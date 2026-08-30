from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_jurisdiction_population_fixture import (  # noqa: E402
    write_four_jurisdiction_population_fixture,
)
from monetary_execution_fixture import (  # noqa: E402
    write_monetary_execution_fixture,
)
from test_analysis_binding import _plan, _planned_estimand  # noqa: E402

from microtx_sim.causal.analysis_plan import (  # noqa: E402
    PopulationOutcomeMetric,
)
from microtx_sim.causal.batch import (  # noqa: E402
    PolicyBatchSpec,
    resolve_policy_run_inputs,
)
from microtx_sim.cli import main
from microtx_sim.config import (  # noqa: E402
    PopulationExecutionMode,
    PopulationProjectionConfig,
)
from microtx_sim.core.ledger import LedgerStorageError
from microtx_sim.data.lineage import build_profile_input_lineage  # noqa: E402
from microtx_sim.data.monetary_execution import (  # noqa: E402
    build_monetary_output_currency_semantics,
)
from microtx_sim.policy_config import (  # noqa: E402
    AnalysisPlanSelection,
    load_policy_config,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "policy_prototype.toml"
CAMPAIGN_CONFIG = ROOT / "configs" / "policy_campaign.toml"


class PolicyCliTests(unittest.TestCase):
    def test_campaign_candidate_validation_fails_before_treatment(self) -> None:
        stderr = io.StringIO()
        with (
            patch("microtx_sim.cli.run_policy_batch") as run_batch,
            patch("microtx_sim.cli.run_sensitivity_analysis") as sensitivity,
            redirect_stderr(stderr),
        ):
            code = main(("policy-validate", str(CAMPAIGN_CONFIG)))
        self.assertEqual(code, 2)
        self.assertIn(
            "campaign population preflight failed before treatment",
            stderr.getvalue(),
        )
        self.assertIn("ILLUSTRATIVE", stderr.getvalue())
        run_batch.assert_not_called()
        sensitivity.assert_not_called()

    def test_invalid_monetary_plan_fails_before_batch_or_sensitivity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_root = root / "profiles"
            profile_root.mkdir()
            profiles, _rate_artifact = write_monetary_execution_fixture(
                profile_root
            )
            adapter = write_four_jurisdiction_population_fixture(
                root / "population"
            )
            profile_lineage = build_profile_input_lineage(
                profiles.country_profiles,
                profile_bundle=profiles,
            )
            currency = build_monetary_output_currency_semantics(
                profile_lineage,
                jurisdiction_codes=profile_lineage.profile_codes,
                target_minor_unit_name="test target minor unit",
            )
            stale_currency = replace(
                currency,
                currency_basis_sha256="0" * 64,
            )

            base = load_policy_config(BASE_CONFIG)
            spec = PolicyBatchSpec(
                seeds=(17,),
                days=1,
                player_count=16,
                scenarios=base.batch.scenarios,
                reference_scenario=base.batch.reference_scenario,
                decision_parameters=base.batch.decision_parameters,
            )
            run_inputs = resolve_policy_run_inputs(
                harm_parameters=base.harm_parameters,
                harm_weights=base.harm_weights,
                opportunity_valuation=base.opportunity_valuation,
                producer_assumptions=base.producer_assumptions,
                epgc_policy=base.epgc_policy,
            )
            planned = _planned_estimand(
                outcome_metric=PopulationOutcomeMetric.SPENDING_CENTS,
                metric_contract_id="player_outcomes.csv:spending_cents",
                currency=stale_currency,
            )
            plan = _plan(
                spec,
                run_inputs,
                adapter,
                estimand=planned,
                overrides={
                    "expected_profile_input_sha256": (
                        profile_lineage.fingerprint_sha256
                    )
                },
            )
            plan_path = root / "analysis-plan.json"
            plan_path.write_text(
                json.dumps(
                    plan.snapshot(),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
                newline="",
            )
            config = replace(
                base,
                batch=spec,
                population=PopulationProjectionConfig(
                    mode=PopulationExecutionMode.PROJECTED_V1,
                    design_bundle_path=adapter.verification.bundle.bundle_path,
                    runtime_mapping_bundle_path=(
                        adapter.mapping_bundle.mapping_path
                    ),
                    adapter_id=adapter.adapter_id,
                ),
                analysis_plan=AnalysisPlanSelection(plan_path),
                output=replace(
                    base.output,
                    output_dir=root / "never-created",
                    run_sensitivity=True,
                ),
            )

            stderr = io.StringIO()
            with (
                patch(
                    "microtx_sim.cli.load_policy_config",
                    return_value=config,
                ),
                patch(
                    "microtx_sim.cli.load_profile_bundle",
                    return_value=profiles,
                ),
                patch(
                    "microtx_sim.cli.resolve_population_projection_adapter",
                    return_value=adapter,
                ),
                patch("microtx_sim.cli.run_policy_batch") as execute_batch,
                patch(
                    "microtx_sim.cli._run_configured_sensitivity"
                ) as execute_sensitivity,
                redirect_stderr(stderr),
            ):
                code = main(("policy-batch", str(BASE_CONFIG)))

            self.assertEqual(code, 2)
            self.assertIn(
                "executed currency/price-period conversion basis",
                stderr.getvalue(),
            )
            self.assertIn("currency_basis_sha256", stderr.getvalue())
            execute_batch.assert_not_called()
            execute_sensitivity.assert_not_called()
            self.assertFalse(config.output.output_dir.exists())

    def test_smoke_normalises_sqlite_storage_failures(self) -> None:
        for failure in (
            sqlite3.OperationalError("forced SQLite failure"),
            LedgerStorageError("forced durability failure"),
        ):
            with self.subTest(failure=type(failure).__name__):
                stderr = io.StringIO()
                with (
                    patch("microtx_sim.cli._smoke", side_effect=failure),
                    redirect_stderr(stderr),
                ):
                    code = main(
                        ("smoke", str(ROOT / "configs" / "smoke.toml"))
                    )
                self.assertEqual(code, 2)
                self.assertIn("error:", stderr.getvalue())
                self.assertIn(str(failure), stderr.getvalue())

    def test_smoke_command_is_deterministic_except_for_elapsed_time(self) -> None:
        payloads: list[dict[str, object]] = []
        for _ in range(2):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(("smoke", str(ROOT / "configs" / "smoke.toml")))
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            payload.pop("elapsed_seconds")
            payloads.append(payload)

        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0]["mode"], "smoke_only")
        self.assertEqual(payloads[0]["cycles"], 3)
        self.assertEqual(payloads[0]["step_history_retention"], "full")
        self.assertEqual(payloads[0]["audit_count"], 16)
        self.assertEqual(payloads[0]["seed_decimal"], str(payloads[0]["seed"]))
        summary = payloads[0]["summary"]
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["players"], 384)

    def test_smoke_audit_count_is_exact_with_final_only_history(self) -> None:
        source = (ROOT / "configs" / "smoke.toml").read_text("utf-8")
        bounded = source.replace(
            'step_history_retention = "full"',
            'step_history_retention = "final_only"',
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "bounded-smoke.toml"
            config_path.write_text(bounded, "utf-8")
            bounded_stdout = io.StringIO()
            with redirect_stdout(bounded_stdout):
                code = main(("smoke", str(config_path)))
            full_stdout = io.StringIO()
            with redirect_stdout(full_stdout):
                full_code = main(
                    ("smoke", str(ROOT / "configs" / "smoke.toml"))
                )

        self.assertEqual(code, 0)
        self.assertEqual(full_code, 0)
        payload = json.loads(bounded_stdout.getvalue())
        full_payload = json.loads(full_stdout.getvalue())
        self.assertEqual(payload["step_history_retention"], "final_only")
        self.assertEqual(payload["cycles"], 3)
        self.assertEqual(payload["audit_count"], 16)
        payload.pop("elapsed_seconds")
        full_payload.pop("elapsed_seconds")
        payload.pop("step_history_retention")
        full_payload.pop("step_history_retention")
        self.assertEqual(payload, full_payload)

    def test_policy_validate_and_small_batch_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            config_path = root / "small.toml"
            text = BASE_CONFIG.read_text("utf-8")
            text = text.replace("seeds = [101, 202, 303]", "seeds = [7]")
            text = text.replace("days = 14", "days = 1")
            text = text.replace("player_count = 1000", "player_count = 12")
            text = text.replace("step_minutes = 60", "step_minutes = 240")
            text = text.replace(
                'output_dir = "artifacts/policy_prototype"',
                f'output_dir = "{output.as_posix()}"',
            )
            text = text.replace("run_sensitivity = true", "run_sensitivity = false")
            config_path.write_text(text, "utf-8")

            validation_stdout = io.StringIO()
            with redirect_stdout(validation_stdout):
                code = main(("policy-validate", str(config_path)))
            self.assertEqual(code, 0)
            validation = json.loads(validation_stdout.getvalue())
            self.assertEqual(validation["scenario_count"], 7)
            self.assertFalse(validation["empirical_validation_claimed"])

            batch_stdout = io.StringIO()
            with redirect_stdout(batch_stdout):
                code = main(
                    (
                        "policy-batch",
                        str(config_path),
                        "--skip-sensitivity",
                    )
                )
            self.assertEqual(code, 0)
            payload = json.loads(batch_stdout.getvalue())
            self.assertEqual(payload["scenario_count"], 7)
            self.assertEqual(payload["seeds"], [7])
            self.assertEqual(payload["seed_decimal_strings"], ["7"])
            self.assertFalse(payload["sensitivity_run"])
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "scenario_summary.csv").exists())
            self.assertTrue((output / "harm_revenue_frontier.svg").exists())
            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertEqual(
                manifest["profile_inputs"]["lineage_status"],
                "registered_profile_bundle",
            )
            self.assertEqual(
                manifest["batch"]["profile_codes"],
                ["UK", "KR", "JP", "BE"],
            )
            self.assertEqual(
                manifest["source_registry_retrieved_on"],
                "2026-08-24",
            )
            self.assertEqual(
                manifest["batch"]["profile_input_fingerprint_sha256"],
                manifest["profile_inputs"]["fingerprint_sha256"],
            )
            self.assertFalse(manifest["sensitivity"]["run"])
            self.assertIsNone(manifest["sensitivity"]["execution_sha256"])
            self.assertIsNone(manifest["sensitivity"]["execution_snapshot"])
            self.assertIsNone(manifest["sensitivity"]["run_inputs_sha256"])

            sensitivity_output = root / "sensitivity-only"
            sensitivity_stdout = io.StringIO()
            with redirect_stdout(sensitivity_stdout):
                code = main(
                    (
                        "policy-sensitivity",
                        str(config_path),
                        "--output",
                        str(sensitivity_output),
                    )
                )
            self.assertEqual(code, 0)
            sensitivity_metadata = json.loads(
                (sensitivity_output / "sensitivity_metadata.json").read_text(
                    "utf-8"
                )
            )
            self.assertEqual(sensitivity_metadata["seeds"], [7])
            self.assertEqual(
                sensitivity_metadata["seed_decimal_strings"],
                ["7"],
            )
            self.assertEqual(
                len(sensitivity_metadata["execution_sha256"]),
                64,
            )
            self.assertEqual(
                sensitivity_metadata["execution_snapshot"]["batch_spec"][
                    "seeds"
                ],
                [7],
            )
            run_input_snapshot = sensitivity_metadata["run_input_snapshot"]
            self.assertEqual(
                run_input_snapshot["batch_spec"],
                sensitivity_metadata["execution_snapshot"]["batch_spec"],
            )
            self.assertEqual(
                run_input_snapshot["model_inputs"],
                sensitivity_metadata["execution_snapshot"]["run_inputs"],
            )
            self.assertEqual(
                sensitivity_metadata["run_input_sha256"],
                sha256(
                    json.dumps(
                        run_input_snapshot,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertNotIn("output_schema_version", sensitivity_metadata)
            self.assertEqual(
                sensitivity_metadata["output_profile"],
                "standalone_sensitivity",
            )
            self.assertEqual(
                sensitivity_metadata["output_profile_schema_version"],
                "1.0",
            )
            self.assertEqual(
                sensitivity_metadata["artifact_files"],
                ["sensitivity.csv", "sensitivity_metadata.json"],
            )
            self.assertRegex(
                sensitivity_metadata["output_profile_schema_sha256"],
                r"^[0-9a-f]{64}$",
            )
            sensitivity_csv = sensitivity_output / "sensitivity.csv"
            self.assertEqual(
                sensitivity_metadata["artifacts"]["sensitivity.csv"],
                {
                    "row_count": 15,
                    "sha256": sha256(sensitivity_csv.read_bytes()).hexdigest(),
                },
            )


if __name__ == "__main__":
    unittest.main()
