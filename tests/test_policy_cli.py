from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.cli import main


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "policy_prototype.toml"


class PolicyCliTests(unittest.TestCase):
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
            self.assertEqual(sensitivity_metadata["output_schema_version"], "2.0")
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
