from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import date
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_population_projection_adapter import _complete_adapter  # noqa: E402

from microtx_sim.causal.analysis_binding import (  # noqa: E402
    AnalysisBindingValidationError,
    resolve_run_analysis_binding,
    validate_analysis_plan_inputs,
)
from microtx_sim.causal.analysis_plan import (  # noqa: E402
    AnalysisEstimandRole,
    AnalysisPlanVerificationError,
    CanonicalPopulationInclusionPredicate,
    FixedSeedStoppingRule,
    PlannedPopulationEstimand,
    PopulationMinorFilter,
    PopulationOutcomeMetric,
    analysis_plan_harm_weights_sha256,
    build_prospective_analysis_plan,
    load_prospective_analysis_plan,
)
from microtx_sim.causal.batch import (  # noqa: E402
    PolicyBatchSpec,
    resolve_policy_run_inputs,
    run_policy_batch,
)
from microtx_sim.causal.design import assess_causal_design  # noqa: E402
from microtx_sim.causal.scenarios import ScenarioId  # noqa: E402
from microtx_sim.cli import main  # noqa: E402
from microtx_sim.consumers.decision import DecisionParameters  # noqa: E402
from microtx_sim.consumers.population import CountryProfile  # noqa: E402
from microtx_sim.config import (  # noqa: E402
    PopulationExecutionMode,
    PopulationProjectionConfig,
)
from microtx_sim.data.population_evidence import (  # noqa: E402
    PopulationEstimandRole,
)
from microtx_sim.data.lineage import build_profile_input_lineage  # noqa: E402
from microtx_sim.data.population_execution import (  # noqa: E402
    population_execution_input_sha256,
)
from microtx_sim.metrics.population_estimands import (  # noqa: E402
    PopulationInclusionField,
    PopulationInclusionRule,
    PopulationInclusionTiming,
    PopulationPeriodSemantics,
)
from microtx_sim.outputs.export import export_policy_batch  # noqa: E402
from microtx_sim.outputs.metric_contracts import (  # noqa: E402
    metric_contract_registry_sha256,
)
from microtx_sim.outputs.schema import (  # noqa: E402
    POLICY_ARTIFACT_FILENAMES,
    TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES,
    TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256,
)
from microtx_sim.policy_config import (  # noqa: E402
    AnalysisPlanSelection,
    PolicyOutputConfig,
    load_policy_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "policy_prototype.toml"
_CUSTOM_PROFILES = (CountryProfile(code="UK"),)
_CUSTOM_PROFILE_INPUT_LINEAGE = build_profile_input_lineage(_CUSTOM_PROFILES)


def _all_players_predicate() -> CanonicalPopulationInclusionPredicate:
    return CanonicalPopulationInclusionPredicate(
        rule=PopulationInclusionRule(
            rule_id="all.projected.players.v1",
            description="All players in the pre-treatment projected cohort.",
            source_fields=tuple(
                sorted(PopulationInclusionField, key=lambda item: item.value)
            ),
            timing=PopulationInclusionTiming.PRETREATMENT,
            evidence_role=PopulationEstimandRole.CALIBRATION,
        ),
        jurisdiction_codes=(),
        age_min_inclusive=0,
        age_max_exclusive=32_768,
        minor_filter=PopulationMinorFilter.ANY,
        monthly_disposable_income_band_ids=(),
        household_type_ids=(),
        gaming_states=(),
        payer_history_states=(),
    )


def _build_plan(
    spec,
    run_inputs,
    adapter,
    *,
    batch_sha256: str | None = None,
    inclusion_predicate: CanonicalPopulationInclusionPredicate | None = None,
):
    estimand = PlannedPopulationEstimand(
        estimand_id="primary.composite-harm.v1",
        role=AnalysisEstimandRole.PRIMARY,
        reference_scenario_id=ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
        comparison_scenario_id=ScenarioId.BASELINE_F2P,
        outcome_metric=PopulationOutcomeMetric.COMPOSITE_HARM,
        metric_contract_id="player_outcomes.csv:composite_harm",
        inclusion_predicate=(
            _all_players_predicate()
            if inclusion_predicate is None
            else inclusion_predicate
        ),
        period=PopulationPeriodSemantics(
            period_start=date(2026, 1, 1),
            period_end=date(2026, 1, 1),
            description="Synthetic policy-run period for structural testing.",
        ),
    )
    return build_prospective_analysis_plan(
        plan_id="test.prospective-analysis.v1",
        expected_causal_design_sha256=assess_causal_design(
            spec.scenarios
        ).design_sha256(),
        expected_batch_spec_sha256=(
            spec.snapshot_sha256() if batch_sha256 is None else batch_sha256
        ),
        expected_model_inputs_sha256=run_inputs.snapshot_sha256(),
        expected_population_input_sha256=population_execution_input_sha256(
            adapter
        ),
        expected_profile_input_sha256=(
            _CUSTOM_PROFILE_INPUT_LINEAGE.fingerprint_sha256
        ),
        expected_metric_contract_sha256=metric_contract_registry_sha256(),
        expected_harm_weights_sha256=analysis_plan_harm_weights_sha256(
            run_inputs.harm_weights
        ),
        expected_output_profile_sha256=(
            TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256
        ),
        stopping_rule=FixedSeedStoppingRule(seeds=spec.seeds),
        estimands=(estimand,),
    )


def _write_plan(path: Path, plan) -> None:
    path.write_text(
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


class AnalysisCompositionTests(unittest.TestCase):
    def _fixture(self, root: Path):
        _verification, _design, mapping_path, _mapping, adapter = (
            _complete_adapter(root)
        )
        base = load_policy_config(CONFIG_PATH)
        spec = PolicyBatchSpec(
            seeds=(17,),
            days=0,
            player_count=12,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        run_inputs = resolve_policy_run_inputs(
            harm_parameters=base.harm_parameters,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        plan = _build_plan(spec, run_inputs, adapter)
        plan_path = root / "analysis-plan.json"
        _write_plan(plan_path, plan)
        loaded = load_prospective_analysis_plan(plan_path)
        config = replace(
            base,
            batch=spec,
            population=PopulationProjectionConfig(
                mode=PopulationExecutionMode.PROJECTED_V1,
                design_bundle_path=(
                    adapter.verification.bundle.bundle_path
                ),
                runtime_mapping_bundle_path=mapping_path,
                adapter_id=adapter.adapter_id,
            ),
            analysis_plan=AnalysisPlanSelection(plan_path),
            output=PolicyOutputConfig(
                root / "output",
                histogram_bins=8,
                include_player_rows=True,
                run_sensitivity=False,
            ),
        )
        return config, run_inputs, adapter, loaded

    def test_opt_in_export_links_separate_exact_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, run_inputs, adapter, loaded = self._fixture(root)
            validate_analysis_plan_inputs(
                loaded.plan,
                batch_spec=config.batch,
                run_inputs=run_inputs,
                population_adapter=adapter,
                profile_input_lineage=_CUSTOM_PROFILE_INPUT_LINEAGE,
            )
            batch = run_policy_batch(
                config.batch,
                country_profiles=_CUSTOM_PROFILES,
                harm_parameters=config.harm_parameters,
                harm_weights=config.harm_weights,
                opportunity_valuation=config.opportunity_valuation,
                producer_assumptions=config.producer_assumptions,
                epgc_policy=config.epgc_policy,
                population_adapter=adapter,
            )
            binding = resolve_run_analysis_binding(loaded.plan, batch)

            paths = export_policy_batch(
                config,
                batch,
                None,
                config_path=CONFIG_PATH,
                repository_root=ROOT,
                output_dir=config.output.output_dir,
                created_utc="2026-08-29T00:00:00+00:00",
                analysis_plan=loaded,
                analysis_binding=binding,
            )

            self.assertEqual(
                {path.name for path in paths.values()},
                set(POLICY_ARTIFACT_FILENAMES).union(
                    TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES
                ),
            )
            manifest = json.loads(
                (config.output.output_dir / "manifest.json").read_text("utf-8")
            )
            self.assertEqual(
                manifest["analysis_plan"]["semantic_sha256"],
                loaded.plan.plan_sha256,
            )
            self.assertEqual(
                manifest["analysis_binding"]["binding_sha256"],
                binding.binding_sha256,
            )
            self.assertFalse(manifest["analysis_binding"]["campaign_ready"])
            self.assertEqual(
                manifest["analysis_output_profile"]["artifact_files"],
                list(TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES),
            )
            self.assertEqual(
                manifest["artifact_files"],
                list(POLICY_ARTIFACT_FILENAMES),
            )
            self.assertTrue(
                (
                    config.output.output_dir
                    / "prospective_analysis"
                    / "target_population_estimands.csv"
                ).is_file()
            )

    def test_export_reopens_plan_before_any_destination_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _run_inputs, adapter, loaded = self._fixture(root)
            batch = run_policy_batch(
                config.batch,
                country_profiles=_CUSTOM_PROFILES,
                harm_parameters=config.harm_parameters,
                harm_weights=config.harm_weights,
                opportunity_valuation=config.opportunity_valuation,
                producer_assumptions=config.producer_assumptions,
                epgc_policy=config.epgc_policy,
                population_adapter=adapter,
            )
            binding = resolve_run_analysis_binding(loaded.plan, batch)
            loaded.plan_path.write_bytes(loaded.plan_path.read_bytes() + b" ")

            with self.assertRaisesRegex(
                AnalysisPlanVerificationError,
                "changed",
            ):
                export_policy_batch(
                    config,
                    batch,
                    None,
                    config_path=CONFIG_PATH,
                    repository_root=ROOT,
                    output_dir=config.output.output_dir,
                    analysis_plan=loaded,
                    analysis_binding=binding,
                )
            self.assertFalse(config.output.output_dir.exists())

    def test_plan_and_no_plan_reruns_reject_existing_profile_before_mutation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, run_inputs, adapter, loaded = self._fixture(root)
            validate_analysis_plan_inputs(
                loaded.plan,
                batch_spec=config.batch,
                run_inputs=run_inputs,
                population_adapter=adapter,
                profile_input_lineage=_CUSTOM_PROFILE_INPUT_LINEAGE,
            )
            batch = run_policy_batch(
                config.batch,
                country_profiles=_CUSTOM_PROFILES,
                harm_parameters=config.harm_parameters,
                harm_weights=config.harm_weights,
                opportunity_valuation=config.opportunity_valuation,
                producer_assumptions=config.producer_assumptions,
                epgc_policy=config.epgc_policy,
                population_adapter=adapter,
            )
            binding = resolve_run_analysis_binding(loaded.plan, batch)
            export_policy_batch(
                config,
                batch,
                None,
                config_path=CONFIG_PATH,
                repository_root=ROOT,
                output_dir=config.output.output_dir,
                analysis_plan=loaded,
                analysis_binding=binding,
            )
            before = {
                path.relative_to(config.output.output_dir).as_posix(): (
                    path.read_bytes()
                )
                for path in config.output.output_dir.rglob("*")
                if path.is_file()
            }

            with self.assertRaisesRegex(
                FileExistsError,
                "prospective analysis output target already exists",
            ):
                export_policy_batch(
                    config,
                    batch,
                    None,
                    config_path=CONFIG_PATH,
                    repository_root=ROOT,
                    output_dir=config.output.output_dir,
                    analysis_plan=loaded,
                    analysis_binding=binding,
                )
            after_plan_rerun = {
                path.relative_to(config.output.output_dir).as_posix(): (
                    path.read_bytes()
                )
                for path in config.output.output_dir.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after_plan_rerun, before)

            without_plan = replace(config, analysis_plan=None)
            with self.assertRaisesRegex(
                FileExistsError,
                "prospective analysis output target already exists",
            ):
                export_policy_batch(
                    without_plan,
                    batch,
                    None,
                    config_path=CONFIG_PATH,
                    repository_root=ROOT,
                    output_dir=without_plan.output.output_dir,
                )
            after = {
                path.relative_to(config.output.output_dir).as_posix(): (
                    path.read_bytes()
                )
                for path in config.output.output_dir.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_late_root_failure_does_not_publish_staged_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _run_inputs, adapter, loaded = self._fixture(root)
            batch = run_policy_batch(
                config.batch,
                country_profiles=_CUSTOM_PROFILES,
                harm_parameters=config.harm_parameters,
                harm_weights=config.harm_weights,
                opportunity_valuation=config.opportunity_valuation,
                producer_assumptions=config.producer_assumptions,
                epgc_policy=config.epgc_policy,
                population_adapter=adapter,
            )
            binding = resolve_run_analysis_binding(loaded.plan, batch)

            with patch(
                "microtx_sim.outputs.export.write_harm_revenue_frontier_svg",
                side_effect=RuntimeError("deliberate late root failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "deliberate late root failure",
                ):
                    export_policy_batch(
                        config,
                        batch,
                        None,
                        config_path=CONFIG_PATH,
                        repository_root=ROOT,
                        output_dir=config.output.output_dir,
                        analysis_plan=loaded,
                        analysis_binding=binding,
                    )

            self.assertFalse(
                (config.output.output_dir / "prospective_analysis").exists()
            )
            self.assertEqual(
                [
                    path
                    for path in root.iterdir()
                    if path.name.startswith(
                        f".{config.output.output_dir.name}.prospective-analysis-"
                    )
                ],
                [],
            )
            interim_manifest = json.loads(
                (config.output.output_dir / "manifest.json").read_text("utf-8")
            )
            self.assertNotIn("analysis_output_profile", interim_manifest)

    def test_final_manifest_failure_rolls_back_published_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _run_inputs, adapter, loaded = self._fixture(root)
            batch = run_policy_batch(
                config.batch,
                country_profiles=_CUSTOM_PROFILES,
                harm_parameters=config.harm_parameters,
                harm_weights=config.harm_weights,
                opportunity_valuation=config.opportunity_valuation,
                producer_assumptions=config.producer_assumptions,
                epgc_policy=config.epgc_policy,
                population_adapter=adapter,
            )
            binding = resolve_run_analysis_binding(loaded.plan, batch)

            from microtx_sim.outputs.writers import write_json_atomic

            def write_then_fail(path, payload):
                write_json_atomic(path, payload)
                raise RuntimeError("deliberate final manifest failure")

            with patch(
                "microtx_sim.outputs.export.write_json_atomic",
                side_effect=write_then_fail,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "deliberate final manifest failure",
                ):
                    export_policy_batch(
                        config,
                        batch,
                        None,
                        config_path=CONFIG_PATH,
                        repository_root=ROOT,
                        output_dir=config.output.output_dir,
                        analysis_plan=loaded,
                        analysis_binding=binding,
                    )

            self.assertFalse(
                (config.output.output_dir / "prospective_analysis").exists()
            )
            interim_manifest = json.loads(
                (config.output.output_dir / "manifest.json").read_text("utf-8")
            )
            self.assertNotIn("analysis_output_profile", interim_manifest)

    def test_preflight_digest_mismatch_precedes_scenario_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, run_inputs, adapter, _loaded = self._fixture(root)
            changed = _build_plan(
                config.batch,
                run_inputs,
                adapter,
                batch_sha256="0" * 64,
            )
            _write_plan(config.analysis_plan.plan_path, changed)

            with (
                patch("microtx_sim.cli.load_policy_config", return_value=config),
                patch(
                    "microtx_sim.cli.resolve_population_projection_adapter",
                    return_value=adapter,
                ),
                patch(
                    "microtx_sim.cli.build_profile_input_lineage",
                    return_value=_CUSTOM_PROFILE_INPUT_LINEAGE,
                ),
                patch("microtx_sim.cli.run_policy_batch") as execute,
            ):
                from microtx_sim.cli import _policy_batch

                with self.assertRaisesRegex(
                    AnalysisBindingValidationError,
                    "batch_spec_sha256",
                ):
                    _policy_batch(
                        CONFIG_PATH,
                        output=root / "never-created",
                        run_sensitivity=False,
                        command=("microtx-sim", "policy-batch"),
                    )
                execute.assert_not_called()

    def test_preflight_unknown_population_level_precedes_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, run_inputs, adapter, _loaded = self._fixture(root)
            predicate = replace(
                _all_players_predicate(),
                jurisdiction_codes=("UK", "ZZ"),
            )
            changed = _build_plan(
                config.batch,
                run_inputs,
                adapter,
                inclusion_predicate=predicate,
            )
            assert config.analysis_plan is not None
            _write_plan(config.analysis_plan.plan_path, changed)

            with (
                patch("microtx_sim.cli.load_policy_config", return_value=config),
                patch(
                    "microtx_sim.cli.resolve_population_projection_adapter",
                    return_value=adapter,
                ),
                patch(
                    "microtx_sim.cli.build_profile_input_lineage",
                    return_value=_CUSTOM_PROFILE_INPUT_LINEAGE,
                ),
                patch("microtx_sim.cli.run_policy_batch") as execute,
            ):
                from microtx_sim.cli import _policy_batch

                with self.assertRaisesRegex(
                    AnalysisBindingValidationError,
                    r"jurisdiction_codes.*outside the exact population adapter domain",
                ):
                    _policy_batch(
                        CONFIG_PATH,
                        output=root / "never-created",
                        run_sensitivity=False,
                        command=("microtx-sim", "policy-batch"),
                    )
                execute.assert_not_called()

    def test_opt_in_policy_cli_composes_profile_without_full_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _run_inputs, adapter, _loaded = self._fixture(root)
            output = root / "cli-output"
            text = CONFIG_PATH.read_text("utf-8")
            text = text.replace("seeds = [101, 202, 303]", "seeds = [17]")
            text = text.replace("days = 14", "days = 0")
            text = text.replace("player_count = 1000", "player_count = 12")
            text = text.replace("step_minutes = 60", "step_minutes = 240")
            text = text.replace(
                'output_dir = "artifacts/policy_prototype"',
                f'output_dir = "{output.as_posix()}"',
            )
            text = text.replace(
                "run_sensitivity = true",
                "run_sensitivity = false",
            )
            assert config.population is not None
            assert config.analysis_plan is not None
            text += f"""

[population]
mode = "projected_v1"
design_bundle_path = "{config.population.design_bundle_path.as_posix()}"
runtime_mapping_bundle_path = "{config.population.runtime_mapping_bundle_path.as_posix()}"
adapter_id = "{config.population.adapter_id}"

[analysis_plan]
plan_path = "{config.analysis_plan.plan_path.as_posix()}"
"""
            config_path = root / "policy-with-plan.toml"
            config_path.write_text(text, "utf-8", newline="")

            def run_fixture_batch(spec, **kwargs):
                kwargs.pop("profile_bundle")
                return run_policy_batch(
                    spec,
                    country_profiles=_CUSTOM_PROFILES,
                    **kwargs,
                )

            stdout = io.StringIO()
            with (
                patch(
                    "microtx_sim.cli.resolve_population_projection_adapter",
                    return_value=adapter,
                ),
                patch(
                    "microtx_sim.cli.build_profile_input_lineage",
                    return_value=_CUSTOM_PROFILE_INPUT_LINEAGE,
                ),
                patch(
                    "microtx_sim.cli.run_policy_batch",
                    side_effect=run_fixture_batch,
                ),
                redirect_stdout(stdout),
            ):
                code = main(
                    (
                        "policy-batch",
                        str(config_path),
                        "--skip-sensitivity",
                    )
                )

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                payload["analysis_plan_sha256"],
                _loaded.plan.plan_sha256,
            )
            self.assertRegex(
                payload["analysis_binding_sha256"],
                r"^[0-9a-f]{64}$",
            )
            self.assertFalse(payload["campaign_ready"])
            self.assertTrue(
                (
                    output
                    / "prospective_analysis"
                    / "target_population_estimand_metadata.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
