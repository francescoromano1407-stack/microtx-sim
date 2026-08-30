from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.causal.analysis_binding import (
    AnalysisBindingValidationError,
    validate_analysis_plan_inputs,
)
from microtx_sim.causal.analysis_plan import (
    AnalysisEstimandRole,
    AnalysisPlanValidationError,
    AnalysisPlanVerificationError,
    PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION,
    analysis_plan_harm_weights_sha256,
    load_prospective_analysis_plan,
    verify_loaded_prospective_analysis_plan,
)
from microtx_sim.causal.batch import resolve_policy_run_inputs
from microtx_sim.cli import _load_policy_profiles
from microtx_sim.data.lineage import build_profile_input_lineage
from microtx_sim.data.population_execution import (
    resolve_population_projection_adapter,
)
from microtx_sim.outputs.schema import PROSPECTIVE_ANALYSIS_SCHEMA_SHA256
from microtx_sim.policy_config import load_policy_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "policy_prospective.toml"
PLAN = ROOT / "inputs" / "prospective-analysis-plan.json"


class CheckedInProspectivePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_policy_config(CONFIG)
        cls.loaded = load_prospective_analysis_plan(PLAN)
        cls.profiles = _load_policy_profiles(cls.config)
        assert cls.config.population is not None
        cls.adapter = resolve_population_projection_adapter(
            cls.config.population,
            cls.profiles,
            player_count=cls.config.batch.player_count,
        )
        cls.profile_lineage = build_profile_input_lineage(
            cls.profiles.country_profiles,
            profile_bundle=cls.profiles,
        )
        cls.run_inputs = resolve_policy_run_inputs(
            harm_parameters=cls.config.harm_parameters,
            harm_weights=cls.config.harm_weights,
            opportunity_valuation=cls.config.opportunity_valuation,
            producer_assumptions=cls.config.producer_assumptions,
            epgc_policy=cls.config.epgc_policy,
        )

    def test_concrete_plan_is_selected_and_fully_preflighted(self) -> None:
        config = self.config
        self.assertIsNotNone(config.analysis_plan)
        assert config.analysis_plan is not None
        self.assertEqual(config.analysis_plan.plan_path, PLAN)
        self.assertIsNotNone(config.population)
        self.assertTrue(config.output.include_player_rows)
        self.assertIsNone(
            validate_analysis_plan_inputs(
                self.loaded.plan,
                batch_spec=config.batch,
                run_inputs=self.run_inputs,
                population_adapter=self.adapter,
                profile_input_lineage=self.profile_lineage,
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "analysis_plan requires projected population execution",
        ):
            replace(config, population=None)

    def test_plan_declares_one_directed_primary_and_no_scenario_average(self) -> None:
        plan = self.loaded.plan
        self.assertEqual(
            plan.schema_version,
            PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION,
        )
        self.assertEqual(
            plan.plan_id,
            "illustrative.prospective.composite-harm.baseline-vs-safe.v2",
        )
        self.assertEqual(
            sum(item.role is AnalysisEstimandRole.PRIMARY for item in plan.estimands),
            1,
        )
        primary = plan.primary_estimand
        self.assertEqual(primary.contrast_direction, "COMPARISON_MINUS_REFERENCE")
        self.assertEqual(primary.reference_scenario_id.value, "safe_fixed_price_subscription")
        self.assertEqual(primary.comparison_scenario_id.value, "baseline_f2p")
        self.assertEqual(primary.outcome_metric.value, "composite_harm")
        assert plan.primary_aggregate_rule is not None
        rule = plan.primary_aggregate_rule.snapshot()
        self.assertEqual(rule["scenario_aggregation"], "SINGLE_DIRECTED_CONTRAST")
        self.assertEqual(rule["scenario_weights"], [])
        self.assertEqual(rule["seed_weighting"], "EQUAL")
        self.assertEqual(rule["exclusion_criteria"], [])
        self.assertFalse(rule["outcome_dependent_exclusion_allowed"])

    def test_harm_weights_seeds_stopping_and_output_contract_are_explicit(self) -> None:
        plan = self.loaded.plan
        self.assertEqual(plan.stopping_rule.seeds, (101, 202, 303))
        stopping = plan.stopping_rule.snapshot()
        self.assertFalse(stopping["early_stopping_allowed"])
        self.assertFalse(stopping["treatment_result_interim_looks_allowed"])
        self.assertEqual(plan.expected_output_profile_sha256, PROSPECTIVE_ANALYSIS_SCHEMA_SHA256)
        self.assertEqual(plan.declared_harm_weights, self.config.harm_weights)
        assert plan.declared_harm_weights is not None
        self.assertEqual(
            plan.expected_harm_weights_sha256,
            analysis_plan_harm_weights_sha256(plan.declared_harm_weights),
        )

    def test_plan_file_bytes_and_semantic_hash_are_attested(self) -> None:
        self.assertEqual(
            verify_loaded_prospective_analysis_plan(self.loaded),
            self.loaded,
        )
        self.assertEqual(self.loaded.semantic_sha256, self.loaded.plan.plan_sha256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_bytes(PLAN.read_bytes())
            loaded = load_prospective_analysis_plan(path)
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                AnalysisPlanVerificationError,
                "changed after it was loaded|changed",
            ):
                verify_loaded_prospective_analysis_plan(loaded)

            snapshot = json.loads(PLAN.read_text("utf-8"))
            snapshot["estimands"][0]["comparison_scenario_id"] = "epgc"
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaisesRegex(
                AnalysisPlanValidationError,
                "plan_sha256|canonical",
            ):
                load_prospective_analysis_plan(path)

    def test_changed_seed_harm_weight_or_scenario_order_fails_preflight(self) -> None:
        cases = (
            (
                "seed",
                {"batch_spec": replace(self.config.batch, seeds=(101, 202))},
            ),
            (
                "harm",
                {
                    "run_inputs": replace(
                        self.run_inputs,
                        harm_weights=replace(
                            self.run_inputs.harm_weights,
                            monetary=2.0,
                        ),
                    )
                },
            ),
            (
                "scenario_order",
                {
                    "batch_spec": replace(
                        self.config.batch,
                        scenarios=tuple(reversed(self.config.batch.scenarios)),
                    )
                },
            ),
        )
        baseline = {
            "batch_spec": self.config.batch,
            "run_inputs": self.run_inputs,
            "population_adapter": self.adapter,
            "profile_input_lineage": self.profile_lineage,
        }
        for label, change in cases:
            with self.subTest(label=label):
                with self.assertRaises(AnalysisBindingValidationError):
                    validate_analysis_plan_inputs(
                        self.loaded.plan,
                        **{**baseline, **change},
                    )


if __name__ == "__main__":
    unittest.main()
