from __future__ import annotations

from dataclasses import dataclass, field, replace
import unittest
from unittest.mock import patch

import numpy as np

from microtx_sim.analysis.sensitivity import (
    SensitivityCase,
    default_sensitivity_cases,
    run_sensitivity_analysis,
)
from microtx_sim.causal.batch import PolicyBatchSpec, resolve_policy_run_inputs
from microtx_sim.causal.scenarios import required_scenarios
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.consumers.population import CountryProfile
from microtx_sim.metrics.harm import (
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmWeights,
)
from microtx_sim.simulation.policy_orchestrator import (
    ProducerAssumptions,
    default_epgc_policy,
    run_policy_scenario,
)


@dataclass(frozen=True)
class _ExtendedSensitivityCase(SensitivityCase):
    mutable_metadata: list[str] = field(default_factory=list)


class SensitivityTests(unittest.TestCase):
    def test_result_retains_exact_spec_cases_threshold_and_run_inputs(self) -> None:
        scenarios = required_scenarios()
        custom_baseline = replace(scenarios[0], label="Custom baseline")
        spec = PolicyBatchSpec(
            seeds=(19,),
            days=0,
            player_count=0,
            scenarios=(custom_baseline, *scenarios[1:]),
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        case = SensitivityCase(
            "decision_temperature",
            (0.4, 0.8),
            expected_direction="none",
        )
        harm_parameters = HarmModelParameters(affordable_spending_share=0.2)
        harm_weights = WelfareHarmWeights(monetary=2.0)
        opportunity_valuation = OpportunityCostValuation(
            adult_sleep_hour_cents=601
        )
        producer_assumptions = ProducerAssumptions(
            development_cost_cents=1_200_001
        )
        epgc_policy = replace(
            default_epgc_policy(),
            maximum_budget_cents=3_000_001,
        )
        with patch(
            "microtx_sim.analysis.sensitivity.run_policy_scenario",
            wraps=run_policy_scenario,
        ) as scenario_runner:
            result = run_sensitivity_analysis(
                spec,
                cases=(item for item in (case,)),
                country_profiles=(CountryProfile(code="XX"),),
                instability_cv_threshold=np.float32(0.125),
                base_harm_parameters=harm_parameters,
                harm_weights=harm_weights,
                opportunity_valuation=opportunity_valuation,
                producer_assumptions=producer_assumptions,
                epgc_policy=epgc_policy,
            )

        self.assertIs(result.batch_spec, spec)
        self.assertEqual(result.cases, (case,))
        self.assertIs(result.cases[0], case)
        self.assertIs(type(result.instability_cv_threshold), float)
        self.assertEqual(
            result.instability_cv_threshold,
            float(np.float32(0.125)),
        )
        execution_snapshot = result.execution_snapshot()
        self.assertEqual(execution_snapshot["batch_spec"], spec.snapshot())
        self.assertEqual(
            execution_snapshot["cases"][0]["parameter"],
            case.parameter,
        )
        self.assertEqual(
            execution_snapshot["numerical_tolerances"]["monotonicity"],
            1e-12,
        )
        self.assertEqual(len(result.execution_sha256()), 64)
        self.assertEqual(
            result.run_input_snapshot()["batch_spec"],
            spec.snapshot(),
        )
        self.assertEqual(len(result.run_input_sha256()), 64)
        for name, value in {
            "harm_parameters": harm_parameters,
            "harm_weights": harm_weights,
            "opportunity_valuation": opportunity_valuation,
            "producer_assumptions": producer_assumptions,
            "epgc_policy": epgc_policy,
        }.items():
            self.assertIs(getattr(result.run_inputs, name), value)
        self.assertEqual(scenario_runner.call_count, 2)
        for call in scenario_runner.call_args_list:
            self.assertIs(call.args[2], custom_baseline)
            self.assertIs(call.kwargs["harm_parameters"], harm_parameters)
            self.assertIs(call.kwargs["harm_weights"], harm_weights)
            self.assertIs(
                call.kwargs["opportunity_valuation"],
                opportunity_valuation,
            )
            self.assertIs(
                call.kwargs["producer_assumptions"],
                producer_assumptions,
            )
            self.assertIs(call.kwargs["epgc_policy"], epgc_policy)

        default_result = run_sensitivity_analysis(
            spec,
            cases=(case,),
            country_profiles=(CountryProfile(code="XX"),),
        )
        self.assertEqual(default_result.run_inputs, resolve_policy_run_inputs())

    def test_retained_spec_distinguishes_equal_rows_from_different_seeds(self) -> None:
        case = SensitivityCase(
            "decision_temperature",
            (0.4, 0.8),
            expected_direction="none",
        )
        common = {
            "days": 0,
            "player_count": 0,
            "decision_parameters": DecisionParameters(step_minutes=240),
        }
        first = run_sensitivity_analysis(
            PolicyBatchSpec(seeds=(1,), **common),
            cases=(case,),
            country_profiles=(CountryProfile(code="XX"),),
        )
        second = run_sensitivity_analysis(
            PolicyBatchSpec(seeds=(2,), **common),
            cases=(case,),
            country_profiles=(CountryProfile(code="XX"),),
        )
        different_config = run_sensitivity_analysis(
            PolicyBatchSpec(
                seeds=(1,),
                days=0,
                player_count=0,
                decision_parameters=DecisionParameters(step_minutes=120),
            ),
            cases=(case,),
            country_profiles=(CountryProfile(code="XX"),),
        )

        self.assertEqual(first.rows, second.rows)
        self.assertNotEqual(first.batch_spec, second.batch_spec)
        self.assertNotEqual(first, second)
        self.assertEqual(first.rows, different_config.rows)
        self.assertNotEqual(first.batch_spec, different_config.batch_spec)
        self.assertNotEqual(first, different_config)

    def test_result_validates_and_freezes_derived_row_claims(self) -> None:
        case = SensitivityCase(
            "decision_temperature",
            (0.4, 0.8),
            expected_direction="none",
        )
        result = run_sensitivity_analysis(
            PolicyBatchSpec(seeds=(5,), days=0, player_count=0),
            cases=(case,),
            country_profiles=(CountryProfile(code="XX"),),
        )

        reordered = replace(result, rows=tuple(reversed(result.rows)))
        self.assertEqual(reordered.rows, result.rows)
        with self.assertRaises(TypeError):
            result.rows[0]["parameter"] = "tampered"  # type: ignore[index]

        wrong_seed_count = [dict(row) for row in result.rows]
        wrong_seed_count[0]["seed_count"] = 2
        with self.assertRaisesRegex(ValueError, "seed_count"):
            replace(result, rows=tuple(wrong_seed_count))

        wrong_direction = [dict(row) for row in result.rows]
        wrong_direction[0]["expected_direction"] = "increasing"
        with self.assertRaisesRegex(ValueError, "expected_direction"):
            replace(result, rows=tuple(wrong_direction))

        missing_column = [dict(row) for row in result.rows]
        missing_column[0].pop("total_revenue_cents")
        with self.assertRaisesRegex(ValueError, "SENSITIVITY_COLUMNS"):
            replace(result, rows=tuple(missing_column))

        wrong_sd = [dict(row) for row in result.rows]
        wrong_sd[0]["harm_sd"] = 999.0
        with self.assertRaisesRegex(ValueError, "harm_sd.*inconsistent"):
            replace(result, rows=tuple(wrong_sd))

        wrong_ci = [dict(row) for row in result.rows]
        wrong_ci[0]["harm_ci95_high"] = 999.0
        with self.assertRaisesRegex(ValueError, "harm_ci95_high.*inconsistent"):
            replace(result, rows=tuple(wrong_ci))

        wrong_metric_type = [dict(row) for row in result.rows]
        wrong_metric_type[0]["total_revenue_cents"] = "0"
        with self.assertRaisesRegex(TypeError, "total_revenue_cents.*numeric"):
            replace(result, rows=tuple(wrong_metric_type))

        unstable_rows = [dict(row) for row in result.rows]
        for row in unstable_rows:
            row["mean_harm"] = 1.0
            row["harm_variance"] = 0.16
            row["harm_sd"] = 0.4
            row["harm_ci95_low"] = 0.216
            row["harm_ci95_high"] = 1.784
            row["harm_coefficient_of_variation"] = 0.4
            row["unstable"] = True
        unstable_result = replace(
            result,
            rows=tuple(unstable_rows),
            instability_cv_threshold=0.35,
            unstable_parameters=(case.parameter,),
        )
        with self.assertRaisesRegex(ValueError, "unstable"):
            replace(unstable_result, instability_cv_threshold=0.5)

    def test_default_grid_reports_monotonicity_uncertainty_and_instability(self) -> None:
        spec = PolicyBatchSpec(
            seeds=(1, 2),
            days=1,
            player_count=20,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        result = run_sensitivity_analysis(
            spec,
            country_profiles=(CountryProfile(code="XX"),),
        )
        self.assertEqual(
            tuple(profile.code for profile in result.country_profiles),
            ("XX",),
        )
        self.assertIsNotNone(result.profile_input_lineage)
        self.assertEqual(
            result.profile_input_lineage.lineage_status,
            "unregistered_custom_profiles",
        )
        self.assertEqual(len(result.rows), 15)
        self.assertTrue(all(row["seed_count"] == 2 for row in result.rows))
        required = {
            "harm_variance",
            "harm_ci95_low",
            "harm_ci95_high",
            "monotonic_observed",
            "unstable",
        }
        self.assertTrue(all(required.issubset(row) for row in result.rows))
        expected_cases = {
            case.parameter
            for case in default_sensitivity_cases()
            if case.expected_direction != "none"
        }
        for parameter in expected_cases:
            parameter_rows = [
                row for row in result.rows if row["parameter"] == parameter
            ]
            self.assertTrue(all(row["monotonic_observed"] for row in parameter_rows))

    def test_identical_spec_is_exactly_reproducible(self) -> None:
        spec = PolicyBatchSpec(
            seeds=(8,),
            days=1,
            player_count=12,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        case = SensitivityCase(
            "paid_random_rewards", (0.0, 0.7), expected_direction="increasing"
        )
        kwargs = {
            "cases": (case,),
            "country_profiles": (CountryProfile(code="XX"),),
        }
        first = run_sensitivity_analysis(spec, **kwargs)
        second = run_sensitivity_analysis(spec, **kwargs)
        self.assertEqual(first, second)

    def test_invalid_sensitivity_definitions_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            SensitivityCase("unknown", (0.0, 1.0))
        with self.assertRaises(ValueError):
            SensitivityCase("paid_random_rewards", (0.7, 0.0))
        with self.assertRaises(ValueError):
            SensitivityCase("paid_random_rewards", (0.0,))
        with self.assertRaises(TypeError):
            SensitivityCase("paid_random_rewards", (False, 0.7))
        for parameter, values, message in (
            ("paid_random_rewards", (-0.1, 0.7), r"\[0, 1\]"),
            ("affordable_spending_share", (0.1, 1.1), r"\[0, 1\]"),
            ("decision_temperature", (0.0, 0.7), r"\(0, 5\]"),
            ("decision_temperature", (0.7, 5.1), r"\(0, 5\]"),
        ):
            with self.subTest(parameter=parameter, values=values):
                with self.assertRaisesRegex(ValueError, message):
                    SensitivityCase(parameter, values)
        spec = PolicyBatchSpec(seeds=(1,), days=0, player_count=0)
        with self.assertRaisesRegex(TypeError, "SensitivityCase"):
            run_sensitivity_analysis(
                spec,
                cases=(
                    _ExtendedSensitivityCase(
                        "paid_random_rewards",
                        (0.0, 0.7),
                    ),
                ),
                country_profiles=(CountryProfile(code="XX"),),
            )
        for threshold in (True, "0.35"):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(TypeError, "threshold"):
                    run_sensitivity_analysis(
                        spec,
                        cases=(
                            SensitivityCase(
                                "paid_random_rewards",
                                (0.0, 0.7),
                            ),
                        ),
                        instability_cv_threshold=threshold,  # type: ignore[arg-type]
                        country_profiles=(CountryProfile(code="XX"),),
                    )

    def test_duplicate_parameter_cases_are_rejected_without_case_id_leakage(self) -> None:
        duplicate_cases = (
            SensitivityCase(
                "paid_random_rewards",
                (0.0, 0.7),
                expected_direction="increasing",
            ),
            SensitivityCase(
                "paid_random_rewards",
                (0.1, 0.6),
                expected_direction="none",
            ),
        )
        with self.assertRaisesRegex(ValueError, "unique parameter names"):
            run_sensitivity_analysis(
                PolicyBatchSpec(
                    seeds=(3,),
                    days=0,
                    player_count=0,
                    decision_parameters=DecisionParameters(step_minutes=240),
                ),
                cases=duplicate_cases,
                country_profiles=(CountryProfile(code="XX"),),
            )

    def test_levels_are_normalized_to_python_floats(self) -> None:
        case = SensitivityCase(
            "paid_random_rewards",
            (np.int64(0), np.float32(0.7)),
        )
        self.assertEqual(case.values, (0.0, float(np.float32(0.7))))
        self.assertTrue(all(type(value) is float for value in case.values))

    def test_result_rejects_lineage_for_different_same_code_profile(self) -> None:
        profile = CountryProfile(code="XX")
        result = run_sensitivity_analysis(
            PolicyBatchSpec(
                seeds=(4,),
                days=0,
                player_count=0,
                decision_parameters=DecisionParameters(step_minutes=240),
            ),
            cases=(SensitivityCase("paid_random_rewards", (0.0, 0.7)),),
            country_profiles=(profile,),
        )

        with self.assertRaisesRegex(
            ValueError,
            "do not match the fingerprinted snapshot",
        ):
            replace(
                result,
                country_profiles=(replace(profile, awareness_mean=0.51),),
            )


if __name__ == "__main__":
    unittest.main()
