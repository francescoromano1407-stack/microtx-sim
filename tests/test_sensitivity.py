from __future__ import annotations

from dataclasses import replace
import unittest

from microtx_sim.analysis.sensitivity import (
    SensitivityCase,
    default_sensitivity_cases,
    run_sensitivity_analysis,
)
from microtx_sim.causal.batch import PolicyBatchSpec
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.consumers.population import CountryProfile


class SensitivityTests(unittest.TestCase):
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
