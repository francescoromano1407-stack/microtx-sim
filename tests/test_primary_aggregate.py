from __future__ import annotations

import inspect
from math import sqrt
import unittest

from microtx_sim.causal.analysis_plan import PrimaryAggregateRule
from microtx_sim.causal.primary_aggregate import (
    NORMAL_95_MONTE_CARLO_INTERVAL,
    PrimaryAggregateValidationError,
    PrimarySeedRealization,
    summarize_primary_realizations,
)


def _rule() -> PrimaryAggregateRule:
    return PrimaryAggregateRule(
        positive_result_interpretation="comparison is higher than reference",
        negative_result_interpretation="comparison is lower than reference",
    )


def _realization(seed: int, numerator: int, denominator: int = 1):
    return PrimarySeedRealization(
        seed=seed,
        numerator=numerator,
        denominator=denominator,
        result_sha256=f"{seed % 10}" * 64,
    )


class PrimaryAggregateArithmeticTests(unittest.TestCase):
    def test_known_mean_sample_sd_mcse_and_normal_interval(self) -> None:
        summary = summarize_primary_realizations(
            _rule(),
            expected_seeds=(1, 2, 3),
            realizations=(
                _realization(1, 1),
                _realization(2, 2),
                _realization(3, 3),
            ),
        )
        self.assertEqual(summary.point_estimate, 2.0)
        self.assertEqual(summary.between_seed_sample_standard_deviation, 1.0)
        self.assertAlmostEqual(summary.monte_carlo_standard_error, 1 / sqrt(3))
        self.assertAlmostEqual(
            summary.interval_lower,
            2.0 - 1.96 / sqrt(3),
        )
        self.assertAlmostEqual(
            summary.interval_upper,
            2.0 + 1.96 / sqrt(3),
        )
        self.assertEqual(summary.retained_seed_count, 3)
        self.assertEqual(summary.excluded_seed_count, 0)

    def test_equal_seed_weighting_is_distinct_from_population_weighting(self) -> None:
        summary = summarize_primary_realizations(
            _rule(),
            expected_seeds=(10, 20),
            realizations=(
                _realization(10, 1, 4),
                _realization(20, 9, 4),
            ),
        )
        self.assertEqual(summary.point_estimate, 1.25)
        self.assertEqual(
            _rule().snapshot()["population_weight_application"],
            "WITHIN_EACH_SEED_BEFORE_CROSS_SEED_AGGREGATION",
        )
        self.assertEqual(_rule().snapshot()["seed_weighting"], "EQUAL")
        self.assertEqual(_rule().snapshot()["scenario_weights"], [])

    def test_one_seed_has_zero_sd_mcse_and_interval_width(self) -> None:
        summary = summarize_primary_realizations(
            _rule(),
            expected_seeds=(7,),
            realizations=(_realization(7, -3, 2),),
        )
        self.assertEqual(summary.point_estimate, -1.5)
        self.assertEqual(summary.between_seed_sample_standard_deviation, 0.0)
        self.assertEqual(summary.monte_carlo_standard_error, 0.0)
        self.assertEqual(summary.interval_lower, -1.5)
        self.assertEqual(summary.interval_upper, -1.5)

    def test_missing_duplicate_or_reordered_pairs_fail_closed(self) -> None:
        cases = (
            (_realization(1, 1),),
            (_realization(1, 1), _realization(1, 2)),
            (_realization(2, 2), _realization(1, 1)),
        )
        for realizations in cases:
            with self.subTest(realizations=realizations):
                with self.assertRaisesRegex(
                    PrimaryAggregateValidationError,
                    "exactly cover the fixed seed set",
                ):
                    summarize_primary_realizations(
                        _rule(),
                        expected_seeds=(1, 2),
                        realizations=realizations,
                    )

    def test_invalid_observations_are_not_excluded(self) -> None:
        with self.assertRaisesRegex(
            PrimaryAggregateValidationError,
            "denominator must be positive",
        ):
            _realization(1, 1, 0)
        with self.assertRaises(TypeError):
            summarize_primary_realizations(
                _rule(),
                expected_seeds=(1,),
                realizations=(float("nan"),),  # type: ignore[arg-type]
            )

    def test_api_has_no_outcome_dependent_exclusion_channel(self) -> None:
        parameters = inspect.signature(summarize_primary_realizations).parameters
        self.assertNotIn("exclude", parameters)
        self.assertFalse(
            _rule().snapshot()["outcome_dependent_exclusion_allowed"]
        )
        self.assertEqual(_rule().snapshot()["exclusion_criteria"], [])
        with self.assertRaises(TypeError):
            summarize_primary_realizations(
                _rule(),
                expected_seeds=(1,),
                realizations=(_realization(1, 99),),
                exclude=lambda value: value > 0,  # type: ignore[call-arg]
            )

    def test_interval_label_is_explicitly_monte_carlo(self) -> None:
        self.assertEqual(
            NORMAL_95_MONTE_CARLO_INTERVAL,
            "NORMAL_95_MONTE_CARLO_MEAN_PLUS_MINUS_1.96_MCSE",
        )
        self.assertIn(
            "not a confidence interval for a real-world population",
            _rule().snapshot()["interval_interpretation"],
        )


if __name__ == "__main__":
    unittest.main()
