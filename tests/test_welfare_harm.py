from __future__ import annotations

import unittest

import numpy as np

from microtx_sim.metrics.harm import (
    HarmComponent,
    WelfareHarmWeights,
    compute_welfare_harm,
)


def _inputs(size: int = 1) -> dict[str, np.ndarray]:
    return {
        "is_minor": np.zeros(size, dtype=np.bool_),
        "disposable_budget_cents": np.full(size, 100_000, dtype=np.int64),
        "intended_spending_limit_cents": np.full(size, 8_000, dtype=np.int64),
        "historical_spending_cents": np.zeros(size, dtype=np.int64),
        "spending_cents": np.full(size, 5_000, dtype=np.int64),
        "opaque_virtual_currency_exposure": np.zeros(size, dtype=np.float64),
        "paid_random_reward_exposure": np.zeros(size, dtype=np.float64),
        "time_pressure_exposure": np.zeros(size, dtype=np.float64),
        "actual_play_minutes": np.full(size, 180, dtype=np.int64),
        "planned_leisure_minutes": np.full(size, 240, dtype=np.int64),
        "sleep_need_minutes": np.full(size, 480, dtype=np.int64),
        "actual_sleep_minutes": np.full(size, 480, dtype=np.int64),
        "sleep_debt_minutes": np.zeros(size, dtype=np.int64),
        "work_study_obligation_minutes": np.full(size, 420, dtype=np.int64),
        "actual_work_study_minutes": np.full(size, 420, dtype=np.int64),
        "social_obligation_minutes": np.full(size, 90, dtype=np.int64),
        "actual_social_minutes": np.full(size, 90, dtype=np.int64),
        "physical_activity_need_minutes": np.full(size, 45, dtype=np.int64),
        "actual_physical_activity_minutes": np.full(size, 45, dtype=np.int64),
        "wellbeing_before": np.full(size, 0.8, dtype=np.float64),
        "wellbeing_after": np.full(size, 0.8, dtype=np.float64),
    }


class WelfareHarmTests(unittest.TestCase):
    def test_planned_transparent_affordable_spending_is_not_harm(self) -> None:
        result = compute_welfare_harm(**_inputs())

        self.assertEqual(int(result.harmful_spending_cents[0]), 0)
        self.assertEqual(int(result.unplanned_spending_cents[0]), 0)
        self.assertTrue(np.all(result.component_scores == 0.0))
        self.assertEqual(int(result.total_monetary_proxy_cents[0]), 0)

    def test_unplanned_opaque_random_and_pressured_spending_is_bounded(self) -> None:
        inputs = _inputs()
        inputs["disposable_budget_cents"][:] = 20_000
        inputs["intended_spending_limit_cents"][:] = 3_000
        inputs["historical_spending_cents"][:] = 2_500
        inputs["opaque_virtual_currency_exposure"][:] = 1.0
        inputs["paid_random_reward_exposure"][:] = 1.0
        inputs["time_pressure_exposure"][:] = 1.0
        result = compute_welfare_harm(**inputs)

        self.assertEqual(int(result.unplanned_spending_cents[0]), 4_500)
        self.assertGreaterEqual(
            int(result.harmful_spending_cents[0]),
            int(result.unplanned_spending_cents[0]),
        )
        self.assertLessEqual(
            int(result.harmful_spending_cents[0]), int(inputs["spending_cents"][0])
        )
        self.assertGreater(float(result.component(HarmComponent.M)[0]), 0.0)

    def test_opportunity_cost_requires_excess_play_and_real_deficits(self) -> None:
        within = _inputs()
        within["spending_cents"][:] = 0
        within["actual_sleep_minutes"][:] = 420
        within["actual_work_study_minutes"][:] = 360
        within["actual_social_minutes"][:] = 60
        no_excess = compute_welfare_harm(**within)

        self.assertEqual(float(no_excess.component(HarmComponent.OC)[0]), 0.0)
        self.assertEqual(float(no_excess.displaced_sleep_minutes[0]), 0.0)
        self.assertEqual(int(no_excess.opportunity_cost_proxy_cents[0]), 0)

        within["actual_play_minutes"][:] = 360
        excess = compute_welfare_harm(**within)
        displaced_total = (
            excess.displaced_sleep_minutes
            + excess.displaced_work_study_minutes
            + excess.displaced_social_minutes
            + excess.displaced_physical_activity_minutes
        )
        self.assertGreater(float(excess.component(HarmComponent.OC)[0]), 0.0)
        self.assertLessEqual(
            float(displaced_total[0]), float(excess.excess_play_minutes[0])
        )
        self.assertGreater(float(excess.component(HarmComponent.S)[0]), 0.0)
        self.assertGreater(float(excess.component(HarmComponent.E)[0]), 0.0)
        self.assertGreater(float(excess.component(HarmComponent.F)[0]), 0.0)

    def test_adult_and_youth_use_separate_valuation_and_wellbeing_stays_separate(self) -> None:
        inputs = _inputs(2)
        inputs["is_minor"][:] = (False, True)
        inputs["spending_cents"][:] = 0
        inputs["actual_play_minutes"][:] = 420
        inputs["actual_sleep_minutes"][:] = 420
        inputs["actual_work_study_minutes"][:] = 300
        inputs["actual_social_minutes"][:] = 45
        inputs["actual_physical_activity_minutes"][:] = 15
        inputs["wellbeing_after"][:] = 0.6
        result = compute_welfare_harm(**inputs)

        self.assertGreater(int(result.adult_opportunity_cost_proxy_cents[0]), 0)
        self.assertEqual(int(result.adult_opportunity_cost_proxy_cents[1]), 0)
        self.assertEqual(int(result.youth_opportunity_cost_proxy_cents[0]), 0)
        self.assertGreater(int(result.youth_opportunity_cost_proxy_cents[1]), 0)
        self.assertGreater(
            int(result.opportunity_cost_proxy_cents[0]),
            int(result.opportunity_cost_proxy_cents[1]),
        )
        np.testing.assert_allclose(result.component(HarmComponent.W), 0.2)

        wellbeing_only = WelfareHarmWeights(
            monetary=0.0,
            opportunity_cost=0.0,
            sleep=0.0,
            education_work=0.0,
            family_social=0.0,
            wellbeing=1.0,
        )
        np.testing.assert_allclose(
            result.composite_harm(wellbeing_only),
            result.component(HarmComponent.W),
        )

    def test_zero_player_and_invalid_input_edges(self) -> None:
        empty = _inputs(0)
        result = compute_welfare_harm(**empty)
        self.assertEqual(result.component_scores.shape, (0, len(HarmComponent)))
        self.assertEqual(result.composite_harm().shape, (0,))

        invalid = _inputs()
        invalid["paid_random_reward_exposure"][:] = 1.1
        with self.assertRaisesRegex(ValueError, "paid_random_reward_exposure"):
            compute_welfare_harm(**invalid)

        invalid = _inputs()
        invalid["spending_cents"][:] = -1
        with self.assertRaisesRegex(ValueError, "spending_cents"):
            compute_welfare_harm(**invalid)


if __name__ == "__main__":
    unittest.main()
