from __future__ import annotations

from dataclasses import fields
import unittest

import numpy as np

from microtx_sim.consumers.population import CountryProfile, initialize_player_table
from microtx_sim.consumers.welfare import PlayerLifeTable, initialize_player_life
from microtx_sim.rng import CounterRNG


def _profile() -> CountryProfile:
    return CountryProfile(
        code="WL",
        age_band_edges=(8, 18, 65, 81),
        age_band_weights=(0.25, 0.60, 0.15),
        monthly_income_median_cents=210_000,
        source_ids=("illustrative-welfare-test",),
    )


class PlayerLifeTests(unittest.TestCase):
    def test_seeded_factory_is_aligned_heterogeneous_and_reproducible(self) -> None:
        players = initialize_player_table(2_000, (_profile(),), CounterRNG(14))
        first = initialize_player_life(players, CounterRNG(918), tick=3)
        second = initialize_player_life(players, CounterRNG(918), tick=3)

        self.assertIsInstance(first, PlayerLifeTable)
        first.validate_alignment(players)
        self.assertEqual(len(first), len(players))
        for descriptor in fields(first):
            np.testing.assert_array_equal(
                getattr(first, descriptor.name), getattr(second, descriptor.name)
            )

        self.assertTrue(
            np.all(first.intended_play_minutes <= first.planned_leisure_minutes)
        )
        self.assertGreater(float(first.planned_leisure_minutes.std()), 20.0)
        self.assertGreater(float(first.sleep_need_minutes.std()), 20.0)
        self.assertGreater(float(first.baseline_game_enjoyment.std()), 0.05)
        self.assertGreater(float(first.habit_strength.std()), 0.03)
        self.assertTrue(np.all(first.intended_spending_limit_cents >= 0))
        self.assertTrue(np.all(first.historical_spending_cents == 0))
        self.assertTrue(np.all(first.actual_play_minutes == 0))

    def test_baselines_are_immutable_while_dynamic_state_is_mutable(self) -> None:
        players = initialize_player_table(20, (_profile(),), CounterRNG(22))
        life = initialize_player_life(players, CounterRNG(23))

        for name in (
            "player_id",
            "planned_leisure_minutes",
            "baseline_game_enjoyment",
            "baseline_vulnerability",
            "intended_spending_limit_cents",
            "intended_play_minutes",
        ):
            self.assertFalse(getattr(life, name).flags.writeable)
        with self.assertRaises(ValueError):
            life.planned_leisure_minutes[0] = 0

        life.sleep_debt_minutes[0] += 10
        life.current_game_progression[0] = 0.5
        life.habit_strength[0] = 0.4
        life.reinforcement_state[0] = -0.2
        life.historical_spending_cents[0] = 199
        life.actual_play_minutes[0] = 45
        life.wellbeing[0] = 0.7
        self.assertEqual(int(life.historical_spending_cents[0]), 199)

    def test_zero_player_population_is_a_valid_seeded_edge_case(self) -> None:
        players = initialize_player_table(0, (_profile(),), CounterRNG(30))
        first = initialize_player_life(players, CounterRNG(31))
        second = initialize_player_life(players, CounterRNG(31))

        self.assertEqual(len(first), 0)
        self.assertGreater(first.nbytes, -1)
        for descriptor in fields(first):
            self.assertEqual(getattr(first, descriptor.name).shape, (0,))
            np.testing.assert_array_equal(
                getattr(first, descriptor.name), getattr(second, descriptor.name)
            )

    def test_alignment_and_tick_validation_fail_closed(self) -> None:
        players = initialize_player_table(5, (_profile(),), CounterRNG(40))
        other = initialize_player_table(
            5, (_profile(),), CounterRNG(40), first_player_id=100
        )
        life = initialize_player_life(players, CounterRNG(41))

        with self.assertRaisesRegex(ValueError, "not aligned"):
            life.validate_alignment(other)
        with self.assertRaisesRegex(ValueError, "tick"):
            initialize_player_life(players, CounterRNG(41), tick=-1)


if __name__ == "__main__":
    unittest.main()
