from __future__ import annotations

import unittest

import numpy as np

from microtx_sim.consumers.decision import (
    DecisionParameters,
    LifeAction,
    choose_life_action,
)
from microtx_sim.consumers.population import CountryProfile, initialize_player_table
from microtx_sim.consumers.welfare import initialize_player_life
from microtx_sim.domain.monetisation import MonetisationVector
from microtx_sim.rng import CounterRNG
from microtx_sim.simulation.policy_day import (
    advance_policy_day,
    create_policy_state,
)


def _cohort(size: int, seed: int = 71):
    rng = CounterRNG(seed)
    players = initialize_player_table(size, (CountryProfile(code="XX"),), rng)
    return players, initialize_player_life(players, rng), rng


class PolicyDecisionTests(unittest.TestCase):
    def test_purchase_alternative_obeys_cap_and_cooling_off(self) -> None:
        players, life, rng = _cohort(20)
        params = DecisionParameters(step_minutes=60)
        common = dict(
            players=players,
            life=life,
            rng=rng,
            tick=4,
            minute_of_day=720,
            remaining_sleep_minutes=life.sleep_need_minutes,
            remaining_work_study_minutes=life.work_study_obligation_minutes,
            remaining_social_minutes=life.social_obligation_minutes,
            remaining_physical_minutes=life.physical_activity_need_minutes,
            daily_play_minutes=np.zeros(len(players)),
            available_budget_cents=np.full(len(players), 10_000, dtype=np.int64),
            parameters=params,
        )
        capped = MonetisationVector(
            direct_price_cents=100,
            spending_cap_cents=200,
            purchase_friction=0.0,
        )
        choice = choose_life_action(
            mechanics=capped,
            cap_period_spend_cents=np.full(len(players), 200, dtype=np.int64),
            last_purchase_tick=np.full(len(players), -1, dtype=np.int64),
            **common,
        )
        self.assertTrue(
            np.all(np.isneginf(choice.deterministic_utilities[:, LifeAction.PURCHASE]))
        )
        cooling = MonetisationVector(
            direct_price_cents=100,
            cooling_off_hours=24,
            purchase_friction=0.0,
        )
        choice = choose_life_action(
            mechanics=cooling,
            cap_period_spend_cents=np.zeros(len(players), dtype=np.int64),
            last_purchase_tick=np.full(len(players), 3, dtype=np.int64),
            **common,
        )
        self.assertTrue(
            np.all(np.isneginf(choice.deterministic_utilities[:, LifeAction.PURCHASE]))
        )

    def test_day_conserves_time_and_budget_and_updates_state(self) -> None:
        players, life, rng = _cohort(80)
        state = create_policy_state(players, life)
        initial_budget = state.available_budget_cents.copy()
        result = advance_policy_day(
            players,
            state,
            MonetisationVector(
                direct_price_cents=199,
                opaque_virtual_currency=0.7,
                paid_random_rewards=0.6,
                time_limited_offers=0.7,
                purchase_friction=0.2,
                spending_cap_cents=500,
            ),
            rng,
            day=0,
            parameters=DecisionParameters(step_minutes=60),
        )
        np.testing.assert_array_equal(result.action_minutes.sum(axis=1), 1_440)
        self.assertTrue(np.all(state.player_spend_cents <= initial_budget))
        self.assertTrue(np.all(state.cap_period_spend_cents <= 500))
        np.testing.assert_array_equal(
            state.player_spend_by_source_cents.sum(axis=1),
            state.player_spend_cents,
        )
        np.testing.assert_array_equal(
            state.life.actual_play_minutes,
            result.action_minutes[:, LifeAction.PLAY],
        )
        self.assertEqual(state.completed_days, 1)

    def test_identical_seed_and_state_give_identical_day(self) -> None:
        players_a, life_a, rng_a = _cohort(50, 900)
        players_b, life_b, rng_b = _cohort(50, 900)
        state_a = create_policy_state(players_a, life_a)
        state_b = create_policy_state(players_b, life_b)
        mechanics = MonetisationVector(
            direct_price_cents=299,
            paid_random_rewards=0.5,
            time_limited_offers=0.5,
            purchase_friction=0.3,
        )
        params = DecisionParameters(step_minutes=120)
        result_a = advance_policy_day(
            players_a, state_a, mechanics, rng_a, day=0, parameters=params
        )
        result_b = advance_policy_day(
            players_b, state_b, mechanics, rng_b, day=0, parameters=params
        )
        np.testing.assert_array_equal(result_a.action_minutes, result_b.action_minutes)
        np.testing.assert_array_equal(result_a.spend_cents, result_b.spend_cents)
        np.testing.assert_array_equal(state_a.life.habit_strength, state_b.life.habit_strength)

    def test_zero_player_day_is_valid(self) -> None:
        players, life, rng = _cohort(0)
        state = create_policy_state(players, life)
        result = advance_policy_day(
            players,
            state,
            MonetisationVector(),
            rng,
            day=0,
            parameters=DecisionParameters(step_minutes=240),
        )
        self.assertEqual(result.action_minutes.shape, (0, len(LifeAction)))
        self.assertEqual(state.completed_days, 1)


if __name__ == "__main__":
    unittest.main()
