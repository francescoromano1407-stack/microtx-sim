from __future__ import annotations

import unittest

import numpy as np

from microtx_sim.causal.interventions import MechanismCap, NullIntervention
from microtx_sim.causal.paired_worlds import compare_outcomes
from microtx_sim.metrics.outcomes import OutcomeSnapshot
from microtx_sim.types import MonetisationMechanism


def _outcome(harm_shift: float = 0.0, spend_shift: int = 0) -> OutcomeSnapshot:
    harm = np.zeros((3, 7), dtype=np.float64)
    harm[:, 0] = harm_shift
    return OutcomeSnapshot(
        tick=10,
        player_harm=harm,
        player_spend_cents=np.array([0, 100, 200], dtype=np.int64) + spend_shift,
        player_income_cents=np.array([100_000] * 3, dtype=np.int64),
        player_debt_cents=np.zeros(3, dtype=np.int64),
        firm_cash_cents=np.array([1_000_000], dtype=np.int64),
        firm_operating_margin_cents=np.array([100_000], dtype=np.int64),
        firm_safe_revenue_share=np.array([0.9], dtype=np.float64),
        state_subsidy_outlay_cents=np.array([0], dtype=np.int64),
    )


class _MockWorld:
    def __init__(self) -> None:
        self.calls: list[tuple[MonetisationMechanism, float, tuple[int, ...] | None]] = []

    def cap_mechanism(self, *, mechanism, maximum, game_ids) -> None:
        self.calls.append((mechanism, maximum, game_ids))


class CausalTests(unittest.TestCase):
    def test_null_difference_is_exactly_zero(self) -> None:
        outcome = _outcome()
        paired, effect = compare_outcomes(outcome, outcome)
        self.assertTrue(np.all(paired.player_harm_difference == 0.0))
        self.assertEqual(effect.total_spend_effect_cents, 0)
        self.assertEqual(effect.affected_player_share, 0.0)

    def test_effect_keeps_components_and_individual_pairing(self) -> None:
        treated = _outcome(harm_shift=0.7, spend_shift=50)
        control = _outcome()
        paired, effect = compare_outcomes(treated, control)
        self.assertEqual(paired.player_harm_difference.shape, (3, 7))
        self.assertEqual(effect.total_spend_effect_cents, 150)
        self.assertGreater(effect.mean_composite_harm_effect, 0.0)
        self.assertEqual(effect.affected_player_share, 1.0)

    def test_researcher_intervention_is_explicit(self) -> None:
        world = _MockWorld()
        NullIntervention().apply(world)
        self.assertEqual(world.calls, [])
        MechanismCap(
            MonetisationMechanism.RANDOM_REWARD,
            maximum=0.1,
            game_ids=(1, 2),
        ).apply(world)
        self.assertEqual(
            world.calls,
            [(MonetisationMechanism.RANDOM_REWARD, 0.1, (1, 2))],
        )


if __name__ == "__main__":
    unittest.main()

