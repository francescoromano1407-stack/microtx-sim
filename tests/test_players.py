from __future__ import annotations

import unittest

import numpy as np

from microtx_sim.agents.players import TRAIT_NAMES, classify_spend_segments
from microtx_sim.rng import CounterRNG
from microtx_sim.systems.initialization import CountryProfile, initialize_player_table
from microtx_sim.types import HarmDimension, Motive, SpendSegment


def _profiles() -> tuple[CountryProfile, CountryProfile]:
    return (
        CountryProfile(
            code="IT",
            population_weight=0.62,
            adult_age=18,
            monthly_income_median_cents=175_000,
            minor_stored_card_probability=0.10,
            guardian_supervision_mean=0.72,
            source_ids=("official-table-placeholder-it",),
        ),
        CountryProfile(
            code="SE",
            population_weight=0.38,
            adult_age=18,
            monthly_income_median_cents=285_000,
            minor_stored_card_probability=0.18,
            guardian_supervision_mean=0.61,
            awareness_mean=0.61,
            trait_means=(0.43, 0.50, 0.49, 0.50, 0.61, 0.57),
            source_ids=("official-table-placeholder-se",),
        ),
    )


class PlayerPopulationTests(unittest.TestCase):
    def test_player_table_columnar_invariants_and_minor_finances(self) -> None:
        players = initialize_player_table(4_000, _profiles(), CounterRNG(441))

        self.assertEqual(len(players), 4_000)
        self.assertEqual(players.player_id.dtype, np.dtype(np.int64))
        self.assertEqual(players.household_id.dtype, np.dtype(np.int64))
        for money in (
            players.monthly_disposable_income_cents,
            players.liquidity_cents,
            players.credit_limit_cents,
            players.allowance_cents,
            players.household_liquidity_cents,
        ):
            self.assertEqual(money.dtype, np.dtype(np.int64))
            self.assertTrue(np.all(money >= 0))

        self.assertEqual(players.traits.shape, (4_000, len(TRAIT_NAMES)))
        self.assertEqual(players.motive_weights.shape, (4_000, len(Motive)))
        self.assertEqual(players.harm_state.shape, (4_000, len(HarmDimension)))
        np.testing.assert_allclose(players.motive_weights.sum(axis=1), 1.0, atol=2e-6)
        self.assertTrue(np.all(players.current_game == -1))
        self.assertTrue(np.all(players.harm_state == 0.0))

        minors = players.is_minor
        adults = ~minors
        self.assertTrue(np.any(minors))
        self.assertTrue(np.any(adults))
        self.assertTrue(np.all(players.allowance_cents[adults] == 0))
        self.assertTrue(np.all(players.credit_limit_cents[minors] == 0))
        self.assertTrue(np.all(players.guardian_supervision[adults] == 0.0))
        self.assertTrue(np.any(players.allowance_cents[minors] > 0))
        # Card access and supervision are distinct state, so neither implies the other.
        self.assertEqual(players.has_stored_payment_access.dtype, np.dtype(np.bool_))
        self.assertEqual(players.guardian_supervision.dtype, np.dtype(np.float32))

    def test_traits_are_heterogeneous_correlated_and_motives_overlap(self) -> None:
        players = initialize_player_table(8_000, _profiles(), CounterRNG(9917))

        self.assertTrue(np.all(players.traits.std(axis=0) > 0.08))
        impulsivity = players.trait("impulsivity")
        self_control = players.trait("self_control")
        reward = players.trait("reward_sensitivity")
        self.assertLess(np.corrcoef(impulsivity, self_control)[0, 1], -0.30)
        self.assertGreater(np.corrcoef(impulsivity, reward)[0, 1], 0.25)
        self.assertTrue(np.all(players.motive_weights > 0.0))
        self.assertEqual(
            np.unique(np.argmax(players.motive_weights, axis=1)).size, len(Motive)
        )

    def test_baseline_vulnerability_is_immutable_but_harm_is_dynamic(self) -> None:
        players = initialize_player_table(100, _profiles(), CounterRNG(88))

        self.assertFalse(players.baseline_vulnerability.flags.writeable)
        with self.assertRaises(ValueError):
            players.baseline_vulnerability[0] = 1.0
        players.harm_state[0, 0] = 0.25
        self.assertAlmostEqual(float(players.harm_state[0, 0]), 0.25)

    def test_whale_is_ex_post_distribution_and_income_share_label(self) -> None:
        spend = np.array([0, 100, 200, 300, 400, 50_000], dtype=np.int64)
        income = np.array(
            [100_000, 100_000, 100_000, 100_000, 100_000, 80_000],
            dtype=np.int64,
        )
        segments = classify_spend_segments(
            spend,
            income,
            whale_quantile=0.80,
            whale_income_share=0.10,
        )

        self.assertEqual(segments[0], SpendSegment.NON_PAYER.value)
        self.assertEqual(segments[-1], SpendSegment.WHALE.value)
        self.assertIn(SpendSegment.MINNOW.value, segments)
        self.assertIn(SpendSegment.DOLPHIN.value, segments)

        # Being at the top of a spend distribution is insufficient if the
        # amount is negligible relative to that player's resources.
        high_income = income.copy()
        high_income[-1] = 10_000_000
        changed = classify_spend_segments(
            spend,
            high_income,
            whale_quantile=0.80,
            whale_income_share=0.10,
        )
        self.assertEqual(changed[-1], SpendSegment.DOLPHIN.value)

    def test_factory_is_reproducible_and_has_no_intrinsic_whale_column(self) -> None:
        first = initialize_player_table(300, _profiles(), CounterRNG(63), tick=4)
        second = initialize_player_table(300, _profiles(), CounterRNG(63), tick=4)

        for name in (
            "age_years",
            "jurisdiction",
            "monthly_disposable_income_cents",
            "traits",
            "motive_weights",
            "baseline_vulnerability",
        ):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertFalse(hasattr(first, "whale"))
        self.assertFalse(hasattr(first, "archetype"))


if __name__ == "__main__":
    unittest.main()
