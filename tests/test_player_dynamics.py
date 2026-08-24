from __future__ import annotations

import inspect
import unittest

import numpy as np

from microtx_sim.core.ledger import Ledger
from microtx_sim.domain.games import GameTable
from microtx_sim.rng import CounterRNG
from microtx_sim.systems.initialization import CountryProfile, initialize_player_table
from microtx_sim.systems.player_dynamics import (
    PlayerDynamicsConfig,
    StepResult,
    step_player_dynamics,
)
from microtx_sim.types import HarmDimension, MonetisationMechanism, Motive


def _profile() -> CountryProfile:
    return CountryProfile(
        code="PP",
        monthly_income_median_cents=190_000,
        minor_allowance_median_cents=3_500,
        minor_stored_card_probability=0.40,
        minor_guardian_consent_probability=0.45,
        source_ids=("illustrative-test-profile",),
    )


def _minor_profile() -> CountryProfile:
    return CountryProfile(
        code="MN",
        adult_age=18,
        age_band_edges=(8, 18),
        age_band_weights=(1.0,),
        monthly_income_median_cents=180_000,
        minor_allowance_median_cents=6_000,
        minor_stored_card_probability=1.0,
        minor_guardian_consent_probability=0.0,
        guardian_supervision_mean=0.10,
        source_ids=("illustrative-minor-test-profile",),
    )


def _games(count: int = 5) -> GameTable:
    games = GameTable.create(
        game_count=count,
        company_count=max(1, min(count, 3)),
        stat_dimensions=4,
    )
    games.public_score[...] = np.linspace(0.25, 0.90, count)
    games.public_rank[...] = np.arange(1, count + 1, dtype=np.int64)
    games.monetisation[...] = np.linspace(
        0.35,
        0.88,
        count * len(MonetisationMechanism),
    ).reshape(count, len(MonetisationMechanism))
    games.price_cents[...] = 99 + 100 * np.arange(count, dtype=np.int64)
    return games


def _prepare_exposed_minors(
    *, stored_card: bool, supervision: float
) -> tuple[object, GameTable]:
    players = initialize_player_table(600, (_minor_profile(),), CounterRNG(901))
    players.current_game[...] = 0
    players.awareness[...] = 1.0
    players.has_stored_payment_access[...] = stored_card
    players.guardian_supervision[...] = supervision
    players.guardian_consent[...] = False
    players.household_id[...] = 0
    players.household_liquidity_cents[...] = 10_000
    players.traits[:, :] = 1.0
    players.traits[:, 4] = 0.0  # financial literacy
    players.traits[:, 5] = 0.0  # self control
    players.motive_weights[:, :] = 0.02
    players.motive_weights[:, Motive.COLLECTION] = 0.92

    games = _games(1)
    games.quality[...] = 1.0
    games.novelty[...] = 1.0
    games.competitive_integrity[...] = 1.0
    games.monetisation[...] = 1.0
    games.price_cents[...] = 99
    return players, games


class PlayerDynamicsTests(unittest.TestCase):
    def test_step_reconciles_money_rank_and_separate_harm_columns(self) -> None:
        players = initialize_player_table(1_200, (_profile(),), CounterRNG(73))
        games = _games()
        ledger = Ledger()
        liquid_before = players.liquidity_cents.copy()
        credit_before = players.credit_limit_cents.copy()

        result = step_player_dynamics(
            players,
            games,
            CounterRNG(812),
            ledger,
            tick=4,
            config=PlayerDynamicsConfig(
                chunk_size=37,
                base_unauthorised_card_hazard_per_exposed_minor_day=0.0,
            ),
        )

        self.assertIsInstance(result, StepResult)
        self.assertNotIn("world", inspect.signature(step_player_dynamics).parameters)
        self.assertEqual(result.player_spend_cents.dtype, np.dtype(np.int64))
        self.assertEqual(result.harm_delta.shape, (len(players), len(HarmDimension)))
        self.assertTrue(np.all(result.player_spend_cents >= 0))
        self.assertTrue(
            np.all(result.player_unsafe_spend_cents <= result.player_spend_cents)
        )
        self.assertEqual(
            int(result.game_revenue_cents.sum()),
            int(result.player_spend_cents.sum()),
        )
        self.assertEqual(
            int(games.revenue_cents.sum()), int(result.player_spend_cents.sum())
        )
        self.assertEqual(
            int(games.active_players.sum()), int(np.count_nonzero(result.active))
        )
        self.assertEqual(
            int((liquid_before - players.liquidity_cents).sum())
            + int((credit_before - players.credit_limit_cents).sum()),
            int(result.player_spend_cents.sum()),
        )
        self.assertEqual(
            sum(entry.amount_cents for entry in ledger.entries),
            int(result.player_spend_cents.sum()),
        )
        ledger.assert_balanced()

        for game_id in games.game_id:
            competitor = (result.chosen_game == game_id) & (result.matches_played > 0)
            ranks = np.sort(result.competitive_rank[competitor])
            np.testing.assert_array_equal(
                ranks, np.arange(1, ranks.size + 1, dtype=np.int32)
            )
        self.assertTrue(np.all(result.competitive_rank[result.matches_played == 0] == 0))
        self.assertTrue(
            np.all(result.matchmaking_bracket[result.matches_played == 0] == -1)
        )
        # The seven dimensions are stored and updated independently, not copied
        # from one composite or diagnosis column.
        self.assertGreater(np.unique(result.harm_delta, axis=1).shape[1], 1)

    def test_spend_never_exceeds_resources_and_unconsented_minors_are_blocked(self) -> None:
        players = initialize_player_table(1_000, (_profile(),), CounterRNG(18))
        players.guardian_consent[players.is_minor] = False
        games = _games(4)
        liquid_before = players.liquidity_cents.copy()
        credit_before = players.credit_limit_cents.copy()

        result = step_player_dynamics(
            players,
            games,
            CounterRNG(234),
            Ledger(),
            tick=9,
            config=PlayerDynamicsConfig(
                base_unauthorised_card_hazard_per_exposed_minor_day=0.0
            ),
        )

        adults = ~players.is_minor
        minors = players.is_minor
        self.assertTrue(
            np.all(
                result.player_spend_cents[adults]
                <= liquid_before[adults] + credit_before[adults]
            )
        )
        self.assertTrue(np.all(result.player_spend_cents[minors] == 0))
        self.assertTrue(np.all(result.player_unauthorised_spend_cents == 0))
        self.assertFalse(np.any(result.unauthorised_card_event))
        self.assertTrue(np.all(players.liquidity_cents >= 0))
        self.assertTrue(np.all(players.credit_limit_cents >= 0))

    def test_rare_card_hazard_requires_every_exposure_condition_and_is_capped(self) -> None:
        config = PlayerDynamicsConfig(
            base_unauthorised_card_hazard_per_exposed_minor_day=1.0,
            low_supervision_threshold=0.35,
            unauthorised_household_fraction=0.10,
            unauthorised_household_cap_cents=500,
        )
        missing_card, games = _prepare_exposed_minors(
            stored_card=False, supervision=0.0
        )
        no_card_result = step_player_dynamics(
            missing_card,
            games,
            CounterRNG(55),
            Ledger(),
            tick=1,
            config=config,
        )
        self.assertFalse(np.any(no_card_result.unauthorised_card_event))
        self.assertTrue(np.all(no_card_result.player_spend_cents == 0))

        supervised, games = _prepare_exposed_minors(
            stored_card=True, supervision=0.90
        )
        supervised_result = step_player_dynamics(
            supervised,
            games,
            CounterRNG(55),
            Ledger(),
            tick=1,
            config=config,
        )
        self.assertFalse(np.any(supervised_result.unauthorised_card_event))
        self.assertTrue(np.all(supervised_result.player_spend_cents == 0))

        exposed, games = _prepare_exposed_minors(
            stored_card=True, supervision=0.05
        )
        ledger = Ledger()
        exposed_result = step_player_dynamics(
            exposed,
            games,
            CounterRNG(55),
            ledger,
            tick=1,
            config=config,
        )
        self.assertTrue(np.all(exposed_result.unauthorised_card_event))
        self.assertGreater(int(exposed_result.player_spend_cents.sum()), 0)
        self.assertLessEqual(int(exposed_result.player_spend_cents.sum()), 500)
        np.testing.assert_array_equal(
            exposed_result.player_spend_cents,
            exposed_result.player_unauthorised_spend_cents,
        )
        self.assertTrue(
            all(entry.kind == "purchase_unauthorised" for entry in ledger.entries)
        )
        self.assertEqual(
            int(exposed.household_liquidity_cents[0]),
            10_000 - int(exposed_result.player_spend_cents.sum()),
        )
        self.assertTrue(
            np.all(
                exposed.household_liquidity_cents
                == exposed.household_liquidity_cents[0]
            )
        )

    def test_dense_choice_and_full_step_are_chunk_independent(self) -> None:
        first_players = initialize_player_table(
            777, (_profile(),), CounterRNG(602), tick=2
        )
        second_players = initialize_player_table(
            777, (_profile(),), CounterRNG(602), tick=2
        )
        first_games = _games(6)
        second_games = _games(6)
        first_ledger = Ledger()
        second_ledger = Ledger()
        base = PlayerDynamicsConfig(
            base_unauthorised_card_hazard_per_exposed_minor_day=0.0
        )

        first = step_player_dynamics(
            first_players,
            first_games,
            CounterRNG(919),
            first_ledger,
            tick=11,
            config=base,
            chunk_size=13,
        )
        second = step_player_dynamics(
            second_players,
            second_games,
            CounterRNG(919),
            second_ledger,
            tick=11,
            config=base,
            chunk_size=256,
        )

        for name in (
            "chosen_game",
            "active",
            "activity_minutes",
            "matches_played",
            "matchmaking_bracket",
            "competitive_rank",
            "competitive_rating",
            "player_spend_cents",
            "player_unsafe_spend_cents",
            "player_unauthorised_spend_cents",
            "unauthorised_card_event",
            "harm_delta",
            "game_revenue_cents",
            "game_unsafe_revenue_cents",
        ):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
        self.assertEqual(first.counters, second.counters)
        np.testing.assert_array_equal(
            first_players.liquidity_cents, second_players.liquidity_cents
        )
        np.testing.assert_array_equal(first_players.harm_state, second_players.harm_state)
        np.testing.assert_array_equal(first_games.revenue_cents, second_games.revenue_cents)
        self.assertEqual(first_ledger.entries, second_ledger.entries)


if __name__ == "__main__":
    unittest.main()
