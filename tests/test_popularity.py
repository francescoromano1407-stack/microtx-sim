from __future__ import annotations

import unittest

import numpy as np

from microtx_sim.domain.games import GameTable
from microtx_sim.rng import CounterRNG
from microtx_sim.systems.initialization import CountryProfile, initialize_player_table
from microtx_sim.systems.popularity import PopularitySystem


class PopularityTests(unittest.TestCase):
    def test_truth_is_exact_and_public_ranking_is_delayed(self) -> None:
        players = initialize_player_table(
            12,
            (CountryProfile(code="XX"),),
            CounterRNG(3),
        )
        games = GameTable.create(game_count=3, company_count=2, stat_dimensions=4)
        players.current_game[:] = np.array([0] * 7 + [1] * 4 + [2], dtype=np.int32)
        system = PopularitySystem(game_count=3, delay_days=2, noise_sd=0.0)
        first = system.observe_truth(
            tick=0,
            players=players,
            games=games,
            period_revenue_cents=np.array([700, 800, 900], dtype=np.int64),
        )
        np.testing.assert_array_equal(first.active_players, [7, 4, 1])

        players.current_game[:] = np.array([2] * 8 + [1] * 3 + [0], dtype=np.int32)
        second = system.observe_truth(
            tick=1,
            players=players,
            games=games,
            period_revenue_cents=np.array([100, 600, 1600], dtype=np.int64),
        )
        self.assertIsNone(system.publish(tick=1, games=games, rng=CounterRNG(9)))
        publication = system.publish(tick=2, games=games, rng=CounterRNG(9))
        assert publication is not None
        self.assertEqual(publication.data_tick, first.tick)
        np.testing.assert_array_equal(games.active_players, second.active_players)
        self.assertEqual(set(publication.rank.tolist()), {1, 2, 3})

    def test_public_noise_is_reproducible_but_not_truth(self) -> None:
        players = initialize_player_table(
            30, (CountryProfile(code="XX"),), CounterRNG(4)
        )
        games_a = GameTable.create(game_count=3, company_count=2, stat_dimensions=3)
        games_b = GameTable.create(game_count=3, company_count=2, stat_dimensions=3)
        players.current_game[:] = np.arange(30, dtype=np.int32) % 3
        first = PopularitySystem(game_count=3, delay_days=0, noise_sd=0.2)
        second = PopularitySystem(game_count=3, delay_days=0, noise_sd=0.2)
        revenue = np.array([100, 200, 300], dtype=np.int64)
        truth = first.observe_truth(tick=0, players=players, games=games_a, period_revenue_cents=revenue)
        second.observe_truth(tick=0, players=players, games=games_b, period_revenue_cents=revenue)
        pub_a = first.publish(tick=0, games=games_a, rng=CounterRNG(88))
        pub_b = second.publish(tick=0, games=games_b, rng=CounterRNG(88))
        np.testing.assert_array_equal(pub_a.score, pub_b.score)
        self.assertTrue(np.any(pub_a.score != truth.score))


if __name__ == "__main__":
    unittest.main()
