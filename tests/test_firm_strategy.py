from __future__ import annotations

import unittest

import numpy as np

from microtx_sim.agents.companies import CompanyObservation, FirmIntent
from microtx_sim.core.ledger import Ledger
from microtx_sim.domain.games import GameTable
from microtx_sim.rng import CounterRNG
from microtx_sim.systems.firm_strategy import (
    FirmStrategySystem,
    capture_period_telemetry,
    create_firms,
)
from microtx_sim.types import FirmAction, MonetisationMechanism


class _ObservationOnlyPolicy:
    def __init__(self, firm_id: int, received: list[CompanyObservation]) -> None:
        self.firm_id = firm_id
        self.received = received

    def __call__(self, observation: CompanyObservation) -> FirmIntent:
        self.received.append(observation)
        return FirmIntent(self.firm_id, FirmAction.HOLD, 0)


class FirmStrategyTests(unittest.TestCase):
    @staticmethod
    def _market(company_count: int, *, stat_dimensions: int = 4) -> GameTable:
        games = GameTable.create(
            game_count=company_count,
            company_count=company_count,
            stat_dimensions=stat_dimensions,
        )
        games.active_players[:] = np.arange(1, company_count + 1) * 12_000
        games.revenue_cents[:] = np.arange(1, company_count + 1) * 900_000
        games.true_popularity[:] = np.linspace(200.0, 20.0, company_count)
        games.public_score[:] = np.linspace(0.9, 0.2, company_count)
        return games

    def test_factory_is_heterogeneous_and_policy_receives_only_observation(self) -> None:
        games = self._market(4)
        rng = CounterRNG(91)
        firms = create_firms(company_count=4, games=games, rng=rng)
        trait_vectors = {
            (
                firm.risk_aversion,
                firm.compliance_culture,
                firm.ethics_weight,
                firm.analytics_capability,
                firm.exploration_tendency,
            )
            for firm in firms
        }
        self.assertEqual(len(trait_vectors), len(firms))
        self.assertGreater(len({firm.state.cash_cents for firm in firms}), 1)

        received: list[CompanyObservation] = []
        policies = {
            firm.firm_id: _ObservationOnlyPolicy(firm.firm_id, received)
            for firm in firms
        }
        system = FirmStrategySystem(
            firms,
            rng=rng,
            policies=policies,
            public_signal_delay=1,
            public_signal_noise=0.0,
        )
        old_rank = games.public_rank.copy()
        system.publish_public_ranking(tick=0, games=games)
        games.public_rank[:] = games.public_rank[::-1]
        system.publish_public_ranking(tick=1, games=games)
        telemetry = capture_period_telemetry(
            tick=1, games=games, firms=firms, rng=rng
        )
        intents = system.collect_intents(
            tick=1, games=games, period_telemetry=telemetry
        )

        self.assertEqual(len(intents), len(firms))
        self.assertEqual(len(received), len(firms))
        for firm, observation in zip(firms, received, strict=True):
            self.assertIsInstance(observation, CompanyObservation)
            self.assertFalse(hasattr(observation, "true_popularity"))
            self.assertFalse(hasattr(observation, "world"))
            expected = tuple(old_rank[game] for game in firm.state.game_ids)
            self.assertEqual(observation.public_rank_by_game, expected)
            self.assertEqual(observation.signal_age_days, 1)

        # Latent market columns cannot alter an observation when its two inputs
        # (eligible public ranking and own telemetry) are held fixed.
        first = system.build_observation(tick=1, telemetry=telemetry[0])
        games.true_popularity[:] = -9_999.0
        games.public_score[:] = 9_999.0
        second = system.build_observation(tick=1, telemetry=telemetry[0])
        self.assertEqual(first, second)

    def test_content_release_contains_tradeoff_and_cash_is_posted(self) -> None:
        games = self._market(3, stat_dimensions=5)
        rng = CounterRNG(27)
        firms = create_firms(company_count=3, games=games, rng=rng)
        system = FirmStrategySystem(firms, rng=rng, public_signal_noise=0.0)
        telemetry = capture_period_telemetry(
            tick=2, games=games, firms=firms, rng=rng
        )
        old_frontier = games.stat_frontier[0].copy()
        before_cash = {firm.firm_id: firm.state.cash_cents for firm in firms}
        intents = (
            FirmIntent(
                0,
                FirmAction.RELEASE_CONTENT,
                500_000,
                target_game_id=0,
                committed_cost_cents=80_000,
            ),
            FirmIntent(
                1,
                FirmAction.BUY_RESEARCH,
                100_000,
                committed_cost_cents=35_000,
            ),
            FirmIntent(
                2,
                FirmAction.ACQUIRE_USERS,
                100_000,
                target_game_id=2,
                committed_cost_cents=25_000,
            ),
        )
        ledger = Ledger()
        result = system.resolve(
            tick=2,
            games=games,
            intents=intents,
            ledger=ledger,
            period_telemetry=telemetry,
        )

        content = result.records[0].content_candidate
        self.assertIsNotNone(content)
        assert content is not None
        content.validate_against(old_frontier)
        self.assertTrue(np.any(content.stats > old_frontier))
        self.assertTrue(np.any(content.stats < old_frontier))
        self.assertTrue(np.any(games.stat_frontier[0] > old_frontier))
        self.assertGreater(result.promotion_pressure[2], 0.0)
        self.assertGreater(firms[1].state.analytics_investment, 0.0)

        ledger.assert_balanced()
        net = ledger.account_net_cents()
        for firm, record in zip(firms, result.records, strict=True):
            cash_delta = before_cash[firm.firm_id] - firm.state.cash_cents
            self.assertEqual(cash_delta, record.charged_cents)
            self.assertEqual(net[f"firm:{firm.firm_id}:cash"], -cash_delta)
        self.assertEqual(sum(record.charged_cents for record in result.records), 140_000)

    def test_ordinary_actions_and_hidden_evasion_kernel_have_effects(self) -> None:
        games = self._market(6)
        rng = CounterRNG(314)
        firms = create_firms(company_count=6, games=games, rng=rng)
        system = FirmStrategySystem(firms, rng=rng)
        telemetry = capture_period_telemetry(
            tick=3, games=games, firms=firms, rng=rng
        )
        old_monetisation = float(
            games.monetisation[0, MonetisationMechanism.RANDOM_REWARD]
        )
        before_cash = sum(firm.state.cash_cents for firm in firms)
        intents = (
            FirmIntent(
                0,
                FirmAction.ADJUST_MONETISATION,
                1,
                target_game_id=0,
                mechanism=MonetisationMechanism.RANDOM_REWARD,
                intensity_delta=0.07,
            ),
            FirmIntent(
                1,
                FirmAction.BUY_RESEARCH,
                1,
                committed_cost_cents=20_000,
            ),
            FirmIntent(
                2,
                FirmAction.INVEST_COMPLIANCE,
                1,
                committed_cost_cents=22_000,
            ),
            FirmIntent(
                3,
                FirmAction.ACQUIRE_USERS,
                1,
                target_game_id=3,
                committed_cost_cents=18_000,
            ),
            FirmIntent(4, FirmAction.APPLY_SUBSIDY, 1),
            FirmIntent(5, FirmAction.EVADE, 1, target_game_id=5),
        )
        ledger = Ledger()
        result = system.resolve(
            tick=3,
            games=games,
            intents=intents,
            ledger=ledger,
            period_telemetry=telemetry,
        )

        self.assertAlmostEqual(
            games.monetisation[0, MonetisationMechanism.RANDOM_REWARD],
            old_monetisation + 0.07,
        )
        self.assertGreater(firms[1].state.analytics_investment, 0.0)
        self.assertGreater(firms[2].state.compliance_investment, 0.0)
        self.assertGreater(result.promotion_pressure[3], 0.0)
        self.assertEqual([item.firm_id for item in result.subsidy_applications], [4])
        evasion = result.firm_kernel_state[5]
        self.assertGreater(evasion.evasion_level, 0.0)
        self.assertGreater(evasion.detection_risk, 0.0)
        self.assertGreater(evasion.hidden_savings_cents, 0)
        self.assertNotIn("evasion_level", CompanyObservation.__dataclass_fields__)

        charged = sum(record.charged_cents for record in result.records)
        after_cash = sum(firm.state.cash_cents for firm in firms)
        self.assertEqual(before_cash - after_cash, charged)
        self.assertEqual(ledger.total_flow_cents(), charged)
        ledger.assert_balanced()

    def test_agreements_require_exact_reciprocal_compatible_proposals(self) -> None:
        games = self._market(5)
        rng = CounterRNG(808)
        firms = create_firms(company_count=5, games=games, rng=rng)
        system = FirmStrategySystem(firms, rng=rng)
        old_collusive = games.monetisation.copy()
        intents = (
            FirmIntent(
                0,
                FirmAction.PROPOSE_COLLABORATION,
                1,
                target_firm_id=1,
            ),
            FirmIntent(
                1,
                FirmAction.PROPOSE_COLLABORATION,
                1,
                target_firm_id=0,
            ),
            FirmIntent(
                2,
                FirmAction.PROPOSE_COLLUSION,
                1,
                target_firm_id=3,
            ),
            FirmIntent(
                3,
                FirmAction.PROPOSE_COLLUSION,
                1,
                target_firm_id=2,
            ),
            FirmIntent(
                4,
                FirmAction.PROPOSE_COLLABORATION,
                1,
                target_firm_id=0,
            ),
        )
        result = system.resolve(
            tick=4,
            games=games,
            intents=intents,
            ledger=Ledger(),
        )

        self.assertEqual(result.collaborations, ((0, 1),))
        self.assertEqual(result.collusions, ((2, 3),))
        self.assertTrue(result.records[0].accepted)
        self.assertTrue(result.records[1].accepted)
        self.assertTrue(result.records[2].accepted)
        self.assertTrue(result.records[3].accepted)
        self.assertFalse(result.records[4].accepted)
        self.assertIn("reciprocal", result.records[0].reason)
        self.assertIn("no compatible", result.records[4].reason)
        self.assertGreater(result.promotion_pressure[[0, 1]].sum(), 0.0)
        self.assertTrue(np.any(games.monetisation[[2, 3]] > old_collusive[[2, 3]]))
        self.assertGreater(result.firm_kernel_state[2].collusion_exposure, 0.0)
        self.assertEqual(result.firm_kernel_state[0].collusion_exposure, 0.0)

    def test_default_step_collects_before_resolving_and_logs_every_firm(self) -> None:
        games = self._market(4)
        rng = CounterRNG(1_337)
        firms = create_firms(company_count=4, games=games, rng=rng)
        system = FirmStrategySystem(firms, rng=rng)
        telemetry = capture_period_telemetry(
            tick=0, games=games, firms=firms, rng=rng
        )
        ledger = Ledger()

        result = system.step(
            tick=0,
            games=games,
            period_telemetry=telemetry,
            ledger=ledger,
        )

        self.assertEqual(len(result.intents), len(firms))
        self.assertEqual(len(result.records), len(firms))
        self.assertEqual(system.intent_log, result.records)
        self.assertTrue(
            all(record.intent.information_fingerprint for record in result.records)
        )
        ledger.assert_balanced()


if __name__ == "__main__":
    unittest.main()
