from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from microtx_sim.config import ConfigurationError, load_config
from microtx_sim.core.ledger import Ledger
from microtx_sim.domain.games import ContentPlanner, GameTable, OwnGameSnapshot
from microtx_sim.types import ProvenanceStatus


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndDomainTests(unittest.TestCase):
    def test_simulation_config_validates_the_root_seed_domain(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        maximum = (1 << 64) - 1
        replace(config, run=replace(config.run, seed=maximum)).validate()

        for value in (-1, 1 << 64):
            with self.subTest(value=value):
                invalid = replace(config, run=replace(config.run, seed=value))
                with self.assertRaisesRegex(
                    ConfigurationError,
                    r"run.seed must be in \[0, 2\*\*64 - 1\]",
                ):
                    invalid.validate()
        for value in (True, 1.0, np.int64(1)):
            with self.subTest(value=value):
                invalid = replace(config, run=replace(config.run, seed=value))
                with self.assertRaisesRegex(
                    ConfigurationError,
                    "run.seed must be a Python integer",
                ):
                    invalid.validate()

    def test_synthetic_smoke_is_valid_but_not_a_campaign(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        self.assertEqual(config.run.player_count, 384)
        with self.assertRaisesRegex(ConfigurationError, "CALIBRATED"):
            load_config(ROOT / "configs" / "smoke.toml", campaign=True)

    def test_campaign_rejects_synthetic_permission_even_if_status_is_calibrated(
        self,
    ) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        promoted = replace(
            config,
            meta=replace(
                config.meta,
                provenance_status=ProvenanceStatus.CALIBRATED,
            ),
        )

        with self.assertRaisesRegex(ConfigurationError, "allow_synthetic=false"):
            promoted.validate(campaign=True)

    def test_calendar_intervals_must_align_exactly_with_tick_size(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        misaligned = replace(config, run=replace(config.run, tick_days=2))
        with self.assertRaisesRegex(ConfigurationError, "divisible by tick_days"):
            misaligned.validate()

    def test_research_report_cost_must_be_positive(self) -> None:
        config = load_config(ROOT / "configs" / "smoke.toml")
        invalid = replace(
            config,
            information=replace(
                config.information,
                research_report_cost_cents=0,
            ),
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "research_report_cost_cents",
        ):
            invalid.validate()

    def test_ledger_balances_and_rejects_duplicate_reference(self) -> None:
        ledger = Ledger()
        ledger.transfer(
            tick=1,
            debit_account="player:7:liquid",
            credit_account="firm:2:cash",
            amount_cents=499,
            kind="purchase",
            reference="purchase-1",
        )
        ledger.transfer(
            tick=2,
            debit_account="firm:2:cash",
            credit_account="player:7:liquid",
            amount_cents=199,
            kind="refund",
            reference="refund-1",
        )
        ledger.assert_balanced()
        self.assertEqual(ledger.account_net_cents()["firm:2:cash"], 300)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ledger.transfer(
                tick=3,
                debit_account="player:7:liquid",
                credit_account="firm:2:cash",
                amount_cents=1,
                kind="purchase",
                reference="purchase-1",
            )

    def test_content_search_is_exact_and_never_fully_dominates(self) -> None:
        snapshot = OwnGameSnapshot(
            game_id=0,
            stat_frontier=np.array([1.0, 1.03, 1.06, 1.09]),
            demand_weights=np.array([0.45, 0.30, 0.20, 0.05]),
            active_players_estimate=10_000,
            price_cents=499,
            content_cost_cents=800_000,
            estimated_conversion=0.08,
            estimated_audit_probability=0.12,
            estimated_fine_cents=5_000_000,
            reputation_sensitivity=0.25,
            analytics_quality=0.7,
        )
        planner = ContentPlanner(boost_grid=(0.03, 0.07))
        candidates = planner.enumerate_candidates(snapshot)
        expected_masks = (2 ** len(snapshot.stat_frontier) - 2) * 2
        self.assertEqual(len(candidates), expected_masks)
        for candidate in candidates:
            candidate.validate_against(snapshot.stat_frontier)
            self.assertTrue(np.any(candidate.stats > snapshot.stat_frontier))
            self.assertTrue(np.any(candidate.stats < snapshot.stat_frontier))

        games = GameTable.create(game_count=2, company_count=2, stat_dimensions=4)
        chosen = planner.choose(snapshot)
        old_frontier = games.stat_frontier[0].copy()
        games.apply_content(chosen)
        self.assertTrue(np.all(games.stat_frontier[0] >= old_frontier))
        self.assertTrue(np.any(games.stat_frontier[0] > old_frontier))
        games.validate()


if __name__ == "__main__":
    unittest.main()
