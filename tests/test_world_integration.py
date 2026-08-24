from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from microtx_sim.causal import NullIntervention, run_paired_worlds
from microtx_sim.config import load_config
from microtx_sim.core.engine import SimulationEngine
from microtx_sim.core.world import World
from microtx_sim.data.profiles import ProfileValidationError
from microtx_sim.types import MonetisationMechanism


class WorldIntegrationTests(unittest.TestCase):
    def test_run_configuration_sets_the_firm_research_cost_scale(self) -> None:
        config = load_config("configs/smoke.toml")
        lower = replace(
            config,
            information=replace(
                config.information,
                research_report_cost_cents=100_000,
            ),
        )
        higher = replace(
            config,
            information=replace(
                config.information,
                research_report_cost_cents=900_000,
            ),
        )
        low_world = World.create(lower, campaign=False)
        high_world = World.create(higher, campaign=False)
        self.assertTrue(
            all(
                high.research_cost_cents > low.research_cost_cents
                for low, high in zip(low_world.firms, high_world.firms, strict=True)
            )
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("configs/smoke.toml")

    def test_three_cycle_smoke_connects_all_major_systems(self) -> None:
        world = World.create(self.config)
        result = SimulationEngine.run(world)

        self.assertEqual(result.cycles, 3)
        self.assertEqual(len(world.step_history), 3)
        self.assertEqual(len(result.final_outcome.player_spend_cents), 384)
        self.assertEqual(len(result.final_outcome.firm_cash_cents), 3)
        self.assertEqual(len(result.final_outcome.state_subsidy_outlay_cents), 4)
        self.assertGreater(
            sum(len(step.audit_resolutions) for step in world.step_history), 0
        )
        self.assertGreater(len(world.ledger.entries), 0)
        world.ledger.assert_balanced()

    def test_null_paired_worlds_are_exactly_identical(self) -> None:
        result = run_paired_worlds(
            self.config,
            treated=NullIntervention(),
            control=NullIntervention(),
            cycles=3,
        )

        paired = result.paired_outcome
        self.assertTrue(np.all(paired.player_harm_difference == 0.0))
        self.assertTrue(np.all(paired.player_spend_difference_cents == 0))
        self.assertTrue(np.all(paired.player_debt_difference_cents == 0))
        self.assertTrue(np.all(paired.firm_cash_difference_cents == 0))
        self.assertTrue(np.all(paired.state_subsidy_difference_cents == 0))
        self.assertEqual(result.effect.mean_composite_harm_effect, 0.0)
        self.assertEqual(result.effect.total_operating_margin_effect_cents, 0)

    def test_base_scenario_cannot_silently_consume_synthetic_profiles(self) -> None:
        base = load_config("configs/base.toml")
        with self.assertRaisesRegex(ProfileValidationError, "allow_synthetic=false"):
            World.create(base)

    def test_mechanism_cap_persists_across_firm_decisions(self) -> None:
        world = World.create(self.config)
        mechanism = MonetisationMechanism.RANDOM_REWARD
        world.cap_mechanism(mechanism=mechanism, maximum=0.10, game_ids=None)
        world.games.monetisation[:, int(mechanism)] = 0.90
        world.step()
        self.assertTrue(
            np.all(world.games.monetisation[:, int(mechanism)] <= 0.10)
        )

    def test_one_day_firm_and_ranking_schedule_has_one_public_pipeline(self) -> None:
        config = replace(
            self.config,
            market=replace(
                self.config.market,
                ranking_interval=1,
                firm_decision_interval=1,
            ),
            information=replace(
                self.config.information,
                public_signal_noise=1.0,
                public_signal_delay=1,
            ),
        )
        world = World.create(config)
        result = SimulationEngine.run(world, cycles=3)
        self.assertEqual(result.cycles, 3)
        self.assertEqual(len(world.step_history), 3)

    def test_subsidies_wait_for_review_and_use_one_home_jurisdiction(self) -> None:
        world = World.create(self.config)
        first = world.step()
        self.assertEqual(first.subsidies_paid_cents, 0)
        world.run(3)
        self.assertEqual(sum(int(value) for value in world.firm_subsidy_cents), 3_600_000)
        self.assertEqual(
            sum(int(value) for value in world.state_subsidy_outlay_cents),
            3_600_000,
        )
        self.assertLessEqual(int(world.state_subsidy_outlay_cents.max()), 1_200_000)


if __name__ == "__main__":
    unittest.main()
