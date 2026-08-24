from __future__ import annotations

import unittest

import numpy as np

from microtx_sim.causal import NullIntervention, run_paired_worlds
from microtx_sim.config import load_config
from microtx_sim.core.engine import SimulationEngine
from microtx_sim.core.world import World


class WorldIntegrationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
