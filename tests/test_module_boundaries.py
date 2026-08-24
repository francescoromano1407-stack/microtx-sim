from __future__ import annotations

import unittest

from microtx_sim.companies import FirmStrategySystem
from microtx_sim.consumers import PlayerDynamicsSystem
from microtx_sim.core.engine import SimulationEngine
from microtx_sim.core.world import World, WorldStep
from microtx_sim.market import PopularitySystem
from microtx_sim.simulation import SimulationOrchestrator
from microtx_sim.states import RegulationSystem
from microtx_sim.systems.firm_strategy import (
    FirmStrategySystem as LegacyFirmStrategySystem,
)
from microtx_sim.systems.player_dynamics import (
    PlayerDynamicsSystem as LegacyPlayerDynamicsSystem,
)
from microtx_sim.systems.popularity import PopularitySystem as LegacyPopularitySystem
from microtx_sim.systems.regulation import RegulationSystem as LegacyRegulationSystem


class ModuleBoundaryTests(unittest.TestCase):
    def test_domain_packages_are_canonical_and_legacy_imports_are_compatible(self) -> None:
        self.assertIs(FirmStrategySystem, LegacyFirmStrategySystem)
        self.assertIs(PlayerDynamicsSystem, LegacyPlayerDynamicsSystem)
        self.assertIs(PopularitySystem, LegacyPopularitySystem)
        self.assertIs(RegulationSystem, LegacyRegulationSystem)
        self.assertEqual(FirmStrategySystem.__module__, "microtx_sim.companies.logic")
        self.assertEqual(PlayerDynamicsSystem.__module__, "microtx_sim.consumers.logic")
        self.assertEqual(PopularitySystem.__module__, "microtx_sim.market.popularity")
        self.assertEqual(RegulationSystem.__module__, "microtx_sim.states.logic")

    def test_day_and_orchestration_are_not_implemented_on_world(self) -> None:
        self.assertEqual(WorldStep.__module__, "microtx_sim.simulation.day")
        self.assertTrue(issubclass(SimulationEngine, SimulationOrchestrator))
        for migrated_method in (
            "_renew_income",
            "_run_firm_decision",
            "_credit_firm_revenue",
            "_publish_ranking",
            "_run_audits",
            "_pay_subsidies",
            "_accrue_interest",
        ):
            self.assertFalse(hasattr(World, migrated_method), migrated_method)


if __name__ == "__main__":
    unittest.main()
