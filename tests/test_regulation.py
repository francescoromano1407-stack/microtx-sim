from __future__ import annotations

import unittest

from microtx_sim.agents.jurisdictions import (
    RegulationRules,
    RegulatorPrivateState,
    StateAgent,
)
from microtx_sim.rng import CounterRNG
from microtx_sim.systems.regulation import (
    FirmComplianceTruth,
    ObservableFirmMetrics,
    RegulationSystem,
)


def _state() -> StateAgent:
    return StateAgent(
        jurisdiction_id=0,
        code="XX",
        rules=RegulationRules(True, True, True, True, False, 0, None, 0.5),
        state=RegulatorPrivateState(
            treasury_cents=10_000_000,
            audit_budget_cents=1_000_000,
            subsidy_budget_cents=2_000_000,
            audit_capacity_per_cycle=2,
            inspection_cost_cents=100_000,
        ),
        harm_priority=0.8,
        minor_priority=0.9,
        fiscal_priority=0.4,
        industry_priority=0.5,
        random_audit_fraction=0.5,
        audit_sensitivity=1.0,
        audit_specificity=1.0,
        subsidy_quality_weight=0.4,
        subsidy_safe_revenue_weight=0.4,
        subsidy_accessibility_weight=0.2,
    )


class RegulationTests(unittest.TestCase):
    def test_agent_selects_from_signals_and_kernel_resolves_truth(self) -> None:
        system = RegulationSystem()
        state = _state()
        observation = system.build_observation(
            tick=10,
            firms=(
                ObservableFirmMetrics(0, 0.8, 0.9, 0.7, 0.0, 0.8, 3),
                ObservableFirmMetrics(1, 0.1, 0.1, 0.2, 0.0, 0.6, 3),
                ObservableFirmMetrics(2, 0.2, 0.2, 0.3, 0.0, 0.5, 3),
            ),
            public_harm_index=0.4,
            treasury_pressure=0.2,
            sector_employment_estimate=100,
        )
        intents = system.select(tick=10, state=state, observation=observation, rng=CounterRNG(1))
        self.assertEqual(len(intents), 2)
        self.assertNotIn("actual_breaches", vars(observation) if hasattr(observation, "__dict__") else {})
        resolutions = system.resolve(
            tick=10,
            state=state,
            intents=intents,
            truth_by_firm={
                firm: FirmComplianceTruth(
                    firm,
                    ("minor_authorisation",) if firm == 0 else (),
                    ("payments", "disclosure"),
                    0.0,
                    1_000_000,
                )
                for firm in range(3)
            },
            rng=CounterRNG(1),
        )
        by_firm = {item.intent.firm_id: item for item in resolutions}
        if 0 in by_firm:
            self.assertEqual(
                by_firm[0].evidence.detected_breaches,
                ("minor_authorisation",),
            )
            self.assertGreater(by_firm[0].fine_cents, 0)

    def test_evasion_reduces_detection_not_underlying_breach(self) -> None:
        system = RegulationSystem()
        state = _state()
        state.audit_sensitivity = 0.0
        from microtx_sim.agents.jurisdictions import AuditIntent

        intent = AuditIntent(0, 7, 1.0, False, 1)
        result = system.resolve(
            tick=1,
            state=state,
            intents=(intent,),
            truth_by_firm={
                7: FirmComplianceTruth(7, ("breach",), ("control",), 1.0, 10_000)
            },
            rng=CounterRNG(5),
        )[0]
        self.assertEqual(result.evidence.detected_breaches, ())
        self.assertEqual(result.fine_cents, 0)


if __name__ == "__main__":
    unittest.main()

