from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from microtx_sim.agents.jurisdictions import (
    AuditEvidence,
    AuditIntent,
    SubsidyApplicationView,
)
from microtx_sim.config import load_config
from microtx_sim.core.world import World
from microtx_sim.simulation import government_phase
from microtx_sim.simulation.accounting import (
    INT64_MAX,
    _checked_grouped_money,
    credit_firm_revenue,
)
from microtx_sim.simulation.policy_day import PURCHASE_REVENUE_SOURCES
from microtx_sim.simulation.policy_orchestrator import (
    ProducerAssumptions,
    _conventional_revenue,
)
from microtx_sim.states.logic import (
    AuditResolution,
    ObservableFirmMetrics,
    RegulationSystem,
)


ROOT = Path(__file__).resolve().parents[1]


def _world() -> World:
    return World.create(load_config(ROOT / "configs" / "smoke.toml"))


def _array_bytes(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    return array.dtype.str, array.shape, array.tobytes()


def _financial_snapshot(world: World) -> dict[str, object]:
    arrays = (
        "firm_revenue_cents",
        "firm_unsafe_revenue_cents",
        "firm_subsidy_cents",
        "firm_fine_assessed_cents",
        "firm_fine_paid_cents",
        "state_subsidy_outlay_cents",
        "_public_detections",
    )
    return {
        "firm_cash": tuple(firm.state.cash_cents for firm in world.firms),
        "states": deepcopy(world.states),
        "arrays": tuple(
            (name, _array_bytes(getattr(world, name))) for name in arrays
        ),
        "ledger": world.ledger.entries,
        "pending_subsidies": tuple(world._pending_subsidies),
    }


def _revenue_result(
    world: World,
    *,
    revenue: np.ndarray | None = None,
    unsafe: np.ndarray | None = None,
) -> SimpleNamespace:
    game_count = len(world.games.game_id)
    return SimpleNamespace(
        game_revenue_cents=(
            np.zeros(game_count, dtype=np.int64) if revenue is None else revenue
        ),
        game_unsafe_revenue_cents=(
            np.zeros(game_count, dtype=np.int64) if unsafe is None else unsafe
        ),
    )


def _policy_revenue_state(
    values: tuple[int, ...],
    *,
    channel: str,
) -> SimpleNamespace:
    cents = np.asarray(values, dtype=np.int64)
    purchase = np.zeros(
        (len(cents), len(PURCHASE_REVENUE_SOURCES)),
        dtype=np.int64,
    )
    fixed = np.zeros(len(cents), dtype=np.int64)
    subscription = np.zeros(len(cents), dtype=np.int64)
    if channel == "direct_purchase":
        purchase[:, 0] = cents
    elif channel == "fixed_price":
        fixed[:] = cents
    elif channel == "subscription":
        subscription[:] = cents
    else:
        raise AssertionError(channel)
    return SimpleNamespace(
        player_spend_by_source_cents=purchase,
        access_fixed_cents=fixed,
        access_subscription_cents=subscription,
    )


def _policy_scenario_without_access_revenue() -> SimpleNamespace:
    return SimpleNamespace(
        fixed_access_price_cents=0,
        subscription_price_cents=0,
    )


def _run_controlled_audits(
    world: World,
    *,
    fines: tuple[tuple[int, int], ...],
    detected_firms: frozenset[int] | None = None,
    tick: int = 7,
) -> tuple[AuditResolution, ...]:
    detected = (
        frozenset(firm_id for firm_id, _ in fines)
        if detected_firms is None
        else detected_firms
    )
    firm_ids = tuple(firm_id for firm_id, _ in fines)
    fine_by_firm = dict(fines)

    def metrics(
        controlled_world: World,
        *,
        tick: int,
        player_result: object,
        jurisdiction_id: int,
    ) -> tuple[ObservableFirmMetrics, ...]:
        del tick, player_result, jurisdiction_id
        return tuple(
            ObservableFirmMetrics(
                firm.firm_id,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0,
            )
            for firm in controlled_world.firms
        )

    def select(
        system: RegulationSystem,
        *,
        tick: int,
        state: object,
        observation: object,
        rng: object,
    ) -> tuple[AuditIntent, ...]:
        del system, observation, rng
        if state.jurisdiction_id != 0:
            return ()
        return tuple(
            AuditIntent(
                jurisdiction_id=0,
                firm_id=firm_id,
                risk_score=1.0,
                random_floor_selection=False,
                information_as_of=tick,
            )
            for firm_id in firm_ids
        )

    def resolve(
        system: RegulationSystem,
        *,
        tick: int,
        state: object,
        intents: tuple[AuditIntent, ...],
        truth_by_firm: object,
        rng: object,
    ) -> tuple[AuditResolution, ...]:
        del system, truth_by_firm, rng
        resolutions: list[AuditResolution] = []
        for intent in intents:
            breaches = ("controlled_breach",) if intent.firm_id in detected else ()
            evidence = AuditEvidence(
                jurisdiction_id=state.jurisdiction_id,
                firm_id=intent.firm_id,
                detected_breaches=breaches,
                tested_controls=("controlled",),
                evidence_strength=1.0,
                false_positive_probability=0.0,
                tick=tick,
            )
            state.observe_audit(evidence)
            resolutions.append(
                AuditResolution(
                    intent=intent,
                    evidence=evidence,
                    fine_cents=fine_by_firm[intent.firm_id],
                )
            )
        audit_cost = len(resolutions) * state.state.inspection_cost_cents
        state.state.audit_budget_cents = max(
            0,
            state.state.audit_budget_cents - audit_cost,
        )
        state.state.treasury_cents = max(
            0,
            state.state.treasury_cents - audit_cost,
        )
        return tuple(resolutions)

    with (
        patch.object(government_phase, "build_observable_firm_metrics", metrics),
        patch.object(government_phase, "build_compliance_truth", return_value={}),
        patch.object(RegulationSystem, "select", new=select),
        patch.object(RegulationSystem, "resolve", new=resolve),
    ):
        return government_phase.run_audits(
            world,
            tick=tick,
            player_result=object(),
        )


def _application(
    firm_id: int,
    requested_cents: int,
    jurisdiction_id: int,
    *,
    quality: float = 1.0,
) -> SubsidyApplicationView:
    return SubsidyApplicationView(
        firm_id=firm_id,
        requested_cents=requested_cents,
        verified_quality=quality,
        verified_design_safety_score=quality,
        verified_accessibility=quality,
        jobs_estimate=100,
        evidence_age_days=0,
        submitted_tick=0,
        eligible_jurisdictions=(jurisdiction_id,),
    )


class FirmRevenueBoundaryTests(unittest.TestCase):
    def test_exact_int64_boundary_succeeds_transactionally(self) -> None:
        world = _world()
        firm_id = 0
        rows = np.flatnonzero(world.games.company_id == firm_id)
        self.assertGreaterEqual(len(rows), 2)
        revenue = np.zeros(len(world.games.game_id), dtype=np.int64)
        unsafe = np.zeros_like(revenue)
        revenue[rows[:2]] = (INT64_MAX - 1, 1)
        unsafe[rows[:2]] = (INT64_MAX - 1, 1)
        world.firms[firm_id].state.cash_cents = 0

        credit_firm_revenue(
            world,
            _revenue_result(world, revenue=revenue, unsafe=unsafe),
        )

        self.assertEqual(world.firms[firm_id].state.cash_cents, INT64_MAX)
        self.assertEqual(int(world.firm_revenue_cents[firm_id]), INT64_MAX)
        self.assertEqual(
            int(world.firm_unsafe_revenue_cents[firm_id]),
            INT64_MAX,
        )

    def test_each_overflow_site_leaves_all_financial_state_unchanged(self) -> None:
        cases = (
            "grouped_revenue",
            "firm_cash",
            "cumulative_revenue",
            "cumulative_unsafe_revenue",
        )
        for case in cases:
            with self.subTest(case=case):
                world = _world()
                firm_id = 0
                rows = np.flatnonzero(world.games.company_id == firm_id)
                revenue = np.zeros(len(world.games.game_id), dtype=np.int64)
                unsafe = np.zeros_like(revenue)
                if case == "grouped_revenue":
                    revenue[rows[:2]] = (INT64_MAX, 1)
                elif case == "firm_cash":
                    revenue[rows[0]] = 1
                    world.firms[firm_id].state.cash_cents = INT64_MAX
                elif case == "cumulative_revenue":
                    revenue[rows[0]] = 1
                    world.firm_revenue_cents[firm_id] = INT64_MAX
                else:
                    unsafe[rows[0]] = 1
                    world.firm_unsafe_revenue_cents[firm_id] = INT64_MAX
                before = _financial_snapshot(world)

                with self.assertRaises(OverflowError):
                    credit_firm_revenue(
                        world,
                        _revenue_result(
                            world,
                            revenue=revenue,
                            unsafe=unsafe,
                        ),
                    )

                self.assertEqual(_financial_snapshot(world), before)

    def test_group_ids_must_be_integer_typed(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            _checked_grouped_money(
                np.asarray([1], dtype=np.int64),
                np.asarray([0.0], dtype=np.float64),
                1,
                label="test grouping",
            )


class PolicyRevenueBoundaryTests(unittest.TestCase):
    def test_each_policy_revenue_channel_accepts_exact_int64_maximum(self) -> None:
        for channel in ("direct_purchase", "fixed_price", "subscription"):
            with self.subTest(channel=channel):
                revenue = _conventional_revenue(
                    _policy_revenue_state(
                        (INT64_MAX - 1, 1),
                        channel=channel,
                    ),
                    _policy_scenario_without_access_revenue(),
                    ProducerAssumptions(),
                )

                self.assertEqual(revenue[channel], INT64_MAX)

    def test_each_policy_revenue_channel_rejects_int64_maximum_plus_one(
        self,
    ) -> None:
        for channel in ("direct_purchase", "fixed_price", "subscription"):
            with self.subTest(channel=channel):
                with self.assertRaisesRegex(OverflowError, "outside int64"):
                    _conventional_revenue(
                        _policy_revenue_state(
                            (INT64_MAX, 1),
                            channel=channel,
                        ),
                        _policy_scenario_without_access_revenue(),
                        ProducerAssumptions(),
                    )

    def test_policy_revenue_rejects_vector_that_would_wrap_to_zero(self) -> None:
        with self.assertRaisesRegex(OverflowError, "outside int64"):
            _conventional_revenue(
                _policy_revenue_state(
                    (INT64_MAX, INT64_MAX, 2),
                    channel="direct_purchase",
                ),
                _policy_scenario_without_access_revenue(),
                ProducerAssumptions(),
            )

    def test_policy_revenue_rejects_negative_component_even_if_net_is_zero(
        self,
    ) -> None:
        with self.assertRaisesRegex(OverflowError, "outside int64"):
            _conventional_revenue(
                _policy_revenue_state(
                    (-1, 1),
                    channel="direct_purchase",
                ),
                _policy_scenario_without_access_revenue(),
                ProducerAssumptions(),
            )


class AuditBoundaryTests(unittest.TestCase):
    def test_exact_int64_boundaries_succeed_and_references_are_unique(self) -> None:
        world = _world()
        state = world.states[0]
        state.state.treasury_cents = INT64_MAX
        state.state.audit_capacity_per_cycle = 1
        state.state.inspection_cost_cents = 1
        world.firms[0].state.cash_cents = 1
        world.firm_fine_assessed_cents[0] = INT64_MAX - 1
        world.firm_fine_paid_cents[0] = INT64_MAX - 1
        world._public_detections[0] = INT64_MAX - 1

        resolutions = _run_controlled_audits(world, fines=((0, 1),))

        self.assertEqual(len(resolutions), 1)
        self.assertEqual(state.state.treasury_cents, INT64_MAX)
        self.assertEqual(state.state.audit_budget_cents, 0)
        self.assertEqual(world.firms[0].state.cash_cents, 0)
        self.assertEqual(int(world.firm_fine_assessed_cents[0]), INT64_MAX)
        self.assertEqual(int(world.firm_fine_paid_cents[0]), INT64_MAX)
        self.assertEqual(int(world._public_detections[0]), INT64_MAX)
        references = tuple(entry.reference for entry in world.ledger.entries)
        self.assertEqual(references, ("audit:7:0", "fine:7:0:0"))
        self.assertEqual(len(references), len(set(references)))

    def test_each_audit_overflow_rolls_back_state_beliefs_arrays_and_ledger(
        self,
    ) -> None:
        cases = (
            "assessed",
            "paid",
            "treasury",
            "detections",
        )
        for case in cases:
            with self.subTest(case=case):
                world = _world()
                state = world.states[0]
                state.state.audit_capacity_per_cycle = 1
                state.state.inspection_cost_cents = 0
                state.state.treasury_cents = INT64_MAX if case == "treasury" else 10
                world.firms[0].state.cash_cents = 1
                if case == "assessed":
                    world.firm_fine_assessed_cents[0] = INT64_MAX
                elif case == "paid":
                    world.firm_fine_paid_cents[0] = INT64_MAX
                elif case == "detections":
                    world._public_detections[0] = INT64_MAX
                before = _financial_snapshot(world)

                with self.assertRaises(OverflowError):
                    _run_controlled_audits(world, fines=((0, 1),))

                self.assertEqual(_financial_snapshot(world), before)

    def test_late_audit_overflow_rolls_back_an_earlier_planned_resolution(self) -> None:
        world = _world()
        state = world.states[0]
        state.state.audit_capacity_per_cycle = 2
        state.state.inspection_cost_cents = 0
        state.state.treasury_cents = 10
        world.firms[0].state.cash_cents = 1
        world.firms[1].state.cash_cents = 1
        world.firm_fine_assessed_cents[1] = INT64_MAX
        before = _financial_snapshot(world)

        with self.assertRaises(OverflowError):
            _run_controlled_audits(world, fines=((0, 1), (1, 1)))

        self.assertEqual(_financial_snapshot(world), before)

    def test_duplicate_audit_reference_fails_before_any_commit(self) -> None:
        world = _world()
        state = world.states[0]
        state.state.audit_capacity_per_cycle = 1
        state.state.inspection_cost_cents = 0
        state.state.treasury_cents = 10
        world.firms[0].state.cash_cents = 1
        world.ledger.transfer(
            tick=7,
            debit_account="existing:source",
            credit_account="existing:destination",
            amount_cents=1,
            kind="existing",
            reference="fine:7:0:0",
        )
        before = _financial_snapshot(world)

        with self.assertRaisesRegex(ValueError, "duplicate ledger reference"):
            _run_controlled_audits(world, fines=((0, 1),))

        self.assertEqual(_financial_snapshot(world), before)

    def test_audit_array_length_is_preflighted_before_resolution(self) -> None:
        world = _world()
        world.firm_fine_paid_cents = np.zeros(
            len(world.firms) + 1,
            dtype=np.int64,
        )
        before = _financial_snapshot(world)

        with self.assertRaisesRegex(TypeError, "length"):
            _run_controlled_audits(world, fines=((0, 1),))

        self.assertEqual(_financial_snapshot(world), before)


class SubsidyBoundaryTests(unittest.TestCase):
    def test_exact_int64_boundary_succeeds_and_reference_is_unique(self) -> None:
        world = _world()
        state = world.states[0]
        state.state.treasury_cents = INT64_MAX
        state.state.subsidy_budget_cents = INT64_MAX
        world.firms[0].state.cash_cents = 0
        world._pending_subsidies[:] = [_application(0, INT64_MAX, 0)]

        total = government_phase.review_subsidies(world, tick=1)

        self.assertEqual(total, INT64_MAX)
        self.assertEqual(state.state.treasury_cents, 0)
        self.assertEqual(state.state.subsidy_budget_cents, 0)
        self.assertEqual(world.firms[0].state.cash_cents, INT64_MAX)
        self.assertEqual(int(world.firm_subsidy_cents[0]), INT64_MAX)
        self.assertEqual(int(world.state_subsidy_outlay_cents[0]), INT64_MAX)
        self.assertEqual(world._pending_subsidies, [])
        references = tuple(entry.reference for entry in world.ledger.entries)
        self.assertEqual(references, ("subsidy:1:0:0",))
        self.assertEqual(len(references), len(set(references)))

    def test_each_subsidy_destination_overflow_rolls_back_everything(self) -> None:
        cases = ("firm_cash", "firm_subsidy", "state_outlay")
        for case in cases:
            with self.subTest(case=case):
                world = _world()
                state = world.states[0]
                state.state.treasury_cents = 1
                state.state.subsidy_budget_cents = 1
                if case == "firm_cash":
                    world.firms[0].state.cash_cents = INT64_MAX
                elif case == "firm_subsidy":
                    world.firm_subsidy_cents[0] = INT64_MAX
                else:
                    world.state_subsidy_outlay_cents[0] = INT64_MAX
                world._pending_subsidies[:] = [_application(0, 1, 0)]
                before = _financial_snapshot(world)

                with self.assertRaises(OverflowError):
                    government_phase.review_subsidies(world, tick=1)

                self.assertEqual(_financial_snapshot(world), before)

    def test_late_subsidy_overflow_rolls_back_an_earlier_planned_award(self) -> None:
        world = _world()
        state = world.states[0]
        state.state.treasury_cents = 2
        state.state.subsidy_budget_cents = 2
        world.firms[0].state.cash_cents = 0
        world.firms[1].state.cash_cents = INT64_MAX
        world._pending_subsidies[:] = [
            _application(0, 1, 0, quality=1.0),
            _application(1, 1, 0, quality=0.0),
        ]
        before = _financial_snapshot(world)

        with self.assertRaises(OverflowError):
            government_phase.review_subsidies(world, tick=1)

        self.assertEqual(_financial_snapshot(world), before)

    def test_total_outlay_overflow_across_states_is_fully_transactional(self) -> None:
        world = _world()
        first_state, second_state = world.states[:2]
        first_state.state.treasury_cents = INT64_MAX
        first_state.state.subsidy_budget_cents = INT64_MAX
        second_state.state.treasury_cents = 1
        second_state.state.subsidy_budget_cents = 1
        world.firms[0].state.cash_cents = 0
        world.firms[1].state.cash_cents = 0
        world._pending_subsidies[:] = [
            _application(0, INT64_MAX, 0),
            _application(1, 1, 1),
        ]
        before = _financial_snapshot(world)

        with self.assertRaises(OverflowError):
            government_phase.review_subsidies(world, tick=1)

        self.assertEqual(_financial_snapshot(world), before)

    def test_duplicate_subsidy_reference_fails_before_any_commit(self) -> None:
        world = _world()
        state = world.states[0]
        state.state.treasury_cents = 1
        state.state.subsidy_budget_cents = 1
        world.firms[0].state.cash_cents = 0
        world._pending_subsidies[:] = [_application(0, 1, 0)]
        world.ledger.transfer(
            tick=1,
            debit_account="existing:source",
            credit_account="existing:destination",
            amount_cents=1,
            kind="existing",
            reference="subsidy:1:0:0",
        )
        before = _financial_snapshot(world)

        with self.assertRaisesRegex(ValueError, "duplicate ledger reference"):
            government_phase.review_subsidies(world, tick=1)

        self.assertEqual(_financial_snapshot(world), before)

    def test_subsidy_array_length_is_preflighted_before_awards(self) -> None:
        world = _world()
        world.state_subsidy_outlay_cents = np.zeros(
            len(world.states) + 1,
            dtype=np.int64,
        )
        world._pending_subsidies[:] = [_application(0, 1, 0)]
        before = _financial_snapshot(world)

        with self.assertRaisesRegex(TypeError, "length"):
            government_phase.review_subsidies(world, tick=1)

        self.assertEqual(_financial_snapshot(world), before)


if __name__ == "__main__":
    unittest.main()
