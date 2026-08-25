"""Government-facing observation, audit, enforcement, and subsidy phases.

Regulators receive constructed, noisy evidence. Latent compliance truth enters
only the kernel-side audit resolver and is never passed to a StateAgent policy.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..agents.jurisdictions import StateAgent
from ..consumers.logic import StepResult
from ..core.ledger import LedgerEntry
from ..rng import stable_stream_id
from ..states.logic import (
    AuditResolution,
    FirmComplianceTruth,
    ObservableFirmMetrics,
)
from ..types import HarmDimension, MonetisationMechanism
from .accounting import INT64_MAX

if TYPE_CHECKING:
    from ..core.world import World


IntArray = npt.NDArray[np.int64]
_COMPLAINT_STREAM = stable_stream_id("player-complaint-report")


def firm_for_player(world: "World") -> IntArray:
    """Return the owning firm for each player's current game, or ``-1``."""

    result = np.full(len(world.players), -1, dtype=np.int64)
    for row, game_id in enumerate(world.games.game_id):
        mask = world.players.current_game == int(game_id)
        result[mask] = int(world.games.company_id[row])
    return result


def build_observable_firm_metrics(
    world: "World",
    *,
    tick: int,
    player_result: StepResult,
    jurisdiction_id: int,
) -> tuple[ObservableFirmMetrics, ...]:
    """Construct regulator-visible complaint and anomaly signals."""

    owner = firm_for_player(world)
    jurisdiction = world.players.jurisdiction == jurisdiction_id
    valid = jurisdiction & (owner >= 0)
    regret = world.players.harm_state[:, HarmDimension.REGRET].astype(np.float64)
    unauthorised = player_result.player_unauthorised_spend_cents > 0
    report_probability = np.clip(
        0.02 + 0.32 * regret + 0.70 * unauthorised,
        0.0,
        1.0,
    )
    reports = valid & world.rng.bernoulli(
        world.players.player_id,
        tick,
        _COMPLAINT_STREAM,
        jurisdiction_id,
        probability=report_probability,
    )
    burden = np.divide(
        player_result.player_spend_cents.astype(np.float64),
        np.maximum(1, world.players.monthly_disposable_income_cents),
    )
    # Spending burden remains latent until a player or household reports it.
    anomalous = reports & (burden > 0.10)
    minor_report = reports & world.players.is_minor
    metrics: list[ObservableFirmMetrics] = []
    for firm_id in range(len(world.firms)):
        exposed = valid & (owner == firm_id)
        denominator = max(1, int(np.count_nonzero(exposed)))
        minor_denominator = max(
            1,
            int(np.count_nonzero(exposed & world.players.is_minor)),
        )
        metrics.append(
            ObservableFirmMetrics(
                firm_id=firm_id,
                complaint_rate=float(
                    np.count_nonzero(reports & exposed) / denominator
                ),
                reported_minor_harm_rate=float(
                    np.count_nonzero(minor_report & exposed) / minor_denominator
                ),
                public_spend_anomaly=float(
                    np.count_nonzero(anomalous & exposed) / denominator
                ),
                past_public_detection=float(
                    min(1.0, world._public_detections[firm_id] / 3.0)
                ),
                signal_precision=0.65,
                signal_age_days=0,
            )
        )
    return tuple(metrics)


def build_compliance_truth(
    world: "World",
    *,
    state: StateAgent,
    player_result: StepResult,
) -> dict[int, FirmComplianceTruth]:
    """Build audit ground truth inside the kernel after policy selection."""

    if world._last_firm_resolution is None:
        kernel = {
            firm.firm_id: (firm.compliance_culture, 0.0) for firm in world.firms
        }
    else:
        kernel = {
            item.firm_id: (item.compliance_effectiveness, item.evasion_level)
            for item in world._last_firm_resolution.firm_kernel_state
        }
    owner = firm_for_player(world)
    truth: dict[int, FirmComplianceTruth] = {}
    for firm in world.firms:
        rows = np.flatnonzero(world.games.company_id == firm.firm_id)
        mechanisms = world.games.monetisation[rows]
        compliance, evasion = kernel[firm.firm_id]
        breaches: list[str] = []
        if (
            state.rules.paid_random_rewards_restricted
            and np.any(
                mechanisms[:, MonetisationMechanism.RANDOM_REWARD] > 0.02
            )
        ):
            breaches.append("paid_random_rewards")
        if (
            state.rules.odds_disclosure_required
            and np.any(
                mechanisms[:, MonetisationMechanism.RANDOM_REWARD] > 0.02
            )
            and compliance < 0.70
        ):
            breaches.append("odds_disclosure")
        if (
            state.rules.real_money_price_required
            and np.any(
                mechanisms[:, MonetisationMechanism.PRICE_OBFUSCATION] > compliance
            )
        ):
            breaches.append("price_transparency")
        if np.any(
            mechanisms[:, MonetisationMechanism.POWER_SALE]
            > state.rules.maximum_power_sale_intensity
        ):
            breaches.append("power_sale_limit")
        firm_minor_unauthorised = (
            (owner == firm.firm_id)
            & (world.players.jurisdiction == state.jurisdiction_id)
            & (player_result.player_unauthorised_spend_cents > 0)
        )
        if (
            state.rules.parental_authorisation_required
            and np.any(firm_minor_unauthorised)
            and compliance < 0.85
        ):
            breaches.append("parental_authorisation")
        if (
            state.rules.direct_exhortation_to_minors_banned
            and np.any(
                0.5
                * (
                    mechanisms[:, MonetisationMechanism.SOCIAL_PRESSURE]
                    + mechanisms[:, MonetisationMechanism.ARTIFICIAL_SCARCITY]
                )
                > 0.60
            )
            and compliance < 0.65
        ):
            breaches.append("minor_exhortation")
        truth[firm.firm_id] = FirmComplianceTruth(
            firm_id=firm.firm_id,
            actual_breaches=tuple(breaches),
            auditable_controls=(
                "price_display",
                "probability_disclosure",
                "parental_authorisation",
                "transaction_log",
            ),
            evasion_intensity=evasion,
            maximum_fine_cents=world.config.regulation.maximum_fine_cents,
        )
    return truth


def run_audits(
    world: "World",
    *,
    tick: int,
    player_result: StepResult,
) -> tuple[AuditResolution, ...]:
    """Run policy selection on observations, then resolve imperfect audits."""

    planned_states = tuple(deepcopy(state) for state in world.states)
    planned_firm_cash = [
        _nonnegative_int64(firm.state.cash_cents, label="firm cash")
        for firm in world.firms
    ]
    planned_assessed = _nonnegative_int64_array(
        world.firm_fine_assessed_cents,
        label="cumulative assessed fines",
        expected_length=len(world.firms),
    )
    planned_paid = _nonnegative_int64_array(
        world.firm_fine_paid_cents,
        label="cumulative paid fines",
        expected_length=len(world.firms),
    )
    planned_detections = _nonnegative_int64_array(
        world._public_detections,
        label="public detection count",
        expected_length=len(world.firms),
    )
    planned_entries: list[LedgerEntry] = []
    all_resolutions: list[AuditResolution] = []
    for state in planned_states:
        # Audit budget is a period appropriation, constrained by the treasury.
        treasury = _nonnegative_int64(
            state.state.treasury_cents,
            label="state treasury",
        )
        capacity = _nonnegative_int64(
            state.state.audit_capacity_per_cycle,
            label="audit capacity",
        )
        inspection_cost = _nonnegative_int64(
            state.state.inspection_cost_cents,
            label="inspection cost",
        )
        state.state.audit_budget_cents = min(
            treasury,
            capacity * inspection_cost,
        )
        metrics = build_observable_firm_metrics(
            world,
            tick=tick,
            player_result=player_result,
            jurisdiction_id=state.jurisdiction_id,
        )
        public_harm = float(
            np.clip(
                np.mean(
                    [
                        0.45 * item.complaint_rate
                        + 0.35 * item.reported_minor_harm_rate
                        + 0.20 * item.public_spend_anomaly
                        for item in metrics
                    ]
                ),
                0.0,
                1.0,
            )
        )
        observation = world.regulation_system.build_observation(
            tick=tick,
            firms=metrics,
            public_harm_index=public_harm,
            treasury_pressure=float(
                1.0 - state.state.treasury_cents / max(1, 36_000_000)
            ),
            sector_employment_estimate=50.0 * len(world.firms),
        )
        intents = world.regulation_system.select(
            tick=tick,
            state=state,
            observation=observation,
            rng=world.rng,
        )
        resolutions = world.regulation_system.resolve(
            tick=tick,
            state=state,
            intents=intents,
            truth_by_firm=build_compliance_truth(
                world,
                state=state,
                player_result=player_result,
            ),
            rng=world.rng,
        )
        audit_cost = _nonnegative_int64(
            len(resolutions) * inspection_cost,
            label="audit cost",
        )
        _nonnegative_int64(
            state.state.audit_budget_cents,
            label="remaining audit budget",
        )
        _nonnegative_int64(
            state.state.treasury_cents,
            label="state treasury",
        )
        if audit_cost:
            planned_entries.append(
                LedgerEntry(
                    tick=tick,
                    debit_account=f"state:{state.jurisdiction_id}:treasury",
                    credit_account="sector:audit-services",
                    amount_cents=audit_cost,
                    kind="regulatory_audit",
                    reference=f"audit:{tick}:{state.jurisdiction_id}",
                )
            )
        for resolution in resolutions:
            firm_id = int(resolution.intent.firm_id)
            if firm_id < 0 or firm_id >= len(world.firms):
                raise ValueError("audit resolution references an unknown firm")
            assessed = _nonnegative_int64(
                resolution.fine_cents,
                label="assessed fine",
            )
            planned_assessed[firm_id] = _checked_add_int64(
                int(planned_assessed[firm_id]),
                assessed,
                label="cumulative assessed fines",
            )
            collected = min(planned_firm_cash[firm_id], assessed)
            if collected:
                planned_paid[firm_id] = _checked_add_int64(
                    int(planned_paid[firm_id]),
                    collected,
                    label="cumulative paid fines",
                )
                state.state.treasury_cents = _checked_add_int64(
                    int(state.state.treasury_cents),
                    collected,
                    label="state treasury",
                )
                planned_firm_cash[firm_id] -= collected
                planned_entries.append(
                    LedgerEntry(
                        tick=tick,
                        debit_account=f"firm:{firm_id}:cash",
                        credit_account=f"state:{state.jurisdiction_id}:treasury",
                        amount_cents=collected,
                        kind="regulatory_fine",
                        reference=(
                            f"fine:{tick}:{state.jurisdiction_id}:{firm_id}"
                        ),
                    )
                )
            if resolution.evidence.detected_breaches:
                planned_detections[firm_id] = _checked_add_int64(
                    int(planned_detections[firm_id]),
                    1,
                    label="public detection count",
                )
        all_resolutions.extend(resolutions)

    _validate_new_ledger_references(world, planned_entries)

    # Resolution ran only on detached StateAgent copies. All arithmetic and
    # ledger validation has now succeeded, so these assignments cannot expose a
    # partially applied audit when a late boundary check fails.
    for target, source in zip(world.states, planned_states):
        _commit_regulator_private_state(target, source)
    for firm, cash in zip(world.firms, planned_firm_cash):
        firm.state.cash_cents = cash
    world.firm_fine_assessed_cents[:] = planned_assessed
    world.firm_fine_paid_cents[:] = planned_paid
    world._public_detections[:] = planned_detections
    world.ledger.extend(planned_entries)
    return tuple(all_resolutions)


def review_subsidies(world: "World", *, tick: int) -> int:
    """Review mature applications in eligible home jurisdictions once."""

    total = 0
    mature = [
        replace(
            application,
            evidence_age_days=(
                application.evidence_age_days
                + tick
                - application.submitted_tick
            ),
        )
        for application in world._pending_subsidies
        if application.submitted_tick < tick
    ]
    future = [
        application
        for application in world._pending_subsidies
        if application.submitted_tick >= tick
    ]
    # Later submissions replace older dossiers from the same company.
    latest_by_firm = {application.firm_id: application for application in mature}
    planned_treasury = [
        _nonnegative_int64(state.state.treasury_cents, label="state treasury")
        for state in world.states
    ]
    planned_budgets = [
        _nonnegative_int64(
            state.state.subsidy_budget_cents,
            label="state subsidy budget",
        )
        for state in world.states
    ]
    planned_firm_cash = [
        _nonnegative_int64(firm.state.cash_cents, label="firm cash")
        for firm in world.firms
    ]
    planned_firm_subsidies = _nonnegative_int64_array(
        world.firm_subsidy_cents,
        label="cumulative firm subsidy",
        expected_length=len(world.firms),
    )
    planned_state_outlays = _nonnegative_int64_array(
        world.state_subsidy_outlay_cents,
        label="cumulative state subsidy",
        expected_length=len(world.states),
    )
    planned_entries: list[LedgerEntry] = []
    for state in world.states:
        state_id = int(state.jurisdiction_id)
        applications = tuple(
            application
            for application in latest_by_firm.values()
            if state_id in application.eligible_jurisdictions
        )
        for award in state.award_subsidies(applications):
            firm_id = int(award.firm_id)
            if firm_id < 0 or firm_id >= len(world.firms):
                raise ValueError("subsidy award references an unknown firm")
            award_cents = _nonnegative_int64(
                award.award_cents,
                label="subsidy award",
            )
            available = min(
                planned_treasury[state_id],
                planned_budgets[state_id],
            )
            paid = min(available, award_cents)
            if paid <= 0:
                continue
            planned_firm_cash[firm_id] = _checked_add_int64(
                planned_firm_cash[firm_id],
                paid,
                label="firm cash",
            )
            planned_firm_subsidies[firm_id] = _checked_add_int64(
                int(planned_firm_subsidies[firm_id]),
                paid,
                label="cumulative firm subsidy",
            )
            planned_state_outlays[state_id] = _checked_add_int64(
                int(planned_state_outlays[state_id]),
                paid,
                label="cumulative state subsidy",
            )
            total = _checked_add_int64(
                total,
                paid,
                label="total subsidy outlay",
            )
            planned_treasury[state_id] -= paid
            planned_budgets[state_id] -= paid
            planned_entries.append(
                LedgerEntry(
                    tick=tick,
                    debit_account=f"state:{state_id}:treasury",
                    credit_account=f"firm:{firm_id}:cash",
                    amount_cents=paid,
                    kind="conditional_subsidy",
                    reference=f"subsidy:{tick}:{state_id}:{firm_id}",
                )
            )

    _validate_new_ledger_references(world, planned_entries)

    for state, treasury, budget in zip(
        world.states,
        planned_treasury,
        planned_budgets,
    ):
        state.state.treasury_cents = treasury
        state.state.subsidy_budget_cents = budget
    for firm, cash in zip(world.firms, planned_firm_cash):
        firm.state.cash_cents = cash
    world.firm_subsidy_cents[:] = planned_firm_subsidies
    world.state_subsidy_outlay_cents[:] = planned_state_outlays
    world.ledger.extend(planned_entries)
    world._pending_subsidies[:] = future
    return total


def _nonnegative_int64(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{label} cannot be negative")
    if integer > INT64_MAX:
        raise OverflowError(f"{label} is outside int64")
    return integer


def _nonnegative_int64_array(
    values: npt.NDArray[np.int64],
    *,
    label: str,
    expected_length: int,
) -> npt.NDArray[np.int64]:
    array = np.asarray(values)
    if array.dtype != np.dtype(np.int64) or array.shape != (expected_length,):
        raise TypeError(
            f"{label} must be an int64 array of length {expected_length}"
        )
    if np.any(array < 0):
        raise ValueError(f"{label} cannot be negative")
    return array.copy()


def _checked_add_int64(current: object, increment: object, *, label: str) -> int:
    left = _nonnegative_int64(current, label=label)
    right = _nonnegative_int64(increment, label=label)
    if left > INT64_MAX - right:
        raise OverflowError(f"{label} would overflow int64")
    return left + right


def _validate_new_ledger_references(
    world: "World",
    entries: list[LedgerEntry],
) -> None:
    existing = {entry.reference for entry in world.ledger.entries}
    planned: set[str] = set()
    for entry in entries:
        if entry.reference in existing or entry.reference in planned:
            raise ValueError(f"duplicate ledger reference: {entry.reference}")
        planned.add(entry.reference)


def _commit_regulator_private_state(target: StateAgent, source: StateAgent) -> None:
    target.state.treasury_cents = source.state.treasury_cents
    target.state.audit_budget_cents = source.state.audit_budget_cents
    target.state.subsidy_budget_cents = source.state.subsidy_budget_cents
    target.state.audit_capacity_per_cycle = source.state.audit_capacity_per_cycle
    target.state.inspection_cost_cents = source.state.inspection_cost_cents
    target.state.compliance_alpha.clear()
    target.state.compliance_alpha.update(source.state.compliance_alpha)
    target.state.compliance_beta.clear()
    target.state.compliance_beta.update(source.state.compliance_beta)
