"""Government-facing observation, audit, enforcement, and subsidy phases.

Regulators receive constructed, noisy evidence. Latent compliance truth enters
only the kernel-side audit resolver and is never passed to a StateAgent policy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..agents.jurisdictions import StateAgent
from ..consumers.logic import StepResult
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

    all_resolutions: list[AuditResolution] = []
    for state in world.states:
        # Audit budget is a period appropriation, constrained by the treasury.
        state.state.audit_budget_cents = min(
            state.state.treasury_cents,
            state.state.audit_capacity_per_cycle
            * state.state.inspection_cost_cents,
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
        audit_cost = len(resolutions) * state.state.inspection_cost_cents
        if audit_cost:
            world.ledger.transfer(
                tick=tick,
                debit_account=f"state:{state.jurisdiction_id}:treasury",
                credit_account="sector:audit-services",
                amount_cents=audit_cost,
                kind="regulatory_audit",
                reference=f"audit:{tick}:{state.jurisdiction_id}",
            )
        for resolution in resolutions:
            firm = world.firms[resolution.intent.firm_id]
            firm_id = firm.firm_id
            assessed = resolution.fine_cents
            if world.firm_fine_assessed_cents[firm_id] > INT64_MAX - assessed:
                raise OverflowError("cumulative assessed fines would overflow int64")
            world.firm_fine_assessed_cents[firm_id] += assessed
            collected = min(firm.state.cash_cents, resolution.fine_cents)
            if collected:
                firm.state.cash_cents -= collected
                if world.firm_fine_paid_cents[firm_id] > INT64_MAX - collected:
                    raise OverflowError("cumulative paid fines would overflow int64")
                if state.state.treasury_cents > INT64_MAX - collected:
                    raise OverflowError("state treasury would overflow int64")
                world.firm_fine_paid_cents[firm_id] += collected
                state.state.treasury_cents += collected
                world.ledger.transfer(
                    tick=tick,
                    debit_account=f"firm:{firm.firm_id}:cash",
                    credit_account=f"state:{state.jurisdiction_id}:treasury",
                    amount_cents=collected,
                    kind="regulatory_fine",
                    reference=(
                        f"fine:{tick}:{state.jurisdiction_id}:{firm.firm_id}"
                    ),
                )
            if resolution.evidence.detected_breaches:
                world._public_detections[firm.firm_id] += 1
        all_resolutions.extend(resolutions)
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
    for state in world.states:
        applications = tuple(
            application
            for application in latest_by_firm.values()
            if state.jurisdiction_id in application.eligible_jurisdictions
        )
        for award in state.award_subsidies(applications):
            available = min(
                state.state.treasury_cents,
                state.state.subsidy_budget_cents,
            )
            paid = min(available, award.award_cents)
            if paid <= 0:
                continue
            firm = world.firms[award.firm_id]
            if firm.state.cash_cents > INT64_MAX - paid:
                raise OverflowError("firm cash would overflow int64")
            state.state.treasury_cents -= paid
            state.state.subsidy_budget_cents -= paid
            firm.state.cash_cents += paid
            if world.firm_subsidy_cents[award.firm_id] > INT64_MAX - paid:
                raise OverflowError("cumulative firm subsidy would overflow int64")
            if (
                world.state_subsidy_outlay_cents[state.jurisdiction_id]
                > INT64_MAX - paid
            ):
                raise OverflowError("cumulative state subsidy would overflow int64")
            world.firm_subsidy_cents[award.firm_id] += paid
            world.state_subsidy_outlay_cents[state.jurisdiction_id] += paid
            total += paid
            world.ledger.transfer(
                tick=tick,
                debit_account=f"state:{state.jurisdiction_id}:treasury",
                credit_account=f"firm:{firm.firm_id}:cash",
                amount_cents=paid,
                kind="conditional_subsidy",
                reference=(
                    f"subsidy:{tick}:{state.jurisdiction_id}:{firm.firm_id}"
                ),
            )
    world._pending_subsidies[:] = future
    return total
