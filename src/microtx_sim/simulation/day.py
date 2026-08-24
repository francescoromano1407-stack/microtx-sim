"""Authoritative ordering for one simulated tick (the daily lifecycle)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..companies.logic import FirmResolution
from ..consumers.logic import StepResult
from ..market.popularity import PublishedRanking
from ..metrics.outcomes import OutcomeSnapshot
from ..states.logic import AuditResolution
from ..types import EventKind
from .accounting import (
    accrue_interest,
    checked_accumulate,
    credit_firm_revenue,
    outcome_snapshot,
    renew_income,
)
from .company_phase import run_company_decision
from .government_phase import review_subsidies, run_audits
from .market_phase import publish_market_ranking

if TYPE_CHECKING:
    from ..core.world import World


@dataclass(frozen=True, slots=True)
class WorldStep:
    """All kernel outputs produced during one simulated tick."""

    tick: int
    player_result: StepResult
    firm_resolution: FirmResolution | None
    published_ranking: PublishedRanking | None
    audit_resolutions: tuple[AuditResolution, ...]
    subsidies_paid_cents: int
    outcome: OutcomeSnapshot


def schedule_initial_events(world: "World") -> None:
    """Install the deterministic starting schedule for a new world."""

    world.events.schedule(0, EventKind.FIRM_DECISION, priority=0)
    world.events.schedule(0, EventKind.PUBLISH_RANKING, priority=20)
    world.events.schedule(0, EventKind.AUDIT_DUE, priority=30)
    world.events.schedule(0, EventKind.SUBSIDY_REVIEW, priority=40)
    world.events.schedule(30, "income_renewal", priority=-10)


def advance_day(world: "World") -> WorldStep:
    """Advance exactly one configured tick while preserving causal ordering."""

    tick = world.tick
    due = world.events.pop_due(tick)
    firm_resolution: FirmResolution | None = None
    published: PublishedRanking | None = None
    audits: tuple[AuditResolution, ...] = ()
    subsidies_paid = 0

    # Pre-consumption phase: resources arrive and firms act on prior signals.
    for event in due:
        if event.kind == "income_renewal":
            renew_income(world, tick=tick)
            world.events.schedule(tick + 30, "income_renewal", priority=-10)
        elif event.kind == EventKind.FIRM_DECISION:
            firm_resolution = run_company_decision(world, tick=tick)
            world.events.schedule(
                tick + world.config.market.firm_decision_interval,
                EventKind.FIRM_DECISION,
                priority=0,
            )

    # Consumer phase: every player evaluates the complete game set in blocks.
    player_result = world.player_system.step(
        world.players,
        world.games,
        world.rng,
        world.ledger,
        tick=tick,
    )
    world._last_player_result = player_result
    checked_accumulate(
        world.player_total_spend_cents,
        player_result.player_spend_cents,
        label="cumulative player spend",
    )
    checked_accumulate(
        world.player_total_unsafe_spend_cents,
        player_result.player_unsafe_spend_cents,
        label="cumulative unsafe player spend",
    )
    checked_accumulate(
        world.player_total_unauthorised_cents,
        player_result.player_unauthorised_spend_cents,
        label="cumulative unauthorised player spend",
    )
    checked_accumulate(
        world._period_game_revenue_cents,
        player_result.game_revenue_cents,
        label="period game revenue",
    )
    world._latest_game_active_players[:] = world.games.active_players
    credit_firm_revenue(world, player_result)
    accrue_interest(world)

    # Post-consumption phase: public information and government actions may use
    # this tick's realised outcomes, but firms cannot react until a later tick.
    for event in due:
        if event.kind == EventKind.PUBLISH_RANKING:
            published = publish_market_ranking(
                world,
                tick=tick,
                player_result=player_result,
            )
            world.events.schedule(
                tick + world.config.market.ranking_interval,
                EventKind.PUBLISH_RANKING,
                priority=20,
            )
        elif event.kind == EventKind.AUDIT_DUE:
            audits = run_audits(world, tick=tick, player_result=player_result)
            world.events.schedule(
                tick + world._audit_interval,
                EventKind.AUDIT_DUE,
                priority=30,
            )
        elif event.kind == EventKind.SUBSIDY_REVIEW:
            subsidies_paid = review_subsidies(world, tick=tick)
            world.events.schedule(
                tick + world._subsidy_interval,
                EventKind.SUBSIDY_REVIEW,
                priority=40,
            )

    # Novelty decays smoothly between content releases.
    world.games.novelty[:] *= np.exp(-0.01 * world.config.run.tick_days)
    world.ledger.assert_balanced()
    outcome = outcome_snapshot(world, tick=tick)
    world.recorder.record(outcome)
    step = WorldStep(
        tick=tick,
        player_result=player_result,
        firm_resolution=firm_resolution,
        published_ranking=published,
        audit_resolutions=audits,
        subsidies_paid_cents=subsidies_paid,
        outcome=outcome,
    )
    world._step_history.append(step)
    world.tick += world.config.run.tick_days
    return step
