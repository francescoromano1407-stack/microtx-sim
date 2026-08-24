"""Kernel coordination for periodic mobile-game company decisions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..companies.logic import FirmResolution, capture_period_telemetry

if TYPE_CHECKING:
    from ..core.world import World


def run_company_decision(world: "World", *, tick: int) -> FirmResolution:
    """Collect bounded company observations, resolve intents, and store effects.

    Scheduling is deliberately owned by :mod:`microtx_sim.simulation.day`; this
    phase knows how a decision is resolved, but not when the next one occurs.
    """

    telemetry = capture_period_telemetry(
        tick=tick,
        games=world.games,
        firms=world.firms,
        rng=world.rng,
        period_revenue_cents=world._period_game_revenue_cents,
        active_players=world._latest_game_active_players,
    )
    intents = world.firm_system.collect_intents(
        tick=tick,
        games=world.games,
        period_telemetry=telemetry,
    )
    result = world.firm_system.resolve(
        tick=tick,
        games=world.games,
        intents=intents,
        ledger=world.ledger,
        period_telemetry=telemetry,
    )
    world._period_game_revenue_cents.fill(0)
    world._enforce_mechanism_caps()
    world._promotion_pressure *= 0.65
    world._promotion_pressure += result.promotion_pressure
    world._pending_subsidies.extend(
        replace(
            application,
            eligible_jurisdictions=(
                int(world._firm_home_jurisdiction[application.firm_id]),
            ),
        )
        for application in result.subsidy_applications
    )
    world._last_firm_resolution = result
    return result
