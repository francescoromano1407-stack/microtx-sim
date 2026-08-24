"""Kernel coordination for true popularity and delayed public rankings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..companies.logic import PublicRankingSnapshot
from ..consumers.logic import StepResult
from ..market.popularity import PublishedRanking

if TYPE_CHECKING:
    from ..core.world import World


def publish_market_ranking(
    world: "World",
    *,
    tick: int,
    player_result: StepResult,
) -> PublishedRanking | None:
    """Update latent popularity and publish only the configured noisy signal."""

    world.popularity_system.observe_truth(
        tick=tick,
        players=world.players,
        games=world.games,
        period_revenue_cents=player_result.game_revenue_cents,
    )
    published = world.popularity_system.publish(
        tick=tick,
        games=world.games,
        rng=world.rng,
        promotion_pressure=world._promotion_pressure,
    )
    if published is not None:
        # Company policies receive the released board, never latent popularity.
        world.firm_system.record_public_ranking(
            PublicRankingSnapshot.from_game_table(
                as_of=published.published_tick,
                data_tick=published.data_tick,
                games=world.games,
            )
        )
    world._last_published_ranking = published
    return published
