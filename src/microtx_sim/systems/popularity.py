from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..agents.players import PlayerTable
from ..domain.games import GameTable
from ..rng import CounterRNG, stable_stream_id


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_RANKING_NOISE_STREAM = stable_stream_id("public-ranking-noise")


@dataclass(frozen=True, slots=True)
class TruthRankingSnapshot:
    tick: int
    active_players: IntArray
    score: FloatArray


@dataclass(frozen=True, slots=True)
class PublishedRanking:
    published_tick: int
    data_tick: int
    score: FloatArray
    rank: IntArray
    expected_noise_sd: float


class PopularitySystem:
    """Separates exact latent popularity from delayed public rankings."""

    __slots__ = ("game_count", "delay_days", "noise_sd", "_history", "_previous_active")

    def __init__(self, *, game_count: int, delay_days: int, noise_sd: float) -> None:
        if game_count <= 0 or delay_days < 0 or not 0.0 <= noise_sd <= 1.0:
            raise ValueError("invalid popularity-system configuration")
        self.game_count = game_count
        self.delay_days = delay_days
        self.noise_sd = noise_sd
        self._history: list[TruthRankingSnapshot] = []
        self._previous_active = np.zeros(game_count, dtype=np.int64)

    @staticmethod
    def _unit_scale(values: FloatArray) -> FloatArray:
        if len(values) == 0:
            return values.copy()
        minimum = float(values.min())
        span = float(values.max()) - minimum
        if span <= 0.0:
            return np.zeros_like(values)
        return (values - minimum) / span

    def observe_truth(
        self,
        *,
        tick: int,
        players: PlayerTable,
        games: GameTable,
        period_revenue_cents: IntArray,
    ) -> TruthRankingSnapshot:
        if len(games.game_id) != self.game_count:
            raise ValueError("game count changed")
        revenue = np.asarray(period_revenue_cents, dtype=np.int64)
        if revenue.shape != (self.game_count,) or np.any(revenue < 0):
            raise ValueError("period revenue must be non-negative per game")
        assigned = players.current_game.astype(np.int64, copy=False)
        valid = (assigned >= 0) & (assigned < self.game_count)
        active = np.bincount(assigned[valid], minlength=self.game_count).astype(np.int64)
        momentum = np.divide(
            active - self._previous_active,
            np.maximum(1, self._previous_active),
            dtype=np.float64,
        )
        revenue_per_active = np.divide(
            revenue.astype(np.float64),
            np.maximum(1, active),
        )
        score = (
            0.34 * self._unit_scale(np.log1p(active.astype(np.float64)))
            + 0.18 * games.quality
            + 0.15 * games.competitive_integrity
            + 0.13 * games.novelty
            + 0.12 * self._unit_scale(np.log1p(revenue_per_active))
            + 0.08 * self._unit_scale(np.clip(momentum, -1.0, 1.0))
        )
        snapshot = TruthRankingSnapshot(tick=tick, active_players=active, score=score)
        self._history.append(snapshot)
        self._previous_active = active.copy()
        games.active_players[:] = active
        games.true_popularity[:] = score
        return snapshot

    def publish(
        self,
        *,
        tick: int,
        games: GameTable,
        rng: CounterRNG,
        promotion_pressure: FloatArray | None = None,
    ) -> PublishedRanking:
        if not self._history:
            raise RuntimeError("truth must be observed before a ranking is published")
        eligible_tick = tick - self.delay_days
        source = self._history[0]
        for snapshot in self._history:
            if snapshot.tick <= eligible_tick:
                source = snapshot
            else:
                break
        promotion = (
            np.zeros(self.game_count, dtype=np.float64)
            if promotion_pressure is None
            else np.asarray(promotion_pressure, dtype=np.float64)
        )
        if promotion.shape != (self.game_count,) or np.any(promotion < 0.0):
            raise ValueError("promotion pressure must be non-negative per game")
        ids = games.game_id.astype(np.int64, copy=False)
        noise = rng.normal(ids, tick, _RANKING_NOISE_STREAM, 0, scale=self.noise_sd)
        public_score = source.score + noise + 0.04 * np.log1p(promotion)
        # Stable tie-breaking by game ID. Lexsort's final key is primary.
        order = np.lexsort((ids, -public_score))
        rank = np.empty(self.game_count, dtype=np.int64)
        rank[order] = np.arange(1, self.game_count + 1, dtype=np.int64)
        games.public_score[:] = public_score
        games.public_rank[:] = rank
        result = PublishedRanking(
            published_tick=tick,
            data_tick=source.tick,
            score=public_score.copy(),
            rank=rank.copy(),
            expected_noise_sd=self.noise_sd,
        )
        return result

    def trim_history(self, *, before_tick: int) -> None:
        """Drop obsolete truth snapshots while retaining the newest predecessor."""

        if len(self._history) <= 1:
            return
        keep_from = 0
        for index, snapshot in enumerate(self._history[:-1]):
            if self._history[index + 1].tick < before_tick:
                keep_from = index + 1
            else:
                break
        if keep_from:
            del self._history[:keep_from]

