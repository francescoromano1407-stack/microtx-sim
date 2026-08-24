from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp

import numpy as np
import numpy.typing as npt

from ..types import MonetisationMechanism


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class OwnGameSnapshot:
    """Information a firm legitimately knows about one of its own games."""

    game_id: int
    stat_frontier: FloatArray
    demand_weights: FloatArray
    active_players_estimate: int
    price_cents: int
    content_cost_cents: int
    estimated_conversion: float
    estimated_audit_probability: float
    estimated_fine_cents: int
    reputation_sensitivity: float
    analytics_quality: float


@dataclass(frozen=True, slots=True)
class ContentCandidate:
    game_id: int
    stats: FloatArray
    boosted_dimensions: tuple[int, ...]
    weakened_dimension: int
    boost_rate: float
    perceived_conversion: float
    perceived_npv_cents: int

    def validate_against(self, frontier: FloatArray) -> None:
        if self.stats.shape != frontier.shape:
            raise ValueError("candidate and frontier dimensions differ")
        if not np.any(self.stats > frontier):
            raise ValueError("content must improve at least one dimension")
        if not np.any(self.stats < frontier):
            raise ValueError("content must contain at least one trade-off")
        if np.all(self.stats >= frontier):
            raise ValueError("content cannot dominate the entire frontier")


class ContentPlanner:
    """Exact bounded search over non-dominating competitive content candidates.

    The planner optimises the firm's perceived demand model. It never sees market
    truth, player vulnerabilities, or a regulator's hidden state.
    """

    __slots__ = ("boost_grid",)

    def __init__(self, boost_grid: tuple[float, ...] = (0.02, 0.05, 0.08, 0.12)):
        if not boost_grid or any(rate <= 0.0 or rate >= 0.5 for rate in boost_grid):
            raise ValueError("boost rates must be in (0, 0.5)")
        self.boost_grid = tuple(sorted(set(boost_grid)))

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + exp(-value))
        exp_value = exp(value)
        return exp_value / (1.0 + exp_value)

    def enumerate_candidates(self, snapshot: OwnGameSnapshot) -> tuple[ContentCandidate, ...]:
        frontier = np.asarray(snapshot.stat_frontier, dtype=np.float64)
        demand = np.asarray(snapshot.demand_weights, dtype=np.float64)
        if frontier.ndim != 1 or len(frontier) < 2 or frontier.shape != demand.shape:
            raise ValueError("frontier and demand must be equal one-dimensional vectors")
        if np.any(frontier <= 0) or np.any(demand < 0) or demand.sum() <= 0:
            raise ValueError("frontier must be positive and demand weights non-negative")
        demand = demand / demand.sum()
        dimension_ids = tuple(range(len(frontier)))
        candidates: list[ContentCandidate] = []

        # Every proper, non-empty subset is evaluated. This is an exact search over
        # the firm's deliberately finite R&D choice set, not sampling alternatives.
        for subset_size in range(1, len(frontier)):
            for boosted in combinations(dimension_ids, subset_size):
                outside = tuple(index for index in dimension_ids if index not in boosted)
                weakened = min(outside, key=lambda index: (demand[index], index))
                for boost_rate in self.boost_grid:
                    stats = frontier.copy()
                    stats[list(boosted)] *= 1.0 + boost_rate
                    stats[weakened] *= 1.0 - max(0.01, 0.55 * boost_rate)

                    relative_change = np.log(stats / frontier)
                    appeal = float(np.dot(demand, relative_change))
                    logit = (
                        -2.4
                        + 18.0 * appeal * (0.5 + snapshot.analytics_quality)
                        + 0.8 * snapshot.estimated_conversion
                    )
                    conversion = self._sigmoid(logit)
                    gross_cents = int(
                        round(
                            snapshot.active_players_estimate
                            * conversion
                            * snapshot.price_cents
                        )
                    )
                    balance_risk = boost_rate**2 * (subset_size / len(frontier))
                    expected_penalty = int(
                        round(
                            snapshot.estimated_audit_probability
                            * snapshot.estimated_fine_cents
                            * balance_risk
                        )
                    )
                    reputation_cost = int(
                        round(
                            gross_cents
                            * snapshot.reputation_sensitivity
                            * balance_risk
                        )
                    )
                    production_cost = int(
                        round(
                            snapshot.content_cost_cents
                            * (0.75 + 0.5 * subset_size / len(frontier))
                        )
                    )
                    candidate = ContentCandidate(
                        game_id=snapshot.game_id,
                        stats=stats,
                        boosted_dimensions=boosted,
                        weakened_dimension=weakened,
                        boost_rate=boost_rate,
                        perceived_conversion=conversion,
                        perceived_npv_cents=(
                            gross_cents - production_cost - expected_penalty - reputation_cost
                        ),
                    )
                    candidate.validate_against(frontier)
                    candidates.append(candidate)
        return tuple(candidates)

    def choose(self, snapshot: OwnGameSnapshot) -> ContentCandidate:
        candidates = self.enumerate_candidates(snapshot)
        return max(
            candidates,
            key=lambda item: (
                item.perceived_npv_cents,
                -item.boost_rate,
                tuple(-index for index in item.boosted_dimensions),
            ),
        )


@dataclass(slots=True)
class GameTable:
    game_id: IntArray
    company_id: IntArray
    quality: FloatArray
    competitive_integrity: FloatArray
    novelty: FloatArray
    monetisation: FloatArray
    stat_frontier: FloatArray
    price_cents: IntArray
    active_players: IntArray
    revenue_cents: IntArray
    true_popularity: FloatArray
    public_score: FloatArray
    public_rank: IntArray

    @classmethod
    def create(
        cls,
        *,
        game_count: int,
        company_count: int,
        stat_dimensions: int,
    ) -> "GameTable":
        game_id = np.arange(game_count, dtype=np.int64)
        company_id = (game_id % company_count).astype(np.int64)
        quality = np.linspace(0.46, 0.72, game_count, dtype=np.float64)
        integrity = np.linspace(0.78, 0.58, game_count, dtype=np.float64)
        novelty = np.full(game_count, 0.55, dtype=np.float64)
        mechanisms = len(MonetisationMechanism)
        monetisation = np.empty((game_count, mechanisms), dtype=np.float64)
        for game in range(game_count):
            base = 0.18 + 0.06 * (game % 4)
            monetisation[game] = np.clip(
                base + 0.025 * np.arange(mechanisms, dtype=np.float64), 0.0, 1.0
            )
        stat_frontier = np.ones((game_count, stat_dimensions), dtype=np.float64)
        stat_frontier += 0.03 * (
            np.arange(game_count, dtype=np.float64)[:, None]
            + np.arange(stat_dimensions, dtype=np.float64)[None, :]
        )
        price_cents = (199 + 100 * (game_id % 5)).astype(np.int64)
        public_rank = np.arange(1, game_count + 1, dtype=np.int64)
        return cls(
            game_id=game_id,
            company_id=company_id,
            quality=quality,
            competitive_integrity=integrity,
            novelty=novelty,
            monetisation=monetisation,
            stat_frontier=stat_frontier,
            price_cents=price_cents,
            active_players=np.zeros(game_count, dtype=np.int64),
            revenue_cents=np.zeros(game_count, dtype=np.int64),
            true_popularity=np.zeros(game_count, dtype=np.float64),
            public_score=np.zeros(game_count, dtype=np.float64),
            public_rank=public_rank,
        )

    def validate(self) -> None:
        game_count = len(self.game_id)
        one_dimensional = (
            self.company_id,
            self.quality,
            self.competitive_integrity,
            self.novelty,
            self.price_cents,
            self.active_players,
            self.revenue_cents,
            self.true_popularity,
            self.public_score,
            self.public_rank,
        )
        if any(len(array) != game_count for array in one_dimensional):
            raise ValueError("game columns have inconsistent lengths")
        if self.monetisation.shape[0] != game_count or self.stat_frontier.shape[0] != game_count:
            raise ValueError("game matrices have inconsistent row counts")
        if np.any((self.monetisation < 0.0) | (self.monetisation > 1.0)):
            raise ValueError("monetisation intensities must be in [0, 1]")
        if np.any(self.price_cents <= 0) or np.any(self.active_players < 0):
            raise ValueError("prices must be positive and populations non-negative")
        if set(self.public_rank.tolist()) != set(range(1, game_count + 1)):
            raise ValueError("public ranks must be a permutation")

    def apply_content(self, candidate: ContentCandidate) -> None:
        game = candidate.game_id
        old_frontier = self.stat_frontier[game].copy()
        candidate.validate_against(old_frontier)
        self.stat_frontier[game] = np.maximum(old_frontier, candidate.stats)
        self.novelty[game] = min(1.0, self.novelty[game] + 0.18)
        self.competitive_integrity[game] = max(
            0.0,
            self.competitive_integrity[game]
            - candidate.boost_rate * len(candidate.boosted_dimensions) / len(old_frontier),
        )

