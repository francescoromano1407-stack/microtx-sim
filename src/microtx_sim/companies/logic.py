"""Firm population, information boundaries, and simultaneous strategy resolution.

This module deliberately keeps two layers apart:

* :class:`FirmAgent` receives only a fallible :class:`CompanyObservation` built
  from its own telemetry and an old public ranking;
* :class:`FirmStrategySystem` resolves the resulting intents against simulation
  state and owns the latent compliance/evasion kernel that firms cannot inspect.

The separation is useful for causal experiments: changing a latent outcome does
not silently give a company clairvoyant information, and all firms decide from
the same pre-resolution state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, isfinite, log1p
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from ..agents.companies import CompanyObservation, FirmAgent, FirmIntent, FirmPrivateState
from ..agents.jurisdictions import SubsidyApplicationView
from ..core.ledger import Ledger
from ..domain.games import ContentCandidate, ContentPlanner, GameTable, OwnGameSnapshot
from ..rng import CounterRNG
from ..types import FirmAction, MonetisationMechanism


FloatArray = npt.NDArray[np.float64]


class FirmPolicy(Protocol):
    """A pluggable policy whose entire information surface is one observation."""

    def __call__(self, observation: CompanyObservation) -> FirmIntent: ...


@dataclass(frozen=True, slots=True)
class PublicRankingSnapshot:
    """A ranking with separate publication time and underlying data time.

    ``as_of`` is the publication tick used for availability. ``data_tick`` is
    the latent-observation tick represented by the delayed public signal.
    """

    as_of: int
    game_ids: tuple[int, ...]
    public_rank: tuple[int, ...]
    data_tick: int | None = None

    def __post_init__(self) -> None:
        if self.as_of < 0:
            raise ValueError("ranking as_of cannot be negative")
        if self.data_tick is None:
            object.__setattr__(self, "data_tick", self.as_of)
        if self.data_tick is None or not 0 <= self.data_tick <= self.as_of:
            raise ValueError("ranking data_tick must be between zero and publication")
        if len(self.game_ids) != len(self.public_rank):
            raise ValueError("ranking columns are inconsistent")
        if len(set(self.game_ids)) != len(self.game_ids):
            raise ValueError("ranking game ids must be unique")
        expected = set(range(1, len(self.game_ids) + 1))
        if set(self.public_rank) != expected:
            raise ValueError("public ranks must be a permutation")

    @classmethod
    def from_game_table(
        cls,
        *,
        as_of: int,
        games: GameTable,
        data_tick: int | None = None,
    ) -> "PublicRankingSnapshot":
        """Capture only the explicitly public columns of ``games``."""

        return cls(
            as_of=int(as_of),
            game_ids=tuple(int(value) for value in games.game_id),
            public_rank=tuple(int(value) for value in games.public_rank),
            data_tick=data_tick,
        )

    def as_mapping(self) -> dict[int, int]:
        return dict(zip(self.game_ids, self.public_rank, strict=True))


@dataclass(frozen=True, slots=True)
class FirmTelemetry:
    """Period aggregates measured by one firm for its own portfolio only."""

    as_of: int
    firm_id: int
    game_ids: tuple[int, ...]
    active_player_estimates: tuple[int, ...]
    revenue_estimates_cents: tuple[int, ...]
    novelty_estimates: tuple[float, ...]
    mechanism_demand_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.as_of < 0 or self.firm_id < 0:
            raise ValueError("telemetry identifiers and time must be non-negative")
        columns = (
            self.active_player_estimates,
            self.revenue_estimates_cents,
            self.novelty_estimates,
        )
        if any(len(column) != len(self.game_ids) for column in columns):
            raise ValueError("firm telemetry game columns are inconsistent")
        if len(set(self.game_ids)) != len(self.game_ids):
            raise ValueError("telemetry game ids must be unique")
        if any(value < 0 for value in self.active_player_estimates):
            raise ValueError("active-player telemetry cannot be negative")
        if any(value < 0 for value in self.revenue_estimates_cents):
            raise ValueError("revenue telemetry cannot be negative")
        if any(not 0.0 <= value <= 1.0 for value in self.novelty_estimates):
            raise ValueError("novelty telemetry must be in [0, 1]")
        if len(self.mechanism_demand_coefficients) != len(MonetisationMechanism):
            raise ValueError("one demand coefficient is required per mechanism")
        if any(
            not isfinite(value) or value < 0.0
            for value in self.mechanism_demand_coefficients
        ):
            raise ValueError("mechanism demand coefficients must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class FirmKernelView:
    """Read-only engine output; this view is never included in firm observations."""

    firm_id: int
    compliance_effectiveness: float
    evasion_level: float
    detection_risk: float
    collusion_exposure: float
    hidden_savings_cents: int


@dataclass(slots=True)
class _FirmKernelState:
    compliance_effectiveness: float
    evasion_level: float = 0.0
    detection_exposure: float = 0.0
    collusion_exposure: float = 0.0
    hidden_savings_cents: int = 0

    def view(self, firm_id: int) -> FirmKernelView:
        return FirmKernelView(
            firm_id=firm_id,
            compliance_effectiveness=float(np.clip(self.compliance_effectiveness, 0.0, 1.0)),
            evasion_level=float(np.clip(self.evasion_level, 0.0, 1.0)),
            detection_risk=float(1.0 - exp(-max(0.0, self.detection_exposure))),
            collusion_exposure=float(max(0.0, self.collusion_exposure)),
            hidden_savings_cents=int(self.hidden_savings_cents),
        )


@dataclass(frozen=True, slots=True)
class IntentRecord:
    """Auditable resolution of one intent, including rejected proposals."""

    tick: int
    intent: FirmIntent
    accepted: bool
    reason: str
    cash_before_cents: int
    cash_after_cents: int
    charged_cents: int = 0
    ledger_reference: str | None = None
    partner_firm_id: int | None = None
    content_candidate: ContentCandidate | None = None


@dataclass(frozen=True, slots=True)
class FirmResolution:
    """All firm-side effects produced by a simultaneous decision round."""

    tick: int
    intents: tuple[FirmIntent, ...]
    records: tuple[IntentRecord, ...]
    promotion_pressure: FloatArray
    firm_kernel_state: tuple[FirmKernelView, ...]
    subsidy_applications: tuple[SubsidyApplicationView, ...]
    collaborations: tuple[tuple[int, int], ...]
    collusions: tuple[tuple[int, int], ...]

    @property
    def compliance_state(self) -> dict[int, float]:
        return {
            state.firm_id: state.compliance_effectiveness
            for state in self.firm_kernel_state
        }

    @property
    def evasion_state(self) -> dict[int, float]:
        return {state.firm_id: state.evasion_level for state in self.firm_kernel_state}


def _sigmoid(value: npt.ArrayLike) -> FloatArray:
    values = np.asarray(value, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_value = np.exp(values[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def create_firms(
    *,
    company_count: int,
    games: GameTable,
    rng: CounterRNG,
    tick: int = 0,
    base_cash_cents: int = 8_000_000,
    base_research_cost_cents: int = 250_000,
) -> tuple[FirmAgent, ...]:
    """Create a correlated, continuously heterogeneous firm population.

    No behavioural archetype is assigned.  Scale, culture, ethics, analytics,
    risk tolerance and exploration are correlated but remain continuous, so
    strategy categories can emerge from decisions rather than labels.
    """

    if isinstance(company_count, bool) or not isinstance(company_count, (int, np.integer)):
        raise TypeError("company_count must be an integer")
    if company_count <= 0:
        raise ValueError("company_count must be positive")
    if tick < 0 or base_cash_cents <= 0 or base_research_cost_cents <= 0:
        raise ValueError(
            "tick must be non-negative and base cash/research cost positive"
        )
    if len(games.game_id) == 0:
        raise ValueError("at least one game is required")
    company_ids = np.asarray(games.company_id, dtype=np.int64)
    if np.any((company_ids < 0) | (company_ids >= company_count)):
        raise ValueError("game table contains an out-of-range company id")

    entity_ids = np.arange(company_count, dtype=np.int64)
    independent = np.column_stack(
        [
            np.asarray(rng.normal(entity_ids, tick, "firm_factory_traits", lane))
            for lane in range(6)
        ]
    )
    # A fixed positive-definite loading matrix creates realistic correlations:
    # analytics follows scale, while aggressive exploration tends to oppose
    # compliance culture and ethics without making either deterministic.
    loadings = np.asarray(
        [
            [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.22, 0.98, 0.00, 0.00, 0.00, 0.00],
            [0.10, 0.35, 0.93, 0.00, 0.00, 0.00],
            [0.48, 0.08, 0.05, 0.87, 0.00, 0.00],
            [-0.18, -0.31, -0.22, 0.12, 0.90, 0.00],
            [-0.12, 0.10, 0.08, -0.05, 0.18, 0.96],
        ],
        dtype=np.float64,
    )
    latent = independent @ loadings.T
    traits = _sigmoid(latent * np.asarray([0.85, 0.95, 0.95, 1.00, 1.05, 0.70]))
    scale_noise = np.asarray(rng.normal(entity_ids, tick, "firm_factory_scale", 0))
    scale = np.exp(np.clip(0.45 * scale_noise, -1.2, 1.2))

    firms: list[FirmAgent] = []
    for firm_id in range(company_count):
        rows = np.flatnonzero(company_ids == firm_id)
        if not rows.size:
            raise ValueError(f"firm {firm_id} has no game")
        game_ids = tuple(int(games.game_id[row]) for row in rows)
        cash = int(round(base_cash_cents * scale[firm_id] * (0.8 + 0.2 * rows.size)))
        operating_scale = cash / max(base_cash_cents, 1)
        content_cost = max(25_000, int(round(900_000 * operating_scale**0.72)))
        acquisition_cost = max(10_000, int(round(360_000 * operating_scale**0.68)))
        compliance_cost = max(10_000, int(round(300_000 * operating_scale**0.62)))
        research_cost = max(
            10_000,
            int(round(base_research_cost_cents * operating_scale**0.66)),
        )

        trust = {
            other: float(
                np.clip(
                    rng.uniform(
                        firm_id,
                        tick,
                        "firm_factory_partner_trust",
                        other,
                        low=0.08,
                        high=0.72,
                    ),
                    0.0,
                    1.0,
                )
            )
            for other in range(company_count)
            if other != firm_id
        }
        firms.append(
            FirmAgent(
                firm_id=firm_id,
                state=FirmPrivateState(
                    cash_cents=cash,
                    game_ids=game_ids,
                    collusive_trust=trust,
                ),
                risk_aversion=float(traits[firm_id, 0]),
                compliance_culture=float(traits[firm_id, 1]),
                ethics_weight=float(traits[firm_id, 2]),
                analytics_capability=float(traits[firm_id, 3]),
                exploration_tendency=float(traits[firm_id, 4]),
                discount_rate=float(0.004 + 0.026 * traits[firm_id, 5]),
                content_cost_cents=content_cost,
                acquisition_cost_cents=acquisition_cost,
                compliance_cost_cents=compliance_cost,
                research_cost_cents=research_cost,
            )
        )
    return tuple(firms)


def capture_period_telemetry(
    *,
    tick: int,
    games: GameTable,
    firms: Sequence[FirmAgent],
    rng: CounterRNG,
    period_revenue_cents: npt.ArrayLike | None = None,
    active_players: npt.ArrayLike | None = None,
) -> tuple[FirmTelemetry, ...]:
    """Measure only portfolio-local period aggregates with firm-specific error."""

    game_row = {int(game_id): row for row, game_id in enumerate(games.game_id)}
    revenue_source = (
        games.revenue_cents
        if period_revenue_cents is None
        else np.asarray(period_revenue_cents, dtype=np.int64)
    )
    active_source = (
        games.active_players
        if active_players is None
        else np.asarray(active_players, dtype=np.int64)
    )
    expected_shape = games.game_id.shape
    if revenue_source.shape != expected_shape or np.any(revenue_source < 0):
        raise ValueError("period revenue telemetry must align with games")
    if active_source.shape != expected_shape or np.any(active_source < 0):
        raise ValueError("active-player telemetry must align with games")
    telemetry: list[FirmTelemetry] = []
    for firm in sorted(firms, key=lambda item: item.firm_id):
        rows = np.asarray([game_row[game] for game in firm.state.game_ids], dtype=np.int64)
        if np.any(games.company_id[rows] != firm.firm_id):
            raise ValueError("a firm attempted to capture another firm's telemetry")
        ids = np.asarray(firm.state.game_ids, dtype=np.int64)
        measurement_sigma = 0.025 + 0.13 * (1.0 - firm.analytics_capability)
        active_error = np.exp(
            np.clip(
                measurement_sigma
                * np.asarray(rng.normal(ids, tick, "firm_telemetry_active", firm.firm_id)),
                -0.7,
                0.7,
            )
        )
        revenue_error = np.exp(
            np.clip(
                measurement_sigma
                * np.asarray(rng.normal(ids, tick, "firm_telemetry_revenue", firm.firm_id)),
                -0.7,
                0.7,
            )
        )
        novelty_error = measurement_sigma * np.asarray(
            rng.normal(ids, tick, "firm_telemetry_novelty", firm.firm_id)
        )
        active = np.rint(active_source[rows] * active_error).astype(np.int64)
        revenue = np.rint(revenue_source[rows] * revenue_error).astype(np.int64)
        novelty = np.clip(games.novelty[rows] + novelty_error, 0.0, 1.0)

        weights = np.maximum(revenue.astype(np.float64), 1.0)
        revealed_intensity = np.average(games.monetisation[rows], axis=0, weights=weights)
        mechanism_noise = np.asarray(
            rng.normal(
                np.full(len(MonetisationMechanism), firm.firm_id, dtype=np.int64),
                tick,
                "firm_telemetry_mechanism_response",
                np.arange(len(MonetisationMechanism), dtype=np.int64),
            )
        )
        demand = np.clip(
            (0.20 + revealed_intensity)
            * np.exp(np.clip(measurement_sigma * mechanism_noise, -0.6, 0.6)),
            0.0,
            2.0,
        )
        telemetry.append(
            FirmTelemetry(
                as_of=tick,
                firm_id=firm.firm_id,
                game_ids=firm.state.game_ids,
                active_player_estimates=tuple(int(value) for value in active),
                revenue_estimates_cents=tuple(int(value) for value in revenue),
                novelty_estimates=tuple(float(value) for value in novelty),
                mechanism_demand_coefficients=tuple(float(value) for value in demand),
            )
        )
    return tuple(telemetry)


class FirmStrategySystem:
    """Build local observations, collect simultaneous intents, then resolve them."""

    def __init__(
        self,
        firms: Sequence[FirmAgent],
        *,
        rng: CounterRNG,
        content_planner: ContentPlanner | None = None,
        policies: Mapping[int, FirmPolicy] | None = None,
        public_signal_delay: int = 1,
        public_signal_noise: float = 0.12,
        audit_probability_prior: float = 0.12,
        expected_fine_cents: int = 5_000_000,
        regulatory_uncertainty: float = 0.35,
        subsidy_success_prior: float = 0.30,
        expected_subsidy_cents: int = 1_200_000,
        research_precision_gain: float = 0.22,
        evasion_concealment_cost_share: float = 0.10,
        collaboration_efficiency: float = 0.08,
        collusion_intensity_gain: float = 0.025,
    ) -> None:
        self.firms = tuple(sorted(firms, key=lambda firm: firm.firm_id))
        if not self.firms:
            raise ValueError("at least one firm is required")
        ids = tuple(firm.firm_id for firm in self.firms)
        if len(set(ids)) != len(ids) or any(firm_id < 0 for firm_id in ids):
            raise ValueError("firm ids must be unique and non-negative")
        if public_signal_delay < 0:
            raise ValueError("public signal delay cannot be negative")
        bounded = (
            public_signal_noise,
            audit_probability_prior,
            regulatory_uncertainty,
            subsidy_success_prior,
            research_precision_gain,
            evasion_concealment_cost_share,
            collaboration_efficiency,
            collusion_intensity_gain,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("strategy probabilities and rates must be in [0, 1]")
        if expected_fine_cents < 0 or expected_subsidy_cents < 0:
            raise ValueError("expected transfers cannot be negative")

        self.rng = rng
        self.content_planner = content_planner or ContentPlanner()
        self.policies = dict(policies or {})
        unknown_policies = set(self.policies) - set(ids)
        if unknown_policies:
            raise ValueError(f"policies supplied for unknown firms: {sorted(unknown_policies)}")
        self.public_signal_delay = int(public_signal_delay)
        self.public_signal_noise = float(public_signal_noise)
        self.audit_probability_prior = float(audit_probability_prior)
        self.expected_fine_cents = int(expected_fine_cents)
        self.regulatory_uncertainty = float(regulatory_uncertainty)
        self.subsidy_success_prior = float(subsidy_success_prior)
        self.expected_subsidy_cents = int(expected_subsidy_cents)
        self.research_precision_gain = float(research_precision_gain)
        self.evasion_concealment_cost_share = float(evasion_concealment_cost_share)
        self.collaboration_efficiency = float(collaboration_efficiency)
        self.collusion_intensity_gain = float(collusion_intensity_gain)

        self._firm_by_id = {firm.firm_id: firm for firm in self.firms}
        self._rankings: dict[int, PublicRankingSnapshot] = {}
        self._latest_telemetry: dict[int, FirmTelemetry] = {}
        self._latest_observation: dict[int, CompanyObservation] = {}
        self._kernel = {
            firm.firm_id: _FirmKernelState(
                compliance_effectiveness=float(
                    np.clip(
                        0.08 + 0.62 * firm.compliance_culture + 0.18 * firm.ethics_weight,
                        0.0,
                        0.92,
                    )
                )
            )
            for firm in self.firms
        }
        self._intent_log: list[IntentRecord] = []
        self._ledger_serial = 0

    @property
    def intent_log(self) -> tuple[IntentRecord, ...]:
        return tuple(self._intent_log)

    def publish_public_ranking(self, *, tick: int, games: GameTable) -> PublicRankingSnapshot:
        """Publish a noisy ordering derived solely from the table's public rank."""

        if tick < 0:
            raise ValueError("tick cannot be negative")
        game_ids = np.asarray(games.game_id, dtype=np.int64)
        ranks = np.asarray(games.public_rank, dtype=np.float64)
        if len(game_ids) == 0:
            raise ValueError("cannot publish an empty ranking")
        noise = np.asarray(
            self.rng.normal(game_ids, tick, "firm_public_ranking_noise", 0),
            dtype=np.float64,
        )
        perceived_score = -ranks + self.public_signal_noise * len(game_ids) * noise
        ordering = np.lexsort((game_ids, -perceived_score))
        noisy_rank = np.empty(len(game_ids), dtype=np.int64)
        noisy_rank[ordering] = np.arange(1, len(game_ids) + 1, dtype=np.int64)
        snapshot = PublicRankingSnapshot(
            as_of=tick,
            game_ids=tuple(int(value) for value in game_ids),
            public_rank=tuple(int(value) for value in noisy_rank),
            data_tick=tick,
        )
        self.record_public_ranking(snapshot)
        return snapshot

    def record_public_ranking(self, snapshot: PublicRankingSnapshot) -> None:
        existing = self._rankings.get(snapshot.as_of)
        if existing is not None and existing != snapshot:
            raise ValueError("conflicting public rankings for the same source tick")
        self._rankings[snapshot.as_of] = snapshot

    def _eligible_ranking(self, tick: int) -> PublicRankingSnapshot | None:
        cutoff = tick - self.public_signal_delay
        eligible = [as_of for as_of in self._rankings if as_of <= cutoff]
        if not eligible:
            return None
        return self._rankings[max(eligible)]

    @staticmethod
    def _normalise_telemetry(
        period_telemetry: Iterable[FirmTelemetry] | Mapping[int, FirmTelemetry],
    ) -> dict[int, FirmTelemetry]:
        values = (
            tuple(period_telemetry.values())
            if isinstance(period_telemetry, Mapping)
            else tuple(period_telemetry)
        )
        result: dict[int, FirmTelemetry] = {}
        for item in values:
            if item.firm_id in result:
                raise ValueError(f"duplicate telemetry for firm {item.firm_id}")
            result[item.firm_id] = item
        return result

    def build_observation(self, *, tick: int, telemetry: FirmTelemetry) -> CompanyObservation:
        """Create one observation without reading latent market or player state."""

        firm = self._firm_by_id.get(telemetry.firm_id)
        if firm is None:
            raise ValueError(f"telemetry belongs to unknown firm {telemetry.firm_id}")
        if telemetry.game_ids != firm.state.game_ids:
            raise ValueError("telemetry does not match the firm's own portfolio")
        if telemetry.as_of > tick:
            raise ValueError("future telemetry cannot be observed")

        snapshot = self._eligible_ranking(tick)
        if snapshot is None:
            unknown_rank = max(1, sum(len(item.state.game_ids) for item in self.firms) + 1)
            own_ranks = tuple(unknown_rank for _ in telemetry.game_ids)
            concentration = 1.0 / max(1, len(self.firms))
            competitor_pressure = 0.5
            signal_age = max(1, self.public_signal_delay)
        else:
            rank_by_game = snapshot.as_mapping()
            missing_rank = len(snapshot.game_ids) + 1
            own_ranks = tuple(rank_by_game.get(game_id, missing_rank) for game_id in telemetry.game_ids)
            reciprocal = np.asarray(
                [1.0 / rank for rank in snapshot.public_rank], dtype=np.float64
            )
            shares = reciprocal / reciprocal.sum()
            concentration = float(np.dot(shares, shares))
            own_set = set(telemetry.game_ids)
            competitor_ranks = [
                rank
                for game_id, rank in zip(
                    snapshot.game_ids, snapshot.public_rank, strict=True
                )
                if game_id not in own_set
            ]
            if competitor_ranks:
                scale = max(1, len(snapshot.game_ids) - 1)
                competitor_pressure = float(
                    np.clip((min(own_ranks) - min(competitor_ranks) + scale) / (2.0 * scale), 0.0, 1.0)
                )
            else:
                competitor_pressure = 0.0
            assert snapshot.data_tick is not None
            signal_age = tick - snapshot.data_tick

        # ``_latest_telemetry`` is the completed previous decision round.  It is
        # replaced only after observations for every firm have been constructed.
        previous = self._latest_telemetry.get(firm.firm_id)
        if previous is None:
            growth = 0.0
        else:
            previous_active = sum(previous.active_player_estimates)
            current_active = sum(telemetry.active_player_estimates)
            growth = float(
                np.clip((current_active - previous_active) / max(1, previous_active), -1.0, 3.0)
            )

        research_stock = float(np.clip(firm.state.analytics_investment, 0.0, 1.0))
        belief_sigma = self.regulatory_uncertainty * (
            1.0 - 0.75 * research_stock
        ) * (1.15 - 0.45 * firm.analytics_capability)
        audit_noise = float(
            self.rng.normal(firm.firm_id, tick, "firm_audit_belief", 0)
        )
        fine_noise = float(
            self.rng.normal(firm.firm_id, tick, "firm_fine_belief", 0)
        )
        subsidy_noise = float(
            self.rng.normal(firm.firm_id, tick, "firm_subsidy_belief", 0)
        )
        audit_probability = float(
            np.clip(self.audit_probability_prior + 0.18 * belief_sigma * audit_noise, 0.0, 1.0)
        )
        expected_fine = int(
            round(
                self.expected_fine_cents
                * exp(float(np.clip(0.35 * belief_sigma * fine_noise, -1.0, 1.0)))
            )
        )
        subsidy_probability = float(
            np.clip(self.subsidy_success_prior + 0.20 * belief_sigma * subsidy_noise, 0.0, 1.0)
        )
        return CompanyObservation(
            as_of=tick,
            own_game_ids=telemetry.game_ids,
            own_active_estimates=telemetry.active_player_estimates,
            own_revenue_estimates_cents=telemetry.revenue_estimates_cents,
            own_novelty_estimates=telemetry.novelty_estimates,
            public_rank_by_game=own_ranks,
            mechanism_demand_coefficients=telemetry.mechanism_demand_coefficients,
            market_growth_estimate=growth,
            competitor_pressure_estimate=competitor_pressure,
            concentration_estimate=concentration,
            audit_probability_mean=audit_probability,
            expected_fine_cents=expected_fine,
            regulatory_uncertainty=float(
                np.clip(self.regulatory_uncertainty * (1.0 - 0.65 * research_stock), 0.0, 1.0)
            ),
            subsidy_success_probability=subsidy_probability,
            expected_subsidy_cents=self.expected_subsidy_cents,
            research_precision_gain=float(
                np.clip(self.research_precision_gain * (1.0 - research_stock), 0.0, 1.0)
            ),
            signal_age_days=signal_age,
        )

    @staticmethod
    def _candidate_count(firm: FirmAgent) -> int:
        del firm
        # One draw per semantic action, even when the action is infeasible.
        return len(FirmAction)

    def _select_partner(
        self,
        *,
        firm: FirmAgent,
        action: FirmAction,
        tick: int,
    ) -> int | None:
        competitors = [candidate for candidate in self.firms if candidate.firm_id != firm.firm_id]
        if not competitors:
            return None
        snapshot = self._eligible_ranking(tick)
        rank_map = snapshot.as_mapping() if snapshot is not None else {}
        missing = (len(snapshot.game_ids) + 1) if snapshot is not None else 1
        own_best = min((rank_map.get(game, missing) for game in firm.state.game_ids), default=missing)

        scored: list[tuple[float, int]] = []
        for candidate in competitors:
            other_best = min(
                (rank_map.get(game, missing) for game in candidate.state.game_ids),
                default=missing,
            )
            trust = firm.state.collusive_trust.get(candidate.firm_id, 0.0)
            if action is FirmAction.PROPOSE_COLLABORATION:
                rank_component = 1.0 / (1.0 + abs(own_best - other_best))
                score = 0.68 * rank_component + 0.32 * trust
            else:
                market_relevance = 1.0 / max(1, other_best)
                score = 0.62 * trust + 0.38 * market_relevance
            tie_break = float(
                self.rng.uniform(
                    firm.firm_id,
                    tick,
                    "firm_partner_tie_break",
                    candidate.firm_id,
                )
            )
            scored.append((score + tie_break * 1e-9, candidate.firm_id))
        return max(scored, key=lambda item: (item[0], -item[1]))[1]

    def collect_intents(
        self,
        *,
        tick: int,
        period_telemetry: Iterable[FirmTelemetry] | Mapping[int, FirmTelemetry],
        games: GameTable | None = None,
    ) -> tuple[FirmIntent, ...]:
        """Collect every decision before any game, cash, or kernel state changes."""

        if tick < 0:
            raise ValueError("tick cannot be negative")
        telemetry_by_firm = self._normalise_telemetry(period_telemetry)
        expected_ids = set(self._firm_by_id)
        if set(telemetry_by_firm) != expected_ids:
            missing = sorted(expected_ids - set(telemetry_by_firm))
            extra = sorted(set(telemetry_by_firm) - expected_ids)
            raise ValueError(f"telemetry coverage mismatch; missing={missing}, extra={extra}")

        observations = {
            firm.firm_id: self.build_observation(
                tick=tick, telemetry=telemetry_by_firm[firm.firm_id]
            )
            for firm in self.firms
        }
        intents: list[FirmIntent] = []
        for firm in self.firms:
            observation = observations[firm.firm_id]
            policy = self.policies.get(firm.firm_id)
            if policy is None:
                candidate_count = self._candidate_count(firm)
                shocks = np.asarray(
                    self.rng.normal(
                        np.full(candidate_count, firm.firm_id, dtype=np.int64),
                        tick,
                        "firm_action_choice",
                        np.arange(candidate_count, dtype=np.int64),
                    ),
                    dtype=np.float64,
                )
                intent = firm.decide(
                    observation,
                    action_shocks=tuple(float(value) for value in shocks),
                )
            else:
                # Deliberately pass exactly one value.  In particular, no GameTable,
                # world/kernel handle, player table, or regulator state is exposed.
                intent = policy(observation)
            if not isinstance(intent, FirmIntent):
                raise TypeError("firm policies must return FirmIntent")
            if intent.firm_id != firm.firm_id:
                raise ValueError("a policy returned an intent for another firm")
            if intent.action in {
                FirmAction.PROPOSE_COLLABORATION,
                FirmAction.PROPOSE_COLLUSION,
            } and intent.target_firm_id is None:
                intent = replace(
                    intent,
                    target_firm_id=self._select_partner(
                        firm=firm, action=intent.action, tick=tick
                    ),
                )
            intents.append(intent)

        # Updating memory happens after all observations were constructed, so one
        # firm's decision order cannot affect another firm's growth estimate.
        self._latest_telemetry = dict(telemetry_by_firm)
        self._latest_observation = observations
        return tuple(intents)

    @staticmethod
    def _game_rows(games: GameTable) -> dict[int, int]:
        rows = {int(game_id): row for row, game_id in enumerate(games.game_id)}
        if len(rows) != len(games.game_id):
            raise ValueError("game ids must be unique")
        return rows

    def _reference(self, tick: int, firm_id: int, action: FirmAction) -> str:
        reference = f"firm-strategy:{tick}:{self._ledger_serial}:{firm_id}:{action.value}"
        self._ledger_serial += 1
        return reference

    def _charge(
        self,
        *,
        tick: int,
        firm: FirmAgent,
        action: FirmAction,
        amount_cents: int,
        destination: str,
        ledger: Ledger,
    ) -> str | None:
        if amount_cents <= 0:
            return None
        reference = self._reference(tick, firm.firm_id, action)
        ledger.transfer(
            tick=tick,
            debit_account=f"firm:{firm.firm_id}:cash",
            credit_account=destination,
            amount_cents=amount_cents,
            kind=action.value,
            reference=reference,
        )
        firm.state.cash_cents -= amount_cents
        return reference

    @staticmethod
    def _configured_cost(intent: FirmIntent, default: int) -> int:
        if intent.committed_cost_cents < 0:
            raise ValueError("committed cost cannot be negative")
        return int(intent.committed_cost_cents or default)

    def _content_snapshot(
        self,
        *,
        firm: FirmAgent,
        game_id: int,
        row: int,
        games: GameTable,
        cost_cents: int,
        tick: int,
    ) -> OwnGameSnapshot:
        telemetry = self._latest_telemetry.get(firm.firm_id)
        observation = self._latest_observation.get(firm.firm_id)
        if telemetry is not None and game_id in telemetry.game_ids:
            position = telemetry.game_ids.index(game_id)
            active = telemetry.active_player_estimates[position]
            revenue = telemetry.revenue_estimates_cents[position]
        else:
            # This fallback is used only for manually supplied test/engine intents;
            # these are still the firm's own operational counters.
            active = int(games.active_players[row])
            revenue = int(games.revenue_cents[row])
        price = int(games.price_cents[row])
        conversion = float(np.clip(revenue / max(1, active * price), 0.0, 1.0))
        dimensions = games.stat_frontier.shape[1]
        demand_noise = np.asarray(
            self.rng.normal(
                np.full(dimensions, firm.firm_id, dtype=np.int64),
                tick,
                "firm_content_demand_belief",
                np.arange(dimensions, dtype=np.int64),
            ),
            dtype=np.float64,
        )
        demand_weights = np.exp(
            np.clip(
                (0.18 + 0.22 * firm.analytics_capability) * demand_noise,
                -1.5,
                1.5,
            )
        )
        audit_probability = (
            observation.audit_probability_mean
            if observation is not None
            else self.audit_probability_prior
        )
        expected_fine = (
            observation.expected_fine_cents
            if observation is not None
            else self.expected_fine_cents
        )
        return OwnGameSnapshot(
            game_id=game_id,
            stat_frontier=games.stat_frontier[row].copy(),
            demand_weights=demand_weights,
            active_players_estimate=active,
            price_cents=price,
            content_cost_cents=cost_cents,
            estimated_conversion=conversion,
            estimated_audit_probability=audit_probability,
            estimated_fine_cents=expected_fine,
            reputation_sensitivity=float(
                np.clip(0.08 + 0.52 * firm.ethics_weight, 0.0, 1.0)
            ),
            analytics_quality=float(
                np.clip(
                    firm.analytics_capability + 0.25 * firm.state.analytics_investment,
                    0.0,
                    1.0,
                )
            ),
        )

    def _record(
        self,
        *,
        tick: int,
        intent: FirmIntent,
        accepted: bool,
        reason: str,
        cash_before: int,
        charged: int = 0,
        ledger_reference: str | None = None,
        partner: int | None = None,
        candidate: ContentCandidate | None = None,
    ) -> IntentRecord:
        record = IntentRecord(
            tick=tick,
            intent=intent,
            accepted=accepted,
            reason=reason,
            cash_before_cents=cash_before,
            cash_after_cents=self._firm_by_id[intent.firm_id].state.cash_cents,
            charged_cents=charged,
            ledger_reference=ledger_reference,
            partner_firm_id=partner,
            content_candidate=candidate,
        )
        self._intent_log.append(record)
        return record

    def _resolve_single(
        self,
        *,
        tick: int,
        intent: FirmIntent,
        games: GameTable,
        game_rows: Mapping[int, int],
        ledger: Ledger,
        promotion: FloatArray,
        applications: list[SubsidyApplicationView],
    ) -> IntentRecord:
        firm = self._firm_by_id[intent.firm_id]
        before = firm.state.cash_cents
        owned_games = set(firm.state.game_ids)

        if intent.action is FirmAction.HOLD:
            return self._record(
                tick=tick,
                intent=intent,
                accepted=True,
                reason="held position",
                cash_before=before,
            )

        if intent.action is FirmAction.RELEASE_CONTENT:
            game_id = intent.target_game_id
            if game_id not in owned_games or game_id not in game_rows:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="content target is not owned by the firm",
                    cash_before=before,
                )
            cost = self._configured_cost(intent, firm.content_cost_cents)
            if before < cost:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="insufficient cash for content",
                    cash_before=before,
                )
            row = game_rows[game_id]
            if game_id != row:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="content planner requires row-aligned game ids",
                    cash_before=before,
                )
            try:
                candidate = self.content_planner.choose(
                    self._content_snapshot(
                        firm=firm,
                        game_id=game_id,
                        row=row,
                        games=games,
                        cost_cents=cost,
                        tick=tick,
                    )
                )
            except ValueError as exc:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason=f"content planning failed: {exc}",
                    cash_before=before,
                )
            reference = self._charge(
                tick=tick,
                firm=firm,
                action=intent.action,
                amount_cents=cost,
                destination="sector:content-production",
                ledger=ledger,
            )
            games.apply_content(candidate)
            return self._record(
                tick=tick,
                intent=intent,
                accepted=True,
                reason="content released",
                cash_before=before,
                charged=cost,
                ledger_reference=reference,
                candidate=candidate,
            )

        if intent.action is FirmAction.ADJUST_MONETISATION:
            game_id = intent.target_game_id
            if game_id not in owned_games or game_id not in game_rows:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="monetisation target is not owned by the firm",
                    cash_before=before,
                )
            if intent.mechanism is None or not isfinite(intent.intensity_delta):
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="monetisation mechanism and finite delta are required",
                    cash_before=before,
                )
            row = game_rows[game_id]
            mechanism = int(intent.mechanism)
            games.monetisation[row, mechanism] = np.clip(
                games.monetisation[row, mechanism] + intent.intensity_delta,
                0.0,
                1.0,
            )
            return self._record(
                tick=tick,
                intent=intent,
                accepted=True,
                reason="monetisation adjusted",
                cash_before=before,
            )

        if intent.action in {
            FirmAction.BUY_RESEARCH,
            FirmAction.INVEST_COMPLIANCE,
            FirmAction.ACQUIRE_USERS,
        }:
            defaults = {
                FirmAction.BUY_RESEARCH: firm.research_cost_cents,
                FirmAction.INVEST_COMPLIANCE: firm.compliance_cost_cents,
                FirmAction.ACQUIRE_USERS: firm.acquisition_cost_cents,
            }
            destinations = {
                FirmAction.BUY_RESEARCH: "sector:research",
                FirmAction.INVEST_COMPLIANCE: "sector:compliance",
                FirmAction.ACQUIRE_USERS: "sector:advertising",
            }
            cost = self._configured_cost(intent, defaults[intent.action])
            if before < cost:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="insufficient cash for investment",
                    cash_before=before,
                )
            if intent.action is FirmAction.ACQUIRE_USERS and (
                intent.target_game_id not in owned_games
                or intent.target_game_id not in game_rows
            ):
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="acquisition target is not owned by the firm",
                    cash_before=before,
                )
            reference = self._charge(
                tick=tick,
                firm=firm,
                action=intent.action,
                amount_cents=cost,
                destination=destinations[intent.action],
                ledger=ledger,
            )
            if intent.action is FirmAction.BUY_RESEARCH:
                dose = cost / max(1, firm.research_cost_cents)
                firm.state.analytics_investment = float(
                    1.0 - (1.0 - np.clip(firm.state.analytics_investment, 0.0, 1.0)) * exp(-0.32 * dose)
                )
            elif intent.action is FirmAction.INVEST_COMPLIANCE:
                dose = cost / max(1, firm.compliance_cost_cents)
                firm.state.compliance_investment = float(
                    1.0 - (1.0 - np.clip(firm.state.compliance_investment, 0.0, 1.0)) * exp(-0.38 * dose)
                )
                kernel = self._kernel[firm.firm_id]
                kernel.compliance_effectiveness = float(
                    1.0 - (1.0 - kernel.compliance_effectiveness) * exp(-0.34 * dose)
                )
                kernel.evasion_level *= exp(-0.22 * dose)
            else:
                dose = cost / max(1, firm.acquisition_cost_cents)
                firm.state.acquisition_stock = float(
                    firm.state.acquisition_stock * exp(-0.08)
                    + dose * (0.45 + firm.analytics_capability)
                )
                row = game_rows[intent.target_game_id]
                promotion[row] += log1p(dose) * (
                    0.35 + 0.65 * firm.analytics_capability
                )
            return self._record(
                tick=tick,
                intent=intent,
                accepted=True,
                reason=f"{intent.action.value} completed",
                cash_before=before,
                charged=cost,
                ledger_reference=reference,
            )

        if intent.action is FirmAction.EVADE:
            game_id = intent.target_game_id
            if game_id is not None and game_id not in owned_games:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="evasion target is not owned by the firm",
                    cash_before=before,
                )
            endogenous_cost = max(
                1,
                int(
                    round(
                        firm.compliance_cost_cents
                        * self.evasion_concealment_cost_share
                        * (0.55 + 0.45 * firm.analytics_capability)
                    )
                ),
            )
            cost = max(intent.committed_cost_cents, endogenous_cost)
            if before < cost:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="insufficient cash to conceal evasion",
                    cash_before=before,
                )
            reference = self._charge(
                tick=tick,
                firm=firm,
                action=intent.action,
                amount_cents=cost,
                destination="sector:concealment-cost",
                ledger=ledger,
            )
            kernel = self._kernel[firm.firm_id]
            implementation_shock = float(
                self.rng.normal(firm.firm_id, tick, "firm_evasion_effectiveness", 0)
            )
            increment = float(
                np.clip(
                    (0.035 + 0.16 * firm.exploration_tendency)
                    * exp(float(np.clip(0.14 * implementation_shock, -0.4, 0.4))),
                    0.01,
                    0.30,
                )
            )
            kernel.evasion_level = float(
                1.0 - (1.0 - kernel.evasion_level) * exp(-increment)
            )
            saved = int(
                round(
                    firm.compliance_cost_cents
                    * increment
                    * (0.65 + 0.55 * firm.exploration_tendency)
                )
            )
            kernel.hidden_savings_cents += saved
            opacity = 1.0 - 0.45 * firm.analytics_capability
            kernel.detection_exposure += (
                increment
                * opacity
                * (1.0 - 0.70 * kernel.compliance_effectiveness)
                * (0.65 + firm.risk_aversion)
            )
            return self._record(
                tick=tick,
                intent=intent,
                accepted=True,
                reason="evasion entered latent enforcement state",
                cash_before=before,
                charged=cost,
                ledger_reference=reference,
            )

        if intent.action is FirmAction.APPLY_SUBSIDY:
            rows = np.asarray([game_rows[game] for game in firm.state.game_ids], dtype=np.int64)
            observation = self._latest_observation.get(firm.firm_id)
            requested = (
                observation.expected_subsidy_cents
                if observation is not None
                else self.expected_subsidy_cents
            )
            if requested <= 0:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="no positive subsidy amount available",
                    cash_before=before,
                )
            # This is the same administrative burden the default policy prices
            # into its perceived subsidy NPV in ``FirmAgent.decide``.
            application_cost = max(
                intent.committed_cost_cents,
                int(round(0.08 * firm.research_cost_cents)),
            )
            if before < application_cost:
                return self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="insufficient cash for subsidy application",
                    cash_before=before,
                )
            reference = self._charge(
                tick=tick,
                firm=firm,
                action=intent.action,
                amount_cents=application_cost,
                destination="sector:grant-administration",
                ledger=ledger,
            )
            intensity = np.mean(games.monetisation[rows], axis=1)
            design_safety = float(
                np.clip(
                    np.average(1.0 - intensity, weights=np.maximum(games.revenue_cents[rows], 1)),
                    0.0,
                    1.0,
                )
            )
            applications.append(
                SubsidyApplicationView(
                    firm_id=firm.firm_id,
                    requested_cents=int(requested),
                    verified_quality=float(np.clip(np.mean(games.quality[rows]), 0.0, 1.0)),
                    verified_design_safety_score=design_safety,
                    verified_accessibility=float(
                        np.clip(np.mean(games.competitive_integrity[rows]), 0.0, 1.0)
                    ),
                    jobs_estimate=max(
                        1,
                        int(
                            round(
                                games.active_players[rows].sum(dtype=np.int64) / 8_000
                                + games.revenue_cents[rows].sum(dtype=np.int64) / 80_000_000
                            )
                        ),
                    ),
                    evidence_age_days=max(0, tick - (observation.as_of if observation else tick)),
                    submitted_tick=tick,
                )
            )
            return self._record(
                tick=tick,
                intent=intent,
                accepted=True,
                reason="subsidy application submitted",
                cash_before=before,
                charged=application_cost,
                ledger_reference=reference,
            )

        return self._record(
            tick=tick,
            intent=intent,
            accepted=False,
            reason="agreement proposal requires bilateral resolution",
            cash_before=before,
        )

    def _resolve_agreement(
        self,
        *,
        tick: int,
        first_intent: FirmIntent,
        second_intent: FirmIntent,
        games: GameTable,
        game_rows: Mapping[int, int],
        ledger: Ledger,
        promotion: FloatArray,
    ) -> tuple[IntentRecord, IntentRecord]:
        first = self._firm_by_id[first_intent.firm_id]
        second = self._firm_by_id[second_intent.firm_id]
        first_before = first.state.cash_cents
        second_before = second.state.cash_cents
        first_cost = max(0, first_intent.committed_cost_cents)
        second_cost = max(0, second_intent.committed_cost_cents)
        if first.state.cash_cents < first_cost or second.state.cash_cents < second_cost:
            reason = "one partner cannot fund the agreement"
            return (
                self._record(
                    tick=tick,
                    intent=first_intent,
                    accepted=False,
                    reason=reason,
                    cash_before=first_before,
                    partner=second.firm_id,
                ),
                self._record(
                    tick=tick,
                    intent=second_intent,
                    accepted=False,
                    reason=reason,
                    cash_before=second_before,
                    partner=first.firm_id,
                ),
            )

        destination = (
            "sector:joint-production"
            if first_intent.action is FirmAction.PROPOSE_COLLABORATION
            else "sector:agreement-coordination"
        )
        first_reference = self._charge(
            tick=tick,
            firm=first,
            action=first_intent.action,
            amount_cents=first_cost,
            destination=destination,
            ledger=ledger,
        )
        second_reference = self._charge(
            tick=tick,
            firm=second,
            action=second_intent.action,
            amount_cents=second_cost,
            destination=destination,
            ledger=ledger,
        )
        first_rows = np.asarray([game_rows[game] for game in first.state.game_ids], dtype=np.int64)
        second_rows = np.asarray([game_rows[game] for game in second.state.game_ids], dtype=np.int64)
        capability = 0.5 * (first.analytics_capability + second.analytics_capability)

        if first_intent.action is FirmAction.PROPOSE_COLLABORATION:
            first_audience = log1p(int(games.active_players[first_rows].sum(dtype=np.int64)))
            second_audience = log1p(int(games.active_players[second_rows].sum(dtype=np.int64)))
            promotion[first_rows] += self.collaboration_efficiency * second_audience * (0.5 + capability)
            promotion[second_rows] += self.collaboration_efficiency * first_audience * (0.5 + capability)
            reason = "reciprocal collaboration formed"
        else:
            combined_rows = np.concatenate((first_rows, second_rows))
            mechanism = int(np.argmax(np.mean(games.monetisation[combined_rows], axis=0)))
            increment = self.collusion_intensity_gain * (
                0.45 + 0.55 * (first.exploration_tendency + second.exploration_tendency) / 2.0
            )
            games.monetisation[combined_rows, mechanism] = np.clip(
                games.monetisation[combined_rows, mechanism] + increment, 0.0, 1.0
            )
            games.competitive_integrity[combined_rows] = np.clip(
                games.competitive_integrity[combined_rows] - 0.4 * increment,
                0.0,
                1.0,
            )
            for firm in (first, second):
                kernel = self._kernel[firm.firm_id]
                kernel.collusion_exposure += increment * (1.15 - 0.45 * firm.analytics_capability)
                kernel.detection_exposure += increment * (0.5 + firm.risk_aversion)
            reason = "reciprocal collusion formed"

        first.state.collusive_trust[second.firm_id] = float(
            np.clip(first.state.collusive_trust.get(second.firm_id, 0.0) + 0.06, 0.0, 1.0)
        )
        second.state.collusive_trust[first.firm_id] = float(
            np.clip(second.state.collusive_trust.get(first.firm_id, 0.0) + 0.06, 0.0, 1.0)
        )
        return (
            self._record(
                tick=tick,
                intent=first_intent,
                accepted=True,
                reason=reason,
                cash_before=first_before,
                charged=first_cost,
                ledger_reference=first_reference,
                partner=second.firm_id,
            ),
            self._record(
                tick=tick,
                intent=second_intent,
                accepted=True,
                reason=reason,
                cash_before=second_before,
                charged=second_cost,
                ledger_reference=second_reference,
                partner=first.firm_id,
            ),
        )

    def resolve(
        self,
        *,
        tick: int,
        games: GameTable,
        intents: Iterable[FirmIntent],
        ledger: Ledger,
        period_telemetry: Iterable[FirmTelemetry] | Mapping[int, FirmTelemetry] | None = None,
    ) -> FirmResolution:
        """Resolve a complete simultaneous intent set and post every cash movement."""

        intent_tuple = tuple(intents)
        by_firm: dict[int, FirmIntent] = {}
        for intent in intent_tuple:
            if intent.firm_id not in self._firm_by_id:
                raise ValueError(f"intent belongs to unknown firm {intent.firm_id}")
            if intent.firm_id in by_firm:
                raise ValueError(f"multiple intents supplied for firm {intent.firm_id}")
            by_firm[intent.firm_id] = intent
        if set(by_firm) != set(self._firm_by_id):
            raise ValueError("resolution requires exactly one intent per firm")
        if period_telemetry is not None:
            telemetry = self._normalise_telemetry(period_telemetry)
            if set(telemetry) != set(self._firm_by_id):
                raise ValueError("resolution telemetry must cover every firm")
            self._latest_telemetry = telemetry

        game_rows = self._game_rows(games)
        for firm in self.firms:
            if any(game not in game_rows for game in firm.state.game_ids):
                raise ValueError(f"firm {firm.firm_id} owns an unknown game")
            rows = [game_rows[game] for game in firm.state.game_ids]
            if any(int(games.company_id[row]) != firm.firm_id for row in rows):
                raise ValueError("firm portfolios and game ownership disagree")

        promotion = np.zeros(len(games.game_id), dtype=np.float64)
        applications: list[SubsidyApplicationView] = []
        records: dict[int, IntentRecord] = {}
        proposal_actions = {
            FirmAction.PROPOSE_COLLABORATION,
            FirmAction.PROPOSE_COLLUSION,
        }

        for firm_id, intent in sorted(by_firm.items()):
            if intent.action not in proposal_actions:
                records[firm_id] = self._resolve_single(
                    tick=tick,
                    intent=intent,
                    games=games,
                    game_rows=game_rows,
                    ledger=ledger,
                    promotion=promotion,
                    applications=applications,
                )

        matched: set[int] = set()
        collaborations: list[tuple[int, int]] = []
        collusions: list[tuple[int, int]] = []
        for firm_id, intent in sorted(by_firm.items()):
            if intent.action not in proposal_actions or firm_id in matched:
                continue
            target = intent.target_firm_id
            counterpart = by_firm.get(target) if target is not None else None
            compatible = (
                counterpart is not None
                and counterpart.action is intent.action
                and counterpart.target_firm_id == firm_id
                and target != firm_id
                and target not in matched
            )
            if not compatible:
                before = self._firm_by_id[firm_id].state.cash_cents
                records[firm_id] = self._record(
                    tick=tick,
                    intent=intent,
                    accepted=False,
                    reason="no compatible reciprocal proposal",
                    cash_before=before,
                    partner=target,
                )
                continue
            assert target is not None and counterpart is not None
            first_id, second_id = sorted((firm_id, target))
            first_intent = by_firm[first_id]
            second_intent = by_firm[second_id]
            first_record, second_record = self._resolve_agreement(
                tick=tick,
                first_intent=first_intent,
                second_intent=second_intent,
                games=games,
                game_rows=game_rows,
                ledger=ledger,
                promotion=promotion,
            )
            records[first_id] = first_record
            records[second_id] = second_record
            matched.update((first_id, second_id))
            if first_record.accepted:
                pair = (first_id, second_id)
                if intent.action is FirmAction.PROPOSE_COLLABORATION:
                    collaborations.append(pair)
                else:
                    collusions.append(pair)

        games.validate()
        ledger.assert_balanced()
        ordered_records = tuple(records[firm.firm_id] for firm in self.firms)
        return FirmResolution(
            tick=tick,
            intents=tuple(by_firm[firm.firm_id] for firm in self.firms),
            records=ordered_records,
            promotion_pressure=promotion,
            firm_kernel_state=tuple(
                self._kernel[firm.firm_id].view(firm.firm_id) for firm in self.firms
            ),
            subsidy_applications=tuple(
                sorted(applications, key=lambda application: application.firm_id)
            ),
            collaborations=tuple(sorted(collaborations)),
            collusions=tuple(sorted(collusions)),
        )

    def step(
        self,
        *,
        tick: int,
        games: GameTable,
        period_telemetry: Iterable[FirmTelemetry] | Mapping[int, FirmTelemetry],
        ledger: Ledger,
        public_ranking: PublicRankingSnapshot | None = None,
    ) -> FirmResolution:
        """Publish/record public data, decide simultaneously, and resolve one round."""

        if public_ranking is None:
            self.publish_public_ranking(tick=tick, games=games)
        else:
            self.record_public_ranking(public_ranking)
        intents = self.collect_intents(
            tick=tick,
            games=games,
            period_telemetry=period_telemetry,
        )
        return self.resolve(
            tick=tick,
            games=games,
            intents=intents,
            ledger=ledger,
            period_telemetry=period_telemetry,
        )


# Descriptive alias retained for callers that prefer the initialization verb.
initialize_firms = create_firms


__all__ = [
    "FirmKernelView",
    "FirmPolicy",
    "FirmResolution",
    "FirmStrategySystem",
    "FirmTelemetry",
    "IntentRecord",
    "PublicRankingSnapshot",
    "capture_period_telemetry",
    "create_firms",
    "initialize_firms",
]
