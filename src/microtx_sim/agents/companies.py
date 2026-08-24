from __future__ import annotations

from dataclasses import dataclass, field
from math import log1p
from typing import Iterable

import numpy as np

from ..types import FirmAction, MonetisationMechanism


@dataclass(frozen=True, slots=True)
class CompanyObservation:
    """Fallible information available to one company at a decision point."""

    as_of: int
    own_game_ids: tuple[int, ...]
    own_active_estimates: tuple[int, ...]
    own_revenue_estimates_cents: tuple[int, ...]
    own_novelty_estimates: tuple[float, ...]
    public_rank_by_game: tuple[int, ...]
    mechanism_demand_coefficients: tuple[float, ...]
    market_growth_estimate: float
    competitor_pressure_estimate: float
    concentration_estimate: float
    audit_probability_mean: float
    expected_fine_cents: int
    regulatory_uncertainty: float
    subsidy_success_probability: float
    expected_subsidy_cents: int
    research_precision_gain: float
    signal_age_days: int

    def __post_init__(self) -> None:
        game_columns = (
            self.own_active_estimates,
            self.own_revenue_estimates_cents,
            self.own_novelty_estimates,
            self.public_rank_by_game,
        )
        if any(len(column) != len(self.own_game_ids) for column in game_columns):
            raise ValueError("company observation game columns are inconsistent")
        probabilities = (
            self.audit_probability_mean,
            self.subsidy_success_probability,
            self.research_precision_gain,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("probability-like observations must be in [0, 1]")


@dataclass(slots=True)
class FirmPrivateState:
    """State owned and therefore directly known by the firm itself."""

    cash_cents: int
    game_ids: tuple[int, ...]
    compliance_investment: float = 0.0
    analytics_investment: float = 0.0
    acquisition_stock: float = 0.0
    subsidy_receivable_cents: int = 0
    collusive_trust: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FirmIntent:
    firm_id: int
    action: FirmAction
    perceived_value_cents: int
    target_game_id: int | None = None
    target_firm_id: int | None = None
    mechanism: MonetisationMechanism | None = None
    intensity_delta: float = 0.0
    committed_cost_cents: int = 0
    information_fingerprint: str = ""


@dataclass(slots=True)
class FirmAgent:
    firm_id: int
    state: FirmPrivateState
    risk_aversion: float
    compliance_culture: float
    ethics_weight: float
    analytics_capability: float
    discount_rate: float
    exploration_tendency: float
    content_cost_cents: int
    acquisition_cost_cents: int
    compliance_cost_cents: int
    research_cost_cents: int

    def __post_init__(self) -> None:
        bounded = (
            self.risk_aversion,
            self.compliance_culture,
            self.ethics_weight,
            self.analytics_capability,
            self.exploration_tendency,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("firm traits must be in [0, 1]")
        if self.state.cash_cents < 0:
            raise ValueError("firm cash cannot start negative")

    def _fingerprint(self, observation: CompanyObservation) -> str:
        ranks = ",".join(str(rank) for rank in observation.public_rank_by_game)
        return f"t={observation.as_of}|age={observation.signal_age_days}|ranks={ranks}"

    def decide(
        self,
        observation: CompanyObservation,
        *,
        action_shocks: Iterable[float] | None = None,
    ) -> FirmIntent:
        """Choose the highest perceived-value feasible intent from private beliefs."""

        if tuple(observation.own_game_ids) != tuple(self.state.game_ids):
            raise ValueError("observation does not describe the firm's own portfolio")
        if not observation.own_game_ids:
            return FirmIntent(self.firm_id, FirmAction.HOLD, 0)

        cash = self.state.cash_cents
        revenue = sum(observation.own_revenue_estimates_cents)
        active = sum(observation.own_active_estimates)
        average_revenue_per_active = revenue / max(1, active)
        weakest_novelty_index = min(
            range(len(observation.own_game_ids)),
            key=lambda index: (observation.own_novelty_estimates[index], index),
        )
        update_game = observation.own_game_ids[weakest_novelty_index]
        update_gap = 1.0 - observation.own_novelty_estimates[weakest_novelty_index]
        rank_pressure = sum(observation.public_rank_by_game) / max(
            1, len(observation.public_rank_by_game)
        )

        demand = np.asarray(observation.mechanism_demand_coefficients, dtype=np.float64)
        if demand.shape != (len(MonetisationMechanism),):
            raise ValueError("one demand coefficient is required per mechanism")
        mechanism_id = int(np.argmax(demand))
        mechanism = MonetisationMechanism(mechanism_id)

        detection_cost = (
            observation.audit_probability_mean
            * observation.expected_fine_cents
            * (0.4 + self.risk_aversion)
        )
        reputation_scale = revenue * (0.15 + self.ethics_weight * 0.5)
        information_value = (
            observation.regulatory_uncertainty
            * observation.research_precision_gain
            * max(revenue, active * 100)
            * (0.5 + self.analytics_capability)
        )

        candidates: list[FirmIntent] = [FirmIntent(self.firm_id, FirmAction.HOLD, 0)]
        update_value = int(
            round(
                active
                * average_revenue_per_active
                * update_gap
                * (0.35 + self.analytics_capability)
                - self.content_cost_cents
            )
        )
        candidates.append(
            FirmIntent(
                self.firm_id,
                FirmAction.RELEASE_CONTENT,
                update_value,
                target_game_id=update_game,
                committed_cost_cents=self.content_cost_cents,
            )
        )

        monetisation_gain = max(0.0, demand[mechanism_id]) * max(revenue, active * 120)
        monetisation_cost = detection_cost + reputation_scale * max(0.0, demand[mechanism_id])
        candidates.append(
            FirmIntent(
                self.firm_id,
                FirmAction.ADJUST_MONETISATION,
                int(round(monetisation_gain - monetisation_cost)),
                target_game_id=update_game,
                mechanism=mechanism,
                intensity_delta=0.025 + 0.055 * self.exploration_tendency,
            )
        )
        candidates.append(
            FirmIntent(
                self.firm_id,
                FirmAction.BUY_RESEARCH,
                int(round(information_value - self.research_cost_cents)),
                committed_cost_cents=self.research_cost_cents,
            )
        )
        avoided_penalty = detection_cost * (0.35 + 0.45 * self.compliance_culture)
        candidates.append(
            FirmIntent(
                self.firm_id,
                FirmAction.INVEST_COMPLIANCE,
                int(round(avoided_penalty - self.compliance_cost_cents)),
                committed_cost_cents=self.compliance_cost_cents,
            )
        )
        acquisition_return = (
            active
            * average_revenue_per_active
            * observation.competitor_pressure_estimate
            * log1p(rank_pressure)
            * 0.12
        )
        candidates.append(
            FirmIntent(
                self.firm_id,
                FirmAction.ACQUIRE_USERS,
                int(round(acquisition_return - self.acquisition_cost_cents)),
                target_game_id=update_game,
                committed_cost_cents=self.acquisition_cost_cents,
            )
        )
        collaboration_value = int(
            round(
                max(0.0, 1.0 - observation.concentration_estimate)
                * active
                * average_revenue_per_active
                * 0.08
            )
        )
        candidates.append(
            FirmIntent(
                self.firm_id,
                FirmAction.PROPOSE_COLLABORATION,
                collaboration_value,
            )
        )
        collusion_margin = (
            observation.concentration_estimate * revenue * 0.11
            - detection_cost
            - revenue * self.ethics_weight * 0.18
        )
        candidates.append(
            FirmIntent(
                self.firm_id,
                FirmAction.PROPOSE_COLLUSION,
                int(round(collusion_margin)),
            )
        )
        evasion_saved_cost = self.compliance_cost_cents * (
            0.7 + observation.regulatory_uncertainty
        )
        evasion_value = (
            evasion_saved_cost
            - detection_cost * (1.1 - 0.5 * observation.regulatory_uncertainty)
            - reputation_scale * (self.ethics_weight + self.compliance_culture)
        )
        candidates.append(
            FirmIntent(
                self.firm_id,
                FirmAction.EVADE,
                int(round(evasion_value)),
                target_game_id=update_game,
            )
        )
        subsidy_value = int(
            round(
                observation.subsidy_success_probability
                * observation.expected_subsidy_cents
                - 0.08 * self.research_cost_cents
            )
        )
        candidates.append(
            FirmIntent(self.firm_id, FirmAction.APPLY_SUBSIDY, subsidy_value)
        )

        shocks = list(action_shocks or ())
        if shocks and len(shocks) != len(candidates):
            raise ValueError("one bounded-rationality shock is required per candidate")
        if not shocks:
            shocks = [0.0] * len(candidates)
        feasible = (
            True,
            cash >= self.content_cost_cents,
            True,
            cash >= self.research_cost_cents,
            cash >= self.compliance_cost_cents,
            cash >= self.acquisition_cost_cents,
            True,
            True,
            True,
            True,
        )
        if len(candidates) != len(FirmAction) or len(feasible) != len(candidates):
            raise AssertionError("firm candidate ordering must match FirmAction")
        scale = max(1_000.0, revenue * 0.01)
        selected_index = max(
            (index for index, allowed in enumerate(feasible) if allowed),
            key=lambda index: (
                candidates[index].perceived_value_cents
                + int(round(scale * self.exploration_tendency * shocks[index])),
                -index,
            ),
        )
        selected = candidates[selected_index]
        return FirmIntent(
            firm_id=selected.firm_id,
            action=selected.action,
            perceived_value_cents=selected.perceived_value_cents,
            target_game_id=selected.target_game_id,
            target_firm_id=selected.target_firm_id,
            mechanism=selected.mechanism,
            intensity_delta=selected.intensity_delta,
            committed_cost_cents=selected.committed_cost_cents,
            information_fingerprint=self._fingerprint(observation),
        )
