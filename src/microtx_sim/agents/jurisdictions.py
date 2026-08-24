from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True, slots=True)
class RegulationRules:
    odds_disclosure_required: bool
    real_money_price_required: bool
    parental_authorisation_required: bool
    direct_exhortation_to_minors_banned: bool
    paid_random_rewards_restricted: bool
    cooling_off_days: int
    minor_monthly_cap_cents: int | None
    maximum_power_sale_intensity: float

    def __post_init__(self) -> None:
        if self.cooling_off_days < 0:
            raise ValueError("cooling-off period cannot be negative")
        if self.minor_monthly_cap_cents is not None and self.minor_monthly_cap_cents < 0:
            raise ValueError("minor spending cap cannot be negative")
        if not 0.0 <= self.maximum_power_sale_intensity <= 1.0:
            raise ValueError("power-sale limit must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FirmRiskSignal:
    firm_id: int
    complaint_rate: float
    minor_harm_signal: float
    spend_anomaly: float
    past_detection_signal: float
    evidence_precision: float
    signal_age_days: int

    def __post_init__(self) -> None:
        if self.signal_age_days < 0:
            raise ValueError("signal age cannot be negative")
        for value in (
            self.complaint_rate,
            self.minor_harm_signal,
            self.spend_anomaly,
            self.past_detection_signal,
            self.evidence_precision,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("risk signals must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RegulatorObservation:
    as_of: int
    firm_signals: tuple[FirmRiskSignal, ...]
    public_harm_index: float
    treasury_pressure: float
    sector_employment_estimate: float
    average_signal_age_days: float


@dataclass(frozen=True, slots=True)
class AuditIntent:
    jurisdiction_id: int
    firm_id: int
    risk_score: float
    random_floor_selection: bool
    information_as_of: int


@dataclass(frozen=True, slots=True)
class AuditEvidence:
    jurisdiction_id: int
    firm_id: int
    detected_breaches: tuple[str, ...]
    tested_controls: tuple[str, ...]
    evidence_strength: float
    false_positive_probability: float
    tick: int


@dataclass(frozen=True, slots=True)
class SubsidyApplicationView:
    firm_id: int
    requested_cents: int
    verified_quality: float
    verified_design_safety_score: float
    verified_accessibility: float
    jobs_estimate: int
    evidence_age_days: int
    submitted_tick: int = 0
    eligible_jurisdictions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.firm_id < 0 or self.requested_cents < 0 or self.submitted_tick < 0:
            raise ValueError("subsidy application identifiers and money must be non-negative")
        if self.jobs_estimate < 0 or self.evidence_age_days < 0:
            raise ValueError("subsidy evidence metadata must be non-negative")
        for value in (
            self.verified_quality,
            self.verified_design_safety_score,
            self.verified_accessibility,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("verified subsidy scores must be in [0, 1]")
        if any(value < 0 for value in self.eligible_jurisdictions):
            raise ValueError("eligible jurisdiction ids must be non-negative")
        if len(set(self.eligible_jurisdictions)) != len(self.eligible_jurisdictions):
            raise ValueError("eligible jurisdictions must be unique")


@dataclass(frozen=True, slots=True)
class SubsidyIntent:
    jurisdiction_id: int
    firm_id: int
    award_cents: int
    score: float
    conditional: bool


@dataclass(slots=True)
class RegulatorPrivateState:
    treasury_cents: int
    audit_budget_cents: int
    subsidy_budget_cents: int
    audit_capacity_per_cycle: int
    inspection_cost_cents: int
    compliance_alpha: dict[int, float] = field(default_factory=dict)
    compliance_beta: dict[int, float] = field(default_factory=dict)


@dataclass(slots=True)
class StateAgent:
    jurisdiction_id: int
    code: str
    rules: RegulationRules
    state: RegulatorPrivateState
    harm_priority: float
    minor_priority: float
    fiscal_priority: float
    industry_priority: float
    random_audit_fraction: float
    audit_sensitivity: float
    audit_specificity: float
    subsidy_quality_weight: float
    subsidy_safe_revenue_weight: float
    subsidy_accessibility_weight: float

    def __post_init__(self) -> None:
        bounded = (
            self.harm_priority,
            self.minor_priority,
            self.fiscal_priority,
            self.industry_priority,
            self.random_audit_fraction,
            self.audit_sensitivity,
            self.audit_specificity,
            self.subsidy_quality_weight,
            self.subsidy_safe_revenue_weight,
            self.subsidy_accessibility_weight,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("state preferences and accuracies must be in [0, 1]")

    def _risk_score(self, signal: FirmRiskSignal) -> float:
        recency = 1.0 / (1.0 + signal.signal_age_days / 30.0)
        raw = (
            self.harm_priority * (0.45 * signal.spend_anomaly + 0.25 * signal.complaint_rate)
            + self.minor_priority * 0.55 * signal.minor_harm_signal
            + 0.25 * signal.past_detection_signal
        )
        return raw * (0.35 + 0.65 * signal.evidence_precision) * recency

    def select_audits(
        self,
        observation: RegulatorObservation,
        *,
        random_keys: Iterable[float],
    ) -> tuple[AuditIntent, ...]:
        signals = tuple(observation.firm_signals)
        if not signals or self.state.audit_capacity_per_cycle <= 0:
            return ()
        affordable = self.state.audit_budget_cents // max(1, self.state.inspection_cost_cents)
        capacity = min(len(signals), self.state.audit_capacity_per_cycle, affordable)
        if capacity <= 0:
            return ()
        keys = list(random_keys)
        if len(keys) != len(signals):
            raise ValueError("one random audit key is required per observable firm")

        random_count = min(capacity, int(round(capacity * self.random_audit_fraction)))
        if self.random_audit_fraction > 0.0 and random_count == 0:
            random_count = 1
        risk_count = capacity - random_count
        scored = [(self._risk_score(signal), signal) for signal in signals]
        risk_selected = sorted(scored, key=lambda item: (-item[0], item[1].firm_id))[
            :risk_count
        ]
        selected_ids = {signal.firm_id for _, signal in risk_selected}
        remaining = [
            (keys[index], self._risk_score(signal), signal)
            for index, signal in enumerate(signals)
            if signal.firm_id not in selected_ids
        ]
        random_selected = sorted(remaining, key=lambda item: (item[0], item[2].firm_id))[
            :random_count
        ]
        intents = [
            AuditIntent(
                self.jurisdiction_id,
                signal.firm_id,
                score,
                False,
                observation.as_of,
            )
            for score, signal in risk_selected
        ]
        intents.extend(
            AuditIntent(
                self.jurisdiction_id,
                signal.firm_id,
                score,
                True,
                observation.as_of,
            )
            for _, score, signal in random_selected
        )
        return tuple(sorted(intents, key=lambda intent: intent.firm_id))

    def observe_audit(self, evidence: AuditEvidence) -> None:
        if evidence.jurisdiction_id != self.jurisdiction_id:
            raise ValueError("audit evidence belongs to a different jurisdiction")
        firm = evidence.firm_id
        self.state.compliance_alpha.setdefault(firm, 1.0)
        self.state.compliance_beta.setdefault(firm, 1.0)
        weight = max(0.0, min(1.0, evidence.evidence_strength))
        if evidence.detected_breaches:
            self.state.compliance_beta[firm] += weight
        else:
            self.state.compliance_alpha[firm] += weight

    def award_subsidies(
        self,
        applications: Iterable[SubsidyApplicationView],
    ) -> tuple[SubsidyIntent, ...]:
        scored: list[tuple[float, SubsidyApplicationView]] = []
        for application in applications:
            recency = 1.0 / (1.0 + application.evidence_age_days / 90.0)
            score = recency * (
                self.subsidy_quality_weight * application.verified_quality
                + self.subsidy_safe_revenue_weight
                * application.verified_design_safety_score
                + self.subsidy_accessibility_weight * application.verified_accessibility
                + self.industry_priority * min(1.0, application.jobs_estimate / 100.0)
            )
            scored.append((score, application))
        budget = min(self.state.subsidy_budget_cents, self.state.treasury_cents)
        awards: list[SubsidyIntent] = []
        for score, application in sorted(
            scored, key=lambda item: (-item[0], item[1].firm_id)
        ):
            if budget <= 0:
                break
            award = min(application.requested_cents, budget)
            if award <= 0:
                continue
            awards.append(
                SubsidyIntent(
                    self.jurisdiction_id,
                    application.firm_id,
                    award,
                    score,
                    conditional=True,
                )
            )
            budget -= award
        return tuple(awards)
