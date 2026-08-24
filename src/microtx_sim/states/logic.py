from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..agents.jurisdictions import (
    AuditEvidence,
    AuditIntent,
    FirmRiskSignal,
    RegulatorObservation,
    StateAgent,
)
from ..rng import CounterRNG, stable_stream_id


_AUDIT_RANDOM_FLOOR = stable_stream_id("audit-random-floor")
_AUDIT_DETECTION = stable_stream_id("audit-detection")
_AUDIT_FALSE_POSITIVE = stable_stream_id("audit-false-positive")


@dataclass(frozen=True, slots=True)
class ObservableFirmMetrics:
    firm_id: int
    complaint_rate: float
    reported_minor_harm_rate: float
    public_spend_anomaly: float
    past_public_detection: float
    signal_precision: float
    signal_age_days: int


@dataclass(frozen=True, slots=True)
class FirmComplianceTruth:
    """Kernel-only audit target; never passed to `StateAgent.select_audits`."""

    firm_id: int
    actual_breaches: tuple[str, ...]
    auditable_controls: tuple[str, ...]
    evasion_intensity: float
    maximum_fine_cents: int


@dataclass(frozen=True, slots=True)
class AuditResolution:
    intent: AuditIntent
    evidence: AuditEvidence
    fine_cents: int


class RegulationSystem:
    """Converts public signals to audits and hidden compliance to finite evidence."""

    __slots__ = ()

    @staticmethod
    def build_observation(
        *,
        tick: int,
        firms: tuple[ObservableFirmMetrics, ...],
        public_harm_index: float,
        treasury_pressure: float,
        sector_employment_estimate: float,
    ) -> RegulatorObservation:
        signals = tuple(
            FirmRiskSignal(
                firm_id=item.firm_id,
                complaint_rate=float(np.clip(item.complaint_rate, 0.0, 1.0)),
                minor_harm_signal=float(
                    np.clip(item.reported_minor_harm_rate, 0.0, 1.0)
                ),
                spend_anomaly=float(np.clip(item.public_spend_anomaly, 0.0, 1.0)),
                past_detection_signal=float(
                    np.clip(item.past_public_detection, 0.0, 1.0)
                ),
                evidence_precision=float(np.clip(item.signal_precision, 0.0, 1.0)),
                signal_age_days=item.signal_age_days,
            )
            for item in firms
        )
        mean_age = (
            float(np.mean([item.signal_age_days for item in firms])) if firms else 0.0
        )
        return RegulatorObservation(
            as_of=tick,
            firm_signals=signals,
            public_harm_index=float(np.clip(public_harm_index, 0.0, 1.0)),
            treasury_pressure=float(np.clip(treasury_pressure, 0.0, 1.0)),
            sector_employment_estimate=max(0.0, sector_employment_estimate),
            average_signal_age_days=mean_age,
        )

    def select(
        self,
        *,
        tick: int,
        state: StateAgent,
        observation: RegulatorObservation,
        rng: CounterRNG,
    ) -> tuple[AuditIntent, ...]:
        firm_ids = np.asarray(
            [item.firm_id for item in observation.firm_signals], dtype=np.int64
        )
        keys = rng.uniform(firm_ids, tick, _AUDIT_RANDOM_FLOOR, state.jurisdiction_id)
        return state.select_audits(observation, random_keys=keys)

    def resolve(
        self,
        *,
        tick: int,
        state: StateAgent,
        intents: tuple[AuditIntent, ...],
        truth_by_firm: dict[int, FirmComplianceTruth],
        rng: CounterRNG,
    ) -> tuple[AuditResolution, ...]:
        resolutions: list[AuditResolution] = []
        for intent in intents:
            truth = truth_by_firm[intent.firm_id]
            evasion = float(np.clip(truth.evasion_intensity, 0.0, 1.0))
            effective_sensitivity = state.audit_sensitivity * (1.0 - 0.65 * evasion)
            detected: list[str] = []
            for breach_index, breach in enumerate(truth.actual_breaches):
                detected_draw = bool(
                    rng.bernoulli(
                        np.asarray([truth.firm_id], dtype=np.int64),
                        tick,
                        _AUDIT_DETECTION,
                        state.jurisdiction_id * 1_000 + breach_index,
                        probability=effective_sensitivity,
                    )[0]
                )
                if detected_draw:
                    detected.append(breach)

            if not truth.actual_breaches and truth.auditable_controls:
                false_positive = bool(
                    rng.bernoulli(
                        np.asarray([truth.firm_id], dtype=np.int64),
                        tick,
                        _AUDIT_FALSE_POSITIVE,
                        state.jurisdiction_id,
                        probability=1.0 - state.audit_specificity,
                    )[0]
                )
                if false_positive:
                    detected.append("inconclusive_control_failure")

            strength = float(
                np.clip(
                    0.45
                    + 0.45 * effective_sensitivity
                    + 0.10 * len(truth.auditable_controls) / 8.0,
                    0.0,
                    1.0,
                )
            )
            evidence = AuditEvidence(
                jurisdiction_id=state.jurisdiction_id,
                firm_id=truth.firm_id,
                detected_breaches=tuple(detected),
                tested_controls=truth.auditable_controls,
                evidence_strength=strength,
                false_positive_probability=1.0 - state.audit_specificity,
                tick=tick,
            )
            fine_share = 0.0 if not detected else min(1.0, 0.25 + 0.15 * len(detected))
            fine_cents = int(round(truth.maximum_fine_cents * strength * fine_share))
            resolution = AuditResolution(intent=intent, evidence=evidence, fine_cents=fine_cents)
            resolutions.append(resolution)
            state.observe_audit(evidence)

        total_cost = len(resolutions) * state.state.inspection_cost_cents
        state.state.audit_budget_cents = max(
            0, state.state.audit_budget_cents - total_cost
        )
        state.state.treasury_cents = max(0, state.state.treasury_cents - total_cost)
        return tuple(resolutions)

