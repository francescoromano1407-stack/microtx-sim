"""Fail-closed registry for the retrospective synthetic policy design.

The seven policy scenarios predate any empirical preregistration.  This module
therefore records their exact atomic-factor matrix without promoting it to a
campaign-ready causal design.  Canonical registry construction rejects any
factor drift under a canonical scenario identifier; descriptive assessment is
available separately so custom scenario batches can be recorded honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from typing import Sequence

from ..domain.monetisation import MonetisationVector
from .scenarios import ScenarioId, ScenarioSpec, required_scenarios


CAUSAL_DESIGN_SCHEMA_VERSION = "1.0"


class CausalDesignStatus(str, Enum):
    """Scientific status of the registered policy design."""

    RETROSPECTIVE_SYNTHETIC = "RETROSPECTIVE_SYNTHETIC"


class ContrastClassification(str, Enum):
    """Classification determined only by the atomic factor differences."""

    IDENTITY = "identity"
    SINGLE_FACTOR = "single_factor"
    BUNDLE = "bundle"


class ContrastStructuralScope(str, Enum):
    """Whether a contrast changes mechanics, financing, or both."""

    IDENTITY = "identity"
    MECHANICS_ONLY = "mechanics_only"
    FINANCING_ONLY = "financing_only"
    MIXED = "mixed"


class ContrastRole(str, Enum):
    """Non-preregistered reporting roles attached to directed contrasts."""

    EXHAUSTIVE_PAIRWISE_DIAGNOSTIC = "exhaustive_pairwise_diagnostic"
    REPORTED_EFFECT_VS_SAFE = "reported_effect_vs_safe"
    DECLARED_CATALOGUE_CHECK = "declared_catalogue_check"


class CampaignValidationError(RuntimeError):
    """Raised when a retrospective design is used as campaign-ready."""

    def __init__(self, blockers: tuple[str, ...]) -> None:
        self.blockers = blockers
        super().__init__(
            "causal design is not campaign-ready: " + ", ".join(blockers)
        )


class CausalFactor(str, Enum):
    """The canonical fourteen mechanic and three scenario-level factors."""

    DIRECT_PRICE_CENTS = "direct_price_cents"
    OPAQUE_VIRTUAL_CURRENCY = "opaque_virtual_currency"
    PAID_RANDOM_REWARDS = "paid_random_rewards"
    PROGRESSION_GATES = "progression_gates"
    TIME_LIMITED_OFFERS = "time_limited_offers"
    DAILY_STREAK_PRESSURE = "daily_streak_pressure"
    PAY_TO_PROGRESS = "pay_to_progress"
    PAY_TO_WIN = "pay_to_win"
    SOCIAL_GUILD_PRESSURE = "social_guild_pressure"
    PURCHASE_FRICTION = "purchase_friction"
    SPENDING_CAP_CENTS = "spending_cap_cents"
    COOLING_OFF_HOURS = "cooling_off_hours"
    REAL_CURRENCY_PRICE_DISPLAY = "real_currency_price_display"
    PERSONALIZED_OFFERS = "personalized_offers"
    FIXED_ACCESS_PRICE_CENTS = "fixed_access_price_cents"
    SUBSCRIPTION_PRICE_CENTS = "subscription_price_cents"
    EPGC_ENABLED = "epgc_enabled"


class FactorSource(str, Enum):
    """Location from which an atomic factor is extracted."""

    MONETISATION_VECTOR = "monetisation_vector"
    SCENARIO = "scenario"


class FactorValueKind(str, Enum):
    """Primitive value contract for an atomic causal factor."""

    NONNEGATIVE_INTEGER = "nonnegative_integer"
    OPTIONAL_NONNEGATIVE_INTEGER = "optional_nonnegative_integer"
    UNIT_INTERVAL_REAL = "unit_interval_real"
    BOOLEAN = "boolean"


FactorValue = int | float | bool | None


@dataclass(frozen=True, slots=True)
class AtomicFactorSpec:
    """Name, source, and primitive type of one atomic design factor."""

    factor: CausalFactor
    source: FactorSource
    value_kind: FactorValueKind

    def __post_init__(self) -> None:
        if type(self.factor) is not CausalFactor:
            raise TypeError("factor must be a CausalFactor")
        if type(self.source) is not FactorSource:
            raise TypeError("source must be a FactorSource")
        if type(self.value_kind) is not FactorValueKind:
            raise TypeError("value_kind must be a FactorValueKind")

    def snapshot(self) -> dict[str, str]:
        return {
            "name": self.factor.value,
            "source": self.source.value,
            "value_kind": self.value_kind.value,
        }


_REAL_MECHANIC_FACTORS = frozenset(
    {
        CausalFactor.OPAQUE_VIRTUAL_CURRENCY,
        CausalFactor.PAID_RANDOM_REWARDS,
        CausalFactor.PROGRESSION_GATES,
        CausalFactor.TIME_LIMITED_OFFERS,
        CausalFactor.DAILY_STREAK_PRESSURE,
        CausalFactor.PAY_TO_PROGRESS,
        CausalFactor.PAY_TO_WIN,
        CausalFactor.SOCIAL_GUILD_PRESSURE,
        CausalFactor.PURCHASE_FRICTION,
    }
)
_BOOLEAN_MECHANIC_FACTORS = frozenset(
    {
        CausalFactor.REAL_CURRENCY_PRICE_DISPLAY,
        CausalFactor.PERSONALIZED_OFFERS,
    }
)
_SCENARIO_FACTORS = (
    CausalFactor.FIXED_ACCESS_PRICE_CENTS,
    CausalFactor.SUBSCRIPTION_PRICE_CENTS,
    CausalFactor.EPGC_ENABLED,
)
_MECHANIC_FACTORS = tuple(CausalFactor)[:14]


def _factor_kind(factor: CausalFactor) -> FactorValueKind:
    if factor in _REAL_MECHANIC_FACTORS:
        return FactorValueKind.UNIT_INTERVAL_REAL
    if factor is CausalFactor.SPENDING_CAP_CENTS:
        return FactorValueKind.OPTIONAL_NONNEGATIVE_INTEGER
    if factor in _BOOLEAN_MECHANIC_FACTORS or factor is CausalFactor.EPGC_ENABLED:
        return FactorValueKind.BOOLEAN
    return FactorValueKind.NONNEGATIVE_INTEGER


ATOMIC_FACTOR_SPECS = tuple(
    AtomicFactorSpec(
        factor=factor,
        source=(
            FactorSource.SCENARIO
            if factor in _SCENARIO_FACTORS
            else FactorSource.MONETISATION_VECTOR
        ),
        value_kind=_factor_kind(factor),
    )
    for factor in CausalFactor
)
ATOMIC_FACTOR_NAMES = tuple(spec.factor.value for spec in ATOMIC_FACTOR_SPECS)
_FACTOR_SPEC_BY_NAME = {spec.factor: spec for spec in ATOMIC_FACTOR_SPECS}
_FACTOR_INDEX = {
    spec.factor: index for index, spec in enumerate(ATOMIC_FACTOR_SPECS)
}


_DOMAIN_MECHANIC_FIELDS = tuple(item.name for item in fields(MonetisationVector))
if _DOMAIN_MECHANIC_FIELDS != tuple(factor.value for factor in _MECHANIC_FACTORS):
    raise RuntimeError(
        "causal factor registry does not exactly match MonetisationVector fields"
    )
_DOMAIN_SCENARIO_FIELDS = tuple(item.name for item in fields(ScenarioSpec))
if _DOMAIN_SCENARIO_FIELDS != (
    "scenario_id",
    "label",
    "mechanics",
    "fixed_access_price_cents",
    "subscription_price_cents",
    "epgc_enabled",
    "description",
):
    raise RuntimeError(
        "causal factor registry must be reviewed for changed ScenarioSpec fields"
    )
if len(ATOMIC_FACTOR_SPECS) != 17:
    raise RuntimeError("causal factor registry must contain exactly 17 factors")


@dataclass(frozen=True, slots=True)
class ScenarioFactorVector:
    """One scenario's values aligned to :data:`ATOMIC_FACTOR_SPECS`."""

    scenario_id: ScenarioId
    values: tuple[FactorValue, ...]

    def __post_init__(self) -> None:
        if type(self.scenario_id) is not ScenarioId:
            raise TypeError("scenario_id must be a ScenarioId")
        if type(self.values) is not tuple:
            raise TypeError("values must be a tuple")
        if len(self.values) != len(ATOMIC_FACTOR_SPECS):
            raise ValueError("scenario vector must contain exactly 17 factor values")
        for spec, value in zip(ATOMIC_FACTOR_SPECS, self.values):
            _validate_factor_value(spec, value)

    def factor_value(self, factor: CausalFactor) -> FactorValue:
        if type(factor) is not CausalFactor:
            raise TypeError("factor must be a CausalFactor")
        return self.values[_FACTOR_INDEX[factor]]

    def snapshot(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id.value,
            "factors": {
                spec.factor.value: value
                for spec, value in zip(ATOMIC_FACTOR_SPECS, self.values)
            },
        }


@dataclass(frozen=True, slots=True)
class FactorDifference:
    """One exact directed factor change in a scenario contrast."""

    factor: CausalFactor
    reference_value: FactorValue
    comparison_value: FactorValue

    def __post_init__(self) -> None:
        if type(self.factor) is not CausalFactor:
            raise TypeError("factor must be a CausalFactor")
        spec = _FACTOR_SPEC_BY_NAME[self.factor]
        _validate_factor_value(spec, self.reference_value)
        _validate_factor_value(spec, self.comparison_value)
        if _factor_values_equal(
            self.reference_value,
            self.comparison_value,
        ):
            raise ValueError("factor difference values must differ")

    def snapshot(self) -> dict[str, object]:
        return {
            "factor": self.factor.value,
            "reference_value": self.reference_value,
            "comparison_value": self.comparison_value,
        }


@dataclass(frozen=True, slots=True)
class ContrastSpec:
    """Typed directed contrast with classification derived from exact diffs."""

    reference_scenario_id: ScenarioId
    comparison_scenario_id: ScenarioId
    classification: ContrastClassification
    factor_differences: tuple[FactorDifference, ...]

    def __post_init__(self) -> None:
        if type(self.reference_scenario_id) is not ScenarioId:
            raise TypeError("reference_scenario_id must be a ScenarioId")
        if type(self.comparison_scenario_id) is not ScenarioId:
            raise TypeError("comparison_scenario_id must be a ScenarioId")
        if type(self.classification) is not ContrastClassification:
            raise TypeError("classification must be a ContrastClassification")
        if type(self.factor_differences) is not tuple or any(
            type(item) is not FactorDifference for item in self.factor_differences
        ):
            raise TypeError(
                "factor_differences must be a tuple of FactorDifference instances"
            )
        factors = tuple(item.factor for item in self.factor_differences)
        if len(set(factors)) != len(factors):
            raise ValueError("contrast factor differences must be unique")
        if tuple(sorted(factors, key=_FACTOR_INDEX.__getitem__)) != factors:
            raise ValueError("contrast factor differences must use canonical order")
        expected = _contrast_classification(len(self.factor_differences))
        if self.classification is not expected:
            raise ValueError(
                "contrast classification does not match its factor differences"
            )

    @property
    def contrast_id(self) -> str:
        return (
            f"{self.reference_scenario_id.value}"
            f"__to__{self.comparison_scenario_id.value}"
        )

    @property
    def differing_factors(self) -> tuple[CausalFactor, ...]:
        return tuple(item.factor for item in self.factor_differences)

    @property
    def structural_scope(self) -> ContrastStructuralScope:
        factors = frozenset(self.differing_factors)
        if not factors:
            return ContrastStructuralScope.IDENTITY
        if factors.issubset(_MECHANIC_FACTOR_SET):
            return ContrastStructuralScope.MECHANICS_ONLY
        if factors.issubset(_SCENARIO_FACTOR_SET):
            return ContrastStructuralScope.FINANCING_ONLY
        return ContrastStructuralScope.MIXED

    @property
    def roles(self) -> tuple[ContrastRole, ...]:
        roles = [ContrastRole.EXHAUSTIVE_PAIRWISE_DIAGNOSTIC]
        pair = (self.reference_scenario_id, self.comparison_scenario_id)
        if self.reference_scenario_id is ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION:
            roles.append(ContrastRole.REPORTED_EFFECT_VS_SAFE)
        if pair in _DECLARED_CATALOGUE_PAIRS:
            roles.append(ContrastRole.DECLARED_CATALOGUE_CHECK)
        return tuple(roles)

    def snapshot(self) -> dict[str, object]:
        return {
            "contrast_id": self.contrast_id,
            "reference_scenario_id": self.reference_scenario_id.value,
            "comparison_scenario_id": self.comparison_scenario_id.value,
            "classification": self.classification.value,
            "structural_scope": self.structural_scope.value,
            "roles": [role.value for role in self.roles],
            "planned_estimand": False,
            "preregistered": False,
            "factor_differences": [
                item.snapshot() for item in self.factor_differences
            ],
        }


@dataclass(frozen=True, slots=True)
class ScenarioMatrixMismatch:
    """Exact canonical-to-observed factor drift for one scenario identifier."""

    scenario_id: ScenarioId
    factor_differences: tuple[FactorDifference, ...]

    def __post_init__(self) -> None:
        if type(self.scenario_id) is not ScenarioId:
            raise TypeError("scenario_id must be a ScenarioId")
        if type(self.factor_differences) is not tuple:
            raise TypeError("factor_differences must be a tuple")
        if not self.factor_differences or any(
            type(item) is not FactorDifference for item in self.factor_differences
        ):
            raise ValueError("a scenario mismatch needs factor differences")

    def snapshot(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id.value,
            "factor_differences": [
                item.snapshot() for item in self.factor_differences
            ],
        }


class CausalDesignValidationError(ValueError):
    """Base class for invalid canonical causal-design declarations."""


class CausalFactorMatrixValidationError(CausalDesignValidationError):
    """Raised when canonical scenario IDs carry changed atomic factors."""

    def __init__(
        self,
        mismatches: tuple[ScenarioMatrixMismatch, ...],
    ) -> None:
        self.mismatches = mismatches
        changed = [
            f"{item.scenario_id.value}:"
            + ",".join(
                difference.factor.value
                for difference in item.factor_differences
            )
            for item in mismatches
        ]
        super().__init__(
            "scenario factor matrix is not canonical; changed="
            + ";".join(changed)
        )


_BASE_CAMPAIGN_BLOCKERS = (
    "retrospective_synthetic_design",
    "causal_design_not_preregistered",
    "empirical_calibration_required",
)
_NONCANONICAL_BLOCKER = "scenario_factor_matrix_not_canonical"

_DECLARED_CATALOGUE_PAIRS = (
    (ScenarioId.BASELINE_F2P, ScenarioId.TRANSPARENT_DIRECT_PRICE),
    (ScenarioId.BASELINE_F2P, ScenarioId.NO_RANDOM_REWARDS),
    (ScenarioId.BASELINE_F2P, ScenarioId.NO_TIME_LIMITED_PRESSURE),
    (ScenarioId.BASELINE_F2P, ScenarioId.SPENDING_CAP_COOLING_OFF),
    (ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION, ScenarioId.EPGC),
)

_MECHANIC_FACTOR_SET = frozenset(_MECHANIC_FACTORS)
_SCENARIO_FACTOR_SET = frozenset(_SCENARIO_FACTORS)


class _DesignView:
    """Shared read-only projections for canonical and observed designs."""

    __slots__ = ()

    scenario_matrix: tuple[ScenarioFactorVector, ...]
    contrasts: tuple[ContrastSpec, ...]

    def factor_value(
        self,
        scenario_id: ScenarioId,
        factor: CausalFactor,
    ) -> FactorValue:
        if type(scenario_id) is not ScenarioId:
            raise TypeError("scenario_id must be a ScenarioId")
        if type(factor) is not CausalFactor:
            raise TypeError("factor must be a CausalFactor")
        row = next(
            item for item in self.scenario_matrix if item.scenario_id is scenario_id
        )
        return row.factor_value(factor)

    def contrast(
        self,
        reference_scenario_id: ScenarioId,
        comparison_scenario_id: ScenarioId,
    ) -> ContrastSpec:
        if type(reference_scenario_id) is not ScenarioId:
            raise TypeError("reference_scenario_id must be a ScenarioId")
        if type(comparison_scenario_id) is not ScenarioId:
            raise TypeError("comparison_scenario_id must be a ScenarioId")
        return next(
            item
            for item in self.contrasts
            if item.reference_scenario_id is reference_scenario_id
            and item.comparison_scenario_id is comparison_scenario_id
        )

    def scenario_matrix_snapshot(self) -> list[dict[str, object]]:
        return [item.snapshot() for item in self.scenario_matrix]

    def scenario_matrix_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": CAUSAL_DESIGN_SCHEMA_VERSION,
                "factor_names": list(ATOMIC_FACTOR_NAMES),
                "scenario_matrix": self.scenario_matrix_snapshot(),
            }
        )

    def contrasts_snapshot(self) -> list[dict[str, object]]:
        return [item.snapshot() for item in self.contrasts]

    def contrasts_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": CAUSAL_DESIGN_SCHEMA_VERSION,
                "factor_names": list(ATOMIC_FACTOR_NAMES),
                "contrast_scope": "exhaustive_directed_pairwise_diagnostics",
                "contrasts": self.contrasts_snapshot(),
            }
        )

    def design_snapshot(self) -> dict[str, object]:
        """Return the shared observed-design payload, without assessment data."""

        return _design_snapshot(self)

    def design_sha256(self) -> str:
        return _canonical_sha256(self.design_snapshot())

    @property
    def reported_effect_vs_safe_contrast_ids(self) -> tuple[str, ...]:
        safe = ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION.value
        return tuple(f"{safe}__to__{scenario_id.value}" for scenario_id in ScenarioId)

    @property
    def declared_catalogue_contrast_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{reference.value}__to__{comparison.value}"
            for reference, comparison in _DECLARED_CATALOGUE_PAIRS
        )

    def validate_for_campaign(self) -> None:
        """Fail closed because this design is retrospective and synthetic."""

        blockers = getattr(self, "campaign_blockers")
        raise CampaignValidationError(blockers)


@dataclass(frozen=True, slots=True)
class CausalDesignRegistry(_DesignView):
    """The exact canonical seven-scenario retrospective synthetic design."""

    scenarios: tuple[ScenarioSpec, ...]
    status: CausalDesignStatus = field(
        default=CausalDesignStatus.RETROSPECTIVE_SYNTHETIC,
        init=False,
    )
    preregistered: bool = field(default=False, init=False)
    campaign_ready: bool = field(default=False, init=False)
    canonical_match: bool = field(default=True, init=False)
    campaign_blockers: tuple[str, ...] = field(
        default=_BASE_CAMPAIGN_BLOCKERS,
        init=False,
    )
    scenario_matrix: tuple[ScenarioFactorVector, ...] = field(
        init=False,
        repr=False,
    )
    contrasts: tuple[ContrastSpec, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        scenarios, observed = _validated_scenario_matrix(self.scenarios)
        canonical = _canonical_scenario_matrix()
        mismatches = _matrix_mismatches(canonical, observed)
        if mismatches:
            raise CausalFactorMatrixValidationError(mismatches)
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "scenario_matrix", observed)
        object.__setattr__(self, "contrasts", _build_contrasts(observed))

    def snapshot(self) -> dict[str, object]:
        return self.design_snapshot()

    def snapshot_sha256(self) -> str:
        return self.design_sha256()

    def manifest_payload(self, *, run_input_sha256: str) -> dict[str, object]:
        payload = self.snapshot()
        payload["design_sha256"] = self.design_sha256()
        payload["run_input_sha256"] = _validate_sha256(
            run_input_sha256,
            name="run_input_sha256",
        )
        return payload


@dataclass(frozen=True, slots=True)
class CausalDesignAssessment(_DesignView):
    """Descriptive assessment that does not mislabel a custom matrix."""

    scenarios: tuple[ScenarioSpec, ...]
    status: CausalDesignStatus = field(
        default=CausalDesignStatus.RETROSPECTIVE_SYNTHETIC,
        init=False,
    )
    preregistered: bool = field(default=False, init=False)
    campaign_ready: bool = field(default=False, init=False)
    canonical_match: bool = field(init=False)
    campaign_blockers: tuple[str, ...] = field(init=False)
    scenario_matrix: tuple[ScenarioFactorVector, ...] = field(
        init=False,
        repr=False,
    )
    contrasts: tuple[ContrastSpec, ...] = field(init=False, repr=False)
    canonical_mismatches: tuple[ScenarioMatrixMismatch, ...] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        scenarios, observed = _validated_scenario_matrix(self.scenarios)
        canonical = _canonical_scenario_matrix()
        mismatches = _matrix_mismatches(canonical, observed)
        canonical_match = not mismatches
        blockers = _BASE_CAMPAIGN_BLOCKERS + (
            () if canonical_match else (_NONCANONICAL_BLOCKER,)
        )
        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "scenario_matrix", observed)
        object.__setattr__(self, "contrasts", _build_contrasts(observed))
        object.__setattr__(self, "canonical_match", canonical_match)
        object.__setattr__(self, "campaign_blockers", blockers)
        object.__setattr__(self, "canonical_mismatches", mismatches)

    def snapshot(self) -> dict[str, object]:
        payload = _design_snapshot(self)
        payload["canonical_scenario_matrix_sha256"] = (
            build_causal_design_registry(
                required_scenarios()
            ).scenario_matrix_sha256()
        )
        payload["canonical_mismatches"] = [
            item.snapshot() for item in self.canonical_mismatches
        ]
        return payload

    def snapshot_sha256(self) -> str:
        return _canonical_sha256(self.snapshot())

    def manifest_payload(self, *, run_input_sha256: str) -> dict[str, object]:
        payload = self.snapshot()
        payload["design_sha256"] = self.design_sha256()
        payload["assessment_sha256"] = self.snapshot_sha256()
        payload["canonical_design_sha256"] = build_causal_design_registry(
            required_scenarios()
        ).design_sha256()
        payload["run_input_sha256"] = _validate_sha256(
            run_input_sha256,
            name="run_input_sha256",
        )
        return payload


def build_causal_design_registry(
    scenarios: Sequence[ScenarioSpec],
) -> CausalDesignRegistry:
    """Build the canonical registry or reject any atomic-factor drift."""

    return CausalDesignRegistry(tuple(scenarios))


def assess_causal_design(
    scenarios: Sequence[ScenarioSpec],
) -> CausalDesignAssessment:
    """Describe canonical or custom factors while retaining campaign blockers."""

    return CausalDesignAssessment(tuple(scenarios))


def _validated_scenario_matrix(
    scenarios: Sequence[ScenarioSpec],
) -> tuple[tuple[ScenarioSpec, ...], tuple[ScenarioFactorVector, ...]]:
    selected = tuple(scenarios)
    if len(selected) != len(ScenarioId):
        raise ValueError("causal design requires exactly seven scenarios")
    if any(type(item) is not ScenarioSpec for item in selected):
        raise TypeError("causal design scenarios must be exact ScenarioSpec instances")
    by_id: dict[ScenarioId, ScenarioSpec] = {}
    for scenario in selected:
        if scenario.scenario_id in by_id:
            raise ValueError("causal design scenario ids must be unique")
        by_id[scenario.scenario_id] = scenario
    if set(by_id) != set(ScenarioId):
        missing = sorted(item.value for item in set(ScenarioId) - set(by_id))
        raise ValueError(f"causal design is missing canonical scenario ids: {missing}")
    ordered = tuple(by_id[scenario_id] for scenario_id in ScenarioId)
    matrix = tuple(_scenario_factor_vector(scenario) for scenario in ordered)
    return ordered, matrix


def _scenario_factor_vector(scenario: ScenarioSpec) -> ScenarioFactorVector:
    mechanic_values = tuple(
        getattr(scenario.mechanics, factor.value) for factor in _MECHANIC_FACTORS
    )
    scenario_values: tuple[FactorValue, ...] = (
        scenario.fixed_access_price_cents,
        scenario.subscription_price_cents,
        scenario.epgc_enabled,
    )
    return ScenarioFactorVector(
        scenario_id=scenario.scenario_id,
        values=mechanic_values + scenario_values,
    )


def _canonical_scenario_matrix() -> tuple[ScenarioFactorVector, ...]:
    _, matrix = _validated_scenario_matrix(required_scenarios())
    return matrix


def _factor_differences(
    reference: ScenarioFactorVector,
    comparison: ScenarioFactorVector,
) -> tuple[FactorDifference, ...]:
    return tuple(
        FactorDifference(
            factor=spec.factor,
            reference_value=reference_value,
            comparison_value=comparison_value,
        )
        for spec, reference_value, comparison_value in zip(
            ATOMIC_FACTOR_SPECS,
            reference.values,
            comparison.values,
        )
        if not _factor_values_equal(reference_value, comparison_value)
    )


def _build_contrasts(
    matrix: tuple[ScenarioFactorVector, ...],
) -> tuple[ContrastSpec, ...]:
    return tuple(
        ContrastSpec(
            reference_scenario_id=reference.scenario_id,
            comparison_scenario_id=comparison.scenario_id,
            classification=_contrast_classification(len(differences)),
            factor_differences=differences,
        )
        for reference in matrix
        for comparison in matrix
        for differences in (_factor_differences(reference, comparison),)
    )


def _contrast_classification(difference_count: int) -> ContrastClassification:
    if difference_count == 0:
        return ContrastClassification.IDENTITY
    if difference_count == 1:
        return ContrastClassification.SINGLE_FACTOR
    return ContrastClassification.BUNDLE


def _matrix_mismatches(
    canonical: tuple[ScenarioFactorVector, ...],
    observed: tuple[ScenarioFactorVector, ...],
) -> tuple[ScenarioMatrixMismatch, ...]:
    if tuple(item.scenario_id for item in canonical) != tuple(
        item.scenario_id for item in observed
    ):
        raise ValueError("canonical and observed scenario matrices are not aligned")
    return tuple(
        ScenarioMatrixMismatch(
            scenario_id=canonical_row.scenario_id,
            factor_differences=differences,
        )
        for canonical_row, observed_row in zip(canonical, observed)
        for differences in (_factor_differences(canonical_row, observed_row),)
        if differences
    )


def _design_snapshot(design: _DesignView) -> dict[str, object]:
    status = getattr(design, "status")
    campaign_blockers = getattr(design, "campaign_blockers")
    return {
        "schema_version": CAUSAL_DESIGN_SCHEMA_VERSION,
        "status": status.value,
        "preregistered": bool(getattr(design, "preregistered")),
        "campaign_ready": bool(getattr(design, "campaign_ready")),
        "canonical_match": bool(getattr(design, "canonical_match")),
        "campaign_blockers": list(campaign_blockers),
        "factor_names": list(ATOMIC_FACTOR_NAMES),
        "factor_specs": [spec.snapshot() for spec in ATOMIC_FACTOR_SPECS],
        "scenario_matrix": design.scenario_matrix_snapshot(),
        "scenario_matrix_sha256": design.scenario_matrix_sha256(),
        "contrast_scope": "exhaustive_directed_pairwise_diagnostics",
        "contrast_count": len(design.contrasts),
        "planned_estimands": False,
        "preregistered_estimands": False,
        "reported_effect_vs_safe_contrast_ids": list(
            design.reported_effect_vs_safe_contrast_ids
        ),
        "declared_catalogue_contrast_ids": list(
            design.declared_catalogue_contrast_ids
        ),
        "contrasts": design.contrasts_snapshot(),
        "contrasts_sha256": design.contrasts_sha256(),
    }


def _validate_factor_value(spec: AtomicFactorSpec, value: FactorValue) -> None:
    kind = spec.value_kind
    if kind is FactorValueKind.BOOLEAN:
        if type(value) is not bool:
            raise TypeError(f"{spec.factor.value} must be a boolean")
        return
    if kind is FactorValueKind.UNIT_INTERVAL_REAL:
        if type(value) is not float:
            raise TypeError(f"{spec.factor.value} must be a float")
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{spec.factor.value} must be finite and in [0, 1]")
        return
    if kind is FactorValueKind.OPTIONAL_NONNEGATIVE_INTEGER and value is None:
        return
    if type(value) is not int:
        qualifier = "an integer or None" if (
            kind is FactorValueKind.OPTIONAL_NONNEGATIVE_INTEGER
        ) else "an integer"
        raise TypeError(f"{spec.factor.value} must be {qualifier}")
    if value < 0:
        raise ValueError(f"{spec.factor.value} cannot be negative")


def _factor_values_equal(
    left: FactorValue,
    right: FactorValue,
) -> bool:
    """Compare primitive factors exactly, including the sign of floating zero."""

    if type(left) is not type(right):
        return False
    if type(left) is float:
        return left.hex() == right.hex()
    return left == right


def _validate_sha256(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "ATOMIC_FACTOR_NAMES",
    "ATOMIC_FACTOR_SPECS",
    "CAUSAL_DESIGN_SCHEMA_VERSION",
    "AtomicFactorSpec",
    "CampaignValidationError",
    "CausalDesignValidationError",
    "CausalFactorMatrixValidationError",
    "CausalDesignAssessment",
    "CausalDesignRegistry",
    "CausalDesignStatus",
    "CausalFactor",
    "ContrastClassification",
    "ContrastRole",
    "ContrastStructuralScope",
    "ContrastSpec",
    "FactorDifference",
    "FactorSource",
    "FactorValueKind",
    "ScenarioFactorVector",
    "ScenarioMatrixMismatch",
    "assess_causal_design",
    "build_causal_design_registry",
]
