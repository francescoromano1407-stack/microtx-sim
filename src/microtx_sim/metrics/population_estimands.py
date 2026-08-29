"""Exact, fail-closed target-population estimand primitives.

This module is intentionally independent of the legacy batch and output paths.
It does not change the equal-player semantics of any released CSV table.  A
future runtime integration can use these objects to re-attest design weights and
resolve the declared target, projection, balance, metric, and output identities
before it publishes a separately versioned target-population result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
import math
import re
from typing import Sequence

import numpy as np

from ..agents.players import PlayerTable, projected_population_assignment_sha256
from ..data.population_evidence import PopulationEstimandRole


POPULATION_ESTIMAND_SCHEMA_VERSION = "1.0"
EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION = "1.0"
TARGET_POPULATION_OUTPUT_PROFILE = "target_population_estimands"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CURRENCY = re.compile(r"[A-Z]{3}\Z")
_MAX_EXACT_INTEGER_BITS = 4096
_MAX_TARGET_POPULATION_COUNT = 2**63 - 1


class PopulationEstimandValidationError(ValueError):
    """Raised when an estimand declaration or calculation is not auditable."""


class PopulationAnalysisUnit(str, Enum):
    """The only analysis unit currently compatible with per-player outcomes."""

    PLAYER_PERSON = "PLAYER_PERSON"


class PopulationInclusionTiming(str, Enum):
    """Timing at which an inclusion/exclusion field is defined."""

    PRETREATMENT = "PRETREATMENT"
    POSTTREATMENT = "POSTTREATMENT"


class PopulationInclusionField(str, Enum):
    """Whitelisted pre-treatment fields from the joint-population contract."""

    AGE_YEARS = "AGE_YEARS"
    JURISDICTION = "JURISDICTION"
    HOUSEHOLD_INCOME_BAND = "HOUSEHOLD_INCOME_BAND"
    HOUSEHOLD_TYPE = "HOUSEHOLD_TYPE"
    GAMING_STATE = "PRETREATMENT_GAMING_STATE"
    PAYER_HISTORY_STATE = "PRETREATMENT_PAYER_HISTORY_STATE"
    IS_MINOR = "PRETREATMENT_IS_MINOR"


class PopulationMetricKind(str, Enum):
    """Storage/semantic family of one player-level metric."""

    SCORE = "SCORE"
    RATIO = "RATIO"
    COUNT = "COUNT"
    TIME = "TIME"
    MONEY_MINOR_UNITS = "MONEY_MINOR_UNITS"
    OTHER = "OTHER"


class PopulationMetricScale(str, Enum):
    """Whether a per-player metric may coherently be expanded to a total."""

    NONADDITIVE = "NONADDITIVE"
    ADDITIVE_PER_ANALYSIS_UNIT = "ADDITIVE_PER_ANALYSIS_UNIT"


class PopulationContrast(str, Enum):
    """The scenario contrast represented by the estimand."""

    NONE = "NONE"
    TREATED_MINUS_CONTROL = "TREATED_MINUS_CONTROL"


class PopulationEstimandAlgorithm(str, Enum):
    """Versioned algorithms with exact rational accumulation."""

    WEIGHTED_MEAN_V1 = "EXACT_RATIONAL_WEIGHTED_MEAN_V1"
    PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1 = (
        "EXACT_RATIONAL_PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1"
    )
    WEIGHTED_QUANTILE_V1 = "EXACT_RATIONAL_WEIGHTED_INVERSE_CDF_QUANTILE_V1"


class PopulationNormalization(str, Enum):
    """How a weighted player reduction is placed on its reporting scale."""

    DIVIDE_BY_WEIGHT_SUM = "DIVIDE_BY_EXACT_DESIGN_WEIGHT_SUM"
    TARGET_POPULATION_TOTAL = (
        "DIVIDE_BY_EXACT_DESIGN_WEIGHT_SUM_THEN_MULTIPLY_TARGET_COUNT"
    )
    WEIGHTED_INVERSE_CDF = "EXACT_WEIGHTED_INVERSE_CDF_NO_INTERPOLATION"


class PopulationCurrencyRounding(str, Enum):
    """Rounding behavior of the isolated primitive."""

    NONE_EXACT_RATIONAL = "NONE_EXACT_RATIONAL"


@dataclass(frozen=True, slots=True)
class PopulationInclusionRule:
    """A pre-treatment, calibration-derived inclusion declaration.

    The actual included ordered player IDs live in :class:`ExactPopulationWeights`.
    This type records a declaration only: its enums reject an overtly post-
    treatment or validation-derived label, but they do not execute a predicate or
    prove how a caller selected IDs. A future integration must bind an executable
    canonical predicate, pre-treatment covariate snapshot, and selection result.
    """

    rule_id: str
    description: str
    source_fields: tuple[PopulationInclusionField, ...]
    timing: PopulationInclusionTiming
    evidence_role: PopulationEstimandRole

    def __post_init__(self) -> None:
        _validate_identifier(self.rule_id, name="inclusion rule_id")
        _validate_text(self.description, name="inclusion description")
        if type(self.source_fields) is not tuple:
            raise TypeError("inclusion source_fields must be an immutable tuple")
        if any(type(item) is not PopulationInclusionField for item in self.source_fields):
            raise TypeError(
                "inclusion source_fields must contain PopulationInclusionField values"
            )
        if len(set(self.source_fields)) != len(self.source_fields):
            raise PopulationEstimandValidationError(
                "inclusion source_fields cannot repeat"
            )
        if self.source_fields != tuple(sorted(self.source_fields, key=lambda item: item.value)):
            raise PopulationEstimandValidationError(
                "inclusion source_fields must use ascending canonical order"
            )
        if type(self.timing) is not PopulationInclusionTiming:
            raise TypeError("inclusion timing must be PopulationInclusionTiming")
        if self.timing is not PopulationInclusionTiming.PRETREATMENT:
            raise PopulationEstimandValidationError(
                "population exclusions must be defined before treatment"
            )
        if type(self.evidence_role) is not PopulationEstimandRole:
            raise TypeError("inclusion evidence_role must be PopulationEstimandRole")
        if self.evidence_role is not PopulationEstimandRole.CALIBRATION:
            raise PopulationEstimandValidationError(
                "held-out validation evidence cannot define population exclusions"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "source_fields": [item.value for item in self.source_fields],
            "timing": self.timing.value,
            "evidence_role": self.evidence_role.value,
        }


@dataclass(frozen=True, slots=True)
class PopulationPeriodSemantics:
    """Calendar period and interpretation attached to the metric."""

    period_start: date
    period_end: date
    description: str

    def __post_init__(self) -> None:
        if type(self.period_start) is not date or type(self.period_end) is not date:
            raise TypeError("population estimand periods must be calendar dates")
        if self.period_end < self.period_start:
            raise PopulationEstimandValidationError(
                "population estimand period ends before it starts"
            )
        _validate_text(self.description, name="period description")

    def snapshot(self) -> dict[str, str]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class PopulationCurrencySemantics:
    """Currency/price-basis contract for money-valued player outcomes."""

    currency_code: str
    minor_unit_name: str
    price_period_start: date
    price_period_end: date
    currency_basis_sha256: str
    rounding: PopulationCurrencyRounding = (
        PopulationCurrencyRounding.NONE_EXACT_RATIONAL
    )

    def __post_init__(self) -> None:
        if type(self.currency_code) is not str or not _CURRENCY.fullmatch(
            self.currency_code
        ):
            raise PopulationEstimandValidationError(
                "currency_code must be exactly three uppercase ASCII letters"
            )
        _validate_text(self.minor_unit_name, name="currency minor_unit_name")
        if (
            type(self.price_period_start) is not date
            or type(self.price_period_end) is not date
        ):
            raise TypeError("currency price periods must be calendar dates")
        if self.price_period_end < self.price_period_start:
            raise PopulationEstimandValidationError(
                "currency price period ends before it starts"
            )
        _validate_sha256(self.currency_basis_sha256, name="currency_basis_sha256")
        if type(self.rounding) is not PopulationCurrencyRounding:
            raise TypeError("currency rounding must be PopulationCurrencyRounding")
        if self.rounding is not PopulationCurrencyRounding.NONE_EXACT_RATIONAL:
            raise PopulationEstimandValidationError(
                "the exact primitive does not perform currency rounding"
            )

    def snapshot(self) -> dict[str, str]:
        return {
            "currency_code": self.currency_code,
            "minor_unit_name": self.minor_unit_name,
            "price_period_start": self.price_period_start.isoformat(),
            "price_period_end": self.price_period_end.isoformat(),
            "currency_basis_sha256": self.currency_basis_sha256,
            "rounding": self.rounding.value,
        }


@dataclass(frozen=True, slots=True)
class ExactPopulationWeights:
    """Immutable ordered player IDs and canonical exact rational design weights."""

    schema_version: str
    player_ids: tuple[int, ...]
    weight_numerators: tuple[int, ...]
    weight_denominators: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.schema_version != EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION:
            raise PopulationEstimandValidationError(
                "unsupported exact population-weights schema version"
            )
        for name in ("player_ids", "weight_numerators", "weight_denominators"):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be an immutable tuple")
        size = len(self.player_ids)
        if size == 0:
            raise PopulationEstimandValidationError(
                "a population estimand requires at least one included player"
            )
        if len(self.weight_numerators) != size or len(self.weight_denominators) != size:
            raise PopulationEstimandValidationError(
                "ordered player IDs and exact weights must have equal lengths"
            )
        for index, player_id in enumerate(self.player_ids):
            _validate_exact_int(
                player_id,
                name=f"player_ids[{index}]",
                minimum=0,
                maximum=2**63 - 1,
            )
        if len(set(self.player_ids)) != size:
            raise PopulationEstimandValidationError("player_ids must be unique")
        for index, (numerator, denominator) in enumerate(
            zip(self.weight_numerators, self.weight_denominators)
        ):
            _validate_fraction_parts(
                numerator,
                denominator,
                name=f"design weight[{index}]",
                positive=True,
            )

    @property
    def fractions(self) -> tuple[Fraction, ...]:
        return tuple(
            Fraction(numerator, denominator)
            for numerator, denominator in zip(
                self.weight_numerators,
                self.weight_denominators,
            )
        )

    @property
    def weight_sum(self) -> Fraction:
        return sum(self.fractions, start=Fraction(0, 1))

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "player_count": len(self.player_ids),
            "player_ids_decimal": [str(value) for value in self.player_ids],
            "weights": [
                {
                    "numerator_decimal": str(numerator),
                    "denominator_decimal": str(denominator),
                }
                for numerator, denominator in zip(
                    self.weight_numerators,
                    self.weight_denominators,
                )
            ],
        }

    @property
    def design_sha256(self) -> str:
        return _canonical_sha256(self.snapshot())


@dataclass(frozen=True, slots=True)
class PopulationEstimandSpec:
    """Strict declaration of one target-population estimand."""

    schema_version: str
    estimand_id: str
    target_population_id: str
    target_evidence_sha256: str
    design_weights_sha256: str
    runtime_projection_sha256: str
    balance_report_sha256: str
    metric_contract_sha256: str
    output_profile_id: str
    output_profile_schema_sha256: str
    analysis_unit: PopulationAnalysisUnit
    inclusion_rule: PopulationInclusionRule
    metric_name: str
    metric_kind: PopulationMetricKind
    metric_scale: PopulationMetricScale
    contrast: PopulationContrast
    algorithm: PopulationEstimandAlgorithm
    normalization: PopulationNormalization
    period: PopulationPeriodSemantics
    currency: PopulationCurrencySemantics | None = None
    target_population_count: int | None = None
    quantile_probability_numerator: int | None = None
    quantile_probability_denominator: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != POPULATION_ESTIMAND_SCHEMA_VERSION:
            raise PopulationEstimandValidationError(
                "unsupported population-estimand schema version"
            )
        _validate_identifier(self.estimand_id, name="estimand_id")
        _validate_identifier(
            self.target_population_id,
            name="target_population_id",
        )
        for name in (
            "target_evidence_sha256",
            "design_weights_sha256",
            "runtime_projection_sha256",
            "balance_report_sha256",
            "metric_contract_sha256",
            "output_profile_schema_sha256",
        ):
            _validate_sha256(getattr(self, name), name=name)
        if self.output_profile_id != TARGET_POPULATION_OUTPUT_PROFILE:
            raise PopulationEstimandValidationError(
                "population estimands require the dedicated target-population "
                "output profile"
            )
        if type(self.analysis_unit) is not PopulationAnalysisUnit:
            raise TypeError("analysis_unit must be PopulationAnalysisUnit")
        if self.analysis_unit is not PopulationAnalysisUnit.PLAYER_PERSON:
            raise PopulationEstimandValidationError(
                "per-player design weights require PLAYER_PERSON as analysis unit"
            )
        if type(self.inclusion_rule) is not PopulationInclusionRule:
            raise TypeError("inclusion_rule must be PopulationInclusionRule")
        _validate_identifier(self.metric_name, name="metric_name")
        for name, expected_type in (
            ("metric_kind", PopulationMetricKind),
            ("metric_scale", PopulationMetricScale),
            ("contrast", PopulationContrast),
            ("algorithm", PopulationEstimandAlgorithm),
            ("normalization", PopulationNormalization),
        ):
            if type(getattr(self, name)) is not expected_type:
                raise TypeError(f"{name} must be {expected_type.__name__}")
        if type(self.period) is not PopulationPeriodSemantics:
            raise TypeError("period must be PopulationPeriodSemantics")
        if self.currency is not None and type(self.currency) is not PopulationCurrencySemantics:
            raise TypeError("currency must be PopulationCurrencySemantics or None")
        if self.metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS:
            if self.currency is None:
                raise PopulationEstimandValidationError(
                    "money metrics require explicit currency semantics"
                )
        elif self.currency is not None:
            raise PopulationEstimandValidationError(
                "currency semantics are only valid for money metrics"
            )

        is_quantile = self.algorithm is PopulationEstimandAlgorithm.WEIGHTED_QUANTILE_V1
        is_paired = (
            self.algorithm
            is PopulationEstimandAlgorithm.PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1
        )
        if is_quantile:
            if self.contrast is not PopulationContrast.NONE:
                raise PopulationEstimandValidationError(
                    "weighted quantiles do not accept a scenario contrast"
                )
            if self.normalization is not PopulationNormalization.WEIGHTED_INVERSE_CDF:
                raise PopulationEstimandValidationError(
                    "weighted quantiles require inverse-CDF normalization"
                )
        elif is_paired:
            if self.contrast is not PopulationContrast.TREATED_MINUS_CONTROL:
                raise PopulationEstimandValidationError(
                    "paired differences require TREATED_MINUS_CONTROL"
                )
            if self.normalization is PopulationNormalization.WEIGHTED_INVERSE_CDF:
                raise PopulationEstimandValidationError(
                    "paired means cannot use quantile normalization"
                )
        else:
            if self.contrast is not PopulationContrast.NONE:
                raise PopulationEstimandValidationError(
                    "a one-world weighted mean cannot declare a paired contrast"
                )
            if self.normalization is PopulationNormalization.WEIGHTED_INVERSE_CDF:
                raise PopulationEstimandValidationError(
                    "weighted means cannot use quantile normalization"
                )

        if is_quantile:
            if (
                self.quantile_probability_numerator is None
                or self.quantile_probability_denominator is None
            ):
                raise PopulationEstimandValidationError(
                    "weighted quantiles require an exact probability"
                )
            _validate_fraction_parts(
                self.quantile_probability_numerator,
                self.quantile_probability_denominator,
                name="quantile probability",
                minimum=0,
                maximum=1,
            )
        elif (
            self.quantile_probability_numerator is not None
            or self.quantile_probability_denominator is not None
        ):
            raise PopulationEstimandValidationError(
                "quantile probability is only valid for weighted quantiles"
            )

        is_total = self.normalization is PopulationNormalization.TARGET_POPULATION_TOTAL
        if is_total:
            if self.metric_scale is not PopulationMetricScale.ADDITIVE_PER_ANALYSIS_UNIT:
                raise PopulationEstimandValidationError(
                    "target totals require an additive per-analysis-unit metric"
                )
            _validate_exact_int(
                self.target_population_count,
                name="target_population_count",
                minimum=1,
                maximum=_MAX_TARGET_POPULATION_COUNT,
            )
        elif self.target_population_count is not None:
            raise PopulationEstimandValidationError(
                "target_population_count is only valid for target totals"
            )

    @property
    def quantile_probability(self) -> Fraction | None:
        if self.quantile_probability_numerator is None:
            return None
        assert self.quantile_probability_denominator is not None
        return Fraction(
            self.quantile_probability_numerator,
            self.quantile_probability_denominator,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "estimand_id": self.estimand_id,
            "target_population_id": self.target_population_id,
            "target_evidence_sha256": self.target_evidence_sha256,
            "design_weights_sha256": self.design_weights_sha256,
            "runtime_projection_sha256": self.runtime_projection_sha256,
            "balance_report_sha256": self.balance_report_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "output_profile_id": self.output_profile_id,
            "output_profile_schema_sha256": self.output_profile_schema_sha256,
            "analysis_unit": self.analysis_unit.value,
            "inclusion_rule": self.inclusion_rule.snapshot(),
            "metric_name": self.metric_name,
            "metric_kind": self.metric_kind.value,
            "metric_scale": self.metric_scale.value,
            "contrast": self.contrast.value,
            "algorithm": self.algorithm.value,
            "normalization": self.normalization.value,
            "period": self.period.snapshot(),
            "currency": self.currency.snapshot() if self.currency is not None else None,
            "target_population_count_decimal": (
                str(self.target_population_count)
                if self.target_population_count is not None
                else None
            ),
            "quantile_probability": (
                {
                    "numerator_decimal": str(self.quantile_probability_numerator),
                    "denominator_decimal": str(self.quantile_probability_denominator),
                }
                if self.quantile_probability_numerator is not None
                else None
            ),
        }

    @property
    def estimand_sha256(self) -> str:
        return _canonical_sha256(self.snapshot())


@dataclass(frozen=True, slots=True)
class PopulationEstimandResult:
    """Content-addressed exact result from one declared population estimand."""

    schema_version: str
    estimand_sha256: str
    design_weights_sha256: str
    algorithm: PopulationEstimandAlgorithm
    metric_name: str
    contrast: PopulationContrast
    normalization: PopulationNormalization
    player_count: int
    numerator: int
    denominator: int
    weight_sum_numerator: int
    weight_sum_denominator: int
    target_population_count: int | None
    result_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != POPULATION_ESTIMAND_SCHEMA_VERSION:
            raise PopulationEstimandValidationError(
                "unsupported population-estimand result schema version"
            )
        _validate_sha256(self.estimand_sha256, name="estimand_sha256")
        _validate_sha256(
            self.design_weights_sha256,
            name="design_weights_sha256",
        )
        if type(self.algorithm) is not PopulationEstimandAlgorithm:
            raise TypeError("result algorithm must be PopulationEstimandAlgorithm")
        _validate_identifier(self.metric_name, name="result metric_name")
        if type(self.contrast) is not PopulationContrast:
            raise TypeError("result contrast must be PopulationContrast")
        if type(self.normalization) is not PopulationNormalization:
            raise TypeError("result normalization must be PopulationNormalization")
        _validate_exact_int(self.player_count, name="result player_count", minimum=1)
        _validate_fraction_parts(
            self.numerator,
            self.denominator,
            name="result value",
        )
        _validate_fraction_parts(
            self.weight_sum_numerator,
            self.weight_sum_denominator,
            name="result weight sum",
            positive=True,
        )
        if self.target_population_count is not None:
            _validate_exact_int(
                self.target_population_count,
                name="result target_population_count",
                minimum=1,
                maximum=_MAX_TARGET_POPULATION_COUNT,
            )
        _require_finite_float(self.value_fraction, name="population estimand result")
        _validate_sha256(self.result_sha256, name="result_sha256")
        expected = _canonical_sha256(self.attestation_payload())
        if self.result_sha256 != expected:
            raise PopulationEstimandValidationError(
                "result_sha256 does not match the population estimand payload"
            )

    @property
    def value_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def value(self) -> float:
        return float(self.value_fraction)

    @property
    def weight_sum(self) -> Fraction:
        return Fraction(self.weight_sum_numerator, self.weight_sum_denominator)

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "estimand_sha256": self.estimand_sha256,
            "design_weights_sha256": self.design_weights_sha256,
            "algorithm": self.algorithm.value,
            "metric_name": self.metric_name,
            "contrast": self.contrast.value,
            "normalization": self.normalization.value,
            "player_count": self.player_count,
            "numerator_decimal": str(self.numerator),
            "denominator_decimal": str(self.denominator),
            "weight_sum_numerator_decimal": str(self.weight_sum_numerator),
            "weight_sum_denominator_decimal": str(self.weight_sum_denominator),
            "target_population_count_decimal": (
                str(self.target_population_count)
                if self.target_population_count is not None
                else None
            ),
        }

    def snapshot(self) -> dict[str, object]:
        return {**self.attestation_payload(), "result_sha256": self.result_sha256}


def exact_population_weights_from_projected_players(
    players: PlayerTable,
) -> ExactPopulationWeights:
    """Rebuild exact ordered design weights from a validated projection sidecar."""

    if type(players) is not PlayerTable:
        raise TypeError("players must be a PlayerTable")
    assignment = players.projected_population
    if assignment is None:
        raise PopulationEstimandValidationError(
            "exact target-population weights require a projected PlayerTable"
        )
    observed_assignment_sha256 = projected_population_assignment_sha256(
        assignment.metadata,
        players.player_id,
        assignment.cell_index,
    )
    if observed_assignment_sha256 != assignment.assignment_sha256:
        raise PopulationEstimandValidationError(
            "projected population assignment was mutated after attestation"
        )
    cells = assignment.metadata.cells
    numerators = tuple(
        cells[int(cell_index)].analysis_weight[0]
        for cell_index in assignment.cell_index
    )
    denominators = tuple(
        cells[int(cell_index)].analysis_weight[1]
        for cell_index in assignment.cell_index
    )
    weights = ExactPopulationWeights(
        schema_version=EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
        player_ids=tuple(int(player_id) for player_id in players.player_id),
        weight_numerators=numerators,
        weight_denominators=denominators,
    )
    if weights.weight_sum != 1:
        raise PopulationEstimandValidationError(
            "projected player analysis weights do not reconstruct unit target mass"
        )
    return weights


def weighted_mean(
    spec: PopulationEstimandSpec,
    weights: ExactPopulationWeights,
    outcomes: Sequence[object],
) -> PopulationEstimandResult:
    """Compute an exact weighted mean or declared target-population total."""

    _require_algorithm(spec, PopulationEstimandAlgorithm.WEIGHTED_MEAN_V1)
    _validate_design_binding(spec, weights)
    values = _outcome_fractions(
        outcomes,
        expected_size=len(weights.player_ids),
        metric_kind=spec.metric_kind,
        name="outcomes",
    )
    return _mean_result(spec, weights, values)


def paired_weighted_mean_difference(
    spec: PopulationEstimandSpec,
    treated_weights: ExactPopulationWeights,
    treated_outcomes: Sequence[object],
    control_weights: ExactPopulationWeights,
    control_outcomes: Sequence[object],
) -> PopulationEstimandResult:
    """Compute treated-minus-control on one exactly aligned weighted cohort."""

    _require_algorithm(
        spec,
        PopulationEstimandAlgorithm.PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1,
    )
    if type(treated_weights) is not ExactPopulationWeights or type(
        control_weights
    ) is not ExactPopulationWeights:
        raise TypeError("paired population weights must be ExactPopulationWeights")
    if treated_weights.player_ids != control_weights.player_ids:
        raise PopulationEstimandValidationError(
            "paired effects require identical ordered player IDs"
        )
    if (
        treated_weights.weight_numerators != control_weights.weight_numerators
        or treated_weights.weight_denominators != control_weights.weight_denominators
    ):
        raise PopulationEstimandValidationError(
            "paired effects reject scenario-dependent design weights"
        )
    _validate_design_binding(spec, treated_weights)
    _validate_design_binding(spec, control_weights)
    treated = _outcome_fractions(
        treated_outcomes,
        expected_size=len(treated_weights.player_ids),
        metric_kind=spec.metric_kind,
        name="treated_outcomes",
    )
    control = _outcome_fractions(
        control_outcomes,
        expected_size=len(control_weights.player_ids),
        metric_kind=spec.metric_kind,
        name="control_outcomes",
    )
    differences: list[Fraction] = []
    for index, (treated_value, control_value) in enumerate(zip(treated, control)):
        difference = treated_value - control_value
        _require_finite_float(
            difference,
            name=f"paired outcome difference[{index}]",
        )
        differences.append(difference)
    return _mean_result(spec, treated_weights, tuple(differences))


def weighted_quantile(
    spec: PopulationEstimandSpec,
    weights: ExactPopulationWeights,
    outcomes: Sequence[object],
) -> PopulationEstimandResult:
    """Return the lower weighted inverse-CDF quantile without interpolation.

    Rows are ordered exactly by ``(outcome, player_id)``.  The selected value is
    the smallest observed outcome whose cumulative exact weight is greater than
    or equal to ``q * total_weight``.  At ``q=0`` this is the minimum; at an
    exact cumulative boundary it is the value on that lower boundary.  Player
    IDs make ordering deterministic even when equal outcomes are supplied.
    """

    _require_algorithm(spec, PopulationEstimandAlgorithm.WEIGHTED_QUANTILE_V1)
    _validate_design_binding(spec, weights)
    values = _outcome_fractions(
        outcomes,
        expected_size=len(weights.player_ids),
        metric_kind=spec.metric_kind,
        name="outcomes",
    )
    probability = spec.quantile_probability
    assert probability is not None
    total_weight = weights.weight_sum
    threshold = probability * total_weight
    ordered = sorted(
        zip(values, weights.player_ids, weights.fractions),
        key=lambda item: (item[0], item[1]),
    )
    cumulative = Fraction(0, 1)
    selected = ordered[0][0]
    for value, _player_id, weight in ordered:
        selected = value
        cumulative += weight
        if probability == 0 or cumulative >= threshold:
            break
    return _build_result(spec, weights, selected, total_weight)


def _mean_result(
    spec: PopulationEstimandSpec,
    weights: ExactPopulationWeights,
    values: tuple[Fraction, ...],
) -> PopulationEstimandResult:
    total_weight = weights.weight_sum
    weighted_sum = sum(
        (weight * value for weight, value in zip(weights.fractions, values)),
        start=Fraction(0, 1),
    )
    estimate = weighted_sum / total_weight
    if spec.normalization is PopulationNormalization.TARGET_POPULATION_TOTAL:
        assert spec.target_population_count is not None
        estimate *= spec.target_population_count
    _require_finite_float(estimate, name="population estimand result")
    return _build_result(spec, weights, estimate, total_weight)


def _build_result(
    spec: PopulationEstimandSpec,
    weights: ExactPopulationWeights,
    estimate: Fraction,
    total_weight: Fraction,
) -> PopulationEstimandResult:
    payload = {
        "schema_version": POPULATION_ESTIMAND_SCHEMA_VERSION,
        "estimand_sha256": spec.estimand_sha256,
        "design_weights_sha256": weights.design_sha256,
        "algorithm": spec.algorithm.value,
        "metric_name": spec.metric_name,
        "contrast": spec.contrast.value,
        "normalization": spec.normalization.value,
        "player_count": len(weights.player_ids),
        "numerator_decimal": str(estimate.numerator),
        "denominator_decimal": str(estimate.denominator),
        "weight_sum_numerator_decimal": str(total_weight.numerator),
        "weight_sum_denominator_decimal": str(total_weight.denominator),
        "target_population_count_decimal": (
            str(spec.target_population_count)
            if spec.target_population_count is not None
            else None
        ),
    }
    return PopulationEstimandResult(
        schema_version=POPULATION_ESTIMAND_SCHEMA_VERSION,
        estimand_sha256=spec.estimand_sha256,
        design_weights_sha256=weights.design_sha256,
        algorithm=spec.algorithm,
        metric_name=spec.metric_name,
        contrast=spec.contrast,
        normalization=spec.normalization,
        player_count=len(weights.player_ids),
        numerator=estimate.numerator,
        denominator=estimate.denominator,
        weight_sum_numerator=total_weight.numerator,
        weight_sum_denominator=total_weight.denominator,
        target_population_count=spec.target_population_count,
        result_sha256=_canonical_sha256(payload),
    )


def _validate_design_binding(
    spec: PopulationEstimandSpec,
    weights: ExactPopulationWeights,
) -> None:
    if type(spec) is not PopulationEstimandSpec:
        raise TypeError("spec must be PopulationEstimandSpec")
    if type(weights) is not ExactPopulationWeights:
        raise TypeError("weights must be ExactPopulationWeights")
    if spec.design_weights_sha256 != weights.design_sha256:
        raise PopulationEstimandValidationError(
            "estimand design_weights_sha256 does not re-attest the supplied weights"
        )


def _require_algorithm(
    spec: PopulationEstimandSpec,
    expected: PopulationEstimandAlgorithm,
) -> None:
    if type(spec) is not PopulationEstimandSpec:
        raise TypeError("spec must be PopulationEstimandSpec")
    if spec.algorithm is not expected:
        raise PopulationEstimandValidationError(
            f"estimand algorithm must be {expected.value}"
        )


def _outcome_fractions(
    values: Sequence[object],
    *,
    expected_size: int,
    metric_kind: PopulationMetricKind,
    name: str,
) -> tuple[Fraction, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if len(supplied) != expected_size:
        raise PopulationEstimandValidationError(
            f"{name} must align one-for-one with ordered design player IDs"
        )
    output: list[Fraction] = []
    for index, value in enumerate(supplied):
        item_name = f"{name}[{index}]"
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{item_name} must be numeric, not boolean")
        if metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS and not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(
                f"{item_name} must be exact integer currency minor units"
            )
        if isinstance(value, Fraction):
            fraction = value
        elif isinstance(value, (int, np.integer)):
            fraction = Fraction(int(value), 1)
        elif isinstance(value, (float, np.floating)):
            normalized = float(value)
            if not math.isfinite(normalized):
                raise PopulationEstimandValidationError(
                    f"{item_name} must be finite"
                )
            fraction = Fraction.from_float(normalized)
        else:
            raise TypeError(
                f"{item_name} must be int, float, or Fraction"
            )
        _validate_integer_size(fraction.numerator, name=f"{item_name} numerator")
        _validate_integer_size(fraction.denominator, name=f"{item_name} denominator")
        _require_finite_float(fraction, name=item_name)
        output.append(fraction)
    return tuple(output)


def _validate_fraction_parts(
    numerator: object,
    denominator: object,
    *,
    name: str,
    positive: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    _validate_exact_int(numerator, name=f"{name} numerator")
    _validate_exact_int(denominator, name=f"{name} denominator", minimum=1)
    assert type(numerator) is int and type(denominator) is int
    if positive and numerator <= 0:
        raise PopulationEstimandValidationError(f"{name} must be positive")
    fraction = Fraction(numerator, denominator)
    if fraction.numerator != numerator or fraction.denominator != denominator:
        raise PopulationEstimandValidationError(
            f"{name} must use a reduced canonical fraction"
        )
    if minimum is not None and fraction < minimum:
        raise PopulationEstimandValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and fraction > maximum:
        raise PopulationEstimandValidationError(f"{name} must be at most {maximum}")


def _validate_exact_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a strict Python integer")
    _validate_integer_size(value, name=name)
    if minimum is not None and value < minimum:
        raise PopulationEstimandValidationError(
            f"{name} must be at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise PopulationEstimandValidationError(
            f"{name} must be at most {maximum}"
        )


def _validate_integer_size(value: int, *, name: str) -> None:
    if abs(value).bit_length() > _MAX_EXACT_INTEGER_BITS:
        raise PopulationEstimandValidationError(
            f"{name} exceeds the exact-integer safety limit"
        )


def _require_finite_float(value: Fraction, *, name: str) -> None:
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise OverflowError(f"{name} is outside the finite float64 range") from exc
    if not math.isfinite(normalized):
        raise OverflowError(f"{name} is outside the finite float64 range")


def _validate_identifier(value: object, *, name: str) -> None:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise PopulationEstimandValidationError(
            f"{name} must be a stable ASCII identifier"
        )


def _validate_text(value: object, *, name: str) -> None:
    if type(value) is not str or not value.strip() or value.strip() != value:
        raise PopulationEstimandValidationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )
    if len(value) > 1024:
        raise PopulationEstimandValidationError(f"{name} is too long")


def _validate_sha256(value: object, *, name: str) -> None:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PopulationEstimandValidationError(
            f"{name} must be lowercase SHA-256 hex"
        )


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION",
    "POPULATION_ESTIMAND_SCHEMA_VERSION",
    "TARGET_POPULATION_OUTPUT_PROFILE",
    "ExactPopulationWeights",
    "PopulationAnalysisUnit",
    "PopulationContrast",
    "PopulationCurrencyRounding",
    "PopulationCurrencySemantics",
    "PopulationEstimandAlgorithm",
    "PopulationEstimandResult",
    "PopulationEstimandSpec",
    "PopulationEstimandValidationError",
    "PopulationInclusionField",
    "PopulationInclusionRule",
    "PopulationInclusionTiming",
    "PopulationMetricKind",
    "PopulationMetricScale",
    "PopulationNormalization",
    "PopulationPeriodSemantics",
    "exact_population_weights_from_projected_players",
    "paired_weighted_mean_difference",
    "weighted_mean",
    "weighted_quantile",
]
