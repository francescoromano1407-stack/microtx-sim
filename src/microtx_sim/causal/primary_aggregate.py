"""Plan-level aggregation of exact per-seed PRIMARY realizations.

Population-analysis weights have already been applied inside each retained
``SeedAnalysisBinding``.  This layer gives every independently configured seed
equal weight and never selects seeds by the sign or favourability of an
outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from math import isfinite, sqrt
from typing import Final

from ..rng import validate_seed
from .analysis_binding import RunAnalysisBinding
from .analysis_plan import PrimaryAggregateRule


PRIMARY_AGGREGATE_SCHEMA_VERSION: Final[str] = "1.0"
NORMAL_95_MONTE_CARLO_INTERVAL: Final[str] = (
    "NORMAL_95_MONTE_CARLO_MEAN_PLUS_MINUS_1.96_MCSE"
)


class PrimaryAggregateValidationError(ValueError):
    """Raised when the declared complete fixed-seed aggregate cannot be made."""


@dataclass(frozen=True, slots=True)
class PrimarySeedRealization:
    """One exact, already population-weighted PRIMARY seed realization."""

    seed: int
    numerator: int
    denominator: int
    result_sha256: str

    def __post_init__(self) -> None:
        validate_seed(self.seed, name="primary realization seed")
        if type(self.numerator) is not int or type(self.denominator) is not int:
            raise TypeError(
                "primary realization numerator/denominator must be integers"
            )
        if self.denominator <= 0:
            raise PrimaryAggregateValidationError(
                "primary realization denominator must be positive"
            )
        canonical = Fraction(self.numerator, self.denominator)
        if (
            canonical.numerator != self.numerator
            or canonical.denominator != self.denominator
        ):
            raise PrimaryAggregateValidationError(
                "primary realization fraction must be reduced and canonical"
            )
        _sha256_digest(self.result_sha256, name="result_sha256")

    @property
    def value_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def snapshot(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "seed_decimal": str(self.seed),
            "numerator_decimal": str(self.numerator),
            "denominator_decimal": str(self.denominator),
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True, slots=True)
class PrimaryMonteCarloSummary:
    """Outcome-blind arithmetic summary over a complete fixed seed set."""

    retained_seed_count: int
    excluded_seed_count: int
    point_estimate: float
    between_seed_sample_standard_deviation: float
    monte_carlo_standard_error: float
    interval_lower: float
    interval_upper: float

    def __post_init__(self) -> None:
        if (
            type(self.retained_seed_count) is not int
            or self.retained_seed_count < 1
        ):
            raise PrimaryAggregateValidationError(
                "retained_seed_count must be a positive integer"
            )
        if self.excluded_seed_count != 0:
            raise PrimaryAggregateValidationError(
                "the declared rule permits no seed exclusions"
            )
        for name in (
            "point_estimate",
            "between_seed_sample_standard_deviation",
            "monte_carlo_standard_error",
            "interval_lower",
            "interval_upper",
        ):
            value = getattr(self, name)
            if type(value) is not float or not isfinite(value):
                raise PrimaryAggregateValidationError(
                    f"{name} must be finite float"
                )
        if (
            self.between_seed_sample_standard_deviation < 0.0
            or self.monte_carlo_standard_error < 0.0
            or self.interval_lower > self.interval_upper
        ):
            raise PrimaryAggregateValidationError(
                "primary Monte Carlo dispersion or interval is invalid"
            )


def summarize_primary_realizations(
    rule: PrimaryAggregateRule,
    *,
    expected_seeds: tuple[int, ...],
    realizations: tuple[PrimarySeedRealization, ...],
) -> PrimaryMonteCarloSummary:
    """Apply the declared equal-seed normal Monte Carlo convention.

    Missing, duplicate, reordered, or invalid realizations fail closed.  There
    is intentionally no exclusion argument and therefore no API through which
    outcome favourability can affect retention.
    """

    if type(rule) is not PrimaryAggregateRule:
        raise TypeError("rule must be PrimaryAggregateRule")
    PrimaryAggregateRule.__post_init__(rule)
    if type(expected_seeds) is not tuple or not expected_seeds:
        raise TypeError("expected_seeds must be a non-empty exact integer tuple")
    for index, seed in enumerate(expected_seeds):
        validate_seed(seed, name=f"expected_seeds[{index}]")
    if expected_seeds != tuple(sorted(expected_seeds)) or len(
        set(expected_seeds)
    ) != len(expected_seeds):
        raise PrimaryAggregateValidationError(
            "expected seeds must be unique and in ascending canonical order"
        )
    if type(realizations) is not tuple or any(
        type(item) is not PrimarySeedRealization for item in realizations
    ):
        raise TypeError(
            "realizations must be an exact tuple of PrimarySeedRealization"
        )
    for realization in realizations:
        PrimarySeedRealization.__post_init__(realization)
    observed_seeds = tuple(item.seed for item in realizations)
    if observed_seeds != expected_seeds:
        raise PrimaryAggregateValidationError(
            "primary realizations must exactly cover the fixed seed set in order"
        )

    exact_values = tuple(item.value_fraction for item in realizations)
    exact_mean = sum(exact_values, Fraction(0, 1)) / len(exact_values)
    point = float(exact_mean)
    if len(exact_values) == 1:
        standard_deviation = 0.0
    else:
        exact_sum_of_squares = sum(
            ((value - exact_mean) ** 2 for value in exact_values),
            Fraction(0, 1),
        )
        standard_deviation = sqrt(
            float(exact_sum_of_squares / (len(exact_values) - 1))
        )
    mcse = standard_deviation / sqrt(len(exact_values))
    half_width = 1.96 * mcse
    return PrimaryMonteCarloSummary(
        retained_seed_count=len(exact_values),
        excluded_seed_count=0,
        point_estimate=point,
        between_seed_sample_standard_deviation=standard_deviation,
        monte_carlo_standard_error=mcse,
        interval_lower=point - half_width,
        interval_upper=point + half_width,
    )


@dataclass(frozen=True, slots=True)
class PlanPrimaryAggregate:
    """Content-addressed plan-level PRIMARY result and Monte Carlo metadata."""

    schema_version: str
    binding: RunAnalysisBinding
    realizations: tuple[PrimarySeedRealization, ...]
    summary: PrimaryMonteCarloSummary
    aggregate_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != PRIMARY_AGGREGATE_SCHEMA_VERSION:
            raise PrimaryAggregateValidationError(
                "unsupported primary aggregate schema version"
            )
        if type(self.binding) is not RunAnalysisBinding:
            raise TypeError("binding must be RunAnalysisBinding")
        RunAnalysisBinding.__post_init__(self.binding)
        rule = self.binding.plan.primary_aggregate_rule
        if type(rule) is not PrimaryAggregateRule:
            raise PrimaryAggregateValidationError(
                "plan does not declare a plan-level primary aggregate"
            )
        expected = _primary_realizations(self.binding)
        if self.realizations != expected:
            raise PrimaryAggregateValidationError(
                "aggregate realizations differ from exact retained seed bindings"
            )
        observed_summary = summarize_primary_realizations(
            rule,
            expected_seeds=self.binding.seeds,
            realizations=self.realizations,
        )
        if self.summary != observed_summary:
            raise PrimaryAggregateValidationError(
                "primary aggregate summary differs from retained realizations"
            )
        _sha256_digest(self.aggregate_sha256, name="aggregate_sha256")
        if self.aggregate_sha256 != _canonical_sha256(
            self.attestation_payload()
        ):
            raise PrimaryAggregateValidationError(
                "primary aggregate SHA-256 differs from its exact payload"
            )

    def attestation_payload(self) -> dict[str, object]:
        return _aggregate_attestation_payload(
            self.binding,
            self.realizations,
            self.summary,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "aggregate_sha256": self.aggregate_sha256,
        }


def compute_plan_primary_aggregate(
    binding: RunAnalysisBinding,
) -> PlanPrimaryAggregate:
    """Compute the schema-v2 plan aggregate from exact retained seed bindings."""

    if type(binding) is not RunAnalysisBinding:
        raise TypeError("binding must be RunAnalysisBinding")
    RunAnalysisBinding.__post_init__(binding)
    rule = binding.plan.primary_aggregate_rule
    if type(rule) is not PrimaryAggregateRule:
        raise PrimaryAggregateValidationError(
            "plan does not declare a plan-level primary aggregate"
        )
    realizations = _primary_realizations(binding)
    summary = summarize_primary_realizations(
        rule,
        expected_seeds=binding.seeds,
        realizations=realizations,
    )
    digest = _canonical_sha256(
        _aggregate_attestation_payload(binding, realizations, summary)
    )
    return PlanPrimaryAggregate(
        schema_version=PRIMARY_AGGREGATE_SCHEMA_VERSION,
        binding=binding,
        realizations=realizations,
        summary=summary,
        aggregate_sha256=digest,
    )


def _aggregate_attestation_payload(
    binding: RunAnalysisBinding,
    realizations: tuple[PrimarySeedRealization, ...],
    summary: PrimaryMonteCarloSummary,
) -> dict[str, object]:
    plan = binding.plan
    primary = plan.primary_estimand
    rule = plan.primary_aggregate_rule
    if type(rule) is not PrimaryAggregateRule:
        raise PrimaryAggregateValidationError(
            "plan does not declare a plan-level primary aggregate"
        )
    metric_sha256s = tuple(
        item.metric_contract_sha256
        for item in binding.seed_bindings
        if item.planned_estimand == primary
    )
    if len(set(metric_sha256s)) != 1:
        raise PrimaryAggregateValidationError(
            "primary seed metric-contract identities differ"
        )
    return {
        "schema_version": PRIMARY_AGGREGATE_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "primary_estimand_id": primary.estimand_id,
        "reference_scenario": primary.reference_scenario_id.value,
        "comparison_scenario": primary.comparison_scenario_id.value,
        "contrast_direction": primary.contrast_direction,
        "outcome_metric": primary.outcome_metric.value,
        "outcome_unit": primary.outcome_semantics.unit,
        "primary_aggregate_rule": rule.snapshot(),
        "retained_realizations": [item.snapshot() for item in realizations],
        "retained_seed_count": summary.retained_seed_count,
        "excluded_seed_count": summary.excluded_seed_count,
        "excluded_seeds": [],
        "point_estimate": summary.point_estimate,
        "between_seed_sample_standard_deviation": (
            summary.between_seed_sample_standard_deviation
        ),
        "monte_carlo_standard_error": summary.monte_carlo_standard_error,
        "interval_method": NORMAL_95_MONTE_CARLO_INTERVAL,
        "interval_lower": summary.interval_lower,
        "interval_upper": summary.interval_upper,
        "interval_interpretation": (
            "Monte Carlo variability of the configured simulator output mean; "
            "not a confidence interval for a real-world population."
        ),
        "population_weighting": (
            "Exact plan-declared population-analysis weights are applied "
            "within each seed before equal weighting across seeds."
        ),
        "scenario_weighting": (
            "None: the PRIMARY is one directed comparison-minus-reference "
            "contrast and no secondary scenario is averaged into it."
        ),
        "plan_sha256": plan.plan_sha256,
        "binding_sha256": binding.binding_sha256,
        "population_input_sha256": binding.population_input_sha256,
        "population_lineage_sha256": binding.population_lineage_sha256,
        "profile_input_sha256": binding.profile_input_sha256,
        "metric_contract_registry_sha256": (
            binding.metric_contract_registry_sha256
        ),
        "primary_metric_contract_sha256": metric_sha256s[0],
        "harm_weights_sha256": binding.harm_weights_sha256,
        "analysis_population_predicate_sha256": (
            primary.inclusion_predicate.predicate_sha256
        ),
        "synthetic_only": True,
        "empirical_validation_claimed": False,
        "campaign_ready": False,
    }


def _primary_realizations(
    binding: RunAnalysisBinding,
) -> tuple[PrimarySeedRealization, ...]:
    primary = binding.plan.primary_estimand
    selected = tuple(
        item
        for item in binding.seed_bindings
        if item.planned_estimand == primary
    )
    if tuple(item.seed for item in selected) != binding.seeds:
        raise PrimaryAggregateValidationError(
            "primary seed bindings do not exactly cover the fixed seed set"
        )
    return tuple(
        PrimarySeedRealization(
            seed=item.seed,
            numerator=item.result.numerator,
            denominator=item.result.denominator,
            result_sha256=item.result.result_sha256,
        )
        for item in selected
    )


def _sha256_digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PrimaryAggregateValidationError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "NORMAL_95_MONTE_CARLO_INTERVAL",
    "PRIMARY_AGGREGATE_SCHEMA_VERSION",
    "PlanPrimaryAggregate",
    "PrimaryAggregateValidationError",
    "PrimaryMonteCarloSummary",
    "PrimarySeedRealization",
    "compute_plan_primary_aggregate",
    "summarize_primary_realizations",
]
