"""Prospective, content-addressed analysis-plan declarations.

Schema version 1 is deliberately *not* a preregistration mechanism.  It can
freeze a proposed analysis before policy outcomes are evaluated, bind that
proposal to exact execution inputs, and execute a canonical pre-treatment
population predicate.  It cannot prove that the declaration existed at an
external time or was deposited with an independent registry.  Consequently
``registration_status``, ``preregistered``, and ``campaign_ready`` are fixed to
their fail-closed values in this schema.

No default plan is supplied.  A caller must opt in with a regular, non-symlink
JSON file and must compare every expected digest with independently resolved
runtime objects before using the declaration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Final, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from ..agents.players import ProjectedPopulationCellMetadata
from ..data.population_evidence import (
    PopulationEstimandRole,
    PopulationGamingState,
    PopulationPayerHistoryState,
)
from ..data.lineage import profile_lineage_fingerprint_matches
from ..metrics.harm import WelfareHarmWeights
from ..metrics.population_estimands import (
    PopulationCurrencySemantics,
    PopulationEstimandAlgorithm,
    PopulationInclusionField,
    PopulationInclusionRule,
    PopulationInclusionTiming,
    PopulationMetricKind,
    PopulationMetricScale,
    PopulationNormalization,
    PopulationPeriodSemantics,
)
from ..rng import validate_seed
from .scenarios import ScenarioId, required_scenarios


ANALYSIS_PLAN_SCHEMA_VERSION: Final[str] = "1.0"
PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION: Final[str] = "2.0"
CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION: Final[str] = "3.0"
EXPLORATORY_ANALYSIS_PLAN_SCHEMA_VERSION: Final[str] = "1.0"
EXPLORATORY_ANALYSIS_PLAN_KIND: Final[str] = (
    "EXPLORATORY_SYNTHETIC_NON_EMPIRICAL"
)
MAX_ANALYSIS_PLAN_BYTES: Final[int] = 1024 * 1024
ALL_MONTHLY_DISPOSABLE_INCOME_BANDS_ID: Final[str] = (
    "runtime.personal.monthly.income.all"
)
ALL_HOUSEHOLD_TYPES_ID: Final[str] = "household.all"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CONTRACT_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_JURISDICTION_CODE = re.compile(r"[A-Z][A-Z0-9-]{1,7}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_CAMPAIGN_BLOCKERS: Final[tuple[str, ...]] = (
    "analysis_plan.external_registration=unregistered",
    "analysis_plan.schema_v1=campaign_ineligible",
    "analysis_plan.execution_calendar_anchor=unbound",
    "analysis_plan.cross_seed_aggregation_uncertainty=unresolved",
    "analysis_plan.model_implementation_environment_identity=unbound",
)
_V2_CAMPAIGN_BLOCKERS: Final[tuple[str, ...]] = (
    "analysis_plan.external_registration=unregistered",
    "analysis_plan.schema_v2=campaign_ineligible",
    "analysis_plan.execution_calendar_anchor=unbound",
    "analysis_plan.model_implementation_environment_identity=unbound",
)
_V3_CAMPAIGN_BLOCKERS: Final[tuple[str, ...]] = (
    "analysis_plan.external_registration=unregistered",
    "analysis_plan.execution_calendar_anchor=unbound",
    "analysis_plan.population_empirical_validation=missing",
    "analysis_plan.population_uncertainty=unquantified",
    "analysis_plan.monetary_source_bundle_signature=missing",
    "analysis_plan.monetary_simulation_bridge=unvalidated",
    "analysis_plan.monetary_rate_uncertainty=unquantified",
    "analysis_plan.parameter_distributions=uncalibrated",
    "analysis_plan.execution_attestation=unverified",
)
_EXPLORATORY_PLAN_BLOCKERS: Final[tuple[str, ...]] = (
    "exploratory_plan.empirical_interpretation=prohibited",
    "exploratory_plan.external_registration=unregistered",
    "exploratory_plan.monetary_values=model_equivalent_not_observed",
    "exploratory_plan.monetary_rate_uncertainty=unquantified",
    "exploratory_plan.parameter_distributions=uncalibrated",
    "exploratory_plan.population_empirical_validation=missing",
    "exploratory_plan.population_generalization=prohibited",
    "exploratory_plan.population_uncertainty=unquantified",
    "exploratory_plan.production_campaign_authority=none",
    "exploratory_plan.real_world_causal_claims=prohibited",
)
_CANONICAL_INCLUSION_FIELDS: Final[tuple[PopulationInclusionField, ...]] = tuple(
    sorted(PopulationInclusionField, key=lambda item: item.value)
)


class AnalysisPlanValidationError(ValueError):
    """Raised when an analysis-plan declaration is malformed."""


class AnalysisPlanVerificationError(AnalysisPlanValidationError):
    """Raised when file or runtime evidence differs from a plan."""


class AnalysisPlanCampaignError(RuntimeError):
    """Raised because schema-v1 plans cannot claim campaign readiness."""

    def __init__(self, blockers: tuple[str, ...]) -> None:
        self.blockers = blockers
        super().__init__(
            "analysis plan is not campaign-ready: " + ", ".join(blockers)
        )


class AnalysisPlanRegistrationStatus(str, Enum):
    """External registration status supported by schema version 1."""

    UNREGISTERED = "UNREGISTERED"


class AnalysisEstimandRole(str, Enum):
    """Prospectively declared reporting role for an estimand."""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"


class PopulationMinorFilter(str, Enum):
    """Canonical minor-status restriction in an inclusion predicate."""

    ANY = "ANY"
    MINOR_ONLY = "MINOR_ONLY"
    ADULT_ONLY = "ADULT_ONLY"


class PopulationOutcomeMetric(str, Enum):
    """Whitelisted one-dimensional per-player policy outcomes."""

    SPENDING_CENTS = "spending_cents"
    HARMFUL_SPENDING_CENTS = "harmful_spending_cents"
    UNPLANNED_SPENDING_CENTS = "unplanned_spending_cents"
    MONETARY_HARM_PROXY_CENTS = "monetary_harm_proxy_cents"
    OPPORTUNITY_COST_PROXY_CENTS = "opportunity_cost_proxy_cents"
    ADULT_OPPORTUNITY_COST_PROXY_CENTS = "adult_opportunity_cost_proxy_cents"
    YOUTH_OPPORTUNITY_COST_PROXY_CENTS = "youth_opportunity_cost_proxy_cents"
    TOTAL_MONETARY_PROXY_CENTS = "total_monetary_proxy_cents"
    COMPOSITE_HARM = "composite_harm"
    MONETARY_HARM_SCORE = "monetary_harm_score"
    OPPORTUNITY_COST_SCORE = "opportunity_cost_score"
    SLEEP_BURDEN_SCORE = "sleep_burden_score"
    EDUCATION_WORK_BURDEN_SCORE = "education_work_burden_score"
    FAMILY_SOCIAL_BURDEN_SCORE = "family_social_burden_score"
    WELLBEING_BURDEN_SCORE = "wellbeing_burden_score"
    ENJOYMENT = "enjoyment"
    HIGH_RISK_INDICATOR = "high_risk_indicator"
    EXCESS_PLAY_MINUTES = "excess_play_minutes"
    DISPLACED_SLEEP_MINUTES = "displaced_sleep_minutes"
    DISPLACED_WORK_STUDY_MINUTES = "displaced_work_study_minutes"
    DISPLACED_SOCIAL_MINUTES = "displaced_social_minutes"
    DISPLACED_PHYSICAL_ACTIVITY_MINUTES = (
        "displaced_physical_activity_minutes"
    )


@dataclass(frozen=True, slots=True)
class PopulationOutcomeMetricSemantics:
    """Authoritative routing and unit semantics for a whitelisted outcome."""

    metric: PopulationOutcomeMetric
    metric_name: str
    result_path: str
    component_index: int | None
    metric_kind: PopulationMetricKind
    metric_scale: PopulationMetricScale
    storage_dtype: str
    unit: str

    def __post_init__(self) -> None:
        if type(self.metric) is not PopulationOutcomeMetric:
            raise TypeError("metric must be PopulationOutcomeMetric")
        for name in ("metric_name", "result_path", "storage_dtype", "unit"):
            _nonempty_text(getattr(self, name), name=name)
        if self.metric_name != self.metric.value:
            raise AnalysisPlanValidationError(
                "outcome metric_name must equal its canonical enum value"
            )
        if self.component_index is not None:
            _strict_int(
                self.component_index,
                name="outcome component_index",
                minimum=0,
                maximum=5,
            )
        if type(self.metric_kind) is not PopulationMetricKind:
            raise TypeError("metric_kind must be PopulationMetricKind")
        if type(self.metric_scale) is not PopulationMetricScale:
            raise TypeError("metric_scale must be PopulationMetricScale")
        if self.storage_dtype not in {"bool", "float64", "int64"}:
            raise AnalysisPlanValidationError(
                "outcome storage_dtype is not supported by schema v1"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "metric_name": self.metric_name,
            "result_path": self.result_path,
            "component_index": self.component_index,
            "metric_kind": self.metric_kind.value,
            "metric_scale": self.metric_scale.value,
            "storage_dtype": self.storage_dtype,
            "unit": self.unit,
        }


def _outcome_semantics_registry() -> Mapping[
    PopulationOutcomeMetric,
    PopulationOutcomeMetricSemantics,
]:
    money = PopulationMetricKind.MONEY_MINOR_UNITS
    additive = PopulationMetricScale.ADDITIVE_PER_ANALYSIS_UNIT
    nonadditive = PopulationMetricScale.NONADDITIVE
    score = PopulationMetricKind.SCORE
    time = PopulationMetricKind.TIME

    def item(
        metric: PopulationOutcomeMetric,
        result_path: str,
        *,
        component_index: int | None = None,
        metric_kind: PopulationMetricKind,
        metric_scale: PopulationMetricScale,
        storage_dtype: str,
        unit: str,
    ) -> PopulationOutcomeMetricSemantics:
        return PopulationOutcomeMetricSemantics(
            metric=metric,
            metric_name=metric.value,
            result_path=result_path,
            component_index=component_index,
            metric_kind=metric_kind,
            metric_scale=metric_scale,
            storage_dtype=storage_dtype,
            unit=unit,
        )

    entries = {
        PopulationOutcomeMetric.SPENDING_CENTS: item(
            PopulationOutcomeMetric.SPENDING_CENTS,
            "PolicyScenarioResult.spending_cents",
            metric_kind=money,
            metric_scale=additive,
            storage_dtype="int64",
            unit="currency_minor_unit",
        ),
        PopulationOutcomeMetric.HARMFUL_SPENDING_CENTS: item(
            PopulationOutcomeMetric.HARMFUL_SPENDING_CENTS,
            "PolicyScenarioResult.harm.harmful_spending_cents",
            metric_kind=money,
            metric_scale=additive,
            storage_dtype="int64",
            unit="currency_minor_unit",
        ),
        PopulationOutcomeMetric.UNPLANNED_SPENDING_CENTS: item(
            PopulationOutcomeMetric.UNPLANNED_SPENDING_CENTS,
            "PolicyScenarioResult.harm.unplanned_spending_cents",
            metric_kind=money,
            metric_scale=additive,
            storage_dtype="int64",
            unit="currency_minor_unit",
        ),
        PopulationOutcomeMetric.MONETARY_HARM_PROXY_CENTS: item(
            PopulationOutcomeMetric.MONETARY_HARM_PROXY_CENTS,
            "PolicyScenarioResult.harm.monetary_harm_proxy_cents",
            metric_kind=money,
            metric_scale=additive,
            storage_dtype="int64",
            unit="currency_minor_unit",
        ),
        PopulationOutcomeMetric.OPPORTUNITY_COST_PROXY_CENTS: item(
            PopulationOutcomeMetric.OPPORTUNITY_COST_PROXY_CENTS,
            "PolicyScenarioResult.harm.opportunity_cost_proxy_cents",
            metric_kind=money,
            metric_scale=additive,
            storage_dtype="int64",
            unit="currency_minor_unit",
        ),
        PopulationOutcomeMetric.ADULT_OPPORTUNITY_COST_PROXY_CENTS: item(
            PopulationOutcomeMetric.ADULT_OPPORTUNITY_COST_PROXY_CENTS,
            "PolicyScenarioResult.harm.adult_opportunity_cost_proxy_cents",
            metric_kind=money,
            metric_scale=additive,
            storage_dtype="int64",
            unit="currency_minor_unit",
        ),
        PopulationOutcomeMetric.YOUTH_OPPORTUNITY_COST_PROXY_CENTS: item(
            PopulationOutcomeMetric.YOUTH_OPPORTUNITY_COST_PROXY_CENTS,
            "PolicyScenarioResult.harm.youth_opportunity_cost_proxy_cents",
            metric_kind=money,
            metric_scale=additive,
            storage_dtype="int64",
            unit="currency_minor_unit",
        ),
        PopulationOutcomeMetric.TOTAL_MONETARY_PROXY_CENTS: item(
            PopulationOutcomeMetric.TOTAL_MONETARY_PROXY_CENTS,
            "PolicyScenarioResult.harm.total_monetary_proxy_cents",
            metric_kind=money,
            metric_scale=additive,
            storage_dtype="int64",
            unit="currency_minor_unit",
        ),
        PopulationOutcomeMetric.COMPOSITE_HARM: item(
            PopulationOutcomeMetric.COMPOSITE_HARM,
            "PolicyScenarioResult.composite_harm",
            metric_kind=score,
            metric_scale=nonadditive,
            storage_dtype="float64",
            unit="model_score",
        ),
        PopulationOutcomeMetric.ENJOYMENT: item(
            PopulationOutcomeMetric.ENJOYMENT,
            "PolicyScenarioResult.enjoyment",
            metric_kind=score,
            metric_scale=nonadditive,
            storage_dtype="float64",
            unit="model_score",
        ),
        PopulationOutcomeMetric.HIGH_RISK_INDICATOR: item(
            PopulationOutcomeMetric.HIGH_RISK_INDICATOR,
            "PolicyScenarioResult.high_risk",
            metric_kind=PopulationMetricKind.RATIO,
            metric_scale=nonadditive,
            storage_dtype="bool",
            unit="indicator",
        ),
    }
    component_metrics = (
        PopulationOutcomeMetric.MONETARY_HARM_SCORE,
        PopulationOutcomeMetric.OPPORTUNITY_COST_SCORE,
        PopulationOutcomeMetric.SLEEP_BURDEN_SCORE,
        PopulationOutcomeMetric.EDUCATION_WORK_BURDEN_SCORE,
        PopulationOutcomeMetric.FAMILY_SOCIAL_BURDEN_SCORE,
        PopulationOutcomeMetric.WELLBEING_BURDEN_SCORE,
    )
    for index, metric in enumerate(component_metrics):
        entries[metric] = item(
            metric,
            "PolicyScenarioResult.harm.component_scores",
            component_index=index,
            metric_kind=score,
            metric_scale=nonadditive,
            storage_dtype="float64",
            unit="model_score",
        )
    minute_metrics = {
        PopulationOutcomeMetric.EXCESS_PLAY_MINUTES: "excess_play_minutes",
        PopulationOutcomeMetric.DISPLACED_SLEEP_MINUTES: (
            "displaced_sleep_minutes"
        ),
        PopulationOutcomeMetric.DISPLACED_WORK_STUDY_MINUTES: (
            "displaced_work_study_minutes"
        ),
        PopulationOutcomeMetric.DISPLACED_SOCIAL_MINUTES: (
            "displaced_social_minutes"
        ),
        PopulationOutcomeMetric.DISPLACED_PHYSICAL_ACTIVITY_MINUTES: (
            "displaced_physical_activity_minutes"
        ),
    }
    for metric, attribute in minute_metrics.items():
        entries[metric] = item(
            metric,
            f"PolicyScenarioResult.harm.{attribute}",
            metric_kind=time,
            metric_scale=additive,
            storage_dtype="float64",
            unit="minute",
        )
    if set(entries) != set(PopulationOutcomeMetric):
        raise RuntimeError("population outcome registry is incomplete")
    return MappingProxyType(entries)


def population_outcome_semantics(
    metric: PopulationOutcomeMetric,
) -> PopulationOutcomeMetricSemantics:
    """Return immutable routing semantics for one whitelisted metric."""

    if type(metric) is not PopulationOutcomeMetric:
        raise TypeError("metric must be PopulationOutcomeMetric")
    return _POPULATION_OUTCOME_SEMANTICS[metric]


@dataclass(frozen=True, slots=True)
class CanonicalPopulationInclusionPredicate:
    """Executable pre-treatment predicate over projected joint-population cells.

    Empty categorical tuples mean "all declared values".  Unlike the reusable
    :class:`PopulationInclusionRule`, this object carries the canonical filters
    and can evaluate them against an attested projected assignment.
    """

    rule: PopulationInclusionRule
    jurisdiction_codes: tuple[str, ...]
    age_min_inclusive: int
    age_max_exclusive: int
    minor_filter: PopulationMinorFilter
    monthly_disposable_income_band_ids: tuple[str, ...]
    household_type_ids: tuple[str, ...]
    gaming_states: tuple[PopulationGamingState, ...]
    payer_history_states: tuple[PopulationPayerHistoryState, ...]

    def __post_init__(self) -> None:
        if type(self.rule) is not PopulationInclusionRule:
            raise TypeError("inclusion rule must be PopulationInclusionRule")
        PopulationInclusionRule.__post_init__(self.rule)
        if self.rule.source_fields != _CANONICAL_INCLUSION_FIELDS:
            raise AnalysisPlanValidationError(
                "analysis inclusion rule must declare every canonical pre-treatment field"
            )
        if self.rule.timing is not PopulationInclusionTiming.PRETREATMENT:
            raise AnalysisPlanValidationError(
                "analysis inclusion predicate must be pre-treatment"
            )
        if self.rule.evidence_role is not PopulationEstimandRole.CALIBRATION:
            raise AnalysisPlanValidationError(
                "validation evidence cannot define analysis inclusion"
            )
        _strict_int(
            self.age_min_inclusive,
            name="inclusion age_min_inclusive",
            minimum=0,
            maximum=32_767,
        )
        _strict_int(
            self.age_max_exclusive,
            name="inclusion age_max_exclusive",
            minimum=1,
            maximum=32_768,
        )
        if self.age_min_inclusive >= self.age_max_exclusive:
            raise AnalysisPlanValidationError(
                "inclusion age interval must be non-empty"
            )
        if type(self.minor_filter) is not PopulationMinorFilter:
            raise TypeError("minor_filter must be PopulationMinorFilter")
        _canonical_code_tuple(self.jurisdiction_codes)
        _canonical_identifier_tuple(
            self.monthly_disposable_income_band_ids,
            name="monthly_disposable_income_band_ids",
        )
        if (
            ALL_MONTHLY_DISPOSABLE_INCOME_BANDS_ID
            in self.monthly_disposable_income_band_ids
            and self.monthly_disposable_income_band_ids
            != (ALL_MONTHLY_DISPOSABLE_INCOME_BANDS_ID,)
        ):
            raise AnalysisPlanValidationError(
                "the canonical all-income selector must be used alone"
            )
        _canonical_identifier_tuple(
            self.household_type_ids,
            name="household_type_ids",
        )
        if (
            ALL_HOUSEHOLD_TYPES_ID in self.household_type_ids
            and self.household_type_ids != (ALL_HOUSEHOLD_TYPES_ID,)
        ):
            raise AnalysisPlanValidationError(
                "the canonical all-household selector must be used alone"
            )
        _canonical_enum_tuple(
            self.gaming_states,
            enum_type=PopulationGamingState,
            name="gaming_states",
        )
        _canonical_enum_tuple(
            self.payer_history_states,
            enum_type=PopulationPayerHistoryState,
            name="payer_history_states",
        )

    @property
    def predicate_sha256(self) -> str:
        return _canonical_sha256(self.snapshot())

    def snapshot(self) -> dict[str, object]:
        return {
            "rule": self.rule.snapshot(),
            "jurisdiction_codes": list(self.jurisdiction_codes),
            "age_min_inclusive": self.age_min_inclusive,
            "age_max_exclusive": self.age_max_exclusive,
            "minor_filter": self.minor_filter.value,
            "monthly_disposable_income_band_ids": list(
                self.monthly_disposable_income_band_ids
            ),
            "household_type_ids": list(self.household_type_ids),
            "gaming_states": [item.value for item in self.gaming_states],
            "payer_history_states": [
                item.value for item in self.payer_history_states
            ],
        }

    def evaluate(
        self,
        *,
        jurisdiction_codes: tuple[str, ...],
        jurisdiction: npt.NDArray[np.int16],
        age_years: npt.NDArray[np.int16],
        is_minor: npt.NDArray[np.bool_],
        projected_cells: tuple[ProjectedPopulationCellMetadata, ...],
        cell_indices: tuple[int, ...],
    ) -> npt.NDArray[np.bool_]:
        return evaluate_population_inclusion(
            self,
            jurisdiction_codes=jurisdiction_codes,
            jurisdiction=jurisdiction,
            age_years=age_years,
            is_minor=is_minor,
            projected_cells=projected_cells,
            cell_indices=cell_indices,
        )


def evaluate_population_inclusion(
    predicate: CanonicalPopulationInclusionPredicate,
    *,
    jurisdiction_codes: tuple[str, ...],
    jurisdiction: npt.NDArray[np.int16],
    age_years: npt.NDArray[np.int16],
    is_minor: npt.NDArray[np.bool_],
    projected_cells: tuple[ProjectedPopulationCellMetadata, ...],
    cell_indices: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Evaluate one canonical predicate over exact pre-treatment memberships."""

    if type(predicate) is not CanonicalPopulationInclusionPredicate:
        raise TypeError(
            "predicate must be CanonicalPopulationInclusionPredicate"
        )
    CanonicalPopulationInclusionPredicate.__post_init__(predicate)
    _runtime_jurisdiction_codes(jurisdiction_codes)
    expected_arrays = (
        (jurisdiction, np.dtype(np.int16), "jurisdiction"),
        (age_years, np.dtype(np.int16), "age_years"),
        (is_minor, np.dtype(np.bool_), "is_minor"),
    )
    size: int | None = None
    for values, expected_dtype, name in expected_arrays:
        if (
            type(values) is not np.ndarray
            or values.ndim != 1
            or values.dtype != expected_dtype
        ):
            raise TypeError(
                f"{name} must be a one-dimensional {expected_dtype.name} array"
            )
        if size is None:
            size = int(values.size)
        elif values.size != size:
            raise AnalysisPlanValidationError(
                "pre-treatment inclusion arrays must have equal length"
            )
    assert size is not None
    if type(projected_cells) is not tuple or not projected_cells or any(
        type(cell) is not ProjectedPopulationCellMetadata
        for cell in projected_cells
    ):
        raise TypeError(
            "projected_cells must be a non-empty exact tuple of cell metadata"
        )
    for cell in projected_cells:
        ProjectedPopulationCellMetadata.__post_init__(cell)
    if type(cell_indices) is not tuple or any(
        type(index) is not int for index in cell_indices
    ):
        raise TypeError("cell_indices must be an exact tuple of Python integers")
    if len(cell_indices) != size:
        raise AnalysisPlanValidationError(
            "cell_indices must contain one entry per player"
        )
    indices = np.asarray(cell_indices, dtype=np.int64)
    if indices.size and (
        np.any(indices < 0) or np.any(indices >= len(projected_cells))
    ):
        raise AnalysisPlanValidationError("cell_indices contain an unknown cell")
    if jurisdiction.size and (
        np.any(jurisdiction < 0)
        or np.any(jurisdiction >= len(jurisdiction_codes))
    ):
        raise AnalysisPlanValidationError(
            "jurisdiction indices fall outside jurisdiction_codes"
        )
    selected_cells = np.zeros(len(projected_cells), dtype=np.bool_)
    all_income_bands = predicate.monthly_disposable_income_band_ids == (
        ALL_MONTHLY_DISPOSABLE_INCOME_BANDS_ID,
    )
    all_household_types = predicate.household_type_ids == (
        ALL_HOUSEHOLD_TYPES_ID,
    )
    for cell_index, cell in enumerate(projected_cells):
        if not 0 <= cell.jurisdiction_index < len(jurisdiction_codes):
            raise AnalysisPlanVerificationError(
                "projected cell has an unknown jurisdiction index"
            )
        if jurisdiction_codes[cell.jurisdiction_index] != cell.jurisdiction_code:
            raise AnalysisPlanVerificationError(
                "projected cell jurisdiction code/index is inconsistent"
            )
        positions = indices == cell_index
        if np.any(jurisdiction[positions] != cell.jurisdiction_index):
            raise AnalysisPlanVerificationError(
                "player jurisdiction differs from projected joint cell"
            )
        if np.any(age_years[positions] < cell.age_min_inclusive) or np.any(
            age_years[positions] >= cell.age_max_exclusive
        ):
            raise AnalysisPlanVerificationError(
                "player age differs from projected joint cell"
            )
        selected_cells[cell_index] = all(
            (
                not predicate.jurisdiction_codes
                or cell.jurisdiction_code in predicate.jurisdiction_codes,
                not predicate.monthly_disposable_income_band_ids
                or all_income_bands
                or cell.monthly_disposable_income_band_id
                in predicate.monthly_disposable_income_band_ids,
                not predicate.household_type_ids
                or all_household_types
                or cell.household_type in predicate.household_type_ids,
                not predicate.gaming_states
                or _gaming_state(cell.baseline_gamer) in predicate.gaming_states,
                not predicate.payer_history_states
                or _payer_state(cell.baseline_ever_payer)
                in predicate.payer_history_states,
            )
        )

    mask = selected_cells[indices] if indices.size else np.zeros(0, dtype=np.bool_)
    mask &= age_years >= predicate.age_min_inclusive
    mask &= age_years < predicate.age_max_exclusive
    if predicate.minor_filter is PopulationMinorFilter.MINOR_ONLY:
        mask &= is_minor
    elif predicate.minor_filter is PopulationMinorFilter.ADULT_ONLY:
        mask &= ~is_minor
    immutable = np.array(mask, dtype=np.bool_, copy=True)
    immutable.setflags(write=False)
    return immutable


@dataclass(frozen=True, slots=True)
class FixedSeedStoppingRule:
    """Outcome-blind rule requiring exactly one declared fixed seed set."""

    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.seeds) is not tuple or not self.seeds:
            raise TypeError("fixed-seed stopping rule requires a non-empty tuple")
        observed = tuple(
            validate_seed(seed, name=f"stopping seeds[{index}]")
            for index, seed in enumerate(self.seeds)
        )
        if len(set(observed)) != len(observed):
            raise AnalysisPlanValidationError("stopping seeds must be unique")
        if observed != tuple(sorted(observed)):
            raise AnalysisPlanValidationError(
                "stopping seeds must use ascending canonical order"
            )

    @property
    def rule_id(self) -> str:
        return "FIXED_SEED_SET_V1"

    def snapshot(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "seeds": list(self.seeds),
            "seed_decimal_strings": [str(seed) for seed in self.seeds],
            "seed_count": len(self.seeds),
            "seed_count_decimal": str(len(self.seeds)),
            "early_stopping_allowed": False,
            "treatment_result_interim_looks_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class PrimaryAggregateRule:
    """Outcome-blind cross-seed rule for the single PRIMARY estimand.

    Schema v2 deliberately permits no result-dependent exclusions.  Every
    fixed seed must supply one finite, exact, paired primary realization; a
    missing or invalid realization is an error instead of a reason to retain a
    more favourable subset.
    """

    positive_result_interpretation: str
    negative_result_interpretation: str

    def __post_init__(self) -> None:
        _nonempty_text(
            self.positive_result_interpretation,
            name="positive_result_interpretation",
        )
        _nonempty_text(
            self.negative_result_interpretation,
            name="negative_result_interpretation",
        )
        if (
            self.positive_result_interpretation
            == self.negative_result_interpretation
        ):
            raise AnalysisPlanValidationError(
                "positive and negative result interpretations must differ"
            )

    @property
    def rule_id(self) -> str:
        return "COMPLETE_FIXED_SEED_PRIMARY_PAIRS_V1"

    def snapshot(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "analysis_unit": "INDEPENDENT_MONTE_CARLO_SEED",
            "primary_estimand_selection": "EXACTLY_ONE_DECLARED_PRIMARY",
            "scenario_aggregation": "SINGLE_DIRECTED_CONTRAST",
            "scenario_weights": [],
            "population_weight_application": (
                "WITHIN_EACH_SEED_BEFORE_CROSS_SEED_AGGREGATION"
            ),
            "seed_weighting": "EQUAL",
            "valid_realization_criteria": [
                "seed_is_in_fixed_stopping_rule",
                "paired_reference_and_comparison_observations_are_present",
                "predeclared_pretreatment_population_predicate_is_applied",
                "population_weighted_primary_estimand_is_exact_and_finite",
            ],
            "exclusion_criteria": [],
            "invalid_or_missing_realization_handling": "FAIL_CLOSED",
            "outcome_dependent_exclusion_allowed": False,
            "point_estimator": "ARITHMETIC_MEAN_OF_RETAINED_SEED_ESTIMANDS",
            "between_seed_standard_deviation": (
                "SAMPLE_STANDARD_DEVIATION_DDOF_1_ZERO_IF_ONE_SEED"
            ),
            "monte_carlo_standard_error": (
                "BETWEEN_SEED_SD_DIVIDED_BY_SQRT_RETAINED_SEED_COUNT"
            ),
            "interval_method": "NORMAL_95_MONTE_CARLO_MEAN_PLUS_MINUS_1.96_MCSE",
            "one_seed_interval": "ZERO_WIDTH_AT_POINT_ESTIMATE",
            "interval_interpretation": (
                "Monte Carlo variability of the configured simulator mean; "
                "not a confidence interval for a real-world population"
            ),
            "positive_result_interpretation": (
                self.positive_result_interpretation
            ),
            "negative_result_interpretation": (
                self.negative_result_interpretation
            ),
        }


@dataclass(frozen=True, slots=True)
class PlannedPopulationEstimand:
    """One directed, population-weighted per-player scenario contrast."""

    estimand_id: str
    role: AnalysisEstimandRole
    reference_scenario_id: ScenarioId
    comparison_scenario_id: ScenarioId
    outcome_metric: PopulationOutcomeMetric
    metric_contract_id: str
    inclusion_predicate: CanonicalPopulationInclusionPredicate
    period: PopulationPeriodSemantics
    currency: PopulationCurrencySemantics | None = None

    def __post_init__(self) -> None:
        _identifier(self.estimand_id, name="estimand_id")
        if type(self.role) is not AnalysisEstimandRole:
            raise TypeError("estimand role must be AnalysisEstimandRole")
        if type(self.reference_scenario_id) is not ScenarioId:
            raise TypeError("reference_scenario_id must be ScenarioId")
        if type(self.comparison_scenario_id) is not ScenarioId:
            raise TypeError("comparison_scenario_id must be ScenarioId")
        if self.reference_scenario_id is self.comparison_scenario_id:
            raise AnalysisPlanValidationError(
                "estimand reference and comparison scenarios must differ"
            )
        if type(self.outcome_metric) is not PopulationOutcomeMetric:
            raise TypeError("outcome_metric must be PopulationOutcomeMetric")
        _contract_identifier(self.metric_contract_id, name="metric_contract_id")
        if type(self.inclusion_predicate) is not CanonicalPopulationInclusionPredicate:
            raise TypeError(
                "inclusion_predicate must be CanonicalPopulationInclusionPredicate"
            )
        CanonicalPopulationInclusionPredicate.__post_init__(
            self.inclusion_predicate
        )
        if type(self.period) is not PopulationPeriodSemantics:
            raise TypeError("period must be PopulationPeriodSemantics")
        PopulationPeriodSemantics.__post_init__(self.period)
        semantics = population_outcome_semantics(self.outcome_metric)
        if semantics.metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS:
            if type(self.currency) is not PopulationCurrencySemantics:
                raise AnalysisPlanValidationError(
                    "money outcome estimands require currency semantics"
                )
            PopulationCurrencySemantics.__post_init__(self.currency)
        elif self.currency is not None:
            raise AnalysisPlanValidationError(
                "non-money outcome estimands cannot declare currency semantics"
            )

    @property
    def contrast_direction(self) -> str:
        return "COMPARISON_MINUS_REFERENCE"

    @property
    def algorithm(self) -> PopulationEstimandAlgorithm:
        return PopulationEstimandAlgorithm.PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1

    @property
    def normalization(self) -> PopulationNormalization:
        return PopulationNormalization.DIVIDE_BY_WEIGHT_SUM

    @property
    def estimand_sha256(self) -> str:
        return _canonical_sha256(self.snapshot())

    @property
    def specification_sha256(self) -> str:
        """Identity excluding the reporting label and PRIMARY/SECONDARY role."""

        payload = self.snapshot()
        del payload["estimand_id"]
        del payload["role"]
        return _canonical_sha256(payload)

    @property
    def outcome_semantics(self) -> PopulationOutcomeMetricSemantics:
        return population_outcome_semantics(self.outcome_metric)

    def snapshot(self) -> dict[str, object]:
        return {
            "estimand_id": self.estimand_id,
            "role": self.role.value,
            "reference_scenario_id": self.reference_scenario_id.value,
            "comparison_scenario_id": self.comparison_scenario_id.value,
            "contrast_direction": self.contrast_direction,
            "outcome": self.outcome_semantics.snapshot(),
            "metric_contract_id": self.metric_contract_id,
            "inclusion_predicate": self.inclusion_predicate.snapshot(),
            "period": self.period.snapshot(),
            "currency": (
                self.currency.snapshot() if self.currency is not None else None
            ),
            "algorithm": self.algorithm.value,
            "normalization": self.normalization.value,
        }


@dataclass(frozen=True, slots=True)
class ProspectiveAnalysisPlan:
    """Immutable semantic plan plus its self-attested canonical digest."""

    schema_version: str
    plan_id: str
    expected_causal_design_sha256: str
    expected_batch_spec_sha256: str
    expected_model_inputs_sha256: str
    expected_population_input_sha256: str
    expected_profile_input_sha256: str
    expected_metric_contract_sha256: str
    expected_harm_weights_sha256: str
    expected_output_profile_sha256: str
    stopping_rule: FixedSeedStoppingRule
    estimands: tuple[PlannedPopulationEstimand, ...]
    plan_sha256: str
    declared_harm_weights: WelfareHarmWeights | None = None
    primary_aggregate_rule: PrimaryAggregateRule | None = None
    amendment_json: str | None = None
    registration_status: AnalysisPlanRegistrationStatus = field(
        default=AnalysisPlanRegistrationStatus.UNREGISTERED,
        init=False,
    )
    preregistered: bool = field(default=False, init=False)
    campaign_ready: bool = field(default=False, init=False)
    campaign_blockers: tuple[str, ...] = field(
        default=(),
        init=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version not in {
            ANALYSIS_PLAN_SCHEMA_VERSION,
            PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION,
            CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION,
        }:
            raise AnalysisPlanValidationError(
                "unsupported prospective analysis-plan schema version"
            )
        object.__setattr__(
            self,
            "campaign_blockers",
            _campaign_blockers(self.schema_version),
        )
        _identifier(self.plan_id, name="plan_id")
        for name in (
            "expected_causal_design_sha256",
            "expected_batch_spec_sha256",
            "expected_model_inputs_sha256",
            "expected_population_input_sha256",
            "expected_profile_input_sha256",
            "expected_metric_contract_sha256",
            "expected_harm_weights_sha256",
            "expected_output_profile_sha256",
            "plan_sha256",
        ):
            _sha256_digest(getattr(self, name), name=name)
        if type(self.stopping_rule) is not FixedSeedStoppingRule:
            raise TypeError("stopping_rule must be FixedSeedStoppingRule")
        FixedSeedStoppingRule.__post_init__(self.stopping_rule)
        if type(self.estimands) is not tuple or not self.estimands or any(
            type(estimand) is not PlannedPopulationEstimand
            for estimand in self.estimands
        ):
            raise TypeError(
                "estimands must be a non-empty exact tuple of planned estimands"
            )
        for estimand in self.estimands:
            PlannedPopulationEstimand.__post_init__(estimand)
        ids = tuple(estimand.estimand_id for estimand in self.estimands)
        if len(set(ids)) != len(ids):
            raise AnalysisPlanValidationError("estimand IDs must be unique")
        if ids != tuple(sorted(ids)):
            raise AnalysisPlanValidationError(
                "estimands must use ascending estimand_id order"
            )
        hashes = tuple(
            estimand.specification_sha256 for estimand in self.estimands
        )
        if len(set(hashes)) != len(hashes):
            raise AnalysisPlanValidationError(
                "estimands must have unique semantic specifications"
            )
        primary_count = sum(
            estimand.role is AnalysisEstimandRole.PRIMARY
            for estimand in self.estimands
        )
        if primary_count != 1:
            raise AnalysisPlanValidationError(
                "analysis plan must declare exactly one PRIMARY estimand"
            )
        if self.schema_version == ANALYSIS_PLAN_SCHEMA_VERSION:
            if (
                self.declared_harm_weights is not None
                or self.primary_aggregate_rule is not None
                or self.amendment_json is not None
            ):
                raise AnalysisPlanValidationError(
                    "schema-v1 plans cannot declare a plan-level aggregate"
                )
        else:
            if type(self.declared_harm_weights) is not WelfareHarmWeights:
                raise TypeError(
                    "schema-v2 declared_harm_weights must be WelfareHarmWeights"
                )
            WelfareHarmWeights.__post_init__(self.declared_harm_weights)
            if (
                analysis_plan_harm_weights_sha256(
                    self.declared_harm_weights
                )
                != self.expected_harm_weights_sha256
            ):
                raise AnalysisPlanValidationError(
                    "declared harm-weight vector differs from its expected digest"
                )
            if type(self.primary_aggregate_rule) is not PrimaryAggregateRule:
                raise TypeError(
                    "schema-v2 primary_aggregate_rule must be PrimaryAggregateRule"
                )
            PrimaryAggregateRule.__post_init__(self.primary_aggregate_rule)
            if self.schema_version == PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION:
                if self.amendment_json is not None:
                    raise AnalysisPlanValidationError(
                        "schema-v2 plans cannot contain a campaign amendment"
                    )
            else:
                if type(self.amendment_json) is not str:
                    raise TypeError(
                        "schema-v3 plans require a canonical amendment payload"
                    )
                amendment = _parse_amendment_json(self.amendment_json)
                _validate_campaign_amendment(amendment)
                scientific_change = _required_mapping(
                    amendment,
                    "scientific_change",
                )
                if (
                    scientific_change["current_estimand_id"]
                    != self.primary_estimand.estimand_id
                    or scientific_change["current_specification_sha256"]
                    != self.primary_estimand.specification_sha256
                ):
                    raise AnalysisPlanValidationError(
                        "amendment current primary estimand identity differs "
                        "from the successor plan"
                    )
                if len(self.stopping_rule.seeds) < 100:
                    raise AnalysisPlanValidationError(
                        "schema-v3 campaign plans require at least 100 fixed seeds"
                    )
        if self.registration_status is not AnalysisPlanRegistrationStatus.UNREGISTERED:
            raise AnalysisPlanValidationError(
                "schema-v1 analysis plans must remain UNREGISTERED"
            )
        if self.preregistered or self.campaign_ready:
            raise AnalysisPlanValidationError(
                "schema-v1 analysis plans cannot be preregistered or campaign-ready"
            )
        if self.campaign_blockers != _campaign_blockers(self.schema_version):
            raise AnalysisPlanValidationError(
                "analysis-plan campaign blockers are fixed by schema version"
            )
        if self.plan_sha256 != _canonical_sha256(self.attestation_payload()):
            raise AnalysisPlanValidationError(
                "plan_sha256 does not match the canonical plan payload"
            )

    @property
    def primary_estimand(self) -> PlannedPopulationEstimand:
        return next(
            estimand
            for estimand in self.estimands
            if estimand.role is AnalysisEstimandRole.PRIMARY
        )

    @property
    def amendment(self) -> dict[str, object] | None:
        """Return a detached copy of the canonical schema-v3 amendment."""

        return (
            _parse_amendment_json(self.amendment_json)
            if self.amendment_json is not None
            else None
        )

    def attestation_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "expected_causal_design_sha256": (
                self.expected_causal_design_sha256
            ),
            "expected_batch_spec_sha256": self.expected_batch_spec_sha256,
            "expected_model_inputs_sha256": self.expected_model_inputs_sha256,
            "expected_population_input_sha256": (
                self.expected_population_input_sha256
            ),
            "expected_profile_input_sha256": (
                self.expected_profile_input_sha256
            ),
            "expected_metric_contract_sha256": (
                self.expected_metric_contract_sha256
            ),
            "expected_harm_weights_sha256": self.expected_harm_weights_sha256,
            "expected_output_profile_sha256": (
                self.expected_output_profile_sha256
            ),
            "stopping_rule": self.stopping_rule.snapshot(),
            "estimands": [estimand.snapshot() for estimand in self.estimands],
            "registration_status": self.registration_status.value,
            "preregistered": self.preregistered,
            "campaign_ready": self.campaign_ready,
            "campaign_blockers": list(self.campaign_blockers),
        }
        if self.schema_version in {
            PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION,
            CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION,
        }:
            assert self.declared_harm_weights is not None
            assert self.primary_aggregate_rule is not None
            payload["declared_harm_weights"] = _harm_weights_snapshot(
                self.declared_harm_weights
            )
            payload["primary_aggregate_rule"] = (
                self.primary_aggregate_rule.snapshot()
            )
        if self.schema_version == CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION:
            assert self.amendment_json is not None
            payload["amendment"] = _parse_amendment_json(self.amendment_json)
        return payload

    def snapshot(self) -> dict[str, object]:
        return {**self.attestation_payload(), "plan_sha256": self.plan_sha256}

    def validate_for_campaign(self) -> None:
        raise AnalysisPlanCampaignError(self.campaign_blockers)


@dataclass(frozen=True, slots=True)
class LoadedProspectiveAnalysisPlan:
    """A semantic plan bound to the exact regular-file bytes that supplied it."""

    plan_path: Path
    byte_length: int
    file_sha256: str
    semantic_sha256: str
    plan: ProspectiveAnalysisPlan

    def __post_init__(self) -> None:
        if not isinstance(self.plan_path, Path) or not self.plan_path.is_absolute():
            raise TypeError("plan_path must be an absolute Path")
        lexical_path = Path(os.path.normpath(os.fspath(self.plan_path)))
        if ".." in self.plan_path.parts or lexical_path != self.plan_path:
            raise AnalysisPlanValidationError(
                "plan_path must be lexically canonical"
            )
        _strict_int(
            self.byte_length,
            name="analysis plan byte_length",
            minimum=1,
            maximum=MAX_ANALYSIS_PLAN_BYTES,
        )
        _sha256_digest(self.file_sha256, name="analysis plan file_sha256")
        _sha256_digest(
            self.semantic_sha256,
            name="analysis plan semantic_sha256",
        )
        if type(self.plan) is not ProspectiveAnalysisPlan:
            raise TypeError("plan must be ProspectiveAnalysisPlan")
        ProspectiveAnalysisPlan.__post_init__(self.plan)
        if self.semantic_sha256 != self.plan.plan_sha256:
            raise AnalysisPlanValidationError(
                "loaded semantic digest differs from the plan digest"
            )
        observed = _read_regular_file(self.plan_path)
        if (
            len(observed) != self.byte_length
            or sha256(observed).hexdigest() != self.file_sha256
        ):
            raise AnalysisPlanVerificationError(
                "analysis plan file changed: loaded metadata differ from its exact bytes"
            )
        observed_plan = _plan_from_snapshot(_parse_json_object(observed))
        if observed_plan != self.plan:
            raise AnalysisPlanVerificationError(
                "loaded analysis-plan object differs from its exact file declaration"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.plan.schema_version,
            "plan_path": str(self.plan_path),
            "byte_length": self.byte_length,
            "file_sha256": self.file_sha256,
            "semantic_sha256": self.semantic_sha256,
            "plan": self.plan.snapshot(),
        }

    def manifest_payload(self) -> dict[str, object]:
        """Return the complete immutable file and semantic plan attestation."""

        return self.snapshot()


@dataclass(frozen=True, slots=True)
class ExploratoryAnalysisPlan:
    """Strict, content-addressed plan for non-empirical model exploration.

    This is deliberately a separate artifact type rather than a campaign-plan
    status.  Its schema permanently prohibits preregistration, empirical or
    real-world causal interpretation, population generalization, observed-
    spending labels, and production-campaign authorization.
    """

    identity_payload_json: str
    plan_sha256: str

    def __post_init__(self) -> None:
        if type(self.identity_payload_json) is not str:
            raise TypeError("exploratory identity_payload_json must be text")
        try:
            payload = json.loads(self.identity_payload_json)
        except (TypeError, ValueError) as exc:
            raise AnalysisPlanValidationError(
                "exploratory plan identity is not valid JSON"
            ) from exc
        if type(payload) is not dict:
            raise AnalysisPlanValidationError(
                "exploratory plan identity must be a JSON object"
            )
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if canonical != self.identity_payload_json:
            raise AnalysisPlanValidationError(
                "exploratory identity payload must use canonical JSON"
            )
        _validate_exploratory_plan_payload(payload)
        _sha256_digest(self.plan_sha256, name="exploratory plan_sha256")
        if self.plan_sha256 != _canonical_sha256(payload):
            raise AnalysisPlanValidationError(
                "exploratory plan_sha256 differs from its canonical identity"
            )

    @property
    def identity_payload(self) -> dict[str, object]:
        return json.loads(self.identity_payload_json)

    @property
    def schema_version(self) -> str:
        return EXPLORATORY_ANALYSIS_PLAN_SCHEMA_VERSION

    @property
    def plan_id(self) -> str:
        return str(self.identity_payload["plan_id"])

    @property
    def primary_estimand(self) -> PlannedPopulationEstimand:
        scientific = _required_mapping(
            self.identity_payload,
            "scientific_estimand",
        )
        return _estimand_from_snapshot(
            _required_mapping(scientific, "estimand")
        )

    @property
    def stopping_rule(self) -> FixedSeedStoppingRule:
        return _stopping_from_snapshot(
            _required_mapping(self.identity_payload, "stopping_rule")
        )

    @property
    def preregistered(self) -> bool:
        return False

    @property
    def campaign_ready(self) -> bool:
        return False

    @property
    def campaign_blockers(self) -> tuple[str, ...]:
        return _EXPLORATORY_PLAN_BLOCKERS

    def snapshot(self) -> dict[str, object]:
        return {**self.identity_payload, "plan_sha256": self.plan_sha256}

    def validate_for_campaign(self) -> None:
        raise AnalysisPlanCampaignError(self.campaign_blockers)


@dataclass(frozen=True, slots=True)
class LoadedExploratoryAnalysisPlan:
    """Exact file-byte and semantic identity for an exploratory plan."""

    plan_path: Path
    byte_length: int
    file_sha256: str
    semantic_sha256: str
    plan: ExploratoryAnalysisPlan

    def __post_init__(self) -> None:
        if not isinstance(self.plan_path, Path) or not self.plan_path.is_absolute():
            raise TypeError("exploratory plan_path must be an absolute Path")
        _strict_int(
            self.byte_length,
            name="exploratory plan byte_length",
            minimum=1,
            maximum=MAX_ANALYSIS_PLAN_BYTES,
        )
        _sha256_digest(self.file_sha256, name="exploratory plan file_sha256")
        _sha256_digest(
            self.semantic_sha256,
            name="exploratory plan semantic_sha256",
        )
        if type(self.plan) is not ExploratoryAnalysisPlan:
            raise TypeError("plan must be ExploratoryAnalysisPlan")
        ExploratoryAnalysisPlan.__post_init__(self.plan)
        if self.semantic_sha256 != self.plan.plan_sha256:
            raise AnalysisPlanValidationError(
                "loaded exploratory semantic digest differs from the plan"
            )
        observed = _read_regular_file(self.plan_path)
        if (
            len(observed) != self.byte_length
            or sha256(observed).hexdigest() != self.file_sha256
        ):
            raise AnalysisPlanVerificationError(
                "exploratory plan file changed after loading"
            )
        if _exploratory_plan_from_snapshot(_parse_json_object(observed)) != self.plan:
            raise AnalysisPlanVerificationError(
                "exploratory plan differs from its exact file declaration"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.plan.schema_version,
            "plan_path": str(self.plan_path),
            "byte_length": self.byte_length,
            "file_sha256": self.file_sha256,
            "semantic_sha256": self.semantic_sha256,
            "plan": self.plan.snapshot(),
        }


def analysis_plan_harm_weights_sha256(weights: WelfareHarmWeights) -> str:
    """Hash the exact versioned harm-weight payload expected by a plan."""

    if type(weights) is not WelfareHarmWeights:
        raise TypeError("weights must be WelfareHarmWeights")
    WelfareHarmWeights.__post_init__(weights)
    payload = {
        "schema_version": ANALYSIS_PLAN_SCHEMA_VERSION,
        "weights": {
            name: getattr(weights, name)
            for name in weights.__dataclass_fields__
        },
    }
    return _canonical_sha256(payload)


def build_prospective_analysis_plan(
    *,
    plan_id: str,
    expected_causal_design_sha256: str,
    expected_batch_spec_sha256: str,
    expected_model_inputs_sha256: str,
    expected_population_input_sha256: str,
    expected_profile_input_sha256: str,
    expected_metric_contract_sha256: str,
    expected_harm_weights_sha256: str,
    expected_output_profile_sha256: str,
    stopping_rule: FixedSeedStoppingRule,
    estimands: Sequence[PlannedPopulationEstimand],
    declared_harm_weights: WelfareHarmWeights | None = None,
    primary_aggregate_rule: PrimaryAggregateRule | None = None,
    amendment: Mapping[str, object] | None = None,
) -> ProspectiveAnalysisPlan:
    """Canonicalize estimands and build a self-attested v1, v2, or v3 plan.

    Supplying both ``declared_harm_weights`` and ``primary_aggregate_rule``
    selects schema v2.  Omitting both preserves the legacy schema-v1 builder
    behavior.  Supplying only one is rejected.
    """

    if isinstance(estimands, (str, bytes, bytearray)) or not isinstance(
        estimands,
        Sequence,
    ):
        raise TypeError("estimands must be a sequence")
    selected = tuple(estimands)
    if any(type(item) is not PlannedPopulationEstimand for item in selected):
        raise TypeError("estimands must contain PlannedPopulationEstimand values")
    selected = tuple(sorted(selected, key=lambda item: item.estimand_id))
    aggregate_fields = (
        declared_harm_weights is not None,
        primary_aggregate_rule is not None,
    )
    if aggregate_fields not in {(False, False), (True, True)}:
        raise AnalysisPlanValidationError(
            "declared_harm_weights and primary_aggregate_rule must be supplied together"
        )
    if amendment is not None and not all(aggregate_fields):
        raise AnalysisPlanValidationError(
            "a campaign amendment requires the schema-v2 aggregate declarations"
        )
    schema_version = (
        CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION
        if amendment is not None
        else (
            PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION
            if all(aggregate_fields)
            else ANALYSIS_PLAN_SCHEMA_VERSION
        )
    )
    amendment_json = None
    if amendment is not None:
        if not isinstance(amendment, Mapping):
            raise TypeError("amendment must be a mapping")
        amendment_payload = json.loads(
            json.dumps(
                amendment,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        _validate_campaign_amendment(amendment_payload)
        amendment_json = json.dumps(
            amendment_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    if declared_harm_weights is not None:
        if type(declared_harm_weights) is not WelfareHarmWeights:
            raise TypeError("declared_harm_weights must be WelfareHarmWeights")
        WelfareHarmWeights.__post_init__(declared_harm_weights)
        if (
            analysis_plan_harm_weights_sha256(declared_harm_weights)
            != expected_harm_weights_sha256
        ):
            raise AnalysisPlanValidationError(
                "declared harm-weight vector differs from expected_harm_weights_sha256"
            )
    if primary_aggregate_rule is not None:
        if type(primary_aggregate_rule) is not PrimaryAggregateRule:
            raise TypeError("primary_aggregate_rule must be PrimaryAggregateRule")
        PrimaryAggregateRule.__post_init__(primary_aggregate_rule)
    payload = _plan_attestation_payload(
        schema_version=schema_version,
        plan_id=plan_id,
        expected_causal_design_sha256=expected_causal_design_sha256,
        expected_batch_spec_sha256=expected_batch_spec_sha256,
        expected_model_inputs_sha256=expected_model_inputs_sha256,
        expected_population_input_sha256=expected_population_input_sha256,
        expected_profile_input_sha256=expected_profile_input_sha256,
        expected_metric_contract_sha256=expected_metric_contract_sha256,
        expected_harm_weights_sha256=expected_harm_weights_sha256,
        expected_output_profile_sha256=expected_output_profile_sha256,
        stopping_rule=stopping_rule,
        estimands=selected,
        declared_harm_weights=declared_harm_weights,
        primary_aggregate_rule=primary_aggregate_rule,
        amendment_json=amendment_json,
    )
    return ProspectiveAnalysisPlan(
        schema_version=schema_version,
        plan_id=plan_id,
        expected_causal_design_sha256=expected_causal_design_sha256,
        expected_batch_spec_sha256=expected_batch_spec_sha256,
        expected_model_inputs_sha256=expected_model_inputs_sha256,
        expected_population_input_sha256=expected_population_input_sha256,
        expected_profile_input_sha256=expected_profile_input_sha256,
        expected_metric_contract_sha256=expected_metric_contract_sha256,
        expected_harm_weights_sha256=expected_harm_weights_sha256,
        expected_output_profile_sha256=expected_output_profile_sha256,
        stopping_rule=stopping_rule,
        estimands=selected,
        plan_sha256=_canonical_sha256(payload),
        declared_harm_weights=declared_harm_weights,
        primary_aggregate_rule=primary_aggregate_rule,
        amendment_json=amendment_json,
    )


def verify_prospective_analysis_plan_bindings(
    plan: ProspectiveAnalysisPlan,
    *,
    causal_design_sha256: str,
    batch_spec_sha256: str,
    model_inputs_sha256: str,
    population_input_sha256: str,
    profile_input_sha256: str,
    metric_contract_sha256: str,
    harm_weights_sha256: str,
    output_profile_sha256: str,
    seeds: tuple[int, ...],
) -> ProspectiveAnalysisPlan:
    """Compare every prospective identity with independently resolved inputs."""

    if type(plan) is not ProspectiveAnalysisPlan:
        raise TypeError("plan must be ProspectiveAnalysisPlan")
    ProspectiveAnalysisPlan.__post_init__(plan)
    observed = {
        "expected_causal_design_sha256": causal_design_sha256,
        "expected_batch_spec_sha256": batch_spec_sha256,
        "expected_model_inputs_sha256": model_inputs_sha256,
        "expected_population_input_sha256": population_input_sha256,
        "expected_profile_input_sha256": profile_input_sha256,
        "expected_metric_contract_sha256": metric_contract_sha256,
        "expected_harm_weights_sha256": harm_weights_sha256,
        "expected_output_profile_sha256": output_profile_sha256,
    }
    mismatches: list[str] = []
    for field_name, value in observed.items():
        _sha256_digest(value, name=field_name.removeprefix("expected_"))
        if field_name == "expected_profile_input_sha256":
            matches = profile_lineage_fingerprint_matches(
                getattr(plan, field_name),
                value,
            )
        else:
            matches = getattr(plan, field_name) == value
        if not matches:
            mismatches.append(field_name.removeprefix("expected_"))
    if type(seeds) is not tuple:
        raise TypeError("runtime seeds must be an exact tuple")
    try:
        FixedSeedStoppingRule(seeds=seeds)
    except (TypeError, ValueError) as exc:
        raise AnalysisPlanVerificationError(
            f"runtime seeds are invalid: {exc}"
        ) from exc
    if seeds != plan.stopping_rule.seeds:
        mismatches.append("fixed_seed_stopping_rule")
    if mismatches:
        raise AnalysisPlanVerificationError(
            "analysis plan bindings differ from resolved runtime inputs: "
            + ", ".join(mismatches)
        )
    return plan


def load_prospective_analysis_plan(
    path: str | Path,
) -> LoadedProspectiveAnalysisPlan:
    """Securely load and re-attest one supported JSON plan file."""

    candidate = _absolute_path(path)
    observed = _read_regular_file(candidate)
    raw = _parse_json_object(observed)
    plan = _plan_from_snapshot(raw)
    return LoadedProspectiveAnalysisPlan(
        plan_path=candidate,
        byte_length=len(observed),
        file_sha256=sha256(observed).hexdigest(),
        semantic_sha256=plan.plan_sha256,
        plan=plan,
    )


def verify_loaded_prospective_analysis_plan(
    loaded: LoadedProspectiveAnalysisPlan,
) -> LoadedProspectiveAnalysisPlan:
    """Reopen the selected file and reject mutation or path substitution."""

    if type(loaded) is not LoadedProspectiveAnalysisPlan:
        raise TypeError("loaded must be LoadedProspectiveAnalysisPlan")
    LoadedProspectiveAnalysisPlan.__post_init__(loaded)
    observed = load_prospective_analysis_plan(loaded.plan_path)
    if observed != loaded:
        raise AnalysisPlanVerificationError(
            "analysis plan file changed after it was loaded"
        )
    return observed


def build_exploratory_analysis_plan(
    parent_loaded: LoadedProspectiveAnalysisPlan,
    *,
    plan_id: str = (
        "illustrative.exploratory.synthetic.composite-harm.baseline-vs-safe.v1"
    ),
) -> ExploratoryAnalysisPlan:
    """Derive a non-empirical exploratory plan from the exact campaign plan.

    The primary estimand, directed scenario contrast, population predicate,
    harm weights, aggregate rule, fixed seeds, and declared technical inputs
    are copied without reinterpretation.  This builder cannot enable empirical
    claims or campaign execution.
    """

    verified = verify_loaded_prospective_analysis_plan(parent_loaded)
    parent = verified.plan
    if parent.schema_version != CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION:
        raise AnalysisPlanValidationError(
            "exploratory plan requires the exact schema-v3 campaign parent"
        )
    if len(parent.stopping_rule.seeds) != 150:
        raise AnalysisPlanValidationError(
            "exploratory plan requires the declared 150-seed parent set"
        )
    if (
        parent.declared_harm_weights is None
        or parent.primary_aggregate_rule is None
        or parent.amendment is None
    ):
        raise AnalysisPlanValidationError(
            "exploratory parent lacks aggregate or amendment declarations"
        )
    amendment = parent.amendment
    scientific_change = _required_mapping(amendment, "scientific_change")
    if (
        _required_bool(scientific_change, "primary_estimand_changed")
        or _required_string(scientific_change, "current_estimand_id")
        != parent.primary_estimand.estimand_id
        or _required_string(scientific_change, "current_specification_sha256")
        != parent.primary_estimand.specification_sha256
    ):
        raise AnalysisPlanValidationError(
            "exploratory parent does not preserve the declared primary estimand"
        )
    population = _required_mapping(amendment, "population_contract")
    monetary = _required_mapping(amendment, "monetary_contract")
    uncertainty = _required_mapping(amendment, "uncertainty_design")
    convergence = _required_mapping(amendment, "convergence_rule")
    primary = parent.primary_estimand
    aggregate = parent.primary_aggregate_rule.snapshot()
    scenario_definitions = _canonical_exploratory_scenario_definitions()
    payload: dict[str, object] = {
        "schema_version": EXPLORATORY_ANALYSIS_PLAN_SCHEMA_VERSION,
        "plan_kind": EXPLORATORY_ANALYSIS_PLAN_KIND,
        "plan_id": plan_id,
        "parent_plan": {
            "schema_version": parent.schema_version,
            "plan_id": parent.plan_id,
            "plan_sha256": parent.plan_sha256,
            "file_sha256": verified.file_sha256,
            "byte_length": verified.byte_length,
            "relationship": "EXPLORATORY_DERIVATIVE_NO_SCIENTIFIC_CHANGE",
        },
        "purpose": {
            "purpose_id": "MODEL_INTERNAL_EXPLORATORY_ENGINEERING_V1",
            "description": (
                "Explore synthetic simulator behavior under the preserved "
                "directed policy contrast; no empirical estimation or "
                "real-world policy inference is permitted."
            ),
            "synthetic_model_exploration": True,
            "empirical_analysis": False,
            "confirmatory_analysis": False,
            "production_campaign": False,
        },
        "scenario_definitions": scenario_definitions,
        "ordered_scenario_set_sha256": _exploratory_scenario_set_sha256(
            scenario_definitions
        ),
        "scientific_estimand": {
            "derived_without_change": True,
            "estimand_id": primary.estimand_id,
            "specification_sha256": primary.specification_sha256,
            "reference_scenario_id": primary.reference_scenario_id.value,
            "comparison_scenario_id": primary.comparison_scenario_id.value,
            "contrast_direction": primary.contrast_direction,
            "metric_contract_id": primary.metric_contract_id,
            "population_predicate_sha256": (
                primary.inclusion_predicate.predicate_sha256
            ),
            "estimand": primary.snapshot(),
        },
        "declared_harm_weights": _harm_weights_snapshot(
            parent.declared_harm_weights
        ),
        "expected_harm_weights_sha256": parent.expected_harm_weights_sha256,
        "stopping_rule": parent.stopping_rule.snapshot(),
        "population_weighting": {
            "mode": _required_string(population, "mode"),
            "design_id": _required_string(population, "design_id"),
            "design_sha256": _required_string(population, "design_sha256"),
            "runtime_mapping_id": _required_string(
                population,
                "runtime_mapping_id",
            ),
            "runtime_mapping_sha256": _required_string(
                population,
                "runtime_mapping_sha256",
            ),
            "adapter_id": _required_string(population, "adapter_id"),
            "adapter_sha256": _required_string(population, "adapter_sha256"),
            "population_input_sha256": _required_string(
                population,
                "population_input_sha256",
            ),
            "population_predicate_sha256": (
                primary.inclusion_predicate.predicate_sha256
            ),
            "weight_representation": "EXACT_RATIONAL",
            "application": aggregate["population_weight_application"],
            "applied_within_seed_before_cross_seed_aggregation": True,
            "identical_across_paired_scenarios": True,
            "structural_balance_is_empirical_validation": False,
            "empirical_validation_claimed": False,
            "target_population_generalization_allowed": False,
            "uncertainty_status": _required_string(
                population,
                "uncertainty_status",
            ),
        },
        "monetary_semantics": {
            "semantic_label": (
                "SIMULATED_MODEL_EQUIVALENT_TARGET_CURRENCY_VALUES"
            ),
            "target_currency": _required_string(monetary, "target_currency"),
            "quote_convention": _required_string(
                monetary,
                "quote_convention",
            ),
            "scale_convention": _required_string(
                monetary,
                "scale_convention",
            ),
            "rate_period_start": _required_string(
                monetary,
                "rate_period_start",
            ),
            "rate_period_end": _required_string(monetary, "rate_period_end"),
            "price_period_start": _required_string(
                monetary,
                "price_period_start",
            ),
            "price_period_end": _required_string(
                monetary,
                "price_period_end",
            ),
            "missing_date_policy": _required_string(
                monetary,
                "missing_date_policy",
            ),
            "rounding_rule": _required_string(monetary, "rounding_rule"),
            "rounding_boundary": _required_string(
                monetary,
                "rounding_boundary",
            ),
            "conversion_basis_sha256": _required_string(
                monetary,
                "conversion_basis_sha256",
            ),
            "source_bundle_id": _required_string(
                monetary,
                "source_bundle_id",
            ),
            "source_bundle_semantic_sha256": _required_string(
                monetary,
                "source_bundle_semantic_sha256",
            ),
            "source_bundle_signature_status": _required_string(
                monetary,
                "source_bundle_signature_status",
            ),
            "simulation_bridge_status": _required_string(
                monetary,
                "simulation_bridge_status",
            ),
            "rate_uncertainty_status": _required_string(
                monetary,
                "rate_uncertainty_status",
            ),
            "exact_rational_conversion_required": True,
            "conversion_before_population_weighting": True,
            "raw_cross_currency_pooling_allowed": False,
            "observed_real_world_spending_claimed": False,
            "empirical_monetary_interpretation_allowed": False,
        },
        "uncertainty_design": json.loads(
            json.dumps(
                uncertainty,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        ),
        "convergence_rule": json.loads(
            json.dumps(
                convergence,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        ),
        "interpretation_limits": {
            "model_internal_results_only": True,
            "empirical_estimate_claimed": False,
            "external_validity_claimed": False,
            "target_population_generalization_allowed": False,
            "real_world_causal_claims_allowed": False,
            "observed_spending_claims_allowed": False,
            "preregistration_claimed": False,
            "production_campaign_authorized": False,
            "allowed_result_label": "EXPLORATORY_SYNTHETIC_MODEL_RESULT",
        },
        "registration_status": AnalysisPlanRegistrationStatus.UNREGISTERED.value,
        "preregistered": False,
        "campaign_ready": False,
        "execution_status": "NOT_EXECUTED",
        "simulation_execution_performed": False,
        "blockers": list(_EXPLORATORY_PLAN_BLOCKERS),
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ExploratoryAnalysisPlan(
        identity_payload_json=canonical,
        plan_sha256=_canonical_sha256(payload),
    )


def load_exploratory_analysis_plan(
    path: str | Path,
) -> LoadedExploratoryAnalysisPlan:
    """Securely load and re-attest one exploratory analysis-plan file."""

    candidate = _absolute_path(path)
    observed = _read_regular_file(candidate)
    plan = _exploratory_plan_from_snapshot(_parse_json_object(observed))
    return LoadedExploratoryAnalysisPlan(
        plan_path=candidate,
        byte_length=len(observed),
        file_sha256=sha256(observed).hexdigest(),
        semantic_sha256=plan.plan_sha256,
        plan=plan,
    )


def verify_loaded_exploratory_analysis_plan(
    loaded: LoadedExploratoryAnalysisPlan,
) -> LoadedExploratoryAnalysisPlan:
    """Reopen an exploratory plan and reject any file or semantic drift."""

    if type(loaded) is not LoadedExploratoryAnalysisPlan:
        raise TypeError("loaded must be LoadedExploratoryAnalysisPlan")
    LoadedExploratoryAnalysisPlan.__post_init__(loaded)
    observed = load_exploratory_analysis_plan(loaded.plan_path)
    if observed != loaded:
        raise AnalysisPlanVerificationError(
            "exploratory plan changed after it was loaded"
        )
    return observed


def verify_exploratory_analysis_plan_parent(
    plan: ExploratoryAnalysisPlan,
    parent_loaded: LoadedProspectiveAnalysisPlan,
) -> ExploratoryAnalysisPlan:
    """Verify the exact parent file and unchanged primary scientific identity."""

    if type(plan) is not ExploratoryAnalysisPlan:
        raise TypeError("plan must be ExploratoryAnalysisPlan")
    ExploratoryAnalysisPlan.__post_init__(plan)
    parent = verify_loaded_prospective_analysis_plan(parent_loaded)
    declared_parent = _required_mapping(plan.identity_payload, "parent_plan")
    scientific = _required_mapping(plan.identity_payload, "scientific_estimand")
    mismatches: list[str] = []
    if declared_parent.get("plan_id") != parent.plan.plan_id:
        mismatches.append("parent_plan_id")
    if declared_parent.get("plan_sha256") != parent.plan.plan_sha256:
        mismatches.append("parent_plan_sha256")
    if declared_parent.get("file_sha256") != parent.file_sha256:
        mismatches.append("parent_file_sha256")
    if scientific.get("estimand_id") != parent.plan.primary_estimand.estimand_id:
        mismatches.append("primary_estimand_id")
    if (
        scientific.get("specification_sha256")
        != parent.plan.primary_estimand.specification_sha256
    ):
        mismatches.append("primary_estimand_specification_sha256")
    if mismatches:
        raise AnalysisPlanVerificationError(
            "exploratory plan differs from its exact parent: "
            + ", ".join(mismatches)
        )
    rebuilt = build_exploratory_analysis_plan(
        parent,
        plan_id=plan.plan_id,
    )
    if rebuilt != plan:
        raise AnalysisPlanVerificationError(
            "exploratory plan is not the canonical derivative of its exact parent"
        )
    return plan


def _plan_attestation_payload(
    *,
    schema_version: str = ANALYSIS_PLAN_SCHEMA_VERSION,
    plan_id: str,
    expected_causal_design_sha256: str,
    expected_batch_spec_sha256: str,
    expected_model_inputs_sha256: str,
    expected_population_input_sha256: str,
    expected_profile_input_sha256: str,
    expected_metric_contract_sha256: str,
    expected_harm_weights_sha256: str,
    expected_output_profile_sha256: str,
    stopping_rule: FixedSeedStoppingRule,
    estimands: tuple[PlannedPopulationEstimand, ...],
    declared_harm_weights: WelfareHarmWeights | None = None,
    primary_aggregate_rule: PrimaryAggregateRule | None = None,
    amendment_json: str | None = None,
) -> dict[str, object]:
    if type(stopping_rule) is not FixedSeedStoppingRule:
        raise TypeError("stopping_rule must be FixedSeedStoppingRule")
    payload = {
        "schema_version": schema_version,
        "plan_id": plan_id,
        "expected_causal_design_sha256": expected_causal_design_sha256,
        "expected_batch_spec_sha256": expected_batch_spec_sha256,
        "expected_model_inputs_sha256": expected_model_inputs_sha256,
        "expected_population_input_sha256": expected_population_input_sha256,
        "expected_profile_input_sha256": expected_profile_input_sha256,
        "expected_metric_contract_sha256": expected_metric_contract_sha256,
        "expected_harm_weights_sha256": expected_harm_weights_sha256,
        "expected_output_profile_sha256": expected_output_profile_sha256,
        "stopping_rule": stopping_rule.snapshot(),
        "estimands": [estimand.snapshot() for estimand in estimands],
        "registration_status": AnalysisPlanRegistrationStatus.UNREGISTERED.value,
        "preregistered": False,
        "campaign_ready": False,
        "campaign_blockers": list(_campaign_blockers(schema_version)),
    }
    if schema_version in {
        PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION,
        CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION,
    }:
        if type(declared_harm_weights) is not WelfareHarmWeights:
            raise TypeError("schema-v2 declared_harm_weights must be WelfareHarmWeights")
        if type(primary_aggregate_rule) is not PrimaryAggregateRule:
            raise TypeError("schema-v2 primary_aggregate_rule must be PrimaryAggregateRule")
        payload["declared_harm_weights"] = _harm_weights_snapshot(
            declared_harm_weights
        )
        payload["primary_aggregate_rule"] = primary_aggregate_rule.snapshot()
        if schema_version == CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION:
            if type(amendment_json) is not str:
                raise TypeError("schema-v3 amendment_json must be canonical text")
            amendment = _parse_amendment_json(amendment_json)
            _validate_campaign_amendment(amendment)
            payload["amendment"] = amendment
        elif amendment_json is not None:
            raise AnalysisPlanValidationError(
                "schema-v2 plans cannot contain a campaign amendment"
            )
    elif (
        declared_harm_weights is not None
        or primary_aggregate_rule is not None
        or amendment_json is not None
    ):
        raise AnalysisPlanValidationError(
            "schema-v1 plans cannot contain schema-v2 aggregate declarations"
        )
    return payload


_PLAN_KEYS_V1 = frozenset(
    {
        "schema_version",
        "plan_id",
        "expected_causal_design_sha256",
        "expected_batch_spec_sha256",
        "expected_model_inputs_sha256",
        "expected_population_input_sha256",
        "expected_profile_input_sha256",
        "expected_metric_contract_sha256",
        "expected_harm_weights_sha256",
        "expected_output_profile_sha256",
        "stopping_rule",
        "estimands",
        "registration_status",
        "preregistered",
        "campaign_ready",
        "campaign_blockers",
        "plan_sha256",
    }
)
_PLAN_KEYS_V2 = frozenset(
    set(_PLAN_KEYS_V1).union(
        {"declared_harm_weights", "primary_aggregate_rule"}
    )
)
_PLAN_KEYS_V3 = frozenset(set(_PLAN_KEYS_V2).union({"amendment"}))
_EXPLORATORY_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "plan_kind",
        "plan_id",
        "parent_plan",
        "purpose",
        "scenario_definitions",
        "ordered_scenario_set_sha256",
        "scientific_estimand",
        "declared_harm_weights",
        "expected_harm_weights_sha256",
        "stopping_rule",
        "population_weighting",
        "monetary_semantics",
        "uncertainty_design",
        "convergence_rule",
        "interpretation_limits",
        "registration_status",
        "preregistered",
        "campaign_ready",
        "execution_status",
        "simulation_execution_performed",
        "blockers",
    }
)
_EXPLORATORY_PLAN_KEYS = frozenset(
    set(_EXPLORATORY_IDENTITY_KEYS).union({"plan_sha256"})
)
_STOPPING_KEYS = frozenset(
    {
        "rule_id",
        "seeds",
        "seed_decimal_strings",
        "seed_count",
        "seed_count_decimal",
        "early_stopping_allowed",
        "treatment_result_interim_looks_allowed",
    }
)
_ESTIMAND_KEYS = frozenset(
    {
        "estimand_id",
        "role",
        "reference_scenario_id",
        "comparison_scenario_id",
        "contrast_direction",
        "outcome",
        "metric_contract_id",
        "inclusion_predicate",
        "period",
        "currency",
        "algorithm",
        "normalization",
    }
)
_OUTCOME_KEYS = frozenset(
    {
        "metric",
        "metric_name",
        "result_path",
        "component_index",
        "metric_kind",
        "metric_scale",
        "storage_dtype",
        "unit",
    }
)
_PREDICATE_KEYS = frozenset(
    {
        "rule",
        "jurisdiction_codes",
        "age_min_inclusive",
        "age_max_exclusive",
        "minor_filter",
        "monthly_disposable_income_band_ids",
        "household_type_ids",
        "gaming_states",
        "payer_history_states",
    }
)
_RULE_KEYS = frozenset(
    {"rule_id", "description", "source_fields", "timing", "evidence_role"}
)
_PERIOD_KEYS = frozenset({"period_start", "period_end", "description"})
_CURRENCY_KEYS = frozenset(
    {
        "currency_code",
        "minor_unit_name",
        "price_period_start",
        "price_period_end",
        "currency_basis_sha256",
        "rounding",
    }
)
_HARM_WEIGHT_KEYS = frozenset(
    {
        "monetary",
        "opportunity_cost",
        "sleep",
        "education_work",
        "family_social",
        "wellbeing",
    }
)
_PRIMARY_AGGREGATE_RULE_KEYS = frozenset(
    {
        "rule_id",
        "analysis_unit",
        "primary_estimand_selection",
        "scenario_aggregation",
        "scenario_weights",
        "population_weight_application",
        "seed_weighting",
        "valid_realization_criteria",
        "exclusion_criteria",
        "invalid_or_missing_realization_handling",
        "outcome_dependent_exclusion_allowed",
        "point_estimator",
        "between_seed_standard_deviation",
        "monte_carlo_standard_error",
        "interval_method",
        "one_seed_interval",
        "interval_interpretation",
        "positive_result_interpretation",
        "negative_result_interpretation",
    }
)


def _plan_from_snapshot(row: Mapping[str, object]) -> ProspectiveAnalysisPlan:
    schema_version = _required_string(row, "schema_version")
    if schema_version not in {
        ANALYSIS_PLAN_SCHEMA_VERSION,
        PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION,
        CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION,
    }:
        raise AnalysisPlanValidationError(
            "unsupported prospective analysis-plan schema version"
        )
    _exact_keys(
        row,
        (
            _PLAN_KEYS_V3
            if schema_version == CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION
            else (
                _PLAN_KEYS_V2
                if schema_version == PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION
                else _PLAN_KEYS_V1
            )
        ),
        name="analysis plan",
    )
    stopping = _stopping_from_snapshot(
        _required_mapping(row, "stopping_rule")
    )
    raw_estimands = _required_list(row, "estimands")
    estimands = tuple(
        _estimand_from_snapshot(
            _require_mapping(item, name=f"estimands[{index}]")
        )
        for index, item in enumerate(raw_estimands)
    )
    try:
        registration = AnalysisPlanRegistrationStatus(
            _required_string(row, "registration_status")
        )
    except ValueError as exc:
        raise AnalysisPlanValidationError(
            "schema-v1 registration_status must be UNREGISTERED"
        ) from exc
    if registration is not AnalysisPlanRegistrationStatus.UNREGISTERED:
        raise AnalysisPlanValidationError(
            "schema-v1 registration_status must be UNREGISTERED"
        )
    if _required_bool(row, "preregistered"):
        raise AnalysisPlanValidationError(
            "schema-v1 preregistered must be false"
        )
    if _required_bool(row, "campaign_ready"):
        raise AnalysisPlanValidationError(
            "schema-v1 campaign_ready must be false"
        )
    blockers = tuple(
        _strict_json_string(value, name=f"campaign_blockers[{index}]")
        for index, value in enumerate(_required_list(row, "campaign_blockers"))
    )
    if blockers != _campaign_blockers(schema_version):
        raise AnalysisPlanValidationError(
            "campaign_blockers differ from the fixed fail-closed schema set"
        )
    declared_harm_weights = (
        _harm_weights_from_snapshot(
            _required_mapping(row, "declared_harm_weights")
        )
        if schema_version in {
            PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION,
            CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION,
        }
        else None
    )
    primary_aggregate_rule = (
        _primary_aggregate_rule_from_snapshot(
            _required_mapping(row, "primary_aggregate_rule")
        )
        if schema_version in {
            PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION,
            CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION,
        }
        else None
    )
    amendment_json = None
    if schema_version == CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION:
        amendment_row = _required_mapping(row, "amendment")
        _validate_campaign_amendment(dict(amendment_row))
        amendment_json = json.dumps(
            amendment_row,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    plan = ProspectiveAnalysisPlan(
        schema_version=schema_version,
        plan_id=_required_string(row, "plan_id"),
        expected_causal_design_sha256=_required_string(
            row,
            "expected_causal_design_sha256",
        ),
        expected_batch_spec_sha256=_required_string(
            row,
            "expected_batch_spec_sha256",
        ),
        expected_model_inputs_sha256=_required_string(
            row,
            "expected_model_inputs_sha256",
        ),
        expected_population_input_sha256=_required_string(
            row,
            "expected_population_input_sha256",
        ),
        expected_profile_input_sha256=_required_string(
            row,
            "expected_profile_input_sha256",
        ),
        expected_metric_contract_sha256=_required_string(
            row,
            "expected_metric_contract_sha256",
        ),
        expected_harm_weights_sha256=_required_string(
            row,
            "expected_harm_weights_sha256",
        ),
        expected_output_profile_sha256=_required_string(
            row,
            "expected_output_profile_sha256",
        ),
        stopping_rule=stopping,
        estimands=estimands,
        plan_sha256=_required_string(row, "plan_sha256"),
        declared_harm_weights=declared_harm_weights,
        primary_aggregate_rule=primary_aggregate_rule,
        amendment_json=amendment_json,
    )
    if plan.snapshot() != dict(row):
        raise AnalysisPlanValidationError(
            "analysis plan JSON is not the canonical schema-v1 snapshot"
        )
    return plan


def _exploratory_plan_from_snapshot(
    row: Mapping[str, object],
) -> ExploratoryAnalysisPlan:
    if type(row) is not dict:
        raise AnalysisPlanValidationError(
            "exploratory analysis plan must be a JSON object"
        )
    _exact_keys(row, _EXPLORATORY_PLAN_KEYS, name="exploratory analysis plan")
    payload = {key: row[key] for key in _EXPLORATORY_IDENTITY_KEYS}
    _validate_exploratory_plan_payload(payload)
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ExploratoryAnalysisPlan(
        identity_payload_json=canonical,
        plan_sha256=_required_string(row, "plan_sha256"),
    )


def _stopping_from_snapshot(row: Mapping[str, object]) -> FixedSeedStoppingRule:
    _exact_keys(row, _STOPPING_KEYS, name="fixed-seed stopping rule")
    seeds = tuple(
        _strict_json_int(value, name=f"stopping seeds[{index}]")
        for index, value in enumerate(_required_list(row, "seeds"))
    )
    _required_int(row, "seed_count")
    if _required_bool(row, "early_stopping_allowed"):
        raise AnalysisPlanValidationError(
            "fixed-seed stopping rule cannot allow early stopping"
        )
    if _required_bool(row, "treatment_result_interim_looks_allowed"):
        raise AnalysisPlanValidationError(
            "fixed-seed stopping rule cannot allow treatment-result interim looks"
        )
    stopping = FixedSeedStoppingRule(seeds=seeds)
    if stopping.snapshot() != dict(row):
        raise AnalysisPlanValidationError(
            "fixed-seed stopping rule is not canonical"
        )
    return stopping


def _estimand_from_snapshot(row: Mapping[str, object]) -> PlannedPopulationEstimand:
    _exact_keys(row, _ESTIMAND_KEYS, name="planned population estimand")
    try:
        role = AnalysisEstimandRole(_required_string(row, "role"))
        reference = ScenarioId(_required_string(row, "reference_scenario_id"))
        comparison = ScenarioId(_required_string(row, "comparison_scenario_id"))
        metric = PopulationOutcomeMetric(
            _required_string(_required_mapping(row, "outcome"), "metric")
        )
    except ValueError as exc:
        raise AnalysisPlanValidationError(
            "planned estimand contains an unknown enum value"
        ) from exc
    outcome_row = _required_mapping(row, "outcome")
    _exact_keys(outcome_row, _OUTCOME_KEYS, name="population outcome semantics")
    component_index = outcome_row.get("component_index")
    if component_index is not None:
        _strict_json_int(component_index, name="outcome component_index")
    if population_outcome_semantics(metric).snapshot() != dict(outcome_row):
        raise AnalysisPlanValidationError(
            "population outcome semantics differ from the whitelist"
        )
    period = _period_from_snapshot(_required_mapping(row, "period"))
    currency_row = row.get("currency")
    currency = (
        None
        if currency_row is None
        else _currency_from_snapshot(
            _require_mapping(currency_row, name="estimand currency")
        )
    )
    estimand = PlannedPopulationEstimand(
        estimand_id=_required_string(row, "estimand_id"),
        role=role,
        reference_scenario_id=reference,
        comparison_scenario_id=comparison,
        outcome_metric=metric,
        metric_contract_id=_required_string(row, "metric_contract_id"),
        inclusion_predicate=_predicate_from_snapshot(
            _required_mapping(row, "inclusion_predicate")
        ),
        period=period,
        currency=currency,
    )
    if estimand.snapshot() != dict(row):
        raise AnalysisPlanValidationError(
            "planned estimand is not the canonical schema-v1 snapshot"
        )
    return estimand


def _predicate_from_snapshot(
    row: Mapping[str, object],
) -> CanonicalPopulationInclusionPredicate:
    _exact_keys(row, _PREDICATE_KEYS, name="population inclusion predicate")
    rule_row = _required_mapping(row, "rule")
    _exact_keys(rule_row, _RULE_KEYS, name="population inclusion rule")
    try:
        source_fields = tuple(
            PopulationInclusionField(
                _strict_json_string(value, name=f"source_fields[{index}]")
            )
            for index, value in enumerate(
                _required_list(rule_row, "source_fields")
            )
        )
        timing = PopulationInclusionTiming(
            _required_string(rule_row, "timing")
        )
        evidence_role = PopulationEstimandRole(
            _required_string(rule_row, "evidence_role")
        )
        minor_filter = PopulationMinorFilter(
            _required_string(row, "minor_filter")
        )
        gaming_states = tuple(
            PopulationGamingState(
                _strict_json_string(value, name=f"gaming_states[{index}]")
            )
            for index, value in enumerate(_required_list(row, "gaming_states"))
        )
        payer_states = tuple(
            PopulationPayerHistoryState(
                _strict_json_string(
                    value,
                    name=f"payer_history_states[{index}]",
                )
            )
            for index, value in enumerate(
                _required_list(row, "payer_history_states")
            )
        )
    except ValueError as exc:
        raise AnalysisPlanValidationError(
            "population inclusion predicate contains an unknown enum value"
        ) from exc
    rule = PopulationInclusionRule(
        rule_id=_required_string(rule_row, "rule_id"),
        description=_required_string(rule_row, "description"),
        source_fields=source_fields,
        timing=timing,
        evidence_role=evidence_role,
    )
    predicate = CanonicalPopulationInclusionPredicate(
        rule=rule,
        jurisdiction_codes=_string_tuple_from_snapshot(
            row,
            "jurisdiction_codes",
        ),
        age_min_inclusive=_required_int(row, "age_min_inclusive"),
        age_max_exclusive=_required_int(row, "age_max_exclusive"),
        minor_filter=minor_filter,
        monthly_disposable_income_band_ids=_string_tuple_from_snapshot(
            row,
            "monthly_disposable_income_band_ids",
        ),
        household_type_ids=_string_tuple_from_snapshot(
            row,
            "household_type_ids",
        ),
        gaming_states=gaming_states,
        payer_history_states=payer_states,
    )
    if predicate.snapshot() != dict(row):
        raise AnalysisPlanValidationError(
            "population inclusion predicate is not canonical"
        )
    return predicate


def _period_from_snapshot(row: Mapping[str, object]) -> PopulationPeriodSemantics:
    _exact_keys(row, _PERIOD_KEYS, name="estimand period")
    period = PopulationPeriodSemantics(
        period_start=_iso_date(row.get("period_start"), name="period_start"),
        period_end=_iso_date(row.get("period_end"), name="period_end"),
        description=_required_string(row, "description"),
    )
    if period.snapshot() != dict(row):
        raise AnalysisPlanValidationError("estimand period is not canonical")
    return period


def _currency_from_snapshot(
    row: Mapping[str, object],
) -> PopulationCurrencySemantics:
    _exact_keys(row, _CURRENCY_KEYS, name="estimand currency")
    from ..metrics.population_estimands import PopulationCurrencyRounding

    try:
        rounding = PopulationCurrencyRounding(
            _required_string(row, "rounding")
        )
    except ValueError as exc:
        raise AnalysisPlanValidationError(
            "estimand currency rounding is invalid"
        ) from exc
    currency = PopulationCurrencySemantics(
        currency_code=_required_string(row, "currency_code"),
        minor_unit_name=_required_string(row, "minor_unit_name"),
        price_period_start=_iso_date(
            row.get("price_period_start"),
            name="price_period_start",
        ),
        price_period_end=_iso_date(
            row.get("price_period_end"),
            name="price_period_end",
        ),
        currency_basis_sha256=_required_string(
            row,
            "currency_basis_sha256",
        ),
        rounding=rounding,
    )
    if currency.snapshot() != dict(row):
        raise AnalysisPlanValidationError("estimand currency is not canonical")
    return currency


def _harm_weights_from_snapshot(
    row: Mapping[str, object],
) -> WelfareHarmWeights:
    _exact_keys(row, _HARM_WEIGHT_KEYS, name="declared harm weights")
    try:
        weights = WelfareHarmWeights(
            **{name: row[name] for name in _HARM_WEIGHT_KEYS}
        )
    except (TypeError, ValueError) as exc:
        raise AnalysisPlanValidationError(
            f"declared harm weights are invalid: {exc}"
        ) from exc
    if _harm_weights_snapshot(weights) != dict(row):
        raise AnalysisPlanValidationError(
            "declared harm weights are not a canonical snapshot"
        )
    return weights


def _primary_aggregate_rule_from_snapshot(
    row: Mapping[str, object],
) -> PrimaryAggregateRule:
    _exact_keys(
        row,
        _PRIMARY_AGGREGATE_RULE_KEYS,
        name="primary aggregate rule",
    )
    rule = PrimaryAggregateRule(
        positive_result_interpretation=_required_string(
            row,
            "positive_result_interpretation",
        ),
        negative_result_interpretation=_required_string(
            row,
            "negative_result_interpretation",
        ),
    )
    if rule.snapshot() != dict(row):
        raise AnalysisPlanValidationError(
            "primary aggregate rule is not the canonical outcome-blind rule"
        )
    return rule


def _absolute_path(value: str | Path) -> Path:
    if type(value) is not str and not isinstance(value, Path):
        raise TypeError("analysis plan path must be str or Path")
    if not os.fspath(value):
        raise ValueError("analysis plan path cannot be empty")
    return Path(os.path.abspath(os.fspath(value)))


def _read_regular_file(path: Path) -> bytes:
    def is_reparse(metadata: os.stat_result) -> bool:
        marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        attributes = getattr(metadata, "st_file_attributes", 0)
        return bool(attributes & marker)

    def same_identity(left: os.stat_result, right: os.stat_result) -> bool:
        return (
            left.st_dev,
            left.st_ino,
            left.st_mode,
            left.st_size,
            left.st_mtime_ns,
        ) == (
            right.st_dev,
            right.st_ino,
            right.st_mode,
            right.st_size,
            right.st_mtime_ns,
        )

    try:
        before = path.lstat()
    except OSError as exc:
        raise AnalysisPlanVerificationError(
            f"cannot inspect analysis plan file: {path}"
        ) from exc
    if path.is_symlink() or is_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise AnalysisPlanVerificationError(
            "analysis plan path must name a regular non-symlink file"
        )
    if before.st_size <= 0 or before.st_size > MAX_ANALYSIS_PLAN_BYTES:
        raise AnalysisPlanValidationError(
            "analysis plan byte length is outside schema-v1 limits"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AnalysisPlanVerificationError(
            f"cannot open analysis plan file: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or is_reparse(opened):
            raise AnalysisPlanVerificationError(
                "opened analysis plan object is not a regular file"
            )
        if not same_identity(before, opened):
            raise AnalysisPlanVerificationError(
                "analysis plan file changed while it was opened"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, MAX_ANALYSIS_PLAN_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_ANALYSIS_PLAN_BYTES:
                raise AnalysisPlanValidationError(
                    "analysis plan exceeds the schema-v1 byte limit"
                )
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise AnalysisPlanVerificationError(
            "analysis plan file changed while it was read"
        ) from exc
    if (
        path.is_symlink()
        or is_reparse(after_path)
        or not stat.S_ISREG(after_path.st_mode)
    ):
        raise AnalysisPlanVerificationError(
            "analysis plan path changed to a non-regular alias"
        )
    if not same_identity(opened, after_open) or not same_identity(
        after_open,
        after_path,
    ):
        raise AnalysisPlanVerificationError(
            "analysis plan file changed while it was read"
        )
    observed = b"".join(chunks)
    if len(observed) != after_open.st_size:
        raise AnalysisPlanVerificationError(
            "analysis plan file was not read completely"
        )
    return observed


def _parse_json_object(observed: bytes) -> dict[str, object]:
    try:
        text = observed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnalysisPlanValidationError(
            "analysis plan must be valid UTF-8 JSON"
        ) from exc

    def reject_constant(value: str) -> object:
        raise AnalysisPlanValidationError(
            f"analysis plan JSON cannot contain {value}"
        )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise AnalysisPlanValidationError(
                    f"analysis plan JSON repeats key {key!r}"
                )
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except AnalysisPlanValidationError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise AnalysisPlanValidationError(
            "analysis plan must be a valid bounded JSON object"
        ) from exc
    if type(parsed) is not dict:
        raise AnalysisPlanValidationError(
            "analysis plan JSON root must be an object"
        )
    return parsed


def _exact_keys(
    row: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    actual = set(row)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise AnalysisPlanValidationError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


def _required_mapping(
    row: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    return _require_mapping(row.get(key), name=key)


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise AnalysisPlanValidationError(f"{name} must be a JSON object")
    return value


def _required_list(row: Mapping[str, object], key: str) -> list[object]:
    value = row.get(key)
    if type(value) is not list:
        raise AnalysisPlanValidationError(f"{key} must be a JSON array")
    return value


def _required_string(row: Mapping[str, object], key: str) -> str:
    return _strict_json_string(row.get(key), name=key)


def _strict_json_string(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise AnalysisPlanValidationError(f"{name} must be JSON text")
    return value


def _required_int(row: Mapping[str, object], key: str) -> int:
    return _strict_json_int(row.get(key), name=key)


def _required_bool(row: Mapping[str, object], key: str) -> bool:
    value = row.get(key)
    if type(value) is not bool:
        raise AnalysisPlanValidationError(f"{key} must be a JSON boolean")
    return value


def _strict_json_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise AnalysisPlanValidationError(f"{name} must be a JSON integer")
    return value


def _string_tuple_from_snapshot(
    row: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    return tuple(
        _strict_json_string(value, name=f"{key}[{index}]")
        for index, value in enumerate(_required_list(row, key))
    )


def _iso_date(value: object, *, name: str) -> date:
    if type(value) is not str:
        raise AnalysisPlanValidationError(f"{name} must be an ISO date")
    try:
        observed = date.fromisoformat(value)
    except ValueError as exc:
        raise AnalysisPlanValidationError(f"{name} must be an ISO date") from exc
    if observed.isoformat() != value:
        raise AnalysisPlanValidationError(f"{name} must be a canonical ISO date")
    return observed


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise AnalysisPlanValidationError(
            f"{name} must be a canonical 1-128 character identifier"
        )
    return value


def _contract_identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not _CONTRACT_IDENTIFIER.fullmatch(value):
        raise AnalysisPlanValidationError(
            f"{name} must be a canonical output-contract identifier"
        )
    return value


def _nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise AnalysisPlanValidationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )
    return value


def _sha256_digest(value: object, *, name: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise AnalysisPlanValidationError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact Python integer")
    if not minimum <= value <= maximum:
        raise AnalysisPlanValidationError(
            f"{name} must be in [{minimum}, {maximum}]"
        )
    return value


def _canonical_code_tuple(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or not _JURISDICTION_CODE.fullmatch(value)
        for value in values
    ):
        raise AnalysisPlanValidationError(
            "jurisdiction_codes must be an exact tuple of canonical codes"
        )
    if values != tuple(sorted(set(values))):
        raise AnalysisPlanValidationError(
            "jurisdiction_codes must be unique and ascending"
        )


def _runtime_jurisdiction_codes(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or not values or any(
        type(value) is not str or not _JURISDICTION_CODE.fullmatch(value)
        for value in values
    ):
        raise TypeError(
            "runtime jurisdiction_codes must be a non-empty exact tuple of codes"
        )
    if len(set(values)) != len(values):
        raise AnalysisPlanValidationError(
            "runtime jurisdiction_codes must be unique"
        )


def _canonical_identifier_tuple(
    values: tuple[str, ...],
    *,
    name: str,
) -> None:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    for index, value in enumerate(values):
        _identifier(value, name=f"{name}[{index}]")
    if values != tuple(sorted(set(values))):
        raise AnalysisPlanValidationError(
            f"{name} must be unique and ascending"
        )


def _canonical_enum_tuple(
    values: tuple[Enum, ...],
    *,
    enum_type: type[Enum],
    name: str,
) -> None:
    if type(values) is not tuple or any(type(value) is not enum_type for value in values):
        raise TypeError(f"{name} must be an exact tuple of {enum_type.__name__}")
    if values != tuple(sorted(set(values), key=lambda item: str(item.value))):
        raise AnalysisPlanValidationError(f"{name} must be unique and canonical")


def _gaming_state(value: bool) -> PopulationGamingState:
    return (
        PopulationGamingState.GAMER
        if value
        else PopulationGamingState.NON_GAMER
    )


def _payer_state(value: bool) -> PopulationPayerHistoryState:
    return (
        PopulationPayerHistoryState.EVER_PAYER
        if value
        else PopulationPayerHistoryState.NEVER_PAYER
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _harm_weights_snapshot(weights: WelfareHarmWeights) -> dict[str, float]:
    if type(weights) is not WelfareHarmWeights:
        raise TypeError("weights must be WelfareHarmWeights")
    WelfareHarmWeights.__post_init__(weights)
    return {
        name: getattr(weights, name)
        for name in weights.__dataclass_fields__
    }


def _canonical_exploratory_scenario_definitions() -> list[dict[str, object]]:
    """Return the full seven-scenario catalogue in stable enum order."""

    scenarios = required_scenarios()
    if tuple(item.scenario_id for item in scenarios) != tuple(ScenarioId):
        raise AnalysisPlanValidationError(
            "required scenario catalogue differs from ScenarioId order"
        )
    return [
        {
            "ordinal": ordinal,
            "scenario_id": scenario.scenario_id.value,
            "label": scenario.label,
            "description": scenario.description,
            "mechanics": {
                field_name: getattr(scenario.mechanics, field_name)
                for field_name in scenario.mechanics.__dataclass_fields__
            },
            "fixed_access_price_cents": scenario.fixed_access_price_cents,
            "subscription_price_cents": scenario.subscription_price_cents,
            "epgc_enabled": scenario.epgc_enabled,
        }
        for ordinal, scenario in enumerate(scenarios)
    ]


def _exploratory_scenario_set_sha256(
    definitions: Sequence[Mapping[str, object]],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": EXPLORATORY_ANALYSIS_PLAN_SCHEMA_VERSION,
            "ordering": "SCENARIO_ID_ENUM_DECLARATION_ORDER",
            "scenario_definitions": list(definitions),
        }
    )


def _validate_exploratory_plan_payload(row: Mapping[str, object]) -> None:
    """Validate the fixed non-empirical interpretation boundary."""

    if type(row) is not dict:
        raise AnalysisPlanValidationError(
            "exploratory plan payload must be a JSON object"
        )
    _exact_keys(row, _EXPLORATORY_IDENTITY_KEYS, name="exploratory plan identity")
    if _required_string(row, "schema_version") != EXPLORATORY_ANALYSIS_PLAN_SCHEMA_VERSION:
        raise AnalysisPlanValidationError(
            "unsupported exploratory analysis-plan schema version"
        )
    if _required_string(row, "plan_kind") != EXPLORATORY_ANALYSIS_PLAN_KIND:
        raise AnalysisPlanValidationError("exploratory plan_kind is fixed")
    _identifier(_required_string(row, "plan_id"), name="exploratory plan_id")

    parent = _required_mapping(row, "parent_plan")
    _exact_keys(
        parent,
        frozenset(
            {
                "schema_version",
                "plan_id",
                "plan_sha256",
                "file_sha256",
                "byte_length",
                "relationship",
            }
        ),
        name="exploratory parent_plan",
    )
    if _required_string(parent, "schema_version") != CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION:
        raise AnalysisPlanValidationError(
            "exploratory parent must be the schema-v3 campaign plan"
        )
    _identifier(_required_string(parent, "plan_id"), name="parent plan_id")
    _sha256_digest(_required_string(parent, "plan_sha256"), name="parent plan_sha256")
    _sha256_digest(_required_string(parent, "file_sha256"), name="parent file_sha256")
    if _required_int(parent, "byte_length") <= 0:
        raise AnalysisPlanValidationError("parent byte_length must be positive")
    if _required_string(
        parent,
        "relationship",
    ) != "EXPLORATORY_DERIVATIVE_NO_SCIENTIFIC_CHANGE":
        raise AnalysisPlanValidationError(
            "exploratory parent relationship cannot claim a scientific change"
        )

    purpose = _required_mapping(row, "purpose")
    _exact_keys(
        purpose,
        frozenset(
            {
                "purpose_id",
                "description",
                "synthetic_model_exploration",
                "empirical_analysis",
                "confirmatory_analysis",
                "production_campaign",
            }
        ),
        name="exploratory purpose",
    )
    if _required_string(
        purpose,
        "purpose_id",
    ) != "MODEL_INTERNAL_EXPLORATORY_ENGINEERING_V1":
        raise AnalysisPlanValidationError("exploratory purpose_id is fixed")
    _nonempty_text(_required_string(purpose, "description"), name="purpose description")
    if not _required_bool(purpose, "synthetic_model_exploration"):
        raise AnalysisPlanValidationError(
            "exploratory plan must identify synthetic model exploration"
        )
    for field_name in (
        "empirical_analysis",
        "confirmatory_analysis",
        "production_campaign",
    ):
        if _required_bool(purpose, field_name):
            raise AnalysisPlanValidationError(
                f"exploratory purpose requires {field_name}=false"
            )

    scenario_definitions = _required_list(row, "scenario_definitions")
    canonical_scenarios = _canonical_exploratory_scenario_definitions()
    if scenario_definitions != canonical_scenarios:
        raise AnalysisPlanValidationError(
            "exploratory scenario definitions differ from the complete ordered "
            "seven-scenario catalogue"
        )
    scenario_set_sha256 = _required_string(
        row,
        "ordered_scenario_set_sha256",
    )
    _sha256_digest(
        scenario_set_sha256,
        name="exploratory ordered scenario set",
    )
    if scenario_set_sha256 != _exploratory_scenario_set_sha256(
        canonical_scenarios
    ):
        raise AnalysisPlanValidationError(
            "ordered_scenario_set_sha256 differs from the scenario definitions"
        )

    scientific = _required_mapping(row, "scientific_estimand")
    _exact_keys(
        scientific,
        frozenset(
            {
                "derived_without_change",
                "estimand_id",
                "specification_sha256",
                "reference_scenario_id",
                "comparison_scenario_id",
                "contrast_direction",
                "metric_contract_id",
                "population_predicate_sha256",
                "estimand",
            }
        ),
        name="exploratory scientific_estimand",
    )
    if not _required_bool(scientific, "derived_without_change"):
        raise AnalysisPlanValidationError(
            "exploratory estimand must be derived without scientific change"
        )
    estimand = _estimand_from_snapshot(_required_mapping(scientific, "estimand"))
    if estimand.role is not AnalysisEstimandRole.PRIMARY:
        raise AnalysisPlanValidationError(
            "exploratory scientific estimand must remain PRIMARY"
        )
    if (
        _required_string(scientific, "estimand_id") != estimand.estimand_id
        or _required_string(scientific, "specification_sha256")
        != estimand.specification_sha256
        or _required_string(scientific, "reference_scenario_id")
        != estimand.reference_scenario_id.value
        or _required_string(scientific, "comparison_scenario_id")
        != estimand.comparison_scenario_id.value
        or _required_string(scientific, "contrast_direction")
        != estimand.contrast_direction
        or _required_string(scientific, "metric_contract_id")
        != estimand.metric_contract_id
        or _required_string(scientific, "population_predicate_sha256")
        != estimand.inclusion_predicate.predicate_sha256
    ):
        raise AnalysisPlanValidationError(
            "exploratory explicit contrast differs from its estimand"
        )

    harm_weights = _harm_weights_from_snapshot(
        _required_mapping(row, "declared_harm_weights")
    )
    expected_harm_sha = _required_string(row, "expected_harm_weights_sha256")
    _sha256_digest(expected_harm_sha, name="exploratory harm weights")
    if analysis_plan_harm_weights_sha256(harm_weights) != expected_harm_sha:
        raise AnalysisPlanValidationError(
            "exploratory harm weights differ from their declared identity"
        )

    stopping = _stopping_from_snapshot(_required_mapping(row, "stopping_rule"))
    if len(stopping.seeds) != 150:
        raise AnalysisPlanValidationError(
            "exploratory analysis requires exactly 150 fixed seeds"
        )

    population = _required_mapping(row, "population_weighting")
    _exact_keys(
        population,
        frozenset(
            {
                "mode",
                "design_id",
                "design_sha256",
                "runtime_mapping_id",
                "runtime_mapping_sha256",
                "adapter_id",
                "adapter_sha256",
                "population_input_sha256",
                "population_predicate_sha256",
                "weight_representation",
                "application",
                "applied_within_seed_before_cross_seed_aggregation",
                "identical_across_paired_scenarios",
                "structural_balance_is_empirical_validation",
                "empirical_validation_claimed",
                "target_population_generalization_allowed",
                "uncertainty_status",
            }
        ),
        name="exploratory population_weighting",
    )
    for field_name in (
        "design_id",
        "runtime_mapping_id",
        "adapter_id",
    ):
        _identifier(_required_string(population, field_name), name=field_name)
    for field_name in (
        "design_sha256",
        "runtime_mapping_sha256",
        "adapter_sha256",
        "population_input_sha256",
        "population_predicate_sha256",
    ):
        _sha256_digest(_required_string(population, field_name), name=field_name)
    if _required_string(population, "mode") != "projected_v1":
        raise AnalysisPlanValidationError("exploratory population mode must be projected_v1")
    if _required_string(population, "weight_representation") != "EXACT_RATIONAL":
        raise AnalysisPlanValidationError("population weights must remain exact rational")
    if _required_string(
        population,
        "application",
    ) != "WITHIN_EACH_SEED_BEFORE_CROSS_SEED_AGGREGATION":
        raise AnalysisPlanValidationError(
            "population weights must be applied within each seed"
        )
    for field_name in (
        "applied_within_seed_before_cross_seed_aggregation",
        "identical_across_paired_scenarios",
    ):
        if not _required_bool(population, field_name):
            raise AnalysisPlanValidationError(
                f"population weighting requires {field_name}=true"
            )
    for field_name in (
        "structural_balance_is_empirical_validation",
        "empirical_validation_claimed",
        "target_population_generalization_allowed",
    ):
        if _required_bool(population, field_name):
            raise AnalysisPlanValidationError(
                f"exploratory population requires {field_name}=false"
            )
    if _required_string(population, "uncertainty_status") != "UNQUANTIFIED":
        raise AnalysisPlanValidationError(
            "exploratory population uncertainty must remain unquantified"
        )
    if (
        _required_string(population, "population_predicate_sha256")
        != estimand.inclusion_predicate.predicate_sha256
    ):
        raise AnalysisPlanValidationError(
            "population weighting predicate differs from the estimand"
        )

    monetary = _required_mapping(row, "monetary_semantics")
    _exact_keys(
        monetary,
        frozenset(
            {
                "semantic_label",
                "target_currency",
                "quote_convention",
                "scale_convention",
                "rate_period_start",
                "rate_period_end",
                "price_period_start",
                "price_period_end",
                "missing_date_policy",
                "rounding_rule",
                "rounding_boundary",
                "conversion_basis_sha256",
                "source_bundle_id",
                "source_bundle_semantic_sha256",
                "source_bundle_signature_status",
                "simulation_bridge_status",
                "rate_uncertainty_status",
                "exact_rational_conversion_required",
                "conversion_before_population_weighting",
                "raw_cross_currency_pooling_allowed",
                "observed_real_world_spending_claimed",
                "empirical_monetary_interpretation_allowed",
            }
        ),
        name="exploratory monetary_semantics",
    )
    if _required_string(
        monetary,
        "semantic_label",
    ) != "SIMULATED_MODEL_EQUIVALENT_TARGET_CURRENCY_VALUES":
        raise AnalysisPlanValidationError(
            "exploratory monetary values must remain model-equivalent"
        )
    for field_name in (
        "target_currency",
        "quote_convention",
        "scale_convention",
        "rate_period_start",
        "rate_period_end",
        "price_period_start",
        "price_period_end",
        "missing_date_policy",
        "rounding_rule",
        "rounding_boundary",
        "source_bundle_id",
    ):
        _nonempty_text(_required_string(monetary, field_name), name=field_name)
    for field_name in (
        "conversion_basis_sha256",
        "source_bundle_semantic_sha256",
    ):
        _sha256_digest(_required_string(monetary, field_name), name=field_name)
    if _required_string(monetary, "source_bundle_signature_status") != "MISSING":
        raise AnalysisPlanValidationError(
            "current exploratory monetary source signature remains missing"
        )
    if _required_string(monetary, "simulation_bridge_status") != "ILLUSTRATIVE":
        raise AnalysisPlanValidationError(
            "current exploratory monetary bridge remains illustrative"
        )
    if _required_string(monetary, "rate_uncertainty_status") != "UNQUANTIFIED":
        raise AnalysisPlanValidationError(
            "exploratory monetary-rate uncertainty must remain unquantified"
        )
    for field_name in (
        "exact_rational_conversion_required",
        "conversion_before_population_weighting",
    ):
        if not _required_bool(monetary, field_name):
            raise AnalysisPlanValidationError(
                f"monetary semantics require {field_name}=true"
            )
    for field_name in (
        "raw_cross_currency_pooling_allowed",
        "observed_real_world_spending_claimed",
        "empirical_monetary_interpretation_allowed",
    ):
        if _required_bool(monetary, field_name):
            raise AnalysisPlanValidationError(
                f"exploratory monetary semantics require {field_name}=false"
            )

    uncertainty = _required_mapping(row, "uncertainty_design")
    if _required_string(uncertainty, "schema_version") != "1.0":
        raise AnalysisPlanValidationError("unsupported exploratory uncertainty schema")
    if _required_string(uncertainty, "oat_role") != "DIAGNOSTIC_ONLY":
        raise AnalysisPlanValidationError("exploratory OAT must remain diagnostic")
    seed_uncertainty = _required_mapping(uncertainty, "seed_uncertainty")
    if (
        _required_string(seed_uncertainty, "status")
        != "QUANTIFIED_WHEN_COMPLETE"
        or _required_int(seed_uncertainty, "fixed_seed_count") != 150
        or not _required_bool(seed_uncertainty, "common_random_numbers")
        or not _required_bool(
            seed_uncertainty,
            "identical_pretreatment_cohorts",
        )
        or not _required_bool(
            seed_uncertainty,
            "population_weights_applied_within_seed",
        )
        or _required_bool(
            seed_uncertainty,
            "outcome_dependent_seed_exclusion_allowed",
        )
    ):
        raise AnalysisPlanValidationError(
            "exploratory seed-uncertainty declaration is not fail closed"
        )
    parameter_uncertainty = _required_mapping(
        uncertainty,
        "parameter_uncertainty",
    )
    if (
        _required_string(parameter_uncertainty, "status")
        != "ILLUSTRATIVE_DESIGN_ONLY"
        or _required_string(parameter_uncertainty, "probability_interpretation")
        != "NONE"
    ):
        raise AnalysisPlanValidationError(
            "exploratory parameter uncertainty cannot claim calibration"
        )
    _sha256_digest(
        _required_string(parameter_uncertainty, "design_sha256"),
        name="exploratory parameter design",
    )
    monetary_uncertainty = _required_mapping(
        uncertainty,
        "monetary_rate_uncertainty",
    )
    population_uncertainty = _required_mapping(
        uncertainty,
        "population_uncertainty",
    )
    combined_uncertainty = _required_mapping(
        uncertainty,
        "combined_uncertainty",
    )
    if (
        _required_string(monetary_uncertainty, "status") != "UNQUANTIFIED"
        or _required_string(population_uncertainty, "status") != "UNQUANTIFIED"
        or _required_string(combined_uncertainty, "status")
        != "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST"
    ):
        raise AnalysisPlanValidationError(
            "exploratory required uncertainty components remain unavailable"
        )

    convergence = _required_mapping(row, "convergence_rule")
    required_convergence = frozenset(
        {
            "schema_version",
            "block_size",
            "minimum_retained_seeds",
            "maximum_mcse",
            "maximum_interval_width",
            "maximum_absolute_change",
            "maximum_relative_change",
            "maximum_invalid_rate",
            "consecutive_passing_checkpoints",
            "sensitivity_instability_allowed",
            "outcome_dependent_seed_exclusion_allowed",
            "required_uncertainty_component_handling",
        }
    )
    _exact_keys(convergence, required_convergence, name="exploratory convergence_rule")
    if _required_string(convergence, "schema_version") != "1.0":
        raise AnalysisPlanValidationError("unsupported exploratory convergence schema")
    for field_name in (
        "block_size",
        "minimum_retained_seeds",
        "consecutive_passing_checkpoints",
    ):
        if _required_int(convergence, field_name) <= 0:
            raise AnalysisPlanValidationError(
                f"exploratory convergence {field_name} must be positive"
            )
    if _required_int(convergence, "minimum_retained_seeds") < 100:
        raise AnalysisPlanValidationError(
            "exploratory convergence requires at least 100 retained seeds"
        )
    for field_name in (
        "maximum_mcse",
        "maximum_interval_width",
        "maximum_absolute_change",
        "maximum_relative_change",
    ):
        value = convergence.get(field_name)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise AnalysisPlanValidationError(
                f"exploratory convergence {field_name} must be positive"
            )
    invalid_rate = convergence.get("maximum_invalid_rate")
    if (
        type(invalid_rate) not in {int, float}
        or isinstance(invalid_rate, bool)
        or not np.isfinite(float(invalid_rate))
        or not 0.0 <= float(invalid_rate) < 1.0
    ):
        raise AnalysisPlanValidationError(
            "exploratory maximum_invalid_rate must be in [0, 1)"
        )
    if (
        _required_bool(convergence, "sensitivity_instability_allowed")
        or _required_bool(
            convergence,
            "outcome_dependent_seed_exclusion_allowed",
        )
        or _required_string(
            convergence,
            "required_uncertainty_component_handling",
        )
        != "FAIL_CLOSED"
    ):
        raise AnalysisPlanValidationError(
            "exploratory convergence must fail closed"
        )

    limits = _required_mapping(row, "interpretation_limits")
    _exact_keys(
        limits,
        frozenset(
            {
                "model_internal_results_only",
                "empirical_estimate_claimed",
                "external_validity_claimed",
                "target_population_generalization_allowed",
                "real_world_causal_claims_allowed",
                "observed_spending_claims_allowed",
                "preregistration_claimed",
                "production_campaign_authorized",
                "allowed_result_label",
            }
        ),
        name="exploratory interpretation_limits",
    )
    if not _required_bool(limits, "model_internal_results_only"):
        raise AnalysisPlanValidationError(
            "exploratory results must be labelled model-internal"
        )
    for field_name in (
        "empirical_estimate_claimed",
        "external_validity_claimed",
        "target_population_generalization_allowed",
        "real_world_causal_claims_allowed",
        "observed_spending_claims_allowed",
        "preregistration_claimed",
        "production_campaign_authorized",
    ):
        if _required_bool(limits, field_name):
            raise AnalysisPlanValidationError(
                f"exploratory interpretation requires {field_name}=false"
            )
    if _required_string(
        limits,
        "allowed_result_label",
    ) != "EXPLORATORY_SYNTHETIC_MODEL_RESULT":
        raise AnalysisPlanValidationError("exploratory result label is fixed")

    if _required_string(row, "registration_status") != "UNREGISTERED":
        raise AnalysisPlanValidationError(
            "exploratory plan cannot claim external registration"
        )
    if _required_bool(row, "preregistered") or _required_bool(row, "campaign_ready"):
        raise AnalysisPlanValidationError(
            "exploratory plan cannot be preregistered or campaign-ready"
        )
    if _required_string(row, "execution_status") != "NOT_EXECUTED":
        raise AnalysisPlanValidationError(
            "the generated exploratory plan must remain not executed"
        )
    if _required_bool(row, "simulation_execution_performed"):
        raise AnalysisPlanValidationError(
            "plan generation cannot claim a simulation execution"
        )
    blockers = tuple(
        _strict_json_string(value, name=f"blockers[{index}]")
        for index, value in enumerate(_required_list(row, "blockers"))
    )
    if blockers != _EXPLORATORY_PLAN_BLOCKERS:
        raise AnalysisPlanValidationError(
            "exploratory blockers differ from the fixed schema set"
        )


_AMENDMENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "amendment_schema_version",
        "parent_plan",
        "scientific_change",
        "changed_inputs",
        "population_contract",
        "monetary_contract",
        "uncertainty_design",
        "convergence_rule",
        "execution_attestation",
        "simulation_flow",
        "readiness_consequences",
    }
)


def _parse_amendment_json(value: str) -> dict[str, object]:
    if type(value) is not str or not value:
        raise AnalysisPlanValidationError("amendment payload must be JSON text")
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisPlanValidationError("amendment payload is invalid JSON") from exc
    if type(decoded) is not dict:
        raise AnalysisPlanValidationError("amendment payload must be an object")
    canonical = json.dumps(
        decoded,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if canonical != value:
        raise AnalysisPlanValidationError(
            "amendment payload must use canonical JSON encoding"
        )
    return decoded


def _amendment_mapping(
    row: Mapping[str, object], key: str
) -> Mapping[str, object]:
    return _required_mapping(row, key)


def _validate_campaign_amendment(row: Mapping[str, object]) -> None:
    """Validate the immutable scientific/provenance delta carried by schema v3."""

    if not isinstance(row, Mapping):
        raise AnalysisPlanValidationError("campaign amendment must be a mapping")
    _exact_keys(row, _AMENDMENT_KEYS, name="campaign amendment")
    if _required_string(row, "amendment_schema_version") != "1.0":
        raise AnalysisPlanValidationError(
            "unsupported campaign amendment schema version"
        )
    parent = _amendment_mapping(row, "parent_plan")
    _exact_keys(
        parent,
        {
            "artifact_path",
            "schema_version",
            "plan_id",
            "plan_sha256",
            "file_sha256",
        },
        name="amendment parent_plan",
    )
    parent_path = _required_string(parent, "artifact_path")
    if "\\" in parent_path or parent_path.startswith("/") or ".." in parent_path.split("/"):
        raise AnalysisPlanValidationError(
            "parent plan artifact_path must be a canonical repository-relative POSIX path"
        )
    if _required_string(parent, "schema_version") != PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION:
        raise AnalysisPlanValidationError("schema-v3 parent must be a schema-v2 plan")
    _identifier(_required_string(parent, "plan_id"), name="parent plan_id")
    _sha256_digest(_required_string(parent, "plan_sha256"), name="parent plan_sha256")
    _sha256_digest(_required_string(parent, "file_sha256"), name="parent file_sha256")

    scientific = _amendment_mapping(row, "scientific_change")
    _exact_keys(
        scientific,
        {
            "primary_estimand_changed",
            "original_estimand_id",
            "current_estimand_id",
            "original_specification_sha256",
            "current_specification_sha256",
            "explanation",
        },
        name="amendment scientific_change",
    )
    if _required_bool(scientific, "primary_estimand_changed"):
        raise AnalysisPlanValidationError(
            "this successor plan declares that the primary estimand is preserved"
        )
    original_id = _required_string(scientific, "original_estimand_id")
    current_id = _required_string(scientific, "current_estimand_id")
    if original_id != current_id:
        raise AnalysisPlanValidationError(
            "preserved primary estimand IDs must be identical"
        )
    original_sha = _required_string(scientific, "original_specification_sha256")
    current_sha = _required_string(scientific, "current_specification_sha256")
    _sha256_digest(original_sha, name="original primary specification")
    _sha256_digest(current_sha, name="current primary specification")
    if original_sha != current_sha:
        raise AnalysisPlanValidationError(
            "preserved primary estimand specifications must be identical"
        )
    _nonempty_text(_required_string(scientific, "explanation"), name="scientific explanation")

    raw_inputs = _required_list(row, "changed_inputs")
    if not raw_inputs:
        raise AnalysisPlanValidationError("changed_inputs cannot be empty")
    roles: list[str] = []
    for index, value in enumerate(raw_inputs):
        item = _require_mapping(value, name=f"changed_inputs[{index}]")
        _exact_keys(
            item,
            {
                "role",
                "artifact_path",
                "schema_version",
                "file_sha256",
                "semantic_sha256",
                "change_type",
                "readiness_status",
                "readiness_consequence",
            },
            name=f"changed_inputs[{index}]",
        )
        role = _required_string(item, "role")
        _identifier(role, name=f"changed_inputs[{index}].role")
        roles.append(role)
        _nonempty_text(
            _required_string(item, "artifact_path"),
            name=f"changed_inputs[{index}].artifact_path",
        )
        _nonempty_text(
            _required_string(item, "schema_version"),
            name=f"changed_inputs[{index}].schema_version",
        )
        _sha256_digest(
            _required_string(item, "file_sha256"),
            name=f"changed_inputs[{index}].file_sha256",
        )
        _sha256_digest(
            _required_string(item, "semantic_sha256"),
            name=f"changed_inputs[{index}].semantic_sha256",
        )
        for field_name in (
            "change_type",
            "readiness_status",
            "readiness_consequence",
        ):
            _nonempty_text(
                _required_string(item, field_name),
                name=f"changed_inputs[{index}].{field_name}",
            )
    if roles != sorted(roles) or len(roles) != len(set(roles)):
        raise AnalysisPlanValidationError(
            "changed_inputs must be unique and sorted by role"
        )

    population = _amendment_mapping(row, "population_contract")
    required_population = {
        "mode",
        "design_id",
        "design_file_sha256",
        "design_sha256",
        "runtime_mapping_id",
        "runtime_mapping_file_sha256",
        "runtime_mapping_sha256",
        "adapter_id",
        "adapter_sha256",
        "apportionment_plan_sha256",
        "population_input_sha256",
        "cell_count",
        "target_design_units",
        "assignment_identity_policy",
        "balance_identity_policy",
        "lineage_identity_policy",
        "uncertainty_status",
        "empirical_validation_claimed",
    }
    _exact_keys(population, required_population, name="population_contract")
    if _required_string(population, "mode") != "projected_v1":
        raise AnalysisPlanValidationError("campaign population mode must be projected_v1")
    for field_name in (
        "design_file_sha256",
        "design_sha256",
        "runtime_mapping_file_sha256",
        "runtime_mapping_sha256",
        "adapter_sha256",
        "apportionment_plan_sha256",
        "population_input_sha256",
    ):
        _sha256_digest(_required_string(population, field_name), name=field_name)
    for field_name in ("cell_count", "target_design_units"):
        _strict_json_int(population.get(field_name), name=field_name)
    if _required_bool(population, "empirical_validation_claimed"):
        raise AnalysisPlanValidationError(
            "the current projected population cannot claim empirical validation"
        )
    if _required_string(population, "uncertainty_status") != "UNQUANTIFIED":
        raise AnalysisPlanValidationError(
            "current population uncertainty must remain UNQUANTIFIED"
        )

    monetary = _amendment_mapping(row, "monetary_contract")
    required_monetary = {
        "source_bundle_id",
        "source_bundle_file_sha256",
        "source_bundle_semantic_sha256",
        "source_artifact_sha256s",
        "conversion_table_sha256",
        "target_currency",
        "quote_convention",
        "scale_convention",
        "rate_period_start",
        "rate_period_end",
        "price_period_start",
        "price_period_end",
        "missing_date_policy",
        "rounding_rule",
        "rounding_boundary",
        "conversion_basis_sha256",
        "source_bundle_signature_status",
        "simulation_bridge_status",
        "rate_uncertainty_status",
        "observed_real_world_spending_claimed",
    }
    _exact_keys(monetary, required_monetary, name="monetary_contract")
    for field_name in (
        "source_bundle_file_sha256",
        "source_bundle_semantic_sha256",
        "conversion_table_sha256",
        "conversion_basis_sha256",
    ):
        _sha256_digest(_required_string(monetary, field_name), name=field_name)
    artifacts = _required_mapping(monetary, "source_artifact_sha256s")
    if not artifacts:
        raise AnalysisPlanValidationError("monetary source artifacts cannot be empty")
    for artifact_name, digest in artifacts.items():
        _nonempty_text(artifact_name, name="monetary artifact name")
        _sha256_digest(digest, name=f"monetary artifact {artifact_name}")
    if _required_string(monetary, "rate_uncertainty_status") != "UNQUANTIFIED":
        raise AnalysisPlanValidationError("point rates do not quantify rate uncertainty")
    if _required_string(monetary, "source_bundle_signature_status") != "MISSING":
        raise AnalysisPlanValidationError("current source-bundle signature is MISSING")
    if _required_string(monetary, "simulation_bridge_status") != "ILLUSTRATIVE":
        raise AnalysisPlanValidationError("current simulation bridge is ILLUSTRATIVE")
    if _required_bool(monetary, "observed_real_world_spending_claimed"):
        raise AnalysisPlanValidationError(
            "converted model values cannot be labelled observed spending"
        )

    uncertainty = _amendment_mapping(row, "uncertainty_design")
    _exact_keys(
        uncertainty,
        {
            "schema_version",
            "seed_uncertainty",
            "parameter_uncertainty",
            "monetary_rate_uncertainty",
            "population_uncertainty",
            "combined_uncertainty",
            "oat_role",
        },
        name="uncertainty_design",
    )
    if _required_string(uncertainty, "schema_version") != "1.0":
        raise AnalysisPlanValidationError("unsupported uncertainty schema")
    if _required_string(uncertainty, "oat_role") != "DIAGNOSTIC_ONLY":
        raise AnalysisPlanValidationError("OAT must remain diagnostic only")
    seed_uncertainty = _required_mapping(uncertainty, "seed_uncertainty")
    _exact_keys(
        seed_uncertainty,
        frozenset(
            {
                "status",
                "fixed_seed_count",
                "population_weights_applied_within_seed",
                "common_random_numbers",
                "identical_pretreatment_cohorts",
                "outcome_dependent_seed_exclusion_allowed",
            }
        ),
        name="seed_uncertainty",
    )
    if _required_string(seed_uncertainty, "status") != "QUANTIFIED_WHEN_COMPLETE":
        raise AnalysisPlanValidationError(
            "seed uncertainty is quantified only for a complete fixed-seed design"
        )
    if _required_int(seed_uncertainty, "fixed_seed_count") < 100:
        raise AnalysisPlanValidationError(
            "seed uncertainty requires at least 100 declared fixed seeds"
        )
    for field_name in (
        "population_weights_applied_within_seed",
        "common_random_numbers",
        "identical_pretreatment_cohorts",
    ):
        if not _required_bool(seed_uncertainty, field_name):
            raise AnalysisPlanValidationError(
                f"seed uncertainty requires {field_name}=true"
            )
    if _required_bool(
        seed_uncertainty,
        "outcome_dependent_seed_exclusion_allowed",
    ):
        raise AnalysisPlanValidationError(
            "outcome-dependent seed exclusion cannot be allowed"
        )

    parameter_uncertainty = _required_mapping(
        uncertainty,
        "parameter_uncertainty",
    )
    _exact_keys(
        parameter_uncertainty,
        frozenset(
            {
                "status",
                "design_id",
                "design_sha256",
                "method",
                "probability_interpretation",
            }
        ),
        name="parameter_uncertainty",
    )
    if _required_string(
        parameter_uncertainty,
        "status",
    ) != "ILLUSTRATIVE_DESIGN_ONLY":
        raise AnalysisPlanValidationError(
            "current parameter uncertainty must remain illustrative"
        )
    _identifier(
        _required_string(parameter_uncertainty, "design_id"),
        name="parameter uncertainty design_id",
    )
    _sha256_digest(
        _required_string(parameter_uncertainty, "design_sha256"),
        name="parameter uncertainty design_sha256",
    )
    if _required_string(
        parameter_uncertainty,
        "method",
    ) != "SEEDED_LATIN_HYPERCUBE_V1":
        raise AnalysisPlanValidationError(
            "unsupported parameter-uncertainty design method"
        )
    if _required_string(
        parameter_uncertainty,
        "probability_interpretation",
    ) != "NONE":
        raise AnalysisPlanValidationError(
            "illustrative ranges cannot receive a probability interpretation"
        )

    monetary_uncertainty = _required_mapping(
        uncertainty,
        "monetary_rate_uncertainty",
    )
    _exact_keys(
        monetary_uncertainty,
        frozenset(
            {"status", "rate_basis_sha256", "point_observation_is_distribution"}
        ),
        name="monetary_rate_uncertainty",
    )
    if _required_string(monetary_uncertainty, "status") != "UNQUANTIFIED":
        raise AnalysisPlanValidationError(
            "point-rate monetary uncertainty must remain unquantified"
        )
    _sha256_digest(
        _required_string(monetary_uncertainty, "rate_basis_sha256"),
        name="monetary uncertainty rate basis",
    )
    if _required_bool(
        monetary_uncertainty,
        "point_observation_is_distribution",
    ):
        raise AnalysisPlanValidationError(
            "an official point observation is not a rate distribution"
        )

    population_uncertainty = _required_mapping(
        uncertainty,
        "population_uncertainty",
    )
    _exact_keys(
        population_uncertainty,
        frozenset(
            {"status", "uncertainty_design_id", "exact_weighting_is_empirical_validation"}
        ),
        name="population_uncertainty",
    )
    if _required_string(population_uncertainty, "status") != "UNQUANTIFIED":
        raise AnalysisPlanValidationError(
            "current population uncertainty must remain unquantified"
        )
    _identifier(
        _required_string(population_uncertainty, "uncertainty_design_id"),
        name="population uncertainty design_id",
    )
    if _required_bool(
        population_uncertainty,
        "exact_weighting_is_empirical_validation",
    ):
        raise AnalysisPlanValidationError(
            "exact weighting cannot claim empirical population validation"
        )

    combined_uncertainty = _required_mapping(
        uncertainty,
        "combined_uncertainty",
    )
    _exact_keys(
        combined_uncertainty,
        frozenset(
            {"status", "double_counting_control", "variance_decomposition_method"}
        ),
        name="combined_uncertainty",
    )
    if _required_string(
        combined_uncertainty,
        "status",
    ) != "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST":
        raise AnalysisPlanValidationError(
            "combined uncertainty must remain unavailable while components are missing"
        )
    if _required_string(
        combined_uncertainty,
        "double_counting_control",
    ) != "one complete seed-parameter-population-rate Cartesian identity":
        raise AnalysisPlanValidationError(
            "combined uncertainty requires one complete Cartesian identity"
        )
    if _required_string(
        combined_uncertainty,
        "variance_decomposition_method",
    ) != "ORTHOGONAL_FINITE_FULL_FACTORIAL_ANOVA_SUM_OF_SQUARES_DIVIDED_BY_N_V1":
        raise AnalysisPlanValidationError(
            "unsupported combined variance-decomposition method"
        )

    convergence = _amendment_mapping(row, "convergence_rule")
    required_convergence = {
        "schema_version",
        "block_size",
        "minimum_retained_seeds",
        "maximum_mcse",
        "maximum_interval_width",
        "maximum_absolute_change",
        "maximum_relative_change",
        "maximum_invalid_rate",
        "consecutive_passing_checkpoints",
        "sensitivity_instability_allowed",
        "outcome_dependent_seed_exclusion_allowed",
        "required_uncertainty_component_handling",
    }
    _exact_keys(convergence, required_convergence, name="convergence_rule")
    minimum = _strict_json_int(
        convergence.get("minimum_retained_seeds"),
        name="minimum_retained_seeds",
    )
    if minimum < 100:
        raise AnalysisPlanValidationError("convergence requires at least 100 seeds")
    for field_name in ("block_size", "consecutive_passing_checkpoints"):
        if _required_int(convergence, field_name) <= 0:
            raise AnalysisPlanValidationError(
                f"convergence {field_name} must be positive"
            )
    for field_name in (
        "maximum_mcse",
        "maximum_interval_width",
        "maximum_absolute_change",
        "maximum_relative_change",
    ):
        value = convergence.get(field_name)
        if (
            type(value) not in {int, float}
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise AnalysisPlanValidationError(
                f"convergence {field_name} must be positive and finite"
            )
    invalid_rate = convergence.get("maximum_invalid_rate")
    if (
        type(invalid_rate) not in {int, float}
        or isinstance(invalid_rate, bool)
        or not np.isfinite(float(invalid_rate))
        or not 0.0 <= float(invalid_rate) < 1.0
    ):
        raise AnalysisPlanValidationError(
            "maximum_invalid_rate must be finite and in [0, 1)"
        )
    if _required_bool(convergence, "sensitivity_instability_allowed"):
        raise AnalysisPlanValidationError("sensitivity instability cannot be allowed")
    if _required_bool(convergence, "outcome_dependent_seed_exclusion_allowed"):
        raise AnalysisPlanValidationError("outcome-dependent exclusion cannot be allowed")
    if _required_string(convergence, "required_uncertainty_component_handling") != "FAIL_CLOSED":
        raise AnalysisPlanValidationError("missing uncertainty components must fail closed")

    attestation = _amendment_mapping(row, "execution_attestation")
    for field_name in (
        "receipt_schema_version",
        "pre_run_required",
        "post_run_required",
        "clean_tree_required",
        "environment_match_required",
        "mismatch_handling",
    ):
        if field_name not in attestation:
            raise AnalysisPlanValidationError(
                f"execution_attestation missing {field_name}"
            )
    if any(
        not _required_bool(attestation, field_name)
        for field_name in (
            "pre_run_required",
            "post_run_required",
            "clean_tree_required",
            "environment_match_required",
        )
    ):
        raise AnalysisPlanValidationError("all execution attestation gates are required")
    if _required_string(attestation, "mismatch_handling") != "REJECT_OR_INVALIDATE":
        raise AnalysisPlanValidationError("attestation mismatches must invalidate the run")

    flow = _amendment_mapping(row, "simulation_flow")
    if _required_string(flow, "execution_layer") != "policy_welfare_v1":
        raise AnalysisPlanValidationError(
            "the preserved primary estimand requires policy_welfare_v1"
        )
    if flow.get("strategic_world_layer_included") is not False:
        raise AnalysisPlanValidationError(
            "the strategic World layer cannot be silently composed"
        )

    readiness = _amendment_mapping(row, "readiness_consequences")
    if readiness.get("campaign_ready") is not False:
        raise AnalysisPlanValidationError("schema-v3 amendment must fail closed")
    blockers = readiness.get("blockers")
    if type(blockers) is not list or not all(type(item) is str for item in blockers):
        raise AnalysisPlanValidationError("readiness blockers must be a string list")
    missing_blockers = sorted(set(_V3_CAMPAIGN_BLOCKERS).difference(blockers))
    if missing_blockers:
        raise AnalysisPlanValidationError(
            "readiness consequences omit fixed blockers: " + ", ".join(missing_blockers)
        )


def _campaign_blockers(schema_version: str) -> tuple[str, ...]:
    if schema_version == ANALYSIS_PLAN_SCHEMA_VERSION:
        return _CAMPAIGN_BLOCKERS
    if schema_version == PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION:
        return _V2_CAMPAIGN_BLOCKERS
    if schema_version == CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION:
        return _V3_CAMPAIGN_BLOCKERS
    raise AnalysisPlanValidationError(
        "unsupported prospective analysis-plan schema version"
    )


_POPULATION_OUTCOME_SEMANTICS = _outcome_semantics_registry()


__all__ = [
    "ANALYSIS_PLAN_SCHEMA_VERSION",
    "ALL_HOUSEHOLD_TYPES_ID",
    "ALL_MONTHLY_DISPOSABLE_INCOME_BANDS_ID",
    "CAMPAIGN_ANALYSIS_PLAN_SCHEMA_VERSION",
    "EXPLORATORY_ANALYSIS_PLAN_KIND",
    "EXPLORATORY_ANALYSIS_PLAN_SCHEMA_VERSION",
    "PROSPECTIVE_ANALYSIS_PLAN_SCHEMA_VERSION",
    "MAX_ANALYSIS_PLAN_BYTES",
    "AnalysisEstimandRole",
    "AnalysisPlanCampaignError",
    "AnalysisPlanRegistrationStatus",
    "AnalysisPlanValidationError",
    "AnalysisPlanVerificationError",
    "CanonicalPopulationInclusionPredicate",
    "FixedSeedStoppingRule",
    "ExploratoryAnalysisPlan",
    "LoadedExploratoryAnalysisPlan",
    "LoadedProspectiveAnalysisPlan",
    "PlannedPopulationEstimand",
    "PopulationMinorFilter",
    "PopulationOutcomeMetric",
    "PopulationOutcomeMetricSemantics",
    "PrimaryAggregateRule",
    "ProspectiveAnalysisPlan",
    "analysis_plan_harm_weights_sha256",
    "build_exploratory_analysis_plan",
    "build_prospective_analysis_plan",
    "evaluate_population_inclusion",
    "load_prospective_analysis_plan",
    "load_exploratory_analysis_plan",
    "population_outcome_semantics",
    "verify_loaded_prospective_analysis_plan",
    "verify_loaded_exploratory_analysis_plan",
    "verify_exploratory_analysis_plan_parent",
    "verify_prospective_analysis_plan_bindings",
]
