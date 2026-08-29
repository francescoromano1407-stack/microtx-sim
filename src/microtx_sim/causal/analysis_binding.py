"""Post-run binding of prospective plans to exact population results.

The plan schema deliberately remains unregistered and campaign-ineligible.
This module does not change that status: it resolves a plan against a fully
retained :class:`PolicyBatchResult`, executes only the plan's canonical
pre-treatment predicate, and produces exact writer-ready population estimands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Mapping

import numpy as np
import numpy.typing as npt

from ..agents.players import ProjectedPopulationCellMetadata
from ..data.lineage import ProfileInputLineage
from ..data.population_execution import (
    PopulationExecutionLineage,
    PopulationSeedExecutionRecord,
    population_execution_input_sha256,
)
from ..data.population_evidence import (
    PopulationGamingState,
    PopulationPayerHistoryState,
)
from ..data.population_projection import (
    PopulationProjectionAdapter,
    verify_population_projection_adapter,
)
from ..metrics.population_estimands import (
    EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
    POPULATION_ESTIMAND_SCHEMA_VERSION,
    TARGET_POPULATION_OUTPUT_PROFILE,
    ExactPopulationWeights,
    PopulationAnalysisUnit,
    PopulationContrast,
    PopulationEstimandAlgorithm,
    PopulationEstimandResult,
    PopulationEstimandSpec,
    PopulationInclusionRule,
    PopulationMetricKind,
    PopulationNormalization,
    paired_weighted_mean_difference,
)
from ..rng import validate_seed
from ..simulation.policy_orchestrator import PolicyScenarioResult
from .analysis_plan import (
    PlannedPopulationEstimand,
    PopulationOutcomeMetric,
    PopulationOutcomeMetricSemantics,
    ProspectiveAnalysisPlan,
    analysis_plan_harm_weights_sha256,
    evaluate_population_inclusion,
    verify_prospective_analysis_plan_bindings,
)
from .batch import PolicyBatchResult, PolicyBatchSpec, PolicyRunInputs
from .design import assess_causal_design
from .scenarios import ScenarioId

if TYPE_CHECKING:
    from ..outputs.metric_contracts import OutputMetricContract


ANALYSIS_BINDING_SCHEMA_VERSION: Final[str] = "1.0"

_CAMPAIGN_BLOCKERS: Final[tuple[str, ...]] = (
    "analysis_binding.external_registration=unregistered",
    "analysis_binding.schema_v1=campaign_ineligible",
    "analysis_binding.execution_calendar_anchor=unbound",
    "analysis_binding.cross_seed_aggregation_uncertainty=unresolved",
    "analysis_binding.model_implementation_environment_identity=unbound",
)

_EXPECTED_PLAYER_CONTRACT_COLUMN: Final[
    Mapping[PopulationOutcomeMetric, str]
] = MappingProxyType(
    {
        PopulationOutcomeMetric.COMPOSITE_HARM: "composite_harm",
        PopulationOutcomeMetric.MONETARY_HARM_SCORE: "monetary_harm",
        PopulationOutcomeMetric.OPPORTUNITY_COST_SCORE: "opportunity_cost",
        PopulationOutcomeMetric.SLEEP_BURDEN_SCORE: "sleep_burden",
        PopulationOutcomeMetric.EDUCATION_WORK_BURDEN_SCORE: (
            "education_work_burden"
        ),
        PopulationOutcomeMetric.FAMILY_SOCIAL_BURDEN_SCORE: "social_burden",
        PopulationOutcomeMetric.WELLBEING_BURDEN_SCORE: "wellbeing_burden",
        PopulationOutcomeMetric.ENJOYMENT: "enjoyment",
        PopulationOutcomeMetric.HIGH_RISK_INDICATOR: "high_risk",
    }
)


class AnalysisBindingValidationError(ValueError):
    """Raised when a plan cannot be bound to exact execution truth."""


@dataclass(frozen=True, slots=True)
class SeedAnalysisBinding:
    """One planned estimand resolved for one fixed seed."""

    schema_version: str
    seed: int
    planned_estimand: PlannedPopulationEstimand
    population_seed_record_sha256: str
    eligibility_sha256: str
    reference_outcome_sha256: str
    comparison_outcome_sha256: str
    metric_contract_sha256: str
    selected_weights: ExactPopulationWeights
    spec: PopulationEstimandSpec
    result: PopulationEstimandResult
    binding_sha256: str
    preregistered: bool = field(default=False, init=False)
    campaign_ready: bool = field(default=False, init=False)
    campaign_blockers: tuple[str, ...] = field(
        default=_CAMPAIGN_BLOCKERS,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != ANALYSIS_BINDING_SCHEMA_VERSION:
            raise AnalysisBindingValidationError(
                "unsupported seed analysis-binding schema version"
            )
        validate_seed(self.seed, name="analysis binding seed")
        if type(self.planned_estimand) is not PlannedPopulationEstimand:
            raise TypeError(
                "planned_estimand must be PlannedPopulationEstimand"
            )
        PlannedPopulationEstimand.__post_init__(self.planned_estimand)
        for name in (
            "population_seed_record_sha256",
            "eligibility_sha256",
            "reference_outcome_sha256",
            "comparison_outcome_sha256",
            "metric_contract_sha256",
            "binding_sha256",
        ):
            _sha256_digest(getattr(self, name), name=name)
        if type(self.selected_weights) is not ExactPopulationWeights:
            raise TypeError("selected_weights must be ExactPopulationWeights")
        ExactPopulationWeights.__post_init__(self.selected_weights)
        if type(self.spec) is not PopulationEstimandSpec:
            raise TypeError("spec must be PopulationEstimandSpec")
        PopulationEstimandSpec.__post_init__(self.spec)
        PopulationInclusionRule.__post_init__(self.spec.inclusion_rule)
        if self.spec.currency is not None:
            type(self.spec.currency).__post_init__(self.spec.currency)
        type(self.spec.period).__post_init__(self.spec.period)
        if type(self.result) is not PopulationEstimandResult:
            raise TypeError("result must be PopulationEstimandResult")
        PopulationEstimandResult.__post_init__(self.result)

        planned = self.planned_estimand
        semantics = planned.outcome_semantics
        if semantics.metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS:
            raise AnalysisBindingValidationError(
                "analysis binding schema v1 prohibits money outcomes without "
                "an executed currency/price-period conversion"
            )
        _contract, expected_contract_sha256 = _resolve_metric_contract(
            planned,
            semantics,
        )
        if self.metric_contract_sha256 != expected_contract_sha256:
            raise AnalysisBindingValidationError(
                "selected metric-contract snapshot digest differs from the registry"
            )
        expected_spec_bindings = {
            "estimand_id": _resolved_estimand_id(planned, self.seed),
            "target_population_id": _target_population_id(planned),
            "design_weights_sha256": self.selected_weights.design_sha256,
            "metric_contract_sha256": self.metric_contract_sha256,
            "output_profile_id": TARGET_POPULATION_OUTPUT_PROFILE,
            "output_profile_schema_sha256": (
                _target_output_profile_sha256()
            ),
            "analysis_unit": PopulationAnalysisUnit.PLAYER_PERSON,
            "inclusion_rule": planned.inclusion_predicate.rule,
            "metric_name": semantics.metric_name,
            "metric_kind": semantics.metric_kind,
            "metric_scale": semantics.metric_scale,
            "contrast": PopulationContrast.TREATED_MINUS_CONTROL,
            "algorithm": (
                PopulationEstimandAlgorithm.PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1
            ),
            "normalization": PopulationNormalization.DIVIDE_BY_WEIGHT_SUM,
            "period": planned.period,
            "currency": planned.currency,
        }
        mismatches = sorted(
            name
            for name, expected in expected_spec_bindings.items()
            if getattr(self.spec, name) != expected
        )
        if mismatches:
            raise AnalysisBindingValidationError(
                "resolved estimand spec differs from its plan/weights: "
                + ", ".join(mismatches)
            )
        if (
            self.result.estimand_sha256 != self.spec.estimand_sha256
            or self.result.design_weights_sha256
            != self.selected_weights.design_sha256
            or self.result.algorithm is not self.spec.algorithm
            or self.result.metric_name != self.spec.metric_name
            or self.result.contrast is not self.spec.contrast
            or self.result.normalization is not self.spec.normalization
            or self.result.player_count != len(self.selected_weights.player_ids)
        ):
            raise AnalysisBindingValidationError(
                "population estimand result differs from its exact resolved spec"
            )
        if self.preregistered or self.campaign_ready:
            raise AnalysisBindingValidationError(
                "schema-v1 run bindings cannot be preregistered or campaign-ready"
            )
        if self.campaign_blockers != _CAMPAIGN_BLOCKERS:
            raise AnalysisBindingValidationError(
                "schema-v1 analysis-binding campaign blockers are fixed"
            )
        if self.binding_sha256 != _canonical_sha256(
            self.attestation_payload()
        ):
            raise AnalysisBindingValidationError(
                "seed analysis binding SHA-256 differs from its exact payload"
            )

    @property
    def selected_player_count(self) -> int:
        return len(self.selected_weights.player_ids)

    @property
    def writer_pair(
        self,
    ) -> tuple[PopulationEstimandSpec, PopulationEstimandResult]:
        SeedAnalysisBinding.__post_init__(self)
        return self.spec, self.result

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "seed_decimal": str(self.seed),
            "planned_estimand": self.planned_estimand.snapshot(),
            "planned_estimand_sha256": (
                self.planned_estimand.estimand_sha256
            ),
            "reference_scenario_id": (
                self.planned_estimand.reference_scenario_id.value
            ),
            "comparison_scenario_id": (
                self.planned_estimand.comparison_scenario_id.value
            ),
            "contrast_direction": self.planned_estimand.contrast_direction,
            "population_seed_record_sha256": (
                self.population_seed_record_sha256
            ),
            "eligibility_sha256": self.eligibility_sha256,
            "selected_player_count": self.selected_player_count,
            "selected_player_count_decimal": str(self.selected_player_count),
            "selected_design_weights_sha256": (
                self.selected_weights.design_sha256
            ),
            "reference_outcome_sha256": self.reference_outcome_sha256,
            "comparison_outcome_sha256": self.comparison_outcome_sha256,
            "metric_contract_id": self.planned_estimand.metric_contract_id,
            "metric_contract_sha256": self.metric_contract_sha256,
            "spec": self.spec.snapshot(),
            "result": self.result.snapshot(),
            "preregistered": self.preregistered,
            "campaign_ready": self.campaign_ready,
            "campaign_blockers": list(self.campaign_blockers),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "binding_sha256": self.binding_sha256,
        }


@dataclass(frozen=True, slots=True)
class RunAnalysisBinding:
    """Content-addressed resolution of a plan against one completed batch."""

    schema_version: str
    plan: ProspectiveAnalysisPlan
    causal_design_sha256: str
    batch_spec_sha256: str
    model_inputs_sha256: str
    population_input_sha256: str
    profile_input_sha256: str
    population_lineage_sha256: str
    metric_contract_registry_sha256: str
    harm_weights_sha256: str
    output_profile_schema_sha256: str
    seeds: tuple[int, ...]
    seed_bindings: tuple[SeedAnalysisBinding, ...]
    binding_sha256: str
    preregistered: bool = field(default=False, init=False)
    campaign_ready: bool = field(default=False, init=False)
    campaign_blockers: tuple[str, ...] = field(
        default=_CAMPAIGN_BLOCKERS,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != ANALYSIS_BINDING_SCHEMA_VERSION:
            raise AnalysisBindingValidationError(
                "unsupported run analysis-binding schema version"
            )
        if type(self.plan) is not ProspectiveAnalysisPlan:
            raise TypeError("plan must be ProspectiveAnalysisPlan")
        ProspectiveAnalysisPlan.__post_init__(self.plan)
        _reject_unresolved_money_estimands(self.plan)
        for name in (
            "causal_design_sha256",
            "batch_spec_sha256",
            "model_inputs_sha256",
            "population_input_sha256",
            "profile_input_sha256",
            "population_lineage_sha256",
            "metric_contract_registry_sha256",
            "harm_weights_sha256",
            "output_profile_schema_sha256",
            "binding_sha256",
        ):
            _sha256_digest(getattr(self, name), name=name)
        expected_plan_bindings = {
            "expected_causal_design_sha256": self.causal_design_sha256,
            "expected_batch_spec_sha256": self.batch_spec_sha256,
            "expected_model_inputs_sha256": self.model_inputs_sha256,
            "expected_population_input_sha256": self.population_input_sha256,
            "expected_profile_input_sha256": self.profile_input_sha256,
            "expected_metric_contract_sha256": (
                self.metric_contract_registry_sha256
            ),
            "expected_harm_weights_sha256": self.harm_weights_sha256,
            "expected_output_profile_sha256": (
                self.output_profile_schema_sha256
            ),
        }
        mismatches = sorted(
            name.removeprefix("expected_")
            for name, expected in expected_plan_bindings.items()
            if getattr(self.plan, name) != expected
        )
        if mismatches:
            raise AnalysisBindingValidationError(
                "run binding differs from its prospective plan: "
                + ", ".join(mismatches)
            )
        if type(self.seeds) is not tuple or self.seeds != tuple(
            sorted(self.seeds)
        ):
            raise AnalysisBindingValidationError(
                "run analysis-binding seeds must be an ascending exact tuple"
            )
        for seed in self.seeds:
            validate_seed(seed, name="run analysis-binding seed")
        if self.seeds != self.plan.stopping_rule.seeds:
            raise AnalysisBindingValidationError(
                "run analysis-binding seeds differ from the fixed stopping rule"
            )
        if type(self.seed_bindings) is not tuple or any(
            type(item) is not SeedAnalysisBinding
            for item in self.seed_bindings
        ):
            raise TypeError(
                "seed_bindings must be an exact tuple of SeedAnalysisBinding"
            )
        for item in self.seed_bindings:
            SeedAnalysisBinding.__post_init__(item)
        expected_order = tuple(
            (seed, estimand.estimand_id)
            for seed in self.seeds
            for estimand in self.plan.estimands
        )
        observed_order = tuple(
            (item.seed, item.planned_estimand.estimand_id)
            for item in self.seed_bindings
        )
        if observed_order != expected_order:
            raise AnalysisBindingValidationError(
                "seed bindings do not exactly cover the fixed seeds and plan estimands"
            )
        plan_estimands = {
            item.estimand_id: item for item in self.plan.estimands
        }
        if any(
            item.planned_estimand
            != plan_estimands[item.planned_estimand.estimand_id]
            for item in self.seed_bindings
        ):
            raise AnalysisBindingValidationError(
                "seed binding scenario direction or estimand fields differ from the plan"
            )
        spec_ids = tuple(item.spec.estimand_id for item in self.seed_bindings)
        if len(set(spec_ids)) != len(spec_ids):
            raise AnalysisBindingValidationError(
                "resolved writer estimand IDs must be unique"
            )
        if self.output_profile_schema_sha256 != (
            _target_output_profile_sha256()
        ):
            raise AnalysisBindingValidationError(
                "run binding targets the wrong output-profile schema"
            )
        if self.preregistered or self.campaign_ready:
            raise AnalysisBindingValidationError(
                "schema-v1 run bindings cannot be preregistered or campaign-ready"
            )
        if self.campaign_blockers != _CAMPAIGN_BLOCKERS:
            raise AnalysisBindingValidationError(
                "schema-v1 analysis-binding campaign blockers are fixed"
            )
        if self.binding_sha256 != _canonical_sha256(
            self.attestation_payload()
        ):
            raise AnalysisBindingValidationError(
                "run analysis binding SHA-256 differs from its exact payload"
            )

    @property
    def writer_pairs(
        self,
    ) -> tuple[tuple[PopulationEstimandSpec, PopulationEstimandResult], ...]:
        RunAnalysisBinding.__post_init__(self)
        return tuple(item.writer_pair for item in self.seed_bindings)

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan.plan_id,
            "plan_sha256": self.plan.plan_sha256,
            "causal_design_sha256": self.causal_design_sha256,
            "batch_spec_sha256": self.batch_spec_sha256,
            "model_inputs_sha256": self.model_inputs_sha256,
            "population_input_sha256": self.population_input_sha256,
            "profile_input_sha256": self.profile_input_sha256,
            "population_lineage_sha256": self.population_lineage_sha256,
            "metric_contract_registry_sha256": (
                self.metric_contract_registry_sha256
            ),
            "harm_weights_sha256": self.harm_weights_sha256,
            "output_profile_id": TARGET_POPULATION_OUTPUT_PROFILE,
            "output_profile_schema_sha256": self.output_profile_schema_sha256,
            "seeds": list(self.seeds),
            "seed_decimal_strings": [str(seed) for seed in self.seeds],
            "seed_bindings": [item.snapshot() for item in self.seed_bindings],
            "preregistered": self.preregistered,
            "campaign_ready": self.campaign_ready,
            "campaign_blockers": list(self.campaign_blockers),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "binding_sha256": self.binding_sha256,
        }

    def manifest_payload(self) -> dict[str, object]:
        RunAnalysisBinding.__post_init__(self)
        return self.snapshot()


def validate_analysis_plan_inputs(
    plan: ProspectiveAnalysisPlan,
    *,
    batch_spec: PolicyBatchSpec,
    run_inputs: PolicyRunInputs,
    population_adapter: PopulationProjectionAdapter,
    profile_input_lineage: ProfileInputLineage,
) -> None:
    """Fail before treatment when any prospective runtime input differs."""

    if type(plan) is not ProspectiveAnalysisPlan:
        raise TypeError("plan must be ProspectiveAnalysisPlan")
    if type(batch_spec) is not PolicyBatchSpec:
        raise TypeError("batch_spec must be PolicyBatchSpec")
    if type(run_inputs) is not PolicyRunInputs:
        raise TypeError("run_inputs must be PolicyRunInputs")
    if type(population_adapter) is not PopulationProjectionAdapter:
        raise TypeError(
            "population_adapter must be PopulationProjectionAdapter"
        )
    if type(profile_input_lineage) is not ProfileInputLineage:
        raise TypeError(
            "profile_input_lineage must be ProfileInputLineage"
        )
    try:
        ProspectiveAnalysisPlan.__post_init__(plan)
        _reattest_batch_spec(batch_spec)
        _reject_unresolved_money_estimands(plan)
        for estimand in plan.estimands:
            _resolve_metric_contract(estimand, estimand.outcome_semantics)
        PolicyRunInputs.__post_init__(run_inputs)
        ProfileInputLineage.__post_init__(profile_input_lineage)
        adapter = verify_population_projection_adapter(population_adapter)
        _validate_inclusion_predicate_domains(plan, adapter)
        verify_prospective_analysis_plan_bindings(
            plan,
            causal_design_sha256=assess_causal_design(
                batch_spec.scenarios
            ).design_sha256(),
            batch_spec_sha256=batch_spec.snapshot_sha256(),
            model_inputs_sha256=run_inputs.snapshot_sha256(),
            population_input_sha256=population_execution_input_sha256(
                adapter
            ),
            profile_input_sha256=profile_input_lineage.fingerprint_sha256,
            metric_contract_sha256=_metric_contract_registry_sha256(),
            harm_weights_sha256=analysis_plan_harm_weights_sha256(
                run_inputs.harm_weights
            ),
            output_profile_sha256=(
                _target_output_profile_sha256()
            ),
            seeds=batch_spec.seeds,
        )
        _validate_estimand_period_durations(plan, batch_spec)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, AnalysisBindingValidationError):
            raise
        raise AnalysisBindingValidationError(
            f"analysis plan pre-execution validation failed: {exc}"
        ) from exc


def _validate_inclusion_predicate_domains(
    plan: ProspectiveAnalysisPlan,
    adapter: PopulationProjectionAdapter,
) -> None:
    """Reject explicit categorical levels absent from the exact adapter domain."""

    cells = tuple(item.projection_cell for item in adapter.cells)
    domains: tuple[
        tuple[str, frozenset[object]], ...
    ] = (
        (
            "jurisdiction_codes",
            frozenset(cell.jurisdiction_code for cell in cells),
        ),
        (
            "monthly_disposable_income_band_ids",
            frozenset(
                cell.monthly_disposable_income_band_id for cell in cells
            ),
        ),
        (
            "household_type_ids",
            frozenset(cell.household_type for cell in cells),
        ),
        (
            "gaming_states",
            frozenset(
                PopulationGamingState.GAMER
                if cell.baseline_gamer
                else PopulationGamingState.NON_GAMER
                for cell in cells
            ),
        ),
        (
            "payer_history_states",
            frozenset(
                PopulationPayerHistoryState.EVER_PAYER
                if cell.baseline_ever_payer
                else PopulationPayerHistoryState.NEVER_PAYER
                for cell in cells
            ),
        ),
    )
    for estimand in plan.estimands:
        predicate = estimand.inclusion_predicate
        for field_name, domain in domains:
            requested = getattr(predicate, field_name)
            if not requested:
                continue
            missing = set(requested).difference(domain)
            if missing:
                rendered_missing = sorted(_domain_value(item) for item in missing)
                rendered_domain = sorted(_domain_value(item) for item in domain)
                raise AnalysisBindingValidationError(
                    f"estimand {estimand.estimand_id!r} inclusion field "
                    f"{field_name} contains values outside the exact population "
                    f"adapter domain: missing={rendered_missing!r}; "
                    f"domain={rendered_domain!r}"
                )


def _domain_value(value: object) -> str:
    if isinstance(value, (PopulationGamingState, PopulationPayerHistoryState)):
        return value.value
    return str(value)


def resolve_run_analysis_binding(
    plan: ProspectiveAnalysisPlan,
    batch: PolicyBatchResult,
) -> RunAnalysisBinding:
    """Resolve a prospective plan to exact per-seed paired estimand results."""

    if type(plan) is not ProspectiveAnalysisPlan:
        raise TypeError("plan must be ProspectiveAnalysisPlan")
    if type(batch) is not PolicyBatchResult:
        raise TypeError("batch must be PolicyBatchResult")
    try:
        ProspectiveAnalysisPlan.__post_init__(plan)
        PolicyBatchResult.__post_init__(batch)
        lineage = batch.population_execution_lineage
        if type(lineage) is not PopulationExecutionLineage:
            raise AnalysisBindingValidationError(
                "analysis binding requires exact projected population lineage"
            )
        PopulationExecutionLineage.__post_init__(lineage)
        profile_lineage = batch.profile_input_lineage
        if type(profile_lineage) is not ProfileInputLineage:
            raise AnalysisBindingValidationError(
                "analysis binding requires exact profile input lineage"
            )
        ProfileInputLineage.__post_init__(profile_lineage)
        validate_analysis_plan_inputs(
            plan,
            batch_spec=batch.spec,
            run_inputs=batch.run_inputs,
            population_adapter=lineage.adapter,
            profile_input_lineage=profile_lineage,
        )
        if lineage.input_sha256 != plan.expected_population_input_sha256:
            raise AnalysisBindingValidationError(
                "post-run population input differs from the prospective plan"
            )
        if (
            profile_lineage.fingerprint_sha256
            != plan.expected_profile_input_sha256
        ):
            raise AnalysisBindingValidationError(
                "post-run profile input differs from the prospective plan"
            )
        if tuple(record.seed for record in lineage.seed_records) != (
            plan.stopping_rule.seeds
        ):
            raise AnalysisBindingValidationError(
                "post-run population records differ from the fixed stopping rule"
            )

        record_by_key = {
            (record.result.seed, record.result.scenario.scenario_id): record
            for record in batch.records
        }
        projected_cells = _projected_cells(
            lineage.adapter,
            batch.country_profiles,
        )
        jurisdiction_codes = tuple(
            profile.code for profile in batch.country_profiles
        )
        seed_bindings: list[SeedAnalysisBinding] = []
        for seed in plan.stopping_rule.seeds:
            population_record = lineage.record_for_seed(seed)
            PopulationSeedExecutionRecord.__post_init__(population_record)
            for planned in plan.estimands:
                reference = _required_result(
                    record_by_key,
                    seed=seed,
                    scenario_id=planned.reference_scenario_id,
                )
                comparison = _required_result(
                    record_by_key,
                    seed=seed,
                    scenario_id=planned.comparison_scenario_id,
                )
                seed_bindings.append(
                    _resolve_seed_estimand(
                        planned,
                        seed=seed,
                        population_record=population_record,
                        reference=reference,
                        comparison=comparison,
                        jurisdiction_codes=jurisdiction_codes,
                        projected_cells=projected_cells,
                        target_evidence_sha256=(
                            lineage.adapter.calibration_target_sha256
                        ),
                    )
                )

        causal_design_sha256 = assess_causal_design(
            batch.spec.scenarios
        ).design_sha256()
        batch_spec_sha256 = batch.spec.snapshot_sha256()
        model_inputs_sha256 = batch.run_inputs.snapshot_sha256()
        population_input_sha256 = population_execution_input_sha256(
            lineage.adapter
        )
        profile_input_sha256 = profile_lineage.fingerprint_sha256
        metric_registry_sha256 = _metric_contract_registry_sha256()
        harm_weights_sha256 = analysis_plan_harm_weights_sha256(
            batch.run_inputs.harm_weights
        )
        payload = _run_attestation_payload(
            plan=plan,
            causal_design_sha256=causal_design_sha256,
            batch_spec_sha256=batch_spec_sha256,
            model_inputs_sha256=model_inputs_sha256,
            population_input_sha256=population_input_sha256,
            profile_input_sha256=profile_input_sha256,
            population_lineage_sha256=lineage.lineage_sha256,
            metric_contract_registry_sha256=metric_registry_sha256,
            harm_weights_sha256=harm_weights_sha256,
            seeds=plan.stopping_rule.seeds,
            seed_bindings=tuple(seed_bindings),
        )
        return RunAnalysisBinding(
            schema_version=ANALYSIS_BINDING_SCHEMA_VERSION,
            plan=plan,
            causal_design_sha256=causal_design_sha256,
            batch_spec_sha256=batch_spec_sha256,
            model_inputs_sha256=model_inputs_sha256,
            population_input_sha256=population_input_sha256,
            profile_input_sha256=profile_input_sha256,
            population_lineage_sha256=lineage.lineage_sha256,
            metric_contract_registry_sha256=metric_registry_sha256,
            harm_weights_sha256=harm_weights_sha256,
            output_profile_schema_sha256=(
                _target_output_profile_sha256()
            ),
            seeds=plan.stopping_rule.seeds,
            seed_bindings=tuple(seed_bindings),
            binding_sha256=_canonical_sha256(payload),
        )
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        if isinstance(exc, AnalysisBindingValidationError):
            raise
        raise AnalysisBindingValidationError(
            f"post-run analysis binding failed: {exc}"
        ) from exc


def _resolve_seed_estimand(
    planned: PlannedPopulationEstimand,
    *,
    seed: int,
    population_record: PopulationSeedExecutionRecord,
    reference: PolicyScenarioResult,
    comparison: PolicyScenarioResult,
    jurisdiction_codes: tuple[str, ...],
    projected_cells: tuple[ProjectedPopulationCellMetadata, ...],
    target_evidence_sha256: str,
) -> SeedAnalysisBinding:
    predicate = planned.inclusion_predicate
    reference_mask = evaluate_population_inclusion(
        predicate,
        jurisdiction_codes=jurisdiction_codes,
        jurisdiction=reference.jurisdiction,
        age_years=reference.age_years,
        is_minor=reference.is_minor,
        projected_cells=projected_cells,
        cell_indices=population_record.cell_indices,
    )
    comparison_mask = evaluate_population_inclusion(
        predicate,
        jurisdiction_codes=jurisdiction_codes,
        jurisdiction=comparison.jurisdiction,
        age_years=comparison.age_years,
        is_minor=comparison.is_minor,
        projected_cells=projected_cells,
        cell_indices=population_record.cell_indices,
    )
    if not np.array_equal(reference_mask, comparison_mask):
        raise AnalysisBindingValidationError(
            "pre-treatment eligibility differs across compared branches"
        )
    selected_positions = np.flatnonzero(reference_mask)
    if selected_positions.size == 0:
        raise AnalysisBindingValidationError(
            f"planned estimand {planned.estimand_id!r} selects no players"
        )
    full_weights = population_record.exact_weights
    expected_ids = np.asarray(full_weights.player_ids, dtype=np.int64)
    if not np.array_equal(reference.player_ids, expected_ids) or not np.array_equal(
        comparison.player_ids,
        expected_ids,
    ):
        raise AnalysisBindingValidationError(
            "compared branches differ from exact population player ordering"
        )
    positions = tuple(int(index) for index in selected_positions)
    selected_weights = ExactPopulationWeights(
        schema_version=EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
        player_ids=tuple(full_weights.player_ids[index] for index in positions),
        weight_numerators=tuple(
            full_weights.weight_numerators[index] for index in positions
        ),
        weight_denominators=tuple(
            full_weights.weight_denominators[index] for index in positions
        ),
    )
    semantics = planned.outcome_semantics
    contract, contract_sha256 = _resolve_metric_contract(planned, semantics)
    reference_outcome = _outcome_array(reference, semantics)[reference_mask]
    comparison_outcome = _outcome_array(comparison, semantics)[comparison_mask]
    reference_outcome_sha256 = _selected_outcome_sha256(
        semantics,
        selected_weights.player_ids,
        reference_outcome,
    )
    comparison_outcome_sha256 = _selected_outcome_sha256(
        semantics,
        selected_weights.player_ids,
        comparison_outcome,
    )
    spec = PopulationEstimandSpec(
        schema_version=POPULATION_ESTIMAND_SCHEMA_VERSION,
        estimand_id=_resolved_estimand_id(planned, seed),
        target_population_id=_target_population_id(planned),
        target_evidence_sha256=target_evidence_sha256,
        design_weights_sha256=selected_weights.design_sha256,
        runtime_projection_sha256=(
            population_record.runtime_projection_sha256
        ),
        balance_report_sha256=population_record.balance.balance_sha256,
        metric_contract_sha256=contract_sha256,
        output_profile_id=TARGET_POPULATION_OUTPUT_PROFILE,
        output_profile_schema_sha256=(
            _target_output_profile_sha256()
        ),
        analysis_unit=PopulationAnalysisUnit.PLAYER_PERSON,
        inclusion_rule=predicate.rule,
        metric_name=semantics.metric_name,
        metric_kind=semantics.metric_kind,
        metric_scale=semantics.metric_scale,
        contrast=PopulationContrast.TREATED_MINUS_CONTROL,
        algorithm=(
            PopulationEstimandAlgorithm.PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1
        ),
        normalization=PopulationNormalization.DIVIDE_BY_WEIGHT_SUM,
        period=planned.period,
        currency=planned.currency,
    )
    reference_values = _primitive_values(reference_outcome, semantics)
    comparison_values = _primitive_values(comparison_outcome, semantics)
    result = paired_weighted_mean_difference(
        spec,
        selected_weights,
        comparison_values,
        selected_weights,
        reference_values,
    )
    eligibility_sha256 = _eligibility_sha256(
        seed=seed,
        predicate_sha256=predicate.predicate_sha256,
        ordered_player_ids_sha256=(
            population_record.ordered_player_ids_sha256
        ),
        mask=reference_mask,
    )
    payload = _seed_attestation_payload(
        seed=seed,
        planned=planned,
        population_seed_record_sha256=(
            population_record.seed_record_sha256
        ),
        eligibility_sha256=eligibility_sha256,
        selected_weights=selected_weights,
        reference_outcome_sha256=reference_outcome_sha256,
        comparison_outcome_sha256=comparison_outcome_sha256,
        metric_contract_sha256=contract_sha256,
        spec=spec,
        result=result,
    )
    del contract
    return SeedAnalysisBinding(
        schema_version=ANALYSIS_BINDING_SCHEMA_VERSION,
        seed=seed,
        planned_estimand=planned,
        population_seed_record_sha256=(
            population_record.seed_record_sha256
        ),
        eligibility_sha256=eligibility_sha256,
        reference_outcome_sha256=reference_outcome_sha256,
        comparison_outcome_sha256=comparison_outcome_sha256,
        metric_contract_sha256=contract_sha256,
        selected_weights=selected_weights,
        spec=spec,
        result=result,
        binding_sha256=_canonical_sha256(payload),
    )


def _projected_cells(
    adapter: PopulationProjectionAdapter,
    country_profiles: tuple[object, ...],
) -> tuple[ProjectedPopulationCellMetadata, ...]:
    codes = tuple(getattr(profile, "code", None) for profile in country_profiles)
    if not codes or any(type(code) is not str for code in codes):
        raise AnalysisBindingValidationError(
            "batch country profiles lack canonical jurisdiction codes"
        )
    if len(set(codes)) != len(codes):
        raise AnalysisBindingValidationError(
            "batch country profile jurisdiction codes repeat"
        )
    code_to_index = {code: index for index, code in enumerate(codes)}
    cells: list[ProjectedPopulationCellMetadata] = []
    for adapter_cell in adapter.cells:
        projected = adapter_cell.projection_cell
        try:
            jurisdiction_index = code_to_index[projected.jurisdiction_code]
        except KeyError as exc:
            raise AnalysisBindingValidationError(
                "population adapter jurisdiction is absent from batch profiles"
            ) from exc
        cells.append(
            ProjectedPopulationCellMetadata(
                cell_id=projected.cell_id,
                jurisdiction_code=projected.jurisdiction_code,
                jurisdiction_index=jurisdiction_index,
                age_min_inclusive=projected.age_min_inclusive,
                age_max_exclusive=projected.age_max_exclusive,
                monthly_disposable_income_band_id=(
                    projected.monthly_disposable_income_band_id
                ),
                monthly_disposable_income_min_cents=(
                    projected.monthly_disposable_income_min_cents
                ),
                monthly_disposable_income_max_cents_exclusive=(
                    projected.monthly_disposable_income_max_cents_exclusive
                ),
                household_type=projected.household_type,
                modeled_players_per_household=(
                    projected.modeled_players_per_household
                ),
                baseline_gamer=projected.baseline_gamer,
                baseline_ever_payer=projected.baseline_ever_payer,
                global_mass=projected.global_mass,
                analysis_weight=adapter_cell.analysis_weight,
            )
        )
    return tuple(cells)


def _resolve_metric_contract(
    planned: PlannedPopulationEstimand,
    semantics: PopulationOutcomeMetricSemantics,
) -> tuple[OutputMetricContract, str]:
    from ..outputs.metric_contracts import (
        OUTPUT_METRIC_CONTRACTS,
        OutputMetricContract,
    )

    by_id = {
        contract.contract_id: contract
        for contract in OUTPUT_METRIC_CONTRACTS.values()
    }
    contract = by_id.get(planned.metric_contract_id)
    if contract is None:
        raise AnalysisBindingValidationError(
            "planned estimand references an unknown output metric contract: "
            f"{planned.metric_contract_id}"
        )
    OutputMetricContract.__post_init__(contract)
    expected_column = _EXPECTED_PLAYER_CONTRACT_COLUMN.get(
        planned.outcome_metric
    )
    if expected_column is None:
        raise AnalysisBindingValidationError(
            "no player-level output metric contract currently supports "
            f"{planned.outcome_metric.value}"
        )
    expected_id = f"player_outcomes.csv:{expected_column}"
    if contract.contract_id != expected_id:
        raise AnalysisBindingValidationError(
            "planned metric contract does not match the selected outcome: "
            f"expected {expected_id}, observed {contract.contract_id}"
        )
    expected_storage = {
        "bool": "boolean",
        "float64": "float",
        "int64": "integer",
    }[semantics.storage_dtype]
    if contract.storage_type != expected_storage:
        raise AnalysisBindingValidationError(
            "planned metric contract storage differs from outcome semantics"
        )
    return contract, _canonical_sha256(contract.snapshot())


def _outcome_array(
    result: PolicyScenarioResult,
    semantics: PopulationOutcomeMetricSemantics,
) -> npt.NDArray[np.generic]:
    route = semantics.result_path.removeprefix("PolicyScenarioResult.")
    if route == "spending_cents":
        values = result.spending_cents
    elif route == "composite_harm":
        values = result.composite_harm
    elif route == "enjoyment":
        values = result.enjoyment
    elif route == "high_risk":
        values = result.high_risk
    elif route == "harm.component_scores":
        if semantics.component_index is None:
            raise AnalysisBindingValidationError(
                "component-score outcome lacks a component index"
            )
        values = result.harm.component_scores[:, semantics.component_index]
    elif route.startswith("harm.") and semantics.component_index is None:
        attribute = route.removeprefix("harm.")
        allowed = {
            "harmful_spending_cents",
            "unplanned_spending_cents",
            "monetary_harm_proxy_cents",
            "opportunity_cost_proxy_cents",
            "adult_opportunity_cost_proxy_cents",
            "youth_opportunity_cost_proxy_cents",
            "total_monetary_proxy_cents",
            "excess_play_minutes",
            "displaced_sleep_minutes",
            "displaced_work_study_minutes",
            "displaced_social_minutes",
            "displaced_physical_activity_minutes",
        }
        if attribute not in allowed:
            raise AnalysisBindingValidationError(
                "outcome semantics contain an unsupported harm route"
            )
        values = getattr(result.harm, attribute)
    else:
        raise AnalysisBindingValidationError(
            "outcome semantics contain an unsupported result route"
        )
    expected_dtype = np.dtype(semantics.storage_dtype)
    if (
        type(values) is not np.ndarray
        or values.ndim != 1
        or values.size != result.player_ids.size
        or values.dtype != expected_dtype
    ):
        raise AnalysisBindingValidationError(
            "selected outcome array differs from its whitelisted shape/dtype"
        )
    if np.issubdtype(expected_dtype, np.floating) and not np.all(
        np.isfinite(values)
    ):
        raise AnalysisBindingValidationError(
            "selected outcome contains non-finite values"
        )
    return values


def _primitive_values(
    values: npt.NDArray[np.generic],
    semantics: PopulationOutcomeMetricSemantics,
) -> npt.NDArray[np.generic]:
    if semantics.storage_dtype == "bool":
        return values.astype(np.int64)
    return values


def _required_result(
    records: Mapping[tuple[int, ScenarioId], object],
    *,
    seed: int,
    scenario_id: ScenarioId,
) -> PolicyScenarioResult:
    record = records.get((seed, scenario_id))
    if record is None:
        raise AnalysisBindingValidationError(
            "batch omitted a plan-required scenario result: "
            f"seed={seed}, scenario={scenario_id.value}"
        )
    result = getattr(record, "result", None)
    if type(result) is not PolicyScenarioResult:
        raise AnalysisBindingValidationError(
            "batch contains a non-exact policy scenario result"
        )
    if result.seed != seed or result.scenario.scenario_id is not scenario_id:
        raise AnalysisBindingValidationError(
            "batch result key differs from the planned scenario direction"
        )
    return result


def _reattest_batch_spec(spec: PolicyBatchSpec) -> None:
    observed = PolicyBatchSpec(
        seeds=spec.seeds,
        days=spec.days,
        player_count=spec.player_count,
        scenarios=spec.scenarios,
        reference_scenario=spec.reference_scenario,
        decision_parameters=spec.decision_parameters,
    )
    if observed != spec:
        raise AnalysisBindingValidationError(
            "batch spec differs from its normalized exact reconstruction"
        )


def _reject_unresolved_money_estimands(
    plan: ProspectiveAnalysisPlan,
) -> None:
    unresolved = tuple(
        estimand.estimand_id
        for estimand in plan.estimands
        if estimand.outcome_semantics.metric_kind
        is PopulationMetricKind.MONEY_MINOR_UNITS
    )
    if unresolved:
        raise AnalysisBindingValidationError(
            "analysis binding schema v1 prohibits money outcomes until an "
            "executed currency/price-period conversion is retained; unresolved "
            "estimands="
            + ",".join(unresolved)
        )


def _validate_estimand_period_durations(
    plan: ProspectiveAnalysisPlan,
    batch_spec: PolicyBatchSpec,
) -> None:
    """Bind declared inclusive durations, but not calendar anchors, to execution."""

    expected_days = max(1, batch_spec.days)
    for estimand in plan.estimands:
        period = estimand.period
        observed_days = (period.period_end - period.period_start).days + 1
        if observed_days != expected_days:
            execution = (
                "zero-day structural snapshot"
                if batch_spec.days == 0
                else f"{batch_spec.days}-day batch"
            )
            raise AnalysisBindingValidationError(
                f"estimand {estimand.estimand_id!r} inclusive period duration "
                f"is {observed_days} day(s), but the executed {execution} "
                f"requires {expected_days} declared day(s)"
            )


def _resolved_estimand_id(
    planned: PlannedPopulationEstimand,
    seed: int,
) -> str:
    return f"planned.{planned.estimand_sha256}.seed.{seed}"


def _target_population_id(planned: PlannedPopulationEstimand) -> str:
    return f"population.{planned.inclusion_predicate.predicate_sha256}"


def _eligibility_sha256(
    *,
    seed: int,
    predicate_sha256: str,
    ordered_player_ids_sha256: str,
    mask: npt.NDArray[np.bool_],
) -> str:
    _sha256_digest(predicate_sha256, name="predicate_sha256")
    _sha256_digest(
        ordered_player_ids_sha256,
        name="ordered_player_ids_sha256",
    )
    if (
        type(mask) is not np.ndarray
        or mask.ndim != 1
        or mask.dtype != np.dtype(np.bool_)
    ):
        raise TypeError("eligibility mask must be a one-dimensional bool array")
    mask_sha256 = sha256(mask.tobytes(order="C")).hexdigest()
    return _canonical_sha256(
        {
            "schema_version": ANALYSIS_BINDING_SCHEMA_VERSION,
            "seed_decimal": str(seed),
            "predicate_sha256": predicate_sha256,
            "ordered_player_ids_sha256": ordered_player_ids_sha256,
            "player_count_decimal": str(mask.size),
            "mask_bytes_sha256": mask_sha256,
            "selected_count_decimal": str(int(np.count_nonzero(mask))),
        }
    )


def _selected_outcome_sha256(
    semantics: PopulationOutcomeMetricSemantics,
    player_ids: tuple[int, ...],
    values: npt.NDArray[np.generic],
) -> str:
    expected_dtype = np.dtype(semantics.storage_dtype)
    if (
        type(values) is not np.ndarray
        or values.ndim != 1
        or values.dtype != expected_dtype
        or values.size != len(player_ids)
    ):
        raise TypeError("selected outcomes differ from their exact semantics")
    little_endian = expected_dtype.newbyteorder("<")
    value_bytes_sha256 = sha256(
        np.asarray(values, dtype=little_endian).tobytes(order="C")
    ).hexdigest()
    return _canonical_sha256(
        {
            "schema_version": ANALYSIS_BINDING_SCHEMA_VERSION,
            "outcome": semantics.snapshot(),
            "player_ids_decimal": [str(item) for item in player_ids],
            "value_bytes_sha256": value_bytes_sha256,
        }
    )


def _seed_attestation_payload(
    *,
    seed: int,
    planned: PlannedPopulationEstimand,
    population_seed_record_sha256: str,
    eligibility_sha256: str,
    selected_weights: ExactPopulationWeights,
    reference_outcome_sha256: str,
    comparison_outcome_sha256: str,
    metric_contract_sha256: str,
    spec: PopulationEstimandSpec,
    result: PopulationEstimandResult,
) -> dict[str, object]:
    return {
        "schema_version": ANALYSIS_BINDING_SCHEMA_VERSION,
        "seed": seed,
        "seed_decimal": str(seed),
        "planned_estimand": planned.snapshot(),
        "planned_estimand_sha256": planned.estimand_sha256,
        "reference_scenario_id": planned.reference_scenario_id.value,
        "comparison_scenario_id": planned.comparison_scenario_id.value,
        "contrast_direction": planned.contrast_direction,
        "population_seed_record_sha256": population_seed_record_sha256,
        "eligibility_sha256": eligibility_sha256,
        "selected_player_count": len(selected_weights.player_ids),
        "selected_player_count_decimal": str(len(selected_weights.player_ids)),
        "selected_design_weights_sha256": selected_weights.design_sha256,
        "reference_outcome_sha256": reference_outcome_sha256,
        "comparison_outcome_sha256": comparison_outcome_sha256,
        "metric_contract_id": planned.metric_contract_id,
        "metric_contract_sha256": metric_contract_sha256,
        "spec": spec.snapshot(),
        "result": result.snapshot(),
        "preregistered": False,
        "campaign_ready": False,
        "campaign_blockers": list(_CAMPAIGN_BLOCKERS),
    }


def _run_attestation_payload(
    *,
    plan: ProspectiveAnalysisPlan,
    causal_design_sha256: str,
    batch_spec_sha256: str,
    model_inputs_sha256: str,
    population_input_sha256: str,
    profile_input_sha256: str,
    population_lineage_sha256: str,
    metric_contract_registry_sha256: str,
    harm_weights_sha256: str,
    seeds: tuple[int, ...],
    seed_bindings: tuple[SeedAnalysisBinding, ...],
) -> dict[str, object]:
    return {
        "schema_version": ANALYSIS_BINDING_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "causal_design_sha256": causal_design_sha256,
        "batch_spec_sha256": batch_spec_sha256,
        "model_inputs_sha256": model_inputs_sha256,
        "population_input_sha256": population_input_sha256,
        "profile_input_sha256": profile_input_sha256,
        "population_lineage_sha256": population_lineage_sha256,
        "metric_contract_registry_sha256": metric_contract_registry_sha256,
        "harm_weights_sha256": harm_weights_sha256,
        "output_profile_id": TARGET_POPULATION_OUTPUT_PROFILE,
        "output_profile_schema_sha256": (
            _target_output_profile_sha256()
        ),
        "seeds": list(seeds),
        "seed_decimal_strings": [str(seed) for seed in seeds],
        "seed_bindings": [item.snapshot() for item in seed_bindings],
        "preregistered": False,
        "campaign_ready": False,
        "campaign_blockers": list(_CAMPAIGN_BLOCKERS),
    }


def _sha256_digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AnalysisBindingValidationError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _metric_contract_registry_sha256() -> str:
    from ..outputs.metric_contracts import metric_contract_registry_sha256

    return metric_contract_registry_sha256()


def _target_output_profile_sha256() -> str:
    from ..outputs.schema import TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256

    return TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256


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
    "ANALYSIS_BINDING_SCHEMA_VERSION",
    "AnalysisBindingValidationError",
    "RunAnalysisBinding",
    "SeedAnalysisBinding",
    "resolve_run_analysis_binding",
    "validate_analysis_plan_inputs",
]
