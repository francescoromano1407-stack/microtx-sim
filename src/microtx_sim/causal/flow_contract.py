"""Fail-closed composition contract for the welfare-policy execution flow.

The repository contains two intentionally separate simulation layers.  The
prospective policy estimands are defined over :class:`PolicyScenarioResult`, so
they cannot be satisfied by the strategic-market ``World`` output or by an
unattested mixture of both layers.  This module makes that boundary explicit
and content-addresses the identities that cross the policy execution flow.

This is a composition and reproducibility contract only.  A passing flow
attestation does not promote provenance, calibration, registration, population,
monetary, uncertainty, convergence, or campaign-readiness status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Final

import numpy as np

from ..data.lineage import profile_lineage_fingerprint_matches
from ..data.monetary_execution import ConvertedMonetaryOutcome, MonetaryOutputBasis
from ..data.population_execution import (
    PopulationExecutionLineage,
    PopulationSeedExecutionRecord,
)
from ..metrics.population_estimands import PopulationMetricKind
from ..rng import validate_seed
from .analysis_binding import (
    RunAnalysisBinding,
    SeedAnalysisBinding,
    resolve_run_analysis_binding,
)
from .analysis_plan import (
    LoadedProspectiveAnalysisPlan,
    ProspectiveAnalysisPlan,
    verify_loaded_prospective_analysis_plan,
)
from .batch import PolicyBatchResult
from .scenarios import ScenarioId


POLICY_FLOW_CONTRACT_SCHEMA_VERSION: Final[str] = "1.0"
POLICY_FLOW_VERIFICATION_SCHEMA_VERSION: Final[str] = "1.0"
POLICY_FLOW_SEED_IDENTITY_SCHEMA_VERSION: Final[str] = "1.0"

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PolicyFlowContractError(ValueError):
    """Raised when execution layers or flow identities cannot be composed."""


class PolicyExecutionLayer(str, Enum):
    """Simulation layers that a future campaign declaration could request."""

    POLICY_WELFARE = "POLICY_WELFARE_V1"
    STRATEGIC_MARKET = "STRATEGIC_MARKET_V1"
    BOTH = "POLICY_WELFARE_AND_STRATEGIC_MARKET_V1"


class PolicyAggregationBasis(str, Enum):
    """Admissible value state at the population-aggregation boundary."""

    DIRECT_NONMONETARY = "DIRECT_NONMONETARY"
    EXACT_TARGET_CURRENCY_BEFORE_WEIGHTING = (
        "EXACT_TARGET_CURRENCY_RATIONAL_BEFORE_POPULATION_WEIGHTING"
    )
    RAW_JURISDICTION_CURRENCY = "RAW_JURISDICTION_CURRENCY_VALUES"


def validate_policy_aggregation_basis(
    metric_kind: PopulationMetricKind,
    basis: PolicyAggregationBasis,
) -> PolicyAggregationBasis:
    """Reject raw cross-currency pooling and mismatched aggregation states."""

    if type(metric_kind) is not PopulationMetricKind:
        raise TypeError("metric_kind must be PopulationMetricKind")
    if type(basis) is not PolicyAggregationBasis:
        raise TypeError("basis must be PolicyAggregationBasis")
    if basis is PolicyAggregationBasis.RAW_JURISDICTION_CURRENCY:
        raise PolicyFlowContractError(
            "raw jurisdiction-currency values cannot cross the population "
            "aggregation boundary; exact target-currency conversion must occur "
            "per observation before weighting"
        )
    if metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS:
        expected = PolicyAggregationBasis.EXACT_TARGET_CURRENCY_BEFORE_WEIGHTING
    else:
        expected = PolicyAggregationBasis.DIRECT_NONMONETARY
    if basis is not expected:
        raise PolicyFlowContractError(
            f"{metric_kind.value} requires aggregation basis {expected.value}"
        )
    return basis


@dataclass(frozen=True, slots=True)
class PolicyFlowContract:
    """Canonical declaration that one prospective plan uses policy outputs only."""

    schema_version: str
    contract_id: str
    execution_layer: PolicyExecutionLayer
    plan_id: str
    plan_sha256: str
    primary_estimand_id: str
    primary_result_path: str
    reference_scenario_id: ScenarioId
    comparison_scenario_id: ScenarioId
    primary_metric_contract_id: str
    expected_causal_design_sha256: str
    expected_batch_spec_sha256: str
    expected_model_inputs_sha256: str
    expected_population_input_sha256: str
    expected_profile_input_sha256: str
    expected_metric_contract_sha256: str
    expected_harm_weights_sha256: str
    expected_output_profile_sha256: str
    strategic_market_outputs_combined: bool
    scientific_readiness_claimed: bool
    contract_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_FLOW_CONTRACT_SCHEMA_VERSION:
            raise PolicyFlowContractError(
                "unsupported policy flow-contract schema version"
            )
        _identifier(self.contract_id, name="contract_id")
        _identifier(self.plan_id, name="plan_id")
        _identifier(self.primary_estimand_id, name="primary_estimand_id")
        _identifier(
            self.primary_metric_contract_id,
            name="primary_metric_contract_id",
            contract=True,
        )
        _require_policy_welfare_layer(self.execution_layer)
        if (
            type(self.primary_result_path) is not str
            or not self.primary_result_path.startswith("PolicyScenarioResult.")
        ):
            raise PolicyFlowContractError(
                "policy-only flow requires a PolicyScenarioResult.* primary outcome"
            )
        if type(self.reference_scenario_id) is not ScenarioId or type(
            self.comparison_scenario_id
        ) is not ScenarioId:
            raise TypeError("flow scenario identities must be ScenarioId values")
        if self.reference_scenario_id is self.comparison_scenario_id:
            raise PolicyFlowContractError(
                "the declared primary estimand must use distinct scenario identities"
            )
        for name in (
            "plan_sha256",
            "expected_causal_design_sha256",
            "expected_batch_spec_sha256",
            "expected_model_inputs_sha256",
            "expected_population_input_sha256",
            "expected_profile_input_sha256",
            "expected_metric_contract_sha256",
            "expected_harm_weights_sha256",
            "expected_output_profile_sha256",
            "contract_sha256",
        ):
            _sha256_digest(getattr(self, name), name=name)
        if self.strategic_market_outputs_combined is not False:
            raise PolicyFlowContractError(
                "policy-only flow cannot claim combined strategic-market outputs"
            )
        if self.scientific_readiness_claimed is not False:
            raise PolicyFlowContractError(
                "flow composition cannot claim scientific or campaign readiness"
            )
        if self.contract_sha256 != _canonical_sha256(self.attestation_payload()):
            raise PolicyFlowContractError(
                "policy flow-contract SHA-256 differs from its canonical payload"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "execution_layer": self.execution_layer.value,
            "plan_id": self.plan_id,
            "plan_sha256": self.plan_sha256,
            "primary_estimand_id": self.primary_estimand_id,
            "primary_result_path": self.primary_result_path,
            "reference_scenario_id": self.reference_scenario_id.value,
            "comparison_scenario_id": self.comparison_scenario_id.value,
            "primary_metric_contract_id": self.primary_metric_contract_id,
            "expected_causal_design_sha256": (
                self.expected_causal_design_sha256
            ),
            "expected_batch_spec_sha256": self.expected_batch_spec_sha256,
            "expected_model_inputs_sha256": self.expected_model_inputs_sha256,
            "expected_population_input_sha256": (
                self.expected_population_input_sha256
            ),
            "expected_profile_input_sha256": self.expected_profile_input_sha256,
            "expected_metric_contract_sha256": (
                self.expected_metric_contract_sha256
            ),
            "expected_harm_weights_sha256": self.expected_harm_weights_sha256,
            "expected_output_profile_sha256": (
                self.expected_output_profile_sha256
            ),
            "strategic_market_outputs_combined": (
                self.strategic_market_outputs_combined
            ),
            "scientific_readiness_claimed": self.scientific_readiness_claimed,
        }

    def snapshot(self) -> dict[str, object]:
        return {**self.attestation_payload(), "contract_sha256": self.contract_sha256}


@dataclass(frozen=True, slots=True)
class PolicyFlowSeedIdentity:
    """Cross-layer identities retained for one primary seed contrast."""

    schema_version: str
    seed: int
    primary_estimand_id: str
    reference_scenario_id: ScenarioId
    comparison_scenario_id: ScenarioId
    cohort_digest: str
    population_seed_record_sha256: str
    runtime_projection_sha256: str
    assignment_sha256: str
    population_balance_sha256: str
    ordered_player_ids_sha256: str
    selected_design_weights_sha256: str
    selected_player_count: int
    jurisdiction_codes: tuple[str, ...]
    jurisdiction_assignment_sha256: str
    primary_metric_contract_id: str
    source_metric_contract_sha256: str
    effective_metric_contract_sha256: str
    metric_kind: PopulationMetricKind
    aggregation_basis: PolicyAggregationBasis
    monetary_basis_sha256: str | None
    reference_outcome_sha256: str
    comparison_outcome_sha256: str
    estimand_result_sha256: str
    seed_binding_sha256: str
    seed_identity_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_FLOW_SEED_IDENTITY_SCHEMA_VERSION:
            raise PolicyFlowContractError(
                "unsupported policy flow seed-identity schema version"
            )
        validate_seed(self.seed, name="policy flow seed")
        _identifier(self.primary_estimand_id, name="primary_estimand_id")
        _identifier(
            self.primary_metric_contract_id,
            name="primary_metric_contract_id",
            contract=True,
        )
        if type(self.reference_scenario_id) is not ScenarioId or type(
            self.comparison_scenario_id
        ) is not ScenarioId:
            raise TypeError("flow scenario identities must be ScenarioId values")
        for name in (
            "cohort_digest",
            "population_seed_record_sha256",
            "runtime_projection_sha256",
            "assignment_sha256",
            "population_balance_sha256",
            "ordered_player_ids_sha256",
            "selected_design_weights_sha256",
            "jurisdiction_assignment_sha256",
            "source_metric_contract_sha256",
            "effective_metric_contract_sha256",
            "reference_outcome_sha256",
            "comparison_outcome_sha256",
            "estimand_result_sha256",
            "seed_binding_sha256",
            "seed_identity_sha256",
        ):
            _sha256_digest(getattr(self, name), name=name)
        if (
            type(self.selected_player_count) is not int
            or self.selected_player_count <= 0
        ):
            raise PolicyFlowContractError(
                "selected_player_count must be a positive exact integer"
            )
        if type(self.jurisdiction_codes) is not tuple or not self.jurisdiction_codes:
            raise TypeError("jurisdiction_codes must be a non-empty exact tuple")
        if any(type(code) is not str or not code for code in self.jurisdiction_codes):
            raise TypeError("jurisdiction_codes must contain non-empty strings")
        if len(set(self.jurisdiction_codes)) != len(self.jurisdiction_codes):
            raise PolicyFlowContractError("jurisdiction_codes must be unique")
        validate_policy_aggregation_basis(self.metric_kind, self.aggregation_basis)
        if self.metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS:
            if self.monetary_basis_sha256 is None:
                raise PolicyFlowContractError(
                    "money flow identity requires an exact monetary basis digest"
                )
            _sha256_digest(
                self.monetary_basis_sha256,
                name="monetary_basis_sha256",
            )
        elif self.monetary_basis_sha256 is not None:
            raise PolicyFlowContractError(
                "non-money flow identity cannot retain a monetary basis"
            )
        if self.seed_identity_sha256 != _canonical_sha256(
            self.attestation_payload()
        ):
            raise PolicyFlowContractError(
                "policy flow seed identity SHA-256 differs from its payload"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "seed_decimal": str(self.seed),
            "primary_estimand_id": self.primary_estimand_id,
            "reference_scenario_id": self.reference_scenario_id.value,
            "comparison_scenario_id": self.comparison_scenario_id.value,
            "cohort_digest": self.cohort_digest,
            "population_seed_record_sha256": self.population_seed_record_sha256,
            "runtime_projection_sha256": self.runtime_projection_sha256,
            "assignment_sha256": self.assignment_sha256,
            "population_balance_sha256": self.population_balance_sha256,
            "ordered_player_ids_sha256": self.ordered_player_ids_sha256,
            "selected_design_weights_sha256": (
                self.selected_design_weights_sha256
            ),
            "selected_player_count": self.selected_player_count,
            "selected_player_count_decimal": str(self.selected_player_count),
            "jurisdiction_codes": list(self.jurisdiction_codes),
            "jurisdiction_assignment_sha256": (
                self.jurisdiction_assignment_sha256
            ),
            "primary_metric_contract_id": self.primary_metric_contract_id,
            "source_metric_contract_sha256": self.source_metric_contract_sha256,
            "effective_metric_contract_sha256": (
                self.effective_metric_contract_sha256
            ),
            "metric_kind": self.metric_kind.value,
            "aggregation_basis": self.aggregation_basis.value,
            "monetary_basis_sha256": self.monetary_basis_sha256,
            "reference_outcome_sha256": self.reference_outcome_sha256,
            "comparison_outcome_sha256": self.comparison_outcome_sha256,
            "estimand_result_sha256": self.estimand_result_sha256,
            "seed_binding_sha256": self.seed_binding_sha256,
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "seed_identity_sha256": self.seed_identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class PolicyFlowVerification:
    """Content-addressed proof that one completed batch satisfies the contract."""

    schema_version: str
    contract: PolicyFlowContract
    analysis_plan_file_sha256: str
    analysis_binding_sha256: str
    population_lineage_sha256: str
    seed_identities: tuple[PolicyFlowSeedIdentity, ...]
    campaign_blockers: tuple[str, ...]
    verification_sha256: str
    campaign_ready: bool = field(default=False, init=False)
    scientific_readiness_claimed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_FLOW_VERIFICATION_SCHEMA_VERSION:
            raise PolicyFlowContractError(
                "unsupported policy flow-verification schema version"
            )
        if type(self.contract) is not PolicyFlowContract:
            raise TypeError("contract must be PolicyFlowContract")
        PolicyFlowContract.__post_init__(self.contract)
        for name in (
            "analysis_plan_file_sha256",
            "analysis_binding_sha256",
            "population_lineage_sha256",
            "verification_sha256",
        ):
            _sha256_digest(getattr(self, name), name=name)
        if type(self.seed_identities) is not tuple or not self.seed_identities:
            raise TypeError("seed_identities must be a non-empty exact tuple")
        for item in self.seed_identities:
            if type(item) is not PolicyFlowSeedIdentity:
                raise TypeError(
                    "seed_identities must contain PolicyFlowSeedIdentity values"
                )
            PolicyFlowSeedIdentity.__post_init__(item)
        seeds = tuple(item.seed for item in self.seed_identities)
        if seeds != tuple(sorted(seeds)) or len(set(seeds)) != len(seeds):
            raise PolicyFlowContractError(
                "flow seed identities must be unique and canonically ordered"
            )
        if type(self.campaign_blockers) is not tuple or any(
            type(item) is not str or not item for item in self.campaign_blockers
        ):
            raise TypeError("campaign_blockers must be an exact tuple of text")
        if self.campaign_ready or self.scientific_readiness_claimed:
            raise PolicyFlowContractError(
                "flow verification cannot promote scientific campaign readiness"
            )
        if self.verification_sha256 != _canonical_sha256(
            self.attestation_payload()
        ):
            raise PolicyFlowContractError(
                "policy flow verification SHA-256 differs from its payload"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract": self.contract.snapshot(),
            "analysis_plan_file_sha256": self.analysis_plan_file_sha256,
            "analysis_binding_sha256": self.analysis_binding_sha256,
            "population_lineage_sha256": self.population_lineage_sha256,
            "seed_identities": [item.snapshot() for item in self.seed_identities],
            "campaign_ready": self.campaign_ready,
            "scientific_readiness_claimed": self.scientific_readiness_claimed,
            "campaign_blockers": list(self.campaign_blockers),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "verification_sha256": self.verification_sha256,
        }


def build_policy_flow_contract(
    plan: ProspectiveAnalysisPlan,
    *,
    execution_layer: PolicyExecutionLayer = PolicyExecutionLayer.POLICY_WELFARE,
    contract_id: str | None = None,
) -> PolicyFlowContract:
    """Bind a prospective ``PolicyScenarioResult`` estimand to one layer."""

    if type(plan) is not ProspectiveAnalysisPlan:
        raise TypeError("plan must be ProspectiveAnalysisPlan")
    ProspectiveAnalysisPlan.__post_init__(plan)
    _require_policy_welfare_layer(execution_layer)
    primary = plan.primary_estimand
    selected_contract_id = contract_id or f"{plan.plan_id}.policy-flow.v1"
    values = {
        "schema_version": POLICY_FLOW_CONTRACT_SCHEMA_VERSION,
        "contract_id": selected_contract_id,
        "execution_layer": execution_layer,
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "primary_estimand_id": primary.estimand_id,
        "primary_result_path": primary.outcome_semantics.result_path,
        "reference_scenario_id": primary.reference_scenario_id,
        "comparison_scenario_id": primary.comparison_scenario_id,
        "primary_metric_contract_id": primary.metric_contract_id,
        "expected_causal_design_sha256": plan.expected_causal_design_sha256,
        "expected_batch_spec_sha256": plan.expected_batch_spec_sha256,
        "expected_model_inputs_sha256": plan.expected_model_inputs_sha256,
        "expected_population_input_sha256": plan.expected_population_input_sha256,
        "expected_profile_input_sha256": plan.expected_profile_input_sha256,
        "expected_metric_contract_sha256": plan.expected_metric_contract_sha256,
        "expected_harm_weights_sha256": plan.expected_harm_weights_sha256,
        "expected_output_profile_sha256": plan.expected_output_profile_sha256,
        "strategic_market_outputs_combined": False,
        "scientific_readiness_claimed": False,
    }
    payload = _policy_flow_contract_payload(**values)
    return PolicyFlowContract(
        **values,
        contract_sha256=_canonical_sha256(payload),
    )


def attest_policy_only_flow(
    contract: PolicyFlowContract,
    *,
    analysis_plan: LoadedProspectiveAnalysisPlan,
    batch: PolicyBatchResult,
    analysis_binding: RunAnalysisBinding,
) -> PolicyFlowVerification:
    """Re-attest a completed policy flow without conferring readiness."""

    try:
        return _attest_policy_only_flow(
            contract,
            analysis_plan=analysis_plan,
            batch=batch,
            analysis_binding=analysis_binding,
        )
    except PolicyFlowContractError:
        raise
    except (TypeError, ValueError, RuntimeError, OSError, KeyError, IndexError) as exc:
        raise PolicyFlowContractError(
            f"policy-only flow attestation failed closed: {exc}"
        ) from exc


def _attest_policy_only_flow(
    contract: PolicyFlowContract,
    *,
    analysis_plan: LoadedProspectiveAnalysisPlan,
    batch: PolicyBatchResult,
    analysis_binding: RunAnalysisBinding,
) -> PolicyFlowVerification:
    if type(contract) is not PolicyFlowContract:
        raise TypeError("contract must be PolicyFlowContract")
    PolicyFlowContract.__post_init__(contract)
    if type(analysis_plan) is not LoadedProspectiveAnalysisPlan:
        raise TypeError("analysis_plan must be LoadedProspectiveAnalysisPlan")
    verified_plan = verify_loaded_prospective_analysis_plan(analysis_plan)
    plan = verified_plan.plan
    if contract.plan_id != plan.plan_id or contract.plan_sha256 != plan.plan_sha256:
        raise PolicyFlowContractError(
            "flow contract plan identity differs from the re-attested plan file"
        )
    expected_contract = build_policy_flow_contract(
        plan,
        execution_layer=contract.execution_layer,
        contract_id=contract.contract_id,
    )
    if contract != expected_contract:
        raise PolicyFlowContractError(
            "flow contract differs from the canonical declaration for its plan"
        )

    if type(batch) is not PolicyBatchResult:
        raise TypeError("batch must be PolicyBatchResult")
    PolicyBatchResult.__post_init__(batch)
    if type(analysis_binding) is not RunAnalysisBinding:
        raise TypeError("analysis_binding must be RunAnalysisBinding")
    RunAnalysisBinding.__post_init__(analysis_binding)
    observed_binding = resolve_run_analysis_binding(plan, batch)
    if analysis_binding.binding_sha256 != observed_binding.binding_sha256:
        raise PolicyFlowContractError(
            "supplied analysis binding differs from the re-attested plan and batch"
        )
    if analysis_binding.plan.plan_sha256 != contract.plan_sha256:
        raise PolicyFlowContractError(
            "analysis binding plan differs from the flow contract"
        )

    expected_hashes = {
        "expected_causal_design_sha256": analysis_binding.causal_design_sha256,
        "expected_batch_spec_sha256": analysis_binding.batch_spec_sha256,
        "expected_model_inputs_sha256": analysis_binding.model_inputs_sha256,
        "expected_population_input_sha256": analysis_binding.population_input_sha256,
        "expected_profile_input_sha256": analysis_binding.profile_input_sha256,
        "expected_metric_contract_sha256": (
            analysis_binding.metric_contract_registry_sha256
        ),
        "expected_harm_weights_sha256": analysis_binding.harm_weights_sha256,
        "expected_output_profile_sha256": (
            analysis_binding.output_profile_schema_sha256
        ),
    }
    mismatches: list[str] = []
    for name, observed in expected_hashes.items():
        if name == "expected_profile_input_sha256":
            matches = profile_lineage_fingerprint_matches(
                getattr(contract, name),
                observed,
            )
        else:
            matches = getattr(contract, name) == observed
        if not matches:
            mismatches.append(name)
    if mismatches:
        raise PolicyFlowContractError(
            "flow contract input identities differ from execution: "
            + ", ".join(mismatches)
        )

    lineage = batch.population_execution_lineage
    if type(lineage) is not PopulationExecutionLineage:
        raise PolicyFlowContractError(
            "policy-only prospective flow requires projected-population lineage"
        )
    PopulationExecutionLineage.__post_init__(lineage)
    if lineage.lineage_sha256 != analysis_binding.population_lineage_sha256:
        raise PolicyFlowContractError(
            "population lineage differs from the analysis binding"
        )

    primary = plan.primary_estimand
    primary_bindings = {
        item.seed: item
        for item in analysis_binding.seed_bindings
        if item.planned_estimand == primary
    }
    if tuple(sorted(primary_bindings)) != plan.stopping_rule.seeds:
        raise PolicyFlowContractError(
            "primary seed bindings do not exactly cover the stopping rule"
        )
    record_by_key = {
        (record.result.seed, record.result.scenario.scenario_id): record
        for record in batch.records
    }
    seed_identities = tuple(
        _build_seed_identity(
            contract,
            seed=seed,
            seed_binding=primary_bindings[seed],
            population_record=lineage.record_for_seed(seed),
            batch=batch,
            record_by_key=record_by_key,
        )
        for seed in plan.stopping_rule.seeds
    )
    payload = _policy_flow_verification_payload(
        contract=contract,
        analysis_plan_file_sha256=verified_plan.file_sha256,
        analysis_binding_sha256=analysis_binding.binding_sha256,
        population_lineage_sha256=lineage.lineage_sha256,
        seed_identities=seed_identities,
        campaign_blockers=plan.campaign_blockers,
    )
    return PolicyFlowVerification(
        schema_version=POLICY_FLOW_VERIFICATION_SCHEMA_VERSION,
        contract=contract,
        analysis_plan_file_sha256=verified_plan.file_sha256,
        analysis_binding_sha256=analysis_binding.binding_sha256,
        population_lineage_sha256=lineage.lineage_sha256,
        seed_identities=seed_identities,
        campaign_blockers=plan.campaign_blockers,
        verification_sha256=_canonical_sha256(payload),
    )


def _build_seed_identity(
    contract: PolicyFlowContract,
    *,
    seed: int,
    seed_binding: SeedAnalysisBinding,
    population_record: PopulationSeedExecutionRecord,
    batch: PolicyBatchResult,
    record_by_key: dict[tuple[int, ScenarioId], object],
) -> PolicyFlowSeedIdentity:
    SeedAnalysisBinding.__post_init__(seed_binding)
    PopulationSeedExecutionRecord.__post_init__(population_record)
    if seed_binding.seed != seed or population_record.seed != seed:
        raise PolicyFlowContractError("seed identity changed across the policy flow")
    planned = seed_binding.planned_estimand
    if (
        planned.estimand_id != contract.primary_estimand_id
        or planned.reference_scenario_id is not contract.reference_scenario_id
        or planned.comparison_scenario_id is not contract.comparison_scenario_id
        or planned.metric_contract_id != contract.primary_metric_contract_id
    ):
        raise PolicyFlowContractError(
            "primary estimand, scenario, or metric identity changed across the flow"
        )
    try:
        reference_record = record_by_key[(seed, contract.reference_scenario_id)]
        comparison_record = record_by_key[(seed, contract.comparison_scenario_id)]
    except KeyError as exc:
        raise PolicyFlowContractError(
            "primary flow is missing a declared scenario branch"
        ) from exc
    reference = getattr(reference_record, "result", None)
    comparison = getattr(comparison_record, "result", None)
    if reference is None or comparison is None:
        raise PolicyFlowContractError(
            "primary flow contains an invalid scenario record"
        )
    if (
        reference.seed != seed
        or comparison.seed != seed
        or reference.scenario.scenario_id is not contract.reference_scenario_id
        or comparison.scenario.scenario_id is not contract.comparison_scenario_id
    ):
        raise PolicyFlowContractError(
            "scenario or seed identity differs from the batch record key"
        )
    seed_records = tuple(
        record for record in batch.records if record.result.seed == seed
    )
    cohort_digests = {record.cohort_digest for record in seed_records}
    if cohort_digests != {batch.cohort_digest_by_seed[seed]}:
        raise PolicyFlowContractError(
            "scenario branches do not retain one common pre-treatment cohort"
        )
    cohort_digest = batch.cohort_digest_by_seed[seed]
    if population_record.cohort_digest != cohort_digest:
        raise PolicyFlowContractError(
            "population execution and scenario branches use different cohorts"
        )

    full_ids = np.asarray(population_record.exact_weights.player_ids, dtype=np.int64)
    if not np.array_equal(reference.player_ids, full_ids) or not np.array_equal(
        comparison.player_ids,
        full_ids,
    ):
        raise PolicyFlowContractError(
            "paired scenarios differ from the projected population player order"
        )
    position_by_id = {int(player_id): index for index, player_id in enumerate(full_ids)}
    try:
        positions = tuple(
            position_by_id[player_id]
            for player_id in seed_binding.selected_weights.player_ids
        )
    except KeyError as exc:
        raise PolicyFlowContractError(
            "selected population weights contain an unknown player ID"
        ) from exc
    if positions != tuple(sorted(positions)):
        raise PolicyFlowContractError(
            "selected population weights do not preserve projected player order"
        )
    expected_fractions = tuple(
        population_record.exact_weights.fractions[position]
        for position in positions
    )
    if expected_fractions != seed_binding.selected_weights.fractions:
        raise PolicyFlowContractError(
            "selected population weights differ from the projection assignment"
        )
    reference_jurisdiction = tuple(
        int(reference.jurisdiction[position]) for position in positions
    )
    comparison_jurisdiction = tuple(
        int(comparison.jurisdiction[position]) for position in positions
    )
    if reference_jurisdiction != comparison_jurisdiction:
        raise PolicyFlowContractError(
            "jurisdiction assignment differs across paired policy branches"
        )
    if any(
        value < 0 or value >= len(population_record.jurisdiction_codes)
        for value in reference_jurisdiction
    ):
        raise PolicyFlowContractError(
            "paired policy jurisdiction assignment falls outside lineage codes"
        )
    jurisdiction_assignment_sha256 = _canonical_sha256(
        {
            "jurisdiction_codes": list(population_record.jurisdiction_codes),
            "player_ids_decimal": [
                str(player_id) for player_id in seed_binding.selected_weights.player_ids
            ],
            "jurisdiction_indices_decimal": [
                str(value) for value in reference_jurisdiction
            ],
        }
    )

    metric_kind = planned.outcome_semantics.metric_kind
    if metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS:
        basis = PolicyAggregationBasis.EXACT_TARGET_CURRENCY_BEFORE_WEIGHTING
        monetary_basis = seed_binding.monetary_output_basis
        reference_execution = seed_binding.reference_monetary_execution
        comparison_execution = seed_binding.comparison_monetary_execution
        if (
            type(monetary_basis) is not MonetaryOutputBasis
            or type(reference_execution) is not ConvertedMonetaryOutcome
            or type(comparison_execution) is not ConvertedMonetaryOutcome
        ):
            raise PolicyFlowContractError(
                "money flow lacks exact per-observation conversion executions"
            )
        if (
            reference_execution.jurisdiction_indices != reference_jurisdiction
            or comparison_execution.jurisdiction_indices != comparison_jurisdiction
        ):
            raise PolicyFlowContractError(
                "monetary conversion jurisdiction identity differs from the cohort"
            )
        monetary_basis_sha256: str | None = monetary_basis.basis_sha256
    else:
        basis = PolicyAggregationBasis.DIRECT_NONMONETARY
        if (
            seed_binding.monetary_output_basis is not None
            or seed_binding.reference_monetary_execution is not None
            or seed_binding.comparison_monetary_execution is not None
        ):
            raise PolicyFlowContractError(
                "non-money flow unexpectedly contains monetary conversion state"
            )
        monetary_basis_sha256 = None
    validate_policy_aggregation_basis(metric_kind, basis)

    values = {
        "schema_version": POLICY_FLOW_SEED_IDENTITY_SCHEMA_VERSION,
        "seed": seed,
        "primary_estimand_id": contract.primary_estimand_id,
        "reference_scenario_id": contract.reference_scenario_id,
        "comparison_scenario_id": contract.comparison_scenario_id,
        "cohort_digest": cohort_digest,
        "population_seed_record_sha256": population_record.seed_record_sha256,
        "runtime_projection_sha256": population_record.runtime_projection_sha256,
        "assignment_sha256": population_record.assignment_sha256,
        "population_balance_sha256": population_record.balance.balance_sha256,
        "ordered_player_ids_sha256": population_record.ordered_player_ids_sha256,
        "selected_design_weights_sha256": (
            seed_binding.selected_weights.design_sha256
        ),
        "selected_player_count": len(seed_binding.selected_weights.player_ids),
        "jurisdiction_codes": population_record.jurisdiction_codes,
        "jurisdiction_assignment_sha256": jurisdiction_assignment_sha256,
        "primary_metric_contract_id": contract.primary_metric_contract_id,
        "source_metric_contract_sha256": (
            seed_binding.source_metric_contract_sha256
        ),
        "effective_metric_contract_sha256": seed_binding.metric_contract_sha256,
        "metric_kind": metric_kind,
        "aggregation_basis": basis,
        "monetary_basis_sha256": monetary_basis_sha256,
        "reference_outcome_sha256": seed_binding.reference_outcome_sha256,
        "comparison_outcome_sha256": seed_binding.comparison_outcome_sha256,
        "estimand_result_sha256": seed_binding.result.result_sha256,
        "seed_binding_sha256": seed_binding.binding_sha256,
    }
    payload = _policy_flow_seed_identity_payload(**values)
    return PolicyFlowSeedIdentity(
        **values,
        seed_identity_sha256=_canonical_sha256(payload),
    )


def _require_policy_welfare_layer(layer: PolicyExecutionLayer) -> None:
    if type(layer) is not PolicyExecutionLayer:
        raise TypeError("execution_layer must be PolicyExecutionLayer")
    if layer is not PolicyExecutionLayer.POLICY_WELFARE:
        raise PolicyFlowContractError(
            "the declared PolicyScenarioResult estimand supports only "
            "POLICY_WELFARE_V1; strategic-market or BOTH composition has no "
            "typed scientific adapter and is rejected"
        )


def _policy_flow_contract_payload(**values: object) -> dict[str, object]:
    return {
        "schema_version": values["schema_version"],
        "contract_id": values["contract_id"],
        "execution_layer": values["execution_layer"].value,
        "plan_id": values["plan_id"],
        "plan_sha256": values["plan_sha256"],
        "primary_estimand_id": values["primary_estimand_id"],
        "primary_result_path": values["primary_result_path"],
        "reference_scenario_id": values["reference_scenario_id"].value,
        "comparison_scenario_id": values["comparison_scenario_id"].value,
        "primary_metric_contract_id": values["primary_metric_contract_id"],
        "expected_causal_design_sha256": values["expected_causal_design_sha256"],
        "expected_batch_spec_sha256": values["expected_batch_spec_sha256"],
        "expected_model_inputs_sha256": values["expected_model_inputs_sha256"],
        "expected_population_input_sha256": values[
            "expected_population_input_sha256"
        ],
        "expected_profile_input_sha256": values["expected_profile_input_sha256"],
        "expected_metric_contract_sha256": values[
            "expected_metric_contract_sha256"
        ],
        "expected_harm_weights_sha256": values["expected_harm_weights_sha256"],
        "expected_output_profile_sha256": values[
            "expected_output_profile_sha256"
        ],
        "strategic_market_outputs_combined": values[
            "strategic_market_outputs_combined"
        ],
        "scientific_readiness_claimed": values["scientific_readiness_claimed"],
    }


def _policy_flow_seed_identity_payload(**values: object) -> dict[str, object]:
    return {
        "schema_version": values["schema_version"],
        "seed": values["seed"],
        "seed_decimal": str(values["seed"]),
        "primary_estimand_id": values["primary_estimand_id"],
        "reference_scenario_id": values["reference_scenario_id"].value,
        "comparison_scenario_id": values["comparison_scenario_id"].value,
        "cohort_digest": values["cohort_digest"],
        "population_seed_record_sha256": values[
            "population_seed_record_sha256"
        ],
        "runtime_projection_sha256": values["runtime_projection_sha256"],
        "assignment_sha256": values["assignment_sha256"],
        "population_balance_sha256": values["population_balance_sha256"],
        "ordered_player_ids_sha256": values["ordered_player_ids_sha256"],
        "selected_design_weights_sha256": values[
            "selected_design_weights_sha256"
        ],
        "selected_player_count": values["selected_player_count"],
        "selected_player_count_decimal": str(values["selected_player_count"]),
        "jurisdiction_codes": list(values["jurisdiction_codes"]),
        "jurisdiction_assignment_sha256": values[
            "jurisdiction_assignment_sha256"
        ],
        "primary_metric_contract_id": values["primary_metric_contract_id"],
        "source_metric_contract_sha256": values[
            "source_metric_contract_sha256"
        ],
        "effective_metric_contract_sha256": values[
            "effective_metric_contract_sha256"
        ],
        "metric_kind": values["metric_kind"].value,
        "aggregation_basis": values["aggregation_basis"].value,
        "monetary_basis_sha256": values["monetary_basis_sha256"],
        "reference_outcome_sha256": values["reference_outcome_sha256"],
        "comparison_outcome_sha256": values["comparison_outcome_sha256"],
        "estimand_result_sha256": values["estimand_result_sha256"],
        "seed_binding_sha256": values["seed_binding_sha256"],
    }


def _policy_flow_verification_payload(
    *,
    contract: PolicyFlowContract,
    analysis_plan_file_sha256: str,
    analysis_binding_sha256: str,
    population_lineage_sha256: str,
    seed_identities: tuple[PolicyFlowSeedIdentity, ...],
    campaign_blockers: tuple[str, ...],
) -> dict[str, object]:
    return {
        "schema_version": POLICY_FLOW_VERIFICATION_SCHEMA_VERSION,
        "contract": contract.snapshot(),
        "analysis_plan_file_sha256": analysis_plan_file_sha256,
        "analysis_binding_sha256": analysis_binding_sha256,
        "population_lineage_sha256": population_lineage_sha256,
        "seed_identities": [item.snapshot() for item in seed_identities],
        "campaign_ready": False,
        "scientific_readiness_claimed": False,
        "campaign_blockers": list(campaign_blockers),
    }


def _identifier(value: object, *, name: str, contract: bool = False) -> None:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be non-empty text")
    pattern = (
        re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
        if contract
        else _IDENTIFIER
    )
    if pattern.fullmatch(value) is None:
        raise PolicyFlowContractError(f"{name} is not a canonical identifier")


def _sha256_digest(value: object, *, name: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise PolicyFlowContractError(f"{name} must be a lowercase SHA-256 digest")


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
    "POLICY_FLOW_CONTRACT_SCHEMA_VERSION",
    "POLICY_FLOW_SEED_IDENTITY_SCHEMA_VERSION",
    "POLICY_FLOW_VERIFICATION_SCHEMA_VERSION",
    "PolicyAggregationBasis",
    "PolicyExecutionLayer",
    "PolicyFlowContract",
    "PolicyFlowContractError",
    "PolicyFlowSeedIdentity",
    "PolicyFlowVerification",
    "attest_policy_only_flow",
    "build_policy_flow_contract",
    "validate_policy_aggregation_basis",
]
