"""Fail-closed metadata for non-executing exploratory validation.

This contract is deliberately separate from production campaign receipts.  It
describes an illustrative computational design that has been resolved but not
executed, and gives future exporters one canonical payload to embed without
inventing monetary or empirical language.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import PurePosixPath
import re
from typing import Final

from ..policy_config import (
    EXPLORATORY_ARTIFACT_NAMESPACE,
    EXPLORATORY_ESTIMAND_INTERPRETATION,
    EXPLORATORY_EXECUTION_KIND,
    EXPLORATORY_INTERNAL_MONETARY_UNIT,
    EXPLORATORY_MONETARY_AMOUNT_SEMANTICS,
    EXPLORATORY_POPULATION_BASIS,
    EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE,
    EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE,
)


EXPLORATORY_VALIDATION_SCHEMA_VERSION: Final[str] = "1.0"
EXPLORATORY_VALIDATION_OUTPUT_PROFILE: Final[str] = (
    "exploratory_synthetic_validation"
)
EXPLORATORY_INTERPRETATION_WORDING: Final[str] = (
    "This is an exploratory computational simulation using an illustrative, "
    "non-empirical projected population. Results are conditional on model "
    "assumptions; they are not empirical estimates, are not "
    "population-representative claims, and are not causal evidence about "
    "real-world players. Monetary outputs are model-equivalent amounts, not "
    "observed real-world spending. No real-world generalisation is permitted."
)
EXPLORATORY_INTERNAL_UNIT_WORDING: Final[str] = (
    "simulation_cents are internal synthetic model units, not real currency, "
    "not cross-country monetary results, and not observed spending."
)
EXPLORATORY_LAUNCH_COMMAND: Final[tuple[str, ...]] = (
    "microtx-sim",
    "policy-batch",
    "configs/policy_exploratory_synthetic.toml",
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")


class ExploratoryValidationMetadataError(ValueError):
    """Raised when exploratory metadata could overstate execution or validity."""


def build_exploratory_validation_metadata(
    *,
    configuration_path: str,
    configuration_sha256: str,
    exploratory_plan_path: str,
    exploratory_plan_id: str,
    exploratory_plan_sha256: str,
    exploratory_plan_file_sha256: str,
    scientific_parent_plan_path: str,
    scientific_parent_plan_id: str,
    scientific_parent_plan_sha256: str,
    scientific_parent_plan_file_sha256: str,
    scientific_parent_registration_status: str,
    primary_estimand_id: str,
    population_adapter_id: str,
    population_adapter_sha256: str,
    population_execution_input_sha256: str,
    scenario_snapshots: Sequence[Mapping[str, object]],
    target_currency: str,
    target_minor_unit_name: str,
    quote_convention: str,
    scale_convention: str,
    rate_period_start: str,
    rate_period_end: str,
    target_price_period_start: str,
    target_price_period_end: str,
    missing_date_policy: str,
    identity_missing_date_policy: str,
    rounding_method: str,
    rounding_scope: str,
    point_rate_status: str,
    rate_uncertainty_status: str,
    source_bundle_signature_status: str,
    simulation_bridge_status: str,
    observed_real_world_spending: bool,
    raw_cross_currency_pooling: str,
    monetary_source_bundle_sha256: str,
    monetary_source_artifact_sha256: str,
    monetary_conversion_table_sha256: str,
    monetary_conversion_basis_sha256: str,
    monetary_rate_evidence_sha256: str,
    profile_input_lineage_sha256: str,
    uncertainty_design: Mapping[str, object],
    convergence_rule: Mapping[str, object],
    fixed_seed_count: int,
    parameter_design_path: str,
    parameter_design_file_sha256: str,
    production_configuration_sha256: str,
    production_plan_sha256: str,
) -> dict[str, object]:
    """Build the canonical validation-only metadata payload.

    No cohort, population assignment, scenario outcome, monetary estimate, or
    convergence result is accepted by this API.  That omission is intentional:
    the payload records identities and execution requirements, never results.
    """

    for name, value in (
        ("configuration_sha256", configuration_sha256),
        ("exploratory_plan_sha256", exploratory_plan_sha256),
        ("exploratory_plan_file_sha256", exploratory_plan_file_sha256),
        ("scientific_parent_plan_sha256", scientific_parent_plan_sha256),
        (
            "scientific_parent_plan_file_sha256",
            scientific_parent_plan_file_sha256,
        ),
        ("population_adapter_sha256", population_adapter_sha256),
        (
            "population_execution_input_sha256",
            population_execution_input_sha256,
        ),
        ("monetary_source_bundle_sha256", monetary_source_bundle_sha256),
        ("monetary_source_artifact_sha256", monetary_source_artifact_sha256),
        (
            "monetary_conversion_table_sha256",
            monetary_conversion_table_sha256,
        ),
        (
            "monetary_conversion_basis_sha256",
            monetary_conversion_basis_sha256,
        ),
        ("monetary_rate_evidence_sha256", monetary_rate_evidence_sha256),
        ("profile_input_lineage_sha256", profile_input_lineage_sha256),
        ("parameter_design_file_sha256", parameter_design_file_sha256),
        ("production_configuration_sha256", production_configuration_sha256),
        ("production_plan_sha256", production_plan_sha256),
    ):
        _digest(value, name=name)
    for name, value in (
        ("exploratory_plan_id", exploratory_plan_id),
        ("scientific_parent_plan_id", scientific_parent_plan_id),
        ("primary_estimand_id", primary_estimand_id),
        ("population_adapter_id", population_adapter_id),
    ):
        _identifier(value, name=name)
    configuration_path = _relative_path(
        configuration_path, name="configuration_path"
    )
    exploratory_plan_path = _relative_path(
        exploratory_plan_path, name="exploratory_plan_path"
    )
    scientific_parent_plan_path = _relative_path(
        scientific_parent_plan_path, name="scientific_parent_plan_path"
    )
    parameter_design_path = _relative_path(
        parameter_design_path, name="parameter_design_path"
    )
    _text(
        scientific_parent_registration_status,
        name="scientific_parent_registration_status",
    )
    for name, value in (
        ("target_currency", target_currency),
        ("target_minor_unit_name", target_minor_unit_name),
        ("quote_convention", quote_convention),
        ("scale_convention", scale_convention),
        ("rate_period_start", rate_period_start),
        ("rate_period_end", rate_period_end),
        ("target_price_period_start", target_price_period_start),
        ("target_price_period_end", target_price_period_end),
        ("missing_date_policy", missing_date_policy),
        ("identity_missing_date_policy", identity_missing_date_policy),
        ("rounding_method", rounding_method),
        ("rounding_scope", rounding_scope),
        ("point_rate_status", point_rate_status),
        ("rate_uncertainty_status", rate_uncertainty_status),
        ("source_bundle_signature_status", source_bundle_signature_status),
        ("simulation_bridge_status", simulation_bridge_status),
        ("raw_cross_currency_pooling", raw_cross_currency_pooling),
    ):
        _text(value, name=name)
    if type(observed_real_world_spending) is not bool:
        raise ExploratoryValidationMetadataError(
            "observed_real_world_spending must be boolean"
        )
    fixed_monetary_fields = {
        "point_rate_status": "OFFICIAL_POINT_OBSERVATION",
        "rate_uncertainty_status": "UNQUANTIFIED",
        "source_bundle_signature_status": "MISSING",
        "simulation_bridge_status": "ILLUSTRATIVE",
        "observed_real_world_spending": False,
        "raw_cross_currency_pooling": "REJECT",
    }
    observed_monetary_fields = {
        "point_rate_status": point_rate_status,
        "rate_uncertainty_status": rate_uncertainty_status,
        "source_bundle_signature_status": source_bundle_signature_status,
        "simulation_bridge_status": simulation_bridge_status,
        "observed_real_world_spending": observed_real_world_spending,
        "raw_cross_currency_pooling": raw_cross_currency_pooling,
    }
    if observed_monetary_fields != fixed_monetary_fields:
        raise ExploratoryValidationMetadataError(
            "exploratory monetary readiness fields differ from the fixed contract"
        )
    if type(fixed_seed_count) is not int or fixed_seed_count != 150:
        raise ExploratoryValidationMetadataError(
            "exploratory validation requires exactly 150 fixed seeds"
        )
    uncertainty_payload = _json_copy(uncertainty_design)
    convergence_payload = _json_copy(convergence_rule)
    if not isinstance(uncertainty_payload, dict) or not isinstance(
        convergence_payload, dict
    ):
        raise ExploratoryValidationMetadataError(
            "uncertainty and convergence declarations must be objects"
        )

    scenarios = [_json_copy(item) for item in scenario_snapshots]
    if not scenarios:
        raise ExploratoryValidationMetadataError(
            "exploratory validation requires at least one declared scenario"
        )
    scenario_ids: list[str] = []
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ExploratoryValidationMetadataError(
                f"scenario_snapshots[{index}] must be an object"
            )
        scenario_id = scenario.get("scenario_id")
        if type(scenario_id) is not str:
            raise ExploratoryValidationMetadataError(
                f"scenario_snapshots[{index}] lacks a string scenario_id"
            )
        _identifier(scenario_id, name=f"scenario_snapshots[{index}].scenario_id")
        scenario_ids.append(scenario_id)
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ExploratoryValidationMetadataError("scenario IDs must be unique")
    scenario_set_payload = {
        "ordered_scenario_ids": scenario_ids,
        "scenarios": scenarios,
    }
    scenario_set_sha256 = _canonical_sha256(scenario_set_payload)
    paired_identity_payload = {
        "population_adapter_sha256": population_adapter_sha256,
        "population_execution_input_sha256": population_execution_input_sha256,
        "scenario_set_sha256": scenario_set_sha256,
        "ordered_scenario_ids": scenario_ids,
        "identical_pretreatment_cohorts_required": True,
        "identical_population_assignments_across_scenarios_required": True,
        "identical_population_weights_across_scenarios_required": True,
        "population_weights_applied_within_seed_before_aggregation": True,
    }
    population_scenario_binding_sha256 = _canonical_sha256(
        paired_identity_payload
    )

    identity_payload: dict[str, object] = {
        "schema_version": EXPLORATORY_VALIDATION_SCHEMA_VERSION,
        "output_profile": EXPLORATORY_VALIDATION_OUTPUT_PROFILE,
        "artifact_namespace": EXPLORATORY_ARTIFACT_NAMESPACE,
        "execution_kind": EXPLORATORY_EXECUTION_KIND,
        "validation_status": "VALIDATED_NOT_EXECUTED",
        "interpretation_wording": EXPLORATORY_INTERPRETATION_WORDING,
        "campaign_ready": False,
        "production_campaign": False,
        "claims": {
            "empirical_validation_claimed": False,
            "population_inference_claimed": False,
            "real_world_causal_effect_claimed": False,
            "generalisation_claimed": False,
            "observed_real_world_spending_claimed": False,
            "production_readiness_claimed": False,
        },
        "execution": {
            "status": "NOT_EXECUTED",
            "validation_only": True,
            "cohort_initialized": False,
            "scenario_executed": False,
            "policy_batch_dispatched": False,
            "sensitivity_dispatched": False,
            "reproduce_dispatched": False,
            "intended_launch_command": list(EXPLORATORY_LAUNCH_COMMAND),
        },
        "configuration": {
            "path": configuration_path,
            "sha256": configuration_sha256,
            "run_purpose": "exploratory",
            "full_exploratory_config": True,
        },
        "exploratory_analysis_plan": {
            "path": exploratory_plan_path,
            "plan_id": exploratory_plan_id,
            "plan_sha256": exploratory_plan_sha256,
            "file_sha256": exploratory_plan_file_sha256,
            "campaign_ready": False,
            "empirical_interpretation_allowed": False,
            "production_authority": "NONE",
            "primary_estimand_id": primary_estimand_id,
        },
        "scientific_parent_plan": {
            "path": scientific_parent_plan_path,
            "plan_id": scientific_parent_plan_id,
            "plan_sha256": scientific_parent_plan_sha256,
            "file_sha256": scientific_parent_plan_file_sha256,
            "registration_status": scientific_parent_registration_status,
            "primary_estimand_id": primary_estimand_id,
        },
        "population_scenario_identity": {
            **paired_identity_payload,
            "population_adapter_id": population_adapter_id,
            "population_basis": EXPLORATORY_POPULATION_BASIS,
            "scenario_set": scenario_set_payload,
            "binding_sha256": population_scenario_binding_sha256,
            "same_weighted_population_scenario_identity_declared": True,
            "realized_assignment_status": "DECLARED_NOT_EXECUTED",
        },
        "monetary_contract": {
            "target_currency": target_currency,
            "target_minor_unit_name": target_minor_unit_name,
            "quote_convention": quote_convention,
            "scale_convention": scale_convention,
            "rate_period_start": rate_period_start,
            "rate_period_end": rate_period_end,
            "target_price_period_start": target_price_period_start,
            "target_price_period_end": target_price_period_end,
            "missing_date_policy": missing_date_policy,
            "identity_missing_date_policy": identity_missing_date_policy,
            "rounding_method": rounding_method,
            "rounding_scope": rounding_scope,
            "point_rate_status": point_rate_status,
            "rate_uncertainty_status": rate_uncertainty_status,
            "source_bundle_signature_status": source_bundle_signature_status,
            "simulation_bridge_status": simulation_bridge_status,
            "internal_monetary_unit": EXPLORATORY_INTERNAL_MONETARY_UNIT,
            "raw_internal_unit_output_role": (
                EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE
            ),
            "monetary_amount_semantics": (
                EXPLORATORY_MONETARY_AMOUNT_SEMANTICS
            ),
            "internal_unit_wording": EXPLORATORY_INTERNAL_UNIT_WORDING,
            "source_bundle_sha256": monetary_source_bundle_sha256,
            "source_artifact_sha256": monetary_source_artifact_sha256,
            "conversion_table_sha256": monetary_conversion_table_sha256,
            "conversion_basis_sha256": monetary_conversion_basis_sha256,
            "rate_evidence_sha256": monetary_rate_evidence_sha256,
            "raw_simulation_cents_are_real_money": False,
            "raw_simulation_cents_allowed_as_cross_country_result": False,
            "observed_real_world_spending": observed_real_world_spending,
            "observed_fx_conversion": True,
            "internal_to_real_money_bridge_empirically_calibrated": False,
            "raw_cross_currency_pooling": raw_cross_currency_pooling,
            "model_equivalent_label_required": True,
        },
        "output_interpretation": {
            "estimand_interpretation": (
                EXPLORATORY_ESTIMAND_INTERPRETATION
            ),
            "unweighted_output_role": EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE,
            "population_weighted_primary_estimand_required": True,
            "human_readable_label_required": (
                EXPLORATORY_INTERPRETATION_WORDING
            ),
        },
        "uncertainty_and_convergence": {
            "fixed_seed_count": fixed_seed_count,
            "component_statuses": {
                "seed": "QUANTIFIED_WHEN_COMPLETE",
                "parameter": "ILLUSTRATIVE_DESIGN_ONLY",
                "monetary_rate": "UNQUANTIFIED",
                "population": "UNQUANTIFIED",
                "combined": "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST",
            },
            "oat_role": "DIAGNOSTIC_ONLY",
            "parameter_design_file": {
                "path": parameter_design_path,
                "file_sha256": parameter_design_file_sha256,
                "design_id": uncertainty_payload.get(
                    "parameter_uncertainty", {}
                ).get("design_id"),
                "design_sha256": uncertainty_payload.get(
                    "parameter_uncertainty", {}
                ).get("design_sha256"),
            },
            "uncertainty_design": uncertainty_payload,
            "convergence_rule": convergence_payload,
            "convergence_status": "NOT_EVALUATED_NOT_EXECUTED",
            "monte_carlo_diagnostics": "NOT_COMPUTED",
            "sensitivity_status": "NOT_EXECUTED",
            "convergence_claimed": False,
        },
        "profile_input_lineage_sha256": profile_input_lineage_sha256,
        "production_boundary": {
            "configuration_path": "configs/policy_campaign.toml",
            "configuration_sha256": production_configuration_sha256,
            "plan_path": "inputs/prospective-analysis-plan-amendment-v3.json",
            "plan_file_sha256": production_plan_sha256,
            "production_configuration_used_as_execution_input": False,
            "production_configuration_modified_by_validation": False,
            "scientific_parent_plan_reused": True,
            "scientific_parent_plan_modified_by_validation": False,
            "production_campaign_authority_inherited": False,
            "production_gates_altered": False,
        },
    }
    metadata = {
        **identity_payload,
        "metadata_sha256": _canonical_sha256(identity_payload),
    }
    validate_exploratory_validation_metadata(metadata)
    return metadata


def validate_exploratory_validation_metadata(
    metadata: Mapping[str, object],
) -> None:
    """Reject any metadata that promotes validation into execution or evidence."""

    if not isinstance(metadata, Mapping):
        raise TypeError("exploratory validation metadata must be a mapping")
    payload = _json_copy(metadata)
    if not isinstance(payload, dict):
        raise ExploratoryValidationMetadataError("metadata must be an object")
    expected_top_level = {
        "schema_version",
        "output_profile",
        "artifact_namespace",
        "execution_kind",
        "validation_status",
        "interpretation_wording",
        "campaign_ready",
        "production_campaign",
        "claims",
        "execution",
        "configuration",
        "exploratory_analysis_plan",
        "scientific_parent_plan",
        "population_scenario_identity",
        "monetary_contract",
        "output_interpretation",
        "uncertainty_and_convergence",
        "profile_input_lineage_sha256",
        "production_boundary",
        "metadata_sha256",
    }
    if set(payload) != expected_top_level:
        raise ExploratoryValidationMetadataError(
            "exploratory validation metadata fields differ from the contract"
        )
    fixed = {
        "schema_version": EXPLORATORY_VALIDATION_SCHEMA_VERSION,
        "output_profile": EXPLORATORY_VALIDATION_OUTPUT_PROFILE,
        "artifact_namespace": EXPLORATORY_ARTIFACT_NAMESPACE,
        "execution_kind": EXPLORATORY_EXECUTION_KIND,
        "validation_status": "VALIDATED_NOT_EXECUTED",
        "interpretation_wording": EXPLORATORY_INTERPRETATION_WORDING,
        "campaign_ready": False,
        "production_campaign": False,
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise ExploratoryValidationMetadataError(
            "exploratory validation fixed scope fields were altered"
        )
    if payload.get("claims") != {
        "empirical_validation_claimed": False,
        "population_inference_claimed": False,
        "real_world_causal_effect_claimed": False,
        "generalisation_claimed": False,
        "observed_real_world_spending_claimed": False,
        "production_readiness_claimed": False,
    }:
        raise ExploratoryValidationMetadataError(
            "exploratory validation cannot make empirical, population, causal, "
            "generalisation, or production claims"
        )
    if payload.get("execution") != {
        "status": "NOT_EXECUTED",
        "validation_only": True,
        "cohort_initialized": False,
        "scenario_executed": False,
        "policy_batch_dispatched": False,
        "sensitivity_dispatched": False,
        "reproduce_dispatched": False,
        "intended_launch_command": list(EXPLORATORY_LAUNCH_COMMAND),
    }:
        raise ExploratoryValidationMetadataError(
            "exploratory validation execution state must remain not executed"
        )
    monetary = _mapping(payload.get("monetary_contract"), name="monetary_contract")
    expected_monetary_keys = {
        "target_currency",
        "target_minor_unit_name",
        "quote_convention",
        "scale_convention",
        "rate_period_start",
        "rate_period_end",
        "target_price_period_start",
        "target_price_period_end",
        "missing_date_policy",
        "identity_missing_date_policy",
        "rounding_method",
        "rounding_scope",
        "point_rate_status",
        "rate_uncertainty_status",
        "source_bundle_signature_status",
        "simulation_bridge_status",
        "internal_monetary_unit",
        "raw_internal_unit_output_role",
        "monetary_amount_semantics",
        "internal_unit_wording",
        "source_bundle_sha256",
        "source_artifact_sha256",
        "conversion_table_sha256",
        "conversion_basis_sha256",
        "rate_evidence_sha256",
        "raw_simulation_cents_are_real_money",
        "raw_simulation_cents_allowed_as_cross_country_result",
        "observed_real_world_spending",
        "observed_fx_conversion",
        "internal_to_real_money_bridge_empirically_calibrated",
        "raw_cross_currency_pooling",
        "model_equivalent_label_required",
    }
    if set(monetary) != expected_monetary_keys:
        raise ExploratoryValidationMetadataError(
            "exploratory monetary fields differ from the strict contract"
        )
    for key, expected in {
        "target_currency": "EUR",
        "target_minor_unit_name": "euro cent",
        "quote_convention": "target minor units per source minor unit",
        "scale_convention": (
            "local nominal monthly anchor minor units per 180000 simulation_cents"
        ),
        "rate_period_start": "2024-01-01",
        "rate_period_end": "2024-12-31",
        "target_price_period_start": "2024-01-01",
        "target_price_period_end": "2024-12-31",
        "missing_date_policy": (
            "use the official annual observation without local date filling or imputation"
        ),
        "identity_missing_date_policy": (
            "not applicable to an identity conversion"
        ),
        "rounding_method": "nearest_minor_unit_half_away_from_zero",
        "rounding_scope": "AFTER_AGGREGATION",
        "internal_monetary_unit": EXPLORATORY_INTERNAL_MONETARY_UNIT,
        "raw_internal_unit_output_role": (
            EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE
        ),
        "monetary_amount_semantics": EXPLORATORY_MONETARY_AMOUNT_SEMANTICS,
        "internal_unit_wording": EXPLORATORY_INTERNAL_UNIT_WORDING,
        "point_rate_status": "OFFICIAL_POINT_OBSERVATION",
        "rate_uncertainty_status": "UNQUANTIFIED",
        "source_bundle_signature_status": "MISSING",
        "simulation_bridge_status": "ILLUSTRATIVE",
        "raw_simulation_cents_are_real_money": False,
        "raw_simulation_cents_allowed_as_cross_country_result": False,
        "observed_real_world_spending": False,
        "observed_fx_conversion": True,
        "internal_to_real_money_bridge_empirically_calibrated": False,
        "raw_cross_currency_pooling": "REJECT",
        "model_equivalent_label_required": True,
    }.items():
        if monetary.get(key) != expected:
            raise ExploratoryValidationMetadataError(
                f"exploratory monetary field {key} violates the unit contract"
            )
    for key in (
        "source_bundle_sha256",
        "source_artifact_sha256",
        "conversion_table_sha256",
        "conversion_basis_sha256",
        "rate_evidence_sha256",
    ):
        _digest(monetary.get(key), name=f"monetary_contract.{key}")
    interpretation = _mapping(
        payload.get("output_interpretation"), name="output_interpretation"
    )
    if interpretation != {
        "estimand_interpretation": EXPLORATORY_ESTIMAND_INTERPRETATION,
        "unweighted_output_role": EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE,
        "population_weighted_primary_estimand_required": True,
        "human_readable_label_required": EXPLORATORY_INTERPRETATION_WORDING,
    }:
        raise ExploratoryValidationMetadataError(
            "exploratory result interpretation wording is not canonical"
        )
    uncertainty = _mapping(
        payload.get("uncertainty_and_convergence"),
        name="uncertainty_and_convergence",
    )
    expected_uncertainty_fixed = {
        "fixed_seed_count": 150,
        "component_statuses": {
            "seed": "QUANTIFIED_WHEN_COMPLETE",
            "parameter": "ILLUSTRATIVE_DESIGN_ONLY",
            "monetary_rate": "UNQUANTIFIED",
            "population": "UNQUANTIFIED",
            "combined": "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST",
        },
        "oat_role": "DIAGNOSTIC_ONLY",
        "convergence_status": "NOT_EVALUATED_NOT_EXECUTED",
        "monte_carlo_diagnostics": "NOT_COMPUTED",
        "sensitivity_status": "NOT_EXECUTED",
        "convergence_claimed": False,
    }
    if any(
        uncertainty.get(key) != value
        for key, value in expected_uncertainty_fixed.items()
    ):
        raise ExploratoryValidationMetadataError(
            "exploratory uncertainty/convergence status overstates execution"
        )
    design = _mapping(
        uncertainty.get("uncertainty_design"), name="uncertainty_design"
    )
    if set(design) != {
        "schema_version",
        "seed_uncertainty",
        "parameter_uncertainty",
        "monetary_rate_uncertainty",
        "population_uncertainty",
        "combined_uncertainty",
        "oat_role",
    } or design.get("schema_version") != "1.0":
        raise ExploratoryValidationMetadataError(
            "exploratory uncertainty design differs from the strict contract"
        )
    parameter_design_file = _mapping(
        uncertainty.get("parameter_design_file"),
        name="parameter_design_file",
    )
    if set(parameter_design_file) != {
        "path",
        "file_sha256",
        "design_id",
        "design_sha256",
    }:
        raise ExploratoryValidationMetadataError(
            "parameter design file identity fields differ from the contract"
        )
    _relative_path(
        parameter_design_file.get("path"), name="parameter_design_file.path"
    )
    _digest(
        parameter_design_file.get("file_sha256"),
        name="parameter_design_file.file_sha256",
    )
    component_paths = {
        "seed_uncertainty": "QUANTIFIED_WHEN_COMPLETE",
        "parameter_uncertainty": "ILLUSTRATIVE_DESIGN_ONLY",
        "monetary_rate_uncertainty": "UNQUANTIFIED",
        "population_uncertainty": "UNQUANTIFIED",
        "combined_uncertainty": (
            "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST"
        ),
    }
    for component, expected_status in component_paths.items():
        row = _mapping(design.get(component), name=component)
        if row.get("status") != expected_status:
            raise ExploratoryValidationMetadataError(
                f"exploratory {component} status differs from the plan"
            )
    seed_component = _mapping(
        design.get("seed_uncertainty"), name="seed_uncertainty"
    )
    if seed_component != {
        "common_random_numbers": True,
        "fixed_seed_count": 150,
        "identical_pretreatment_cohorts": True,
        "outcome_dependent_seed_exclusion_allowed": False,
        "population_weights_applied_within_seed": True,
        "status": "QUANTIFIED_WHEN_COMPLETE",
    }:
        raise ExploratoryValidationMetadataError(
            "exploratory seed uncertainty differs from the fixed design"
        )
    parameter_component = _mapping(
        design.get("parameter_uncertainty"), name="parameter_uncertainty"
    )
    if (
        set(parameter_component)
        != {
            "design_id",
            "design_sha256",
            "method",
            "probability_interpretation",
            "status",
        }
        or parameter_component.get("method") != "SEEDED_LATIN_HYPERCUBE_V1"
        or parameter_component.get("probability_interpretation") != "NONE"
        or parameter_component.get("status") != "ILLUSTRATIVE_DESIGN_ONLY"
    ):
        raise ExploratoryValidationMetadataError(
            "exploratory parameter uncertainty differs from the fixed design"
        )
    if (
        parameter_design_file.get("design_id")
        != parameter_component.get("design_id")
        or parameter_design_file.get("design_sha256")
        != parameter_component.get("design_sha256")
    ):
        raise ExploratoryValidationMetadataError(
            "parameter design file identity differs from the uncertainty plan"
        )
    _identifier(
        parameter_design_file.get("design_id"),
        name="parameter_design_file.design_id",
    )
    _digest(
        parameter_design_file.get("design_sha256"),
        name="parameter_design_file.design_sha256",
    )
    rate_component = _mapping(
        design.get("monetary_rate_uncertainty"),
        name="monetary_rate_uncertainty",
    )
    if (
        set(rate_component)
        != {"point_observation_is_distribution", "rate_basis_sha256", "status"}
        or rate_component.get("point_observation_is_distribution") is not False
        or rate_component.get("status") != "UNQUANTIFIED"
    ):
        raise ExploratoryValidationMetadataError(
            "exploratory monetary-rate uncertainty differs from the fixed design"
        )
    _digest(
        rate_component.get("rate_basis_sha256"),
        name="monetary_rate_uncertainty.rate_basis_sha256",
    )
    population_component = _mapping(
        design.get("population_uncertainty"), name="population_uncertainty"
    )
    if (
        set(population_component)
        != {"exact_weighting_is_empirical_validation", "status", "uncertainty_design_id"}
        or population_component.get("exact_weighting_is_empirical_validation")
        is not False
        or population_component.get("status") != "UNQUANTIFIED"
    ):
        raise ExploratoryValidationMetadataError(
            "exploratory population uncertainty differs from the fixed design"
        )
    _identifier(
        population_component.get("uncertainty_design_id"),
        name="population_uncertainty.uncertainty_design_id",
    )
    combined_component = _mapping(
        design.get("combined_uncertainty"), name="combined_uncertainty"
    )
    if combined_component != {
        "double_counting_control": (
            "one complete seed-parameter-population-rate Cartesian identity"
        ),
        "status": "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST",
        "variance_decomposition_method": (
            "ORTHOGONAL_FINITE_FULL_FACTORIAL_ANOVA_SUM_OF_SQUARES_DIVIDED_BY_N_V1"
        ),
    }:
        raise ExploratoryValidationMetadataError(
            "exploratory combined uncertainty differs from the fixed design"
        )
    if design.get("oat_role") != "DIAGNOSTIC_ONLY":
        raise ExploratoryValidationMetadataError(
            "exploratory OAT role must be diagnostic only"
        )
    convergence = _mapping(
        uncertainty.get("convergence_rule"), name="convergence_rule"
    )
    if convergence != {
        "schema_version": "1.0",
        "block_size": 50,
        "minimum_retained_seeds": 100,
        "maximum_mcse": 0.0025,
        "maximum_interval_width": 0.01,
        "maximum_absolute_change": 0.0025,
        "maximum_relative_change": 0.025,
        "maximum_invalid_rate": 0.0,
        "consecutive_passing_checkpoints": 2,
        "sensitivity_instability_allowed": False,
        "outcome_dependent_seed_exclusion_allowed": False,
        "required_uncertainty_component_handling": "FAIL_CLOSED",
    }:
        raise ExploratoryValidationMetadataError(
            "exploratory convergence rule is not fail-closed"
        )
    binding = _mapping(
        payload.get("population_scenario_identity"),
        name="population_scenario_identity",
    )
    for key, expected in {
        "population_basis": EXPLORATORY_POPULATION_BASIS,
        "identical_pretreatment_cohorts_required": True,
        "identical_population_assignments_across_scenarios_required": True,
        "identical_population_weights_across_scenarios_required": True,
        "population_weights_applied_within_seed_before_aggregation": True,
        "same_weighted_population_scenario_identity_declared": True,
        "realized_assignment_status": "DECLARED_NOT_EXECUTED",
    }.items():
        if binding.get(key) != expected:
            raise ExploratoryValidationMetadataError(
                f"population/scenario identity field {key} is not fail-closed"
            )
    scenario_set = _mapping(binding.get("scenario_set"), name="scenario_set")
    if set(scenario_set) != {"ordered_scenario_ids", "scenarios"}:
        raise ExploratoryValidationMetadataError(
            "scenario-set identity fields differ from the contract"
        )
    if binding.get("ordered_scenario_ids") != scenario_set.get(
        "ordered_scenario_ids"
    ):
        raise ExploratoryValidationMetadataError(
            "scenario-set ordering differs from the paired identity"
        )
    observed_scenario_set_sha256 = _canonical_sha256(scenario_set)
    if binding.get("scenario_set_sha256") != observed_scenario_set_sha256:
        raise ExploratoryValidationMetadataError(
            "scenario_set_sha256 does not match the exact scenario declarations"
        )
    for key in (
        "population_adapter_sha256",
        "population_execution_input_sha256",
        "scenario_set_sha256",
        "binding_sha256",
    ):
        _digest(binding.get(key), name=f"population_scenario_identity.{key}")
    paired = {
        "population_adapter_sha256": binding.get("population_adapter_sha256"),
        "population_execution_input_sha256": binding.get(
            "population_execution_input_sha256"
        ),
        "scenario_set_sha256": binding.get("scenario_set_sha256"),
        "ordered_scenario_ids": binding.get("ordered_scenario_ids"),
        "identical_pretreatment_cohorts_required": True,
        "identical_population_assignments_across_scenarios_required": True,
        "identical_population_weights_across_scenarios_required": True,
        "population_weights_applied_within_seed_before_aggregation": True,
    }
    if binding.get("binding_sha256") != _canonical_sha256(paired):
        raise ExploratoryValidationMetadataError(
            "population/scenario binding SHA-256 does not match its identities"
        )
    production = _mapping(
        payload.get("production_boundary"), name="production_boundary"
    )
    if (
        production.get("production_configuration_used_as_execution_input")
        is not False
        or production.get("production_configuration_modified_by_validation")
        is not False
        or production.get("scientific_parent_plan_reused")
        is not True
        or production.get("scientific_parent_plan_modified_by_validation")
        is not False
        or production.get("production_campaign_authority_inherited") is not False
        or production.get("production_gates_altered") is not False
    ):
        raise ExploratoryValidationMetadataError(
            "exploratory validation must leave production inputs untouched"
        )
    observed_digest = payload.pop("metadata_sha256", None)
    _digest(observed_digest, name="metadata_sha256")
    if observed_digest != _canonical_sha256(payload):
        raise ExploratoryValidationMetadataError(
            "metadata_sha256 does not match the exploratory validation payload"
        )


def require_exploratory_manifest_metadata(
    *,
    exploratory_requested: bool,
    metadata: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Require the canonical review payload exactly for exploratory exports.

    This deliberately has no receipt semantics.  It only prevents future
    exploratory output from omitting, or non-exploratory output from acquiring,
    the mandatory non-empirical interpretation contract.
    """

    if type(exploratory_requested) is not bool:
        raise TypeError("exploratory_requested must be boolean")
    if not exploratory_requested:
        if metadata is not None:
            raise ExploratoryValidationMetadataError(
                "exploratory validation metadata is forbidden for "
                "non-exploratory output"
            )
        return None
    if metadata is None:
        raise ExploratoryValidationMetadataError(
            "exploratory output requires verified validation metadata"
        )
    validate_exploratory_validation_metadata(metadata)
    payload = _json_copy(metadata)
    assert isinstance(payload, dict)
    return payload


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ExploratoryValidationMetadataError(
            "exploratory metadata must be canonical JSON-compatible"
        ) from exc


def _json_copy(value: object) -> object:
    return json.loads(_canonical_json(value))


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ExploratoryValidationMetadataError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ExploratoryValidationMetadataError(
            f"{name} must be a canonical identifier"
        )
    return value


def _relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ExploratoryValidationMetadataError(
            f"{name} must be a POSIX relative path"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ExploratoryValidationMetadataError(
            f"{name} must be a canonical repository-relative path"
        )
    if path.as_posix() != value:
        raise ExploratoryValidationMetadataError(
            f"{name} must be lexically canonical"
        )
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ExploratoryValidationMetadataError(
            f"{name} must be non-empty text without surrounding whitespace"
        )
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExploratoryValidationMetadataError(f"{name} must be an object")
    return value


__all__ = [
    "EXPLORATORY_ARTIFACT_NAMESPACE",
    "EXPLORATORY_ESTIMAND_INTERPRETATION",
    "EXPLORATORY_EXECUTION_KIND",
    "EXPLORATORY_INTERNAL_MONETARY_UNIT",
    "EXPLORATORY_INTERNAL_UNIT_WORDING",
    "EXPLORATORY_INTERPRETATION_WORDING",
    "EXPLORATORY_LAUNCH_COMMAND",
    "EXPLORATORY_MONETARY_AMOUNT_SEMANTICS",
    "EXPLORATORY_POPULATION_BASIS",
    "EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE",
    "EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE",
    "EXPLORATORY_VALIDATION_OUTPUT_PROFILE",
    "EXPLORATORY_VALIDATION_SCHEMA_VERSION",
    "ExploratoryValidationMetadataError",
    "build_exploratory_validation_metadata",
    "require_exploratory_manifest_metadata",
    "validate_exploratory_validation_metadata",
]
