"""Build the fail-closed schema-v3 campaign successor plan without simulation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from microtx_sim.analysis.uncertainty import (
    ConvergenceRule,
    canonical_sha256,
    load_parameter_uncertainty_design,
)
from microtx_sim.causal.analysis_plan import (
    FixedSeedStoppingRule,
    build_prospective_analysis_plan,
    load_prospective_analysis_plan,
    verify_loaded_prospective_analysis_plan,
)
from microtx_sim.causal.batch import resolve_policy_run_inputs
from microtx_sim.data.lineage import build_profile_input_lineage
from microtx_sim.data.population_execution import (
    population_execution_input_sha256,
    resolve_population_projection_adapter,
)
from microtx_sim.data.profiles import load_profile_bundle
from microtx_sim.data.rate_evidence import load_and_verify_rate_evidence_bundle
from microtx_sim.outputs.schema import CAMPAIGN_ANALYSIS_SCHEMA_SHA256
from microtx_sim.outputs.writers import write_text_atomic
from microtx_sim.policy_config import load_policy_config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "policy_campaign.toml"
PARENT_PLAN_PATH = REPOSITORY_ROOT / "inputs" / "prospective-analysis-plan.json"
PLAN_PATH = (
    REPOSITORY_ROOT / "inputs" / "prospective-analysis-plan-amendment-v3.json"
)
PARAMETER_DESIGN_PATH = (
    REPOSITORY_ROOT / "inputs" / "parameter-uncertainty-design-v1.json"
)
EXECUTION_RECEIPT_SCHEMA_PATH = (
    REPOSITORY_ROOT / "schemas" / "execution-receipt.schema.json"
)
OUTPUT_SCHEMA_IMPLEMENTATION_PATH = (
    REPOSITORY_ROOT / "src" / "microtx_sim" / "outputs" / "schema.py"
)


_READINESS_BLOCKERS = (
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


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _changed_input(
    *,
    role: str,
    path: Path,
    schema_version: str,
    semantic_sha256: str,
    readiness_status: str,
    readiness_consequence: str,
) -> dict[str, object]:
    return {
        "role": role,
        "artifact_path": path.relative_to(REPOSITORY_ROOT).as_posix(),
        "schema_version": schema_version,
        "file_sha256": _file_sha256(path),
        "semantic_sha256": semantic_sha256,
        "change_type": "BOUND_SUCCESSOR_IMPLEMENTATION_INPUT",
        "readiness_status": readiness_status,
        "readiness_consequence": readiness_consequence,
    }


def build_plan() -> object:
    """Re-attest every bound input and build the immutable successor plan."""

    config = load_policy_config(CONFIG_PATH)
    parent_loaded = verify_loaded_prospective_analysis_plan(
        load_prospective_analysis_plan(PARENT_PLAN_PATH)
    )
    parent = parent_loaded.plan
    if not config.full_campaign_config or config.campaign is None:
        raise RuntimeError("campaign plan builder requires the full campaign config")
    if config.campaign.campaign_ready:
        raise RuntimeError("the current successor plan must remain fail closed")
    if config.population is None or config.monetary_contract is None:
        raise RuntimeError("campaign population and monetary contracts are required")
    if config.uncertainty is None or config.convergence is None:
        raise RuntimeError("campaign uncertainty and convergence are required")
    if config.analysis_plan is None:
        raise RuntimeError("campaign analysis-plan selection is required")
    if (
        config.analysis_plan.parent_plan_id != parent.plan_id
        or config.analysis_plan.parent_plan_sha256 != parent.plan_sha256
    ):
        raise RuntimeError("configured parent plan identity does not re-attest")

    profiles = load_profile_bundle(
        config.monetary_contract.profile_path,
        config.population.source_registry_path,
        source_bundle_path=config.monetary_contract.source_bundle_path,
        population_bundle_path=config.population.evidence_bundle_path,
        campaign=False,
    )
    profile_lineage = build_profile_input_lineage(
        profiles.country_profiles,
        profile_bundle=profiles,
    )
    adapter = resolve_population_projection_adapter(
        config.population,
        profiles,
        player_count=config.batch.player_count,
        campaign=False,
    )
    run_inputs = resolve_policy_run_inputs(
        harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
    )
    parameter_design = load_parameter_uncertainty_design(
        config.uncertainty.parameter_design_path
    )
    rate_bundle, rate_results = load_and_verify_rate_evidence_bundle(
        config.monetary_contract.source_bundle_path,
        required_source_registry_sha256=profiles.source_registry_sha256,
    )
    rate_result_set_sha256 = canonical_sha256(
        [result.snapshot() for result in rate_results]
    )
    if rate_result_set_sha256 != config.monetary_contract.rate_evidence_sha256:
        raise RuntimeError("configured rate-evidence result-set hash differs")

    verification = adapter.verification
    evidence_result_set_sha256 = canonical_sha256(
        [result.snapshot() for result in verification.evidence_results]
    )
    changed_inputs = [
        _changed_input(
            role="campaign_output_contract",
            path=OUTPUT_SCHEMA_IMPLEMENTATION_PATH,
            schema_version="1.0",
            semantic_sha256=CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
            readiness_status="STRUCTURALLY_ATTESTED",
            readiness_consequence=(
                "production-shaped artifacts and receipt references are required; "
                "the profile does not itself imply readiness"
            ),
        ),
        _changed_input(
            role="execution_receipt_schema",
            path=EXECUTION_RECEIPT_SCHEMA_PATH,
            schema_version="1.0",
            semantic_sha256=_file_sha256(EXECUTION_RECEIPT_SCHEMA_PATH),
            readiness_status="IMPLEMENTED_NOT_EXECUTED",
            readiness_consequence=(
                "a clean-tree pre-run and matching post-run receipt remain required"
            ),
        ),
        _changed_input(
            role="monetary_conversion_profile",
            path=config.monetary_contract.profile_path,
            schema_version=str(profiles.jurisdiction_schema_version),
            semantic_sha256=profile_lineage.fingerprint_sha256,
            readiness_status="BLOCKED",
            readiness_consequence=(
                "the exact rational production basis is bound, but the "
                "simulation-cent bridge remains illustrative"
            ),
        ),
        _changed_input(
            role="monetary_conversion_table",
            path=config.monetary_contract.conversion_table_path,
            schema_version="1",
            semantic_sha256=canonical_sha256(
                [
                    result.snapshot()
                    for result in rate_results
                    if result.artifact_sha256
                    == config.monetary_contract.conversion_table_sha256
                ]
            ),
            readiness_status="POINT_RATES_ONLY",
            readiness_consequence=(
                "the table is exact and verified but supplies no rate distribution"
            ),
        ),
        _changed_input(
            role="monetary_official_rate_artifact",
            path=config.monetary_contract.source_artifact_path,
            schema_version="ECB_EXR_CSV_2024",
            semantic_sha256=canonical_sha256(
                [
                    result.snapshot()
                    for result in rate_results
                    if result.artifact_sha256
                    == config.monetary_contract.source_artifact_sha256
                ]
            ),
            readiness_status="OFFICIAL_POINT_OBSERVATION",
            readiness_consequence=(
                "official observations do not quantify monetary-rate uncertainty"
            ),
        ),
        _changed_input(
            role="monetary_source_bundle",
            path=config.monetary_contract.source_bundle_path,
            schema_version=str(rate_bundle.schema_version),
            semantic_sha256=rate_result_set_sha256,
            readiness_status="BLOCKED_SIGNATURE_MISSING",
            readiness_consequence=(
                "artifact recipes re-attest, but the source bundle signature is missing"
            ),
        ),
        _changed_input(
            role="parameter_uncertainty_design",
            path=PARAMETER_DESIGN_PATH,
            schema_version="1.0",
            semantic_sha256=parameter_design.design.design_sha256,
            readiness_status="ILLUSTRATIVE_DESIGN_ONLY",
            readiness_consequence=(
                "deterministic joint draws are supported, but their ranges are not "
                "calibrated probability distributions"
            ),
        ),
        _changed_input(
            role="population_design",
            path=config.population.design_bundle_path,
            schema_version=str(verification.bundle.schema_version),
            semantic_sha256=adapter.calibration_target_sha256,
            readiness_status="BLOCKED",
            readiness_consequence=(
                "the complete illustrative design is not empirical population validation"
            ),
        ),
        _changed_input(
            role="population_evidence_bundle",
            path=config.population.evidence_bundle_path,
            schema_version=str(verification.evidence_bundle.schema_version),
            semantic_sha256=evidence_result_set_sha256,
            readiness_status="BLOCKED",
            readiness_consequence=(
                "content-addressed calibration evidence does not supply a valid "
                "population-uncertainty design"
            ),
        ),
        _changed_input(
            role="population_joint_artifact",
            path=(
                REPOSITORY_ROOT
                / "data"
                / "provenance"
                / "population_artifacts"
                / "joint_population.csv"
            ),
            schema_version="1",
            semantic_sha256=_file_sha256(
                REPOSITORY_ROOT
                / "data"
                / "provenance"
                / "population_artifacts"
                / "joint_population.csv"
            ),
            readiness_status="STRUCTURALLY_ATTESTED",
            readiness_consequence=(
                "exact cell masses and balance are design facts, not empirical validation"
            ),
        ),
        _changed_input(
            role="population_runtime_mapping",
            path=config.population.runtime_mapping_bundle_path,
            schema_version=str(adapter.mapping_bundle.schema_version),
            semantic_sha256=adapter.mapping_bundle.mapping_sha256,
            readiness_status="STRUCTURALLY_ATTESTED",
            readiness_consequence=(
                "the source-to-runtime mapping is fixed; per-seed assignment, balance, "
                "execution, and lineage identities are still required"
            ),
        ),
    ]
    changed_inputs.sort(key=lambda row: str(row["role"]))

    primary = parent.primary_estimand
    rule = ConvergenceRule(
        block_size=config.convergence.block_size,
        minimum_retained_seeds=config.convergence.minimum_retained_seeds,
        maximum_mcse=config.convergence.maximum_mcse,
        maximum_interval_width=config.convergence.maximum_interval_width,
        maximum_absolute_change=config.convergence.maximum_absolute_change,
        maximum_relative_change=config.convergence.maximum_relative_change,
        maximum_invalid_rate=config.convergence.maximum_invalid_rate,
        consecutive_passing_checkpoints=(
            config.convergence.consecutive_passing_checkpoints
        ),
    )
    amendment = {
        "amendment_schema_version": "1.0",
        "parent_plan": {
            "artifact_path": PARENT_PLAN_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "schema_version": parent.schema_version,
            "plan_id": parent.plan_id,
            "plan_sha256": parent.plan_sha256,
            "file_sha256": parent_loaded.file_sha256,
        },
        "scientific_change": {
            "primary_estimand_changed": False,
            "original_estimand_id": primary.estimand_id,
            "current_estimand_id": primary.estimand_id,
            "original_specification_sha256": primary.specification_sha256,
            "current_specification_sha256": primary.specification_sha256,
            "explanation": (
                "The original directed composite-harm estimand, population predicate, "
                "metric contract, scenario direction, horizon semantics, and harm "
                "weights are preserved. This successor changes implementation inputs, "
                "the fixed seed policy, uncertainty, convergence, output, and technical "
                "attestation contracts only."
            ),
        },
        "changed_inputs": changed_inputs,
        "population_contract": {
            "mode": "projected_v1",
            "design_id": verification.bundle.design_id,
            "design_file_sha256": verification.bundle.bundle_sha256,
            "design_sha256": verification.bundle.bundle_sha256,
            "runtime_mapping_id": adapter.mapping_bundle.mapping_id,
            "runtime_mapping_file_sha256": adapter.mapping_bundle.mapping_sha256,
            "runtime_mapping_sha256": adapter.mapping_bundle.mapping_sha256,
            "adapter_id": adapter.adapter_id,
            "adapter_sha256": adapter.adapter_sha256,
            "apportionment_plan_sha256": adapter.apportionment_sha256,
            "population_input_sha256": population_execution_input_sha256(adapter),
            "cell_count": len(adapter.cells),
            "target_design_units": (
                adapter.apportionment_plan.calibration_target.total_population_count
            ),
            "assignment_identity_policy": "PER_SEED_REQUIRED",
            "balance_identity_policy": "EXACT_PER_SEED_REQUIRED",
            "lineage_identity_policy": "COMPLETE_FIXED_SET_REQUIRED",
            "uncertainty_status": "UNQUANTIFIED",
            "empirical_validation_claimed": False,
        },
        "monetary_contract": {
            "source_bundle_id": rate_bundle.bundle_id,
            "source_bundle_file_sha256": rate_bundle.bundle_sha256,
            "source_bundle_semantic_sha256": rate_result_set_sha256,
            "source_artifact_sha256s": {
                config.monetary_contract.conversion_table_path.name: (
                    config.monetary_contract.conversion_table_sha256
                ),
                config.monetary_contract.source_artifact_path.name: (
                    config.monetary_contract.source_artifact_sha256
                ),
            },
            "conversion_table_sha256": (
                config.monetary_contract.conversion_table_sha256
            ),
            "target_currency": config.monetary_contract.target_currency,
            "quote_convention": config.monetary_contract.quote_convention,
            "scale_convention": config.monetary_contract.scale_convention,
            "rate_period_start": config.monetary_contract.rate_period_start,
            "rate_period_end": config.monetary_contract.rate_period_end,
            "price_period_start": (
                config.monetary_contract.target_price_period_start
            ),
            "price_period_end": config.monetary_contract.target_price_period_end,
            "missing_date_policy": config.monetary_contract.missing_date_policy,
            "rounding_rule": config.monetary_contract.rounding_method,
            "rounding_boundary": config.monetary_contract.rounding_scope,
            "conversion_basis_sha256": (
                config.monetary_contract.conversion_basis_sha256
            ),
            "source_bundle_signature_status": (
                config.monetary_contract.source_bundle_signature_status
            ),
            "simulation_bridge_status": (
                config.monetary_contract.simulation_bridge_status
            ),
            "rate_uncertainty_status": "UNQUANTIFIED",
            "observed_real_world_spending_claimed": False,
        },
        "uncertainty_design": {
            "schema_version": "1.0",
            "seed_uncertainty": {
                "status": "QUANTIFIED_WHEN_COMPLETE",
                "fixed_seed_count": len(config.batch.seeds),
                "population_weights_applied_within_seed": True,
                "common_random_numbers": True,
                "identical_pretreatment_cohorts": True,
                "outcome_dependent_seed_exclusion_allowed": False,
            },
            "parameter_uncertainty": {
                "status": "ILLUSTRATIVE_DESIGN_ONLY",
                "design_id": parameter_design.design.design_id,
                "design_sha256": parameter_design.design.design_sha256,
                "method": parameter_design.design.method,
                "probability_interpretation": "NONE",
            },
            "monetary_rate_uncertainty": {
                "status": "UNQUANTIFIED",
                "rate_basis_sha256": (
                    config.monetary_contract.conversion_basis_sha256
                ),
                "point_observation_is_distribution": False,
            },
            "population_uncertainty": {
                "status": "UNQUANTIFIED",
                "uncertainty_design_id": (
                    config.population_contract.uncertainty_design_id
                    if config.population_contract is not None
                    else "UNAVAILABLE_NO_ADMISSIBLE_POPULATION_UNCERTAINTY_DESIGN"
                ),
                "exact_weighting_is_empirical_validation": False,
            },
            "combined_uncertainty": {
                "status": "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST",
                "double_counting_control": (
                    "one complete seed-parameter-population-rate Cartesian identity"
                ),
                "variance_decomposition_method": (
                    config.uncertainty.variance_decomposition_method
                ),
            },
            "oat_role": "DIAGNOSTIC_ONLY",
        },
        "convergence_rule": rule.snapshot(),
        "execution_attestation": {
            "receipt_schema_version": config.execution_receipt.schema_version,
            "pre_run_required": True,
            "post_run_required": True,
            "clean_tree_required": True,
            "environment_match_required": True,
            "mismatch_handling": "REJECT_OR_INVALIDATE",
        },
        "simulation_flow": {
            "execution_layer": "policy_welfare_v1",
            "strategic_world_layer_included": False,
            "incompatibility_status": "FAIL_CLOSED_NO_ADAPTER",
            "monetary_conversion_before_population_aggregation": True,
            "raw_cross_currency_pooling_allowed": False,
        },
        "readiness_consequences": {
            "campaign_ready": False,
            "blockers": list(_READINESS_BLOCKERS),
        },
    }
    return build_prospective_analysis_plan(
        plan_id="illustrative.prospective.composite-harm.baseline-vs-safe.v3",
        expected_causal_design_sha256=parent.expected_causal_design_sha256,
        expected_batch_spec_sha256=config.batch.snapshot_sha256(),
        expected_model_inputs_sha256=run_inputs.snapshot_sha256(),
        expected_population_input_sha256=population_execution_input_sha256(adapter),
        expected_profile_input_sha256=profile_lineage.fingerprint_sha256,
        expected_metric_contract_sha256=parent.expected_metric_contract_sha256,
        expected_harm_weights_sha256=parent.expected_harm_weights_sha256,
        expected_output_profile_sha256=CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
        stopping_rule=FixedSeedStoppingRule(seeds=config.batch.seeds),
        estimands=parent.estimands,
        declared_harm_weights=parent.declared_harm_weights,
        primary_aggregate_rule=parent.primary_aggregate_rule,
        amendment=amendment,
    )


def main() -> None:
    plan = build_plan()
    text = json.dumps(
        plan.snapshot(),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    write_text_atomic(PLAN_PATH, text)
    loaded = verify_loaded_prospective_analysis_plan(
        load_prospective_analysis_plan(PLAN_PATH)
    )
    if loaded.plan != plan:
        raise RuntimeError("written plan differs from the programmatic result")
    print(
        json.dumps(
            {
                "campaign_ready": False,
                "file_sha256": loaded.file_sha256,
                "full_campaign_run": False,
                "plan_id": plan.plan_id,
                "plan_path": str(PLAN_PATH),
                "plan_sha256": plan.plan_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
