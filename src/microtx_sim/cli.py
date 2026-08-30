from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from typing import Sequence

from .analysis import run_sensitivity_analysis
from .causal.analysis_binding import (
    resolve_run_analysis_binding,
    validate_analysis_plan_inputs,
)
from .causal.analysis_plan import (
    LoadedProspectiveAnalysisPlan,
    load_prospective_analysis_plan,
    verify_loaded_prospective_analysis_plan,
)
from .causal.batch import resolve_policy_run_inputs, run_policy_batch
from .config import ConfigurationError, load_config
from .core.ledger import LedgerStorageError
from .core.world import World
from .data.lineage import ProfileInputLineage, build_profile_input_lineage
from .data.profiles import (
    DEFAULT_JURISDICTIONS_PATH,
    ProfileBundle,
    ProfileValidationError,
    load_profile_bundle,
)
from .data.population_execution import resolve_population_projection_adapter
from .execution_attestation import CampaignExecutionRejectedError
from .outputs import export_policy_batch
from .outputs.schema import (
    SENSITIVITY_COLUMNS,
    stamp_standalone_sensitivity_schema,
)
from .outputs.writers import write_csv_atomic, write_json_atomic
from .policy_config import (
    PolicyConfigurationError,
    PolicyRunPurpose,
    load_policy_config,
)
from .simulation import SimulationOrchestrator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microtx-sim",
        description="Causal agent-based model of the mobile-game market.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="validate configuration and evidence contracts"
    )
    validate.add_argument("config", type=Path)

    smoke = commands.add_parser(
        "smoke", help="run only the short structural smoke check"
    )
    smoke.add_argument("config", type=Path)

    policy_validate = commands.add_parser(
        "policy-validate",
        help="validate the synthetic seven-scenario policy configuration",
    )
    policy_validate.add_argument("config", type=Path)

    exploratory_validate = commands.add_parser(
        "exploratory-validate",
        help=(
            "validate the fixed exploratory synthetic campaign contract "
            "without initializing or running any simulation"
        ),
    )
    exploratory_validate.add_argument("config", type=Path)

    campaign_preflight = commands.add_parser(
        "campaign-preflight",
        help=(
            "generate the fail-closed campaign receipt and validation report "
            "without running any simulation"
        ),
    )
    campaign_preflight.add_argument("config", type=Path)
    campaign_preflight.add_argument(
        "--output",
        type=Path,
        help=(
            "optional validation-report path; the execution receipt still uses "
            "the path declared by the campaign configuration"
        ),
    )

    policy_batch = commands.add_parser(
        "policy-batch",
        help="run and export the configured synthetic counterfactual batch",
    )
    policy_batch.add_argument("config", type=Path)
    policy_batch.add_argument("--output", type=Path)
    policy_batch.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="omit the optional OAT sensitivity grid",
    )

    sensitivity = commands.add_parser(
        "policy-sensitivity",
        help="run only the configured synthetic sensitivity analysis",
    )
    sensitivity.add_argument("config", type=Path)
    sensitivity.add_argument("--output", type=Path)

    reproduce = commands.add_parser(
        "reproduce",
        help="reproduce the complete synthetic tables, metadata, and charts",
    )
    reproduce.add_argument("config", type=Path)
    reproduce.add_argument("--output", type=Path)
    return parser


def _validate(config_path: Path) -> dict[str, object]:
    config = load_config(config_path, campaign=False)
    profiles = load_profile_bundle(campaign=False)
    if config.population is not None:
        resolve_population_projection_adapter(
            config.population,
            profiles,
            player_count=config.run.player_count,
        )
    return {
        "status": "ok",
        "scenario": config.meta.name,
        "scenario_provenance": config.meta.provenance_status.value,
        "profile_provenance": profiles.profile_status.value,
        "jurisdictions": [profile.code for profile in profiles.country_profiles],
        "source_records": len(profiles.sources),
        "campaign_ready": False,
        **(
            {"population_mode": config.population.mode.value}
            if config.population is not None
            else {}
        ),
        "caveats": list(profiles.caveats),
    }


def _smoke(config_path: Path) -> dict[str, object]:
    config = load_config(config_path, campaign=False)
    with World.create(config, campaign=False) as world:
        result = SimulationOrchestrator.run(world, campaign=False)
        return {
            "status": "ok",
            "mode": "smoke_only",
            "scenario": config.meta.name,
            "seed": config.run.seed,
            "seed_decimal": str(config.run.seed),
            "cycles": result.cycles,
            "elapsed_seconds": round(result.elapsed_seconds, 6),
            "step_history_retention": config.run.step_history_retention.value,
            "ledger_backend": config.run.ledger_backend.value,
            "summary": result.summary,
            "audit_count": world.audit_count,
            "ledger_entries": world.ledger.entry_count(),
        }


def _policy_validate(config_path: Path) -> dict[str, object]:
    config = load_policy_config(config_path)
    if config.run_purpose is PolicyRunPurpose.EXPLORATORY:
        return _exploratory_validate(config_path)
    campaign = config.run_purpose is PolicyRunPurpose.CAMPAIGN
    population_adapter = None
    profile_input_lineage = None
    if config.population is not None:
        profiles = _load_policy_profiles(config)
        profile_input_lineage = build_profile_input_lineage(
            profiles.country_profiles,
            profile_bundle=profiles,
        )
        population_adapter = resolve_population_projection_adapter(
            config.population,
            profiles,
            player_count=config.batch.player_count,
            campaign=campaign,
        )
        if campaign:
            _validate_policy_campaign_provenance(config)
            profiles.validate_for_campaign()
    analysis_plan = _resolve_configured_analysis_plan(
        config,
        population_adapter=population_adapter,
        profile_input_lineage=profile_input_lineage,
    )
    return {
        "status": "ok",
        "mode": (
            "campaign_policy_preflight"
            if campaign
            else "synthetic_policy_prototype"
        ),
        "run_purpose": config.run_purpose.value,
        "scenario": config.name,
        "provenance_status": config.provenance_status,
        "scenario_count": len(config.batch.scenarios),
        "seeds": list(config.batch.seeds),
        "days": config.batch.days,
        "player_count": config.batch.player_count,
        "personalized_offers_enabled": any(
            scenario.mechanics.personalized_offers
            for scenario in config.batch.scenarios
        ),
        "empirical_validation_claimed": False,
        **(
            {"population_mode": config.population.mode.value}
            if config.population is not None
            else {}
        ),
        **(
            {
                "analysis_plan_sha256": analysis_plan.plan.plan_sha256,
                "analysis_plan_registration_status": (
                    analysis_plan.plan.registration_status.value
                ),
                "campaign_ready": False,
            }
            if analysis_plan is not None
            else {}
        ),
    }


def _exploratory_validate(config_path: Path) -> dict[str, object]:
    """Resolve the fixed exploratory design without initializing a cohort."""

    from .analysis.uncertainty import (
        ConvergenceRule,
        canonical_sha256,
        load_parameter_uncertainty_design,
        verify_loaded_parameter_uncertainty_design,
    )
    from .causal.analysis_plan import (
        load_exploratory_analysis_plan,
        verify_exploratory_analysis_plan_parent,
        verify_loaded_exploratory_analysis_plan,
    )
    from .data.population_execution import population_execution_input_sha256
    from .data.rate_evidence import load_and_verify_rate_evidence_bundle
    from .outputs.exploratory import (
        EXPLORATORY_ARTIFACT_NAMESPACE,
        EXPLORATORY_ESTIMAND_INTERPRETATION,
        EXPLORATORY_EXECUTION_KIND,
        EXPLORATORY_INTERNAL_MONETARY_UNIT,
        EXPLORATORY_MONETARY_AMOUNT_SEMANTICS,
        EXPLORATORY_POPULATION_BASIS,
        EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE,
        EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE,
        build_exploratory_validation_metadata,
    )

    repository_root = Path(__file__).resolve().parents[2]
    config_path = config_path.resolve(strict=True)
    expected_config_path = (
        repository_root / "configs" / "policy_exploratory_synthetic.toml"
    ).resolve(strict=True)
    if config_path != expected_config_path:
        raise PolicyConfigurationError(
            "exploratory-validate is bound to "
            "configs/policy_exploratory_synthetic.toml"
        )
    production_paths = {
        "configuration": repository_root / "configs" / "policy_campaign.toml",
        "scientific_parent": (
            repository_root
            / "inputs"
            / "prospective-analysis-plan-amendment-v3.json"
        ),
    }
    production_before = {
        name: sha256(path.read_bytes()).hexdigest()
        for name, path in production_paths.items()
    }

    config = load_policy_config(config_path)
    if config.run_purpose is not PolicyRunPurpose.EXPLORATORY:
        raise PolicyConfigurationError(
            "exploratory-validate requires run_purpose = 'exploratory'"
        )
    if not config.full_exploratory_config or config.exploratory is None:
        raise PolicyConfigurationError(
            "exploratory-validate requires the complete [exploratory] contract"
        )
    control = config.exploratory
    fixed_control = {
        "artifact_namespace": EXPLORATORY_ARTIFACT_NAMESPACE,
        "execution_kind": EXPLORATORY_EXECUTION_KIND,
        "population_basis": EXPLORATORY_POPULATION_BASIS,
        "estimand_interpretation": EXPLORATORY_ESTIMAND_INTERPRETATION,
        "monetary_amount_semantics": EXPLORATORY_MONETARY_AMOUNT_SEMANTICS,
        "unweighted_output_role": EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE,
        "internal_monetary_unit": EXPLORATORY_INTERNAL_MONETARY_UNIT,
        "raw_internal_unit_output_role": (
            EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE
        ),
        "allow_synthetic": True,
        "campaign_ready": False,
        "production_campaign": False,
        "empirical_claims": False,
        "population_inference_claims": False,
        "causal_claims": False,
        "generalisation_claims": False,
        "identical_pretreatment_cohorts": True,
        "identical_population_weights_across_scenarios": True,
    }
    control_snapshot = control.snapshot()
    if any(control_snapshot.get(key) != value for key, value in fixed_control.items()):
        raise PolicyConfigurationError(
            "exploratory control differs from the fixed non-empirical contract"
        )
    if config.monetary_contract is None:
        raise PolicyConfigurationError(
            "exploratory validation requires an explicit monetary contract"
        )

    profiles = _load_policy_profiles(config)
    profile_input_lineage = build_profile_input_lineage(
        profiles.country_profiles,
        profile_bundle=profiles,
    )
    if config.population is None:
        raise PolicyConfigurationError(
            "exploratory validation requires projected population mode"
        )
    population_adapter = resolve_population_projection_adapter(
        config.population,
        profiles,
        player_count=config.batch.player_count,
        campaign=False,
    )
    scientific_parent = _resolve_configured_analysis_plan(
        config,
        population_adapter=population_adapter,
        profile_input_lineage=profile_input_lineage,
    )
    if scientific_parent is None:
        raise PolicyConfigurationError(
            "exploratory validation requires the scientific parent plan"
        )
    selection = config.analysis_plan
    if (
        selection is None
        or selection.expected_plan_id is None
        or selection.expected_plan_sha256 is None
        or selection.parent_plan_path is None
        or selection.parent_plan_id is None
        or selection.parent_plan_sha256 is None
    ):
        raise PolicyConfigurationError(
            "exploratory scientific-parent selection identities are incomplete"
        )
    if (
        scientific_parent.plan.plan_id != selection.expected_plan_id
        or scientific_parent.plan.plan_sha256
        != selection.expected_plan_sha256
    ):
        raise PolicyConfigurationError(
            "re-attested v3 scientific parent differs from [analysis_plan]"
        )
    scientific_grandparent = verify_loaded_prospective_analysis_plan(
        load_prospective_analysis_plan(selection.parent_plan_path)
    )
    if (
        scientific_grandparent.plan.plan_id != selection.parent_plan_id
        or scientific_grandparent.plan.plan_sha256
        != selection.parent_plan_sha256
    ):
        raise PolicyConfigurationError(
            "re-attested v2 parent differs from [analysis_plan]"
        )
    amendment = scientific_parent.plan.amendment
    parent_binding = (
        amendment.get("parent_plan")
        if isinstance(amendment, dict)
        else None
    )
    if (
        not isinstance(parent_binding, dict)
        or parent_binding.get("plan_id")
        != scientific_grandparent.plan.plan_id
        or parent_binding.get("plan_sha256")
        != scientific_grandparent.plan.plan_sha256
        or parent_binding.get("file_sha256")
        != scientific_grandparent.file_sha256
    ):
        raise PolicyConfigurationError(
            "v3 scientific plan amendment differs from its exact v2 parent"
        )
    exploratory_plan = verify_loaded_exploratory_analysis_plan(
        load_exploratory_analysis_plan(control.exploratory_plan_path)
    )
    verify_exploratory_analysis_plan_parent(
        exploratory_plan.plan,
        scientific_parent,
    )
    if (
        exploratory_plan.plan.plan_id != control.exploratory_plan_id
        or exploratory_plan.plan.plan_sha256 != control.exploratory_plan_sha256
    ):
        raise PolicyConfigurationError(
            "exploratory sidecar plan differs from its configured identity"
        )
    if (
        exploratory_plan.plan.primary_estimand.estimand_id
        != control.primary_estimand_id
        or scientific_parent.plan.primary_estimand.estimand_id
        != control.primary_estimand_id
    ):
        raise PolicyConfigurationError(
            "exploratory and scientific-parent primary estimands differ"
        )
    if exploratory_plan.plan.stopping_rule.seeds != config.batch.seeds:
        raise PolicyConfigurationError(
            "exploratory sidecar fixed seeds differ from the configuration"
        )
    uncertainty_design = exploratory_plan.plan.identity_payload.get(
        "uncertainty_design"
    )
    convergence_rule = exploratory_plan.plan.identity_payload.get(
        "convergence_rule"
    )
    if not isinstance(uncertainty_design, dict) or not isinstance(
        convergence_rule, dict
    ):
        raise PolicyConfigurationError(
            "exploratory sidecar uncertainty/convergence declarations are malformed"
        )
    if config.uncertainty is None or config.convergence is None:
        raise PolicyConfigurationError(
            "exploratory validation requires uncertainty and convergence contracts"
        )
    uncertainty = config.uncertainty
    convergence = config.convergence
    parameter_design = verify_loaded_parameter_uncertainty_design(
        load_parameter_uncertainty_design(uncertainty.parameter_design_path)
    )
    config_convergence_snapshot = ConvergenceRule(
        block_size=convergence.block_size,
        minimum_retained_seeds=convergence.minimum_retained_seeds,
        maximum_mcse=convergence.maximum_mcse,
        maximum_interval_width=convergence.maximum_interval_width,
        maximum_absolute_change=convergence.maximum_absolute_change,
        maximum_relative_change=convergence.maximum_relative_change,
        maximum_invalid_rate=convergence.maximum_invalid_rate,
        consecutive_passing_checkpoints=(
            convergence.consecutive_passing_checkpoints
        ),
    ).snapshot()
    if config_convergence_snapshot != convergence_rule:
        raise PolicyConfigurationError(
            "exploratory sidecar convergence rule differs from configuration"
        )
    seed_uncertainty = uncertainty_design.get("seed_uncertainty")
    parameter_uncertainty = uncertainty_design.get("parameter_uncertainty")
    rate_uncertainty = uncertainty_design.get("monetary_rate_uncertainty")
    population_uncertainty = uncertainty_design.get("population_uncertainty")
    combined_uncertainty = uncertainty_design.get("combined_uncertainty")
    if not all(
        isinstance(item, dict)
        for item in (
            seed_uncertainty,
            parameter_uncertainty,
            rate_uncertainty,
            population_uncertainty,
            combined_uncertainty,
        )
    ):
        raise PolicyConfigurationError(
            "exploratory sidecar uncertainty components are incomplete"
        )
    if config.population_contract is None:
        raise PolicyConfigurationError(
            "exploratory validation requires the exact population contract"
        )
    expected_seed_uncertainty = {
        "common_random_numbers": uncertainty.common_random_numbers,
        "fixed_seed_count": len(config.batch.seeds),
        "identical_pretreatment_cohorts": (
            uncertainty.identical_pretreatment_cohorts
        ),
        "outcome_dependent_seed_exclusion_allowed": (
            uncertainty.outcome_dependent_seed_exclusion
        ),
        "population_weights_applied_within_seed": (
            uncertainty.population_weights_within_seed
        ),
        "status": "QUANTIFIED_WHEN_COMPLETE",
    }
    expected_parameter_uncertainty = {
        "design_id": parameter_design.design.design_id,
        "design_sha256": parameter_design.design.design_sha256,
        "method": parameter_design.design.method,
        "probability_interpretation": "NONE",
        "status": "ILLUSTRATIVE_DESIGN_ONLY",
    }
    expected_rate_uncertainty = {
        "point_observation_is_distribution": False,
        "rate_basis_sha256": config.monetary_contract.conversion_basis_sha256,
        "status": "UNQUANTIFIED",
    }
    expected_population_uncertainty = {
        "exact_weighting_is_empirical_validation": False,
        "status": "UNQUANTIFIED",
        "uncertainty_design_id": (
            config.population_contract.uncertainty_design_id
        ),
    }
    expected_combined_uncertainty = {
        "double_counting_control": (
            "one complete seed-parameter-population-rate Cartesian identity"
        ),
        "status": "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST",
        "variance_decomposition_method": (
            uncertainty.variance_decomposition_method
        ),
    }
    if (
        uncertainty.seed_design != "FIXED_ASCENDING"
        or uncertainty.minimum_retained_seeds != 100
        or len(config.batch.seeds) != 150
        or uncertainty.oat_role != "DIAGNOSTIC_ONLY"
        or uncertainty_design.get("schema_version") != "1.0"
        or uncertainty_design.get("oat_role") != uncertainty.oat_role
        or parameter_design.design.design_id != uncertainty.parameter_design_id
        or parameter_design.design.design_sha256
        != uncertainty.parameter_design_sha256
        or parameter_design.design.calibrated_probability_design
        or seed_uncertainty != expected_seed_uncertainty
        or parameter_uncertainty != expected_parameter_uncertainty
        or rate_uncertainty != expected_rate_uncertainty
        or population_uncertainty != expected_population_uncertainty
        or combined_uncertainty != expected_combined_uncertainty
        or uncertainty.parameter_uncertainty.value != "UNQUANTIFIED"
        or uncertainty.monetary_rate_uncertainty.value != "UNQUANTIFIED"
        or uncertainty.population_uncertainty.value != "UNQUANTIFIED"
        or not uncertainty.combined_uncertainty_required
        or config.monetary_contract.rate_uncertainty_status.value
        != "UNQUANTIFIED"
        or config.population_contract.uncertainty_status.value
        != "UNQUANTIFIED"
    ):
        raise PolicyConfigurationError(
            "exploratory sidecar uncertainty identities differ from configuration"
        )

    monetary = config.monetary_contract
    sidecar_monetary = exploratory_plan.plan.identity_payload.get(
        "monetary_semantics"
    )
    expected_sidecar_monetary = {
        "conversion_basis_sha256": monetary.conversion_basis_sha256,
        "conversion_before_population_weighting": True,
        "empirical_monetary_interpretation_allowed": False,
        "exact_rational_conversion_required": True,
        "missing_date_policy": monetary.missing_date_policy,
        "observed_real_world_spending_claimed": False,
        "price_period_end": monetary.target_price_period_end,
        "price_period_start": monetary.target_price_period_start,
        "quote_convention": monetary.quote_convention,
        "rate_period_end": monetary.rate_period_end,
        "rate_period_start": monetary.rate_period_start,
        "rate_uncertainty_status": monetary.rate_uncertainty_status.value,
        "raw_cross_currency_pooling_allowed": False,
        "rounding_boundary": monetary.rounding_scope,
        "rounding_rule": monetary.rounding_method,
        "scale_convention": monetary.scale_convention,
        "semantic_label": (
            "SIMULATED_MODEL_EQUIVALENT_TARGET_CURRENCY_VALUES"
        ),
        "simulation_bridge_status": monetary.simulation_bridge_status,
        "source_bundle_id": monetary.bundle_id,
        "source_bundle_semantic_sha256": monetary.rate_evidence_sha256,
        "source_bundle_signature_status": (
            monetary.source_bundle_signature_status
        ),
        "target_currency": monetary.target_currency,
    }
    if sidecar_monetary != expected_sidecar_monetary:
        raise PolicyConfigurationError(
            "exploratory sidecar monetary semantics differ from configuration"
        )
    rate_bundle, rate_results = load_and_verify_rate_evidence_bundle(
        monetary.source_bundle_path,
        required_source_registry_sha256=profiles.source_registry_sha256,
    )
    rate_evidence_sha256 = canonical_sha256(
        [result.snapshot() for result in rate_results]
    )
    if (
        rate_bundle.bundle_sha256 != monetary.source_bundle_sha256
        or rate_evidence_sha256 != monetary.rate_evidence_sha256
        or sha256(monetary.source_artifact_path.read_bytes()).hexdigest()
        != monetary.source_artifact_sha256
        or sha256(monetary.conversion_table_path.read_bytes()).hexdigest()
        != monetary.conversion_table_sha256
    ):
        raise PolicyConfigurationError(
            "exploratory monetary identities differ from re-attested evidence"
        )

    batch_snapshot = config.batch.snapshot()
    scenarios = batch_snapshot.get("scenarios")
    if not isinstance(scenarios, list):
        raise PolicyConfigurationError(
            "exploratory scenario snapshot is malformed"
        )
    production_after = {
        name: sha256(path.read_bytes()).hexdigest()
        for name, path in production_paths.items()
    }
    if production_before != production_after:
        raise PolicyConfigurationError(
            "production campaign inputs changed during exploratory validation"
        )

    return build_exploratory_validation_metadata(
        configuration_path=config_path.relative_to(repository_root).as_posix(),
        configuration_sha256=sha256(config_path.read_bytes()).hexdigest(),
        exploratory_plan_path=(
            exploratory_plan.plan_path.relative_to(repository_root).as_posix()
        ),
        exploratory_plan_id=exploratory_plan.plan.plan_id,
        exploratory_plan_sha256=exploratory_plan.plan.plan_sha256,
        exploratory_plan_file_sha256=exploratory_plan.file_sha256,
        scientific_parent_plan_path=(
            scientific_parent.plan_path.relative_to(repository_root).as_posix()
        ),
        scientific_parent_plan_id=scientific_parent.plan.plan_id,
        scientific_parent_plan_sha256=scientific_parent.plan.plan_sha256,
        scientific_parent_plan_file_sha256=scientific_parent.file_sha256,
        scientific_parent_registration_status=(
            scientific_parent.plan.registration_status.value
        ),
        primary_estimand_id=control.primary_estimand_id,
        population_adapter_id=population_adapter.adapter_id,
        population_adapter_sha256=population_adapter.adapter_sha256,
        population_execution_input_sha256=(
            population_execution_input_sha256(population_adapter)
        ),
        scenario_snapshots=scenarios,
        target_currency=monetary.target_currency,
        target_minor_unit_name=monetary.target_minor_unit_name,
        quote_convention=monetary.quote_convention,
        scale_convention=monetary.scale_convention,
        rate_period_start=monetary.rate_period_start,
        rate_period_end=monetary.rate_period_end,
        target_price_period_start=monetary.target_price_period_start,
        target_price_period_end=monetary.target_price_period_end,
        missing_date_policy=monetary.missing_date_policy,
        identity_missing_date_policy=monetary.identity_missing_date_policy,
        rounding_method=monetary.rounding_method,
        rounding_scope=monetary.rounding_scope,
        point_rate_status=monetary.point_rate_status,
        rate_uncertainty_status=monetary.rate_uncertainty_status.value,
        source_bundle_signature_status=(
            monetary.source_bundle_signature_status
        ),
        simulation_bridge_status=monetary.simulation_bridge_status,
        observed_real_world_spending=monetary.observed_real_world_spending,
        raw_cross_currency_pooling=monetary.raw_cross_currency_pooling,
        monetary_source_bundle_sha256=monetary.source_bundle_sha256,
        monetary_source_artifact_sha256=monetary.source_artifact_sha256,
        monetary_conversion_table_sha256=monetary.conversion_table_sha256,
        monetary_conversion_basis_sha256=monetary.conversion_basis_sha256,
        monetary_rate_evidence_sha256=rate_evidence_sha256,
        profile_input_lineage_sha256=(
            profile_input_lineage.fingerprint_sha256
        ),
        uncertainty_design=uncertainty_design,
        convergence_rule=convergence_rule,
        fixed_seed_count=len(config.batch.seeds),
        parameter_design_path=(
            parameter_design.design_path.relative_to(repository_root).as_posix()
        ),
        parameter_design_file_sha256=parameter_design.file_sha256,
        production_configuration_sha256=production_before["configuration"],
        production_plan_sha256=production_before["scientific_parent"],
    )


def _campaign_preflight(
    config_path: Path,
    *,
    output: Path | None,
) -> dict[str, object]:
    """Generate technical preflight evidence without executing realizations."""

    from .campaign_preflight import (
        ReceiptAttemptStatus,
        build_policy_campaign_execution_receipt_spec,
        build_policy_campaign_preflight_spec,
        run_pre_campaign_validation,
        write_pre_campaign_validation_report,
    )
    from .execution_attestation import (
        ExecutionVerificationPhase,
        build_execution_receipt,
        verify_execution_receipt,
        write_execution_receipt_atomic,
    )

    selected_config = config_path.resolve(strict=True)
    repository_root = _repository_root_for_path(selected_config)
    receipt_spec = build_policy_campaign_execution_receipt_spec(
        selected_config,
        repository_root=repository_root,
    )
    preflight_spec = build_policy_campaign_preflight_spec(
        selected_config,
        repository_root=repository_root,
        receipt_spec=receipt_spec,
    )
    report = run_pre_campaign_validation(preflight_spec)
    config = load_policy_config(selected_config)
    if config.execution_receipt is None:
        raise PolicyConfigurationError(
            "campaign-preflight requires [execution_receipt]"
        )
    report_path = (
        output.resolve()
        if output is not None
        else config.execution_receipt.receipt_path.parent
        / "pre-campaign-validation-report.json"
    )
    write_pre_campaign_validation_report(report_path, report)

    report_payload = report.identity_payload
    receipt_attempt = report_payload["execution_receipt"]
    receipt_path: str | None = None
    if (
        isinstance(receipt_attempt, dict)
        and receipt_attempt.get("status")
        == ReceiptAttemptStatus.GENERATED_AND_PREVERIFIED.value
    ):
        receipt = build_execution_receipt(receipt_spec)
        verify_execution_receipt(
            receipt,
            receipt_spec,
            phase=ExecutionVerificationPhase.PRE_EXECUTION,
        )
        written_receipt = write_execution_receipt_atomic(
            config.execution_receipt.receipt_path,
            receipt,
        )
        receipt_path = written_receipt.resolve().as_posix()

    return {
        "status": "PRE_CAMPAIGN_VALIDATION_COMPLETE_FAIL_CLOSED",
        "campaign_ready": False,
        "full_campaign_intentionally_not_run": True,
        "report_path": report_path.resolve().as_posix(),
        "report_sha256": report.report_sha256,
        "execution_receipt_path": receipt_path,
        "execution_receipt_sha256": (
            receipt_attempt.get("execution_receipt_sha256")
            if isinstance(receipt_attempt, dict)
            else None
        ),
        "passed_checks": report_payload["passed_checks"],
        "failed_checks": report_payload["failed_checks"],
        "unresolved_blockers": report_payload["unresolved_blockers"],
        "convergence_status": report_payload["convergence"]["status"],
    }


def _policy_batch(
    config_path: Path,
    *,
    output: Path | None,
    run_sensitivity: bool | None,
    command: Sequence[str],
) -> dict[str, object]:
    config = load_policy_config(config_path)
    campaign = config.run_purpose is PolicyRunPurpose.CAMPAIGN
    exploratory = config.run_purpose is PolicyRunPurpose.EXPLORATORY
    exploratory_validation_metadata = None
    if exploratory:
        from .outputs.exploratory import EXPLORATORY_LAUNCH_COMMAND

        repository_root = Path(__file__).resolve().parents[2]
        expected_config = (
            repository_root / "configs" / "policy_exploratory_synthetic.toml"
        ).resolve(strict=True)
        observed_config = config_path.resolve(strict=True)
        if observed_config != expected_config:
            raise CampaignExecutionRejectedError(
                "exploratory policy-batch is bound to "
                "configs/policy_exploratory_synthetic.toml"
            )
        if (
            len(command) != 3
            or tuple(command[:2]) != EXPLORATORY_LAUNCH_COMMAND[:2]
            or Path(command[2]).resolve(strict=True) != expected_config
        ):
            raise CampaignExecutionRejectedError(
                "exploratory model work may only use the reviewed "
                "microtx-sim policy-batch command"
            )
        if output is not None:
            raise CampaignExecutionRejectedError(
                "exploratory --output overrides are prohibited; the reviewed "
                "isolated artifact directory is fixed by configuration"
            )
        if run_sensitivity is not None:
            raise CampaignExecutionRejectedError(
                "exploratory sensitivity overrides are prohibited; retained "
                "sensitivity is fixed by configuration"
            )
        exploratory_validation_metadata = _exploratory_validate(config_path)
        if config.exploratory is None or not config.exploratory.execution_enabled:
            raise CampaignExecutionRejectedError(
                "exploratory execution is not explicitly enabled by the "
                "reviewed configuration"
            )
    execution_receipt_spec = None
    execution_receipt = None
    execution_pre_verification = None
    if campaign:
        (
            execution_receipt_spec,
            execution_receipt,
            execution_pre_verification,
        ) = (
            _preverify_campaign_execution(
                config_path,
                config=config,
                command=command,
            )
        )
    sensitivity_enabled = (
        config.output.run_sensitivity
        if run_sensitivity is None
        else run_sensitivity
    )
    profiles = _load_policy_profiles(config)
    profile_input_lineage = build_profile_input_lineage(
        profiles.country_profiles,
        profile_bundle=profiles,
    )
    population_adapter = (
        resolve_population_projection_adapter(
            config.population,
            profiles,
            player_count=config.batch.player_count,
            campaign=campaign,
        )
        if config.population is not None
        else None
    )
    if campaign:
        _validate_policy_campaign_provenance(config)
        profiles.validate_for_campaign()
    analysis_plan = _resolve_configured_analysis_plan(
        config,
        population_adapter=population_adapter,
        profile_input_lineage=profile_input_lineage,
    )
    repository_root = Path(__file__).resolve().parents[2]
    destination = _resolve_output(
        output if output is not None else config.output.output_dir,
        repository_root=repository_root,
    )
    checkpoint_recorder = None
    if exploratory:
        from .outputs.checkpoints import ExploratoryCheckpointRecorder
        from .outputs.exploratory_results import preflight_exploratory_output

        if config.exploratory is None or config.exploratory_checkpoint is None:
            raise PolicyConfigurationError(
                "exploratory execution requires explicit checkpoint policy"
            )
        preflight_exploratory_output(destination)
        checkpoint_recorder = ExploratoryCheckpointRecorder.start(
            config.exploratory_checkpoint.directory,
            expected_seeds=config.batch.seeds,
            config_sha256=sha256(config_path.read_bytes()).hexdigest(),
            exploratory_plan_id=config.exploratory.exploratory_plan_id,
            exploratory_plan_sha256=(
                config.exploratory.exploratory_plan_sha256
            ),
            launch_command=command,
        )
    try:
        batch = run_policy_batch(
            config.batch,
            profile_bundle=profiles,
            harm_parameters=config.harm_parameters,
            harm_weights=config.harm_weights,
            opportunity_valuation=config.opportunity_valuation,
            producer_assumptions=config.producer_assumptions,
            epgc_policy=config.epgc_policy,
            population_adapter=population_adapter,
            campaign=campaign,
            campaign_receipt=execution_receipt,
            campaign_verification=execution_pre_verification,
            checkpoint_callback=checkpoint_recorder,
        )
        if checkpoint_recorder is not None:
            checkpoint_recorder.mark_model_batch_complete()
        analysis_binding = (
            resolve_run_analysis_binding(analysis_plan.plan, batch)
            if analysis_plan is not None
            else None
        )
        sensitivity = (
            _run_configured_sensitivity(
                config,
                profile_bundle=profiles,
                population_adapter=population_adapter,
            )
            if sensitivity_enabled
            else None
        )
        execution_verification = None
        if campaign:
            # Rebuild the same identity after model work and before publication.
            from .execution_attestation import (
                ExecutionVerificationPhase,
                require_campaign_execution,
                verify_execution_receipt,
                write_execution_attestation_atomic,
            )

            assert execution_receipt_spec is not None
            assert execution_receipt is not None
            execution_verification = verify_execution_receipt(
                execution_receipt,
                execution_receipt_spec,
                phase=ExecutionVerificationPhase.POST_EXECUTION,
            )
            require_campaign_execution(execution_receipt, execution_verification)
            assert config.execution_receipt is not None
            write_execution_attestation_atomic(
                config.execution_receipt.attestation_path,
                execution_verification,
            )
        if exploratory:
            from .outputs.exploratory_results import export_exploratory_results

            if analysis_binding is None or checkpoint_recorder is None:
                raise RuntimeError(
                    "exploratory execution requires a weighted analysis binding "
                    "and checkpoint attempt"
                )
            assert exploratory_validation_metadata is not None
            paths = export_exploratory_results(
                config,
                batch,
                sensitivity,
                analysis_binding,
                config_path=config_path,
                output_dir=destination,
                command=command,
                exploratory_validation_metadata=(
                    exploratory_validation_metadata
                ),
                checkpoint_attempt_id=checkpoint_recorder.attempt_id,
            )
            checkpoint_recorder.mark_complete()
        else:
            paths = export_policy_batch(
                config,
                batch,
                sensitivity,
                config_path=config_path,
                repository_root=repository_root,
                output_dir=destination,
                command=command,
                analysis_plan=analysis_plan,
                analysis_binding=analysis_binding,
                execution_receipt=execution_receipt,
                execution_verification=execution_verification,
                exploratory_validation_metadata=(
                    exploratory_validation_metadata
                ),
            )
    except KeyboardInterrupt as error:
        if checkpoint_recorder is not None:
            try:
                checkpoint_recorder.mark_interrupted()
            except BaseException as checkpoint_error:
                error.add_note(
                    "the interrupted exploratory attempt could not update "
                    f"progress.json: {checkpoint_error}"
                )
        raise
    except BaseException as error:
        if checkpoint_recorder is not None:
            try:
                checkpoint_recorder.mark_failed(error)
            except BaseException as checkpoint_error:
                error.add_note(
                    "the exploratory failure marker also failed: "
                    f"{checkpoint_error}"
                )
        raise
    return {
        "status": "ok",
        "mode": (
            "campaign_policy_batch"
            if campaign
            else (
                "exploratory_policy_batch"
                if exploratory
                else "synthetic_policy_batch"
            )
        ),
        "run_purpose": config.run_purpose.value,
        "scenario": config.name,
        "scenario_count": len(config.batch.scenarios),
        "seeds": list(batch.spec.seeds),
        "seed_decimal_strings": [str(seed) for seed in batch.spec.seeds],
        "seed_count": len(config.batch.seeds),
        "player_count": config.batch.player_count,
        "days": config.batch.days,
        "sensitivity_run": sensitivity is not None,
        "unstable_parameters": (
            list(sensitivity.unstable_parameters) if sensitivity else []
        ),
        "output_dir": str(destination.resolve()),
        "artifacts": sorted(path.name for path in paths.values()),
        "empirical_validation_claimed": False,
        **(
            {"population_mode": config.population.mode.value}
            if config.population is not None
            else {}
        ),
        **(
            {
                "analysis_plan_sha256": analysis_plan.plan.plan_sha256,
                "analysis_binding_sha256": analysis_binding.binding_sha256,
                "campaign_ready": False,
            }
            if analysis_plan is not None and analysis_binding is not None
            else {}
        ),
    }


def _preverify_campaign_execution(
    config_path: Path,
    *,
    config,
    command: Sequence[str],
):
    """Create, persist, verify, and enforce the pre-run campaign receipt.

    This function is deliberately called before profile construction or the
    first realization.  The configured command is part of the identity and no
    CLI override is accepted for a full campaign.  Schema v1 always fails at
    ``require_campaign_execution`` after preserving the preflight receipt.
    """

    from .campaign_preflight import (
        build_policy_campaign_execution_receipt_spec,
    )
    from .execution_attestation import (
        ExecutionVerificationPhase,
        build_execution_receipt,
        require_campaign_execution,
        verify_execution_receipt,
        write_execution_receipt_atomic,
    )

    policy = config.execution_receipt
    if policy is None:
        raise PolicyConfigurationError(
            "campaign execution requires [execution_receipt]"
        )
    observed_command = tuple(command)
    if observed_command != policy.run_command:
        raise PolicyConfigurationError(
            "campaign command differs from execution_receipt.run_command"
        )
    selected_config = config_path.resolve(strict=True)
    repository_root = _repository_root_for_path(selected_config)
    receipt_spec = build_policy_campaign_execution_receipt_spec(
        selected_config,
        repository_root=repository_root,
    )
    if receipt_spec.run_command != observed_command:
        raise PolicyConfigurationError(
            "re-attested receipt command differs from the invoked campaign command"
        )
    receipt = build_execution_receipt(receipt_spec)
    verification = verify_execution_receipt(
        receipt,
        receipt_spec,
        phase=ExecutionVerificationPhase.PRE_EXECUTION,
    )
    write_execution_receipt_atomic(policy.receipt_path, receipt)
    require_campaign_execution(receipt, verification)
    return receipt_spec, receipt, verification


def _policy_sensitivity(
    config_path: Path,
    *,
    output: Path | None,
) -> dict[str, object]:
    config = load_policy_config(config_path)
    campaign = config.run_purpose is PolicyRunPurpose.CAMPAIGN
    if campaign:
        raise CampaignExecutionRejectedError(
            "standalone policy-sensitivity is not an attested campaign command; "
            "campaign sensitivity may only run inside the configured full "
            "campaign execution after its receipt gate opens"
        )
    if config.run_purpose is PolicyRunPurpose.EXPLORATORY:
        raise CampaignExecutionRejectedError(
            "standalone policy-sensitivity is outside the reviewed "
            "exploratory policy-batch command; no model work was dispatched"
        )
    profiles = _load_policy_profiles(config)
    profile_input_lineage = build_profile_input_lineage(
        profiles.country_profiles,
        profile_bundle=profiles,
    )
    population_adapter = (
        resolve_population_projection_adapter(
            config.population,
            profiles,
            player_count=config.batch.player_count,
            campaign=campaign,
        )
        if config.population is not None
        else None
    )
    if campaign:
        _validate_policy_campaign_provenance(config)
        profiles.validate_for_campaign()
    analysis_plan = _resolve_configured_analysis_plan(
        config,
        population_adapter=population_adapter,
        profile_input_lineage=profile_input_lineage,
    )
    result = _run_configured_sensitivity(
        config,
        profile_bundle=profiles,
        population_adapter=population_adapter,
    )
    if analysis_plan is not None:
        analysis_plan = verify_loaded_prospective_analysis_plan(analysis_plan)
    repository_root = Path(__file__).resolve().parents[2]
    destination = _resolve_output(
        output if output is not None else config.output.output_dir,
        repository_root=repository_root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv_atomic(
        destination / "sensitivity.csv",
        result.rows,
        canonical_columns=SENSITIVITY_COLUMNS,
        allow_extra_columns=False,
    )
    csv_sha256 = sha256(csv_path.read_bytes()).hexdigest()
    metadata_path = write_json_atomic(
        destination / "sensitivity_metadata.json",
        stamp_standalone_sensitivity_schema({
            "synthetic_only": True,
            "empirical_validation_claimed": False,
            "config": str(config_path.resolve()),
            "seeds": list(result.batch_spec.seeds),
            "seed_decimal_strings": [
                str(seed) for seed in result.batch_spec.seeds
            ],
            "execution_sha256": result.execution_sha256(),
            "execution_snapshot": result.execution_snapshot(),
            "run_inputs_sha256": result.run_inputs.snapshot_sha256(),
            "run_input_sha256": result.run_input_sha256(),
            "run_input_snapshot": result.run_input_snapshot(),
            "artifacts": {
                "sensitivity.csv": {
                    "row_count": len(result.rows),
                    "sha256": csv_sha256,
                }
            },
            "unstable_parameters": list(result.unstable_parameters),
            "profile_inputs": (
                result.profile_input_lineage.manifest_payload()
                if result.profile_input_lineage is not None
                else None
            ),
            **(
                {
                    "analysis_plan": analysis_plan.manifest_payload(),
                    "analysis_plan_scope": (
                        "Expected execution-input identities validated before "
                        "the sensitivity run; this standalone profile does not "
                        "bind treatment-result estimands."
                    ),
                }
                if analysis_plan is not None
                else {}
            ),
        }),
    )
    return {
        "status": "ok",
        "mode": "synthetic_sensitivity",
        "rows": len(result.rows),
        "unstable_parameters": list(result.unstable_parameters),
        "artifacts": [str(csv_path.resolve()), str(metadata_path.resolve())],
        **(
            {
                "analysis_plan_sha256": analysis_plan.plan.plan_sha256,
                "campaign_ready": False,
            }
            if analysis_plan is not None
            else {}
        ),
    }


def _resolve_configured_analysis_plan(
    config,
    *,
    population_adapter,
    profile_input_lineage: ProfileInputLineage | None,
) -> LoadedProspectiveAnalysisPlan | None:
    """Load, re-attest, and preflight an opt-in plan before execution."""

    selection = config.analysis_plan
    if selection is None:
        return None
    if population_adapter is None:
        raise ValueError(
            "analysis plan execution requires a projected population adapter"
        )
    if profile_input_lineage is None:
        raise ValueError(
            "analysis plan execution requires exact profile input lineage"
        )
    loaded = load_prospective_analysis_plan(selection.plan_path)
    loaded = verify_loaded_prospective_analysis_plan(loaded)
    run_inputs = resolve_policy_run_inputs(
        harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
    )
    validate_analysis_plan_inputs(
        loaded.plan,
        batch_spec=config.batch,
        run_inputs=run_inputs,
        population_adapter=population_adapter,
        profile_input_lineage=profile_input_lineage,
    )
    return loaded


def _validate_policy_campaign_provenance(config) -> None:
    if config.provenance_status != "calibrated":
        raise PolicyConfigurationError(
            "campaign policy execution requires provenance_status = 'calibrated'; "
            f"got {config.provenance_status!r}"
        )


def _load_policy_profiles(config) -> ProfileBundle:
    """Load the exact profile/evidence selection declared for a policy run."""

    population = config.population
    if population is None or population.evidence_bundle_path is None:
        return load_profile_bundle(campaign=False)
    if population.source_registry_path is None:
        raise ValueError(
            "configured population evidence requires a source registry path"
        )
    return load_profile_bundle(
        jurisdictions_path=(
            config.monetary_contract.profile_path
            if config.monetary_contract is not None
            else DEFAULT_JURISDICTIONS_PATH
        ),
        sources_path=population.source_registry_path,
        source_bundle_path=(
            config.monetary_contract.source_bundle_path
            if config.monetary_contract is not None
            else None
        ),
        population_bundle_path=population.evidence_bundle_path,
        campaign=False,
    )


def _run_configured_sensitivity(
    config,
    *,
    profile_bundle: ProfileBundle | None = None,
    population_adapter=None,
):
    return run_sensitivity_analysis(
        config.batch,
        profile_bundle=profile_bundle,
        base_harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
        population_adapter=population_adapter,
    )


def _resolve_output(path: Path, *, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _repository_root_for_path(path: Path) -> Path:
    selected = path.resolve(strict=True)
    for candidate in (selected.parent, *selected.parents):
        if (candidate / ".git").exists():
            return candidate.resolve(strict=True)
    raise PolicyConfigurationError(
        f"configuration is not contained in a Git worktree: {selected}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            payload = _validate(args.config)
        elif args.command == "smoke":
            payload = _smoke(args.config)
        elif args.command == "policy-validate":
            payload = _policy_validate(args.config)
        elif args.command == "exploratory-validate":
            payload = _exploratory_validate(args.config)
        elif args.command == "campaign-preflight":
            payload = _campaign_preflight(args.config, output=args.output)
        elif args.command == "policy-sensitivity":
            payload = _policy_sensitivity(args.config, output=args.output)
        elif args.command in {"policy-batch", "reproduce"}:
            requested = list(argv) if argv is not None else sys.argv[1:]
            payload = _policy_batch(
                args.config,
                output=args.output,
                run_sensitivity=(
                    True
                    if args.command == "reproduce"
                    else False
                    if args.skip_sensitivity
                    else None
                ),
                command=("microtx-sim", *requested),
            )
        else:
            raise AssertionError(args.command)
    except (
        ConfigurationError,
        PolicyConfigurationError,
        ProfileValidationError,
        LedgerStorageError,
        CampaignExecutionRejectedError,
        sqlite3.DatabaseError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
