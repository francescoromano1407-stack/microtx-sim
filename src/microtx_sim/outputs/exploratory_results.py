"""Dedicated non-empirical result profile for the exploratory campaign."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Final, Mapping, Sequence

from ..analysis.sensitivity import SensitivityResult
from ..analysis.uncertainty import (
    ConvergenceRule,
    RealizationIdentity,
    UncertaintyAvailability,
    UncertaintyComponentStatus,
    UncertaintyRealization,
    decompose_joint_uncertainty,
    evaluate_blockwise_convergence,
    final_sufficiency_judgment,
    summarize_seed_uncertainty,
)
from ..causal.analysis_binding import RunAnalysisBinding
from ..causal.batch import PolicyBatchCheckpoint, PolicyBatchResult
from ..execution.checkpoints import (
    CheckpointStatus,
    ResumableCheckpointStore,
)
from ..execution.optimized_runner import (
    CheckpointedPolicyBatch,
    CheckpointedSensitivityResult,
)
from ..execution.streaming_analysis import (
    CheckpointedRunAnalysisBinding,
    CompactSeedAnalysisBinding,
)
from ..policy_config import PolicyPrototypeConfig, PolicyRunPurpose
from .checkpoints import EXPLORATORY_CHECKPOINT_COLUMNS
from .exploratory import EXPLORATORY_INTERPRETATION_WORDING
from .uncertainty import write_joint_uncertainty_outputs
from .writers import write_csv_atomic, write_json_atomic, write_text_atomic


EXPLORATORY_RESULT_SCHEMA_VERSION: Final[str] = "1.0"
EXPLORATORY_RESULT_OUTPUT_PROFILE: Final[str] = (
    "exploratory_synthetic_weighted_primary_and_diagnostics"
)
EXPLORATORY_RESULT_ARTIFACTS: Final[tuple[str, ...]] = (
    "weighted_primary_estimand.csv",
    "scenario_diagnostics.csv",
    "sensitivity_diagnostics.csv",
    "uncertainty_realizations.csv",
    "convergence_checkpoints.csv",
    "uncertainty_summary.json",
    "nonempirical_metadata.json",
    "summary.md",
    "manifest.json",
)
EXPLORATORY_FINALIZATION_ATTESTATION_SCHEMA: Final[str] = (
    "microtx_sim.exploratory_finalization_attestation.v1"
)
WEIGHTED_PRIMARY_COLUMNS: Final[tuple[str, ...]] = (
    "seed",
    "primary_estimand_id",
    "reference_scenario_id",
    "comparison_scenario_id",
    "contrast_direction",
    "metric_name",
    "unit",
    "weighted_estimate",
    "exact_numerator_decimal",
    "exact_denominator_decimal",
    "selected_player_count",
    "population_weights_sha256",
    "population_seed_record_sha256",
    "pretreatment_cohort_sha256",
    "result_sha256",
    "binding_sha256",
    "interpretation",
)
SENSITIVITY_DIAGNOSTIC_COLUMNS: Final[tuple[str, ...]] = (
    "parameter",
    "parameter_value",
    "scenario_id",
    "seed_count",
    "mean_harm",
    "harm_variance",
    "harm_sd",
    "harm_ci95_low",
    "harm_ci95_high",
    "harm_coefficient_of_variation",
    "opportunity_cost_burden",
    "expected_direction",
    "monotonic_expected",
    "monotonic_observed",
    "unstable",
    "interpretation",
)


def _profile_descriptor() -> dict[str, object]:
    return {
        "schema_version": EXPLORATORY_RESULT_SCHEMA_VERSION,
        "output_profile": EXPLORATORY_RESULT_OUTPUT_PROFILE,
        "artifact_files": list(EXPLORATORY_RESULT_ARTIFACTS),
        "weighted_primary_columns": list(WEIGHTED_PRIMARY_COLUMNS),
        "scenario_diagnostic_columns": list(EXPLORATORY_CHECKPOINT_COLUMNS),
        "sensitivity_diagnostic_columns": list(
            SENSITIVITY_DIAGNOSTIC_COLUMNS
        ),
        "monetary_output_policy": {
            "raw_simulation_cents_exposed": False,
            "raw_cross_country_pooling_allowed": False,
            "observed_spending_claim_allowed": False,
        },
        "campaign_ready": False,
    }


EXPLORATORY_RESULT_SCHEMA_SHA256: Final[str] = sha256(
    json.dumps(
        _profile_descriptor(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def export_exploratory_results(
    config: PolicyPrototypeConfig,
    batch: PolicyBatchResult,
    sensitivity: SensitivityResult | None,
    analysis_binding: RunAnalysisBinding,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    command: Sequence[str],
    exploratory_validation_metadata: Mapping[str, object],
    checkpoint_attempt_id: str,
) -> dict[str, Path]:
    """Publish a strict exploratory bundle with no raw monetary aggregates."""

    if config.run_purpose is not PolicyRunPurpose.EXPLORATORY:
        raise ValueError("exploratory result export requires exploratory config")
    if config.exploratory is None or config.monetary_contract is None:
        raise ValueError("exploratory result export requires complete contracts")
    if batch.spec != config.batch or analysis_binding.seeds != batch.spec.seeds:
        raise ValueError("exploratory result identities differ from configuration")
    if not isinstance(exploratory_validation_metadata, Mapping):
        raise TypeError("exploratory validation metadata must be a mapping")
    config_file = Path(config_path)
    repository_root = config_file.resolve().parent.parent
    expected_destination = (
        repository_root / config.output.output_dir
    ).resolve()
    if Path(output_dir).resolve() != expected_destination:
        raise ValueError(
            "exploratory result output must use the configured isolated "
            "artifact namespace"
        )
    destination = preflight_exploratory_output(expected_destination)

    primary_bindings = tuple(
        item
        for item in analysis_binding.seed_bindings
        if item.planned_estimand.estimand_id
        == config.exploratory.primary_estimand_id
    )
    if tuple(item.seed for item in primary_bindings) != batch.spec.seeds:
        raise ValueError(
            "primary exploratory binding does not cover the fixed seed set"
        )
    weighted_rows = [
        {
            "seed": item.seed,
            "primary_estimand_id": item.planned_estimand.estimand_id,
            "reference_scenario_id": (
                item.planned_estimand.reference_scenario_id.value
            ),
            "comparison_scenario_id": (
                item.planned_estimand.comparison_scenario_id.value
            ),
            "contrast_direction": item.planned_estimand.contrast_direction,
            "metric_name": item.result.metric_name,
            "unit": item.planned_estimand.outcome_semantics.unit,
            "weighted_estimate": item.result.value,
            "exact_numerator_decimal": str(item.result.numerator),
            "exact_denominator_decimal": str(item.result.denominator),
            "selected_player_count": item.selected_player_count,
            "population_weights_sha256": item.selected_weights.design_sha256,
            "population_seed_record_sha256": (
                item.population_seed_record_sha256
            ),
            "pretreatment_cohort_sha256": (
                batch.cohort_digest_by_seed[item.seed]
            ),
            "result_sha256": item.result.result_sha256,
            "binding_sha256": item.binding_sha256,
            "interpretation": "CONDITIONAL_ON_MODEL_ASSUMPTIONS_NOT_EMPIRICAL",
        }
        for item in primary_bindings
    ]
    full_checkpoint = PolicyBatchCheckpoint(
        spec=batch.spec,
        completed_seeds=batch.spec.seeds,
        records=batch.records,
        cohort_digest_by_seed=batch.cohort_digest_by_seed,
    )
    scenario_rows = full_checkpoint.nonmonetary_diagnostic_rows()
    sensitivity_rows = _sensitivity_rows(sensitivity)

    config_sha256 = _file_sha256(config_file)
    realizations = _primary_realizations(
        config,
        batch,
        primary_bindings,
    )
    seed_summary = summarize_seed_uncertainty(
        realizations,
        expected_seeds=batch.spec.seeds,
    )
    decomposition = decompose_joint_uncertainty(realizations)
    components = (
        UncertaintyComponentStatus(
            source="seed",
            availability=UncertaintyAvailability.QUANTIFIED,
            variance=seed_summary.sample_standard_deviation**2,
            method="SAMPLE_VARIANCE_ACROSS_FIXED_SEEDS_DDOF_1",
            blocker=None,
        ),
        UncertaintyComponentStatus(
            source="parameter",
            availability=UncertaintyAvailability.UNQUANTIFIED,
            variance=None,
            method=None,
            blocker="illustrative ranges are not calibrated distributions",
        ),
        UncertaintyComponentStatus(
            source="monetary_rate",
            availability=UncertaintyAvailability.UNQUANTIFIED,
            variance=None,
            method=None,
            blocker="official point rate has no admissible uncertainty model",
        ),
        UncertaintyComponentStatus(
            source="population",
            availability=UncertaintyAvailability.UNQUANTIFIED,
            variance=None,
            method=None,
            blocker="no admissible population uncertainty design",
        ),
        UncertaintyComponentStatus(
            source="combined",
            availability=UncertaintyAvailability.UNAVAILABLE,
            variance=None,
            method=None,
            blocker="required uncertainty components are unavailable",
        ),
    )
    rule = _convergence_rule(config)
    instability = bool(sensitivity and sensitivity.unstable_parameters)
    checkpoints = evaluate_blockwise_convergence(
        realizations,
        expected_seeds=batch.spec.seeds,
        rule=rule,
        sensitivity_instability=instability,
        required_components_available=False,
    )
    judgment = final_sufficiency_judgment(
        convergence_status=checkpoints[-1].status,
        components=components,
    )

    paths = {
        "weighted_primary_estimand": write_csv_atomic(
            destination / "weighted_primary_estimand.csv",
            weighted_rows,
            canonical_columns=WEIGHTED_PRIMARY_COLUMNS,
            allow_extra_columns=False,
        ),
        "scenario_diagnostics": write_csv_atomic(
            destination / "scenario_diagnostics.csv",
            scenario_rows,
            canonical_columns=EXPLORATORY_CHECKPOINT_COLUMNS,
            allow_extra_columns=False,
        ),
        "sensitivity_diagnostics": write_csv_atomic(
            destination / "sensitivity_diagnostics.csv",
            sensitivity_rows,
            canonical_columns=SENSITIVITY_DIAGNOSTIC_COLUMNS,
            allow_extra_columns=False,
        ),
    }
    paths.update(
        write_joint_uncertainty_outputs(
            destination,
            realizations=realizations,
            primary_seed_realizations=realizations,
            primary_seed_summary=seed_summary,
            components=components,
            variance_decomposition=decomposition,
            convergence_checkpoints=checkpoints,
            sufficiency_judgment=judgment,
            expected_seeds=batch.spec.seeds,
            convergence_rule=rule,
            plan_id=config.exploratory.exploratory_plan_id,
            plan_sha256=config.exploratory.exploratory_plan_sha256,
            config_sha256=config_sha256,
            sensitivity_instability=instability,
            output_profile=EXPLORATORY_RESULT_OUTPUT_PROFILE,
            output_profile_schema_version=EXPLORATORY_RESULT_SCHEMA_VERSION,
            output_profile_schema_sha256=EXPLORATORY_RESULT_SCHEMA_SHA256,
        )
    )
    nonempirical = {
        **_profile_descriptor(),
        "output_profile_schema_sha256": EXPLORATORY_RESULT_SCHEMA_SHA256,
        "interpretation_wording": EXPLORATORY_INTERPRETATION_WORDING,
        "monetary_interpretation": {
            "internal_unit": "simulation_cents",
            "raw_internal_units_published": False,
            "raw_cross_country_pooling": "REJECTED",
            "official_fx_point_observation_retained_in_contract": True,
            "internal_to_real_money_bridge": "ILLUSTRATIVE_NOT_CALIBRATED",
            "allowed_final_label": "MODEL_EQUIVALENT_AMOUNT",
            "observed_real_world_spending": False,
        },
        "population_interpretation": {
            "basis": "ILLUSTRATIVE_NON_EMPIRICAL",
            "exact_weights_used_for_primary_estimand": True,
            "unweighted_outputs_role": "DIAGNOSTIC_ONLY",
            "representativeness_claim": False,
        },
        "checkpoint_attempt_id": checkpoint_attempt_id,
        "campaign_ready": False,
    }
    paths["nonempirical_metadata"] = write_json_atomic(
        destination / "nonempirical_metadata.json",
        nonempirical,
    )
    paths["summary"] = write_text_atomic(
        destination / "summary.md",
        _summary(seed_summary.snapshot(), checkpoints[-1].snapshot(), judgment),
    )
    manifest = {
        **_profile_descriptor(),
        "output_profile_schema_sha256": EXPLORATORY_RESULT_SCHEMA_SHA256,
        "status": "EXPLORATORY_EXECUTION_COMPLETE_SCIENTIFICALLY_INSUFFICIENT",
        "configuration": {
            "path": config_file.resolve().as_posix(),
            "sha256": config_sha256,
        },
        "exploratory_plan": {
            "plan_id": config.exploratory.exploratory_plan_id,
            "plan_sha256": config.exploratory.exploratory_plan_sha256,
        },
        "scientific_parent_plan": {
            "plan_id": analysis_binding.plan.plan_id,
            "plan_sha256": analysis_binding.plan.plan_sha256,
        },
        "analysis_binding_sha256": analysis_binding.binding_sha256,
        "population_lineage_sha256": (
            batch.population_execution_lineage.lineage_sha256
            if batch.population_execution_lineage is not None
            else None
        ),
        "model_inputs_sha256": batch.run_input_sha256(),
        "command": list(command),
        "checkpoint_attempt_id": checkpoint_attempt_id,
        "preexecution_validation_identity_sha256": _canonical_sha256(
            exploratory_validation_metadata
        ),
        "final_sufficiency_judgment": judgment,
        "convergence_status": checkpoints[-1].status.value,
        "campaign_ready": False,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in paths.values()
        },
    }
    paths["manifest"] = write_json_atomic(
        destination / "manifest.json",
        manifest,
    )
    actual = {path.name for path in paths.values()}
    if actual != set(EXPLORATORY_RESULT_ARTIFACTS):
        raise RuntimeError(
            "exploratory result artifact set differs from its strict profile"
        )
    return paths


def export_checkpointed_exploratory_results(
    config: PolicyPrototypeConfig,
    batch: CheckpointedPolicyBatch,
    sensitivity: CheckpointedSensitivityResult | None,
    analysis_binding: CheckpointedRunAnalysisBinding,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    command: Sequence[str],
    exploratory_validation_metadata: Mapping[str, object],
) -> dict[str, Path]:
    """Finalize the resumable exploratory run without retaining all arrays.

    Every final artifact is first written below the immutable attempt namespace.
    Publication starts only after that complete staged bundle validates.  A crash
    during publication can therefore copy only the missing byte-identical files
    on the next launch and never needs to repeat completed model work.
    """

    _validate_checkpointed_export_inputs(
        config,
        batch,
        sensitivity,
        analysis_binding,
        config_path=config_path,
        output_dir=output_dir,
        exploratory_validation_metadata=exploratory_validation_metadata,
    )
    config_file = Path(config_path).resolve(strict=True)
    destination = Path(output_dir).resolve()
    store = batch.store
    resumed = resume_staged_exploratory_finalization(store, destination)
    if resumed is not None:
        return resumed

    primary_bindings = tuple(
        item
        for item in analysis_binding.seed_bindings
        if item.planned_estimand.estimand_id
        == config.exploratory.primary_estimand_id
    )
    assert config.exploratory is not None
    weighted_rows = [_compact_weighted_row(item) for item in primary_bindings]
    scenario_rows = list(analysis_binding.scenario_diagnostic_rows)
    sensitivity_rows = _sensitivity_rows(sensitivity)
    realizations = _checkpointed_primary_realizations(
        config,
        batch,
        primary_bindings,
    )
    seed_summary = summarize_seed_uncertainty(
        realizations,
        expected_seeds=batch.spec.seeds,
    )
    decomposition = decompose_joint_uncertainty(realizations)
    components = _exploratory_uncertainty_components(seed_summary)
    rule = _convergence_rule(config)
    instability = bool(sensitivity and sensitivity.unstable_parameters)
    checkpoints = evaluate_blockwise_convergence(
        realizations,
        expected_seeds=batch.spec.seeds,
        rule=rule,
        sensitivity_instability=instability,
        required_components_available=False,
    )
    judgment = final_sufficiency_judgment(
        convergence_status=checkpoints[-1].status,
        components=components,
    )
    end_to_end_parity = _end_to_end_backend_parity(
        config,
        batch,
        sensitivity,
        primary_bindings,
        observed_checkpoints=checkpoints,
        observed_judgment=judgment,
        convergence_rule=rule,
    )
    config_sha256 = _file_sha256(config_file)
    stage = _finalization_stage(store)
    stage.mkdir(parents=True, exist_ok=True)
    paths = {
        "weighted_primary_estimand": write_csv_atomic(
            stage / "weighted_primary_estimand.csv",
            weighted_rows,
            canonical_columns=WEIGHTED_PRIMARY_COLUMNS,
            allow_extra_columns=False,
        ),
        "scenario_diagnostics": write_csv_atomic(
            stage / "scenario_diagnostics.csv",
            scenario_rows,
            canonical_columns=EXPLORATORY_CHECKPOINT_COLUMNS,
            allow_extra_columns=False,
        ),
        "sensitivity_diagnostics": write_csv_atomic(
            stage / "sensitivity_diagnostics.csv",
            sensitivity_rows,
            canonical_columns=SENSITIVITY_DIAGNOSTIC_COLUMNS,
            allow_extra_columns=False,
        ),
    }
    paths.update(
        write_joint_uncertainty_outputs(
            stage,
            realizations=realizations,
            primary_seed_realizations=realizations,
            primary_seed_summary=seed_summary,
            components=components,
            variance_decomposition=decomposition,
            convergence_checkpoints=checkpoints,
            sufficiency_judgment=judgment,
            expected_seeds=batch.spec.seeds,
            convergence_rule=rule,
            plan_id=config.exploratory.exploratory_plan_id,
            plan_sha256=config.exploratory.exploratory_plan_sha256,
            config_sha256=config_sha256,
            sensitivity_instability=instability,
            output_profile=EXPLORATORY_RESULT_OUTPUT_PROFILE,
            output_profile_schema_version=EXPLORATORY_RESULT_SCHEMA_VERSION,
            output_profile_schema_sha256=EXPLORATORY_RESULT_SCHEMA_SHA256,
        )
    )
    execution_metadata = _checkpointed_execution_metadata(
        config,
        batch,
        analysis_binding,
        end_to_end_parity=end_to_end_parity,
    )
    nonempirical = {
        **_profile_descriptor(),
        "output_profile_schema_sha256": EXPLORATORY_RESULT_SCHEMA_SHA256,
        "interpretation_wording": EXPLORATORY_INTERPRETATION_WORDING,
        "monetary_interpretation": {
            "internal_unit": "simulation_cents",
            "raw_internal_units_published": False,
            "raw_cross_country_pooling": "REJECTED",
            "official_fx_point_observation_retained_in_contract": True,
            "internal_to_real_money_bridge": "ILLUSTRATIVE_NOT_CALIBRATED",
            "allowed_final_label": "MODEL_EQUIVALENT_AMOUNT",
            "observed_real_world_spending": False,
        },
        "population_interpretation": {
            "basis": "ILLUSTRATIVE_NON_EMPIRICAL",
            "exact_weights_used_for_primary_estimand": True,
            "unweighted_outputs_role": "DIAGNOSTIC_ONLY",
            "representativeness_claim": False,
        },
        "execution": execution_metadata,
        "checkpoint_attempt_id": store.identity.attempt_id,
        "campaign_ready": False,
    }
    paths["nonempirical_metadata"] = write_json_atomic(
        stage / "nonempirical_metadata.json",
        nonempirical,
    )
    paths["summary"] = write_text_atomic(
        stage / "summary.md",
        _summary(seed_summary.snapshot(), checkpoints[-1].snapshot(), judgment),
    )
    manifest = {
        **_profile_descriptor(),
        "output_profile_schema_sha256": EXPLORATORY_RESULT_SCHEMA_SHA256,
        "status": "EXPLORATORY_EXECUTION_COMPLETE_SCIENTIFICALLY_INSUFFICIENT",
        "configuration": {
            "path": config_file.as_posix(),
            "sha256": config_sha256,
        },
        "exploratory_plan": {
            "plan_id": config.exploratory.exploratory_plan_id,
            "plan_sha256": config.exploratory.exploratory_plan_sha256,
        },
        "scientific_parent_plan": {
            "plan_id": analysis_binding.plan.plan_id,
            "plan_sha256": analysis_binding.plan.plan_sha256,
        },
        "analysis_binding_sha256": analysis_binding.binding_sha256,
        "population_lineage_sha256": (
            analysis_binding.population_lineage_sha256
        ),
        "model_inputs_sha256": analysis_binding.model_inputs_sha256,
        "policy_run_input_sha256": analysis_binding.policy_run_input_sha256,
        "command": list(command),
        "execution": execution_metadata,
        "checkpoint_attempt_id": store.identity.attempt_id,
        "preexecution_validation_identity_sha256": _canonical_sha256(
            exploratory_validation_metadata
        ),
        "final_sufficiency_judgment": judgment,
        "convergence_status": checkpoints[-1].status.value,
        "unstable_parameters": (
            list(sensitivity.unstable_parameters) if sensitivity else []
        ),
        "campaign_ready": False,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in paths.values()
        },
    }
    paths["manifest"] = write_json_atomic(stage / "manifest.json", manifest)
    _require_complete_staged_bundle(stage)
    attest_staged_exploratory_finalization(store, stage)
    published = _publish_staged_bundle(stage, destination)
    output_checksums = {
        path.name: _file_sha256(path) for path in published.values()
    }
    store.mark_complete(output_checksums)
    return published


def resume_staged_exploratory_finalization(
    store: ResumableCheckpointStore,
    output_dir: str | Path,
) -> dict[str, Path] | None:
    """Finish or verify a previously staged publication, if one exists."""

    if type(store) is not ResumableCheckpointStore:
        raise TypeError("store must be ResumableCheckpointStore")
    destination = Path(output_dir).resolve()
    stage = _finalization_stage(store)
    attestation = _finalization_attestation_path(store)
    staged_names = _staged_artifact_names(stage)
    final_names = {
        name for name in EXPLORATORY_RESULT_ARTIFACTS
        if (destination / name).is_file()
    }
    if not staged_names:
        if final_names or attestation.exists() or attestation.is_symlink():
            raise FileExistsError(
                "exploratory final artifacts exist without the attested "
                "attempt staging bundle"
            )
        return None
    expected = set(EXPLORATORY_RESULT_ARTIFACTS)
    if staged_names != expected:
        if final_names or attestation.exists() or attestation.is_symlink():
            raise FileExistsError(
                "partial final artifacts cannot be recovered from an "
                "incomplete staging bundle"
            )
        return None
    if not attestation.is_file() or attestation.is_symlink():
        if final_names:
            raise FileExistsError(
                "published exploratory artifacts lack a valid finalization "
                "attestation"
            )
        return None
    _require_complete_staged_bundle(stage)
    _verify_finalization_attestation(store, stage)
    published = _publish_staged_bundle(stage, destination)
    checksums = {path.name: _file_sha256(path) for path in published.values()}
    if store.status is not CheckpointStatus.COMPLETE:
        store.mark_complete(checksums)
    else:
        declared = {
            str(item["path"]): str(item["sha256"])
            for item in store.progress_snapshot["output_paths"]["final_outputs"]
        }
        if declared != checksums:
            raise ValueError(
                "completed checkpoint final-output checksums changed"
            )
    return published


def preflight_exploratory_output(output_dir: str | Path) -> Path:
    """Reject stale final outputs before any model work is dispatched."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    existing = [
        name
        for name in EXPLORATORY_RESULT_ARTIFACTS
        if (destination / name).exists()
    ]
    if existing:
        raise FileExistsError(
            "exploratory final artifacts already exist and will not be "
            f"overwritten: {', '.join(existing)}"
        )
    return destination


def _validate_checkpointed_export_inputs(
    config: PolicyPrototypeConfig,
    batch: CheckpointedPolicyBatch,
    sensitivity: CheckpointedSensitivityResult | None,
    analysis_binding: CheckpointedRunAnalysisBinding,
    *,
    config_path: str | Path,
    output_dir: str | Path,
    exploratory_validation_metadata: Mapping[str, object],
) -> None:
    if config.run_purpose is not PolicyRunPurpose.EXPLORATORY:
        raise ValueError("checkpointed export requires exploratory config")
    if (
        config.exploratory is None
        or config.monetary_contract is None
        or config.execution_engine is None
    ):
        raise ValueError("checkpointed export requires complete contracts")
    if type(batch) is not CheckpointedPolicyBatch:
        raise TypeError("batch must be CheckpointedPolicyBatch")
    if type(analysis_binding) is not CheckpointedRunAnalysisBinding:
        raise TypeError(
            "analysis_binding must be CheckpointedRunAnalysisBinding"
        )
    if sensitivity is not None and type(sensitivity) is not CheckpointedSensitivityResult:
        raise TypeError(
            "sensitivity must be CheckpointedSensitivityResult or None"
        )
    if batch.spec != config.batch or analysis_binding.seeds != batch.spec.seeds:
        raise ValueError("checkpointed result identities differ from configuration")
    if tuple(batch.store.work_plan.seeds) != batch.spec.seeds:
        raise ValueError("checkpoint work-plan seeds differ from configuration")
    if batch.store.remaining_main_seeds:
        raise ValueError("main checkpoint work remains incomplete")
    if batch.store.remaining_sensitivity_units:
        raise ValueError("sensitivity checkpoint work remains incomplete")
    sensitivity_declared = bool(batch.store.work_plan.sensitivity_units)
    if sensitivity_declared != (sensitivity is not None):
        raise ValueError("sensitivity result differs from the declared work plan")
    if analysis_binding.backend_identity_sha256 != (
        batch.store.identity.backend.identity_sha256
    ):
        raise ValueError("analysis binding backend identity changed")
    if batch.store.identity.analysis_plan_sha256 != (
        config.exploratory.exploratory_plan_sha256
    ):
        raise ValueError("checkpoint exploratory plan identity changed")
    config_file = Path(config_path).resolve(strict=True)
    if _file_sha256(config_file) != batch.store.identity.configuration_sha256:
        raise ValueError("configuration bytes changed after checkpoint creation")
    repository_root = config_file.parent.parent
    expected_destination = (repository_root / config.output.output_dir).resolve()
    if Path(output_dir).resolve() != expected_destination:
        raise ValueError(
            "checkpointed exploratory output must use the configured isolated "
            "artifact namespace"
        )
    if not isinstance(exploratory_validation_metadata, Mapping):
        raise TypeError("exploratory validation metadata must be a mapping")
    primary = tuple(
        item
        for item in analysis_binding.seed_bindings
        if item.planned_estimand.estimand_id
        == config.exploratory.primary_estimand_id
    )
    if tuple(item.seed for item in primary) != batch.spec.seeds:
        raise ValueError(
            "primary checkpointed binding does not cover the fixed seed set"
        )


def _compact_weighted_row(
    item: CompactSeedAnalysisBinding,
) -> dict[str, object]:
    return {
        "seed": item.seed,
        "primary_estimand_id": item.planned_estimand.estimand_id,
        "reference_scenario_id": (
            item.planned_estimand.reference_scenario_id.value
        ),
        "comparison_scenario_id": (
            item.planned_estimand.comparison_scenario_id.value
        ),
        "contrast_direction": item.planned_estimand.contrast_direction,
        "metric_name": item.result.metric_name,
        "unit": item.planned_estimand.outcome_semantics.unit,
        "weighted_estimate": item.result.value,
        "exact_numerator_decimal": str(item.result.numerator),
        "exact_denominator_decimal": str(item.result.denominator),
        "selected_player_count": item.selected_player_count,
        "population_weights_sha256": item.population_weights_sha256,
        "population_seed_record_sha256": item.population_seed_record_sha256,
        "pretreatment_cohort_sha256": item.pretreatment_cohort_sha256,
        "result_sha256": item.result.result_sha256,
        "binding_sha256": item.binding_sha256,
        "interpretation": "CONDITIONAL_ON_MODEL_ASSUMPTIONS_NOT_EMPIRICAL",
    }


def _checkpointed_primary_realizations(
    config: PolicyPrototypeConfig,
    batch: CheckpointedPolicyBatch,
    bindings: Sequence[CompactSeedAnalysisBinding],
    *,
    cpu_reference: bool = False,
) -> tuple[UncertaintyRealization, ...]:
    assert config.exploratory is not None
    assert config.population_contract is not None
    assert config.monetary_contract is not None
    parameter_sha256 = batch.run_inputs.snapshot_sha256()
    scenario_id = "baseline_f2p-minus-safe_fixed_price_subscription"
    return tuple(
        UncertaintyRealization(
            identity=RealizationIdentity(
                seed=item.seed,
                parameter_draw_id="fixed_configured_model_parameters_v1",
                parameter_draw_sha256=parameter_sha256,
                population_design_id=config.population_contract.design_id,
                population_replicate_id="fixed_exact_projected_design_v1",
                population_design_sha256=(
                    config.population_contract.design_sha256
                ),
                monetary_rate_draw_id="official_point_observation_v1",
                monetary_rate_basis_id=(
                    config.monetary_contract.conversion_basis_id
                ),
                monetary_rate_basis_sha256=(
                    config.monetary_contract.conversion_basis_sha256
                ),
                scenario_id=scenario_id,
                primary_estimand_id=config.exploratory.primary_estimand_id,
                pretreatment_cohort_sha256=item.pretreatment_cohort_sha256,
                population_weights_sha256=item.population_weights_sha256,
            ),
            estimate=(
                item.cpu_reference_result.value
                if cpu_reference
                else item.result.value
            ),
            valid=True,
        )
        for item in bindings
    )


def _exploratory_uncertainty_components(seed_summary):
    return (
        UncertaintyComponentStatus(
            source="seed",
            availability=UncertaintyAvailability.QUANTIFIED,
            variance=seed_summary.sample_standard_deviation**2,
            method="SAMPLE_VARIANCE_ACROSS_FIXED_SEEDS_DDOF_1",
            blocker=None,
        ),
        UncertaintyComponentStatus(
            source="parameter",
            availability=UncertaintyAvailability.UNQUANTIFIED,
            variance=None,
            method=None,
            blocker="illustrative ranges are not calibrated distributions",
        ),
        UncertaintyComponentStatus(
            source="monetary_rate",
            availability=UncertaintyAvailability.UNQUANTIFIED,
            variance=None,
            method=None,
            blocker="official point rate has no admissible uncertainty model",
        ),
        UncertaintyComponentStatus(
            source="population",
            availability=UncertaintyAvailability.UNQUANTIFIED,
            variance=None,
            method=None,
            blocker="no admissible population uncertainty design",
        ),
        UncertaintyComponentStatus(
            source="combined",
            availability=UncertaintyAvailability.UNAVAILABLE,
            variance=None,
            method=None,
            blocker="required uncertainty components are unavailable",
        ),
    )


def _end_to_end_backend_parity(
    config: PolicyPrototypeConfig,
    batch: CheckpointedPolicyBatch,
    sensitivity: CheckpointedSensitivityResult | None,
    bindings: Sequence[CompactSeedAnalysisBinding],
    *,
    observed_checkpoints: Sequence[object],
    observed_judgment: Mapping[str, object],
    convergence_rule: ConvergenceRule,
) -> dict[str, object]:
    """Require GPU arithmetic to preserve final scientific decisions."""

    reference_realizations = _checkpointed_primary_realizations(
        config,
        batch,
        bindings,
        cpu_reference=True,
    )
    reference_summary = summarize_seed_uncertainty(
        reference_realizations,
        expected_seeds=batch.spec.seeds,
    )
    reference_components = _exploratory_uncertainty_components(
        reference_summary
    )
    reference_instability = bool(
        sensitivity and sensitivity.cpu_reference_unstable_parameters
    )
    reference_checkpoints = evaluate_blockwise_convergence(
        reference_realizations,
        expected_seeds=batch.spec.seeds,
        rule=convergence_rule,
        sensitivity_instability=reference_instability,
        required_components_available=False,
    )
    reference_judgment = final_sufficiency_judgment(
        convergence_status=reference_checkpoints[-1].status,
        components=reference_components,
    )
    observed_statuses = [item.status.value for item in observed_checkpoints]
    reference_statuses = [
        item.status.value for item in reference_checkpoints
    ]
    observed_blockers = [list(item.blockers) for item in observed_checkpoints]
    reference_blockers = [
        list(item.blockers) for item in reference_checkpoints
    ]
    values_bitwise_equal = all(
        item.result.result_sha256 == item.cpu_reference_result.result_sha256
        for item in bindings
    )
    within_tolerance = all(
        item.continuous_parity_within_tolerance for item in bindings
    )
    direction_equal = all(
        item.estimand_direction_matches_cpu_reference for item in bindings
    )
    sensitivity_equal = bool(
        sensitivity is None or sensitivity.conclusions_match_cpu_reference
    )
    convergence_equal = (
        observed_statuses == reference_statuses
        and observed_blockers == reference_blockers
    )
    judgment_equal = dict(observed_judgment) == dict(reference_judgment)
    maximum_difference = max(
        (item.continuous_absolute_difference for item in bindings),
        default=0.0,
    )
    backend_mode = batch.store.identity.backend.resolved_backend
    passed = bool(
        within_tolerance
        and direction_equal
        and sensitivity_equal
        and convergence_equal
        and judgment_equal
        and (backend_mode == "gpu" or values_bitwise_equal)
    )
    payload = {
        "reference_backend": "cpu",
        "observed_backend": backend_mode,
        "continuous_absolute_tolerance": (
            5e-13 if backend_mode == "gpu" else 0.0
        ),
        "continuous_relative_tolerance": (
            5e-13 if backend_mode == "gpu" else 0.0
        ),
        "primary_seed_values_bitwise_equal": values_bitwise_equal,
        "primary_seed_values_within_tolerance": within_tolerance,
        "primary_estimand_directions_equal": direction_equal,
        "maximum_absolute_primary_estimand_difference": maximum_difference,
        "sensitivity_conclusions_equal": sensitivity_equal,
        "sensitivity_maximum_absolute_harm_difference": (
            sensitivity.maximum_absolute_harm_difference
            if sensitivity is not None
            else 0.0
        ),
        "observed_convergence_statuses": observed_statuses,
        "cpu_reference_convergence_statuses": reference_statuses,
        "observed_convergence_blockers": observed_blockers,
        "cpu_reference_convergence_blockers": reference_blockers,
        "convergence_decisions_equal": convergence_equal,
        "observed_final_sufficiency_judgment": dict(observed_judgment),
        "cpu_reference_final_sufficiency_judgment": dict(
            reference_judgment
        ),
        "declared_conclusions_equal": judgment_equal,
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            "execution backend changes a primary, sensitivity, convergence, "
            "or final-sufficiency conclusion relative to CPU"
        )
    return payload


def _checkpointed_execution_metadata(
    config: PolicyPrototypeConfig,
    batch: CheckpointedPolicyBatch,
    analysis_binding: CheckpointedRunAnalysisBinding,
    *,
    end_to_end_parity: Mapping[str, object],
) -> dict[str, object]:
    engine = config.execution_engine
    assert engine is not None
    store = batch.store
    progress = store.progress_snapshot
    return {
        "implementation_id": store.identity.implementation_id,
        "run_id": store.identity.run_id,
        "attempt_id": store.identity.attempt_id,
        "execution_identity_sha256": store.identity.identity_sha256,
        "execution_identity": store.identity.snapshot(),
        "source_tree_sha256": store.identity.source_tree_sha256,
        "git_commit": store.identity.git_commit,
        "git_branch": store.identity.git_branch,
        "runtime_identity_sha256": store.identity.runtime.identity_sha256,
        "runtime_identity": store.identity.runtime.snapshot(),
        "backend_identity_sha256": store.identity.backend.identity_sha256,
        "backend_contract": store.identity.backend.snapshot(),
        "backend_runtime": dict(batch.backend_metadata),
        "backend_parity": dict(batch.backend_parity),
        "backend_end_to_end_parity": dict(end_to_end_parity),
        "continuous_result_tolerance": (
            5e-13
            if store.identity.backend.resolved_backend == "gpu"
            else 0.0
        ),
        "scheduler": {
            "host_executor": engine.host_executor,
            "process_start_method": "spawn",
            "configured_worker_count": engine.host_workers,
            "effective_worker_count": store.identity.backend.worker_count,
            "native_threads_per_worker": engine.native_threads_per_worker,
            "maximum_in_flight_units": engine.max_in_flight_units,
            "memory_limit_mb": engine.memory_limit_mb,
            "estimated_worker_memory_mb": engine.estimated_worker_memory_mb,
            "scheduling_policy": engine.scheduling_policy,
            "checkpoint_writer": "COORDINATOR_ONLY",
            "commit_order": "DECLARED_WORK_PLAN_ORDER",
            "gpu_batch_size": engine.gpu_batch_size,
            "gpu_max_batch_bytes": engine.gpu_max_batch_bytes,
        },
        "ledger_contract": (
            {
                "backend": config.ledger.backend.value,
                "path": config.ledger.path.as_posix(),
                "persistent": config.ledger.persistent,
                "temporary": config.ledger.temporary,
                "execution_role": (
                    "DECLARED_PERSISTENT_CONFIG_CONTRACT; "
                    "RESUMABLE_RESULTS_ARE_AUTHORITATIVE_CHECKPOINT_BLOCKS"
                ),
            }
            if config.ledger is not None
            else None
        ),
        "checkpoint": {
            "schema_id": progress["schema_id"],
            "schema_version": progress["schema_version"],
            "attempt_directory": store.attempt_dir.as_posix(),
            "checkpoint_path": store.checkpoint_path.as_posix(),
            "progress_path": store.progress_path.as_posix(),
            "finalization_attestation_path": (
                _finalization_attestation_path(store).as_posix()
            ),
            "finalization_attestation_schema": (
                EXPLORATORY_FINALIZATION_ATTESTATION_SCHEMA
            ),
            "checkpoint_identity_sha256": store.checkpoint_sha256,
            "checkpoint_file_sha256_at_finalization": (
                store.checkpoint_file_sha256
            ),
            "work_plan_sha256": store.work_plan.identity_sha256,
            "work_plan": store.work_plan.snapshot(),
            "progress_at_finalization": progress,
            "single_coordinator_lease": {
                "path": (
                    store.attempt_dir.parent
                    / f".{store.identity.attempt_id}.lock"
                ).as_posix(),
                "mechanism": "NONBLOCKING_OS_BYTE_RANGE_LOCK",
                "crash_release": True,
            },
        },
        "lineage": (
            {
                "identity_sha256": store.lineage.identity_sha256,
                "payload": store.lineage.snapshot(),
            }
            if store.lineage is not None
            else None
        ),
        "analysis_binding_sha256": analysis_binding.binding_sha256,
        "campaign_ready": False,
    }


def _finalization_stage(store: ResumableCheckpointStore) -> Path:
    return store.attempt_dir / "finalization-staging"


def _finalization_attestation_path(store: ResumableCheckpointStore) -> Path:
    return store.attempt_dir / "finalization_attestation.json"


def attest_staged_exploratory_finalization(
    store: ResumableCheckpointStore,
    stage: str | Path,
) -> Path:
    """Atomically bind a complete staged set to the execution identity."""

    if type(store) is not ResumableCheckpointStore:
        raise TypeError("store must be ResumableCheckpointStore")
    selected = Path(stage)
    if selected.resolve() != _finalization_stage(store).resolve():
        raise ValueError("finalization staging directory differs from attempt")
    _require_complete_staged_bundle(selected)
    core = _finalization_attestation_core(store, selected)
    payload = {
        **core,
        "attestation_sha256": _canonical_sha256(core),
    }
    path = _finalization_attestation_path(store)
    if path.is_symlink():
        raise ValueError("finalization attestation cannot be a symlink")
    if path.exists():
        _verify_finalization_attestation(store, selected)
        return path
    write_json_atomic(path, payload)
    _verify_finalization_attestation(store, selected)
    return path


def _finalization_attestation_core(
    store: ResumableCheckpointStore,
    stage: Path,
) -> dict[str, object]:
    _validate_staged_manifest(store, stage)
    return {
        "schema_id": EXPLORATORY_FINALIZATION_ATTESTATION_SCHEMA,
        "schema_version": "1.0",
        "execution_identity_sha256": store.identity.identity_sha256,
        "run_id": store.identity.run_id,
        "attempt_id": store.identity.attempt_id,
        "configuration_sha256": store.identity.configuration_sha256,
        "analysis_plan_sha256": store.identity.analysis_plan_sha256,
        "work_plan_sha256": store.work_plan.identity_sha256,
        "artifact_set": {
            name: {
                "bytes": (stage / name).stat().st_size,
                "sha256": _file_sha256(stage / name),
            }
            for name in sorted(EXPLORATORY_RESULT_ARTIFACTS)
        },
        "campaign_ready": False,
    }


def _verify_finalization_attestation(
    store: ResumableCheckpointStore,
    stage: Path,
) -> None:
    path = _finalization_attestation_path(store)
    if path.is_symlink() or not path.is_file():
        raise ValueError("finalization attestation is missing or not regular")
    try:
        observed = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("finalization attestation is unreadable") from exc
    if type(observed) is not dict:
        raise ValueError("finalization attestation must be an object")
    claimed = observed.get("attestation_sha256")
    core = {
        key: value
        for key, value in observed.items()
        if key != "attestation_sha256"
    }
    if type(claimed) is not str or claimed != _canonical_sha256(core):
        raise ValueError("finalization attestation identity hash mismatch")
    expected = _finalization_attestation_core(store, stage)
    if _canonical_sha256(core) != _canonical_sha256(expected):
        raise ValueError(
            "staged exploratory artifacts differ from their finalization "
            "attestation"
        )


def _validate_staged_manifest(
    store: ResumableCheckpointStore,
    stage: Path,
) -> None:
    try:
        manifest = json.loads((stage / "manifest.json").read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("staged exploratory manifest is unreadable") from exc
    if type(manifest) is not dict:
        raise ValueError("staged exploratory manifest must be an object")
    execution = manifest.get("execution")
    configuration = manifest.get("configuration")
    plan = manifest.get("exploratory_plan")
    if not all(type(item) is dict for item in (execution, configuration, plan)):
        raise ValueError("staged exploratory manifest identities are malformed")
    expected = {
        "execution_identity_sha256": store.identity.identity_sha256,
        "run_id": store.identity.run_id,
        "attempt_id": store.identity.attempt_id,
        "configuration_sha256": store.identity.configuration_sha256,
        "analysis_plan_sha256": store.identity.analysis_plan_sha256,
        "checkpoint_attempt_id": store.identity.attempt_id,
        "campaign_ready": False,
    }
    observed = {
        "execution_identity_sha256": execution.get(
            "execution_identity_sha256"
        ),
        "run_id": execution.get("run_id"),
        "attempt_id": execution.get("attempt_id"),
        "configuration_sha256": configuration.get("sha256"),
        "analysis_plan_sha256": plan.get("plan_sha256"),
        "checkpoint_attempt_id": manifest.get("checkpoint_attempt_id"),
        "campaign_ready": manifest.get("campaign_ready"),
    }
    if observed != expected:
        raise ValueError("staged exploratory manifest identity mismatch")
    unstable = manifest.get("unstable_parameters")
    if (
        type(unstable) is not list
        or any(type(item) is not str for item in unstable)
    ):
        raise ValueError(
            "staged exploratory manifest has invalid unstable parameters"
        )
    if manifest.get("status") != (
        "EXPLORATORY_EXECUTION_COMPLETE_SCIENTIFICALLY_INSUFFICIENT"
    ):
        raise ValueError("staged exploratory manifest status is invalid")
    artifacts = manifest.get("artifacts")
    expected_names = set(EXPLORATORY_RESULT_ARTIFACTS) - {"manifest.json"}
    if type(artifacts) is not dict or set(artifacts) != expected_names:
        raise ValueError("staged manifest artifact inventory is incomplete")
    for name in expected_names:
        item = artifacts[name]
        if type(item) is not dict or item != {
            "bytes": (stage / name).stat().st_size,
            "sha256": _file_sha256(stage / name),
        }:
            raise ValueError(
                f"staged manifest artifact identity mismatch: {name}"
            )


def _require_complete_staged_bundle(stage: Path) -> None:
    expected = set(EXPLORATORY_RESULT_ARTIFACTS)
    observed = _staged_artifact_names(stage)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            "exploratory staging bundle differs from its strict profile: "
            f"missing={missing}, extra={extra}"
        )
    for name in expected:
        path = stage / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"invalid staged exploratory artifact: {name}")


def _staged_artifact_names(stage: Path) -> set[str]:
    if stage.is_symlink():
        raise RuntimeError("finalization staging path cannot be a symlink")
    if not stage.exists():
        return set()
    if not stage.is_dir():
        raise RuntimeError("finalization staging path is not a real directory")
    expected = set(EXPLORATORY_RESULT_ARTIFACTS)
    observed: set[str] = set()
    temporary_patterns = tuple(
        re.compile(rf"^\.{re.escape(name)}\..+\.tmp$") for name in expected
    )
    for path in stage.iterdir():
        if path.name in expected:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(
                    f"invalid staged exploratory artifact: {path.name}"
                )
            observed.add(path.name)
            continue
        if (
            not path.is_symlink()
            and path.is_file()
            and any(pattern.fullmatch(path.name) for pattern in temporary_patterns)
        ):
            # A process death may leave an unreferenced atomic-writer temp.
            # It is ignored for regeneration and can never be published.
            continue
        raise RuntimeError(
            f"unexpected entry in exploratory staging: {path.name}"
        )
    return observed


def _publish_staged_bundle(
    stage: Path,
    destination: Path,
) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    published: dict[str, Path] = {}
    for name in EXPLORATORY_RESULT_ARTIFACTS:
        source = stage / name
        target = destination / name
        source_sha256 = _file_sha256(source)
        if target.exists():
            if target.is_symlink() or not target.is_file():
                raise FileExistsError(
                    f"exploratory output target is not a regular file: {name}"
                )
            if _file_sha256(target) != source_sha256:
                raise FileExistsError(
                    "exploratory final artifact differs from the attested "
                    f"staging copy: {name}"
                )
        else:
            _copy_file_atomic(source, target)
            if _file_sha256(target) != source_sha256:
                raise OSError(f"published exploratory checksum mismatch: {name}")
        published[name.rsplit(".", 1)[0]] = target
    return published


def _copy_file_atomic(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        try:
            # Same-volume hard-link creation is atomic and refuses to replace a
            # concurrently appearing final artifact.  The temporary name is
            # removed only after the destination directory entry exists.
            os.link(temporary, destination)
        except FileExistsError:
            if _file_sha256(destination) != _file_sha256(source):
                raise FileExistsError(
                    "exploratory output appeared during publication: "
                    f"{destination.name}"
                )
        finally:
            temporary.unlink(missing_ok=True)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _primary_realizations(config, batch, bindings) -> tuple[UncertaintyRealization, ...]:
    assert config.exploratory is not None
    assert config.population_contract is not None
    assert config.monetary_contract is not None
    parameter_sha256 = batch.run_input_sha256()
    scenario_id = "baseline_f2p-minus-safe_fixed_price_subscription"
    return tuple(
        UncertaintyRealization(
            identity=RealizationIdentity(
                seed=item.seed,
                parameter_draw_id="fixed_configured_model_parameters_v1",
                parameter_draw_sha256=parameter_sha256,
                population_design_id=config.population_contract.design_id,
                population_replicate_id="fixed_exact_projected_design_v1",
                population_design_sha256=(
                    config.population_contract.design_sha256
                ),
                monetary_rate_draw_id="official_point_observation_v1",
                monetary_rate_basis_id=(
                    config.monetary_contract.conversion_basis_id
                ),
                monetary_rate_basis_sha256=(
                    config.monetary_contract.conversion_basis_sha256
                ),
                scenario_id=scenario_id,
                primary_estimand_id=config.exploratory.primary_estimand_id,
                pretreatment_cohort_sha256=(
                    batch.cohort_digest_by_seed[item.seed]
                ),
                population_weights_sha256=(
                    item.selected_weights.design_sha256
                ),
            ),
            estimate=item.result.value,
            valid=True,
        )
        for item in bindings
    )


def _convergence_rule(config: PolicyPrototypeConfig) -> ConvergenceRule:
    if config.convergence is None:
        raise ValueError("exploratory convergence configuration is required")
    item = config.convergence
    return ConvergenceRule(
        block_size=item.block_size,
        minimum_retained_seeds=item.minimum_retained_seeds,
        maximum_mcse=item.maximum_mcse,
        maximum_interval_width=item.maximum_interval_width,
        maximum_absolute_change=item.maximum_absolute_change,
        maximum_relative_change=item.maximum_relative_change,
        maximum_invalid_rate=item.maximum_invalid_rate,
        consecutive_passing_checkpoints=item.consecutive_passing_checkpoints,
    )


def _sensitivity_rows(
    sensitivity: SensitivityResult | None,
) -> list[dict[str, object]]:
    if sensitivity is None:
        return []
    return [
        {
            key: row[key]
            for key in SENSITIVITY_DIAGNOSTIC_COLUMNS
            if key != "interpretation"
        }
        | {"interpretation": "OAT_MODEL_DIAGNOSTIC_ONLY"}
        for row in sensitivity.rows
    ]


def _summary(seed_summary, checkpoint, judgment) -> str:
    return "\n".join(
        (
            "# Exploratory synthetic model results",
            "",
            f"> {EXPLORATORY_INTERPRETATION_WORDING}",
            "",
            "## Primary weighted model estimand",
            "",
            f"Point estimate: {seed_summary['point_estimate']:.10g}",
            f"Monte Carlo standard error: {seed_summary['monte_carlo_standard_error']:.10g}",
            f"Monte Carlo interval: [{seed_summary['interval_lower']:.10g}, {seed_summary['interval_upper']:.10g}]",
            "",
            "## Sufficiency",
            "",
            f"Convergence status: {checkpoint['status']}",
            f"Sufficient: {str(judgment['sufficient']).lower()}",
            f"Blockers: {', '.join(judgment['blockers'])}",
            "",
            "Raw simulation_cents are not published or pooled across countries in this output profile.",
        )
    ) + "\n"


def _canonical_sha256(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "EXPLORATORY_FINALIZATION_ATTESTATION_SCHEMA",
    "EXPLORATORY_RESULT_ARTIFACTS",
    "EXPLORATORY_RESULT_OUTPUT_PROFILE",
    "EXPLORATORY_RESULT_SCHEMA_SHA256",
    "EXPLORATORY_RESULT_SCHEMA_VERSION",
    "SENSITIVITY_DIAGNOSTIC_COLUMNS",
    "WEIGHTED_PRIMARY_COLUMNS",
    "attest_staged_exploratory_finalization",
    "export_checkpointed_exploratory_results",
    "export_exploratory_results",
    "preflight_exploratory_output",
    "resume_staged_exploratory_finalization",
]
