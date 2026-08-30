"""Dedicated non-empirical result profile for the exploratory campaign."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
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
    "EXPLORATORY_RESULT_ARTIFACTS",
    "EXPLORATORY_RESULT_OUTPUT_PROFILE",
    "EXPLORATORY_RESULT_SCHEMA_SHA256",
    "EXPLORATORY_RESULT_SCHEMA_VERSION",
    "SENSITIVITY_DIAGNOSTIC_COLUMNS",
    "WEIGHTED_PRIMARY_COLUMNS",
    "export_exploratory_results",
    "preflight_exploratory_output",
]
