"""Production-shaped uncertainty and convergence output writer."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..analysis.uncertainty import (
    ConvergenceCheckpoint,
    ConvergenceRule,
    ConvergenceStatus,
    SeedUncertaintySummary,
    UncertaintyAvailability,
    UncertaintyComponentStatus,
    UncertaintyRealization,
    VarianceDecomposition,
    decompose_joint_uncertainty,
    evaluate_blockwise_convergence,
    final_sufficiency_judgment,
    summarize_seed_uncertainty,
)
from .schema import (
    CAMPAIGN_ANALYSIS_OUTPUT_PROFILE,
    CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
    CAMPAIGN_ANALYSIS_SCHEMA_VERSION,
    CONVERGENCE_CHECKPOINT_COLUMNS,
    UNCERTAINTY_REALIZATION_COLUMNS,
)
from .writers import write_csv_atomic, write_json_atomic


def _realization_row(item: UncertaintyRealization) -> dict[str, object]:
    identity = item.identity
    return {
        "seed": identity.seed,
        "parameter_draw_id": identity.parameter_draw_id,
        "parameter_draw_sha256": identity.parameter_draw_sha256,
        "population_design_id": identity.population_design_id,
        "population_replicate_id": identity.population_replicate_id,
        "population_design_sha256": identity.population_design_sha256,
        "monetary_rate_draw_id": identity.monetary_rate_draw_id,
        "monetary_rate_basis_id": identity.monetary_rate_basis_id,
        "monetary_rate_basis_sha256": identity.monetary_rate_basis_sha256,
        "scenario_id": identity.scenario_id,
        "primary_estimand_id": identity.primary_estimand_id,
        "pretreatment_cohort_sha256": identity.pretreatment_cohort_sha256,
        "population_weights_sha256": identity.population_weights_sha256,
        "identity_sha256": identity.identity_sha256,
        "estimate": item.estimate if item.estimate is not None else "",
        "valid": item.valid,
        "invalid_reason": item.invalid_reason or "",
    }


def _checkpoint_row(item: ConvergenceCheckpoint) -> dict[str, object]:
    payload = item.snapshot()
    payload["blockers"] = "|".join(item.blockers)
    for key, value in tuple(payload.items()):
        if value is None:
            payload[key] = ""
    return payload


def write_joint_uncertainty_outputs(
    output_dir: str | Path,
    *,
    realizations: Sequence[UncertaintyRealization],
    primary_seed_realizations: Sequence[UncertaintyRealization],
    primary_seed_summary: SeedUncertaintySummary | None,
    components: Sequence[UncertaintyComponentStatus],
    variance_decomposition: VarianceDecomposition,
    convergence_checkpoints: Sequence[ConvergenceCheckpoint],
    sufficiency_judgment: Mapping[str, object],
    expected_seeds: Sequence[int],
    convergence_rule: ConvergenceRule,
    plan_id: str,
    plan_sha256: str,
    config_sha256: str,
    rejected_count: int = 0,
    excluded_count: int = 0,
    sensitivity_instability: bool | Mapping[int, bool] = False,
    output_profile: str = CAMPAIGN_ANALYSIS_OUTPUT_PROFILE,
    output_profile_schema_version: str = CAMPAIGN_ANALYSIS_SCHEMA_VERSION,
    output_profile_schema_sha256: str = CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
) -> dict[str, Path]:
    """Write mutually consistent uncertainty artifacts without readiness promotion.

    Joint realizations and the fixed-input seed series are separate arguments:
    a combined Cartesian design contains repeated seed IDs, while convergence
    is defined only for one declared fixed-input seed series.  Every supplied
    summary is independently recomputed before any file is created.
    """

    joint_rows = tuple(realizations)
    seed_rows = tuple(primary_seed_realizations)
    component_rows = tuple(components)
    checkpoint_rows_input = tuple(convergence_checkpoints)
    if any(type(item) is not UncertaintyRealization for item in joint_rows):
        raise TypeError("realizations must contain UncertaintyRealization values")
    if any(type(item) is not UncertaintyRealization for item in seed_rows):
        raise TypeError(
            "primary_seed_realizations must contain UncertaintyRealization values"
        )
    joint_by_identity = {
        item.identity.identity_sha256: item for item in joint_rows
    }
    if len(joint_by_identity) != len(joint_rows):
        raise ValueError("joint realization identities must be unique")
    if any(
        joint_by_identity.get(item.identity.identity_sha256) != item
        for item in seed_rows
    ):
        raise ValueError(
            "primary seed realizations must be present in the joint realization set"
        )
    if type(variance_decomposition) is not VarianceDecomposition:
        raise TypeError("variance_decomposition must be VarianceDecomposition")
    VarianceDecomposition.__post_init__(variance_decomposition)
    observed_decomposition = decompose_joint_uncertainty(joint_rows)
    if variance_decomposition != observed_decomposition:
        raise ValueError(
            "variance decomposition differs from the declared joint realizations"
        )
    if type(convergence_rule) is not ConvergenceRule:
        raise TypeError("convergence_rule must be ConvergenceRule")
    ConvergenceRule.__post_init__(convergence_rule)
    if any(type(item) is not UncertaintyComponentStatus for item in component_rows):
        raise TypeError(
            "components must contain UncertaintyComponentStatus values"
        )
    for item in component_rows:
        UncertaintyComponentStatus.__post_init__(item)
    component_payload = {item.source: item.snapshot() for item in component_rows}
    if len(component_payload) != len(component_rows):
        raise ValueError("uncertainty component sources must be unique")
    required_sources = {
        "seed",
        "parameter",
        "monetary_rate",
        "population",
        "combined",
    }
    if set(component_payload) != required_sources:
        raise ValueError(
            "campaign uncertainty outputs require exactly seed, parameter, "
            "monetary_rate, population, and combined component statuses"
        )
    expected = tuple(expected_seeds)
    complete_valid_seed_series = (
        len(seed_rows) == len(expected)
        and tuple(item.identity.seed for item in seed_rows) == expected
        and all(item.valid for item in seed_rows)
    )
    if complete_valid_seed_series:
        observed_seed_summary = summarize_seed_uncertainty(
            seed_rows,
            expected_seeds=expected,
        )
        if primary_seed_summary != observed_seed_summary:
            raise ValueError(
                "primary seed summary differs from the fixed-input seed realizations"
            )
    elif primary_seed_summary is not None:
        raise ValueError(
            "a seed summary requires a complete valid declared fixed-seed series"
        )
    seed_component = next(
        item for item in component_rows if item.source == "seed"
    )
    if primary_seed_summary is None:
        if seed_component.availability is UncertaintyAvailability.QUANTIFIED:
            raise ValueError(
                "seed uncertainty cannot be quantified without a complete seed summary"
            )
    else:
        SeedUncertaintySummary.__post_init__(primary_seed_summary)
        expected_seed_variance = primary_seed_summary.sample_standard_deviation**2
        if (
            seed_component.availability is not UncertaintyAvailability.QUANTIFIED
            or seed_component.variance is None
            or not np.isclose(
                seed_component.variance,
                expected_seed_variance,
                rtol=1e-12,
                atol=1e-15,
            )
        ):
            raise ValueError(
                "quantified seed component must equal the seed-summary sample variance"
            )
    _validate_component_against_decomposition(
        next(item for item in component_rows if item.source == "parameter"),
        variance_decomposition.between_parameter_variance,
        name="parameter",
    )
    _validate_component_against_decomposition(
        next(item for item in component_rows if item.source == "population"),
        variance_decomposition.between_population_variance,
        name="population",
    )
    _validate_component_against_decomposition(
        next(item for item in component_rows if item.source == "monetary_rate"),
        variance_decomposition.between_rate_variance,
        name="monetary_rate",
    )
    combined_component = next(
        item for item in component_rows if item.source == "combined"
    )
    if combined_component.availability is UncertaintyAvailability.QUANTIFIED:
        if (
            not variance_decomposition.identifiable
            or variance_decomposition.total_joint_variance is None
            or combined_component.variance is None
            or not np.isclose(
                combined_component.variance,
                variance_decomposition.total_joint_variance,
                rtol=1e-12,
                atol=1e-15,
            )
        ):
            raise ValueError(
                "combined uncertainty requires an identifiable total joint variance"
            )
    required_components_available = all(
        item.availability is UncertaintyAvailability.QUANTIFIED
        for item in component_rows
    )
    observed_checkpoints = evaluate_blockwise_convergence(
        seed_rows,
        expected_seeds=expected,
        rule=convergence_rule,
        rejected_count=rejected_count,
        excluded_count=excluded_count,
        sensitivity_instability=sensitivity_instability,
        required_components_available=required_components_available,
    )
    if checkpoint_rows_input != observed_checkpoints:
        raise ValueError(
            "convergence checkpoints differ from the declared seed series and rule"
        )
    if not isinstance(sufficiency_judgment, Mapping):
        raise TypeError("sufficiency_judgment must be a mapping")
    final_status = (
        observed_checkpoints[-1].status
        if observed_checkpoints
        else ConvergenceStatus.NON_CONVERGED
    )
    observed_judgment = final_sufficiency_judgment(
        convergence_status=final_status,
        components=component_rows,
    )
    if dict(sufficiency_judgment) != observed_judgment:
        raise ValueError(
            "sufficiency judgment differs from convergence and component statuses"
        )
    if observed_judgment.get("campaign_ready") is not False:
        raise ValueError(
            "the uncertainty writer cannot independently claim campaign readiness"
        )
    if type(plan_id) is not str or not plan_id or any(
        character.isspace() for character in plan_id
    ):
        raise ValueError("plan_id must be a non-empty identifier")
    for name, value in (
        ("plan_sha256", plan_sha256),
        ("config_sha256", config_sha256),
        ("output_profile_schema_sha256", output_profile_schema_sha256),
    ):
        if type(value) is not str or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{name} must be lowercase SHA-256 hex")
    if type(output_profile) is not str or not output_profile.strip():
        raise ValueError("output_profile must be non-empty text")
    if (
        type(output_profile_schema_version) is not str
        or not output_profile_schema_version.strip()
    ):
        raise ValueError(
            "output_profile_schema_version must be non-empty text"
        )
    destination = Path(output_dir)
    realization_rows = [_realization_row(item) for item in joint_rows]
    checkpoint_rows = [_checkpoint_row(item) for item in observed_checkpoints]
    summary = {
        "output_profile": output_profile,
        "output_profile_schema_version": output_profile_schema_version,
        "output_profile_schema_sha256": output_profile_schema_sha256,
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "config_sha256": config_sha256,
        "fixed_seed_set": list(expected),
        "convergence_rule": convergence_rule.snapshot(),
        "primary_point_estimate": (
            primary_seed_summary.point_estimate
            if primary_seed_summary is not None
            else None
        ),
        "seed_uncertainty": (
            primary_seed_summary.snapshot()
            if primary_seed_summary is not None
            else {
                "availability": "UNAVAILABLE",
                "blocker": "no complete fixed-seed primary result",
            }
        ),
        "parameter_uncertainty": component_payload.get("parameter"),
        "monetary_rate_uncertainty": component_payload.get("monetary_rate"),
        "population_uncertainty": component_payload.get("population"),
        "combined_uncertainty": component_payload.get("combined"),
        "all_uncertainty_components": component_payload,
        "variance_decomposition": variance_decomposition.snapshot(),
        "convergence": (
            observed_checkpoints[-1].snapshot()
            if observed_checkpoints
            else {
                "status": "NON_CONVERGED",
                "blockers": ["no_convergence_checkpoint"],
            }
        ),
        "final_sufficiency_judgment": observed_judgment,
        "outcome_dependent_seed_exclusion_allowed": False,
        "oat_role": "DIAGNOSTIC_ONLY",
        "campaign_ready": False,
    }
    return {
        "uncertainty_realizations": write_csv_atomic(
            destination / "uncertainty_realizations.csv",
            realization_rows,
            canonical_columns=UNCERTAINTY_REALIZATION_COLUMNS,
            allow_extra_columns=False,
        ),
        "convergence_checkpoints": write_csv_atomic(
            destination / "convergence_checkpoints.csv",
            checkpoint_rows,
            canonical_columns=CONVERGENCE_CHECKPOINT_COLUMNS,
            allow_extra_columns=False,
        ),
        "uncertainty_summary": write_json_atomic(
            destination / "uncertainty_summary.json",
            summary,
        ),
    }


def _validate_component_against_decomposition(
    component: UncertaintyComponentStatus,
    decomposition_variance: float | None,
    *,
    name: str,
) -> None:
    """Reject quantified factor components not supported by the decomposition."""

    if component.availability is not UncertaintyAvailability.QUANTIFIED:
        return
    if (
        decomposition_variance is None
        or component.variance is None
        or not np.isclose(
            component.variance,
            decomposition_variance,
            rtol=1e-12,
            atol=1e-15,
        )
    ):
        raise ValueError(
            f"quantified {name} component differs from the variance decomposition"
        )


__all__ = ["write_joint_uncertainty_outputs"]
