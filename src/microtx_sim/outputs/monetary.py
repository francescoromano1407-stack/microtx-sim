"""Separate production-shaped output for monetary model equivalents.

The legacy simulator-unit tables remain diagnostic.  This writer accepts only
post-conversion, population-weighted analysis bindings, aggregates their exact
rational seed estimates, and performs the sole monetary rounding operation at
the serialized output boundary.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import json
from math import sqrt
from pathlib import Path

from ..causal.analysis_binding import RunAnalysisBinding, SeedAnalysisBinding
from ..data.monetary_execution import round_target_minor_units
from .writers import _render_csv, _render_json, write_text_atomic


PRODUCTION_MONETARY_SCHEMA_VERSION = "1.0"
PRODUCTION_MONETARY_ARTIFACT_FILENAMES = (
    "production_monetary_estimates.csv",
    "production_monetary_metadata.json",
)
PRODUCTION_MONETARY_COLUMNS = (
    "estimand_id",
    "reference_scenario_id",
    "comparison_scenario_id",
    "contrast_direction",
    "estimate_label",
    "target_currency",
    "target_minor_unit_name",
    "reference_period_start",
    "reference_period_end",
    "population_identifier",
    "population_weighting_rule",
    "converted_monetary_estimate_minor_units",
    "unrounded_estimate_numerator_decimal",
    "unrounded_estimate_denominator_decimal",
    "retained_seed_count_decimal",
    "between_seed_sample_standard_deviation_minor_units",
    "monte_carlo_standard_error_minor_units",
    "interval_method",
    "interval_lower_minor_units",
    "interval_upper_minor_units",
    "rounding_rule",
    "rounding_point",
    "rounding_count_decimal",
    "conversion_bundle_id",
    "conversion_bundle_sha256",
    "conversion_basis_sha256",
    "source_artifact_sha256s",
    "source_bundle_signature_status",
    "plan_sha256",
    "analysis_binding_sha256",
    "metric_contract_sha256",
    "execution_sha256s",
    "empirically_calibrated_real_money_interpretation_available",
    "campaign_ready",
)


def write_production_monetary_outputs(
    output_dir: str | Path,
    binding: RunAnalysisBinding,
) -> dict[str, Path]:
    """Write monetary estimates separately from raw simulator-unit output."""

    if type(binding) is not RunAnalysisBinding:
        raise TypeError("binding must be RunAnalysisBinding")
    RunAnalysisBinding.__post_init__(binding)
    grouped = _money_bindings(binding)
    if not grouped:
        raise ValueError("production monetary output requires a monetary estimand")
    rows = [_estimate_row(binding, items) for items in grouped]
    metadata = _metadata(binding, grouped, rows)
    csv_text = _render_csv(
        rows,
        canonical_columns=PRODUCTION_MONETARY_COLUMNS,
        allow_extra_columns=False,
    )
    metadata_text = _render_json(metadata)
    destination = Path(output_dir)
    return {
        "production_monetary_estimates": write_text_atomic(
            destination / PRODUCTION_MONETARY_ARTIFACT_FILENAMES[0],
            csv_text,
        ),
        "production_monetary_metadata": write_text_atomic(
            destination / PRODUCTION_MONETARY_ARTIFACT_FILENAMES[1],
            metadata_text,
        ),
    }


def monetary_lineage_payload(binding: RunAnalysisBinding) -> dict[str, object]:
    """Return the complete monetary manifest lineage without writing files."""

    if type(binding) is not RunAnalysisBinding:
        raise TypeError("binding must be RunAnalysisBinding")
    RunAnalysisBinding.__post_init__(binding)
    grouped = _money_bindings(binding)
    if not grouped:
        return {"present": False, "campaign_ready": False}
    rows = [_estimate_row(binding, items) for items in grouped]
    return _metadata(binding, grouped, rows)


def _money_bindings(
    binding: RunAnalysisBinding,
) -> tuple[tuple[SeedAnalysisBinding, ...], ...]:
    by_estimand: dict[str, list[SeedAnalysisBinding]] = {}
    for item in binding.seed_bindings:
        if item.monetary_output_basis is not None:
            by_estimand.setdefault(
                item.planned_estimand.estimand_id,
                [],
            ).append(item)
    output: list[tuple[SeedAnalysisBinding, ...]] = []
    for estimand_id in sorted(by_estimand):
        items = tuple(by_estimand[estimand_id])
        if tuple(item.seed for item in items) != binding.seeds:
            raise ValueError(
                f"monetary estimand {estimand_id!r} does not cover every fixed seed"
            )
        first = items[0]
        if any(
            item.planned_estimand != first.planned_estimand
            or item.monetary_output_basis != first.monetary_output_basis
            or item.spec.target_population_id != first.spec.target_population_id
            for item in items
        ):
            raise ValueError(
                f"monetary estimand {estimand_id!r} changes basis or population"
            )
        output.append(items)
    return tuple(output)


def _estimate_row(
    binding: RunAnalysisBinding,
    items: tuple[SeedAnalysisBinding, ...],
) -> dict[str, object]:
    first = items[0]
    basis = first.monetary_output_basis
    assert basis is not None
    exact_values = tuple(item.result.value_fraction for item in items)
    exact_mean = sum(exact_values, Fraction(0, 1)) / len(exact_values)
    if len(exact_values) == 1:
        standard_deviation = 0.0
    else:
        exact_sum_squares = sum(
            ((value - exact_mean) ** 2 for value in exact_values),
            Fraction(0, 1),
        )
        standard_deviation = sqrt(
            float(exact_sum_squares / (len(exact_values) - 1))
        )
    mcse = standard_deviation / sqrt(len(exact_values))
    half_width = 1.96 * mcse
    source_hashes = sorted(
        {row.rate_artifact_sha256 for row in basis.jurisdictions}
    )
    execution_hashes: list[str] = []
    for item in items:
        if (
            item.reference_monetary_execution is None
            or item.comparison_monetary_execution is None
        ):
            raise ValueError(
                "production monetary output requires both scenario executions"
            )
        execution_hashes.extend(
            (
                item.reference_monetary_execution.execution_sha256,
                item.comparison_monetary_execution.execution_sha256,
            )
        )
    return {
        "estimand_id": first.planned_estimand.estimand_id,
        "reference_scenario_id": first.planned_estimand.reference_scenario_id.value,
        "comparison_scenario_id": (
            first.planned_estimand.comparison_scenario_id.value
        ),
        "contrast_direction": first.planned_estimand.contrast_direction,
        "estimate_label": (
            f"{basis.target_currency}-equivalent model amount; not observed "
            "real-world spending"
        ),
        "target_currency": basis.target_currency,
        "target_minor_unit_name": basis.target_minor_unit_name,
        "reference_period_start": basis.rate_period_start.isoformat(),
        "reference_period_end": basis.rate_period_end.isoformat(),
        "population_identifier": first.spec.target_population_id,
        "population_weighting_rule": (
            "the same exact pre-treatment projected-population weights are "
            "applied separately to reference and comparison scenarios within "
            "each seed before equal-seed aggregation"
        ),
        # This call is the sole monetary rounding boundary.
        "converted_monetary_estimate_minor_units": round_target_minor_units(
            exact_mean
        ),
        "unrounded_estimate_numerator_decimal": str(exact_mean.numerator),
        "unrounded_estimate_denominator_decimal": str(exact_mean.denominator),
        "retained_seed_count_decimal": str(len(exact_values)),
        "between_seed_sample_standard_deviation_minor_units": standard_deviation,
        "monte_carlo_standard_error_minor_units": mcse,
        "interval_method": "NORMAL_95_MONTE_CARLO_MEAN_PLUS_MINUS_1.96_MCSE",
        "interval_lower_minor_units": float(exact_mean) - half_width,
        "interval_upper_minor_units": float(exact_mean) + half_width,
        "rounding_rule": basis.rounding_method,
        "rounding_point": "final production monetary estimate only",
        "rounding_count_decimal": "1",
        "conversion_bundle_id": basis.source_bundle_id,
        "conversion_bundle_sha256": basis.source_bundle_sha256,
        "conversion_basis_sha256": basis.basis_sha256,
        "source_artifact_sha256s": json.dumps(source_hashes, separators=(",", ":")),
        "source_bundle_signature_status": basis.source_bundle_signature_status,
        "plan_sha256": binding.plan.plan_sha256,
        "analysis_binding_sha256": binding.binding_sha256,
        "metric_contract_sha256": first.metric_contract_sha256,
        "execution_sha256s": json.dumps(execution_hashes, separators=(",", ":")),
        "empirically_calibrated_real_money_interpretation_available": False,
        "campaign_ready": False,
    }


def _metadata(
    binding: RunAnalysisBinding,
    grouped: tuple[tuple[SeedAnalysisBinding, ...], ...],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    bases = {
        items[0].monetary_output_basis.basis_sha256: (
            items[0].monetary_output_basis
        )
        for items in grouped
        if items[0].monetary_output_basis is not None
    }
    applied_weights = []
    for items in grouped:
        for item in items:
            counts = Counter(item.selected_weights.fractions)
            applied_weights.append(
                {
                    "estimand_id": item.planned_estimand.estimand_id,
                    "seed_decimal": str(item.seed),
                    "design_weights_sha256": item.selected_weights.design_sha256,
                    "selected_player_count_decimal": str(
                        len(item.selected_weights.player_ids)
                    ),
                    "weight_sum_numerator_decimal": str(
                        item.selected_weights.weight_sum.numerator
                    ),
                    "weight_sum_denominator_decimal": str(
                        item.selected_weights.weight_sum.denominator
                    ),
                    "weight_value_counts": [
                        {
                            "numerator_decimal": str(weight.numerator),
                            "denominator_decimal": str(weight.denominator),
                            "player_count_decimal": str(counts[weight]),
                        }
                        for weight in sorted(counts)
                    ],
                    "same_weights_used_for_reference_and_comparison": True,
                }
            )
    source_files = sorted(
        {
            (
                row.rate_artifact_id,
                row.rate_artifact_sha256,
                row.rate_artifact_byte_length,
            )
            for basis in bases.values()
            for row in basis.jurisdictions
        }
    )
    blockers = sorted(
        {
            blocker
            for basis in bases.values()
            for blocker in basis.campaign_blockers
        }.union(binding.campaign_blockers)
    )
    payload: dict[str, object] = {
        "schema_version": PRODUCTION_MONETARY_SCHEMA_VERSION,
        "present": True,
        "output_role": "separate production-shaped monetary model-equivalent output",
        "diagnostic_simulator_unit_outputs_separate": True,
        "raw_simulation_cents_allowed_as_final_estimand": False,
        "observed_real_world_spending_claimed": False,
        "plan_id": binding.plan.plan_id,
        "plan_sha256": binding.plan.plan_sha256,
        "analysis_binding_sha256": binding.binding_sha256,
        "profile_input_sha256": binding.profile_input_sha256,
        "population_input_sha256": binding.population_input_sha256,
        "population_lineage_sha256": binding.population_lineage_sha256,
        "metric_contract_registry_sha256": (
            binding.metric_contract_registry_sha256
        ),
        "monetary_bases": [bases[digest].snapshot() for digest in sorted(bases)],
        "source_files": [
            {
                "artifact_id": artifact_id,
                "sha256": digest,
                "byte_length": byte_length,
            }
            for artifact_id, digest, byte_length in source_files
        ],
        "applied_population_weights": applied_weights,
        "conversion_before_aggregation": True,
        "raw_cross_jurisdiction_summation_allowed": False,
        "single_rounding_rule": (
            "nearest target minor unit, signed half away from zero, once at "
            "the final production monetary estimate boundary"
        ),
        "rounding_operation_count_per_reported_estimate": 1,
        "records": rows,
        "campaign_ready": False,
        "campaign_readiness_blockers": blockers,
    }
    payload["lineage_sha256"] = _canonical_sha256(payload)
    return payload


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
    "PRODUCTION_MONETARY_ARTIFACT_FILENAMES",
    "PRODUCTION_MONETARY_COLUMNS",
    "PRODUCTION_MONETARY_SCHEMA_VERSION",
    "monetary_lineage_payload",
    "write_production_monetary_outputs",
]
