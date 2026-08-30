"""Writer for the separate plan-level PRIMARY Monte Carlo aggregate."""

from __future__ import annotations

from pathlib import Path

from ..causal.primary_aggregate import (
    NORMAL_95_MONTE_CARLO_INTERVAL,
    PlanPrimaryAggregate,
)
from .schema import (
    PRIMARY_AGGREGATE_ARTIFACT_FILENAMES,
    PRIMARY_AGGREGATE_COLUMNS,
    PROSPECTIVE_ANALYSIS_ARTIFACT_FILENAMES,
    PROSPECTIVE_ANALYSIS_OUTPUT_PROFILE,
    PROSPECTIVE_ANALYSIS_SCHEMA_SHA256,
    PROSPECTIVE_ANALYSIS_SCHEMA_VERSION,
)
from .writers import _render_csv, _render_json, write_text_atomic


def write_primary_aggregate(
    output_dir: str | Path,
    aggregate: PlanPrimaryAggregate,
) -> dict[str, Path]:
    """Write one re-attested aggregate CSV and complete metadata snapshot."""

    if type(aggregate) is not PlanPrimaryAggregate:
        raise TypeError("aggregate must be PlanPrimaryAggregate")
    PlanPrimaryAggregate.__post_init__(aggregate)
    payload = aggregate.snapshot()
    row = {
        "plan_id": payload["plan_id"],
        "primary_estimand_id": payload["primary_estimand_id"],
        "reference_scenario": payload["reference_scenario"],
        "comparison_scenario": payload["comparison_scenario"],
        "outcome_metric": payload["outcome_metric"],
        "outcome_unit": payload["outcome_unit"],
        "contrast_direction": payload["contrast_direction"],
        "point_estimate": payload["point_estimate"],
        "between_seed_sample_standard_deviation": payload[
            "between_seed_sample_standard_deviation"
        ],
        "retained_seed_count": payload["retained_seed_count"],
        "monte_carlo_standard_error": payload[
            "monte_carlo_standard_error"
        ],
        "interval_method": NORMAL_95_MONTE_CARLO_INTERVAL,
        "interval_lower": payload["interval_lower"],
        "interval_upper": payload["interval_upper"],
        "exclusion_count": payload["excluded_seed_count"],
        "plan_sha256": payload["plan_sha256"],
        "binding_sha256": payload["binding_sha256"],
        "population_input_sha256": payload["population_input_sha256"],
        "population_lineage_sha256": payload["population_lineage_sha256"],
        "profile_input_sha256": payload["profile_input_sha256"],
        "metric_contract_registry_sha256": payload[
            "metric_contract_registry_sha256"
        ],
        "primary_metric_contract_sha256": payload[
            "primary_metric_contract_sha256"
        ],
        "harm_weights_sha256": payload["harm_weights_sha256"],
        "analysis_population_predicate_sha256": payload[
            "analysis_population_predicate_sha256"
        ],
        "aggregate_sha256": payload["aggregate_sha256"],
    }
    metadata = {
        "output_profile": PROSPECTIVE_ANALYSIS_OUTPUT_PROFILE,
        "output_profile_schema_version": PROSPECTIVE_ANALYSIS_SCHEMA_VERSION,
        "output_profile_schema_sha256": PROSPECTIVE_ANALYSIS_SCHEMA_SHA256,
        "output_profile_artifact_files": list(
            PROSPECTIVE_ANALYSIS_ARTIFACT_FILENAMES
        ),
        "aggregate_artifact_files": list(
            PRIMARY_AGGREGATE_ARTIFACT_FILENAMES
        ),
        "primary_aggregate": payload,
        "interval_scope": (
            "Monte Carlo variability of simulator output only; not a confidence "
            "interval for a real-world population."
        ),
        "legacy_root_output_v3_changed": False,
        "raw_simulation_cents_relabelled": False,
        "synthetic_only": True,
        "campaign_ready": False,
    }
    csv_text = _render_csv(
        [row],
        canonical_columns=PRIMARY_AGGREGATE_COLUMNS,
        allow_extra_columns=False,
    )
    metadata_text = _render_json(metadata)
    destination = Path(output_dir)
    return {
        "primary_aggregate": write_text_atomic(
            destination / "primary_aggregate.csv",
            csv_text,
        ),
        "primary_aggregate_metadata": write_text_atomic(
            destination / "primary_aggregate_metadata.json",
            metadata_text,
        ),
    }


__all__ = ["write_primary_aggregate"]
