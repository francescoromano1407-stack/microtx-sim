"""Versioned, reproducible tabular and SVG simulation outputs."""

from .plots import (
    render_epgc_subsidy_requirement_svg,
    render_harm_distribution_svg,
    render_harm_revenue_frontier_svg,
    render_opportunity_cost_decomposition_svg,
    render_spending_distribution_svg,
    write_epgc_subsidy_requirement_svg,
    write_harm_distribution_svg,
    write_harm_revenue_frontier_svg,
    write_opportunity_cost_decomposition_svg,
    write_spending_distribution_svg,
)
from .schema import (
    MANIFEST_SCHEMA_SHA256,
    MANIFEST_SCHEMA_VERSION,
    OUTPUT_SCHEMA_VERSION,
    STANDALONE_SENSITIVITY_PROFILE,
    STANDALONE_SENSITIVITY_SCHEMA_SHA256,
    STANDALONE_SENSITIVITY_SCHEMA_VERSION,
    manifest_schema_descriptor,
    stamp_standalone_sensitivity_schema,
)
from .metric_contracts import (
    OUTPUT_METRIC_CONTRACTS,
    metric_contract_registry_sha256,
)
from .export import export_policy_batch, render_human_summary
from .manifest import build_run_manifest
from .writers import write_batch_artifacts, write_csv_atomic, write_json_atomic

__all__ = [
    "MANIFEST_SCHEMA_SHA256",
    "MANIFEST_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "STANDALONE_SENSITIVITY_PROFILE",
    "STANDALONE_SENSITIVITY_SCHEMA_SHA256",
    "STANDALONE_SENSITIVITY_SCHEMA_VERSION",
    "OUTPUT_METRIC_CONTRACTS",
    "build_run_manifest",
    "export_policy_batch",
    "render_epgc_subsidy_requirement_svg",
    "render_harm_distribution_svg",
    "render_harm_revenue_frontier_svg",
    "render_human_summary",
    "render_opportunity_cost_decomposition_svg",
    "render_spending_distribution_svg",
    "metric_contract_registry_sha256",
    "manifest_schema_descriptor",
    "stamp_standalone_sensitivity_schema",
    "write_batch_artifacts",
    "write_csv_atomic",
    "write_epgc_subsidy_requirement_svg",
    "write_harm_distribution_svg",
    "write_harm_revenue_frontier_svg",
    "write_json_atomic",
    "write_opportunity_cost_decomposition_svg",
    "write_spending_distribution_svg",
]
