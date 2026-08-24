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
from .schema import OUTPUT_SCHEMA_VERSION
from .export import export_policy_batch, render_human_summary
from .manifest import build_run_manifest
from .writers import write_batch_artifacts, write_csv_atomic, write_json_atomic

__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "build_run_manifest",
    "export_policy_batch",
    "render_epgc_subsidy_requirement_svg",
    "render_harm_distribution_svg",
    "render_harm_revenue_frontier_svg",
    "render_human_summary",
    "render_opportunity_cost_decomposition_svg",
    "render_spending_distribution_svg",
    "write_batch_artifacts",
    "write_csv_atomic",
    "write_epgc_subsidy_requirement_svg",
    "write_harm_distribution_svg",
    "write_harm_revenue_frontier_svg",
    "write_json_atomic",
    "write_opportunity_cost_decomposition_svg",
    "write_spending_distribution_svg",
]
