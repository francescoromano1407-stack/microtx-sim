"""End-to-end export of tables, metadata, human summary, and SVG charts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

from ..analysis.sensitivity import SensitivityResult
from ..causal.batch import PolicyBatchResult
from ..causal.scenarios import ScenarioId
from ..policy_config import PolicyPrototypeConfig
from .manifest import build_run_manifest
from .plots import (
    write_epgc_subsidy_requirement_svg,
    write_harm_distribution_svg,
    write_harm_revenue_frontier_svg,
    write_opportunity_cost_decomposition_svg,
    write_spending_distribution_svg,
)
from .schema import (
    OPPORTUNITY_DECOMPOSITION_COLUMNS,
    OUTPUT_SCHEMA_VERSION,
    PLAYER_OUTCOME_COLUMNS,
    POLICY_ARTIFACT_FILENAMES,
)
from .writers import (
    preflight_csv_rows,
    write_batch_artifacts,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


def export_policy_batch(
    config: PolicyPrototypeConfig,
    batch: PolicyBatchResult,
    sensitivity: SensitivityResult | None,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    output_dir: str | Path | None = None,
    created_utc: str | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Persist a complete, self-describing synthetic result bundle."""

    destination = Path(output_dir) if output_dir is not None else config.output.output_dir
    batch_lineage = batch.profile_input_lineage
    sensitivity_lineage = (
        sensitivity.profile_input_lineage if sensitivity is not None else None
    )
    if batch_lineage is None:
        raise ValueError("policy batch export requires profile input lineage")
    if sensitivity is not None and sensitivity_lineage is None:
        raise ValueError("sensitivity export requires profile input lineage")
    if sensitivity_lineage is not None and (
        batch_lineage.fingerprint_sha256
        != sensitivity_lineage.fingerprint_sha256
    ):
        raise ValueError(
            "batch and sensitivity results used different profile inputs"
        )
    manifest = build_run_manifest(
        config,
        batch,
        config_path=config_path,
        repository_root=repository_root,
        created_utc=created_utc,
        command=command,
    )
    manifest["sensitivity"] = {
        "run": sensitivity is not None,
        "profile_input_fingerprint_sha256": (
            sensitivity_lineage.fingerprint_sha256
            if sensitivity_lineage is not None
            else None
        ),
    }
    sensitivity_rows = list(sensitivity.rows) if sensitivity is not None else []
    player_rows = batch.player_rows() if config.output.include_player_rows else []
    opportunity_rows = batch.opportunity_rows()
    preflight_csv_rows(
        player_rows,
        canonical_columns=PLAYER_OUTCOME_COLUMNS,
        allow_extra_columns=False,
    )
    preflight_csv_rows(
        opportunity_rows,
        canonical_columns=OPPORTUNITY_DECOMPOSITION_COLUMNS,
        allow_extra_columns=False,
    )
    paths = write_batch_artifacts(
        destination,
        batch.seed_rows(),
        batch.scenario_rows(),
        batch.epgc_rows(),
        sensitivity_rows,
        manifest,
    )
    paths["player_outcomes"] = write_csv_atomic(
        destination / "player_outcomes.csv",
        player_rows,
        canonical_columns=PLAYER_OUTCOME_COLUMNS,
        allow_extra_columns=False,
    )
    paths["opportunity_cost_decomposition"] = write_csv_atomic(
        destination / "opportunity_cost_decomposition.csv",
        opportunity_rows,
        canonical_columns=OPPORTUNITY_DECOMPOSITION_COLUMNS,
        allow_extra_columns=False,
    )
    paths["summary"] = write_text_atomic(
        destination / "summary.md",
        render_human_summary(batch, sensitivity),
    )

    baseline_players = [
        row
        for row in player_rows
        if row["scenario_id"] == ScenarioId.BASELINE_F2P.value
    ]
    paths["harm_distribution"] = write_harm_distribution_svg(
        destination / "harm_distribution.svg",
        [float(row["composite_harm"]) for row in baseline_players],
        bins=config.output.histogram_bins,
        title="Baseline F2P harm distribution",
    )
    paths["spending_distribution"] = write_spending_distribution_svg(
        destination / "spending_distribution.svg",
        [float(row["spending_cents"]) for row in baseline_players],
        bins=config.output.histogram_bins,
        title="Baseline F2P spending distribution",
    )
    summary_rows = batch.scenario_rows()
    paths["harm_revenue_frontier"] = write_harm_revenue_frontier_svg(
        destination / "harm_revenue_frontier.svg",
        summary_rows,
        scenario_key="scenario_id",
        revenue_key="total_revenue_cents_mean",
        harm_key="mean_harm_mean",
    )
    baseline_opportunity = [
        {
            "component": row["component"],
            "value": row["mean_minutes"],
        }
        for row in opportunity_rows
        if row["scenario_id"] == ScenarioId.BASELINE_F2P.value
        and row["component"] != "all_displaced_activities"
    ]
    paths["opportunity_cost_plot"] = write_opportunity_cost_decomposition_svg(
        destination / "opportunity_cost_decomposition.svg",
        baseline_opportunity,
        title="Baseline F2P displaced-activity decomposition",
    )
    epgc_plot_rows = [
        {
            "scenario": f"EPGC seed {row['seed']}",
            "minimum_public_contribution_cents": row[
                "minimum_public_contribution_cents"
            ],
        }
        for row in batch.epgc_rows()
    ]
    paths["epgc_subsidy_plot"] = write_epgc_subsidy_requirement_svg(
        destination / "epgc_subsidy_requirement.svg",
        epgc_plot_rows,
    )

    # The initial manifest is written before charts so failures never describe
    # non-existent outputs.  Once all writes succeed, replace it with complete
    # file names, sizes, and hashes (excluding the self-referential manifest).
    manifest["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    manifest["artifact_files"] = list(POLICY_ARTIFACT_FILENAMES)
    manifest["artifacts"] = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
        for path in sorted(destination.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "manifest.json"
    }
    paths["manifest"] = write_json_atomic(destination / "manifest.json", manifest)
    actual = {path.name for path in paths.values()}
    expected = set(POLICY_ARTIFACT_FILENAMES)
    if actual != expected:
        raise RuntimeError(
            f"exported artifact set differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return paths


def render_human_summary(
    batch: PolicyBatchResult,
    sensitivity: SensitivityResult | None,
) -> str:
    """Render a concise Markdown summary that labels every result synthetic."""

    lines = [
        "# Synthetic EU-GAME-HARM policy prototype results",
        "",
        "> These figures are generated by illustrative assumptions. They are not empirical estimates, clinical findings, or legal conclusions.",
        "",
        "| Scenario | Revenue (mean cents) | Spending (mean cents) | Harmful spending | Mean harm | Δ harm vs safe | Enjoyment | High-risk outcomes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch.scenario_rows():
        lines.append(
            "| {label} | {revenue:.0f} | {spending:.0f} | {harmful:.0f} | "
            "{harm:.4f} | {effect:.4f} | {enjoyment:.4f} | {risk:.2f} |".format(
                label=row["scenario_label"],
                revenue=row["total_revenue_cents_mean"],
                spending=row["total_spending_cents_mean"],
                harmful=row["harmful_spending_cents_mean"],
                harm=row["mean_harm_mean"],
                effect=row["mean_harm_effect_vs_safe_mean"],
                enjoyment=row["mean_enjoyment_mean"],
                risk=row["high_risk_count_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The comparisons are causal only inside the declared structural model because every branch uses the same synthetic cohort and common random coordinates. Later empirical work must calibrate and validate every behavioural, welfare, cost, and financing parameter.",
        ]
    )
    if sensitivity is not None:
        unstable = ", ".join(sensitivity.unstable_parameters) or "none flagged"
        lines.extend(
            [
                "",
                "## Sensitivity diagnostic",
                "",
                f"Parameters flagged as unstable under the configured synthetic grid: {unstable}.",
            ]
        )
    return "\n".join(lines) + "\n"


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["export_policy_batch", "render_human_summary"]
