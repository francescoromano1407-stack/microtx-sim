"""End-to-end export of tables, metadata, human summary, and SVG charts."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping, Sequence

from ..analysis.sensitivity import SensitivityResult
from ..causal.analysis_binding import (
    RunAnalysisBinding,
    resolve_run_analysis_binding,
)
from ..causal.analysis_plan import (
    LoadedProspectiveAnalysisPlan,
    verify_loaded_prospective_analysis_plan,
)
from ..causal.primary_aggregate import compute_plan_primary_aggregate
from ..causal.batch import PolicyBatchResult, resolve_policy_run_inputs
from ..causal.scenarios import ScenarioId
from ..policy_config import PolicyPrototypeConfig
from .manifest import build_run_manifest
from .monetary import (
    PRODUCTION_MONETARY_ARTIFACT_FILENAMES,
    write_production_monetary_outputs,
)
from .population import write_target_population_estimands
from .prospective import write_primary_aggregate
from .plots import (
    write_epgc_subsidy_requirement_svg,
    write_harm_distribution_svg,
    write_harm_revenue_frontier_svg,
    write_opportunity_cost_decomposition_svg,
    write_spending_distribution_svg,
)
from .schema import (
    OPPORTUNITY_DECOMPOSITION_COLUMNS,
    PLAYER_OUTCOME_COLUMNS,
    POLICY_ARTIFACT_FILENAMES,
    PROSPECTIVE_ANALYSIS_ARTIFACT_FILENAMES,
    PROSPECTIVE_ANALYSIS_OUTPUT_PROFILE,
    TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES,
    stamp_manifest_schema,
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
    analysis_plan: LoadedProspectiveAnalysisPlan | None = None,
    analysis_binding: RunAnalysisBinding | None = None,
) -> dict[str, Path]:
    """Persist a complete, self-describing synthetic result bundle."""

    if not isinstance(config, PolicyPrototypeConfig):
        raise TypeError("config must be PolicyPrototypeConfig")
    if not isinstance(batch, PolicyBatchResult):
        raise TypeError("batch must be PolicyBatchResult")
    if sensitivity is not None and not isinstance(sensitivity, SensitivityResult):
        raise TypeError("sensitivity must be SensitivityResult or None")
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
    configured_run_inputs = resolve_policy_run_inputs(
        harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
    )
    if batch.spec != config.batch:
        raise ValueError(
            "configuration batch specification does not match the executed batch"
        )
    if batch.run_inputs != configured_run_inputs:
        raise ValueError(
            "configuration model inputs do not match the executed batch"
        )
    if sensitivity is not None:
        if sensitivity.batch_spec != batch.spec:
            raise ValueError(
                "batch and sensitivity used different batch specifications"
            )
        if sensitivity.run_inputs != batch.run_inputs:
            raise ValueError(
                "batch and sensitivity used different resolved model inputs"
            )
        batch_population = batch.population_execution_lineage
        sensitivity_population = sensitivity.population_execution_lineage
        if (batch_population is None) != (sensitivity_population is None):
            raise ValueError(
                "batch and sensitivity used different population execution modes"
            )
        if batch_population is not None and sensitivity_population is not None:
            if (
                batch_population.manifest_payload()
                != sensitivity_population.manifest_payload()
            ):
                raise ValueError(
                    "batch and sensitivity used different population executions"
                )
    destination = (
        Path(output_dir)
        if output_dir is not None
        else config.output.output_dir
    )
    manifest = build_run_manifest(
        config,
        batch,
        config_path=config_path,
        repository_root=repository_root,
        created_utc=created_utc,
        command=command,
        analysis_plan=analysis_plan,
        analysis_binding=analysis_binding,
    )
    sensitivity_snapshot = (
        sensitivity.execution_snapshot() if sensitivity is not None else None
    )
    manifest["sensitivity"] = {
        "run": sensitivity is not None,
        "execution_sha256": (
            sensitivity.execution_sha256()
            if sensitivity is not None
            else None
        ),
        "execution_snapshot": sensitivity_snapshot,
        "run_inputs_sha256": (
            sensitivity.run_inputs.snapshot_sha256()
            if sensitivity is not None
            else None
        ),
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
    analysis_destination = destination / "prospective_analysis"
    _reject_existing_analysis_output(analysis_destination)
    analysis_stage: Path | None = None
    staged_analysis_paths: dict[str, Path] = {}
    analysis_file_identities: dict[str, tuple[int, str]] = {}
    analysis_output_profile: dict[str, object] | None = None
    if analysis_plan is not None and analysis_binding is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        analysis_stage = Path(
            tempfile.mkdtemp(
                dir=destination.parent,
                prefix=f".{destination.name}.prospective-analysis-",
            )
        )
        try:
            staged_analysis_paths = write_target_population_estimands(
                analysis_stage,
                analysis_binding.writer_pairs,
                metadata={
                    "analysis_plan": analysis_plan.manifest_payload(),
                    "analysis_binding": analysis_binding.manifest_payload(),
                    "composition_scope": (
                        "The run binding resolves the declared plan against exact "
                        "execution lineage; the standalone writer independently "
                        "re-attests only each supplied spec/result pair."
                    ),
                },
            )
            primary_aggregate = None
            analysis_artifact_files = (
                TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES
            )
            if analysis_plan.plan.primary_aggregate_rule is not None:
                primary_aggregate = compute_plan_primary_aggregate(
                    analysis_binding
                )
                staged_analysis_paths.update(
                    write_primary_aggregate(
                        analysis_stage,
                        primary_aggregate,
                    )
                )
                analysis_artifact_files = (
                    PROSPECTIVE_ANALYSIS_ARTIFACT_FILENAMES
                )
            if analysis_binding.monetary_output_bases:
                staged_analysis_paths.update(
                    write_production_monetary_outputs(
                        analysis_stage,
                        analysis_binding,
                    )
                )
                analysis_artifact_files = (
                    analysis_artifact_files
                    + PRODUCTION_MONETARY_ARTIFACT_FILENAMES
                )
            analysis_file_identities = {
                path.name: (path.stat().st_size, _digest(path))
                for path in staged_analysis_paths.values()
            }
            analysis_output_profile = {
                "directory": "prospective_analysis",
                "output_profile": (
                    PROSPECTIVE_ANALYSIS_OUTPUT_PROFILE
                    if primary_aggregate is not None
                    else "target_population_estimands"
                ),
                "output_profile_schema_sha256": (
                    analysis_binding.output_profile_schema_sha256
                ),
                "artifact_files": list(analysis_artifact_files),
                "record_count_decimal": str(len(analysis_binding.writer_pairs)),
                "binding_sha256": analysis_binding.binding_sha256,
                "primary_aggregate_sha256": (
                    primary_aggregate.aggregate_sha256
                    if primary_aggregate is not None
                    else None
                ),
                "production_monetary_output_present": bool(
                    analysis_binding.monetary_output_bases
                ),
                "campaign_ready": False,
                "artifacts": {
                    name: {
                        "relative_path": (
                            Path("prospective_analysis") / name
                        ).as_posix(),
                        "bytes": size,
                        "sha256": digest,
                    }
                    for name, (size, digest) in sorted(
                        analysis_file_identities.items()
                    )
                },
            }
        except BaseException:
            _remove_owned_analysis_directory(
                analysis_stage,
                expected_parent=destination.parent,
                require_complete=False,
            )
            analysis_stage = None
            raise

    try:
        # This interim manifest deliberately omits analysis_output_profile.  A
        # later root-writer failure therefore cannot leave a manifest claiming
        # an unpublished optional directory.
        paths = write_batch_artifacts(
            destination,
            batch.seed_rows(),
            batch.scenario_rows(),
            batch.epgc_rows(),
            sensitivity_rows,
            manifest,
        )
        interim_manifest_text = paths["manifest"].read_text(encoding="utf-8")
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

        # The initial manifest is written before charts so failures never
        # describe non-existent outputs. Once all ordinary root writes succeed,
        # prepare the complete file inventory without publishing it yet.
        final_manifest = stamp_manifest_schema(
            manifest,
            artifact_files=POLICY_ARTIFACT_FILENAMES,
        )
        final_manifest["artifacts"] = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
            for path in sorted(destination.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "manifest.json"
        }
        if analysis_output_profile is not None:
            final_manifest["analysis_output_profile"] = analysis_output_profile
            paths["analysis_estimands"] = (
                analysis_destination / "target_population_estimands.csv"
            )
            paths["analysis_metadata"] = (
                analysis_destination
                / "target_population_estimand_metadata.json"
            )
            if (
                analysis_plan is not None
                and analysis_plan.plan.primary_aggregate_rule is not None
            ):
                paths["analysis_primary_aggregate"] = (
                    analysis_destination / "primary_aggregate.csv"
                )
                paths["analysis_primary_aggregate_metadata"] = (
                    analysis_destination / "primary_aggregate_metadata.json"
                )
            if analysis_binding is not None and (
                analysis_binding.monetary_output_bases
            ):
                paths["analysis_production_monetary_estimates"] = (
                    analysis_destination
                    / "production_monetary_estimates.csv"
                )
                paths["analysis_production_monetary_metadata"] = (
                    analysis_destination
                    / "production_monetary_metadata.json"
                )

        actual = {path.name for path in paths.values()}
        expected = set(POLICY_ARTIFACT_FILENAMES)
        if analysis_output_profile is not None:
            expected.update(analysis_output_profile["artifact_files"])
        if actual != expected:
            raise RuntimeError(
                "exported artifact set differs: "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

        published_analysis = False
        if analysis_stage is not None:
            _reject_existing_analysis_output(analysis_destination)
            _verify_staged_analysis_identities(
                analysis_stage,
                expected_parent=destination.parent,
                expected_file_identities=analysis_file_identities,
            )
            assert analysis_plan is not None
            assert analysis_binding is not None
            _reattest_analysis_for_publication(
                analysis_plan,
                analysis_binding,
                batch,
            )
            os.replace(analysis_stage, analysis_destination)
            analysis_stage = None
            published_analysis = True
        try:
            paths["manifest"] = write_json_atomic(
                destination / "manifest.json",
                final_manifest,
            )
        except BaseException as error:
            if published_analysis:
                rollback_error: BaseException | None = None
                try:
                    _remove_owned_analysis_directory(
                        analysis_destination,
                        expected_parent=destination,
                        require_complete=True,
                        expected_file_identities=analysis_file_identities,
                    )
                except BaseException as cleanup_error:
                    rollback_error = cleanup_error
                try:
                    write_text_atomic(
                        destination / "manifest.json",
                        interim_manifest_text,
                    )
                except BaseException as restore_error:
                    if rollback_error is None:
                        rollback_error = restore_error
                    else:
                        rollback_error.add_note(
                            "restoring the non-claiming interim manifest also "
                            f"failed: {restore_error}"
                        )
                if rollback_error is not None:
                    rollback_error.add_note(
                        "failed while rolling back a newly published "
                        "prospective-analysis profile"
                    )
                    raise rollback_error from error
            raise
        return paths
    finally:
        if analysis_stage is not None:
            _remove_owned_analysis_directory(
                analysis_stage,
                expected_parent=destination.parent,
                require_complete=False,
            )


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


def _reject_existing_analysis_output(path: Path) -> None:
    """Require a fresh optional-profile target without following leaf links."""

    if os.path.lexists(path):
        raise FileExistsError(
            "prospective analysis output target already exists; choose a fresh "
            "output directory or remove the exact target after reviewing it: "
            f"{path}"
        )


def _verify_staged_analysis_identities(
    path: Path,
    *,
    expected_parent: Path,
    expected_file_identities: Mapping[str, tuple[int, str]],
) -> None:
    """Recheck every staged byte immediately before atomic publication."""

    target_absolute = os.path.normcase(os.path.abspath(os.fspath(path)))
    parent_absolute = os.path.normcase(
        os.path.abspath(os.fspath(expected_parent))
    )
    if os.path.dirname(target_absolute) != parent_absolute:
        raise RuntimeError(
            "refusing prospective-analysis publication outside its expected parent"
        )
    expected_names = set(expected_file_identities)
    allowed_contracts = {
        frozenset(TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES),
        frozenset(PROSPECTIVE_ANALYSIS_ARTIFACT_FILENAMES),
        frozenset(
            TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES
            + PRODUCTION_MONETARY_ARTIFACT_FILENAMES
        ),
        frozenset(
            PROSPECTIVE_ANALYSIS_ARTIFACT_FILENAMES
            + PRODUCTION_MONETARY_ARTIFACT_FILENAMES
        ),
    }
    if frozenset(expected_names) not in allowed_contracts:
        raise RuntimeError(
            "staged prospective-analysis identities do not cover the exact "
            "artifact contract"
        )
    if not os.path.lexists(path):
        raise RuntimeError("staged prospective-analysis directory disappeared")
    target_status = path.lstat()
    if stat.S_ISLNK(target_status.st_mode) or not stat.S_ISDIR(
        target_status.st_mode
    ):
        raise RuntimeError(
            "staged prospective-analysis target is not an owned directory"
        )
    children = sorted(path.iterdir(), key=lambda item: item.name)
    if {child.name for child in children} != expected_names or len(
        children
    ) != len(expected_names):
        raise RuntimeError(
            "staged prospective-analysis artifact set changed before publication"
        )
    for child in children:
        child_status = child.lstat()
        if stat.S_ISLNK(child_status.st_mode) or not stat.S_ISREG(
            child_status.st_mode
        ):
            raise RuntimeError(
                "staged prospective-analysis artifact became a link or non-file"
            )
        expected_size, expected_digest = expected_file_identities[child.name]
        if (
            child_status.st_size != expected_size
            or _digest(child) != expected_digest
        ):
            raise RuntimeError(
                "staged prospective-analysis artifact identity changed before "
                f"publication: {child.name}"
            )


def _reattest_analysis_for_publication(
    analysis_plan: LoadedProspectiveAnalysisPlan,
    analysis_binding: RunAnalysisBinding,
    batch: PolicyBatchResult,
) -> None:
    """Reopen plan/evidence inputs after root writers and before publication."""

    verified_plan = verify_loaded_prospective_analysis_plan(analysis_plan)
    if verified_plan != analysis_plan:
        raise RuntimeError(
            "prospective analysis plan changed before publication"
        )
    observed_binding = resolve_run_analysis_binding(verified_plan.plan, batch)
    if observed_binding != analysis_binding:
        raise RuntimeError(
            "prospective analysis binding changed before publication"
        )


def _remove_owned_analysis_directory(
    path: Path,
    *,
    expected_parent: Path,
    require_complete: bool,
    expected_file_identities: Mapping[str, tuple[int, str]] | None = None,
) -> None:
    """Remove only an exact two-file directory created by this export call."""

    target_absolute = os.path.normcase(os.path.abspath(os.fspath(path)))
    parent_absolute = os.path.normcase(
        os.path.abspath(os.fspath(expected_parent))
    )
    if os.path.dirname(target_absolute) != parent_absolute:
        raise RuntimeError(
            "refusing prospective-analysis cleanup outside its expected parent"
        )
    if not os.path.lexists(path):
        return
    target_status = path.lstat()
    if stat.S_ISLNK(target_status.st_mode) or not stat.S_ISDIR(
        target_status.st_mode
    ):
        raise RuntimeError(
            "refusing prospective-analysis cleanup of a non-directory or link"
        )

    allowed_names = set(
        PROSPECTIVE_ANALYSIS_ARTIFACT_FILENAMES
        + PRODUCTION_MONETARY_ARTIFACT_FILENAMES
    )
    expected_names = (
        set(expected_file_identities)
        if expected_file_identities is not None
        else allowed_names
    )
    if not expected_names.issubset(allowed_names):
        raise RuntimeError(
            "refusing prospective-analysis cleanup with an unknown artifact contract"
        )
    children = sorted(path.iterdir(), key=lambda item: item.name)
    observed_names = {child.name for child in children}
    valid_names = (
        observed_names == expected_names
        if require_complete
        else observed_names.issubset(expected_names)
    )
    if not valid_names or len(observed_names) != len(children):
        raise RuntimeError(
            "refusing prospective-analysis cleanup with unexpected contents"
        )
    if expected_file_identities is not None and (
        set(expected_file_identities) != expected_names
    ):
        raise RuntimeError(
            "refusing prospective-analysis cleanup without complete owned-file "
            "identities"
        )
    for child in children:
        child_status = child.lstat()
        if stat.S_ISLNK(child_status.st_mode) or not stat.S_ISREG(
            child_status.st_mode
        ):
            raise RuntimeError(
                "refusing prospective-analysis cleanup of linked or non-file "
                "contents"
            )
        if expected_file_identities is not None:
            expected_size, expected_digest = expected_file_identities[child.name]
            if (
                child_status.st_size != expected_size
                or _digest(child) != expected_digest
            ):
                raise RuntimeError(
                    "refusing prospective-analysis cleanup after owned-file "
                    "identity changed"
                )
    for child in children:
        child.unlink()
    path.rmdir()


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["export_policy_batch", "render_human_summary"]
