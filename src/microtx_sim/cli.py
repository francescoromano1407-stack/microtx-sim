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
    )
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
        # The same complete identity is rebuilt after all model work and before
        # any result is published.  A future receipt schema may open the gate;
        # schema v1 cannot reach this branch because its pre-execution gate is
        # intentionally fixed closed.
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
    repository_root = Path(__file__).resolve().parents[2]
    destination = _resolve_output(
        output if output is not None else config.output.output_dir,
        repository_root=repository_root,
    )
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
    )
    return {
        "status": "ok",
        "mode": "campaign_policy_batch" if campaign else "synthetic_policy_batch",
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
