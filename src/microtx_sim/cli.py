from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from typing import Sequence

from .analysis import run_sensitivity_analysis
from .causal.batch import run_policy_batch
from .config import ConfigurationError, load_config
from .core.ledger import LedgerStorageError
from .core.world import World
from .data.profiles import ProfileBundle, ProfileValidationError, load_profile_bundle
from .outputs import export_policy_batch
from .outputs.schema import (
    SENSITIVITY_COLUMNS,
    stamp_standalone_sensitivity_schema,
)
from .outputs.writers import write_csv_atomic, write_json_atomic
from .policy_config import PolicyConfigurationError, load_policy_config
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
    return {
        "status": "ok",
        "scenario": config.meta.name,
        "scenario_provenance": config.meta.provenance_status.value,
        "profile_provenance": profiles.profile_status.value,
        "jurisdictions": [profile.code for profile in profiles.country_profiles],
        "source_records": len(profiles.sources),
        "campaign_ready": False,
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
    return {
        "status": "ok",
        "mode": "synthetic_policy_prototype",
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
    }


def _policy_batch(
    config_path: Path,
    *,
    output: Path | None,
    run_sensitivity: bool | None,
    command: Sequence[str],
) -> dict[str, object]:
    config = load_policy_config(config_path)
    sensitivity_enabled = (
        config.output.run_sensitivity
        if run_sensitivity is None
        else run_sensitivity
    )
    profiles = load_profile_bundle(campaign=False)
    batch = run_policy_batch(
        config.batch,
        profile_bundle=profiles,
        harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
    )
    sensitivity = (
        _run_configured_sensitivity(config, profile_bundle=profiles)
        if sensitivity_enabled
        else None
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
    )
    return {
        "status": "ok",
        "mode": "synthetic_policy_batch",
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
    }


def _policy_sensitivity(
    config_path: Path,
    *,
    output: Path | None,
) -> dict[str, object]:
    config = load_policy_config(config_path)
    profiles = load_profile_bundle(campaign=False)
    result = _run_configured_sensitivity(config, profile_bundle=profiles)
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
        }),
    )
    return {
        "status": "ok",
        "mode": "synthetic_sensitivity",
        "rows": len(result.rows),
        "unstable_parameters": list(result.unstable_parameters),
        "artifacts": [str(csv_path.resolve()), str(metadata_path.resolve())],
    }


def _run_configured_sensitivity(
    config,
    *,
    profile_bundle: ProfileBundle | None = None,
):
    return run_sensitivity_analysis(
        config.batch,
        profile_bundle=profile_bundle,
        base_harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
    )


def _resolve_output(path: Path, *, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            payload = _validate(args.config)
        elif args.command == "smoke":
            payload = _smoke(args.config)
        elif args.command == "policy-validate":
            payload = _policy_validate(args.config)
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
