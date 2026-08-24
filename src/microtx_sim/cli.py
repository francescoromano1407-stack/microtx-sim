from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .config import ConfigurationError, load_config
from .core.engine import SimulationEngine
from .core.world import World
from .data.profiles import ProfileValidationError, load_profile_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="microtx-sim",
        description="Scheletro causale agent-based per il mercato mobile.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser(
        "validate", help="valida configurazione e contratti delle fonti"
    )
    validate.add_argument("config", type=Path)

    smoke = commands.add_parser(
        "smoke", help="esegue soltanto il controllo strutturale breve"
    )
    smoke.add_argument("config", type=Path)
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
    world = World.create(config, campaign=False)
    result = SimulationEngine.run(world, campaign=False)
    return {
        "status": "ok",
        "mode": "smoke_only",
        "scenario": config.meta.name,
        "cycles": result.cycles,
        "elapsed_seconds": round(result.elapsed_seconds, 6),
        "summary": result.summary,
        "audit_count": sum(
            len(step.audit_resolutions) for step in world.step_history
        ),
        "ledger_entries": len(world.ledger.entries),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = (
            _validate(args.config)
            if args.command == "validate"
            else _smoke(args.config)
        )
    except (ConfigurationError, ProfileValidationError, OSError, ValueError) as exc:
        print(f"errore: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
