"""Structured provenance metadata for synthetic policy runs."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

import numpy as np

from ..causal.batch import PolicyBatchResult
from ..policy_config import PolicyPrototypeConfig


def build_run_manifest(
    config: PolicyPrototypeConfig,
    batch: PolicyBatchResult,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    created_utc: str | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    """Build a self-contained run manifest without claiming empirical validity."""

    config_file = Path(config_path).resolve()
    repository = Path(repository_root).resolve()
    source_registry = repository / "data" / "provenance" / "sources.toml"
    git_commit, git_dirty = _git_state(repository)
    return {
        "run_name": config.name,
        "created_utc": created_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance_status": config.provenance_status,
        "synthetic_only": True,
        "empirical_validation_claimed": False,
        "notes": config.notes,
        "config_path": str(config_file),
        "config_sha256": _file_digest(config_file),
        "source_registry_sha256": (
            _file_digest(source_registry) if source_registry.exists() else None
        ),
        "repository": {
            "root": str(repository),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "command": list(command) if command is not None else None,
        "batch": {
            "seeds": list(config.batch.seeds),
            "days": config.batch.days,
            "player_count": config.batch.player_count,
            "step_minutes": config.batch.decision_parameters.step_minutes,
            "reference_scenario": config.batch.reference_scenario.value,
            "cohort_digest_by_seed": {
                str(seed): digest
                for seed, digest in batch.cohort_digest_by_seed.items()
            },
        },
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id.value,
                "label": scenario.label,
                "description": scenario.description,
                "mechanics": asdict(scenario.mechanics),
                "fixed_access_price_cents": scenario.fixed_access_price_cents,
                "subscription_price_cents": scenario.subscription_price_cents,
                "epgc_enabled": scenario.epgc_enabled,
            }
            for scenario in config.batch.scenarios
        ],
        "decision_parameters": asdict(config.batch.decision_parameters),
        "harm": {
            "equation": "H_i = w_M*M_i + w_T*OC_i + w_S*S_i + w_E*E_i + w_F*F_i + w_W*W_i",
            "parameters": asdict(config.harm_parameters),
            "weights": asdict(config.harm_weights),
            "opportunity_valuation": asdict(config.opportunity_valuation),
        },
        "producer_assumptions": asdict(config.producer_assumptions),
        "epgc_policy": asdict(config.epgc_policy),
        "random_stream_contract": {
            "generator": "CounterRNG/SplitMix64",
            "coordinates": ["seed", "player_id", "tick", "stream", "draw_index"],
            "named_streams": [
                "player-life:*",
                "policy:life-action-gumbel",
                "policy:reward-prediction-error",
                "policy:purchase-revenue-source",
                "policy:access-adoption",
                "policy:access-plan",
            ],
        },
        "scope_limits": [
            "All population, behavioural, harm, cost, and financing values are synthetic.",
            "The outputs identify effects inside the structural model, not real-world causal effects.",
            "No clinical diagnosis, real-user profiling, targeting, or data collection is implemented.",
            "Public financing is a policy simulation, not a legal conclusion or subsidy application.",
        ],
    }


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repository: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit or None, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


__all__ = ["build_run_manifest"]
