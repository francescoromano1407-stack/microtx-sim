"""Structured provenance metadata for synthetic policy runs."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

import numpy as np

from ..causal.batch import (
    PolicyBatchResult,
    PolicyRunInputs,
    resolve_policy_run_inputs,
)
from ..causal.design import assess_causal_design
from ..policy_config import PolicyPrototypeConfig
from .metric_contracts import build_metric_contract_manifest_payload


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
    configured_run_inputs = resolve_policy_run_inputs(
        harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
    )
    _validate_config_matches_batch(config, batch, configured_run_inputs)
    git_commit, git_dirty = _git_state(repository)
    profile_lineage = batch.profile_input_lineage
    if profile_lineage is None:
        profile_inputs: dict[str, object] = {
            "lineage_status": "unavailable_legacy_result",
            "profile_codes": [profile.code for profile in batch.country_profiles],
            "fingerprint_sha256": None,
            "snapshot": None,
            "jurisdictions": {"path": None, "sha256": None},
            "source_registry": {
                "path": None,
                "sha256": None,
                "retrieved_on": None,
            },
            "metric_contract_summary": {"count": 0, "status_counts": {}},
            "money_scale_summary": {
                "count": 0,
                "currencies": [],
                "anchor_status_counts": {},
                "scale_status_counts": {},
            },
        }
    else:
        profile_inputs = profile_lineage.manifest_payload()
    run_input_snapshot = batch.run_input_snapshot()
    run_input_sha256 = batch.run_input_sha256()
    causal_design = assess_causal_design(
        batch.spec.scenarios
    ).manifest_payload(run_input_sha256=run_input_sha256)
    effective_config_snapshot = _effective_config_snapshot(
        config,
        configured_run_inputs,
    )
    config_file_sha256 = _file_digest(config_file)
    jurisdictions_metadata = profile_inputs["jurisdictions"]
    source_registry_metadata = profile_inputs["source_registry"]
    if not isinstance(jurisdictions_metadata, dict) or not isinstance(
        source_registry_metadata, dict
    ):
        raise ValueError("profile input file lineage is malformed")
    output_metric_contracts = build_metric_contract_manifest_payload(
        configuration_status=config.provenance_status,
        profile_lineage_status=str(profile_inputs["lineage_status"]),
        profile_dependencies_calibrated=(
            _profile_dependencies_calibrated(profile_inputs)
        ),
        profile_input_fingerprint_sha256=(
            str(profile_inputs["fingerprint_sha256"])
            if profile_inputs["fingerprint_sha256"] is not None
            else None
        ),
        run_source_retrieved_on=(
            profile_lineage.source_retrieved_on
            if profile_lineage is not None
            else None
        ),
        monetary_outputs_cross_country_comparable=(
            _money_outputs_cross_country_comparable(profile_inputs)
        ),
        run_input_sha256=run_input_sha256,
    )
    return {
        "run_name": config.name,
        "created_utc": created_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance_status": config.provenance_status,
        "synthetic_only": True,
        "empirical_validation_claimed": False,
        "notes": config.notes,
        "config_path": str(config_file),
        "config_sha256": config_file_sha256,
        "config_file_sha256_observed_at_export": config_file_sha256,
        "config_file_attestation": (
            "export-time observation only; execution truth is run_input_sha256"
        ),
        "effective_config_sha256": _canonical_sha256(
            effective_config_snapshot
        ),
        "config_snapshot": effective_config_snapshot,
        "run_input_sha256": run_input_sha256,
        "run_input_snapshot": run_input_snapshot,
        "jurisdictions_sha256": jurisdictions_metadata.get("sha256"),
        "source_registry_sha256": source_registry_metadata.get("sha256"),
        "source_registry_retrieved_on": source_registry_metadata.get(
            "retrieved_on"
        ),
        "profile_inputs": profile_inputs,
        "causal_design": causal_design,
        "output_metric_contracts": output_metric_contracts,
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
            "seeds": list(batch.spec.seeds),
            "seed_decimal_strings": [str(seed) for seed in batch.spec.seeds],
            "days": batch.spec.days,
            "player_count": batch.spec.player_count,
            "step_minutes": batch.spec.decision_parameters.step_minutes,
            "reference_scenario": batch.spec.reference_scenario.value,
            "profile_codes": list(profile_inputs["profile_codes"]),
            "profile_input_fingerprint_sha256": profile_inputs[
                "fingerprint_sha256"
            ],
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
            for scenario in batch.spec.scenarios
        ],
        "decision_parameters": asdict(batch.spec.decision_parameters),
        "harm": {
            "equation": "H_i = w_M*M_i + w_T*OC_i + w_S*S_i + w_E*E_i + w_F*F_i + w_W*W_i",
            "parameters": asdict(batch.run_inputs.harm_parameters),
            "weights": asdict(batch.run_inputs.harm_weights),
            "opportunity_valuation": asdict(
                batch.run_inputs.opportunity_valuation
            ),
        },
        "producer_assumptions": asdict(batch.run_inputs.producer_assumptions),
        "epgc_policy": asdict(batch.run_inputs.epgc_policy),
        "random_stream_contract": {
            "generator": "CounterRNG/SplitMix64",
            "coordinates": ["seed", "player_id", "tick", "stream", "draw_index"],
            "root_seed": {
                "accepted_runtime_type": "strict Python int (bool excluded)",
                "minimum_decimal": "0",
                "maximum_decimal": "18446744073709551615",
                "normalization": "none; out-of-range values are rejected",
                "json_exact_encoding": "batch.seed_decimal_strings",
            },
            "batch_seed_order": "unique ascending numeric order",
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


def _validate_config_matches_batch(
    config: PolicyPrototypeConfig,
    batch: PolicyBatchResult,
    configured_run_inputs: PolicyRunInputs,
) -> None:
    """Reject metadata claims that differ from the execution truth."""

    if config.batch != batch.spec:
        raise ValueError(
            "configuration batch specification does not match the executed batch"
        )
    if configured_run_inputs != batch.run_inputs:
        raise ValueError(
            "configuration model inputs do not match the executed batch"
        )


def _effective_config_snapshot(
    config: PolicyPrototypeConfig,
    run_inputs: PolicyRunInputs,
) -> dict[str, object]:
    return {
        "meta": {
            "name": config.name,
            "provenance_status": config.provenance_status,
            "notes": config.notes,
        },
        "batch_spec": config.batch.snapshot(),
        "model_inputs": run_inputs.snapshot(),
        "output": {
            "output_dir": str(config.output.output_dir),
            "histogram_bins": config.output.histogram_bins,
            "include_player_rows": config.output.include_player_rows,
            "run_sensitivity": config.output.run_sensitivity,
        },
    }


def _canonical_sha256(snapshot: dict[str, object]) -> str:
    encoded = json.dumps(
        snapshot,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _money_outputs_cross_country_comparable(
    profile_inputs: dict[str, object],
) -> bool:
    """Require an explicit true claim on every retained money-scale contract."""

    snapshot = profile_inputs.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    bundle = snapshot.get("profile_bundle")
    if not isinstance(bundle, dict):
        return False
    scales = bundle.get("money_scales")
    return bool(scales) and isinstance(scales, list) and all(
        isinstance(scale, dict)
        and scale.get("cross_country_comparable") is True
        for scale in scales
    )


def _profile_dependencies_calibrated(
    profile_inputs: dict[str, object],
) -> bool:
    """Mirror the transitive profile campaign gate from fingerprinted values."""

    if profile_inputs.get("lineage_status") != "registered_profile_bundle":
        return False
    snapshot = profile_inputs.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    bundle = snapshot.get("profile_bundle")
    profiles = snapshot.get("country_profiles")
    if not isinstance(bundle, dict) or not isinstance(profiles, list):
        return False
    if bundle.get("profile_status") != "CALIBRATED":
        return False
    sources = bundle.get("sources")
    contracts = bundle.get("metric_contracts")
    scales = bundle.get("money_scales")
    if not all(isinstance(rows, list) and rows for rows in (sources, contracts, scales)):
        return False
    assert isinstance(sources, list)
    assert isinstance(contracts, list)
    assert isinstance(scales, list)
    if any(
        not isinstance(contract, dict) or contract.get("status") != "CALIBRATED"
        for contract in contracts
    ):
        return False
    if any(
        not isinstance(scale, dict)
        or scale.get("anchor_status") != "CALIBRATED"
        or scale.get("scale_status") != "CALIBRATED"
        for scale in scales
    ):
        return False

    referenced_source_ids: set[str] = set()
    for row in (*profiles, *contracts, *scales):
        if not isinstance(row, dict):
            return False
        source_ids = row.get("source_ids")
        if not isinstance(source_ids, list) or any(
            not isinstance(source_id, str) for source_id in source_ids
        ):
            return False
        referenced_source_ids.update(source_ids)
    source_statuses = {
        source.get("id"): source.get("calibration_status")
        for source in sources
        if isinstance(source, dict)
    }
    return bool(referenced_source_ids) and all(
        source_statuses.get(source_id) == "CALIBRATED"
        for source_id in referenced_source_ids
    )


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
