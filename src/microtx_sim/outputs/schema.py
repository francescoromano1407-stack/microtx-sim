"""Stable, machine-readable contracts for exported simulation results.

Version 2 exhaustively declared every policy-table column.  Version 3 preserves
those filenames and columns exactly and introduces a separately versioned
manifest envelope.  Output-v2 manifests had no independent manifest version;
they are legacy artifacts, not payloads that current writers may relabel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Final

from ..metrics.reporting import REPEATED_SEED_METRIC_STEMS


LEGACY_OUTPUT_SCHEMA_VERSION: Final[str] = "2.0"
OUTPUT_SCHEMA_VERSION: Final[str] = "3.0"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.0"
STANDALONE_SENSITIVITY_PROFILE: Final[str] = "standalone_sensitivity"
STANDALONE_SENSITIVITY_SCHEMA_VERSION: Final[str] = "1.0"

SEED_RESULT_V1_PREFIX_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "scenario_label",
    "seed",
    "cohort_digest",
    "days",
    "player_count",
    "total_revenue_cents",
    "producer_cost_cents",
    "producer_profit_cents",
    "revenue_direct_purchase_cents",
    "revenue_opaque_virtual_currency_cents",
    "revenue_paid_random_rewards_cents",
    "revenue_fixed_price_cents",
    "revenue_subscription_cents",
    "revenue_public_contract_cents",
    "revenue_institutional_licensing_cents",
    "revenue_non_targeted_sponsorship_cents",
    "total_spending_cents",
    "harmful_spending_cents",
    "unplanned_spending_cents",
    "mean_harm",
    "harm_p50",
    "harm_p90",
    "total_opportunity_cost_proxy_cents",
    "mean_opportunity_cost_score",
    "mean_sleep_burden",
    "mean_education_work_burden",
    "mean_social_burden",
    "mean_wellbeing_burden",
    "mean_enjoyment",
    "high_risk_count",
    "high_risk_share",
    "mean_harm_effect_vs_safe",
    "epgc_minimum_public_contribution_cents",
)

_SEED_RESULT_V2_EXTENSION_COLUMNS: Final[tuple[str, ...]] = (
    "adult_opportunity_cost_proxy_cents",
    "epgc_profit_safe_cents",
    "harm_p10",
    "harm_variance_players",
    "harmful_spending_effect_vs_safe_cents",
    "high_risk_mean_age",
    "high_risk_mean_baseline_vulnerability",
    "high_risk_mean_budget_cents",
    "high_risk_minor_share",
    "mean_monetary_harm",
    "spend_p10_cents",
    "spend_p50_cents",
    "spend_p90_cents",
    "total_revenue_effect_vs_safe_cents",
    "total_spending_effect_vs_safe_cents",
    "youth_opportunity_cost_proxy_cents",
)

SEED_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    SEED_RESULT_V1_PREFIX_COLUMNS + _SEED_RESULT_V2_EXTENSION_COLUMNS
)

SCENARIO_SUMMARY_V1_PREFIX_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "scenario_label",
    "seed_count",
    "player_count",
    "days",
    "total_revenue_cents_mean",
    "total_revenue_cents_sd",
    "total_spending_cents_mean",
    "harmful_spending_cents_mean",
    "mean_harm_mean",
    "mean_harm_sd",
    "mean_harm_ci95_low",
    "mean_harm_ci95_high",
    "mean_harm_effect_vs_safe_mean",
    "total_opportunity_cost_proxy_cents_mean",
    "mean_opportunity_cost_score_mean",
    "mean_sleep_burden_mean",
    "mean_education_work_burden_mean",
    "mean_social_burden_mean",
    "mean_wellbeing_burden_mean",
    "mean_enjoyment_mean",
    "high_risk_count_mean",
    "epgc_minimum_public_contribution_cents_mean",
)

_UNCERTAINTY_SUFFIXES: Final[tuple[str, ...]] = (
    "mean",
    "variance",
    "sd",
    "ci95_low",
    "ci95_high",
)

_SCENARIO_SUMMARY_V1_DERIVED: Final[frozenset[str]] = frozenset(
    f"{metric}_{suffix}"
    for metric in REPEATED_SEED_METRIC_STEMS
    for suffix in _UNCERTAINTY_SUFFIXES
)

SCENARIO_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    SCENARIO_SUMMARY_V1_PREFIX_COLUMNS
    + tuple(
        sorted(
            _SCENARIO_SUMMARY_V1_DERIVED.difference(
                SCENARIO_SUMMARY_V1_PREFIX_COLUMNS
            )
        )
    )
)

EPGC_FINANCING_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "seed",
    "public_contract_revenue_cents",
    "minimum_public_contribution_cents",
    "maximum_budget_cents",
    "profit_safe_cents",
    "feasible_under_budget_cap",
    "sustainable_under_policy",
    "clawback_cents",
    "penalty_cents",
)

SENSITIVITY_V1_PREFIX_COLUMNS: Final[tuple[str, ...]] = (
    "parameter",
    "parameter_value",
    "scenario_id",
    "seed_count",
    "mean_harm",
    "harm_variance",
    "harm_sd",
    "harm_ci95_low",
    "harm_ci95_high",
    "total_revenue_cents",
    "opportunity_cost_burden",
    "minimum_public_contribution_cents",
    "expected_direction",
    "monotonic_observed",
    "unstable",
)

_SENSITIVITY_V2_EXTENSION_COLUMNS: Final[tuple[str, ...]] = (
    "harm_coefficient_of_variation",
    "monotonic_expected",
)

SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    SENSITIVITY_V1_PREFIX_COLUMNS + _SENSITIVITY_V2_EXTENSION_COLUMNS
)

PLAYER_OUTCOME_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "seed",
    "player_id",
    "age_years",
    "is_minor",
    "baseline_vulnerability",
    "spending_cents",
    "harmful_spending_cents",
    "composite_harm",
    "monetary_harm",
    "opportunity_cost",
    "sleep_burden",
    "education_work_burden",
    "social_burden",
    "wellbeing_burden",
    "opportunity_cost_proxy_cents",
    "enjoyment",
    "high_risk",
)

OPPORTUNITY_DECOMPOSITION_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "component",
    "mean_minutes",
    "mean_burden",
    "monetary_proxy_cents",
)

TABLE_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "seed_results.csv": SEED_RESULT_COLUMNS,
        "scenario_summary.csv": SCENARIO_SUMMARY_COLUMNS,
        "epgc_financing.csv": EPGC_FINANCING_COLUMNS,
        "sensitivity.csv": SENSITIVITY_COLUMNS,
        "player_outcomes.csv": PLAYER_OUTCOME_COLUMNS,
        "opportunity_cost_decomposition.csv": (
            OPPORTUNITY_DECOMPOSITION_COLUMNS
        ),
    }
)

ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "seed_results.csv",
    "scenario_summary.csv",
    "epgc_financing.csv",
    "sensitivity.csv",
    "manifest.json",
)

POLICY_ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "seed_results.csv",
    "scenario_summary.csv",
    "player_outcomes.csv",
    "opportunity_cost_decomposition.csv",
    "epgc_financing.csv",
    "sensitivity.csv",
    "manifest.json",
    "summary.md",
    "harm_distribution.svg",
    "spending_distribution.svg",
    "harm_revenue_frontier.svg",
    "opportunity_cost_decomposition.svg",
    "epgc_subsidy_requirement.svg",
)

STANDALONE_SENSITIVITY_ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "sensitivity.csv",
    "sensitivity_metadata.json",
)


def manifest_schema_descriptor() -> dict[str, object]:
    """Return the deterministic contract for the versioned manifest envelope.

    Domain payloads inside a manifest retain their own schema/version contracts.
    This descriptor owns the four fields which identify the on-disk output and
    manifest contracts, while deliberately allowing additional domain metadata.
    """

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "legacy_output_schema_versions_without_manifest_schema": ["1.0", "2.0"],
        "required_fields": [
            "artifact_files",
            "manifest_schema_sha256",
            "manifest_schema_version",
            "output_schema_version",
        ],
        "reserved_fields": {
            "artifact_files": {
                "type": "array[string]",
                "allowed_exact_values": [
                    list(ARTIFACT_FILENAMES),
                    list(POLICY_ARTIFACT_FILENAMES),
                ],
            },
            "manifest_schema_sha256": {
                "type": "lowercase-sha256",
                "value": "sha256(canonical manifest schema descriptor)",
            },
            "manifest_schema_version": {
                "type": "string",
                "exact_value": MANIFEST_SCHEMA_VERSION,
            },
            "output_schema_version": {
                "type": "string",
                "exact_value": OUTPUT_SCHEMA_VERSION,
            },
        },
        "additional_domain_metadata_allowed": True,
    }


def _manifest_schema_digest() -> str:
    encoded = json.dumps(
        manifest_schema_descriptor(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


MANIFEST_SCHEMA_SHA256: Final[str] = _manifest_schema_digest()

_MANIFEST_RESERVED_FIELDS = frozenset(
    {
        "artifact_files",
        "manifest_schema_sha256",
        "manifest_schema_version",
        "output_schema_version",
    }
)


def stamp_manifest_schema(
    manifest: Mapping[str, object],
    *,
    artifact_files: Sequence[str],
) -> dict[str, object]:
    """Validate and stamp the reserved manifest-version fields.

    Missing reserved fields mean the caller supplied unversioned metadata for a
    new bundle.  Explicit legacy or conflicting declarations are rejected; the
    writer never upgrades an already-versioned output-v2 manifest in place.
    """

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    selected_files = tuple(artifact_files)
    if selected_files not in {ARTIFACT_FILENAMES, POLICY_ARTIFACT_FILENAMES}:
        raise ValueError("artifact_files must match a registered output profile")

    payload = dict(manifest)
    unknown_manifest_fields = sorted(
        key
        for key in payload
        if isinstance(key, str)
        and key.startswith("manifest_schema_")
        and key not in _MANIFEST_RESERVED_FIELDS
    )
    if unknown_manifest_fields:
        raise ValueError(
            "unknown manifest schema fields: " + ", ".join(unknown_manifest_fields)
        )
    present_reserved_fields = _MANIFEST_RESERVED_FIELDS.intersection(payload)
    if present_reserved_fields and present_reserved_fields != _MANIFEST_RESERVED_FIELDS:
        missing = sorted(_MANIFEST_RESERVED_FIELDS.difference(payload))
        raise ValueError(
            "manifest reserved schema fields must be supplied together; missing "
            + ", ".join(missing)
        )

    declared_output_version = payload.get("output_schema_version")
    if present_reserved_fields and declared_output_version != OUTPUT_SCHEMA_VERSION:
        raise ValueError(
            "manifest output_schema_version conflicts with the writer schema"
        )
    declared_manifest_version = payload.get("manifest_schema_version")
    if present_reserved_fields and declared_manifest_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "manifest_schema_version conflicts with the writer schema"
        )
    declared_manifest_digest = payload.get("manifest_schema_sha256")
    if present_reserved_fields and declared_manifest_digest != MANIFEST_SCHEMA_SHA256:
        raise ValueError("manifest_schema_sha256 conflicts with the writer schema")
    declared_files = payload.get("artifact_files")
    if present_reserved_fields and declared_files not in (
        selected_files, list(selected_files)
    ):
        raise ValueError("manifest artifact_files conflicts with stable filenames")

    payload["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    payload["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    payload["manifest_schema_sha256"] = MANIFEST_SCHEMA_SHA256
    payload["artifact_files"] = list(selected_files)
    return payload


def standalone_sensitivity_schema_descriptor() -> dict[str, object]:
    """Return the contract for the two-file standalone sensitivity profile."""

    return {
        "output_profile": STANDALONE_SENSITIVITY_PROFILE,
        "output_profile_schema_version": STANDALONE_SENSITIVITY_SCHEMA_VERSION,
        "artifact_files": list(STANDALONE_SENSITIVITY_ARTIFACT_FILENAMES),
        "table_columns": {"sensitivity.csv": list(SENSITIVITY_COLUMNS)},
        "metadata_filename": "sensitivity_metadata.json",
        "additional_metadata_allowed": True,
    }


def _standalone_sensitivity_schema_digest() -> str:
    encoded = json.dumps(
        standalone_sensitivity_schema_descriptor(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


STANDALONE_SENSITIVITY_SCHEMA_SHA256: Final[str] = (
    _standalone_sensitivity_schema_digest()
)


def stamp_standalone_sensitivity_schema(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Stamp metadata for the standalone sensitivity output profile.

    This profile is not a complete output-v3 bundle and must not carry the full
    bundle's ``output_schema_version`` or manifest envelope.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("sensitivity metadata must be a mapping")
    payload = dict(metadata)
    forbidden_full_bundle_fields = sorted(
        key
        for key in payload
        if isinstance(key, str)
        and (
            key == "output_schema_version"
            or key.startswith("manifest_schema_")
        )
    )
    if forbidden_full_bundle_fields:
        raise ValueError(
            "standalone sensitivity metadata cannot claim full-bundle fields: "
            + ", ".join(forbidden_full_bundle_fields)
        )
    reserved = {
        "artifact_files": list(STANDALONE_SENSITIVITY_ARTIFACT_FILENAMES),
        "output_profile": STANDALONE_SENSITIVITY_PROFILE,
        "output_profile_schema_sha256": STANDALONE_SENSITIVITY_SCHEMA_SHA256,
        "output_profile_schema_version": STANDALONE_SENSITIVITY_SCHEMA_VERSION,
    }
    present = set(reserved).intersection(payload)
    unknown_profile_schema_fields = sorted(
        key
        for key in payload
        if isinstance(key, str)
        and key.startswith("output_profile_schema_")
        and key not in reserved
    )
    if unknown_profile_schema_fields:
        raise ValueError(
            "unknown standalone sensitivity schema fields: "
            + ", ".join(unknown_profile_schema_fields)
        )
    if present and present != set(reserved):
        raise ValueError(
            "standalone sensitivity schema fields must be supplied together"
        )
    if present and any(payload[key] != value for key, value in reserved.items()):
        raise ValueError("standalone sensitivity schema fields conflict")
    payload.update(reserved)
    return payload


__all__ = [
    "ARTIFACT_FILENAMES",
    "EPGC_FINANCING_COLUMNS",
    "LEGACY_OUTPUT_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_SHA256",
    "MANIFEST_SCHEMA_VERSION",
    "OUTPUT_SCHEMA_VERSION",
    "OPPORTUNITY_DECOMPOSITION_COLUMNS",
    "PLAYER_OUTCOME_COLUMNS",
    "POLICY_ARTIFACT_FILENAMES",
    "REPEATED_SEED_METRIC_STEMS",
    "SCENARIO_SUMMARY_COLUMNS",
    "SCENARIO_SUMMARY_V1_PREFIX_COLUMNS",
    "SEED_RESULT_COLUMNS",
    "SEED_RESULT_V1_PREFIX_COLUMNS",
    "SENSITIVITY_COLUMNS",
    "SENSITIVITY_V1_PREFIX_COLUMNS",
    "STANDALONE_SENSITIVITY_ARTIFACT_FILENAMES",
    "STANDALONE_SENSITIVITY_PROFILE",
    "STANDALONE_SENSITIVITY_SCHEMA_SHA256",
    "STANDALONE_SENSITIVITY_SCHEMA_VERSION",
    "TABLE_COLUMNS",
    "manifest_schema_descriptor",
    "standalone_sensitivity_schema_descriptor",
    "stamp_manifest_schema",
    "stamp_standalone_sensitivity_schema",
]
