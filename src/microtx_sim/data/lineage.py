"""Content-addressed lineage for policy population inputs."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..consumers.population import CountryProfile
from .profiles import (
    ProfileBundle,
    ProfileValidationError,
    _is_sha256,
    _parse_iso_date,
    load_profile_bundle,
)


_PROFILE_INPUT_SCHEMA_VERSION = 1
_REGISTERED_PROFILE_LINEAGE = "registered_profile_bundle"
_UNREGISTERED_PROFILE_LINEAGE = "unregistered_custom_profiles"
_UNREGISTERED_BUNDLE_LINEAGE = "unregistered_profile_bundle"


@dataclass(frozen=True, slots=True)
class ProfileInputLineage:
    """Immutable, content-addressed description of policy population inputs."""

    lineage_status: str
    profile_codes: tuple[str, ...]
    fingerprint_sha256: str
    snapshot_json: str
    jurisdictions_path: str | None = None
    jurisdictions_sha256: str | None = None
    source_registry_path: str | None = None
    source_registry_sha256: str | None = None
    source_retrieved_on: date | None = None

    def __post_init__(self) -> None:
        if self.lineage_status not in {
            _REGISTERED_PROFILE_LINEAGE,
            _UNREGISTERED_PROFILE_LINEAGE,
            _UNREGISTERED_BUNDLE_LINEAGE,
        }:
            raise ProfileValidationError("unknown profile lineage status")
        if not self.profile_codes or any(not code.strip() for code in self.profile_codes):
            raise ProfileValidationError("profile lineage needs non-empty profile codes")
        if len(set(self.profile_codes)) != len(self.profile_codes):
            raise ProfileValidationError("profile lineage repeats a profile code")
        if not _is_sha256(self.fingerprint_sha256):
            raise ProfileValidationError("profile fingerprint must be a SHA-256 digest")
        expected = sha256(self.snapshot_json.encode("utf-8")).hexdigest()
        if expected != self.fingerprint_sha256:
            raise ProfileValidationError("profile fingerprint does not match its snapshot")
        try:
            snapshot = json.loads(self.snapshot_json)
        except json.JSONDecodeError as exc:
            raise ProfileValidationError("profile snapshot must be valid JSON") from exc
        if not isinstance(snapshot, dict):
            raise ProfileValidationError("profile snapshot root must be an object")
        if self.snapshot_json != _canonical_snapshot_json(snapshot):
            raise ProfileValidationError("profile snapshot JSON must be canonical")
        if snapshot.get("schema_version") != _PROFILE_INPUT_SCHEMA_VERSION:
            raise ProfileValidationError("unsupported profile snapshot schema version")
        if snapshot.get("lineage_status") != self.lineage_status:
            raise ProfileValidationError(
                "profile snapshot lineage status does not match lineage metadata"
            )
        if snapshot.get("country_profile_type") != (
            "microtx_sim.consumers.population.CountryProfile"
        ):
            raise ProfileValidationError("profile snapshot has an unknown profile type")
        snapshot_profiles = snapshot.get("country_profiles")
        if not isinstance(snapshot_profiles, list) or any(
            not isinstance(profile, dict) for profile in snapshot_profiles
        ):
            raise ProfileValidationError("profile snapshot country profiles are malformed")
        snapshot_codes = tuple(
            str(profile.get("code", "")) for profile in snapshot_profiles
        )
        if snapshot_codes != self.profile_codes:
            raise ProfileValidationError(
                "profile snapshot codes do not match profile lineage metadata"
            )

        file_lineage = snapshot.get("file_lineage")
        if not isinstance(file_lineage, dict):
            raise ProfileValidationError("profile snapshot file lineage is malformed")
        jurisdictions = file_lineage.get("jurisdictions")
        source_registry = file_lineage.get("source_registry")
        if not isinstance(jurisdictions, dict) or not isinstance(
            source_registry, dict
        ):
            raise ProfileValidationError("profile snapshot file lineage is malformed")
        snapshot_jurisdictions_path = jurisdictions.get("path")
        snapshot_jurisdictions_sha256 = jurisdictions.get("sha256")
        snapshot_source_registry_path = source_registry.get("path")
        snapshot_source_registry_sha256 = source_registry.get("sha256")
        snapshot_retrieved_text = source_registry.get("retrieved_on")
        snapshot_retrieved_on = (
            _parse_iso_date(
                snapshot_retrieved_text,
                "profile snapshot source_registry.retrieved_on",
            )
            if snapshot_retrieved_text is not None
            else None
        )
        if (
            snapshot_jurisdictions_path != self.jurisdictions_path
            or snapshot_jurisdictions_sha256 != self.jurisdictions_sha256
            or snapshot_source_registry_path != self.source_registry_path
            or snapshot_source_registry_sha256 != self.source_registry_sha256
            or snapshot_retrieved_on != self.source_retrieved_on
        ):
            raise ProfileValidationError(
                "published profile file lineage does not match its fingerprinted snapshot"
            )

        for name in ("jurisdictions_path", "source_registry_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ProfileValidationError(f"{name} must be non-empty text")
        for name in ("jurisdictions_sha256", "source_registry_sha256"):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise ProfileValidationError(f"{name} is not a SHA-256 digest")
        if self.source_retrieved_on is not None and type(self.source_retrieved_on) is not date:
            raise ProfileValidationError(
                "profile source retrieval metadata must be an ISO calendar date"
            )
        bundle = snapshot.get("profile_bundle")
        if self.lineage_status == _REGISTERED_PROFILE_LINEAGE:
            if not isinstance(bundle, dict) or any(
                value is None
                for value in (
                    self.jurisdictions_path,
                    self.jurisdictions_sha256,
                    self.source_registry_path,
                    self.source_registry_sha256,
                    self.source_retrieved_on,
                )
            ):
                raise ProfileValidationError(
                    "registered profile lineage requires complete loaded-file metadata"
                )
        elif any(
            value is not None
            for value in (
                self.jurisdictions_path,
                self.jurisdictions_sha256,
                self.source_registry_path,
                self.source_registry_sha256,
                self.source_retrieved_on,
            )
        ):
            raise ProfileValidationError(
                "unregistered profiles cannot claim registered file lineage"
            )
        if self.lineage_status == _UNREGISTERED_PROFILE_LINEAGE and bundle is not None:
            raise ProfileValidationError(
                "bare custom profile lineage cannot claim a profile bundle"
            )
        if self.lineage_status == _UNREGISTERED_BUNDLE_LINEAGE and not isinstance(
            bundle, dict
        ):
            raise ProfileValidationError(
                "unregistered profile-bundle lineage requires its bundle snapshot"
            )

    @property
    def snapshot(self) -> dict[str, object]:
        """Return a fresh JSON-compatible copy of the fingerprinted snapshot."""

        value = json.loads(self.snapshot_json)
        if not isinstance(value, dict):
            raise ProfileValidationError("profile snapshot root must be an object")
        return value

    def validate_country_profiles(
        self,
        country_profiles: Sequence[CountryProfile],
    ) -> None:
        """Require retained profiles to equal the fingerprinted values exactly."""

        profiles = tuple(country_profiles)
        if not profiles or any(
            not isinstance(profile, CountryProfile) for profile in profiles
        ):
            raise ProfileValidationError(
                "profile lineage requires retained CountryProfile values"
            )
        expected = self.snapshot.get("country_profiles")
        actual = [_snapshot_dataclass(profile) for profile in profiles]
        if expected != actual:
            raise ProfileValidationError(
                "retained country profiles do not match the fingerprinted snapshot"
            )

    def manifest_payload(self) -> dict[str, object]:
        """Return deterministic run-manifest metadata without status promotion."""

        snapshot = self.snapshot
        bundle = snapshot.get("profile_bundle")
        bundle_payload = bundle if isinstance(bundle, dict) else {}
        metric_contracts = bundle_payload.get("metric_contracts", [])
        money_scales = bundle_payload.get("money_scales", [])
        if not isinstance(metric_contracts, list) or not isinstance(money_scales, list):
            raise ProfileValidationError("profile snapshot contract tables are malformed")
        file_lineage = snapshot.get("file_lineage")
        if not isinstance(file_lineage, dict):
            raise ProfileValidationError("profile snapshot file lineage is malformed")
        return {
            "lineage_status": self.lineage_status,
            "profile_codes": list(self.profile_codes),
            "fingerprint_sha256": self.fingerprint_sha256,
            "snapshot": snapshot,
            "jurisdictions": dict(file_lineage["jurisdictions"]),
            "source_registry": dict(file_lineage["source_registry"]),
            "metric_contract_summary": {
                "count": len(metric_contracts),
                "status_counts": _status_counts(metric_contracts, "status"),
            },
            "money_scale_summary": {
                "count": len(money_scales),
                "currencies": sorted(
                    {
                        str(item["currency"])
                        for item in money_scales
                        if isinstance(item, dict) and "currency" in item
                    }
                ),
                "anchor_status_counts": _status_counts(
                    money_scales, "anchor_status"
                ),
                "scale_status_counts": _status_counts(
                    money_scales, "scale_status"
                ),
            },
        }


def build_profile_input_lineage(
    country_profiles: Sequence[CountryProfile],
    *,
    profile_bundle: ProfileBundle | None = None,
) -> ProfileInputLineage:
    """Snapshot and fingerprint the exact profile tuple used by a policy run."""

    profiles = tuple(country_profiles)
    if not profiles:
        raise ValueError("at least one country profile is required")
    if any(not isinstance(profile, CountryProfile) for profile in profiles):
        raise TypeError("country_profiles must contain CountryProfile instances")
    if profile_bundle is not None and profiles != profile_bundle.country_profiles:
        raise ProfileValidationError(
            "profile bundle country profiles differ from the profiles being fingerprinted"
        )

    bundle_snapshot: dict[str, object] | None = None
    if profile_bundle is not None:
        bundle_snapshot = {
            "profile_status": profile_bundle.profile_status.value,
            "caveats": list(profile_bundle.caveats),
            "sources": [
                _snapshot_dataclass(profile_bundle.sources[source_id])
                for source_id in sorted(profile_bundle.sources)
            ],
            "metric_contracts": [
                _snapshot_dataclass(contract) for contract in profile_bundle.contracts
            ],
            "money_scales": [
                _snapshot_dataclass(scale) for scale in profile_bundle.money_scales
            ],
        }
    registered_bundle = profile_bundle is not None and _matches_loaded_profile_bundle(
        profile_bundle
    )
    if registered_bundle:
        lineage_status = _REGISTERED_PROFILE_LINEAGE
    elif profile_bundle is not None:
        lineage_status = _UNREGISTERED_BUNDLE_LINEAGE
    else:
        lineage_status = _UNREGISTERED_PROFILE_LINEAGE
    jurisdictions_path = (
        str(profile_bundle.jurisdictions_path)
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.jurisdictions_path is not None
        else None
    )
    jurisdictions_sha256 = (
        profile_bundle.jurisdictions_sha256
        if registered_bundle and profile_bundle is not None
        else None
    )
    source_registry_path = (
        str(profile_bundle.source_registry_path)
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.source_registry_path is not None
        else None
    )
    source_registry_sha256 = (
        profile_bundle.source_registry_sha256
        if registered_bundle and profile_bundle is not None
        else None
    )
    source_retrieved_on = (
        profile_bundle.source_retrieved_on
        if registered_bundle and profile_bundle is not None
        else None
    )
    snapshot = {
        "schema_version": _PROFILE_INPUT_SCHEMA_VERSION,
        "lineage_status": lineage_status,
        "country_profile_type": (
            "microtx_sim.consumers.population.CountryProfile"
        ),
        "country_profiles": [_snapshot_dataclass(profile) for profile in profiles],
        "profile_bundle": bundle_snapshot,
        "file_lineage": {
            "jurisdictions": {
                "path": jurisdictions_path,
                "sha256": jurisdictions_sha256,
            },
            "source_registry": {
                "path": source_registry_path,
                "sha256": source_registry_sha256,
                "retrieved_on": (
                    source_retrieved_on.isoformat()
                    if source_retrieved_on is not None
                    else None
                ),
            },
        },
    }
    snapshot_json = _canonical_snapshot_json(snapshot)
    return ProfileInputLineage(
        lineage_status=lineage_status,
        profile_codes=tuple(profile.code for profile in profiles),
        fingerprint_sha256=sha256(snapshot_json.encode("utf-8")).hexdigest(),
        snapshot_json=snapshot_json,
        jurisdictions_path=jurisdictions_path,
        jurisdictions_sha256=jurisdictions_sha256,
        source_registry_path=source_registry_path,
        source_registry_sha256=source_registry_sha256,
        source_retrieved_on=source_retrieved_on,
    )


def resolve_profile_inputs(
    *,
    country_profiles: Sequence[CountryProfile] | None = None,
    profile_bundle: ProfileBundle | None = None,
) -> tuple[tuple[CountryProfile, ...], ProfileInputLineage]:
    """Resolve registered or custom inputs without conflating their lineage."""

    if country_profiles is not None and profile_bundle is not None:
        raise ValueError("supply country_profiles or profile_bundle, not both")
    if profile_bundle is None and country_profiles is None:
        profile_bundle = load_profile_bundle(campaign=False)
    profiles = (
        profile_bundle.country_profiles
        if profile_bundle is not None
        else tuple(country_profiles or ())
    )
    return profiles, build_profile_input_lineage(
        profiles,
        profile_bundle=profile_bundle,
    )


def _matches_loaded_profile_bundle(bundle: ProfileBundle) -> bool:
    """Return whether bundle contents exactly match its declared input files."""

    if any(
        value is None
        for value in (
            bundle.jurisdictions_path,
            bundle.jurisdictions_sha256,
            bundle.source_registry_path,
            bundle.source_registry_sha256,
            bundle.source_retrieved_on,
        )
    ):
        return False
    assert bundle.jurisdictions_path is not None
    assert bundle.source_registry_path is not None
    try:
        loaded = load_profile_bundle(
            bundle.jurisdictions_path,
            bundle.source_registry_path,
            campaign=False,
        )
    except ProfileValidationError:
        return False
    return (
        loaded.country_profiles == bundle.country_profiles
        and loaded.sources == bundle.sources
        and loaded.profile_status is bundle.profile_status
        and loaded.caveats == bundle.caveats
        and loaded.contracts == bundle.contracts
        and loaded.money_scales == bundle.money_scales
        and loaded.jurisdictions_path == bundle.jurisdictions_path
        and loaded.source_registry_path == bundle.source_registry_path
        and loaded.jurisdictions_sha256 == bundle.jurisdictions_sha256
        and loaded.source_registry_sha256 == bundle.source_registry_sha256
        and loaded.source_retrieved_on == bundle.source_retrieved_on
    )


def _snapshot_dataclass(value: object) -> dict[str, object]:
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("profile snapshots require dataclass instances")
    return {
        descriptor.name: _json_value(getattr(value, descriptor.name))
        for descriptor in fields(value)
    }


def _json_value(value: object) -> object:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _snapshot_dataclass(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported profile snapshot value: {type(value).__name__}")


def _canonical_snapshot_json(snapshot: Mapping[str, object]) -> str:
    return json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _status_counts(rows: Sequence[object], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or field not in row:
            continue
        status = str(row[field])
        counts[status] = counts.get(status, 0) + 1
    return {status: counts[status] for status in sorted(counts)}


__all__ = [
    "ProfileInputLineage",
    "build_profile_input_lineage",
    "resolve_profile_inputs",
]
