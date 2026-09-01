"""Content-addressed lineage for policy population inputs."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from typing import Final, Mapping, Sequence

import numpy as np

from ..consumers.population import CountryProfile
from .profiles import (
    ProfileBundle,
    ProfileValidationError,
    _is_sha256,
    _parse_iso_date,
    load_profile_bundle,
    monetary_evidence_assessment_from_snapshot,
    population_evidence_assessment_from_snapshot,
)


_PROFILE_INPUT_SCHEMA_VERSION = 4
_SUPPORTED_PROFILE_INPUT_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
_REGISTERED_PROFILE_LINEAGE = "registered_profile_bundle"
_UNREGISTERED_PROFILE_LINEAGE = "unregistered_custom_profiles"
_UNREGISTERED_BUNDLE_LINEAGE = "unregistered_profile_bundle"

# These are the only historical plan fingerprints known to have been computed
# with the portable-v1 recipe on a Windows checkout.  The corresponding values
# are the canonical recipe's digest of the same checked-in profile snapshots.
# The mapping is directional: an immutable historical plan may name the legacy
# digest while a newly resolved runtime must emit the canonical digest.  It is
# deliberately not a general digest-alias mechanism.
_PROFILE_LINEAGE_FINGERPRINT_MIGRATIONS: Final[
    frozenset[tuple[str, str]]
] = frozenset(
    {
        (
            "8458d4c844e4a1e810d76e0a83e41e742d97e595373432b95e9e493322232dd4",
            "ce1c4592c3968215f6ec9fa9b7d907f42fc25feca4e9c5f795b2e72244c9ff56",
        ),
        (
            "119e5a9cbc919808520c395b4346d50e4a12fe9d5ec095f76816ad7c1fe38658",
            "5abda0b7383ba4051889bf05aa53f3faff729e2a564b5cac9864617b972f42e8",
        ),
    }
)


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
    source_bundle_path: str | None = None
    source_bundle_sha256: str | None = None
    population_bundle_path: str | None = None
    population_bundle_sha256: str | None = None

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
        try:
            snapshot = json.loads(self.snapshot_json)
        except json.JSONDecodeError as exc:
            raise ProfileValidationError("profile snapshot must be valid JSON") from exc
        if not isinstance(snapshot, dict):
            raise ProfileValidationError("profile snapshot root must be an object")
        if set(snapshot) != {
            "schema_version",
            "lineage_status",
            "country_profile_type",
            "country_profiles",
            "profile_bundle",
            "file_lineage",
        }:
            raise ProfileValidationError(
                "profile snapshot root fields do not match its schema"
            )
        if self.snapshot_json != _canonical_snapshot_json(snapshot):
            raise ProfileValidationError("profile snapshot JSON must be canonical")
        expected_legacy = sha256(self.snapshot_json.encode("utf-8")).hexdigest()
        expected_portable_v1 = _profile_lineage_fingerprint_sha256_v1(snapshot)
        expected_portable = _profile_lineage_fingerprint_sha256(snapshot)
        if self.fingerprint_sha256 not in {
            expected_legacy,
            expected_portable_v1,
            expected_portable,
        }:
            raise ProfileValidationError("profile fingerprint does not match its snapshot")
        snapshot_schema_version = snapshot.get("schema_version")
        if (
            type(snapshot_schema_version) is not int
            or snapshot_schema_version not in _SUPPORTED_PROFILE_INPUT_SCHEMA_VERSIONS
        ):
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
        expected_lineage_fields = {"jurisdictions", "source_registry"}
        if snapshot_schema_version >= 3:
            expected_lineage_fields.add("source_bundle")
        if snapshot_schema_version == 4:
            expected_lineage_fields.add("population_bundle")
        if set(file_lineage) != expected_lineage_fields:
            raise ProfileValidationError(
                "profile snapshot file lineage fields do not match its schema"
            )
        jurisdictions = file_lineage.get("jurisdictions")
        source_registry = file_lineage.get("source_registry")
        if not isinstance(jurisdictions, dict) or not isinstance(
            source_registry, dict
        ):
            raise ProfileValidationError("profile snapshot file lineage is malformed")
        if set(jurisdictions) != {"path", "sha256"} or set(source_registry) != {
            "path",
            "sha256",
            "retrieved_on",
        }:
            raise ProfileValidationError(
                "profile snapshot file-lineage record fields are malformed"
            )
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
        if snapshot_schema_version >= 3:
            source_bundle = file_lineage.get("source_bundle")
            if not isinstance(source_bundle, dict) or set(source_bundle) != {
                "path",
                "sha256",
                "source_registry_sha256",
                "signature_status",
            }:
                raise ProfileValidationError(
                    "profile snapshot source-bundle lineage is malformed"
                )
            snapshot_source_bundle_path = source_bundle.get("path")
            snapshot_source_bundle_sha256 = source_bundle.get("sha256")
            bound_registry_sha256 = source_bundle.get("source_registry_sha256")
            signature_status = source_bundle.get("signature_status")
            if snapshot_source_bundle_path is None:
                if any(
                    value is not None
                    for value in (
                        snapshot_source_bundle_sha256,
                        bound_registry_sha256,
                        signature_status,
                    )
                ):
                    raise ProfileValidationError(
                        "absent source-bundle lineage cannot claim evidence metadata"
                    )
            elif (
                not isinstance(snapshot_source_bundle_path, str)
                or not snapshot_source_bundle_path
                or not _is_sha256(snapshot_source_bundle_sha256)
                or bound_registry_sha256 != snapshot_source_registry_sha256
                or signature_status != "MISSING"
            ):
                raise ProfileValidationError(
                    "profile snapshot source-bundle lineage is inconsistent"
                )
        else:
            snapshot_source_bundle_path = None
            snapshot_source_bundle_sha256 = None
        if snapshot_schema_version == 4:
            population_bundle = file_lineage.get("population_bundle")
            if not isinstance(population_bundle, dict) or set(population_bundle) != {
                "path",
                "sha256",
                "source_registry_sha256",
                "signature_status",
            }:
                raise ProfileValidationError(
                    "profile snapshot population-bundle lineage is malformed"
                )
            snapshot_population_bundle_path = population_bundle.get("path")
            snapshot_population_bundle_sha256 = population_bundle.get("sha256")
            population_registry_sha256 = population_bundle.get(
                "source_registry_sha256"
            )
            population_signature_status = population_bundle.get(
                "signature_status"
            )
            if snapshot_population_bundle_path is None:
                if any(
                    value is not None
                    for value in (
                        snapshot_population_bundle_sha256,
                        population_registry_sha256,
                        population_signature_status,
                    )
                ):
                    raise ProfileValidationError(
                        "absent population-bundle lineage cannot claim evidence metadata"
                    )
            elif (
                not isinstance(snapshot_population_bundle_path, str)
                or not snapshot_population_bundle_path
                or not _is_sha256(snapshot_population_bundle_sha256)
                or population_registry_sha256 != snapshot_source_registry_sha256
                or population_signature_status != "MISSING"
            ):
                raise ProfileValidationError(
                    "profile snapshot population-bundle lineage is inconsistent"
                )
        else:
            snapshot_population_bundle_path = None
            snapshot_population_bundle_sha256 = None
        if (
            snapshot_jurisdictions_path != self.jurisdictions_path
            or snapshot_jurisdictions_sha256 != self.jurisdictions_sha256
            or snapshot_source_registry_path != self.source_registry_path
            or snapshot_source_registry_sha256 != self.source_registry_sha256
            or snapshot_retrieved_on != self.source_retrieved_on
            or snapshot_source_bundle_path != self.source_bundle_path
            or snapshot_source_bundle_sha256 != self.source_bundle_sha256
            or snapshot_population_bundle_path != self.population_bundle_path
            or snapshot_population_bundle_sha256
            != self.population_bundle_sha256
        ):
            raise ProfileValidationError(
                "published profile file lineage does not match its fingerprinted snapshot"
            )

        for name in (
            "jurisdictions_path",
            "source_registry_path",
            "source_bundle_path",
            "population_bundle_path",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ProfileValidationError(f"{name} must be non-empty text")
        for name in (
            "jurisdictions_sha256",
            "source_registry_sha256",
            "source_bundle_sha256",
            "population_bundle_sha256",
        ):
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
                self.source_bundle_path,
                self.source_bundle_sha256,
                self.population_bundle_path,
                self.population_bundle_sha256,
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
        if snapshot_schema_version == 1 and isinstance(bundle, dict) and (
            "monetary_conversions" in bundle
        ):
            raise ProfileValidationError(
                "profile snapshot schema version 1 cannot contain monetary conversions"
            )
        _validate_snapshot_schema_fields(
            snapshot_schema_version,
            bundle,
            registered_lineage=(
                self.lineage_status == _REGISTERED_PROFILE_LINEAGE
            ),
        )
        self._validate_registered_files(snapshot)

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
        self._validate_registered_files(snapshot)
        bundle = snapshot.get("profile_bundle")
        bundle_payload = bundle if isinstance(bundle, dict) else {}
        metric_contracts = bundle_payload.get("metric_contracts", [])
        money_scales = bundle_payload.get("money_scales", [])
        monetary_conversions = bundle_payload.get("monetary_conversions", [])
        source_evidence_bundle = bundle_payload.get("source_evidence_bundle")
        rate_evidence_results = bundle_payload.get("rate_evidence_results", [])
        population_evidence_bundle = bundle_payload.get(
            "population_evidence_bundle"
        )
        population_evidence_results = bundle_payload.get(
            "population_evidence_results", []
        )
        monetary_evidence_assessment = bundle_payload.get(
            "monetary_evidence_assessment"
        )
        population_evidence_assessment = bundle_payload.get(
            "population_evidence_assessment"
        )
        if (
            not isinstance(metric_contracts, list)
            or not isinstance(money_scales, list)
            or not isinstance(monetary_conversions, list)
            or not isinstance(rate_evidence_results, list)
            or not isinstance(population_evidence_results, list)
        ):
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
            "source_bundle": (
                dict(file_lineage["source_bundle"])
                if "source_bundle" in file_lineage
                else {
                    "path": None,
                    "sha256": None,
                    "source_registry_sha256": None,
                    "signature_status": None,
                }
            ),
            "population_bundle": (
                dict(file_lineage["population_bundle"])
                if "population_bundle" in file_lineage
                else {
                    "path": None,
                    "sha256": None,
                    "source_registry_sha256": None,
                    "signature_status": None,
                }
            ),
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
            "monetary_conversion_summary": {
                "count": len(monetary_conversions),
                "methods": sorted(
                    {
                        str(item["method"])
                        for item in monetary_conversions
                        if isinstance(item, dict) and "method" in item
                    }
                ),
                "source_currencies": sorted(
                    {
                        str(item["source_currency"])
                        for item in monetary_conversions
                        if isinstance(item, dict) and "source_currency" in item
                    }
                ),
                "target_currencies": sorted(
                    {
                        str(item["target_currency"])
                        for item in monetary_conversions
                        if isinstance(item, dict) and "target_currency" in item
                    }
                ),
                "rate_period_starts": _distinct_text(
                    monetary_conversions,
                    "rate_period_start",
                ),
                "rate_period_ends": _distinct_text(
                    monetary_conversions,
                    "rate_period_end",
                ),
                "target_price_period_starts": _distinct_text(
                    monetary_conversions,
                    "target_price_period_start",
                ),
                "target_price_period_ends": _distinct_text(
                    monetary_conversions,
                    "target_price_period_end",
                ),
                "estimands": _distinct_text(
                    monetary_conversions,
                    "estimand",
                ),
                "population_bases": _distinct_text(
                    monetary_conversions,
                    "population_base",
                ),
                "comparison_groups": _distinct_text(
                    monetary_conversions,
                    "comparison_group",
                ),
                "retrieval_dates": _distinct_text(
                    monetary_conversions,
                    "retrieved_on",
                ),
                "rounding_scopes": _distinct_text(
                    monetary_conversions,
                    "rounding_scope",
                ),
                "aggregation_units": _distinct_text(
                    monetary_conversions,
                    "aggregation_unit",
                ),
                "status_counts": _status_counts(
                    monetary_conversions,
                    "status",
                ),
            },
            "source_evidence_summary": {
                "present": isinstance(source_evidence_bundle, dict),
                "artifact_count": (
                    len(source_evidence_bundle.get("artifacts", []))
                    if isinstance(source_evidence_bundle, dict)
                    and isinstance(source_evidence_bundle.get("artifacts"), list)
                    else 0
                ),
                "binding_count": (
                    len(source_evidence_bundle.get("bindings", []))
                    if isinstance(source_evidence_bundle, dict)
                    and isinstance(source_evidence_bundle.get("bindings"), list)
                    else 0
                ),
                "verified_result_count": len(rate_evidence_results),
                "signature_status": (
                    source_evidence_bundle.get("signature", {}).get("status")
                    if isinstance(source_evidence_bundle, dict)
                    and isinstance(source_evidence_bundle.get("signature"), dict)
                    else None
                ),
            },
            "population_evidence_summary": {
                "present": isinstance(population_evidence_bundle, dict),
                "artifact_count": (
                    len(population_evidence_bundle.get("artifacts", []))
                    if isinstance(population_evidence_bundle, dict)
                    and isinstance(
                        population_evidence_bundle.get("artifacts"), list
                    )
                    else 0
                ),
                "binding_count": (
                    len(population_evidence_bundle.get("bindings", []))
                    if isinstance(population_evidence_bundle, dict)
                    and isinstance(
                        population_evidence_bundle.get("bindings"), list
                    )
                    else 0
                ),
                "verified_result_count": len(population_evidence_results),
                "signature_status": (
                    population_evidence_bundle.get("signature", {}).get("status")
                    if isinstance(population_evidence_bundle, dict)
                    and isinstance(
                        population_evidence_bundle.get("signature"), dict
                    )
                    else None
                ),
            },
            "monetary_evidence_assessment": (
                dict(monetary_evidence_assessment)
                if isinstance(monetary_evidence_assessment, dict)
                else {
                    "structure_coherent": False,
                    "source_rate_evidence_bound": False,
                    "source_bundle_signature_bound": False,
                    "output_design_binding_bound": False,
                    "population_binding_bound": False,
                    "preregistration_bound": False,
                    "public_output_comparability": False,
                    "blockers": [
                        "monetary_conversion.structure=unavailable",
                        "monetary_conversion.source_rate_binding=missing",
                        "monetary_conversion.source_bundle_signature=missing",
                        "monetary_conversion.output_design_binding=missing",
                        "monetary_conversion.population_binding=missing",
                        "monetary_conversion.preregistration_binding=missing",
                    ],
                }
            ),
            "population_evidence_assessment": (
                dict(population_evidence_assessment)
                if isinstance(population_evidence_assessment, dict)
                else {
                    "structure_coherent": False,
                    "source_population_evidence_bound": False,
                    "calibration_targets_bound": False,
                    "heldout_validation_targets_bound": False,
                    "source_bundle_signature_bound": False,
                    "sampling_plan_bound": False,
                    "runtime_projection_bound": False,
                    "output_estimand_binding_bound": False,
                    "balance_validation_bound": False,
                    "public_population_comparability": False,
                    "blockers": [
                        "population.structure=unavailable",
                        "population.source_evidence=missing",
                        "population.calibration_targets=missing",
                        "population.heldout_validation_targets=missing",
                        "population.source_bundle_signature=missing",
                        "population.sampling_plan=missing",
                        "population.runtime_projection=missing",
                        "population.output_estimand_binding=missing",
                        "population.balance_validation=missing",
                    ],
                }
            ),
        }

    def _validate_registered_files(self, snapshot: Mapping[str, object]) -> None:
        """Re-attest every registered claim against the current source files."""

        if self.lineage_status != _REGISTERED_PROFILE_LINEAGE:
            return
        assert self.jurisdictions_path is not None
        assert self.source_registry_path is not None
        try:
            loaded = load_profile_bundle(
                self.jurisdictions_path,
                self.source_registry_path,
                source_bundle_path=(
                    self.source_bundle_path
                    if snapshot.get("schema_version") in {3, 4}
                    else None
                ),
                population_bundle_path=(
                    self.population_bundle_path
                    if snapshot.get("schema_version") == 4
                    else None
                ),
                campaign=False,
            )
        except (OSError, ProfileValidationError) as exc:
            raise ProfileValidationError(
                "registered profile lineage files are unavailable or invalid"
            ) from exc
        if (
            str(loaded.jurisdictions_path) != self.jurisdictions_path
            or loaded.jurisdictions_sha256 != self.jurisdictions_sha256
            or str(loaded.source_registry_path) != self.source_registry_path
            or loaded.source_registry_sha256 != self.source_registry_sha256
            or loaded.source_retrieved_on != self.source_retrieved_on
            or (
                str(loaded.source_evidence_bundle.bundle_path)
                if loaded.source_evidence_bundle is not None
                else None
            )
            != self.source_bundle_path
            or (
                loaded.source_evidence_bundle.bundle_sha256
                if loaded.source_evidence_bundle is not None
                else None
            )
            != self.source_bundle_sha256
            or (
                str(loaded.population_evidence_bundle.bundle_path)
                if loaded.population_evidence_bundle is not None
                else None
            )
            != self.population_bundle_path
            or (
                loaded.population_evidence_bundle.bundle_sha256
                if loaded.population_evidence_bundle is not None
                else None
            )
            != self.population_bundle_sha256
        ):
            raise ProfileValidationError(
                "registered profile lineage no longer matches its claimed files"
            )
        expected_bundle = _profile_bundle_snapshot(loaded, registered=True)
        snapshot_version = snapshot.get("schema_version")
        if snapshot_version in {1, 2, 3}:
            try:
                with Path(self.jurisdictions_path).open("rb") as handle:
                    claimed_schema_version = tomllib.load(handle).get(
                        "schema_version"
                    )
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ProfileValidationError(
                    "registered profile lineage files are unavailable or invalid"
                ) from exc
            if type(claimed_schema_version) is not int or (
                snapshot_version == 1
                and (
                    claimed_schema_version != 1
                    or loaded.monetary_conversions
                )
            ) or (
                snapshot_version == 2
                and claimed_schema_version not in {1, 2}
            ) or (
                snapshot_version == 3
                and claimed_schema_version not in {1, 2, 3}
            ):
                raise ProfileValidationError(
                    "legacy profile snapshot requires a compatible jurisdiction file"
                )
            expected_bundle = _downgrade_profile_bundle_snapshot(
                expected_bundle,
                version=snapshot_version,
            )
        if (
            snapshot.get("country_profiles")
            != [_snapshot_dataclass(profile) for profile in loaded.country_profiles]
            or snapshot.get("profile_bundle") != expected_bundle
        ):
            raise ProfileValidationError(
                "registered profile snapshot values do not match their input files"
            )


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

    registered_bundle = profile_bundle is not None and _matches_loaded_profile_bundle(
        profile_bundle
    )
    if registered_bundle:
        lineage_status = _REGISTERED_PROFILE_LINEAGE
    elif profile_bundle is not None:
        lineage_status = _UNREGISTERED_BUNDLE_LINEAGE
    else:
        lineage_status = _UNREGISTERED_PROFILE_LINEAGE
    bundle_snapshot: dict[str, object] | None = None
    if profile_bundle is not None:
        bundle_snapshot = _profile_bundle_snapshot(
            profile_bundle,
            registered=registered_bundle,
        )
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
    source_bundle_path = (
        str(profile_bundle.source_evidence_bundle.bundle_path)
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.source_evidence_bundle is not None
        else None
    )
    source_bundle_sha256 = (
        profile_bundle.source_evidence_bundle.bundle_sha256
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.source_evidence_bundle is not None
        else None
    )
    source_bundle_registry_sha256 = (
        profile_bundle.source_evidence_bundle.source_registry_sha256
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.source_evidence_bundle is not None
        else None
    )
    source_bundle_signature_status = (
        profile_bundle.source_evidence_bundle.signature.status.value
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.source_evidence_bundle is not None
        else None
    )
    population_bundle_path = (
        str(profile_bundle.population_evidence_bundle.bundle_path)
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.population_evidence_bundle is not None
        else None
    )
    population_bundle_sha256 = (
        profile_bundle.population_evidence_bundle.bundle_sha256
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.population_evidence_bundle is not None
        else None
    )
    population_bundle_registry_sha256 = (
        profile_bundle.population_evidence_bundle.source_registry_sha256
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.population_evidence_bundle is not None
        else None
    )
    population_bundle_signature_status = (
        profile_bundle.population_evidence_bundle.signature.status.value
        if registered_bundle
        and profile_bundle is not None
        and profile_bundle.population_evidence_bundle is not None
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
            "source_bundle": {
                "path": source_bundle_path,
                "sha256": source_bundle_sha256,
                "source_registry_sha256": source_bundle_registry_sha256,
                "signature_status": source_bundle_signature_status,
            },
            "population_bundle": {
                "path": population_bundle_path,
                "sha256": population_bundle_sha256,
                "source_registry_sha256": population_bundle_registry_sha256,
                "signature_status": population_bundle_signature_status,
            },
        },
    }
    snapshot_json = _canonical_snapshot_json(snapshot)
    return ProfileInputLineage(
        lineage_status=lineage_status,
        profile_codes=tuple(profile.code for profile in profiles),
        fingerprint_sha256=_profile_lineage_fingerprint_sha256(snapshot),
        snapshot_json=snapshot_json,
        jurisdictions_path=jurisdictions_path,
        jurisdictions_sha256=jurisdictions_sha256,
        source_registry_path=source_registry_path,
        source_registry_sha256=source_registry_sha256,
        source_retrieved_on=source_retrieved_on,
        source_bundle_path=source_bundle_path,
        source_bundle_sha256=source_bundle_sha256,
        population_bundle_path=population_bundle_path,
        population_bundle_sha256=population_bundle_sha256,
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

    return bundle.matches_registered_files()


def _profile_bundle_snapshot(
    bundle: ProfileBundle,
    *,
    registered: bool,
) -> dict[str, object]:
    conversions: list[dict[str, object]] = []
    for conversion in bundle.monetary_conversions:
        snapshot = _snapshot_dataclass(conversion)
        snapshot["rate_numerator_decimal"] = str(conversion.rate_numerator)
        snapshot["rate_denominator_decimal"] = str(conversion.rate_denominator)
        conversions.append(snapshot)
    source_evidence_snapshot: dict[str, object] | None = None
    rate_evidence_results: list[dict[str, object]] = []
    if bundle.source_evidence_bundle is not None:
        from .rate_evidence import (
            RateEvidenceValidationError,
            verify_rate_evidence_bundle,
        )

        try:
            verified = verify_rate_evidence_bundle(bundle.source_evidence_bundle)
        except RateEvidenceValidationError as exc:
            raise ProfileValidationError(
                f"profile source evidence cannot be re-attested: {exc}"
            ) from exc
        source_evidence_snapshot = bundle.source_evidence_bundle.snapshot()
        rate_evidence_results = [result.snapshot() for result in verified]
    population_evidence_snapshot: dict[str, object] | None = None
    population_evidence_results: list[dict[str, object]] = []
    if bundle.population_evidence_bundle is not None:
        from .population_evidence import (
            PopulationEvidenceValidationError,
            verify_population_evidence_bundle,
        )

        try:
            verified_population = verify_population_evidence_bundle(
                bundle.population_evidence_bundle,
                expected_source_registry_sha256=bundle.source_registry_sha256,
            )
        except PopulationEvidenceValidationError as exc:
            raise ProfileValidationError(
                f"profile population evidence cannot be re-attested: {exc}"
            ) from exc
        population_evidence_snapshot = bundle.population_evidence_bundle.snapshot()
        population_evidence_results = [
            result.snapshot() for result in verified_population
        ]
    evidence_assessment = bundle.monetary_evidence_assessment(
        registered=registered
    )
    population_assessment = bundle.population_evidence_assessment(
        registered=registered
    )
    return {
        "jurisdiction_schema_version": bundle.jurisdiction_schema_version,
        "source_catalogue_schema_version": bundle.source_catalogue_schema_version,
        "profile_status": bundle.profile_status.value,
        "caveats": list(bundle.caveats),
        "sources": [
            _snapshot_dataclass(bundle.sources[source_id])
            for source_id in sorted(bundle.sources)
        ],
        "metric_contracts": [
            _snapshot_dataclass(contract) for contract in bundle.contracts
        ],
        "money_scales": [
            _snapshot_dataclass(scale) for scale in bundle.money_scales
        ],
        "monetary_conversions": conversions,
        "source_evidence_bundle": source_evidence_snapshot,
        "rate_evidence_results": rate_evidence_results,
        "monetary_evidence_assessment": _snapshot_dataclass(
            evidence_assessment
        ),
        "population_evidence_bundle": population_evidence_snapshot,
        "population_evidence_results": population_evidence_results,
        "population_evidence_assessment": _snapshot_dataclass(
            population_assessment
        ),
    }


def _validate_snapshot_schema_fields(
    schema_version: int,
    bundle: object,
    *,
    registered_lineage: bool,
) -> None:
    """Reject downgrade claims and internally inconsistent evidence flags."""

    if not isinstance(bundle, dict):
        return
    base_fields = {
        "profile_status",
        "caveats",
        "sources",
        "metric_contracts",
        "money_scales",
    }
    v3_fields = {
        "jurisdiction_schema_version",
        "source_catalogue_schema_version",
        "source_evidence_bundle",
        "rate_evidence_results",
        "monetary_evidence_assessment",
    }
    v4_fields = {
        "population_evidence_bundle",
        "population_evidence_results",
        "population_evidence_assessment",
    }
    expected_fields = set(base_fields)
    if schema_version >= 2:
        expected_fields.add("monetary_conversions")
    if schema_version >= 3:
        expected_fields.update(v3_fields)
    if schema_version == 4:
        expected_fields.update(v4_fields)
    if schema_version < 3:
        present_v3_fields = sorted(v3_fields.intersection(bundle))
        if present_v3_fields:
            raise ProfileValidationError(
                "legacy profile snapshot cannot contain schema-v3 evidence fields: "
                + ", ".join(present_v3_fields)
            )
    if schema_version < 4:
        present_v4_fields = sorted(v4_fields.intersection(bundle))
        if present_v4_fields:
            raise ProfileValidationError(
                "legacy profile snapshot cannot contain schema-v4 population fields: "
                + ", ".join(present_v4_fields)
            )
    if set(bundle) != expected_fields:
        raise ProfileValidationError(
            "profile bundle snapshot fields do not match its schema"
        )
    if schema_version < 3:
        conversions = bundle.get("monetary_conversions", [])
        if isinstance(conversions, list) and any(
            isinstance(conversion, dict)
            and bool({"conversion_id", "rate_binding_id"}.intersection(conversion))
            for conversion in conversions
        ):
            raise ProfileValidationError(
                "legacy profile snapshot cannot contain schema-v3 conversion fields"
            )
        return

    missing = sorted(v3_fields.difference(bundle))
    if missing:
        raise ProfileValidationError(
            "profile snapshot schema version 3 is missing evidence fields: "
            + ", ".join(missing)
        )
    jurisdiction_version = bundle.get("jurisdiction_schema_version")
    source_version = bundle.get("source_catalogue_schema_version")
    if (
        type(jurisdiction_version) is not int
        or jurisdiction_version not in {1, 2, 3}
        or type(source_version) is not int
        or source_version != 1
    ):
        raise ProfileValidationError(
            "profile snapshot contains unsupported source schema metadata"
        )
    results = bundle.get("rate_evidence_results")
    if not isinstance(results, list) or any(
        not isinstance(result, dict) for result in results
    ):
        raise ProfileValidationError(
            "profile snapshot rate evidence results are malformed"
        )
    from .rate_evidence import (
        RateEvidenceValidationError,
        validate_rate_evidence_snapshot,
    )

    try:
        validate_rate_evidence_snapshot(
            bundle.get("source_evidence_bundle"),
            results,
        )
    except RateEvidenceValidationError as exc:
        raise ProfileValidationError(
            f"profile snapshot source evidence is malformed: {exc}"
        ) from exc
    monetary_evidence_assessment_from_snapshot(
        bundle.get("monetary_evidence_assessment"),
        registered_lineage=registered_lineage,
        bundle_snapshot=bundle,
    )
    if schema_version == 4:
        population_results = bundle.get("population_evidence_results")
        if not isinstance(population_results, list) or any(
            not isinstance(result, dict) for result in population_results
        ):
            raise ProfileValidationError(
                "profile snapshot population evidence results are malformed"
            )
        population_evidence_assessment_from_snapshot(
            bundle.get("population_evidence_assessment"),
            registered_lineage=registered_lineage,
            bundle_snapshot=bundle,
        )


def _downgrade_profile_bundle_snapshot(
    bundle: dict[str, object],
    *,
    version: object,
) -> dict[str, object]:
    """Project current typed values onto an exact legacy snapshot surface."""

    if version not in {1, 2, 3}:
        raise ProfileValidationError("unsupported legacy profile snapshot version")
    for field in (
        "population_evidence_bundle",
        "population_evidence_results",
        "population_evidence_assessment",
    ):
        bundle.pop(field, None)
    if version == 3:
        return bundle
    for field in (
        "jurisdiction_schema_version",
        "source_catalogue_schema_version",
        "source_evidence_bundle",
        "rate_evidence_results",
        "monetary_evidence_assessment",
    ):
        bundle.pop(field, None)
    conversions = bundle.get("monetary_conversions")
    if isinstance(conversions, list):
        for conversion in conversions:
            if isinstance(conversion, dict):
                conversion.pop("conversion_id", None)
                conversion.pop("rate_binding_id", None)
                conversion.pop("quote_convention", None)
                conversion.pop("scale_convention", None)
                conversion.pop("timing_convention", None)
                conversion.pop("missing_date_policy", None)
    if version == 1:
        bundle.pop("monetary_conversions", None)
    return bundle


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


def _profile_lineage_fingerprint_sha256(
    snapshot: Mapping[str, object],
) -> str:
    """Hash profile content without binding nested evidence to a worktree."""

    portable = json.loads(_canonical_snapshot_json(snapshot))
    _normalize_portable_lineage_paths(portable)
    portable_json = _canonical_snapshot_json(portable)
    return sha256(portable_json.encode("utf-8")).hexdigest()


def _profile_lineage_fingerprint_sha256_v1(
    snapshot: Mapping[str, object],
) -> str:
    """Recompute the first portable recipe for backward verification only.

    The first recipe normalized the explicit ``file_lineage`` records but
    missed ``bundle_path`` fields nested in source- and population-evidence
    snapshots.  Those absolute paths made otherwise identical fingerprints
    differ between Windows and Linux.  Existing manifests remain readable,
    while newly built lineages use the complete recursive normalization above.
    """

    portable = json.loads(_canonical_snapshot_json(snapshot))
    file_lineage = portable.get("file_lineage")
    if isinstance(file_lineage, dict):
        for record in file_lineage.values():
            if isinstance(record, dict) and "path" in record:
                record["path"] = _portable_lineage_path(record["path"])
    portable_json = _canonical_snapshot_json(portable)
    return sha256(portable_json.encode("utf-8")).hexdigest()


def profile_lineage_fingerprint_matches(
    expected_sha256: str,
    observed_sha256: str,
) -> bool:
    """Match a canonical fingerprint or one attested historical migration.

    ``expected_sha256`` is the value frozen in an immutable plan and
    ``observed_sha256`` is the value emitted by the current runtime.  Legacy
    migrations are accepted only in that direction and only for the two
    snapshots whose path-only difference was independently reproduced.
    """

    if not _is_sha256(expected_sha256) or not _is_sha256(observed_sha256):
        raise ProfileValidationError(
            "profile fingerprint comparison requires SHA-256 digests"
        )
    return (
        expected_sha256 == observed_sha256
        or (expected_sha256, observed_sha256)
        in _PROFILE_LINEAGE_FINGERPRINT_MIGRATIONS
    )


def _normalize_portable_lineage_paths(value: object) -> None:
    """Normalize the declared local-path locations in schema v4."""

    if not isinstance(value, dict):
        return
    file_lineage = value.get("file_lineage")
    if isinstance(file_lineage, dict):
        for record in file_lineage.values():
            if isinstance(record, dict) and "path" in record:
                record["path"] = _portable_lineage_path(record["path"])
    profile_bundle = value.get("profile_bundle")
    if not isinstance(profile_bundle, dict):
        return
    for evidence_key in (
        "source_evidence_bundle",
        "population_evidence_bundle",
    ):
        evidence = profile_bundle.get(evidence_key)
        if isinstance(evidence, dict) and "bundle_path" in evidence:
            evidence["bundle_path"] = _portable_lineage_path(
                evidence["bundle_path"]
            )


def _portable_lineage_path(value: object) -> object:
    """Represent repository paths consistently across checkout locations."""

    if value is None or not isinstance(value, str):
        return value
    # Snapshot verification can happen on a different operating system from
    # the one that produced an older manifest, so accept both path separators.
    parts = tuple(
        part for part in value.replace("\\", "/").split("/") if part
    )
    repository_directories = {
        "configs",
        "data",
        "docs",
        "inputs",
        "src",
        "tests",
        "tools",
    }
    # Use the last declared repository-root component.  A checkout itself may
    # live below an ancestor named ``data`` or ``src``; taking the first match
    # would then retain part of the host-specific checkout path.
    candidate_indices = tuple(
        index
        for index, part in enumerate(parts)
        if part.casefold() in repository_directories
    )
    if candidate_indices:
        index = candidate_indices[-1]
        return "<repository>/" + "/".join(parts[index:])
    return "<external>"


def _status_counts(rows: Sequence[object], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or field not in row:
            continue
        status = str(row[field])
        counts[status] = counts.get(status, 0) + 1
    return {status: counts[status] for status in sorted(counts)}


def _distinct_text(rows: Sequence[object], field: str) -> list[str]:
    return sorted(
        {
            str(row[field])
            for row in rows
            if isinstance(row, dict) and field in row
        }
    )


__all__ = [
    "ProfileInputLineage",
    "build_profile_input_lineage",
    "profile_lineage_fingerprint_matches",
    "resolve_profile_inputs",
]
