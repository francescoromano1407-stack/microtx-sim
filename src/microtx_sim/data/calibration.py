"""Fail-closed loader for the UK-adults-2024 calibration evidence bundle.

The bundle described here is deliberately an evidence artefact, not a runtime
configuration.  Schema version 1 records quantified targets, a content-addressed
source manifest, and exact ONS age-by-sex population weights.  It also preserves
known identification gaps as typed, campaign-blocking rows.  Loading therefore
cannot silently turn a proxy, normative choice, or unquantified concept into a
calibration target.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Mapping, Sequence
from urllib.parse import urlparse


UK_ADULTS_2024_CALIBRATION_SCHEMA_VERSION = 1
UK_ADULTS_2024_CALIBRATION_BUNDLE_ID = "uk-adults-2024-v1"

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH = (
    _PROJECT_ROOT / "inputs" / "calibration" / UK_ADULTS_2024_CALIBRATION_BUNDLE_ID
)

TARGET_CSV_COLUMNS = (
    "target_id",
    "concept",
    "estimand_role",
    "evidence_status",
    "geography",
    "population",
    "reference_period",
    "subgroup",
    "value",
    "unit",
    "lower_ci",
    "upper_ci",
    "source_id",
    "source_file",
    "source_locator",
    "transformation",
    "runtime_mapping",
    "limitations",
)

POPULATION_WEIGHT_CSV_COLUMNS = (
    "age_band",
    "age_min_inclusive",
    "age_max_inclusive",
    "sex",
    "population_count",
    "adult_population_weight",
    "source_id",
    "estimand_role",
)

_COMPANION_FILENAMES = (
    "targets.csv",
    "population_weights.csv",
    "source_manifest.json",
)
_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "bundle_id",
        "status",
        "files",
        "frs_rounded_margin",
        "unsupported_concepts",
        "campaign_ready",
        "blockers",
    }
)
_FILE_DESCRIPTOR_KEYS = frozenset({"sha256", "byte_length"})
_FRS_MARGIN_KEYS = frozenset(
    {"target_id", "published_sum_percent", "normalization_applied"}
)
_SOURCE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "bundle_id",
        "raw_cache_root",
        "raw_cache_tracked",
        "verified_at",
        "sources",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "relative_path",
        "sha256",
        "byte_length",
        "official_url",
        "licence",
        "notes",
        "publisher",
        "title",
        "version",
        "publication_date",
        "retrieved_at",
        "evidence_role",
    }
)

_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_POSITIVE_INTEGER_PATTERN = re.compile(r"[1-9][0-9]*\Z")
_NONNEGATIVE_INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_CANONICAL_DECIMAL_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z"
)

_MAX_BUNDLE_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_TARGET_BYTES = 16 * 1024 * 1024
_MAX_POPULATION_WEIGHT_BYTES = 1024 * 1024
_MAX_SOURCE_BYTES = 1024 * 1024 * 1024
_POPULATION_WEIGHT_TOLERANCE = Decimal("5e-15")

_EXPECTED_POPULATION_COUNTS = MappingProxyType(
    {
        ("18-24", 18, 24, "FEMALE"): 2_821_237,
        ("18-24", 18, 24, "MALE"): 2_970_284,
        ("25-34", 25, 34, "FEMALE"): 4_754_911,
        ("25-34", 25, 34, "MALE"): 4_590_686,
        ("35-44", 35, 44, "FEMALE"): 4_805_400,
        ("35-44", 35, 44, "MALE"): 4_499_492,
        ("45-54", 45, 54, "FEMALE"): 4_343_516,
        ("45-54", 45, 54, "MALE"): 4_169_961,
        ("55-64", 55, 64, "FEMALE"): 4_554_117,
        ("55-64", 55, 64, "MALE"): 4_350_983,
    }
)
UK_ADULTS_2024_POPULATION_COUNT = sum(_EXPECTED_POPULATION_COUNTS.values())

UNSUPPORTED_CONCEPT_STATUSES = MappingProxyType(
    {
        "decision.temperature": "UNQUANTIFIED",
        "planned_unplanned_spending": "UNQUANTIFIED",
        "uk_adult_high_risk_prevalence": "UNQUANTIFIED",
        "simulation_cent_monetary_bridge": "UNQUANTIFIED",
        "composite_harm_weights": "NORMATIVE",
    }
)
UNSUPPORTED_CONCEPTS = tuple(UNSUPPORTED_CONCEPT_STATUSES)


class CalibrationBundleValidationError(ValueError):
    """Raised when a UK-adults calibration declaration is malformed."""


class CalibrationBundleVerificationError(CalibrationBundleValidationError):
    """Raised when declared bytes differ from the observed local evidence."""


class EstimandRole(str, Enum):
    """Disjoint uses assigned before targets are inspected."""

    CALIBRATION = "CALIBRATION"
    VALIDATION = "VALIDATION"
    DIAGNOSTIC = "DIAGNOSTIC"


class EvidenceStatus(str, Enum):
    """Identification status of one target or documented concept."""

    QUANTIFIED = "QUANTIFIED"
    UNQUANTIFIED = "UNQUANTIFIED"
    NORMATIVE = "NORMATIVE"
    PROXY = "PROXY"


class SourceEvidenceRole(str, Enum):
    """Pre-assigned role of a source record in the compact manifest."""

    CALIBRATION = "CALIBRATION"
    VALIDATION = "VALIDATION"
    DIAGNOSTIC = "DIAGNOSTIC"
    REFERENCE = "REFERENCE"
    MIXED = "MIXED"


@dataclass(frozen=True, slots=True)
class CalibrationSource:
    """One content-addressed local source plus bibliographic metadata."""

    source_id: str
    relative_path: str
    sha256: str
    byte_length: int
    official_url: str | None
    licence: str
    notes: str
    publisher: str
    title: str
    version: str
    publication_date: date | None
    retrieved_at: datetime | None
    evidence_role: SourceEvidenceRole


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """One typed target, holdout, diagnostic, or explicit evidence gap."""

    target_id: str
    concept: str
    estimand_role: EstimandRole
    evidence_status: EvidenceStatus
    geography: str
    population: str
    reference_period: str
    subgroup: str
    value: Decimal | None
    unit: str
    lower_ci: Decimal | None
    upper_ci: Decimal | None
    source_id: str
    source_file: str
    source_locator: str
    transformation: str
    runtime_mapping: str
    limitations: str


@dataclass(frozen=True, slots=True)
class PopulationWeight:
    """One exact ONS age-by-sex cell for UK residents aged 18--64."""

    age_band: str
    age_min_inclusive: int
    age_max_inclusive: int
    sex: str
    population_count: int
    adult_population_weight: Decimal
    source_id: str
    estimand_role: EstimandRole


@dataclass(frozen=True, slots=True)
class UKAdults2024CalibrationBundle:
    """Verified schema-v1 evidence bundle; intentionally not campaign-ready."""

    schema_version: int
    bundle_id: str
    status: str
    targets: tuple[CalibrationTarget, ...]
    population_weights: tuple[PopulationWeight, ...]
    sources: tuple[CalibrationSource, ...]
    blockers: tuple[str, ...]
    bundle_path: Path
    bundle_sha256: str

    @property
    def campaign_ready(self) -> bool:
        """Return the declared fail-closed campaign status."""

        return False

    @property
    def source_by_id(self) -> Mapping[str, CalibrationSource]:
        """Return an immutable source lookup."""

        return MappingProxyType({source.source_id: source for source in self.sources})

    def validate_for_campaign(self) -> None:
        """Reject execution because schema v1 intentionally preserves blockers."""

        raise CalibrationBundleValidationError(
            "UK-adults-2024 calibration bundle is PARTIAL and cannot authorize "
            "a campaign; blockers=" + ", ".join(self.blockers)
        )


def load_uk_adults_2024_calibration_bundle(
    path: str | Path = DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
    *,
    repository_root: str | Path | None = None,
) -> UKAdults2024CalibrationBundle:
    """Load and re-attest all declared files in the partial calibration bundle.

    ``path`` may name the bundle directory or ``calibration_bundle.json``.  A
    custom path outside this checkout must provide ``repository_root`` so that
    source-manifest paths cannot be resolved against an inferred, ambiguous root.
    """

    supplied_path = Path(path)
    bundle_path = (
        supplied_path
        if supplied_path.name == "calibration_bundle.json"
        else supplied_path / "calibration_bundle.json"
    )
    bundle_bytes = _secure_read_regular_file(
        bundle_path,
        maximum_bytes=_MAX_BUNDLE_BYTES,
        description="calibration bundle",
    )
    bundle_root = bundle_path.resolve(strict=True).parent
    repo_root = _resolve_repository_root(
        bundle_root,
        repository_root=repository_root,
    )
    raw_bundle = _parse_json_object(
        bundle_bytes,
        name="calibration bundle",
    )
    _exact_keys(raw_bundle, _BUNDLE_KEYS, name="calibration bundle")
    schema_version = _strict_int(
        raw_bundle.get("schema_version"),
        name="schema_version",
        minimum=1,
    )
    if schema_version != UK_ADULTS_2024_CALIBRATION_SCHEMA_VERSION:
        raise CalibrationBundleValidationError(
            f"unsupported calibration schema version: {schema_version}"
        )
    bundle_id = _required_identifier(raw_bundle, "bundle_id")
    if bundle_id != UK_ADULTS_2024_CALIBRATION_BUNDLE_ID:
        raise CalibrationBundleValidationError(
            "calibration bundle_id must be uk-adults-2024-v1"
        )
    status = _required_text(raw_bundle, "status")
    if status != "PARTIAL":
        raise CalibrationBundleValidationError(
            "schema-v1 calibration status must remain PARTIAL"
        )
    if raw_bundle.get("campaign_ready") is not False:
        raise CalibrationBundleValidationError(
            "schema-v1 calibration bundle must declare campaign_ready=false"
        )

    file_records = _required_mapping(raw_bundle, "files")
    _exact_keys(
        file_records,
        frozenset(_COMPANION_FILENAMES),
        name="calibration bundle files",
    )
    companion_bytes: dict[str, bytes] = {}
    for filename in _COMPANION_FILENAMES:
        record = _required_mapping(file_records, filename)
        _exact_keys(
            record,
            _FILE_DESCRIPTOR_KEYS,
            name=f"file descriptor {filename}",
        )
        expected_hash = _required_sha256(record, "sha256")
        expected_length = _strict_int(
            record.get("byte_length"),
            name=f"{filename} byte_length",
            minimum=1,
        )
        maximum = {
            "targets.csv": _MAX_TARGET_BYTES,
            "population_weights.csv": _MAX_POPULATION_WEIGHT_BYTES,
            "source_manifest.json": _MAX_MANIFEST_BYTES,
        }[filename]
        companion_bytes[filename] = _secure_read_regular_file(
            bundle_root / filename,
            expected_byte_length=expected_length,
            expected_sha256=expected_hash,
            maximum_bytes=maximum,
            description=filename,
            containing_root=bundle_root,
        )

    sources, raw_cache_root = _parse_and_verify_source_manifest(
        companion_bytes["source_manifest.json"],
        repository_root=repo_root,
        expected_bundle_id=bundle_id,
    )
    source_by_id = {source.source_id: source for source in sources}
    targets = _parse_targets_csv(
        companion_bytes["targets.csv"],
        source_by_id=source_by_id,
    )
    population_weights = _parse_population_weights_csv(
        companion_bytes["population_weights.csv"],
        source_by_id=source_by_id,
    )
    _validate_cross_file_contracts(
        raw_bundle,
        targets=targets,
        population_weights=population_weights,
        sources=sources,
        raw_cache_root=raw_cache_root,
    )
    blockers = _parse_blockers(raw_bundle.get("blockers"))
    _validate_blockers(blockers)
    return UKAdults2024CalibrationBundle(
        schema_version=schema_version,
        bundle_id=bundle_id,
        status=status,
        targets=targets,
        population_weights=population_weights,
        sources=sources,
        blockers=blockers,
        bundle_path=bundle_path.resolve(strict=True),
        bundle_sha256=sha256(bundle_bytes).hexdigest(),
    )


def load_calibration_bundle(
    path: str | Path = DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
    *,
    repository_root: str | Path | None = None,
) -> UKAdults2024CalibrationBundle:
    """Compatibility spelling for the only calibration schema currently known."""

    return load_uk_adults_2024_calibration_bundle(
        path,
        repository_root=repository_root,
    )


def _parse_and_verify_source_manifest(
    content: bytes,
    *,
    repository_root: Path,
    expected_bundle_id: str,
) -> tuple[tuple[CalibrationSource, ...], Path]:
    raw = _parse_json_object(content, name="source manifest")
    _exact_keys(raw, _SOURCE_MANIFEST_KEYS, name="source manifest")
    version = _strict_int(
        raw.get("schema_version"),
        name="source manifest schema_version",
        minimum=1,
    )
    if version != UK_ADULTS_2024_CALIBRATION_SCHEMA_VERSION:
        raise CalibrationBundleValidationError(
            "source manifest schema_version must be strict integer 1"
        )
    if _required_identifier(raw, "bundle_id") != expected_bundle_id:
        raise CalibrationBundleValidationError(
            "source manifest bundle_id differs from calibration bundle"
        )
    if raw.get("raw_cache_tracked") is not False:
        raise CalibrationBundleValidationError(
            "source manifest must declare raw_cache_tracked=false"
        )
    _parse_iso_date(
        _required_text(raw, "verified_at"),
        name="source manifest verified_at",
    )
    raw_cache_text = _required_text(raw, "raw_cache_root")
    _validate_relative_posix_path(raw_cache_text, name="raw_cache_root")
    raw_cache_root = repository_root.joinpath(
        *PurePosixPath(raw_cache_text).parts
    )
    _assert_directory_within_root(
        raw_cache_root,
        containing_root=repository_root,
        description="raw calibration cache",
    )

    raw_sources = raw.get("sources")
    if type(raw_sources) is not list or not raw_sources:
        raise CalibrationBundleValidationError(
            "source manifest sources must be a non-empty array"
        )
    sources: list[CalibrationSource] = []
    for index, item in enumerate(raw_sources):
        if not isinstance(item, Mapping):
            raise CalibrationBundleValidationError(
                f"source manifest sources[{index}] must be an object"
            )
        _exact_keys(item, _SOURCE_KEYS, name=f"source manifest sources[{index}]")
        source_id = _required_identifier(item, "source_id")
        relative_path = _required_text(item, "relative_path")
        _validate_relative_posix_path(
            relative_path,
            name=f"source {source_id} relative_path",
        )
        source_hash = _required_sha256(item, "sha256")
        byte_length = _strict_int(
            item.get("byte_length"),
            name=f"source {source_id} byte_length",
            minimum=1,
            maximum=_MAX_SOURCE_BYTES,
        )
        raw_official_url = item.get("official_url")
        if raw_official_url is None:
            official_url = None
        elif type(raw_official_url) is str:
            official_url = raw_official_url.strip()
            if not official_url:
                raise CalibrationBundleValidationError(
                    f"source {source_id} official_url must not be blank"
                )
            _validate_https_url(
                official_url,
                name=f"source {source_id} official_url",
            )
        else:
            raise CalibrationBundleValidationError(
                f"source {source_id} official_url must be null or an HTTPS URL"
            )
        raw_publication_date = item.get("publication_date")
        if raw_publication_date is None:
            publication_date = None
        elif type(raw_publication_date) is str:
            publication_date = _parse_iso_date(
                raw_publication_date,
                name=f"source {source_id} publication_date",
            )
        else:
            raise CalibrationBundleValidationError(
                f"source {source_id} publication_date must be null or an ISO "
                "calendar date"
            )
        raw_retrieved_at = item.get("retrieved_at")
        if raw_retrieved_at is None:
            retrieved_at = None
        elif type(raw_retrieved_at) is str:
            retrieved_at = _parse_aware_datetime(
                raw_retrieved_at,
                name=f"source {source_id} retrieved_at",
            )
        else:
            raise CalibrationBundleValidationError(
                f"source {source_id} retrieved_at must be null or an ISO-8601 "
                "datetime"
            )
        if (
            retrieved_at is not None
            and publication_date is not None
            and retrieved_at.date() < publication_date
        ):
            raise CalibrationBundleValidationError(
                f"source {source_id} retrieved_at predates publication_date"
            )
        try:
            evidence_role = SourceEvidenceRole(
                _required_text(item, "evidence_role")
            )
        except ValueError as exc:
            raise CalibrationBundleValidationError(
                f"source {source_id} evidence_role is invalid"
            ) from exc
        source = CalibrationSource(
            source_id=source_id,
            relative_path=relative_path,
            sha256=source_hash,
            byte_length=byte_length,
            official_url=official_url,
            licence=_required_text(item, "licence"),
            notes=_optional_trimmed_text(item, "notes"),
            publisher=_required_text(item, "publisher"),
            title=_required_text(item, "title"),
            version=_required_text(item, "version"),
            publication_date=publication_date,
            retrieved_at=retrieved_at,
            evidence_role=evidence_role,
        )
        source_path = raw_cache_root.joinpath(
            *PurePosixPath(source.relative_path).parts
        )
        _secure_read_regular_file(
            source_path,
            expected_byte_length=source.byte_length,
            expected_sha256=source.sha256,
            maximum_bytes=_MAX_SOURCE_BYTES,
            description=f"source {source.source_id}",
            containing_root=raw_cache_root,
        )
        sources.append(source)
    source_ids = tuple(source.source_id for source in sources)
    if len(set(source_ids)) != len(source_ids):
        raise CalibrationBundleValidationError("source manifest repeats source_id")
    if source_ids != tuple(sorted(source_ids)):
        raise CalibrationBundleValidationError(
            "source manifest sources must be ordered by source_id"
        )
    relative_paths = tuple(source.relative_path for source in sources)
    if len(set(relative_paths)) != len(relative_paths):
        raise CalibrationBundleValidationError(
            "source manifest repeats a relative_path"
        )
    return tuple(sources), raw_cache_root


def _parse_targets_csv(
    content: bytes,
    *,
    source_by_id: Mapping[str, CalibrationSource],
) -> tuple[CalibrationTarget, ...]:
    rows = _parse_strict_csv(
        content,
        columns=TARGET_CSV_COLUMNS,
        description="targets.csv",
    )
    targets: list[CalibrationTarget] = []
    for line_number, row in enumerate(rows, start=2):
        target_id = _csv_identifier(row, "target_id", line_number=line_number)
        concept = _csv_identifier(row, "concept", line_number=line_number)
        role = _csv_enum(
            row,
            "estimand_role",
            EstimandRole,
            line_number=line_number,
        )
        status = _csv_enum(
            row,
            "evidence_status",
            EvidenceStatus,
            line_number=line_number,
        )
        assert isinstance(role, EstimandRole)
        assert isinstance(status, EvidenceStatus)
        _validate_role_status(role, status, target_id=target_id)
        value = _csv_optional_decimal(row, "value", line_number=line_number)
        lower_ci = _csv_optional_decimal(
            row,
            "lower_ci",
            line_number=line_number,
        )
        upper_ci = _csv_optional_decimal(
            row,
            "upper_ci",
            line_number=line_number,
        )
        _validate_numeric_target_fields(
            target_id=target_id,
            status=status,
            value=value,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
        )
        geography = _csv_text(row, "geography", line_number=line_number)
        source_id = _csv_identifier(row, "source_id", line_number=line_number)
        try:
            source = source_by_id[source_id]
        except KeyError as exc:
            raise CalibrationBundleValidationError(
                f"targets.csv row {line_number} refers to unknown source_id"
            ) from exc
        source_file = _csv_text(row, "source_file", line_number=line_number)
        if source_file != source.relative_path:
            raise CalibrationBundleValidationError(
                f"target {target_id} source_file differs from source manifest"
            )
        _validate_target_source_role(target_id=target_id, role=role, source=source)
        targets.append(
            CalibrationTarget(
                target_id=target_id,
                concept=concept,
                estimand_role=role,
                evidence_status=status,
                geography=geography,
                population=_csv_text(row, "population", line_number=line_number),
                reference_period=_csv_text(
                    row,
                    "reference_period",
                    line_number=line_number,
                ),
                subgroup=_csv_text(row, "subgroup", line_number=line_number),
                value=value,
                unit=_csv_text(row, "unit", line_number=line_number),
                lower_ci=lower_ci,
                upper_ci=upper_ci,
                source_id=source_id,
                source_file=source_file,
                source_locator=_csv_text(
                    row,
                    "source_locator",
                    line_number=line_number,
                ),
                transformation=_csv_text(
                    row,
                    "transformation",
                    line_number=line_number,
                ),
                runtime_mapping=_csv_text(
                    row,
                    "runtime_mapping",
                    line_number=line_number,
                ),
                limitations=_csv_text(
                    row,
                    "limitations",
                    line_number=line_number,
                ),
            )
        )
    target_ids = tuple(target.target_id for target in targets)
    if len(set(target_ids)) != len(target_ids):
        raise CalibrationBundleValidationError("targets.csv repeats target_id")
    if not any(target.estimand_role is EstimandRole.CALIBRATION for target in targets):
        raise CalibrationBundleValidationError(
            "targets.csv must contain at least one CALIBRATION target"
        )
    if not any(target.estimand_role is EstimandRole.VALIDATION for target in targets):
        raise CalibrationBundleValidationError(
            "targets.csv must contain at least one VALIDATION target"
        )
    _validate_disjoint_target_records(targets)
    _validate_unsupported_targets(targets)
    return tuple(targets)


def _parse_population_weights_csv(
    content: bytes,
    *,
    source_by_id: Mapping[str, CalibrationSource],
) -> tuple[PopulationWeight, ...]:
    rows = _parse_strict_csv(
        content,
        columns=POPULATION_WEIGHT_CSV_COLUMNS,
        description="population_weights.csv",
    )
    weights: list[PopulationWeight] = []
    for line_number, row in enumerate(rows, start=2):
        role = _csv_enum(
            row,
            "estimand_role",
            EstimandRole,
            line_number=line_number,
        )
        assert isinstance(role, EstimandRole)
        if role is not EstimandRole.CALIBRATION:
            raise CalibrationBundleValidationError(
                f"population_weights.csv row {line_number} must be CALIBRATION"
            )
        source_id = _csv_identifier(row, "source_id", line_number=line_number)
        try:
            source = source_by_id[source_id]
        except KeyError as exc:
            raise CalibrationBundleValidationError(
                f"population_weights.csv row {line_number} has unknown source_id"
            ) from exc
        if source.evidence_role not in {
            SourceEvidenceRole.CALIBRATION,
            SourceEvidenceRole.MIXED,
        }:
            raise CalibrationBundleValidationError(
                f"population weight source {source_id} is not pre-assigned to "
                "CALIBRATION"
            )
        age_min = _csv_nonnegative_integer(
            row,
            "age_min_inclusive",
            line_number=line_number,
        )
        age_max = _csv_nonnegative_integer(
            row,
            "age_max_inclusive",
            line_number=line_number,
        )
        if age_max < age_min:
            raise CalibrationBundleValidationError(
                f"population_weights.csv row {line_number} has reversed ages"
            )
        weight = _csv_required_decimal(
            row,
            "adult_population_weight",
            line_number=line_number,
        )
        if weight <= 0 or weight >= 1:
            raise CalibrationBundleValidationError(
                f"population_weights.csv row {line_number} weight must be in (0, 1)"
            )
        weights.append(
            PopulationWeight(
                age_band=_csv_text(row, "age_band", line_number=line_number),
                age_min_inclusive=age_min,
                age_max_inclusive=age_max,
                sex=_csv_text(row, "sex", line_number=line_number),
                population_count=_csv_positive_integer(
                    row,
                    "population_count",
                    line_number=line_number,
                ),
                adult_population_weight=weight,
                source_id=source_id,
                estimand_role=role,
            )
        )
    _validate_exact_population_weights(weights)
    return tuple(weights)


def _validate_cross_file_contracts(
    raw_bundle: Mapping[str, object],
    *,
    targets: Sequence[CalibrationTarget],
    population_weights: Sequence[PopulationWeight],
    sources: Sequence[CalibrationSource],
    raw_cache_root: Path,
) -> None:
    del population_weights, sources, raw_cache_root
    unsupported = raw_bundle.get("unsupported_concepts")
    if type(unsupported) is not list or tuple(unsupported) != UNSUPPORTED_CONCEPTS:
        raise CalibrationBundleValidationError(
            "unsupported_concepts must exactly match the schema-v1 ordered set"
        )
    margin = _required_mapping(raw_bundle, "frs_rounded_margin")
    _exact_keys(margin, _FRS_MARGIN_KEYS, name="frs_rounded_margin")
    target_id = _required_identifier(margin, "target_id")
    if target_id not in {target.target_id for target in targets}:
        raise CalibrationBundleValidationError(
            "frs_rounded_margin target_id does not identify a target row"
        )
    published_sum = _strict_int(
        margin.get("published_sum_percent"),
        name="frs_rounded_margin published_sum_percent",
        minimum=0,
    )
    if published_sum != 101:
        raise CalibrationBundleValidationError(
            "FRS published rounded margins must retain their observed 101 percent sum"
        )
    if margin.get("normalization_applied") is not False:
        raise CalibrationBundleValidationError(
            "FRS rounded margins must not be silently normalized"
        )


def _validate_role_status(
    role: EstimandRole,
    status: EvidenceStatus,
    *,
    target_id: str,
) -> None:
    if role in {EstimandRole.CALIBRATION, EstimandRole.VALIDATION}:
        if status is not EvidenceStatus.QUANTIFIED:
            raise CalibrationBundleValidationError(
                f"target {target_id} with role {role.value} must be QUANTIFIED"
            )


def _validate_numeric_target_fields(
    *,
    target_id: str,
    status: EvidenceStatus,
    value: Decimal | None,
    lower_ci: Decimal | None,
    upper_ci: Decimal | None,
) -> None:
    quantified = status in {EvidenceStatus.QUANTIFIED, EvidenceStatus.PROXY}
    if quantified and value is None:
        raise CalibrationBundleValidationError(
            f"target {target_id} with status {status.value} needs a numeric value"
        )
    if not quantified and any(item is not None for item in (value, lower_ci, upper_ci)):
        raise CalibrationBundleValidationError(
            f"target {target_id} with status {status.value} must not carry numbers"
        )
    if (lower_ci is None) != (upper_ci is None):
        raise CalibrationBundleValidationError(
            f"target {target_id} confidence bounds must be both present or both absent"
        )
    if lower_ci is not None and value is not None and upper_ci is not None:
        if not lower_ci <= value <= upper_ci:
            raise CalibrationBundleValidationError(
                f"target {target_id} value lies outside its confidence interval"
            )


def _validate_target_source_role(
    *,
    target_id: str,
    role: EstimandRole,
    source: CalibrationSource,
) -> None:
    permitted = {
        EstimandRole.CALIBRATION: {
            SourceEvidenceRole.CALIBRATION,
            SourceEvidenceRole.MIXED,
        },
        EstimandRole.VALIDATION: {
            SourceEvidenceRole.VALIDATION,
            SourceEvidenceRole.MIXED,
        },
        EstimandRole.DIAGNOSTIC: {
            SourceEvidenceRole.DIAGNOSTIC,
            SourceEvidenceRole.REFERENCE,
            SourceEvidenceRole.MIXED,
        },
    }[role]
    if source.evidence_role not in permitted:
        raise CalibrationBundleValidationError(
            f"target {target_id} role conflicts with source {source.source_id} "
            "evidence_role"
        )


def _validate_disjoint_target_records(
    targets: Sequence[CalibrationTarget],
) -> None:
    def record_key(target: CalibrationTarget) -> tuple[str, str, str]:
        return (target.source_id, target.source_file, target.source_locator)

    calibration_records = {
        record_key(target)
        for target in targets
        if target.estimand_role is EstimandRole.CALIBRATION
    }
    validation_records = {
        record_key(target)
        for target in targets
        if target.estimand_role is EstimandRole.VALIDATION
    }
    overlap = sorted(calibration_records.intersection(validation_records))
    if overlap:
        raise CalibrationBundleValidationError(
            "CALIBRATION and VALIDATION targets reuse source records: "
            + repr(overlap)
        )
    calibration_source_ids = {
        target.source_id
        for target in targets
        if target.estimand_role is EstimandRole.CALIBRATION
    }
    validation_source_ids = {
        target.source_id
        for target in targets
        if target.estimand_role is EstimandRole.VALIDATION
    }
    overlapping_sources = sorted(
        calibration_source_ids.intersection(validation_source_ids)
    )
    if overlapping_sources:
        raise CalibrationBundleValidationError(
            "CALIBRATION and VALIDATION targets must use disjoint source_id values: "
            + ", ".join(overlapping_sources)
        )


def _validate_unsupported_targets(
    targets: Sequence[CalibrationTarget],
) -> None:
    for concept, required_status in UNSUPPORTED_CONCEPT_STATUSES.items():
        matches = [target for target in targets if target.concept == concept]
        if len(matches) != 1:
            raise CalibrationBundleValidationError(
                f"unsupported concept {concept} must appear in exactly one target row"
            )
        target = matches[0]
        if target.estimand_role is EstimandRole.CALIBRATION:
            raise CalibrationBundleValidationError(
                f"unsupported concept {concept} cannot be promoted to CALIBRATION"
            )
        if target.evidence_status.value != required_status:
            raise CalibrationBundleValidationError(
                f"unsupported concept {concept} must remain {required_status}"
            )


def _validate_exact_population_weights(
    weights: Sequence[PopulationWeight],
) -> None:
    observed_keys = tuple(
        (
            weight.age_band,
            weight.age_min_inclusive,
            weight.age_max_inclusive,
            weight.sex,
        )
        for weight in weights
    )
    expected_keys = tuple(_EXPECTED_POPULATION_COUNTS)
    if observed_keys != expected_keys:
        raise CalibrationBundleValidationError(
            "population weights must contain the canonical ordered UK 18--64 "
            "age-band-by-sex cells"
        )
    for weight in weights:
        key = (
            weight.age_band,
            weight.age_min_inclusive,
            weight.age_max_inclusive,
            weight.sex,
        )
        expected_count = _EXPECTED_POPULATION_COUNTS[key]
        if weight.population_count != expected_count:
            raise CalibrationBundleValidationError(
                f"population cell {weight.age_band}/{weight.sex} count differs "
                "from the extracted ONS mid-2024 value"
            )
        expected_weight = Decimal(expected_count) / Decimal(
            UK_ADULTS_2024_POPULATION_COUNT
        )
        if (
            abs(weight.adult_population_weight - expected_weight)
            > _POPULATION_WEIGHT_TOLERANCE
        ):
            raise CalibrationBundleValidationError(
                f"population cell {weight.age_band}/{weight.sex} weight is not "
                "count / 41860587"
            )
    total_count = sum(weight.population_count for weight in weights)
    if total_count != UK_ADULTS_2024_POPULATION_COUNT:
        raise CalibrationBundleValidationError(
            "population weights do not total 41,860,587 UK adults aged 18--64"
        )
    total_weight = sum(
        (weight.adult_population_weight for weight in weights),
        start=Decimal(0),
    )
    if abs(total_weight - Decimal(1)) > _POPULATION_WEIGHT_TOLERANCE:
        raise CalibrationBundleValidationError(
            "adult_population_weight values must sum to one"
        )


def _validate_blockers(blockers: tuple[str, ...]) -> None:
    joined = "\n".join(blockers)
    missing = [concept for concept in UNSUPPORTED_CONCEPTS if concept not in joined]
    if missing:
        raise CalibrationBundleValidationError(
            "campaign blockers must name every unsupported concept: "
            + ", ".join(missing)
        )


def _parse_blockers(value: object) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise CalibrationBundleValidationError(
            "blockers must be a non-empty array of strings"
        )
    blockers: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str or not item.strip() or item != item.strip():
            raise CalibrationBundleValidationError(
                f"blockers[{index}] must be non-empty trimmed text"
            )
        blockers.append(item)
    if len(set(blockers)) != len(blockers):
        raise CalibrationBundleValidationError("blockers must not contain duplicates")
    if tuple(blockers) != tuple(sorted(blockers)):
        raise CalibrationBundleValidationError("blockers must use ascending order")
    return tuple(blockers)


def _parse_strict_csv(
    content: bytes,
    *,
    columns: Sequence[str],
    description: str,
) -> tuple[dict[str, str], ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CalibrationBundleVerificationError(
            f"{description} must be strict UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise CalibrationBundleVerificationError(
            f"{description} must not contain a UTF-8 BOM"
        )
    if "\x00" in text:
        raise CalibrationBundleVerificationError(
            f"{description} contains a NUL character"
        )
    try:
        parsed = list(
            csv.reader(
                io.StringIO(text, newline=""),
                delimiter=",",
                quotechar='"',
                doublequote=True,
                escapechar=None,
                skipinitialspace=False,
                strict=True,
            )
        )
    except csv.Error as exc:
        raise CalibrationBundleVerificationError(
            f"{description} is not strict CSV"
        ) from exc
    if len(parsed) < 2:
        raise CalibrationBundleVerificationError(
            f"{description} requires a header and at least one data row"
        )
    if parsed[0] != list(columns):
        raise CalibrationBundleVerificationError(
            f"{description} header does not exactly match schema version 1"
        )
    width = len(columns)
    rows: list[dict[str, str]] = []
    for line_number, row in enumerate(parsed[1:], start=2):
        if len(row) != width:
            raise CalibrationBundleVerificationError(
                f"{description} row {line_number} has wrong width"
            )
        if any("\r" in value or "\n" in value for value in row):
            raise CalibrationBundleVerificationError(
                f"{description} row {line_number} contains multiline data"
            )
        rows.append(dict(zip(columns, row, strict=True)))
    return tuple(rows)


def _parse_json_object(content: bytes, *, name: str) -> Mapping[str, object]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CalibrationBundleVerificationError(
            f"{name} must be strict UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise CalibrationBundleVerificationError(
            f"{name} must not contain a UTF-8 BOM"
        )
    if "\x00" in text:
        raise CalibrationBundleVerificationError(
            f"{name} contains a NUL character"
        )

    def reject_constant(value: str) -> object:
        raise CalibrationBundleValidationError(
            f"{name} contains non-finite JSON number {value}"
        )

    def reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise CalibrationBundleValidationError(
                    f"{name} repeats JSON key {key!r}"
                )
            output[key] = value
        return output

    try:
        raw = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_pairs,
        )
    except json.JSONDecodeError as exc:
        raise CalibrationBundleValidationError(f"invalid {name} JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise CalibrationBundleValidationError(f"{name} root must be an object")
    return raw


def _secure_read_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
    expected_byte_length: int | None = None,
    expected_sha256: str | None = None,
    containing_root: Path | None = None,
) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CalibrationBundleVerificationError(
            f"{description} cannot be read: {path}"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise CalibrationBundleVerificationError(
            f"{description} must be a regular file, not a link"
        )
    if metadata.st_size <= 0 or metadata.st_size > maximum_bytes:
        raise CalibrationBundleVerificationError(
            f"{description} byte length is outside the permitted range"
        )
    if expected_byte_length is not None and metadata.st_size != expected_byte_length:
        raise CalibrationBundleVerificationError(
            f"{description} byte length differs from its declaration"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CalibrationBundleVerificationError(
            f"{description} cannot be resolved"
        ) from exc
    if containing_root is not None:
        try:
            resolved_root = containing_root.resolve(strict=True)
        except OSError as exc:
            raise CalibrationBundleVerificationError(
                f"containing root for {description} cannot be resolved"
            ) from exc
        if not resolved.is_relative_to(resolved_root):
            raise CalibrationBundleVerificationError(
                f"{description} escapes its declared root"
            )
    try:
        with path.open("rb") as stream:
            content = stream.read(maximum_bytes + 1)
    except OSError as exc:
        raise CalibrationBundleVerificationError(
            f"{description} cannot be read"
        ) from exc
    if len(content) > maximum_bytes or len(content) != metadata.st_size:
        raise CalibrationBundleVerificationError(
            f"{description} changed or exceeded its size limit while reading"
        )
    if expected_sha256 is not None and sha256(content).hexdigest() != expected_sha256:
        raise CalibrationBundleVerificationError(
            f"{description} SHA-256 differs from its declaration"
        )
    return content


def _resolve_repository_root(
    bundle_root: Path,
    *,
    repository_root: str | Path | None,
) -> Path:
    if repository_root is None:
        if bundle_root.is_relative_to(_PROJECT_ROOT):
            candidate = _PROJECT_ROOT
        else:
            raise CalibrationBundleValidationError(
                "repository_root is required for a bundle outside the project checkout"
            )
    else:
        candidate = Path(repository_root)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise CalibrationBundleVerificationError(
            "repository_root cannot be resolved"
        ) from exc
    if not resolved.is_dir():
        raise CalibrationBundleVerificationError(
            "repository_root must be a directory"
        )
    return resolved


def _assert_directory_within_root(
    path: Path,
    *,
    containing_root: Path,
    description: str,
) -> None:
    try:
        resolved = path.resolve(strict=True)
        root = containing_root.resolve(strict=True)
    except OSError as exc:
        raise CalibrationBundleVerificationError(
            f"{description} cannot be resolved"
        ) from exc
    if not resolved.is_dir() or not resolved.is_relative_to(root):
        raise CalibrationBundleVerificationError(
            f"{description} must be a directory inside repository_root"
        )


def _validate_relative_posix_path(value: str, *, name: str) -> None:
    path = PurePosixPath(value)
    windows_reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(":" in part for part in path.parts)
        or any(part.split(".", maxsplit=1)[0].upper() in windows_reserved for part in path.parts)
    ):
        raise CalibrationBundleValidationError(
            f"{name} must be a safe repository-relative POSIX path"
        )


def _validate_https_url(value: str, *, name: str) -> None:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise CalibrationBundleValidationError(
            f"{name} must be an HTTPS URL without credentials or a fragment"
        )


def _parse_iso_date(value: str, *, name: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise CalibrationBundleValidationError(
            f"{name} must be an ISO calendar date"
        ) from exc
    if parsed.isoformat() != value:
        raise CalibrationBundleValidationError(
            f"{name} must use canonical YYYY-MM-DD format"
        )
    return parsed


def _parse_aware_datetime(value: str, *, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CalibrationBundleValidationError(
            f"{name} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CalibrationBundleValidationError(
            f"{name} must include an explicit UTC offset"
        )
    if parsed.isoformat() != value:
        raise CalibrationBundleValidationError(
            f"{name} must use Python's canonical ISO-8601 representation"
        )
    return parsed


def _exact_keys(
    values: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if any(type(key) is not str for key in values):
        raise CalibrationBundleValidationError(f"{name} keys must be strings")
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise CalibrationBundleValidationError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


def _required_mapping(
    values: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    value = values.get(field)
    if not isinstance(value, Mapping):
        raise CalibrationBundleValidationError(f"{field} must be an object")
    return value


def _required_text(values: Mapping[str, object], field: str) -> str:
    value = values.get(field)
    if type(value) is not str or not value.strip() or value != value.strip():
        raise CalibrationBundleValidationError(
            f"{field} must be non-empty text without surrounding whitespace"
        )
    return value


def _optional_trimmed_text(values: Mapping[str, object], field: str) -> str:
    value = values.get(field)
    if type(value) is not str or value != value.strip():
        raise CalibrationBundleValidationError(
            f"{field} must be text without surrounding whitespace"
        )
    return value


def _required_identifier(values: Mapping[str, object], field: str) -> str:
    value = values.get(field)
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise CalibrationBundleValidationError(
            f"{field} must be a canonical ASCII identifier"
        )
    return value


def _required_sha256(values: Mapping[str, object], field: str) -> str:
    value = values.get(field)
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise CalibrationBundleValidationError(
            f"{field} must be lowercase SHA-256 hex"
        )
    return value


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise CalibrationBundleValidationError(f"{name} must be a strict integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise CalibrationBundleValidationError(f"{name} must be {bounds}")
    return value


def _csv_text(
    row: Mapping[str, str],
    field: str,
    *,
    line_number: int,
) -> str:
    value = row[field]
    if not value.strip() or value != value.strip():
        raise CalibrationBundleValidationError(
            f"CSV row {line_number} field {field} must be non-empty trimmed text"
        )
    return value


def _csv_identifier(
    row: Mapping[str, str],
    field: str,
    *,
    line_number: int,
) -> str:
    value = row[field]
    if _ID_PATTERN.fullmatch(value) is None:
        raise CalibrationBundleValidationError(
            f"CSV row {line_number} field {field} is not a canonical identifier"
        )
    return value


def _csv_enum(
    row: Mapping[str, str],
    field: str,
    enum_type: type[Enum],
    *,
    line_number: int,
) -> Enum:
    try:
        return enum_type(row[field])
    except ValueError as exc:
        raise CalibrationBundleValidationError(
            f"CSV row {line_number} field {field} is invalid"
        ) from exc


def _csv_optional_decimal(
    row: Mapping[str, str],
    field: str,
    *,
    line_number: int,
) -> Decimal | None:
    value = row[field]
    if value == "":
        return None
    return _parse_canonical_decimal(
        value,
        name=f"CSV row {line_number} field {field}",
    )


def _csv_required_decimal(
    row: Mapping[str, str],
    field: str,
    *,
    line_number: int,
) -> Decimal:
    value = row[field]
    if not value:
        raise CalibrationBundleValidationError(
            f"CSV row {line_number} field {field} requires a decimal"
        )
    return _parse_canonical_decimal(
        value,
        name=f"CSV row {line_number} field {field}",
    )


def _parse_canonical_decimal(value: str, *, name: str) -> Decimal:
    if _CANONICAL_DECIMAL_PATTERN.fullmatch(value) is None:
        raise CalibrationBundleValidationError(
            f"{name} must be a canonical finite decimal without exponent notation"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise CalibrationBundleValidationError(f"{name} is not a decimal") from exc
    if not parsed.is_finite():
        raise CalibrationBundleValidationError(f"{name} must be finite")
    return parsed


def _csv_positive_integer(
    row: Mapping[str, str],
    field: str,
    *,
    line_number: int,
) -> int:
    value = row[field]
    if _POSITIVE_INTEGER_PATTERN.fullmatch(value) is None:
        raise CalibrationBundleValidationError(
            f"CSV row {line_number} field {field} must be a canonical positive integer"
        )
    return int(value)


def _csv_nonnegative_integer(
    row: Mapping[str, str],
    field: str,
    *,
    line_number: int,
) -> int:
    value = row[field]
    if _NONNEGATIVE_INTEGER_PATTERN.fullmatch(value) is None:
        raise CalibrationBundleValidationError(
            f"CSV row {line_number} field {field} must be a canonical non-negative "
            "integer"
        )
    return int(value)


__all__ = [
    "CalibrationBundleValidationError",
    "CalibrationBundleVerificationError",
    "CalibrationSource",
    "CalibrationTarget",
    "DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH",
    "EstimandRole",
    "EvidenceStatus",
    "POPULATION_WEIGHT_CSV_COLUMNS",
    "PopulationWeight",
    "SourceEvidenceRole",
    "TARGET_CSV_COLUMNS",
    "UKAdults2024CalibrationBundle",
    "UK_ADULTS_2024_CALIBRATION_BUNDLE_ID",
    "UK_ADULTS_2024_CALIBRATION_SCHEMA_VERSION",
    "UK_ADULTS_2024_POPULATION_COUNT",
    "UNSUPPORTED_CONCEPTS",
    "UNSUPPORTED_CONCEPT_STATUSES",
    "load_calibration_bundle",
    "load_uk_adults_2024_calibration_bundle",
]
