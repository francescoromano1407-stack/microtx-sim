"""Fail-closed, content-addressed evidence for exact monetary rates.

Schema version 1 deliberately supports only bundles with a missing signature
and tightly specified CSV interpreters for an exact rational table or an
official ECB annual EXR response. It can prove that a declared rational was
extracted from particular bytes by a particular recipe; it cannot prove that
the publisher, rate choice, or downstream estimand is scientifically valid.
Consequently every v1 bundle remains campaign-blocking.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tomllib
from typing import Mapping

from ..types import ProvenanceStatus


RATE_EVIDENCE_SCHEMA_VERSION = 1
EXACT_CSV_INTERPRETER_V1 = "exact_csv_positive_rational/1"
ECB_EXR_SOURCE_PER_EUR_INTERPRETER_V1 = (
    "ecb_exr_source_per_eur_to_target_minor_rational/1"
)
MAX_RATE_ARTIFACT_BYTES = 16 * 1024 * 1024
MAX_RATE_BUNDLE_BYTES = 1024 * 1024

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RATE_EVIDENCE_BUNDLE_PATH = (
    _PROJECT_ROOT / "data" / "provenance" / "source_bundle.toml"
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "bundle_id",
        "provenance_status",
        "source_registry_sha256",
        "artifact_root",
        "notes",
        "artifacts",
        "bindings",
        "signature",
    }
)
_ARTIFACT_KEYS = frozenset(
    {"artifact_id", "relative_path", "media_type", "sha256", "byte_length"}
)
_BINDING_KEYS = frozenset(
    {
        "binding_id",
        "artifact_id",
        "source_id",
        "jurisdiction_code",
        "source_currency",
        "target_currency",
        "method",
        "rate_period_start",
        "rate_period_end",
        "retrieved_on",
        "rate_numerator",
        "rate_denominator",
        "recipe_json",
    }
)
_SIGNATURE_KEYS = frozenset({"status", "algorithm", "key_id", "value"})
_BUNDLE_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "bundle_id",
        "provenance_status",
        "source_registry_sha256",
        "artifact_root",
        "notes",
        "artifacts",
        "bindings",
        "signature",
        "bundle_path",
        "bundle_sha256",
        "bundle_byte_length",
        "campaign_ready",
        "campaign_blockers",
    }
)
_BINDING_SNAPSHOT_KEYS = _BINDING_KEYS.union(
    {"rate_numerator_decimal", "rate_denominator_decimal"}
)
_RESULT_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "bundle_sha256",
        "source_registry_sha256",
        "binding_id",
        "binding_sha256",
        "artifact_id",
        "artifact_sha256",
        "artifact_byte_length",
        "recipe_sha256",
        "rate_numerator",
        "rate_denominator",
        "rate_numerator_decimal",
        "rate_denominator_decimal",
        "evidence_sha256",
    }
)
_RECIPE_KEYS = frozenset(
    {
        "schema_version",
        "interpreter",
        "row_match",
        "numerator_column",
        "denominator_column",
    }
)
_ECB_RECIPE_KEYS = frozenset(
    {
        "schema_version",
        "interpreter",
        "row_match",
        "source_row_match",
        "observation_column",
        "source_minor_unit_exponent",
        "target_minor_unit_exponent",
    }
)
_ROW_MATCH_KEYS = (
    "jurisdiction_code",
    "method",
    "rate_period_end",
    "rate_period_start",
    "retrieved_on",
    "source_currency",
    "source_id",
    "target_currency",
)
_ROW_MATCH_KEY_SET = frozenset(_ROW_MATCH_KEYS)
_ECB_SOURCE_ROW_MATCH_KEYS = frozenset(
    {
        "KEY",
        "FREQ",
        "CURRENCY",
        "CURRENCY_DENOM",
        "EXR_TYPE",
        "EXR_SUFFIX",
        "TIME_PERIOD",
    }
)
_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_COLUMN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_POSITIVE_INTEGER_PATTERN = re.compile(r"[1-9][0-9]*\Z")
_POSITIVE_DECIMAL_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z"
)


class RateEvidenceValidationError(ValueError):
    """Raised when a rate-evidence declaration is malformed."""


class RateEvidenceVerificationError(RateEvidenceValidationError):
    """Raised when declared evidence does not match the observed bytes."""


class RateEvidenceMethod(str, Enum):
    """Rate families permitted by schema version 1."""

    FX = "FX"
    PPP = "PPP"


class RateEvidenceSignatureStatus(str, Enum):
    """Signature states understood by schema version 1."""

    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class RateEvidenceArtifact:
    """One regular CSV file whose complete byte sequence is declared."""

    artifact_id: str
    relative_path: str
    media_type: str
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        _validate_id(self.artifact_id, name="artifact_id")
        _validate_relative_posix_path(
            self.relative_path,
            name=f"artifact {self.artifact_id} relative_path",
        )
        if self.media_type != "text/csv":
            raise RateEvidenceValidationError(
                f"artifact {self.artifact_id} media_type must be text/csv"
            )
        _validate_sha256(self.sha256, name=f"artifact {self.artifact_id} sha256")
        _validate_strict_int(
            self.byte_length,
            name=f"artifact {self.artifact_id} byte_length",
            minimum=1,
            maximum=MAX_RATE_ARTIFACT_BYTES,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "sha256": self.sha256,
            "byte_length": self.byte_length,
        }


@dataclass(frozen=True, slots=True)
class RateEvidenceSignature:
    """Explicitly missing signature declaration for schema version 1."""

    status: RateEvidenceSignatureStatus
    algorithm: str
    key_id: str
    value: str

    def __post_init__(self) -> None:
        if type(self.status) is not RateEvidenceSignatureStatus:
            raise RateEvidenceValidationError(
                "rate-evidence signature status is invalid"
            )
        if self.status is not RateEvidenceSignatureStatus.MISSING:
            raise RateEvidenceValidationError(
                "rate-evidence schema v1 supports missing signatures only"
            )
        if (self.algorithm, self.key_id, self.value) != ("NONE", "", ""):
            raise RateEvidenceValidationError(
                "a missing rate-evidence signature must use algorithm NONE and "
                "empty key/value fields"
            )

    def snapshot(self) -> dict[str, str]:
        return {
            "status": self.status.value,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class RateEvidenceBinding:
    """Exact semantic selector and expected rational for one source row."""

    binding_id: str
    artifact_id: str
    source_id: str
    jurisdiction_code: str
    source_currency: str
    target_currency: str
    method: RateEvidenceMethod
    rate_period_start: date
    rate_period_end: date
    retrieved_on: date
    rate_numerator: int
    rate_denominator: int
    recipe_json: str

    def __post_init__(self) -> None:
        _validate_id(self.binding_id, name="binding_id")
        _validate_id(self.artifact_id, name="artifact_id")
        _validate_id(self.source_id, name="source_id")
        _validate_jurisdiction_code(self.jurisdiction_code)
        _validate_currency(self.source_currency, name="source_currency")
        _validate_currency(self.target_currency, name="target_currency")
        if type(self.method) is not RateEvidenceMethod:
            raise RateEvidenceValidationError("rate-evidence method must be FX or PPP")
        for name in ("rate_period_start", "rate_period_end", "retrieved_on"):
            if type(getattr(self, name)) is not date:
                raise RateEvidenceValidationError(
                    f"rate-evidence {name} must be an ISO calendar date"
                )
        if self.rate_period_end < self.rate_period_start:
            raise RateEvidenceValidationError("rate period ends before it starts")
        if self.retrieved_on < self.rate_period_end:
            raise RateEvidenceValidationError(
                "rate retrieval date cannot predate the rate-period end"
            )
        _validate_strict_int(
            self.rate_numerator,
            name="rate_numerator",
            minimum=1,
        )
        _validate_strict_int(
            self.rate_denominator,
            name="rate_denominator",
            minimum=1,
        )
        if math.gcd(self.rate_numerator, self.rate_denominator) != 1:
            raise RateEvidenceValidationError(
                "declared rate numerator and denominator must be in lowest terms"
            )
        recipe = _parse_recipe_json(self.recipe_json)
        expected_match = self.row_match
        if recipe["row_match"] != expected_match:
            raise RateEvidenceValidationError(
                f"binding {self.binding_id} recipe row_match does not exactly "
                "match its typed metadata"
            )
        if recipe["interpreter"] == ECB_EXR_SOURCE_PER_EUR_INTERPRETER_V1:
            if self.method is not RateEvidenceMethod.FX:
                raise RateEvidenceValidationError(
                    "the ECB EXR interpreter is valid only for FX bindings"
                )
            if (
                self.rate_period_start != date(self.rate_period_start.year, 1, 1)
                or self.rate_period_end
                != date(self.rate_period_start.year, 12, 31)
            ):
                raise RateEvidenceValidationError(
                    "the ECB annual EXR interpreter requires one calendar year"
                )
            expected_source_match = {
                "KEY": (
                    f"EXR.A.{self.source_currency}."
                    f"{self.target_currency}.SP00.A"
                ),
                "FREQ": "A",
                "CURRENCY": self.source_currency,
                "CURRENCY_DENOM": self.target_currency,
                "EXR_TYPE": "SP00",
                "EXR_SUFFIX": "A",
                "TIME_PERIOD": str(self.rate_period_start.year),
            }
            if recipe["source_row_match"] != expected_source_match:
                raise RateEvidenceValidationError(
                    f"binding {self.binding_id} ECB source_row_match does not "
                    "match its typed metadata"
                )

    @property
    def rate(self) -> Fraction:
        """Return the declared rate without floating-point conversion."""

        return Fraction(self.rate_numerator, self.rate_denominator)

    @property
    def row_match(self) -> dict[str, str]:
        return {
            "jurisdiction_code": self.jurisdiction_code,
            "method": self.method.value,
            "rate_period_end": self.rate_period_end.isoformat(),
            "rate_period_start": self.rate_period_start.isoformat(),
            "retrieved_on": self.retrieved_on.isoformat(),
            "source_currency": self.source_currency,
            "source_id": self.source_id,
            "target_currency": self.target_currency,
        }

    @property
    def recipe_sha256(self) -> str:
        return sha256(self.recipe_json.encode("utf-8")).hexdigest()

    @property
    def binding_sha256(self) -> str:
        return sha256(_canonical_json(self.snapshot()).encode("utf-8")).hexdigest()

    def snapshot(self) -> dict[str, object]:
        return {
            "binding_id": self.binding_id,
            "artifact_id": self.artifact_id,
            "source_id": self.source_id,
            "jurisdiction_code": self.jurisdiction_code,
            "source_currency": self.source_currency,
            "target_currency": self.target_currency,
            "method": self.method.value,
            "rate_period_start": self.rate_period_start.isoformat(),
            "rate_period_end": self.rate_period_end.isoformat(),
            "retrieved_on": self.retrieved_on.isoformat(),
            "rate_numerator": self.rate_numerator,
            "rate_denominator": self.rate_denominator,
            "rate_numerator_decimal": str(self.rate_numerator),
            "rate_denominator_decimal": str(self.rate_denominator),
            "recipe_json": self.recipe_json,
        }


@dataclass(frozen=True, slots=True)
class RateEvidenceResult:
    """Verified extraction result tied to bundle, artifact, binding, and recipe."""

    bundle_sha256: str
    source_registry_sha256: str
    binding_id: str
    binding_sha256: str
    artifact_id: str
    artifact_sha256: str
    artifact_byte_length: int
    recipe_sha256: str
    rate_numerator: int
    rate_denominator: int
    evidence_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "bundle_sha256",
            "source_registry_sha256",
            "binding_sha256",
            "artifact_sha256",
            "recipe_sha256",
            "evidence_sha256",
        ):
            _validate_sha256(getattr(self, name), name=name)
        _validate_id(self.binding_id, name="binding_id")
        _validate_id(self.artifact_id, name="artifact_id")
        _validate_strict_int(
            self.artifact_byte_length,
            name="artifact_byte_length",
            minimum=1,
            maximum=MAX_RATE_ARTIFACT_BYTES,
        )
        _validate_strict_int(self.rate_numerator, name="rate_numerator", minimum=1)
        _validate_strict_int(
            self.rate_denominator,
            name="rate_denominator",
            minimum=1,
        )
        if math.gcd(self.rate_numerator, self.rate_denominator) != 1:
            raise RateEvidenceValidationError(
                "verified rate numerator and denominator must be in lowest terms"
            )
        expected = sha256(
            _canonical_json(self.attestation_payload()).encode("utf-8")
        ).hexdigest()
        if self.evidence_sha256 != expected:
            raise RateEvidenceValidationError(
                "evidence_sha256 does not match the verified result payload"
            )

    @property
    def rate(self) -> Fraction:
        return Fraction(self.rate_numerator, self.rate_denominator)

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": RATE_EVIDENCE_SCHEMA_VERSION,
            "bundle_sha256": self.bundle_sha256,
            "source_registry_sha256": self.source_registry_sha256,
            "binding_id": self.binding_id,
            "binding_sha256": self.binding_sha256,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "artifact_byte_length": self.artifact_byte_length,
            "recipe_sha256": self.recipe_sha256,
            "rate_numerator": self.rate_numerator,
            "rate_denominator": self.rate_denominator,
            "rate_numerator_decimal": str(self.rate_numerator),
            "rate_denominator_decimal": str(self.rate_denominator),
        }

    def snapshot(self) -> dict[str, object]:
        return {**self.attestation_payload(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class RateEvidenceBundle:
    """Parsed schema-v1 bundle plus its own observed file identity."""

    schema_version: int
    bundle_id: str
    provenance_status: ProvenanceStatus
    source_registry_sha256: str
    artifact_root: str
    notes: str
    artifacts: tuple[RateEvidenceArtifact, ...]
    bindings: tuple[RateEvidenceBinding, ...]
    signature: RateEvidenceSignature
    bundle_path: Path
    bundle_sha256: str
    bundle_byte_length: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != RATE_EVIDENCE_SCHEMA_VERSION
        ):
            raise RateEvidenceValidationError(
                f"unsupported rate-evidence schema version: {self.schema_version!r}"
            )
        _validate_id(self.bundle_id, name="bundle_id")
        if type(self.provenance_status) is not ProvenanceStatus:
            raise RateEvidenceValidationError("invalid bundle provenance status")
        _validate_sha256(
            self.source_registry_sha256,
            name="source_registry_sha256",
        )
        _validate_relative_posix_path(self.artifact_root, name="artifact_root")
        if not isinstance(self.notes, str) or not self.notes.strip():
            raise RateEvidenceValidationError(
                "rate-evidence bundle notes cannot be empty"
            )
        if type(self.artifacts) is not tuple or any(
            type(item) is not RateEvidenceArtifact for item in self.artifacts
        ):
            raise RateEvidenceValidationError(
                "rate-evidence artifacts must be an immutable typed tuple"
            )
        if type(self.bindings) is not tuple or any(
            type(item) is not RateEvidenceBinding for item in self.bindings
        ):
            raise RateEvidenceValidationError(
                "rate-evidence bindings must be an immutable typed tuple"
            )
        if type(self.signature) is not RateEvidenceSignature:
            raise RateEvidenceValidationError("invalid rate-evidence signature")
        if not isinstance(self.bundle_path, Path) or not self.bundle_path.is_absolute():
            raise RateEvidenceValidationError("bundle_path must be an absolute Path")
        lexical_bundle_path = Path(os.path.normpath(os.fspath(self.bundle_path)))
        if (
            ".." in self.bundle_path.parts
            or lexical_bundle_path != self.bundle_path
        ):
            raise RateEvidenceValidationError(
                "bundle_path must be lexically canonical without dot segments"
            )
        _validate_sha256(self.bundle_sha256, name="bundle_sha256")
        _validate_strict_int(
            self.bundle_byte_length,
            name="bundle_byte_length",
            minimum=1,
            maximum=MAX_RATE_BUNDLE_BYTES,
        )
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        artifact_paths = tuple(item.relative_path for item in self.artifacts)
        binding_ids = tuple(item.binding_id for item in self.bindings)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise RateEvidenceValidationError("rate-evidence artifact ids repeat")
        if len(set(binding_ids)) != len(binding_ids):
            raise RateEvidenceValidationError("rate-evidence binding ids repeat")
        if len({path.casefold() for path in artifact_paths}) != len(artifact_paths):
            raise RateEvidenceValidationError(
                "rate-evidence artifact paths repeat under case-insensitive lookup"
            )
        if artifact_ids != tuple(sorted(artifact_ids)):
            raise RateEvidenceValidationError(
                "rate-evidence artifacts must use ascending artifact_id order"
            )
        if binding_ids != tuple(sorted(binding_ids)):
            raise RateEvidenceValidationError(
                "rate-evidence bindings must use ascending binding_id order"
            )
        referenced = {binding.artifact_id for binding in self.bindings}
        declared = set(artifact_ids)
        if referenced != declared:
            raise RateEvidenceValidationError(
                "rate-evidence artifacts must be referenced exactly; "
                f"missing={sorted(referenced - declared)}, "
                f"unreferenced={sorted(declared - referenced)}"
            )
        semantic_keys = tuple(
            (
                item.source_id,
                item.jurisdiction_code,
                item.source_currency,
                item.target_currency,
                item.method,
                item.rate_period_start,
                item.rate_period_end,
            )
            for item in self.bindings
        )
        if len(set(semantic_keys)) != len(semantic_keys):
            raise RateEvidenceValidationError(
                "rate-evidence bundle repeats a semantic rate binding"
            )

    @property
    def campaign_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.provenance_status is not ProvenanceStatus.CALIBRATED:
            blockers.append(
                "rate_evidence_bundle_status=" + self.provenance_status.value
            )
        blockers.append("rate_evidence_bundle_signature_missing")
        if not self.artifacts:
            blockers.append("rate_evidence_bundle_empty")
        return tuple(blockers)

    @property
    def campaign_ready(self) -> bool:
        return False

    def validate_for_campaign(self) -> None:
        raise RateEvidenceVerificationError(
            "rate-evidence schema v1 is not campaign-ready: "
            + ", ".join(self.campaign_blockers)
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_id": self.bundle_id,
            "provenance_status": self.provenance_status.value,
            "source_registry_sha256": self.source_registry_sha256,
            "artifact_root": self.artifact_root,
            "notes": self.notes,
            "artifacts": [item.snapshot() for item in self.artifacts],
            "bindings": [item.snapshot() for item in self.bindings],
            "signature": self.signature.snapshot(),
            "bundle_path": str(self.bundle_path),
            "bundle_sha256": self.bundle_sha256,
            "bundle_byte_length": self.bundle_byte_length,
            "campaign_ready": False,
            "campaign_blockers": list(self.campaign_blockers),
        }


def exact_csv_rational_recipe_json(
    *,
    source_id: str,
    jurisdiction_code: str,
    source_currency: str,
    target_currency: str,
    method: RateEvidenceMethod,
    rate_period_start: date,
    rate_period_end: date,
    retrieved_on: date,
    numerator_column: str = "rate_numerator",
    denominator_column: str = "rate_denominator",
) -> str:
    """Build the only canonical extraction recipe accepted by schema v1."""

    _validate_id(source_id, name="source_id")
    _validate_jurisdiction_code(jurisdiction_code)
    _validate_currency(source_currency, name="source_currency")
    _validate_currency(target_currency, name="target_currency")
    if type(method) is not RateEvidenceMethod:
        raise RateEvidenceValidationError("rate-evidence method must be FX or PPP")
    for name, value in (
        ("rate_period_start", rate_period_start),
        ("rate_period_end", rate_period_end),
        ("retrieved_on", retrieved_on),
    ):
        if type(value) is not date:
            raise RateEvidenceValidationError(f"{name} must be an ISO calendar date")
    if rate_period_end < rate_period_start:
        raise RateEvidenceValidationError("rate period ends before it starts")
    if retrieved_on < rate_period_end:
        raise RateEvidenceValidationError(
            "rate retrieval date cannot predate the rate-period end"
        )
    _validate_column_name(numerator_column, name="numerator_column")
    _validate_column_name(denominator_column, name="denominator_column")
    if numerator_column == denominator_column:
        raise RateEvidenceValidationError(
            "numerator and denominator columns must be distinct"
        )
    if numerator_column in _ROW_MATCH_KEY_SET or (
        denominator_column in _ROW_MATCH_KEY_SET
    ):
        raise RateEvidenceValidationError(
            "rational-value columns cannot also be semantic selector columns"
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "interpreter": EXACT_CSV_INTERPRETER_V1,
        "row_match": {
            "jurisdiction_code": jurisdiction_code,
            "method": method.value,
            "rate_period_end": rate_period_end.isoformat(),
            "rate_period_start": rate_period_start.isoformat(),
            "retrieved_on": retrieved_on.isoformat(),
            "source_currency": source_currency,
            "source_id": source_id,
            "target_currency": target_currency,
        },
        "numerator_column": numerator_column,
        "denominator_column": denominator_column,
    }
    return _canonical_json(payload)


def load_rate_evidence_bundle(
    path: str | Path = DEFAULT_RATE_EVIDENCE_BUNDLE_PATH,
) -> RateEvidenceBundle:
    """Parse one strict schema-v1 bundle without yet reading its artifacts."""

    bundle_path = Path(path)
    observed = _secure_read_regular_file(
        bundle_path,
        maximum_bytes=MAX_RATE_BUNDLE_BYTES,
        description="rate-evidence bundle",
    )
    try:
        text = observed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RateEvidenceValidationError(
            "rate-evidence bundle must be UTF-8"
        ) from exc
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RateEvidenceValidationError(
            f"invalid rate-evidence TOML: {exc}"
        ) from exc
    _require_mapping(raw, name="rate-evidence root")
    _exact_keys(raw, _TOP_LEVEL_KEYS, name="rate-evidence root")
    schema_version = _required_int(raw, "schema_version", minimum=1)
    if schema_version != RATE_EVIDENCE_SCHEMA_VERSION:
        raise RateEvidenceValidationError(
            f"unsupported rate-evidence schema version: {schema_version}"
        )
    artifacts_raw = _required_table_list(raw, "artifacts")
    bindings_raw = _required_table_list(raw, "bindings")
    signature_raw = _required_table(raw, "signature")
    artifacts = tuple(_parse_artifact(row) for row in artifacts_raw)
    bindings = tuple(_parse_binding(row) for row in bindings_raw)
    signature = _parse_signature(signature_raw)
    return RateEvidenceBundle(
        schema_version=schema_version,
        bundle_id=_required_string(raw, "bundle_id"),
        provenance_status=_parse_provenance_status(
            _required_string(raw, "provenance_status")
        ),
        source_registry_sha256=_required_string(raw, "source_registry_sha256"),
        artifact_root=_required_string(raw, "artifact_root"),
        notes=_required_string(raw, "notes"),
        artifacts=artifacts,
        bindings=bindings,
        signature=signature,
        bundle_path=bundle_path.resolve(strict=True),
        bundle_sha256=sha256(observed).hexdigest(),
        bundle_byte_length=len(observed),
    )


def verify_rate_evidence_bundle(
    bundle: RateEvidenceBundle,
    *,
    required_source_registry_sha256: str | None = None,
) -> tuple[RateEvidenceResult, ...]:
    """Re-attest the bundle and execute each whitelisted exact recipe."""

    if type(bundle) is not RateEvidenceBundle:
        raise TypeError("bundle must be a RateEvidenceBundle")
    reloaded = load_rate_evidence_bundle(bundle.bundle_path)
    if reloaded != bundle:
        raise RateEvidenceVerificationError(
            "rate-evidence bundle metadata no longer match its declared file"
        )
    if required_source_registry_sha256 is not None:
        _validate_sha256(
            required_source_registry_sha256,
            name="required_source_registry_sha256",
        )
        if required_source_registry_sha256 != bundle.source_registry_sha256:
            raise RateEvidenceVerificationError(
                "rate-evidence bundle source_registry_sha256 does not match "
                "the required source catalogue"
            )
    if not bundle.artifacts:
        return ()

    bundle_parent = bundle.bundle_path.parent
    artifact_root = bundle_parent.joinpath(
        *PurePosixPath(bundle.artifact_root).parts
    )
    _assert_directory_without_links(
        artifact_root,
        chain_start=bundle_parent,
        description="rate-evidence artifact root",
    )
    resolved_root = artifact_root.resolve(strict=True)
    resolved_parent = bundle_parent.resolve(strict=True)
    if not resolved_root.is_relative_to(resolved_parent):
        raise RateEvidenceVerificationError(
            "rate-evidence artifact root escapes the bundle directory"
        )

    artifact_bytes: dict[str, bytes] = {}
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in bundle.artifacts}
    for artifact in bundle.artifacts:
        candidate = artifact_root.joinpath(
            *PurePosixPath(artifact.relative_path).parts
        )
        _assert_path_chain_without_links(
            candidate,
            chain_start=artifact_root,
            description=f"artifact {artifact.artifact_id}",
        )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise RateEvidenceVerificationError(
                f"artifact {artifact.artifact_id} cannot be resolved"
            ) from exc
        if not resolved.is_relative_to(resolved_root):
            raise RateEvidenceVerificationError(
                f"artifact {artifact.artifact_id} escapes its declared root"
            )
        content = _secure_read_regular_file(
            candidate,
            expected_byte_length=artifact.byte_length,
            expected_sha256=artifact.sha256,
            maximum_bytes=MAX_RATE_ARTIFACT_BYTES,
            description=f"artifact {artifact.artifact_id}",
        )
        _assert_path_chain_without_links(
            candidate,
            chain_start=artifact_root,
            description=f"artifact {artifact.artifact_id}",
        )
        artifact_bytes[artifact.artifact_id] = content

    results: list[RateEvidenceResult] = []
    for binding in bundle.bindings:
        artifact = artifacts_by_id[binding.artifact_id]
        numerator, denominator = _execute_exact_csv_recipe(
            artifact_bytes[binding.artifact_id],
            binding=binding,
        )
        if (numerator, denominator) != (
            binding.rate_numerator,
            binding.rate_denominator,
        ):
            raise RateEvidenceVerificationError(
                f"binding {binding.binding_id} extracted {numerator}/{denominator} "
                f"but declared {binding.rate_numerator}/{binding.rate_denominator}"
            )
        payload = {
            "schema_version": RATE_EVIDENCE_SCHEMA_VERSION,
            "bundle_sha256": bundle.bundle_sha256,
            "source_registry_sha256": bundle.source_registry_sha256,
            "binding_id": binding.binding_id,
            "binding_sha256": binding.binding_sha256,
            "artifact_id": artifact.artifact_id,
            "artifact_sha256": artifact.sha256,
            "artifact_byte_length": artifact.byte_length,
            "recipe_sha256": binding.recipe_sha256,
            "rate_numerator": numerator,
            "rate_denominator": denominator,
            "rate_numerator_decimal": str(numerator),
            "rate_denominator_decimal": str(denominator),
        }
        results.append(
            RateEvidenceResult(
                bundle_sha256=bundle.bundle_sha256,
                source_registry_sha256=bundle.source_registry_sha256,
                binding_id=binding.binding_id,
                binding_sha256=binding.binding_sha256,
                artifact_id=artifact.artifact_id,
                artifact_sha256=artifact.sha256,
                artifact_byte_length=artifact.byte_length,
                recipe_sha256=binding.recipe_sha256,
                rate_numerator=numerator,
                rate_denominator=denominator,
                evidence_sha256=sha256(
                    _canonical_json(payload).encode("utf-8")
                ).hexdigest(),
            )
        )
    return tuple(results)


def load_and_verify_rate_evidence_bundle(
    path: str | Path = DEFAULT_RATE_EVIDENCE_BUNDLE_PATH,
    *,
    required_source_registry_sha256: str | None = None,
) -> tuple[RateEvidenceBundle, tuple[RateEvidenceResult, ...]]:
    """Load and immediately re-attest a rate-evidence bundle."""

    bundle = load_rate_evidence_bundle(path)
    return bundle, verify_rate_evidence_bundle(
        bundle,
        required_source_registry_sha256=required_source_registry_sha256,
    )


def validate_rate_evidence_snapshot(
    bundle_snapshot: object,
    result_snapshots: object,
) -> tuple[RateEvidenceBundle | None, tuple[RateEvidenceResult, ...]]:
    """Rebuild canonical serialized evidence without trusting derived claims.

    Registered lineage separately re-reads the files and artifacts.  This
    validator also protects unregistered snapshots from publishing invented
    ``campaign_ready`` values, blocker lists, or extraction attestations.
    """

    if not isinstance(result_snapshots, list):
        raise RateEvidenceValidationError(
            "rate-evidence result snapshots must be an array"
        )
    if bundle_snapshot is None:
        if result_snapshots:
            raise RateEvidenceValidationError(
                "rate-evidence results require a source bundle snapshot"
            )
        return None, ()
    bundle_row = _require_mapping(
        bundle_snapshot,
        name="rate-evidence bundle snapshot",
    )
    _exact_keys(
        bundle_row,
        _BUNDLE_SNAPSHOT_KEYS,
        name="rate-evidence bundle snapshot",
    )
    if bundle_row.get("campaign_ready") is not False:
        raise RateEvidenceValidationError(
            "rate-evidence schema v1 campaign_ready must be false"
        )
    campaign_blockers = bundle_row.get("campaign_blockers")
    if not isinstance(campaign_blockers, list) or any(
        not isinstance(blocker, str) or not blocker
        for blocker in campaign_blockers
    ):
        raise RateEvidenceValidationError(
            "rate-evidence campaign blockers are malformed"
        )
    artifacts_raw = _required_table_list(bundle_row, "artifacts")
    bindings_raw = _required_table_list(bundle_row, "bindings")
    artifacts = tuple(_parse_artifact(row) for row in artifacts_raw)
    bindings = tuple(_binding_from_snapshot(row) for row in bindings_raw)
    signature = _parse_signature(_required_table(bundle_row, "signature"))
    bundle = RateEvidenceBundle(
        schema_version=_required_int(
            bundle_row,
            "schema_version",
            minimum=1,
        ),
        bundle_id=_required_string(bundle_row, "bundle_id"),
        provenance_status=_parse_provenance_status(
            _required_string(bundle_row, "provenance_status")
        ),
        source_registry_sha256=_required_string(
            bundle_row,
            "source_registry_sha256",
        ),
        artifact_root=_required_string(bundle_row, "artifact_root"),
        notes=_required_string(bundle_row, "notes"),
        artifacts=artifacts,
        bindings=bindings,
        signature=signature,
        bundle_path=Path(_required_string(bundle_row, "bundle_path")),
        bundle_sha256=_required_string(bundle_row, "bundle_sha256"),
        bundle_byte_length=_required_int(
            bundle_row,
            "bundle_byte_length",
            minimum=1,
            maximum=MAX_RATE_BUNDLE_BYTES,
        ),
    )
    if campaign_blockers != list(bundle.campaign_blockers):
        raise RateEvidenceValidationError(
            "rate-evidence campaign blockers are not canonical"
        )
    if bundle.snapshot() != dict(bundle_row):
        raise RateEvidenceValidationError(
            "rate-evidence bundle snapshot is not canonical"
        )

    results_raw = tuple(
        _require_mapping(row, name=f"rate-evidence result[{index}]")
        for index, row in enumerate(result_snapshots)
    )
    results = tuple(_result_from_snapshot(row) for row in results_raw)
    if tuple(result.binding_id for result in results) != tuple(
        binding.binding_id for binding in bindings
    ):
        raise RateEvidenceValidationError(
            "rate-evidence results do not exactly cover ordered bindings"
        )
    artifacts_by_id = {
        artifact.artifact_id: artifact for artifact in artifacts
    }
    bindings_by_id = {binding.binding_id: binding for binding in bindings}
    for result in results:
        binding = bindings_by_id[result.binding_id]
        artifact = artifacts_by_id[binding.artifact_id]
        if (
            result.bundle_sha256 != bundle.bundle_sha256
            or result.source_registry_sha256 != bundle.source_registry_sha256
            or result.binding_sha256 != binding.binding_sha256
            or result.artifact_id != artifact.artifact_id
            or result.artifact_sha256 != artifact.sha256
            or result.artifact_byte_length != artifact.byte_length
            or result.recipe_sha256 != binding.recipe_sha256
            or result.rate_numerator != binding.rate_numerator
            or result.rate_denominator != binding.rate_denominator
        ):
            raise RateEvidenceValidationError(
                "rate-evidence result does not match its bundle declarations"
            )
    observed_results = verify_rate_evidence_bundle(
        bundle,
        required_source_registry_sha256=bundle.source_registry_sha256,
    )
    if observed_results != results:
        raise RateEvidenceValidationError(
            "rate-evidence result snapshots do not match re-extracted bytes"
        )
    return bundle, results


def _binding_from_snapshot(
    row: Mapping[str, object],
) -> RateEvidenceBinding:
    _exact_keys(row, _BINDING_SNAPSHOT_KEYS, name="rate-evidence binding snapshot")
    numerator = _required_int(row, "rate_numerator", minimum=1)
    denominator = _required_int(row, "rate_denominator", minimum=1)
    if (
        row.get("rate_numerator_decimal") != str(numerator)
        or row.get("rate_denominator_decimal") != str(denominator)
    ):
        raise RateEvidenceValidationError(
            "rate-evidence binding decimal mirrors are not lossless"
        )
    parsed_row = {
        key: row[key]
        for key in _BINDING_KEYS
    }
    for field in ("rate_period_start", "rate_period_end", "retrieved_on"):
        parsed_row[field] = _snapshot_date(row.get(field), name=field)
    binding = _parse_binding(parsed_row)
    if binding.snapshot() != dict(row):
        raise RateEvidenceValidationError(
            "rate-evidence binding snapshot is not canonical"
        )
    return binding


def _result_from_snapshot(
    row: Mapping[str, object],
) -> RateEvidenceResult:
    _exact_keys(row, _RESULT_SNAPSHOT_KEYS, name="rate-evidence result snapshot")
    schema_version = _required_int(row, "schema_version", minimum=1)
    if schema_version != RATE_EVIDENCE_SCHEMA_VERSION:
        raise RateEvidenceValidationError(
            "rate-evidence result has an unsupported schema version"
        )
    numerator = _required_int(row, "rate_numerator", minimum=1)
    denominator = _required_int(row, "rate_denominator", minimum=1)
    if (
        row.get("rate_numerator_decimal") != str(numerator)
        or row.get("rate_denominator_decimal") != str(denominator)
    ):
        raise RateEvidenceValidationError(
            "rate-evidence result decimal mirrors are not lossless"
        )
    result = RateEvidenceResult(
        bundle_sha256=_required_string(row, "bundle_sha256"),
        source_registry_sha256=_required_string(
            row,
            "source_registry_sha256",
        ),
        binding_id=_required_string(row, "binding_id"),
        binding_sha256=_required_string(row, "binding_sha256"),
        artifact_id=_required_string(row, "artifact_id"),
        artifact_sha256=_required_string(row, "artifact_sha256"),
        artifact_byte_length=_required_int(
            row,
            "artifact_byte_length",
            minimum=1,
            maximum=MAX_RATE_ARTIFACT_BYTES,
        ),
        recipe_sha256=_required_string(row, "recipe_sha256"),
        rate_numerator=numerator,
        rate_denominator=denominator,
        evidence_sha256=_required_string(row, "evidence_sha256"),
    )
    if result.snapshot() != dict(row):
        raise RateEvidenceValidationError(
            "rate-evidence result snapshot is not canonical"
        )
    return result


def _snapshot_date(value: object, *, name: str) -> date:
    if not isinstance(value, str):
        raise RateEvidenceValidationError(
            f"rate-evidence snapshot {name} must be an ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise RateEvidenceValidationError(
            f"rate-evidence snapshot {name} must be an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise RateEvidenceValidationError(
            f"rate-evidence snapshot {name} must be canonical"
        )
    return parsed


def _parse_artifact(row: Mapping[str, object]) -> RateEvidenceArtifact:
    _exact_keys(row, _ARTIFACT_KEYS, name="rate-evidence artifact")
    return RateEvidenceArtifact(
        artifact_id=_required_string(row, "artifact_id"),
        relative_path=_required_string(row, "relative_path"),
        media_type=_required_string(row, "media_type"),
        sha256=_required_string(row, "sha256"),
        byte_length=_required_int(
            row,
            "byte_length",
            minimum=1,
            maximum=MAX_RATE_ARTIFACT_BYTES,
        ),
    )


def _parse_binding(row: Mapping[str, object]) -> RateEvidenceBinding:
    _exact_keys(row, _BINDING_KEYS, name="rate-evidence binding")
    try:
        method = RateEvidenceMethod(_required_string(row, "method"))
    except ValueError as exc:
        raise RateEvidenceValidationError(
            "rate-evidence binding method must be FX or PPP"
        ) from exc
    return RateEvidenceBinding(
        binding_id=_required_string(row, "binding_id"),
        artifact_id=_required_string(row, "artifact_id"),
        source_id=_required_string(row, "source_id"),
        jurisdiction_code=_required_string(row, "jurisdiction_code"),
        source_currency=_required_string(row, "source_currency"),
        target_currency=_required_string(row, "target_currency"),
        method=method,
        rate_period_start=_required_date(row, "rate_period_start"),
        rate_period_end=_required_date(row, "rate_period_end"),
        retrieved_on=_required_date(row, "retrieved_on"),
        rate_numerator=_required_int(row, "rate_numerator", minimum=1),
        rate_denominator=_required_int(row, "rate_denominator", minimum=1),
        recipe_json=_required_string(row, "recipe_json"),
    )


def _parse_signature(row: Mapping[str, object]) -> RateEvidenceSignature:
    _exact_keys(row, _SIGNATURE_KEYS, name="rate-evidence signature")
    try:
        status = RateEvidenceSignatureStatus(_required_string(row, "status"))
    except ValueError as exc:
        raise RateEvidenceValidationError(
            "rate-evidence schema v1 signature status must be MISSING"
        ) from exc
    return RateEvidenceSignature(
        status=status,
        algorithm=_required_string(row, "algorithm", allow_empty=True),
        key_id=_required_string(row, "key_id", allow_empty=True),
        value=_required_string(row, "value", allow_empty=True),
    )


def _parse_recipe_json(recipe_json: object) -> dict[str, object]:
    if not isinstance(recipe_json, str) or not recipe_json:
        raise RateEvidenceValidationError("rate-evidence recipe_json must be text")
    try:
        parsed = json.loads(recipe_json)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RateEvidenceValidationError(
            "rate-evidence recipe_json must be valid finite JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise RateEvidenceValidationError("rate-evidence recipe root must be an object")
    try:
        canonical = _canonical_json(parsed)
    except (TypeError, ValueError) as exc:
        raise RateEvidenceValidationError(
            "rate-evidence recipe must contain finite JSON values"
        ) from exc
    if canonical != recipe_json:
        raise RateEvidenceValidationError(
            "rate-evidence recipe_json must use canonical JSON"
        )
    interpreter = parsed.get("interpreter")
    if interpreter == EXACT_CSV_INTERPRETER_V1:
        expected_keys = _RECIPE_KEYS
    elif interpreter == ECB_EXR_SOURCE_PER_EUR_INTERPRETER_V1:
        expected_keys = _ECB_RECIPE_KEYS
    else:
        raise RateEvidenceValidationError(
            "rate-evidence recipe uses a non-whitelisted interpreter"
        )
    _exact_keys(parsed, expected_keys, name="rate-evidence recipe")
    if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
        raise RateEvidenceValidationError(
            "rate-evidence recipe schema_version must be strict integer 1"
        )
    row_match = parsed["row_match"]
    if not isinstance(row_match, dict):
        raise RateEvidenceValidationError("recipe row_match must be an object")
    _exact_keys(row_match, _ROW_MATCH_KEY_SET, name="recipe row_match")
    if any(not isinstance(value, str) or not value for value in row_match.values()):
        raise RateEvidenceValidationError(
            "recipe row_match values must be non-empty strings"
        )
    if interpreter == EXACT_CSV_INTERPRETER_V1:
        for field in ("numerator_column", "denominator_column"):
            value = parsed[field]
            if not isinstance(value, str):
                raise RateEvidenceValidationError(f"recipe {field} must be text")
            _validate_column_name(value, name=field)
        if parsed["numerator_column"] == parsed["denominator_column"]:
            raise RateEvidenceValidationError(
                "recipe rational-value columns must be distinct"
            )
        if parsed["numerator_column"] in _ROW_MATCH_KEY_SET or (
            parsed["denominator_column"] in _ROW_MATCH_KEY_SET
        ):
            raise RateEvidenceValidationError(
                "recipe rational-value columns cannot be selector columns"
            )
    else:
        source_row_match = parsed["source_row_match"]
        if not isinstance(source_row_match, dict):
            raise RateEvidenceValidationError(
                "ECB recipe source_row_match must be an object"
            )
        _exact_keys(
            source_row_match,
            _ECB_SOURCE_ROW_MATCH_KEYS,
            name="ECB recipe source_row_match",
        )
        if any(
            not isinstance(value, str) or not value
            for value in source_row_match.values()
        ):
            raise RateEvidenceValidationError(
                "ECB recipe source_row_match values must be non-empty strings"
            )
        observation_column = parsed["observation_column"]
        if not isinstance(observation_column, str):
            raise RateEvidenceValidationError(
                "ECB recipe observation_column must be text"
            )
        _validate_column_name(
            observation_column,
            name="observation_column",
        )
        if observation_column in _ECB_SOURCE_ROW_MATCH_KEYS:
            raise RateEvidenceValidationError(
                "ECB recipe observation column cannot be a selector column"
            )
        for field in (
            "source_minor_unit_exponent",
            "target_minor_unit_exponent",
        ):
            _validate_strict_int(
                parsed[field],
                name=f"ECB recipe {field}",
                minimum=0,
                maximum=9,
            )
    return parsed


def _execute_exact_csv_recipe(
    content: bytes,
    *,
    binding: RateEvidenceBinding,
) -> tuple[int, int]:
    recipe = _parse_recipe_json(binding.recipe_json)
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RateEvidenceVerificationError(
            f"artifact {binding.artifact_id} must be strict UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise RateEvidenceVerificationError(
            f"artifact {binding.artifact_id} must not contain a UTF-8 BOM"
        )
    if "\x00" in text:
        raise RateEvidenceVerificationError(
            f"artifact {binding.artifact_id} contains a NUL character"
        )
    try:
        rows = list(
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
        raise RateEvidenceVerificationError(
            f"artifact {binding.artifact_id} is not strict CSV"
        ) from exc
    if len(rows) < 2:
        raise RateEvidenceVerificationError(
            f"artifact {binding.artifact_id} needs a header and data row"
        )
    header = rows[0]
    if not header or any(
        not name or name.strip() != name or "\r" in name or "\n" in name
        for name in header
    ):
        raise RateEvidenceVerificationError(
            f"artifact {binding.artifact_id} has an invalid CSV header"
        )
    if len(set(header)) != len(header):
        raise RateEvidenceVerificationError(
            f"artifact {binding.artifact_id} repeats a CSV column"
        )
    if recipe["interpreter"] == EXACT_CSV_INTERPRETER_V1:
        selector = binding.row_match
        value_columns = {
            str(recipe["numerator_column"]),
            str(recipe["denominator_column"]),
        }
    else:
        selector = recipe["source_row_match"]
        assert isinstance(selector, dict)
        value_columns = {str(recipe["observation_column"])}
    required_columns = {*selector, *value_columns}
    missing = sorted(required_columns.difference(header))
    if missing:
        raise RateEvidenceVerificationError(
            f"artifact {binding.artifact_id} lacks recipe columns: {missing}"
        )
    width = len(header)
    matched: list[dict[str, str]] = []
    for index, row in enumerate(rows[1:], start=2):
        if not row or len(row) != width:
            raise RateEvidenceVerificationError(
                f"artifact {binding.artifact_id} row {index} has wrong width"
            )
        if any("\r" in value or "\n" in value for value in row):
            raise RateEvidenceVerificationError(
                f"artifact {binding.artifact_id} row {index} has multiline data"
            )
        mapped = dict(zip(header, row, strict=True))
        if all(mapped[key] == value for key, value in selector.items()):
            matched.append(mapped)
    if len(matched) != 1:
        raise RateEvidenceVerificationError(
            f"binding {binding.binding_id} must match exactly one CSV row; "
            f"matched={len(matched)}"
        )
    row = matched[0]
    if recipe["interpreter"] == EXACT_CSV_INTERPRETER_V1:
        numerator = _parse_positive_canonical_integer(
            row[str(recipe["numerator_column"])],
            name=f"binding {binding.binding_id} numerator",
        )
        denominator = _parse_positive_canonical_integer(
            row[str(recipe["denominator_column"])],
            name=f"binding {binding.binding_id} denominator",
        )
        if math.gcd(numerator, denominator) != 1:
            raise RateEvidenceVerificationError(
                f"binding {binding.binding_id} CSV rational is not in lowest terms"
            )
        return numerator, denominator
    observation = _parse_positive_canonical_decimal(
        row[str(recipe["observation_column"])],
        name=f"binding {binding.binding_id} ECB observation",
    )
    source_exponent = int(recipe["source_minor_unit_exponent"])
    target_exponent = int(recipe["target_minor_unit_exponent"])
    rate = Fraction(10**target_exponent, 10**source_exponent) / observation
    return rate.numerator, rate.denominator


def _secure_read_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
    expected_byte_length: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    candidate = Path(path)
    before = _lstat_regular_file(candidate, description=description)
    if before.st_size > maximum_bytes:
        raise RateEvidenceVerificationError(
            f"{description} exceeds the {maximum_bytes}-byte safety limit"
        )
    if expected_byte_length is not None and before.st_size != expected_byte_length:
        raise RateEvidenceVerificationError(
            f"{description} byte length changed: expected {expected_byte_length}, "
            f"observed {before.st_size}"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RateEvidenceVerificationError(f"cannot open {description}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _is_reparse(opened):
            raise RateEvidenceVerificationError(
                f"{description} must be a non-reparse regular file"
            )
        if not _same_file_identity(before, opened):
            raise RateEvidenceVerificationError(
                f"{description} changed while it was opened"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise RateEvidenceVerificationError(
                    f"{description} exceeds the {maximum_bytes}-byte safety limit"
                )
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = _lstat_regular_file(candidate, description=description)
    if not _same_file_identity(opened, after_open) or not _same_file_identity(
        after_open,
        after_path,
    ):
        raise RateEvidenceVerificationError(f"{description} changed while read")
    content = b"".join(chunks)
    if len(content) != after_open.st_size:
        raise RateEvidenceVerificationError(f"{description} was not read completely")
    if expected_byte_length is not None and len(content) != expected_byte_length:
        raise RateEvidenceVerificationError(
            f"{description} byte length does not match its declaration"
        )
    observed_sha256 = sha256(content).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RateEvidenceVerificationError(
            f"{description} SHA-256 does not match its declaration"
        )
    return content


def _lstat_regular_file(path: Path, *, description: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise RateEvidenceVerificationError(f"{description} does not exist") from exc
    if path.is_symlink() or _is_reparse(observed):
        raise RateEvidenceVerificationError(
            f"{description} cannot be a symlink or reparse point"
        )
    if not stat.S_ISREG(observed.st_mode):
        raise RateEvidenceVerificationError(f"{description} must be a regular file")
    return observed


def _assert_directory_without_links(
    path: Path,
    *,
    chain_start: Path,
    description: str,
) -> None:
    _assert_path_chain_without_links(
        path,
        chain_start=chain_start,
        description=description,
    )
    try:
        observed = path.lstat()
    except OSError as exc:
        raise RateEvidenceVerificationError(f"{description} does not exist") from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise RateEvidenceVerificationError(f"{description} must be a directory")


def _assert_path_chain_without_links(
    path: Path,
    *,
    chain_start: Path,
    description: str,
) -> None:
    try:
        relative = path.relative_to(chain_start)
    except ValueError as exc:
        raise RateEvidenceVerificationError(
            f"{description} is not lexically contained in its declared root"
        ) from exc
    current = chain_start
    candidates = (
        current,
        *(
            current.joinpath(*relative.parts[:index])
            for index in range(1, len(relative.parts) + 1)
        ),
    )
    for component in candidates:
        try:
            observed = component.lstat()
        except OSError as exc:
            raise RateEvidenceVerificationError(
                f"{description} path component does not exist: {component.name}"
            ) from exc
        if component.is_symlink() or _is_reparse(observed):
            raise RateEvidenceVerificationError(
                f"{description} path contains a symlink or reparse point"
            )


def _is_reparse(observed: os.stat_result) -> bool:
    attributes = getattr(observed, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        getattr(left, "st_mtime_ns", None),
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        getattr(right, "st_mtime_ns", None),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _validate_relative_posix_path(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise RateEvidenceValidationError(f"{name} must be non-empty text")
    if "\\" in value:
        raise RateEvidenceValidationError(f"{name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise RateEvidenceValidationError(f"{name} must be a canonical relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RateEvidenceValidationError(
            f"{name} cannot contain empty, dot, or parent components"
        )
    windows_reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if not path.parts or any(
        ":" in part
        or part.endswith((" ", "."))
        or part.split(".", maxsplit=1)[0].upper() in windows_reserved
        for part in path.parts
    ):
        raise RateEvidenceValidationError(
            f"{name} cannot be drive-qualified, an alternate stream, or a "
            "reserved platform path"
        )


def _validate_id(value: object, *, name: str) -> None:
    if not isinstance(value, str) or _ID_PATTERN.fullmatch(value) is None:
        raise RateEvidenceValidationError(
            f"{name} must be a canonical ASCII identifier"
        )


def _validate_column_name(value: object, *, name: str) -> None:
    if not isinstance(value, str) or _COLUMN_PATTERN.fullmatch(value) is None:
        raise RateEvidenceValidationError(
            f"{name} must be a canonical ASCII CSV column name"
        )


def _validate_jurisdiction_code(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 2
        or any(character < "A" or character > "Z" for character in value)
    ):
        raise RateEvidenceValidationError(
            "jurisdiction_code must be a two-letter uppercase ASCII code"
        )


def _validate_currency(value: object, *, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 3
        or any(character < "A" or character > "Z" for character in value)
    ):
        raise RateEvidenceValidationError(
            f"{name} must be a three-letter uppercase ASCII currency code"
        )


def _validate_sha256(value: object, *, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise RateEvidenceValidationError(f"{name} must be lowercase SHA-256 hex")


def _validate_strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise RateEvidenceValidationError(f"{name} must be a strict integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise RateEvidenceValidationError(f"{name} must be {bound}")


def _parse_positive_canonical_integer(value: str, *, name: str) -> int:
    if _POSITIVE_INTEGER_PATTERN.fullmatch(value) is None:
        raise RateEvidenceVerificationError(
            f"{name} must be a canonical positive decimal integer"
        )
    return int(value)


def _parse_positive_canonical_decimal(value: str, *, name: str) -> Fraction:
    if _POSITIVE_DECIMAL_PATTERN.fullmatch(value) is None:
        raise RateEvidenceVerificationError(
            f"{name} must be a canonical non-negative decimal"
        )
    parsed = Fraction(value)
    if parsed <= 0:
        raise RateEvidenceVerificationError(f"{name} must be positive")
    return parsed


def _exact_keys(
    values: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if any(type(key) is not str for key in values):
        raise RateEvidenceValidationError(f"{name} keys must be strings")
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise RateEvidenceValidationError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RateEvidenceValidationError(f"{name} must be a table")
    return value


def _required_table(
    values: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    return _require_mapping(values.get(field), name=field)


def _required_table_list(
    values: Mapping[str, object],
    field: str,
) -> tuple[Mapping[str, object], ...]:
    raw = values.get(field)
    if type(raw) is not list:
        raise RateEvidenceValidationError(f"{field} must be an array of tables")
    output: list[Mapping[str, object]] = []
    for index, value in enumerate(raw):
        output.append(_require_mapping(value, name=f"{field}[{index}]"))
    return tuple(output)


def _required_string(
    values: Mapping[str, object],
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = values.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "text" if allow_empty else "non-empty text"
        raise RateEvidenceValidationError(f"{field} must be {qualifier}")
    if not allow_empty and value != value.strip():
        raise RateEvidenceValidationError(
            f"{field} cannot contain surrounding whitespace"
        )
    return value


def _required_int(
    values: Mapping[str, object],
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = values.get(field)
    _validate_strict_int(value, name=field, minimum=minimum, maximum=maximum)
    assert type(value) is int
    return value


def _required_date(values: Mapping[str, object], field: str) -> date:
    value = values.get(field)
    if type(value) is not date:
        raise RateEvidenceValidationError(
            f"{field} must be an unquoted TOML calendar date"
        )
    return value


def _parse_provenance_status(value: str) -> ProvenanceStatus:
    try:
        return ProvenanceStatus(value)
    except ValueError as exc:
        raise RateEvidenceValidationError(
            f"invalid rate-evidence provenance status: {value}"
        ) from exc


__all__ = [
    "DEFAULT_RATE_EVIDENCE_BUNDLE_PATH",
    "EXACT_CSV_INTERPRETER_V1",
    "MAX_RATE_ARTIFACT_BYTES",
    "RATE_EVIDENCE_SCHEMA_VERSION",
    "RateEvidenceArtifact",
    "RateEvidenceBinding",
    "RateEvidenceBundle",
    "RateEvidenceMethod",
    "RateEvidenceResult",
    "RateEvidenceSignature",
    "RateEvidenceSignatureStatus",
    "RateEvidenceValidationError",
    "RateEvidenceVerificationError",
    "exact_csv_rational_recipe_json",
    "load_and_verify_rate_evidence_bundle",
    "load_rate_evidence_bundle",
    "validate_rate_evidence_snapshot",
    "verify_rate_evidence_bundle",
]
