"""Deterministic, fail-closed execution receipts for campaign preflight.

The receipt in this module attests *identity*, not scientific validity.  A
verified digest means that the repository, interpreter, dependencies,
configuration, plan, and declared inputs still match the pre-run declaration.
It does not authenticate sources, calibrate the model, validate a population,
or establish campaign readiness.

Schema version 1 therefore fixes ``campaign_ready`` to ``False``.  Callers can
still build and verify a pre-campaign receipt when an input is explicitly
unavailable; the resulting campaign-execution gate remains closed and names the
missing identity instead of assigning it a fabricated digest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Final


EXECUTION_RECEIPT_SCHEMA_VERSION: Final[str] = "1.0"
EXECUTION_RECEIPT_SCHEMA_ID: Final[str] = (
    "https://microtx-sim.invalid/schemas/execution-receipt-1.0.json"
)
CANONICAL_IDENTITY_ALGORITHM: Final[str] = (
    "microtx_sim.execution_receipt.canonical_json_utf8.v1"
)
SOURCE_TREE_ALGORITHM: Final[str] = (
    "microtx_sim.git_tracked_worktree.raw_bytes_and_modes.v1"
)
DEPENDENCY_SET_ALGORITHM: Final[str] = (
    "microtx_sim.installed_python_distributions.v1"
)
MAX_RECEIPT_BYTES: Final[int] = 8 * 1024 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}\Z")
_PACKAGE_NORMALIZER = re.compile(r"[-_.]+")

REQUIRED_FILE_ARTIFACT_IDS: Final[frozenset[str]] = frozenset(
    {
        "campaign_configuration",
        "prospective_analysis_plan",
        "population_source_registry",
        "population_evidence",
        "population_source_artifact",
        "population_design",
        "population_runtime_mapping",
        "monetary_source_bundle",
        "monetary_source_artifact",
        "monetary_conversion_table",
    }
)

REQUIRED_SEMANTIC_IDENTITY_IDS: Final[frozenset[str]] = frozenset(
    {
        "prospective_plan_semantic",
        "population_adapter",
        "population_execution",
        "population_assignment",
        "population_balance",
        "population_lineage",
        "monetary_conversion_basis",
        "monetary_rate_evidence",
        "metric_contract_registry",
        "output_schema",
        "manifest_schema",
    }
)

_IDENTITY_PAYLOAD_KEYS = frozenset(
    {
        "receipt_contract",
        "encoding_contract",
        "repository",
        "source_tree",
        "python_runtime",
        "operating_system",
        "dependency_environment",
        "project_files",
        "campaign",
        "input_artifacts",
        "input_identities",
        "expected_output_artifacts",
        "ledger",
        "run",
        "model_version",
    }
)


class ExecutionAttestationError(ValueError):
    """Base error for malformed, incomplete, or unverifiable receipts."""


class ExecutionReceiptMismatchError(ExecutionAttestationError):
    """Raised when a pre-run receipt differs from a fresh identity."""


class CampaignExecutionRejectedError(RuntimeError):
    """Raised when a verified receipt still has a closed campaign gate."""


class ExecutionVerificationPhase(str, Enum):
    PRE_EXECUTION = "PRE_EXECUTION"
    POST_EXECUTION = "POST_EXECUTION"


@dataclass(frozen=True, slots=True)
class FileArtifactSpec:
    """One repository-relative input whose exact raw bytes must match."""

    artifact_id: str
    path: Path
    expected_sha256: str
    schema_version: str
    expected_byte_length: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, name="artifact_id")
        if not isinstance(self.path, Path):
            raise TypeError("file artifact path must be a Path")
        _sha256_digest(self.expected_sha256, name="expected_sha256")
        _nonempty_text(self.schema_version, name="artifact schema_version")
        if self.expected_byte_length is not None:
            _strict_int(
                self.expected_byte_length,
                name="expected_byte_length",
                minimum=0,
            )


@dataclass(frozen=True, slots=True)
class DeclaredIdentity:
    """A canonical semantic identity or an explicit unavailable declaration."""

    identity_id: str
    schema_version: str
    sha256: str | None = None
    unavailable_reason: str | None = None
    required_for_execution: bool = True

    def __post_init__(self) -> None:
        _identifier(self.identity_id, name="identity_id")
        _nonempty_text(self.schema_version, name="identity schema_version")
        if type(self.required_for_execution) is not bool:
            raise TypeError("required_for_execution must be boolean")
        available = self.sha256 is not None
        unavailable = self.unavailable_reason is not None
        if available == unavailable:
            raise ExecutionAttestationError(
                "declared identity requires exactly one of sha256 or "
                "unavailable_reason"
            )
        if available:
            _sha256_digest(self.sha256, name=f"{self.identity_id} sha256")
        else:
            _nonempty_text(
                self.unavailable_reason,
                name=f"{self.identity_id} unavailable_reason",
            )

    @property
    def available(self) -> bool:
        return self.sha256 is not None

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": "AVAILABLE" if self.available else "UNAVAILABLE",
            "sha256": self.sha256,
            "hash_scope": (
                "declared canonical semantic payload"
                if self.available
                else None
            ),
            "unavailable_reason": self.unavailable_reason,
            "required_for_execution": self.required_for_execution,
        }


@dataclass(frozen=True, slots=True)
class ExecutionReceiptSpec:
    """Complete deterministic request used for both pre- and post-run checks."""

    repository_root: Path
    input_artifacts: tuple[FileArtifactSpec, ...]
    input_identities: tuple[DeclaredIdentity, ...]
    plan_id: str
    plan_sha256: str
    expected_output_artifacts: tuple[str, ...]
    ledger_backend: str
    ledger_configuration: Mapping[str, object]
    run_command: tuple[str, ...]
    execution_mode: str
    model_version: str
    scientific_readiness_blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path):
            raise TypeError("repository_root must be a Path")
        if type(self.input_artifacts) is not tuple or any(
            type(item) is not FileArtifactSpec for item in self.input_artifacts
        ):
            raise TypeError("input_artifacts must be an exact tuple of FileArtifactSpec")
        if type(self.input_identities) is not tuple or any(
            type(item) is not DeclaredIdentity for item in self.input_identities
        ):
            raise TypeError(
                "input_identities must be an exact tuple of DeclaredIdentity"
            )
        artifact_ids = tuple(item.artifact_id for item in self.input_artifacts)
        identity_ids = tuple(item.identity_id for item in self.input_identities)
        _unique_ids(artifact_ids, name="file artifact IDs")
        _unique_ids(identity_ids, name="semantic identity IDs")
        missing_artifacts = sorted(REQUIRED_FILE_ARTIFACT_IDS.difference(artifact_ids))
        missing_identities = sorted(
            REQUIRED_SEMANTIC_IDENTITY_IDS.difference(identity_ids)
        )
        if missing_artifacts or missing_identities:
            raise ExecutionAttestationError(
                "execution receipt declarations are incomplete: "
                f"missing_artifacts={missing_artifacts}, "
                f"missing_identities={missing_identities}"
            )
        _identifier(self.plan_id, name="plan_id")
        _sha256_digest(self.plan_sha256, name="plan_sha256")
        semantic_plan = next(
            item
            for item in self.input_identities
            if item.identity_id == "prospective_plan_semantic"
        )
        if semantic_plan.sha256 != self.plan_sha256:
            raise ExecutionAttestationError(
                "prospective_plan_semantic must equal plan_sha256"
            )
        _validate_expected_outputs(self.expected_output_artifacts)
        _nonempty_text(self.ledger_backend, name="ledger_backend")
        _canonical_json_bytes(self.ledger_configuration)
        if type(self.run_command) is not tuple or not self.run_command:
            raise TypeError("run_command must be a non-empty exact tuple")
        for index, value in enumerate(self.run_command):
            _nonempty_text(value, name=f"run_command[{index}]")
        _nonempty_text(self.execution_mode, name="execution_mode")
        _nonempty_text(self.model_version, name="model_version")
        blockers = _canonical_blockers(
            self.scientific_readiness_blockers,
            name="scientific_readiness_blockers",
        )
        if any(value.startswith("execution_identity.") for value in blockers):
            raise ExecutionAttestationError(
                "scientific readiness blockers cannot use the reserved "
                "execution_identity namespace"
            )


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Immutable canonical identity plus separate fail-closed status fields."""

    identity_payload_json: str
    execution_receipt_sha256: str
    input_completeness_blockers: tuple[str, ...]
    scientific_readiness_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.identity_payload_json) is not str:
            raise TypeError("identity_payload_json must be text")
        try:
            payload = _parse_json_object(self.identity_payload_json.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ExecutionAttestationError(
                "identity payload must be valid UTF-8 text"
            ) from exc
        canonical = _canonical_json_bytes(payload).decode("utf-8")
        if canonical != self.identity_payload_json:
            raise ExecutionAttestationError(
                "identity_payload_json must use canonical serialization"
            )
        _validate_identity_payload_shape(payload)
        _sha256_digest(
            self.execution_receipt_sha256,
            name="execution_receipt_sha256",
        )
        observed = sha256(canonical.encode("utf-8")).hexdigest()
        if observed != self.execution_receipt_sha256:
            raise ExecutionAttestationError(
                "execution_receipt_sha256 does not match identity_payload"
            )
        object.__setattr__(
            self,
            "input_completeness_blockers",
            _canonical_blockers(
                self.input_completeness_blockers,
                name="input_completeness_blockers",
            ),
        )
        readiness = _canonical_blockers(
            self.scientific_readiness_blockers,
            name="scientific_readiness_blockers",
        )
        if "execution_receipt.scientific_readiness=not_established" not in readiness:
            raise ExecutionAttestationError(
                "schema-v1 receipt must retain its fixed readiness blocker"
            )
        object.__setattr__(self, "scientific_readiness_blockers", readiness)
        if set(self.input_completeness_blockers).intersection(readiness):
            raise ExecutionAttestationError(
                "input completeness and scientific readiness blockers must be disjoint"
            )

    @property
    def identity_payload(self) -> dict[str, object]:
        return _parse_json_object(self.identity_payload_json.encode("utf-8"))

    @property
    def campaign_execution_blockers(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self.input_completeness_blockers).union(
                    self.scientific_readiness_blockers
                )
            )
        )

    @property
    def campaign_execution_admissible(self) -> bool:
        return False

    def snapshot(self) -> dict[str, object]:
        return {
            "$schema": EXECUTION_RECEIPT_SCHEMA_ID,
            "receipt_schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
            "identity_payload": self.identity_payload,
            "execution_receipt_sha256": self.execution_receipt_sha256,
            "identity_verification": {
                "status": "VERIFIED",
                "all_referenced_file_hashes_match": True,
                "identity_verification_implies_scientific_readiness": False,
            },
            "execution_gate": {
                "campaign_execution_admissible": False,
                "blockers": list(self.campaign_execution_blockers),
            },
            "scientific_readiness": {
                "campaign_ready": False,
                "blockers": list(self.scientific_readiness_blockers),
            },
        }

    @classmethod
    def from_snapshot(cls, value: Mapping[str, object]) -> "ExecutionReceipt":
        expected = {
            "$schema",
            "receipt_schema_version",
            "identity_payload",
            "execution_receipt_sha256",
            "identity_verification",
            "execution_gate",
            "scientific_readiness",
        }
        _exact_keys(value, expected, name="execution receipt")
        if value.get("$schema") != EXECUTION_RECEIPT_SCHEMA_ID:
            raise ExecutionAttestationError("execution receipt $schema is unsupported")
        if value.get("receipt_schema_version") != EXECUTION_RECEIPT_SCHEMA_VERSION:
            raise ExecutionAttestationError(
                "execution receipt schema_version is unsupported"
            )
        payload = _require_mapping(value.get("identity_payload"), name="identity_payload")
        identity_verification = _require_mapping(
            value.get("identity_verification"),
            name="identity_verification",
        )
        _exact_keys(
            identity_verification,
            {
                "status",
                "all_referenced_file_hashes_match",
                "identity_verification_implies_scientific_readiness",
            },
            name="identity_verification",
        )
        if identity_verification != {
            "status": "VERIFIED",
            "all_referenced_file_hashes_match": True,
            "identity_verification_implies_scientific_readiness": False,
        }:
            raise ExecutionAttestationError(
                "identity_verification must retain its fail-closed fixed values"
            )
        execution_gate = _require_mapping(
            value.get("execution_gate"), name="execution_gate"
        )
        _exact_keys(
            execution_gate,
            {"campaign_execution_admissible", "blockers"},
            name="execution_gate",
        )
        if execution_gate.get("campaign_execution_admissible") is not False:
            raise ExecutionAttestationError(
                "schema-v1 campaign_execution_admissible must be false"
            )
        readiness = _require_mapping(
            value.get("scientific_readiness"), name="scientific_readiness"
        )
        _exact_keys(
            readiness,
            {"campaign_ready", "blockers"},
            name="scientific_readiness",
        )
        if readiness.get("campaign_ready") is not False:
            raise ExecutionAttestationError(
                "schema-v1 scientific_readiness.campaign_ready must be false"
            )
        readiness_blockers = _blockers_from_json(
            readiness.get("blockers"), name="scientific_readiness.blockers"
        )
        gate_blockers = _blockers_from_json(
            execution_gate.get("blockers"), name="execution_gate.blockers"
        )
        readiness_set = set(readiness_blockers)
        completeness = tuple(
            blocker for blocker in gate_blockers if blocker not in readiness_set
        )
        receipt = cls(
            identity_payload_json=_canonical_json_bytes(payload).decode("utf-8"),
            execution_receipt_sha256=_required_string(
                value,
                "execution_receipt_sha256",
            ),
            input_completeness_blockers=completeness,
            scientific_readiness_blockers=readiness_blockers,
        )
        if receipt.snapshot() != dict(value):
            raise ExecutionAttestationError(
                "execution receipt status fields are not canonical"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class ExecutionReceiptVerification:
    """Deterministic result of comparing a receipt with current execution state."""

    phase: ExecutionVerificationPhase
    expected_receipt_sha256: str
    observed_receipt_sha256: str
    campaign_execution_admissible: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.phase) is not ExecutionVerificationPhase:
            raise TypeError("phase must be ExecutionVerificationPhase")
        _sha256_digest(
            self.expected_receipt_sha256,
            name="expected_receipt_sha256",
        )
        _sha256_digest(
            self.observed_receipt_sha256,
            name="observed_receipt_sha256",
        )
        if self.expected_receipt_sha256 != self.observed_receipt_sha256:
            raise ExecutionReceiptMismatchError(
                "verification digests differ"
            )
        if type(self.campaign_execution_admissible) is not bool:
            raise TypeError("campaign_execution_admissible must be boolean")
        if self.campaign_execution_admissible:
            raise ExecutionAttestationError(
                "schema-v1 verification cannot admit campaign execution"
            )
        object.__setattr__(
            self,
            "blockers",
            _canonical_blockers(self.blockers, name="verification blockers"),
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
            "phase": self.phase.value,
            "status": "VERIFIED",
            "expected_receipt_sha256": self.expected_receipt_sha256,
            "observed_receipt_sha256": self.observed_receipt_sha256,
            "identity_payload_match": True,
            "campaign_execution_admissible": False,
            "blockers": list(self.blockers),
        }


def build_execution_receipt(spec: ExecutionReceiptSpec) -> ExecutionReceipt:
    """Build one receipt after two stable, clean observations of all identities."""

    if type(spec) is not ExecutionReceiptSpec:
        raise TypeError("spec must be ExecutionReceiptSpec")
    ExecutionReceiptSpec.__post_init__(spec)
    first = _identity_payload_once(spec)
    second = _identity_payload_once(spec)
    if _canonical_json_bytes(first) != _canonical_json_bytes(second):
        changed = sorted(
            key for key in _IDENTITY_PAYLOAD_KEYS if first.get(key) != second.get(key)
        )
        raise ExecutionAttestationError(
            "execution identity changed while the receipt was being built: "
            + ", ".join(changed)
        )
    canonical = _canonical_json_bytes(first).decode("utf-8")
    completeness = tuple(
        sorted(
            f"execution_identity.{item.identity_id}=unavailable:{item.unavailable_reason}"
            for item in spec.input_identities
            if item.required_for_execution and not item.available
        )
    )
    readiness = tuple(
        sorted(
            set(spec.scientific_readiness_blockers).union(
                {"execution_receipt.scientific_readiness=not_established"}
            )
        )
    )
    return ExecutionReceipt(
        identity_payload_json=canonical,
        execution_receipt_sha256=sha256(canonical.encode("utf-8")).hexdigest(),
        input_completeness_blockers=completeness,
        scientific_readiness_blockers=readiness,
    )


def verify_execution_receipt(
    receipt: ExecutionReceipt,
    spec: ExecutionReceiptSpec,
    *,
    phase: ExecutionVerificationPhase,
) -> ExecutionReceiptVerification:
    """Recompute the full identity and reject every pre/post-run mismatch."""

    if type(receipt) is not ExecutionReceipt:
        raise TypeError("receipt must be ExecutionReceipt")
    ExecutionReceipt.__post_init__(receipt)
    if type(phase) is not ExecutionVerificationPhase:
        raise TypeError("phase must be ExecutionVerificationPhase")
    observed = build_execution_receipt(spec)
    if receipt.execution_receipt_sha256 != observed.execution_receipt_sha256 or (
        receipt.identity_payload_json != observed.identity_payload_json
    ):
        expected_payload = receipt.identity_payload
        observed_payload = observed.identity_payload
        changed = sorted(
            key
            for key in _IDENTITY_PAYLOAD_KEYS
            if expected_payload.get(key) != observed_payload.get(key)
        )
        raise ExecutionReceiptMismatchError(
            "execution receipt identity mismatch"
            + (": " + ", ".join(changed) if changed else "")
        )
    if receipt.snapshot() != observed.snapshot():
        raise ExecutionReceiptMismatchError(
            "execution receipt readiness or completeness declaration changed"
        )
    return ExecutionReceiptVerification(
        phase=phase,
        expected_receipt_sha256=receipt.execution_receipt_sha256,
        observed_receipt_sha256=observed.execution_receipt_sha256,
        campaign_execution_admissible=False,
        blockers=receipt.campaign_execution_blockers,
    )


def require_campaign_execution(
    receipt: ExecutionReceipt,
    verification: ExecutionReceiptVerification,
) -> None:
    """Fail closed unless a future receipt schema explicitly opens every gate."""

    if type(receipt) is not ExecutionReceipt:
        raise TypeError("receipt must be ExecutionReceipt")
    if type(verification) is not ExecutionReceiptVerification:
        raise TypeError("verification must be ExecutionReceiptVerification")
    if (
        verification.expected_receipt_sha256
        != receipt.execution_receipt_sha256
        or verification.observed_receipt_sha256
        != receipt.execution_receipt_sha256
    ):
        raise ExecutionReceiptMismatchError(
            "verification does not belong to the supplied receipt"
        )
    raise CampaignExecutionRejectedError(
        "campaign execution rejected: " + "; ".join(verification.blockers)
    )


def attach_verified_execution_receipt(
    manifest: Mapping[str, object],
    *,
    receipt: ExecutionReceipt,
    verification: ExecutionReceiptVerification,
    receipt_relative_path: str = "execution_receipt.json",
) -> dict[str, object]:
    """Attach a post-run receipt reference without promoting readiness."""

    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    if "execution_receipt" in manifest or "execution_attestation" in manifest:
        raise ExecutionAttestationError(
            "manifest already contains an execution receipt or attestation"
        )
    if "campaign_ready" in manifest and manifest.get("campaign_ready") is not False:
        raise ExecutionAttestationError(
            "cannot attach a fail-closed receipt to a campaign-ready manifest"
        )
    if type(receipt) is not ExecutionReceipt:
        raise TypeError("receipt must be ExecutionReceipt")
    if type(verification) is not ExecutionReceiptVerification:
        raise TypeError("verification must be ExecutionReceiptVerification")
    if verification.phase is not ExecutionVerificationPhase.POST_EXECUTION:
        raise ExecutionAttestationError(
            "final manifest requires POST_EXECUTION verification"
        )
    if (
        verification.expected_receipt_sha256
        != receipt.execution_receipt_sha256
        or verification.observed_receipt_sha256
        != receipt.execution_receipt_sha256
        or verification.blockers != receipt.campaign_execution_blockers
    ):
        raise ExecutionReceiptMismatchError(
            "post-run verification does not match the supplied receipt"
        )
    path = _canonical_relative_path(
        receipt_relative_path,
        name="receipt_relative_path",
    )
    payload = _json_detached_copy(manifest)
    payload["execution_receipt"] = {
        "path": path,
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "execution_receipt_sha256": receipt.execution_receipt_sha256,
        "identity_verification_status": "VERIFIED",
        "campaign_ready": False,
    }
    payload["execution_attestation"] = verification.snapshot()
    return payload


def write_execution_receipt_atomic(
    path: str | Path,
    receipt: ExecutionReceipt,
) -> Path:
    """Write a deterministic receipt document without replacing different bytes.

    Repeating a write of the same receipt is idempotent.  An existing receipt
    with different bytes is treated as material evidence and is never silently
    overwritten.
    """

    if type(receipt) is not ExecutionReceipt:
        raise TypeError("receipt must be ExecutionReceipt")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            receipt.snapshot(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ExecutionAttestationError(
                "existing execution receipt must be a regular file"
            )
        try:
            existing = destination.read_bytes()
        except OSError as exc:
            raise ExecutionAttestationError(
                "existing execution receipt cannot be read"
            ) from exc
        if existing == rendered:
            return destination
        raise ExecutionAttestationError(
            "refusing to overwrite an existing different execution receipt"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                existing = destination.read_bytes()
            except OSError as exc:
                raise ExecutionAttestationError(
                    "concurrent execution receipt cannot be read"
                ) from exc
            if existing != rendered:
                raise ExecutionAttestationError(
                    "refusing to overwrite a concurrent different execution receipt"
                )
        except OSError as exc:
            raise ExecutionAttestationError(
                "execution receipt cannot be installed atomically"
            ) from exc
        temporary.unlink(missing_ok=True)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_execution_attestation_atomic(
    path: str | Path,
    verification: ExecutionReceiptVerification,
) -> Path:
    """Persist one pre/post verification without replacing different evidence."""

    if type(verification) is not ExecutionReceiptVerification:
        raise TypeError("verification must be ExecutionReceiptVerification")
    ExecutionReceiptVerification.__post_init__(verification)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            verification.snapshot(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ExecutionAttestationError(
                "existing execution attestation must be a regular file"
            )
        try:
            existing = destination.read_bytes()
        except OSError as exc:
            raise ExecutionAttestationError(
                "existing execution attestation cannot be read"
            ) from exc
        if existing == rendered:
            return destination
        raise ExecutionAttestationError(
            "refusing to overwrite an existing different execution attestation"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                existing = destination.read_bytes()
            except OSError as exc:
                raise ExecutionAttestationError(
                    "concurrent execution attestation cannot be read"
                ) from exc
            if existing != rendered:
                raise ExecutionAttestationError(
                    "refusing to overwrite a concurrent different execution "
                    "attestation"
                )
        except OSError as exc:
            raise ExecutionAttestationError(
                "execution attestation cannot be installed atomically"
            ) from exc
        temporary.unlink(missing_ok=True)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_execution_receipt(path: str | Path) -> ExecutionReceipt:
    """Load one regular JSON receipt and revalidate its self-attestation."""

    candidate = Path(path)
    if candidate.is_symlink():
        raise ExecutionAttestationError("execution receipt cannot be a symlink")
    try:
        observed = candidate.read_bytes()
    except OSError as exc:
        raise ExecutionAttestationError(
            f"execution receipt cannot be read: {candidate}"
        ) from exc
    if not observed or len(observed) > MAX_RECEIPT_BYTES:
        raise ExecutionAttestationError(
            "execution receipt byte length is outside the supported range"
        )
    return ExecutionReceipt.from_snapshot(_parse_json_object(observed))


def _identity_payload_once(spec: ExecutionReceiptSpec) -> dict[str, object]:
    repository_root = spec.repository_root.resolve(strict=True)
    if not repository_root.is_dir():
        raise ExecutionAttestationError("repository_root must be a directory")
    repository = _repository_identity(repository_root)
    source_tree = _source_tree_identity(repository_root)
    artifacts = {
        item.artifact_id: _attest_input_artifact(repository_root, item)
        for item in sorted(spec.input_artifacts, key=lambda value: value.artifact_id)
    }
    identities = {
        item.identity_id: item.snapshot()
        for item in sorted(spec.input_identities, key=lambda value: value.identity_id)
    }
    pyproject = _attest_project_file(repository_root, "pyproject.toml")
    uv_lock = _attest_project_file(repository_root, "uv.lock")
    runtime = _python_runtime_identity()
    operating_system = _operating_system_identity()
    dependency_environment = _dependency_environment_identity()
    ledger_configuration = _json_detached_copy(spec.ledger_configuration)
    ledger_payload = {
        "backend": spec.ledger_backend,
        "configuration": ledger_configuration,
        "configuration_sha256": _canonical_sha256(ledger_configuration),
    }
    outputs = sorted(
        _canonical_relative_path(item, name="expected output artifact")
        for item in spec.expected_output_artifacts
    )
    payload = {
        "receipt_contract": {
            "schema_id": EXECUTION_RECEIPT_SCHEMA_ID,
            "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
            "identity_algorithm": CANONICAL_IDENTITY_ALGORITHM,
        },
        "encoding_contract": {
            "canonical_identity_encoding": "UTF-8",
            "canonical_identity_serialization": (
                "JSON sort_keys=true separators=(',',':') ensure_ascii=false "
                "allow_nan=false"
            ),
            "path_encoding": "UTF-8 with POSIX '/' separators",
            "file_hash_encoding": "raw bytes without newline transformation",
            "command_version_output_encoding": (
                "UTF-8 with terminal CR/LF removed"
            ),
        },
        "repository": repository,
        "source_tree": source_tree,
        "python_runtime": runtime,
        "operating_system": operating_system,
        "dependency_environment": dependency_environment,
        "project_files": {
            "pyproject.toml": pyproject,
            "uv.lock": uv_lock,
        },
        "campaign": {
            "configuration_artifact_id": "campaign_configuration",
            "prospective_plan_artifact_id": "prospective_analysis_plan",
            "plan_id": spec.plan_id,
            "plan_sha256": spec.plan_sha256,
        },
        "input_artifacts": artifacts,
        "input_identities": identities,
        "expected_output_artifacts": outputs,
        "ledger": ledger_payload,
        "run": {
            "command": list(spec.run_command),
            "execution_mode": spec.execution_mode,
        },
        "model_version": spec.model_version,
    }
    _validate_identity_payload_shape(payload)
    return payload


def _repository_identity(repository_root: Path) -> dict[str, object]:
    top = Path(_git_text(repository_root, "rev-parse", "--show-toplevel"))
    if top.resolve(strict=True) != repository_root:
        raise ExecutionAttestationError(
            "repository_root differs from Git's top-level worktree"
        )
    status = _git_bytes(
        repository_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    if status:
        raise ExecutionAttestationError(
            "execution receipt requires a clean working tree"
        )
    branch = _git_text(
        repository_root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
    )
    _nonempty_text(branch, name="active Git branch")
    commit = _git_text(repository_root, "rev-parse", "--verify", "HEAD")
    if not _GIT_OBJECT_ID.fullmatch(commit):
        raise ExecutionAttestationError("Git commit has an unsupported identity")
    object_format = _git_text(
        repository_root,
        "rev-parse",
        "--show-object-format",
    )
    if object_format not in {"sha1", "sha256"}:
        raise ExecutionAttestationError("Git object format is unsupported")
    expected_object_length = 40 if object_format == "sha1" else 64
    if len(commit) != expected_object_length:
        raise ExecutionAttestationError(
            "Git commit length does not match its object format"
        )
    tree_object = _git_text(repository_root, "rev-parse", "HEAD^{tree}")
    if (
        not _GIT_OBJECT_ID.fullmatch(tree_object)
        or len(tree_object) != expected_object_length
    ):
        raise ExecutionAttestationError("Git tree object has an unsupported identity")
    return {
        "root": _normalized_absolute_path(repository_root),
        "branch": branch,
        "commit": commit,
        "object_format": object_format,
        "git_tree_object": tree_object,
        "working_tree_clean": True,
        "status_porcelain_v1": "",
    }


def _source_tree_identity(repository_root: Path) -> dict[str, object]:
    observed = _git_bytes(repository_root, "ls-files", "--stage", "-z")
    raw_entries = observed.split(b"\0")
    if raw_entries and raw_entries[-1] == b"":
        raw_entries.pop()
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_entry in raw_entries:
        try:
            raw_metadata, raw_path = raw_entry.split(b"\t", maxsplit=1)
            raw_mode, raw_object_id, raw_stage = raw_metadata.split(b" ")
            mode = raw_mode.decode("ascii", errors="strict")
            object_id = raw_object_id.decode("ascii", errors="strict")
            stage = raw_stage.decode("ascii", errors="strict")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ExecutionAttestationError(
                "Git index entry has an unsupported encoding"
            ) from exc
        if stage != "0":
            raise ExecutionAttestationError(
                "Git index contains an unresolved non-zero-stage entry"
            )
        if mode not in {"100644", "100755"}:
            raise ExecutionAttestationError(
                f"Git-tracked source has unsupported mode {mode}"
            )
        if not _GIT_OBJECT_ID.fullmatch(object_id):
            raise ExecutionAttestationError(
                "Git index object has an unsupported identity"
            )
        try:
            path_text = raw_path.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ExecutionAttestationError(
                "Git-tracked paths must be valid UTF-8"
            ) from exc
        canonical = _canonical_relative_path(path_text, name="Git-tracked path")
        if canonical in seen:
            raise ExecutionAttestationError("Git-tracked source path repeats")
        seen.add(canonical)
        candidate = repository_root.joinpath(*PurePosixPath(canonical).parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise ExecutionAttestationError(
                f"Git-tracked source is not a regular file: {canonical}"
            )
        content = candidate.read_bytes()
        entries.append(
            {
                "path": canonical,
                "git_mode": mode,
                "byte_length": len(content),
                "sha256": sha256(content).hexdigest(),
            }
        )
    entries.sort(key=lambda item: str(item["path"]).encode("utf-8"))
    manifest = {
        "algorithm": SOURCE_TREE_ALGORITHM,
        "path_encoding": "UTF-8 POSIX relative path",
        "content_encoding": "raw bytes",
        "mode_encoding": "Git index octal mode",
        "files": entries,
    }
    return {
        "algorithm": SOURCE_TREE_ALGORITHM,
        "tracked_file_count": len(entries),
        "tracked_byte_length": sum(int(item["byte_length"]) for item in entries),
        "source_tree_sha256": _canonical_sha256(manifest),
    }


def _attest_input_artifact(
    repository_root: Path,
    spec: FileArtifactSpec,
) -> dict[str, object]:
    resolved, relative = _repository_file(repository_root, spec.path)
    content = resolved.read_bytes()
    observed_sha256 = sha256(content).hexdigest()
    if observed_sha256 != spec.expected_sha256:
        raise ExecutionAttestationError(
            f"artifact {spec.artifact_id} SHA-256 mismatch"
        )
    if (
        spec.expected_byte_length is not None
        and len(content) != spec.expected_byte_length
    ):
        raise ExecutionAttestationError(
            f"artifact {spec.artifact_id} byte-length mismatch"
        )
    return {
        "schema_version": spec.schema_version,
        "path": relative,
        "path_base": "repository_root",
        "byte_length": len(content),
        "sha256": observed_sha256,
        "hash_scope": "raw file bytes",
    }


def _attest_project_file(repository_root: Path, name: str) -> dict[str, object]:
    resolved, relative = _repository_file(repository_root, Path(name))
    content = resolved.read_bytes()
    return {
        "path": relative,
        "path_base": "repository_root",
        "byte_length": len(content),
        "sha256": sha256(content).hexdigest(),
        "hash_scope": "raw file bytes",
    }


def _python_runtime_identity() -> dict[str, object]:
    executable = Path(sys.executable).resolve(strict=True)
    if not executable.is_file():
        raise ExecutionAttestationError("Python executable is not a regular file")
    content = executable.read_bytes()
    implementation = platform.python_implementation()
    version = platform.python_version()
    _nonempty_text(implementation, name="Python implementation")
    _nonempty_text(version, name="Python version")
    return {
        "implementation": implementation,
        "version": version,
        "version_info": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
            sys.version_info.releaselevel,
            sys.version_info.serial,
        ],
        "full_version": sys.version,
        "compiler": platform.python_compiler(),
        "cache_tag": sys.implementation.cache_tag,
        "byte_order": sys.byteorder,
        "filesystem_encoding": sys.getfilesystemencoding(),
        "executable_path": _normalized_absolute_path(executable),
        "executable_byte_length": len(content),
        "executable_sha256": sha256(content).hexdigest(),
        "prefix": _normalized_absolute_path(Path(sys.prefix)),
        "base_prefix": _normalized_absolute_path(Path(sys.base_prefix)),
        "virtual_environment": sys.prefix != sys.base_prefix,
    }


def _operating_system_identity() -> dict[str, object]:
    return {
        "system": platform.system() or "UNAVAILABLE",
        "release": platform.release() or "UNAVAILABLE",
        "version": platform.version() or "UNAVAILABLE",
        "platform": platform.platform(aliased=False, terse=False),
        "machine": platform.machine() or "UNAVAILABLE",
        "architecture": list(platform.architecture()),
        "processor": platform.processor() or "UNAVAILABLE",
    }


def _dependency_environment_identity() -> dict[str, object]:
    packages_by_name: dict[str, dict[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ExecutionAttestationError(
                "installed distribution lacks a canonical package name"
            )
        version = distribution.version
        if not isinstance(version, str) or not version.strip():
            raise ExecutionAttestationError(
                f"installed distribution {raw_name!r} lacks a version"
            )
        normalized = _PACKAGE_NORMALIZER.sub("-", raw_name).lower()
        row = {
            "name": raw_name,
            "normalized_name": normalized,
            "version": version,
        }
        previous = packages_by_name.get(normalized)
        if previous is not None and previous != row:
            raise ExecutionAttestationError(
                f"installed distribution identity repeats: {normalized}"
            )
        packages_by_name[normalized] = row
    packages = [packages_by_name[key] for key in sorted(packages_by_name)]
    package_payload = {
        "algorithm": DEPENDENCY_SET_ALGORITHM,
        "packages": packages,
    }
    versions = {row["normalized_name"]: row["version"] for row in packages}
    return {
        "algorithm": DEPENDENCY_SET_ALGORITHM,
        "package_count": len(packages),
        "packages": packages,
        "dependency_set_sha256": _canonical_sha256(package_payload),
        "package_manager_runtime": {
            "pip_version": versions.get("pip"),
            "setuptools_version": versions.get("setuptools"),
            "uv_distribution_version": versions.get("uv"),
            "uv_cli": _external_command_identity("uv", "--version"),
        },
    }


def _external_command_identity(name: str, *version_args: str) -> dict[str, object]:
    """Resolve and hash an external runtime without inventing missing metadata."""

    _nonempty_text(name, name="external command name")
    resolved_name = shutil.which(name)
    if resolved_name is None:
        return {
            "command": name,
            "status": "UNAVAILABLE",
            "executable_path": None,
            "executable_byte_length": None,
            "executable_sha256": None,
            "version_output": None,
        }
    executable = Path(resolved_name).resolve(strict=True)
    if not executable.is_file():
        raise ExecutionAttestationError(
            f"resolved external command {name!r} is not a regular file"
        )
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    try:
        completed = subprocess.run(
            (str(executable), *version_args),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ExecutionAttestationError(
            f"resolved external command {name!r} cannot be executed"
        ) from exc
    if completed.returncode != 0:
        raise ExecutionAttestationError(
            f"resolved external command {name!r} version query failed"
        )
    try:
        version_output = completed.stdout.decode(
            "utf-8", errors="strict"
        ).rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise ExecutionAttestationError(
            f"resolved external command {name!r} version is not UTF-8"
        ) from exc
    _nonempty_text(version_output, name=f"{name} version output")
    content = executable.read_bytes()
    return {
        "command": name,
        "status": "AVAILABLE",
        "executable_path": _normalized_absolute_path(executable),
        "executable_byte_length": len(content),
        "executable_sha256": sha256(content).hexdigest(),
        "version_output": version_output,
    }


def _repository_file(repository_root: Path, path: Path) -> tuple[Path, str]:
    candidate = path if path.is_absolute() else repository_root / path
    if candidate.is_symlink():
        raise ExecutionAttestationError("input artifacts cannot be symlinks")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ExecutionAttestationError(
            f"input artifact cannot be resolved: {path}"
        ) from exc
    if not resolved.is_file() or not resolved.is_relative_to(repository_root):
        raise ExecutionAttestationError(
            "input artifact must be a regular file inside repository_root"
        )
    relative = resolved.relative_to(repository_root).as_posix()
    return resolved, _canonical_relative_path(relative, name="input artifact path")


def _git_text(repository_root: Path, *args: str) -> str:
    try:
        value = _git_bytes(repository_root, *args).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ExecutionAttestationError("Git output must be UTF-8") from exc
    return value.rstrip("\r\n")


def _git_bytes(repository_root: Path, *args: str) -> bytes:
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=repository_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ExecutionAttestationError("Git executable is unavailable") from exc
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ExecutionAttestationError(
            "Git identity command failed"
            + (f": {message}" if message else "")
        )
    return completed.stdout


def _validate_identity_payload_shape(value: Mapping[str, object]) -> None:
    _exact_keys(value, _IDENTITY_PAYLOAD_KEYS, name="identity payload")
    contract = _require_mapping(value.get("receipt_contract"), name="receipt_contract")
    if contract != {
        "schema_id": EXECUTION_RECEIPT_SCHEMA_ID,
        "schema_version": EXECUTION_RECEIPT_SCHEMA_VERSION,
        "identity_algorithm": CANONICAL_IDENTITY_ALGORITHM,
    }:
        raise ExecutionAttestationError("receipt_contract is unsupported")
    repository = _require_mapping(value.get("repository"), name="repository")
    if repository.get("working_tree_clean") is not True:
        raise ExecutionAttestationError("identity payload working tree is not clean")
    commit = repository.get("commit")
    if type(commit) is not str or not _GIT_OBJECT_ID.fullmatch(commit):
        raise ExecutionAttestationError("identity payload commit is malformed")
    source_tree = _require_mapping(value.get("source_tree"), name="source_tree")
    _sha256_digest(
        source_tree.get("source_tree_sha256"),
        name="source_tree_sha256",
    )
    campaign = _require_mapping(value.get("campaign"), name="campaign")
    _sha256_digest(campaign.get("plan_sha256"), name="campaign plan_sha256")
    artifacts = _require_mapping(value.get("input_artifacts"), name="input_artifacts")
    identities = _require_mapping(value.get("input_identities"), name="input_identities")
    if not REQUIRED_FILE_ARTIFACT_IDS.issubset(artifacts):
        raise ExecutionAttestationError("identity payload file artifacts are incomplete")
    if not REQUIRED_SEMANTIC_IDENTITY_IDS.issubset(identities):
        raise ExecutionAttestationError(
            "identity payload semantic identities are incomplete"
        )
    _canonical_json_bytes(value)


def _parse_json_object(observed: bytes) -> dict[str, object]:
    try:
        text = observed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExecutionAttestationError("receipt JSON must be UTF-8") from exc

    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ExecutionAttestationError(
                    f"receipt JSON repeats object key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ExecutionAttestationError(
            f"receipt JSON contains non-finite constant {value}"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ExecutionAttestationError("execution receipt JSON is invalid") from exc
    if type(value) is not dict:
        raise ExecutionAttestationError("execution receipt JSON must be an object")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    _validate_json_value(value, path="$")
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ExecutionAttestationError(
            "value is not canonical UTF-8 JSON"
        ) from exc


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ExecutionAttestationError(f"{path} contains NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ExecutionAttestationError(f"{path} has a non-string key")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    if type(value) in {list, tuple}:
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    raise ExecutionAttestationError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


def _json_detached_copy(value: object) -> dict[str, object]:
    encoded = _canonical_json_bytes(value)
    return _parse_json_object(encoded)


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _normalized_absolute_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    rendered = resolved.as_posix()
    if not rendered or "\\" in rendered:
        raise ExecutionAttestationError("absolute path normalization failed")
    return rendered


def _canonical_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ExecutionAttestationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )
    if "\\" in value or "\x00" in value:
        raise ExecutionAttestationError(
            f"{name} must use POSIX separators and contain no NUL"
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or re.match(r"[A-Za-z]:", value)
        or path.parts in {(), (".",)}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ExecutionAttestationError(f"{name} is not a canonical relative path")
    rendered = path.as_posix()
    if rendered != value:
        raise ExecutionAttestationError(f"{name} is not lexically canonical")
    try:
        rendered.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ExecutionAttestationError(f"{name} must be valid UTF-8") from exc
    return rendered


def _validate_expected_outputs(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or not values:
        raise TypeError("expected_output_artifacts must be a non-empty exact tuple")
    normalized = tuple(
        _canonical_relative_path(value, name="expected output artifact")
        for value in values
    )
    if len(set(normalized)) != len(normalized):
        raise ExecutionAttestationError("expected output artifact paths repeat")


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ExecutionAttestationError(
            f"{name} must be a canonical 1-192 character identifier"
        )
    return value


def _nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ExecutionAttestationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )
    return value


def _sha256_digest(value: object, *, name: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ExecutionAttestationError(f"{name} must be lowercase SHA-256 hex")
    return value


def _strict_int(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ExecutionAttestationError(
            f"{name} must be an exact integer greater than or equal to {minimum}"
        )
    return value


def _unique_ids(values: Sequence[str], *, name: str) -> None:
    if len(set(values)) != len(values):
        raise ExecutionAttestationError(f"{name} must be unique")


def _canonical_blockers(values: object, *, name: str) -> tuple[str, ...]:
    if type(values) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    selected = tuple(_nonempty_text(item, name=name) for item in values)
    return tuple(sorted(set(selected)))


def _blockers_from_json(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ExecutionAttestationError(f"{name} must be an array")
    selected = tuple(_nonempty_text(item, name=name) for item in value)
    canonical = tuple(sorted(set(selected)))
    if selected != canonical:
        raise ExecutionAttestationError(f"{name} must be unique and sorted")
    return canonical


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise ExecutionAttestationError(
            f"{name} keys differ: missing={sorted(set(expected) - actual)}, "
            f"unknown={sorted(actual - set(expected))}"
        )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ExecutionAttestationError(f"{name} must be an object")
    return value  # type: ignore[return-value]


def _required_string(value: Mapping[str, object], field: str) -> str:
    return _nonempty_text(value.get(field), name=field)


__all__ = [
    "CANONICAL_IDENTITY_ALGORITHM",
    "DEPENDENCY_SET_ALGORITHM",
    "EXECUTION_RECEIPT_SCHEMA_ID",
    "EXECUTION_RECEIPT_SCHEMA_VERSION",
    "MAX_RECEIPT_BYTES",
    "REQUIRED_FILE_ARTIFACT_IDS",
    "REQUIRED_SEMANTIC_IDENTITY_IDS",
    "SOURCE_TREE_ALGORITHM",
    "CampaignExecutionRejectedError",
    "DeclaredIdentity",
    "ExecutionAttestationError",
    "ExecutionReceipt",
    "ExecutionReceiptMismatchError",
    "ExecutionReceiptSpec",
    "ExecutionReceiptVerification",
    "ExecutionVerificationPhase",
    "FileArtifactSpec",
    "attach_verified_execution_receipt",
    "build_execution_receipt",
    "load_execution_receipt",
    "require_campaign_execution",
    "verify_execution_receipt",
    "write_execution_attestation_atomic",
    "write_execution_receipt_atomic",
]
