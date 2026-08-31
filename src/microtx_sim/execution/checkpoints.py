"""Crash-safe resumable execution checkpoints.

The checkpoint index is the sole authority for completed work.  Result payloads
are first written as immutable, content-addressed blocks and are made visible by
one atomic replacement of ``checkpoint.json``.  A crash can therefore leave an
unreferenced block, but can never expose a partially committed work unit.

All writes are performed by the coordinating process.  Worker processes return
payloads to that coordinator; they never mutate the checkpoint directory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from threading import RLock
from typing import Final


CHECKPOINT_SCHEMA_ID: Final[str] = "microtx_sim.resumable_checkpoint.v2"
PROGRESS_SCHEMA_ID: Final[str] = "microtx_sim.execution_progress.v2"
CHECKPOINT_SCHEMA_VERSION: Final[str] = "2.0"
CHECKPOINT_CANONICALIZATION: Final[str] = (
    "microtx_sim.execution_checkpoint.canonical_json_utf8.v1"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}\Z")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{6}\Z")


class CheckpointError(RuntimeError):
    """Base class for fail-closed checkpoint failures."""


class IncompatibleCheckpointError(CheckpointError):
    """Raised when an execution identity or work plan changed."""


class CheckpointCorruptError(CheckpointError):
    """Raised when a checkpoint or referenced payload fails validation."""


class CheckpointIncompleteError(CheckpointError):
    """Raised when completion is requested with work still outstanding."""


class DuplicateWorkUnitError(CheckpointError):
    """Raised before executing or committing an already completed unit."""


class ExecutionPhase(str, Enum):
    MAIN_BATCH = "main_batch"
    SENSITIVITY = "sensitivity"
    FINALIZATION = "finalization"
    COMPLETE = "complete"


class CheckpointStatus(str, Enum):
    READY = "READY"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    COMPUTE_COMPLETE_EXPORT_PENDING = "COMPUTE_COMPLETE_EXPORT_PENDING"
    COMPLETE = "COMPLETE"


@dataclass(frozen=True, slots=True)
class BackendIdentity:
    """Resolved execution backend; an automatic choice remains explicit."""

    requested_backend: str
    resolved_backend: str
    library: str
    library_version: str
    device_name: str
    device_id: str
    precision_mode: str
    worker_count: int
    batch_size: int
    scheduling_policy: str
    native_thread_runtime: str
    native_thread_library_path: str
    native_thread_library_sha256: str
    native_thread_getter_symbol: str
    native_thread_setter_symbol: str
    native_thread_limit: int
    fallback_policy: str = "FORBID_SILENT_FALLBACK"
    compute_capability: str | None = None
    driver_version: str | None = None
    runtime_version: str | None = None

    def __post_init__(self) -> None:
        if self.requested_backend not in {"cpu", "gpu", "auto"}:
            raise ValueError("requested_backend must be cpu, gpu, or auto")
        if self.resolved_backend not in {"cpu", "gpu"}:
            raise ValueError("resolved_backend must be cpu or gpu")
        if (
            self.requested_backend in {"cpu", "gpu"}
            and self.requested_backend != self.resolved_backend
        ):
            raise ValueError("an explicit backend cannot resolve to another backend")
        for name, value in (
            ("library", self.library),
            ("library_version", self.library_version),
            ("device_name", self.device_name),
            ("device_id", self.device_id),
            ("precision_mode", self.precision_mode),
            ("scheduling_policy", self.scheduling_policy),
            ("native_thread_runtime", self.native_thread_runtime),
            ("native_thread_getter_symbol", self.native_thread_getter_symbol),
            ("native_thread_setter_symbol", self.native_thread_setter_symbol),
        ):
            _text(value, name=name)
        _normalized_path(
            self.native_thread_library_path,
            name="native_thread_library_path",
        )
        _digest(
            self.native_thread_library_sha256,
            name="native_thread_library_sha256",
        )
        if type(self.native_thread_limit) is not int or self.native_thread_limit != 1:
            raise ValueError("native_thread_limit must be exactly one")
        if self.fallback_policy != "FORBID_SILENT_FALLBACK":
            raise ValueError("backend identity must forbid silent fallback")
        for name, value in (
            ("worker_count", self.worker_count),
            ("batch_size", self.batch_size),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.resolved_backend == "gpu":
            for name, value in (
                ("compute_capability", self.compute_capability),
                ("driver_version", self.driver_version),
                ("runtime_version", self.runtime_version),
            ):
                _text(value, name=name)

    def snapshot(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "library": self.library,
            "library_version": self.library_version,
            "device_name": self.device_name,
            "device_id": self.device_id,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "precision_mode": self.precision_mode,
            "worker_count": self.worker_count,
            "batch_size": self.batch_size,
            "scheduling_policy": self.scheduling_policy,
            "native_thread_runtime": self.native_thread_runtime,
            "native_thread_library_path": _normalized_path(
                self.native_thread_library_path,
                name="native_thread_library_path",
            ),
            "native_thread_library_sha256": (
                self.native_thread_library_sha256
            ),
            "native_thread_getter_symbol": self.native_thread_getter_symbol,
            "native_thread_setter_symbol": self.native_thread_setter_symbol,
            "native_thread_limit": self.native_thread_limit,
            "fallback_policy": self.fallback_policy,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.snapshot())


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Interpreter, platform, and installed-dependency identity."""

    python_implementation: str
    python_version: str
    python_executable: str
    python_executable_sha256: str
    dependency_lock_sha256: str
    installed_dependencies_sha256: str
    operating_system: str
    os_release: str
    machine_architecture: str
    processor: str

    def __post_init__(self) -> None:
        for name, value in (
            ("python_implementation", self.python_implementation),
            ("python_version", self.python_version),
            ("operating_system", self.operating_system),
            ("os_release", self.os_release),
            ("machine_architecture", self.machine_architecture),
            ("processor", self.processor),
        ):
            _text(value, name=name)
        _normalized_path(self.python_executable, name="python_executable")
        for name, value in (
            ("python_executable_sha256", self.python_executable_sha256),
            ("dependency_lock_sha256", self.dependency_lock_sha256),
            (
                "installed_dependencies_sha256",
                self.installed_dependencies_sha256,
            ),
        ):
            _digest(value, name=name)

    def snapshot(self) -> dict[str, object]:
        return {
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "python_executable": _normalized_path(
                self.python_executable, name="python_executable"
            ),
            "python_executable_sha256": self.python_executable_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "installed_dependencies_sha256": self.installed_dependencies_sha256,
            "operating_system": self.operating_system,
            "os_release": self.os_release,
            "machine_architecture": self.machine_architecture,
            "processor": self.processor,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.snapshot())


@dataclass(frozen=True, slots=True)
class SensitivityWorkUnit:
    """One atomic sensitivity unit in declared execution order."""

    parameter_id: str
    level_id: str
    seed: int

    def __post_init__(self) -> None:
        _identifier(self.parameter_id, name="parameter_id")
        _identifier(self.level_id, name="level_id")
        _seed(self.seed)

    def snapshot(self) -> dict[str, object]:
        return {
            "phase": ExecutionPhase.SENSITIVITY.value,
            "parameter_id": self.parameter_id,
            "level_id": self.level_id,
            "seed": self.seed,
        }

    @property
    def unit_id(self) -> str:
        return "sensitivity:" + canonical_sha256(self.snapshot())


@dataclass(frozen=True, slots=True)
class ExecutionWorkPlan:
    """Immutable enumeration of every progress-denominator work unit."""

    seeds: tuple[int, ...]
    scenario_ids: tuple[str, ...]
    sensitivity_units: tuple[SensitivityWorkUnit, ...] = ()

    def __post_init__(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("work-plan seeds must be non-empty and unique")
        for seed in self.seeds:
            _seed(seed)
        if not self.scenario_ids or len(set(self.scenario_ids)) != len(
            self.scenario_ids
        ):
            raise ValueError("scenario_ids must be non-empty and unique")
        for scenario_id in self.scenario_ids:
            _identifier(scenario_id, name="scenario_id")
        unit_ids = [unit.unit_id for unit in self.sensitivity_units]
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("sensitivity work units must be unique")
        unknown_seeds = {
            unit.seed for unit in self.sensitivity_units if unit.seed not in self.seeds
        }
        if unknown_seeds:
            raise ValueError("sensitivity units contain undeclared seeds")

    @classmethod
    def build(
        cls,
        *,
        seeds: Sequence[int],
        scenario_ids: Sequence[str],
        sensitivity_units: Sequence[SensitivityWorkUnit] = (),
    ) -> "ExecutionWorkPlan":
        return cls(
            seeds=tuple(seeds),
            scenario_ids=tuple(scenario_ids),
            sensitivity_units=tuple(sensitivity_units),
        )

    @property
    def main_total_units(self) -> int:
        return len(self.seeds) * len(self.scenario_ids)

    @property
    def sensitivity_total_units(self) -> int:
        return len(self.sensitivity_units)

    @property
    def total_units(self) -> int:
        return self.main_total_units + self.sensitivity_total_units

    @property
    def seed_set_sha256(self) -> str:
        return canonical_sha256({"seeds": list(self.seeds)})

    def main_descriptor(self, seed: int, scenario_id: str) -> dict[str, object]:
        if seed not in self.seeds or scenario_id not in self.scenario_ids:
            raise ValueError("main work unit is not declared")
        return {
            "phase": ExecutionPhase.MAIN_BATCH.value,
            "seed": seed,
            "scenario_id": scenario_id,
        }

    def main_unit_id(self, seed: int, scenario_id: str) -> str:
        return "main:" + canonical_sha256(
            self.main_descriptor(seed, scenario_id)
        )

    def main_unit_ids(self, seed: int) -> tuple[str, ...]:
        return tuple(
            self.main_unit_id(seed, scenario_id)
            for scenario_id in self.scenario_ids
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_id": CHECKPOINT_SCHEMA_ID,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "seeds": list(self.seeds),
            "scenario_ids": list(self.scenario_ids),
            "sensitivity_units": [
                unit.snapshot() for unit in self.sensitivity_units
            ],
            "expected_work_units": {
                "main_batch": self.main_total_units,
                "sensitivity": self.sensitivity_total_units,
                "overall": self.total_units,
            },
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.snapshot())


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """Every field that must match before completed work may be reused."""

    run_id: str
    attempt_id: str
    implementation_id: str
    source_tree_sha256: str
    git_commit: str
    git_branch: str
    configuration_sha256: str
    analysis_plan_id: str
    analysis_plan_sha256: str
    seed_set_sha256: str
    work_plan_sha256: str
    backend: BackendIdentity
    runtime: RuntimeIdentity
    execution_kind: str = "exploratory_synthetic"
    payload_schema_id: str = "microtx_sim.execution_payload.json.v1"

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("implementation_id", self.implementation_id),
            ("analysis_plan_id", self.analysis_plan_id),
            ("execution_kind", self.execution_kind),
            ("payload_schema_id", self.payload_schema_id),
        ):
            _identifier(value, name=name)
        if not _ATTEMPT_ID.fullmatch(self.attempt_id):
            raise ValueError("attempt_id must have form attempt-000001")
        for name, value in (
            ("source_tree_sha256", self.source_tree_sha256),
            ("configuration_sha256", self.configuration_sha256),
            ("analysis_plan_sha256", self.analysis_plan_sha256),
            ("seed_set_sha256", self.seed_set_sha256),
            ("work_plan_sha256", self.work_plan_sha256),
        ):
            _digest(value, name=name)
        if type(self.git_commit) is not str or not _GIT_COMMIT.fullmatch(
            self.git_commit
        ):
            raise ValueError("git_commit must be a full lowercase commit hash")
        _text(self.git_branch, name="git_branch")

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_id": CHECKPOINT_SCHEMA_ID,
            "canonicalization": CHECKPOINT_CANONICALIZATION,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "implementation_id": self.implementation_id,
            "source_tree_sha256": self.source_tree_sha256,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "configuration_sha256": self.configuration_sha256,
            "analysis_plan_id": self.analysis_plan_id,
            "analysis_plan_sha256": self.analysis_plan_sha256,
            "seed_set_sha256": self.seed_set_sha256,
            "work_plan_sha256": self.work_plan_sha256,
            "backend": self.backend.snapshot(),
            "backend_identity_sha256": self.backend.identity_sha256,
            "runtime": self.runtime.snapshot(),
            "runtime_identity_sha256": self.runtime.identity_sha256,
            "execution_kind": self.execution_kind,
            "payload_schema_id": self.payload_schema_id,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.snapshot())


@dataclass(frozen=True, slots=True)
class PriorExecutionLineage:
    """Observed identity and failure state of a superseded incomplete run."""

    run_id: str | None
    run_identity_status: str
    attempt_id: str
    configuration_sha256: str
    analysis_plan_sha256: str
    source_identity_status: str
    source_tree_sha256: str | None
    git_commit: str | None
    observed_status: str
    reason_final_outputs_unavailable: str
    progress_artifact_path: str
    progress_artifact_sha256: str
    final_outputs_available: bool = False

    def __post_init__(self) -> None:
        if self.run_identity_status == "RECORDED_AND_VERIFIED":
            _identifier(self.run_id, name="previous.run_id")
        elif self.run_identity_status == "NOT_RECORDED_BY_CHECKPOINT_SCHEMA_V1":
            if self.run_id is not None:
                raise ValueError("unattested v1 run identity must remain null")
        else:
            raise ValueError("previous run_identity_status is invalid")
        if not _ATTEMPT_ID.fullmatch(self.attempt_id):
            raise ValueError("previous attempt_id is invalid")
        for name, value in (
            ("configuration_sha256", self.configuration_sha256),
            ("analysis_plan_sha256", self.analysis_plan_sha256),
            ("progress_artifact_sha256", self.progress_artifact_sha256),
        ):
            _digest(value, name=f"previous.{name}")
        if self.source_identity_status == "RECORDED_AND_VERIFIED":
            _digest(
                self.source_tree_sha256,
                name="previous.source_tree_sha256",
            )
            if type(self.git_commit) is not str or not _GIT_COMMIT.fullmatch(
                self.git_commit
            ):
                raise ValueError("previous git_commit is invalid")
        elif (
            self.source_identity_status
            == "NOT_RECORDED_BY_CHECKPOINT_SCHEMA_V1"
        ):
            if self.source_tree_sha256 is not None or self.git_commit is not None:
                raise ValueError(
                    "unattested v1 source identity must remain null"
                )
        else:
            raise ValueError("previous source_identity_status is invalid")
        if self.observed_status not in {"INCOMPLETE", "INTERRUPTED"}:
            raise ValueError("previous status must be INCOMPLETE or INTERRUPTED")
        _text(
            self.reason_final_outputs_unavailable,
            name="reason_final_outputs_unavailable",
        )
        _relative_path(
            self.progress_artifact_path, name="progress_artifact_path"
        )
        if self.final_outputs_available is not False:
            raise ValueError("superseded lineage cannot claim final outputs")

    def snapshot(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_identity_status": self.run_identity_status,
            "attempt_id": self.attempt_id,
            "configuration_sha256": self.configuration_sha256,
            "analysis_plan_sha256": self.analysis_plan_sha256,
            "source_identity_status": self.source_identity_status,
            "source_tree_sha256": self.source_tree_sha256,
            "git_commit": self.git_commit,
            "observed_status": self.observed_status,
            "reason_final_outputs_unavailable": (
                self.reason_final_outputs_unavailable
            ),
            "progress_artifact_path": _relative_path(
                self.progress_artifact_path, name="progress_artifact_path"
            ),
            "progress_artifact_sha256": self.progress_artifact_sha256,
            "final_outputs_available": False,
        }


@dataclass(frozen=True, slots=True)
class ExecutionLineage:
    """Immutable link from one incomplete attempt to its successor."""

    previous: PriorExecutionLineage
    successor_run_id: str
    successor_attempt_id: str
    successor_implementation_id: str
    relation: str = "SUPERSEDES_INCOMPLETE_EXECUTION_ONLY"
    scientific_plan_changed: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("successor_run_id", self.successor_run_id),
            ("successor_implementation_id", self.successor_implementation_id),
        ):
            _identifier(value, name=name)
        if not _ATTEMPT_ID.fullmatch(self.successor_attempt_id):
            raise ValueError("successor_attempt_id is invalid")
        if (
            self.previous.run_id is not None
            and self.previous.run_id == self.successor_run_id
        ):
            raise ValueError("a successor must use a new run_id")
        if self.previous.attempt_id == self.successor_attempt_id:
            raise ValueError("a successor must use a new attempt_id")
        if self.relation != "SUPERSEDES_INCOMPLETE_EXECUTION_ONLY":
            raise ValueError("lineage relation cannot alter scientific semantics")
        if self.scientific_plan_changed is not False:
            raise ValueError("execution-only lineage cannot change the plan")

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_id": CHECKPOINT_SCHEMA_ID,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "relation": self.relation,
            "scientific_plan_changed": False,
            "previous_execution": self.previous.snapshot(),
            "successor_execution": {
                "run_id": self.successor_run_id,
                "attempt_id": self.successor_attempt_id,
                "implementation_id": self.successor_implementation_id,
            },
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.snapshot())


def canonical_sha256(value: object) -> str:
    """Hash strict JSON with deterministic ordering and UTF-8 encoding."""

    return sha256(_canonical_json_bytes(value)).hexdigest()


def canonical_level_id(value: int | float) -> str:
    """Return an exact, deterministic sensitivity-level identity."""

    if type(value) is int:
        return f"int:{value}"
    if type(value) is float and math.isfinite(value):
        return f"float64:{value.hex()}"
    raise ValueError("sensitivity level must be a finite int or float")


class ResumableCheckpointStore:
    """Single-coordinator transactional store for main and sensitivity work."""

    def __init__(
        self,
        *,
        attempt_dir: Path,
        identity: ExecutionIdentity,
        work_plan: ExecutionWorkPlan,
        lineage: ExecutionLineage | None,
        state: dict[str, object],
        clock: Callable[[], datetime],
    ) -> None:
        self.attempt_dir = attempt_dir
        self.identity = identity
        self.work_plan = work_plan
        self.lineage = lineage
        self._state = state
        self._clock = clock
        self._lock = RLock()

    @classmethod
    def create(
        cls,
        progress_root: str | Path,
        *,
        identity: ExecutionIdentity,
        work_plan: ExecutionWorkPlan,
        lineage: ExecutionLineage | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> "ResumableCheckpointStore":
        """Create a new attempt without modifying any existing attempt."""

        _validate_identity_plan(identity, work_plan)
        _validate_lineage(identity, lineage)
        root = Path(progress_root)
        _ensure_real_directory(root, create=True)
        attempt_dir = root / identity.attempt_id
        try:
            attempt_dir.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(
                f"checkpoint attempt already exists: {identity.attempt_id}"
            ) from exc
        _ensure_real_directory(attempt_dir, create=False)
        _ensure_real_directory(attempt_dir / "units", create=True)
        _ensure_real_directory(
            attempt_dir / "units" / ExecutionPhase.MAIN_BATCH.value,
            create=True,
        )
        _ensure_real_directory(
            attempt_dir / "units" / "sensitivity", create=True
        )

        effective_clock = clock or _utc_now
        created_at = _timestamp(effective_clock())
        identity_path = attempt_dir / "execution_identity.json"
        work_plan_path = attempt_dir / "work_plan.json"
        lineage_path = attempt_dir / "execution_lineage.json"
        _atomic_write_json(identity_path, identity.snapshot(), replace=False)
        _atomic_write_json(work_plan_path, work_plan.snapshot(), replace=False)
        lineage_file_sha256: str | None = None
        if lineage is not None:
            _atomic_write_json(lineage_path, lineage.snapshot(), replace=False)
            lineage_file_sha256 = _file_sha256(lineage_path)

        immutable_files: dict[str, object] = {
            "execution_identity": _file_reference(
                attempt_dir, identity_path, role="EXECUTION_IDENTITY"
            ),
            "work_plan": _file_reference(
                attempt_dir, work_plan_path, role="WORK_PLAN"
            ),
            "execution_lineage": (
                {
                    "path": lineage_path.relative_to(attempt_dir).as_posix(),
                    "sha256": lineage_file_sha256,
                    "role": "EXECUTION_LINEAGE",
                }
                if lineage is not None
                else None
            ),
        }
        state: dict[str, object] = {
            "schema_id": CHECKPOINT_SCHEMA_ID,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "canonicalization": CHECKPOINT_CANONICALIZATION,
            "generation": 0,
            "run_id": identity.run_id,
            "attempt_id": identity.attempt_id,
            "execution_identity_sha256": identity.identity_sha256,
            "source_tree_sha256": identity.source_tree_sha256,
            "git_commit": identity.git_commit,
            "configuration_sha256": identity.configuration_sha256,
            "analysis_plan_sha256": identity.analysis_plan_sha256,
            "backend_identity_sha256": identity.backend.identity_sha256,
            "runtime_identity_sha256": identity.runtime.identity_sha256,
            "work_plan_sha256": work_plan.identity_sha256,
            "lineage_sha256": (
                lineage.identity_sha256 if lineage is not None else None
            ),
            "immutable_files": immutable_files,
            "created_at": created_at,
            "updated_at": created_at,
            "status": CheckpointStatus.READY.value,
            "phase": ExecutionPhase.MAIN_BATCH.value,
            "current_work_unit": None,
            "current_parameter": None,
            "current_level": None,
            "current_seed": None,
            "in_progress_unit_ids": [],
            "completed_units": [],
            "expected_work_units": {
                "main_batch": work_plan.main_total_units,
                "sensitivity": work_plan.sensitivity_total_units,
                "overall": work_plan.total_units,
            },
            "resume_count": 0,
            "events": [],
            "error": None,
            "final_outputs": [],
            "output_paths": {
                "checkpoint": "checkpoint.json",
                "progress": "progress.json",
                "payload_root": "units",
            },
        }
        store = cls(
            attempt_dir=attempt_dir,
            identity=identity,
            work_plan=work_plan,
            lineage=lineage,
            state=state,
            clock=effective_clock,
        )
        store._commit_state(state, increment=False)
        return store

    @classmethod
    def load(
        cls,
        attempt_dir: str | Path,
        *,
        expected_identity: ExecutionIdentity,
        expected_work_plan: ExecutionWorkPlan,
        expected_lineage: ExecutionLineage | None = None,
        clock: Callable[[], datetime] | None = None,
        repair_progress: bool = True,
    ) -> "ResumableCheckpointStore":
        """Load only after verifying every immutable identity and payload."""

        directory = Path(attempt_dir)
        _ensure_real_directory(directory, create=False)
        if directory.name != expected_identity.attempt_id:
            raise IncompatibleCheckpointError(
                "attempt directory differs from expected attempt_id"
            )
        _validate_identity_plan(expected_identity, expected_work_plan)
        _validate_lineage(expected_identity, expected_lineage)
        state = _read_json_object(directory / "checkpoint.json")
        _verify_state_digest(state)
        _assert_equal(
            state.get("schema_id"),
            CHECKPOINT_SCHEMA_ID,
            "checkpoint schema ID",
        )
        _assert_equal(
            state.get("schema_version"),
            CHECKPOINT_SCHEMA_VERSION,
            "checkpoint schema version",
        )
        _assert_equal(
            state.get("execution_identity_sha256"),
            expected_identity.identity_sha256,
            "execution identity",
        )
        _assert_equal(
            state.get("work_plan_sha256"),
            expected_work_plan.identity_sha256,
            "work plan",
        )
        for observed, expected, name in (
            (
                state.get("source_tree_sha256"),
                expected_identity.source_tree_sha256,
                "source tree",
            ),
            (state.get("git_commit"), expected_identity.git_commit, "Git commit"),
            (
                state.get("configuration_sha256"),
                expected_identity.configuration_sha256,
                "configuration",
            ),
            (
                state.get("analysis_plan_sha256"),
                expected_identity.analysis_plan_sha256,
                "analysis plan",
            ),
            (
                state.get("backend_identity_sha256"),
                expected_identity.backend.identity_sha256,
                "backend identity",
            ),
            (
                state.get("runtime_identity_sha256"),
                expected_identity.runtime.identity_sha256,
                "runtime identity",
            ),
        ):
            _assert_equal(observed, expected, name)
        _assert_equal(
            state.get("lineage_sha256"),
            (
                expected_lineage.identity_sha256
                if expected_lineage is not None
                else None
            ),
            "execution lineage",
        )
        _assert_equal(state.get("run_id"), expected_identity.run_id, "run_id")
        _assert_equal(
            state.get("attempt_id"), expected_identity.attempt_id, "attempt_id"
        )
        immutable = state.get("immutable_files")
        if type(immutable) is not dict:
            raise CheckpointCorruptError("immutable_files must be an object")
        _verify_immutable_file(
            directory,
            immutable.get("execution_identity"),
            expected_identity.snapshot(),
            name="execution identity",
        )
        _verify_immutable_file(
            directory,
            immutable.get("work_plan"),
            expected_work_plan.snapshot(),
            name="work plan",
        )
        lineage_reference = immutable.get("execution_lineage")
        if expected_lineage is None:
            if lineage_reference is not None:
                raise IncompatibleCheckpointError(
                    "unexpected execution lineage is present"
                )
        else:
            _verify_immutable_file(
                directory,
                lineage_reference,
                expected_lineage.snapshot(),
                name="execution lineage",
            )
        _validate_state_structure(state, expected_work_plan)
        store = cls(
            attempt_dir=directory,
            identity=expected_identity,
            work_plan=expected_work_plan,
            lineage=expected_lineage,
            state=state,
            clock=clock or _utc_now,
        )
        store._verify_all_payload_blocks()
        if repair_progress:
            store._write_progress()
        return store

    @classmethod
    def resume(
        cls,
        attempt_dir: str | Path,
        *,
        expected_identity: ExecutionIdentity,
        expected_work_plan: ExecutionWorkPlan,
        expected_lineage: ExecutionLineage | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> "ResumableCheckpointStore":
        """Verify a checkpoint and requeue only units not atomically committed."""

        store = cls.load(
            attempt_dir,
            expected_identity=expected_identity,
            expected_work_plan=expected_work_plan,
            expected_lineage=expected_lineage,
            clock=clock,
        )
        with store._lock:
            if store.status is CheckpointStatus.COMPLETE:
                return store
            candidate = deepcopy(store._state)
            events = _list(candidate, "events")
            events.append(
                {
                    "event": "RESUME",
                    "at": store._now(),
                    "previous_status": candidate["status"],
                    "requeued_in_progress_unit_ids": list(
                        _list(candidate, "in_progress_unit_ids")
                    ),
                    "previous_error": deepcopy(candidate.get("error")),
                }
            )
            candidate["in_progress_unit_ids"] = []
            candidate["current_work_unit"] = None
            candidate["error"] = None
            candidate["resume_count"] = _integer(
                candidate.get("resume_count"), name="resume_count"
            ) + 1
            store._set_phase_and_running_status(candidate)
            store._commit_state(candidate)
        return store

    @property
    def status(self) -> CheckpointStatus:
        return CheckpointStatus(str(self._state["status"]))

    @property
    def checkpoint_path(self) -> Path:
        return self.attempt_dir / "checkpoint.json"

    @property
    def progress_path(self) -> Path:
        return self.attempt_dir / "progress.json"

    @property
    def checkpoint_sha256(self) -> str:
        """Canonical state identity (distinct from pretty-file byte hash)."""

        return _digest(
            self._state.get("checkpoint_sha256"), name="checkpoint_sha256"
        )

    @property
    def checkpoint_file_sha256(self) -> str:
        return _file_sha256(self.checkpoint_path)

    @property
    def completed_main_seeds(self) -> tuple[int, ...]:
        completed = self._completed_ids()
        return tuple(
            seed
            for seed in self.work_plan.seeds
            if all(
                unit_id in completed
                for unit_id in self.work_plan.main_unit_ids(seed)
            )
        )

    @property
    def remaining_main_seeds(self) -> tuple[int, ...]:
        completed = set(self.completed_main_seeds)
        return tuple(seed for seed in self.work_plan.seeds if seed not in completed)

    @property
    def completed_sensitivity_units(self) -> tuple[SensitivityWorkUnit, ...]:
        completed = self._completed_ids()
        return tuple(
            unit
            for unit in self.work_plan.sensitivity_units
            if unit.unit_id in completed
        )

    @property
    def remaining_sensitivity_units(self) -> tuple[SensitivityWorkUnit, ...]:
        completed = self._completed_ids()
        return tuple(
            unit
            for unit in self.work_plan.sensitivity_units
            if unit.unit_id not in completed
        )

    @property
    def progress_snapshot(self) -> dict[str, object]:
        return self._build_progress()

    def begin_main_seed(self, seed: int) -> None:
        """Declare one seed block in flight without marking any unit complete."""

        unit_ids = self.work_plan.main_unit_ids(seed)
        self._begin_units(
            unit_ids,
            {
                "phase": ExecutionPhase.MAIN_BATCH.value,
                "seed": seed,
                "scenario_id": None,
                "atomic_block_units": len(unit_ids),
            },
        )

    def commit_main_seed(
        self,
        seed: int,
        scenario_payloads: Mapping[str, object],
    ) -> None:
        """Atomically commit every declared scenario result for one seed."""

        if not isinstance(scenario_payloads, Mapping):
            raise TypeError("scenario_payloads must be a mapping")
        expected = set(self.work_plan.scenario_ids)
        observed = set(scenario_payloads)
        if observed != expected or any(type(key) is not str for key in observed):
            raise ValueError(
                "scenario payloads must match the complete declared scenario set"
            )
        entries = []
        for scenario_id in self.work_plan.scenario_ids:
            descriptor = self.work_plan.main_descriptor(seed, scenario_id)
            entries.append(
                _payload_entry(
                    unit_id=self.work_plan.main_unit_id(seed, scenario_id),
                    descriptor=descriptor,
                    payload=scenario_payloads[scenario_id],
                )
            )
        self._commit_block(
            phase=ExecutionPhase.MAIN_BATCH,
            block_key={"seed": seed},
            entries=entries,
        )

    def begin_sensitivity(
        self, parameter_id: str, level_id: str, seed: int
    ) -> None:
        unit = self._sensitivity_unit(parameter_id, level_id, seed)
        self._begin_units((unit.unit_id,), unit.snapshot())

    def commit_sensitivity(
        self,
        parameter_id: str,
        level_id: str,
        seed: int,
        payload: object,
    ) -> None:
        """Atomically commit one parameter/level/seed sensitivity payload."""

        unit = self._sensitivity_unit(parameter_id, level_id, seed)
        self._commit_block(
            phase=ExecutionPhase.SENSITIVITY,
            block_key={
                "parameter_id": parameter_id,
                "level_id": level_id,
                "seed": seed,
            },
            entries=[
                _payload_entry(
                    unit_id=unit.unit_id,
                    descriptor=unit.snapshot(),
                    payload=payload,
                )
            ],
        )

    def load_main_seed_payload(self, seed: int) -> dict[str, object]:
        """Return detached, checksum-verified scenario payloads for one seed."""

        result: dict[str, object] = {}
        records = self._records_by_id()
        for scenario_id in self.work_plan.scenario_ids:
            unit_id = self.work_plan.main_unit_id(seed, scenario_id)
            record = records.get(unit_id)
            if record is None:
                raise CheckpointIncompleteError(
                    f"main seed {seed} has not been committed"
                )
            result[scenario_id] = self._load_record_payload(record)
        return result

    def load_sensitivity_payload(
        self, parameter_id: str, level_id: str, seed: int
    ) -> object:
        unit = self._sensitivity_unit(parameter_id, level_id, seed)
        record = self._records_by_id().get(unit.unit_id)
        if record is None:
            raise CheckpointIncompleteError(
                "sensitivity work unit has not been committed"
            )
        return self._load_record_payload(record)

    def mark_interrupted(self, reason: str) -> None:
        self._mark_error(CheckpointStatus.INTERRUPTED, "INTERRUPTION", reason)

    def mark_failed(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception")
        self._mark_error(
            CheckpointStatus.FAILED,
            type(error).__name__,
            str(error) or type(error).__name__,
        )

    def mark_complete(self, output_checksums: Mapping[str, str]) -> None:
        """Record final artifact paths only after all declared work completed."""

        with self._lock:
            if len(self._completed_ids()) != self.work_plan.total_units:
                raise CheckpointIncompleteError(
                    "cannot complete while declared work units remain"
                )
            if not isinstance(output_checksums, Mapping) or not output_checksums:
                raise ValueError("at least one final output checksum is required")
            outputs = []
            for path, digest in sorted(output_checksums.items()):
                outputs.append(
                    {
                        "path": _relative_path(path, name="output path"),
                        "sha256": _digest(digest, name="output checksum"),
                    }
                )
            candidate = deepcopy(self._state)
            candidate["final_outputs"] = outputs
            candidate["status"] = CheckpointStatus.COMPLETE.value
            candidate["phase"] = ExecutionPhase.COMPLETE.value
            candidate["current_work_unit"] = None
            candidate["in_progress_unit_ids"] = []
            candidate["error"] = None
            self._commit_state(candidate)

    def _begin_units(
        self, unit_ids: Sequence[str], descriptor: Mapping[str, object]
    ) -> None:
        with self._lock:
            completed = self._completed_ids()
            active = set(_list(self._state, "in_progress_unit_ids"))
            duplicate = [
                unit_id
                for unit_id in unit_ids
                if unit_id in completed or unit_id in active
            ]
            if duplicate:
                raise DuplicateWorkUnitError(
                    f"work unit is already completed or in flight: {duplicate[0]}"
                )
            if self.status is CheckpointStatus.COMPLETE:
                raise CheckpointIncompleteError("completed checkpoint is immutable")
            candidate = deepcopy(self._state)
            candidate["in_progress_unit_ids"] = [
                *_list(candidate, "in_progress_unit_ids"),
                *unit_ids,
            ]
            candidate["current_work_unit"] = _json_copy(descriptor)
            candidate["status"] = CheckpointStatus.RUNNING.value
            candidate["phase"] = descriptor["phase"]
            candidate["error"] = None
            self._commit_state(candidate)

    def _commit_block(
        self,
        *,
        phase: ExecutionPhase,
        block_key: Mapping[str, object],
        entries: Sequence[dict[str, object]],
    ) -> None:
        with self._lock:
            completed = self._completed_ids()
            unit_ids = [str(entry["unit_id"]) for entry in entries]
            duplicate = [unit_id for unit_id in unit_ids if unit_id in completed]
            if duplicate:
                raise DuplicateWorkUnitError(
                    f"work unit is already committed: {duplicate[0]}"
                )
            if self.status is CheckpointStatus.COMPLETE:
                raise CheckpointIncompleteError("completed checkpoint is immutable")
            timestamp = self._now()
            core: dict[str, object] = {
                "schema_id": CHECKPOINT_SCHEMA_ID,
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "canonicalization": CHECKPOINT_CANONICALIZATION,
                "execution_identity_sha256": self.identity.identity_sha256,
                "work_plan_sha256": self.work_plan.identity_sha256,
                "payload_schema_id": self.identity.payload_schema_id,
                "phase": phase.value,
                "block_key": _json_copy(block_key),
                "completed_at": timestamp,
                "entries": list(entries),
            }
            block_identity = canonical_sha256(core)
            envelope = dict(core)
            envelope["block_sha256"] = block_identity
            suffix = block_identity
            if phase is ExecutionPhase.MAIN_BATCH:
                seed = _integer(block_key.get("seed"), name="block seed")
                filename = f"seed-{seed}-{suffix}.json"
            else:
                filename = f"unit-{suffix}.json"
            block_path = self.attempt_dir / "units" / phase.value / filename
            _atomic_write_json(block_path, envelope, replace=False)
            relative = block_path.relative_to(self.attempt_dir).as_posix()
            file_digest = _file_sha256(block_path)
            new_records = []
            for entry in entries:
                new_records.append(
                    {
                        "unit_id": entry["unit_id"],
                        "descriptor": deepcopy(entry["descriptor"]),
                        "payload_sha256": entry["payload_sha256"],
                        "block_path": relative,
                        "block_sha256": block_identity,
                        "block_file_sha256": file_digest,
                    }
                )
            candidate = deepcopy(self._state)
            records = _list(candidate, "completed_units")
            records.extend(new_records)
            order = self._unit_order()
            records.sort(key=lambda record: order[str(record["unit_id"])])
            committed_ids = set(unit_ids)
            candidate["in_progress_unit_ids"] = [
                unit_id
                for unit_id in _list(candidate, "in_progress_unit_ids")
                if unit_id not in committed_ids
            ]
            active = _list(candidate, "in_progress_unit_ids")
            candidate["current_work_unit"] = (
                self._descriptor_for_unit(str(active[-1])) if active else None
            )
            candidate["error"] = None
            self._set_phase_and_running_status(candidate)
            self._commit_state(candidate)

    def _mark_error(
        self, status: CheckpointStatus, error_type: str, message: str
    ) -> None:
        _text(message, name="error message")
        with self._lock:
            if self.status is CheckpointStatus.COMPLETE:
                raise CheckpointIncompleteError("completed checkpoint is immutable")
            candidate = deepcopy(self._state)
            candidate["status"] = status.value
            candidate["error"] = {
                "type": error_type,
                "message": message,
                "at": self._now(),
                "in_progress_unit_ids": list(
                    _list(candidate, "in_progress_unit_ids")
                ),
            }
            self._commit_state(candidate)

    def _set_phase_and_running_status(self, state: dict[str, object]) -> None:
        completed = {
            str(record["unit_id"])
            for record in _record_list(state, "completed_units")
        }
        main_ids = {
            self.work_plan.main_unit_id(seed, scenario_id)
            for seed in self.work_plan.seeds
            for scenario_id in self.work_plan.scenario_ids
        }
        sensitivity_ids = {
            unit.unit_id for unit in self.work_plan.sensitivity_units
        }
        active = _list(state, "in_progress_unit_ids")
        if active:
            descriptor = self._descriptor_for_unit(str(active[-1]))
            state["phase"] = descriptor["phase"]
            state["current_work_unit"] = descriptor
            state["status"] = CheckpointStatus.RUNNING.value
        elif not main_ids.issubset(completed):
            state["phase"] = ExecutionPhase.MAIN_BATCH.value
            state["current_work_unit"] = None
            state["status"] = CheckpointStatus.RUNNING.value
        elif not sensitivity_ids.issubset(completed):
            state["phase"] = ExecutionPhase.SENSITIVITY.value
            state["current_work_unit"] = None
            state["status"] = CheckpointStatus.RUNNING.value
        else:
            state["phase"] = ExecutionPhase.FINALIZATION.value
            state["current_work_unit"] = None
            state["status"] = (
                CheckpointStatus.COMPUTE_COMPLETE_EXPORT_PENDING.value
            )

    def _commit_state(
        self, state: dict[str, object], *, increment: bool = True
    ) -> None:
        candidate = deepcopy(state)
        if increment:
            candidate["generation"] = _integer(
                candidate.get("generation"), name="generation"
            ) + 1
        records = _record_list(candidate, "completed_units")
        completed_ids = {str(record["unit_id"]) for record in records}
        main_ids = {
            self.work_plan.main_unit_id(seed, scenario_id)
            for seed in self.work_plan.seeds
            for scenario_id in self.work_plan.scenario_ids
        }
        sensitivity_ids = {
            unit.unit_id for unit in self.work_plan.sensitivity_units
        }
        candidate["work_unit_counts"] = {
            "main_batch": {
                "expected": len(main_ids),
                "completed": len(completed_ids & main_ids),
                "remaining": len(main_ids - completed_ids),
            },
            "sensitivity": {
                "expected": len(sensitivity_ids),
                "completed": len(completed_ids & sensitivity_ids),
                "remaining": len(sensitivity_ids - completed_ids),
            },
            "overall": {
                "expected": self.work_plan.total_units,
                "completed": len(completed_ids),
                "remaining": self.work_plan.total_units - len(completed_ids),
            },
        }
        candidate["completed_seed_identifiers"] = [
            seed
            for seed in self.work_plan.seeds
            if all(
                unit_id in completed_ids
                for unit_id in self.work_plan.main_unit_ids(seed)
            )
        ]
        current = candidate.get("current_work_unit")
        if current is not None and type(current) is not dict:
            raise CheckpointCorruptError("current_work_unit must be an object")
        candidate["current_parameter"] = (
            current.get("parameter_id") if current is not None else None
        )
        candidate["current_level"] = (
            current.get("level_id") if current is not None else None
        )
        candidate["current_seed"] = (
            current.get("seed") if current is not None else None
        )
        candidate["updated_at"] = self._now()
        core = {key: value for key, value in candidate.items() if key != "checkpoint_sha256"}
        candidate["checkpoint_sha256"] = canonical_sha256(core)
        _atomic_write_json(self.checkpoint_path, candidate, replace=True)
        self._state = candidate
        self._write_progress()

    def _write_progress(self) -> None:
        _atomic_write_json(self.progress_path, self._build_progress(), replace=True)

    def _build_progress(self) -> dict[str, object]:
        completed = self._completed_ids()
        main_ids = {
            self.work_plan.main_unit_id(seed, scenario_id)
            for seed in self.work_plan.seeds
            for scenario_id in self.work_plan.scenario_ids
        }
        sensitivity_ids = {
            unit.unit_id for unit in self.work_plan.sensitivity_units
        }
        main_completed = len(completed & main_ids)
        sensitivity_completed = len(completed & sensitivity_ids)
        overall_completed = len(completed)
        current = self._state.get("current_work_unit")
        if current is not None and type(current) is not dict:
            raise CheckpointCorruptError("current_work_unit must be an object")
        records = _record_list(self._state, "completed_units")
        payload_files: dict[str, str] = {}
        for record in records:
            payload_files[str(record["block_path"])] = str(
                record["block_file_sha256"]
            )
        return {
            "schema_id": PROGRESS_SCHEMA_ID,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.identity.run_id,
            "attempt_id": self.identity.attempt_id,
            "status": self._state["status"],
            "current_phase": self._state["phase"],
            "current_parameter": (
                current.get("parameter_id") if current is not None else None
            ),
            "current_level": (
                current.get("level_id") if current is not None else None
            ),
            "current_seed": current.get("seed") if current is not None else None,
            "current_scenario": (
                current.get("scenario_id") if current is not None else None
            ),
            "overall": _progress_fraction(
                overall_completed, self.work_plan.total_units
            ),
            "main_batch": _progress_fraction(
                main_completed, self.work_plan.main_total_units
            ),
            "sensitivity": _progress_fraction(
                sensitivity_completed, self.work_plan.sensitivity_total_units
            ),
            "completed_main_seeds": list(self.completed_main_seeds),
            "remaining_main_seeds": list(self.remaining_main_seeds),
            "completed_sensitivity_units": [
                unit.snapshot() for unit in self.completed_sensitivity_units
            ],
            "remaining_sensitivity_unit_count": len(
                self.remaining_sensitivity_units
            ),
            "in_progress_unit_ids": list(
                _list(self._state, "in_progress_unit_ids")
            ),
            "execution_identity_sha256": self.identity.identity_sha256,
            "source_tree_sha256": self.identity.source_tree_sha256,
            "git_commit": self.identity.git_commit,
            "configuration_sha256": self.identity.configuration_sha256,
            "analysis_plan_sha256": self.identity.analysis_plan_sha256,
            "backend_identity_sha256": self.identity.backend.identity_sha256,
            "runtime_identity_sha256": self.identity.runtime.identity_sha256,
            "work_plan_sha256": self.work_plan.identity_sha256,
            "checkpoint": {
                "path": "checkpoint.json",
                "sha256": _file_sha256(self.checkpoint_path),
                "identity_sha256": self._state["checkpoint_sha256"],
                "generation": self._state["generation"],
            },
            "payload_blocks": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(payload_files.items())
            ],
            "output_paths": {
                "attempt_directory": self.attempt_dir.as_posix(),
                "checkpoint": "checkpoint.json",
                "progress": "progress.json",
                "payload_root": "units",
                "final_outputs": deepcopy(self._state["final_outputs"]),
            },
            "created_at": self._state["created_at"],
            "updated_at": self._state["updated_at"],
            "resume_count": self._state["resume_count"],
            "error": deepcopy(self._state.get("error")),
            "campaign_ready": False,
        }

    def _completed_ids(self) -> set[str]:
        return {
            str(record["unit_id"])
            for record in _record_list(self._state, "completed_units")
        }

    def _records_by_id(self) -> dict[str, dict[str, object]]:
        return {
            str(record["unit_id"]): record
            for record in _record_list(self._state, "completed_units")
        }

    def _unit_order(self) -> dict[str, int]:
        ordered = []
        for seed in self.work_plan.seeds:
            ordered.extend(self.work_plan.main_unit_ids(seed))
        ordered.extend(unit.unit_id for unit in self.work_plan.sensitivity_units)
        return {unit_id: index for index, unit_id in enumerate(ordered)}

    def _descriptor_for_unit(self, unit_id: str) -> dict[str, object]:
        for seed in self.work_plan.seeds:
            for scenario_id in self.work_plan.scenario_ids:
                if self.work_plan.main_unit_id(seed, scenario_id) == unit_id:
                    return self.work_plan.main_descriptor(seed, scenario_id)
        for unit in self.work_plan.sensitivity_units:
            if unit.unit_id == unit_id:
                return unit.snapshot()
        raise CheckpointCorruptError(f"unknown work unit {unit_id}")

    def _sensitivity_unit(
        self, parameter_id: str, level_id: str, seed: int
    ) -> SensitivityWorkUnit:
        matches = [
            unit
            for unit in self.work_plan.sensitivity_units
            if (
                unit.parameter_id == parameter_id
                and unit.level_id == level_id
                and unit.seed == seed
            )
        ]
        if len(matches) != 1:
            raise ValueError("sensitivity work unit is not declared")
        return matches[0]

    def _verify_all_payload_blocks(self) -> None:
        checked: set[str] = set()
        for record in _record_list(self._state, "completed_units"):
            path = str(record["block_path"])
            if path not in checked:
                self._verify_block_group(path)
                checked.add(path)

    def _verify_block_group(self, relative_path: str) -> dict[str, object]:
        records = [
            record
            for record in _record_list(self._state, "completed_units")
            if record["block_path"] == relative_path
        ]
        if not records:
            raise CheckpointCorruptError("payload block has no index records")
        block_path = _safe_child(self.attempt_dir, relative_path)
        expected_file_hashes = {str(record["block_file_sha256"]) for record in records}
        if len(expected_file_hashes) != 1 or _file_sha256(block_path) not in expected_file_hashes:
            raise CheckpointCorruptError("payload block file checksum mismatch")
        block = _read_json_object(block_path)
        claimed = block.get("block_sha256")
        _digest(claimed, name="block_sha256")
        core = {key: value for key, value in block.items() if key != "block_sha256"}
        if canonical_sha256(core) != claimed:
            raise CheckpointCorruptError("payload block identity mismatch")
        if claimed != records[0]["block_sha256"]:
            raise CheckpointCorruptError("checkpoint references another block identity")
        _assert_equal(
            block.get("execution_identity_sha256"),
            self.identity.identity_sha256,
            "payload execution identity",
        )
        _assert_equal(
            block.get("work_plan_sha256"),
            self.work_plan.identity_sha256,
            "payload work plan",
        )
        _assert_equal(
            block.get("payload_schema_id"),
            self.identity.payload_schema_id,
            "payload schema",
        )
        phases = {
            str(record["descriptor"].get("phase"))  # type: ignore[union-attr]
            for record in records
        }
        if len(phases) != 1 or block.get("phase") not in phases:
            raise CheckpointCorruptError("payload block phase is inconsistent")
        block_key = block.get("block_key")
        if type(block_key) is not dict:
            raise CheckpointCorruptError("payload block key must be an object")
        if block.get("phase") == ExecutionPhase.MAIN_BATCH.value:
            seeds = {
                record["descriptor"].get("seed")  # type: ignore[union-attr]
                for record in records
            }
            if len(seeds) != 1 or block_key != {"seed": next(iter(seeds))}:
                raise CheckpointCorruptError("main payload block key is inconsistent")
        else:
            descriptor = records[0]["descriptor"]
            if type(descriptor) is not dict:
                raise CheckpointCorruptError("payload descriptor must be an object")
            expected_key = {
                "parameter_id": descriptor.get("parameter_id"),
                "level_id": descriptor.get("level_id"),
                "seed": descriptor.get("seed"),
            }
            if block_key != expected_key:
                raise CheckpointCorruptError(
                    "sensitivity payload block key is inconsistent"
                )
        entries = block.get("entries")
        if type(entries) is not list:
            raise CheckpointCorruptError("payload entries must be an array")
        by_id: dict[str, dict[str, object]] = {}
        for entry in entries:
            if type(entry) is not dict:
                raise CheckpointCorruptError("payload entry must be an object")
            unit_id = entry.get("unit_id")
            if type(unit_id) is not str or unit_id in by_id:
                raise CheckpointCorruptError("payload unit IDs must be unique")
            if canonical_sha256(entry.get("payload")) != entry.get("payload_sha256"):
                raise CheckpointCorruptError("payload checksum mismatch")
            by_id[unit_id] = entry
        if set(by_id) != {str(record["unit_id"]) for record in records}:
            raise CheckpointCorruptError("block entries differ from checkpoint index")
        for record in records:
            entry = by_id[str(record["unit_id"])]
            if entry.get("payload_sha256") != record.get("payload_sha256"):
                raise CheckpointCorruptError("indexed payload checksum mismatch")
            if _canonical_json_bytes(entry.get("descriptor")) != _canonical_json_bytes(
                record.get("descriptor")
            ):
                raise CheckpointCorruptError("indexed work descriptor mismatch")
        return block

    def _load_record_payload(self, record: dict[str, object]) -> object:
        block = self._verify_block_group(str(record["block_path"]))
        for entry in block["entries"]:  # type: ignore[index]
            if entry["unit_id"] == record["unit_id"]:  # type: ignore[index]
                return _json_copy(entry["payload"])  # type: ignore[index]
        raise CheckpointCorruptError("payload entry disappeared")

    def _now(self) -> str:
        return _timestamp(self._clock())


def next_attempt_id(progress_root: str | Path) -> str:
    """Return the next monotonic ID without creating or modifying an attempt."""

    root = Path(progress_root)
    _ensure_real_directory(root, create=True)
    numbers = []
    for child in root.iterdir():
        if _ATTEMPT_ID.fullmatch(child.name):
            _ensure_real_directory(child, create=False)
            numbers.append(int(child.name.removeprefix("attempt-")))
    number = max(numbers, default=0) + 1
    if number > 999_999:
        raise RuntimeError("checkpoint attempt namespace exhausted")
    return f"attempt-{number:06d}"


def format_console_progress(progress: Mapping[str, object]) -> str:
    """Format progress from completed/declared units, never elapsed time."""

    overall = progress.get("overall")
    main = progress.get("main_batch")
    sensitivity = progress.get("sensitivity")
    if not all(type(value) is dict for value in (overall, main, sensitivity)):
        raise ValueError("progress payload lacks exact work-unit sections")
    current = [
        f"phase={progress.get('current_phase')}",
        f"seed={progress.get('current_seed')}",
    ]
    if progress.get("current_parameter") is not None:
        current.extend(
            (
                f"parameter={progress.get('current_parameter')}",
                f"level={progress.get('current_level')}",
            )
        )
    return (
        f"overall {overall['percentage_display']} "  # type: ignore[index]
        f"({overall['completed_units']}/{overall['total_units']}); "  # type: ignore[index]
        f"main {main['percentage_display']} "  # type: ignore[index]
        f"({main['completed_units']}/{main['total_units']}); "  # type: ignore[index]
        f"sensitivity {sensitivity['percentage_display']} "  # type: ignore[index]
        f"({sensitivity['completed_units']}/{sensitivity['total_units']}); "  # type: ignore[index]
        + " ".join(current)
    )


def _validate_identity_plan(
    identity: ExecutionIdentity, work_plan: ExecutionWorkPlan
) -> None:
    if identity.work_plan_sha256 != work_plan.identity_sha256:
        raise IncompatibleCheckpointError(
            "execution identity work-plan hash is incompatible"
        )
    if identity.seed_set_sha256 != work_plan.seed_set_sha256:
        raise IncompatibleCheckpointError(
            "execution identity seed-set hash is incompatible"
        )


def _validate_lineage(
    identity: ExecutionIdentity, lineage: ExecutionLineage | None
) -> None:
    if lineage is None:
        return
    if (
        lineage.successor_run_id != identity.run_id
        or lineage.successor_attempt_id != identity.attempt_id
        or lineage.successor_implementation_id != identity.implementation_id
    ):
        raise IncompatibleCheckpointError(
            "execution lineage does not identify this successor"
        )
    if lineage.previous.analysis_plan_sha256 != identity.analysis_plan_sha256:
        raise IncompatibleCheckpointError(
            "execution-only supersession must preserve the scientific plan hash"
        )


def _validate_state_structure(
    state: dict[str, object], work_plan: ExecutionWorkPlan
) -> None:
    for name in ("created_at", "updated_at"):
        _parse_timestamp(state.get(name), name=name)
    _integer(state.get("generation"), name="generation", allow_zero=True)
    _integer(state.get("resume_count"), name="resume_count", allow_zero=True)
    try:
        status = CheckpointStatus(str(state.get("status")))
        ExecutionPhase(str(state.get("phase")))
    except ValueError as exc:
        raise CheckpointCorruptError("checkpoint status or phase is invalid") from exc
    expected = state.get("expected_work_units")
    if expected != {
        "main_batch": work_plan.main_total_units,
        "sensitivity": work_plan.sensitivity_total_units,
        "overall": work_plan.total_units,
    }:
        raise IncompatibleCheckpointError("expected work-unit counts changed")
    records = _record_list(state, "completed_units")
    active = _list(state, "in_progress_unit_ids")
    if any(type(unit_id) is not str for unit_id in active) or len(active) != len(
        set(active)
    ):
        raise CheckpointCorruptError("in-progress unit IDs must be unique text")

    descriptors: dict[str, dict[str, object]] = {}
    main_by_seed: dict[int, set[str]] = {seed: set() for seed in work_plan.seeds}
    for seed in work_plan.seeds:
        for scenario_id in work_plan.scenario_ids:
            descriptor = work_plan.main_descriptor(seed, scenario_id)
            unit_id = work_plan.main_unit_id(seed, scenario_id)
            descriptors[unit_id] = descriptor
            main_by_seed[seed].add(unit_id)
    for unit in work_plan.sensitivity_units:
        descriptors[unit.unit_id] = unit.snapshot()

    observed: set[str] = set()
    for index, record in enumerate(records):
        unit_id = record.get("unit_id")
        if type(unit_id) is not str or unit_id in observed:
            raise CheckpointCorruptError("completed work-unit IDs must be unique")
        if unit_id not in descriptors:
            raise IncompatibleCheckpointError(
                f"checkpoint contains undeclared work unit {unit_id}"
            )
        observed.add(unit_id)
        if _canonical_json_bytes(record.get("descriptor")) != _canonical_json_bytes(
            descriptors[unit_id]
        ):
            raise IncompatibleCheckpointError(
                f"work descriptor changed for completed unit {unit_id}"
            )
        for name in ("payload_sha256", "block_sha256", "block_file_sha256"):
            try:
                _digest(record.get(name), name=f"completed_units[{index}].{name}")
            except ValueError as exc:
                raise CheckpointCorruptError(str(exc)) from exc
        try:
            _relative_path(
                record.get("block_path"),
                name=f"completed_units[{index}].block_path",
            )
        except ValueError as exc:
            raise CheckpointCorruptError(str(exc)) from exc
    unknown_active = set(active) - set(descriptors)
    if unknown_active:
        raise CheckpointCorruptError("checkpoint has undeclared in-progress units")
    if observed & set(active):
        raise CheckpointCorruptError("a completed unit cannot remain in progress")

    # Main results are committed as one complete seed block: a partial seed is
    # evidence of a corrupt or non-conforming writer and is never resumed.
    for seed, seed_ids in main_by_seed.items():
        count = len(observed & seed_ids)
        if count not in {0, len(seed_ids)}:
            raise CheckpointCorruptError(
                f"main seed {seed} was not committed as one atomic block"
            )
        if count:
            block_paths = {
                str(record["block_path"])
                for record in records
                if str(record["unit_id"]) in seed_ids
            }
            if len(block_paths) != 1:
                raise CheckpointCorruptError(
                    f"main seed {seed} spans more than one payload block"
                )
    sensitivity_ids = {unit.unit_id for unit in work_plan.sensitivity_units}
    sensitivity_block_counts: dict[str, int] = {}
    for record in records:
        if str(record["unit_id"]) in sensitivity_ids:
            path = str(record["block_path"])
            sensitivity_block_counts[path] = sensitivity_block_counts.get(path, 0) + 1
    if any(count != 1 for count in sensitivity_block_counts.values()):
        raise CheckpointCorruptError(
            "sensitivity payload blocks must contain one declared unit"
        )
    if len(observed) > work_plan.total_units:
        raise CheckpointCorruptError("completed-unit count exceeds work plan")
    expected_counts = {
        "main_batch": {
            "expected": work_plan.main_total_units,
            "completed": len(
                observed
                & {
                    work_plan.main_unit_id(seed, scenario_id)
                    for seed in work_plan.seeds
                    for scenario_id in work_plan.scenario_ids
                }
            ),
            "remaining": work_plan.main_total_units
            - len(
                observed
                & {
                    work_plan.main_unit_id(seed, scenario_id)
                    for seed in work_plan.seeds
                    for scenario_id in work_plan.scenario_ids
                }
            ),
        },
        "sensitivity": {
            "expected": work_plan.sensitivity_total_units,
            "completed": len(
                observed
                & {unit.unit_id for unit in work_plan.sensitivity_units}
            ),
            "remaining": work_plan.sensitivity_total_units
            - len(
                observed
                & {unit.unit_id for unit in work_plan.sensitivity_units}
            ),
        },
        "overall": {
            "expected": work_plan.total_units,
            "completed": len(observed),
            "remaining": work_plan.total_units - len(observed),
        },
    }
    if state.get("work_unit_counts") != expected_counts:
        raise CheckpointCorruptError("checkpoint work-unit counts are inconsistent")
    expected_completed_seeds = [
        seed
        for seed, seed_ids in main_by_seed.items()
        if seed_ids.issubset(observed)
    ]
    if state.get("completed_seed_identifiers") != expected_completed_seeds:
        raise CheckpointCorruptError(
            "checkpoint completed seed identifiers are inconsistent"
        )
    current = state.get("current_work_unit")
    if current is not None and type(current) is not dict:
        raise CheckpointCorruptError("current_work_unit must be an object")
    expected_current_fields = {
        "current_parameter": (
            current.get("parameter_id") if current is not None else None
        ),
        "current_level": current.get("level_id") if current is not None else None,
        "current_seed": current.get("seed") if current is not None else None,
    }
    for name, value in expected_current_fields.items():
        if state.get(name) != value:
            raise CheckpointCorruptError(f"{name} is inconsistent")
    if state.get("output_paths") != {
        "checkpoint": "checkpoint.json",
        "progress": "progress.json",
        "payload_root": "units",
    }:
        raise CheckpointCorruptError("checkpoint output paths are inconsistent")
    if status is CheckpointStatus.COMPLETE:
        if len(observed) != work_plan.total_units:
            raise CheckpointCorruptError("complete checkpoint has missing work")
        if not _list(state, "final_outputs"):
            raise CheckpointCorruptError("complete checkpoint lacks final outputs")
    final_outputs = _list(state, "final_outputs")
    for output in final_outputs:
        if type(output) is not dict:
            raise CheckpointCorruptError("final output entry must be an object")
        try:
            _relative_path(output.get("path"), name="final output path")
            _digest(output.get("sha256"), name="final output checksum")
        except ValueError as exc:
            raise CheckpointCorruptError(str(exc)) from exc
    events = _list(state, "events")
    if any(type(event) is not dict for event in events):
        raise CheckpointCorruptError("checkpoint events must be objects")


def _verify_state_digest(state: dict[str, object]) -> None:
    claimed = state.get("checkpoint_sha256")
    try:
        _digest(claimed, name="checkpoint_sha256")
    except ValueError as exc:
        raise CheckpointCorruptError(str(exc)) from exc
    core = {key: value for key, value in state.items() if key != "checkpoint_sha256"}
    if canonical_sha256(core) != claimed:
        raise CheckpointCorruptError("checkpoint identity hash mismatch")


def _verify_immutable_file(
    attempt_dir: Path,
    reference: object,
    expected_payload: object,
    *,
    name: str,
) -> None:
    if type(reference) is not dict:
        raise CheckpointCorruptError(f"{name} file reference is missing")
    try:
        relative = _relative_path(reference.get("path"), name=f"{name} path")
        expected_sha = _digest(
            reference.get("sha256"), name=f"{name} file checksum"
        )
    except ValueError as exc:
        raise CheckpointCorruptError(str(exc)) from exc
    path = _safe_child(attempt_dir, relative)
    if _file_sha256(path) != expected_sha:
        raise CheckpointCorruptError(f"{name} file checksum mismatch")
    observed = _read_json_object(path)
    if _canonical_json_bytes(observed) != _canonical_json_bytes(expected_payload):
        raise IncompatibleCheckpointError(f"{name} payload changed")


def _payload_entry(
    *, unit_id: str, descriptor: Mapping[str, object], payload: object
) -> dict[str, object]:
    detached = _json_copy(payload)
    return {
        "unit_id": unit_id,
        "descriptor": _json_copy(descriptor),
        "payload": detached,
        "payload_sha256": canonical_sha256(detached),
    }


def _progress_fraction(completed: int, total: int) -> dict[str, object]:
    if type(completed) is not int or type(total) is not int:
        raise TypeError("progress counts must be integers")
    if completed < 0 or total < 0 or completed > total:
        raise ValueError("progress counts are outside the declared denominator")
    remaining = total - completed
    if total == 0:
        numerator, denominator = 100, 1
        percentage = 100.0
    else:
        raw_numerator = completed * 100
        divisor = math.gcd(raw_numerator, total)
        numerator = raw_numerator // divisor
        denominator = total // divisor
        percentage = raw_numerator / total
    return {
        "completed_units": completed,
        "total_units": total,
        "remaining_units": remaining,
        "percentage": percentage,
        "percentage_display": f"{percentage:.6f}%",
        "percentage_exact": {
            "numerator": numerator,
            "denominator": denominator,
            "unit": "percent",
        },
        "basis": "COMPLETED_DECLARED_WORK_UNITS",
    }


def _file_reference(root: Path, path: Path, *, role: str) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _file_sha256(path),
        "role": role,
    }


def _atomic_write_json(path: Path, payload: object, *, replace: bool) -> None:
    encoded = _render_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_real_directory(path.parent, create=False)
    if not replace and path.exists():
        raise FileExistsError(f"immutable checkpoint file exists: {path.name}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and path.exists():
            raise FileExistsError(f"immutable checkpoint file exists: {path.name}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _render_json_bytes(value: object) -> bytes:
    _validate_json(value, path="$")
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_json_bytes(value: object) -> bytes:
    _validate_json(value, path="$")
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_json(value: object, *, path: str) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ValueError(f"{path} has a non-string or empty key")
            _validate_json(item, path=f"{path}.{key}")
        return
    if type(value) in {list, tuple}:
        for index, item in enumerate(value):
            _validate_json(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported {type(value).__name__}")


def _json_copy(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _read_json_object(path: Path) -> dict[str, object]:
    _ensure_real_file(path)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")

        def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise CheckpointCorruptError(
                        f"duplicate JSON key {key!r} in {path.name}"
                    )
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise CheckpointCorruptError(
                f"non-finite JSON constant {value} in {path.name}"
            )

        payload = json.loads(
            text,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except CheckpointCorruptError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptError(
            f"checkpoint JSON is unreadable: {path.name}"
        ) from exc
    if type(payload) is not dict:
        raise CheckpointCorruptError(f"{path.name} must contain a JSON object")
    try:
        _validate_json(payload, path="$" )
    except ValueError as exc:
        raise CheckpointCorruptError(str(exc)) from exc
    return payload


def _file_sha256(path: Path) -> str:
    _ensure_real_file(path)
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child(root: Path, relative: str) -> Path:
    normalized = _relative_path(relative, name="checkpoint relative path")
    path = root.joinpath(*PurePosixPath(normalized).parts)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise CheckpointCorruptError("checkpoint path escapes attempt directory") from exc
    _ensure_real_file(path)
    return path


def _ensure_real_directory(path: Path, *, create: bool) -> None:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    try:
        info = path.lstat()
    except OSError as exc:
        raise CheckpointError(f"checkpoint directory is unavailable: {path}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or bool(getattr(info, "st_file_attributes", 0) & reparse)
    ):
        raise CheckpointError("checkpoint directories must not be links/reparse points")


def _ensure_real_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CheckpointCorruptError(f"checkpoint file is missing: {path.name}") from exc
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or bool(getattr(info, "st_file_attributes", 0) & reparse)
    ):
        raise CheckpointCorruptError("checkpoint files must be real regular files")


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        # Windows does not expose portable directory fsync.  The file itself
        # was fsynced before atomic replacement, which is the strongest
        # portable guarantee available through Python's standard library.
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty text without outer whitespace")
    if "\x00" in value:
        raise ValueError(f"{name} cannot contain NUL")
    return value


def _seed(value: object) -> int:
    if type(value) is not int or value < 0 or value >= 2**63:
        raise ValueError("seed must be a non-negative signed 64-bit integer")
    return value


def _integer(value: object, *, name: str, allow_zero: bool = True) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise CheckpointCorruptError(f"{name} must be a valid integer")
    return value


def _normalized_path(
    value: object, *, name: str, require_absolute: bool = True
) -> str:
    text = _text(value, name=name).replace("\\", "/")
    if "//" in text or "/./" in f"/{text}/" or "/../" in f"/{text}/":
        raise ValueError(f"{name} must be a normalized path")
    if require_absolute and not (
        text.startswith("/") or re.match(r"[A-Za-z]:/", text)
    ):
        raise ValueError(f"{name} must be an absolute path")
    return text


def _relative_path(value: object, *, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a relative POSIX path")
    rendered = _normalized_path(value, name=name, require_absolute=False)
    pure = PurePosixPath(rendered)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"{name} must stay within its declared root")
    if ":" in pure.parts[0]:
        raise ValueError(f"{name} cannot contain a drive prefix")
    return pure.as_posix()


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("checkpoint clock must return an aware datetime")
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _parse_timestamp(value: object, *, name: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise CheckpointCorruptError(f"{name} must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise CheckpointCorruptError(f"{name} is invalid") from exc
    if _timestamp(parsed) != value:
        raise CheckpointCorruptError(f"{name} is not canonical")
    return parsed


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _record_list(
    mapping: Mapping[str, object], name: str
) -> list[dict[str, object]]:
    value = mapping.get(name)
    if type(value) is not list or any(type(item) is not dict for item in value):
        raise CheckpointCorruptError(f"{name} must be an array of objects")
    return value  # type: ignore[return-value]


def _list(mapping: Mapping[str, object], name: str) -> list[object]:
    value = mapping.get(name)
    if type(value) is not list:
        raise CheckpointCorruptError(f"{name} must be an array")
    return value


def _assert_equal(observed: object, expected: object, name: str) -> None:
    if observed != expected:
        raise IncompatibleCheckpointError(f"{name} does not match")


__all__ = [
    "CHECKPOINT_CANONICALIZATION",
    "CHECKPOINT_SCHEMA_ID",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROGRESS_SCHEMA_ID",
    "BackendIdentity",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointIncompleteError",
    "CheckpointStatus",
    "DuplicateWorkUnitError",
    "ExecutionIdentity",
    "ExecutionLineage",
    "ExecutionPhase",
    "ExecutionWorkPlan",
    "IncompatibleCheckpointError",
    "PriorExecutionLineage",
    "ResumableCheckpointStore",
    "RuntimeIdentity",
    "SensitivityWorkUnit",
    "canonical_sha256",
    "canonical_level_id",
    "format_console_progress",
    "next_attempt_id",
]
