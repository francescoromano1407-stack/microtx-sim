"""Build and verify an exact configured execution/checkpoint session."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from ..execution_attestation import (
    _dependency_environment_identity,
    _operating_system_identity,
    _python_runtime_identity,
    _repository_identity,
    _source_tree_identity,
)
from ..policy_config import PolicyPrototypeConfig, PolicyRunPurpose
from .backends import (
    ExecutionBackendConfig,
    ResolvedExecutionBackend,
    resolve_execution_backend,
)
from .checkpoints import (
    BackendIdentity,
    ExecutionIdentity,
    ExecutionLineage,
    ExecutionWorkPlan,
    PriorExecutionLineage,
    ResumableCheckpointStore,
    RuntimeIdentity,
)
from .kernels import CompositeHarmParityReport, validate_composite_harm_parity
from .native_threads import (
    NativeThreadAttestation,
    enforce_numpy_native_thread_limit,
)


def resolve_configured_backend(
    config: PolicyPrototypeConfig,
) -> ResolvedExecutionBackend:
    engine = config.execution_engine
    if config.run_purpose is not PolicyRunPurpose.EXPLORATORY or engine is None:
        raise ValueError("execution backend requires the exploratory contract")
    return resolve_execution_backend(
        ExecutionBackendConfig(
            mode=engine.backend,
            device_index=engine.gpu_device_index,
            batch_size=engine.gpu_batch_size,
            max_batch_bytes=engine.gpu_max_batch_bytes,
            gpu_memory_fraction=engine.gpu_memory_fraction,
            precision_mode=engine.precision_mode,
        )
    )


def preflight_backend_parity(
    config: PolicyPrototypeConfig,
    backend: ResolvedExecutionBackend,
) -> CompositeHarmParityReport:
    """Require bounded deterministic numerical parity before dispatch."""

    scores = np.linspace(
        0.0,
        1.0,
        num=6 * 257,
        dtype=np.float64,
    ).reshape(257, 6)
    weights = config.harm_weights.as_array()
    report = validate_composite_harm_parity(
        scores,
        weights,
        backend=backend,
        categorical_thresholds=(0.35,),
        mean_direction_reference=0.35,
        raise_on_failure=True,
    )
    if backend.metadata.resolved_mode == "cpu" and not report.bitwise_equal:
        raise RuntimeError("explicit CPU backend is not bitwise equivalent")
    return report


def build_execution_identity(
    config: PolicyPrototypeConfig,
    *,
    config_path: str | Path,
    work_plan: ExecutionWorkPlan,
    backend: ResolvedExecutionBackend,
    repository_root: str | Path,
) -> ExecutionIdentity:
    """Attest clean source, runtime, configuration, plan, and scheduler."""

    if config.run_purpose is not PolicyRunPurpose.EXPLORATORY:
        raise ValueError("optimized identity is exploratory-only")
    engine = config.execution_engine
    exploratory = config.exploratory
    if engine is None or exploratory is None:
        raise ValueError("optimized identity requires complete exploratory config")
    root = Path(repository_root).resolve(strict=True)
    repository = _repository_identity(root)
    source = _source_tree_identity(root)
    runtime = _runtime_identity(root)
    worker_count = _effective_worker_count(config, backend)
    native_threads = enforce_numpy_native_thread_limit(1)
    backend_identity = checkpoint_backend_identity(
        config,
        backend,
        worker_count=worker_count,
        native_threads=native_threads,
    )
    selected_config = Path(config_path).resolve(strict=True)
    return ExecutionIdentity(
        run_id=engine.run_id,
        attempt_id=engine.attempt_id,
        implementation_id=engine.implementation_id,
        source_tree_sha256=str(source["source_tree_sha256"]),
        git_commit=str(repository["commit"]),
        git_branch=str(repository["branch"]),
        configuration_sha256=_file_sha256(selected_config),
        analysis_plan_id=exploratory.exploratory_plan_id,
        analysis_plan_sha256=exploratory.exploratory_plan_sha256,
        seed_set_sha256=work_plan.seed_set_sha256,
        work_plan_sha256=work_plan.identity_sha256,
        backend=backend_identity,
        runtime=runtime,
        payload_schema_id=(
            "microtx_sim.policy_and_sensitivity_checkpoint_payload.v2"
        ),
    )


def checkpoint_backend_identity(
    config: PolicyPrototypeConfig,
    backend: ResolvedExecutionBackend,
    *,
    worker_count: int,
    native_threads: NativeThreadAttestation | None = None,
) -> BackendIdentity:
    engine = config.execution_engine
    if engine is None:
        raise ValueError("backend identity requires execution_engine config")
    metadata = backend.metadata
    is_gpu = metadata.resolved_mode == "gpu"
    attestation = (
        native_threads
        if native_threads is not None
        else enforce_numpy_native_thread_limit(1)
    )
    return BackendIdentity(
        requested_backend=metadata.requested_mode,
        resolved_backend=metadata.resolved_mode,
        library=metadata.implementation,
        library_version=metadata.implementation_version,
        device_name=metadata.device_name,
        device_id=(
            f"cuda:{metadata.device_index}" if is_gpu else "cpu"
        ),
        compute_capability=(metadata.compute_capability if is_gpu else None),
        driver_version=(
            str(metadata.driver_version) if is_gpu else None
        ),
        runtime_version=(
            str(metadata.runtime_version) if is_gpu else None
        ),
        precision_mode=metadata.precision_mode,
        worker_count=worker_count,
        batch_size=metadata.batch_size,
        scheduling_policy=engine.scheduling_policy,
        native_thread_runtime=attestation.runtime,
        native_thread_library_path=attestation.library_path,
        native_thread_library_sha256=attestation.library_sha256,
        native_thread_getter_symbol=attestation.getter_symbol,
        native_thread_setter_symbol=attestation.setter_symbol,
        native_thread_limit=attestation.enforced_thread_count,
    )


def build_previous_execution_lineage(
    config: PolicyPrototypeConfig,
    *,
    repository_root: str | Path,
) -> ExecutionLineage:
    """Bind exact observed v1 facts without inventing absent source identity."""

    engine = config.execution_engine
    if engine is None:
        raise ValueError("execution lineage requires execution_engine config")
    lineage_path = engine.previous_attempt_lineage_path.resolve(strict=True)
    raw = json.loads(lineage_path.read_text("utf-8"))
    previous = raw.get("previous_execution")
    successor = raw.get("superseding_execution")
    if type(previous) is not dict or type(successor) is not dict:
        raise ValueError("previous execution lineage artifact is malformed")
    root = Path(repository_root).resolve(strict=True)
    progress_relative = (
        "artifacts/policy_exploratory_synthetic/progress/"
        "attempt-000001/progress.json"
    )
    progress_path = root.joinpath(*progress_relative.split("/"))
    progress = json.loads(progress_path.read_text("utf-8"))
    expected = {
        "attempt_id": engine.supersedes_attempt_id,
        "status": "INTERRUPTED",
        "config_sha256": previous.get("configuration_sha256"),
        "plan_sha256": previous.get("analysis_plan_sha256"),
    }
    observed = {
        "attempt_id": progress.get("attempt_id"),
        "status": progress.get("status"),
        "config_sha256": progress.get("config_sha256"),
        "plan_sha256": progress.get("exploratory_plan_sha256"),
    }
    if observed != expected:
        raise ValueError("observed v1 progress differs from checked-in lineage")
    if previous.get("git_commit") is not None or previous.get(
        "source_tree_sha256"
    ) is not None:
        raise ValueError("v1 lineage must not retroactively claim source identity")
    if successor.get("attempt_id") != engine.attempt_id or successor.get(
        "run_id"
    ) != engine.run_id:
        raise ValueError("lineage successor differs from execution configuration")
    prior = PriorExecutionLineage(
        run_id=None,
        run_identity_status="NOT_RECORDED_BY_CHECKPOINT_SCHEMA_V1",
        attempt_id=engine.supersedes_attempt_id,
        configuration_sha256=str(previous["configuration_sha256"]),
        analysis_plan_sha256=str(previous["analysis_plan_sha256"]),
        source_identity_status="NOT_RECORDED_BY_CHECKPOINT_SCHEMA_V1",
        source_tree_sha256=None,
        git_commit=None,
        observed_status="INTERRUPTED",
        reason_final_outputs_unavailable=(
            "Specific interruption reason was not recorded by checkpoint "
            "schema 1.0; no final campaign outputs were produced."
        ),
        progress_artifact_path=progress_relative,
        progress_artifact_sha256=_file_sha256(progress_path),
    )
    return ExecutionLineage(
        previous=prior,
        successor_run_id=engine.run_id,
        successor_attempt_id=engine.attempt_id,
        successor_implementation_id=engine.implementation_id,
    )


def open_configured_checkpoint(
    config: PolicyPrototypeConfig,
    *,
    identity: ExecutionIdentity,
    work_plan: ExecutionWorkPlan,
    lineage: ExecutionLineage,
) -> ResumableCheckpointStore:
    checkpoint = config.exploratory_checkpoint
    engine = config.execution_engine
    if checkpoint is None or engine is None:
        raise ValueError("configured checkpoint requires exploratory execution")
    if not checkpoint.enabled or not engine.resume_enabled:
        raise ValueError("resumable checkpointing must remain enabled")
    attempt_dir = checkpoint.directory / engine.attempt_id
    if attempt_dir.exists():
        return ResumableCheckpointStore.resume(
            attempt_dir,
            expected_identity=identity,
            expected_work_plan=work_plan,
            expected_lineage=lineage,
        )
    return ResumableCheckpointStore.create(
        checkpoint.directory,
        identity=identity,
        work_plan=work_plan,
        lineage=lineage,
    )


def _runtime_identity(repository_root: Path) -> RuntimeIdentity:
    runtime = _python_runtime_identity()
    operating_system = _operating_system_identity()
    dependencies = _dependency_environment_identity()
    lock_path = repository_root / "uv.lock"
    return RuntimeIdentity(
        python_implementation=str(runtime["implementation"]),
        python_version=str(runtime["version"]),
        python_executable=str(runtime["executable_path"]),
        python_executable_sha256=str(runtime["executable_sha256"]),
        dependency_lock_sha256=_file_sha256(lock_path),
        installed_dependencies_sha256=str(
            dependencies["dependency_set_sha256"]
        ),
        operating_system=str(operating_system["system"]),
        os_release=str(operating_system["release"]),
        machine_architecture=str(operating_system["machine"]),
        processor=str(operating_system["processor"]),
    )


def _effective_worker_count(
    config: PolicyPrototypeConfig,
    backend: ResolvedExecutionBackend,
) -> int:
    engine = config.execution_engine
    if engine is None:
        raise ValueError("worker count requires execution_engine config")
    if backend.metadata.resolved_mode == "gpu":
        return 1
    memory_bound = engine.memory_limit_mb // engine.estimated_worker_memory_mb
    return max(1, min(engine.host_workers, memory_bound))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "build_execution_identity",
    "build_previous_execution_lineage",
    "checkpoint_backend_identity",
    "open_configured_checkpoint",
    "preflight_backend_parity",
    "resolve_configured_backend",
]
