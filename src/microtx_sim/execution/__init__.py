"""Explicit, attestable execution backends for numerical kernels.

The package deliberately has no dependency on campaign orchestration.  A caller
must resolve and record a backend before it starts work; an explicitly requested
GPU is never replaced by a CPU implementation.
"""

from .backends import (
    BackendMode,
    BackendProbe,
    BackendUnavailableError,
    ExecutionBackendConfig,
    ExecutionBackendMetadata,
    ResolvedExecutionBackend,
    resolve_execution_backend,
    probe_gpu_backend,
)
from .kernels import (
    CompositeHarmParityReport,
    NumericalParityError,
    NumericalParityTolerance,
    compute_composite_harm,
    validate_composite_harm_parity,
)
from .checkpoints import (
    CHECKPOINT_SCHEMA_ID,
    CHECKPOINT_SCHEMA_VERSION,
    PROGRESS_SCHEMA_ID,
    BackendIdentity,
    CheckpointCorruptError,
    CheckpointError,
    CheckpointIncompleteError,
    CheckpointStatus,
    DuplicateWorkUnitError,
    ExecutionIdentity,
    ExecutionLineage,
    ExecutionPhase,
    ExecutionWorkPlan,
    IncompatibleCheckpointError,
    PriorExecutionLineage,
    ResumableCheckpointStore,
    RuntimeIdentity,
    SensitivityWorkUnit,
    canonical_level_id,
    format_console_progress,
    next_attempt_id,
)
from .native_threads import (
    NativeThreadAttestation,
    NativeThreadControlError,
    enforce_numpy_native_thread_limit,
)

__all__ = [
    "CHECKPOINT_SCHEMA_ID",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROGRESS_SCHEMA_ID",
    "BackendIdentity",
    "BackendMode",
    "BackendProbe",
    "BackendUnavailableError",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointIncompleteError",
    "CheckpointStatus",
    "CompositeHarmParityReport",
    "ExecutionBackendConfig",
    "ExecutionBackendMetadata",
    "ExecutionIdentity",
    "ExecutionLineage",
    "ExecutionPhase",
    "ExecutionWorkPlan",
    "DuplicateWorkUnitError",
    "IncompatibleCheckpointError",
    "NumericalParityError",
    "NumericalParityTolerance",
    "NativeThreadAttestation",
    "NativeThreadControlError",
    "PriorExecutionLineage",
    "ResumableCheckpointStore",
    "ResolvedExecutionBackend",
    "RuntimeIdentity",
    "SensitivityWorkUnit",
    "canonical_level_id",
    "compute_composite_harm",
    "enforce_numpy_native_thread_limit",
    "format_console_progress",
    "next_attempt_id",
    "probe_gpu_backend",
    "resolve_execution_backend",
    "validate_composite_harm_parity",
]
