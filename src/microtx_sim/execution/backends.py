"""Backend selection, device identity, and bounded array-batch planning.

CuPy is an optional execution dependency because it exposes NumPy-compatible
arrays while supporting fused CUDA kernels.  Importing this module never imports
CuPy.  ``gpu`` requests fail closed when CuPy, its CUDA runtime, or a compatible
device is unavailable.  ``auto`` is an explicit selection policy, not a GPU
request with an implicit fallback: its resolved backend and reason are included
in the backend identity.

Only float64 is admitted for model-facing floating-point kernels.  TF32 and
mixed precision are intentionally absent from the contract.  Integer counter
RNG kernels are required to be bitwise identical to the CPU reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import importlib
import json
from numbers import Integral, Real
import platform
from types import MappingProxyType, ModuleType
from typing import Any, Iterator, Mapping

import numpy as np


_BACKEND_SCHEMA = "microtx.execution-backend.v1"
_PRECISION_MODE = "FLOAT64_STRICT_INTEGER_EXACT"


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly selected execution backend cannot run."""


class BackendMode(str, Enum):
    """User-facing backend selection modes."""

    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


@dataclass(frozen=True, slots=True)
class ExecutionBackendConfig:
    """Strict resource and precision contract for numerical execution."""

    mode: BackendMode | str = BackendMode.CPU
    device_index: int = 0
    batch_size: int = 65_536
    max_batch_bytes: int = 256 * 1024 * 1024
    gpu_memory_fraction: float = 0.50
    precision_mode: str = _PRECISION_MODE

    def __post_init__(self) -> None:
        try:
            mode = self.mode if isinstance(self.mode, BackendMode) else BackendMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("mode must be one of: cpu, gpu, auto") from exc
        object.__setattr__(self, "mode", mode)
        for name in ("device_index", "batch_size", "max_batch_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.device_index < 0:
            raise ValueError("device_index cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be positive")
        fraction = self.gpu_memory_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, Real):
            raise TypeError("gpu_memory_fraction must be a real number")
        fraction = float(fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be in (0, 1]")
        object.__setattr__(self, "gpu_memory_fraction", fraction)
        if self.precision_mode != _PRECISION_MODE:
            raise ValueError(
                f"precision_mode must be the strict value {_PRECISION_MODE!r}"
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": _BACKEND_SCHEMA,
            "requested_mode": self.mode.value,
            "device_index": self.device_index,
            "batch_size": self.batch_size,
            "max_batch_bytes": self.max_batch_bytes,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "precision_mode": self.precision_mode,
        }


@dataclass(frozen=True, slots=True)
class BackendProbe:
    """Read-only probe result; it is not an authorization to execute."""

    available: bool
    implementation: str
    reason: str
    device_count: int = 0
    package_version: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionBackendMetadata:
    """Stable backend/device facts suitable for a receipt or manifest."""

    schema_version: str
    requested_mode: str
    resolved_mode: str
    resolution: str
    implementation: str
    implementation_version: str
    device_index: int | None
    device_name: str
    compute_capability: str | None
    driver_version: int | None
    runtime_version: int | None
    total_memory_bytes: int | None
    precision_mode: str
    batch_size: int
    max_batch_bytes: int
    gpu_memory_fraction: float
    kernel_placement: Mapping[str, str]
    backend_identity_sha256: str

    def __post_init__(self) -> None:
        placement = {
            str(key): str(value) for key, value in self.kernel_placement.items()
        }
        object.__setattr__(
            self,
            "kernel_placement",
            MappingProxyType(dict(sorted(placement.items()))),
        )
        if (
            len(self.backend_identity_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.backend_identity_sha256
            )
        ):
            raise ValueError("backend_identity_sha256 must be a lowercase SHA-256")
        if _canonical_sha256(self.identity_payload()) != self.backend_identity_sha256:
            raise ValueError("backend identity payload does not match its SHA-256")

    def identity_payload(self) -> dict[str, object]:
        """Return the exact payload hashed as ``backend_identity_sha256``."""

        return {
            "schema_version": self.schema_version,
            "requested_mode": self.requested_mode,
            "resolved_mode": self.resolved_mode,
            "resolution": self.resolution,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "runtime_version": self.runtime_version,
            "total_memory_bytes": self.total_memory_bytes,
            "precision_mode": self.precision_mode,
            "batch_size": self.batch_size,
            "max_batch_bytes": self.max_batch_bytes,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "kernel_placement": dict(sorted(self.kernel_placement.items())),
        }


@dataclass(frozen=True, slots=True)
class ResolvedExecutionBackend:
    """Resolved backend plus bounded-memory helpers used by numerical kernels."""

    config: ExecutionBackendConfig
    metadata: ExecutionBackendMetadata
    _array_module: ModuleType = field(repr=False, compare=False)

    @property
    def mode(self) -> BackendMode:
        return BackendMode(self.metadata.resolved_mode)

    @property
    def array_module(self) -> ModuleType:
        return self._array_module

    def effective_batch_size(
        self,
        *,
        bytes_per_item: int,
        fixed_bytes: int = 0,
    ) -> int:
        """Return a positive batch bound under all declared memory limits."""

        for value, name in (
            (bytes_per_item, "bytes_per_item"),
            (fixed_bytes, "fixed_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
        bytes_per_item = int(bytes_per_item)
        fixed_bytes = int(fixed_bytes)
        if bytes_per_item <= 0:
            raise ValueError("bytes_per_item must be positive")
        if fixed_bytes < 0:
            raise ValueError("fixed_bytes cannot be negative")
        byte_limit = self.config.max_batch_bytes
        if self.mode is BackendMode.GPU:
            total = self.metadata.total_memory_bytes
            if total is None or total <= 0:
                raise BackendUnavailableError("GPU total-memory identity is unavailable")
            byte_limit = min(
                byte_limit,
                max(1, int(total * self.config.gpu_memory_fraction)),
            )
        by_memory = (byte_limit - fixed_bytes) // bytes_per_item
        if by_memory < 1:
            raise MemoryError(
                "declared batch memory limit cannot hold one work item"
            )
        return min(self.config.batch_size, by_memory)

    def iter_batches(
        self,
        item_count: int,
        *,
        bytes_per_item: int,
        fixed_bytes: int = 0,
    ) -> Iterator[slice]:
        if isinstance(item_count, bool) or not isinstance(item_count, Integral):
            raise TypeError("item_count must be an integer")
        item_count = int(item_count)
        if item_count < 0:
            raise ValueError("item_count cannot be negative")
        size = self.effective_batch_size(
            bytes_per_item=bytes_per_item,
            fixed_bytes=fixed_bytes,
        )
        for start in range(0, item_count, size):
            yield slice(start, min(start + size, item_count))

    def synchronize(self) -> None:
        if self.mode is BackendMode.GPU:
            self._array_module.cuda.get_current_stream().synchronize()


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decode_device_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    return str(value)


def _property(properties: Mapping[object, object], name: str) -> object | None:
    if name in properties:
        return properties[name]
    encoded = name.encode("ascii")
    return properties.get(encoded)


def _load_cupy() -> ModuleType:
    try:
        return importlib.import_module("cupy")
    except Exception as exc:  # Import can fail while loading the CUDA DLLs.
        raise BackendUnavailableError(
            f"CuPy/CUDA import failed: {type(exc).__name__}: {exc}"
        ) from exc


def probe_gpu_backend() -> BackendProbe:
    """Probe optional CuPy/CUDA availability without changing backend state."""

    try:
        cupy = _load_cupy()
        count = int(cupy.cuda.runtime.getDeviceCount())
        if count < 1:
            return BackendProbe(
                available=False,
                implementation="cupy.cuda",
                reason="CUDA runtime reported zero devices",
                package_version=str(cupy.__version__),
            )
        return BackendProbe(
            available=True,
            implementation="cupy.cuda",
            reason="AVAILABLE",
            device_count=count,
            package_version=str(cupy.__version__),
        )
    except BackendUnavailableError as exc:
        return BackendProbe(
            available=False,
            implementation="cupy.cuda",
            reason=str(exc),
        )
    except Exception as exc:  # Driver/runtime calls can fail after import.
        return BackendProbe(
            available=False,
            implementation="cupy.cuda",
            reason=f"CUDA probe failed: {type(exc).__name__}: {exc}",
        )


def _metadata(
    config: ExecutionBackendConfig,
    *,
    resolved_mode: BackendMode,
    resolution: str,
    implementation: str,
    implementation_version: str,
    device_index: int | None,
    device_name: str,
    compute_capability: str | None,
    driver_version: int | None,
    runtime_version: int | None,
    total_memory_bytes: int | None,
) -> ExecutionBackendMetadata:
    placement = (
        {
            "composite_harm_reporting": "gpu_float64_batched",
            "counter_rng": "cpu_reference_exact",
            "categorical_decisions": "cpu_reference_exact",
            "model_state_transitions": "cpu_reference",
        }
        if resolved_mode is BackendMode.GPU
        else {
            "composite_harm_reporting": "cpu_reference",
            "counter_rng": "cpu_reference",
            "categorical_decisions": "cpu_reference",
            "model_state_transitions": "cpu_reference",
        }
    )
    payload: dict[str, object] = {
        "schema_version": _BACKEND_SCHEMA,
        "requested_mode": config.mode.value,
        "resolved_mode": resolved_mode.value,
        "resolution": resolution,
        "implementation": implementation,
        "implementation_version": implementation_version,
        "device_index": device_index,
        "device_name": device_name,
        "compute_capability": compute_capability,
        "driver_version": driver_version,
        "runtime_version": runtime_version,
        "total_memory_bytes": total_memory_bytes,
        "precision_mode": config.precision_mode,
        "batch_size": config.batch_size,
        "max_batch_bytes": config.max_batch_bytes,
        "gpu_memory_fraction": config.gpu_memory_fraction,
        "kernel_placement": dict(sorted(placement.items())),
    }
    return ExecutionBackendMetadata(
        **payload,
        backend_identity_sha256=_canonical_sha256(payload),
    )


def _resolve_cpu(
    config: ExecutionBackendConfig,
    *,
    resolution: str,
) -> ResolvedExecutionBackend:
    metadata = _metadata(
        config,
        resolved_mode=BackendMode.CPU,
        resolution=resolution,
        implementation="numpy",
        implementation_version=np.__version__,
        device_index=None,
        device_name=platform.processor() or platform.machine() or "UNKNOWN_CPU",
        compute_capability=None,
        driver_version=None,
        runtime_version=None,
        total_memory_bytes=None,
    )
    return ResolvedExecutionBackend(config, metadata, np)


def _resolve_gpu(
    config: ExecutionBackendConfig,
    *,
    resolution: str,
) -> ResolvedExecutionBackend:
    cupy = _load_cupy()
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
        if config.device_index >= count:
            raise BackendUnavailableError(
                f"GPU device_index {config.device_index} is outside [0, {count - 1}]"
            )
        with cupy.cuda.Device(config.device_index):
            properties = cupy.cuda.runtime.getDeviceProperties(config.device_index)
            _free_memory, total_memory = cupy.cuda.runtime.memGetInfo()
            driver_version = int(cupy.cuda.runtime.driverGetVersion())
            runtime_version = int(cupy.cuda.runtime.runtimeGetVersion())
        major = int(_property(properties, "major") or 0)
        minor = int(_property(properties, "minor") or 0)
        name = _decode_device_name(_property(properties, "name") or "UNKNOWN_GPU")
    except BackendUnavailableError:
        raise
    except Exception as exc:
        raise BackendUnavailableError(
            f"CUDA device initialization failed: {type(exc).__name__}: {exc}"
        ) from exc
    metadata = _metadata(
        config,
        resolved_mode=BackendMode.GPU,
        resolution=resolution,
        implementation="cupy.cuda",
        implementation_version=str(cupy.__version__),
        device_index=config.device_index,
        device_name=name,
        compute_capability=f"{major}.{minor}",
        driver_version=driver_version,
        runtime_version=runtime_version,
        total_memory_bytes=int(total_memory),
    )
    return ResolvedExecutionBackend(config, metadata, cupy)


def resolve_execution_backend(
    config: ExecutionBackendConfig,
) -> ResolvedExecutionBackend:
    """Resolve one explicit backend contract with no silent GPU fallback."""

    if not isinstance(config, ExecutionBackendConfig):
        raise TypeError("config must be ExecutionBackendConfig")
    if config.mode is BackendMode.CPU:
        return _resolve_cpu(config, resolution="EXPLICIT_CPU")
    if config.mode is BackendMode.GPU:
        probe = probe_gpu_backend()
        if not probe.available:
            raise BackendUnavailableError(
                "GPU was explicitly requested but is unavailable: " + probe.reason
            )
        return _resolve_gpu(config, resolution="EXPLICIT_GPU")
    probe = probe_gpu_backend()
    if probe.available:
        return _resolve_gpu(config, resolution="AUTO_SELECTED_GPU")
    return _resolve_cpu(
        config,
        resolution="AUTO_SELECTED_CPU_NO_COMPATIBLE_GPU: " + probe.reason,
    )


__all__ = [
    "BackendMode",
    "BackendProbe",
    "BackendUnavailableError",
    "ExecutionBackendConfig",
    "ExecutionBackendMetadata",
    "ResolvedExecutionBackend",
    "probe_gpu_backend",
    "resolve_execution_backend",
]
