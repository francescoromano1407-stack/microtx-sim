"""Bounded numerical kernels that cannot perturb random or policy decisions.

The current GPU-eligible kernel is the post-simulation composite-harm reporting
view. Random draws, utilities, categorical choices, state transitions, money,
and scenario order remain on the exact CPU reference path. GPU output is
float64 and must pass the declared parity contract before it can be accepted.

The six-term reduction can differ in its last floating-point bits because a GPU
BLAS implementation may use another valid reduction order. The default
tolerance is deliberately much tighter than any published precision. Threshold
classification is tested exactly and is never covered by numeric tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Iterable

import numpy as np
import numpy.typing as npt

from .backends import BackendMode, ResolvedExecutionBackend


FloatArray = npt.NDArray[np.float64]
_COMPOSITE_COMPONENT_COUNT = 6
_GPU_BYTES_PER_ROW = (_COMPOSITE_COMPONENT_COUNT + 1) * 8
_GPU_FIXED_BYTES = _COMPOSITE_COMPONENT_COUNT * 8


class NumericalParityError(RuntimeError):
    """Raised when an accelerated result violates its parity contract."""


@dataclass(frozen=True, slots=True)
class NumericalParityTolerance:
    """Tolerance only for continuous float64 reporting outputs."""

    absolute: float = 5e-13
    relative: float = 5e-13

    def __post_init__(self) -> None:
        for name in ("absolute", "relative"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            value = float(value)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


def _validated_inputs(
    component_scores: npt.ArrayLike,
    weights: npt.ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    scores = np.asarray(component_scores)
    if scores.dtype != np.dtype(np.float64):
        raise TypeError("component_scores must use float64")
    if scores.ndim != 2 or scores.shape[1] != _COMPOSITE_COMPONENT_COUNT:
        raise ValueError("component_scores must have shape (players, 6)")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("component_scores must be finite and in [0, 1]")
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (_COMPOSITE_COMPONENT_COUNT,):
        raise ValueError("weights must have shape (6,)")
    if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(weight_array.sum())
    if not isfinite(total) or total <= 0.0:
        raise ValueError("weights must have a finite positive sum")
    return scores, weight_array / total


def _compute_validated(
    scores: FloatArray,
    normalized_weights: FloatArray,
    backend: ResolvedExecutionBackend,
) -> FloatArray:
    if backend.mode is BackendMode.CPU:
        return np.asarray(scores @ normalized_weights, dtype=np.float64)
    output = np.empty(scores.shape[0], dtype=np.float64)
    if output.size == 0:
        return output
    cupy = backend.array_module
    device_index = backend.metadata.device_index
    if device_index is None:
        raise RuntimeError("resolved GPU backend has no device index")
    with cupy.cuda.Device(device_index):
        device_weights = cupy.asarray(normalized_weights, dtype=cupy.float64)
        for batch in backend.iter_batches(
            output.size,
            bytes_per_item=_GPU_BYTES_PER_ROW,
            fixed_bytes=_GPU_FIXED_BYTES,
        ):
            device_scores = cupy.asarray(scores[batch], dtype=cupy.float64)
            device_result = device_scores @ device_weights
            output[batch] = cupy.asnumpy(device_result)
            # Release batch references before allocating the next batch. CuPy's
            # pool may reuse the storage, but two live batches are never kept.
            del device_scores, device_result
    if not np.all(np.isfinite(output)) or np.any((output < 0.0) | (output > 1.0)):
        raise ArithmeticError("composite-harm backend produced invalid values")
    return output


def compute_composite_harm(
    component_scores: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    backend: ResolvedExecutionBackend,
) -> FloatArray:
    """Compute the reporting composite on an explicitly resolved backend.

    The CPU branch is exactly the pre-existing NumPy expression. The GPU branch
    transfers and reduces bounded row batches, retaining only the final
    one-dimensional result on the host. No model state or random stream is
    moved to the device.
    """

    if not isinstance(backend, ResolvedExecutionBackend):
        raise TypeError("backend must be a ResolvedExecutionBackend")
    scores, normalized_weights = _validated_inputs(component_scores, weights)
    return _compute_validated(scores, normalized_weights, backend)


@dataclass(frozen=True, slots=True)
class CompositeHarmParityReport:
    backend_identity_sha256: str
    row_count: int
    absolute_tolerance: float
    relative_tolerance: float
    bitwise_equal: bool
    within_tolerance: bool
    threshold_classifications_equal: bool
    mean_direction_equal: bool
    maximum_absolute_difference: float
    maximum_relative_difference: float

    @property
    def passed(self) -> bool:
        return bool(
            self.within_tolerance
            and self.threshold_classifications_equal
            and self.mean_direction_equal
        )


def _direction(value: float, reference: float) -> int:
    difference = value - reference
    return 1 if difference > 0.0 else (-1 if difference < 0.0 else 0)


def validate_composite_harm_parity(
    component_scores: npt.ArrayLike,
    weights: npt.ArrayLike,
    *,
    backend: ResolvedExecutionBackend,
    tolerance: NumericalParityTolerance | None = None,
    categorical_thresholds: Iterable[float] = (0.35,),
    mean_direction_reference: float = 0.0,
    raise_on_failure: bool = False,
) -> CompositeHarmParityReport:
    """Prove continuous tolerance and exact derived classifications.

    ``categorical_thresholds`` are compared with exact boolean equality; no
    epsilon is applied. Aggregate-mean direction relative to
    ``mean_direction_reference`` must also remain identical.
    """

    if not isinstance(backend, ResolvedExecutionBackend):
        raise TypeError("backend must be a ResolvedExecutionBackend")
    declared = tolerance or NumericalParityTolerance()
    if not isinstance(declared, NumericalParityTolerance):
        raise TypeError("tolerance must be NumericalParityTolerance")
    scores, normalized_weights = _validated_inputs(component_scores, weights)
    reference = np.asarray(scores @ normalized_weights, dtype=np.float64)
    accelerated = _compute_validated(scores, normalized_weights, backend)
    absolute = np.abs(reference - accelerated)
    denominator = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    relative = absolute / denominator
    thresholds: list[float] = []
    for raw_threshold in categorical_thresholds:
        if isinstance(raw_threshold, bool) or not isinstance(raw_threshold, Real):
            raise TypeError("categorical thresholds must be real numbers")
        threshold = float(raw_threshold)
        if not isfinite(threshold):
            raise ValueError("categorical thresholds must be finite")
        thresholds.append(threshold)
    classifications_equal = all(
        np.array_equal(reference >= threshold, accelerated >= threshold)
        for threshold in thresholds
    )
    mean_reference = float(reference.mean()) if reference.size else 0.0
    mean_accelerated = float(accelerated.mean()) if accelerated.size else 0.0
    report = CompositeHarmParityReport(
        backend_identity_sha256=backend.metadata.backend_identity_sha256,
        row_count=int(reference.size),
        absolute_tolerance=declared.absolute,
        relative_tolerance=declared.relative,
        bitwise_equal=np.array_equal(reference, accelerated),
        within_tolerance=np.allclose(
            reference,
            accelerated,
            atol=declared.absolute,
            rtol=declared.relative,
        ),
        threshold_classifications_equal=classifications_equal,
        mean_direction_equal=(
            _direction(mean_reference, mean_direction_reference)
            == _direction(mean_accelerated, mean_direction_reference)
        ),
        maximum_absolute_difference=(float(absolute.max()) if absolute.size else 0.0),
        maximum_relative_difference=(float(relative.max()) if relative.size else 0.0),
    )
    if raise_on_failure and not report.passed:
        raise NumericalParityError(
            "accelerated composite-harm kernel failed the declared parity contract"
        )
    return report


__all__ = [
    "CompositeHarmParityReport",
    "NumericalParityError",
    "NumericalParityTolerance",
    "compute_composite_harm",
    "validate_composite_harm_parity",
]
