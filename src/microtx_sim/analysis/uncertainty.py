"""Fail-closed uncertainty propagation and blockwise convergence contracts.

This module deliberately separates Monte Carlo seed variation from uncertainty
about model parameters, the target population, and monetary rates.  A missing
uncertainty design is represented by ``UNQUANTIFIED``; it is never silently
translated into a zero variance component.

The estimand supplied to this layer must already be paired across scenarios and
population weighted within a seed.  Consequently the seed summary cannot
accidentally pool player rows, currencies, or unpaired scenario branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import erf, isfinite, sqrt
import os
from pathlib import Path
import stat
from statistics import NormalDist
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from ..rng import validate_seed


UNCERTAINTY_SCHEMA_VERSION: Final[str] = "1.0"
CONVERGENCE_SCHEMA_VERSION: Final[str] = "1.0"
NORMAL_95_Z: Final[float] = 1.96


class UncertaintyValidationError(ValueError):
    """Raised when an uncertainty declaration or realization is ambiguous."""


class UncertaintyAvailability(str, Enum):
    QUANTIFIED = "QUANTIFIED"
    UNQUANTIFIED = "UNQUANTIFIED"
    UNAVAILABLE = "UNAVAILABLE"


class ParameterProvenanceStatus(str, Enum):
    CALIBRATED_DISTRIBUTION = "CALIBRATED_DISTRIBUTION"
    ILLUSTRATIVE_RANGE = "ILLUSTRATIVE_RANGE"


class ConvergenceStatus(str, Enum):
    CONVERGED = "CONVERGED"
    NON_CONVERGED = "NON_CONVERGED"
    INSUFFICIENT_PRECISION = "INSUFFICIENT_PRECISION"
    UNSTABLE = "UNSTABLE"


def canonical_sha256(payload: object) -> str:
    """Hash a JSON identity with UTF-8 and deterministic key ordering."""

    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise UncertaintyValidationError(f"{name} must be lowercase SHA-256 hex")
    return value


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or len(value) > 256:
        raise UncertaintyValidationError(f"{name} must be non-empty text")
    if any(character.isspace() for character in value):
        raise UncertaintyValidationError(f"{name} cannot contain whitespace")
    return value


def _finite(value: object, *, name: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise UncertaintyValidationError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise UncertaintyValidationError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ParameterDeclaration:
    """One declared uncertain parameter and its evidentiary interpretation."""

    parameter_id: str
    source: str
    provenance_status: ParameterProvenanceStatus
    nominal_value: float
    lower_bound: float
    upper_bound: float
    probability_distribution: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.parameter_id, name="parameter_id")
        if type(self.source) is not str or not self.source.strip():
            raise UncertaintyValidationError("parameter source must be non-empty")
        if type(self.provenance_status) is not ParameterProvenanceStatus:
            raise TypeError("provenance_status must be ParameterProvenanceStatus")
        nominal = _finite(self.nominal_value, name="nominal_value")
        lower = _finite(self.lower_bound, name="lower_bound")
        upper = _finite(self.upper_bound, name="upper_bound")
        if not lower < upper or not lower <= nominal <= upper:
            raise UncertaintyValidationError(
                "parameter bounds must be ordered and contain the nominal value"
            )
        if self.provenance_status is ParameterProvenanceStatus.ILLUSTRATIVE_RANGE:
            if self.probability_distribution is not None:
                raise UncertaintyValidationError(
                    "an illustrative range cannot be relabelled as a probability "
                    "distribution"
                )
        elif self.probability_distribution not in {"UNIFORM"}:
            raise UncertaintyValidationError(
                "calibrated distributions require an implemented distribution"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "parameter_id": self.parameter_id,
            "source": self.source,
            "provenance_status": self.provenance_status.value,
            "nominal_value": self.nominal_value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "probability_distribution": self.probability_distribution,
        }


@dataclass(frozen=True, slots=True)
class ParameterUncertaintyDesign:
    """Seeded Latin-hypercube design with optional justified correlations."""

    design_id: str
    design_seed: int
    draw_count: int
    parameters: tuple[ParameterDeclaration, ...]
    correlation_matrix: tuple[tuple[float, ...], ...] | None = None
    correlation_source: str | None = None
    method: str = "SEEDED_LATIN_HYPERCUBE_V1"

    def __post_init__(self) -> None:
        _identifier(self.design_id, name="design_id")
        validate_seed(self.design_seed, name="parameter design seed")
        if type(self.draw_count) is not int or isinstance(self.draw_count, bool):
            raise TypeError("draw_count must be an integer")
        if self.draw_count < 2:
            raise UncertaintyValidationError("draw_count must be at least two")
        if self.method != "SEEDED_LATIN_HYPERCUBE_V1":
            raise UncertaintyValidationError("unsupported parameter design method")
        if type(self.parameters) is not tuple or not self.parameters or any(
            type(item) is not ParameterDeclaration for item in self.parameters
        ):
            raise TypeError("parameters must be a non-empty exact tuple")
        ids = tuple(item.parameter_id for item in self.parameters)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise UncertaintyValidationError(
                "parameter declarations must be unique and sorted by ID"
            )
        dimension = len(self.parameters)
        if self.correlation_matrix is None:
            if self.correlation_source is not None:
                raise UncertaintyValidationError(
                    "correlation_source requires a correlation matrix"
                )
            return
        if type(self.correlation_source) is not str or not self.correlation_source.strip():
            raise UncertaintyValidationError(
                "a non-identity correlation matrix requires provenance"
            )
        matrix = np.asarray(self.correlation_matrix, dtype=np.float64)
        if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
            raise UncertaintyValidationError("correlation matrix has invalid shape")
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
            raise UncertaintyValidationError("correlation matrix must be symmetric")
        if not np.allclose(np.diag(matrix), 1.0, rtol=0.0, atol=1e-12):
            raise UncertaintyValidationError("correlation diagonal must equal one")
        eigenvalues = np.linalg.eigvalsh(matrix)
        if float(eigenvalues.min()) < -1e-12:
            raise UncertaintyValidationError(
                "correlation matrix must be positive semidefinite"
            )

    @property
    def design_sha256(self) -> str:
        return canonical_sha256(self.snapshot())

    @property
    def calibrated_probability_design(self) -> bool:
        return all(
            item.provenance_status
            is ParameterProvenanceStatus.CALIBRATED_DISTRIBUTION
            for item in self.parameters
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": UNCERTAINTY_SCHEMA_VERSION,
            "design_id": self.design_id,
            "method": self.method,
            "design_seed": self.design_seed,
            "draw_count": self.draw_count,
            "parameters": [item.snapshot() for item in self.parameters],
            "correlation_matrix": (
                [list(row) for row in self.correlation_matrix]
                if self.correlation_matrix is not None
                else None
            ),
            "correlation_source": self.correlation_source,
            "probability_interpretation": (
                "CALIBRATED_JOINT_DISTRIBUTION"
                if self.calibrated_probability_design
                else "NONE_ILLUSTRATIVE_DESIGN_POINTS"
            ),
            "oat_role": "DIAGNOSTIC_ONLY",
        }


@dataclass(frozen=True, slots=True)
class ParameterDraw:
    draw_id: str
    design_id: str
    design_sha256: str
    ordinal: int
    values: tuple[tuple[str, float], ...]
    probability_interpretation: bool
    draw_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.draw_id, name="draw_id")
        _identifier(self.design_id, name="design_id")
        _digest(self.design_sha256, name="design_sha256")
        _digest(self.draw_sha256, name="draw_sha256")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise UncertaintyValidationError("draw ordinal must be non-negative")
        if type(self.probability_interpretation) is not bool:
            raise TypeError("probability_interpretation must be boolean")
        if type(self.values) is not tuple or not self.values:
            raise UncertaintyValidationError("draw values cannot be empty")
        ids = tuple(item[0] for item in self.values)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise UncertaintyValidationError("draw values must be sorted and unique")
        for parameter_id, value in self.values:
            _identifier(parameter_id, name="draw parameter_id")
            _finite(value, name=f"draw {parameter_id}")
        if self.draw_sha256 != canonical_sha256(self.attestation_payload()):
            raise UncertaintyValidationError("draw_sha256 does not match its payload")

    def attestation_payload(self) -> dict[str, object]:
        return {
            "draw_id": self.draw_id,
            "design_id": self.design_id,
            "design_sha256": self.design_sha256,
            "ordinal": self.ordinal,
            "values": {name: value for name, value in self.values},
            "probability_interpretation": self.probability_interpretation,
        }

    def snapshot(self) -> dict[str, object]:
        return {**self.attestation_payload(), "draw_sha256": self.draw_sha256}


@dataclass(frozen=True, slots=True)
class LoadedParameterUncertaintyDesign:
    design_path: Path
    byte_length: int
    file_sha256: str
    design: ParameterUncertaintyDesign

    def __post_init__(self) -> None:
        if not isinstance(self.design_path, Path) or not self.design_path.is_absolute():
            raise TypeError("design_path must be an absolute Path")
        if type(self.byte_length) is not int or self.byte_length <= 0:
            raise UncertaintyValidationError("design byte_length must be positive")
        _digest(self.file_sha256, name="design file_sha256")
        if type(self.design) is not ParameterUncertaintyDesign:
            raise TypeError("design must be ParameterUncertaintyDesign")
        observed = _read_design_file(self.design_path)
        if len(observed) != self.byte_length or sha256(observed).hexdigest() != self.file_sha256:
            raise UncertaintyValidationError(
                "parameter uncertainty design changed after loading"
            )
        if parameter_design_from_snapshot(json.loads(observed)) != self.design:
            raise UncertaintyValidationError(
                "loaded parameter design differs from its file"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "design_path": str(self.design_path),
            "byte_length": self.byte_length,
            "file_sha256": self.file_sha256,
            "design_sha256": self.design.design_sha256,
            "design": self.design.snapshot(),
        }


def parameter_design_from_snapshot(
    row: Mapping[str, object],
) -> ParameterUncertaintyDesign:
    expected = {
        "schema_version",
        "design_id",
        "method",
        "design_seed",
        "draw_count",
        "parameters",
        "correlation_matrix",
        "correlation_source",
        "probability_interpretation",
        "oat_role",
    }
    if type(row) is not dict or set(row) != expected:
        raise UncertaintyValidationError(
            "parameter design keys differ from the strict schema"
        )
    if row["schema_version"] != UNCERTAINTY_SCHEMA_VERSION:
        raise UncertaintyValidationError("unsupported parameter design schema")
    raw_parameters = row["parameters"]
    if type(raw_parameters) is not list:
        raise UncertaintyValidationError("parameters must be a JSON array")
    parameters: list[ParameterDeclaration] = []
    parameter_keys = {
        "parameter_id",
        "source",
        "provenance_status",
        "nominal_value",
        "lower_bound",
        "upper_bound",
        "probability_distribution",
    }
    for index, value in enumerate(raw_parameters):
        if type(value) is not dict or set(value) != parameter_keys:
            raise UncertaintyValidationError(
                f"parameters[{index}] differs from the strict schema"
            )
        try:
            status = ParameterProvenanceStatus(value["provenance_status"])
        except (TypeError, ValueError) as exc:
            raise UncertaintyValidationError(
                f"parameters[{index}] has invalid provenance status"
            ) from exc
        parameters.append(
            ParameterDeclaration(
                parameter_id=value["parameter_id"],
                source=value["source"],
                provenance_status=status,
                nominal_value=value["nominal_value"],
                lower_bound=value["lower_bound"],
                upper_bound=value["upper_bound"],
                probability_distribution=value["probability_distribution"],
            )
        )
    raw_correlation = row["correlation_matrix"]
    correlation = None
    if raw_correlation is not None:
        if type(raw_correlation) is not list or any(type(item) is not list for item in raw_correlation):
            raise UncertaintyValidationError("correlation_matrix must be nested arrays")
        correlation = tuple(tuple(float(value) for value in item) for item in raw_correlation)
    design = ParameterUncertaintyDesign(
        design_id=row["design_id"],
        design_seed=row["design_seed"],
        draw_count=row["draw_count"],
        parameters=tuple(parameters),
        correlation_matrix=correlation,
        correlation_source=row["correlation_source"],
        method=row["method"],
    )
    if design.snapshot() != dict(row):
        raise UncertaintyValidationError(
            "parameter design is not the canonical schema snapshot"
        )
    return design


def _read_design_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise UncertaintyValidationError(
            f"cannot inspect parameter design: {path}"
        ) from exc
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        path.is_symlink()
        or bool(getattr(metadata, "st_file_attributes", 0) & marker)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise UncertaintyValidationError(
            "parameter design must be a regular non-symlink file"
        )
    if metadata.st_size <= 0 or metadata.st_size > 1024 * 1024:
        raise UncertaintyValidationError("parameter design file size is invalid")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise UncertaintyValidationError(
            f"cannot read parameter design: {path}"
        ) from exc


def load_parameter_uncertainty_design(
    path: str | Path,
) -> LoadedParameterUncertaintyDesign:
    candidate = Path(os.path.abspath(os.fspath(path)))
    observed = _read_design_file(candidate)
    try:
        raw = json.loads(observed)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UncertaintyValidationError(
            "parameter uncertainty design must be valid UTF-8 JSON"
        ) from exc
    design = parameter_design_from_snapshot(raw)
    return LoadedParameterUncertaintyDesign(
        design_path=candidate,
        byte_length=len(observed),
        file_sha256=sha256(observed).hexdigest(),
        design=design,
    )


def verify_loaded_parameter_uncertainty_design(
    loaded: LoadedParameterUncertaintyDesign,
) -> LoadedParameterUncertaintyDesign:
    if type(loaded) is not LoadedParameterUncertaintyDesign:
        raise TypeError("loaded must be LoadedParameterUncertaintyDesign")
    LoadedParameterUncertaintyDesign.__post_init__(loaded)
    observed = load_parameter_uncertainty_design(loaded.design_path)
    if observed != loaded:
        raise UncertaintyValidationError(
            "parameter uncertainty design changed after it was loaded"
        )
    return observed


def _permutation(count: int, *, design_seed: int, parameter_id: str) -> list[int]:
    keyed = []
    for index in range(count):
        key = sha256(
            f"lhs-v1\0{design_seed}\0{parameter_id}\0{index}".encode("utf-8")
        ).digest()
        keyed.append((key, index))
    return [index for _, index in sorted(keyed)]


def generate_parameter_draws(
    design: ParameterUncertaintyDesign,
) -> tuple[ParameterDraw, ...]:
    """Generate deterministic, content-addressed joint parameter design points."""

    if type(design) is not ParameterUncertaintyDesign:
        raise TypeError("design must be ParameterUncertaintyDesign")
    ParameterUncertaintyDesign.__post_init__(design)
    count = design.draw_count
    dimension = len(design.parameters)
    uniforms = np.empty((count, dimension), dtype=np.float64)
    for column, parameter in enumerate(design.parameters):
        permutation = _permutation(
            count,
            design_seed=design.design_seed,
            parameter_id=parameter.parameter_id,
        )
        for row, stratum in enumerate(permutation):
            uniforms[row, column] = (stratum + 0.5) / count
    if design.correlation_matrix is not None:
        # A deterministic Gaussian rank transform supplies correlated design
        # points.  The declared matrix is a design input, not inferred evidence.
        normal = NormalDist()
        z = np.vectorize(normal.inv_cdf)(uniforms)
        matrix = np.asarray(design.correlation_matrix, dtype=np.float64)
        values, vectors = np.linalg.eigh(matrix)
        root = vectors @ np.diag(np.sqrt(np.clip(values, 0.0, None))) @ vectors.T
        correlated = z @ root.T
        uniforms = 0.5 * (1.0 + np.vectorize(erf)(correlated / sqrt(2.0)))
        uniforms = np.clip(uniforms, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    output: list[ParameterDraw] = []
    design_sha = design.design_sha256
    for ordinal in range(count):
        values = tuple(
            (
                parameter.parameter_id,
                parameter.lower_bound
                + (parameter.upper_bound - parameter.lower_bound)
                * float(uniforms[ordinal, column]),
            )
            for column, parameter in enumerate(design.parameters)
        )
        draw_id = f"{design.design_id}.draw-{ordinal:04d}"
        payload = {
            "draw_id": draw_id,
            "design_id": design.design_id,
            "design_sha256": design_sha,
            "ordinal": ordinal,
            "values": {name: value for name, value in values},
            "probability_interpretation": design.calibrated_probability_design,
        }
        output.append(
            ParameterDraw(
                draw_id=draw_id,
                design_id=design.design_id,
                design_sha256=design_sha,
                ordinal=ordinal,
                values=values,
                probability_interpretation=design.calibrated_probability_design,
                draw_sha256=canonical_sha256(payload),
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class RealizationIdentity:
    """Complete identity needed to prevent accidental cross-design pooling."""

    seed: int
    parameter_draw_id: str
    parameter_draw_sha256: str
    population_design_id: str
    population_replicate_id: str
    population_design_sha256: str
    monetary_rate_draw_id: str
    monetary_rate_basis_id: str
    monetary_rate_basis_sha256: str
    scenario_id: str
    primary_estimand_id: str
    pretreatment_cohort_sha256: str
    population_weights_sha256: str

    def __post_init__(self) -> None:
        validate_seed(self.seed, name="uncertainty realization seed")
        for name in (
            "parameter_draw_id",
            "population_design_id",
            "population_replicate_id",
            "monetary_rate_draw_id",
            "monetary_rate_basis_id",
            "scenario_id",
            "primary_estimand_id",
        ):
            _identifier(getattr(self, name), name=name)
        for name in (
            "parameter_draw_sha256",
            "population_design_sha256",
            "monetary_rate_basis_sha256",
            "pretreatment_cohort_sha256",
            "population_weights_sha256",
        ):
            _digest(getattr(self, name), name=name)

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.snapshot())

    def snapshot(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "seed_decimal": str(self.seed),
            "parameter_draw_id": self.parameter_draw_id,
            "parameter_draw_sha256": self.parameter_draw_sha256,
            "population_design_id": self.population_design_id,
            "population_replicate_id": self.population_replicate_id,
            "population_design_sha256": self.population_design_sha256,
            "monetary_rate_draw_id": self.monetary_rate_draw_id,
            "monetary_rate_basis_id": self.monetary_rate_basis_id,
            "monetary_rate_basis_sha256": self.monetary_rate_basis_sha256,
            "scenario_id": self.scenario_id,
            "primary_estimand_id": self.primary_estimand_id,
            "pretreatment_cohort_sha256": self.pretreatment_cohort_sha256,
            "population_weights_sha256": self.population_weights_sha256,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyRealization:
    """One already-paired, already-weighted primary-estimand realization."""

    identity: RealizationIdentity
    estimate: float | None
    valid: bool
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.identity) is not RealizationIdentity:
            raise TypeError("identity must be RealizationIdentity")
        RealizationIdentity.__post_init__(self.identity)
        if type(self.valid) is not bool:
            raise TypeError("valid must be boolean")
        if self.valid:
            _finite(self.estimate, name="realization estimate")
            if self.invalid_reason is not None:
                raise UncertaintyValidationError(
                    "a valid realization cannot have an invalid reason"
                )
        else:
            if self.estimate is not None:
                raise UncertaintyValidationError(
                    "an invalid realization cannot carry an estimate"
                )
            if type(self.invalid_reason) is not str or not self.invalid_reason.strip():
                raise UncertaintyValidationError(
                    "an invalid realization requires a reason"
                )

    def snapshot(self) -> dict[str, object]:
        return {
            "identity": self.identity.snapshot(),
            "identity_sha256": self.identity.identity_sha256,
            "estimate": self.estimate,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


def _fixed_analysis_identity(identity: RealizationIdentity) -> tuple[object, ...]:
    """Return every non-seed design identity used by seed-only diagnostics."""

    return (
        identity.parameter_draw_id,
        identity.parameter_draw_sha256,
        identity.population_design_id,
        identity.population_replicate_id,
        identity.population_design_sha256,
        identity.monetary_rate_draw_id,
        identity.monetary_rate_basis_id,
        identity.monetary_rate_basis_sha256,
        identity.scenario_id,
        identity.primary_estimand_id,
    )


@dataclass(frozen=True, slots=True)
class SeedUncertaintySummary:
    retained_seed_count: int
    point_estimate: float
    sample_standard_deviation: float
    monte_carlo_standard_error: float
    interval_lower: float
    interval_upper: float

    def __post_init__(self) -> None:
        if (
            type(self.retained_seed_count) is not int
            or isinstance(self.retained_seed_count, bool)
            or self.retained_seed_count <= 0
        ):
            raise UncertaintyValidationError(
                "retained_seed_count must be a positive integer"
            )
        for name in (
            "point_estimate",
            "sample_standard_deviation",
            "monte_carlo_standard_error",
            "interval_lower",
            "interval_upper",
        ):
            _finite(getattr(self, name), name=name)
        if self.sample_standard_deviation < 0.0:
            raise UncertaintyValidationError(
                "sample_standard_deviation cannot be negative"
            )
        if self.monte_carlo_standard_error < 0.0:
            raise UncertaintyValidationError(
                "monte_carlo_standard_error cannot be negative"
            )
        expected_mcse = self.sample_standard_deviation / sqrt(
            self.retained_seed_count
        )
        if not np.isclose(
            self.monte_carlo_standard_error,
            expected_mcse,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise UncertaintyValidationError(
                "monte_carlo_standard_error differs from sd / sqrt(seed count)"
            )
        expected_lower = (
            self.point_estimate
            - NORMAL_95_Z * self.monte_carlo_standard_error
        )
        expected_upper = (
            self.point_estimate
            + NORMAL_95_Z * self.monte_carlo_standard_error
        )
        if not np.isclose(
            self.interval_lower,
            expected_lower,
            rtol=1e-12,
            atol=1e-15,
        ) or not np.isclose(
            self.interval_upper,
            expected_upper,
            rtol=1e-12,
            atol=1e-15,
        ):
            raise UncertaintyValidationError(
                "Monte Carlo interval differs from the declared normal method"
            )

    @property
    def interval_width(self) -> float:
        return self.interval_upper - self.interval_lower

    def snapshot(self) -> dict[str, object]:
        return {
            "retained_seed_count": self.retained_seed_count,
            "point_estimate": self.point_estimate,
            "sample_standard_deviation": self.sample_standard_deviation,
            "monte_carlo_standard_error": self.monte_carlo_standard_error,
            "interval_method": "NORMAL_95_MONTE_CARLO_MEAN_PLUS_MINUS_1.96_MCSE",
            "interval_lower": self.interval_lower,
            "interval_upper": self.interval_upper,
            "interval_width": self.interval_width,
        }


def summarize_seed_uncertainty(
    realizations: Sequence[UncertaintyRealization],
    *,
    expected_seeds: Sequence[int],
) -> SeedUncertaintySummary:
    """Summarize one complete fixed-input seed set with no exclusion API."""

    expected = tuple(
        validate_seed(seed, name=f"expected_seeds[{index}]")
        for index, seed in enumerate(expected_seeds)
    )
    if not expected or expected != tuple(sorted(expected)) or len(expected) != len(set(expected)):
        raise UncertaintyValidationError(
            "expected seeds must be non-empty, unique, and ascending"
        )
    rows = tuple(realizations)
    if any(type(row) is not UncertaintyRealization for row in rows):
        raise TypeError("realizations must contain UncertaintyRealization values")
    for row in rows:
        UncertaintyRealization.__post_init__(row)
    if len(rows) != len(expected):
        raise UncertaintyValidationError(
            "the complete fixed seed set is required; exclusions are prohibited"
        )
    observed = tuple(row.identity.seed for row in rows)
    if observed != expected:
        raise UncertaintyValidationError(
            "realizations must exactly follow the declared fixed seed set"
        )
    if any(not row.valid for row in rows):
        raise UncertaintyValidationError(
            "invalid fixed-seed realizations fail closed and cannot be excluded"
        )
    fixed_identity = [_fixed_analysis_identity(row.identity) for row in rows]
    if len(set(fixed_identity)) != 1:
        raise UncertaintyValidationError(
            "seed-only uncertainty requires fixed parameter, population, rate, "
            "scenario, and estimand identities"
        )
    values = np.asarray([float(row.estimate) for row in rows], dtype=np.float64)
    point = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    mcse = sd / sqrt(len(values))
    return SeedUncertaintySummary(
        retained_seed_count=len(values),
        point_estimate=point,
        sample_standard_deviation=sd,
        monte_carlo_standard_error=mcse,
        interval_lower=point - NORMAL_95_Z * mcse,
        interval_upper=point + NORMAL_95_Z * mcse,
    )


@dataclass(frozen=True, slots=True)
class ConvergenceRule:
    block_size: int = 50
    minimum_retained_seeds: int = 100
    maximum_mcse: float = 0.0025
    maximum_interval_width: float = 0.01
    maximum_absolute_change: float = 0.0025
    maximum_relative_change: float = 0.025
    maximum_invalid_rate: float = 0.0
    consecutive_passing_checkpoints: int = 2

    def __post_init__(self) -> None:
        for name in (
            "block_size",
            "minimum_retained_seeds",
            "consecutive_passing_checkpoints",
        ):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value <= 0:
                raise UncertaintyValidationError(f"{name} must be positive integer")
        if self.minimum_retained_seeds < 100:
            raise UncertaintyValidationError(
                "campaign convergence requires at least 100 retained seeds"
            )
        for name in (
            "maximum_mcse",
            "maximum_interval_width",
            "maximum_absolute_change",
            "maximum_relative_change",
        ):
            if _finite(getattr(self, name), name=name) <= 0.0:
                raise UncertaintyValidationError(f"{name} must be positive")
        invalid_rate = _finite(self.maximum_invalid_rate, name="maximum_invalid_rate")
        if invalid_rate < 0.0 or invalid_rate >= 1.0:
            raise UncertaintyValidationError(
                "maximum_invalid_rate must be in [0, 1)"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": CONVERGENCE_SCHEMA_VERSION,
            "block_size": self.block_size,
            "minimum_retained_seeds": self.minimum_retained_seeds,
            "maximum_mcse": self.maximum_mcse,
            "maximum_interval_width": self.maximum_interval_width,
            "maximum_absolute_change": self.maximum_absolute_change,
            "maximum_relative_change": self.maximum_relative_change,
            "maximum_invalid_rate": self.maximum_invalid_rate,
            "consecutive_passing_checkpoints": self.consecutive_passing_checkpoints,
            "sensitivity_instability_allowed": False,
            "outcome_dependent_seed_exclusion_allowed": False,
            "required_uncertainty_component_handling": "FAIL_CLOSED",
        }


@dataclass(frozen=True, slots=True)
class ConvergenceCheckpoint:
    completed_realization_count: int
    retained_seed_count: int
    cumulative_point_estimate: float | None
    cumulative_monte_carlo_standard_error: float | None
    interval_width: float | None
    absolute_change: float | None
    relative_change: float | None
    invalid_count: int
    rejected_count: int
    excluded_count: int
    sensitivity_instability: bool
    status: ConvergenceStatus
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "completed_realization_count",
            "retained_seed_count",
            "invalid_count",
            "rejected_count",
            "excluded_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool) or value < 0:
                raise UncertaintyValidationError(
                    f"{name} must be a non-negative integer"
                )
        if self.retained_seed_count + self.invalid_count != self.completed_realization_count:
            raise UncertaintyValidationError(
                "retained and invalid counts must equal completed realizations"
            )
        if type(self.sensitivity_instability) is not bool:
            raise TypeError("sensitivity_instability must be boolean")
        if type(self.status) is not ConvergenceStatus:
            raise TypeError("checkpoint status must be ConvergenceStatus")
        if type(self.blockers) is not tuple or any(
            type(item) is not str or not item for item in self.blockers
        ):
            raise TypeError("checkpoint blockers must be an exact tuple of text")
        if len(self.blockers) != len(set(self.blockers)):
            raise UncertaintyValidationError("checkpoint blockers must be unique")
        if self.status is ConvergenceStatus.CONVERGED and self.blockers:
            raise UncertaintyValidationError(
                "a converged checkpoint cannot carry blockers"
            )
        if self.status is not ConvergenceStatus.CONVERGED and not self.blockers:
            raise UncertaintyValidationError(
                "a non-converged checkpoint requires an explicit blocker"
            )
        numeric = (
            "cumulative_point_estimate",
            "cumulative_monte_carlo_standard_error",
            "interval_width",
            "absolute_change",
            "relative_change",
        )
        for name in numeric:
            value = getattr(self, name)
            if value is not None:
                _finite(value, name=f"checkpoint {name}")
        if self.retained_seed_count:
            if any(
                getattr(self, name) is None
                for name in (
                    "cumulative_point_estimate",
                    "cumulative_monte_carlo_standard_error",
                    "interval_width",
                )
            ):
                raise UncertaintyValidationError(
                    "retained realizations require estimate, MCSE, and interval width"
                )
        elif any(
            getattr(self, name) is not None
            for name in (
                "cumulative_point_estimate",
                "cumulative_monte_carlo_standard_error",
                "interval_width",
                "absolute_change",
                "relative_change",
            )
        ):
            raise UncertaintyValidationError(
                "a checkpoint without retained realizations cannot report diagnostics"
            )
        if (
            self.cumulative_monte_carlo_standard_error is not None
            and self.cumulative_monte_carlo_standard_error < 0.0
        ) or (self.interval_width is not None and self.interval_width < 0.0):
            raise UncertaintyValidationError("checkpoint precision cannot be negative")
        if self.absolute_change is not None and self.absolute_change < 0.0:
            raise UncertaintyValidationError("absolute_change cannot be negative")
        if self.relative_change is not None and self.relative_change < 0.0:
            raise UncertaintyValidationError("relative_change cannot be negative")

    def snapshot(self) -> dict[str, object]:
        return {
            "completed_realization_count": self.completed_realization_count,
            "retained_seed_count": self.retained_seed_count,
            "cumulative_point_estimate": self.cumulative_point_estimate,
            "cumulative_monte_carlo_standard_error": (
                self.cumulative_monte_carlo_standard_error
            ),
            "interval_width": self.interval_width,
            "absolute_change_from_previous": self.absolute_change,
            "relative_change_from_previous": self.relative_change,
            "invalid_count": self.invalid_count,
            "rejected_count": self.rejected_count,
            "excluded_count": self.excluded_count,
            "sensitivity_instability": self.sensitivity_instability,
            "status": self.status.value,
            "blockers": list(self.blockers),
        }


def evaluate_blockwise_convergence(
    realizations: Sequence[UncertaintyRealization],
    *,
    expected_seeds: Sequence[int],
    rule: ConvergenceRule,
    rejected_count: int = 0,
    excluded_count: int = 0,
    sensitivity_instability: bool | Mapping[int, bool] = False,
    required_components_available: bool = True,
) -> tuple[ConvergenceCheckpoint, ...]:
    """Recompute cumulative seed diagnostics after deterministic blocks."""

    if type(rule) is not ConvergenceRule:
        raise TypeError("rule must be ConvergenceRule")
    ConvergenceRule.__post_init__(rule)
    if type(rejected_count) is not int or rejected_count < 0:
        raise UncertaintyValidationError("rejected_count must be non-negative")
    if type(excluded_count) is not int or excluded_count < 0:
        raise UncertaintyValidationError("excluded_count must be non-negative")
    if type(required_components_available) is not bool:
        raise TypeError("required_components_available must be boolean")
    expected = tuple(
        validate_seed(seed, name=f"expected_seeds[{index}]")
        for index, seed in enumerate(expected_seeds)
    )
    if expected != tuple(sorted(expected)) or len(expected) != len(set(expected)):
        raise UncertaintyValidationError("expected seeds must be unique and ascending")
    rows = tuple(realizations)
    if any(type(row) is not UncertaintyRealization for row in rows):
        raise TypeError("realizations must contain UncertaintyRealization values")
    for row in rows:
        UncertaintyRealization.__post_init__(row)
    if len(rows) > len(expected):
        raise UncertaintyValidationError("more realizations than declared seeds")
    if tuple(row.identity.seed for row in rows) != expected[: len(rows)]:
        raise UncertaintyValidationError(
            "completed realizations must be a prefix of the fixed seed set"
        )
    if len({_fixed_analysis_identity(row.identity) for row in rows}) > 1:
        raise UncertaintyValidationError(
            "seed convergence requires fixed parameter, population, rate, "
            "scenario, and estimand identities"
        )
    if isinstance(sensitivity_instability, Mapping):
        instability_by_count = dict(sensitivity_instability)
        if any(
            type(key) is not int
            or isinstance(key, bool)
            or key <= 0
            or type(value) is not bool
            for key, value in instability_by_count.items()
        ):
            raise UncertaintyValidationError(
                "sensitivity-instability checkpoints require positive integer "
                "keys and boolean values"
            )
    elif type(sensitivity_instability) is bool:
        instability_by_count = {}
    else:
        raise TypeError("sensitivity_instability must be bool or mapping")
    block_ends = list(range(rule.block_size, len(rows) + 1, rule.block_size))
    if rows and (not block_ends or block_ends[-1] != len(rows)):
        block_ends.append(len(rows))
    if isinstance(sensitivity_instability, Mapping) and not set(
        instability_by_count
    ).issubset(block_ends):
        raise UncertaintyValidationError(
            "sensitivity-instability keys must name emitted checkpoints"
        )
    checkpoints: list[ConvergenceCheckpoint] = []
    previous: float | None = None
    passing_streak = 0
    for end in block_ends:
        prefix = rows[:end]
        valid = [row for row in prefix if row.valid]
        invalid_count = len(prefix) - len(valid)
        estimate = mcse = width = None
        if valid:
            values = np.asarray([float(row.estimate) for row in valid])
            estimate = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            mcse = sd / sqrt(len(values))
            width = 2.0 * NORMAL_95_Z * mcse
        absolute = (
            abs(estimate - previous)
            if estimate is not None and previous is not None
            else None
        )
        relative = None
        if absolute is not None and previous is not None:
            relative = 0.0 if absolute == 0.0 else (
                absolute / abs(previous) if previous != 0.0 else None
            )
        unstable = (
            bool(instability_by_count.get(end, False))
            if isinstance(sensitivity_instability, Mapping)
            else bool(sensitivity_instability)
        )
        blockers: list[str] = []
        attempted = end + rejected_count + excluded_count
        invalid_rate = (
            (invalid_count + rejected_count + excluded_count) / attempted
            if attempted
            else 1.0
        )
        if unstable:
            blockers.append("sensitivity_instability")
        if invalid_rate > rule.maximum_invalid_rate:
            blockers.append("invalid_run_rate")
        if excluded_count:
            blockers.append("seed_exclusion_prohibited")
        if len(valid) < rule.minimum_retained_seeds:
            blockers.append("minimum_retained_seed_count")
        if mcse is None or mcse > rule.maximum_mcse:
            blockers.append("monte_carlo_standard_error")
        if width is None or width > rule.maximum_interval_width:
            blockers.append("interval_width")
        if absolute is None or absolute > rule.maximum_absolute_change:
            blockers.append("absolute_block_change")
        if relative is None or relative > rule.maximum_relative_change:
            blockers.append("relative_block_change")
        if not required_components_available:
            blockers.append("required_uncertainty_component_unavailable")
        if end % rule.block_size:
            blockers.append("incomplete_deterministic_block")
        precision_only = set(blockers).issubset(
            {"monte_carlo_standard_error", "interval_width"}
        )
        if unstable or invalid_rate > rule.maximum_invalid_rate or excluded_count:
            status = ConvergenceStatus.UNSTABLE
            passing_streak = 0
        elif precision_only and blockers:
            status = ConvergenceStatus.INSUFFICIENT_PRECISION
            passing_streak = 0
        elif blockers:
            status = ConvergenceStatus.NON_CONVERGED
            passing_streak = 0
        else:
            passing_streak += 1
            if passing_streak >= rule.consecutive_passing_checkpoints:
                status = ConvergenceStatus.CONVERGED
            else:
                status = ConvergenceStatus.NON_CONVERGED
                blockers.append("consecutive_passing_checkpoints")
        checkpoints.append(
            ConvergenceCheckpoint(
                completed_realization_count=end,
                retained_seed_count=len(valid),
                cumulative_point_estimate=estimate,
                cumulative_monte_carlo_standard_error=mcse,
                interval_width=width,
                absolute_change=absolute,
                relative_change=relative,
                invalid_count=invalid_count,
                rejected_count=rejected_count,
                excluded_count=excluded_count,
                sensitivity_instability=unstable,
                status=status,
                blockers=tuple(blockers),
            )
        )
        ConvergenceCheckpoint.__post_init__(checkpoints[-1])
        previous = estimate
    return tuple(checkpoints)


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    seed_only_variance: float | None
    between_parameter_variance: float | None
    between_population_variance: float | None
    between_rate_variance: float | None
    residual_or_interaction_variance: float | None
    total_joint_variance: float | None
    method: str
    identifiable: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "seed_only_variance",
            "between_parameter_variance",
            "between_population_variance",
            "between_rate_variance",
            "residual_or_interaction_variance",
            "total_joint_variance",
        ):
            value = getattr(self, name)
            if value is not None and _finite(value, name=name) < 0.0:
                raise UncertaintyValidationError(
                    f"{name} must be non-negative when available"
                )
        if type(self.method) is not str or not self.method.strip():
            raise UncertaintyValidationError(
                "variance decomposition requires an explicit method"
            )
        if type(self.identifiable) is not bool:
            raise TypeError("variance identifiable must be boolean")
        if type(self.blockers) is not tuple or any(
            type(item) is not str or not item for item in self.blockers
        ):
            raise TypeError("variance blockers must be an exact tuple of text")
        if len(self.blockers) != len(set(self.blockers)):
            raise UncertaintyValidationError("variance blockers must be unique")
        components = (
            self.seed_only_variance,
            self.between_parameter_variance,
            self.between_population_variance,
            self.between_rate_variance,
            self.residual_or_interaction_variance,
        )
        if self.identifiable:
            if self.blockers or self.total_joint_variance is None or any(
                value is None for value in components
            ):
                raise UncertaintyValidationError(
                    "identifiable decomposition requires every component and no blocker"
                )
            if not np.isclose(
                sum(float(value) for value in components),
                self.total_joint_variance,
                rtol=1e-10,
                atol=1e-14,
            ):
                raise UncertaintyValidationError(
                    "identified variance components do not sum to total variance"
                )

    def snapshot(self) -> dict[str, object]:
        return {
            "seed_only_variance": self.seed_only_variance,
            "between_parameter_variance": self.between_parameter_variance,
            "between_population_variance": self.between_population_variance,
            "between_rate_variance": self.between_rate_variance,
            "residual_or_interaction_variance": (
                self.residual_or_interaction_variance
            ),
            "total_joint_variance": self.total_joint_variance,
            "method": self.method,
            "identifiable": self.identifiable,
            "blockers": list(self.blockers),
        }


def decompose_joint_uncertainty(
    realizations: Iterable[UncertaintyRealization],
) -> VarianceDecomposition:
    """Orthogonal finite-design ANOVA decomposition for a full factorial.

    This is a descriptive decomposition of the declared finite design, not a
    population variance estimator.  It is emitted only for a unique, balanced,
    complete seed × parameter × population × rate Cartesian product.  A factor
    with a single level is explicitly unavailable rather than assigned zero.
    """

    rows = tuple(realizations)
    if any(type(row) is not UncertaintyRealization for row in rows):
        raise TypeError("realizations must contain UncertaintyRealization values")
    for row in rows:
        UncertaintyRealization.__post_init__(row)
    if not rows:
        return VarianceDecomposition(
            None, None, None, None, None, None,
            "UNAVAILABLE_INCOMPLETE_OR_INVALID_REALIZATIONS",
            False,
            ("complete_valid_joint_design_required",),
        )
    estimands = {(row.identity.scenario_id, row.identity.primary_estimand_id) for row in rows}
    if len(estimands) != 1:
        raise UncertaintyValidationError(
            "joint decomposition cannot pool scenarios or estimands"
        )
    identity_sha256s = [row.identity.identity_sha256 for row in rows]
    if len(identity_sha256s) != len(set(identity_sha256s)):
        raise UncertaintyValidationError(
            "joint design contains duplicate realization identities"
        )
    # Within a seed and population replicate, parameter/rate draws must reuse
    # the same pre-treatment cohort and exact design-weight identity.  This is
    # the observable common-random-number boundary available to this layer.
    shared_random_state: dict[
        tuple[int, str, str, str], tuple[str, str]
    ] = {}
    for row in rows:
        state_key = (
            row.identity.seed,
            row.identity.population_design_id,
            row.identity.population_design_sha256,
            row.identity.population_replicate_id,
        )
        state = (
            row.identity.pretreatment_cohort_sha256,
            row.identity.population_weights_sha256,
        )
        previous_state = shared_random_state.setdefault(state_key, state)
        if previous_state != state:
            raise UncertaintyValidationError(
                "joint design changed the pre-treatment cohort or population "
                "weights within a seed and population replicate"
            )
    if any(not row.valid for row in rows):
        return VarianceDecomposition(
            None, None, None, None, None, None,
            "UNAVAILABLE_INCOMPLETE_OR_INVALID_REALIZATIONS",
            False,
            ("complete_valid_joint_design_required",),
        )
    keys = [
        (
            row.identity.seed,
            (
                row.identity.parameter_draw_id,
                row.identity.parameter_draw_sha256,
            ),
            (
                row.identity.population_design_id,
                row.identity.population_design_sha256,
                row.identity.population_replicate_id,
            ),
            (
                row.identity.monetary_rate_draw_id,
                row.identity.monetary_rate_basis_id,
                row.identity.monetary_rate_basis_sha256,
            ),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise UncertaintyValidationError("joint design contains duplicate cells")
    levels = [tuple(sorted({key[index] for key in keys}, key=str)) for index in range(4)]
    expected_count = int(np.prod([len(items) for items in levels]))
    values = np.asarray([float(row.estimate) for row in rows], dtype=np.float64)
    total = float(np.mean((values - float(values.mean())) ** 2))
    if expected_count != len(rows):
        return VarianceDecomposition(
            None, None, None, None, None, total,
            "TOTAL_FINITE_DESIGN_VARIANCE_ONLY_UNBALANCED_OR_INCOMPLETE",
            False,
            ("balanced_complete_factorial_required",),
        )
    grand = float(values.mean())
    components: list[float | None] = []
    identified_sum = 0.0
    for factor_index, factor_levels in enumerate(levels):
        if len(factor_levels) < 2:
            components.append(None)
            continue
        component = 0.0
        for level in factor_levels:
            selected = np.asarray(
                [value for value, key in zip(values, keys, strict=True) if key[factor_index] == level]
            )
            component += len(selected) * (float(selected.mean()) - grand) ** 2
        component /= len(values)
        component = max(0.0, float(component))
        components.append(component)
        identified_sum += component
    blockers = tuple(
        name
        for name, component in zip(
            (
                "seed_component_single_level",
                "parameter_component_single_level",
                "population_component_single_level",
                "rate_component_single_level",
            ),
            components,
            strict=True,
        )
        if component is None
    )
    residual = None if blockers else max(0.0, total - identified_sum)
    return VarianceDecomposition(
        seed_only_variance=components[0],
        between_parameter_variance=components[1],
        between_population_variance=components[2],
        between_rate_variance=components[3],
        residual_or_interaction_variance=residual,
        total_joint_variance=total,
        method=(
            "ORTHOGONAL_FINITE_FULL_FACTORIAL_ANOVA_SUM_OF_SQUARES_DIVIDED_BY_N_V1"
        ),
        identifiable=not blockers,
        blockers=blockers,
    )


@dataclass(frozen=True, slots=True)
class UncertaintyComponentStatus:
    source: str
    availability: UncertaintyAvailability
    variance: float | None
    method: str | None
    blocker: str | None

    def __post_init__(self) -> None:
        _identifier(self.source, name="uncertainty source")
        if type(self.availability) is not UncertaintyAvailability:
            raise TypeError("availability must be UncertaintyAvailability")
        if self.availability is UncertaintyAvailability.QUANTIFIED:
            if self.variance is None or _finite(self.variance, name="variance") < 0.0:
                raise UncertaintyValidationError(
                    "quantified component requires non-negative variance"
                )
            if type(self.method) is not str or not self.method.strip():
                raise UncertaintyValidationError(
                    "quantified component requires a method"
                )
            if self.blocker is not None:
                raise UncertaintyValidationError(
                    "quantified component cannot carry a blocker"
                )
        else:
            if self.variance is not None:
                raise UncertaintyValidationError(
                    "unavailable components must not be assigned zero variance"
                )
            if type(self.blocker) is not str or not self.blocker.strip():
                raise UncertaintyValidationError(
                    "unavailable component requires an explicit blocker"
                )

    def snapshot(self) -> dict[str, object]:
        return {
            "source": self.source,
            "availability": self.availability.value,
            "variance": self.variance,
            "method": self.method,
            "blocker": self.blocker,
        }


def final_sufficiency_judgment(
    *,
    convergence_status: ConvergenceStatus,
    components: Sequence[UncertaintyComponentStatus],
    required_sources: Sequence[str] = (
        "seed",
        "parameter",
        "monetary_rate",
        "population",
        "combined",
    ),
) -> dict[str, object]:
    """Return a separate fail-closed scientific sufficiency judgment."""

    if type(convergence_status) is not ConvergenceStatus:
        raise TypeError("convergence_status must be ConvergenceStatus")
    declared_components = tuple(components)
    if any(
        type(component) is not UncertaintyComponentStatus
        for component in declared_components
    ):
        raise TypeError(
            "components must contain UncertaintyComponentStatus values"
        )
    for component in declared_components:
        UncertaintyComponentStatus.__post_init__(component)
    if type(required_sources) not in {tuple, list} or any(
        type(source) is not str or not source for source in required_sources
    ):
        raise TypeError("required_sources must be a sequence of non-empty text")
    if len(required_sources) != len(set(required_sources)):
        raise UncertaintyValidationError("required_sources must be unique")
    by_source = {component.source: component for component in declared_components}
    if len(by_source) != len(declared_components):
        raise UncertaintyValidationError(
            "uncertainty component sources must be unique"
        )
    missing = [source for source in required_sources if source not in by_source]
    unavailable = [
        source
        for source in required_sources
        if source in by_source
        and by_source[source].availability is not UncertaintyAvailability.QUANTIFIED
    ]
    blockers = [f"uncertainty_component_missing:{source}" for source in missing]
    blockers.extend(
        f"uncertainty_component_unavailable:{source}" for source in unavailable
    )
    if convergence_status is not ConvergenceStatus.CONVERGED:
        blockers.append(f"convergence:{convergence_status.value}")
    return {
        "sufficient": not blockers,
        "judgment": "SUFFICIENT" if not blockers else "INSUFFICIENT",
        "campaign_ready": False,
        "blockers": blockers,
    }


__all__ = [
    "CONVERGENCE_SCHEMA_VERSION",
    "UNCERTAINTY_SCHEMA_VERSION",
    "ConvergenceCheckpoint",
    "ConvergenceRule",
    "ConvergenceStatus",
    "LoadedParameterUncertaintyDesign",
    "ParameterDeclaration",
    "ParameterDraw",
    "ParameterProvenanceStatus",
    "ParameterUncertaintyDesign",
    "RealizationIdentity",
    "SeedUncertaintySummary",
    "UncertaintyAvailability",
    "UncertaintyComponentStatus",
    "UncertaintyRealization",
    "UncertaintyValidationError",
    "VarianceDecomposition",
    "canonical_sha256",
    "decompose_joint_uncertainty",
    "evaluate_blockwise_convergence",
    "final_sufficiency_judgment",
    "generate_parameter_draws",
    "load_parameter_uncertainty_design",
    "parameter_design_from_snapshot",
    "summarize_seed_uncertainty",
    "verify_loaded_parameter_uncertainty_design",
]
