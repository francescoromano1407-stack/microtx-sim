from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from hashlib import sha256
import json
import re
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..types import HarmDimension, Motive, SpendSegment


TRAIT_NAMES: Final[tuple[str, ...]] = (
    "impulsivity",
    "reward_sensitivity",
    "social_susceptibility",
    "loss_aversion",
    "financial_literacy",
    "self_control",
)

_MONEY_COLUMNS: Final[tuple[str, ...]] = (
    "monthly_disposable_income_cents",
    "liquidity_cents",
    "credit_limit_cents",
    "allowance_cents",
    "household_liquidity_cents",
)

_SHA256_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

SOURCE_RECORDED_SEX_FEMALE: Final[str] = "FEMALE"
SOURCE_RECORDED_SEX_MALE: Final[str] = "MALE"
SOURCE_RECORDED_SEX_UNAVAILABLE: Final[str] = ""
SOURCE_RECORDED_SEX_DTYPE: Final[np.dtype[np.str_]] = np.dtype("<U6")
_SOURCE_RECORDED_SEX_VALUES: Final[tuple[str, str]] = (
    SOURCE_RECORDED_SEX_FEMALE,
    SOURCE_RECORDED_SEX_MALE,
)


def _validate_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace")
    return value


def _validate_exact_rational(
    value: object,
    *,
    name: str,
    allow_zero: bool,
) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a (numerator, denominator) tuple")
    numerator, denominator = value
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
    ):
        raise TypeError(f"{name} numerator and denominator must be Python integers")
    if numerator < 0 or (numerator == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} numerator must be {qualifier}")
    if denominator <= 0:
        raise ValueError(f"{name} denominator must be positive")
    reduced = Fraction(numerator, denominator)
    if (reduced.numerator, reduced.denominator) != value:
        raise ValueError(f"{name} must be in lowest terms")
    return value


def source_recorded_sex_sha256(values: NDArray[np.str_]) -> str:
    """Hash one exact synthetic-player allocation of source-recorded sex.

    Empty strings mean that the cited source does not cover that runtime row;
    they must never be interpreted as a third sex or as gender identity.
    """

    if type(values) is not np.ndarray:
        raise TypeError("source-recorded sex values must be an exact NumPy array")
    if values.ndim != 1 or values.dtype != SOURCE_RECORDED_SEX_DTYPE:
        raise TypeError(
            "source-recorded sex values must be a one-dimensional <U6 array"
        )
    allowed = (*_SOURCE_RECORDED_SEX_VALUES, SOURCE_RECORDED_SEX_UNAVAILABLE)
    if not np.all(np.isin(values, allowed)):
        raise ValueError(
            "source-recorded sex values must be FEMALE, MALE, or empty outside "
            "the declared source scope"
        )
    digest = sha256(b"microtx-sim.source-recorded-sex.v1\0")
    digest.update(values.size.to_bytes(8, "little", signed=False))
    for raw_value in values:
        encoded = str(raw_value).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def source_recorded_sex_derivation_input_sha256(
    player_id: NDArray[np.int64],
    age_years: NDArray[np.int16],
    jurisdiction: NDArray[np.int16],
    cell_index: NDArray[np.int32],
) -> str:
    """Hash every ordered runtime column used to derive a sex allocation.

    The projection metadata and evidence bundle have their own content
    addresses. This digest closes the remaining provenance gap by binding the
    per-player inputs that can change age-band membership or deterministic
    ranking without changing the legacy projection-cell assignment.
    """

    arrays = (
        (player_id, np.dtype(np.int64), "player_id"),
        (age_years, np.dtype(np.int16), "age_years"),
        (jurisdiction, np.dtype(np.int16), "jurisdiction"),
        (cell_index, np.dtype(np.int32), "cell_index"),
    )
    expected_shape: tuple[int, ...] | None = None
    for values, dtype, name in arrays:
        if type(values) is not np.ndarray:
            raise TypeError(f"{name} must be an exact NumPy array")
        if values.ndim != 1 or values.dtype != dtype:
            raise TypeError(f"{name} must be a one-dimensional {dtype} array")
        if expected_shape is None:
            expected_shape = values.shape
        elif values.shape != expected_shape:
            raise ValueError("source-recorded sex derivation inputs must align")

    digest = sha256(b"microtx-sim.source-recorded-sex-derivation-input.v1\0")
    assert expected_shape is not None
    digest.update(expected_shape[0].to_bytes(8, "little", signed=False))
    digest.update(np.asarray(player_id, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(age_years, dtype="<i2").tobytes(order="C"))
    digest.update(np.asarray(jurisdiction, dtype="<i2").tobytes(order="C"))
    digest.update(np.asarray(cell_index, dtype="<i4").tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectedPopulationSexBinding:
    """Typed source and scope for an aggregate-informed synthetic sex field.

    The field records the source's binary ``FEMALE``/``MALE`` categories. It
    is deliberately named *sex*, not gender, and makes no gender-identity
    inference. Values outside the declared jurisdiction and age interval are
    absent rather than imputed from an unrelated population.
    """

    source_id: str
    evidence_bundle_id: str
    evidence_bundle_sha256: str
    population_weights_sha256: str
    jurisdiction_code: str
    age_min_inclusive: int
    age_max_inclusive: int
    assignment_method: str
    derivation_input_sha256: str
    sex_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "evidence_bundle_id",
            "jurisdiction_code",
            "assignment_method",
        ):
            _validate_identifier(getattr(self, name), name=name)
        for name in (
            "evidence_bundle_sha256",
            "population_weights_sha256",
            "derivation_input_sha256",
            "sex_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
        for name in ("age_min_inclusive", "age_max_inclusive"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a Python integer")
        if not 0 <= self.age_min_inclusive <= self.age_max_inclusive < 32_768:
            raise ValueError(
                "source-recorded sex age interval must be non-empty and fit int16"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "field": "sex",
            "source_categories": list(_SOURCE_RECORDED_SEX_VALUES),
            "source_id": self.source_id,
            "evidence_bundle_id": self.evidence_bundle_id,
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "population_weights_sha256": self.population_weights_sha256,
            "jurisdiction_code": self.jurisdiction_code,
            "age_min_inclusive": self.age_min_inclusive,
            "age_max_inclusive": self.age_max_inclusive,
            "assignment_method": self.assignment_method,
            "derivation_input_sha256": self.derivation_input_sha256,
            "sex_sha256": self.sex_sha256,
            "interpretation": "source-recorded sex; not inferred gender identity",
            "out_of_scope_value": SOURCE_RECORDED_SEX_UNAVAILABLE,
        }


@dataclass(frozen=True, slots=True)
class ProjectedPopulationCellMetadata:
    """Immutable runtime meaning and exact weights for one projected cell.

    The income interval is deliberately named for the runtime player column. It
    is not a claim that a source household-income band has the same estimand.
    Any conversion from source evidence belongs in the upstream synthesis plan.
    """

    cell_id: str
    jurisdiction_code: str
    jurisdiction_index: int
    age_min_inclusive: int
    age_max_exclusive: int
    monthly_disposable_income_band_id: str
    monthly_disposable_income_min_cents: int
    monthly_disposable_income_max_cents_exclusive: int
    household_type: str
    modeled_players_per_household: int
    baseline_gamer: bool
    baseline_ever_payer: bool
    global_mass: tuple[int, int]
    analysis_weight: tuple[int, int]

    def __post_init__(self) -> None:
        _validate_identifier(self.cell_id, name="cell_id")
        _validate_identifier(self.jurisdiction_code, name="jurisdiction_code")
        _validate_identifier(
            self.monthly_disposable_income_band_id,
            name="monthly_disposable_income_band_id",
        )
        _validate_identifier(self.household_type, name="household_type")
        for name in (
            "jurisdiction_index",
            "age_min_inclusive",
            "age_max_exclusive",
            "monthly_disposable_income_min_cents",
            "monthly_disposable_income_max_cents_exclusive",
            "modeled_players_per_household",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a Python integer")
        if self.jurisdiction_index < 0:
            raise ValueError("jurisdiction_index cannot be negative")
        if not 0 <= self.age_min_inclusive < self.age_max_exclusive <= 32_768:
            raise ValueError("age interval must be non-empty and fit int16 ages")
        if (
            self.monthly_disposable_income_min_cents < 0
            or self.monthly_disposable_income_max_cents_exclusive
            <= self.monthly_disposable_income_min_cents
            or self.monthly_disposable_income_max_cents_exclusive
            > np.iinfo(np.int64).max
        ):
            raise ValueError(
                "monthly disposable income interval must be non-empty, non-negative, "
                "and fit int64 cents"
            )
        if self.modeled_players_per_household <= 0:
            raise ValueError("modeled_players_per_household must be positive")
        maximum_household_income = (
            self.monthly_disposable_income_max_cents_exclusive - 1
        ) * self.modeled_players_per_household
        if maximum_household_income > 2**53:
            raise ValueError(
                "runtime income upper bound times modeled household size must be "
                "at most 2**53 cents for exact float64 household-resource input"
            )
        if not isinstance(self.baseline_gamer, bool):
            raise TypeError("baseline_gamer must be a bool")
        if not isinstance(self.baseline_ever_payer, bool):
            raise TypeError("baseline_ever_payer must be a bool")
        mass = _validate_exact_rational(
            self.global_mass,
            name="global_mass",
            allow_zero=True,
        )
        weight = _validate_exact_rational(
            self.analysis_weight,
            name="analysis_weight",
            allow_zero=True,
        )
        if mass[0] == 0 and weight != (0, 1):
            raise ValueError("zero-mass cells must have analysis_weight (0, 1)")
        if mass[0] > 0 and weight[0] == 0:
            raise ValueError("positive-mass cells must have positive analysis weight")

    def snapshot(self) -> dict[str, object]:
        """Return the canonical JSON-safe representation used by assignment hashes."""

        return {
            "age_max_exclusive": self.age_max_exclusive,
            "age_min_inclusive": self.age_min_inclusive,
            "analysis_weight": list(self.analysis_weight),
            "baseline_ever_payer": self.baseline_ever_payer,
            "baseline_gamer": self.baseline_gamer,
            "cell_id": self.cell_id,
            "global_mass": list(self.global_mass),
            "household_type": self.household_type,
            "jurisdiction_code": self.jurisdiction_code,
            "jurisdiction_index": self.jurisdiction_index,
            "modeled_players_per_household": self.modeled_players_per_household,
            "monthly_disposable_income_band_id": (
                self.monthly_disposable_income_band_id
            ),
            "monthly_disposable_income_max_cents_exclusive": (
                self.monthly_disposable_income_max_cents_exclusive
            ),
            "monthly_disposable_income_min_cents": (
                self.monthly_disposable_income_min_cents
            ),
        }


def projected_population_plan_sha256(
    projection_id: str,
    cells: tuple[ProjectedPopulationCellMetadata, ...],
) -> str:
    """Hash the exact content of one runtime population-projection plan."""

    _validate_identifier(projection_id, name="projection_id")
    if type(cells) is not tuple:
        raise TypeError("cells must be an immutable tuple")
    if not cells:
        raise ValueError("cells must be a non-empty tuple")
    if any(type(cell) is not ProjectedPopulationCellMetadata for cell in cells):
        raise TypeError("cells must contain ProjectedPopulationCellMetadata")
    cell_ids = tuple(cell.cell_id for cell in cells)
    if tuple(sorted(cell_ids)) != cell_ids:
        raise ValueError("cells must be in canonical cell_id order")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("cell_id values must be unique")
    total_mass = sum(
        (Fraction(*cell.global_mass) for cell in cells),
        start=Fraction(0, 1),
    )
    if total_mass != 1:
        raise ValueError("projected cell global masses must sum exactly to one")
    payload = {
        "cells": [cell.snapshot() for cell in cells],
        "projection_id": projection_id,
        "schema_version": 1,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = sha256(b"microtx-sim.projected-population-plan.v1\0")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectedPopulationMetadata:
    """Content-addressed runtime plan and ordered projected-cell metadata.

    ``projection_sha256`` is derived from ``projection_id`` and the exact cell
    metadata.  It is not a caller-provided digest of an upstream static design.
    A future adapter must bind that upstream design separately.
    """

    projection_id: str
    projection_sha256: str
    cells: tuple[ProjectedPopulationCellMetadata, ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.projection_id, name="projection_id")
        if not isinstance(self.projection_sha256, str):
            raise TypeError("projection_sha256 must be a string")
        if _SHA256_PATTERN.fullmatch(self.projection_sha256) is None:
            raise ValueError(
                "projection_sha256 must be a lowercase hexadecimal SHA-256"
            )
        expected_sha256 = projected_population_plan_sha256(
            self.projection_id,
            self.cells,
        )
        if self.projection_sha256 != expected_sha256:
            raise ValueError(
                "projection_sha256 does not match the exact runtime projection plan"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "cells": [cell.snapshot() for cell in self.cells],
            "projection_id": self.projection_id,
            "projection_sha256": self.projection_sha256,
            "schema_version": 1,
        }


def projected_population_assignment_sha256(
    metadata: ProjectedPopulationMetadata,
    player_id: NDArray[np.int64],
    cell_index: NDArray[np.int32],
    *,
    age_years: NDArray[np.int16] | None = None,
    jurisdiction: NDArray[np.int16] | None = None,
    sex: NDArray[np.str_] | None = None,
    sex_binding: ProjectedPopulationSexBinding | None = None,
) -> str:
    """Hash runtime projection metadata, ordered IDs, and bound assignments.

    The legacy digest recipe is retained byte-for-byte when no sex binding is
    present. A source-recorded sex vector selects a separate v2 domain and is
    accepted only together with its exact typed source binding.
    """

    if type(metadata) is not ProjectedPopulationMetadata:
        raise TypeError("metadata must be ProjectedPopulationMetadata")
    ids = np.asarray(player_id)
    indices = np.asarray(cell_index)
    if ids.ndim != 1 or ids.dtype != np.dtype(np.int64):
        raise TypeError("player_id must be a one-dimensional int64 array")
    if indices.ndim != 1 or indices.dtype != np.dtype(np.int32):
        raise TypeError("cell_index must be a one-dimensional int32 array")
    if ids.shape != indices.shape:
        raise ValueError("player_id and cell_index must have the same shape")
    if (sex is None) != (sex_binding is None):
        raise ValueError("sex and sex_binding must either both be present or both absent")
    if sex_binding is not None:
        if type(sex_binding) is not ProjectedPopulationSexBinding:
            raise TypeError(
                "sex_binding must be an exact ProjectedPopulationSexBinding"
            )
        ProjectedPopulationSexBinding.__post_init__(sex_binding)
        assert sex is not None
        observed_sex_sha256 = source_recorded_sex_sha256(sex)
        if sex.shape != ids.shape:
            raise ValueError("source-recorded sex must have one value per player")
        if observed_sex_sha256 != sex_binding.sex_sha256:
            raise ValueError("sex_sha256 does not match source-recorded sex values")
        if age_years is None or jurisdiction is None:
            raise ValueError(
                "age_years and jurisdiction are required with source-recorded sex"
            )
        observed_derivation_sha256 = (
            source_recorded_sex_derivation_input_sha256(
                ids,
                age_years,
                jurisdiction,
                indices,
            )
        )
        if observed_derivation_sha256 != sex_binding.derivation_input_sha256:
            raise ValueError(
                "source-recorded sex derivation inputs differ from their binding"
            )

    encoded_metadata = json.dumps(
        metadata.snapshot(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = sha256(
        b"microtx-sim.projected-population-assignment.v1\0"
        if sex_binding is None
        else b"microtx-sim.projected-population-assignment.v2\0"
    )
    digest.update(len(encoded_metadata).to_bytes(8, "little"))
    digest.update(encoded_metadata)
    digest.update(ids.size.to_bytes(8, "little"))
    digest.update(np.asarray(ids, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(indices, dtype="<i4").tobytes(order="C"))
    if sex_binding is not None:
        encoded_binding = json.dumps(
            sex_binding.snapshot(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(encoded_binding).to_bytes(8, "little"))
        digest.update(encoded_binding)
        assert sex is not None
        digest.update(np.asarray(sex, dtype=SOURCE_RECORDED_SEX_DTYPE).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProjectedPopulationAssignment:
    """Immutable per-player lookup into exact projected-population metadata."""

    metadata: ProjectedPopulationMetadata
    cell_index: NDArray[np.int32]
    assignment_sha256: str
    sex_binding: ProjectedPopulationSexBinding | None = None

    def __post_init__(self) -> None:
        if type(self.metadata) is not ProjectedPopulationMetadata:
            raise TypeError("metadata must be ProjectedPopulationMetadata")
        if not isinstance(self.cell_index, np.ndarray):
            raise TypeError("cell_index must be a NumPy array")
        if self.cell_index.ndim != 1 or self.cell_index.dtype != np.dtype(np.int32):
            raise TypeError("cell_index must be a one-dimensional int32 array")
        if self.cell_index.size and (
            np.any(self.cell_index < 0)
            or np.any(self.cell_index >= len(self.metadata.cells))
        ):
            raise ValueError("cell_index contains an unknown projected cell")
        if not isinstance(self.assignment_sha256, str):
            raise TypeError("assignment_sha256 must be a string")
        if _SHA256_PATTERN.fullmatch(self.assignment_sha256) is None:
            raise ValueError(
                "assignment_sha256 must be a lowercase hexadecimal SHA-256"
            )
        if self.sex_binding is not None:
            if type(self.sex_binding) is not ProjectedPopulationSexBinding:
                raise TypeError(
                    "sex_binding must be an exact ProjectedPopulationSexBinding"
                )
            ProjectedPopulationSexBinding.__post_init__(self.sex_binding)
        immutable_index = np.array(self.cell_index, dtype=np.int32, copy=True)
        immutable_index.flags.writeable = False
        object.__setattr__(self, "cell_index", immutable_index)

    def cell_for_player_position(self, position: int) -> ProjectedPopulationCellMetadata:
        if isinstance(position, bool) or not isinstance(position, (int, np.integer)):
            raise TypeError("position must be an integer")
        if not 0 <= int(position) < self.cell_index.size:
            raise IndexError("player position is outside the assignment")
        return self.metadata.cells[int(self.cell_index[int(position)])]


@dataclass(frozen=True, slots=True)
class PlayerTable:
    """Structure-of-arrays state for a heterogeneous player population.

    The dataclass is frozen so columns cannot accidentally be replaced while a
    simulation is running.  Dynamic columns (for example ``harm_state``) remain
    mutable NumPy arrays.  ``baseline_vulnerability`` is additionally copied
    and write-protected because it is a pre-treatment covariate in the causal
    design.

    Jurisdictions are integer codes into ``jurisdiction_codes``.  Motive and
    harm columns use the ordering of :class:`~microtx_sim.types.Motive` and
    :class:`~microtx_sim.types.HarmDimension`, respectively.
    """

    player_id: NDArray[np.int64]
    age_years: NDArray[np.int16]
    jurisdiction: NDArray[np.int16]
    household_id: NDArray[np.int64]
    is_minor: NDArray[np.bool_]

    monthly_disposable_income_cents: NDArray[np.int64]
    liquidity_cents: NDArray[np.int64]
    credit_limit_cents: NDArray[np.int64]
    allowance_cents: NDArray[np.int64]
    household_liquidity_cents: NDArray[np.int64]

    has_stored_payment_access: NDArray[np.bool_]
    guardian_supervision: NDArray[np.float32]
    guardian_consent: NDArray[np.bool_]

    traits: NDArray[np.float32]
    motive_weights: NDArray[np.float32]
    baseline_vulnerability: NDArray[np.float32]
    harm_state: NDArray[np.float32]

    current_game: NDArray[np.int32]
    awareness: NDArray[np.float32]

    jurisdiction_codes: tuple[str, ...]
    adult_age_by_jurisdiction: tuple[int, ...]
    projected_population: ProjectedPopulationAssignment | None = None
    sex: NDArray[np.str_] | None = None

    def __post_init__(self) -> None:
        if self.player_id.ndim != 1:
            raise ValueError("player_id must be one-dimensional")
        n_players = self.player_id.shape[0]

        one_dimensional = (
            "age_years",
            "jurisdiction",
            "household_id",
            "is_minor",
            *_MONEY_COLUMNS,
            "has_stored_payment_access",
            "guardian_supervision",
            "guardian_consent",
            "baseline_vulnerability",
            "current_game",
            "awareness",
        )
        for name in one_dimensional:
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != n_players:
                raise ValueError(f"{name} must have shape ({n_players},)")

        expected_dtypes: dict[str, np.dtype[object]] = {
            "player_id": np.dtype(np.int64),
            "age_years": np.dtype(np.int16),
            "jurisdiction": np.dtype(np.int16),
            "household_id": np.dtype(np.int64),
            "is_minor": np.dtype(np.bool_),
            **{name: np.dtype(np.int64) for name in _MONEY_COLUMNS},
            "has_stored_payment_access": np.dtype(np.bool_),
            "guardian_supervision": np.dtype(np.float32),
            "guardian_consent": np.dtype(np.bool_),
            "traits": np.dtype(np.float32),
            "motive_weights": np.dtype(np.float32),
            "baseline_vulnerability": np.dtype(np.float32),
            "harm_state": np.dtype(np.float32),
            "current_game": np.dtype(np.int32),
            "awareness": np.dtype(np.float32),
        }
        for name, dtype in expected_dtypes.items():
            if getattr(self, name).dtype != dtype:
                raise TypeError(f"{name} must use dtype {dtype}")

        if self.sex is not None:
            if type(self.sex) is not np.ndarray:
                raise TypeError("sex must be an exact NumPy array when present")
            if self.sex.ndim != 1 or self.sex.shape != (n_players,):
                raise ValueError(f"sex must have shape ({n_players},)")
            if self.sex.dtype != SOURCE_RECORDED_SEX_DTYPE:
                raise TypeError("sex must use dtype <U6")
            source_recorded_sex_sha256(self.sex)

        if self.traits.shape != (n_players, len(TRAIT_NAMES)):
            raise ValueError(
                f"traits must have shape ({n_players}, {len(TRAIT_NAMES)})"
            )
        if self.motive_weights.shape != (n_players, len(Motive)):
            raise ValueError(
                f"motive_weights must have shape ({n_players}, {len(Motive)})"
            )
        if self.harm_state.shape != (n_players, len(HarmDimension)):
            raise ValueError(
                f"harm_state must have shape ({n_players}, {len(HarmDimension)})"
            )

        if len(self.jurisdiction_codes) != len(self.adult_age_by_jurisdiction):
            raise ValueError("jurisdiction metadata lengths differ")
        if len(set(self.jurisdiction_codes)) != len(self.jurisdiction_codes):
            raise ValueError("jurisdiction codes must be unique")
        if n_players:
            if np.unique(self.player_id).size != n_players:
                raise ValueError("player_id values must be unique")
            if np.any(self.age_years < 0):
                raise ValueError("age_years cannot be negative")
            if np.any(self.household_id < 0):
                raise ValueError("household_id cannot be negative")
            if np.any(self.jurisdiction < 0) or np.any(
                self.jurisdiction >= len(self.jurisdiction_codes)
            ):
                raise ValueError("jurisdiction contains an unknown code")

            adult_ages = np.asarray(self.adult_age_by_jurisdiction, dtype=np.int16)
            expected_minor = self.age_years < adult_ages[self.jurisdiction]
            if not np.array_equal(self.is_minor, expected_minor):
                raise ValueError("is_minor is inconsistent with age and jurisdiction")

        for name in _MONEY_COLUMNS:
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "guardian_supervision",
            "traits",
            "motive_weights",
            "baseline_vulnerability",
            "awareness",
        ):
            value = getattr(self, name)
            if not np.all(np.isfinite(value)) or np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError(f"{name} values must be finite and in [0, 1]")
        if not np.all(np.isfinite(self.harm_state)) or np.any(self.harm_state < 0.0):
            raise ValueError("harm_state must contain finite, non-negative values")
        if n_players and not np.allclose(
            self.motive_weights.sum(axis=1), 1.0, atol=2e-6
        ):
            raise ValueError("every motive_weights row must sum to one")
        if np.any(self.allowance_cents[~self.is_minor] != 0):
            raise ValueError("allowance_cents is reserved for minors")
        if np.any(self.credit_limit_cents[self.is_minor] != 0):
            raise ValueError(
                "minors cannot own credit; stored-card access is represented separately"
            )
        if np.any(self.guardian_supervision[~self.is_minor] != 0.0):
            raise ValueError("guardian_supervision is reserved for minors")
        if np.any(self.guardian_consent[~self.is_minor]):
            raise ValueError("guardian_consent is reserved for minors")
        if np.any(self.current_game < -1):
            raise ValueError("current_game must be -1 (none) or a non-negative game id")

        if self.projected_population is not None:
            _validate_player_projection(self, self.projected_population)
        elif self.sex is not None:
            raise ValueError(
                "source-recorded sex requires projected-population lineage"
            )

        immutable_baseline = np.array(
            self.baseline_vulnerability, dtype=np.float32, copy=True
        )
        immutable_baseline.flags.writeable = False
        object.__setattr__(self, "baseline_vulnerability", immutable_baseline)
        if self.sex is not None:
            immutable_sex = np.array(
                self.sex,
                dtype=SOURCE_RECORDED_SEX_DTYPE,
                copy=True,
            )
            immutable_sex.flags.writeable = False
            object.__setattr__(self, "sex", immutable_sex)

    def __len__(self) -> int:
        return int(self.player_id.size)

    def trait(self, name: str) -> NDArray[np.float32]:
        """Return a zero-copy view of a named continuous trait."""

        try:
            index = TRAIT_NAMES.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown trait: {name}") from exc
        return self.traits[:, index]

    def motive(self, motive: Motive | int) -> NDArray[np.float32]:
        """Return a zero-copy view of one motive weight."""

        return self.motive_weights[:, int(motive)]

    def classify_spending(
        self,
        spend_cents: NDArray[np.integer] | list[int],
        *,
        income_cents: NDArray[np.integer] | list[int] | None = None,
        whale_quantile: float = 0.99,
        whale_income_share: float = 0.10,
    ) -> NDArray[np.str_]:
        """Classify observed period spending without creating player types.

        If ``income_cents`` is omitted, monthly disposable income is used, so
        ``spend_cents`` should then cover the same monthly period.
        """

        denominator = (
            self.monthly_disposable_income_cents
            if income_cents is None
            else income_cents
        )
        return classify_spend_segments(
            spend_cents,
            denominator,
            whale_quantile=whale_quantile,
            whale_income_share=whale_income_share,
        )

    @property
    def nbytes(self) -> int:
        """Memory owned by NumPy columns (metadata excluded)."""

        total = 0
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if isinstance(value, np.ndarray):
                total += value.nbytes
        if self.projected_population is not None:
            total += self.projected_population.cell_index.nbytes
        return total


def require_treatment_eligible_player_table(
    players: PlayerTable,
    *,
    operation: str,
) -> PlayerTable:
    """Reject structural point-zero fields before any behavioural mutation."""

    if not isinstance(players, PlayerTable):
        raise TypeError("players must be a PlayerTable")
    if not isinstance(operation, str) or not operation or operation.strip() != operation:
        raise ValueError("operation must be non-empty text without surrounding spaces")
    assignment = players.projected_population
    binding = (
        assignment.sex_binding
        if type(assignment) is ProjectedPopulationAssignment
        else None
    )
    if players.sex is not None or binding is not None:
        raise ValueError(
            f"{operation} rejects source-recorded sex: the current binding is "
            "structural point-zero evidence only and is not treatment-eligible"
        )
    return players


def _validate_player_projection(
    players: PlayerTable,
    assignment: ProjectedPopulationAssignment,
) -> None:
    if type(assignment) is not ProjectedPopulationAssignment:
        raise TypeError("projected_population must be ProjectedPopulationAssignment")
    if assignment.cell_index.shape != players.player_id.shape:
        raise ValueError("projected cell_index must have one value per player")
    expected_sha256 = projected_population_assignment_sha256(
        assignment.metadata,
        players.player_id,
        assignment.cell_index,
        age_years=players.age_years,
        jurisdiction=players.jurisdiction,
        sex=players.sex,
        sex_binding=assignment.sex_binding,
    )
    if assignment.assignment_sha256 != expected_sha256:
        raise ValueError("projected population assignment hash does not match PlayerTable")

    binding = assignment.sex_binding
    if (players.sex is None) != (binding is None):
        raise ValueError(
            "PlayerTable sex and projected-population sex binding differ"
        )
    if binding is not None:
        assert players.sex is not None
        observed_derivation_sha256 = (
            source_recorded_sex_derivation_input_sha256(
                players.player_id,
                players.age_years,
                players.jurisdiction,
                assignment.cell_index,
            )
        )
        if observed_derivation_sha256 != binding.derivation_input_sha256:
            raise ValueError(
                "source-recorded sex derivation inputs differ from their binding"
            )
        try:
            jurisdiction_index = players.jurisdiction_codes.index(
                binding.jurisdiction_code
            )
        except ValueError as exc:
            raise ValueError(
                "source-recorded sex binding has an unknown jurisdiction"
            ) from exc
        in_scope = (
            (players.jurisdiction == jurisdiction_index)
            & (players.age_years >= binding.age_min_inclusive)
            & (players.age_years <= binding.age_max_inclusive)
        )
        if np.any(
            ~np.isin(players.sex[in_scope], _SOURCE_RECORDED_SEX_VALUES)
        ):
            raise ValueError(
                "every player in the source-recorded sex scope must be FEMALE or MALE"
            )
        if np.any(players.sex[~in_scope] != SOURCE_RECORDED_SEX_UNAVAILABLE):
            raise ValueError(
                "source-recorded sex must be empty outside its declared source scope"
            )

    cells = assignment.metadata.cells
    counts = np.bincount(
        assignment.cell_index.astype(np.int64, copy=False),
        minlength=len(cells),
    )
    cell_groups: list[tuple[int, str, str]] = []
    group_specification: dict[
        tuple[int, str, str],
        tuple[int, int, int],
    ] = {}
    for index, (cell, raw_count) in enumerate(zip(cells, counts, strict=True)):
        count = int(raw_count)
        mass = Fraction(*cell.global_mass)
        weight = Fraction(*cell.analysis_weight)
        if mass == 0:
            if count != 0:
                raise ValueError("zero-mass projected cells cannot contain players")
            if weight != 0:
                raise ValueError("zero-mass projected cells must have zero analysis weight")
        else:
            if count == 0:
                raise ValueError("every positive-mass projected cell must be represented")
            if weight != mass / count:
                raise ValueError(
                    "projected cell analysis weight must equal global mass / sample count"
                )

        if not 0 <= cell.jurisdiction_index < len(players.jurisdiction_codes):
            raise ValueError("projected cell has an unknown jurisdiction index")
        if players.jurisdiction_codes[cell.jurisdiction_index] != cell.jurisdiction_code:
            raise ValueError(
                "projected cell jurisdiction code/index does not match PlayerTable"
            )

        key = (
            cell.jurisdiction_index,
            cell.monthly_disposable_income_band_id,
            cell.household_type,
        )
        specification = (
            cell.modeled_players_per_household,
            cell.monthly_disposable_income_min_cents,
            cell.monthly_disposable_income_max_cents_exclusive,
        )
        previous_specification = group_specification.setdefault(key, specification)
        if previous_specification != specification:
            raise ValueError(
                "a projected jurisdiction/income-band/household-type group must "
                "have one runtime income interval and modeled household size"
            )
        cell_groups.append(key)

        positions = assignment.cell_index == index
        if not count:
            continue
        if np.any(players.jurisdiction[positions] != cell.jurisdiction_index):
            raise ValueError("player jurisdiction is inconsistent with projected cell")
        if np.any(players.age_years[positions] < cell.age_min_inclusive) or np.any(
            players.age_years[positions] >= cell.age_max_exclusive
        ):
            raise ValueError("player age is outside its projected cell interval")
        incomes = players.monthly_disposable_income_cents[positions]
        if np.any(incomes < cell.monthly_disposable_income_min_cents) or np.any(
            incomes >= cell.monthly_disposable_income_max_cents_exclusive
        ):
            raise ValueError(
                "player monthly disposable income is outside its projected runtime interval"
            )

    if not len(players):
        return
    canonical_groups = {
        key: index for index, key in enumerate(sorted(set(cell_groups)))
    }
    cell_group_index = np.asarray(
        [canonical_groups[key] for key in cell_groups],
        dtype=np.int32,
    )
    cell_group_size = np.asarray(
        [
            group_specification[key][0]
            for key in sorted(canonical_groups, key=canonical_groups.__getitem__)
        ],
        dtype=np.int64,
    )
    player_group = cell_group_index[assignment.cell_index]
    order = np.argsort(players.household_id, kind="stable")
    ordered_household = players.household_id[order]
    ordered_group = player_group[order]
    same_household = ordered_household[1:] == ordered_household[:-1]
    if np.any(ordered_group[1:][same_household] != ordered_group[:-1][same_household]):
        raise ValueError(
            "a projected household cannot cross jurisdiction, income band, or household type"
        )

    household_start = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(~same_household).astype(np.int64) + 1,
        )
    )
    household_end = np.concatenate(
        (household_start[1:], np.asarray([len(players)], dtype=np.int64))
    )
    household_size = household_end - household_start
    household_group = ordered_group[household_start]
    allowed_size = cell_group_size[household_group]
    if np.any(household_size <= 0) or np.any(household_size > allowed_size):
        raise ValueError("projected household exceeds its modeled player capacity")
    partial_group = household_group[household_size < allowed_size]
    if partial_group.size:
        partial_counts = np.bincount(
            partial_group.astype(np.int64, copy=False),
            minlength=len(canonical_groups),
        )
        if np.any(partial_counts > 1):
            raise ValueError(
                "each projected population group may have at most one partial household"
            )


def classify_spend_segments(
    spend_cents: NDArray[np.integer] | list[int],
    disposable_income_cents: NDArray[np.integer] | list[int],
    *,
    whale_quantile: float = 0.99,
    whale_income_share: float = 0.10,
) -> NDArray[np.str_]:
    """Return retrospective spend segments for one observation window.

    ``whale`` is assigned only when spending is both in the configured upper
    tail of the *observed payer distribution* and large relative to that
    player's disposable income.  Consequently the label can change across
    windows and is never an intrinsic player attribute.
    """

    spend = np.asarray(spend_cents)
    income = np.asarray(disposable_income_cents)
    if spend.ndim != 1 or income.ndim != 1 or spend.shape != income.shape:
        raise ValueError("spend_cents and disposable_income_cents need equal 1-D shapes")
    if not np.issubdtype(spend.dtype, np.integer) or not np.issubdtype(
        income.dtype, np.integer
    ):
        raise TypeError("spending and income must be integer cents")
    if np.any(spend < 0) or np.any(income < 0):
        raise ValueError("spending and income cannot be negative")
    if not 0.0 < whale_quantile < 1.0:
        raise ValueError("whale_quantile must be in (0, 1)")
    if whale_income_share < 0.0:
        raise ValueError("whale_income_share cannot be negative")

    segments = np.full(spend.shape, SpendSegment.NON_PAYER.value, dtype="<U9")
    payer = spend > 0
    if not np.any(payer):
        return segments

    payer_spend = spend[payer]
    payer_median = float(np.median(payer_spend))
    segments[payer & (spend <= payer_median)] = SpendSegment.MINNOW.value
    segments[payer & (spend > payer_median)] = SpendSegment.DOLPHIN.value

    upper_tail = float(np.quantile(payer_spend, whale_quantile))
    spend_share = np.divide(
        spend.astype(np.float64),
        income.astype(np.float64),
        out=np.full(spend.shape, np.inf, dtype=np.float64),
        where=income > 0,
    )
    whale = payer & (spend >= upper_tail) & (spend_share >= whale_income_share)
    segments[whale] = SpendSegment.WHALE.value
    return segments
