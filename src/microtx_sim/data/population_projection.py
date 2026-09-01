"""Fail-closed bridge from static population targets to runtime players.

This schema-versioned adapter keeps three concepts separate:

* verified source household-income and household-type semantics;
* an explicit, content-addressed conversion to runtime *personal monthly
  disposable-income* intervals and modeled household capacities; and
* the already-attested integer sample counts in a population apportionment
  plan.

Schema v1 retains its discrete-uniform runtime interval semantics. Schema v2
adds an exact, bounded log-normal personal-income declaration without changing
the v1 file shape or digest recipes. The adapter never reallocates cells. Its
runtime initializer consumes the
static plan's exact counts through the exact-count population primitive.  The
result binds the adapter, runtime plan, ordered player ids, and per-player
assignment, but makes no authenticity, balance, held-out, configured-use, or
campaign-readiness claim.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..agents.players import (
    PlayerTable,
    ProjectedPopulationAssignment,
    ProjectedPopulationCellMetadata,
    ProjectedPopulationSexBinding,
    SOURCE_RECORDED_SEX_DTYPE,
    projected_population_assignment_sha256,
    require_treatment_eligible_player_table,
    source_recorded_sex_derivation_input_sha256,
    source_recorded_sex_sha256,
)
from ..consumers.population import (
    CountryProfile,
    PROJECTED_INCOME_BOUNDARY_RULE,
    PROJECTED_INCOME_MINOR_GAMING_ADJUSTMENT,
    PROJECTED_INCOME_MINOR_GAMING_ADJUSTMENT_REASON,
    PROJECTED_INCOME_MODEL_FAMILY,
    PROJECTED_INCOME_ROUNDING_RULE,
    PROJECTED_INCOME_TARGET_QUANTITY,
    PopulationProjectionCell,
    PopulationProjectionIncomeModel,
    PopulationProjectionSampleCount,
    initialize_projected_player_table_from_exact_counts,
)
from ..rng import CounterRNG, validate_seed
from .population_design import (
    PopulationApportionmentPlan,
    PopulationCalibrationTarget,
    PopulationDesignVerification,
    build_population_calibration_target,
    validate_population_apportionment_snapshot,
    verify_population_design_bundle,
)
from .population_evidence import (
    PopulationGamingState,
    PopulationPayerHistoryState,
)


POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION = 1
POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION_V2 = 2
POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION = 1
POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION_V2 = 2
POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION = 1
POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V2 = 2
POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V3 = 3
MAX_POPULATION_RUNTIME_MAPPING_BYTES = 16 * 1024 * 1024

SOURCE_INCOME_CONCEPT = "source_household_income"
RUNTIME_INCOME_CONCEPT = "runtime_personal_monthly_disposable_income_cents"

if RUNTIME_INCOME_CONCEPT != PROJECTED_INCOME_TARGET_QUANTITY:  # pragma: no cover
    raise RuntimeError("runtime income concept constants diverged")

_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CURRENCY_PATTERN = re.compile(r"[A-Z]{3}\Z")

_MAPPING_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "mapping_id",
        "design_id",
        "design_bundle_sha256",
        "domain_sha256",
        "source_income_concept",
        "runtime_income_concept",
        "entries",
    }
)
_MAPPING_ENTRY_KEYS = frozenset(
    {
        "jurisdiction_code",
        "source_household_income_band_id",
        "source_household_income_definition",
        "source_household_income_currency",
        "source_household_income_period",
        "source_household_income_lower_unbounded",
        "source_household_income_lower_bound",
        "source_household_income_upper_unbounded",
        "source_household_income_upper_bound",
        "source_household_type_id",
        "source_household_type_definition",
        "runtime_personal_monthly_disposable_income_band_id",
        "runtime_personal_monthly_disposable_income_currency",
        "runtime_personal_monthly_disposable_income_min_cents",
        "runtime_personal_monthly_disposable_income_max_cents_exclusive",
        "modeled_players_per_household",
        "conversion_recipe_id",
        "conversion_recipe_sha256",
    }
)
_MAPPING_V2_INCOME_MODEL_KEY = (
    "runtime_personal_monthly_disposable_income_model"
)
_MAPPING_ENTRY_KEYS_V2 = _MAPPING_ENTRY_KEYS | {_MAPPING_V2_INCOME_MODEL_KEY}
_MAPPING_V2_INCOME_MODEL_KEYS = frozenset(
    {
        "target_quantity",
        "model_family",
        "median_cents",
        "log_sigma",
        "lower_bound_cents",
        "upper_bound_cents_inclusive",
        "currency",
        "time_period",
        "source_id",
        "calibration_target",
        "transformation",
        "boundary_rule",
        "rounding_rule",
        "minor_gaming_adjustment",
        "minor_gaming_adjustment_reason",
    }
)


class PopulationProjectionValidationError(ValueError):
    """Raised when a source-to-runtime projection declaration is malformed."""


class PopulationProjectionVerificationError(PopulationProjectionValidationError):
    """Raised when exact bytes, upstream lineage, or runtime execution diverges."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise PopulationProjectionValidationError(
            f"{name} must match {_ID_PATTERN.pattern}"
        )
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise PopulationProjectionValidationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )
    return value


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise PopulationProjectionValidationError(
            f"{name} must be a lowercase hexadecimal SHA-256"
        )
    return value


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise PopulationProjectionValidationError(
            f"{name} must be a Python integer"
        )
    if value < minimum or (maximum is not None and value > maximum):
        qualifier = f"at least {minimum}"
        if maximum is not None:
            qualifier += f" and at most {maximum}"
        raise PopulationProjectionValidationError(f"{name} must be {qualifier}")
    return value


def _exact_fraction(
    value: object,
    *,
    name: str,
    nonnegative: bool = True,
) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise PopulationProjectionValidationError(
            f"{name} must be an immutable (numerator, denominator) tuple"
        )
    numerator, denominator = value
    if type(numerator) is not int or type(denominator) is not int:
        raise PopulationProjectionValidationError(
            f"{name} numerator and denominator must be Python integers"
        )
    if denominator <= 0 or (nonnegative and numerator < 0):
        raise PopulationProjectionValidationError(
            f"{name} must have a positive denominator and non-negative numerator"
        )
    if abs(numerator).bit_length() > 4096 or denominator.bit_length() > 4096:
        raise PopulationProjectionValidationError(
            f"{name} exceeds the schema-v1 exact-integer safety limit"
        )
    reduced = Fraction(numerator, denominator)
    if (reduced.numerator, reduced.denominator) != value:
        raise PopulationProjectionValidationError(
            f"{name} must be expressed in lowest terms"
        )
    return value


def _income_model_snapshot(
    model: PopulationProjectionIncomeModel,
) -> dict[str, object]:
    if type(model) is not PopulationProjectionIncomeModel:
        raise TypeError("model must be an exact PopulationProjectionIncomeModel")
    return {
        "target_quantity": model.target_quantity,
        "model_family": model.model_family,
        "median_cents": model.median_cents,
        "log_sigma": list(model.log_sigma),
        "lower_bound_cents": model.lower_bound_cents,
        "upper_bound_cents_inclusive": model.upper_bound_cents_inclusive,
        "currency": model.currency,
        "time_period": model.time_period,
        "source_id": model.source_id,
        "calibration_target": model.calibration_target,
        "transformation": model.transformation,
        "boundary_rule": model.boundary_rule,
        "rounding_rule": model.rounding_rule,
        "minor_gaming_adjustment": model.minor_gaming_adjustment,
        "minor_gaming_adjustment_reason": model.minor_gaming_adjustment_reason,
    }


@dataclass(frozen=True, slots=True)
class PopulationRuntimeMappingEntry:
    """One explicit source household-income/type to runtime mapping."""

    jurisdiction_code: str
    source_household_income_band_id: str
    source_household_income_definition: str
    source_household_income_currency: str
    source_household_income_period: str
    source_household_income_lower_unbounded: bool
    source_household_income_lower_bound: tuple[int, int]
    source_household_income_upper_unbounded: bool
    source_household_income_upper_bound: tuple[int, int]
    source_household_type_id: str
    source_household_type_definition: str
    runtime_personal_monthly_disposable_income_band_id: str
    runtime_personal_monthly_disposable_income_currency: str
    runtime_personal_monthly_disposable_income_min_cents: int
    runtime_personal_monthly_disposable_income_max_cents_exclusive: int
    modeled_players_per_household: int
    conversion_recipe_id: str
    conversion_recipe_sha256: str
    income_model: PopulationProjectionIncomeModel | None = None

    def __post_init__(self) -> None:
        _identifier(self.jurisdiction_code, name="jurisdiction_code")
        _identifier(
            self.source_household_income_band_id,
            name="source_household_income_band_id",
        )
        _text(
            self.source_household_income_definition,
            name="source_household_income_definition",
        )
        if type(self.source_household_income_currency) is not str or (
            _CURRENCY_PATTERN.fullmatch(self.source_household_income_currency) is None
        ):
            raise PopulationProjectionValidationError(
                "source_household_income_currency must be uppercase ISO-like text"
            )
        _text(
            self.source_household_income_period,
            name="source_household_income_period",
        )
        if type(self.source_household_income_lower_unbounded) is not bool or type(
            self.source_household_income_upper_unbounded
        ) is not bool:
            raise PopulationProjectionValidationError(
                "source household-income bound flags must be booleans"
            )
        _exact_fraction(
            self.source_household_income_lower_bound,
            name="source_household_income_lower_bound",
        )
        _exact_fraction(
            self.source_household_income_upper_bound,
            name="source_household_income_upper_bound",
        )
        if self.source_household_income_lower_unbounded and (
            self.source_household_income_lower_bound != (0, 1)
        ):
            raise PopulationProjectionValidationError(
                "unbounded source lower bound must use the 0/1 sentinel"
            )
        if self.source_household_income_upper_unbounded and (
            self.source_household_income_upper_bound != (0, 1)
        ):
            raise PopulationProjectionValidationError(
                "unbounded source upper bound must use the 0/1 sentinel"
            )
        if (
            not self.source_household_income_lower_unbounded
            and not self.source_household_income_upper_unbounded
            and Fraction(*self.source_household_income_lower_bound)
            >= Fraction(*self.source_household_income_upper_bound)
        ):
            raise PopulationProjectionValidationError(
                "source household-income interval is empty or reversed"
            )
        _identifier(
            self.source_household_type_id,
            name="source_household_type_id",
        )
        _text(
            self.source_household_type_definition,
            name="source_household_type_definition",
        )
        _identifier(
            self.runtime_personal_monthly_disposable_income_band_id,
            name="runtime_personal_monthly_disposable_income_band_id",
        )
        if (
            self.runtime_personal_monthly_disposable_income_band_id.casefold()
            == self.source_household_income_band_id.casefold()
        ):
            raise PopulationProjectionValidationError(
                "runtime personal-income band id must not alias its source "
                "household-income band id"
            )
        if type(self.runtime_personal_monthly_disposable_income_currency) is not str or (
            _CURRENCY_PATTERN.fullmatch(
                self.runtime_personal_monthly_disposable_income_currency
            )
            is None
        ):
            raise PopulationProjectionValidationError(
                "runtime personal-income currency must be uppercase ISO-like text"
            )
        minimum = _strict_int(
            self.runtime_personal_monthly_disposable_income_min_cents,
            name="runtime personal monthly disposable-income minimum cents",
        )
        maximum = _strict_int(
            self.runtime_personal_monthly_disposable_income_max_cents_exclusive,
            name="runtime personal monthly disposable-income maximum cents",
            minimum=1,
            maximum=np.iinfo(np.int64).max,
        )
        if maximum <= minimum:
            raise PopulationProjectionValidationError(
                "runtime personal monthly disposable-income interval is empty or reversed"
            )
        household_size = _strict_int(
            self.modeled_players_per_household,
            name="modeled_players_per_household",
            minimum=1,
        )
        if (maximum - 1) * household_size > 2**53:
            raise PopulationProjectionValidationError(
                "runtime income upper bound times modeled household size must be "
                "at most 2**53 cents"
            )
        _identifier(self.conversion_recipe_id, name="conversion_recipe_id")
        _digest(self.conversion_recipe_sha256, name="conversion_recipe_sha256")
        if self.income_model is not None:
            if type(self.income_model) is not PopulationProjectionIncomeModel:
                raise PopulationProjectionValidationError(
                    "income_model must be an exact PopulationProjectionIncomeModel"
                )
            if (
                self.income_model.target_quantity != RUNTIME_INCOME_CONCEPT
                or self.income_model.currency
                != self.runtime_personal_monthly_disposable_income_currency
                or self.income_model.lower_bound_cents != minimum
                or self.income_model.upper_bound_cents_inclusive != maximum - 1
            ):
                raise PopulationProjectionValidationError(
                    "schema-v2 income-model quantity, currency, and bounds must "
                    "exactly match the declared runtime income interval"
                )

    @property
    def semantic_key(self) -> tuple[str, str, str]:
        return (
            self.jurisdiction_code,
            self.source_household_income_band_id,
            self.source_household_type_id,
        )

    @property
    def runtime_alias_key(self) -> tuple[str, str, str]:
        return (
            self.jurisdiction_code,
            self.source_household_type_id,
            self.runtime_personal_monthly_disposable_income_band_id,
        )

    @property
    def mapping_entry_sha256(self) -> str:
        return sha256(_canonical_json(self.snapshot()).encode("utf-8")).hexdigest()

    def snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "jurisdiction_code": self.jurisdiction_code,
            "source_household_income_band_id": (
                self.source_household_income_band_id
            ),
            "source_household_income_definition": (
                self.source_household_income_definition
            ),
            "source_household_income_currency": (
                self.source_household_income_currency
            ),
            "source_household_income_period": self.source_household_income_period,
            "source_household_income_lower_unbounded": (
                self.source_household_income_lower_unbounded
            ),
            "source_household_income_lower_bound": list(
                self.source_household_income_lower_bound
            ),
            "source_household_income_upper_unbounded": (
                self.source_household_income_upper_unbounded
            ),
            "source_household_income_upper_bound": list(
                self.source_household_income_upper_bound
            ),
            "source_household_type_id": self.source_household_type_id,
            "source_household_type_definition": (
                self.source_household_type_definition
            ),
            "runtime_personal_monthly_disposable_income_band_id": (
                self.runtime_personal_monthly_disposable_income_band_id
            ),
            "runtime_personal_monthly_disposable_income_currency": (
                self.runtime_personal_monthly_disposable_income_currency
            ),
            "runtime_personal_monthly_disposable_income_min_cents": (
                self.runtime_personal_monthly_disposable_income_min_cents
            ),
            "runtime_personal_monthly_disposable_income_max_cents_exclusive": (
                self.runtime_personal_monthly_disposable_income_max_cents_exclusive
            ),
            "modeled_players_per_household": self.modeled_players_per_household,
            "conversion_recipe_id": self.conversion_recipe_id,
            "conversion_recipe_sha256": self.conversion_recipe_sha256,
        }
        if self.income_model is not None:
            snapshot[_MAPPING_V2_INCOME_MODEL_KEY] = _income_model_snapshot(
                self.income_model
            )
        return snapshot


@dataclass(frozen=True, slots=True)
class PopulationRuntimeMappingBundle:
    """Strict file-backed source-to-runtime conversion declaration."""

    mapping_id: str
    design_id: str
    design_bundle_sha256: str
    domain_sha256: str
    source_income_concept: str
    runtime_income_concept: str
    entries: tuple[PopulationRuntimeMappingEntry, ...]
    mapping_path: Path
    mapping_sha256: str
    mapping_byte_length: int
    schema_version: int = POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version not in (
            POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION,
            POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION_V2,
        ):
            raise PopulationProjectionValidationError(
                "runtime mapping schema_version must be integer 1 or 2"
            )
        _identifier(self.mapping_id, name="mapping_id")
        _identifier(self.design_id, name="design_id")
        _digest(self.design_bundle_sha256, name="design_bundle_sha256")
        _digest(self.domain_sha256, name="domain_sha256")
        if self.source_income_concept != SOURCE_INCOME_CONCEPT:
            raise PopulationProjectionValidationError(
                f"source_income_concept must be {SOURCE_INCOME_CONCEPT}"
            )
        if self.runtime_income_concept != RUNTIME_INCOME_CONCEPT:
            raise PopulationProjectionValidationError(
                f"runtime_income_concept must be {RUNTIME_INCOME_CONCEPT}"
            )
        if self.source_income_concept == self.runtime_income_concept:
            raise PopulationProjectionValidationError(
                "source and runtime income concepts must remain distinct"
            )
        if type(self.entries) is not tuple or not self.entries or any(
            type(entry) is not PopulationRuntimeMappingEntry
            for entry in self.entries
        ):
            raise PopulationProjectionValidationError(
                "entries must be a non-empty immutable tuple of exact mapping entries"
            )
        if self.schema_version == POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION:
            if any(entry.income_model is not None for entry in self.entries):
                raise PopulationProjectionValidationError(
                    "schema-v1 mapping entries must not declare an income model"
                )
        elif any(entry.income_model is None for entry in self.entries):
            raise PopulationProjectionValidationError(
                "every schema-v2 mapping entry must declare an exact income model"
            )
        keys = tuple(entry.semantic_key for entry in self.entries)
        if keys != tuple(sorted(keys)):
            raise PopulationProjectionValidationError(
                "mapping entries must use canonical semantic-key order"
            )
        folded_keys = tuple(
            tuple(component.casefold() for component in key) for key in keys
        )
        if len(set(folded_keys)) != len(folded_keys):
            raise PopulationProjectionValidationError(
                "mapping entries repeat a source semantic key"
            )
        runtime_aliases: dict[tuple[str, str, str], str] = {}
        for entry in self.entries:
            alias_key = tuple(
                component.casefold() for component in entry.runtime_alias_key
            )
            previous = runtime_aliases.setdefault(
                alias_key,
                entry.source_household_income_band_id.casefold(),
            )
            if previous != entry.source_household_income_band_id.casefold():
                raise PopulationProjectionValidationError(
                    "distinct source income bands must not alias one runtime band "
                    "within a jurisdiction/household type"
                )
        if not isinstance(self.mapping_path, Path) or not self.mapping_path.is_absolute():
            raise PopulationProjectionValidationError(
                "mapping_path must be an absolute pathlib.Path"
            )
        lexical = Path(os.path.normpath(os.fspath(self.mapping_path)))
        if ".." in self.mapping_path.parts or lexical != self.mapping_path:
            raise PopulationProjectionValidationError(
                "mapping_path must be lexically canonical"
            )
        _digest(self.mapping_sha256, name="mapping_sha256")
        _strict_int(
            self.mapping_byte_length,
            name="mapping_byte_length",
            minimum=1,
            maximum=MAX_POPULATION_RUNTIME_MAPPING_BYTES,
        )
        observed = _secure_read_regular_file(self.mapping_path)
        if (
            sha256(observed).hexdigest() != self.mapping_sha256
            or len(observed) != self.mapping_byte_length
        ):
            raise PopulationProjectionVerificationError(
                "runtime mapping metadata do not match its exact file bytes"
            )
        declaration = _parse_mapping_declaration(observed)
        if declaration != self.declaration_snapshot():
            raise PopulationProjectionVerificationError(
                "runtime mapping object differs from its exact file declaration"
            )

    def declaration_snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mapping_id": self.mapping_id,
            "design_id": self.design_id,
            "design_bundle_sha256": self.design_bundle_sha256,
            "domain_sha256": self.domain_sha256,
            "source_income_concept": self.source_income_concept,
            "runtime_income_concept": self.runtime_income_concept,
            "entries": [entry.snapshot() for entry in self.entries],
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.declaration_snapshot(),
            "mapping_path": str(self.mapping_path),
            "mapping_sha256": self.mapping_sha256,
            "mapping_byte_length": self.mapping_byte_length,
        }


def _secure_read_regular_file(path: Path) -> bytes:
    def is_reparse(observed: os.stat_result) -> bool:
        attributes = getattr(observed, "st_file_attributes", 0)
        marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(attributes & marker)

    def same_identity(left: os.stat_result, right: os.stat_result) -> bool:
        return (
            left.st_dev,
            left.st_ino,
            left.st_mode,
            left.st_size,
            left.st_mtime_ns,
        ) == (
            right.st_dev,
            right.st_ino,
            right.st_mode,
            right.st_size,
            right.st_mtime_ns,
        )

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PopulationProjectionVerificationError(
            f"cannot inspect runtime mapping file: {path}"
        ) from exc
    if (
        path.is_symlink()
        or is_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise PopulationProjectionVerificationError(
            "runtime mapping path must name a regular non-symlink file"
        )
    if metadata.st_size <= 0 or metadata.st_size > MAX_POPULATION_RUNTIME_MAPPING_BYTES:
        raise PopulationProjectionValidationError(
            "runtime mapping byte length is outside schema-v1 limits"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PopulationProjectionVerificationError(
            f"cannot read runtime mapping file: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or is_reparse(opened):
            raise PopulationProjectionVerificationError(
                "opened runtime mapping object is not a regular file"
            )
        if not same_identity(metadata, opened):
            raise PopulationProjectionVerificationError(
                "runtime mapping file changed while it was opened"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(
                    1024 * 1024,
                    MAX_POPULATION_RUNTIME_MAPPING_BYTES + 1 - total,
                ),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_POPULATION_RUNTIME_MAPPING_BYTES:
                raise PopulationProjectionValidationError(
                    "runtime mapping exceeds schema-v1 byte limit"
                )
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise PopulationProjectionVerificationError(
            "runtime mapping file changed while it was read"
        ) from exc
    if path.is_symlink() or is_reparse(after_path) or not stat.S_ISREG(
        after_path.st_mode
    ):
        raise PopulationProjectionVerificationError(
            "runtime mapping file changed to a non-regular or aliased path"
        )
    if not same_identity(opened, after_open) or not same_identity(
        after_open,
        after_path,
    ):
        raise PopulationProjectionVerificationError(
            "runtime mapping file changed while it was read"
        )
    observed = b"".join(chunks)
    if len(observed) != after_open.st_size:
        raise PopulationProjectionVerificationError(
            "runtime mapping file was not read completely"
        )
    return observed


def _reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PopulationProjectionValidationError(
                f"runtime mapping JSON repeats key {key!r}"
            )
        result[key] = value
    return result


def _parse_mapping_declaration(observed: bytes) -> dict[str, object]:
    try:
        text = observed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PopulationProjectionValidationError(
            "runtime mapping must be UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise PopulationProjectionValidationError(
            "runtime mapping must not contain a UTF-8 BOM"
        )
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=lambda value: (_raise_invalid_json_constant(value)),
        )
    except json.JSONDecodeError as exc:
        raise PopulationProjectionValidationError(
            f"invalid runtime mapping JSON: {exc}"
        ) from exc
    if type(raw) is not dict:
        raise PopulationProjectionValidationError(
            "runtime mapping root must be a JSON object"
        )
    if set(raw) != _MAPPING_ROOT_KEYS:
        raise PopulationProjectionValidationError(
            "runtime mapping root fields differ from schema v1"
        )
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int or schema_version not in (
        POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION,
        POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION_V2,
    ):
        raise PopulationProjectionValidationError(
            "runtime mapping schema_version must be integer 1 or 2"
        )
    entries_raw = raw.get("entries")
    if type(entries_raw) is not list or not entries_raw:
        raise PopulationProjectionValidationError(
            "runtime mapping entries must be a non-empty JSON array"
        )
    entries = tuple(
        _parse_mapping_entry(item, schema_version=schema_version)
        for item in entries_raw
    )
    declaration = {
        "schema_version": schema_version,
        "mapping_id": _identifier(raw.get("mapping_id"), name="mapping_id"),
        "design_id": _identifier(raw.get("design_id"), name="design_id"),
        "design_bundle_sha256": _digest(
            raw.get("design_bundle_sha256"),
            name="design_bundle_sha256",
        ),
        "domain_sha256": _digest(raw.get("domain_sha256"), name="domain_sha256"),
        "source_income_concept": _text(
            raw.get("source_income_concept"),
            name="source_income_concept",
        ),
        "runtime_income_concept": _text(
            raw.get("runtime_income_concept"),
            name="runtime_income_concept",
        ),
        "entries": [entry.snapshot() for entry in entries],
    }
    # Exercise whole-bundle invariants without recursively reopening a file.
    keys = tuple(entry.semantic_key for entry in entries)
    folded_keys = tuple(
        tuple(component.casefold() for component in key) for key in keys
    )
    if keys != tuple(sorted(keys)) or len(set(folded_keys)) != len(folded_keys):
        raise PopulationProjectionValidationError(
            "runtime mapping entries are not unique canonical semantic keys"
        )
    aliases: dict[tuple[str, str, str], str] = {}
    for entry in entries:
        alias_key = tuple(
            component.casefold() for component in entry.runtime_alias_key
        )
        previous = aliases.setdefault(
            alias_key,
            entry.source_household_income_band_id.casefold(),
        )
        if previous != entry.source_household_income_band_id.casefold():
            raise PopulationProjectionValidationError(
                "distinct source income bands alias one runtime band"
            )
    if declaration["source_income_concept"] != SOURCE_INCOME_CONCEPT:
        raise PopulationProjectionValidationError(
            f"source_income_concept must be {SOURCE_INCOME_CONCEPT}"
        )
    if declaration["runtime_income_concept"] != RUNTIME_INCOME_CONCEPT:
        raise PopulationProjectionValidationError(
            f"runtime_income_concept must be {RUNTIME_INCOME_CONCEPT}"
        )
    return declaration


def _raise_invalid_json_constant(value: str) -> object:
    raise PopulationProjectionValidationError(
        f"runtime mapping JSON constant {value!r} is not permitted"
    )


def _parse_income_model(value: object) -> PopulationProjectionIncomeModel:
    if type(value) is not dict or set(value) != _MAPPING_V2_INCOME_MODEL_KEYS:
        raise PopulationProjectionValidationError(
            "runtime income-model fields differ from schema v2"
        )
    row = value
    raw_log_sigma = row.get("log_sigma")
    if type(raw_log_sigma) is not list or len(raw_log_sigma) != 2:
        raise PopulationProjectionValidationError(
            "income-model log_sigma must be a two-integer JSON array"
        )
    log_sigma = tuple(raw_log_sigma)
    _exact_fraction(
        log_sigma,
        name="income-model log_sigma",
        nonnegative=False,
    )
    try:
        return PopulationProjectionIncomeModel(
            target_quantity=_text(
                row.get("target_quantity"),
                name="income-model target_quantity",
            ),
            model_family=_text(
                row.get("model_family"),
                name="income-model model_family",
            ),
            median_cents=_strict_int(
                row.get("median_cents"),
                name="income-model median_cents",
                minimum=1,
                maximum=2**53,
            ),
            log_sigma=log_sigma,  # type: ignore[arg-type]
            lower_bound_cents=_strict_int(
                row.get("lower_bound_cents"),
                name="income-model lower_bound_cents",
                maximum=2**53,
            ),
            upper_bound_cents_inclusive=_strict_int(
                row.get("upper_bound_cents_inclusive"),
                name="income-model upper_bound_cents_inclusive",
                maximum=2**53,
            ),
            currency=_text(
                row.get("currency"),
                name="income-model currency",
            ),
            time_period=_text(
                row.get("time_period"),
                name="income-model time_period",
            ),
            source_id=_identifier(
                row.get("source_id"),
                name="income-model source_id",
            ),
            calibration_target=_text(
                row.get("calibration_target"),
                name="income-model calibration_target",
            ),
            transformation=_text(
                row.get("transformation"),
                name="income-model transformation",
            ),
            boundary_rule=_text(
                row.get("boundary_rule"),
                name="income-model boundary_rule",
            ),
            rounding_rule=_text(
                row.get("rounding_rule"),
                name="income-model rounding_rule",
            ),
            minor_gaming_adjustment=_text(
                row.get("minor_gaming_adjustment"),
                name="income-model minor_gaming_adjustment",
            ),
            minor_gaming_adjustment_reason=_text(
                row.get("minor_gaming_adjustment_reason"),
                name="income-model minor_gaming_adjustment_reason",
            ),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PopulationProjectionValidationError):
            raise
        raise PopulationProjectionValidationError(
            f"invalid schema-v2 runtime income model: {exc}"
        ) from exc


def _parse_mapping_entry(
    value: object,
    *,
    schema_version: int,
) -> PopulationRuntimeMappingEntry:
    expected_keys = (
        _MAPPING_ENTRY_KEYS
        if schema_version == POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION
        else _MAPPING_ENTRY_KEYS_V2
    )
    if type(value) is not dict or set(value) != expected_keys:
        raise PopulationProjectionValidationError(
            f"runtime mapping entry fields differ from schema v{schema_version}"
        )
    row = value
    lower = row.get("source_household_income_lower_bound")
    upper = row.get("source_household_income_upper_bound")
    if type(lower) is not list or len(lower) != 2:
        raise PopulationProjectionValidationError(
            "source lower bound must be a two-integer JSON array"
        )
    if type(upper) is not list or len(upper) != 2:
        raise PopulationProjectionValidationError(
            "source upper bound must be a two-integer JSON array"
        )
    income_model = None
    if schema_version == POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION_V2:
        income_model = _parse_income_model(
            row.get(_MAPPING_V2_INCOME_MODEL_KEY)
        )
    return PopulationRuntimeMappingEntry(
        jurisdiction_code=row.get("jurisdiction_code"),  # type: ignore[arg-type]
        source_household_income_band_id=row.get(  # type: ignore[arg-type]
            "source_household_income_band_id"
        ),
        source_household_income_definition=row.get(  # type: ignore[arg-type]
            "source_household_income_definition"
        ),
        source_household_income_currency=row.get(  # type: ignore[arg-type]
            "source_household_income_currency"
        ),
        source_household_income_period=row.get(  # type: ignore[arg-type]
            "source_household_income_period"
        ),
        source_household_income_lower_unbounded=row.get(  # type: ignore[arg-type]
            "source_household_income_lower_unbounded"
        ),
        source_household_income_lower_bound=tuple(lower),  # type: ignore[arg-type]
        source_household_income_upper_unbounded=row.get(  # type: ignore[arg-type]
            "source_household_income_upper_unbounded"
        ),
        source_household_income_upper_bound=tuple(upper),  # type: ignore[arg-type]
        source_household_type_id=row.get(  # type: ignore[arg-type]
            "source_household_type_id"
        ),
        source_household_type_definition=row.get(  # type: ignore[arg-type]
            "source_household_type_definition"
        ),
        runtime_personal_monthly_disposable_income_band_id=row.get(  # type: ignore[arg-type]
            "runtime_personal_monthly_disposable_income_band_id"
        ),
        runtime_personal_monthly_disposable_income_currency=row.get(  # type: ignore[arg-type]
            "runtime_personal_monthly_disposable_income_currency"
        ),
        runtime_personal_monthly_disposable_income_min_cents=row.get(  # type: ignore[arg-type]
            "runtime_personal_monthly_disposable_income_min_cents"
        ),
        runtime_personal_monthly_disposable_income_max_cents_exclusive=row.get(  # type: ignore[arg-type]
            "runtime_personal_monthly_disposable_income_max_cents_exclusive"
        ),
        modeled_players_per_household=row.get(  # type: ignore[arg-type]
            "modeled_players_per_household"
        ),
        conversion_recipe_id=row.get("conversion_recipe_id"),  # type: ignore[arg-type]
        conversion_recipe_sha256=row.get(  # type: ignore[arg-type]
            "conversion_recipe_sha256"
        ),
        income_model=income_model,
    )


def load_population_runtime_mapping_bundle(
    path: str | Path,
) -> PopulationRuntimeMappingBundle:
    """Load and content-address one strict source-to-runtime mapping file."""

    candidate = Path(path)
    mapping_path = Path(os.path.abspath(os.fspath(candidate)))
    observed = _secure_read_regular_file(mapping_path)
    declaration = _parse_mapping_declaration(observed)
    schema_version = declaration["schema_version"]
    entries = tuple(
        _parse_mapping_entry(
            item,
            schema_version=schema_version,  # type: ignore[arg-type]
        )
        for item in declaration["entries"]
    )
    return PopulationRuntimeMappingBundle(
        mapping_id=declaration["mapping_id"],  # type: ignore[arg-type]
        design_id=declaration["design_id"],  # type: ignore[arg-type]
        design_bundle_sha256=declaration["design_bundle_sha256"],  # type: ignore[arg-type]
        domain_sha256=declaration["domain_sha256"],  # type: ignore[arg-type]
        source_income_concept=declaration["source_income_concept"],  # type: ignore[arg-type]
        runtime_income_concept=declaration["runtime_income_concept"],  # type: ignore[arg-type]
        entries=entries,
        mapping_path=mapping_path,
        mapping_sha256=sha256(observed).hexdigest(),
        mapping_byte_length=len(observed),
        schema_version=schema_version,  # type: ignore[arg-type]
    )


def _reverify_design(
    verification: PopulationDesignVerification,
) -> PopulationDesignVerification:
    if type(verification) is not PopulationDesignVerification:
        raise TypeError("verification must be a PopulationDesignVerification")
    observed = verify_population_design_bundle(
        verification.bundle,
        population_evidence_bundle=verification.evidence_bundle,
        population_evidence_results=verification.evidence_results,
    )
    if observed != verification:
        raise PopulationProjectionVerificationError(
            "population design verification differs from re-attested bytes"
        )
    return observed


def _reverify_mapping(
    mapping: PopulationRuntimeMappingBundle,
) -> PopulationRuntimeMappingBundle:
    if type(mapping) is not PopulationRuntimeMappingBundle:
        raise TypeError("mapping_bundle must be a PopulationRuntimeMappingBundle")
    observed = load_population_runtime_mapping_bundle(mapping.mapping_path)
    if observed != mapping:
        raise PopulationProjectionVerificationError(
            "runtime mapping differs from its current exact bytes"
        )
    return observed


def _verified_target_and_plan(
    verification: PopulationDesignVerification,
    plan: PopulationApportionmentPlan,
) -> tuple[PopulationDesignVerification, PopulationCalibrationTarget, PopulationApportionmentPlan]:
    fresh = _reverify_design(verification)
    if type(plan) is not PopulationApportionmentPlan:
        raise TypeError("apportionment_plan must be a PopulationApportionmentPlan")
    target = build_population_calibration_target(fresh)
    if plan.calibration_target != target:
        raise PopulationProjectionVerificationError(
            "apportionment plan does not belong to the verified calibration target"
        )
    observed_plan = validate_population_apportionment_snapshot(
        plan.snapshot(),
        target,
    )
    if observed_plan != plan:
        raise PopulationProjectionVerificationError(
            "apportionment plan differs from its exact verified target"
        )
    return fresh, target, observed_plan


def _verify_mapping_against_design(
    mapping: PopulationRuntimeMappingBundle,
    verification: PopulationDesignVerification,
) -> PopulationRuntimeMappingBundle:
    observed = _reverify_mapping(mapping)
    bundle = verification.bundle
    if (
        observed.design_id != bundle.design_id
        or observed.design_bundle_sha256 != bundle.bundle_sha256
        or observed.domain_sha256 != bundle.domain_sha256
    ):
        raise PopulationProjectionVerificationError(
            "runtime mapping lineage differs from the verified population design"
        )
    expected: dict[tuple[str, str, str], tuple[object, object]] = {}
    household_by_id = {
        household.household_type_id: household
        for household in bundle.household_types
    }
    for jurisdiction in bundle.jurisdictions:
        income_bands = tuple(
            band
            for band in bundle.income_bands
            if band.jurisdiction_code == jurisdiction.jurisdiction_code
        )
        for income_band in income_bands:
            for household in bundle.household_types:
                expected[
                    (
                        jurisdiction.jurisdiction_code,
                        income_band.income_band_id,
                        household.household_type_id,
                    )
                ] = (income_band, household)
    observed_by_key = {entry.semantic_key: entry for entry in observed.entries}
    missing = tuple(sorted(set(expected).difference(observed_by_key)))
    extra = tuple(sorted(set(observed_by_key).difference(expected)))
    if missing or extra:
        raise PopulationProjectionVerificationError(
            "runtime mapping must exactly cover every declared "
            "jurisdiction/source-income-band/household-type key; "
            f"missing={missing!r}; extra={extra!r}"
        )
    for key, (income_band, household) in expected.items():
        entry = observed_by_key[key]
        if (
            entry.source_household_income_definition != income_band.definition
            or entry.source_household_income_currency != income_band.currency
            or entry.source_household_income_period != income_band.period
            or entry.source_household_income_lower_unbounded
            is not income_band.lower_unbounded
            or entry.source_household_income_lower_bound
            != (
                income_band.lower_bound_numerator,
                income_band.lower_bound_denominator,
            )
            or entry.source_household_income_upper_unbounded
            is not income_band.upper_unbounded
            or entry.source_household_income_upper_bound
            != (
                income_band.upper_bound_numerator,
                income_band.upper_bound_denominator,
            )
            or entry.source_household_type_definition != household.definition
        ):
            raise PopulationProjectionVerificationError(
                "runtime mapping source semantics differ from the exact design "
                f"for key {key!r}"
            )
    if set(household_by_id) != {
        entry.source_household_type_id for entry in observed.entries
    }:
        raise PopulationProjectionVerificationError(
            "runtime mapping household types differ from the verified domain"
        )
    return observed


@dataclass(frozen=True, slots=True)
class PopulationProjectionAdapterCell:
    """One exact static ordinal mapped to one runtime projection cell."""

    cell_ordinal: int
    evidence_cell_id: str
    projection_cell: PopulationProjectionCell
    sample_count: int
    analysis_weight: tuple[int, int]
    expansion_weight: tuple[int, int]
    mapping_entry_sha256: str

    def __post_init__(self) -> None:
        _strict_int(self.cell_ordinal, name="cell_ordinal")
        _identifier(self.evidence_cell_id, name="evidence_cell_id")
        if type(self.projection_cell) is not PopulationProjectionCell:
            raise PopulationProjectionValidationError(
                "projection_cell must be an exact PopulationProjectionCell"
            )
        _strict_int(self.sample_count, name="sample_count")
        _exact_fraction(self.analysis_weight, name="analysis_weight")
        _exact_fraction(self.expansion_weight, name="expansion_weight")
        _digest(self.mapping_entry_sha256, name="mapping_entry_sha256")
        mass = Fraction(*self.projection_cell.global_mass)
        analysis = Fraction(*self.analysis_weight)
        if self.sample_count == 0:
            if mass != 0 or analysis != 0 or Fraction(*self.expansion_weight) != 0:
                raise PopulationProjectionValidationError(
                    "only a zero-mass adapter cell may have zero samples/weights"
                )
        elif mass <= 0 or analysis != mass / self.sample_count:
            raise PopulationProjectionValidationError(
                "adapter analysis weight must equal exact mass / sample count"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "cell_ordinal": self.cell_ordinal,
            "evidence_cell_id": self.evidence_cell_id,
            "projection_cell": _projection_cell_snapshot(self.projection_cell),
            "sample_count": self.sample_count,
            "analysis_weight": list(self.analysis_weight),
            "expansion_weight": list(self.expansion_weight),
            "mapping_entry_sha256": self.mapping_entry_sha256,
        }


def _projection_cell_snapshot(cell: PopulationProjectionCell) -> dict[str, object]:
    if type(cell) is not PopulationProjectionCell:
        raise TypeError("cell must be an exact PopulationProjectionCell")
    snapshot: dict[str, object] = {
        "cell_id": cell.cell_id,
        "jurisdiction_code": cell.jurisdiction_code,
        "age_min_inclusive": cell.age_min_inclusive,
        "age_max_exclusive": cell.age_max_exclusive,
        "monthly_disposable_income_band_id": (
            cell.monthly_disposable_income_band_id
        ),
        "monthly_disposable_income_min_cents": (
            cell.monthly_disposable_income_min_cents
        ),
        "monthly_disposable_income_max_cents_exclusive": (
            cell.monthly_disposable_income_max_cents_exclusive
        ),
        "household_type": cell.household_type,
        "modeled_players_per_household": cell.modeled_players_per_household,
        "baseline_gamer": cell.baseline_gamer,
        "baseline_ever_payer": cell.baseline_ever_payer,
        "global_mass": list(cell.global_mass),
    }
    if cell.income_model is not None:
        snapshot[_MAPPING_V2_INCOME_MODEL_KEY] = _income_model_snapshot(
            cell.income_model
        )
    return snapshot


def _expected_adapter_cells(
    verification: PopulationDesignVerification,
    plan: PopulationApportionmentPlan,
    mapping: PopulationRuntimeMappingBundle,
) -> tuple[PopulationProjectionAdapterCell, ...]:
    bundle = verification.bundle
    age_by_id = {band.age_band_id: band for band in bundle.age_bands}
    mapping_by_key = {entry.semantic_key: entry for entry in mapping.entries}
    result: list[PopulationProjectionAdapterCell] = []
    for apportioned in plan.cells:
        static = apportioned.calibration_cell
        age = age_by_id[static.age_band_id]
        entry = mapping_by_key[
            (
                static.jurisdiction_code,
                static.income_band_id,
                static.household_type_id,
            )
        ]
        mass = static.target_mass
        projection_cell = PopulationProjectionCell(
            cell_id=f"cell.{static.cell_ordinal:020d}",
            jurisdiction_code=static.jurisdiction_code,
            age_min_inclusive=age.age_min_inclusive,
            age_max_exclusive=age.age_max_exclusive,
            monthly_disposable_income_band_id=(
                entry.runtime_personal_monthly_disposable_income_band_id
            ),
            monthly_disposable_income_min_cents=(
                entry.runtime_personal_monthly_disposable_income_min_cents
            ),
            monthly_disposable_income_max_cents_exclusive=(
                entry.runtime_personal_monthly_disposable_income_max_cents_exclusive
            ),
            household_type=static.household_type_id,
            modeled_players_per_household=entry.modeled_players_per_household,
            baseline_gamer=(static.gaming_state is PopulationGamingState.GAMER),
            baseline_ever_payer=(
                static.payer_history_state
                is PopulationPayerHistoryState.EVER_PAYER
            ),
            global_mass=(mass.numerator, mass.denominator),
            income_model=entry.income_model,
        )
        result.append(
            PopulationProjectionAdapterCell(
                cell_ordinal=static.cell_ordinal,
                evidence_cell_id=static.evidence_cell_id,
                projection_cell=projection_cell,
                sample_count=apportioned.sample_count,
                analysis_weight=(
                    apportioned.analysis_weight_numerator,
                    apportioned.analysis_weight_denominator,
                ),
                expansion_weight=(
                    apportioned.expansion_weight_numerator,
                    apportioned.expansion_weight_denominator,
                ),
                mapping_entry_sha256=entry.mapping_entry_sha256,
            )
        )
    return tuple(result)


def _adapter_schema_version(mapping: PopulationRuntimeMappingBundle) -> int:
    return (
        POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION
        if mapping.schema_version == POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION
        else POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION_V2
    )


@dataclass(frozen=True, slots=True)
class PopulationProjectionAdapter:
    """Content-addressed static-plan-to-runtime projection adapter."""

    verification: PopulationDesignVerification
    apportionment_plan: PopulationApportionmentPlan
    mapping_bundle: PopulationRuntimeMappingBundle
    adapter_id: str
    cells: tuple[PopulationProjectionAdapterCell, ...]
    adapter_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.adapter_id, name="adapter_id")
        fresh, _target, plan = _verified_target_and_plan(
            self.verification,
            self.apportionment_plan,
        )
        mapping = _verify_mapping_against_design(self.mapping_bundle, fresh)
        if type(self.cells) is not tuple or any(
            type(cell) is not PopulationProjectionAdapterCell for cell in self.cells
        ):
            raise PopulationProjectionValidationError(
                "adapter cells must be an immutable tuple of exact adapter cells"
            )
        expected_cells = _expected_adapter_cells(fresh, plan, mapping)
        if self.cells != expected_cells:
            raise PopulationProjectionVerificationError(
                "adapter cells differ from the verified static plan and runtime mapping"
            )
        if tuple(cell.cell_ordinal for cell in self.cells) != tuple(
            range(len(self.cells))
        ):
            raise PopulationProjectionValidationError(
                "adapter cells must retain contiguous static ordinals"
            )
        _digest(self.adapter_sha256, name="adapter_sha256")
        expected_sha256 = sha256(
            _canonical_json(self.attestation_payload()).encode("utf-8")
        ).hexdigest()
        if self.adapter_sha256 != expected_sha256:
            raise PopulationProjectionValidationError(
                "adapter_sha256 does not match its exact lineage and cells"
            )

    @property
    def runtime_projection_id(self) -> str:
        return f"adapter.{self.adapter_sha256}"

    @property
    def calibration_target_sha256(self) -> str:
        return self.apportionment_plan.calibration_target_sha256

    @property
    def apportionment_sha256(self) -> str:
        return self.apportionment_plan.apportionment_sha256

    @property
    def mapping_id(self) -> str:
        return self.mapping_bundle.mapping_id

    @property
    def mapping_sha256(self) -> str:
        return self.mapping_bundle.mapping_sha256

    @property
    def schema_version(self) -> int:
        return _adapter_schema_version(self.mapping_bundle)

    @property
    def authenticity_verified(self) -> bool:
        return False

    @property
    def balance_verified(self) -> bool:
        return False

    @property
    def campaign_ready(self) -> bool:
        return False

    def attestation_payload(self) -> dict[str, object]:
        plan = self.apportionment_plan
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "design_verification_sha256": self.verification.verification_sha256,
            "design_id": plan.design_id,
            "design_bundle_sha256": plan.design_bundle_sha256,
            "domain_sha256": plan.domain_sha256,
            "calibration_target_sha256": plan.calibration_target_sha256,
            "apportionment_sha256": plan.apportionment_sha256,
            "mapping_id": self.mapping_bundle.mapping_id,
            "mapping_sha256": self.mapping_bundle.mapping_sha256,
            "player_count": plan.player_count,
            "first_player_id": plan.first_player_id,
            "last_player_id_exclusive": plan.last_player_id_exclusive,
            "cell_count": len(self.cells),
            "cells": [cell.snapshot() for cell in self.cells],
            "authenticity_verified": False,
            "balance_verified": False,
            "campaign_ready": False,
        }
        if self.schema_version == POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION_V2:
            payload["mapping_schema_version"] = self.mapping_bundle.schema_version
        return payload

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "adapter_sha256": self.adapter_sha256,
            "runtime_projection_id": self.runtime_projection_id,
        }


def build_population_projection_adapter(
    verification: PopulationDesignVerification,
    apportionment_plan: PopulationApportionmentPlan,
    mapping_bundle: PopulationRuntimeMappingBundle,
    *,
    adapter_id: str,
) -> PopulationProjectionAdapter:
    """Bind verified static cells/counts to explicit runtime assumptions."""

    _identifier(adapter_id, name="adapter_id")
    fresh, _target, plan = _verified_target_and_plan(
        verification,
        apportionment_plan,
    )
    mapping = _verify_mapping_against_design(mapping_bundle, fresh)
    cells = _expected_adapter_cells(fresh, plan, mapping)
    adapter_schema_version = _adapter_schema_version(mapping)
    payload: dict[str, object] = {
        "schema_version": adapter_schema_version,
        "adapter_id": adapter_id,
        "design_verification_sha256": fresh.verification_sha256,
        "design_id": plan.design_id,
        "design_bundle_sha256": plan.design_bundle_sha256,
        "domain_sha256": plan.domain_sha256,
        "calibration_target_sha256": plan.calibration_target_sha256,
        "apportionment_sha256": plan.apportionment_sha256,
        "mapping_id": mapping.mapping_id,
        "mapping_sha256": mapping.mapping_sha256,
        "player_count": plan.player_count,
        "first_player_id": plan.first_player_id,
        "last_player_id_exclusive": plan.last_player_id_exclusive,
        "cell_count": len(cells),
        "cells": [cell.snapshot() for cell in cells],
        "authenticity_verified": False,
        "balance_verified": False,
        "campaign_ready": False,
    }
    if adapter_schema_version == POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION_V2:
        payload["mapping_schema_version"] = mapping.schema_version
    return PopulationProjectionAdapter(
        verification=fresh,
        apportionment_plan=plan,
        mapping_bundle=mapping,
        adapter_id=adapter_id,
        cells=cells,
        adapter_sha256=sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    )


def verify_population_projection_adapter(
    adapter: PopulationProjectionAdapter,
) -> PopulationProjectionAdapter:
    """Reopen every bound byte artifact and rebuild an adapter exactly."""

    if type(adapter) is not PopulationProjectionAdapter:
        raise TypeError("adapter must be a PopulationProjectionAdapter")
    observed = build_population_projection_adapter(
        adapter.verification,
        adapter.apportionment_plan,
        adapter.mapping_bundle,
        adapter_id=adapter.adapter_id,
    )
    if observed != adapter:
        raise PopulationProjectionVerificationError(
            "population projection adapter differs from rebuilt exact inputs"
        )
    return observed


def population_projection_ordered_player_ids_sha256(
    player_ids: NDArray[np.int64],
) -> str:
    """Hash one exact ordered int64 player-id vector for execution lineage."""

    if not isinstance(player_ids, np.ndarray):
        raise TypeError("player_ids must be a NumPy array")
    if player_ids.ndim != 1 or player_ids.dtype != np.dtype(np.int64):
        raise TypeError("player_ids must be a one-dimensional int64 array")
    digest = sha256(
        b"microtx-sim.population-projection-ordered-player-ids.v1\0"
    )
    digest.update(player_ids.size.to_bytes(8, "little", signed=False))
    digest.update(np.asarray(player_ids, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()


def _population_projection_execution_attestation_payload(
    adapter: PopulationProjectionAdapter,
    *,
    initialization_seed: int,
    initialization_tick: int,
    runtime_projection_id: str,
    runtime_projection_sha256: str,
    assignment_sha256: str,
    ordered_player_ids_sha256: str,
    sex_binding: ProjectedPopulationSexBinding | None = None,
) -> dict[str, object]:
    plan = adapter.apportionment_plan
    if sex_binding is not None:
        if type(sex_binding) is not ProjectedPopulationSexBinding:
            raise TypeError(
                "sex_binding must be an exact ProjectedPopulationSexBinding"
            )
        ProjectedPopulationSexBinding.__post_init__(sex_binding)
        execution_schema_version = POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V3
    else:
        execution_schema_version = (
            POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION
            if adapter.schema_version == POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION
            else POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V2
        )
    payload: dict[str, object] = {
        "schema_version": execution_schema_version,
        "adapter_sha256": adapter.adapter_sha256,
        "apportionment_sha256": adapter.apportionment_sha256,
        "calibration_target_sha256": adapter.calibration_target_sha256,
        "mapping_id": adapter.mapping_id,
        "mapping_sha256": adapter.mapping_sha256,
        "player_count": plan.player_count,
        "first_player_id": plan.first_player_id,
        "initialization_seed": initialization_seed,
        "initialization_seed_decimal": str(initialization_seed),
        "initialization_tick": initialization_tick,
        "initialization_tick_decimal": str(initialization_tick),
        "runtime_projection_id": runtime_projection_id,
        "runtime_projection_sha256": runtime_projection_sha256,
        "assignment_sha256": assignment_sha256,
        "ordered_player_ids_sha256": ordered_player_ids_sha256,
        "campaign_ready": False,
    }
    if execution_schema_version in {
        POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V2,
        POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V3,
    }:
        payload["adapter_schema_version"] = adapter.schema_version
        payload["mapping_schema_version"] = adapter.mapping_bundle.schema_version
    if sex_binding is not None:
        payload["source_recorded_sex"] = sex_binding.snapshot()
        payload["sex_sha256"] = sex_binding.sex_sha256
    return payload


def population_projection_execution_sha256(
    adapter: PopulationProjectionAdapter,
    *,
    initialization_seed: int,
    initialization_tick: int,
    runtime_projection_sha256: str,
    assignment_sha256: str,
    ordered_player_ids_sha256: str,
    sex_binding: ProjectedPopulationSexBinding | None = None,
) -> str:
    """Recompute an execution address from its detached exact bindings."""

    observed = verify_population_projection_adapter(adapter)
    selected_seed = validate_seed(
        initialization_seed,
        name="population initialization seed",
    )
    selected_tick = validate_seed(
        initialization_tick,
        name="population initialization tick",
    )
    for value, name in (
        (runtime_projection_sha256, "runtime_projection_sha256"),
        (assignment_sha256, "assignment_sha256"),
        (ordered_player_ids_sha256, "ordered_player_ids_sha256"),
    ):
        _digest(value, name=name)
    payload = _population_projection_execution_attestation_payload(
        observed,
        initialization_seed=selected_seed,
        initialization_tick=selected_tick,
        runtime_projection_id=observed.runtime_projection_id,
        runtime_projection_sha256=runtime_projection_sha256,
        assignment_sha256=assignment_sha256,
        ordered_player_ids_sha256=ordered_player_ids_sha256,
        sex_binding=sex_binding,
    )
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PopulationProjectionExecution:
    """Content-addressed execution of one exact projection adapter."""

    adapter: PopulationProjectionAdapter
    players: PlayerTable
    initialization_seed: int
    initialization_tick: int
    runtime_projection_id: str
    runtime_projection_sha256: str
    assignment_sha256: str
    ordered_player_ids_sha256: str
    execution_sha256: str

    def __post_init__(self) -> None:
        _validate_execution(self)

    @property
    def apportionment_sha256(self) -> str:
        return self.adapter.apportionment_sha256

    @property
    def calibration_target_sha256(self) -> str:
        return self.adapter.calibration_target_sha256

    @property
    def mapping_id(self) -> str:
        return self.adapter.mapping_id

    @property
    def mapping_sha256(self) -> str:
        return self.adapter.mapping_sha256

    @property
    def campaign_ready(self) -> bool:
        return False

    def attestation_payload(self) -> dict[str, object]:
        assignment = self.players.projected_population
        sex_binding = (
            assignment.sex_binding
            if type(assignment) is ProjectedPopulationAssignment
            else None
        )
        return _population_projection_execution_attestation_payload(
            self.adapter,
            initialization_seed=self.initialization_seed,
            initialization_tick=self.initialization_tick,
            runtime_projection_id=self.runtime_projection_id,
            runtime_projection_sha256=self.runtime_projection_sha256,
            assignment_sha256=self.assignment_sha256,
            ordered_player_ids_sha256=self.ordered_player_ids_sha256,
            sex_binding=sex_binding,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "execution_sha256": self.execution_sha256,
        }


def _expected_runtime_metadata_cells(
    adapter: PopulationProjectionAdapter,
    players: PlayerTable,
) -> tuple[ProjectedPopulationCellMetadata, ...]:
    code_to_index = {
        code: index for index, code in enumerate(players.jurisdiction_codes)
    }
    result: list[ProjectedPopulationCellMetadata] = []
    for cell in adapter.cells:
        projected = cell.projection_cell
        result.append(
            ProjectedPopulationCellMetadata(
                cell_id=projected.cell_id,
                jurisdiction_code=projected.jurisdiction_code,
                jurisdiction_index=code_to_index[projected.jurisdiction_code],
                age_min_inclusive=projected.age_min_inclusive,
                age_max_exclusive=projected.age_max_exclusive,
                monthly_disposable_income_band_id=(
                    projected.monthly_disposable_income_band_id
                ),
                monthly_disposable_income_min_cents=(
                    projected.monthly_disposable_income_min_cents
                ),
                monthly_disposable_income_max_cents_exclusive=(
                    projected.monthly_disposable_income_max_cents_exclusive
                ),
                household_type=projected.household_type,
                modeled_players_per_household=(
                    projected.modeled_players_per_household
                ),
                baseline_gamer=projected.baseline_gamer,
                baseline_ever_payer=projected.baseline_ever_payer,
                global_mass=projected.global_mass,
                analysis_weight=cell.analysis_weight,
            )
        )
    return tuple(result)


def _validate_execution(execution: PopulationProjectionExecution) -> None:
    if type(execution.adapter) is not PopulationProjectionAdapter:
        raise PopulationProjectionValidationError(
            "execution adapter must be an exact PopulationProjectionAdapter"
        )
    adapter = verify_population_projection_adapter(execution.adapter)
    validate_seed(
        execution.initialization_seed,
        name="population initialization seed",
    )
    validate_seed(
        execution.initialization_tick,
        name="population initialization tick",
    )
    if type(execution.players) is not PlayerTable:
        raise PopulationProjectionValidationError(
            "execution players must be an exact PlayerTable"
        )
    assignment = execution.players.projected_population
    if type(assignment) is not ProjectedPopulationAssignment:
        raise PopulationProjectionVerificationError(
            "execution players must carry an exact projected-population assignment"
        )
    plan = adapter.apportionment_plan
    expected_ids = np.arange(
        plan.first_player_id,
        plan.last_player_id_exclusive,
        dtype=np.int64,
    )
    if not np.array_equal(execution.players.player_id, expected_ids):
        raise PopulationProjectionVerificationError(
            "execution player ids differ from the exact apportionment interval"
        )
    player_codes = execution.players.jurisdiction_codes
    expected_codes = tuple(
        item.jurisdiction_code for item in adapter.verification.bundle.jurisdictions
    )
    if len(set(player_codes)) != len(player_codes) or set(player_codes) != set(
        expected_codes
    ):
        raise PopulationProjectionVerificationError(
            "execution jurisdiction code set must be unique and exactly match "
            "the verified design; supplied order is retained at runtime"
        )
    expected_metadata_cells = _expected_runtime_metadata_cells(
        adapter,
        execution.players,
    )
    if assignment.metadata.cells != expected_metadata_cells:
        raise PopulationProjectionVerificationError(
            "runtime projected cells differ from the exact adapter cells/counts"
        )
    counts = np.bincount(
        assignment.cell_index.astype(np.int64, copy=False),
        minlength=len(adapter.cells),
    )
    expected_counts = np.asarray(
        [cell.sample_count for cell in adapter.cells],
        dtype=np.int64,
    )
    if not np.array_equal(counts, expected_counts):
        raise PopulationProjectionVerificationError(
            "runtime assignment counts differ from the static apportionment plan"
        )
    if execution.runtime_projection_id != adapter.runtime_projection_id or (
        assignment.metadata.projection_id != adapter.runtime_projection_id
    ):
        raise PopulationProjectionVerificationError(
            "runtime projection id does not bind the exact adapter digest"
        )
    if execution.runtime_projection_sha256 != assignment.metadata.projection_sha256:
        raise PopulationProjectionVerificationError(
            "runtime projection digest differs from PlayerTable metadata"
        )
    if execution.assignment_sha256 != assignment.assignment_sha256:
        raise PopulationProjectionVerificationError(
            "assignment digest differs from PlayerTable metadata"
        )
    try:
        expected_assignment_sha256 = projected_population_assignment_sha256(
            assignment.metadata,
            execution.players.player_id,
            assignment.cell_index,
            age_years=execution.players.age_years,
            jurisdiction=execution.players.jurisdiction,
            sex=execution.players.sex,
            sex_binding=assignment.sex_binding,
        )
    except (TypeError, ValueError) as exc:
        raise PopulationProjectionVerificationError(
            "runtime assignment values do not verify against their binding"
        ) from exc
    if assignment.assignment_sha256 != expected_assignment_sha256:
        raise PopulationProjectionVerificationError(
            "runtime assignment values differ from their content address"
        )
    if assignment.sex_binding is not None:
        observed_derivation_sha256 = (
            source_recorded_sex_derivation_input_sha256(
                execution.players.player_id,
                execution.players.age_years,
                execution.players.jurisdiction,
                assignment.cell_index,
            )
        )
        if (
            observed_derivation_sha256
            != assignment.sex_binding.derivation_input_sha256
        ):
            raise PopulationProjectionVerificationError(
                "source-recorded sex derivation inputs differ from their binding"
            )
    expected_ids_sha256 = population_projection_ordered_player_ids_sha256(
        execution.players.player_id
    )
    if execution.ordered_player_ids_sha256 != expected_ids_sha256:
        raise PopulationProjectionVerificationError(
            "ordered player-id digest does not match execution players"
        )
    for value, name in (
        (execution.runtime_projection_sha256, "runtime_projection_sha256"),
        (execution.assignment_sha256, "assignment_sha256"),
        (execution.ordered_player_ids_sha256, "ordered_player_ids_sha256"),
        (execution.execution_sha256, "execution_sha256"),
    ):
        _digest(value, name=name)
    expected_execution_sha256 = sha256(
        _canonical_json(execution.attestation_payload()).encode("utf-8")
    ).hexdigest()
    if execution.execution_sha256 != expected_execution_sha256:
        raise PopulationProjectionValidationError(
            "execution_sha256 does not match exact adapter/runtime lineage"
        )


def initialize_population_projection(
    adapter: PopulationProjectionAdapter,
    country_profiles: Sequence[CountryProfile],
    rng: CounterRNG,
    *,
    tick: int = 0,
) -> PopulationProjectionExecution:
    """Initialize players from the adapter's exact static sample counts."""

    observed = verify_population_projection_adapter(adapter)
    if type(rng) is not CounterRNG:
        raise TypeError("rng must be an exact CounterRNG")
    initialization_seed = validate_seed(
        rng.seed,
        name="population initialization seed",
    )
    initialization_tick = validate_seed(
        tick,
        name="population initialization tick",
    )
    profiles = tuple(country_profiles)
    if not profiles or any(type(profile) is not CountryProfile for profile in profiles):
        raise TypeError(
            "country_profiles must contain exact CountryProfile instances"
        )
    expected_codes = tuple(
        item.jurisdiction_code
        for item in observed.verification.bundle.jurisdictions
    )
    profile_codes = tuple(profile.code for profile in profiles)
    if len(set(profile_codes)) != len(profile_codes) or set(profile_codes) != set(
        expected_codes
    ):
        raise PopulationProjectionVerificationError(
            "country profile code set must be unique and exactly match verified "
            "jurisdictions; supplied order is retained at runtime"
        )
    cells = tuple(cell.projection_cell for cell in observed.cells)
    counts = tuple(
        PopulationProjectionSampleCount(
            cell_id=cell.projection_cell.cell_id,
            sample_count=cell.sample_count,
        )
        for cell in observed.cells
    )
    plan = observed.apportionment_plan
    players = initialize_projected_player_table_from_exact_counts(
        plan.player_count,
        profiles,
        rng,
        cells,
        counts,
        projection_id=observed.runtime_projection_id,
        tick=initialization_tick,
        first_player_id=plan.first_player_id,
    )
    assignment = players.projected_population
    if type(assignment) is not ProjectedPopulationAssignment:
        raise PopulationProjectionVerificationError(
            "exact-count initializer did not return a projected assignment"
        )
    ids_sha256 = population_projection_ordered_player_ids_sha256(players.player_id)
    payload = _population_projection_execution_attestation_payload(
        observed,
        initialization_seed=initialization_seed,
        initialization_tick=initialization_tick,
        runtime_projection_id=observed.runtime_projection_id,
        runtime_projection_sha256=assignment.metadata.projection_sha256,
        assignment_sha256=assignment.assignment_sha256,
        ordered_player_ids_sha256=ids_sha256,
    )
    return PopulationProjectionExecution(
        adapter=observed,
        players=players,
        initialization_seed=initialization_seed,
        initialization_tick=initialization_tick,
        runtime_projection_id=observed.runtime_projection_id,
        runtime_projection_sha256=assignment.metadata.projection_sha256,
        assignment_sha256=assignment.assignment_sha256,
        ordered_player_ids_sha256=ids_sha256,
        execution_sha256=sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    )


def bind_population_projection_source_recorded_sex(
    execution: PopulationProjectionExecution,
    sex: NDArray[np.str_],
    binding: ProjectedPopulationSexBinding,
) -> PopulationProjectionExecution:
    """Return a new execution with one immutable, content-addressed sex field.

    This is a pre-treatment projection binding, not a behavioural update. The
    supplied vector must use an empty string outside the binding's declared
    jurisdiction/age scope. Existing unbound executions remain byte-for-byte
    compatible with the historical digest recipe.
    """

    observed = verify_population_projection_execution(execution)
    assignment = observed.players.projected_population
    if type(assignment) is not ProjectedPopulationAssignment:
        raise PopulationProjectionVerificationError(
            "source-recorded sex requires a projected-population assignment"
        )
    if observed.players.sex is not None or assignment.sex_binding is not None:
        raise PopulationProjectionValidationError(
            "projected execution already has a source-recorded sex binding"
        )
    if type(binding) is not ProjectedPopulationSexBinding:
        raise TypeError("binding must be an exact ProjectedPopulationSexBinding")
    ProjectedPopulationSexBinding.__post_init__(binding)
    observed_sex_sha256 = source_recorded_sex_sha256(sex)
    if observed_sex_sha256 != binding.sex_sha256:
        raise PopulationProjectionValidationError(
            "source-recorded sex vector differs from its typed binding"
        )
    if sex.shape != observed.players.player_id.shape:
        raise PopulationProjectionValidationError(
            "source-recorded sex must have one value per projected player"
        )

    selected_sex = np.array(sex, dtype=SOURCE_RECORDED_SEX_DTYPE, copy=True)
    assignment_sha256 = projected_population_assignment_sha256(
        assignment.metadata,
        observed.players.player_id,
        assignment.cell_index,
        age_years=observed.players.age_years,
        jurisdiction=observed.players.jurisdiction,
        sex=selected_sex,
        sex_binding=binding,
    )
    bound_assignment = ProjectedPopulationAssignment(
        metadata=assignment.metadata,
        cell_index=assignment.cell_index,
        assignment_sha256=assignment_sha256,
        sex_binding=binding,
    )
    bound_players = replace(
        observed.players,
        sex=selected_sex,
        projected_population=bound_assignment,
    )
    payload = _population_projection_execution_attestation_payload(
        observed.adapter,
        initialization_seed=observed.initialization_seed,
        initialization_tick=observed.initialization_tick,
        runtime_projection_id=observed.runtime_projection_id,
        runtime_projection_sha256=observed.runtime_projection_sha256,
        assignment_sha256=assignment_sha256,
        ordered_player_ids_sha256=observed.ordered_player_ids_sha256,
        sex_binding=binding,
    )
    return PopulationProjectionExecution(
        adapter=observed.adapter,
        players=bound_players,
        initialization_seed=observed.initialization_seed,
        initialization_tick=observed.initialization_tick,
        runtime_projection_id=observed.runtime_projection_id,
        runtime_projection_sha256=observed.runtime_projection_sha256,
        assignment_sha256=assignment_sha256,
        ordered_player_ids_sha256=observed.ordered_player_ids_sha256,
        execution_sha256=sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    )


def verify_population_projection_execution(
    execution: PopulationProjectionExecution,
) -> PopulationProjectionExecution:
    """Re-attest an execution, including current bound files and player arrays."""

    if type(execution) is not PopulationProjectionExecution:
        raise TypeError("execution must be a PopulationProjectionExecution")
    _validate_execution(execution)
    return execution


def require_treatment_eligible_population_projection(
    execution: PopulationProjectionExecution,
    *,
    operation: str,
) -> PopulationProjectionExecution:
    """Verify an execution and reject point-zero-only structural bindings."""

    observed = verify_population_projection_execution(execution)
    require_treatment_eligible_player_table(
        observed.players,
        operation=operation,
    )
    return observed


__all__ = [
    "MAX_POPULATION_RUNTIME_MAPPING_BYTES",
    "POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION",
    "POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION_V2",
    "POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION",
    "POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V2",
    "POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V3",
    "POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION",
    "POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION_V2",
    "RUNTIME_INCOME_CONCEPT",
    "SOURCE_INCOME_CONCEPT",
    "PopulationProjectionAdapter",
    "PopulationProjectionAdapterCell",
    "PopulationProjectionExecution",
    "PopulationProjectionValidationError",
    "PopulationProjectionVerificationError",
    "PopulationRuntimeMappingBundle",
    "PopulationRuntimeMappingEntry",
    "build_population_projection_adapter",
    "bind_population_projection_source_recorded_sex",
    "initialize_population_projection",
    "load_population_runtime_mapping_bundle",
    "population_projection_execution_sha256",
    "population_projection_ordered_player_ids_sha256",
    "require_treatment_eligible_population_projection",
    "verify_population_projection_adapter",
    "verify_population_projection_execution",
]
