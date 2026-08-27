"""Fail-closed, content-addressed evidence for joint population targets.

Schema version 1 proves that typed joint population cells were extracted from
particular UTF-8 CSV bytes by one whitelisted recipe.  It deliberately supports
only an explicitly missing signature, so content-addressing alone cannot make a
bundle campaign-ready or establish publisher authenticity.
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


POPULATION_EVIDENCE_SCHEMA_VERSION = 1
EXACT_CSV_JOINT_POPULATION_INTERPRETER_V1 = (
    "exact_csv_joint_population_cells/1"
)
MAX_POPULATION_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_POPULATION_BUNDLE_BYTES = 1024 * 1024

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH = (
    _PROJECT_ROOT / "data" / "provenance" / "population_bundle.toml"
)

POPULATION_CELL_CSV_COLUMNS = (
    "target_population_id",
    "jurisdiction_code",
    "estimand_role",
    "cell_id",
    "age_min_inclusive",
    "age_max_exclusive",
    "household_income_band",
    "household_type",
    "gaming_state",
    "payer_history_state",
    "target_mass_numerator",
    "target_mass_denominator",
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
        "target_population_id",
        "jurisdiction_code",
        "geography",
        "reference_period_start",
        "reference_period_end",
        "population_base",
        "universe",
        "unit_of_analysis",
        "eligibility",
        "exclusion",
        "age_min_inclusive",
        "age_max_exclusive",
        "household_income_definition",
        "household_income_currency",
        "household_income_period",
        "household_income_equivalisation",
        "household_definition",
        "gaming_definition",
        "payer_definition",
        "zero_spender_treatment",
        "estimand_role",
        "status",
        "source_ids",
        "retrieved_on",
        "recipe_json",
    }
)
_SIGNATURE_KEYS = frozenset({"status", "algorithm", "key_id", "value"})
_BUNDLE_SNAPSHOT_KEYS = frozenset(
    {
        *_TOP_LEVEL_KEYS,
        "bundle_path",
        "bundle_sha256",
        "bundle_byte_length",
        "campaign_ready",
        "campaign_blockers",
    }
)
_CELL_KEYS = frozenset(
    {
        "cell_id",
        "age_min_inclusive",
        "age_max_exclusive",
        "household_income_band",
        "household_type",
        "gaming_state",
        "payer_history_state",
        "target_mass_numerator",
        "target_mass_denominator",
        "target_mass_numerator_decimal",
        "target_mass_denominator_decimal",
    }
)
_RESULT_KEYS = frozenset(
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
        "target_population_id",
        "jurisdiction_code",
        "estimand_role",
        "cell_count",
        "cells_sha256",
        "cells",
        "total_mass_numerator",
        "total_mass_denominator",
        "total_mass_numerator_decimal",
        "total_mass_denominator_decimal",
        "evidence_sha256",
    }
)
_RECIPE_KEYS = frozenset(
    {"schema_version", "interpreter", "row_match", "cell_columns"}
)
_ROW_MATCH_KEYS = (
    "estimand_role",
    "jurisdiction_code",
    "target_population_id",
)
_ROW_MATCH_KEY_SET = frozenset(_ROW_MATCH_KEYS)

_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_POSITIVE_INTEGER_PATTERN = re.compile(r"[1-9][0-9]*\Z")
_NONNEGATIVE_INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)\Z")


class PopulationEvidenceValidationError(ValueError):
    """Raised when a population-evidence declaration is malformed."""


class PopulationEvidenceVerificationError(PopulationEvidenceValidationError):
    """Raised when declared population evidence differs from observed bytes."""


class PopulationEstimandRole(str, Enum):
    """Disjoint scientific roles for population targets."""

    CALIBRATION = "CALIBRATION"
    VALIDATION = "VALIDATION"


class PopulationGamingState(str, Enum):
    """Exhaustive gaming-state categories accepted by schema v1."""

    GAMER = "GAMER"
    NON_GAMER = "NON_GAMER"


class PopulationPayerHistoryState(str, Enum):
    """Exhaustive payer-history categories accepted by schema v1."""

    EVER_PAYER = "EVER_PAYER"
    NEVER_PAYER = "NEVER_PAYER"


class PopulationEvidenceSignatureStatus(str, Enum):
    """Signature states understood by population-evidence schema v1."""

    MISSING = "MISSING"


@dataclass(frozen=True, slots=True)
class PopulationEvidenceArtifact:
    """One regular CSV file whose entire byte sequence is declared."""

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
            raise PopulationEvidenceValidationError(
                f"artifact {self.artifact_id} media_type must be text/csv"
            )
        _validate_sha256(self.sha256, name=f"artifact {self.artifact_id} sha256")
        _validate_strict_int(
            self.byte_length,
            name=f"artifact {self.artifact_id} byte_length",
            minimum=1,
            maximum=MAX_POPULATION_ARTIFACT_BYTES,
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
class PopulationEvidenceSignature:
    """Explicitly missing signature declaration for schema version 1."""

    status: PopulationEvidenceSignatureStatus
    algorithm: str
    key_id: str
    value: str

    def __post_init__(self) -> None:
        if type(self.status) is not PopulationEvidenceSignatureStatus:
            raise PopulationEvidenceValidationError(
                "population-evidence signature status is invalid"
            )
        if self.status is not PopulationEvidenceSignatureStatus.MISSING:
            raise PopulationEvidenceValidationError(
                "population-evidence schema v1 supports missing signatures only"
            )
        if (self.algorithm, self.key_id, self.value) != ("NONE", "", ""):
            raise PopulationEvidenceValidationError(
                "a missing population-evidence signature must use algorithm NONE "
                "and empty key/value fields"
            )

    def snapshot(self) -> dict[str, str]:
        return {
            "status": self.status.value,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class PopulationEvidenceCell:
    """One exact joint age-income-household-gaming-payer population cell."""

    cell_id: str
    age_min_inclusive: int
    age_max_exclusive: int
    household_income_band: str
    household_type: str
    gaming_state: PopulationGamingState
    payer_history_state: PopulationPayerHistoryState
    target_mass_numerator: int
    target_mass_denominator: int

    def __post_init__(self) -> None:
        _validate_id(self.cell_id, name="cell_id")
        _validate_strict_int(
            self.age_min_inclusive,
            name=f"cell {self.cell_id} age_min_inclusive",
            minimum=0,
            maximum=199,
        )
        _validate_strict_int(
            self.age_max_exclusive,
            name=f"cell {self.cell_id} age_max_exclusive",
            minimum=1,
            maximum=200,
        )
        if self.age_max_exclusive <= self.age_min_inclusive:
            raise PopulationEvidenceValidationError(
                f"cell {self.cell_id} age bounds are empty or reversed"
            )
        _validate_id(
            self.household_income_band,
            name=f"cell {self.cell_id} household_income_band",
        )
        _validate_id(
            self.household_type,
            name=f"cell {self.cell_id} household_type",
        )
        if type(self.gaming_state) is not PopulationGamingState:
            raise PopulationEvidenceValidationError(
                f"cell {self.cell_id} gaming_state is invalid"
            )
        if type(self.payer_history_state) is not PopulationPayerHistoryState:
            raise PopulationEvidenceValidationError(
                f"cell {self.cell_id} payer_history_state is invalid"
            )
        _validate_strict_int(
            self.target_mass_numerator,
            name=f"cell {self.cell_id} target_mass_numerator",
            minimum=0,
        )
        _validate_strict_int(
            self.target_mass_denominator,
            name=f"cell {self.cell_id} target_mass_denominator",
            minimum=1,
        )
        if math.gcd(
            self.target_mass_numerator,
            self.target_mass_denominator,
        ) != 1:
            raise PopulationEvidenceValidationError(
                f"cell {self.cell_id} target mass must be in lowest terms"
            )

    @property
    def target_mass(self) -> Fraction:
        return Fraction(self.target_mass_numerator, self.target_mass_denominator)

    @property
    def semantic_key(self) -> tuple[object, ...]:
        return (
            self.age_min_inclusive,
            self.age_max_exclusive,
            self.household_income_band,
            self.household_type,
            self.gaming_state.value,
            self.payer_history_state.value,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "age_min_inclusive": self.age_min_inclusive,
            "age_max_exclusive": self.age_max_exclusive,
            "household_income_band": self.household_income_band,
            "household_type": self.household_type,
            "gaming_state": self.gaming_state.value,
            "payer_history_state": self.payer_history_state.value,
            "target_mass_numerator": self.target_mass_numerator,
            "target_mass_denominator": self.target_mass_denominator,
            "target_mass_numerator_decimal": str(self.target_mass_numerator),
            "target_mass_denominator_decimal": str(self.target_mass_denominator),
        }


# Keep the shorter spelling as a source-compatible alias for early callers while
# exposing the evidence-qualified public name used by the rest of the provenance
# API.  Both names identify the same strict dataclass (there is no subclassing or
# conversion path that could weaken the exact-type checks below).
PopulationCell = PopulationEvidenceCell


@dataclass(frozen=True, slots=True)
class PopulationEvidenceBinding:
    """Typed target-population metadata plus an exact CSV extraction recipe."""

    binding_id: str
    artifact_id: str
    target_population_id: str
    jurisdiction_code: str
    geography: str
    reference_period_start: date
    reference_period_end: date
    population_base: str
    universe: str
    unit_of_analysis: str
    eligibility: str
    exclusion: str
    age_min_inclusive: int
    age_max_exclusive: int
    household_income_definition: str
    household_income_currency: str
    household_income_period: str
    household_income_equivalisation: str
    household_definition: str
    gaming_definition: str
    payer_definition: str
    zero_spender_treatment: str
    estimand_role: PopulationEstimandRole
    status: ProvenanceStatus
    source_ids: tuple[str, ...]
    retrieved_on: date
    recipe_json: str

    def __post_init__(self) -> None:
        for name in ("binding_id", "artifact_id", "target_population_id"):
            _validate_id(getattr(self, name), name=name)
        _validate_jurisdiction_code(self.jurisdiction_code)
        for name in (
            "geography",
            "population_base",
            "universe",
            "unit_of_analysis",
            "eligibility",
            "exclusion",
            "household_income_definition",
            "household_income_period",
            "household_income_equivalisation",
            "household_definition",
            "gaming_definition",
            "payer_definition",
            "zero_spender_treatment",
        ):
            _validate_text(getattr(self, name), name=name)
        _validate_currency(
            self.household_income_currency,
            name="household_income_currency",
        )
        for name in (
            "reference_period_start",
            "reference_period_end",
            "retrieved_on",
        ):
            if type(getattr(self, name)) is not date:
                raise PopulationEvidenceValidationError(
                    f"population-evidence {name} must be an ISO calendar date"
                )
        if self.reference_period_end < self.reference_period_start:
            raise PopulationEvidenceValidationError(
                "population reference period ends before it starts"
            )
        if self.retrieved_on < self.reference_period_end:
            raise PopulationEvidenceValidationError(
                "population retrieval date cannot predate the reference-period end"
            )
        _validate_strict_int(
            self.age_min_inclusive,
            name="age_min_inclusive",
            minimum=0,
            maximum=199,
        )
        _validate_strict_int(
            self.age_max_exclusive,
            name="age_max_exclusive",
            minimum=1,
            maximum=200,
        )
        if self.age_max_exclusive <= self.age_min_inclusive:
            raise PopulationEvidenceValidationError(
                "target-population age scope is empty or reversed"
            )
        if type(self.estimand_role) is not PopulationEstimandRole:
            raise PopulationEvidenceValidationError(
                "population estimand_role must be CALIBRATION or VALIDATION"
            )
        if type(self.status) is not ProvenanceStatus:
            raise PopulationEvidenceValidationError(
                "population-evidence binding status is invalid"
            )
        if type(self.source_ids) is not tuple or not self.source_ids:
            raise PopulationEvidenceValidationError(
                "population-evidence source_ids must be a non-empty immutable tuple"
            )
        for source_id in self.source_ids:
            _validate_id(source_id, name="source_id")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise PopulationEvidenceValidationError(
                "population-evidence binding repeats a source id"
            )
        if self.source_ids != tuple(sorted(self.source_ids)):
            raise PopulationEvidenceValidationError(
                "population-evidence source_ids must use ascending order"
            )
        recipe = _parse_recipe_json(self.recipe_json)
        if recipe["row_match"] != self.row_match:
            raise PopulationEvidenceValidationError(
                f"binding {self.binding_id} recipe row_match does not exactly "
                "match its typed metadata"
            )

    @property
    def row_match(self) -> dict[str, str]:
        return {
            "estimand_role": self.estimand_role.value,
            "jurisdiction_code": self.jurisdiction_code,
            "target_population_id": self.target_population_id,
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
            "target_population_id": self.target_population_id,
            "jurisdiction_code": self.jurisdiction_code,
            "geography": self.geography,
            "reference_period_start": self.reference_period_start.isoformat(),
            "reference_period_end": self.reference_period_end.isoformat(),
            "population_base": self.population_base,
            "universe": self.universe,
            "unit_of_analysis": self.unit_of_analysis,
            "eligibility": self.eligibility,
            "exclusion": self.exclusion,
            "age_min_inclusive": self.age_min_inclusive,
            "age_max_exclusive": self.age_max_exclusive,
            "household_income_definition": self.household_income_definition,
            "household_income_currency": self.household_income_currency,
            "household_income_period": self.household_income_period,
            "household_income_equivalisation": self.household_income_equivalisation,
            "household_definition": self.household_definition,
            "gaming_definition": self.gaming_definition,
            "payer_definition": self.payer_definition,
            "zero_spender_treatment": self.zero_spender_treatment,
            "estimand_role": self.estimand_role.value,
            "status": self.status.value,
            "source_ids": list(self.source_ids),
            "retrieved_on": self.retrieved_on.isoformat(),
            "recipe_json": self.recipe_json,
        }


@dataclass(frozen=True, slots=True)
class PopulationEvidenceResult:
    """Verified joint-cell extraction tied to all declared input identities."""

    bundle_sha256: str
    source_registry_sha256: str
    binding_id: str
    binding_sha256: str
    artifact_id: str
    artifact_sha256: str
    artifact_byte_length: int
    recipe_sha256: str
    target_population_id: str
    jurisdiction_code: str
    estimand_role: PopulationEstimandRole
    cells: tuple[PopulationEvidenceCell, ...]
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
        for name in ("binding_id", "artifact_id", "target_population_id"):
            _validate_id(getattr(self, name), name=name)
        _validate_jurisdiction_code(self.jurisdiction_code)
        if type(self.estimand_role) is not PopulationEstimandRole:
            raise PopulationEvidenceValidationError(
                "population result estimand_role is invalid"
            )
        _validate_strict_int(
            self.artifact_byte_length,
            name="artifact_byte_length",
            minimum=1,
            maximum=MAX_POPULATION_ARTIFACT_BYTES,
        )
        _validate_cells(self.cells)
        expected = sha256(
            _canonical_json(self.attestation_payload()).encode("utf-8")
        ).hexdigest()
        if self.evidence_sha256 != expected:
            raise PopulationEvidenceValidationError(
                "evidence_sha256 does not match the population result payload"
            )

    @property
    def cells_sha256(self) -> str:
        return sha256(
            _canonical_json([cell.snapshot() for cell in self.cells]).encode("utf-8")
        ).hexdigest()

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": POPULATION_EVIDENCE_SCHEMA_VERSION,
            "bundle_sha256": self.bundle_sha256,
            "source_registry_sha256": self.source_registry_sha256,
            "binding_id": self.binding_id,
            "binding_sha256": self.binding_sha256,
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "artifact_byte_length": self.artifact_byte_length,
            "recipe_sha256": self.recipe_sha256,
            "target_population_id": self.target_population_id,
            "jurisdiction_code": self.jurisdiction_code,
            "estimand_role": self.estimand_role.value,
            "cell_count": len(self.cells),
            "cells_sha256": self.cells_sha256,
            "cells": [cell.snapshot() for cell in self.cells],
            "total_mass_numerator": 1,
            "total_mass_denominator": 1,
            "total_mass_numerator_decimal": "1",
            "total_mass_denominator_decimal": "1",
        }

    def snapshot(self) -> dict[str, object]:
        return {**self.attestation_payload(), "evidence_sha256": self.evidence_sha256}


@dataclass(frozen=True, slots=True)
class PopulationEvidenceBundle:
    """Parsed schema-v1 population bundle plus observed file identity."""

    schema_version: int
    bundle_id: str
    provenance_status: ProvenanceStatus
    source_registry_sha256: str
    artifact_root: str
    notes: str
    artifacts: tuple[PopulationEvidenceArtifact, ...]
    bindings: tuple[PopulationEvidenceBinding, ...]
    signature: PopulationEvidenceSignature
    bundle_path: Path
    bundle_sha256: str
    bundle_byte_length: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != POPULATION_EVIDENCE_SCHEMA_VERSION
        ):
            raise PopulationEvidenceValidationError(
                "unsupported population-evidence schema version: "
                f"{self.schema_version!r}"
            )
        _validate_id(self.bundle_id, name="bundle_id")
        if type(self.provenance_status) is not ProvenanceStatus:
            raise PopulationEvidenceValidationError(
                "invalid population bundle provenance status"
            )
        _validate_sha256(
            self.source_registry_sha256,
            name="source_registry_sha256",
        )
        _validate_relative_posix_path(self.artifact_root, name="artifact_root")
        _validate_text(self.notes, name="notes")
        if type(self.artifacts) is not tuple or any(
            type(item) is not PopulationEvidenceArtifact for item in self.artifacts
        ):
            raise PopulationEvidenceValidationError(
                "population-evidence artifacts must be an immutable typed tuple"
            )
        if type(self.bindings) is not tuple or any(
            type(item) is not PopulationEvidenceBinding for item in self.bindings
        ):
            raise PopulationEvidenceValidationError(
                "population-evidence bindings must be an immutable typed tuple"
            )
        if type(self.signature) is not PopulationEvidenceSignature:
            raise PopulationEvidenceValidationError(
                "invalid population-evidence signature"
            )
        if not isinstance(self.bundle_path, Path) or not self.bundle_path.is_absolute():
            raise PopulationEvidenceValidationError(
                "population bundle_path must be an absolute Path"
            )
        lexical_path = Path(os.path.normpath(os.fspath(self.bundle_path)))
        if ".." in self.bundle_path.parts or lexical_path != self.bundle_path:
            raise PopulationEvidenceValidationError(
                "population bundle_path must be lexically canonical without "
                "dot segments"
            )
        _validate_sha256(self.bundle_sha256, name="bundle_sha256")
        _validate_strict_int(
            self.bundle_byte_length,
            name="bundle_byte_length",
            minimum=1,
            maximum=MAX_POPULATION_BUNDLE_BYTES,
        )

        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        artifact_paths = tuple(item.relative_path for item in self.artifacts)
        binding_ids = tuple(item.binding_id for item in self.bindings)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise PopulationEvidenceValidationError(
                "population-evidence artifact ids repeat"
            )
        if len(set(binding_ids)) != len(binding_ids):
            raise PopulationEvidenceValidationError(
                "population-evidence binding ids repeat"
            )
        if len({path.casefold() for path in artifact_paths}) != len(artifact_paths):
            raise PopulationEvidenceValidationError(
                "population-evidence artifact paths repeat under "
                "case-insensitive lookup"
            )
        if artifact_ids != tuple(sorted(artifact_ids)):
            raise PopulationEvidenceValidationError(
                "population-evidence artifacts must use ascending artifact_id order"
            )
        if binding_ids != tuple(sorted(binding_ids)):
            raise PopulationEvidenceValidationError(
                "population-evidence bindings must use ascending binding_id order"
            )
        referenced = {binding.artifact_id for binding in self.bindings}
        declared = set(artifact_ids)
        if referenced != declared:
            raise PopulationEvidenceValidationError(
                "population-evidence artifacts must be referenced exactly; "
                f"missing={sorted(referenced - declared)}, "
                f"unreferenced={sorted(declared - referenced)}"
            )
        semantic_keys = tuple(
            (
                item.target_population_id,
                item.jurisdiction_code,
                item.estimand_role.value,
            )
            for item in self.bindings
        )
        if len(set(semantic_keys)) != len(semantic_keys):
            raise PopulationEvidenceValidationError(
                "population-evidence bundle repeats a semantic target binding"
            )
        if (
            self.provenance_status is not ProvenanceStatus.CALIBRATED
            and any(
                binding.status is ProvenanceStatus.CALIBRATED
                for binding in self.bindings
            )
        ):
            raise PopulationEvidenceValidationError(
                "a non-calibrated population bundle cannot contain CALIBRATED bindings"
            )

    @property
    def campaign_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.provenance_status is not ProvenanceStatus.CALIBRATED:
            blockers.append(
                "population_evidence_bundle_status=" + self.provenance_status.value
            )
        blockers.append("population_evidence_bundle_signature_missing")
        if not self.artifacts:
            blockers.append("population_evidence_bundle_empty")
        roles = {binding.estimand_role for binding in self.bindings}
        if PopulationEstimandRole.CALIBRATION not in roles:
            blockers.append("population_evidence_calibration_binding_missing")
        if PopulationEstimandRole.VALIDATION not in roles:
            blockers.append("population_evidence_validation_binding_missing")
        non_calibrated = tuple(
            binding.binding_id
            for binding in self.bindings
            if binding.status is not ProvenanceStatus.CALIBRATED
        )
        if non_calibrated:
            blockers.append(
                "population_evidence_non_calibrated_bindings="
                + ",".join(non_calibrated)
            )
        return tuple(blockers)

    @property
    def campaign_ready(self) -> bool:
        return False

    def validate_for_campaign(self) -> None:
        raise PopulationEvidenceVerificationError(
            "population-evidence schema v1 is not campaign-ready: "
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


def exact_csv_joint_population_recipe_json(
    *,
    target_population_id: str,
    jurisdiction_code: str,
    estimand_role: PopulationEstimandRole,
) -> str:
    """Build the only canonical joint-cell recipe accepted by schema v1."""

    _validate_id(target_population_id, name="target_population_id")
    _validate_jurisdiction_code(jurisdiction_code)
    if type(estimand_role) is not PopulationEstimandRole:
        raise PopulationEvidenceValidationError(
            "estimand_role must be CALIBRATION or VALIDATION"
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "interpreter": EXACT_CSV_JOINT_POPULATION_INTERPRETER_V1,
        "row_match": {
            "estimand_role": estimand_role.value,
            "jurisdiction_code": jurisdiction_code,
            "target_population_id": target_population_id,
        },
        "cell_columns": list(POPULATION_CELL_CSV_COLUMNS),
    }
    return _canonical_json(payload)


def load_population_evidence_bundle(
    path: str | Path = DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH,
    *,
    expected_source_registry_sha256: str | None = None,
) -> PopulationEvidenceBundle:
    """Parse one strict schema-v1 bundle without trusting artifact claims."""

    bundle_path = Path(path)
    observed = _secure_read_regular_file(
        bundle_path,
        maximum_bytes=MAX_POPULATION_BUNDLE_BYTES,
        description="population-evidence bundle",
    )
    try:
        text = observed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PopulationEvidenceValidationError(
            "population-evidence bundle must be UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise PopulationEvidenceValidationError(
            "population-evidence bundle must not contain a UTF-8 BOM"
        )
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PopulationEvidenceValidationError(
            f"invalid population-evidence TOML: {exc}"
        ) from exc
    _require_mapping(raw, name="population-evidence root")
    _exact_keys(raw, _TOP_LEVEL_KEYS, name="population-evidence root")
    schema_version = _required_int(raw, "schema_version", minimum=1)
    if schema_version != POPULATION_EVIDENCE_SCHEMA_VERSION:
        raise PopulationEvidenceValidationError(
            f"unsupported population-evidence schema version: {schema_version}"
        )
    bundle = PopulationEvidenceBundle(
        schema_version=schema_version,
        bundle_id=_required_string(raw, "bundle_id"),
        provenance_status=_parse_provenance_status(
            _required_string(raw, "provenance_status")
        ),
        source_registry_sha256=_required_string(
            raw,
            "source_registry_sha256",
        ),
        artifact_root=_required_string(raw, "artifact_root"),
        notes=_required_string(raw, "notes"),
        artifacts=tuple(
            _parse_artifact(row)
            for row in _required_table_list(raw, "artifacts")
        ),
        bindings=tuple(
            _parse_binding(row)
            for row in _required_table_list(raw, "bindings")
        ),
        signature=_parse_signature(_required_table(raw, "signature")),
        bundle_path=bundle_path.resolve(strict=True),
        bundle_sha256=sha256(observed).hexdigest(),
        bundle_byte_length=len(observed),
    )
    _validate_expected_source_registry(
        bundle.source_registry_sha256,
        expected_source_registry_sha256,
    )
    return bundle


def verify_population_evidence_bundle(
    bundle: PopulationEvidenceBundle,
    *,
    expected_source_registry_sha256: str | None = None,
) -> tuple[PopulationEvidenceResult, ...]:
    """Reopen the bundle and every artifact, then rerun every exact recipe."""

    if type(bundle) is not PopulationEvidenceBundle:
        raise TypeError("bundle must be a PopulationEvidenceBundle")
    reloaded = load_population_evidence_bundle(bundle.bundle_path)
    if reloaded != bundle:
        raise PopulationEvidenceVerificationError(
            "population-evidence bundle metadata no longer match its declared file"
        )
    _validate_expected_source_registry(
        bundle.source_registry_sha256,
        expected_source_registry_sha256,
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
        description="population-evidence artifact root",
    )
    resolved_root = artifact_root.resolve(strict=True)
    resolved_parent = bundle_parent.resolve(strict=True)
    if not resolved_root.is_relative_to(resolved_parent):
        raise PopulationEvidenceVerificationError(
            "population-evidence artifact root escapes the bundle directory"
        )

    artifacts_by_id = {artifact.artifact_id: artifact for artifact in bundle.artifacts}
    rows_by_artifact: dict[str, tuple[dict[str, str], ...]] = {}
    for artifact in bundle.artifacts:
        candidate = artifact_root.joinpath(
            *PurePosixPath(artifact.relative_path).parts
        )
        _assert_path_chain_without_links(
            candidate,
            chain_start=artifact_root,
            description=f"population artifact {artifact.artifact_id}",
        )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise PopulationEvidenceVerificationError(
                f"population artifact {artifact.artifact_id} cannot be resolved"
            ) from exc
        if not resolved.is_relative_to(resolved_root):
            raise PopulationEvidenceVerificationError(
                f"population artifact {artifact.artifact_id} escapes its declared root"
            )
        content = _secure_read_regular_file(
            candidate,
            expected_byte_length=artifact.byte_length,
            expected_sha256=artifact.sha256,
            maximum_bytes=MAX_POPULATION_ARTIFACT_BYTES,
            description=f"population artifact {artifact.artifact_id}",
        )
        _assert_path_chain_without_links(
            candidate,
            chain_start=artifact_root,
            description=f"population artifact {artifact.artifact_id}",
        )
        rows_by_artifact[artifact.artifact_id] = _parse_population_csv_rows(
            content,
            artifact_id=artifact.artifact_id,
        )

    bindings_by_artifact: dict[str, tuple[PopulationEvidenceBinding, ...]] = {}
    for artifact in bundle.artifacts:
        bindings_by_artifact[artifact.artifact_id] = tuple(
            binding
            for binding in bundle.bindings
            if binding.artifact_id == artifact.artifact_id
        )
    for artifact_id, rows in rows_by_artifact.items():
        candidates = bindings_by_artifact[artifact_id]
        for index, row in enumerate(rows, start=2):
            owners = tuple(
                binding
                for binding in candidates
                if all(row[key] == value for key, value in binding.row_match.items())
            )
            if len(owners) != 1:
                raise PopulationEvidenceVerificationError(
                    f"population artifact {artifact_id} row {index} must be owned "
                    f"by exactly one binding; matched={len(owners)}"
                )

    results: list[PopulationEvidenceResult] = []
    for binding in bundle.bindings:
        _parse_recipe_json(binding.recipe_json)
        selected = tuple(
            row
            for row in rows_by_artifact[binding.artifact_id]
            if all(row[key] == value for key, value in binding.row_match.items())
        )
        if not selected:
            raise PopulationEvidenceVerificationError(
                f"binding {binding.binding_id} must match at least one CSV row"
            )
        cells = tuple(
            _cell_from_csv_row(row, binding=binding)
            for row in selected
        )
        _validate_cells_for_binding(cells, binding=binding)
        results.append(
            _build_result(
                bundle=bundle,
                binding=binding,
                artifact=artifacts_by_id[binding.artifact_id],
                cells=cells,
            )
        )
    return tuple(results)


def load_and_verify_population_evidence_bundle(
    path: str | Path = DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH,
    *,
    expected_source_registry_sha256: str | None = None,
) -> tuple[PopulationEvidenceBundle, tuple[PopulationEvidenceResult, ...]]:
    """Load and immediately re-attest a population-evidence bundle."""

    bundle = load_population_evidence_bundle(
        path,
        expected_source_registry_sha256=expected_source_registry_sha256,
    )
    return bundle, verify_population_evidence_bundle(
        bundle,
        expected_source_registry_sha256=expected_source_registry_sha256,
    )


def validate_population_evidence_snapshot(
    bundle_snapshot: object,
    result_snapshots: object,
) -> tuple[
    PopulationEvidenceBundle | None,
    tuple[PopulationEvidenceResult, ...],
]:
    """Rebuild and re-attest serialized population evidence fail-closed."""

    if type(result_snapshots) is not list:
        raise PopulationEvidenceValidationError(
            "population-evidence result snapshots must be an array"
        )
    if bundle_snapshot is None:
        if result_snapshots:
            raise PopulationEvidenceValidationError(
                "population-evidence results require a bundle snapshot"
            )
        return None, ()
    bundle_row = _require_mapping(
        bundle_snapshot,
        name="population-evidence bundle snapshot",
    )
    _exact_keys(
        bundle_row,
        _BUNDLE_SNAPSHOT_KEYS,
        name="population-evidence bundle snapshot",
    )
    if bundle_row.get("campaign_ready") is not False:
        raise PopulationEvidenceValidationError(
            "population-evidence schema v1 campaign_ready must be false"
        )
    blockers = bundle_row.get("campaign_blockers")
    if type(blockers) is not list or any(
        type(blocker) is not str or not blocker for blocker in blockers
    ):
        raise PopulationEvidenceValidationError(
            "population-evidence campaign blockers are malformed"
        )
    artifacts = tuple(
        _parse_artifact(row)
        for row in _required_table_list(bundle_row, "artifacts")
    )
    bindings = tuple(
        _binding_from_snapshot(row)
        for row in _required_table_list(bundle_row, "bindings")
    )
    bundle = PopulationEvidenceBundle(
        schema_version=_required_int(bundle_row, "schema_version", minimum=1),
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
        signature=_parse_signature(_required_table(bundle_row, "signature")),
        bundle_path=Path(_required_string(bundle_row, "bundle_path")),
        bundle_sha256=_required_string(bundle_row, "bundle_sha256"),
        bundle_byte_length=_required_int(
            bundle_row,
            "bundle_byte_length",
            minimum=1,
            maximum=MAX_POPULATION_BUNDLE_BYTES,
        ),
    )
    if blockers != list(bundle.campaign_blockers):
        raise PopulationEvidenceValidationError(
            "population-evidence campaign blockers are not canonical"
        )
    if bundle.snapshot() != dict(bundle_row):
        raise PopulationEvidenceValidationError(
            "population-evidence bundle snapshot is not canonical"
        )

    results = tuple(
        _result_from_snapshot(
            _require_mapping(row, name=f"population result[{index}]")
        )
        for index, row in enumerate(result_snapshots)
    )
    if tuple(result.binding_id for result in results) != tuple(
        binding.binding_id for binding in bindings
    ):
        raise PopulationEvidenceValidationError(
            "population-evidence results do not exactly cover ordered bindings"
        )
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
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
            or result.target_population_id != binding.target_population_id
            or result.jurisdiction_code != binding.jurisdiction_code
            or result.estimand_role is not binding.estimand_role
        ):
            raise PopulationEvidenceValidationError(
                "population-evidence result does not match its bundle declarations"
            )
        _validate_cells_for_binding(result.cells, binding=binding)
    observed = verify_population_evidence_bundle(
        bundle,
        expected_source_registry_sha256=bundle.source_registry_sha256,
    )
    if observed != results:
        raise PopulationEvidenceValidationError(
            "population-evidence result snapshots do not match re-extracted bytes"
        )
    return bundle, results


def _build_result(
    *,
    bundle: PopulationEvidenceBundle,
    binding: PopulationEvidenceBinding,
    artifact: PopulationEvidenceArtifact,
    cells: tuple[PopulationEvidenceCell, ...],
) -> PopulationEvidenceResult:
    cell_snapshots = [cell.snapshot() for cell in cells]
    cells_sha256 = sha256(
        _canonical_json(cell_snapshots).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": POPULATION_EVIDENCE_SCHEMA_VERSION,
        "bundle_sha256": bundle.bundle_sha256,
        "source_registry_sha256": bundle.source_registry_sha256,
        "binding_id": binding.binding_id,
        "binding_sha256": binding.binding_sha256,
        "artifact_id": artifact.artifact_id,
        "artifact_sha256": artifact.sha256,
        "artifact_byte_length": artifact.byte_length,
        "recipe_sha256": binding.recipe_sha256,
        "target_population_id": binding.target_population_id,
        "jurisdiction_code": binding.jurisdiction_code,
        "estimand_role": binding.estimand_role.value,
        "cell_count": len(cells),
        "cells_sha256": cells_sha256,
        "cells": cell_snapshots,
        "total_mass_numerator": 1,
        "total_mass_denominator": 1,
        "total_mass_numerator_decimal": "1",
        "total_mass_denominator_decimal": "1",
    }
    return PopulationEvidenceResult(
        bundle_sha256=bundle.bundle_sha256,
        source_registry_sha256=bundle.source_registry_sha256,
        binding_id=binding.binding_id,
        binding_sha256=binding.binding_sha256,
        artifact_id=artifact.artifact_id,
        artifact_sha256=artifact.sha256,
        artifact_byte_length=artifact.byte_length,
        recipe_sha256=binding.recipe_sha256,
        target_population_id=binding.target_population_id,
        jurisdiction_code=binding.jurisdiction_code,
        estimand_role=binding.estimand_role,
        cells=cells,
        evidence_sha256=sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    )


def _binding_from_snapshot(
    row: Mapping[str, object],
) -> PopulationEvidenceBinding:
    _exact_keys(
        row,
        _BINDING_KEYS,
        name="population-evidence binding snapshot",
    )
    parsed = dict(row)
    for field in (
        "reference_period_start",
        "reference_period_end",
        "retrieved_on",
    ):
        parsed[field] = _snapshot_date(row.get(field), name=field)
    binding = _parse_binding(parsed)
    if binding.snapshot() != dict(row):
        raise PopulationEvidenceValidationError(
            "population-evidence binding snapshot is not canonical"
        )
    return binding


def _cell_from_snapshot(row: Mapping[str, object]) -> PopulationEvidenceCell:
    _exact_keys(row, _CELL_KEYS, name="population cell snapshot")
    numerator = _required_int(row, "target_mass_numerator", minimum=0)
    denominator = _required_int(row, "target_mass_denominator", minimum=1)
    if (
        row.get("target_mass_numerator_decimal") != str(numerator)
        or row.get("target_mass_denominator_decimal") != str(denominator)
    ):
        raise PopulationEvidenceValidationError(
            "population cell mass decimal mirrors are not lossless"
        )
    try:
        gaming_state = PopulationGamingState(
            _required_string(row, "gaming_state")
        )
        payer_state = PopulationPayerHistoryState(
            _required_string(row, "payer_history_state")
        )
    except ValueError as exc:
        raise PopulationEvidenceValidationError(
            "population cell contains an invalid categorical state"
        ) from exc
    cell = PopulationEvidenceCell(
        cell_id=_required_string(row, "cell_id"),
        age_min_inclusive=_required_int(row, "age_min_inclusive", minimum=0),
        age_max_exclusive=_required_int(row, "age_max_exclusive", minimum=1),
        household_income_band=_required_string(row, "household_income_band"),
        household_type=_required_string(row, "household_type"),
        gaming_state=gaming_state,
        payer_history_state=payer_state,
        target_mass_numerator=numerator,
        target_mass_denominator=denominator,
    )
    if cell.snapshot() != dict(row):
        raise PopulationEvidenceValidationError(
            "population cell snapshot is not canonical"
        )
    return cell


def _result_from_snapshot(
    row: Mapping[str, object],
) -> PopulationEvidenceResult:
    _exact_keys(row, _RESULT_KEYS, name="population-evidence result snapshot")
    schema_version = _required_int(row, "schema_version", minimum=1)
    if schema_version != POPULATION_EVIDENCE_SCHEMA_VERSION:
        raise PopulationEvidenceValidationError(
            "population result has an unsupported schema version"
        )
    cells_raw = row.get("cells")
    if type(cells_raw) is not list:
        raise PopulationEvidenceValidationError(
            "population result cells must be an array"
        )
    cells = tuple(
        _cell_from_snapshot(
            _require_mapping(cell, name=f"population cell[{index}]")
        )
        for index, cell in enumerate(cells_raw)
    )
    if _required_int(row, "cell_count", minimum=1) != len(cells):
        raise PopulationEvidenceValidationError(
            "population result cell_count does not match its cells"
        )
    observed_cells_sha256 = sha256(
        _canonical_json([cell.snapshot() for cell in cells]).encode("utf-8")
    ).hexdigest()
    if _required_string(row, "cells_sha256") != observed_cells_sha256:
        raise PopulationEvidenceValidationError(
            "population result cells_sha256 does not match its cells"
        )
    total_numerator = _required_int(row, "total_mass_numerator", minimum=1)
    total_denominator = _required_int(row, "total_mass_denominator", minimum=1)
    if (
        (total_numerator, total_denominator) != (1, 1)
        or row.get("total_mass_numerator_decimal") != "1"
        or row.get("total_mass_denominator_decimal") != "1"
    ):
        raise PopulationEvidenceValidationError(
            "population result total mass must be canonical 1/1"
        )
    try:
        role = PopulationEstimandRole(_required_string(row, "estimand_role"))
    except ValueError as exc:
        raise PopulationEvidenceValidationError(
            "population result estimand_role is invalid"
        ) from exc
    result = PopulationEvidenceResult(
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
            maximum=MAX_POPULATION_ARTIFACT_BYTES,
        ),
        recipe_sha256=_required_string(row, "recipe_sha256"),
        target_population_id=_required_string(row, "target_population_id"),
        jurisdiction_code=_required_string(row, "jurisdiction_code"),
        estimand_role=role,
        cells=cells,
        evidence_sha256=_required_string(row, "evidence_sha256"),
    )
    if result.snapshot() != dict(row):
        raise PopulationEvidenceValidationError(
            "population-evidence result snapshot is not canonical"
        )
    return result


def _snapshot_date(value: object, *, name: str) -> date:
    if type(value) is not str:
        raise PopulationEvidenceValidationError(
            f"population snapshot {name} must be an ISO date"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise PopulationEvidenceValidationError(
            f"population snapshot {name} must be an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise PopulationEvidenceValidationError(
            f"population snapshot {name} must be canonical"
        )
    return parsed


def _parse_artifact(
    row: Mapping[str, object],
) -> PopulationEvidenceArtifact:
    _exact_keys(row, _ARTIFACT_KEYS, name="population-evidence artifact")
    return PopulationEvidenceArtifact(
        artifact_id=_required_string(row, "artifact_id"),
        relative_path=_required_string(row, "relative_path"),
        media_type=_required_string(row, "media_type"),
        sha256=_required_string(row, "sha256"),
        byte_length=_required_int(
            row,
            "byte_length",
            minimum=1,
            maximum=MAX_POPULATION_ARTIFACT_BYTES,
        ),
    )


def _parse_binding(
    row: Mapping[str, object],
) -> PopulationEvidenceBinding:
    _exact_keys(row, _BINDING_KEYS, name="population-evidence binding")
    try:
        role = PopulationEstimandRole(_required_string(row, "estimand_role"))
    except ValueError as exc:
        raise PopulationEvidenceValidationError(
            "population binding estimand_role must be CALIBRATION or VALIDATION"
        ) from exc
    raw_source_ids = row.get("source_ids")
    if type(raw_source_ids) is not list or not raw_source_ids or any(
        type(source_id) is not str for source_id in raw_source_ids
    ):
        raise PopulationEvidenceValidationError(
            "population binding source_ids must be a non-empty string array"
        )
    return PopulationEvidenceBinding(
        binding_id=_required_string(row, "binding_id"),
        artifact_id=_required_string(row, "artifact_id"),
        target_population_id=_required_string(row, "target_population_id"),
        jurisdiction_code=_required_string(row, "jurisdiction_code"),
        geography=_required_string(row, "geography"),
        reference_period_start=_required_date(row, "reference_period_start"),
        reference_period_end=_required_date(row, "reference_period_end"),
        population_base=_required_string(row, "population_base"),
        universe=_required_string(row, "universe"),
        unit_of_analysis=_required_string(row, "unit_of_analysis"),
        eligibility=_required_string(row, "eligibility"),
        exclusion=_required_string(row, "exclusion"),
        age_min_inclusive=_required_int(row, "age_min_inclusive", minimum=0),
        age_max_exclusive=_required_int(row, "age_max_exclusive", minimum=1),
        household_income_definition=_required_string(
            row,
            "household_income_definition",
        ),
        household_income_currency=_required_string(
            row,
            "household_income_currency",
        ),
        household_income_period=_required_string(
            row,
            "household_income_period",
        ),
        household_income_equivalisation=_required_string(
            row,
            "household_income_equivalisation",
        ),
        household_definition=_required_string(row, "household_definition"),
        gaming_definition=_required_string(row, "gaming_definition"),
        payer_definition=_required_string(row, "payer_definition"),
        zero_spender_treatment=_required_string(row, "zero_spender_treatment"),
        estimand_role=role,
        status=_parse_provenance_status(_required_string(row, "status")),
        source_ids=tuple(raw_source_ids),
        retrieved_on=_required_date(row, "retrieved_on"),
        recipe_json=_required_string(row, "recipe_json"),
    )


def _parse_signature(
    row: Mapping[str, object],
) -> PopulationEvidenceSignature:
    _exact_keys(row, _SIGNATURE_KEYS, name="population-evidence signature")
    try:
        signature_status = PopulationEvidenceSignatureStatus(
            _required_string(row, "status")
        )
    except ValueError as exc:
        raise PopulationEvidenceValidationError(
            "population-evidence schema v1 signature status must be MISSING"
        ) from exc
    return PopulationEvidenceSignature(
        status=signature_status,
        algorithm=_required_string(row, "algorithm", allow_empty=True),
        key_id=_required_string(row, "key_id", allow_empty=True),
        value=_required_string(row, "value", allow_empty=True),
    )


def _parse_recipe_json(recipe_json: object) -> dict[str, object]:
    if type(recipe_json) is not str or not recipe_json:
        raise PopulationEvidenceValidationError(
            "population-evidence recipe_json must be text"
        )
    try:
        parsed = json.loads(recipe_json)
    except (json.JSONDecodeError, ValueError) as exc:
        raise PopulationEvidenceValidationError(
            "population-evidence recipe_json must be valid finite JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise PopulationEvidenceValidationError(
            "population-evidence recipe root must be an object"
        )
    _exact_keys(parsed, _RECIPE_KEYS, name="population-evidence recipe")
    try:
        canonical = _canonical_json(parsed)
    except (TypeError, ValueError) as exc:
        raise PopulationEvidenceValidationError(
            "population-evidence recipe must contain finite JSON values"
        ) from exc
    if canonical != recipe_json:
        raise PopulationEvidenceValidationError(
            "population-evidence recipe_json must use canonical JSON"
        )
    if type(parsed["schema_version"]) is not int or parsed["schema_version"] != 1:
        raise PopulationEvidenceValidationError(
            "population-evidence recipe schema_version must be strict integer 1"
        )
    if parsed["interpreter"] != EXACT_CSV_JOINT_POPULATION_INTERPRETER_V1:
        raise PopulationEvidenceValidationError(
            "population-evidence recipe uses a non-whitelisted interpreter"
        )
    row_match = parsed["row_match"]
    if not isinstance(row_match, dict):
        raise PopulationEvidenceValidationError(
            "population recipe row_match must be an object"
        )
    _exact_keys(row_match, _ROW_MATCH_KEY_SET, name="population recipe row_match")
    if any(type(value) is not str or not value for value in row_match.values()):
        raise PopulationEvidenceValidationError(
            "population recipe row_match values must be non-empty strings"
        )
    _validate_id(row_match["target_population_id"], name="target_population_id")
    _validate_jurisdiction_code(row_match["jurisdiction_code"])
    try:
        PopulationEstimandRole(row_match["estimand_role"])
    except ValueError as exc:
        raise PopulationEvidenceValidationError(
            "population recipe estimand_role is invalid"
        ) from exc
    if parsed["cell_columns"] != list(POPULATION_CELL_CSV_COLUMNS):
        raise PopulationEvidenceValidationError(
            "population recipe cell_columns must match the schema-v1 exact header"
        )
    return parsed


def _parse_population_csv_rows(
    content: bytes,
    *,
    artifact_id: str,
) -> tuple[dict[str, str], ...]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PopulationEvidenceVerificationError(
            f"population artifact {artifact_id} must be strict UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise PopulationEvidenceVerificationError(
            f"population artifact {artifact_id} must not contain a UTF-8 BOM"
        )
    if "\x00" in text:
        raise PopulationEvidenceVerificationError(
            f"population artifact {artifact_id} contains a NUL character"
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
        raise PopulationEvidenceVerificationError(
            f"population artifact {artifact_id} is not strict CSV"
        ) from exc
    if len(rows) < 2:
        raise PopulationEvidenceVerificationError(
            f"population artifact {artifact_id} needs a header and data row"
        )
    if rows[0] != list(POPULATION_CELL_CSV_COLUMNS):
        raise PopulationEvidenceVerificationError(
            f"population artifact {artifact_id} header does not exactly match "
            "the schema-v1 joint-cell columns"
        )
    width = len(POPULATION_CELL_CSV_COLUMNS)
    mapped_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows[1:], start=2):
        if not row or len(row) != width:
            raise PopulationEvidenceVerificationError(
                f"population artifact {artifact_id} row {index} has wrong width"
            )
        if any("\r" in value or "\n" in value for value in row):
            raise PopulationEvidenceVerificationError(
                f"population artifact {artifact_id} row {index} has multiline data"
            )
        mapped = dict(zip(POPULATION_CELL_CSV_COLUMNS, row, strict=True))
        _validate_csv_selector_fields(mapped, artifact_id=artifact_id, row=index)
        mapped_rows.append(mapped)
    return tuple(mapped_rows)


def _validate_csv_selector_fields(
    row_value: Mapping[str, str],
    *,
    artifact_id: str,
    row: int,
) -> None:
    try:
        _validate_id(
            row_value["target_population_id"],
            name="target_population_id",
        )
        _validate_jurisdiction_code(row_value["jurisdiction_code"])
        PopulationEstimandRole(row_value["estimand_role"])
    except (PopulationEvidenceValidationError, ValueError) as exc:
        raise PopulationEvidenceVerificationError(
            f"population artifact {artifact_id} row {row} has invalid selector fields"
        ) from exc


def _cell_from_csv_row(
    row: Mapping[str, str],
    *,
    binding: PopulationEvidenceBinding,
) -> PopulationEvidenceCell:
    try:
        gaming_state = PopulationGamingState(row["gaming_state"])
        payer_state = PopulationPayerHistoryState(row["payer_history_state"])
    except ValueError as exc:
        raise PopulationEvidenceVerificationError(
            f"binding {binding.binding_id} CSV contains an invalid categorical state"
        ) from exc
    cell = PopulationEvidenceCell(
        cell_id=row["cell_id"],
        age_min_inclusive=_parse_nonnegative_canonical_integer(
            row["age_min_inclusive"],
            name=f"binding {binding.binding_id} age_min_inclusive",
        ),
        age_max_exclusive=_parse_positive_canonical_integer(
            row["age_max_exclusive"],
            name=f"binding {binding.binding_id} age_max_exclusive",
        ),
        household_income_band=row["household_income_band"],
        household_type=row["household_type"],
        gaming_state=gaming_state,
        payer_history_state=payer_state,
        target_mass_numerator=_parse_nonnegative_canonical_integer(
            row["target_mass_numerator"],
            name=f"binding {binding.binding_id} target_mass_numerator",
        ),
        target_mass_denominator=_parse_positive_canonical_integer(
            row["target_mass_denominator"],
            name=f"binding {binding.binding_id} target_mass_denominator",
        ),
    )
    return cell


def _validate_cells(cells: tuple[PopulationEvidenceCell, ...]) -> None:
    if type(cells) is not tuple or not cells or any(
        type(cell) is not PopulationEvidenceCell for cell in cells
    ):
        raise PopulationEvidenceValidationError(
            "population cells must be a non-empty immutable typed tuple"
        )
    cell_ids = tuple(cell.cell_id for cell in cells)
    if len(set(cell_ids)) != len(cell_ids):
        raise PopulationEvidenceValidationError("population cell ids repeat")
    if cell_ids != tuple(sorted(cell_ids)):
        raise PopulationEvidenceValidationError(
            "population cells must use ascending cell_id order"
        )
    semantic_keys = tuple(cell.semantic_key for cell in cells)
    if len(set(semantic_keys)) != len(semantic_keys):
        raise PopulationEvidenceValidationError(
            "population cells repeat a joint semantic cell"
        )
    strata: dict[
        tuple[str, str, str, str],
        list[PopulationEvidenceCell],
    ] = {}
    for cell in cells:
        stratum = (
            cell.household_income_band,
            cell.household_type,
            cell.gaming_state.value,
            cell.payer_history_state.value,
        )
        strata.setdefault(stratum, []).append(cell)
    for stratum, stratum_cells in strata.items():
        ordered = sorted(
            stratum_cells,
            key=lambda cell: (
                cell.age_min_inclusive,
                cell.age_max_exclusive,
                cell.cell_id,
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            if current.age_min_inclusive < previous.age_max_exclusive:
                raise PopulationEvidenceValidationError(
                    "population cells overlap in age within joint stratum "
                    f"{stratum}: {previous.cell_id}, {current.cell_id}"
                )
    total = sum((cell.target_mass for cell in cells), start=Fraction(0, 1))
    if total != Fraction(1, 1):
        raise PopulationEvidenceValidationError(
            f"population cell target masses must sum exactly to 1; observed={total}"
        )


def _validate_cells_for_binding(
    cells: tuple[PopulationEvidenceCell, ...],
    *,
    binding: PopulationEvidenceBinding,
) -> None:
    _validate_cells(cells)
    outside = tuple(
        cell.cell_id
        for cell in cells
        if cell.age_min_inclusive < binding.age_min_inclusive
        or cell.age_max_exclusive > binding.age_max_exclusive
    )
    if outside:
        raise PopulationEvidenceValidationError(
            f"binding {binding.binding_id} has cells outside its age scope: "
            + ", ".join(outside)
        )
    uncovered = tuple(
        age
        for age in range(binding.age_min_inclusive, binding.age_max_exclusive)
        if not any(
            cell.age_min_inclusive <= age < cell.age_max_exclusive
            for cell in cells
        )
    )
    if uncovered:
        raise PopulationEvidenceValidationError(
            f"binding {binding.binding_id} does not cover every age in its scope; "
            f"first_uncovered={uncovered[0]}"
        )


def _validate_expected_source_registry(
    observed: str,
    expected: str | None,
) -> None:
    if expected is None:
        return
    _validate_sha256(expected, name="expected_source_registry_sha256")
    if observed != expected:
        raise PopulationEvidenceVerificationError(
            "population-evidence bundle source_registry_sha256 does not match "
            "the expected source catalogue"
        )


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
        raise PopulationEvidenceVerificationError(
            f"{description} exceeds the {maximum_bytes}-byte safety limit"
        )
    if expected_byte_length is not None and before.st_size != expected_byte_length:
        raise PopulationEvidenceVerificationError(
            f"{description} byte length changed: expected {expected_byte_length}, "
            f"observed {before.st_size}"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise PopulationEvidenceVerificationError(
            f"cannot open {description}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _is_reparse(opened):
            raise PopulationEvidenceVerificationError(
                f"{description} must be a non-reparse regular file"
            )
        if not _same_file_identity(before, opened):
            raise PopulationEvidenceVerificationError(
                f"{description} changed while it was opened"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise PopulationEvidenceVerificationError(
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
        raise PopulationEvidenceVerificationError(f"{description} changed while read")
    content = b"".join(chunks)
    if len(content) != after_open.st_size:
        raise PopulationEvidenceVerificationError(
            f"{description} was not read completely"
        )
    if expected_byte_length is not None and len(content) != expected_byte_length:
        raise PopulationEvidenceVerificationError(
            f"{description} byte length does not match its declaration"
        )
    observed_sha256 = sha256(content).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise PopulationEvidenceVerificationError(
            f"{description} SHA-256 does not match its declaration"
        )
    return content


def _lstat_regular_file(path: Path, *, description: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise PopulationEvidenceVerificationError(
            f"{description} does not exist"
        ) from exc
    if path.is_symlink() or _is_reparse(observed):
        raise PopulationEvidenceVerificationError(
            f"{description} cannot be a symlink or reparse point"
        )
    if not stat.S_ISREG(observed.st_mode):
        raise PopulationEvidenceVerificationError(
            f"{description} must be a regular file"
        )
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
        raise PopulationEvidenceVerificationError(
            f"{description} does not exist"
        ) from exc
    if not stat.S_ISDIR(observed.st_mode):
        raise PopulationEvidenceVerificationError(
            f"{description} must be a directory"
        )


def _assert_path_chain_without_links(
    path: Path,
    *,
    chain_start: Path,
    description: str,
) -> None:
    try:
        relative = path.relative_to(chain_start)
    except ValueError as exc:
        raise PopulationEvidenceVerificationError(
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
            raise PopulationEvidenceVerificationError(
                f"{description} path component does not exist: {component.name}"
            ) from exc
        if component.is_symlink() or _is_reparse(observed):
            raise PopulationEvidenceVerificationError(
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
    if type(value) is not str or not value:
        raise PopulationEvidenceValidationError(f"{name} must be non-empty text")
    if "\\" in value:
        raise PopulationEvidenceValidationError(f"{name} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise PopulationEvidenceValidationError(
            f"{name} must be a canonical relative path"
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise PopulationEvidenceValidationError(
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
        raise PopulationEvidenceValidationError(
            f"{name} cannot be drive-qualified, an alternate stream, or a "
            "reserved platform path"
        )


def _validate_id(value: object, *, name: str) -> None:
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise PopulationEvidenceValidationError(
            f"{name} must be a canonical ASCII identifier"
        )


def _validate_text(value: object, *, name: str) -> None:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise PopulationEvidenceValidationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )


def _validate_jurisdiction_code(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 2
        or any(character < "A" or character > "Z" for character in value)
    ):
        raise PopulationEvidenceValidationError(
            "jurisdiction_code must be a two-letter uppercase ASCII code"
        )


def _validate_currency(value: object, *, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 3
        or any(character < "A" or character > "Z" for character in value)
    ):
        raise PopulationEvidenceValidationError(
            f"{name} must be a three-letter uppercase ASCII currency code"
        )


def _validate_sha256(value: object, *, name: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise PopulationEvidenceValidationError(
            f"{name} must be lowercase SHA-256 hex"
        )


def _validate_strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise PopulationEvidenceValidationError(f"{name} must be a strict integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise PopulationEvidenceValidationError(f"{name} must be {bound}")


def _parse_positive_canonical_integer(value: str, *, name: str) -> int:
    if _POSITIVE_INTEGER_PATTERN.fullmatch(value) is None:
        raise PopulationEvidenceVerificationError(
            f"{name} must be a canonical positive decimal integer"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise PopulationEvidenceVerificationError(
            f"{name} exceeds the supported exact-integer safety limit"
        ) from exc


def _parse_nonnegative_canonical_integer(value: str, *, name: str) -> int:
    if _NONNEGATIVE_INTEGER_PATTERN.fullmatch(value) is None:
        raise PopulationEvidenceVerificationError(
            f"{name} must be a canonical non-negative decimal integer"
        )
    try:
        return int(value)
    except ValueError as exc:
        raise PopulationEvidenceVerificationError(
            f"{name} exceeds the supported exact-integer safety limit"
        ) from exc


def _exact_keys(
    values: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if any(type(key) is not str for key in values):
        raise PopulationEvidenceValidationError(f"{name} keys must be strings")
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise PopulationEvidenceValidationError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PopulationEvidenceValidationError(f"{name} must be a table")
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
        raise PopulationEvidenceValidationError(
            f"{field} must be an array of tables"
        )
    return tuple(
        _require_mapping(value, name=f"{field}[{index}]")
        for index, value in enumerate(raw)
    )


def _required_string(
    values: Mapping[str, object],
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = values.get(field)
    if type(value) is not str or (not allow_empty and not value.strip()):
        qualifier = "text" if allow_empty else "non-empty text"
        raise PopulationEvidenceValidationError(f"{field} must be {qualifier}")
    if not allow_empty and value != value.strip():
        raise PopulationEvidenceValidationError(
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
        raise PopulationEvidenceValidationError(
            f"{field} must be an unquoted TOML calendar date"
        )
    return value


def _parse_provenance_status(value: str) -> ProvenanceStatus:
    try:
        return ProvenanceStatus(value)
    except ValueError as exc:
        raise PopulationEvidenceValidationError(
            f"invalid population-evidence provenance status: {value}"
        ) from exc


__all__ = [
    "DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH",
    "EXACT_CSV_JOINT_POPULATION_INTERPRETER_V1",
    "MAX_POPULATION_ARTIFACT_BYTES",
    "POPULATION_CELL_CSV_COLUMNS",
    "POPULATION_EVIDENCE_SCHEMA_VERSION",
    "PopulationCell",
    "PopulationEvidenceCell",
    "PopulationEstimandRole",
    "PopulationEvidenceArtifact",
    "PopulationEvidenceBinding",
    "PopulationEvidenceBundle",
    "PopulationEvidenceResult",
    "PopulationEvidenceSignature",
    "PopulationEvidenceSignatureStatus",
    "PopulationEvidenceValidationError",
    "PopulationEvidenceVerificationError",
    "PopulationGamingState",
    "PopulationPayerHistoryState",
    "exact_csv_joint_population_recipe_json",
    "load_and_verify_population_evidence_bundle",
    "load_population_evidence_bundle",
    "validate_population_evidence_snapshot",
    "verify_population_evidence_bundle",
]
