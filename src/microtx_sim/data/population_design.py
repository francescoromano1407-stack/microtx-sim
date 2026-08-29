"""Exact, fail-closed population-design and integer-apportionment contracts.

Schema version 1 is deliberately a static contract.  It binds an exact
population-evidence bundle, complete categorical domains, disjoint
calibration/validation source partitions, and one deterministic Hamilton
sample plan.  Runtime projection, output estimands, and balance results are
outside this schema, so even a complete schema-v1 design cannot by itself make
a campaign ready.

The record and cluster hashes in schema v1 are manifest declarations, not proof
against role-specific salting or source-unit aliases.  Without signed immutable
source-unit keys they cannot establish authenticity or held-out readiness; the
public readiness properties therefore remain false.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import tomllib
from typing import Mapping

from .population_evidence import (
    PopulationEstimandRole,
    PopulationEvidenceBinding,
    PopulationEvidenceBundle,
    PopulationEvidenceCell,
    PopulationEvidenceResult,
    PopulationEvidenceValidationError,
    PopulationGamingState,
    PopulationPayerHistoryState,
    validate_population_evidence_snapshot,
    verify_population_evidence_bundle,
)
from ..types import ProvenanceStatus


POPULATION_DESIGN_SCHEMA_VERSION = 1
EXACT_RATIONAL_HAMILTON_V1 = "exact_rational_hamilton/1"
CANONICAL_SOURCE_RECORD_ID_V1 = "canonical_source_record_sha256/1"
CANONICAL_SOURCE_CLUSTER_ID_V1 = "canonical_source_cluster_sha256/1"
SHA256_CLUSTER_THRESHOLD_V1 = "sha256_cluster_threshold/1"
MAX_POPULATION_DESIGN_BYTES = 64 * 1024 * 1024
MAX_TARGET_POPULATION_COUNT = (1 << 63) - 1
MAX_SAMPLE_PLAYER_COUNT = (1 << 63) - 1

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POPULATION_DESIGN_BUNDLE_PATH = (
    _PROJECT_ROOT / "data" / "provenance" / "population_design.toml"
)

_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CURRENCY_PATTERN = re.compile(r"[A-Z]{3}\Z")

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "design_id",
        "provenance_status",
        "notes",
        "population_evidence_bundle_sha256",
        "population_evidence_result_sha256s",
        "hamilton_recipe",
        "domains",
        "jurisdictions",
        "partition",
    }
)
_DOMAIN_KEYS = frozenset(
    {
        "income_missing_policy",
        "household_missing_policy",
        "gaming_states",
        "payer_history_states",
        "age_bands",
        "income_bands",
        "household_types",
    }
)
_AGE_BAND_KEYS = frozenset(
    {"ordinal", "age_band_id", "age_min_inclusive", "age_max_exclusive"}
)
_INCOME_BAND_KEYS = frozenset(
    {
        "ordinal",
        "jurisdiction_code",
        "income_band_id",
        "definition",
        "currency",
        "period",
        "lower_unbounded",
        "lower_bound_numerator",
        "lower_bound_denominator",
        "upper_unbounded",
        "upper_bound_numerator",
        "upper_bound_denominator",
    }
)
_HOUSEHOLD_TYPE_KEYS = frozenset(
    {"ordinal", "household_type_id", "definition"}
)
_JURISDICTION_KEYS = frozenset(
    {
        "jurisdiction_code",
        "target_population_count",
        "calibration_binding_id",
        "calibration_target_population_id",
        "calibration_evidence_sha256",
        "validation_binding_id",
        "validation_target_population_id",
        "validation_evidence_sha256",
    }
)
_PARTITION_KEYS = frozenset(
    {
        "identity_namespace",
        "record_id_recipe",
        "cluster_id_recipe",
        "role_assignment_recipe",
        "assignment_seed_sha256",
        "calibration_threshold_numerator",
        "calibration_threshold_denominator",
        "records",
    }
)
_PARTITION_RECORD_KEYS = frozenset(
    {
        "record_identity_sha256",
        "cluster_identity_sha256",
        "estimand_role",
        "binding_id",
        "cell_id",
        "record_weight_numerator",
        "record_weight_denominator",
    }
)

_AGE_BAND_SNAPSHOT_KEYS = frozenset(
    {*_AGE_BAND_KEYS}
)
_INCOME_BAND_SNAPSHOT_KEYS = frozenset(
    {
        *_INCOME_BAND_KEYS,
        "lower_bound_numerator_decimal",
        "lower_bound_denominator_decimal",
        "upper_bound_numerator_decimal",
        "upper_bound_denominator_decimal",
    }
)
_HOUSEHOLD_TYPE_SNAPSHOT_KEYS = frozenset({*_HOUSEHOLD_TYPE_KEYS})
_JURISDICTION_SNAPSHOT_KEYS = frozenset(
    {*_JURISDICTION_KEYS, "target_population_count_decimal"}
)
_PARTITION_RECORD_SNAPSHOT_KEYS = frozenset(
    {
        *_PARTITION_RECORD_KEYS,
        "record_weight_numerator_decimal",
        "record_weight_denominator_decimal",
    }
)
_BUNDLE_SNAPSHOT_KEYS = frozenset(
    {
        *_TOP_LEVEL_KEYS,
        "bundle_path",
        "bundle_sha256",
        "bundle_byte_length",
        "domain_sha256",
        "partition_sha256",
        "declaration_complete",
        "campaign_ready",
        "campaign_blockers",
    }
)

_CALIBRATION_CELL_SNAPSHOT_KEYS = frozenset(
    {
        "cell_ordinal",
        "jurisdiction_code",
        "age_band_id",
        "income_band_id",
        "household_type_id",
        "gaming_state",
        "payer_history_state",
        "evidence_cell_id",
        "target_mass_numerator",
        "target_mass_denominator",
        "target_mass_numerator_decimal",
        "target_mass_denominator_decimal",
        "target_population_numerator",
        "target_population_denominator",
        "target_population_numerator_decimal",
        "target_population_denominator_decimal",
    }
)
_CALIBRATION_TARGET_KEYS = frozenset(
    {
        "schema_version",
        "design_id",
        "design_bundle_sha256",
        "population_evidence_bundle_sha256",
        "calibration_evidence_sha256s",
        "domain_sha256",
        "total_population_count",
        "total_population_count_decimal",
        "cell_count",
        "cells",
    }
)
_CALIBRATION_TARGET_SNAPSHOT_KEYS = frozenset(
    {*_CALIBRATION_TARGET_KEYS, "calibration_target_sha256"}
)
_APPORTIONMENT_CELL_KEYS = frozenset(
    {
        *_CALIBRATION_CELL_SNAPSHOT_KEYS,
        "sample_count",
        "sample_count_decimal",
        "analysis_weight_numerator",
        "analysis_weight_denominator",
        "analysis_weight_numerator_decimal",
        "analysis_weight_denominator_decimal",
        "expansion_weight_numerator",
        "expansion_weight_denominator",
        "expansion_weight_numerator_decimal",
        "expansion_weight_denominator_decimal",
    }
)
_APPORTIONMENT_KEYS = frozenset(
    {
        "schema_version",
        "recipe",
        "calibration_target_sha256",
        "design_id",
        "design_bundle_sha256",
        "domain_sha256",
        "player_count",
        "player_count_decimal",
        "first_player_id",
        "first_player_id_decimal",
        "last_player_id_exclusive",
        "last_player_id_exclusive_decimal",
        "total_population_count",
        "total_population_count_decimal",
        "cell_count",
        "cells",
    }
)
_APPORTIONMENT_SNAPSHOT_KEYS = frozenset(
    {*_APPORTIONMENT_KEYS, "apportionment_sha256"}
)

_GAMING_DOMAIN = (
    PopulationGamingState.GAMER,
    PopulationGamingState.NON_GAMER,
)
_PAYER_DOMAIN = (
    PopulationPayerHistoryState.EVER_PAYER,
    PopulationPayerHistoryState.NEVER_PAYER,
)


class PopulationDesignValidationError(ValueError):
    """Raised when a population-design declaration is malformed."""


class PopulationDesignVerificationError(PopulationDesignValidationError):
    """Raised when a design differs from its bound evidence or exact bytes."""


@dataclass(frozen=True, slots=True)
class PopulationAgeBand:
    ordinal: int
    age_band_id: str
    age_min_inclusive: int
    age_max_exclusive: int

    def __post_init__(self) -> None:
        _strict_int(self.ordinal, name="age band ordinal", minimum=0)
        _identifier(self.age_band_id, name="age_band_id")
        _strict_int(
            self.age_min_inclusive,
            name=f"age band {self.age_band_id} age_min_inclusive",
            minimum=0,
            maximum=199,
        )
        _strict_int(
            self.age_max_exclusive,
            name=f"age band {self.age_band_id} age_max_exclusive",
            minimum=1,
            maximum=200,
        )
        if self.age_max_exclusive <= self.age_min_inclusive:
            raise PopulationDesignValidationError(
                f"age band {self.age_band_id} is empty or reversed"
            )

    @property
    def semantic_key(self) -> tuple[int, int]:
        return self.age_min_inclusive, self.age_max_exclusive

    def snapshot(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "age_band_id": self.age_band_id,
            "age_min_inclusive": self.age_min_inclusive,
            "age_max_exclusive": self.age_max_exclusive,
        }


@dataclass(frozen=True, slots=True)
class PopulationIncomeBand:
    ordinal: int
    jurisdiction_code: str
    income_band_id: str
    definition: str
    currency: str
    period: str
    lower_unbounded: bool
    lower_bound_numerator: int
    lower_bound_denominator: int
    upper_unbounded: bool
    upper_bound_numerator: int
    upper_bound_denominator: int

    def __post_init__(self) -> None:
        _strict_int(self.ordinal, name="income band ordinal", minimum=0)
        _jurisdiction_code(self.jurisdiction_code)
        _identifier(self.income_band_id, name="income_band_id")
        _text(self.definition, name=f"income band {self.income_band_id} definition")
        if type(self.currency) is not str or _CURRENCY_PATTERN.fullmatch(
            self.currency
        ) is None:
            raise PopulationDesignValidationError(
                f"income band {self.income_band_id} currency must be uppercase ISO-like text"
            )
        _text(self.period, name=f"income band {self.income_band_id} period")
        if type(self.lower_unbounded) is not bool or type(
            self.upper_unbounded
        ) is not bool:
            raise PopulationDesignValidationError(
                f"income band {self.income_band_id} bound flags must be booleans"
            )
        _reduced_fraction_parts(
            self.lower_bound_numerator,
            self.lower_bound_denominator,
            name=f"income band {self.income_band_id} lower bound",
        )
        _reduced_fraction_parts(
            self.upper_bound_numerator,
            self.upper_bound_denominator,
            name=f"income band {self.income_band_id} upper bound",
        )
        if self.lower_unbounded and (
            self.lower_bound_numerator,
            self.lower_bound_denominator,
        ) != (0, 1):
            raise PopulationDesignValidationError(
                f"income band {self.income_band_id} unbounded lower sentinel must be 0/1"
            )
        if self.upper_unbounded and (
            self.upper_bound_numerator,
            self.upper_bound_denominator,
        ) != (0, 1):
            raise PopulationDesignValidationError(
                f"income band {self.income_band_id} unbounded upper sentinel must be 0/1"
            )
        if not self.lower_unbounded and not self.upper_unbounded:
            if self.lower_bound >= self.upper_bound:
                raise PopulationDesignValidationError(
                    f"income band {self.income_band_id} is empty or reversed"
                )

    @property
    def lower_bound(self) -> Fraction:
        return Fraction(self.lower_bound_numerator, self.lower_bound_denominator)

    @property
    def upper_bound(self) -> Fraction:
        return Fraction(self.upper_bound_numerator, self.upper_bound_denominator)

    def snapshot(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "jurisdiction_code": self.jurisdiction_code,
            "income_band_id": self.income_band_id,
            "definition": self.definition,
            "currency": self.currency,
            "period": self.period,
            "lower_unbounded": self.lower_unbounded,
            "lower_bound_numerator": self.lower_bound_numerator,
            "lower_bound_denominator": self.lower_bound_denominator,
            "lower_bound_numerator_decimal": str(self.lower_bound_numerator),
            "lower_bound_denominator_decimal": str(self.lower_bound_denominator),
            "upper_unbounded": self.upper_unbounded,
            "upper_bound_numerator": self.upper_bound_numerator,
            "upper_bound_denominator": self.upper_bound_denominator,
            "upper_bound_numerator_decimal": str(self.upper_bound_numerator),
            "upper_bound_denominator_decimal": str(self.upper_bound_denominator),
        }


@dataclass(frozen=True, slots=True)
class PopulationHouseholdType:
    ordinal: int
    household_type_id: str
    definition: str

    def __post_init__(self) -> None:
        _strict_int(self.ordinal, name="household type ordinal", minimum=0)
        _identifier(self.household_type_id, name="household_type_id")
        _text(
            self.definition,
            name=f"household type {self.household_type_id} definition",
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "household_type_id": self.household_type_id,
            "definition": self.definition,
        }


@dataclass(frozen=True, slots=True)
class PopulationDesignJurisdiction:
    jurisdiction_code: str
    target_population_count: int
    calibration_binding_id: str
    calibration_target_population_id: str
    calibration_evidence_sha256: str
    validation_binding_id: str
    validation_target_population_id: str
    validation_evidence_sha256: str

    def __post_init__(self) -> None:
        _jurisdiction_code(self.jurisdiction_code)
        _strict_int(
            self.target_population_count,
            name=f"{self.jurisdiction_code} target_population_count",
            minimum=1,
            maximum=MAX_TARGET_POPULATION_COUNT,
        )
        for name in (
            "calibration_binding_id",
            "calibration_target_population_id",
            "validation_binding_id",
            "validation_target_population_id",
        ):
            _identifier(getattr(self, name), name=name)
        for name in (
            "calibration_evidence_sha256",
            "validation_evidence_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        if self.calibration_binding_id == self.validation_binding_id:
            raise PopulationDesignValidationError(
                f"{self.jurisdiction_code} calibration and validation binding ids must differ"
            )
        if (
            self.calibration_target_population_id
            == self.validation_target_population_id
        ):
            raise PopulationDesignValidationError(
                f"{self.jurisdiction_code} calibration and validation target ids must differ"
            )
        if self.calibration_evidence_sha256 == self.validation_evidence_sha256:
            raise PopulationDesignValidationError(
                f"{self.jurisdiction_code} calibration and validation evidence digests must differ"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "jurisdiction_code": self.jurisdiction_code,
            "target_population_count": self.target_population_count,
            "target_population_count_decimal": str(self.target_population_count),
            "calibration_binding_id": self.calibration_binding_id,
            "calibration_target_population_id": self.calibration_target_population_id,
            "calibration_evidence_sha256": self.calibration_evidence_sha256,
            "validation_binding_id": self.validation_binding_id,
            "validation_target_population_id": self.validation_target_population_id,
            "validation_evidence_sha256": self.validation_evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class PopulationPartitionRecord:
    record_identity_sha256: str
    cluster_identity_sha256: str
    estimand_role: PopulationEstimandRole
    binding_id: str
    cell_id: str
    record_weight_numerator: int
    record_weight_denominator: int

    def __post_init__(self) -> None:
        _sha256(self.record_identity_sha256, name="record_identity_sha256")
        _sha256(self.cluster_identity_sha256, name="cluster_identity_sha256")
        if type(self.estimand_role) is not PopulationEstimandRole:
            raise PopulationDesignValidationError(
                "partition record estimand_role must be CALIBRATION or VALIDATION"
            )
        _identifier(self.binding_id, name="partition record binding_id")
        _identifier(self.cell_id, name="partition record cell_id")
        _reduced_fraction_parts(
            self.record_weight_numerator,
            self.record_weight_denominator,
            name=f"partition record {self.record_identity_sha256} weight",
            positive=True,
        )

    @property
    def record_weight(self) -> Fraction:
        return Fraction(self.record_weight_numerator, self.record_weight_denominator)

    def snapshot(self) -> dict[str, object]:
        return {
            "record_identity_sha256": self.record_identity_sha256,
            "cluster_identity_sha256": self.cluster_identity_sha256,
            "estimand_role": self.estimand_role.value,
            "binding_id": self.binding_id,
            "cell_id": self.cell_id,
            "record_weight_numerator": self.record_weight_numerator,
            "record_weight_denominator": self.record_weight_denominator,
            "record_weight_numerator_decimal": str(self.record_weight_numerator),
            "record_weight_denominator_decimal": str(self.record_weight_denominator),
        }


@dataclass(frozen=True, slots=True)
class PopulationPartitionSpec:
    identity_namespace: str
    record_id_recipe: str
    cluster_id_recipe: str
    role_assignment_recipe: str
    assignment_seed_sha256: str
    calibration_threshold_numerator: int
    calibration_threshold_denominator: int
    records: tuple[PopulationPartitionRecord, ...]

    def __post_init__(self) -> None:
        _identifier(self.identity_namespace, name="partition identity_namespace")
        if self.record_id_recipe != CANONICAL_SOURCE_RECORD_ID_V1:
            raise PopulationDesignValidationError(
                "unsupported partition record_id_recipe"
            )
        if self.cluster_id_recipe != CANONICAL_SOURCE_CLUSTER_ID_V1:
            raise PopulationDesignValidationError(
                "unsupported partition cluster_id_recipe"
            )
        if self.role_assignment_recipe != SHA256_CLUSTER_THRESHOLD_V1:
            raise PopulationDesignValidationError(
                "unsupported partition role_assignment_recipe"
            )
        _sha256(self.assignment_seed_sha256, name="assignment_seed_sha256")
        _reduced_fraction_parts(
            self.calibration_threshold_numerator,
            self.calibration_threshold_denominator,
            name="calibration threshold",
            positive=True,
        )
        if not Fraction(
            self.calibration_threshold_numerator,
            self.calibration_threshold_denominator,
        ) < 1:
            raise PopulationDesignValidationError(
                "calibration threshold must be strictly between zero and one"
            )
        if type(self.records) is not tuple or any(
            type(record) is not PopulationPartitionRecord for record in self.records
        ):
            raise PopulationDesignValidationError(
                "partition records must be an immutable typed tuple"
            )
        record_ids = tuple(record.record_identity_sha256 for record in self.records)
        if record_ids != tuple(sorted(record_ids)):
            raise PopulationDesignValidationError(
                "partition records must use ascending record identity order"
            )
        if len(set(record_ids)) != len(record_ids):
            raise PopulationDesignValidationError(
                "partition record identities repeat"
            )
        cluster_bindings: dict[str, tuple[PopulationEstimandRole, str]] = {}
        for record in self.records:
            expected_role = assigned_population_partition_role(
                identity_namespace=self.identity_namespace,
                assignment_seed_sha256=self.assignment_seed_sha256,
                cluster_identity_sha256=record.cluster_identity_sha256,
                calibration_threshold_numerator=(
                    self.calibration_threshold_numerator
                ),
                calibration_threshold_denominator=(
                    self.calibration_threshold_denominator
                ),
            )
            if record.estimand_role is not expected_role:
                raise PopulationDesignValidationError(
                    "partition record role differs from deterministic cluster assignment: "
                    f"record={record.record_identity_sha256}"
                )
            identity = (record.estimand_role, record.binding_id)
            previous = cluster_bindings.setdefault(
                record.cluster_identity_sha256,
                identity,
            )
            if previous != identity:
                raise PopulationDesignValidationError(
                    "a partition cluster crosses estimand roles or target bindings: "
                    f"cluster={record.cluster_identity_sha256}"
                )

    @property
    def calibration_threshold(self) -> Fraction:
        return Fraction(
            self.calibration_threshold_numerator,
            self.calibration_threshold_denominator,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "identity_namespace": self.identity_namespace,
            "record_id_recipe": self.record_id_recipe,
            "cluster_id_recipe": self.cluster_id_recipe,
            "role_assignment_recipe": self.role_assignment_recipe,
            "assignment_seed_sha256": self.assignment_seed_sha256,
            "calibration_threshold_numerator": (
                self.calibration_threshold_numerator
            ),
            "calibration_threshold_denominator": (
                self.calibration_threshold_denominator
            ),
            "records": [record.snapshot() for record in self.records],
        }


@dataclass(frozen=True, slots=True)
class PopulationDesignBundle:
    schema_version: int
    design_id: str
    provenance_status: ProvenanceStatus
    notes: str
    population_evidence_bundle_sha256: str
    population_evidence_result_sha256s: tuple[str, ...]
    hamilton_recipe: str
    income_missing_policy: str
    household_missing_policy: str
    gaming_states: tuple[PopulationGamingState, ...]
    payer_history_states: tuple[PopulationPayerHistoryState, ...]
    age_bands: tuple[PopulationAgeBand, ...]
    income_bands: tuple[PopulationIncomeBand, ...]
    household_types: tuple[PopulationHouseholdType, ...]
    jurisdictions: tuple[PopulationDesignJurisdiction, ...]
    partition: PopulationPartitionSpec
    bundle_path: Path
    bundle_sha256: str
    bundle_byte_length: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != POPULATION_DESIGN_SCHEMA_VERSION
        ):
            raise PopulationDesignValidationError(
                f"unsupported population-design schema version: {self.schema_version!r}"
            )
        _identifier(self.design_id, name="design_id")
        if type(self.provenance_status) is not ProvenanceStatus:
            raise PopulationDesignValidationError(
                "population-design provenance_status is invalid"
            )
        _text(self.notes, name="notes")
        _sha256(
            self.population_evidence_bundle_sha256,
            name="population_evidence_bundle_sha256",
        )
        if type(self.population_evidence_result_sha256s) is not tuple:
            raise PopulationDesignValidationError(
                "population_evidence_result_sha256s must be an immutable tuple"
            )
        for digest in self.population_evidence_result_sha256s:
            _sha256(digest, name="population evidence result digest")
        if self.hamilton_recipe != EXACT_RATIONAL_HAMILTON_V1:
            raise PopulationDesignValidationError(
                "unsupported population-design Hamilton recipe"
            )
        if self.income_missing_policy != "REJECT":
            raise PopulationDesignValidationError(
                "income_missing_policy must be REJECT in schema v1"
            )
        if self.household_missing_policy != "REJECT":
            raise PopulationDesignValidationError(
                "household_missing_policy must be REJECT in schema v1"
            )
        if self.gaming_states != _GAMING_DOMAIN:
            raise PopulationDesignValidationError(
                "gaming_states must declare GAMER then NON_GAMER exactly"
            )
        if self.payer_history_states != _PAYER_DOMAIN:
            raise PopulationDesignValidationError(
                "payer_history_states must declare EVER_PAYER then NEVER_PAYER exactly"
            )
        _typed_tuple(self.age_bands, PopulationAgeBand, name="age_bands")
        _typed_tuple(self.income_bands, PopulationIncomeBand, name="income_bands")
        _typed_tuple(
            self.household_types,
            PopulationHouseholdType,
            name="household_types",
        )
        _typed_tuple(
            self.jurisdictions,
            PopulationDesignJurisdiction,
            name="jurisdictions",
        )
        if type(self.partition) is not PopulationPartitionSpec:
            raise PopulationDesignValidationError(
                "partition must be a PopulationPartitionSpec"
            )
        if not isinstance(self.bundle_path, Path) or not self.bundle_path.is_absolute():
            raise PopulationDesignValidationError(
                "population design bundle_path must be an absolute Path"
            )
        lexical_path = Path(os.path.normpath(os.fspath(self.bundle_path)))
        if ".." in self.bundle_path.parts or lexical_path != self.bundle_path:
            raise PopulationDesignValidationError(
                "population design bundle_path must be lexically canonical"
            )
        _sha256(self.bundle_sha256, name="population design bundle_sha256")
        _strict_int(
            self.bundle_byte_length,
            name="population design bundle_byte_length",
            minimum=1,
            maximum=MAX_POPULATION_DESIGN_BYTES,
        )
        self._validate_domains()
        self._validate_jurisdictions_and_partition()

    def _validate_domains(self) -> None:
        for name, values in (
            ("age bands", self.age_bands),
            ("household types", self.household_types),
        ):
            ordinals = tuple(value.ordinal for value in values)
            if ordinals != tuple(range(len(values))):
                raise PopulationDesignValidationError(
                    f"{name} must use contiguous declared ordinals from zero"
                )
        _unique_domain_ids(
            tuple(band.age_band_id for band in self.age_bands),
            name="age band ids",
        )
        _unique_domain_ids(
            tuple(item.household_type_id for item in self.household_types),
            name="household type ids",
        )
        for left, right in zip(self.age_bands, self.age_bands[1:]):
            if left.age_max_exclusive != right.age_min_inclusive:
                raise PopulationDesignValidationError(
                    "age bands must be gap-free and non-overlapping in ordinal order"
                )
        income_codes = tuple(band.jurisdiction_code for band in self.income_bands)
        if income_codes != tuple(sorted(income_codes)):
            raise PopulationDesignValidationError(
                "income bands must be grouped in ascending jurisdiction order"
            )
        grouped: dict[str, tuple[PopulationIncomeBand, ...]] = {
            code: tuple(
                band for band in self.income_bands if band.jurisdiction_code == code
            )
            for code in dict.fromkeys(income_codes)
        }
        harmonized: tuple[tuple[int, str, str], ...] | None = None
        for code, bands in grouped.items():
            if tuple(band.ordinal for band in bands) != tuple(range(len(bands))):
                raise PopulationDesignValidationError(
                    f"{code} income bands must use contiguous ordinals from zero"
                )
            _unique_domain_ids(
                tuple(band.income_band_id for band in bands),
                name=f"{code} income band ids",
            )
            if not bands[0].lower_unbounded:
                raise PopulationDesignValidationError(
                    f"{code} first income band must have an unbounded lower edge"
                )
            if not bands[-1].upper_unbounded:
                raise PopulationDesignValidationError(
                    f"{code} last income band must have an unbounded upper edge"
                )
            if len({band.currency for band in bands}) != 1 or len(
                {band.period for band in bands}
            ) != 1:
                raise PopulationDesignValidationError(
                    f"{code} income bands must use one exact currency and period"
                )
            for index, band in enumerate(bands):
                if index > 0 and band.lower_unbounded:
                    raise PopulationDesignValidationError(
                        f"only {code}'s first income band may be lower-unbounded"
                    )
                if index < len(bands) - 1 and band.upper_unbounded:
                    raise PopulationDesignValidationError(
                        f"only {code}'s last income band may be upper-unbounded"
                    )
            for left, right in zip(bands, bands[1:]):
                if left.upper_bound != right.lower_bound:
                    raise PopulationDesignValidationError(
                        f"{code} income bands must be gap-free and non-overlapping"
                    )
            declarations = tuple(
                (band.ordinal, band.income_band_id, band.definition) for band in bands
            )
            if harmonized is None:
                harmonized = declarations
            elif declarations != harmonized:
                raise PopulationDesignValidationError(
                    "jurisdictions must use the same harmonized income-band "
                    "ordinals, ids, and definitions"
                )

    def _validate_jurisdictions_and_partition(self) -> None:
        codes = tuple(item.jurisdiction_code for item in self.jurisdictions)
        if codes != tuple(sorted(codes)):
            raise PopulationDesignValidationError(
                "jurisdictions must use ascending jurisdiction_code order"
            )
        if len(set(codes)) != len(codes):
            raise PopulationDesignValidationError("jurisdiction codes repeat")
        income_codes = {band.jurisdiction_code for band in self.income_bands}
        if income_codes != set(codes):
            raise PopulationDesignValidationError(
                "income-band jurisdictions must exactly cover design jurisdictions"
            )
        binding_roles: dict[str, PopulationEstimandRole] = {}
        ordered_result_digests: list[str] = []
        for jurisdiction in self.jurisdictions:
            pairs = (
                (
                    jurisdiction.calibration_binding_id,
                    PopulationEstimandRole.CALIBRATION,
                    jurisdiction.calibration_evidence_sha256,
                ),
                (
                    jurisdiction.validation_binding_id,
                    PopulationEstimandRole.VALIDATION,
                    jurisdiction.validation_evidence_sha256,
                ),
            )
            for binding_id, role, digest in pairs:
                if binding_id in binding_roles:
                    raise PopulationDesignValidationError(
                        f"population-design binding id repeats: {binding_id}"
                    )
                binding_roles[binding_id] = role
                ordered_result_digests.append(digest)
        if tuple(ordered_result_digests) != self.population_evidence_result_sha256s:
            raise PopulationDesignValidationError(
                "population evidence result digests must exactly follow jurisdiction "
                "order with CALIBRATION then VALIDATION"
            )
        populated_components = (
            bool(self.age_bands),
            bool(self.income_bands),
            bool(self.household_types),
            bool(self.jurisdictions),
            bool(self.partition.records),
            bool(self.population_evidence_result_sha256s),
        )
        if any(populated_components) and not all(populated_components):
            raise PopulationDesignValidationError(
                "population-design domains, targets, evidence results, and partition "
                "records must be populated together"
            )
        seen_bindings: set[str] = set()
        for record in self.partition.records:
            expected_role = binding_roles.get(record.binding_id)
            if expected_role is None:
                raise PopulationDesignValidationError(
                    f"partition record references unknown binding {record.binding_id}"
                )
            if record.estimand_role is not expected_role:
                raise PopulationDesignValidationError(
                    f"partition record role does not match binding {record.binding_id}"
                )
            seen_bindings.add(record.binding_id)
        if seen_bindings != set(binding_roles):
            raise PopulationDesignValidationError(
                "partition records must cover every calibration and validation binding"
            )

    @property
    def domain_sha256(self) -> str:
        return sha256(
            _canonical_json(self.domain_snapshot()).encode("utf-8")
        ).hexdigest()

    @property
    def partition_sha256(self) -> str:
        return sha256(
            _canonical_json(self.partition.snapshot()).encode("utf-8")
        ).hexdigest()

    @property
    def declaration_complete(self) -> bool:
        """Whether all schema-v1 declaration sections are populated.

        This is not an authenticity, held-out-validation, or campaign-readiness
        claim.  Record and cluster hashes are manifest declarations only.
        """

        return bool(self.jurisdictions) and bool(self.partition.records)

    @property
    def campaign_ready(self) -> bool:
        return False

    @property
    def campaign_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.provenance_status is not ProvenanceStatus.CALIBRATED:
            blockers.append(
                "population_design_status=" + self.provenance_status.value
            )
        if not self.jurisdictions:
            blockers.append("population_design_empty")
        blockers.extend(
            (
                "population_design_signature_missing",
                "population_design_partition_source_unit_keys_unverified",
                "population_design_heldout_readiness_unverified",
                "population_design_runtime_projection_unverified",
                "population_design_output_estimand_unverified",
                "population_design_balance_unverified",
            )
        )
        return tuple(blockers)

    def validate_for_campaign(self) -> None:
        raise PopulationDesignVerificationError(
            "population-design schema v1 is a static-only contract: "
            + ", ".join(self.campaign_blockers)
        )

    def domain_snapshot(self) -> dict[str, object]:
        return {
            "income_missing_policy": self.income_missing_policy,
            "household_missing_policy": self.household_missing_policy,
            "gaming_states": [state.value for state in self.gaming_states],
            "payer_history_states": [state.value for state in self.payer_history_states],
            "age_bands": [band.snapshot() for band in self.age_bands],
            "income_bands": [band.snapshot() for band in self.income_bands],
            "household_types": [item.snapshot() for item in self.household_types],
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "design_id": self.design_id,
            "provenance_status": self.provenance_status.value,
            "notes": self.notes,
            "population_evidence_bundle_sha256": (
                self.population_evidence_bundle_sha256
            ),
            "population_evidence_result_sha256s": list(
                self.population_evidence_result_sha256s
            ),
            "hamilton_recipe": self.hamilton_recipe,
            "domains": self.domain_snapshot(),
            "jurisdictions": [item.snapshot() for item in self.jurisdictions],
            "partition": self.partition.snapshot(),
            "bundle_path": str(self.bundle_path),
            "bundle_sha256": self.bundle_sha256,
            "bundle_byte_length": self.bundle_byte_length,
            "domain_sha256": self.domain_sha256,
            "partition_sha256": self.partition_sha256,
            "declaration_complete": self.declaration_complete,
            "campaign_ready": False,
            "campaign_blockers": list(self.campaign_blockers),
        }


@dataclass(frozen=True, slots=True)
class PopulationDesignVerification:
    """A design whose exact bytes and exact evidence have been re-attested."""

    bundle: PopulationDesignBundle
    evidence_bundle: PopulationEvidenceBundle
    evidence_results: tuple[PopulationEvidenceResult, ...]
    verification_sha256: str

    def __post_init__(self) -> None:
        if type(self.bundle) is not PopulationDesignBundle:
            raise PopulationDesignValidationError(
                "verification bundle must be a PopulationDesignBundle"
            )
        if type(self.evidence_bundle) is not PopulationEvidenceBundle:
            raise PopulationDesignValidationError(
                "verification evidence_bundle must be a PopulationEvidenceBundle"
            )
        _typed_tuple(
            self.evidence_results,
            PopulationEvidenceResult,
            name="verification evidence_results",
        )
        reloaded_bundle = load_population_design_bundle(self.bundle.bundle_path)
        if reloaded_bundle != self.bundle:
            raise PopulationDesignVerificationError(
                "population-design verification does not match its exact bundle bytes"
            )
        try:
            observed_results = verify_population_evidence_bundle(
                self.evidence_bundle,
                expected_source_registry_sha256=(
                    self.evidence_bundle.source_registry_sha256
                ),
            )
        except PopulationEvidenceValidationError as exc:
            raise PopulationDesignVerificationError(
                "population-design verification evidence could not be re-attested"
            ) from exc
        if observed_results != self.evidence_results:
            raise PopulationDesignVerificationError(
                "population-design verification results differ from exact evidence bytes"
            )
        _verify_design_evidence_bindings(
            self.bundle,
            evidence_bundle=self.evidence_bundle,
            evidence_results=self.evidence_results,
        )
        _sha256(self.verification_sha256, name="verification_sha256")
        expected = sha256(
            _canonical_json(self.attestation_payload()).encode("utf-8")
        ).hexdigest()
        if self.verification_sha256 != expected:
            raise PopulationDesignValidationError(
                "verification_sha256 does not match its design/evidence identities"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": POPULATION_DESIGN_SCHEMA_VERSION,
            "design_id": self.bundle.design_id,
            "design_bundle_sha256": self.bundle.bundle_sha256,
            "domain_sha256": self.bundle.domain_sha256,
            "partition_sha256": self.bundle.partition_sha256,
            "population_evidence_bundle_sha256": self.evidence_bundle.bundle_sha256,
            "population_evidence_result_sha256s": [
                result.evidence_sha256 for result in self.evidence_results
            ],
            "evidence_reverified": True,
            "authenticity_verified": False,
            "heldout_ready": False,
            "campaign_ready": False,
        }

    @property
    def evidence_reverified(self) -> bool:
        return True

    @property
    def authenticity_verified(self) -> bool:
        return False

    @property
    def heldout_ready(self) -> bool:
        return False

    @property
    def campaign_ready(self) -> bool:
        return False


def assigned_population_partition_role(
    *,
    identity_namespace: str,
    assignment_seed_sha256: str,
    cluster_identity_sha256: str,
    calibration_threshold_numerator: int,
    calibration_threshold_denominator: int,
) -> PopulationEstimandRole:
    """Return the canonical cluster-level role for schema-v1 partitioning."""

    _identifier(identity_namespace, name="identity_namespace")
    _sha256(assignment_seed_sha256, name="assignment_seed_sha256")
    _sha256(cluster_identity_sha256, name="cluster_identity_sha256")
    _reduced_fraction_parts(
        calibration_threshold_numerator,
        calibration_threshold_denominator,
        name="calibration threshold",
        positive=True,
    )
    threshold = Fraction(
        calibration_threshold_numerator,
        calibration_threshold_denominator,
    )
    if threshold >= 1:
        raise PopulationDesignValidationError(
            "calibration threshold must be strictly between zero and one"
        )
    payload = _canonical_json(
        {
            "assignment_seed_sha256": assignment_seed_sha256,
            "cluster_identity_sha256": cluster_identity_sha256,
            "identity_namespace": identity_namespace,
            "recipe": SHA256_CLUSTER_THRESHOLD_V1,
        }
    ).encode("utf-8")
    score = int.from_bytes(sha256(payload).digest(), "big")
    calibration = (
        score * calibration_threshold_denominator
        < calibration_threshold_numerator * (1 << 256)
    )
    return (
        PopulationEstimandRole.CALIBRATION
        if calibration
        else PopulationEstimandRole.VALIDATION
    )


@dataclass(frozen=True, slots=True)
class PopulationCalibrationCell:
    """One calibration-only cell after jurisdiction counts are normalized."""

    cell_ordinal: int
    jurisdiction_code: str
    age_band_id: str
    income_band_id: str
    household_type_id: str
    gaming_state: PopulationGamingState
    payer_history_state: PopulationPayerHistoryState
    evidence_cell_id: str
    target_mass_numerator: int
    target_mass_denominator: int
    target_population_numerator: int
    target_population_denominator: int

    def __post_init__(self) -> None:
        _strict_int(self.cell_ordinal, name="cell_ordinal", minimum=0)
        _jurisdiction_code(self.jurisdiction_code)
        for value, name in (
            (self.age_band_id, "age_band_id"),
            (self.income_band_id, "income_band_id"),
            (self.household_type_id, "household_type_id"),
            (self.evidence_cell_id, "evidence_cell_id"),
        ):
            _identifier(value, name=name)
        if type(self.gaming_state) is not PopulationGamingState:
            raise PopulationDesignValidationError(
                "calibration cell gaming_state is invalid"
            )
        if type(self.payer_history_state) is not PopulationPayerHistoryState:
            raise PopulationDesignValidationError(
                "calibration cell payer_history_state is invalid"
            )
        _reduced_fraction_parts(
            self.target_mass_numerator,
            self.target_mass_denominator,
            name=f"calibration cell {self.cell_ordinal} target mass",
            nonnegative=True,
        )
        _reduced_fraction_parts(
            self.target_population_numerator,
            self.target_population_denominator,
            name=f"calibration cell {self.cell_ordinal} target population",
            nonnegative=True,
        )

    @property
    def target_mass(self) -> Fraction:
        return Fraction(self.target_mass_numerator, self.target_mass_denominator)

    @property
    def target_population(self) -> Fraction:
        return Fraction(
            self.target_population_numerator,
            self.target_population_denominator,
        )

    @property
    def semantic_key(self) -> tuple[object, ...]:
        return (
            self.jurisdiction_code,
            self.age_band_id,
            self.income_band_id,
            self.household_type_id,
            self.gaming_state.value,
            self.payer_history_state.value,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "cell_ordinal": self.cell_ordinal,
            "jurisdiction_code": self.jurisdiction_code,
            "age_band_id": self.age_band_id,
            "income_band_id": self.income_band_id,
            "household_type_id": self.household_type_id,
            "gaming_state": self.gaming_state.value,
            "payer_history_state": self.payer_history_state.value,
            "evidence_cell_id": self.evidence_cell_id,
            "target_mass_numerator": self.target_mass_numerator,
            "target_mass_denominator": self.target_mass_denominator,
            "target_mass_numerator_decimal": str(self.target_mass_numerator),
            "target_mass_denominator_decimal": str(self.target_mass_denominator),
            "target_population_numerator": self.target_population_numerator,
            "target_population_denominator": self.target_population_denominator,
            "target_population_numerator_decimal": str(
                self.target_population_numerator
            ),
            "target_population_denominator_decimal": str(
                self.target_population_denominator
            ),
        }


@dataclass(frozen=True, slots=True)
class PopulationCalibrationTarget:
    """Calibration-only allocator input; validation results are intentionally absent."""

    design_id: str
    design_bundle_sha256: str
    population_evidence_bundle_sha256: str
    calibration_evidence_sha256s: tuple[str, ...]
    domain_sha256: str
    total_population_count: int
    cells: tuple[PopulationCalibrationCell, ...]
    calibration_target_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.design_id, name="calibration target design_id")
        for value, name in (
            (self.design_bundle_sha256, "design_bundle_sha256"),
            (
                self.population_evidence_bundle_sha256,
                "population_evidence_bundle_sha256",
            ),
            (self.domain_sha256, "domain_sha256"),
            (self.calibration_target_sha256, "calibration_target_sha256"),
        ):
            _sha256(value, name=name)
        if type(self.calibration_evidence_sha256s) is not tuple or not (
            self.calibration_evidence_sha256s
        ):
            raise PopulationDesignValidationError(
                "calibration evidence digests must be a non-empty immutable tuple"
            )
        for digest in self.calibration_evidence_sha256s:
            _sha256(digest, name="calibration evidence digest")
        _strict_int(
            self.total_population_count,
            name="total_population_count",
            minimum=1,
            maximum=MAX_TARGET_POPULATION_COUNT * len(
                self.calibration_evidence_sha256s
            ),
        )
        _typed_tuple(self.cells, PopulationCalibrationCell, name="calibration cells")
        if not self.cells:
            raise PopulationDesignValidationError(
                "calibration target must contain declared cells"
            )
        if tuple(cell.cell_ordinal for cell in self.cells) != tuple(
            range(len(self.cells))
        ):
            raise PopulationDesignValidationError(
                "calibration cells must use contiguous canonical ordinals"
            )
        semantic_keys = tuple(cell.semantic_key for cell in self.cells)
        if len(set(semantic_keys)) != len(semantic_keys):
            raise PopulationDesignValidationError(
                "calibration target repeats a semantic cell"
            )
        if sum((cell.target_mass for cell in self.cells), Fraction()) != 1:
            raise PopulationDesignValidationError(
                "calibration target masses must sum to exactly one"
            )
        if sum(
            (cell.target_population for cell in self.cells),
            Fraction(),
        ) != self.total_population_count:
            raise PopulationDesignValidationError(
                "calibration target population amounts do not match total count"
            )
        for cell in self.cells:
            if cell.target_population != (
                cell.target_mass * self.total_population_count
            ):
                raise PopulationDesignValidationError(
                    "calibration cell target population differs from its target mass"
                )
        expected = sha256(
            _canonical_json(self.attestation_payload()).encode("utf-8")
        ).hexdigest()
        if self.calibration_target_sha256 != expected:
            raise PopulationDesignValidationError(
                "calibration_target_sha256 does not match its exact payload"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": POPULATION_DESIGN_SCHEMA_VERSION,
            "design_id": self.design_id,
            "design_bundle_sha256": self.design_bundle_sha256,
            "population_evidence_bundle_sha256": (
                self.population_evidence_bundle_sha256
            ),
            "calibration_evidence_sha256s": list(
                self.calibration_evidence_sha256s
            ),
            "domain_sha256": self.domain_sha256,
            "total_population_count": self.total_population_count,
            "total_population_count_decimal": str(self.total_population_count),
            "cell_count": len(self.cells),
            "cells": [cell.snapshot() for cell in self.cells],
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "calibration_target_sha256": self.calibration_target_sha256,
        }


@dataclass(frozen=True, slots=True)
class PopulationApportionmentCell:
    """One immutable Hamilton count and its exact per-player weights."""

    calibration_cell: PopulationCalibrationCell
    sample_count: int
    analysis_weight_numerator: int
    analysis_weight_denominator: int
    expansion_weight_numerator: int
    expansion_weight_denominator: int

    def __post_init__(self) -> None:
        if type(self.calibration_cell) is not PopulationCalibrationCell:
            raise PopulationDesignValidationError(
                "apportionment calibration_cell is invalid"
            )
        _strict_int(self.sample_count, name="sample_count", minimum=0)
        _reduced_fraction_parts(
            self.analysis_weight_numerator,
            self.analysis_weight_denominator,
            name=f"cell {self.calibration_cell.cell_ordinal} analysis weight",
            nonnegative=True,
        )
        _reduced_fraction_parts(
            self.expansion_weight_numerator,
            self.expansion_weight_denominator,
            name=f"cell {self.calibration_cell.cell_ordinal} expansion weight",
            nonnegative=True,
        )
        mass = self.calibration_cell.target_mass
        population = self.calibration_cell.target_population
        if self.sample_count == 0:
            if mass != 0 or self.analysis_weight != 0 or self.expansion_weight != 0:
                raise PopulationDesignValidationError(
                    "only a zero-mass cell may have zero sample count and weights"
                )
        else:
            if self.analysis_weight != mass / self.sample_count:
                raise PopulationDesignValidationError(
                    "analysis weight does not equal target mass / sample count"
                )
            if self.expansion_weight != population / self.sample_count:
                raise PopulationDesignValidationError(
                    "expansion weight does not equal target population / sample count"
                )

    @property
    def analysis_weight(self) -> Fraction:
        return Fraction(
            self.analysis_weight_numerator,
            self.analysis_weight_denominator,
        )

    @property
    def expansion_weight(self) -> Fraction:
        return Fraction(
            self.expansion_weight_numerator,
            self.expansion_weight_denominator,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            **self.calibration_cell.snapshot(),
            "sample_count": self.sample_count,
            "sample_count_decimal": str(self.sample_count),
            "analysis_weight_numerator": self.analysis_weight_numerator,
            "analysis_weight_denominator": self.analysis_weight_denominator,
            "analysis_weight_numerator_decimal": str(
                self.analysis_weight_numerator
            ),
            "analysis_weight_denominator_decimal": str(
                self.analysis_weight_denominator
            ),
            "expansion_weight_numerator": self.expansion_weight_numerator,
            "expansion_weight_denominator": self.expansion_weight_denominator,
            "expansion_weight_numerator_decimal": str(
                self.expansion_weight_numerator
            ),
            "expansion_weight_denominator_decimal": str(
                self.expansion_weight_denominator
            ),
        }


def _exact_hamilton_counts(
    cells: tuple[PopulationCalibrationCell, ...],
    player_count: int,
) -> tuple[int, ...]:
    """Return the unique schema-v1 largest-remainder allocation."""

    quotas = tuple(cell.target_mass * player_count for cell in cells)
    counts = [quota.numerator // quota.denominator for quota in quotas]
    remaining = player_count - sum(counts)
    remainder_order = sorted(
        range(len(cells)),
        key=lambda index: (
            -(quotas[index] - counts[index]),
            cells[index].cell_ordinal,
        ),
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1
    if sum(counts) != player_count:
        raise PopulationDesignValidationError(
            "Hamilton allocation does not preserve player_count"
        )
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class PopulationApportionmentPlan:
    calibration_target: PopulationCalibrationTarget
    recipe: str
    calibration_target_sha256: str
    design_id: str
    design_bundle_sha256: str
    domain_sha256: str
    player_count: int
    first_player_id: int
    total_population_count: int
    cells: tuple[PopulationApportionmentCell, ...]
    apportionment_sha256: str

    def __post_init__(self) -> None:
        if type(self.calibration_target) is not PopulationCalibrationTarget:
            raise PopulationDesignValidationError(
                "apportionment calibration_target must be a PopulationCalibrationTarget"
            )
        if self.recipe != EXACT_RATIONAL_HAMILTON_V1:
            raise PopulationDesignValidationError(
                "apportionment uses an unsupported recipe"
            )
        _sha256(
            self.calibration_target_sha256,
            name="calibration_target_sha256",
        )
        _identifier(self.design_id, name="apportionment design_id")
        _sha256(self.design_bundle_sha256, name="design_bundle_sha256")
        _sha256(self.domain_sha256, name="domain_sha256")
        target = self.calibration_target
        if (
            self.calibration_target_sha256 != target.calibration_target_sha256
            or self.design_id != target.design_id
            or self.design_bundle_sha256 != target.design_bundle_sha256
            or self.domain_sha256 != target.domain_sha256
            or self.total_population_count != target.total_population_count
        ):
            raise PopulationDesignValidationError(
                "apportionment lineage differs from its exact calibration target"
            )
        _strict_int(
            self.player_count,
            name="player_count",
            minimum=1,
            maximum=MAX_SAMPLE_PLAYER_COUNT,
        )
        _strict_int(
            self.first_player_id,
            name="first_player_id",
            minimum=0,
            maximum=MAX_SAMPLE_PLAYER_COUNT,
        )
        if self.first_player_id + self.player_count > MAX_SAMPLE_PLAYER_COUNT:
            raise PopulationDesignValidationError(
                "apportionment player id interval exceeds the supported range"
            )
        _strict_int(
            self.total_population_count,
            name="total_population_count",
            minimum=1,
        )
        _typed_tuple(
            self.cells,
            PopulationApportionmentCell,
            name="apportionment cells",
        )
        if not self.cells or tuple(
            cell.calibration_cell.cell_ordinal for cell in self.cells
        ) != tuple(range(len(self.cells))):
            raise PopulationDesignValidationError(
                "apportionment cells must exactly retain canonical calibration ordinals"
            )
        if tuple(cell.calibration_cell for cell in self.cells) != target.cells:
            raise PopulationDesignValidationError(
                "apportionment cells differ from the exact calibration target"
            )
        if sum(cell.sample_count for cell in self.cells) != self.player_count:
            raise PopulationDesignValidationError(
                "apportionment counts do not sum to player_count"
            )
        expected_counts = _exact_hamilton_counts(
            tuple(cell.calibration_cell for cell in self.cells),
            self.player_count,
        )
        observed_counts = tuple(cell.sample_count for cell in self.cells)
        if observed_counts != expected_counts:
            raise PopulationDesignValidationError(
                "apportionment counts differ from exact deterministic Hamilton allocation"
            )
        weighted_mass = sum(
            (
                cell.analysis_weight * cell.sample_count
                for cell in self.cells
            ),
            Fraction(),
        )
        if weighted_mass != 1:
            raise PopulationDesignValidationError(
                "apportionment analysis weights do not reconstruct unit mass"
            )
        weighted_population = sum(
            (
                cell.expansion_weight * cell.sample_count
                for cell in self.cells
            ),
            Fraction(),
        )
        if weighted_population != self.total_population_count:
            raise PopulationDesignValidationError(
                "apportionment expansion weights do not reconstruct population count"
            )
        _sha256(self.apportionment_sha256, name="apportionment_sha256")
        expected = sha256(
            _canonical_json(self.attestation_payload()).encode("utf-8")
        ).hexdigest()
        if self.apportionment_sha256 != expected:
            raise PopulationDesignValidationError(
                "apportionment_sha256 does not match its exact payload"
            )

    @property
    def last_player_id_exclusive(self) -> int:
        return self.first_player_id + self.player_count

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": POPULATION_DESIGN_SCHEMA_VERSION,
            "recipe": self.recipe,
            "calibration_target_sha256": self.calibration_target_sha256,
            "design_id": self.design_id,
            "design_bundle_sha256": self.design_bundle_sha256,
            "domain_sha256": self.domain_sha256,
            "player_count": self.player_count,
            "player_count_decimal": str(self.player_count),
            "first_player_id": self.first_player_id,
            "first_player_id_decimal": str(self.first_player_id),
            "last_player_id_exclusive": self.last_player_id_exclusive,
            "last_player_id_exclusive_decimal": str(
                self.last_player_id_exclusive
            ),
            "total_population_count": self.total_population_count,
            "total_population_count_decimal": str(self.total_population_count),
            "cell_count": len(self.cells),
            "cells": [cell.snapshot() for cell in self.cells],
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "apportionment_sha256": self.apportionment_sha256,
        }


def load_population_design_bundle(
    path: str | Path = DEFAULT_POPULATION_DESIGN_BUNDLE_PATH,
) -> PopulationDesignBundle:
    """Parse one strict schema-v1 design without trusting evidence references."""

    bundle_path = Path(path)
    observed = _secure_read_regular_file(
        bundle_path,
        maximum_bytes=MAX_POPULATION_DESIGN_BYTES,
        description="population-design bundle",
    )
    try:
        text = observed.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PopulationDesignValidationError(
            "population-design bundle must be UTF-8"
        ) from exc
    if text.startswith("\ufeff"):
        raise PopulationDesignValidationError(
            "population-design bundle must not contain a UTF-8 BOM"
        )
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise PopulationDesignValidationError(
            f"invalid population-design TOML: {exc}"
        ) from exc
    root = _mapping(raw, name="population-design root")
    _exact_keys(root, _TOP_LEVEL_KEYS, name="population-design root")
    schema_version = _required_int(root, "schema_version", minimum=1)
    if schema_version != POPULATION_DESIGN_SCHEMA_VERSION:
        raise PopulationDesignValidationError(
            f"unsupported population-design schema version: {schema_version}"
        )
    domains = _required_table(root, "domains")
    _exact_keys(domains, _DOMAIN_KEYS, name="population-design domains")
    partition_row = _required_table(root, "partition")
    _exact_keys(partition_row, _PARTITION_KEYS, name="population-design partition")
    bundle = PopulationDesignBundle(
        schema_version=schema_version,
        design_id=_required_string(root, "design_id"),
        provenance_status=_parse_provenance_status(
            _required_string(root, "provenance_status")
        ),
        notes=_required_string(root, "notes"),
        population_evidence_bundle_sha256=_required_string(
            root,
            "population_evidence_bundle_sha256",
        ),
        population_evidence_result_sha256s=tuple(
            _required_string_list(root, "population_evidence_result_sha256s")
        ),
        hamilton_recipe=_required_string(root, "hamilton_recipe"),
        income_missing_policy=_required_string(domains, "income_missing_policy"),
        household_missing_policy=_required_string(
            domains,
            "household_missing_policy",
        ),
        gaming_states=_parse_gaming_states(domains.get("gaming_states")),
        payer_history_states=_parse_payer_states(
            domains.get("payer_history_states")
        ),
        age_bands=tuple(
            _parse_age_band(row)
            for row in _required_table_list(domains, "age_bands")
        ),
        income_bands=tuple(
            _parse_income_band(row)
            for row in _required_table_list(domains, "income_bands")
        ),
        household_types=tuple(
            _parse_household_type(row)
            for row in _required_table_list(domains, "household_types")
        ),
        jurisdictions=tuple(
            _parse_jurisdiction(row)
            for row in _required_table_list(root, "jurisdictions")
        ),
        partition=_parse_partition(partition_row),
        bundle_path=bundle_path.resolve(strict=True),
        bundle_sha256=sha256(observed).hexdigest(),
        bundle_byte_length=len(observed),
    )
    return bundle


def verify_population_design_bundle(
    bundle: PopulationDesignBundle,
    *,
    population_evidence_bundle: PopulationEvidenceBundle,
    population_evidence_results: tuple[PopulationEvidenceResult, ...],
) -> PopulationDesignVerification:
    """Reopen design/evidence bytes and verify every static semantic binding."""

    if type(bundle) is not PopulationDesignBundle:
        raise TypeError("bundle must be a PopulationDesignBundle")
    if type(population_evidence_bundle) is not PopulationEvidenceBundle:
        raise TypeError(
            "population_evidence_bundle must be a PopulationEvidenceBundle"
        )
    if type(population_evidence_results) is not tuple or any(
        type(result) is not PopulationEvidenceResult
        for result in population_evidence_results
    ):
        raise TypeError(
            "population_evidence_results must be a typed immutable tuple"
        )
    reloaded = load_population_design_bundle(bundle.bundle_path)
    if reloaded != bundle:
        raise PopulationDesignVerificationError(
            "population-design metadata no longer match its declared file"
        )
    try:
        observed_results = verify_population_evidence_bundle(
            population_evidence_bundle,
            expected_source_registry_sha256=(
                population_evidence_bundle.source_registry_sha256
            ),
        )
    except PopulationEvidenceValidationError as exc:
        raise PopulationDesignVerificationError(
            "population-design evidence could not be re-attested"
        ) from exc
    if observed_results != population_evidence_results:
        raise PopulationDesignVerificationError(
            "supplied population evidence results differ from re-extracted bytes"
        )
    _verify_design_evidence_bindings(
        bundle,
        evidence_bundle=population_evidence_bundle,
        evidence_results=population_evidence_results,
    )
    payload = {
        "schema_version": POPULATION_DESIGN_SCHEMA_VERSION,
        "design_id": bundle.design_id,
        "design_bundle_sha256": bundle.bundle_sha256,
        "domain_sha256": bundle.domain_sha256,
        "partition_sha256": bundle.partition_sha256,
        "population_evidence_bundle_sha256": (
            population_evidence_bundle.bundle_sha256
        ),
        "population_evidence_result_sha256s": [
            result.evidence_sha256 for result in population_evidence_results
        ],
        "evidence_reverified": True,
        "authenticity_verified": False,
        "heldout_ready": False,
        "campaign_ready": False,
    }
    return PopulationDesignVerification(
        bundle=bundle,
        evidence_bundle=population_evidence_bundle,
        evidence_results=population_evidence_results,
        verification_sha256=sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    )


def load_and_verify_population_design_bundle(
    path: str | Path = DEFAULT_POPULATION_DESIGN_BUNDLE_PATH,
    *,
    population_evidence_bundle: PopulationEvidenceBundle,
    population_evidence_results: tuple[PopulationEvidenceResult, ...],
) -> PopulationDesignVerification:
    """Load a design and immediately re-attest its exact evidence dependencies."""

    bundle = load_population_design_bundle(path)
    return verify_population_design_bundle(
        bundle,
        population_evidence_bundle=population_evidence_bundle,
        population_evidence_results=population_evidence_results,
    )


def validate_population_design_snapshot(
    bundle_snapshot: object,
    population_evidence_bundle_snapshot: object,
    population_evidence_result_snapshots: object,
) -> PopulationDesignVerification:
    """Reopen all bytes and rebuild a serialized design attestation fail-closed."""

    bundle_row = _mapping(bundle_snapshot, name="population-design bundle snapshot")
    _exact_keys(
        bundle_row,
        _BUNDLE_SNAPSHOT_KEYS,
        name="population-design bundle snapshot",
    )
    bundle_path = bundle_row.get("bundle_path")
    if type(bundle_path) is not str or not bundle_path:
        raise PopulationDesignValidationError(
            "population-design snapshot bundle_path must be non-empty text"
        )
    bundle = load_population_design_bundle(bundle_path)
    if _canonical_json(bundle.snapshot()) != _canonical_json(dict(bundle_row)):
        raise PopulationDesignValidationError(
            "population-design bundle snapshot is not canonical or no longer matches bytes"
        )
    try:
        evidence_bundle, evidence_results = validate_population_evidence_snapshot(
            population_evidence_bundle_snapshot,
            population_evidence_result_snapshots,
        )
    except PopulationEvidenceValidationError as exc:
        raise PopulationDesignValidationError(
            "population-design evidence snapshots are invalid"
        ) from exc
    if evidence_bundle is None:
        raise PopulationDesignValidationError(
            "population-design snapshot requires a population-evidence bundle"
        )
    return verify_population_design_bundle(
        bundle,
        population_evidence_bundle=evidence_bundle,
        population_evidence_results=evidence_results,
    )


def build_population_calibration_target(
    verification: PopulationDesignVerification,
) -> PopulationCalibrationTarget:
    """Project verified declarations into a calibration-only typed target."""

    if type(verification) is not PopulationDesignVerification:
        raise TypeError("verification must be a PopulationDesignVerification")
    bundle = verification.bundle
    if not bundle.declaration_complete or (
        bundle.provenance_status is not ProvenanceStatus.CALIBRATED
    ):
        raise PopulationDesignVerificationError(
            "population design is not a complete CALIBRATED declaration"
        )
    # Recompute semantic checks so hand-constructed verification-like objects do
    # not become an alternate path that exposes validation data to allocation.
    _verify_design_evidence_bindings(
        bundle,
        evidence_bundle=verification.evidence_bundle,
        evidence_results=verification.evidence_results,
    )
    results_by_id = {
        result.binding_id: result for result in verification.evidence_results
    }
    total_population_count = sum(
        jurisdiction.target_population_count
        for jurisdiction in bundle.jurisdictions
    )
    age_by_key = {band.semantic_key: band for band in bundle.age_bands}
    household_by_id = {
        item.household_type_id: item for item in bundle.household_types
    }
    cells: list[PopulationCalibrationCell] = []
    for jurisdiction in bundle.jurisdictions:
        result = results_by_id[jurisdiction.calibration_binding_id]
        jurisdiction_income_bands = tuple(
            band
            for band in bundle.income_bands
            if band.jurisdiction_code == jurisdiction.jurisdiction_code
        )
        result_by_semantic_key = {
            cell.semantic_key: cell for cell in result.cells
        }
        for age_band in bundle.age_bands:
            for income_band in jurisdiction_income_bands:
                for household_type in bundle.household_types:
                    for gaming_state in bundle.gaming_states:
                        for payer_state in bundle.payer_history_states:
                            semantic_key = (
                                age_band.age_min_inclusive,
                                age_band.age_max_exclusive,
                                income_band.income_band_id,
                                household_type.household_type_id,
                                gaming_state.value,
                                payer_state.value,
                            )
                            evidence_cell = result_by_semantic_key[semantic_key]
                            global_mass = (
                                Fraction(
                                    jurisdiction.target_population_count,
                                    total_population_count,
                                )
                                * evidence_cell.target_mass
                            )
                            target_population = (
                                global_mass * total_population_count
                            )
                            cells.append(
                                PopulationCalibrationCell(
                                    cell_ordinal=len(cells),
                                    jurisdiction_code=jurisdiction.jurisdiction_code,
                                    age_band_id=age_by_key[
                                        (
                                            evidence_cell.age_min_inclusive,
                                            evidence_cell.age_max_exclusive,
                                        )
                                    ].age_band_id,
                                    income_band_id=evidence_cell.household_income_band,
                                    household_type_id=household_by_id[
                                        evidence_cell.household_type
                                    ].household_type_id,
                                    gaming_state=evidence_cell.gaming_state,
                                    payer_history_state=(
                                        evidence_cell.payer_history_state
                                    ),
                                    evidence_cell_id=evidence_cell.cell_id,
                                    target_mass_numerator=global_mass.numerator,
                                    target_mass_denominator=global_mass.denominator,
                                    target_population_numerator=(
                                        target_population.numerator
                                    ),
                                    target_population_denominator=(
                                        target_population.denominator
                                    ),
                                )
                            )
    calibration_digests = tuple(
        jurisdiction.calibration_evidence_sha256
        for jurisdiction in bundle.jurisdictions
    )
    payload = {
        "schema_version": POPULATION_DESIGN_SCHEMA_VERSION,
        "design_id": bundle.design_id,
        "design_bundle_sha256": bundle.bundle_sha256,
        "population_evidence_bundle_sha256": (
            bundle.population_evidence_bundle_sha256
        ),
        "calibration_evidence_sha256s": list(calibration_digests),
        "domain_sha256": bundle.domain_sha256,
        "total_population_count": total_population_count,
        "total_population_count_decimal": str(total_population_count),
        "cell_count": len(cells),
        "cells": [cell.snapshot() for cell in cells],
    }
    return PopulationCalibrationTarget(
        design_id=bundle.design_id,
        design_bundle_sha256=bundle.bundle_sha256,
        population_evidence_bundle_sha256=(
            bundle.population_evidence_bundle_sha256
        ),
        calibration_evidence_sha256s=calibration_digests,
        domain_sha256=bundle.domain_sha256,
        total_population_count=total_population_count,
        cells=tuple(cells),
        calibration_target_sha256=sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    )


def apportion_population_hamilton(
    target: PopulationCalibrationTarget,
    player_count: int,
    *,
    first_player_id: int = 0,
) -> PopulationApportionmentPlan:
    """Allocate exact cell masses with deterministic largest remainders."""

    if type(target) is not PopulationCalibrationTarget:
        raise TypeError("target must be a PopulationCalibrationTarget")
    _strict_int(
        player_count,
        name="player_count",
        minimum=1,
        maximum=MAX_SAMPLE_PLAYER_COUNT,
    )
    _strict_int(
        first_player_id,
        name="first_player_id",
        minimum=0,
        maximum=MAX_SAMPLE_PLAYER_COUNT,
    )
    if first_player_id + player_count > MAX_SAMPLE_PLAYER_COUNT:
        raise PopulationDesignValidationError(
            "apportionment player id interval exceeds the supported range"
        )
    counts = _exact_hamilton_counts(target.cells, player_count)
    unrepresented = tuple(
        cell.cell_ordinal
        for cell, count in zip(target.cells, counts)
        if cell.target_mass > 0 and count == 0
    )
    if unrepresented:
        raise PopulationDesignVerificationError(
            "Hamilton plan leaves positive-mass cells unrepresented; increase "
            f"player_count; first_cell_ordinal={unrepresented[0]}"
        )
    apportioned_cells: list[PopulationApportionmentCell] = []
    for cell, count in zip(target.cells, counts):
        if count == 0:
            analysis_weight = Fraction()
            expansion_weight = Fraction()
        else:
            analysis_weight = cell.target_mass / count
            expansion_weight = cell.target_population / count
        apportioned_cells.append(
            PopulationApportionmentCell(
                calibration_cell=cell,
                sample_count=count,
                analysis_weight_numerator=analysis_weight.numerator,
                analysis_weight_denominator=analysis_weight.denominator,
                expansion_weight_numerator=expansion_weight.numerator,
                expansion_weight_denominator=expansion_weight.denominator,
            )
        )
    payload = {
        "schema_version": POPULATION_DESIGN_SCHEMA_VERSION,
        "recipe": EXACT_RATIONAL_HAMILTON_V1,
        "calibration_target_sha256": target.calibration_target_sha256,
        "design_id": target.design_id,
        "design_bundle_sha256": target.design_bundle_sha256,
        "domain_sha256": target.domain_sha256,
        "player_count": player_count,
        "player_count_decimal": str(player_count),
        "first_player_id": first_player_id,
        "first_player_id_decimal": str(first_player_id),
        "last_player_id_exclusive": first_player_id + player_count,
        "last_player_id_exclusive_decimal": str(first_player_id + player_count),
        "total_population_count": target.total_population_count,
        "total_population_count_decimal": str(target.total_population_count),
        "cell_count": len(apportioned_cells),
        "cells": [cell.snapshot() for cell in apportioned_cells],
    }
    return PopulationApportionmentPlan(
        calibration_target=target,
        recipe=EXACT_RATIONAL_HAMILTON_V1,
        calibration_target_sha256=target.calibration_target_sha256,
        design_id=target.design_id,
        design_bundle_sha256=target.design_bundle_sha256,
        domain_sha256=target.domain_sha256,
        player_count=player_count,
        first_player_id=first_player_id,
        total_population_count=target.total_population_count,
        cells=tuple(apportioned_cells),
        apportionment_sha256=sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    )


def validate_population_apportionment_snapshot(
    snapshot: object,
    target: PopulationCalibrationTarget,
) -> PopulationApportionmentPlan:
    """Recompute Hamilton counts and reject any serialized plan divergence."""

    if type(target) is not PopulationCalibrationTarget:
        raise TypeError("target must be a PopulationCalibrationTarget")
    row = _mapping(snapshot, name="population apportionment snapshot")
    _exact_keys(
        row,
        _APPORTIONMENT_SNAPSHOT_KEYS,
        name="population apportionment snapshot",
    )
    player_count = row.get("player_count")
    first_player_id = row.get("first_player_id")
    _strict_int(
        player_count,
        name="snapshot player_count",
        minimum=1,
        maximum=MAX_SAMPLE_PLAYER_COUNT,
    )
    _strict_int(
        first_player_id,
        name="snapshot first_player_id",
        minimum=0,
        maximum=MAX_SAMPLE_PLAYER_COUNT,
    )
    assert type(player_count) is int
    assert type(first_player_id) is int
    expected = apportion_population_hamilton(
        target,
        player_count,
        first_player_id=first_player_id,
    )
    if _canonical_json(expected.snapshot()) != _canonical_json(dict(row)):
        raise PopulationDesignValidationError(
            "population apportionment snapshot is not canonical or was tampered"
        )
    return expected


def _verify_design_evidence_bindings(
    bundle: PopulationDesignBundle,
    *,
    evidence_bundle: PopulationEvidenceBundle,
    evidence_results: tuple[PopulationEvidenceResult, ...],
) -> None:
    if bundle.population_evidence_bundle_sha256 != evidence_bundle.bundle_sha256:
        raise PopulationDesignVerificationError(
            "population-design evidence bundle digest does not match verified bytes"
        )
    observed_digests = tuple(result.evidence_sha256 for result in evidence_results)
    if bundle.population_evidence_result_sha256s != observed_digests:
        raise PopulationDesignVerificationError(
            "population-design evidence result digests/order do not match verified results"
        )
    if bundle.jurisdictions and (
        evidence_bundle.provenance_status is not ProvenanceStatus.CALIBRATED
    ):
        raise PopulationDesignVerificationError(
            "a populated design requires CALIBRATED population evidence"
        )
    results_by_binding = {result.binding_id: result for result in evidence_results}
    bindings_by_id = {
        binding.binding_id: binding for binding in evidence_bundle.bindings
    }
    expected_order: list[str] = []
    result_by_binding_and_cell: dict[
        tuple[str, str], PopulationEvidenceCell
    ] = {}
    for jurisdiction in bundle.jurisdictions:
        jurisdiction_income_bands = tuple(
            band
            for band in bundle.income_bands
            if band.jurisdiction_code == jurisdiction.jurisdiction_code
        )
        expected_semantic_keys = {
            (
                age_band.age_min_inclusive,
                age_band.age_max_exclusive,
                income_band.income_band_id,
                household_type.household_type_id,
                gaming_state.value,
                payer_state.value,
            )
            for age_band in bundle.age_bands
            for income_band in jurisdiction_income_bands
            for household_type in bundle.household_types
            for gaming_state in bundle.gaming_states
            for payer_state in bundle.payer_history_states
        }
        pair = (
            (
                PopulationEstimandRole.CALIBRATION,
                jurisdiction.calibration_binding_id,
                jurisdiction.calibration_target_population_id,
                jurisdiction.calibration_evidence_sha256,
            ),
            (
                PopulationEstimandRole.VALIDATION,
                jurisdiction.validation_binding_id,
                jurisdiction.validation_target_population_id,
                jurisdiction.validation_evidence_sha256,
            ),
        )
        pair_bindings: list[PopulationEvidenceBinding] = []
        for role, binding_id, target_population_id, evidence_sha in pair:
            expected_order.append(binding_id)
            result = results_by_binding.get(binding_id)
            binding = bindings_by_id.get(binding_id)
            if result is None or binding is None:
                raise PopulationDesignVerificationError(
                    f"population-design binding is absent from evidence: {binding_id}"
                )
            if (
                result.evidence_sha256 != evidence_sha
                or result.estimand_role is not role
                or binding.estimand_role is not role
                or result.target_population_id != target_population_id
                or binding.target_population_id != target_population_id
                or result.jurisdiction_code != jurisdiction.jurisdiction_code
                or binding.jurisdiction_code != jurisdiction.jurisdiction_code
                or binding.status is not ProvenanceStatus.CALIBRATED
            ):
                raise PopulationDesignVerificationError(
                    f"population-design {role.value} binding metadata differ from "
                    f"verified evidence for {jurisdiction.jurisdiction_code}"
                )
            observed_semantic_keys = {cell.semantic_key for cell in result.cells}
            if observed_semantic_keys != expected_semantic_keys or len(
                result.cells
            ) != len(expected_semantic_keys):
                missing = expected_semantic_keys - observed_semantic_keys
                extra = observed_semantic_keys - expected_semantic_keys
                raise PopulationDesignVerificationError(
                    "population evidence does not exactly cover the declared Cartesian "
                    f"domain for {binding_id}; missing={len(missing)}, extra={len(extra)}"
                )
            for cell in result.cells:
                result_by_binding_and_cell[(binding_id, cell.cell_id)] = cell
            pair_bindings.append(binding)
        _verify_role_binding_semantics(
            pair_bindings[0],
            pair_bindings[1],
            bundle=bundle,
            jurisdiction=jurisdiction,
        )
    if tuple(expected_order) != tuple(result.binding_id for result in evidence_results):
        raise PopulationDesignVerificationError(
            "verified evidence results must be ordered by jurisdiction with "
            "CALIBRATION then VALIDATION"
        )
    weights: dict[tuple[str, str], Fraction] = {}
    for record in bundle.partition.records:
        key = (record.binding_id, record.cell_id)
        if key not in result_by_binding_and_cell:
            raise PopulationDesignVerificationError(
                "partition record references a cell absent from its exact evidence "
                f"result: binding={record.binding_id}, cell={record.cell_id}"
            )
        weights[key] = weights.get(key, Fraction()) + record.record_weight
    for key, cell in result_by_binding_and_cell.items():
        if weights.get(key, Fraction()) != cell.target_mass:
            raise PopulationDesignVerificationError(
                "partition record weights do not exactly reconstruct evidence cell: "
                f"binding={key[0]}, cell={key[1]}"
            )


def _verify_role_binding_semantics(
    calibration: PopulationEvidenceBinding,
    validation: PopulationEvidenceBinding,
    *,
    bundle: PopulationDesignBundle,
    jurisdiction: PopulationDesignJurisdiction,
) -> None:
    fields = (
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
    )
    if any(
        getattr(calibration, field) != getattr(validation, field)
        for field in fields
    ):
        raise PopulationDesignVerificationError(
            "calibration and validation bindings must describe exactly the same "
            f"population semantics for {jurisdiction.jurisdiction_code}"
        )
    if not bundle.age_bands:
        raise PopulationDesignVerificationError(
            "a populated design requires an explicit age domain"
        )
    if (
        calibration.age_min_inclusive != bundle.age_bands[0].age_min_inclusive
        or calibration.age_max_exclusive
        != bundle.age_bands[-1].age_max_exclusive
    ):
        raise PopulationDesignVerificationError(
            "evidence age scope differs from the complete declared age domain"
        )
    if not bundle.income_bands:
        raise PopulationDesignVerificationError(
            "a populated design requires an explicit income domain"
        )
    if (
        calibration.household_income_currency
        != next(
            band.currency
            for band in bundle.income_bands
            if band.jurisdiction_code == jurisdiction.jurisdiction_code
        )
        or calibration.household_income_period
        != next(
            band.period
            for band in bundle.income_bands
            if band.jurisdiction_code == jurisdiction.jurisdiction_code
        )
    ):
        raise PopulationDesignVerificationError(
            "evidence income currency/period differ from the declared income domain"
        )


def _parse_age_band(row: Mapping[str, object]) -> PopulationAgeBand:
    _exact_keys(row, _AGE_BAND_KEYS, name="population-design age band")
    return PopulationAgeBand(
        ordinal=_required_int(row, "ordinal", minimum=0),
        age_band_id=_required_string(row, "age_band_id"),
        age_min_inclusive=_required_int(
            row,
            "age_min_inclusive",
            minimum=0,
            maximum=199,
        ),
        age_max_exclusive=_required_int(
            row,
            "age_max_exclusive",
            minimum=1,
            maximum=200,
        ),
    )


def _parse_income_band(row: Mapping[str, object]) -> PopulationIncomeBand:
    _exact_keys(row, _INCOME_BAND_KEYS, name="population-design income band")
    return PopulationIncomeBand(
        ordinal=_required_int(row, "ordinal", minimum=0),
        jurisdiction_code=_required_string(row, "jurisdiction_code"),
        income_band_id=_required_string(row, "income_band_id"),
        definition=_required_string(row, "definition"),
        currency=_required_string(row, "currency"),
        period=_required_string(row, "period"),
        lower_unbounded=_required_bool(row, "lower_unbounded"),
        lower_bound_numerator=_required_int(
            row,
            "lower_bound_numerator",
            minimum=-(1 << 63),
            maximum=(1 << 63) - 1,
        ),
        lower_bound_denominator=_required_int(
            row,
            "lower_bound_denominator",
            minimum=1,
            maximum=(1 << 63) - 1,
        ),
        upper_unbounded=_required_bool(row, "upper_unbounded"),
        upper_bound_numerator=_required_int(
            row,
            "upper_bound_numerator",
            minimum=-(1 << 63),
            maximum=(1 << 63) - 1,
        ),
        upper_bound_denominator=_required_int(
            row,
            "upper_bound_denominator",
            minimum=1,
            maximum=(1 << 63) - 1,
        ),
    )


def _parse_household_type(
    row: Mapping[str, object],
) -> PopulationHouseholdType:
    _exact_keys(
        row,
        _HOUSEHOLD_TYPE_KEYS,
        name="population-design household type",
    )
    return PopulationHouseholdType(
        ordinal=_required_int(row, "ordinal", minimum=0),
        household_type_id=_required_string(row, "household_type_id"),
        definition=_required_string(row, "definition"),
    )


def _parse_jurisdiction(
    row: Mapping[str, object],
) -> PopulationDesignJurisdiction:
    _exact_keys(
        row,
        _JURISDICTION_KEYS,
        name="population-design jurisdiction",
    )
    return PopulationDesignJurisdiction(
        jurisdiction_code=_required_string(row, "jurisdiction_code"),
        target_population_count=_required_int(
            row,
            "target_population_count",
            minimum=1,
            maximum=MAX_TARGET_POPULATION_COUNT,
        ),
        calibration_binding_id=_required_string(
            row,
            "calibration_binding_id",
        ),
        calibration_target_population_id=_required_string(
            row,
            "calibration_target_population_id",
        ),
        calibration_evidence_sha256=_required_string(
            row,
            "calibration_evidence_sha256",
        ),
        validation_binding_id=_required_string(row, "validation_binding_id"),
        validation_target_population_id=_required_string(
            row,
            "validation_target_population_id",
        ),
        validation_evidence_sha256=_required_string(
            row,
            "validation_evidence_sha256",
        ),
    )


def _parse_partition(row: Mapping[str, object]) -> PopulationPartitionSpec:
    return PopulationPartitionSpec(
        identity_namespace=_required_string(row, "identity_namespace"),
        record_id_recipe=_required_string(row, "record_id_recipe"),
        cluster_id_recipe=_required_string(row, "cluster_id_recipe"),
        role_assignment_recipe=_required_string(row, "role_assignment_recipe"),
        assignment_seed_sha256=_required_string(row, "assignment_seed_sha256"),
        calibration_threshold_numerator=_required_int(
            row,
            "calibration_threshold_numerator",
            minimum=1,
        ),
        calibration_threshold_denominator=_required_int(
            row,
            "calibration_threshold_denominator",
            minimum=1,
        ),
        records=tuple(
            _parse_partition_record(record)
            for record in _required_table_list(row, "records")
        ),
    )


def _parse_partition_record(
    row: Mapping[str, object],
) -> PopulationPartitionRecord:
    _exact_keys(
        row,
        _PARTITION_RECORD_KEYS,
        name="population-design partition record",
    )
    role_text = _required_string(row, "estimand_role")
    try:
        role = PopulationEstimandRole(role_text)
    except ValueError as exc:
        raise PopulationDesignValidationError(
            "partition record estimand_role must be CALIBRATION or VALIDATION"
        ) from exc
    return PopulationPartitionRecord(
        record_identity_sha256=_required_string(
            row,
            "record_identity_sha256",
        ),
        cluster_identity_sha256=_required_string(
            row,
            "cluster_identity_sha256",
        ),
        estimand_role=role,
        binding_id=_required_string(row, "binding_id"),
        cell_id=_required_string(row, "cell_id"),
        record_weight_numerator=_required_int(
            row,
            "record_weight_numerator",
            minimum=1,
        ),
        record_weight_denominator=_required_int(
            row,
            "record_weight_denominator",
            minimum=1,
        ),
    )


def _parse_gaming_states(value: object) -> tuple[PopulationGamingState, ...]:
    strings = _strict_string_array(value, name="gaming_states")
    try:
        return tuple(PopulationGamingState(item) for item in strings)
    except ValueError as exc:
        raise PopulationDesignValidationError("gaming_states are invalid") from exc


def _parse_payer_states(
    value: object,
) -> tuple[PopulationPayerHistoryState, ...]:
    strings = _strict_string_array(value, name="payer_history_states")
    try:
        return tuple(PopulationPayerHistoryState(item) for item in strings)
    except ValueError as exc:
        raise PopulationDesignValidationError(
            "payer_history_states are invalid"
        ) from exc


def _secure_read_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
) -> bytes:
    candidate = Path(path)
    before = _lstat_regular_file(candidate, description=description)
    if before.st_size > maximum_bytes:
        raise PopulationDesignVerificationError(
            f"{description} exceeds the {maximum_bytes}-byte safety limit"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise PopulationDesignVerificationError(
            f"cannot open {description}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _is_reparse(opened):
            raise PopulationDesignVerificationError(
                f"{description} must be a non-reparse regular file"
            )
        if not _same_file_identity(before, opened):
            raise PopulationDesignVerificationError(
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
                raise PopulationDesignVerificationError(
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
        raise PopulationDesignVerificationError(f"{description} changed while read")
    content = b"".join(chunks)
    if len(content) != after_open.st_size:
        raise PopulationDesignVerificationError(
            f"{description} was not read completely"
        )
    return content


def _lstat_regular_file(path: Path, *, description: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise PopulationDesignVerificationError(
            f"{description} does not exist"
        ) from exc
    if path.is_symlink() or _is_reparse(observed):
        raise PopulationDesignVerificationError(
            f"{description} cannot be a symlink or reparse point"
        )
    if not stat.S_ISREG(observed.st_mode):
        raise PopulationDesignVerificationError(
            f"{description} must be a regular file"
        )
    return observed


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


def _exact_keys(
    values: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    if any(type(key) is not str for key in values):
        raise PopulationDesignValidationError(f"{name} keys must be strings")
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise PopulationDesignValidationError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PopulationDesignValidationError(f"{name} must be a table")
    return value


def _required_table(
    values: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    return _mapping(values.get(field), name=field)


def _required_table_list(
    values: Mapping[str, object],
    field: str,
) -> tuple[Mapping[str, object], ...]:
    raw = values.get(field)
    if type(raw) is not list:
        raise PopulationDesignValidationError(
            f"{field} must be an array of tables"
        )
    return tuple(
        _mapping(item, name=f"{field}[{index}]")
        for index, item in enumerate(raw)
    )


def _required_string(values: Mapping[str, object], field: str) -> str:
    value = values.get(field)
    if type(value) is not str or not value.strip() or value != value.strip():
        raise PopulationDesignValidationError(
            f"{field} must be non-empty text without surrounding whitespace"
        )
    return value


def _required_string_list(
    values: Mapping[str, object],
    field: str,
) -> tuple[str, ...]:
    return _strict_string_array(values.get(field), name=field)


def _strict_string_array(value: object, *, name: str) -> tuple[str, ...]:
    if type(value) is not list or any(
        type(item) is not str or not item for item in value
    ):
        raise PopulationDesignValidationError(f"{name} must be an array of text")
    return tuple(value)


def _required_bool(values: Mapping[str, object], field: str) -> bool:
    value = values.get(field)
    if type(value) is not bool:
        raise PopulationDesignValidationError(f"{field} must be a boolean")
    return value


def _required_int(
    values: Mapping[str, object],
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = values.get(field)
    _strict_int(value, name=field, minimum=minimum, maximum=maximum)
    assert type(value) is int
    return value


def _parse_provenance_status(value: str) -> ProvenanceStatus:
    try:
        return ProvenanceStatus(value)
    except ValueError as exc:
        raise PopulationDesignValidationError(
            f"invalid population-design provenance status: {value}"
        ) from exc


def _typed_tuple(value: object, item_type: type, *, name: str) -> None:
    if type(value) is not tuple or any(type(item) is not item_type for item in value):
        raise PopulationDesignValidationError(
            f"{name} must be an immutable tuple of exact {item_type.__name__} values"
        )


def _unique_domain_ids(values: tuple[str, ...], *, name: str) -> None:
    if len({value.casefold() for value in values}) != len(values):
        raise PopulationDesignValidationError(
            f"{name} repeat under case-insensitive matching"
        )


def _identifier(value: object, *, name: str) -> None:
    if type(value) is not str or _ID_PATTERN.fullmatch(value) is None:
        raise PopulationDesignValidationError(
            f"{name} must be a canonical ASCII identifier"
        )


def _text(value: object, *, name: str) -> None:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise PopulationDesignValidationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )


def _jurisdiction_code(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 2
        or any(character < "A" or character > "Z" for character in value)
    ):
        raise PopulationDesignValidationError(
            "jurisdiction_code must be a two-letter uppercase ASCII code"
        )


def _sha256(value: object, *, name: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise PopulationDesignValidationError(
            f"{name} must be lowercase SHA-256 hex"
        )


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise PopulationDesignValidationError(f"{name} must be a strict integer")
    if value < minimum or (maximum is not None and value > maximum):
        interval = (
            f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        )
        raise PopulationDesignValidationError(f"{name} must be {interval}")


def _reduced_fraction_parts(
    numerator: object,
    denominator: object,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if type(numerator) is not int or type(denominator) is not int:
        raise PopulationDesignValidationError(
            f"{name} numerator and denominator must be strict integers"
        )
    if denominator <= 0:
        raise PopulationDesignValidationError(
            f"{name} denominator must be positive"
        )
    if positive and numerator <= 0:
        raise PopulationDesignValidationError(f"{name} must be positive")
    if nonnegative and numerator < 0:
        raise PopulationDesignValidationError(f"{name} must be non-negative")
    if math.gcd(numerator, denominator) != 1:
        raise PopulationDesignValidationError(f"{name} must be in lowest terms")


__all__ = [
    "CANONICAL_SOURCE_CLUSTER_ID_V1",
    "CANONICAL_SOURCE_RECORD_ID_V1",
    "DEFAULT_POPULATION_DESIGN_BUNDLE_PATH",
    "EXACT_RATIONAL_HAMILTON_V1",
    "MAX_POPULATION_DESIGN_BYTES",
    "MAX_SAMPLE_PLAYER_COUNT",
    "MAX_TARGET_POPULATION_COUNT",
    "POPULATION_DESIGN_SCHEMA_VERSION",
    "SHA256_CLUSTER_THRESHOLD_V1",
    "PopulationAgeBand",
    "PopulationApportionmentCell",
    "PopulationApportionmentPlan",
    "PopulationCalibrationCell",
    "PopulationCalibrationTarget",
    "PopulationDesignBundle",
    "PopulationDesignJurisdiction",
    "PopulationDesignValidationError",
    "PopulationDesignVerification",
    "PopulationDesignVerificationError",
    "PopulationHouseholdType",
    "PopulationIncomeBand",
    "PopulationPartitionRecord",
    "PopulationPartitionSpec",
    "apportion_population_hamilton",
    "assigned_population_partition_role",
    "build_population_calibration_target",
    "load_and_verify_population_design_bundle",
    "load_population_design_bundle",
    "validate_population_apportionment_snapshot",
    "validate_population_design_snapshot",
    "verify_population_design_bundle",
]
