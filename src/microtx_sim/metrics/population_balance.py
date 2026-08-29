"""Exact pre-treatment balance checks for projected target populations.

The artifact in this module is deliberately narrower than an empirical-data
claim.  It re-attests one already-built population-projection execution and
answers whether its exact joint-cell assignment agrees with its Hamilton plan,
and whether the runtime pre-treatment columns remain inside the mapped cell
domains.  It makes no claim about source quality, source identity, held-out
performance, treatment outcomes, or fitness for a research campaign.

All static-to-runtime interpretation is supplied by
``PopulationProjectionAdapter``.  This module does not define another mapping
contract.  A successful artifact is content-addressed over the adapter,
execution, ordered player IDs, exact cell counts and weights, and the runtime
jurisdiction/age/income/household columns that were actually checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
import re

import numpy as np

from ..agents.players import (
    PlayerTable,
    ProjectedPopulationAssignment,
    ProjectedPopulationCellMetadata,
    ProjectedPopulationMetadata,
    projected_population_assignment_sha256,
    projected_population_plan_sha256,
)
from ..consumers.population import PopulationProjectionCell
from ..data.population_design import (
    MAX_SAMPLE_PLAYER_COUNT,
    PopulationApportionmentCell,
    PopulationApportionmentPlan,
    PopulationCalibrationCell,
    PopulationCalibrationTarget,
)
from ..data.population_evidence import (
    PopulationGamingState,
    PopulationPayerHistoryState,
)
from ..data.population_projection import (
    PopulationProjectionAdapter,
    PopulationProjectionAdapterCell,
    PopulationProjectionExecution,
    PopulationRuntimeMappingBundle,
    PopulationRuntimeMappingEntry,
    population_projection_ordered_player_ids_sha256,
    verify_population_projection_adapter,
    verify_population_projection_execution,
)


POPULATION_BALANCE_SCHEMA_VERSION = "1.0"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_MAX_EXACT_INTEGER_BITS = 4096


class PopulationBalanceValidationError(ValueError):
    """Raised when an exact population-balance artifact cannot be trusted."""


@dataclass(frozen=True, slots=True)
class PopulationBalanceCellResult:
    """Exact planned and realized quantities for one full joint cell."""

    cell_ordinal: int
    evidence_cell_id: str
    jurisdiction_code: str
    age_band_id: str
    income_band_id: str
    household_type_id: str
    gaming_state: PopulationGamingState
    payer_history_state: PopulationPayerHistoryState
    projected_cell_id: str
    mapping_entry_sha256: str

    runtime_age_min_inclusive: int
    runtime_age_max_exclusive: int
    runtime_income_band_id: str
    runtime_income_min_cents: int
    runtime_income_max_cents_exclusive: int
    runtime_household_type: str
    modeled_players_per_household: int

    planned_sample_count: int
    realized_sample_count: int
    sample_count_discrepancy: int

    target_mass_numerator: int
    target_mass_denominator: int
    sidecar_mass_numerator: int
    sidecar_mass_denominator: int
    declared_mass_discrepancy_numerator: int
    declared_mass_discrepancy_denominator: int

    analysis_weight_numerator: int
    analysis_weight_denominator: int
    realized_mass_numerator: int
    realized_mass_denominator: int
    realized_mass_discrepancy_numerator: int
    realized_mass_discrepancy_denominator: int

    def __post_init__(self) -> None:
        _strict_int(
            self.cell_ordinal,
            name="balance cell_ordinal",
            minimum=0,
            maximum=MAX_SAMPLE_PLAYER_COUNT,
        )
        for value, name in (
            (self.evidence_cell_id, "evidence_cell_id"),
            (self.age_band_id, "age_band_id"),
            (self.income_band_id, "income_band_id"),
            (self.household_type_id, "household_type_id"),
            (self.projected_cell_id, "projected_cell_id"),
            (self.runtime_income_band_id, "runtime_income_band_id"),
            (self.runtime_household_type, "runtime_household_type"),
        ):
            if name in {
                "projected_cell_id",
                "runtime_income_band_id",
                "runtime_household_type",
            }:
                _nonempty_text(value, name=name)
            else:
                _identifier(value, name=name)
        _jurisdiction_code(self.jurisdiction_code)
        _sha256(self.mapping_entry_sha256, name="mapping_entry_sha256")
        if type(self.gaming_state) is not PopulationGamingState:
            raise TypeError("gaming_state must be PopulationGamingState")
        if type(self.payer_history_state) is not PopulationPayerHistoryState:
            raise TypeError(
                "payer_history_state must be PopulationPayerHistoryState"
            )

        _strict_int(
            self.runtime_age_min_inclusive,
            name="runtime_age_min_inclusive",
            minimum=0,
            maximum=32_767,
        )
        _strict_int(
            self.runtime_age_max_exclusive,
            name="runtime_age_max_exclusive",
            minimum=1,
            maximum=32_768,
        )
        if self.runtime_age_min_inclusive >= self.runtime_age_max_exclusive:
            raise PopulationBalanceValidationError(
                "runtime balance age interval must be non-empty"
            )
        _strict_int(
            self.runtime_income_min_cents,
            name="runtime_income_min_cents",
            minimum=0,
            maximum=np.iinfo(np.int64).max,
        )
        _strict_int(
            self.runtime_income_max_cents_exclusive,
            name="runtime_income_max_cents_exclusive",
            minimum=1,
            maximum=np.iinfo(np.int64).max,
        )
        if (
            self.runtime_income_min_cents
            >= self.runtime_income_max_cents_exclusive
        ):
            raise PopulationBalanceValidationError(
                "runtime balance income interval must be non-empty"
            )
        _strict_int(
            self.modeled_players_per_household,
            name="modeled_players_per_household",
            minimum=1,
            maximum=MAX_SAMPLE_PLAYER_COUNT,
        )

        for value, name in (
            (self.planned_sample_count, "planned_sample_count"),
            (self.realized_sample_count, "realized_sample_count"),
        ):
            _strict_int(
                value,
                name=name,
                minimum=0,
                maximum=MAX_SAMPLE_PLAYER_COUNT,
            )
        _strict_int(
            self.sample_count_discrepancy,
            name="sample_count_discrepancy",
            minimum=-MAX_SAMPLE_PLAYER_COUNT,
            maximum=MAX_SAMPLE_PLAYER_COUNT,
        )

        target_mass = _fraction(
            self.target_mass_numerator,
            self.target_mass_denominator,
            name="target_mass",
            nonnegative=True,
        )
        sidecar_mass = _fraction(
            self.sidecar_mass_numerator,
            self.sidecar_mass_denominator,
            name="sidecar_mass",
            nonnegative=True,
        )
        declared_discrepancy = _fraction(
            self.declared_mass_discrepancy_numerator,
            self.declared_mass_discrepancy_denominator,
            name="declared_mass_discrepancy",
        )
        analysis_weight = _fraction(
            self.analysis_weight_numerator,
            self.analysis_weight_denominator,
            name="analysis_weight",
            nonnegative=True,
        )
        realized_mass = _fraction(
            self.realized_mass_numerator,
            self.realized_mass_denominator,
            name="realized_mass",
            nonnegative=True,
        )
        realized_discrepancy = _fraction(
            self.realized_mass_discrepancy_numerator,
            self.realized_mass_discrepancy_denominator,
            name="realized_mass_discrepancy",
        )

        expected_count_discrepancy = (
            self.realized_sample_count - self.planned_sample_count
        )
        if self.sample_count_discrepancy != expected_count_discrepancy:
            raise PopulationBalanceValidationError(
                "sample_count_discrepancy is not realized minus planned count"
            )
        if declared_discrepancy != sidecar_mass - target_mass:
            raise PopulationBalanceValidationError(
                "declared mass discrepancy does not match sidecar minus target mass"
            )
        if realized_mass != analysis_weight * self.realized_sample_count:
            raise PopulationBalanceValidationError(
                "realized mass is not analysis weight times realized count"
            )
        if realized_discrepancy != realized_mass - target_mass:
            raise PopulationBalanceValidationError(
                "realized mass discrepancy does not match realized minus target mass"
            )
        if self.planned_sample_count == 0:
            if target_mass != 0 or analysis_weight != 0:
                raise PopulationBalanceValidationError(
                    "only a zero-mass balance cell may have zero planned players"
                )
        elif analysis_weight != target_mass / self.planned_sample_count:
            raise PopulationBalanceValidationError(
                "balance analysis weight differs from target mass / planned count"
            )

        if target_mass > 0 and self.realized_sample_count == 0:
            raise PopulationBalanceValidationError(
                "positive target-mass balance cell has no realized players"
            )
        if (
            self.sample_count_discrepancy != 0
            or declared_discrepancy != 0
            or realized_discrepancy != 0
        ):
            raise PopulationBalanceValidationError(
                "projected joint-cell counts or masses differ from the exact plan"
            )

    @property
    def target_mass(self) -> Fraction:
        return Fraction(self.target_mass_numerator, self.target_mass_denominator)

    @property
    def sidecar_mass(self) -> Fraction:
        return Fraction(self.sidecar_mass_numerator, self.sidecar_mass_denominator)

    @property
    def analysis_weight(self) -> Fraction:
        return Fraction(
            self.analysis_weight_numerator,
            self.analysis_weight_denominator,
        )

    @property
    def realized_mass(self) -> Fraction:
        return Fraction(
            self.realized_mass_numerator,
            self.realized_mass_denominator,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "cell_ordinal": self.cell_ordinal,
            "cell_ordinal_decimal": str(self.cell_ordinal),
            "evidence_cell_id": self.evidence_cell_id,
            "joint_cell": {
                "jurisdiction_code": self.jurisdiction_code,
                "age_band_id": self.age_band_id,
                "income_band_id": self.income_band_id,
                "household_type_id": self.household_type_id,
                "gaming_state": self.gaming_state.value,
                "payer_history_state": self.payer_history_state.value,
            },
            "runtime_mapping": {
                "projected_cell_id": self.projected_cell_id,
                "mapping_entry_sha256": self.mapping_entry_sha256,
                "age_min_inclusive": self.runtime_age_min_inclusive,
                "age_max_exclusive": self.runtime_age_max_exclusive,
                "income_band_id": self.runtime_income_band_id,
                "income_min_cents": self.runtime_income_min_cents,
                "income_max_cents_exclusive": (
                    self.runtime_income_max_cents_exclusive
                ),
                "household_type": self.runtime_household_type,
                "modeled_players_per_household": (
                    self.modeled_players_per_household
                ),
            },
            "counts": {
                "planned": self.planned_sample_count,
                "planned_decimal": str(self.planned_sample_count),
                "realized": self.realized_sample_count,
                "realized_decimal": str(self.realized_sample_count),
                "discrepancy": self.sample_count_discrepancy,
                "discrepancy_decimal": str(self.sample_count_discrepancy),
            },
            "target_mass": _fraction_snapshot(self.target_mass),
            "sidecar_mass": _fraction_snapshot(self.sidecar_mass),
            "declared_mass_discrepancy": _fraction_snapshot(
                Fraction(
                    self.declared_mass_discrepancy_numerator,
                    self.declared_mass_discrepancy_denominator,
                )
            ),
            "analysis_weight": _fraction_snapshot(self.analysis_weight),
            "realized_mass": _fraction_snapshot(self.realized_mass),
            "realized_mass_discrepancy": _fraction_snapshot(
                Fraction(
                    self.realized_mass_discrepancy_numerator,
                    self.realized_mass_discrepancy_denominator,
                )
            ),
        }


@dataclass(frozen=True, slots=True)
class PopulationRuntimeMembershipAttestation:
    """Separate hashes and pass statements for checked runtime memberships."""

    player_count: int
    runtime_projection_sha256: str
    assignment_sha256: str
    ordered_player_ids_sha256: str
    household_count: int
    partial_household_count: int
    jurisdiction_members_checked: int
    age_members_checked: int
    income_members_checked: int
    household_members_checked: int
    jurisdiction_membership_passed: bool
    age_membership_passed: bool
    income_membership_passed: bool
    household_membership_passed: bool
    jurisdiction_membership_sha256: str
    age_membership_sha256: str
    income_membership_sha256: str
    household_membership_sha256: str
    membership_sha256: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.player_count, "membership player_count"),
            (self.household_count, "membership household_count"),
            (
                self.partial_household_count,
                "membership partial_household_count",
            ),
            (
                self.jurisdiction_members_checked,
                "jurisdiction_members_checked",
            ),
            (self.age_members_checked, "age_members_checked"),
            (self.income_members_checked, "income_members_checked"),
            (self.household_members_checked, "household_members_checked"),
        ):
            _strict_int(
                value,
                name=name,
                minimum=0,
                maximum=MAX_SAMPLE_PLAYER_COUNT,
            )
        if self.household_count > self.player_count:
            raise PopulationBalanceValidationError(
                "household_count cannot exceed player_count"
            )
        if self.partial_household_count > self.household_count:
            raise PopulationBalanceValidationError(
                "partial_household_count cannot exceed household_count"
            )
        checked = (
            self.jurisdiction_members_checked,
            self.age_members_checked,
            self.income_members_checked,
            self.household_members_checked,
        )
        if checked != (self.player_count,) * 4:
            raise PopulationBalanceValidationError(
                "every runtime membership check must cover every ordered player"
            )
        for value, name in (
            (
                self.jurisdiction_membership_passed,
                "jurisdiction_membership_passed",
            ),
            (self.age_membership_passed, "age_membership_passed"),
            (self.income_membership_passed, "income_membership_passed"),
            (self.household_membership_passed, "household_membership_passed"),
        ):
            if type(value) is not bool:
                raise TypeError(f"{name} must be a strict bool")
            if not value:
                raise PopulationBalanceValidationError(
                    f"{name} must be true for an attested balance artifact"
                )
        for value, name in (
            (
                self.runtime_projection_sha256,
                "membership runtime_projection_sha256",
            ),
            (self.assignment_sha256, "membership assignment_sha256"),
            (
                self.ordered_player_ids_sha256,
                "membership ordered_player_ids_sha256",
            ),
            (
                self.jurisdiction_membership_sha256,
                "jurisdiction_membership_sha256",
            ),
            (self.age_membership_sha256, "age_membership_sha256"),
            (self.income_membership_sha256, "income_membership_sha256"),
            (self.household_membership_sha256, "household_membership_sha256"),
            (self.membership_sha256, "membership_sha256"),
        ):
            _sha256(value, name=name)
        expected = _canonical_sha256(
            b"microtx-sim.population-runtime-membership.v1\0",
            self.attestation_payload(),
        )
        if self.membership_sha256 != expected:
            raise PopulationBalanceValidationError(
                "membership_sha256 does not match its exact attestation payload"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": POPULATION_BALANCE_SCHEMA_VERSION,
            "player_count": self.player_count,
            "player_count_decimal": str(self.player_count),
            "runtime_projection_sha256": self.runtime_projection_sha256,
            "assignment_sha256": self.assignment_sha256,
            "ordered_player_ids_sha256": self.ordered_player_ids_sha256,
            "household_count": self.household_count,
            "household_count_decimal": str(self.household_count),
            "partial_household_count": self.partial_household_count,
            "partial_household_count_decimal": str(
                self.partial_household_count
            ),
            "jurisdiction": {
                "members_checked": self.jurisdiction_members_checked,
                "passed": self.jurisdiction_membership_passed,
                "values_sha256": self.jurisdiction_membership_sha256,
            },
            "age": {
                "members_checked": self.age_members_checked,
                "passed": self.age_membership_passed,
                "values_sha256": self.age_membership_sha256,
            },
            "income": {
                "members_checked": self.income_members_checked,
                "passed": self.income_membership_passed,
                "values_sha256": self.income_membership_sha256,
            },
            "household": {
                "members_checked": self.household_members_checked,
                "passed": self.household_membership_passed,
                "values_sha256": self.household_membership_sha256,
            },
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "membership_sha256": self.membership_sha256,
        }


@dataclass(frozen=True, slots=True)
class PopulationBalanceArtifact:
    """Content-addressed exact balance result for one projection execution."""

    schema_version: str
    execution_sha256: str
    adapter_sha256: str
    mapping_sha256: str
    apportionment_sha256: str
    calibration_target_sha256: str
    design_id: str
    design_bundle_sha256: str
    domain_sha256: str
    runtime_projection_id: str
    runtime_projection_sha256: str
    assignment_sha256: str
    ordered_player_ids_sha256: str
    player_count: int
    cells: tuple[PopulationBalanceCellResult, ...]
    runtime_membership: PopulationRuntimeMembershipAttestation
    exact_balance_passed: bool
    balance_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != POPULATION_BALANCE_SCHEMA_VERSION:
            raise PopulationBalanceValidationError(
                "unsupported population-balance schema version"
            )
        for value, name in (
            (self.execution_sha256, "execution_sha256"),
            (self.adapter_sha256, "adapter_sha256"),
            (self.mapping_sha256, "mapping_sha256"),
            (self.apportionment_sha256, "apportionment_sha256"),
            (self.calibration_target_sha256, "calibration_target_sha256"),
            (self.design_bundle_sha256, "design_bundle_sha256"),
            (self.domain_sha256, "domain_sha256"),
            (self.runtime_projection_sha256, "runtime_projection_sha256"),
            (self.assignment_sha256, "assignment_sha256"),
            (self.ordered_player_ids_sha256, "ordered_player_ids_sha256"),
            (self.balance_sha256, "balance_sha256"),
        ):
            _sha256(value, name=name)
        _identifier(self.design_id, name="design_id")
        _identifier(self.runtime_projection_id, name="runtime_projection_id")
        _strict_int(
            self.player_count,
            name="balance player_count",
            minimum=1,
            maximum=MAX_SAMPLE_PLAYER_COUNT,
        )
        if type(self.cells) is not tuple or any(
            type(cell) is not PopulationBalanceCellResult for cell in self.cells
        ):
            raise TypeError(
                "balance cells must be an immutable tuple of "
                "PopulationBalanceCellResult"
            )
        if not self.cells:
            raise PopulationBalanceValidationError(
                "population balance must contain every declared joint cell"
            )
        for cell in self.cells:
            PopulationBalanceCellResult.__post_init__(cell)
        if tuple(cell.cell_ordinal for cell in self.cells) != tuple(
            range(len(self.cells))
        ):
            raise PopulationBalanceValidationError(
                "population balance cells must use contiguous canonical ordinals"
            )
        if len({cell.projected_cell_id for cell in self.cells}) != len(self.cells):
            raise PopulationBalanceValidationError(
                "population balance repeats a projected cell"
            )
        if sum(cell.planned_sample_count for cell in self.cells) != self.player_count:
            raise PopulationBalanceValidationError(
                "planned balance counts do not sum to player_count"
            )
        if sum(cell.realized_sample_count for cell in self.cells) != self.player_count:
            raise PopulationBalanceValidationError(
                "realized balance counts do not sum to player_count"
            )
        if sum((cell.target_mass for cell in self.cells), Fraction()) != 1:
            raise PopulationBalanceValidationError(
                "balance target masses do not sum exactly to one"
            )
        if sum((cell.sidecar_mass for cell in self.cells), Fraction()) != 1:
            raise PopulationBalanceValidationError(
                "balance sidecar masses do not sum exactly to one"
            )
        if sum((cell.realized_mass for cell in self.cells), Fraction()) != 1:
            raise PopulationBalanceValidationError(
                "balance realized masses do not sum exactly to one"
            )
        if type(self.runtime_membership) is not PopulationRuntimeMembershipAttestation:
            raise TypeError(
                "runtime_membership must be PopulationRuntimeMembershipAttestation"
            )
        PopulationRuntimeMembershipAttestation.__post_init__(
            self.runtime_membership
        )
        if self.runtime_membership.player_count != self.player_count:
            raise PopulationBalanceValidationError(
                "runtime membership player count differs from balance"
            )
        membership_bindings = (
            self.runtime_membership.runtime_projection_sha256,
            self.runtime_membership.assignment_sha256,
            self.runtime_membership.ordered_player_ids_sha256,
        )
        if membership_bindings != (
            self.runtime_projection_sha256,
            self.assignment_sha256,
            self.ordered_player_ids_sha256,
        ):
            raise PopulationBalanceValidationError(
                "runtime membership identities differ from the balance artifact"
            )
        if type(self.exact_balance_passed) is not bool:
            raise TypeError("exact_balance_passed must be a strict bool")
        if not self.exact_balance_passed:
            raise PopulationBalanceValidationError(
                "an exact balance artifact cannot attest a failed comparison"
            )
        expected = _canonical_sha256(
            b"microtx-sim.population-balance.v1\0",
            self.attestation_payload(),
        )
        if self.balance_sha256 != expected:
            raise PopulationBalanceValidationError(
                "balance_sha256 does not match its exact attestation payload"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "execution_sha256": self.execution_sha256,
            "adapter_sha256": self.adapter_sha256,
            "mapping_sha256": self.mapping_sha256,
            "apportionment_sha256": self.apportionment_sha256,
            "calibration_target_sha256": self.calibration_target_sha256,
            "design_id": self.design_id,
            "design_bundle_sha256": self.design_bundle_sha256,
            "domain_sha256": self.domain_sha256,
            "runtime_projection_id": self.runtime_projection_id,
            "runtime_projection_sha256": self.runtime_projection_sha256,
            "assignment_sha256": self.assignment_sha256,
            "ordered_player_ids_sha256": self.ordered_player_ids_sha256,
            "player_count": self.player_count,
            "player_count_decimal": str(self.player_count),
            "cell_count": len(self.cells),
            "cells": [cell.snapshot() for cell in self.cells],
            "runtime_membership": self.runtime_membership.snapshot(),
            "exact_balance_passed": self.exact_balance_passed,
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "balance_sha256": self.balance_sha256,
        }


def build_population_balance_artifact(
    execution: PopulationProjectionExecution,
) -> PopulationBalanceArtifact:
    """Re-attest one projection execution and require exact pre-treatment balance."""

    _reverify_execution(execution)
    adapter = execution.adapter
    plan = adapter.apportionment_plan
    players = execution.players
    assignment = players.projected_population
    assert assignment is not None

    sidecar_indices = _require_exact_cell_alignment(adapter, assignment)
    counts = _realized_counts(assignment, len(adapter.cells))
    cell_results = tuple(
        _build_cell_result(
            adapter_cell,
            plan.cells[adapter_cell.cell_ordinal],
            assignment.metadata.cells[
                sidecar_indices[adapter_cell.cell_ordinal]
            ],
            counts[sidecar_indices[adapter_cell.cell_ordinal]],
        )
        for adapter_cell in adapter.cells
    )
    membership = _attest_runtime_membership(players, assignment)

    common = {
        "schema_version": POPULATION_BALANCE_SCHEMA_VERSION,
        "execution_sha256": execution.execution_sha256,
        "adapter_sha256": adapter.adapter_sha256,
        "mapping_sha256": adapter.mapping_bundle.mapping_sha256,
        "apportionment_sha256": plan.apportionment_sha256,
        "calibration_target_sha256": plan.calibration_target_sha256,
        "design_id": plan.design_id,
        "design_bundle_sha256": plan.design_bundle_sha256,
        "domain_sha256": plan.domain_sha256,
        "runtime_projection_id": execution.runtime_projection_id,
        "runtime_projection_sha256": execution.runtime_projection_sha256,
        "assignment_sha256": execution.assignment_sha256,
        "ordered_player_ids_sha256": execution.ordered_player_ids_sha256,
        "player_count": plan.player_count,
        "cells": cell_results,
        "runtime_membership": membership,
        "exact_balance_passed": True,
    }
    payload = {
        "schema_version": POPULATION_BALANCE_SCHEMA_VERSION,
        "execution_sha256": common["execution_sha256"],
        "adapter_sha256": common["adapter_sha256"],
        "mapping_sha256": common["mapping_sha256"],
        "apportionment_sha256": common["apportionment_sha256"],
        "calibration_target_sha256": common["calibration_target_sha256"],
        "design_id": common["design_id"],
        "design_bundle_sha256": common["design_bundle_sha256"],
        "domain_sha256": common["domain_sha256"],
        "runtime_projection_id": common["runtime_projection_id"],
        "runtime_projection_sha256": common["runtime_projection_sha256"],
        "assignment_sha256": common["assignment_sha256"],
        "ordered_player_ids_sha256": common["ordered_player_ids_sha256"],
        "player_count": plan.player_count,
        "player_count_decimal": str(plan.player_count),
        "cell_count": len(cell_results),
        "cells": [cell.snapshot() for cell in cell_results],
        "runtime_membership": membership.snapshot(),
        "exact_balance_passed": True,
    }
    return PopulationBalanceArtifact(
        **common,
        balance_sha256=_canonical_sha256(
            b"microtx-sim.population-balance.v1\0",
            payload,
        ),
    )


def validate_population_balance_snapshot(
    snapshot: object,
    execution: PopulationProjectionExecution,
) -> PopulationBalanceArtifact:
    """Recompute a balance artifact and reject non-canonical or stale snapshots."""

    _validate_snapshot_tree(snapshot, name="population balance snapshot")
    if type(snapshot) is not dict:
        raise TypeError("population balance snapshot must be an exact dict")
    expected = build_population_balance_artifact(execution)
    if _canonical_json(snapshot) != _canonical_json(expected.snapshot()):
        raise PopulationBalanceValidationError(
            "population balance snapshot is non-canonical, stale, or was mutated"
        )
    return expected


def _reverify_execution(execution: object) -> None:
    if type(execution) is not PopulationProjectionExecution:
        raise TypeError("execution must be PopulationProjectionExecution")
    adapter = execution.adapter
    players = execution.players
    if type(adapter) is not PopulationProjectionAdapter:
        raise TypeError("execution adapter must be PopulationProjectionAdapter")
    if type(players) is not PlayerTable:
        raise TypeError("execution players must be PlayerTable")
    _revalidate_adapter_types(adapter)
    try:
        verified_adapter = verify_population_projection_adapter(adapter)
    except (TypeError, ValueError, OSError) as exc:
        raise PopulationBalanceValidationError(
            "projection adapter could not be re-attested"
        ) from exc
    if verified_adapter is not None and verified_adapter != adapter:
        raise PopulationBalanceValidationError(
            "projection adapter reverification returned different content"
        )

    assignment = players.projected_population
    if type(assignment) is not ProjectedPopulationAssignment:
        raise PopulationBalanceValidationError(
            "population balance requires an exact projected PlayerTable sidecar"
        )
    _revalidate_sidecar(players, assignment)

    # The sidecar bounds check above intentionally precedes the adapter's
    # bincount, so an out-of-domain mutated int32 cannot force a giant allocation.
    try:
        observed_execution = verify_population_projection_execution(execution)
    except (TypeError, ValueError, OSError) as exc:
        raise PopulationBalanceValidationError(
            "projection execution is stale, mutated, or mismatched"
        ) from exc
    if observed_execution is not execution:
        raise PopulationBalanceValidationError(
            "projection execution reverification returned an alias object"
        )

    plan = adapter.apportionment_plan
    if len(players) != plan.player_count:
        raise PopulationBalanceValidationError(
            "projected player count differs from exact apportionment plan"
        )
    expected_ids = np.arange(
        plan.first_player_id,
        plan.last_player_id_exclusive,
        dtype=np.int64,
    )
    if not np.array_equal(players.player_id, expected_ids):
        raise PopulationBalanceValidationError(
            "ordered player IDs differ from the apportionment interval"
        )
    if execution.assignment_sha256 != assignment.assignment_sha256:
        raise PopulationBalanceValidationError(
            "execution assignment identity differs from PlayerTable sidecar"
        )
    if execution.runtime_projection_id != assignment.metadata.projection_id:
        raise PopulationBalanceValidationError(
            "execution projection id differs from PlayerTable sidecar"
        )
    if execution.runtime_projection_sha256 != assignment.metadata.projection_sha256:
        raise PopulationBalanceValidationError(
            "execution projection digest differs from PlayerTable sidecar"
        )
    observed_player_ids_sha256 = population_projection_ordered_player_ids_sha256(
        players.player_id
    )
    if execution.ordered_player_ids_sha256 != observed_player_ids_sha256:
        raise PopulationBalanceValidationError(
            "execution ordered-player identity differs from PlayerTable"
        )


def _revalidate_adapter_types(adapter: PopulationProjectionAdapter) -> None:
    if type(adapter.mapping_bundle) is not PopulationRuntimeMappingBundle:
        raise TypeError(
            "adapter mapping_bundle must be PopulationRuntimeMappingBundle"
        )
    if type(adapter.mapping_bundle.entries) is not tuple or any(
        type(entry) is not PopulationRuntimeMappingEntry
        for entry in adapter.mapping_bundle.entries
    ):
        raise TypeError(
            "adapter mapping entries must contain exact PopulationRuntimeMappingEntry"
        )
    for entry in adapter.mapping_bundle.entries:
        try:
            PopulationRuntimeMappingEntry.__post_init__(entry)
        except (TypeError, ValueError) as exc:
            raise PopulationBalanceValidationError(
                "adapter mapping entry is stale, mutated, or polymorphic"
            ) from exc
    if type(adapter.apportionment_plan) is not PopulationApportionmentPlan:
        raise TypeError(
            "adapter apportionment_plan must be PopulationApportionmentPlan"
        )
    plan = adapter.apportionment_plan
    if type(plan.calibration_target) is not PopulationCalibrationTarget:
        raise TypeError(
            "plan calibration_target must be PopulationCalibrationTarget"
        )
    if type(plan.calibration_target.cells) is not tuple or any(
        type(cell) is not PopulationCalibrationCell
        for cell in plan.calibration_target.cells
    ):
        raise TypeError(
            "calibration target cells must contain exact PopulationCalibrationCell"
        )
    if type(plan.cells) is not tuple or any(
        type(cell) is not PopulationApportionmentCell for cell in plan.cells
    ):
        raise TypeError(
            "plan cells must contain exact PopulationApportionmentCell"
        )
    if type(adapter.cells) is not tuple or any(
        type(cell) is not PopulationProjectionAdapterCell for cell in adapter.cells
    ):
        raise TypeError(
            "adapter cells must contain exact PopulationProjectionAdapterCell"
        )
    for cell in adapter.cells:
        if type(cell.projection_cell) is not PopulationProjectionCell:
            raise TypeError(
                "adapter projection cells must be exact PopulationProjectionCell"
            )
        if type(cell.projection_cell.global_mass) is not tuple:
            raise TypeError(
                "adapter projection global_mass must be an exact tuple"
            )
        for value, name in (
            (cell.analysis_weight, "adapter analysis_weight"),
            (cell.expansion_weight, "adapter expansion_weight"),
        ):
            if type(value) is not tuple or len(value) != 2:
                raise TypeError(f"{name} must be an exact rational tuple")
            _fraction(value[0], value[1], name=name, nonnegative=True)

    _revalidate_plan(plan)


def _revalidate_plan(plan: PopulationApportionmentPlan) -> None:
    try:
        for calibration_cell in plan.calibration_target.cells:
            PopulationCalibrationCell.__post_init__(calibration_cell)
            _fraction(
                calibration_cell.target_mass_numerator,
                calibration_cell.target_mass_denominator,
                name="calibration target mass",
                nonnegative=True,
            )
            _fraction(
                calibration_cell.target_population_numerator,
                calibration_cell.target_population_denominator,
                name="calibration target population",
                nonnegative=True,
            )
        PopulationCalibrationTarget.__post_init__(plan.calibration_target)
        for cell in plan.cells:
            PopulationApportionmentCell.__post_init__(cell)
            _fraction(
                cell.analysis_weight_numerator,
                cell.analysis_weight_denominator,
                name="apportionment analysis weight",
                nonnegative=True,
            )
            _fraction(
                cell.expansion_weight_numerator,
                cell.expansion_weight_denominator,
                name="apportionment expansion weight",
                nonnegative=True,
            )
        PopulationApportionmentPlan.__post_init__(plan)
    except (TypeError, ValueError) as exc:
        raise PopulationBalanceValidationError(
            "apportionment plan is stale, mutated, invalid, or too large"
        ) from exc


def _revalidate_sidecar(
    players: PlayerTable,
    assignment: ProjectedPopulationAssignment,
) -> None:
    if type(players.jurisdiction_codes) is not tuple or any(
        type(code) is not str for code in players.jurisdiction_codes
    ):
        raise TypeError("jurisdiction_codes must be an exact tuple of strings")
    if type(players.adult_age_by_jurisdiction) is not tuple or any(
        type(age) is not int for age in players.adult_age_by_jurisdiction
    ):
        raise TypeError(
            "adult_age_by_jurisdiction must be an exact tuple of integers"
        )
    for index, adult_age in enumerate(players.adult_age_by_jurisdiction):
        _strict_int(
            adult_age,
            name=f"adult_age_by_jurisdiction[{index}]",
            minimum=0,
            maximum=32_767,
        )
    if len(players.jurisdiction_codes) != len(
        players.adult_age_by_jurisdiction
    ) or len(set(players.jurisdiction_codes)) != len(players.jurisdiction_codes):
        raise PopulationBalanceValidationError(
            "PlayerTable jurisdiction metadata are inconsistent"
        )
    metadata = assignment.metadata
    if type(metadata) is not ProjectedPopulationMetadata:
        raise TypeError("projected metadata must be ProjectedPopulationMetadata")
    if type(metadata.cells) is not tuple or any(
        type(cell) is not ProjectedPopulationCellMetadata for cell in metadata.cells
    ):
        raise TypeError(
            "projected metadata cells must contain exact "
            "ProjectedPopulationCellMetadata"
        )
    for cell in metadata.cells:
        if type(cell.global_mass) is not tuple:
            raise TypeError("projected global_mass must be an exact tuple")
        if type(cell.analysis_weight) is not tuple:
            raise TypeError("projected analysis_weight must be an exact tuple")
        ProjectedPopulationCellMetadata.__post_init__(cell)
        _fraction(
            cell.global_mass[0],
            cell.global_mass[1],
            name="projected global mass",
            nonnegative=True,
        )
        _fraction(
            cell.analysis_weight[0],
            cell.analysis_weight[1],
            name="projected analysis weight",
            nonnegative=True,
        )
    expected_projection_sha256 = projected_population_plan_sha256(
        metadata.projection_id,
        metadata.cells,
    )
    if metadata.projection_sha256 != expected_projection_sha256:
        raise PopulationBalanceValidationError(
            "runtime projection sidecar was mutated after attestation"
        )

    for array, dtype, name in (
        (players.player_id, np.dtype(np.int64), "player_id"),
        (players.jurisdiction, np.dtype(np.int16), "jurisdiction"),
        (players.age_years, np.dtype(np.int16), "age_years"),
        (players.is_minor, np.dtype(np.bool_), "is_minor"),
        (
            players.monthly_disposable_income_cents,
            np.dtype(np.int64),
            "monthly_disposable_income_cents",
        ),
        (players.household_id, np.dtype(np.int64), "household_id"),
        (assignment.cell_index, np.dtype(np.int32), "cell_index"),
    ):
        _exact_vector(array, dtype=dtype, name=name, size=len(players))
    if len(set(int(value) for value in players.player_id)) != len(players):
        raise PopulationBalanceValidationError("player_id values must remain unique")
    if len(players) and np.any(players.household_id < 0):
        raise PopulationBalanceValidationError(
            "household_id values cannot be negative"
        )
    if len(players) and (
        np.any(assignment.cell_index < 0)
        or np.any(assignment.cell_index >= len(metadata.cells))
    ):
        raise PopulationBalanceValidationError(
            "projected cell_index contains an unknown cell"
        )
    expected_assignment_sha256 = projected_population_assignment_sha256(
        metadata,
        players.player_id,
        assignment.cell_index,
    )
    if assignment.assignment_sha256 != expected_assignment_sha256:
        raise PopulationBalanceValidationError(
            "runtime projection assignment was mutated after attestation"
        )


def _require_exact_cell_alignment(
    adapter: PopulationProjectionAdapter,
    assignment: ProjectedPopulationAssignment,
) -> tuple[int, ...]:
    plan = adapter.apportionment_plan
    adapter_cells = adapter.cells
    sidecar_cells = assignment.metadata.cells
    if not (
        len(adapter_cells) == len(plan.cells) == len(sidecar_cells)
    ):
        raise PopulationBalanceValidationError(
            "plan, adapter, and sidecar have missing or extra joint cells"
        )
    expected_ordinals = tuple(range(len(plan.cells)))
    if tuple(cell.cell_ordinal for cell in adapter_cells) != expected_ordinals:
        raise PopulationBalanceValidationError(
            "adapter joint cells are not in exact plan-ordinal order"
        )
    sidecar_index_by_id = {
        cell.cell_id: index for index, cell in enumerate(sidecar_cells)
    }
    adapter_ids = tuple(cell.projection_cell.cell_id for cell in adapter_cells)
    if len(sidecar_index_by_id) != len(sidecar_cells) or set(adapter_ids) != set(
        sidecar_index_by_id
    ):
        raise PopulationBalanceValidationError(
            "adapter and sidecar projected cell identities differ"
        )
    sidecar_indices = tuple(
        sidecar_index_by_id[cell_id] for cell_id in adapter_ids
    )

    for ordinal, (adapter_cell, plan_cell) in enumerate(
        zip(adapter_cells, plan.cells, strict=True)
    ):
        sidecar_cell = sidecar_cells[sidecar_indices[ordinal]]
        calibration = plan_cell.calibration_cell
        projection = adapter_cell.projection_cell
        if adapter_cell.cell_ordinal != ordinal:
            raise PopulationBalanceValidationError(
                "adapter joint-cell ordinal differs from exact plan"
            )
        if (
            adapter_cell.evidence_cell_id != calibration.evidence_cell_id
            or projection.jurisdiction_code != calibration.jurisdiction_code
            or projection.baseline_gamer
            is not (calibration.gaming_state is PopulationGamingState.GAMER)
            or projection.baseline_ever_payer
            is not (
                calibration.payer_history_state
                is PopulationPayerHistoryState.EVER_PAYER
            )
        ):
            raise PopulationBalanceValidationError(
                "adapter joint-cell semantics differ from exact calibration cell"
            )
        expected_sidecar = (
            projection.cell_id,
            projection.jurisdiction_code,
            projection.age_min_inclusive,
            projection.age_max_exclusive,
            projection.monthly_disposable_income_band_id,
            projection.monthly_disposable_income_min_cents,
            projection.monthly_disposable_income_max_cents_exclusive,
            projection.household_type,
            projection.modeled_players_per_household,
            projection.baseline_gamer,
            projection.baseline_ever_payer,
            projection.global_mass,
            adapter_cell.analysis_weight,
        )
        observed_sidecar = (
            sidecar_cell.cell_id,
            sidecar_cell.jurisdiction_code,
            sidecar_cell.age_min_inclusive,
            sidecar_cell.age_max_exclusive,
            sidecar_cell.monthly_disposable_income_band_id,
            sidecar_cell.monthly_disposable_income_min_cents,
            sidecar_cell.monthly_disposable_income_max_cents_exclusive,
            sidecar_cell.household_type,
            sidecar_cell.modeled_players_per_household,
            sidecar_cell.baseline_gamer,
            sidecar_cell.baseline_ever_payer,
            sidecar_cell.global_mass,
            sidecar_cell.analysis_weight,
        )
        if observed_sidecar != expected_sidecar:
            raise PopulationBalanceValidationError(
                "sidecar joint-cell metadata differs from the exact adapter"
            )
        if (
            adapter_cell.sample_count != plan_cell.sample_count
            or adapter_cell.analysis_weight
            != (
                plan_cell.analysis_weight_numerator,
                plan_cell.analysis_weight_denominator,
            )
            or adapter_cell.expansion_weight
            != (
                plan_cell.expansion_weight_numerator,
                plan_cell.expansion_weight_denominator,
            )
            or Fraction(*projection.global_mass) != calibration.target_mass
        ):
            raise PopulationBalanceValidationError(
                "adapter count, weight, or mass differs from exact apportionment plan"
            )
    return sidecar_indices


def _realized_counts(
    assignment: ProjectedPopulationAssignment,
    cell_count: int,
) -> tuple[int, ...]:
    counts = [0] * cell_count
    for raw_index in assignment.cell_index:
        index = int(raw_index)
        if not 0 <= index < cell_count:
            raise PopulationBalanceValidationError(
                "sidecar assignment references an unknown joint cell"
            )
        counts[index] += 1
    return tuple(counts)


def _build_cell_result(
    adapter_cell: PopulationProjectionAdapterCell,
    plan_cell: PopulationApportionmentCell,
    sidecar_cell: ProjectedPopulationCellMetadata,
    realized_count: int,
) -> PopulationBalanceCellResult:
    calibration = plan_cell.calibration_cell
    projection = adapter_cell.projection_cell
    target_mass = calibration.target_mass
    sidecar_mass = Fraction(*sidecar_cell.global_mass)
    weight = Fraction(*sidecar_cell.analysis_weight)
    realized_mass = weight * realized_count
    declared_discrepancy = sidecar_mass - target_mass
    realized_discrepancy = realized_mass - target_mass
    return PopulationBalanceCellResult(
        cell_ordinal=calibration.cell_ordinal,
        evidence_cell_id=calibration.evidence_cell_id,
        jurisdiction_code=calibration.jurisdiction_code,
        age_band_id=calibration.age_band_id,
        income_band_id=calibration.income_band_id,
        household_type_id=calibration.household_type_id,
        gaming_state=calibration.gaming_state,
        payer_history_state=calibration.payer_history_state,
        projected_cell_id=projection.cell_id,
        mapping_entry_sha256=adapter_cell.mapping_entry_sha256,
        runtime_age_min_inclusive=projection.age_min_inclusive,
        runtime_age_max_exclusive=projection.age_max_exclusive,
        runtime_income_band_id=projection.monthly_disposable_income_band_id,
        runtime_income_min_cents=(
            projection.monthly_disposable_income_min_cents
        ),
        runtime_income_max_cents_exclusive=(
            projection.monthly_disposable_income_max_cents_exclusive
        ),
        runtime_household_type=projection.household_type,
        modeled_players_per_household=projection.modeled_players_per_household,
        planned_sample_count=plan_cell.sample_count,
        realized_sample_count=realized_count,
        sample_count_discrepancy=realized_count - plan_cell.sample_count,
        target_mass_numerator=target_mass.numerator,
        target_mass_denominator=target_mass.denominator,
        sidecar_mass_numerator=sidecar_mass.numerator,
        sidecar_mass_denominator=sidecar_mass.denominator,
        declared_mass_discrepancy_numerator=declared_discrepancy.numerator,
        declared_mass_discrepancy_denominator=declared_discrepancy.denominator,
        analysis_weight_numerator=weight.numerator,
        analysis_weight_denominator=weight.denominator,
        realized_mass_numerator=realized_mass.numerator,
        realized_mass_denominator=realized_mass.denominator,
        realized_mass_discrepancy_numerator=realized_discrepancy.numerator,
        realized_mass_discrepancy_denominator=realized_discrepancy.denominator,
    )


def _attest_runtime_membership(
    players: PlayerTable,
    assignment: ProjectedPopulationAssignment,
) -> PopulationRuntimeMembershipAttestation:
    cells = assignment.metadata.cells
    n_players = len(players)
    player_cell = assignment.cell_index

    expected_jurisdiction = np.fromiter(
        (cells[int(index)].jurisdiction_index for index in player_cell),
        dtype=np.int16,
        count=n_players,
    )
    if not np.array_equal(players.jurisdiction, expected_jurisdiction):
        raise PopulationBalanceValidationError(
            "runtime jurisdiction membership differs from projected joint cells"
        )

    age_minimum = np.fromiter(
        (cells[int(index)].age_min_inclusive for index in player_cell),
        dtype=np.int16,
        count=n_players,
    )
    age_maximum = np.fromiter(
        (cells[int(index)].age_max_exclusive for index in player_cell),
        dtype=np.int32,
        count=n_players,
    )
    if np.any(players.age_years < age_minimum) or np.any(
        players.age_years.astype(np.int32, copy=False) >= age_maximum
    ):
        raise PopulationBalanceValidationError(
            "runtime age membership falls outside a projected joint cell"
        )
    adult_ages = np.asarray(
        players.adult_age_by_jurisdiction,
        dtype=np.int16,
    )
    expected_minor = players.age_years < adult_ages[players.jurisdiction]
    if not np.array_equal(players.is_minor, expected_minor):
        raise PopulationBalanceValidationError(
            "runtime age/minor membership is internally inconsistent"
        )

    income_minimum = np.fromiter(
        (
            cells[int(index)].monthly_disposable_income_min_cents
            for index in player_cell
        ),
        dtype=np.int64,
        count=n_players,
    )
    income_maximum = np.fromiter(
        (
            cells[int(index)].monthly_disposable_income_max_cents_exclusive
            for index in player_cell
        ),
        dtype=np.int64,
        count=n_players,
    )
    if np.any(players.monthly_disposable_income_cents < income_minimum) or np.any(
        players.monthly_disposable_income_cents >= income_maximum
    ):
        raise PopulationBalanceValidationError(
            "runtime income membership falls outside a projected joint cell"
        )

    household_count, partial_count = _check_household_membership(
        players,
        assignment,
    )
    jurisdiction_digest = _array_digest(
        b"microtx-sim.population-balance.jurisdiction.v1\0",
        (
            ("player_id", players.player_id, np.dtype("<i8")),
            ("cell_index", player_cell, np.dtype("<i4")),
            ("jurisdiction", players.jurisdiction, np.dtype("<i2")),
        ),
        text_values=players.jurisdiction_codes,
    )
    age_digest = _array_digest(
        b"microtx-sim.population-balance.age.v1\0",
        (
            ("player_id", players.player_id, np.dtype("<i8")),
            ("cell_index", player_cell, np.dtype("<i4")),
            ("age_years", players.age_years, np.dtype("<i2")),
            ("is_minor", players.is_minor, np.dtype("u1")),
            ("adult_ages", adult_ages, np.dtype("<i2")),
        ),
    )
    income_digest = _array_digest(
        b"microtx-sim.population-balance.income.v1\0",
        (
            ("player_id", players.player_id, np.dtype("<i8")),
            ("cell_index", player_cell, np.dtype("<i4")),
            (
                "monthly_disposable_income_cents",
                players.monthly_disposable_income_cents,
                np.dtype("<i8"),
            ),
        ),
    )
    household_digest = _array_digest(
        b"microtx-sim.population-balance.household.v1\0",
        (
            ("player_id", players.player_id, np.dtype("<i8")),
            ("cell_index", player_cell, np.dtype("<i4")),
            ("household_id", players.household_id, np.dtype("<i8")),
        ),
    )
    payload = {
        "schema_version": POPULATION_BALANCE_SCHEMA_VERSION,
        "player_count": n_players,
        "player_count_decimal": str(n_players),
        "runtime_projection_sha256": assignment.metadata.projection_sha256,
        "assignment_sha256": assignment.assignment_sha256,
        "ordered_player_ids_sha256": (
            population_projection_ordered_player_ids_sha256(players.player_id)
        ),
        "household_count": household_count,
        "household_count_decimal": str(household_count),
        "partial_household_count": partial_count,
        "partial_household_count_decimal": str(partial_count),
        "jurisdiction": {
            "members_checked": n_players,
            "passed": True,
            "values_sha256": jurisdiction_digest,
        },
        "age": {
            "members_checked": n_players,
            "passed": True,
            "values_sha256": age_digest,
        },
        "income": {
            "members_checked": n_players,
            "passed": True,
            "values_sha256": income_digest,
        },
        "household": {
            "members_checked": n_players,
            "passed": True,
            "values_sha256": household_digest,
        },
    }
    return PopulationRuntimeMembershipAttestation(
        player_count=n_players,
        runtime_projection_sha256=assignment.metadata.projection_sha256,
        assignment_sha256=assignment.assignment_sha256,
        ordered_player_ids_sha256=(
            population_projection_ordered_player_ids_sha256(players.player_id)
        ),
        household_count=household_count,
        partial_household_count=partial_count,
        jurisdiction_members_checked=n_players,
        age_members_checked=n_players,
        income_members_checked=n_players,
        household_members_checked=n_players,
        jurisdiction_membership_passed=True,
        age_membership_passed=True,
        income_membership_passed=True,
        household_membership_passed=True,
        jurisdiction_membership_sha256=jurisdiction_digest,
        age_membership_sha256=age_digest,
        income_membership_sha256=income_digest,
        household_membership_sha256=household_digest,
        membership_sha256=_canonical_sha256(
            b"microtx-sim.population-runtime-membership.v1\0",
            payload,
        ),
    )


def _check_household_membership(
    players: PlayerTable,
    assignment: ProjectedPopulationAssignment,
) -> tuple[int, int]:
    if not len(players):
        return 0, 0
    cells = assignment.metadata.cells
    group_keys = tuple(
        (
            cell.jurisdiction_code,
            cell.monthly_disposable_income_band_id,
            cell.household_type,
        )
        for cell in cells
    )
    group_specifications: dict[tuple[str, str, str], tuple[int, int, int]] = {}
    for cell, key in zip(cells, group_keys, strict=True):
        specification = (
            cell.modeled_players_per_household,
            cell.monthly_disposable_income_min_cents,
            cell.monthly_disposable_income_max_cents_exclusive,
        )
        previous = group_specifications.setdefault(key, specification)
        if previous != specification:
            raise PopulationBalanceValidationError(
                "runtime household group has conflicting interval or capacity assumptions"
            )
    canonical_keys = tuple(sorted(group_specifications))
    group_ordinal = {key: ordinal for ordinal, key in enumerate(canonical_keys)}
    cell_group = np.asarray(
        [group_ordinal[key] for key in group_keys],
        dtype=np.int32,
    )
    player_group = cell_group[assignment.cell_index]
    order = np.lexsort((players.player_id, players.household_id))
    ordered_household = players.household_id[order]
    ordered_group = player_group[order]
    same_household = ordered_household[1:] == ordered_household[:-1]
    if np.any(
        ordered_group[1:][same_household]
        != ordered_group[:-1][same_household]
    ):
        raise PopulationBalanceValidationError(
            "runtime household crosses jurisdiction, income-band, or household-type groups"
        )
    household_start = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(~same_household).astype(np.int64) + 1,
        )
    )
    household_end = np.concatenate(
        (
            household_start[1:],
            np.asarray([len(players)], dtype=np.int64),
        )
    )
    household_sizes = household_end - household_start
    household_groups = ordered_group[household_start]
    capacity_by_group = np.asarray(
        [group_specifications[key][0] for key in canonical_keys],
        dtype=np.int64,
    )
    allowed_sizes = capacity_by_group[household_groups]
    if np.any(household_sizes <= 0) or np.any(household_sizes > allowed_sizes):
        raise PopulationBalanceValidationError(
            "runtime household exceeds its mapped modeled-player capacity"
        )
    partial_groups = household_groups[household_sizes < allowed_sizes]
    if partial_groups.size:
        partial_by_group = np.bincount(
            partial_groups.astype(np.int64, copy=False),
            minlength=len(canonical_keys),
        )
        if np.any(partial_by_group > 1):
            raise PopulationBalanceValidationError(
                "runtime household group has more than one partial household"
            )
    return int(household_start.size), int(partial_groups.size)


def _exact_vector(
    value: object,
    *,
    dtype: np.dtype[object],
    name: str,
    size: int,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be an exact NumPy array")
    if value.ndim != 1 or value.shape != (size,) or value.dtype != dtype:
        raise TypeError(f"{name} must have shape ({size},) and dtype {dtype}")
    return value


def _array_digest(
    domain: bytes,
    arrays: tuple[tuple[str, np.ndarray, np.dtype[object]], ...],
    *,
    text_values: tuple[str, ...] = (),
) -> str:
    digest = sha256(domain)
    if type(text_values) is not tuple or any(
        type(value) is not str for value in text_values
    ):
        raise TypeError("array-digest text_values must be an exact string tuple")
    digest.update(len(text_values).to_bytes(8, "little"))
    for value in text_values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    for name, value, wire_dtype in arrays:
        encoded_name = name.encode("ascii")
        digest.update(len(encoded_name).to_bytes(2, "little"))
        digest.update(encoded_name)
        digest.update(value.size.to_bytes(8, "little"))
        digest.update(np.asarray(value, dtype=wire_dtype).tobytes(order="C"))
    return digest.hexdigest()


def _fraction(
    numerator: object,
    denominator: object,
    *,
    name: str,
    nonnegative: bool = False,
) -> Fraction:
    _strict_int(numerator, name=f"{name} numerator")
    _strict_int(denominator, name=f"{name} denominator", minimum=1)
    assert type(numerator) is int and type(denominator) is int
    if nonnegative and numerator < 0:
        raise PopulationBalanceValidationError(f"{name} must be non-negative")
    value = Fraction(numerator, denominator)
    if value.numerator != numerator or value.denominator != denominator:
        raise PopulationBalanceValidationError(
            f"{name} must be a reduced canonical fraction"
        )
    return value


def _fraction_snapshot(value: Fraction) -> dict[str, object]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "numerator_decimal": str(value.numerator),
        "denominator_decimal": str(value.denominator),
    }


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a strict Python integer")
    if value.bit_length() > _MAX_EXACT_INTEGER_BITS:
        raise PopulationBalanceValidationError(
            f"{name} exceeds the {_MAX_EXACT_INTEGER_BITS}-bit exact limit"
        )
    if minimum is not None and value < minimum:
        raise PopulationBalanceValidationError(
            f"{name} must be at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise PopulationBalanceValidationError(
            f"{name} must be at most {maximum}"
        )


def _identifier(value: object, *, name: str) -> None:
    if type(value) is not str or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise PopulationBalanceValidationError(
            f"{name} must be a canonical identifier"
        )


def _nonempty_text(value: object, *, name: str) -> None:
    if type(value) is not str or not value or value.strip() != value:
        raise PopulationBalanceValidationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )


def _jurisdiction_code(value: object) -> None:
    if (
        type(value) is not str
        or len(value) != 2
        or any(character < "A" or character > "Z" for character in value)
    ):
        raise PopulationBalanceValidationError(
            "jurisdiction_code must be a two-letter uppercase ASCII code"
        )


def _sha256(value: object, *, name: str) -> None:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise PopulationBalanceValidationError(
            f"{name} must be lowercase SHA-256 hex"
        )


def _canonical_json(value: object) -> str:
    _validate_snapshot_tree(value, name="canonical JSON value")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(domain: bytes, payload: object) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    digest = sha256(domain)
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    return digest.hexdigest()


def _validate_snapshot_tree(value: object, *, name: str) -> None:
    if value is None or type(value) in (str, bool):
        return
    if type(value) is int:
        _strict_int(value, name=name)
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_snapshot_tree(item, name=f"{name}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{name} keys must be exact strings")
            _validate_snapshot_tree(item, name=f"{name}.{key}")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise PopulationBalanceValidationError(
            f"{name} contains a non-finite number"
        )
    raise TypeError(
        f"{name} must contain only exact JSON primitive/container types"
    )


__all__ = [
    "POPULATION_BALANCE_SCHEMA_VERSION",
    "PopulationBalanceArtifact",
    "PopulationBalanceCellResult",
    "PopulationBalanceValidationError",
    "PopulationRuntimeMembershipAttestation",
    "build_population_balance_artifact",
    "validate_population_balance_snapshot",
]
