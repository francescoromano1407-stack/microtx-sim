"""Opt-in population execution resolution and immutable run lineage.

The static design, runtime mapping, projected assignment, exact design weights,
and realized balance are related but distinct attestations.  This module binds
them at the run boundary without changing legacy population initialization or
promoting any evidence/readiness gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from typing import Sequence

import numpy as np

from ..agents.players import (
    ProjectedPopulationCellMetadata,
    ProjectedPopulationMetadata,
    projected_population_assignment_sha256,
    projected_population_plan_sha256,
)
from ..config import (
    PopulationExecutionMode,
    PopulationProjectionConfig,
)
from ..metrics.population_balance import (
    PopulationBalanceArtifact,
    build_population_balance_artifact,
)
from ..metrics.population_estimands import (
    ExactPopulationWeights,
    exact_population_weights_from_projected_players,
)
from ..rng import validate_seed
from .population_design import (
    apportion_population_hamilton,
    build_population_calibration_target,
    load_and_verify_population_design_bundle,
)
from .population_evidence import (
    PopulationEvidenceBundle,
    verify_population_evidence_bundle,
)
from .population_projection import (
    POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION_V2,
    POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION_V2,
    PopulationProjectionAdapter,
    PopulationProjectionExecution,
    build_population_projection_adapter,
    load_population_runtime_mapping_bundle,
    population_projection_execution_sha256,
    population_projection_ordered_player_ids_sha256,
    verify_population_projection_adapter,
    verify_population_projection_execution,
)
from .profiles import ProfileBundle


POPULATION_EXECUTION_INPUT_SCHEMA_VERSION = "1.0"
POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION = "2.0"
CAMPAIGN_POPULATION_ADAPTER_ID = "campaign.standardized.population.v2"
CAMPAIGN_POPULATION_EVIDENCE_BUNDLE_ID = (
    "illustrative-four-country-joint-population-2024-v1"
)
CAMPAIGN_POPULATION_DESIGN_ID = (
    "standardized-four-country-person-population-2024-v1"
)
CAMPAIGN_POPULATION_RUNTIME_MAPPING_ID = (
    "standardized-four-country-runtime-income-v2"
)

_CAMPAIGN_BLOCKERS = (
    "population.source_authenticity=not_verified",
    "population.heldout_validation=not_independently_verified",
    "population.gaming_payer_labels=sidecar_only",
    "population.output_estimands=explicit_separate_profile_required",
)


class PopulationExecutionValidationError(ValueError):
    """Raised when population execution identities cannot be re-attested."""


def resolve_population_projection_adapter(
    selection: PopulationProjectionConfig,
    profile_bundle: ProfileBundle,
    *,
    player_count: int,
    first_player_id: int = 0,
    campaign: bool = False,
) -> PopulationProjectionAdapter:
    """Resolve one configured adapter against the selected profile evidence.

    No fallback is permitted: selecting projected mode with absent, empty, or
    illustrative inputs fails before any player or scenario is initialized.
    """

    if type(selection) is not PopulationProjectionConfig:
        raise TypeError("selection must be PopulationProjectionConfig")
    if type(campaign) is not bool:
        raise TypeError("campaign must be a strict boolean")
    if selection.mode is not PopulationExecutionMode.PROJECTED_V1:
        raise PopulationExecutionValidationError(
            "unsupported population execution mode"
        )
    if type(profile_bundle) is not ProfileBundle:
        raise TypeError("profile_bundle must be ProfileBundle")
    evidence_bundle = profile_bundle.population_evidence_bundle
    if type(evidence_bundle) is not PopulationEvidenceBundle:
        raise PopulationExecutionValidationError(
            "projected population mode requires registered population evidence"
        )
    evidence_results = verify_population_evidence_bundle(
        evidence_bundle,
        expected_source_registry_sha256=evidence_bundle.source_registry_sha256,
    )
    verification = load_and_verify_population_design_bundle(
        selection.design_bundle_path,
        population_evidence_bundle=evidence_bundle,
        population_evidence_results=evidence_results,
    )
    target = build_population_calibration_target(verification)
    plan = apportion_population_hamilton(
        target,
        player_count,
        first_player_id=first_player_id,
    )
    mapping = load_population_runtime_mapping_bundle(
        selection.runtime_mapping_bundle_path
    )
    adapter = build_population_projection_adapter(
        verification,
        plan,
        mapping,
        adapter_id=selection.adapter_id,
    )
    if campaign:
        return validate_population_campaign_preflight(adapter)
    return adapter


def validate_population_campaign_preflight(
    adapter: PopulationProjectionAdapter,
) -> PopulationProjectionAdapter:
    """Re-attest every population input required before campaign treatment.

    This gate intentionally evaluates the exact verified objects rather than a
    caller-controlled readiness flag.  Schema-v1 evidence/design declarations
    remain incapable of passing because their authenticity, signature, and
    held-out validation contracts are incomplete.
    """

    observed = verify_population_projection_adapter(adapter)
    blockers: list[str] = []
    verification = observed.verification
    try:
        verification.evidence_bundle.validate_for_campaign()
    except ValueError as exc:
        blockers.append(str(exc))
    try:
        verification.bundle.validate_for_campaign()
    except ValueError as exc:
        blockers.append(str(exc))
    expected_codes = {"BE", "JP", "KR", "UK"}
    observed_codes = {
        item.jurisdiction_code for item in verification.bundle.jurisdictions
    }
    if observed_codes != expected_codes:
        blockers.append(
            "population_jurisdictions=" + ",".join(sorted(observed_codes))
        )
    if observed.adapter_id != CAMPAIGN_POPULATION_ADAPTER_ID:
        blockers.append("population_adapter_id=" + observed.adapter_id)
    if not observed.authenticity_verified:
        blockers.append("population_adapter_authenticity=not_verified")
    if not observed.campaign_ready:
        blockers.append("population_adapter_campaign_ready=false")
    if (
        verification.evidence_bundle.bundle_id
        != CAMPAIGN_POPULATION_EVIDENCE_BUNDLE_ID
    ):
        blockers.append(
            "population_evidence_bundle_id="
            + verification.evidence_bundle.bundle_id
        )
    if verification.bundle.design_id != CAMPAIGN_POPULATION_DESIGN_ID:
        blockers.append("population_design_id=" + verification.bundle.design_id)
    mapping = observed.mapping_bundle
    if mapping.mapping_id != CAMPAIGN_POPULATION_RUNTIME_MAPPING_ID:
        blockers.append("population_runtime_mapping_id=" + mapping.mapping_id)
    evidence_sources_by_jurisdiction: dict[str, set[str]] = {}
    for binding in verification.evidence_bundle.bindings:
        evidence_sources_by_jurisdiction.setdefault(
            binding.jurisdiction_code,
            set(),
        ).update(binding.source_ids)
    unbound_mapping_sources = sorted(
        {
            (
                entry.jurisdiction_code,
                entry.income_model.source_id
                if entry.income_model is not None
                else "missing-income-model",
            )
            for entry in mapping.entries
            if entry.income_model is None
            or entry.income_model.source_id
            not in evidence_sources_by_jurisdiction.get(
                entry.jurisdiction_code,
                set(),
            )
        }
    )
    if unbound_mapping_sources:
        blockers.append(
            "population_runtime_mapping_sources_unbound="
            + ",".join(
                f"{code}:{source_id}"
                for code, source_id in unbound_mapping_sources
            )
        )
    if mapping.schema_version != POPULATION_RUNTIME_MAPPING_SCHEMA_VERSION_V2:
        blockers.append(
            "population_runtime_mapping_schema_version="
            + str(mapping.schema_version)
        )
    if observed.schema_version != POPULATION_PROJECTION_ADAPTER_SCHEMA_VERSION_V2:
        blockers.append(
            "population_adapter_schema_version=" + str(observed.schema_version)
        )
    if not mapping.entries:
        blockers.append("population_runtime_mapping_empty")
    if observed.apportionment_plan.player_count <= 0:
        blockers.append("population_projected_cohort_empty")
    if blockers:
        raise PopulationExecutionValidationError(
            "campaign population preflight failed before treatment: "
            + " | ".join(blockers)
        )
    return observed


def population_execution_input_snapshot(
    adapter: PopulationProjectionAdapter,
) -> dict[str, object]:
    """Return the exact opt-in initializer identity for a run-input digest."""

    observed = verify_population_projection_adapter(adapter)
    return {
        "schema_version": POPULATION_EXECUTION_INPUT_SCHEMA_VERSION,
        "mode": PopulationExecutionMode.PROJECTED_V1.value,
        "adapter": observed.snapshot(),
        "campaign_ready": False,
        "campaign_blockers": list(_CAMPAIGN_BLOCKERS),
    }


def population_execution_input_sha256(
    adapter: PopulationProjectionAdapter,
) -> str:
    return _canonical_sha256(population_execution_input_snapshot(adapter))


def population_policy_pretreatment_sha256(
    *,
    policy_days: int,
    player_ids: np.ndarray,
    is_minor: np.ndarray,
    age_years: np.ndarray,
    jurisdiction: np.ndarray,
    baseline_vulnerability: np.ndarray,
    disposable_budget_cents: np.ndarray,
) -> str:
    """Hash the exact policy-facing pre-treatment arrays for one cohort.

    The helper intentionally accepts arrays rather than a policy-result object,
    keeping this data-lineage module independent of the policy orchestrator.
    """

    selected_days = _policy_days(policy_days)
    contracts = (
        ("player_ids", player_ids, np.dtype(np.int64)),
        ("is_minor", is_minor, np.dtype(np.bool_)),
        ("age_years", age_years, np.dtype(np.int16)),
        ("jurisdiction", jurisdiction, np.dtype(np.int16)),
        (
            "baseline_vulnerability",
            baseline_vulnerability,
            np.dtype(np.float32),
        ),
        (
            "disposable_budget_cents",
            disposable_budget_cents,
            np.dtype(np.int64),
        ),
    )
    expected_size: int | None = None
    digest = sha256(b"microtx-sim.population-policy-pretreatment.v1\0")
    days_bytes = str(selected_days).encode("ascii")
    digest.update(len(days_bytes).to_bytes(8, "little", signed=False))
    digest.update(days_bytes)
    for name, values, expected_dtype in contracts:
        if type(values) is not np.ndarray:
            raise TypeError(f"{name} must be an exact NumPy array")
        if values.ndim != 1 or values.dtype != expected_dtype:
            raise TypeError(
                f"{name} must be a one-dimensional {expected_dtype.name} array"
            )
        if expected_size is None:
            expected_size = int(values.size)
        elif values.size != expected_size:
            raise ValueError("policy pre-treatment arrays must have equal size")
        name_bytes = name.encode("ascii")
        digest.update(len(name_bytes).to_bytes(8, "little", signed=False))
        digest.update(name_bytes)
        digest.update(values.size.to_bytes(8, "little", signed=False))
        little_endian = expected_dtype.newbyteorder("<")
        digest.update(np.asarray(values, dtype=little_endian).tobytes(order="C"))
    return digest.hexdigest()


def _policy_days(value: object) -> int:
    if type(value) is not int:
        raise TypeError("policy_days must be an exact integer")
    if value < 0:
        raise ValueError("policy_days cannot be negative")
    return value


def _policy_disposable_budget_cents(
    monthly_disposable_income_cents: np.ndarray,
    policy_days: int,
) -> np.ndarray:
    if (
        type(monthly_disposable_income_cents) is not np.ndarray
        or monthly_disposable_income_cents.ndim != 1
        or monthly_disposable_income_cents.dtype != np.dtype(np.int64)
    ):
        raise TypeError(
            "monthly_disposable_income_cents must be a one-dimensional int64 array"
        )
    months = max(1, (_policy_days(policy_days) + 29) // 30)
    maximum = np.iinfo(np.int64).max
    if months and np.any(monthly_disposable_income_cents > maximum // months):
        raise OverflowError("disposable budget would overflow int64")
    return (monthly_disposable_income_cents * months).astype(np.int64)


@dataclass(frozen=True, slots=True)
class PopulationSeedExecutionRecord:
    """Detached exact population execution truth for one simulation seed."""

    seed: int
    initialization_tick: int
    policy_days: int
    policy_pretreatment_sha256: str
    cohort_digest: str
    projection_execution_sha256: str
    adapter_sha256: str
    runtime_projection_sha256: str
    assignment_sha256: str
    ordered_player_ids_sha256: str
    jurisdiction_codes: tuple[str, ...]
    cell_indices: tuple[int, ...]
    exact_weights: ExactPopulationWeights
    balance: PopulationBalanceArtifact
    seed_record_sha256: str

    def __post_init__(self) -> None:
        validate_seed(self.seed, name="population execution seed")
        validate_seed(
            self.initialization_tick,
            name="population initialization tick",
        )
        _policy_days(self.policy_days)
        for value, name in (
            (
                self.policy_pretreatment_sha256,
                "policy_pretreatment_sha256",
            ),
            (self.cohort_digest, "cohort_digest"),
            (
                self.projection_execution_sha256,
                "projection_execution_sha256",
            ),
            (self.adapter_sha256, "adapter_sha256"),
            (self.runtime_projection_sha256, "runtime_projection_sha256"),
            (self.assignment_sha256, "assignment_sha256"),
            (self.ordered_player_ids_sha256, "ordered_player_ids_sha256"),
            (self.seed_record_sha256, "seed_record_sha256"),
        ):
            _sha256(value, name=name)
        if type(self.exact_weights) is not ExactPopulationWeights:
            raise TypeError("exact_weights must be ExactPopulationWeights")
        if type(self.balance) is not PopulationBalanceArtifact:
            raise TypeError("balance must be PopulationBalanceArtifact")
        if type(self.cell_indices) is not tuple or any(
            type(index) is not int for index in self.cell_indices
        ):
            raise TypeError("cell_indices must be an immutable tuple of exact integers")
        if type(self.jurisdiction_codes) is not tuple or any(
            type(code) is not str for code in self.jurisdiction_codes
        ):
            raise TypeError("jurisdiction_codes must be an immutable tuple of strings")
        # Explicit class calls do not trust a subclass override and repeat the
        # nested content-address validation if a caller used object.__setattr__.
        ExactPopulationWeights.__post_init__(self.exact_weights)
        PopulationBalanceArtifact.__post_init__(self.balance)
        balance_codes = {cell.jurisdiction_code for cell in self.balance.cells}
        if (
            not self.jurisdiction_codes
            or len(set(self.jurisdiction_codes)) != len(self.jurisdiction_codes)
            or set(self.jurisdiction_codes) != balance_codes
        ):
            raise PopulationExecutionValidationError(
                "seed jurisdiction codes must be unique and exactly match the "
                "balanced jurisdiction scope"
            )
        if self.exact_weights.weight_sum != Fraction(1, 1):
            raise PopulationExecutionValidationError(
                "projected exact design weights must sum exactly to one"
            )
        if len(self.exact_weights.player_ids) != self.balance.player_count:
            raise PopulationExecutionValidationError(
                "exact weights and balance player counts differ"
            )
        if len(self.cell_indices) != self.balance.player_count:
            raise PopulationExecutionValidationError(
                "cell indices and balance player counts differ"
            )
        if any(
            index < 0 or index >= len(self.balance.cells)
            for index in self.cell_indices
        ):
            raise PopulationExecutionValidationError(
                "cell indices fall outside the balanced cell domain"
            )
        observed_counts = np.bincount(
            np.asarray(self.cell_indices, dtype=np.int64),
            minlength=len(self.balance.cells),
        )
        expected_counts = np.asarray(
            [cell.realized_sample_count for cell in self.balance.cells],
            dtype=np.int64,
        )
        if not np.array_equal(observed_counts, expected_counts):
            raise PopulationExecutionValidationError(
                "cell indices do not reproduce the balanced realized counts"
            )
        player_ids = np.asarray(self.exact_weights.player_ids, dtype=np.int64)
        observed_ids_sha256 = population_projection_ordered_player_ids_sha256(
            player_ids
        )
        if observed_ids_sha256 != self.ordered_player_ids_sha256:
            raise PopulationExecutionValidationError(
                "exact weights do not bind the ordered projected player ids"
            )
        expected_balance_bindings = {
            "execution_sha256": self.projection_execution_sha256,
            "adapter_sha256": self.adapter_sha256,
            "runtime_projection_sha256": self.runtime_projection_sha256,
            "assignment_sha256": self.assignment_sha256,
            "ordered_player_ids_sha256": self.ordered_player_ids_sha256,
        }
        mismatches = sorted(
            name
            for name, expected in expected_balance_bindings.items()
            if getattr(self.balance, name) != expected
        )
        if mismatches:
            raise PopulationExecutionValidationError(
                "balance does not bind the projected seed execution: "
                + ", ".join(mismatches)
            )
        if not self.balance.exact_balance_passed:
            raise PopulationExecutionValidationError(
                "population execution requires an exact passing balance result"
            )
        expected_record_sha256 = _canonical_sha256(self.attestation_payload())
        if self.seed_record_sha256 != expected_record_sha256:
            raise PopulationExecutionValidationError(
                "seed_record_sha256 does not match its exact payload"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "seed_decimal": str(self.seed),
            "initialization_tick": self.initialization_tick,
            "initialization_tick_decimal": str(self.initialization_tick),
            "policy_days": self.policy_days,
            "policy_pretreatment_sha256": self.policy_pretreatment_sha256,
            "cohort_digest": self.cohort_digest,
            "projection_execution_sha256": self.projection_execution_sha256,
            "adapter_sha256": self.adapter_sha256,
            "runtime_projection_sha256": self.runtime_projection_sha256,
            "assignment_sha256": self.assignment_sha256,
            "ordered_player_ids_sha256": self.ordered_player_ids_sha256,
            "jurisdiction_codes": list(self.jurisdiction_codes),
            "cell_indices": list(self.cell_indices),
            "exact_weights_sha256": self.exact_weights.design_sha256,
            "exact_weights": self.exact_weights.snapshot(),
            "balance": self.balance.snapshot(),
            "campaign_ready": False,
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "seed_record_sha256": self.seed_record_sha256,
        }


def build_population_seed_execution_record(
    execution: PopulationProjectionExecution,
    *,
    seed: int,
    cohort_digest: str,
    policy_days: int,
) -> PopulationSeedExecutionRecord:
    """Re-attest and detach projection, weights, and balance for one seed."""

    if type(execution) is not PopulationProjectionExecution:
        raise TypeError("execution must be PopulationProjectionExecution")
    observed = verify_population_projection_execution(execution)
    selected_seed = validate_seed(seed, name="population execution seed")
    if selected_seed != observed.initialization_seed:
        raise PopulationExecutionValidationError(
            "population execution seed differs from its initialization seed"
        )
    selected_days = _policy_days(policy_days)
    disposable_budget_cents = _policy_disposable_budget_cents(
        observed.players.monthly_disposable_income_cents,
        selected_days,
    )
    policy_pretreatment_sha256 = population_policy_pretreatment_sha256(
        policy_days=selected_days,
        player_ids=observed.players.player_id,
        is_minor=observed.players.is_minor,
        age_years=observed.players.age_years,
        jurisdiction=observed.players.jurisdiction,
        baseline_vulnerability=observed.players.baseline_vulnerability,
        disposable_budget_cents=disposable_budget_cents,
    )
    weights = exact_population_weights_from_projected_players(observed.players)
    balance = build_population_balance_artifact(observed)
    assignment = observed.players.projected_population
    assert assignment is not None
    cell_indices = tuple(int(index) for index in assignment.cell_index)
    payload = {
        "seed": selected_seed,
        "seed_decimal": str(selected_seed),
        "initialization_tick": observed.initialization_tick,
        "initialization_tick_decimal": str(observed.initialization_tick),
        "policy_days": selected_days,
        "policy_pretreatment_sha256": policy_pretreatment_sha256,
        "cohort_digest": cohort_digest,
        "projection_execution_sha256": observed.execution_sha256,
        "adapter_sha256": observed.adapter.adapter_sha256,
        "runtime_projection_sha256": observed.runtime_projection_sha256,
        "assignment_sha256": observed.assignment_sha256,
        "ordered_player_ids_sha256": observed.ordered_player_ids_sha256,
        "jurisdiction_codes": list(observed.players.jurisdiction_codes),
        "cell_indices": list(cell_indices),
        "exact_weights_sha256": weights.design_sha256,
        "exact_weights": weights.snapshot(),
        "balance": balance.snapshot(),
        "campaign_ready": False,
    }
    _sha256(cohort_digest, name="cohort_digest")
    return PopulationSeedExecutionRecord(
        seed=selected_seed,
        initialization_tick=observed.initialization_tick,
        policy_days=selected_days,
        policy_pretreatment_sha256=policy_pretreatment_sha256,
        cohort_digest=cohort_digest,
        projection_execution_sha256=observed.execution_sha256,
        adapter_sha256=observed.adapter.adapter_sha256,
        runtime_projection_sha256=observed.runtime_projection_sha256,
        assignment_sha256=observed.assignment_sha256,
        ordered_player_ids_sha256=observed.ordered_player_ids_sha256,
        jurisdiction_codes=observed.players.jurisdiction_codes,
        cell_indices=cell_indices,
        exact_weights=weights,
        balance=balance,
        seed_record_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class PopulationExecutionLineage:
    """Content-addressed opt-in population input plus per-seed executions."""

    schema_version: str
    mode: PopulationExecutionMode
    adapter: PopulationProjectionAdapter
    seed_records: tuple[PopulationSeedExecutionRecord, ...]
    lineage_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION:
            raise PopulationExecutionValidationError(
                "unsupported population execution-lineage schema version"
            )
        if type(self.mode) is not PopulationExecutionMode or (
            self.mode is not PopulationExecutionMode.PROJECTED_V1
        ):
            raise PopulationExecutionValidationError(
                "population execution lineage requires projected_v1 mode"
            )
        if type(self.adapter) is not PopulationProjectionAdapter:
            raise TypeError("lineage adapter must be PopulationProjectionAdapter")
        adapter = verify_population_projection_adapter(self.adapter)
        if type(self.seed_records) is not tuple or any(
            type(record) is not PopulationSeedExecutionRecord
            for record in self.seed_records
        ):
            raise TypeError(
                "seed_records must be an immutable tuple of exact records"
            )
        if not self.seed_records:
            raise PopulationExecutionValidationError(
                "population execution lineage requires at least one seed record"
            )
        if self.seed_records != tuple(
            sorted(self.seed_records, key=lambda record: record.seed)
        ):
            raise PopulationExecutionValidationError(
                "population seed records must use ascending seed order"
            )
        seeds = tuple(record.seed for record in self.seed_records)
        if len(set(seeds)) != len(seeds):
            raise PopulationExecutionValidationError(
                "population seed records cannot repeat seeds"
            )
        plan = adapter.apportionment_plan
        expected_ids = tuple(range(plan.first_player_id, plan.last_player_id_exclusive))
        for record in self.seed_records:
            PopulationSeedExecutionRecord.__post_init__(record)
            if record.adapter_sha256 != adapter.adapter_sha256:
                raise PopulationExecutionValidationError(
                    "population seed record uses a different projection adapter"
                )
            if record.exact_weights.player_ids != expected_ids:
                raise PopulationExecutionValidationError(
                    "population seed weights differ from the apportionment id interval"
                )
            _validate_seed_assignment_against_adapter(record, adapter)
            bindings = {
                "mapping_sha256": adapter.mapping_sha256,
                "apportionment_sha256": adapter.apportionment_sha256,
                "calibration_target_sha256": adapter.calibration_target_sha256,
                "design_bundle_sha256": plan.design_bundle_sha256,
                "domain_sha256": plan.domain_sha256,
                "runtime_projection_id": adapter.runtime_projection_id,
            }
            mismatches = sorted(
                name
                for name, expected in bindings.items()
                if getattr(record.balance, name) != expected
            )
            if mismatches:
                raise PopulationExecutionValidationError(
                    "population seed balance differs from lineage input: "
                    + ", ".join(mismatches)
                )
        _sha256(self.lineage_sha256, name="lineage_sha256")
        if self.lineage_sha256 != _canonical_sha256(self.attestation_payload()):
            raise PopulationExecutionValidationError(
                "lineage_sha256 does not match its exact execution payload"
            )

    @property
    def input_sha256(self) -> str:
        return population_execution_input_sha256(self.adapter)

    @property
    def campaign_ready(self) -> bool:
        return False

    @property
    def public_population_comparability(self) -> bool:
        return False

    def record_for_seed(self, seed: int) -> PopulationSeedExecutionRecord:
        selected = validate_seed(seed, name="population execution seed")
        for record in self.seed_records:
            if record.seed == selected:
                return record
        raise KeyError(f"population execution has no seed {selected}")

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "input_sha256": population_execution_input_sha256(self.adapter),
            "input": population_execution_input_snapshot(self.adapter),
            "seeds": [record.seed for record in self.seed_records],
            "seed_decimal_strings": [
                str(record.seed) for record in self.seed_records
            ],
            "seed_records": [record.snapshot() for record in self.seed_records],
            "campaign_ready": False,
            "public_population_comparability": False,
            "campaign_blockers": list(_CAMPAIGN_BLOCKERS),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "lineage_sha256": self.lineage_sha256,
        }

    def manifest_payload(self) -> dict[str, object]:
        """Re-attest bound files and return a detached manifest payload."""

        PopulationExecutionLineage.__post_init__(self)
        return self.snapshot()


def build_population_execution_lineage(
    adapter: PopulationProjectionAdapter,
    seed_records: Sequence[PopulationSeedExecutionRecord],
) -> PopulationExecutionLineage:
    """Canonicalize and content-address per-seed projected executions."""

    observed = verify_population_projection_adapter(adapter)
    if isinstance(seed_records, (str, bytes, bytearray)):
        raise TypeError("seed_records must be a sequence")
    records = tuple(seed_records)
    if any(type(record) is not PopulationSeedExecutionRecord for record in records):
        raise TypeError("seed_records must contain PopulationSeedExecutionRecord")
    records = tuple(sorted(records, key=lambda record: record.seed))
    payload = {
        "schema_version": POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION,
        "mode": PopulationExecutionMode.PROJECTED_V1.value,
        "input_sha256": population_execution_input_sha256(observed),
        "input": population_execution_input_snapshot(observed),
        "seeds": [record.seed for record in records],
        "seed_decimal_strings": [str(record.seed) for record in records],
        "seed_records": [record.snapshot() for record in records],
        "campaign_ready": False,
        "public_population_comparability": False,
        "campaign_blockers": list(_CAMPAIGN_BLOCKERS),
    }
    return PopulationExecutionLineage(
        schema_version=POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION,
        mode=PopulationExecutionMode.PROJECTED_V1,
        adapter=observed,
        seed_records=records,
        lineage_sha256=_canonical_sha256(payload),
    )


def _validate_seed_assignment_against_adapter(
    record: PopulationSeedExecutionRecord,
    adapter: PopulationProjectionAdapter,
) -> None:
    """Reconstruct assignment and weights from detached indices and adapter."""

    expected_execution_sha256 = population_projection_execution_sha256(
        adapter,
        initialization_seed=record.seed,
        initialization_tick=record.initialization_tick,
        runtime_projection_sha256=record.runtime_projection_sha256,
        assignment_sha256=record.assignment_sha256,
        ordered_player_ids_sha256=record.ordered_player_ids_sha256,
    )
    if record.projection_execution_sha256 != expected_execution_sha256:
        raise PopulationExecutionValidationError(
            "population seed/tick do not reproduce the projection execution digest"
        )
    expected_codes = {
        jurisdiction.jurisdiction_code
        for jurisdiction in adapter.verification.bundle.jurisdictions
    }
    if set(record.jurisdiction_codes) != expected_codes:
        raise PopulationExecutionValidationError(
            "population seed jurisdiction scope differs from the adapter"
        )
    code_to_index = {
        code: index for index, code in enumerate(record.jurisdiction_codes)
    }
    metadata_cells: list[ProjectedPopulationCellMetadata] = []
    for adapter_cell in adapter.cells:
        projected = adapter_cell.projection_cell
        metadata_cells.append(
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
                analysis_weight=adapter_cell.analysis_weight,
            )
        )
    cells = tuple(metadata_cells)
    metadata = ProjectedPopulationMetadata(
        projection_id=adapter.runtime_projection_id,
        projection_sha256=projected_population_plan_sha256(
            adapter.runtime_projection_id,
            cells,
        ),
        cells=cells,
    )
    if metadata.projection_sha256 != record.runtime_projection_sha256:
        raise PopulationExecutionValidationError(
            "population seed runtime projection differs from the adapter"
        )
    player_ids = np.asarray(record.exact_weights.player_ids, dtype=np.int64)
    cell_indices = np.asarray(record.cell_indices, dtype=np.int32)
    observed_assignment_sha256 = projected_population_assignment_sha256(
        metadata,
        player_ids,
        cell_indices,
    )
    if observed_assignment_sha256 != record.assignment_sha256:
        raise PopulationExecutionValidationError(
            "population seed cell indices do not reproduce its assignment digest"
        )
    expected_counts = np.asarray(
        [cell.sample_count for cell in adapter.cells],
        dtype=np.int64,
    )
    observed_counts = np.bincount(
        cell_indices.astype(np.int64, copy=False),
        minlength=len(adapter.cells),
    )
    if not np.array_equal(observed_counts, expected_counts):
        raise PopulationExecutionValidationError(
            "population seed cell indices differ from adapter sample counts"
        )
    expected_weights = tuple(
        Fraction(*adapter.cells[index].analysis_weight)
        for index in record.cell_indices
    )
    if record.exact_weights.fractions != expected_weights:
        raise PopulationExecutionValidationError(
            "population seed exact weights differ from assigned adapter cells"
        )


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PopulationExecutionValidationError(
            f"{name} must be lowercase SHA-256 hex"
        )
    return value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "CAMPAIGN_POPULATION_ADAPTER_ID",
    "CAMPAIGN_POPULATION_DESIGN_ID",
    "CAMPAIGN_POPULATION_EVIDENCE_BUNDLE_ID",
    "CAMPAIGN_POPULATION_RUNTIME_MAPPING_ID",
    "POPULATION_EXECUTION_INPUT_SCHEMA_VERSION",
    "POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION",
    "PopulationExecutionLineage",
    "PopulationExecutionValidationError",
    "PopulationSeedExecutionRecord",
    "build_population_execution_lineage",
    "build_population_seed_execution_record",
    "population_execution_input_sha256",
    "population_execution_input_snapshot",
    "population_policy_pretreatment_sha256",
    "resolve_population_projection_adapter",
    "validate_population_campaign_preflight",
]
