"""Run the deterministic UK-adult point-zero initializer audit.

This command initializes the existing projected population and player-life
tables for seed 101.  It does not initialize a scenario, execute a policy day,
or run a campaign.  Comparisons between unlike constructs (for example,
intended play and diary-observed gaming) are reported as diagnostics and never
used as fit gates.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from microtx_sim.agents.players import (
    PlayerTable,
    ProjectedPopulationAssignment,
    ProjectedPopulationSexBinding,
    SOURCE_RECORDED_SEX_DTYPE,
    SOURCE_RECORDED_SEX_FEMALE,
    SOURCE_RECORDED_SEX_MALE,
    SOURCE_RECORDED_SEX_UNAVAILABLE,
    projected_population_assignment_sha256,
    source_recorded_sex_sha256,
)
from microtx_sim.consumers.welfare import PlayerLifeTable, initialize_player_life
from microtx_sim.data.calibration import (
    DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
    CalibrationTarget,
    PopulationWeight,
    UKAdults2024CalibrationBundle,
    load_uk_adults_2024_calibration_bundle,
)
from microtx_sim.data.population_execution import (
    resolve_population_projection_adapter,
)
from microtx_sim.data.population_projection import (
    PopulationProjectionExecution,
    initialize_population_projection,
    verify_population_projection_execution,
)
from microtx_sim.data.profiles import (
    DEFAULT_JURISDICTIONS_PATH,
    ProfileBundle,
    load_profile_bundle,
)
from microtx_sim.data.uk_adults_runtime import (
    UK_ADULTS_2024_SEX_ASSIGNMENT_METHOD,
    UK_ADULTS_2024_SEX_JURISDICTION,
    UK_ADULTS_2024_SEX_MAX_AGE,
    UK_ADULTS_2024_SEX_MIN_AGE,
    bind_uk_adults_2024_source_recorded_sex,
    uk_adults_2024_population_weights_sha256,
    verify_uk_adults_2024_source_recorded_sex,
)
from microtx_sim.policy_config import load_policy_config
from microtx_sim.rng import CounterRNG


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "policy_exploratory_synthetic.toml"
)
AUDIT_SCHEMA_VERSION = 1
AUDIT_ID = "uk-adults-2024-point-zero-seed-101-v1"
INITIALIZATION_SEED = 101
UK_MIN_AGE = 18
UK_MAX_AGE = 64
AGE_TOTAL_VARIATION_MAX = Fraction(1, 50)  # 2 percentage points.

_AGE_BANDS = (
    ("18-24", 18, 24),
    ("25-34", 25, 34),
    ("35-44", 35, 44),
    ("45-54", 45, 54),
    ("55-64", 55, 64),
)
_OUTPUT_QUANTUM = Decimal("0.000001")


class PointZeroAuditError(RuntimeError):
    """Raised when the initializer audit cannot produce a trustworthy report."""


@dataclass(frozen=True, slots=True)
class _InitializerState:
    config: object
    profiles: ProfileBundle
    players: PlayerTable
    life: PlayerLifeTable
    projection_execution: PopulationProjectionExecution
    population_execution_sha256: str
    runtime_projection_sha256: str
    assignment_sha256: str


def _target(bundle: UKAdults2024CalibrationBundle, target_id: str) -> CalibrationTarget:
    try:
        return bundle.target_by_id[target_id]
    except KeyError as exc:
        raise PointZeroAuditError(
            f"verified calibration bundle is missing target {target_id}"
        ) from exc


def _target_value(
    bundle: UKAdults2024CalibrationBundle,
    target_id: str,
) -> Decimal:
    target = _target(bundle, target_id)
    if target.value is None:
        raise PointZeroAuditError(f"target {target_id} is not quantified")
    return target.value


def _decimal_text(value: Decimal | Fraction | int | float) -> str:
    if isinstance(value, Fraction):
        selected = Decimal(value.numerator) / Decimal(value.denominator)
    elif isinstance(value, Decimal):
        selected = value
    elif isinstance(value, int):
        selected = Decimal(value)
    else:
        selected = Decimal.from_float(float(value))
    return format(selected.quantize(_OUTPUT_QUANTUM, rounding=ROUND_HALF_EVEN), "f")


def _mean_int(values: np.ndarray) -> Fraction:
    if values.ndim != 1 or values.size == 0:
        raise PointZeroAuditError("a point-zero mean requires a non-empty vector")
    return Fraction(int(np.sum(values, dtype=np.int64)), int(values.size))


def _load_declared_profiles(config: object) -> ProfileBundle:
    population = getattr(config, "population", None)
    monetary = getattr(config, "monetary_contract", None)
    if population is None:
        raise PointZeroAuditError("point-zero audit requires projected population mode")
    evidence_path = getattr(population, "evidence_bundle_path", None)
    source_registry_path = getattr(population, "source_registry_path", None)
    if evidence_path is None or source_registry_path is None:
        raise PointZeroAuditError(
            "point-zero audit requires declared population evidence and source paths"
        )
    return load_profile_bundle(
        jurisdictions_path=(
            monetary.profile_path
            if monetary is not None
            else DEFAULT_JURISDICTIONS_PATH
        ),
        sources_path=source_registry_path,
        source_bundle_path=(
            monetary.source_bundle_path if monetary is not None else None
        ),
        population_bundle_path=evidence_path,
        campaign=False,
    )


def _initialize_only(
    config_path: Path,
    *,
    calibration_bundle: UKAdults2024CalibrationBundle | None = None,
    repository_root: Path | None = None,
) -> _InitializerState:
    config = load_policy_config(config_path)
    if INITIALIZATION_SEED not in config.batch.seeds:
        raise PointZeroAuditError(
            f"configured exploratory seed set does not contain {INITIALIZATION_SEED}"
        )
    if config.population is None:
        raise PointZeroAuditError("configured population projection is absent")
    profiles = _load_declared_profiles(config)
    adapter = resolve_population_projection_adapter(
        config.population,
        profiles,
        player_count=config.batch.player_count,
        campaign=False,
    )
    population = verify_population_projection_execution(
        initialize_population_projection(
            adapter,
            profiles.country_profiles,
            CounterRNG(INITIALIZATION_SEED),
        )
    )
    if calibration_bundle is not None:
        population = bind_uk_adults_2024_source_recorded_sex(
            population,
            calibration_bundle,
            repository_root=repository_root,
        )
        population = verify_population_projection_execution(population)
    players = population.players
    life = initialize_player_life(players, CounterRNG(INITIALIZATION_SEED))
    life.validate_alignment(players)
    return _InitializerState(
        config=config,
        profiles=profiles,
        players=players,
        life=life,
        projection_execution=population,
        population_execution_sha256=population.execution_sha256,
        runtime_projection_sha256=population.runtime_projection_sha256,
        assignment_sha256=population.assignment_sha256,
    )


def _sidecar_vectors(
    players: PlayerTable,
) -> tuple[
    ProjectedPopulationAssignment,
    np.ndarray,
    tuple[Fraction, ...],
]:
    assignment = players.projected_population
    if type(assignment) is not ProjectedPopulationAssignment:
        raise PointZeroAuditError(
            "point-zero audit requires the projected population assignment sidecar"
        )
    expected_assignment_sha256 = projected_population_assignment_sha256(
        assignment.metadata,
        players.player_id,
        assignment.cell_index,
        age_years=players.age_years,
        jurisdiction=players.jurisdiction,
        sex=players.sex,
        sex_binding=assignment.sex_binding,
    )
    if assignment.assignment_sha256 != expected_assignment_sha256:
        raise PointZeroAuditError(
            "projected population sidecar assignment digest does not verify"
        )
    cells = assignment.metadata.cells
    indices = assignment.cell_index.astype(np.int64, copy=False)
    gamer_by_cell = np.asarray(
        [cell.baseline_gamer for cell in cells],
        dtype=np.bool_,
    )
    exact_weight_by_cell = tuple(
        Fraction(*cell.analysis_weight) for cell in cells
    )
    return assignment, gamer_by_cell[indices], exact_weight_by_cell


def _weighted_share(
    mask: np.ndarray,
    selected: np.ndarray,
    cell_index: np.ndarray,
    exact_weight_by_cell: Sequence[Fraction],
) -> Fraction:
    if mask.dtype != np.dtype(np.bool_) or selected.dtype != np.dtype(np.bool_):
        raise TypeError("weighted-share masks must be boolean arrays")
    if (
        mask.ndim != 1
        or selected.ndim != 1
        or mask.shape != selected.shape
        or cell_index.ndim != 1
        or cell_index.dtype != np.dtype(np.int32)
        or cell_index.shape != selected.shape
    ):
        raise TypeError(
            "weighted-share masks and int32 cell indices must be aligned vectors"
        )
    weights = tuple(exact_weight_by_cell)
    if not weights or any(type(weight) is not Fraction for weight in weights):
        raise TypeError("weighted-share cell weights must be exact Fractions")
    if np.any(cell_index < 0) or np.any(cell_index >= len(weights)):
        raise PointZeroAuditError("weighted-share cell index is out of range")

    def mass(selection: np.ndarray) -> Fraction:
        counts = np.bincount(
            cell_index[selection].astype(np.int64, copy=False),
            minlength=len(weights),
        )
        return sum(
            (
                Fraction(int(count), 1) * weight
                for count, weight in zip(counts, weights, strict=True)
            ),
            start=Fraction(0, 1),
        )

    denominator = mass(selected)
    if denominator <= 0:
        raise PointZeroAuditError("selected point-zero population has no positive mass")
    return mass(selected & mask) / denominator


def _runtime_sex_vector(players: PlayerTable) -> np.ndarray | None:
    """Return the exact source-recorded vector, including out-of-scope blanks."""

    if type(players) is not PlayerTable or "sex" not in {
        descriptor.name for descriptor in fields(PlayerTable)
    }:
        return None

    value = players.sex
    if (
        type(value) is not np.ndarray
        or value.ndim != 1
        or value.shape != (len(players),)
        or value.dtype != SOURCE_RECORDED_SEX_DTYPE
    ):
        return None
    if not np.all(
        np.isin(
            value,
            (
                SOURCE_RECORDED_SEX_FEMALE,
                SOURCE_RECORDED_SEX_MALE,
                SOURCE_RECORDED_SEX_UNAVAILABLE,
            ),
        )
    ):
        return None
    return value


def _attested_runtime_sex_vector(
    players: PlayerTable,
    execution: PopulationProjectionExecution,
    selected: np.ndarray,
) -> tuple[np.ndarray | None, bool, bool, str]:
    """Require both a typed sex column and an execution-bound column digest.

    Projection-execution schema v3 binds both fields. Returning any other
    vector would make an age-by-sex fit appear reproducible when it is not part
    of the content-addressed runtime lineage, so older or mismatched executions
    remain fail-closed.
    """

    vector = _runtime_sex_vector(players)
    field_available = vector is not None
    if vector is None:
        return None, False, False, "UNVERIFIED_FIELD_ABSENT"
    observed = verify_population_projection_execution(execution)
    if observed.players is not players:
        return None, True, False, "UNVERIFIED_EXECUTION_PLAYER_MISMATCH"
    assignment = observed.players.projected_population
    if type(assignment) is not ProjectedPopulationAssignment or type(
        assignment.sex_binding
    ) is not ProjectedPopulationSexBinding:
        return None, True, False, "UNVERIFIED_EXECUTION_BINDING_ABSENT"
    binding = assignment.sex_binding
    payload = observed.attestation_payload()
    if (
        payload.get("sex_sha256") != binding.sex_sha256
        or payload.get("source_recorded_sex") != binding.snapshot()
        or source_recorded_sex_sha256(vector) != binding.sex_sha256
    ):
        return None, True, False, "UNVERIFIED_EXECUTION_BINDING_MISMATCH"
    try:
        jurisdiction_index = players.jurisdiction_codes.index(
            binding.jurisdiction_code
        )
    except ValueError:
        return None, True, False, "UNVERIFIED_EXECUTION_SCOPE_UNKNOWN"
    bound_scope = (
        (players.jurisdiction == jurisdiction_index)
        & (players.age_years >= binding.age_min_inclusive)
        & (players.age_years <= binding.age_max_inclusive)
    )
    if not np.array_equal(bound_scope, selected):
        return None, True, False, "UNVERIFIED_EXECUTION_SCOPE_MISMATCH"
    if np.any(
        ~np.isin(
            vector[selected],
            (SOURCE_RECORDED_SEX_FEMALE, SOURCE_RECORDED_SEX_MALE),
        )
    ):
        return None, True, False, "UNVERIFIED_IN_SCOPE_VALUE_MISSING"
    if np.any(vector[~selected] != SOURCE_RECORDED_SEX_UNAVAILABLE):
        return None, True, False, "UNVERIFIED_OUT_OF_SCOPE_VALUE_PRESENT"
    return vector, True, True, "VERIFIED"


def _calibration_bundle_attestation(
    bundle: object,
    *,
    repository_root: Path | None,
) -> tuple[bool, str]:
    """Reopen an exact bundle before claiming that its bytes were attested."""

    if type(bundle) is not UKAdults2024CalibrationBundle:
        return False, "UNVERIFIED_NOT_STRICT_LOADER_OUTPUT"
    try:
        observed = load_uk_adults_2024_calibration_bundle(
            bundle.bundle_path,
            repository_root=repository_root,
        )
    except Exception:
        return False, "UNVERIFIED_REATTESTATION_FAILED"
    if observed != bundle:
        return False, "UNVERIFIED_REATTESTATION_MISMATCH"
    return True, "VERIFIED"


def _runtime_calibration_binding(
    bundle: object,
    state: _InitializerState,
    *,
    repository_root: Path | None,
) -> tuple[bool, dict[str, object]]:
    """Compare only a typed execution binding; path substrings never qualify.

    Projection-execution schema v3 may bind the UK bundle only for the
    source-recorded sex sidecar. Older executions remain ``UNVERIFIED`` even
    if a path happens to contain the calibration id or digest.
    """

    execution = verify_population_projection_execution(
        state.projection_execution
    )
    payload = execution.attestation_payload()
    assignment = execution.players.projected_population
    binding = (
        assignment.sex_binding
        if type(assignment) is ProjectedPopulationAssignment
        else None
    )
    typed = payload.get("source_recorded_sex")
    required_id = getattr(bundle, "bundle_id", None)
    required_sha256 = getattr(bundle, "bundle_sha256", None)
    expected_source_id: str | None = None
    expected_weights_sha256: str | None = None
    canonical_verification_status = "UNVERIFIED_NOT_STRICT_LOADER_OUTPUT"
    if type(bundle) is UKAdults2024CalibrationBundle:
        source_ids = {row.source_id for row in bundle.population_weights}
        if len(source_ids) == 1:
            expected_source_id = next(iter(source_ids))
        expected_weights_sha256 = uk_adults_2024_population_weights_sha256(
            bundle
        )
        try:
            verified = verify_uk_adults_2024_source_recorded_sex(
                execution,
                bundle,
                repository_root=repository_root,
            )
        except Exception:
            canonical_verification_status = "UNVERIFIED_CANONICAL_RECOMPUTATION_FAILED"
        else:
            canonical_verification_status = (
                "VERIFIED"
                if verified is execution
                else "UNVERIFIED_EXECUTION_IDENTITY_MISMATCH"
            )
    typed_available = bool(
        type(binding) is ProjectedPopulationSexBinding
        and isinstance(typed, dict)
        and typed == binding.snapshot()
        and payload.get("sex_sha256") == binding.sex_sha256
    )
    bound = bool(
        typed_available
        and canonical_verification_status == "VERIFIED"
        and binding.evidence_bundle_id == required_id
        and binding.evidence_bundle_sha256 == required_sha256
        and binding.source_id == expected_source_id
        and binding.population_weights_sha256 == expected_weights_sha256
        and binding.assignment_method == UK_ADULTS_2024_SEX_ASSIGNMENT_METHOD
        and binding.jurisdiction_code == UK_ADULTS_2024_SEX_JURISDICTION
        and binding.age_min_inclusive == UK_ADULTS_2024_SEX_MIN_AGE
        and binding.age_max_inclusive == UK_ADULTS_2024_SEX_MAX_AGE
    )
    runtime_evidence = execution.adapter.verification.evidence_bundle
    return bound, {
        "binding_status": (
            "VERIFIED_MATCH"
            if bound
            else "VERIFIED_MISMATCH"
            if typed_available
            else "UNVERIFIED"
        ),
        "typed_calibration_binding_available": typed_available,
        "required_bundle_id": required_id,
        "required_bundle_sha256": required_sha256,
        "required_source_id": expected_source_id,
        "required_population_weights_sha256": expected_weights_sha256,
        "required_assignment_method": UK_ADULTS_2024_SEX_ASSIGNMENT_METHOD,
        "canonical_recomputation_status": canonical_verification_status,
        "runtime_population_evidence_bundle_id": runtime_evidence.bundle_id,
        "runtime_population_evidence_bundle_sha256": (
            runtime_evidence.bundle_sha256
        ),
        "bound_source_id": (
            binding.source_id if type(binding) is ProjectedPopulationSexBinding else None
        ),
        "bound_population_weights_sha256": (
            binding.population_weights_sha256
            if type(binding) is ProjectedPopulationSexBinding
            else None
        ),
        "bound_assignment_method": (
            binding.assignment_method
            if type(binding) is ProjectedPopulationSexBinding
            else None
        ),
        "limitation": (
            "A verified match proves only the source-recorded UK age-by-sex "
            "sidecar binding. It does not prove that every calibration target or "
            "behavioural weight is consumed by the runtime; matching path text "
            "never qualifies as a binding."
        ),
    }


def _gate(
    gate_id: str,
    category: str,
    passed: bool,
    criterion: str,
    observed: object,
) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "category": category,
        "status": "PASS" if passed else "FAIL",
        "criterion": criterion,
        "observed": observed,
    }


def _incomparable(
    diagnostic_id: str,
    simulator_construct: str,
    evidence_construct: str,
    simulator_value: Fraction,
    evidence_value: Decimal,
    unit: str,
    limitation: str,
) -> dict[str, object]:
    return {
        "diagnostic_id": diagnostic_id,
        "status": "INCOMPARABLE_DIAGNOSTIC",
        "gate_applied": False,
        "simulator_construct": simulator_construct,
        "evidence_construct": evidence_construct,
        "simulator_value": _decimal_text(simulator_value),
        "evidence_value": _decimal_text(evidence_value),
        "unit": unit,
        "descriptive_ratio": _decimal_text(
            simulator_value / Fraction(evidence_value)
        ),
        "limitation": limitation,
    }


def _age_targets(
    weights: Sequence[PopulationWeight],
) -> tuple[dict[str, Fraction], dict[tuple[str, str], Fraction], int]:
    total = sum(weight.population_count for weight in weights)
    if total <= 0:
        raise PointZeroAuditError("calibration age-by-sex population is empty")
    by_band: dict[str, int] = {name: 0 for name, _, _ in _AGE_BANDS}
    by_joint_cell: dict[tuple[str, str], int] = {}
    for weight in weights:
        if weight.age_band not in by_band:
            raise PointZeroAuditError(
                f"unexpected calibration population age band {weight.age_band}"
            )
        by_band[weight.age_band] += weight.population_count
        key = (weight.age_band, weight.sex)
        if key in by_joint_cell:
            raise PointZeroAuditError(
                f"duplicate calibration population cell {weight.age_band}/{weight.sex}"
            )
        by_joint_cell[key] = weight.population_count
    expected_joint_cells = {
        (name, sex)
        for name, _, _ in _AGE_BANDS
        for sex in ("FEMALE", "MALE")
    }
    if set(by_joint_cell) != expected_joint_cells:
        raise PointZeroAuditError(
            "calibration population must contain exactly five age bands by two sexes"
        )
    return (
        {name: Fraction(count, total) for name, count in by_band.items()},
        {key: Fraction(count, total) for key, count in by_joint_cell.items()},
        total,
    )


def _finite_bounded_state(players: PlayerTable, life: PlayerLifeTable) -> bool:
    minute_columns = (
        life.planned_leisure_minutes,
        life.sleep_need_minutes,
        life.work_study_obligation_minutes,
        life.social_obligation_minutes,
        life.physical_activity_need_minutes,
        life.intended_play_minutes,
        life.sleep_debt_minutes,
    )
    continuous_columns = (
        players.baseline_vulnerability,
        life.baseline_game_enjoyment,
        life.financial_sensitivity,
        life.delay_discounting,
        life.social_pressure_susceptibility,
        life.scarcity_fomo_susceptibility,
        life.wellbeing,
    )
    return all(
        np.all(np.isfinite(column))
        and np.all(column >= 0)
        and np.all(column <= 1_440)
        for column in minute_columns
    ) and all(
        np.all(np.isfinite(column))
        and np.all(column >= 0.0)
        and np.all(column <= 1.0)
        for column in continuous_columns
    )


def build_report(
    bundle: UKAdults2024CalibrationBundle,
    state: _InitializerState,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Build the canonical report from initializer state; no treatment is run."""

    players = state.players
    life = state.life
    assignment, sidecar_gamer, exact_weight_by_cell = _sidecar_vectors(players)
    try:
        uk_index = players.jurisdiction_codes.index("UK")
    except ValueError as exc:
        raise PointZeroAuditError(
            "initialized population has no UK jurisdiction"
        ) from exc
    selected = (
        (players.jurisdiction == uk_index)
        & (players.age_years >= UK_MIN_AGE)
        & (players.age_years <= UK_MAX_AGE)
    )
    selected_count = int(np.count_nonzero(selected))
    if selected_count == 0:
        raise PointZeroAuditError("initialized population has no UK adults aged 18-64")

    target_age_shares, target_joint_shares, target_population = _age_targets(
        bundle.population_weights
    )
    age_rows: list[dict[str, object]] = []
    total_variation = Fraction(0, 1)
    maximum_absolute_difference = Fraction(0, 1)
    for name, lower, upper in _AGE_BANDS:
        age_mask = (players.age_years >= lower) & (players.age_years <= upper)
        observed = _weighted_share(
            age_mask,
            selected,
            assignment.cell_index,
            exact_weight_by_cell,
        )
        target = target_age_shares[name]
        difference = observed - target
        absolute = abs(difference)
        total_variation += absolute / 2
        maximum_absolute_difference = max(maximum_absolute_difference, absolute)
        age_rows.append(
            {
                "age_band": name,
                "initialized_weighted_share": _decimal_text(observed),
                "ons_target_share": _decimal_text(target),
                "difference_percentage_points": _decimal_text(difference * 100),
            }
        )

    selected_gamer = selected & sidecar_gamer
    selected_non_gamer = selected & ~sidecar_gamer
    gamer_count = int(np.count_nonzero(selected_gamer))
    non_gamer_count = int(np.count_nonzero(selected_non_gamer))
    if gamer_count == 0 or non_gamer_count == 0:
        raise PointZeroAuditError(
            "point-zero selection must retain sidecar gamers and non-gamers"
        )
    non_gamer_positive_play = Fraction(
        int(np.count_nonzero(life.intended_play_minutes[selected_non_gamer] > 0)),
        non_gamer_count,
    )
    non_gamer_positive_spending_limit = Fraction(
        int(
            np.count_nonzero(
                life.intended_spending_limit_cents[selected_non_gamer] > 0
            )
        ),
        non_gamer_count,
    )
    (
        runtime_sex,
        runtime_sex_field_available,
        runtime_sex_lineage_attested,
        runtime_sex_status,
    ) = _attested_runtime_sex_vector(
        players,
        state.projection_execution,
        selected,
    )
    joint_total_variation: Fraction | None = None
    if runtime_sex is not None:
        joint_total_variation = Fraction(0, 1)
        for name, lower, upper in _AGE_BANDS:
            age_mask = (players.age_years >= lower) & (players.age_years <= upper)
            for sex in ("FEMALE", "MALE"):
                observed = _weighted_share(
                    age_mask & (runtime_sex == sex),
                    selected,
                    assignment.cell_index,
                    exact_weight_by_cell,
                )
                joint_total_variation += abs(
                    observed - target_joint_shares[(name, sex)]
                ) / 2

    purchase_probability = getattr(life, "purchase_probability", None)
    valid_purchase_probability = bool(
        isinstance(purchase_probability, np.ndarray)
        and purchase_probability.ndim == 1
        and purchase_probability.shape == (len(players),)
        and np.all(np.isfinite(purchase_probability))
        and np.all(purchase_probability >= 0.0)
        and np.all(purchase_probability <= 1.0)
    )
    non_gamer_purchase_probability_zero = bool(
        valid_purchase_probability
        and np.all(purchase_probability[selected_non_gamer] == 0.0)
    )

    bundle_attested, bundle_attestation_status = (
        _calibration_bundle_attestation(
            bundle,
            repository_root=repository_root,
        )
    )
    bundle_runtime_bound, runtime_binding_observed = (
        _runtime_calibration_binding(
            bundle,
            state,
            repository_root=repository_root,
        )
    )

    gates = [
        _gate(
            "calibration_bundle_attested",
            "STRUCTURAL",
            bundle_attested,
            (
                "The exact strict-loader output was reopened and all declared "
                "bundle, companion, and source bytes were re-attested."
            ),
            {
                "bundle_id": bundle.bundle_id,
                "bundle_sha256": bundle.bundle_sha256,
                "bundle_status": bundle.status,
                "attestation_status": bundle_attestation_status,
            },
        ),
        _gate(
            "configured_seed_and_cohort",
            "STRUCTURAL",
            INITIALIZATION_SEED in state.config.batch.seeds
            and len(players) == state.config.batch.player_count,
            (
                "Seed 101 is declared and the initializer realizes the "
                "configured cohort size."
            ),
            {
                "seed": INITIALIZATION_SEED,
                "configured_player_count": state.config.batch.player_count,
                "initialized_player_count": len(players),
            },
        ),
        _gate(
            "uk_adult_selection_nonempty",
            "STRUCTURAL",
            selected_count > 0,
            "The projected initializer contains at least one UK resident aged 18-64.",
            {"selected_player_count": selected_count},
        ),
        _gate(
            "calibration_bundle_runtime_binding",
            "STRUCTURAL",
            bundle_runtime_bound,
            (
                "The verified source-recorded sex sidecar binds both the UK-adult "
                "calibration bundle id and digest in the projection execution."
            ),
            runtime_binding_observed,
        ),
        _gate(
            "runtime_sex_state",
            "STRUCTURAL",
            runtime_sex is not None and runtime_sex_lineage_attested,
            (
                "Every in-scope UK resident aged 18-64 has an explicit FEMALE "
                "or MALE runtime sex value bound by the projection-execution "
                "digest."
            ),
            {
                "field_available": runtime_sex_field_available,
                "lineage_attested": runtime_sex_lineage_attested,
                "attestation_status": runtime_sex_status,
                "ons_age_by_sex_cells": len(bundle.population_weights),
            },
        ),
        _gate(
            "ons_age_sex_joint_fit",
            "PLAUSIBILITY",
            joint_total_variation is not None
            and joint_total_variation <= AGE_TOTAL_VARIATION_MAX,
            "Weighted UK 5-by-2 age-sex total-variation distance is at most 0.020000.",
            {
                "runtime_sex_available": runtime_sex is not None,
                "runtime_sex_lineage_attested": runtime_sex_lineage_attested,
                "total_variation_distance": (
                    _decimal_text(joint_total_variation)
                    if joint_total_variation is not None
                    else None
                ),
                "threshold_role": (
                    "predeclared engineering plausibility threshold, not an "
                    "inferential confidence bound"
                ),
            },
        ),
        _gate(
            "ons_age_marginal_fit",
            "PLAUSIBILITY",
            total_variation <= AGE_TOTAL_VARIATION_MAX,
            "Weighted UK age-margin total-variation distance is at most 0.020000.",
            {
                "total_variation_distance": _decimal_text(total_variation),
                "maximum_absolute_band_difference": _decimal_text(
                    maximum_absolute_difference
                ),
            },
        ),
        _gate(
            "gamer_participation_identified",
            "IDENTIFICATION",
            False,
            (
                "A representative source identifies P(gamer | ONS age-by-sex "
                "cell), is encoded in a typed calibration target, and is "
                "consumed by the runtime participation hurdle."
            ),
            {
                "identified": False,
                "status": "UNIDENTIFIED_IN_SCHEMA_V1",
                "baseline_gamer_role": (
                    "illustrative projected-population metadata; not an "
                    "empirical UK age-by-sex participation estimate"
                ),
            },
        ),
        _gate(
            "non_gamer_zero_play_intention",
            "STRUCTURAL",
            non_gamer_positive_play == 0,
            "All sidecar non-gamers have intended_play_minutes equal to zero.",
            {
                "positive_share": _decimal_text(non_gamer_positive_play),
                "non_gamer_count": non_gamer_count,
            },
        ),
        _gate(
            "non_gamer_zero_spending_limit",
            "STRUCTURAL",
            non_gamer_positive_spending_limit == 0,
            "All sidecar non-gamers have intended_spending_limit_cents equal to zero.",
            {
                "positive_share": _decimal_text(
                    non_gamer_positive_spending_limit
                ),
                "non_gamer_count": non_gamer_count,
            },
        ),
        _gate(
            "non_gamer_zero_purchase_probability",
            "STRUCTURAL",
            non_gamer_purchase_probability_zero,
            (
                "An explicit purchase_probability field exists and equals zero "
                "for every sidecar non-gamer."
            ),
            {
                "field_available_and_valid": valid_purchase_probability,
                "all_non_gamer_values_zero": non_gamer_purchase_probability_zero,
            },
        ),
        _gate(
            "initializer_state_finite_and_bounded",
            "PLAUSIBILITY",
            _finite_bounded_state(players, life),
            (
                "Initializer minute fields are finite in [0,1440] and unit "
                "fields in [0,1]."
            ),
            {"checked_player_count": len(players)},
        ),
    ]
    passed = sum(gate["status"] == "PASS" for gate in gates)
    failed = len(gates) - passed

    selected_sleep = life.sleep_need_minutes[selected]
    selected_work = life.work_study_obligation_minutes[selected]
    selected_social = life.social_obligation_minutes[selected]
    selected_play = life.intended_play_minutes[selected]
    gamer_play = life.intended_play_minutes[selected_gamer]
    age_18_40_gamer = selected_gamer & (players.age_years <= 40)
    if not np.any(age_18_40_gamer):
        raise PointZeroAuditError(
            "point-zero selection has no sidecar gamers aged 18-40"
        )

    diagnostics = [
        _incomparable(
            "sleep_need_vs_ons_sleeping",
            "initialized sleep need",
            "ONS realised primary-activity sleeping",
            _mean_int(selected_sleep),
            _target_value(bundle, "time_sleeping_mean_march_2024"),
            "minutes_per_day",
            (
                "A physiological need is not realised diary time; the ONS "
                "population also includes ages 65+."
            ),
        ),
        _incomparable(
            "work_obligation_vs_ons_working",
            "initialized work/study obligation",
            "ONS realised working",
            _mean_int(selected_work),
            _target_value(bundle, "time_working_mean_march_2024"),
            "minutes_per_day",
            (
                "An assigned work/study obligation is not realised working "
                "time and includes study."
            ),
        ),
        _incomparable(
            "social_obligation_vs_ons_socialising",
            "initialized social obligation",
            "ONS realised primary-activity socialising",
            _mean_int(selected_social),
            _target_value(bundle, "time_socialising_mean_march_2024"),
            "minutes_per_day",
            (
                "An obligation is not realised diary time, and simultaneous "
                "activities may be omitted by ONS."
            ),
        ),
        _incomparable(
            "play_intention_vs_ons_gaming",
            "initialized intended play",
            "ONS realised unconditional primary-activity gaming",
            _mean_int(selected_play),
            _target_value(bundle, "time_gaming_mean_march_2024"),
            "minutes_per_day",
            (
                "An intention is not realised diary time; the ONS mean includes "
                "gamers and non-gamers aged 18+."
            ),
        ),
        _incomparable(
            "gamer_play_intention_vs_ofcom",
            "initialized intended play among sidecar gamers aged 18-64",
            "Ofcom/Ampere gamer-reported weekly play ages 16-64",
            _mean_int(gamer_play) * 7,
            _target_value(bundle, "ofcom_gamer_weekly_play_mean"),
            "minutes_per_week",
            (
                "The simulator field is an intention; Ofcom is a third-party "
                "self-report with a different age range and no published interval."
            ),
        ),
        _incomparable(
            "gamer_play_intention_vs_open_play",
            "initialized intended play among sidecar gamers aged 18-40",
            "selected Open Play reported weekly play ages 18-40",
            _mean_int(life.intended_play_minutes[age_18_40_gamer]) * 7,
            _target_value(bundle, "open_play_weekly_play_mean"),
            "minutes_per_week",
            (
                "Open Play is a selected trace-sharing sample and the simulator "
                "field is an intention, so this is not a population-fit residual."
            ),
        ),
    ]

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "PASS" if failed == 0 else "FAIL_CLOSED",
        "exit_code": 0 if failed == 0 else 1,
        "initialization": {
            "seed": INITIALIZATION_SEED,
            "configured_player_count": state.config.batch.player_count,
            "selected_population": "UK usual residents aged 18-64",
            "selected_player_count": selected_count,
            "population_execution_sha256": state.population_execution_sha256,
            "runtime_projection_sha256": state.runtime_projection_sha256,
            "assignment_sha256": state.assignment_sha256,
        },
        "execution_scope": {
            "population_initializer_executed": True,
            "player_life_initializer_executed": True,
            "scenario_initialized": False,
            "policy_day_executed": False,
            "campaign_executed": False,
        },
        "calibration_bundle": {
            "bundle_id": bundle.bundle_id,
            "bundle_sha256": bundle.bundle_sha256,
            "status": bundle.status,
            "campaign_ready": bundle.campaign_ready,
            "ons_population_18_64": target_population,
        },
        "population_diagnostics": {
            "age_margin": age_rows,
            "total_variation_distance": _decimal_text(total_variation),
            "age_sex_joint_total_variation_distance": (
                _decimal_text(joint_total_variation)
                if joint_total_variation is not None
                else None
            ),
            "runtime_sex_state_available": runtime_sex is not None,
            "runtime_sex_field_available": runtime_sex_field_available,
            "runtime_sex_lineage_attested": runtime_sex_lineage_attested,
            "runtime_sex_attestation_status": runtime_sex_status,
            "runtime_sex_counts": (
                {
                    "FEMALE": int(
                        np.count_nonzero(
                            runtime_sex[selected] == SOURCE_RECORDED_SEX_FEMALE
                        )
                    ),
                    "MALE": int(
                        np.count_nonzero(
                            runtime_sex[selected] == SOURCE_RECORDED_SEX_MALE
                        )
                    ),
                    "out_of_scope_unavailable": int(
                        np.count_nonzero(
                            runtime_sex[~selected]
                            == SOURCE_RECORDED_SEX_UNAVAILABLE
                        )
                    ),
                }
                if runtime_sex is not None
                else None
            ),
            "baseline_gamer_metadata_count": gamer_count,
            "baseline_non_gamer_metadata_count": non_gamer_count,
            "baseline_gamer_metadata_is_behaviorally_binding": (
                non_gamer_positive_play == 0
                and non_gamer_positive_spending_limit == 0
                and non_gamer_purchase_probability_zero
            ),
            "baseline_gamer_behavior_binding_checks": {
                "non_gamer_zero_play_intention": non_gamer_positive_play == 0,
                "non_gamer_zero_spending_limit": (
                    non_gamer_positive_spending_limit == 0
                ),
                "non_gamer_zero_purchase_probability": (
                    non_gamer_purchase_probability_zero
                ),
            },
            "current_game_none_share": _decimal_text(
                Fraction(
                    int(np.count_nonzero(players.current_game[selected] == -1)),
                    selected_count,
                )
            ),
        },
        "initializer_means": {
            "sleep_need_minutes_per_day": _decimal_text(_mean_int(selected_sleep)),
            "work_study_obligation_minutes_per_day": _decimal_text(
                _mean_int(selected_work)
            ),
            "social_obligation_minutes_per_day": _decimal_text(
                _mean_int(selected_social)
            ),
            "intended_play_minutes_per_day": _decimal_text(
                _mean_int(selected_play)
            ),
            "sidecar_gamer_intended_play_minutes_per_day": _decimal_text(
                _mean_int(gamer_play)
            ),
            "sidecar_gamer_intended_play_minutes_per_week": _decimal_text(
                _mean_int(gamer_play) * 7
            ),
            "sidecar_gamer_age_18_40_intended_play_minutes_per_week": (
                _decimal_text(
                    _mean_int(life.intended_play_minutes[age_18_40_gamer]) * 7
                )
            ),
            "sidecar_non_gamer_intended_play_minutes_per_day": _decimal_text(
                _mean_int(life.intended_play_minutes[selected_non_gamer])
            ),
        },
        "gates": gates,
        "gate_summary": {
            "passed": passed,
            "failed": failed,
            "total": len(gates),
        },
        "construct_incomparable_diagnostics": diagnostics,
        "interpretation": (
            "PASS means only that the declared point-zero structural and plausibility "
            "gates passed. It does not authorize a scenario, policy run, campaign, "
            "causal claim, or population inference."
        ),
    }


def run_point_zero_audit(
    *,
    bundle_path: Path = DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Verify inputs, initialize seed 101, and return the canonical report."""

    bundle = load_uk_adults_2024_calibration_bundle(
        bundle_path,
        repository_root=repository_root,
    )
    state = _initialize_only(
        config_path,
        calibration_bundle=bundle,
        repository_root=repository_root,
    )
    return build_report(
        bundle,
        state,
        repository_root=repository_root,
    )


def _error_report(exc: Exception) -> dict[str, object]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": AUDIT_ID,
        "status": "ERROR_FAIL_CLOSED",
        "exit_code": 1,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "execution_scope": {
            "scenario_initialized": False,
            "policy_day_executed": False,
            "campaign_executed": False,
        },
    }


class _FailClosedArgumentParser(argparse.ArgumentParser):
    """Convert command-line validation failures into the audit error path."""

    def error(self, message: str) -> None:
        raise PointZeroAuditError(f"argument parsing failed: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _FailClosedArgumentParser(
        add_help=False,
        description=(
            "Run the deterministic seed-101 UK-adult initializer-only point-zero "
            "audit. No scenario, policy day, or campaign is executed."
        )
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        dest="help_requested",
        help="Return a fail-closed JSON explanation without running the audit.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
        help="Calibration bundle directory or calibration_bundle.json path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Exploratory configuration whose point-zero initializer is audited.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Required only when the calibration bundle is outside this checkout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent JSON for human inspection; canonical key ordering is retained.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    pretty = False
    try:
        args = _parser().parse_args(argv)
        pretty = bool(args.pretty)
        if args.help_requested:
            raise PointZeroAuditError(
                "help requested; exit 0 is reserved exclusively for a passing audit"
            )
        report = run_point_zero_audit(
            bundle_path=args.bundle,
            config_path=args.config,
            repository_root=args.repository_root,
        )
    except Exception as exc:  # The command-line boundary is deliberately fail-closed.
        report = _error_report(exc)
    print(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        )
    )
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
