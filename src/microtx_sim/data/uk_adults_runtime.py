"""Fail-closed UK-adult runtime bindings backed by the 2024 ONS bundle.

This module adds only the source-recorded ONS ``FEMALE``/``MALE`` category to
synthetic UK residents aged 18--64. It does not infer gender identity, does not
label players outside that source scope, and does not authorize a campaign.
The aggregate cells identify conditional proportions, not individual sex, so
the per-player allocation is explicitly synthetic and deterministic.
"""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from ..agents.players import (
    ProjectedPopulationAssignment,
    ProjectedPopulationSexBinding,
    SOURCE_RECORDED_SEX_DTYPE,
    SOURCE_RECORDED_SEX_FEMALE,
    SOURCE_RECORDED_SEX_MALE,
    SOURCE_RECORDED_SEX_UNAVAILABLE,
    source_recorded_sex_derivation_input_sha256,
    source_recorded_sex_sha256,
)
from .calibration import (
    EstimandRole,
    UKAdults2024CalibrationBundle,
    load_uk_adults_2024_calibration_bundle,
)
from .population_projection import (
    PopulationProjectionExecution,
    PopulationProjectionValidationError,
    bind_population_projection_source_recorded_sex,
    verify_population_projection_execution,
)


UK_ADULTS_2024_SEX_ASSIGNMENT_METHOD = (
    "sha256-ranked-coupled-hamilton-within-age-band-v1"
)
UK_ADULTS_2024_SEX_JURISDICTION = "UK"
UK_ADULTS_2024_SEX_MIN_AGE = 18
UK_ADULTS_2024_SEX_MAX_AGE = 64


def uk_adults_2024_population_weights_sha256(
    bundle: UKAdults2024CalibrationBundle,
) -> str:
    """Hash the exact typed ONS age-by-sex rows consumed by the runtime."""

    if type(bundle) is not UKAdults2024CalibrationBundle:
        raise TypeError("bundle must be an exact UKAdults2024CalibrationBundle")
    payload = {
        "schema_version": 1,
        "bundle_id": bundle.bundle_id,
        "rows": [
            {
                "age_band": row.age_band,
                "age_min_inclusive": row.age_min_inclusive,
                "age_max_inclusive": row.age_max_inclusive,
                "sex": row.sex,
                "population_count": row.population_count,
                "adult_population_weight": format(
                    row.adult_population_weight,
                    "f",
                ),
                "source_id": row.source_id,
                "estimand_role": row.estimand_role.value,
            }
            for row in bundle.population_weights
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _ranked_positions(
    player_ids: np.ndarray,
    positions: np.ndarray,
    *,
    bundle_sha256: str,
    age_band: str,
    runtime_cell_id: str,
) -> tuple[int, ...]:
    domain = b"microtx-sim.uk-adults-2024-sex-rank.v1\0"

    def rank(position: int) -> tuple[bytes, int]:
        digest = sha256(domain)
        for value in (bundle_sha256, age_band, runtime_cell_id):
            encoded = value.encode("ascii")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
        player_id = int(player_ids[position])
        digest.update(player_id.to_bytes(8, "little", signed=True))
        return digest.digest(), player_id

    return tuple(sorted((int(value) for value in positions), key=rank))


def _hamilton_binary_count(
    sample_count: int,
    female_population: int,
    male_population: int,
) -> int:
    if sample_count < 0 or female_population <= 0 or male_population <= 0:
        raise PopulationProjectionValidationError(
            "ONS sex allocation requires non-negative samples and positive cells"
        )
    total = female_population + male_population
    female_quota = Fraction(sample_count * female_population, total)
    male_quota = Fraction(sample_count * male_population, total)
    female_floor = female_quota.numerator // female_quota.denominator
    male_floor = male_quota.numerator // male_quota.denominator
    remaining = sample_count - female_floor - male_floor
    if remaining not in {0, 1}:
        raise RuntimeError("binary Hamilton allocation did not preserve sample size")
    if remaining:
        female_remainder = female_quota - female_floor
        male_remainder = male_quota - male_floor
        # FEMALE is the documented canonical tie-break used by the source rows.
        if female_remainder >= male_remainder:
            female_floor += 1
    return female_floor


def _cell_tie_break(
    *,
    bundle_sha256: str,
    age_band: str,
    runtime_cell_id: str,
) -> bytes:
    digest = sha256(b"microtx-sim.uk-adults-2024-sex-cell-rank.v1\0")
    for value in (bundle_sha256, age_band, runtime_cell_id):
        encoded = value.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.digest()


def _coupled_hamilton_female_counts(
    sample_counts: dict[int, int],
    *,
    female_population: int,
    male_population: int,
    bundle_sha256: str,
    age_band: str,
    assignment: ProjectedPopulationAssignment,
) -> dict[int, int]:
    """Preserve the age-band total while minimizing within-cell rounding error."""

    if not sample_counts:
        return {}
    if any(value <= 0 for value in sample_counts.values()):
        raise PopulationProjectionValidationError(
            "coupled Hamilton allocation requires positive runtime-cell samples"
        )
    population_total = female_population + male_population
    total_samples = sum(sample_counts.values())
    target_female = _hamilton_binary_count(
        total_samples,
        female_population,
        male_population,
    )
    quotas = {
        cell_index: Fraction(count * female_population, population_total)
        for cell_index, count in sample_counts.items()
    }
    result = {
        cell_index: quota.numerator // quota.denominator
        for cell_index, quota in quotas.items()
    }
    remaining = target_female - sum(result.values())
    if not 0 <= remaining <= len(result):
        raise RuntimeError("coupled Hamilton remainder is outside cell bounds")
    ranked_cells = sorted(
        result,
        key=lambda cell_index: (
            -(quotas[cell_index] - result[cell_index]),
            _cell_tie_break(
                bundle_sha256=bundle_sha256,
                age_band=age_band,
                runtime_cell_id=(
                    assignment.metadata.cells[cell_index].cell_id
                ),
            ),
            cell_index,
        ),
    )
    for cell_index in ranked_cells[:remaining]:
        result[cell_index] += 1
    if sum(result.values()) != target_female or any(
        not 0 <= result[cell_index] <= sample_counts[cell_index]
        for cell_index in result
    ):
        raise RuntimeError("coupled Hamilton allocation failed exact conservation")
    return result


def _reattest_bundle(
    bundle: UKAdults2024CalibrationBundle,
    *,
    repository_root: str | Path | None,
) -> UKAdults2024CalibrationBundle:
    if type(bundle) is not UKAdults2024CalibrationBundle:
        raise TypeError("bundle must be an exact UKAdults2024CalibrationBundle")
    observed_bundle = load_uk_adults_2024_calibration_bundle(
        bundle.bundle_path,
        repository_root=repository_root,
    )
    if observed_bundle != bundle:
        raise PopulationProjectionValidationError(
            "UK-adult calibration bundle differs from re-attested source bytes"
        )
    return observed_bundle


def _expected_uk_adults_2024_source_recorded_sex(
    execution: PopulationProjectionExecution,
    bundle: UKAdults2024CalibrationBundle,
) -> tuple[
    PopulationProjectionExecution,
    np.ndarray,
    ProjectedPopulationSexBinding,
]:
    observed_execution = verify_population_projection_execution(execution)
    players = observed_execution.players
    assignment = players.projected_population
    if type(assignment) is not ProjectedPopulationAssignment:
        raise PopulationProjectionValidationError(
            "UK-adult sex binding requires projected-population lineage"
        )
    if UK_ADULTS_2024_SEX_JURISDICTION not in players.jurisdiction_codes:
        raise PopulationProjectionValidationError(
            "projected population has no UK jurisdiction for the ONS sex binding"
        )

    rows_by_band: dict[str, dict[str, object]] = {}
    source_ids: set[str] = set()
    for row in bundle.population_weights:
        if row.estimand_role is not EstimandRole.CALIBRATION:
            raise PopulationProjectionValidationError(
                "ONS population sex rows must retain CALIBRATION role"
            )
        if row.sex not in {SOURCE_RECORDED_SEX_FEMALE, SOURCE_RECORDED_SEX_MALE}:
            raise PopulationProjectionValidationError(
                "ONS population sex rows must use FEMALE or MALE"
            )
        source_ids.add(row.source_id)
        band = rows_by_band.setdefault(
            row.age_band,
            {
                "age_min_inclusive": row.age_min_inclusive,
                "age_max_inclusive": row.age_max_inclusive,
                "counts": {},
            },
        )
        if (
            band["age_min_inclusive"] != row.age_min_inclusive
            or band["age_max_inclusive"] != row.age_max_inclusive
        ):
            raise PopulationProjectionValidationError(
                "ONS population rows disagree on an age-band interval"
            )
        counts = band["counts"]
        assert isinstance(counts, dict)
        if row.sex in counts:
            raise PopulationProjectionValidationError(
                "ONS population rows duplicate an age-by-sex cell"
            )
        counts[row.sex] = row.population_count
    if len(source_ids) != 1:
        raise PopulationProjectionValidationError(
            "ONS population sex rows must bind one common source"
        )
    if any(
        set(band["counts"]) != {
            SOURCE_RECORDED_SEX_FEMALE,
            SOURCE_RECORDED_SEX_MALE,
        }
        for band in rows_by_band.values()
    ):
        raise PopulationProjectionValidationError(
            "every ONS age band must contain FEMALE and MALE cells"
        )

    uk_index = players.jurisdiction_codes.index(UK_ADULTS_2024_SEX_JURISDICTION)
    sex = np.full(
        len(players),
        SOURCE_RECORDED_SEX_UNAVAILABLE,
        dtype=SOURCE_RECORDED_SEX_DTYPE,
    )
    covered = np.zeros(len(players), dtype=np.bool_)
    for age_band, band in sorted(rows_by_band.items()):
        lower = int(band["age_min_inclusive"])
        upper = int(band["age_max_inclusive"])
        band_mask = (
            (players.jurisdiction == uk_index)
            & (players.age_years >= lower)
            & (players.age_years <= upper)
        )
        covered |= band_mask
        counts = band["counts"]
        assert isinstance(counts, dict)
        female_population = int(counts[SOURCE_RECORDED_SEX_FEMALE])
        male_population = int(counts[SOURCE_RECORDED_SEX_MALE])
        positions_by_cell = {
            int(cell_index): np.flatnonzero(
                band_mask & (assignment.cell_index == int(cell_index))
            )
            for cell_index in np.unique(assignment.cell_index[band_mask])
        }
        female_by_cell = _coupled_hamilton_female_counts(
            {
                cell_index: int(positions.size)
                for cell_index, positions in positions_by_cell.items()
            },
            female_population=female_population,
            male_population=male_population,
            bundle_sha256=bundle.bundle_sha256,
            age_band=age_band,
            assignment=assignment,
        )
        for cell_index, positions in sorted(positions_by_cell.items()):
            ordered = _ranked_positions(
                players.player_id,
                positions,
                bundle_sha256=bundle.bundle_sha256,
                age_band=age_band,
                runtime_cell_id=assignment.metadata.cells[cell_index].cell_id,
            )
            female_count = female_by_cell[cell_index]
            sex[np.asarray(ordered[:female_count], dtype=np.int64)] = (
                SOURCE_RECORDED_SEX_FEMALE
            )
            sex[np.asarray(ordered[female_count:], dtype=np.int64)] = (
                SOURCE_RECORDED_SEX_MALE
            )

    expected_scope = (
        (players.jurisdiction == uk_index)
        & (players.age_years >= UK_ADULTS_2024_SEX_MIN_AGE)
        & (players.age_years <= UK_ADULTS_2024_SEX_MAX_AGE)
    )
    if not np.any(expected_scope):
        raise PopulationProjectionValidationError(
            "projected population contains no UK residents aged 18--64"
        )
    if not np.array_equal(covered, expected_scope):
        raise PopulationProjectionValidationError(
            "ONS age bands do not exactly cover the UK 18--64 runtime scope"
        )
    if np.any(
        ~np.isin(
            sex[expected_scope],
            (SOURCE_RECORDED_SEX_FEMALE, SOURCE_RECORDED_SEX_MALE),
        )
    ):
        raise RuntimeError("ONS sex allocation left an in-scope player unassigned")

    sex_sha256 = source_recorded_sex_sha256(sex)
    binding = ProjectedPopulationSexBinding(
        source_id=next(iter(source_ids)),
        evidence_bundle_id=bundle.bundle_id,
        evidence_bundle_sha256=bundle.bundle_sha256,
        population_weights_sha256=uk_adults_2024_population_weights_sha256(
            bundle
        ),
        jurisdiction_code=UK_ADULTS_2024_SEX_JURISDICTION,
        age_min_inclusive=UK_ADULTS_2024_SEX_MIN_AGE,
        age_max_inclusive=UK_ADULTS_2024_SEX_MAX_AGE,
        assignment_method=UK_ADULTS_2024_SEX_ASSIGNMENT_METHOD,
        derivation_input_sha256=(
            source_recorded_sex_derivation_input_sha256(
                players.player_id,
                players.age_years,
                players.jurisdiction,
                assignment.cell_index,
            )
        ),
        sex_sha256=sex_sha256,
    )
    return observed_execution, sex, binding


def bind_uk_adults_2024_source_recorded_sex(
    execution: PopulationProjectionExecution,
    bundle: UKAdults2024CalibrationBundle,
    *,
    repository_root: str | Path | None = None,
) -> PopulationProjectionExecution:
    """Bind deterministic ONS sex categories to the UK 18--64 runtime scope.

    Hamilton remainders are coupled across runtime cells within each ONS age
    band. The aggregate band count is therefore exact, while deterministic
    player ranking keeps the allocation reproducible. This is a structural
    point-zero binding only and cannot authorize treatment execution.
    """

    observed_bundle = _reattest_bundle(
        bundle,
        repository_root=repository_root,
    )
    observed_execution, sex, binding = (
        _expected_uk_adults_2024_source_recorded_sex(
            execution,
            observed_bundle,
        )
    )
    return bind_population_projection_source_recorded_sex(
        observed_execution,
        sex,
        binding,
    )


def verify_uk_adults_2024_source_recorded_sex(
    execution: PopulationProjectionExecution,
    bundle: UKAdults2024CalibrationBundle,
    *,
    repository_root: str | Path | None = None,
) -> PopulationProjectionExecution:
    """Recompute the canonical UK-adult allocation and all source bindings."""

    observed_bundle = _reattest_bundle(
        bundle,
        repository_root=repository_root,
    )
    observed_execution, expected_sex, expected_binding = (
        _expected_uk_adults_2024_source_recorded_sex(
            execution,
            observed_bundle,
        )
    )
    assignment = observed_execution.players.projected_population
    if (
        type(assignment) is not ProjectedPopulationAssignment
        or observed_execution.players.sex is None
        or assignment.sex_binding is None
    ):
        raise PopulationProjectionValidationError(
            "UK-adult source-recorded sex binding is absent"
        )
    if assignment.sex_binding != expected_binding:
        raise PopulationProjectionValidationError(
            "UK-adult source-recorded sex binding differs from the canonical "
            "source, scope, weights, method, derivation inputs, or allocation"
        )
    if not np.array_equal(observed_execution.players.sex, expected_sex):
        raise PopulationProjectionValidationError(
            "UK-adult source-recorded sex values differ from canonical allocation"
        )
    return observed_execution


__all__ = [
    "UK_ADULTS_2024_SEX_ASSIGNMENT_METHOD",
    "UK_ADULTS_2024_SEX_JURISDICTION",
    "UK_ADULTS_2024_SEX_MAX_AGE",
    "UK_ADULTS_2024_SEX_MIN_AGE",
    "bind_uk_adults_2024_source_recorded_sex",
    "uk_adults_2024_population_weights_sha256",
    "verify_uk_adults_2024_source_recorded_sex",
]
