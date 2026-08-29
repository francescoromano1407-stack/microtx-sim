from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Final, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from ..agents.players import (
    PlayerTable,
    ProjectedPopulationAssignment,
    ProjectedPopulationCellMetadata,
    ProjectedPopulationMetadata,
    TRAIT_NAMES,
    projected_population_assignment_sha256,
    projected_population_plan_sha256,
)
from ..types import HarmDimension, Motive


class CounterRNGLike(Protocol):
    """Minimal counter-based RNG contract used by population factories."""

    def uniform(
        self,
        entity_ids: NDArray[np.int64],
        tick: int,
        stream: int,
        draw_index: int,
    ) -> NDArray[np.float64]: ...

    def normal(
        self,
        entity_ids: NDArray[np.int64],
        tick: int,
        stream: int,
        draw_index: int,
    ) -> NDArray[np.float64]: ...


_DEFAULT_AGE_EDGES: Final[tuple[int, ...]] = (8, 13, 18, 25, 35, 45, 55, 65, 81)
_DEFAULT_AGE_WEIGHTS: Final[tuple[float, ...]] = (
    0.07,
    0.10,
    0.14,
    0.20,
    0.17,
    0.14,
    0.10,
    0.08,
)
_DEFAULT_TRAIT_MEANS: Final[tuple[float, ...]] = (0.48, 0.52, 0.47, 0.51, 0.53, 0.52)
_DEFAULT_TRAIT_SCALES: Final[tuple[float, ...]] = (0.18, 0.17, 0.18, 0.16, 0.18, 0.18)
_DEFAULT_TRAIT_CORRELATION: Final[tuple[tuple[float, ...], ...]] = (
    (1.00, 0.45, 0.25, 0.10, -0.20, -0.55),
    (0.45, 1.00, 0.30, 0.15, -0.10, -0.35),
    (0.25, 0.30, 1.00, 0.20, -0.05, -0.20),
    (0.10, 0.15, 0.20, 1.00, 0.05, 0.10),
    (-0.20, -0.10, -0.05, 0.05, 1.00, 0.35),
    (-0.55, -0.35, -0.20, 0.10, 0.35, 1.00),
)


@dataclass(frozen=True, slots=True)
class CountryProfile:
    """Configurable, provenance-addressable inputs for one jurisdiction.

    Defaults are illustrative priors suitable for smoke tests, not empirical
    claims.  Scientific configurations should override them with values keyed
    to official tables in ``source_ids``.
    """

    code: str
    population_weight: float = 1.0
    adult_age: int = 18
    age_band_edges: tuple[int, ...] = _DEFAULT_AGE_EDGES
    age_band_weights: tuple[float, ...] = _DEFAULT_AGE_WEIGHTS

    monthly_income_median_cents: int = 180_000
    income_log_sigma: float = 0.55
    income_peak_age: float = 45.0
    income_age_spread: float = 25.0
    income_age_floor: float = 0.28
    minor_allowance_median_cents: int = 2_500

    personal_liquidity_months: float = 1.6
    household_liquidity_months: float = 2.2
    credit_access_probability: float = 0.72
    credit_limit_income_multiple: float = 1.8
    adult_stored_payment_probability: float = 0.90

    mean_players_per_household: float = 1.25
    minor_stored_card_probability: float = 0.13
    minor_guardian_consent_probability: float = 0.35
    guardian_supervision_mean: float = 0.66

    awareness_mean: float = 0.50
    trait_means: tuple[float, ...] = _DEFAULT_TRAIT_MEANS
    trait_scales: tuple[float, ...] = _DEFAULT_TRAIT_SCALES
    trait_correlation: tuple[tuple[float, ...], ...] = _DEFAULT_TRAIT_CORRELATION
    motive_logits: tuple[float, ...] = (0.05, 0.02, 0.00, 0.04, 0.06)
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code or self.code.strip() != self.code:
            raise ValueError("code must be non-empty and have no surrounding whitespace")
        if not np.isfinite(self.population_weight) or self.population_weight <= 0.0:
            raise ValueError("population_weight must be positive")
        if isinstance(self.adult_age, bool) or not isinstance(
            self.adult_age, (int, np.integer)
        ) or not 13 <= self.adult_age <= 25:
            raise ValueError("adult_age must be between 13 and 25")
        if len(self.age_band_edges) < 2 or any(
            right <= left
            for left, right in zip(self.age_band_edges, self.age_band_edges[1:])
        ):
            raise ValueError("age_band_edges must be strictly increasing")
        if any(
            isinstance(edge, bool) or not isinstance(edge, (int, np.integer))
            for edge in self.age_band_edges
        ):
            raise TypeError("age_band_edges must contain integer ages")
        if len(self.age_band_weights) != len(self.age_band_edges) - 1:
            raise ValueError("age_band_weights must have one value per age band")
        if any(
            not np.isfinite(weight) or weight < 0.0
            for weight in self.age_band_weights
        ) or sum(self.age_band_weights) <= 0.0:
            raise ValueError("age_band_weights must be non-negative with positive sum")
        if isinstance(self.monthly_income_median_cents, bool) or not isinstance(
            self.monthly_income_median_cents, (int, np.integer)
        ):
            raise TypeError("monthly_income_median_cents must be integer cents")
        if self.monthly_income_median_cents <= 0:
            raise ValueError("monthly_income_median_cents must be positive")
        if isinstance(self.minor_allowance_median_cents, bool) or not isinstance(
            self.minor_allowance_median_cents, (int, np.integer)
        ):
            raise TypeError("minor_allowance_median_cents must be integer cents")
        if self.minor_allowance_median_cents < 0:
            raise ValueError("minor_allowance_median_cents cannot be negative")
        for name in (
            "income_log_sigma",
            "income_age_spread",
            "personal_liquidity_months",
            "household_liquidity_months",
            "credit_limit_income_multiple",
            "mean_players_per_household",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not np.isfinite(self.income_peak_age):
            raise ValueError("income_peak_age must be finite")
        if not np.isfinite(self.income_age_floor) or not 0.0 < self.income_age_floor <= 1.0:
            raise ValueError("income_age_floor must be in (0, 1]")
        for name in (
            "credit_access_probability",
            "adult_stored_payment_probability",
            "minor_stored_card_probability",
            "minor_guardian_consent_probability",
            "guardian_supervision_mean",
            "awareness_mean",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if len(self.trait_means) != len(TRAIT_NAMES) or len(self.trait_scales) != len(
            TRAIT_NAMES
        ):
            raise ValueError("trait means/scales must match TRAIT_NAMES")
        if any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.trait_means
        ):
            raise ValueError("trait_means must be in [0, 1]")
        if any(not np.isfinite(value) or value <= 0.0 for value in self.trait_scales):
            raise ValueError("trait_scales must be positive")
        if len(self.motive_logits) != len(Motive):
            raise ValueError("motive_logits must match Motive")
        if any(not np.isfinite(value) for value in self.motive_logits):
            raise ValueError("motive_logits must be finite")

        correlation = np.asarray(self.trait_correlation, dtype=np.float64)
        shape = (len(TRAIT_NAMES), len(TRAIT_NAMES))
        if correlation.shape != shape:
            raise ValueError(f"trait_correlation must have shape {shape}")
        if not np.all(np.isfinite(correlation)):
            raise ValueError("trait_correlation must be finite")
        if not np.allclose(correlation, correlation.T, atol=1e-10):
            raise ValueError("trait_correlation must be symmetric")
        if not np.allclose(np.diag(correlation), 1.0, atol=1e-10):
            raise ValueError("trait_correlation diagonal must equal one")
        try:
            np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as exc:
            raise ValueError("trait_correlation must be positive definite") from exc


@dataclass(frozen=True, slots=True)
class PopulationProjectionCell:
    """One exact target cell after upstream evidence-to-runtime conversion.

    ``monthly_disposable_income_*`` is an explicit runtime interval for the
    PlayerTable column.  This primitive intentionally does not infer it from a
    source household-income label or perform currency/equivalisation work.
    """

    cell_id: str
    jurisdiction_code: str
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

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "jurisdiction_code",
            "monthly_disposable_income_band_id",
            "household_type",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value or value.strip() != value:
                raise ValueError(
                    f"{name} must be non-empty and have no surrounding whitespace"
                )
        for name in (
            "age_min_inclusive",
            "age_max_exclusive",
            "monthly_disposable_income_min_cents",
            "monthly_disposable_income_max_cents_exclusive",
            "modeled_players_per_household",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a Python integer")
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
        if not isinstance(self.global_mass, tuple) or len(self.global_mass) != 2:
            raise TypeError("global_mass must be a (numerator, denominator) tuple")
        numerator, denominator = self.global_mass
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
        ):
            raise TypeError(
                "global_mass numerator and denominator must be Python integers"
            )
        if numerator < 0:
            raise ValueError("global_mass numerator cannot be negative")
        if denominator <= 0:
            raise ValueError("global_mass denominator must be positive")
        reduced = Fraction(numerator, denominator)
        if (reduced.numerator, reduced.denominator) != self.global_mass:
            raise ValueError("global_mass must be in lowest terms")


class _Stream:
    JURISDICTION = 100
    AGE_BAND = 101
    AGE_WITHIN_BAND = 102
    HOUSEHOLD = 103
    HOUSEHOLD_RESOURCES = 104
    INCOME = 105
    LIQUIDITY = 106
    CREDIT_ACCESS = 107
    CREDIT_LIMIT = 108
    PAYMENT_ACCESS = 109
    GUARDIAN_CONSENT = 110
    SUPERVISION = 111
    TRAITS = 112
    MOTIVES = 113
    AWARENESS = 114


class _ProjectionStream:
    """Reserved counter coordinates for the opt-in projection primitive."""

    CELL_ASSIGNMENT = 1_100
    AGE_WITHIN_CELL = 1_101
    MONTHLY_DISPOSABLE_INCOME = 1_102
    HOUSEHOLD_ORDER = 1_103
    HOUSEHOLD_RESOURCES = 1_104
    LIQUIDITY = 1_105
    CREDIT_ACCESS = 1_106
    CREDIT_LIMIT = 1_107
    PAYMENT_ACCESS = 1_108
    GUARDIAN_CONSENT = 1_109
    SUPERVISION = 1_110
    TRAITS = 1_111
    MOTIVES = 1_112
    AWARENESS = 1_113


def initialize_player_table(
    player_count: int,
    country_profiles: Sequence[CountryProfile],
    rng: CounterRNGLike,
    *,
    tick: int = 0,
    first_player_id: int = 0,
) -> PlayerTable:
    """Create a heterogeneous population without assigning behavioural types.

    All random draws are keyed by entity id, tick, stream and draw index.  The
    implementation is vectorised by jurisdiction; it does not form player by
    player Python objects.
    """

    if isinstance(player_count, bool) or not isinstance(player_count, (int, np.integer)):
        raise TypeError("player_count must be an integer")
    if player_count < 0:
        raise ValueError("player_count cannot be negative")
    if not country_profiles:
        raise ValueError("at least one country profile is required")
    if len(country_profiles) > np.iinfo(np.int16).max:
        raise ValueError("too many country profiles for int16 jurisdiction codes")
    codes = tuple(profile.code for profile in country_profiles)
    if len(set(codes)) != len(codes):
        raise ValueError("country profile codes must be unique")
    if isinstance(first_player_id, bool) or not isinstance(
        first_player_id, (int, np.integer)
    ):
        raise TypeError("first_player_id must be an integer")
    if first_player_id < 0 or first_player_id + player_count > np.iinfo(np.int64).max:
        raise ValueError("player id range is outside int64")

    player_id = np.arange(
        first_player_id, first_player_id + player_count, dtype=np.int64
    )
    jurisdiction = _sample_jurisdictions(player_id, country_profiles, rng, tick)

    age_years = np.empty(player_count, dtype=np.int16)
    household_id = np.empty(player_count, dtype=np.int64)
    monthly_income = np.empty(player_count, dtype=np.int64)
    liquidity = np.empty(player_count, dtype=np.int64)
    credit_limit = np.zeros(player_count, dtype=np.int64)
    allowance = np.zeros(player_count, dtype=np.int64)
    household_liquidity = np.empty(player_count, dtype=np.int64)
    stored_payment = np.zeros(player_count, dtype=np.bool_)
    guardian_supervision = np.zeros(player_count, dtype=np.float32)
    guardian_consent = np.zeros(player_count, dtype=np.bool_)
    traits = np.empty((player_count, len(TRAIT_NAMES)), dtype=np.float32)
    motive_weights = np.empty((player_count, len(Motive)), dtype=np.float32)
    awareness = np.empty(player_count, dtype=np.float32)

    household_offset = 0
    for country_index, profile in enumerate(country_profiles):
        mask = jurisdiction == country_index
        positions = np.flatnonzero(mask)
        entity_ids = player_id[positions]
        count = positions.size
        if not count:
            continue

        ages = _sample_ages(entity_ids, profile, rng, tick)
        age_years[positions] = ages
        minor = ages < profile.adult_age

        number_households = max(
            1, int(np.ceil(count / profile.mean_players_per_household))
        )
        hh_uniform = _uniform(rng, entity_ids, tick, _Stream.HOUSEHOLD, 0)
        local_household = np.minimum(
            (hh_uniform * number_households).astype(np.int64), number_households - 1
        )
        player_households = household_offset + local_household
        household_id[positions] = player_households
        unique_households, household_inverse = np.unique(
            player_households, return_inverse=True
        )
        household_resource_noise = _normal(
            rng, unique_households, tick, _Stream.HOUSEHOLD_RESOURCES, country_index
        )
        hh_resources = _money(
            profile.monthly_income_median_cents
            * profile.household_liquidity_months
            * np.exp(np.clip(0.65 * household_resource_noise, -4.0, 4.0))
        )
        household_liquidity[positions] = hh_resources[household_inverse]
        household_offset += number_households

        income_noise = _normal(rng, entity_ids, tick, _Stream.INCOME, 0)
        adult_age_factor = profile.income_age_floor + (
            1.0 - profile.income_age_floor
        ) * np.exp(
            -0.5
            * ((ages.astype(np.float64) - profile.income_peak_age) / profile.income_age_spread)
            ** 2
        )
        adult_income = _money(
            profile.monthly_income_median_cents
            * adult_age_factor
            * np.exp(np.clip(profile.income_log_sigma * income_noise, -5.0, 5.0))
        )
        allowance_age_factor = np.clip(
            (ages.astype(np.float64) - profile.age_band_edges[0] + 1.0)
            / max(profile.adult_age - profile.age_band_edges[0], 1),
            0.15,
            1.0,
        )
        minor_allowance = _money(
            profile.minor_allowance_median_cents
            * allowance_age_factor
            * np.exp(np.clip(0.35 * income_noise, -3.0, 3.0))
        )
        income_for_players = np.where(minor, minor_allowance, adult_income)
        monthly_income[positions] = income_for_players
        allowance[positions] = np.where(minor, minor_allowance, 0)

        liquidity_noise = _normal(rng, entity_ids, tick, _Stream.LIQUIDITY, 0)
        personal_months = profile.personal_liquidity_months * np.exp(
            np.clip(0.55 * liquidity_noise, -4.0, 4.0)
        )
        minor_balance_fraction = 0.25 + 1.75 * _uniform(
            rng, entity_ids, tick, _Stream.LIQUIDITY, 1
        )
        liquidity_for_players = _money(
            np.where(
                minor,
                minor_allowance * minor_balance_fraction,
                adult_income * personal_months,
            )
        )
        liquidity[positions] = liquidity_for_players

        credit_access = (
            _uniform(rng, entity_ids, tick, _Stream.CREDIT_ACCESS, 0)
            < profile.credit_access_probability
        ) & ~minor
        credit_noise = _normal(rng, entity_ids, tick, _Stream.CREDIT_LIMIT, 0)
        credit_values = _money(
            adult_income
            * profile.credit_limit_income_multiple
            * np.exp(np.clip(0.30 * credit_noise, -2.0, 2.0))
        )
        credit_limit[positions] = np.where(credit_access, credit_values, 0)

        payment_draw = _uniform(rng, entity_ids, tick, _Stream.PAYMENT_ACCESS, 0)
        stored_payment[positions] = np.where(
            minor,
            payment_draw < profile.minor_stored_card_probability,
            payment_draw < profile.adult_stored_payment_probability,
        )
        guardian_consent[positions] = minor & (
            _uniform(rng, entity_ids, tick, _Stream.GUARDIAN_CONSENT, 0)
            < profile.minor_guardian_consent_probability
        )
        supervision_noise = _normal(rng, entity_ids, tick, _Stream.SUPERVISION, 0)
        supervision = _sigmoid(
            _logit(profile.guardian_supervision_mean) + 0.85 * supervision_noise
        )
        guardian_supervision[positions] = np.where(minor, supervision, 0.0).astype(
            np.float32
        )

        profile_traits = _sample_traits(entity_ids, profile, rng, tick)
        traits[positions] = profile_traits
        motive_weights[positions] = _sample_motives(
            entity_ids, profile, profile_traits, rng, tick
        )
        literacy = profile_traits[:, TRAIT_NAMES.index("financial_literacy")]
        self_control = profile_traits[:, TRAIT_NAMES.index("self_control")]
        awareness_noise = _normal(rng, entity_ids, tick, _Stream.AWARENESS, 0)
        awareness_values = _sigmoid(
            _logit(profile.awareness_mean)
            + 0.9 * (literacy - 0.5)
            + 0.35 * (self_control - 0.5)
            + 0.55 * awareness_noise
        )
        awareness[positions] = awareness_values.astype(np.float32)

    adult_ages = np.asarray(
        [profile.adult_age for profile in country_profiles], dtype=np.int16
    )
    is_minor = age_years < adult_ages[jurisdiction]
    baseline = _baseline_vulnerability(
        age_years,
        is_minor,
        monthly_income,
        jurisdiction,
        country_profiles,
        traits,
    )

    return PlayerTable(
        player_id=player_id,
        age_years=age_years,
        jurisdiction=jurisdiction,
        household_id=household_id,
        is_minor=is_minor,
        monthly_disposable_income_cents=monthly_income,
        liquidity_cents=liquidity,
        credit_limit_cents=credit_limit,
        allowance_cents=allowance,
        household_liquidity_cents=household_liquidity,
        has_stored_payment_access=stored_payment,
        guardian_supervision=guardian_supervision,
        guardian_consent=guardian_consent,
        traits=traits,
        motive_weights=motive_weights,
        baseline_vulnerability=baseline,
        harm_state=np.zeros(
            (player_count, len(HarmDimension)), dtype=np.float32
        ),
        current_game=np.full(player_count, -1, dtype=np.int32),
        awareness=awareness,
        jurisdiction_codes=codes,
        adult_age_by_jurisdiction=tuple(
            profile.adult_age for profile in country_profiles
        ),
    )


def initialize_projected_player_table(
    player_count: int,
    country_profiles: Sequence[CountryProfile],
    rng: CounterRNGLike,
    projection_cells: Sequence[PopulationProjectionCell],
    *,
    projection_id: str,
    tick: int = 0,
    first_player_id: int = 0,
) -> PlayerTable:
    """Create an opt-in population from exact, already-resolved target cells.

    This is deliberately separate from :func:`initialize_player_table`, whose
    legacy marginal generator and RNG coordinates remain unchanged.  The
    caller is responsible for verifying evidence and resolving source concepts
    into the explicit runtime cell fields accepted here. The returned runtime
    projection digest is computed from those exact cells; callers cannot attach
    an unrelated upstream plan digest. Baseline gamer and payer-history labels
    remain immutable sidecar metadata; this initializer does not map them into
    current games, payment access, or spending history.
    """

    if isinstance(player_count, bool) or not isinstance(player_count, (int, np.integer)):
        raise TypeError("player_count must be an integer")
    if player_count < 0:
        raise ValueError("player_count cannot be negative")
    if not country_profiles:
        raise ValueError("at least one country profile is required")
    if len(country_profiles) > np.iinfo(np.int16).max:
        raise ValueError("too many country profiles for int16 jurisdiction codes")
    codes = tuple(profile.code for profile in country_profiles)
    if len(set(codes)) != len(codes):
        raise ValueError("country profile codes must be unique")
    if isinstance(first_player_id, bool) or not isinstance(
        first_player_id, (int, np.integer)
    ):
        raise TypeError("first_player_id must be an integer")
    if first_player_id < 0 or first_player_id + player_count > np.iinfo(np.int64).max:
        raise ValueError("player id range is outside int64")

    raw_cells = tuple(projection_cells)
    if not raw_cells:
        raise ValueError("at least one projection cell is required")
    if any(type(cell) is not PopulationProjectionCell for cell in raw_cells):
        raise TypeError("projection_cells must contain PopulationProjectionCell")
    cell_ids = tuple(cell.cell_id for cell in raw_cells)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("projection cell_id values must be unique")
    cells = tuple(sorted(raw_cells, key=lambda cell: cell.cell_id))

    code_to_index = {code: index for index, code in enumerate(codes)}
    unknown_codes = sorted(
        {cell.jurisdiction_code for cell in cells}.difference(code_to_index)
    )
    if unknown_codes:
        raise ValueError(
            "projection cells reference unknown jurisdiction codes: "
            + ", ".join(unknown_codes)
        )
    masses = tuple(Fraction(*cell.global_mass) for cell in cells)
    if sum(masses, start=Fraction(0, 1)) != 1:
        raise ValueError("projection cell global masses must sum exactly to one")

    group_specification: dict[
        tuple[str, str, str],
        tuple[int, int, int],
    ] = {}
    for cell in cells:
        key = (
            cell.jurisdiction_code,
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
                "a jurisdiction/income-band/household-type group must have one "
                "runtime income interval and modeled household size"
            )

    counts = _hamilton_cell_counts(player_count, cells, masses)
    missing_positive = [
        cell.cell_id
        for cell, mass, count in zip(cells, masses, counts, strict=True)
        if mass > 0 and count == 0
    ]
    if missing_positive:
        raise ValueError(
            "player_count is too small for every positive-mass projection cell "
            "to be represented under Hamilton allocation: "
            + ", ".join(missing_positive)
        )

    metadata_cells: list[ProjectedPopulationCellMetadata] = []
    for cell, mass, count in zip(cells, masses, counts, strict=True):
        analysis_weight = Fraction(0, 1) if mass == 0 else mass / count
        metadata_cells.append(
            ProjectedPopulationCellMetadata(
                cell_id=cell.cell_id,
                jurisdiction_code=cell.jurisdiction_code,
                jurisdiction_index=code_to_index[cell.jurisdiction_code],
                age_min_inclusive=cell.age_min_inclusive,
                age_max_exclusive=cell.age_max_exclusive,
                monthly_disposable_income_band_id=(
                    cell.monthly_disposable_income_band_id
                ),
                monthly_disposable_income_min_cents=(
                    cell.monthly_disposable_income_min_cents
                ),
                monthly_disposable_income_max_cents_exclusive=(
                    cell.monthly_disposable_income_max_cents_exclusive
                ),
                household_type=cell.household_type,
                modeled_players_per_household=cell.modeled_players_per_household,
                baseline_gamer=cell.baseline_gamer,
                baseline_ever_payer=cell.baseline_ever_payer,
                global_mass=(mass.numerator, mass.denominator),
                analysis_weight=(
                    analysis_weight.numerator,
                    analysis_weight.denominator,
                ),
            )
        )
    immutable_metadata_cells = tuple(metadata_cells)
    metadata = ProjectedPopulationMetadata(
        projection_id=projection_id,
        projection_sha256=projected_population_plan_sha256(
            projection_id,
            immutable_metadata_cells,
        ),
        cells=immutable_metadata_cells,
    )

    player_id = np.arange(
        first_player_id, first_player_id + player_count, dtype=np.int64
    )
    canonical_slots = np.repeat(
        np.arange(len(cells), dtype=np.int32),
        np.asarray(counts, dtype=np.int64),
    )
    assignment_draw = _uniform(
        rng,
        player_id,
        tick,
        _ProjectionStream.CELL_ASSIGNMENT,
        0,
    )
    assignment_order = np.lexsort((player_id, assignment_draw))
    cell_index = np.empty(player_count, dtype=np.int32)
    cell_index[assignment_order] = canonical_slots

    jurisdiction_by_cell = np.asarray(
        [code_to_index[cell.jurisdiction_code] for cell in cells],
        dtype=np.int16,
    )
    jurisdiction = jurisdiction_by_cell[cell_index]
    age_years = np.empty(player_count, dtype=np.int16)
    monthly_income = np.empty(player_count, dtype=np.int64)
    for index, cell in enumerate(cells):
        positions = np.flatnonzero(cell_index == index)
        if not positions.size:
            continue
        entity_ids = player_id[positions]
        age_draw = _uniform(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.AGE_WITHIN_CELL,
            index,
        )
        age_width = cell.age_max_exclusive - cell.age_min_inclusive
        age_offset = np.minimum(
            np.floor(age_draw * age_width).astype(np.int64),
            age_width - 1,
        )
        age_years[positions] = (
            cell.age_min_inclusive + age_offset
        ).astype(np.int16)

        income_draw = _uniform(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.MONTHLY_DISPOSABLE_INCOME,
            index,
        )
        income_width = (
            cell.monthly_disposable_income_max_cents_exclusive
            - cell.monthly_disposable_income_min_cents
        )
        income_offset = np.minimum(
            np.floor(income_draw * income_width).astype(np.int64),
            income_width - 1,
        )
        monthly_income[positions] = (
            cell.monthly_disposable_income_min_cents + income_offset
        )

    household_id = np.empty(player_count, dtype=np.int64)
    household_liquidity = np.empty(player_count, dtype=np.int64)
    cell_group_key = tuple(
        (
            cell.jurisdiction_code,
            cell.monthly_disposable_income_band_id,
            cell.household_type,
        )
        for cell in cells
    )
    household_offset = 0
    for group_index, key in enumerate(sorted(set(cell_group_key))):
        group_cell_indices = np.asarray(
            [index for index, candidate in enumerate(cell_group_key) if candidate == key],
            dtype=np.int32,
        )
        positions = np.flatnonzero(np.isin(cell_index, group_cell_indices))
        if not positions.size:
            continue
        size, _income_min, _income_max = group_specification[key]
        entity_ids = player_id[positions]
        household_draw = _uniform(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.HOUSEHOLD_ORDER,
            group_index,
        )
        shuffled_positions = positions[np.lexsort((entity_ids, household_draw))]
        local_household = np.arange(positions.size, dtype=np.int64) // size
        number_households = int(local_household[-1]) + 1
        household_id[shuffled_positions] = household_offset + local_household

        household_income = np.bincount(
            local_household,
            weights=monthly_income[shuffled_positions].astype(np.float64),
            minlength=number_households,
        )
        household_entities = np.arange(
            household_offset,
            household_offset + number_households,
            dtype=np.int64,
        )
        country_index = code_to_index[key[0]]
        profile = country_profiles[country_index]
        resource_noise = _normal(
            rng,
            household_entities,
            tick,
            _ProjectionStream.HOUSEHOLD_RESOURCES,
            group_index,
        )
        resources = _money(
            household_income
            * profile.household_liquidity_months
            * np.exp(np.clip(0.65 * resource_noise, -4.0, 4.0))
        )
        household_liquidity[shuffled_positions] = resources[local_household]
        household_offset += number_households

    adult_ages = np.asarray(
        [profile.adult_age for profile in country_profiles], dtype=np.int16
    )
    is_minor = age_years < adult_ages[jurisdiction]
    allowance = np.where(is_minor, monthly_income, 0).astype(np.int64)
    liquidity = np.empty(player_count, dtype=np.int64)
    credit_limit = np.zeros(player_count, dtype=np.int64)
    stored_payment = np.zeros(player_count, dtype=np.bool_)
    guardian_supervision = np.zeros(player_count, dtype=np.float32)
    guardian_consent = np.zeros(player_count, dtype=np.bool_)
    traits = np.empty((player_count, len(TRAIT_NAMES)), dtype=np.float32)
    motive_weights = np.empty((player_count, len(Motive)), dtype=np.float32)
    awareness = np.empty(player_count, dtype=np.float32)

    for country_index, profile in enumerate(country_profiles):
        positions = np.flatnonzero(jurisdiction == country_index)
        if not positions.size:
            continue
        entity_ids = player_id[positions]
        minor = is_minor[positions]
        income_for_players = monthly_income[positions]

        liquidity_noise = _normal(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.LIQUIDITY,
            0,
        )
        personal_months = profile.personal_liquidity_months * np.exp(
            np.clip(0.55 * liquidity_noise, -4.0, 4.0)
        )
        minor_balance_fraction = 0.25 + 1.75 * _uniform(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.LIQUIDITY,
            1,
        )
        liquidity[positions] = _money(
            np.where(
                minor,
                income_for_players * minor_balance_fraction,
                income_for_players * personal_months,
            )
        )

        credit_access = (
            _uniform(
                rng,
                entity_ids,
                tick,
                _ProjectionStream.CREDIT_ACCESS,
                0,
            )
            < profile.credit_access_probability
        ) & ~minor
        credit_noise = _normal(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.CREDIT_LIMIT,
            0,
        )
        credit_values = _money(
            income_for_players
            * profile.credit_limit_income_multiple
            * np.exp(np.clip(0.30 * credit_noise, -2.0, 2.0))
        )
        credit_limit[positions] = np.where(credit_access, credit_values, 0)

        payment_draw = _uniform(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.PAYMENT_ACCESS,
            0,
        )
        stored_payment[positions] = np.where(
            minor,
            payment_draw < profile.minor_stored_card_probability,
            payment_draw < profile.adult_stored_payment_probability,
        )
        guardian_consent[positions] = minor & (
            _uniform(
                rng,
                entity_ids,
                tick,
                _ProjectionStream.GUARDIAN_CONSENT,
                0,
            )
            < profile.minor_guardian_consent_probability
        )
        supervision_noise = _normal(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.SUPERVISION,
            0,
        )
        supervision = _sigmoid(
            _logit(profile.guardian_supervision_mean) + 0.85 * supervision_noise
        )
        guardian_supervision[positions] = np.where(
            minor,
            supervision,
            0.0,
        ).astype(np.float32)

        profile_traits = _sample_projected_traits(entity_ids, profile, rng, tick)
        traits[positions] = profile_traits
        motive_weights[positions] = _sample_projected_motives(
            entity_ids,
            profile,
            profile_traits,
            rng,
            tick,
        )
        literacy = profile_traits[:, TRAIT_NAMES.index("financial_literacy")]
        self_control = profile_traits[:, TRAIT_NAMES.index("self_control")]
        awareness_noise = _normal(
            rng,
            entity_ids,
            tick,
            _ProjectionStream.AWARENESS,
            0,
        )
        awareness[positions] = _sigmoid(
            _logit(profile.awareness_mean)
            + 0.9 * (literacy - 0.5)
            + 0.35 * (self_control - 0.5)
            + 0.55 * awareness_noise
        ).astype(np.float32)

    baseline = _baseline_vulnerability(
        age_years,
        is_minor,
        monthly_income,
        jurisdiction,
        country_profiles,
        traits,
    )
    assignment = ProjectedPopulationAssignment(
        metadata=metadata,
        cell_index=cell_index,
        assignment_sha256=projected_population_assignment_sha256(
            metadata,
            player_id,
            cell_index,
        ),
    )
    return PlayerTable(
        player_id=player_id,
        age_years=age_years,
        jurisdiction=jurisdiction,
        household_id=household_id,
        is_minor=is_minor,
        monthly_disposable_income_cents=monthly_income,
        liquidity_cents=liquidity,
        credit_limit_cents=credit_limit,
        allowance_cents=allowance,
        household_liquidity_cents=household_liquidity,
        has_stored_payment_access=stored_payment,
        guardian_supervision=guardian_supervision,
        guardian_consent=guardian_consent,
        traits=traits,
        motive_weights=motive_weights,
        baseline_vulnerability=baseline,
        harm_state=np.zeros((player_count, len(HarmDimension)), dtype=np.float32),
        current_game=np.full(player_count, -1, dtype=np.int32),
        awareness=awareness,
        jurisdiction_codes=codes,
        adult_age_by_jurisdiction=tuple(
            profile.adult_age for profile in country_profiles
        ),
        projected_population=assignment,
    )


def _hamilton_cell_counts(
    player_count: int,
    cells: Sequence[PopulationProjectionCell],
    masses: Sequence[Fraction],
) -> tuple[int, ...]:
    quotas = tuple(player_count * mass for mass in masses)
    floors = [quota.numerator // quota.denominator for quota in quotas]
    remaining = player_count - sum(floors)
    remainder_order = sorted(
        range(len(cells)),
        key=lambda index: (
            -(quotas[index] - floors[index]),
            cells[index].cell_id,
        ),
    )
    for index in remainder_order[:remaining]:
        floors[index] += 1
    if sum(floors) != player_count:
        raise RuntimeError("Hamilton allocation did not preserve player_count")
    for mass, count in zip(masses, floors, strict=True):
        if mass == 0 and count != 0:
            raise RuntimeError("Hamilton allocation assigned a zero-mass cell")
    return tuple(floors)


def initialize_players(
    player_count: int,
    country_profiles: Sequence[CountryProfile],
    rng: CounterRNGLike,
    *,
    tick: int = 0,
    first_player_id: int = 0,
) -> PlayerTable:
    """Compatibility alias for :func:`initialize_player_table`."""

    return initialize_player_table(
        player_count,
        country_profiles,
        rng,
        tick=tick,
        first_player_id=first_player_id,
    )


def _sample_jurisdictions(
    player_ids: NDArray[np.int64],
    profiles: Sequence[CountryProfile],
    rng: CounterRNGLike,
    tick: int,
) -> NDArray[np.int16]:
    weights = np.asarray([profile.population_weight for profile in profiles], dtype=np.float64)
    cumulative = np.cumsum(weights / weights.sum())
    cumulative[-1] = 1.0
    draws = _uniform(rng, player_ids, tick, _Stream.JURISDICTION, 0)
    return np.searchsorted(cumulative, draws, side="right").astype(np.int16)


def _sample_ages(
    entity_ids: NDArray[np.int64],
    profile: CountryProfile,
    rng: CounterRNGLike,
    tick: int,
) -> NDArray[np.int16]:
    weights = np.asarray(profile.age_band_weights, dtype=np.float64)
    cumulative = np.cumsum(weights / weights.sum())
    cumulative[-1] = 1.0
    band_draw = _uniform(rng, entity_ids, tick, _Stream.AGE_BAND, 0)
    bands = np.searchsorted(cumulative, band_draw, side="right")
    edges = np.asarray(profile.age_band_edges, dtype=np.int16)
    low = edges[bands]
    width = edges[bands + 1] - low
    within = _uniform(rng, entity_ids, tick, _Stream.AGE_WITHIN_BAND, 0)
    return (low + np.minimum((within * width).astype(np.int16), width - 1)).astype(
        np.int16
    )


def _sample_traits(
    entity_ids: NDArray[np.int64],
    profile: CountryProfile,
    rng: CounterRNGLike,
    tick: int,
) -> NDArray[np.float32]:
    independent = np.column_stack(
        [
            _normal(rng, entity_ids, tick, _Stream.TRAITS, draw_index)
            for draw_index in range(len(TRAIT_NAMES))
        ]
    )
    cholesky = np.linalg.cholesky(
        np.asarray(profile.trait_correlation, dtype=np.float64)
    )
    correlated = independent @ cholesky.T
    means = np.asarray(profile.trait_means, dtype=np.float64)
    scales = np.asarray(profile.trait_scales, dtype=np.float64)
    return np.clip(means + correlated * scales, 0.0, 1.0).astype(np.float32)


def _sample_motives(
    entity_ids: NDArray[np.int64],
    profile: CountryProfile,
    traits: NDArray[np.float32],
    rng: CounterRNGLike,
    tick: int,
) -> NDArray[np.float32]:
    noise = np.column_stack(
        [
            _normal(rng, entity_ids, tick, _Stream.MOTIVES, draw_index)
            for draw_index in range(len(Motive))
        ]
    )
    impulsivity, reward, social, loss_aversion, literacy, self_control = traits.T
    logits = np.broadcast_to(
        np.asarray(profile.motive_logits, dtype=np.float64),
        (entity_ids.size, len(Motive)),
    ).copy()
    logits[:, Motive.COMPETITION] += 0.75 * reward + 0.25 * self_control
    logits[:, Motive.COLLECTION] += 0.65 * reward + 0.45 * loss_aversion
    logits[:, Motive.SOCIAL] += 1.05 * social + 0.15 * impulsivity
    logits[:, Motive.EXPLORATION] += 0.45 * reward + 0.30 * literacy
    logits[:, Motive.RELAXATION] += 0.55 * self_control - 0.30 * impulsivity
    logits += 0.45 * noise
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def _sample_projected_traits(
    entity_ids: NDArray[np.int64],
    profile: CountryProfile,
    rng: CounterRNGLike,
    tick: int,
) -> NDArray[np.float32]:
    independent = np.column_stack(
        [
            _normal(rng, entity_ids, tick, _ProjectionStream.TRAITS, draw_index)
            for draw_index in range(len(TRAIT_NAMES))
        ]
    )
    cholesky = np.linalg.cholesky(
        np.asarray(profile.trait_correlation, dtype=np.float64)
    )
    correlated = independent @ cholesky.T
    means = np.asarray(profile.trait_means, dtype=np.float64)
    scales = np.asarray(profile.trait_scales, dtype=np.float64)
    return np.clip(means + correlated * scales, 0.0, 1.0).astype(np.float32)


def _sample_projected_motives(
    entity_ids: NDArray[np.int64],
    profile: CountryProfile,
    traits: NDArray[np.float32],
    rng: CounterRNGLike,
    tick: int,
) -> NDArray[np.float32]:
    noise = np.column_stack(
        [
            _normal(rng, entity_ids, tick, _ProjectionStream.MOTIVES, draw_index)
            for draw_index in range(len(Motive))
        ]
    )
    impulsivity, reward, social, loss_aversion, literacy, self_control = traits.T
    logits = np.broadcast_to(
        np.asarray(profile.motive_logits, dtype=np.float64),
        (entity_ids.size, len(Motive)),
    ).copy()
    logits[:, Motive.COMPETITION] += 0.75 * reward + 0.25 * self_control
    logits[:, Motive.COLLECTION] += 0.65 * reward + 0.45 * loss_aversion
    logits[:, Motive.SOCIAL] += 1.05 * social + 0.15 * impulsivity
    logits[:, Motive.EXPLORATION] += 0.45 * reward + 0.30 * literacy
    logits[:, Motive.RELAXATION] += 0.55 * self_control - 0.30 * impulsivity
    logits += 0.45 * noise
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.astype(np.float32)


def _baseline_vulnerability(
    ages: NDArray[np.int16],
    minors: NDArray[np.bool_],
    income: NDArray[np.int64],
    jurisdiction: NDArray[np.int16],
    profiles: Sequence[CountryProfile],
    traits: NDArray[np.float32],
) -> NDArray[np.float32]:
    impulsivity, reward, social, _loss_aversion, literacy, self_control = traits.T
    country_median = np.asarray(
        [profile.monthly_income_median_cents for profile in profiles], dtype=np.float64
    )[jurisdiction]
    income_ratio = income.astype(np.float64) / np.maximum(country_median, 1.0)
    income_strain = np.clip(1.0 - income_ratio, 0.0, 1.0)
    youth_gradient = np.clip((25.0 - ages.astype(np.float64)) / 20.0, 0.0, 1.0)
    latent = (
        -1.35
        + 1.35 * impulsivity
        + 0.90 * reward
        + 0.40 * social
        - 1.10 * self_control
        - 0.55 * literacy
        + 0.35 * income_strain
        + 0.35 * youth_gradient
        + 0.15 * minors
    )
    return _sigmoid(latent).astype(np.float32)


def _uniform(
    rng: CounterRNGLike,
    entity_ids: NDArray[np.int64],
    tick: int,
    stream: int,
    draw_index: int,
) -> NDArray[np.float64]:
    values = np.asarray(
        rng.uniform(entity_ids, tick, stream, draw_index), dtype=np.float64
    )
    if values.shape != entity_ids.shape:
        raise ValueError("CounterRNG.uniform returned an unexpected shape")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values >= 1.0):
        raise ValueError("CounterRNG.uniform must return finite values in [0, 1)")
    return values


def _normal(
    rng: CounterRNGLike,
    entity_ids: NDArray[np.int64],
    tick: int,
    stream: int,
    draw_index: int,
) -> NDArray[np.float64]:
    values = np.asarray(
        rng.normal(entity_ids, tick, stream, draw_index), dtype=np.float64
    )
    if values.shape != entity_ids.shape:
        raise ValueError("CounterRNG.normal returned an unexpected shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("CounterRNG.normal must return finite values")
    return values


def _money(values: NDArray[np.floating] | float) -> NDArray[np.int64]:
    as_float = np.asarray(values, dtype=np.float64)
    ceiling = float(np.iinfo(np.int64).max)
    return np.rint(np.clip(as_float, 0.0, ceiling)).astype(np.int64)


def _sigmoid(values: NDArray[np.floating] | float) -> NDArray[np.float64]:
    value = np.asarray(values, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _logit(probability: float) -> float:
    clipped = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))
