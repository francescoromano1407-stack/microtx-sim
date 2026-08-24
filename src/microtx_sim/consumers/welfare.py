"""Synthetic player time, commitment, habit, and wellbeing state.

The main :class:`~microtx_sim.agents.players.PlayerTable` owns demographic and
financial state.  ``PlayerLifeTable`` is an aligned structure-of-arrays table
for welfare variables that should not be inferred from spending after
treatment.  Baseline columns are copied and write-protected; dynamic columns
remain mutable NumPy arrays for a later transition system.

All initial distributions below are illustrative research priors.  They are
deliberately simple, bounded, and keyed by the counter-based RNG.  They are not
empirical prevalence estimates or clinical measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..agents.players import PlayerTable
from ..rng import CounterRNG
from ..types import Motive


Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]
Float32Array = NDArray[np.float32]


_BASELINE_COLUMNS: Final[tuple[str, ...]] = (
    "player_id",
    "planned_leisure_minutes",
    "sleep_need_minutes",
    "work_study_obligation_minutes",
    "social_obligation_minutes",
    "physical_activity_need_minutes",
    "baseline_game_enjoyment",
    "financial_sensitivity",
    "delay_discounting",
    "social_pressure_susceptibility",
    "scarcity_fomo_susceptibility",
    "baseline_vulnerability",
    "intended_spending_limit_cents",
    "intended_play_minutes",
)

_PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "baseline_game_enjoyment",
    "financial_sensitivity",
    "delay_discounting",
    "social_pressure_susceptibility",
    "scarcity_fomo_susceptibility",
    "baseline_vulnerability",
    "current_game_progression",
    "habit_strength",
    "wellbeing",
)


@dataclass(frozen=True, slots=True)
class PlayerLifeTable:
    """Columnar welfare state aligned one-for-one with ``PlayerTable``.

    Minutes without a period suffix are daily baseline allocations.  The
    dynamic ``actual_play_minutes`` and ``historical_spending_cents`` columns
    are cumulative totals.  ``reinforcement_state`` is a signed prediction-
    error/habit signal in ``[-1, 1]``; it is not a clinical construct.
    """

    player_id: Int64Array

    planned_leisure_minutes: Int32Array
    sleep_need_minutes: Int32Array
    work_study_obligation_minutes: Int32Array
    social_obligation_minutes: Int32Array
    physical_activity_need_minutes: Int32Array

    baseline_game_enjoyment: Float32Array
    financial_sensitivity: Float32Array
    delay_discounting: Float32Array
    social_pressure_susceptibility: Float32Array
    scarcity_fomo_susceptibility: Float32Array
    baseline_vulnerability: Float32Array

    intended_spending_limit_cents: Int64Array
    intended_play_minutes: Int32Array

    sleep_debt_minutes: Int32Array
    current_game_progression: Float32Array
    habit_strength: Float32Array
    reinforcement_state: Float32Array
    historical_spending_cents: Int64Array
    actual_play_minutes: Int64Array
    wellbeing: Float32Array

    def __post_init__(self) -> None:
        if self.player_id.ndim != 1:
            raise ValueError("player_id must be one-dimensional")
        size = self.player_id.size
        expected_dtypes: dict[str, np.dtype[object]] = {
            "player_id": np.dtype(np.int64),
            "planned_leisure_minutes": np.dtype(np.int32),
            "sleep_need_minutes": np.dtype(np.int32),
            "work_study_obligation_minutes": np.dtype(np.int32),
            "social_obligation_minutes": np.dtype(np.int32),
            "physical_activity_need_minutes": np.dtype(np.int32),
            "baseline_game_enjoyment": np.dtype(np.float32),
            "financial_sensitivity": np.dtype(np.float32),
            "delay_discounting": np.dtype(np.float32),
            "social_pressure_susceptibility": np.dtype(np.float32),
            "scarcity_fomo_susceptibility": np.dtype(np.float32),
            "baseline_vulnerability": np.dtype(np.float32),
            "intended_spending_limit_cents": np.dtype(np.int64),
            "intended_play_minutes": np.dtype(np.int32),
            "sleep_debt_minutes": np.dtype(np.int32),
            "current_game_progression": np.dtype(np.float32),
            "habit_strength": np.dtype(np.float32),
            "reinforcement_state": np.dtype(np.float32),
            "historical_spending_cents": np.dtype(np.int64),
            "actual_play_minutes": np.dtype(np.int64),
            "wellbeing": np.dtype(np.float32),
        }
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{descriptor.name} must be a NumPy array")
            if value.ndim != 1 or value.size != size:
                raise ValueError(f"{descriptor.name} must have shape ({size},)")
            if value.dtype != expected_dtypes[descriptor.name]:
                raise TypeError(
                    f"{descriptor.name} must use dtype "
                    f"{expected_dtypes[descriptor.name]}"
                )

        if size and np.unique(self.player_id).size != size:
            raise ValueError("player_id values must be unique")
        if np.any(self.player_id < 0):
            raise ValueError("player_id cannot be negative")

        for name in (
            "planned_leisure_minutes",
            "sleep_need_minutes",
            "work_study_obligation_minutes",
            "social_obligation_minutes",
            "physical_activity_need_minutes",
            "intended_spending_limit_cents",
            "intended_play_minutes",
            "sleep_debt_minutes",
            "historical_spending_cents",
            "actual_play_minutes",
        ):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} cannot be negative")
        if np.any(self.intended_play_minutes > self.planned_leisure_minutes):
            raise ValueError("intended play cannot exceed planned leisure")

        for name in _PROBABILITY_COLUMNS:
            value = getattr(self, name)
            if not np.all(np.isfinite(value)) or np.any(
                (value < 0.0) | (value > 1.0)
            ):
                raise ValueError(f"{name} must contain finite values in [0, 1]")
        if not np.all(np.isfinite(self.reinforcement_state)) or np.any(
            (self.reinforcement_state < -1.0) | (self.reinforcement_state > 1.0)
        ):
            raise ValueError(
                "reinforcement_state must contain finite values in [-1, 1]"
            )

        # Pre-treatment commitments and traits must not drift when dynamic
        # columns are updated.  Copying also prevents writeability changes on
        # this table from affecting the source PlayerTable.
        for name in _BASELINE_COLUMNS:
            immutable = np.array(getattr(self, name), copy=True)
            immutable.flags.writeable = False
            object.__setattr__(self, name, immutable)

    def __len__(self) -> int:
        return int(self.player_id.size)

    def validate_alignment(self, players: PlayerTable) -> None:
        """Fail if this welfare table is not aligned with ``players``."""

        if not isinstance(players, PlayerTable):
            raise TypeError("players must be a PlayerTable")
        if not np.array_equal(self.player_id, players.player_id):
            raise ValueError("PlayerLifeTable is not aligned with PlayerTable")

    @property
    def nbytes(self) -> int:
        """Return memory owned by the table's NumPy columns."""

        return sum(getattr(self, item.name).nbytes for item in fields(self))


def initialize_player_life(
    players: PlayerTable,
    rng: CounterRNG,
    *,
    tick: int = 0,
) -> PlayerLifeTable:
    """Create deterministic illustrative welfare state for ``players``.

    Continuous traits use clipped normal or affine synthetic priors.  Time
    allocations use rounded clipped normals conditional on age and obligations.
    Spending commitments are fractions of the existing disposable-budget field.
    Each draw has a named counter stream, so population ordering and unrelated
    random calls cannot change the result.
    """

    if not isinstance(players, PlayerTable):
        raise TypeError("players must be a PlayerTable")
    if not isinstance(rng, CounterRNG):
        raise TypeError("rng must be a CounterRNG")
    if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)):
        raise TypeError("tick must be an integer")
    if tick < 0:
        raise ValueError("tick cannot be negative")

    player_id = players.player_id
    age = players.age_years.astype(np.float64)
    minor = players.is_minor
    older_adult = age >= 65.0
    reward = players.trait("reward_sensitivity").astype(np.float64)
    social = players.trait("social_susceptibility").astype(np.float64)
    loss_aversion = players.trait("loss_aversion").astype(np.float64)
    literacy = players.trait("financial_literacy").astype(np.float64)
    impulsivity = players.trait("impulsivity").astype(np.float64)
    self_control = players.trait("self_control").astype(np.float64)
    relaxation = players.motive(Motive.RELAXATION).astype(np.float64)
    collection = players.motive(Motive.COLLECTION).astype(np.float64)

    work_centre = np.where(
        minor,
        np.where(age < 13.0, 330.0, 390.0),
        np.where(older_adult, 90.0, 420.0),
    )
    work_study = _rounded_minutes(
        work_centre
        + rng.normal(player_id, tick, "player-life:work-study", 0, scale=75.0),
        minimum=0,
        maximum=600,
    )
    planned_leisure = _rounded_minutes(
        285.0
        - 0.27 * work_study.astype(np.float64)
        + 35.0 * minor.astype(np.float64)
        + rng.normal(player_id, tick, "player-life:leisure", 0, scale=48.0),
        minimum=60,
        maximum=480,
    )
    sleep_centre = np.where(age < 13.0, 600.0, np.where(minor, 540.0, 480.0))
    sleep_need = _rounded_minutes(
        sleep_centre
        + rng.normal(player_id, tick, "player-life:sleep-need", 0, scale=32.0),
        minimum=390,
        maximum=660,
    )
    social_obligation = _rounded_minutes(
        42.0
        + 78.0 * social
        + rng.normal(player_id, tick, "player-life:social-obligation", 0, scale=24.0),
        minimum=15,
        maximum=240,
    )
    physical_need = _rounded_minutes(
        38.0
        + 18.0 * minor.astype(np.float64)
        + rng.normal(player_id, tick, "player-life:physical-need", 0, scale=13.0),
        minimum=15,
        maximum=120,
    )

    enjoyment = np.clip(
        0.34
        + 0.24 * reward
        + 0.22 * relaxation
        + rng.normal(player_id, tick, "player-life:enjoyment", 0, scale=0.11),
        0.0,
        1.0,
    )
    liquidity_months = np.divide(
        players.liquidity_cents.astype(np.float64),
        np.maximum(players.monthly_disposable_income_cents.astype(np.float64), 1.0),
    )
    resource_strain = 1.0 - np.clip(liquidity_months / 1.5, 0.0, 1.0)
    financial_sensitivity = np.clip(
        0.15 + 0.42 * literacy + 0.25 * loss_aversion + 0.25 * resource_strain,
        0.0,
        1.0,
    )
    delay_discounting = np.clip(
        0.10 + 0.55 * impulsivity + 0.25 * reward - 0.30 * self_control,
        0.0,
        1.0,
    )
    fomo = np.clip(
        0.10 + 0.32 * loss_aversion + 0.30 * reward + 0.38 * collection,
        0.0,
        1.0,
    )
    vulnerability = players.baseline_vulnerability.astype(np.float64)

    spending_fraction = np.clip(
        0.012
        + 0.065 * (1.0 - financial_sensitivity)
        + 0.018 * self_control
        + 0.020 * minor.astype(np.float64),
        0.005,
        0.12,
    )
    intended_limit = _rounded_money(
        players.monthly_disposable_income_cents.astype(np.float64)
        * spending_fraction
    )
    intended_play = _rounded_minutes(
        planned_leisure.astype(np.float64)
        * np.clip(0.18 + 0.48 * enjoyment + 0.12 * reward, 0.10, 0.85)
        + rng.normal(player_id, tick, "player-life:intended-play", 0, scale=18.0),
        minimum=0,
        maximum=480,
    )
    intended_play = np.minimum(intended_play, planned_leisure).astype(np.int32)

    sleep_debt = _rounded_minutes(
        12.0
        + 70.0 * vulnerability
        + rng.normal(player_id, tick, "player-life:sleep-debt", 0, scale=38.0),
        minimum=0,
        maximum=480,
    )
    progression = np.where(
        players.current_game >= 0,
        np.power(
            rng.uniform(player_id, tick, "player-life:progression", 0), 1.8
        )
        * 0.35,
        0.0,
    )
    habit = np.clip(
        0.08
        + 0.30 * vulnerability
        + 0.25 * enjoyment
        + 0.15 * delay_discounting
        + rng.normal(player_id, tick, "player-life:habit", 0, scale=0.08),
        0.0,
        1.0,
    )
    reinforcement = np.zeros(len(players), dtype=np.float32)
    wellbeing = np.clip(
        0.86
        - 0.24 * vulnerability
        - 0.18 * (sleep_debt.astype(np.float64) / 480.0)
        + rng.normal(player_id, tick, "player-life:wellbeing", 0, scale=0.06),
        0.0,
        1.0,
    )

    table = PlayerLifeTable(
        player_id=np.asarray(player_id, dtype=np.int64),
        planned_leisure_minutes=planned_leisure,
        sleep_need_minutes=sleep_need,
        work_study_obligation_minutes=work_study,
        social_obligation_minutes=social_obligation,
        physical_activity_need_minutes=physical_need,
        baseline_game_enjoyment=enjoyment.astype(np.float32),
        financial_sensitivity=financial_sensitivity.astype(np.float32),
        delay_discounting=delay_discounting.astype(np.float32),
        social_pressure_susceptibility=social.astype(np.float32),
        scarcity_fomo_susceptibility=fomo.astype(np.float32),
        baseline_vulnerability=vulnerability.astype(np.float32),
        intended_spending_limit_cents=intended_limit,
        intended_play_minutes=intended_play,
        sleep_debt_minutes=sleep_debt,
        current_game_progression=progression.astype(np.float32),
        habit_strength=habit.astype(np.float32),
        reinforcement_state=reinforcement,
        historical_spending_cents=np.zeros(len(players), dtype=np.int64),
        actual_play_minutes=np.zeros(len(players), dtype=np.int64),
        wellbeing=wellbeing.astype(np.float32),
    )
    table.validate_alignment(players)
    return table


def _rounded_minutes(
    values: NDArray[np.floating] | float,
    *,
    minimum: int,
    maximum: int,
) -> Int32Array:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("minute prior produced a non-finite value")
    return np.rint(np.clip(array, minimum, maximum)).astype(np.int32)


def _rounded_money(values: NDArray[np.floating] | float) -> Int64Array:
    array = np.asarray(values, dtype=np.float64)
    maximum = float(np.iinfo(np.int64).max)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > maximum):
        raise OverflowError("spending commitment is outside int64 cents")
    return np.rint(array).astype(np.int64)


__all__ = ["PlayerLifeTable", "initialize_player_life"]
