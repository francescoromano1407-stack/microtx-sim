"""Interpretable discrete choices among gaming and alternative activities.

Each time step evaluates the full action set for every synthetic player.  A
counter-based Gumbel shock turns the documented utilities into a multinomial
logit choice while preserving common random numbers across counterfactuals.
The equations are policy-research assumptions, not empirical estimates and not
instructions for optimising engagement or spending.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite

import numpy as np
import numpy.typing as npt

from ..agents.players import PlayerTable
from ..domain.monetisation import MonetisationVector
from ..rng import CounterRNG
from .welfare import PlayerLifeTable


class LifeAction(IntEnum):
    """Mutually exclusive uses of one simulated time step."""

    PLAY = 0
    PURCHASE = 1
    STOP = 2
    SLEEP = 3
    STUDY_WORK = 4
    SOCIALIZE = 5
    EXERCISE = 6
    OTHER = 7


ACTION_NAMES = tuple(action.name.lower() for action in LifeAction)


@dataclass(frozen=True, slots=True)
class DecisionParameters:
    """Illustrative coefficients for the activity-choice equation."""

    step_minutes: int = 30
    temperature: float = 0.65
    habit_persistence: float = 0.97
    habit_learning_rate: float = 0.025
    reinforcement_learning_rate: float = 0.12

    def __post_init__(self) -> None:
        if isinstance(self.step_minutes, bool) or not isinstance(
            self.step_minutes, int
        ):
            raise TypeError("step_minutes must be an integer")
        if self.step_minutes <= 0 or 1_440 % self.step_minutes:
            raise ValueError("step_minutes must be a positive divisor of 1440")
        for name in (
            "temperature",
            "habit_persistence",
            "habit_learning_rate",
            "reinforcement_learning_rate",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.temperature > 5.0:
            raise ValueError("temperature must not exceed 5")
        if self.habit_persistence > 1.0:
            raise ValueError("habit_persistence must be in (0, 1]")
        if self.habit_learning_rate > 1.0:
            raise ValueError("habit_learning_rate must be in (0, 1]")
        if self.reinforcement_learning_rate > 1.0:
            raise ValueError("reinforcement_learning_rate must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ActionChoice:
    """One vectorised action draw and its feasible monetary charge."""

    action: npt.NDArray[np.int8]
    purchase_cents: npt.NDArray[np.int64]
    deterministic_utilities: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        size = self.action.size
        if self.action.ndim != 1 or self.action.dtype != np.dtype(np.int8):
            raise TypeError("action must be a one-dimensional int8 array")
        if np.any((self.action < 0) | (self.action >= len(LifeAction))):
            raise ValueError("action contains an unknown code")
        if self.purchase_cents.shape != (size,) or self.purchase_cents.dtype != np.int64:
            raise TypeError("purchase_cents must be a matching int64 array")
        if np.any(self.purchase_cents < 0):
            raise ValueError("purchase_cents cannot be negative")
        if self.deterministic_utilities.shape != (size, len(LifeAction)):
            raise ValueError("utility matrix has the wrong shape")
        if np.any(np.isnan(self.deterministic_utilities)):
            raise ValueError("utilities cannot contain NaN")
        if np.any(self.purchase_cents[self.action != LifeAction.PURCHASE] != 0):
            raise ValueError("only purchase actions may carry a charge")


def choose_life_action(
    players: PlayerTable,
    life: PlayerLifeTable,
    mechanics: MonetisationVector,
    rng: CounterRNG,
    *,
    tick: int,
    minute_of_day: int,
    remaining_sleep_minutes: npt.ArrayLike,
    remaining_work_study_minutes: npt.ArrayLike,
    remaining_social_minutes: npt.ArrayLike,
    remaining_physical_minutes: npt.ArrayLike,
    daily_play_minutes: npt.ArrayLike,
    available_budget_cents: npt.ArrayLike,
    cap_period_spend_cents: npt.ArrayLike,
    last_purchase_tick: npt.ArrayLike,
    parameters: DecisionParameters | None = None,
) -> ActionChoice:
    """Choose one action per player using full-set utility maximisation.

    If ``V_ia`` is the deterministic utility below and ``epsilon_ia`` is an
    independent standard Gumbel field, the selected action is

    ``argmax_a(V_ia + temperature * epsilon_ia)``.

    The purchase alternative is assigned negative infinity when affordability,
    pre-commitment safeguards, cooling-off, or guardian consent makes the
    transaction infeasible.  Thus stochastic choice cannot bypass a hard rule.
    """

    params = parameters or DecisionParameters()
    if not isinstance(params, DecisionParameters):
        raise TypeError("parameters must be DecisionParameters")
    if not isinstance(players, PlayerTable) or not isinstance(life, PlayerLifeTable):
        raise TypeError("players and life must be aligned player tables")
    life.validate_alignment(players)
    if not isinstance(mechanics, MonetisationVector):
        raise TypeError("mechanics must be a MonetisationVector")
    if not isinstance(rng, CounterRNG):
        raise TypeError("rng must be a CounterRNG")
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
        raise ValueError("tick must be a non-negative integer")
    if (
        isinstance(minute_of_day, bool)
        or not isinstance(minute_of_day, int)
        or not 0 <= minute_of_day < 1_440
    ):
        raise ValueError("minute_of_day must be an integer in [0, 1440)")

    size = len(players)
    remaining_sleep = _nonnegative_float_vector(
        remaining_sleep_minutes, size, "remaining_sleep_minutes"
    )
    remaining_work = _nonnegative_float_vector(
        remaining_work_study_minutes, size, "remaining_work_study_minutes"
    )
    remaining_social = _nonnegative_float_vector(
        remaining_social_minutes, size, "remaining_social_minutes"
    )
    remaining_physical = _nonnegative_float_vector(
        remaining_physical_minutes, size, "remaining_physical_minutes"
    )
    daily_play = _nonnegative_float_vector(
        daily_play_minutes, size, "daily_play_minutes"
    )
    available_budget = _nonnegative_int_vector(
        available_budget_cents, size, "available_budget_cents"
    )
    cap_spend = _nonnegative_int_vector(
        cap_period_spend_cents, size, "cap_period_spend_cents"
    )
    last_purchase = np.asarray(last_purchase_tick)
    if last_purchase.shape != (size,) or last_purchase.dtype.kind not in {"i", "u"}:
        raise TypeError("last_purchase_tick must be a matching integer array")
    last_purchase = last_purchase.astype(np.int64, copy=False)
    if np.any(last_purchase > tick):
        raise ValueError("last_purchase_tick cannot be in the future")

    utilities = np.zeros((size, len(LifeAction)), dtype=np.float64)
    if not size:
        return ActionChoice(
            action=np.empty(0, dtype=np.int8),
            purchase_cents=np.empty(0, dtype=np.int64),
            deterministic_utilities=utilities,
        )

    sleep_urgency = _urgency(remaining_sleep, life.sleep_need_minutes)
    work_urgency = _urgency(
        remaining_work, life.work_study_obligation_minutes
    )
    social_urgency = _urgency(remaining_social, life.social_obligation_minutes)
    physical_urgency = _urgency(
        remaining_physical, life.physical_activity_need_minutes
    )
    leisure_overrun = np.divide(
        np.maximum(daily_play - life.intended_play_minutes, 0.0),
        np.maximum(life.planned_leisure_minutes.astype(np.float64), 1.0),
    )
    hour = minute_of_day / 60.0
    night = 1.0 if hour >= 22.0 or hour < 7.0 else 0.0
    work_window = 1.0 if 8.0 <= hour < 18.0 else 0.0
    social_window = 1.0 if 17.0 <= hour < 23.0 else 0.0
    daylight = 1.0 if 7.0 <= hour < 21.0 else 0.0

    mechanics_play_pull = (
        0.35 * mechanics.progression_gates * (1.0 - life.current_game_progression)
        + 0.35
        * mechanics.daily_streak_pressure
        * life.scarcity_fomo_susceptibility
        + 0.30
        * mechanics.social_guild_pressure
        * life.social_pressure_susceptibility
    )
    utilities[:, LifeAction.PLAY] = (
        -0.45
        + 1.45 * life.baseline_game_enjoyment
        + 1.05 * life.habit_strength
        + 0.35 * life.reinforcement_state
        + mechanics_play_pull
        - 0.80 * sleep_urgency
        - 0.60 * work_urgency
        - 0.65 * leisure_overrun
    )

    price = mechanics.direct_price_cents
    price_burden = np.divide(
        float(price),
        np.maximum(players.monthly_disposable_income_cents.astype(np.float64), 1.0),
    )
    utilities[:, LifeAction.PURCHASE] = (
        -2.20
        + 1.70 * mechanics.purchase_pressure * life.delay_discounting
        + 0.80 * life.baseline_vulnerability
        + 0.55 * life.habit_strength
        + 0.45 * mechanics.paid_random_rewards
        + 0.35 * mechanics.pay_to_progress
        + 0.25 * mechanics.pay_to_win
        - 1.35 * mechanics.purchase_friction
        - 4.00 * price_burden * (0.5 + life.financial_sensitivity)
    )
    self_control = players.trait("self_control").astype(np.float64)
    utilities[:, LifeAction.STOP] = (
        0.20
        + 0.70 * self_control
        + 0.35 * sleep_urgency
        + 0.25 * leisure_overrun
        - 0.30 * life.habit_strength
    )
    utilities[:, LifeAction.SLEEP] = (
        -0.30 + 1.85 * sleep_urgency + 1.15 * night - 0.40 * daylight
    )
    utilities[:, LifeAction.STUDY_WORK] = (
        -0.20 + 1.95 * work_urgency + 1.05 * work_window
    )
    utilities[:, LifeAction.SOCIALIZE] = (
        -0.05 + 1.45 * social_urgency + 0.55 * social_window
    )
    utilities[:, LifeAction.EXERCISE] = (
        -0.10 + 1.35 * physical_urgency + 0.35 * daylight
    )
    utilities[:, LifeAction.OTHER] = 0.35 + 0.15 * (1.0 - work_window)

    purchase_allowed = np.full(size, price > 0, dtype=np.bool_)
    purchase_allowed &= available_budget >= price
    if mechanics.spending_cap_cents is not None:
        purchase_allowed &= cap_spend <= mechanics.spending_cap_cents - price
    if mechanics.cooling_off_hours:
        cooling_steps = int(
            np.ceil(mechanics.cooling_off_hours * 60 / params.step_minutes)
        )
        purchase_allowed &= (last_purchase < 0) | (
            tick - last_purchase >= cooling_steps
        )
    purchase_allowed &= (~players.is_minor) | (
        players.has_stored_payment_access & players.guardian_consent
    )
    utilities[~purchase_allowed, LifeAction.PURCHASE] = -np.inf

    stochastic = utilities.copy()
    for action in LifeAction:
        uniforms = rng.uniform(
            players.player_id,
            tick,
            "policy:life-action-gumbel",
            int(action),
        )
        uniforms = np.clip(uniforms, 1e-15, 1.0 - 1e-15)
        shock = -np.log(-np.log(uniforms))
        stochastic[:, int(action)] += params.temperature * shock
    action = np.argmax(stochastic, axis=1).astype(np.int8)
    purchase = np.where(action == LifeAction.PURCHASE, price, 0).astype(np.int64)
    return ActionChoice(
        action=action,
        purchase_cents=purchase,
        deterministic_utilities=utilities,
    )


def _urgency(
    remaining: npt.NDArray[np.float64],
    baseline: npt.NDArray[np.integer],
) -> npt.NDArray[np.float64]:
    return np.clip(
        np.divide(
            remaining,
            np.maximum(baseline.astype(np.float64), 1.0),
        ),
        0.0,
        2.0,
    )


def _nonnegative_float_vector(
    values: npt.ArrayLike, size: int, name: str
) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be a finite non-negative ({size},) vector")
    return array


def _nonnegative_int_vector(
    values: npt.ArrayLike, size: int, name: str
) -> npt.NDArray[np.int64]:
    array = np.asarray(values)
    if array.shape != (size,) or array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must be a ({size},) integer vector")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise OverflowError(f"{name} exceeds int64")
    result = array.astype(np.int64, copy=False)
    if np.any(result < 0):
        raise ValueError(f"{name} cannot be negative")
    return result


__all__ = [
    "ACTION_NAMES",
    "ActionChoice",
    "DecisionParameters",
    "LifeAction",
    "choose_life_action",
]
