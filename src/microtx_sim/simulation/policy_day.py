"""One simulated day of competing activities and purchase opportunities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..agents.players import PlayerTable
from ..consumers.decision import (
    DecisionParameters,
    LifeAction,
    choose_life_action,
)
from ..consumers.welfare import PlayerLifeTable
from ..domain.monetisation import MonetisationVector
from ..rng import CounterRNG


PURCHASE_REVENUE_SOURCES = (
    "direct_purchase",
    "opaque_virtual_currency",
    "paid_random_rewards",
)


@dataclass(slots=True)
class PolicyState:
    """Mutable scenario state; baseline traits remain in immutable tables."""

    life: PlayerLifeTable
    available_budget_cents: npt.NDArray[np.int64]
    cap_period_spend_cents: npt.NDArray[np.int64]
    last_purchase_tick: npt.NDArray[np.int64]
    player_spend_cents: npt.NDArray[np.int64]
    player_spend_by_source_cents: npt.NDArray[np.int64]
    access_fixed_cents: npt.NDArray[np.int64]
    access_subscription_cents: npt.NDArray[np.int64]
    action_minutes: npt.NDArray[np.int64]
    cumulative_enjoyment: npt.NDArray[np.float64]
    initial_wellbeing: npt.NDArray[np.float32]
    completed_days: int = 0

    def __post_init__(self) -> None:
        size = len(self.life)
        one_dimensional = (
            "available_budget_cents",
            "cap_period_spend_cents",
            "last_purchase_tick",
            "player_spend_cents",
            "access_fixed_cents",
            "access_subscription_cents",
            "cumulative_enjoyment",
            "initial_wellbeing",
        )
        for name in one_dimensional:
            if getattr(self, name).shape != (size,):
                raise ValueError(f"{name} must have shape ({size},)")
        if self.player_spend_by_source_cents.shape != (
            size,
            len(PURCHASE_REVENUE_SOURCES),
        ):
            raise ValueError("player_spend_by_source_cents has the wrong shape")
        if self.action_minutes.shape != (size, len(LifeAction)):
            raise ValueError("action_minutes has the wrong shape")
        for name in (
            "available_budget_cents",
            "cap_period_spend_cents",
            "last_purchase_tick",
            "player_spend_cents",
            "player_spend_by_source_cents",
            "access_fixed_cents",
            "access_subscription_cents",
            "action_minutes",
        ):
            if getattr(self, name).dtype != np.dtype(np.int64):
                raise TypeError(f"{name} must use int64")
        if self.cumulative_enjoyment.dtype != np.dtype(np.float64):
            raise TypeError("cumulative_enjoyment must use float64")
        if self.initial_wellbeing.dtype != np.dtype(np.float32):
            raise TypeError("initial_wellbeing must use float32")
        for name in (
            "available_budget_cents",
            "cap_period_spend_cents",
            "player_spend_cents",
            "player_spend_by_source_cents",
            "access_fixed_cents",
            "access_subscription_cents",
            "action_minutes",
            "cumulative_enjoyment",
        ):
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} cannot contain negative values")
        if self.completed_days < 0:
            raise ValueError("completed_days cannot be negative")


@dataclass(frozen=True, slots=True)
class PolicyDayResult:
    """Auditable time, money, deficit, and enjoyment outputs for one day."""

    day: int
    action_minutes: npt.NDArray[np.int64]
    spend_cents: npt.NDArray[np.int64]
    spend_by_source_cents: npt.NDArray[np.int64]
    sleep_minutes: npt.NDArray[np.int64]
    work_study_minutes: npt.NDArray[np.int64]
    social_minutes: npt.NDArray[np.int64]
    physical_minutes: npt.NDArray[np.int64]
    sleep_shortfall_minutes: npt.NDArray[np.int64]
    work_study_shortfall_minutes: npt.NDArray[np.int64]
    social_shortfall_minutes: npt.NDArray[np.int64]
    physical_shortfall_minutes: npt.NDArray[np.int64]
    enjoyment: npt.NDArray[np.float64]


def create_policy_state(players: PlayerTable, life: PlayerLifeTable) -> PolicyState:
    """Create a fresh mutable state for one counterfactual branch."""

    life.validate_alignment(players)
    size = len(players)
    return PolicyState(
        life=life,
        available_budget_cents=players.monthly_disposable_income_cents.copy(),
        cap_period_spend_cents=np.zeros(size, dtype=np.int64),
        last_purchase_tick=np.full(size, -1, dtype=np.int64),
        player_spend_cents=np.zeros(size, dtype=np.int64),
        player_spend_by_source_cents=np.zeros(
            (size, len(PURCHASE_REVENUE_SOURCES)), dtype=np.int64
        ),
        access_fixed_cents=np.zeros(size, dtype=np.int64),
        access_subscription_cents=np.zeros(size, dtype=np.int64),
        action_minutes=np.zeros((size, len(LifeAction)), dtype=np.int64),
        cumulative_enjoyment=np.zeros(size, dtype=np.float64),
        initial_wellbeing=life.wellbeing.copy(),
    )


def advance_policy_day(
    players: PlayerTable,
    state: PolicyState,
    mechanics: MonetisationVector,
    rng: CounterRNG,
    *,
    day: int,
    parameters: DecisionParameters | None = None,
) -> PolicyDayResult:
    """Advance one exact 1,440-minute allocation day.

    Every player receives the same number of decision opportunities, and each
    opportunity evaluates every action.  Complexity is ``O(P * A * T)`` with
    eight fixed actions and ``T = 1440 / step_minutes``; no alternatives are
    sampled away.
    """

    params = parameters or DecisionParameters()
    if day != state.completed_days:
        raise ValueError("days must be advanced consecutively from zero")
    state.life.validate_alignment(players)
    size = len(players)
    step_minutes = params.step_minutes
    steps_per_day = 1_440 // step_minutes
    if day and day % 30 == 0:
        state.available_budget_cents[:] = players.monthly_disposable_income_cents
        state.cap_period_spend_cents.fill(0)

    daily_action = np.zeros((size, len(LifeAction)), dtype=np.int64)
    daily_spend = np.zeros(size, dtype=np.int64)
    daily_sources = np.zeros(
        (size, len(PURCHASE_REVENUE_SOURCES)), dtype=np.int64
    )
    remaining_sleep = (
        state.life.sleep_need_minutes.astype(np.float64)
        + np.minimum(state.life.sleep_debt_minutes, 120).astype(np.float64)
    )
    remaining_work = state.life.work_study_obligation_minutes.astype(np.float64)
    remaining_social = state.life.social_obligation_minutes.astype(np.float64)
    remaining_physical = state.life.physical_activity_need_minutes.astype(np.float64)
    daily_play = np.zeros(size, dtype=np.float64)

    for within_day_step in range(steps_per_day):
        global_tick = day * steps_per_day + within_day_step
        choice = choose_life_action(
            players,
            state.life,
            mechanics,
            rng,
            tick=global_tick,
            minute_of_day=within_day_step * step_minutes,
            remaining_sleep_minutes=remaining_sleep,
            remaining_work_study_minutes=remaining_work,
            remaining_social_minutes=remaining_social,
            remaining_physical_minutes=remaining_physical,
            daily_play_minutes=daily_play,
            available_budget_cents=state.available_budget_cents,
            cap_period_spend_cents=state.cap_period_spend_cents,
            last_purchase_tick=state.last_purchase_tick,
            parameters=params,
        )
        action = choice.action
        rows = np.arange(size, dtype=np.int64)
        daily_action[rows, action.astype(np.int64)] += step_minutes

        play = action == LifeAction.PLAY
        purchase = action == LifeAction.PURCHASE
        stop = action == LifeAction.STOP
        sleep = action == LifeAction.SLEEP
        work = action == LifeAction.STUDY_WORK
        social = action == LifeAction.SOCIALIZE
        physical = action == LifeAction.EXERCISE

        daily_play[play] += step_minutes
        remaining_sleep[sleep] = np.maximum(
            0.0, remaining_sleep[sleep] - step_minutes
        )
        remaining_work[work] = np.maximum(
            0.0, remaining_work[work] - step_minutes
        )
        remaining_social[social] = np.maximum(
            0.0, remaining_social[social] - step_minutes
        )
        remaining_physical[physical] = np.maximum(
            0.0, remaining_physical[physical] - step_minutes
        )

        charge = choice.purchase_cents
        if np.any(charge > state.available_budget_cents):
            raise RuntimeError("purchase decision exceeded available budget")
        _checked_add_inplace(daily_spend, charge, "daily spend")
        _checked_add_inplace(state.player_spend_cents, charge, "player spend")
        _checked_add_inplace(
            state.cap_period_spend_cents, charge, "cap-period spend"
        )
        _checked_add_inplace(
            state.life.historical_spending_cents, charge, "historical spend"
        )
        state.available_budget_cents -= charge
        state.last_purchase_tick[purchase] = global_tick

        if np.any(purchase):
            source_draw = rng.uniform(
                players.player_id,
                global_tick,
                "policy:purchase-revenue-source",
                0,
            )
            random_source = purchase & (
                source_draw < mechanics.paid_random_rewards
            )
            opaque_threshold = mechanics.paid_random_rewards + (
                1.0 - mechanics.paid_random_rewards
            ) * mechanics.opaque_virtual_currency
            opaque_source = purchase & ~random_source & (
                source_draw < opaque_threshold
            )
            direct_source = purchase & ~random_source & ~opaque_source
            for source_index, mask in enumerate(
                (direct_source, opaque_source, random_source)
            ):
                source_charge = np.where(mask, charge, 0).astype(np.int64)
                _checked_add_inplace(
                    daily_sources[:, source_index],
                    source_charge,
                    f"daily {PURCHASE_REVENUE_SOURCES[source_index]} revenue",
                )
                _checked_add_inplace(
                    state.player_spend_by_source_cents[:, source_index],
                    source_charge,
                    f"cumulative {PURCHASE_REVENUE_SOURCES[source_index]} revenue",
                )

        _update_learning_state(
            players,
            state.life,
            rng,
            global_tick,
            play=play,
            purchase=purchase,
            stop=stop,
            mechanics=mechanics,
            parameters=params,
        )

    sleep_minutes = daily_action[:, LifeAction.SLEEP]
    work_minutes = daily_action[:, LifeAction.STUDY_WORK]
    social_minutes = daily_action[:, LifeAction.SOCIALIZE]
    physical_minutes = daily_action[:, LifeAction.EXERCISE]
    sleep_shortfall = np.maximum(
        state.life.sleep_need_minutes.astype(np.int64) - sleep_minutes, 0
    )
    sleep_recovery = np.maximum(
        sleep_minutes - state.life.sleep_need_minutes.astype(np.int64), 0
    )
    new_debt = (
        state.life.sleep_debt_minutes.astype(np.int64)
        + sleep_shortfall
        - sleep_recovery
    )
    state.life.sleep_debt_minutes[:] = np.clip(
        new_debt, 0, 7 * 1_440
    ).astype(np.int32)
    work_shortfall = np.maximum(
        state.life.work_study_obligation_minutes.astype(np.int64) - work_minutes,
        0,
    )
    social_shortfall = np.maximum(
        state.life.social_obligation_minutes.astype(np.int64) - social_minutes,
        0,
    )
    physical_shortfall = np.maximum(
        state.life.physical_activity_need_minutes.astype(np.int64) - physical_minutes,
        0,
    )
    excess_play = np.maximum(
        daily_play - state.life.intended_play_minutes.astype(np.float64), 0.0
    )
    intended = np.maximum(
        state.life.intended_play_minutes.astype(np.float64), 1.0
    )
    within_intention = np.minimum(daily_play, intended) / intended
    enjoyment = np.clip(
        state.life.baseline_game_enjoyment.astype(np.float64) * within_intention
        + 0.10 * state.life.current_game_progression.astype(np.float64)
        - 0.35
        * np.divide(
            excess_play,
            np.maximum(state.life.planned_leisure_minutes.astype(np.float64), 1.0),
        ),
        0.0,
        1.0,
    )
    wellbeing_delta = (
        0.018 * enjoyment
        + 0.010 * _completion(social_minutes, state.life.social_obligation_minutes)
        + 0.010
        * _completion(physical_minutes, state.life.physical_activity_need_minutes)
        - 0.025 * _completion(sleep_shortfall, state.life.sleep_need_minutes)
        - 0.020
        * _completion(work_shortfall, state.life.work_study_obligation_minutes)
        - 0.018
        * np.divide(
            excess_play,
            np.maximum(state.life.planned_leisure_minutes.astype(np.float64), 1.0),
        )
    )
    state.life.wellbeing[:] = np.clip(
        state.life.wellbeing.astype(np.float64) + wellbeing_delta,
        0.0,
        1.0,
    ).astype(np.float32)
    _checked_add_inplace(state.action_minutes, daily_action, "action minutes")
    state.life.actual_play_minutes[:] += daily_play.astype(np.int64)
    state.cumulative_enjoyment += enjoyment
    state.completed_days += 1

    if size and not np.all(daily_action.sum(axis=1) == 1_440):
        raise RuntimeError("daily time allocation does not conserve 1440 minutes")
    if not np.array_equal(daily_sources.sum(axis=1), daily_spend):
        raise RuntimeError("purchase revenue sources do not reconcile")
    return PolicyDayResult(
        day=day,
        action_minutes=daily_action,
        spend_cents=daily_spend,
        spend_by_source_cents=daily_sources,
        sleep_minutes=sleep_minutes.copy(),
        work_study_minutes=work_minutes.copy(),
        social_minutes=social_minutes.copy(),
        physical_minutes=physical_minutes.copy(),
        sleep_shortfall_minutes=sleep_shortfall,
        work_study_shortfall_minutes=work_shortfall,
        social_shortfall_minutes=social_shortfall,
        physical_shortfall_minutes=physical_shortfall,
        enjoyment=enjoyment,
    )


def _update_learning_state(
    players: PlayerTable,
    life: PlayerLifeTable,
    rng: CounterRNG,
    tick: int,
    *,
    play: npt.NDArray[np.bool_],
    purchase: npt.NDArray[np.bool_],
    stop: npt.NDArray[np.bool_],
    mechanics: MonetisationVector,
    parameters: DecisionParameters,
) -> None:
    engaged = play | purchase
    habit = parameters.habit_persistence * life.habit_strength.astype(np.float64)
    habit += parameters.habit_learning_rate * engaged.astype(np.float64)
    habit -= 0.60 * parameters.habit_learning_rate * stop.astype(np.float64)
    life.habit_strength[:] = np.clip(habit, 0.0, 1.0).astype(np.float32)

    reward_draw = rng.uniform(
        players.player_id, tick, "policy:reward-prediction-error", 0
    )
    observed_reward = (
        life.baseline_game_enjoyment.astype(np.float64)
        + 0.25
        * (
            reward_draw
            < 0.15 + 0.25 * mechanics.daily_streak_pressure
        ).astype(np.float64)
        + 0.15 * mechanics.pay_to_win * purchase.astype(np.float64)
    )
    prediction = 0.50 + 0.35 * life.reinforcement_state.astype(np.float64)
    error = np.clip(observed_reward - prediction, -1.0, 1.0)
    reinforcement = 0.98 * life.reinforcement_state.astype(np.float64)
    reinforcement[engaged] += parameters.reinforcement_learning_rate * error[engaged]
    life.reinforcement_state[:] = np.clip(
        reinforcement, -1.0, 1.0
    ).astype(np.float32)

    progression_gain = (
        parameters.step_minutes
        / 1_200.0
        * (1.0 - 0.55 * mechanics.progression_gates)
        * play.astype(np.float64)
        + 0.035 * mechanics.pay_to_progress * purchase.astype(np.float64)
    )
    life.current_game_progression[:] = np.clip(
        life.current_game_progression.astype(np.float64) + progression_gain,
        0.0,
        1.0,
    ).astype(np.float32)


def _completion(
    numerator: npt.ArrayLike, denominator: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    return np.clip(
        np.divide(top, np.maximum(bottom, 1.0)),
        0.0,
        1.0,
    )


def _checked_add_inplace(
    target: npt.NDArray[np.int64],
    increment: npt.NDArray[np.int64],
    label: str,
) -> None:
    if target.shape != increment.shape:
        raise ValueError(f"{label} shapes differ")
    if np.any(increment < 0) or np.any(target > np.iinfo(np.int64).max - increment):
        raise OverflowError(f"{label} would overflow int64")
    target += increment


__all__ = [
    "PURCHASE_REVENUE_SOURCES",
    "PolicyDayResult",
    "PolicyState",
    "advance_policy_day",
    "create_policy_state",
]
