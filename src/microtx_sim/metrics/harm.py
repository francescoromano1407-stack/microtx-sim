"""Pure multidimensional welfare-harm and opportunity-cost calculations.

This module deliberately separates model assumptions from state mutation.  It
does not classify all play or all spending as harmful: monetary harm requires
unplanned spending, mechanic-linked opacity/pressure/randomness, or financial
strain; opportunity cost requires play beyond planned leisure and a plausible
activity deficit.  The resulting scores are research proxies, not diagnoses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
_INT64_MAX = int(np.iinfo(np.int64).max)


class HarmComponent(IntEnum):
    """The six components in ``H = M + OC + S + E + F + W``."""

    M = 0
    OC = 1
    S = 2
    E = 3
    F = 4
    W = 5

    MONETARY = M
    OPPORTUNITY_COST = OC
    SLEEP = S
    EDUCATION_WORK = E
    FAMILY_SOCIAL = F
    WELLBEING = W


@dataclass(frozen=True, slots=True)
class WelfareHarmWeights:
    """Non-negative reporting weights; components remain stored separately."""

    monetary: float = 1.0
    opportunity_cost: float = 1.0
    sleep: float = 1.0
    education_work: float = 1.0
    family_social: float = 1.0
    wellbeing: float = 1.0

    def as_array(self) -> FloatArray:
        values = np.asarray(
            (
                self.monetary,
                self.opportunity_cost,
                self.sleep,
                self.education_work,
                self.family_social,
                self.wellbeing,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("harm weights must be finite and non-negative")
        if values.sum() <= 0.0:
            raise ValueError("at least one harm weight must be positive")
        return values


@dataclass(frozen=True, slots=True)
class OpportunityCostValuation:
    """Illustrative monetary proxies in simulation cents per displaced hour.

    Adult work/study time receives the largest monetary proxy.  Youth values
    are non-wage educational/welfare resource proxies; the non-monetary burden
    scores remain the primary youth outcome.
    """

    adult_sleep_hour_cents: int = 600
    adult_work_study_hour_cents: int = 1_800
    adult_social_hour_cents: int = 500
    adult_physical_activity_hour_cents: int = 450
    youth_sleep_hour_cents: int = 400
    youth_education_hour_cents: int = 500
    youth_family_social_hour_cents: int = 350
    youth_physical_activity_hour_cents: int = 300

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _scalar_nonnegative_int(getattr(self, name), name)

    def adult_rates(self) -> FloatArray:
        return np.asarray(
            (
                self.adult_sleep_hour_cents,
                self.adult_work_study_hour_cents,
                self.adult_social_hour_cents,
                self.adult_physical_activity_hour_cents,
            ),
            dtype=np.float64,
        )

    def youth_rates(self) -> FloatArray:
        return np.asarray(
            (
                self.youth_sleep_hour_cents,
                self.youth_education_hour_cents,
                self.youth_family_social_hour_cents,
                self.youth_physical_activity_hour_cents,
            ),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class HarmModelParameters:
    """Explicit illustrative assumptions used by :func:`compute_welfare_harm`."""

    affordable_spending_share: float = 0.10
    opaque_spending_weight: float = 0.35
    random_reward_spending_weight: float = 0.35
    time_pressure_spending_weight: float = 0.35
    sleep_debt_weight: float = 0.25

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class WelfareHarmResult:
    """Player-level component scores and non-overloaded monetary proxies."""

    component_scores: FloatArray
    harmful_spending_cents: IntArray
    unplanned_spending_cents: IntArray
    monetary_harm_proxy_cents: IntArray
    opportunity_cost_proxy_cents: IntArray
    adult_opportunity_cost_proxy_cents: IntArray
    youth_opportunity_cost_proxy_cents: IntArray
    total_monetary_proxy_cents: IntArray
    excess_play_minutes: FloatArray
    displaced_sleep_minutes: FloatArray
    displaced_work_study_minutes: FloatArray
    displaced_social_minutes: FloatArray
    displaced_physical_activity_minutes: FloatArray

    def __post_init__(self) -> None:
        if self.harmful_spending_cents.ndim != 1:
            raise ValueError("harmful_spending_cents must be one-dimensional")
        size = self.harmful_spending_cents.size
        if self.component_scores.shape != (size, len(HarmComponent)):
            raise ValueError("component_scores must have six columns per player")
        if not np.all(np.isfinite(self.component_scores)) or np.any(
            (self.component_scores < 0.0) | (self.component_scores > 1.0)
        ):
            raise ValueError("component scores must be finite and in [0, 1]")

        money_columns = (
            self.harmful_spending_cents,
            self.unplanned_spending_cents,
            self.monetary_harm_proxy_cents,
            self.opportunity_cost_proxy_cents,
            self.adult_opportunity_cost_proxy_cents,
            self.youth_opportunity_cost_proxy_cents,
            self.total_monetary_proxy_cents,
        )
        for values in money_columns:
            if values.ndim != 1 or values.size != size:
                raise ValueError("money result columns must have equal 1-D shapes")
            if values.dtype != np.dtype(np.int64) or np.any(values < 0):
                raise TypeError("money result columns must be non-negative int64")

        minute_columns = (
            self.excess_play_minutes,
            self.displaced_sleep_minutes,
            self.displaced_work_study_minutes,
            self.displaced_social_minutes,
            self.displaced_physical_activity_minutes,
        )
        for values in minute_columns:
            if values.ndim != 1 or values.size != size:
                raise ValueError("minute result columns must have equal 1-D shapes")
            if values.dtype != np.dtype(np.float64):
                raise TypeError("minute result columns must use float64")
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("minute result columns must be finite and non-negative")

        if not np.array_equal(
            self.adult_opportunity_cost_proxy_cents
            + self.youth_opportunity_cost_proxy_cents,
            self.opportunity_cost_proxy_cents,
        ):
            raise ValueError("adult/youth opportunity proxies do not reconcile")
        if not np.array_equal(
            self.monetary_harm_proxy_cents + self.opportunity_cost_proxy_cents,
            self.total_monetary_proxy_cents,
        ):
            raise ValueError("total monetary proxy does not reconcile")

    def component(self, component: HarmComponent | int) -> FloatArray:
        """Return one non-monetary burden-score column without copying."""

        return self.component_scores[:, int(component)]

    def composite_harm(
        self,
        weights: WelfareHarmWeights | None = None,
    ) -> FloatArray:
        """Return a weighted reporting view without discarding components."""

        values = (weights or WelfareHarmWeights()).as_array()
        return self.component_scores @ (values / values.sum())


def compute_welfare_harm(
    *,
    is_minor: npt.ArrayLike,
    disposable_budget_cents: npt.ArrayLike,
    intended_spending_limit_cents: npt.ArrayLike,
    historical_spending_cents: npt.ArrayLike,
    spending_cents: npt.ArrayLike,
    opaque_virtual_currency_exposure: npt.ArrayLike,
    paid_random_reward_exposure: npt.ArrayLike,
    time_pressure_exposure: npt.ArrayLike,
    actual_play_minutes: npt.ArrayLike,
    planned_leisure_minutes: npt.ArrayLike,
    sleep_need_minutes: npt.ArrayLike,
    actual_sleep_minutes: npt.ArrayLike,
    sleep_debt_minutes: npt.ArrayLike,
    work_study_obligation_minutes: npt.ArrayLike,
    actual_work_study_minutes: npt.ArrayLike,
    social_obligation_minutes: npt.ArrayLike,
    actual_social_minutes: npt.ArrayLike,
    physical_activity_need_minutes: npt.ArrayLike,
    actual_physical_activity_minutes: npt.ArrayLike,
    wellbeing_before: npt.ArrayLike,
    wellbeing_after: npt.ArrayLike,
    parameters: HarmModelParameters | None = None,
    valuation: OpportunityCostValuation | None = None,
) -> WelfareHarmResult:
    """Calculate six welfare components without mutating any input.

    Opportunity displacement is allocated proportionally across observed
    activity deficits and is capped by play beyond planned leisure.  Therefore
    ordinary gaming within the player's leisure allocation has zero opportunity
    cost even when another activity happens to fall below its baseline target.
    """

    params = parameters or HarmModelParameters()
    values = valuation or OpportunityCostValuation()
    if not isinstance(params, HarmModelParameters):
        raise TypeError("parameters must be HarmModelParameters")
    if not isinstance(values, OpportunityCostValuation):
        raise TypeError("valuation must be OpportunityCostValuation")

    minor = _bool_vector(is_minor, "is_minor")
    size = minor.size
    budget = _money_vector(disposable_budget_cents, "disposable_budget_cents", size)
    intended_limit = _money_vector(
        intended_spending_limit_cents, "intended_spending_limit_cents", size
    )
    historical = _money_vector(
        historical_spending_cents, "historical_spending_cents", size
    )
    spending = _money_vector(spending_cents, "spending_cents", size)
    opaque = _unit_vector(
        opaque_virtual_currency_exposure,
        "opaque_virtual_currency_exposure",
        size,
    )
    random_reward = _unit_vector(
        paid_random_reward_exposure,
        "paid_random_reward_exposure",
        size,
    )
    time_pressure = _unit_vector(
        time_pressure_exposure,
        "time_pressure_exposure",
        size,
    )

    actual_play = _nonnegative_vector(actual_play_minutes, "actual_play_minutes", size)
    leisure = _nonnegative_vector(
        planned_leisure_minutes, "planned_leisure_minutes", size
    )
    sleep_need = _nonnegative_vector(sleep_need_minutes, "sleep_need_minutes", size)
    actual_sleep = _nonnegative_vector(
        actual_sleep_minutes, "actual_sleep_minutes", size
    )
    sleep_debt = _nonnegative_vector(sleep_debt_minutes, "sleep_debt_minutes", size)
    work_need = _nonnegative_vector(
        work_study_obligation_minutes,
        "work_study_obligation_minutes",
        size,
    )
    actual_work = _nonnegative_vector(
        actual_work_study_minutes, "actual_work_study_minutes", size
    )
    social_need = _nonnegative_vector(
        social_obligation_minutes, "social_obligation_minutes", size
    )
    actual_social = _nonnegative_vector(
        actual_social_minutes, "actual_social_minutes", size
    )
    physical_need = _nonnegative_vector(
        physical_activity_need_minutes, "physical_activity_need_minutes", size
    )
    actual_physical = _nonnegative_vector(
        actual_physical_activity_minutes,
        "actual_physical_activity_minutes",
        size,
    )
    before = _unit_vector(wellbeing_before, "wellbeing_before", size)
    after = _unit_vector(wellbeing_after, "wellbeing_after", size)

    if np.any(historical > _INT64_MAX - spending):
        raise OverflowError("historical plus current spending would overflow int64")
    total_spending = historical + spending
    remaining_commitment = np.maximum(intended_limit - historical, 0)
    unplanned = np.maximum(spending - remaining_commitment, 0).astype(np.int64)

    spending_float = spending.astype(np.float64)
    unplanned_share = np.divide(
        unplanned.astype(np.float64),
        spending_float,
        out=np.zeros(size, dtype=np.float64),
        where=spending > 0,
    )
    mechanic_risk = 1.0 - (
        (1.0 - params.opaque_spending_weight * opaque)
        * (1.0 - params.random_reward_spending_weight * random_reward)
        * (1.0 - params.time_pressure_spending_weight * time_pressure)
    )
    affordable_threshold = (
        budget.astype(np.float64) * params.affordable_spending_share
    )
    strain_current = np.minimum(
        spending_float,
        np.maximum(total_spending.astype(np.float64) - affordable_threshold, 0.0),
    )
    strain_share = np.divide(
        strain_current,
        spending_float,
        out=np.zeros(size, dtype=np.float64),
        where=spending > 0,
    )
    harmful_share = 1.0 - (
        (1.0 - unplanned_share) * (1.0 - mechanic_risk) * (1.0 - strain_share)
    )
    harmful = np.rint(spending_float * np.clip(harmful_share, 0.0, 1.0)).astype(
        np.int64
    )
    harmful = np.minimum(spending, np.maximum(harmful, unplanned)).astype(np.int64)
    monetary_score = np.divide(
        harmful.astype(np.float64),
        np.maximum(budget.astype(np.float64), 1.0),
        out=np.zeros(size, dtype=np.float64),
        where=harmful > 0,
    )
    monetary_score = np.clip(monetary_score, 0.0, 1.0)

    excess_play = np.maximum(actual_play - leisure, 0.0)
    deficits = np.column_stack(
        (
            np.maximum(sleep_need - actual_sleep, 0.0),
            np.maximum(work_need - actual_work, 0.0),
            np.maximum(social_need - actual_social, 0.0),
            np.maximum(physical_need - actual_physical, 0.0),
        )
    )
    total_deficit = deficits.sum(axis=1)
    allocation_scale = np.divide(
        excess_play,
        total_deficit,
        out=np.zeros(size, dtype=np.float64),
        where=total_deficit > 0.0,
    )
    allocation_scale = np.minimum(allocation_scale, 1.0)
    displaced = deficits * allocation_scale[:, None]
    displaced_sleep = displaced[:, 0]
    displaced_work = displaced[:, 1]
    displaced_social = displaced[:, 2]
    displaced_physical = displaced[:, 3]

    sleep_ratio = _safe_ratio(displaced_sleep, sleep_need)
    work_ratio = _safe_ratio(displaced_work, work_need)
    social_ratio = _safe_ratio(displaced_social, social_need)
    physical_ratio = _safe_ratio(displaced_physical, physical_need)
    adult_oc_score = (
        0.20 * sleep_ratio
        + 0.45 * work_ratio
        + 0.20 * social_ratio
        + 0.15 * physical_ratio
    )
    youth_oc_score = (
        0.30 * sleep_ratio
        + 0.40 * work_ratio
        + 0.20 * social_ratio
        + 0.10 * physical_ratio
    )
    opportunity_score = np.where(minor, youth_oc_score, adult_oc_score)
    opportunity_score = np.where(excess_play > 0.0, opportunity_score, 0.0)

    sleep_score = np.clip(
        _safe_ratio(
            displaced_sleep + params.sleep_debt_weight * sleep_debt,
            sleep_need,
        ),
        0.0,
        1.0,
    )
    education_work_score = np.clip(work_ratio, 0.0, 1.0)
    family_social_score = np.clip(social_ratio, 0.0, 1.0)
    wellbeing_score = np.clip(before - after, 0.0, 1.0)

    component_scores = np.column_stack(
        (
            monetary_score,
            np.clip(opportunity_score, 0.0, 1.0),
            sleep_score,
            education_work_score,
            family_social_score,
            wellbeing_score,
        )
    ).astype(np.float64)

    adult_proxy = _rounded_proxy(
        (displaced / 60.0) @ values.adult_rates(), "adult opportunity cost"
    )
    youth_proxy = _rounded_proxy(
        (displaced / 60.0) @ values.youth_rates(), "youth opportunity cost"
    )
    adult_proxy = np.where(minor, 0, adult_proxy).astype(np.int64)
    youth_proxy = np.where(minor, youth_proxy, 0).astype(np.int64)
    if np.any(adult_proxy > _INT64_MAX - youth_proxy):
        raise OverflowError("opportunity cost proxy would overflow int64")
    opportunity_proxy = adult_proxy + youth_proxy
    if np.any(harmful > _INT64_MAX - opportunity_proxy):
        raise OverflowError("total monetary proxy would overflow int64")
    total_proxy = harmful + opportunity_proxy

    return WelfareHarmResult(
        component_scores=component_scores,
        harmful_spending_cents=harmful,
        unplanned_spending_cents=unplanned,
        monetary_harm_proxy_cents=harmful.copy(),
        opportunity_cost_proxy_cents=opportunity_proxy,
        adult_opportunity_cost_proxy_cents=adult_proxy,
        youth_opportunity_cost_proxy_cents=youth_proxy,
        total_monetary_proxy_cents=total_proxy,
        excess_play_minutes=excess_play.astype(np.float64),
        displaced_sleep_minutes=displaced_sleep.astype(np.float64),
        displaced_work_study_minutes=displaced_work.astype(np.float64),
        displaced_social_minutes=displaced_social.astype(np.float64),
        displaced_physical_activity_minutes=displaced_physical.astype(np.float64),
    )


def _bool_vector(values: npt.ArrayLike, name: str) -> BoolArray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must contain booleans")
    return np.asarray(array, dtype=np.bool_)


def _money_vector(values: npt.ArrayLike, name: str, size: int) -> IntArray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{name} must have shape ({size},)")
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integer cents")
    if array.dtype.kind == "u" and np.any(array > _INT64_MAX):
        raise OverflowError(f"{name} exceeds signed int64")
    result = array.astype(np.int64, copy=False)
    if np.any(result < 0):
        raise ValueError(f"{name} cannot be negative")
    return result


def _nonnegative_vector(values: npt.ArrayLike, name: str, size: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return array


def _unit_vector(values: npt.ArrayLike, name: str, size: int) -> FloatArray:
    array = _nonnegative_vector(values, name, size)
    if np.any(array > 1.0):
        raise ValueError(f"{name} must contain values in [0, 1]")
    return array


def _safe_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(numerator.shape, dtype=np.float64),
        where=denominator > 0.0,
    )


def _rounded_proxy(values: FloatArray, name: str) -> IntArray:
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} produced an invalid value")
    if np.any(values > _INT64_MAX):
        raise OverflowError(f"{name} exceeds signed int64")
    return np.rint(values).astype(np.int64)


def _scalar_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    if result > _INT64_MAX:
        raise OverflowError(f"{name} exceeds signed int64")
    return result


__all__ = [
    "HarmComponent",
    "HarmModelParameters",
    "OpportunityCostValuation",
    "WelfareHarmResult",
    "WelfareHarmWeights",
    "compute_welfare_harm",
]
