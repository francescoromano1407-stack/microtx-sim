"""Orchestrate one complete synthetic policy scenario over a fixed cohort."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping

import numpy as np
import numpy.typing as npt

from ..agents.players import PlayerTable
from ..causal.scenarios import ScenarioSpec
from ..consumers.decision import DecisionParameters, LifeAction
from ..consumers.welfare import PlayerLifeTable
from ..funding import EPGCFirmInputs, EPGCPolicy, EPGCResult, evaluate_epgc
from ..metrics.harm import (
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmResult,
    WelfareHarmWeights,
    compute_welfare_harm,
)
from ..rng import CounterRNG, validate_seed
from .policy_day import (
    PURCHASE_REVENUE_SOURCES,
    PolicyState,
    advance_policy_day,
    create_policy_state,
)


@dataclass(frozen=True, slots=True)
class ProducerAssumptions:
    """Illustrative cost and non-behavioural revenue assumptions."""

    development_cost_cents: int = 1_200_000
    maintenance_cost_cents_per_day: int = 20_000
    institutional_license_count: int = 20
    institutional_license_price_cents: int = 12_000
    non_targeted_sponsorship_revenue_cents: int = 100_000
    accessibility_eligible: bool = True
    multilingual_support_eligible: bool = True
    cultural_value_eligible: bool = True
    safety_certified: bool = True

    def __post_init__(self) -> None:
        for name in (
            "development_cost_cents",
            "maintenance_cost_cents_per_day",
            "institutional_license_count",
            "institutional_license_price_cents",
            "non_targeted_sponsorship_revenue_cents",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


def default_epgc_policy() -> EPGCPolicy:
    """Return the documented synthetic payment schedule used by the example."""

    return EPGCPolicy(
        access_payment_cents_per_eligible_access=250,
        institutional_license_payment_cents_per_license=8_000,
        availability_payment_cents_per_period=300_000,
        accessibility_bonus_cents=120_000,
        multilingual_bonus_cents=100_000,
        cultural_value_bonus_cents=100_000,
        safety_certification_bonus_cents=150_000,
        prohibited_mechanics_penalty_cents=100_000,
        prohibited_mechanics_clawback_basis_points=10_000,
        maximum_budget_cents=3_000_000,
    )


@dataclass(frozen=True, slots=True)
class PolicyScenarioResult:
    """Player-level welfare outcomes and auditable producer accounts."""

    scenario: ScenarioSpec
    seed: int
    days: int
    player_ids: npt.NDArray[np.int64]
    is_minor: npt.NDArray[np.bool_]
    age_years: npt.NDArray[np.int16]
    jurisdiction: npt.NDArray[np.int16]
    baseline_vulnerability: npt.NDArray[np.float32]
    disposable_budget_cents: npt.NDArray[np.int64]
    spending_cents: npt.NDArray[np.int64]
    harm: WelfareHarmResult
    composite_harm: npt.NDArray[np.float64]
    enjoyment: npt.NDArray[np.float64]
    high_risk: npt.NDArray[np.bool_]
    action_minutes: npt.NDArray[np.int64]
    revenue_composition_cents: Mapping[str, int]
    total_revenue_cents: int
    producer_cost_cents: int
    producer_profit_cents: int
    epgc: EPGCResult | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", validate_seed(self.seed))
        size = self.player_ids.size
        for name in (
            "is_minor",
            "age_years",
            "jurisdiction",
            "baseline_vulnerability",
            "disposable_budget_cents",
            "spending_cents",
            "composite_harm",
            "enjoyment",
            "high_risk",
        ):
            if getattr(self, name).shape != (size,):
                raise ValueError(f"{name} must have shape ({size},)")
        if self.action_minutes.shape != (size, len(LifeAction)):
            raise ValueError("action_minutes has the wrong shape")
        if self.harm.component_scores.shape != (size, 6):
            raise ValueError("harm result is not aligned with players")
        if not np.all(np.isfinite(self.composite_harm)) or np.any(
            (self.composite_harm < 0.0) | (self.composite_harm > 1.0)
        ):
            raise ValueError("composite_harm must be finite and in [0, 1]")
        if not np.all(np.isfinite(self.enjoyment)) or np.any(
            (self.enjoyment < 0.0) | (self.enjoyment > 1.0)
        ):
            raise ValueError("enjoyment must be finite and in [0, 1]")
        revenue = {str(key): int(value) for key, value in self.revenue_composition_cents.items()}
        if any(value < 0 for value in revenue.values()):
            raise ValueError("revenue components cannot be negative")
        if sum(revenue.values()) != self.total_revenue_cents:
            raise ValueError("revenue composition does not reconcile")
        if self.producer_profit_cents != self.total_revenue_cents - self.producer_cost_cents:
            raise ValueError("producer profit does not reconcile")
        object.__setattr__(self, "revenue_composition_cents", MappingProxyType(revenue))


def run_policy_scenario(
    players: PlayerTable,
    initial_life: PlayerLifeTable,
    scenario: ScenarioSpec,
    *,
    seed: int,
    days: int,
    decision_parameters: DecisionParameters | None = None,
    harm_parameters: HarmModelParameters | None = None,
    harm_weights: WelfareHarmWeights | None = None,
    opportunity_valuation: OpportunityCostValuation | None = None,
    producer_assumptions: ProducerAssumptions | None = None,
    epgc_policy: EPGCPolicy | None = None,
) -> PolicyScenarioResult:
    """Run one branch from an immutable shared pre-treatment cohort."""

    seed = validate_seed(seed)
    if isinstance(days, bool) or not isinstance(days, int):
        raise TypeError("days must be an integer")
    if days < 0:
        raise ValueError("days cannot be negative")
    initial_life.validate_alignment(players)
    params = decision_parameters or DecisionParameters()
    producer = producer_assumptions or ProducerAssumptions()
    policy = epgc_policy or default_epgc_policy()
    rng = CounterRNG(seed)
    life = clone_player_life(initial_life)
    state = create_policy_state(players, life)
    historical_before = life.historical_spending_cents.copy()
    _apply_access_revenue(players, state, scenario, rng)
    for day in range(days):
        advance_policy_day(
            players,
            state,
            scenario.mechanics,
            rng,
            day=day,
            parameters=params,
        )

    months = max(1, (days + 29) // 30)
    disposable_budget = _checked_scale_money(
        players.monthly_disposable_income_cents, months, "disposable budget"
    )
    intended_limit = _checked_scale_money(
        life.intended_spending_limit_cents, months, "intended spending limit"
    )
    planned_game_leisure = _checked_scale_minutes(
        np.minimum(life.intended_play_minutes, life.planned_leisure_minutes), days
    )
    actual_sleep = state.action_minutes[:, LifeAction.SLEEP]
    actual_work = state.action_minutes[:, LifeAction.STUDY_WORK]
    actual_social = state.action_minutes[:, LifeAction.SOCIALIZE]
    actual_physical = state.action_minutes[:, LifeAction.EXERCISE]
    harm = compute_welfare_harm(
        is_minor=players.is_minor,
        disposable_budget_cents=disposable_budget,
        intended_spending_limit_cents=intended_limit,
        historical_spending_cents=historical_before,
        spending_cents=state.player_spend_cents,
        opaque_virtual_currency_exposure=np.full(
            len(players), scenario.mechanics.opaque_virtual_currency
        ),
        paid_random_reward_exposure=np.full(
            len(players), scenario.mechanics.paid_random_rewards
        ),
        time_pressure_exposure=np.full(
            len(players), scenario.mechanics.time_limited_offers
        ),
        actual_play_minutes=life.actual_play_minutes,
        planned_leisure_minutes=planned_game_leisure,
        sleep_need_minutes=_checked_scale_minutes(life.sleep_need_minutes, days),
        actual_sleep_minutes=actual_sleep,
        sleep_debt_minutes=life.sleep_debt_minutes,
        work_study_obligation_minutes=_checked_scale_minutes(
            life.work_study_obligation_minutes, days
        ),
        actual_work_study_minutes=actual_work,
        social_obligation_minutes=_checked_scale_minutes(
            life.social_obligation_minutes, days
        ),
        actual_social_minutes=actual_social,
        physical_activity_need_minutes=_checked_scale_minutes(
            life.physical_activity_need_minutes, days
        ),
        actual_physical_activity_minutes=actual_physical,
        wellbeing_before=state.initial_wellbeing,
        wellbeing_after=life.wellbeing,
        parameters=harm_parameters,
        valuation=opportunity_valuation,
    )
    composite = harm.composite_harm(harm_weights)
    enjoyment = (
        state.cumulative_enjoyment / days
        if days
        else np.zeros(len(players), dtype=np.float64)
    )
    harmful_share = np.divide(
        harm.harmful_spending_cents.astype(np.float64),
        np.maximum(disposable_budget.astype(np.float64), 1.0),
    )
    high_risk = (
        (composite >= 0.35)
        | (harmful_share >= 0.10)
        | (harm.component_scores[:, 2] >= 0.50)
    )

    conventional = _conventional_revenue(state, scenario, producer)
    costs = _checked_scalar_sum(
        producer.development_cost_cents,
        producer.maintenance_cost_cents_per_day * days,
        label="producer costs",
    )
    epgc_result: EPGCResult | None = None
    public_contract = 0
    institutional = conventional["institutional_licensing"]
    sponsorship = conventional["non_targeted_sponsorship"]
    if scenario.epgc_enabled:
        epgc_result = evaluate_epgc(
            policy,
            EPGCFirmInputs(
                fixed_price_revenue_cents=conventional["fixed_price"],
                institutional_licensing_revenue_cents=institutional,
                non_targeted_sponsorship_revenue_cents=sponsorship,
                development_cost_cents=producer.development_cost_cents,
                maintenance_cost_cents=producer.maintenance_cost_cents_per_day
                * days,
                eligible_access_count=len(players),
                eligible_institutional_license_count=(
                    producer.institutional_license_count
                ),
                availability_period_count=max(1, months),
                accessibility_eligible=producer.accessibility_eligible,
                multilingual_support_eligible=(
                    producer.multilingual_support_eligible
                ),
                cultural_value_eligible=producer.cultural_value_eligible,
                safety_certified=producer.safety_certified,
                prohibited_mechanics_enabled=_has_prohibited_mechanics(scenario),
            ),
        )
        public_contract = epgc_result.public_contract_revenue_cents
    revenue = {
        **conventional,
        "public_contract": public_contract,
    }
    total_revenue = _checked_scalar_sum(*revenue.values(), label="total revenue")
    return PolicyScenarioResult(
        scenario=scenario,
        seed=seed,
        days=days,
        player_ids=players.player_id.copy(),
        is_minor=players.is_minor.copy(),
        age_years=players.age_years.copy(),
        jurisdiction=players.jurisdiction.copy(),
        baseline_vulnerability=players.baseline_vulnerability.copy(),
        disposable_budget_cents=disposable_budget,
        spending_cents=state.player_spend_cents.copy(),
        harm=harm,
        composite_harm=composite,
        enjoyment=enjoyment,
        high_risk=high_risk,
        action_minutes=state.action_minutes.copy(),
        revenue_composition_cents=revenue,
        total_revenue_cents=total_revenue,
        producer_cost_cents=costs,
        producer_profit_cents=total_revenue - costs,
        epgc=epgc_result,
    )


def clone_player_life(source: PlayerLifeTable) -> PlayerLifeTable:
    """Deep-copy a branch so no mutable post-treatment state is shared."""

    return PlayerLifeTable(
        **{
            descriptor.name: np.array(
                getattr(source, descriptor.name), copy=True
            )
            for descriptor in fields(source)
        }
    )


def _apply_access_revenue(
    players: PlayerTable,
    state: PolicyState,
    scenario: ScenarioSpec,
    rng: CounterRNG,
) -> None:
    if not scenario.fixed_access_price_cents and not scenario.subscription_price_cents:
        return
    interest = np.clip(
        0.12
        + 0.58 * state.life.baseline_game_enjoyment.astype(np.float64)
        - 0.20 * state.life.financial_sensitivity.astype(np.float64),
        0.02,
        0.85,
    )
    adopt = rng.uniform(
        players.player_id, 0, "policy:access-adoption", 0
    ) < interest
    plan_draw = rng.uniform(players.player_id, 0, "policy:access-plan", 0)
    choose_subscription = (
        scenario.subscription_price_cents > 0
    ) & (plan_draw < 0.50)
    proposed = np.where(
        choose_subscription,
        scenario.subscription_price_cents,
        scenario.fixed_access_price_cents,
    ).astype(np.int64)
    if scenario.fixed_access_price_cents == 0:
        proposed[:] = scenario.subscription_price_cents
        choose_subscription[:] = True
    authorised = (~players.is_minor) | (
        players.guardian_consent & players.has_stored_payment_access
    )
    within_commitment = proposed <= state.life.intended_spending_limit_cents
    eligible = (
        adopt
        & authorised
        & within_commitment
        & (proposed > 0)
        & (proposed <= state.available_budget_cents)
    )
    charge = np.where(eligible, proposed, 0).astype(np.int64)
    _checked_add_money(state.player_spend_cents, charge, "access spending")
    _checked_add_money(
        state.life.historical_spending_cents, charge, "access spending history"
    )
    state.available_budget_cents -= charge
    # Two columns are appended conceptually by _conventional_revenue; the
    # microtransaction source table deliberately remains three-column.
    state.access_fixed_cents[:] = np.where(
        eligible & ~choose_subscription, charge, 0
    ).astype(np.int64)
    state.access_subscription_cents[:] = np.where(
        eligible & choose_subscription, charge, 0
    ).astype(np.int64)


def _conventional_revenue(
    state: PolicyState,
    scenario: ScenarioSpec,
    producer: ProducerAssumptions,
) -> dict[str, int]:
    values = {
        source: int(state.player_spend_by_source_cents[:, index].sum(dtype=np.int64))
        for index, source in enumerate(PURCHASE_REVENUE_SOURCES)
    }
    values["fixed_price"] = int(state.access_fixed_cents.sum(dtype=np.int64))
    values["subscription"] = int(
        state.access_subscription_cents.sum(dtype=np.int64)
    )
    alternative_model = bool(
        scenario.fixed_access_price_cents or scenario.subscription_price_cents
    )
    values["institutional_licensing"] = (
        producer.institutional_license_count
        * producer.institutional_license_price_cents
        if alternative_model
        else 0
    )
    values["non_targeted_sponsorship"] = (
        producer.non_targeted_sponsorship_revenue_cents
        if alternative_model
        else 0
    )
    return values


def _has_prohibited_mechanics(scenario: ScenarioSpec) -> bool:
    mechanics = scenario.mechanics
    return bool(
        mechanics.paid_random_rewards > 0.0
        or mechanics.time_limited_offers > 0.0
        or mechanics.opaque_virtual_currency > 0.0
        or mechanics.personalized_offers
    )


def _checked_scale_money(
    values: npt.NDArray[np.int64], multiplier: int, label: str
) -> npt.NDArray[np.int64]:
    if multiplier < 0:
        raise ValueError(f"{label} multiplier cannot be negative")
    if multiplier and np.any(values > np.iinfo(np.int64).max // multiplier):
        raise OverflowError(f"{label} would overflow int64")
    return (values * multiplier).astype(np.int64)


def _checked_scale_minutes(
    values: npt.NDArray[np.integer], multiplier: int
) -> npt.NDArray[np.int64]:
    array = values.astype(np.int64)
    return _checked_scale_money(array, multiplier, "time target")


def _checked_add_money(
    target: npt.NDArray[np.int64], increment: npt.NDArray[np.int64], label: str
) -> None:
    if np.any(target > np.iinfo(np.int64).max - increment):
        raise OverflowError(f"{label} would overflow int64")
    target += increment


def _checked_scalar_sum(*values: int, label: str) -> int:
    total = sum(int(value) for value in values)
    if total < 0 or total > np.iinfo(np.int64).max:
        raise OverflowError(f"{label} is outside int64")
    return total


__all__ = [
    "PolicyScenarioResult",
    "ProducerAssumptions",
    "clone_player_life",
    "default_epgc_policy",
    "run_policy_scenario",
]
