"""Money-flow aggregation and outcome accounting for the simulation kernel.

This module contains no agent policy. It mutates only kernel-owned balances and
records the corresponding double-entry transfers, keeping behavioural choices
separate from accounting consequences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..consumers.logic import StepResult
from ..metrics.outcomes import OutcomeSnapshot

if TYPE_CHECKING:
    from ..core.world import World


IntArray = npt.NDArray[np.int64]
INT64_MAX = int(np.iinfo(np.int64).max)


def checked_accumulate(
    target: IntArray,
    increment: IntArray,
    *,
    label: str,
) -> None:
    """Add aligned non-negative cent arrays after an explicit overflow check."""

    values = np.asarray(increment, dtype=np.int64)
    if values.shape != target.shape or np.any(values < 0) or np.any(target < 0):
        raise ValueError(f"{label} needs aligned non-negative int64 arrays")
    if np.any(target > INT64_MAX - values):
        raise OverflowError(f"{label} would overflow int64")
    target += values


def renew_income(world: "World", *, tick: int) -> None:
    """Credit one income/allowance renewal without exposing it to policies."""

    essential_share = world.player_system.config.essential_spend_share
    adult_inflow = np.rint(
        world.players.monthly_disposable_income_cents * (1.0 - essential_share)
    ).astype(np.int64)
    inflow = np.where(
        world.players.is_minor,
        world.players.allowance_cents,
        adult_inflow,
    ).astype(np.int64)
    if np.any(world.players.liquidity_cents > INT64_MAX - inflow):
        raise OverflowError("player liquidity would overflow")
    world.players.liquidity_cents[:] += inflow
    for row in np.flatnonzero(inflow > 0):
        player_id = int(world.players.player_id[row])
        jurisdiction = int(world.players.jurisdiction[row])
        world.ledger.transfer(
            tick=tick,
            debit_account=f"external:income:{jurisdiction}",
            credit_account=f"player:{player_id}:liquid",
            amount_cents=int(inflow[row]),
            kind="disposable_income",
            reference=f"income:{tick}:{player_id}",
        )


def credit_firm_revenue(world: "World", result: StepResult) -> None:
    """Aggregate exact game revenue into company balances and kernel totals."""

    company_ids = np.asarray(world.games.company_id)
    by_firm = _checked_grouped_money(
        result.game_revenue_cents,
        company_ids,
        len(world.firms),
        label="firm revenue aggregation",
    )
    unsafe = _checked_grouped_money(
        result.game_unsafe_revenue_cents,
        company_ids,
        len(world.firms),
        label="unsafe firm revenue aggregation",
    )
    planned_cash: list[int] = []
    for firm in world.firms:
        revenue = int(by_firm[firm.firm_id])
        cash = int(firm.state.cash_cents)
        if cash < 0 or cash > INT64_MAX or revenue > INT64_MAX - cash:
            raise OverflowError("firm cash would overflow int64")
        planned_cash.append(cash + revenue)

    planned_revenue = world.firm_revenue_cents.copy()
    checked_accumulate(
        planned_revenue,
        by_firm,
        label="cumulative firm revenue",
    )
    planned_unsafe_revenue = world.firm_unsafe_revenue_cents.copy()
    checked_accumulate(
        planned_unsafe_revenue,
        unsafe,
        label="cumulative unsafe firm revenue",
    )

    # Every possible failure is above this line; committing detached values is
    # therefore one non-raising transaction from the kernel's perspective.
    for firm, cash in zip(world.firms, planned_cash):
        firm.state.cash_cents = cash
    world.firm_revenue_cents[:] = planned_revenue
    world.firm_unsafe_revenue_cents[:] = planned_unsafe_revenue


def _checked_grouped_money(
    values: IntArray,
    group_ids: npt.NDArray[np.integer],
    group_count: int,
    *,
    label: str,
) -> IntArray:
    """Return exact non-negative int64 group totals without wraparound."""

    money = np.asarray(values)
    groups = np.asarray(group_ids)
    if (
        money.dtype != np.dtype(np.int64)
        or money.ndim != 1
        or groups.ndim != 1
        or not np.issubdtype(groups.dtype, np.integer)
        or money.shape != groups.shape
    ):
        raise ValueError(f"{label} needs aligned one-dimensional int64 arrays")
    if np.any(money < 0):
        raise ValueError(f"{label} needs non-negative int64 values")
    totals = [0] * group_count
    for group_value, money_value in zip(groups, money):
        group = int(group_value)
        if group < 0 or group >= group_count:
            raise ValueError(f"{label} references an unknown firm")
        total = totals[group] + int(money_value)
        if total > INT64_MAX:
            raise OverflowError(f"{label} would overflow int64")
        totals[group] = total
    return np.asarray(totals, dtype=np.int64)


def accrue_interest(world: "World") -> None:
    """Accrue player credit interest for the current tick in exact cents."""

    principal = world._initial_credit_limit_cents - world.players.credit_limit_cents
    raw_interest = (
        principal.astype(np.float64)
        * world.config.behavior.daily_credit_interest_rate
        * world.config.run.tick_days
    )
    if (
        not np.all(np.isfinite(raw_interest))
        or np.any(raw_interest > 2**53)
        or np.any(raw_interest < 0.0)
    ):
        raise OverflowError("interest calculation exceeded exact-cent range")
    interest = np.rint(raw_interest).astype(np.int64)
    checked_accumulate(
        world.player_interest_cents,
        interest,
        label="cumulative player interest",
    )


def outcome_snapshot(world: "World", *, tick: int | None = None) -> OutcomeSnapshot:
    """Build an immutable research outcome from latent kernel state."""

    cash_values = [firm.state.cash_cents for firm in world.firms]
    if any(value < 0 or value > INT64_MAX for value in cash_values):
        raise OverflowError("firm cash is outside the reportable int64 range")
    firm_cash = np.asarray(cash_values, dtype=np.int64)
    outstanding_fines = world.firm_fine_assessed_cents - world.firm_fine_paid_cents
    margin_values = [
        int(firm_cash[index])
        - int(world._initial_firm_cash_cents[index])
        - int(world.firm_subsidy_cents[index])
        - int(outstanding_fines[index])
        for index in range(len(world.firms))
    ]
    int64_min = int(np.iinfo(np.int64).min)
    if any(value < int64_min or value > INT64_MAX for value in margin_values):
        raise OverflowError("firm margin is outside the reportable int64 range")
    margin = np.asarray(margin_values, dtype=np.int64)
    safe_share = np.divide(
        world.firm_revenue_cents - world.firm_unsafe_revenue_cents,
        world.firm_revenue_cents,
        out=np.ones(len(world.firms), dtype=np.float64),
        where=world.firm_revenue_cents > 0,
    )
    debt = world._initial_credit_limit_cents - world.players.credit_limit_cents
    checked_accumulate(
        debt,
        world.player_interest_cents,
        label="reported player debt",
    )
    return OutcomeSnapshot(
        tick=world.tick if tick is None else tick,
        player_harm=world.players.harm_state.astype(np.float64, copy=True),
        player_spend_cents=world.player_total_spend_cents.copy(),
        player_income_cents=world.players.monthly_disposable_income_cents.copy(),
        player_debt_cents=debt.astype(np.int64, copy=False),
        firm_cash_cents=firm_cash,
        firm_operating_margin_cents=margin,
        firm_safe_revenue_share=safe_share,
        state_subsidy_outlay_cents=world.state_subsidy_outlay_cents.copy(),
    )
