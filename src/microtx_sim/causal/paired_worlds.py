from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..metrics.outcomes import HarmWeights, OutcomeSnapshot
from ..config import SimulationConfig
from ..core.engine import RunResult, SimulationEngine
from ..core.world import World
from ..data.profiles import ProfileBundle
from .interventions import Intervention, NullIntervention


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PairedOutcome:
    player_harm_difference: FloatArray
    player_spend_difference_cents: IntArray
    player_debt_difference_cents: IntArray
    firm_margin_difference_cents: IntArray
    firm_cash_difference_cents: IntArray
    state_subsidy_difference_cents: IntArray


@dataclass(frozen=True, slots=True)
class RegimeEffect:
    estimand: str
    mean_composite_harm_effect: float
    total_spend_effect_cents: int
    total_debt_effect_cents: int
    total_operating_margin_effect_cents: int
    total_subsidy_effect_cents: int
    affected_player_share: float


@dataclass(frozen=True, slots=True)
class PairedWorldRun:
    """Outputs from two structurally identical counterfactual markets."""

    treated_run: RunResult
    control_run: RunResult
    paired_outcome: PairedOutcome
    effect: RegimeEffect


def compare_outcomes(
    treated: OutcomeSnapshot,
    control: OutcomeSnapshot,
    *,
    estimand: str = "market_regime_total_effect",
    weights: HarmWeights | None = None,
) -> tuple[PairedOutcome, RegimeEffect]:
    """Compare structurally paired worlds without regressing away vulnerability."""

    if treated.player_harm.shape != control.player_harm.shape:
        raise ValueError("paired worlds must contain the same players and harm dimensions")
    if treated.firm_cash_cents.shape != control.firm_cash_cents.shape:
        raise ValueError("paired worlds must contain the same firms")
    if treated.state_subsidy_outlay_cents.shape != control.state_subsidy_outlay_cents.shape:
        raise ValueError("paired worlds must contain the same jurisdictions")

    paired = PairedOutcome(
        player_harm_difference=treated.player_harm - control.player_harm,
        player_spend_difference_cents=(
            treated.player_spend_cents - control.player_spend_cents
        ),
        player_debt_difference_cents=(
            treated.player_debt_cents - control.player_debt_cents
        ),
        firm_margin_difference_cents=(
            treated.firm_operating_margin_cents
            - control.firm_operating_margin_cents
        ),
        firm_cash_difference_cents=treated.firm_cash_cents - control.firm_cash_cents,
        state_subsidy_difference_cents=(
            treated.state_subsidy_outlay_cents
            - control.state_subsidy_outlay_cents
        ),
    )
    weight_array = (weights or HarmWeights()).as_array()
    individual_composite = paired.player_harm_difference @ (
        weight_array / weight_array.sum()
    )
    affected_share = (
        float(np.count_nonzero(np.abs(individual_composite) > 1e-12))
        / len(individual_composite)
        if len(individual_composite)
        else 0.0
    )
    effect = RegimeEffect(
        estimand=estimand,
        mean_composite_harm_effect=(
            float(individual_composite.mean()) if len(individual_composite) else 0.0
        ),
        total_spend_effect_cents=int(
            paired.player_spend_difference_cents.sum(dtype=np.int64)
        ),
        total_debt_effect_cents=int(
            paired.player_debt_difference_cents.sum(dtype=np.int64)
        ),
        total_operating_margin_effect_cents=int(
            paired.firm_margin_difference_cents.sum(dtype=np.int64)
        ),
        total_subsidy_effect_cents=int(
            paired.state_subsidy_difference_cents.sum(dtype=np.int64)
        ),
        affected_player_share=affected_share,
    )
    return paired, effect


def run_paired_worlds(
    config: SimulationConfig,
    *,
    treated: Intervention,
    control: Intervention | None = None,
    cycles: int | None = None,
    campaign: bool = False,
    profiles: ProfileBundle | None = None,
) -> PairedWorldRun:
    """Run an explicit treated/control pair with common random numbers.

    Each branch owns separate mutable state. Counter-based random streams share
    coordinates, so an action occurring only in one branch cannot shift later
    exogenous draws in the other branch.
    """

    if not config.causal.common_random_numbers:
        raise ValueError("paired worlds require common_random_numbers=true")
    control_intervention = control or NullIntervention()
    treated_world = World.create(config, profiles=profiles, campaign=campaign)
    control_world = World.create(config, profiles=profiles, campaign=campaign)
    _assert_structural_pair(treated_world, control_world)

    treated.apply(treated_world)
    control_intervention.apply(control_world)
    treated_run = SimulationEngine.run(
        treated_world, cycles=cycles, campaign=campaign
    )
    control_run = SimulationEngine.run(
        control_world, cycles=cycles, campaign=campaign
    )
    paired, effect = compare_outcomes(
        treated_run.final_outcome,
        control_run.final_outcome,
        estimand=config.causal.estimand,
    )
    return PairedWorldRun(
        treated_run=treated_run,
        control_run=control_run,
        paired_outcome=paired,
        effect=effect,
    )


def _assert_structural_pair(treated: World, control: World) -> None:
    """Fail before treatment if latent populations are not exactly paired."""

    player_columns = (
        "player_id",
        "age_years",
        "jurisdiction",
        "household_id",
        "is_minor",
        "monthly_disposable_income_cents",
        "allowance_cents",
        "has_stored_payment_access",
        "guardian_supervision",
        "guardian_consent",
        "traits",
        "motive_weights",
        "baseline_vulnerability",
    )
    game_columns = (
        "game_id",
        "company_id",
        "quality",
        "competitive_integrity",
        "novelty",
        "monetisation",
        "stat_frontier",
        "price_cents",
    )
    for name in player_columns:
        if not np.array_equal(
            getattr(treated.players, name), getattr(control.players, name)
        ):
            raise ValueError(f"paired player column differs before treatment: {name}")
    for name in game_columns:
        if not np.array_equal(
            getattr(treated.games, name), getattr(control.games, name)
        ):
            raise ValueError(f"paired game column differs before treatment: {name}")
    if tuple(firm.firm_id for firm in treated.firms) != tuple(
        firm.firm_id for firm in control.firms
    ):
        raise ValueError("paired firm identities differ before treatment")
    if tuple(state.code for state in treated.states) != tuple(
        state.code for state in control.states
    ):
        raise ValueError("paired jurisdiction identities differ before treatment")
