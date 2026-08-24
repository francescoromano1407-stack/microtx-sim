from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..metrics.outcomes import HarmWeights, OutcomeSnapshot


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

