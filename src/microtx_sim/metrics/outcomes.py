from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class HarmWeights:
    financial_stress: float = 1.0
    essential_displacement: float = 1.0
    debt: float = 1.0
    unauthorised_spend: float = 1.0
    loss_of_control: float = 1.0
    functioning_impairment: float = 1.0
    regret: float = 1.0

    def as_array(self) -> FloatArray:
        result = np.asarray(
            (
                self.financial_stress,
                self.essential_displacement,
                self.debt,
                self.unauthorised_spend,
                self.loss_of_control,
                self.functioning_impairment,
                self.regret,
            ),
            dtype=np.float64,
        )
        if np.any(result < 0.0) or not np.all(np.isfinite(result)):
            raise ValueError("harm weights must be finite and non-negative")
        if result.sum() <= 0.0:
            raise ValueError("at least one harm weight must be positive")
        return result


@dataclass(frozen=True, slots=True)
class OutcomeSnapshot:
    """Research output for a single tick.

    Harm dimensions are never collapsed destructively. The composite index is an
    explicitly weighted reporting view over the stored matrix.
    """

    tick: int
    player_harm: FloatArray
    player_spend_cents: IntArray
    player_income_cents: IntArray
    player_debt_cents: IntArray
    firm_cash_cents: IntArray
    firm_operating_margin_cents: IntArray
    firm_safe_revenue_share: FloatArray
    state_subsidy_outlay_cents: IntArray

    def __post_init__(self) -> None:
        players = len(self.player_spend_cents)
        if self.player_harm.shape != (players, 7):
            raise ValueError("player_harm must have one row and seven dimensions per player")
        if len(self.player_income_cents) != players or len(self.player_debt_cents) != players:
            raise ValueError("player outcome columns have inconsistent lengths")
        firms = len(self.firm_cash_cents)
        if (
            len(self.firm_operating_margin_cents) != firms
            or len(self.firm_safe_revenue_share) != firms
        ):
            raise ValueError("firm outcome columns have inconsistent lengths")

    def composite_harm(self, weights: HarmWeights | None = None) -> FloatArray:
        weight_array = (weights or HarmWeights()).as_array()
        return self.player_harm @ (weight_array / weight_array.sum())

    def summary(self) -> dict[str, float | int]:
        composite = self.composite_harm()
        return {
            "tick": self.tick,
            "players": len(self.player_spend_cents),
            "total_spend_cents": sum(
                int(value) for value in self.player_spend_cents
            ),
            "players_with_debt": int(np.count_nonzero(self.player_debt_cents > 0)),
            "mean_composite_harm": float(composite.mean()) if len(composite) else 0.0,
            "p99_composite_harm": (
                float(np.quantile(composite, 0.99)) if len(composite) else 0.0
            ),
            "solvent_firms": int(np.count_nonzero(self.firm_cash_cents >= 0)),
            "mean_safe_revenue_share": (
                float(self.firm_safe_revenue_share.mean())
                if len(self.firm_safe_revenue_share)
                else 0.0
            ),
            "subsidy_outlay_cents": sum(
                int(value) for value in self.state_subsidy_outlay_cents
            ),
        }


class OutcomeRecorder:
    """Stores aggregate time series plus optional individual final snapshots."""

    __slots__ = ("record_individual", "_summaries", "_latest")

    def __init__(self, *, record_individual: bool = True) -> None:
        self.record_individual = record_individual
        self._summaries: list[dict[str, float | int]] = []
        self._latest: OutcomeSnapshot | None = None

    @property
    def summaries(self) -> tuple[dict[str, float | int], ...]:
        return tuple(dict(item) for item in self._summaries)

    @property
    def latest(self) -> OutcomeSnapshot | None:
        return self._latest

    def record(self, snapshot: OutcomeSnapshot) -> None:
        if self._summaries and snapshot.tick <= int(self._summaries[-1]["tick"]):
            raise ValueError("outcome ticks must increase strictly")
        self._summaries.append(snapshot.summary())
        if self.record_individual:
            self._latest = OutcomeSnapshot(
                tick=snapshot.tick,
                player_harm=snapshot.player_harm.copy(),
                player_spend_cents=snapshot.player_spend_cents.copy(),
                player_income_cents=snapshot.player_income_cents.copy(),
                player_debt_cents=snapshot.player_debt_cents.copy(),
                firm_cash_cents=snapshot.firm_cash_cents.copy(),
                firm_operating_margin_cents=snapshot.firm_operating_margin_cents.copy(),
                firm_safe_revenue_share=snapshot.firm_safe_revenue_share.copy(),
                state_subsidy_outlay_cents=snapshot.state_subsidy_outlay_cents.copy(),
            )
