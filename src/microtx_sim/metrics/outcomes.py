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
        with np.errstate(over="ignore", invalid="ignore"):
            total = float(result.sum())
        if not np.isfinite(total):
            raise ValueError("harm weight sum must be finite")
        if total <= 0.0:
            raise ValueError("at least one harm weight must be positive")
        return result


@dataclass(frozen=True, slots=True)
class OutcomeSnapshot:
    """Research output for a single tick.

    Harm dimensions are never collapsed destructively. The composite index is an
    explicitly weighted reporting view over the stored matrix.
    """

    tick: int
    player_ids: IntArray
    player_harm: FloatArray
    player_spend_cents: IntArray
    player_income_cents: IntArray
    player_debt_cents: IntArray
    firm_ids: IntArray
    firm_cash_cents: IntArray
    firm_operating_margin_cents: IntArray
    firm_safe_revenue_share: FloatArray
    jurisdiction_ids: IntArray
    state_subsidy_outlay_cents: IntArray

    def __post_init__(self) -> None:
        array_columns = (
            "player_ids",
            "player_harm",
            "player_spend_cents",
            "player_income_cents",
            "player_debt_cents",
            "firm_ids",
            "firm_cash_cents",
            "firm_operating_margin_cents",
            "firm_safe_revenue_share",
            "jurisdiction_ids",
            "state_subsidy_outlay_cents",
        )
        for name in array_columns:
            if type(getattr(self, name)) is not np.ndarray:
                raise TypeError(f"{name} must be a numpy array")

        identity_columns = {
            "player_ids": self.player_ids,
            "firm_ids": self.firm_ids,
            "jurisdiction_ids": self.jurisdiction_ids,
        }
        for name, values in identity_columns.items():
            if type(values) is not np.ndarray or values.dtype != np.dtype(np.int64):
                raise TypeError(f"{name} must be an int64 numpy array")
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if np.any(values < 0):
                raise ValueError(f"{name} must be non-negative")
            if not _has_unique_int64_ids(values):
                raise ValueError(f"{name} must be unique")

        players = len(self.player_ids)
        _validate_player_harm(self.player_harm, expected_players=players)
        if (
            len(self.player_spend_cents) != players
            or len(self.player_income_cents) != players
            or len(self.player_debt_cents) != players
        ):
            raise ValueError("player outcome columns have inconsistent lengths")
        firms = len(self.firm_ids)
        if (
            len(self.firm_cash_cents) != firms
            or len(self.firm_operating_margin_cents) != firms
            or len(self.firm_safe_revenue_share) != firms
        ):
            raise ValueError("firm outcome columns have inconsistent lengths")
        if len(self.state_subsidy_outlay_cents) != len(self.jurisdiction_ids):
            raise ValueError("jurisdiction outcome columns have inconsistent lengths")
        for name in array_columns:
            object.__setattr__(
                self,
                name,
                _immutable_array_copy(getattr(self, name)),
            )

    def composite_harm(self, weights: HarmWeights | None = None) -> FloatArray:
        _validate_player_harm(
            self.player_harm,
            expected_players=len(self.player_ids),
        )
        weight_array = (weights or HarmWeights()).as_array()
        with np.errstate(over="ignore", invalid="ignore"):
            composite = self.player_harm @ (
                weight_array / weight_array.sum()
            )
        if not np.all(np.isfinite(composite)):
            raise OverflowError("weighted composite harm is not finite")
        return composite

    def summary(self) -> dict[str, float | int]:
        composite = self.composite_harm()
        with np.errstate(over="ignore", invalid="ignore"):
            mean_composite = (
                float(composite.mean()) if len(composite) else 0.0
            )
        if not np.isfinite(mean_composite):
            raise OverflowError("mean composite harm is not finite")
        return {
            "tick": self.tick,
            "players": len(self.player_spend_cents),
            "total_spend_cents": sum(
                int(value) for value in self.player_spend_cents
            ),
            "players_with_debt": int(np.count_nonzero(self.player_debt_cents > 0)),
            "mean_composite_harm": mean_composite,
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
        if type(snapshot) is not OutcomeSnapshot:
            raise TypeError("snapshot must be an OutcomeSnapshot")
        if self._summaries and snapshot.tick <= int(self._summaries[-1]["tick"]):
            raise ValueError("outcome ticks must increase strictly")
        self._summaries.append(snapshot.summary())
        if self.record_individual:
            self._latest = snapshot


def _validate_player_harm(
    values: object,
    *,
    expected_players: int,
) -> None:
    if type(values) is not np.ndarray or values.dtype != np.dtype(np.float64):
        raise TypeError("player_harm must be a float64 numpy array")
    if values.shape != (expected_players, 7):
        raise ValueError(
            "player_harm must have one row and seven dimensions per player"
        )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("player_harm must be finite and in [0, 1]")


def _has_unique_int64_ids(values: IntArray) -> bool:
    if values.size < 2 or bool(np.all(values[1:] > values[:-1])):
        return True
    return np.unique(values).size == values.size


def _immutable_array_copy(values: np.ndarray) -> np.ndarray:
    """Return an independent C-order array backed by immutable bytes."""

    copied = np.frombuffer(values.tobytes(order="C"), dtype=values.dtype)
    return copied.reshape(values.shape)
