from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Final

import numpy as np
from numpy.typing import NDArray

from ..types import HarmDimension, Motive, SpendSegment


TRAIT_NAMES: Final[tuple[str, ...]] = (
    "impulsivity",
    "reward_sensitivity",
    "social_susceptibility",
    "loss_aversion",
    "financial_literacy",
    "self_control",
)

_MONEY_COLUMNS: Final[tuple[str, ...]] = (
    "monthly_disposable_income_cents",
    "liquidity_cents",
    "credit_limit_cents",
    "allowance_cents",
    "household_liquidity_cents",
)


@dataclass(frozen=True, slots=True)
class PlayerTable:
    """Structure-of-arrays state for a heterogeneous player population.

    The dataclass is frozen so columns cannot accidentally be replaced while a
    simulation is running.  Dynamic columns (for example ``harm_state``) remain
    mutable NumPy arrays.  ``baseline_vulnerability`` is additionally copied
    and write-protected because it is a pre-treatment covariate in the causal
    design.

    Jurisdictions are integer codes into ``jurisdiction_codes``.  Motive and
    harm columns use the ordering of :class:`~microtx_sim.types.Motive` and
    :class:`~microtx_sim.types.HarmDimension`, respectively.
    """

    player_id: NDArray[np.int64]
    age_years: NDArray[np.int16]
    jurisdiction: NDArray[np.int16]
    household_id: NDArray[np.int64]
    is_minor: NDArray[np.bool_]

    monthly_disposable_income_cents: NDArray[np.int64]
    liquidity_cents: NDArray[np.int64]
    credit_limit_cents: NDArray[np.int64]
    allowance_cents: NDArray[np.int64]
    household_liquidity_cents: NDArray[np.int64]

    has_stored_payment_access: NDArray[np.bool_]
    guardian_supervision: NDArray[np.float32]
    guardian_consent: NDArray[np.bool_]

    traits: NDArray[np.float32]
    motive_weights: NDArray[np.float32]
    baseline_vulnerability: NDArray[np.float32]
    harm_state: NDArray[np.float32]

    current_game: NDArray[np.int32]
    awareness: NDArray[np.float32]

    jurisdiction_codes: tuple[str, ...]
    adult_age_by_jurisdiction: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.player_id.ndim != 1:
            raise ValueError("player_id must be one-dimensional")
        n_players = self.player_id.shape[0]

        one_dimensional = (
            "age_years",
            "jurisdiction",
            "household_id",
            "is_minor",
            *_MONEY_COLUMNS,
            "has_stored_payment_access",
            "guardian_supervision",
            "guardian_consent",
            "baseline_vulnerability",
            "current_game",
            "awareness",
        )
        for name in one_dimensional:
            value = getattr(self, name)
            if value.ndim != 1 or value.shape[0] != n_players:
                raise ValueError(f"{name} must have shape ({n_players},)")

        expected_dtypes: dict[str, np.dtype[object]] = {
            "player_id": np.dtype(np.int64),
            "age_years": np.dtype(np.int16),
            "jurisdiction": np.dtype(np.int16),
            "household_id": np.dtype(np.int64),
            "is_minor": np.dtype(np.bool_),
            **{name: np.dtype(np.int64) for name in _MONEY_COLUMNS},
            "has_stored_payment_access": np.dtype(np.bool_),
            "guardian_supervision": np.dtype(np.float32),
            "guardian_consent": np.dtype(np.bool_),
            "traits": np.dtype(np.float32),
            "motive_weights": np.dtype(np.float32),
            "baseline_vulnerability": np.dtype(np.float32),
            "harm_state": np.dtype(np.float32),
            "current_game": np.dtype(np.int32),
            "awareness": np.dtype(np.float32),
        }
        for name, dtype in expected_dtypes.items():
            if getattr(self, name).dtype != dtype:
                raise TypeError(f"{name} must use dtype {dtype}")

        if self.traits.shape != (n_players, len(TRAIT_NAMES)):
            raise ValueError(
                f"traits must have shape ({n_players}, {len(TRAIT_NAMES)})"
            )
        if self.motive_weights.shape != (n_players, len(Motive)):
            raise ValueError(
                f"motive_weights must have shape ({n_players}, {len(Motive)})"
            )
        if self.harm_state.shape != (n_players, len(HarmDimension)):
            raise ValueError(
                f"harm_state must have shape ({n_players}, {len(HarmDimension)})"
            )

        if len(self.jurisdiction_codes) != len(self.adult_age_by_jurisdiction):
            raise ValueError("jurisdiction metadata lengths differ")
        if len(set(self.jurisdiction_codes)) != len(self.jurisdiction_codes):
            raise ValueError("jurisdiction codes must be unique")
        if n_players:
            if np.unique(self.player_id).size != n_players:
                raise ValueError("player_id values must be unique")
            if np.any(self.age_years < 0):
                raise ValueError("age_years cannot be negative")
            if np.any(self.household_id < 0):
                raise ValueError("household_id cannot be negative")
            if np.any(self.jurisdiction < 0) or np.any(
                self.jurisdiction >= len(self.jurisdiction_codes)
            ):
                raise ValueError("jurisdiction contains an unknown code")

            adult_ages = np.asarray(self.adult_age_by_jurisdiction, dtype=np.int16)
            expected_minor = self.age_years < adult_ages[self.jurisdiction]
            if not np.array_equal(self.is_minor, expected_minor):
                raise ValueError("is_minor is inconsistent with age and jurisdiction")

        for name in _MONEY_COLUMNS:
            if np.any(getattr(self, name) < 0):
                raise ValueError(f"{name} cannot be negative")
        for name in (
            "guardian_supervision",
            "traits",
            "motive_weights",
            "baseline_vulnerability",
            "awareness",
        ):
            value = getattr(self, name)
            if not np.all(np.isfinite(value)) or np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError(f"{name} values must be finite and in [0, 1]")
        if not np.all(np.isfinite(self.harm_state)) or np.any(self.harm_state < 0.0):
            raise ValueError("harm_state must contain finite, non-negative values")
        if n_players and not np.allclose(
            self.motive_weights.sum(axis=1), 1.0, atol=2e-6
        ):
            raise ValueError("every motive_weights row must sum to one")
        if np.any(self.allowance_cents[~self.is_minor] != 0):
            raise ValueError("allowance_cents is reserved for minors")
        if np.any(self.credit_limit_cents[self.is_minor] != 0):
            raise ValueError(
                "minors cannot own credit; stored-card access is represented separately"
            )
        if np.any(self.guardian_supervision[~self.is_minor] != 0.0):
            raise ValueError("guardian_supervision is reserved for minors")
        if np.any(self.guardian_consent[~self.is_minor]):
            raise ValueError("guardian_consent is reserved for minors")
        if np.any(self.current_game < -1):
            raise ValueError("current_game must be -1 (none) or a non-negative game id")

        immutable_baseline = np.array(
            self.baseline_vulnerability, dtype=np.float32, copy=True
        )
        immutable_baseline.flags.writeable = False
        object.__setattr__(self, "baseline_vulnerability", immutable_baseline)

    def __len__(self) -> int:
        return int(self.player_id.size)

    def trait(self, name: str) -> NDArray[np.float32]:
        """Return a zero-copy view of a named continuous trait."""

        try:
            index = TRAIT_NAMES.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown trait: {name}") from exc
        return self.traits[:, index]

    def motive(self, motive: Motive | int) -> NDArray[np.float32]:
        """Return a zero-copy view of one motive weight."""

        return self.motive_weights[:, int(motive)]

    def classify_spending(
        self,
        spend_cents: NDArray[np.integer] | list[int],
        *,
        income_cents: NDArray[np.integer] | list[int] | None = None,
        whale_quantile: float = 0.99,
        whale_income_share: float = 0.10,
    ) -> NDArray[np.str_]:
        """Classify observed period spending without creating player types.

        If ``income_cents`` is omitted, monthly disposable income is used, so
        ``spend_cents`` should then cover the same monthly period.
        """

        denominator = (
            self.monthly_disposable_income_cents
            if income_cents is None
            else income_cents
        )
        return classify_spend_segments(
            spend_cents,
            denominator,
            whale_quantile=whale_quantile,
            whale_income_share=whale_income_share,
        )

    @property
    def nbytes(self) -> int:
        """Memory owned by NumPy columns (metadata excluded)."""

        total = 0
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if isinstance(value, np.ndarray):
                total += value.nbytes
        return total


def classify_spend_segments(
    spend_cents: NDArray[np.integer] | list[int],
    disposable_income_cents: NDArray[np.integer] | list[int],
    *,
    whale_quantile: float = 0.99,
    whale_income_share: float = 0.10,
) -> NDArray[np.str_]:
    """Return retrospective spend segments for one observation window.

    ``whale`` is assigned only when spending is both in the configured upper
    tail of the *observed payer distribution* and large relative to that
    player's disposable income.  Consequently the label can change across
    windows and is never an intrinsic player attribute.
    """

    spend = np.asarray(spend_cents)
    income = np.asarray(disposable_income_cents)
    if spend.ndim != 1 or income.ndim != 1 or spend.shape != income.shape:
        raise ValueError("spend_cents and disposable_income_cents need equal 1-D shapes")
    if not np.issubdtype(spend.dtype, np.integer) or not np.issubdtype(
        income.dtype, np.integer
    ):
        raise TypeError("spending and income must be integer cents")
    if np.any(spend < 0) or np.any(income < 0):
        raise ValueError("spending and income cannot be negative")
    if not 0.0 < whale_quantile < 1.0:
        raise ValueError("whale_quantile must be in (0, 1)")
    if whale_income_share < 0.0:
        raise ValueError("whale_income_share cannot be negative")

    segments = np.full(spend.shape, SpendSegment.NON_PAYER.value, dtype="<U9")
    payer = spend > 0
    if not np.any(payer):
        return segments

    payer_spend = spend[payer]
    payer_median = float(np.median(payer_spend))
    segments[payer & (spend <= payer_median)] = SpendSegment.MINNOW.value
    segments[payer & (spend > payer_median)] = SpendSegment.DOLPHIN.value

    upper_tail = float(np.quantile(payer_spend, whale_quantile))
    spend_share = np.divide(
        spend.astype(np.float64),
        income.astype(np.float64),
        out=np.full(spend.shape, np.inf, dtype=np.float64),
        where=income > 0,
    )
    whale = payer & (spend >= upper_tail) & (spend_share >= whale_income_share)
    segments[whale] = SpendSegment.WHALE.value
    return segments
