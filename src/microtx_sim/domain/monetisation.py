"""Explicit research variables for a game's monetisation configuration.

The fields in :class:`MonetisationVector` are abstract intervention coordinates,
not commercial optimisation advice.  Risk-oriented intensities are normalised
to ``[0, 1]``.  Monetary and temporal safeguards retain interpretable cent/hour
units and are applied by explicit methods rather than hidden utility constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


_INT64_MAX = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class MonetisationVector:
    """One transparent, immutable monetisation intervention vector.

    ``purchase_friction`` is the only continuous coordinate where larger means
    safer/slower.  All other continuous mechanism coordinates measure greater
    exposure when larger.  ``spending_cap_cents=None`` means that no cap is
    active; a cap of zero prohibits monetary purchases.  Personalised offers
    are deliberately disabled by default.
    """

    direct_price_cents: int = 0
    opaque_virtual_currency: float = 0.0
    paid_random_rewards: float = 0.0
    progression_gates: float = 0.0
    time_limited_offers: float = 0.0
    daily_streak_pressure: float = 0.0
    pay_to_progress: float = 0.0
    pay_to_win: float = 0.0
    social_guild_pressure: float = 0.0
    purchase_friction: float = 1.0
    spending_cap_cents: int | None = None
    cooling_off_hours: int = 0
    real_currency_price_display: bool = True
    personalized_offers: bool = False

    def __post_init__(self) -> None:
        _nonnegative_int(self.direct_price_cents, "direct_price_cents")
        if self.spending_cap_cents is not None:
            _nonnegative_int(self.spending_cap_cents, "spending_cap_cents")
        _nonnegative_int(self.cooling_off_hours, "cooling_off_hours")
        for name in (
            "opaque_virtual_currency",
            "paid_random_rewards",
            "progression_gates",
            "time_limited_offers",
            "daily_streak_pressure",
            "pay_to_progress",
            "pay_to_win",
            "social_guild_pressure",
            "purchase_friction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not isinstance(self.real_currency_price_display, bool):
            raise TypeError("real_currency_price_display must be boolean")
        if not isinstance(self.personalized_offers, bool):
            raise TypeError("personalized_offers must be boolean")

    @property
    def price_transparency(self) -> float:
        """Return an interpretable ``[0, 1]`` real-price transparency score."""

        real_display = 1.0 if self.real_currency_price_display else 0.0
        return 0.55 * (1.0 - float(self.opaque_virtual_currency)) + 0.45 * real_display

    @property
    def purchase_pressure(self) -> float:
        """Return a documented weighted mean of purchase-pressure mechanisms."""

        # Weights sum exactly to one.  Paid randomness and limited-time offers
        # receive the largest weights because they combine transaction demand
        # with uncertain reward or time pressure in the welfare model.
        return float(
            0.08 * self.opaque_virtual_currency
            + 0.15 * self.paid_random_rewards
            + 0.12 * self.progression_gates
            + 0.14 * self.time_limited_offers
            + 0.12 * self.daily_streak_pressure
            + 0.11 * self.pay_to_progress
            + 0.11 * self.pay_to_win
            + 0.10 * self.social_guild_pressure
            + 0.07 * (1.0 - self.purchase_friction)
        )

    @property
    def risk_exposure(self) -> float:
        """Return pressure/opacity exposure after structural safeguards.

        An active cap and a cooling-off period receive conservative bounded
        mitigation credits.  The absolute adequacy of a cap is budget-specific
        and must still be evaluated with :meth:`constrain_purchase` and player
        resources; this aggregate is only a comparable research coordinate.
        """

        raw = (
            0.68 * self.purchase_pressure
            + 0.27 * (1.0 - self.price_transparency)
            + 0.05 * float(self.personalized_offers)
        )
        cap_mitigation = 0.15 if self.spending_cap_cents is not None else 0.0
        cooling_mitigation = min(float(self.cooling_off_hours) / 168.0, 1.0) * 0.15
        return min(1.0, max(0.0, raw * (1.0 - cap_mitigation - cooling_mitigation)))

    def remaining_spending_cap_cents(self, already_spent_cents: int) -> int | None:
        """Return remaining cap capacity, or ``None`` when no cap is active."""

        spent = _nonnegative_int(already_spent_cents, "already_spent_cents")
        if self.spending_cap_cents is None:
            return None
        return max(0, self.spending_cap_cents - spent)

    def cooling_off_active(self, hours_since_last_purchase: float | None) -> bool:
        """Whether a prior purchase still blocks another monetary transaction."""

        if hours_since_last_purchase is None or self.cooling_off_hours == 0:
            return False
        if isinstance(hours_since_last_purchase, bool) or not isinstance(
            hours_since_last_purchase, Real
        ):
            raise TypeError("hours_since_last_purchase must be real or None")
        hours = float(hours_since_last_purchase)
        if not isfinite(hours) or hours < 0.0:
            raise ValueError("hours_since_last_purchase must be finite and non-negative")
        return hours < self.cooling_off_hours

    def constrain_purchase(
        self,
        proposed_cents: int,
        *,
        already_spent_cents: int = 0,
        hours_since_last_purchase: float | None = None,
    ) -> int:
        """Apply cooling-off and spending-cap constraints to a proposed amount."""

        proposed = _nonnegative_int(proposed_cents, "proposed_cents")
        if self.cooling_off_active(hours_since_last_purchase):
            return 0
        remaining = self.remaining_spending_cap_cents(already_spent_cents)
        return proposed if remaining is None else min(proposed, remaining)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} cannot be negative")
    if result > _INT64_MAX:
        raise OverflowError(f"{name} must fit signed int64 cents/hours")
    return result


__all__ = ["MonetisationVector"]
