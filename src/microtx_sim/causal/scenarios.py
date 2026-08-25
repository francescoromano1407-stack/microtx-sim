"""Named policy scenarios for the synthetic EU-GAME-HARM prototype.

The values in this catalogue are transparent *illustrative assumptions*. They
are not estimates of any commercial game and must not be interpreted as design
advice. Scenario construction is kept separate from the decision engine so a
counterfactual changes only the declared intervention vector.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from ..domain.monetisation import MonetisationVector


class ScenarioId(str, Enum):
    """Stable identifiers for the seven required counterfactual regimes."""

    BASELINE_F2P = "baseline_f2p"
    TRANSPARENT_DIRECT_PRICE = "transparent_direct_price"
    NO_RANDOM_REWARDS = "no_random_rewards"
    NO_TIME_LIMITED_PRESSURE = "no_time_limited_pressure"
    SPENDING_CAP_COOLING_OFF = "spending_cap_cooling_off"
    SAFE_FIXED_PRICE_SUBSCRIPTION = "safe_fixed_price_subscription"
    EPGC = "epgc"


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """One explicit monetisation and financing intervention."""

    scenario_id: ScenarioId
    label: str
    mechanics: MonetisationVector
    fixed_access_price_cents: int = 0
    subscription_price_cents: int = 0
    epgc_enabled: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if type(self.scenario_id) is not ScenarioId:
            raise TypeError("scenario_id must be a ScenarioId")
        if type(self.label) is not str:
            raise TypeError("label must be a string")
        if not self.label.strip():
            raise ValueError("scenario label cannot be empty")
        if type(self.mechanics) is not MonetisationVector:
            raise TypeError("mechanics must be a MonetisationVector")
        for name in ("fixed_access_price_cents", "subscription_price_cents"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be integer cents")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.epgc_enabled, bool):
            raise TypeError("epgc_enabled must be a boolean")
        if type(self.description) is not str:
            raise TypeError("description must be a string")
        if self.mechanics.personalized_offers:
            raise ValueError(
                "research scenarios keep personalized offers disabled"
            )


def required_scenarios() -> tuple[ScenarioSpec, ...]:
    """Return the canonical seven-scenario policy comparison.

    The first five regimes use the same purchase price and differ only in the
    named safety intervention where practical. The last two are alternative
    business models and therefore change the revenue structure as well.
    """

    baseline = MonetisationVector(
        direct_price_cents=299,
        opaque_virtual_currency=0.75,
        paid_random_rewards=0.70,
        progression_gates=0.55,
        time_limited_offers=0.70,
        daily_streak_pressure=0.60,
        pay_to_progress=0.55,
        pay_to_win=0.50,
        social_guild_pressure=0.50,
        purchase_friction=0.20,
        spending_cap_cents=None,
        cooling_off_hours=0,
        real_currency_price_display=False,
        personalized_offers=False,
    )
    safe = MonetisationVector(
        direct_price_cents=0,
        opaque_virtual_currency=0.0,
        paid_random_rewards=0.0,
        progression_gates=0.05,
        time_limited_offers=0.0,
        daily_streak_pressure=0.05,
        pay_to_progress=0.0,
        pay_to_win=0.0,
        social_guild_pressure=0.10,
        purchase_friction=0.85,
        spending_cap_cents=2_500,
        cooling_off_hours=24,
        real_currency_price_display=True,
        personalized_offers=False,
    )
    return (
        ScenarioSpec(
            ScenarioId.BASELINE_F2P,
            "Baseline free-to-play",
            baseline,
            description="Illustrative high-pressure free-to-play reference.",
        ),
        ScenarioSpec(
            ScenarioId.TRANSPARENT_DIRECT_PRICE,
            "Transparent direct price",
            replace(
                baseline,
                opaque_virtual_currency=0.0,
                purchase_friction=0.65,
                real_currency_price_display=True,
            ),
            description="Pricing presentation changes; other pressures remain.",
        ),
        ScenarioSpec(
            ScenarioId.NO_RANDOM_REWARDS,
            "No paid randomized rewards",
            replace(baseline, paid_random_rewards=0.0),
            description="Paid randomized rewards are removed in isolation.",
        ),
        ScenarioSpec(
            ScenarioId.NO_TIME_LIMITED_PRESSURE,
            "No time-limited purchase pressure",
            replace(baseline, time_limited_offers=0.0),
            description="Purchase deadlines are removed in isolation.",
        ),
        ScenarioSpec(
            ScenarioId.SPENDING_CAP_COOLING_OFF,
            "Spending cap and cooling-off",
            replace(
                baseline,
                spending_cap_cents=2_500,
                cooling_off_hours=24,
                purchase_friction=0.70,
                real_currency_price_display=True,
            ),
            description="A binding rolling cap and time-based pause are applied.",
        ),
        ScenarioSpec(
            ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
            "Safe fixed-price or subscription",
            safe,
            fixed_access_price_cents=1_499,
            subscription_price_cents=499,
            description="Revenue comes from transparent access, not pressure.",
        ),
        ScenarioSpec(
            ScenarioId.EPGC,
            "European Public-Value Game Contract",
            safe,
            fixed_access_price_cents=299,
            epgc_enabled=True,
            description="Safe access is supplemented by a capped public contract.",
        ),
    )


def scenario_by_id(scenario_id: ScenarioId | str) -> ScenarioSpec:
    """Resolve a stable identifier or raise a clear error."""

    wanted = ScenarioId(scenario_id)
    for scenario in required_scenarios():
        if scenario.scenario_id is wanted:
            return scenario
    raise AssertionError(f"unreachable scenario id: {wanted}")


__all__ = ["ScenarioId", "ScenarioSpec", "required_scenarios", "scenario_by_id"]
