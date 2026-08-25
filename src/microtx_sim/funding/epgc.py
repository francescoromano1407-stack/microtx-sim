"""Pure European Public-Value Game Contract (EPGC) financing model.

The model intentionally uses only eligibility, availability, public-value
features, conventional revenue, and production costs.  It is independent of
the behavioural simulation and has no agent-state side effects.

The reported sustainability equation is exactly::

    Profit_safe = (
        PublicContractRevenue
        + FixedPriceRevenue
        + InstitutionalLicensingRevenue
        + NonTargetedSponsorshipRevenue
        - DevelopmentCost
        - MaintenanceCost
    )

All monetary quantities are signed-64-bit-compatible integer cents.  Python's
integer arithmetic is used for intermediate calculations, followed by explicit
range checks; no floating-point money is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass


INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
_BASIS_POINTS = 10_000


def _validate_nonnegative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0 or value > INT64_MAX:
        raise ValueError(f"{name} must be in [0, {INT64_MAX}]")


def _validate_bool(value: bool, *, name: str) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")


def _checked_sum(*values: int, label: str) -> int:
    total = sum(values)
    if total < 0 or total > INT64_MAX:
        raise OverflowError(f"{label} is outside the non-negative int64 range")
    return total


def _checked_product(left: int, right: int, *, label: str) -> int:
    result = left * right
    if result < 0 or result > INT64_MAX:
        raise OverflowError(f"{label} is outside the non-negative int64 range")
    return result


def _checked_signed(value: int, *, label: str) -> int:
    if value < INT64_MIN or value > INT64_MAX:
        raise OverflowError(f"{label} is outside the signed int64 range")
    return value


@dataclass(frozen=True, slots=True)
class EPGCPolicy:
    """Public payment schedule and enforcement terms for one EPGC regime."""

    access_payment_cents_per_eligible_access: int
    institutional_license_payment_cents_per_license: int
    availability_payment_cents_per_period: int
    accessibility_bonus_cents: int
    multilingual_bonus_cents: int
    cultural_value_bonus_cents: int
    safety_certification_bonus_cents: int
    prohibited_mechanics_penalty_cents: int
    prohibited_mechanics_clawback_basis_points: int
    maximum_budget_cents: int

    def __post_init__(self) -> None:
        for name in (
            "access_payment_cents_per_eligible_access",
            "institutional_license_payment_cents_per_license",
            "availability_payment_cents_per_period",
            "accessibility_bonus_cents",
            "multilingual_bonus_cents",
            "cultural_value_bonus_cents",
            "safety_certification_bonus_cents",
            "prohibited_mechanics_penalty_cents",
            "maximum_budget_cents",
        ):
            value = getattr(self, name)
            _validate_nonnegative_int(value, name=name)
            object.__setattr__(self, name, int(value))
        clawback_basis_points = self.prohibited_mechanics_clawback_basis_points
        _validate_nonnegative_int(
            clawback_basis_points,
            name="prohibited_mechanics_clawback_basis_points",
        )
        object.__setattr__(
            self,
            "prohibited_mechanics_clawback_basis_points",
            int(clawback_basis_points),
        )
        if not 1 <= self.prohibited_mechanics_clawback_basis_points <= _BASIS_POINTS:
            raise ValueError(
                "prohibited_mechanics_clawback_basis_points must be in [1, 10000]"
            )
        if self.prohibited_mechanics_penalty_cents <= 0:
            raise ValueError("prohibited_mechanics_penalty_cents must be positive")


@dataclass(frozen=True, slots=True)
class EPGCFirmInputs:
    """Cost, conventional-revenue, eligibility, and certification assumptions."""

    fixed_price_revenue_cents: int
    institutional_licensing_revenue_cents: int
    non_targeted_sponsorship_revenue_cents: int
    development_cost_cents: int
    maintenance_cost_cents: int
    eligible_access_count: int
    eligible_institutional_license_count: int
    availability_period_count: int
    accessibility_eligible: bool
    multilingual_support_eligible: bool
    cultural_value_eligible: bool
    safety_certified: bool
    prohibited_mechanics_enabled: bool

    def __post_init__(self) -> None:
        for name in (
            "fixed_price_revenue_cents",
            "institutional_licensing_revenue_cents",
            "non_targeted_sponsorship_revenue_cents",
            "development_cost_cents",
            "maintenance_cost_cents",
            "eligible_access_count",
            "eligible_institutional_license_count",
            "availability_period_count",
        ):
            _validate_nonnegative_int(getattr(self, name), name=name)
        for name in (
            "accessibility_eligible",
            "multilingual_support_eligible",
            "cultural_value_eligible",
            "safety_certified",
            "prohibited_mechanics_enabled",
        ):
            _validate_bool(getattr(self, name), name=name)


@dataclass(frozen=True, slots=True)
class EPGCResult:
    """Auditable payment decomposition and safe-profit calculation."""

    access_payment_cents: int
    institutional_license_payment_cents: int
    availability_payment_cents: int
    accessibility_bonus_cents: int
    multilingual_bonus_cents: int
    cultural_value_bonus_cents: int
    safety_certification_bonus_cents: int
    gross_eligible_public_contract_cents: int
    budget_limited_public_contract_cents: int
    clawback_cents: int
    penalty_cents: int
    public_contract_revenue_cents: int
    fixed_price_revenue_cents: int
    institutional_licensing_revenue_cents: int
    non_targeted_sponsorship_revenue_cents: int
    development_cost_cents: int
    maintenance_cost_cents: int
    maximum_budget_cents: int
    minimum_public_contribution_cents: int
    feasible_under_budget_cap: bool
    profit_safe_cents: int
    sustainable_under_policy: bool

    def __post_init__(self) -> None:
        money_fields = (
            "access_payment_cents",
            "institutional_license_payment_cents",
            "availability_payment_cents",
            "accessibility_bonus_cents",
            "multilingual_bonus_cents",
            "cultural_value_bonus_cents",
            "safety_certification_bonus_cents",
            "gross_eligible_public_contract_cents",
            "budget_limited_public_contract_cents",
            "clawback_cents",
            "penalty_cents",
            "public_contract_revenue_cents",
            "fixed_price_revenue_cents",
            "institutional_licensing_revenue_cents",
            "non_targeted_sponsorship_revenue_cents",
            "development_cost_cents",
            "maintenance_cost_cents",
            "maximum_budget_cents",
            "minimum_public_contribution_cents",
        )
        for name in money_fields:
            _validate_nonnegative_int(getattr(self, name), name=name)
        _validate_bool(
            self.feasible_under_budget_cap,
            name="feasible_under_budget_cap",
        )
        _validate_bool(
            self.sustainable_under_policy,
            name="sustainable_under_policy",
        )
        if isinstance(self.profit_safe_cents, bool) or not isinstance(
            self.profit_safe_cents, int
        ):
            raise TypeError("profit_safe_cents must be an integer")
        _checked_signed(self.profit_safe_cents, label="profit_safe_cents")

        gross_components = _checked_sum(
            self.access_payment_cents,
            self.institutional_license_payment_cents,
            self.availability_payment_cents,
            self.accessibility_bonus_cents,
            self.multilingual_bonus_cents,
            self.cultural_value_bonus_cents,
            self.safety_certification_bonus_cents,
            label="gross eligible public contract",
        )
        if gross_components != self.gross_eligible_public_contract_cents:
            raise ValueError("gross public contract does not match its components")
        if self.budget_limited_public_contract_cents != min(
            self.gross_eligible_public_contract_cents,
            self.maximum_budget_cents,
        ):
            raise ValueError("budget-limited contract does not apply the budget cap")
        if self.clawback_cents + self.penalty_cents > (
            self.budget_limited_public_contract_cents
        ):
            raise ValueError("sanctions exceed the budget-limited contract")
        expected_public_revenue = (
            self.budget_limited_public_contract_cents
            - self.clawback_cents
            - self.penalty_cents
        )
        if self.public_contract_revenue_cents != expected_public_revenue:
            raise ValueError("public contract revenue does not reconcile sanctions")

        private_revenue = _checked_sum(
            self.fixed_price_revenue_cents,
            self.institutional_licensing_revenue_cents,
            self.non_targeted_sponsorship_revenue_cents,
            label="conventional revenue",
        )
        costs = _checked_sum(
            self.development_cost_cents,
            self.maintenance_cost_cents,
            label="safe production cost",
        )
        expected_minimum = max(0, costs - private_revenue)
        if self.minimum_public_contribution_cents != expected_minimum:
            raise ValueError("minimum public contribution is inconsistent")
        if self.feasible_under_budget_cap != (
            expected_minimum <= self.maximum_budget_cents
        ):
            raise ValueError("budget-cap feasibility is inconsistent")

        safe_revenue = _checked_sum(
            self.public_contract_revenue_cents,
            private_revenue,
            label="total safe revenue",
        )
        expected_profit = _checked_signed(
            safe_revenue - costs,
            label="safe profit",
        )
        if self.profit_safe_cents != expected_profit:
            raise ValueError("profit_safe_cents does not satisfy the EPGC equation")
        if self.sustainable_under_policy != (expected_profit >= 0):
            raise ValueError("sustainability flag is inconsistent with safe profit")


def evaluate_epgc(policy: EPGCPolicy, inputs: EPGCFirmInputs) -> EPGCResult:
    """Evaluate one policy/firm pair without mutating either input.

    The minimum public contribution is the exact residual after conventional
    safe revenue is subtracted from development and maintenance costs.  Cap
    feasibility reports whether that residual fits within the public budget;
    actual sustainability additionally reflects eligibility and any sanctions.
    """

    if not isinstance(policy, EPGCPolicy):
        raise TypeError("policy must be an EPGCPolicy")
    if not isinstance(inputs, EPGCFirmInputs):
        raise TypeError("inputs must be EPGCFirmInputs")

    access_payment = _checked_product(
        policy.access_payment_cents_per_eligible_access,
        inputs.eligible_access_count,
        label="eligible-access payment",
    )
    institutional_payment = _checked_product(
        policy.institutional_license_payment_cents_per_license,
        inputs.eligible_institutional_license_count,
        label="institutional-license payment",
    )
    availability_payment = _checked_product(
        policy.availability_payment_cents_per_period,
        inputs.availability_period_count,
        label="availability payment",
    )
    accessibility_bonus = (
        policy.accessibility_bonus_cents if inputs.accessibility_eligible else 0
    )
    multilingual_bonus = (
        policy.multilingual_bonus_cents
        if inputs.multilingual_support_eligible
        else 0
    )
    cultural_bonus = (
        policy.cultural_value_bonus_cents if inputs.cultural_value_eligible else 0
    )
    safety_bonus = (
        policy.safety_certification_bonus_cents if inputs.safety_certified else 0
    )
    gross_eligible = _checked_sum(
        access_payment,
        institutional_payment,
        availability_payment,
        accessibility_bonus,
        multilingual_bonus,
        cultural_bonus,
        safety_bonus,
        label="gross eligible public contract",
    )
    budget_limited = min(gross_eligible, policy.maximum_budget_cents)

    if inputs.prohibited_mechanics_enabled:
        clawback = (
            budget_limited
            * policy.prohibited_mechanics_clawback_basis_points
            // _BASIS_POINTS
        )
        penalty = min(
            policy.prohibited_mechanics_penalty_cents,
            budget_limited - clawback,
        )
    else:
        clawback = 0
        penalty = 0
    public_contract_revenue = budget_limited - clawback - penalty

    private_revenue = _checked_sum(
        inputs.fixed_price_revenue_cents,
        inputs.institutional_licensing_revenue_cents,
        inputs.non_targeted_sponsorship_revenue_cents,
        label="conventional revenue",
    )
    costs = _checked_sum(
        inputs.development_cost_cents,
        inputs.maintenance_cost_cents,
        label="safe production cost",
    )
    minimum_public_contribution = max(0, costs - private_revenue)
    safe_revenue = _checked_sum(
        public_contract_revenue,
        private_revenue,
        label="total safe revenue",
    )
    profit_safe = _checked_signed(safe_revenue - costs, label="safe profit")

    return EPGCResult(
        access_payment_cents=access_payment,
        institutional_license_payment_cents=institutional_payment,
        availability_payment_cents=availability_payment,
        accessibility_bonus_cents=accessibility_bonus,
        multilingual_bonus_cents=multilingual_bonus,
        cultural_value_bonus_cents=cultural_bonus,
        safety_certification_bonus_cents=safety_bonus,
        gross_eligible_public_contract_cents=gross_eligible,
        budget_limited_public_contract_cents=budget_limited,
        clawback_cents=clawback,
        penalty_cents=penalty,
        public_contract_revenue_cents=public_contract_revenue,
        fixed_price_revenue_cents=inputs.fixed_price_revenue_cents,
        institutional_licensing_revenue_cents=(
            inputs.institutional_licensing_revenue_cents
        ),
        non_targeted_sponsorship_revenue_cents=(
            inputs.non_targeted_sponsorship_revenue_cents
        ),
        development_cost_cents=inputs.development_cost_cents,
        maintenance_cost_cents=inputs.maintenance_cost_cents,
        maximum_budget_cents=policy.maximum_budget_cents,
        minimum_public_contribution_cents=minimum_public_contribution,
        feasible_under_budget_cap=(
            minimum_public_contribution <= policy.maximum_budget_cents
        ),
        profit_safe_cents=profit_safe,
        sustainable_under_policy=profit_safe >= 0,
    )


__all__ = [
    "EPGCFirmInputs",
    "EPGCPolicy",
    "EPGCResult",
    "evaluate_epgc",
]
