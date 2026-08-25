"""Shared metric stems used by aggregation, schemas, and provenance."""

from __future__ import annotations

from typing import Final


REPEATED_SEED_METRIC_STEMS: Final[tuple[str, ...]] = (
    "total_revenue_cents",
    "producer_profit_cents",
    "total_spending_cents",
    "harmful_spending_cents",
    "mean_harm",
    "mean_harm_effect_vs_safe",
    "mean_opportunity_cost_score",
    "mean_sleep_burden",
    "mean_education_work_burden",
    "mean_social_burden",
    "mean_wellbeing_burden",
    "mean_enjoyment",
    "high_risk_count",
    "total_opportunity_cost_proxy_cents",
    "epgc_minimum_public_contribution_cents",
    "revenue_direct_purchase_cents",
    "revenue_fixed_price_cents",
    "revenue_institutional_licensing_cents",
    "revenue_non_targeted_sponsorship_cents",
    "revenue_opaque_virtual_currency_cents",
    "revenue_paid_random_rewards_cents",
    "revenue_public_contract_cents",
    "revenue_subscription_cents",
)


__all__ = ["REPEATED_SEED_METRIC_STEMS"]
