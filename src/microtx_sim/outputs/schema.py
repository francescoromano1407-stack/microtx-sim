"""Stable, machine-readable column contracts for exported simulation results.

Version 2 exhaustively declares every column in a policy table and rejects
undeclared row keys.  The named v1 prefix tuples preserve the migration
boundary: v2 keeps the released non-empty header order while empty tables now
expose the complete schema.  The generic CSV writer remains flexible.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from ..metrics.reporting import REPEATED_SEED_METRIC_STEMS


OUTPUT_SCHEMA_VERSION: Final[str] = "2.0"

SEED_RESULT_V1_PREFIX_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "scenario_label",
    "seed",
    "cohort_digest",
    "days",
    "player_count",
    "total_revenue_cents",
    "producer_cost_cents",
    "producer_profit_cents",
    "revenue_direct_purchase_cents",
    "revenue_opaque_virtual_currency_cents",
    "revenue_paid_random_rewards_cents",
    "revenue_fixed_price_cents",
    "revenue_subscription_cents",
    "revenue_public_contract_cents",
    "revenue_institutional_licensing_cents",
    "revenue_non_targeted_sponsorship_cents",
    "total_spending_cents",
    "harmful_spending_cents",
    "unplanned_spending_cents",
    "mean_harm",
    "harm_p50",
    "harm_p90",
    "total_opportunity_cost_proxy_cents",
    "mean_opportunity_cost_score",
    "mean_sleep_burden",
    "mean_education_work_burden",
    "mean_social_burden",
    "mean_wellbeing_burden",
    "mean_enjoyment",
    "high_risk_count",
    "high_risk_share",
    "mean_harm_effect_vs_safe",
    "epgc_minimum_public_contribution_cents",
)

_SEED_RESULT_V2_EXTENSION_COLUMNS: Final[tuple[str, ...]] = (
    "adult_opportunity_cost_proxy_cents",
    "epgc_profit_safe_cents",
    "harm_p10",
    "harm_variance_players",
    "harmful_spending_effect_vs_safe_cents",
    "high_risk_mean_age",
    "high_risk_mean_baseline_vulnerability",
    "high_risk_mean_budget_cents",
    "high_risk_minor_share",
    "mean_monetary_harm",
    "spend_p10_cents",
    "spend_p50_cents",
    "spend_p90_cents",
    "total_revenue_effect_vs_safe_cents",
    "total_spending_effect_vs_safe_cents",
    "youth_opportunity_cost_proxy_cents",
)

SEED_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    SEED_RESULT_V1_PREFIX_COLUMNS + _SEED_RESULT_V2_EXTENSION_COLUMNS
)

SCENARIO_SUMMARY_V1_PREFIX_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "scenario_label",
    "seed_count",
    "player_count",
    "days",
    "total_revenue_cents_mean",
    "total_revenue_cents_sd",
    "total_spending_cents_mean",
    "harmful_spending_cents_mean",
    "mean_harm_mean",
    "mean_harm_sd",
    "mean_harm_ci95_low",
    "mean_harm_ci95_high",
    "mean_harm_effect_vs_safe_mean",
    "total_opportunity_cost_proxy_cents_mean",
    "mean_opportunity_cost_score_mean",
    "mean_sleep_burden_mean",
    "mean_education_work_burden_mean",
    "mean_social_burden_mean",
    "mean_wellbeing_burden_mean",
    "mean_enjoyment_mean",
    "high_risk_count_mean",
    "epgc_minimum_public_contribution_cents_mean",
)

_UNCERTAINTY_SUFFIXES: Final[tuple[str, ...]] = (
    "mean",
    "variance",
    "sd",
    "ci95_low",
    "ci95_high",
)

_SCENARIO_SUMMARY_V1_DERIVED: Final[frozenset[str]] = frozenset(
    f"{metric}_{suffix}"
    for metric in REPEATED_SEED_METRIC_STEMS
    for suffix in _UNCERTAINTY_SUFFIXES
)

SCENARIO_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    SCENARIO_SUMMARY_V1_PREFIX_COLUMNS
    + tuple(
        sorted(
            _SCENARIO_SUMMARY_V1_DERIVED.difference(
                SCENARIO_SUMMARY_V1_PREFIX_COLUMNS
            )
        )
    )
)

EPGC_FINANCING_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "seed",
    "public_contract_revenue_cents",
    "minimum_public_contribution_cents",
    "maximum_budget_cents",
    "profit_safe_cents",
    "feasible_under_budget_cap",
    "sustainable_under_policy",
    "clawback_cents",
    "penalty_cents",
)

SENSITIVITY_V1_PREFIX_COLUMNS: Final[tuple[str, ...]] = (
    "parameter",
    "parameter_value",
    "scenario_id",
    "seed_count",
    "mean_harm",
    "harm_variance",
    "harm_sd",
    "harm_ci95_low",
    "harm_ci95_high",
    "total_revenue_cents",
    "opportunity_cost_burden",
    "minimum_public_contribution_cents",
    "expected_direction",
    "monotonic_observed",
    "unstable",
)

_SENSITIVITY_V2_EXTENSION_COLUMNS: Final[tuple[str, ...]] = (
    "harm_coefficient_of_variation",
    "monotonic_expected",
)

SENSITIVITY_COLUMNS: Final[tuple[str, ...]] = (
    SENSITIVITY_V1_PREFIX_COLUMNS + _SENSITIVITY_V2_EXTENSION_COLUMNS
)

PLAYER_OUTCOME_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "seed",
    "player_id",
    "age_years",
    "is_minor",
    "baseline_vulnerability",
    "spending_cents",
    "harmful_spending_cents",
    "composite_harm",
    "monetary_harm",
    "opportunity_cost",
    "sleep_burden",
    "education_work_burden",
    "social_burden",
    "wellbeing_burden",
    "opportunity_cost_proxy_cents",
    "enjoyment",
    "high_risk",
)

OPPORTUNITY_DECOMPOSITION_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "component",
    "mean_minutes",
    "mean_burden",
    "monetary_proxy_cents",
)

TABLE_COLUMNS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "seed_results.csv": SEED_RESULT_COLUMNS,
        "scenario_summary.csv": SCENARIO_SUMMARY_COLUMNS,
        "epgc_financing.csv": EPGC_FINANCING_COLUMNS,
        "sensitivity.csv": SENSITIVITY_COLUMNS,
        "player_outcomes.csv": PLAYER_OUTCOME_COLUMNS,
        "opportunity_cost_decomposition.csv": (
            OPPORTUNITY_DECOMPOSITION_COLUMNS
        ),
    }
)

ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "seed_results.csv",
    "scenario_summary.csv",
    "epgc_financing.csv",
    "sensitivity.csv",
    "manifest.json",
)

POLICY_ARTIFACT_FILENAMES: Final[tuple[str, ...]] = (
    "seed_results.csv",
    "scenario_summary.csv",
    "player_outcomes.csv",
    "opportunity_cost_decomposition.csv",
    "epgc_financing.csv",
    "sensitivity.csv",
    "manifest.json",
    "summary.md",
    "harm_distribution.svg",
    "spending_distribution.svg",
    "harm_revenue_frontier.svg",
    "opportunity_cost_decomposition.svg",
    "epgc_subsidy_requirement.svg",
)


__all__ = [
    "ARTIFACT_FILENAMES",
    "EPGC_FINANCING_COLUMNS",
    "OUTPUT_SCHEMA_VERSION",
    "OPPORTUNITY_DECOMPOSITION_COLUMNS",
    "PLAYER_OUTCOME_COLUMNS",
    "POLICY_ARTIFACT_FILENAMES",
    "REPEATED_SEED_METRIC_STEMS",
    "SCENARIO_SUMMARY_COLUMNS",
    "SCENARIO_SUMMARY_V1_PREFIX_COLUMNS",
    "SEED_RESULT_COLUMNS",
    "SEED_RESULT_V1_PREFIX_COLUMNS",
    "SENSITIVITY_COLUMNS",
    "SENSITIVITY_V1_PREFIX_COLUMNS",
    "TABLE_COLUMNS",
]
