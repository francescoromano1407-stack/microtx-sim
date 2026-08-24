"""Strict TOML configuration for the synthetic policy-prototype runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import tomllib

from .causal.batch import PolicyBatchSpec
from .consumers.decision import DecisionParameters
from .funding import EPGCPolicy
from .metrics.harm import (
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmWeights,
)
from .simulation.policy_orchestrator import ProducerAssumptions


class PolicyConfigurationError(ValueError):
    """Raised when the policy-prototype configuration is ambiguous or unsafe."""


@dataclass(frozen=True, slots=True)
class PolicyOutputConfig:
    output_dir: Path
    histogram_bins: int = 20
    include_player_rows: bool = True
    run_sensitivity: bool = True

    def __post_init__(self) -> None:
        if not str(self.output_dir):
            raise ValueError("output_dir cannot be empty")
        if isinstance(self.histogram_bins, bool) or not isinstance(
            self.histogram_bins, int
        ):
            raise TypeError("histogram_bins must be an integer")
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")
        if not isinstance(self.include_player_rows, bool):
            raise TypeError("include_player_rows must be boolean")
        if not isinstance(self.run_sensitivity, bool):
            raise TypeError("run_sensitivity must be boolean")


@dataclass(frozen=True, slots=True)
class PolicyPrototypeConfig:
    name: str
    provenance_status: str
    notes: str
    batch: PolicyBatchSpec
    harm_parameters: HarmModelParameters
    harm_weights: WelfareHarmWeights
    opportunity_valuation: OpportunityCostValuation
    producer_assumptions: ProducerAssumptions
    epgc_policy: EPGCPolicy
    output: PolicyOutputConfig

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("configuration name cannot be empty")
        if self.provenance_status != "synthetic":
            raise ValueError(
                "the policy prototype accepts synthetic provenance only"
            )


def load_policy_config(path: str | Path) -> PolicyPrototypeConfig:
    """Load a strict, fully typed policy prototype configuration."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        _strict_top_level(raw)
        meta = _section(raw, "meta")
        run = _section(raw, "policy_run")
        decision = _section(raw, "decision")
        harm = _section(raw, "harm")
        weights = _section(raw, "harm_weights")
        valuation = _section(raw, "opportunity_valuation")
        producer = _section(raw, "producer")
        epgc = _section(raw, "epgc")
        output = _section(raw, "output")
        _exact_keys(meta, {"name", "provenance_status", "notes"}, "meta")
        _exact_keys(run, {"seeds", "days", "player_count"}, "policy_run")
        _exact_keys(
            decision,
            {
                "step_minutes",
                "temperature",
                "habit_persistence",
                "habit_learning_rate",
                "reinforcement_learning_rate",
            },
            "decision",
        )
        _exact_keys(
            harm,
            {
                "affordable_spending_share",
                "opaque_spending_weight",
                "random_reward_spending_weight",
                "time_pressure_spending_weight",
                "sleep_debt_weight",
            },
            "harm",
        )
        _exact_keys(
            weights,
            {
                "monetary",
                "opportunity_cost",
                "sleep",
                "education_work",
                "family_social",
                "wellbeing",
            },
            "harm_weights",
        )
        _exact_keys(
            valuation,
            {
                "adult_sleep_hour_cents",
                "adult_work_study_hour_cents",
                "adult_social_hour_cents",
                "adult_physical_activity_hour_cents",
                "youth_sleep_hour_cents",
                "youth_education_hour_cents",
                "youth_family_social_hour_cents",
                "youth_physical_activity_hour_cents",
            },
            "opportunity_valuation",
        )
        _exact_keys(
            producer,
            {
                "development_cost_cents",
                "maintenance_cost_cents_per_day",
                "institutional_license_count",
                "institutional_license_price_cents",
                "non_targeted_sponsorship_revenue_cents",
                "accessibility_eligible",
                "multilingual_support_eligible",
                "cultural_value_eligible",
                "safety_certified",
            },
            "producer",
        )
        _exact_keys(
            epgc,
            {
                "access_payment_cents_per_eligible_access",
                "institutional_license_payment_cents_per_license",
                "availability_payment_cents_per_period",
                "accessibility_bonus_cents",
                "multilingual_bonus_cents",
                "cultural_value_bonus_cents",
                "safety_certification_bonus_cents",
                "prohibited_mechanics_penalty_cents",
                "prohibited_mechanics_clawback_basis_points",
                "maximum_budget_cents",
            },
            "epgc",
        )
        _exact_keys(
            output,
            {
                "output_dir",
                "histogram_bins",
                "include_player_rows",
                "run_sensitivity",
            },
            "output",
        )
        decision_parameters = DecisionParameters(**decision)
        batch = PolicyBatchSpec(
            seeds=tuple(run["seeds"]),
            days=run["days"],
            player_count=run["player_count"],
            decision_parameters=decision_parameters,
        )
        return PolicyPrototypeConfig(
            name=str(meta["name"]),
            provenance_status=str(meta["provenance_status"]),
            notes=str(meta["notes"]),
            batch=batch,
            harm_parameters=HarmModelParameters(**harm),
            harm_weights=WelfareHarmWeights(**weights),
            opportunity_valuation=OpportunityCostValuation(**valuation),
            producer_assumptions=ProducerAssumptions(**producer),
            epgc_policy=EPGCPolicy(**epgc),
            output=PolicyOutputConfig(
                output_dir=Path(output["output_dir"]),
                histogram_bins=output["histogram_bins"],
                include_player_rows=output["include_player_rows"],
                run_sensitivity=output["run_sensitivity"],
            ),
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        if isinstance(exc, PolicyConfigurationError):
            raise
        raise PolicyConfigurationError(
            f"invalid policy configuration {config_path}: {exc}"
        ) from exc


def _strict_top_level(raw: Mapping[str, object]) -> None:
    expected = {
        "meta",
        "policy_run",
        "decision",
        "harm",
        "harm_weights",
        "opportunity_valuation",
        "producer",
        "epgc",
        "output",
    }
    _exact_keys(raw, expected, "top level")


def _section(raw: Mapping[str, object], name: str) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise PolicyConfigurationError(f"[{name}] must be a TOML table")
    return value


def _exact_keys(
    values: Mapping[str, object], expected: set[str], name: str
) -> None:
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise PolicyConfigurationError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


__all__ = [
    "PolicyConfigurationError",
    "PolicyOutputConfig",
    "PolicyPrototypeConfig",
    "load_policy_config",
]
