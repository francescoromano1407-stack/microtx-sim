"""Repeated-seed, common-cohort counterfactual policy batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from math import sqrt
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..consumers.decision import DecisionParameters
from ..consumers.population import CountryProfile, initialize_player_table
from ..consumers.welfare import initialize_player_life
from ..data.profiles import load_profile_bundle
from ..funding import EPGCPolicy
from ..metrics.harm import (
    HarmComponent,
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmWeights,
)
from ..rng import CounterRNG
from ..simulation.policy_orchestrator import (
    PolicyScenarioResult,
    ProducerAssumptions,
    run_policy_scenario,
)
from .scenarios import ScenarioId, ScenarioSpec, required_scenarios


@dataclass(frozen=True, slots=True)
class PolicyBatchSpec:
    """A small reproducible scenario-by-seed design."""

    seeds: tuple[int, ...] = (101, 202, 303)
    days: int = 14
    player_count: int = 500
    scenarios: tuple[ScenarioSpec, ...] = field(default_factory=required_scenarios)
    reference_scenario: ScenarioId = ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION
    decision_parameters: DecisionParameters = field(default_factory=DecisionParameters)

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        for seed in self.seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seeds must contain integers")
        for name in ("days", "player_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        if len(set(ids)) != len(ids):
            raise ValueError("scenario ids must be unique")
        if set(ids) != set(ScenarioId):
            missing = sorted(item.value for item in set(ScenarioId) - set(ids))
            extra = sorted(str(item) for item in set(ids) - set(ScenarioId))
            raise ValueError(
                f"batch must contain exactly the seven required scenarios; "
                f"missing={missing}, extra={extra}"
            )
        if self.reference_scenario not in ids:
            raise ValueError("reference scenario is not present in the batch")


@dataclass(frozen=True, slots=True)
class SeedScenarioRecord:
    """One scenario run plus its paired effect against the safe reference."""

    result: PolicyScenarioResult
    cohort_digest: str
    mean_harm_effect_vs_safe: float
    total_spending_effect_vs_safe_cents: int
    harmful_spending_effect_vs_safe_cents: int
    total_revenue_effect_vs_safe_cents: int


@dataclass(frozen=True, slots=True)
class PolicyBatchResult:
    """All individual outputs and tidy summary projections for one batch."""

    spec: PolicyBatchSpec
    records: tuple[SeedScenarioRecord, ...]
    cohort_digest_by_seed: Mapping[int, str]

    def __post_init__(self) -> None:
        expected = len(self.spec.seeds) * len(self.spec.scenarios)
        if len(self.records) != expected:
            raise ValueError(f"expected {expected} seed-scenario records")
        keys = {
            (record.result.seed, record.result.scenario.scenario_id)
            for record in self.records
        }
        if len(keys) != expected:
            raise ValueError("duplicate or missing seed-scenario result")
        digests = {int(seed): str(value) for seed, value in self.cohort_digest_by_seed.items()}
        if set(digests) != set(self.spec.seeds):
            raise ValueError("cohort digest metadata do not match seeds")
        for record in self.records:
            if digests[record.result.seed] != record.cohort_digest:
                raise ValueError("record cohort digest is inconsistent")
        object.__setattr__(self, "cohort_digest_by_seed", MappingProxyType(digests))

    def seed_rows(self) -> list[dict[str, object]]:
        """Return one machine-readable aggregate row per seed and scenario."""

        return [_seed_row(record) for record in self.records]

    def scenario_rows(self) -> list[dict[str, object]]:
        """Aggregate repeated seeds with variance and normal 95% intervals."""

        seed_rows = self.seed_rows()
        output: list[dict[str, object]] = []
        for scenario in self.spec.scenarios:
            rows = [
                row
                for row in seed_rows
                if row["scenario_id"] == scenario.scenario_id.value
            ]
            result: dict[str, object] = {
                "scenario_id": scenario.scenario_id.value,
                "scenario_label": scenario.label,
                "seed_count": len(rows),
                "player_count": self.spec.player_count,
                "days": self.spec.days,
            }
            metrics = (
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
            )
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in rows])
                mean, variance, standard_deviation, low, high = _uncertainty(values)
                result[f"{metric}_mean"] = mean
                result[f"{metric}_variance"] = variance
                result[f"{metric}_sd"] = standard_deviation
                result[f"{metric}_ci95_low"] = low
                result[f"{metric}_ci95_high"] = high
            for key in sorted(rows[0]):
                if key.startswith("revenue_") and key.endswith("_cents"):
                    values = np.asarray([float(row[key]) for row in rows])
                    mean, variance, standard_deviation, low, high = _uncertainty(
                        values
                    )
                    result[f"{key}_mean"] = mean
                    result[f"{key}_variance"] = variance
                    result[f"{key}_sd"] = standard_deviation
                    result[f"{key}_ci95_low"] = low
                    result[f"{key}_ci95_high"] = high
            output.append(result)
        return output

    def player_rows(self) -> list[dict[str, object]]:
        """Return player-level synthetic distributions for plots and auditing."""

        rows: list[dict[str, object]] = []
        for record in self.records:
            result = record.result
            for index, player_id in enumerate(result.player_ids):
                rows.append(
                    {
                        "scenario_id": result.scenario.scenario_id.value,
                        "seed": result.seed,
                        "player_id": int(player_id),
                        "age_years": int(result.age_years[index]),
                        "is_minor": bool(result.is_minor[index]),
                        "baseline_vulnerability": float(
                            result.baseline_vulnerability[index]
                        ),
                        "spending_cents": int(result.spending_cents[index]),
                        "harmful_spending_cents": int(
                            result.harm.harmful_spending_cents[index]
                        ),
                        "composite_harm": float(result.composite_harm[index]),
                        "monetary_harm": float(
                            result.harm.component_scores[index, HarmComponent.M]
                        ),
                        "opportunity_cost": float(
                            result.harm.component_scores[index, HarmComponent.OC]
                        ),
                        "sleep_burden": float(
                            result.harm.component_scores[index, HarmComponent.S]
                        ),
                        "education_work_burden": float(
                            result.harm.component_scores[index, HarmComponent.E]
                        ),
                        "social_burden": float(
                            result.harm.component_scores[index, HarmComponent.F]
                        ),
                        "wellbeing_burden": float(
                            result.harm.component_scores[index, HarmComponent.W]
                        ),
                        "opportunity_cost_proxy_cents": int(
                            result.harm.opportunity_cost_proxy_cents[index]
                        ),
                        "enjoyment": float(result.enjoyment[index]),
                        "high_risk": bool(result.high_risk[index]),
                    }
                )
        return rows

    def epgc_rows(self) -> list[dict[str, object]]:
        """Return the EPGC audit trail for each seed."""

        rows: list[dict[str, object]] = []
        for record in self.records:
            result = record.result
            if result.epgc is None:
                continue
            epgc = result.epgc
            rows.append(
                {
                    "scenario_id": result.scenario.scenario_id.value,
                    "seed": result.seed,
                    "public_contract_revenue_cents": epgc.public_contract_revenue_cents,
                    "minimum_public_contribution_cents": (
                        epgc.minimum_public_contribution_cents
                    ),
                    "maximum_budget_cents": epgc.maximum_budget_cents,
                    "profit_safe_cents": epgc.profit_safe_cents,
                    "feasible_under_budget_cap": epgc.feasible_under_budget_cap,
                    "sustainable_under_policy": epgc.sustainable_under_policy,
                    "clawback_cents": epgc.clawback_cents,
                    "penalty_cents": epgc.penalty_cents,
                }
            )
        return rows

    def opportunity_rows(self) -> list[dict[str, object]]:
        """Return scenario-level displaced-activity decomposition rows."""

        output: list[dict[str, object]] = []
        components = (
            ("sleep", "displaced_sleep_minutes", HarmComponent.S),
            ("work_study", "displaced_work_study_minutes", HarmComponent.E),
            ("family_social", "displaced_social_minutes", HarmComponent.F),
            ("physical_activity", "displaced_physical_activity_minutes", None),
        )
        for scenario in self.spec.scenarios:
            records = [
                item
                for item in self.records
                if item.result.scenario.scenario_id is scenario.scenario_id
            ]
            for label, minute_name, burden_component in components:
                minute_arrays = [
                    getattr(item.result.harm, minute_name) for item in records
                ]
                minute_values = (
                    np.concatenate(minute_arrays)
                    if minute_arrays
                    else np.empty(0, dtype=np.float64)
                )
                if burden_component is None:
                    burden_values = np.empty(0, dtype=np.float64)
                else:
                    burden_values = np.concatenate(
                        [
                            item.result.harm.component_scores[:, burden_component]
                            for item in records
                        ]
                    )
                output.append(
                    {
                        "scenario_id": scenario.scenario_id.value,
                        "component": label,
                        "mean_minutes": _mean(minute_values),
                        "mean_burden": _mean(burden_values),
                        "monetary_proxy_cents": "",
                    }
                )
            output.append(
                {
                    "scenario_id": scenario.scenario_id.value,
                    "component": "all_displaced_activities",
                    "mean_minutes": sum(
                        float(row["mean_minutes"])
                        for row in output[-len(components) :]
                    ),
                    "mean_burden": float(
                        np.mean(
                            [
                                _mean(
                                    item.result.harm.component_scores[
                                        :, HarmComponent.OC
                                    ]
                                )
                                for item in records
                            ]
                        )
                    )
                    if records
                    else 0.0,
                    "monetary_proxy_cents": float(
                        np.mean(
                            [
                                _python_sum(
                                    item.result.harm.opportunity_cost_proxy_cents
                                )
                                for item in records
                            ]
                        )
                    )
                    if records
                    else 0.0,
                }
            )
        return output


def run_policy_batch(
    spec: PolicyBatchSpec,
    *,
    country_profiles: Sequence[CountryProfile] | None = None,
    harm_parameters: HarmModelParameters | None = None,
    harm_weights: WelfareHarmWeights | None = None,
    opportunity_valuation: OpportunityCostValuation | None = None,
    producer_assumptions: ProducerAssumptions | None = None,
    epgc_policy: EPGCPolicy | None = None,
) -> PolicyBatchResult:
    """Run all scenarios on the same seeded cohort within each replication."""

    if not isinstance(spec, PolicyBatchSpec):
        raise TypeError("spec must be PolicyBatchSpec")
    profiles = tuple(country_profiles) if country_profiles is not None else (
        load_profile_bundle(campaign=False).country_profiles
    )
    if not profiles:
        raise ValueError("at least one country profile is required")
    records: list[SeedScenarioRecord] = []
    digests: dict[int, str] = {}
    for seed in spec.seeds:
        rng = CounterRNG(seed)
        players = initialize_player_table(spec.player_count, profiles, rng)
        life = initialize_player_life(players, rng)
        digest = _cohort_digest(players, life)
        digests[seed] = digest
        scenario_results: dict[ScenarioId, PolicyScenarioResult] = {}
        for scenario in spec.scenarios:
            scenario_results[scenario.scenario_id] = run_policy_scenario(
                players,
                life,
                scenario,
                seed=seed,
                days=spec.days,
                decision_parameters=spec.decision_parameters,
                harm_parameters=harm_parameters,
                harm_weights=harm_weights,
                opportunity_valuation=opportunity_valuation,
                producer_assumptions=producer_assumptions,
                epgc_policy=epgc_policy,
            )
        reference = scenario_results[spec.reference_scenario]
        for scenario in spec.scenarios:
            result = scenario_results[scenario.scenario_id]
            if not np.array_equal(result.player_ids, reference.player_ids):
                raise RuntimeError("counterfactual branches lost player alignment")
            harm_difference = result.composite_harm - reference.composite_harm
            records.append(
                SeedScenarioRecord(
                    result=result,
                    cohort_digest=digest,
                    mean_harm_effect_vs_safe=(
                        float(harm_difference.mean()) if len(harm_difference) else 0.0
                    ),
                    total_spending_effect_vs_safe_cents=(
                        _python_sum(result.spending_cents)
                        - _python_sum(reference.spending_cents)
                    ),
                    harmful_spending_effect_vs_safe_cents=(
                        _python_sum(result.harm.harmful_spending_cents)
                        - _python_sum(reference.harm.harmful_spending_cents)
                    ),
                    total_revenue_effect_vs_safe_cents=(
                        result.total_revenue_cents - reference.total_revenue_cents
                    ),
                )
            )
    return PolicyBatchResult(
        spec=spec,
        records=tuple(records),
        cohort_digest_by_seed=digests,
    )


def _seed_row(record: SeedScenarioRecord) -> dict[str, object]:
    result = record.result
    harm = result.harm.component_scores
    high = result.high_risk
    row: dict[str, object] = {
        "scenario_id": result.scenario.scenario_id.value,
        "scenario_label": result.scenario.label,
        "seed": result.seed,
        "cohort_digest": record.cohort_digest,
        "days": result.days,
        "player_count": len(result.player_ids),
        "total_revenue_cents": result.total_revenue_cents,
        "producer_cost_cents": result.producer_cost_cents,
        "producer_profit_cents": result.producer_profit_cents,
        "total_spending_cents": _python_sum(result.spending_cents),
        "harmful_spending_cents": _python_sum(result.harm.harmful_spending_cents),
        "unplanned_spending_cents": _python_sum(result.harm.unplanned_spending_cents),
        "mean_harm": _mean(result.composite_harm),
        "harm_variance_players": _variance(result.composite_harm),
        "harm_p10": _quantile(result.composite_harm, 0.10),
        "harm_p50": _quantile(result.composite_harm, 0.50),
        "harm_p90": _quantile(result.composite_harm, 0.90),
        "spend_p10_cents": _quantile(result.spending_cents, 0.10),
        "spend_p50_cents": _quantile(result.spending_cents, 0.50),
        "spend_p90_cents": _quantile(result.spending_cents, 0.90),
        "mean_monetary_harm": _mean(harm[:, HarmComponent.M]),
        "mean_opportunity_cost_score": _mean(harm[:, HarmComponent.OC]),
        "mean_sleep_burden": _mean(harm[:, HarmComponent.S]),
        "mean_education_work_burden": _mean(harm[:, HarmComponent.E]),
        "mean_social_burden": _mean(harm[:, HarmComponent.F]),
        "mean_wellbeing_burden": _mean(harm[:, HarmComponent.W]),
        "total_opportunity_cost_proxy_cents": _python_sum(
            result.harm.opportunity_cost_proxy_cents
        ),
        "adult_opportunity_cost_proxy_cents": _python_sum(
            result.harm.adult_opportunity_cost_proxy_cents
        ),
        "youth_opportunity_cost_proxy_cents": _python_sum(
            result.harm.youth_opportunity_cost_proxy_cents
        ),
        "mean_enjoyment": _mean(result.enjoyment),
        "high_risk_count": int(np.count_nonzero(high)),
        "high_risk_share": float(np.mean(high)) if len(high) else 0.0,
        "high_risk_mean_age": _masked_mean(result.age_years, high),
        "high_risk_minor_share": _masked_mean(result.is_minor, high),
        "high_risk_mean_budget_cents": _masked_mean(
            result.disposable_budget_cents, high
        ),
        "high_risk_mean_baseline_vulnerability": _masked_mean(
            result.baseline_vulnerability, high
        ),
        "mean_harm_effect_vs_safe": record.mean_harm_effect_vs_safe,
        "total_spending_effect_vs_safe_cents": (
            record.total_spending_effect_vs_safe_cents
        ),
        "harmful_spending_effect_vs_safe_cents": (
            record.harmful_spending_effect_vs_safe_cents
        ),
        "total_revenue_effect_vs_safe_cents": (
            record.total_revenue_effect_vs_safe_cents
        ),
        "epgc_minimum_public_contribution_cents": (
            result.epgc.minimum_public_contribution_cents if result.epgc else 0
        ),
        "epgc_profit_safe_cents": result.epgc.profit_safe_cents if result.epgc else 0,
    }
    for source, value in result.revenue_composition_cents.items():
        row[f"revenue_{source}_cents"] = value
    return row


def _cohort_digest(players: object, life: object) -> str:
    digest = sha256()
    for table in (players, life):
        for name in table.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(table, name)
            if isinstance(value, np.ndarray):
                digest.update(name.encode("utf-8"))
                digest.update(value.dtype.str.encode("ascii"))
                digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
                digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _uncertainty(values: np.ndarray) -> tuple[float, float, float, float, float]:
    if not values.size:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    mean = float(values.mean())
    variance = float(values.var(ddof=1)) if values.size > 1 else 0.0
    standard_deviation = sqrt(variance)
    half_width = 1.96 * standard_deviation / sqrt(values.size)
    return mean, variance, standard_deviation, mean - half_width, mean + half_width


def _python_sum(values: np.ndarray) -> int:
    return sum(int(value) for value in values)


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _variance(values: np.ndarray) -> float:
    return float(np.var(values)) if values.size else 0.0


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if values.size else 0.0


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(values[mask])) if np.any(mask) else 0.0


__all__ = [
    "PolicyBatchResult",
    "PolicyBatchSpec",
    "SeedScenarioRecord",
    "run_policy_batch",
]
