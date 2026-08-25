"""One-at-a-time sensitivity analysis using common cohorts and random fields."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Literal, Sequence

import numpy as np

from ..causal.batch import PolicyBatchSpec
from ..causal.scenarios import ScenarioId, scenario_by_id
from ..consumers.population import CountryProfile, initialize_player_table
from ..consumers.welfare import initialize_player_life
from ..data.lineage import (
    ProfileInputLineage,
    resolve_profile_inputs,
)
from ..data.profiles import ProfileBundle
from ..metrics.harm import HarmModelParameters, HarmComponent
from ..metrics.harm import OpportunityCostValuation, WelfareHarmWeights
from ..funding import EPGCPolicy
from ..rng import CounterRNG
from ..simulation.policy_orchestrator import ProducerAssumptions, run_policy_scenario


Direction = Literal["increasing", "decreasing", "none"]
_SUPPORTED_PARAMETERS = frozenset(
    {
        "paid_random_rewards",
        "time_limited_offers",
        "opaque_virtual_currency",
        "affordable_spending_share",
        "decision_temperature",
    }
)
_CV_ZERO_MEAN_TOLERANCE = 1e-12
_MONOTONIC_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class SensitivityCase:
    """A named OAT parameter grid and expected primary-harm direction."""

    parameter: str
    values: tuple[float, ...]
    scenario_id: ScenarioId = ScenarioId.BASELINE_F2P
    expected_direction: Direction = "none"

    def __post_init__(self) -> None:
        if self.parameter not in _SUPPORTED_PARAMETERS:
            raise ValueError(f"unsupported sensitivity parameter: {self.parameter}")
        raw_values = tuple(self.values)
        if len(raw_values) < 2:
            raise ValueError("a sensitivity case needs at least two levels")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            for value in raw_values
        ):
            raise TypeError("sensitivity levels must be numeric")
        values = tuple(float(value) for value in raw_values)
        if len(set(values)) != len(values):
            raise ValueError("sensitivity levels must be unique")
        if tuple(sorted(values)) != values:
            raise ValueError("sensitivity levels must be strictly increasing")
        if not all(np.isfinite(value) for value in values):
            raise ValueError("sensitivity levels must be finite")
        object.__setattr__(self, "values", values)
        if self.expected_direction not in ("increasing", "decreasing", "none"):
            raise ValueError("unknown expected direction")


def default_sensitivity_cases() -> tuple[SensitivityCase, ...]:
    """Return a compact face-valid synthetic sensitivity grid."""

    return (
        SensitivityCase(
            "paid_random_rewards", (0.0, 0.35, 0.70), expected_direction="increasing"
        ),
        SensitivityCase(
            "time_limited_offers", (0.0, 0.35, 0.70), expected_direction="increasing"
        ),
        SensitivityCase(
            "opaque_virtual_currency", (0.0, 0.375, 0.75), expected_direction="increasing"
        ),
        SensitivityCase(
            "affordable_spending_share", (0.05, 0.10, 0.20), expected_direction="decreasing"
        ),
        SensitivityCase(
            "decision_temperature", (0.40, 0.65, 1.00), expected_direction="none"
        ),
    )


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Tidy level summaries and parameters flagged as unstable."""

    rows: tuple[dict[str, object], ...]
    unstable_parameters: tuple[str, ...]
    country_profiles: tuple[CountryProfile, ...] = ()
    profile_input_lineage: ProfileInputLineage | None = None

    def __post_init__(self) -> None:
        profiles = tuple(self.country_profiles)
        if any(not isinstance(profile, CountryProfile) for profile in profiles):
            raise TypeError("country_profiles must contain CountryProfile instances")
        object.__setattr__(self, "country_profiles", profiles)
        if self.profile_input_lineage is not None:
            if not isinstance(self.profile_input_lineage, ProfileInputLineage):
                raise TypeError("profile_input_lineage must be ProfileInputLineage")
            self.profile_input_lineage.validate_country_profiles(profiles)


def run_sensitivity_analysis(
    batch_spec: PolicyBatchSpec,
    *,
    cases: Sequence[SensitivityCase] | None = None,
    country_profiles: Sequence[CountryProfile] | None = None,
    profile_bundle: ProfileBundle | None = None,
    instability_cv_threshold: float = 0.35,
    base_harm_parameters: HarmModelParameters | None = None,
    harm_weights: WelfareHarmWeights | None = None,
    opportunity_valuation: OpportunityCostValuation | None = None,
    producer_assumptions: ProducerAssumptions | None = None,
    epgc_policy: EPGCPolicy | None = None,
) -> SensitivityResult:
    """Evaluate OAT levels with identical cohorts and shocks within each seed.

    ``unstable`` means either that an expected direction is violated beyond a
    small numerical tolerance or that between-seed dispersion is large relative
    to the mean.  It is a model diagnostic, not an empirical uncertainty claim.
    """

    if not isinstance(batch_spec, PolicyBatchSpec):
        raise TypeError("batch_spec must be PolicyBatchSpec")
    selected = tuple(cases) if cases is not None else default_sensitivity_cases()
    if not selected:
        raise ValueError("at least one sensitivity case is required")
    parameters = tuple(case.parameter for case in selected)
    if len(set(parameters)) != len(parameters):
        raise ValueError(
            "sensitivity cases must use unique parameter names because the "
            "output schema has no case identifier"
        )
    if not np.isfinite(instability_cv_threshold) or instability_cv_threshold < 0:
        raise ValueError("instability_cv_threshold must be finite and non-negative")
    profiles, profile_lineage = resolve_profile_inputs(
        country_profiles=country_profiles,
        profile_bundle=profile_bundle,
    )
    cohorts = {}
    for seed in batch_spec.seeds:
        rng = CounterRNG(seed)
        players = initialize_player_table(batch_spec.player_count, profiles, rng)
        cohorts[seed] = (players, initialize_player_life(players, rng))

    rows: list[dict[str, object]] = []
    unstable: set[str] = set()
    for case in selected:
        level_metrics: list[tuple[float, float]] = []
        level_rows: list[dict[str, object]] = []
        for value in case.values:
            harm_by_seed: list[float] = []
            revenue_by_seed: list[float] = []
            opportunity_by_seed: list[float] = []
            subsidy_by_seed: list[float] = []
            for seed in batch_spec.seeds:
                players, life = cohorts[seed]
                scenario, decision, harm_parameters = _case_configuration(
                    case,
                    value,
                    batch_spec,
                    base_harm_parameters or HarmModelParameters(),
                )
                result = run_policy_scenario(
                    players,
                    life,
                    scenario,
                    seed=seed,
                    days=batch_spec.days,
                    decision_parameters=decision,
                    harm_parameters=harm_parameters,
                    harm_weights=harm_weights,
                    opportunity_valuation=opportunity_valuation,
                    producer_assumptions=producer_assumptions,
                    epgc_policy=epgc_policy,
                )
                harm_by_seed.append(
                    float(result.composite_harm.mean())
                    if len(result.composite_harm)
                    else 0.0
                )
                revenue_by_seed.append(float(result.total_revenue_cents))
                opportunity_by_seed.append(
                    float(result.harm.component_scores[:, HarmComponent.OC].mean())
                    if len(result.player_ids)
                    else 0.0
                )
                subsidy_by_seed.append(
                    float(result.epgc.minimum_public_contribution_cents)
                    if result.epgc
                    else 0.0
                )
            harm_stats = _stats(harm_by_seed)
            revenue_stats = _stats(revenue_by_seed)
            opportunity_stats = _stats(opportunity_by_seed)
            subsidy_stats = _stats(subsidy_by_seed)
            level_metrics.append((value, harm_stats[0]))
            coefficient_of_variation = (
                harm_stats[2] / abs(harm_stats[0])
                if abs(harm_stats[0]) > _CV_ZERO_MEAN_TOLERANCE
                else (0.0 if harm_stats[2] == 0.0 else float("inf"))
            )
            level_row: dict[str, object] = {
                "parameter": case.parameter,
                "parameter_value": value,
                "scenario_id": case.scenario_id.value,
                "seed_count": len(batch_spec.seeds),
                "mean_harm": harm_stats[0],
                "harm_variance": harm_stats[1],
                "harm_sd": harm_stats[2],
                "harm_ci95_low": harm_stats[3],
                "harm_ci95_high": harm_stats[4],
                "harm_coefficient_of_variation": coefficient_of_variation,
                "total_revenue_cents": revenue_stats[0],
                "opportunity_cost_burden": opportunity_stats[0],
                "minimum_public_contribution_cents": subsidy_stats[0],
                "expected_direction": case.expected_direction,
            }
            level_rows.append(level_row)
            if coefficient_of_variation > instability_cv_threshold:
                unstable.add(case.parameter)
        monotonic = _monotonic(level_metrics, case.expected_direction)
        if case.expected_direction != "none" and not monotonic:
            unstable.add(case.parameter)
        for row in level_rows:
            row["monotonic_expected"] = case.expected_direction != "none"
            row["monotonic_observed"] = monotonic
            row["unstable"] = case.parameter in unstable
            rows.append(row)
    return SensitivityResult(
        rows=tuple(rows),
        unstable_parameters=tuple(sorted(unstable)),
        country_profiles=profiles,
        profile_input_lineage=profile_lineage,
    )


def _case_configuration(
    case: SensitivityCase,
    value: float,
    batch_spec: PolicyBatchSpec,
    base_harm_parameters: HarmModelParameters,
):
    scenario = scenario_by_id(case.scenario_id)
    decision = batch_spec.decision_parameters
    harm_parameters = base_harm_parameters
    if case.parameter in {
        "paid_random_rewards",
        "time_limited_offers",
        "opaque_virtual_currency",
    }:
        scenario = replace(
            scenario,
            mechanics=replace(scenario.mechanics, **{case.parameter: value}),
        )
    elif case.parameter == "affordable_spending_share":
        harm_parameters = replace(
            harm_parameters, affordable_spending_share=value
        )
    elif case.parameter == "decision_temperature":
        decision = replace(decision, temperature=value)
    else:
        raise AssertionError(case.parameter)
    return scenario, decision, harm_parameters


def _monotonic(
    level_metrics: Sequence[tuple[float, float]], direction: Direction
) -> bool:
    if direction == "none":
        return True
    values = [metric for _, metric in sorted(level_metrics)]
    if direction == "increasing":
        return all(
            right + _MONOTONIC_TOLERANCE >= left
            for left, right in zip(values, values[1:])
        )
    return all(
        right <= left + _MONOTONIC_TOLERANCE
        for left, right in zip(values, values[1:])
    )


def _stats(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean()) if array.size else 0.0
    variance = float(array.var(ddof=1)) if array.size > 1 else 0.0
    standard_deviation = sqrt(variance)
    half = 1.96 * standard_deviation / sqrt(array.size) if array.size else 0.0
    return mean, variance, standard_deviation, mean - half, mean + half


__all__ = [
    "SensitivityCase",
    "SensitivityResult",
    "default_sensitivity_cases",
    "run_sensitivity_analysis",
]
