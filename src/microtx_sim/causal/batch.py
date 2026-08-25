"""Repeated-seed, common-cohort counterfactual policy batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
from json import dumps
from math import sqrt
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ..consumers.decision import DecisionParameters
from ..consumers.population import CountryProfile, initialize_player_table
from ..consumers.welfare import initialize_player_life
from ..data.lineage import (
    ProfileInputLineage,
    resolve_profile_inputs,
)
from ..data.profiles import ProfileBundle
from ..funding import EPGCPolicy
from ..metrics.harm import (
    HarmComponent,
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmResult,
    WelfareHarmWeights,
)
from ..metrics.outcomes import _immutable_array_copy
from ..metrics.reporting import REPEATED_SEED_METRIC_STEMS
from ..rng import CounterRNG, validate_seed
from ..simulation.policy_orchestrator import (
    PolicyScenarioResult,
    ProducerAssumptions,
    default_epgc_policy,
    run_policy_scenario,
)
from .scenarios import ScenarioId, ScenarioSpec, required_scenarios


_POLICY_PRETREATMENT_RESULT_CONTRACTS = (
    ("player_ids", np.dtype(np.int64)),
    ("is_minor", np.dtype(np.bool_)),
    ("age_years", np.dtype(np.int16)),
    ("jurisdiction", np.dtype(np.int16)),
    ("baseline_vulnerability", np.dtype(np.float32)),
    ("disposable_budget_cents", np.dtype(np.int64)),
)

_POLICY_RESULT_ARRAY_CONTRACTS = (
    *_POLICY_PRETREATMENT_RESULT_CONTRACTS,
    ("spending_cents", np.dtype(np.int64)),
    ("composite_harm", np.dtype(np.float64)),
    ("enjoyment", np.dtype(np.float64)),
    ("high_risk", np.dtype(np.bool_)),
    ("action_minutes", np.dtype(np.int64)),
)


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
        seeds = tuple(self.seeds)
        if not seeds:
            raise ValueError("at least one seed is required")
        validated_seeds = tuple(
            validate_seed(seed, name=f"seeds[{index}]")
            for index, seed in enumerate(seeds)
        )
        if len(set(validated_seeds)) != len(validated_seeds):
            raise ValueError("seeds must be unique")
        object.__setattr__(self, "seeds", tuple(sorted(validated_seeds)))
        scenarios = tuple(self.scenarios)
        for index, scenario in enumerate(scenarios):
            if type(scenario) is not ScenarioSpec:
                raise TypeError(f"scenarios[{index}] must be a ScenarioSpec")
        object.__setattr__(self, "scenarios", scenarios)
        for name in ("days", "player_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, int(value))
        ids = tuple(scenario.scenario_id for scenario in scenarios)
        if len(set(ids)) != len(ids):
            raise ValueError("scenario ids must be unique")
        if set(ids) != set(ScenarioId):
            missing = sorted(item.value for item in set(ScenarioId) - set(ids))
            extra = sorted(str(item) for item in set(ids) - set(ScenarioId))
            raise ValueError(
                f"batch must contain exactly the seven required scenarios; "
                f"missing={missing}, extra={extra}"
            )
        if not isinstance(self.reference_scenario, ScenarioId):
            raise TypeError("reference_scenario must be a ScenarioId")
        if self.reference_scenario is not ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION:
            raise ValueError(
                "the effect_vs_safe output contract requires "
                "safe_fixed_price_subscription as its reference scenario"
            )
        if type(self.decision_parameters) is not DecisionParameters:
            raise TypeError("decision_parameters must be DecisionParameters")

    def snapshot(self) -> dict[str, object]:
        """Return the exact normalized batch design as JSON-compatible values."""

        snapshot = _primitive_snapshot(asdict(self))
        if not isinstance(snapshot, dict):
            raise AssertionError("PolicyBatchSpec snapshot must be a dictionary")
        return snapshot

    def snapshot_sha256(self) -> str:
        """Return the canonical SHA-256 digest of :meth:`snapshot`."""

        return _snapshot_sha256(self.snapshot())


@dataclass(frozen=True, slots=True)
class PolicyRunInputs:
    """Fully materialized model inputs shared by every scenario execution."""

    harm_parameters: HarmModelParameters
    harm_weights: WelfareHarmWeights
    opportunity_valuation: OpportunityCostValuation
    producer_assumptions: ProducerAssumptions
    epgc_policy: EPGCPolicy

    def __post_init__(self) -> None:
        expected_types = {
            "harm_parameters": HarmModelParameters,
            "harm_weights": WelfareHarmWeights,
            "opportunity_valuation": OpportunityCostValuation,
            "producer_assumptions": ProducerAssumptions,
            "epgc_policy": EPGCPolicy,
        }
        for name, expected_type in expected_types.items():
            if type(getattr(self, name)) is not expected_type:
                raise TypeError(f"{name} must be {expected_type.__name__}")
        self.harm_weights.as_array()

    def snapshot(self) -> dict[str, object]:
        """Return a detached primitive snapshot for manifests and comparisons."""

        snapshot = _primitive_snapshot(asdict(self))
        if not isinstance(snapshot, dict):
            raise AssertionError("PolicyRunInputs snapshot must be a dictionary")
        return snapshot

    def snapshot_sha256(self) -> str:
        """Return the canonical SHA-256 digest of :meth:`snapshot`."""

        return _snapshot_sha256(self.snapshot())


def resolve_policy_run_inputs(
    *,
    harm_parameters: HarmModelParameters | None = None,
    harm_weights: WelfareHarmWeights | None = None,
    opportunity_valuation: OpportunityCostValuation | None = None,
    producer_assumptions: ProducerAssumptions | None = None,
    epgc_policy: EPGCPolicy | None = None,
) -> PolicyRunInputs:
    """Resolve every optional policy input exactly once at the run boundary."""

    return PolicyRunInputs(
        harm_parameters=(
            harm_parameters
            if harm_parameters is not None
            else HarmModelParameters()
        ),
        harm_weights=(
            harm_weights
            if harm_weights is not None
            else WelfareHarmWeights()
        ),
        opportunity_valuation=(
            opportunity_valuation
            if opportunity_valuation is not None
            else OpportunityCostValuation()
        ),
        producer_assumptions=(
            producer_assumptions
            if producer_assumptions is not None
            else ProducerAssumptions()
        ),
        epgc_policy=(
            epgc_policy
            if epgc_policy is not None
            else default_epgc_policy()
        ),
    )


def build_policy_run_input_snapshot(
    *,
    batch_spec: PolicyBatchSpec,
    run_inputs: PolicyRunInputs,
    profile_input_fingerprint_sha256: str | None,
) -> dict[str, object]:
    """Build the canonical execution-input payload shared by all exports."""

    if type(batch_spec) is not PolicyBatchSpec:
        raise TypeError("batch_spec must be PolicyBatchSpec")
    if type(run_inputs) is not PolicyRunInputs:
        raise TypeError("run_inputs must be PolicyRunInputs")
    fingerprint = profile_input_fingerprint_sha256
    if fingerprint is not None:
        if not isinstance(fingerprint, str):
            raise TypeError(
                "profile_input_fingerprint_sha256 must be a string or None"
            )
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError(
                "profile_input_fingerprint_sha256 must be lowercase SHA-256 hex"
            )
    return {
        "batch_spec": batch_spec.snapshot(),
        "model_inputs": run_inputs.snapshot(),
        "profile_input_fingerprint_sha256": fingerprint,
    }


def policy_run_input_sha256(
    *,
    batch_spec: PolicyBatchSpec,
    run_inputs: PolicyRunInputs,
    profile_input_fingerprint_sha256: str | None,
) -> str:
    """Hash the canonical execution-input payload shared by all exports."""

    return _snapshot_sha256(
        build_policy_run_input_snapshot(
            batch_spec=batch_spec,
            run_inputs=run_inputs,
            profile_input_fingerprint_sha256=(
                profile_input_fingerprint_sha256
            ),
        )
    )


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
    run_inputs: PolicyRunInputs
    country_profiles: tuple[CountryProfile, ...] = ()
    profile_input_lineage: ProfileInputLineage | None = None

    def __post_init__(self) -> None:
        if type(self.spec) is not PolicyBatchSpec:
            raise TypeError("spec must be PolicyBatchSpec")
        if type(self.run_inputs) is not PolicyRunInputs:
            raise TypeError("run_inputs must be PolicyRunInputs")
        profiles = tuple(self.country_profiles)
        if any(not isinstance(profile, CountryProfile) for profile in profiles):
            raise TypeError("country_profiles must contain CountryProfile instances")
        if self.spec.player_count and not profiles:
            raise ValueError("country_profiles are required for non-empty policy cohorts")
        object.__setattr__(self, "country_profiles", profiles)
        expected = len(self.spec.seeds) * len(self.spec.scenarios)
        if len(self.records) != expected:
            raise ValueError(f"expected {expected} seed-scenario records")
        normalized_records: list[SeedScenarioRecord] = []
        for record in self.records:
            if type(record) is not SeedScenarioRecord:
                raise TypeError("records must contain SeedScenarioRecord instances")
            frozen_result = _readonly_policy_result(record.result)
            record = replace(record, result=frozen_result)
            if (
                type(record.result.days) is not int
                or record.result.days != self.spec.days
            ):
                raise ValueError("record result days do not match the batch spec")
            if record.result.player_ids.size != self.spec.player_count:
                raise ValueError(
                    "record result player count does not match the batch spec"
                )
            _validate_policy_pretreatment_result(
                record.result,
                expected_player_count=self.spec.player_count,
                country_profiles=profiles,
            )
            _validate_policy_result_outcomes(
                record.result,
                harm_weights=self.run_inputs.harm_weights,
            )
            normalized_records.append(record)
        record_by_key = {
            (record.result.seed, record.result.scenario.scenario_id): record
            for record in normalized_records
        }
        if len(record_by_key) != expected:
            raise ValueError("duplicate seed-scenario result")
        expected_keys = {
            (seed, scenario.scenario_id)
            for seed in self.spec.seeds
            for scenario in self.spec.scenarios
        }
        if set(record_by_key) != expected_keys:
            raise ValueError("seed-scenario records do not match the batch spec")
        records = tuple(
            record_by_key[(seed, scenario.scenario_id)]
            for seed in self.spec.seeds
            for scenario in self.spec.scenarios
        )
        for seed in self.spec.seeds:
            reference = record_by_key[
                (seed, self.spec.reference_scenario)
            ].result
            for scenario in self.spec.scenarios:
                result = record_by_key[(seed, scenario.scenario_id)].result
                if result.scenario != scenario:
                    raise ValueError(
                        "record result scenario does not exactly match the batch spec: "
                        f"{scenario.scenario_id.value}"
                    )
                _assert_policy_pretreatment_alignment(
                    result,
                    reference,
                )
                _validate_record_effects(
                    record_by_key[(seed, scenario.scenario_id)],
                    reference,
                )
        object.__setattr__(self, "records", records)
        provided_digests: dict[int, str] = {}
        for raw_seed, value in self.cohort_digest_by_seed.items():
            seed = validate_seed(raw_seed, name="cohort_digest_by_seed key")
            if seed in provided_digests:
                raise ValueError("cohort digest metadata contain duplicate seeds")
            provided_digests[seed] = str(value)
        if set(provided_digests) != set(self.spec.seeds):
            raise ValueError("cohort digest metadata do not match seeds")
        digests = {
            seed: provided_digests[seed]
            for seed in self.spec.seeds
        }
        for record in records:
            if digests[record.result.seed] != record.cohort_digest:
                raise ValueError("record cohort digest is inconsistent")
        object.__setattr__(self, "cohort_digest_by_seed", MappingProxyType(digests))
        if self.profile_input_lineage is not None:
            if not isinstance(self.profile_input_lineage, ProfileInputLineage):
                raise TypeError("profile_input_lineage must be ProfileInputLineage")
            self.profile_input_lineage.validate_country_profiles(profiles)

    def run_input_snapshot(self) -> dict[str, object]:
        """Return the canonical design, model inputs, and profile locator."""

        return build_policy_run_input_snapshot(
            batch_spec=self.spec,
            run_inputs=self.run_inputs,
            profile_input_fingerprint_sha256=(
                self.profile_input_lineage.fingerprint_sha256
                if self.profile_input_lineage is not None
                else None
            ),
        )

    def run_input_sha256(self) -> str:
        """Hash :meth:`run_input_snapshot` canonically."""

        return _snapshot_sha256(self.run_input_snapshot())

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
            for metric in REPEATED_SEED_METRIC_STEMS:
                values = np.asarray([float(row[metric]) for row in rows])
                mean, variance, standard_deviation, low, high = _uncertainty(values)
                result[f"{metric}_mean"] = mean
                result[f"{metric}_variance"] = variance
                result[f"{metric}_sd"] = standard_deviation
                result[f"{metric}_ci95_low"] = low
                result[f"{metric}_ci95_high"] = high
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
    profile_bundle: ProfileBundle | None = None,
    harm_parameters: HarmModelParameters | None = None,
    harm_weights: WelfareHarmWeights | None = None,
    opportunity_valuation: OpportunityCostValuation | None = None,
    producer_assumptions: ProducerAssumptions | None = None,
    epgc_policy: EPGCPolicy | None = None,
) -> PolicyBatchResult:
    """Run all scenarios on the same seeded cohort within each replication."""

    if type(spec) is not PolicyBatchSpec:
        raise TypeError("spec must be PolicyBatchSpec")
    run_inputs = resolve_policy_run_inputs(
        harm_parameters=harm_parameters,
        harm_weights=harm_weights,
        opportunity_valuation=opportunity_valuation,
        producer_assumptions=producer_assumptions,
        epgc_policy=epgc_policy,
    )
    profiles, profile_lineage = resolve_profile_inputs(
        country_profiles=country_profiles,
        profile_bundle=profile_bundle,
    )
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
            result = run_policy_scenario(
                players,
                life,
                scenario,
                seed=seed,
                days=spec.days,
                decision_parameters=spec.decision_parameters,
                harm_parameters=run_inputs.harm_parameters,
                harm_weights=run_inputs.harm_weights,
                opportunity_valuation=run_inputs.opportunity_valuation,
                producer_assumptions=run_inputs.producer_assumptions,
                epgc_policy=run_inputs.epgc_policy,
            )
            if _cohort_digest(players, life) != digest:
                raise RuntimeError(
                    "policy branch mutated the shared pre-treatment cohort"
                )
            if result.scenario != scenario:
                raise ValueError(
                    "policy runner returned a result for a different scenario"
                )
            _validate_policy_pretreatment_result(
                result,
                expected_player_count=spec.player_count,
                country_profiles=profiles,
            )
            _validate_policy_result_outcomes(
                result,
                harm_weights=run_inputs.harm_weights,
            )
            scenario_results[scenario.scenario_id] = result
        reference = scenario_results[spec.reference_scenario]
        for scenario in spec.scenarios:
            result = scenario_results[scenario.scenario_id]
            _assert_policy_pretreatment_alignment(
                result,
                reference,
            )
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
        run_inputs=run_inputs,
        country_profiles=profiles,
        profile_input_lineage=profile_lineage,
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


def _readonly_array_copy(values: np.ndarray) -> np.ndarray:
    return _immutable_array_copy(values)


def _readonly_policy_result(result: PolicyScenarioResult) -> PolicyScenarioResult:
    """Give a retained batch an immutable, independently owned result snapshot."""

    if type(result) is not PolicyScenarioResult:
        raise TypeError("record result must be PolicyScenarioResult")
    for name, expected_dtype in _POLICY_RESULT_ARRAY_CONTRACTS:
        values = getattr(result, name)
        if type(values) is not np.ndarray:
            raise TypeError(f"policy result field {name} must be a numpy array")
        if values.dtype != expected_dtype:
            raise ValueError(
                f"policy result field {name} must have dtype {expected_dtype.name}"
            )
    if type(result.harm) is not WelfareHarmResult:
        raise TypeError("policy result harm must be WelfareHarmResult")
    harm_updates: dict[str, np.ndarray] = {}
    for descriptor in fields(result.harm):
        values = getattr(result.harm, descriptor.name)
        if type(values) is not np.ndarray:
            raise TypeError(
                f"policy harm result field {descriptor.name} must be a numpy array"
            )
        harm_updates[descriptor.name] = _readonly_array_copy(values)
    frozen_harm = replace(result.harm, **harm_updates)
    result_updates = {
        name: _readonly_array_copy(getattr(result, name))
        for name, _ in _POLICY_RESULT_ARRAY_CONTRACTS
    }
    frozen_result = replace(result, harm=frozen_harm, **result_updates)
    for name in (
        "total_revenue_cents",
        "producer_cost_cents",
        "producer_profit_cents",
    ):
        if type(getattr(frozen_result, name)) is not int:
            raise TypeError(f"policy result field {name} must be a built-in integer")
    return frozen_result


def _validate_record_effects(
    record: SeedScenarioRecord,
    reference: PolicyScenarioResult,
) -> None:
    result = record.result
    with np.errstate(over="ignore", invalid="ignore"):
        harm_difference = result.composite_harm - reference.composite_harm
        expected_mean = (
            float(harm_difference.mean()) if harm_difference.size else 0.0
        )
    if not np.isfinite(expected_mean):
        raise ValueError("recomputed mean_harm_effect_vs_safe must be finite")
    actual_mean = record.mean_harm_effect_vs_safe
    if type(actual_mean) is not float:
        raise TypeError("mean_harm_effect_vs_safe must be a built-in float")
    if not np.isfinite(actual_mean):
        raise ValueError("mean_harm_effect_vs_safe must be finite")
    if actual_mean != expected_mean:
        raise ValueError("mean_harm_effect_vs_safe does not match paired results")

    expected_integer_effects = {
        "total_spending_effect_vs_safe_cents": (
            _python_sum(result.spending_cents)
            - _python_sum(reference.spending_cents)
        ),
        "harmful_spending_effect_vs_safe_cents": (
            _python_sum(result.harm.harmful_spending_cents)
            - _python_sum(reference.harm.harmful_spending_cents)
        ),
        "total_revenue_effect_vs_safe_cents": (
            result.total_revenue_cents - reference.total_revenue_cents
        ),
    }
    for name, expected_value in expected_integer_effects.items():
        actual_value = getattr(record, name)
        if type(actual_value) is not int:
            raise TypeError(f"{name} must be a built-in integer")
        if actual_value != expected_value:
            raise ValueError(f"{name} does not match paired results")


def _assert_policy_pretreatment_alignment(
    comparison: PolicyScenarioResult,
    reference: PolicyScenarioResult,
) -> None:
    """Fail closed if a policy branch changes an invariant input field."""

    if comparison.seed != reference.seed:
        raise ValueError("policy counterfactual branches use different seeds")
    if comparison.days != reference.days:
        raise ValueError("policy counterfactual branches use different day horizons")
    if comparison.player_ids.size != reference.player_ids.size:
        raise ValueError("policy counterfactual branches have different player counts")
    for name, _ in _POLICY_PRETREATMENT_RESULT_CONTRACTS:
        comparison_values = getattr(comparison, name)
        reference_values = getattr(reference, name)
        if (
            comparison_values.shape != reference_values.shape
            or not np.array_equal(comparison_values, reference_values)
        ):
            raise ValueError(
                "policy counterfactual branches differ on pre-treatment/exogenous "
                f"field {name}: comparison={comparison.scenario.scenario_id.value}, "
                f"reference={reference.scenario.scenario_id.value}, seed={comparison.seed}"
            )


def _validate_policy_pretreatment_result(
    result: PolicyScenarioResult,
    *,
    expected_player_count: int,
    country_profiles: tuple[CountryProfile, ...] = (),
) -> None:
    if type(result) is not PolicyScenarioResult:
        raise TypeError("record result must be PolicyScenarioResult")
    expected_shape = (expected_player_count,)
    for name, expected_dtype in _POLICY_PRETREATMENT_RESULT_CONTRACTS:
        values = getattr(result, name)
        if type(values) is not np.ndarray:
            raise TypeError(f"policy pre-treatment field {name} must be a numpy array")
        if values.dtype != expected_dtype:
            raise ValueError(
                f"policy pre-treatment field {name} must have dtype "
                f"{expected_dtype.name}"
            )
        if values.shape != expected_shape:
            raise ValueError(
                f"policy pre-treatment field {name} must have shape "
                f"{expected_shape}"
            )

    player_ids = result.player_ids
    if np.any(player_ids < 0):
        raise ValueError("policy pre-treatment field player_ids must be non-negative")
    if not _has_unique_integer_ids(player_ids):
        raise ValueError("policy pre-treatment field player_ids must be unique")

    if np.any(result.age_years < 0):
        raise ValueError("policy pre-treatment field age_years cannot be negative")

    jurisdictions = result.jurisdiction
    if np.any(jurisdictions < 0):
        raise ValueError(
            "policy pre-treatment field jurisdiction contains an unknown code"
        )
    if country_profiles and np.any(jurisdictions >= len(country_profiles)):
        raise ValueError(
            "policy pre-treatment field jurisdiction contains an unknown code"
        )

    vulnerability = result.baseline_vulnerability
    if (
        not np.all(np.isfinite(vulnerability))
        or np.any(vulnerability < 0.0)
        or np.any(vulnerability > 1.0)
    ):
        raise ValueError(
            "policy pre-treatment field baseline_vulnerability must be finite "
            "and in [0, 1]"
        )

    if np.any(result.disposable_budget_cents < 0):
        raise ValueError(
            "policy pre-treatment field disposable_budget_cents cannot be negative"
        )

    if country_profiles and expected_player_count:
        adult_ages = np.asarray(
            [profile.adult_age for profile in country_profiles],
            dtype=np.int16,
        )
        expected_minor = result.age_years < adult_ages[jurisdictions]
        if not np.array_equal(result.is_minor, expected_minor):
            raise ValueError(
                "policy pre-treatment field is_minor is inconsistent with age "
                "and jurisdiction"
            )


def _validate_policy_result_outcomes(
    result: PolicyScenarioResult,
    *,
    harm_weights: WelfareHarmWeights,
) -> None:
    if np.any(result.spending_cents < 0):
        raise ValueError("policy result field spending_cents cannot be negative")
    if np.any(result.spending_cents > result.disposable_budget_cents):
        raise ValueError(
            "policy result field spending_cents cannot exceed disposable budget"
        )
    if np.any(result.action_minutes < 0):
        raise ValueError("policy result field action_minutes cannot be negative")
    with np.errstate(over="ignore", invalid="ignore"):
        expected_composite = result.harm.composite_harm(harm_weights)
    if not np.all(np.isfinite(expected_composite)):
        raise ValueError("recomputed composite_harm must be finite")
    if not np.array_equal(result.composite_harm, expected_composite):
        raise ValueError("composite_harm does not match component scores and weights")


def _has_unique_integer_ids(values: np.ndarray) -> bool:
    if values.size < 2 or bool(np.all(values[1:] > values[:-1])):
        return True
    return np.unique(values).size == values.size


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


def _primitive_snapshot(value: object) -> object:
    if isinstance(value, ScenarioId):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _primitive_snapshot(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_primitive_snapshot(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _snapshot_sha256(snapshot: dict[str, object]) -> str:
    encoded = dumps(
        snapshot,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "PolicyBatchResult",
    "PolicyBatchSpec",
    "PolicyRunInputs",
    "REPEATED_SEED_METRIC_STEMS",
    "SeedScenarioRecord",
    "build_policy_run_input_snapshot",
    "policy_run_input_sha256",
    "run_policy_batch",
    "resolve_policy_run_inputs",
]
