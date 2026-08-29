"""One-at-a-time sensitivity analysis using common cohorts and random fields."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import sqrt
from types import MappingProxyType
from typing import Literal

import numpy as np

from ..causal.batch import (
    PolicyBatchSpec,
    PolicyRunInputs,
    _cohort_digest,
    build_policy_run_input_snapshot,
    policy_run_input_sha256,
    resolve_policy_run_inputs,
)
from ..causal.scenarios import ScenarioId
from ..consumers.population import CountryProfile, initialize_player_table
from ..consumers.welfare import initialize_player_life
from ..data.lineage import (
    ProfileInputLineage,
    resolve_profile_inputs,
)
from ..data.population_execution import (
    PopulationExecutionLineage,
    PopulationSeedExecutionRecord,
    build_population_execution_lineage,
    build_population_seed_execution_record,
)
from ..data.population_projection import (
    PopulationProjectionAdapter,
    initialize_population_projection,
    verify_population_projection_adapter,
)
from ..data.profiles import ProfileBundle
from ..funding import EPGCPolicy
from ..metrics.harm import (
    HarmComponent,
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmWeights,
)
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
_ROW_IDENTITY_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class SensitivityCase:
    """A named OAT parameter grid and expected primary-harm direction."""

    parameter: str
    values: tuple[float, ...]
    scenario_id: ScenarioId = ScenarioId.BASELINE_F2P
    expected_direction: Direction = "none"

    def __post_init__(self) -> None:
        if type(self.parameter) is not str:
            raise TypeError("sensitivity parameter must be a string")
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
        if self.parameter == "decision_temperature":
            if any(not 0.0 < value <= 5.0 for value in values):
                raise ValueError(
                    "decision_temperature levels must be in (0, 5]"
                )
        elif any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(
                f"{self.parameter} levels must be in [0, 1]"
            )
        object.__setattr__(self, "values", values)
        if type(self.scenario_id) is not ScenarioId:
            raise TypeError("scenario_id must be a ScenarioId")
        if type(self.expected_direction) is not str:
            raise TypeError("expected_direction must be a string")
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

    batch_spec: PolicyBatchSpec
    cases: tuple[SensitivityCase, ...]
    instability_cv_threshold: float
    run_inputs: PolicyRunInputs
    rows: tuple[Mapping[str, object], ...]
    unstable_parameters: tuple[str, ...]
    country_profiles: tuple[CountryProfile, ...] = ()
    profile_input_lineage: ProfileInputLineage | None = None
    population_execution_lineage: PopulationExecutionLineage | None = None

    def __post_init__(self) -> None:
        if type(self.batch_spec) is not PolicyBatchSpec:
            raise TypeError("batch_spec must be PolicyBatchSpec")
        cases = _validated_cases(self.cases)
        threshold = _validated_instability_threshold(
            self.instability_cv_threshold
        )
        if type(self.run_inputs) is not PolicyRunInputs:
            raise TypeError("run_inputs must be PolicyRunInputs")
        rows, derived_unstable_parameters = _canonical_sensitivity_rows(
            self.rows,
            batch_spec=self.batch_spec,
            cases=cases,
            instability_cv_threshold=threshold,
        )
        unstable_parameters = tuple(self.unstable_parameters)
        if any(
            type(parameter) is not str
            for parameter in unstable_parameters
        ):
            raise TypeError("unstable_parameters must contain strings")
        if len(set(unstable_parameters)) != len(unstable_parameters):
            raise ValueError("unstable_parameters must be unique")
        if unstable_parameters != derived_unstable_parameters:
            raise ValueError(
                "unstable_parameters do not match the retained threshold and rows"
            )
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "instability_cv_threshold", threshold)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "unstable_parameters", unstable_parameters)
        profiles = tuple(self.country_profiles)
        if any(not isinstance(profile, CountryProfile) for profile in profiles):
            raise TypeError("country_profiles must contain CountryProfile instances")
        object.__setattr__(self, "country_profiles", profiles)
        if self.profile_input_lineage is not None:
            if not isinstance(self.profile_input_lineage, ProfileInputLineage):
                raise TypeError("profile_input_lineage must be ProfileInputLineage")
            self.profile_input_lineage.validate_country_profiles(profiles)
        population_lineage = self.population_execution_lineage
        if population_lineage is not None:
            if type(population_lineage) is not PopulationExecutionLineage:
                raise TypeError(
                    "population_execution_lineage must be "
                    "PopulationExecutionLineage or None"
                )
            PopulationExecutionLineage.__post_init__(population_lineage)
            if tuple(
                record.seed for record in population_lineage.seed_records
            ) != self.batch_spec.seeds:
                raise ValueError(
                    "population execution seed records do not match sensitivity seeds"
                )
            if (
                population_lineage.adapter.apportionment_plan.player_count
                != self.batch_spec.player_count
            ):
                raise ValueError(
                    "population execution player count does not match batch spec"
                )
            if any(
                record.policy_days != self.batch_spec.days
                for record in population_lineage.seed_records
            ):
                raise ValueError(
                    "population execution policy horizon does not match sensitivity"
                )

    def execution_snapshot(self) -> dict[str, object]:
        """Return the exact OAT design and resolved execution inputs."""

        payload = {
            "schema_version": "1.0",
            "batch_spec": self.batch_spec.snapshot(),
            "cases": [
                {
                    "parameter": case.parameter,
                    "values": list(case.values),
                    "scenario_id": case.scenario_id.value,
                    "expected_direction": case.expected_direction,
                }
                for case in self.cases
            ],
            "instability_cv_threshold": self.instability_cv_threshold,
            "numerical_tolerances": {
                "coefficient_of_variation_zero_mean": (
                    _CV_ZERO_MEAN_TOLERANCE
                ),
                "monotonicity": _MONOTONIC_TOLERANCE,
                "row_identity": _ROW_IDENTITY_TOLERANCE,
            },
            "run_inputs": self.run_inputs.snapshot(),
            "profile_input_fingerprint_sha256": (
                self.profile_input_lineage.fingerprint_sha256
                if self.profile_input_lineage is not None
                else None
            ),
        }
        if self.population_execution_lineage is None:
            return payload
        return {
            **payload,
            "schema_version": "2.0",
            "population_execution": (
                self.population_execution_lineage.manifest_payload()
            ),
        }

    def run_input_snapshot(self) -> dict[str, object]:
        """Return the canonical batch/model/profile execution inputs."""

        return build_policy_run_input_snapshot(
            batch_spec=self.batch_spec,
            run_inputs=self.run_inputs,
            profile_input_fingerprint_sha256=(
                self.profile_input_lineage.fingerprint_sha256
                if self.profile_input_lineage is not None
                else None
            ),
            population_adapter=(
                self.population_execution_lineage.adapter
                if self.population_execution_lineage is not None
                else None
            ),
        )

    def run_input_sha256(self) -> str:
        """Hash :meth:`run_input_snapshot` canonically."""

        return policy_run_input_sha256(
            batch_spec=self.batch_spec,
            run_inputs=self.run_inputs,
            profile_input_fingerprint_sha256=(
                self.profile_input_lineage.fingerprint_sha256
                if self.profile_input_lineage is not None
                else None
            ),
            population_adapter=(
                self.population_execution_lineage.adapter
                if self.population_execution_lineage is not None
                else None
            ),
        )

    def execution_sha256(self) -> str:
        """Hash the canonical :meth:`execution_snapshot` payload."""

        encoded = json.dumps(
            self.execution_snapshot(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def _validated_cases(
    cases: Sequence[SensitivityCase],
) -> tuple[SensitivityCase, ...]:
    selected = tuple(cases)
    if not selected:
        raise ValueError("at least one sensitivity case is required")
    if any(type(case) is not SensitivityCase for case in selected):
        raise TypeError("cases must contain SensitivityCase instances")
    parameters = tuple(case.parameter for case in selected)
    if len(set(parameters)) != len(parameters):
        raise ValueError(
            "sensitivity cases must use unique parameter names because the "
            "output schema has no case identifier"
        )
    return selected


def _validated_instability_threshold(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("instability_cv_threshold must be numeric")
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "instability_cv_threshold must be finite and non-negative"
        )
    return threshold


def _canonical_sensitivity_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    batch_spec: PolicyBatchSpec,
    cases: tuple[SensitivityCase, ...],
    instability_cv_threshold: float,
) -> tuple[tuple[Mapping[str, object], ...], tuple[str, ...]]:
    from ..outputs.schema import SENSITIVITY_COLUMNS

    expected_columns = frozenset(SENSITIVITY_COLUMNS)
    expected_keys = tuple(
        (case.parameter, value, case.scenario_id.value)
        for case in cases
        for value in case.values
    )
    provided_rows: dict[tuple[str, float, str], dict[str, object]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise TypeError("sensitivity rows must be mappings")
        supplied = dict(raw_row)
        supplied_columns = frozenset(supplied)
        if supplied_columns != expected_columns:
            missing = sorted(expected_columns - supplied_columns)
            extra = sorted(supplied_columns - expected_columns)
            raise ValueError(
                "sensitivity row columns do not match SENSITIVITY_COLUMNS; "
                f"missing={missing}, extra={extra}"
            )
        row = {column: supplied[column] for column in SENSITIVITY_COLUMNS}
        parameter = row["parameter"]
        raw_value = row["parameter_value"]
        scenario_id = row["scenario_id"]
        seed_count = row["seed_count"]
        if type(parameter) is not str or type(scenario_id) is not str:
            raise TypeError(
                "sensitivity row parameter and scenario_id must be strings"
            )
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("sensitivity row parameter_value must be numeric")
        parameter_value = float(raw_value)
        if not np.isfinite(parameter_value):
            raise ValueError("sensitivity row parameter_value must be finite")
        row["parameter_value"] = parameter_value
        if isinstance(seed_count, bool) or not isinstance(
            seed_count,
            (int, np.integer),
        ):
            raise TypeError("sensitivity row seed_count must be an integer")
        normalized_seed_count = int(seed_count)
        row["seed_count"] = normalized_seed_count
        if normalized_seed_count != len(batch_spec.seeds):
            raise ValueError(
                "sensitivity row seed_count does not match batch_spec.seeds"
            )
        key = (parameter, parameter_value, scenario_id)
        if key in provided_rows:
            raise ValueError("duplicate sensitivity row")
        provided_rows[key] = row
    if set(provided_rows) != set(expected_keys):
        raise ValueError("sensitivity rows do not match the selected cases")
    canonical_rows: list[Mapping[str, object]] = []
    unstable_parameters: set[str] = set()
    for case in cases:
        case_rows = [
            provided_rows[(case.parameter, value, case.scenario_id.value)]
            for value in case.values
        ]
        level_metrics: list[tuple[float, float]] = []
        coefficient_values: list[float] = []
        for value, row in zip(case.values, case_rows):
            expected_direction = row["expected_direction"]
            if type(expected_direction) is not str:
                raise TypeError(
                    "sensitivity row expected_direction must be a string"
                )
            if expected_direction != case.expected_direction:
                raise ValueError(
                    "sensitivity row expected_direction does not match its case"
                )
            monotonic_expected = row.get("monotonic_expected")
            if not isinstance(monotonic_expected, bool):
                raise TypeError(
                    "sensitivity row monotonic_expected must be boolean"
                )
            if monotonic_expected != (case.expected_direction != "none"):
                raise ValueError(
                    "sensitivity row monotonic_expected does not match its case"
                )
            mean_harm_value = _row_float(row, "mean_harm")
            variance_value = _row_float(
                row,
                "harm_variance",
                minimum=0.0,
            )
            standard_deviation_value = _row_float(
                row,
                "harm_sd",
                minimum=0.0,
            )
            ci_low_value = _row_float(row, "harm_ci95_low")
            ci_high_value = _row_float(row, "harm_ci95_high")
            _row_float(row, "total_revenue_cents", minimum=0.0)
            _row_float(
                row,
                "opportunity_cost_burden",
                minimum=0.0,
                maximum=1.0,
            )
            _row_float(
                row,
                "minimum_public_contribution_cents",
                minimum=0.0,
            )
            coefficient_value = _row_float(
                row,
                "harm_coefficient_of_variation",
                allow_positive_infinity=True,
                minimum=0.0,
            )
            if not 0.0 <= mean_harm_value <= 1.0:
                raise ValueError("sensitivity row mean_harm must be in [0, 1]")
            expected_standard_deviation = sqrt(variance_value)
            _require_row_identity(
                "harm_sd",
                standard_deviation_value,
                expected_standard_deviation,
            )
            half_width = (
                1.96
                * standard_deviation_value
                / sqrt(len(batch_spec.seeds))
            )
            _require_row_identity(
                "harm_ci95_low",
                ci_low_value,
                mean_harm_value - half_width,
            )
            _require_row_identity(
                "harm_ci95_high",
                ci_high_value,
                mean_harm_value + half_width,
            )
            expected_coefficient = (
                standard_deviation_value / abs(mean_harm_value)
                if abs(mean_harm_value) > _CV_ZERO_MEAN_TOLERANCE
                else (
                    0.0
                    if standard_deviation_value == 0.0
                    else float("inf")
                )
            )
            if not (
                np.isposinf(coefficient_value)
                and np.isposinf(expected_coefficient)
            ):
                _require_row_identity(
                    "harm_coefficient_of_variation",
                    coefficient_value,
                    expected_coefficient,
                )
            if np.isnan(coefficient_value) or coefficient_value < 0.0:
                raise ValueError(
                    "sensitivity row harm_coefficient_of_variation must be "
                    "non-negative and not NaN"
                )
            level_metrics.append((value, mean_harm_value))
            coefficient_values.append(coefficient_value)
        monotonic_observed = _monotonic(
            level_metrics,
            case.expected_direction,
        )
        case_unstable = any(
            value > instability_cv_threshold
            for value in coefficient_values
        ) or (
            case.expected_direction != "none" and not monotonic_observed
        )
        if case_unstable:
            unstable_parameters.add(case.parameter)
        for row in case_rows:
            stored_monotonic = row.get("monotonic_observed")
            stored_unstable = row.get("unstable")
            if not isinstance(stored_monotonic, bool):
                raise TypeError(
                    "sensitivity row monotonic_observed must be boolean"
                )
            if not isinstance(stored_unstable, bool):
                raise TypeError("sensitivity row unstable must be boolean")
            if stored_monotonic != monotonic_observed:
                raise ValueError(
                    "sensitivity row monotonic_observed is inconsistent"
                )
            if stored_unstable != case_unstable:
                raise ValueError(
                    "sensitivity row unstable flag is inconsistent"
                )
            canonical_rows.append(MappingProxyType(row))
    return tuple(canonical_rows), tuple(sorted(unstable_parameters))


def _row_float(
    row: dict[str, object],
    name: str,
    *,
    allow_positive_infinity: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = row[name]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"sensitivity row {name} must be numeric")
    normalized = float(value)
    if not np.isfinite(normalized) and not (
        allow_positive_infinity and np.isposinf(normalized)
    ):
        raise ValueError(f"sensitivity row {name} must be finite")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"sensitivity row {name} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"sensitivity row {name} must be at most {maximum}")
    normalized = 0.0 if normalized == 0.0 else normalized
    row[name] = normalized
    return normalized


def _require_row_identity(name: str, observed: float, expected: float) -> None:
    if not np.isclose(
        observed,
        expected,
        rtol=_ROW_IDENTITY_TOLERANCE,
        atol=_ROW_IDENTITY_TOLERANCE,
    ):
        raise ValueError(f"sensitivity row {name} is inconsistent")


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
    population_adapter: PopulationProjectionAdapter | None = None,
) -> SensitivityResult:
    """Evaluate OAT levels with identical cohorts and shocks within each seed.

    ``unstable`` means either that an expected direction is violated beyond a
    small numerical tolerance or that between-seed dispersion is large relative
    to the mean.  It is a model diagnostic, not an empirical uncertainty claim.
    """

    if type(batch_spec) is not PolicyBatchSpec:
        raise TypeError("batch_spec must be PolicyBatchSpec")
    selected = _validated_cases(
        tuple(cases) if cases is not None else default_sensitivity_cases()
    )
    instability_cv_threshold = _validated_instability_threshold(
        instability_cv_threshold
    )
    run_inputs = resolve_policy_run_inputs(
        harm_parameters=base_harm_parameters,
        harm_weights=harm_weights,
        opportunity_valuation=opportunity_valuation,
        producer_assumptions=producer_assumptions,
        epgc_policy=epgc_policy,
    )
    profiles, profile_lineage = resolve_profile_inputs(
        country_profiles=country_profiles,
        profile_bundle=profile_bundle,
    )
    cohorts = {}
    population_seed_records: list[PopulationSeedExecutionRecord] = []
    if population_adapter is not None:
        if type(population_adapter) is not PopulationProjectionAdapter:
            raise TypeError(
                "population_adapter must be PopulationProjectionAdapter or None"
            )
        population_adapter = verify_population_projection_adapter(
            population_adapter
        )
        if (
            population_adapter.apportionment_plan.player_count
            != batch_spec.player_count
        ):
            raise ValueError(
                "population adapter player count does not match the batch spec"
            )
    for seed in batch_spec.seeds:
        rng = CounterRNG(seed)
        population_execution = None
        if population_adapter is None:
            players = initialize_player_table(batch_spec.player_count, profiles, rng)
        else:
            population_execution = initialize_population_projection(
                population_adapter,
                profiles,
                rng,
            )
            players = population_execution.players
        life = initialize_player_life(players, rng)
        cohorts[seed] = (players, life)
        if population_execution is not None:
            population_seed_records.append(
                build_population_seed_execution_record(
                    population_execution,
                    seed=seed,
                    cohort_digest=_cohort_digest(players, life),
                    policy_days=batch_spec.days,
                )
            )

    population_lineage = (
        build_population_execution_lineage(
            population_adapter,
            population_seed_records,
        )
        if population_adapter is not None
        else None
    )

    rows: list[dict[str, object]] = []
    unstable: set[str] = set()
    for case in selected:
        level_metrics: list[tuple[float, float]] = []
        level_rows: list[dict[str, object]] = []
        for value in case.values:
            scenario, decision, harm_parameters = _case_configuration(
                case,
                value,
                batch_spec,
                run_inputs.harm_parameters,
            )
            harm_by_seed: list[float] = []
            revenue_by_seed: list[float] = []
            opportunity_by_seed: list[float] = []
            subsidy_by_seed: list[float] = []
            for seed in batch_spec.seeds:
                players, life = cohorts[seed]
                result = run_policy_scenario(
                    players,
                    life,
                    scenario,
                    seed=seed,
                    days=batch_spec.days,
                    decision_parameters=decision,
                    harm_parameters=harm_parameters,
                    harm_weights=run_inputs.harm_weights,
                    opportunity_valuation=run_inputs.opportunity_valuation,
                    producer_assumptions=run_inputs.producer_assumptions,
                    epgc_policy=run_inputs.epgc_policy,
                )
                if population_lineage is not None:
                    expected_ids = np.asarray(
                        population_lineage.record_for_seed(
                            seed
                        ).exact_weights.player_ids,
                        dtype=np.int64,
                    )
                    if not np.array_equal(result.player_ids, expected_ids):
                        raise ValueError(
                            "population execution player ids do not match a "
                            "sensitivity result"
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
        batch_spec=batch_spec,
        cases=selected,
        instability_cv_threshold=instability_cv_threshold,
        run_inputs=run_inputs,
        rows=tuple(rows),
        unstable_parameters=tuple(sorted(unstable)),
        country_profiles=profiles,
        profile_input_lineage=profile_lineage,
        population_execution_lineage=population_lineage,
    )


def _case_configuration(
    case: SensitivityCase,
    value: float,
    batch_spec: PolicyBatchSpec,
    base_harm_parameters: HarmModelParameters,
):
    scenario = next(
        scenario
        for scenario in batch_spec.scenarios
        if scenario.scenario_id is case.scenario_id
    )
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
