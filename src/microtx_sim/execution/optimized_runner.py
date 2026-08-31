"""Bounded, checkpoint-backed exploratory policy execution.

The host scheduler owns complete seeds: a worker creates one pre-treatment
cohort and runs every scenario in declared order.  The coordinator alone
commits result payloads.  Completed arrays are therefore released after each
seed and later decoded one seed at a time for analysis/finalization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
import multiprocessing as mp
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from ..analysis.sensitivity import (
    SensitivityCase,
    _CV_ZERO_MEAN_TOLERANCE,
    _case_configuration,
    _monotonic,
    _stats,
    _validated_cases,
    _validated_instability_threshold,
    default_sensitivity_cases,
)
from ..causal.batch import (
    PolicyBatchResult,
    PolicyBatchSpec,
    PolicyRunInputs,
    SeedScenarioRecord,
    _cohort_digest,
    resolve_policy_run_inputs,
    run_policy_batch,
)
from ..consumers.population import CountryProfile, initialize_player_table
from ..consumers.welfare import initialize_player_life
from ..data.lineage import ProfileInputLineage, resolve_profile_inputs
from ..data.population_execution import (
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
from ..policy_config import ExploratoryExecutionEngineConfig
from ..rng import CounterRNG
from ..simulation.policy_orchestrator import (
    ProducerAssumptions,
    run_policy_scenario,
)
from .backends import (
    BackendMode,
    ExecutionBackendConfig,
    ResolvedExecutionBackend,
    resolve_execution_backend,
)
from .checkpoints import (
    CheckpointIncompleteError,
    ExecutionWorkPlan,
    ResumableCheckpointStore,
    SensitivityWorkUnit,
    canonical_level_id,
)
from .native_threads import enforce_numpy_native_thread_limit
from .result_codec import (
    decode_seed_scenario_record,
    encode_seed_scenario_record,
)
from .kernels import validate_composite_harm_parity


_WorkItem = TypeVar("_WorkItem")
_WorkResult = TypeVar("_WorkResult")


@dataclass(frozen=True, slots=True)
class _ProcessWorkerContract:
    """Pickle-safe runtime contract re-attested in every spawned worker."""

    backend_config: ExecutionBackendConfig
    backend_identity_sha256: str
    resolved_backend: str
    native_thread_runtime: str
    native_thread_library_path: str
    native_thread_library_sha256: str
    native_thread_getter_symbol: str
    native_thread_setter_symbol: str
    native_thread_limit: int


@dataclass(frozen=True, slots=True)
class _PolicyProcessContext:
    """Static, pickle-safe model inputs transferred once per worker."""

    spec: PolicyBatchSpec
    country_profiles: tuple[CountryProfile, ...]
    run_inputs: PolicyRunInputs
    population_adapter: PopulationProjectionAdapter
    runtime: _ProcessWorkerContract


@dataclass(frozen=True, slots=True)
class _SensitivityProcessContext:
    """Static sensitivity inputs transferred once per spawned worker."""

    policy: _PolicyProcessContext
    cases: tuple[SensitivityCase, ...]


_PROCESS_POLICY_CONTEXT: _PolicyProcessContext | None = None
_PROCESS_SENSITIVITY_CONTEXT: _SensitivityProcessContext | None = None
_PROCESS_BACKEND: ResolvedExecutionBackend | None = None


@dataclass(slots=True)
class CheckpointedPolicyBatch:
    """A memory-bounded completed batch backed by verified checkpoint blocks."""

    spec: PolicyBatchSpec
    store: ResumableCheckpointStore
    run_inputs: PolicyRunInputs
    country_profiles: tuple[CountryProfile, ...]
    profile_input_lineage: ProfileInputLineage
    population_adapter: PopulationProjectionAdapter
    backend_metadata: Mapping[str, object]
    backend_parity: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.spec) is not PolicyBatchSpec:
            raise TypeError("spec must be PolicyBatchSpec")
        if type(self.store) is not ResumableCheckpointStore:
            raise TypeError("store must be ResumableCheckpointStore")
        if type(self.run_inputs) is not PolicyRunInputs:
            raise TypeError("run_inputs must be PolicyRunInputs")
        if tuple(self.store.work_plan.seeds) != self.spec.seeds:
            raise ValueError("checkpoint seeds differ from the policy batch")
        scenario_ids = tuple(
            scenario.scenario_id.value for scenario in self.spec.scenarios
        )
        if self.store.work_plan.scenario_ids != scenario_ids:
            raise ValueError("checkpoint scenario order differs from the batch")
        if self.store.remaining_main_seeds:
            raise CheckpointIncompleteError(
                "checkpointed policy batch still has incomplete seeds"
            )
        profiles = tuple(self.country_profiles)
        self.profile_input_lineage.validate_country_profiles(profiles)
        self.country_profiles = profiles
        self.population_adapter = verify_population_projection_adapter(
            self.population_adapter
        )
        if (
            self.population_adapter.apportionment_plan.player_count
            != self.spec.player_count
        ):
            raise ValueError("population adapter player count differs from batch")
        self.backend_metadata = MappingProxyType(dict(self.backend_metadata))
        self.backend_parity = MappingProxyType(dict(self.backend_parity))

    def load_seed_batch(self, seed: int) -> PolicyBatchResult:
        """Decode and fully revalidate one complete seed, never the full run."""

        if seed not in self.spec.seeds:
            raise KeyError(f"undeclared batch seed: {seed}")
        payloads = self.store.load_main_seed_payload(seed)
        records = tuple(
            decode_seed_scenario_record(
                payloads[scenario.scenario_id.value],
                batch_spec=self.spec,
            )
            for scenario in self.spec.scenarios
        )
        digests = {record.cohort_digest for record in records}
        if len(digests) != 1:
            raise ValueError("checkpoint scenarios do not share one cohort digest")
        digest = next(iter(digests))
        population_record = self.population_seed_record(
            seed,
            expected_cohort_digest=digest,
        )
        seed_spec = replace(self.spec, seeds=(seed,))
        lineage = build_population_execution_lineage(
            self.population_adapter,
            (population_record,),
        )
        return PolicyBatchResult(
            spec=seed_spec,
            records=records,
            cohort_digest_by_seed={seed: digest},
            run_inputs=self.run_inputs,
            country_profiles=self.country_profiles,
            profile_input_lineage=self.profile_input_lineage,
            population_execution_lineage=lineage,
            execution_backend_mode=(
                "gpu"
                if self.store.identity.backend.resolved_backend == "gpu"
                else "cpu_explicit"
            ),
            continuous_result_tolerance=(
                5e-13
                if self.store.identity.backend.resolved_backend == "gpu"
                else 0.0
            ),
        )

    def population_seed_record(
        self,
        seed: int,
        *,
        expected_cohort_digest: str | None,
    ) -> PopulationSeedExecutionRecord:
        """Recreate only deterministic pre-treatment attestation metadata."""

        rng = CounterRNG(seed)
        execution = initialize_population_projection(
            self.population_adapter,
            self.country_profiles,
            rng,
        )
        life = initialize_player_life(execution.players, rng)
        observed_digest = _cohort_digest(execution.players, life)
        if (
            expected_cohort_digest is not None
            and observed_digest != expected_cohort_digest
        ):
            raise ValueError(
                "checkpoint cohort differs from deterministic projected population"
            )
        return build_population_seed_execution_record(
            execution,
            seed=seed,
            cohort_digest=observed_digest,
            policy_days=self.spec.days,
        )

    def iter_seed_batches(self):
        for seed in self.spec.seeds:
            yield self.load_seed_batch(seed)


@dataclass(frozen=True, slots=True)
class CheckpointedSensitivityResult:
    """Compact OAT result assembled from atomic sensitivity work units."""

    batch_spec: PolicyBatchSpec
    cases: tuple[SensitivityCase, ...]
    instability_cv_threshold: float
    run_inputs: PolicyRunInputs
    rows: tuple[Mapping[str, object], ...]
    unstable_parameters: tuple[str, ...]
    cpu_reference_rows: tuple[Mapping[str, object], ...]
    cpu_reference_unstable_parameters: tuple[str, ...]
    conclusions_match_cpu_reference: bool
    maximum_absolute_harm_difference: float
    country_profiles: tuple[CountryProfile, ...]
    profile_input_lineage: ProfileInputLineage
    population_lineage_sha256: str | None


def sensitivity_work_units(
    batch_spec: PolicyBatchSpec,
    cases: Sequence[SensitivityCase] | None = None,
) -> tuple[SensitivityWorkUnit, ...]:
    selected = _validated_cases(
        tuple(cases) if cases is not None else default_sensitivity_cases()
    )
    return tuple(
        SensitivityWorkUnit(
            parameter_id=case.parameter,
            level_id=_level_id(case.values[index]),
            seed=seed,
        )
        for case in selected
        for index, _value in enumerate(case.values)
        for seed in batch_spec.seeds
    )


def expected_execution_work_plan(
    batch_spec: PolicyBatchSpec,
    *,
    sensitivity_enabled: bool,
    cases: Sequence[SensitivityCase] | None = None,
) -> ExecutionWorkPlan:
    return ExecutionWorkPlan.build(
        seeds=batch_spec.seeds,
        scenario_ids=tuple(
            scenario.scenario_id.value for scenario in batch_spec.scenarios
        ),
        sensitivity_units=(
            sensitivity_work_units(batch_spec, cases)
            if sensitivity_enabled
            else ()
        ),
    )


def _process_worker_contract(
    store: ResumableCheckpointStore,
    backend: ResolvedExecutionBackend,
) -> _ProcessWorkerContract:
    identity = store.identity.backend
    return _ProcessWorkerContract(
        backend_config=backend.config,
        backend_identity_sha256=backend.metadata.backend_identity_sha256,
        resolved_backend=backend.metadata.resolved_mode,
        native_thread_runtime=identity.native_thread_runtime,
        native_thread_library_path=identity.native_thread_library_path,
        native_thread_library_sha256=identity.native_thread_library_sha256,
        native_thread_getter_symbol=identity.native_thread_getter_symbol,
        native_thread_setter_symbol=identity.native_thread_setter_symbol,
        native_thread_limit=identity.native_thread_limit,
    )


def _initialize_process_runtime(
    contract: _ProcessWorkerContract,
) -> ResolvedExecutionBackend:
    """Recreate and verify non-pickleable runtime state after Windows spawn."""

    native = enforce_numpy_native_thread_limit(contract.native_thread_limit)
    observed_native = {
        "native_thread_runtime": native.runtime,
        "native_thread_library_path": native.library_path,
        "native_thread_library_sha256": native.library_sha256,
        "native_thread_getter_symbol": native.getter_symbol,
        "native_thread_setter_symbol": native.setter_symbol,
        "native_thread_limit": native.enforced_thread_count,
    }
    mismatches = [
        name
        for name, value in observed_native.items()
        if value != getattr(contract, name)
    ]
    if mismatches:
        raise RuntimeError(
            "spawned worker native-thread identity differs from coordinator: "
            + ", ".join(mismatches)
        )
    backend = resolve_execution_backend(contract.backend_config)
    if (
        backend.metadata.backend_identity_sha256
        != contract.backend_identity_sha256
        or backend.metadata.resolved_mode != contract.resolved_backend
    ):
        raise RuntimeError(
            "spawned worker backend identity differs from coordinator"
        )
    return backend


def _initialize_policy_process_worker(context: _PolicyProcessContext) -> None:
    global _PROCESS_BACKEND, _PROCESS_POLICY_CONTEXT
    _PROCESS_POLICY_CONTEXT = context
    _PROCESS_BACKEND = _initialize_process_runtime(context.runtime)


def _initialize_sensitivity_process_worker(
    context: _SensitivityProcessContext,
) -> None:
    global _PROCESS_BACKEND, _PROCESS_SENSITIVITY_CONTEXT
    _PROCESS_SENSITIVITY_CONTEXT = context
    _PROCESS_BACKEND = _initialize_process_runtime(context.policy.runtime)


def _compute_main_seed_payload(
    context: _PolicyProcessContext,
    backend: ResolvedExecutionBackend,
    seed: int,
) -> dict[str, object]:
    seed_spec = replace(context.spec, seeds=(seed,))
    result = run_policy_batch(
        seed_spec,
        country_profiles=context.country_profiles,
        harm_parameters=context.run_inputs.harm_parameters,
        harm_weights=context.run_inputs.harm_weights,
        opportunity_valuation=context.run_inputs.opportunity_valuation,
        producer_assumptions=context.run_inputs.producer_assumptions,
        epgc_policy=context.run_inputs.epgc_policy,
        population_adapter=context.population_adapter,
        execution_backend=backend,
    )
    return {
        record.result.scenario.scenario_id.value: encode_seed_scenario_record(
            record,
            batch_spec=context.spec,
        )
        for record in result.records
    }


def _process_main_seed(seed: int) -> dict[str, object]:
    if _PROCESS_POLICY_CONTEXT is None or _PROCESS_BACKEND is None:
        raise RuntimeError("spawned policy worker was not initialized")
    return _compute_main_seed_payload(
        _PROCESS_POLICY_CONTEXT,
        _PROCESS_BACKEND,
        seed,
    )


def _compute_sensitivity_payload(
    context: _SensitivityProcessContext,
    backend: ResolvedExecutionBackend,
    unit: SensitivityWorkUnit,
) -> dict[str, object]:
    case_by_parameter = {case.parameter: case for case in context.cases}
    case = case_by_parameter[unit.parameter_id]
    level_index = _level_index(case, unit.level_id)
    value = case.values[level_index]
    policy = context.policy
    rng = CounterRNG(unit.seed)
    population = initialize_population_projection(
        policy.population_adapter,
        policy.country_profiles,
        rng,
    )
    players = population.players
    life = initialize_player_life(players, rng)
    cohort_digest = _cohort_digest(players, life)
    scenario, decision, harm_parameters = _case_configuration(
        case,
        value,
        policy.spec,
        policy.run_inputs.harm_parameters,
    )
    result = run_policy_scenario(
        players,
        life,
        scenario,
        seed=unit.seed,
        days=policy.spec.days,
        decision_parameters=decision,
        harm_parameters=harm_parameters,
        harm_weights=policy.run_inputs.harm_weights,
        opportunity_valuation=policy.run_inputs.opportunity_valuation,
        producer_assumptions=policy.run_inputs.producer_assumptions,
        epgc_policy=policy.run_inputs.epgc_policy,
        execution_backend=backend,
    )
    if not np.array_equal(result.player_ids, players.player_id):
        raise ValueError("sensitivity result changed population ordering")
    cpu_reference_composite = result.harm.composite_harm(
        policy.run_inputs.harm_weights
    )
    observed_mean_harm = (
        float(result.composite_harm.mean())
        if result.composite_harm.size
        else 0.0
    )
    cpu_reference_mean_harm = (
        float(cpu_reference_composite.mean())
        if cpu_reference_composite.size
        else 0.0
    )
    if backend.mode is BackendMode.GPU:
        if not np.isclose(
            observed_mean_harm,
            cpu_reference_mean_harm,
            atol=5e-13,
            rtol=5e-13,
        ):
            raise ValueError(
                "GPU sensitivity mean exceeds its continuous tolerance"
            )
    elif not np.array_equal(result.composite_harm, cpu_reference_composite):
        raise ValueError("CPU sensitivity composite is not bitwise-identical")
    return {
        "schema_version": "microtx_sim.sensitivity_unit.v2",
        "parameter": case.parameter,
        "level_id": unit.level_id,
        "parameter_value": value,
        "scenario_id": case.scenario_id.value,
        "seed": unit.seed,
        "cohort_digest": cohort_digest,
        "monetary_semantics": {
            "internal_unit": "simulation_cents",
            "interpretation": "INTERNAL_MODEL_UNIT_NOT_REAL_MONEY",
            "observed_real_world_spending": False,
            "raw_cross_country_pooling": "PROHIBITED",
        },
        "mean_harm": observed_mean_harm,
        "cpu_reference_mean_harm": cpu_reference_mean_harm,
        "total_revenue_cents": result.total_revenue_cents,
        "opportunity_cost_burden": (
            float(result.harm.component_scores[:, HarmComponent.OC].mean())
            if result.player_ids.size
            else 0.0
        ),
        "minimum_public_contribution_cents": (
            result.epgc.minimum_public_contribution_cents
            if result.epgc is not None
            else 0
        ),
    }


def _process_sensitivity_unit(
    unit: SensitivityWorkUnit,
) -> dict[str, object]:
    if _PROCESS_SENSITIVITY_CONTEXT is None or _PROCESS_BACKEND is None:
        raise RuntimeError("spawned sensitivity worker was not initialized")
    return _compute_sensitivity_payload(
        _PROCESS_SENSITIVITY_CONTEXT,
        _PROCESS_BACKEND,
        unit,
    )


def run_checkpointed_policy_batch(
    spec: PolicyBatchSpec,
    *,
    store: ResumableCheckpointStore,
    execution_config: ExploratoryExecutionEngineConfig,
    backend: ResolvedExecutionBackend,
    country_profiles: Sequence[CountryProfile] | None = None,
    profile_bundle: ProfileBundle | None = None,
    harm_parameters: HarmModelParameters | None = None,
    harm_weights: WelfareHarmWeights | None = None,
    opportunity_valuation: OpportunityCostValuation | None = None,
    producer_assumptions: ProducerAssumptions | None = None,
    epgc_policy: EPGCPolicy | None = None,
    population_adapter: PopulationProjectionAdapter,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> CheckpointedPolicyBatch:
    """Execute only missing seeds and atomically retain each complete seed."""

    if type(execution_config) is not ExploratoryExecutionEngineConfig:
        raise TypeError(
            "execution_config must be ExploratoryExecutionEngineConfig"
        )
    if type(store) is not ResumableCheckpointStore:
        raise TypeError("store must be ResumableCheckpointStore")
    adapter = verify_population_projection_adapter(population_adapter)
    profiles, profile_lineage = resolve_profile_inputs(
        country_profiles=country_profiles,
        profile_bundle=profile_bundle,
    )
    run_inputs = resolve_policy_run_inputs(
        harm_parameters=harm_parameters,
        harm_weights=harm_weights,
        opportunity_valuation=opportunity_valuation,
        producer_assumptions=producer_assumptions,
        epgc_policy=epgc_policy,
    )
    _validate_backend_store_contract(store, backend, execution_config)
    parity = _preflight_backend_parity(run_inputs, backend)
    process_context = _PolicyProcessContext(
        spec=spec,
        country_profiles=profiles,
        run_inputs=run_inputs,
        population_adapter=adapter,
        runtime=_process_worker_contract(store, backend),
    )

    def compute(seed: int) -> dict[str, object]:
        return _compute_main_seed_payload(
            process_context,
            backend,
            seed,
        )

    def begin(seed: int) -> None:
        store.begin_main_seed(seed)
        _notify_progress(store, progress_callback)

    def commit(seed: int, payload: dict[str, object]) -> None:
        store.commit_main_seed(seed, payload)
        _notify_progress(store, progress_callback)

    _bounded_execute(
        store.remaining_main_seeds,
        compute=compute,
        begin=begin,
        commit=commit,
        worker_count=_effective_worker_count(execution_config, backend),
        max_in_flight=_effective_in_flight(execution_config, backend),
        process_compute=_process_main_seed,
        process_initializer=_initialize_policy_process_worker,
        process_initargs=(process_context,),
    )
    return CheckpointedPolicyBatch(
        spec=spec,
        store=store,
        run_inputs=run_inputs,
        country_profiles=profiles,
        profile_input_lineage=profile_lineage,
        population_adapter=adapter,
        backend_metadata={
            **backend.metadata.identity_payload(),
            "backend_identity_sha256": (
                backend.metadata.backend_identity_sha256
            ),
            "checkpoint_backend_contract": store.identity.backend.snapshot(),
            "checkpoint_backend_identity_sha256": (
                store.identity.backend.identity_sha256
            ),
        },
        backend_parity=asdict(parity) | {"passed": parity.passed},
    )


def run_checkpointed_sensitivity(
    batch: CheckpointedPolicyBatch,
    *,
    execution_config: ExploratoryExecutionEngineConfig,
    backend: ResolvedExecutionBackend,
    cases: Sequence[SensitivityCase] | None = None,
    instability_cv_threshold: float = 0.35,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> CheckpointedSensitivityResult:
    """Execute and resume one exact ``(parameter, level, seed)`` per unit."""

    selected = _validated_cases(
        tuple(cases) if cases is not None else default_sensitivity_cases()
    )
    threshold = _validated_instability_threshold(instability_cv_threshold)
    declared = sensitivity_work_units(batch.spec, selected)
    if batch.store.work_plan.sensitivity_units != declared:
        raise ValueError("checkpoint sensitivity work plan differs from design")
    _validate_backend_store_contract(batch.store, backend, execution_config)
    process_context = _SensitivityProcessContext(
        policy=_PolicyProcessContext(
            spec=batch.spec,
            country_profiles=batch.country_profiles,
            run_inputs=batch.run_inputs,
            population_adapter=batch.population_adapter,
            runtime=_process_worker_contract(batch.store, backend),
        ),
        cases=selected,
    )

    def compute(unit: SensitivityWorkUnit) -> dict[str, object]:
        return _compute_sensitivity_payload(
            process_context,
            backend,
            unit,
        )

    def begin(unit: SensitivityWorkUnit) -> None:
        batch.store.begin_sensitivity(
            unit.parameter_id,
            unit.level_id,
            unit.seed,
        )
        _notify_progress(batch.store, progress_callback)

    def commit(
        unit: SensitivityWorkUnit,
        payload: dict[str, object],
    ) -> None:
        batch.store.commit_sensitivity(
            unit.parameter_id,
            unit.level_id,
            unit.seed,
            payload,
        )
        _notify_progress(batch.store, progress_callback)

    _bounded_execute(
        batch.store.remaining_sensitivity_units,
        compute=compute,
        begin=begin,
        commit=commit,
        worker_count=_effective_worker_count(execution_config, backend),
        max_in_flight=_effective_in_flight(execution_config, backend),
        process_compute=_process_sensitivity_unit,
        process_initializer=_initialize_sensitivity_process_worker,
        process_initargs=(process_context,),
    )
    (
        rows,
        unstable,
        cpu_reference_rows,
        cpu_reference_unstable,
    ) = _assemble_sensitivity_rows(
        batch,
        selected,
        threshold,
    )
    conclusions_match = (
        unstable == cpu_reference_unstable
        and all(
            bool(observed["monotonic_observed"])
            == bool(reference["monotonic_observed"])
            and bool(observed["unstable"]) == bool(reference["unstable"])
            for observed, reference in zip(
                rows, cpu_reference_rows, strict=True
            )
        )
    )
    maximum_difference = max(
        (
            abs(float(observed["mean_harm"]) - float(reference["mean_harm"]))
            for observed, reference in zip(
                rows, cpu_reference_rows, strict=True
            )
        ),
        default=0.0,
    )
    if not conclusions_match:
        raise ValueError(
            "accelerated sensitivity changes instability or monotonicity conclusions"
        )
    return CheckpointedSensitivityResult(
        batch_spec=batch.spec,
        cases=selected,
        instability_cv_threshold=threshold,
        run_inputs=batch.run_inputs,
        rows=rows,
        unstable_parameters=unstable,
        cpu_reference_rows=cpu_reference_rows,
        cpu_reference_unstable_parameters=cpu_reference_unstable,
        conclusions_match_cpu_reference=conclusions_match,
        maximum_absolute_harm_difference=maximum_difference,
        country_profiles=batch.country_profiles,
        profile_input_lineage=batch.profile_input_lineage,
        population_lineage_sha256=None,
    )


def _assemble_sensitivity_rows(
    batch: CheckpointedPolicyBatch,
    cases: tuple[SensitivityCase, ...],
    threshold: float,
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[str, ...],
    tuple[Mapping[str, object], ...],
    tuple[str, ...],
]:
    rows: list[dict[str, object]] = []
    cpu_rows: list[dict[str, object]] = []
    unstable: set[str] = set()
    cpu_unstable: set[str] = set()
    cohort_by_seed: dict[int, str] = {}
    case_level_rows: list[
        tuple[
            SensitivityCase,
            list[dict[str, object]],
            list[tuple[float, float]],
            list[dict[str, object]],
            list[tuple[float, float]],
        ]
    ] = []
    for case in cases:
        level_rows: list[dict[str, object]] = []
        level_metrics: list[tuple[float, float]] = []
        cpu_level_rows: list[dict[str, object]] = []
        cpu_level_metrics: list[tuple[float, float]] = []
        for level_index, value in enumerate(case.values):
            payloads = [
                _sensitivity_payload(
                    batch.store.load_sensitivity_payload(
                        case.parameter,
                        _level_id(value),
                        seed,
                    ),
                    case=case,
                    level_index=level_index,
                    seed=seed,
                )
                for seed in batch.spec.seeds
            ]
            for payload in payloads:
                seed = int(payload["seed"])
                digest = str(payload["cohort_digest"])
                prior = cohort_by_seed.setdefault(seed, digest)
                if prior != digest:
                    raise ValueError(
                        "sensitivity units do not share one cohort per seed"
                    )
            harm_stats = _stats([float(item["mean_harm"]) for item in payloads])
            cpu_harm_stats = _stats(
                [float(item["cpu_reference_mean_harm"]) for item in payloads]
            )
            revenue_stats = _stats(
                [float(item["total_revenue_cents"]) for item in payloads]
            )
            opportunity_stats = _stats(
                [float(item["opportunity_cost_burden"]) for item in payloads]
            )
            subsidy_stats = _stats(
                [
                    float(item["minimum_public_contribution_cents"])
                    for item in payloads
                ]
            )
            coefficient = (
                harm_stats[2] / abs(harm_stats[0])
                if abs(harm_stats[0]) > _CV_ZERO_MEAN_TOLERANCE
                else (0.0 if harm_stats[2] == 0.0 else float("inf"))
            )
            cpu_coefficient = (
                cpu_harm_stats[2] / abs(cpu_harm_stats[0])
                if abs(cpu_harm_stats[0]) > _CV_ZERO_MEAN_TOLERANCE
                else (0.0 if cpu_harm_stats[2] == 0.0 else float("inf"))
            )
            level_metrics.append((value, harm_stats[0]))
            cpu_level_metrics.append((value, cpu_harm_stats[0]))
            shared = {
                "parameter": case.parameter,
                "parameter_value": value,
                "scenario_id": case.scenario_id.value,
                "seed_count": len(batch.spec.seeds),
                "total_revenue_cents": revenue_stats[0],
                "opportunity_cost_burden": opportunity_stats[0],
                "minimum_public_contribution_cents": subsidy_stats[0],
                "expected_direction": case.expected_direction,
            }
            level_rows.append(
                {
                    **shared,
                    "mean_harm": harm_stats[0],
                    "harm_variance": harm_stats[1],
                    "harm_sd": harm_stats[2],
                    "harm_ci95_low": harm_stats[3],
                    "harm_ci95_high": harm_stats[4],
                    "harm_coefficient_of_variation": coefficient,
                }
            )
            cpu_level_rows.append(
                {
                    **shared,
                    "mean_harm": cpu_harm_stats[0],
                    "harm_variance": cpu_harm_stats[1],
                    "harm_sd": cpu_harm_stats[2],
                    "harm_ci95_low": cpu_harm_stats[3],
                    "harm_ci95_high": cpu_harm_stats[4],
                    "harm_coefficient_of_variation": cpu_coefficient,
                }
            )
            if coefficient > threshold:
                unstable.add(case.parameter)
            if cpu_coefficient > threshold:
                cpu_unstable.add(case.parameter)
        case_level_rows.append(
            (
                case,
                level_rows,
                level_metrics,
                cpu_level_rows,
                cpu_level_metrics,
            )
        )
    for (
        case,
        level_rows,
        level_metrics,
        cpu_level_rows,
        cpu_level_metrics,
    ) in case_level_rows:
        monotonic = _monotonic(level_metrics, case.expected_direction)
        cpu_monotonic = _monotonic(
            cpu_level_metrics, case.expected_direction
        )
        if case.expected_direction != "none" and not monotonic:
            unstable.add(case.parameter)
        if case.expected_direction != "none" and not cpu_monotonic:
            cpu_unstable.add(case.parameter)
        for row in level_rows:
            row["monotonic_expected"] = case.expected_direction != "none"
            row["monotonic_observed"] = monotonic
            row["unstable"] = case.parameter in unstable
            rows.append(row)
        for row in cpu_level_rows:
            row["monotonic_expected"] = case.expected_direction != "none"
            row["monotonic_observed"] = cpu_monotonic
            row["unstable"] = case.parameter in cpu_unstable
            cpu_rows.append(row)
    return (
        tuple(MappingProxyType(row) for row in rows),
        tuple(sorted(unstable)),
        tuple(MappingProxyType(row) for row in cpu_rows),
        tuple(sorted(cpu_unstable)),
    )


def _sensitivity_payload(
    value: object,
    *,
    case: SensitivityCase,
    level_index: int,
    seed: int,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint sensitivity payload is not an object")
    expected = {
        "schema_version": "microtx_sim.sensitivity_unit.v2",
        "parameter": case.parameter,
        "level_id": _level_id(case.values[level_index]),
        "parameter_value": case.values[level_index],
        "scenario_id": case.scenario_id.value,
        "seed": seed,
    }
    mismatches = [name for name, item in expected.items() if value.get(name) != item]
    if mismatches:
        raise ValueError(
            "checkpoint sensitivity identity mismatch: " + ", ".join(mismatches)
        )
    if value.get("monetary_semantics") != {
        "internal_unit": "simulation_cents",
        "interpretation": "INTERNAL_MODEL_UNIT_NOT_REAL_MONEY",
        "observed_real_world_spending": False,
        "raw_cross_country_pooling": "PROHIBITED",
    }:
        raise ValueError(
            "checkpoint sensitivity monetary semantics are missing or changed"
        )
    for name in (
        "cohort_digest",
        "mean_harm",
        "cpu_reference_mean_harm",
        "total_revenue_cents",
        "opportunity_cost_burden",
        "minimum_public_contribution_cents",
    ):
        if name not in value:
            raise ValueError(f"checkpoint sensitivity payload lacks {name}")
    return value


def _bounded_execute(
    items: Sequence[_WorkItem],
    *,
    compute: Callable[[_WorkItem], _WorkResult],
    begin: Callable[[_WorkItem], None],
    commit: Callable[[_WorkItem, _WorkResult], None],
    worker_count: int,
    max_in_flight: int,
    process_compute: Callable[[_WorkItem], _WorkResult] | None = None,
    process_initializer: Callable[..., None] | None = None,
    process_initargs: tuple[object, ...] = (),
) -> None:
    for value, name in (
        (worker_count, "worker_count"),
        (max_in_flight, "max_in_flight"),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if max_in_flight > worker_count:
        raise ValueError("max_in_flight cannot exceed worker_count")
    pending = tuple(items)
    if len(set(pending)) != len(pending):
        raise ValueError("scheduler work items must be unique")
    declared_order = {item: index for index, item in enumerate(pending)}
    if not pending:
        return
    if worker_count == 1:
        for item in pending:
            begin(item)
            commit(item, compute(item))
        return
    if process_compute is None or process_initializer is None:
        raise ValueError(
            "parallel execution requires explicit pickle-safe process worker "
            "and initializer"
        )
    iterator = iter(enumerate(pending))
    futures: dict[Future[_WorkResult], tuple[int, _WorkItem]] = {}
    ready: dict[int, tuple[_WorkItem, _WorkResult]] = {}
    next_commit_index = 0
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp.get_context("spawn"),
        initializer=process_initializer,
        initargs=process_initargs,
    ) as pool:
        for _ in range(min(max_in_flight, len(pending))):
            index, item = next(iterator)
            begin(item)
            futures[pool.submit(process_compute, item)] = (index, item)
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            ordered_done = sorted(
                done,
                key=lambda future: declared_order[futures[future][1]],
            )
            for future in ordered_done:
                index, item = futures.pop(future)
                ready[index] = (item, future.result())
            while next_commit_index in ready:
                item, result = ready.pop(next_commit_index)
                commit(item, result)
                next_commit_index += 1
                try:
                    following_index, following = next(iterator)
                except StopIteration:
                    continue
                begin(following)
                futures[pool.submit(process_compute, following)] = (
                    following_index,
                    following,
                )
    if ready or next_commit_index != len(pending):
        raise RuntimeError("process scheduler did not commit every work item")


def _effective_worker_count(
    config: ExploratoryExecutionEngineConfig,
    backend: ResolvedExecutionBackend,
) -> int:
    if backend.mode is BackendMode.GPU:
        return 1
    memory_bound = config.memory_limit_mb // config.estimated_worker_memory_mb
    return max(1, min(config.host_workers, memory_bound))


def _effective_in_flight(
    config: ExploratoryExecutionEngineConfig,
    backend: ResolvedExecutionBackend,
) -> int:
    return min(
        config.max_in_flight_units,
        _effective_worker_count(config, backend),
    )


def _validate_backend_store_contract(
    store: ResumableCheckpointStore,
    backend: ResolvedExecutionBackend,
    config: ExploratoryExecutionEngineConfig,
) -> None:
    identity = store.identity.backend
    expected_workers = _effective_worker_count(config, backend)
    native_threads = enforce_numpy_native_thread_limit(1)
    observed = {
        "requested_backend": backend.metadata.requested_mode,
        "resolved_backend": backend.metadata.resolved_mode,
        "library": backend.metadata.implementation,
        "library_version": backend.metadata.implementation_version,
        "device_name": backend.metadata.device_name,
        "precision_mode": backend.metadata.precision_mode,
        "worker_count": expected_workers,
        "batch_size": backend.metadata.batch_size,
        "scheduling_policy": config.scheduling_policy,
        "native_thread_runtime": native_threads.runtime,
        "native_thread_library_path": native_threads.library_path,
        "native_thread_library_sha256": native_threads.library_sha256,
        "native_thread_getter_symbol": native_threads.getter_symbol,
        "native_thread_setter_symbol": native_threads.setter_symbol,
        "native_thread_limit": native_threads.enforced_thread_count,
    }
    mismatches = [
        name for name, value in observed.items() if getattr(identity, name) != value
    ]
    if mismatches:
        raise ValueError(
            "checkpoint backend identity differs from runtime: "
            + ", ".join(mismatches)
        )


def _notify_progress(
    store: ResumableCheckpointStore,
    callback: Callable[[Mapping[str, object]], None] | None,
) -> None:
    snapshot = store.progress_snapshot
    if callback is None:
        print(format_progress(snapshot), flush=True)
    else:
        callback(snapshot)


def _preflight_backend_parity(
    run_inputs: PolicyRunInputs,
    backend: ResolvedExecutionBackend,
):
    scores = np.linspace(0.0, 1.0, 6 * 257, dtype=np.float64).reshape(257, 6)
    report = validate_composite_harm_parity(
        scores,
        run_inputs.harm_weights.as_array(),
        backend=backend,
        categorical_thresholds=(0.35,),
        mean_direction_reference=0.35,
        raise_on_failure=True,
    )
    if backend.mode is BackendMode.CPU and not report.bitwise_equal:
        raise RuntimeError("CPU backend failed bitwise reference parity")
    return report


def format_progress(snapshot: Mapping[str, object]) -> str:
    overall = snapshot["overall"]
    main = snapshot["main_batch"]
    sensitivity = snapshot["sensitivity"]
    if not all(isinstance(item, Mapping) for item in (overall, main, sensitivity)):
        raise TypeError("progress sections must be mappings")
    return (
        f"[{snapshot['current_phase']}] overall="
        f"{overall['percentage_display']} "
        f"({overall['completed_units']}/{overall['total_units']}) "
        f"main={main['percentage_display']} "
        f"sensitivity={sensitivity['percentage_display']} "
        f"parameter={snapshot['current_parameter']} "
        f"level={snapshot['current_level']} seed={snapshot['current_seed']}"
    )


def _level_id(value: float) -> str:
    return canonical_level_id(value)


def _level_index(case: SensitivityCase, level_id: str) -> int:
    matches = [
        index
        for index, value in enumerate(case.values)
        if canonical_level_id(value) == level_id
    ]
    if len(matches) != 1:
        raise ValueError("sensitivity level identity is invalid or ambiguous")
    return matches[0]


__all__ = [
    "CheckpointedPolicyBatch",
    "CheckpointedSensitivityResult",
    "expected_execution_work_plan",
    "format_progress",
    "run_checkpointed_policy_batch",
    "run_checkpointed_sensitivity",
    "sensitivity_work_units",
]
