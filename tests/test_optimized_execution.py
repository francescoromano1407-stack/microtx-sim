from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_binding import _plan as _analysis_plan  # noqa: E402
from test_population_projection_adapter import _complete_adapter  # noqa: E402

from microtx_sim.causal.analysis_binding import (  # noqa: E402
    resolve_run_analysis_binding,
)
from microtx_sim.analysis.sensitivity import run_sensitivity_analysis  # noqa: E402
from microtx_sim.causal.batch import (  # noqa: E402
    PolicyBatchCheckpoint,
    PolicyBatchSpec,
    resolve_policy_run_inputs,
    run_policy_batch,
)
from microtx_sim.consumers.decision import DecisionParameters  # noqa: E402
from microtx_sim.consumers.population import CountryProfile  # noqa: E402
from microtx_sim.execution.backends import (  # noqa: E402
    ExecutionBackendConfig,
    resolve_execution_backend,
)
from microtx_sim.execution.checkpoints import (  # noqa: E402
    BackendIdentity,
    ExecutionIdentity,
    ExecutionWorkPlan,
    ResumableCheckpointStore,
    RuntimeIdentity,
    canonical_sha256,
)
from microtx_sim.execution.native_threads import (  # noqa: E402
    enforce_numpy_native_thread_limit,
)
from microtx_sim.execution.optimized_runner import (  # noqa: E402
    CheckpointedPolicyBatch,
    _bounded_execute,
    expected_execution_work_plan,
    run_checkpointed_policy_batch,
    run_checkpointed_sensitivity,
)
from microtx_sim.execution.result_codec import (  # noqa: E402
    ResultCheckpointCodecError,
    decode_seed_scenario_record,
    encode_seed_scenario_record,
)
from microtx_sim.execution.streaming_analysis import (  # noqa: E402
    _CanonicalArrayHash,
    resolve_checkpointed_run_analysis_binding,
)
from microtx_sim.policy_config import (  # noqa: E402
    ExploratoryExecutionEngineConfig,
    load_policy_config,
)
from microtx_sim.analysis.uncertainty import (  # noqa: E402
    evaluate_blockwise_convergence,
    final_sufficiency_judgment,
    summarize_seed_uncertainty,
)
from microtx_sim.outputs.exploratory_results import (  # noqa: E402
    _checkpointed_execution_metadata,
    _checkpointed_primary_realizations,
    _convergence_rule,
    _end_to_end_backend_parity,
    _exploratory_uncertainty_components,
)


_PROFILE = (CountryProfile(code="UK"),)


def _canonical_projection(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _runtime_identity() -> RuntimeIdentity:
    return RuntimeIdentity(
        python_implementation="CPython",
        python_version="fixture",
        python_executable="C:/runtime/python.exe",
        python_executable_sha256="1" * 64,
        dependency_lock_sha256="2" * 64,
        installed_dependencies_sha256="3" * 64,
        operating_system="Windows",
        os_release="fixture",
        machine_architecture="AMD64",
        processor="fixture",
    )


def _backend_identity(*, worker_count: int = 2) -> BackendIdentity:
    backend = resolve_execution_backend(ExecutionBackendConfig(mode="cpu"))
    native = enforce_numpy_native_thread_limit(1)
    return BackendIdentity(
        requested_backend=backend.metadata.requested_mode,
        resolved_backend=backend.metadata.resolved_mode,
        library=backend.metadata.implementation,
        library_version=backend.metadata.implementation_version,
        device_name=backend.metadata.device_name,
        device_id="cpu",
        precision_mode=backend.metadata.precision_mode,
        worker_count=worker_count,
        batch_size=backend.metadata.batch_size,
        scheduling_policy="ONE_SEED_OWNS_COMMON_COHORT_AND_ALL_SCENARIOS",
        native_thread_runtime=native.runtime,
        native_thread_library_path=native.library_path,
        native_thread_library_sha256=native.library_sha256,
        native_thread_getter_symbol=native.getter_symbol,
        native_thread_setter_symbol=native.setter_symbol,
        native_thread_limit=native.enforced_thread_count,
    )


def _identity(
    work_plan: ExecutionWorkPlan,
    *,
    worker_count: int = 2,
) -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="fixture.optimized.run.v2",
        attempt_id="attempt-000002",
        implementation_id="fixture.optimized.implementation.v2",
        source_tree_sha256="4" * 64,
        git_commit="a" * 40,
        git_branch="codex/accelerated-resumable-exploratory",
        configuration_sha256="5" * 64,
        analysis_plan_id="fixture.analysis.plan.v1",
        analysis_plan_sha256="6" * 64,
        seed_set_sha256=work_plan.seed_set_sha256,
        work_plan_sha256=work_plan.identity_sha256,
        backend=_backend_identity(worker_count=worker_count),
        runtime=_runtime_identity(),
        payload_schema_id=(
            "microtx_sim.policy_and_sensitivity_checkpoint_payload.v2"
        ),
    )


def _engine_config(
    root: Path,
    *,
    worker_count: int = 2,
) -> ExploratoryExecutionEngineConfig:
    return ExploratoryExecutionEngineConfig(
        implementation_id="fixture.optimized.implementation.v2",
        run_id="fixture.optimized.run.v2",
        attempt_id="attempt-000002",
        supersedes_attempt_id="attempt-000001",
        previous_attempt_lineage_path=root / "lineage.json",
        backend="cpu",
        gpu_device_index=0,
        gpu_batch_size=65_536,
        gpu_max_batch_bytes=256 * 1024 * 1024,
        gpu_memory_fraction=0.5,
        precision_mode="FLOAT64_STRICT_INTEGER_EXACT",
        host_executor="BOUNDED_PROCESS_POOL_SPAWN",
        host_workers=worker_count,
        max_in_flight_units=worker_count,
        memory_limit_mb=2048,
        estimated_worker_memory_mb=512,
        native_threads_per_worker=1,
        scheduling_policy="ONE_SEED_OWNS_COMMON_COHORT_AND_ALL_SCENARIOS",
        resume_enabled=True,
        checkpoint_schema_version="microtx_sim.resumable_checkpoint.v2",
        main_checkpoint_granularity="COMPLETE_SEED_ALL_SCENARIOS",
        sensitivity_checkpoint_granularity="PARAMETER_LEVEL_SEED",
        progress_schema_version="microtx_sim.execution_progress.v2",
    )


def test_streaming_canonical_array_hash_is_exact_for_every_key_position() -> None:
    for key in ("a", "m", "z"):
        payload: dict[str, object] = {"a": 1, "m": "middle", "z": False}
        payload[key] = None
        values = ({"seed": 101}, {"seed": 303, "nested": [1, 2]})
        stream = _CanonicalArrayHash(payload, array_key=key)
        for value in values:
            stream.append(value)
        expected = dict(payload)
        expected[key] = list(values)
        assert stream.finish() == canonical_sha256(expected)


def test_result_checkpoint_codec_is_lossless_and_rejects_corruption() -> None:
    spec = PolicyBatchSpec(
        seeds=(17,),
        days=2,
        player_count=32,
        decision_parameters=DecisionParameters(step_minutes=240),
    )
    batch = run_policy_batch(spec, country_profiles=_PROFILE)
    for original in batch.records:
        payload = encode_seed_scenario_record(original, batch_spec=spec)
        assert payload["monetary_semantics"] == {
            "internal_unit": "simulation_cents",
            "interpretation": "INTERNAL_MODEL_UNIT_NOT_REAL_MONEY",
            "observed_real_world_spending": False,
            "raw_cross_country_pooling": "PROHIBITED",
        }
        decoded = decode_seed_scenario_record(payload, batch_spec=spec)
        assert decoded.cohort_digest == original.cohort_digest
        assert decoded.mean_harm_effect_vs_safe == original.mean_harm_effect_vs_safe
        assert decoded.result.scenario == original.result.scenario
        assert decoded.result.seed == original.result.seed
        assert decoded.result.days == original.result.days
        assert decoded.result.revenue_composition_cents == (
            original.result.revenue_composition_cents
        )
        assert decoded.result.total_revenue_cents == (
            original.result.total_revenue_cents
        )
        assert decoded.result.producer_cost_cents == (
            original.result.producer_cost_cents
        )
        assert decoded.result.producer_profit_cents == (
            original.result.producer_profit_cents
        )
        assert decoded.result.epgc == original.result.epgc
        for name in (
            "player_ids",
            "spending_cents",
            "composite_harm",
            "high_risk",
            "action_minutes",
        ):
            assert np.array_equal(
                getattr(decoded.result, name),
                getattr(original.result, name),
            )

    corrupted = encode_seed_scenario_record(batch.records[0], batch_spec=spec)
    corrupted["arrays"]["spending_cents"]["raw_sha256"] = "0" * 64
    with pytest.raises(ResultCheckpointCodecError, match="checksum"):
        decode_seed_scenario_record(corrupted, batch_spec=spec)


def test_parallel_scheduler_requires_explicit_pickle_safe_process_worker() -> None:
    spec = PolicyBatchSpec(
        seeds=(17, 29, 41),
        days=2,
        player_count=96,
        decision_parameters=DecisionParameters(step_minutes=240),
    )
    backend = resolve_execution_backend(ExecutionBackendConfig(mode="cpu"))
    enforce_numpy_native_thread_limit(1)

    def compute(seed: int) -> bytes:
        batch = run_policy_batch(
            replace(spec, seeds=(seed,)),
            country_profiles=_PROFILE,
            execution_backend=backend,
        )
        return _canonical_projection(
            {
                "seed_rows": batch.seed_rows(),
                "scenario_rows": batch.scenario_rows(),
                "cohort": dict(batch.cohort_digest_by_seed),
            }
        )

    committed: dict[int, bytes] = {}
    begun: list[int] = []
    with pytest.raises(ValueError, match="pickle-safe process worker"):
        _bounded_execute(
            spec.seeds,
            compute=compute,
            begin=begun.append,
            commit=lambda seed, value: committed.__setitem__(seed, value),
            worker_count=2,
            max_in_flight=2,
        )
    assert committed == {}
    assert begun == []


def test_checkpointed_streaming_analysis_preserves_reference_binding_hashes(
    tmp_path: Path,
) -> None:
    population_root = tmp_path / "population"
    population_root.mkdir()
    _verification, _apportionment, _path, _mapping, adapter = _complete_adapter(
        population_root
    )
    spec = PolicyBatchSpec(
        seeds=(17, 29),
        days=1,
        player_count=12,
        decision_parameters=DecisionParameters(step_minutes=240),
    )
    inputs = resolve_policy_run_inputs()
    reference = run_policy_batch(
        spec,
        country_profiles=_PROFILE,
        harm_parameters=inputs.harm_parameters,
        harm_weights=inputs.harm_weights,
        opportunity_valuation=inputs.opportunity_valuation,
        producer_assumptions=inputs.producer_assumptions,
        epgc_policy=inputs.epgc_policy,
        population_adapter=adapter,
    )
    plan = _analysis_plan(spec, inputs, adapter)
    reference_binding = resolve_run_analysis_binding(plan, reference)
    work_plan = ExecutionWorkPlan.build(
        seeds=spec.seeds,
        scenario_ids=tuple(
            scenario.scenario_id.value for scenario in spec.scenarios
        ),
    )
    identity = _identity(work_plan)
    store = ResumableCheckpointStore.create(
        tmp_path / "progress",
        identity=identity,
        work_plan=work_plan,
    )
    for index, seed in enumerate(spec.seeds):
        records = [
            record for record in reference.records if record.result.seed == seed
        ]
        store.begin_main_seed(seed)
        store.commit_main_seed(
            seed,
            {
                record.result.scenario.scenario_id.value: (
                    encode_seed_scenario_record(record, batch_spec=spec)
                )
                for record in records
            },
        )
        if index == 0:
            store.begin_main_seed(spec.seeds[1])
            store.mark_interrupted("bounded injected interruption")
            store = ResumableCheckpointStore.resume(
                store.attempt_dir,
                expected_identity=identity,
                expected_work_plan=work_plan,
            )
            assert store.completed_main_seeds == (spec.seeds[0],)
            assert store.remaining_main_seeds == (spec.seeds[1],)
    assert reference.profile_input_lineage is not None
    batch = CheckpointedPolicyBatch(
        spec=spec,
        store=store,
        run_inputs=inputs,
        country_profiles=_PROFILE,
        profile_input_lineage=reference.profile_input_lineage,
        population_adapter=adapter,
        backend_metadata={},
        backend_parity={},
    )
    observed = resolve_checkpointed_run_analysis_binding(plan, batch)
    assert observed.binding_sha256 == reference_binding.binding_sha256
    assert (
        observed.population_lineage_sha256
        == reference_binding.population_lineage_sha256
    )
    assert [item.binding_sha256 for item in observed.seed_bindings] == [
        item.binding_sha256 for item in reference_binding.seed_bindings
    ]
    assert [item.result for item in observed.seed_bindings] == [
        item.result for item in reference_binding.seed_bindings
    ]
    expected_diagnostics: list[dict[str, object]] = []
    for index, seed in enumerate(spec.seeds, start=1):
        prefix = tuple(
            record
            for record in reference.records
            if record.result.seed in spec.seeds[:index]
        )
        checkpoint = PolicyBatchCheckpoint(
            spec=spec,
            completed_seeds=spec.seeds[:index],
            records=prefix,
            cohort_digest_by_seed={
                selected: reference.cohort_digest_by_seed[selected]
                for selected in spec.seeds[:index]
            },
        )
        expected_diagnostics.extend(
            row
            for row in checkpoint.nonmonetary_diagnostic_rows()
            if row["seed"] == seed
        )
    assert observed.scenario_diagnostic_rows == tuple(expected_diagnostics)
    repository_config = load_policy_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "policy_exploratory_synthetic.toml"
    )
    realizations = _checkpointed_primary_realizations(
        repository_config,
        batch,
        observed.seed_bindings,
    )
    summary = summarize_seed_uncertainty(
        realizations,
        expected_seeds=spec.seeds,
    )
    components = _exploratory_uncertainty_components(summary)
    rule = _convergence_rule(repository_config)
    convergence = evaluate_blockwise_convergence(
        realizations,
        expected_seeds=spec.seeds,
        rule=rule,
        sensitivity_instability=False,
        required_components_available=False,
    )
    judgment = final_sufficiency_judgment(
        convergence_status=convergence[-1].status,
        components=components,
    )
    parity = _end_to_end_backend_parity(
        repository_config,
        batch,
        None,
        observed.seed_bindings,
        observed_checkpoints=convergence,
        observed_judgment=judgment,
        convergence_rule=rule,
    )
    assert parity["passed"] is True
    assert parity["primary_seed_values_bitwise_equal"] is True
    assert parity["convergence_decisions_equal"] is True
    assert parity["declared_conclusions_equal"] is True
    assert repository_config.ledger is not None
    execution_metadata = _checkpointed_execution_metadata(
        repository_config,
        batch,
        observed,
        end_to_end_parity=parity,
    )
    assert execution_metadata["scheduler"]["host_executor"] == (
        "BOUNDED_PROCESS_POOL_SPAWN"
    )
    assert execution_metadata["scheduler"]["process_start_method"] == "spawn"
    assert execution_metadata["scheduler"]["checkpoint_writer"] == (
        "COORDINATOR_ONLY"
    )
    assert execution_metadata["scheduler"]["commit_order"] == (
        "DECLARED_WORK_PLAN_ORDER"
    )
    assert execution_metadata["ledger_contract"] == {
        "backend": "sqlite",
        "path": repository_config.ledger.path.as_posix(),
        "persistent": True,
        "temporary": False,
        "execution_role": (
            "DECLARED_PERSISTENT_CONFIG_CONTRACT; "
            "RESUMABLE_RESULTS_ARE_AUTHORITATIVE_CHECKPOINT_BLOCKS"
        ),
    }
    incompatible = (
        replace(
            observed.seed_bindings[0],
            estimand_direction_matches_cpu_reference=False,
        ),
        *observed.seed_bindings[1:],
    )
    with pytest.raises(ValueError, match="changes a primary"):
        _end_to_end_backend_parity(
            repository_config,
            batch,
            None,
            incompatible,
            observed_checkpoints=convergence,
            observed_judgment=judgment,
            convergence_rule=rule,
        )


def test_checkpointed_parallel_main_and_sensitivity_match_cpu_reference(
    tmp_path: Path,
) -> None:
    population_root = tmp_path / "population"
    population_root.mkdir()
    _verification, _apportionment, _path, _mapping, adapter = _complete_adapter(
        population_root
    )
    spec = PolicyBatchSpec(
        seeds=(17, 29),
        days=1,
        player_count=12,
        decision_parameters=DecisionParameters(step_minutes=240),
    )
    engine = _engine_config(tmp_path)
    backend = resolve_execution_backend(
        ExecutionBackendConfig(
            mode="cpu",
            batch_size=engine.gpu_batch_size,
            max_batch_bytes=engine.gpu_max_batch_bytes,
            gpu_memory_fraction=engine.gpu_memory_fraction,
            precision_mode=engine.precision_mode,
        )
    )
    work_plan = expected_execution_work_plan(
        spec,
        sensitivity_enabled=True,
    )
    store = ResumableCheckpointStore.create(
        tmp_path / "progress",
        identity=_identity(work_plan),
        work_plan=work_plan,
    )
    progress: list[dict[str, object]] = []
    checkpointed = run_checkpointed_policy_batch(
        spec,
        store=store,
        execution_config=engine,
        backend=backend,
        country_profiles=_PROFILE,
        population_adapter=adapter,
        progress_callback=lambda value: progress.append(dict(value)),
    )
    serial_engine = _engine_config(tmp_path, worker_count=1)
    serial_store = ResumableCheckpointStore.create(
        tmp_path / "progress-serial",
        identity=_identity(work_plan, worker_count=1),
        work_plan=work_plan,
    )
    serial_checkpointed = run_checkpointed_policy_batch(
        spec,
        store=serial_store,
        execution_config=serial_engine,
        backend=backend,
        country_profiles=_PROFILE,
        population_adapter=adapter,
        progress_callback=lambda _value: None,
    )
    for seed in spec.seeds:
        assert store.load_main_seed_payload(seed) == (
            serial_store.load_main_seed_payload(seed)
        )
    reference_batch = run_policy_batch(
        spec,
        country_profiles=_PROFILE,
        population_adapter=adapter,
    )
    checkpoint_rows = [
        row
        for seed_batch in checkpointed.iter_seed_batches()
        for row in seed_batch.seed_rows()
    ]
    assert checkpoint_rows == reference_batch.seed_rows()
    observed = run_checkpointed_sensitivity(
        checkpointed,
        execution_config=engine,
        backend=backend,
        progress_callback=lambda value: progress.append(dict(value)),
    )
    serial_observed = run_checkpointed_sensitivity(
        serial_checkpointed,
        execution_config=serial_engine,
        backend=backend,
        progress_callback=lambda _value: None,
    )
    assert [dict(row) for row in observed.rows] == [
        dict(row) for row in serial_observed.rows
    ]
    for unit in work_plan.sensitivity_units:
        assert store.load_sensitivity_payload(
            unit.parameter_id,
            unit.level_id,
            unit.seed,
        ) == serial_store.load_sensitivity_payload(
            unit.parameter_id,
            unit.level_id,
            unit.seed,
        )
    expected = run_sensitivity_analysis(
        spec,
        country_profiles=_PROFILE,
        population_adapter=adapter,
        execution_backend=backend,
    )
    assert [dict(row) for row in observed.rows] == [
        dict(row) for row in expected.rows
    ]
    assert [dict(row) for row in observed.cpu_reference_rows] == [
        dict(row) for row in expected.rows
    ]
    assert observed.unstable_parameters == expected.unstable_parameters
    assert (
        observed.cpu_reference_unstable_parameters
        == expected.unstable_parameters
    )
    assert observed.conclusions_match_cpu_reference
    assert observed.maximum_absolute_harm_difference == 0.0
    assert store.remaining_main_seeds == ()
    assert store.remaining_sensitivity_units == ()
    assert progress[0]["current_seed"] in spec.seeds
    assert progress[-1]["overall"]["percentage_display"] == "100.000000%"
    assert progress[-1]["main_batch"]["percentage_display"] == "100.000000%"
    assert progress[-1]["sensitivity"]["percentage_display"] == "100.000000%"
