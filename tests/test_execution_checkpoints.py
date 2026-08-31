from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import microtx_sim.execution.checkpoints as checkpoint_module
from microtx_sim.execution.checkpoints import (
    BackendIdentity,
    CheckpointCorruptError,
    CheckpointIncompleteError,
    DuplicateWorkUnitError,
    ExecutionIdentity,
    ExecutionLineage,
    ExecutionWorkPlan,
    IncompatibleCheckpointError,
    PriorExecutionLineage,
    ResumableCheckpointStore,
    RuntimeIdentity,
    SensitivityWorkUnit,
    canonical_level_id,
    format_console_progress,
    next_attempt_id,
)
from microtx_sim.execution.native_threads import NativeThreadAttestation


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        result = self.value
        self.value += timedelta(microseconds=1)
        return result


def _runtime() -> RuntimeIdentity:
    return RuntimeIdentity(
        python_implementation="CPython",
        python_version="3.13.7",
        python_executable="C:/runtime/python.exe",
        python_executable_sha256="1" * 64,
        dependency_lock_sha256="2" * 64,
        installed_dependencies_sha256="3" * 64,
        operating_system="Windows",
        os_release="fixture",
        machine_architecture="AMD64",
        processor="fixture-cpu",
    )


def _backend() -> BackendIdentity:
    return BackendIdentity(
        requested_backend="cpu",
        resolved_backend="cpu",
        library="numpy",
        library_version="2.3.2",
        device_name="fixture-host",
        device_id="cpu:0",
        precision_mode="float64",
        worker_count=2,
        batch_size=128,
        scheduling_policy="bounded_seed_blocks",
        native_thread_runtime="fixture-openblas",
        native_thread_library_path="C:/runtime/openblas.dll",
        native_thread_library_sha256="9" * 64,
        native_thread_getter_symbol="fixture_get_num_threads",
        native_thread_setter_symbol="fixture_set_num_threads",
        native_thread_limit=1,
    )


def _plan() -> ExecutionWorkPlan:
    scenarios = tuple(f"scenario_{index}" for index in range(7))
    sensitivity = (
        SensitivityWorkUnit("temperature", "low", 101),
        SensitivityWorkUnit("temperature", "high", 101),
    )
    return ExecutionWorkPlan.build(
        seeds=(101, 102),
        scenario_ids=scenarios,
        sensitivity_units=sensitivity,
    )


def _identity(
    plan: ExecutionWorkPlan,
    *,
    attempt_id: str = "attempt-000002",
    configuration_sha256: str = "5" * 64,
    backend: BackendIdentity | None = None,
) -> ExecutionIdentity:
    return ExecutionIdentity(
        run_id="exploratory.optimized.run.v2",
        attempt_id=attempt_id,
        implementation_id="execution.optimized.v2",
        source_tree_sha256="4" * 64,
        git_commit="a" * 40,
        git_branch="codex/accelerated-resumable-exploratory",
        configuration_sha256=configuration_sha256,
        analysis_plan_id="illustrative.exploratory.plan.v1",
        analysis_plan_sha256=(
            "5915bc42752dd77b984a67bdab5a79a040d99fc84f381de2d3aeb4c813bc2414"
        ),
        seed_set_sha256=plan.seed_set_sha256,
        work_plan_sha256=plan.identity_sha256,
        backend=backend or _backend(),
        runtime=_runtime(),
    )


def _lineage(identity: ExecutionIdentity) -> ExecutionLineage:
    return ExecutionLineage(
        previous=PriorExecutionLineage(
            run_id=None,
            run_identity_status="NOT_RECORDED_BY_CHECKPOINT_SCHEMA_V1",
            attempt_id="attempt-000001",
            configuration_sha256=(
                "10f0969f5fb005f3dd83507cc76a3faa46dd33aa6c0c0a9c937445c37f22089d"
            ),
            analysis_plan_sha256=(
                "5915bc42752dd77b984a67bdab5a79a040d99fc84f381de2d3aeb4c813bc2414"
            ),
            source_identity_status="NOT_RECORDED_BY_CHECKPOINT_SCHEMA_V1",
            source_tree_sha256=None,
            git_commit=None,
            observed_status="INTERRUPTED",
            reason_final_outputs_unavailable=(
                "The observed v1 progress record is INTERRUPTED and no final "
                "campaign outputs are present."
            ),
            progress_artifact_path=(
                "artifacts/policy_exploratory_synthetic/progress/"
                "attempt-000001/progress.json"
            ),
            progress_artifact_sha256="7" * 64,
        ),
        successor_run_id=identity.run_id,
        successor_attempt_id=identity.attempt_id,
        successor_implementation_id=identity.implementation_id,
    )


def _main_payloads(plan: ExecutionWorkPlan, seed: int) -> dict[str, object]:
    return {
        scenario_id: {
            "seed": seed,
            "scenario_id": scenario_id,
            "records": [{"player_id": 1, "harm": seed / 1000.0}],
        }
        for scenario_id in plan.scenario_ids
    }


class ResumableCheckpointStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "progress"
        self.root.mkdir()
        self.plan = _plan()
        self.identity = _identity(self.plan)
        self.lineage = _lineage(self.identity)
        self.clock = _Clock()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _create(self) -> ResumableCheckpointStore:
        return ResumableCheckpointStore.create(
            self.root,
            identity=self.identity,
            work_plan=self.plan,
            lineage=self.lineage,
            clock=self.clock,
        )

    def _load(self) -> ResumableCheckpointStore:
        return ResumableCheckpointStore.load(
            self.root / self.identity.attempt_id,
            expected_identity=self.identity,
            expected_work_plan=self.plan,
            expected_lineage=self.lineage,
            clock=self.clock,
        )

    def test_main_seed_commit_is_atomic_seven_units_and_payloads_round_trip(
        self,
    ) -> None:
        store = self._create()
        initial = store.progress_snapshot
        self.assertEqual(initial["main_batch"]["completed_units"], 0)
        self.assertEqual(initial["main_batch"]["total_units"], 14)

        store.begin_main_seed(101)
        in_progress = store.progress_snapshot
        self.assertEqual(in_progress["current_phase"], "main_batch")
        self.assertEqual(in_progress["current_seed"], 101)
        self.assertEqual(len(in_progress["in_progress_unit_ids"]), 7)
        payloads = _main_payloads(self.plan, 101)
        store.commit_main_seed(101, payloads)

        progress = store.progress_snapshot
        self.assertEqual(progress["main_batch"]["completed_units"], 7)
        self.assertEqual(progress["main_batch"]["percentage"], 50.0)
        self.assertEqual(
            progress["main_batch"]["percentage_exact"],
            {"numerator": 50, "denominator": 1, "unit": "percent"},
        )
        self.assertEqual(progress["overall"]["completed_units"], 7)
        self.assertEqual(
            progress["overall"]["percentage_exact"],
            {"numerator": 175, "denominator": 4, "unit": "percent"},
        )
        self.assertEqual(store.load_main_seed_payload(101), payloads)
        self.assertEqual(store.remaining_main_seeds, (102,))
        self.assertTrue(progress["payload_blocks"])

        loaded = self._load()
        self.assertEqual(loaded.load_main_seed_payload(101), payloads)
        self.assertEqual(loaded.completed_main_seeds, (101,))

    def test_sensitivity_unit_progress_is_exact_and_machine_readable(self) -> None:
        store = self._create()
        store.commit_main_seed(101, _main_payloads(self.plan, 101))
        store.commit_main_seed(102, _main_payloads(self.plan, 102))
        store.begin_sensitivity("temperature", "low", 101)
        current = store.progress_snapshot
        self.assertEqual(current["current_parameter"], "temperature")
        self.assertEqual(current["current_level"], "low")
        store.commit_sensitivity(
            "temperature", "low", 101, {"estimate": 0.125}
        )
        progress = store.progress_snapshot
        self.assertEqual(progress["sensitivity"]["completed_units"], 1)
        self.assertEqual(progress["sensitivity"]["percentage"], 50.0)
        self.assertEqual(progress["overall"]["completed_units"], 15)
        self.assertEqual(progress["overall"]["total_units"], 16)
        self.assertEqual(
            store.load_sensitivity_payload("temperature", "low", 101),
            {"estimate": 0.125},
        )
        on_disk = json.loads(store.progress_path.read_text("utf-8"))
        self.assertEqual(on_disk["overall"], progress["overall"])
        self.assertIn("15/16", format_console_progress(progress))

    def test_interruption_resume_preserves_complete_payload_and_requeues_active(
        self,
    ) -> None:
        store = self._create()
        first = _main_payloads(self.plan, 101)
        store.commit_main_seed(101, first)
        store.begin_main_seed(102)
        store.mark_interrupted("operator requested stop")

        resumed = ResumableCheckpointStore.resume(
            store.attempt_dir,
            expected_identity=self.identity,
            expected_work_plan=self.plan,
            expected_lineage=self.lineage,
            clock=self.clock,
        )
        self.assertEqual(resumed.load_main_seed_payload(101), first)
        self.assertEqual(resumed.remaining_main_seeds, (102,))
        self.assertEqual(resumed.progress_snapshot["in_progress_unit_ids"], [])
        self.assertEqual(resumed.progress_snapshot["resume_count"], 1)
        resumed.begin_main_seed(102)

    def test_duplicate_units_are_rejected_before_reexecution(self) -> None:
        store = self._create()
        store.commit_main_seed(101, _main_payloads(self.plan, 101))
        with self.assertRaises(DuplicateWorkUnitError):
            store.begin_main_seed(101)
        with self.assertRaises(DuplicateWorkUnitError):
            store.commit_main_seed(101, _main_payloads(self.plan, 101))

        store.commit_main_seed(102, _main_payloads(self.plan, 102))
        store.commit_sensitivity("temperature", "low", 101, {"x": 1})
        with self.assertRaises(DuplicateWorkUnitError):
            store.begin_sensitivity("temperature", "low", 101)

    def test_crash_before_index_publish_leaves_orphan_but_no_completed_units(
        self,
    ) -> None:
        store = self._create()
        original = checkpoint_module._atomic_write_json

        def fail_checkpoint(path: Path, payload: object, *, replace: bool) -> None:
            if path.name == "checkpoint.json":
                raise OSError("injected crash before index publish")
            original(path, payload, replace=replace)

        with patch.object(
            checkpoint_module, "_atomic_write_json", side_effect=fail_checkpoint
        ):
            with self.assertRaisesRegex(OSError, "injected crash"):
                store.commit_main_seed(101, _main_payloads(self.plan, 101))

        self.assertTrue(list((store.attempt_dir / "units" / "main_batch").glob("*.json")))
        reloaded = self._load()
        self.assertEqual(reloaded.completed_main_seeds, ())
        self.assertEqual(reloaded.progress_snapshot["overall"]["completed_units"], 0)

    def test_incompatible_configuration_backend_and_plan_are_rejected(self) -> None:
        store = self._create()
        changed_config = replace(self.identity, configuration_sha256="8" * 64)
        with self.assertRaises(IncompatibleCheckpointError):
            ResumableCheckpointStore.load(
                store.attempt_dir,
                expected_identity=changed_config,
                expected_work_plan=self.plan,
                expected_lineage=self.lineage,
            )

        for changed_identity in (
            replace(self.identity, source_tree_sha256="b" * 64),
            replace(
                self.identity,
                runtime=replace(
                    self.identity.runtime,
                    installed_dependencies_sha256="c" * 64,
                ),
            ),
            replace(
                self.identity,
                backend=replace(
                    self.identity.backend,
                    native_thread_library_sha256="d" * 64,
                ),
            ),
        ):
            with self.subTest(identity=changed_identity.identity_sha256):
                with self.assertRaises(IncompatibleCheckpointError):
                    ResumableCheckpointStore.load(
                        store.attempt_dir,
                        expected_identity=changed_identity,
                        expected_work_plan=self.plan,
                        expected_lineage=self.lineage,
                    )

        gpu = BackendIdentity(
            requested_backend="gpu",
            resolved_backend="gpu",
            library="cupy",
            library_version="13.6.0",
            device_name="fixture-gpu",
            device_id="gpu:0",
            compute_capability="8.9",
            driver_version="fixture",
            runtime_version="13.0",
            precision_mode="float64",
            worker_count=1,
            batch_size=64,
            scheduling_policy="bounded_gpu_batches",
            native_thread_runtime="fixture-openblas",
            native_thread_library_path="C:/runtime/openblas.dll",
            native_thread_library_sha256="9" * 64,
            native_thread_getter_symbol="fixture_get_num_threads",
            native_thread_setter_symbol="fixture_set_num_threads",
            native_thread_limit=1,
        )
        changed_backend = replace(self.identity, backend=gpu)
        with self.assertRaises(IncompatibleCheckpointError):
            ResumableCheckpointStore.load(
                store.attempt_dir,
                expected_identity=changed_backend,
                expected_work_plan=self.plan,
                expected_lineage=self.lineage,
            )

        changed_plan = ExecutionWorkPlan.build(
            seeds=(101, 102, 103),
            scenario_ids=self.plan.scenario_ids,
            sensitivity_units=self.plan.sensitivity_units,
        )
        with self.assertRaises(IncompatibleCheckpointError):
            ResumableCheckpointStore.load(
                store.attempt_dir,
                expected_identity=self.identity,
                expected_work_plan=changed_plan,
                expected_lineage=self.lineage,
            )

    def test_partial_checkpoint_and_altered_payload_fail_closed(self) -> None:
        store = self._create()
        store.commit_main_seed(101, _main_payloads(self.plan, 101))
        block = next((store.attempt_dir / "units" / "main_batch").glob("*.json"))
        original = block.read_bytes()
        block.write_bytes(original.replace(b'"harm": 0.101', b'"harm": 9.999'))
        with self.assertRaises(CheckpointCorruptError):
            self._load()

        block.write_bytes(original)
        store.checkpoint_path.write_text('{"schema_version":', "utf-8")
        with self.assertRaises(CheckpointCorruptError):
            self._load()

    def test_stale_progress_is_rebuilt_from_authoritative_checkpoint(self) -> None:
        store = self._create()
        store.commit_main_seed(101, _main_payloads(self.plan, 101))
        store.progress_path.write_text('{"wrong": true}\n', "utf-8")
        loaded = self._load()
        repaired = json.loads(loaded.progress_path.read_text("utf-8"))
        self.assertEqual(repaired["main_batch"]["completed_units"], 7)
        self.assertEqual(repaired["checkpoint"]["identity_sha256"], loaded._state["checkpoint_sha256"])

    def test_completion_requires_all_work_and_final_output_checksums(self) -> None:
        store = self._create()
        with self.assertRaises(CheckpointIncompleteError):
            store.mark_complete({"manifest.json": "9" * 64})
        for seed in self.plan.seeds:
            store.commit_main_seed(seed, _main_payloads(self.plan, seed))
        for unit in self.plan.sensitivity_units:
            store.commit_sensitivity(
                unit.parameter_id, unit.level_id, unit.seed, {"ok": True}
            )
        self.assertEqual(
            store.progress_snapshot["status"],
            "COMPUTE_COMPLETE_EXPORT_PENDING",
        )
        store.mark_complete({"manifest.json": "9" * 64})
        self.assertEqual(store.progress_snapshot["status"], "COMPLETE")
        self.assertEqual(store.progress_snapshot["overall"]["percentage"], 100.0)

    def test_previous_attempt_is_preserved_and_next_id_is_monotonic(self) -> None:
        previous = self.root / "attempt-000001"
        previous.mkdir()
        marker = previous / "progress.json"
        marker.write_text('{"status":"INTERRUPTED"}\n', "utf-8")
        before = marker.read_bytes()
        self.assertEqual(next_attempt_id(self.root), "attempt-000002")
        self._create()
        self.assertEqual(marker.read_bytes(), before)
        self.assertEqual(next_attempt_id(self.root), "attempt-000003")
        with self.assertRaises(FileExistsError):
            self._create()

    def test_v1_lineage_does_not_retroactively_claim_source_identity(self) -> None:
        snapshot = self.lineage.snapshot()["previous_execution"]
        self.assertEqual(
            snapshot["source_identity_status"],
            "NOT_RECORDED_BY_CHECKPOINT_SCHEMA_V1",
        )
        self.assertEqual(
            snapshot["run_identity_status"],
            "NOT_RECORDED_BY_CHECKPOINT_SCHEMA_V1",
        )
        self.assertIsNone(snapshot["run_id"])
        self.assertIsNone(snapshot["source_tree_sha256"])
        self.assertIsNone(snapshot["git_commit"])
        with self.assertRaisesRegex(ValueError, "must remain null"):
            replace(
                self.lineage.previous,
                source_tree_sha256="a" * 64,
            )

    def test_sensitivity_level_identity_preserves_exact_float_bits(self) -> None:
        self.assertEqual(canonical_level_id(1), "int:1")
        self.assertEqual(canonical_level_id(0.1), f"float64:{(0.1).hex()}")
        self.assertNotEqual(canonical_level_id(1), canonical_level_id(1.0))
        with self.assertRaises(ValueError):
            canonical_level_id(float("nan"))

    def test_native_thread_resume_identity_excludes_only_invocation_history(
        self,
    ) -> None:
        first = NativeThreadAttestation(
            runtime="fixture-openblas",
            library_path="C:/runtime/openblas.dll",
            library_sha256="9" * 64,
            getter_symbol="get_threads",
            setter_symbol="set_threads",
            previous_thread_count=24,
            enforced_thread_count=1,
        )
        second = replace(first, previous_thread_count=1)
        self.assertNotEqual(first.snapshot(), second.snapshot())
        self.assertEqual(first.identity_snapshot(), second.identity_snapshot())


if __name__ == "__main__":
    unittest.main()
