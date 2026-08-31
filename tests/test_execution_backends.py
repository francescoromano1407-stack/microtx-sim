from __future__ import annotations

import unittest

import numpy as np

from microtx_sim.execution import (
    BackendMode,
    BackendUnavailableError,
    ExecutionBackendConfig,
    NumericalParityTolerance,
    compute_composite_harm,
    probe_gpu_backend,
    resolve_execution_backend,
    validate_composite_harm_parity,
)
from microtx_sim.causal.scenarios import required_scenarios
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.consumers.population import CountryProfile, initialize_player_table
from microtx_sim.consumers.welfare import initialize_player_life
from microtx_sim.rng import CounterRNG
from microtx_sim.simulation.policy_orchestrator import run_policy_scenario


class ExecutionBackendTests(unittest.TestCase):
    def test_backend_config_is_strict_and_content_addressed(self) -> None:
        config = ExecutionBackendConfig(
            mode="cpu",
            batch_size=64,
            max_batch_bytes=4_096,
        )
        first = resolve_execution_backend(config)
        second = resolve_execution_backend(config)
        self.assertIs(config.mode, BackendMode.CPU)
        self.assertEqual(
            first.metadata.backend_identity_sha256,
            second.metadata.backend_identity_sha256,
        )
        self.assertEqual(first.metadata.resolution, "EXPLICIT_CPU")
        self.assertEqual(
            first.metadata.identity_payload()["kernel_placement"],
            {
                "categorical_decisions": "cpu_reference",
                "composite_harm_reporting": "cpu_reference",
                "counter_rng": "cpu_reference",
                "model_state_transitions": "cpu_reference",
            },
        )
        self.assertEqual(len(first.metadata.backend_identity_sha256), 64)
        with self.assertRaises(TypeError):
            first.metadata.kernel_placement["counter_rng"] = "gpu"  # type: ignore[index]
        for invalid in ("cuda", "", True):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    ExecutionBackendConfig(mode=invalid)  # type: ignore[arg-type]

    def test_batch_planning_enforces_item_and_memory_bounds(self) -> None:
        backend = resolve_execution_backend(
            ExecutionBackendConfig(
                mode="cpu",
                batch_size=10,
                max_batch_bytes=32,
            )
        )
        self.assertEqual(backend.effective_batch_size(bytes_per_item=8), 4)
        self.assertEqual(
            backend.effective_batch_size(bytes_per_item=8, fixed_bytes=8),
            3,
        )
        self.assertEqual(
            list(backend.iter_batches(10, bytes_per_item=8)),
            [slice(0, 4), slice(4, 8), slice(8, 10)],
        )
        with self.assertRaises(MemoryError):
            backend.effective_batch_size(bytes_per_item=33)
        with self.assertRaises(MemoryError):
            backend.effective_batch_size(bytes_per_item=8, fixed_bytes=32)

    def test_cpu_composite_kernel_is_bitwise_reference_equivalent(self) -> None:
        backend = resolve_execution_backend(
            ExecutionBackendConfig(mode="cpu", batch_size=7, max_batch_bytes=512)
        )
        rng = np.random.default_rng(20260901)
        scores = rng.random((33, 6), dtype=np.float64)
        weights = np.asarray([1.0, 1.5, 0.8, 1.2, 0.9, 1.1], dtype=np.float64)
        expected = scores @ (weights / weights.sum())
        actual = compute_composite_harm(scores, weights, backend=backend)
        np.testing.assert_array_equal(actual, expected)
        report = validate_composite_harm_parity(
            scores,
            weights,
            backend=backend,
            categorical_thresholds=(0.35, 0.50),
        )
        self.assertTrue(report.passed)
        self.assertTrue(report.bitwise_equal)
        self.assertEqual(report.maximum_absolute_difference, 0.0)

    def test_composite_kernel_validates_precision_and_shapes(self) -> None:
        backend = resolve_execution_backend(ExecutionBackendConfig(mode="cpu"))
        with self.assertRaisesRegex(TypeError, "float64"):
            compute_composite_harm(
                np.zeros((2, 6), dtype=np.float32),
                np.ones(6),
                backend=backend,
            )
        with self.assertRaisesRegex(ValueError, "players, 6"):
            compute_composite_harm(
                np.zeros((2, 5), dtype=np.float64),
                np.ones(6),
                backend=backend,
            )
        with self.assertRaisesRegex(ValueError, "positive sum"):
            compute_composite_harm(
                np.zeros((2, 6), dtype=np.float64),
                np.zeros(6),
                backend=backend,
            )
        self.assertEqual(NumericalParityTolerance().absolute, 5e-13)

    def test_cpu_backend_is_used_in_tiny_policy_flow_without_semantic_change(self) -> None:
        backend = resolve_execution_backend(
            ExecutionBackendConfig(mode="cpu", batch_size=2, max_batch_bytes=256)
        )
        players = initialize_player_table(
            4,
            (CountryProfile(code="XX"),),
            CounterRNG(41),
        )
        life = initialize_player_life(players, CounterRNG(41))
        kwargs = {
            "seed": 41,
            "days": 1,
            "decision_parameters": DecisionParameters(step_minutes=240),
        }
        reference = run_policy_scenario(
            players,
            life,
            required_scenarios()[0],
            **kwargs,
        )
        selected = run_policy_scenario(
            players,
            life,
            required_scenarios()[0],
            execution_backend=backend,
            **kwargs,
        )
        np.testing.assert_array_equal(selected.composite_harm, reference.composite_harm)
        np.testing.assert_array_equal(selected.high_risk, reference.high_risk)
        np.testing.assert_array_equal(selected.spending_cents, reference.spending_cents)
        np.testing.assert_array_equal(selected.action_minutes, reference.action_minutes)

    def test_explicit_gpu_request_fails_instead_of_falling_back_when_unavailable(self) -> None:
        probe = probe_gpu_backend()
        if probe.available:
            self.skipTest("a CUDA GPU is available on this test host")
        with self.assertRaisesRegex(
            BackendUnavailableError,
            "explicitly requested.*unavailable",
        ):
            resolve_execution_backend(ExecutionBackendConfig(mode="gpu"))

    def test_auto_records_its_resolution(self) -> None:
        probe = probe_gpu_backend()
        backend = resolve_execution_backend(
            ExecutionBackendConfig(mode="auto", batch_size=128)
        )
        if probe.available:
            self.assertIs(backend.mode, BackendMode.GPU)
            self.assertEqual(backend.metadata.resolution, "AUTO_SELECTED_GPU")
        else:
            self.assertIs(backend.mode, BackendMode.CPU)
            self.assertIn(
                "AUTO_SELECTED_CPU_NO_COMPATIBLE_GPU",
                backend.metadata.resolution,
            )

    def test_gpu_composite_parity_when_cuda_is_available(self) -> None:
        probe = probe_gpu_backend()
        if not probe.available:
            self.skipTest(
                "CuPy/CUDA is unavailable; explicit GPU failure is tested separately"
            )
        backend = resolve_execution_backend(
            ExecutionBackendConfig(
                mode="gpu",
                batch_size=31,
                max_batch_bytes=4_096,
            )
        )
        rng = np.random.default_rng(20260901)
        scores = rng.random((67, 6), dtype=np.float64)
        weights = np.asarray([1.0, 1.5, 0.8, 1.2, 0.9, 1.1], dtype=np.float64)
        report = validate_composite_harm_parity(
            scores,
            weights,
            backend=backend,
            categorical_thresholds=(0.35, 0.50),
            raise_on_failure=True,
        )
        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
