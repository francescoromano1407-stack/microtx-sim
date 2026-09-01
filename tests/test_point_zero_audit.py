from __future__ import annotations

import csv
from contextlib import redirect_stdout
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run_point_zero_audit.py"
BUNDLE_ROOT = ROOT / "inputs" / "calibration" / "uk-adults-2024-v1"


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_point_zero_audit", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _committed_bundle_stub() -> SimpleNamespace:
    """Read tracked aggregates without pretending the ignored cache was verified."""

    with (BUNDLE_ROOT / "targets.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        targets = {
            row["target_id"]: SimpleNamespace(
                target_id=row["target_id"],
                value=Decimal(row["value"]) if row["value"] else None,
            )
            for row in csv.DictReader(handle)
        }
    with (BUNDLE_ROOT / "population_weights.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        weights = tuple(
            SimpleNamespace(
                age_band=row["age_band"],
                sex=row["sex"],
                population_count=int(row["population_count"]),
            )
            for row in csv.DictReader(handle)
        )
    bundle_path = BUNDLE_ROOT / "calibration_bundle.json"
    return SimpleNamespace(
        bundle_id="uk-adults-2024-v1",
        bundle_sha256=sha256(bundle_path.read_bytes()).hexdigest(),
        status="PARTIAL",
        campaign_ready=False,
        target_by_id=targets,
        population_weights=weights,
    )


def _gates_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        row["gate_id"]: row
        for row in report["gates"]  # type: ignore[index]
    }


class PointZeroAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.state = tool._initialize_only(tool.DEFAULT_CONFIG_PATH)

    def test_seed_101_initializer_only_report_reproduces_point_zero(self) -> None:
        report = tool.build_report(_committed_bundle_stub(), self.state)

        self.assertEqual(report["status"], "FAIL_CLOSED")
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(report["initialization"]["seed"], 101)
        self.assertEqual(
            report["initialization"]["configured_player_count"], 50_000
        )
        self.assertEqual(
            report["initialization"]["selected_player_count"], 10_024
        )
        self.assertEqual(
            report["execution_scope"],
            {
                "population_initializer_executed": True,
                "player_life_initializer_executed": True,
                "scenario_initialized": False,
                "policy_day_executed": False,
                "campaign_executed": False,
            },
        )
        self.assertEqual(
            report["initializer_means"],
            {
                "sleep_need_minutes_per_day": "480.222765",
                "work_study_obligation_minutes_per_day": "419.989425",
                "social_obligation_minutes_per_day": "78.600658",
                "intended_play_minutes_per_day": "82.828811",
                "sidecar_gamer_intended_play_minutes_per_day": "83.068008",
                "sidecar_gamer_intended_play_minutes_per_week": "581.476057",
                "sidecar_gamer_age_18_40_intended_play_minutes_per_week": (
                    "584.664768"
                ),
                "sidecar_non_gamer_intended_play_minutes_per_day": "82.553146",
            },
        )
        population = report["population_diagnostics"]
        self.assertEqual(population["total_variation_distance"], "0.033053")
        self.assertEqual(population["baseline_gamer_metadata_count"], 5_367)
        self.assertEqual(population["baseline_non_gamer_metadata_count"], 4_657)
        self.assertFalse(
            population["baseline_gamer_metadata_is_behaviorally_binding"]
        )
        self.assertFalse(population["runtime_sex_state_available"])
        self.assertIsNone(population["age_sex_joint_total_variation_distance"])

        gates = _gates_by_id(report)
        self.assertEqual(gates["configured_seed_and_cohort"]["status"], "PASS")
        self.assertEqual(
            gates["uk_adult_selection_nonempty"]["status"], "PASS"
        )
        self.assertEqual(
            gates["initializer_state_finite_and_bounded"]["status"], "PASS"
        )
        self.assertEqual(gates["calibration_bundle_attested"]["status"], "FAIL")
        self.assertEqual(
            gates["calibration_bundle_attested"]["observed"][
                "attestation_status"
            ],
            "UNVERIFIED_NOT_STRICT_LOADER_OUTPUT",
        )
        self.assertEqual(
            gates["calibration_bundle_runtime_binding"]["status"], "FAIL"
        )
        binding = gates["calibration_bundle_runtime_binding"]["observed"]
        self.assertEqual(binding["binding_status"], "UNVERIFIED")
        self.assertFalse(binding["typed_calibration_binding_available"])
        for gate_id in (
            "runtime_sex_state",
            "ons_age_sex_joint_fit",
            "ons_age_marginal_fit",
            "gamer_participation_identified",
            "non_gamer_zero_play_intention",
            "non_gamer_zero_spending_limit",
            "non_gamer_zero_purchase_probability",
        ):
            self.assertEqual(gates[gate_id]["status"], "FAIL")
        self.assertEqual(
            gates["non_gamer_zero_play_intention"]["observed"],
            {"positive_share": "0.997423", "non_gamer_count": 4_657},
        )
        self.assertEqual(
            population["baseline_gamer_behavior_binding_checks"],
            {
                "non_gamer_zero_play_intention": False,
                "non_gamer_zero_spending_limit": False,
                "non_gamer_zero_purchase_probability": False,
            },
        )
        self.assertFalse(population["runtime_sex_field_available"])
        self.assertFalse(population["runtime_sex_lineage_attested"])
        self.assertEqual(
            population["runtime_sex_attestation_status"],
            "UNVERIFIED_FIELD_ABSENT",
        )

        first = json.dumps(
            report, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        second = json.dumps(
            tool.build_report(_committed_bundle_stub(), self.state),
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(first, second)

    def test_construct_incomparable_diagnostics_are_not_fit_gates(self) -> None:
        report = tool.build_report(_committed_bundle_stub(), self.state)
        diagnostics = report["construct_incomparable_diagnostics"]

        self.assertEqual(len(diagnostics), 6)
        self.assertEqual(
            {row["status"] for row in diagnostics},
            {"INCOMPARABLE_DIAGNOSTIC"},
        )
        self.assertEqual({row["gate_applied"] for row in diagnostics}, {False})
        self.assertEqual(
            {
                row["diagnostic_id"]: row["descriptive_ratio"]
                for row in diagnostics
            },
            {
                "sleep_need_vs_ons_sleeping": "0.926177",
                "work_obligation_vs_ons_working": "3.153074",
                "social_obligation_vs_ons_socialising": "2.551969",
                "play_intention_vs_ons_gaming": "4.930286",
                "gamer_play_intention_vs_ofcom": "1.394427",
                "gamer_play_intention_vs_open_play": "0.582179",
            },
        )

    def test_strict_local_bundle_binds_selected_sex_without_treatment(self) -> None:
        raw_cache = ROOT / "data" / "public_calibration_sources_uk_adults_2024"
        if not raw_cache.is_dir():
            self.skipTest("ignored source cache is not present in this checkout")
        bundle = tool.load_uk_adults_2024_calibration_bundle(
            tool.DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH
        )
        state = tool._initialize_only(
            tool.DEFAULT_CONFIG_PATH,
            calibration_bundle=bundle,
        )
        report = tool.build_report(bundle, state)
        gates = _gates_by_id(report)

        self.assertEqual(report["status"], "FAIL_CLOSED")
        self.assertEqual(report["exit_code"], 1)
        self.assertEqual(
            report["gate_summary"],
            {"passed": 6, "failed": 6, "total": 12},
        )
        self.assertEqual(gates["calibration_bundle_attested"]["status"], "PASS")
        self.assertEqual(
            gates["calibration_bundle_runtime_binding"]["status"], "PASS"
        )
        self.assertEqual(gates["runtime_sex_state"]["status"], "PASS")
        self.assertEqual(
            gates["gamer_participation_identified"]["status"], "FAIL"
        )
        self.assertEqual(
            report["population_diagnostics"][
                "age_sex_joint_total_variation_distance"
            ],
            "0.033053",
        )
        self.assertEqual(
            report["population_diagnostics"]["runtime_sex_counts"],
            {
                "FEMALE": 5_093,
                "MALE": 4_931,
                "out_of_scope_unavailable": 39_976,
            },
        )
        self.assertEqual(
            report["execution_scope"],
            {
                "population_initializer_executed": True,
                "player_life_initializer_executed": True,
                "scenario_initialized": False,
                "policy_day_executed": False,
                "campaign_executed": False,
            },
        )

    def test_tool_has_no_treatment_execution_dependency(self) -> None:
        source = TOOL_PATH.read_text(encoding="utf-8")
        for symbol in (
            "run_policy_scenario",
            "run_policy_batch",
            "run_policy_day",
            "SimulationOrchestrator",
            "World.create",
        ):
            self.assertNotIn(symbol, source)

    def test_main_emits_canonical_json_and_uses_success_exit_code(self) -> None:
        expected = {
            "schema_version": 1,
            "audit_id": tool.AUDIT_ID,
            "status": "PASS",
            "exit_code": 0,
        }
        stdout = io.StringIO()
        with patch.object(
            tool, "run_point_zero_audit", return_value=expected
        ), redirect_stdout(stdout):
            exit_code = tool.main([])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            stdout.getvalue(),
            json.dumps(
                expected,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
        )

    def test_main_converts_validation_errors_to_fail_closed_json(self) -> None:
        stdout = io.StringIO()
        with patch.object(
            tool,
            "run_point_zero_audit",
            side_effect=ValueError("deliberate test failure"),
        ), redirect_stdout(stdout):
            exit_code = tool.main([])

        self.assertEqual(exit_code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "ERROR_FAIL_CLOSED")
        self.assertEqual(payload["exit_code"], 1)
        self.assertEqual(
            payload["error"],
            {"type": "ValueError", "message": "deliberate test failure"},
        )
        self.assertEqual(
            payload["execution_scope"],
            {
                "scenario_initialized": False,
                "policy_day_executed": False,
                "campaign_executed": False,
            },
        )

    def test_weighted_share_uses_exact_fraction_arithmetic(self) -> None:
        scale = 10**30
        first_weight = Fraction(scale + 1, scale)
        selected = np.asarray([True, True], dtype=np.bool_)
        mask = np.asarray([True, False], dtype=np.bool_)
        cell_index = np.asarray([0, 1], dtype=np.int32)

        observed = tool._weighted_share(
            mask,
            selected,
            cell_index,
            (first_weight, Fraction(1, 1)),
        )

        self.assertEqual(observed, Fraction(scale + 1, 2 * scale + 1))
        self.assertNotEqual(observed, Fraction(1, 2))

    def test_runtime_binding_rejects_config_path_substring(self) -> None:
        bundle = _committed_bundle_stub()
        poisoned_snapshot = {
            "evidence_bundle_path": (
                f"C:/not-a-binding/{bundle.bundle_id}/{bundle.bundle_sha256}"
            )
        }
        poisoned_config = SimpleNamespace(
            batch=self.state.config.batch,
            population=SimpleNamespace(snapshot=lambda: poisoned_snapshot),
        )
        report = tool.build_report(
            bundle,
            replace(self.state, config=poisoned_config),
        )
        gate = _gates_by_id(report)["calibration_bundle_runtime_binding"]

        self.assertEqual(gate["status"], "FAIL")
        self.assertEqual(gate["observed"]["binding_status"], "UNVERIFIED")
        self.assertFalse(
            gate["observed"]["typed_calibration_binding_available"]
        )

    def test_unattested_runtime_sex_vector_remains_fail_closed(self) -> None:
        plausible = np.full(len(self.state.players), "FEMALE", dtype="<U6")
        uk_index = self.state.players.jurisdiction_codes.index("UK")
        selected = (
            (self.state.players.jurisdiction == uk_index)
            & (self.state.players.age_years >= 18)
            & (self.state.players.age_years <= 64)
        )
        with patch.object(
            tool,
            "_runtime_sex_vector",
            return_value=plausible,
        ):
            vector, field_available, lineage_attested, status = (
                tool._attested_runtime_sex_vector(
                    self.state.players,
                    self.state.projection_execution,
                    selected,
                )
            )

        self.assertIsNone(vector)
        self.assertTrue(field_available)
        self.assertFalse(lineage_attested)
        self.assertEqual(status, "UNVERIFIED_EXECUTION_BINDING_ABSENT")

    def test_unknown_cli_option_exits_one_with_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(TOOL_PATH), "--not-a-real-option"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stderr, "")
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ERROR_FAIL_CLOSED")
        self.assertEqual(payload["exit_code"], 1)
        self.assertEqual(payload["error"]["type"], "PointZeroAuditError")
        self.assertIn("argument parsing failed", payload["error"]["message"])

    def test_help_is_json_and_cannot_claim_a_passing_exit(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = tool.main(["--help"])

        self.assertEqual(exit_code, 1)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "ERROR_FAIL_CLOSED")
        self.assertEqual(payload["exit_code"], 1)
        self.assertIn("exit 0 is reserved", payload["error"]["message"])

    def test_missing_bundle_subprocess_exits_one_without_treatment(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL_PATH),
                "--bundle",
                str(ROOT / "inputs" / "calibration" / "does-not-exist"),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 1)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["status"], "ERROR_FAIL_CLOSED")
        self.assertEqual(payload["exit_code"], 1)
        self.assertEqual(
            payload["execution_scope"],
            {
                "scenario_initialized": False,
                "policy_day_executed": False,
                "campaign_executed": False,
            },
        )


if __name__ == "__main__":
    unittest.main()
