from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from microtx_sim.causal.scenarios import ScenarioId
from microtx_sim.outputs.checkpoints import EXPLORATORY_CHECKPOINT_COLUMNS
from microtx_sim.outputs.exploratory import EXPLORATORY_INTERPRETATION_WORDING
from microtx_sim.outputs.exploratory_results import (
    EXPLORATORY_RESULT_ARTIFACTS,
    SENSITIVITY_DIAGNOSTIC_COLUMNS,
    WEIGHTED_PRIMARY_COLUMNS,
    export_exploratory_results,
    preflight_exploratory_output,
)
from microtx_sim.policy_config import load_policy_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "policy_exploratory_synthetic.toml"


class ExploratoryExecutionTests(unittest.TestCase):
    def test_output_contract_contains_no_raw_monetary_columns(self) -> None:
        for columns in (
            EXPLORATORY_CHECKPOINT_COLUMNS,
            WEIGHTED_PRIMARY_COLUMNS,
            SENSITIVITY_DIAGNOSTIC_COLUMNS,
        ):
            self.assertFalse(any("cents" in column for column in columns))
        self.assertNotIn("player_outcomes.csv", EXPLORATORY_RESULT_ARTIFACTS)
        self.assertNotIn("scenario_summary.csv", EXPLORATORY_RESULT_ARTIFACTS)

    def test_preflight_preserves_progress_and_rejects_final_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            progress = output / "progress" / "attempt-000001"
            progress.mkdir(parents=True)
            marker = progress / "progress.json"
            marker.write_text("{}\n", "utf-8")

            self.assertEqual(preflight_exploratory_output(output), output)
            self.assertTrue(marker.is_file())
            (output / "manifest.json").write_text("{}\n", "utf-8")
            with self.assertRaisesRegex(FileExistsError, "will not be overwritten"):
                preflight_exploratory_output(output)
            self.assertTrue(marker.is_file())

    def test_dedicated_export_is_nonempirical_and_fail_closed_without_model_run(
        self,
    ) -> None:
        config = load_policy_config(CONFIG)
        assert config.exploratory is not None
        seeds = config.batch.seeds
        run_input_sha256 = "c" * 64
        cohort_by_seed = {seed: f"{seed:064x}" for seed in seeds}
        planned = SimpleNamespace(
            estimand_id=config.exploratory.primary_estimand_id,
            reference_scenario_id=ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
            comparison_scenario_id=ScenarioId.BASELINE_F2P,
            contrast_direction="COMPARISON_MINUS_REFERENCE",
            outcome_semantics=SimpleNamespace(unit="model_score"),
        )
        bindings = tuple(
            SimpleNamespace(
                seed=seed,
                planned_estimand=planned,
                result=SimpleNamespace(
                    metric_name="composite_harm",
                    value=0.1 + seed / 1_000_000.0,
                    numerator=seed + 1000,
                    denominator=10_000,
                    result_sha256=f"{seed + 10_000:064x}",
                ),
                selected_weights=SimpleNamespace(
                    design_sha256=f"{seed + 20_000:064x}"
                ),
                population_seed_record_sha256=f"{seed + 30_000:064x}",
                selected_player_count=config.batch.player_count,
                binding_sha256=f"{seed + 40_000:064x}",
            )
            for seed in seeds
        )
        analysis_binding = SimpleNamespace(
            seeds=seeds,
            seed_bindings=bindings,
            binding_sha256="d" * 64,
            plan=SimpleNamespace(plan_id="parent.plan.v1", plan_sha256="e" * 64),
        )
        batch = SimpleNamespace(
            spec=config.batch,
            records=(),
            cohort_digest_by_seed=cohort_by_seed,
            run_input_sha256=lambda: run_input_sha256,
            population_execution_lineage=SimpleNamespace(
                lineage_sha256="f" * 64
            ),
        )
        empty_checkpoint = SimpleNamespace(
            nonmonetary_diagnostic_rows=lambda: []
        )
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            config_copy = repository / "configs" / CONFIG.name
            config_copy.parent.mkdir(parents=True)
            config_copy.write_bytes(CONFIG.read_bytes())
            output = repository / config.output.output_dir
            with patch(
                "microtx_sim.outputs.exploratory_results.PolicyBatchCheckpoint",
                return_value=empty_checkpoint,
            ):
                paths = export_exploratory_results(
                    config,
                    batch,
                    None,
                    analysis_binding,
                    config_path=config_copy,
                    output_dir=output,
                    command=(
                        "microtx-sim",
                        "policy-batch",
                        "configs/policy_exploratory_synthetic.toml",
                    ),
                    exploratory_validation_metadata={
                        "validation_status": "VALIDATED_NOT_EXECUTED"
                    },
                    checkpoint_attempt_id="attempt-000001",
                )

            self.assertEqual(
                {path.name for path in paths.values()},
                set(EXPLORATORY_RESULT_ARTIFACTS),
            )
            metadata = json.loads(
                (output / "nonempirical_metadata.json").read_text("utf-8")
            )
            self.assertEqual(
                metadata["interpretation_wording"],
                EXPLORATORY_INTERPRETATION_WORDING,
            )
            self.assertFalse(metadata["campaign_ready"])
            self.assertFalse(
                metadata["monetary_interpretation"][
                    "raw_internal_units_published"
                ]
            )
            uncertainty = json.loads(
                (output / "uncertainty_summary.json").read_text("utf-8")
            )
            self.assertFalse(uncertainty["campaign_ready"])
            self.assertEqual(
                uncertainty["final_sufficiency_judgment"]["judgment"],
                "INSUFFICIENT",
            )
            manifest = json.loads(
                (output / "manifest.json").read_text("utf-8")
            )
            self.assertFalse(manifest["campaign_ready"])
            self.assertNotEqual(manifest["convergence_status"], "CONVERGED")


if __name__ == "__main__":
    unittest.main()
