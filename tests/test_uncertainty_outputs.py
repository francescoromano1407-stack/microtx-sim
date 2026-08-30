from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.analysis.uncertainty import (
    ConvergenceRule,
    ConvergenceStatus,
    RealizationIdentity,
    UncertaintyAvailability,
    UncertaintyComponentStatus,
    UncertaintyRealization,
    decompose_joint_uncertainty,
    evaluate_blockwise_convergence,
    final_sufficiency_judgment,
    summarize_seed_uncertainty,
)
from microtx_sim.outputs.schema import (
    CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
    CONVERGENCE_CHECKPOINT_COLUMNS,
    UNCERTAINTY_REALIZATION_COLUMNS,
)
from microtx_sim.outputs.uncertainty import write_joint_uncertainty_outputs


def _row(seed: int, value: float) -> UncertaintyRealization:
    return UncertaintyRealization(
        identity=RealizationIdentity(
            seed=seed,
            parameter_draw_id="nominal",
            parameter_draw_sha256="1" * 64,
            population_design_id="design",
            population_replicate_id="deterministic",
            population_design_sha256="2" * 64,
            monetary_rate_draw_id="official-point-rate",
            monetary_rate_basis_id="ecb-2024",
            monetary_rate_basis_sha256="3" * 64,
            scenario_id="baseline-minus-safe",
            primary_estimand_id="primary.composite-harm.v1",
            pretreatment_cohort_sha256="4" * 64,
            population_weights_sha256="5" * 64,
        ),
        estimate=value,
        valid=True,
    )


class UncertaintyOutputTests(unittest.TestCase):
    def test_writer_separates_components_convergence_and_final_judgment(self) -> None:
        rows = tuple(_row(seed, 0.1) for seed in range(1, 101))
        seed_summary = summarize_seed_uncertainty(
            rows,
            expected_seeds=tuple(range(1, 101)),
        )
        convergence = evaluate_blockwise_convergence(
            rows,
            expected_seeds=tuple(range(1, 101)),
            rule=ConvergenceRule(block_size=50),
            required_components_available=False,
        )
        components = (
            UncertaintyComponentStatus(
                "seed",
                UncertaintyAvailability.QUANTIFIED,
                0.0,
                "sample variance of complete fixed seed estimates",
                None,
            ),
            UncertaintyComponentStatus(
                "parameter",
                UncertaintyAvailability.UNQUANTIFIED,
                None,
                None,
                "illustrative ranges have no probability interpretation",
            ),
            UncertaintyComponentStatus(
                "monetary_rate",
                UncertaintyAvailability.UNQUANTIFIED,
                None,
                None,
                "official point rate only",
            ),
            UncertaintyComponentStatus(
                "population",
                UncertaintyAvailability.UNQUANTIFIED,
                None,
                None,
                "deterministic exact weights are not population validation",
            ),
            UncertaintyComponentStatus(
                "combined",
                UncertaintyAvailability.UNAVAILABLE,
                None,
                None,
                "required components unavailable",
            ),
        )
        judgment = final_sufficiency_judgment(
            convergence_status=convergence[-1].status,
            components=components,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_joint_uncertainty_outputs(
                directory,
                realizations=rows,
                primary_seed_realizations=rows,
                primary_seed_summary=seed_summary,
                components=components,
                variance_decomposition=decompose_joint_uncertainty(rows),
                convergence_checkpoints=convergence,
                sufficiency_judgment=judgment,
                expected_seeds=tuple(range(1, 101)),
                convergence_rule=ConvergenceRule(block_size=50),
                plan_id="plan-v3",
                plan_sha256="6" * 64,
                config_sha256="7" * 64,
            )
            self.assertEqual(len(paths), 3)
            with paths["uncertainty_realizations"].open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), UNCERTAINTY_REALIZATION_COLUMNS)
                self.assertEqual(len(list(reader)), 100)
            with paths["convergence_checkpoints"].open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), CONVERGENCE_CHECKPOINT_COLUMNS)
                self.assertEqual(len(list(reader)), 2)
            summary = json.loads(
                paths["uncertainty_summary"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                summary["output_profile_schema_sha256"],
                CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
            )
            self.assertEqual(
                summary["monetary_rate_uncertainty"]["availability"],
                "UNQUANTIFIED",
            )
            self.assertEqual(summary["convergence"]["status"], "NON_CONVERGED")
            self.assertFalse(summary["final_sufficiency_judgment"]["sufficient"])
            self.assertFalse(summary["campaign_ready"])

    def test_writer_rejects_independent_readiness_promotion(self) -> None:
        rows = tuple(_row(seed, 0.1) for seed in range(1, 101))
        rule = ConvergenceRule(block_size=50)
        components = (
            UncertaintyComponentStatus(
                "seed",
                UncertaintyAvailability.QUANTIFIED,
                0.0,
                "sample variance of complete fixed seed estimates",
                None,
            ),
            UncertaintyComponentStatus(
                "parameter",
                UncertaintyAvailability.UNQUANTIFIED,
                None,
                None,
                "illustrative ranges",
            ),
            UncertaintyComponentStatus(
                "monetary_rate",
                UncertaintyAvailability.UNQUANTIFIED,
                None,
                None,
                "point rate only",
            ),
            UncertaintyComponentStatus(
                "population",
                UncertaintyAvailability.UNQUANTIFIED,
                None,
                None,
                "no population uncertainty design",
            ),
            UncertaintyComponentStatus(
                "combined",
                UncertaintyAvailability.UNAVAILABLE,
                None,
                None,
                "required components unavailable",
            ),
        )
        convergence = evaluate_blockwise_convergence(
            rows,
            expected_seeds=tuple(range(1, 101)),
            rule=rule,
            required_components_available=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "sufficiency judgment differs"):
                write_joint_uncertainty_outputs(
                    directory,
                    realizations=rows,
                    primary_seed_realizations=rows,
                    primary_seed_summary=summarize_seed_uncertainty(
                        rows,
                        expected_seeds=tuple(range(1, 101)),
                    ),
                    components=components,
                    variance_decomposition=decompose_joint_uncertainty(rows),
                    convergence_checkpoints=convergence,
                    sufficiency_judgment={"campaign_ready": True},
                    expected_seeds=tuple(range(1, 101)),
                    convergence_rule=rule,
                    plan_id="plan-v3",
                    plan_sha256="6" * 64,
                    config_sha256="7" * 64,
                )


if __name__ == "__main__":
    unittest.main()
