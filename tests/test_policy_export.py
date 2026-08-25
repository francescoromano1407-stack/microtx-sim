from __future__ import annotations

import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from microtx_sim.analysis.sensitivity import (
    SensitivityCase,
    run_sensitivity_analysis,
)
from microtx_sim.causal.batch import (
    PolicyBatchResult,
    PolicyBatchSpec,
    run_policy_batch,
)
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.consumers.population import CountryProfile
from microtx_sim.data.profiles import load_profile_bundle
from microtx_sim.outputs.export import export_policy_batch
from microtx_sim.outputs.schema import (
    EPGC_FINANCING_COLUMNS,
    OPPORTUNITY_DECOMPOSITION_COLUMNS,
    PLAYER_OUTCOME_COLUMNS,
    POLICY_ARTIFACT_FILENAMES,
    SCENARIO_SUMMARY_COLUMNS,
    SEED_RESULT_COLUMNS,
    SENSITIVITY_COLUMNS,
)
from microtx_sim.policy_config import PolicyOutputConfig, load_policy_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "policy_prototype.toml"
JURISDICTIONS_PATH = ROOT / "configs" / "jurisdictions.toml"
SOURCES_PATH = ROOT / "data" / "provenance" / "sources.toml"


class PolicyExportTests(unittest.TestCase):
    def test_complete_export_contains_tables_metadata_summary_and_valid_svgs(self) -> None:
        base = load_policy_config(CONFIG_PATH)
        spec = PolicyBatchSpec(
            seeds=(5, 6),
            days=1,
            player_count=16,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        batch = run_policy_batch(
            spec,
            country_profiles=(CountryProfile(code="XX"),),
            harm_parameters=base.harm_parameters,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            config = replace(
                base,
                batch=spec,
                output=PolicyOutputConfig(output, 8, True, False),
            )
            paths = export_policy_batch(
                config,
                batch,
                None,
                config_path=CONFIG_PATH,
                repository_root=ROOT,
                created_utc="2026-01-01T00:00:00+00:00",
                command=("microtx-sim", "policy-batch"),
            )
            self.assertEqual(
                {path.name for path in paths.values()},
                set(POLICY_ARTIFACT_FILENAMES),
            )
            with (output / "scenario_summary.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 7)
            with (output / "player_outcomes.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                players = list(csv.DictReader(handle))
            self.assertEqual(len(players), 2 * 7 * 16)
            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["output_schema_version"], "2.0")
            self.assertTrue(manifest["synthetic_only"])
            self.assertFalse(manifest["empirical_validation_claimed"])
            self.assertEqual(
                manifest["artifact_files"], list(POLICY_ARTIFACT_FILENAMES)
            )
            self.assertEqual(len(manifest["scenarios"]), 7)
            self.assertEqual(manifest["batch"]["seeds"], [5, 6])
            self.assertEqual(manifest["batch"]["profile_codes"], ["XX"])
            self.assertEqual(
                manifest["profile_inputs"]["lineage_status"],
                "unregistered_custom_profiles",
            )
            self.assertEqual(
                manifest["profile_inputs"]["fingerprint_sha256"],
                batch.profile_input_lineage.fingerprint_sha256,
            )
            self.assertIsNone(manifest["jurisdictions_sha256"])
            self.assertIsNone(manifest["source_registry_sha256"])
            self.assertIsNone(manifest["source_registry_retrieved_on"])
            self.assertEqual(
                manifest["profile_inputs"]["snapshot"]["country_profiles"][0][
                    "code"
                ],
                "XX",
            )
            self.assertIn("scenario_summary.csv", manifest["artifacts"])
            self.assertIn("illustrative assumptions", (output / "summary.md").read_text("utf-8"))
            contracts = {
                "seed_results.csv": SEED_RESULT_COLUMNS,
                "scenario_summary.csv": SCENARIO_SUMMARY_COLUMNS,
                "epgc_financing.csv": EPGC_FINANCING_COLUMNS,
                "sensitivity.csv": SENSITIVITY_COLUMNS,
                "player_outcomes.csv": PLAYER_OUTCOME_COLUMNS,
                "opportunity_cost_decomposition.csv": (
                    OPPORTUNITY_DECOMPOSITION_COLUMNS
                ),
            }
            for filename, columns in contracts.items():
                with (output / filename).open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(next(csv.reader(handle)), list(columns))
            for svg in output.glob("*.svg"):
                ET.parse(svg)

    def test_manifest_lineage_matches_injected_profile_bundle(self) -> None:
        base = load_policy_config(CONFIG_PATH)
        spec = PolicyBatchSpec(
            seeds=(31,),
            days=0,
            player_count=0,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jurisdictions = root / "jurisdictions.toml"
            jurisdictions.write_text(
                JURISDICTIONS_PATH.read_text(encoding="utf-8").replace(
                    "population_weight = 0.25",
                    "population_weight = 0.30",
                    1,
                ),
                encoding="utf-8",
            )
            bundle = load_profile_bundle(jurisdictions, SOURCES_PATH)
            batch = run_policy_batch(
                spec,
                profile_bundle=bundle,
                harm_parameters=base.harm_parameters,
                harm_weights=base.harm_weights,
                opportunity_valuation=base.opportunity_valuation,
                producer_assumptions=base.producer_assumptions,
                epgc_policy=base.epgc_policy,
            )
            output = root / "bundle"
            config = replace(
                base,
                batch=spec,
                output=PolicyOutputConfig(output, 8, True, False),
            )
            export_policy_batch(
                config,
                batch,
                None,
                config_path=CONFIG_PATH,
                repository_root=ROOT,
                output_dir=output,
                created_utc="2026-01-01T00:00:00+00:00",
            )

            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            lineage = manifest["profile_inputs"]
            self.assertEqual(
                lineage["lineage_status"],
                "registered_profile_bundle",
            )
            self.assertEqual(lineage["profile_codes"], ["UK", "KR", "JP", "BE"])
            self.assertEqual(
                lineage["fingerprint_sha256"],
                batch.profile_input_lineage.fingerprint_sha256,
            )
            self.assertEqual(
                lineage["jurisdictions"],
                {
                    "path": str(jurisdictions.resolve()),
                    "sha256": sha256(jurisdictions.read_bytes()).hexdigest(),
                },
            )
            self.assertEqual(
                lineage["source_registry"]["path"],
                str(SOURCES_PATH.resolve()),
            )
            self.assertEqual(
                lineage["source_registry"]["sha256"],
                sha256(SOURCES_PATH.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                lineage["source_registry"]["retrieved_on"],
                "2026-08-24",
            )
            self.assertEqual(
                lineage["snapshot"]["country_profiles"][0]["population_weight"],
                0.30,
            )
            self.assertEqual(
                lineage["metric_contract_summary"]["count"],
                len(bundle.contracts),
            )
            self.assertEqual(lineage["money_scale_summary"]["count"], 4)
            self.assertEqual(
                manifest["jurisdictions_sha256"],
                lineage["jurisdictions"]["sha256"],
            )
            self.assertEqual(
                manifest["source_registry_sha256"],
                lineage["source_registry"]["sha256"],
            )

    def test_export_rejects_undeclared_player_and_opportunity_fields(self) -> None:
        base = load_policy_config(CONFIG_PATH)
        spec = PolicyBatchSpec(
            seeds=(5,),
            days=1,
            player_count=2,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        batch = run_policy_batch(
            spec,
            country_profiles=(CountryProfile(code="XX"),),
            harm_parameters=base.harm_parameters,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        rows_by_method = {
            "player_rows": batch.player_rows(),
            "opportunity_rows": batch.opportunity_rows(),
        }
        for method_name, rows in rows_by_method.items():
            invalid_row = dict(rows[0])
            invalid_row["not_in_schema"] = 1
            with (
                self.subTest(method=method_name),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "bundle"
                config = replace(
                    base,
                    batch=spec,
                    output=PolicyOutputConfig(output, 8, True, False),
                )
                with patch.object(
                    PolicyBatchResult,
                    method_name,
                    return_value=[invalid_row],
                ), self.assertRaisesRegex(
                    ValueError,
                    "undeclared columns: not_in_schema",
                ):
                    export_policy_batch(
                        config,
                        batch,
                        None,
                        config_path=CONFIG_PATH,
                        repository_root=ROOT,
                        output_dir=output,
                        created_utc="2026-01-01T00:00:00+00:00",
                    )
                self.assertFalse(output.exists())

    def test_export_requires_matching_batch_and_sensitivity_lineage(self) -> None:
        base = load_policy_config(CONFIG_PATH)
        spec = PolicyBatchSpec(
            seeds=(7,),
            days=0,
            player_count=0,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        profile = CountryProfile(code="XX")
        batch = run_policy_batch(spec, country_profiles=(profile,))
        sensitivity = run_sensitivity_analysis(
            spec,
            cases=(SensitivityCase("paid_random_rewards", (0.0, 0.7)),),
            country_profiles=(profile,),
        )

        cases = (
            (
                replace(batch, profile_input_lineage=None),
                None,
                "batch export requires profile input lineage",
            ),
            (
                batch,
                replace(sensitivity, profile_input_lineage=None),
                "sensitivity export requires profile input lineage",
            ),
            (
                batch,
                run_sensitivity_analysis(
                    spec,
                    cases=(
                        SensitivityCase("paid_random_rewards", (0.0, 0.7)),
                    ),
                    country_profiles=(
                        replace(profile, awareness_mean=0.51),
                    ),
                ),
                "different profile inputs",
            ),
        )
        for candidate_batch, candidate_sensitivity, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "bundle"
                config = replace(
                    base,
                    batch=spec,
                    output=PolicyOutputConfig(output, 8, True, False),
                )
                with self.assertRaisesRegex(ValueError, message):
                    export_policy_batch(
                        config,
                        candidate_batch,
                        candidate_sensitivity,
                        config_path=CONFIG_PATH,
                        repository_root=ROOT,
                        output_dir=output,
                        created_utc="2026-01-01T00:00:00+00:00",
                    )
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
