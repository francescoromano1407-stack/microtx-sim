from __future__ import annotations

import csv
from copy import deepcopy
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
from microtx_sim.causal.scenarios import required_scenarios
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.consumers.population import CountryProfile
from microtx_sim.data.profiles import load_profile_bundle
from microtx_sim.outputs.export import export_policy_batch
from microtx_sim.outputs.manifest import (
    _money_conversion_structure_coherent,
    _money_outputs_cross_country_comparable,
)
from microtx_sim.outputs.schema import (
    EPGC_FINANCING_COLUMNS,
    MANIFEST_SCHEMA_SHA256,
    MANIFEST_SCHEMA_VERSION,
    OPPORTUNITY_DECOMPOSITION_COLUMNS,
    OUTPUT_SCHEMA_VERSION,
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


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _comparable_profile_inputs_fixture() -> dict[str, object]:
    """Minimal serialized fixture for the manifest's fail-closed mirror."""

    scale_specs = (
        ("UK", "GBP", 4, 2),
        ("KR", "KRW", 6, 3),
        ("JP", "JPY", 8, 4),
        ("BE", "EUR", 10, 5),
    )
    scales = [
        {
            "jurisdiction_code": code,
            "currency": currency,
            "reported_income_values": [nominal_anchor],
            "source_period": "2025-01-01/2025-12-31",
            "nominal_monthly_anchor_minor_units": nominal_anchor,
            "anchor_selection": "test-only anchor",
            "simulation_monthly_anchor_cents": simulation_anchor,
            "anchor_status": "CALIBRATED",
            "scale_status": "CALIBRATED",
            "source_ids": ["TEST_ONLY_SOURCE"],
            "condition": "test-only users",
            "denominator": "test-only users",
            "cross_country_comparable": False,
        }
        for code, currency, nominal_anchor, simulation_anchor in scale_specs
    ]
    conversions = [
        {
            "jurisdiction_code": code,
            "source_currency": source_currency,
            "target_currency": "TST",
            "method": "FX",
            "rate_numerator": 1,
            "rate_denominator": 2,
            "rate_numerator_decimal": "1",
            "rate_denominator_decimal": "2",
            "rate_period_start": "2025-01-01",
            "rate_period_end": "2025-12-31",
            "target_price_period_start": "2025-01-01",
            "target_price_period_end": "2025-12-31",
            "estimand": "test-only comparable amount",
            "population_base": "test-only common population",
            "comparison_group": "test-only common basis",
            "aggregation_unit": "one test-only jurisdiction-seed total",
            "status": "CALIBRATED",
            "source_ids": ["TEST_ONLY_SOURCE"],
            "retrieved_on": "2026-08-24",
            "rounding_method": "nearest_minor_unit_half_away_from_zero",
            "rounding_scope": "AFTER_AGGREGATION",
            "notes": "test-only conversion",
            "conversion_id": None,
            "rate_binding_id": None,
        }
        for code, source_currency, _nominal, _simulation in scale_specs
    ]
    return {
        "lineage_status": "registered_profile_bundle",
        "snapshot": {
            "profile_bundle": {
                "jurisdiction_schema_version": 2,
                "sources": [
                    {
                        "id": "TEST_ONLY_SOURCE",
                        "publisher": "Test fixture",
                        "title": "Test-only monetary conversion",
                        "url": "https://example.invalid/test-only-source",
                        "period": "2025-01-01/2025-12-31",
                        "geography": "Test jurisdictions",
                        "supports": ["foreign_exchange_rate"],
                        "calibration_status": "CALIBRATED",
                        "retrieved_on": "2026-08-24",
                    }
                ],
                "money_scales": scales,
                "monetary_conversions": conversions,
            }
        },
    }


def _tree_snapshot(root: Path) -> tuple[tuple[str, bytes | None], ...]:
    return tuple(
        (
            path.relative_to(root).as_posix(),
            None if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
    )


class PolicyExportTests(unittest.TestCase):
    def test_complete_export_contains_tables_metadata_summary_and_valid_svgs(self) -> None:
        base = load_policy_config(CONFIG_PATH)
        spec = PolicyBatchSpec(
            seeds=(5, 6),
            days=1,
            player_count=16,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        profile = CountryProfile(code="XX")
        batch = run_policy_batch(
            spec,
            country_profiles=(profile,),
            harm_parameters=base.harm_parameters,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        sensitivity = run_sensitivity_analysis(
            spec,
            cases=(
                SensitivityCase(
                    "paid_random_rewards",
                    (0.0, 0.7),
                    expected_direction="increasing",
                ),
            ),
            country_profiles=(profile,),
            base_harm_parameters=base.harm_parameters,
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
                sensitivity,
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
            self.assertEqual(
                manifest["output_schema_version"], OUTPUT_SCHEMA_VERSION
            )
            self.assertEqual(
                manifest["manifest_schema_version"], MANIFEST_SCHEMA_VERSION
            )
            self.assertEqual(
                manifest["manifest_schema_sha256"], MANIFEST_SCHEMA_SHA256
            )
            self.assertTrue(manifest["synthetic_only"])
            self.assertFalse(manifest["empirical_validation_claimed"])
            self.assertEqual(
                manifest["artifact_files"], list(POLICY_ARTIFACT_FILENAMES)
            )
            self.assertEqual(len(manifest["scenarios"]), 7)
            self.assertEqual(manifest["batch"]["seeds"], [5, 6])
            self.assertEqual(
                manifest["config_sha256"],
                sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["config_file_sha256_observed_at_export"],
                manifest["config_sha256"],
            )
            self.assertEqual(
                manifest["effective_config_sha256"],
                _canonical_sha256(manifest["config_snapshot"]),
            )
            self.assertEqual(
                manifest["run_input_sha256"],
                _canonical_sha256(manifest["run_input_snapshot"]),
            )
            causal_design = manifest["causal_design"]
            self.assertEqual(causal_design["schema_version"], "1.0")
            self.assertEqual(
                causal_design["status"],
                "RETROSPECTIVE_SYNTHETIC",
            )
            self.assertFalse(causal_design["preregistered"])
            self.assertFalse(causal_design["planned_estimands"])
            self.assertFalse(causal_design["preregistered_estimands"])
            self.assertFalse(causal_design["campaign_ready"])
            self.assertTrue(causal_design["canonical_match"])
            self.assertEqual(
                causal_design["campaign_blockers"],
                [
                    "retrospective_synthetic_design",
                    "causal_design_not_preregistered",
                    "empirical_calibration_required",
                ],
            )
            self.assertEqual(len(causal_design["factor_names"]), 17)
            self.assertEqual(len(causal_design["factor_specs"]), 17)
            self.assertEqual(len(causal_design["scenario_matrix"]), 7)
            self.assertEqual(causal_design["contrast_count"], 49)
            self.assertEqual(len(causal_design["contrasts"]), 49)
            self.assertEqual(
                causal_design["contrast_scope"],
                "exhaustive_directed_pairwise_diagnostics",
            )
            self.assertEqual(causal_design["canonical_mismatches"], [])
            self.assertEqual(
                causal_design["scenario_matrix_sha256"],
                _canonical_sha256(
                    {
                        "schema_version": causal_design["schema_version"],
                        "factor_names": causal_design["factor_names"],
                        "scenario_matrix": causal_design["scenario_matrix"],
                    }
                ),
            )
            self.assertEqual(
                causal_design["scenario_matrix_sha256"],
                causal_design["canonical_scenario_matrix_sha256"],
            )
            self.assertEqual(
                causal_design["contrasts_sha256"],
                _canonical_sha256(
                    {
                        "schema_version": causal_design["schema_version"],
                        "factor_names": causal_design["factor_names"],
                        "contrast_scope": causal_design["contrast_scope"],
                        "contrasts": causal_design["contrasts"],
                    }
                ),
            )
            design_snapshot = {
                key: value
                for key, value in causal_design.items()
                if key
                not in {
                    "design_sha256",
                    "assessment_sha256",
                    "canonical_design_sha256",
                    "canonical_scenario_matrix_sha256",
                    "canonical_mismatches",
                    "run_input_sha256",
                }
            }
            self.assertEqual(
                causal_design["design_sha256"],
                _canonical_sha256(design_snapshot),
            )
            assessment_snapshot = {
                key: value
                for key, value in causal_design.items()
                if key
                not in {
                    "design_sha256",
                    "assessment_sha256",
                    "canonical_design_sha256",
                    "run_input_sha256",
                }
            }
            self.assertEqual(
                causal_design["assessment_sha256"],
                _canonical_sha256(assessment_snapshot),
            )
            self.assertEqual(
                causal_design["design_sha256"],
                causal_design["canonical_design_sha256"],
            )
            self.assertEqual(
                causal_design["run_input_sha256"],
                manifest["run_input_sha256"],
            )
            self.assertEqual(len(causal_design["canonical_design_sha256"]), 64)
            self.assertEqual(
                manifest["output_metric_contracts"]["run_input_lineage"][
                    "run_input_sha256"
                ],
                manifest["run_input_sha256"],
            )
            sensitivity_payload = manifest["sensitivity"]
            self.assertTrue(sensitivity_payload["run"])
            self.assertEqual(
                sensitivity_payload["execution_sha256"],
                _canonical_sha256(sensitivity_payload["execution_snapshot"]),
            )
            self.assertEqual(
                sensitivity_payload["execution_snapshot"]["cases"][0][
                    "parameter"
                ],
                "paid_random_rewards",
            )
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

    def test_custom_factor_matrix_exports_descriptively_but_blocks_campaign(
        self,
    ) -> None:
        base = load_policy_config(CONFIG_PATH)
        canonical = required_scenarios()
        custom_baseline = replace(
            canonical[0],
            mechanics=replace(
                canonical[0].mechanics,
                paid_random_rewards=0.69,
            ),
        )
        spec = PolicyBatchSpec(
            seeds=(37,),
            days=0,
            player_count=0,
            scenarios=(custom_baseline, *canonical[1:]),
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        profile = CountryProfile(code="XX")
        batch = run_policy_batch(
            spec,
            country_profiles=(profile,),
            harm_parameters=base.harm_parameters,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "custom-bundle"
            config = replace(
                base,
                batch=spec,
                output=PolicyOutputConfig(output, 8, False, False),
            )
            export_policy_batch(
                config,
                batch,
                None,
                config_path=CONFIG_PATH,
                repository_root=ROOT,
                created_utc="2026-01-01T00:00:00+00:00",
            )

            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            causal_design = manifest["causal_design"]
            self.assertFalse(causal_design["canonical_match"])
            self.assertFalse(causal_design["campaign_ready"])
            self.assertIn(
                "scenario_factor_matrix_not_canonical",
                causal_design["campaign_blockers"],
            )
            self.assertEqual(causal_design["contrast_count"], 49)
            self.assertEqual(len(causal_design["contrasts"]), 49)
            self.assertNotEqual(
                causal_design["scenario_matrix_sha256"],
                causal_design["canonical_scenario_matrix_sha256"],
            )
            self.assertNotEqual(
                causal_design["design_sha256"],
                causal_design["canonical_design_sha256"],
            )
            self.assertEqual(
                causal_design["run_input_sha256"],
                manifest["run_input_sha256"],
            )
            self.assertEqual(
                causal_design["canonical_mismatches"],
                [
                    {
                        "scenario_id": "baseline_f2p",
                        "factor_differences": [
                            {
                                "factor": "paid_random_rewards",
                                "reference_value": 0.7,
                                "comparison_value": 0.69,
                            }
                        ],
                    }
                ],
            )

    def test_execution_mismatches_fail_before_any_export_mutation(self) -> None:
        base = load_policy_config(CONFIG_PATH)
        spec = PolicyBatchSpec(
            seeds=(7,),
            days=0,
            player_count=0,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        profile = CountryProfile(code="XX")
        run_kwargs = {
            "harm_parameters": base.harm_parameters,
            "harm_weights": base.harm_weights,
            "opportunity_valuation": base.opportunity_valuation,
            "producer_assumptions": base.producer_assumptions,
            "epgc_policy": base.epgc_policy,
        }
        batch = run_policy_batch(
            spec,
            country_profiles=(profile,),
            **run_kwargs,
        )
        sensitivity_case = SensitivityCase(
            "paid_random_rewards",
            (0.0, 0.7),
        )
        sensitivity = run_sensitivity_analysis(
            spec,
            cases=(sensitivity_case,),
            country_profiles=(profile,),
            base_harm_parameters=base.harm_parameters,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        config = replace(
            base,
            batch=spec,
            output=PolicyOutputConfig(Path("unused"), 8, True, False),
        )
        different_spec = replace(spec, seeds=(8,))
        different_sensitivity_spec = run_sensitivity_analysis(
            different_spec,
            cases=(sensitivity_case,),
            country_profiles=(profile,),
            base_harm_parameters=base.harm_parameters,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        different_harm = replace(
            base.harm_parameters,
            affordable_spending_share=0.2,
        )
        different_sensitivity_inputs = run_sensitivity_analysis(
            spec,
            cases=(sensitivity_case,),
            country_profiles=(profile,),
            base_harm_parameters=different_harm,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        mismatches = (
            (
                replace(config, batch=different_spec),
                sensitivity,
                "configuration batch specification",
            ),
            (
                replace(config, harm_parameters=different_harm),
                sensitivity,
                "configuration model inputs",
            ),
            (
                config,
                different_sensitivity_spec,
                "different batch specifications",
            ),
            (
                config,
                different_sensitivity_inputs,
                "different resolved model inputs",
            ),
        )
        for candidate_config, candidate_sensitivity, message in mismatches:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "new-bundle"
                with self.assertRaisesRegex(ValueError, message):
                    export_policy_batch(
                        candidate_config,
                        batch,
                        candidate_sensitivity,
                        config_path=CONFIG_PATH,
                        repository_root=ROOT,
                        output_dir=output,
                        created_utc="2026-01-01T00:00:00+00:00",
                    )
                self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing-bundle"
            (output / "nested").mkdir(parents=True)
            (output / "sentinel.bin").write_bytes(b"unchanged")
            (output / "nested" / "keep.txt").write_text("keep", "utf-8")
            before = _tree_snapshot(output)
            with self.assertRaisesRegex(ValueError, "configuration model inputs"):
                export_policy_batch(
                    replace(config, harm_parameters=different_harm),
                    batch,
                    sensitivity,
                    config_path=CONFIG_PATH,
                    repository_root=ROOT,
                    output_dir=output,
                    created_utc="2026-01-01T00:00:00+00:00",
                )
            self.assertEqual(_tree_snapshot(output), before)

    def test_manifest_currency_contract_structure_mirror_fails_closed(self) -> None:
        profile_inputs = _comparable_profile_inputs_fixture()
        self.assertTrue(_money_conversion_structure_coherent(profile_inputs))
        self.assertFalse(_money_outputs_cross_country_comparable(profile_inputs))

        mutations = (
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ].pop(),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__("status", "ILLUSTRATIVE"),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__("comparison_group", "different basis"),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__("rate_numerator", 2),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__("source_ids", []),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__(
                "source_ids",
                ["TEST_ONLY_SOURCE", "TEST_ONLY_SOURCE"],
            ),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__("target_currency", "123"),
            lambda value: (
                value["snapshot"]["profile_bundle"]["money_scales"][
                    0
                ].__setitem__("jurisdiction_code", 7),
                value["snapshot"]["profile_bundle"]["monetary_conversions"][
                    0
                ].__setitem__("jurisdiction_code", 7),
            ),
            lambda value: (
                value["snapshot"]["profile_bundle"]["money_scales"][
                    0
                ].__setitem__("jurisdiction_code", ["AA"]),
                value["snapshot"]["profile_bundle"]["monetary_conversions"][
                    0
                ].__setitem__("jurisdiction_code", ["AA"]),
            ),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__("rate_numerator_decimal", "999"),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__("target_price_period_end", "2099-12-31"),
            lambda value: value["snapshot"]["profile_bundle"][
                "monetary_conversions"
            ][0].__setitem__("retrieved_on", "not-a-date"),
            lambda value: (
                value["snapshot"]["profile_bundle"]["monetary_conversions"][
                    0
                ].__setitem__("retrieved_on", "2025-06-01"),
                value["snapshot"]["profile_bundle"]["sources"][0].__setitem__(
                    "retrieved_on", "2025-06-01"
                ),
            ),
            lambda value: value["snapshot"]["profile_bundle"]["sources"][
                0
            ].__setitem__("supports", []),
            lambda value: value["snapshot"]["profile_bundle"]["sources"][
                0
            ].__setitem__("supports", [[]]),
            lambda value: value["snapshot"]["profile_bundle"]["sources"][
                0
            ].__setitem__("period", "different period"),
            lambda value: value["snapshot"]["profile_bundle"]["money_scales"][
                0
            ].__setitem__("scale_status", "ILLUSTRATIVE"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                changed = deepcopy(profile_inputs)
                mutate(changed)
                self.assertFalse(
                    _money_conversion_structure_coherent(changed)
                )

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
                lineage["monetary_conversion_summary"],
                {
                    "count": 0,
                    "methods": [],
                    "source_currencies": [],
                    "target_currencies": [],
                    "rate_period_starts": [],
                    "rate_period_ends": [],
                    "target_price_period_starts": [],
                    "target_price_period_ends": [],
                    "estimands": [],
                    "population_bases": [],
                    "comparison_groups": [],
                    "retrieval_dates": [],
                    "rounding_scopes": [],
                    "aggregation_units": [],
                    "status_counts": {},
                },
            )
            self.assertEqual(lineage["snapshot"]["schema_version"], 4)
            self.assertEqual(
                lineage["snapshot"]["profile_bundle"]["monetary_conversions"],
                [],
            )
            self.assertEqual(
                manifest["jurisdictions_sha256"],
                lineage["jurisdictions"]["sha256"],
            )
            self.assertEqual(
                manifest["source_registry_sha256"],
                lineage["source_registry"]["sha256"],
            )
            self.assertEqual(
                manifest["population_readiness"]["schema_version"],
                "1.0",
            )
            self.assertEqual(
                manifest["population_readiness"]["typed_assessment"],
                lineage["population_evidence_assessment"],
            )
            population_assessment = lineage["population_evidence_assessment"]
            population_gate = manifest["population_readiness"]["manifest_gate"]
            self.assertEqual(
                population_gate,
                {
                    field: population_assessment[field] is True
                    for field in population_gate
                },
            )
            self.assertTrue(population_gate["structure_coherent"])
            self.assertFalse(population_gate["public_population_comparability"])

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
