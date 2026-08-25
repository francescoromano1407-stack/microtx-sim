from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import unittest

from microtx_sim.causal.batch import (
    PolicyBatchSpec,
    REPEATED_SEED_METRIC_STEMS,
    run_policy_batch,
)
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.data.profiles import load_profile_bundle
from microtx_sim.outputs.manifest import build_run_manifest
from microtx_sim.outputs.metric_contracts import (
    METRIC_CONTRACT_SCHEMA_VERSION,
    OUTPUT_METRIC_CONTRACTS,
    MetricContractValidationError,
    MetricRole,
    metric_contract_registry_sha256,
    metric_contract_registry_snapshot,
    validate_metric_contract_registry,
    validate_metric_contracts_for_campaign,
)
from microtx_sim.outputs.schema import OUTPUT_SCHEMA_VERSION, TABLE_COLUMNS
from microtx_sim.policy_config import load_policy_config
from microtx_sim.types import ProvenanceStatus


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "policy_prototype.toml"


class OutputMetricContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_config = load_policy_config(CONFIG_PATH)
        cls.bundle = load_profile_bundle()
        cls.spec = PolicyBatchSpec(
            seeds=(73,),
            days=0,
            player_count=0,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        cls.config = replace(base_config, batch=cls.spec)
        cls.batch = run_policy_batch(
            cls.spec,
            profile_bundle=cls.bundle,
            harm_parameters=cls.config.harm_parameters,
            harm_weights=cls.config.harm_weights,
            opportunity_valuation=cls.config.opportunity_valuation,
            producer_assumptions=cls.config.producer_assumptions,
            epgc_policy=cls.config.epgc_policy,
        )

    def test_registry_exhaustively_matches_all_six_table_schemas(self) -> None:
        expected = tuple(
            (artifact, column)
            for artifact, columns in TABLE_COLUMNS.items()
            for column in columns
        )
        self.assertEqual(len(expected), 220)
        self.assertEqual(tuple(OUTPUT_METRIC_CONTRACTS), expected)
        self.assertEqual(
            Counter(contract.role for contract in OUTPUT_METRIC_CONTRACTS.values()),
            {
                MetricRole.IDENTIFIER: 15,
                MetricRole.DESIGN: 9,
                MetricRole.DERIVED: 196,
            },
        )
        validate_metric_contract_registry(OUTPUT_METRIC_CONTRACTS)

    def test_every_derived_metric_has_truthful_unpromoted_provenance(self) -> None:
        derived = tuple(
            contract
            for contract in OUTPUT_METRIC_CONTRACTS.values()
            if contract.role is MetricRole.DERIVED
        )
        self.assertEqual(len(derived), 196)
        for contract in derived:
            with self.subTest(contract=contract.contract_id):
                self.assertTrue(contract.recipe_id)
                self.assertTrue(contract.recipe_version)
                self.assertTrue(contract.source_version)
                self.assertTrue(contract.formula)
                self.assertTrue(contract.inputs)
                self.assertTrue(contract.lineage_ids)
                self.assertIs(contract.status, ProvenanceStatus.SYNTHETIC)
                self.assertIsNone(contract.source_retrieved_on)

    def test_every_contract_links_the_authoritative_run_input_digest(self) -> None:
        for contract in OUTPUT_METRIC_CONTRACTS.values():
            with self.subTest(contract=contract.contract_id):
                self.assertIn("run_input_sha256", contract.lineage_ids)

    def test_only_sensitivity_contracts_link_the_sensitivity_execution_digest(
        self,
    ) -> None:
        for contract in OUTPUT_METRIC_CONTRACTS.values():
            with self.subTest(contract=contract.contract_id):
                if contract.artifact == "sensitivity.csv":
                    self.assertIn(
                        "sensitivity.execution_sha256",
                        contract.lineage_ids,
                    )
                else:
                    self.assertNotIn(
                        "sensitivity.execution_sha256",
                        contract.lineage_ids,
                    )

    def test_registry_digest_and_snapshot_are_frozen(self) -> None:
        self.assertEqual(METRIC_CONTRACT_SCHEMA_VERSION, "1.0")
        self.assertEqual(OUTPUT_SCHEMA_VERSION, "2.0")
        self.assertEqual(
            metric_contract_registry_sha256(),
            "ddda63b154eddc572213ee0da76303de972a26fed7b8e348b395831bdc56975a",
        )
        snapshot = metric_contract_registry_snapshot()
        self.assertEqual(len(snapshot), 220)
        self.assertEqual(
            [(item["artifact"], item["column"]) for item in snapshot],
            list(OUTPUT_METRIC_CONTRACTS),
        )

    def test_summary_stems_share_one_public_contract_and_correct_unit_algebra(self) -> None:
        summary_columns = set(TABLE_COLUMNS["scenario_summary.csv"][5:])
        expected = {
            f"{stem}_{suffix}"
            for stem in REPEATED_SEED_METRIC_STEMS
            for suffix in ("mean", "variance", "sd", "ci95_low", "ci95_high")
        }
        self.assertEqual(len(REPEATED_SEED_METRIC_STEMS), 23)
        self.assertEqual(summary_columns, expected)
        for stem in REPEATED_SEED_METRIC_STEMS:
            base = OUTPUT_METRIC_CONTRACTS[("seed_results.csv", stem)].unit
            variance = OUTPUT_METRIC_CONTRACTS[
                ("scenario_summary.csv", f"{stem}_variance")
            ]
            self.assertEqual(variance.unit, base.squared())
            self.assertIn("ddof=1", variance.uncertainty_semantics or "")
            for suffix in ("mean", "sd", "ci95_low", "ci95_high"):
                contract = OUTPUT_METRIC_CONTRACTS[
                    ("scenario_summary.csv", f"{stem}_{suffix}")
                ]
                self.assertEqual(contract.unit, base)

    def test_seed_contracts_publish_the_strict_unsigned_64_bit_domain(self) -> None:
        for artifact in (
            "seed_results.csv",
            "epgc_financing.csv",
            "player_outcomes.csv",
        ):
            with self.subTest(artifact=artifact):
                contract = OUTPUT_METRIC_CONTRACTS[(artifact, "seed")]
                self.assertIn("[0, 2**64 - 1]", contract.range_semantics)
                self.assertIn("booleans", contract.missing_value_semantics)
                self.assertIn(
                    "modulo-wrapped aliases",
                    contract.missing_value_semantics,
                )

    def test_one_seed_summary_has_declared_zero_width_monte_carlo_interval(self) -> None:
        row = self.batch.scenario_rows()[0]
        self.assertEqual(set(row), set(TABLE_COLUMNS["scenario_summary.csv"]))
        for stem in REPEATED_SEED_METRIC_STEMS:
            with self.subTest(stem=stem):
                self.assertEqual(row[f"{stem}_variance"], 0.0)
                self.assertEqual(row[f"{stem}_sd"], 0.0)
                self.assertEqual(row[f"{stem}_ci95_low"], row[f"{stem}_mean"])
                self.assertEqual(row[f"{stem}_ci95_high"], row[f"{stem}_mean"])

    def test_semantic_traps_are_explicit(self) -> None:
        player_variance = OUTPUT_METRIC_CONTRACTS[
            ("seed_results.csv", "harm_variance_players")
        ]
        self.assertIn("ddof=0", player_variance.uncertainty_semantics or "")
        high_risk = OUTPUT_METRIC_CONTRACTS[
            ("seed_results.csv", "high_risk_mean_age")
        ]
        self.assertIn("encoded as 0", high_risk.missing_value_semantics.lower())
        high_risk_share = OUTPUT_METRIC_CONTRACTS[
            ("seed_results.csv", "high_risk_share")
        ]
        self.assertEqual(
            high_risk_share.population_base,
            "one synthetic scenario-seed cohort",
        )
        self.assertIn("all players", high_risk_share.condition)
        producer_cost = OUTPUT_METRIC_CONTRACTS[
            ("seed_results.csv", "producer_cost_cents")
        ]
        self.assertNotIn(
            "reduce to 0",
            producer_cost.missing_value_semantics,
        )
        self.assertGreater(self.batch.seed_rows()[0]["producer_cost_cents"], 0)
        opportunity = OUTPUT_METRIC_CONTRACTS[
            (
                "opportunity_cost_decomposition.csv",
                "monetary_proxy_cents",
            )
        ]
        self.assertTrue(opportunity.nullable)
        self.assertIn("blank", opportunity.missing_value_semantics.lower())
        self.assertEqual(
            opportunity.inputs,
            ("WelfareHarmResult.opportunity_cost_proxy_cents",),
        )
        self.assertEqual(
            opportunity.population_base,
            "configured scenario-seed cohort totals",
        )

    def test_opportunity_contracts_follow_their_distinct_reductions(self) -> None:
        artifact = "opportunity_cost_decomposition.csv"
        minutes = OUTPUT_METRIC_CONTRACTS[(artifact, "mean_minutes")]
        burden = OUTPUT_METRIC_CONTRACTS[(artifact, "mean_burden")]
        monetary = OUTPUT_METRIC_CONTRACTS[(artifact, "monetary_proxy_cents")]
        self.assertIn(
            "opportunity_cost_decomposition.csv:component mean_minutes rows",
            minutes.inputs,
        )
        self.assertEqual(
            burden.inputs,
            ("WelfareHarmResult.component_scores[:, S/E/F/OC]",),
        )
        self.assertNotEqual(minutes.inputs, burden.inputs)
        self.assertNotEqual(burden.inputs, monetary.inputs)
        self.assertIn("mean over scenario seeds", monetary.formula)

    def test_epgc_recipes_name_upstream_equation_inputs(self) -> None:
        minimum = OUTPUT_METRIC_CONTRACTS[
            ("epgc_financing.csv", "minimum_public_contribution_cents")
        ]
        self.assertEqual(
            minimum.implementation,
            "microtx_sim.funding.epgc.evaluate_epgc",
        )
        self.assertEqual(
            minimum.inputs,
            (
                "EPGCResult.development_cost_cents",
                "EPGCResult.maintenance_cost_cents",
                "EPGCResult.fixed_price_revenue_cents",
                "EPGCResult.institutional_licensing_revenue_cents",
                "EPGCResult.non_targeted_sponsorship_revenue_cents",
            ),
        )
        feasibility = OUTPUT_METRIC_CONTRACTS[
            ("epgc_financing.csv", "feasible_under_budget_cap")
        ]
        self.assertEqual(
            feasibility.inputs,
            (
                "EPGCResult.minimum_public_contribution_cents",
                "EPGCResult.maximum_budget_cents",
            ),
        )

    def test_paired_and_case_wide_recipes_name_their_actual_inputs(self) -> None:
        paired = OUTPUT_METRIC_CONTRACTS[
            ("seed_results.csv", "mean_harm_effect_vs_safe")
        ]
        self.assertEqual(
            paired.inputs,
            (
                "PolicyScenarioResult.composite_harm[scenario]",
                "PolicyScenarioResult.composite_harm[safe-reference]",
                "PolicyBatchSpec.reference_scenario",
            ),
        )
        self.assertEqual(
            paired.implementation,
            "microtx_sim.causal.batch.run_policy_batch",
        )
        unstable = OUTPUT_METRIC_CONTRACTS[
            ("sensitivity.csv", "unstable")
        ]
        self.assertIn("all parameter levels", unstable.population_base)
        self.assertIn("instability_cv_threshold", unstable.inputs)
        self.assertIn("any level", unstable.formula)
        monotonic = OUTPUT_METRIC_CONTRACTS[
            ("sensitivity.csv", "monotonic_observed")
        ]
        self.assertIn("1e-12", monotonic.formula)
        cv = OUTPUT_METRIC_CONTRACTS[
            ("sensitivity.csv", "harm_coefficient_of_variation")
        ]
        self.assertIn("abs(mean_harm) > 1e-12", cv.formula)
        self.assertIn(
            "microtx_sim.analysis.sensitivity._CV_ZERO_MEAN_TOLERANCE=1e-12",
            cv.inputs,
        )

    def test_source_locators_name_the_functions_that_compute_the_values(self) -> None:
        self.assertEqual(
            OUTPUT_METRIC_CONTRACTS[
                ("seed_results.csv", "cohort_digest")
            ].implementation,
            "microtx_sim.causal.batch._cohort_digest",
        )
        for column in ("total_revenue_cents", "producer_profit_cents"):
            with self.subTest(column=column):
                self.assertEqual(
                    OUTPUT_METRIC_CONTRACTS[
                        ("seed_results.csv", column)
                    ].implementation,
                    "microtx_sim.simulation.policy_orchestrator.run_policy_scenario",
                )

    def test_current_registry_fails_campaign_promotion_closed(self) -> None:
        with self.assertRaisesRegex(
            MetricContractValidationError,
            "196 derived output contracts are not CALIBRATED",
        ):
            validate_metric_contracts_for_campaign(
                configuration_status="CALIBRATED",
                profile_lineage_status="registered_profile_bundle",
                profile_dependencies_calibrated=True,
                run_source_retrieved_on=self.bundle.source_retrieved_on,
                monetary_outputs_cross_country_comparable=True,
            )

    def test_manifest_embeds_exact_registry_and_run_source_lineage(self) -> None:
        manifest = build_run_manifest(
            self.config,
            self.batch,
            config_path=CONFIG_PATH,
            repository_root=ROOT,
            created_utc="2026-01-01T00:00:00+00:00",
        )
        payload = manifest["output_metric_contracts"]
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload["contract_count"], 220)
        self.assertEqual(payload["registry_sha256"], metric_contract_registry_sha256())
        self.assertEqual(payload["contracts"], metric_contract_registry_snapshot())
        self.assertEqual(
            payload["status_counts"],
            {ProvenanceStatus.SYNTHETIC.value: 220},
        )
        self.assertFalse(payload["campaign_ready"])
        self.assertIn(
            "profile input dependencies are not all CALIBRATED",
            payload["campaign_blockers"],
        )
        lineage = payload["run_input_lineage"]
        self.assertIsInstance(lineage, dict)
        assert isinstance(lineage, dict)
        self.assertEqual(
            lineage["profile_input_fingerprint_sha256"],
            self.batch.profile_input_lineage.fingerprint_sha256,
        )
        self.assertEqual(lineage["profile_source_retrieved_on"], "2026-08-24")
        self.assertFalse(lineage["profile_dependencies_calibrated"])
        self.assertFalse(lineage["monetary_outputs_cross_country_comparable"])
        seed_contract = manifest["random_stream_contract"]
        self.assertEqual(
            seed_contract["root_seed"]["maximum_decimal"],
            "18446744073709551615",
        )
        self.assertEqual(
            seed_contract["batch_seed_order"],
            "unique ascending numeric order",
        )
        self.assertEqual(
            manifest["batch"]["seed_decimal_strings"],
            [str(seed) for seed in self.config.batch.seeds],
        )


if __name__ == "__main__":
    unittest.main()
