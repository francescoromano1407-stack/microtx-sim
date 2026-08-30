from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microtx_sim.cli import _policy_batch
from microtx_sim.policy_config import (
    AnalysisPlanSelection,
    PolicyConfigurationError,
    PolicyRunPurpose,
    PolicySimulationLayer,
    UncertaintyAvailability,
    load_policy_config,
)
from microtx_sim.config import PopulationExecutionMode


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "policy_prototype.toml"
CAMPAIGN_CONFIG = ROOT / "configs" / "policy_campaign.toml"


class PolicyConfigTests(unittest.TestCase):
    def test_checked_in_campaign_candidate_requires_projected_population(self) -> None:
        config = load_policy_config(CAMPAIGN_CONFIG)
        self.assertIs(config.run_purpose, PolicyRunPurpose.CAMPAIGN)
        self.assertTrue(config.full_campaign_config)
        self.assertEqual(config.provenance_status, "illustrative")
        self.assertIsNotNone(config.population)
        assert config.population is not None
        self.assertIs(
            config.population.mode,
            PopulationExecutionMode.PROJECTED_V1,
        )
        self.assertIsNotNone(config.analysis_plan)
        assert config.analysis_plan is not None
        self.assertEqual(
            config.analysis_plan.parent_plan_id,
            "illustrative.prospective.composite-harm.baseline-vs-safe.v2",
        )
        self.assertTrue(config.output.include_player_rows)
        self.assertEqual(config.batch.player_count, 50_000)
        self.assertEqual(config.batch.days, 14)
        self.assertEqual(len(config.batch.seeds), 150)
        self.assertEqual(config.batch.seeds, tuple(sorted(config.batch.seeds)))
        self.assertTrue({101, 202, 303}.issubset(config.batch.seeds))
        assert config.campaign is not None
        self.assertFalse(config.campaign.allow_synthetic)
        self.assertTrue(config.campaign.fail_closed)
        self.assertFalse(config.campaign.campaign_ready)
        self.assertIs(
            config.campaign.simulation_layer,
            PolicySimulationLayer.POLICY_ORCHESTRATOR,
        )
        assert config.uncertainty is not None
        self.assertEqual(config.uncertainty.minimum_retained_seeds, 100)
        self.assertIs(
            config.uncertainty.population_uncertainty,
            UncertaintyAvailability.UNQUANTIFIED,
        )
        self.assertIs(
            config.uncertainty.monetary_rate_uncertainty,
            UncertaintyAvailability.UNQUANTIFIED,
        )
        assert config.convergence is not None
        self.assertEqual(config.convergence.block_size, 50)
        self.assertEqual(config.convergence.required_status, "CONVERGED")
        assert config.population_contract is not None
        self.assertTrue(
            config.population_contract.require_per_seed_assignment_identity
        )
        self.assertEqual(
            config.population_contract.weight_application,
            "WITHIN_SEED_BEFORE_CROSS_SEED_AGGREGATION",
        )
        assert config.monetary_contract is not None
        self.assertEqual(config.monetary_contract.target_currency, "EUR")
        self.assertEqual(
            config.monetary_contract.target_minor_unit_name,
            "euro cent",
        )
        self.assertFalse(
            config.monetary_contract.observed_real_world_spending
        )
        assert config.output_contract is not None
        self.assertEqual(len(config.output_contract.expected_artifacts), 25)
        assert config.ledger is not None
        self.assertTrue(config.ledger.persistent)
        self.assertFalse(config.ledger.temporary)
        assert config.execution_receipt is not None
        self.assertTrue(config.execution_receipt.require_clean_working_tree)

    def test_full_campaign_sections_are_collectively_required(self) -> None:
        original = CAMPAIGN_CONFIG.read_text("utf-8")
        incomplete = original.split("\n[execution_receipt]\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete-campaign.toml"
            path.write_text(incomplete, "utf-8")
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "missing required sections: execution_receipt",
            ):
                load_policy_config(path)

    def test_full_campaign_seed_design_is_strict_and_large_enough(self) -> None:
        original = CAMPAIGN_CONFIG.read_text("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unsorted = root / "unsorted-seeds.toml"
            unsorted.write_text(
                original.replace(
                    "seeds = [101, 102, 103",
                    "seeds = [102, 101, 103",
                    1,
                ),
                "utf-8",
            )
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "strictly ascending",
            ):
                load_policy_config(unsorted)

            too_few = root / "too-few-seeds.toml"
            lines = original.splitlines()
            lines = [
                "seeds = [101, 202, 303]"
                if line.startswith("seeds = [")
                else line
                for line in lines
            ]
            too_few.write_text("\n".join(lines) + "\n", "utf-8")
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "at least 100 seeds",
            ):
                load_policy_config(too_few)

    def test_full_campaign_fail_closed_invariants_are_not_overridable(self) -> None:
        original = CAMPAIGN_CONFIG.read_text("utf-8")
        mutations = (
            (
                "allow_synthetic = false",
                "allow_synthetic = true",
                "allow_synthetic = false",
            ),
            (
                "campaign_ready = false",
                "campaign_ready = true",
                "campaign_ready must remain false",
            ),
            (
                'backend = "sqlite"',
                'backend = "memory"',
                "SQLite ledger",
            ),
            (
                "temporary = false",
                "temporary = true",
                "cannot be temporary",
            ),
            (
                "observed_real_world_spending = false",
                "observed_real_world_spending = true",
                "not observed spending",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (old, new, message) in enumerate(mutations):
                with self.subTest(mutation=new):
                    path = root / f"unsafe-{index}.toml"
                    path.write_text(original.replace(old, new, 1), "utf-8")
                    with self.assertRaisesRegex(
                        PolicyConfigurationError,
                        message,
                    ):
                        load_policy_config(path)

    def test_full_campaign_tables_remain_strict(self) -> None:
        original = CAMPAIGN_CONFIG.read_text("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = root / "unknown-campaign-key.toml"
            unknown.write_text(
                original.replace(
                    'primary_estimand_id = "primary.composite-harm.baseline-vs-safe.v1"',
                    'primary_estimand_id = "primary.composite-harm.baseline-vs-safe.v1"\nunknown = true',
                    1,
                ),
                "utf-8",
            )
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "campaign keys differ",
            ):
                load_policy_config(unknown)

            incomplete_identity = root / "incomplete-plan-identity.toml"
            incomplete_identity.write_text(
                original.replace(
                    'expected_plan_id = "illustrative.prospective.composite-harm.baseline-vs-safe.v3"\n',
                    "",
                    1,
                ),
                "utf-8",
            )
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "identities must be supplied together",
            ):
                load_policy_config(incomplete_identity)

    def test_checked_in_policy_config_is_strict_and_complete(self) -> None:
        config = load_policy_config(CONFIG)
        self.assertEqual(config.provenance_status, "synthetic")
        self.assertEqual(len(config.batch.scenarios), 7)
        self.assertEqual(config.batch.seeds, (101, 202, 303))
        self.assertEqual(config.batch.player_count, 1000)
        self.assertIs(config.run_purpose, PolicyRunPurpose.DEVELOPMENT)
        self.assertNotIn("run_purpose", config.batch.snapshot())
        self.assertFalse(
            any(item.mechanics.personalized_offers for item in config.batch.scenarios)
        )
        self.assertGreater(config.epgc_policy.maximum_budget_cents, 0)

    def test_unknown_or_missing_toml_keys_are_rejected(self) -> None:
        original = CONFIG.read_text("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.toml"
            unknown.write_text(
                original.replace(
                    'run_sensitivity = true',
                    'run_sensitivity = true\nunknown_option = 1',
                ),
                "utf-8",
            )
            with self.assertRaises(PolicyConfigurationError):
                load_policy_config(unknown)
            missing = Path(directory) / "missing.toml"
            missing.write_text(
                original.replace('histogram_bins = 20\n', ''), "utf-8"
            )
            with self.assertRaises(PolicyConfigurationError):
                load_policy_config(missing)

    def test_population_projection_is_strict_and_opt_in(self) -> None:
        original = CONFIG.read_text("utf-8")
        self.assertIsNone(load_policy_config(CONFIG).population)
        population = """

[population]
mode = "projected_v1"
design_bundle_path = "inputs/design.toml"
runtime_mapping_bundle_path = "inputs/runtime-mapping.toml"
adapter_id = "policy.population.v1"
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured_path = root / "projected.toml"
            configured_path.write_text(original + population, "utf-8")
            configured = load_policy_config(configured_path)
            self.assertIsNotNone(configured.population)
            assert configured.population is not None
            self.assertIs(
                configured.population.mode,
                PopulationExecutionMode.PROJECTED_V1,
            )
            self.assertEqual(
                configured.population.design_bundle_path,
                (root / "inputs" / "design.toml").resolve(),
            )
            self.assertEqual(
                configured.population.runtime_mapping_bundle_path,
                (root / "inputs" / "runtime-mapping.toml").resolve(),
            )

            malformed = root / "malformed.toml"
            malformed.write_text(
                original
                + population.replace(
                    'adapter_id = "policy.population.v1"',
                    'adapter_id = "policy.population.v1"\nunknown = true',
                ),
                "utf-8",
            )
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "population keys differ",
            ):
                load_policy_config(malformed)

    def test_campaign_run_purpose_requires_population_plan_rows_and_cohort(
        self,
    ) -> None:
        original = CONFIG.read_text("utf-8")
        campaign = original.replace(
            "[meta]\n",
            '[meta]\nrun_purpose = "campaign"\n',
            1,
        ).replace(
            'provenance_status = "synthetic"',
            'provenance_status = "calibrated"',
            1,
        )
        population = """

[population]
mode = "projected_v1"
design_bundle_path = "inputs/design.toml"
runtime_mapping_bundle_path = "inputs/runtime-mapping.toml"
adapter_id = "policy.population.v1"
"""
        analysis_plan = """

[analysis_plan]
plan_path = "inputs/prospective-analysis-plan.json"
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            valid_path = root / "campaign.toml"
            valid_path.write_text(
                campaign + population + analysis_plan,
                "utf-8",
            )
            configured = load_policy_config(valid_path)
            self.assertIs(configured.run_purpose, PolicyRunPurpose.CAMPAIGN)
            self.assertNotIn("run_purpose", configured.batch.snapshot())

            missing_population = root / "missing-population.toml"
            missing_population.write_text(campaign, "utf-8")
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                r"campaign policy runs require \[population\]",
            ):
                load_policy_config(missing_population)

            missing_plan = root / "missing-plan.toml"
            missing_plan.write_text(campaign + population, "utf-8")
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                r"campaign policy runs require an \[analysis_plan\]",
            ):
                load_policy_config(missing_plan)

            empty_cohort = root / "empty-cohort.toml"
            empty_cohort.write_text(
                campaign.replace("player_count = 1000", "player_count = 0")
                + population
                + analysis_plan,
                "utf-8",
            )
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "positive player cohort",
            ):
                load_policy_config(empty_cohort)

            missing_rows = root / "missing-rows.toml"
            missing_rows.write_text(
                campaign.replace(
                    "include_player_rows = true",
                    "include_player_rows = false",
                )
                + population
                + analysis_plan,
                "utf-8",
            )
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                r"output\.include_player_rows = true",
            ):
                load_policy_config(missing_rows)

            invalid_purpose = root / "invalid-purpose.toml"
            invalid_purpose.write_text(
                campaign.replace(
                    'run_purpose = "campaign"',
                    'run_purpose = "production-ish"',
                ),
                "utf-8",
            )
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "meta.run_purpose",
            ):
                load_policy_config(invalid_purpose)

    def test_analysis_plan_selection_is_strict_and_opt_in(self) -> None:
        original = CONFIG.read_text("utf-8")
        self.assertIsNone(load_policy_config(CONFIG).analysis_plan)
        population = """

[population]
mode = "projected_v1"
design_bundle_path = "inputs/design.toml"
runtime_mapping_bundle_path = "inputs/runtime-mapping.toml"
adapter_id = "policy.population.v1"
"""
        selection = """

[analysis_plan]
plan_path = "inputs/prospective-analysis-plan.json"
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured_path = root / "planned.toml"
            configured_path.write_text(
                original + population + selection,
                "utf-8",
            )

            configured = load_policy_config(configured_path)

            self.assertEqual(
                configured.analysis_plan,
                AnalysisPlanSelection(
                    root / "inputs" / "prospective-analysis-plan.json"
                ),
            )
            assert configured.analysis_plan is not None
            self.assertEqual(
                configured.analysis_plan.snapshot(),
                {
                    "plan_path": str(
                        root / "inputs" / "prospective-analysis-plan.json"
                    )
                },
            )

            invalid_sections = (
                "\n[analysis_plan]\n",
                '\n[analysis_plan]\nplan_path = ""\n',
                '\n[analysis_plan]\nplan_path = 1\n',
                (
                    '\n[analysis_plan]\nplan_path = "plan.json"\n'
                    "unknown = true\n"
                ),
            )
            for index, invalid in enumerate(invalid_sections):
                with self.subTest(index=index):
                    malformed = root / f"malformed-plan-{index}.toml"
                    malformed.write_text(
                        original + population + invalid,
                        "utf-8",
                    )
                    with self.assertRaises(PolicyConfigurationError):
                        load_policy_config(malformed)

            missing_population = root / "missing-population.toml"
            missing_population.write_text(original + selection, "utf-8")
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                "analysis_plan requires projected population execution",
            ):
                load_policy_config(missing_population)

    def test_analysis_plan_section_must_be_a_table(self) -> None:
        original = CONFIG.read_text("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory) / "malformed-plan.toml"
            malformed.write_text(
                'analysis_plan = "plan.json"\n\n' + original,
                "utf-8",
            )
            with self.assertRaisesRegex(
                PolicyConfigurationError,
                r"\[analysis_plan\] must be a TOML table",
            ):
                load_policy_config(malformed)

    def test_analysis_plan_requires_player_rows_before_execution(self) -> None:
        original = CONFIG.read_text("utf-8").replace(
            "include_player_rows = true",
            "include_player_rows = false",
        )
        population_and_plan = """

[population]
mode = "projected_v1"
design_bundle_path = "inputs/design.toml"
runtime_mapping_bundle_path = "inputs/runtime-mapping.toml"
adapter_id = "policy.population.v1"

[analysis_plan]
plan_path = "inputs/prospective-analysis-plan.json"
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "plan-without-player-rows.toml"
            config_path.write_text(
                original + population_and_plan,
                "utf-8",
            )
            with patch("microtx_sim.cli.run_policy_batch") as execute:
                with self.assertRaisesRegex(
                    PolicyConfigurationError,
                    r"analysis_plan requires output\.include_player_rows = true",
                ):
                    _policy_batch(
                        config_path,
                        output=Path(directory) / "never-created",
                        run_sensitivity=False,
                        command=("microtx-sim", "policy-batch"),
                    )
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
