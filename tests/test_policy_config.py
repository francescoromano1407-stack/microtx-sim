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
        self.assertEqual(config.provenance_status, "illustrative")
        self.assertIsNotNone(config.population)
        assert config.population is not None
        self.assertIs(
            config.population.mode,
            PopulationExecutionMode.PROJECTED_V1,
        )
        self.assertIsNotNone(config.analysis_plan)
        self.assertTrue(config.output.include_player_rows)
        self.assertEqual(config.batch.player_count, 50_000)

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
