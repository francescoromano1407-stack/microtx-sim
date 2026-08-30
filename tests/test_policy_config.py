from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from microtx_sim.cli import _policy_batch
from microtx_sim.policy_config import (
    AnalysisPlanSelection,
    EXPLORATORY_ARTIFACT_NAMESPACE,
    EXPLORATORY_ESTIMAND_INTERPRETATION,
    EXPLORATORY_EXECUTION_KIND,
    EXPLORATORY_INTERNAL_MONETARY_UNIT,
    EXPLORATORY_MONETARY_AMOUNT_SEMANTICS,
    EXPLORATORY_PLAN_ID,
    EXPLORATORY_POPULATION_BASIS,
    EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE,
    EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE,
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
EXPLORATORY_CONFIG = ROOT / "configs" / "policy_exploratory_synthetic.toml"


def _without_toml_tables(text: str, *table_names: str) -> str:
    omitted = set(table_names)
    output: list[str] = []
    skipping = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            skipping = stripped[1:-1] in omitted
        if not skipping:
            output.append(line)
    return "\n".join(output).rstrip() + "\n"


def _toml_table(text: str, table_name: str) -> str:
    output: list[str] = []
    copying = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            if copying and current != table_name:
                break
            copying = current == table_name
        if copying:
            output.append(line)
    if not output:
        raise AssertionError(f"missing TOML table {table_name}")
    return "\n".join(output) + "\n"


def _exploratory_config_text() -> str:
    original = CAMPAIGN_CONFIG.read_text("utf-8")
    selected = _without_toml_tables(
        original,
        "campaign",
        "output_contract",
        "execution_receipt",
    )
    selected = selected.replace(
        'run_purpose = "campaign"\nfull_campaign_config = true',
        (
            'run_purpose = "exploratory"\n'
            "full_campaign_config = false\n"
            "full_exploratory_config = true"
        ),
        1,
    )
    selected = selected.replace(
        'output_dir = "artifacts/policy_campaign_BLOCKED"',
        'output_dir = "artifacts/policy_exploratory_synthetic"',
        1,
    ).replace(
        "../artifacts/policy_campaign_BLOCKED/campaign-ledger.sqlite3",
        (
            "../artifacts/policy_exploratory_synthetic/"
            "exploratory-ledger.sqlite3"
        ),
        1,
    )
    return selected + """
[exploratory]
exploratory_plan_path = "../inputs/exploratory-synthetic-analysis-plan-v1.json"
exploratory_plan_id = "illustrative.exploratory.synthetic.composite-harm.baseline-vs-safe.v1"
exploratory_plan_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
artifact_namespace = "policy_exploratory_synthetic"
execution_kind = "COMPUTATIONAL_SIMULATION"
population_basis = "ILLUSTRATIVE_NON_EMPIRICAL"
estimand_interpretation = "CONDITIONAL_ON_MODEL_ASSUMPTIONS"
monetary_amount_semantics = "SYNTHETIC_MODEL_EQUIVALENT_NOT_OBSERVED_SPENDING"
unweighted_output_role = "DIAGNOSTIC_ONLY"
internal_monetary_unit = "simulation_cents"
raw_internal_unit_output_role = "DIAGNOSTIC_ONLY_NOT_A_CROSS_COUNTRY_MONETARY_RESULT"
execution_enabled = true
allow_synthetic = true
campaign_ready = false
production_campaign = false
empirical_claims = false
population_inference_claims = false
causal_claims = false
generalisation_claims = false
identical_pretreatment_cohorts = true
identical_population_weights_across_scenarios = true
primary_estimand_id = "primary.composite-harm.baseline-vs-safe.v1"

[exploratory_checkpoint]
enabled = true
interval_seeds = 1
directory = "../artifacts/policy_exploratory_synthetic/progress"
atomic_writes = true
preserve_prior_attempts = true
resume_mode = "RESTART_FROM_ZERO_PRESERVE_PRIOR_ATTEMPT"
partial_result_profile = "NONMONETARY_UNWEIGHTED_SEED_SCENARIO_DIAGNOSTICS_ONLY"
"""


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
        self.assertFalse(config.full_exploratory_config)
        self.assertIsNone(config.exploratory)

    def test_full_exploratory_contract_is_explicit_and_non_production(self) -> None:
        config = load_policy_config(EXPLORATORY_CONFIG)

        self.assertIs(config.run_purpose, PolicyRunPurpose.EXPLORATORY)
        self.assertTrue(config.full_exploratory_config)
        self.assertFalse(config.full_campaign_config)
        self.assertIsNone(config.campaign)
        self.assertIsNone(config.output_contract)
        self.assertIsNone(config.execution_receipt)
        exploratory = config.exploratory
        assert exploratory is not None
        self.assertEqual(exploratory.exploratory_plan_id, EXPLORATORY_PLAN_ID)
        self.assertEqual(
            exploratory.exploratory_plan_sha256,
            "5915bc42752dd77b984a67bdab5a79a040d99fc84f381de2d3aeb4c813bc2414",
        )
        self.assertEqual(
            exploratory.artifact_namespace,
            EXPLORATORY_ARTIFACT_NAMESPACE,
        )
        self.assertEqual(exploratory.execution_kind, EXPLORATORY_EXECUTION_KIND)
        self.assertEqual(
            exploratory.population_basis,
            EXPLORATORY_POPULATION_BASIS,
        )
        self.assertEqual(
            exploratory.estimand_interpretation,
            EXPLORATORY_ESTIMAND_INTERPRETATION,
        )
        self.assertEqual(
            exploratory.monetary_amount_semantics,
            EXPLORATORY_MONETARY_AMOUNT_SEMANTICS,
        )
        self.assertEqual(
            exploratory.unweighted_output_role,
            EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE,
        )
        self.assertEqual(
            exploratory.internal_monetary_unit,
            EXPLORATORY_INTERNAL_MONETARY_UNIT,
        )
        self.assertEqual(
            exploratory.raw_internal_unit_output_role,
            EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE,
        )
        self.assertTrue(exploratory.allow_synthetic)
        self.assertTrue(exploratory.execution_enabled)
        for name in (
            "campaign_ready",
            "production_campaign",
            "empirical_claims",
            "population_inference_claims",
            "causal_claims",
            "generalisation_claims",
        ):
            self.assertFalse(getattr(exploratory, name), name)
        self.assertTrue(exploratory.identical_pretreatment_cohorts)
        self.assertTrue(
            exploratory.identical_population_weights_across_scenarios
        )
        self.assertEqual(
            config.output.output_dir,
            Path("artifacts/policy_exploratory_synthetic"),
        )
        checkpoint = config.exploratory_checkpoint
        assert checkpoint is not None
        self.assertTrue(checkpoint.enabled)
        self.assertEqual(checkpoint.interval_seeds, 1)
        self.assertFalse("resume" == checkpoint.resume_mode)
        self.assertEqual(checkpoint.directory.name, "progress")
        assert config.ledger is not None
        self.assertEqual(
            config.ledger.path.parent.name,
            EXPLORATORY_ARTIFACT_NAMESPACE,
        )
        assert config.analysis_plan is not None
        self.assertEqual(
            config.analysis_plan.expected_plan_id,
            "illustrative.prospective.composite-harm.baseline-vs-safe.v3",
        )
        self.assertEqual(
            config.analysis_plan.parent_plan_id,
            "illustrative.prospective.composite-harm.baseline-vs-safe.v2",
        )

    def test_full_exploratory_claims_labels_and_paths_fail_closed(self) -> None:
        original = _exploratory_config_text()
        mutations = (
            (
                'exploratory_plan_id = "illustrative.exploratory.synthetic.composite-harm.baseline-vs-safe.v1"',
                'exploratory_plan_id = "illustrative.exploratory.synthetic.other.v1"',
                "must select the versioned synthetic exploratory sidecar",
            ),
            (
                'exploratory_plan_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"',
                'exploratory_plan_sha256 = "BLOCKED_PENDING_EXPLORATORY_PLAN"',
                "must be a resolved lowercase SHA-256",
            ),
            ("allow_synthetic = true", "allow_synthetic = false", "must be true"),
            (
                "execution_enabled = true",
                "execution_enabled = false",
                "execution_enabled must be true",
            ),
            ("campaign_ready = false", "campaign_ready = true", "must be false"),
            (
                "production_campaign = false",
                "production_campaign = true",
                "must be false",
            ),
            ("empirical_claims = false", "empirical_claims = true", "must be false"),
            (
                "population_inference_claims = false",
                "population_inference_claims = true",
                "must be false",
            ),
            ("causal_claims = false", "causal_claims = true", "must be false"),
            (
                "generalisation_claims = false",
                "generalisation_claims = true",
                "must be false",
            ),
            (
                'execution_kind = "COMPUTATIONAL_SIMULATION"',
                'execution_kind = "PRODUCTION"',
                "COMPUTATIONAL_SIMULATION",
            ),
            (
                'internal_monetary_unit = "simulation_cents"',
                'internal_monetary_unit = "EUR"',
                "simulation_cents",
            ),
            (
                'output_dir = "artifacts/policy_exploratory_synthetic"',
                'output_dir = "artifacts/policy_campaign_BLOCKED"',
                "output_dir must be isolated",
            ),
            (
                "../artifacts/policy_exploratory_synthetic/exploratory-ledger.sqlite3",
                "../artifacts/policy_campaign_BLOCKED/campaign-ledger.sqlite3",
                "ledger path must be the isolated",
            ),
            (
                "interval_seeds = 1",
                "interval_seeds = 50",
                "interval_seeds must be 1",
            ),
            (
                'directory = "../artifacts/policy_exploratory_synthetic/progress"',
                'directory = "../artifacts/policy_campaign_BLOCKED/progress"',
                "checkpoint directory must be isolated",
            ),
            (
                'expected_plan_sha256 = "1f27f290179cb054da10d97fce877ca9b17582f82f7d18e76788575afb18a023"',
                'expected_plan_sha256 = "BLOCKED_PENDING_PARENT_PLAN"',
                "identities must use resolved lowercase SHA-256",
            ),
            (
                'expected_plan_id = "illustrative.prospective.composite-harm.baseline-vs-safe.v3"',
                'expected_plan_id = "illustrative.prospective.composite-harm.baseline-vs-safe.v4"',
                "must retain the production-v3 scientific analysis plan binding",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (old, new, message) in enumerate(mutations):
                with self.subTest(new=new):
                    path = root / f"unsafe-exploratory-{index}.toml"
                    path.write_text(original.replace(old, new, 1), "utf-8")
                    with self.assertRaisesRegex(
                        PolicyConfigurationError,
                        message,
                    ):
                        load_policy_config(path)

    def test_full_exploratory_forbids_production_sections_and_unknown_keys(
        self,
    ) -> None:
        original_campaign = CAMPAIGN_CONFIG.read_text("utf-8")
        exploratory = _exploratory_config_text()
        cases = (
            (
                "campaign",
                exploratory + "\n" + _toml_table(original_campaign, "campaign"),
                r"forbids \[campaign\]",
            ),
            (
                "output_contract",
                exploratory
                + "\n"
                + _toml_table(original_campaign, "output_contract"),
                r"forbids production \[output_contract\]",
            ),
            (
                "execution_receipt",
                exploratory
                + "\n"
                + _toml_table(original_campaign, "execution_receipt"),
                r"forbids production \[execution_receipt\]",
            ),
            (
                "unknown_key",
                exploratory.replace(
                    'execution_kind = "COMPUTATIONAL_SIMULATION"',
                    'execution_kind = "COMPUTATIONAL_SIMULATION"\nunknown = true',
                    1,
                ),
                "exploratory keys differ",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, text, message in cases:
                with self.subTest(name=name):
                    path = root / f"forbidden-{name}.toml"
                    path.write_text(text, "utf-8")
                    with self.assertRaisesRegex(
                        PolicyConfigurationError,
                        message,
                    ):
                        load_policy_config(path)

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
