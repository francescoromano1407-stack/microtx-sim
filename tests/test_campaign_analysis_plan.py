from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.analysis.uncertainty import ConvergenceRule
from microtx_sim.causal.analysis_plan import (
    AnalysisPlanValidationError,
    FixedSeedStoppingRule,
    build_prospective_analysis_plan,
    load_prospective_analysis_plan,
)
from microtx_sim.causal.analysis_binding import validate_analysis_plan_inputs
from microtx_sim.causal.batch import resolve_policy_run_inputs
from microtx_sim.cli import _load_policy_profiles
from microtx_sim.data.lineage import build_profile_input_lineage
from microtx_sim.data.population_execution import resolve_population_projection_adapter
from microtx_sim.outputs.schema import CAMPAIGN_ANALYSIS_SCHEMA_SHA256
from microtx_sim.policy_config import load_policy_config


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "inputs" / "prospective-analysis-plan.json"
CAMPAIGN_PLAN_PATH = ROOT / "inputs" / "prospective-analysis-plan-amendment-v3.json"
CAMPAIGN_CONFIG_PATH = ROOT / "configs" / "policy_campaign.toml"


_BLOCKERS = [
    "analysis_plan.external_registration=unregistered",
    "analysis_plan.execution_calendar_anchor=unbound",
    "analysis_plan.population_empirical_validation=missing",
    "analysis_plan.population_uncertainty=unquantified",
    "analysis_plan.monetary_source_bundle_signature=missing",
    "analysis_plan.monetary_simulation_bridge=unvalidated",
    "analysis_plan.monetary_rate_uncertainty=unquantified",
    "analysis_plan.parameter_distributions=uncalibrated",
    "analysis_plan.execution_attestation=unverified",
]


def _amendment() -> dict[str, object]:
    parent_loaded = load_prospective_analysis_plan(PARENT_PATH)
    primary = parent_loaded.plan.primary_estimand
    return {
        "amendment_schema_version": "1.0",
        "parent_plan": {
            "artifact_path": "inputs/prospective-analysis-plan.json",
            "schema_version": parent_loaded.plan.schema_version,
            "plan_id": parent_loaded.plan.plan_id,
            "plan_sha256": parent_loaded.plan.plan_sha256,
            "file_sha256": parent_loaded.file_sha256,
        },
        "scientific_change": {
            "primary_estimand_changed": False,
            "original_estimand_id": primary.estimand_id,
            "current_estimand_id": primary.estimand_id,
            "original_specification_sha256": primary.specification_sha256,
            "current_specification_sha256": primary.specification_sha256,
            "explanation": "The scientific estimand is preserved; only implementation and uncertainty contracts change.",
        },
        "changed_inputs": [
            {
                "role": "monetary_source_bundle",
                "artifact_path": "inputs/monetary/ecb-eur-fx-2024-v1/bundle.toml",
                "schema_version": "1",
                "file_sha256": "1" * 64,
                "semantic_sha256": "2" * 64,
                "change_type": "BOUND_LATEST_IMPLEMENTATION",
                "readiness_status": "BLOCKED",
                "readiness_consequence": "signature and rate uncertainty remain unavailable",
            },
            {
                "role": "population_design",
                "artifact_path": "data/provenance/population_design.toml",
                "schema_version": "1",
                "file_sha256": "3" * 64,
                "semantic_sha256": "4" * 64,
                "change_type": "BOUND_LATEST_IMPLEMENTATION",
                "readiness_status": "BLOCKED",
                "readiness_consequence": "illustrative design is not empirical validation",
            },
        ],
        "population_contract": {
            "mode": "projected_v1",
            "design_id": "population-v1",
            "design_file_sha256": "1" * 64,
            "design_sha256": "2" * 64,
            "runtime_mapping_id": "mapping-v2",
            "runtime_mapping_file_sha256": "3" * 64,
            "runtime_mapping_sha256": "4" * 64,
            "adapter_id": "adapter-v2",
            "adapter_sha256": "5" * 64,
            "apportionment_plan_sha256": "6" * 64,
            "population_input_sha256": "7" * 64,
            "cell_count": 864,
            "target_design_units": 40000,
            "assignment_identity_policy": "PER_SEED_REQUIRED",
            "balance_identity_policy": "EXACT_PER_SEED_REQUIRED",
            "lineage_identity_policy": "COMPLETE_FIXED_SET_REQUIRED",
            "uncertainty_status": "UNQUANTIFIED",
            "empirical_validation_claimed": False,
        },
        "monetary_contract": {
            "source_bundle_id": "ecb-eur-fx-2024-v1",
            "source_bundle_file_sha256": "8" * 64,
            "source_bundle_semantic_sha256": "9" * 64,
            "source_artifact_sha256s": {
                "conversion_rates.csv": "a" * 64,
                "ecb_exr_annual_2024.csv": "b" * 64,
            },
            "conversion_table_sha256": "a" * 64,
            "target_currency": "EUR",
            "quote_convention": "target minor units per source minor unit",
            "scale_convention": "exact rational minor-unit conversion",
            "rate_period_start": "2024-01-01",
            "rate_period_end": "2024-12-31",
            "price_period_start": "2024-01-01",
            "price_period_end": "2024-12-31",
            "missing_date_policy": "ANNUAL_OBSERVATION_EXACT_PERIOD_ONLY",
            "rounding_rule": "nearest_minor_unit_half_away_from_zero",
            "rounding_boundary": "AFTER_AGGREGATION",
            "conversion_basis_sha256": "c" * 64,
            "source_bundle_signature_status": "MISSING",
            "simulation_bridge_status": "ILLUSTRATIVE",
            "rate_uncertainty_status": "UNQUANTIFIED",
            "observed_real_world_spending_claimed": False,
        },
        "uncertainty_design": {
            "schema_version": "1.0",
            "seed_uncertainty": {
                "status": "QUANTIFIED_WHEN_COMPLETE",
                "fixed_seed_count": 100,
                "population_weights_applied_within_seed": True,
                "common_random_numbers": True,
                "identical_pretreatment_cohorts": True,
                "outcome_dependent_seed_exclusion_allowed": False,
            },
            "parameter_uncertainty": {
                "status": "ILLUSTRATIVE_DESIGN_ONLY",
                "design_id": "illustrative-joint-v1",
                "design_sha256": "d" * 64,
                "method": "SEEDED_LATIN_HYPERCUBE_V1",
                "probability_interpretation": "NONE",
            },
            "monetary_rate_uncertainty": {
                "status": "UNQUANTIFIED",
                "rate_basis_sha256": "c" * 64,
                "point_observation_is_distribution": False,
            },
            "population_uncertainty": {
                "status": "UNQUANTIFIED",
                "uncertainty_design_id": "UNAVAILABLE_NO_ADMISSIBLE_DESIGN",
                "exact_weighting_is_empirical_validation": False,
            },
            "combined_uncertainty": {
                "status": "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST",
                "double_counting_control": (
                    "one complete seed-parameter-population-rate Cartesian identity"
                ),
                "variance_decomposition_method": (
                    "ORTHOGONAL_FINITE_FULL_FACTORIAL_ANOVA_"
                    "SUM_OF_SQUARES_DIVIDED_BY_N_V1"
                ),
            },
            "oat_role": "DIAGNOSTIC_ONLY",
        },
        "convergence_rule": ConvergenceRule().snapshot(),
        "execution_attestation": {
            "receipt_schema_version": "1.0",
            "pre_run_required": True,
            "post_run_required": True,
            "clean_tree_required": True,
            "environment_match_required": True,
            "mismatch_handling": "REJECT_OR_INVALIDATE",
        },
        "simulation_flow": {
            "execution_layer": "policy_welfare_v1",
            "strategic_world_layer_included": False,
            "incompatibility_status": "FAIL_CLOSED_NO_ADAPTER",
        },
        "readiness_consequences": {
            "campaign_ready": False,
            "blockers": list(_BLOCKERS),
        },
    }


def _plan(
    *,
    amendment: dict[str, object] | None = None,
    seed_count: int = 100,
    estimands: tuple[object, ...] | None = None,
):
    parent = load_prospective_analysis_plan(PARENT_PATH).plan
    return build_prospective_analysis_plan(
        plan_id="illustrative.prospective.composite-harm.baseline-vs-safe.v3",
        expected_causal_design_sha256=parent.expected_causal_design_sha256,
        expected_batch_spec_sha256="d" * 64,
        expected_model_inputs_sha256=parent.expected_model_inputs_sha256,
        expected_population_input_sha256="e" * 64,
        expected_profile_input_sha256="f" * 64,
        expected_metric_contract_sha256=parent.expected_metric_contract_sha256,
        expected_harm_weights_sha256=parent.expected_harm_weights_sha256,
        expected_output_profile_sha256=CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
        stopping_rule=FixedSeedStoppingRule(seeds=tuple(range(1, seed_count + 1))),
        estimands=parent.estimands if estimands is None else estimands,
        declared_harm_weights=parent.declared_harm_weights,
        primary_aggregate_rule=parent.primary_aggregate_rule,
        amendment=amendment if amendment is not None else _amendment(),
    )


class CampaignAnalysisPlanTests(unittest.TestCase):
    def test_checked_in_successor_binds_latest_inputs_without_changing_primary(self) -> None:
        config = load_policy_config(CAMPAIGN_CONFIG_PATH)
        parent = load_prospective_analysis_plan(PARENT_PATH).plan
        loaded = load_prospective_analysis_plan(CAMPAIGN_PLAN_PATH)
        self.assertEqual(loaded.plan.schema_version, "3.0")
        self.assertEqual(
            loaded.plan.primary_estimand.specification_sha256,
            parent.primary_estimand.specification_sha256,
        )
        self.assertEqual(loaded.plan.stopping_rule.seeds, config.batch.seeds)
        self.assertFalse(loaded.plan.campaign_ready)
        assert config.analysis_plan is not None
        self.assertEqual(config.analysis_plan.expected_plan_id, loaded.plan.plan_id)
        self.assertEqual(
            config.analysis_plan.expected_plan_sha256,
            loaded.plan.plan_sha256,
        )
        self.assertNotIn("BLOCKED_PENDING_", CAMPAIGN_PLAN_PATH.read_text("utf-8"))

        profiles = _load_policy_profiles(config)
        assert config.population is not None
        adapter = resolve_population_projection_adapter(
            config.population,
            profiles,
            player_count=config.batch.player_count,
            campaign=False,
        )
        lineage = build_profile_input_lineage(
            profiles.country_profiles,
            profile_bundle=profiles,
        )
        run_inputs = resolve_policy_run_inputs(
            harm_parameters=config.harm_parameters,
            harm_weights=config.harm_weights,
            opportunity_valuation=config.opportunity_valuation,
            producer_assumptions=config.producer_assumptions,
            epgc_policy=config.epgc_policy,
        )
        self.assertIsNone(
            validate_analysis_plan_inputs(
                loaded.plan,
                batch_spec=config.batch,
                run_inputs=run_inputs,
                population_adapter=adapter,
                profile_input_lineage=lineage,
            )
        )

    def test_v3_successor_is_canonical_and_round_trips(self) -> None:
        plan = _plan()
        self.assertEqual(plan.schema_version, "3.0")
        self.assertEqual(len(plan.stopping_rule.seeds), 100)
        self.assertFalse(plan.campaign_ready)
        self.assertIsNotNone(plan.amendment)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(
                json.dumps(plan.snapshot(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            loaded = load_prospective_analysis_plan(path)
            self.assertEqual(loaded.plan, plan)

    def test_v3_rejects_fewer_than_one_hundred_fixed_seeds(self) -> None:
        with self.assertRaisesRegex(AnalysisPlanValidationError, "at least 100"):
            _plan(seed_count=99)

    def test_primary_estimand_cannot_be_silently_changed(self) -> None:
        amendment = _amendment()
        scientific = dict(amendment["scientific_change"])
        scientific["current_specification_sha256"] = "0" * 64
        amendment["scientific_change"] = scientific
        with self.assertRaisesRegex(AnalysisPlanValidationError, "must be identical"):
            _plan(amendment=amendment)

    def test_declared_current_hash_must_bind_actual_successor_estimand(self) -> None:
        parent = load_prospective_analysis_plan(PARENT_PATH).plan
        primary = parent.primary_estimand
        changed_primary = replace(
            primary,
            period=replace(
                primary.period,
                description=primary.period.description + " Changed silently.",
            ),
        )
        estimands = tuple(
            changed_primary if item.estimand_id == primary.estimand_id else item
            for item in parent.estimands
        )
        with self.assertRaisesRegex(
            AnalysisPlanValidationError,
            "differs from the successor plan",
        ):
            _plan(estimands=estimands)

    def test_point_rate_cannot_be_relabelled_as_quantified_uncertainty(self) -> None:
        amendment = _amendment()
        monetary = dict(amendment["monetary_contract"])
        monetary["rate_uncertainty_status"] = "QUANTIFIED"
        amendment["monetary_contract"] = monetary
        with self.assertRaisesRegex(AnalysisPlanValidationError, "do not quantify"):
            _plan(amendment=amendment)

    def test_seed_uncertainty_requires_identical_pretreatment_cohorts(self) -> None:
        amendment = _amendment()
        uncertainty = dict(amendment["uncertainty_design"])
        seed = dict(uncertainty["seed_uncertainty"])
        seed["identical_pretreatment_cohorts"] = False
        uncertainty["seed_uncertainty"] = seed
        amendment["uncertainty_design"] = uncertainty
        with self.assertRaisesRegex(
            AnalysisPlanValidationError,
            "identical_pretreatment_cohorts",
        ):
            _plan(amendment=amendment)

    def test_convergence_thresholds_must_be_positive_and_finite(self) -> None:
        amendment = _amendment()
        convergence = dict(amendment["convergence_rule"])
        convergence["maximum_mcse"] = 0.0
        amendment["convergence_rule"] = convergence
        with self.assertRaisesRegex(
            AnalysisPlanValidationError,
            "maximum_mcse",
        ):
            _plan(amendment=amendment)

    def test_amendment_readiness_cannot_omit_fixed_blocker(self) -> None:
        amendment = _amendment()
        readiness = dict(amendment["readiness_consequences"])
        readiness["blockers"] = _BLOCKERS[:-1]
        amendment["readiness_consequences"] = readiness
        with self.assertRaisesRegex(AnalysisPlanValidationError, "omit fixed blockers"):
            _plan(amendment=amendment)


if __name__ == "__main__":
    unittest.main()
