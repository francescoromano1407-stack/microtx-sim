from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.causal.analysis_plan import (
    EXPLORATORY_ANALYSIS_PLAN_KIND,
    EXPLORATORY_ANALYSIS_PLAN_SCHEMA_VERSION,
    AnalysisPlanCampaignError,
    AnalysisPlanValidationError,
    AnalysisPlanVerificationError,
    ExploratoryAnalysisPlan,
    build_exploratory_analysis_plan,
    load_exploratory_analysis_plan,
    load_prospective_analysis_plan,
    verify_exploratory_analysis_plan_parent,
    verify_loaded_exploratory_analysis_plan,
    verify_loaded_prospective_analysis_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "inputs" / "prospective-analysis-plan-amendment-v3.json"
EXPLORATORY_PATH = ROOT / "inputs" / "exploratory-synthetic-analysis-plan-v1.json"


def _canonical(payload: object) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rebuild(payload: dict[str, object]) -> ExploratoryAnalysisPlan:
    canonical = _canonical(payload)
    return ExploratoryAnalysisPlan(
        identity_payload_json=canonical,
        plan_sha256=sha256(canonical.encode("utf-8")).hexdigest(),
    )


class ExploratoryAnalysisPlanTests(unittest.TestCase):
    def _parent(self):
        return verify_loaded_prospective_analysis_plan(
            load_prospective_analysis_plan(PARENT_PATH)
        )

    def test_checked_in_sidecar_is_canonical_and_exactly_parent_derived(self) -> None:
        parent = self._parent()
        loaded = verify_loaded_exploratory_analysis_plan(
            load_exploratory_analysis_plan(EXPLORATORY_PATH)
        )
        verify_exploratory_analysis_plan_parent(loaded.plan, parent)
        rebuilt = build_exploratory_analysis_plan(
            parent,
            plan_id=loaded.plan.plan_id,
        )

        self.assertEqual(loaded.plan, rebuilt)
        self.assertEqual(
            loaded.file_sha256,
            sha256(EXPLORATORY_PATH.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            loaded.plan.schema_version,
            EXPLORATORY_ANALYSIS_PLAN_SCHEMA_VERSION,
        )
        self.assertEqual(
            loaded.plan.identity_payload["plan_kind"],
            EXPLORATORY_ANALYSIS_PLAN_KIND,
        )
        self.assertEqual(
            loaded.plan.primary_estimand.specification_sha256,
            parent.plan.primary_estimand.specification_sha256,
        )
        self.assertEqual(
            loaded.plan.primary_estimand.reference_scenario_id,
            parent.plan.primary_estimand.reference_scenario_id,
        )
        self.assertEqual(
            loaded.plan.primary_estimand.comparison_scenario_id,
            parent.plan.primary_estimand.comparison_scenario_id,
        )
        self.assertEqual(loaded.plan.stopping_rule.seeds, parent.plan.stopping_rule.seeds)
        self.assertEqual(len(loaded.plan.stopping_rule.seeds), 150)
        scenarios = loaded.plan.identity_payload["scenario_definitions"]
        self.assertEqual(
            [item["scenario_id"] for item in scenarios],
            [
                "baseline_f2p",
                "transparent_direct_price",
                "no_random_rewards",
                "no_time_limited_pressure",
                "spending_cap_cooling_off",
                "safe_fixed_price_subscription",
                "epgc",
            ],
        )
        self.assertTrue(
            all(
                set(item)
                == {
                    "ordinal",
                    "scenario_id",
                    "label",
                    "description",
                    "mechanics",
                    "fixed_access_price_cents",
                    "subscription_price_cents",
                    "epgc_enabled",
                }
                for item in scenarios
            )
        )
        self.assertEqual(
            len(loaded.plan.identity_payload["ordered_scenario_set_sha256"]),
            64,
        )

    def test_sidecar_records_weighting_monetary_uncertainty_and_limits(self) -> None:
        payload = load_exploratory_analysis_plan(EXPLORATORY_PATH).plan.identity_payload
        population = payload["population_weighting"]
        monetary = payload["monetary_semantics"]
        uncertainty = payload["uncertainty_design"]
        limits = payload["interpretation_limits"]

        self.assertEqual(population["weight_representation"], "EXACT_RATIONAL")
        self.assertTrue(
            population["applied_within_seed_before_cross_seed_aggregation"]
        )
        self.assertTrue(population["identical_across_paired_scenarios"])
        self.assertFalse(population["empirical_validation_claimed"])
        self.assertFalse(population["target_population_generalization_allowed"])
        self.assertEqual(
            monetary["semantic_label"],
            "SIMULATED_MODEL_EQUIVALENT_TARGET_CURRENCY_VALUES",
        )
        self.assertTrue(monetary["exact_rational_conversion_required"])
        self.assertFalse(monetary["observed_real_world_spending_claimed"])
        self.assertEqual(monetary["rate_uncertainty_status"], "UNQUANTIFIED")
        self.assertEqual(
            uncertainty["parameter_uncertainty"]["status"],
            "ILLUSTRATIVE_DESIGN_ONLY",
        )
        self.assertEqual(
            uncertainty["population_uncertainty"]["status"],
            "UNQUANTIFIED",
        )
        self.assertTrue(limits["model_internal_results_only"])
        self.assertFalse(limits["real_world_causal_claims_allowed"])
        self.assertFalse(limits["production_campaign_authorized"])

    def test_sidecar_cannot_claim_preregistration_or_campaign_readiness(self) -> None:
        loaded = load_exploratory_analysis_plan(EXPLORATORY_PATH)
        payload = loaded.plan.identity_payload
        payload["preregistered"] = True
        with self.assertRaisesRegex(
            AnalysisPlanValidationError,
            "cannot be preregistered",
        ):
            _rebuild(payload)
        with self.assertRaises(AnalysisPlanCampaignError):
            loaded.plan.validate_for_campaign()

    def test_sidecar_cannot_relabel_model_values_as_observed_spending(self) -> None:
        payload = load_exploratory_analysis_plan(EXPLORATORY_PATH).plan.identity_payload
        monetary = dict(payload["monetary_semantics"])
        monetary["observed_real_world_spending_claimed"] = True
        payload["monetary_semantics"] = monetary
        with self.assertRaisesRegex(
            AnalysisPlanValidationError,
            "observed_real_world_spending_claimed=false",
        ):
            _rebuild(payload)

    def test_explicit_contrast_cannot_differ_from_embedded_estimand(self) -> None:
        payload = load_exploratory_analysis_plan(EXPLORATORY_PATH).plan.identity_payload
        scientific = dict(payload["scientific_estimand"])
        scientific["comparison_scenario_id"] = "safe_fixed_price_subscription"
        payload["scientific_estimand"] = scientific
        with self.assertRaisesRegex(
            AnalysisPlanValidationError,
            "explicit contrast differs",
        ):
            _rebuild(payload)

    def test_scenario_catalogue_cannot_be_relabelled_or_reordered(self) -> None:
        payload = load_exploratory_analysis_plan(EXPLORATORY_PATH).plan.identity_payload
        scenarios = [dict(item) for item in payload["scenario_definitions"]]
        scenarios[0]["label"] = "Altered label"
        payload["scenario_definitions"] = scenarios
        with self.assertRaisesRegex(
            AnalysisPlanValidationError,
            "complete ordered seven-scenario catalogue",
        ):
            _rebuild(payload)

    def test_parent_identity_is_reverified_not_merely_well_formed(self) -> None:
        parent = self._parent()
        payload = load_exploratory_analysis_plan(EXPLORATORY_PATH).plan.identity_payload
        parent_identity = dict(payload["parent_plan"])
        parent_identity["file_sha256"] = "0" * 64
        payload["parent_plan"] = parent_identity
        forged = _rebuild(payload)
        with self.assertRaisesRegex(
            AnalysisPlanVerificationError,
            "parent_file_sha256",
        ):
            verify_exploratory_analysis_plan_parent(forged, parent)

    def test_file_loader_rejects_semantic_tampering(self) -> None:
        payload = load_exploratory_analysis_plan(EXPLORATORY_PATH).plan.snapshot()
        payload["campaign_ready"] = True
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "exploratory.json"
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                AnalysisPlanValidationError,
                "campaign-ready",
            ):
                load_exploratory_analysis_plan(path)


if __name__ == "__main__":
    unittest.main()
