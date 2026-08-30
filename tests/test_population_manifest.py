from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_analysis_composition import (  # noqa: E402
    _CUSTOM_PROFILE_INPUT_LINEAGE,
    _CUSTOM_PROFILES,
    _all_players_predicate,
    _build_plan,
    _write_plan,
)
from test_population_projection_adapter import _complete_adapter  # noqa: E402

from microtx_sim.causal.analysis_binding import (  # noqa: E402
    resolve_run_analysis_binding,
    validate_analysis_plan_inputs,
)
from microtx_sim.causal.analysis_plan import (  # noqa: E402
    load_prospective_analysis_plan,
)
from microtx_sim.causal.batch import (  # noqa: E402
    PolicyBatchSpec,
    resolve_policy_run_inputs,
    run_policy_batch,
)
from microtx_sim.config import (  # noqa: E402
    PopulationExecutionMode,
    PopulationProjectionConfig,
)
from microtx_sim.consumers.decision import DecisionParameters  # noqa: E402
from microtx_sim.data.population_evidence import (  # noqa: E402
    PopulationGamingState,
)
from microtx_sim.outputs.manifest import build_run_manifest  # noqa: E402
from microtx_sim.policy_config import (  # noqa: E402
    AnalysisPlanSelection,
    PolicyOutputConfig,
    PolicyRunPurpose,
    load_policy_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "policy_prototype.toml"


class PopulationManifestTests(unittest.TestCase):
    def _fixture(self, root: Path):
        _verification, _design, mapping_path, _mapping, adapter = (
            _complete_adapter(root)
        )
        base = load_policy_config(CONFIG_PATH)
        spec = PolicyBatchSpec(
            seeds=(73,),
            days=0,
            player_count=12,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        run_inputs = resolve_policy_run_inputs(
            harm_parameters=base.harm_parameters,
            harm_weights=base.harm_weights,
            opportunity_valuation=base.opportunity_valuation,
            producer_assumptions=base.producer_assumptions,
            epgc_policy=base.epgc_policy,
        )
        gamer_predicate = replace(
            _all_players_predicate(),
            gaming_states=(PopulationGamingState.GAMER,),
        )
        plan = _build_plan(
            spec,
            run_inputs,
            adapter,
            inclusion_predicate=gamer_predicate,
        )
        plan_path = root / "analysis-plan.json"
        _write_plan(plan_path, plan)
        loaded = load_prospective_analysis_plan(plan_path)
        config = replace(
            base,
            batch=spec,
            population=PopulationProjectionConfig(
                mode=PopulationExecutionMode.PROJECTED_V1,
                design_bundle_path=adapter.verification.bundle.bundle_path,
                runtime_mapping_bundle_path=mapping_path,
                adapter_id=adapter.adapter_id,
            ),
            analysis_plan=AnalysisPlanSelection(plan_path),
            output=PolicyOutputConfig(
                root / "output",
                histogram_bins=8,
                include_player_rows=True,
                run_sensitivity=False,
            ),
        )
        validate_analysis_plan_inputs(
            loaded.plan,
            batch_spec=spec,
            run_inputs=run_inputs,
            population_adapter=adapter,
            profile_input_lineage=_CUSTOM_PROFILE_INPUT_LINEAGE,
        )
        batch = run_policy_batch(
            spec,
            country_profiles=_CUSTOM_PROFILES,
            harm_parameters=config.harm_parameters,
            harm_weights=config.harm_weights,
            opportunity_valuation=config.opportunity_valuation,
            producer_assumptions=config.producer_assumptions,
            epgc_policy=config.epgc_policy,
            population_adapter=adapter,
        )
        binding = resolve_run_analysis_binding(loaded.plan, batch)
        return config, batch, adapter, loaded, binding

    def test_manifest_separates_unweighted_root_from_weighted_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, batch, adapter, loaded, binding = self._fixture(root)

            without_binding = build_run_manifest(
                replace(config, analysis_plan=None),
                batch,
                config_path=CONFIG_PATH,
                repository_root=ROOT,
                created_utc="2026-08-30T00:00:00+00:00",
            )
            unbound_profile = without_binding["population_output_contract"]
            self.assertFalse(
                unbound_profile["prospective_population_profile"]["weighted"]
            )

            manifest = build_run_manifest(
                config,
                batch,
                config_path=CONFIG_PATH,
                repository_root=ROOT,
                created_utc="2026-08-30T00:00:00+00:00",
                analysis_plan=loaded,
                analysis_binding=binding,
            )
            contract = manifest["population_output_contract"]
            self.assertEqual(contract["run_purpose"], "development")
            legacy = contract["legacy_root_tables"]
            self.assertFalse(legacy["weighted"])
            self.assertFalse(legacy["population_estimate"])
            self.assertEqual(legacy["role"], "diagnostic")
            self.assertEqual(
                legacy["interpretation"],
                "unweighted synthetic-player summaries",
            )
            self.assertFalse(legacy["units_reinterpreted"])

            prospective = contract["prospective_population_profile"]
            self.assertTrue(prospective["binding_present"])
            self.assertTrue(prospective["profile_declared"])
            self.assertFalse(prospective["publication_claimed"])
            self.assertTrue(prospective["weighted"])
            self.assertTrue(prospective["analysis_binding_verified"])
            self.assertFalse(prospective["campaign_ready"])
            self.assertEqual(
                prospective["binding_sha256"],
                binding.binding_sha256,
            )

            projected = contract["projected_population_lineage"]
            self.assertEqual(
                projected["evidence"]["bundle_id"],
                adapter.verification.evidence_bundle.bundle_id,
            )
            self.assertEqual(
                projected["evidence"]["population_evidence_bundle_sha256"],
                adapter.verification.evidence_bundle.bundle_sha256,
            )
            self.assertEqual(
                projected["design"]["design_id"],
                adapter.verification.bundle.design_id,
            )
            self.assertEqual(
                projected["design"]["design_bundle_sha256"],
                adapter.verification.bundle.bundle_sha256,
            )
            self.assertEqual(
                projected["adapter"],
                {
                    "adapter_id": adapter.adapter_id,
                    "schema_version": adapter.schema_version,
                    "adapter_sha256": adapter.adapter_sha256,
                    "runtime_projection_id": adapter.runtime_projection_id,
                    "authenticity_verified": False,
                    "campaign_ready": False,
                },
            )
            self.assertEqual(
                projected["runtime_mapping"]["mapping_sha256"],
                adapter.mapping_sha256,
            )

            seed = projected["seed_profiles"][0]
            full = seed["full_cohort"]
            self.assertEqual(full["player_count"], 12)
            self.assertEqual(
                full["exact_analysis_weight_total"]["numerator_decimal"],
                "1",
            )
            self.assertEqual(
                full["exact_analysis_weight_total"]["denominator_decimal"],
                "1",
            )
            self.assertEqual(
                full["exact_expansion_weight_total"]["numerator_decimal"],
                "1000",
            )
            self.assertEqual(
                full["exact_expansion_weight_total"],
                full["exact_target_population_total"],
            )
            self.assertEqual(
                full["cell_identity"]["assignment_sha256"],
                batch.population_execution_lineage.seed_records[0].assignment_sha256,
            )
            self.assertTrue(
                full["pre_treatment_balance"]["exact_balance_passed"]
            )

            selected = seed["selected_profiles"][0]
            self.assertEqual(selected["full_player_count"], 12)
            self.assertEqual(selected["selected_player_count"], 6)
            self.assertEqual(selected["excluded_player_count"], 6)
            self.assertEqual(
                selected["exact_analysis_weight_total"]["numerator_decimal"],
                "1",
            )
            self.assertEqual(
                selected["exact_analysis_weight_total"]["denominator_decimal"],
                "2",
            )
            self.assertEqual(
                selected["exact_expansion_weight_total"]["numerator_decimal"],
                "500",
            )
            self.assertEqual(
                selected["planned_estimand_id"],
                binding.seed_bindings[0].planned_estimand.estimand_id,
            )
            self.assertEqual(
                selected["resolved_estimand_id"],
                binding.seed_bindings[0].spec.estimand_id,
            )
            self.assertEqual(
                selected["result_sha256"],
                binding.seed_bindings[0].result.result_sha256,
            )
            self.assertEqual(
                selected["target_evidence_sha256"],
                binding.seed_bindings[0].spec.target_evidence_sha256,
            )
            self.assertEqual(
                selected["balance_report_sha256"],
                binding.seed_bindings[0].spec.balance_report_sha256,
            )
            gate = contract["campaign_gate"]
            self.assertFalse(gate["enforced"])
            self.assertFalse(gate["passed"])
            self.assertIn(
                "population.analysis_binding.campaign_ready=false",
                gate["blockers"],
            )
            self.assertEqual(manifest["run_purpose"], "development")
            self.assertEqual(
                manifest["config_snapshot"]["meta"]["run_purpose"],
                "development",
            )

    def test_campaign_manifest_gate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, batch, _adapter, loaded, binding = self._fixture(root)
            campaign_config = replace(
                config,
                run_purpose=PolicyRunPurpose.CAMPAIGN,
                provenance_status="calibrated",
            )
            with self.assertRaisesRegex(
                ValueError,
                "campaign population manifest gate failed closed",
            ):
                build_run_manifest(
                    campaign_config,
                    batch,
                    config_path=CONFIG_PATH,
                    repository_root=ROOT,
                    created_utc="2026-08-30T00:00:00+00:00",
                    analysis_plan=loaded,
                    analysis_binding=binding,
                )


if __name__ == "__main__":
    unittest.main()
