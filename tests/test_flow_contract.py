from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_jurisdiction_population_fixture import (  # noqa: E402
    REGISTERED_PROFILE_CODES,
    write_four_jurisdiction_population_fixture,
)
from test_analysis_binding import (  # noqa: E402
    _plan,
    _planned_estimand,
    _unsafe_clone,
)

from microtx_sim.causal.analysis_binding import (  # noqa: E402
    resolve_run_analysis_binding,
)
from microtx_sim.causal.analysis_plan import (  # noqa: E402
    PopulationOutcomeMetric,
    load_prospective_analysis_plan,
)
from microtx_sim.causal.batch import (  # noqa: E402
    PolicyBatchSpec,
    resolve_policy_run_inputs,
    run_policy_batch,
)
from microtx_sim.causal.flow_contract import (  # noqa: E402
    PolicyAggregationBasis,
    PolicyExecutionLayer,
    PolicyFlowContractError,
    attest_policy_only_flow,
    build_policy_flow_contract,
    validate_policy_aggregation_basis,
)
from microtx_sim.causal.scenarios import ScenarioId  # noqa: E402
from microtx_sim.consumers.decision import DecisionParameters  # noqa: E402
from microtx_sim.data.lineage import build_profile_input_lineage  # noqa: E402
from microtx_sim.data.monetary_execution import (  # noqa: E402
    build_monetary_output_currency_semantics,
)
from microtx_sim.data.profiles import load_profile_bundle  # noqa: E402
from microtx_sim.metrics.population_estimands import (  # noqa: E402
    PopulationMetricKind,
    paired_weighted_mean_difference,
)


def _write_plan(path: Path, plan: object) -> None:
    snapshot = plan.snapshot()  # type: ignore[attr-defined]
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="",
    )


class PolicyFlowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls._temporary.name)
        cls.profiles = load_profile_bundle()
        cls.profile_lineage = build_profile_input_lineage(
            cls.profiles.country_profiles,
            profile_bundle=cls.profiles,
        )
        cls.adapter = write_four_jurisdiction_population_fixture(
            cls.root / "population",
            player_count=16,
        )
        cls.spec = PolicyBatchSpec(
            seeds=(29, 17),
            days=1,
            player_count=16,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        cls.run_inputs = resolve_policy_run_inputs()
        cls.plan = _plan(
            cls.spec,
            cls.run_inputs,
            cls.adapter,
            overrides={
                "expected_profile_input_sha256": (
                    cls.profile_lineage.fingerprint_sha256
                )
            },
        )
        cls.plan_path = cls.root / "policy-flow-plan.json"
        _write_plan(cls.plan_path, cls.plan)
        cls.loaded_plan = load_prospective_analysis_plan(cls.plan_path)
        cls.batch = run_policy_batch(
            cls.spec,
            profile_bundle=cls.profiles,
            harm_parameters=cls.run_inputs.harm_parameters,
            harm_weights=cls.run_inputs.harm_weights,
            opportunity_valuation=cls.run_inputs.opportunity_valuation,
            producer_assumptions=cls.run_inputs.producer_assumptions,
            epgc_policy=cls.run_inputs.epgc_policy,
            population_adapter=cls.adapter,
        )
        cls.binding = resolve_run_analysis_binding(cls.plan, cls.batch)
        cls.contract = build_policy_flow_contract(cls.plan)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_end_to_end_policy_flow_preserves_all_pair_identities(self) -> None:
        verification = attest_policy_only_flow(
            self.contract,
            analysis_plan=self.loaded_plan,
            batch=self.batch,
            analysis_binding=self.binding,
        )

        self.assertFalse(verification.campaign_ready)
        self.assertFalse(verification.scientific_readiness_claimed)
        self.assertEqual(
            verification.contract.execution_layer,
            PolicyExecutionLayer.POLICY_WELFARE,
        )
        self.assertEqual(
            tuple(item.seed for item in verification.seed_identities),
            self.spec.seeds,
        )
        self.assertEqual(len(verification.seed_identities), len(self.spec.seeds))
        lineage = self.batch.population_execution_lineage
        assert lineage is not None
        for identity in verification.seed_identities:
            with self.subTest(seed=identity.seed):
                population = lineage.record_for_seed(identity.seed)
                self.assertEqual(
                    identity.reference_scenario_id,
                    ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
                )
                self.assertEqual(
                    identity.comparison_scenario_id,
                    ScenarioId.BASELINE_F2P,
                )
                self.assertEqual(
                    identity.cohort_digest,
                    self.batch.cohort_digest_by_seed[identity.seed],
                )
                self.assertEqual(
                    identity.population_seed_record_sha256,
                    population.seed_record_sha256,
                )
                self.assertEqual(
                    identity.ordered_player_ids_sha256,
                    population.ordered_player_ids_sha256,
                )
                self.assertEqual(
                    identity.jurisdiction_codes,
                    REGISTERED_PROFILE_CODES,
                )
                self.assertEqual(
                    identity.primary_metric_contract_id,
                    "player_outcomes.csv:composite_harm",
                )
                self.assertEqual(
                    identity.aggregation_basis,
                    PolicyAggregationBasis.DIRECT_NONMONETARY,
                )
                self.assertIsNone(identity.monetary_basis_sha256)
                seed_binding = next(
                    item
                    for item in self.binding.seed_bindings
                    if item.seed == identity.seed
                )
                self.assertEqual(
                    identity.selected_design_weights_sha256,
                    seed_binding.selected_weights.design_sha256,
                )
                self.assertEqual(
                    identity.effective_metric_contract_sha256,
                    seed_binding.metric_contract_sha256,
                )

    def test_strategic_or_combined_execution_is_rejected(self) -> None:
        for layer in (
            PolicyExecutionLayer.STRATEGIC_MARKET,
            PolicyExecutionLayer.BOTH,
        ):
            with self.subTest(layer=layer), self.assertRaisesRegex(
                PolicyFlowContractError,
                "no typed scientific adapter",
            ):
                build_policy_flow_contract(self.plan, execution_layer=layer)

    def test_stale_plan_and_mismatched_plan_fail_closed(self) -> None:
        stale_path = self.root / "stale-plan.json"
        _write_plan(stale_path, self.plan)
        stale = load_prospective_analysis_plan(stale_path)
        stale_path.write_bytes(stale_path.read_bytes() + b" ")
        with self.assertRaisesRegex(
            PolicyFlowContractError,
            "plan file changed|analysis plan file changed",
        ):
            attest_policy_only_flow(
                self.contract,
                analysis_plan=stale,
                batch=self.batch,
                analysis_binding=self.binding,
            )

        mismatched = _plan(
            self.spec,
            self.run_inputs,
            self.adapter,
            seeds=(17,),
            overrides={
                "expected_profile_input_sha256": (
                    self.profile_lineage.fingerprint_sha256
                )
            },
        )
        mismatch_path = self.root / "mismatched-plan.json"
        _write_plan(mismatch_path, mismatched)
        loaded_mismatch = load_prospective_analysis_plan(mismatch_path)
        mismatched_contract = build_policy_flow_contract(mismatched)
        with self.assertRaisesRegex(
            PolicyFlowContractError,
            "stopping rule|plan and batch|batch_spec|failed closed",
        ):
            attest_policy_only_flow(
                mismatched_contract,
                analysis_plan=loaded_mismatch,
                batch=self.batch,
                analysis_binding=self.binding,
            )

    def test_pair_jurisdiction_weight_and_metric_tampering_is_rejected(self) -> None:
        comparison_index = next(
            index
            for index, record in enumerate(self.batch.records)
            if record.result.seed == self.spec.seeds[0]
            and record.result.scenario.scenario_id is ScenarioId.BASELINE_F2P
        )
        comparison_record = self.batch.records[comparison_index]
        changed_jurisdiction = comparison_record.result.jurisdiction.copy()
        changed_jurisdiction[0] = np.int16(
            (int(changed_jurisdiction[0]) + 1) % len(REGISTERED_PROFILE_CODES)
        )
        changed_result = replace(
            comparison_record.result,
            jurisdiction=changed_jurisdiction,
        )
        changed_records = list(self.batch.records)
        changed_records[comparison_index] = replace(
            comparison_record,
            result=changed_result,
        )
        changed_batch = _unsafe_clone(self.batch, records=tuple(changed_records))
        with self.assertRaisesRegex(
            PolicyFlowContractError,
            "pre-treatment|jurisdiction",
        ):
            attest_policy_only_flow(
                self.contract,
                analysis_plan=self.loaded_plan,
                batch=changed_batch,
                analysis_binding=self.binding,
            )

        primary_binding = self.binding.seed_bindings[0]
        weights = primary_binding.selected_weights
        changed_weights = _unsafe_clone(
            weights,
            weight_numerators=(
                weights.weight_numerators[0] + weights.weight_denominators[0],
                *weights.weight_numerators[1:],
            ),
        )
        changed_seed_binding = _unsafe_clone(
            primary_binding,
            selected_weights=changed_weights,
        )
        changed_binding = _unsafe_clone(
            self.binding,
            seed_bindings=(changed_seed_binding, *self.binding.seed_bindings[1:]),
        )
        with self.assertRaises(PolicyFlowContractError):
            attest_policy_only_flow(
                self.contract,
                analysis_plan=self.loaded_plan,
                batch=self.batch,
                analysis_binding=changed_binding,
            )

        changed_seed_binding = _unsafe_clone(
            primary_binding,
            metric_contract_sha256="0" * 64,
        )
        changed_binding = _unsafe_clone(
            self.binding,
            seed_bindings=(changed_seed_binding, *self.binding.seed_bindings[1:]),
        )
        with self.assertRaisesRegex(
            PolicyFlowContractError,
            "metric-contract|metric contract",
        ):
            attest_policy_only_flow(
                self.contract,
                analysis_plan=self.loaded_plan,
                batch=self.batch,
                analysis_binding=changed_binding,
            )

    def test_raw_cross_currency_pooling_is_rejected_and_conversion_is_retained(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            PolicyFlowContractError,
            "raw jurisdiction-currency values cannot cross",
        ):
            validate_policy_aggregation_basis(
                PopulationMetricKind.MONEY_MINOR_UNITS,
                PolicyAggregationBasis.RAW_JURISDICTION_CURRENCY,
            )

        currency = build_monetary_output_currency_semantics(
            self.profile_lineage,
            jurisdiction_codes=self.profile_lineage.profile_codes,
            target_minor_unit_name="euro cent",
        )
        money_estimand = _planned_estimand(
            outcome_metric=PopulationOutcomeMetric.SPENDING_CENTS,
            metric_contract_id="player_outcomes.csv:spending_cents",
            currency=currency,
        )
        money_plan = _plan(
            self.spec,
            self.run_inputs,
            self.adapter,
            estimand=money_estimand,
            overrides={
                "expected_profile_input_sha256": (
                    self.profile_lineage.fingerprint_sha256
                )
            },
        )
        money_path = self.root / "money-plan.json"
        _write_plan(money_path, money_plan)
        loaded_money = load_prospective_analysis_plan(money_path)
        money_binding = resolve_run_analysis_binding(money_plan, self.batch)
        verification = attest_policy_only_flow(
            build_policy_flow_contract(money_plan),
            analysis_plan=loaded_money,
            batch=self.batch,
            analysis_binding=money_binding,
        )
        for identity in verification.seed_identities:
            self.assertEqual(
                identity.aggregation_basis,
                PolicyAggregationBasis.EXACT_TARGET_CURRENCY_BEFORE_WEIGHTING,
            )
            self.assertIsNotNone(identity.monetary_basis_sha256)

    def test_null_versus_null_population_contrast_is_exactly_zero(self) -> None:
        binding = self.binding.seed_bindings[0]
        baseline = next(
            record.result
            for record in self.batch.records
            if record.result.seed == binding.seed
            and record.result.scenario.scenario_id is ScenarioId.BASELINE_F2P
        )
        result = paired_weighted_mean_difference(
            binding.spec,
            binding.selected_weights,
            baseline.composite_harm,
            binding.selected_weights,
            baseline.composite_harm,
        )
        self.assertEqual(result.numerator, 0)
        self.assertEqual(result.denominator, 1)


if __name__ == "__main__":
    unittest.main()
