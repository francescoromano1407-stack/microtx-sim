from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field, fields, replace
import json
import unittest
from unittest.mock import patch

import numpy as np

from microtx_sim.causal.batch import (
    PolicyBatchResult,
    PolicyBatchSpec,
    PolicyRunInputs,
    resolve_policy_run_inputs,
    run_policy_batch,
)
from microtx_sim.causal.scenarios import ScenarioId, ScenarioSpec, required_scenarios
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.consumers.population import CountryProfile
from microtx_sim.domain.monetisation import MonetisationVector
from microtx_sim.metrics.harm import (
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmWeights,
)
from microtx_sim.simulation.policy_orchestrator import (
    ProducerAssumptions,
    default_epgc_policy,
    run_policy_scenario,
)


PROFILE = (CountryProfile(code="XX"),)


@dataclass(frozen=True)
class _ExtendedMonetisationVector(MonetisationVector):
    mutable_metadata: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _ExtendedPolicyBatchSpec(PolicyBatchSpec):
    mutable_metadata: list[str] = field(default_factory=list)


class _MutableInt(int):
    pass


def _projection_bytes(batch: PolicyBatchResult) -> bytes:
    return json.dumps(
        {
            "seed_rows": batch.seed_rows(),
            "scenario_rows": batch.scenario_rows(),
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class PolicyBatchTests(unittest.TestCase):
    def test_batch_retains_fully_resolved_default_and_custom_inputs(self) -> None:
        spec = PolicyBatchSpec(seeds=(13,), days=0, player_count=0)
        default_result = run_policy_batch(spec, country_profiles=PROFILE)
        expected_defaults = resolve_policy_run_inputs()

        self.assertEqual(spec.snapshot()["seeds"], [13])
        self.assertEqual(len(spec.snapshot_sha256()), 64)
        self.assertEqual(default_result.run_inputs, expected_defaults)
        self.assertEqual(
            default_result.run_inputs.snapshot(),
            expected_defaults.snapshot(),
        )
        self.assertEqual(len(default_result.run_inputs.snapshot_sha256()), 64)
        self.assertEqual(
            default_result.run_input_snapshot()["batch_spec"],
            spec.snapshot(),
        )
        self.assertEqual(len(default_result.run_input_sha256()), 64)

        harm_parameters = HarmModelParameters(affordable_spending_share=0.2)
        harm_weights = WelfareHarmWeights(monetary=2.0)
        opportunity_valuation = OpportunityCostValuation(
            adult_sleep_hour_cents=601
        )
        producer_assumptions = ProducerAssumptions(
            development_cost_cents=1_200_001
        )
        epgc_policy = replace(
            default_epgc_policy(),
            maximum_budget_cents=3_000_001,
        )
        with patch(
            "microtx_sim.causal.batch.run_policy_scenario",
            wraps=run_policy_scenario,
        ) as scenario_runner:
            custom_result = run_policy_batch(
                spec,
                country_profiles=PROFILE,
                harm_parameters=harm_parameters,
                harm_weights=harm_weights,
                opportunity_valuation=opportunity_valuation,
                producer_assumptions=producer_assumptions,
                epgc_policy=epgc_policy,
            )

        for name, value in {
            "harm_parameters": harm_parameters,
            "harm_weights": harm_weights,
            "opportunity_valuation": opportunity_valuation,
            "producer_assumptions": producer_assumptions,
            "epgc_policy": epgc_policy,
        }.items():
            self.assertIs(getattr(custom_result.run_inputs, name), value)
        self.assertEqual(scenario_runner.call_count, len(spec.scenarios))
        for call in scenario_runner.call_args_list:
            self.assertIs(call.kwargs["harm_parameters"], harm_parameters)
            self.assertIs(call.kwargs["harm_weights"], harm_weights)
            self.assertIs(
                call.kwargs["opportunity_valuation"],
                opportunity_valuation,
            )
            self.assertIs(
                call.kwargs["producer_assumptions"],
                producer_assumptions,
            )
            self.assertIs(call.kwargs["epgc_policy"], epgc_policy)
        self.assertEqual(
            custom_result.run_inputs,
            PolicyRunInputs(
                harm_parameters=harm_parameters,
                harm_weights=harm_weights,
                opportunity_valuation=opportunity_valuation,
                producer_assumptions=producer_assumptions,
                epgc_policy=epgc_policy,
            ),
        )
        self.assertNotEqual(
            default_result.run_inputs.snapshot_sha256(),
            custom_result.run_inputs.snapshot_sha256(),
        )
        with self.assertRaises(FrozenInstanceError):
            setattr(
                custom_result.run_inputs,
                "harm_parameters",
                HarmModelParameters(),
            )
        with self.assertRaisesRegex(TypeError, "run_inputs"):
            replace(custom_result, run_inputs=object())
        with self.assertRaisesRegex(TypeError, "harm_parameters"):
            run_policy_batch(
                spec,
                country_profiles=PROFILE,
                harm_parameters=object(),  # type: ignore[arg-type]
            )
        for value in ([], 1, "true"):
            with self.subTest(accessibility_eligible=value):
                with self.assertRaisesRegex(TypeError, "must be a boolean"):
                    ProducerAssumptions(  # type: ignore[arg-type]
                        accessibility_eligible=value
                    )

    def test_batch_seed_domain_is_strict_canonical_and_unique(self) -> None:
        maximum = (1 << 64) - 1
        self.assertEqual(
            PolicyBatchSpec(seeds=(maximum, 0, 7)).seeds,
            (0, 7, maximum),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            PolicyBatchSpec(seeds=(7, 1, 7))
        for value in (-1, 1 << 64):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"\[0, 2\*\*64 - 1\]"):
                    PolicyBatchSpec(seeds=(value,))
        for value in (True, 1.0, np.int64(1)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(TypeError, "Python integer"):
                    PolicyBatchSpec(seeds=(value,))  # type: ignore[arg-type]

    def test_batch_materializes_and_validates_scenario_iterables(self) -> None:
        scenario_list = list(required_scenarios())
        expected_scenarios = tuple(scenario_list)
        from_list = PolicyBatchSpec(scenarios=scenario_list)  # type: ignore[arg-type]
        scenario_list.clear()

        self.assertIsInstance(from_list.scenarios, tuple)
        self.assertEqual(from_list.scenarios, expected_scenarios)
        self.assertEqual(
            tuple(item["scenario_id"] for item in from_list.snapshot()["scenarios"]),
            tuple(item.scenario_id.value for item in expected_scenarios),
        )

        from_generator = PolicyBatchSpec(  # type: ignore[arg-type]
            scenarios=(scenario for scenario in expected_scenarios)
        )
        self.assertEqual(from_generator.scenarios, expected_scenarios)
        self.assertEqual(from_generator.snapshot(), from_list.snapshot())

        with self.assertRaisesRegex(TypeError, r"scenarios\[1\].*ScenarioSpec"):
            PolicyBatchSpec(  # type: ignore[arg-type]
                scenarios=(expected_scenarios[0], object())
            )

        with self.assertRaisesRegex(TypeError, "decision_parameters"):
            PolicyBatchSpec(decision_parameters={})  # type: ignore[arg-type]

        normalized = PolicyBatchSpec(
            seeds=(_MutableInt(4),),
            days=_MutableInt(0),
            player_count=_MutableInt(0),
        )
        self.assertIs(type(normalized.seeds[0]), int)
        self.assertIs(type(normalized.days), int)
        self.assertIs(type(normalized.player_count), int)
        normalized_scenario = replace(
            expected_scenarios[0],
            fixed_access_price_cents=_MutableInt(7),
        )
        normalized_producer = ProducerAssumptions(
            development_cost_cents=_MutableInt(1_200_000)
        )
        self.assertIs(type(normalized_scenario.fixed_access_price_cents), int)
        self.assertIs(type(normalized_producer.development_cost_cents), int)

        with self.assertRaisesRegex(TypeError, "spec must be PolicyBatchSpec"):
            run_policy_batch(
                _ExtendedPolicyBatchSpec(days=0, player_count=0),
                country_profiles=PROFILE,
            )

    def test_scenario_specs_reject_wrong_or_mutable_nested_values(self) -> None:
        baseline = required_scenarios()[0]
        invalid_values = (
            ("scenario_id", baseline.scenario_id.value, "ScenarioId"),
            ("label", [], "string"),
            ("mechanics", object(), "MonetisationVector"),
            (
                "mechanics",
                _ExtendedMonetisationVector(),
                "MonetisationVector",
            ),
            ("epgc_enabled", [], "boolean"),
            ("description", [], "string"),
        )
        for field_name, value, message in invalid_values:
            with self.subTest(field_name=field_name):
                arguments = {
                    "scenario_id": baseline.scenario_id,
                    "label": baseline.label,
                    "mechanics": baseline.mechanics,
                    "fixed_access_price_cents": baseline.fixed_access_price_cents,
                    "subscription_price_cents": baseline.subscription_price_cents,
                    "epgc_enabled": baseline.epgc_enabled,
                    "description": baseline.description,
                    field_name: value,
                }
                with self.assertRaisesRegex(TypeError, message):
                    ScenarioSpec(**arguments)  # type: ignore[arg-type]

    def test_seed_order_produces_identical_values_and_bytes(self) -> None:
        parameters = {
            "days": 1,
            "player_count": 12,
            "decision_parameters": DecisionParameters(step_minutes=240),
        }
        forward = run_policy_batch(
            PolicyBatchSpec(seeds=(17, 3, 11), **parameters),
            country_profiles=PROFILE,
        )
        reverse = run_policy_batch(
            PolicyBatchSpec(seeds=(11, 3, 17), **parameters),
            country_profiles=PROFILE,
        )

        self.assertEqual(forward.spec.seeds, (3, 11, 17))
        reordered_digests = replace(
            forward,
            cohort_digest_by_seed=dict(
                reversed(tuple(forward.cohort_digest_by_seed.items()))
            ),
        )
        self.assertEqual(
            tuple(reordered_digests.cohort_digest_by_seed),
            forward.spec.seeds,
        )
        reordered_records = replace(
            forward,
            records=tuple(reversed(forward.records)),
        )
        self.assertEqual(forward.seed_rows(), reverse.seed_rows())
        self.assertEqual(forward.scenario_rows(), reverse.scenario_rows())
        self.assertEqual(forward.seed_rows(), reordered_records.seed_rows())
        self.assertEqual(forward.scenario_rows(), reordered_records.scenario_rows())
        self.assertEqual(_projection_bytes(forward), _projection_bytes(reverse))
        self.assertEqual(
            _projection_bytes(forward),
            _projection_bytes(reordered_records),
        )

    def test_result_and_digest_metadata_reject_invalid_seeds(self) -> None:
        batch = run_policy_batch(
            PolicyBatchSpec(seeds=(1,), days=0, player_count=0),
            country_profiles=PROFILE,
        )
        result = batch.records[0].result
        digest = batch.cohort_digest_by_seed[1]

        wrong_result = replace(result, seed=2)
        wrong_record = replace(batch.records[0], result=wrong_result)
        with self.assertRaisesRegex(ValueError, "match the batch spec"):
            replace(batch, records=(wrong_record, *batch.records[1:]))

        for value in (-1, 1 << 64):
            with self.subTest(boundary="result", value=value):
                with self.assertRaisesRegex(ValueError, r"\[0, 2\*\*64 - 1\]"):
                    replace(result, seed=value)
            with self.subTest(boundary="digest", value=value):
                with self.assertRaisesRegex(ValueError, r"\[0, 2\*\*64 - 1\]"):
                    replace(batch, cohort_digest_by_seed={value: digest})
        for value in (True, 1.0, np.int64(1)):
            with self.subTest(boundary="result", value=value):
                with self.assertRaisesRegex(TypeError, "Python integer"):
                    replace(result, seed=value)
            with self.subTest(boundary="digest", value=value):
                with self.assertRaisesRegex(TypeError, "Python integer"):
                    replace(batch, cohort_digest_by_seed={value: digest})

    def test_effect_vs_safe_schema_rejects_a_different_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "effect_vs_safe"):
            PolicyBatchSpec(reference_scenario=ScenarioId.BASELINE_F2P)

    def test_catalogue_contains_exactly_seven_explicit_scenarios(self) -> None:
        scenarios = required_scenarios()
        self.assertEqual(tuple(item.scenario_id for item in scenarios), tuple(ScenarioId))
        self.assertTrue(all(not item.mechanics.personalized_offers for item in scenarios))
        baseline = scenarios[0].mechanics
        no_random = scenarios[2].mechanics
        no_time = scenarios[3].mechanics
        self.assertEqual(no_random.paid_random_rewards, 0.0)
        self.assertEqual(no_time.time_limited_offers, 0.0)
        self.assertEqual(no_random.time_limited_offers, baseline.time_limited_offers)
        self.assertEqual(no_time.paid_random_rewards, baseline.paid_random_rewards)

    def test_complete_two_seed_batch_reports_all_required_outcomes(self) -> None:
        batch = run_policy_batch(
            PolicyBatchSpec(
                seeds=(11, 22),
                days=1,
                player_count=32,
                decision_parameters=DecisionParameters(step_minutes=120),
            ),
            country_profiles=PROFILE,
        )
        self.assertEqual(len(batch.records), 14)
        self.assertEqual(len(batch.seed_rows()), 14)
        self.assertEqual(len(batch.scenario_rows()), 7)
        self.assertEqual(len(batch.epgc_rows()), 2)
        for seed in (11, 22):
            seed_records = [item for item in batch.records if item.result.seed == seed]
            self.assertEqual(len({item.cohort_digest for item in seed_records}), 1)
        for record in batch.records:
            result = record.result
            self.assertTrue(
                np.all(result.spending_cents <= result.disposable_budget_cents)
            )
            self.assertEqual(
                result.total_revenue_cents,
                sum(result.revenue_composition_cents.values()),
            )
            if result.scenario.scenario_id is ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION:
                self.assertEqual(record.mean_harm_effect_vs_safe, 0.0)
        epgc = [
            item.result
            for item in batch.records
            if item.result.scenario.scenario_id is ScenarioId.EPGC
        ]
        self.assertTrue(all(item.epgc is not None for item in epgc))
        self.assertTrue(
            all(item.revenue_composition_cents["public_contract"] > 0 for item in epgc)
        )
        summary = batch.scenario_rows()[0]
        self.assertIn("mean_harm_variance", summary)
        self.assertIn("mean_harm_ci95_low", summary)
        self.assertIn("mean_harm_ci95_high", summary)
        revenue_variances = [
            key
            for key in summary
            if key.startswith("revenue_") and key.endswith("_variance")
        ]
        self.assertTrue(revenue_variances)
        for variance_key in revenue_variances:
            stem = variance_key.removesuffix("_variance")
            self.assertIn(f"{stem}_mean", summary)
            self.assertIn(f"{stem}_sd", summary)
            self.assertIn(f"{stem}_ci95_low", summary)
            self.assertIn(f"{stem}_ci95_high", summary)

        opportunity_rows = batch.opportunity_rows()
        self.assertEqual(len(opportunity_rows), 7 * 5)
        self.assertEqual(
            {
                row["component"]
                for row in opportunity_rows
                if row["scenario_id"] == ScenarioId.BASELINE_F2P.value
            },
            {
                "sleep",
                "work_study",
                "family_social",
                "physical_activity",
                "all_displaced_activities",
            },
        )

    def test_scenario_iteration_order_cannot_change_results(self) -> None:
        params = dict(
            seeds=(909,),
            days=1,
            player_count=24,
            decision_parameters=DecisionParameters(step_minutes=120),
        )
        forward = run_policy_batch(
            PolicyBatchSpec(**params), country_profiles=PROFILE
        )
        reverse = run_policy_batch(
            PolicyBatchSpec(scenarios=tuple(reversed(required_scenarios())), **params),
            country_profiles=PROFILE,
        )
        forward_rows = {
            row["scenario_id"]: row for row in forward.seed_rows()
        }
        reverse_rows = {
            row["scenario_id"]: row for row in reverse.seed_rows()
        }
        self.assertEqual(forward_rows, reverse_rows)

    def test_zero_player_batch_is_valid_and_finite(self) -> None:
        batch = run_policy_batch(
            PolicyBatchSpec(
                seeds=(3,),
                days=0,
                player_count=0,
                decision_parameters=DecisionParameters(step_minutes=240),
            ),
            country_profiles=PROFILE,
        )
        self.assertEqual(len(batch.records), 7)
        for row in batch.seed_rows():
            self.assertEqual(row["player_count"], 0)
            self.assertEqual(row["mean_harm"], 0.0)
            self.assertEqual(row["total_spending_cents"], 0)
        for row in batch.opportunity_rows():
            self.assertEqual(row["mean_minutes"], 0.0)
            self.assertEqual(row["mean_burden"], 0.0)

    def test_custom_profile_tuple_is_retained_and_content_fingerprinted(self) -> None:
        spec = PolicyBatchSpec(
            seeds=(17,),
            days=0,
            player_count=0,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        original_profile = CountryProfile(code="CUSTOM")
        changed_profile = replace(original_profile, awareness_mean=0.51)
        original = run_policy_batch(
            spec,
            country_profiles=(original_profile,),
        )
        changed = run_policy_batch(
            spec,
            country_profiles=(changed_profile,),
        )

        self.assertEqual(original.country_profiles, (original_profile,))
        self.assertIsNotNone(original.profile_input_lineage)
        lineage = original.profile_input_lineage
        assert lineage is not None
        self.assertEqual(lineage.lineage_status, "unregistered_custom_profiles")
        self.assertIsNone(lineage.source_registry_sha256)
        snapshot_profiles = lineage.snapshot["country_profiles"]
        self.assertIsInstance(snapshot_profiles, list)
        snapshot_profile = snapshot_profiles[0]
        self.assertEqual(
            set(snapshot_profile),
            {descriptor.name for descriptor in fields(CountryProfile)},
        )
        self.assertNotEqual(
            original.profile_input_lineage.fingerprint_sha256,
            changed.profile_input_lineage.fingerprint_sha256,
        )
        with self.assertRaisesRegex(
            ValueError,
            "do not match the fingerprinted snapshot",
        ):
            replace(
                changed,
                profile_input_lineage=original.profile_input_lineage,
            )


if __name__ == "__main__":
    unittest.main()
