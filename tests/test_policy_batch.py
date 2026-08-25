from __future__ import annotations

from dataclasses import fields, replace
import json
import unittest

import numpy as np

from microtx_sim.causal.batch import (
    PolicyBatchResult,
    PolicyBatchSpec,
    run_policy_batch,
)
from microtx_sim.causal.scenarios import ScenarioId, required_scenarios
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.consumers.population import CountryProfile


PROFILE = (CountryProfile(code="XX"),)


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
