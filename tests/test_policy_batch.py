from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field, fields, replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_population_projection_adapter import _complete_adapter  # noqa: E402

from microtx_sim.causal.batch import (
    PolicyBatchCheckpoint,
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
from microtx_sim.data.population_execution import (
    build_population_seed_execution_record,
)
from microtx_sim.execution_attestation import (
    CampaignExecutionRejectedError,
    ExecutionVerificationPhase,
)
from microtx_sim.data.population_projection import initialize_population_projection
from microtx_sim.metrics.harm import (
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmWeights,
)
from microtx_sim.outputs.checkpoints import ExploratoryCheckpointRecorder
from microtx_sim.simulation.policy_orchestrator import (
    ProducerAssumptions,
    default_epgc_policy,
    run_policy_scenario,
)


PROFILE = (CountryProfile(code="XX"),)


def _campaign_attestation_kwargs() -> dict[str, object]:
    return {
        "campaign_receipt": object(),
        "campaign_verification": SimpleNamespace(
            phase=ExecutionVerificationPhase.PRE_EXECUTION
        ),
    }


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
    def test_checkpoint_callback_receives_complete_seed_prefixes_only(self) -> None:
        spec = PolicyBatchSpec(
            seeds=(5, 6),
            days=0,
            player_count=4,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        checkpoints: list[PolicyBatchCheckpoint] = []

        run_policy_batch(
            spec,
            country_profiles=PROFILE,
            checkpoint_callback=checkpoints.append,
        )

        self.assertEqual(
            [item.completed_seeds for item in checkpoints],
            [(5,), (5, 6)],
        )
        self.assertEqual(
            [len(item.records) for item in checkpoints],
            [len(spec.scenarios), 2 * len(spec.scenarios)],
        )
        rows = checkpoints[-1].nonmonetary_diagnostic_rows()
        self.assertTrue(rows)
        self.assertFalse(any("cents" in key for row in rows for key in row))
        self.assertEqual(
            {row["interpretation"] for row in rows},
            {"UNWEIGHTED_DIAGNOSTIC_ONLY"},
        )
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / "progress"
            first = ExploratoryCheckpointRecorder.start(
                progress,
                expected_seeds=spec.seeds,
                config_sha256="a" * 64,
                exploratory_plan_id="exploratory.test.v1",
                exploratory_plan_sha256="b" * 64,
                launch_command=("microtx-sim", "policy-batch", "test.toml"),
            )
            first(checkpoints[0])
            first.mark_interrupted()
            first_payload = json.loads(first.progress_path.read_text("utf-8"))
            self.assertEqual(first_payload["status"], "INTERRUPTED")
            self.assertEqual(first_payload["retained_seed_count"], 1)
            self.assertFalse(first_payload["resume_supported"])
            self.assertTrue(first.partial_results_path.is_file())
            second = ExploratoryCheckpointRecorder.start(
                progress,
                expected_seeds=spec.seeds,
                config_sha256="a" * 64,
                exploratory_plan_id="exploratory.test.v1",
                exploratory_plan_sha256="b" * 64,
                launch_command=("microtx-sim", "policy-batch", "test.toml"),
            )
            self.assertEqual(first.attempt_id, "attempt-000001")
            self.assertEqual(second.attempt_id, "attempt-000002")
            self.assertTrue(first.progress_path.is_file())

    def test_campaign_requires_attestation_before_any_initializer(self) -> None:
        spec = PolicyBatchSpec(seeds=(13,), days=0, player_count=12)
        with (
            patch(
                "microtx_sim.causal.batch.validate_population_campaign_preflight"
            ) as population_gate,
            patch("microtx_sim.causal.batch.initialize_player_table") as legacy,
            patch(
                "microtx_sim.causal.batch.initialize_population_projection"
            ) as projected,
            patch("microtx_sim.causal.batch.run_policy_scenario") as scenario,
        ):
            with self.assertRaisesRegex(
                CampaignExecutionRejectedError,
                "preverified execution receipt",
            ):
                run_policy_batch(
                    spec,
                    country_profiles=PROFILE,
                    campaign=True,
                )
        population_gate.assert_not_called()
        legacy.assert_not_called()
        projected.assert_not_called()
        scenario.assert_not_called()

    def test_campaign_without_adapter_rejects_before_any_initializer(self) -> None:
        spec = PolicyBatchSpec(seeds=(13,), days=0, player_count=12)
        with (
            patch("microtx_sim.causal.batch.require_campaign_execution"),
            patch("microtx_sim.causal.batch.initialize_player_table") as legacy,
            patch(
                "microtx_sim.causal.batch.initialize_population_projection"
            ) as projected,
            patch("microtx_sim.causal.batch.run_policy_scenario") as scenario,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "legacy population fallback is prohibited",
            ):
                run_policy_batch(
                    spec,
                    country_profiles=PROFILE,
                    campaign=True,
                    **_campaign_attestation_kwargs(),
                )
        legacy.assert_not_called()
        projected.assert_not_called()
        scenario.assert_not_called()

    def test_campaign_initializes_one_population_per_seed_for_all_scenarios(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            spec = PolicyBatchSpec(
                seeds=(17, 29),
                days=0,
                player_count=12,
                decision_parameters=DecisionParameters(step_minutes=240),
            )
            with (
                patch(
                    "microtx_sim.causal.batch.validate_population_campaign_preflight",
                    side_effect=lambda adapter: adapter,
                ),
                patch("microtx_sim.causal.batch.require_campaign_execution"),
                patch(
                    "microtx_sim.causal.batch.initialize_population_projection",
                    wraps=initialize_population_projection,
                ) as initialize,
                patch(
                    "microtx_sim.causal.batch.run_policy_scenario",
                    wraps=run_policy_scenario,
                ) as scenario,
            ):
                result = run_policy_batch(
                    spec,
                    country_profiles=(CountryProfile(code="UK"),),
                    population_adapter=adapter,
                    campaign=True,
                    **_campaign_attestation_kwargs(),
                )

            self.assertEqual(initialize.call_count, len(spec.seeds))
            self.assertEqual(
                scenario.call_count,
                len(spec.seeds) * len(spec.scenarios),
            )
            self.assertIsNotNone(result.population_execution_lineage)

    def test_campaign_reattests_balance_and_weights_before_first_scenario(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            spec = PolicyBatchSpec(
                seeds=(31,),
                days=0,
                player_count=12,
                decision_parameters=DecisionParameters(step_minutes=240),
            )

            def invalid_balance(*args, **kwargs):
                record = build_population_seed_execution_record(*args, **kwargs)
                object.__setattr__(record.balance, "exact_balance_passed", False)
                return record

            with (
                patch(
                    "microtx_sim.causal.batch.validate_population_campaign_preflight",
                    side_effect=lambda adapter: adapter,
                ),
                patch("microtx_sim.causal.batch.require_campaign_execution"),
                patch(
                    "microtx_sim.causal.batch.build_population_seed_execution_record",
                    side_effect=invalid_balance,
                ),
                patch("microtx_sim.causal.batch.run_policy_scenario") as scenario,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "exact balance artifact",
                ):
                    run_policy_batch(
                        spec,
                        country_profiles=(CountryProfile(code="UK"),),
                        population_adapter=adapter,
                        campaign=True,
                        **_campaign_attestation_kwargs(),
                    )
            scenario.assert_not_called()

            def invalid_weights(*args, **kwargs):
                record = build_population_seed_execution_record(*args, **kwargs)
                weights = record.exact_weights
                altered = replace(
                    weights,
                    weight_numerators=(
                        weights.weight_numerators[0]
                        + weights.weight_denominators[0],
                        *weights.weight_numerators[1:],
                    ),
                )
                object.__setattr__(record, "exact_weights", altered)
                return record

            with (
                patch(
                    "microtx_sim.causal.batch.validate_population_campaign_preflight",
                    side_effect=lambda adapter: adapter,
                ),
                patch("microtx_sim.causal.batch.require_campaign_execution"),
                patch(
                    "microtx_sim.causal.batch.build_population_seed_execution_record",
                    side_effect=invalid_weights,
                ),
                patch("microtx_sim.causal.batch.run_policy_scenario") as scenario,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "weights must sum exactly to one",
                ):
                    run_policy_batch(
                        spec,
                        country_profiles=(CountryProfile(code="UK"),),
                        population_adapter=adapter,
                        campaign=True,
                        **_campaign_attestation_kwargs(),
                    )
            scenario.assert_not_called()

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

    def test_batch_rejects_drift_in_every_pretreatment_result_field(self) -> None:
        batch = run_policy_batch(
            PolicyBatchSpec(seeds=(31,), days=0, player_count=4),
            country_profiles=PROFILE,
        )
        record = batch.records[0]
        field_names = (
            "player_ids",
            "is_minor",
            "age_years",
            "jurisdiction",
            "baseline_vulnerability",
            "disposable_budget_cents",
        )

        for name in field_names:
            with self.subTest(field=name, drift="value"):
                changed = getattr(record.result, name).copy()
                if changed.dtype == np.bool_:
                    changed[0] = not bool(changed[0])
                else:
                    changed[0] += 1
                changed_result = replace(record.result, **{name: changed})
                changed_records = list(batch.records)
                changed_records[0] = replace(record, result=changed_result)
                with self.assertRaisesRegex(ValueError, name):
                    replace(batch, records=tuple(changed_records))

        changed_dtype = record.result.player_ids.astype(np.int32)
        changed_result = replace(record.result, player_ids=changed_dtype)
        changed_records = list(batch.records)
        changed_records[0] = replace(record, result=changed_result)
        with self.assertRaisesRegex(ValueError, "player_ids"):
            replace(batch, records=tuple(changed_records))

        shared_dtype_drift = tuple(
            replace(
                item,
                result=replace(
                    item.result,
                    player_ids=item.result.player_ids.astype(np.int32),
                ),
            )
            for item in batch.records
        )
        with self.assertRaisesRegex(ValueError, "dtype int64"):
            replace(batch, records=shared_dtype_drift)

        rank_drift = record.result.player_ids.reshape(2, 2)
        changed_result = replace(record.result, player_ids=rank_drift)
        changed_records = list(batch.records)
        changed_records[0] = replace(record, result=changed_result)
        with self.assertRaisesRegex(ValueError, r"shape \(4,\)"):
            replace(batch, records=tuple(changed_records))

        changed_scenario = replace(
            record.result.scenario,
            fixed_access_price_cents=(
                record.result.scenario.fixed_access_price_cents + 1
            ),
        )
        changed_result = replace(record.result, scenario=changed_scenario)
        changed_records = list(batch.records)
        changed_records[0] = replace(record, result=changed_result)
        with self.assertRaisesRegex(ValueError, "exactly match the batch spec"):
            replace(batch, records=tuple(changed_records))

        changed_days = replace(record.result, days=record.result.days + 1)
        changed_records = list(batch.records)
        changed_records[0] = replace(record, result=changed_days)
        with self.assertRaisesRegex(ValueError, "days"):
            replace(batch, records=tuple(changed_records))

        wrong_count_spec = replace(
            batch.spec,
            player_count=batch.spec.player_count + 1,
        )
        with self.assertRaisesRegex(ValueError, "player count"):
            replace(batch, spec=wrong_count_spec)

    def test_batch_rejects_shared_invalid_pretreatment_domains(self) -> None:
        batch = run_policy_batch(
            PolicyBatchSpec(seeds=(37,), days=0, player_count=4),
            country_profiles=PROFILE,
        )

        with self.assertRaisesRegex(
            ValueError,
            "country_profiles are required for non-empty policy cohorts",
        ):
            replace(
                batch,
                country_profiles=(),
                profile_input_lineage=None,
            )

        def replace_field_in_every_record(
            field_name: str,
            values: np.ndarray,
        ) -> tuple:
            return tuple(
                replace(
                    record,
                    result=replace(
                        record.result,
                        **{field_name: values.copy()},
                    ),
                )
                for record in batch.records
            )

        first = batch.records[0].result
        invalid_cases: list[tuple[str, np.ndarray, str]] = []

        duplicate_ids = first.player_ids.copy()
        duplicate_ids[1] = duplicate_ids[0]
        invalid_cases.append(("player_ids", duplicate_ids, "unique"))

        negative_ids = first.player_ids.copy()
        negative_ids[0] = -1
        invalid_cases.append(("player_ids", negative_ids, "non-negative"))

        negative_age = first.age_years.copy()
        negative_age[0] = -1
        invalid_cases.append(("age_years", negative_age, "cannot be negative"))

        negative_jurisdiction = first.jurisdiction.copy()
        negative_jurisdiction[0] = -1
        invalid_cases.append(
            ("jurisdiction", negative_jurisdiction, "unknown code")
        )

        unknown_jurisdiction = first.jurisdiction.copy()
        unknown_jurisdiction[0] = len(PROFILE)
        invalid_cases.append(
            ("jurisdiction", unknown_jurisdiction, "unknown code")
        )

        for invalid_value in (np.inf, -0.01, 1.01):
            vulnerability = first.baseline_vulnerability.copy()
            vulnerability[0] = invalid_value
            invalid_cases.append(
                (
                    "baseline_vulnerability",
                    vulnerability,
                    r"finite and in \[0, 1\]",
                )
            )

        negative_budget = first.disposable_budget_cents.copy()
        negative_budget[0] = -1
        invalid_cases.append(
            ("disposable_budget_cents", negative_budget, "cannot be negative")
        )

        inconsistent_minor = first.is_minor.copy()
        inconsistent_minor[0] = not bool(inconsistent_minor[0])
        invalid_cases.append(("is_minor", inconsistent_minor, "inconsistent"))

        for field_name, values, message in invalid_cases:
            with self.subTest(field=field_name, message=message):
                records = replace_field_in_every_record(field_name, values)
                with self.assertRaisesRegex(ValueError, message):
                    replace(batch, records=records)

    def test_batch_recomputes_every_retained_effect_scalar(self) -> None:
        batch = run_policy_batch(
            PolicyBatchSpec(seeds=(39,), days=0, player_count=4),
            country_profiles=PROFILE,
        )
        record = batch.records[0]

        with self.assertRaisesRegex(ValueError, "must be finite"):
            replace(
                batch,
                records=(
                    replace(record, mean_harm_effect_vs_safe=float("nan")),
                    *batch.records[1:],
                ),
            )
        with self.assertRaisesRegex(TypeError, "built-in float"):
            replace(
                batch,
                records=(
                    replace(record, mean_harm_effect_vs_safe=np.float64(0.0)),
                    *batch.records[1:],
                ),
            )
        with self.assertRaisesRegex(ValueError, "does not match paired results"):
            replace(
                batch,
                records=(
                    replace(
                        record,
                        mean_harm_effect_vs_safe=(
                            record.mean_harm_effect_vs_safe + 0.25
                        ),
                    ),
                    *batch.records[1:],
                ),
            )

        integer_fields = (
            "total_spending_effect_vs_safe_cents",
            "harmful_spending_effect_vs_safe_cents",
            "total_revenue_effect_vs_safe_cents",
        )
        for name in integer_fields:
            with self.subTest(field=name, failure="type"):
                changed = replace(record, **{name: np.int64(0)})
                with self.assertRaisesRegex(TypeError, "built-in integer"):
                    replace(
                        batch,
                        records=(changed, *batch.records[1:]),
                    )
            with self.subTest(field=name, failure="value"):
                changed = replace(record, **{name: getattr(record, name) + 1})
                with self.assertRaisesRegex(
                    ValueError,
                    "does not match paired results",
                ):
                    replace(
                        batch,
                        records=(changed, *batch.records[1:]),
                    )

    def test_batch_owns_read_only_copies_of_all_result_arrays(self) -> None:
        batch = run_policy_batch(
            PolicyBatchSpec(seeds=(43,), days=0, player_count=4),
            country_profiles=PROFILE,
        )
        result = batch.records[0].result
        top_level_arrays = (
            "player_ids",
            "is_minor",
            "age_years",
            "jurisdiction",
            "baseline_vulnerability",
            "disposable_budget_cents",
            "spending_cents",
            "composite_harm",
            "enjoyment",
            "high_risk",
            "action_minutes",
        )
        for name in top_level_arrays:
            with self.subTest(field=name):
                values = getattr(result, name)
                self.assertFalse(values.flags.writeable)
                with self.assertRaises(ValueError):
                    values.setflags(write=True)
        for descriptor in fields(result.harm):
            with self.subTest(harm_field=descriptor.name):
                values = getattr(result.harm, descriptor.name)
                self.assertFalse(values.flags.writeable)
                with self.assertRaises(ValueError):
                    values.setflags(write=True)

        source_ids = result.player_ids.copy()
        source_result = replace(result, player_ids=source_ids)
        source_record = replace(batch.records[0], result=source_result)
        retained = replace(
            batch,
            records=(source_record, *batch.records[1:]),
        )
        source_ids[0] = 999
        self.assertNotEqual(int(retained.records[0].result.player_ids[0]), 999)
        with self.assertRaises(ValueError):
            retained.records[0].result.player_ids[0] = 999

    def test_batch_rejects_shared_invalid_outcome_arrays(self) -> None:
        batch = run_policy_batch(
            PolicyBatchSpec(seeds=(47,), days=0, player_count=4),
            country_profiles=PROFILE,
        )

        def replace_result_in_every_record(**changes) -> tuple:
            return tuple(
                replace(
                    record,
                    result=replace(record.result, **changes),
                )
                for record in batch.records
            )

        result = batch.records[0].result
        cases = (
            (
                {"spending_cents": result.spending_cents.astype(np.int32)},
                "spending_cents must have dtype int64",
            ),
            (
                {
                    "spending_cents": np.full(
                        result.spending_cents.shape,
                        -1,
                        dtype=np.int64,
                    )
                },
                "spending_cents cannot be negative",
            ),
            (
                {
                    "spending_cents": (
                        result.disposable_budget_cents + 1
                    ).astype(np.int64)
                },
                "cannot exceed disposable budget",
            ),
            (
                {"high_risk": result.high_risk.astype(np.int8)},
                "high_risk must have dtype bool",
            ),
            (
                {"action_minutes": result.action_minutes.astype(np.int32)},
                "action_minutes must have dtype int64",
            ),
            (
                {
                    "action_minutes": np.full(
                        result.action_minutes.shape,
                        -1,
                        dtype=np.int64,
                    )
                },
                "action_minutes cannot be negative",
            ),
        )
        for changes, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    replace(
                        batch,
                        records=replace_result_in_every_record(**changes),
                    )

        forged_composite = np.full(
            result.composite_harm.shape,
            0.25,
            dtype=np.float64,
        )
        with self.assertRaisesRegex(ValueError, "does not match component"):
            replace(
                batch,
                records=replace_result_in_every_record(
                    composite_harm=forged_composite,
                ),
            )

        component_scores = result.harm.component_scores.copy()
        component_scores[:, 0] = 0.5
        forged_harm = replace(result.harm, component_scores=component_scores)
        with self.assertRaisesRegex(ValueError, "does not match component"):
            replace(
                batch,
                records=replace_result_in_every_record(harm=forged_harm),
            )

    def test_branch_cannot_mutate_the_shared_pre_treatment_cohort(self) -> None:
        def mutating_runner(players, life, scenario, **kwargs):
            result = run_policy_scenario(
                players,
                life,
                scenario,
                **kwargs,
            )
            life.wellbeing[0] += 0.01
            return result

        with patch(
            "microtx_sim.causal.batch.run_policy_scenario",
            side_effect=mutating_runner,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "mutated the shared pre-treatment cohort",
            ):
                run_policy_batch(
                    PolicyBatchSpec(seeds=(41,), days=0, player_count=4),
                    country_profiles=PROFILE,
                )

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
