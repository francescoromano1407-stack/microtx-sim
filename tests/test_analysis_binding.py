from __future__ import annotations

from dataclasses import fields, replace
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_population_projection_adapter import _complete_adapter  # noqa: E402

from microtx_sim.causal.analysis_binding import (  # noqa: E402
    ANALYSIS_BINDING_SCHEMA_VERSION,
    AnalysisBindingValidationError,
    resolve_run_analysis_binding,
    validate_analysis_plan_inputs,
)
from microtx_sim.causal.analysis_plan import (  # noqa: E402
    AnalysisEstimandRole,
    CanonicalPopulationInclusionPredicate,
    FixedSeedStoppingRule,
    PlannedPopulationEstimand,
    PopulationMinorFilter,
    PopulationOutcomeMetric,
    analysis_plan_harm_weights_sha256,
    build_prospective_analysis_plan,
)
from microtx_sim.causal.batch import (  # noqa: E402
    PolicyBatchResult,
    PolicyBatchSpec,
    PolicyRunInputs,
    resolve_policy_run_inputs,
    run_policy_batch,
)
from microtx_sim.causal.design import assess_causal_design  # noqa: E402
from microtx_sim.causal.scenarios import ScenarioId  # noqa: E402
from microtx_sim.consumers.decision import DecisionParameters  # noqa: E402
from microtx_sim.consumers.population import CountryProfile  # noqa: E402
from microtx_sim.data.lineage import build_profile_input_lineage  # noqa: E402
from microtx_sim.data.population_evidence import PopulationEstimandRole  # noqa: E402
from microtx_sim.data.population_evidence import (  # noqa: E402
    PopulationGamingState,
    PopulationPayerHistoryState,
)
from microtx_sim.data.population_execution import (  # noqa: E402
    PopulationExecutionLineage,
    PopulationSeedExecutionRecord,
    population_execution_input_sha256,
)
from microtx_sim.metrics.population_estimands import (  # noqa: E402
    ExactPopulationWeights,
    PopulationCurrencySemantics,
    PopulationInclusionField,
    PopulationInclusionRule,
    PopulationInclusionTiming,
    PopulationPeriodSemantics,
    paired_weighted_mean_difference,
)
from microtx_sim.outputs.metric_contracts import (  # noqa: E402
    metric_contract_registry_sha256,
)
from microtx_sim.outputs.population import (  # noqa: E402
    write_target_population_estimands,
)
from microtx_sim.outputs.schema import (  # noqa: E402
    TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256,
)


_CANONICAL_INCLUSION_FIELDS = tuple(
    sorted(PopulationInclusionField, key=lambda item: item.value)
)
_PROFILES = (CountryProfile(code="UK"),)
_PROFILE_INPUT_LINEAGE = build_profile_input_lineage(_PROFILES)


def _predicate(
    *,
    age_min_inclusive: int = 0,
    age_max_exclusive: int = 32_768,
) -> CanonicalPopulationInclusionPredicate:
    return CanonicalPopulationInclusionPredicate(
        rule=PopulationInclusionRule(
            rule_id="all.projected.players",
            description="All players selected from pre-treatment projection fields.",
            source_fields=_CANONICAL_INCLUSION_FIELDS,
            timing=PopulationInclusionTiming.PRETREATMENT,
            evidence_role=PopulationEstimandRole.CALIBRATION,
        ),
        jurisdiction_codes=(),
        age_min_inclusive=age_min_inclusive,
        age_max_exclusive=age_max_exclusive,
        minor_filter=PopulationMinorFilter.ANY,
        monthly_disposable_income_band_ids=(),
        household_type_ids=(),
        gaming_states=(),
        payer_history_states=(),
    )


def _period() -> PopulationPeriodSemantics:
    return PopulationPeriodSemantics(
        period_start=date(2026, 1, 1),
        period_end=date(2026, 1, 1),
        description="One-day synthetic policy horizon.",
    )


def _planned_estimand(
    *,
    reference: ScenarioId = ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
    comparison: ScenarioId = ScenarioId.BASELINE_F2P,
    outcome_metric: PopulationOutcomeMetric = (
        PopulationOutcomeMetric.COMPOSITE_HARM
    ),
    metric_contract_id: str = "player_outcomes.csv:composite_harm",
    predicate: CanonicalPopulationInclusionPredicate | None = None,
    currency: PopulationCurrencySemantics | None = None,
) -> PlannedPopulationEstimand:
    return PlannedPopulationEstimand(
        estimand_id="primary.policy.effect",
        role=AnalysisEstimandRole.PRIMARY,
        reference_scenario_id=reference,
        comparison_scenario_id=comparison,
        outcome_metric=outcome_metric,
        metric_contract_id=metric_contract_id,
        inclusion_predicate=_predicate() if predicate is None else predicate,
        period=_period(),
        currency=currency,
    )


def _plan(
    spec: PolicyBatchSpec,
    run_inputs: PolicyRunInputs,
    adapter: object,
    *,
    estimand: PlannedPopulationEstimand | None = None,
    estimands: tuple[PlannedPopulationEstimand, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    overrides: dict[str, str] | None = None,
):
    if estimand is not None and estimands is not None:
        raise ValueError("choose estimand or estimands, not both")
    bindings = {
        "expected_causal_design_sha256": assess_causal_design(
            spec.scenarios
        ).design_sha256(),
        "expected_batch_spec_sha256": spec.snapshot_sha256(),
        "expected_model_inputs_sha256": run_inputs.snapshot_sha256(),
        "expected_population_input_sha256": population_execution_input_sha256(
            adapter
        ),
        "expected_profile_input_sha256": (
            _PROFILE_INPUT_LINEAGE.fingerprint_sha256
        ),
        "expected_metric_contract_sha256": metric_contract_registry_sha256(),
        "expected_harm_weights_sha256": analysis_plan_harm_weights_sha256(
            run_inputs.harm_weights
        ),
        "expected_output_profile_sha256": (
            TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256
        ),
    }
    bindings.update(overrides or {})
    return build_prospective_analysis_plan(
        plan_id="test.prospective.analysis",
        stopping_rule=FixedSeedStoppingRule(
            spec.seeds if seeds is None else seeds
        ),
        estimands=(
            estimands
            if estimands is not None
            else (_planned_estimand() if estimand is None else estimand,)
        ),
        **bindings,
    )


def _run_batch(
    spec: PolicyBatchSpec,
    run_inputs: PolicyRunInputs,
    adapter: object,
) -> PolicyBatchResult:
    return run_policy_batch(
        spec,
        country_profiles=_PROFILES,
        harm_parameters=run_inputs.harm_parameters,
        harm_weights=run_inputs.harm_weights,
        opportunity_valuation=run_inputs.opportunity_valuation,
        producer_assumptions=run_inputs.producer_assumptions,
        epgc_policy=run_inputs.epgc_policy,
        population_adapter=adapter,
    )


def _unsafe_clone(value: object, **changes: object):
    clone = object.__new__(type(value))
    for item in fields(value):
        object.__setattr__(
            clone,
            item.name,
            changes.get(item.name, getattr(value, item.name)),
        )
    return clone


class AnalysisBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        _verification, _apportionment, _path, _mapping, cls.adapter = (
            _complete_adapter(root)
        )
        cls.spec = PolicyBatchSpec(
            seeds=(29, 17),
            days=1,
            player_count=12,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        cls.run_inputs = resolve_policy_run_inputs()
        cls.planned = _planned_estimand()
        cls.plan = _plan(
            cls.spec,
            cls.run_inputs,
            cls.adapter,
            estimand=cls.planned,
        )
        cls.batch = _run_batch(cls.spec, cls.run_inputs, cls.adapter)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_resolves_deterministic_writer_ready_exact_pairs(self) -> None:
        self.assertIsNone(
            validate_analysis_plan_inputs(
                self.plan,
                batch_spec=self.spec,
                run_inputs=self.run_inputs,
                population_adapter=self.adapter,
                profile_input_lineage=_PROFILE_INPUT_LINEAGE,
            )
        )
        binding = resolve_run_analysis_binding(self.plan, self.batch)
        self.assertEqual(binding.schema_version, ANALYSIS_BINDING_SCHEMA_VERSION)
        self.assertEqual(binding.seeds, (17, 29))
        self.assertEqual(
            tuple(item.seed for item in binding.seed_bindings),
            (17, 29),
        )
        self.assertEqual(len(binding.writer_pairs), 2)
        self.assertFalse(binding.preregistered)
        self.assertFalse(binding.campaign_ready)
        expected_campaign_blockers = (
            "analysis_binding.external_registration=unregistered",
            "analysis_binding.schema_v1=campaign_ineligible",
            "analysis_binding.execution_calendar_anchor=unbound",
            "analysis_binding.cross_seed_aggregation_uncertainty=unresolved",
            "analysis_binding.model_implementation_environment_identity=unbound",
        )
        self.assertEqual(binding.campaign_blockers, expected_campaign_blockers)
        self.assertEqual(
            binding.profile_input_sha256,
            _PROFILE_INPUT_LINEAGE.fingerprint_sha256,
        )
        self.assertEqual(
            binding.binding_sha256,
            binding.manifest_payload()["binding_sha256"],
        )

        assert self.batch.population_execution_lineage is not None
        records = {
            (record.result.seed, record.result.scenario.scenario_id): record.result
            for record in self.batch.records
        }
        observed_weight_hashes: list[str] = []
        for item in binding.seed_bindings:
            self.assertEqual(
                item.campaign_blockers,
                expected_campaign_blockers,
            )
            population_record = (
                self.batch.population_execution_lineage.record_for_seed(item.seed)
            )
            self.assertEqual(
                item.selected_weights,
                population_record.exact_weights,
            )
            self.assertEqual(
                item.spec.design_weights_sha256,
                population_record.exact_weights.design_sha256,
            )
            expected = paired_weighted_mean_difference(
                item.spec,
                population_record.exact_weights,
                records[(item.seed, ScenarioId.BASELINE_F2P)].composite_harm,
                population_record.exact_weights,
                records[
                    (item.seed, ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION)
                ].composite_harm,
            )
            self.assertEqual(item.result, expected)
            observed_weight_hashes.append(item.spec.design_weights_sha256)
        self.assertEqual(len(set(observed_weight_hashes)), 2)

        with tempfile.TemporaryDirectory() as directory:
            paths = write_target_population_estimands(
                Path(directory),
                binding.writer_pairs,
            )
            self.assertTrue(paths["estimands"].is_file())
            self.assertTrue(paths["metadata"].is_file())

    def test_swapped_scenario_direction_is_exactly_negated_and_bound(self) -> None:
        forward = resolve_run_analysis_binding(self.plan, self.batch)
        reversed_estimand = _planned_estimand(
            reference=ScenarioId.BASELINE_F2P,
            comparison=ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
        )
        reversed_plan = _plan(
            self.spec,
            self.run_inputs,
            self.adapter,
            estimand=reversed_estimand,
        )
        reversed_binding = resolve_run_analysis_binding(
            reversed_plan,
            self.batch,
        )
        for forward_item, reversed_item in zip(
            forward.seed_bindings,
            reversed_binding.seed_bindings,
        ):
            self.assertEqual(
                reversed_item.result.value_fraction,
                -forward_item.result.value_fraction,
            )
            self.assertNotEqual(
                reversed_item.spec.estimand_sha256,
                forward_item.spec.estimand_sha256,
            )
        with self.assertRaises(AnalysisBindingValidationError):
            replace(
                forward.seed_bindings[0],
                planned_estimand=reversed_estimand,
            )

    def test_changed_country_profile_is_rejected_preflight_and_post_run(self) -> None:
        changed_profiles = (
            replace(
                _PROFILES[0],
                monthly_income_median_cents=(
                    _PROFILES[0].monthly_income_median_cents + 1
                ),
            ),
        )
        changed_lineage = build_profile_input_lineage(changed_profiles)
        self.assertNotEqual(
            changed_lineage.fingerprint_sha256,
            _PROFILE_INPUT_LINEAGE.fingerprint_sha256,
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "profile_input_sha256",
        ):
            validate_analysis_plan_inputs(
                self.plan,
                batch_spec=self.spec,
                run_inputs=self.run_inputs,
                population_adapter=self.adapter,
                profile_input_lineage=changed_lineage,
            )

        changed_batch = _unsafe_clone(
            self.batch,
            country_profiles=changed_profiles,
            profile_input_lineage=changed_lineage,
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "profile_input_sha256|profile input",
        ):
            resolve_run_analysis_binding(self.plan, changed_batch)

    def test_every_prospective_digest_and_stopping_rule_is_preflighted(self) -> None:
        digest_fields = (
            "expected_causal_design_sha256",
            "expected_batch_spec_sha256",
            "expected_model_inputs_sha256",
            "expected_population_input_sha256",
            "expected_profile_input_sha256",
            "expected_metric_contract_sha256",
            "expected_harm_weights_sha256",
            "expected_output_profile_sha256",
        )
        for field_name in digest_fields:
            with self.subTest(field_name=field_name):
                observed = getattr(self.plan, field_name)
                wrong = "0" * 64 if observed != "0" * 64 else "1" * 64
                changed = _plan(
                    self.spec,
                    self.run_inputs,
                    self.adapter,
                    overrides={field_name: wrong},
                )
                with self.assertRaisesRegex(
                    AnalysisBindingValidationError,
                    field_name.removeprefix("expected_"),
                ):
                    validate_analysis_plan_inputs(
                        changed,
                        batch_spec=self.spec,
                        run_inputs=self.run_inputs,
                        population_adapter=self.adapter,
                        profile_input_lineage=_PROFILE_INPUT_LINEAGE,
                    )
        changed_stopping = _plan(
            self.spec,
            self.run_inputs,
            self.adapter,
            seeds=(17,),
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "fixed_seed_stopping_rule",
        ):
            validate_analysis_plan_inputs(
                changed_stopping,
                batch_spec=self.spec,
                run_inputs=self.run_inputs,
                population_adapter=self.adapter,
                profile_input_lineage=_PROFILE_INPUT_LINEAGE,
            )

    def test_explicit_categorical_levels_must_exist_in_adapter_domain(self) -> None:
        cells = tuple(item.projection_cell for item in self.adapter.cells)
        valid_jurisdiction = cells[0].jurisdiction_code
        valid_income = cells[0].monthly_disposable_income_band_id
        valid_household = cells[0].household_type
        fully_explicit = replace(
            _predicate(),
            jurisdiction_codes=(valid_jurisdiction,),
            monthly_disposable_income_band_ids=(valid_income,),
            household_type_ids=(valid_household,),
            gaming_states=(PopulationGamingState.GAMER,),
            payer_history_states=(
                PopulationPayerHistoryState.EVER_PAYER,
            ),
        )
        validate_analysis_plan_inputs(
            _plan(
                self.spec,
                self.run_inputs,
                self.adapter,
                estimand=_planned_estimand(predicate=fully_explicit),
            ),
            batch_spec=self.spec,
            run_inputs=self.run_inputs,
            population_adapter=self.adapter,
            profile_input_lineage=_PROFILE_INPUT_LINEAGE,
        )

        invalid_selections = (
            (
                "jurisdiction_codes",
                tuple(sorted((valid_jurisdiction, "ZZ"))),
            ),
            (
                "monthly_disposable_income_band_ids",
                tuple(sorted((valid_income, "runtime.personal.monthly.typo"))),
            ),
            (
                "household_type_ids",
                tuple(sorted((valid_household, "household.typo"))),
            ),
        )
        for field_name, values in invalid_selections:
            with self.subTest(field_name=field_name):
                predicate = replace(_predicate(), **{field_name: values})
                plan = _plan(
                    self.spec,
                    self.run_inputs,
                    self.adapter,
                    estimand=_planned_estimand(predicate=predicate),
                )
                with self.assertRaisesRegex(
                    AnalysisBindingValidationError,
                    rf"{field_name}.*outside the exact population adapter domain",
                ):
                    validate_analysis_plan_inputs(
                        plan,
                        batch_spec=self.spec,
                        run_inputs=self.run_inputs,
                        population_adapter=self.adapter,
                        profile_input_lineage=_PROFILE_INPUT_LINEAGE,
                    )

    def test_wrong_selected_metric_contract_fails_before_execution(self) -> None:
        planned = _planned_estimand(
            metric_contract_id="player_outcomes.csv:enjoyment"
        )
        plan = _plan(
            self.spec,
            self.run_inputs,
            self.adapter,
            estimand=planned,
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "does not match the selected outcome",
        ):
            validate_analysis_plan_inputs(
                plan,
                batch_spec=self.spec,
                run_inputs=self.run_inputs,
                population_adapter=self.adapter,
                profile_input_lineage=_PROFILE_INPUT_LINEAGE,
            )

    def test_money_estimands_fail_closed_without_executed_conversion(self) -> None:
        currency = PopulationCurrencySemantics(
            currency_code="EUR",
            minor_unit_name="cent",
            price_period_start=date(2026, 1, 1),
            price_period_end=date(2026, 1, 1),
            currency_basis_sha256="9" * 64,
        )
        planned = _planned_estimand(
            outcome_metric=PopulationOutcomeMetric.SPENDING_CENTS,
            metric_contract_id="player_outcomes.csv:spending_cents",
            currency=currency,
        )
        plan = _plan(
            self.spec,
            self.run_inputs,
            self.adapter,
            estimand=planned,
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "executed currency/price-period conversion",
        ):
            validate_analysis_plan_inputs(
                plan,
                batch_spec=self.spec,
                run_inputs=self.run_inputs,
                population_adapter=self.adapter,
                profile_input_lineage=_PROFILE_INPUT_LINEAGE,
            )

    def test_inclusive_period_duration_must_equal_executed_horizon(self) -> None:
        mismatched_secondary = replace(
            self.planned,
            estimand_id="secondary.enjoyment",
            role=AnalysisEstimandRole.SECONDARY,
            outcome_metric=PopulationOutcomeMetric.ENJOYMENT,
            metric_contract_id="player_outcomes.csv:enjoyment",
            period=PopulationPeriodSemantics(
                period_start=date(2040, 7, 10),
                period_end=date(2040, 7, 11),
                description="Two declared days for a one-day execution.",
            ),
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "inclusive period duration.*1-day batch",
        ):
            validate_analysis_plan_inputs(
                _plan(
                    self.spec,
                    self.run_inputs,
                    self.adapter,
                    estimands=(self.planned, mismatched_secondary),
                ),
                batch_spec=self.spec,
                run_inputs=self.run_inputs,
                population_adapter=self.adapter,
                profile_input_lineage=_PROFILE_INPUT_LINEAGE,
            )

        shifted_anchor = replace(
            self.planned,
            period=PopulationPeriodSemantics(
                period_start=date(2040, 7, 10),
                period_end=date(2040, 7, 10),
                description="One declared day at an unexecuted calendar anchor.",
            ),
        )
        validate_analysis_plan_inputs(
            _plan(
                self.spec,
                self.run_inputs,
                self.adapter,
                estimand=shifted_anchor,
            ),
            batch_spec=self.spec,
            run_inputs=self.run_inputs,
            population_adapter=self.adapter,
            profile_input_lineage=_PROFILE_INPUT_LINEAGE,
        )

    def test_zero_day_structural_snapshot_requires_one_declared_day(self) -> None:
        structural_spec = PolicyBatchSpec(
            seeds=self.spec.seeds,
            days=0,
            player_count=self.spec.player_count,
            decision_parameters=self.spec.decision_parameters,
        )
        validate_analysis_plan_inputs(
            _plan(
                structural_spec,
                self.run_inputs,
                self.adapter,
                estimand=self.planned,
            ),
            batch_spec=structural_spec,
            run_inputs=self.run_inputs,
            population_adapter=self.adapter,
            profile_input_lineage=_PROFILE_INPUT_LINEAGE,
        )

        two_day_period = replace(
            self.planned,
            period=PopulationPeriodSemantics(
                period_start=date(2026, 1, 1),
                period_end=date(2026, 1, 2),
                description="Two days cannot label a structural snapshot.",
            ),
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "zero-day structural snapshot.*requires 1 declared day",
        ):
            validate_analysis_plan_inputs(
                _plan(
                    structural_spec,
                    self.run_inputs,
                    self.adapter,
                    estimand=two_day_period,
                ),
                batch_spec=structural_spec,
                run_inputs=self.run_inputs,
                population_adapter=self.adapter,
                profile_input_lineage=_PROFILE_INPUT_LINEAGE,
            )

    def test_empty_selected_population_is_rejected_post_run(self) -> None:
        planned = _planned_estimand(
            predicate=_predicate(
                age_min_inclusive=0,
                age_max_exclusive=1,
            )
        )
        plan = _plan(
            self.spec,
            self.run_inputs,
            self.adapter,
            estimand=planned,
        )
        validate_analysis_plan_inputs(
            plan,
            batch_spec=self.spec,
            run_inputs=self.run_inputs,
            population_adapter=self.adapter,
            profile_input_lineage=_PROFILE_INPUT_LINEAGE,
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "selects no players",
        ):
            resolve_run_analysis_binding(plan, self.batch)

    def test_post_run_batch_mismatch_is_rejected(self) -> None:
        changed_spec = PolicyBatchSpec(
            seeds=self.spec.seeds,
            days=2,
            player_count=12,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        changed_batch = _run_batch(
            changed_spec,
            self.run_inputs,
            self.adapter,
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "batch_spec",
        ):
            resolve_run_analysis_binding(self.plan, changed_batch)

    def test_omitted_result_is_rejected_during_batch_reattestation(self) -> None:
        omitted = _unsafe_clone(
            self.batch,
            records=self.batch.records[:-1],
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "expected .* seed-scenario records",
        ):
            resolve_run_analysis_binding(self.plan, omitted)

    def test_altered_exact_weights_are_rejected_before_analysis(self) -> None:
        lineage = self.batch.population_execution_lineage
        assert lineage is not None
        original = lineage.seed_records[0]
        weights = original.exact_weights
        changed_numerators = (
            weights.weight_numerators[0] + weights.weight_denominators[0],
            *weights.weight_numerators[1:],
        )
        altered_weights = _unsafe_clone(
            weights,
            weight_numerators=changed_numerators,
        )
        self.assertIsInstance(altered_weights, ExactPopulationWeights)
        altered_record = _unsafe_clone(
            original,
            exact_weights=altered_weights,
        )
        self.assertIsInstance(altered_record, PopulationSeedExecutionRecord)
        altered_lineage = _unsafe_clone(
            lineage,
            seed_records=(altered_record, *lineage.seed_records[1:]),
        )
        self.assertIsInstance(altered_lineage, PopulationExecutionLineage)
        altered_batch = _unsafe_clone(
            self.batch,
            population_execution_lineage=altered_lineage,
        )
        with self.assertRaises(AnalysisBindingValidationError):
            resolve_run_analysis_binding(self.plan, altered_batch)

    def test_cross_seed_weights_cannot_be_substituted(self) -> None:
        binding = resolve_run_analysis_binding(self.plan, self.batch)
        first, second = binding.seed_bindings
        self.assertNotEqual(
            first.selected_weights.design_sha256,
            second.selected_weights.design_sha256,
        )
        with self.assertRaisesRegex(
            AnalysisBindingValidationError,
            "resolved estimand spec differs",
        ):
            replace(first, selected_weights=second.selected_weights)


if __name__ == "__main__":
    unittest.main()
