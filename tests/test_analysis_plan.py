from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from microtx_sim.agents.players import ProjectedPopulationCellMetadata
from microtx_sim.causal.analysis_plan import (
    ANALYSIS_PLAN_SCHEMA_VERSION,
    MAX_ANALYSIS_PLAN_BYTES,
    AnalysisEstimandRole,
    AnalysisPlanCampaignError,
    AnalysisPlanRegistrationStatus,
    AnalysisPlanValidationError,
    AnalysisPlanVerificationError,
    CanonicalPopulationInclusionPredicate,
    FixedSeedStoppingRule,
    LoadedProspectiveAnalysisPlan,
    PlannedPopulationEstimand,
    PopulationMinorFilter,
    PopulationOutcomeMetric,
    ProspectiveAnalysisPlan,
    analysis_plan_harm_weights_sha256,
    build_prospective_analysis_plan,
    evaluate_population_inclusion,
    load_prospective_analysis_plan,
    population_outcome_semantics,
    verify_loaded_prospective_analysis_plan,
    verify_prospective_analysis_plan_bindings,
)
from microtx_sim.causal.scenarios import ScenarioId
from microtx_sim.data.population_evidence import (
    PopulationEstimandRole,
    PopulationGamingState,
    PopulationPayerHistoryState,
)
from microtx_sim.metrics.harm import WelfareHarmWeights
from microtx_sim.metrics.population_estimands import (
    PopulationCurrencySemantics,
    PopulationInclusionField,
    PopulationInclusionRule,
    PopulationInclusionTiming,
    PopulationMetricKind,
    PopulationMetricScale,
    PopulationPeriodSemantics,
)


_DIGESTS = {
    "expected_causal_design_sha256": "1" * 64,
    "expected_batch_spec_sha256": "2" * 64,
    "expected_model_inputs_sha256": "3" * 64,
    "expected_population_input_sha256": "4" * 64,
    "expected_profile_input_sha256": "5" * 64,
    "expected_metric_contract_sha256": "6" * 64,
    "expected_harm_weights_sha256": "7" * 64,
    "expected_output_profile_sha256": "8" * 64,
}
_CANONICAL_INCLUSION_FIELDS = tuple(
    sorted(PopulationInclusionField, key=lambda item: item.value)
)


def _rule(
    *,
    source_fields: tuple[PopulationInclusionField, ...] = (
        _CANONICAL_INCLUSION_FIELDS
    ),
    timing: PopulationInclusionTiming = PopulationInclusionTiming.PRETREATMENT,
    evidence_role: PopulationEstimandRole = PopulationEstimandRole.CALIBRATION,
) -> PopulationInclusionRule:
    return PopulationInclusionRule(
        rule_id="eligible.projected.players",
        description="Canonical pre-treatment target-population eligibility.",
        source_fields=source_fields,
        timing=timing,
        evidence_role=evidence_role,
    )


def _predicate(
    *,
    rule: PopulationInclusionRule | None = None,
    jurisdiction_codes: tuple[str, ...] = ("BE",),
    age_min_inclusive: int = 12,
    age_max_exclusive: int = 65,
    minor_filter: PopulationMinorFilter = PopulationMinorFilter.ANY,
    income_bands: tuple[str, ...] = ("low",),
    household_types: tuple[str, ...] = ("family",),
    gaming_states: tuple[PopulationGamingState, ...] = (
        PopulationGamingState.GAMER,
    ),
    payer_states: tuple[PopulationPayerHistoryState, ...] = (
        PopulationPayerHistoryState.NEVER_PAYER,
    ),
) -> CanonicalPopulationInclusionPredicate:
    return CanonicalPopulationInclusionPredicate(
        rule=_rule() if rule is None else rule,
        jurisdiction_codes=jurisdiction_codes,
        age_min_inclusive=age_min_inclusive,
        age_max_exclusive=age_max_exclusive,
        minor_filter=minor_filter,
        monthly_disposable_income_band_ids=income_bands,
        household_type_ids=household_types,
        gaming_states=gaming_states,
        payer_history_states=payer_states,
    )


def _currency() -> PopulationCurrencySemantics:
    return PopulationCurrencySemantics(
        currency_code="EUR",
        minor_unit_name="cent",
        price_period_start=date(2025, 1, 1),
        price_period_end=date(2025, 12, 31),
        currency_basis_sha256="8" * 64,
    )


def _period() -> PopulationPeriodSemantics:
    return PopulationPeriodSemantics(
        period_start=date(2025, 1, 1),
        period_end=date(2025, 12, 31),
        description="One calendar-year per-player policy outcome.",
    )


def _estimand(
    *,
    estimand_id: str = "primary.harmful_spending",
    role: AnalysisEstimandRole = AnalysisEstimandRole.PRIMARY,
    reference: ScenarioId = ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
    comparison: ScenarioId = ScenarioId.BASELINE_F2P,
    metric: PopulationOutcomeMetric = PopulationOutcomeMetric.HARMFUL_SPENDING_CENTS,
    currency: PopulationCurrencySemantics | None = None,
    predicate: CanonicalPopulationInclusionPredicate | None = None,
) -> PlannedPopulationEstimand:
    semantics = population_outcome_semantics(metric)
    if currency is None and semantics.metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS:
        currency = _currency()
    return PlannedPopulationEstimand(
        estimand_id=estimand_id,
        role=role,
        reference_scenario_id=reference,
        comparison_scenario_id=comparison,
        outcome_metric=metric,
        metric_contract_id=f"population.player.{metric.value}.v1",
        inclusion_predicate=_predicate() if predicate is None else predicate,
        period=_period(),
        currency=currency,
    )


def _plan(
    *,
    estimands: tuple[PlannedPopulationEstimand, ...] | None = None,
    seeds: tuple[int, ...] = (11, 22),
    digests: dict[str, str] | None = None,
) -> ProspectiveAnalysisPlan:
    values = dict(_DIGESTS if digests is None else digests)
    return build_prospective_analysis_plan(
        plan_id="prospective.population.policy.v1",
        stopping_rule=FixedSeedStoppingRule(seeds=seeds),
        estimands=(_estimand(),) if estimands is None else estimands,
        **values,
    )


def _canonical_bytes(plan: ProspectiveAnalysisPlan) -> bytes:
    return json.dumps(
        plan.snapshot(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _cell(
    *,
    cell_id: str,
    jurisdiction_code: str,
    jurisdiction_index: int,
    age_min: int,
    age_max: int,
    income_band: str,
    household_type: str,
    gamer: bool,
    ever_payer: bool,
) -> ProjectedPopulationCellMetadata:
    return ProjectedPopulationCellMetadata(
        cell_id=cell_id,
        jurisdiction_code=jurisdiction_code,
        jurisdiction_index=jurisdiction_index,
        age_min_inclusive=age_min,
        age_max_exclusive=age_max,
        monthly_disposable_income_band_id=income_band,
        monthly_disposable_income_min_cents=0,
        monthly_disposable_income_max_cents_exclusive=100_000,
        household_type=household_type,
        modeled_players_per_household=2,
        baseline_gamer=gamer,
        baseline_ever_payer=ever_payer,
        global_mass=(1, 2),
        analysis_weight=(1, 1),
    )


class AnalysisPlanContractTests(unittest.TestCase):
    def test_schema_v1_is_immutable_content_addressed_and_fail_closed(self) -> None:
        plan = _plan()
        self.assertEqual(plan.schema_version, ANALYSIS_PLAN_SCHEMA_VERSION)
        self.assertEqual(plan.registration_status, AnalysisPlanRegistrationStatus.UNREGISTERED)
        self.assertFalse(plan.preregistered)
        self.assertFalse(plan.campaign_ready)
        self.assertEqual(
            plan.campaign_blockers,
            (
                "analysis_plan.external_registration=unregistered",
                "analysis_plan.schema_v1=campaign_ineligible",
                "analysis_plan.execution_calendar_anchor=unbound",
                "analysis_plan.cross_seed_aggregation_uncertainty=unresolved",
                "analysis_plan.model_implementation_environment_identity=unbound",
            ),
        )
        self.assertEqual(len(plan.plan_sha256), 64)
        self.assertEqual(plan.primary_estimand.role, AnalysisEstimandRole.PRIMARY)
        with self.assertRaises(FrozenInstanceError):
            plan.campaign_ready = True  # type: ignore[misc]
        with self.assertRaises(AnalysisPlanCampaignError):
            plan.validate_for_campaign()

    def test_exactly_one_primary_estimand_is_required(self) -> None:
        secondary = _estimand(
            estimand_id="secondary.enjoyment",
            role=AnalysisEstimandRole.SECONDARY,
            metric=PopulationOutcomeMetric.ENJOYMENT,
            currency=None,
        )
        with self.assertRaisesRegex(AnalysisPlanValidationError, "exactly one PRIMARY"):
            _plan(estimands=(secondary,))
        another_primary = replace(
            secondary,
            estimand_id="primary.enjoyment",
            role=AnalysisEstimandRole.PRIMARY,
        )
        with self.assertRaisesRegex(AnalysisPlanValidationError, "exactly one PRIMARY"):
            _plan(estimands=(_estimand(), another_primary))

    def test_duplicate_semantic_estimand_cannot_be_relabelled_secondary(self) -> None:
        primary = _estimand()
        duplicate = replace(
            primary,
            estimand_id="secondary.duplicate.label",
            role=AnalysisEstimandRole.SECONDARY,
        )
        self.assertEqual(
            primary.specification_sha256,
            duplicate.specification_sha256,
        )
        with self.assertRaisesRegex(
            AnalysisPlanValidationError,
            "unique semantic specifications",
        ):
            _plan(estimands=(primary, duplicate))

    def test_contrast_is_directed_nonidentity_and_not_canonically_sorted(self) -> None:
        estimand = _estimand()
        swapped = replace(
            estimand,
            reference_scenario_id=estimand.comparison_scenario_id,
            comparison_scenario_id=estimand.reference_scenario_id,
        )
        self.assertNotEqual(swapped.estimand_sha256, estimand.estimand_sha256)
        self.assertEqual(swapped.contrast_direction, "COMPARISON_MINUS_REFERENCE")
        with self.assertRaisesRegex(AnalysisPlanValidationError, "must differ"):
            replace(
                estimand,
                comparison_scenario_id=estimand.reference_scenario_id,
            )

    def test_estimand_order_is_canonical_and_deterministic(self) -> None:
        primary = _estimand(estimand_id="z.primary")
        secondary = _estimand(
            estimand_id="a.secondary",
            role=AnalysisEstimandRole.SECONDARY,
            metric=PopulationOutcomeMetric.ENJOYMENT,
            currency=None,
        )
        first = _plan(estimands=(primary, secondary))
        second = _plan(estimands=(secondary, primary))
        self.assertEqual(first, second)
        self.assertEqual(first.plan_sha256, second.plan_sha256)
        self.assertEqual(
            tuple(item.estimand_id for item in first.estimands),
            ("a.secondary", "z.primary"),
        )
        with self.assertRaisesRegex(AnalysisPlanValidationError, "ascending"):
            ProspectiveAnalysisPlan(
                **{
                    **{
                        field: getattr(first, field)
                        for field in _DIGESTS
                    },
                    "schema_version": first.schema_version,
                    "plan_id": first.plan_id,
                    "stopping_rule": first.stopping_rule,
                    "estimands": tuple(reversed(first.estimands)),
                    "plan_sha256": first.plan_sha256,
                }
            )

    def test_metric_whitelist_exposes_writer_ready_semantics(self) -> None:
        self.assertEqual(len(PopulationOutcomeMetric), 22)
        for metric in PopulationOutcomeMetric:
            semantics = population_outcome_semantics(metric)
            self.assertEqual(semantics.metric, metric)
            self.assertEqual(semantics.metric_name, metric.value)
            self.assertTrue(semantics.result_path.startswith("PolicyScenarioResult."))
            self.assertIn(semantics.metric_kind, PopulationMetricKind)
            self.assertIn(semantics.metric_scale, PopulationMetricScale)
        money = population_outcome_semantics(PopulationOutcomeMetric.SPENDING_CENTS)
        self.assertEqual(money.metric_kind, PopulationMetricKind.MONEY_MINOR_UNITS)
        self.assertEqual(money.storage_dtype, "int64")
        component = population_outcome_semantics(
            PopulationOutcomeMetric.SLEEP_BURDEN_SCORE
        )
        self.assertEqual(component.component_index, 2)

    def test_money_metrics_require_currency_and_nonmoney_forbid_it(self) -> None:
        with self.assertRaisesRegex(AnalysisPlanValidationError, "require currency"):
            PlannedPopulationEstimand(
                estimand_id="money.no.currency",
                role=AnalysisEstimandRole.PRIMARY,
                reference_scenario_id=ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
                comparison_scenario_id=ScenarioId.BASELINE_F2P,
                outcome_metric=PopulationOutcomeMetric.SPENDING_CENTS,
                metric_contract_id="population.player.spending_cents.v1",
                inclusion_predicate=_predicate(),
                period=_period(),
                currency=None,
            )
        with self.assertRaisesRegex(AnalysisPlanValidationError, "cannot declare currency"):
            _estimand(metric=PopulationOutcomeMetric.ENJOYMENT, currency=_currency())

    def test_fixed_seed_rule_rejects_outcome_adaptive_or_polymorphic_values(self) -> None:
        for invalid in ((), (2, 1), (1, 1), (True,), (-1,), (2**64,)):
            with self.subTest(invalid=invalid), self.assertRaises((TypeError, ValueError)):
                FixedSeedStoppingRule(seeds=invalid)  # type: ignore[arg-type]
        snapshot = FixedSeedStoppingRule(seeds=(1, 3)).snapshot()
        self.assertFalse(snapshot["early_stopping_allowed"])
        self.assertFalse(snapshot["treatment_result_interim_looks_allowed"])

    def test_every_runtime_digest_and_seed_set_is_verified(self) -> None:
        plan = _plan()
        kwargs = {
            "causal_design_sha256": plan.expected_causal_design_sha256,
            "batch_spec_sha256": plan.expected_batch_spec_sha256,
            "model_inputs_sha256": plan.expected_model_inputs_sha256,
            "population_input_sha256": plan.expected_population_input_sha256,
            "profile_input_sha256": plan.expected_profile_input_sha256,
            "metric_contract_sha256": plan.expected_metric_contract_sha256,
            "harm_weights_sha256": plan.expected_harm_weights_sha256,
            "output_profile_sha256": plan.expected_output_profile_sha256,
            "seeds": plan.stopping_rule.seeds,
        }
        self.assertIs(
            verify_prospective_analysis_plan_bindings(plan, **kwargs),
            plan,
        )
        for key in tuple(kwargs)[:-1]:
            changed = dict(kwargs)
            changed[key] = "f" * 64
            with self.subTest(key=key), self.assertRaises(
                AnalysisPlanVerificationError
            ):
                verify_prospective_analysis_plan_bindings(plan, **changed)
        with self.assertRaises(AnalysisPlanVerificationError):
            verify_prospective_analysis_plan_bindings(
                plan,
                **{**kwargs, "seeds": (11, 23)},
            )
        with self.assertRaises((TypeError, AnalysisPlanValidationError)):
            verify_prospective_analysis_plan_bindings(
                plan,
                **{**kwargs, "model_inputs_sha256": 3},  # type: ignore[arg-type]
            )

    def test_harm_weight_digest_is_deterministic_and_value_sensitive(self) -> None:
        baseline = WelfareHarmWeights()
        self.assertEqual(
            analysis_plan_harm_weights_sha256(baseline),
            analysis_plan_harm_weights_sha256(WelfareHarmWeights()),
        )
        self.assertNotEqual(
            analysis_plan_harm_weights_sha256(baseline),
            analysis_plan_harm_weights_sha256(
                WelfareHarmWeights(monetary=2.0)
            ),
        )


class PopulationInclusionPredicateTests(unittest.TestCase):
    def test_predicate_executes_all_joint_cell_and_player_filters(self) -> None:
        cells = (
            _cell(
                cell_id="be.child.low.family.gamer.never",
                jurisdiction_code="BE",
                jurisdiction_index=0,
                age_min=12,
                age_max=18,
                income_band="low",
                household_type="family",
                gamer=True,
                ever_payer=False,
            ),
            _cell(
                cell_id="DE.adult.high.single.non_gamer.ever",
                jurisdiction_code="DE",
                jurisdiction_index=1,
                age_min=16,
                age_max=65,
                income_band="high",
                household_type="single",
                gamer=False,
                ever_payer=True,
            ),
        )
        predicate = _predicate(
            age_min_inclusive=12,
            age_max_exclusive=18,
            minor_filter=PopulationMinorFilter.MINOR_ONLY,
        )
        observed = evaluate_population_inclusion(
            predicate,
            jurisdiction_codes=("BE", "DE"),
            jurisdiction=np.asarray([0, 1], dtype=np.int16),
            age_years=np.asarray([16, 17], dtype=np.int16),
            # Minor status is jurisdiction-derived, not hard-coded as age < 18.
            is_minor=np.asarray([True, False], dtype=np.bool_),
            projected_cells=cells,
            cell_indices=(0, 1),
        )
        np.testing.assert_array_equal(observed, np.asarray([True, False]))
        self.assertFalse(observed.flags.writeable)
        with self.assertRaises(ValueError):
            observed[0] = False

    def test_runtime_assignment_must_match_projected_cell(self) -> None:
        cells = (
            _cell(
                cell_id="be.child.low.family.gamer.never",
                jurisdiction_code="BE",
                jurisdiction_index=0,
                age_min=12,
                age_max=18,
                income_band="low",
                household_type="family",
                gamer=True,
                ever_payer=False,
            ),
        )
        kwargs = {
            "jurisdiction_codes": ("BE",),
            "jurisdiction": np.asarray([0], dtype=np.int16),
            "age_years": np.asarray([16], dtype=np.int16),
            "is_minor": np.asarray([True], dtype=np.bool_),
            "projected_cells": cells,
            "cell_indices": (0,),
        }
        self.assertTrue(_predicate().evaluate(**kwargs)[0])
        with self.assertRaisesRegex(AnalysisPlanVerificationError, "age differs"):
            _predicate().evaluate(
                **{**kwargs, "age_years": np.asarray([20], dtype=np.int16)}
            )
        with self.assertRaises(TypeError):
            _predicate().evaluate(
                **{**kwargs, "age_years": np.asarray([16], dtype=np.int64)}
            )

    def test_all_canonical_pretreatment_fields_are_mandatory(self) -> None:
        incomplete = _rule(source_fields=_CANONICAL_INCLUSION_FIELDS[:-1])
        with self.assertRaisesRegex(AnalysisPlanValidationError, "every canonical"):
            _predicate(rule=incomplete)

    def test_posttreatment_and_validation_derived_rules_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "before treatment"):
            _rule(timing=PopulationInclusionTiming.POSTTREATMENT)
        with self.assertRaisesRegex(ValueError, "validation evidence"):
            _rule(evidence_role=PopulationEstimandRole.VALIDATION)

    def test_filter_order_duplicates_and_polymorphic_enums_are_rejected(self) -> None:
        with self.assertRaisesRegex(AnalysisPlanValidationError, "ascending"):
            _predicate(jurisdiction_codes=("DE", "BE"))
        with self.assertRaisesRegex(AnalysisPlanValidationError, "ascending"):
            _predicate(income_bands=("low", "low"))
        with self.assertRaises(TypeError):
            _predicate(gaming_states=("GAMER",))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            _predicate(minor_filter="ANY")  # type: ignore[arg-type]


class AnalysisPlanLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.directory = Path(self._temporary.name)
        self.plan = _plan()

    def _write(self, name: str, content: bytes) -> Path:
        path = self.directory / name
        path.write_bytes(content)
        return path

    def _write_snapshot(
        self,
        snapshot: dict[str, object],
        *,
        name: str = "analysis-plan.json",
        pretty: bool = False,
    ) -> Path:
        content = json.dumps(
            snapshot,
            allow_nan=False,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return self._write(name, content)

    def test_secure_load_records_exact_file_and_semantic_identities(self) -> None:
        content = _canonical_bytes(self.plan)
        path = self._write("analysis-plan.json", content)
        loaded = load_prospective_analysis_plan(path)
        self.assertEqual(loaded.plan_path, path.absolute())
        self.assertEqual(loaded.byte_length, len(content))
        self.assertEqual(loaded.file_sha256, sha256(content).hexdigest())
        self.assertEqual(loaded.semantic_sha256, self.plan.plan_sha256)
        self.assertEqual(loaded.plan, self.plan)
        self.assertEqual(loaded.manifest_payload(), loaded.snapshot())
        self.assertEqual(verify_loaded_prospective_analysis_plan(loaded), loaded)

    def test_formatting_changes_file_digest_but_not_semantic_digest(self) -> None:
        compact = load_prospective_analysis_plan(
            self._write_snapshot(self.plan.snapshot(), name="compact.json")
        )
        pretty = load_prospective_analysis_plan(
            self._write_snapshot(
                self.plan.snapshot(),
                name="pretty.json",
                pretty=True,
            )
        )
        self.assertNotEqual(compact.file_sha256, pretty.file_sha256)
        self.assertEqual(compact.semantic_sha256, pretty.semantic_sha256)
        self.assertEqual(compact.plan, pretty.plan)

    def test_exact_keys_and_duplicate_keys_are_rejected(self) -> None:
        unknown = self.plan.snapshot()
        unknown["external_registration_id"] = "invented"
        with self.assertRaisesRegex(AnalysisPlanValidationError, "unknown"):
            load_prospective_analysis_plan(self._write_snapshot(unknown))
        duplicate = b'{"schema_version":"1.0","schema_version":"1.0"}'
        with self.assertRaisesRegex(AnalysisPlanValidationError, "repeats key"):
            load_prospective_analysis_plan(self._write("duplicate.json", duplicate))
        with self.assertRaisesRegex(AnalysisPlanValidationError, "root must be an object"):
            load_prospective_analysis_plan(self._write("list.json", b"[]"))

    def test_registration_and_readiness_cannot_be_claimed_in_json(self) -> None:
        mutations: tuple[tuple[str, object], ...] = (
            ("registration_status", "REGISTERED"),
            ("preregistered", True),
            ("preregistered", 0),
            ("campaign_ready", True),
            ("campaign_ready", 0),
            ("campaign_blockers", []),
        )
        for index, (key, value) in enumerate(mutations):
            snapshot = self.plan.snapshot()
            snapshot[key] = value
            with self.subTest(key=key), self.assertRaises(AnalysisPlanValidationError):
                load_prospective_analysis_plan(
                    self._write_snapshot(snapshot, name=f"forged-{index}.json")
                )

    def test_polymorphic_json_values_and_noncanonical_filters_are_rejected(self) -> None:
        seeds = self.plan.snapshot()
        seeds["stopping_rule"]["seeds"][0] = True  # type: ignore[index]
        with self.assertRaisesRegex(AnalysisPlanValidationError, "JSON integer"):
            load_prospective_analysis_plan(
                self._write_snapshot(seeds, name="boolean-seed.json")
            )
        stopping_flag = self.plan.snapshot()
        stopping_flag["stopping_rule"]["early_stopping_allowed"] = 0  # type: ignore[index]
        with self.assertRaisesRegex(AnalysisPlanValidationError, "JSON boolean"):
            load_prospective_analysis_plan(
                self._write_snapshot(stopping_flag, name="integer-flag.json")
            )
        component_plan = _plan(
            estimands=(
                _estimand(
                    metric=PopulationOutcomeMetric.MONETARY_HARM_SCORE,
                    currency=None,
                ),
            )
        )
        component = component_plan.snapshot()
        component["estimands"][0]["outcome"]["component_index"] = 0.0  # type: ignore[index]
        with self.assertRaisesRegex(AnalysisPlanValidationError, "JSON integer"):
            load_prospective_analysis_plan(
                self._write_snapshot(component, name="float-component.json")
            )
        filters = self.plan.snapshot()
        filters["estimands"][0]["inclusion_predicate"][  # type: ignore[index]
            "jurisdiction_codes"
        ] = ["DE", "BE"]
        with self.assertRaisesRegex(AnalysisPlanValidationError, "ascending"):
            load_prospective_analysis_plan(
                self._write_snapshot(filters, name="unordered-filter.json")
            )

    def test_posttreatment_and_validation_inclusion_json_is_rejected(self) -> None:
        for key, value in (
            ("timing", PopulationInclusionTiming.POSTTREATMENT.value),
            ("evidence_role", PopulationEstimandRole.VALIDATION.value),
        ):
            snapshot = self.plan.snapshot()
            rule = snapshot["estimands"][0]["inclusion_predicate"]["rule"]  # type: ignore[index]
            rule[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_prospective_analysis_plan(
                    self._write_snapshot(snapshot, name=f"forged-{key}.json")
                )

    def test_digest_or_seed_mutation_breaks_semantic_attestation(self) -> None:
        for index, mutate in enumerate(("digest", "seed")):
            snapshot = self.plan.snapshot()
            if mutate == "digest":
                snapshot["expected_model_inputs_sha256"] = "f" * 64
            else:
                snapshot["stopping_rule"]["seeds"][1] = 23  # type: ignore[index]
                snapshot["stopping_rule"]["seed_decimal_strings"][1] = "23"  # type: ignore[index]
            with self.subTest(mutate=mutate), self.assertRaises(
                AnalysisPlanValidationError
            ):
                load_prospective_analysis_plan(
                    self._write_snapshot(snapshot, name=f"mutated-{index}.json")
                )

    def test_loaded_file_mutation_is_detected_even_if_semantics_are_unchanged(self) -> None:
        path = self._write("analysis-plan.json", _canonical_bytes(self.plan))
        loaded = load_prospective_analysis_plan(path)
        path.write_bytes(_canonical_bytes(self.plan) + b"\n")
        with self.assertRaisesRegex(
            AnalysisPlanVerificationError,
            "changed|differ",
        ):
            verify_loaded_prospective_analysis_plan(loaded)

    def test_symlink_alias_is_rejected(self) -> None:
        target = self._write("analysis-plan.json", _canonical_bytes(self.plan))
        alias = self.directory / "analysis-plan-alias.json"
        try:
            alias.symlink_to(target)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        with self.assertRaisesRegex(AnalysisPlanVerificationError, "non-symlink"):
            load_prospective_analysis_plan(alias)

    def test_empty_oversized_invalid_utf8_and_nonfiles_are_rejected(self) -> None:
        with self.assertRaises(AnalysisPlanValidationError):
            load_prospective_analysis_plan(self._write("empty.json", b""))
        with self.assertRaises(AnalysisPlanValidationError):
            load_prospective_analysis_plan(
                self._write("oversized.json", b" " * (MAX_ANALYSIS_PLAN_BYTES + 1))
            )
        with self.assertRaises(AnalysisPlanValidationError):
            load_prospective_analysis_plan(self._write("invalid.json", b"\xff"))
        with self.assertRaises(AnalysisPlanVerificationError):
            load_prospective_analysis_plan(self.directory)

    def test_loaded_wrapper_rejects_polymorphic_or_forged_identity(self) -> None:
        path = self._write("analysis-plan.json", _canonical_bytes(self.plan))
        loaded = load_prospective_analysis_plan(str(path))
        with self.assertRaises(TypeError):
            LoadedProspectiveAnalysisPlan(
                plan_path=str(loaded.plan_path),  # type: ignore[arg-type]
                byte_length=loaded.byte_length,
                file_sha256=loaded.file_sha256,
                semantic_sha256=loaded.semantic_sha256,
                plan=loaded.plan,
            )
        with self.assertRaises(AnalysisPlanValidationError):
            replace(loaded, file_sha256="F" * 64)
        with self.assertRaises(AnalysisPlanValidationError):
            replace(loaded, semantic_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
