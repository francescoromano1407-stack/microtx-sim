from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date
from fractions import Fraction
import unittest

import numpy as np

from microtx_sim.data.population_evidence import PopulationEstimandRole
from microtx_sim.metrics.population_estimands import (
    EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
    POPULATION_ESTIMAND_SCHEMA_VERSION,
    TARGET_POPULATION_OUTPUT_PROFILE,
    ExactPopulationWeights,
    PopulationAnalysisUnit,
    PopulationContrast,
    PopulationCurrencySemantics,
    PopulationEstimandAlgorithm,
    PopulationEstimandSpec,
    PopulationEstimandValidationError,
    PopulationInclusionField,
    PopulationInclusionRule,
    PopulationInclusionTiming,
    PopulationMetricKind,
    PopulationMetricScale,
    PopulationNormalization,
    PopulationPeriodSemantics,
    paired_weighted_mean_difference,
    weighted_mean,
    weighted_quantile,
)


def _weights(
    *,
    player_ids: tuple[int, ...] = (10, 20),
    numerators: tuple[int, ...] = (9, 1),
    denominators: tuple[int, ...] = (1, 1),
) -> ExactPopulationWeights:
    return ExactPopulationWeights(
        schema_version=EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
        player_ids=player_ids,
        weight_numerators=numerators,
        weight_denominators=denominators,
    )


def _inclusion_rule() -> PopulationInclusionRule:
    return PopulationInclusionRule(
        rule_id="all.target.eligible",
        description="All target-eligible players, selected before treatment.",
        source_fields=(
            PopulationInclusionField.AGE_YEARS,
            PopulationInclusionField.JURISDICTION,
        ),
        timing=PopulationInclusionTiming.PRETREATMENT,
        evidence_role=PopulationEstimandRole.CALIBRATION,
    )


def _currency() -> PopulationCurrencySemantics:
    return PopulationCurrencySemantics(
        currency_code="EUR",
        minor_unit_name="cent",
        price_period_start=date(2025, 1, 1),
        price_period_end=date(2025, 12, 31),
        currency_basis_sha256="d" * 64,
    )


def _spec(
    weights: ExactPopulationWeights,
    *,
    algorithm: PopulationEstimandAlgorithm = (
        PopulationEstimandAlgorithm.WEIGHTED_MEAN_V1
    ),
    normalization: PopulationNormalization | None = None,
    metric_kind: PopulationMetricKind = PopulationMetricKind.SCORE,
    metric_scale: PopulationMetricScale = PopulationMetricScale.NONADDITIVE,
    target_population_count: int | None = None,
    quantile: tuple[int, int] | None = None,
    currency: PopulationCurrencySemantics | None = None,
) -> PopulationEstimandSpec:
    if normalization is None:
        normalization = (
            PopulationNormalization.WEIGHTED_INVERSE_CDF
            if algorithm is PopulationEstimandAlgorithm.WEIGHTED_QUANTILE_V1
            else PopulationNormalization.DIVIDE_BY_WEIGHT_SUM
        )
    contrast = (
        PopulationContrast.TREATED_MINUS_CONTROL
        if algorithm
        is PopulationEstimandAlgorithm.PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1
        else PopulationContrast.NONE
    )
    return PopulationEstimandSpec(
        schema_version=POPULATION_ESTIMAND_SCHEMA_VERSION,
        estimand_id="population.estimand.test",
        target_population_id="target.players.test",
        target_evidence_sha256="a" * 64,
        design_weights_sha256=weights.design_sha256,
        runtime_projection_sha256="b" * 64,
        balance_report_sha256="c" * 64,
        metric_contract_sha256="e" * 64,
        output_profile_id=TARGET_POPULATION_OUTPUT_PROFILE,
        output_profile_schema_sha256="f" * 64,
        analysis_unit=PopulationAnalysisUnit.PLAYER_PERSON,
        inclusion_rule=_inclusion_rule(),
        metric_name=(
            "spending_cents"
            if metric_kind is PopulationMetricKind.MONEY_MINOR_UNITS
            else "composite_harm"
        ),
        metric_kind=metric_kind,
        metric_scale=metric_scale,
        contrast=contrast,
        algorithm=algorithm,
        normalization=normalization,
        period=PopulationPeriodSemantics(
            period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 31),
            description="One declared simulation horizon.",
        ),
        currency=currency,
        target_population_count=target_population_count,
        quantile_probability_numerator=(quantile[0] if quantile else None),
        quantile_probability_denominator=(quantile[1] if quantile else None),
    )


class PopulationEstimandTests(unittest.TestCase):
    def test_exact_weighted_mean_is_content_addressed_and_immutable(self) -> None:
        weights = _weights()
        spec = _spec(weights)

        result = weighted_mean(spec, weights, (0, 1))

        self.assertEqual(result.value_fraction, Fraction(1, 10))
        self.assertEqual(result.weight_sum, Fraction(10, 1))
        self.assertEqual(result.numerator, 1)
        self.assertEqual(result.denominator, 10)
        self.assertEqual(result.design_weights_sha256, weights.design_sha256)
        self.assertEqual(result.estimand_sha256, spec.estimand_sha256)
        self.assertEqual(len(result.result_sha256), 64)
        self.assertEqual(result.snapshot()["numerator_decimal"], "1")
        with self.assertRaises(FrozenInstanceError):
            result.numerator = 2  # type: ignore[misc]
        with self.assertRaises(TypeError):
            weights.player_ids[0] = 99  # type: ignore[index]

    def test_paired_difference_uses_one_ordered_design(self) -> None:
        weights = _weights()
        spec = _spec(
            weights,
            algorithm=(
                PopulationEstimandAlgorithm.PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1
            ),
        )

        result = paired_weighted_mean_difference(
            spec,
            weights,
            (0, 10),
            weights,
            (1, 0),
        )

        self.assertEqual(result.value_fraction, Fraction(1, 10))
        self.assertEqual(result.contrast, PopulationContrast.TREATED_MINUS_CONTROL)

        reordered = _weights(player_ids=(20, 10))
        with self.assertRaisesRegex(ValueError, "identical ordered player IDs"):
            paired_weighted_mean_difference(
                spec,
                weights,
                (0, 10),
                reordered,
                (0, 1),
            )

        scenario_dependent = _weights(numerators=(8, 2))
        with self.assertRaisesRegex(ValueError, "scenario-dependent design weights"):
            paired_weighted_mean_difference(
                spec,
                weights,
                (0, 10),
                scenario_dependent,
                (1, 0),
            )

    def test_weighted_quantile_uses_lower_inverse_cdf_without_interpolation(
        self,
    ) -> None:
        weights = _weights(
            player_ids=(20, 10, 30),
            numerators=(1, 1, 2),
            denominators=(1, 1, 1),
        )
        median = _spec(
            weights,
            algorithm=PopulationEstimandAlgorithm.WEIGHTED_QUANTILE_V1,
            quantile=(1, 2),
        )

        tied = weighted_quantile(median, weights, (10, 10, 20))
        boundary = weighted_quantile(median, weights, (0, 10, 20))

        self.assertEqual(tied.value_fraction, Fraction(10, 1))
        self.assertEqual(boundary.value_fraction, Fraction(10, 1))

        two_equal_weights = _weights(
            player_ids=(1, 2),
            numerators=(1, 1),
            denominators=(1, 1),
        )
        lower_median = _spec(
            two_equal_weights,
            algorithm=PopulationEstimandAlgorithm.WEIGHTED_QUANTILE_V1,
            quantile=(1, 2),
        )
        self.assertEqual(
            weighted_quantile(lower_median, two_equal_weights, (0, 10)).value,
            0.0,
        )

    def test_target_total_requires_count_additivity_and_currency_contract(self) -> None:
        weights = _weights(numerators=(1, 1))
        total_spec = _spec(
            weights,
            normalization=PopulationNormalization.TARGET_POPULATION_TOTAL,
            metric_kind=PopulationMetricKind.MONEY_MINOR_UNITS,
            metric_scale=PopulationMetricScale.ADDITIVE_PER_ANALYSIS_UNIT,
            target_population_count=1_000,
            currency=_currency(),
        )

        total = weighted_mean(total_spec, weights, (101, 103))

        self.assertEqual(total.value_fraction, Fraction(102_000, 1))
        self.assertEqual(total.target_population_count, 1_000)

        with self.assertRaisesRegex(ValueError, "additive"):
            _spec(
                weights,
                normalization=PopulationNormalization.TARGET_POPULATION_TOTAL,
                target_population_count=1_000,
            )
        with self.assertRaises((TypeError, PopulationEstimandValidationError)):
            _spec(
                weights,
                normalization=PopulationNormalization.TARGET_POPULATION_TOTAL,
                metric_scale=PopulationMetricScale.ADDITIVE_PER_ANALYSIS_UNIT,
            )
        with self.assertRaisesRegex(ValueError, "only valid for target totals"):
            _spec(weights, target_population_count=1_000)
        with self.assertRaisesRegex(ValueError, "currency semantics"):
            _spec(weights, metric_kind=PopulationMetricKind.MONEY_MINOR_UNITS)
        with self.assertRaisesRegex(ValueError, "only valid for money"):
            _spec(weights, currency=_currency())
        with self.assertRaisesRegex(
            TypeError,
            "exact rational target-currency minor units",
        ):
            weighted_mean(total_spec, weights, (101.0, 103.0))

    def test_inclusions_cannot_be_posttreatment_or_validation_derived(self) -> None:
        common = {
            "rule_id": "eligible.players",
            "description": "A deliberately invalid exclusion rule.",
            "source_fields": (PopulationInclusionField.AGE_YEARS,),
        }
        with self.assertRaisesRegex(ValueError, "before treatment"):
            PopulationInclusionRule(
                **common,
                timing=PopulationInclusionTiming.POSTTREATMENT,
                evidence_role=PopulationEstimandRole.CALIBRATION,
            )
        with self.assertRaisesRegex(ValueError, "validation evidence"):
            PopulationInclusionRule(
                **common,
                timing=PopulationInclusionTiming.PRETREATMENT,
                evidence_role=PopulationEstimandRole.VALIDATION,
            )
        with self.assertRaisesRegex(ValueError, "canonical order"):
            PopulationInclusionRule(
                **{
                    **common,
                    "source_fields": (
                        PopulationInclusionField.JURISDICTION,
                        PopulationInclusionField.AGE_YEARS,
                    ),
                },
                timing=PopulationInclusionTiming.PRETREATMENT,
                evidence_role=PopulationEstimandRole.CALIBRATION,
            )

    def test_zero_nonfinite_and_misaligned_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            _weights(numerators=(0, 1))
        with self.assertRaisesRegex(ValueError, "at least one"):
            _weights(player_ids=(), numerators=(), denominators=())
        with self.assertRaisesRegex(ValueError, "reduced canonical"):
            _weights(numerators=(2, 1), denominators=(4, 1))

        weights = _weights()
        spec = _spec(weights)
        with self.assertRaisesRegex(ValueError, "align one-for-one"):
            weighted_mean(spec, weights, (0,))
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    weighted_mean(spec, weights, (value, 0.0))
        with self.assertRaisesRegex(TypeError, "not boolean"):
            weighted_mean(spec, weights, (True, 0.0))

    def test_exact_arithmetic_preserves_values_beyond_float_integer_precision(
        self,
    ) -> None:
        denominator = 2**53 + 3
        weights = _weights(
            numerators=(denominator - 2, 2),
            denominators=(denominator, denominator),
        )
        spec = _spec(weights)

        result = weighted_mean(spec, weights, (0, denominator))

        self.assertEqual(weights.weight_sum, Fraction(1, 1))
        self.assertEqual(result.value_fraction, Fraction(2, 1))

        paired = _spec(
            _weights(player_ids=(1,), numerators=(1,), denominators=(1,)),
            algorithm=(
                PopulationEstimandAlgorithm.PAIRED_WEIGHTED_MEAN_DIFFERENCE_V1
            ),
        )
        single = _weights(player_ids=(1,), numerators=(1,), denominators=(1,))
        with self.assertRaisesRegex(OverflowError, "finite float64"):
            paired_weighted_mean_difference(
                paired,
                single,
                (1e308,),
                single,
                (-1e308,),
            )

    def test_digest_tampering_and_wrong_design_are_rejected(self) -> None:
        weights = _weights()
        spec = _spec(weights)
        result = weighted_mean(spec, weights, (0, 1))

        with self.assertRaisesRegex(ValueError, "result_sha256"):
            replace(result, numerator=result.numerator + result.denominator)

        wrong_digest = replace(spec, design_weights_sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "does not re-attest"):
            weighted_mean(wrong_digest, weights, (0, 1))

        with self.assertRaisesRegex(ValueError, "schema version"):
            replace(spec, schema_version="2.0")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            replace(spec, balance_report_sha256="not-a-digest")

        changed_metric_contract = replace(spec, metric_contract_sha256="0" * 64)
        changed_output_schema = replace(
            spec,
            output_profile_schema_sha256="1" * 64,
        )
        metric_result = weighted_mean(changed_metric_contract, weights, (0, 1))
        output_result = weighted_mean(changed_output_schema, weights, (0, 1))
        self.assertNotEqual(spec.estimand_sha256, changed_metric_contract.estimand_sha256)
        self.assertNotEqual(spec.estimand_sha256, changed_output_schema.estimand_sha256)
        self.assertNotEqual(result.result_sha256, metric_result.result_sha256)
        self.assertNotEqual(result.result_sha256, output_result.result_sha256)
        with self.assertRaisesRegex(ValueError, "dedicated target-population"):
            replace(spec, output_profile_id="policy_output_v2")


if __name__ == "__main__":
    unittest.main()
