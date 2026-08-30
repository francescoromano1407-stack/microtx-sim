from __future__ import annotations

from dataclasses import replace
import unittest

from microtx_sim.analysis.uncertainty import (
    ConvergenceRule,
    ConvergenceStatus,
    ParameterDeclaration,
    ParameterProvenanceStatus,
    ParameterUncertaintyDesign,
    RealizationIdentity,
    UncertaintyAvailability,
    UncertaintyComponentStatus,
    UncertaintyRealization,
    UncertaintyValidationError,
    decompose_joint_uncertainty,
    evaluate_blockwise_convergence,
    final_sufficiency_judgment,
    generate_parameter_draws,
    summarize_seed_uncertainty,
)


def _identity(
    seed: int,
    *,
    parameter: str = "1",
    population: str = "1",
    rate: str = "1",
) -> RealizationIdentity:
    return RealizationIdentity(
        seed=seed,
        parameter_draw_id=f"parameter-{parameter}",
        parameter_draw_sha256=parameter * 64,
        population_design_id="population-design",
        population_replicate_id=f"population-{population}",
        population_design_sha256=population * 64,
        monetary_rate_draw_id=f"rate-{rate}",
        monetary_rate_basis_id=f"rate-basis-{rate}",
        monetary_rate_basis_sha256=rate * 64,
        scenario_id="baseline-minus-safe",
        primary_estimand_id="primary.composite-harm.v1",
        pretreatment_cohort_sha256="a" * 64,
        population_weights_sha256="b" * 64,
    )


def _realization(
    seed: int,
    estimate: float,
    *,
    parameter: str = "1",
    population: str = "1",
    rate: str = "1",
) -> UncertaintyRealization:
    return UncertaintyRealization(
        identity=_identity(
            seed,
            parameter=parameter,
            population=population,
            rate=rate,
        ),
        estimate=estimate,
        valid=True,
    )


class ParameterUncertaintyTests(unittest.TestCase):
    def _design(self) -> ParameterUncertaintyDesign:
        return ParameterUncertaintyDesign(
            design_id="illustrative-joint-v1",
            design_seed=9182,
            draw_count=8,
            parameters=(
                ParameterDeclaration(
                    parameter_id="decision.temperature",
                    source="declared engineering range",
                    provenance_status=ParameterProvenanceStatus.ILLUSTRATIVE_RANGE,
                    nominal_value=0.65,
                    lower_bound=0.5,
                    upper_bound=0.8,
                ),
                ParameterDeclaration(
                    parameter_id="harm.sleep_debt_weight",
                    source="declared engineering range",
                    provenance_status=ParameterProvenanceStatus.ILLUSTRATIVE_RANGE,
                    nominal_value=0.25,
                    lower_bound=0.1,
                    upper_bound=0.4,
                ),
            ),
        )

    def test_seeded_lhs_is_reproducible_content_addressed_and_not_probabilistic(self) -> None:
        design = self._design()
        first = generate_parameter_draws(design)
        second = generate_parameter_draws(design)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len({item.draw_sha256 for item in first}), 8)
        self.assertTrue(all(not item.probability_interpretation for item in first))

    def test_illustrative_range_cannot_be_silently_relabelled_distribution(self) -> None:
        with self.assertRaisesRegex(
            UncertaintyValidationError,
            "cannot be relabelled",
        ):
            ParameterDeclaration(
                parameter_id="decision.temperature",
                source="OAT diagnostic",
                provenance_status=ParameterProvenanceStatus.ILLUSTRATIVE_RANGE,
                nominal_value=0.65,
                lower_bound=0.5,
                upper_bound=0.8,
                probability_distribution="UNIFORM",
            )

    def test_correlation_requires_provenance(self) -> None:
        design = self._design()
        with self.assertRaisesRegex(UncertaintyValidationError, "provenance"):
            replace(
                design,
                correlation_matrix=((1.0, 0.2), (0.2, 1.0)),
                correlation_source=None,
            )


class SeedAndConvergenceTests(unittest.TestCase):
    def test_seed_summary_reports_sd_mcse_and_interval(self) -> None:
        rows = tuple(
            _realization(seed, value)
            for seed, value in zip((1, 2, 3), (0.1, 0.2, 0.3), strict=True)
        )
        summary = summarize_seed_uncertainty(rows, expected_seeds=(1, 2, 3))
        self.assertAlmostEqual(summary.point_estimate, 0.2)
        self.assertAlmostEqual(summary.sample_standard_deviation, 0.1)
        self.assertAlmostEqual(summary.monte_carlo_standard_error, 0.1 / 3**0.5)
        self.assertGreater(summary.interval_width, 0.0)

    def test_seed_summary_has_no_outcome_dependent_exclusion_path(self) -> None:
        invalid = UncertaintyRealization(
            identity=_identity(2),
            estimate=None,
            valid=False,
            invalid_reason="non-finite output",
        )
        with self.assertRaisesRegex(UncertaintyValidationError, "fail closed"):
            summarize_seed_uncertainty(
                (_realization(1, 1.0), invalid),
                expected_seeds=(1, 2),
            )
        with self.assertRaisesRegex(UncertaintyValidationError, "complete fixed"):
            summarize_seed_uncertainty(
                (_realization(1, 1.0),),
                expected_seeds=(1, 2),
            )

    def test_seed_only_summary_rejects_changed_parameter_identity(self) -> None:
        with self.assertRaisesRegex(UncertaintyValidationError, "fixed parameter"):
            summarize_seed_uncertainty(
                (
                    _realization(1, 1.0, parameter="1"),
                    _realization(2, 1.0, parameter="2"),
                ),
                expected_seeds=(1, 2),
            )

    def test_blockwise_example_converges_only_after_two_passing_checkpoints(self) -> None:
        values = [0.1000] * 50 + [0.1020] * 50 + [0.0995] * 50
        rows = tuple(_realization(seed, value) for seed, value in enumerate(values, 1))
        rule = ConvergenceRule(
            block_size=50,
            minimum_retained_seeds=100,
            maximum_mcse=0.001,
            maximum_interval_width=0.004,
            maximum_absolute_change=0.002,
            maximum_relative_change=0.02,
            maximum_invalid_rate=0.0,
            consecutive_passing_checkpoints=2,
        )
        checkpoints = evaluate_blockwise_convergence(
            rows,
            expected_seeds=tuple(range(1, 151)),
            rule=rule,
        )
        self.assertEqual([item.retained_seed_count for item in checkpoints], [50, 100, 150])
        self.assertAlmostEqual(checkpoints[0].cumulative_point_estimate or 0.0, 0.1000)
        self.assertAlmostEqual(checkpoints[1].cumulative_point_estimate or 0.0, 0.1010)
        self.assertAlmostEqual(checkpoints[2].cumulative_point_estimate or 0.0, 0.1005)
        self.assertIs(checkpoints[1].status, ConvergenceStatus.NON_CONVERGED)
        self.assertIs(checkpoints[2].status, ConvergenceStatus.CONVERGED)

    def test_invalid_or_excluded_realization_is_unstable(self) -> None:
        rows = [_realization(seed, 0.1) for seed in range(1, 101)]
        rows[9] = UncertaintyRealization(
            identity=_identity(10),
            estimate=None,
            valid=False,
            invalid_reason="incomplete result",
        )
        checkpoint = evaluate_blockwise_convergence(
            rows,
            expected_seeds=tuple(range(1, 101)),
            rule=ConvergenceRule(block_size=100),
        )[-1]
        self.assertIs(checkpoint.status, ConvergenceStatus.UNSTABLE)
        self.assertIn("invalid_run_rate", checkpoint.blockers)

    def test_partial_final_block_cannot_complete_consecutive_rule(self) -> None:
        rows = tuple(_realization(seed, 0.1) for seed in range(1, 126))
        checkpoints = evaluate_blockwise_convergence(
            rows,
            expected_seeds=tuple(range(1, 151)),
            rule=ConvergenceRule(
                block_size=50,
                consecutive_passing_checkpoints=2,
            ),
        )
        self.assertEqual(
            [item.completed_realization_count for item in checkpoints],
            [50, 100, 125],
        )
        self.assertIs(checkpoints[-1].status, ConvergenceStatus.NON_CONVERGED)
        self.assertIn(
            "incomplete_deterministic_block",
            checkpoints[-1].blockers,
        )

    def test_convergence_rejects_changed_fixed_input_identity(self) -> None:
        rows = [
            _realization(seed, 0.1, parameter="1")
            for seed in range(1, 101)
        ]
        rows[-1] = _realization(100, 0.1, parameter="2")
        with self.assertRaisesRegex(
            UncertaintyValidationError,
            "fixed parameter",
        ):
            evaluate_blockwise_convergence(
                rows,
                expected_seeds=tuple(range(1, 101)),
                rule=ConvergenceRule(block_size=50),
            )

    def test_missing_required_component_prevents_converged_label(self) -> None:
        rows = tuple(_realization(seed, 0.1) for seed in range(1, 151))
        checkpoints = evaluate_blockwise_convergence(
            rows,
            expected_seeds=tuple(range(1, 151)),
            rule=ConvergenceRule(block_size=50),
            required_components_available=False,
        )
        self.assertIs(checkpoints[-1].status, ConvergenceStatus.NON_CONVERGED)
        self.assertIn(
            "required_uncertainty_component_unavailable",
            checkpoints[-1].blockers,
        )


class CombinedUncertaintyTests(unittest.TestCase):
    def test_full_factorial_variance_decomposition_sums_without_double_counting(self) -> None:
        rows = []
        for seed in (1, 2):
            for parameter in ("1", "2"):
                for population in ("1", "2"):
                    for rate in ("1", "2"):
                        value = (
                            0.1 * seed
                            + 0.2 * int(parameter)
                            + 0.3 * int(population)
                            + 0.4 * int(rate)
                        )
                        rows.append(
                            _realization(
                                seed,
                                value,
                                parameter=parameter,
                                population=population,
                                rate=rate,
                            )
                        )
        result = decompose_joint_uncertainty(rows)
        self.assertTrue(result.identifiable)
        components = (
            result.seed_only_variance,
            result.between_parameter_variance,
            result.between_population_variance,
            result.between_rate_variance,
            result.residual_or_interaction_variance,
        )
        self.assertTrue(all(value is not None for value in components))
        self.assertAlmostEqual(sum(float(value) for value in components), result.total_joint_variance or -1.0)

    def test_single_level_components_are_unavailable_not_zero(self) -> None:
        result = decompose_joint_uncertainty(
            (_realization(1, 0.1), _realization(2, 0.2))
        )
        self.assertIsNone(result.between_parameter_variance)
        self.assertIsNone(result.between_population_variance)
        self.assertIsNone(result.between_rate_variance)
        self.assertIsNone(result.residual_or_interaction_variance)
        self.assertFalse(result.identifiable)

    def test_distinct_rate_draw_ids_under_one_basis_are_decomposed(self) -> None:
        rows = []
        for seed in (1, 2):
            for parameter in ("1", "2"):
                for population in ("1", "2"):
                    for rate in ("1", "2"):
                        identity = _identity(
                            seed,
                            parameter=parameter,
                            population=population,
                            rate="1",
                        )
                        identity = replace(
                            identity,
                            monetary_rate_draw_id=f"rate-{rate}",
                        )
                        rows.append(
                            UncertaintyRealization(
                                identity=identity,
                                estimate=float(rate),
                                valid=True,
                            )
                        )
        result = decompose_joint_uncertainty(rows)
        self.assertTrue(result.identifiable)
        self.assertGreater(result.between_rate_variance or 0.0, 0.0)

    def test_joint_design_rejects_changed_shared_random_state(self) -> None:
        first = _realization(1, 0.1, parameter="1")
        second = _realization(1, 0.2, parameter="2")
        second = replace(
            second,
            identity=replace(
                second.identity,
                pretreatment_cohort_sha256="c" * 64,
            ),
        )
        with self.assertRaisesRegex(
            UncertaintyValidationError,
            "pre-treatment cohort",
        ):
            decompose_joint_uncertainty((first, second))

    def test_invalid_rows_cannot_hide_cross_estimand_pooling(self) -> None:
        invalid = UncertaintyRealization(
            identity=replace(
                _identity(2),
                primary_estimand_id="different-estimand",
            ),
            estimate=None,
            valid=False,
            invalid_reason="incomplete output",
        )
        with self.assertRaisesRegex(
            UncertaintyValidationError,
            "cannot pool scenarios or estimands",
        ):
            decompose_joint_uncertainty((_realization(1, 0.1), invalid))

    def test_unquantified_component_cannot_be_assigned_zero(self) -> None:
        with self.assertRaisesRegex(UncertaintyValidationError, "must not be assigned"):
            UncertaintyComponentStatus(
                source="monetary_rate",
                availability=UncertaintyAvailability.UNQUANTIFIED,
                variance=0.0,
                method=None,
                blocker="point rate only",
            )

    def test_final_judgment_separates_convergence_from_scientific_sufficiency(self) -> None:
        components = (
            UncertaintyComponentStatus(
                source="seed",
                availability=UncertaintyAvailability.QUANTIFIED,
                variance=0.01,
                method="sample variance",
                blocker=None,
            ),
            UncertaintyComponentStatus(
                source="parameter",
                availability=UncertaintyAvailability.UNQUANTIFIED,
                variance=None,
                method=None,
                blocker="illustrative ranges only",
            ),
        )
        result = final_sufficiency_judgment(
            convergence_status=ConvergenceStatus.CONVERGED,
            components=components,
        )
        self.assertFalse(result["sufficient"])
        self.assertFalse(result["campaign_ready"])
        self.assertIn(
            "uncertainty_component_unavailable:parameter",
            result["blockers"],
        )


if __name__ == "__main__":
    unittest.main()
