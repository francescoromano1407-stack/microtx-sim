from __future__ import annotations

from dataclasses import replace
import unittest

from microtx_sim.causal.design import (
    ATOMIC_FACTOR_NAMES,
    CampaignValidationError,
    CausalFactorMatrixValidationError,
    CausalDesignStatus,
    CausalFactor,
    ContrastClassification,
    ContrastRole,
    ContrastStructuralScope,
    FactorDifference,
    ScenarioFactorVector,
    assess_causal_design,
    build_causal_design_registry,
)
from microtx_sim.causal.scenarios import (
    ScenarioId,
    ScenarioSpec,
    required_scenarios,
)


EXPECTED_FACTOR_NAMES = (
    "direct_price_cents",
    "opaque_virtual_currency",
    "paid_random_rewards",
    "progression_gates",
    "time_limited_offers",
    "daily_streak_pressure",
    "pay_to_progress",
    "pay_to_win",
    "social_guild_pressure",
    "purchase_friction",
    "spending_cap_cents",
    "cooling_off_hours",
    "real_currency_price_display",
    "personalized_offers",
    "fixed_access_price_cents",
    "subscription_price_cents",
    "epgc_enabled",
)

EXPECTED_MATRIX_VALUES = (
    (
        299,
        0.75,
        0.70,
        0.55,
        0.70,
        0.60,
        0.55,
        0.50,
        0.50,
        0.20,
        None,
        0,
        False,
        False,
        0,
        0,
        False,
    ),
    (
        299,
        0.0,
        0.70,
        0.55,
        0.70,
        0.60,
        0.55,
        0.50,
        0.50,
        0.65,
        None,
        0,
        True,
        False,
        0,
        0,
        False,
    ),
    (
        299,
        0.75,
        0.0,
        0.55,
        0.70,
        0.60,
        0.55,
        0.50,
        0.50,
        0.20,
        None,
        0,
        False,
        False,
        0,
        0,
        False,
    ),
    (
        299,
        0.75,
        0.70,
        0.55,
        0.0,
        0.60,
        0.55,
        0.50,
        0.50,
        0.20,
        None,
        0,
        False,
        False,
        0,
        0,
        False,
    ),
    (
        299,
        0.75,
        0.70,
        0.55,
        0.70,
        0.60,
        0.55,
        0.50,
        0.50,
        0.70,
        2_500,
        24,
        True,
        False,
        0,
        0,
        False,
    ),
    (
        0,
        0.0,
        0.0,
        0.05,
        0.0,
        0.05,
        0.0,
        0.0,
        0.10,
        0.85,
        2_500,
        24,
        True,
        False,
        1_499,
        499,
        False,
    ),
    (
        0,
        0.0,
        0.0,
        0.05,
        0.0,
        0.05,
        0.0,
        0.0,
        0.10,
        0.85,
        2_500,
        24,
        True,
        False,
        299,
        0,
        True,
    ),
)

EXPECTED_MATRIX_SHA256 = (
    "71800380fa68cd6b5e4f0bd71a0d72702e6f9000f4410634e9306192e6bd5b2a"
)
EXPECTED_CONTRASTS_SHA256 = (
    "94c8eee73fd9eede8bead628fc73b4056cd009fef1e48e853150a27c239abd91"
)
EXPECTED_DESIGN_SHA256 = (
    "347188a2c629b85ff444b0c21ad8ca3c1c918708db76b0699e4642d9f7861dd7"
)


class CausalDesignTests(unittest.TestCase):
    def test_atomic_factor_order_and_canonical_matrix_are_frozen(self) -> None:
        registry = build_causal_design_registry(required_scenarios())

        self.assertEqual(ATOMIC_FACTOR_NAMES, EXPECTED_FACTOR_NAMES)
        self.assertEqual(
            tuple(row.scenario_id for row in registry.scenario_matrix),
            tuple(ScenarioId),
        )
        self.assertEqual(
            tuple(row.values for row in registry.scenario_matrix),
            EXPECTED_MATRIX_VALUES,
        )
        self.assertEqual(registry.scenario_matrix_sha256(), EXPECTED_MATRIX_SHA256)

    def test_registry_is_explicitly_retrospective_and_fails_campaign_use(self) -> None:
        registry = build_causal_design_registry(required_scenarios())

        self.assertIs(
            registry.status,
            CausalDesignStatus.RETROSPECTIVE_SYNTHETIC,
        )
        self.assertEqual(registry.status.value, "RETROSPECTIVE_SYNTHETIC")
        self.assertFalse(registry.preregistered)
        self.assertFalse(registry.campaign_ready)
        self.assertTrue(registry.canonical_match)
        self.assertEqual(
            registry.campaign_blockers,
            (
                "retrospective_synthetic_design",
                "causal_design_not_preregistered",
                "empirical_calibration_required",
            ),
        )
        with self.assertRaises(CampaignValidationError) as caught:
            registry.validate_for_campaign()
        self.assertEqual(caught.exception.blockers, registry.campaign_blockers)

    def test_all_pairwise_contrasts_are_diagnostic_and_not_estimands(self) -> None:
        registry = build_causal_design_registry(required_scenarios())
        snapshots = registry.contrasts_snapshot()

        self.assertEqual(len(registry.contrasts), 49)
        self.assertEqual(
            tuple(
                (
                    contrast.reference_scenario_id,
                    contrast.comparison_scenario_id,
                )
                for contrast in registry.contrasts
            ),
            tuple(
                (reference, comparison)
                for reference in ScenarioId
                for comparison in ScenarioId
            ),
        )
        self.assertEqual(
            {
                classification: sum(
                    contrast.classification is classification
                    for contrast in registry.contrasts
                )
                for classification in ContrastClassification
            },
            {
                ContrastClassification.IDENTITY: 7,
                ContrastClassification.SINGLE_FACTOR: 4,
                ContrastClassification.BUNDLE: 38,
            },
        )
        self.assertEqual(
            {
                scope: sum(
                    contrast.structural_scope is scope
                    for contrast in registry.contrasts
                )
                for scope in ContrastStructuralScope
            },
            {
                ContrastStructuralScope.IDENTITY: 7,
                ContrastStructuralScope.MECHANICS_ONLY: 20,
                ContrastStructuralScope.FINANCING_ONLY: 2,
                ContrastStructuralScope.MIXED: 20,
            },
        )
        self.assertTrue(
            all(
                ContrastRole.EXHAUSTIVE_PAIRWISE_DIAGNOSTIC in contrast.roles
                for contrast in registry.contrasts
            )
        )
        self.assertTrue(all(not item["planned_estimand"] for item in snapshots))
        self.assertTrue(all(not item["preregistered"] for item in snapshots))
        payload = registry.snapshot()
        self.assertEqual(
            payload["contrast_scope"],
            "exhaustive_directed_pairwise_diagnostics",
        )
        self.assertFalse(payload["planned_estimands"])
        self.assertFalse(payload["preregistered_estimands"])

    def test_reported_and_catalogue_contrast_roles_are_exact(self) -> None:
        registry = build_causal_design_registry(required_scenarios())
        reported = tuple(
            contrast.contrast_id
            for contrast in registry.contrasts
            if ContrastRole.REPORTED_EFFECT_VS_SAFE in contrast.roles
        )
        declared = tuple(
            contrast.contrast_id
            for contrast in registry.contrasts
            if ContrastRole.DECLARED_CATALOGUE_CHECK in contrast.roles
        )

        self.assertEqual(reported, registry.reported_effect_vs_safe_contrast_ids)
        self.assertEqual(declared, registry.declared_catalogue_contrast_ids)
        self.assertEqual(len(reported), 7)
        self.assertEqual(len(declared), 5)
        self.assertEqual(
            set(reported) & set(declared),
            {"safe_fixed_price_subscription__to__epgc"},
        )
        self.assertEqual(
            registry.declared_catalogue_contrast_ids,
            (
                "baseline_f2p__to__transparent_direct_price",
                "baseline_f2p__to__no_random_rewards",
                "baseline_f2p__to__no_time_limited_pressure",
                "baseline_f2p__to__spending_cap_cooling_off",
                "safe_fixed_price_subscription__to__epgc",
            ),
        )
        diagnostic_only = tuple(
            contrast
            for contrast in registry.contrasts
            if contrast.roles
            == (ContrastRole.EXHAUSTIVE_PAIRWISE_DIAGNOSTIC,)
        )
        self.assertEqual(len(diagnostic_only), 38)

    def test_exact_contrast_factor_differences_are_retained(self) -> None:
        registry = build_causal_design_registry(required_scenarios())

        identity = registry.contrast(
            ScenarioId.BASELINE_F2P,
            ScenarioId.BASELINE_F2P,
        )
        self.assertIs(identity.classification, ContrastClassification.IDENTITY)
        self.assertEqual(identity.factor_differences, ())

        no_random = registry.contrast(
            ScenarioId.BASELINE_F2P,
            ScenarioId.NO_RANDOM_REWARDS,
        )
        self.assertIs(
            no_random.classification,
            ContrastClassification.SINGLE_FACTOR,
        )
        self.assertEqual(
            no_random.factor_differences,
            (
                FactorDifference(
                    CausalFactor.PAID_RANDOM_REWARDS,
                    0.70,
                    0.0,
                ),
            ),
        )

        expected_factors = {
            ScenarioId.TRANSPARENT_DIRECT_PRICE: (
                CausalFactor.OPAQUE_VIRTUAL_CURRENCY,
                CausalFactor.PURCHASE_FRICTION,
                CausalFactor.REAL_CURRENCY_PRICE_DISPLAY,
            ),
            ScenarioId.SPENDING_CAP_COOLING_OFF: (
                CausalFactor.PURCHASE_FRICTION,
                CausalFactor.SPENDING_CAP_CENTS,
                CausalFactor.COOLING_OFF_HOURS,
                CausalFactor.REAL_CURRENCY_PRICE_DISPLAY,
            ),
        }
        for comparison, factors in expected_factors.items():
            with self.subTest(comparison=comparison):
                contrast = registry.contrast(ScenarioId.BASELINE_F2P, comparison)
                self.assertEqual(contrast.differing_factors, factors)

        safe_to_epgc = registry.contrast(
            ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
            ScenarioId.EPGC,
        )
        self.assertEqual(
            safe_to_epgc.differing_factors,
            (
                CausalFactor.FIXED_ACCESS_PRICE_CENTS,
                CausalFactor.SUBSCRIPTION_PRICE_CENTS,
                CausalFactor.EPGC_ENABLED,
            ),
        )
        self.assertIs(
            safe_to_epgc.structural_scope,
            ContrastStructuralScope.FINANCING_ONLY,
        )
        self.assertEqual(
            tuple(
                len(
                    registry.contrast(
                        ScenarioId.SAFE_FIXED_PRICE_SUBSCRIPTION,
                        scenario_id,
                    ).factor_differences
                )
                for scenario_id in ScenarioId
            ),
            (15, 13, 14, 14, 12, 0, 3),
        )

    def test_order_and_nondesign_labels_cannot_change_snapshots_or_hashes(self) -> None:
        scenarios = required_scenarios()
        canonical = build_causal_design_registry(scenarios)
        reverse = build_causal_design_registry(tuple(reversed(scenarios)))
        relabelled = build_causal_design_registry(
            (
                replace(
                    scenarios[0],
                    label="A different descriptive label",
                    description="Different descriptive prose.",
                ),
                *scenarios[1:],
            )
        )

        self.assertEqual(canonical.snapshot(), reverse.snapshot())
        self.assertEqual(canonical.snapshot(), relabelled.snapshot())
        self.assertEqual(canonical.scenario_matrix_sha256(), EXPECTED_MATRIX_SHA256)
        self.assertEqual(canonical.contrasts_sha256(), EXPECTED_CONTRASTS_SHA256)
        self.assertEqual(canonical.snapshot_sha256(), EXPECTED_DESIGN_SHA256)
        self.assertEqual(reverse.snapshot_sha256(), EXPECTED_DESIGN_SHA256)
        self.assertEqual(relabelled.snapshot_sha256(), EXPECTED_DESIGN_SHA256)

        assessment = assess_causal_design(tuple(reversed(scenarios)))
        assessment_payload = assessment.manifest_payload(
            run_input_sha256="e" * 64
        )
        self.assertEqual(
            assessment.design_snapshot(),
            canonical.design_snapshot(),
        )
        self.assertEqual(
            assessment_payload["design_sha256"],
            EXPECTED_DESIGN_SHA256,
        )
        self.assertEqual(
            assessment_payload["canonical_design_sha256"],
            EXPECTED_DESIGN_SHA256,
        )
        self.assertEqual(
            assessment_payload["assessment_sha256"],
            assessment.snapshot_sha256(),
        )

    def test_factor_drift_fails_registry_but_is_descriptively_assessed(self) -> None:
        scenarios = required_scenarios()
        changed_baseline = replace(
            scenarios[0],
            mechanics=replace(
                scenarios[0].mechanics,
                paid_random_rewards=0.69,
            ),
        )
        custom = (changed_baseline, *scenarios[1:])

        with self.assertRaisesRegex(
            CausalFactorMatrixValidationError,
            "baseline_f2p:paid_random_rewards",
        ) as caught:
            build_causal_design_registry(custom)
        self.assertEqual(len(caught.exception.mismatches), 1)

        assessment = assess_causal_design(tuple(reversed(custom)))
        self.assertFalse(assessment.canonical_match)
        self.assertFalse(assessment.campaign_ready)
        self.assertIn(
            "scenario_factor_matrix_not_canonical",
            assessment.campaign_blockers,
        )
        self.assertNotEqual(
            assessment.scenario_matrix_sha256(),
            EXPECTED_MATRIX_SHA256,
        )
        self.assertEqual(len(assessment.canonical_mismatches), 1)
        mismatch = assessment.canonical_mismatches[0]
        self.assertIs(mismatch.scenario_id, ScenarioId.BASELINE_F2P)
        self.assertEqual(
            mismatch.factor_differences,
            (
                FactorDifference(
                    CausalFactor.PAID_RANDOM_REWARDS,
                    0.70,
                    0.69,
                ),
            ),
        )
        payload = assessment.manifest_payload(run_input_sha256="a" * 64)
        self.assertFalse(payload["canonical_match"])
        self.assertEqual(payload["run_input_sha256"], "a" * 64)
        self.assertEqual(len(payload["design_sha256"]), 64)
        self.assertEqual(payload["design_sha256"], assessment.design_sha256())
        self.assertEqual(
            payload["assessment_sha256"],
            assessment.snapshot_sha256(),
        )
        self.assertEqual(payload["canonical_design_sha256"], EXPECTED_DESIGN_SHA256)

        signed_zero_transparent = replace(
            scenarios[1],
            mechanics=replace(
                scenarios[1].mechanics,
                opaque_virtual_currency=-0.0,
            ),
        )
        signed_zero_scenarios = (
            scenarios[0],
            signed_zero_transparent,
            *scenarios[2:],
        )
        with self.assertRaisesRegex(
            CausalFactorMatrixValidationError,
            "transparent_direct_price:opaque_virtual_currency",
        ):
            build_causal_design_registry(signed_zero_scenarios)
        signed_zero_assessment = assess_causal_design(signed_zero_scenarios)
        self.assertFalse(signed_zero_assessment.canonical_match)
        self.assertNotEqual(
            signed_zero_assessment.design_sha256(),
            EXPECTED_DESIGN_SHA256,
        )

    def test_wrong_factor_and_primitive_types_fail_closed(self) -> None:
        registry = build_causal_design_registry(required_scenarios())
        baseline_values = registry.scenario_matrix[0].values

        with self.assertRaisesRegex(TypeError, "factor must be a CausalFactor"):
            registry.factor_value(  # type: ignore[arg-type]
                ScenarioId.BASELINE_F2P,
                "paid_random_rewards",
            )
        with self.assertRaisesRegex(TypeError, "factor must be a CausalFactor"):
            FactorDifference(  # type: ignore[arg-type]
                "paid_random_rewards",
                0.70,
                0.0,
            )
        with self.assertRaisesRegex(TypeError, "values must be a tuple"):
            ScenarioFactorVector(  # type: ignore[arg-type]
                ScenarioId.BASELINE_F2P,
                list(baseline_values),
            )
        wrong_float_type = list(baseline_values)
        wrong_float_type[1] = 0
        with self.assertRaisesRegex(TypeError, "must be a float"):
            ScenarioFactorVector(
                ScenarioId.BASELINE_F2P,
                tuple(wrong_float_type),
            )

    def test_scenario_set_and_manifest_linkage_are_strict(self) -> None:
        scenarios = required_scenarios()
        duplicate = (*scenarios[:-1], scenarios[0])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_causal_design_registry(duplicate)
        with self.assertRaisesRegex(TypeError, "exact ScenarioSpec"):
            build_causal_design_registry(  # type: ignore[arg-type]
                (*scenarios[:-1], object())
            )

        registry = build_causal_design_registry(scenarios)
        with self.assertRaisesRegex(TypeError, "must be a string"):
            registry.manifest_payload(run_input_sha256=1)  # type: ignore[arg-type]
        for invalid in ("A" * 64, "a" * 63, "z" * 64):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
                    registry.manifest_payload(run_input_sha256=invalid)
        payload = registry.manifest_payload(run_input_sha256="f" * 64)
        self.assertEqual(payload["design_sha256"], EXPECTED_DESIGN_SHA256)
        self.assertEqual(payload["run_input_sha256"], "f" * 64)


if __name__ == "__main__":
    unittest.main()
