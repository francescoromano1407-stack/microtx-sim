from __future__ import annotations

from dataclasses import fields, replace
import unittest

from microtx_sim.funding import (
    EPGCFirmInputs,
    EPGCPolicy,
    evaluate_epgc,
)
from microtx_sim.funding.epgc import INT64_MAX


def _policy(**overrides: int) -> EPGCPolicy:
    values = {
        "access_payment_cents_per_eligible_access": 100,
        "institutional_license_payment_cents_per_license": 1_000,
        "availability_payment_cents_per_period": 500,
        "accessibility_bonus_cents": 200,
        "multilingual_bonus_cents": 300,
        "cultural_value_bonus_cents": 400,
        "safety_certification_bonus_cents": 600,
        "prohibited_mechanics_penalty_cents": 100,
        "prohibited_mechanics_clawback_basis_points": 5_000,
        "maximum_budget_cents": 10_000,
    }
    values.update(overrides)
    return EPGCPolicy(**values)


def _inputs(**overrides: object) -> EPGCFirmInputs:
    values: dict[str, object] = {
        "fixed_price_revenue_cents": 2_000,
        "institutional_licensing_revenue_cents": 1_500,
        "non_targeted_sponsorship_revenue_cents": 500,
        "development_cost_cents": 5_000,
        "maintenance_cost_cents": 2_000,
        "eligible_access_count": 10,
        "eligible_institutional_license_count": 1,
        "availability_period_count": 1,
        "accessibility_eligible": False,
        "multilingual_support_eligible": False,
        "cultural_value_eligible": False,
        "safety_certified": False,
        "prohibited_mechanics_enabled": False,
    }
    values.update(overrides)
    return EPGCFirmInputs(**values)


class EPGCTests(unittest.TestCase):
    def test_safe_profit_uses_the_explicit_financing_equation(self) -> None:
        result = evaluate_epgc(_policy(), _inputs())

        expected = (
            result.public_contract_revenue_cents
            + result.fixed_price_revenue_cents
            + result.institutional_licensing_revenue_cents
            + result.non_targeted_sponsorship_revenue_cents
            - result.development_cost_cents
            - result.maintenance_cost_cents
        )
        self.assertEqual(result.profit_safe_cents, expected)
        self.assertEqual(result.public_contract_revenue_cents, 2_500)
        self.assertFalse(result.sustainable_under_policy)

    def test_minimum_public_contribution_is_the_exact_residual(self) -> None:
        result = evaluate_epgc(
            _policy(maximum_budget_cents=5_000),
            _inputs(
                fixed_price_revenue_cents=1_000,
                institutional_licensing_revenue_cents=500,
                non_targeted_sponsorship_revenue_cents=0,
                development_cost_cents=3_000,
                maintenance_cost_cents=1_000,
            ),
        )

        self.assertEqual(result.minimum_public_contribution_cents, 2_500)
        self.assertTrue(result.feasible_under_budget_cap)
        self.assertTrue(result.sustainable_under_policy)

    def test_budget_cap_can_make_the_required_contribution_infeasible(self) -> None:
        result = evaluate_epgc(
            _policy(maximum_budget_cents=900),
            _inputs(
                fixed_price_revenue_cents=0,
                institutional_licensing_revenue_cents=0,
                non_targeted_sponsorship_revenue_cents=0,
                development_cost_cents=1_500,
                maintenance_cost_cents=0,
            ),
        )

        self.assertEqual(result.minimum_public_contribution_cents, 1_500)
        self.assertFalse(result.feasible_under_budget_cap)
        self.assertEqual(result.budget_limited_public_contract_cents, 900)
        self.assertEqual(result.profit_safe_cents, -600)

    def test_public_value_bonuses_are_separate_auditable_components(self) -> None:
        result = evaluate_epgc(
            _policy(),
            _inputs(
                accessibility_eligible=True,
                multilingual_support_eligible=True,
                cultural_value_eligible=True,
                safety_certified=True,
            ),
        )

        self.assertEqual(result.accessibility_bonus_cents, 200)
        self.assertEqual(result.multilingual_bonus_cents, 300)
        self.assertEqual(result.cultural_value_bonus_cents, 400)
        self.assertEqual(result.safety_certification_bonus_cents, 600)
        self.assertEqual(result.gross_eligible_public_contract_cents, 4_000)

    def test_prohibited_mechanics_trigger_clawback_and_penalty(self) -> None:
        result = evaluate_epgc(
            _policy(
                access_payment_cents_per_eligible_access=1_000,
                institutional_license_payment_cents_per_license=0,
                availability_payment_cents_per_period=0,
                prohibited_mechanics_penalty_cents=100,
                prohibited_mechanics_clawback_basis_points=5_000,
            ),
            _inputs(
                fixed_price_revenue_cents=0,
                institutional_licensing_revenue_cents=0,
                non_targeted_sponsorship_revenue_cents=0,
                development_cost_cents=1_000,
                maintenance_cost_cents=0,
                eligible_access_count=1,
                eligible_institutional_license_count=0,
                availability_period_count=0,
                prohibited_mechanics_enabled=True,
            ),
        )

        self.assertEqual(result.clawback_cents, 500)
        self.assertEqual(result.penalty_cents, 100)
        self.assertEqual(result.public_contract_revenue_cents, 400)
        self.assertEqual(result.profit_safe_cents, -600)

    def test_zero_cost_case_requires_no_public_contribution(self) -> None:
        result = evaluate_epgc(
            _policy(
                access_payment_cents_per_eligible_access=0,
                institutional_license_payment_cents_per_license=0,
                availability_payment_cents_per_period=0,
                maximum_budget_cents=0,
            ),
            _inputs(
                fixed_price_revenue_cents=0,
                institutional_licensing_revenue_cents=0,
                non_targeted_sponsorship_revenue_cents=0,
                development_cost_cents=0,
                maintenance_cost_cents=0,
                eligible_access_count=0,
                eligible_institutional_license_count=0,
                availability_period_count=0,
            ),
        )

        self.assertEqual(result.minimum_public_contribution_cents, 0)
        self.assertTrue(result.feasible_under_budget_cap)
        self.assertEqual(result.profit_safe_cents, 0)
        self.assertTrue(result.sustainable_under_policy)

    def test_integer_extremes_are_exact_or_rejected_before_overflow(self) -> None:
        exact = evaluate_epgc(
            _policy(
                access_payment_cents_per_eligible_access=INT64_MAX,
                institutional_license_payment_cents_per_license=0,
                availability_payment_cents_per_period=0,
                maximum_budget_cents=INT64_MAX,
            ),
            _inputs(
                fixed_price_revenue_cents=0,
                institutional_licensing_revenue_cents=0,
                non_targeted_sponsorship_revenue_cents=0,
                development_cost_cents=INT64_MAX,
                maintenance_cost_cents=0,
                eligible_access_count=1,
                eligible_institutional_license_count=0,
                availability_period_count=0,
            ),
        )
        self.assertEqual(exact.profit_safe_cents, 0)

        with self.assertRaisesRegex(OverflowError, "eligible-access payment"):
            evaluate_epgc(
                _policy(
                    access_payment_cents_per_eligible_access=INT64_MAX,
                    institutional_license_payment_cents_per_license=0,
                    availability_payment_cents_per_period=0,
                    maximum_budget_cents=INT64_MAX,
                ),
                _inputs(eligible_access_count=2),
            )
        with self.assertRaises(ValueError):
            _policy(prohibited_mechanics_clawback_basis_points=10_001)
        with self.assertRaises(ValueError):
            _inputs(development_cost_cents=-1)

    def test_result_and_policy_validate_their_invariants(self) -> None:
        result = evaluate_epgc(_policy(), _inputs())
        with self.assertRaisesRegex(ValueError, "EPGC equation"):
            replace(result, profit_safe_cents=result.profit_safe_cents + 1)
        with self.assertRaises(TypeError):
            _inputs(eligible_access_count=True)

    def test_public_api_contains_no_prohibited_performance_metrics(self) -> None:
        prohibited = ("playtime", "retention", "conversion", "spend")
        api_names = {
            item.name.lower()
            for dataclass_type in (EPGCPolicy, EPGCFirmInputs)
            for item in fields(dataclass_type)
        }
        for fragment in prohibited:
            self.assertFalse(
                any(fragment in name for name in api_names),
                msg=f"prohibited metric leaked into EPGC API: {fragment}",
            )


if __name__ == "__main__":
    unittest.main()
