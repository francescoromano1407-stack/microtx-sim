from __future__ import annotations

from dataclasses import replace
import unittest

from microtx_sim.domain.monetisation import MonetisationVector


class MonetisationVectorTests(unittest.TestCase):
    def test_safe_defaults_are_explicit_and_personalisation_is_off(self) -> None:
        vector = MonetisationVector(direct_price_cents=499)

        self.assertEqual(vector.direct_price_cents, 499)
        self.assertFalse(vector.personalized_offers)
        self.assertTrue(vector.real_currency_price_display)
        self.assertEqual(vector.price_transparency, 1.0)
        self.assertEqual(vector.purchase_pressure, 0.0)
        self.assertEqual(vector.risk_exposure, 0.0)

    def test_all_pressure_coordinates_have_interpretable_monotonic_effects(self) -> None:
        safe = MonetisationVector(direct_price_cents=499)
        risky = MonetisationVector(
            direct_price_cents=499,
            opaque_virtual_currency=1.0,
            paid_random_rewards=1.0,
            progression_gates=1.0,
            time_limited_offers=1.0,
            daily_streak_pressure=1.0,
            pay_to_progress=1.0,
            pay_to_win=1.0,
            social_guild_pressure=1.0,
            purchase_friction=0.0,
            real_currency_price_display=False,
            personalized_offers=True,
        )
        protected = replace(
            risky,
            spending_cap_cents=2_000,
            cooling_off_hours=24,
        )

        self.assertEqual(risky.purchase_pressure, 1.0)
        self.assertEqual(risky.price_transparency, 0.0)
        self.assertGreater(risky.risk_exposure, safe.risk_exposure)
        self.assertLess(protected.risk_exposure, risky.risk_exposure)

    def test_each_mechanic_has_an_isolated_interpretable_effect(self) -> None:
        safe = MonetisationVector(direct_price_cents=499)
        self.assertEqual(safe.constrain_purchase(safe.direct_price_cents), 499)

        pressure_coordinates = (
            "opaque_virtual_currency",
            "paid_random_rewards",
            "progression_gates",
            "time_limited_offers",
            "daily_streak_pressure",
            "pay_to_progress",
            "pay_to_win",
            "social_guild_pressure",
        )
        for coordinate in pressure_coordinates:
            with self.subTest(mechanic=coordinate):
                exposed = replace(safe, **{coordinate: 1.0})
                self.assertGreater(exposed.purchase_pressure, safe.purchase_pressure)
                self.assertGreater(exposed.risk_exposure, safe.risk_exposure)

        low_friction = replace(safe, purchase_friction=0.0)
        self.assertGreater(low_friction.purchase_pressure, safe.purchase_pressure)
        self.assertGreater(low_friction.risk_exposure, safe.risk_exposure)

        hidden_real_price = replace(safe, real_currency_price_display=False)
        self.assertLess(hidden_real_price.price_transparency, safe.price_transparency)
        self.assertGreater(hidden_real_price.risk_exposure, safe.risk_exposure)

        personalized = replace(safe, personalized_offers=True)
        self.assertGreater(personalized.risk_exposure, safe.risk_exposure)

        exposed = MonetisationVector(paid_random_rewards=1.0)
        capped = replace(exposed, spending_cap_cents=1_000)
        cooled = replace(exposed, cooling_off_hours=24)
        self.assertLess(capped.risk_exposure, exposed.risk_exposure)
        self.assertLess(cooled.risk_exposure, exposed.risk_exposure)

    def test_cap_and_cooling_off_constrain_actual_amounts(self) -> None:
        vector = MonetisationVector(
            spending_cap_cents=1_000,
            cooling_off_hours=24,
        )

        self.assertEqual(vector.remaining_spending_cap_cents(700), 300)
        self.assertEqual(
            vector.constrain_purchase(
                900,
                already_spent_cents=700,
                hours_since_last_purchase=30,
            ),
            300,
        )
        self.assertEqual(
            vector.constrain_purchase(
                100,
                already_spent_cents=0,
                hours_since_last_purchase=4,
            ),
            0,
        )
        self.assertEqual(
            vector.constrain_purchase(100, hours_since_last_purchase=None), 100
        )

    def test_invalid_mechanisms_and_money_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "paid_random_rewards"):
            MonetisationVector(paid_random_rewards=1.01)
        with self.assertRaisesRegex(TypeError, "personalized_offers"):
            MonetisationVector(personalized_offers=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "spending_cap_cents"):
            MonetisationVector(spending_cap_cents=-1)
        with self.assertRaisesRegex(ValueError, "hours_since_last_purchase"):
            MonetisationVector(cooling_off_hours=12).cooling_off_active(-0.5)


if __name__ == "__main__":
    unittest.main()
