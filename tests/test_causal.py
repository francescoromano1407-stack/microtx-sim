from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from microtx_sim.causal.interventions import (
    AuditRegime,
    CompositeIntervention,
    MechanismCap,
    NullIntervention,
    SubsidyRegime,
)
from microtx_sim.causal.paired_worlds import PairedOutcome, compare_outcomes
from microtx_sim.metrics.outcomes import HarmWeights, OutcomeRecorder, OutcomeSnapshot
from microtx_sim.types import MonetisationMechanism


def _outcome(
    harm_shift: float = 0.0,
    spend_shift: int = 0,
    *,
    income_shift: int = 0,
    debt_shift: int = 0,
    cash_shift: int = 0,
    margin_shift: int = 0,
    subsidy_shift: int = 0,
    tick: int = 10,
    player_count: int = 3,
    firm_count: int = 1,
    state_count: int = 1,
) -> OutcomeSnapshot:
    harm = np.zeros((player_count, 7), dtype=np.float64)
    harm[:, 0] = harm_shift
    return OutcomeSnapshot(
        tick=tick,
        player_ids=np.arange(player_count, dtype=np.int64),
        player_harm=harm,
        player_spend_cents=(
            np.arange(player_count, dtype=np.int64) * 100 + spend_shift
        ),
        player_income_cents=(
            np.full(player_count, 100_000 + income_shift, dtype=np.int64)
        ),
        player_debt_cents=np.full(player_count, debt_shift, dtype=np.int64),
        firm_ids=np.arange(firm_count, dtype=np.int64),
        firm_cash_cents=np.full(
            firm_count,
            1_000_000 + cash_shift,
            dtype=np.int64,
        ),
        firm_operating_margin_cents=np.full(
            firm_count,
            100_000 + margin_shift,
            dtype=np.int64,
        ),
        firm_safe_revenue_share=np.full(firm_count, 0.9, dtype=np.float64),
        jurisdiction_ids=np.arange(state_count, dtype=np.int64),
        state_subsidy_outlay_cents=np.full(
            state_count,
            subsidy_shift,
            dtype=np.int64,
        ),
    )


class _MockWorld:
    def __init__(self) -> None:
        self.calls: list[tuple[MonetisationMechanism, float, tuple[int, ...] | None]] = []
        self.audit_calls: list[dict[str, object]] = []
        self.subsidy_calls: list[dict[str, object]] = []

    def cap_mechanism(self, *, mechanism, maximum, game_ids) -> None:
        self.calls.append((mechanism, maximum, game_ids))

    def configure_audit_regime(self, **kwargs) -> None:
        self.audit_calls.append(kwargs)

    def configure_subsidy_regime(self, **kwargs) -> None:
        self.subsidy_calls.append(kwargs)


class CausalTests(unittest.TestCase):
    def test_null_difference_is_exactly_zero(self) -> None:
        outcome = _outcome()
        paired, effect = compare_outcomes(outcome, outcome)
        self.assertTrue(np.all(paired.player_harm_difference == 0.0))
        self.assertTrue(np.all(paired.player_spend_difference_cents == 0))
        self.assertTrue(
            np.all(
                paired.player_income_negative_control_difference_cents == 0
            )
        )
        self.assertTrue(paired.player_income_negative_control_passed)
        self.assertTrue(np.all(paired.player_debt_difference_cents == 0))
        self.assertTrue(np.all(paired.firm_margin_difference_cents == 0))
        self.assertTrue(np.all(paired.firm_cash_difference_cents == 0))
        self.assertTrue(np.all(paired.state_subsidy_difference_cents == 0))
        self.assertEqual(effect.total_spend_effect_cents, 0)
        self.assertEqual(effect.total_debt_effect_cents, 0)
        self.assertEqual(effect.total_operating_margin_effect_cents, 0)
        self.assertEqual(effect.total_subsidy_effect_cents, 0)
        self.assertEqual(effect.affected_player_share, 0.0)

    def test_known_shift_recovers_every_paired_effect_exactly(self) -> None:
        treated = _outcome(
            harm_shift=0.7,
            spend_shift=50,
            debt_shift=7,
            cash_shift=500,
            margin_shift=125,
            subsidy_shift=40,
        )
        control = _outcome()
        paired, effect = compare_outcomes(treated, control)
        self.assertEqual(paired.player_harm_difference.shape, (3, 7))
        np.testing.assert_array_equal(
            paired.player_harm_difference[:, 0],
            np.full(3, 0.7),
        )
        np.testing.assert_array_equal(
            paired.player_spend_difference_cents,
            np.full(3, 50),
        )
        np.testing.assert_array_equal(
            paired.player_income_negative_control_difference_cents,
            np.zeros(3, dtype=np.int64),
        )
        np.testing.assert_array_equal(
            paired.player_debt_difference_cents,
            np.full(3, 7),
        )
        np.testing.assert_array_equal(
            paired.firm_margin_difference_cents,
            np.array([125]),
        )
        np.testing.assert_array_equal(
            paired.firm_cash_difference_cents,
            np.array([500]),
        )
        np.testing.assert_array_equal(
            paired.state_subsidy_difference_cents,
            np.array([40]),
        )
        self.assertEqual(effect.total_spend_effect_cents, 150)
        self.assertEqual(effect.total_debt_effect_cents, 21)
        self.assertEqual(effect.total_operating_margin_effect_cents, 125)
        self.assertEqual(effect.total_subsidy_effect_cents, 40)
        self.assertAlmostEqual(effect.mean_composite_harm_effect, 0.1)
        self.assertEqual(effect.affected_player_share, 1.0)

    def test_signed_harm_effects_are_retained(self) -> None:
        _, effect = compare_outcomes(
            _outcome(),
            _outcome(harm_shift=0.7),
        )

        self.assertAlmostEqual(effect.mean_composite_harm_effect, -0.1)

    def test_player_harm_snapshot_domain_is_fail_closed(self) -> None:
        baseline = _outcome(player_count=2)
        invalid_cases = (
            (np.nan, "finite"),
            (np.inf, "finite"),
            (-0.01, r"\[0, 1\]"),
            (1.01, r"\[0, 1\]"),
        )
        for value, message in invalid_cases:
            with self.subTest(value=value, boundary="construction"):
                harm = baseline.player_harm.copy()
                harm[0, 0] = value
                with self.assertRaisesRegex(ValueError, message):
                    replace(baseline, player_harm=harm)

            with self.subTest(value=value, boundary="retained"):
                with self.assertRaises(ValueError):
                    baseline.player_harm[0, 0] = value

        with self.assertRaisesRegex(TypeError, "float64"):
            replace(
                baseline,
                player_harm=baseline.player_harm.astype(np.float32),
            )

    def test_extreme_multirow_harm_cannot_produce_an_infinite_mean(self) -> None:
        baseline = _outcome(player_count=2)
        extreme_harm = baseline.player_harm.copy()
        extreme_harm[:] = 1e308

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            replace(baseline, player_harm=extreme_harm)
        self.assertEqual(baseline.summary()["mean_composite_harm"], 0.0)

    def test_extreme_harm_weights_cannot_silently_collapse_an_effect(self) -> None:
        extreme = HarmWeights(*([1e308] * 7))
        with self.assertRaisesRegex(ValueError, "weight sum must be finite"):
            compare_outcomes(
                _outcome(harm_shift=0.7),
                _outcome(),
                weights=extreme,
            )

    def test_player_income_difference_is_an_explicit_negative_control(self) -> None:
        paired, _ = compare_outcomes(
            _outcome(income_shift=25),
            _outcome(),
        )

        np.testing.assert_array_equal(
            paired.player_income_negative_control_difference_cents,
            np.full(3, 25),
        )
        self.assertFalse(paired.player_income_negative_control_passed)

    def test_paired_outcomes_reject_tick_dtype_and_domain_mismatches(self) -> None:
        control = _outcome()
        player_order = np.array([1, 0, 2])
        reordered_players = replace(
            control,
            player_ids=control.player_ids[player_order],
            player_harm=control.player_harm[player_order],
            player_spend_cents=control.player_spend_cents[player_order],
            player_income_cents=control.player_income_cents[player_order],
            player_debt_cents=control.player_debt_cents[player_order],
        )
        two_firms = _outcome(firm_count=2)
        firm_order = np.array([1, 0])
        reordered_firms = replace(
            two_firms,
            firm_ids=two_firms.firm_ids[firm_order],
            firm_cash_cents=two_firms.firm_cash_cents[firm_order],
            firm_operating_margin_cents=(
                two_firms.firm_operating_margin_cents[firm_order]
            ),
            firm_safe_revenue_share=(
                two_firms.firm_safe_revenue_share[firm_order]
            ),
        )
        two_states = _outcome(state_count=2)
        state_order = np.array([1, 0])
        reordered_states = replace(
            two_states,
            jurisdiction_ids=two_states.jurisdiction_ids[state_order],
            state_subsidy_outlay_cents=(
                two_states.state_subsidy_outlay_cents[state_order]
            ),
        )
        mismatches = (
            ("tick", _outcome(tick=11), "same tick"),
            ("players", _outcome(player_count=2), "player_ids"),
            ("firms", _outcome(firm_count=2), "firm_ids"),
            ("states", _outcome(state_count=2), "jurisdiction_ids"),
            ("player_order", reordered_players, "ordered player_ids"),
            (
                "dtype",
                replace(
                    control,
                    player_spend_cents=control.player_spend_cents.astype(np.int32),
                ),
                "dtype int64",
            ),
        )
        for label, treated, message in mismatches:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    compare_outcomes(treated, control)

        with self.assertRaisesRegex(TypeError, "non-empty"):
            compare_outcomes(control, control, estimand="")

        int32 = replace(
            control,
            player_spend_cents=control.player_spend_cents.astype(np.int32),
        )
        with self.assertRaisesRegex(ValueError, "dtype int64"):
            compare_outcomes(int32, int32)

        invalid_share = replace(
            control,
            firm_safe_revenue_share=np.array([1.01], dtype=np.float64),
        )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            compare_outcomes(invalid_share, control)

        with self.assertRaisesRegex(ValueError, "ordered firm_ids"):
            compare_outcomes(reordered_firms, two_firms)
        with self.assertRaisesRegex(ValueError, "ordered jurisdiction_ids"):
            compare_outcomes(reordered_states, two_states)

    def test_paired_identity_contract_allows_unsorted_nonnegative_ids_only(self) -> None:
        unsorted = _outcome(player_count=3, firm_count=2, state_count=2)
        player_order = np.array([2, 0, 1])
        firm_order = np.array([1, 0])
        state_order = np.array([1, 0])
        unsorted = replace(
            unsorted,
            player_ids=unsorted.player_ids[player_order],
            player_harm=unsorted.player_harm[player_order],
            player_spend_cents=unsorted.player_spend_cents[player_order],
            player_income_cents=unsorted.player_income_cents[player_order],
            player_debt_cents=unsorted.player_debt_cents[player_order],
            firm_ids=unsorted.firm_ids[firm_order],
            firm_cash_cents=unsorted.firm_cash_cents[firm_order],
            firm_operating_margin_cents=(
                unsorted.firm_operating_margin_cents[firm_order]
            ),
            firm_safe_revenue_share=unsorted.firm_safe_revenue_share[firm_order],
            jurisdiction_ids=unsorted.jurisdiction_ids[state_order],
            state_subsidy_outlay_cents=(
                unsorted.state_subsidy_outlay_cents[state_order]
            ),
        )
        paired, _ = compare_outcomes(unsorted, unsorted)
        self.assertTrue(np.all(paired.player_harm_difference == 0.0))

        empty = _outcome(player_count=0, firm_count=0, state_count=0)
        paired, effect = compare_outcomes(empty, empty)
        self.assertEqual(paired.player_harm_difference.shape, (0, 7))
        self.assertEqual(effect.affected_player_share, 0.0)

        for name in ("player_ids", "firm_ids", "jurisdiction_ids"):
            with self.subTest(field=name):
                invalid = _outcome(player_count=2, firm_count=2, state_count=2)
                values = getattr(invalid, name).copy()
                values[0] = -1
                with self.assertRaisesRegex(ValueError, "non-negative"):
                    replace(invalid, **{name: values})
                with self.assertRaises(ValueError):
                    getattr(invalid, name)[0] = -1

    def test_paired_integer_difference_rejects_int64_wraparound(self) -> None:
        control = _outcome()
        boundaries = (
            (
                np.iinfo(np.int64).max,
                -1,
            ),
            (
                np.iinfo(np.int64).min,
                1,
            ),
        )
        fields = (
            "player_spend_cents",
            "player_income_cents",
            "player_debt_cents",
            "firm_operating_margin_cents",
            "firm_cash_cents",
            "state_subsidy_outlay_cents",
        )
        for field in fields:
            for treated_value, control_value in boundaries:
                with self.subTest(
                    field=field,
                    treated_value=treated_value,
                    control_value=control_value,
                ):
                    treated_values = getattr(control, field).copy()
                    control_values = getattr(control, field).copy()
                    treated_values[0] = treated_value
                    control_values[0] = control_value
                    treated = replace(control, **{field: treated_values})
                    comparison = replace(control, **{field: control_values})
                    with self.assertRaisesRegex(OverflowError, "exceeds int64"):
                        compare_outcomes(treated, comparison)

    def test_outcome_snapshot_owns_immutable_arrays_and_recorder_reuses_it(
        self,
    ) -> None:
        baseline = _outcome(player_count=3, firm_count=2, state_count=2)
        names = (
            "player_ids",
            "player_harm",
            "player_spend_cents",
            "player_income_cents",
            "player_debt_cents",
            "firm_ids",
            "firm_cash_cents",
            "firm_operating_margin_cents",
            "firm_safe_revenue_share",
            "jurisdiction_ids",
            "state_subsidy_outlay_cents",
        )
        sources = {name: getattr(baseline, name).copy() for name in names}
        snapshot = replace(baseline, **sources)

        for name in names:
            with self.subTest(field=name):
                source = sources[name]
                retained = getattr(snapshot, name)
                expected = retained.copy()
                self.assertFalse(np.shares_memory(retained, source))
                source.flat[0] = 1 if source.flat[0] == 0 else 0
                np.testing.assert_array_equal(retained, expected)
                with self.assertRaises(ValueError):
                    retained.flat[0] = retained.flat[0]
                with self.assertRaises(ValueError):
                    retained.setflags(write=True)

        recorder = OutcomeRecorder()
        recorder.record(snapshot)
        latest = recorder.latest
        assert latest is not None
        self.assertIs(latest, snapshot)

    def test_paired_outcome_owns_immutable_arrays(self) -> None:
        paired, effect = compare_outcomes(
            _outcome(
                harm_shift=0.7,
                spend_shift=50,
                debt_shift=7,
                cash_shift=500,
                margin_shift=125,
                subsidy_shift=40,
            ),
            _outcome(),
        )
        names = tuple(PairedOutcome.__dataclass_fields__)
        sources = {name: getattr(paired, name).copy() for name in names}
        retained = PairedOutcome(**sources)

        malformed = dict(sources)
        malformed["player_harm_difference"] = np.zeros((3, 6), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "seven harm dimensions"):
            PairedOutcome(**malformed)

        for name in names:
            with self.subTest(field=name):
                source = sources[name]
                values = getattr(retained, name)
                expected = values.copy()
                self.assertFalse(np.shares_memory(values, source))
                source.flat[0] = 0
                np.testing.assert_array_equal(values, expected)
                with self.assertRaises(ValueError):
                    values.flat[0] = values.flat[0]
                with self.assertRaises(ValueError):
                    values.setflags(write=True)

        self.assertEqual(effect.total_spend_effect_cents, 150)
        self.assertEqual(effect.total_debt_effect_cents, 21)

    def test_researcher_intervention_is_explicit(self) -> None:
        world = _MockWorld()
        NullIntervention().apply(world)
        self.assertEqual(world.calls, [])
        MechanismCap(
            MonetisationMechanism.RANDOM_REWARD,
            maximum=0.1,
            game_ids=(1, 2),
        ).apply(world)
        self.assertEqual(
            world.calls,
            [(MonetisationMechanism.RANDOM_REWARD, 0.1, (1, 2))],
        )

    def test_regulation_and_public_funding_compose_explicitly(self) -> None:
        world = _MockWorld()
        intervention = CompositeIntervention(
            (
                AuditRegime(interval_days=14, sensitivity=0.9),
                SubsidyRegime(
                    budget_cents_per_state=2_000_000,
                    design_safety_weight=0.7,
                ),
            )
        )
        intervention.apply(world)
        self.assertEqual(world.audit_calls[0]["interval_days"], 14)
        self.assertEqual(world.audit_calls[0]["sensitivity"], 0.9)
        self.assertEqual(
            world.subsidy_calls[0]["budget_cents_per_state"], 2_000_000
        )
        self.assertEqual(world.subsidy_calls[0]["design_safety_weight"], 0.7)


if __name__ == "__main__":
    unittest.main()
