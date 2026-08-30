from __future__ import annotations

from dataclasses import fields, replace
import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import microtx_sim.simulation.day as simulation_day

from microtx_sim.causal import (
    BalanceMismatchKind,
    MechanismCap,
    NegativeControlValidationError,
    NullIntervention,
    PreTreatmentBalanceError,
    assess_pre_treatment_balance,
    run_paired_worlds,
)
from microtx_sim.config import (
    ConfigurationError,
    PopulationExecutionMode,
    PopulationProjectionConfig,
    StepHistoryRetention,
    load_config,
)
from microtx_sim.core.engine import SimulationEngine
from microtx_sim.core.ledger import Ledger
from microtx_sim.core.world import World
from microtx_sim.data.population_design import PopulationDesignVerificationError
from microtx_sim.data.profiles import ProfileValidationError
from microtx_sim.types import LedgerBackend, MonetisationMechanism, ProvenanceStatus


class _RecordingIntervention:
    name = "recording"

    def __init__(self) -> None:
        self.apply_count = 0

    def apply(self, world: World) -> None:
        self.apply_count += 1


class _IncomeMutationIntervention:
    name = "invalid-income-mutation"

    def apply(self, world: World) -> None:
        world.players.monthly_disposable_income_cents[0] += 1


class WorldIntegrationTests(unittest.TestCase):
    @staticmethod
    def _registered_population_selection() -> PopulationProjectionConfig:
        root = Path(__file__).resolve().parents[1]
        return PopulationProjectionConfig(
            mode=PopulationExecutionMode.PROJECTED_V1,
            design_bundle_path=(
                root / "data" / "provenance" / "population_design.toml"
            ),
            runtime_mapping_bundle_path=(
                root
                / "data"
                / "provenance"
                / "population_runtime_mapping.json"
            ),
            adapter_id="campaign.standardized.population.v2",
        )

    @staticmethod
    def _assert_array_dataclass_equal(
        case: unittest.TestCase,
        left: object,
        right: object,
    ) -> None:
        case.assertIs(type(left), type(right))
        for field in fields(left):  # type: ignore[arg-type]
            left_value = getattr(left, field.name)
            right_value = getattr(right, field.name)
            if type(left_value) is np.ndarray:
                case.assertEqual(left_value.dtype, right_value.dtype)
                case.assertEqual(left_value.shape, right_value.shape)
                case.assertEqual(left_value.tobytes(), right_value.tobytes())
            else:
                case.assertEqual(left_value, right_value)

    def test_run_configuration_sets_the_firm_research_cost_scale(self) -> None:
        config = load_config("configs/smoke.toml")
        lower = replace(
            config,
            information=replace(
                config.information,
                research_report_cost_cents=100_000,
            ),
        )
        higher = replace(
            config,
            information=replace(
                config.information,
                research_report_cost_cents=900_000,
            ),
        )
        low_world = World.create(lower, campaign=False)
        high_world = World.create(higher, campaign=False)
        self.assertTrue(
            all(
                high.research_cost_cents > low.research_cost_cents
                for low, high in zip(low_world.firms, high_world.firms, strict=True)
            )
        )

    def test_configured_population_failure_never_falls_back_to_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configured = replace(
                self.config,
                population=PopulationProjectionConfig(
                    mode=PopulationExecutionMode.PROJECTED_V1,
                    design_bundle_path=(root / "missing-design.toml").resolve(),
                    runtime_mapping_bundle_path=(
                        root / "missing-mapping.json"
                    ).resolve(),
                    adapter_id="world.population.missing",
                ),
            )
            with patch(
                "microtx_sim.core.world.initialize_player_table"
            ) as legacy_initializer:
                with self.assertRaises(PopulationDesignVerificationError):
                    World.create(configured)
            legacy_initializer.assert_not_called()

    def test_direct_world_construction_rejects_invalid_retention_early(self) -> None:
        invalid = replace(
            self.config,
            run=replace(
                self.config.run,
                step_history_retention="invalid",  # type: ignore[arg-type]
            ),
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "step_history_retention",
        ):
            World(
                config=invalid,
                profiles=object(),  # type: ignore[arg-type]
                rng=object(),  # type: ignore[arg-type]
                players=object(),  # type: ignore[arg-type]
                games=object(),  # type: ignore[arg-type]
                firms=(),
                states=(),
            )

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config("configs/smoke.toml")

    def test_three_cycle_smoke_connects_all_major_systems(self) -> None:
        world = World.create(self.config)
        result = SimulationEngine.run(world)

        self.assertEqual(result.cycles, 3)
        self.assertEqual(len(world.step_history), 3)
        self.assertEqual(len(result.final_outcome.player_spend_cents), 384)
        self.assertEqual(len(result.final_outcome.firm_cash_cents), 3)
        self.assertEqual(len(result.final_outcome.state_subsidy_outlay_cents), 4)
        self.assertEqual(
            world.player_system.config.household_peer_influence,
            self.config.behavior.household_peer_influence,
        )
        self.assertGreater(
            sum(len(step.audit_resolutions) for step in world.step_history), 0
        )
        self.assertGreater(len(world.ledger.entries), 0)
        world.ledger.assert_balanced()

    def test_sqlite_world_is_byte_exact_and_releases_temporary_store(self) -> None:
        sqlite_config = replace(
            self.config,
            run=replace(
                self.config.run,
                ledger_backend=LedgerBackend.SQLITE,
            ),
        )
        memory_world = World.create(self.config)
        sqlite_world = World.create(sqlite_config)
        sqlite_path = sqlite_world.ledger.path
        self.assertIsNotNone(sqlite_path)
        assert sqlite_path is not None
        self.assertTrue(sqlite_path.exists())

        memory_result = SimulationEngine.run(memory_world)
        sqlite_result = SimulationEngine.run(sqlite_world)

        self._assert_array_dataclass_equal(
            self,
            memory_result.final_outcome,
            sqlite_result.final_outcome,
        )
        self.assertEqual(memory_result.summary, sqlite_result.summary)
        self.assertEqual(memory_world.ledger.entries, sqlite_world.ledger.entries)
        self.assertEqual(
            memory_world.ledger.logical_sha256(),
            sqlite_world.ledger.logical_sha256(),
        )
        memory_world.close()
        sqlite_world.close()
        self.assertFalse(sqlite_path.exists())

    def test_world_leaves_caller_owned_persistent_ledger_open(self) -> None:
        sqlite_config = replace(
            self.config,
            run=replace(
                self.config.run,
                ledger_backend=LedgerBackend.SQLITE,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world.sqlite3"
            ledger = Ledger.create(path)
            world = World.create(sqlite_config, ledger=ledger)
            self.assertFalse(world.owns_ledger)
            world.step()
            world.close()
            self.assertTrue(world.closed)
            self.assertFalse(ledger.closed)
            self.assertGreater(ledger.entry_count(), 0)
            ledger.close()
            self.assertTrue(path.exists())

    def test_campaign_requires_explicit_persistent_ledger_before_profiles(self) -> None:
        candidate = replace(
            self.config,
            meta=replace(
                self.config.meta,
                provenance_status=ProvenanceStatus.CALIBRATED,
            ),
            run=replace(
                self.config.run,
                allow_synthetic=False,
                ledger_backend=LedgerBackend.SQLITE,
            ),
            population=self._registered_population_selection(),
        )
        with self.assertRaisesRegex(
            ConfigurationError,
            "explicit persistent ledger",
        ):
            World.create(candidate, campaign=True)

    def test_orchestrator_cannot_promote_a_temporary_ledger_to_campaign(self) -> None:
        candidate = replace(
            self.config,
            meta=replace(
                self.config.meta,
                provenance_status=ProvenanceStatus.CALIBRATED,
            ),
            run=replace(
                self.config.run,
                allow_synthetic=False,
                ledger_backend=LedgerBackend.SQLITE,
            ),
            population=self._registered_population_selection(),
        )
        ledger = Ledger.temporary()
        fake_world = SimpleNamespace(
            config=candidate,
            profiles=SimpleNamespace(validate_for_campaign=lambda: None),
            players=[object()],
            ledger=ledger,
        )
        with self.assertRaisesRegex(ConfigurationError, "non-temporary SQLite"):
            SimulationEngine.run(fake_world, cycles=1, campaign=True)
        ledger.close()

    def test_final_only_history_is_bounded_without_changing_results(self) -> None:
        bounded_config = replace(
            self.config,
            run=replace(
                self.config.run,
                step_history_retention=StepHistoryRetention.FINAL_ONLY,
            ),
        )
        full_world = World.create(self.config)
        bounded_world = World.create(bounded_config)

        full_run = SimulationEngine.run(full_world)
        bounded_run = SimulationEngine.run(bounded_world)

        self.assertEqual(len(full_world.step_history), 3)
        self.assertEqual(len(bounded_world.step_history), 1)
        self.assertEqual(
            bounded_world.step_history[0].tick,
            bounded_run.final_outcome.tick,
        )
        self.assertIs(
            bounded_world.step_history[0].outcome,
            bounded_run.final_outcome,
        )
        self.assertEqual(
            full_world.audit_count,
            sum(
                len(step.audit_resolutions)
                for step in full_world.step_history
            ),
        )
        self.assertEqual(bounded_world.audit_count, full_world.audit_count)
        self.assertEqual(full_world.audit_count, 16)
        self.assertEqual(full_world.ledger.entries, bounded_world.ledger.entries)
        self.assertEqual(full_run.summary, bounded_run.summary)
        self._assert_array_dataclass_equal(
            self,
            full_run.final_outcome,
            bounded_run.final_outcome,
        )

    def test_final_only_retains_latest_successful_step_across_calls(self) -> None:
        bounded_config = replace(
            self.config,
            run=replace(
                self.config.run,
                step_history_retention=StepHistoryRetention.FINAL_ONLY,
            ),
        )
        world = World.create(bounded_config)
        self.assertEqual(world.step_history, ())
        self.assertEqual(world.audit_count, 0)

        first = world.step()
        self.assertEqual(len(world.step_history), 1)
        self.assertIs(world.step_history[0], first)
        second = world.step()
        self.assertEqual(len(world.step_history), 1)
        self.assertIs(world.step_history[0], second)

        final_outcome = world.run(2)
        self.assertEqual(len(world.step_history), 1)
        self.assertEqual(world.step_history[0].tick, 3)
        self.assertIs(world.step_history[0].outcome, final_outcome)

    def test_final_only_releases_prior_player_payloads(self) -> None:
        bounded_config = replace(
            self.config,
            run=replace(
                self.config.run,
                step_history_retention=StepHistoryRetention.FINAL_ONLY,
            ),
        )
        world = World.create(bounded_config)
        prior = world.step()
        step_payload = weakref.ref(prior.player_result.player_spend_cents)
        outcome_payload = weakref.ref(prior.outcome.player_harm)
        del prior

        world.step()
        gc.collect()

        self.assertIsNone(step_payload())
        self.assertIsNone(outcome_payload())

    def test_individual_recorder_flag_is_orthogonal_to_step_history(self) -> None:
        final_outcomes = []
        for retention, expected_steps in (
            (StepHistoryRetention.FULL, 3),
            (StepHistoryRetention.FINAL_ONLY, 1),
        ):
            with self.subTest(retention=retention.value):
                config = replace(
                    self.config,
                    run=replace(
                        self.config.run,
                        step_history_retention=retention,
                    ),
                    causal=replace(
                        self.config.causal,
                        record_individual_outcomes=False,
                    ),
                )
                world = World.create(config)
                result = SimulationEngine.run(world)

                self.assertIsNone(world.recorder.latest)
                self.assertEqual(len(world.recorder.summaries), 3)
                self.assertEqual(len(world.step_history), expected_steps)
                self.assertIs(
                    world.step_history[-1].outcome,
                    result.final_outcome,
                )
                final_outcomes.append(result.final_outcome)

        self._assert_array_dataclass_equal(
            self,
            final_outcomes[0],
            final_outcomes[1],
        )

    def test_zero_cycle_rejections_do_not_fabricate_retained_state(self) -> None:
        bounded_config = replace(
            self.config,
            run=replace(
                self.config.run,
                step_history_retention=StepHistoryRetention.FINAL_ONLY,
            ),
        )
        world = World.create(bounded_config)

        with self.assertRaisesRegex(ValueError, "positive"):
            SimulationEngine.run(world, cycles=0)
        with self.assertRaisesRegex(ValueError, "positive"):
            world.run(0)

        self.assertEqual(world.tick, 0)
        self.assertEqual(world.step_history, ())
        self.assertEqual(world.audit_count, 0)
        self.assertEqual(world.recorder.summaries, ())
        self.assertIsNone(world.recorder.latest)

    def test_world_step_rejects_a_caller_owned_outer_ledger_transaction(
        self,
    ) -> None:
        world = World.create(self.config)
        with self.assertRaisesRegex(RuntimeError, "root ledger transaction"):
            with world.ledger.transaction():
                world.step()

        self.assertTrue(world.poisoned)
        self.assertEqual(world.tick, 0)
        self.assertEqual(world.step_history, ())
        self.assertEqual(world.audit_count, 0)
        self.assertEqual(world.recorder.summaries, ())
        self.assertEqual(world.ledger.entry_count(), 0)
        world.close()

        raw_world = World.create(self.config)
        raw_world.ledger._connection.execute("BEGIN IMMEDIATE")
        try:
            with self.assertRaisesRegex(RuntimeError, "root ledger transaction"):
                raw_world.step()
        finally:
            raw_world.ledger._connection.execute("ROLLBACK")
        self.assertTrue(raw_world.poisoned)
        self.assertEqual(raw_world.tick, 0)
        self.assertEqual(raw_world.ledger.entry_count(), 0)
        raw_world.close()

    def test_failed_late_step_does_not_replace_last_completed_history(self) -> None:
        bounded_config = replace(
            self.config,
            run=replace(
                self.config.run,
                step_history_retention=StepHistoryRetention.FINAL_ONLY,
            ),
            regulation=replace(
                self.config.regulation,
                audit_interval=1,
            ),
        )
        world = World.create(bounded_config)
        completed = world.step()
        audit_count = world.audit_count
        completed_ledger_count = world.ledger.entry_count()
        completed_ledger_hash = world.ledger.logical_sha256()

        with (
            patch(
                "microtx_sim.simulation.day.run_audits",
                wraps=simulation_day.run_audits,
            ) as audit_runner,
            patch(
                "microtx_sim.simulation.day.outcome_snapshot",
                side_effect=RuntimeError("late outcome failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "late outcome failure"):
                world.step()

        audit_runner.assert_called_once()
        self.assertEqual(world.tick, 1)
        self.assertEqual(len(world.step_history), 1)
        self.assertIs(world.step_history[0], completed)
        self.assertEqual(world.audit_count, audit_count)
        self.assertEqual(len(world.recorder.summaries), 1)
        self.assertIs(world.recorder.latest, completed.outcome)
        self.assertTrue(world.poisoned)
        self.assertEqual(world.ledger.entry_count(), completed_ledger_count)
        self.assertEqual(world.ledger.logical_sha256(), completed_ledger_hash)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            world.step()

    def test_retention_mode_does_not_change_paired_effects(self) -> None:
        bounded_config = replace(
            self.config,
            run=replace(
                self.config.run,
                step_history_retention=StepHistoryRetention.FINAL_ONLY,
            ),
        )
        intervention = MechanismCap(
            MonetisationMechanism.RANDOM_REWARD,
            maximum=0.0,
        )

        full = run_paired_worlds(
            self.config,
            treated=intervention,
            cycles=3,
        )
        bounded = run_paired_worlds(
            bounded_config,
            treated=intervention,
            cycles=3,
        )

        self.assertEqual(full.effect, bounded.effect)
        for field in fields(full.effect):
            full_value = getattr(full.effect, field.name)
            bounded_value = getattr(bounded.effect, field.name)
            if type(full_value) is float:
                self.assertEqual(full_value.hex(), bounded_value.hex())
            else:
                self.assertEqual(full_value, bounded_value)
        self._assert_array_dataclass_equal(
            self,
            full.paired_outcome,
            bounded.paired_outcome,
        )

    def test_null_paired_worlds_are_exactly_identical(self) -> None:
        result = run_paired_worlds(
            self.config,
            treated=NullIntervention(),
            control=NullIntervention(),
            cycles=3,
        )

        paired = result.paired_outcome
        self.assertTrue(result.pre_treatment_balance.balanced)
        self.assertGreater(len(result.pre_treatment_balance.checked_paths), 0)
        self.assertIn(
            "world.regulation_system.__attributes__",
            result.pre_treatment_balance.checked_paths,
        )
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
        self.assertEqual(result.effect.mean_composite_harm_effect, 0.0)
        self.assertEqual(result.effect.total_spend_effect_cents, 0)
        self.assertEqual(result.effect.total_debt_effect_cents, 0)
        self.assertEqual(result.effect.total_operating_margin_effect_cents, 0)
        self.assertEqual(result.effect.total_subsidy_effect_cents, 0)
        self.assertEqual(result.effect.affected_player_share, 0.0)

    def test_sqlite_paired_worlds_use_distinct_caller_owned_ledgers(self) -> None:
        sqlite_config = replace(
            self.config,
            run=replace(
                self.config.run,
                ledger_backend=LedgerBackend.SQLITE,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            treated_ledger = Ledger.create(root / "treated.sqlite3")
            control_ledger = Ledger.create(root / "control.sqlite3")
            result = run_paired_worlds(
                sqlite_config,
                treated=NullIntervention(),
                control=NullIntervention(),
                cycles=1,
                treated_ledger=treated_ledger,
                control_ledger=control_ledger,
            )
            self.assertTrue(result.pre_treatment_balance.balanced)
            self.assertFalse(treated_ledger.closed)
            self.assertFalse(control_ledger.closed)
            self.assertFalse(treated_ledger.shares_storage_with(control_ledger))
            self.assertEqual(
                treated_ledger.logical_sha256(),
                control_ledger.logical_sha256(),
            )
            treated_ledger.close()
            control_ledger.close()

    def test_paired_worlds_reject_shared_sqlite_storage(self) -> None:
        sqlite_config = replace(
            self.config,
            run=replace(
                self.config.run,
                ledger_backend=LedgerBackend.SQLITE,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            shared = Ledger.create(Path(directory) / "shared.sqlite3")
            with self.assertRaisesRegex(ValueError, "physically distinct"):
                run_paired_worlds(
                    sqlite_config,
                    treated=NullIntervention(),
                    cycles=1,
                    treated_ledger=shared,
                    control_ledger=shared,
                )
            shared.close()

    def test_pre_treatment_imbalance_fails_before_intervention_or_run(self) -> None:
        treated_world = World.create(self.config)
        control_world = World.create(self.config)
        control_world.players.age_years[0] += 1
        report = assess_pre_treatment_balance(treated_world, control_world)

        self.assertFalse(report.balanced)
        self.assertIn(
            "world.players.age_years",
            tuple(item.path for item in report.mismatches),
        )
        treated_intervention = _RecordingIntervention()
        control_intervention = _RecordingIntervention()
        with (
            patch(
                "microtx_sim.causal.paired_worlds.World.create",
                side_effect=(treated_world, control_world),
            ),
            patch(
                "microtx_sim.causal.paired_worlds.SimulationOrchestrator.run"
            ) as runner,
        ):
            with self.assertRaises(PreTreatmentBalanceError) as raised:
                run_paired_worlds(
                    self.config,
                    treated=treated_intervention,
                    control=control_intervention,
                    cycles=1,
                )

        self.assertEqual(raised.exception.report, report)
        self.assertEqual(treated_intervention.apply_count, 0)
        self.assertEqual(control_intervention.apply_count, 0)
        runner.assert_not_called()

    def test_pre_treatment_balance_detects_mutable_alias_topology(self) -> None:
        treated_world = World.create(self.config)
        control_world = World.create(self.config)
        treated_world.firms[1].state.collusive_trust = (
            treated_world.firms[0].state.collusive_trust
        )

        report = assess_pre_treatment_balance(treated_world, control_world)

        self.assertFalse(report.balanced)
        self.assertIn(
            BalanceMismatchKind.ALIAS,
            tuple(item.kind for item in report.mismatches),
        )

    def test_pre_treatment_balance_ignores_immutable_alias_topology(self) -> None:
        treated_world = World.create(self.config)
        control_world = World.create(self.config)
        treated_world.metadata_alias_a = self.config.meta
        treated_world.metadata_alias_b = treated_world.metadata_alias_a
        control_world.metadata_alias_a = replace(self.config.meta)
        control_world.metadata_alias_b = replace(self.config.meta)
        self.assertIs(
            treated_world.metadata_alias_a,
            treated_world.metadata_alias_b,
        )
        self.assertIsNot(
            control_world.metadata_alias_a,
            control_world.metadata_alias_b,
        )

        report = assess_pre_treatment_balance(treated_world, control_world)

        self.assertTrue(report.balanced)

    def test_pre_treatment_balance_handles_cycles_and_compares_their_topology(
        self,
    ) -> None:
        treated_world = World.create(self.config)
        control_world = World.create(self.config)
        treated_world.cycle = []
        treated_world.cycle.append(treated_world.cycle)
        control_world.cycle = []
        control_world.cycle.append(control_world.cycle)

        self.assertTrue(
            assess_pre_treatment_balance(treated_world, control_world).balanced
        )

        control_inner_cycle: list[object] = []
        control_inner_cycle.append(control_inner_cycle)
        control_world.cycle = [control_inner_cycle]
        report = assess_pre_treatment_balance(treated_world, control_world)
        mismatch_by_path = {item.path: item.kind for item in report.mismatches}
        self.assertEqual(
            mismatch_by_path["world.cycle[0]"],
            BalanceMismatchKind.ALIAS,
        )

    def test_pre_treatment_balance_detects_numpy_view_alias_topology(self) -> None:
        treated_world = World.create(self.config)
        control_world = World.create(self.config)
        treated_world.player_total_unsafe_spend_cents = (
            treated_world.player_total_spend_cents.view()
        )
        self.assertIsNot(
            treated_world.player_total_unsafe_spend_cents,
            treated_world.player_total_spend_cents,
        )
        self.assertTrue(
            np.shares_memory(
                treated_world.player_total_unsafe_spend_cents,
                treated_world.player_total_spend_cents,
            )
        )
        self.assertFalse(
            np.shares_memory(
                control_world.player_total_unsafe_spend_cents,
                control_world.player_total_spend_cents,
            )
        )

        report = assess_pre_treatment_balance(treated_world, control_world)

        mismatch_by_path = {item.path: item.kind for item in report.mismatches}
        self.assertEqual(
            mismatch_by_path["world.player_total_unsafe_spend_cents"],
            BalanceMismatchKind.ALIAS,
        )

    def test_pre_treatment_balance_detects_cross_branch_alias_at_other_path(
        self,
    ) -> None:
        treated_world = World.create(self.config)
        control_world = World.create(self.config)
        treated_world.player_total_spend_cents = control_world.player_interest_cents

        report = assess_pre_treatment_balance(treated_world, control_world)

        mismatch_by_path = {item.path: item.kind for item in report.mismatches}
        self.assertEqual(
            mismatch_by_path["world.player_total_spend_cents"],
            BalanceMismatchKind.SHARED_MUTABLE,
        )

    def test_pre_treatment_balance_rejects_shared_mutable_player_array(self) -> None:
        treated_world = World.create(self.config)
        control_world = World.create(self.config)
        object.__setattr__(
            control_world.players,
            "age_years",
            treated_world.players.age_years.view(),
        )
        self.assertIsNot(
            control_world.players.age_years,
            treated_world.players.age_years,
        )
        self.assertTrue(
            np.shares_memory(
                control_world.players.age_years,
                treated_world.players.age_years,
            )
        )

        report = assess_pre_treatment_balance(treated_world, control_world)

        self.assertFalse(report.balanced)
        self.assertIn(
            BalanceMismatchKind.SHARED_MUTABLE,
            tuple(item.kind for item in report.mismatches),
        )
        self.assertIn(
            "world.players.age_years",
            tuple(item.path for item in report.mismatches),
        )

    def test_shared_config_is_allowed_but_mutable_profile_templates_are_not(
        self,
    ) -> None:
        independently_loaded_treated = World.create(self.config)
        independently_loaded_control = World.create(self.config)
        self.assertIs(
            independently_loaded_treated.config,
            independently_loaded_control.config,
        )
        self.assertTrue(
            assess_pre_treatment_balance(
                independently_loaded_treated,
                independently_loaded_control,
            ).balanced
        )

        shared_profiles = independently_loaded_treated.profiles
        metadata_treated = World.create(self.config, profiles=shared_profiles)
        metadata_control = World.create(self.config, profiles=shared_profiles)
        self.assertIs(
            metadata_treated.profiles.state_agents[0],
            metadata_control.profiles.state_agents[0],
        )
        metadata_report = assess_pre_treatment_balance(
            metadata_treated,
            metadata_control,
        )
        mismatch_by_path = {
            item.path: item.kind for item in metadata_report.mismatches
        }
        self.assertEqual(
            mismatch_by_path["world.profiles.state_agents[0]"],
            BalanceMismatchKind.SHARED_MUTABLE,
        )

        paired = run_paired_worlds(
            self.config,
            treated=NullIntervention(),
            control=NullIntervention(),
            cycles=1,
            profiles=shared_profiles,
        )
        self.assertTrue(paired.pre_treatment_balance.balanced)

    def test_non_null_treatment_leaves_income_negative_control_unchanged(self) -> None:
        result = run_paired_worlds(
            self.config,
            treated=MechanismCap(
                MonetisationMechanism.RANDOM_REWARD,
                maximum=0.0,
            ),
            cycles=3,
        )

        self.assertTrue(
            result.paired_outcome.player_income_negative_control_passed
        )
        self.assertTrue(
            np.all(
                result.paired_outcome
                .player_income_negative_control_difference_cents
                == 0
            )
        )

    def test_run_rejects_nonzero_income_negative_control(self) -> None:
        with self.assertRaises(NegativeControlValidationError) as raised:
            run_paired_worlds(
                self.config,
                treated=_IncomeMutationIntervention(),
                cycles=1,
            )

        self.assertEqual(raised.exception.field, "player_income_cents")
        self.assertEqual(raised.exception.nonzero_count, 1)

    def test_base_scenario_cannot_silently_consume_synthetic_profiles(self) -> None:
        base = load_config("configs/base.toml")
        with self.assertRaisesRegex(ProfileValidationError, "allow_synthetic=false"):
            World.create(base)

    def test_mechanism_cap_persists_across_firm_decisions(self) -> None:
        world = World.create(self.config)
        mechanism = MonetisationMechanism.RANDOM_REWARD
        world.cap_mechanism(mechanism=mechanism, maximum=0.10, game_ids=None)
        world.games.monetisation[:, int(mechanism)] = 0.90
        world.step()
        self.assertTrue(
            np.all(world.games.monetisation[:, int(mechanism)] <= 0.10)
        )

    def test_one_day_firm_and_ranking_schedule_has_one_public_pipeline(self) -> None:
        config = replace(
            self.config,
            market=replace(
                self.config.market,
                ranking_interval=1,
                firm_decision_interval=1,
            ),
            information=replace(
                self.config.information,
                public_signal_noise=1.0,
                public_signal_delay=1,
            ),
        )
        world = World.create(config)
        result = SimulationEngine.run(world, cycles=3)
        self.assertEqual(result.cycles, 3)
        self.assertEqual(len(world.step_history), 3)

    def test_subsidies_wait_for_review_and_use_one_home_jurisdiction(self) -> None:
        world = World.create(self.config)
        first = world.step()
        self.assertEqual(first.subsidies_paid_cents, 0)
        world.run(3)
        self.assertEqual(sum(int(value) for value in world.firm_subsidy_cents), 3_600_000)
        self.assertEqual(
            sum(int(value) for value in world.state_subsidy_outlay_cents),
            3_600_000,
        )
        self.assertLessEqual(int(world.state_subsidy_outlay_cents.max()), 1_200_000)


if __name__ == "__main__":
    unittest.main()
