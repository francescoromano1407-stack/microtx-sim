from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_population_projection_adapter import _complete_adapter  # noqa: E402

from microtx_sim.analysis.sensitivity import (  # noqa: E402
    SensitivityCase,
    run_sensitivity_analysis,
)
from microtx_sim.causal.batch import (  # noqa: E402
    PolicyBatchSpec,
    _cohort_digest,
    run_policy_batch,
)
from microtx_sim.consumers.decision import DecisionParameters  # noqa: E402
from microtx_sim.consumers.population import CountryProfile  # noqa: E402
from microtx_sim.consumers.welfare import initialize_player_life  # noqa: E402
from microtx_sim.config import (  # noqa: E402
    PopulationExecutionMode,
    PopulationProjectionConfig,
    load_config,
)
from microtx_sim.core.world import World  # noqa: E402
from microtx_sim.data.population_execution import (  # noqa: E402
    POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION,
    PopulationExecutionValidationError,
    build_population_execution_lineage,
    build_population_seed_execution_record,
    population_policy_pretreatment_sha256,
)
from microtx_sim.data.population_projection import (  # noqa: E402
    initialize_population_projection,
)
from microtx_sim.metrics.population_estimands import (  # noqa: E402
    EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
    ExactPopulationWeights,
)
from microtx_sim.rng import CounterRNG  # noqa: E402
from microtx_sim.outputs.manifest import build_run_manifest  # noqa: E402
from microtx_sim.policy_config import load_policy_config  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class PopulationExecutionTests(unittest.TestCase):
    def test_seed_record_detaches_exact_weights_and_balance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            rng = CounterRNG(701)
            execution = initialize_population_projection(
                adapter,
                (CountryProfile(code="UK"),),
                rng,
                tick=11,
            )
            life = initialize_player_life(execution.players, rng)
            cohort_digest = _cohort_digest(execution.players, life)
            record = build_population_seed_execution_record(
                execution,
                seed=701,
                cohort_digest=cohort_digest,
                policy_days=31,
            )
            lineage = build_population_execution_lineage(adapter, (record,))

            self.assertEqual(
                lineage.schema_version,
                POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION,
            )
            self.assertEqual(record.exact_weights.weight_sum, 1)
            self.assertTrue(record.balance.exact_balance_passed)
            self.assertEqual(record.balance.assignment_sha256, record.assignment_sha256)
            self.assertEqual(lineage.record_for_seed(701), record)
            self.assertEqual(execution.initialization_seed, 701)
            self.assertEqual(execution.initialization_tick, 11)
            self.assertEqual(record.initialization_tick, 11)
            self.assertEqual(record.policy_days, 31)
            expected_budget = (
                execution.players.monthly_disposable_income_cents * 2
            ).astype(np.int64)
            self.assertEqual(
                record.policy_pretreatment_sha256,
                population_policy_pretreatment_sha256(
                    policy_days=31,
                    player_ids=execution.players.player_id,
                    is_minor=execution.players.is_minor,
                    age_years=execution.players.age_years,
                    jurisdiction=execution.players.jurisdiction,
                    baseline_vulnerability=(
                        execution.players.baseline_vulnerability
                    ),
                    disposable_budget_cents=expected_budget,
                ),
            )
            self.assertFalse(lineage.campaign_ready)
            self.assertFalse(lineage.public_population_comparability)
            self.assertEqual(lineage.manifest_payload(), lineage.snapshot())

            with self.assertRaisesRegex(
                PopulationExecutionValidationError,
                "seed_record_sha256",
            ):
                replace(record, cohort_digest="0" * 64)

            fractions = record.exact_weights.fractions
            left = 0
            right = next(
                index
                for index, value in enumerate(fractions)
                if value != fractions[left]
            )
            altered_fractions = list(fractions)
            altered_fractions[left], altered_fractions[right] = (
                altered_fractions[right],
                altered_fractions[left],
            )
            swapped = ExactPopulationWeights(
                schema_version=EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
                player_ids=record.exact_weights.player_ids,
                weight_numerators=tuple(
                    value.numerator
                    for value in altered_fractions
                ),
                weight_denominators=tuple(
                    value.denominator
                    for value in altered_fractions
                ),
            )
            object.__setattr__(record, "exact_weights", swapped)
            encoded = json.dumps(
                record.attestation_payload(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            object.__setattr__(record, "seed_record_sha256", sha256(encoded).hexdigest())
            with self.assertRaisesRegex(
                PopulationExecutionValidationError,
                "weights differ from assigned adapter cells",
            ):
                build_population_execution_lineage(adapter, (record,))

    def test_execution_binds_actual_seed_tick_and_policy_pretreatment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            profiles = (CountryProfile(code="UK"),)
            first = initialize_population_projection(
                adapter,
                profiles,
                CounterRNG(701),
                tick=3,
            )
            different_tick = initialize_population_projection(
                adapter,
                profiles,
                CounterRNG(701),
                tick=4,
            )
            different_seed = initialize_population_projection(
                adapter,
                profiles,
                CounterRNG(702),
                tick=3,
            )
            self.assertNotEqual(
                first.execution_sha256,
                different_tick.execution_sha256,
            )
            self.assertNotEqual(
                first.execution_sha256,
                different_seed.execution_sha256,
            )
            life = initialize_player_life(first.players, CounterRNG(701))
            with self.assertRaisesRegex(
                PopulationExecutionValidationError,
                "initialization seed",
            ):
                build_population_seed_execution_record(
                    first,
                    seed=702,
                    cohort_digest=_cohort_digest(first.players, life),
                    policy_days=31,
                )

            record = build_population_seed_execution_record(
                first,
                seed=701,
                cohort_digest=_cohort_digest(first.players, life),
                policy_days=31,
            )
            changed_age = first.players.age_years.copy()
            changed_age[0] = np.int16(changed_age[0] + 1)
            changed_digest = population_policy_pretreatment_sha256(
                policy_days=31,
                player_ids=first.players.player_id,
                is_minor=first.players.is_minor,
                age_years=changed_age,
                jurisdiction=first.players.jurisdiction,
                baseline_vulnerability=first.players.baseline_vulnerability,
                disposable_budget_cents=(
                    first.players.monthly_disposable_income_cents * 2
                ).astype(np.int64),
            )
            self.assertNotEqual(record.policy_pretreatment_sha256, changed_digest)

    def test_batch_and_sensitivity_share_one_population_execution_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            spec = PolicyBatchSpec(
                seeds=(17,),
                days=1,
                player_count=12,
                decision_parameters=DecisionParameters(step_minutes=240),
            )
            profiles = (CountryProfile(code="UK"),)
            batch = run_policy_batch(
                spec,
                country_profiles=profiles,
                population_adapter=adapter,
            )
            sensitivity = run_sensitivity_analysis(
                spec,
                cases=(
                    SensitivityCase(
                        "paid_random_rewards",
                        (0.0, 0.7),
                        expected_direction="increasing",
                    ),
                ),
                country_profiles=profiles,
                population_adapter=adapter,
            )

            self.assertIsNotNone(batch.population_execution_lineage)
            self.assertIsNotNone(sensitivity.population_execution_lineage)
            assert batch.population_execution_lineage is not None
            assert sensitivity.population_execution_lineage is not None
            self.assertEqual(
                batch.population_execution_lineage.manifest_payload(),
                sensitivity.population_execution_lineage.manifest_payload(),
            )
            self.assertEqual(batch.run_input_sha256(), sensitivity.run_input_sha256())
            self.assertEqual(
                batch.run_input_snapshot()["run_input_schema_version"],
                "2.0",
            )
            self.assertIn(
                "population_execution_input",
                batch.run_input_snapshot(),
            )
            expected_ids = np.asarray(
                batch.population_execution_lineage.seed_records[
                    0
                ].exact_weights.player_ids,
                dtype=np.int64,
            )
            for record in batch.records:
                self.assertTrue(np.array_equal(record.result.player_ids, expected_ids))

    def test_batch_rejects_relabelled_projected_pretreatment_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            spec = PolicyBatchSpec(
                seeds=(19,),
                days=1,
                player_count=12,
                decision_parameters=DecisionParameters(step_minutes=240),
            )
            batch = run_policy_batch(
                spec,
                country_profiles=(CountryProfile(code="UK"),),
                population_adapter=adapter,
            )
            altered_records = tuple(
                replace(
                    record,
                    result=replace(
                        record.result,
                        age_years=np.full_like(record.result.age_years, 50),
                        is_minor=np.zeros_like(record.result.is_minor),
                    ),
                )
                for record in batch.records
            )

            with self.assertRaisesRegex(
                ValueError,
                "pre-treatment fields do not match",
            ):
                replace(batch, records=altered_records)

    def test_legacy_run_input_shape_remains_unversioned(self) -> None:
        spec = PolicyBatchSpec(
            seeds=(23,),
            days=0,
            player_count=2,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        batch = run_policy_batch(
            spec,
            country_profiles=(CountryProfile(code="XX"),),
        )

        self.assertIsNone(batch.population_execution_lineage)
        self.assertEqual(
            set(batch.run_input_snapshot()),
            {
                "batch_spec",
                "model_inputs",
                "profile_input_fingerprint_sha256",
            },
        )

    def test_projected_batch_rejects_adapter_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            spec = PolicyBatchSpec(
                seeds=(1,),
                days=0,
                player_count=11,
            )
            with self.assertRaisesRegex(ValueError, "player count"):
                run_policy_batch(
                    spec,
                    country_profiles=(CountryProfile(code="UK"),),
                    population_adapter=adapter,
                )

    def test_direct_world_constructor_rejects_different_configured_adapter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            base = load_config(ROOT / "configs" / "smoke.toml")
            run_config = replace(
                base,
                run=replace(base.run, player_count=12),
            )
            world = World.create(run_config)
            try:
                execution = initialize_population_projection(
                    adapter,
                    (CountryProfile(code="UK"),),
                    CounterRNG(run_config.run.seed),
                )
                mismatched = replace(
                    run_config,
                    population=PopulationProjectionConfig(
                        mode=PopulationExecutionMode.PROJECTED_V1,
                        design_bundle_path=(
                            adapter.verification.bundle.bundle_path
                        ),
                        runtime_mapping_bundle_path=(
                            adapter.mapping_bundle.mapping_path
                        ),
                        adapter_id="different.adapter",
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "does not match the configured population",
                ):
                    World(
                        config=mismatched,
                        profiles=world.profiles,
                        rng=world.rng,
                        players=execution.players,
                        games=world.games,
                        firms=world.firms,
                        states=world.states,
                        population_projection_execution=execution,
                    )
            finally:
                world.close()

    def test_manifest_publishes_separate_execution_without_promoting_readiness(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(root)
            base = load_policy_config(ROOT / "configs" / "policy_prototype.toml")
            spec = replace(
                base.batch,
                seeds=(29,),
                days=0,
                player_count=12,
            )
            config = replace(
                base,
                batch=spec,
                population=PopulationProjectionConfig(
                    mode=PopulationExecutionMode.PROJECTED_V1,
                    design_bundle_path=adapter.verification.bundle.bundle_path,
                    runtime_mapping_bundle_path=adapter.mapping_bundle.mapping_path,
                    adapter_id=adapter.adapter_id,
                ),
            )
            batch = run_policy_batch(
                spec,
                country_profiles=(CountryProfile(code="UK"),),
                harm_parameters=config.harm_parameters,
                harm_weights=config.harm_weights,
                opportunity_valuation=config.opportunity_valuation,
                producer_assumptions=config.producer_assumptions,
                epgc_policy=config.epgc_policy,
                population_adapter=adapter,
            )
            manifest = build_run_manifest(
                config,
                batch,
                config_path=ROOT / "configs" / "policy_prototype.toml",
                repository_root=ROOT,
                created_utc="2026-08-29T00:00:00+00:00",
            )

            self.assertEqual(
                manifest["population_execution"],
                batch.population_execution_lineage.manifest_payload(),
            )
            self.assertFalse(
                manifest["population_execution"]["campaign_ready"]
            )
            self.assertFalse(
                manifest["population_readiness"]["manifest_gate"][
                    "public_population_comparability"
                ]
            )
            with self.assertRaisesRegex(
                ValueError,
                "population selection",
            ):
                build_run_manifest(
                    replace(config, population=None),
                    batch,
                    config_path=ROOT / "configs" / "policy_prototype.toml",
                    repository_root=ROOT,
                )


if __name__ == "__main__":
    unittest.main()
