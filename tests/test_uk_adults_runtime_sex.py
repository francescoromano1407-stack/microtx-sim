from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_population_projection_adapter import _complete_adapter  # noqa: E402

from microtx_sim.agents.players import (  # noqa: E402
    SOURCE_RECORDED_SEX_DTYPE,
    SOURCE_RECORDED_SEX_FEMALE,
    SOURCE_RECORDED_SEX_MALE,
    SOURCE_RECORDED_SEX_UNAVAILABLE,
    projected_population_assignment_sha256,
    require_treatment_eligible_player_table,
    source_recorded_sex_sha256,
)
from microtx_sim.consumers.logic import step_player_dynamics  # noqa: E402
from microtx_sim.consumers.population import CountryProfile  # noqa: E402
from microtx_sim.core.world import World  # noqa: E402
from microtx_sim.data.calibration import (  # noqa: E402
    EstimandRole,
    PopulationWeight,
    UKAdults2024CalibrationBundle,
)
from microtx_sim.data.population_execution import (  # noqa: E402
    PopulationExecutionValidationError,
    build_population_seed_execution_record,
)
from microtx_sim.data.population_projection import (  # noqa: E402
    POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V3,
    PopulationProjectionValidationError,
    bind_population_projection_source_recorded_sex,
    initialize_population_projection,
    population_projection_execution_sha256,
    require_treatment_eligible_population_projection,
    verify_population_projection_execution,
)
from microtx_sim.data.uk_adults_runtime import (  # noqa: E402
    UK_ADULTS_2024_SEX_ASSIGNMENT_METHOD,
    bind_uk_adults_2024_source_recorded_sex,
    uk_adults_2024_population_weights_sha256,
    verify_uk_adults_2024_source_recorded_sex,
)
from microtx_sim.rng import CounterRNG  # noqa: E402
from microtx_sim.simulation.orchestrator import advance_cycles  # noqa: E402
from microtx_sim.simulation.policy_day import advance_policy_day  # noqa: E402
from microtx_sim.simulation.policy_orchestrator import (  # noqa: E402
    run_policy_scenario,
)


_ONS_COUNTS = (
    ("18-24", 18, 24, "FEMALE", 2_821_237),
    ("18-24", 18, 24, "MALE", 2_970_284),
    ("25-34", 25, 34, "FEMALE", 4_754_911),
    ("25-34", 25, 34, "MALE", 4_590_686),
    ("35-44", 35, 44, "FEMALE", 4_805_400),
    ("35-44", 35, 44, "MALE", 4_499_492),
    ("45-54", 45, 54, "FEMALE", 4_343_516),
    ("45-54", 45, 54, "MALE", 4_169_961),
    ("55-64", 55, 64, "FEMALE", 4_554_117),
    ("55-64", 55, 64, "MALE", 4_350_983),
)


def _bundle(path: Path) -> UKAdults2024CalibrationBundle:
    total = sum(row[4] for row in _ONS_COUNTS)
    weights = tuple(
        PopulationWeight(
            age_band=age_band,
            age_min_inclusive=lower,
            age_max_inclusive=upper,
            sex=sex,
            population_count=count,
            adult_population_weight=Decimal(count) / Decimal(total),
            source_id="ons_population_mid_2024",
            estimand_role=EstimandRole.CALIBRATION,
        )
        for age_band, lower, upper, sex, count in _ONS_COUNTS
    )
    return UKAdults2024CalibrationBundle(
        schema_version=1,
        bundle_id="uk-adults-2024-v1",
        status="PARTIAL",
        targets=(),
        population_weights=weights,
        sources=(),
        blockers=("campaign remains blocked",),
        bundle_path=path,
        bundle_sha256=sha256(b"strict UK adult calibration fixture").hexdigest(),
    )


def _binary_hamilton_female_count(
    sample_count: int,
    female_population: int,
    male_population: int,
) -> int:
    total = female_population + male_population
    female_quota = Fraction(sample_count * female_population, total)
    male_quota = Fraction(sample_count * male_population, total)
    female = female_quota.numerator // female_quota.denominator
    male = male_quota.numerator // male_quota.denominator
    if sample_count - female - male:
        if female_quota - female >= male_quota - male:
            female += 1
    return female


class UKAdultsRuntimeSexTests(unittest.TestCase):
    def _execution(self, root: Path):
        adapter = _complete_adapter(root)[-1]
        return initialize_population_projection(
            adapter,
            (CountryProfile(code="UK"),),
            CounterRNG(903),
        )

    def test_binding_is_scoped_deterministic_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = _bundle(root / "calibration_bundle.json")
            execution = self._execution(root)
            legacy_assignment = execution.players.projected_population
            assert legacy_assignment is not None
            self.assertIsNone(execution.players.sex)
            self.assertIsNone(legacy_assignment.sex_binding)
            self.assertNotIn("sex_sha256", execution.attestation_payload())
            self.assertEqual(
                legacy_assignment.assignment_sha256,
                projected_population_assignment_sha256(
                    legacy_assignment.metadata,
                    execution.players.player_id,
                    legacy_assignment.cell_index,
                ),
            )

            with patch(
                "microtx_sim.data.uk_adults_runtime."
                "load_uk_adults_2024_calibration_bundle",
                return_value=bundle,
            ):
                first = bind_uk_adults_2024_source_recorded_sex(
                    execution,
                    bundle,
                )
                independent_root = root / "independent"
                independent_root.mkdir()
                independent_execution = self._execution(independent_root)
                second = bind_uk_adults_2024_source_recorded_sex(
                    independent_execution,
                    bundle,
                )

            np.testing.assert_array_equal(first.players.sex, second.players.sex)
            self.assertEqual(first.assignment_sha256, second.assignment_sha256)
            self.assertEqual(first.execution_sha256, second.execution_sha256)
            self.assertNotEqual(first.assignment_sha256, execution.assignment_sha256)
            self.assertEqual(
                first.runtime_projection_sha256,
                execution.runtime_projection_sha256,
            )
            self.assertFalse(first.players.sex.flags.writeable)  # type: ignore[union-attr]

            uk_index = first.players.jurisdiction_codes.index("UK")
            selected = (
                (first.players.jurisdiction == uk_index)
                & (first.players.age_years >= 18)
                & (first.players.age_years <= 64)
            )
            assert first.players.sex is not None
            self.assertTrue(
                np.all(
                    np.isin(
                        first.players.sex[selected],
                        (SOURCE_RECORDED_SEX_FEMALE, SOURCE_RECORDED_SEX_MALE),
                    )
                )
            )
            self.assertTrue(
                np.all(
                    first.players.sex[~selected]
                    == SOURCE_RECORDED_SEX_UNAVAILABLE
                )
            )
            counts_by_band = {
                age_band: {
                    sex: count
                    for current_band, _, _, sex, count in _ONS_COUNTS
                    if current_band == age_band
                }
                for age_band in {row[0] for row in _ONS_COUNTS}
            }
            for age_band, lower, upper, _, _ in _ONS_COUNTS[::2]:
                band = selected & (first.players.age_years >= lower) & (
                    first.players.age_years <= upper
                )
                expected_female = _binary_hamilton_female_count(
                    int(np.count_nonzero(band)),
                    counts_by_band[age_band][SOURCE_RECORDED_SEX_FEMALE],
                    counts_by_band[age_band][SOURCE_RECORDED_SEX_MALE],
                )
                self.assertEqual(
                    int(
                        np.count_nonzero(
                            first.players.sex[band]
                            == SOURCE_RECORDED_SEX_FEMALE
                        )
                    ),
                    expected_female,
                )

            assignment = first.players.projected_population
            assert assignment is not None and assignment.sex_binding is not None
            binding = assignment.sex_binding
            self.assertEqual(
                binding.assignment_method,
                UK_ADULTS_2024_SEX_ASSIGNMENT_METHOD,
            )
            self.assertEqual(
                binding.population_weights_sha256,
                uk_adults_2024_population_weights_sha256(bundle),
            )
            self.assertEqual(
                binding.sex_sha256,
                source_recorded_sex_sha256(first.players.sex),
            )
            self.assertEqual(
                assignment.assignment_sha256,
                projected_population_assignment_sha256(
                    assignment.metadata,
                    first.players.player_id,
                    assignment.cell_index,
                    age_years=first.players.age_years,
                    jurisdiction=first.players.jurisdiction,
                    sex=first.players.sex,
                    sex_binding=binding,
                ),
            )
            payload = first.attestation_payload()
            self.assertEqual(
                payload["schema_version"],
                POPULATION_PROJECTION_EXECUTION_SCHEMA_VERSION_V3,
            )
            self.assertEqual(payload["sex_sha256"], binding.sex_sha256)
            self.assertEqual(
                payload["source_recorded_sex"],
                binding.snapshot(),
            )
            self.assertEqual(
                first.execution_sha256,
                population_projection_execution_sha256(
                    first.adapter,
                    initialization_seed=first.initialization_seed,
                    initialization_tick=first.initialization_tick,
                    runtime_projection_sha256=first.runtime_projection_sha256,
                    assignment_sha256=first.assignment_sha256,
                    ordered_player_ids_sha256=first.ordered_player_ids_sha256,
                    sex_binding=binding,
                ),
            )
            self.assertIs(verify_population_projection_execution(first), first)

    def test_mutation_and_out_of_scope_imputation_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = _bundle(root / "calibration_bundle.json")
            execution = self._execution(root)
            with patch(
                "microtx_sim.data.uk_adults_runtime."
                "load_uk_adults_2024_calibration_bundle",
                return_value=bundle,
            ):
                bound = bind_uk_adults_2024_source_recorded_sex(
                    execution,
                    bundle,
                )
            assignment = bound.players.projected_population
            assert assignment is not None and assignment.sex_binding is not None
            assert bound.players.sex is not None

            mutated = bound.players.sex.copy()
            selected = np.flatnonzero(mutated != SOURCE_RECORDED_SEX_UNAVAILABLE)
            mutated[selected[0]] = (
                SOURCE_RECORDED_SEX_MALE
                if mutated[selected[0]] == SOURCE_RECORDED_SEX_FEMALE
                else SOURCE_RECORDED_SEX_FEMALE
            )
            with self.assertRaisesRegex(ValueError, "sex_sha256"):
                replace(bound.players, sex=mutated)

            out_of_scope = np.flatnonzero(
                bound.players.sex == SOURCE_RECORDED_SEX_UNAVAILABLE
            )
            self.assertTrue(out_of_scope.size)
            invalid = bound.players.sex.copy()
            invalid[out_of_scope[0]] = SOURCE_RECORDED_SEX_FEMALE
            invalid_binding = replace(
                assignment.sex_binding,
                sex_sha256=source_recorded_sex_sha256(invalid),
            )
            with self.assertRaisesRegex(
                ValueError,
                "empty outside its declared source scope",
            ):
                bind_population_projection_source_recorded_sex(
                    execution,
                    invalid,
                    invalid_binding,
                )

            with self.assertRaisesRegex(
                PopulationProjectionValidationError,
                "already has",
            ):
                bind_population_projection_source_recorded_sex(
                    bound,
                    bound.players.sex,
                    assignment.sex_binding,
                )

            with self.assertRaisesRegex(
                PopulationExecutionValidationError,
                "point-zero execution only",
            ):
                build_population_seed_execution_record(
                    bound,
                    seed=bound.initialization_seed,
                    cohort_digest="0" * 64,
                    policy_days=0,
                )

            bound.players.sex.flags.writeable = True
            bound.players.sex[selected[0]] = mutated[selected[0]]
            bound.players.sex.flags.writeable = False
            with self.assertRaisesRegex(
                PopulationProjectionValidationError,
                "runtime assignment values do not verify",
            ):
                verify_population_projection_execution(bound)

    def test_age_band_input_mutation_fails_generic_execution_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = _bundle(root / "calibration_bundle.json")
            with patch(
                "microtx_sim.data.uk_adults_runtime."
                "load_uk_adults_2024_calibration_bundle",
                return_value=bundle,
            ):
                bound = bind_uk_adults_2024_source_recorded_sex(
                    self._execution(root),
                    bundle,
                )
            assignment = bound.players.projected_population
            assert assignment is not None
            candidate = None
            for position in np.flatnonzero(
                (bound.players.age_years >= 25)
                & (bound.players.age_years <= 34)
            ):
                cell = assignment.metadata.cells[
                    int(assignment.cell_index[int(position)])
                ]
                if cell.age_min_inclusive <= 24 < cell.age_max_exclusive:
                    candidate = int(position)
                    break
            self.assertIsNotNone(candidate)
            assert candidate is not None
            bound.players.age_years[candidate] = 24

            with self.assertRaisesRegex(
                PopulationProjectionValidationError,
                "derivation inputs|runtime assignment values",
            ):
                verify_population_projection_execution(bound)

    def test_canonical_verifier_rejects_self_consistent_forged_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = _bundle(root / "calibration_bundle.json")
            execution = self._execution(root)
            with patch(
                "microtx_sim.data.uk_adults_runtime."
                "load_uk_adults_2024_calibration_bundle",
                return_value=bundle,
            ):
                canonical = bind_uk_adults_2024_source_recorded_sex(
                    execution,
                    bundle,
                )
            assignment = canonical.players.projected_population
            assert assignment is not None and assignment.sex_binding is not None
            uk_index = execution.players.jurisdiction_codes.index("UK")
            selected = (
                (execution.players.jurisdiction == uk_index)
                & (execution.players.age_years >= 18)
                & (execution.players.age_years <= 64)
            )
            forged_sex = np.full(
                len(execution.players),
                SOURCE_RECORDED_SEX_UNAVAILABLE,
                dtype=SOURCE_RECORDED_SEX_DTYPE,
            )
            forged_sex[selected] = SOURCE_RECORDED_SEX_FEMALE
            forged_binding = replace(
                assignment.sex_binding,
                population_weights_sha256="0" * 64,
                assignment_method="forged-self-consistent-allocation-v1",
                sex_sha256=source_recorded_sex_sha256(forged_sex),
            )
            forged = bind_population_projection_source_recorded_sex(
                execution,
                forged_sex,
                forged_binding,
            )
            self.assertIs(verify_population_projection_execution(forged), forged)
            with patch(
                "microtx_sim.data.uk_adults_runtime."
                "load_uk_adults_2024_calibration_bundle",
                return_value=bundle,
            ):
                with self.assertRaisesRegex(
                    PopulationProjectionValidationError,
                    "canonical",
                ):
                    verify_uk_adults_2024_source_recorded_sex(
                        forged,
                        bundle,
                    )

    def test_point_zero_binding_is_rejected_at_treatment_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = _bundle(root / "calibration_bundle.json")
            with patch(
                "microtx_sim.data.uk_adults_runtime."
                "load_uk_adults_2024_calibration_bundle",
                return_value=bundle,
            ):
                bound = bind_uk_adults_2024_source_recorded_sex(
                    self._execution(root),
                    bundle,
                )

            for operation in (
                lambda: require_treatment_eligible_player_table(
                    bound.players,
                    operation="test treatment",
                ),
                lambda: require_treatment_eligible_population_projection(
                    bound,
                    operation="test treatment",
                ),
                lambda: run_policy_scenario(
                    bound.players,
                    None,  # type: ignore[arg-type]
                    None,  # type: ignore[arg-type]
                    seed=1,
                    days=0,
                ),
                lambda: advance_policy_day(
                    bound.players,
                    None,  # type: ignore[arg-type]
                    None,  # type: ignore[arg-type]
                    None,  # type: ignore[arg-type]
                    day=0,
                ),
                lambda: step_player_dynamics(
                    bound.players,
                    None,  # type: ignore[arg-type]
                    None,  # type: ignore[arg-type]
                    None,  # type: ignore[arg-type]
                    tick=0,
                ),
                lambda: advance_cycles(
                    SimpleNamespace(
                        players=bound.players,
                        population_projection_execution=bound,
                        step=lambda: None,
                    ),
                    1,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "point-zero"):
                    operation()

            world = object.__new__(World)
            world.players = bound.players
            world.population_projection_execution = bound
            with self.assertRaisesRegex(ValueError, "point-zero"):
                world._require_runnable()

    def test_exact_dtype_and_binary_source_categories_are_required(self) -> None:
        values = np.asarray(["FEMALE", "MALE"], dtype=SOURCE_RECORDED_SEX_DTYPE)
        self.assertEqual(len(source_recorded_sex_sha256(values)), 64)
        with self.assertRaisesRegex(TypeError, "<U6"):
            source_recorded_sex_sha256(values.astype(object))
        invalid = np.asarray(["FEMALE", "OTHER"], dtype=SOURCE_RECORDED_SEX_DTYPE)
        with self.assertRaisesRegex(ValueError, "FEMALE, MALE"):
            source_recorded_sex_sha256(invalid)


if __name__ == "__main__":
    unittest.main()
