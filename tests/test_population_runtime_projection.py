from __future__ import annotations

from dataclasses import fields, replace
import unittest

import numpy as np

from microtx_sim.agents.players import (
    ProjectedPopulationAssignment,
    ProjectedPopulationCellMetadata,
    ProjectedPopulationMetadata,
    projected_population_assignment_sha256,
    projected_population_plan_sha256,
)
from microtx_sim.consumers.population import (
    CountryProfile,
    PopulationProjectionCell,
    initialize_player_table,
    initialize_projected_player_table,
)
from microtx_sim.consumers.welfare import initialize_player_life
from microtx_sim.causal.batch import _cohort_digest
from microtx_sim.metrics.population_estimands import (
    PopulationEstimandValidationError,
    exact_population_weights_from_projected_players,
)
from microtx_sim.rng import CounterRNG


def _profiles() -> tuple[CountryProfile, CountryProfile]:
    return (
        CountryProfile(code="IT", adult_age=18),
        CountryProfile(code="SE", adult_age=18),
    )


def _cell(
    cell_id: str,
    *,
    jurisdiction_code: str = "IT",
    age_min_inclusive: int = 18,
    age_max_exclusive: int = 30,
    income_band: str = "runtime.low",
    income_min: int = 1_000,
    income_max: int = 2_000,
    household_type: str = "family",
    household_size: int = 2,
    gamer: bool = False,
    payer: bool = False,
    mass: tuple[int, int] = (1, 1),
) -> PopulationProjectionCell:
    return PopulationProjectionCell(
        cell_id=cell_id,
        jurisdiction_code=jurisdiction_code,
        age_min_inclusive=age_min_inclusive,
        age_max_exclusive=age_max_exclusive,
        monthly_disposable_income_band_id=income_band,
        monthly_disposable_income_min_cents=income_min,
        monthly_disposable_income_max_cents_exclusive=income_max,
        household_type=household_type,
        modeled_players_per_household=household_size,
        baseline_gamer=gamer,
        baseline_ever_payer=payer,
        global_mass=mass,
    )


def _projection_cells() -> tuple[PopulationProjectionCell, ...]:
    # Deliberately non-canonical input order: the initializer must canonicalize it.
    return (
        _cell(
            "se.adult",
            jurisdiction_code="SE",
            age_min_inclusive=20,
            age_max_exclusive=40,
            income_band="runtime.high",
            income_min=3_000,
            income_max=5_000,
            household_type="single",
            household_size=1,
            gamer=True,
            payer=True,
            mass=(1, 2),
        ),
        _cell(
            "it.minor",
            age_min_inclusive=12,
            age_max_exclusive=18,
            gamer=True,
            mass=(1, 4),
        ),
        _cell(
            "se.zero",
            jurisdiction_code="SE",
            age_min_inclusive=40,
            age_max_exclusive=50,
            income_band="runtime.unused",
            income_min=5_000,
            income_max=6_000,
            household_type="unused",
            household_size=1,
            mass=(0, 1),
        ),
        _cell("it.adult", payer=True, mass=(1, 4)),
    )


class RuntimePopulationProjectionTests(unittest.TestCase):
    def test_content_hashes_reject_polymorphic_snapshot_types(self) -> None:
        players = initialize_projected_player_table(
            20,
            _profiles(),
            CounterRNG(780),
            _projection_cells(),
            projection_id="strict-types.plan.v1",
        )
        assignment = players.projected_population
        assert assignment is not None

        class HiddenCell(ProjectedPopulationCellMetadata):
            __slots__ = ()

            def snapshot(self) -> dict[str, object]:
                return {"hidden": "constant"}

        base_cell = assignment.metadata.cells[0]
        hidden_cell = HiddenCell(
            **{
                descriptor.name: getattr(base_cell, descriptor.name)
                for descriptor in fields(ProjectedPopulationCellMetadata)
            }
        )
        with self.assertRaisesRegex(TypeError, "ProjectedPopulationCellMetadata"):
            projected_population_plan_sha256("hidden.plan", (hidden_cell,))

        class HiddenMetadata(ProjectedPopulationMetadata):
            __slots__ = ()

            def snapshot(self) -> dict[str, object]:
                return {"hidden": "constant"}

        hidden_metadata = HiddenMetadata(
            projection_id=assignment.metadata.projection_id,
            projection_sha256=assignment.metadata.projection_sha256,
            cells=assignment.metadata.cells,
        )
        with self.assertRaisesRegex(TypeError, "ProjectedPopulationMetadata"):
            projected_population_assignment_sha256(
                hidden_metadata,
                players.player_id,
                assignment.cell_index,
            )

        class HiddenProjectionCell(PopulationProjectionCell):
            __slots__ = ()

        base_projection_cell = _projection_cells()[0]
        hidden_projection_cell = HiddenProjectionCell(
            **{
                descriptor.name: getattr(base_projection_cell, descriptor.name)
                for descriptor in fields(PopulationProjectionCell)
            }
        )
        with self.assertRaisesRegex(TypeError, "PopulationProjectionCell"):
            initialize_projected_player_table(
                1,
                _profiles(),
                CounterRNG(780),
                (hidden_projection_cell,),
                projection_id="hidden.input",
            )

    def test_exact_projection_is_canonical_reproducible_and_consistent(self) -> None:
        cells = _projection_cells()
        first = initialize_projected_player_table(
            20,
            _profiles(),
            CounterRNG(781),
            cells,
            projection_id="fixture.plan.v1",
        )
        reordered = initialize_projected_player_table(
            20,
            _profiles(),
            CounterRNG(781),
            tuple(reversed(cells)),
            projection_id="fixture.plan.v1",
        )

        assignment = first.projected_population
        self.assertIsNotNone(assignment)
        assert assignment is not None
        self.assertFalse(assignment.cell_index.flags.writeable)
        self.assertEqual(
            tuple(cell.cell_id for cell in assignment.metadata.cells),
            ("it.adult", "it.minor", "se.adult", "se.zero"),
        )
        np.testing.assert_array_equal(
            np.bincount(assignment.cell_index, minlength=4),
            np.asarray([5, 5, 10, 0]),
        )
        self.assertEqual(
            tuple(cell.analysis_weight for cell in assignment.metadata.cells),
            ((1, 20), (1, 20), (1, 20), (0, 1)),
        )
        self.assertEqual(
            assignment.metadata.projection_sha256,
            projected_population_plan_sha256(
                assignment.metadata.projection_id,
                assignment.metadata.cells,
            ),
        )
        self.assertEqual(
            assignment.assignment_sha256,
            projected_population_assignment_sha256(
                assignment.metadata,
                first.player_id,
                assignment.cell_index,
            ),
        )
        self.assertEqual(
            assignment.assignment_sha256,
            reordered.projected_population.assignment_sha256,  # type: ignore[union-attr]
        )
        for name in (
            "age_years",
            "jurisdiction",
            "household_id",
            "monthly_disposable_income_cents",
            "household_liquidity_cents",
            "traits",
            "motive_weights",
        ):
            np.testing.assert_array_equal(getattr(first, name), getattr(reordered, name))

        # Baseline gaming and payer history are retained only in the sidecar.
        self.assertTrue(np.all(first.current_game == -1))
        self.assertTrue(
            any(cell.baseline_gamer for cell in assignment.metadata.cells)
        )
        self.assertTrue(
            any(cell.baseline_ever_payer for cell in assignment.metadata.cells)
        )

        for household in np.unique(first.household_id):
            positions = np.flatnonzero(first.household_id == household)
            household_cells = [
                assignment.metadata.cells[int(index)]
                for index in assignment.cell_index[positions]
            ]
            keys = {
                (
                    cell.jurisdiction_code,
                    cell.monthly_disposable_income_band_id,
                    cell.household_type,
                )
                for cell in household_cells
            }
            self.assertEqual(len(keys), 1)
            self.assertLessEqual(
                positions.size,
                household_cells[0].modeled_players_per_household,
            )
            self.assertEqual(
                np.unique(first.household_liquidity_cents[positions]).size,
                1,
            )

    def test_hamilton_uses_exact_mass_and_canonical_cell_id_tie_break(self) -> None:
        cells = tuple(
            _cell(cell_id, household_size=2, mass=(1, 3))
            for cell_id in ("c", "a", "b")
        )
        players = initialize_projected_player_table(
            4,
            _profiles(),
            CounterRNG(4),
            cells,
            projection_id="tie.plan",
        )
        assignment = players.projected_population
        assert assignment is not None

        self.assertEqual(
            tuple(cell.cell_id for cell in assignment.metadata.cells),
            ("a", "b", "c"),
        )
        np.testing.assert_array_equal(
            np.bincount(assignment.cell_index, minlength=3),
            np.asarray([2, 1, 1]),
        )
        self.assertEqual(
            tuple(cell.analysis_weight for cell in assignment.metadata.cells),
            ((1, 6), (1, 3), (1, 3)),
        )

    def test_positive_cells_must_all_be_represented(self) -> None:
        cells = (
            _cell("large", mass=(99, 100)),
            _cell("tiny-a", mass=(1, 200)),
            _cell("tiny-b", mass=(1, 200)),
        )

        with self.assertRaisesRegex(ValueError, "too small.*tiny-a.*tiny-b"):
            initialize_projected_player_table(
                3,
                _profiles(),
                CounterRNG(4),
                cells,
                projection_id="small.plan",
            )

    def test_arbitrary_precision_rational_metadata_is_not_narrowed(self) -> None:
        denominator = 2**70 + 1
        lower = 2**69
        cells = (
            _cell("a", household_size=1, mass=(lower, denominator)),
            _cell(
                "b",
                household_type="single",
                household_size=1,
                mass=(denominator - lower, denominator),
            ),
        )
        players = initialize_projected_player_table(
            2,
            _profiles(),
            CounterRNG(5),
            cells,
            projection_id="large-rational.plan",
        )
        assignment = players.projected_population
        assert assignment is not None

        self.assertGreater(assignment.metadata.cells[0].global_mass[1], 2**63)
        self.assertEqual(
            assignment.metadata.cells[0].analysis_weight,
            (lower, denominator),
        )

    def test_assignment_and_player_fields_are_cross_validated(self) -> None:
        players = initialize_projected_player_table(
            20,
            _profiles(),
            CounterRNG(781),
            _projection_cells(),
            projection_id="fixture.plan.v1",
        )
        assignment = players.projected_population
        assert assignment is not None
        with self.assertRaises(ValueError):
            assignment.cell_index[0] = 0

        bad_income = players.monthly_disposable_income_cents.copy()
        first_cell = assignment.cell_for_player_position(0)
        bad_income[0] = first_cell.monthly_disposable_income_max_cents_exclusive
        with self.assertRaisesRegex(ValueError, "outside its projected runtime interval"):
            replace(players, monthly_disposable_income_cents=bad_income)

        changed_index = assignment.cell_index.copy()
        first = 0
        second = int(np.flatnonzero(changed_index != changed_index[first])[0])
        changed_index[first], changed_index[second] = (
            changed_index[second],
            changed_index[first],
        )
        wrong_hash = ProjectedPopulationAssignment(
            metadata=assignment.metadata,
            cell_index=changed_index,
            assignment_sha256=assignment.assignment_sha256,
        )
        with self.assertRaisesRegex(ValueError, "hash does not match"):
            replace(players, projected_population=wrong_hash)
        with self.assertRaisesRegex(ValueError, "exact runtime projection plan"):
            replace(assignment.metadata, projection_sha256="0" * 64)

    def test_household_income_float_precision_boundary_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, r"at most 2\*\*53 cents"):
            _cell(
                "unsafe",
                income_min=2**52,
                income_max=2**52 + 2,
                household_size=3,
            )

        safe = _cell(
            "safe",
            income_min=2**52,
            income_max=2**52 + 1,
            household_size=2,
        )
        players = initialize_projected_player_table(
            2,
            _profiles(),
            CounterRNG(17),
            (safe,),
            projection_id="precision-boundary.plan",
        )
        self.assertTrue(
            np.all(players.monthly_disposable_income_cents == 2**52)
        )

    def test_legacy_factory_has_no_projection_sidecar(self) -> None:
        players = initialize_player_table(25, _profiles(), CounterRNG(9))

        self.assertIsNone(players.projected_population)

    def test_projected_cohort_digest_binds_the_nested_assignment(self) -> None:
        rng = CounterRNG(91)
        players = initialize_projected_player_table(
            20,
            _profiles(),
            rng,
            _projection_cells(),
            projection_id="fixture.plan.v1",
        )
        life = initialize_player_life(players, rng)
        assignment = players.projected_population
        assert assignment is not None
        changed_projection_id = "different-projection-plan"
        changed_metadata = replace(
            assignment.metadata,
            projection_id=changed_projection_id,
            projection_sha256=projected_population_plan_sha256(
                changed_projection_id,
                assignment.metadata.cells,
            ),
        )
        changed_assignment = ProjectedPopulationAssignment(
            metadata=changed_metadata,
            cell_index=assignment.cell_index,
            assignment_sha256=projected_population_assignment_sha256(
                changed_metadata,
                players.player_id,
                assignment.cell_index,
            ),
        )
        same_player_arrays_different_plan = replace(
            players,
            projected_population=changed_assignment,
        )

        self.assertNotEqual(
            _cohort_digest(players, life),
            _cohort_digest(same_player_arrays_different_plan, life),
        )

    def test_consumers_reject_a_mutated_nested_assignment(self) -> None:
        rng = CounterRNG(92)
        players = initialize_projected_player_table(
            20,
            _profiles(),
            rng,
            _projection_cells(),
            projection_id="fixture.plan.v1",
        )
        life = initialize_player_life(players, rng)
        assignment = players.projected_population
        assert assignment is not None
        first = 0
        second = int(
            np.flatnonzero(assignment.cell_index != assignment.cell_index[first])[0]
        )
        assignment.cell_index.flags.writeable = True
        assignment.cell_index[first], assignment.cell_index[second] = (
            assignment.cell_index[second],
            assignment.cell_index[first],
        )
        assignment.cell_index.flags.writeable = False

        with self.assertRaisesRegex(RuntimeError, "mutated after attestation"):
            _cohort_digest(players, life)
        with self.assertRaisesRegex(
            PopulationEstimandValidationError,
            "mutated after attestation",
        ):
            exact_population_weights_from_projected_players(players)

    def test_projected_sidecar_materializes_exact_ordered_estimand_weights(
        self,
    ) -> None:
        players = initialize_projected_player_table(
            20,
            _profiles(),
            CounterRNG(818),
            _projection_cells(),
            projection_id="fixture.plan.v1",
        )
        weights = exact_population_weights_from_projected_players(players)

        self.assertEqual(weights.player_ids, tuple(range(20)))
        self.assertEqual(weights.weight_sum, 1)
        with self.assertRaisesRegex(
            PopulationEstimandValidationError,
            "require a projected PlayerTable",
        ):
            exact_population_weights_from_projected_players(
                initialize_player_table(20, _profiles(), CounterRNG(818))
            )


if __name__ == "__main__":
    unittest.main()
