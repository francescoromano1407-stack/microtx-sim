from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from microtx_sim.consumers.population import CountryProfile
from microtx_sim.data.population_design import (
    apportion_population_hamilton,
    build_population_calibration_target,
)
from microtx_sim.data.population_projection import (
    RUNTIME_INCOME_CONCEPT,
    SOURCE_INCOME_CONCEPT,
    PopulationProjectionAdapterCell,
    PopulationProjectionExecution,
    build_population_projection_adapter,
    initialize_population_projection,
    load_population_runtime_mapping_bundle,
)
from microtx_sim.metrics.population_balance import (
    POPULATION_BALANCE_SCHEMA_VERSION,
    PopulationBalanceValidationError,
    build_population_balance_artifact,
    validate_population_balance_snapshot,
)
from microtx_sim.rng import CounterRNG
from tests.test_population_design import _write_complete_fixture


def _mapping_payload(verification: object) -> dict[str, object]:
    bundle = verification.bundle
    entries: list[dict[str, object]] = []
    for income_index, income_band in enumerate(bundle.income_bands):
        for household in bundle.household_types:
            entries.append(
                {
                    "jurisdiction_code": income_band.jurisdiction_code,
                    "source_household_income_band_id": income_band.income_band_id,
                    "source_household_income_definition": income_band.definition,
                    "source_household_income_currency": income_band.currency,
                    "source_household_income_period": income_band.period,
                    "source_household_income_lower_unbounded": (
                        income_band.lower_unbounded
                    ),
                    "source_household_income_lower_bound": [
                        income_band.lower_bound_numerator,
                        income_band.lower_bound_denominator,
                    ],
                    "source_household_income_upper_unbounded": (
                        income_band.upper_unbounded
                    ),
                    "source_household_income_upper_bound": [
                        income_band.upper_bound_numerator,
                        income_band.upper_bound_denominator,
                    ],
                    "source_household_type_id": household.household_type_id,
                    "source_household_type_definition": household.definition,
                    "runtime_personal_monthly_disposable_income_band_id": (
                        f"runtime.{income_band.income_band_id}"
                    ),
                    "runtime_personal_monthly_disposable_income_currency": "GBP",
                    "runtime_personal_monthly_disposable_income_min_cents": (
                        1_000 + income_index * 2_000
                    ),
                    "runtime_personal_monthly_disposable_income_max_cents_exclusive": (
                        2_000 + income_index * 2_000
                    ),
                    "modeled_players_per_household": 2,
                    "conversion_recipe_id": "test.explicit.conversion.v1",
                    "conversion_recipe_sha256": sha256(
                        b"test-only explicit source-to-runtime conversion"
                    ).hexdigest(),
                }
            )
    entries.sort(
        key=lambda row: (
            row["jurisdiction_code"],
            row["source_household_income_band_id"],
            row["source_household_type_id"],
        )
    )
    return {
        "schema_version": 1,
        "mapping_id": "test.population.runtime.mapping",
        "design_id": bundle.design_id,
        "design_bundle_sha256": bundle.bundle_sha256,
        "domain_sha256": bundle.domain_sha256,
        "source_income_concept": SOURCE_INCOME_CONCEPT,
        "runtime_income_concept": RUNTIME_INCOME_CONCEPT,
        "entries": entries,
    }


def _execution(
    root: Path,
    *,
    seed: int = 610,
) -> tuple[PopulationProjectionExecution, Path]:
    _design_path, _evidence, _results, verification = _write_complete_fixture(root)
    target = build_population_calibration_target(verification)
    plan = apportion_population_hamilton(
        target,
        16,
        first_player_id=100,
    )
    mapping_path = root / "runtime_mapping.json"
    mapping_path.write_text(
        json.dumps(
            _mapping_payload(verification),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
        newline="",
    )
    mapping = load_population_runtime_mapping_bundle(mapping_path)
    adapter = build_population_projection_adapter(
        verification,
        plan,
        mapping,
        adapter_id="test.population.adapter.v1",
    )
    execution = initialize_population_projection(
        adapter,
        (CountryProfile(code="UK", adult_age=18),),
        CounterRNG(seed),
    )
    return execution, mapping_path


class PopulationBalanceArtifactTests(unittest.TestCase):
    def test_exact_joint_balance_is_content_addressed_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution, _mapping_path = _execution(Path(directory))

            first = build_population_balance_artifact(execution)
            repeated = build_population_balance_artifact(execution)

            self.assertEqual(first, repeated)
            self.assertEqual(first.schema_version, POPULATION_BALANCE_SCHEMA_VERSION)
            self.assertTrue(first.exact_balance_passed)
            self.assertEqual(first.execution_sha256, execution.execution_sha256)
            self.assertEqual(first.adapter_sha256, execution.adapter.adapter_sha256)
            self.assertEqual(
                first.apportionment_sha256,
                execution.adapter.apportionment_plan.apportionment_sha256,
            )
            self.assertEqual(len(first.cells), 8)
            self.assertEqual(
                tuple(cell.cell_ordinal for cell in first.cells),
                tuple(range(8)),
            )
            self.assertTrue(
                all(cell.sample_count_discrepancy == 0 for cell in first.cells)
            )
            self.assertTrue(
                all(cell.target_mass == cell.sidecar_mass for cell in first.cells)
            )
            self.assertTrue(
                all(cell.target_mass == cell.realized_mass for cell in first.cells)
            )
            self.assertEqual(sum(cell.realized_mass for cell in first.cells), 1)
            self.assertTrue(first.runtime_membership.age_membership_passed)
            self.assertTrue(first.runtime_membership.income_membership_passed)
            self.assertTrue(first.runtime_membership.household_membership_passed)
            self.assertEqual(
                validate_population_balance_snapshot(first.snapshot(), execution),
                first,
            )

    def test_artifact_binds_runtime_values_even_when_membership_still_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution, _mapping_path = _execution(Path(directory), seed=611)
            original = build_population_balance_artifact(execution)
            assignment = execution.players.projected_population
            assert assignment is not None
            first_cell = assignment.cell_for_player_position(0)
            current = int(execution.players.age_years[0])
            adult_age = execution.players.adult_age_by_jurisdiction[
                int(execution.players.jurisdiction[0])
            ]
            candidates = (
                current + 1,
                current - 1,
                first_cell.age_min_inclusive,
            )
            replacement = next(
                candidate
                for candidate in candidates
                if first_cell.age_min_inclusive
                <= candidate
                < first_cell.age_max_exclusive
                and (candidate < adult_age) is bool(execution.players.is_minor[0])
                and candidate != current
            )
            execution.players.age_years[0] = replacement

            changed = build_population_balance_artifact(execution)
            self.assertNotEqual(
                original.runtime_membership.age_membership_sha256,
                changed.runtime_membership.age_membership_sha256,
            )
            self.assertNotEqual(original.balance_sha256, changed.balance_sha256)
            with self.assertRaisesRegex(
                PopulationBalanceValidationError,
                "stale|mutated",
            ):
                validate_population_balance_snapshot(original.snapshot(), execution)

    def test_adult_age_threshold_metadata_is_bound_even_if_flags_do_not_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution, _mapping_path = _execution(Path(directory), seed=681)
            original = build_population_balance_artifact(execution)
            players = execution.players
            original_threshold = players.adult_age_by_jurisdiction[0]
            replacement = next(
                threshold
                for threshold in range(1, 80)
                if threshold != original_threshold
                and np.array_equal(players.age_years < threshold, players.is_minor)
            )
            object.__setattr__(
                players,
                "adult_age_by_jurisdiction",
                (replacement,),
            )

            changed = build_population_balance_artifact(execution)
            self.assertNotEqual(
                original.runtime_membership.age_membership_sha256,
                changed.runtime_membership.age_membership_sha256,
            )
            self.assertNotEqual(original.balance_sha256, changed.balance_sha256)

    def test_runtime_age_income_jurisdiction_and_household_are_separate_checks(
        self,
    ) -> None:
        mutations = ("age", "income", "jurisdiction", "minor", "household")
        for offset, mutation in enumerate(mutations):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                execution, _mapping_path = _execution(
                    Path(directory),
                    seed=620 + offset,
                )
                players = execution.players
                assignment = players.projected_population
                assert assignment is not None
                cell = assignment.cell_for_player_position(0)
                if mutation == "age":
                    players.age_years[0] = cell.age_max_exclusive
                    expected = "age membership"
                elif mutation == "income":
                    players.monthly_disposable_income_cents[0] = (
                        cell.monthly_disposable_income_max_cents_exclusive
                    )
                    expected = "income membership"
                elif mutation == "jurisdiction":
                    players.jurisdiction[0] = 1
                    expected = "jurisdiction membership"
                elif mutation == "minor":
                    players.is_minor[0] = ~players.is_minor[0]
                    expected = "age/minor membership"
                else:
                    first_group = (
                        cell.jurisdiction_code,
                        cell.monthly_disposable_income_band_id,
                        cell.household_type,
                    )
                    other_position = next(
                        index
                        for index in range(1, len(players))
                        if (
                            assignment.cell_for_player_position(index).jurisdiction_code,
                            assignment.cell_for_player_position(
                                index
                            ).monthly_disposable_income_band_id,
                            assignment.cell_for_player_position(index).household_type,
                        )
                        != first_group
                    )
                    players.household_id[other_position] = players.household_id[0]
                    expected = "household crosses"
                with self.assertRaisesRegex(
                    PopulationBalanceValidationError,
                    expected,
                ):
                    build_population_balance_artifact(execution)

    def test_stale_assignment_weights_and_positive_cell_loss_are_rejected(self) -> None:
        cases = ("assignment", "weight", "positive-cell")
        for offset, case in enumerate(cases):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                execution, _mapping_path = _execution(
                    Path(directory),
                    seed=630 + offset,
                )
                assignment = execution.players.projected_population
                assert assignment is not None
                if case == "assignment":
                    first = 0
                    second = int(
                        np.flatnonzero(
                            assignment.cell_index != assignment.cell_index[first]
                        )[0]
                    )
                    assignment.cell_index.flags.writeable = True
                    assignment.cell_index[first], assignment.cell_index[second] = (
                        assignment.cell_index[second],
                        assignment.cell_index[first],
                    )
                    assignment.cell_index.flags.writeable = False
                elif case == "weight":
                    object.__setattr__(
                        assignment.metadata.cells[0],
                        "analysis_weight",
                        (1, 999),
                    )
                else:
                    assignment.cell_index.flags.writeable = True
                    assignment.cell_index[assignment.cell_index == 0] = 1
                    assignment.cell_index.flags.writeable = False
                with self.assertRaisesRegex(
                    PopulationBalanceValidationError,
                    "mutated|mismatched|sidecar|execution",
                ):
                    build_population_balance_artifact(execution)

    def test_sidecar_rational_tuple_subclasses_are_rejected(self) -> None:
        class HiddenTuple(tuple):
            pass

        for offset, field_name in enumerate(("global_mass", "analysis_weight")):
            with self.subTest(field=field_name), tempfile.TemporaryDirectory() as directory:
                execution, _mapping_path = _execution(
                    Path(directory),
                    seed=635 + offset,
                )
                assignment = execution.players.projected_population
                assert assignment is not None
                cell = assignment.metadata.cells[0]
                object.__setattr__(
                    cell,
                    field_name,
                    HiddenTuple(getattr(cell, field_name)),
                )

                with self.assertRaisesRegex(TypeError, "exact tuple"):
                    build_population_balance_artifact(execution)

    def test_plan_and_cell_set_mismatches_are_rejected(self) -> None:
        for offset, case in enumerate(("plan", "missing-cell")):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                execution, _mapping_path = _execution(
                    Path(directory),
                    seed=640 + offset,
                )
                if case == "plan":
                    object.__setattr__(
                        execution.adapter.apportionment_plan,
                        "apportionment_sha256",
                        "f" * 64,
                    )
                else:
                    object.__setattr__(
                        execution.adapter,
                        "cells",
                        execution.adapter.cells[:-1],
                    )
                with self.assertRaisesRegex(
                    PopulationBalanceValidationError,
                    "adapter|plan|mismatched",
                ):
                    build_population_balance_artifact(execution)

    def test_mapping_bytes_are_reopened_before_balance_is_attested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution, mapping_path = _execution(Path(directory), seed=650)
            mapping_path.write_text("{}", encoding="utf-8", newline="")

            with self.assertRaisesRegex(
                PopulationBalanceValidationError,
                "adapter could not be re-attested",
            ):
                build_population_balance_artifact(execution)

    def test_nested_artifact_rows_are_revalidated_after_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution, _mapping_path = _execution(Path(directory), seed=655)
            artifact = build_population_balance_artifact(execution)
            object.__setattr__(
                artifact.cells[0],
                "realized_sample_count",
                artifact.cells[0].realized_sample_count + 1,
            )

            with self.assertRaisesRegex(
                PopulationBalanceValidationError,
                "discrepancy|realized",
            ):
                type(artifact).__post_init__(artifact)

    def test_polymorphic_inputs_and_snapshot_containers_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution, _mapping_path = _execution(Path(directory), seed=660)

            class HiddenExecution(PopulationProjectionExecution):
                __slots__ = ()

            hidden_execution = HiddenExecution(
                **{
                    descriptor.name: getattr(execution, descriptor.name)
                    for descriptor in fields(PopulationProjectionExecution)
                }
            )
            with self.assertRaisesRegex(TypeError, "PopulationProjectionExecution"):
                build_population_balance_artifact(hidden_execution)

            class HiddenAdapterCell(PopulationProjectionAdapterCell):
                __slots__ = ()

            base = execution.adapter.cells[0]
            hidden_cell = HiddenAdapterCell(
                **{
                    descriptor.name: getattr(base, descriptor.name)
                    for descriptor in fields(PopulationProjectionAdapterCell)
                }
            )
            object.__setattr__(
                execution.adapter,
                "cells",
                (hidden_cell, *execution.adapter.cells[1:]),
            )
            with self.assertRaisesRegex(TypeError, "PopulationProjectionAdapterCell"):
                build_population_balance_artifact(execution)

        with tempfile.TemporaryDirectory() as directory:
            execution, _mapping_path = _execution(Path(directory), seed=661)
            artifact = build_population_balance_artifact(execution)

            class HiddenDict(dict):
                pass

            with self.assertRaises(TypeError):
                validate_population_balance_snapshot(
                    HiddenDict(artifact.snapshot()),
                    execution,
                )

    def test_nonfinite_and_oversized_snapshot_values_fail_closed(self) -> None:
        for offset, (name, value, expected) in enumerate(
            (
                ("nonfinite", float("nan"), "non-finite"),
                ("oversized", 2**5000, "4096-bit"),
            )
        ):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                execution, _mapping_path = _execution(
                    Path(directory),
                    seed=670 + offset,
                )
                artifact = build_population_balance_artifact(execution)
                snapshot = deepcopy(artifact.snapshot())
                snapshot["player_count"] = value
                with self.assertRaisesRegex(
                    (PopulationBalanceValidationError, TypeError),
                    expected,
                ):
                    validate_population_balance_snapshot(snapshot, execution)

    def test_nonfinite_runtime_polymorphism_fails_before_array_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            execution, _mapping_path = _execution(Path(directory), seed=680)
            ages = execution.players.age_years.astype(np.float64)
            ages[0] = np.nan
            object.__setattr__(execution.players, "age_years", ages)

            with self.assertRaisesRegex(TypeError, "age_years"):
                build_population_balance_artifact(execution)


if __name__ == "__main__":
    unittest.main()
