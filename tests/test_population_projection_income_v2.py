from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_population_projection_adapter import (  # noqa: E402
    _complete_adapter,
    _mapping_payload,
    _write_mapping,
)

from microtx_sim.consumers.population import (  # noqa: E402
    CountryProfile,
    PROJECTED_INCOME_BOUNDARY_RULE,
    PROJECTED_INCOME_MINOR_GAMING_ADJUSTMENT,
    PROJECTED_INCOME_MINOR_GAMING_ADJUSTMENT_REASON,
    PROJECTED_INCOME_MODEL_FAMILY,
    PROJECTED_INCOME_ROUNDING_RULE,
    PROJECTED_INCOME_TARGET_QUANTITY,
    PopulationProjectionCell,
    PopulationProjectionIncomeModel,
    PopulationProjectionSampleCount,
    initialize_projected_player_table_from_exact_counts,
)
from microtx_sim.data.population_projection import (  # noqa: E402
    PopulationProjectionValidationError,
    build_population_projection_adapter,
    initialize_population_projection,
)
from microtx_sim.rng import CounterRNG  # noqa: E402


_MODEL_KEY = "runtime_personal_monthly_disposable_income_model"


def _income_model_payload(entry: dict[str, object]) -> dict[str, object]:
    minimum = entry["runtime_personal_monthly_disposable_income_min_cents"]
    maximum = entry[
        "runtime_personal_monthly_disposable_income_max_cents_exclusive"
    ]
    assert type(minimum) is int
    assert type(maximum) is int
    return {
        "target_quantity": PROJECTED_INCOME_TARGET_QUANTITY,
        "model_family": PROJECTED_INCOME_MODEL_FAMILY,
        "median_cents": (minimum + maximum - 1) // 2,
        "log_sigma": [1, 2],
        "lower_bound_cents": minimum,
        "upper_bound_cents_inclusive": maximum - 1,
        "currency": entry[
            "runtime_personal_monthly_disposable_income_currency"
        ],
        "time_period": "calendar-year 2024 monthly personal disposable income",
        "source_id": entry["conversion_recipe_id"],
        "calibration_target": "cell-specific personal monthly income median",
        "transformation": (
            "source household-income calibration converted to personal monthly "
            "cents before fitting"
        ),
        "boundary_rule": PROJECTED_INCOME_BOUNDARY_RULE,
        "rounding_rule": PROJECTED_INCOME_ROUNDING_RULE,
        "minor_gaming_adjustment": PROJECTED_INCOME_MINOR_GAMING_ADJUSTMENT,
        "minor_gaming_adjustment_reason": (
            PROJECTED_INCOME_MINOR_GAMING_ADJUSTMENT_REASON
        ),
    }


def _v2_mapping_payload(verification: object) -> dict[str, object]:
    payload = deepcopy(_mapping_payload(verification))
    payload["schema_version"] = 2
    payload["mapping_id"] = "test.source-to-runtime.v2"
    entries = payload["entries"]
    assert type(entries) is list
    for entry in entries:
        assert type(entry) is dict
        entry[_MODEL_KEY] = _income_model_payload(entry)
    return payload


def _income_model(
    *,
    median_cents: int = 2_000,
    transformation: str = "direct deterministic test calibration",
) -> PopulationProjectionIncomeModel:
    return PopulationProjectionIncomeModel(
        target_quantity=PROJECTED_INCOME_TARGET_QUANTITY,
        model_family=PROJECTED_INCOME_MODEL_FAMILY,
        median_cents=median_cents,
        log_sigma=(1, 2),
        lower_bound_cents=1_000,
        upper_bound_cents_inclusive=3_000,
        currency="GBP",
        time_period="calendar-year 2024",
        source_id="official.test.table",
        calibration_target="personal monthly disposable-income median",
        transformation=transformation,
        boundary_rule=PROJECTED_INCOME_BOUNDARY_RULE,
        rounding_rule=PROJECTED_INCOME_ROUNDING_RULE,
        minor_gaming_adjustment=PROJECTED_INCOME_MINOR_GAMING_ADJUSTMENT,
        minor_gaming_adjustment_reason=(
            PROJECTED_INCOME_MINOR_GAMING_ADJUSTMENT_REASON
        ),
    )


def _cell(
    *,
    cell_id: str = "cell.0",
    income_model: PopulationProjectionIncomeModel | None,
    global_mass: tuple[int, int] = (1, 1),
) -> PopulationProjectionCell:
    return PopulationProjectionCell(
        cell_id=cell_id,
        jurisdiction_code="UK",
        age_min_inclusive=18,
        age_max_exclusive=40,
        monthly_disposable_income_band_id="runtime.income.all",
        monthly_disposable_income_min_cents=1_000,
        monthly_disposable_income_max_cents_exclusive=3_001,
        household_type="household.all",
        modeled_players_per_household=1,
        baseline_gamer=True,
        baseline_ever_payer=False,
        global_mass=global_mass,
        income_model=income_model,
    )


class _CoordinateRNG:
    """Small deterministic protocol fixture that records income RNG methods."""

    def __init__(self) -> None:
        self.income_uniform_calls = 0
        self.income_normal_calls = 0

    def uniform(
        self,
        entity_ids: np.ndarray,
        tick: int,
        stream: int,
        draw_index: int,
    ) -> np.ndarray:
        del tick, draw_index
        if stream == 1_102:
            self.income_uniform_calls += 1
            values = np.asarray([0.0, 0.5, np.nextafter(1.0, 0.0)])
            return values[entity_ids.astype(np.int64)]
        return np.zeros(entity_ids.shape, dtype=np.float64)

    def normal(
        self,
        entity_ids: np.ndarray,
        tick: int,
        stream: int,
        draw_index: int,
    ) -> np.ndarray:
        del tick, draw_index
        if stream == 1_102:
            self.income_normal_calls += 1
            values = np.asarray([-100.0, 0.0, 100.0])
            return values[entity_ids.astype(np.int64)]
        return np.zeros(entity_ids.shape, dtype=np.float64)


class PopulationProjectionIncomeV2Tests(unittest.TestCase):
    def test_v1_declaration_attestations_and_uniform_income_path_are_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verification, _plan, _path, mapping, adapter = _complete_adapter(
                Path(directory)
            )
            self.assertEqual(mapping.schema_version, 1)
            self.assertEqual(
                mapping.declaration_snapshot(),
                _mapping_payload(verification),
            )
            self.assertTrue(
                all(entry.income_model is None for entry in mapping.entries)
            )
            self.assertEqual(adapter.snapshot()["schema_version"], 1)
            self.assertNotIn("mapping_schema_version", adapter.snapshot())
            execution = initialize_population_projection(
                adapter,
                (CountryProfile(code="UK"),),
                CounterRNG(991),
            )
            self.assertEqual(execution.snapshot()["schema_version"], 1)
            self.assertNotIn("adapter_schema_version", execution.snapshot())

        rng = _CoordinateRNG()
        players = initialize_projected_player_table_from_exact_counts(
            3,
            (CountryProfile(code="UK"),),
            rng,
            (_cell(income_model=None),),
            (PopulationProjectionSampleCount("cell.0", 3),),
            projection_id="test.v1.uniform",
        )
        self.assertEqual(rng.income_uniform_calls, 1)
        self.assertEqual(rng.income_normal_calls, 0)
        self.assertEqual(
            players.monthly_disposable_income_cents.tolist(),
            [1_000, 2_000, 3_000],
        )

    def test_v2_loads_projects_and_propagates_mapping_adapter_execution_versions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verification, plan, _path, v1_mapping, v1_adapter = _complete_adapter(
                root
            )
            payload = _v2_mapping_payload(verification)
            mapping = _write_mapping(root / "runtime-mapping-v2.json", payload)
            adapter = build_population_projection_adapter(
                verification,
                plan,
                mapping,
                adapter_id="test.population-projection.v2",
            )

            self.assertEqual(mapping.schema_version, 2)
            self.assertEqual(mapping.declaration_snapshot(), payload)
            self.assertTrue(
                all(entry.income_model is not None for entry in mapping.entries)
            )
            self.assertTrue(
                all(
                    cell.projection_cell.income_model is not None
                    for cell in adapter.cells
                )
            )
            self.assertNotEqual(mapping.mapping_sha256, v1_mapping.mapping_sha256)
            self.assertNotEqual(adapter.adapter_sha256, v1_adapter.adapter_sha256)
            self.assertEqual(adapter.snapshot()["schema_version"], 2)
            self.assertEqual(adapter.snapshot()["mapping_schema_version"], 2)

            first = initialize_population_projection(
                adapter,
                (CountryProfile(code="UK"),),
                CounterRNG(992),
            )
            second = initialize_population_projection(
                adapter,
                (CountryProfile(code="UK"),),
                CounterRNG(992),
            )
            np.testing.assert_array_equal(
                first.players.monthly_disposable_income_cents,
                second.players.monthly_disposable_income_cents,
            )
            self.assertEqual(first.snapshot()["schema_version"], 2)
            self.assertEqual(first.snapshot()["adapter_schema_version"], 2)
            self.assertEqual(first.snapshot()["mapping_schema_version"], 2)
            assignment = first.players.projected_population
            assert assignment is not None
            for position, cell_index in enumerate(assignment.cell_index):
                cell = adapter.cells[int(cell_index)].projection_cell
                income = int(first.players.monthly_disposable_income_cents[position])
                self.assertGreaterEqual(
                    income,
                    cell.monthly_disposable_income_min_cents,
                )
                self.assertLess(
                    income,
                    cell.monthly_disposable_income_max_cents_exclusive,
                )

    def test_v2_uses_counter_normal_then_exp_and_censors_without_resampling(
        self,
    ) -> None:
        rng = _CoordinateRNG()
        players = initialize_projected_player_table_from_exact_counts(
            3,
            (CountryProfile(code="UK"),),
            rng,
            (_cell(income_model=_income_model()),),
            (PopulationProjectionSampleCount("cell.0", 3),),
            projection_id="test.v2.log-normal",
        )
        self.assertEqual(rng.income_uniform_calls, 0)
        self.assertEqual(rng.income_normal_calls, 1)
        self.assertEqual(
            players.monthly_disposable_income_cents.tolist(),
            [1_000, 2_000, 3_000],
        )

    def test_v2_rejects_missing_or_malformed_income_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verification, _plan, _path, _mapping, _adapter = _complete_adapter(root)
            base = _v2_mapping_payload(verification)

            attacks: dict[str, object] = {
                "missing-model": None,
                "zero-sigma": ("log_sigma", [0, 1]),
                "unreduced-sigma": ("log_sigma", [2, 4]),
                "wrong-target": ("target_quantity", "household_income"),
                "outside-median": ("median_cents", 999_999),
                "wrong-bound": ("lower_bound_cents", 0),
                "wrong-currency": ("currency", "EUR"),
                "missing-source": ("source_id", ""),
                "conditional-minor-adjustment": (
                    "minor_gaming_adjustment",
                    "CONDITIONAL_SELECTION",
                ),
                "unsupported-none-reason": (
                    "minor_gaming_adjustment_reason",
                    "UNDECLARED",
                ),
            }
            for label, mutation in attacks.items():
                candidate = deepcopy(base)
                entries = candidate["entries"]
                assert type(entries) is list and type(entries[0]) is dict
                if mutation is None:
                    del entries[0][_MODEL_KEY]
                else:
                    field, value = mutation
                    model = entries[0][_MODEL_KEY]
                    assert type(model) is dict
                    model[field] = value
                with self.subTest(label=label), self.assertRaises(
                    PopulationProjectionValidationError
                ):
                    _write_mapping(root / f"invalid-{label}.json", candidate)

            v1_with_model = _mapping_payload(verification)
            entries = v1_with_model["entries"]
            assert type(entries) is list and type(entries[0]) is dict
            entries[0][_MODEL_KEY] = _income_model_payload(entries[0])
            with self.assertRaises(PopulationProjectionValidationError):
                _write_mapping(root / "invalid-v1-model.json", v1_with_model)

    def test_v2_model_metadata_changes_mapping_entry_and_adapter_identities(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verification, plan, _path, _mapping, _adapter = _complete_adapter(root)
            first_payload = _v2_mapping_payload(verification)
            second_payload = deepcopy(first_payload)
            second_entries = second_payload["entries"]
            assert type(second_entries) is list and type(second_entries[0]) is dict
            second_model = second_entries[0][_MODEL_KEY]
            assert type(second_model) is dict
            second_model["transformation"] = "documented sensitivity transformation"

            first_mapping = _write_mapping(root / "first.json", first_payload)
            second_mapping = _write_mapping(root / "second.json", second_payload)
            first_adapter = build_population_projection_adapter(
                verification,
                plan,
                first_mapping,
                adapter_id="test.identity.v2",
            )
            second_adapter = build_population_projection_adapter(
                verification,
                plan,
                second_mapping,
                adapter_id="test.identity.v2",
            )
            self.assertNotEqual(
                first_mapping.mapping_sha256,
                second_mapping.mapping_sha256,
            )
            self.assertNotEqual(
                first_mapping.entries[0].mapping_entry_sha256,
                second_mapping.entries[0].mapping_entry_sha256,
            )
            self.assertNotEqual(
                first_adapter.adapter_sha256,
                second_adapter.adapter_sha256,
            )

    def test_initializer_rejects_conflicting_models_within_runtime_group(
        self,
    ) -> None:
        first = _cell(
            cell_id="cell.0",
            income_model=_income_model(),
            global_mass=(1, 2),
        )
        second = replace(
            first,
            cell_id="cell.1",
            global_mass=(1, 2),
            income_model=_income_model(
                transformation="different undocumented cell transformation"
            ),
        )
        with self.assertRaisesRegex(ValueError, "one runtime income interval"):
            initialize_projected_player_table_from_exact_counts(
                2,
                (CountryProfile(code="UK"),),
                _CoordinateRNG(),
                (first, second),
                (
                    PopulationProjectionSampleCount("cell.0", 1),
                    PopulationProjectionSampleCount("cell.1", 1),
                ),
                projection_id="test.v2.conflict",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
