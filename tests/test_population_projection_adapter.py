from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

# Reuse the exact file-backed design/evidence fixture rather than introducing a
# second, subtly different source-contract generator in this test module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from four_jurisdiction_population_fixture import (  # noqa: E402
    CANONICAL_DESIGN_CODES,
    REGISTERED_PROFILE_CODES,
    write_four_jurisdiction_population_fixture,
)
from monetary_execution_fixture import (  # noqa: E402
    write_monetary_execution_fixture,
)
from test_population_design import (  # noqa: E402
    _design_text,
    _write_complete_fixture,
    _write_evidence,
)

from microtx_sim.consumers.population import (  # noqa: E402
    CountryProfile,
    PopulationProjectionCell,
    PopulationProjectionSampleCount,
    initialize_projected_player_table_from_exact_counts,
)
from microtx_sim.data.population_design import (  # noqa: E402
    apportion_population_hamilton,
    build_population_calibration_target,
    load_and_verify_population_design_bundle,
)
from microtx_sim.data.population_execution import (  # noqa: E402
    build_population_execution_lineage,
    build_population_seed_execution_record,
)
from microtx_sim.data.population_projection import (  # noqa: E402
    RUNTIME_INCOME_CONCEPT,
    SOURCE_INCOME_CONCEPT,
    PopulationProjectionAdapter,
    PopulationProjectionAdapterCell,
    PopulationProjectionExecution,
    PopulationProjectionValidationError,
    PopulationProjectionVerificationError,
    PopulationRuntimeMappingBundle,
    build_population_projection_adapter,
    initialize_population_projection,
    load_population_runtime_mapping_bundle,
    population_projection_ordered_player_ids_sha256,
    verify_population_projection_adapter,
    verify_population_projection_execution,
)
from microtx_sim.data.population_evidence import (  # noqa: E402
    load_and_verify_population_evidence_bundle,
)
from microtx_sim.rng import CounterRNG  # noqa: E402


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _mapping_payload(verification: object) -> dict[str, object]:
    bundle = verification.bundle
    entries: list[dict[str, object]] = []
    for income_band in bundle.income_bands:
        for household in bundle.household_types:
            minimum = 1_000 + 2_000 * income_band.ordinal
            maximum = minimum + 2_000
            recipe_id = (
                f"convert.{income_band.jurisdiction_code.lower()}."
                f"{income_band.income_band_id.split('.')[-1]}."
                f"{household.household_type_id.split('.')[-1]}"
            )
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
                        "runtime.personal.monthly."
                        + income_band.income_band_id.split(".")[-1]
                    ),
                    "runtime_personal_monthly_disposable_income_currency": "GBP",
                    "runtime_personal_monthly_disposable_income_min_cents": minimum,
                    "runtime_personal_monthly_disposable_income_max_cents_exclusive": (
                        maximum
                    ),
                    "modeled_players_per_household": 2,
                    "conversion_recipe_id": recipe_id,
                    "conversion_recipe_sha256": sha256(
                        recipe_id.encode("utf-8")
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
        "mapping_id": "test.source-to-runtime.v1",
        "design_id": bundle.design_id,
        "design_bundle_sha256": bundle.bundle_sha256,
        "domain_sha256": bundle.domain_sha256,
        "source_income_concept": SOURCE_INCOME_CONCEPT,
        "runtime_income_concept": RUNTIME_INCOME_CONCEPT,
        "entries": entries,
    }


def _write_mapping(
    path: Path,
    payload: dict[str, object],
) -> PopulationRuntimeMappingBundle:
    path.write_text(
        _canonical_json(payload) + "\n",
        encoding="utf-8",
        newline="",
    )
    return load_population_runtime_mapping_bundle(path)


def _complete_adapter(
    root: Path,
) -> tuple[object, object, Path, PopulationRuntimeMappingBundle, PopulationProjectionAdapter]:
    _design_path, _evidence, _results, verification = _write_complete_fixture(root)
    target = build_population_calibration_target(verification)
    plan = apportion_population_hamilton(target, 12, first_player_id=100)
    mapping_path = root / "runtime-mapping.json"
    mapping = _write_mapping(mapping_path, _mapping_payload(verification))
    adapter = build_population_projection_adapter(
        verification,
        plan,
        mapping,
        adapter_id="test.population-projection.v1",
    )
    return verification, plan, mapping_path, mapping, adapter


def _zero_cell_verification(root: Path) -> object:
    evidence, _results = _write_evidence(root)
    artifact_path = root / "population_artifacts" / "joint.csv"
    original = artifact_path.read_bytes()
    lines = original.decode("utf-8").splitlines()
    masses = ((1, 4), (1, 4), (1, 4), (1, 4), (0, 1), (0, 1), (0, 1), (0, 1))
    for index, (numerator, denominator) in enumerate(masses, start=1):
        row = lines[index].split(",")
        row[-2:] = [str(numerator), str(denominator)]
        lines[index] = ",".join(row)
    changed = ("\n".join(lines) + "\n").encode("utf-8")
    artifact_path.write_bytes(changed)

    evidence_path = root / "population_bundle.toml"
    declaration = evidence_path.read_text(encoding="utf-8")
    declaration = declaration.replace(
        sha256(original).hexdigest(),
        sha256(changed).hexdigest(),
        1,
    ).replace(
        f"byte_length = {len(original)}",
        f"byte_length = {len(changed)}",
        1,
    )
    evidence_path.write_text(declaration, encoding="utf-8", newline="")
    evidence, results = load_and_verify_population_evidence_bundle(evidence_path)
    design_path = root / "population_design.toml"
    design_path.write_text(
        _design_text(evidence, results),
        encoding="utf-8",
        newline="",
    )
    return load_and_verify_population_design_bundle(
        design_path,
        population_evidence_bundle=evidence,
        population_evidence_results=results,
    )


class PopulationProjectionAdapterTests(unittest.TestCase):
    def test_adapter_maps_every_static_cell_and_executes_exact_plan_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verification, plan, _path, mapping, adapter = _complete_adapter(
                Path(directory)
            )

            self.assertEqual(len(adapter.cells), len(plan.cells))
            self.assertEqual(
                [cell.cell_ordinal for cell in adapter.cells],
                list(range(8)),
            )
            self.assertEqual(
                [cell.sample_count for cell in adapter.cells],
                [2, 2, 2, 2, 1, 1, 1, 1],
            )
            self.assertEqual(
                [cell.projection_cell.cell_id for cell in adapter.cells],
                [f"cell.{ordinal:020d}" for ordinal in range(8)],
            )
            self.assertEqual(
                {cell.projection_cell.age_min_inclusive for cell in adapter.cells},
                {8},
            )
            self.assertEqual(
                {cell.projection_cell.age_max_exclusive for cell in adapter.cells},
                {70},
            )
            self.assertEqual(
                [cell.projection_cell.baseline_gamer for cell in adapter.cells],
                [True, True, False, False, True, True, False, False],
            )
            self.assertEqual(
                [cell.projection_cell.baseline_ever_payer for cell in adapter.cells],
                [True, False, True, False, True, False, True, False],
            )
            self.assertEqual(
                {cell.projection_cell.household_type for cell in adapter.cells},
                {"household.all"},
            )
            self.assertEqual(adapter.mapping_id, mapping.mapping_id)
            self.assertEqual(adapter.mapping_sha256, mapping.mapping_sha256)
            self.assertFalse(adapter.authenticity_verified)
            self.assertFalse(adapter.balance_verified)
            self.assertFalse(adapter.campaign_ready)
            self.assertEqual(verify_population_projection_adapter(adapter), adapter)

            with patch(
                "microtx_sim.consumers.population._hamilton_cell_counts",
                side_effect=AssertionError("runtime must not reapportion"),
            ):
                execution = initialize_population_projection(
                    adapter,
                    (CountryProfile(code="UK"),),
                    CounterRNG(901),
                )

            assignment = execution.players.projected_population
            assert assignment is not None
            observed_counts = np.bincount(
                assignment.cell_index.astype(np.int64, copy=False),
                minlength=len(adapter.cells),
            )
            self.assertEqual(observed_counts.tolist(), [2, 2, 2, 2, 1, 1, 1, 1])
            np.testing.assert_array_equal(
                execution.players.player_id,
                np.arange(100, 112, dtype=np.int64),
            )
            self.assertEqual(
                execution.runtime_projection_id,
                f"adapter.{adapter.adapter_sha256}",
            )
            self.assertEqual(
                execution.ordered_player_ids_sha256,
                population_projection_ordered_player_ids_sha256(
                    execution.players.player_id
                ),
            )
            self.assertFalse(execution.campaign_ready)
            self.assertIs(
                verify_population_projection_execution(execution), execution
            )
            self.assertEqual(
                execution.adapter.verification.verification_sha256,
                verification.verification_sha256,
            )

    def test_four_jurisdiction_execution_accepts_reordered_profile_scope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            monetary_root = root / "monetary"
            monetary_root.mkdir()
            profile_bundle, _artifact_path = write_monetary_execution_fixture(
                monetary_root
            )
            adapter = write_four_jurisdiction_population_fixture(
                root / "population"
            )
            profiles = profile_bundle.country_profiles

            self.assertEqual(
                tuple(
                    jurisdiction.jurisdiction_code
                    for jurisdiction in adapter.verification.bundle.jurisdictions
                ),
                CANONICAL_DESIGN_CODES,
            )
            self.assertEqual(
                tuple(profile.code for profile in profiles),
                REGISTERED_PROFILE_CODES,
            )
            self.assertEqual(adapter.apportionment_plan.player_count, 16)
            self.assertFalse(adapter.campaign_ready)

            execution = initialize_population_projection(
                adapter,
                profiles,
                CounterRNG(907),
            )
            self.assertEqual(
                execution.players.jurisdiction_codes,
                REGISTERED_PROFILE_CODES,
            )
            code_to_index = {
                code: index
                for index, code in enumerate(execution.players.jurisdiction_codes)
            }
            assignment = execution.players.projected_population
            assert assignment is not None
            self.assertTrue(
                all(
                    cell.jurisdiction_index == code_to_index[cell.jurisdiction_code]
                    for cell in assignment.metadata.cells
                )
            )
            self.assertIs(
                verify_population_projection_execution(execution),
                execution,
            )
            seed_record = build_population_seed_execution_record(
                execution,
                seed=907,
                cohort_digest="7" * 64,
                policy_days=1,
            )
            self.assertEqual(
                seed_record.jurisdiction_codes,
                REGISTERED_PROFILE_CODES,
            )
            lineage = build_population_execution_lineage(adapter, (seed_record,))
            self.assertEqual(
                lineage.record_for_seed(907).jurisdiction_codes,
                REGISTERED_PROFILE_CODES,
            )
            self.assertEqual(lineage.manifest_payload(), lineage.snapshot())

            invalid_scopes = {
                "wrong": (CountryProfile(code="GB"), *profiles[1:]),
                "missing": profiles[:-1],
                "duplicate": (profiles[0], profiles[0], profiles[2], profiles[3]),
            }
            for label, invalid_profiles in invalid_scopes.items():
                with self.subTest(label=label), self.assertRaisesRegex(
                    PopulationProjectionVerificationError,
                    "unique and exactly match",
                ):
                    initialize_population_projection(
                        adapter,
                        invalid_profiles,
                        CounterRNG(908),
                    )

    def test_zero_cells_are_preserved_in_adapter_and_runtime_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            verification = _zero_cell_verification(root)
            plan = apportion_population_hamilton(
                build_population_calibration_target(verification),
                8,
            )
            mapping = _write_mapping(
                root / "mapping.json",
                _mapping_payload(verification),
            )
            adapter = build_population_projection_adapter(
                verification,
                plan,
                mapping,
                adapter_id="test.zero-cell-projection.v1",
            )
            self.assertEqual(len(adapter.cells), len(plan.calibration_target.cells))
            self.assertEqual(
                [cell.sample_count for cell in adapter.cells],
                [2, 2, 2, 2, 0, 0, 0, 0],
            )
            execution = initialize_population_projection(
                adapter,
                (CountryProfile(code="UK"),),
                CounterRNG(906),
            )
            assignment = execution.players.projected_population
            assert assignment is not None
            self.assertEqual(len(assignment.metadata.cells), 8)
            self.assertEqual(
                np.bincount(
                    assignment.cell_index.astype(np.int64, copy=False),
                    minlength=8,
                ).tolist(),
                [2, 2, 2, 2, 0, 0, 0, 0],
            )
            self.assertEqual(
                [cell.analysis_weight for cell in assignment.metadata.cells[4:]],
                [(0, 1)] * 4,
            )

    def test_adapter_is_deterministic_and_mapping_bytes_are_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            rebuilt = verify_population_projection_adapter(adapter)
            self.assertEqual(rebuilt.adapter_sha256, adapter.adapter_sha256)
            self.assertEqual(rebuilt.snapshot(), adapter.snapshot())

            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                PopulationProjectionVerificationError,
                "current exact bytes",
            ):
                verify_population_projection_adapter(adapter)

    def test_mapping_requires_exact_complete_source_semantics(self) -> None:
        for attack in ("missing", "extra", "definition"):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _design, _evidence, _results, verification = _write_complete_fixture(root)
                target = build_population_calibration_target(verification)
                plan = apportion_population_hamilton(target, 12)
                payload = _mapping_payload(verification)
                entries = payload["entries"]
                assert isinstance(entries, list)
                if attack == "missing":
                    entries.pop()
                elif attack == "extra":
                    forged = deepcopy(entries[-1])
                    forged["source_household_type_id"] = "household.extra"
                    forged["source_household_type_definition"] = "extra"
                    entries.append(forged)
                    entries.sort(
                        key=lambda row: (
                            row["jurisdiction_code"],
                            row["source_household_income_band_id"],
                            row["source_household_type_id"],
                        )
                    )
                elif attack == "definition":
                    entries[0]["source_household_income_definition"] = "aliased income"
                mapping = _write_mapping(root / "mapping.json", payload)
                with self.assertRaises(PopulationProjectionVerificationError):
                    build_population_projection_adapter(
                        verification,
                        plan,
                        mapping,
                        adapter_id="invalid.mapping",
                    )

    def test_mapping_rejects_duplicate_and_runtime_band_aliases(self) -> None:
        for attack in (
            "duplicate",
            "runtime-alias",
            "runtime-case-alias",
            "source-runtime-alias",
            "income-concept-alias",
            "boolean-numeric-alias",
        ):
            with self.subTest(attack=attack), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _design, _evidence, _results, verification = _write_complete_fixture(root)
                payload = _mapping_payload(verification)
                entries = payload["entries"]
                assert isinstance(entries, list)
                if attack == "duplicate":
                    entries.insert(1, deepcopy(entries[0]))
                elif attack == "runtime-alias":
                    entries[1][
                        "runtime_personal_monthly_disposable_income_band_id"
                    ] = entries[0][
                        "runtime_personal_monthly_disposable_income_band_id"
                    ]
                elif attack == "runtime-case-alias":
                    entries[1][
                        "runtime_personal_monthly_disposable_income_band_id"
                    ] = str(
                        entries[0][
                            "runtime_personal_monthly_disposable_income_band_id"
                        ]
                    ).upper()
                elif attack == "source-runtime-alias":
                    entries[0][
                        "runtime_personal_monthly_disposable_income_band_id"
                    ] = entries[0]["source_household_income_band_id"]
                elif attack == "income-concept-alias":
                    payload["runtime_income_concept"] = SOURCE_INCOME_CONCEPT
                else:
                    entries[0]["modeled_players_per_household"] = True
                path = root / "invalid.json"
                path.write_text(
                    _canonical_json(payload) + "\n",
                    encoding="utf-8",
                    newline="",
                )
                with self.assertRaises(PopulationProjectionValidationError):
                    load_population_runtime_mapping_bundle(path)

    def test_plan_must_belong_to_the_exact_verified_target(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_root = Path(first_directory)
            verification, _plan, _path, mapping, _adapter = _complete_adapter(
                first_root
            )
            second_root = Path(second_directory)
            design_path, evidence, results, _second = _write_complete_fixture(second_root)
            design_path.write_text(
                design_path.read_text(encoding="utf-8").replace(
                    "Test-only complete static declarations",
                    "Distinct complete static declarations",
                    1,
                ),
                encoding="utf-8",
                newline="",
            )
            second_verification = load_and_verify_population_design_bundle(
                design_path,
                population_evidence_bundle=evidence,
                population_evidence_results=results,
            )
            second_plan = apportion_population_hamilton(
                build_population_calibration_target(second_verification),
                12,
            )
            with self.assertRaisesRegex(
                PopulationProjectionVerificationError,
                "does not belong",
            ):
                build_population_projection_adapter(
                    verification,
                    second_plan,
                    mapping,
                    adapter_id="mismatched.plan",
                )

    def test_polymorphic_contract_objects_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            verification, plan, _path, mapping, adapter = _complete_adapter(
                Path(directory)
            )

            class HiddenMapping(PopulationRuntimeMappingBundle):
                __slots__ = ()

            hidden_mapping = HiddenMapping(
                **{
                    descriptor.name: getattr(mapping, descriptor.name)
                    for descriptor in fields(PopulationRuntimeMappingBundle)
                }
            )
            with self.assertRaisesRegex(TypeError, "mapping_bundle"):
                build_population_projection_adapter(
                    verification,
                    plan,
                    hidden_mapping,
                    adapter_id="hidden.mapping",
                )

            class HiddenAdapter(PopulationProjectionAdapter):
                __slots__ = ()

            hidden_adapter = HiddenAdapter(
                **{
                    descriptor.name: getattr(adapter, descriptor.name)
                    for descriptor in fields(PopulationProjectionAdapter)
                }
            )
            with self.assertRaisesRegex(TypeError, "adapter"):
                verify_population_projection_adapter(hidden_adapter)

            class HiddenAdapterCell(PopulationProjectionAdapterCell):
                __slots__ = ()

            base_cell = adapter.cells[0]
            hidden_cell = HiddenAdapterCell(
                **{
                    descriptor.name: getattr(base_cell, descriptor.name)
                    for descriptor in fields(PopulationProjectionAdapterCell)
                }
            )
            with self.assertRaisesRegex(
                PopulationProjectionValidationError,
                "exact adapter cells",
            ):
                replace(adapter, cells=(hidden_cell, *adapter.cells[1:]))

    def test_execution_rejects_wrong_profile_scope_and_mutated_player_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _verification, _plan, _path, _mapping, adapter = _complete_adapter(
                Path(directory)
            )
            with self.assertRaisesRegex(
                PopulationProjectionVerificationError,
                "code set",
            ):
                initialize_population_projection(
                    adapter,
                    (CountryProfile(code="GB"),),
                    CounterRNG(902),
                )
            execution = initialize_population_projection(
                adapter,
                (CountryProfile(code="UK"),),
                CounterRNG(903),
            )
            execution.players.player_id[0] = 999
            with self.assertRaisesRegex(
                PopulationProjectionVerificationError,
                "player ids",
            ):
                verify_population_projection_execution(execution)


class ExactCountPopulationInitializerTests(unittest.TestCase):
    @staticmethod
    def _cells() -> tuple[PopulationProjectionCell, PopulationProjectionCell]:
        return (
            PopulationProjectionCell(
                cell_id="cell.0",
                jurisdiction_code="UK",
                age_min_inclusive=18,
                age_max_exclusive=40,
                monthly_disposable_income_band_id="runtime.low",
                monthly_disposable_income_min_cents=1_000,
                monthly_disposable_income_max_cents_exclusive=2_000,
                household_type="household.a",
                modeled_players_per_household=1,
                baseline_gamer=True,
                baseline_ever_payer=False,
                global_mass=(1, 2),
            ),
            PopulationProjectionCell(
                cell_id="cell.1",
                jurisdiction_code="UK",
                age_min_inclusive=18,
                age_max_exclusive=40,
                monthly_disposable_income_band_id="runtime.high",
                monthly_disposable_income_min_cents=2_000,
                monthly_disposable_income_max_cents_exclusive=3_000,
                household_type="household.a",
                modeled_players_per_household=1,
                baseline_gamer=False,
                baseline_ever_payer=True,
                global_mass=(1, 2),
            ),
        )

    def test_exact_counts_are_keyed_canonical_and_content_addressed(self) -> None:
        cells = self._cells()
        profiles = (CountryProfile(code="UK"),)
        balanced = initialize_projected_player_table_from_exact_counts(
            4,
            profiles,
            CounterRNG(904),
            cells,
            (
                PopulationProjectionSampleCount("cell.0", 2),
                PopulationProjectionSampleCount("cell.1", 2),
            ),
            projection_id="exact-count.test",
        )
        skewed = initialize_projected_player_table_from_exact_counts(
            4,
            profiles,
            CounterRNG(904),
            cells,
            (
                PopulationProjectionSampleCount("cell.0", 1),
                PopulationProjectionSampleCount("cell.1", 3),
            ),
            projection_id="exact-count.test",
        )
        assert balanced.projected_population is not None
        assert skewed.projected_population is not None
        self.assertNotEqual(
            balanced.projected_population.metadata.projection_sha256,
            skewed.projected_population.metadata.projection_sha256,
        )
        self.assertEqual(
            [cell.analysis_weight for cell in balanced.projected_population.metadata.cells],
            [(1, 4), (1, 4)],
        )
        self.assertEqual(
            [cell.analysis_weight for cell in skewed.projected_population.metadata.cells],
            [(1, 2), (1, 6)],
        )

        with self.assertRaisesRegex(ValueError, "canonical cell_id order"):
            initialize_projected_player_table_from_exact_counts(
                4,
                profiles,
                CounterRNG(904),
                cells,
                (
                    PopulationProjectionSampleCount("cell.1", 2),
                    PopulationProjectionSampleCount("cell.0", 2),
                ),
                projection_id="exact-count.invalid",
            )

    def test_exact_counts_reject_alias_types_and_mass_count_mismatch(self) -> None:
        cells = self._cells()
        profiles = (CountryProfile(code="UK"),)

        class HiddenCount(PopulationProjectionSampleCount):
            __slots__ = ()

        with self.assertRaisesRegex(TypeError, "PopulationProjectionSampleCount"):
            initialize_projected_player_table_from_exact_counts(
                4,
                profiles,
                CounterRNG(905),
                cells,
                (
                    HiddenCount("cell.0", 2),
                    PopulationProjectionSampleCount("cell.1", 2),
                ),
                projection_id="exact-count.hidden",
            )
        with self.assertRaisesRegex(ValueError, "sum to player_count"):
            initialize_projected_player_table_from_exact_counts(
                4,
                profiles,
                CounterRNG(905),
                cells,
                (
                    PopulationProjectionSampleCount("cell.0", 1),
                    PopulationProjectionSampleCount("cell.1", 1),
                ),
                projection_id="exact-count.bad-sum",
            )


if __name__ == "__main__":
    unittest.main()
