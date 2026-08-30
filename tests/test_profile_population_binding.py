from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from microtx_sim.data.lineage import (
    ProfileInputLineage,
    build_profile_input_lineage,
)
from microtx_sim.data.population_evidence import (
    POPULATION_CELL_CSV_COLUMNS,
    PopulationEstimandRole,
    exact_csv_joint_population_recipe_json,
)
from microtx_sim.data.profiles import (
    DEFAULT_JURISDICTIONS_PATH,
    DEFAULT_POPULATION_BUNDLE_PATH,
    DEFAULT_SOURCES_PATH,
    ProfileValidationError,
    load_profile_bundle,
)
from microtx_sim.outputs.manifest import _population_readiness_payload


_FIXTURES = Path(__file__).parent / "fixtures"
_FALSE_POPULATION_GATE = {
    "structure_coherent": False,
    "source_population_evidence_bound": False,
    "calibration_targets_bound": False,
    "heldout_validation_targets_bound": False,
    "source_bundle_signature_bound": False,
    "sampling_plan_bound": False,
    "runtime_projection_bound": False,
    "output_estimand_binding_bound": False,
    "balance_validation_bound": False,
    "public_population_comparability": False,
}
_JURISDICTION_SPECS = (
    ("BE", "Belgium", "EUR"),
    ("JP", "Japan", "JPY"),
    ("KR", "South Korea", "KRW"),
    ("UK", "United Kingdom", "GBP"),
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _population_csv_bytes(
    role: PopulationEstimandRole,
    *,
    income_band: str = "income.all",
    copy_calibration_masses: bool = False,
    omit_non_gamer: bool = False,
) -> bytes:
    rows = [list(POPULATION_CELL_CSV_COLUMNS)]
    gaming_states = ("GAMER",) if omit_non_gamer else ("GAMER", "NON_GAMER")
    cell_count = len(gaming_states) * 2 * 2
    for code, _geography, _currency in _JURISDICTION_SPECS:
        target_id = f"{code.lower()}.mobile_players.{role.value.lower()}"
        selector = [target_id, code, role.value]
        cell_index = 0
        for gaming_state in gaming_states:
            for payer_state in ("EVER_PAYER", "NEVER_PAYER"):
                for age_min, age_max in ((10, 25), (25, 70)):
                    cell_index += 1
                    if (
                        role is PopulationEstimandRole.CALIBRATION
                        or copy_calibration_masses
                    ):
                        mass = ("1", str(cell_count))
                    elif cell_index == cell_count:
                        mass = ("2", str(cell_count + 1))
                    else:
                        mass = ("1", str(cell_count + 1))
                    rows.append(
                        [
                            *selector,
                            f"cell.{cell_index:02d}",
                            str(age_min),
                            str(age_max),
                            income_band,
                            "household.all",
                            gaming_state,
                            payer_state,
                            *mass,
                        ]
                    )
    return (
        "\n".join(",".join(row) for row in rows) + "\n"
    ).encode("utf-8")


def _write_complete_population_fixture(
    root: Path,
    *,
    source_geography_override: tuple[str, str] | None = None,
    reuse_calibration_sources_for_validation: bool = False,
    reuse_calibration_source_urls_for_validation: bool = False,
    validation_eligibility_override: str | None = None,
    validation_income_band: str | None = None,
    copy_calibration_targets_to_validation: bool = False,
    calibration_omit_non_gamer: bool = False,
    source_period_override: tuple[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    jurisdictions = root / "jurisdictions.toml"
    sources = root / "sources.toml"
    population = root / "population_bundle.toml"
    jurisdictions.write_bytes(DEFAULT_JURISDICTIONS_PATH.read_bytes())

    source_text = DEFAULT_SOURCES_PATH.read_text(encoding="utf-8")
    source_blocks = []
    override_source_id, override_geography = (
        source_geography_override
        if source_geography_override is not None
        else (None, None)
    )
    override_period_source_id, override_period = (
        source_period_override
        if source_period_override is not None
        else (None, None)
    )
    for code, geography, _currency in _JURISDICTION_SPECS:
        for role in PopulationEstimandRole:
            source_id = f"POP_{code}_{role.value}"
            declared_geography = (
                override_geography
                if source_id == override_source_id
                else geography
            )
            declared_period = (
                override_period
                if source_id == override_period_source_id
                else "2025"
            )
            url_role = (
                PopulationEstimandRole.CALIBRATION
                if reuse_calibration_source_urls_for_validation
                and role is PopulationEstimandRole.VALIDATION
                else role
            )
            source_url_id = f"POP_{code}_{url_role.value}"
            source_blocks.append(
                f'''\n[[source]]
id = "{source_id}"
publisher = "Test population authority"
title = "Test-only {role.value.lower()} population cells"
url = "https://example.invalid/{source_url_id.lower()}"
period = "{declared_period}"
geography = "{declared_geography}"
supports = ["age_structure", "income_distribution", "household_composition", "gaming_reach", "conditional_payer_rate"]
calibration_status = "CALIBRATED"
'''
            )
    sources.write_text(
        source_text + "".join(source_blocks),
        encoding="utf-8",
        newline="",
    )
    source_sha256 = sha256(sources.read_bytes()).hexdigest()

    artifact_root = root / "population_artifacts"
    artifact_root.mkdir()
    artifacts: list[tuple[str, str, bytes]] = []
    for role in PopulationEstimandRole:
        slug = role.value.lower()
        content = _population_csv_bytes(
            role,
            income_band=(
                validation_income_band
                if role is PopulationEstimandRole.VALIDATION
                and validation_income_band is not None
                else "income.all"
            ),
            copy_calibration_masses=(
                copy_calibration_targets_to_validation
                and role is PopulationEstimandRole.VALIDATION
            ),
            omit_non_gamer=(
                calibration_omit_non_gamer
                and role is PopulationEstimandRole.CALIBRATION
            ),
        )
        relative_path = f"joint_{slug}.csv"
        (artifact_root / relative_path).write_bytes(content)
        artifacts.append((f"joint.{slug}", relative_path, content))

    artifact_tables = []
    for artifact_id, relative_path, content in artifacts:
        artifact_tables.append(
            f'''[[artifacts]]
artifact_id = "{artifact_id}"
relative_path = "{relative_path}"
media_type = "text/csv"
sha256 = "{sha256(content).hexdigest()}"
byte_length = {len(content)}

'''
        )

    bindings = []
    for role in PopulationEstimandRole:
        slug = role.value.lower()
        for code, geography, currency in _JURISDICTION_SPECS:
            target_id = f"{code.lower()}.mobile_players.{slug}"
            source_role = (
                PopulationEstimandRole.CALIBRATION
                if reuse_calibration_sources_for_validation
                and role is PopulationEstimandRole.VALIDATION
                else role
            )
            eligibility = (
                validation_eligibility_override
                if role is PopulationEstimandRole.VALIDATION
                and validation_eligibility_override is not None
                else "usual residents in the declared age range"
            )
            recipe = exact_csv_joint_population_recipe_json(
                target_population_id=target_id,
                jurisdiction_code=code,
                estimand_role=role,
            )
            bindings.append(
                f'''[[bindings]]
binding_id = "population.{slug}.{code.lower()}"
artifact_id = "joint.{slug}"
target_population_id = "{target_id}"
jurisdiction_code = "{code}"
geography = "{geography}"
reference_period_start = 2025-01-01
reference_period_end = 2025-12-31
population_base = "resident population"
universe = "mobile-game players and non-players"
unit_of_analysis = "person"
eligibility = "{eligibility}"
exclusion = "institutional residents"
age_min_inclusive = 10
age_max_exclusive = 70
household_income_definition = "gross household income"
household_income_currency = "{currency}"
household_income_period = "annual"
household_income_equivalisation = "modified OECD scale"
household_definition = "shared dwelling and budget"
gaming_definition = "mobile-game play in reference period"
payer_definition = "ever paid before the reference-period end"
zero_spender_treatment = "retained as never-payer cells"
estimand_role = "{role.value}"
status = "CALIBRATED"
source_ids = ["POP_{code}_{source_role.value}"]
retrieved_on = 2026-08-24
recipe_json = {json.dumps(recipe)}

'''
            )

    population.write_text(
        f'''schema_version = 1
bundle_id = "complete-test-population-evidence"
provenance_status = "CALIBRATED"
source_registry_sha256 = "{source_sha256}"
artifact_root = "population_artifacts"
notes = "Test-only complete population evidence; not substantive evidence."

{"".join(artifact_tables)}{"".join(bindings)}[signature]
status = "MISSING"
algorithm = "NONE"
key_id = ""
value = ""
''',
        encoding="utf-8",
        newline="",
    )
    return jurisdictions, sources, population


class ProfilePopulationBindingTests(unittest.TestCase):
    def test_complete_country_specific_evidence_clears_only_source_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(Path(directory))
            bundle = load_profile_bundle(
                paths[0],
                paths[1],
                source_bundle_path=None,
                population_bundle_path=paths[2],
                campaign=False,
            )

            assessment = bundle.population_evidence_assessment()
            self.assertTrue(assessment.structure_coherent)
            self.assertTrue(assessment.source_population_evidence_bound)
            self.assertFalse(assessment.calibration_targets_bound)
            self.assertFalse(assessment.heldout_validation_targets_bound)
            self.assertFalse(assessment.source_bundle_signature_bound)
            self.assertFalse(assessment.sampling_plan_bound)
            self.assertFalse(assessment.runtime_projection_bound)
            self.assertFalse(assessment.output_estimand_binding_bound)
            self.assertFalse(assessment.balance_validation_bound)
            self.assertFalse(assessment.public_population_comparability)
            self.assertEqual(
                assessment.blockers,
                (
                    "population.calibration_targets=missing",
                    "population.heldout_validation_targets=missing",
                    "population.source_bundle_signature=missing",
                    "population.sampling_plan=missing",
                    "population.runtime_projection=missing",
                    "population.output_estimand_binding=missing",
                    "population.balance_validation=missing",
                ),
            )

            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            self.assertEqual(lineage.snapshot["schema_version"], 4)
            payload = lineage.manifest_payload()
            self.assertEqual(
                payload["population_evidence_summary"]["verified_result_count"],
                8,
            )
            readiness = _population_readiness_payload(
                payload,
                profile_lineage=lineage,
            )
            self.assertTrue(
                readiness["manifest_gate"][
                    "source_population_evidence_bound"
                ]
            )
            self.assertFalse(
                readiness["manifest_gate"]["calibration_targets_bound"]
            )
            self.assertFalse(
                readiness["manifest_gate"][
                    "heldout_validation_targets_bound"
                ]
            )
            self.assertFalse(
                readiness["manifest_gate"][
                    "public_population_comparability"
                ]
            )

            unregistered = replace(
                bundle,
                jurisdictions_path=None,
                source_registry_path=None,
                jurisdictions_sha256=None,
                source_registry_sha256=None,
            )
            with self.assertRaisesRegex(
                ProfileValidationError,
                "cannot claim registered population evidence",
            ):
                unregistered.population_evidence_assessment(registered=True)

    def test_validation_targets_reusing_calibration_sources_are_not_held_out(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                reuse_calibration_sources_for_validation=True,
            )
            bundle = load_profile_bundle(
                paths[0],
                paths[1],
                source_bundle_path=None,
                population_bundle_path=paths[2],
                campaign=False,
            )

            assessment = bundle.population_evidence_assessment()
            self.assertTrue(assessment.source_population_evidence_bound)
            self.assertFalse(assessment.calibration_targets_bound)
            self.assertFalse(assessment.heldout_validation_targets_bound)
            self.assertIn(
                "population.heldout_validation_targets=missing",
                assessment.blockers,
            )

    def test_incomplete_gaming_payer_support_cannot_bind_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                calibration_omit_non_gamer=True,
            )
            bundle = load_profile_bundle(
                paths[0],
                paths[1],
                source_bundle_path=None,
                population_bundle_path=paths[2],
                campaign=False,
            )

            assessment = bundle.population_evidence_assessment()
            self.assertFalse(assessment.structure_coherent)
            self.assertFalse(assessment.source_population_evidence_bound)
            self.assertFalse(assessment.calibration_targets_bound)
            self.assertTrue(
                any(
                    blocker.startswith(
                        "population.structure=incomplete_joint_support:"
                    )
                    for blocker in assessment.blockers
                )
            )

    def test_copied_validation_targets_are_not_held_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                copy_calibration_targets_to_validation=True,
            )
            bundle = load_profile_bundle(
                paths[0],
                paths[1],
                source_bundle_path=None,
                population_bundle_path=paths[2],
                campaign=False,
            )

            assessment = bundle.population_evidence_assessment()
            self.assertTrue(assessment.structure_coherent)
            self.assertFalse(assessment.calibration_targets_bound)
            self.assertFalse(assessment.heldout_validation_targets_bound)
            self.assertIn(
                "population.heldout_validation_targets=missing",
                assessment.blockers,
            )

    def test_validation_sources_with_aliased_urls_are_not_held_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                reuse_calibration_source_urls_for_validation=True,
            )
            bundle = load_profile_bundle(
                paths[0],
                paths[1],
                source_bundle_path=None,
                population_bundle_path=paths[2],
                campaign=False,
            )

            assessment = bundle.population_evidence_assessment()
            self.assertTrue(assessment.structure_coherent)
            self.assertFalse(assessment.calibration_targets_bound)
            self.assertFalse(assessment.heldout_validation_targets_bound)
            self.assertIn(
                "population.heldout_validation_targets=missing",
                assessment.blockers,
            )

    def test_validation_targets_with_a_different_basis_are_not_held_out(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                validation_eligibility_override=(
                    "a deliberately different validation population"
                ),
            )
            bundle = load_profile_bundle(
                paths[0],
                paths[1],
                source_bundle_path=None,
                population_bundle_path=paths[2],
                campaign=False,
            )

            assessment = bundle.population_evidence_assessment()
            self.assertTrue(assessment.source_population_evidence_bound)
            self.assertFalse(assessment.calibration_targets_bound)
            self.assertFalse(assessment.heldout_validation_targets_bound)
            self.assertIn(
                "population.heldout_validation_targets=missing",
                assessment.blockers,
            )

    def test_validation_targets_with_different_cells_are_not_held_out(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                validation_income_band="income.different",
            )
            bundle = load_profile_bundle(
                paths[0],
                paths[1],
                source_bundle_path=None,
                population_bundle_path=paths[2],
                campaign=False,
            )

            assessment = bundle.population_evidence_assessment()
            self.assertTrue(assessment.structure_coherent)
            self.assertFalse(assessment.calibration_targets_bound)
            self.assertFalse(assessment.heldout_validation_targets_bound)
            self.assertIn(
                "population.heldout_validation_targets=missing",
                assessment.blockers,
            )

    def test_cross_country_population_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                source_geography_override=(
                    "POP_UK_CALIBRATION",
                    "South Korea",
                ),
            )
            with self.assertRaisesRegex(
                ProfileValidationError,
                "sources without explicit United Kingdom applicability",
            ):
                load_profile_bundle(
                    paths[0],
                    paths[1],
                    source_bundle_path=None,
                    population_bundle_path=paths[2],
                    campaign=False,
                )

    def test_qualified_population_source_geography_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                source_geography_override=(
                    "POP_UK_CALIBRATION",
                    "United Kingdom; ages 8-17",
                ),
            )
            with self.assertRaisesRegex(
                ProfileValidationError,
                "sources without explicit United Kingdom applicability",
            ):
                load_profile_bundle(
                    paths[0],
                    paths[1],
                    source_bundle_path=None,
                    population_bundle_path=paths[2],
                    campaign=False,
                )

    def test_population_source_period_must_match_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_complete_population_fixture(
                Path(directory),
                source_period_override=("POP_UK_CALIBRATION", "2024"),
            )
            with self.assertRaisesRegex(
                ProfileValidationError,
                "without an exact reference-period declaration",
            ):
                load_profile_bundle(
                    paths[0],
                    paths[1],
                    source_bundle_path=None,
                    population_bundle_path=paths[2],
                    campaign=False,
                )

    def test_default_bundle_is_bound_into_schema_v4_lineage(self) -> None:
        bundle = load_profile_bundle(campaign=False)
        evidence = bundle.population_evidence_bundle
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertTrue(bundle.matches_registered_files())

        assessment = bundle.population_evidence_assessment()
        self.assertEqual(
            assessment.blockers,
            (
                "population.source_evidence=missing",
                "population.calibration_targets=missing",
                "population.heldout_validation_targets=missing",
                "population.source_bundle_signature=missing",
                "population.sampling_plan=missing",
                "population.runtime_projection=missing",
                "population.output_estimand_binding=missing",
                "population.balance_validation=missing",
            ),
        )
        with self.assertRaisesRegex(
            ProfileValidationError,
            "population.source_evidence=missing",
        ):
            bundle.validate_population_for_campaign()

        lineage = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )
        snapshot = lineage.snapshot
        self.assertEqual(snapshot["schema_version"], 4)
        self.assertEqual(
            snapshot["file_lineage"]["population_bundle"],
            {
                "path": str(DEFAULT_POPULATION_BUNDLE_PATH.resolve()),
                "sha256": sha256(
                    DEFAULT_POPULATION_BUNDLE_PATH.read_bytes()
                ).hexdigest(),
                "source_registry_sha256": bundle.source_registry_sha256,
                "signature_status": "MISSING",
            },
        )

        payload = lineage.manifest_payload()
        self.assertEqual(
            payload["population_evidence_summary"],
            {
                "present": True,
                "artifact_count": 1,
                "binding_count": 8,
                "verified_result_count": 8,
                "signature_status": "MISSING",
            },
        )
        self.assertEqual(
            payload["population_evidence_assessment"],
            json.loads(_canonical_json(asdict(assessment))),
        )

    def test_manifest_population_gate_requires_the_exact_lineage_payload(self) -> None:
        bundle = load_profile_bundle(campaign=False)
        lineage = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )
        payload = lineage.manifest_payload()
        readiness = _population_readiness_payload(
            payload,
            profile_lineage=lineage,
        )

        self.assertEqual(readiness["schema_version"], "1.0")
        self.assertEqual(
            readiness["typed_assessment"],
            payload["population_evidence_assessment"],
        )
        expected_registered_gate = dict(_FALSE_POPULATION_GATE)
        expected_registered_gate["structure_coherent"] = True
        self.assertEqual(
            readiness["manifest_gate"],
            expected_registered_gate,
        )

        without_lineage = _population_readiness_payload(
            payload,
            profile_lineage=None,
        )
        self.assertEqual(without_lineage["manifest_gate"], _FALSE_POPULATION_GATE)

        promoted = deepcopy(payload)
        promoted_assessment = promoted["snapshot"]["profile_bundle"][
            "population_evidence_assessment"
        ]
        for field in _FALSE_POPULATION_GATE:
            promoted_assessment[field] = True
        promoted_assessment["blockers"] = []
        promoted["population_evidence_assessment"] = deepcopy(
            promoted_assessment
        )
        promoted_readiness = _population_readiness_payload(
            promoted,
            profile_lineage=lineage,
        )
        self.assertEqual(
            promoted_readiness["manifest_gate"],
            _FALSE_POPULATION_GATE,
        )

    def test_schema_v3_lineage_cannot_claim_population_readiness(self) -> None:
        snapshot_json = (
            _FIXTURES / "profile_lineage_v3.json"
        ).read_text(encoding="utf-8").strip()
        lineage = ProfileInputLineage(
            lineage_status="unregistered_custom_profiles",
            profile_codes=("ZZ",),
            fingerprint_sha256=sha256(snapshot_json.encode("utf-8")).hexdigest(),
            snapshot_json=snapshot_json,
        )

        readiness = _population_readiness_payload(
            lineage.manifest_payload(),
            profile_lineage=lineage,
        )
        self.assertEqual(readiness["manifest_gate"], _FALSE_POPULATION_GATE)
        self.assertEqual(
            readiness["typed_assessment"]["blockers"][0],
            "population.structure=unavailable",
        )

    def test_population_snapshot_tampering_is_rejected(self) -> None:
        bundle = load_profile_bundle(campaign=False)
        lineage = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )

        variants = []
        promoted = lineage.snapshot
        promoted["profile_bundle"]["population_evidence_assessment"][
            "source_population_evidence_bound"
        ] = True
        variants.append(promoted)

        forged_bundle = lineage.snapshot
        forged_bundle["profile_bundle"]["population_evidence_bundle"][
            "campaign_ready"
        ] = True
        variants.append(forged_bundle)

        reordered = lineage.snapshot
        blockers = reordered["profile_bundle"][
            "population_evidence_assessment"
        ]["blockers"]
        blockers[0], blockers[1] = blockers[1], blockers[0]
        variants.append(reordered)

        for index, snapshot in enumerate(variants):
            with self.subTest(index=index):
                snapshot_json = _canonical_json(snapshot)
                with self.assertRaises(ProfileValidationError):
                    replace(
                        lineage,
                        snapshot_json=snapshot_json,
                        fingerprint_sha256=sha256(
                            snapshot_json.encode("utf-8")
                        ).hexdigest(),
                    )

    def test_registered_lineage_rejects_population_bundle_byte_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jurisdictions = root / "jurisdictions.toml"
            sources = root / "sources.toml"
            population = root / "population_bundle.toml"
            jurisdictions.write_bytes(DEFAULT_JURISDICTIONS_PATH.read_bytes())
            sources.write_bytes(DEFAULT_SOURCES_PATH.read_bytes())
            population.write_bytes(DEFAULT_POPULATION_BUNDLE_PATH.read_bytes())
            shutil.copytree(
                DEFAULT_POPULATION_BUNDLE_PATH.with_name(
                    "population_artifacts"
                ),
                root / "population_artifacts",
            )

            bundle = load_profile_bundle(
                jurisdictions,
                sources,
                source_bundle_path=None,
                population_bundle_path=population,
                campaign=False,
            )
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            self.assertEqual(
                lineage.lineage_status,
                "registered_profile_bundle",
            )

            population.write_bytes(population.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                ProfileValidationError,
                "population-evidence bundle metadata no longer match",
            ):
                replace(lineage)


if __name__ == "__main__":
    unittest.main()
