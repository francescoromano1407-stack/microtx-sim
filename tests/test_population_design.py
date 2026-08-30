from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.data.population_design import (
    CANONICAL_SOURCE_CLUSTER_ID_V1,
    CANONICAL_SOURCE_RECORD_ID_V1,
    DEFAULT_POPULATION_DESIGN_BUNDLE_PATH,
    EXACT_RATIONAL_HAMILTON_V1,
    POPULATION_DESIGN_SCHEMA_VERSION,
    SHA256_CLUSTER_THRESHOLD_V1,
    PopulationApportionmentPlan,
    PopulationDesignVerification,
    PopulationDesignValidationError,
    PopulationDesignVerificationError,
    apportion_population_hamilton,
    assigned_population_partition_role,
    build_population_calibration_target,
    load_and_verify_population_design_bundle,
    load_population_design_bundle,
    validate_population_apportionment_snapshot,
    validate_population_design_snapshot,
    verify_population_design_bundle,
)
from microtx_sim.data.population_evidence import (
    DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH,
    POPULATION_CELL_CSV_COLUMNS,
    PopulationEstimandRole,
    exact_csv_joint_population_recipe_json,
    load_and_verify_population_evidence_bundle,
)
from microtx_sim.types import ProvenanceStatus


_SOURCE_REGISTRY_SHA256 = "a" * 64
_PARTITION_SEED = "1" * 64
_CALIBRATION_BINDING_ID = "00.calibration.uk"
_VALIDATION_BINDING_ID = "01.validation.uk"
_CALIBRATION_TARGET_ID = "uk.mobile.calibration"
_VALIDATION_TARGET_ID = "uk.mobile.validation"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _population_rows(
    *,
    role: PopulationEstimandRole,
    masses: tuple[Fraction, ...],
) -> list[list[str]]:
    rows: list[list[str]] = []
    index = 0
    for income_band in ("income.lower", "income.upper"):
        for gaming_state in ("GAMER", "NON_GAMER"):
            for payer_state in ("EVER_PAYER", "NEVER_PAYER"):
                mass = masses[index]
                prefix = "cal" if role is PopulationEstimandRole.CALIBRATION else "val"
                rows.append(
                    [
                        (
                            _CALIBRATION_TARGET_ID
                            if role is PopulationEstimandRole.CALIBRATION
                            else _VALIDATION_TARGET_ID
                        ),
                        "UK",
                        role.value,
                        f"{prefix}.{index:02d}",
                        "8",
                        "70",
                        income_band,
                        "household.all",
                        gaming_state,
                        payer_state,
                        str(mass.numerator),
                        str(mass.denominator),
                    ]
                )
                index += 1
    return rows


def _write_evidence(
    root: Path,
    *,
    validation_masses: tuple[Fraction, ...] | None = None,
) -> tuple[object, tuple[object, ...]]:
    calibration_masses = (Fraction(1, 8),) * 8
    validation_masses = validation_masses or (
        Fraction(1, 16),
        Fraction(1, 16),
        Fraction(1, 8),
        Fraction(1, 8),
        Fraction(1, 8),
        Fraction(1, 8),
        Fraction(3, 16),
        Fraction(3, 16),
    )
    rows = [
        *_population_rows(
            role=PopulationEstimandRole.CALIBRATION,
            masses=calibration_masses,
        ),
        *_population_rows(
            role=PopulationEstimandRole.VALIDATION,
            masses=validation_masses,
        ),
    ]
    content = (
        ",".join(POPULATION_CELL_CSV_COLUMNS)
        + "\n"
        + "\n".join(",".join(row) for row in rows)
        + "\n"
    ).encode("utf-8")
    artifact_root = root / "population_artifacts"
    artifact_root.mkdir()
    (artifact_root / "joint.csv").write_bytes(content)
    bindings: list[str] = []
    for role, binding_id, target_id, source_id in (
        (
            PopulationEstimandRole.CALIBRATION,
            _CALIBRATION_BINDING_ID,
            _CALIBRATION_TARGET_ID,
            "SRC_CALIBRATION",
        ),
        (
            PopulationEstimandRole.VALIDATION,
            _VALIDATION_BINDING_ID,
            _VALIDATION_TARGET_ID,
            "SRC_VALIDATION",
        ),
    ):
        recipe = exact_csv_joint_population_recipe_json(
            target_population_id=target_id,
            jurisdiction_code="UK",
            estimand_role=role,
        )
        bindings.append(
            f'''[[bindings]]
binding_id = "{binding_id}"
artifact_id = "joint.population"
target_population_id = "{target_id}"
jurisdiction_code = "UK"
geography = "United Kingdom"
reference_period_start = 2025-01-01
reference_period_end = 2025-12-31
population_base = "resident population"
universe = "mobile-game players and non-players"
unit_of_analysis = "person"
eligibility = "usual residents aged 8 to 69"
exclusion = "institutional residents"
age_min_inclusive = 8
age_max_exclusive = 70
household_income_definition = "gross household income"
household_income_currency = "GBP"
household_income_period = "annual"
household_income_equivalisation = "none"
household_definition = "declared source household"
gaming_definition = "mobile-game play in reference period"
payer_definition = "ever paid before the reference-period endpoint"
zero_spender_treatment = "retained as never-payer cells"
estimand_role = "{role.value}"
status = "CALIBRATED"
source_ids = ["{source_id}"]
retrieved_on = 2026-01-15
recipe_json = {json.dumps(recipe)}
'''
        )
    evidence_text = f'''schema_version = 1
bundle_id = "test-population-evidence"
provenance_status = "CALIBRATED"
source_registry_sha256 = "{_SOURCE_REGISTRY_SHA256}"
artifact_root = "population_artifacts"
notes = "Test-only exact joint target declarations."

[[artifacts]]
artifact_id = "joint.population"
relative_path = "joint.csv"
media_type = "text/csv"
sha256 = "{sha256(content).hexdigest()}"
byte_length = {len(content)}

{"".join(bindings)}
[signature]
status = "MISSING"
algorithm = "NONE"
key_id = ""
value = ""
'''
    evidence_path = root / "population_bundle.toml"
    evidence_path.write_text(evidence_text, encoding="utf-8", newline="")
    return load_and_verify_population_evidence_bundle(evidence_path)


def _cluster_for_role(role: PopulationEstimandRole, ordinal: int) -> str:
    candidate = 0
    while True:
        digest = sha256(
            f"cluster:{role.value}:{ordinal}:{candidate}".encode("utf-8")
        ).hexdigest()
        observed = assigned_population_partition_role(
            identity_namespace="test.population.units",
            assignment_seed_sha256=_PARTITION_SEED,
            cluster_identity_sha256=digest,
            calibration_threshold_numerator=1,
            calibration_threshold_denominator=2,
        )
        if observed is role:
            return digest
        candidate += 1


def _design_text(evidence_bundle: object, evidence_results: tuple[object, ...]) -> str:
    calibration_result, validation_result = evidence_results
    records: list[tuple[str, str]] = []
    for role, result in (
        (PopulationEstimandRole.CALIBRATION, calibration_result),
        (PopulationEstimandRole.VALIDATION, validation_result),
    ):
        for ordinal, cell in enumerate(result.cells):
            if cell.target_mass == 0:
                continue
            record_id = sha256(
                f"record:{role.value}:{ordinal}".encode("utf-8")
            ).hexdigest()
            cluster_id = _cluster_for_role(role, ordinal)
            records.append(
                (
                    record_id,
                    f'''[[partition.records]]
record_identity_sha256 = "{record_id}"
cluster_identity_sha256 = "{cluster_id}"
estimand_role = "{role.value}"
binding_id = "{result.binding_id}"
cell_id = "{cell.cell_id}"
record_weight_numerator = {cell.target_mass_numerator}
record_weight_denominator = {cell.target_mass_denominator}
''',
                )
            )
    record_text = "".join(text for _, text in sorted(records))
    return f'''schema_version = 1
design_id = "test-population-design"
provenance_status = "CALIBRATED"
notes = "Test-only complete static declarations; not an authenticity claim."
population_evidence_bundle_sha256 = "{evidence_bundle.bundle_sha256}"
population_evidence_result_sha256s = ["{calibration_result.evidence_sha256}", "{validation_result.evidence_sha256}"]
hamilton_recipe = "{EXACT_RATIONAL_HAMILTON_V1}"

[domains]
income_missing_policy = "REJECT"
household_missing_policy = "REJECT"
gaming_states = ["GAMER", "NON_GAMER"]
payer_history_states = ["EVER_PAYER", "NEVER_PAYER"]

[[domains.age_bands]]
ordinal = 0
age_band_id = "age.8-69"
age_min_inclusive = 8
age_max_exclusive = 70

[[domains.income_bands]]
ordinal = 0
jurisdiction_code = "UK"
income_band_id = "income.lower"
definition = "lower harmonized household-income category"
currency = "GBP"
period = "annual"
lower_unbounded = true
lower_bound_numerator = 0
lower_bound_denominator = 1
upper_unbounded = false
upper_bound_numerator = 50000
upper_bound_denominator = 1

[[domains.income_bands]]
ordinal = 1
jurisdiction_code = "UK"
income_band_id = "income.upper"
definition = "upper harmonized household-income category"
currency = "GBP"
period = "annual"
lower_unbounded = false
lower_bound_numerator = 50000
lower_bound_denominator = 1
upper_unbounded = true
upper_bound_numerator = 0
upper_bound_denominator = 1

[[domains.household_types]]
ordinal = 0
household_type_id = "household.all"
definition = "all declared source household types"

[[jurisdictions]]
jurisdiction_code = "UK"
target_population_count = 1000
calibration_binding_id = "{_CALIBRATION_BINDING_ID}"
calibration_target_population_id = "{_CALIBRATION_TARGET_ID}"
calibration_evidence_sha256 = "{calibration_result.evidence_sha256}"
validation_binding_id = "{_VALIDATION_BINDING_ID}"
validation_target_population_id = "{_VALIDATION_TARGET_ID}"
validation_evidence_sha256 = "{validation_result.evidence_sha256}"

[partition]
identity_namespace = "test.population.units"
record_id_recipe = "{CANONICAL_SOURCE_RECORD_ID_V1}"
cluster_id_recipe = "{CANONICAL_SOURCE_CLUSTER_ID_V1}"
role_assignment_recipe = "{SHA256_CLUSTER_THRESHOLD_V1}"
assignment_seed_sha256 = "{_PARTITION_SEED}"
calibration_threshold_numerator = 1
calibration_threshold_denominator = 2

{record_text}'''


def _write_complete_fixture(
    root: Path,
    *,
    validation_masses: tuple[Fraction, ...] | None = None,
) -> tuple[Path, object, tuple[object, ...], object]:
    evidence_bundle, evidence_results = _write_evidence(
        root,
        validation_masses=validation_masses,
    )
    design_path = root / "population_design.toml"
    design_path.write_text(
        _design_text(evidence_bundle, evidence_results),
        encoding="utf-8",
        newline="",
    )
    verification = load_and_verify_population_design_bundle(
        design_path,
        population_evidence_bundle=evidence_bundle,
        population_evidence_results=evidence_results,
    )
    return design_path, evidence_bundle, evidence_results, verification


class DefaultPopulationDesignTests(unittest.TestCase):
    def test_default_is_complete_illustrative_and_campaign_blocking(self) -> None:
        evidence_bundle, evidence_results = (
            load_and_verify_population_evidence_bundle(
                DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH
            )
        )
        verification = load_and_verify_population_design_bundle(
            population_evidence_bundle=evidence_bundle,
            population_evidence_results=evidence_results,
        )
        bundle = verification.bundle

        self.assertEqual(
            bundle.bundle_path,
            DEFAULT_POPULATION_DESIGN_BUNDLE_PATH.resolve(),
        )
        self.assertEqual(bundle.schema_version, POPULATION_DESIGN_SCHEMA_VERSION)
        self.assertIs(bundle.provenance_status, ProvenanceStatus.ILLUSTRATIVE)
        self.assertEqual(len(bundle.age_bands), 6)
        self.assertEqual(len(bundle.income_bands), 12)
        self.assertEqual(len(bundle.household_types), 3)
        self.assertEqual(
            tuple(item.jurisdiction_code for item in bundle.jurisdictions),
            ("BE", "JP", "KR", "UK"),
        )
        self.assertGreater(len(bundle.partition.records), 0)
        self.assertTrue(bundle.declaration_complete)
        self.assertFalse(bundle.campaign_ready)
        self.assertTrue(verification.evidence_reverified)
        self.assertFalse(verification.authenticity_verified)
        self.assertFalse(verification.heldout_ready)
        target = build_population_calibration_target(verification)
        self.assertEqual(len(target.cells), 864)
        self.assertEqual(target.total_population_count, 40_000)
        self.assertEqual(
            sum((cell.target_mass for cell in target.cells), Fraction()),
            Fraction(1, 1),
        )
        self.assertIn(
            "population_design_partition_source_unit_keys_unverified",
            bundle.campaign_blockers,
        )
        with self.assertRaisesRegex(
            PopulationDesignVerificationError,
            "static-only contract",
        ):
            bundle.validate_for_campaign()

        rebuilt = validate_population_design_snapshot(
            bundle.snapshot(),
            evidence_bundle.snapshot(),
            [result.snapshot() for result in evidence_results],
        )
        self.assertEqual(rebuilt, verification)


class CompletePopulationDesignTests(unittest.TestCase):
    def test_complete_design_reverifies_and_builds_calibration_only_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, verification = _write_complete_fixture(Path(directory))

            self.assertTrue(verification.bundle.declaration_complete)
            self.assertFalse(verification.bundle.campaign_ready)
            self.assertFalse(verification.heldout_ready)
            target = build_population_calibration_target(verification)
            self.assertEqual(len(target.cells), 8)
            self.assertEqual(
                [cell.cell_ordinal for cell in target.cells],
                list(range(8)),
            )
            self.assertEqual(
                sum((cell.target_mass for cell in target.cells), Fraction()),
                1,
            )
            self.assertFalse(hasattr(target, "validation_evidence_sha256s"))
            self.assertEqual(
                target.calibration_evidence_sha256s,
                (verification.evidence_results[0].evidence_sha256,),
            )

    def test_hamilton_is_exact_deterministic_and_uses_declared_ordinal_ties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, verification = _write_complete_fixture(Path(directory))
            target = build_population_calibration_target(verification)

            plan = apportion_population_hamilton(
                target,
                12,
                first_player_id=100,
            )
            self.assertEqual(
                [cell.sample_count for cell in plan.cells],
                [2, 2, 2, 2, 1, 1, 1, 1],
            )
            self.assertEqual(plan.last_player_id_exclusive, 112)
            self.assertEqual(
                sum(cell.sample_count for cell in plan.cells),
                12,
            )
            self.assertEqual(plan.cells[0].analysis_weight, Fraction(1, 16))
            self.assertEqual(plan.cells[-1].analysis_weight, Fraction(1, 8))
            self.assertEqual(plan.cells[0].expansion_weight, Fraction(125, 2))
            repeated = apportion_population_hamilton(
                target,
                12,
                first_player_id=100,
            )
            self.assertEqual(repeated, plan)
            self.assertEqual(
                validate_population_apportionment_snapshot(
                    plan.snapshot(),
                    target,
                ),
                plan,
            )

    def test_positive_cells_cannot_be_silently_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, verification = _write_complete_fixture(Path(directory))
            target = build_population_calibration_target(verification)
            with self.assertRaisesRegex(
                PopulationDesignVerificationError,
                "positive-mass cells unrepresented",
            ):
                apportion_population_hamilton(target, 7)

    def test_validation_values_are_not_allocator_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            _, _, _, first = _write_complete_fixture(Path(first_directory))
            changed_validation = (
                Fraction(1, 32),
                Fraction(3, 32),
                Fraction(1, 8),
                Fraction(1, 8),
                Fraction(1, 8),
                Fraction(1, 8),
                Fraction(3, 16),
                Fraction(3, 16),
            )
            _, _, _, second = _write_complete_fixture(
                Path(second_directory),
                validation_masses=changed_validation,
            )
            first_target = build_population_calibration_target(first)
            second_target = build_population_calibration_target(second)
            self.assertEqual(
                [cell.target_mass for cell in first_target.cells],
                [cell.target_mass for cell in second_target.cells],
            )
            self.assertEqual(
                [cell.sample_count for cell in apportion_population_hamilton(first_target, 12).cells],
                [cell.sample_count for cell in apportion_population_hamilton(second_target, 12).cells],
            )


class PopulationDesignAdversarialTests(unittest.TestCase):
    def test_verification_constructor_reopens_exact_bound_evidence(self) -> None:
        design = load_population_design_bundle(DEFAULT_POPULATION_DESIGN_BUNDLE_PATH)
        evidence, results = load_and_verify_population_evidence_bundle(
            DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH
        )
        forged_evidence = replace(evidence, bundle_sha256="f" * 64)
        payload = {
            "schema_version": POPULATION_DESIGN_SCHEMA_VERSION,
            "design_id": design.design_id,
            "design_bundle_sha256": design.bundle_sha256,
            "domain_sha256": design.domain_sha256,
            "partition_sha256": design.partition_sha256,
            "population_evidence_bundle_sha256": forged_evidence.bundle_sha256,
            "population_evidence_result_sha256s": [
                result.evidence_sha256 for result in results
            ],
            "evidence_reverified": True,
            "authenticity_verified": False,
            "heldout_ready": False,
            "campaign_ready": False,
        }
        verification_sha256 = sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        with self.assertRaises(PopulationDesignVerificationError):
            PopulationDesignVerification(
                bundle=design,
                evidence_bundle=forged_evidence,
                evidence_results=results,
                verification_sha256=verification_sha256,
            )

    def test_apportionment_constructor_recomputes_hamilton_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, verification = _write_complete_fixture(Path(directory))
            target = build_population_calibration_target(verification)
            plan = apportion_population_hamilton(target, 12)
            with self.assertRaisesRegex(
                PopulationDesignValidationError,
                "lineage differs from its exact calibration target",
            ):
                replace(plan, calibration_target_sha256="f" * 64)
            forged_counts = (1, 2, 2, 2, 1, 1, 1, 2)
            forged_cells = []
            for cell, count in zip(plan.cells, forged_counts, strict=True):
                mass = cell.calibration_cell.target_mass
                population = cell.calibration_cell.target_population
                analysis_weight = mass / count
                expansion_weight = population / count
                forged_cells.append(
                    replace(
                        cell,
                        sample_count=count,
                        analysis_weight_numerator=analysis_weight.numerator,
                        analysis_weight_denominator=analysis_weight.denominator,
                        expansion_weight_numerator=expansion_weight.numerator,
                        expansion_weight_denominator=expansion_weight.denominator,
                    )
                )
            payload = plan.attestation_payload()
            payload["cells"] = [cell.snapshot() for cell in forged_cells]
            forged_sha256 = sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()

            with self.assertRaisesRegex(
                PopulationDesignValidationError,
                "differ from exact deterministic Hamilton",
            ):
                PopulationApportionmentPlan(
                    calibration_target=target,
                    recipe=plan.recipe,
                    calibration_target_sha256=plan.calibration_target_sha256,
                    design_id=plan.design_id,
                    design_bundle_sha256=plan.design_bundle_sha256,
                    domain_sha256=plan.domain_sha256,
                    player_count=plan.player_count,
                    first_player_id=plan.first_player_id,
                    total_population_count=plan.total_population_count,
                    cells=tuple(forged_cells),
                    apportionment_sha256=forged_sha256,
                )

    def test_strict_toml_rejects_boolean_alias_unknown_fields_and_domain_reorder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path, _, _, _ = _write_complete_fixture(root)
            original = design_path.read_text(encoding="utf-8")
            variants = (
                original.replace("schema_version = 1", "schema_version = true", 1),
                original.replace(
                    'design_id = "test-population-design"',
                    'design_id = "test-population-design"\nunknown = "laundered"',
                    1,
                ),
                original.replace(
                    'gaming_states = ["GAMER", "NON_GAMER"]',
                    'gaming_states = ["NON_GAMER", "GAMER"]',
                    1,
                ),
            )
            for index, text in enumerate(variants):
                candidate = root / f"invalid-{index}.toml"
                candidate.write_text(text, encoding="utf-8", newline="")
                with self.subTest(index=index):
                    with self.assertRaises(PopulationDesignValidationError):
                        load_population_design_bundle(candidate)

    def test_income_currency_gap_and_incomplete_cartesian_evidence_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path, evidence_bundle, evidence_results, _ = (
                _write_complete_fixture(root)
            )
            original = design_path.read_text(encoding="utf-8")
            variants = (
                original.replace(
                    "upper_bound_numerator = 50000",
                    "upper_bound_numerator = 49999",
                    1,
                ),
                original.replace('currency = "GBP"', 'currency = "EUR"', 2),
            )
            for index, text in enumerate(variants):
                candidate = root / f"income-invalid-{index}.toml"
                candidate.write_text(text, encoding="utf-8", newline="")
                with self.subTest(index=index):
                    with self.assertRaises(PopulationDesignValidationError):
                        load_and_verify_population_design_bundle(
                            candidate,
                            population_evidence_bundle=evidence_bundle,
                            population_evidence_results=evidence_results,
                        )

            # Relabeling one declared evidence cell leaves a duplicate and one
            # missing Cartesian cell; the exact evidence layer rejects it first.
            artifact_path = root / "population_artifacts" / "joint.csv"
            content = artifact_path.read_text(encoding="utf-8")
            artifact_path.write_text(
                content.replace("income.upper", "income.lower", 1),
                encoding="utf-8",
                newline="",
            )
            with self.assertRaises(PopulationDesignVerificationError):
                verify_population_design_bundle(
                    load_population_design_bundle(design_path),
                    population_evidence_bundle=evidence_bundle,
                    population_evidence_results=evidence_results,
                )

    def test_partition_role_weight_and_identity_attacks_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path, evidence_bundle, evidence_results, _ = (
                _write_complete_fixture(root)
            )
            original = design_path.read_text(encoding="utf-8")
            first_record = next(
                line.split('"')[1]
                for line in original.splitlines()
                if line.startswith("record_identity_sha256")
            )
            record_lines = [
                line.split('"')[1]
                for line in original.splitlines()
                if line.startswith("record_identity_sha256")
            ]
            variants = (
                original.replace(
                    'estimand_role = "CALIBRATION"',
                    'estimand_role = "VALIDATION"',
                    1,
                ),
                original.replace(
                    "record_weight_numerator = 1",
                    "record_weight_numerator = 2",
                    1,
                ),
                original.replace(record_lines[1], first_record, 1),
            )
            for index, text in enumerate(variants):
                candidate = root / f"partition-invalid-{index}.toml"
                candidate.write_text(text, encoding="utf-8", newline="")
                with self.subTest(index=index):
                    with self.assertRaises(PopulationDesignValidationError):
                        load_and_verify_population_design_bundle(
                            candidate,
                            population_evidence_bundle=evidence_bundle,
                            population_evidence_results=evidence_results,
                        )

    def test_snapshot_and_apportionment_tampering_fail_even_with_boolean_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, evidence_bundle, evidence_results, verification = (
                _write_complete_fixture(Path(directory))
            )
            forged = deepcopy(verification.bundle.snapshot())
            forged["declaration_complete"] = 1
            with self.assertRaises(PopulationDesignValidationError):
                validate_population_design_snapshot(
                    forged,
                    evidence_bundle.snapshot(),
                    [result.snapshot() for result in evidence_results],
                )

            target = build_population_calibration_target(verification)
            plan = apportion_population_hamilton(target, 12)
            forged_plan = deepcopy(plan.snapshot())
            forged_plan["cells"][0]["sample_count"] = 3
            with self.assertRaises(PopulationDesignValidationError):
                validate_population_apportionment_snapshot(forged_plan, target)

            numeric_alias = deepcopy(plan.snapshot())
            numeric_alias["player_count"] = True
            with self.assertRaises(PopulationDesignValidationError):
                validate_population_apportionment_snapshot(numeric_alias, target)

    def test_design_or_evidence_byte_changes_invalidate_retained_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_path, evidence_bundle, evidence_results, verification = (
                _write_complete_fixture(root)
            )
            design_path.write_text(
                design_path.read_text(encoding="utf-8").replace(
                    "Test-only complete static declarations",
                    "Changed complete static declarations",
                    1,
                ),
                encoding="utf-8",
                newline="",
            )
            with self.assertRaisesRegex(
                PopulationDesignVerificationError,
                "metadata no longer match",
            ):
                verify_population_design_bundle(
                    verification.bundle,
                    population_evidence_bundle=evidence_bundle,
                    population_evidence_results=evidence_results,
                )


if __name__ == "__main__":
    unittest.main()
