from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.data.population_evidence import (
    DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH,
    EXACT_CSV_JOINT_POPULATION_INTERPRETER_V1,
    POPULATION_CELL_CSV_COLUMNS,
    POPULATION_EVIDENCE_SCHEMA_VERSION,
    PopulationEstimandRole,
    PopulationEvidenceCell,
    PopulationEvidenceSignatureStatus,
    PopulationEvidenceValidationError,
    PopulationEvidenceVerificationError,
    PopulationGamingState,
    PopulationPayerHistoryState,
    exact_csv_joint_population_recipe_json,
    load_and_verify_population_evidence_bundle,
    load_population_evidence_bundle,
    validate_population_evidence_snapshot,
    verify_population_evidence_bundle,
)
from microtx_sim.types import ProvenanceStatus


_SOURCE_REGISTRY_SHA256 = "a" * 64
_SOURCE_IDS = ("TEST_GAMING", "TEST_HOUSEHOLD")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _recipe(role: PopulationEstimandRole) -> str:
    return exact_csv_joint_population_recipe_json(
        target_population_id=f"uk.mobile_players.{role.value.lower()}",
        jurisdiction_code="UK",
        estimand_role=role,
    )


def _rows(
    *,
    roles: tuple[PopulationEstimandRole, ...],
    masses: tuple[tuple[str, str], tuple[str, str]] = (("1", "3"), ("2", "3")),
) -> list[list[str]]:
    rows: list[list[str]] = []
    for role in roles:
        selector = [
            f"uk.mobile_players.{role.value.lower()}",
            "UK",
            role.value,
        ]
        rows.extend(
            (
                [
                    *selector,
                    "cell.01",
                    "8",
                    "18",
                    "income.lower",
                    "household.children",
                    "GAMER",
                    "NEVER_PAYER",
                    *masses[0],
                ],
                [
                    *selector,
                    "cell.02",
                    "18",
                    "70",
                    "income.upper",
                    "household.adults",
                    "NON_GAMER",
                    "EVER_PAYER",
                    *masses[1],
                ],
            )
        )
    return rows


def _csv_bytes(rows: list[list[str]]) -> bytes:
    # Fixture values intentionally need no CSV quoting; production parsing still
    # uses Python's strict CSV implementation and validates the exact width.
    lines = [",".join(POPULATION_CELL_CSV_COLUMNS)]
    lines.extend(",".join(row) for row in rows)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _bundle_text(
    artifact: bytes,
    *,
    roles: tuple[PopulationEstimandRole, ...],
    relative_path: str = "joint_cells.csv",
    recipes: dict[PopulationEstimandRole, str] | None = None,
    top_extra: str = "",
    artifact_extra: str = "",
    binding_extra: str = "",
    signature_extra: str = "",
) -> str:
    bindings: list[str] = []
    recipes = recipes or {role: _recipe(role) for role in roles}
    for role in roles:
        slug = role.value.lower()
        bindings.append(
            f'''[[bindings]]
binding_id = "population.{slug}.uk"
artifact_id = "joint.population"
target_population_id = "uk.mobile_players.{slug}"
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
household_income_equivalisation = "modified OECD scale"
household_definition = "shared dwelling and budget"
gaming_definition = "mobile-game play in reference period"
payer_definition = "ever paid for mobile-game content"
zero_spender_treatment = "retained as never-payer cells"
estimand_role = "{role.value}"
status = "CALIBRATED"
source_ids = ["{_SOURCE_IDS[0]}", "{_SOURCE_IDS[1]}"]
retrieved_on = 2026-01-15
recipe_json = {json.dumps(recipes[role])}
{binding_extra}'''
        )
    return f'''schema_version = 1
bundle_id = "test-population-evidence"
provenance_status = "CALIBRATED"
source_registry_sha256 = "{_SOURCE_REGISTRY_SHA256}"
artifact_root = "population_artifacts"
notes = "Test-only exact joint population cells; not substantive evidence."
{top_extra}
[[artifacts]]
artifact_id = "joint.population"
relative_path = {json.dumps(relative_path)}
media_type = "text/csv"
sha256 = "{sha256(artifact).hexdigest()}"
byte_length = {len(artifact)}
{artifact_extra}
{"".join(bindings)}
[signature]
status = "MISSING"
algorithm = "NONE"
key_id = ""
value = ""
{signature_extra}'''


def _write_fixture(
    root: Path,
    *,
    roles: tuple[PopulationEstimandRole, ...] = (
        PopulationEstimandRole.CALIBRATION,
        PopulationEstimandRole.VALIDATION,
    ),
    rows: list[list[str]] | None = None,
    artifact: bytes | None = None,
    **bundle_options: object,
) -> tuple[Path, Path]:
    if artifact is None:
        artifact = _csv_bytes(_rows(roles=roles) if rows is None else rows)
    artifact_root = root / "population_artifacts"
    artifact_root.mkdir()
    artifact_path = artifact_root / "joint_cells.csv"
    artifact_path.write_bytes(artifact)
    bundle_path = root / "population_bundle.toml"
    bundle_path.write_text(
        _bundle_text(artifact, roles=roles, **bundle_options),
        encoding="utf-8",
        newline="",
    )
    return bundle_path, artifact_path


class DefaultPopulationEvidenceTests(unittest.TestCase):
    def test_default_bundle_is_complete_illustrative_and_campaign_blocking(self) -> None:
        source_registry = DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH.with_name(
            "sources.toml"
        )
        source_registry_sha256 = sha256(source_registry.read_bytes()).hexdigest()
        bundle, results = load_and_verify_population_evidence_bundle(
            expected_source_registry_sha256=source_registry_sha256,
        )

        self.assertEqual(
            bundle.bundle_path,
            DEFAULT_POPULATION_EVIDENCE_BUNDLE_PATH.resolve(),
        )
        self.assertEqual(bundle.schema_version, POPULATION_EVIDENCE_SCHEMA_VERSION)
        self.assertIs(bundle.provenance_status, ProvenanceStatus.ILLUSTRATIVE)
        self.assertEqual(bundle.source_registry_sha256, source_registry_sha256)
        self.assertEqual(len(bundle.artifacts), 1)
        self.assertEqual(len(bundle.bindings), 8)
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(len(result.cells) for result in results), 1_728)
        self.assertIs(
            bundle.signature.status,
            PopulationEvidenceSignatureStatus.MISSING,
        )
        self.assertFalse(bundle.campaign_ready)
        self.assertIn(
            "population_evidence_bundle_status=ILLUSTRATIVE",
            bundle.campaign_blockers,
        )
        self.assertIn(
            "population_evidence_bundle_signature_missing",
            bundle.campaign_blockers,
        )
        self.assertTrue(
            any(
                blocker.startswith("population_evidence_non_calibrated_bindings=")
                for blocker in bundle.campaign_blockers
            )
        )
        with self.assertRaisesRegex(
            PopulationEvidenceVerificationError,
            "schema v1 is not campaign-ready",
        ):
            bundle.validate_for_campaign()

        rebuilt_bundle, rebuilt_results = validate_population_evidence_snapshot(
            bundle.snapshot(),
            [result.snapshot() for result in results],
        )
        self.assertEqual(rebuilt_bundle, bundle)
        self.assertEqual(rebuilt_results, results)

    def test_absent_snapshot_allows_no_results_only(self) -> None:
        self.assertEqual(validate_population_evidence_snapshot(None, []), (None, ()))
        with self.assertRaisesRegex(
            PopulationEvidenceValidationError,
            "require a bundle snapshot",
        ):
            validate_population_evidence_snapshot(None, [{}])


class ExactPopulationEvidenceTests(unittest.TestCase):
    def test_populated_bundle_extracts_exact_joint_cells_and_keeps_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(Path(directory))
            bundle, results = load_and_verify_population_evidence_bundle(
                bundle_path,
                expected_source_registry_sha256=_SOURCE_REGISTRY_SHA256,
            )

            self.assertEqual(
                {result.estimand_role for result in results},
                {
                    PopulationEstimandRole.CALIBRATION,
                    PopulationEstimandRole.VALIDATION,
                },
            )
            self.assertEqual(
                tuple(result.binding_id for result in results),
                tuple(binding.binding_id for binding in bundle.bindings),
            )
            for binding, result in zip(bundle.bindings, results, strict=True):
                self.assertEqual(binding.source_ids, _SOURCE_IDS)
                self.assertEqual(binding.retrieved_on.isoformat(), "2026-01-15")
                self.assertIs(binding.status, ProvenanceStatus.CALIBRATED)
                self.assertEqual(result.binding_sha256, binding.binding_sha256)
                self.assertEqual(result.bundle_sha256, bundle.bundle_sha256)
                self.assertEqual(
                    result.source_registry_sha256,
                    _SOURCE_REGISTRY_SHA256,
                )
                self.assertEqual(result.artifact_sha256, bundle.artifacts[0].sha256)
                self.assertEqual(
                    sum((cell.target_mass for cell in result.cells), Fraction()),
                    Fraction(1, 1),
                )
                self.assertTrue(
                    all(type(cell) is PopulationEvidenceCell for cell in result.cells)
                )
                self.assertIs(
                    result.cells[0].gaming_state,
                    PopulationGamingState.GAMER,
                )
                self.assertIs(
                    result.cells[0].payer_history_state,
                    PopulationPayerHistoryState.NEVER_PAYER,
                )
                self.assertEqual(
                    result.evidence_sha256,
                    sha256(
                        _canonical_json(result.attestation_payload()).encode("utf-8")
                    ).hexdigest(),
                )

            self.assertFalse(bundle.campaign_ready)
            self.assertEqual(
                bundle.campaign_blockers,
                ("population_evidence_bundle_signature_missing",),
            )
            rebuilt_bundle, rebuilt_results = validate_population_evidence_snapshot(
                bundle.snapshot(),
                [result.snapshot() for result in results],
            )
            self.assertEqual(rebuilt_bundle, bundle)
            self.assertEqual(rebuilt_results, results)

    def test_large_rational_integers_are_lossless(self) -> None:
        huge = 1 << 200
        denominator = 2 * huge + 1
        masses = (
            (str(huge), str(denominator)),
            (str(huge + 1), str(denominator)),
        )
        roles = (PopulationEstimandRole.CALIBRATION,)
        with tempfile.TemporaryDirectory() as directory:
            rows = _rows(roles=roles, masses=masses)
            bundle_path, _ = _write_fixture(
                Path(directory),
                roles=roles,
                rows=rows,
            )
            bundle, results = load_and_verify_population_evidence_bundle(bundle_path)
            cell = results[0].cells[0]
            snapshot = cell.snapshot()

            self.assertGreater(cell.target_mass_denominator, 2**53)
            self.assertEqual(cell.target_mass_numerator, huge)
            self.assertEqual(snapshot["target_mass_numerator_decimal"], str(huge))
            self.assertEqual(
                snapshot["target_mass_denominator_decimal"],
                str(denominator),
            )
            rebuilt_bundle, rebuilt_results = validate_population_evidence_snapshot(
                bundle.snapshot(),
                [results[0].snapshot()],
            )
            self.assertEqual(rebuilt_bundle, bundle)
            self.assertEqual(rebuilt_results, results)

    def test_recipe_is_canonical_whitelisted_and_metadata_bound(self) -> None:
        recipe = _recipe(PopulationEstimandRole.CALIBRATION)
        parsed = json.loads(recipe)
        self.assertEqual(
            parsed["interpreter"],
            EXACT_CSV_JOINT_POPULATION_INTERPRETER_V1,
        )
        self.assertEqual(recipe, _canonical_json(parsed))

        invalid_recipes: list[str] = []
        changed = deepcopy(parsed)
        changed["interpreter"] = "arbitrary_python/1"
        invalid_recipes.append(_canonical_json(changed))
        changed = deepcopy(parsed)
        changed["row_match"]["jurisdiction_code"] = "BE"
        invalid_recipes.append(_canonical_json(changed))
        changed = deepcopy(parsed)
        changed["unexpected"] = True
        invalid_recipes.append(_canonical_json(changed))
        invalid_recipes.append(" " + recipe)

        roles = (PopulationEstimandRole.CALIBRATION,)
        for invalid_recipe in invalid_recipes:
            with (
                self.subTest(recipe=invalid_recipe[:60]),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(
                    Path(directory),
                    roles=roles,
                    recipes={PopulationEstimandRole.CALIBRATION: invalid_recipe},
                )
                with self.assertRaises(PopulationEvidenceValidationError):
                    load_population_evidence_bundle(bundle_path)

    def test_expected_source_registry_digest_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(Path(directory))
            with self.assertRaisesRegex(
                PopulationEvidenceVerificationError,
                "expected source catalogue",
            ):
                load_population_evidence_bundle(
                    bundle_path,
                    expected_source_registry_sha256="b" * 64,
                )
            bundle = load_population_evidence_bundle(bundle_path)
            with self.assertRaisesRegex(
                PopulationEvidenceVerificationError,
                "expected source catalogue",
            ):
                verify_population_evidence_bundle(
                    bundle,
                    expected_source_registry_sha256="b" * 64,
                )

    def test_bundle_schema_uses_exact_keys_and_strict_types(self) -> None:
        roles = (PopulationEstimandRole.CALIBRATION,)
        mutations = {
            "boolean schema": ("schema_version = 1", "schema_version = true"),
            "unknown key": (
                'bundle_id = "test-population-evidence"',
                'bundle_id = "test-population-evidence"\nunknown = "field"',
            ),
            "quoted date": (
                "reference_period_start = 2025-01-01",
                'reference_period_start = "2025-01-01"',
            ),
            "boolean age": ("age_min_inclusive = 8", "age_min_inclusive = true"),
            "scalar source ids": (
                'source_ids = ["TEST_GAMING", "TEST_HOUSEHOLD"]',
                'source_ids = "TEST_GAMING"',
            ),
        }
        for label, (old, new) in mutations.items():
            with (
                self.subTest(label=label),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(Path(directory), roles=roles)
                text = bundle_path.read_text(encoding="utf-8")
                bundle_path.write_text(
                    text.replace(old, new, 1),
                    encoding="utf-8",
                    newline="",
                )
                with self.assertRaises(PopulationEvidenceValidationError):
                    load_population_evidence_bundle(bundle_path)

    def test_csv_rejects_numeric_laundering_and_unreduced_mass(self) -> None:
        roles = (PopulationEstimandRole.CALIBRATION,)
        invalid_values = ("0.5", "1e0", "nan", "true", "+1", "01", "-1")
        for value in invalid_values:
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                rows = _rows(roles=roles)
                rows[0][-2] = value
                bundle_path, _ = _write_fixture(
                    Path(directory),
                    roles=roles,
                    rows=rows,
                )
                bundle = load_population_evidence_bundle(bundle_path)
                with self.assertRaises(PopulationEvidenceVerificationError):
                    verify_population_evidence_bundle(bundle)

        with tempfile.TemporaryDirectory() as directory:
            rows = _rows(roles=roles, masses=(("0", "1"), ("1", "1")))
            bundle_path, _ = _write_fixture(Path(directory), roles=roles, rows=rows)
            _bundle, results = load_and_verify_population_evidence_bundle(bundle_path)
            self.assertEqual(results[0].cells[0].target_mass, Fraction(0, 1))

        with tempfile.TemporaryDirectory() as directory:
            rows = _rows(roles=roles)
            rows[0][-2:] = ["2", "6"]
            bundle_path, _ = _write_fixture(Path(directory), roles=roles, rows=rows)
            bundle = load_population_evidence_bundle(bundle_path)
            with self.assertRaisesRegex(
                PopulationEvidenceValidationError,
                "lowest terms",
            ):
                verify_population_evidence_bundle(bundle)

    def test_cells_must_be_unique_cover_age_scope_and_sum_exactly_one(self) -> None:
        roles = (PopulationEstimandRole.CALIBRATION,)

        duplicate_id = _rows(roles=roles)
        duplicate_id[1][3] = duplicate_id[0][3]
        duplicate_semantic = _rows(roles=roles)
        duplicate_semantic[1][4:10] = duplicate_semantic[0][4:10]
        overlapping_stratum = _rows(roles=roles)
        overlapping_stratum[1][4:6] = ["15", "70"]
        overlapping_stratum[1][6:10] = overlapping_stratum[0][6:10]
        gap = _rows(roles=roles)
        gap[1][4] = "19"
        outside = _rows(roles=roles)
        outside[0][4] = "7"
        wrong_total = _rows(roles=roles)
        wrong_total[1][-2:] = ["1", "3"]
        cases = {
            "ids repeat": duplicate_id,
            "semantic cell": duplicate_semantic,
            "overlap in age": overlapping_stratum,
            "cover every age": gap,
            "outside its age scope": outside,
            "sum exactly to 1": wrong_total,
        }
        for expected, rows in cases.items():
            with (
                self.subTest(expected=expected),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(
                    Path(directory),
                    roles=roles,
                    rows=rows,
                )
                bundle = load_population_evidence_bundle(bundle_path)
                with self.assertRaisesRegex(
                    PopulationEvidenceValidationError,
                    expected,
                ):
                    verify_population_evidence_bundle(bundle)

    def test_every_csv_row_must_be_owned_by_exactly_one_binding(self) -> None:
        roles = (PopulationEstimandRole.CALIBRATION,)
        rows = _rows(roles=roles)
        unowned = list(rows[0])
        unowned[0] = "uk.mobile_players.validation"
        unowned[2] = "VALIDATION"
        rows.append(unowned)
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(Path(directory), roles=roles, rows=rows)
            bundle = load_population_evidence_bundle(bundle_path)
            with self.assertRaisesRegex(
                PopulationEvidenceVerificationError,
                "owned by exactly one binding",
            ):
                verify_population_evidence_bundle(bundle)

    def test_artifact_exact_bytes_and_containment_are_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, artifact_path = _write_fixture(Path(directory))
            bundle = load_population_evidence_bundle(bundle_path)
            artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")
            with self.assertRaisesRegex(
                PopulationEvidenceVerificationError,
                "byte length changed",
            ):
                verify_population_evidence_bundle(bundle)

        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(
                Path(directory),
                relative_path="../joint_cells.csv",
            )
            with self.assertRaisesRegex(
                PopulationEvidenceValidationError,
                "parent components",
            ):
                load_population_evidence_bundle(bundle_path)

    def test_artifact_symlinks_are_never_followed(self) -> None:
        roles = (PopulationEstimandRole.CALIBRATION,)
        artifact = _csv_bytes(_rows(roles=roles))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path, _ = _write_fixture(
                root,
                roles=roles,
                artifact=artifact,
                relative_path="linked.csv",
            )
            outside = root / "outside.csv"
            outside.write_bytes(artifact)
            linked = root / "population_artifacts" / "linked.csv"
            try:
                linked.symlink_to(outside)
            except OSError:
                self.skipTest("this host does not permit test symlink creation")

            bundle = load_population_evidence_bundle(bundle_path)
            with self.assertRaisesRegex(
                PopulationEvidenceVerificationError,
                "symlink or reparse point",
            ):
                verify_population_evidence_bundle(bundle)

    def test_csv_is_exact_utf8_with_exact_header_and_no_multiline_cells(self) -> None:
        roles = (PopulationEstimandRole.CALIBRATION,)
        valid = _csv_bytes(_rows(roles=roles))
        extra_header = valid.replace(
            b"target_mass_denominator\n",
            b"target_mass_denominator,unexpected\n",
            1,
        )
        multiline = valid.replace(b"income.lower", b'"income.\nlower"', 1)
        invalid_artifacts = (
            b"\xef\xbb\xbf" + valid,
            valid + b"\xff",
            valid.replace(b"income.lower", b"income.\x00lower", 1),
            extra_header,
            multiline,
        )
        for artifact in invalid_artifacts:
            with (
                self.subTest(prefix=artifact[:20]),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(
                    Path(directory),
                    roles=roles,
                    artifact=artifact,
                )
                bundle = load_population_evidence_bundle(bundle_path)
                with self.assertRaises(PopulationEvidenceVerificationError):
                    verify_population_evidence_bundle(bundle)

    def test_snapshot_validation_rejects_boolean_and_coordinated_forgeries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(Path(directory))
            bundle, results = load_and_verify_population_evidence_bundle(bundle_path)
            bundle_snapshot = bundle.snapshot()
            result_snapshots = [result.snapshot() for result in results]

            for forged_value in (True, 0):
                forged_bundle = deepcopy(bundle_snapshot)
                forged_bundle["campaign_ready"] = forged_value
                with self.assertRaisesRegex(
                    PopulationEvidenceValidationError,
                    "must be false",
                ):
                    validate_population_evidence_snapshot(
                        forged_bundle,
                        result_snapshots,
                    )

            forged_blockers = deepcopy(bundle_snapshot)
            forged_blockers["campaign_blockers"] = []
            with self.assertRaisesRegex(
                PopulationEvidenceValidationError,
                "not canonical",
            ):
                validate_population_evidence_snapshot(
                    forged_blockers,
                    result_snapshots,
                )

            forged_decimal = deepcopy(result_snapshots)
            forged_decimal[0]["cells"][0]["target_mass_numerator_decimal"] = "1.0"
            with self.assertRaisesRegex(
                PopulationEvidenceValidationError,
                "decimal mirrors",
            ):
                validate_population_evidence_snapshot(
                    bundle_snapshot,
                    forged_decimal,
                )

            coordinated_results = deepcopy(result_snapshots)
            changed = coordinated_results[0]
            changed["cells"][0]["target_mass_numerator"] = 1
            changed["cells"][0]["target_mass_denominator"] = 4
            changed["cells"][0]["target_mass_denominator_decimal"] = "4"
            changed["cells"][1]["target_mass_numerator"] = 3
            changed["cells"][1]["target_mass_denominator"] = 4
            changed["cells"][1]["target_mass_numerator_decimal"] = "3"
            changed["cells"][1]["target_mass_denominator_decimal"] = "4"
            changed["cells_sha256"] = sha256(
                _canonical_json(changed["cells"]).encode("utf-8")
            ).hexdigest()
            payload = dict(changed)
            payload.pop("evidence_sha256")
            changed["evidence_sha256"] = sha256(
                _canonical_json(payload).encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(
                PopulationEvidenceValidationError,
                "do not match re-extracted bytes",
            ):
                validate_population_evidence_snapshot(
                    bundle_snapshot,
                    coordinated_results,
                )

            coordinated_bundle = deepcopy(bundle_snapshot)
            coordinated_results = deepcopy(result_snapshots)
            coordinated_bundle["bundle_sha256"] = "b" * 64
            for result in coordinated_results:
                result["bundle_sha256"] = "b" * 64
                payload = dict(result)
                payload.pop("evidence_sha256")
                result["evidence_sha256"] = sha256(
                    _canonical_json(payload).encode("utf-8")
                ).hexdigest()
            with self.assertRaisesRegex(
                PopulationEvidenceValidationError,
                "metadata no longer match",
            ):
                validate_population_evidence_snapshot(
                    coordinated_bundle,
                    coordinated_results,
                )


if __name__ == "__main__":
    unittest.main()
