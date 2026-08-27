from __future__ import annotations

from copy import deepcopy
from datetime import date
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import microtx_sim.data.rate_evidence as rate_evidence_module
from microtx_sim.data.rate_evidence import (
    DEFAULT_RATE_EVIDENCE_BUNDLE_PATH,
    EXACT_CSV_INTERPRETER_V1,
    RATE_EVIDENCE_SCHEMA_VERSION,
    RateEvidenceMethod,
    RateEvidenceSignatureStatus,
    RateEvidenceValidationError,
    RateEvidenceVerificationError,
    exact_csv_rational_recipe_json,
    load_and_verify_rate_evidence_bundle,
    load_rate_evidence_bundle,
    validate_rate_evidence_snapshot,
    verify_rate_evidence_bundle,
)
from microtx_sim.types import ProvenanceStatus


_SOURCE_ID = "TEST_RATE_SOURCE"
_PERIOD_START = date(2025, 1, 1)
_PERIOD_END = date(2025, 12, 31)
_RETRIEVED_ON = date(2026, 1, 15)
_SOURCE_REGISTRY_SHA256 = "a" * 64
_CSV_HEADER = (
    "source_id,jurisdiction_code,source_currency,target_currency,method,"
    "rate_period_start,rate_period_end,retrieved_on,rate_numerator,"
    "rate_denominator\n"
)
_CSV_ROW = (
    "TEST_RATE_SOURCE,UK,GBP,EUR,FX,2025-01-01,2025-12-31,"
    "2026-01-15,2469,2000\n"
)


def _recipe() -> str:
    return exact_csv_rational_recipe_json(
        source_id=_SOURCE_ID,
        jurisdiction_code="UK",
        source_currency="GBP",
        target_currency="EUR",
        method=RateEvidenceMethod.FX,
        rate_period_start=_PERIOD_START,
        rate_period_end=_PERIOD_END,
        retrieved_on=_RETRIEVED_ON,
    )


def _bundle_text(
    artifact: bytes,
    *,
    recipe_json: str | None = None,
    relative_path: str = "rates.csv",
    rate_numerator: str = "2469",
    rate_denominator: str = "2000",
    artifact_extra: str = "",
    binding_extra: str = "",
    signature_extra: str = "",
    top_extra: str = "",
) -> str:
    recipe_json = _recipe() if recipe_json is None else recipe_json
    return f'''schema_version = 1
bundle_id = "test-rate-evidence"
provenance_status = "CALIBRATED"
source_registry_sha256 = "{_SOURCE_REGISTRY_SHA256}"
artifact_root = "rate_artifacts"
notes = "Test-only exact source artifact; not substantive evidence."
{top_extra}
[[artifacts]]
artifact_id = "test-rates"
relative_path = {json.dumps(relative_path)}
media_type = "text/csv"
sha256 = "{sha256(artifact).hexdigest()}"
byte_length = {len(artifact)}
{artifact_extra}
[[bindings]]
binding_id = "uk-gbp-eur-2025"
artifact_id = "test-rates"
source_id = "{_SOURCE_ID}"
jurisdiction_code = "UK"
source_currency = "GBP"
target_currency = "EUR"
method = "FX"
rate_period_start = 2025-01-01
rate_period_end = 2025-12-31
retrieved_on = 2026-01-15
rate_numerator = {rate_numerator}
rate_denominator = {rate_denominator}
recipe_json = {json.dumps(recipe_json)}
{binding_extra}
[signature]
status = "MISSING"
algorithm = "NONE"
key_id = ""
value = ""
{signature_extra}'''


def _write_fixture(
    root: Path,
    *,
    artifact: bytes | None = None,
    **bundle_options: str,
) -> tuple[Path, Path]:
    artifact = (
        (_CSV_HEADER + _CSV_ROW).encode("utf-8")
        if artifact is None
        else artifact
    )
    artifact_root = root / "rate_artifacts"
    artifact_root.mkdir()
    artifact_path = artifact_root / "rates.csv"
    artifact_path.write_bytes(artifact)
    bundle_path = root / "source_bundle.toml"
    bundle_path.write_text(
        _bundle_text(artifact, **bundle_options),
        encoding="utf-8",
        newline="",
    )
    return bundle_path, artifact_path


class DefaultRateEvidenceTests(unittest.TestCase):
    def test_default_bundle_is_empty_and_campaign_blocking(self) -> None:
        source_registry = DEFAULT_RATE_EVIDENCE_BUNDLE_PATH.with_name(
            "sources.toml"
        )
        source_registry_sha256 = sha256(source_registry.read_bytes()).hexdigest()
        bundle, results = load_and_verify_rate_evidence_bundle(
            required_source_registry_sha256=source_registry_sha256,
        )

        self.assertEqual(
            bundle.bundle_path,
            DEFAULT_RATE_EVIDENCE_BUNDLE_PATH.resolve(),
        )
        self.assertEqual(bundle.schema_version, RATE_EVIDENCE_SCHEMA_VERSION)
        self.assertIs(bundle.provenance_status, ProvenanceStatus.ILLUSTRATIVE)
        self.assertEqual(bundle.source_registry_sha256, source_registry_sha256)
        self.assertEqual(bundle.artifacts, ())
        self.assertEqual(bundle.bindings, ())
        self.assertIs(
            bundle.signature.status,
            RateEvidenceSignatureStatus.MISSING,
        )
        self.assertEqual(results, ())
        self.assertFalse(bundle.campaign_ready)
        self.assertEqual(
            bundle.campaign_blockers,
            (
                "rate_evidence_bundle_status=ILLUSTRATIVE",
                "rate_evidence_bundle_signature_missing",
                "rate_evidence_bundle_empty",
            ),
        )
        with self.assertRaisesRegex(
            RateEvidenceVerificationError,
            "schema v1 is not campaign-ready",
        ):
            bundle.validate_for_campaign()

        noncanonical_path = deepcopy(bundle.snapshot())
        path = bundle.bundle_path
        noncanonical_path["bundle_path"] = str(
            path.parent / ".." / path.parent.name / path.name
        )
        with self.assertRaisesRegex(
            RateEvidenceValidationError,
            "lexically canonical",
        ):
            validate_rate_evidence_snapshot(noncanonical_path, [])


class ExactRateEvidenceTests(unittest.TestCase):
    def test_exact_csv_binding_verifies_content_and_integer_rational(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(Path(directory))

            bundle = load_rate_evidence_bundle(bundle_path)
            results = verify_rate_evidence_bundle(bundle)

        self.assertEqual(len(results), 1)
        binding = bundle.bindings[0]
        result = results[0]
        self.assertEqual(binding.recipe_sha256, sha256(_recipe().encode()).hexdigest())
        self.assertEqual(binding.rate, Fraction(2469, 2000))
        self.assertEqual(result.rate, Fraction(2469, 2000))
        self.assertEqual(result.binding_sha256, binding.binding_sha256)
        self.assertEqual(result.bundle_sha256, bundle.bundle_sha256)
        self.assertEqual(result.source_registry_sha256, _SOURCE_REGISTRY_SHA256)
        self.assertEqual(result.artifact_sha256, bundle.artifacts[0].sha256)
        self.assertEqual(
            result.evidence_sha256,
            sha256(
                json.dumps(
                    result.attestation_payload(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        self.assertFalse(bundle.campaign_ready)
        self.assertEqual(
            bundle.campaign_blockers,
            ("rate_evidence_bundle_signature_missing",),
        )

    def test_large_exact_rates_have_lossless_decimal_mirrors(self) -> None:
        numerator = (1 << 60) + 1
        artifact = (
            _CSV_HEADER
            + _CSV_ROW.replace("2469,2000", f"{numerator},1")
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(
                Path(directory),
                artifact=artifact,
                rate_numerator=str(numerator),
                rate_denominator="1",
            )
            bundle, results = load_and_verify_rate_evidence_bundle(bundle_path)

        binding_snapshot = bundle.bindings[0].snapshot()
        result_snapshot = results[0].snapshot()
        self.assertEqual(binding_snapshot["rate_numerator"], numerator)
        self.assertEqual(
            binding_snapshot["rate_numerator_decimal"], str(numerator)
        )
        self.assertEqual(result_snapshot["rate_numerator"], numerator)
        self.assertEqual(
            result_snapshot["rate_numerator_decimal"], str(numerator)
        )

    def test_serialized_bundle_and_results_are_rebuilt_canonically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(Path(directory))
            bundle, results = load_and_verify_rate_evidence_bundle(bundle_path)

            rebuilt_bundle, rebuilt_results = validate_rate_evidence_snapshot(
                bundle.snapshot(),
                [result.snapshot() for result in results],
            )
            self.assertEqual(rebuilt_bundle, bundle)
            self.assertEqual(rebuilt_results, results)

            forged_readiness = deepcopy(bundle.snapshot())
            forged_readiness["campaign_ready"] = True
            forged_readiness["campaign_blockers"] = []
            with self.assertRaisesRegex(
                RateEvidenceValidationError,
                "must be false",
            ):
                validate_rate_evidence_snapshot(
                    forged_readiness,
                    [result.snapshot() for result in results],
                )

            numeric_false_alias = deepcopy(bundle.snapshot())
            numeric_false_alias["campaign_ready"] = 0
            with self.assertRaisesRegex(
                RateEvidenceValidationError,
                "must be false",
            ):
                validate_rate_evidence_snapshot(
                    numeric_false_alias,
                    [result.snapshot() for result in results],
                )

            forged_result = deepcopy(results[0].snapshot())
            forged_result["rate_numerator_decimal"] = "rounded"
            with self.assertRaisesRegex(
                RateEvidenceValidationError,
                "decimal mirrors",
            ):
                validate_rate_evidence_snapshot(
                    bundle.snapshot(),
                    [forged_result],
                )

            coordinated_bundle = deepcopy(bundle.snapshot())
            coordinated_result = deepcopy(results[0].snapshot())
            invented_bundle_sha256 = "b" * 64
            coordinated_bundle["bundle_sha256"] = invented_bundle_sha256
            coordinated_result["bundle_sha256"] = invented_bundle_sha256
            coordinated_payload = dict(coordinated_result)
            coordinated_payload.pop("evidence_sha256")
            coordinated_result["evidence_sha256"] = sha256(
                json.dumps(
                    coordinated_payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(
                RateEvidenceValidationError,
                "metadata no longer match",
            ):
                validate_rate_evidence_snapshot(
                    coordinated_bundle,
                    [coordinated_result],
                )

    def test_verification_can_require_the_exact_source_registry_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(Path(directory))
            bundle, results = load_and_verify_rate_evidence_bundle(
                bundle_path,
                required_source_registry_sha256=_SOURCE_REGISTRY_SHA256,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(bundle.source_registry_sha256, _SOURCE_REGISTRY_SHA256)
            with self.assertRaisesRegex(
                RateEvidenceVerificationError,
                "required source catalogue",
            ):
                verify_rate_evidence_bundle(
                    bundle,
                    required_source_registry_sha256="b" * 64,
                )

    def test_recipe_is_canonical_and_interpreter_is_whitelisted(self) -> None:
        recipe = _recipe()
        parsed = json.loads(recipe)
        self.assertEqual(parsed["interpreter"], EXACT_CSV_INTERPRETER_V1)
        self.assertEqual(
            recipe,
            json.dumps(
                parsed,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

        invalid_recipes = []
        changed = dict(parsed)
        changed["interpreter"] = "pandas-or-arbitrary-code"
        invalid_recipes.append(
            json.dumps(changed, sort_keys=True, separators=(",", ":"))
        )
        changed = dict(parsed)
        changed["unexpected"] = True
        invalid_recipes.append(
            json.dumps(changed, sort_keys=True, separators=(",", ":"))
        )
        invalid_recipes.append(" " + recipe)
        for invalid_recipe in invalid_recipes:
            with (
                self.subTest(recipe=invalid_recipe[:50]),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(
                    Path(directory),
                    recipe_json=invalid_recipe,
                )
                with self.assertRaises(RateEvidenceValidationError):
                    load_rate_evidence_bundle(bundle_path)

    def test_recipe_selector_must_equal_typed_binding_metadata(self) -> None:
        parsed = json.loads(_recipe())
        parsed["row_match"]["jurisdiction_code"] = "BE"
        changed = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(
                Path(directory),
                recipe_json=changed,
            )
            with self.assertRaisesRegex(
                RateEvidenceValidationError,
                "row_match does not exactly match",
            ):
                load_rate_evidence_bundle(bundle_path)

    def test_toml_schema_uses_exact_keys_at_every_level(self) -> None:
        cases = (
            {"top_extra": "unexpected = true"},
            {"artifact_extra": "unexpected = true"},
            {"binding_extra": "unexpected = true"},
            {"signature_extra": "unexpected = true"},
        )
        for options in cases:
            with (
                self.subTest(options=options),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(Path(directory), **options)
                with self.assertRaisesRegex(
                    RateEvidenceValidationError,
                    "keys differ",
                ):
                    load_rate_evidence_bundle(bundle_path)

    def test_toml_numeric_fields_reject_float_and_unreduced_rates(self) -> None:
        cases = (
            ("2469.0", "2000", "strict integer"),
            ("4938", "4000", "lowest terms"),
        )
        for numerator, denominator, message in cases:
            with (
                self.subTest(numerator=numerator),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(
                    Path(directory),
                    rate_numerator=numerator,
                    rate_denominator=denominator,
                )
                with self.assertRaisesRegex(RateEvidenceValidationError, message):
                    load_rate_evidence_bundle(bundle_path)

    def test_artifact_path_must_be_canonical_and_contained(self) -> None:
        for relative_path in (
            "../outside.csv",
            "folder\\rates.csv",
            "/rates.csv",
            "folder/rates.csv:stream",
            "NUL.csv",
        ):
            with (
                self.subTest(path=relative_path),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(
                    Path(directory),
                    relative_path=relative_path,
                )
                with self.assertRaises(RateEvidenceValidationError):
                    load_rate_evidence_bundle(bundle_path)

    def test_content_byte_length_and_hash_are_reattested(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, artifact_path = _write_fixture(Path(directory))
            bundle = load_rate_evidence_bundle(bundle_path)
            original = artifact_path.read_bytes()
            artifact_path.write_bytes(original[:-1] + b"X")
            with self.assertRaisesRegex(
                RateEvidenceVerificationError,
                "SHA-256",
            ):
                verify_rate_evidence_bundle(bundle)

        with tempfile.TemporaryDirectory() as directory:
            bundle_path, artifact_path = _write_fixture(Path(directory))
            bundle = load_rate_evidence_bundle(bundle_path)
            artifact_path.write_bytes(artifact_path.read_bytes() + b"X")
            with self.assertRaisesRegex(
                RateEvidenceVerificationError,
                "byte length changed",
            ):
                verify_rate_evidence_bundle(bundle)

    def test_bundle_file_is_reattested_before_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle_path, _ = _write_fixture(Path(directory))
            bundle = load_rate_evidence_bundle(bundle_path)
            bundle_path.write_text(
                bundle_path.read_text("utf-8") + "\n",
                encoding="utf-8",
                newline="",
            )
            with self.assertRaisesRegex(
                RateEvidenceVerificationError,
                "metadata no longer match",
            ):
                verify_rate_evidence_bundle(bundle)

    def test_csv_selection_requires_exactly_one_matching_row(self) -> None:
        artifacts = (
            _CSV_HEADER.encode("utf-8"),
            (_CSV_HEADER + _CSV_ROW + _CSV_ROW).encode("utf-8"),
        )
        for artifact in artifacts:
            with (
                self.subTest(rows=artifact.count(b"\n")),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(Path(directory), artifact=artifact)
                bundle = load_rate_evidence_bundle(bundle_path)
                with self.assertRaisesRegex(
                    RateEvidenceVerificationError,
                    "header and data row|match exactly one CSV row",
                ):
                    verify_rate_evidence_bundle(bundle)

    def test_csv_rational_accepts_only_canonical_positive_integers(self) -> None:
        for value in ("02469", "+2469", "2469.0", "0", "1e3"):
            row = _CSV_ROW.replace(",2469,2000", f",{value},2000")
            artifact = (_CSV_HEADER + row).encode("utf-8")
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                bundle_path, _ = _write_fixture(Path(directory), artifact=artifact)
                bundle = load_rate_evidence_bundle(bundle_path)
                with self.assertRaisesRegex(
                    RateEvidenceVerificationError,
                    "canonical positive decimal integer",
                ):
                    verify_rate_evidence_bundle(bundle)

    def test_csv_fraction_must_be_reduced_and_equal_the_binding(self) -> None:
        cases = (
            (
                (_CSV_HEADER + _CSV_ROW.replace(",2469,2000", ",4938,4000")).encode(),
                "not in lowest terms",
            ),
            (
                (_CSV_HEADER + _CSV_ROW.replace(",2469,2000", ",2470,2001")).encode(),
                "but declared",
            ),
        )
        for artifact, message in cases:
            with (
                self.subTest(message=message),
                tempfile.TemporaryDirectory() as directory,
            ):
                bundle_path, _ = _write_fixture(Path(directory), artifact=artifact)
                bundle = load_rate_evidence_bundle(bundle_path)
                with self.assertRaisesRegex(RateEvidenceVerificationError, message):
                    verify_rate_evidence_bundle(bundle)

    def test_symlink_or_reparse_artifact_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path, _ = _write_fixture(root)
            bundle = load_rate_evidence_bundle(bundle_path)
            declared_root = root / "rate_artifacts"
            actual_root = root / "actual_rate_artifacts"
            declared_root.rename(actual_root)
            try:
                declared_root.symlink_to(actual_root, target_is_directory=True)
            except OSError:
                self.assertTrue(
                    rate_evidence_module._is_reparse(  # noqa: SLF001
                        SimpleNamespace(st_file_attributes=0x400)
                    )
                )
                return
            with self.assertRaisesRegex(
                RateEvidenceVerificationError,
                "symlink or reparse",
            ):
                verify_rate_evidence_bundle(bundle)

    def test_bundle_rejects_unreferenced_artifacts(self) -> None:
        artifact = (_CSV_HEADER + _CSV_ROW).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle_path, _ = _write_fixture(root, artifact=artifact)
            text = bundle_path.read_text("utf-8")
            extra = f'''[[artifacts]]
artifact_id = "unused"
relative_path = "unused.csv"
media_type = "text/csv"
sha256 = "{sha256(b'unused').hexdigest()}"
byte_length = 6

'''
            text = text.replace("[[bindings]]", extra + "[[bindings]]")
            bundle_path.write_text(text, encoding="utf-8", newline="")
            with self.assertRaisesRegex(
                RateEvidenceValidationError,
                "referenced exactly",
            ):
                load_rate_evidence_bundle(bundle_path)


if __name__ == "__main__":
    unittest.main()
