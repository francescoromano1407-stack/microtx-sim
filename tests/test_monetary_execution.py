from __future__ import annotations

import csv
from dataclasses import replace
from datetime import timedelta
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from microtx_sim.data.lineage import build_profile_input_lineage
from microtx_sim.data.monetary_execution import (
    MONETARY_OUTPUT_AGGREGATION_UNIT,
    ConvertedMonetaryOutcome,
    MonetaryOutputExecutionValidationError,
    build_monetary_output_currency_semantics,
    convert_monetary_outcome,
    round_target_minor_units,
    resolve_monetary_output_basis,
)
from microtx_sim.data.profiles import ProfileValidationError, load_profile_bundle
from tests.monetary_execution_fixture import (
    DEFAULT_TARGET_PER_SIMULATION as _TARGET_PER_SIMULATION,
    JURISDICTIONS,
    PERIOD_END as _PERIOD_END,
    PERIOD_START as _PERIOD_START,
    SOURCES,
    write_monetary_execution_fixture as _write_execution_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
MONETARY_ROOT = ROOT / "inputs" / "monetary" / "ecb-eur-fx-2024-v1"
MONETARY_BUNDLE = MONETARY_ROOT / "bundle.toml"
MONETARY_ARTIFACT = MONETARY_ROOT / "artifacts" / "conversion_rates.csv"
MONETARY_OFFICIAL_SOURCE = (
    MONETARY_ROOT / "artifacts" / "ecb_exr_annual_2024.csv"
)


class MonetaryExecutionTests(unittest.TestCase):
    def test_resolves_reverified_jurisdiction_specific_basis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, _artifact = _write_execution_fixture(Path(directory))
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            currency = build_monetary_output_currency_semantics(
                lineage,
                jurisdiction_codes=lineage.profile_codes,
                target_minor_unit_name="test minor unit",
            )
            basis = resolve_monetary_output_basis(
                lineage,
                currency,
                jurisdiction_codes=lineage.profile_codes,
            )

            self.assertEqual(basis.target_currency, "TST")
            self.assertEqual(basis.target_minor_unit_name, "test minor unit")
            self.assertEqual(basis.price_period_start, _PERIOD_START)
            self.assertEqual(basis.price_period_end, _PERIOD_END)
            self.assertEqual(basis.profile_input_sha256, lineage.fingerprint_sha256)
            self.assertEqual(basis.jurisdiction_codes, lineage.profile_codes)
            self.assertEqual(currency.currency_basis_sha256, basis.basis_sha256)
            self.assertEqual(currency, basis.currency_semantics)
            self.assertEqual(
                {
                    row.jurisdiction_code: row.target_per_simulation
                    for row in basis.jurisdictions
                },
                _TARGET_PER_SIMULATION,
            )
            self.assertEqual(
                len(
                    {
                        row.target_per_simulation
                        for row in basis.jurisdictions
                    }
                ),
                4,
            )
            self.assertFalse(basis.campaign_ready)
            self.assertTrue(basis.campaign_blockers)
            self.assertEqual(basis.snapshot()["basis_sha256"], basis.basis_sha256)
            for row in basis.jurisdictions:
                self.assertEqual(len(row.money_scale_sha256), 64)
                self.assertEqual(len(row.monetary_conversion_sha256), 64)
                self.assertEqual(len(row.rate_binding_sha256), 64)
                self.assertEqual(len(row.rate_evidence_sha256), 64)

            # Jurisdiction-specific scale ratios are expected: exact conversion
            # happens before any cross-jurisdiction population aggregation.
            self.assertTrue(
                bundle.monetary_evidence_assessment().structure_coherent
            )

    def test_converts_exactly_and_rounds_only_at_final_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, _artifact = _write_execution_fixture(Path(directory))
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            currency = build_monetary_output_currency_semantics(
                lineage,
                jurisdiction_codes=lineage.profile_codes,
                target_minor_unit_name="test minor unit",
            )
            basis = resolve_monetary_output_basis(
                lineage,
                currency,
                jurisdiction_codes=lineage.profile_codes,
            )
            outcome = convert_monetary_outcome(
                basis,
                player_ids=np.arange(8, dtype=np.int64),
                jurisdiction_indices=np.asarray(
                    [0, 0, 1, 1, 2, 2, 3, 3],
                    dtype=np.int16,
                ),
                jurisdiction_codes=lineage.profile_codes,
                raw_values=np.asarray(
                    [1, -1, 1, -1, 1, -1, 1, -1],
                    dtype=np.int64,
                ),
            )

            self.assertEqual(
                outcome.converted_values,
                (
                    Fraction(1, 2),
                    Fraction(-1, 2),
                    Fraction(3, 2),
                    Fraction(-3, 2),
                    Fraction(2, 1),
                    Fraction(-2, 1),
                    Fraction(5, 2),
                    Fraction(-5, 2),
                ),
            )
            self.assertTrue(
                all(type(value) is Fraction for value in outcome.converted_values)
            )
            self.assertEqual(round_target_minor_units(Fraction(1, 2)), 1)
            self.assertEqual(round_target_minor_units(Fraction(-1, 2)), -1)
            self.assertEqual(len(outcome.raw_values_sha256), 64)
            self.assertEqual(len(outcome.converted_values_sha256), 64)
            self.assertEqual(len(outcome.player_ids_sha256), 64)
            self.assertEqual(len(outcome.execution_sha256), 64)
            self.assertNotIn("player_ids_decimal", outcome.snapshot())
            self.assertNotIn("jurisdiction_indices_decimal", outcome.snapshot())
            self.assertNotIn("raw_values_decimal", outcome.snapshot())
            self.assertNotIn("converted_values_decimal", outcome.snapshot())
            self.assertEqual(
                outcome.snapshot()["execution_sha256"],
                outcome.execution_sha256,
            )
            with self.assertRaisesRegex(
                MonetaryOutputExecutionValidationError,
                "converted values",
            ):
                replace(
                    outcome,
                    converted_values=(Fraction(0, 1), *outcome.converted_values[1:]),
                )

    def test_basis_rejects_currency_period_digest_and_order_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, _artifact = _write_execution_fixture(Path(directory))
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            currency = build_monetary_output_currency_semantics(
                lineage,
                jurisdiction_codes=lineage.profile_codes,
                target_minor_unit_name="test minor unit",
            )
            cases = (
                (replace(currency, currency_code="EUR"), "target currency"),
                (
                    replace(
                        currency,
                        price_period_end=currency.price_period_end
                        + timedelta(days=1),
                    ),
                    "price period",
                ),
                (
                    replace(currency, currency_basis_sha256="f" * 64),
                    "currency_basis_sha256",
                ),
            )
            for changed, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(
                        MonetaryOutputExecutionValidationError,
                        message,
                    ):
                        resolve_monetary_output_basis(
                            lineage,
                            changed,
                            jurisdiction_codes=lineage.profile_codes,
                        )
            with self.assertRaisesRegex(
                MonetaryOutputExecutionValidationError,
                "ordered jurisdiction codes",
            ):
                resolve_monetary_output_basis(
                    lineage,
                    currency,
                    jurisdiction_codes=tuple(reversed(lineage.profile_codes)),
                )

    def test_rejects_nonexecuted_rounding_scope_and_aggregation_unit(self) -> None:
        cases = (
            ("PER_OBSERVATION", MONETARY_OUTPUT_AGGREGATION_UNIT, "AFTER_AGGREGATION"),
            ("AFTER_AGGREGATION", "one arbitrary observation", "aggregation unit"),
        )
        for rounding_scope, aggregation_unit, message in cases:
            with self.subTest(rounding_scope=rounding_scope):
                with tempfile.TemporaryDirectory() as directory:
                    bundle, _artifact = _write_execution_fixture(
                        Path(directory),
                        rounding_scope=rounding_scope,
                        aggregation_unit=aggregation_unit,
                    )
                    lineage = build_profile_input_lineage(
                        bundle.country_profiles,
                        profile_bundle=bundle,
                    )
                    with self.assertRaisesRegex(
                        MonetaryOutputExecutionValidationError,
                        message,
                    ):
                        build_monetary_output_currency_semantics(
                            lineage,
                            jurisdiction_codes=lineage.profile_codes,
                            target_minor_unit_name="test minor unit",
                        )

    def test_checked_in_official_basis_executes_but_remains_campaign_blocked(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        lineage = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )
        currency = build_monetary_output_currency_semantics(
            lineage,
            jurisdiction_codes=lineage.profile_codes,
            target_minor_unit_name="cent",
        )
        basis = resolve_monetary_output_basis(
            lineage,
            currency,
            jurisdiction_codes=lineage.profile_codes,
        )
        self.assertEqual(basis.target_currency, "EUR")
        self.assertEqual(basis.source_bundle_id, "ecb-eur-fx-2024-v1")
        self.assertEqual(basis.source_bundle_signature_status, "MISSING")
        self.assertEqual(
            {row.rate_artifact_id for row in basis.jurisdictions},
            {
                "ecb-eur-fx-2024-conversion-rates",
                "ecb-exr-annual-2024-official-response",
            },
        )
        self.assertFalse(basis.campaign_ready)
        self.assertIn(
            "monetary_output_execution.simulation_to_local_currency_bridge=unvalidated",
            basis.campaign_blockers,
        )
        with self.assertRaisesRegex(
            ProfileValidationError,
            "currency_scale=ILLUSTRATIVE|source_bundle_signature=missing",
        ):
            bundle.validate_monetary_comparability_for_campaign()

    def test_checked_in_bundle_recomputes_hashes_and_exact_ecb_transformations(
        self,
    ) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        evidence = bundle.source_evidence_bundle
        assert evidence is not None
        self.assertEqual(evidence.bundle_path, MONETARY_BUNDLE.resolve())
        self.assertEqual(
            evidence.bundle_sha256,
            sha256(MONETARY_BUNDLE.read_bytes()).hexdigest(),
        )
        declared_artifacts = {
            artifact.relative_path: artifact for artifact in evidence.artifacts
        }
        self.assertEqual(
            set(declared_artifacts),
            {"conversion_rates.csv", "ecb_exr_annual_2024.csv"},
        )
        for path in (MONETARY_ARTIFACT, MONETARY_OFFICIAL_SOURCE):
            artifact = declared_artifacts[path.name]
            self.assertEqual(artifact.byte_length, path.stat().st_size)
            self.assertEqual(
                artifact.sha256,
                sha256(path.read_bytes()).hexdigest(),
            )
        self.assertEqual(evidence.signature.status.value, "MISSING")
        self.assertEqual(evidence.signature.algorithm, "NONE")

        with MONETARY_ARTIFACT.open(encoding="utf-8", newline="") as handle:
            rows = {
                row["jurisdiction_code"]: row
                for row in csv.DictReader(handle)
            }
        self.assertEqual(set(rows), {"UK", "KR", "JP", "BE"})
        with MONETARY_OFFICIAL_SOURCE.open(
            encoding="utf-8",
            newline="",
        ) as handle:
            official_rows = {
                row["CURRENCY"]: row for row in csv.DictReader(handle)
            }
        self.assertEqual(set(official_rows), {"GBP", "JPY", "KRW"})
        expected_series = {
            "UK": ("EXR.A.GBP.EUR.SP00.A", "0.8466166015625", 2),
            "KR": ("EXR.A.KRW.EUR.SP00.A", "1475.4041015624998", 0),
            "JP": ("EXR.A.JPY.EUR.SP00.A", "163.8519140625", 0),
        }
        conversions = {
            item.jurisdiction_code: item for item in bundle.monetary_conversions
        }
        for code, (series_id, observation, source_exponent) in expected_series.items():
            row = rows[code]
            self.assertEqual(row["series_id"], series_id)
            self.assertEqual(row["observation_value"], observation)
            self.assertEqual(
                official_rows[row["source_currency"]]["KEY"],
                series_id,
            )
            self.assertEqual(
                official_rows[row["source_currency"]]["OBS_VALUE"],
                observation,
            )
            self.assertEqual(row["publication_date_status"], "NOT_PROVIDED_BY_ENDPOINT")
            self.assertTrue(row["permanent_identifier"])
            expected = Fraction(100, 10**source_exponent) / Fraction(observation)
            declared = Fraction(
                int(row["rate_numerator"]),
                int(row["rate_denominator"]),
            )
            self.assertEqual(declared, expected)
            self.assertEqual(conversions[code].conversion_ratio, expected)
        self.assertEqual(conversions["BE"].conversion_ratio, Fraction(1, 1))
        self.assertEqual(rows["BE"]["timing_convention"], "IDENTITY_SAME_CURRENCY")

    def test_checked_in_preflight_rejects_missing_or_inconsistent_contract_fields(
        self,
    ) -> None:
        original = JURISDICTIONS.read_text(encoding="utf-8")
        mutations = (
            (
                "missing target currency",
                original.replace('target_currency = "EUR"\n', "", 1),
            ),
            (
                "missing reference period",
                original.replace('rate_period_start = "2024-01-01"\n', "", 1),
            ),
            (
                "invalid FX factor",
                original.replace("rate_denominator = 4334677", "rate_denominator = 0", 1),
            ),
            (
                "missing scale factor",
                original.replace("simulation_monthly_anchor_cents = 180000\n", "", 1),
            ),
            (
                "missing quote convention",
                original.replace(
                    'quote_convention = "target minor units per source minor unit"\n',
                    "",
                    1,
                ),
            ),
            (
                "inconsistent target currency",
                original.replace('target_currency = "EUR"', 'target_currency = "USD"', 1),
            ),
            (
                "missing source identity",
                original.replace('source_ids = ["ECB_EXR_A_2024_EUR"]\n', "", 1),
            ),
        )
        for label, jurisdiction_text in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jurisdictions = root / "jurisdictions.toml"
                jurisdictions.write_text(jurisdiction_text, encoding="utf-8")
                with self.assertRaises(ProfileValidationError):
                    load_profile_bundle(
                        jurisdictions,
                        SOURCES,
                        source_bundle_path=MONETARY_BUNDLE,
                        population_bundle_path=None,
                    )

    def test_changed_checked_source_bytes_or_metadata_fail_reverification(self) -> None:
        for mutation in ("artifact", "metadata"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                jurisdictions = root / "jurisdictions.toml"
                sources = root / "sources.toml"
                monetary = root / "monetary"
                jurisdictions.write_bytes(JURISDICTIONS.read_bytes())
                sources.write_bytes(SOURCES.read_bytes())
                shutil.copytree(MONETARY_ROOT, monetary)
                if mutation == "artifact":
                    artifact = monetary / "artifacts" / "ecb_exr_annual_2024.csv"
                    artifact.write_bytes(artifact.read_bytes() + b"# changed\n")
                else:
                    sources.write_text(
                        sources.read_text(encoding="utf-8").replace(
                            "ECB annual euro foreign-exchange reference rates",
                            "Changed ECB metadata",
                            1,
                        ),
                        encoding="utf-8",
                    )
                with self.assertRaises(ProfileValidationError):
                    load_profile_bundle(
                        jurisdictions,
                        sources,
                        source_bundle_path=monetary / "bundle.toml",
                        population_bundle_path=None,
                    )

    def test_reopens_rate_artifact_and_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, artifact = _write_execution_fixture(Path(directory))
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            artifact.write_bytes(artifact.read_bytes() + b"# mutation\n")
            with self.assertRaisesRegex(
                MonetaryOutputExecutionValidationError,
                "re-attest profile inputs",
            ):
                build_monetary_output_currency_semantics(
                    lineage,
                    jurisdiction_codes=lineage.profile_codes,
                    target_minor_unit_name="test minor unit",
                )

    def test_conversion_rejects_misalignment_types_and_unknown_jurisdiction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, _artifact = _write_execution_fixture(Path(directory))
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            currency = build_monetary_output_currency_semantics(
                lineage,
                jurisdiction_codes=lineage.profile_codes,
                target_minor_unit_name="test minor unit",
            )
            basis = resolve_monetary_output_basis(
                lineage,
                currency,
                jurisdiction_codes=lineage.profile_codes,
            )
            with self.assertRaisesRegex(
                MonetaryOutputExecutionValidationError,
                "must align",
            ):
                convert_monetary_outcome(
                    basis,
                    player_ids=(0, 1),
                    jurisdiction_indices=(0,),
                    jurisdiction_codes=lineage.profile_codes,
                    raw_values=(1, 2),
                )
            with self.assertRaisesRegex(
                MonetaryOutputExecutionValidationError,
                "at most 3",
            ):
                convert_monetary_outcome(
                    basis,
                    player_ids=(0,),
                    jurisdiction_indices=(4,),
                    jurisdiction_codes=lineage.profile_codes,
                    raw_values=(1,),
                )
            with self.assertRaisesRegex(TypeError, "exact integer"):
                convert_monetary_outcome(
                    basis,
                    player_ids=(0,),
                    jurisdiction_indices=(0,),
                    jurisdiction_codes=lineage.profile_codes,
                    raw_values=(1.0,),
                )


if __name__ == "__main__":
    unittest.main()
