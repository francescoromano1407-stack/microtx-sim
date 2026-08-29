from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
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
    resolve_monetary_output_basis,
)
from microtx_sim.data.profiles import load_profile_bundle
from tests.monetary_execution_fixture import (
    DEFAULT_TARGET_PER_SIMULATION as _TARGET_PER_SIMULATION,
    JURISDICTIONS,
    PERIOD_END as _PERIOD_END,
    PERIOD_START as _PERIOD_START,
    SOURCES,
    write_monetary_execution_fixture as _write_execution_fixture,
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

            # The legacy aggregate gate deliberately rejects unequal global
            # scale ratios. Player-level execution is safe because it retains
            # and applies each player's jurisdiction before pooling.
            self.assertFalse(
                bundle.monetary_evidence_assessment().structure_coherent
            )

    def test_converts_once_per_player_with_signed_half_away_rounding(self) -> None:
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
                (1, -1, 2, -2, 2, -2, 3, -3),
            )
            self.assertTrue(
                all(type(value) is int for value in outcome.converted_values)
            )
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
                    converted_values=(0, *outcome.converted_values[1:]),
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
            ("AFTER_AGGREGATION", MONETARY_OUTPUT_AGGREGATION_UNIT, "PER_OBSERVATION"),
            ("PER_OBSERVATION", "one arbitrary observation", "aggregation unit"),
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

    def test_default_declaration_only_profiles_fail_closed(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        lineage = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )
        with self.assertRaisesRegex(
            MonetaryOutputExecutionValidationError,
            "schema version 3",
        ):
            build_monetary_output_currency_semantics(
                lineage,
                jurisdiction_codes=lineage.profile_codes,
                target_minor_unit_name="cent",
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
