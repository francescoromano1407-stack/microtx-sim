from __future__ import annotations

from datetime import date
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.data.lineage import build_profile_input_lineage
from microtx_sim.data.profiles import (
    ProfileBundle,
    ProfileValidationError,
    load_profile_bundle,
)
from microtx_sim.data.rate_evidence import (
    RateEvidenceMethod,
    exact_csv_rational_recipe_json,
)
from microtx_sim.outputs.manifest import (
    _monetary_comparability_payload,
    _money_outputs_cross_country_comparable,
    _source_rate_evidence_is_bound,
)


ROOT = Path(__file__).resolve().parents[1]
JURISDICTIONS = ROOT / "configs" / "jurisdictions.toml"
SOURCES = ROOT / "data" / "provenance" / "sources.toml"

_SOURCE_ID = "TEST_ONLY_RATE_SNAPSHOT"
_RETRIEVED_ON = date(2026, 8, 24)
_PERIOD_START = date(2025, 1, 1)
_PERIOD_END = date(2025, 12, 31)
_CSV_HEADER = (
    "source_id,jurisdiction_code,source_currency,target_currency,method,"
    "rate_period_start,rate_period_end,retrieved_on,rate_numerator,"
    "rate_denominator\n"
)


class ProfileRateBindingTests(unittest.TestCase):
    def test_verified_rates_clear_only_the_source_extraction_subgate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle, artifact_path = _write_bound_profile_fixture(Path(directory))
            assessment = bundle.monetary_evidence_assessment()
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            payload = lineage.manifest_payload()

            self.assertTrue(assessment.structure_coherent)
            self.assertTrue(assessment.source_rate_evidence_bound)
            self.assertFalse(assessment.source_bundle_signature_bound)
            self.assertFalse(assessment.output_design_binding_bound)
            self.assertFalse(assessment.population_binding_bound)
            self.assertFalse(assessment.preregistration_bound)
            self.assertFalse(assessment.public_output_comparability)
            self.assertNotIn(
                "monetary_conversion.source_rate_binding=missing",
                assessment.blockers,
            )
            self.assertIn(
                "monetary_conversion.source_bundle_signature=missing",
                assessment.blockers,
            )
            self.assertIn(
                "monetary_conversion.output_design_binding=missing",
                assessment.blockers,
            )
            self.assertEqual(lineage.snapshot["schema_version"], 4)
            self.assertEqual(
                payload["source_evidence_summary"],
                {
                    "present": True,
                    "artifact_count": 1,
                    "binding_count": 4,
                    "verified_result_count": 4,
                    "signature_status": "MISSING",
                },
            )
            self.assertEqual(
                payload["monetary_evidence_assessment"]["blockers"],
                list(assessment.blockers),
            )
            self.assertFalse(_source_rate_evidence_is_bound(payload))
            untrusted_manifest_assessment = _monetary_comparability_payload(payload)
            self.assertFalse(
                untrusted_manifest_assessment["typed_assessment"][
                    "source_rate_evidence_bound"
                ]
            )
            self.assertFalse(
                untrusted_manifest_assessment["manifest_gate"][
                    "source_rate_evidence_bound"
                ]
            )
            self.assertTrue(
                _source_rate_evidence_is_bound(
                    payload,
                    profile_lineage=lineage,
                )
            )
            self.assertFalse(
                _money_outputs_cross_country_comparable(
                    payload,
                    profile_lineage=lineage,
                )
            )
            manifest_assessment = _monetary_comparability_payload(
                payload,
                profile_lineage=lineage,
            )
            self.assertTrue(
                manifest_assessment["manifest_gate"][
                    "source_rate_evidence_bound"
                ]
            )
            self.assertFalse(
                manifest_assessment["manifest_gate"][
                    "public_output_comparability"
                ]
            )
            with self.assertRaisesRegex(
                ProfileValidationError,
                "source_bundle_signature=missing",
            ):
                bundle.validate_monetary_comparability_for_campaign()

            # Registered lineage must re-read the exact artifact bytes, not only
            # trust the fingerprinted copy embedded in the snapshot.
            artifact_path.write_bytes(artifact_path.read_bytes() + b"#mutated\n")
            with self.assertRaisesRegex(
                ProfileValidationError,
                "files are unavailable or invalid",
            ):
                lineage.manifest_payload()

    def test_schema_v3_rejects_missing_or_numerically_different_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, _artifact_path = _write_bound_profile_fixture(root)
            assert bundle.jurisdictions_path is not None
            assert bundle.source_registry_path is not None
            assert bundle.source_evidence_bundle is not None

            with self.assertRaisesRegex(
                ProfileValidationError,
                "require a source evidence bundle",
            ):
                load_profile_bundle(
                    bundle.jurisdictions_path,
                    bundle.source_registry_path,
                    source_bundle_path=None,
                )

            text = bundle.jurisdictions_path.read_text(encoding="utf-8")
            uk = bundle.monetary_conversion("UK")
            changed = text.replace(
                f"rate_numerator = {uk.rate_numerator}",
                f"rate_numerator = {uk.rate_numerator + 1}",
                1,
            )
            changed_path = root / "jurisdictions-changed.toml"
            changed_path.write_text(changed, encoding="utf-8", newline="")
            with self.assertRaisesRegex(
                ProfileValidationError,
                "does not match its verified rate binding",
            ):
                load_profile_bundle(
                    changed_path,
                    bundle.source_registry_path,
                    source_bundle_path=bundle.source_evidence_bundle.bundle_path,
                )

    def test_programmatic_conversion_claims_remain_unregistered(self) -> None:
        base = load_profile_bundle(JURISDICTIONS, SOURCES)
        unregistered = ProfileBundle(
            country_profiles=base.country_profiles,
            state_agents=base.state_agents,
            sources=base.sources,
            profile_status=base.profile_status,
            caveats=base.caveats,
            contracts=base.contracts,
            money_scales=base.money_scales,
            monetary_conversions=(),
            source_evidence_bundle=base.source_evidence_bundle,
        )
        assessment = unregistered.monetary_evidence_assessment()
        self.assertFalse(assessment.source_rate_evidence_bound)
        self.assertFalse(assessment.public_output_comparability)

    def test_compatible_decoy_source_cannot_launder_binding_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ProfileValidationError,
                "rate-binding source does not declare method-compatible scope",
            ):
                _write_bound_profile_fixture(
                    Path(directory),
                    launder_binding_source=True,
                )


def _write_bound_profile_fixture(
    root: Path,
    *,
    launder_binding_source: bool = False,
) -> tuple[ProfileBundle, Path]:
    rate_period = (
        "unmatched-test-period"
        if launder_binding_source
        else "2025-01-01/2025-12-31"
    )
    source_supports = (
        '["median_equivalised_disposable_income"]'
        if launder_binding_source
        else '["foreign_exchange_rate", "median_equivalised_disposable_income"]'
    )
    decoy_source = (
        '''

[[source]]
id = "TEST_ONLY_RATE_DECOY"
publisher = "Test fixture"
title = "Compatible metadata decoy without the bound artifact row"
url = "https://example.invalid/rate-decoy"
period = "2025-01-01/2025-12-31"
geography = "Test jurisdictions"
supports = ["foreign_exchange_rate"]
calibration_status = "CALIBRATED"
'''
        if launder_binding_source
        else ""
    )
    sources_text = SOURCES.read_text(encoding="utf-8") + f'''

[[source]]
id = "{_SOURCE_ID}"
publisher = "Test fixture"
title = "Content-addressed test-only rational rates"
url = "https://example.invalid/content-addressed-test-rates"
period = "{rate_period}"
geography = "Test jurisdictions"
supports = {source_supports}
calibration_status = "CALIBRATED"
{decoy_source}'''
    sources_path = root / "sources.toml"
    sources_path.write_text(sources_text, encoding="utf-8", newline="")
    source_registry_sha256 = sha256(sources_path.read_bytes()).hexdigest()

    # Load only to reuse the exact local-to-simulation ratios already declared
    # in the jurisdiction contract. This fixture is algebraic, not empirical.
    base = load_profile_bundle(
        JURISDICTIONS,
        sources_path,
        source_bundle_path=None,
    )
    ratios = {
        scale.jurisdiction_code: Fraction(
            scale.simulation_monthly_anchor_cents,
            scale.nominal_monthly_anchor_minor_units,
        )
        for scale in base.money_scales
    }
    currencies = {
        scale.jurisdiction_code: scale.currency for scale in base.money_scales
    }
    binding_ids = {
        code: f"{code.lower()}-{currencies[code].lower()}-tst-2025"
        for code in currencies
    }

    rows = []
    for code in sorted(currencies):
        ratio = ratios[code]
        rows.append(
            f"{_SOURCE_ID},{code},{currencies[code]},TST,FX,"
            "2025-01-01,2025-12-31,2026-08-24,"
            f"{ratio.numerator},{ratio.denominator}\n"
        )
    artifact = (_CSV_HEADER + "".join(rows)).encode("utf-8")
    artifact_root = root / "rate_artifacts"
    artifact_root.mkdir()
    artifact_path = artifact_root / "rates.csv"
    artifact_path.write_bytes(artifact)

    binding_tables = []
    for code in sorted(currencies, key=lambda item: binding_ids[item]):
        ratio = ratios[code]
        recipe = exact_csv_rational_recipe_json(
            source_id=_SOURCE_ID,
            jurisdiction_code=code,
            source_currency=currencies[code],
            target_currency="TST",
            method=RateEvidenceMethod.FX,
            rate_period_start=_PERIOD_START,
            rate_period_end=_PERIOD_END,
            retrieved_on=_RETRIEVED_ON,
        )
        binding_tables.append(
            f'''
[[bindings]]
binding_id = "{binding_ids[code]}"
artifact_id = "test-rates"
source_id = "{_SOURCE_ID}"
jurisdiction_code = "{code}"
source_currency = "{currencies[code]}"
target_currency = "TST"
method = "FX"
rate_period_start = 2025-01-01
rate_period_end = 2025-12-31
retrieved_on = 2026-08-24
rate_numerator = {ratio.numerator}
rate_denominator = {ratio.denominator}
recipe_json = {json.dumps(recipe)}
'''
        )
    source_bundle_text = f'''schema_version = 1
bundle_id = "test-profile-rate-evidence"
provenance_status = "CALIBRATED"
source_registry_sha256 = "{source_registry_sha256}"
artifact_root = "rate_artifacts"
notes = "Test-only exact extraction fixture; not an empirical rate choice."

[[artifacts]]
artifact_id = "test-rates"
relative_path = "rates.csv"
media_type = "text/csv"
sha256 = "{sha256(artifact).hexdigest()}"
byte_length = {len(artifact)}
{''.join(binding_tables)}
[signature]
status = "MISSING"
algorithm = "NONE"
key_id = ""
value = ""
'''
    source_bundle_path = root / "source_bundle.toml"
    source_bundle_path.write_text(
        source_bundle_text,
        encoding="utf-8",
        newline="",
    )

    jurisdiction_text = JURISDICTIONS.read_text(encoding="utf-8")
    jurisdiction_text = jurisdiction_text.replace(
        "schema_version = 2", "schema_version = 3", 1
    )
    jurisdiction_text = jurisdiction_text.replace(
        "median_equivalised_disposable_income_minor_units = 300000\n"
        'income_status = "ILLUSTRATIVE"',
        "median_equivalised_disposable_income_minor_units = 300000\n"
        f'income_source = "{_SOURCE_ID}"\n'
        'income_status = "ILLUSTRATIVE"',
        1,
    )
    jurisdiction_text = jurisdiction_text.replace(
        'income_status = "ANCHORED"', 'income_status = "CALIBRATED"'
    ).replace(
        'income_status = "ILLUSTRATIVE"', 'income_status = "CALIBRATED"'
    ).replace(
        'currency_scale_status = "ILLUSTRATIVE"',
        'currency_scale_status = "CALIBRATED"',
    )
    conversion_tables = []
    conversion_source_ids = (
        f'["{_SOURCE_ID}", "TEST_ONLY_RATE_DECOY"]'
        if launder_binding_source
        else f'["{_SOURCE_ID}"]'
    )
    for code in currencies:
        ratio = ratios[code]
        conversion_tables.append(
            f'''
[[monetary_conversion]]
conversion_id = "{code.lower()}-to-tst-2025"
rate_binding_id = "{binding_ids[code]}"
jurisdiction_code = "{code}"
source_currency = "{currencies[code]}"
target_currency = "TST"
method = "FX"
rate_numerator = {ratio.numerator}
rate_denominator = {ratio.denominator}
rate_period_start = "2025-01-01"
rate_period_end = "2025-12-31"
target_price_period_start = "2025-01-01"
target_price_period_end = "2025-12-31"
estimand = "test-only comparable monetary amount"
population_base = "test-only common population"
comparison_group = "test-only exact extraction fixture"
rounding_method = "nearest_minor_unit_half_away_from_zero"
rounding_scope = "AFTER_AGGREGATION"
aggregation_unit = "one test-only jurisdiction-seed total"
status = "CALIBRATED"
source_ids = {conversion_source_ids}
retrieved_on = "2026-08-24"
notes = "Algebraic test fixture; not an empirical rate choice."
'''
        )
    jurisdictions_path = root / "jurisdictions.toml"
    jurisdictions_path.write_text(
        jurisdiction_text + "".join(conversion_tables),
        encoding="utf-8",
        newline="",
    )
    return (
        load_profile_bundle(
            jurisdictions_path,
            sources_path,
            source_bundle_path=source_bundle_path,
        ),
        artifact_path,
    )


if __name__ == "__main__":
    unittest.main()
