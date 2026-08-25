from __future__ import annotations

from dataclasses import replace
from datetime import date
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.data import ProfileInputLineage, build_profile_input_lineage
from microtx_sim.data.profiles import (
    MonetaryConversionContract,
    MonetaryConversionMethod,
    MonetaryRoundingScope,
    MoneyScaleContract,
    ProfileBundle,
    ProfileValidationError,
    SourceProvenance,
    load_country_profiles,
    load_profile_bundle,
)
from microtx_sim.types import ProvenanceStatus


ROOT = Path(__file__).resolve().parents[1]
JURISDICTIONS = ROOT / "configs" / "jurisdictions.toml"
SOURCES = ROOT / "data" / "provenance" / "sources.toml"

_TEST_ONLY_SOURCE_ID = "TEST_ONLY_MONETARY_CONVERSION"
_TEST_ONLY_SOURCE_TOML = """

[[source]]
id = "TEST_ONLY_MONETARY_CONVERSION"
publisher = "Test fixture"
title = "Test-only monetary conversion arithmetic"
url = "https://example.invalid/test-only-monetary-conversion"
period = "2025-01-01/2025-12-31"
geography = "Test jurisdictions"
supports = ["foreign_exchange_rate"]
calibration_status = "ILLUSTRATIVE"
"""
_TEST_ONLY_UK_CONVERSION_TOML = """

[[monetary_conversion]]
jurisdiction_code = "UK"
source_currency = "GBP"
target_currency = "TST"
method = "FX"
rate_numerator = 180000
rate_denominator = 305525
rate_period_start = "2025-01-01"
rate_period_end = "2025-12-31"
target_price_period_start = "2025-01-01"
target_price_period_end = "2025-12-31"
estimand = "test-only comparable monetary amount"
population_base = "test-only common population"
comparison_group = "test-only parser fixture"
rounding_method = "nearest_minor_unit_half_away_from_zero"
rounding_scope = "AFTER_AGGREGATION"
aggregation_unit = "one test-only jurisdiction-seed total"
status = "ILLUSTRATIVE"
source_ids = ["TEST_ONLY_MONETARY_CONVERSION"]
retrieved_on = "2026-08-24"
notes = "Test-only algebraic fixture; not an empirical rate."
"""


class ProfileLoadingTests(unittest.TestCase):
    def test_parses_four_profiles_and_keeps_money_units_separate(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)

        self.assertEqual(
            tuple(profile.code for profile in bundle.country_profiles),
            ("UK", "KR", "JP", "BE"),
        )
        self.assertEqual(bundle.profile_status, ProvenanceStatus.ILLUSTRATIVE)
        self.assertEqual(len(bundle.state_agents), 4)
        self.assertEqual(bundle.source_retrieved_on, date(2026, 8, 24))
        self.assertEqual(
            {source.retrieved_on for source in bundle.sources.values()},
            {date(2026, 8, 24)},
        )
        self.assertEqual(bundle.jurisdictions_path, JURISDICTIONS.resolve())
        self.assertEqual(bundle.source_registry_path, SOURCES.resolve())
        self.assertEqual(
            bundle.jurisdictions_sha256,
            sha256(JURISDICTIONS.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            bundle.source_registry_sha256,
            sha256(SOURCES.read_bytes()).hexdigest(),
        )
        self.assertEqual(bundle.monetary_conversions, ())
        with self.assertRaises(KeyError):
            bundle.monetary_conversion("UK")

        # Prices and player incomes share one internal unit.  Unsupported nominal
        # country rankings are not smuggled into the agent table.
        self.assertEqual(
            {profile.monthly_income_median_cents for profile in bundle.country_profiles},
            {180_000},
        )
        nominal = {
            scale.jurisdiction_code: scale.nominal_monthly_anchor_minor_units
            for scale in bundle.money_scales
        }
        self.assertEqual(nominal["UK"], 305_525)  # £36,663 / 12, in pence
        self.assertEqual(nominal["KR"], 3_515_000)  # central monthly quintile
        self.assertEqual(nominal["JP"], 300_000)
        self.assertEqual(nominal["BE"], 260_825)  # €31,299 / 12, in cents
        self.assertEqual(
            bundle.money_scale("JP").anchor_status,
            ProvenanceStatus.ILLUSTRATIVE,
        )
        self.assertTrue(
            all(
                scale.scale_status is ProvenanceStatus.ILLUSTRATIVE
                and not scale.cross_country_comparable
                for scale in bundle.money_scales
            )
        )
        with self.assertRaisesRegex(ProfileValidationError, "cross-country"):
            bundle.money_scale("UK").nominal_ratio_to(bundle.money_scale("KR"))
        self.assertTrue(any("must not" in caveat for caveat in bundle.caveats))

        with self.assertRaisesRegex(ProfileValidationError, "SYNTHETIC"):
            bundle.validate_for_run(allow_synthetic=False)
        bundle.validate_for_run(allow_synthetic=True)

    def test_monetary_conversion_is_exact_signed_and_currency_safe(self) -> None:
        contract = MonetaryConversionContract(
            jurisdiction_code="UK",
            source_currency="GBP",
            target_currency="TST",
            method=MonetaryConversionMethod.FX,
            rate_numerator=1,
            rate_denominator=2,
            rate_period_start=date(2025, 1, 1),
            rate_period_end=date(2025, 12, 31),
            target_price_period_start=date(2025, 1, 1),
            target_price_period_end=date(2025, 12, 31),
            estimand="test-only comparable monetary amount",
            population_base="test-only common population",
            comparison_group="test-only exact arithmetic",
            rounding_method="nearest_minor_unit_half_away_from_zero",
            rounding_scope=MonetaryRoundingScope.AFTER_AGGREGATION,
            aggregation_unit="one test-only jurisdiction-seed total",
            status=ProvenanceStatus.ILLUSTRATIVE,
            source_ids=("TEST_ONLY_CONVERSION",),
            retrieved_on=date(2026, 8, 24),
            notes="Arithmetic fixture only; not an observed exchange rate.",
        )

        self.assertEqual(contract.conversion_ratio, Fraction(1, 2))
        self.assertEqual(
            contract.comparison_signature,
            (
                "TST",
                "FX",
                "2025-01-01",
                "2025-12-31",
                "2025-01-01",
                "2025-12-31",
                "test-only comparable monetary amount",
                "test-only common population",
                "test-only exact arithmetic",
                "nearest_minor_unit_half_away_from_zero",
                "AFTER_AGGREGATION",
                "one test-only jurisdiction-seed total",
            ),
        )
        expected = {
            -5: -3,
            -4: -2,
            -3: -2,
            -2: -1,
            -1: -1,
            0: 0,
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
        }
        self.assertEqual(
            {
                amount: contract.convert_minor_units(amount, currency="GBP")
                for amount in expected
            },
            expected,
        )
        self.assertEqual(
            contract.convert_many_minor_units((1, 1), currency="GBP"),
            1,
        )
        per_observation = replace(
            contract,
            rounding_scope=MonetaryRoundingScope.PER_OBSERVATION,
        )
        self.assertEqual(
            per_observation.convert_many_minor_units((1, 1), currency="GBP"),
            2,
        )
        with self.assertRaisesRegex(ProfileValidationError, "GBP.*EUR"):
            per_observation.convert_many_minor_units((), currency="EUR")
        with self.assertRaisesRegex(ProfileValidationError, "GBP.*EUR"):
            contract.convert_minor_units(1, currency="EUR")
        for invalid in (True, 1.0):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ProfileValidationError, "strict integer"):
                    contract.convert_minor_units(invalid, currency="GBP")
        for invalid_currency in ("gbp", "123", "12!"):
            with self.subTest(invalid_currency=invalid_currency):
                with self.assertRaisesRegex(ProfileValidationError, "ISO-style"):
                    replace(contract, source_currency=invalid_currency)
        with self.assertRaisesRegex(ProfileValidationError, "rounding method"):
            replace(contract, rounding_method="bankers_rounding")
        with self.assertRaisesRegex(ProfileValidationError, "immutable tuple"):
            replace(contract, source_ids=["TEST_ONLY_CONVERSION"])
        with self.assertRaisesRegex(ProfileValidationError, "target price period"):
            replace(
                contract,
                target_price_period_end=date(2026, 12, 31),
            )
        with self.assertRaisesRegex(ProfileValidationError, "cannot predate"):
            replace(contract, retrieved_on=date(2025, 12, 30))

    def test_schema_v2_parses_an_explicit_test_only_conversion(self) -> None:
        jurisdiction_text = (
            JURISDICTIONS.read_text(encoding="utf-8")
            + _TEST_ONLY_UK_CONVERSION_TOML
        )
        source_text = SOURCES.read_text(encoding="utf-8") + _TEST_ONLY_SOURCE_TOML

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jurisdictions_path = root / "jurisdictions.toml"
            sources_path = root / "sources.toml"
            jurisdictions_path.write_text(jurisdiction_text, encoding="utf-8")
            sources_path.write_text(source_text, encoding="utf-8")

            bundle = load_profile_bundle(jurisdictions_path, sources_path)
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            conversion_summary_count = lineage.manifest_payload()[
                "monetary_conversion_summary"
            ]["count"]

        self.assertEqual(len(bundle.monetary_conversions), 1)
        self.assertEqual(lineage.lineage_status, "registered_profile_bundle")
        self.assertEqual(conversion_summary_count, 1)
        conversion = bundle.monetary_conversion("UK")
        self.assertEqual(conversion.source_currency, "GBP")
        self.assertEqual(conversion.target_currency, "TST")
        self.assertIs(conversion.method, MonetaryConversionMethod.FX)
        self.assertEqual(conversion.conversion_ratio, Fraction(180_000, 305_525))
        self.assertEqual(conversion.retrieved_on, date(2026, 8, 24))
        self.assertIn("not an empirical rate", conversion.notes)

    def test_schema_v2_conversion_tables_reject_unknown_fields(self) -> None:
        variants = (
            _TEST_ONLY_UK_CONVERSION_TOML.replace(
                "rounding_method =",
                "rounding_methd =",
                1,
            ),
            _TEST_ONLY_UK_CONVERSION_TOML + "rate = 0.999\n",
        )
        source_text = SOURCES.read_text(encoding="utf-8") + _TEST_ONLY_SOURCE_TOML
        for conversion_text in variants:
            with self.subTest(conversion_text=conversion_text[-40:]):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    jurisdictions_path = root / "jurisdictions.toml"
                    sources_path = root / "sources.toml"
                    jurisdictions_path.write_text(
                        JURISDICTIONS.read_text(encoding="utf-8")
                        + conversion_text,
                        encoding="utf-8",
                    )
                    sources_path.write_text(source_text, encoding="utf-8")
                    with self.assertRaisesRegex(
                        ProfileValidationError,
                        "unknown fields",
                    ):
                        load_profile_bundle(jurisdictions_path, sources_path)

    def test_jurisdiction_schema_v1_remains_readable_without_v2_fields(self) -> None:
        text = JURISDICTIONS.read_text(encoding="utf-8").replace(
            "schema_version = 2",
            "schema_version = 1",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "jurisdictions.toml"
            path.write_text(text, encoding="utf-8")
            bundle = load_profile_bundle(path, SOURCES)

        self.assertEqual(bundle.monetary_conversions, ())

    def test_jurisdiction_schema_rejects_unknown_or_v2_fields_in_v1(self) -> None:
        original = JURISDICTIONS.read_text(encoding="utf-8")
        unsupported = original.replace("schema_version = 2", "schema_version = 3", 1)
        boolean_alias = original.replace(
            "schema_version = 2", "schema_version = true", 1
        )
        float_alias = original.replace(
            "schema_version = 2", "schema_version = 2.0", 1
        )
        v1_with_monetary_field = original.replace(
            "schema_version = 2",
            "schema_version = 1",
            1,
        ).replace(
            'income_period = "annual"',
            'income_period = "annual"\nsimulation_monthly_anchor_cents = 180000',
            1,
        )
        for text, message in (
            (unsupported, "unsupported jurisdiction schema_version"),
            (boolean_alias, "unsupported jurisdiction schema_version"),
            (float_alias, "unsupported jurisdiction schema_version"),
            (v1_with_monetary_field, "version-2 monetary fields"),
        ):
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "jurisdictions.toml"
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaisesRegex(ProfileValidationError, message):
                        load_profile_bundle(path, SOURCES)

    def test_source_schema_requires_a_strict_integer_version(self) -> None:
        original = SOURCES.read_text(encoding="utf-8")
        for replacement in ("schema_version = true", "schema_version = 1.0"):
            with self.subTest(replacement=replacement):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "sources.toml"
                    path.write_text(
                        original.replace("schema_version = 1", replacement, 1),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        ProfileValidationError,
                        "unsupported source schema_version",
                    ):
                        load_profile_bundle(JURISDICTIONS, path)

    def test_profile_numbers_reject_non_finite_values_at_ingestion(self) -> None:
        original = JURISDICTIONS.read_text(encoding="utf-8")
        variants = (
            original.replace("income_log_sigma = 0.62", "income_log_sigma = inf", 1),
            original.replace("population_weight = 0.25", "population_weight = nan", 1),
            original.replace(
                "age_band_weights = [0.07,",
                "age_band_weights = [inf,",
                1,
            ),
        )
        for text in variants:
            with self.subTest(marker=text[:80]):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "jurisdictions.toml"
                    path.write_text(text, encoding="utf-8")
                    with self.assertRaises(ProfileValidationError):
                        load_profile_bundle(path, SOURCES)

    def test_sources_contracts_rules_and_synthetic_audit_are_integral(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)

        self.assertGreater(len(bundle.sources), 20)
        for profile in bundle.country_profiles:
            self.assertTrue(profile.source_ids)
            self.assertTrue(set(profile.source_ids).issubset(bundle.sources))
        for contract in bundle.contracts:
            self.assertTrue(contract.condition.strip())
            self.assertTrue(contract.denominator.strip())
            self.assertIsInstance(contract.status, ProvenanceStatus)
            self.assertTrue(set(contract.source_ids).issubset(bundle.sources))

        states = {state.code: state for state in bundle.state_agents}
        self.assertFalse(states["UK"].rules.odds_disclosure_required)
        self.assertTrue(states["BE"].rules.paid_random_rewards_restricted)
        self.assertTrue(states["JP"].rules.odds_disclosure_required)
        self.assertTrue(
            all(state.state.audit_capacity_per_cycle == 2 for state in states.values())
        )
        audit_contract = next(
            contract
            for contract in bundle.contracts
            if contract.metric == "audit_capacity_per_cycle"
        )
        self.assertEqual(audit_contract.status, ProvenanceStatus.SYNTHETIC)

        contracts = {
            (contract.jurisdiction_code, contract.metric): contract
            for contract in bundle.contracts
        }
        self.assertEqual(
            contracts[("KR", "odds_disclosure_required")].source_ids,
            ("MCST_ODDS_DISCLOSURE_2024",),
        )
        self.assertEqual(
            contracts[("JP", "complete_gacha_restricted")].source_ids,
            ("JP_COMPLETE_GACHA_FAQ",),
        )
        self.assertEqual(
            contracts[("BE", "paid_random_rewards_restricted")].status,
            ProvenanceStatus.ILLUSTRATIVE,
        )

        self.assertEqual(
            load_country_profiles(JURISDICTIONS, SOURCES), bundle.country_profiles
        )

    def test_unknown_source_reference_is_rejected(self) -> None:
        text = JURISDICTIONS.read_text(encoding="utf-8").replace(
            'income_source = "ONS_HDI_FYE2024"',
            'income_source = "MISSING_SOURCE"',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            bad_path = Path(directory) / "jurisdictions.toml"
            bad_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "MISSING_SOURCE"):
                load_profile_bundle(bad_path, SOURCES)

    def test_rule_source_must_declare_compatible_scope(self) -> None:
        text = JURISDICTIONS.read_text(encoding="utf-8").replace(
            'odds_disclosure_required_source = "MCST_ODDS_DISCLOSURE_2024"',
            'odds_disclosure_required_source = "JP_COMPLETE_GACHA_FAQ"',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            bad_path = Path(directory) / "jurisdictions.toml"
            bad_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "compatible scope"):
                load_profile_bundle(bad_path, SOURCES)

    def test_source_catalogue_retrieval_date_must_be_canonical_iso_date(self) -> None:
        text = SOURCES.read_text(encoding="utf-8").replace(
            'retrieved_on = "2026-08-24"',
            'retrieved_on = "20260824"',
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            bad_path = Path(directory) / "sources.toml"
            bad_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ProfileValidationError, "ISO date"):
                load_profile_bundle(JURISDICTIONS, bad_path)

    def test_conversion_source_scope_and_retrieval_date_are_validated(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        scale = bundle.money_scale("UK")
        incompatible_source = _test_conversion_source(
            bundle,
            supports=("purchasing_power_parity",),
        )
        sources = dict(bundle.sources)
        sources[incompatible_source.id] = incompatible_source
        conversion = _conversion_for_scale(
            scale,
            source_id=incompatible_source.id,
        )
        with self.assertRaisesRegex(ProfileValidationError, "compatible scope"):
            replace(
                bundle,
                sources=sources,
                monetary_conversions=(conversion,),
            )
        sources[incompatible_source.id] = replace(
            incompatible_source,
            supports=("currency_conversion",),
        )
        with self.assertRaisesRegex(ProfileValidationError, "compatible scope"):
            replace(
                bundle,
                sources=sources,
                monetary_conversions=(conversion,),
            )

        compatible_source = replace(
            incompatible_source,
            supports=("foreign_exchange_rate",),
        )
        sources[compatible_source.id] = compatible_source
        sources[compatible_source.id] = replace(
            compatible_source,
            period="different test period",
        )
        with self.assertRaisesRegex(
            ProfileValidationError,
            "rate period does not match",
        ):
            replace(
                bundle,
                sources=sources,
                monetary_conversions=(conversion,),
            )

        sources[compatible_source.id] = compatible_source
        wrong_date = replace(conversion, retrieved_on=date(2026, 8, 23))
        with self.assertRaisesRegex(
            ProfileValidationError,
            "retrieval date does not match",
        ):
            replace(
                bundle,
                sources=sources,
                monetary_conversions=(wrong_date,),
            )

        wrong_currency = replace(conversion, source_currency="EUR")
        with self.assertRaisesRegex(
            ProfileValidationError,
            "source currency EUR does not match money-scale currency GBP",
        ):
            replace(
                bundle,
                sources=sources,
                monetary_conversions=(wrong_currency,),
            )

    def test_registered_lineage_binds_every_published_file_claim(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        lineage = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )

        self.assertEqual(lineage.lineage_status, "registered_profile_bundle")
        with self.assertRaisesRegex(
            ProfileValidationError,
            "does not match its fingerprinted snapshot",
        ):
            replace(lineage, jurisdictions_sha256="0" * 64)
        with self.assertRaisesRegex(ProfileValidationError, "lineage status"):
            replace(lineage, lineage_status="unregistered_profile_bundle")

    def test_registered_lineage_constructor_rejects_nonexistent_claims(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        valid = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )
        with tempfile.TemporaryDirectory() as directory:
            missing_jurisdictions = str(Path(directory) / "missing-jurisdictions.toml")
            missing_sources = str(Path(directory) / "missing-sources.toml")
            snapshot = valid.snapshot
            file_lineage = snapshot["file_lineage"]
            assert isinstance(file_lineage, dict)
            file_lineage["jurisdictions"] = {
                "path": missing_jurisdictions,
                "sha256": "1" * 64,
            }
            file_lineage["source_registry"] = {
                "path": missing_sources,
                "sha256": "2" * 64,
                "retrieved_on": "2026-08-24",
            }
            snapshot_json = json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            with self.assertRaisesRegex(
                ProfileValidationError,
                "files are unavailable or invalid",
            ):
                ProfileInputLineage(
                    lineage_status="registered_profile_bundle",
                    profile_codes=valid.profile_codes,
                    fingerprint_sha256=sha256(
                        snapshot_json.encode("utf-8")
                    ).hexdigest(),
                    snapshot_json=snapshot_json,
                    jurisdictions_path=missing_jurisdictions,
                    jurisdictions_sha256="1" * 64,
                    source_registry_path=missing_sources,
                    source_registry_sha256="2" * 64,
                    source_retrieved_on=date(2026, 8, 24),
                )

    def test_registered_lineage_is_rechecked_when_manifest_is_built(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jurisdictions = root / "jurisdictions.toml"
            sources = root / "sources.toml"
            jurisdictions.write_bytes(JURISDICTIONS.read_bytes())
            sources.write_bytes(SOURCES.read_bytes())
            bundle = load_profile_bundle(jurisdictions, sources)
            lineage = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            jurisdictions.write_text(
                jurisdictions.read_text(encoding="utf-8") + "\n# changed\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ProfileValidationError,
                "no longer matches",
            ):
                lineage.manifest_payload()

    def test_profile_lineage_schema_v1_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jurisdictions = root / "jurisdictions.toml"
            sources = root / "sources.toml"
            jurisdictions.write_text(
                JURISDICTIONS.read_text(encoding="utf-8").replace(
                    "schema_version = 2",
                    "schema_version = 1",
                    1,
                ),
                encoding="utf-8",
            )
            sources.write_bytes(SOURCES.read_bytes())
            bundle = load_profile_bundle(jurisdictions, sources)
            current = build_profile_input_lineage(
                bundle.country_profiles,
                profile_bundle=bundle,
            )
            snapshot = current.snapshot
            snapshot["schema_version"] = 1
            bundle_snapshot = snapshot["profile_bundle"]
            assert isinstance(bundle_snapshot, dict)
            bundle_snapshot.pop("monetary_conversions")
            snapshot_json = json.dumps(
                snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            legacy = ProfileInputLineage(
                lineage_status=current.lineage_status,
                profile_codes=current.profile_codes,
                fingerprint_sha256=sha256(
                    snapshot_json.encode("utf-8")
                ).hexdigest(),
                snapshot_json=snapshot_json,
                jurisdictions_path=current.jurisdictions_path,
                jurisdictions_sha256=current.jurisdictions_sha256,
                source_registry_path=current.source_registry_path,
                source_registry_sha256=current.source_registry_sha256,
                source_retrieved_on=current.source_retrieved_on,
            )
            summary_count = legacy.manifest_payload()[
                "monetary_conversion_summary"
            ]["count"]

        self.assertEqual(legacy.snapshot["schema_version"], 1)
        self.assertEqual(summary_count, 0)

    def test_profile_lineage_schema_requires_a_strict_integer_version(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        current = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )
        for alias in (True, 2.0):
            with self.subTest(alias=alias):
                snapshot = current.snapshot
                snapshot["schema_version"] = alias
                snapshot_json = json.dumps(
                    snapshot,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                with self.assertRaisesRegex(
                    ProfileValidationError,
                    "unsupported profile snapshot schema version",
                ):
                    replace(
                        current,
                        snapshot_json=snapshot_json,
                        fingerprint_sha256=sha256(
                            snapshot_json.encode("utf-8")
                        ).hexdigest(),
                    )

    def test_lineage_snapshots_and_summarises_monetary_conversions(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        candidate = _coherent_campaign_candidate(bundle)

        lineage = build_profile_input_lineage(
            candidate.country_profiles,
            profile_bundle=candidate,
        )
        profile_bundle_snapshot = lineage.snapshot["profile_bundle"]
        assert isinstance(profile_bundle_snapshot, dict)
        conversions = profile_bundle_snapshot["monetary_conversions"]
        assert isinstance(conversions, list)
        self.assertEqual(
            [item["jurisdiction_code"] for item in conversions],
            ["UK", "KR", "JP", "BE"],
        )
        self.assertTrue(
            all(item["target_currency"] == "TST" for item in conversions)
        )
        self.assertTrue(
            all(
                item["rounding_method"]
                == "nearest_minor_unit_half_away_from_zero"
                for item in conversions
            )
        )
        self.assertTrue(
            all(
                item["rate_numerator_decimal"]
                == str(item["rate_numerator"])
                and item["rate_denominator_decimal"]
                == str(item["rate_denominator"])
                for item in conversions
            )
        )
        summary = lineage.manifest_payload()["monetary_conversion_summary"]
        self.assertEqual(
            summary,
            {
                "aggregation_units": ["one test-only jurisdiction-seed total"],
                "comparison_groups": ["test-only common basis"],
                "count": 4,
                "estimands": ["test-only comparable monetary amount"],
                "methods": ["FX"],
                "population_bases": ["test-only common population"],
                "rate_period_ends": ["2025-12-31"],
                "rate_period_starts": ["2025-01-01"],
                "retrieval_dates": ["2026-08-24"],
                "rounding_scopes": ["AFTER_AGGREGATION"],
                "source_currencies": ["EUR", "GBP", "JPY", "KRW"],
                "target_currencies": ["TST"],
                "target_price_period_ends": ["2025-12-31"],
                "target_price_period_starts": ["2025-01-01"],
                "status_counts": {"CALIBRATED": 4},
            },
        )

        without_conversions = _campaign_candidate(bundle)
        baseline_lineage = build_profile_input_lineage(
            without_conversions.country_profiles,
            profile_bundle=without_conversions,
        )
        self.assertNotEqual(
            lineage.fingerprint_sha256,
            baseline_lineage.fingerprint_sha256,
        )
        large_rate = (1 << 60) + 1
        large_conversion = replace(
            candidate.monetary_conversions[0],
            rate_numerator=large_rate,
        )
        large_candidate = replace(
            candidate,
            monetary_conversions=(
                large_conversion,
                *candidate.monetary_conversions[1:],
            ),
        )
        large_lineage = build_profile_input_lineage(
            large_candidate.country_profiles,
            profile_bundle=large_candidate,
        )
        large_bundle_snapshot = large_lineage.snapshot["profile_bundle"]
        assert isinstance(large_bundle_snapshot, dict)
        large_snapshot = large_bundle_snapshot["monetary_conversions"][0]
        self.assertEqual(large_snapshot["rate_numerator"], large_rate)
        self.assertEqual(large_snapshot["rate_numerator_decimal"], str(large_rate))

    def test_manual_or_incomplete_bundle_cannot_claim_registered_files(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        incomplete = replace(
            bundle,
            jurisdictions_path=None,
            source_registry_path=None,
            jurisdictions_sha256=None,
            source_registry_sha256=None,
        )
        incomplete_lineage = build_profile_input_lineage(
            incomplete.country_profiles,
            profile_bundle=incomplete,
        )
        self.assertEqual(
            incomplete_lineage.lineage_status,
            "unregistered_profile_bundle",
        )
        self.assertIsNone(incomplete_lineage.jurisdictions_path)
        self.assertIsNone(incomplete_lineage.source_registry_sha256)

        changed_profiles = (
            replace(bundle.country_profiles[0], awareness_mean=0.51),
            *bundle.country_profiles[1:],
        )
        changed_bundle = replace(bundle, country_profiles=changed_profiles)
        changed_lineage = build_profile_input_lineage(
            changed_profiles,
            profile_bundle=changed_bundle,
        )
        self.assertEqual(
            changed_lineage.lineage_status,
            "unregistered_profile_bundle",
        )
        self.assertIsNone(changed_lineage.jurisdictions_sha256)
        self.assertIsNone(changed_lineage.source_registry_path)

    def test_profile_bundle_rejects_mutable_outer_collections(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        for field, value in (
            ("country_profiles", list(bundle.country_profiles)),
            ("monetary_conversions", list(bundle.monetary_conversions)),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ProfileValidationError, "immutable tuple"):
                    replace(bundle, **{field: value})

    def test_campaign_rejects_every_non_calibrated_dependency(self) -> None:
        with self.assertRaisesRegex(
            ProfileValidationError,
            "monetary comparability: UK.monetary_conversion=missing",
        ):
            load_profile_bundle(JURISDICTIONS, SOURCES, campaign=True)
        with self.assertRaisesRegex(ProfileValidationError, "CALIBRATED"):
            load_country_profiles(JURISDICTIONS, SOURCES, campaign=True)

    def test_campaign_rejects_missing_monetary_conversion_coverage(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        candidate = _campaign_candidate(bundle)

        with self.assertRaisesRegex(
            ProfileValidationError,
            "UK.monetary_conversion=missing",
        ):
            candidate.validate_for_campaign()

    def test_coherent_test_only_conversions_clear_structure_not_evidence(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        candidate = _coherent_campaign_candidate(bundle)

        candidate.validate_monetary_contract_structure()
        with self.assertRaisesRegex(
            ProfileValidationError,
            "source_rate_binding=missing",
        ):
            candidate.validate_monetary_comparability_for_campaign()
        with self.assertRaisesRegex(
            ProfileValidationError,
            "profile_file_lineage=unregistered_or_changed",
        ):
            candidate.validate_for_campaign()
        self.assertEqual(
            tuple(
                candidate.monetary_conversion(code).jurisdiction_code
                for code in ("UK", "KR", "JP", "BE")
            ),
            ("UK", "KR", "JP", "BE"),
        )
        self.assertEqual(
            {
                scale.currency_scale_to_sim
                / candidate.monetary_conversion(
                    scale.jurisdiction_code
                ).conversion_ratio
                for scale in candidate.money_scales
            },
            {Fraction(1, 1)},
        )

    def test_campaign_rejects_inconsistent_conversion_basis(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        candidate = _coherent_campaign_candidate(bundle)
        changes = (
            {"target_currency": "ZZZ"},
            {"estimand": "different test estimand"},
            {"population_base": "different test population"},
            {"comparison_group": "different test-only basis"},
            {"rounding_scope": MonetaryRoundingScope.PER_OBSERVATION},
            {"aggregation_unit": "different aggregation unit"},
        )
        for change in changes:
            with self.subTest(change=change):
                changed = replace(candidate.monetary_conversions[-1], **change)
                inconsistent = replace(
                    candidate,
                    monetary_conversions=(
                        *candidate.monetary_conversions[:-1],
                        changed,
                    ),
                )

                with self.assertRaisesRegex(
                    ProfileValidationError,
                    "monetary_conversion.comparison_basis=inconsistent",
                ):
                    inconsistent.validate_monetary_comparability_for_campaign()

    def test_campaign_rejects_incoherent_internal_monetary_scale(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        candidate = _coherent_campaign_candidate(bundle)
        final = candidate.monetary_conversions[-1]
        changed = replace(final, rate_numerator=final.rate_numerator + 1)
        incoherent = replace(
            candidate,
            monetary_conversions=(*candidate.monetary_conversions[:-1], changed),
        )

        with self.assertRaisesRegex(
            ProfileValidationError,
            "monetary_conversion.internal_scale=incoherent",
        ):
            incoherent.validate_monetary_comparability_for_campaign()

    def test_campaign_checks_source_referenced_only_by_money_scale(self) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        source_id = "ONS_HDI_FYE2024"
        candidate = _campaign_candidate(
            bundle,
            scale_source_id=source_id,
        )

        with self.assertRaisesRegex(
            ProfileValidationError,
            rf"source:{source_id}=ANCHORED",
        ):
            candidate.validate_for_campaign()


def _campaign_candidate(
    bundle: ProfileBundle,
    *,
    scale_source_id: str | None = None,
    extra_sources: tuple[SourceProvenance, ...] = (),
    monetary_conversions: tuple[MonetaryConversionContract, ...] = (),
) -> ProfileBundle:
    """Build a narrow gate fixture with no failures except those under test."""

    profiles = tuple(
        replace(profile, source_ids=()) for profile in bundle.country_profiles
    )
    contracts = tuple(
        replace(
            contract,
            status=ProvenanceStatus.CALIBRATED,
            source_ids=(),
        )
        for contract in bundle.contracts
    )
    scales = tuple(
        replace(
            scale,
            anchor_status=ProvenanceStatus.CALIBRATED,
            scale_status=ProvenanceStatus.CALIBRATED,
            source_ids=(scale_source_id,) if index == 0 and scale_source_id else (),
        )
        for index, scale in enumerate(bundle.money_scales)
    )
    sources = dict(bundle.sources)
    for source in extra_sources:
        if source.id in sources:
            raise AssertionError(f"duplicate test source id: {source.id}")
        sources[source.id] = source
    return ProfileBundle(
        country_profiles=profiles,
        state_agents=bundle.state_agents,
        sources=sources,
        profile_status=ProvenanceStatus.CALIBRATED,
        caveats=bundle.caveats,
        contracts=contracts,
        money_scales=scales,
        monetary_conversions=monetary_conversions,
    )


def _test_conversion_source(
    bundle: ProfileBundle,
    *,
    supports: tuple[str, ...] = ("foreign_exchange_rate",),
) -> SourceProvenance:
    """Return a calibrated source used only to exercise gate mechanics."""

    retrieved_on = bundle.source_retrieved_on
    if retrieved_on is None:
        raise AssertionError("loaded profile fixture must retain a retrieval date")
    return SourceProvenance(
        id=_TEST_ONLY_SOURCE_ID,
        publisher="Test fixture",
        title="Test-only monetary conversion contract",
        url="https://example.invalid/test-only-monetary-conversion",
        period="2025-01-01/2025-12-31",
        geography="Test jurisdictions",
        supports=supports,
        calibration_status=ProvenanceStatus.CALIBRATED,
        retrieved_on=retrieved_on,
    )


def _conversion_for_scale(
    scale: MoneyScaleContract,
    *,
    source_id: str,
    retrieved_on: date = date(2026, 8, 24),
) -> MonetaryConversionContract:
    """Make an algebraically coherent test contract, never an empirical rate."""

    return MonetaryConversionContract(
        jurisdiction_code=scale.jurisdiction_code,
        source_currency=scale.currency,
        target_currency="TST",
        method=MonetaryConversionMethod.FX,
        rate_numerator=scale.simulation_monthly_anchor_cents,
        rate_denominator=scale.nominal_monthly_anchor_minor_units,
        rate_period_start=date(2025, 1, 1),
        rate_period_end=date(2025, 12, 31),
        target_price_period_start=date(2025, 1, 1),
        target_price_period_end=date(2025, 12, 31),
        estimand="test-only comparable monetary amount",
        population_base="test-only common population",
        comparison_group="test-only common basis",
        rounding_method="nearest_minor_unit_half_away_from_zero",
        rounding_scope=MonetaryRoundingScope.AFTER_AGGREGATION,
        aggregation_unit="one test-only jurisdiction-seed total",
        status=ProvenanceStatus.CALIBRATED,
        source_ids=(source_id,),
        retrieved_on=retrieved_on,
        notes="Algebraic gate fixture only; not an observed exchange rate.",
    )


def _coherent_campaign_candidate(bundle: ProfileBundle) -> ProfileBundle:
    """Promote every dependency using explicit, test-only coherent contracts."""

    source = _test_conversion_source(bundle)
    assert source.retrieved_on is not None
    conversions = tuple(
        _conversion_for_scale(
            scale,
            source_id=source.id,
            retrieved_on=source.retrieved_on,
        )
        for scale in bundle.money_scales
    )
    return _campaign_candidate(
        bundle,
        extra_sources=(source,),
        monetary_conversions=conversions,
    )


if __name__ == "__main__":
    unittest.main()
