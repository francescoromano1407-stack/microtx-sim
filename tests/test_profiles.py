from __future__ import annotations

from dataclasses import replace
from datetime import date
from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from microtx_sim.data import build_profile_input_lineage
from microtx_sim.data.profiles import (
    ProfileBundle,
    ProfileValidationError,
    load_country_profiles,
    load_profile_bundle,
)
from microtx_sim.types import ProvenanceStatus


ROOT = Path(__file__).resolve().parents[1]
JURISDICTIONS = ROOT / "configs" / "jurisdictions.toml"
SOURCES = ROOT / "data" / "provenance" / "sources.toml"


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

    def test_campaign_rejects_every_non_calibrated_dependency(self) -> None:
        with self.assertRaisesRegex(ProfileValidationError, "CALIBRATED"):
            load_profile_bundle(JURISDICTIONS, SOURCES, campaign=True)
        with self.assertRaisesRegex(ProfileValidationError, "CALIBRATED"):
            load_country_profiles(JURISDICTIONS, SOURCES, campaign=True)

    def test_campaign_rejects_noncomparable_multi_jurisdiction_money_scales(
        self,
    ) -> None:
        bundle = load_profile_bundle(JURISDICTIONS, SOURCES)
        candidate = _campaign_candidate(bundle)

        with self.assertRaisesRegex(
            ProfileValidationError,
            "cross_country_comparable=false",
        ):
            candidate.validate_for_campaign()

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
    return ProfileBundle(
        country_profiles=profiles,
        state_agents=bundle.state_agents,
        sources=bundle.sources,
        profile_status=ProvenanceStatus.CALIBRATED,
        caveats=bundle.caveats,
        contracts=contracts,
        money_scales=scales,
    )


if __name__ == "__main__":
    unittest.main()
