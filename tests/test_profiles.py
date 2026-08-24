from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from microtx_sim.data.profiles import (
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
        self.assertEqual(nominal["UK"], 3_055)  # 36,663 / 12, nearest unit
        self.assertEqual(nominal["KR"], 3_515_000)  # central monthly quintile
        self.assertEqual(nominal["JP"], 300_000)
        self.assertEqual(nominal["BE"], 2_608)  # 31,299 / 12, nearest unit
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

    def test_campaign_rejects_every_non_calibrated_dependency(self) -> None:
        with self.assertRaisesRegex(ProfileValidationError, "CALIBRATED"):
            load_profile_bundle(JURISDICTIONS, SOURCES, campaign=True)
        with self.assertRaisesRegex(ProfileValidationError, "CALIBRATED"):
            load_country_profiles(JURISDICTIONS, SOURCES, campaign=True)


if __name__ == "__main__":
    unittest.main()
