from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import unittest

from microtx_sim.causal.batch import _cohort_digest
from microtx_sim.consumers.population import initialize_player_table
from microtx_sim.consumers.welfare import initialize_player_life
from microtx_sim.data.lineage import ProfileInputLineage, build_profile_input_lineage
from microtx_sim.data.profiles import load_profile_bundle
from microtx_sim.rng import CounterRNG


_FIXTURES = Path(__file__).parent / "fixtures"
_REGISTERED_LEGACY_COHORT_DIGESTS = {
    101: "c81c33c2e2d749d94c5fdb6c3ac18cf30ae02b18c628fd6727240bbb862c8117",
    202: "4bc170e57ea43ed70019ae554e3a6a31efb78345dba86d981417c4b87eaa1910",
    303: "11cf006d488b9fd8f7dba292f03e35bc48ffe4dfe431d62ff9ece88b533cea6b",
}
_REGISTERED_PROFILE_LINEAGE_V3_PROJECTION_SHA256 = (
    "a7631f28f0d3e0172e292612e3e1813891da712e30693214262c0974eed00ad9"
)


def _normalized_registered_v3_projection_sha(snapshot: dict[str, object]) -> str:
    """Fingerprint the v3 surface while excluding host-specific input paths."""

    normalized = json.loads(
        json.dumps(
            snapshot,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    file_lineage = normalized["file_lineage"]
    file_lineage["jurisdictions"]["path"] = "<JURISDICTIONS>"
    file_lineage["source_registry"]["path"] = "<SOURCES>"
    file_lineage["source_bundle"]["path"] = "<SOURCE_BUNDLE>"
    source_evidence = normalized["profile_bundle"]["source_evidence_bundle"]
    source_evidence["bundle_path"] = "<SOURCE_BUNDLE>"
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


class RegisteredLegacyPopulationRegressionTests(unittest.TestCase):
    def test_default_1000_player_cohort_digests_remain_frozen(self) -> None:
        bundle = load_profile_bundle(campaign=False)
        self.assertTrue(bundle.matches_registered_files())

        for seed, expected_digest in _REGISTERED_LEGACY_COHORT_DIGESTS.items():
            with self.subTest(seed=seed):
                rng = CounterRNG(seed)
                players = initialize_player_table(
                    1_000,
                    bundle.country_profiles,
                    rng,
                )
                life = initialize_player_life(players, rng)

                self.assertEqual(_cohort_digest(players, life), expected_digest)

    def test_frozen_unregistered_profile_lineage_v3_fixture_remains_readable(
        self,
    ) -> None:
        snapshot_json = (
            _FIXTURES / "profile_lineage_v3.json"
        ).read_text(encoding="utf-8").strip()
        lineage = ProfileInputLineage(
            lineage_status="unregistered_custom_profiles",
            profile_codes=("ZZ",),
            fingerprint_sha256=sha256(snapshot_json.encode("utf-8")).hexdigest(),
            snapshot_json=snapshot_json,
        )

        self.assertEqual(lineage.snapshot["schema_version"], 3)
        self.assertEqual(
            lineage.manifest_payload()["monetary_conversion_summary"]["count"],
            0,
        )

    def test_registered_profile_lineage_v3_projection_remains_frozen(self) -> None:
        bundle = load_profile_bundle(campaign=False)
        lineage = build_profile_input_lineage(
            bundle.country_profiles,
            profile_bundle=bundle,
        )

        self.assertEqual(lineage.lineage_status, "registered_profile_bundle")
        self.assertEqual(lineage.snapshot["schema_version"], 4)

        projection = json.loads(lineage.snapshot_json)
        projection["schema_version"] = 3
        projection["file_lineage"].pop("population_bundle")
        projection["profile_bundle"].pop("population_evidence_bundle")
        projection["profile_bundle"].pop("population_evidence_results")
        projection["profile_bundle"].pop("population_evidence_assessment")
        projection_json = json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        legacy = ProfileInputLineage(
            lineage_status=lineage.lineage_status,
            profile_codes=lineage.profile_codes,
            fingerprint_sha256=sha256(
                projection_json.encode("utf-8")
            ).hexdigest(),
            snapshot_json=projection_json,
            jurisdictions_path=lineage.jurisdictions_path,
            jurisdictions_sha256=lineage.jurisdictions_sha256,
            source_registry_path=lineage.source_registry_path,
            source_registry_sha256=lineage.source_registry_sha256,
            source_retrieved_on=lineage.source_retrieved_on,
            source_bundle_path=lineage.source_bundle_path,
            source_bundle_sha256=lineage.source_bundle_sha256,
        )

        self.assertEqual(legacy.snapshot["schema_version"], 3)
        self.assertFalse(
            legacy.manifest_payload()["population_evidence_summary"]["present"]
        )
        self.assertEqual(
            _normalized_registered_v3_projection_sha(legacy.snapshot),
            _REGISTERED_PROFILE_LINEAGE_V3_PROJECTION_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
