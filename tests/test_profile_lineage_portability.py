from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import unittest

from microtx_sim.cli import _load_policy_profiles
from microtx_sim.data.lineage import (
    _profile_lineage_fingerprint_sha256,
    _profile_lineage_fingerprint_sha256_v1,
    build_profile_input_lineage,
    profile_lineage_fingerprint_matches,
)
from microtx_sim.policy_config import load_policy_config


ROOT = Path(__file__).resolve().parents[1]


class ProfileLineagePortabilityTests(unittest.TestCase):
    def test_nested_evidence_bundle_paths_are_worktree_portable(self) -> None:
        windows = {
            "file_lineage": {
                "population_bundle": {
                    "path": (
                        r"C:\Users\researcher\microtx-sim\inputs\population.toml"
                    )
                }
            },
            "profile_bundle": {
                "source_evidence_bundle": {
                    "bundle_path": (
                        r"C:\Users\researcher\microtx-sim\inputs\rates.toml"
                    )
                },
                "population_evidence_bundle": {
                    "bundle_path": (
                        r"C:\Users\researcher\microtx-sim\inputs\population.toml"
                    ),
                    "artifacts": [{"relative_path": "joint.csv"}],
                }
            },
        }
        linux = {
            "file_lineage": {
                "population_bundle": {
                    "path": "/home/runner/microtx-sim/inputs/population.toml"
                }
            },
            "profile_bundle": {
                "source_evidence_bundle": {
                    "bundle_path": "/home/runner/microtx-sim/inputs/rates.toml"
                },
                "population_evidence_bundle": {
                    "bundle_path": "/home/runner/microtx-sim/inputs/population.toml",
                    "artifacts": [{"relative_path": "joint.csv"}],
                }
            },
        }

        self.assertEqual(
            _profile_lineage_fingerprint_sha256(windows),
            _profile_lineage_fingerprint_sha256(linux),
        )
        self.assertNotEqual(
            _profile_lineage_fingerprint_sha256_v1(windows),
            _profile_lineage_fingerprint_sha256_v1(linux),
        )

        changed_artifact = {
            **linux,
            "profile_bundle": {
                **linux["profile_bundle"],
                "population_evidence_bundle": {
                    **linux["profile_bundle"]["population_evidence_bundle"],
                    "artifacts": [{"relative_path": "different.csv"}],
                }
            },
        }
        self.assertNotEqual(
            _profile_lineage_fingerprint_sha256(linux),
            _profile_lineage_fingerprint_sha256(changed_artifact),
        )

    def test_previous_portable_fingerprint_remains_verifiable(self) -> None:
        config = load_policy_config(ROOT / "configs" / "policy_prospective.toml")
        profiles = _load_policy_profiles(config)
        current = build_profile_input_lineage(
            profiles.country_profiles,
            profile_bundle=profiles,
        )
        previous_fingerprint = _profile_lineage_fingerprint_sha256_v1(
            current.snapshot
        )

        previous = replace(
            current,
            fingerprint_sha256=previous_fingerprint,
        )

        self.assertEqual(previous.fingerprint_sha256, previous_fingerprint)
        self.assertNotEqual(previous.fingerprint_sha256, current.fingerprint_sha256)

        raw_legacy_fingerprint = sha256(
            current.snapshot_json.encode("utf-8")
        ).hexdigest()
        raw_legacy = replace(
            current,
            fingerprint_sha256=raw_legacy_fingerprint,
        )
        self.assertEqual(
            raw_legacy.fingerprint_sha256,
            raw_legacy_fingerprint,
        )

    def test_only_attested_directional_migrations_match(self) -> None:
        migrations = (
            (
                "8458d4c844e4a1e810d76e0a83e41e742d97e595373432b95e9e493322232dd4",
                "ce1c4592c3968215f6ec9fa9b7d907f42fc25feca4e9c5f795b2e72244c9ff56",
            ),
            (
                "119e5a9cbc919808520c395b4346d50e4a12fe9d5ec095f76816ad7c1fe38658",
                "5abda0b7383ba4051889bf05aa53f3faff729e2a564b5cac9864617b972f42e8",
            ),
        )
        for legacy, canonical in migrations:
            with self.subTest(legacy=legacy):
                self.assertTrue(
                    profile_lineage_fingerprint_matches(legacy, canonical)
                )
                self.assertTrue(
                    profile_lineage_fingerprint_matches(canonical, canonical)
                )
                self.assertFalse(
                    profile_lineage_fingerprint_matches(canonical, legacy)
                )

        self.assertFalse(
            profile_lineage_fingerprint_matches("a" * 64, "b" * 64)
        )


if __name__ == "__main__":
    unittest.main()
