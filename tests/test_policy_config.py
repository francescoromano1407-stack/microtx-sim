from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from microtx_sim.policy_config import (
    PolicyConfigurationError,
    load_policy_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "policy_prototype.toml"


class PolicyConfigTests(unittest.TestCase):
    def test_checked_in_policy_config_is_strict_and_complete(self) -> None:
        config = load_policy_config(CONFIG)
        self.assertEqual(config.provenance_status, "synthetic")
        self.assertEqual(len(config.batch.scenarios), 7)
        self.assertEqual(config.batch.seeds, (101, 202, 303))
        self.assertEqual(config.batch.player_count, 1000)
        self.assertFalse(
            any(item.mechanics.personalized_offers for item in config.batch.scenarios)
        )
        self.assertGreater(config.epgc_policy.maximum_budget_cents, 0)

    def test_unknown_or_missing_toml_keys_are_rejected(self) -> None:
        original = CONFIG.read_text("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.toml"
            unknown.write_text(
                original.replace(
                    'run_sensitivity = true',
                    'run_sensitivity = true\nunknown_option = 1',
                ),
                "utf-8",
            )
            with self.assertRaises(PolicyConfigurationError):
                load_policy_config(unknown)
            missing = Path(directory) / "missing.toml"
            missing.write_text(
                original.replace('histogram_bins = 20\n', ''), "utf-8"
            )
            with self.assertRaises(PolicyConfigurationError):
                load_policy_config(missing)


if __name__ == "__main__":
    unittest.main()
