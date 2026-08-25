from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.cli import main


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "policy_prototype.toml"


class PolicyCliTests(unittest.TestCase):
    def test_smoke_command_is_deterministic_except_for_elapsed_time(self) -> None:
        payloads: list[dict[str, object]] = []
        for _ in range(2):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(("smoke", str(ROOT / "configs" / "smoke.toml")))
            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            payload.pop("elapsed_seconds")
            payloads.append(payload)

        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual(payloads[0]["mode"], "smoke_only")
        self.assertEqual(payloads[0]["cycles"], 3)
        summary = payloads[0]["summary"]
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["players"], 384)

    def test_policy_validate_and_small_batch_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            config_path = root / "small.toml"
            text = BASE_CONFIG.read_text("utf-8")
            text = text.replace("seeds = [101, 202, 303]", "seeds = [7]")
            text = text.replace("days = 14", "days = 1")
            text = text.replace("player_count = 1000", "player_count = 12")
            text = text.replace("step_minutes = 60", "step_minutes = 240")
            text = text.replace(
                'output_dir = "artifacts/policy_prototype"',
                f'output_dir = "{output.as_posix()}"',
            )
            text = text.replace("run_sensitivity = true", "run_sensitivity = false")
            config_path.write_text(text, "utf-8")

            validation_stdout = io.StringIO()
            with redirect_stdout(validation_stdout):
                code = main(("policy-validate", str(config_path)))
            self.assertEqual(code, 0)
            validation = json.loads(validation_stdout.getvalue())
            self.assertEqual(validation["scenario_count"], 7)
            self.assertFalse(validation["empirical_validation_claimed"])

            batch_stdout = io.StringIO()
            with redirect_stdout(batch_stdout):
                code = main(
                    (
                        "policy-batch",
                        str(config_path),
                        "--skip-sensitivity",
                    )
                )
            self.assertEqual(code, 0)
            payload = json.loads(batch_stdout.getvalue())
            self.assertEqual(payload["scenario_count"], 7)
            self.assertFalse(payload["sensitivity_run"])
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "scenario_summary.csv").exists())
            self.assertTrue((output / "harm_revenue_frontier.svg").exists())
            manifest = json.loads((output / "manifest.json").read_text("utf-8"))
            self.assertEqual(
                manifest["profile_inputs"]["lineage_status"],
                "registered_profile_bundle",
            )
            self.assertEqual(
                manifest["batch"]["profile_codes"],
                ["UK", "KR", "JP", "BE"],
            )
            self.assertEqual(
                manifest["source_registry_retrieved_on"],
                "2026-08-24",
            )
            self.assertEqual(
                manifest["batch"]["profile_input_fingerprint_sha256"],
                manifest["profile_inputs"]["fingerprint_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
