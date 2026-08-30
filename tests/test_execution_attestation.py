from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import microtx_sim.execution_attestation as attestation_module
from microtx_sim.execution_attestation import (
    EXECUTION_RECEIPT_SCHEMA_ID,
    EXECUTION_RECEIPT_SCHEMA_VERSION,
    REQUIRED_FILE_ARTIFACT_IDS,
    REQUIRED_SEMANTIC_IDENTITY_IDS,
    SOURCE_TREE_ALGORITHM,
    CampaignExecutionRejectedError,
    DeclaredIdentity,
    ExecutionAttestationError,
    ExecutionReceipt,
    ExecutionReceiptMismatchError,
    ExecutionReceiptSpec,
    ExecutionReceiptVerification,
    ExecutionVerificationPhase,
    FileArtifactSpec,
    attach_verified_execution_receipt,
    build_execution_receipt,
    load_execution_receipt,
    require_campaign_execution,
    verify_execution_receipt,
    write_execution_attestation_atomic,
    write_execution_receipt_atomic,
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ("git", *args),
        cwd=repository,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            completed.stderr.decode("utf-8", errors="replace")
        )
    return completed.stdout.decode("utf-8", errors="strict").strip()


class ExecutionAttestationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.repository = Path(self._temporary.name).resolve()
        _git(self.repository, "init", "-b", "main")
        _git(self.repository, "config", "user.email", "attestation@example.invalid")
        _git(self.repository, "config", "user.name", "Attestation Test")
        _git(self.repository, "config", "core.autocrlf", "false")

        (self.repository / ".gitignore").write_text(
            "/run-artifacts/\n", encoding="utf-8", newline=""
        )
        (self.repository / "pyproject.toml").write_text(
            "[project]\nname = \"attestation-fixture\"\nversion = \"1.0\"\n",
            encoding="utf-8",
            newline="",
        )
        (self.repository / "uv.lock").write_text(
            "version = 1\nrevision = 1\n",
            encoding="utf-8",
            newline="",
        )
        input_root = self.repository / "inputs" / "attestation"
        input_root.mkdir(parents=True)
        artifacts: list[FileArtifactSpec] = []
        for artifact_id in sorted(REQUIRED_FILE_ARTIFACT_IDS):
            relative = Path("inputs") / "attestation" / f"{artifact_id}.artifact"
            content = f"artifact_id={artifact_id}\nschema_version=fixture-v1\n".encode()
            (self.repository / relative).write_bytes(content)
            artifacts.append(
                FileArtifactSpec(
                    artifact_id=artifact_id,
                    path=relative,
                    expected_sha256=sha256(content).hexdigest(),
                    expected_byte_length=len(content),
                    schema_version="fixture-v1",
                )
            )

        plan_sha256 = _digest("canonical prospective plan fixture")
        identities = tuple(
            DeclaredIdentity(
                identity_id=identity_id,
                schema_version="fixture-v1",
                sha256=(
                    plan_sha256
                    if identity_id == "prospective_plan_semantic"
                    else _digest(f"semantic:{identity_id}")
                ),
            )
            for identity_id in sorted(REQUIRED_SEMANTIC_IDENTITY_IDS)
        )
        _git(self.repository, "add", "--all")
        _git(self.repository, "commit", "-m", "attestation fixture")
        self.spec = ExecutionReceiptSpec(
            repository_root=self.repository,
            input_artifacts=tuple(artifacts),
            input_identities=identities,
            plan_id="fixture.prospective.plan.v1",
            plan_sha256=plan_sha256,
            expected_output_artifacts=(
                "manifest.json",
                "execution_receipt.json",
                "seed_results.csv",
            ),
            ledger_backend="sqlite",
            ledger_configuration={
                "path": "state/campaign.sqlite3",
                "journal_mode": "WAL",
                "temporary": False,
            },
            run_command=("microtx-sim", "policy-run", "configs/policy_campaign.toml"),
            execution_mode="PRE_CAMPAIGN_VALIDATION_ONLY",
            model_version="microtx-sim/fixture-v1",
            scientific_readiness_blockers=(
                "monetary.rate_uncertainty=unquantified",
                "population.empirical_validation=missing",
            ),
        )

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def test_identity_is_canonical_deterministic_and_has_no_timestamp(self) -> None:
        first = build_execution_receipt(self.spec)
        reordered = replace(
            self.spec,
            input_artifacts=tuple(reversed(self.spec.input_artifacts)),
            input_identities=tuple(reversed(self.spec.input_identities)),
            ledger_configuration={
                "temporary": False,
                "journal_mode": "WAL",
                "path": "state/campaign.sqlite3",
            },
            expected_output_artifacts=tuple(
                reversed(self.spec.expected_output_artifacts)
            ),
        )
        second = build_execution_receipt(reordered)

        self.assertEqual(first.execution_receipt_sha256, second.execution_receipt_sha256)
        self.assertEqual(first.identity_payload_json, second.identity_payload_json)
        expected_json = json.dumps(
            first.identity_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.assertEqual(first.identity_payload_json, expected_json)
        self.assertEqual(
            first.identity_payload["source_tree"]["algorithm"],
            SOURCE_TREE_ALGORITHM,
        )
        self.assertTrue(first.identity_payload["repository"]["working_tree_clean"])
        self.assertNotIn("\\", first.identity_payload["repository"]["root"])
        self.assertFalse(first.campaign_execution_admissible)
        self.assertFalse(first.snapshot()["scientific_readiness"]["campaign_ready"])

        forbidden = {
            "timestamp",
            "created_at",
            "generated_at",
            "observed_at",
            "verified_at",
            "wall_clock",
        }

        def visit(value: object) -> None:
            if isinstance(value, dict):
                self.assertTrue(forbidden.isdisjoint(value))
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(first.snapshot())

    def test_dirty_or_detached_repository_is_rejected(self) -> None:
        (self.repository / "untracked.txt").write_text("dirty", encoding="utf-8")
        with self.assertRaisesRegex(ExecutionAttestationError, "clean working tree"):
            build_execution_receipt(self.spec)
        (self.repository / "untracked.txt").unlink()
        _git(self.repository, "checkout", "--detach")
        with self.assertRaisesRegex(ExecutionAttestationError, "Git identity command"):
            build_execution_receipt(self.spec)

    def test_pre_post_verification_and_manifest_attachment(self) -> None:
        receipt = build_execution_receipt(self.spec)
        pre = verify_execution_receipt(
            receipt,
            self.spec,
            phase=ExecutionVerificationPhase.PRE_EXECUTION,
        )
        post = verify_execution_receipt(
            receipt,
            self.spec,
            phase=ExecutionVerificationPhase.POST_EXECUTION,
        )
        self.assertEqual(pre.phase, ExecutionVerificationPhase.PRE_EXECUTION)
        manifest = {"manifest_schema_version": "fixture-v1", "campaign_ready": False}
        attached = attach_verified_execution_receipt(
            manifest,
            receipt=receipt,
            verification=post,
            receipt_relative_path="attestation/execution_receipt.json",
        )
        self.assertNotIn("execution_receipt", manifest)
        self.assertFalse(attached["campaign_ready"])
        self.assertFalse(attached["execution_receipt"]["campaign_ready"])
        self.assertEqual(attached["execution_attestation"]["phase"], "POST_EXECUTION")
        with self.assertRaisesRegex(ExecutionAttestationError, "POST_EXECUTION"):
            attach_verified_execution_receipt(
                manifest,
                receipt=receipt,
                verification=pre,
            )
        with self.assertRaisesRegex(ExecutionAttestationError, "campaign-ready"):
            attach_verified_execution_receipt(
                {"campaign_ready": True},
                receipt=receipt,
                verification=post,
            )
        forged_blockers = ExecutionReceiptVerification(
            phase=ExecutionVerificationPhase.POST_EXECUTION,
            expected_receipt_sha256=receipt.execution_receipt_sha256,
            observed_receipt_sha256=receipt.execution_receipt_sha256,
            campaign_execution_admissible=False,
            blockers=("different.blocker",),
        )
        with self.assertRaises(ExecutionReceiptMismatchError):
            attach_verified_execution_receipt(
                manifest,
                receipt=receipt,
                verification=forged_blockers,
            )

    def test_clean_source_commit_change_and_runtime_change_are_stale(self) -> None:
        receipt = build_execution_receipt(self.spec)
        (self.repository / ".gitignore").write_text(
            "/run-artifacts/\n/another-ignored-path/\n",
            encoding="utf-8",
            newline="",
        )
        _git(self.repository, "add", ".gitignore")
        _git(self.repository, "commit", "-m", "alter clean source identity")
        with self.assertRaisesRegex(
            ExecutionReceiptMismatchError,
            "repository|source_tree",
        ):
            verify_execution_receipt(
                receipt,
                self.spec,
                phase=ExecutionVerificationPhase.POST_EXECUTION,
            )

        current = build_execution_receipt(self.spec)
        original_runtime = attestation_module._python_runtime_identity

        def changed_runtime() -> dict[str, object]:
            value = original_runtime()
            return {**value, "compiler": str(value["compiler"]) + "-changed"}

        with patch.object(
            attestation_module,
            "_python_runtime_identity",
            side_effect=changed_runtime,
        ):
            with self.assertRaisesRegex(
                ExecutionReceiptMismatchError,
                "python_runtime",
            ):
                verify_execution_receipt(
                    current,
                    self.spec,
                    phase=ExecutionVerificationPhase.POST_EXECUTION,
                )

    def test_altered_or_external_artifact_is_rejected(self) -> None:
        selected = self.spec.input_artifacts[0]
        bad_artifact = replace(selected, expected_sha256="0" * 64)
        bad_spec = replace(
            self.spec,
            input_artifacts=(bad_artifact, *self.spec.input_artifacts[1:]),
        )
        with self.assertRaisesRegex(ExecutionAttestationError, "SHA-256 mismatch"):
            build_execution_receipt(bad_spec)

        with tempfile.TemporaryDirectory() as directory:
            outside = Path(directory) / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            external = replace(
                selected,
                path=outside,
                expected_sha256=sha256(b"{}").hexdigest(),
                expected_byte_length=2,
            )
            external_spec = replace(
                self.spec,
                input_artifacts=(external, *self.spec.input_artifacts[1:]),
            )
            with self.assertRaisesRegex(ExecutionAttestationError, "inside"):
                build_execution_receipt(external_spec)

    def test_unavailable_identity_remains_explicit_and_gate_stays_closed(self) -> None:
        identities = tuple(
            (
                DeclaredIdentity(
                    identity_id=item.identity_id,
                    schema_version=item.schema_version,
                    unavailable_reason="no admissible empirical design exists",
                )
                if item.identity_id == "population_balance"
                else item
            )
            for item in self.spec.input_identities
        )
        incomplete = replace(self.spec, input_identities=identities)
        first = build_execution_receipt(incomplete)
        second = build_execution_receipt(incomplete)
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertIn(
            "execution_identity.population_balance=unavailable:"
            "no admissible empirical design exists",
            first.input_completeness_blockers,
        )
        verification = verify_execution_receipt(
            first,
            incomplete,
            phase=ExecutionVerificationPhase.PRE_EXECUTION,
        )
        with self.assertRaises(CampaignExecutionRejectedError):
            require_campaign_execution(first, verification)

        missing = tuple(
            item
            for item in self.spec.input_identities
            if item.identity_id != "population_balance"
        )
        with self.assertRaisesRegex(ExecutionAttestationError, "incomplete"):
            replace(self.spec, input_identities=missing)

    def test_receipt_write_is_idempotent_and_never_overwrites(self) -> None:
        receipt = build_execution_receipt(self.spec)
        path = self.repository / "run-artifacts" / "execution_receipt.json"
        self.assertEqual(write_execution_receipt_atomic(path, receipt), path)
        original = path.read_bytes()
        self.assertEqual(write_execution_receipt_atomic(path, receipt), path)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(load_execution_receipt(path).snapshot(), receipt.snapshot())

        path.write_text("reserved evidence\n", encoding="utf-8", newline="")
        with self.assertRaisesRegex(ExecutionAttestationError, "refusing to overwrite"):
            write_execution_receipt_atomic(path, receipt)
        self.assertEqual(path.read_text(encoding="utf-8"), "reserved evidence\n")

    def test_post_execution_attestation_is_separate_and_evidence_preserving(
        self,
    ) -> None:
        receipt = build_execution_receipt(self.spec)
        verification = verify_execution_receipt(
            receipt,
            self.spec,
            phase=ExecutionVerificationPhase.POST_EXECUTION,
        )
        path = self.repository / "run-artifacts" / "execution-attestation.json"
        self.assertEqual(
            write_execution_attestation_atomic(path, verification),
            path,
        )
        expected = (
            json.dumps(
                verification.snapshot(),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(
            write_execution_attestation_atomic(path, verification),
            path,
        )

        altered = ExecutionReceiptVerification(
            phase=ExecutionVerificationPhase.PRE_EXECUTION,
            expected_receipt_sha256=receipt.execution_receipt_sha256,
            observed_receipt_sha256=receipt.execution_receipt_sha256,
            campaign_execution_admissible=False,
            blockers=verification.blockers,
        )
        with self.assertRaisesRegex(
            ExecutionAttestationError,
            "refusing to overwrite",
        ):
            write_execution_attestation_atomic(path, altered)
        self.assertEqual(path.read_bytes(), expected)

    def test_loader_rejects_tampering_unknown_fields_and_duplicate_keys(self) -> None:
        receipt = build_execution_receipt(self.spec)
        output = self.repository / "run-artifacts"
        output.mkdir()

        tampered = receipt.snapshot()
        tampered["scientific_readiness"]["campaign_ready"] = True
        path = output / "tampered.json"
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ExecutionAttestationError, "must be false"):
            load_execution_receipt(path)

        unknown = receipt.snapshot()
        unknown["unknown"] = "field"
        path.write_text(json.dumps(unknown), encoding="utf-8")
        with self.assertRaisesRegex(ExecutionAttestationError, "keys differ"):
            load_execution_receipt(path)

        path.write_text('{"same":1,"same":2}', encoding="utf-8")
        with self.assertRaisesRegex(ExecutionAttestationError, "repeats object key"):
            load_execution_receipt(path)

        mismatched = receipt.snapshot()
        mismatched["execution_receipt_sha256"] = "0" * 64
        with self.assertRaisesRegex(ExecutionAttestationError, "does not match"):
            ExecutionReceipt.from_snapshot(mismatched)

    def test_receipt_schema_matches_generated_contract(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "schemas"
            / "execution-receipt.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$id"], EXECUTION_RECEIPT_SCHEMA_ID)
        self.assertEqual(schema["properties"]["receipt_schema_version"]["const"], EXECUTION_RECEIPT_SCHEMA_VERSION)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["identity_payload"]["properties"]
            ["source_tree"]["properties"]["algorithm"]["const"],
            SOURCE_TREE_ALGORITHM,
        )
        receipt = build_execution_receipt(self.spec)
        if importlib.util.find_spec("jsonschema") is not None:
            import jsonschema  # type: ignore[import-not-found]

            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.Draft202012Validator(schema).validate(receipt.snapshot())
            verification = verify_execution_receipt(
                receipt,
                self.spec,
                phase=ExecutionVerificationPhase.POST_EXECUTION,
            )
            jsonschema.Draft202012Validator(
                schema["$defs"]["executionVerification"]
            ).validate(verification.snapshot())


if __name__ == "__main__":
    unittest.main()
