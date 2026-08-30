from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from microtx_sim.campaign_preflight import (
    FileHashExpectation,
    GitObservation,
    PreCampaignValidationError,
    PreCampaignValidationReport,
    PreCampaignValidationSpec,
    ProbeEvidence,
    SemanticArtifactObservation,
    SemanticIdentityExpectation,
    StrictValidationProbe,
    UncertaintyDeclaration,
    build_policy_campaign_execution_receipt_spec,
    build_policy_campaign_preflight_spec,
    observe_git_repository,
    observe_parameter_uncertainty_design,
    observe_prospective_analysis_plan,
    run_pre_campaign_validation,
    write_pre_campaign_validation_report,
)
from microtx_sim.execution_attestation import (
    REQUIRED_FILE_ARTIFACT_IDS,
    REQUIRED_SEMANTIC_IDENTITY_IDS,
    ExecutionAttestationError,
    ExecutionReceiptSpec,
    build_execution_receipt,
)
from microtx_sim.policy_config import load_policy_config


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _stable_git(root: Path, *, clean: bool = True) -> GitObservation:
    return GitObservation(
        repository_root=root,
        active_branch="main",
        exact_commit="1" * 40,
        object_format="sha1",
        status_porcelain="" if clean else "?? pending.txt",
        working_tree_clean=clean,
    )


class _FakeReceipt:
    execution_receipt_sha256 = _digest("fake execution receipt")
    identity_payload = {
        "source_tree": {"source_tree_sha256": _digest("fake source tree")}
    }


class _FakeVerification:
    blockers = ("execution_receipt.scientific_readiness=not_established",)


def _successful_receipt_builder(_spec: object) -> _FakeReceipt:
    return _FakeReceipt()


def _successful_receipt_verifier(
    _receipt: object,
    _spec: object,
    *,
    phase: object,
) -> _FakeVerification:
    assert str(getattr(phase, "value", phase)) == "PRE_EXECUTION"
    return _FakeVerification()


def _fixture_spec(
    tmp_path: Path,
    *,
    strict_probes: tuple[StrictValidationProbe, ...] = (),
    expected_file_sha256: str | None = None,
    expected_semantic_id: str = "fixture.semantic.v1",
    expected_semantic_sha256: str | None = None,
) -> PreCampaignValidationSpec:
    config = tmp_path / "configs" / "policy_campaign.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("[meta]\nname='fixture'\n", encoding="utf-8", newline="")
    artifact = tmp_path / "inputs" / "bound.txt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("bound input\n", encoding="utf-8", newline="")
    semantic = tmp_path / "inputs" / "semantic.json"
    semantic.write_text('{"value":1}\n', encoding="utf-8", newline="")
    semantic_file_sha = sha256(semantic.read_bytes()).hexdigest()
    semantic_sha = _digest("semantic payload")

    def observe() -> SemanticArtifactObservation:
        return SemanticArtifactObservation(
            artifact_id="fixture_semantic_artifact",
            path=semantic.resolve(),
            file_sha256=semantic_file_sha,
            byte_length=len(semantic.read_bytes()),
            schema_version="fixture-v1",
            semantic_id="fixture.semantic.v1",
            semantic_sha256=semantic_sha,
            metadata={"probability_interpretation": "NONE_ILLUSTRATIVE_DESIGN_POINTS"},
        )

    return PreCampaignValidationSpec(
        repository_root=tmp_path,
        campaign_configuration_path=config,
        semantic_expectations=(
            SemanticIdentityExpectation(
                expectation_id="parameter.design",
                expected_semantic_id=expected_semantic_id,
                expected_semantic_sha256=(
                    expected_semantic_sha256 or semantic_sha
                ),
                observer=observe,
            ),
        ),
        file_expectations=(
            FileHashExpectation(
                artifact_id="input.raw_bytes",
                path=artifact,
                expected_sha256=(
                    expected_file_sha256
                    or sha256(artifact.read_bytes()).hexdigest()
                ),
                schema_version="fixture-v1",
            ),
        ),
        strict_probes=strict_probes,
        uncertainty=UncertaintyDeclaration(
            minimum_retained_seeds=100,
            fixed_seeds=tuple(range(100, 200)),
            parameter_probability_interpretation=(
                "NONE_ILLUSTRATIVE_DESIGN_POINTS"
            ),
            monetary_rate_status="UNQUANTIFIED",
            population_status="UNQUANTIFIED",
            combined_uncertainty_required=True,
        ),
        receipt_spec=object(),
        scientific_readiness_blockers=(
            "analysis_plan.registration=unregistered",
        ),
    )


def test_preflight_fails_closed_without_executing_realizations(tmp_path: Path) -> None:
    def population_gate() -> None:
        raise ValueError("population empirical validation is unavailable")

    spec = _fixture_spec(
        tmp_path,
        strict_probes=(
            StrictValidationProbe(
                probe_id="flow.policy_only_contract",
                validator=lambda: ProbeEvidence(
                    {
                        "execution_layer": "POLICY_WELFARE_V1",
                        "strategic_market_outputs_combined": False,
                    }
                ),
            ),
            StrictValidationProbe(
                probe_id="population.campaign_gate",
                validator=population_gate,
            ),
        ),
    )

    def reject_receipt(_spec: object) -> object:
        raise ExecutionAttestationError(
            "execution receipt requires a clean working tree"
        )

    report = run_pre_campaign_validation(
        spec,
        _git_observer=lambda root: _stable_git(root, clean=False),
        _receipt_builder=reject_receipt,
        _receipt_verifier=_successful_receipt_verifier,
    )
    payload = report.snapshot()

    assert report.campaign_ready is False
    assert payload["campaign_ready"] is False
    assert payload["full_campaign_intentionally_not_run"] is True
    assert payload["convergence"]["status"] == "NON_CONVERGED"
    assert payload["convergence"]["retained_seed_count"] == 0
    assert payload["primary_result"]["point_estimate"] is None
    assert payload["execution_receipt"]["status"] == "REJECTED"
    assert "clean working tree" in payload["execution_receipt"]["rejection_reason"]
    assert "population.campaign_gate" in payload["failed_checks"]
    assert "flow.policy_only_contract" in payload["passed_checks"]
    assert (
        payload["uncertainty_components"]["model_parameter"]["state"]
        == "UNQUANTIFIED"
    )
    assert (
        payload["uncertainty_components"]["monetary_rate"]["variance"]
        is None
    )
    assert (
        payload["uncertainty_components"]["population"]["variance"]
        is None
    )
    assert (
        payload["uncertainty_components"]["combined"]["state"]
        == "UNAVAILABLE"
    )
    assert "uncertainty.combined=unavailable" in payload["unresolved_blockers"]


def test_report_is_deterministic_and_write_is_evidence_preserving(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(tmp_path)
    kwargs = {
        "_git_observer": lambda root: _stable_git(root),
        "_receipt_builder": _successful_receipt_builder,
        "_receipt_verifier": _successful_receipt_verifier,
    }
    first = run_pre_campaign_validation(spec, **kwargs)
    second = run_pre_campaign_validation(spec, **kwargs)
    assert first.report_sha256 == second.report_sha256
    assert first.identity_payload_json == second.identity_payload_json
    assert "timestamp" not in first.identity_payload_json
    assert (
        first.snapshot()["execution_receipt"]["status"]
        == "GENERATED_AND_PREVERIFIED"
    )
    assert first.snapshot()["execution_receipt"]["campaign_execution_admissible"] is False

    destination = tmp_path / "artifacts" / "pre-campaign-validation-report.json"
    assert write_pre_campaign_validation_report(destination, first) == destination
    assert write_pre_campaign_validation_report(destination, second) == destination

    changed_spec = _fixture_spec(
        tmp_path,
        strict_probes=(
            StrictValidationProbe(
                probe_id="additional.failure",
                validator=lambda: (_ for _ in ()).throw(ValueError("blocked")),
            ),
        ),
    )
    changed = run_pre_campaign_validation(changed_spec, **kwargs)
    with pytest.raises(
        PreCampaignValidationError,
        match="refusing to overwrite",
    ):
        write_pre_campaign_validation_report(destination, changed)


def test_mismatched_and_placeholder_identities_are_failed_not_trusted(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec(
        tmp_path,
        expected_file_sha256="0" * 64,
        expected_semantic_id="BLOCKED_PENDING_PLAN_ID",
        expected_semantic_sha256="BLOCKED_PENDING_PLAN_SHA256",
    )
    report = run_pre_campaign_validation(
        spec,
        _git_observer=lambda root: _stable_git(root),
        _receipt_builder=_successful_receipt_builder,
        _receipt_verifier=_successful_receipt_verifier,
    ).snapshot()

    assert "input.raw_bytes" in report["failed_checks"]
    assert "parameter.design" in report["failed_checks"]
    assert "preflight.input.raw_bytes=failed" in report["unresolved_blockers"]
    assert "preflight.parameter.design=failed" in report["unresolved_blockers"]
    semantic_checks = {
        item["check_id"]: item for item in report["checks"]
    }
    assert semantic_checks["parameter.design"]["expected_sha256"] is None


def test_repository_plan_and_parameter_observers_recompute_real_identities() -> None:
    repository = Path(__file__).resolve().parents[1]
    plan = observe_prospective_analysis_plan(
        repository / "inputs" / "prospective-analysis-plan.json"
    )
    parameter = observe_parameter_uncertainty_design(
        repository / "inputs" / "parameter-uncertainty-design-v1.json"
    )

    assert plan.path.is_absolute()
    assert plan.file_sha256 == sha256(plan.path.read_bytes()).hexdigest()
    assert len(plan.semantic_sha256) == 64
    assert plan.semantic_id.startswith("illustrative.prospective")
    assert len(parameter.semantic_sha256) == 64
    assert parameter.semantic_id


def test_observers_expose_plan_and_parameter_semantics_without_promoting_them() -> None:
    repository = Path(__file__).resolve().parents[1]
    plan = observe_prospective_analysis_plan(
        repository / "inputs" / "prospective-analysis-plan.json"
    )
    parameter = observe_parameter_uncertainty_design(
        repository / "inputs" / "parameter-uncertainty-design-v1.json"
    )

    assert plan.metadata["campaign_ready"] is False
    assert plan.metadata["registration_status"] == "UNREGISTERED"
    assert len(plan.metadata["campaign_blockers"]) > 0
    assert parameter.metadata["oat_role"] == "DIAGNOSTIC_ONLY"
    assert (
        parameter.metadata["probability_interpretation"]
        == "NONE_ILLUSTRATIVE_DESIGN_POINTS"
    )
    assert parameter.file_sha256 == sha256(parameter.path.read_bytes()).hexdigest()


def test_real_campaign_execution_receipt_spec_is_complete_without_execution() -> None:
    repository = Path(__file__).resolve().parents[1]
    receipt_spec = build_policy_campaign_execution_receipt_spec(
        repository / "configs" / "policy_campaign.toml",
        repository_root=repository,
    )

    assert type(receipt_spec) is ExecutionReceiptSpec
    assert receipt_spec.repository_root == repository
    assert (
        receipt_spec.plan_id
        == "illustrative.prospective.composite-harm.baseline-vs-safe.v3"
    )
    assert (
        receipt_spec.plan_sha256
        == "1f27f290179cb054da10d97fce877ca9b17582f82f7d18e76788575afb18a023"
    )
    artifact_ids = {item.artifact_id for item in receipt_spec.input_artifacts}
    identity_ids = {item.identity_id for item in receipt_spec.input_identities}
    assert REQUIRED_FILE_ARTIFACT_IDS <= artifact_ids
    assert REQUIRED_SEMANTIC_IDENTITY_IDS <= identity_ids
    assert {
        "parent_prospective_analysis_plan",
        "monetary_conversion_profile",
        "parameter_uncertainty_design",
        "execution_receipt_schema",
    } <= artifact_ids
    assert {
        "campaign_output_profile",
        "parameter_uncertainty_design",
        "profile_input_lineage",
        "population_apportionment_plan",
        "population_execution_input",
        "population_uncertainty_design",
    } <= identity_ids

    identities = {
        item.identity_id: item for item in receipt_spec.input_identities
    }
    assert identities["prospective_plan_semantic"].sha256 == receipt_spec.plan_sha256
    assert identities["profile_input_lineage"].schema_version == "4"
    assert identities["output_schema"].schema_version == "3.0"
    assert identities["manifest_schema"].schema_version == "1.0"
    assert identities["campaign_output_profile"].schema_version == "1.0"
    unavailable = {
        identity_id
        for identity_id, identity in identities.items()
        if not identity.available
    }
    assert unavailable == {
        "monetary_rate_uncertainty_design",
        "parameter_probability_distribution",
        "population_assignment",
        "population_balance",
        "population_execution",
        "population_lineage",
        "population_uncertainty_design",
    }
    assert all(
        item.required_for_execution for item in identities.values() if not item.available
    )
    assert receipt_spec.ledger_backend == "sqlite"
    assert receipt_spec.ledger_configuration["persistent"] is True
    assert receipt_spec.ledger_configuration["temporary"] is False
    assert receipt_spec.execution_mode == "FULL_CAMPAIGN"
    assert receipt_spec.run_command[-1] == "configs/policy_campaign.toml"
    assert "pre-campaign-validation-report.json" in (
        receipt_spec.expected_output_artifacts
    )
    assert "population.empirical_validation=missing" in (
        receipt_spec.scientific_readiness_blockers
    )

    preflight_spec = build_policy_campaign_preflight_spec(
        repository / "configs" / "policy_campaign.toml",
        repository_root=repository,
        receipt_spec=receipt_spec,
    )
    assert preflight_spec.receipt_spec is receipt_spec
    assert len(preflight_spec.uncertainty.fixed_seeds) >= 100
    assert preflight_spec.uncertainty.monetary_rate_status == "UNQUANTIFIED"
    assert preflight_spec.uncertainty.population_status == "UNQUANTIFIED"
    assert any(
        probe.probe_id == "population.campaign_gate"
        for probe in preflight_spec.strict_probes
    )


@pytest.mark.parametrize(
    ("field", "expected_message"),
    (
        ("maximum_mcse", "convergence rule differs"),
        ("parameter_design_id", "parameter-design identity differs"),
    ),
)
def test_preflight_rejects_plan_config_uncertainty_binding_drift(
    field: str,
    expected_message: str,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs" / "policy_campaign.toml"
    config = load_policy_config(config_path)
    assert config.convergence is not None
    assert config.uncertainty is not None
    if field == "maximum_mcse":
        tampered = replace(
            config,
            convergence=replace(config.convergence, maximum_mcse=0.00125),
        )
    else:
        tampered = replace(
            config,
            uncertainty=replace(
                config.uncertainty,
                parameter_design_id="illustrative.policy-model-joint-parameters.drift",
            ),
        )
    with patch(
        "microtx_sim.policy_config.load_policy_config",
        return_value=tampered,
    ):
        spec = build_policy_campaign_preflight_spec(
            config_path,
            repository_root=repository,
            receipt_spec=object(),
        )
    probe = next(
        item
        for item in spec.strict_probes
        if item.probe_id == "analysis_plan.runtime_bindings"
    )
    with pytest.raises(PreCampaignValidationError, match=expected_message):
        probe.validator()


def test_real_campaign_receipt_attempt_matches_current_cleanliness() -> None:
    repository = Path(__file__).resolve().parents[1]
    receipt_spec = build_policy_campaign_execution_receipt_spec(
        repository / "configs" / "policy_campaign.toml",
        repository_root=repository,
    )
    git = observe_git_repository(repository)

    if git.working_tree_clean:
        receipt = build_execution_receipt(receipt_spec)
        assert receipt.campaign_execution_admissible is False
        assert any(
            blocker.startswith("execution_identity.population_execution=unavailable")
            for blocker in receipt.input_completeness_blockers
        )
    else:
        with pytest.raises(
            ExecutionAttestationError,
            match="clean working tree",
        ):
            build_execution_receipt(receipt_spec)


def test_report_rejects_forged_readiness_or_convergence(tmp_path: Path) -> None:
    spec = _fixture_spec(tmp_path)
    report = run_pre_campaign_validation(
        spec,
        _git_observer=lambda root: _stable_git(root),
        _receipt_builder=_successful_receipt_builder,
        _receipt_verifier=_successful_receipt_verifier,
    )
    payload = report.identity_payload
    payload["campaign_ready"] = True
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(PreCampaignValidationError, match="campaign_ready"):
        PreCampaignValidationReport(
            identity_payload_json=canonical,
            report_sha256=sha256(canonical.encode()).hexdigest(),
        )

    payload = report.identity_payload
    payload["convergence"]["status"] = "CONVERGED"
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(PreCampaignValidationError, match="NON_CONVERGED"):
        PreCampaignValidationReport(
            identity_payload_json=canonical,
            report_sha256=sha256(canonical.encode()).hexdigest(),
        )

    payload = report.identity_payload
    payload["convergence"]["completed_realization_count"] = 1
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(PreCampaignValidationError, match="no-realization"):
        PreCampaignValidationReport(
            identity_payload_json=canonical,
            report_sha256=sha256(canonical.encode()).hexdigest(),
        )

    payload = report.identity_payload
    payload["primary_result"]["point_estimate"] = 0.0
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    with pytest.raises(PreCampaignValidationError, match="point estimate"):
        PreCampaignValidationReport(
            identity_payload_json=canonical,
            report_sha256=sha256(canonical.encode()).hexdigest(),
        )


def test_git_observation_records_exact_branch_commit_and_dirty_status(
    tmp_path: Path,
) -> None:
    subprocess.run(("git", "init", "-b", "main"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "config", "user.email", "preflight@example.invalid"),
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Preflight Test"),
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(("git", "add", "tracked.txt"), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "commit", "-m", "fixture"),
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
    )
    clean = observe_git_repository(tmp_path.resolve())
    assert clean.active_branch == "main"
    assert clean.exact_commit is not None and len(clean.exact_commit) == 40
    assert clean.working_tree_clean is True

    (tmp_path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = observe_git_repository(tmp_path.resolve())
    assert dirty.working_tree_clean is False
    assert "untracked.txt" in dirty.status_porcelain
