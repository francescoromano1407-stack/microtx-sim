"""Deterministic, non-executing policy-campaign preflight reports.

This module is deliberately incapable of running a simulation.  It reopens
declared files, re-runs semantic loaders and strict gate probes, and attempts a
canonical execution receipt.  A failed probe is evidence for a blocker; it is
never converted into a warning or an inferred zero-uncertainty component.

The report is an inspection artifact, not an authorization token.  Schema 1
therefore fixes ``campaign_ready`` to false, records zero realized seeds, and
marks convergence ``NON_CONVERGED`` even when every structural check passes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
from typing import Any, Final

from .execution_attestation import (
    DeclaredIdentity,
    ExecutionReceiptSpec,
    ExecutionVerificationPhase,
    FileArtifactSpec,
    build_execution_receipt,
    verify_execution_receipt,
)


PRE_CAMPAIGN_REPORT_SCHEMA_VERSION: Final[str] = "1.0"
PRE_CAMPAIGN_REPORT_SCHEMA_ID: Final[str] = (
    "https://microtx-sim.invalid/schemas/pre-campaign-validation-report-1.0.json"
)
PRE_CAMPAIGN_REPORT_IDENTITY_ALGORITHM: Final[str] = (
    "microtx_sim.pre_campaign_report.canonical_json_utf8.v1"
)
PRE_CAMPAIGN_EXECUTION_MODE: Final[str] = (
    "PRE_CAMPAIGN_VALIDATION_ONLY_NO_SIMULATION"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_BLOCKED_IDENTITY = re.compile(r"BLOCKED_PENDING_[A-Z0-9_]+\Z")


class PreCampaignValidationError(ValueError):
    """Raised when a validation request or report is internally ambiguous."""


class PreflightCheckStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"


class ReceiptAttemptStatus(str, Enum):
    GENERATED_AND_PREVERIFIED = "GENERATED_AND_PREVERIFIED"
    REJECTED = "REJECTED"


class UncertaintyComponentState(str, Enum):
    DESIGN_DECLARED_NOT_EXECUTED = "DESIGN_DECLARED_NOT_EXECUTED"
    QUANTIFIED_DESIGN_NOT_EXECUTED = "QUANTIFIED_DESIGN_NOT_EXECUTED"
    UNQUANTIFIED = "UNQUANTIFIED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class FileHashExpectation:
    """Expected raw-byte identity for one repository-contained input."""

    artifact_id: str
    path: Path
    expected_sha256: str
    schema_version: str

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, name="artifact_id")
        if not isinstance(self.path, Path):
            raise TypeError("file expectation path must be a Path")
        _nonempty_text(self.expected_sha256, name="expected_sha256")
        _nonempty_text(self.schema_version, name="schema_version")


@dataclass(frozen=True, slots=True)
class SemanticArtifactObservation:
    """A semantic identity independently rebuilt from exact file bytes."""

    artifact_id: str
    path: Path
    file_sha256: str
    byte_length: int
    schema_version: str
    semantic_id: str
    semantic_sha256: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.artifact_id, name="semantic artifact_id")
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise TypeError("semantic observation path must be absolute")
        _digest(self.file_sha256, name="semantic file_sha256")
        _strict_int(self.byte_length, name="semantic byte_length", minimum=1)
        _nonempty_text(self.schema_version, name="semantic schema_version")
        _nonempty_text(self.semantic_id, name="semantic_id")
        _digest(self.semantic_sha256, name="semantic_sha256")
        _canonical_json_bytes(self.metadata)

    def snapshot(self, *, repository_root: Path) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "path": _repository_relative_path(repository_root, self.path),
            "path_base": "repository_root",
            "byte_length": self.byte_length,
            "file_sha256": self.file_sha256,
            "schema_version": self.schema_version,
            "semantic_id": self.semantic_id,
            "semantic_sha256": self.semantic_sha256,
            "metadata": _json_copy(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SemanticIdentityExpectation:
    """Expected semantic identity and its independent observer."""

    expectation_id: str
    expected_semantic_id: str
    expected_semantic_sha256: str
    observer: Callable[[], SemanticArtifactObservation]

    def __post_init__(self) -> None:
        _identifier(self.expectation_id, name="semantic expectation_id")
        _nonempty_text(self.expected_semantic_id, name="expected_semantic_id")
        _nonempty_text(
            self.expected_semantic_sha256,
            name="expected_semantic_sha256",
        )
        if not callable(self.observer):
            raise TypeError("semantic observer must be callable")


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    """Canonical evidence returned by one strict validation probe."""

    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise TypeError("probe evidence payload must be a mapping")
        _canonical_json_bytes(self.payload)

    @property
    def evidence_sha256(self) -> str:
        return _canonical_sha256(self.payload)


@dataclass(frozen=True, slots=True)
class StrictValidationProbe:
    """One existing strict validator invoked without suppressing its failure."""

    probe_id: str
    validator: Callable[[], ProbeEvidence | Mapping[str, object] | None]

    def __post_init__(self) -> None:
        _identifier(self.probe_id, name="probe_id")
        if not callable(self.validator):
            raise TypeError("probe validator must be callable")


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    check_id: str
    status: PreflightCheckStatus
    detail: str
    expected_sha256: str | None = None
    observed_sha256: str | None = None
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.check_id, name="check_id")
        if type(self.status) is not PreflightCheckStatus:
            raise TypeError("check status must be PreflightCheckStatus")
        _nonempty_text(self.detail, name="check detail")
        for name in ("expected_sha256", "observed_sha256", "evidence_sha256"):
            value = getattr(self, name)
            if value is not None:
                _digest(value, name=f"check {name}")

    def snapshot(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "status": self.status.value,
            "detail": self.detail,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class GitObservation:
    repository_root: Path
    active_branch: str | None
    exact_commit: str | None
    object_format: str | None
    status_porcelain: str
    working_tree_clean: bool
    observation_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path) or not self.repository_root.is_absolute():
            raise TypeError("repository_root must be absolute")
        if self.active_branch is not None:
            _nonempty_text(self.active_branch, name="active_branch")
        if self.exact_commit is not None and not _GIT_OBJECT_ID.fullmatch(
            self.exact_commit
        ):
            raise PreCampaignValidationError("exact_commit is not a Git object ID")
        if self.object_format is not None and self.object_format not in {
            "sha1",
            "sha256",
        }:
            raise PreCampaignValidationError("unsupported Git object format")
        if type(self.status_porcelain) is not str:
            raise TypeError("status_porcelain must be text")
        if type(self.working_tree_clean) is not bool:
            raise TypeError("working_tree_clean must be boolean")
        if self.working_tree_clean != (self.status_porcelain == ""):
            raise PreCampaignValidationError(
                "working_tree_clean differs from status_porcelain"
            )
        if self.observation_error is not None:
            _nonempty_text(self.observation_error, name="Git observation_error")

    def snapshot(self) -> dict[str, object]:
        return {
            "repository_root": _normalized_absolute_path(self.repository_root),
            "active_branch": self.active_branch,
            "exact_commit": self.exact_commit,
            "object_format": self.object_format,
            "status_porcelain": self.status_porcelain,
            "working_tree_clean": self.working_tree_clean,
            "observation_error": self.observation_error,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyDeclaration:
    """Declared design availability before any realization exists."""

    minimum_retained_seeds: int
    fixed_seeds: tuple[int, ...]
    parameter_probability_interpretation: str
    monetary_rate_status: str
    population_status: str
    combined_uncertainty_required: bool

    def __post_init__(self) -> None:
        _strict_int(
            self.minimum_retained_seeds,
            name="minimum_retained_seeds",
            minimum=100,
        )
        if type(self.fixed_seeds) is not tuple or any(
            type(value) is not int or isinstance(value, bool)
            for value in self.fixed_seeds
        ):
            raise TypeError("fixed_seeds must be an exact tuple of integers")
        if self.fixed_seeds != tuple(sorted(self.fixed_seeds)) or len(
            set(self.fixed_seeds)
        ) != len(self.fixed_seeds):
            raise PreCampaignValidationError(
                "fixed_seeds must be unique and strictly ascending"
            )
        _nonempty_text(
            self.parameter_probability_interpretation,
            name="parameter_probability_interpretation",
        )
        for name in ("monetary_rate_status", "population_status"):
            if getattr(self, name) not in {
                "QUANTIFIED",
                "UNQUANTIFIED",
                "UNAVAILABLE",
            }:
                raise PreCampaignValidationError(
                    f"{name} must be QUANTIFIED, UNQUANTIFIED, or UNAVAILABLE"
                )
        if type(self.combined_uncertainty_required) is not bool:
            raise TypeError("combined_uncertainty_required must be boolean")


@dataclass(frozen=True, slots=True)
class PreCampaignValidationSpec:
    """All inputs needed for deterministic validation without model execution."""

    repository_root: Path
    campaign_configuration_path: Path
    semantic_expectations: tuple[SemanticIdentityExpectation, ...]
    file_expectations: tuple[FileHashExpectation, ...]
    strict_probes: tuple[StrictValidationProbe, ...]
    uncertainty: UncertaintyDeclaration
    receipt_spec: object
    scientific_readiness_blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.repository_root, Path):
            raise TypeError("repository_root must be a Path")
        root = self.repository_root.resolve(strict=True)
        if not root.is_dir():
            raise PreCampaignValidationError("repository_root must be a directory")
        object.__setattr__(self, "repository_root", root)
        if not isinstance(self.campaign_configuration_path, Path):
            raise TypeError("campaign_configuration_path must be a Path")
        for name, values, expected_type in (
            (
                "semantic_expectations",
                self.semantic_expectations,
                SemanticIdentityExpectation,
            ),
            ("file_expectations", self.file_expectations, FileHashExpectation),
            ("strict_probes", self.strict_probes, StrictValidationProbe),
        ):
            if type(values) is not tuple or any(
                type(item) is not expected_type for item in values
            ):
                raise TypeError(f"{name} must be an exact tuple of {expected_type.__name__}")
        ids = [item.expectation_id for item in self.semantic_expectations]
        ids.extend(item.artifact_id for item in self.file_expectations)
        ids.extend(item.probe_id for item in self.strict_probes)
        if len(ids) != len(set(ids)):
            raise PreCampaignValidationError("preflight check IDs must be unique")
        if type(self.uncertainty) is not UncertaintyDeclaration:
            raise TypeError("uncertainty must be UncertaintyDeclaration")
        object.__setattr__(
            self,
            "scientific_readiness_blockers",
            _canonical_blockers(self.scientific_readiness_blockers),
        )


@dataclass(frozen=True, slots=True)
class PreCampaignValidationReport:
    """Canonical report whose fixed fields cannot authorize a campaign."""

    identity_payload_json: str
    report_sha256: str

    def __post_init__(self) -> None:
        if type(self.identity_payload_json) is not str:
            raise TypeError("identity_payload_json must be text")
        try:
            payload = json.loads(self.identity_payload_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise PreCampaignValidationError("report payload is not valid JSON") from exc
        if type(payload) is not dict:
            raise PreCampaignValidationError("report payload must be an object")
        canonical = _canonical_json_bytes(payload).decode("utf-8")
        if canonical != self.identity_payload_json:
            raise PreCampaignValidationError(
                "report identity_payload_json must be canonical"
            )
        _validate_report_payload(payload)
        _digest(self.report_sha256, name="report_sha256")
        if _canonical_sha256(payload) != self.report_sha256:
            raise PreCampaignValidationError(
                "report_sha256 does not match identity payload"
            )

    @property
    def identity_payload(self) -> dict[str, object]:
        return json.loads(self.identity_payload_json)

    @property
    def campaign_ready(self) -> bool:
        return False

    def snapshot(self) -> dict[str, object]:
        return {
            **self.identity_payload,
            "report_sha256": self.report_sha256,
        }


def observe_prospective_analysis_plan(path: str | Path) -> SemanticArtifactObservation:
    """Load and re-attest a prospective plan from its exact bytes."""

    from .causal.analysis_plan import (
        load_prospective_analysis_plan,
        verify_loaded_prospective_analysis_plan,
    )

    loaded = verify_loaded_prospective_analysis_plan(
        load_prospective_analysis_plan(path)
    )
    plan = loaded.plan
    return SemanticArtifactObservation(
        artifact_id="prospective_analysis_plan",
        path=loaded.plan_path,
        file_sha256=loaded.file_sha256,
        byte_length=loaded.byte_length,
        schema_version=plan.schema_version,
        semantic_id=plan.plan_id,
        semantic_sha256=plan.plan_sha256,
        metadata={
            "registration_status": plan.registration_status.value,
            "preregistered": plan.preregistered,
            "campaign_ready": False,
            "campaign_blockers": list(plan.campaign_blockers),
            "primary_estimand_id": plan.primary_estimand.estimand_id,
            "primary_estimand_specification_sha256": (
                plan.primary_estimand.specification_sha256
            ),
            "fixed_seeds": list(plan.stopping_rule.seeds),
        },
    )


def observe_parameter_uncertainty_design(
    path: str | Path,
) -> SemanticArtifactObservation:
    """Load and re-attest a deterministic joint-parameter design."""

    from .analysis.uncertainty import (
        load_parameter_uncertainty_design,
        verify_loaded_parameter_uncertainty_design,
    )

    loaded = verify_loaded_parameter_uncertainty_design(
        load_parameter_uncertainty_design(path)
    )
    design = loaded.design
    return SemanticArtifactObservation(
        artifact_id="parameter_uncertainty_design",
        path=loaded.design_path,
        file_sha256=loaded.file_sha256,
        byte_length=loaded.byte_length,
        schema_version="1.0",
        semantic_id=design.design_id,
        semantic_sha256=design.design_sha256,
        metadata={
            "method": design.method,
            "design_seed": design.design_seed,
            "draw_count": design.draw_count,
            "parameter_ids": [item.parameter_id for item in design.parameters],
            "probability_interpretation": (
                "CALIBRATED_JOINT_DISTRIBUTION"
                if design.calibrated_probability_design
                else "NONE_ILLUSTRATIVE_DESIGN_POINTS"
            ),
            "oat_role": "DIAGNOSTIC_ONLY",
        },
    )


def policy_flow_contract_probe(plan_path: str | Path) -> ProbeEvidence:
    """Build the policy-only cross-layer contract without running a batch."""

    from .causal.analysis_plan import load_prospective_analysis_plan
    from .causal.flow_contract import (
        PolicyExecutionLayer,
        build_policy_flow_contract,
    )

    loaded = load_prospective_analysis_plan(plan_path)
    contract = build_policy_flow_contract(
        loaded.plan,
        execution_layer=PolicyExecutionLayer.POLICY_WELFARE,
    )
    return ProbeEvidence(
        {
            "schema_version": contract.schema_version,
            "contract_id": contract.contract_id,
            "contract_sha256": contract.contract_sha256,
            "execution_layer": contract.execution_layer.value,
            "strategic_market_outputs_combined": False,
            "campaign_ready": False,
        }
    )


def rate_evidence_integrity_probe(
    bundle_path: str | Path,
    *,
    required_source_registry_sha256: str | None = None,
) -> ProbeEvidence:
    """Reopen an exact point-rate bundle without claiming rate uncertainty."""

    from .data.rate_evidence import load_and_verify_rate_evidence_bundle

    bundle, results = load_and_verify_rate_evidence_bundle(
        bundle_path,
        required_source_registry_sha256=required_source_registry_sha256,
    )
    evidence_set = {
        "schema_version": "1.0",
        "bundle_id": bundle.bundle_id,
        "bundle_sha256": bundle.bundle_sha256,
        "binding_evidence_sha256": [item.evidence_sha256 for item in results],
        "point_rate_observations_only": True,
        "rate_uncertainty_quantified": False,
    }
    return ProbeEvidence(
        {
            **evidence_set,
            "evidence_set_sha256": _canonical_sha256(evidence_set),
            "source_bundle_signature_status": bundle.signature.status.value,
            "campaign_ready": False,
        }
    )


def rate_evidence_campaign_gate_probe(bundle_path: str | Path) -> ProbeEvidence:
    """Invoke the existing monetary campaign gate; schema-v1 currently fails."""

    from .data.rate_evidence import load_rate_evidence_bundle

    bundle = load_rate_evidence_bundle(bundle_path)
    bundle.validate_for_campaign()
    return ProbeEvidence({"campaign_ready": False})  # pragma: no cover


def build_policy_campaign_execution_receipt_spec(
    config_path: str | Path,
    *,
    repository_root: str | Path,
) -> ExecutionReceiptSpec:
    """Build a complete technical receipt declaration without simulation.

    Per-seed identities that do not exist before execution are represented as
    required-but-unavailable.  A clean-tree receipt can therefore attest all
    currently available inputs while its execution gate remains closed.
    """

    from . import __version__
    from .analysis.uncertainty import (
        canonical_sha256,
        load_parameter_uncertainty_design,
    )
    from .causal.analysis_plan import (
        load_prospective_analysis_plan,
        verify_loaded_prospective_analysis_plan,
    )
    from .data.lineage import build_profile_input_lineage
    from .data.population_execution import (
        population_execution_input_sha256,
        resolve_population_projection_adapter,
    )
    from .data.profiles import load_profile_bundle
    from .data.rate_evidence import load_and_verify_rate_evidence_bundle
    from .outputs.metric_contracts import metric_contract_registry_sha256
    from .outputs.schema import (
        CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
        MANIFEST_SCHEMA_SHA256,
        MANIFEST_SCHEMA_VERSION,
    )
    from .policy_config import load_policy_config

    root = Path(repository_root).resolve(strict=True)
    selected_config = Path(config_path)
    selected_config = (
        selected_config if selected_config.is_absolute() else root / selected_config
    ).resolve(strict=True)
    config = load_policy_config(selected_config)
    required = {
        "analysis_plan": config.analysis_plan,
        "campaign": config.campaign,
        "uncertainty": config.uncertainty,
        "population": config.population,
        "population_contract": config.population_contract,
        "monetary_contract": config.monetary_contract,
        "output_contract": config.output_contract,
        "ledger": config.ledger,
        "execution_receipt": config.execution_receipt,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing or not config.full_campaign_config:
        raise PreCampaignValidationError(
            "execution receipt requires the complete full campaign config: "
            + ", ".join(missing or ["meta.full_campaign_config"])
        )
    analysis = config.analysis_plan
    campaign = config.campaign
    uncertainty = config.uncertainty
    convergence = config.convergence
    population = config.population
    population_contract = config.population_contract
    monetary = config.monetary_contract
    output = config.output_contract
    ledger = config.ledger
    receipt_policy = config.execution_receipt
    assert analysis is not None
    assert campaign is not None
    assert uncertainty is not None
    assert convergence is not None
    assert population is not None
    assert population_contract is not None
    assert monetary is not None
    assert output is not None
    assert ledger is not None
    assert receipt_policy is not None
    if (
        analysis.expected_plan_id is None
        or analysis.expected_plan_sha256 is None
        or analysis.parent_plan_path is None
        or analysis.parent_plan_id is None
        or analysis.parent_plan_sha256 is None
    ):
        raise PreCampaignValidationError(
            "execution receipt requires complete plan and parent identities"
        )
    if population.source_registry_path is None or population.evidence_bundle_path is None:
        raise PreCampaignValidationError(
            "receipt requires explicit population source and evidence paths"
        )

    plan = verify_loaded_prospective_analysis_plan(
        load_prospective_analysis_plan(analysis.plan_path)
    )
    parent = verify_loaded_prospective_analysis_plan(
        load_prospective_analysis_plan(analysis.parent_plan_path)
    )
    if (
        plan.plan.plan_id != analysis.expected_plan_id
        or plan.plan.plan_sha256 != analysis.expected_plan_sha256
        or parent.plan.plan_id != analysis.parent_plan_id
        or parent.plan.plan_sha256 != analysis.parent_plan_sha256
    ):
        raise PreCampaignValidationError(
            "receipt plan declarations differ from re-attested artifacts"
        )
    profiles = load_profile_bundle(
        jurisdictions_path=monetary.profile_path,
        sources_path=population.source_registry_path,
        source_bundle_path=monetary.source_bundle_path,
        population_bundle_path=population.evidence_bundle_path,
        campaign=False,
    )
    profile_lineage = build_profile_input_lineage(
        profiles.country_profiles,
        profile_bundle=profiles,
    )
    adapter = resolve_population_projection_adapter(
        population,
        profiles,
        player_count=config.batch.player_count,
        campaign=False,
    )
    if (
        adapter.adapter_sha256 != population_contract.adapter_sha256
        or population_execution_input_sha256(adapter)
        != population_contract.execution_input_sha256
    ):
        raise PreCampaignValidationError(
            "receipt projected-population identities differ from configuration"
        )
    parameter_design = load_parameter_uncertainty_design(
        uncertainty.parameter_design_path
    )
    rate_bundle, rate_results = load_and_verify_rate_evidence_bundle(
        monetary.source_bundle_path,
        required_source_registry_sha256=profiles.source_registry_sha256,
    )
    rate_evidence_sha256 = canonical_sha256(
        [result.snapshot() for result in rate_results]
    )
    if rate_evidence_sha256 != monetary.rate_evidence_sha256:
        raise PreCampaignValidationError(
            "receipt rate-evidence identity differs from campaign contract"
        )
    if metric_contract_registry_sha256() != output.metric_registry_sha256:
        raise PreCampaignValidationError(
            "receipt metric-contract identity differs from campaign contract"
        )
    if (
        output.output_profile_sha256 != CAMPAIGN_ANALYSIS_SCHEMA_SHA256
        or output.output_schema_sha256 != MANIFEST_SCHEMA_SHA256
    ):
        raise PreCampaignValidationError(
            "receipt output identities differ from campaign contract"
        )

    population_bundle = profiles.population_evidence_bundle
    if population_bundle is None or len(population_bundle.artifacts) != 1:
        raise PreCampaignValidationError(
            "receipt requires exactly one population source artifact"
        )
    population_artifact = population_bundle.artifacts[0]
    population_artifact_path = population_bundle.bundle_path.parent.joinpath(
        *PurePosixPath(population_bundle.artifact_root).parts,
        *PurePosixPath(population_artifact.relative_path).parts,
    ).resolve(strict=True)

    def artifact(
        artifact_id: str,
        path: Path,
        schema_version: str,
        *,
        expected_sha256: str | None = None,
        expected_byte_length: int | None = None,
    ) -> FileArtifactSpec:
        observed = path.read_bytes()
        return FileArtifactSpec(
            artifact_id=artifact_id,
            path=path,
            expected_sha256=expected_sha256 or sha256(observed).hexdigest(),
            schema_version=schema_version,
            expected_byte_length=(
                len(observed)
                if expected_byte_length is None
                else expected_byte_length
            ),
        )

    jurisdictions_sha256 = profiles.jurisdictions_sha256
    if jurisdictions_sha256 is None:
        raise PreCampaignValidationError(
            "receipt requires the monetary profile file identity"
        )
    artifacts = (
        artifact("campaign_configuration", selected_config, "full-policy-campaign-v1"),
        artifact(
            "prospective_analysis_plan",
            plan.plan_path,
            plan.plan.schema_version,
            expected_sha256=plan.file_sha256,
            expected_byte_length=plan.byte_length,
        ),
        artifact(
            "parent_prospective_analysis_plan",
            parent.plan_path,
            parent.plan.schema_version,
            expected_sha256=parent.file_sha256,
            expected_byte_length=parent.byte_length,
        ),
        artifact(
            "population_source_registry",
            population.source_registry_path,
            "1",
            expected_sha256=population_bundle.source_registry_sha256,
        ),
        artifact(
            "population_evidence",
            population_bundle.bundle_path,
            str(population_bundle.schema_version),
            expected_sha256=population_bundle.bundle_sha256,
            expected_byte_length=population_bundle.bundle_byte_length,
        ),
        artifact(
            "population_source_artifact",
            population_artifact_path,
            "1",
            expected_sha256=population_artifact.sha256,
            expected_byte_length=population_artifact.byte_length,
        ),
        artifact(
            "population_design",
            population.design_bundle_path,
            population_contract.design_schema_version,
            expected_sha256=population_contract.design_sha256,
        ),
        artifact(
            "population_runtime_mapping",
            population.runtime_mapping_bundle_path,
            population_contract.runtime_mapping_schema_version,
            expected_sha256=population_contract.runtime_mapping_sha256,
        ),
        artifact(
            "monetary_source_bundle",
            monetary.source_bundle_path,
            str(rate_bundle.schema_version),
            expected_sha256=monetary.source_bundle_sha256,
            expected_byte_length=rate_bundle.bundle_byte_length,
        ),
        artifact(
            "monetary_source_artifact",
            monetary.source_artifact_path,
            "ECB_EXR_CSV_2024",
            expected_sha256=monetary.source_artifact_sha256,
        ),
        artifact(
            "monetary_conversion_table",
            monetary.conversion_table_path,
            "1",
            expected_sha256=monetary.conversion_table_sha256,
        ),
        artifact(
            "monetary_conversion_profile",
            monetary.profile_path,
            str(profiles.jurisdiction_schema_version),
            expected_sha256=jurisdictions_sha256,
        ),
        artifact(
            "parameter_uncertainty_design",
            parameter_design.design_path,
            "1.0",
            expected_sha256=parameter_design.file_sha256,
            expected_byte_length=parameter_design.byte_length,
        ),
        artifact(
            "execution_receipt_schema",
            receipt_policy.schema_path,
            receipt_policy.schema_version,
        ),
    )
    identities = (
        DeclaredIdentity(
            identity_id="prospective_plan_semantic",
            schema_version=plan.plan.schema_version,
            sha256=plan.plan.plan_sha256,
        ),
        DeclaredIdentity(
            identity_id="parent_plan_semantic",
            schema_version=parent.plan.schema_version,
            sha256=parent.plan.plan_sha256,
        ),
        DeclaredIdentity(
            identity_id="population_adapter",
            schema_version=population_contract.adapter_schema_version,
            sha256=adapter.adapter_sha256,
        ),
        DeclaredIdentity(
            identity_id="population_design_target",
            schema_version=population_contract.design_schema_version,
            sha256=adapter.calibration_target_sha256,
        ),
        DeclaredIdentity(
            identity_id="population_apportionment_plan",
            schema_version="1",
            sha256=adapter.apportionment_sha256,
        ),
        DeclaredIdentity(
            identity_id="population_execution_input",
            schema_version=population_contract.execution_input_schema_version,
            sha256=population_execution_input_sha256(adapter),
        ),
        DeclaredIdentity(
            identity_id="population_execution",
            schema_version=population_contract.execution_input_schema_version,
            unavailable_reason=(
                "no seed realization was executed during pre-campaign validation"
            ),
        ),
        DeclaredIdentity(
            identity_id="population_assignment",
            schema_version=population_contract.assignment_schema_version,
            unavailable_reason=(
                "per-seed projected-population assignment does not exist before execution"
            ),
        ),
        DeclaredIdentity(
            identity_id="population_balance",
            schema_version=population_contract.balance_schema_version,
            unavailable_reason=(
                "per-seed realized balance does not exist before execution"
            ),
        ),
        DeclaredIdentity(
            identity_id="population_lineage",
            schema_version=population_contract.lineage_schema_version,
            unavailable_reason=(
                "complete fixed-seed population lineage requires retained executions"
            ),
        ),
        DeclaredIdentity(
            identity_id="population_uncertainty_design",
            schema_version="1.0",
            unavailable_reason=(
                "no admissible alternative-population, resampling, or design-based "
                "uncertainty method is declared"
            ),
        ),
        DeclaredIdentity(
            identity_id="monetary_conversion_basis",
            schema_version="1.0",
            sha256=monetary.conversion_basis_sha256,
        ),
        DeclaredIdentity(
            identity_id="monetary_rate_evidence",
            schema_version="1.0",
            sha256=rate_evidence_sha256,
        ),
        DeclaredIdentity(
            identity_id="monetary_rate_uncertainty_design",
            schema_version="1.0",
            unavailable_reason=(
                "official point-rate observations do not define a rate distribution"
            ),
        ),
        DeclaredIdentity(
            identity_id="parameter_uncertainty_design",
            schema_version="1.0",
            sha256=parameter_design.design.design_sha256,
        ),
        DeclaredIdentity(
            identity_id="parameter_probability_distribution",
            schema_version="1.0",
            unavailable_reason=(
                "declared ranges are illustrative design points, not calibrated "
                "probability distributions"
            ),
        ),
        DeclaredIdentity(
            identity_id="profile_input_lineage",
            schema_version=str(profile_lineage.snapshot["schema_version"]),
            sha256=profile_lineage.fingerprint_sha256,
        ),
        DeclaredIdentity(
            identity_id="metric_contract_registry",
            schema_version=output.metric_registry_schema_version,
            sha256=metric_contract_registry_sha256(),
        ),
        DeclaredIdentity(
            identity_id="output_schema",
            schema_version=output.output_schema_version,
            sha256=output.output_schema_sha256,
        ),
        DeclaredIdentity(
            identity_id="manifest_schema",
            schema_version=MANIFEST_SCHEMA_VERSION,
            sha256=MANIFEST_SCHEMA_SHA256,
        ),
        DeclaredIdentity(
            identity_id="campaign_output_profile",
            schema_version=output.output_profile_schema_version,
            sha256=CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
        ),
    )
    scientific_blockers = set(plan.plan.campaign_blockers)
    scientific_blockers.update(
        {
            "configuration.provenance_status=" + config.provenance_status,
            "monetary.rate_uncertainty=unquantified",
            "monetary.simulation_bridge=illustrative",
            "monetary.source_bundle_signature=missing",
            "parameter.probability_distributions=uncalibrated",
            "population.empirical_validation=missing",
            "population.uncertainty=unquantified",
        }
    )
    ledger_path = ledger.path.resolve()
    if not ledger_path.is_relative_to(root):
        raise PreCampaignValidationError(
            "campaign ledger must remain within the repository"
        )
    return ExecutionReceiptSpec(
        repository_root=root,
        input_artifacts=artifacts,
        input_identities=identities,
        plan_id=plan.plan.plan_id,
        plan_sha256=plan.plan.plan_sha256,
        expected_output_artifacts=output.expected_artifacts,
        ledger_backend=ledger.backend.value,
        ledger_configuration={
            "path": ledger_path.relative_to(root).as_posix(),
            "path_base": "repository_root",
            "persistent": ledger.persistent,
            "temporary": ledger.temporary,
        },
        run_command=receipt_policy.run_command,
        execution_mode=receipt_policy.execution_mode,
        model_version=f"microtx-sim/{__version__}",
        scientific_readiness_blockers=tuple(sorted(scientific_blockers)),
    )


def build_policy_campaign_preflight_spec(
    config_path: str | Path,
    *,
    repository_root: str | Path,
    receipt_spec: object,
) -> PreCampaignValidationSpec:
    """Resolve the strict full-campaign TOML into a non-executing preflight.

    The population adapter built by the returned probes is a static,
    content-addressed apportionment/mapping object.  No player cohort, scenario,
    policy day, or market cycle is initialized.  Both the population and rate
    campaign validators are still invoked and their current failures remain
    report blockers.
    """

    from .policy_config import load_policy_config

    root = Path(repository_root).resolve(strict=True)
    selected_config = Path(config_path)
    selected_config = (
        selected_config
        if selected_config.is_absolute()
        else root / selected_config
    ).resolve(strict=True)
    config = load_policy_config(selected_config)
    required = {
        "analysis_plan": config.analysis_plan,
        "campaign": config.campaign,
        "uncertainty": config.uncertainty,
        "convergence": config.convergence,
        "population": config.population,
        "population_contract": config.population_contract,
        "monetary_contract": config.monetary_contract,
        "output_contract": config.output_contract,
        "ledger": config.ledger,
        "execution_receipt": config.execution_receipt,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing or not config.full_campaign_config:
        raise PreCampaignValidationError(
            "full campaign configuration is incomplete: "
            + ", ".join(missing or ["meta.full_campaign_config"])
        )
    analysis = config.analysis_plan
    campaign = config.campaign
    uncertainty = config.uncertainty
    convergence = config.convergence
    population = config.population
    population_contract = config.population_contract
    monetary = config.monetary_contract
    output = config.output_contract
    assert analysis is not None
    assert campaign is not None
    assert uncertainty is not None
    assert convergence is not None
    assert population is not None
    assert population_contract is not None
    assert monetary is not None
    assert output is not None
    if (
        analysis.expected_plan_id is None
        or analysis.expected_plan_sha256 is None
        or analysis.parent_plan_path is None
        or analysis.parent_plan_id is None
        or analysis.parent_plan_sha256 is None
    ):
        raise PreCampaignValidationError(
            "campaign analysis-plan expected and parent identities are incomplete"
        )

    cache: dict[str, object] = {}

    def resolved_context() -> Mapping[str, object]:
        existing = cache.get("context")
        if isinstance(existing, Mapping):
            return existing
        from .causal.batch import resolve_policy_run_inputs
        from .data.lineage import build_profile_input_lineage
        from .data.population_execution import resolve_population_projection_adapter
        from .data.profiles import load_profile_bundle

        if population.source_registry_path is None or (
            population.evidence_bundle_path is None
        ):
            raise PreCampaignValidationError(
                "projected population requires explicit evidence and source-registry paths"
            )
        profiles = load_profile_bundle(
            jurisdictions_path=monetary.profile_path,
            sources_path=population.source_registry_path,
            source_bundle_path=monetary.source_bundle_path,
            population_bundle_path=population.evidence_bundle_path,
            campaign=False,
        )
        lineage = build_profile_input_lineage(
            profiles.country_profiles,
            profile_bundle=profiles,
        )
        adapter = resolve_population_projection_adapter(
            population,
            profiles,
            player_count=config.batch.player_count,
            campaign=False,
        )
        run_inputs = resolve_policy_run_inputs(
            harm_parameters=config.harm_parameters,
            harm_weights=config.harm_weights,
            opportunity_valuation=config.opportunity_valuation,
            producer_assumptions=config.producer_assumptions,
            epgc_policy=config.epgc_policy,
        )
        result: Mapping[str, object] = {
            "profiles": profiles,
            "lineage": lineage,
            "adapter": adapter,
            "run_inputs": run_inputs,
        }
        cache["context"] = result
        return result

    def configuration_probe() -> ProbeEvidence:
        if "BLOCKED_PENDING_" in repr(config):
            raise PreCampaignValidationError(
                "full campaign configuration contains unresolved BLOCKED_PENDING identities"
            )
        if campaign.campaign_ready:
            raise PreCampaignValidationError(
                "pre-campaign configuration cannot declare campaign_ready=true"
            )
        return ProbeEvidence(
            {
                "name": config.name,
                "run_purpose": config.run_purpose.value,
                "full_campaign_config": config.full_campaign_config,
                "allow_synthetic": campaign.allow_synthetic,
                "fail_closed": campaign.fail_closed,
                "simulation_layer": campaign.simulation_layer.value,
                "campaign_ready": False,
                "seed_count": len(config.batch.seeds),
                "player_count": config.batch.player_count,
                "days": config.batch.days,
            }
        )

    def analysis_binding_probe() -> ProbeEvidence:
        from .analysis.uncertainty import load_parameter_uncertainty_design
        from .causal.analysis_binding import validate_analysis_plan_inputs
        from .causal.analysis_plan import (
            load_prospective_analysis_plan,
            verify_loaded_prospective_analysis_plan,
        )

        loaded = verify_loaded_prospective_analysis_plan(
            load_prospective_analysis_plan(analysis.plan_path)
        )
        parent = verify_loaded_prospective_analysis_plan(
            load_prospective_analysis_plan(analysis.parent_plan_path)
        )
        amendment = loaded.plan.amendment
        if amendment is None:
            raise PreCampaignValidationError(
                "campaign plan is not an explicit schema-v3 amendment"
            )
        parent_binding = amendment.get("parent_plan")
        if not isinstance(parent_binding, Mapping):
            raise PreCampaignValidationError("campaign amendment lacks parent binding")
        scientific_change = amendment.get("scientific_change")
        if not isinstance(scientific_change, Mapping):
            raise PreCampaignValidationError(
                "campaign amendment lacks a scientific-change binding"
            )
        if (
            parent.plan.plan_id != analysis.parent_plan_id
            or parent.plan.plan_sha256 != analysis.parent_plan_sha256
            or parent_binding.get("plan_id") != parent.plan.plan_id
            or parent_binding.get("plan_sha256") != parent.plan.plan_sha256
            or parent_binding.get("file_sha256") != parent.file_sha256
        ):
            raise PreCampaignValidationError(
                "amended-plan parent identity differs from the exact parent artifact"
            )
        parent_primary = parent.plan.primary_estimand
        successor_primary = loaded.plan.primary_estimand
        if (
            scientific_change.get("original_estimand_id")
            != parent_primary.estimand_id
            or scientific_change.get("original_specification_sha256")
            != parent_primary.specification_sha256
            or scientific_change.get("current_estimand_id")
            != successor_primary.estimand_id
            or scientific_change.get("current_specification_sha256")
            != successor_primary.specification_sha256
            or successor_primary.estimand_id != parent_primary.estimand_id
            or successor_primary.specification_sha256
            != parent_primary.specification_sha256
            or loaded.plan.expected_harm_weights_sha256
            != parent.plan.expected_harm_weights_sha256
            or loaded.plan.declared_harm_weights
            != parent.plan.declared_harm_weights
        ):
            raise PreCampaignValidationError(
                "successor primary estimand or declared harm weights differ "
                "from the exact parent plan"
            )
        if loaded.plan.primary_estimand.estimand_id != campaign.primary_estimand_id:
            raise PreCampaignValidationError(
                "campaign primary estimand differs from the amended plan"
            )
        if loaded.plan.stopping_rule.seeds != config.batch.seeds:
            raise PreCampaignValidationError(
                "campaign fixed seeds differ from the amended stopping rule"
            )
        declared_uncertainty = amendment.get("uncertainty_design")
        declared_convergence = amendment.get("convergence_rule")
        if not isinstance(declared_uncertainty, Mapping) or not isinstance(
            declared_convergence,
            Mapping,
        ):
            raise PreCampaignValidationError(
                "campaign amendment lacks uncertainty or convergence declarations"
            )
        parameter_design = load_parameter_uncertainty_design(
            uncertainty.parameter_design_path
        ).design
        if (
            parameter_design.design_id != uncertainty.parameter_design_id
            or parameter_design.design_sha256
            != uncertainty.parameter_design_sha256
        ):
            raise PreCampaignValidationError(
                "campaign parameter-design identity differs from its exact artifact"
            )
        if (
            parameter_design.calibrated_probability_design
            or uncertainty.parameter_uncertainty.value != "UNQUANTIFIED"
        ):
            raise PreCampaignValidationError(
                "the current illustrative parameter ranges must remain "
                "non-probabilistic and unquantified"
            )
        expected_uncertainty = {
            "schema_version": "1.0",
            "seed_uncertainty": {
                "status": "QUANTIFIED_WHEN_COMPLETE",
                "fixed_seed_count": len(config.batch.seeds),
                "population_weights_applied_within_seed": (
                    uncertainty.population_weights_within_seed
                ),
                "common_random_numbers": uncertainty.common_random_numbers,
                "identical_pretreatment_cohorts": (
                    uncertainty.identical_pretreatment_cohorts
                ),
                "outcome_dependent_seed_exclusion_allowed": (
                    uncertainty.outcome_dependent_seed_exclusion
                ),
            },
            "parameter_uncertainty": {
                "status": "ILLUSTRATIVE_DESIGN_ONLY",
                "design_id": uncertainty.parameter_design_id,
                "design_sha256": uncertainty.parameter_design_sha256,
                "method": parameter_design.method,
                "probability_interpretation": "NONE",
            },
            "monetary_rate_uncertainty": {
                "status": uncertainty.monetary_rate_uncertainty.value,
                "rate_basis_sha256": monetary.conversion_basis_sha256,
                "point_observation_is_distribution": False,
            },
            "population_uncertainty": {
                "status": uncertainty.population_uncertainty.value,
                "uncertainty_design_id": (
                    population_contract.uncertainty_design_id
                ),
                "exact_weighting_is_empirical_validation": False,
            },
            "combined_uncertainty": {
                "status": "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST",
                "double_counting_control": (
                    "one complete seed-parameter-population-rate Cartesian identity"
                ),
                "variance_decomposition_method": (
                    uncertainty.variance_decomposition_method
                ),
            },
            "oat_role": uncertainty.oat_role,
        }
        if dict(declared_uncertainty) != expected_uncertainty:
            raise PreCampaignValidationError(
                "amended-plan uncertainty declarations differ from the campaign "
                "configuration or parameter design"
            )
        expected_convergence = {
            "schema_version": "1.0",
            "block_size": convergence.block_size,
            "minimum_retained_seeds": convergence.minimum_retained_seeds,
            "maximum_mcse": convergence.maximum_mcse,
            "maximum_interval_width": convergence.maximum_interval_width,
            "maximum_absolute_change": convergence.maximum_absolute_change,
            "maximum_relative_change": convergence.maximum_relative_change,
            "maximum_invalid_rate": convergence.maximum_invalid_rate,
            "consecutive_passing_checkpoints": (
                convergence.consecutive_passing_checkpoints
            ),
            "sensitivity_instability_allowed": (
                convergence.sensitivity_instability_allowed
            ),
            "outcome_dependent_seed_exclusion_allowed": (
                uncertainty.outcome_dependent_seed_exclusion
            ),
            "required_uncertainty_component_handling": "FAIL_CLOSED",
        }
        if dict(declared_convergence) != expected_convergence:
            raise PreCampaignValidationError(
                "amended-plan convergence rule differs from the campaign "
                "configuration"
            )
        context = resolved_context()
        validate_analysis_plan_inputs(
            loaded.plan,
            batch_spec=config.batch,
            run_inputs=context["run_inputs"],
            population_adapter=context["adapter"],
            profile_input_lineage=context["lineage"],
        )
        return ProbeEvidence(
            {
                "plan_id": loaded.plan.plan_id,
                "plan_sha256": loaded.plan.plan_sha256,
                "parent_plan_id": parent.plan.plan_id,
                "parent_plan_sha256": parent.plan.plan_sha256,
                "primary_estimand_id": loaded.plan.primary_estimand.estimand_id,
                "primary_estimand_specification_sha256": (
                    loaded.plan.primary_estimand.specification_sha256
                ),
                "fixed_seed_count": len(loaded.plan.stopping_rule.seeds),
                "uncertainty_declarations_match": True,
                "convergence_rule_matches": True,
                "all_runtime_bindings_match": True,
                "campaign_ready": False,
            }
        )

    def population_integrity_probe() -> ProbeEvidence:
        from .data.population_execution import population_execution_input_sha256
        from .data.population_projection import verify_population_projection_adapter

        adapter = verify_population_projection_adapter(resolved_context()["adapter"])
        observed = {
            "design_id": adapter.apportionment_plan.design_id,
            "design_sha256": adapter.apportionment_plan.design_bundle_sha256,
            "runtime_mapping_id": adapter.mapping_id,
            "runtime_mapping_sha256": adapter.mapping_sha256,
            "adapter_id": adapter.adapter_id,
            "adapter_sha256": adapter.adapter_sha256,
            "execution_input_sha256": population_execution_input_sha256(adapter),
        }
        expected = {
            "design_id": population_contract.design_id,
            "design_sha256": population_contract.design_sha256,
            "runtime_mapping_id": population_contract.runtime_mapping_id,
            "runtime_mapping_sha256": population_contract.runtime_mapping_sha256,
            "adapter_id": population_contract.adapter_id,
            "adapter_sha256": population_contract.adapter_sha256,
            "execution_input_sha256": population_contract.execution_input_sha256,
        }
        mismatches = sorted(
            name for name, value in expected.items() if observed[name] != value
        )
        if mismatches:
            raise PreCampaignValidationError(
                "projected-population identities mismatch: " + ", ".join(mismatches)
            )
        return ProbeEvidence(
            {
                **observed,
                "apportionment_sha256": adapter.apportionment_sha256,
                "cell_count": len(adapter.cells),
                "player_count": adapter.apportionment_plan.player_count,
                "authenticity_verified": adapter.authenticity_verified,
                "campaign_ready": False,
            }
        )

    def population_campaign_gate() -> ProbeEvidence:
        from .data.population_execution import validate_population_campaign_preflight

        adapter = validate_population_campaign_preflight(
            resolved_context()["adapter"]
        )
        return ProbeEvidence(
            {"adapter_sha256": adapter.adapter_sha256, "campaign_ready": False}
        )  # pragma: no cover

    def output_contract_probe() -> ProbeEvidence:
        from .outputs.metric_contracts import metric_contract_registry_sha256
        from .outputs.schema import (
            CAMPAIGN_ANALYSIS_ARTIFACT_FILENAMES,
            CAMPAIGN_ANALYSIS_OUTPUT_PROFILE,
            CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
            CAMPAIGN_ANALYSIS_SCHEMA_VERSION,
            MANIFEST_SCHEMA_SHA256,
            MANIFEST_SCHEMA_VERSION,
            OUTPUT_SCHEMA_VERSION,
        )

        observed_registry = metric_contract_registry_sha256()
        mismatches: list[str] = []
        if output.metric_registry_sha256 != observed_registry:
            mismatches.append("metric_registry_sha256")
        if output.output_schema_version != OUTPUT_SCHEMA_VERSION:
            mismatches.append("output_schema_version")
        if output.output_profile_id != CAMPAIGN_ANALYSIS_OUTPUT_PROFILE:
            mismatches.append("output_profile_id")
        if output.output_profile_schema_version != CAMPAIGN_ANALYSIS_SCHEMA_VERSION:
            mismatches.append("output_profile_schema_version")
        if output.output_profile_sha256 != CAMPAIGN_ANALYSIS_SCHEMA_SHA256:
            mismatches.append("output_profile_sha256")
        if output.expected_artifacts != CAMPAIGN_ANALYSIS_ARTIFACT_FILENAMES:
            mismatches.append("expected_artifacts")
        if output.output_schema_sha256 != MANIFEST_SCHEMA_SHA256:
            mismatches.append("output_schema_sha256")
        if mismatches:
            raise PreCampaignValidationError(
                "output contracts mismatch: " + ", ".join(sorted(mismatches))
            )
        return ProbeEvidence(
            {
                "metric_registry_sha256": observed_registry,
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "declared_output_schema_sha256": output.output_schema_sha256,
                "output_profile_id": CAMPAIGN_ANALYSIS_OUTPUT_PROFILE,
                "output_profile_schema_version": CAMPAIGN_ANALYSIS_SCHEMA_VERSION,
                "output_profile_sha256": CAMPAIGN_ANALYSIS_SCHEMA_SHA256,
                "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
                "manifest_schema_sha256": MANIFEST_SCHEMA_SHA256,
                "expected_artifacts": list(CAMPAIGN_ANALYSIS_ARTIFACT_FILENAMES),
                "campaign_ready": False,
            }
        )

    if population.source_registry_path is None:
        raise PreCampaignValidationError(
            "population source_registry_path is required for rate evidence"
        )

    def monetary_integrity_probe() -> ProbeEvidence:
        registry_sha = sha256(population.source_registry_path.read_bytes()).hexdigest()
        return rate_evidence_integrity_probe(
            monetary.source_bundle_path,
            required_source_registry_sha256=registry_sha,
        )

    plan_blockers: tuple[str, ...]
    parameter_interpretation = "UNAVAILABLE"
    try:
        plan_observation = observe_prospective_analysis_plan(analysis.plan_path)
        raw_blockers = plan_observation.metadata.get("campaign_blockers", [])
        plan_blockers = tuple(str(item) for item in raw_blockers)
    except (OSError, TypeError, ValueError, RuntimeError):
        plan_blockers = ("analysis_plan.identity=unavailable",)
    try:
        parameter_observation = observe_parameter_uncertainty_design(
            uncertainty.parameter_design_path
        )
        interpretation = parameter_observation.metadata.get(
            "probability_interpretation"
        )
        if type(interpretation) is str:
            parameter_interpretation = interpretation
    except (OSError, TypeError, ValueError, RuntimeError):
        pass

    blockers = set(plan_blockers)
    if config.provenance_status != "calibrated":
        blockers.add(
            "configuration.provenance_status=" + config.provenance_status
        )
    if population_contract.empirical_validation_status != "VALIDATED":
        blockers.add("population.empirical_validation=missing")
    if monetary.source_bundle_signature_status != "VERIFIED":
        blockers.add("monetary.source_bundle_signature=missing")
    if monetary.simulation_bridge_status != "VALIDATED":
        blockers.add("monetary.simulation_bridge=illustrative")

    return PreCampaignValidationSpec(
        repository_root=root,
        campaign_configuration_path=selected_config,
        semantic_expectations=(
            SemanticIdentityExpectation(
                expectation_id="analysis_plan.amended",
                expected_semantic_id=analysis.expected_plan_id,
                expected_semantic_sha256=analysis.expected_plan_sha256,
                observer=lambda: observe_prospective_analysis_plan(
                    analysis.plan_path
                ),
            ),
            SemanticIdentityExpectation(
                expectation_id="analysis_plan.parent",
                expected_semantic_id=analysis.parent_plan_id,
                expected_semantic_sha256=analysis.parent_plan_sha256,
                observer=lambda: observe_prospective_analysis_plan(
                    analysis.parent_plan_path
                ),
            ),
            SemanticIdentityExpectation(
                expectation_id="parameter.design",
                expected_semantic_id=uncertainty.parameter_design_id,
                expected_semantic_sha256=uncertainty.parameter_design_sha256,
                observer=lambda: observe_parameter_uncertainty_design(
                    uncertainty.parameter_design_path
                ),
            ),
        ),
        file_expectations=(
            FileHashExpectation(
                artifact_id="monetary.conversion_table.bytes",
                path=monetary.conversion_table_path,
                expected_sha256=monetary.conversion_table_sha256,
                schema_version="1.0",
            ),
            FileHashExpectation(
                artifact_id="monetary.source_artifact.bytes",
                path=monetary.source_artifact_path,
                expected_sha256=monetary.source_artifact_sha256,
                schema_version="1.0",
            ),
            FileHashExpectation(
                artifact_id="monetary.source_bundle.bytes",
                path=monetary.source_bundle_path,
                expected_sha256=monetary.source_bundle_sha256,
                schema_version="1",
            ),
            FileHashExpectation(
                artifact_id="population.design.bytes",
                path=population.design_bundle_path,
                expected_sha256=population_contract.design_sha256,
                schema_version=population_contract.design_schema_version,
            ),
            FileHashExpectation(
                artifact_id="population.runtime_mapping.bytes",
                path=population.runtime_mapping_bundle_path,
                expected_sha256=population_contract.runtime_mapping_sha256,
                schema_version=population_contract.runtime_mapping_schema_version,
            ),
        ),
        strict_probes=(
            StrictValidationProbe(
                probe_id="analysis_plan.runtime_bindings",
                validator=analysis_binding_probe,
            ),
            StrictValidationProbe(
                probe_id="configuration.strict_full_campaign_schema",
                validator=configuration_probe,
            ),
            StrictValidationProbe(
                probe_id="flow.policy_only_contract",
                validator=lambda: policy_flow_contract_probe(analysis.plan_path),
            ),
            StrictValidationProbe(
                probe_id="monetary.campaign_gate",
                validator=lambda: rate_evidence_campaign_gate_probe(
                    monetary.source_bundle_path
                ),
            ),
            StrictValidationProbe(
                probe_id="monetary.rate_evidence_integrity",
                validator=monetary_integrity_probe,
            ),
            StrictValidationProbe(
                probe_id="outputs.contracts",
                validator=output_contract_probe,
            ),
            StrictValidationProbe(
                probe_id="population.adapter_integrity",
                validator=population_integrity_probe,
            ),
            StrictValidationProbe(
                probe_id="population.campaign_gate",
                validator=population_campaign_gate,
            ),
        ),
        uncertainty=UncertaintyDeclaration(
            minimum_retained_seeds=uncertainty.minimum_retained_seeds,
            fixed_seeds=config.batch.seeds,
            parameter_probability_interpretation=parameter_interpretation,
            monetary_rate_status=uncertainty.monetary_rate_uncertainty.value,
            population_status=uncertainty.population_uncertainty.value,
            combined_uncertainty_required=uncertainty.combined_uncertainty_required,
        ),
        receipt_spec=receipt_spec,
        scientific_readiness_blockers=tuple(sorted(blockers)),
    )


def run_pre_campaign_validation(
    spec: PreCampaignValidationSpec,
    *,
    _git_observer: Callable[[Path], GitObservation] | None = None,
    _receipt_builder: Callable[[Any], Any] | None = None,
    _receipt_verifier: Callable[..., Any] | None = None,
) -> PreCampaignValidationReport:
    """Re-attest declared inputs and report blockers without model execution."""

    if type(spec) is not PreCampaignValidationSpec:
        raise TypeError("spec must be PreCampaignValidationSpec")
    PreCampaignValidationSpec.__post_init__(spec)
    git_observer = _git_observer or observe_git_repository
    receipt_builder = _receipt_builder or build_execution_receipt
    receipt_verifier = _receipt_verifier or verify_execution_receipt

    checks: list[PreflightCheck] = []
    observations: list[dict[str, object]] = []
    blockers = set(spec.scientific_readiness_blockers)

    git_observation = git_observer(spec.repository_root)
    if git_observation.observation_error is None:
        checks.append(
            PreflightCheck(
                check_id="repository.identity_observed",
                status=PreflightCheckStatus.PASSED,
                detail="active branch, exact commit, and porcelain status observed",
            )
        )
    else:
        _append_failure(
            checks,
            blockers,
            check_id="repository.identity_observed",
            detail=git_observation.observation_error,
        )
    if git_observation.working_tree_clean:
        checks.append(
            PreflightCheck(
                check_id="repository.working_tree_clean",
                status=PreflightCheckStatus.PASSED,
                detail="git status --porcelain is empty",
            )
        )
    else:
        _append_failure(
            checks,
            blockers,
            check_id="repository.working_tree_clean",
            detail="git status --porcelain is non-empty",
        )

    configuration_snapshot = _observe_unbound_configuration(
        spec.repository_root,
        spec.campaign_configuration_path,
        checks=checks,
        blockers=blockers,
    )

    for expectation in sorted(
        spec.file_expectations,
        key=lambda item: item.artifact_id,
    ):
        _evaluate_file_expectation(
            spec.repository_root,
            expectation,
            checks=checks,
            observations=observations,
            blockers=blockers,
        )

    parameter_interpretation = spec.uncertainty.parameter_probability_interpretation
    for expectation in sorted(
        spec.semantic_expectations,
        key=lambda item: item.expectation_id,
    ):
        observed = _evaluate_semantic_expectation(
            spec.repository_root,
            expectation,
            checks=checks,
            observations=observations,
            blockers=blockers,
        )
        if observed is not None and expectation.expectation_id == "parameter.design":
            value = observed.metadata.get("probability_interpretation")
            if type(value) is str:
                parameter_interpretation = value

    for probe in sorted(spec.strict_probes, key=lambda item: item.probe_id):
        _evaluate_probe(
            probe,
            checks=checks,
            observations=observations,
            blockers=blockers,
        )

    receipt_payload = _attempt_receipt(
        spec.receipt_spec,
        builder=receipt_builder,
        verifier=receipt_verifier,
        checks=checks,
        blockers=blockers,
    )

    uncertainty_snapshot, uncertainty_blockers = _uncertainty_snapshot(
        spec.uncertainty,
        parameter_probability_interpretation=parameter_interpretation,
    )
    blockers.update(uncertainty_blockers)
    blockers.add("convergence.no_realizations=non_converged")

    checks.sort(key=lambda item: item.check_id)
    passed = [
        item.check_id
        for item in checks
        if item.status is PreflightCheckStatus.PASSED
    ]
    failed = [
        item.check_id
        for item in checks
        if item.status is PreflightCheckStatus.FAILED
    ]
    payload: dict[str, object] = {
        "$schema": PRE_CAMPAIGN_REPORT_SCHEMA_ID,
        "report_schema_version": PRE_CAMPAIGN_REPORT_SCHEMA_VERSION,
        "identity_algorithm": PRE_CAMPAIGN_REPORT_IDENTITY_ALGORITHM,
        "execution_mode": PRE_CAMPAIGN_EXECUTION_MODE,
        "repository": git_observation.snapshot(),
        "campaign_configuration": configuration_snapshot,
        "identity_observations": sorted(
            observations,
            key=lambda item: str(item["observation_id"]),
        ),
        "checks": [item.snapshot() for item in checks],
        "passed_checks": passed,
        "failed_checks": failed,
        "unresolved_blockers": sorted(blockers),
        "uncertainty_components": uncertainty_snapshot,
        "convergence": _non_convergence_snapshot(),
        "execution_receipt": receipt_payload,
        "primary_result": _no_primary_result_snapshot(),
        "full_campaign_intentionally_not_run": True,
        "campaign_ready": False,
    }
    canonical = _canonical_json_bytes(payload).decode("utf-8")
    return PreCampaignValidationReport(
        identity_payload_json=canonical,
        report_sha256=sha256(canonical.encode("utf-8")).hexdigest(),
    )


def observe_git_repository(repository_root: Path) -> GitObservation:
    """Read the exact current Git identity without requiring a clean tree."""

    root = Path(os.path.abspath(os.fspath(repository_root)))
    try:
        top = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve(strict=True)
        if top != root.resolve(strict=True):
            raise PreCampaignValidationError(
                "repository_root differs from Git's top-level worktree"
            )
        commit = _git_text(root, "rev-parse", "--verify", "HEAD")
        object_format = _git_text(root, "rev-parse", "--show-object-format")
        try:
            branch = _git_text(root, "symbolic-ref", "--quiet", "--short", "HEAD")
        except PreCampaignValidationError:
            branch = None
        status = _git_text(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            allow_empty=True,
        )
        return GitObservation(
            repository_root=root,
            active_branch=branch,
            exact_commit=commit,
            object_format=object_format,
            status_porcelain=status,
            working_tree_clean=status == "",
            observation_error=(
                "detached HEAD has no active branch" if branch is None else None
            ),
        )
    except (OSError, ValueError, PreCampaignValidationError) as exc:
        return GitObservation(
            repository_root=root,
            active_branch=None,
            exact_commit=None,
            object_format=None,
            status_porcelain="<unavailable>",
            working_tree_clean=False,
            observation_error=_exception_detail(exc),
        )


def write_pre_campaign_validation_report(
    path: str | Path,
    report: PreCampaignValidationReport,
) -> Path:
    """Atomically write one report, refusing to replace different evidence."""

    if type(report) is not PreCampaignValidationReport:
        raise TypeError("report must be PreCampaignValidationReport")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        json.dumps(
            report.snapshot(),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise PreCampaignValidationError(
                "existing pre-campaign report must be a regular file"
            )
        if destination.read_bytes() == rendered:
            return destination
        raise PreCampaignValidationError(
            "refusing to overwrite a different pre-campaign validation report"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != rendered:
                raise PreCampaignValidationError(
                    "refusing to overwrite a concurrently written different report"
                )
        temporary.unlink(missing_ok=True)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _observe_unbound_configuration(
    repository_root: Path,
    path: Path,
    *,
    checks: list[PreflightCheck],
    blockers: set[str],
) -> dict[str, object]:
    try:
        resolved, relative, content = _read_repository_file(repository_root, path)
        digest = sha256(content).hexdigest()
        checks.append(
            PreflightCheck(
                check_id="configuration.bytes_observed",
                status=PreflightCheckStatus.PASSED,
                detail="campaign configuration raw bytes were hashed",
                observed_sha256=digest,
            )
        )
        return {
            "path": relative,
            "path_base": "repository_root",
            "byte_length": len(content),
            "file_sha256": digest,
        }
    except (OSError, ValueError, PreCampaignValidationError) as exc:
        _append_failure(
            checks,
            blockers,
            check_id="configuration.bytes_observed",
            detail=_exception_detail(exc),
        )
        return {
            "path": _safe_path_text(path),
            "path_base": "repository_root",
            "byte_length": None,
            "file_sha256": None,
        }


def _evaluate_file_expectation(
    repository_root: Path,
    expectation: FileHashExpectation,
    *,
    checks: list[PreflightCheck],
    observations: list[dict[str, object]],
    blockers: set[str],
) -> None:
    check_id = expectation.artifact_id
    try:
        if _BLOCKED_IDENTITY.fullmatch(expectation.expected_sha256):
            raise PreCampaignValidationError(
                "expected file identity is an unresolved BLOCKED_PENDING placeholder"
            )
        _digest(expectation.expected_sha256, name="expected file SHA-256")
        _resolved, relative, content = _read_repository_file(
            repository_root,
            expectation.path,
        )
        observed = sha256(content).hexdigest()
        observations.append(
            {
                "observation_id": check_id,
                "kind": "RAW_FILE_BYTES",
                "path": relative,
                "path_base": "repository_root",
                "schema_version": expectation.schema_version,
                "byte_length": len(content),
                "file_sha256": observed,
            }
        )
        if observed != expectation.expected_sha256:
            raise PreCampaignValidationError("raw file SHA-256 mismatch")
        checks.append(
            PreflightCheck(
                check_id=check_id,
                status=PreflightCheckStatus.PASSED,
                detail="raw file bytes match the declared SHA-256",
                expected_sha256=expectation.expected_sha256,
                observed_sha256=observed,
            )
        )
    except (OSError, TypeError, ValueError, PreCampaignValidationError) as exc:
        expected = (
            expectation.expected_sha256
            if _SHA256.fullmatch(expectation.expected_sha256)
            else None
        )
        _append_failure(
            checks,
            blockers,
            check_id=check_id,
            detail=_exception_detail(exc),
            expected_sha256=expected,
        )


def _evaluate_semantic_expectation(
    repository_root: Path,
    expectation: SemanticIdentityExpectation,
    *,
    checks: list[PreflightCheck],
    observations: list[dict[str, object]],
    blockers: set[str],
) -> SemanticArtifactObservation | None:
    check_id = expectation.expectation_id
    try:
        if _BLOCKED_IDENTITY.fullmatch(expectation.expected_semantic_id) or (
            _BLOCKED_IDENTITY.fullmatch(expectation.expected_semantic_sha256)
        ):
            raise PreCampaignValidationError(
                "expected semantic identity is an unresolved BLOCKED_PENDING placeholder"
            )
        _digest(
            expectation.expected_semantic_sha256,
            name="expected semantic SHA-256",
        )
        observed = expectation.observer()
        if type(observed) is not SemanticArtifactObservation:
            raise TypeError(
                "semantic observer must return SemanticArtifactObservation"
            )
        snapshot = observed.snapshot(repository_root=repository_root)
        observations.append(
            {
                "observation_id": check_id,
                "kind": "SEMANTIC_ARTIFACT",
                **snapshot,
            }
        )
        mismatches: list[str] = []
        if observed.semantic_id != expectation.expected_semantic_id:
            mismatches.append("semantic_id")
        if observed.semantic_sha256 != expectation.expected_semantic_sha256:
            mismatches.append("semantic_sha256")
        if mismatches:
            raise PreCampaignValidationError(
                "semantic identity mismatch: " + ", ".join(mismatches)
            )
        checks.append(
            PreflightCheck(
                check_id=check_id,
                status=PreflightCheckStatus.PASSED,
                detail="semantic loader re-attested the expected identity",
                expected_sha256=expectation.expected_semantic_sha256,
                observed_sha256=observed.semantic_sha256,
            )
        )
        return observed
    except (OSError, TypeError, ValueError, RuntimeError, KeyError) as exc:
        expected = (
            expectation.expected_semantic_sha256
            if _SHA256.fullmatch(expectation.expected_semantic_sha256)
            else None
        )
        _append_failure(
            checks,
            blockers,
            check_id=check_id,
            detail=_exception_detail(exc),
            expected_sha256=expected,
        )
        return None


def _evaluate_probe(
    probe: StrictValidationProbe,
    *,
    checks: list[PreflightCheck],
    observations: list[dict[str, object]],
    blockers: set[str],
) -> None:
    try:
        raw = probe.validator()
        evidence = (
            raw
            if type(raw) is ProbeEvidence
            else ProbeEvidence({} if raw is None else raw)
        )
        observations.append(
            {
                "observation_id": probe.probe_id,
                "kind": "STRICT_GATE_PROBE",
                "evidence_sha256": evidence.evidence_sha256,
                "evidence": _json_copy(evidence.payload),
            }
        )
        checks.append(
            PreflightCheck(
                check_id=probe.probe_id,
                status=PreflightCheckStatus.PASSED,
                detail="strict validator completed without weakening its gate",
                evidence_sha256=evidence.evidence_sha256,
            )
        )
    except (OSError, TypeError, ValueError, RuntimeError, KeyError) as exc:
        _append_failure(
            checks,
            blockers,
            check_id=probe.probe_id,
            detail=_exception_detail(exc),
        )


def _attempt_receipt(
    receipt_spec: object,
    *,
    builder: Callable[[Any], Any],
    verifier: Callable[..., Any],
    checks: list[PreflightCheck],
    blockers: set[str],
) -> dict[str, object]:
    try:
        receipt = builder(receipt_spec)
        verification = verifier(
            receipt,
            receipt_spec,
            phase=ExecutionVerificationPhase.PRE_EXECUTION,
        )
        digest = getattr(receipt, "execution_receipt_sha256")
        _digest(digest, name="execution_receipt_sha256")
        receipt_blockers = tuple(getattr(verification, "blockers"))
        blockers.update(str(item) for item in receipt_blockers)
        checks.append(
            PreflightCheck(
                check_id="execution_receipt.pre_execution_identity",
                status=PreflightCheckStatus.PASSED,
                detail="canonical receipt was generated and reverified before execution",
                observed_sha256=digest,
            )
        )
        identity_payload = getattr(receipt, "identity_payload", {})
        source_tree = (
            identity_payload.get("source_tree", {})
            if isinstance(identity_payload, Mapping)
            else {}
        )
        return {
            "status": ReceiptAttemptStatus.GENERATED_AND_PREVERIFIED.value,
            "execution_receipt_sha256": digest,
            "source_tree_sha256": (
                source_tree.get("source_tree_sha256")
                if isinstance(source_tree, Mapping)
                else None
            ),
            "campaign_execution_admissible": False,
            "blockers": sorted(str(item) for item in receipt_blockers),
            "rejection_type": None,
            "rejection_reason": None,
        }
    except (OSError, TypeError, ValueError, RuntimeError, KeyError) as exc:
        detail = _exception_detail(exc)
        _append_failure(
            checks,
            blockers,
            check_id="execution_receipt.pre_execution_identity",
            detail=detail,
        )
        blockers.add("execution_receipt.pre_execution=rejected")
        return {
            "status": ReceiptAttemptStatus.REJECTED.value,
            "execution_receipt_sha256": None,
            "source_tree_sha256": None,
            "campaign_execution_admissible": False,
            "blockers": ["execution_receipt.pre_execution=rejected"],
            "rejection_type": type(exc).__name__,
            "rejection_reason": detail,
        }


def _uncertainty_snapshot(
    declaration: UncertaintyDeclaration,
    *,
    parameter_probability_interpretation: str,
) -> tuple[dict[str, object], tuple[str, ...]]:
    seed_design_sufficient = (
        len(declaration.fixed_seeds) >= declaration.minimum_retained_seeds
    )
    seed_state = (
        UncertaintyComponentState.DESIGN_DECLARED_NOT_EXECUTED
        if seed_design_sufficient
        else UncertaintyComponentState.UNAVAILABLE
    )
    parameter_calibrated = (
        parameter_probability_interpretation
        == "CALIBRATED_JOINT_DISTRIBUTION"
    )
    parameter_state = (
        UncertaintyComponentState.QUANTIFIED_DESIGN_NOT_EXECUTED
        if parameter_calibrated
        else UncertaintyComponentState.UNQUANTIFIED
    )
    rate_state = _declared_component_state(declaration.monetary_rate_status)
    population_state = _declared_component_state(declaration.population_status)
    all_quantified = (
        seed_design_sufficient
        and parameter_calibrated
        and rate_state
        is UncertaintyComponentState.QUANTIFIED_DESIGN_NOT_EXECUTED
        and population_state
        is UncertaintyComponentState.QUANTIFIED_DESIGN_NOT_EXECUTED
    )
    combined_state = (
        UncertaintyComponentState.DESIGN_DECLARED_NOT_EXECUTED
        if all_quantified
        else UncertaintyComponentState.UNAVAILABLE
    )
    components = {
        "seed": _component_snapshot(
            seed_state,
            reason=(
                "fixed seed design is declared but no seed was executed"
                if seed_design_sufficient
                else "fixed seed design is smaller than the declared minimum"
            ),
        ),
        "model_parameter": _component_snapshot(
            parameter_state,
            reason=(
                "calibrated joint parameter design exists but was not executed"
                if parameter_calibrated
                else "parameter ranges are illustrative and are not a probability distribution"
            ),
        ),
        "monetary_rate": _component_snapshot(
            rate_state,
            reason=(
                "declared rate-uncertainty design was not executed"
                if rate_state
                is UncertaintyComponentState.QUANTIFIED_DESIGN_NOT_EXECUTED
                else "official point-rate evidence does not quantify uncertainty about the rate"
            ),
        ),
        "population": _component_snapshot(
            population_state,
            reason=(
                "declared population-uncertainty design was not executed"
                if population_state
                is UncertaintyComponentState.QUANTIFIED_DESIGN_NOT_EXECUTED
                else "exact design weights do not quantify target-population uncertainty"
            ),
        ),
        "combined": _component_snapshot(
            combined_state,
            reason=(
                "all component designs are declared but no joint realization was executed"
                if all_quantified
                else "one or more required uncertainty components are unquantified or unavailable"
            ),
        ),
    }
    blockers: list[str] = []
    if not seed_design_sufficient:
        blockers.append("uncertainty.seed_design=minimum_not_reached")
    if not parameter_calibrated:
        blockers.append("uncertainty.parameter_distribution=uncalibrated")
    if declaration.monetary_rate_status != "QUANTIFIED":
        blockers.append(
            "uncertainty.monetary_rate="
            + declaration.monetary_rate_status.lower()
        )
    if declaration.population_status != "QUANTIFIED":
        blockers.append(
            "uncertainty.population=" + declaration.population_status.lower()
        )
    if declaration.combined_uncertainty_required and not all_quantified:
        blockers.append("uncertainty.combined=unavailable")
    return components, tuple(sorted(blockers))


def _declared_component_state(value: str) -> UncertaintyComponentState:
    if value == "QUANTIFIED":
        return UncertaintyComponentState.QUANTIFIED_DESIGN_NOT_EXECUTED
    if value == "UNQUANTIFIED":
        return UncertaintyComponentState.UNQUANTIFIED
    return UncertaintyComponentState.UNAVAILABLE


def _component_snapshot(
    state: UncertaintyComponentState,
    *,
    reason: str,
) -> dict[str, object]:
    return {
        "state": state.value,
        "variance": None,
        "interval": None,
        "realization_count": 0,
        "reason": reason,
        "unavailable_components_are_zero": False,
    }


def _non_convergence_snapshot() -> dict[str, object]:
    return {
        "status": "NON_CONVERGED",
        "retained_seed_count": 0,
        "completed_realization_count": 0,
        "checkpoint_count": 0,
        "cumulative_point_estimate": None,
        "cumulative_monte_carlo_standard_error": None,
        "interval_width": None,
        "absolute_change_from_previous_checkpoint": None,
        "relative_change_from_previous_checkpoint": None,
        "invalid_realization_count": 0,
        "rejected_realization_count": 0,
        "excluded_realization_count": 0,
        "sensitivity_instability_status": "NOT_EVALUATED",
        "sufficiency_judgment": "INSUFFICIENT_NO_REALIZATIONS",
    }


def _no_primary_result_snapshot() -> dict[str, object]:
    return {
        "point_estimate": None,
        "seed_uncertainty": None,
        "parameter_uncertainty": None,
        "monetary_rate_uncertainty": None,
        "population_uncertainty": None,
        "total_uncertainty": None,
        "sufficiency_judgment": (
            "INSUFFICIENT_NO_REALIZATIONS_AND_REQUIRED_COMPONENTS_UNAVAILABLE"
        ),
    }


def _append_failure(
    checks: list[PreflightCheck],
    blockers: set[str],
    *,
    check_id: str,
    detail: str,
    expected_sha256: str | None = None,
) -> None:
    checks.append(
        PreflightCheck(
            check_id=check_id,
            status=PreflightCheckStatus.FAILED,
            detail=detail,
            expected_sha256=expected_sha256,
        )
    )
    blockers.add(f"preflight.{check_id}=failed")


def _validate_report_payload(payload: Mapping[str, object]) -> None:
    expected = {
        "$schema",
        "report_schema_version",
        "identity_algorithm",
        "execution_mode",
        "repository",
        "campaign_configuration",
        "identity_observations",
        "checks",
        "passed_checks",
        "failed_checks",
        "unresolved_blockers",
        "uncertainty_components",
        "convergence",
        "execution_receipt",
        "primary_result",
        "full_campaign_intentionally_not_run",
        "campaign_ready",
    }
    if set(payload) != expected:
        raise PreCampaignValidationError("report payload keys differ from schema")
    if payload.get("$schema") != PRE_CAMPAIGN_REPORT_SCHEMA_ID:
        raise PreCampaignValidationError("report $schema is unsupported")
    if payload.get("report_schema_version") != PRE_CAMPAIGN_REPORT_SCHEMA_VERSION:
        raise PreCampaignValidationError("report schema version is unsupported")
    if payload.get("identity_algorithm") != PRE_CAMPAIGN_REPORT_IDENTITY_ALGORITHM:
        raise PreCampaignValidationError("report identity algorithm is unsupported")
    if payload.get("execution_mode") != PRE_CAMPAIGN_EXECUTION_MODE:
        raise PreCampaignValidationError("report execution mode is not pre-campaign only")
    if payload.get("full_campaign_intentionally_not_run") is not True:
        raise PreCampaignValidationError("report must attest that the campaign was not run")
    if payload.get("campaign_ready") is not False:
        raise PreCampaignValidationError("schema-v1 report campaign_ready must be false")
    convergence = payload.get("convergence")
    if not isinstance(convergence, Mapping) or dict(convergence) != (
        _non_convergence_snapshot()
    ):
        raise PreCampaignValidationError(
            "pre-campaign report must retain the exact no-realization "
            "NON_CONVERGED state"
        )
    if payload.get("primary_result") != _no_primary_result_snapshot():
        raise PreCampaignValidationError(
            "pre-campaign report cannot contain a point estimate or uncertainty result"
        )
    components = payload.get("uncertainty_components")
    expected_components = {
        "seed",
        "model_parameter",
        "monetary_rate",
        "population",
        "combined",
    }
    if not isinstance(components, Mapping) or set(components) != expected_components:
        raise PreCampaignValidationError(
            "pre-campaign uncertainty components are incomplete"
        )
    allowed_states = {item.value for item in UncertaintyComponentState}
    for name, raw_component in components.items():
        if not isinstance(raw_component, Mapping):
            raise PreCampaignValidationError(
                f"pre-campaign uncertainty component {name} must be an object"
            )
        if (
            set(raw_component)
            != {
                "state",
                "variance",
                "interval",
                "realization_count",
                "reason",
                "unavailable_components_are_zero",
            }
            or raw_component.get("state") not in allowed_states
            or raw_component.get("variance") is not None
            or raw_component.get("interval") is not None
            or raw_component.get("realization_count") != 0
            or raw_component.get("unavailable_components_are_zero") is not False
            or type(raw_component.get("reason")) is not str
            or not str(raw_component.get("reason")).strip()
        ):
            raise PreCampaignValidationError(
                f"pre-campaign uncertainty component {name} contradicts no execution"
            )
    receipt = payload.get("execution_receipt")
    if not isinstance(receipt, Mapping) or receipt.get(
        "campaign_execution_admissible"
    ) is not False:
        raise PreCampaignValidationError("execution receipt cannot admit a campaign")
    receipt_status = receipt.get("status")
    if receipt_status == ReceiptAttemptStatus.GENERATED_AND_PREVERIFIED.value:
        if (
            not _SHA256.fullmatch(str(receipt.get("execution_receipt_sha256")))
            or not _SHA256.fullmatch(str(receipt.get("source_tree_sha256")))
            or receipt.get("rejection_type") is not None
            or receipt.get("rejection_reason") is not None
        ):
            raise PreCampaignValidationError(
                "generated execution receipt evidence is incomplete"
            )
    elif receipt_status == ReceiptAttemptStatus.REJECTED.value:
        if (
            receipt.get("execution_receipt_sha256") is not None
            or receipt.get("source_tree_sha256") is not None
            or type(receipt.get("rejection_type")) is not str
            or not str(receipt.get("rejection_type")).strip()
            or type(receipt.get("rejection_reason")) is not str
            or not str(receipt.get("rejection_reason")).strip()
        ):
            raise PreCampaignValidationError(
                "rejected execution receipt evidence is inconsistent"
            )
    else:
        raise PreCampaignValidationError("execution receipt status is unsupported")
    _canonical_json_bytes(payload)


def _read_repository_file(
    repository_root: Path,
    path: Path,
) -> tuple[Path, str, bytes]:
    root = repository_root.resolve(strict=True)
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=True)
    try:
        relative_path = resolved.relative_to(root)
    except ValueError as exc:
        raise PreCampaignValidationError(
            "preflight input must remain inside repository_root"
        ) from exc
    metadata = resolved.lstat()
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if (
        resolved.is_symlink()
        or bool(getattr(metadata, "st_file_attributes", 0) & marker)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise PreCampaignValidationError(
            "preflight input must be a regular non-symlink file"
        )
    return resolved, PurePosixPath(relative_path).as_posix(), resolved.read_bytes()


def _repository_relative_path(repository_root: Path, path: Path) -> str:
    root = repository_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PreCampaignValidationError(
            "semantic artifact must remain inside repository_root"
        ) from exc
    return PurePosixPath(relative).as_posix()


def _git_text(
    repository_root: Path,
    *args: str,
    allow_empty: bool = False,
) -> str:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=repository_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise PreCampaignValidationError("Git command could not be executed") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PreCampaignValidationError(
            "Git identity command failed" + (f": {detail}" if detail else "")
        )
    try:
        value = completed.stdout.decode("utf-8", errors="strict").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise PreCampaignValidationError("Git output is not valid UTF-8") from exc
    if not value and not allow_empty:
        raise PreCampaignValidationError("Git identity command returned empty output")
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PreCampaignValidationError(
            "value is not canonical JSON-compatible"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


def _json_copy(value: object) -> object:
    return json.loads(_canonical_json_bytes(value))


def _canonical_blockers(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(
        values,
        Sequence,
    ):
        raise TypeError("scientific_readiness_blockers must be a sequence")
    result: list[str] = []
    for index, value in enumerate(values):
        _nonempty_text(value, name=f"scientific_readiness_blockers[{index}]")
        result.append(value)
    return tuple(sorted(set(result)))


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise PreCampaignValidationError(f"{name} is not a canonical identifier")
    return value


def _nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise PreCampaignValidationError(f"{name} must be non-empty text")
    return value


def _digest(value: object, *, name: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise PreCampaignValidationError(f"{name} must be lowercase SHA-256 hex")
    return value


def _strict_int(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or isinstance(value, bool) or value < minimum:
        raise PreCampaignValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _normalized_absolute_path(path: Path) -> str:
    return Path(os.path.abspath(os.fspath(path))).as_posix()


def _safe_path_text(path: Path) -> str:
    try:
        return PurePosixPath(path).as_posix()
    except (TypeError, ValueError):
        return "<unavailable>"


def _exception_detail(exc: BaseException) -> str:
    message = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {message or 'no detail'}"


__all__ = [
    "FileHashExpectation",
    "GitObservation",
    "PRE_CAMPAIGN_EXECUTION_MODE",
    "PRE_CAMPAIGN_REPORT_IDENTITY_ALGORITHM",
    "PRE_CAMPAIGN_REPORT_SCHEMA_ID",
    "PRE_CAMPAIGN_REPORT_SCHEMA_VERSION",
    "PreCampaignValidationError",
    "PreCampaignValidationReport",
    "PreCampaignValidationSpec",
    "PreflightCheck",
    "PreflightCheckStatus",
    "ProbeEvidence",
    "ReceiptAttemptStatus",
    "SemanticArtifactObservation",
    "SemanticIdentityExpectation",
    "StrictValidationProbe",
    "UncertaintyComponentState",
    "UncertaintyDeclaration",
    "build_policy_campaign_execution_receipt_spec",
    "build_policy_campaign_preflight_spec",
    "observe_git_repository",
    "observe_parameter_uncertainty_design",
    "observe_prospective_analysis_plan",
    "policy_flow_contract_probe",
    "rate_evidence_campaign_gate_probe",
    "rate_evidence_integrity_probe",
    "run_pre_campaign_validation",
    "write_pre_campaign_validation_report",
]
