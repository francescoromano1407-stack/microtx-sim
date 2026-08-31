"""Strict TOML configuration for the synthetic policy-prototype runner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
import os
from pathlib import Path
import re
from typing import Mapping
import tomllib

from .causal.batch import PolicyBatchSpec
from .config import PopulationProjectionConfig, _population_projection_config
from .consumers.decision import DecisionParameters
from .funding import EPGCPolicy
from .metrics.harm import (
    HarmModelParameters,
    OpportunityCostValuation,
    WelfareHarmWeights,
)
from .simulation.policy_orchestrator import ProducerAssumptions
from .types import LedgerBackend


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BLOCKED_IDENTITY = re.compile(r"BLOCKED_PENDING_[A-Z0-9_]+\Z")
_ATTEMPT_ID = re.compile(r"attempt-[0-9]{6}\Z")

EXPLORATORY_ARTIFACT_NAMESPACE = "policy_exploratory_synthetic"
EXPLORATORY_EXECUTION_KIND = "COMPUTATIONAL_SIMULATION"
EXPLORATORY_POPULATION_BASIS = "ILLUSTRATIVE_NON_EMPIRICAL"
EXPLORATORY_ESTIMAND_INTERPRETATION = "CONDITIONAL_ON_MODEL_ASSUMPTIONS"
EXPLORATORY_MONETARY_AMOUNT_SEMANTICS = (
    "SYNTHETIC_MODEL_EQUIVALENT_NOT_OBSERVED_SPENDING"
)
EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE = "DIAGNOSTIC_ONLY"
EXPLORATORY_INTERNAL_MONETARY_UNIT = "simulation_cents"
EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE = (
    "DIAGNOSTIC_ONLY_NOT_A_CROSS_COUNTRY_MONETARY_RESULT"
)
EXPLORATORY_PLAN_FILENAME = (
    "exploratory-synthetic-analysis-plan-v1.json"
)
EXPLORATORY_PLAN_ID = (
    "illustrative.exploratory.synthetic.composite-harm.baseline-vs-safe.v1"
)
EXPLORATORY_PARENT_PLAN_ID = (
    "illustrative.prospective.composite-harm.baseline-vs-safe.v3"
)
EXPLORATORY_PARENT_PLAN_FILENAME = "prospective-analysis-plan-amendment-v3.json"
EXPLORATORY_SCIENTIFIC_PARENT_PLAN_ID = (
    "illustrative.prospective.composite-harm.baseline-vs-safe.v2"
)
EXPLORATORY_SCIENTIFIC_PARENT_PLAN_FILENAME = "prospective-analysis-plan.json"
EXPLORATORY_PRIMARY_ESTIMAND_ID = (
    "primary.composite-harm.baseline-vs-safe.v1"
)


class PolicyConfigurationError(ValueError):
    """Raised when the policy-prototype configuration is ambiguous or unsafe."""


class PolicyRunPurpose(str, Enum):
    """The declared scientific/operational purpose of a policy configuration."""

    DEVELOPMENT = "development"
    EXPLORATORY = "exploratory"
    CAMPAIGN = "campaign"


class PolicySimulationLayer(str, Enum):
    """The one simulation layer authorized by a policy campaign contract."""

    POLICY_ORCHESTRATOR = "policy_orchestrator"


class UncertaintyAvailability(str, Enum):
    """Whether a declared source has an admissible uncertainty design."""

    QUANTIFIED = "QUANTIFIED"
    UNQUANTIFIED = "UNQUANTIFIED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PolicyOutputConfig:
    output_dir: Path
    histogram_bins: int = 20
    include_player_rows: bool = True
    run_sensitivity: bool = True

    def __post_init__(self) -> None:
        if not str(self.output_dir):
            raise ValueError("output_dir cannot be empty")
        if isinstance(self.histogram_bins, bool) or not isinstance(
            self.histogram_bins, int
        ):
            raise TypeError("histogram_bins must be an integer")
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")
        if not isinstance(self.include_player_rows, bool):
            raise TypeError("include_player_rows must be boolean")
        if not isinstance(self.run_sensitivity, bool):
            raise TypeError("run_sensitivity must be boolean")


@dataclass(frozen=True, slots=True)
class AnalysisPlanSelection:
    """Non-verifying locator and optional expected plan identities."""

    plan_path: Path
    expected_plan_id: str | None = None
    expected_plan_sha256: str | None = None
    parent_plan_path: Path | None = None
    parent_plan_id: str | None = None
    parent_plan_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan_path, Path):
            raise TypeError("analysis plan_path must be a Path")
        if not str(self.plan_path):
            raise ValueError("analysis plan_path cannot be empty")
        optional = (
            self.expected_plan_id,
            self.expected_plan_sha256,
            self.parent_plan_path,
            self.parent_plan_id,
            self.parent_plan_sha256,
        )
        if any(value is not None for value in optional) and any(
            value is None for value in optional
        ):
            raise ValueError(
                "analysis plan expected and parent identities must be supplied "
                "together"
            )
        if self.expected_plan_id is not None:
            _identity_text_or_placeholder(
                self.expected_plan_id,
                name="analysis expected_plan_id",
            )
            _identity_sha256_or_placeholder(
                self.expected_plan_sha256,
                name="analysis expected_plan_sha256",
            )
            if not isinstance(self.parent_plan_path, Path) or not str(
                self.parent_plan_path
            ):
                raise TypeError("analysis parent_plan_path must be a Path")
            _identity_text_or_placeholder(
                self.parent_plan_id,
                name="analysis parent_plan_id",
            )
            _identity_sha256_or_placeholder(
                self.parent_plan_sha256,
                name="analysis parent_plan_sha256",
            )

    def snapshot(self) -> dict[str, str]:
        payload = {"plan_path": str(self.plan_path)}
        if self.expected_plan_id is not None:
            assert self.expected_plan_sha256 is not None
            assert self.parent_plan_path is not None
            assert self.parent_plan_id is not None
            assert self.parent_plan_sha256 is not None
            payload.update(
                {
                    "expected_plan_id": self.expected_plan_id,
                    "expected_plan_sha256": self.expected_plan_sha256,
                    "parent_plan_path": str(self.parent_plan_path),
                    "parent_plan_id": self.parent_plan_id,
                    "parent_plan_sha256": self.parent_plan_sha256,
                }
            )
        return payload


@dataclass(frozen=True, slots=True)
class CampaignControlConfig:
    """Fail-closed authorization boundary for a full policy campaign."""

    allow_synthetic: bool
    fail_closed: bool
    simulation_layer: PolicySimulationLayer
    campaign_ready: bool
    primary_estimand_id: str

    def __post_init__(self) -> None:
        if type(self.allow_synthetic) is not bool:
            raise TypeError("campaign allow_synthetic must be boolean")
        if self.allow_synthetic:
            raise ValueError("full campaigns require allow_synthetic = false")
        if type(self.fail_closed) is not bool or not self.fail_closed:
            raise ValueError("full campaigns require fail_closed = true")
        if type(self.simulation_layer) is not PolicySimulationLayer:
            raise TypeError("campaign simulation_layer is invalid")
        if type(self.campaign_ready) is not bool:
            raise TypeError("campaign campaign_ready must be boolean")
        _nonempty_text(self.primary_estimand_id, name="campaign primary_estimand_id")


@dataclass(frozen=True, slots=True)
class ExploratoryControlConfig:
    """Fixed non-production semantics for a synthetic computational run."""

    exploratory_plan_path: Path
    exploratory_plan_id: str
    exploratory_plan_sha256: str
    artifact_namespace: str
    execution_kind: str
    population_basis: str
    estimand_interpretation: str
    monetary_amount_semantics: str
    unweighted_output_role: str
    internal_monetary_unit: str
    raw_internal_unit_output_role: str
    execution_enabled: bool
    allow_synthetic: bool
    campaign_ready: bool
    production_campaign: bool
    empirical_claims: bool
    population_inference_claims: bool
    causal_claims: bool
    generalisation_claims: bool
    identical_pretreatment_cohorts: bool
    identical_population_weights_across_scenarios: bool
    primary_estimand_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.exploratory_plan_path, Path) or not str(
            self.exploratory_plan_path
        ):
            raise TypeError("exploratory_plan_path must be a Path")
        if self.exploratory_plan_path.name != EXPLORATORY_PLAN_FILENAME:
            raise ValueError(
                "exploratory_plan_path must select the versioned exploratory "
                "sidecar plan"
            )
        _identity_text_or_placeholder(
            self.exploratory_plan_id,
            name="exploratory exploratory_plan_id",
        )
        if self.exploratory_plan_id != EXPLORATORY_PLAN_ID:
            raise ValueError(
                "exploratory_plan_id must select the versioned synthetic "
                "exploratory sidecar"
            )
        _identity_sha256_or_placeholder(
            self.exploratory_plan_sha256,
            name="exploratory exploratory_plan_sha256",
        )
        if not _SHA256.fullmatch(self.exploratory_plan_sha256):
            raise ValueError(
                "exploratory exploratory_plan_sha256 must be a resolved "
                "lowercase SHA-256"
            )
        fixed_labels = {
            "artifact_namespace": EXPLORATORY_ARTIFACT_NAMESPACE,
            "execution_kind": EXPLORATORY_EXECUTION_KIND,
            "population_basis": EXPLORATORY_POPULATION_BASIS,
            "estimand_interpretation": EXPLORATORY_ESTIMAND_INTERPRETATION,
            "monetary_amount_semantics": (
                EXPLORATORY_MONETARY_AMOUNT_SEMANTICS
            ),
            "unweighted_output_role": EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE,
            "internal_monetary_unit": EXPLORATORY_INTERNAL_MONETARY_UNIT,
            "raw_internal_unit_output_role": (
                EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE
            ),
        }
        for name, expected in fixed_labels.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"exploratory {name} must be {expected!r}"
                )
        if type(self.allow_synthetic) is not bool or not self.allow_synthetic:
            raise ValueError("exploratory allow_synthetic must be true")
        if (
            type(self.execution_enabled) is not bool
            or not self.execution_enabled
        ):
            raise ValueError("exploratory execution_enabled must be true")
        required_false = (
            "campaign_ready",
            "production_campaign",
            "empirical_claims",
            "population_inference_claims",
            "causal_claims",
            "generalisation_claims",
        )
        for name in required_false:
            if type(getattr(self, name)) is not bool or getattr(self, name):
                raise ValueError(f"exploratory {name} must be false")
        required_true = (
            "identical_pretreatment_cohorts",
            "identical_population_weights_across_scenarios",
        )
        for name in required_true:
            if type(getattr(self, name)) is not bool or not getattr(self, name):
                raise ValueError(f"exploratory {name} must be true")
        _nonempty_text(
            self.primary_estimand_id,
            name="exploratory primary_estimand_id",
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "exploratory_plan_path": str(self.exploratory_plan_path),
            "exploratory_plan_id": self.exploratory_plan_id,
            "exploratory_plan_sha256": self.exploratory_plan_sha256,
            "artifact_namespace": self.artifact_namespace,
            "execution_kind": self.execution_kind,
            "population_basis": self.population_basis,
            "estimand_interpretation": self.estimand_interpretation,
            "monetary_amount_semantics": self.monetary_amount_semantics,
            "unweighted_output_role": self.unweighted_output_role,
            "internal_monetary_unit": self.internal_monetary_unit,
            "raw_internal_unit_output_role": (
                self.raw_internal_unit_output_role
            ),
            "execution_enabled": self.execution_enabled,
            "allow_synthetic": self.allow_synthetic,
            "campaign_ready": self.campaign_ready,
            "production_campaign": self.production_campaign,
            "empirical_claims": self.empirical_claims,
            "population_inference_claims": self.population_inference_claims,
            "causal_claims": self.causal_claims,
            "generalisation_claims": self.generalisation_claims,
            "identical_pretreatment_cohorts": (
                self.identical_pretreatment_cohorts
            ),
            "identical_population_weights_across_scenarios": (
                self.identical_population_weights_across_scenarios
            ),
            "primary_estimand_id": self.primary_estimand_id,
        }


@dataclass(frozen=True, slots=True)
class ExploratoryCheckpointConfig:
    """Atomic resumable intermediate-output policy for exploration."""

    enabled: bool
    interval_seeds: int
    directory: Path
    atomic_writes: bool
    preserve_prior_attempts: bool
    resume_mode: str
    partial_result_profile: str

    def __post_init__(self) -> None:
        for name in ("enabled", "atomic_writes", "preserve_prior_attempts"):
            if type(getattr(self, name)) is not bool or not getattr(self, name):
                raise ValueError(
                    f"exploratory_checkpoint {name} must be true"
                )
        if type(self.interval_seeds) is not int or isinstance(
            self.interval_seeds, bool
        ) or self.interval_seeds != 1:
            raise ValueError(
                "exploratory_checkpoint interval_seeds must be 1"
            )
        if not isinstance(self.directory, Path) or not str(self.directory):
            raise TypeError("exploratory_checkpoint directory must be a Path")
        if self.resume_mode != "RESUME_EXACT_COMPATIBLE_ATTEMPT_ONLY":
            raise ValueError(
                "exploratory_checkpoint resume_mode must require exact "
                "compatible-attempt resume"
            )
        if (
            self.partial_result_profile
            != "ATTESTED_INTERNAL_MODEL_STATE_NOT_INTERPRETABLE_AS_MONETARY_OR_ESTIMAND_OUTPUT"
        ):
            raise ValueError(
                "exploratory_checkpoint partial_result_profile must describe "
                "attested internal state while prohibiting monetary and "
                "estimand interpretation"
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "interval_seeds": self.interval_seeds,
            "directory": str(self.directory),
            "atomic_writes": self.atomic_writes,
            "preserve_prior_attempts": self.preserve_prior_attempts,
            "resume_mode": self.resume_mode,
            "partial_result_profile": self.partial_result_profile,
        }


@dataclass(frozen=True, slots=True)
class ExploratoryExecutionEngineConfig:
    """Explicit optimized execution contract, separate from scientific inputs."""

    implementation_id: str
    run_id: str
    attempt_id: str
    supersedes_attempt_id: str
    previous_attempt_lineage_path: Path
    backend: str
    gpu_device_index: int
    gpu_batch_size: int
    gpu_max_batch_bytes: int
    gpu_memory_fraction: float
    precision_mode: str
    host_executor: str
    host_workers: int
    max_in_flight_units: int
    memory_limit_mb: int
    estimated_worker_memory_mb: int
    native_threads_per_worker: int
    scheduling_policy: str
    resume_enabled: bool
    checkpoint_schema_version: str
    main_checkpoint_granularity: str
    sensitivity_checkpoint_granularity: str
    progress_schema_version: str

    def __post_init__(self) -> None:
        for name in ("implementation_id", "run_id"):
            _nonempty_text(getattr(self, name), name=f"execution_engine {name}")
        for name in ("attempt_id", "supersedes_attempt_id"):
            value = getattr(self, name)
            if type(value) is not str or not _ATTEMPT_ID.fullmatch(value):
                raise ValueError(
                    f"execution_engine {name} must use attempt-NNNNNN"
                )
        if self.attempt_id == self.supersedes_attempt_id:
            raise ValueError(
                "execution_engine attempt_id must not reuse the superseded attempt"
            )
        if not isinstance(self.previous_attempt_lineage_path, Path) or not str(
            self.previous_attempt_lineage_path
        ):
            raise TypeError(
                "execution_engine previous_attempt_lineage_path must be a Path"
            )
        if self.backend not in {"cpu", "gpu", "auto"}:
            raise ValueError("execution_engine backend must be cpu, gpu, or auto")
        for name, minimum in (
            ("gpu_device_index", 0),
            ("gpu_batch_size", 1),
            ("gpu_max_batch_bytes", 1),
            ("host_workers", 1),
            ("max_in_flight_units", 1),
            ("memory_limit_mb", 1),
            ("estimated_worker_memory_mb", 1),
            ("native_threads_per_worker", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(
                    f"execution_engine {name} must be an integer >= {minimum}"
                )
        if self.host_workers > 32:
            raise ValueError("execution_engine host_workers must be at most 32")
        if self.native_threads_per_worker != 1:
            raise ValueError(
                "execution_engine native_threads_per_worker must be 1"
            )
        if self.max_in_flight_units > self.host_workers:
            raise ValueError(
                "execution_engine max_in_flight_units cannot exceed host_workers"
            )
        if (
            self.estimated_worker_memory_mb * self.max_in_flight_units
            > self.memory_limit_mb
        ):
            raise ValueError(
                "execution_engine declared in-flight work exceeds its memory limit"
            )
        if isinstance(self.gpu_memory_fraction, bool) or not isinstance(
            self.gpu_memory_fraction, (int, float)
        ):
            raise TypeError(
                "execution_engine gpu_memory_fraction must be numeric"
            )
        fraction = float(self.gpu_memory_fraction)
        if not isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError(
                "execution_engine gpu_memory_fraction must be in (0, 1]"
            )
        object.__setattr__(self, "gpu_memory_fraction", fraction)
        if self.precision_mode != "FLOAT64_STRICT_INTEGER_EXACT":
            raise ValueError(
                "execution_engine precision_mode must remain "
                "FLOAT64_STRICT_INTEGER_EXACT"
            )
        if self.host_executor != "BOUNDED_PROCESS_POOL_SPAWN":
            raise ValueError(
                "execution_engine host_executor must use the bounded "
                "spawn process-pool contract"
            )
        if (
            self.scheduling_policy
            != "ONE_SEED_OWNS_COMMON_COHORT_AND_ALL_SCENARIOS"
        ):
            raise ValueError(
                "execution_engine scheduling_policy must preserve one common "
                "cohort per seed"
            )
        if type(self.resume_enabled) is not bool or not self.resume_enabled:
            raise ValueError("execution_engine resume_enabled must be true")
        expected = {
            "checkpoint_schema_version": "microtx_sim.resumable_checkpoint.v2",
            "main_checkpoint_granularity": "COMPLETE_SEED_ALL_SCENARIOS",
            "sensitivity_checkpoint_granularity": "PARAMETER_LEVEL_SEED",
            "progress_schema_version": "microtx_sim.execution_progress.v2",
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"execution_engine {name} must be {value!r}"
                )

    def snapshot(self) -> dict[str, object]:
        return {
            name: (
                str(getattr(self, name))
                if name == "previous_attempt_lineage_path"
                else getattr(self, name)
            )
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CampaignUncertaintyConfig:
    """Bindings for fixed-seed and joint uncertainty execution."""

    seed_design: str
    minimum_retained_seeds: int
    common_random_numbers: bool
    identical_pretreatment_cohorts: bool
    population_weights_within_seed: bool
    outcome_dependent_seed_exclusion: bool
    parameter_design_path: Path
    parameter_design_id: str
    parameter_design_sha256: str
    parameter_uncertainty: UncertaintyAvailability
    monetary_rate_uncertainty: UncertaintyAvailability
    population_uncertainty: UncertaintyAvailability
    combined_uncertainty_required: bool
    oat_role: str
    variance_decomposition_method: str

    def __post_init__(self) -> None:
        if self.seed_design != "FIXED_ASCENDING":
            raise ValueError("campaign seed_design must be FIXED_ASCENDING")
        _positive_integer(
            self.minimum_retained_seeds,
            name="uncertainty minimum_retained_seeds",
        )
        if self.minimum_retained_seeds < 100:
            raise ValueError(
                "campaign uncertainty requires at least 100 retained seeds"
            )
        required_true = (
            "common_random_numbers",
            "identical_pretreatment_cohorts",
            "population_weights_within_seed",
            "combined_uncertainty_required",
        )
        for name in required_true:
            if type(getattr(self, name)) is not bool or not getattr(self, name):
                raise ValueError(f"campaign uncertainty requires {name} = true")
        if type(self.outcome_dependent_seed_exclusion) is not bool:
            raise TypeError("outcome_dependent_seed_exclusion must be boolean")
        if self.outcome_dependent_seed_exclusion:
            raise ValueError("outcome-dependent seed exclusion is prohibited")
        if not isinstance(self.parameter_design_path, Path) or not str(
            self.parameter_design_path
        ):
            raise TypeError("parameter_design_path must be a Path")
        _identity_text_or_placeholder(
            self.parameter_design_id,
            name="uncertainty parameter_design_id",
        )
        _identity_sha256_or_placeholder(
            self.parameter_design_sha256,
            name="uncertainty parameter_design_sha256",
        )
        for name in (
            "parameter_uncertainty",
            "monetary_rate_uncertainty",
            "population_uncertainty",
        ):
            if type(getattr(self, name)) is not UncertaintyAvailability:
                raise TypeError(f"uncertainty {name} availability is invalid")
        if self.oat_role != "DIAGNOSTIC_ONLY":
            raise ValueError("OAT sensitivity must be DIAGNOSTIC_ONLY")
        if (
            self.variance_decomposition_method
            != "ORTHOGONAL_FINITE_FULL_FACTORIAL_ANOVA_SUM_OF_SQUARES_DIVIDED_BY_N_V1"
        ):
            raise ValueError("unsupported variance decomposition method")


@dataclass(frozen=True, slots=True)
class CampaignConvergenceConfig:
    """Deterministic blockwise convergence and invalid-run thresholds."""

    block_size: int
    minimum_retained_seeds: int
    maximum_mcse: float
    maximum_interval_width: float
    maximum_absolute_change: float
    maximum_relative_change: float
    maximum_invalid_rate: float
    consecutive_passing_checkpoints: int
    sensitivity_instability_allowed: bool
    required_status: str

    def __post_init__(self) -> None:
        for name in (
            "block_size",
            "minimum_retained_seeds",
            "consecutive_passing_checkpoints",
        ):
            _positive_integer(getattr(self, name), name=f"convergence {name}")
        if self.minimum_retained_seeds < 100:
            raise ValueError(
                "campaign convergence requires at least 100 retained seeds"
            )
        for name in (
            "maximum_mcse",
            "maximum_interval_width",
            "maximum_absolute_change",
            "maximum_relative_change",
        ):
            _positive_finite(getattr(self, name), name=f"convergence {name}")
        _finite_number(
            self.maximum_invalid_rate,
            name="convergence maximum_invalid_rate",
        )
        if not 0.0 <= self.maximum_invalid_rate < 1.0:
            raise ValueError("maximum_invalid_rate must be in [0, 1)")
        if type(self.sensitivity_instability_allowed) is not bool:
            raise TypeError("sensitivity_instability_allowed must be boolean")
        if self.sensitivity_instability_allowed:
            raise ValueError("campaign convergence cannot allow instability")
        if self.required_status != "CONVERGED":
            raise ValueError("campaign convergence required_status must be CONVERGED")


@dataclass(frozen=True, slots=True)
class PopulationContractConfig:
    """Exact projected-population identities and application policies."""

    design_id: str
    design_schema_version: str
    design_sha256: str
    runtime_mapping_id: str
    runtime_mapping_schema_version: str
    runtime_mapping_sha256: str
    adapter_id: str
    adapter_schema_version: str
    adapter_sha256: str
    execution_input_schema_version: str
    execution_input_sha256: str
    assignment_schema_version: str
    balance_schema_version: str
    lineage_schema_version: str
    require_per_seed_execution_identity: bool
    require_per_seed_assignment_identity: bool
    require_per_seed_balance_identity: bool
    require_per_seed_lineage_identity: bool
    apportionment_method: str
    weight_application: str
    identical_weights_across_scenarios: bool
    empirical_validation_status: str
    uncertainty_status: UncertaintyAvailability
    uncertainty_design_id: str

    def __post_init__(self) -> None:
        for name in (
            "design_id",
            "runtime_mapping_id",
            "adapter_id",
            "uncertainty_design_id",
        ):
            _identity_text_or_placeholder(
                getattr(self, name),
                name=f"population contract {name}",
            )
        for name in (
            "design_schema_version",
            "runtime_mapping_schema_version",
            "adapter_schema_version",
            "execution_input_schema_version",
            "assignment_schema_version",
            "balance_schema_version",
            "lineage_schema_version",
        ):
            _nonempty_text(getattr(self, name), name=f"population contract {name}")
        for name in (
            "design_sha256",
            "runtime_mapping_sha256",
            "adapter_sha256",
            "execution_input_sha256",
        ):
            _identity_sha256_or_placeholder(
                getattr(self, name),
                name=f"population contract {name}",
            )
        for name in (
            "require_per_seed_execution_identity",
            "require_per_seed_assignment_identity",
            "require_per_seed_balance_identity",
            "require_per_seed_lineage_identity",
        ):
            if type(getattr(self, name)) is not bool or not getattr(self, name):
                raise ValueError(f"population contract requires {name} = true")
        if self.apportionment_method != "exact_rational_hamilton/1":
            raise ValueError("unsupported population apportionment_method")
        if self.weight_application != "WITHIN_SEED_BEFORE_CROSS_SEED_AGGREGATION":
            raise ValueError("population weights must be applied within each seed")
        if type(self.identical_weights_across_scenarios) is not bool or not (
            self.identical_weights_across_scenarios
        ):
            raise ValueError(
                "population weights must be identical across paired scenarios"
            )
        if self.empirical_validation_status not in {"VALIDATED", "UNAVAILABLE"}:
            raise ValueError("invalid empirical population validation status")
        if type(self.uncertainty_status) is not UncertaintyAvailability:
            raise TypeError("population uncertainty_status is invalid")


@dataclass(frozen=True, slots=True)
class MonetaryContractConfig:
    """Production monetary basis, exact inputs, and uncertainty limitation."""

    profile_path: Path
    source_bundle_path: Path
    source_artifact_path: Path
    conversion_table_path: Path
    bundle_id: str
    source_bundle_sha256: str
    source_artifact_sha256: str
    conversion_table_sha256: str
    conversion_basis_id: str
    conversion_basis_sha256: str
    rate_evidence_sha256: str
    target_currency: str
    target_minor_unit_name: str
    quote_convention: str
    scale_convention: str
    rate_period_start: str
    rate_period_end: str
    target_price_period_start: str
    target_price_period_end: str
    missing_date_policy: str
    identity_missing_date_policy: str
    rounding_method: str
    rounding_scope: str
    point_rate_status: str
    rate_uncertainty_status: UncertaintyAvailability
    source_bundle_signature_status: str
    simulation_bridge_status: str
    observed_real_world_spending: bool
    raw_cross_currency_pooling: str

    def __post_init__(self) -> None:
        for name in (
            "profile_path",
            "source_bundle_path",
            "source_artifact_path",
            "conversion_table_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not str(value):
                raise TypeError(f"monetary {name} must be a Path")
        for name in ("bundle_id", "conversion_basis_id"):
            _identity_text_or_placeholder(
                getattr(self, name), name=f"monetary {name}"
            )
        for name in (
            "source_bundle_sha256",
            "source_artifact_sha256",
            "conversion_table_sha256",
            "conversion_basis_sha256",
            "rate_evidence_sha256",
        ):
            _identity_sha256_or_placeholder(
                getattr(self, name), name=f"monetary {name}"
            )
        if self.target_currency != "EUR":
            raise ValueError("current monetary production contract targets EUR")
        if self.target_minor_unit_name != "euro cent":
            raise ValueError(
                "current monetary production contract uses the euro cent"
            )
        for name in (
            "quote_convention",
            "scale_convention",
            "rate_period_start",
            "rate_period_end",
            "target_price_period_start",
            "target_price_period_end",
            "missing_date_policy",
            "identity_missing_date_policy",
            "rounding_method",
            "rounding_scope",
        ):
            _nonempty_text(getattr(self, name), name=f"monetary {name}")
        if self.point_rate_status != "OFFICIAL_POINT_OBSERVATION":
            raise ValueError("monetary point_rate_status is invalid")
        if type(self.rate_uncertainty_status) is not UncertaintyAvailability:
            raise TypeError("monetary rate_uncertainty_status is invalid")
        if self.source_bundle_signature_status not in {"VERIFIED", "MISSING"}:
            raise ValueError("monetary source bundle signature status is invalid")
        if self.simulation_bridge_status != "ILLUSTRATIVE":
            raise ValueError(
                "the current simulation-cent monetary bridge is ILLUSTRATIVE"
            )
        if type(self.observed_real_world_spending) is not bool:
            raise TypeError("observed_real_world_spending must be boolean")
        if self.observed_real_world_spending:
            raise ValueError("model monetary values are not observed spending")
        if self.raw_cross_currency_pooling != "REJECT":
            raise ValueError("raw cross-currency pooling must be rejected")


@dataclass(frozen=True, slots=True)
class CampaignOutputContractConfig:
    """Metric-registry and complete production-shaped output selection."""

    metric_registry_schema_version: str
    metric_registry_sha256: str
    output_schema_version: str
    output_schema_sha256: str
    output_profile_id: str
    output_profile_schema_version: str
    output_profile_sha256: str
    expected_artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "metric_registry_schema_version",
            "output_schema_version",
            "output_profile_schema_version",
        ):
            _nonempty_text(getattr(self, name), name=f"output contract {name}")
        for name in (
            "metric_registry_sha256",
            "output_schema_sha256",
            "output_profile_sha256",
        ):
            _identity_sha256_or_placeholder(
                getattr(self, name), name=f"output contract {name}"
            )
        if self.output_profile_id != "campaign_joint_uncertainty":
            raise ValueError(
                "full campaign output_profile_id must be campaign_joint_uncertainty"
            )
        if (
            type(self.expected_artifacts) is not tuple
            or not self.expected_artifacts
            or any(
                type(value) is not str or not value.strip()
                for value in self.expected_artifacts
            )
        ):
            raise ValueError("expected_artifacts must be a non-empty tuple")
        if len(set(self.expected_artifacts)) != len(self.expected_artifacts):
            raise ValueError("expected_artifacts cannot contain duplicates")
        required = {
            "manifest.json",
            "execution-receipt.json",
            "execution-attestation.json",
            "uncertainty_realizations.csv",
            "uncertainty_summary.json",
            "convergence_checkpoints.csv",
            "pre-campaign-validation-report.json",
        }
        missing = sorted(required - set(self.expected_artifacts))
        if missing:
            raise ValueError(
                f"full campaign expected_artifacts missing required files: {missing}"
            )


@dataclass(frozen=True, slots=True)
class CampaignLedgerConfig:
    """Persistent, caller-owned policy campaign ledger."""

    backend: LedgerBackend
    path: Path
    persistent: bool
    temporary: bool

    def __post_init__(self) -> None:
        if self.backend is not LedgerBackend.SQLITE:
            raise ValueError("full campaigns require a SQLite ledger")
        if not isinstance(self.path, Path) or not str(self.path):
            raise TypeError("ledger path must be a Path")
        if self.path.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
            raise ValueError("campaign ledger path must identify a SQLite file")
        if type(self.persistent) is not bool or not self.persistent:
            raise ValueError("full campaign ledger must be persistent")
        if type(self.temporary) is not bool or self.temporary:
            raise ValueError("full campaign ledger cannot be temporary")


@dataclass(frozen=True, slots=True)
class ExecutionReceiptPolicyConfig:
    """Pre/post technical identity verification required around execution."""

    schema_path: Path
    schema_version: str
    identity_algorithm: str
    receipt_path: Path
    attestation_path: Path
    require_clean_working_tree: bool
    verify_active_commit: bool
    verify_source_tree: bool
    verify_interpreter: bool
    verify_dependencies: bool
    reject_environment_drift: bool
    manifest_reference_required: bool
    run_command: tuple[str, ...]
    execution_mode: str

    def __post_init__(self) -> None:
        for name in (
            "schema_path",
            "receipt_path",
            "attestation_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, Path) or not str(value):
                raise TypeError(f"execution receipt {name} must be a Path")
        _nonempty_text(self.schema_version, name="execution receipt schema_version")
        if (
            self.identity_algorithm
            != "microtx_sim.execution_receipt.canonical_json_utf8.v1"
        ):
            raise ValueError("unsupported execution receipt identity algorithm")
        for name in (
            "require_clean_working_tree",
            "verify_active_commit",
            "verify_source_tree",
            "verify_interpreter",
            "verify_dependencies",
            "reject_environment_drift",
            "manifest_reference_required",
        ):
            if type(getattr(self, name)) is not bool or not getattr(self, name):
                raise ValueError(f"execution receipt requires {name} = true")
        if (
            type(self.run_command) is not tuple
            or not self.run_command
            or any(type(value) is not str or not value for value in self.run_command)
        ):
            raise ValueError("execution receipt run_command must be non-empty")
        if self.execution_mode != "FULL_CAMPAIGN":
            raise ValueError("execution receipt execution_mode must be FULL_CAMPAIGN")


@dataclass(frozen=True, slots=True)
class PolicyPrototypeConfig:
    name: str
    provenance_status: str
    notes: str
    batch: PolicyBatchSpec
    harm_parameters: HarmModelParameters
    harm_weights: WelfareHarmWeights
    opportunity_valuation: OpportunityCostValuation
    producer_assumptions: ProducerAssumptions
    epgc_policy: EPGCPolicy
    output: PolicyOutputConfig
    population: PopulationProjectionConfig | None = None
    analysis_plan: AnalysisPlanSelection | None = None
    run_purpose: PolicyRunPurpose = PolicyRunPurpose.DEVELOPMENT
    full_campaign_config: bool = False
    full_exploratory_config: bool = False
    campaign: CampaignControlConfig | None = None
    exploratory: ExploratoryControlConfig | None = None
    exploratory_checkpoint: ExploratoryCheckpointConfig | None = None
    execution_engine: ExploratoryExecutionEngineConfig | None = None
    uncertainty: CampaignUncertaintyConfig | None = None
    convergence: CampaignConvergenceConfig | None = None
    population_contract: PopulationContractConfig | None = None
    monetary_contract: MonetaryContractConfig | None = None
    output_contract: CampaignOutputContractConfig | None = None
    ledger: CampaignLedgerConfig | None = None
    execution_receipt: ExecutionReceiptPolicyConfig | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("configuration name cannot be empty")
        if type(self.run_purpose) is not PolicyRunPurpose:
            raise TypeError("run_purpose must be PolicyRunPurpose")
        if type(self.full_campaign_config) is not bool:
            raise TypeError("full_campaign_config must be boolean")
        if type(self.full_exploratory_config) is not bool:
            raise TypeError("full_exploratory_config must be boolean")
        if self.full_campaign_config and self.full_exploratory_config:
            raise ValueError(
                "full_campaign_config and full_exploratory_config are "
                "mutually exclusive"
            )
        allowed_provenance = {
            "synthetic",
            "illustrative",
            "anchored",
            "calibrated",
        }
        if self.provenance_status not in allowed_provenance:
            raise ValueError("policy provenance_status is unsupported")
        if (
            self.run_purpose is PolicyRunPurpose.DEVELOPMENT
            and self.provenance_status != "synthetic"
        ):
            raise ValueError(
                "development policy runs require provenance_status = 'synthetic'"
            )
        if self.population is not None and type(
            self.population
        ) is not PopulationProjectionConfig:
            raise TypeError(
                "population must be PopulationProjectionConfig or None"
            )
        if self.analysis_plan is not None and type(
            self.analysis_plan
        ) is not AnalysisPlanSelection:
            raise TypeError(
                "analysis_plan must be AnalysisPlanSelection or None"
            )
        if self.exploratory is not None and type(
            self.exploratory
        ) is not ExploratoryControlConfig:
            raise TypeError(
                "exploratory must be ExploratoryControlConfig or None"
            )
        if self.exploratory_checkpoint is not None and type(
            self.exploratory_checkpoint
        ) is not ExploratoryCheckpointConfig:
            raise TypeError(
                "exploratory_checkpoint must be "
                "ExploratoryCheckpointConfig or None"
            )
        if self.execution_engine is not None and type(
            self.execution_engine
        ) is not ExploratoryExecutionEngineConfig:
            raise TypeError(
                "execution_engine must be "
                "ExploratoryExecutionEngineConfig or None"
            )
        if self.run_purpose is PolicyRunPurpose.CAMPAIGN:
            if (
                self.exploratory is not None
                or self.exploratory_checkpoint is not None
                or self.execution_engine is not None
                or self.full_exploratory_config
            ):
                raise ValueError(
                    "campaign policy runs cannot declare exploratory semantics"
                )
            if self.population is None:
                raise ValueError(
                    "campaign policy runs require [population] with "
                    "mode = 'projected_v1'"
                )
            if self.population.mode.value != "projected_v1":
                raise ValueError(
                    "campaign policy runs require population.mode = 'projected_v1'"
                )
            if self.analysis_plan is None:
                raise ValueError(
                    "campaign policy runs require an [analysis_plan] selection"
                )
            if self.batch.player_count <= 0:
                raise ValueError(
                    "campaign policy runs require a positive player cohort"
                )
            if not self.output.include_player_rows:
                raise ValueError(
                    "campaign policy runs require output.include_player_rows = true"
                )
        if self.run_purpose is PolicyRunPurpose.EXPLORATORY:
            if not self.full_exploratory_config:
                raise ValueError(
                    "exploratory policy runs require "
                    "meta.full_exploratory_config = true"
                )
            if self.exploratory is None:
                raise ValueError(
                    "exploratory policy runs require an [exploratory] table"
                )
        elif (
            self.full_exploratory_config
            or self.exploratory is not None
            or self.exploratory_checkpoint is not None
            or self.execution_engine is not None
        ):
            raise ValueError(
                "exploratory semantics require run_purpose = 'exploratory'"
            )
        if self.analysis_plan is not None and self.population is None:
            raise ValueError(
                "analysis_plan requires projected population execution"
            )
        if (
            self.analysis_plan is not None
            and not self.output.include_player_rows
        ):
            raise ValueError(
                "analysis_plan requires output.include_player_rows = true "
                "because schema-v1 planned metrics bind to player_outcomes.csv"
            )
        campaign_sections = {
            "campaign": (self.campaign, CampaignControlConfig),
            "uncertainty": (self.uncertainty, CampaignUncertaintyConfig),
            "convergence": (self.convergence, CampaignConvergenceConfig),
            "population_contract": (
                self.population_contract,
                PopulationContractConfig,
            ),
            "monetary_contract": (
                self.monetary_contract,
                MonetaryContractConfig,
            ),
            "output_contract": (
                self.output_contract,
                CampaignOutputContractConfig,
            ),
            "ledger": (self.ledger, CampaignLedgerConfig),
            "execution_receipt": (
                self.execution_receipt,
                ExecutionReceiptPolicyConfig,
            ),
        }
        for name, (value, expected_type) in campaign_sections.items():
            if value is not None and type(value) is not expected_type:
                raise TypeError(f"{name} must be {expected_type.__name__} or None")
        if self.full_campaign_config:
            if self.run_purpose is not PolicyRunPurpose.CAMPAIGN:
                raise ValueError(
                    "meta.full_campaign_config = true requires "
                    "run_purpose = 'campaign'"
                )
            missing = sorted(
                name for name, (value, _) in campaign_sections.items() if value is None
            )
            if missing:
                raise ValueError(
                    "full campaign configuration is missing required sections: "
                    + ", ".join(missing)
                )
            assert self.campaign is not None
            assert self.uncertainty is not None
            assert self.convergence is not None
            assert self.population_contract is not None
            assert self.monetary_contract is not None
            assert self.analysis_plan is not None
            if len(self.batch.seeds) < self.uncertainty.minimum_retained_seeds:
                raise ValueError(
                    "full campaign fixed seed set is smaller than the declared "
                    "minimum_retained_seeds"
                )
            if (
                self.convergence.minimum_retained_seeds
                != self.uncertainty.minimum_retained_seeds
            ):
                raise ValueError(
                    "uncertainty and convergence minimum_retained_seeds must match"
                )
            if self.population is None:
                raise ValueError("full campaign requires projected population")
            if self.population.adapter_id != self.population_contract.adapter_id:
                raise ValueError(
                    "population adapter_id conflicts with population_contract"
                )
            if self.analysis_plan.expected_plan_id is None:
                raise ValueError(
                    "full campaign analysis_plan requires expected and parent "
                    "identities"
                )
            required_unavailable = any(
                value is not UncertaintyAvailability.QUANTIFIED
                for value in (
                    self.uncertainty.parameter_uncertainty,
                    self.uncertainty.monetary_rate_uncertainty,
                    self.uncertainty.population_uncertainty,
                )
            )
            has_placeholder = _contains_blocked_placeholder(
                self.analysis_plan.snapshot()
            ) or _contains_blocked_placeholder(
                {
                    "parameter_design_sha256": (
                        self.uncertainty.parameter_design_sha256
                    ),
                    "population": self.population_contract,
                    "monetary": self.monetary_contract,
                    "output": self.output_contract,
                }
            )
            authenticated_rate_source = (
                self.monetary_contract.source_bundle_signature_status == "VERIFIED"
            )
            empirically_validated_population = (
                self.population_contract.empirical_validation_status == "VALIDATED"
            )
            if (
                required_unavailable
                or has_placeholder
                or not authenticated_rate_source
                or not empirically_validated_population
            ) and self.campaign.campaign_ready:
                raise ValueError(
                    "campaign_ready must remain false while required identities, "
                    "uncertainty, authentication, or population validation are "
                    "unavailable"
                )
        if self.full_exploratory_config:
            _validate_full_exploratory_config(self)


def _validate_full_exploratory_config(config: PolicyPrototypeConfig) -> None:
    """Enforce a complete, isolated, explicitly non-inferential run contract."""

    if config.run_purpose is not PolicyRunPurpose.EXPLORATORY:
        raise ValueError(
            "meta.full_exploratory_config = true requires "
            "run_purpose = 'exploratory'"
        )
    if config.provenance_status not in {"synthetic", "illustrative"}:
        raise ValueError(
            "exploratory policy runs require synthetic or illustrative provenance"
        )
    if config.campaign is not None:
        raise ValueError("full exploratory configuration forbids [campaign]")
    if config.output_contract is not None:
        raise ValueError(
            "full exploratory configuration forbids production [output_contract]"
        )
    if config.execution_receipt is not None:
        raise ValueError(
            "full exploratory configuration forbids production [execution_receipt]"
        )
    required = {
        "exploratory": config.exploratory,
        "exploratory_checkpoint": config.exploratory_checkpoint,
        "execution_engine": config.execution_engine,
        "population": config.population,
        "analysis_plan": config.analysis_plan,
        "uncertainty": config.uncertainty,
        "convergence": config.convergence,
        "population_contract": config.population_contract,
        "monetary_contract": config.monetary_contract,
        "ledger": config.ledger,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError(
            "full exploratory configuration is missing required sections: "
            + ", ".join(missing)
        )
    exploratory = config.exploratory
    checkpoint = config.exploratory_checkpoint
    execution_engine = config.execution_engine
    population = config.population
    analysis = config.analysis_plan
    uncertainty = config.uncertainty
    convergence = config.convergence
    population_contract = config.population_contract
    monetary = config.monetary_contract
    ledger = config.ledger
    assert exploratory is not None
    assert checkpoint is not None
    assert execution_engine is not None
    assert population is not None
    assert analysis is not None
    assert uncertainty is not None
    assert convergence is not None
    assert population_contract is not None
    assert monetary is not None
    assert ledger is not None

    if population.mode.value != "projected_v1":
        raise ValueError(
            "full exploratory configuration requires population.mode = "
            "'projected_v1'"
        )
    if population.adapter_id != population_contract.adapter_id:
        raise ValueError(
            "exploratory population adapter_id conflicts with population_contract"
        )
    if population_contract.empirical_validation_status != "UNAVAILABLE":
        raise ValueError(
            "exploratory population must remain explicitly non-empirical"
        )
    if analysis.expected_plan_id is None:
        raise ValueError(
            "full exploratory analysis_plan requires expected and parent identities"
        )
    assert analysis.parent_plan_path is not None
    assert analysis.parent_plan_id is not None
    assert analysis.expected_plan_sha256 is not None
    assert analysis.parent_plan_sha256 is not None
    if not _SHA256.fullmatch(analysis.expected_plan_sha256) or not _SHA256.fullmatch(
        analysis.parent_plan_sha256
    ):
        raise ValueError(
            "full exploratory analysis_plan identities must use resolved "
            "lowercase SHA-256 values"
        )
    if (
        analysis.plan_path.name != EXPLORATORY_PARENT_PLAN_FILENAME
        or analysis.expected_plan_id != EXPLORATORY_PARENT_PLAN_ID
    ):
        raise ValueError(
            "full exploratory configuration must retain the production-v3 "
            "scientific analysis plan binding"
        )
    if (
        analysis.parent_plan_path.name
        != EXPLORATORY_SCIENTIFIC_PARENT_PLAN_FILENAME
        or analysis.parent_plan_id != EXPLORATORY_SCIENTIFIC_PARENT_PLAN_ID
    ):
        raise ValueError(
            "exploratory scientific analysis selection must retain the "
            "production-v3 plan's v2 parent identity"
        )
    if exploratory.primary_estimand_id != EXPLORATORY_PRIMARY_ESTIMAND_ID:
        raise ValueError(
            "exploratory primary_estimand_id must preserve the declared primary"
        )
    if len(config.batch.seeds) < uncertainty.minimum_retained_seeds:
        raise ValueError(
            "exploratory fixed seed set is smaller than minimum_retained_seeds"
        )
    if convergence.minimum_retained_seeds != uncertainty.minimum_retained_seeds:
        raise ValueError(
            "exploratory uncertainty and convergence minimum seed counts must match"
        )
    if not uncertainty.identical_pretreatment_cohorts:
        raise ValueError(
            "exploratory paired scenarios require identical pre-treatment cohorts"
        )
    if not population_contract.identical_weights_across_scenarios:
        raise ValueError(
            "exploratory paired scenarios require identical population weights"
        )
    if not config.output.include_player_rows:
        raise ValueError(
            "exploratory analysis requires output.include_player_rows = true"
        )
    expected_output = Path("artifacts") / exploratory.artifact_namespace
    if config.output.output_dir != expected_output:
        raise ValueError(
            "exploratory output_dir must be isolated at "
            f"{expected_output.as_posix()!r}"
        )
    expected_checkpoint = (
        Path("artifacts") / exploratory.artifact_namespace / "progress"
    )
    if (
        checkpoint.directory.name != "progress"
        or checkpoint.directory.parent.name != exploratory.artifact_namespace
        or checkpoint.directory.parent.parent.name != "artifacts"
    ):
        raise ValueError(
            "exploratory checkpoint directory must be isolated at "
            f"{expected_checkpoint.as_posix()!r}"
        )
    if execution_engine.supersedes_attempt_id != "attempt-000001":
        raise ValueError(
            "exploratory optimized execution must preserve attempt-000001 "
            "as its superseded incomplete lineage"
        )
    if execution_engine.attempt_id != "attempt-000002":
        raise ValueError(
            "exploratory optimized execution must use the new attempt-000002 identity"
        )
    if execution_engine.previous_attempt_lineage_path.name != (
        "policy-exploratory-attempt-000001.json"
    ):
        raise ValueError(
            "exploratory execution must bind the checked-in previous-attempt lineage"
        )
    if (
        ledger.path.parent.name != exploratory.artifact_namespace
        or ledger.path.parent.parent.name != "artifacts"
        or ledger.path.name != "exploratory-ledger.sqlite3"
    ):
        raise ValueError(
            "exploratory ledger path must be the isolated persistent SQLite "
            "artifact at artifacts/policy_exploratory_synthetic/"
            "exploratory-ledger.sqlite3"
        )
    if monetary.observed_real_world_spending:
        raise ValueError(
            "exploratory monetary amounts cannot claim observed spending"
        )


def load_policy_config(path: str | Path) -> PolicyPrototypeConfig:
    """Load a strict, fully typed policy prototype configuration."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        _strict_top_level(raw)
        meta = _section(raw, "meta")
        run = _section(raw, "policy_run")
        decision = _section(raw, "decision")
        harm = _section(raw, "harm")
        weights = _section(raw, "harm_weights")
        valuation = _section(raw, "opportunity_valuation")
        producer = _section(raw, "producer")
        epgc = _section(raw, "epgc")
        output = _section(raw, "output")
        full_campaign_config = _optional_boolean(
            meta.get("full_campaign_config", False),
            name="meta.full_campaign_config",
        )
        full_exploratory_config = _optional_boolean(
            meta.get("full_exploratory_config", False),
            name="meta.full_exploratory_config",
        )
        population = _population_projection_config(
            raw.get("population"),
            config_path=config_path,
        )
        analysis_plan = _analysis_plan_selection(
            raw.get("analysis_plan"),
            config_path=config_path,
        )
        campaign = _campaign_control(raw.get("campaign"))
        exploratory = _exploratory_control(
            raw.get("exploratory"),
            config_path=config_path,
        )
        exploratory_checkpoint = _exploratory_checkpoint_control(
            raw.get("exploratory_checkpoint"),
            config_path=config_path,
        )
        execution_engine = _exploratory_execution_engine(
            raw.get("execution_engine"),
            config_path=config_path,
        )
        uncertainty = _campaign_uncertainty(
            raw.get("uncertainty"),
            config_path=config_path,
        )
        convergence = _campaign_convergence(raw.get("convergence"))
        population_contract = _population_contract(
            raw.get("population_contract")
        )
        monetary_contract = _monetary_contract(
            raw.get("monetary_contract"),
            config_path=config_path,
        )
        output_contract = _campaign_output_contract(raw.get("output_contract"))
        ledger = _campaign_ledger(
            raw.get("ledger"),
            config_path=config_path,
        )
        execution_receipt = _execution_receipt_policy(
            raw.get("execution_receipt"),
            config_path=config_path,
        )
        _required_and_optional_keys(
            meta,
            {"name", "provenance_status", "notes"},
            {
                "run_purpose",
                "full_campaign_config",
                "full_exploratory_config",
            },
            "meta",
        )
        _exact_keys(run, {"seeds", "days", "player_count"}, "policy_run")
        if full_campaign_config:
            _validate_full_campaign_seeds(run["seeds"])
        if full_exploratory_config:
            _validate_full_exploratory_seeds(run["seeds"])
        _exact_keys(
            decision,
            {
                "step_minutes",
                "temperature",
                "habit_persistence",
                "habit_learning_rate",
                "reinforcement_learning_rate",
            },
            "decision",
        )
        _exact_keys(
            harm,
            {
                "affordable_spending_share",
                "opaque_spending_weight",
                "random_reward_spending_weight",
                "time_pressure_spending_weight",
                "sleep_debt_weight",
            },
            "harm",
        )
        _exact_keys(
            weights,
            {
                "monetary",
                "opportunity_cost",
                "sleep",
                "education_work",
                "family_social",
                "wellbeing",
            },
            "harm_weights",
        )
        _exact_keys(
            valuation,
            {
                "adult_sleep_hour_cents",
                "adult_work_study_hour_cents",
                "adult_social_hour_cents",
                "adult_physical_activity_hour_cents",
                "youth_sleep_hour_cents",
                "youth_education_hour_cents",
                "youth_family_social_hour_cents",
                "youth_physical_activity_hour_cents",
            },
            "opportunity_valuation",
        )
        _exact_keys(
            producer,
            {
                "development_cost_cents",
                "maintenance_cost_cents_per_day",
                "institutional_license_count",
                "institutional_license_price_cents",
                "non_targeted_sponsorship_revenue_cents",
                "accessibility_eligible",
                "multilingual_support_eligible",
                "cultural_value_eligible",
                "safety_certified",
            },
            "producer",
        )
        _exact_keys(
            epgc,
            {
                "access_payment_cents_per_eligible_access",
                "institutional_license_payment_cents_per_license",
                "availability_payment_cents_per_period",
                "accessibility_bonus_cents",
                "multilingual_bonus_cents",
                "cultural_value_bonus_cents",
                "safety_certification_bonus_cents",
                "prohibited_mechanics_penalty_cents",
                "prohibited_mechanics_clawback_basis_points",
                "maximum_budget_cents",
            },
            "epgc",
        )
        _exact_keys(
            output,
            {
                "output_dir",
                "histogram_bins",
                "include_player_rows",
                "run_sensitivity",
            },
            "output",
        )
        decision_parameters = DecisionParameters(**decision)
        batch = PolicyBatchSpec(
            seeds=tuple(run["seeds"]),
            days=run["days"],
            player_count=run["player_count"],
            decision_parameters=decision_parameters,
        )
        return PolicyPrototypeConfig(
            name=str(meta["name"]),
            provenance_status=str(meta["provenance_status"]),
            notes=str(meta["notes"]),
            batch=batch,
            harm_parameters=HarmModelParameters(**harm),
            harm_weights=WelfareHarmWeights(**weights),
            opportunity_valuation=OpportunityCostValuation(**valuation),
            producer_assumptions=ProducerAssumptions(**producer),
            epgc_policy=EPGCPolicy(**epgc),
            output=PolicyOutputConfig(
                output_dir=Path(output["output_dir"]),
                histogram_bins=output["histogram_bins"],
                include_player_rows=output["include_player_rows"],
                run_sensitivity=output["run_sensitivity"],
            ),
            population=population,
            analysis_plan=analysis_plan,
            run_purpose=_policy_run_purpose(meta.get("run_purpose")),
            full_campaign_config=full_campaign_config,
            full_exploratory_config=full_exploratory_config,
            campaign=campaign,
            exploratory=exploratory,
            exploratory_checkpoint=exploratory_checkpoint,
            execution_engine=execution_engine,
            uncertainty=uncertainty,
            convergence=convergence,
            population_contract=population_contract,
            monetary_contract=monetary_contract,
            output_contract=output_contract,
            ledger=ledger,
            execution_receipt=execution_receipt,
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        if isinstance(exc, PolicyConfigurationError):
            raise
        raise PolicyConfigurationError(
            f"invalid policy configuration {config_path}: {exc}"
        ) from exc


def _strict_top_level(raw: Mapping[str, object]) -> None:
    expected = {
        "meta",
        "policy_run",
        "decision",
        "harm",
        "harm_weights",
        "opportunity_valuation",
        "producer",
        "epgc",
        "output",
    }
    actual = set(raw)
    if "population" in actual:
        actual.remove("population")
    if "analysis_plan" in actual:
        actual.remove("analysis_plan")
    for optional in (
        "campaign",
        "exploratory",
        "exploratory_checkpoint",
        "execution_engine",
        "uncertainty",
        "convergence",
        "population_contract",
        "monetary_contract",
        "output_contract",
        "ledger",
        "execution_receipt",
    ):
        actual.discard(optional)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise PolicyConfigurationError(
            f"top level keys differ: missing={missing}, unknown={unknown}"
        )


def _section(raw: Mapping[str, object], name: str) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise PolicyConfigurationError(f"[{name}] must be a TOML table")
    return value


def _analysis_plan_selection(
    value: object,
    *,
    config_path: Path,
) -> AnalysisPlanSelection | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("[analysis_plan] must be a TOML table")
    _required_and_optional_keys(
        value,
        {"plan_path"},
        {
            "expected_plan_id",
            "expected_plan_sha256",
            "parent_plan_path",
            "parent_plan_id",
            "parent_plan_sha256",
        },
        "analysis_plan",
    )
    identity_fields = {
        "expected_plan_id",
        "expected_plan_sha256",
        "parent_plan_path",
        "parent_plan_id",
        "parent_plan_sha256",
    }
    present = identity_fields.intersection(value)
    if present and present != identity_fields:
        raise ValueError(
            "analysis plan expected and parent identities must be supplied together"
        )
    raw_path = value["plan_path"]
    if type(raw_path) is not str or not raw_path:
        raise ValueError("analysis plan_path must be non-empty text")
    selected = _resolved_config_path(
        raw_path,
        config_path=config_path,
        name="analysis plan_path",
    )
    parent_path = (
        _resolved_config_path(
            value["parent_plan_path"],
            config_path=config_path,
            name="analysis parent_plan_path",
        )
        if identity_fields
        and "parent_plan_path" in value
        else None
    )
    return AnalysisPlanSelection(
        plan_path=selected,
        expected_plan_id=value.get("expected_plan_id"),
        expected_plan_sha256=value.get("expected_plan_sha256"),
        parent_plan_path=parent_path,
        parent_plan_id=value.get("parent_plan_id"),
        parent_plan_sha256=value.get("parent_plan_sha256"),
    )


def _campaign_control(value: object) -> CampaignControlConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="campaign")
    _exact_keys(
        values,
        {
            "allow_synthetic",
            "fail_closed",
            "simulation_layer",
            "campaign_ready",
            "primary_estimand_id",
        },
        "campaign",
    )
    try:
        layer = PolicySimulationLayer(values["simulation_layer"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "campaign simulation_layer must be 'policy_orchestrator'"
        ) from exc
    return CampaignControlConfig(
        allow_synthetic=values["allow_synthetic"],
        fail_closed=values["fail_closed"],
        simulation_layer=layer,
        campaign_ready=values["campaign_ready"],
        primary_estimand_id=values["primary_estimand_id"],
    )


def _exploratory_control(
    value: object,
    *,
    config_path: Path,
) -> ExploratoryControlConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="exploratory")
    expected = {
        "exploratory_plan_path",
        "exploratory_plan_id",
        "exploratory_plan_sha256",
        "artifact_namespace",
        "execution_kind",
        "population_basis",
        "estimand_interpretation",
        "monetary_amount_semantics",
        "unweighted_output_role",
        "internal_monetary_unit",
        "raw_internal_unit_output_role",
        "execution_enabled",
        "allow_synthetic",
        "campaign_ready",
        "production_campaign",
        "empirical_claims",
        "population_inference_claims",
        "causal_claims",
        "generalisation_claims",
        "identical_pretreatment_cohorts",
        "identical_population_weights_across_scenarios",
        "primary_estimand_id",
    }
    _exact_keys(values, expected, "exploratory")
    return ExploratoryControlConfig(
        **{
            **values,
            "exploratory_plan_path": _resolved_config_path(
                values["exploratory_plan_path"],
                config_path=config_path,
                name="exploratory exploratory_plan_path",
            ),
        }
    )


def _exploratory_checkpoint_control(
    value: object,
    *,
    config_path: Path,
) -> ExploratoryCheckpointConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="exploratory_checkpoint")
    _exact_keys(
        values,
        {
            "enabled",
            "interval_seeds",
            "directory",
            "atomic_writes",
            "preserve_prior_attempts",
            "resume_mode",
            "partial_result_profile",
        },
        "exploratory_checkpoint",
    )
    return ExploratoryCheckpointConfig(
        **{
            **values,
            "directory": _resolved_config_path(
                values["directory"],
                config_path=config_path,
                name="exploratory_checkpoint directory",
            ),
        }
    )


def _exploratory_execution_engine(
    value: object,
    *,
    config_path: Path,
) -> ExploratoryExecutionEngineConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="execution_engine")
    expected = {
        "implementation_id",
        "run_id",
        "attempt_id",
        "supersedes_attempt_id",
        "previous_attempt_lineage_path",
        "backend",
        "gpu_device_index",
        "gpu_batch_size",
        "gpu_max_batch_bytes",
        "gpu_memory_fraction",
        "precision_mode",
        "host_executor",
        "host_workers",
        "max_in_flight_units",
        "memory_limit_mb",
        "estimated_worker_memory_mb",
        "native_threads_per_worker",
        "scheduling_policy",
        "resume_enabled",
        "checkpoint_schema_version",
        "main_checkpoint_granularity",
        "sensitivity_checkpoint_granularity",
        "progress_schema_version",
    }
    _exact_keys(values, expected, "execution_engine")
    return ExploratoryExecutionEngineConfig(
        **{
            **values,
            "previous_attempt_lineage_path": _resolved_config_path(
                values["previous_attempt_lineage_path"],
                config_path=config_path,
                name="execution_engine previous_attempt_lineage_path",
            ),
        }
    )


def _campaign_uncertainty(
    value: object,
    *,
    config_path: Path,
) -> CampaignUncertaintyConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="uncertainty")
    _exact_keys(
        values,
        {
            "seed_design",
            "minimum_retained_seeds",
            "common_random_numbers",
            "identical_pretreatment_cohorts",
            "population_weights_within_seed",
            "outcome_dependent_seed_exclusion",
            "parameter_design_path",
            "parameter_design_id",
            "parameter_design_sha256",
            "parameter_uncertainty",
            "monetary_rate_uncertainty",
            "population_uncertainty",
            "combined_uncertainty_required",
            "oat_role",
            "variance_decomposition_method",
        },
        "uncertainty",
    )
    return CampaignUncertaintyConfig(
        seed_design=values["seed_design"],
        minimum_retained_seeds=values["minimum_retained_seeds"],
        common_random_numbers=values["common_random_numbers"],
        identical_pretreatment_cohorts=values[
            "identical_pretreatment_cohorts"
        ],
        population_weights_within_seed=values[
            "population_weights_within_seed"
        ],
        outcome_dependent_seed_exclusion=values[
            "outcome_dependent_seed_exclusion"
        ],
        parameter_design_path=_resolved_config_path(
            values["parameter_design_path"],
            config_path=config_path,
            name="uncertainty parameter_design_path",
        ),
        parameter_design_id=values["parameter_design_id"],
        parameter_design_sha256=values["parameter_design_sha256"],
        parameter_uncertainty=_uncertainty_availability(
            values["parameter_uncertainty"],
            name="uncertainty parameter_uncertainty",
        ),
        monetary_rate_uncertainty=_uncertainty_availability(
            values["monetary_rate_uncertainty"],
            name="uncertainty monetary_rate_uncertainty",
        ),
        population_uncertainty=_uncertainty_availability(
            values["population_uncertainty"],
            name="uncertainty population_uncertainty",
        ),
        combined_uncertainty_required=values["combined_uncertainty_required"],
        oat_role=values["oat_role"],
        variance_decomposition_method=values["variance_decomposition_method"],
    )


def _campaign_convergence(
    value: object,
) -> CampaignConvergenceConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="convergence")
    expected = {
        "block_size",
        "minimum_retained_seeds",
        "maximum_mcse",
        "maximum_interval_width",
        "maximum_absolute_change",
        "maximum_relative_change",
        "maximum_invalid_rate",
        "consecutive_passing_checkpoints",
        "sensitivity_instability_allowed",
        "required_status",
    }
    _exact_keys(values, expected, "convergence")
    return CampaignConvergenceConfig(**values)


def _population_contract(value: object) -> PopulationContractConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="population_contract")
    expected = {
        "design_id",
        "design_schema_version",
        "design_sha256",
        "runtime_mapping_id",
        "runtime_mapping_schema_version",
        "runtime_mapping_sha256",
        "adapter_id",
        "adapter_schema_version",
        "adapter_sha256",
        "execution_input_schema_version",
        "execution_input_sha256",
        "assignment_schema_version",
        "balance_schema_version",
        "lineage_schema_version",
        "require_per_seed_execution_identity",
        "require_per_seed_assignment_identity",
        "require_per_seed_balance_identity",
        "require_per_seed_lineage_identity",
        "apportionment_method",
        "weight_application",
        "identical_weights_across_scenarios",
        "empirical_validation_status",
        "uncertainty_status",
        "uncertainty_design_id",
    }
    _exact_keys(values, expected, "population_contract")
    return PopulationContractConfig(
        **{
            **values,
            "uncertainty_status": _uncertainty_availability(
                values["uncertainty_status"],
                name="population_contract uncertainty_status",
            ),
        }
    )


def _monetary_contract(
    value: object,
    *,
    config_path: Path,
) -> MonetaryContractConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="monetary_contract")
    expected = {
        "profile_path",
        "source_bundle_path",
        "source_artifact_path",
        "conversion_table_path",
        "bundle_id",
        "source_bundle_sha256",
        "source_artifact_sha256",
        "conversion_table_sha256",
        "conversion_basis_id",
        "conversion_basis_sha256",
        "rate_evidence_sha256",
        "target_currency",
        "target_minor_unit_name",
        "quote_convention",
        "scale_convention",
        "rate_period_start",
        "rate_period_end",
        "target_price_period_start",
        "target_price_period_end",
        "missing_date_policy",
        "identity_missing_date_policy",
        "rounding_method",
        "rounding_scope",
        "point_rate_status",
        "rate_uncertainty_status",
        "source_bundle_signature_status",
        "simulation_bridge_status",
        "observed_real_world_spending",
        "raw_cross_currency_pooling",
    }
    _exact_keys(values, expected, "monetary_contract")
    paths = {
        name: _resolved_config_path(
            values[name],
            config_path=config_path,
            name=f"monetary {name}",
        )
        for name in (
            "profile_path",
            "source_bundle_path",
            "source_artifact_path",
            "conversion_table_path",
        )
    }
    return MonetaryContractConfig(
        **{
            **values,
            **paths,
            "rate_uncertainty_status": _uncertainty_availability(
                values["rate_uncertainty_status"],
                name="monetary rate_uncertainty_status",
            ),
        }
    )


def _campaign_output_contract(
    value: object,
) -> CampaignOutputContractConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="output_contract")
    expected = {
        "metric_registry_schema_version",
        "metric_registry_sha256",
        "output_schema_version",
        "output_schema_sha256",
        "output_profile_id",
        "output_profile_schema_version",
        "output_profile_sha256",
        "expected_artifacts",
    }
    _exact_keys(values, expected, "output_contract")
    artifacts = values["expected_artifacts"]
    if type(artifacts) is not list:
        raise ValueError("output_contract expected_artifacts must be an array")
    return CampaignOutputContractConfig(
        **{**values, "expected_artifacts": tuple(artifacts)}
    )


def _campaign_ledger(
    value: object,
    *,
    config_path: Path,
) -> CampaignLedgerConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="ledger")
    _exact_keys(
        values,
        {"backend", "path", "persistent", "temporary"},
        "ledger",
    )
    try:
        backend = LedgerBackend(values["backend"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ledger backend must be 'sqlite'") from exc
    return CampaignLedgerConfig(
        backend=backend,
        path=_resolved_config_path(
            values["path"],
            config_path=config_path,
            name="ledger path",
        ),
        persistent=values["persistent"],
        temporary=values["temporary"],
    )


def _execution_receipt_policy(
    value: object,
    *,
    config_path: Path,
) -> ExecutionReceiptPolicyConfig | None:
    if value is None:
        return None
    values = _optional_table(value, name="execution_receipt")
    expected = {
        "schema_path",
        "schema_version",
        "identity_algorithm",
        "receipt_path",
        "attestation_path",
        "require_clean_working_tree",
        "verify_active_commit",
        "verify_source_tree",
        "verify_interpreter",
        "verify_dependencies",
        "reject_environment_drift",
        "manifest_reference_required",
        "run_command",
        "execution_mode",
    }
    _exact_keys(values, expected, "execution_receipt")
    command = values["run_command"]
    if type(command) is not list:
        raise ValueError("execution_receipt run_command must be an array")
    paths = {
        name: _resolved_config_path(
            values[name],
            config_path=config_path,
            name=f"execution receipt {name}",
        )
        for name in ("schema_path", "receipt_path", "attestation_path")
    }
    return ExecutionReceiptPolicyConfig(
        **{**values, **paths, "run_command": tuple(command)}
    )


def _optional_table(value: object, *, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _optional_boolean(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


def _nonempty_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _identity_text_or_placeholder(value: object, *, name: str) -> str:
    result = _nonempty_text(value, name=name)
    if any(character.isspace() for character in result):
        raise ValueError(f"{name} cannot contain whitespace")
    return result


def _identity_sha256_or_placeholder(value: object, *, name: str) -> str:
    if type(value) is not str or not (
        _SHA256.fullmatch(value) or _BLOCKED_IDENTITY.fullmatch(value)
    ):
        raise ValueError(
            f"{name} must be lowercase SHA-256 hex or an explicit "
            "BLOCKED_PENDING_* placeholder"
        )
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite(value: object, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _uncertainty_availability(
    value: object,
    *,
    name: str,
) -> UncertaintyAvailability:
    try:
        return UncertaintyAvailability(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be QUANTIFIED, UNQUANTIFIED, or UNAVAILABLE"
        ) from exc


def _resolved_config_path(
    value: object,
    *,
    config_path: Path,
    name: str,
) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be non-empty text")
    root = Path(os.path.abspath(os.fspath(config_path))).parent
    candidate = Path(value)
    selected = candidate if candidate.is_absolute() else root / candidate
    return Path(os.path.abspath(os.fspath(selected)))


def _validate_full_campaign_seeds(value: object) -> None:
    if type(value) is not list:
        raise ValueError("full campaign seeds must be a TOML array")
    if len(value) < 100:
        raise ValueError("full campaign fixed seed set requires at least 100 seeds")
    if any(type(seed) is not int or isinstance(seed, bool) for seed in value):
        raise ValueError("full campaign seeds must be integers")
    if value != sorted(value) or len(value) != len(set(value)):
        raise ValueError(
            "full campaign fixed seeds must be unique and strictly ascending"
        )


def _validate_full_exploratory_seeds(value: object) -> None:
    if type(value) is not list:
        raise ValueError("full exploratory seeds must be a TOML array")
    if len(value) < 100:
        raise ValueError(
            "full exploratory fixed seed set requires at least 100 seeds"
        )
    if any(type(seed) is not int or isinstance(seed, bool) for seed in value):
        raise ValueError("full exploratory seeds must be integers")
    if value != sorted(value) or len(value) != len(set(value)):
        raise ValueError(
            "full exploratory fixed seeds must be unique and strictly ascending"
        )


def _contains_blocked_placeholder(value: object) -> bool:
    return "BLOCKED_PENDING_" in repr(value)


def _policy_run_purpose(value: object) -> PolicyRunPurpose:
    if value is None:
        return PolicyRunPurpose.DEVELOPMENT
    try:
        return PolicyRunPurpose(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "meta.run_purpose must be 'development', 'exploratory', or 'campaign'"
        ) from exc


def _required_and_optional_keys(
    values: Mapping[str, object],
    required: set[str],
    optional: set[str],
    name: str,
) -> None:
    actual = set(values)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - optional)
    if missing or unknown:
        raise PolicyConfigurationError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


def _exact_keys(
    values: Mapping[str, object], expected: set[str], name: str
) -> None:
    actual = set(values)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise PolicyConfigurationError(
            f"{name} keys differ: missing={missing}, unknown={unknown}"
        )


__all__ = [
    "AnalysisPlanSelection",
    "CampaignControlConfig",
    "CampaignConvergenceConfig",
    "CampaignLedgerConfig",
    "CampaignOutputContractConfig",
    "CampaignUncertaintyConfig",
    "ExecutionReceiptPolicyConfig",
    "EXPLORATORY_ARTIFACT_NAMESPACE",
    "EXPLORATORY_ESTIMAND_INTERPRETATION",
    "EXPLORATORY_EXECUTION_KIND",
    "EXPLORATORY_INTERNAL_MONETARY_UNIT",
    "EXPLORATORY_MONETARY_AMOUNT_SEMANTICS",
    "EXPLORATORY_PLAN_FILENAME",
    "EXPLORATORY_PLAN_ID",
    "EXPLORATORY_POPULATION_BASIS",
    "EXPLORATORY_RAW_INTERNAL_UNIT_OUTPUT_ROLE",
    "EXPLORATORY_UNWEIGHTED_OUTPUT_ROLE",
    "ExploratoryCheckpointConfig",
    "ExploratoryControlConfig",
    "MonetaryContractConfig",
    "PolicyConfigurationError",
    "PolicyOutputConfig",
    "PolicyPrototypeConfig",
    "PolicyRunPurpose",
    "PolicySimulationLayer",
    "PopulationContractConfig",
    "PopulationProjectionConfig",
    "UncertaintyAvailability",
    "load_policy_config",
]
