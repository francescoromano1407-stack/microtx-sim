from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from microtx_sim.outputs.exploratory import (
    EXPLORATORY_INTERNAL_UNIT_WORDING,
    EXPLORATORY_INTERPRETATION_WORDING,
    ExploratoryValidationMetadataError,
    build_exploratory_validation_metadata,
    require_exploratory_manifest_metadata,
    validate_exploratory_validation_metadata,
)
from microtx_sim.outputs.export import export_policy_batch
from microtx_sim.outputs.manifest import build_run_manifest
from microtx_sim.policy_config import load_policy_config


ROOT = Path(__file__).resolve().parents[1]
EXPLORATORY_CONFIG = ROOT / "configs" / "policy_exploratory_synthetic.toml"
PROTOTYPE_CONFIG = ROOT / "configs" / "policy_prototype.toml"


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _metadata() -> dict[str, object]:
    return build_exploratory_validation_metadata(
        configuration_path="configs/policy_exploratory_synthetic.toml",
        configuration_sha256=_digest("exploratory config"),
        exploratory_plan_path="inputs/exploratory-synthetic-analysis-plan-v1.json",
        exploratory_plan_id="illustrative.exploratory.synthetic.v1",
        exploratory_plan_sha256=_digest("exploratory semantic plan"),
        exploratory_plan_file_sha256=_digest("exploratory plan bytes"),
        scientific_parent_plan_path=(
            "inputs/prospective-analysis-plan-amendment-v3.json"
        ),
        scientific_parent_plan_id="illustrative.scientific.parent.v3",
        scientific_parent_plan_sha256=_digest("parent semantic plan"),
        scientific_parent_plan_file_sha256=_digest("parent plan bytes"),
        scientific_parent_registration_status="UNREGISTERED",
        primary_estimand_id="primary.composite-harm.baseline-vs-safe.v1",
        population_adapter_id="campaign.standardized.population.v2",
        population_adapter_sha256=_digest("adapter"),
        population_execution_input_sha256=_digest("population input"),
        scenario_snapshots=(
            {"scenario_id": "baseline_f2p", "mechanics": {"pressure": 1}},
            {
                "scenario_id": "safe_fixed_price_subscription",
                "mechanics": {"pressure": 0},
            },
        ),
        target_currency="EUR",
        target_minor_unit_name="euro cent",
        quote_convention="target minor units per source minor unit",
        scale_convention=(
            "local nominal monthly anchor minor units per 180000 "
            "simulation_cents"
        ),
        rate_period_start="2024-01-01",
        rate_period_end="2024-12-31",
        target_price_period_start="2024-01-01",
        target_price_period_end="2024-12-31",
        missing_date_policy=(
            "use the official annual observation without local date filling "
            "or imputation"
        ),
        identity_missing_date_policy="not applicable to an identity conversion",
        rounding_method="nearest_minor_unit_half_away_from_zero",
        rounding_scope="AFTER_AGGREGATION",
        point_rate_status="OFFICIAL_POINT_OBSERVATION",
        rate_uncertainty_status="UNQUANTIFIED",
        source_bundle_signature_status="MISSING",
        simulation_bridge_status="ILLUSTRATIVE",
        observed_real_world_spending=False,
        raw_cross_currency_pooling="REJECT",
        monetary_source_bundle_sha256=_digest("bundle"),
        monetary_source_artifact_sha256=_digest("source"),
        monetary_conversion_table_sha256=_digest("table"),
        monetary_conversion_basis_sha256=_digest("basis"),
        monetary_rate_evidence_sha256=_digest("rate"),
        profile_input_lineage_sha256=_digest("profile lineage"),
        uncertainty_design={
            "schema_version": "1.0",
            "seed_uncertainty": {
                "status": "QUANTIFIED_WHEN_COMPLETE",
                "fixed_seed_count": 150,
                "common_random_numbers": True,
                "identical_pretreatment_cohorts": True,
                "population_weights_applied_within_seed": True,
                "outcome_dependent_seed_exclusion_allowed": False,
            },
            "parameter_uncertainty": {
                "status": "ILLUSTRATIVE_DESIGN_ONLY",
                "design_id": "illustrative.policy-model-joint-parameters.v1",
                "design_sha256": _digest("parameter design"),
                "method": "SEEDED_LATIN_HYPERCUBE_V1",
                "probability_interpretation": "NONE",
            },
            "monetary_rate_uncertainty": {
                "status": "UNQUANTIFIED",
                "point_observation_is_distribution": False,
                "rate_basis_sha256": _digest("basis"),
            },
            "population_uncertainty": {
                "status": "UNQUANTIFIED",
                "exact_weighting_is_empirical_validation": False,
                "uncertainty_design_id": (
                    "UNAVAILABLE_NO_ADMISSIBLE_POPULATION_UNCERTAINTY_DESIGN"
                ),
            },
            "combined_uncertainty": {
                "status": "UNAVAILABLE_UNTIL_ALL_REQUIRED_COMPONENTS_EXIST",
                "double_counting_control": (
                    "one complete seed-parameter-population-rate Cartesian identity"
                ),
                "variance_decomposition_method": (
                    "ORTHOGONAL_FINITE_FULL_FACTORIAL_ANOVA_"
                    "SUM_OF_SQUARES_DIVIDED_BY_N_V1"
                ),
            },
            "oat_role": "DIAGNOSTIC_ONLY",
        },
        convergence_rule={
            "schema_version": "1.0",
            "block_size": 50,
            "minimum_retained_seeds": 100,
            "maximum_mcse": 0.0025,
            "maximum_interval_width": 0.01,
            "maximum_absolute_change": 0.0025,
            "maximum_relative_change": 0.025,
            "maximum_invalid_rate": 0.0,
            "consecutive_passing_checkpoints": 2,
            "sensitivity_instability_allowed": False,
            "outcome_dependent_seed_exclusion_allowed": False,
            "required_uncertainty_component_handling": "FAIL_CLOSED",
        },
        fixed_seed_count=150,
        parameter_design_path="inputs/parameter-uncertainty-design-v1.json",
        parameter_design_file_sha256=_digest("parameter design file"),
        production_configuration_sha256=_digest("production config"),
        production_plan_sha256=_digest("production plan"),
    )


def _rehash(payload: dict[str, object]) -> None:
    identity = dict(payload)
    identity.pop("metadata_sha256", None)
    payload["metadata_sha256"] = sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def test_metadata_is_deterministic_non_executing_and_dual_plan_bound() -> None:
    first = _metadata()
    second = _metadata()

    assert first == second
    validate_exploratory_validation_metadata(first)
    assert first["campaign_ready"] is False
    assert first["production_campaign"] is False
    assert first["claims"] == {
        "empirical_validation_claimed": False,
        "population_inference_claimed": False,
        "real_world_causal_effect_claimed": False,
        "generalisation_claimed": False,
        "observed_real_world_spending_claimed": False,
        "production_readiness_claimed": False,
    }
    assert first["interpretation_wording"] == EXPLORATORY_INTERPRETATION_WORDING
    assert first["execution"] == {
        "status": "NOT_EXECUTED",
        "validation_only": True,
        "cohort_initialized": False,
        "scenario_executed": False,
        "policy_batch_dispatched": False,
        "sensitivity_dispatched": False,
        "reproduce_dispatched": False,
        "intended_launch_command": [
            "microtx-sim",
            "policy-batch",
            "configs/policy_exploratory_synthetic.toml",
        ],
    }
    assert first["exploratory_analysis_plan"]["production_authority"] == "NONE"
    assert (
        first["scientific_parent_plan"]["plan_id"]
        == "illustrative.scientific.parent.v3"
    )
    boundary = first["production_boundary"]
    assert boundary["production_configuration_used_as_execution_input"] is False
    assert boundary["scientific_parent_plan_reused"] is True
    assert boundary["production_campaign_authority_inherited"] is False
    assert boundary["production_gates_altered"] is False


def test_raw_simulation_cents_cannot_be_relabelled_as_real_money() -> None:
    metadata = _metadata()
    monetary = metadata["monetary_contract"]
    assert monetary["internal_monetary_unit"] == "simulation_cents"
    assert monetary["internal_unit_wording"] == EXPLORATORY_INTERNAL_UNIT_WORDING
    assert monetary["raw_simulation_cents_are_real_money"] is False
    assert monetary["raw_simulation_cents_allowed_as_cross_country_result"] is False
    assert monetary["observed_real_world_spending"] is False
    assert monetary["observed_fx_conversion"] is True
    assert monetary["internal_to_real_money_bridge_empirically_calibrated"] is False
    assert monetary["point_rate_status"] == "OFFICIAL_POINT_OBSERVATION"
    assert monetary["source_bundle_signature_status"] == "MISSING"
    assert monetary["simulation_bridge_status"] == "ILLUSTRATIVE"
    assert monetary["rate_uncertainty_status"] == "UNQUANTIFIED"
    assert monetary["raw_cross_currency_pooling"] == "REJECT"
    assert monetary["model_equivalent_label_required"] is True

    monetary["raw_simulation_cents_are_real_money"] = True
    _rehash(metadata)
    with pytest.raises(ExploratoryValidationMetadataError, match="unit contract"):
        validate_exploratory_validation_metadata(metadata)


def test_population_scenario_pairing_cannot_be_weakened() -> None:
    metadata = _metadata()
    binding = metadata["population_scenario_identity"]
    assert binding["same_weighted_population_scenario_identity_declared"] is True
    assert binding["identical_pretreatment_cohorts_required"] is True
    assert binding["identical_population_assignments_across_scenarios_required"] is True
    assert binding["identical_population_weights_across_scenarios_required"] is True
    assert binding["realized_assignment_status"] == "DECLARED_NOT_EXECUTED"

    binding["identical_population_weights_across_scenarios_required"] = False
    _rehash(metadata)
    with pytest.raises(ExploratoryValidationMetadataError, match="identity field"):
        validate_exploratory_validation_metadata(metadata)

    metadata = _metadata()
    metadata["population_scenario_identity"]["scenario_set"]["scenarios"][0][
        "mechanics"
    ]["pressure"] = 999
    _rehash(metadata)
    with pytest.raises(ExploratoryValidationMetadataError, match="scenario"):
        validate_exploratory_validation_metadata(metadata)


def test_mandatory_non_empirical_wording_cannot_be_paraphrased() -> None:
    metadata = _metadata()
    metadata["interpretation_wording"] = "synthetic only"
    metadata["output_interpretation"][
        "human_readable_label_required"
    ] = "synthetic only"
    _rehash(metadata)
    with pytest.raises(ExploratoryValidationMetadataError, match="fixed scope"):
        validate_exploratory_validation_metadata(metadata)


@pytest.mark.parametrize(
    "claim",
    ("population_inference_claimed", "generalisation_claimed"),
)
def test_population_and_generalisation_claims_cannot_be_promoted(
    claim: str,
) -> None:
    metadata = _metadata()
    metadata["claims"][claim] = True
    _rehash(metadata)
    with pytest.raises(
        ExploratoryValidationMetadataError,
        match="cannot make empirical, population, causal",
    ):
        validate_exploratory_validation_metadata(metadata)


def test_unexecuted_uncertainty_status_cannot_be_promoted() -> None:
    metadata = _metadata()
    uncertainty = metadata["uncertainty_and_convergence"]
    assert uncertainty["fixed_seed_count"] == 150
    assert uncertainty["convergence_status"] == "NOT_EVALUATED_NOT_EXECUTED"
    assert uncertainty["monte_carlo_diagnostics"] == "NOT_COMPUTED"
    assert uncertainty["sensitivity_status"] == "NOT_EXECUTED"
    assert uncertainty["convergence_claimed"] is False

    uncertainty["convergence_status"] = "CONVERGED"
    uncertainty["convergence_claimed"] = True
    _rehash(metadata)
    with pytest.raises(
        ExploratoryValidationMetadataError,
        match="overstates execution",
    ):
        validate_exploratory_validation_metadata(metadata)


def test_manifest_metadata_is_required_only_for_exploratory_output() -> None:
    metadata = _metadata()
    accepted = require_exploratory_manifest_metadata(
        exploratory_requested=True,
        metadata=metadata,
    )
    assert accepted == metadata
    assert accepted is not metadata

    with pytest.raises(
        ExploratoryValidationMetadataError,
        match="requires verified",
    ):
        require_exploratory_manifest_metadata(
            exploratory_requested=True,
            metadata=None,
        )
    with pytest.raises(
        ExploratoryValidationMetadataError,
        match="forbidden",
    ):
        require_exploratory_manifest_metadata(
            exploratory_requested=False,
            metadata=metadata,
        )


def test_export_and_manifest_boundaries_require_and_forbid_review_metadata() -> None:
    exploratory = load_policy_config(EXPLORATORY_CONFIG)
    prototype = load_policy_config(PROTOTYPE_CONFIG)

    with patch("microtx_sim.outputs.export.PolicyBatchResult", dict):
        with pytest.raises(
            ExploratoryValidationMetadataError,
            match="requires verified",
        ):
            export_policy_batch(
                exploratory,
                {},
                None,
                config_path=EXPLORATORY_CONFIG,
                repository_root=ROOT,
            )
        with pytest.raises(
            ExploratoryValidationMetadataError,
            match="forbidden",
        ):
            export_policy_batch(
                prototype,
                {},
                None,
                config_path=PROTOTYPE_CONFIG,
                repository_root=ROOT,
                exploratory_validation_metadata=_metadata(),
            )

    with pytest.raises(
        ExploratoryValidationMetadataError,
        match="requires verified",
    ):
        build_run_manifest(
            exploratory,
            object(),
            config_path=EXPLORATORY_CONFIG,
            repository_root=ROOT,
        )
    with pytest.raises(
        ExploratoryValidationMetadataError,
        match="forbidden",
    ):
        build_run_manifest(
            prototype,
            object(),
            config_path=PROTOTYPE_CONFIG,
            repository_root=ROOT,
            exploratory_validation_metadata=_metadata(),
        )


def test_exploratory_export_rejects_output_override_before_writes(
    tmp_path: Path,
) -> None:
    exploratory = load_policy_config(EXPLORATORY_CONFIG)
    forbidden = tmp_path / "outside-reviewed-namespace"

    with patch("microtx_sim.outputs.export.PolicyBatchResult", dict):
        with pytest.raises(ValueError, match="destination must remain fixed"):
            export_policy_batch(
                exploratory,
                {},
                None,
                config_path=EXPLORATORY_CONFIG,
                repository_root=ROOT,
                output_dir=forbidden,
                exploratory_validation_metadata=_metadata(),
            )

    assert not forbidden.exists()
