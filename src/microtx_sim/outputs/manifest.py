"""Structured provenance metadata for synthetic policy runs."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

import numpy as np

from ..causal.batch import (
    PolicyBatchResult,
    PolicyRunInputs,
    resolve_policy_run_inputs,
)
from ..causal.analysis_binding import (
    RunAnalysisBinding,
    resolve_run_analysis_binding,
)
from ..causal.analysis_plan import (
    LoadedProspectiveAnalysisPlan,
    verify_loaded_prospective_analysis_plan,
)
from ..causal.design import assess_causal_design
from ..data.lineage import ProfileInputLineage
from ..data.profiles import (
    ProfileValidationError,
    monetary_evidence_assessment_from_snapshot,
    monetary_structure_assessment_from_snapshot,
    population_evidence_assessment_from_snapshot,
)
from ..metrics.population_estimands import TARGET_POPULATION_OUTPUT_PROFILE
from ..policy_config import PolicyPrototypeConfig, PolicyRunPurpose
from .metric_contracts import build_metric_contract_manifest_payload


def build_run_manifest(
    config: PolicyPrototypeConfig,
    batch: PolicyBatchResult,
    *,
    config_path: str | Path,
    repository_root: str | Path,
    created_utc: str | None = None,
    command: Sequence[str] | None = None,
    analysis_plan: LoadedProspectiveAnalysisPlan | None = None,
    analysis_binding: RunAnalysisBinding | None = None,
) -> dict[str, object]:
    """Build a self-contained run manifest without claiming empirical validity."""

    config_file = Path(config_path).resolve()
    repository = Path(repository_root).resolve()
    configured_run_inputs = resolve_policy_run_inputs(
        harm_parameters=config.harm_parameters,
        harm_weights=config.harm_weights,
        opportunity_valuation=config.opportunity_valuation,
        producer_assumptions=config.producer_assumptions,
        epgc_policy=config.epgc_policy,
    )
    _validate_config_matches_batch(config, batch, configured_run_inputs)
    analysis_plan, analysis_binding = _validate_analysis_composition(
        config,
        batch,
        analysis_plan=analysis_plan,
        analysis_binding=analysis_binding,
    )
    git_commit, git_dirty = _git_state(repository)
    profile_lineage = batch.profile_input_lineage
    if profile_lineage is None:
        profile_inputs: dict[str, object] = {
            "lineage_status": "unavailable_legacy_result",
            "profile_codes": [profile.code for profile in batch.country_profiles],
            "fingerprint_sha256": None,
            "snapshot": None,
            "jurisdictions": {"path": None, "sha256": None},
            "source_registry": {
                "path": None,
                "sha256": None,
                "retrieved_on": None,
            },
            "source_bundle": {
                "path": None,
                "sha256": None,
                "source_registry_sha256": None,
                "signature_status": None,
            },
            "population_bundle": {
                "path": None,
                "sha256": None,
                "source_registry_sha256": None,
                "signature_status": None,
            },
            "metric_contract_summary": {"count": 0, "status_counts": {}},
            "money_scale_summary": {
                "count": 0,
                "currencies": [],
                "anchor_status_counts": {},
                "scale_status_counts": {},
            },
            "monetary_conversion_summary": {
                "count": 0,
                "methods": [],
                "source_currencies": [],
                "target_currencies": [],
                "rate_period_starts": [],
                "rate_period_ends": [],
                "target_price_period_starts": [],
                "target_price_period_ends": [],
                "estimands": [],
                "population_bases": [],
                "comparison_groups": [],
                "retrieval_dates": [],
                "rounding_scopes": [],
                "aggregation_units": [],
                "status_counts": {},
            },
            "source_evidence_summary": {
                "present": False,
                "artifact_count": 0,
                "binding_count": 0,
                "verified_result_count": 0,
                "signature_status": None,
            },
            "population_evidence_summary": {
                "present": False,
                "artifact_count": 0,
                "binding_count": 0,
                "verified_result_count": 0,
                "signature_status": None,
            },
            "monetary_evidence_assessment": {
                "structure_coherent": False,
                "source_rate_evidence_bound": False,
                "source_bundle_signature_bound": False,
                "output_design_binding_bound": False,
                "population_binding_bound": False,
                "preregistration_bound": False,
                "public_output_comparability": False,
                "blockers": [
                    "monetary_conversion.structure=unavailable",
                    "monetary_conversion.source_rate_binding=missing",
                    "monetary_conversion.source_bundle_signature=missing",
                    "monetary_conversion.output_design_binding=missing",
                    "monetary_conversion.population_binding=missing",
                    "monetary_conversion.preregistration_binding=missing",
                ],
            },
            "population_evidence_assessment": {
                "structure_coherent": False,
                "source_population_evidence_bound": False,
                "calibration_targets_bound": False,
                "heldout_validation_targets_bound": False,
                "source_bundle_signature_bound": False,
                "sampling_plan_bound": False,
                "runtime_projection_bound": False,
                "output_estimand_binding_bound": False,
                "balance_validation_bound": False,
                "public_population_comparability": False,
                "blockers": [
                    "population.structure=unavailable",
                    "population.source_evidence=missing",
                    "population.calibration_targets=missing",
                    "population.heldout_validation_targets=missing",
                    "population.source_bundle_signature=missing",
                    "population.sampling_plan=missing",
                    "population.runtime_projection=missing",
                    "population.output_estimand_binding=missing",
                    "population.balance_validation=missing",
                ],
            },
        }
    else:
        profile_inputs = profile_lineage.manifest_payload()
    population_execution_payload = (
        batch.population_execution_lineage.manifest_payload()
        if batch.population_execution_lineage is not None
        else None
    )
    population_readiness = _population_readiness_payload(
        profile_inputs,
        profile_lineage=profile_lineage,
    )
    population_output_contract = _population_output_contract(
        config,
        batch,
        analysis_binding=analysis_binding,
        population_execution_payload=population_execution_payload,
        population_readiness=population_readiness,
        profile_lineage=profile_lineage,
    )
    campaign_gate = population_output_contract["campaign_gate"]
    if (
        config.run_purpose is PolicyRunPurpose.CAMPAIGN
        and (
            not isinstance(campaign_gate, dict)
            or campaign_gate.get("passed") is not True
        )
    ):
        blockers = campaign_gate.get("blockers")
        rendered = (
            ", ".join(str(item) for item in blockers)
            if isinstance(blockers, list)
            else "population manifest gate is malformed"
        )
        raise ValueError(
            "campaign population manifest gate failed closed: " + rendered
        )
    run_input_snapshot = batch.run_input_snapshot()
    run_input_sha256 = batch.run_input_sha256()
    causal_design = assess_causal_design(
        batch.spec.scenarios
    ).manifest_payload(run_input_sha256=run_input_sha256)
    effective_config_snapshot = _effective_config_snapshot(
        config,
        configured_run_inputs,
    )
    config_file_sha256 = _file_digest(config_file)
    jurisdictions_metadata = profile_inputs["jurisdictions"]
    source_registry_metadata = profile_inputs["source_registry"]
    if not isinstance(jurisdictions_metadata, dict) or not isinstance(
        source_registry_metadata, dict
    ):
        raise ValueError("profile input file lineage is malformed")
    output_metric_contracts = build_metric_contract_manifest_payload(
        configuration_status=config.provenance_status,
        profile_lineage_status=str(profile_inputs["lineage_status"]),
        profile_dependencies_calibrated=(
            _profile_dependencies_calibrated(
                profile_inputs,
                profile_lineage=profile_lineage,
            )
        ),
        profile_input_fingerprint_sha256=(
            str(profile_inputs["fingerprint_sha256"])
            if profile_inputs["fingerprint_sha256"] is not None
            else None
        ),
        run_source_retrieved_on=(
            profile_lineage.source_retrieved_on
            if profile_lineage is not None
            else None
        ),
        monetary_outputs_cross_country_comparable=(
            _money_outputs_cross_country_comparable(
                profile_inputs,
                profile_lineage=profile_lineage,
            )
        ),
        run_input_sha256=run_input_sha256,
    )
    manifest = {
        "run_name": config.name,
        "run_purpose": config.run_purpose.value,
        "created_utc": created_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance_status": config.provenance_status,
        "synthetic_only": True,
        "empirical_validation_claimed": False,
        "notes": config.notes,
        "config_path": str(config_file),
        "config_sha256": config_file_sha256,
        "config_file_sha256_observed_at_export": config_file_sha256,
        "config_file_attestation": (
            "export-time observation only; execution truth is run_input_sha256"
        ),
        "effective_config_sha256": _canonical_sha256(
            effective_config_snapshot
        ),
        "config_snapshot": effective_config_snapshot,
        "run_input_sha256": run_input_sha256,
        "run_input_snapshot": run_input_snapshot,
        "jurisdictions_sha256": jurisdictions_metadata.get("sha256"),
        "source_registry_sha256": source_registry_metadata.get("sha256"),
        "source_registry_retrieved_on": source_registry_metadata.get(
            "retrieved_on"
        ),
        "profile_inputs": profile_inputs,
        "monetary_comparability": _monetary_comparability_payload(
            profile_inputs,
            profile_lineage=profile_lineage,
        ),
        "population_readiness": population_readiness,
        "population_output_contract": population_output_contract,
        "causal_design": causal_design,
        "output_metric_contracts": output_metric_contracts,
        "repository": {
            "root": str(repository),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
        },
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "command": list(command) if command is not None else None,
        "batch": {
            "seeds": list(batch.spec.seeds),
            "seed_decimal_strings": [str(seed) for seed in batch.spec.seeds],
            "days": batch.spec.days,
            "player_count": batch.spec.player_count,
            "step_minutes": batch.spec.decision_parameters.step_minutes,
            "reference_scenario": batch.spec.reference_scenario.value,
            "profile_codes": list(profile_inputs["profile_codes"]),
            "profile_input_fingerprint_sha256": profile_inputs[
                "fingerprint_sha256"
            ],
            "cohort_digest_by_seed": {
                str(seed): digest
                for seed, digest in batch.cohort_digest_by_seed.items()
            },
        },
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id.value,
                "label": scenario.label,
                "description": scenario.description,
                "mechanics": asdict(scenario.mechanics),
                "fixed_access_price_cents": scenario.fixed_access_price_cents,
                "subscription_price_cents": scenario.subscription_price_cents,
                "epgc_enabled": scenario.epgc_enabled,
            }
            for scenario in batch.spec.scenarios
        ],
        "decision_parameters": asdict(batch.spec.decision_parameters),
        "harm": {
            "equation": "H_i = w_M*M_i + w_T*OC_i + w_S*S_i + w_E*E_i + w_F*F_i + w_W*W_i",
            "parameters": asdict(batch.run_inputs.harm_parameters),
            "weights": asdict(batch.run_inputs.harm_weights),
            "opportunity_valuation": asdict(
                batch.run_inputs.opportunity_valuation
            ),
        },
        "producer_assumptions": asdict(batch.run_inputs.producer_assumptions),
        "epgc_policy": asdict(batch.run_inputs.epgc_policy),
        "random_stream_contract": {
            "generator": "CounterRNG/SplitMix64",
            "coordinates": ["seed", "player_id", "tick", "stream", "draw_index"],
            "root_seed": {
                "accepted_runtime_type": "strict Python int (bool excluded)",
                "minimum_decimal": "0",
                "maximum_decimal": "18446744073709551615",
                "normalization": "none; out-of-range values are rejected",
                "json_exact_encoding": "batch.seed_decimal_strings",
            },
            "batch_seed_order": "unique ascending numeric order",
            "named_streams": [
                "player-life:*",
                "policy:life-action-gumbel",
                "policy:reward-prediction-error",
                "policy:purchase-revenue-source",
                "policy:access-adoption",
                "policy:access-plan",
            ],
        },
        "scope_limits": [
            "All population, behavioural, harm, cost, and financing values are synthetic.",
            "The outputs identify effects inside the structural model, not real-world causal effects.",
            "No clinical diagnosis, real-user profiling, targeting, or data collection is implemented.",
            "Public financing is a policy simulation, not a legal conclusion or subsidy application.",
        ],
    }
    if population_execution_payload is not None:
        manifest["population_execution"] = population_execution_payload
    if analysis_plan is not None and analysis_binding is not None:
        manifest["analysis_plan"] = analysis_plan.manifest_payload()
        manifest["analysis_binding"] = analysis_binding.manifest_payload()
        monetary_bases = analysis_binding.monetary_output_bases
        if monetary_bases:
            manifest["prospective_monetary_output_execution"] = {
                "present": True,
                "scope": "prospective_analysis target-population estimands only",
                "basis_count": len(monetary_bases),
                "basis_sha256s": [
                    basis.basis_sha256 for basis in monetary_bases
                ],
                "target_currencies": [
                    basis.target_currency for basis in monetary_bases
                ],
                "interpretation": "target-currency-equivalent model amounts",
                "per_observation_before_contrast_and_weighting": True,
                "observed_currency_recovered": False,
                "legacy_root_outputs_relabelled": False,
                "legacy_root_monetary_outputs_cross_country_comparable": False,
                "campaign_ready": False,
                "campaign_blockers": sorted(
                    {
                        "monetary_output_execution.population_semantics_compatibility="
                        "unreviewed",
                    }.union(
                        blocker
                        for basis in monetary_bases
                        for blocker in basis.campaign_blockers
                    ).union(
                        analysis_binding.campaign_blockers
                    )
                ),
            }
    return manifest


_LEGACY_ROOT_TABLES = (
    "seed_results.csv",
    "scenario_summary.csv",
    "player_outcomes.csv",
    "opportunity_cost_decomposition.csv",
    "epgc_financing.csv",
    "sensitivity.csv",
)


def _population_output_contract(
    config: PolicyPrototypeConfig,
    batch: PolicyBatchResult,
    *,
    analysis_binding: RunAnalysisBinding | None,
    population_execution_payload: dict[str, object] | None,
    population_readiness: dict[str, object],
    profile_lineage: ProfileInputLineage | None,
) -> dict[str, object]:
    """Describe population semantics without changing any legacy output schema."""

    lineage = batch.population_execution_lineage
    binding_verified = analysis_binding is not None
    prospective_profile: dict[str, object] = {
        "binding_present": binding_verified,
        "profile_declared": binding_verified,
        "publication_claimed": False,
        "publication_attestation": (
            "published artifacts are attested separately by analysis_output_profile"
        ),
        "weighted": binding_verified,
        "role": (
            "prospective_target_population_estimands"
            if binding_verified
            else "unavailable"
        ),
        "interpretation": (
            "exact declared target-population weighted estimands"
            if binding_verified
            else "no verified analysis binding; no weighted population claim"
        ),
        "analysis_binding_verified": binding_verified,
        "binding_schema_version": (
            analysis_binding.schema_version if analysis_binding is not None else None
        ),
        "binding_sha256": (
            analysis_binding.binding_sha256 if analysis_binding is not None else None
        ),
        "output_profile_id": (
            TARGET_POPULATION_OUTPUT_PROFILE
            if analysis_binding is not None
            else None
        ),
        "output_profile_schema_sha256": (
            analysis_binding.output_profile_schema_sha256
            if analysis_binding is not None
            else None
        ),
        "campaign_ready": (
            analysis_binding.campaign_ready
            if analysis_binding is not None
            else False
        ),
        "campaign_blockers": (
            list(analysis_binding.campaign_blockers)
            if analysis_binding is not None
            else ["population.analysis_binding=missing"]
        ),
    }

    projected: dict[str, object] | None = None
    blockers: list[str] = []
    if config.population is None:
        blockers.append("population.configuration.projected_v1=missing")
    if lineage is None:
        blockers.append("population.execution_lineage=missing")
    else:
        # This call reopens every registered population file and re-attests the
        # adapter, assignments, exact weights, and balance artifacts.
        if population_execution_payload is None:
            raise ValueError(
                "projected population lineage lacks its re-attested manifest payload"
            )
        adapter = lineage.adapter
        verification = adapter.verification
        evidence = verification.evidence_bundle
        design = verification.bundle
        mapping = adapter.mapping_bundle
        seed_profiles = [
            _population_seed_output_profile(
                record,
                adapter=adapter,
                analysis_binding=analysis_binding,
            )
            for record in lineage.seed_records
        ]
        projected = {
            "present": True,
            "mode": lineage.mode.value,
            "lineage_schema_version": lineage.schema_version,
            "lineage_sha256": lineage.lineage_sha256,
            "population_input_sha256": lineage.input_sha256,
            "evidence": {
                "bundle_id": evidence.bundle_id,
                "schema_version": evidence.schema_version,
                "population_evidence_bundle_sha256": evidence.bundle_sha256,
                "source_registry_sha256": evidence.source_registry_sha256,
                "evidence_result_sha256s": [
                    result.evidence_sha256
                    for result in verification.evidence_results
                ],
                "provenance_status": evidence.provenance_status.value,
                "campaign_ready": evidence.campaign_ready,
                "campaign_blockers": list(evidence.campaign_blockers),
            },
            "design": {
                "design_id": design.design_id,
                "schema_version": design.schema_version,
                "design_bundle_sha256": design.bundle_sha256,
                "verification_sha256": verification.verification_sha256,
                "domain_sha256": design.domain_sha256,
                "partition_sha256": design.partition_sha256,
                "calibration_target_sha256": (
                    adapter.calibration_target_sha256
                ),
                "apportionment_sha256": adapter.apportionment_sha256,
                "provenance_status": design.provenance_status.value,
                "campaign_ready": design.campaign_ready,
                "campaign_blockers": list(design.campaign_blockers),
            },
            "adapter": {
                "adapter_id": adapter.adapter_id,
                "schema_version": adapter.schema_version,
                "adapter_sha256": adapter.adapter_sha256,
                "runtime_projection_id": adapter.runtime_projection_id,
                "authenticity_verified": adapter.authenticity_verified,
                "campaign_ready": adapter.campaign_ready,
            },
            "runtime_mapping": {
                "mapping_id": mapping.mapping_id,
                "schema_version": mapping.schema_version,
                "mapping_sha256": mapping.mapping_sha256,
            },
            "weight_semantics": {
                "analysis_weight_total": (
                    "sum of declared per-player analysis weights; never "
                    "renormalized in the manifest"
                ),
                "expansion_weight_total": (
                    "sum of adapter cell expansion weights selected by each "
                    "seed record's exact ordered cell indices"
                ),
                "exact_representation": "reduced rational numerator/denominator",
                "target_population_count": (
                    adapter.apportionment_plan.total_population_count
                ),
                "target_population_count_decimal": str(
                    adapter.apportionment_plan.total_population_count
                ),
            },
            "seed_profiles": seed_profiles,
        }

        if evidence.campaign_ready is not True:
            blockers.append("population.evidence.campaign_ready=false")
        if design.campaign_ready is not True:
            blockers.append("population.design.campaign_ready=false")
        if adapter.campaign_ready is not True:
            blockers.append("population.adapter.campaign_ready=false")
        if adapter.authenticity_verified is not True:
            blockers.append("population.adapter.authenticity_verified=false")
        if mapping.schema_version < 2:
            blockers.append("population.runtime_mapping.schema_version<2")
        if adapter.schema_version < 2:
            blockers.append("population.adapter.schema_version<2")
        for seed_profile in seed_profiles:
            full = seed_profile["full_cohort"]
            if not isinstance(full, dict):
                blockers.append("population.seed.full_cohort=malformed")
                continue
            if full.get("player_count") != batch.spec.player_count:
                blockers.append("population.seed.full_cohort_count=mismatch")
            if full.get("exact_analysis_weight_total") != _fraction_payload(
                Fraction(1, 1)
            ):
                blockers.append("population.seed.analysis_weight_total!=1")
            if full.get("exact_expansion_weight_total") != _fraction_payload(
                Fraction(adapter.apportionment_plan.total_population_count, 1)
            ):
                blockers.append(
                    "population.seed.expansion_weight_total!=target_population_count"
                )
            balance = full.get("pre_treatment_balance")
            if not isinstance(balance, dict) or (
                balance.get("exact_balance_passed") is not True
            ):
                blockers.append("population.seed.pre_treatment_balance=false")
            selections = seed_profile.get("selected_profiles")
            if analysis_binding is not None and (
                not isinstance(selections, list) or not selections
            ):
                blockers.append("population.seed.selected_profile=missing")
            if isinstance(selections, list):
                for selection_profile in selections:
                    if not isinstance(selection_profile, dict):
                        blockers.append(
                            "population.seed.selected_profile=malformed"
                        )
                        continue
                    selected_count = selection_profile.get(
                        "selected_player_count"
                    )
                    excluded_count = selection_profile.get(
                        "excluded_player_count"
                    )
                    if (
                        type(selected_count) is not int
                        or selected_count <= 0
                        or selected_count > batch.spec.player_count
                        or excluded_count
                        != batch.spec.player_count - selected_count
                    ):
                        blockers.append(
                            "population.seed.selected_excluded_counts=invalid"
                        )
                    selected_weight = selection_profile.get(
                        "exact_analysis_weight_total"
                    )
                    selected_expansion = selection_profile.get(
                        "exact_expansion_weight_total"
                    )
                    if not _positive_fraction_payload(selected_weight):
                        blockers.append(
                            "population.seed.selected_analysis_weight_total<=0"
                        )
                    if not _positive_fraction_payload(selected_expansion):
                        blockers.append(
                            "population.seed.selected_expansion_weight_total<=0"
                        )

    if config.provenance_status != "calibrated":
        blockers.append("population.configuration.provenance_status!=calibrated")
    if config.batch.player_count <= 0:
        blockers.append("population.configuration.player_count<=0")
    if not config.output.include_player_rows:
        blockers.append("population.configuration.include_player_rows=false")
    if config.analysis_plan is None:
        blockers.append("population.analysis_plan=missing")
    if analysis_binding is None:
        blockers.append("population.analysis_binding=missing")
    elif analysis_binding.campaign_ready is not True:
        blockers.append("population.analysis_binding.campaign_ready=false")
    if analysis_binding is not None and (
        analysis_binding.plan.campaign_ready is not True
    ):
        blockers.append("population.analysis_plan.campaign_ready=false")
    if prospective_profile["weighted"] is not True:
        blockers.append("population.prospective_output.weighted=false")
    if profile_lineage is None:
        blockers.append("population.profile_lineage=missing")
    readiness_gate = population_readiness["manifest_gate"]
    if not isinstance(readiness_gate, dict) or (
        readiness_gate.get("public_population_comparability") is not True
    ):
        blockers.append("population.profile_evidence.comparability=false")

    blockers = sorted(set(blockers))
    gate_passed = not blockers
    campaign_requested = config.run_purpose is PolicyRunPurpose.CAMPAIGN
    return {
        "schema_version": "1.0",
        "run_purpose": config.run_purpose.value,
        "legacy_root_tables": {
            "artifact_files": list(_LEGACY_ROOT_TABLES),
            "weighted": False,
            "population_weighting_applied": False,
            "population_estimate": False,
            "role": "diagnostic",
            "interpretation": "unweighted synthetic-player summaries",
            "units_reinterpreted": False,
        },
        "prospective_population_profile": prospective_profile,
        "projected_population_lineage": projected,
        "campaign_gate": {
            "enforced": campaign_requested,
            "passed": gate_passed,
            "campaign_ready": campaign_requested and gate_passed,
            "blockers": blockers,
        },
    }


def _population_seed_output_profile(
    record,
    *,
    adapter,
    analysis_binding: RunAnalysisBinding | None,
) -> dict[str, object]:
    """Return exact full and selected population totals for one seed."""

    full_count = len(record.exact_weights.player_ids)
    position_by_player_id = {
        player_id: position
        for position, player_id in enumerate(record.exact_weights.player_ids)
    }
    full_expansion_total = _expansion_weight_total(
        record,
        adapter=adapter,
        player_ids=record.exact_weights.player_ids,
        position_by_player_id=position_by_player_id,
    )
    selected_profiles: list[dict[str, object]] = []
    if analysis_binding is not None:
        for item in analysis_binding.seed_bindings:
            if item.seed != record.seed:
                continue
            selected_count = item.selected_player_count
            selected_profiles.append(
                {
                    "planned_estimand_id": item.planned_estimand.estimand_id,
                    "resolved_estimand_id": item.spec.estimand_id,
                    "target_population_id": item.spec.target_population_id,
                    "target_evidence_sha256": item.spec.target_evidence_sha256,
                    "estimand_sha256": item.spec.estimand_sha256,
                    "result_sha256": item.result.result_sha256,
                    "estimand_role": item.planned_estimand.role.value,
                    "full_player_count": full_count,
                    "full_player_count_decimal": str(full_count),
                    "selected_player_count": selected_count,
                    "selected_player_count_decimal": str(selected_count),
                    "excluded_player_count": full_count - selected_count,
                    "excluded_player_count_decimal": str(
                        full_count - selected_count
                    ),
                    "exact_analysis_weight_total": _fraction_payload(
                        item.selected_weights.weight_sum
                    ),
                    "exact_expansion_weight_total": _fraction_payload(
                        _expansion_weight_total(
                            record,
                            adapter=adapter,
                            player_ids=item.selected_weights.player_ids,
                            position_by_player_id=position_by_player_id,
                        )
                    ),
                    "selected_design_weights_sha256": (
                        item.selected_weights.design_sha256
                    ),
                    "balance_report_sha256": item.spec.balance_report_sha256,
                    "adapter_sha256": adapter.adapter_sha256,
                    "eligibility_sha256": item.eligibility_sha256,
                    "metric_contract_sha256": item.metric_contract_sha256,
                    "runtime_projection_sha256": (
                        record.runtime_projection_sha256
                    ),
                    "assignment_sha256": record.assignment_sha256,
                    "balance_sha256": record.balance.balance_sha256,
                    "exact_balance_passed": (
                        record.balance.exact_balance_passed
                    ),
                    "population_seed_record_sha256": (
                        item.population_seed_record_sha256
                    ),
                    "binding_sha256": item.binding_sha256,
                }
            )
    return {
        "seed": record.seed,
        "seed_decimal": str(record.seed),
        "seed_record_sha256": record.seed_record_sha256,
        "full_cohort": {
            "player_count": full_count,
            "player_count_decimal": str(full_count),
            "exact_analysis_weight_total": _fraction_payload(
                record.exact_weights.weight_sum
            ),
            "exact_expansion_weight_total": _fraction_payload(
                full_expansion_total
            ),
            "exact_target_population_total": _fraction_payload(
                Fraction(adapter.apportionment_plan.total_population_count, 1)
            ),
            "exact_weights_sha256": record.exact_weights.design_sha256,
            "adapter_sha256": adapter.adapter_sha256,
            "cell_identity": {
                "cell_count": len(adapter.cells),
                "domain_sha256": adapter.apportionment_plan.domain_sha256,
                "runtime_projection_id": adapter.runtime_projection_id,
                "runtime_projection_sha256": record.runtime_projection_sha256,
                "assignment_sha256": record.assignment_sha256,
                "ordered_player_ids_sha256": record.ordered_player_ids_sha256,
            },
            "pre_treatment_balance": {
                "schema_version": record.balance.schema_version,
                "balance_sha256": record.balance.balance_sha256,
                "runtime_membership_sha256": (
                    record.balance.runtime_membership.membership_sha256
                ),
                "exact_balance_passed": record.balance.exact_balance_passed,
            },
        },
        "selected_profiles_available": bool(selected_profiles),
        "selected_profiles": selected_profiles,
    }


def _expansion_weight_total(
    record,
    *,
    adapter,
    player_ids: tuple[int, ...],
    position_by_player_id: dict[int, int],
) -> Fraction:
    """Derive an exact expansion total from adapter cells and seed indices."""

    total = Fraction(0, 1)
    for player_id in player_ids:
        try:
            position = position_by_player_id[player_id]
            cell_index = record.cell_indices[position]
            expansion_weight = adapter.cells[cell_index].expansion_weight
        except (KeyError, IndexError) as exc:
            raise ValueError(
                "selected population cannot be mapped to exact expansion weights"
            ) from exc
        total += Fraction(*expansion_weight)
    return total


def _fraction_payload(value: Fraction) -> dict[str, object]:
    if type(value) is not Fraction:
        raise TypeError("exact population total must be fractions.Fraction")
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "numerator_decimal": str(value.numerator),
        "denominator_decimal": str(value.denominator),
    }


def _positive_fraction_payload(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        numerator = int(str(value["numerator_decimal"]))
        denominator = int(str(value["denominator_decimal"]))
    except (KeyError, TypeError, ValueError):
        return False
    return numerator > 0 and denominator > 0


def _validate_analysis_composition(
    config: PolicyPrototypeConfig,
    batch: PolicyBatchResult,
    *,
    analysis_plan: LoadedProspectiveAnalysisPlan | None,
    analysis_binding: RunAnalysisBinding | None,
) -> tuple[LoadedProspectiveAnalysisPlan | None, RunAnalysisBinding | None]:
    """Re-attest optional plan files and recompute bindings before export."""

    selected = config.analysis_plan
    supplied = (analysis_plan is not None, analysis_binding is not None)
    if selected is None:
        if any(supplied):
            raise ValueError(
                "analysis plan/binding supplied without a configuration selection"
            )
        return None, None
    if supplied != (True, True):
        raise ValueError(
            "configured analysis plan requires both loaded plan and run binding"
        )
    if type(analysis_plan) is not LoadedProspectiveAnalysisPlan:
        raise TypeError(
            "analysis_plan must be LoadedProspectiveAnalysisPlan or None"
        )
    if type(analysis_binding) is not RunAnalysisBinding:
        raise TypeError("analysis_binding must be RunAnalysisBinding or None")
    verified = verify_loaded_prospective_analysis_plan(analysis_plan)
    if verified.plan_path != selected.plan_path:
        raise ValueError(
            "loaded analysis plan path does not match the configuration selection"
        )
    observed_binding = resolve_run_analysis_binding(verified.plan, batch)
    if observed_binding != analysis_binding:
        raise ValueError(
            "analysis binding does not match the re-attested plan and batch"
        )
    return verified, observed_binding


def _validate_config_matches_batch(
    config: PolicyPrototypeConfig,
    batch: PolicyBatchResult,
    configured_run_inputs: PolicyRunInputs,
) -> None:
    """Reject metadata claims that differ from the execution truth."""

    if config.batch != batch.spec:
        raise ValueError(
            "configuration batch specification does not match the executed batch"
        )
    if configured_run_inputs != batch.run_inputs:
        raise ValueError(
            "configuration model inputs do not match the executed batch"
        )
    selection = config.population
    lineage = batch.population_execution_lineage
    if (selection is None) != (lineage is None):
        raise ValueError(
            "configuration population selection does not match executed batch"
        )
    if selection is not None and lineage is not None:
        adapter = lineage.adapter
        if (
            adapter.adapter_id != selection.adapter_id
            or adapter.verification.bundle.bundle_path
            != selection.design_bundle_path
            or adapter.mapping_bundle.mapping_path
            != selection.runtime_mapping_bundle_path
        ):
            raise ValueError(
                "configuration population files/adapter id do not match execution"
            )


def _effective_config_snapshot(
    config: PolicyPrototypeConfig,
    run_inputs: PolicyRunInputs,
) -> dict[str, object]:
    payload = {
        "meta": {
            "name": config.name,
            "run_purpose": config.run_purpose.value,
            "provenance_status": config.provenance_status,
            "notes": config.notes,
        },
        "batch_spec": config.batch.snapshot(),
        "model_inputs": run_inputs.snapshot(),
        "output": {
            "output_dir": str(config.output.output_dir),
            "histogram_bins": config.output.histogram_bins,
            "include_player_rows": config.output.include_player_rows,
            "run_sensitivity": config.output.run_sensitivity,
        },
    }
    if config.population is not None:
        payload["population"] = config.population.snapshot()
    if config.analysis_plan is not None:
        payload["analysis_plan"] = config.analysis_plan.snapshot()
    return payload


def _canonical_sha256(snapshot: dict[str, object]) -> str:
    encoded = json.dumps(
        snapshot,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _money_outputs_cross_country_comparable(
    profile_inputs: dict[str, object],
    *,
    profile_lineage: ProfileInputLineage | None = None,
) -> bool:
    """Require every independent evidence and design gate."""

    return all(
        (
            _money_conversion_structure_coherent(profile_inputs),
            _source_rate_evidence_is_bound(
                profile_inputs,
                profile_lineage=profile_lineage,
            ),
            _source_bundle_signature_is_bound(profile_inputs),
            _legacy_root_monetary_output_design_is_bound(profile_inputs),
            _monetary_population_binding_is_bound(
                profile_inputs,
                profile_lineage=profile_lineage,
            ),
            _monetary_preregistration_is_bound(profile_inputs),
        )
    )


def _monetary_comparability_payload(
    profile_inputs: dict[str, object],
    *,
    profile_lineage: ProfileInputLineage | None = None,
) -> dict[str, object]:
    """Publish typed reason codes plus non-self-promotable manifest gates."""

    assessment = (
        _serialized_monetary_evidence_assessment(profile_inputs)
        if _profile_lineage_reattests_payload(
            profile_inputs,
            profile_lineage=profile_lineage,
        )
        else None
    )
    if assessment is None:
        assessment = {
            "structure_coherent": False,
            "source_rate_evidence_bound": False,
            "source_bundle_signature_bound": False,
            "output_design_binding_bound": False,
            "population_binding_bound": False,
            "preregistration_bound": False,
            "public_output_comparability": False,
            "blockers": [
                "monetary_conversion.structure=unavailable",
                "monetary_conversion.source_rate_binding=missing",
                "monetary_conversion.source_bundle_signature=missing",
                "monetary_conversion.output_design_binding=missing",
                "monetary_conversion.population_binding=missing",
                "monetary_conversion.preregistration_binding=missing",
            ],
        }
    return {
        "typed_assessment": dict(assessment),
        "manifest_gate": {
            "structure_coherent": _money_conversion_structure_coherent(
                profile_inputs
            ),
            "source_rate_evidence_bound": _source_rate_evidence_is_bound(
                profile_inputs,
                profile_lineage=profile_lineage,
            ),
            "source_bundle_signature_bound": _source_bundle_signature_is_bound(
                profile_inputs
            ),
            "output_design_binding_bound": _legacy_root_monetary_output_design_is_bound(
                profile_inputs
            ),
            "population_binding_bound": _monetary_population_binding_is_bound(
                profile_inputs,
                profile_lineage=profile_lineage,
            ),
            "preregistration_bound": _monetary_preregistration_is_bound(
                profile_inputs
            ),
            "public_output_comparability": _money_outputs_cross_country_comparable(
                profile_inputs,
                profile_lineage=profile_lineage,
            ),
        },
    }


def _population_readiness_payload(
    profile_inputs: dict[str, object],
    *,
    profile_lineage: ProfileInputLineage | None = None,
) -> dict[str, object]:
    """Publish population subgates only from re-attested lineage."""

    assessment = (
        _serialized_population_evidence_assessment(profile_inputs)
        if _profile_lineage_reattests_payload(
            profile_inputs,
            profile_lineage=profile_lineage,
        )
        else None
    )
    if assessment is None:
        assessment = {
            "structure_coherent": False,
            "source_population_evidence_bound": False,
            "calibration_targets_bound": False,
            "heldout_validation_targets_bound": False,
            "source_bundle_signature_bound": False,
            "sampling_plan_bound": False,
            "runtime_projection_bound": False,
            "output_estimand_binding_bound": False,
            "balance_validation_bound": False,
            "public_population_comparability": False,
            "blockers": [
                "population.structure=unavailable",
                "population.source_evidence=missing",
                "population.calibration_targets=missing",
                "population.heldout_validation_targets=missing",
                "population.source_bundle_signature=missing",
                "population.sampling_plan=missing",
                "population.runtime_projection=missing",
                "population.output_estimand_binding=missing",
                "population.balance_validation=missing",
            ],
        }
    gate_fields = (
        "structure_coherent",
        "source_population_evidence_bound",
        "calibration_targets_bound",
        "heldout_validation_targets_bound",
        "source_bundle_signature_bound",
        "sampling_plan_bound",
        "runtime_projection_bound",
        "output_estimand_binding_bound",
        "balance_validation_bound",
        "public_population_comparability",
    )
    return {
        "schema_version": "1.0",
        "typed_assessment": dict(assessment),
        "manifest_gate": {
            field: assessment[field] is True for field in gate_fields
        },
    }


def _money_conversion_structure_coherent(
    profile_inputs: dict[str, object],
) -> bool:
    """Apply the typed structural validator plus calibrated scale gates."""

    if profile_inputs.get("lineage_status") != "registered_profile_bundle":
        return False
    snapshot = profile_inputs.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    bundle = snapshot.get("profile_bundle")
    if not isinstance(bundle, dict):
        return False
    try:
        coherent, _blockers = monetary_structure_assessment_from_snapshot(
            bundle
        )
    except ProfileValidationError:
        return False
    if not coherent:
        return False
    scales = bundle.get("money_scales")
    if not isinstance(scales, list):
        return False
    return all(
        isinstance(scale, dict)
        and scale.get("anchor_status") == "CALIBRATED"
        and scale.get("scale_status") == "CALIBRATED"
        for scale in scales
    )


def _source_rate_evidence_is_bound(
    profile_inputs: dict[str, object],
    *,
    profile_lineage: ProfileInputLineage | None = None,
) -> bool:
    """Accept only the schema-v3, re-attested typed extraction subgate."""

    if not _profile_lineage_reattests_payload(
        profile_inputs,
        profile_lineage=profile_lineage,
    ):
        return False
    assessment = _serialized_monetary_evidence_assessment(profile_inputs)
    return bool(
        assessment is not None
        and assessment["source_rate_evidence_bound"] is True
    )


def _profile_lineage_reattests_payload(
    profile_inputs: dict[str, object],
    *,
    profile_lineage: ProfileInputLineage | None,
) -> bool:
    if profile_lineage is None:
        return False
    try:
        return profile_lineage.manifest_payload() == profile_inputs
    except ProfileValidationError:
        return False


def _source_bundle_signature_is_bound(
    profile_inputs: dict[str, object],
) -> bool:
    """Schema-v1 source bundles are explicitly missing a trust-root signature."""

    # Do not accept a serialized true value until a future schema implements
    # detached-signature verification against an external trust root.
    _ = profile_inputs
    return False


def _legacy_root_monetary_output_design_is_bound(
    profile_inputs: dict[str, object],
) -> bool:
    """The legacy root output profile remains raw simulation cents.

    Prospective monetary execution is a separate nested output profile and must
    never promote this global/root gate.
    """

    _ = profile_inputs
    return False


def _monetary_population_binding_is_bound(
    profile_inputs: dict[str, object],
    *,
    profile_lineage: ProfileInputLineage | None = None,
) -> bool:
    """Require the independently validated population-readiness contract."""

    if not _profile_lineage_reattests_payload(
        profile_inputs,
        profile_lineage=profile_lineage,
    ):
        return False
    assessment = _serialized_population_evidence_assessment(profile_inputs)
    return bool(
        assessment is not None
        and assessment["public_population_comparability"] is True
    )


def _monetary_preregistration_is_bound(
    profile_inputs: dict[str, object],
) -> bool:
    """The current causal design is retrospective and not preregistered."""

    _ = profile_inputs
    return False


def _serialized_monetary_evidence_assessment(
    profile_inputs: dict[str, object],
) -> dict[str, object] | None:
    """Validate the exact typed assessment mirrored by registered lineage."""

    if profile_inputs.get("lineage_status") != "registered_profile_bundle":
        return None
    snapshot = profile_inputs.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") not in {
        3,
        4,
    }:
        return None
    bundle = snapshot.get("profile_bundle")
    published = profile_inputs.get("monetary_evidence_assessment")
    if not isinstance(bundle, dict):
        return None
    assessment = bundle.get("monetary_evidence_assessment")
    if (
        not isinstance(assessment, dict)
        or published != assessment
    ):
        return None
    try:
        monetary_evidence_assessment_from_snapshot(
            assessment,
            registered_lineage=True,
            bundle_snapshot=bundle,
        )
    except ProfileValidationError:
        return None
    return assessment


def _serialized_population_evidence_assessment(
    profile_inputs: dict[str, object],
) -> dict[str, object] | None:
    """Validate the exact typed population assessment mirrored by lineage."""

    if profile_inputs.get("lineage_status") != "registered_profile_bundle":
        return None
    snapshot = profile_inputs.get("snapshot")
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 4:
        return None
    bundle = snapshot.get("profile_bundle")
    published = profile_inputs.get("population_evidence_assessment")
    if not isinstance(bundle, dict):
        return None
    assessment = bundle.get("population_evidence_assessment")
    if not isinstance(assessment, dict) or published != assessment:
        return None
    try:
        population_evidence_assessment_from_snapshot(
            assessment,
            registered_lineage=True,
            bundle_snapshot=bundle,
        )
    except ProfileValidationError:
        return None
    return assessment


def _profile_dependencies_calibrated(
    profile_inputs: dict[str, object],
    *,
    profile_lineage: ProfileInputLineage | None = None,
) -> bool:
    """Mirror the transitive profile campaign gate from fingerprinted values."""

    if profile_inputs.get("lineage_status") != "registered_profile_bundle":
        return False
    snapshot = profile_inputs.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    bundle = snapshot.get("profile_bundle")
    profiles = snapshot.get("country_profiles")
    if not isinstance(bundle, dict) or not isinstance(profiles, list):
        return False
    if bundle.get("profile_status") != "CALIBRATED":
        return False
    sources = bundle.get("sources")
    contracts = bundle.get("metric_contracts")
    scales = bundle.get("money_scales")
    conversions = bundle.get("monetary_conversions")
    population_bundle = bundle.get("population_evidence_bundle")
    if not all(
        isinstance(rows, list) and rows
        for rows in (sources, contracts, scales, conversions)
    ):
        return False
    assert isinstance(sources, list)
    assert isinstance(contracts, list)
    assert isinstance(scales, list)
    assert isinstance(conversions, list)
    if any(
        not isinstance(contract, dict) or contract.get("status") != "CALIBRATED"
        for contract in contracts
    ):
        return False
    if any(
        not isinstance(scale, dict)
        or scale.get("anchor_status") != "CALIBRATED"
        or scale.get("scale_status") != "CALIBRATED"
        for scale in scales
    ):
        return False
    if any(
        not isinstance(conversion, dict)
        or conversion.get("status") != "CALIBRATED"
        for conversion in conversions
    ):
        return False

    referenced_source_ids: set[str] = set()
    for row in (*profiles, *contracts, *scales, *conversions):
        if not isinstance(row, dict):
            return False
        source_ids = row.get("source_ids")
        if not isinstance(source_ids, list) or any(
            not isinstance(source_id, str) for source_id in source_ids
        ):
            return False
        referenced_source_ids.update(source_ids)
    if isinstance(population_bundle, dict):
        population_bindings = population_bundle.get("bindings")
        if not isinstance(population_bindings, list):
            return False
        for binding in population_bindings:
            if not isinstance(binding, dict):
                return False
            source_ids = binding.get("source_ids")
            if not isinstance(source_ids, list) or any(
                not isinstance(source_id, str) for source_id in source_ids
            ):
                return False
            referenced_source_ids.update(source_ids)
    source_statuses = {
        source.get("id"): source.get("calibration_status")
        for source in sources
        if isinstance(source, dict)
    }
    population_assessment = _serialized_population_evidence_assessment(
        profile_inputs
    )
    return (
        bool(referenced_source_ids)
        and all(
            source_statuses.get(source_id) == "CALIBRATED"
            for source_id in referenced_source_ids
        )
        and _money_outputs_cross_country_comparable(
            profile_inputs,
            profile_lineage=profile_lineage,
        )
        and population_assessment is not None
        and population_assessment["public_population_comparability"] is True
    )


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repository: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit or None, bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


__all__ = ["build_run_manifest"]
