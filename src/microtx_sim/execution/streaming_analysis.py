"""Memory-bounded exact analysis binding over checkpointed seed blocks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Mapping

import numpy as np

from ..causal.analysis_binding import (
    ANALYSIS_BINDING_SCHEMA_VERSION,
    SeedAnalysisBinding,
    _CAMPAIGN_BLOCKERS as _ANALYSIS_BLOCKERS,
    _metric_contract_registry_sha256,
    _plan_output_profile_id,
    _plan_output_profile_sha256,
    _projected_cells,
    _resolve_plan_monetary_bases,
    _resolve_seed_estimand,
    validate_analysis_plan_inputs,
)
from ..causal.analysis_plan import (
    PlannedPopulationEstimand,
    ProspectiveAnalysisPlan,
    analysis_plan_harm_weights_sha256,
)
from ..causal.batch import (
    PolicyBatchCheckpoint,
    policy_run_input_sha256,
)
from ..causal.design import assess_causal_design
from ..data.population_execution import (
    POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION,
    PopulationSeedExecutionRecord,
    _CAMPAIGN_BLOCKERS as _POPULATION_BLOCKERS,
    population_execution_input_sha256,
    population_execution_input_snapshot,
)
from ..data.lineage import profile_lineage_fingerprint_matches
from ..metrics.population_estimands import PopulationEstimandResult
from .optimized_runner import CheckpointedPolicyBatch


@dataclass(frozen=True, slots=True)
class CompactSeedAnalysisBinding:
    """All final-output fields retained without player-sized weight tuples."""

    seed: int
    planned_estimand: PlannedPopulationEstimand
    result: PopulationEstimandResult
    cpu_reference_result: PopulationEstimandResult
    continuous_absolute_difference: float
    continuous_parity_within_tolerance: bool
    estimand_direction_matches_cpu_reference: bool
    selected_player_count: int
    population_weights_sha256: str
    population_seed_record_sha256: str
    binding_sha256: str
    pretreatment_cohort_sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointedRunAnalysisBinding:
    """Content-addressed run binding with compact per-seed projections."""

    schema_version: str
    plan: ProspectiveAnalysisPlan
    causal_design_sha256: str
    batch_spec_sha256: str
    model_inputs_sha256: str
    policy_run_input_sha256: str
    population_input_sha256: str
    profile_input_sha256: str
    population_lineage_sha256: str
    metric_contract_registry_sha256: str
    harm_weights_sha256: str
    output_profile_schema_sha256: str
    seeds: tuple[int, ...]
    seed_bindings: tuple[CompactSeedAnalysisBinding, ...]
    binding_sha256: str
    scenario_diagnostic_rows: tuple[Mapping[str, object], ...]
    backend_identity_sha256: str
    preregistered: bool = False
    campaign_ready: bool = False


def resolve_checkpointed_run_analysis_binding(
    plan: ProspectiveAnalysisPlan,
    batch: CheckpointedPolicyBatch,
) -> CheckpointedRunAnalysisBinding:
    """Resolve the exact plan while retaining at most one seed's arrays."""

    if type(plan) is not ProspectiveAnalysisPlan:
        raise TypeError("plan must be ProspectiveAnalysisPlan")
    ProspectiveAnalysisPlan.__post_init__(plan)
    if plan.stopping_rule.seeds != batch.spec.seeds:
        raise ValueError("analysis plan seeds differ from checkpoint work plan")
    validate_analysis_plan_inputs(
        plan,
        batch_spec=batch.spec,
        run_inputs=batch.run_inputs,
        population_adapter=batch.population_adapter,
        profile_input_lineage=batch.profile_input_lineage,
    )
    population_input_sha256 = population_execution_input_sha256(
        batch.population_adapter
    )
    if population_input_sha256 != plan.expected_population_input_sha256:
        raise ValueError("checkpoint population input differs from analysis plan")
    if not profile_lineage_fingerprint_matches(
        plan.expected_profile_input_sha256,
        batch.profile_input_lineage.fingerprint_sha256,
    ):
        raise ValueError("checkpoint profile input differs from analysis plan")

    # First pass hashes only deterministic population attestations.  It never
    # loads a scenario payload and retains only one seed record at a time.
    population_payload = {
        "schema_version": POPULATION_EXECUTION_LINEAGE_SCHEMA_VERSION,
        "mode": "projected_v1",
        "input_sha256": population_input_sha256,
        "input": population_execution_input_snapshot(batch.population_adapter),
        "seeds": list(batch.spec.seeds),
        "seed_decimal_strings": [str(seed) for seed in batch.spec.seeds],
        "seed_records": None,
        "campaign_ready": False,
        "public_population_comparability": False,
        "campaign_blockers": list(_POPULATION_BLOCKERS),
    }
    population_stream = _CanonicalArrayHash(
        population_payload,
        array_key="seed_records",
    )
    population_record_sha256: dict[int, str] = {}
    for seed in batch.spec.seeds:
        record = batch.population_seed_record(
            seed,
            expected_cohort_digest=None,
        )
        PopulationSeedExecutionRecord.__post_init__(record)
        population_record_sha256[seed] = record.seed_record_sha256
        population_stream.append(record.snapshot())
    population_lineage_sha256 = population_stream.finish()

    projected_cells = _projected_cells(
        batch.population_adapter,
        batch.country_profiles,
    )
    jurisdiction_codes = tuple(
        profile.code for profile in batch.country_profiles
    )
    monetary_bases = _resolve_plan_monetary_bases(
        plan,
        profile_input_lineage=batch.profile_input_lineage,
        jurisdiction_codes=jurisdiction_codes,
    )
    causal_design_sha256 = assess_causal_design(
        batch.spec.scenarios
    ).design_sha256()
    batch_spec_sha256 = batch.spec.snapshot_sha256()
    model_inputs_sha256 = batch.run_inputs.snapshot_sha256()
    profile_input_sha256 = batch.profile_input_lineage.fingerprint_sha256
    metric_registry_sha256 = _metric_contract_registry_sha256()
    harm_weights_sha256 = analysis_plan_harm_weights_sha256(
        batch.run_inputs.harm_weights
    )
    output_profile_sha256 = _plan_output_profile_sha256(plan)

    run_payload = {
        "schema_version": ANALYSIS_BINDING_SCHEMA_VERSION,
        "plan_id": plan.plan_id,
        "plan_sha256": plan.plan_sha256,
        "causal_design_sha256": causal_design_sha256,
        "batch_spec_sha256": batch_spec_sha256,
        "model_inputs_sha256": model_inputs_sha256,
        "population_input_sha256": population_input_sha256,
        "profile_input_sha256": profile_input_sha256,
        "population_lineage_sha256": population_lineage_sha256,
        "metric_contract_registry_sha256": metric_registry_sha256,
        "harm_weights_sha256": harm_weights_sha256,
        "output_profile_id": _plan_output_profile_id(plan),
        "output_profile_schema_sha256": output_profile_sha256,
        "seeds": list(batch.spec.seeds),
        "seed_decimal_strings": [str(seed) for seed in batch.spec.seeds],
        "monetary_output_bases": [
            basis.snapshot()
            for basis in sorted(
                {item.basis_sha256: item for item in monetary_bases.values()}.values(),
                key=lambda item: item.basis_sha256,
            )
        ],
        "seed_bindings": None,
        "preregistered": False,
        "campaign_ready": False,
        "campaign_blockers": list(_ANALYSIS_BLOCKERS),
    }
    compact_bindings: list[CompactSeedAnalysisBinding] = []
    scenario_rows: list[Mapping[str, object]] = []
    run_stream = _CanonicalArrayHash(run_payload, array_key="seed_bindings")
    for seed_batch in batch.iter_seed_batches():
        seed = seed_batch.spec.seeds[0]
        lineage = seed_batch.population_execution_lineage
        assert lineage is not None
        population_record = lineage.seed_records[0]
        if (
            population_record.seed_record_sha256
            != population_record_sha256[seed]
        ):
            raise ValueError(
                "population record changed between streaming attestation passes"
            )
        record_by_scenario = {
            record.result.scenario.scenario_id: record
            for record in seed_batch.records
        }
        checkpoint = PolicyBatchCheckpoint(
            spec=seed_batch.spec,
            completed_seeds=(seed,),
            records=seed_batch.records,
            cohort_digest_by_seed=seed_batch.cohort_digest_by_seed,
        )
        scenario_rows.extend(checkpoint.nonmonetary_diagnostic_rows())
        for planned in plan.estimands:
            reference = record_by_scenario[planned.reference_scenario_id].result
            comparison = record_by_scenario[planned.comparison_scenario_id].result
            resolved = _resolve_seed_estimand(
                planned,
                seed=seed,
                population_record=population_record,
                reference=reference,
                comparison=comparison,
                jurisdiction_codes=jurisdiction_codes,
                projected_cells=projected_cells,
                target_evidence_sha256=(
                    batch.population_adapter.calibration_target_sha256
                ),
                monetary_output_basis=monetary_bases.get(
                    planned.estimand_id
                ),
            )
            SeedAnalysisBinding.__post_init__(resolved)
            cpu_reference = _resolve_seed_estimand(
                planned,
                seed=seed,
                population_record=population_record,
                reference=replace(
                    reference,
                    composite_harm=reference.harm.composite_harm(
                        batch.run_inputs.harm_weights
                    ),
                ),
                comparison=replace(
                    comparison,
                    composite_harm=comparison.harm.composite_harm(
                        batch.run_inputs.harm_weights
                    ),
                ),
                jurisdiction_codes=jurisdiction_codes,
                projected_cells=projected_cells,
                target_evidence_sha256=(
                    batch.population_adapter.calibration_target_sha256
                ),
                monetary_output_basis=monetary_bases.get(
                    planned.estimand_id
                ),
            )
            SeedAnalysisBinding.__post_init__(cpu_reference)
            if (
                resolved.selected_player_count
                != cpu_reference.selected_player_count
                or resolved.selected_weights.design_sha256
                != cpu_reference.selected_weights.design_sha256
            ):
                raise ValueError(
                    "accelerated and CPU estimands selected different populations"
                )
            absolute_difference = abs(
                resolved.result.value - cpu_reference.result.value
            )
            within_tolerance = bool(
                np.isclose(
                    resolved.result.value,
                    cpu_reference.result.value,
                    atol=5e-13,
                    rtol=5e-13,
                )
            )
            direction_equal = (
                np.sign(resolved.result.value)
                == np.sign(cpu_reference.result.value)
            )
            is_gpu = batch.store.identity.backend.resolved_backend == "gpu"
            if is_gpu and (not within_tolerance or not direction_equal):
                raise ValueError(
                    "GPU arithmetic changes the planned estimand beyond its "
                    "tolerance or direction"
                )
            if not is_gpu and (
                resolved.result.value != cpu_reference.result.value
                or resolved.result.result_sha256
                != cpu_reference.result.result_sha256
            ):
                raise ValueError(
                    "CPU execution is not bitwise-identical to its estimand reference"
                )
            run_stream.append(resolved.snapshot())
            compact_bindings.append(
                CompactSeedAnalysisBinding(
                    seed=seed,
                    planned_estimand=planned,
                    result=resolved.result,
                    cpu_reference_result=cpu_reference.result,
                    continuous_absolute_difference=absolute_difference,
                    continuous_parity_within_tolerance=within_tolerance,
                    estimand_direction_matches_cpu_reference=direction_equal,
                    selected_player_count=resolved.selected_player_count,
                    population_weights_sha256=(
                        resolved.selected_weights.design_sha256
                    ),
                    population_seed_record_sha256=(
                        resolved.population_seed_record_sha256
                    ),
                    binding_sha256=resolved.binding_sha256,
                    pretreatment_cohort_sha256=(
                        seed_batch.cohort_digest_by_seed[seed]
                    ),
                )
            )
        # ``seed_batch`` and the player-sized exact weights become unreachable
        # here before the next checkpoint block is decoded.

    binding_sha256 = run_stream.finish()
    return CheckpointedRunAnalysisBinding(
        schema_version=ANALYSIS_BINDING_SCHEMA_VERSION,
        plan=plan,
        causal_design_sha256=causal_design_sha256,
        batch_spec_sha256=batch_spec_sha256,
        model_inputs_sha256=model_inputs_sha256,
        policy_run_input_sha256=policy_run_input_sha256(
            batch_spec=batch.spec,
            run_inputs=batch.run_inputs,
            profile_input_fingerprint_sha256=profile_input_sha256,
            population_adapter=batch.population_adapter,
        ),
        population_input_sha256=population_input_sha256,
        profile_input_sha256=profile_input_sha256,
        population_lineage_sha256=population_lineage_sha256,
        metric_contract_registry_sha256=metric_registry_sha256,
        harm_weights_sha256=harm_weights_sha256,
        output_profile_schema_sha256=output_profile_sha256,
        seeds=batch.spec.seeds,
        seed_bindings=tuple(compact_bindings),
        binding_sha256=binding_sha256,
        scenario_diagnostic_rows=tuple(scenario_rows),
        backend_identity_sha256=(
            batch.store.identity.backend.identity_sha256
        ),
    )


class _CanonicalArrayHash:
    """Incrementally hash one array field exactly like canonical JSON."""

    def __init__(self, payload: Mapping[str, object], *, array_key: str) -> None:
        if array_key not in payload or payload[array_key] is not None:
            raise ValueError("streamed canonical array requires a None placeholder")
        self._digest = sha256()
        self._array_key = array_key
        self._keys = sorted(payload)
        self._payload = dict(payload)
        self._index = self._keys.index(array_key)
        self._count = 0
        self._finished = False
        self._digest.update(b"{")
        for index, key in enumerate(self._keys[: self._index]):
            if index:
                self._digest.update(b",")
            self._digest.update(_canonical_json_bytes(key))
            self._digest.update(b":")
            self._digest.update(_canonical_json_bytes(self._payload[key]))
        if self._index:
            self._digest.update(b",")
        self._digest.update(_canonical_json_bytes(array_key))
        self._digest.update(b":[")

    def append(self, value: object) -> None:
        if self._finished:
            raise RuntimeError("canonical array hash is already finished")
        if self._count:
            self._digest.update(b",")
        self._digest.update(_canonical_json_bytes(value))
        self._count += 1

    def finish(self) -> str:
        if self._finished:
            raise RuntimeError("canonical array hash is already finished")
        self._digest.update(b"]")
        for key in self._keys[self._index + 1 :]:
            self._digest.update(b",")
            self._digest.update(_canonical_json_bytes(key))
            self._digest.update(b":")
            self._digest.update(_canonical_json_bytes(self._payload[key]))
        self._digest.update(b"}")
        self._finished = True
        return self._digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "CheckpointedRunAnalysisBinding",
    "CompactSeedAnalysisBinding",
    "resolve_checkpointed_run_analysis_binding",
]
