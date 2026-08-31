"""Lossless, JSON-compatible checkpoint codec for policy result records.

The scientific arrays are encoded from canonical little-endian bytes and
compressed before base64 transport.  Decoding reconstructs the strict model
dataclasses, so a resumed result passes the same validation as a freshly
computed result.  No pickle or executable object deserialization is used.
"""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import fields
from hashlib import sha256
from typing import Mapping
import zlib

import numpy as np

from ..causal.batch import PolicyBatchSpec, SeedScenarioRecord
from ..causal.scenarios import ScenarioId
from ..funding import EPGCResult
from ..metrics.harm import WelfareHarmResult
from ..simulation.policy_orchestrator import PolicyScenarioResult


RESULT_CODEC_SCHEMA_VERSION = "microtx_sim.policy_result_checkpoint.v1"

_RESULT_ARRAY_DTYPES = {
    "player_ids": np.dtype(np.int64),
    "is_minor": np.dtype(np.bool_),
    "age_years": np.dtype(np.int16),
    "jurisdiction": np.dtype(np.int16),
    "baseline_vulnerability": np.dtype(np.float32),
    "disposable_budget_cents": np.dtype(np.int64),
    "spending_cents": np.dtype(np.int64),
    "composite_harm": np.dtype(np.float64),
    "enjoyment": np.dtype(np.float64),
    "high_risk": np.dtype(np.bool_),
    "action_minutes": np.dtype(np.int64),
}

_HARM_ARRAY_DTYPES = {
    "component_scores": np.dtype(np.float64),
    "harmful_spending_cents": np.dtype(np.int64),
    "unplanned_spending_cents": np.dtype(np.int64),
    "monetary_harm_proxy_cents": np.dtype(np.int64),
    "opportunity_cost_proxy_cents": np.dtype(np.int64),
    "adult_opportunity_cost_proxy_cents": np.dtype(np.int64),
    "youth_opportunity_cost_proxy_cents": np.dtype(np.int64),
    "total_monetary_proxy_cents": np.dtype(np.int64),
    "excess_play_minutes": np.dtype(np.float64),
    "displaced_sleep_minutes": np.dtype(np.float64),
    "displaced_work_study_minutes": np.dtype(np.float64),
    "displaced_social_minutes": np.dtype(np.float64),
    "displaced_physical_activity_minutes": np.dtype(np.float64),
}


class ResultCheckpointCodecError(ValueError):
    """Raised when a result checkpoint payload is malformed or incompatible."""


def encode_seed_scenario_record(
    record: SeedScenarioRecord,
    *,
    batch_spec: PolicyBatchSpec,
) -> dict[str, object]:
    """Return a lossless canonical-JSON-compatible result payload."""

    if type(record) is not SeedScenarioRecord:
        raise TypeError("record must be SeedScenarioRecord")
    if type(batch_spec) is not PolicyBatchSpec:
        raise TypeError("batch_spec must be PolicyBatchSpec")
    result = record.result
    if result.seed not in batch_spec.seeds:
        raise ResultCheckpointCodecError("record seed is outside the batch design")
    scenario = next(
        (
            item
            for item in batch_spec.scenarios
            if item.scenario_id is result.scenario.scenario_id
        ),
        None,
    )
    if scenario is None or scenario != result.scenario:
        raise ResultCheckpointCodecError(
            "record scenario does not exactly match the batch design"
        )
    arrays = {
        name: _encode_array(getattr(result, name), expected_dtype=dtype)
        for name, dtype in _RESULT_ARRAY_DTYPES.items()
    }
    harm_arrays = {
        name: _encode_array(getattr(result.harm, name), expected_dtype=dtype)
        for name, dtype in _HARM_ARRAY_DTYPES.items()
    }
    epgc = (
        {
            item.name: getattr(result.epgc, item.name)
            for item in fields(EPGCResult)
        }
        if result.epgc is not None
        else None
    )
    return {
        "schema_version": RESULT_CODEC_SCHEMA_VERSION,
        "batch_spec_sha256": batch_spec.snapshot_sha256(),
        "scenario_id": result.scenario.scenario_id.value,
        "seed": result.seed,
        "days": result.days,
        "cohort_digest": record.cohort_digest,
        "monetary_semantics": {
            "internal_unit": "simulation_cents",
            "interpretation": "INTERNAL_MODEL_UNIT_NOT_REAL_MONEY",
            "observed_real_world_spending": False,
            "raw_cross_country_pooling": "PROHIBITED",
        },
        "effects_vs_safe": {
            "mean_harm": record.mean_harm_effect_vs_safe,
            "total_spending_cents": (
                record.total_spending_effect_vs_safe_cents
            ),
            "harmful_spending_cents": (
                record.harmful_spending_effect_vs_safe_cents
            ),
            "total_revenue_cents": (
                record.total_revenue_effect_vs_safe_cents
            ),
        },
        "arrays": arrays,
        "harm_arrays": harm_arrays,
        "revenue_composition_cents": dict(result.revenue_composition_cents),
        "total_revenue_cents": result.total_revenue_cents,
        "producer_cost_cents": result.producer_cost_cents,
        "producer_profit_cents": result.producer_profit_cents,
        "epgc": epgc,
    }


def decode_seed_scenario_record(
    payload: Mapping[str, object],
    *,
    batch_spec: PolicyBatchSpec,
) -> SeedScenarioRecord:
    """Reconstruct and strictly validate a checkpointed result record."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if type(batch_spec) is not PolicyBatchSpec:
        raise TypeError("batch_spec must be PolicyBatchSpec")
    expected_fields = {
        "schema_version",
        "batch_spec_sha256",
        "scenario_id",
        "seed",
        "days",
        "cohort_digest",
        "monetary_semantics",
        "effects_vs_safe",
        "arrays",
        "harm_arrays",
        "revenue_composition_cents",
        "total_revenue_cents",
        "producer_cost_cents",
        "producer_profit_cents",
        "epgc",
    }
    if set(payload) != expected_fields:
        raise ResultCheckpointCodecError(
            "checkpoint result field set differs from the strict schema"
        )
    if payload.get("schema_version") != RESULT_CODEC_SCHEMA_VERSION:
        raise ResultCheckpointCodecError("unsupported result checkpoint schema")
    if payload.get("batch_spec_sha256") != batch_spec.snapshot_sha256():
        raise ResultCheckpointCodecError(
            "checkpoint result uses a different batch specification"
        )
    if payload.get("monetary_semantics") != {
        "internal_unit": "simulation_cents",
        "interpretation": "INTERNAL_MODEL_UNIT_NOT_REAL_MONEY",
        "observed_real_world_spending": False,
        "raw_cross_country_pooling": "PROHIBITED",
    }:
        raise ResultCheckpointCodecError(
            "checkpoint internal monetary semantics are missing or changed"
        )
    try:
        scenario_id = ScenarioId(_strict_string(payload, "scenario_id"))
    except ValueError as error:
        raise ResultCheckpointCodecError(
            "checkpoint scenario identifier is invalid"
        ) from error
    scenario = next(
        (item for item in batch_spec.scenarios if item.scenario_id is scenario_id),
        None,
    )
    if scenario is None:
        raise ResultCheckpointCodecError(
            "checkpoint scenario is outside the batch design"
        )
    seed = _strict_integer(payload, "seed")
    if seed not in batch_spec.seeds:
        raise ResultCheckpointCodecError("checkpoint seed is outside the batch design")
    days = _strict_integer(payload, "days")
    if days != batch_spec.days:
        raise ResultCheckpointCodecError("checkpoint policy horizon changed")

    arrays_payload = _mapping(payload, "arrays")
    if set(arrays_payload) != set(_RESULT_ARRAY_DTYPES):
        raise ResultCheckpointCodecError("checkpoint result array set is incomplete")
    result_arrays = {
        name: _decode_array(arrays_payload[name], expected_dtype=dtype)
        for name, dtype in _RESULT_ARRAY_DTYPES.items()
    }
    harm_payload = _mapping(payload, "harm_arrays")
    if set(harm_payload) != set(_HARM_ARRAY_DTYPES):
        raise ResultCheckpointCodecError("checkpoint harm array set is incomplete")
    harm_arrays = {
        name: _decode_array(harm_payload[name], expected_dtype=dtype)
        for name, dtype in _HARM_ARRAY_DTYPES.items()
    }
    harm = WelfareHarmResult(**harm_arrays)

    epgc_payload = payload.get("epgc")
    if epgc_payload is None:
        epgc = None
    else:
        if not isinstance(epgc_payload, Mapping):
            raise ResultCheckpointCodecError("checkpoint EPGC payload is invalid")
        expected_fields = {item.name for item in fields(EPGCResult)}
        if set(epgc_payload) != expected_fields:
            raise ResultCheckpointCodecError("checkpoint EPGC field set is incomplete")
        epgc = EPGCResult(**dict(epgc_payload))

    revenue = _mapping(payload, "revenue_composition_cents")
    if any(type(key) is not str or type(value) is not int for key, value in revenue.items()):
        raise ResultCheckpointCodecError(
            "checkpoint revenue composition must contain exact integer cents"
        )
    result = PolicyScenarioResult(
        scenario=scenario,
        seed=seed,
        days=days,
        **result_arrays,
        harm=harm,
        revenue_composition_cents=dict(revenue),
        total_revenue_cents=_strict_integer(payload, "total_revenue_cents"),
        producer_cost_cents=_strict_integer(payload, "producer_cost_cents"),
        producer_profit_cents=_strict_integer(payload, "producer_profit_cents"),
        epgc=epgc,
    )
    effects = _mapping(payload, "effects_vs_safe")
    if set(effects) != {
        "mean_harm",
        "total_spending_cents",
        "harmful_spending_cents",
        "total_revenue_cents",
    }:
        raise ResultCheckpointCodecError(
            "checkpoint effect field set differs from the strict schema"
        )
    mean_harm = effects.get("mean_harm")
    if isinstance(mean_harm, bool) or not isinstance(mean_harm, (int, float)):
        raise ResultCheckpointCodecError("checkpoint mean harm effect is invalid")
    return SeedScenarioRecord(
        result=result,
        cohort_digest=_strict_string(payload, "cohort_digest"),
        mean_harm_effect_vs_safe=float(mean_harm),
        total_spending_effect_vs_safe_cents=_strict_integer(
            effects, "total_spending_cents"
        ),
        harmful_spending_effect_vs_safe_cents=_strict_integer(
            effects, "harmful_spending_cents"
        ),
        total_revenue_effect_vs_safe_cents=_strict_integer(
            effects, "total_revenue_cents"
        ),
    )


def _encode_array(
    value: object,
    *,
    expected_dtype: np.dtype,
) -> dict[str, object]:
    if type(value) is not np.ndarray or value.dtype != expected_dtype:
        raise TypeError(f"checkpoint array must use dtype {expected_dtype}")
    canonical_dtype = _canonical_dtype(expected_dtype)
    array = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
    raw = array.tobytes(order="C")
    compressed = zlib.compress(raw, level=1)
    return {
        "encoding": "zlib-base64-canonical-little-endian-c-order-v1",
        "dtype": canonical_dtype.str,
        "shape": list(array.shape),
        "raw_byte_length": len(raw),
        "raw_sha256": sha256(raw).hexdigest(),
        "data": b64encode(compressed).decode("ascii"),
    }


def _decode_array(value: object, *, expected_dtype: np.dtype) -> np.ndarray:
    if not isinstance(value, Mapping):
        raise ResultCheckpointCodecError("checkpoint array payload is invalid")
    if set(value) != {
        "encoding",
        "dtype",
        "shape",
        "raw_byte_length",
        "raw_sha256",
        "data",
    }:
        raise ResultCheckpointCodecError(
            "checkpoint array field set differs from the strict schema"
        )
    canonical_dtype = _canonical_dtype(expected_dtype)
    if (
        value.get("encoding")
        != "zlib-base64-canonical-little-endian-c-order-v1"
        or value.get("dtype") != canonical_dtype.str
    ):
        raise ResultCheckpointCodecError("checkpoint array encoding or dtype changed")
    shape_value = value.get("shape")
    if not isinstance(shape_value, list) or any(
        type(item) is not int or item < 0 for item in shape_value
    ):
        raise ResultCheckpointCodecError("checkpoint array shape is invalid")
    shape = tuple(shape_value)
    expected_bytes = canonical_dtype.itemsize
    for dimension in shape:
        expected_bytes *= dimension
    if value.get("raw_byte_length") != expected_bytes:
        raise ResultCheckpointCodecError("checkpoint array byte length is inconsistent")
    encoded = value.get("data")
    if type(encoded) is not str:
        raise ResultCheckpointCodecError("checkpoint array data is invalid")
    try:
        compressed = b64decode(encoded, validate=True)
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_bytes + 1)
    except (ValueError, zlib.error) as error:
        raise ResultCheckpointCodecError(
            "checkpoint array data cannot be decoded"
        ) from error
    if (
        len(raw) != expected_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or sha256(raw).hexdigest() != value.get("raw_sha256")
    ):
        raise ResultCheckpointCodecError("checkpoint array checksum mismatch")
    return np.frombuffer(raw, dtype=canonical_dtype).copy().reshape(shape)


def _canonical_dtype(value: np.dtype) -> np.dtype:
    return value if value.byteorder == "|" else value.newbyteorder("<")


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    selected = value.get(name)
    if not isinstance(selected, Mapping):
        raise ResultCheckpointCodecError(f"checkpoint {name} must be a mapping")
    return selected


def _strict_integer(value: Mapping[str, object], name: str) -> int:
    selected = value.get(name)
    if type(selected) is not int:
        raise ResultCheckpointCodecError(f"checkpoint {name} must be an integer")
    return selected


def _strict_string(value: Mapping[str, object], name: str) -> str:
    selected = value.get(name)
    if type(selected) is not str or not selected:
        raise ResultCheckpointCodecError(f"checkpoint {name} must be text")
    return selected


__all__ = [
    "RESULT_CODEC_SCHEMA_VERSION",
    "ResultCheckpointCodecError",
    "decode_seed_scenario_record",
    "encode_seed_scenario_record",
]
