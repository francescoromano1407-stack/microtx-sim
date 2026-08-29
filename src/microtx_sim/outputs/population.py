"""Standalone exact target-population estimand output writer.

This module intentionally does not participate in the legacy output-v3 bundle.
It re-attests estimand declarations and exact results, preserves their upstream
identity declarations, renders both payloads before writing either file, and
publishes no calibration, empirical-validation, or campaign-readiness claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path

from ..metrics.population_estimands import (
    PopulationCurrencySemantics,
    PopulationEstimandResult,
    PopulationEstimandSpec,
    PopulationEstimandValidationError,
    PopulationInclusionRule,
    PopulationPeriodSemantics,
)
from .schema import (
    TARGET_POPULATION_ESTIMAND_COLUMNS,
    TARGET_POPULATION_ESTIMAND_METADATA_FIELDS,
    TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256,
    TARGET_POPULATION_OUTPUT_PROFILE,
    stamp_target_population_estimand_schema,
)
from .writers import _render_csv, _render_json, write_text_atomic


PopulationEstimandPair = tuple[PopulationEstimandSpec, PopulationEstimandResult]

_WRITER_METADATA_FIELDS = frozenset(TARGET_POPULATION_ESTIMAND_METADATA_FIELDS)


def write_target_population_estimands(
    output_dir: str | Path,
    estimands: Sequence[PopulationEstimandPair],
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, Path]:
    """Write one deterministic two-file exact-estimand profile.

    Input order is not semantic: records are emitted in ascending
    ``estimand_id`` order.  Repeated IDs or content hashes are rejected instead
    of being silently deduplicated.  Every declaration/result pair is rebuilt
    through its exact dataclass constructor before any destination is touched.

    Upstream SHA-256 values remain declarations copied from the exact estimand
    specs.  This writer does not resolve the referenced target evidence,
    weights, runtime projection, balance report, or metric contract.
    """

    attested_pairs = _prepare_pairs(estimands)
    rows = [_population_estimand_row(spec, result) for spec, result in attested_pairs]
    metadata_payload = _population_estimand_metadata(
        attested_pairs,
        metadata=metadata,
    )

    # Complete schema/value rendering is the preflight boundary: invalid CSV or
    # JSON content cannot create or replace either destination file.
    csv_text = _render_csv(
        rows,
        canonical_columns=TARGET_POPULATION_ESTIMAND_COLUMNS,
        allow_extra_columns=False,
    )
    metadata_text = _render_json(metadata_payload)

    destination = Path(output_dir)
    return {
        "estimands": write_text_atomic(
            destination / "target_population_estimands.csv",
            csv_text,
        ),
        "metadata": write_text_atomic(
            destination / "target_population_estimand_metadata.json",
            metadata_text,
        ),
    }


def _prepare_pairs(
    estimands: Sequence[PopulationEstimandPair],
) -> tuple[PopulationEstimandPair, ...]:
    if isinstance(estimands, (str, bytes, bytearray)) or not isinstance(
        estimands, Sequence
    ):
        raise TypeError("population estimands must be a sequence of exact pairs")
    if len(estimands) == 0:
        raise PopulationEstimandValidationError(
            "the target-population output requires at least one estimand pair"
        )

    attested: list[PopulationEstimandPair] = []
    for index, pair in enumerate(estimands):
        if type(pair) is not tuple or len(pair) != 2:
            raise TypeError(
                f"population estimand pair {index} must be an exact two-item tuple"
            )
        spec = _reattest_spec(pair[0])
        result = _reattest_result(pair[1])
        _validate_pair_binding(spec, result)
        if spec.output_profile_id != TARGET_POPULATION_OUTPUT_PROFILE:
            raise PopulationEstimandValidationError(
                "estimand output_profile_id conflicts with the target-population "
                "output profile"
            )
        if (
            spec.output_profile_schema_sha256
            != TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256
        ):
            raise PopulationEstimandValidationError(
                "estimand output_profile_schema_sha256 conflicts with the "
                "target-population output profile"
            )
        attested.append((spec, result))

    for attribute, label in (
        ("estimand_id", "estimand IDs"),
        ("estimand_sha256", "estimand SHA-256 identities"),
        ("result_sha256", "result SHA-256 identities"),
    ):
        if attribute == "result_sha256":
            values = [getattr(result, attribute) for _spec, result in attested]
        else:
            values = [getattr(spec, attribute) for spec, _result in attested]
        if len(set(values)) != len(values):
            raise PopulationEstimandValidationError(
                f"target-population output cannot contain duplicate {label}"
            )

    return tuple(sorted(attested, key=lambda pair: pair[0].estimand_id))


def _reattest_spec(value: object) -> PopulationEstimandSpec:
    if type(value) is not PopulationEstimandSpec:
        raise TypeError("population estimand spec must be PopulationEstimandSpec")
    spec = value
    inclusion_rule = spec.inclusion_rule
    if type(inclusion_rule) is not PopulationInclusionRule:
        raise TypeError("population inclusion rule must be PopulationInclusionRule")
    reattested_rule = PopulationInclusionRule(
        rule_id=inclusion_rule.rule_id,
        description=inclusion_rule.description,
        source_fields=inclusion_rule.source_fields,
        timing=inclusion_rule.timing,
        evidence_role=inclusion_rule.evidence_role,
    )
    period = spec.period
    if type(period) is not PopulationPeriodSemantics:
        raise TypeError("population estimand period must be PopulationPeriodSemantics")
    reattested_period = PopulationPeriodSemantics(
        period_start=period.period_start,
        period_end=period.period_end,
        description=period.description,
    )
    currency = spec.currency
    if currency is not None:
        if type(currency) is not PopulationCurrencySemantics:
            raise TypeError(
                "population estimand currency must be PopulationCurrencySemantics"
            )
        currency = PopulationCurrencySemantics(
            currency_code=currency.currency_code,
            minor_unit_name=currency.minor_unit_name,
            price_period_start=currency.price_period_start,
            price_period_end=currency.price_period_end,
            currency_basis_sha256=currency.currency_basis_sha256,
            rounding=currency.rounding,
        )
    return PopulationEstimandSpec(
        schema_version=spec.schema_version,
        estimand_id=spec.estimand_id,
        target_population_id=spec.target_population_id,
        target_evidence_sha256=spec.target_evidence_sha256,
        design_weights_sha256=spec.design_weights_sha256,
        runtime_projection_sha256=spec.runtime_projection_sha256,
        balance_report_sha256=spec.balance_report_sha256,
        metric_contract_sha256=spec.metric_contract_sha256,
        output_profile_id=spec.output_profile_id,
        output_profile_schema_sha256=spec.output_profile_schema_sha256,
        analysis_unit=spec.analysis_unit,
        inclusion_rule=reattested_rule,
        metric_name=spec.metric_name,
        metric_kind=spec.metric_kind,
        metric_scale=spec.metric_scale,
        contrast=spec.contrast,
        algorithm=spec.algorithm,
        normalization=spec.normalization,
        period=reattested_period,
        currency=currency,
        target_population_count=spec.target_population_count,
        quantile_probability_numerator=(
            spec.quantile_probability_numerator
        ),
        quantile_probability_denominator=(
            spec.quantile_probability_denominator
        ),
    )


def _reattest_result(value: object) -> PopulationEstimandResult:
    if type(value) is not PopulationEstimandResult:
        raise TypeError("population estimand result must be PopulationEstimandResult")
    result = value
    return PopulationEstimandResult(
        schema_version=result.schema_version,
        estimand_sha256=result.estimand_sha256,
        design_weights_sha256=result.design_weights_sha256,
        algorithm=result.algorithm,
        metric_name=result.metric_name,
        contrast=result.contrast,
        normalization=result.normalization,
        player_count=result.player_count,
        numerator=result.numerator,
        denominator=result.denominator,
        weight_sum_numerator=result.weight_sum_numerator,
        weight_sum_denominator=result.weight_sum_denominator,
        target_population_count=result.target_population_count,
        result_sha256=result.result_sha256,
    )


def _validate_pair_binding(
    spec: PopulationEstimandSpec,
    result: PopulationEstimandResult,
) -> None:
    expected = {
        "schema_version": spec.schema_version,
        "estimand_sha256": spec.estimand_sha256,
        "design_weights_sha256": spec.design_weights_sha256,
        "algorithm": spec.algorithm,
        "metric_name": spec.metric_name,
        "contrast": spec.contrast,
        "normalization": spec.normalization,
        "target_population_count": spec.target_population_count,
    }
    mismatches = sorted(
        name
        for name, expected_value in expected.items()
        if getattr(result, name) != expected_value
    )
    if mismatches:
        raise PopulationEstimandValidationError(
            "population estimand result does not re-attest its spec binding: "
            + ", ".join(mismatches)
        )


def _population_estimand_row(
    spec: PopulationEstimandSpec,
    result: PopulationEstimandResult,
) -> dict[str, object]:
    currency = spec.currency
    probability_numerator = spec.quantile_probability_numerator
    probability_denominator = spec.quantile_probability_denominator
    return {
        "estimand_id": spec.estimand_id,
        "estimand_sha256": spec.estimand_sha256,
        "result_sha256": result.result_sha256,
        "target_population_id": spec.target_population_id,
        "target_evidence_sha256": spec.target_evidence_sha256,
        "design_weights_sha256": spec.design_weights_sha256,
        "runtime_projection_sha256": spec.runtime_projection_sha256,
        "balance_report_sha256": spec.balance_report_sha256,
        "metric_contract_sha256": spec.metric_contract_sha256,
        "output_profile_id": spec.output_profile_id,
        "output_profile_schema_sha256": spec.output_profile_schema_sha256,
        "analysis_unit": spec.analysis_unit.value,
        "inclusion_rule_id": spec.inclusion_rule.rule_id,
        "inclusion_rule_sha256": _canonical_sha256(
            spec.inclusion_rule.snapshot()
        ),
        "metric_name": spec.metric_name,
        "metric_kind": spec.metric_kind.value,
        "metric_scale": spec.metric_scale.value,
        "contrast": spec.contrast.value,
        "algorithm": spec.algorithm.value,
        "normalization": spec.normalization.value,
        "period_start": spec.period.period_start.isoformat(),
        "period_end": spec.period.period_end.isoformat(),
        "currency_code": currency.currency_code if currency is not None else None,
        "currency_basis_sha256": (
            currency.currency_basis_sha256 if currency is not None else None
        ),
        "quantile_probability_numerator_decimal": (
            str(probability_numerator)
            if probability_numerator is not None
            else None
        ),
        "quantile_probability_denominator_decimal": (
            str(probability_denominator)
            if probability_denominator is not None
            else None
        ),
        "player_count_decimal": str(result.player_count),
        "numerator_decimal": str(result.numerator),
        "denominator_decimal": str(result.denominator),
        "weight_sum_numerator_decimal": str(result.weight_sum_numerator),
        "weight_sum_denominator_decimal": str(result.weight_sum_denominator),
        "target_population_count_decimal": (
            str(result.target_population_count)
            if result.target_population_count is not None
            else None
        ),
    }


def _population_estimand_metadata(
    pairs: tuple[PopulationEstimandPair, ...],
    *,
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    if metadata is None:
        payload: dict[str, object] = {}
    elif isinstance(metadata, Mapping):
        payload = dict(metadata)
    else:
        raise TypeError("target-population estimand metadata must be a mapping")
    collisions = sorted(_WRITER_METADATA_FIELDS.intersection(payload))
    if collisions:
        raise ValueError(
            "target-population metadata fields are writer-owned: "
            + ", ".join(collisions)
        )

    records = [
        {"spec": spec.snapshot(), "result": result.snapshot()}
        for spec, result in pairs
    ]
    upstream_identities = [
        {
            "estimand_id": spec.estimand_id,
            "estimand_sha256": spec.estimand_sha256,
            "target_population_id": spec.target_population_id,
            "target_evidence_sha256": spec.target_evidence_sha256,
            "design_weights_sha256": spec.design_weights_sha256,
            "runtime_projection_sha256": spec.runtime_projection_sha256,
            "balance_report_sha256": spec.balance_report_sha256,
            "metric_contract_sha256": spec.metric_contract_sha256,
        }
        for spec, _result in pairs
    ]
    payload.update(
        {
            "record_count_decimal": str(len(pairs)),
            "ordered_estimand_ids": [spec.estimand_id for spec, _result in pairs],
            "ordered_estimand_sha256": [
                spec.estimand_sha256 for spec, _result in pairs
            ],
            "ordered_result_sha256": [
                result.result_sha256 for _spec, result in pairs
            ],
            "record_set_sha256": _canonical_sha256(
                {"ordered_records": records}
            ),
            "records": records,
            "upstream_identity_declarations": upstream_identities,
            "upstream_identity_scope": (
                "Copied from re-attested exact estimand specs; this writer does "
                "not independently resolve or authenticate the referenced "
                "target evidence, design weights, runtime projection, balance "
                "report, or metric contract."
            ),
            "campaign_readiness_basis": (
                "False: this profile has no separate campaign-readiness gate."
            ),
        }
    )
    return stamp_target_population_estimand_schema(payload)


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "PopulationEstimandPair",
    "write_target_population_estimands",
]
