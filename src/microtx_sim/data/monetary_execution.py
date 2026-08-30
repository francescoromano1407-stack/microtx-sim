"""Exact, jurisdiction-aware conversion of simulation-money outcomes.

This module is deliberately narrower than a substantive monetary-comparability
claim.  It re-attests registered profile lineage, binds the exact local-money
scales and verified rate extractions used by one target-currency basis, and
converts player observations exactly before any cross-jurisdiction reduction.

Simulation cents are not inverted into recovered local-currency observations.
For jurisdiction ``j`` the executed model-scale ratio is instead

``(target minor / source minor) / (simulation cent / source minor)``.

The combined exact rational remains unrounded through population weighting,
scenario contrast, and seed aggregation.  It is rounded once, signed
half-away from zero, only by the separate production monetary writer.  This
avoids a lossy intermediate local amount, raw cross-currency pooling, and
double rounding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from fractions import Fraction
from hashlib import sha256
import json
import re
from typing import Final, Mapping, Sequence

import numpy as np

from ..metrics.population_estimands import (
    PopulationCurrencyRounding,
    PopulationCurrencySemantics,
)
from ..types import ProvenanceStatus
from .lineage import ProfileInputLineage
from .profiles import (
    MonetaryConversionContract,
    MonetaryConversionMethod,
    MonetaryRoundingScope,
    MoneyScaleContract,
    ProfileValidationError,
    _monetary_conversion_from_snapshot,
    _money_scale_from_snapshot,
)
from .rate_evidence import (
    RateEvidenceResult,
    RateEvidenceValidationError,
    validate_rate_evidence_snapshot,
)


MONETARY_OUTPUT_EXECUTION_SCHEMA_VERSION: Final[str] = "2.0"
MONETARY_OUTPUT_ROUNDING_METHOD: Final[str] = (
    "nearest_minor_unit_half_away_from_zero"
)
MONETARY_OUTPUT_AGGREGATION_UNIT: Final[str] = (
    "one population-weighted target-currency-equivalent estimand after "
    "conversion and cross-seed aggregation"
)

_CAMPAIGN_BLOCKERS: Final[tuple[str, ...]] = (
    "monetary_output_execution.schema_v2=campaign_ineligible",
    "monetary_output_execution.source_bundle_signature=MISSING",
    "monetary_output_execution.simulation_to_local_currency_bridge=unvalidated",
    "monetary_output_execution.population_comparability=unresolved",
    "monetary_output_execution.external_preregistration=unregistered",
    "monetary_output_execution.substantive_validity=unestablished",
)
_REGISTERED_PROFILE_LINEAGE: Final[str] = "registered_profile_bundle"
_JURISDICTION_CODE = re.compile(r"[A-Z][A-Z0-9]{1,15}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_EXACT_INTEGER_BITS: Final[int] = 4096
_INT64_MIN: Final[int] = -(1 << 63)
_INT64_MAX: Final[int] = (1 << 63) - 1


class MonetaryOutputExecutionValidationError(ValueError):
    """Raised when a monetary basis or its execution cannot be re-attested."""


@dataclass(frozen=True, slots=True)
class JurisdictionMonetaryBasis:
    """One jurisdiction's exact model-scale route to the target minor unit."""

    jurisdiction_code: str
    source_currency: str
    conversion_id: str
    rate_binding_id: str
    rate_artifact_id: str
    nominal_monthly_anchor_minor_units: int
    simulation_monthly_anchor_cents: int
    rate_numerator: int
    rate_denominator: int
    target_per_simulation_numerator: int
    target_per_simulation_denominator: int
    money_scale_sha256: str
    monetary_conversion_sha256: str
    rate_binding_sha256: str
    rate_evidence_sha256: str
    rate_artifact_sha256: str
    rate_artifact_byte_length: int
    anchor_status: str
    scale_status: str
    quote_convention: str
    scale_convention: str
    timing_convention: str
    missing_date_policy: str

    def __post_init__(self) -> None:
        _jurisdiction_code(self.jurisdiction_code)
        _currency_code(self.source_currency, name="source_currency")
        _identifier(self.conversion_id, name="conversion_id")
        _identifier(self.rate_binding_id, name="rate_binding_id")
        _identifier(self.rate_artifact_id, name="rate_artifact_id")
        for name in (
            "nominal_monthly_anchor_minor_units",
            "simulation_monthly_anchor_cents",
            "rate_numerator",
            "rate_denominator",
            "target_per_simulation_numerator",
            "target_per_simulation_denominator",
            "rate_artifact_byte_length",
        ):
            _strict_int(getattr(self, name), name=name, minimum=1)
        for name in (
            "money_scale_sha256",
            "monetary_conversion_sha256",
            "rate_binding_sha256",
            "rate_evidence_sha256",
            "rate_artifact_sha256",
        ):
            _sha256_digest(getattr(self, name), name=name)
        for name in (
            "anchor_status",
            "scale_status",
            "quote_convention",
            "scale_convention",
            "timing_convention",
            "missing_date_policy",
        ):
            _nonempty_text(getattr(self, name), name=name)
        expected = Fraction(
            self.rate_numerator * self.nominal_monthly_anchor_minor_units,
            self.rate_denominator * self.simulation_monthly_anchor_cents,
        )
        observed = Fraction(
            self.target_per_simulation_numerator,
            self.target_per_simulation_denominator,
        )
        if observed != expected:
            raise MonetaryOutputExecutionValidationError(
                "target-per-simulation ratio does not match the exact scale and rate"
            )
        if (
            observed.numerator != self.target_per_simulation_numerator
            or observed.denominator != self.target_per_simulation_denominator
        ):
            raise MonetaryOutputExecutionValidationError(
                "target-per-simulation ratio must be reduced"
            )

    @property
    def target_per_simulation(self) -> Fraction:
        return Fraction(
            self.target_per_simulation_numerator,
            self.target_per_simulation_denominator,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "jurisdiction_code": self.jurisdiction_code,
            "source_currency": self.source_currency,
            "conversion_id": self.conversion_id,
            "rate_binding_id": self.rate_binding_id,
            "rate_artifact_id": self.rate_artifact_id,
            "nominal_monthly_anchor_minor_units_decimal": str(
                self.nominal_monthly_anchor_minor_units
            ),
            "simulation_monthly_anchor_cents_decimal": str(
                self.simulation_monthly_anchor_cents
            ),
            "rate_numerator_decimal": str(self.rate_numerator),
            "rate_denominator_decimal": str(self.rate_denominator),
            "target_per_simulation_numerator_decimal": str(
                self.target_per_simulation_numerator
            ),
            "target_per_simulation_denominator_decimal": str(
                self.target_per_simulation_denominator
            ),
            "money_scale_sha256": self.money_scale_sha256,
            "monetary_conversion_sha256": self.monetary_conversion_sha256,
            "rate_binding_sha256": self.rate_binding_sha256,
            "rate_evidence_sha256": self.rate_evidence_sha256,
            "rate_artifact_sha256": self.rate_artifact_sha256,
            "rate_artifact_byte_length": self.rate_artifact_byte_length,
            "anchor_status": self.anchor_status,
            "scale_status": self.scale_status,
            "quote_convention": self.quote_convention,
            "scale_convention": self.scale_convention,
            "timing_convention": self.timing_convention,
            "missing_date_policy": self.missing_date_policy,
        }


@dataclass(frozen=True, slots=True)
class MonetaryOutputBasis:
    """Content-addressed input basis for one prospective output conversion."""

    schema_version: str
    profile_input_sha256: str
    source_bundle_id: str
    source_bundle_sha256: str
    source_bundle_signature_status: str
    target_currency: str
    target_minor_unit_name: str
    method: MonetaryConversionMethod
    rate_period_start: date
    rate_period_end: date
    price_period_start: date
    price_period_end: date
    estimand: str
    population_base: str
    comparison_group: str
    rounding_method: str
    rounding_scope: MonetaryRoundingScope
    aggregation_unit: str
    jurisdiction_codes: tuple[str, ...]
    jurisdictions: tuple[JurisdictionMonetaryBasis, ...]
    basis_sha256: str
    campaign_ready: bool = field(default=False, init=False)
    campaign_blockers: tuple[str, ...] = field(
        default=_CAMPAIGN_BLOCKERS,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.schema_version != MONETARY_OUTPUT_EXECUTION_SCHEMA_VERSION:
            raise MonetaryOutputExecutionValidationError(
                "unsupported monetary-output basis schema version"
            )
        _sha256_digest(self.profile_input_sha256, name="profile_input_sha256")
        _identifier(self.source_bundle_id, name="source_bundle_id")
        _sha256_digest(self.source_bundle_sha256, name="source_bundle_sha256")
        if self.source_bundle_signature_status != "MISSING":
            raise MonetaryOutputExecutionValidationError(
                "unsupported or unverifiable monetary source-bundle signature status"
            )
        _currency_code(self.target_currency, name="target_currency")
        _nonempty_text(
            self.target_minor_unit_name,
            name="target_minor_unit_name",
        )
        if type(self.method) is not MonetaryConversionMethod:
            raise TypeError("method must be MonetaryConversionMethod")
        for name in (
            "rate_period_start",
            "rate_period_end",
            "price_period_start",
            "price_period_end",
        ):
            if type(getattr(self, name)) is not date:
                raise TypeError(f"{name} must be a calendar date")
        if self.rate_period_end < self.rate_period_start:
            raise MonetaryOutputExecutionValidationError(
                "rate period ends before it starts"
            )
        if self.price_period_end < self.price_period_start:
            raise MonetaryOutputExecutionValidationError(
                "price period ends before it starts"
            )
        for name in ("estimand", "population_base", "comparison_group"):
            _nonempty_text(getattr(self, name), name=name)
        if self.rounding_method != MONETARY_OUTPUT_ROUNDING_METHOD:
            raise MonetaryOutputExecutionValidationError(
                "unsupported monetary-output rounding method"
            )
        if self.rounding_scope is not MonetaryRoundingScope.AFTER_AGGREGATION:
            raise MonetaryOutputExecutionValidationError(
                "monetary-output schema v2 requires AFTER_AGGREGATION rounding"
            )
        if self.aggregation_unit != MONETARY_OUTPUT_AGGREGATION_UNIT:
            raise MonetaryOutputExecutionValidationError(
                "monetary-output schema v2 requires the exact final estimand "
                "aggregation unit"
            )
        _ordered_jurisdiction_codes(self.jurisdiction_codes)
        if type(self.jurisdictions) is not tuple or any(
            type(item) is not JurisdictionMonetaryBasis
            for item in self.jurisdictions
        ):
            raise TypeError(
                "jurisdictions must be an exact tuple of JurisdictionMonetaryBasis"
            )
        for item in self.jurisdictions:
            JurisdictionMonetaryBasis.__post_init__(item)
        if tuple(item.jurisdiction_code for item in self.jurisdictions) != (
            self.jurisdiction_codes
        ):
            raise MonetaryOutputExecutionValidationError(
                "monetary jurisdiction rows do not match ordered jurisdiction codes"
            )
        if self.campaign_ready or self.campaign_blockers != _CAMPAIGN_BLOCKERS:
            raise MonetaryOutputExecutionValidationError(
                "monetary-output schema v2 has fixed non-campaign status"
            )
        _sha256_digest(self.basis_sha256, name="basis_sha256")
        if self.basis_sha256 != _canonical_sha256(self.attestation_payload()):
            raise MonetaryOutputExecutionValidationError(
                "basis_sha256 does not match the canonical monetary basis"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile_input_sha256": self.profile_input_sha256,
            "source_bundle_id": self.source_bundle_id,
            "source_bundle_sha256": self.source_bundle_sha256,
            "source_bundle_signature_status": (
                self.source_bundle_signature_status
            ),
            "target_currency": self.target_currency,
            "target_minor_unit_name": self.target_minor_unit_name,
            "method": self.method.value,
            "rate_period_start": self.rate_period_start.isoformat(),
            "rate_period_end": self.rate_period_end.isoformat(),
            "price_period_start": self.price_period_start.isoformat(),
            "price_period_end": self.price_period_end.isoformat(),
            "estimand": self.estimand,
            "population_base": self.population_base,
            "comparison_group": self.comparison_group,
            "rounding_method": self.rounding_method,
            "rounding_scope": self.rounding_scope.value,
            "aggregation_unit": self.aggregation_unit,
            "jurisdiction_codes": list(self.jurisdiction_codes),
            "jurisdictions": [item.snapshot() for item in self.jurisdictions],
            "campaign_ready": self.campaign_ready,
            "campaign_blockers": list(self.campaign_blockers),
            "conversion_order": [
                "retain_raw_simulation_cents",
                "apply_jurisdiction_local_scale_and_fx_as_exact_rational",
                "apply_common_population_weights_to_each_scenario",
                "form_declared_scenario_contrast",
                "aggregate_fixed seeds equally",
                "round_once_at_production_output_boundary",
            ],
            "raw_cross_jurisdiction_sum_allowed": False,
            "rounding_point": "production monetary output boundary only",
            "empirical_interpretation": (
                "target-currency-equivalent model amount; not observed "
                "real-world spending"
            ),
        }

    def snapshot(self) -> dict[str, object]:
        return {**self.attestation_payload(), "basis_sha256": self.basis_sha256}

    @property
    def currency_semantics(self) -> PopulationCurrencySemantics:
        return PopulationCurrencySemantics(
            currency_code=self.target_currency,
            minor_unit_name=self.target_minor_unit_name,
            price_period_start=self.price_period_start,
            price_period_end=self.price_period_end,
            currency_basis_sha256=self.basis_sha256,
            rounding=PopulationCurrencyRounding.NONE_EXACT_RATIONAL,
        )


@dataclass(frozen=True, slots=True)
class ConvertedMonetaryOutcome:
    """One exact converted vector bound to players, jurisdictions, and basis."""

    schema_version: str
    basis: MonetaryOutputBasis
    player_ids: tuple[int, ...]
    jurisdiction_indices: tuple[int, ...]
    raw_values: tuple[int, ...]
    converted_values: tuple[Fraction, ...]
    player_ids_sha256: str
    raw_values_sha256: str
    converted_values_sha256: str
    jurisdiction_assignment_sha256: str
    execution_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != MONETARY_OUTPUT_EXECUTION_SCHEMA_VERSION:
            raise MonetaryOutputExecutionValidationError(
                "unsupported converted monetary-outcome schema version"
            )
        if type(self.basis) is not MonetaryOutputBasis:
            raise TypeError("basis must be MonetaryOutputBasis")
        MonetaryOutputBasis.__post_init__(self.basis)
        for name in (
            "player_ids",
            "jurisdiction_indices",
            "raw_values",
            "converted_values",
        ):
            if type(getattr(self, name)) is not tuple:
                raise TypeError(f"{name} must be an immutable tuple")
        size = len(self.player_ids)
        if any(
            len(values) != size
            for values in (
                self.jurisdiction_indices,
                self.raw_values,
                self.converted_values,
            )
        ):
            raise MonetaryOutputExecutionValidationError(
                "players, jurisdictions, raw values, and converted values must align"
            )
        for index, player_id in enumerate(self.player_ids):
            _strict_int(
                player_id,
                name=f"player_ids[{index}]",
                minimum=0,
                maximum=_INT64_MAX,
            )
        if len(set(self.player_ids)) != size:
            raise MonetaryOutputExecutionValidationError(
                "converted monetary player IDs must be unique"
            )
        for index, jurisdiction_index in enumerate(self.jurisdiction_indices):
            _strict_int(
                jurisdiction_index,
                name=f"jurisdiction_indices[{index}]",
                minimum=0,
                maximum=len(self.basis.jurisdiction_codes) - 1,
            )
        for index, value in enumerate(self.raw_values):
            _strict_int(
                value,
                name=f"raw_values[{index}]",
                minimum=_INT64_MIN,
                maximum=_INT64_MAX,
            )
        for index, value in enumerate(self.converted_values):
            if type(value) is not Fraction:
                raise TypeError(
                    f"converted_values[{index}] must be an exact Fraction"
                )
            if (
                value.numerator.bit_length() > _MAX_EXACT_INTEGER_BITS
                or value.denominator.bit_length() > _MAX_EXACT_INTEGER_BITS
            ):
                raise MonetaryOutputExecutionValidationError(
                    f"converted_values[{index}] exceeds the exact integer limit"
                )
        for name in (
            "player_ids_sha256",
            "raw_values_sha256",
            "converted_values_sha256",
            "jurisdiction_assignment_sha256",
            "execution_sha256",
        ):
            _sha256_digest(getattr(self, name), name=name)
        if self.player_ids_sha256 != _integer_values_sha256(self.player_ids):
            raise MonetaryOutputExecutionValidationError(
                "player_ids_sha256 does not match player IDs"
            )
        if self.raw_values_sha256 != _integer_values_sha256(self.raw_values):
            raise MonetaryOutputExecutionValidationError(
                "raw_values_sha256 does not match raw values"
            )
        if self.converted_values_sha256 != _fraction_values_sha256(
            self.converted_values
        ):
            raise MonetaryOutputExecutionValidationError(
                "converted_values_sha256 does not match converted values"
            )
        if self.jurisdiction_assignment_sha256 != _canonical_sha256(
            {
                "jurisdiction_codes": list(self.basis.jurisdiction_codes),
                "player_ids_decimal": [str(value) for value in self.player_ids],
                "jurisdiction_indices_decimal": [
                    str(value) for value in self.jurisdiction_indices
                ],
            }
        ):
            raise MonetaryOutputExecutionValidationError(
                "jurisdiction_assignment_sha256 does not match player assignments"
            )
        expected_values = _convert_values(
            self.basis,
            jurisdiction_indices=self.jurisdiction_indices,
            raw_values=self.raw_values,
        )
        if expected_values != self.converted_values:
            raise MonetaryOutputExecutionValidationError(
                "converted values do not match the exact monetary basis"
            )
        if self.execution_sha256 != _canonical_sha256(
            self.attestation_payload()
        ):
            raise MonetaryOutputExecutionValidationError(
                "execution_sha256 does not match converted monetary outcome"
            )

    def attestation_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "basis": self.basis.snapshot(),
            "player_count": len(self.player_ids),
            "player_ids_sha256": self.player_ids_sha256,
            "raw_values_sha256": self.raw_values_sha256,
            "converted_values_sha256": self.converted_values_sha256,
            "converted_value_representation": (
                "exact unrounded target minor-unit rationals"
            ),
            "jurisdiction_assignment_sha256": (
                self.jurisdiction_assignment_sha256
            ),
        }

    def snapshot(self) -> dict[str, object]:
        return {
            **self.attestation_payload(),
            "execution_sha256": self.execution_sha256,
        }


def build_monetary_output_currency_semantics(
    profile_lineage: ProfileInputLineage,
    *,
    jurisdiction_codes: tuple[str, ...],
    target_minor_unit_name: str,
) -> PopulationCurrencySemantics:
    """Derive prospective currency semantics and its exact basis digest."""

    components = _resolve_components(
        profile_lineage,
        jurisdiction_codes=jurisdiction_codes,
    )
    target_currency = components["target_currency"]
    price_period_start = components["price_period_start"]
    price_period_end = components["price_period_end"]
    assert isinstance(target_currency, str)
    assert type(price_period_start) is date
    assert type(price_period_end) is date
    provisional = PopulationCurrencySemantics(
        currency_code=target_currency,
        minor_unit_name=target_minor_unit_name,
        price_period_start=price_period_start,
        price_period_end=price_period_end,
        currency_basis_sha256="0" * 64,
        rounding=PopulationCurrencyRounding.NONE_EXACT_RATIONAL,
    )
    basis = _basis_from_components(components, provisional)
    return basis.currency_semantics


def resolve_monetary_output_basis(
    profile_lineage: ProfileInputLineage,
    currency: PopulationCurrencySemantics,
    *,
    jurisdiction_codes: tuple[str, ...],
) -> MonetaryOutputBasis:
    """Re-attest lineage and bind it to declared target-currency semantics."""

    if type(currency) is not PopulationCurrencySemantics:
        raise TypeError("currency must be PopulationCurrencySemantics")
    PopulationCurrencySemantics.__post_init__(currency)
    components = _resolve_components(
        profile_lineage,
        jurisdiction_codes=jurisdiction_codes,
    )
    basis = _basis_from_components(components, currency)
    if currency.currency_basis_sha256 != basis.basis_sha256:
        raise MonetaryOutputExecutionValidationError(
            "currency_basis_sha256 does not match the re-attested monetary basis"
        )
    if currency != basis.currency_semantics:
        raise MonetaryOutputExecutionValidationError(
            "currency semantics differ from the re-attested monetary basis"
        )
    return basis


def convert_monetary_outcome(
    basis: MonetaryOutputBasis,
    *,
    player_ids: Sequence[object],
    jurisdiction_indices: Sequence[object],
    jurisdiction_codes: tuple[str, ...],
    raw_values: Sequence[object],
) -> ConvertedMonetaryOutcome:
    """Convert one aligned player vector using one exact jurisdiction route."""

    if type(basis) is not MonetaryOutputBasis:
        raise TypeError("basis must be MonetaryOutputBasis")
    MonetaryOutputBasis.__post_init__(basis)
    _ordered_jurisdiction_codes(jurisdiction_codes)
    if jurisdiction_codes != basis.jurisdiction_codes:
        raise MonetaryOutputExecutionValidationError(
            "runtime jurisdiction codes differ from the monetary basis"
        )
    normalized_player_ids = _integer_tuple(
        player_ids,
        name="player_ids",
        minimum=0,
        maximum=_INT64_MAX,
    )
    normalized_indices = _integer_tuple(
        jurisdiction_indices,
        name="jurisdiction_indices",
        minimum=0,
        maximum=len(jurisdiction_codes) - 1,
    )
    normalized_raw = _integer_tuple(
        raw_values,
        name="raw_values",
        minimum=_INT64_MIN,
        maximum=_INT64_MAX,
    )
    size = len(normalized_player_ids)
    if len(normalized_indices) != size or len(normalized_raw) != size:
        raise MonetaryOutputExecutionValidationError(
            "player IDs, jurisdiction indices, and raw values must align"
        )
    if len(set(normalized_player_ids)) != size:
        raise MonetaryOutputExecutionValidationError(
            "converted monetary player IDs must be unique"
        )
    converted = _convert_values(
        basis,
        jurisdiction_indices=normalized_indices,
        raw_values=normalized_raw,
    )
    player_ids_digest = _integer_values_sha256(normalized_player_ids)
    raw_digest = _integer_values_sha256(normalized_raw)
    converted_digest = _fraction_values_sha256(converted)
    assignment_digest = _canonical_sha256(
        {
            "jurisdiction_codes": list(jurisdiction_codes),
            "player_ids_decimal": [str(value) for value in normalized_player_ids],
            "jurisdiction_indices_decimal": [
                str(value) for value in normalized_indices
            ],
        }
    )
    payload = {
        "schema_version": MONETARY_OUTPUT_EXECUTION_SCHEMA_VERSION,
        "basis": basis.snapshot(),
        "player_count": size,
        "player_ids_sha256": player_ids_digest,
        "raw_values_sha256": raw_digest,
        "converted_values_sha256": converted_digest,
        "converted_value_representation": (
            "exact unrounded target minor-unit rationals"
        ),
        "jurisdiction_assignment_sha256": assignment_digest,
    }
    return ConvertedMonetaryOutcome(
        schema_version=MONETARY_OUTPUT_EXECUTION_SCHEMA_VERSION,
        basis=basis,
        player_ids=normalized_player_ids,
        jurisdiction_indices=normalized_indices,
        raw_values=normalized_raw,
        converted_values=converted,
        player_ids_sha256=player_ids_digest,
        raw_values_sha256=raw_digest,
        converted_values_sha256=converted_digest,
        jurisdiction_assignment_sha256=assignment_digest,
        execution_sha256=_canonical_sha256(payload),
    )


def _resolve_components(
    profile_lineage: ProfileInputLineage,
    *,
    jurisdiction_codes: tuple[str, ...],
) -> dict[str, object]:
    if type(profile_lineage) is not ProfileInputLineage:
        raise TypeError("profile_lineage must be ProfileInputLineage")
    _ordered_jurisdiction_codes(jurisdiction_codes)
    try:
        # This is intentionally a fresh constructor call rather than trust in a
        # previously validated object. Registered files and rate artifacts are
        # reopened and compared with the retained snapshot here.
        reattested = ProfileInputLineage(
            lineage_status=profile_lineage.lineage_status,
            profile_codes=profile_lineage.profile_codes,
            fingerprint_sha256=profile_lineage.fingerprint_sha256,
            snapshot_json=profile_lineage.snapshot_json,
            jurisdictions_path=profile_lineage.jurisdictions_path,
            jurisdictions_sha256=profile_lineage.jurisdictions_sha256,
            source_registry_path=profile_lineage.source_registry_path,
            source_registry_sha256=profile_lineage.source_registry_sha256,
            source_retrieved_on=profile_lineage.source_retrieved_on,
            source_bundle_path=profile_lineage.source_bundle_path,
            source_bundle_sha256=profile_lineage.source_bundle_sha256,
            population_bundle_path=profile_lineage.population_bundle_path,
            population_bundle_sha256=profile_lineage.population_bundle_sha256,
        )
        if reattested != profile_lineage:
            raise MonetaryOutputExecutionValidationError(
                "profile lineage differs from its exact reconstruction"
            )
        if reattested.lineage_status != _REGISTERED_PROFILE_LINEAGE:
            raise MonetaryOutputExecutionValidationError(
                "monetary output execution requires registered profile lineage"
            )
        if jurisdiction_codes != reattested.profile_codes:
            raise MonetaryOutputExecutionValidationError(
                "ordered jurisdiction codes differ from profile lineage"
            )
        snapshot = reattested.snapshot
        bundle_row = snapshot.get("profile_bundle")
        if not isinstance(bundle_row, Mapping):
            raise MonetaryOutputExecutionValidationError(
                "registered profile lineage lacks a profile-bundle snapshot"
            )
        if bundle_row.get("jurisdiction_schema_version") != 3:
            raise MonetaryOutputExecutionValidationError(
                "monetary output execution requires jurisdiction schema version 3"
            )
        scales_raw = bundle_row.get("money_scales")
        conversions_raw = bundle_row.get("monetary_conversions")
        if not isinstance(scales_raw, list) or not isinstance(
            conversions_raw, list
        ):
            raise MonetaryOutputExecutionValidationError(
                "profile monetary scale or conversion snapshots are malformed"
            )
        scales = tuple(
            _money_scale_from_snapshot(_mapping(item, name="money scale"))
            for item in scales_raw
        )
        conversions = tuple(
            _monetary_conversion_from_snapshot(
                _mapping(item, name="monetary conversion")
            )
            for item in conversions_raw
        )
        evidence_bundle, evidence_results = validate_rate_evidence_snapshot(
            bundle_row.get("source_evidence_bundle"),
            bundle_row.get("rate_evidence_results"),
        )
    except (
        ProfileValidationError,
        RateEvidenceValidationError,
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        if isinstance(exc, MonetaryOutputExecutionValidationError):
            raise
        raise MonetaryOutputExecutionValidationError(
            f"monetary output basis cannot re-attest profile inputs: {exc}"
        ) from exc

    if evidence_bundle is None or reattested.source_bundle_sha256 is None:
        raise MonetaryOutputExecutionValidationError(
            "monetary output execution requires verified rate evidence"
        )
    if evidence_bundle.bundle_sha256 != reattested.source_bundle_sha256:
        raise MonetaryOutputExecutionValidationError(
            "verified rate bundle differs from profile lineage"
        )
    if evidence_bundle.provenance_status is not ProvenanceStatus.CALIBRATED:
        raise MonetaryOutputExecutionValidationError(
            "rate evidence bundle must be CALIBRATED for output execution"
        )
    scales_by_code = {item.jurisdiction_code: item for item in scales}
    conversions_by_code = {
        item.jurisdiction_code: item for item in conversions
    }
    if set(scales_by_code) != set(jurisdiction_codes) or set(
        conversions_by_code
    ) != set(jurisdiction_codes):
        raise MonetaryOutputExecutionValidationError(
            "money scales and conversions must exactly cover jurisdictions"
        )
    if len(scales_by_code) != len(scales) or len(conversions_by_code) != len(
        conversions
    ):
        raise MonetaryOutputExecutionValidationError(
            "money scales or conversions repeat a jurisdiction"
        )
    for conversion in conversions:
        if conversion.status is not ProvenanceStatus.CALIBRATED:
            raise MonetaryOutputExecutionValidationError(
                f"{conversion.jurisdiction_code} monetary conversion must be CALIBRATED"
            )
        if conversion.rounding_scope is not MonetaryRoundingScope.AFTER_AGGREGATION:
            raise MonetaryOutputExecutionValidationError(
                "monetary output execution schema v2 requires AFTER_AGGREGATION "
                f"rounding; {conversion.jurisdiction_code} declares "
                f"{conversion.rounding_scope.value}"
            )
        if conversion.rounding_method != MONETARY_OUTPUT_ROUNDING_METHOD:
            raise MonetaryOutputExecutionValidationError(
                "monetary conversion rounding method is unsupported"
            )
        if conversion.aggregation_unit != MONETARY_OUTPUT_AGGREGATION_UNIT:
            raise MonetaryOutputExecutionValidationError(
                "monetary output execution schema v2 requires the exact final "
                "estimand aggregation unit; "
                f"{conversion.jurisdiction_code} declares "
                f"{conversion.aggregation_unit!r}"
            )
        for name in (
            "quote_convention",
            "scale_convention",
            "timing_convention",
            "missing_date_policy",
        ):
            if not getattr(conversion, name):
                raise MonetaryOutputExecutionValidationError(
                    f"{conversion.jurisdiction_code} monetary conversion lacks "
                    f"{name}"
                )
    signatures = {item.comparison_signature for item in conversions}
    if len(signatures) != 1:
        raise MonetaryOutputExecutionValidationError(
            "monetary conversions do not share one comparison basis"
        )

    results_by_binding = {item.binding_id: item for item in evidence_results}
    raw_scales_by_code = {
        str(_mapping(item, name="money scale").get("jurisdiction_code")): item
        for item in scales_raw
    }
    raw_conversions_by_code = {
        str(
            _mapping(item, name="monetary conversion").get(
                "jurisdiction_code"
            )
        ): item
        for item in conversions_raw
    }
    rows: list[JurisdictionMonetaryBasis] = []
    for code in jurisdiction_codes:
        scale = scales_by_code[code]
        conversion = conversions_by_code[code]
        if conversion.source_currency != scale.currency:
            raise MonetaryOutputExecutionValidationError(
                f"{code} scale currency differs from conversion source currency"
            )
        binding_id = conversion.rate_binding_id
        conversion_id = conversion.conversion_id
        if binding_id is None or conversion_id is None:
            raise MonetaryOutputExecutionValidationError(
                f"{code} conversion lacks schema-v3 identities"
            )
        result = results_by_binding.get(binding_id)
        if result is None:
            raise MonetaryOutputExecutionValidationError(
                f"{code} conversion lacks an exact verified rate result"
            )
        _validate_rate_result(conversion, result)
        ratio = conversion.conversion_ratio / scale.currency_scale_to_sim
        rows.append(
            JurisdictionMonetaryBasis(
                jurisdiction_code=code,
                source_currency=scale.currency,
                conversion_id=conversion_id,
                rate_binding_id=binding_id,
                rate_artifact_id=result.artifact_id,
                nominal_monthly_anchor_minor_units=(
                    scale.nominal_monthly_anchor_minor_units
                ),
                simulation_monthly_anchor_cents=(
                    scale.simulation_monthly_anchor_cents
                ),
                rate_numerator=conversion.rate_numerator,
                rate_denominator=conversion.rate_denominator,
                target_per_simulation_numerator=ratio.numerator,
                target_per_simulation_denominator=ratio.denominator,
                money_scale_sha256=_canonical_sha256(
                    raw_scales_by_code[code]
                ),
                monetary_conversion_sha256=_canonical_sha256(
                    raw_conversions_by_code[code]
                ),
                rate_binding_sha256=result.binding_sha256,
                rate_evidence_sha256=result.evidence_sha256,
                rate_artifact_sha256=result.artifact_sha256,
                rate_artifact_byte_length=result.artifact_byte_length,
                anchor_status=scale.anchor_status.value,
                scale_status=scale.scale_status.value,
                quote_convention=conversion.quote_convention,
                scale_convention=conversion.scale_convention,
                timing_convention=conversion.timing_convention,
                missing_date_policy=conversion.missing_date_policy,
            )
        )

    first = conversions_by_code[jurisdiction_codes[0]]
    return {
        "profile_input_sha256": reattested.fingerprint_sha256,
        "source_bundle_id": evidence_bundle.bundle_id,
        "source_bundle_sha256": evidence_bundle.bundle_sha256,
        "source_bundle_signature_status": evidence_bundle.signature.status.value,
        "target_currency": first.target_currency,
        "method": first.method,
        "rate_period_start": first.rate_period_start,
        "rate_period_end": first.rate_period_end,
        "price_period_start": first.target_price_period_start,
        "price_period_end": first.target_price_period_end,
        "estimand": first.estimand,
        "population_base": first.population_base,
        "comparison_group": first.comparison_group,
        "rounding_method": first.rounding_method,
        "rounding_scope": first.rounding_scope,
        "aggregation_unit": first.aggregation_unit,
        "jurisdiction_codes": jurisdiction_codes,
        "jurisdictions": tuple(rows),
    }


def _basis_from_components(
    components: Mapping[str, object],
    currency: PopulationCurrencySemantics,
) -> MonetaryOutputBasis:
    if currency.currency_code != components["target_currency"]:
        raise MonetaryOutputExecutionValidationError(
            "declared target currency differs from monetary conversions"
        )
    if (
        currency.price_period_start != components["price_period_start"]
        or currency.price_period_end != components["price_period_end"]
    ):
        raise MonetaryOutputExecutionValidationError(
            "declared price period differs from monetary conversions"
        )
    if currency.rounding is not PopulationCurrencyRounding.NONE_EXACT_RATIONAL:
        raise MonetaryOutputExecutionValidationError(
            "population estimator must retain exact rational results after "
            "jurisdiction conversion and before final output rounding"
        )
    kwargs = {
        "schema_version": MONETARY_OUTPUT_EXECUTION_SCHEMA_VERSION,
        "profile_input_sha256": components["profile_input_sha256"],
        "source_bundle_id": components["source_bundle_id"],
        "source_bundle_sha256": components["source_bundle_sha256"],
        "source_bundle_signature_status": components[
            "source_bundle_signature_status"
        ],
        "target_currency": components["target_currency"],
        "target_minor_unit_name": currency.minor_unit_name,
        "method": components["method"],
        "rate_period_start": components["rate_period_start"],
        "rate_period_end": components["rate_period_end"],
        "price_period_start": components["price_period_start"],
        "price_period_end": components["price_period_end"],
        "estimand": components["estimand"],
        "population_base": components["population_base"],
        "comparison_group": components["comparison_group"],
        "rounding_method": components["rounding_method"],
        "rounding_scope": components["rounding_scope"],
        "aggregation_unit": components["aggregation_unit"],
        "jurisdiction_codes": components["jurisdiction_codes"],
        "jurisdictions": components["jurisdictions"],
    }
    payload = _basis_attestation_payload(**kwargs)
    return MonetaryOutputBasis(
        **kwargs,  # type: ignore[arg-type]
        basis_sha256=_canonical_sha256(payload),
    )


def _basis_attestation_payload(**kwargs: object) -> dict[str, object]:
    method = kwargs["method"]
    rounding_scope = kwargs["rounding_scope"]
    jurisdictions = kwargs["jurisdictions"]
    assert isinstance(method, MonetaryConversionMethod)
    assert isinstance(rounding_scope, MonetaryRoundingScope)
    assert isinstance(jurisdictions, tuple)
    payload = {
        **kwargs,
        "method": method.value,
        "rate_period_start": _date_text(kwargs["rate_period_start"]),
        "rate_period_end": _date_text(kwargs["rate_period_end"]),
        "price_period_start": _date_text(kwargs["price_period_start"]),
        "price_period_end": _date_text(kwargs["price_period_end"]),
        "rounding_scope": rounding_scope.value,
        "jurisdiction_codes": list(kwargs["jurisdiction_codes"]),
        "jurisdictions": [item.snapshot() for item in jurisdictions],
        "campaign_ready": False,
        "campaign_blockers": list(_CAMPAIGN_BLOCKERS),
        "conversion_order": [
            "retain_raw_simulation_cents",
            "apply_jurisdiction_local_scale_and_fx_as_exact_rational",
            "apply_common_population_weights_to_each_scenario",
            "form_declared_scenario_contrast",
            "aggregate_fixed seeds equally",
            "round_once_at_production_output_boundary",
        ],
        "raw_cross_jurisdiction_sum_allowed": False,
        "rounding_point": "production monetary output boundary only",
        "empirical_interpretation": (
            "target-currency-equivalent model amount; not observed "
            "real-world spending"
        ),
    }
    return payload


def _validate_rate_result(
    conversion: MonetaryConversionContract,
    result: RateEvidenceResult,
) -> None:
    if (
        result.binding_id != conversion.rate_binding_id
        or result.rate_numerator != conversion.rate_numerator
        or result.rate_denominator != conversion.rate_denominator
    ):
        raise MonetaryOutputExecutionValidationError(
            f"{conversion.jurisdiction_code} conversion differs from its exact "
            "verified rate result"
        )


def _convert_values(
    basis: MonetaryOutputBasis,
    *,
    jurisdiction_indices: tuple[int, ...],
    raw_values: tuple[int, ...],
) -> tuple[Fraction, ...]:
    rows = basis.jurisdictions
    converted: list[Fraction] = []
    for jurisdiction_index, raw_value in zip(
        jurisdiction_indices,
        raw_values,
        strict=True,
    ):
        row = rows[jurisdiction_index]
        value = Fraction(raw_value, 1) * row.target_per_simulation
        if (
            abs(value.numerator).bit_length() > _MAX_EXACT_INTEGER_BITS
            or value.denominator.bit_length() > _MAX_EXACT_INTEGER_BITS
        ):
            raise MonetaryOutputExecutionValidationError(
                "converted monetary value exceeds the exact-integer safety limit"
            )
        converted.append(value)
    return tuple(converted)


def round_target_minor_units(value: Fraction) -> int:
    """Round one final target-minor-unit estimand exactly once."""

    if type(value) is not Fraction:
        raise TypeError("final monetary output must be an exact Fraction")
    absolute = abs(value.numerator)
    quotient, remainder = divmod(absolute, value.denominator)
    if remainder * 2 >= value.denominator:
        quotient += 1
    return quotient if value.numerator >= 0 else -quotient


def _integer_tuple(
    values: Sequence[object],
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an integer sequence")
    try:
        supplied = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer sequence") from exc
    output: list[int] = []
    for index, value in enumerate(supplied):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name}[{index}] must be an exact integer")
        normalized = int(value)
        _strict_int(
            normalized,
            name=f"{name}[{index}]",
            minimum=minimum,
            maximum=maximum,
        )
        output.append(normalized)
    return tuple(output)


def _integer_values_sha256(values: tuple[int, ...]) -> str:
    return _canonical_sha256(
        {
            "count": len(values),
            "count_decimal": str(len(values)),
            "values_decimal": [str(value) for value in values],
        }
    )


def _fraction_values_sha256(values: tuple[Fraction, ...]) -> str:
    return _canonical_sha256(
        {
            "count": len(values),
            "count_decimal": str(len(values)),
            "values": [
                {
                    "numerator_decimal": str(value.numerator),
                    "denominator_decimal": str(value.denominator),
                }
                for value in values
            ],
        }
    )


def _ordered_jurisdiction_codes(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or not values:
        raise TypeError("jurisdiction_codes must be a non-empty exact tuple")
    for value in values:
        _jurisdiction_code(value)
    if len(set(values)) != len(values):
        raise MonetaryOutputExecutionValidationError(
            "jurisdiction_codes must be unique"
        )


def _jurisdiction_code(value: object) -> None:
    if type(value) is not str or not _JURISDICTION_CODE.fullmatch(value):
        raise MonetaryOutputExecutionValidationError(
            "jurisdiction code must be uppercase canonical ASCII text"
        )


def _currency_code(value: object, *, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 3
        or any(character < "A" or character > "Z" for character in value)
    ):
        raise MonetaryOutputExecutionValidationError(
            f"{name} must be a three-letter uppercase ASCII currency code"
        )


def _identifier(value: object, *, name: str) -> None:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or len(value) > 256
    ):
        raise MonetaryOutputExecutionValidationError(
            f"{name} must be non-empty canonical text"
        )


def _nonempty_text(value: object, *, name: str) -> None:
    if (
        type(value) is not str
        or not value.strip()
        or value.strip() != value
        or len(value) > 2048
    ):
        raise MonetaryOutputExecutionValidationError(
            f"{name} must be non-empty text without surrounding whitespace"
        )


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be a strict Python integer")
    if abs(value).bit_length() > _MAX_EXACT_INTEGER_BITS:
        raise MonetaryOutputExecutionValidationError(
            f"{name} exceeds the exact-integer safety limit"
        )
    if minimum is not None and value < minimum:
        raise MonetaryOutputExecutionValidationError(
            f"{name} must be at least {minimum}"
        )
    if maximum is not None and value > maximum:
        raise MonetaryOutputExecutionValidationError(
            f"{name} must be at most {maximum}"
        )


def _sha256_digest(value: object, *, name: str) -> None:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise MonetaryOutputExecutionValidationError(
            f"{name} must be a lowercase SHA-256 digest"
        )


def _mapping(value: object, *, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise MonetaryOutputExecutionValidationError(f"{name} must be an object")
    return value


def _date_text(value: object) -> str:
    if type(value) is not date:
        raise TypeError("monetary basis period must be a calendar date")
    return value.isoformat()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


__all__ = [
    "MONETARY_OUTPUT_EXECUTION_SCHEMA_VERSION",
    "MONETARY_OUTPUT_AGGREGATION_UNIT",
    "MONETARY_OUTPUT_ROUNDING_METHOD",
    "ConvertedMonetaryOutcome",
    "JurisdictionMonetaryBasis",
    "MonetaryOutputBasis",
    "MonetaryOutputExecutionValidationError",
    "build_monetary_output_currency_semantics",
    "convert_monetary_outcome",
    "round_target_minor_units",
    "resolve_monetary_output_basis",
]
