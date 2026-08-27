from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from fractions import Fraction
from hashlib import sha256
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping
import tomllib

from ..agents.jurisdictions import (
    RegulationRules,
    RegulatorPrivateState,
    StateAgent,
)
from ..consumers.population import CountryProfile
from ..types import ProvenanceStatus

if TYPE_CHECKING:
    from .rate_evidence import RateEvidenceBundle


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JURISDICTIONS_PATH = _PROJECT_ROOT / "configs" / "jurisdictions.toml"
DEFAULT_SOURCES_PATH = _PROJECT_ROOT / "data" / "provenance" / "sources.toml"
DEFAULT_SOURCE_BUNDLE_PATH = (
    _PROJECT_ROOT / "data" / "provenance" / "source_bundle.toml"
)
_USE_REGISTERED_SOURCE_BUNDLE = object()

_EXPECTED_CODES = ("UK", "KR", "JP", "BE")
_SIMULATION_MONTHLY_INCOME_CENTS = 180_000
_RULE_FIELDS = frozenset(
    {
        "paid_random_rewards_restricted",
        "complete_gacha_restricted",
        "odds_disclosure_required",
        "parental_authorisation_required",
        "real_money_price_required",
        "direct_exhortation_to_minors_banned",
    }
)
_RULE_SUPPORTS = {
    "paid_random_rewards_restricted": frozenset(
        {"paid_random_reward_classification", "loot_boxes"}
    ),
    "complete_gacha_restricted": frozenset({"complete_gacha_restriction"}),
    "odds_disclosure_required": frozenset({"odds_disclosure"}),
    "parental_authorisation_required": frozenset(
        {"express_consent", "parental_defaults"}
    ),
    "real_money_price_required": frozenset(
        {"real_money_price", "price_transparency"}
    ),
    "direct_exhortation_to_minors_banned": frozenset(
        {"direct_exhortation_to_minors", "minor_exhortation"}
    ),
}
_NON_METRIC_FIELDS = frozenset(
    {
        "code",
        "name",
        "income_currency",
        "income_period",
        "mobile_spend_population_base",
        "gaming_data_caveat",
        "simulation_monthly_anchor_cents",
    }
)
_CONVERSION_SOURCE_SUPPORTS = {
    "FX": frozenset({"foreign_exchange_rate"}),
    "PPP": frozenset({"purchasing_power_parity"}),
}
_MONETARY_CONVERSION_V2_FIELDS = frozenset(
    {
        "jurisdiction_code",
        "source_currency",
        "target_currency",
        "method",
        "rate_numerator",
        "rate_denominator",
        "rate_period_start",
        "rate_period_end",
        "target_price_period_start",
        "target_price_period_end",
        "estimand",
        "population_base",
        "comparison_group",
        "rounding_method",
        "rounding_scope",
        "aggregation_unit",
        "status",
        "source_ids",
        "retrieved_on",
        "notes",
    }
)
_MONETARY_CONVERSION_V3_FIELDS = _MONETARY_CONVERSION_V2_FIELDS.union(
    {"conversion_id", "rate_binding_id"}
)
_MONETARY_FIXED_BLOCKERS = (
    (
        "source_rate_evidence_bound",
        "monetary_conversion.source_rate_binding=missing",
    ),
    (
        "source_bundle_signature_bound",
        "monetary_conversion.source_bundle_signature=missing",
    ),
    (
        "output_design_binding_bound",
        "monetary_conversion.output_design_binding=missing",
    ),
    (
        "population_binding_bound",
        "monetary_conversion.population_binding=missing",
    ),
    (
        "preregistration_bound",
        "monetary_conversion.preregistration_binding=missing",
    ),
)
_MONETARY_ASSESSMENT_FIELDS = frozenset(
    {
        "structure_coherent",
        "source_rate_evidence_bound",
        "source_bundle_signature_bound",
        "output_design_binding_bound",
        "population_binding_bound",
        "preregistration_bound",
        "public_output_comparability",
        "blockers",
    }
)
_SOURCE_PROVENANCE_SNAPSHOT_FIELDS = frozenset(
    {
        "id",
        "publisher",
        "title",
        "url",
        "period",
        "geography",
        "supports",
        "calibration_status",
        "retrieved_on",
    }
)
_MONEY_SCALE_SNAPSHOT_FIELDS = frozenset(
    {
        "jurisdiction_code",
        "currency",
        "reported_income_values",
        "source_period",
        "nominal_monthly_anchor_minor_units",
        "anchor_selection",
        "anchor_status",
        "source_ids",
        "condition",
        "denominator",
        "simulation_monthly_anchor_cents",
        "scale_status",
        "cross_country_comparable",
    }
)
_MONETARY_CONVERSION_SNAPSHOT_FIELDS = _MONETARY_CONVERSION_V3_FIELDS.union(
    {"rate_numerator_decimal", "rate_denominator_decimal"}
)


class ProfileValidationError(ValueError):
    """Raised when a profile or its evidence contract is internally unsafe."""


# A descriptive alias for callers which treat malformed profiles as configuration
# failures rather than data-validation failures.
ProfileConfigurationError = ProfileValidationError


class MonetaryConversionMethod(str, Enum):
    """Permitted bases for a cross-country monetary conversion."""

    FX = "FX"
    PPP = "PPP"


class MonetaryRoundingScope(str, Enum):
    """Stage at which a declared monetary aggregation is rounded."""

    PER_OBSERVATION = "PER_OBSERVATION"
    AFTER_AGGREGATION = "AFTER_AGGREGATION"


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """One immutable record from the machine-readable source catalogue."""

    id: str
    publisher: str
    title: str
    url: str
    period: str
    geography: str
    supports: tuple[str, ...]
    calibration_status: ProvenanceStatus
    retrieved_on: date | None = None

    def __post_init__(self) -> None:
        for name in ("id", "publisher", "title", "url", "period", "geography"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProfileValidationError(f"source {name} must be non-empty")
        if self.id.strip() != self.id:
            raise ProfileValidationError("source id cannot contain surrounding whitespace")
        if not self.url.startswith(("https://", "http://")):
            raise ProfileValidationError(f"source {self.id} has a non-HTTP URL")
        if not self.supports or any(not item.strip() for item in self.supports):
            raise ProfileValidationError(f"source {self.id} needs non-empty supports")
        if len(set(self.supports)) != len(self.supports):
            raise ProfileValidationError(f"source {self.id} has duplicate supports")
        if not isinstance(self.calibration_status, ProvenanceStatus):
            raise ProfileValidationError(
                f"source {self.id} has an invalid calibration status"
            )
        if self.retrieved_on is not None and type(self.retrieved_on) is not date:
            raise ProfileValidationError(
                f"source {self.id} retrieved_on must be an ISO calendar date"
            )

    @property
    def source_id(self) -> str:
        """Stable alias used by provenance-addressable model inputs."""

        return self.id

    @property
    def status(self) -> ProvenanceStatus:
        return self.calibration_status


@dataclass(frozen=True, slots=True)
class MetricContract:
    """Semantic and provenance contract for one configured quantity.

    ``condition`` says which observations are eligible for a statistic, while
    ``denominator`` names the population over which it is defined.  Keeping the
    two explicit prevents, for example, a payer-only amount from silently being
    applied to every player.
    """

    jurisdiction_code: str
    metric: str
    value: object
    status: ProvenanceStatus
    source_ids: tuple[str, ...]
    condition: str
    denominator: str
    period: str
    currency: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("jurisdiction_code", "metric", "condition", "denominator", "period"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProfileValidationError(
                    f"metric contract {self.metric!r} has empty {name} metadata"
                )
        if not isinstance(self.status, ProvenanceStatus):
            raise ProfileValidationError(
                f"metric contract {self.metric!r} has an invalid status"
            )
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ProfileValidationError(
                f"metric contract {self.metric!r} repeats a source id"
            )
        if any(not source_id.strip() for source_id in self.source_ids):
            raise ProfileValidationError(
                f"metric contract {self.metric!r} has an empty source id"
            )
        if self.currency is not None and (
            len(self.currency) != 3 or self.currency.upper() != self.currency
            or any(
                character < "A" or character > "Z"
                for character in self.currency
            )
        ):
            raise ProfileValidationError(
                f"metric contract {self.metric!r} needs an ISO-style currency code"
            )


@dataclass(frozen=True, slots=True)
class MoneyScaleContract:
    """Typed bridge from a local nominal anchor to simulation money.

    The reported values remain in their local currencies and are never pooled.
    A separate, explicitly illustrative scale maps each country's monthly anchor
    to the same purchasing-power-cent anchor used by prices inside the model.
    This is not an FX or PPP estimate and cannot be used for cross-country income
    comparisons.  A separate dated ``MonetaryConversionContract`` with reviewed
    rates must pass the campaign gate before outputs may pool currencies.
    """

    jurisdiction_code: str
    currency: str
    reported_income_values: tuple[int, ...]
    source_period: str
    nominal_monthly_anchor_minor_units: int
    anchor_selection: str
    anchor_status: ProvenanceStatus
    source_ids: tuple[str, ...]
    condition: str
    denominator: str
    simulation_monthly_anchor_cents: int = _SIMULATION_MONTHLY_INCOME_CENTS
    scale_status: ProvenanceStatus = ProvenanceStatus.ILLUSTRATIVE
    cross_country_comparable: bool = False

    def __post_init__(self) -> None:
        if not self.jurisdiction_code.strip():
            raise ProfileValidationError("money scale needs a jurisdiction code")
        if len(self.currency) != 3 or self.currency.upper() != self.currency:
            raise ProfileValidationError("money scale needs an ISO-style currency code")
        if any(character < "A" or character > "Z" for character in self.currency):
            raise ProfileValidationError("money scale needs an ISO-style currency code")
        if not self.reported_income_values or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.reported_income_values
        ):
            raise ProfileValidationError("reported income values must be positive integers")
        if (
            isinstance(self.nominal_monthly_anchor_minor_units, bool)
            or not isinstance(self.nominal_monthly_anchor_minor_units, int)
            or self.nominal_monthly_anchor_minor_units <= 0
        ):
            raise ProfileValidationError("nominal monthly anchor must be positive")
        if (
            isinstance(self.simulation_monthly_anchor_cents, bool)
            or not isinstance(self.simulation_monthly_anchor_cents, int)
            or self.simulation_monthly_anchor_cents <= 0
        ):
            raise ProfileValidationError("simulation monthly anchor must be positive")
        for name in (
            "source_period",
            "anchor_selection",
            "condition",
            "denominator",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProfileValidationError(f"money scale has empty {name} metadata")
        if not isinstance(self.anchor_status, ProvenanceStatus) or not isinstance(
            self.scale_status, ProvenanceStatus
        ):
            raise ProfileValidationError("money scale statuses are invalid")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ProfileValidationError("money scale repeats a source id")
        if self.cross_country_comparable:
            raise ProfileValidationError(
                "nominal local-currency anchors cannot be marked cross-country comparable"
            )

    @property
    def currency_scale_to_sim(self) -> Fraction:
        """Exact illustrative ratio; no floating-point or currency conversion."""

        return Fraction(
            self.simulation_monthly_anchor_cents,
            self.nominal_monthly_anchor_minor_units,
        )

    def to_simulation_cents(self, amount: int, *, currency: str) -> int:
        """Map a same-currency monthly amount into the internal money unit."""

        if currency != self.currency:
            raise ProfileValidationError(
                f"cannot apply {self.currency} scale to nominal {currency}"
            )
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ProfileValidationError("nominal amount must be a non-negative integer")
        numerator = amount * self.simulation_monthly_anchor_cents
        denominator = self.nominal_monthly_anchor_minor_units
        return _round_positive_ratio(numerator, denominator)

    def nominal_ratio_to(self, other: MoneyScaleContract) -> Fraction:
        """Return a within-jurisdiction ratio and reject nominal country ranking."""

        if self.jurisdiction_code != other.jurisdiction_code:
            raise ProfileValidationError(
                "nominal cross-country comparison is forbidden; use a calibrated "
                "FX/PPP conversion contract"
            )
        if self.currency != other.currency:
            raise ProfileValidationError("nominal amounts use different currencies")
        return Fraction(
            self.nominal_monthly_anchor_minor_units,
            other.nominal_monthly_anchor_minor_units,
        )


@dataclass(frozen=True, slots=True)
class MonetaryConversionContract:
    """Dated exact-rate contract for one local monetary profile.

    ``rate_numerator / rate_denominator`` is expressed in target minor units
    per source minor unit.  Explicit estimand and population metadata preserve
    denominator semantics. ``comparison_group`` binds the rates to one vintage
    so that independently sourced FX or PPP values cannot be pooled merely
    because they happen to share a target currency. Typed period endpoints and
    an explicit aggregation-stage rounding scope prevent two further silent
    transformations. The contract carries no default rate or scientific status.
    """

    jurisdiction_code: str
    source_currency: str
    target_currency: str
    method: MonetaryConversionMethod
    rate_numerator: int
    rate_denominator: int
    rate_period_start: date
    rate_period_end: date
    target_price_period_start: date
    target_price_period_end: date
    estimand: str
    population_base: str
    comparison_group: str
    rounding_method: str
    rounding_scope: MonetaryRoundingScope
    aggregation_unit: str
    status: ProvenanceStatus
    source_ids: tuple[str, ...]
    retrieved_on: date
    notes: str = ""
    conversion_id: str | None = None
    rate_binding_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ids, tuple):
            raise ProfileValidationError(
                "monetary conversion source_ids must be an immutable tuple"
            )
        for name in (
            "jurisdiction_code",
            "estimand",
            "population_base",
            "comparison_group",
            "aggregation_unit",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProfileValidationError(
                    f"monetary conversion has empty {name} metadata"
                )
        for name in (
            "rate_period_start",
            "rate_period_end",
            "target_price_period_start",
            "target_price_period_end",
        ):
            if type(getattr(self, name)) is not date:
                raise ProfileValidationError(
                    f"monetary conversion {name} must be an ISO calendar date"
                )
        if self.rate_period_end < self.rate_period_start:
            raise ProfileValidationError(
                "monetary conversion rate period ends before it starts"
            )
        if self.target_price_period_end < self.target_price_period_start:
            raise ProfileValidationError(
                "monetary conversion target price period ends before it starts"
            )
        if (
            self.target_price_period_start,
            self.target_price_period_end,
        ) != (self.rate_period_start, self.rate_period_end):
            raise ProfileValidationError(
                "target price period must equal the rate period unless a separate "
                "price-adjustment contract is implemented"
            )
        for name in ("source_currency", "target_currency"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 3
                or value.upper() != value
                or any(character < "A" or character > "Z" for character in value)
            ):
                raise ProfileValidationError(
                    f"monetary conversion {name} needs an ISO-style currency code"
                )
        if not isinstance(self.method, MonetaryConversionMethod):
            raise ProfileValidationError("monetary conversion method must be FX or PPP")
        if not isinstance(self.rounding_scope, MonetaryRoundingScope):
            raise ProfileValidationError("monetary conversion rounding scope is invalid")
        for name in ("rate_numerator", "rate_denominator"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ProfileValidationError(
                    f"monetary conversion {name} must be a positive integer"
                )
        if not isinstance(self.status, ProvenanceStatus):
            raise ProfileValidationError("monetary conversion status is invalid")
        if not self.source_ids or any(
            not isinstance(source_id, str) or not source_id.strip()
            for source_id in self.source_ids
        ):
            raise ProfileValidationError(
                "monetary conversion requires non-empty source ids"
            )
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ProfileValidationError("monetary conversion repeats a source id")
        if type(self.retrieved_on) is not date:
            raise ProfileValidationError(
                "monetary conversion retrieved_on must be an ISO calendar date"
            )
        if self.retrieved_on < self.rate_period_end:
            raise ProfileValidationError(
                "monetary conversion retrieval date cannot predate the rate-period end"
            )
        if self.rounding_method != "nearest_minor_unit_half_away_from_zero":
            raise ProfileValidationError(
                "unsupported monetary conversion rounding method"
            )
        if not isinstance(self.notes, str):
            raise ProfileValidationError("monetary conversion notes must be text")
        identifiers = (self.conversion_id, self.rate_binding_id)
        if (identifiers[0] is None) != (identifiers[1] is None):
            raise ProfileValidationError(
                "monetary conversion identity and rate binding must be supplied together"
            )
        for name, value in zip(
            ("conversion_id", "rate_binding_id"), identifiers, strict=True
        ):
            if value is not None and (
                not isinstance(value, str)
                or not value
                or value.strip() != value
            ):
                raise ProfileValidationError(
                    f"monetary conversion {name} must be non-empty canonical text"
                )

    @property
    def conversion_ratio(self) -> Fraction:
        """Exact target-minor/source-minor conversion ratio."""

        return Fraction(self.rate_numerator, self.rate_denominator)

    @property
    def comparison_signature(self) -> tuple[str, ...]:
        """Fields that must agree before jurisdiction values may be pooled."""

        return (
            self.target_currency,
            self.method.value,
            self.rate_period_start.isoformat(),
            self.rate_period_end.isoformat(),
            self.target_price_period_start.isoformat(),
            self.target_price_period_end.isoformat(),
            self.estimand,
            self.population_base,
            self.comparison_group,
            self.rounding_method,
            self.rounding_scope.value,
            self.aggregation_unit,
        )

    @property
    def rate_period_label(self) -> str:
        """Canonical source-catalogue label for the dated rate interval."""

        start = self.rate_period_start.isoformat()
        end = self.rate_period_end.isoformat()
        return start if start == end else f"{start}/{end}"

    def convert_minor_units(self, amount: int, *, currency: str) -> int:
        """Convert one declared aggregation-unit amount using exact arithmetic."""

        if currency != self.source_currency:
            raise ProfileValidationError(
                f"cannot apply {self.source_currency} conversion to {currency}"
            )
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise ProfileValidationError(
                "monetary conversion amount must be a strict integer"
            )
        return _round_signed_ratio(
            amount * self.rate_numerator,
            self.rate_denominator,
        )

    def convert_many_minor_units(
        self,
        amounts: Iterable[int],
        *,
        currency: str,
    ) -> int:
        """Apply the declared rounding stage to a sequence of source amounts."""

        if currency != self.source_currency:
            raise ProfileValidationError(
                f"cannot apply {self.source_currency} conversion to {currency}"
            )
        values = tuple(amounts)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ProfileValidationError(
                "monetary conversion amounts must be strict integers"
            )
        if self.rounding_scope is MonetaryRoundingScope.AFTER_AGGREGATION:
            return self.convert_minor_units(sum(values), currency=currency)
        return sum(
            self.convert_minor_units(value, currency=currency)
            for value in values
        )


@dataclass(frozen=True, slots=True)
class MonetaryEvidenceAssessment:
    """Typed, fail-closed assessment shared by campaign and manifest gates.

    Reproducible source extraction is intentionally independent from the
    still-unimplemented output, population, and external preregistration
    bindings.  A valid rate snapshot can therefore clear only one subgate.
    """

    structure_coherent: bool
    source_rate_evidence_bound: bool
    source_bundle_signature_bound: bool
    output_design_binding_bound: bool
    population_binding_bound: bool
    preregistration_bound: bool
    public_output_comparability: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        boolean_fields = (
            "structure_coherent",
            "source_rate_evidence_bound",
            "source_bundle_signature_bound",
            "output_design_binding_bound",
            "population_binding_bound",
            "preregistration_bound",
            "public_output_comparability",
        )
        if any(type(getattr(self, field)) is not bool for field in boolean_fields):
            raise ProfileValidationError(
                "monetary evidence assessment flags must be booleans"
            )
        if not isinstance(self.blockers, tuple) or any(
            not isinstance(blocker, str) or not blocker
            for blocker in self.blockers
        ):
            raise ProfileValidationError(
                "monetary evidence blockers must be immutable non-empty text"
            )
        if len(set(self.blockers)) != len(self.blockers):
            raise ProfileValidationError(
                "monetary evidence blockers must be unique"
            )
        if any(
            getattr(self, field)
            for field in (
                "source_bundle_signature_bound",
                "output_design_binding_bound",
                "population_binding_bound",
                "preregistration_bound",
            )
        ):
            raise ProfileValidationError(
                "current monetary evidence schema cannot promote signature, "
                "output/design, population, or preregistration gates"
            )
        if self.source_rate_evidence_bound and not self.structure_coherent:
            raise ProfileValidationError(
                "bound source-rate evidence requires coherent monetary structure"
            )
        expected_suffix = tuple(
            blocker
            for field, blocker in _MONETARY_FIXED_BLOCKERS
            if not getattr(self, field)
        )
        if len(self.blockers) < len(expected_suffix) or (
            self.blockers[-len(expected_suffix):] != expected_suffix
        ):
            raise ProfileValidationError(
                "monetary evidence blocker codes do not match their flags"
            )
        structural_blockers = self.blockers[: -len(expected_suffix)]
        if self.structure_coherent and structural_blockers:
            raise ProfileValidationError(
                "coherent monetary structure cannot carry structural blockers"
            )
        if not self.structure_coherent and not structural_blockers:
            raise ProfileValidationError(
                "incoherent monetary structure requires a typed structural blocker"
            )
        if structural_blockers and not _valid_structural_monetary_blockers(
            structural_blockers
        ):
            raise ProfileValidationError(
                "monetary evidence contains unknown or disordered structural blockers"
            )
        expected_public = all(
            (
                self.structure_coherent,
                self.source_rate_evidence_bound,
                self.source_bundle_signature_bound,
                self.output_design_binding_bound,
                self.population_binding_bound,
                self.preregistration_bound,
            )
        )
        if self.public_output_comparability is not expected_public:
            raise ProfileValidationError(
                "monetary evidence public-comparability flag is inconsistent"
            )


def monetary_evidence_assessment_from_snapshot(
    value: object,
    *,
    registered_lineage: bool,
    bundle_snapshot: object | None = None,
) -> MonetaryEvidenceAssessment:
    """Parse one serialized assessment through the authoritative validator."""

    if not isinstance(value, Mapping) or set(value) != _MONETARY_ASSESSMENT_FIELDS:
        raise ProfileValidationError(
            "profile snapshot monetary evidence assessment is malformed"
        )
    boolean_fields = _MONETARY_ASSESSMENT_FIELDS.difference({"blockers"})
    if any(type(value[field]) is not bool for field in boolean_fields):
        raise ProfileValidationError(
            "profile snapshot monetary evidence flags must be booleans"
        )
    blockers = value["blockers"]
    if not isinstance(blockers, (list, tuple)) or any(
        not isinstance(blocker, str) or not blocker for blocker in blockers
    ):
        raise ProfileValidationError(
            "profile snapshot monetary evidence blockers are malformed"
        )
    assessment = MonetaryEvidenceAssessment(
        structure_coherent=value["structure_coherent"],
        source_rate_evidence_bound=value["source_rate_evidence_bound"],
        source_bundle_signature_bound=value["source_bundle_signature_bound"],
        output_design_binding_bound=value["output_design_binding_bound"],
        population_binding_bound=value["population_binding_bound"],
        preregistration_bound=value["preregistration_bound"],
        public_output_comparability=value["public_output_comparability"],
        blockers=tuple(blockers),
    )
    if not registered_lineage and assessment.source_rate_evidence_bound:
        raise ProfileValidationError(
            "unregistered profile lineage cannot claim bound source-rate evidence"
        )
    if bundle_snapshot is not None:
        expected_coherent, expected_structural = (
            monetary_structure_assessment_from_snapshot(bundle_snapshot)
        )
        expected_suffix = tuple(
            blocker
            for field, blocker in _MONETARY_FIXED_BLOCKERS
            if not getattr(assessment, field)
        )
        actual_structural = assessment.blockers[: -len(expected_suffix)]
        if (
            assessment.structure_coherent is not expected_coherent
            or actual_structural != expected_structural
        ):
            raise ProfileValidationError(
                "profile monetary assessment does not match its bundle structure"
            )
    return assessment


def _valid_structural_monetary_blockers(blockers: tuple[str, ...]) -> bool:
    jurisdiction_blockers: list[str] = []
    for code in _EXPECTED_CODES:
        allowed = {
            f"{code}.monetary_conversion=missing",
            f"{code}.monetary_conversion={ProvenanceStatus.ANCHORED.value}",
            f"{code}.monetary_conversion={ProvenanceStatus.ILLUSTRATIVE.value}",
            f"{code}.monetary_conversion={ProvenanceStatus.SYNTHETIC.value}",
        }
        jurisdiction_blockers.extend(
            blocker for blocker in blockers if blocker in allowed
        )
    if jurisdiction_blockers:
        if len(jurisdiction_blockers) != len(blockers):
            return False
        expected = tuple(
            blocker
            for code in _EXPECTED_CODES
            for blocker in blockers
            if blocker.startswith(f"{code}.monetary_conversion=")
        )
        return tuple(blockers) == expected and len(expected) == len(blockers)
    ordered_global = (
        "monetary_conversion.comparison_basis=inconsistent",
        "monetary_conversion.internal_scale=incoherent",
    )
    return tuple(blockers) in {
        ("monetary_conversion.structure=unavailable",),
        (ordered_global[0],),
        (ordered_global[1],),
        ordered_global,
    }


def monetary_structure_assessment_from_snapshot(
    bundle_snapshot: object,
) -> tuple[bool, tuple[str, ...]]:
    """Derive structural blockers only after rebuilding the typed contracts.

    The rebuild is deliberate: a common malformed value in every jurisdiction
    must not survive merely because equality-based comparison signatures still
    agree.  It also prevents a non-calibrated or missing conversion from hiding
    malformed rows, extra jurisdictions, or lossy integer mirrors behind an
    early blocker return.
    """

    if not isinstance(bundle_snapshot, Mapping):
        raise ProfileValidationError("profile bundle snapshot is malformed")
    jurisdiction_schema_version = bundle_snapshot.get(
        "jurisdiction_schema_version"
    )
    if (
        type(jurisdiction_schema_version) is not int
        or jurisdiction_schema_version not in {1, 2, 3}
    ):
        raise ProfileValidationError(
            "profile monetary structure has invalid jurisdiction schema metadata"
        )
    raw_sources = bundle_snapshot.get("sources")
    raw_scales = bundle_snapshot.get("money_scales")
    raw_conversions = bundle_snapshot.get("monetary_conversions")
    if (
        not isinstance(raw_sources, list)
        or not isinstance(raw_scales, list)
        or not isinstance(raw_conversions, list)
        or any(
            not isinstance(row, Mapping)
            for row in (*raw_sources, *raw_scales, *raw_conversions)
        )
    ):
        raise ProfileValidationError(
            "profile monetary structure tables are malformed"
        )
    sources = tuple(_source_from_snapshot(row) for row in raw_sources)
    scales = tuple(_money_scale_from_snapshot(row) for row in raw_scales)
    conversions = tuple(
        _monetary_conversion_from_snapshot(row) for row in raw_conversions
    )
    scale_codes = tuple(scale.jurisdiction_code for scale in scales)
    conversion_codes = tuple(
        conversion.jurisdiction_code for conversion in conversions
    )
    if len(set(scale_codes)) != len(scale_codes) or len(
        set(conversion_codes)
    ) != len(conversion_codes):
        raise ProfileValidationError(
            "profile monetary structure jurisdiction codes are malformed"
        )
    if len(scales) > 1 and scale_codes != _EXPECTED_CODES:
        raise ProfileValidationError(
            "profile monetary structure jurisdiction order does not match "
            "the profile schema"
        )
    unknown_conversion_codes = sorted(
        set(conversion_codes).difference(scale_codes)
    )
    if unknown_conversion_codes:
        raise ProfileValidationError(
            "profile monetary conversions contain unknown jurisdictions: "
            + ", ".join(unknown_conversion_codes)
        )
    conversion_ids = tuple(
        conversion.conversion_id
        for conversion in conversions
        if conversion.conversion_id is not None
    )
    if len(set(conversion_ids)) != len(conversion_ids):
        raise ProfileValidationError(
            "profile monetary conversions repeat a conversion id"
        )
    rate_binding_ids = tuple(
        conversion.rate_binding_id
        for conversion in conversions
        if conversion.rate_binding_id is not None
    )
    if len(set(rate_binding_ids)) != len(rate_binding_ids):
        raise ProfileValidationError(
            "profile monetary conversions repeat a rate-binding id"
        )
    if jurisdiction_schema_version == 3 and any(
        conversion.conversion_id is None
        or conversion.rate_binding_id is None
        for conversion in conversions
    ):
        raise ProfileValidationError(
            "jurisdiction schema version 3 requires monetary conversion ids"
        )
    if jurisdiction_schema_version < 3 and any(
        conversion.conversion_id is not None
        or conversion.rate_binding_id is not None
        for conversion in conversions
    ):
        raise ProfileValidationError(
            "legacy jurisdiction schemas cannot carry monetary conversion ids"
        )
    sources_by_id = {source.id: source for source in sources}
    if len(sources_by_id) != len(sources):
        raise ProfileValidationError("profile source snapshot repeats a source id")
    retrieval_dates = {source.retrieved_on for source in sources}
    if len(retrieval_dates) > 1:
        raise ProfileValidationError(
            "profile source snapshot has inconsistent retrieval dates"
        )
    conversions_by_code = {
        conversion.jurisdiction_code: conversion
        for conversion in conversions
    }
    scales_by_code = {scale.jurisdiction_code: scale for scale in scales}

    referenced_scale_sources = {
        source_id for scale in scales for source_id in scale.source_ids
    }
    unknown_scale_sources = sorted(
        referenced_scale_sources.difference(sources_by_id)
    )
    if unknown_scale_sources:
        raise ProfileValidationError(
            "profile money scales reference unknown source ids: "
            + ", ".join(unknown_scale_sources)
        )
    for conversion in conversions:
        scale = scales_by_code[conversion.jurisdiction_code]
        if conversion.source_currency != scale.currency:
            raise ProfileValidationError(
                "profile monetary conversion source currency does not match "
                "its money scale"
            )
        unknown_source_ids = sorted(
            set(conversion.source_ids).difference(sources_by_id)
        )
        if unknown_source_ids:
            raise ProfileValidationError(
                "profile monetary conversion references unknown source ids: "
                + ", ".join(unknown_source_ids)
            )
        conversion_sources = tuple(
            sources_by_id[source_id] for source_id in conversion.source_ids
        )
        required_supports = _CONVERSION_SOURCE_SUPPORTS[conversion.method.value]
        compatible_sources = tuple(
            source
            for source in conversion_sources
            if required_supports.intersection(source.supports)
        )
        if not compatible_sources:
            raise ProfileValidationError(
                "profile monetary conversion sources do not declare "
                "method-compatible scope"
            )
        if not any(
            source.period == conversion.rate_period_label
            for source in compatible_sources
        ):
            raise ProfileValidationError(
                "profile monetary conversion period does not match a "
                "compatible source"
            )
        if conversion.status is ProvenanceStatus.CALIBRATED and any(
            source.status is not ProvenanceStatus.CALIBRATED
            for source in conversion_sources
        ):
            raise ProfileValidationError(
                "profile calibrated monetary conversion cites "
                "non-calibrated source evidence"
            )
        if {source.retrieved_on for source in conversion_sources} != {
            conversion.retrieved_on
        }:
            raise ProfileValidationError(
                "profile monetary conversion retrieval date does not match "
                "its source records"
            )

    if len(scales) <= 1:
        return False, ("monetary_conversion.structure=unavailable",)

    failures: list[str] = []
    for code in scale_codes:
        conversion = conversions_by_code.get(code)
        if conversion is None:
            failures.append(f"{code}.monetary_conversion=missing")
            continue
        if conversion.status is not ProvenanceStatus.CALIBRATED:
            failures.append(
                f"{code}.monetary_conversion={conversion.status.value}"
            )
    if failures:
        return False, tuple(failures)

    signatures: set[tuple[str, ...]] = set()
    simulation_per_target: set[Fraction] = set()
    for scale in scales:
        code = scale.jurisdiction_code
        conversion = conversions_by_code[code]
        signatures.add(conversion.comparison_signature)
        simulation_per_target.add(
            scale.currency_scale_to_sim / conversion.conversion_ratio
        )
    global_failures: list[str] = []
    if len(signatures) != 1:
        global_failures.append(
            "monetary_conversion.comparison_basis=inconsistent"
        )
    if len(simulation_per_target) != 1:
        global_failures.append("monetary_conversion.internal_scale=incoherent")
    return not global_failures, tuple(global_failures)


def _source_from_snapshot(value: Mapping[object, object]) -> SourceProvenance:
    if set(value) != _SOURCE_PROVENANCE_SNAPSHOT_FIELDS:
        raise ProfileValidationError("profile source snapshot fields are malformed")
    supports = value.get("supports")
    if not isinstance(supports, list):
        raise ProfileValidationError("profile source supports are malformed")
    retrieved_on = value.get("retrieved_on")
    try:
        return SourceProvenance(
            id=value.get("id"),
            publisher=value.get("publisher"),
            title=value.get("title"),
            url=value.get("url"),
            period=value.get("period"),
            geography=value.get("geography"),
            supports=tuple(supports),
            calibration_status=_provenance_status_from_snapshot(
                value.get("calibration_status"),
                context="profile source calibration status",
            ),
            retrieved_on=(
                _parse_iso_date(retrieved_on, "profile source retrieved_on")
                if retrieved_on is not None
                else None
            ),
        )
    except (AttributeError, TypeError) as exc:
        raise ProfileValidationError(
            "profile source snapshot values are malformed"
        ) from exc


def _money_scale_from_snapshot(value: Mapping[object, object]) -> MoneyScaleContract:
    if set(value) != _MONEY_SCALE_SNAPSHOT_FIELDS:
        raise ProfileValidationError(
            "profile money-scale snapshot fields are malformed"
        )
    reported_values = value.get("reported_income_values")
    source_ids = value.get("source_ids")
    if not isinstance(reported_values, list) or not isinstance(source_ids, list):
        raise ProfileValidationError("profile money-scale arrays are malformed")
    if type(value.get("cross_country_comparable")) is not bool:
        raise ProfileValidationError(
            "profile money-scale comparability flag must be boolean"
        )
    try:
        return MoneyScaleContract(
            jurisdiction_code=value.get("jurisdiction_code"),
            currency=value.get("currency"),
            reported_income_values=tuple(reported_values),
            source_period=value.get("source_period"),
            nominal_monthly_anchor_minor_units=value.get(
                "nominal_monthly_anchor_minor_units"
            ),
            anchor_selection=value.get("anchor_selection"),
            anchor_status=_provenance_status_from_snapshot(
                value.get("anchor_status"),
                context="profile money-scale anchor status",
            ),
            source_ids=tuple(source_ids),
            condition=value.get("condition"),
            denominator=value.get("denominator"),
            simulation_monthly_anchor_cents=value.get(
                "simulation_monthly_anchor_cents"
            ),
            scale_status=_provenance_status_from_snapshot(
                value.get("scale_status"),
                context="profile money-scale status",
            ),
            cross_country_comparable=value.get("cross_country_comparable"),
        )
    except (AttributeError, TypeError) as exc:
        raise ProfileValidationError(
            "profile money-scale snapshot values are malformed"
        ) from exc


def _monetary_conversion_from_snapshot(
    value: Mapping[object, object],
) -> MonetaryConversionContract:
    if set(value) != _MONETARY_CONVERSION_SNAPSHOT_FIELDS:
        raise ProfileValidationError(
            "profile monetary-conversion snapshot fields are malformed"
        )
    source_ids = value.get("source_ids")
    if not isinstance(source_ids, list):
        raise ProfileValidationError(
            "profile monetary-conversion source ids are malformed"
        )
    try:
        method = MonetaryConversionMethod(value.get("method"))
        rounding_scope = MonetaryRoundingScope(value.get("rounding_scope"))
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError(
            "profile monetary-conversion enum value is malformed"
        ) from exc
    try:
        conversion = MonetaryConversionContract(
            jurisdiction_code=value.get("jurisdiction_code"),
            source_currency=value.get("source_currency"),
            target_currency=value.get("target_currency"),
            method=method,
            rate_numerator=value.get("rate_numerator"),
            rate_denominator=value.get("rate_denominator"),
            rate_period_start=_parse_iso_date(
                value.get("rate_period_start"),
                "profile monetary-conversion rate_period_start",
            ),
            rate_period_end=_parse_iso_date(
                value.get("rate_period_end"),
                "profile monetary-conversion rate_period_end",
            ),
            target_price_period_start=_parse_iso_date(
                value.get("target_price_period_start"),
                "profile monetary-conversion target_price_period_start",
            ),
            target_price_period_end=_parse_iso_date(
                value.get("target_price_period_end"),
                "profile monetary-conversion target_price_period_end",
            ),
            estimand=value.get("estimand"),
            population_base=value.get("population_base"),
            comparison_group=value.get("comparison_group"),
            rounding_method=value.get("rounding_method"),
            rounding_scope=rounding_scope,
            aggregation_unit=value.get("aggregation_unit"),
            status=_provenance_status_from_snapshot(
                value.get("status"),
                context="profile monetary-conversion status",
            ),
            source_ids=tuple(source_ids),
            retrieved_on=_parse_iso_date(
                value.get("retrieved_on"),
                "profile monetary-conversion retrieved_on",
            ),
            notes=value.get("notes"),
            conversion_id=value.get("conversion_id"),
            rate_binding_id=value.get("rate_binding_id"),
        )
    except (AttributeError, TypeError) as exc:
        raise ProfileValidationError(
            "profile monetary-conversion snapshot values are malformed"
        ) from exc
    if (
        value.get("rate_numerator_decimal") != str(conversion.rate_numerator)
        or value.get("rate_denominator_decimal")
        != str(conversion.rate_denominator)
    ):
        raise ProfileValidationError(
            "profile monetary decimal mirrors are malformed"
        )
    return conversion


def _provenance_status_from_snapshot(
    value: object,
    *,
    context: str,
) -> ProvenanceStatus:
    try:
        return ProvenanceStatus(value)
    except (TypeError, ValueError) as exc:
        raise ProfileValidationError(f"{context} is malformed") from exc


@dataclass(frozen=True, slots=True)
class ProfileBundle:
    """Validated inputs needed to initialise players and jurisdiction agents."""

    country_profiles: tuple[CountryProfile, ...]
    state_agents: tuple[StateAgent, ...]
    sources: Mapping[str, SourceProvenance]
    profile_status: ProvenanceStatus
    caveats: tuple[str, ...]
    contracts: tuple[MetricContract, ...] = ()
    money_scales: tuple[MoneyScaleContract, ...] = ()
    monetary_conversions: tuple[MonetaryConversionContract, ...] = ()
    jurisdictions_path: Path | None = None
    source_registry_path: Path | None = None
    jurisdictions_sha256: str | None = None
    source_registry_sha256: str | None = None
    source_evidence_bundle: RateEvidenceBundle | None = None
    jurisdiction_schema_version: int = 2
    source_catalogue_schema_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "country_profiles",
            "state_agents",
            "caveats",
            "contracts",
            "money_scales",
            "monetary_conversions",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise ProfileValidationError(
                    f"profile bundle {name} must be an immutable tuple"
                )
        frozen_sources = MappingProxyType(dict(self.sources))
        object.__setattr__(self, "sources", frozen_sources)

        profile_codes = tuple(profile.code for profile in self.country_profiles)
        state_codes = tuple(state.code for state in self.state_agents)
        money_codes = tuple(scale.jurisdiction_code for scale in self.money_scales)
        conversion_codes = tuple(
            contract.jurisdiction_code for contract in self.monetary_conversions
        )
        conversion_ids = tuple(
            contract.conversion_id
            for contract in self.monetary_conversions
            if contract.conversion_id is not None
        )
        if profile_codes != _EXPECTED_CODES:
            raise ProfileValidationError(
                f"expected country profiles {_EXPECTED_CODES}; got {profile_codes}"
            )
        if state_codes != profile_codes or money_codes != profile_codes:
            raise ProfileValidationError(
                "country profiles, state agents, and money contracts must align"
            )
        if len(set(conversion_codes)) != len(conversion_codes):
            raise ProfileValidationError(
                "monetary conversion contracts repeat a jurisdiction"
            )
        if len(set(conversion_ids)) != len(conversion_ids):
            raise ProfileValidationError(
                "monetary conversion contracts repeat a conversion id"
            )
        unknown_conversion_codes = sorted(set(conversion_codes).difference(profile_codes))
        if unknown_conversion_codes:
            raise ProfileValidationError(
                "monetary conversions reference unknown jurisdictions: "
                + ", ".join(unknown_conversion_codes)
            )
        if len(frozen_sources) != len(set(frozen_sources)):
            raise ProfileValidationError("source catalogue contains duplicate ids")
        if not isinstance(self.profile_status, ProvenanceStatus):
            raise ProfileValidationError("profile_status is invalid")
        if any(not caveat.strip() for caveat in self.caveats):
            raise ProfileValidationError("bundle caveats cannot be empty")
        if (
            type(self.jurisdiction_schema_version) is not int
            or self.jurisdiction_schema_version not in {1, 2, 3}
        ):
            raise ProfileValidationError(
                "profile bundle has an unsupported jurisdiction schema version"
            )
        if (
            type(self.source_catalogue_schema_version) is not int
            or self.source_catalogue_schema_version != 1
        ):
            raise ProfileValidationError(
                "profile bundle has an unsupported source-catalogue schema version"
            )
        if self.jurisdiction_schema_version < 3 and conversion_ids:
            raise ProfileValidationError(
                "jurisdiction schema versions 1/2 cannot claim rate bindings"
            )
        if self.jurisdiction_schema_version == 3 and any(
            conversion.conversion_id is None
            or conversion.rate_binding_id is None
            for conversion in self.monetary_conversions
        ):
            raise ProfileValidationError(
                "jurisdiction schema version 3 requires conversion and rate-binding ids"
            )

        retrieval_dates = {source.retrieved_on for source in frozen_sources.values()}
        if len(retrieval_dates) > 1:
            raise ProfileValidationError(
                "source catalogue records must share one global retrieval date"
            )
        for name in ("jurisdictions_path", "source_registry_path"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise ProfileValidationError(f"{name} must be a Path when supplied")
        for name in ("jurisdictions_sha256", "source_registry_sha256"):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise ProfileValidationError(
                    f"{name} must be a lowercase SHA-256 digest when supplied"
                )
        if self.source_evidence_bundle is not None:
            from .rate_evidence import RateEvidenceBundle

            if type(self.source_evidence_bundle) is not RateEvidenceBundle:
                raise ProfileValidationError(
                    "source evidence must be a typed RateEvidenceBundle"
                )
            unknown_evidence_sources = sorted(
                {
                    binding.source_id
                    for binding in self.source_evidence_bundle.bindings
                }.difference(frozen_sources)
            )
            if unknown_evidence_sources:
                raise ProfileValidationError(
                    "rate evidence references unknown source ids: "
                    + ", ".join(unknown_evidence_sources)
                )

        referenced = {
            source_id
            for profile in self.country_profiles
            for source_id in profile.source_ids
        }
        referenced.update(
            source_id for contract in self.contracts for source_id in contract.source_ids
        )
        referenced.update(
            source_id for scale in self.money_scales for source_id in scale.source_ids
        )
        referenced.update(
            source_id
            for conversion in self.monetary_conversions
            for source_id in conversion.source_ids
        )
        missing = sorted(referenced.difference(frozen_sources))
        if missing:
            raise ProfileValidationError(
                f"profile references unknown source ids: {', '.join(missing)}"
            )

        scales_by_code = {
            scale.jurisdiction_code: scale for scale in self.money_scales
        }
        for conversion in self.monetary_conversions:
            scale = scales_by_code[conversion.jurisdiction_code]
            if conversion.source_currency != scale.currency:
                raise ProfileValidationError(
                    f"{conversion.jurisdiction_code} monetary conversion source "
                    f"currency {conversion.source_currency} does not match "
                    f"money-scale currency {scale.currency}"
                )
            required_supports = _CONVERSION_SOURCE_SUPPORTS[
                conversion.method.value
            ]
            compatible_sources = tuple(
                frozen_sources[source_id]
                for source_id in conversion.source_ids
                if required_supports.intersection(
                    frozen_sources[source_id].supports
                )
            )
            if not compatible_sources:
                raise ProfileValidationError(
                    f"{conversion.jurisdiction_code} {conversion.method.value} "
                    "conversion sources do not declare compatible scope"
                )
            if not any(
                source.period == conversion.rate_period_label
                for source in compatible_sources
            ):
                raise ProfileValidationError(
                    f"{conversion.jurisdiction_code} monetary conversion rate "
                    "period does not match a compatible source record"
                )
            if (
                conversion.status is ProvenanceStatus.CALIBRATED
                and any(
                    frozen_sources[source_id].status
                    is not ProvenanceStatus.CALIBRATED
                    for source_id in conversion.source_ids
                )
            ):
                raise ProfileValidationError(
                    f"{conversion.jurisdiction_code} monetary conversion cannot "
                    "be CALIBRATED from a non-calibrated source"
                )
            source_dates = {
                frozen_sources[source_id].retrieved_on
                for source_id in conversion.source_ids
            }
            if source_dates != {conversion.retrieved_on}:
                raise ProfileValidationError(
                    f"{conversion.jurisdiction_code} monetary conversion retrieval "
                    "date does not match its source records"
                )
        if self.jurisdiction_schema_version == 3 and self.monetary_conversions:
            _validate_monetary_rate_bindings(self)

    @property
    def provenance(self) -> tuple[MetricContract, ...]:
        """Alias that makes the evidence contracts discoverable to callers."""

        return self.contracts

    @property
    def source_retrieved_on(self) -> date | None:
        """Return the catalogue-wide retrieval date retained on each source."""

        return next(
            (source.retrieved_on for source in self.sources.values()),
            None,
        )

    def matches_registered_files(self) -> bool:
        """Return whether every retained value matches the claimed input files."""

        if any(
            value is None
            for value in (
                self.jurisdictions_path,
                self.jurisdictions_sha256,
                self.source_registry_path,
                self.source_registry_sha256,
                self.source_retrieved_on,
            )
        ):
            return False
        assert self.jurisdictions_path is not None
        assert self.source_registry_path is not None
        try:
            loaded = load_profile_bundle(
                self.jurisdictions_path,
                self.source_registry_path,
                source_bundle_path=(
                    self.source_evidence_bundle.bundle_path
                    if self.source_evidence_bundle is not None
                    else None
                ),
                campaign=False,
            )
        except (OSError, ProfileValidationError):
            return False
        return (
            loaded.country_profiles == self.country_profiles
            and loaded.state_agents == self.state_agents
            and loaded.sources == self.sources
            and loaded.profile_status is self.profile_status
            and loaded.caveats == self.caveats
            and loaded.contracts == self.contracts
            and loaded.money_scales == self.money_scales
            and loaded.monetary_conversions == self.monetary_conversions
            and loaded.jurisdictions_path == self.jurisdictions_path
            and loaded.source_registry_path == self.source_registry_path
            and loaded.jurisdictions_sha256 == self.jurisdictions_sha256
            and loaded.source_registry_sha256 == self.source_registry_sha256
            and loaded.source_evidence_bundle == self.source_evidence_bundle
            and loaded.jurisdiction_schema_version
            == self.jurisdiction_schema_version
            and loaded.source_catalogue_schema_version
            == self.source_catalogue_schema_version
            and loaded.source_retrieved_on == self.source_retrieved_on
        )

    def money_scale(self, jurisdiction_code: str) -> MoneyScaleContract:
        for scale in self.money_scales:
            if scale.jurisdiction_code == jurisdiction_code:
                return scale
        raise KeyError(jurisdiction_code)

    def monetary_conversion(
        self,
        jurisdiction_code: str,
    ) -> MonetaryConversionContract:
        for conversion in self.monetary_conversions:
            if conversion.jurisdiction_code == jurisdiction_code:
                return conversion
        raise KeyError(jurisdiction_code)

    def validate_for_campaign(self) -> None:
        """Reject any campaign containing a non-calibrated dependency."""

        failures: list[str] = []
        if not self.matches_registered_files():
            failures.append("profile_file_lineage=unregistered_or_changed")
        if self.profile_status is not ProvenanceStatus.CALIBRATED:
            failures.append(f"profile_status={self.profile_status.value}")
        failures.extend(
            f"{contract.jurisdiction_code}.{contract.metric}={contract.status.value}"
            for contract in self.contracts
            if contract.status is not ProvenanceStatus.CALIBRATED
        )
        for scale in self.money_scales:
            if scale.anchor_status is not ProvenanceStatus.CALIBRATED:
                failures.append(
                    f"{scale.jurisdiction_code}.income_anchor="
                    f"{scale.anchor_status.value}"
                )
            if scale.scale_status is not ProvenanceStatus.CALIBRATED:
                failures.append(
                    f"{scale.jurisdiction_code}.currency_scale="
                    f"{scale.scale_status.value}"
                )
        monetary_failures = self._monetary_campaign_failures()
        failures.extend(monetary_failures)

        used_source_ids = {
            source_id
            for profile in self.country_profiles
            for source_id in profile.source_ids
        }
        used_source_ids.update(
            source_id for contract in self.contracts for source_id in contract.source_ids
        )
        used_source_ids.update(
            source_id for scale in self.money_scales for source_id in scale.source_ids
        )
        used_source_ids.update(
            source_id
            for conversion in self.monetary_conversions
            for source_id in conversion.source_ids
        )
        for source_id in sorted(used_source_ids):
            source = self.sources[source_id]
            if source.status is not ProvenanceStatus.CALIBRATED:
                failures.append(f"source:{source_id}={source.status.value}")

        if failures:
            preview_limit = 16
            preview = ", ".join(failures[:preview_limit])
            if len(failures) > preview_limit:
                preview += f", ... ({len(failures) - preview_limit} more)"
            hidden_monetary = [
                failure
                for failure in monetary_failures
                if failure not in failures[:preview_limit]
            ]
            if hidden_monetary:
                preview += "; monetary comparability: " + ", ".join(
                    hidden_monetary
                )
            raise ProfileValidationError(
                "Scientific campaigns require CALIBRATED profile dependencies "
                f"and bound comparability evidence; found {preview}"
            )

    def validate_monetary_comparability_for_campaign(self) -> None:
        """Reject pooled-money claims until source rates are evidence-bound."""

        failures = [
            f"{scale.jurisdiction_code}.income_anchor={scale.anchor_status.value}"
            for scale in self.money_scales
            if scale.anchor_status is not ProvenanceStatus.CALIBRATED
        ]
        failures.extend(
            f"{scale.jurisdiction_code}.currency_scale={scale.scale_status.value}"
            for scale in self.money_scales
            if scale.scale_status is not ProvenanceStatus.CALIBRATED
        )
        failures.extend(self._monetary_campaign_failures())
        if failures:
            raise ProfileValidationError(
                "Pooled monetary outputs are not campaign-comparable: "
                + ", ".join(failures)
            )

    def validate_monetary_contract_structure(self) -> None:
        """Validate exact rate mechanics without claiming substantive evidence."""

        failures = [
            f"{scale.jurisdiction_code}.income_anchor={scale.anchor_status.value}"
            for scale in self.money_scales
            if scale.anchor_status is not ProvenanceStatus.CALIBRATED
        ]
        failures.extend(
            f"{scale.jurisdiction_code}.currency_scale={scale.scale_status.value}"
            for scale in self.money_scales
            if scale.scale_status is not ProvenanceStatus.CALIBRATED
        )
        failures.extend(self._monetary_contract_structure_failures())
        if failures:
            raise ProfileValidationError(
                "Pooled monetary contract structure is invalid: "
                + ", ".join(failures)
            )

    def _monetary_campaign_failures(self) -> list[str]:
        """Return fail-closed substantive comparability failures."""

        return list(self.monetary_evidence_assessment().blockers)

    def monetary_evidence_assessment(
        self,
        *,
        registered: bool | None = None,
    ) -> MonetaryEvidenceAssessment:
        """Assess independent monetary evidence gates without self-promotion."""

        structure_failures = self._monetary_contract_structure_failures()
        structure_coherent = not structure_failures and len(self.money_scales) > 1
        if registered is None:
            registered = self.matches_registered_files()
        source_bound = (
            structure_coherent
            and registered
            and self.jurisdiction_schema_version == 3
            and self.source_evidence_bundle is not None
            and _rate_bindings_cover_conversions(self)
        )
        # Schema-v1 bundles deliberately have no verified trust-root signature.
        source_signature_bound = False

        # These require future run-specific output bindings, a calibrated
        # population specification, and an externally immutable preregistration.
        # They cannot be promoted by configuration-file booleans.
        output_design_bound = False
        population_bound = False
        preregistration_bound = False
        blockers = list(structure_failures)
        if not source_bound:
            blockers.append("monetary_conversion.source_rate_binding=missing")
        if not source_signature_bound:
            blockers.append("monetary_conversion.source_bundle_signature=missing")
        if not output_design_bound:
            blockers.append("monetary_conversion.output_design_binding=missing")
        if not population_bound:
            blockers.append("monetary_conversion.population_binding=missing")
        if not preregistration_bound:
            blockers.append("monetary_conversion.preregistration_binding=missing")
        public_comparable = all(
            (
                structure_coherent,
                source_bound,
                source_signature_bound,
                output_design_bound,
                population_bound,
                preregistration_bound,
            )
        )
        return MonetaryEvidenceAssessment(
            structure_coherent=structure_coherent,
            source_rate_evidence_bound=source_bound,
            source_bundle_signature_bound=source_signature_bound,
            output_design_binding_bound=output_design_bound,
            population_binding_bound=population_bound,
            preregistration_bound=preregistration_bound,
            public_output_comparability=public_comparable,
            blockers=tuple(blockers),
        )

    def _monetary_contract_structure_failures(self) -> list[str]:
        """Return exact-rate structural failures without promoting evidence."""

        if len(self.money_scales) <= 1:
            return ["monetary_conversion.structure=unavailable"]
        failures: list[str] = []
        conversions = {
            contract.jurisdiction_code: contract
            for contract in self.monetary_conversions
        }
        for scale in self.money_scales:
            conversion = conversions.get(scale.jurisdiction_code)
            if conversion is None:
                failures.append(
                    f"{scale.jurisdiction_code}.monetary_conversion=missing"
                )
            elif conversion.status is not ProvenanceStatus.CALIBRATED:
                failures.append(
                    f"{scale.jurisdiction_code}.monetary_conversion="
                    f"{conversion.status.value}"
                )
        if failures:
            return failures

        ordered = tuple(conversions[scale.jurisdiction_code] for scale in self.money_scales)
        if len({contract.comparison_signature for contract in ordered}) != 1:
            failures.append("monetary_conversion.comparison_basis=inconsistent")

        # Each local-to-simulation scale divided by its local-to-target rate is
        # the exact number of simulation cents per common target minor unit.
        # Those ratios must agree; matching currency labels alone is insufficient.
        simulation_per_target = {
            scale.currency_scale_to_sim / conversion.conversion_ratio
            for scale, conversion in zip(self.money_scales, ordered, strict=True)
        }
        if len(simulation_per_target) != 1:
            failures.append("monetary_conversion.internal_scale=incoherent")
        return failures

    def validate_for_run(self, *, allow_synthetic: bool) -> None:
        """Apply the scenario's synthetic-data switch to profile dependencies."""

        if allow_synthetic:
            return
        synthetic: list[str] = []
        if self.profile_status is ProvenanceStatus.SYNTHETIC:
            synthetic.append("profile_status")
        synthetic.extend(
            f"{contract.jurisdiction_code}.{contract.metric}"
            for contract in self.contracts
            if contract.status is ProvenanceStatus.SYNTHETIC
        )
        synthetic.extend(
            f"{scale.jurisdiction_code}.money_scale"
            for scale in self.money_scales
            if scale.anchor_status is ProvenanceStatus.SYNTHETIC
            or scale.scale_status is ProvenanceStatus.SYNTHETIC
        )
        synthetic.extend(
            f"{conversion.jurisdiction_code}.monetary_conversion"
            for conversion in self.monetary_conversions
            if conversion.status is ProvenanceStatus.SYNTHETIC
        )
        if synthetic:
            raise ProfileValidationError(
                "Profile bundle contains SYNTHETIC dependencies while "
                "allow_synthetic=false: " + ", ".join(synthetic)
            )


def load_profile_bundle(
    jurisdictions_path: str | Path = DEFAULT_JURISDICTIONS_PATH,
    sources_path: str | Path = DEFAULT_SOURCES_PATH,
    *,
    source_bundle_path: str | Path | None | object = _USE_REGISTERED_SOURCE_BUNDLE,
    campaign: bool = False,
) -> ProfileBundle:
    """Load, validate, and harmonise the four jurisdiction profiles."""

    jurisdiction_file = Path(jurisdictions_path)
    sources_file = Path(sources_path)
    jurisdiction_raw, jurisdiction_sha256 = _read_toml(
        jurisdiction_file,
        "jurisdiction profiles",
    )
    sources_raw, source_registry_sha256 = _read_toml(
        sources_file,
        "source catalogue",
    )
    sources = _parse_sources(sources_raw, sources_file)

    selected_source_bundle_path: Path | None
    if source_bundle_path is _USE_REGISTERED_SOURCE_BUNDLE:
        selected_source_bundle_path = (
            DEFAULT_SOURCE_BUNDLE_PATH
            if sources_file.resolve() == DEFAULT_SOURCES_PATH.resolve()
            else None
        )
    elif source_bundle_path is None:
        selected_source_bundle_path = None
    elif isinstance(source_bundle_path, (str, Path)):
        selected_source_bundle_path = Path(source_bundle_path)
    else:
        raise TypeError("source_bundle_path must be a path or None")

    jurisdiction_schema_version = jurisdiction_raw.get("schema_version")
    if (
        type(jurisdiction_schema_version) is not int
        or jurisdiction_schema_version not in {1, 2, 3}
    ):
        raise ProfileValidationError(
            f"{jurisdiction_file}: unsupported jurisdiction schema_version"
        )
    profile_status = _parse_status(
        jurisdiction_raw.get("profile_status"),
        f"{jurisdiction_file}: profile_status",
    )
    _validate_status_fields(jurisdiction_raw, str(jurisdiction_file))
    _validate_source_references(jurisdiction_raw, sources, jurisdiction_file)

    raw_rows = jurisdiction_raw.get("jurisdiction")
    if not isinstance(raw_rows, list) or any(
        not isinstance(row, dict) for row in raw_rows
    ):
        raise ProfileValidationError(
            f"{jurisdiction_file}: jurisdiction must be an array of tables"
        )
    rows_by_code: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        code = _required_string(row, "code", f"jurisdiction[{index}]")
        if code in rows_by_code:
            raise ProfileValidationError(f"duplicate jurisdiction code {code}")
        rows_by_code[code] = row
    if set(rows_by_code) != set(_EXPECTED_CODES):
        raise ProfileValidationError(
            f"profiles must contain exactly {', '.join(_EXPECTED_CODES)}"
        )
    if jurisdiction_schema_version == 1:
        v2_fields = {
            "monetary_conversion",
            "simulation_monthly_anchor_cents",
            "currency_scale_status",
        }
        present_v2_fields = set(jurisdiction_raw).intersection(v2_fields)
        present_v2_fields.update(
            field
            for row in rows_by_code.values()
            for field in set(row).intersection(v2_fields)
        )
        if present_v2_fields:
            raise ProfileValidationError(
                f"{jurisdiction_file}: profile schema_version 1 cannot contain "
                "version-2 monetary fields: "
                + ", ".join(sorted(present_v2_fields))
            )
    elif jurisdiction_schema_version == 2:
        raw_conversions = jurisdiction_raw.get("monetary_conversion", [])
        if isinstance(raw_conversions, list):
            present_v3_fields = {
                field
                for row in raw_conversions
                if isinstance(row, dict)
                for field in ("conversion_id", "rate_binding_id")
                if field in row
            }
            if present_v3_fields:
                raise ProfileValidationError(
                    f"{jurisdiction_file}: profile schema_version 2 cannot contain "
                    "version-3 evidence fields: "
                    + ", ".join(sorted(present_v3_fields))
                )

    shared = jurisdiction_raw.get("shared_assumptions")
    if not isinstance(shared, dict):
        raise ProfileValidationError(
            f"{jurisdiction_file}: shared_assumptions must be a table"
        )
    audit_capacity = _positive_int(
        shared.get("audit_capacity_per_cycle"),
        "shared_assumptions.audit_capacity_per_cycle",
    )
    audit_status = _parse_status(
        shared.get("audit_capacity_status"),
        "shared_assumptions.audit_capacity_status",
    )
    if audit_status is not ProvenanceStatus.SYNTHETIC:
        raise ProfileValidationError(
            "audit capacity currently has no empirical capacity source and must be "
            "marked SYNTHETIC"
        )

    country_profiles: list[CountryProfile] = []
    state_agents: list[StateAgent] = []
    money_scales: list[MoneyScaleContract] = []
    contracts: list[MetricContract] = []

    for jurisdiction_id, code in enumerate(_EXPECTED_CODES):
        row = rows_by_code[code]
        profile, money_scale = _country_profile(code, row, sources)
        country_profiles.append(profile)
        money_scales.append(money_scale)
        contracts.extend(_jurisdiction_contracts(code, row, sources))
        state_agents.append(
            _state_agent(jurisdiction_id, code, row, audit_capacity=audit_capacity)
        )

    monetary_conversions = _parse_monetary_conversions(
        jurisdiction_raw.get("monetary_conversion", []),
        jurisdiction_file,
        schema_version=jurisdiction_schema_version,
    )

    contracts.extend(_shared_contracts(shared))
    contracts.append(
        MetricContract(
            jurisdiction_code="*",
            metric="state_agent_operating_parameters",
            value="synthetic scaffold defaults",
            status=ProvenanceStatus.SYNTHETIC,
            source_ids=(),
            condition="when a StateAgent is initialised",
            denominator="one regulator operating model per jurisdiction",
            period="simulation initialisation",
            notes=(
                "Budgets, inspection cost, preferences, audit accuracy, and subsidy "
                "weights are synthetic pending jurisdiction-specific calibration."
            ),
        )
    )

    note = jurisdiction_raw.get("notes", "")
    if not isinstance(note, str):
        raise ProfileValidationError(f"{jurisdiction_file}: notes must be text")
    caveats = tuple(
        item
        for item in (
            note.strip(),
            (
                "Nominal GBP, KRW, JPY, and EUR anchors are provenance records only "
                "and must not be ranked or compared across countries."
            ),
            (
                "CountryProfile money uses simulation purchasing-power cents: each "
                "monthly country anchor is mapped to 180000 internal cents by an "
                "explicitly ILLUSTRATIVE scale, not an FX or PPP estimate."
            ),
            (
                "UK and Belgium annual anchors are stored in currency minor units, "
                "divided by 12, and rounded to the nearest minor unit; Korea uses the "
                "central monthly income "
                "quintile; Japan's income anchor is ILLUSTRATIVE."
            ),
            (
                "Audit capacity and all other StateAgent operating parameters not "
                "present in the source tables are SYNTHETIC."
            ),
            (
                "Japan's complete-gacha restriction remains an explicit evidence "
                "contract; the generic RegulationRules type does not broaden it into "
                "a ban on every paid random reward."
            ),
            (
                "Subsidy rates, caps, and instruments remain provenance contracts; "
                "the current StateAgent award budget and scoring weights are SYNTHETIC."
            ),
            _optional_caveat(rows_by_code["KR"], "mobile_spend_population_base"),
            _optional_caveat(rows_by_code["JP"], "gaming_data_caveat"),
        )
        if item
    )

    bundle = ProfileBundle(
        country_profiles=tuple(country_profiles),
        state_agents=tuple(state_agents),
        sources=sources,
        profile_status=profile_status,
        caveats=caveats,
        contracts=tuple(contracts),
        money_scales=tuple(money_scales),
        monetary_conversions=monetary_conversions,
        jurisdictions_path=jurisdiction_file.resolve(),
        source_registry_path=sources_file.resolve(),
        jurisdictions_sha256=jurisdiction_sha256,
        source_registry_sha256=source_registry_sha256,
        source_evidence_bundle=_load_source_evidence_bundle(
            selected_source_bundle_path,
            sources=sources,
            source_registry_sha256=source_registry_sha256,
        ),
        jurisdiction_schema_version=jurisdiction_schema_version,
        source_catalogue_schema_version=1,
    )
    if campaign:
        bundle.validate_for_campaign()
    return bundle


def load_country_profiles(
    jurisdictions_path: str | Path = DEFAULT_JURISDICTIONS_PATH,
    sources_path: str | Path = DEFAULT_SOURCES_PATH,
    *,
    source_bundle_path: str | Path | None | object = _USE_REGISTERED_SOURCE_BUNDLE,
    campaign: bool = False,
) -> tuple[CountryProfile, ...]:
    """Convenience wrapper returning only the player-initialisation profiles."""

    return load_profile_bundle(
        jurisdictions_path,
        sources_path,
        source_bundle_path=source_bundle_path,
        campaign=campaign,
    ).country_profiles


def load_state_agents(
    jurisdictions_path: str | Path = DEFAULT_JURISDICTIONS_PATH,
    sources_path: str | Path = DEFAULT_SOURCES_PATH,
    *,
    source_bundle_path: str | Path | None | object = _USE_REGISTERED_SOURCE_BUNDLE,
    campaign: bool = False,
) -> tuple[StateAgent, ...]:
    """Convenience wrapper returning only the jurisdiction agents."""

    return load_profile_bundle(
        jurisdictions_path,
        sources_path,
        source_bundle_path=source_bundle_path,
        campaign=campaign,
    ).state_agents


def _read_toml(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        encoded = path.read_bytes()
        raw = tomllib.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProfileValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileValidationError(f"{label} {path} must contain a TOML table")
    return raw, sha256(encoded).hexdigest()


def _load_source_evidence_bundle(
    path: Path | None,
    *,
    sources: Mapping[str, SourceProvenance],
    source_registry_sha256: str,
) -> RateEvidenceBundle | None:
    """Load and verify a bundle against the exact source catalogue in use."""

    if path is None:
        return None
    from .rate_evidence import (
        RateEvidenceValidationError,
        load_and_verify_rate_evidence_bundle,
    )

    try:
        bundle, _results = load_and_verify_rate_evidence_bundle(
            path,
            required_source_registry_sha256=source_registry_sha256,
        )
    except RateEvidenceValidationError as exc:
        raise ProfileValidationError(
            f"cannot verify source evidence bundle {path}: {exc}"
        ) from exc
    if getattr(bundle, "source_registry_sha256", None) != source_registry_sha256:
        raise ProfileValidationError(
            "source evidence bundle does not bind the loaded source catalogue"
        )
    unknown_sources = sorted(
        {binding.source_id for binding in bundle.bindings}.difference(sources)
    )
    if unknown_sources:
        raise ProfileValidationError(
            "source evidence bundle references unknown source ids: "
            + ", ".join(unknown_sources)
        )
    return bundle


def _validate_monetary_rate_bindings(bundle: ProfileBundle) -> None:
    """Require every schema-v3 conversion to match verified bytes exactly."""

    evidence = bundle.source_evidence_bundle
    if evidence is None:
        raise ProfileValidationError(
            "jurisdiction schema version 3 monetary conversions require a "
            "source evidence bundle"
        )
    if evidence.source_registry_sha256 != bundle.source_registry_sha256:
        raise ProfileValidationError(
            "monetary rate evidence is bound to a different source catalogue"
        )
    from .rate_evidence import (
        RateEvidenceValidationError,
        verify_rate_evidence_bundle,
    )

    try:
        results = verify_rate_evidence_bundle(
            evidence,
            required_source_registry_sha256=bundle.source_registry_sha256,
        )
    except RateEvidenceValidationError as exc:
        raise ProfileValidationError(
            f"monetary rate evidence cannot be re-attested: {exc}"
        ) from exc
    bindings_by_id = {
        binding.binding_id: binding for binding in evidence.bindings
    }
    results_by_id = {result.binding_id: result for result in results}
    referenced_binding_ids = tuple(
        conversion.rate_binding_id
        for conversion in bundle.monetary_conversions
    )
    if len(set(referenced_binding_ids)) != len(referenced_binding_ids):
        raise ProfileValidationError(
            "monetary conversions repeat a rate-evidence binding"
        )
    for conversion in bundle.monetary_conversions:
        assert conversion.rate_binding_id is not None
        binding = bindings_by_id.get(conversion.rate_binding_id)
        result = results_by_id.get(conversion.rate_binding_id)
        if binding is None or result is None:
            raise ProfileValidationError(
                f"monetary conversion {conversion.conversion_id} references an "
                "unknown or unverified rate binding"
            )
        exact_fields = (
            ("jurisdiction", binding.jurisdiction_code, conversion.jurisdiction_code),
            ("source currency", binding.source_currency, conversion.source_currency),
            ("target currency", binding.target_currency, conversion.target_currency),
            ("method", binding.method.value, conversion.method.value),
            ("rate period start", binding.rate_period_start, conversion.rate_period_start),
            ("rate period end", binding.rate_period_end, conversion.rate_period_end),
            ("retrieval date", binding.retrieved_on, conversion.retrieved_on),
            ("rate numerator", result.rate_numerator, conversion.rate_numerator),
            ("rate denominator", result.rate_denominator, conversion.rate_denominator),
        )
        mismatches = [
            label for label, observed, declared in exact_fields
            if observed != declared
        ]
        if mismatches:
            raise ProfileValidationError(
                f"monetary conversion {conversion.conversion_id} does not match "
                "its verified rate binding: " + ", ".join(mismatches)
            )
        if binding.source_id not in conversion.source_ids:
            raise ProfileValidationError(
                f"monetary conversion {conversion.conversion_id} does not cite "
                "the source owned by its rate binding"
            )
        source = bundle.sources[binding.source_id]
        required_supports = _CONVERSION_SOURCE_SUPPORTS[conversion.method.value]
        if not required_supports.intersection(source.supports):
            raise ProfileValidationError(
                f"monetary conversion {conversion.conversion_id} rate-binding "
                "source does not declare method-compatible scope"
            )
        if source.period != conversion.rate_period_label:
            raise ProfileValidationError(
                f"monetary conversion {conversion.conversion_id} rate-binding "
                "source period does not match the exact rate interval"
            )
        if source.retrieved_on != conversion.retrieved_on:
            raise ProfileValidationError(
                f"monetary conversion {conversion.conversion_id} rate-binding "
                "source retrieval date does not match"
            )
        if (
            conversion.status is ProvenanceStatus.CALIBRATED
            and (
                evidence.provenance_status is not ProvenanceStatus.CALIBRATED
                or source.status is not ProvenanceStatus.CALIBRATED
            )
        ):
            raise ProfileValidationError(
                f"monetary conversion {conversion.conversion_id} cannot be "
                "CALIBRATED from non-calibrated rate evidence"
            )


def _rate_bindings_cover_conversions(bundle: ProfileBundle) -> bool:
    if not bundle.monetary_conversions:
        return False
    try:
        _validate_monetary_rate_bindings(bundle)
    except ProfileValidationError:
        return False
    return all(
        conversion.status is ProvenanceStatus.CALIBRATED
        for conversion in bundle.monetary_conversions
    )


def _parse_sources(
    raw: Mapping[str, Any], path: Path
) -> Mapping[str, SourceProvenance]:
    allowed_catalogue_fields = frozenset(
        {"schema_version", "retrieved_on", "source"}
    )
    unknown_catalogue_fields = sorted(set(raw).difference(allowed_catalogue_fields))
    if unknown_catalogue_fields:
        raise ProfileValidationError(
            f"{path}: source catalogue contains unknown fields: "
            + ", ".join(unknown_catalogue_fields)
        )
    source_schema_version = raw.get("schema_version")
    if type(source_schema_version) is not int or source_schema_version != 1:
        raise ProfileValidationError(f"{path}: unsupported source schema_version")
    retrieved_on = _parse_iso_date(raw.get("retrieved_on"), f"{path}: retrieved_on")
    records = raw.get("source")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ProfileValidationError(f"{path}: source must be an array of tables")

    parsed: dict[str, SourceProvenance] = {}
    allowed_source_fields = frozenset(
        {
            "id",
            "publisher",
            "title",
            "url",
            "period",
            "geography",
            "supports",
            "calibration_status",
        }
    )
    for index, row in enumerate(records):
        context = f"{path}: source[{index}]"
        unknown_source_fields = sorted(set(row).difference(allowed_source_fields))
        if unknown_source_fields:
            raise ProfileValidationError(
                f"{context} contains unknown fields: "
                + ", ".join(unknown_source_fields)
            )
        source_id = _required_string(row, "id", context)
        supports_raw = row.get("supports")
        if not isinstance(supports_raw, list) or any(
            not isinstance(value, str) for value in supports_raw
        ):
            raise ProfileValidationError(f"{context}.supports must be a string array")
        source = SourceProvenance(
            id=source_id,
            publisher=_required_string(row, "publisher", context),
            title=_required_string(row, "title", context),
            url=_required_string(row, "url", context),
            period=_required_string(row, "period", context),
            geography=_required_string(row, "geography", context),
            supports=tuple(supports_raw),
            calibration_status=_parse_status(
                row.get("calibration_status"),
                f"{context}.calibration_status",
            ),
            retrieved_on=retrieved_on,
        )
        if source.id in parsed:
            raise ProfileValidationError(f"{path}: duplicate source id {source.id}")
        parsed[source.id] = source
    if not parsed:
        raise ProfileValidationError(f"{path}: source catalogue cannot be empty")
    return MappingProxyType(parsed)


def _parse_monetary_conversions(
    value: object,
    path: Path,
    *,
    schema_version: int,
) -> tuple[MonetaryConversionContract, ...]:
    """Parse optional v2/v3 FX/PPP contracts without fallback rates."""

    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ProfileValidationError(
            f"{path}: monetary_conversion must be an array of tables"
        )
    parsed: list[MonetaryConversionContract] = []
    for index, raw_row in enumerate(value):
        row = raw_row
        context = f"{path}: monetary_conversion[{index}]"
        allowed_fields = (
            _MONETARY_CONVERSION_V3_FIELDS
            if schema_version == 3
            else _MONETARY_CONVERSION_V2_FIELDS
        )
        unknown_fields = sorted(set(row).difference(allowed_fields))
        if unknown_fields:
            raise ProfileValidationError(
                f"{context} contains unknown fields: {', '.join(unknown_fields)}"
            )
        method_value = _required_string(row, "method", context)
        try:
            method = MonetaryConversionMethod(method_value)
        except ValueError as exc:
            raise ProfileValidationError(
                f"{context}.method must be FX or PPP"
            ) from exc
        rounding_scope_value = _required_string(row, "rounding_scope", context)
        try:
            rounding_scope = MonetaryRoundingScope(rounding_scope_value)
        except ValueError as exc:
            raise ProfileValidationError(
                f"{context}.rounding_scope must be PER_OBSERVATION or "
                "AFTER_AGGREGATION"
            ) from exc
        source_ids_raw = row.get("source_ids")
        if not isinstance(source_ids_raw, list) or not source_ids_raw or any(
            not isinstance(source_id, str) or not source_id.strip()
            for source_id in source_ids_raw
        ):
            raise ProfileValidationError(
                f"{context}.source_ids must be a non-empty text array"
            )
        notes = row.get("notes", "")
        if not isinstance(notes, str):
            raise ProfileValidationError(f"{context}.notes must be text")
        parsed.append(
            MonetaryConversionContract(
                jurisdiction_code=_required_string(
                    row, "jurisdiction_code", context
                ),
                source_currency=_required_string(
                    row, "source_currency", context
                ),
                target_currency=_required_string(
                    row, "target_currency", context
                ),
                method=method,
                rate_numerator=_positive_int(
                    row.get("rate_numerator"),
                    f"{context}.rate_numerator",
                ),
                rate_denominator=_positive_int(
                    row.get("rate_denominator"),
                    f"{context}.rate_denominator",
                ),
                rate_period_start=_parse_iso_date(
                    row.get("rate_period_start"),
                    f"{context}.rate_period_start",
                ),
                rate_period_end=_parse_iso_date(
                    row.get("rate_period_end"),
                    f"{context}.rate_period_end",
                ),
                target_price_period_start=_parse_iso_date(
                    row.get("target_price_period_start"),
                    f"{context}.target_price_period_start",
                ),
                target_price_period_end=_parse_iso_date(
                    row.get("target_price_period_end"),
                    f"{context}.target_price_period_end",
                ),
                estimand=_required_string(row, "estimand", context),
                population_base=_required_string(
                    row, "population_base", context
                ),
                comparison_group=_required_string(
                    row, "comparison_group", context
                ),
                rounding_method=_required_string(
                    row, "rounding_method", context
                ),
                rounding_scope=rounding_scope,
                aggregation_unit=_required_string(
                    row, "aggregation_unit", context
                ),
                status=_parse_status(
                    row.get("status"),
                    f"{context}.status",
                ),
                source_ids=tuple(source_ids_raw),
                retrieved_on=_parse_iso_date(
                    row.get("retrieved_on"),
                    f"{context}.retrieved_on",
                ),
                notes=notes,
                conversion_id=(
                    _required_string(row, "conversion_id", context)
                    if schema_version == 3
                    else None
                ),
                rate_binding_id=(
                    _required_string(row, "rate_binding_id", context)
                    if schema_version == 3
                    else None
                ),
            )
        )
    return tuple(parsed)


def _country_profile(
    code: str,
    row: Mapping[str, Any],
    sources: Mapping[str, SourceProvenance],
) -> tuple[CountryProfile, MoneyScaleContract]:
    currency = _required_string(row, "income_currency", code)
    period = _required_string(row, "income_period", code)
    if period not in {"annual", "monthly"}:
        raise ProfileValidationError(f"{code}.income_period must be annual or monthly")
    income_status = _parse_status(row.get("income_status"), f"{code}.income_status")
    income_sources = _source_ids_for(row, "income")
    if income_status in {ProvenanceStatus.ANCHORED, ProvenanceStatus.CALIBRATED} and not income_sources:
        raise ProfileValidationError(
            f"{code} {income_status.value} income requires an income_source"
        )

    if code == "KR":
        quintiles = _positive_int_tuple(
            row.get("disposable_income_quintile_minor_units"),
            f"{code}.disposable_income_quintile_minor_units",
        )
        if period != "monthly" or len(quintiles) % 2 != 1:
            raise ProfileValidationError(
                "KR income needs an odd number of ordered monthly quintile anchors"
            )
        if any(right <= left for left, right in zip(quintiles, quintiles[1:])):
            raise ProfileValidationError("KR income quintile anchors must increase")
        reported = quintiles
        nominal_monthly = quintiles[len(quintiles) // 2]
        selection = "central (third) reported monthly disposable-income quintile"
        income_sigma = _positive_float(
            row.get("income_within_quintile_log_sigma"),
            f"{code}.income_within_quintile_log_sigma",
        )
        condition = "households in the central disposable-income quintile"
        denominator = "households observed by the cited Korean household survey"
    else:
        reported_value = _positive_int(
            row.get("median_equivalised_disposable_income_minor_units"),
            f"{code}.median_equivalised_disposable_income_minor_units",
        )
        reported = (reported_value,)
        if period == "annual":
            nominal_monthly = _round_positive_ratio(reported_value, 12)
            selection = (
                "reported annual median expressed in currency minor units, "
                "divided by 12 and rounded to the nearest minor unit"
            )
        else:
            nominal_monthly = reported_value
            selection = "reported monthly median used directly"
        income_sigma = _positive_float(
            row.get("income_log_sigma"), f"{code}.income_log_sigma"
        )
        condition = "residents represented by the configured income anchor"
        denominator = "one equivalised household-income observation in the source population"

    scale = MoneyScaleContract(
        jurisdiction_code=code,
        currency=currency,
        reported_income_values=reported,
        source_period=period,
        nominal_monthly_anchor_minor_units=nominal_monthly,
        anchor_selection=selection,
        anchor_status=income_status,
        source_ids=income_sources,
        condition=condition,
        denominator=denominator,
        simulation_monthly_anchor_cents=_positive_int(
            row.get(
                "simulation_monthly_anchor_cents",
                _SIMULATION_MONTHLY_INCOME_CENTS,
            ),
            f"{code}.simulation_monthly_anchor_cents",
        ),
        scale_status=_parse_status(
            row.get("currency_scale_status", "ILLUSTRATIVE"),
            f"{code}.currency_scale_status",
        ),
    )
    source_ids = tuple(
        dict.fromkeys(
            value
            for key, value in row.items()
            if key.endswith("_source") and isinstance(value, str)
        )
    )
    unknown = set(source_ids).difference(sources)
    if unknown:
        raise ProfileValidationError(
            f"{code} references unknown sources: {', '.join(sorted(unknown))}"
        )

    edges = _positive_int_tuple(row.get("age_band_edges"), f"{code}.age_band_edges")
    weights = _nonnegative_float_tuple(
        row.get("age_band_weights"), f"{code}.age_band_weights"
    )
    profile = CountryProfile(
        code=code,
        population_weight=_positive_float(
            row.get("population_weight"), f"{code}.population_weight"
        ),
        adult_age=18,
        age_band_edges=edges,
        age_band_weights=weights,
        monthly_income_median_cents=scale.simulation_monthly_anchor_cents,
        income_log_sigma=income_sigma,
        source_ids=source_ids,
    )
    return profile, scale


def _state_agent(
    jurisdiction_id: int,
    code: str,
    row: Mapping[str, Any],
    *,
    audit_capacity: int,
) -> StateAgent:
    rules = RegulationRules(
        odds_disclosure_required=_required_bool(row, "odds_disclosure_required", code),
        real_money_price_required=_required_bool(
            row, "real_money_price_required", code
        ),
        parental_authorisation_required=_required_bool(
            row, "parental_authorisation_required", code
        ),
        direct_exhortation_to_minors_banned=_required_bool(
            row, "direct_exhortation_to_minors_banned", code
        ),
        paid_random_rewards_restricted=_required_bool(
            row, "paid_random_rewards_restricted", code
        ),
        cooling_off_days=0,
        minor_monthly_cap_cents=None,
        maximum_power_sale_intensity=1.0,
    )

    # No jurisdiction-specific operating-budget series is configured yet.  These
    # values are in the same internal simulation cents as player prices/incomes
    # and are covered by an explicit SYNTHETIC contract in the returned bundle.
    inspection_cost_cents = 10_000
    subsidy_budget_cents = 18_000_000
    private_state = RegulatorPrivateState(
        treasury_cents=36_000_000,
        audit_budget_cents=audit_capacity * inspection_cost_cents,
        subsidy_budget_cents=subsidy_budget_cents,
        audit_capacity_per_cycle=audit_capacity,
        inspection_cost_cents=inspection_cost_cents,
    )
    return StateAgent(
        jurisdiction_id=jurisdiction_id,
        code=code,
        rules=rules,
        state=private_state,
        harm_priority=0.75,
        minor_priority=0.85,
        fiscal_priority=0.45,
        industry_priority=0.55,
        random_audit_fraction=0.20,
        audit_sensitivity=0.85,
        audit_specificity=0.95,
        subsidy_quality_weight=0.35,
        subsidy_safe_revenue_weight=0.45,
        subsidy_accessibility_weight=0.20,
    )


def _jurisdiction_contracts(
    code: str,
    row: Mapping[str, Any],
    sources: Mapping[str, SourceProvenance],
) -> tuple[MetricContract, ...]:
    contracts: list[MetricContract] = []
    for field, raw_value in row.items():
        if (
            field in _NON_METRIC_FIELDS
            or field.endswith("_source")
            or field.endswith("_status")
        ):
            continue
        status, source_ids = _metric_status_and_sources(field, row, sources)
        condition, denominator, period, currency, notes = _metric_semantics(
            code, field, row, sources, source_ids
        )
        contracts.append(
            MetricContract(
                jurisdiction_code=code,
                metric=field,
                value=_freeze_value(raw_value),
                status=status,
                source_ids=source_ids,
                condition=condition,
                denominator=denominator,
                period=period,
                currency=currency,
                notes=notes,
            )
        )
    return tuple(contracts)


def _metric_status_and_sources(
    field: str,
    row: Mapping[str, Any],
    sources: Mapping[str, SourceProvenance],
) -> tuple[ProvenanceStatus, tuple[str, ...]]:
    if field == "population_weight":
        return _parse_status(row.get("population_weight_status"), field), ()
    if field in {"age_band_edges", "age_band_weights"}:
        return _parse_status(row.get("age_weights_status"), field), ()
    if field in {
        "income_log_sigma",
        "income_within_quintile_log_sigma",
    }:
        return _parse_status(row.get("income_shape_status"), field), ()
    if field == "consumption_propensity_by_quintile":
        return _parse_status(
            row.get("consumption_propensity_status"), field
        ), _source_ids_for(row, "income")
    if field.startswith("deprivation_"):
        source_ids = _source_ids_for(row, "deprivation")
    elif field in _RULE_FIELDS:
        status = _parse_status(row.get(f"{field}_status"), field)
        source_ids = _source_ids_for(row, field)
        if status in {ProvenanceStatus.ANCHORED, ProvenanceStatus.CALIBRATED}:
            if not source_ids:
                raise ProfileValidationError(
                    f"{field} with {status.value} status requires its own source"
                )
            if status is ProvenanceStatus.CALIBRATED and any(
                sources[source_id].status is not ProvenanceStatus.CALIBRATED
                for source_id in source_ids
            ):
                raise ProfileValidationError(
                    f"{field} cannot be CALIBRATED from a non-calibrated source"
                )
        expected = _RULE_SUPPORTS[field]
        if source_ids and not any(
            expected.intersection(sources[source_id].supports)
            for source_id in source_ids
        ):
            raise ProfileValidationError(
                f"{field} source metadata does not declare a compatible scope"
            )
        return status, source_ids
    elif field.startswith("subsidy_"):
        source_ids = _source_ids_for(row, "subsidy")
    elif field.startswith(("median_equivalised_", "disposable_income_")):
        return _parse_status(row.get("income_status"), field), _source_ids_for(
            row, "income"
        )
    else:
        source_ids = _source_ids_for(row, "gaming")

    if not source_ids:
        raise ProfileValidationError(f"{field} requires provenance source metadata")
    statuses = {sources[source_id].status for source_id in source_ids}
    if len(statuses) != 1:
        raise ProfileValidationError(f"{field} sources disagree on provenance status")
    return next(iter(statuses)), source_ids


def _metric_semantics(
    code: str,
    field: str,
    row: Mapping[str, Any],
    sources: Mapping[str, SourceProvenance],
    source_ids: tuple[str, ...],
) -> tuple[str, str, str, str | None, str]:
    condition = f"configured observations or agents eligible for {field} in {code}"
    denominator = f"the corresponding {code} population or regulated product set"
    period = "simulation initialisation"
    currency: str | None = None
    notes = ""

    exact: dict[str, tuple[str, str]] = {
        "population_weight": (
            "before jurisdiction assignment",
            "the sum of all configured jurisdiction assignment weights",
        ),
        "age_band_edges": (
            f"players assigned to {code}",
            f"all simulated players assigned to {code}",
        ),
        "age_band_weights": (
            f"players assigned to {code} and eligible for an age band",
            f"all simulated players assigned to {code}",
        ),
        "gaming_reach_minors": (
            "survey respondents aged 8-17",
            "all surveyed respondents aged 8-17",
        ),
        "minor_payer_probability_given_recent_gaming": (
            "respondents aged 8-17 who recently played games",
            "surveyed recent game players aged 8-17",
        ),
        "recent_purchaser_regret_probability": (
            "respondents reporting a recent in-game purchase",
            "surveyed recent in-game purchasers",
        ),
        "recent_purchaser_overspend_probability": (
            "respondents reporting a recent in-game purchase",
            "surveyed recent in-game purchasers",
        ),
        "parental_monitoring_probability": (
            "parents or carers covered by the cited child-spending study",
            "surveyed parents or carers in that study",
        ),
        "gaming_reach_ages_10_69": (
            "survey respondents aged 10-69",
            "all surveyed residents aged 10-69",
        ),
        "mobile_share_given_gamer": (
            "survey respondents classified as game users",
            "all surveyed game users",
        ),
        "mean_mobile_spend_monthly_minor_units": (
            "mobile-game users; inclusion of zero spenders remains unverified",
            "mobile-game users in the cited survey",
        ),
        "mean_iap_spend_monthly_minor_units": (
            "mobile-game users; inclusion of zero spenders remains unverified",
            "mobile-game users in the cited survey",
        ),
        "smartphone_game_reach": (
            "respondents covered by the cited smartphone-game survey",
            "all respondents in the cited survey population",
        ),
        "nonpayer_probability_given_smartphone_game": (
            "respondents who play smartphone games",
            "surveyed smartphone-game players",
        ),
        "payers_below_1500_monthly_probability": (
            "smartphone-game players reporting payment",
            "surveyed smartphone-game payers",
        ),
        "extreme_ever_title_spend_threshold_minor_units": (
            "lifetime spending on one title",
            "one title-level lifetime-spend observation",
        ),
        "extreme_ever_title_spend_probability_given_payer": (
            "smartphone-game players reporting payment",
            "surveyed smartphone-game payers",
        ),
    }
    if field in exact:
        condition, denominator = exact[field]
    elif field.startswith("deprivation_probability_age_"):
        band = field.removeprefix("deprivation_probability_age_").replace("_", "-")
        condition = f"surveyed Belgian residents in age band {band}"
        denominator = f"all surveyed Belgian residents in age band {band}"
    elif field in _RULE_FIELDS:
        condition = f"games or transactions within the legal scope in {code}"
        denominator = "covered games/transactions; this is rule applicability, not prevalence"
        period = "current configured regulatory regime"
    elif field.startswith("subsidy_"):
        condition = f"projects meeting the cited public-funding eligibility rules in {code}"
        denominator = "one eligible project or its qualified expenditure"
        period = "configured funding round"
    elif field.startswith(("median_equivalised_", "disposable_income_")):
        condition = f"households represented by the {code} income statistic"
        denominator = (
            "one surveyed household-income observation"
            if code == "KR"
            else "one equivalised household-income observation"
        )
        period = _required_string(row, "income_period", code)
    elif field in {
        "income_log_sigma",
        "income_within_quintile_log_sigma",
        "consumption_propensity_by_quintile",
    }:
        condition = f"positive simulated disposable incomes in {code}"
        denominator = f"simulated players assigned to {code}"
        period = "monthly simulation distribution"

    if source_ids:
        period = sources[source_ids[0]].period if period == "simulation initialisation" else period
    if "monthly" in field:
        period = "monthly"
    if field.startswith("extreme_ever_"):
        period = "ever per title; not a monthly hazard"
    if field.endswith("_minor_units"):
        currency = _required_string(row, "income_currency", code)
        notes = "Nominal local currency only; not cross-country comparable."
    return condition, denominator, period, currency, notes


def _shared_contracts(shared: Mapping[str, Any]) -> tuple[MetricContract, ...]:
    semantics = {
        "audit_capacity_per_cycle": (
            "audits scheduled within one jurisdiction cycle",
            "one StateAgent per jurisdiction per cycle",
            "per simulation cycle",
        ),
        "base_unauthorised_card_hazard_per_exposed_minor_day": (
            "minor has stored-card access and an opportunity to attempt unauthorised spend",
            "one exposed minor-day",
            "daily hazard",
        ),
        "essential_spend_share_mean": (
            "simulated household resources before discretionary game spending",
            "one simulated household budget",
            "monthly",
        ),
    }
    status_fields = {
        "audit_capacity_per_cycle": "audit_capacity_status",
        "base_unauthorised_card_hazard_per_exposed_minor_day": "rare_event_status",
        "essential_spend_share_mean": "essential_spend_share_status",
    }
    contracts: list[MetricContract] = []
    for metric, (condition, denominator, period) in semantics.items():
        if metric not in shared:
            raise ProfileValidationError(f"shared_assumptions.{metric} is required")
        notes = ""
        if metric == "base_unauthorised_card_hazard_per_exposed_minor_day":
            note = shared.get("rare_event_note", "")
            if not isinstance(note, str) or not note.strip():
                raise ProfileValidationError(
                    "rare event contract requires non-empty rare_event_note"
                )
            notes = note
        contracts.append(
            MetricContract(
                jurisdiction_code="*",
                metric=metric,
                value=_freeze_value(shared[metric]),
                status=_parse_status(
                    shared.get(status_fields[metric]),
                    f"shared_assumptions.{status_fields[metric]}",
                ),
                source_ids=(),
                condition=condition,
                denominator=denominator,
                period=period,
                notes=notes,
            )
        )
    return tuple(contracts)


def _validate_status_fields(value: object, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == "profile_status" or key.endswith("_status"):
                _parse_status(child, child_path)
            _validate_status_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_status_fields(child, f"{path}[{index}]")


def _validate_source_references(
    value: object,
    sources: Mapping[str, SourceProvenance],
    path: Path,
) -> None:
    references: list[tuple[str, str]] = []

    def visit(node: object, location: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                child_location = f"{location}.{key}"
                if key.endswith("_source") or key.endswith("_source_id"):
                    if not isinstance(child, str) or not child.strip():
                        raise ProfileValidationError(
                            f"{child_location} must contain a source id"
                        )
                    references.append((child_location, child))
                elif key.endswith("_source_ids"):
                    if not isinstance(child, list) or any(
                        not isinstance(item, str) or not item.strip() for item in child
                    ):
                        raise ProfileValidationError(
                            f"{child_location} must contain source ids"
                        )
                    references.extend((child_location, item) for item in child)
                visit(child, child_location)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")

    visit(value, str(path))
    missing = [(location, source_id) for location, source_id in references if source_id not in sources]
    if missing:
        location, source_id = missing[0]
        raise ProfileValidationError(f"{location} references unknown source id {source_id}")


def _source_ids_for(row: Mapping[str, Any], prefix: str) -> tuple[str, ...]:
    value = row.get(f"{prefix}_source")
    if value is None:
        return ()
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"{prefix}_source must be a non-empty source id")
    return (value,)


def _parse_status(value: object, context: str) -> ProvenanceStatus:
    if not isinstance(value, str):
        raise ProfileValidationError(f"{context} must be a provenance status string")
    try:
        return ProvenanceStatus(value)
    except ValueError as exc:
        allowed = ", ".join(status.value for status in ProvenanceStatus)
        raise ProfileValidationError(
            f"{context} has unknown status {value!r}; expected one of {allowed}"
        ) from exc


def _required_string(row: Mapping[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"{context}.{key} must be non-empty text")
    return value


def _required_bool(row: Mapping[str, Any], key: str, context: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ProfileValidationError(f"{context}.{key} must be boolean")
    return value


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProfileValidationError(f"{context} must be a positive integer")
    return value


def _positive_float(value: object, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0
    ):
        raise ProfileValidationError(f"{context} must be positive")
    return float(value)


def _positive_int_tuple(value: object, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ProfileValidationError(f"{context} must be a non-empty integer array")
    return tuple(_positive_int(item, context) for item in value)


def _nonnegative_float_tuple(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ProfileValidationError(f"{context} must be a non-empty number array")
    result: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not isfinite(item)
            or item < 0
        ):
            raise ProfileValidationError(f"{context} must contain non-negative numbers")
        result.append(float(item))
    total = sum(result)
    if not isfinite(total) or total <= 0:
        raise ProfileValidationError(f"{context} must have positive total weight")
    return tuple(result)


def _round_positive_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ProfileValidationError("money ratio requires positive inputs")
    return (numerator + denominator // 2) // denominator


def _round_signed_ratio(numerator: int, denominator: int) -> int:
    """Round an exact signed ratio to nearest, with half ties away from zero."""

    if denominator <= 0:
        raise ProfileValidationError("money ratio denominator must be positive")
    if numerator < 0:
        return -((-numerator + denominator // 2) // denominator)
    return (numerator + denominator // 2) // denominator


def _freeze_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _freeze_value(child)) for key, child in value.items())
    return value


def _optional_caveat(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key, "")
    if value == "":
        return ""
    if not isinstance(value, str):
        raise ProfileValidationError(f"{key} caveat must be text")
    return value.strip()


def _parse_iso_date(value: object, context: str) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(
            f"{context} must be an ISO date in YYYY-MM-DD form"
        )
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ProfileValidationError(
            f"{context} must be an ISO date in YYYY-MM-DD form"
        ) from exc
    if parsed.isoformat() != value:
        raise ProfileValidationError(
            f"{context} must be an ISO date in YYYY-MM-DD form"
        )
    return parsed


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


# Readable compatibility names for callers that use "record" or "contract" in
# their domain vocabulary.  They refer to the same immutable dataclasses.
SourceRecord = SourceProvenance
ProvenanceContract = MetricContract


__all__ = [
    "DEFAULT_JURISDICTIONS_PATH",
    "DEFAULT_SOURCE_BUNDLE_PATH",
    "DEFAULT_SOURCES_PATH",
    "MetricContract",
    "MonetaryConversionContract",
    "MonetaryConversionMethod",
    "MonetaryEvidenceAssessment",
    "MonetaryRoundingScope",
    "MoneyScaleContract",
    "ProfileBundle",
    "ProfileConfigurationError",
    "ProfileValidationError",
    "ProvenanceContract",
    "SourceRecord",
    "SourceProvenance",
    "load_country_profiles",
    "load_profile_bundle",
    "load_state_agents",
]
