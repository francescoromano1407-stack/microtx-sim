from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import tomllib

from ..agents.jurisdictions import (
    RegulationRules,
    RegulatorPrivateState,
    StateAgent,
)
from ..consumers.population import CountryProfile
from ..types import ProvenanceStatus


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JURISDICTIONS_PATH = _PROJECT_ROOT / "configs" / "jurisdictions.toml"
DEFAULT_SOURCES_PATH = _PROJECT_ROOT / "data" / "provenance" / "sources.toml"

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
    }
)


class ProfileValidationError(ValueError):
    """Raised when a profile or its evidence contract is internally unsafe."""


# A descriptive alias for callers which treat malformed profiles as configuration
# failures rather than data-validation failures.
ProfileConfigurationError = ProfileValidationError


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
    comparisons.
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
                "purchasing-power contract"
            )
        if self.currency != other.currency:
            raise ProfileValidationError("nominal amounts use different currencies")
        return Fraction(
            self.nominal_monthly_anchor_minor_units,
            other.nominal_monthly_anchor_minor_units,
        )


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

    def __post_init__(self) -> None:
        frozen_sources = MappingProxyType(dict(self.sources))
        object.__setattr__(self, "sources", frozen_sources)

        profile_codes = tuple(profile.code for profile in self.country_profiles)
        state_codes = tuple(state.code for state in self.state_agents)
        money_codes = tuple(scale.jurisdiction_code for scale in self.money_scales)
        if profile_codes != _EXPECTED_CODES:
            raise ProfileValidationError(
                f"expected country profiles {_EXPECTED_CODES}; got {profile_codes}"
            )
        if state_codes != profile_codes or money_codes != profile_codes:
            raise ProfileValidationError(
                "country profiles, state agents, and money contracts must align"
            )
        if len(frozen_sources) != len(set(frozen_sources)):
            raise ProfileValidationError("source catalogue contains duplicate ids")
        if not isinstance(self.profile_status, ProvenanceStatus):
            raise ProfileValidationError("profile_status is invalid")
        if any(not caveat.strip() for caveat in self.caveats):
            raise ProfileValidationError("bundle caveats cannot be empty")

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
        missing = sorted(referenced.difference(frozen_sources))
        if missing:
            raise ProfileValidationError(
                f"profile references unknown source ids: {', '.join(missing)}"
            )

    @property
    def provenance(self) -> tuple[MetricContract, ...]:
        """Alias that makes the evidence contracts discoverable to callers."""

        return self.contracts

    def money_scale(self, jurisdiction_code: str) -> MoneyScaleContract:
        for scale in self.money_scales:
            if scale.jurisdiction_code == jurisdiction_code:
                return scale
        raise KeyError(jurisdiction_code)

    def validate_for_campaign(self) -> None:
        """Reject any campaign containing a non-calibrated dependency."""

        failures: list[str] = []
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

        used_source_ids = {
            source_id
            for profile in self.country_profiles
            for source_id in profile.source_ids
        }
        used_source_ids.update(
            source_id for contract in self.contracts for source_id in contract.source_ids
        )
        for source_id in sorted(used_source_ids):
            source = self.sources[source_id]
            if source.status is not ProvenanceStatus.CALIBRATED:
                failures.append(f"source:{source_id}={source.status.value}")

        if failures:
            preview = ", ".join(failures[:8])
            if len(failures) > 8:
                preview += f", ... ({len(failures) - 8} more)"
            raise ProfileValidationError(
                "Scientific campaigns require every profile dependency to be "
                f"CALIBRATED; found {preview}"
            )

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
        if synthetic:
            raise ProfileValidationError(
                "Profile bundle contains SYNTHETIC dependencies while "
                "allow_synthetic=false: " + ", ".join(synthetic)
            )


def load_profile_bundle(
    jurisdictions_path: str | Path = DEFAULT_JURISDICTIONS_PATH,
    sources_path: str | Path = DEFAULT_SOURCES_PATH,
    *,
    campaign: bool = False,
) -> ProfileBundle:
    """Load, validate, and harmonise the four jurisdiction profiles."""

    jurisdiction_file = Path(jurisdictions_path)
    sources_file = Path(sources_path)
    jurisdiction_raw = _read_toml(jurisdiction_file, "jurisdiction profiles")
    sources_raw = _read_toml(sources_file, "source catalogue")
    sources = _parse_sources(sources_raw, sources_file)

    if jurisdiction_raw.get("schema_version") != 1:
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
    )
    if campaign:
        bundle.validate_for_campaign()
    return bundle


def load_country_profiles(
    jurisdictions_path: str | Path = DEFAULT_JURISDICTIONS_PATH,
    sources_path: str | Path = DEFAULT_SOURCES_PATH,
    *,
    campaign: bool = False,
) -> tuple[CountryProfile, ...]:
    """Convenience wrapper returning only the player-initialisation profiles."""

    return load_profile_bundle(
        jurisdictions_path,
        sources_path,
        campaign=campaign,
    ).country_profiles


def load_state_agents(
    jurisdictions_path: str | Path = DEFAULT_JURISDICTIONS_PATH,
    sources_path: str | Path = DEFAULT_SOURCES_PATH,
    *,
    campaign: bool = False,
) -> tuple[StateAgent, ...]:
    """Convenience wrapper returning only the jurisdiction agents."""

    return load_profile_bundle(
        jurisdictions_path,
        sources_path,
        campaign=campaign,
    ).state_agents


def _read_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ProfileValidationError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileValidationError(f"{label} {path} must contain a TOML table")
    return raw


def _parse_sources(
    raw: Mapping[str, Any], path: Path
) -> Mapping[str, SourceProvenance]:
    if raw.get("schema_version") != 1:
        raise ProfileValidationError(f"{path}: unsupported source schema_version")
    retrieved_on = raw.get("retrieved_on")
    if not isinstance(retrieved_on, str) or not retrieved_on.strip():
        raise ProfileValidationError(f"{path}: retrieved_on must be non-empty text")
    records = raw.get("source")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ProfileValidationError(f"{path}: source must be an array of tables")

    parsed: dict[str, SourceProvenance] = {}
    for index, row in enumerate(records):
        context = f"{path}: source[{index}]"
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
        )
        if source.id in parsed:
            raise ProfileValidationError(f"{path}: duplicate source id {source.id}")
        parsed[source.id] = source
    if not parsed:
        raise ProfileValidationError(f"{path}: source catalogue cannot be empty")
    return MappingProxyType(parsed)


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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
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
        if isinstance(item, bool) or not isinstance(item, (int, float)) or item < 0:
            raise ProfileValidationError(f"{context} must contain non-negative numbers")
        result.append(float(item))
    if sum(result) <= 0:
        raise ProfileValidationError(f"{context} must have positive total weight")
    return tuple(result)


def _round_positive_ratio(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise ProfileValidationError("money ratio requires positive inputs")
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


# Readable compatibility names for callers that use "record" or "contract" in
# their domain vocabulary.  They refer to the same immutable dataclasses.
SourceRecord = SourceProvenance
ProvenanceContract = MetricContract


__all__ = [
    "DEFAULT_JURISDICTIONS_PATH",
    "DEFAULT_SOURCES_PATH",
    "MetricContract",
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
