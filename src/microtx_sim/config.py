from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import tomllib

from .rng import validate_seed
from .types import ProvenanceStatus


class ConfigurationError(ValueError):
    """Raised when a scenario would violate a structural safeguard."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int
    cycles: int
    tick_days: int
    player_count: int
    chunk_size: int
    allow_synthetic: bool


@dataclass(frozen=True, slots=True)
class MarketConfig:
    company_count: int
    game_count: int
    stat_dimensions: int
    ranking_interval: int
    firm_decision_interval: int


@dataclass(frozen=True, slots=True)
class InformationConfig:
    public_signal_noise: float
    public_signal_delay: int
    research_report_cost_cents: int
    research_noise: float


@dataclass(frozen=True, slots=True)
class BehaviorConfig:
    game_choice_temperature: float
    switching_cost: float
    household_peer_influence: float
    base_purchase_logit: float
    unauthorised_card_hazard_per_exposed_minor_day: float
    essential_spend_share: float
    harm_decay: float
    daily_credit_interest_rate: float


@dataclass(frozen=True, slots=True)
class RegulationConfig:
    audit_interval: int
    subsidy_interval: int
    maximum_fine_cents: int
    audit_sensitivity: float
    audit_specificity: float
    random_audit_fraction: float


@dataclass(frozen=True, slots=True)
class CausalConfig:
    common_random_numbers: bool
    estimand: str
    record_individual_outcomes: bool


@dataclass(frozen=True, slots=True)
class MetaConfig:
    name: str
    provenance_status: ProvenanceStatus
    notes: str


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    meta: MetaConfig
    run: RunConfig
    market: MarketConfig
    information: InformationConfig
    behavior: BehaviorConfig
    regulation: RegulationConfig
    causal: CausalConfig

    def validate(self, *, campaign: bool = False) -> None:
        try:
            validate_seed(self.run.seed, name="run.seed")
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(str(exc)) from exc
        positive = {
            "cycles": self.run.cycles,
            "tick_days": self.run.tick_days,
            "player_count": self.run.player_count,
            "chunk_size": self.run.chunk_size,
            "company_count": self.market.company_count,
            "game_count": self.market.game_count,
            "stat_dimensions": self.market.stat_dimensions,
            "ranking_interval": self.market.ranking_interval,
            "firm_decision_interval": self.market.firm_decision_interval,
            "audit_interval": self.regulation.audit_interval,
            "subsidy_interval": self.regulation.subsidy_interval,
            "maximum_fine_cents": self.regulation.maximum_fine_cents,
            "research_report_cost_cents": (
                self.information.research_report_cost_cents
            ),
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ConfigurationError(f"Values must be positive: {', '.join(invalid)}")
        if self.market.company_count > self.market.game_count:
            raise ConfigurationError("Every company needs at least one game")
        if self.information.public_signal_delay < 0:
            raise ConfigurationError("public_signal_delay cannot be negative")
        scheduled_days = {
            "ranking_interval": self.market.ranking_interval,
            "firm_decision_interval": self.market.firm_decision_interval,
            "audit_interval": self.regulation.audit_interval,
            "subsidy_interval": self.regulation.subsidy_interval,
            "public_signal_delay": self.information.public_signal_delay,
            "income_renewal_interval": 30,
        }
        misaligned = [
            name
            for name, days in scheduled_days.items()
            if days % self.run.tick_days != 0
        ]
        if misaligned:
            raise ConfigurationError(
                "Scheduled day intervals must be divisible by tick_days: "
                + ", ".join(misaligned)
            )
        if not 2 <= self.market.stat_dimensions <= 12:
            raise ConfigurationError("stat_dimensions must be between 2 and 12")
        for name, value in {
            "public_signal_noise": self.information.public_signal_noise,
            "research_noise": self.information.research_noise,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be in [0, 1]")
        if not 0.0 < self.behavior.game_choice_temperature <= 1.0:
            raise ConfigurationError("game_choice_temperature must be in (0, 1]")
        if not isfinite(self.behavior.base_purchase_logit):
            raise ConfigurationError("base_purchase_logit must be finite")
        for name, value in {
            "switching_cost": self.behavior.switching_cost,
            "household_peer_influence": self.behavior.household_peer_influence,
            "unauthorised_card_hazard_per_exposed_minor_day": (
                self.behavior.unauthorised_card_hazard_per_exposed_minor_day
            ),
            "essential_spend_share": self.behavior.essential_spend_share,
            "harm_decay": self.behavior.harm_decay,
            "daily_credit_interest_rate": self.behavior.daily_credit_interest_rate,
            "audit_sensitivity": self.regulation.audit_sensitivity,
            "audit_specificity": self.regulation.audit_specificity,
            "random_audit_fraction": self.regulation.random_audit_fraction,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ConfigurationError(f"{name} must be in [0, 1]")
        if (
            self.meta.provenance_status is ProvenanceStatus.SYNTHETIC
            and not self.run.allow_synthetic
        ):
            raise ConfigurationError("Synthetic scenarios require allow_synthetic=true")
        if campaign:
            if self.meta.provenance_status is not ProvenanceStatus.CALIBRATED:
                raise ConfigurationError(
                    "Scientific campaigns require provenance_status=CALIBRATED; "
                    f"got {self.meta.provenance_status.value}"
                )
            if self.run.allow_synthetic:
                raise ConfigurationError(
                    "Scientific campaigns require allow_synthetic=false"
                )


def load_config(path: str | Path, *, campaign: bool = False) -> SimulationConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    try:
        config = SimulationConfig(
            meta=MetaConfig(
                name=str(raw["meta"]["name"]),
                provenance_status=ProvenanceStatus(raw["meta"]["provenance_status"]),
                notes=str(raw["meta"].get("notes", "")),
            ),
            run=RunConfig(**raw["run"]),
            market=MarketConfig(**raw["market"]),
            information=InformationConfig(**raw["information"]),
            behavior=BehaviorConfig(**raw["behavior"]),
            regulation=RegulationConfig(**raw["regulation"]),
            causal=CausalConfig(**raw["causal"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration {config_path}: {exc}") from exc
    config.validate(campaign=campaign)
    return config
