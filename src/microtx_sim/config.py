from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
import os
from pathlib import Path
import re
import tomllib

from .rng import validate_seed
from .types import LedgerBackend, ProvenanceStatus


class ConfigurationError(ValueError):
    """Raised when a scenario would violate a structural safeguard."""


class StepHistoryRetention(str, Enum):
    """In-memory retention policy for completed market steps."""

    FULL = "full"
    FINAL_ONLY = "final_only"


class PopulationExecutionMode(str, Enum):
    """Explicit population initializer selected by a run configuration."""

    PROJECTED_V1 = "projected_v1"


_POPULATION_ADAPTER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


@dataclass(frozen=True, slots=True)
class PopulationProjectionConfig:
    """File locators for an opt-in, separately verified population projection.

    Merely parsing these locators does not verify either file or authorize a
    campaign.  Runtime entry points must load and re-attest both bundles against
    the selected profile evidence before initializing any players.
    """

    mode: PopulationExecutionMode
    design_bundle_path: Path
    runtime_mapping_bundle_path: Path
    adapter_id: str

    def __post_init__(self) -> None:
        if type(self.mode) is not PopulationExecutionMode:
            raise TypeError("population mode must be PopulationExecutionMode")
        if self.mode is not PopulationExecutionMode.PROJECTED_V1:
            raise ValueError("unsupported population execution mode")
        for name in ("design_bundle_path", "runtime_mapping_bundle_path"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise TypeError(f"population {name} must be a Path")
            if not str(value):
                raise ValueError(f"population {name} cannot be empty")
        if type(self.adapter_id) is not str or not _POPULATION_ADAPTER_ID.fullmatch(
            self.adapter_id
        ):
            raise ValueError(
                "population adapter_id must be a stable 1-128 character identifier"
            )

    def snapshot(self) -> dict[str, str]:
        """Return the exact, non-verifying configuration selection."""

        return {
            "mode": self.mode.value,
            "design_bundle_path": str(self.design_bundle_path),
            "runtime_mapping_bundle_path": str(
                self.runtime_mapping_bundle_path
            ),
            "adapter_id": self.adapter_id,
        }


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int
    cycles: int
    tick_days: int
    player_count: int
    chunk_size: int
    allow_synthetic: bool
    step_history_retention: StepHistoryRetention = StepHistoryRetention.FULL
    ledger_backend: LedgerBackend = LedgerBackend.MEMORY


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
    population: PopulationProjectionConfig | None = None

    def validate(self, *, campaign: bool = False) -> None:
        if self.population is not None and type(
            self.population
        ) is not PopulationProjectionConfig:
            raise ConfigurationError(
                "population must be PopulationProjectionConfig or None"
            )
        try:
            validate_seed(self.run.seed, name="run.seed")
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(str(exc)) from exc
        if type(self.run.step_history_retention) is not StepHistoryRetention:
            raise ConfigurationError(
                "step_history_retention must be 'full' or 'final_only'"
            )
        if type(self.run.ledger_backend) is not LedgerBackend:
            raise ConfigurationError("ledger_backend must be 'memory' or 'sqlite'")
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
            if self.run.ledger_backend is not LedgerBackend.SQLITE:
                raise ConfigurationError(
                    "Scientific campaigns require ledger_backend='sqlite'"
                )


def load_config(path: str | Path, *, campaign: bool = False) -> SimulationConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    try:
        run_values = dict(raw["run"])
        try:
            run_values["step_history_retention"] = StepHistoryRetention(
                run_values.get(
                    "step_history_retention",
                    StepHistoryRetention.FULL,
                )
            )
        except ValueError as exc:
            raise ValueError(
                "step_history_retention must be 'full' or 'final_only'"
            ) from exc
        try:
            run_values["ledger_backend"] = LedgerBackend(
                run_values.get("ledger_backend", LedgerBackend.MEMORY)
            )
        except ValueError as exc:
            raise ValueError(
                "ledger_backend must be 'memory' or 'sqlite'"
            ) from exc
        population = _population_projection_config(
            raw.get("population"),
            config_path=config_path,
        )
        config = SimulationConfig(
            meta=MetaConfig(
                name=str(raw["meta"]["name"]),
                provenance_status=ProvenanceStatus(raw["meta"]["provenance_status"]),
                notes=str(raw["meta"].get("notes", "")),
            ),
            run=RunConfig(**run_values),
            market=MarketConfig(**raw["market"]),
            information=InformationConfig(**raw["information"]),
            behavior=BehaviorConfig(**raw["behavior"]),
            regulation=RegulationConfig(**raw["regulation"]),
            causal=CausalConfig(**raw["causal"]),
            population=population,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration {config_path}: {exc}") from exc
    config.validate(campaign=campaign)
    return config


def _population_projection_config(
    value: object,
    *,
    config_path: Path,
) -> PopulationProjectionConfig | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("[population] must be a TOML table")
    expected = {
        "mode",
        "design_bundle_path",
        "runtime_mapping_bundle_path",
        "adapter_id",
    }
    actual = set(value)
    if actual != expected:
        raise ValueError(
            "population keys differ: "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    try:
        mode = PopulationExecutionMode(value["mode"])
    except (TypeError, ValueError) as exc:
        raise ValueError("population mode must be 'projected_v1'") from exc
    # Keep the configured leaf path visible to the secure population loaders.
    # ``Path.resolve`` would dereference a symlink/reparse-point leaf here and
    # thereby prevent those loaders from enforcing their no-alias contract.
    root = Path(os.path.abspath(os.fspath(config_path))).parent

    def resolved_path(field: str) -> Path:
        raw_path = value[field]
        if type(raw_path) is not str or not raw_path:
            raise ValueError(f"population {field} must be non-empty text")
        candidate = Path(raw_path)
        selected = candidate if candidate.is_absolute() else root / candidate
        return Path(os.path.abspath(os.fspath(selected)))

    return PopulationProjectionConfig(
        mode=mode,
        design_bundle_path=resolved_path("design_bundle_path"),
        runtime_mapping_bundle_path=resolved_path(
            "runtime_mapping_bundle_path"
        ),
        adapter_id=value["adapter_id"],
    )
