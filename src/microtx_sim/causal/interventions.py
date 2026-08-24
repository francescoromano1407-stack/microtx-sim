from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..types import MonetisationMechanism


@runtime_checkable
class IntervenableWorld(Protocol):
    def cap_mechanism(
        self,
        *,
        mechanism: MonetisationMechanism,
        maximum: float,
        game_ids: tuple[int, ...] | None,
    ) -> None: ...

    def configure_audit_regime(
        self,
        *,
        interval_days: int | None,
        sensitivity: float | None,
        specificity: float | None,
        random_fraction: float | None,
    ) -> None: ...

    def configure_subsidy_regime(
        self,
        *,
        budget_cents_per_state: int | None,
        interval_days: int | None,
        quality_weight: float | None,
        design_safety_weight: float | None,
        accessibility_weight: float | None,
    ) -> None: ...


@runtime_checkable
class Intervention(Protocol):
    name: str

    def apply(self, world: IntervenableWorld) -> None: ...


@dataclass(frozen=True, slots=True)
class NullIntervention:
    name: str = "null"

    def apply(self, world: IntervenableWorld) -> None:
        del world


@dataclass(frozen=True, slots=True)
class MechanismCap:
    mechanism: MonetisationMechanism
    maximum: float
    game_ids: tuple[int, ...] | None = None
    name: str = "mechanism_cap"

    def __post_init__(self) -> None:
        if not 0.0 <= self.maximum <= 1.0:
            raise ValueError("mechanism cap must be in [0, 1]")

    def apply(self, world: IntervenableWorld) -> None:
        world.cap_mechanism(
            mechanism=self.mechanism,
            maximum=self.maximum,
            game_ids=self.game_ids,
        )


@dataclass(frozen=True, slots=True)
class AuditRegime:
    interval_days: int | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    random_fraction: float | None = None
    name: str = "audit_regime"

    def apply(self, world: IntervenableWorld) -> None:
        world.configure_audit_regime(
            interval_days=self.interval_days,
            sensitivity=self.sensitivity,
            specificity=self.specificity,
            random_fraction=self.random_fraction,
        )


@dataclass(frozen=True, slots=True)
class SubsidyRegime:
    budget_cents_per_state: int | None = None
    interval_days: int | None = None
    quality_weight: float | None = None
    design_safety_weight: float | None = None
    accessibility_weight: float | None = None
    name: str = "subsidy_regime"

    def apply(self, world: IntervenableWorld) -> None:
        world.configure_subsidy_regime(
            budget_cents_per_state=self.budget_cents_per_state,
            interval_days=self.interval_days,
            quality_weight=self.quality_weight,
            design_safety_weight=self.design_safety_weight,
            accessibility_weight=self.accessibility_weight,
        )


@dataclass(frozen=True, slots=True)
class CompositeIntervention:
    interventions: tuple[Intervention, ...]
    name: str = "composite"

    def apply(self, world: IntervenableWorld) -> None:
        for intervention in self.interventions:
            intervention.apply(world)
