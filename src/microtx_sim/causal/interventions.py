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

