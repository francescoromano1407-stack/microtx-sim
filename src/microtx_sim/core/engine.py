from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from ..config import ConfigurationError, SimulationConfig
from ..metrics.outcomes import OutcomeSnapshot
from .world import World


@dataclass(frozen=True, slots=True)
class RunResult:
    cycles: int
    elapsed_seconds: float
    final_outcome: OutcomeSnapshot
    summary: dict[str, float | int]


class SimulationEngine:
    """Thin runner that keeps campaign validation separate from smoke checks."""

    __slots__ = ()

    @staticmethod
    def run(
        world: World,
        *,
        cycles: int | None = None,
        campaign: bool = False,
    ) -> RunResult:
        world.config.validate(campaign=campaign)
        if campaign:
            world.profiles.validate_for_campaign()
        count = world.config.run.cycles if cycles is None else cycles
        if count <= 0:
            raise ValueError("cycles must be positive")
        if not campaign and (count > 32 or len(world.players) > 5_000):
            raise ConfigurationError(
                "Non-campaign execution is guarded to <=32 cycles and <=5,000 players"
            )
        started = perf_counter()
        outcome = world.run(count)
        elapsed = perf_counter() - started
        return RunResult(
            cycles=count,
            elapsed_seconds=elapsed,
            final_outcome=outcome,
            summary=outcome.summary(),
        )
