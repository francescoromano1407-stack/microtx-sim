"""Multi-cycle simulation orchestration and execution guards."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from ..config import ConfigurationError, SimulationConfig
from ..metrics.outcomes import OutcomeSnapshot


class SteppableWorld(Protocol):
    """Minimal interface required by the orchestrator."""

    config: SimulationConfig
    players: object
    profiles: object

    def step(self): ...


@dataclass(frozen=True, slots=True)
class RunResult:
    """Execution metadata and the final immutable outcome."""

    cycles: int
    elapsed_seconds: float
    final_outcome: OutcomeSnapshot
    summary: dict[str, float | int]


def advance_cycles(world: SteppableWorld, cycles: int) -> OutcomeSnapshot:
    """Advance an already validated world for a positive number of cycles."""

    if cycles <= 0:
        raise ValueError("cycles must be positive")
    latest: OutcomeSnapshot | None = None
    for _ in range(cycles):
        latest = world.step().outcome
    assert latest is not None
    return latest


class SimulationOrchestrator:
    """Validate execution mode, advance cycles, and report elapsed time."""

    __slots__ = ()

    @staticmethod
    def run(
        world: SteppableWorld,
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
                "Non-campaign execution is guarded to <=32 cycles and "
                "<=5,000 players"
            )
        started = perf_counter()
        outcome = advance_cycles(world, count)
        elapsed = perf_counter() - started
        return RunResult(
            cycles=count,
            elapsed_seconds=elapsed,
            final_outcome=outcome,
            summary=outcome.summary(),
        )
