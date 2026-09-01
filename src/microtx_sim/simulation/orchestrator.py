"""Multi-cycle simulation orchestration and execution guards."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from ..agents.players import PlayerTable, require_treatment_eligible_player_table
from ..config import ConfigurationError, SimulationConfig
from ..core.ledger import Ledger
from ..data.population_execution import validate_population_campaign_preflight
from ..data.population_projection import (
    PopulationProjectionExecution,
    require_treatment_eligible_population_projection,
)
from ..metrics.population_balance import PopulationBalanceArtifact
from ..metrics.population_estimands import ExactPopulationWeights
from ..metrics.outcomes import OutcomeSnapshot
from ..types import LedgerBackend


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
    players = getattr(world, "players", None)
    if isinstance(players, PlayerTable):
        require_treatment_eligible_player_table(
            players,
            operation="multi-cycle simulation",
        )
        projection = getattr(world, "population_projection_execution", None)
        if projection is not None:
            if type(projection) is not PopulationProjectionExecution:
                raise TypeError(
                    "population_projection_execution must be an exact "
                    "PopulationProjectionExecution when present"
                )
            observed_projection = (
                require_treatment_eligible_population_projection(
                    projection,
                    operation="multi-cycle simulation",
                )
            )
            if observed_projection.players is not players:
                raise ValueError(
                    "population projection execution must bind world.players"
                )
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
            ledger = getattr(world, "ledger", None)
            if (
                not isinstance(ledger, Ledger)
                or ledger.backend is not LedgerBackend.SQLITE
                or ledger.path is None
                or ledger.temporary_store
            ):
                raise ConfigurationError(
                    "Scientific campaigns require a non-temporary SQLite ledger"
                )
            projection = getattr(world, "population_projection_execution", None)
            if type(projection) is not PopulationProjectionExecution:
                raise ConfigurationError(
                    "Scientific campaigns require an installed projected "
                    "population execution; legacy population fallback is prohibited"
                )
            observed_projection = require_treatment_eligible_population_projection(
                projection,
                operation="scientific campaign",
            )
            validate_population_campaign_preflight(observed_projection.adapter)
            balance = getattr(world, "population_balance", None)
            weights = getattr(world, "population_weights", None)
            if (
                type(balance) is not PopulationBalanceArtifact
                or not balance.exact_balance_passed
            ):
                raise ConfigurationError(
                    "Scientific campaigns require passing pre-treatment "
                    "population balance"
                )
            if (
                type(weights) is not ExactPopulationWeights
                or weights.weight_sum != 1
                or len(weights.player_ids) != len(world.players)
            ):
                raise ConfigurationError(
                    "Scientific campaigns require exact full-cohort analysis "
                    "weights summing to one"
                )
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
