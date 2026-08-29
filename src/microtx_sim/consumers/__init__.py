"""Consumer population construction and daily behavioural dynamics."""

from .logic import (
    PlayerDynamicsConfig,
    PlayerDynamicsSystem,
    PlayerStepCounters,
    StepResult,
    step_player_dynamics,
)
from .population import (
    CountryProfile,
    CounterRNGLike,
    PopulationProjectionCell,
    initialize_player_table,
    initialize_players,
    initialize_projected_player_table,
)

__all__ = [
    "CountryProfile",
    "CounterRNGLike",
    "PopulationProjectionCell",
    "PlayerDynamicsConfig",
    "PlayerDynamicsSystem",
    "PlayerStepCounters",
    "StepResult",
    "initialize_player_table",
    "initialize_players",
    "initialize_projected_player_table",
    "step_player_dynamics",
]
