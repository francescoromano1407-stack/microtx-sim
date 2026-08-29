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
    PopulationProjectionSampleCount,
    initialize_player_table,
    initialize_players,
    initialize_projected_player_table,
    initialize_projected_player_table_from_exact_counts,
)

__all__ = [
    "CountryProfile",
    "CounterRNGLike",
    "PopulationProjectionCell",
    "PopulationProjectionSampleCount",
    "PlayerDynamicsConfig",
    "PlayerDynamicsSystem",
    "PlayerStepCounters",
    "StepResult",
    "initialize_player_table",
    "initialize_players",
    "initialize_projected_player_table",
    "initialize_projected_player_table_from_exact_counts",
    "step_player_dynamics",
]
