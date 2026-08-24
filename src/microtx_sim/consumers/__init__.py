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
    initialize_player_table,
    initialize_players,
)

__all__ = [
    "CountryProfile",
    "CounterRNGLike",
    "PlayerDynamicsConfig",
    "PlayerDynamicsSystem",
    "PlayerStepCounters",
    "StepResult",
    "initialize_player_table",
    "initialize_players",
    "step_player_dynamics",
]
