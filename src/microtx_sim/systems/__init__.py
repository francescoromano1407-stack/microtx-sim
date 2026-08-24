"""Backward-compatible import namespace.

The implementation now lives in the domain packages ``consumers``,
``companies``, ``states``, and ``market``.
"""

from ..consumers.population import (
    CountryProfile,
    CounterRNGLike,
    initialize_player_table,
    initialize_players,
)

__all__ = [
    "CountryProfile",
    "CounterRNGLike",
    "initialize_player_table",
    "initialize_players",
]
