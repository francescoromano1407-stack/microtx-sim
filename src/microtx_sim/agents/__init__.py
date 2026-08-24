"""Agent state containers.

The simulation keeps player state in columnar arrays.  Behavioural labels such
as "whale" deliberately do not live here: they are retrospective summaries of
observed spending, not fixed player types.
"""

from .players import (
    PlayerTable,
    TRAIT_NAMES,
    classify_spend_segments,
)
from .companies import CompanyObservation, FirmAgent, FirmIntent, FirmPrivateState
from .jurisdictions import RegulationRules, StateAgent

__all__ = [
    "PlayerTable",
    "TRAIT_NAMES",
    "classify_spend_segments",
    "CompanyObservation",
    "FirmAgent",
    "FirmIntent",
    "FirmPrivateState",
    "RegulationRules",
    "StateAgent",
]
