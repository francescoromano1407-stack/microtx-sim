"""Agent state containers.

The simulation keeps player state in columnar arrays.  Behavioural labels such
as "whale" deliberately do not live here: they are retrospective summaries of
observed spending, not fixed player types.
"""

from .players import (
    PlayerTable,
    ProjectedPopulationAssignment,
    ProjectedPopulationCellMetadata,
    ProjectedPopulationMetadata,
    ProjectedPopulationSexBinding,
    SOURCE_RECORDED_SEX_DTYPE,
    SOURCE_RECORDED_SEX_FEMALE,
    SOURCE_RECORDED_SEX_MALE,
    SOURCE_RECORDED_SEX_UNAVAILABLE,
    TRAIT_NAMES,
    classify_spend_segments,
    projected_population_assignment_sha256,
    projected_population_plan_sha256,
    require_treatment_eligible_player_table,
    source_recorded_sex_derivation_input_sha256,
    source_recorded_sex_sha256,
)
from .companies import CompanyObservation, FirmAgent, FirmIntent, FirmPrivateState
from .jurisdictions import RegulationRules, StateAgent

__all__ = [
    "PlayerTable",
    "ProjectedPopulationAssignment",
    "ProjectedPopulationCellMetadata",
    "ProjectedPopulationMetadata",
    "ProjectedPopulationSexBinding",
    "SOURCE_RECORDED_SEX_DTYPE",
    "SOURCE_RECORDED_SEX_FEMALE",
    "SOURCE_RECORDED_SEX_MALE",
    "SOURCE_RECORDED_SEX_UNAVAILABLE",
    "TRAIT_NAMES",
    "classify_spend_segments",
    "projected_population_assignment_sha256",
    "projected_population_plan_sha256",
    "require_treatment_eligible_player_table",
    "source_recorded_sex_derivation_input_sha256",
    "source_recorded_sex_sha256",
    "CompanyObservation",
    "FirmAgent",
    "FirmIntent",
    "FirmPrivateState",
    "RegulationRules",
    "StateAgent",
]
