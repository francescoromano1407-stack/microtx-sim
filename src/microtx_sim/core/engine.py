"""Compatibility layer for the canonical simulation orchestrator.

New code should import :class:`SimulationOrchestrator` from
``microtx_sim.simulation``. ``SimulationEngine`` remains available so existing
experiments do not break during the module migration.
"""

from ..simulation.orchestrator import RunResult, SimulationOrchestrator


class SimulationEngine(SimulationOrchestrator):
    """Backward-compatible name for :class:`SimulationOrchestrator`."""


__all__ = ["RunResult", "SimulationEngine", "SimulationOrchestrator"]
