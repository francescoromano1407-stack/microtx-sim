from __future__ import annotations

from copy import deepcopy

import numpy as np

from ..agents.companies import FirmAgent
from ..agents.jurisdictions import StateAgent, SubsidyApplicationView
from ..agents.players import PlayerTable
from ..config import (
    ConfigurationError,
    SimulationConfig,
    StepHistoryRetention,
)
from ..data.profiles import ProfileBundle, load_profile_bundle
from ..domain.games import GameTable
from ..metrics.outcomes import OutcomeRecorder, OutcomeSnapshot
from ..rng import CounterRNG
from ..simulation.accounting import outcome_snapshot
from ..simulation.day import WorldStep, advance_day, schedule_initial_events
from ..simulation.orchestrator import advance_cycles
from ..companies.logic import (
    FirmResolution,
    FirmStrategySystem,
    create_firms,
)
from ..consumers.population import initialize_player_table
from ..consumers.logic import (
    PlayerDynamicsConfig,
    PlayerDynamicsSystem,
    StepResult,
)
from ..market.popularity import PopularitySystem, PublishedRanking
from ..states.logic import RegulationSystem
from ..types import MonetisationMechanism
from .events import EventQueue
from .ledger import Ledger


class World:
    """Latent simulation state owned exclusively by the research kernel.

    Agent policies are invoked only through the specialised systems, which
    construct local observations. No `FirmAgent` or `StateAgent` method receives
    this object.
    """

    def __init__(
        self,
        *,
        config: SimulationConfig,
        profiles: ProfileBundle,
        rng: CounterRNG,
        players: PlayerTable,
        games: GameTable,
        firms: tuple[FirmAgent, ...],
        states: tuple[StateAgent, ...],
    ) -> None:
        # ``World.create`` performs campaign-aware validation, but the public
        # constructor must still reject malformed execution contracts before
        # any mutable world state is installed.
        config.validate(campaign=False)
        self.config = config
        self.profiles = profiles
        self.rng = rng
        self.players = players
        self.games = games
        self.firms = firms
        self.states = states
        if not states:
            raise ValueError("at least one jurisdiction is required")
        if tuple(firm.firm_id for firm in firms) != tuple(range(len(firms))):
            raise ValueError("firm ids must be contiguous and position-aligned")
        if tuple(state.jurisdiction_id for state in states) != tuple(
            range(len(states))
        ):
            raise ValueError("jurisdiction ids must be contiguous and position-aligned")
        self.tick = 0
        self.ledger = Ledger()
        self.events = EventQueue()
        self.player_system = PlayerDynamicsSystem(
            PlayerDynamicsConfig(
                tick_days=config.run.tick_days,
                chunk_size=config.run.chunk_size,
                base_unauthorised_card_hazard_per_exposed_minor_day=(
                    config.behavior.unauthorised_card_hazard_per_exposed_minor_day
                ),
                essential_spend_share=config.behavior.essential_spend_share,
                game_choice_temperature=config.behavior.game_choice_temperature,
                switching_cost=config.behavior.switching_cost,
                household_peer_influence=(
                    config.behavior.household_peer_influence
                ),
                base_purchase_logit=config.behavior.base_purchase_logit,
                harm_decay=config.behavior.harm_decay,
            )
        )
        self.firm_system = FirmStrategySystem(
            firms,
            rng=rng,
            # PopularitySystem already owns delay and noise for the public
            # series; the firm system only stores what was actually published.
            public_signal_delay=0,
            public_signal_noise=config.information.public_signal_noise,
            expected_fine_cents=config.regulation.maximum_fine_cents,
            research_precision_gain=max(
                0.01, 1.0 - config.information.research_noise
            ),
        )
        self.popularity_system = PopularitySystem(
            game_count=config.market.game_count,
            delay_days=config.information.public_signal_delay,
            noise_sd=config.information.public_signal_noise,
        )
        self.regulation_system = RegulationSystem()
        self._audit_interval = config.regulation.audit_interval
        self._subsidy_interval = config.regulation.subsidy_interval
        self.recorder = OutcomeRecorder(
            record_individual=config.causal.record_individual_outcomes
        )

        player_count = len(players)
        firm_count = len(firms)
        state_count = len(states)
        self.player_total_spend_cents = np.zeros(player_count, dtype=np.int64)
        self.player_total_unsafe_spend_cents = np.zeros(player_count, dtype=np.int64)
        self.player_total_unauthorised_cents = np.zeros(player_count, dtype=np.int64)
        self.player_interest_cents = np.zeros(player_count, dtype=np.int64)
        self._initial_credit_limit_cents = players.credit_limit_cents.copy()
        self._initial_firm_cash_cents = np.asarray(
            [firm.state.cash_cents for firm in firms], dtype=np.int64
        )
        self.firm_revenue_cents = np.zeros(firm_count, dtype=np.int64)
        self.firm_unsafe_revenue_cents = np.zeros(firm_count, dtype=np.int64)
        self.firm_subsidy_cents = np.zeros(firm_count, dtype=np.int64)
        self.firm_fine_assessed_cents = np.zeros(firm_count, dtype=np.int64)
        self.firm_fine_paid_cents = np.zeros(firm_count, dtype=np.int64)
        self.state_subsidy_outlay_cents = np.zeros(state_count, dtype=np.int64)
        self._public_detections = np.zeros(firm_count, dtype=np.int64)
        self._promotion_pressure = np.zeros(config.market.game_count, dtype=np.float64)
        self._period_game_revenue_cents = np.zeros(
            config.market.game_count, dtype=np.int64
        )
        self._latest_game_active_players = np.zeros(
            config.market.game_count, dtype=np.int64
        )
        self._mechanism_caps = np.ones_like(
            games.monetisation, dtype=np.float64
        )
        self._firm_home_jurisdiction = np.asarray(
            [firm.firm_id % state_count for firm in firms], dtype=np.int64
        )
        self._pending_subsidies: list[SubsidyApplicationView] = []
        self._last_player_result: StepResult | None = None
        self._last_firm_resolution: FirmResolution | None = None
        self._last_published_ranking: PublishedRanking | None = None
        self._step_history: list[WorldStep] = []
        self._audit_count = 0

        schedule_initial_events(self)

    @classmethod
    def create(
        cls,
        config: SimulationConfig,
        *,
        profiles: ProfileBundle | None = None,
        campaign: bool = False,
    ) -> "World":
        config.validate(campaign=campaign)
        profile_bundle = profiles or load_profile_bundle(campaign=campaign)
        profile_bundle.validate_for_run(
            allow_synthetic=config.run.allow_synthetic
        )
        shared_contracts = {
            contract.metric: contract
            for contract in profile_bundle.contracts
            if contract.jurisdiction_code == "*"
        }
        linked_behavior = {
            "base_unauthorised_card_hazard_per_exposed_minor_day": (
                "unauthorised_card_hazard_per_exposed_minor_day",
                config.behavior.unauthorised_card_hazard_per_exposed_minor_day,
            ),
            "essential_spend_share_mean": (
                "essential_spend_share",
                config.behavior.essential_spend_share,
            ),
        }
        for metric, (config_field, configured) in linked_behavior.items():
            contract = shared_contracts.get(metric)
            if contract is None or float(contract.value) != configured:
                raise ConfigurationError(
                    f"behavior.{config_field} diverges from its profile evidence "
                    f"contract {metric}; "
                    "update both inputs explicitly"
                )
        if campaign and profiles is not None:
            profile_bundle.validate_for_campaign()
        rng = CounterRNG(config.run.seed)
        games = GameTable.create(
            game_count=config.market.game_count,
            company_count=config.market.company_count,
            stat_dimensions=config.market.stat_dimensions,
        )
        # Bootstrap the public board without revealing true popularity.
        initial_order = np.lexsort((games.game_id, -games.quality))
        games.public_rank[initial_order] = np.arange(
            1, len(games.game_id) + 1, dtype=np.int64
        )
        games.public_score[:] = games.quality
        players = initialize_player_table(
            config.run.player_count,
            profile_bundle.country_profiles,
            rng,
        )
        firms = create_firms(
            company_count=config.market.company_count,
            games=games,
            rng=rng,
            base_research_cost_cents=(
                config.information.research_report_cost_cents
            ),
        )
        # Profile bundles are immutable except for their agent state; paired
        # worlds must never share those mutable budgets or beliefs.
        states = tuple(deepcopy(state) for state in profile_bundle.state_agents)
        for state in states:
            state.audit_sensitivity = config.regulation.audit_sensitivity
            state.audit_specificity = config.regulation.audit_specificity
            state.random_audit_fraction = config.regulation.random_audit_fraction
        return cls(
            config=config,
            profiles=profile_bundle,
            rng=rng,
            players=players,
            games=games,
            firms=firms,
            states=states,
        )

    @property
    def step_history(self) -> tuple[WorldStep, ...]:
        """Return retained completed steps under the configured policy."""

        return tuple(self._step_history)

    @property
    def audit_count(self) -> int:
        """Return the number of audit resolutions across all completed steps."""

        return self._audit_count

    def _record_completed_step(self, step: WorldStep) -> None:
        """Retain a completed step without changing simulation semantics."""

        retention = self.config.run.step_history_retention
        if retention is StepHistoryRetention.FULL:
            self._step_history.append(step)
        elif retention is StepHistoryRetention.FINAL_ONLY:
            if self._step_history:
                self._step_history[0] = step
            else:
                self._step_history.append(step)
        else:  # Configuration validation makes this unreachable.
            raise AssertionError(f"unsupported step history retention: {retention!r}")
        self._audit_count += len(step.audit_resolutions)

    def cap_mechanism(
        self,
        *,
        mechanism: MonetisationMechanism,
        maximum: float,
        game_ids: tuple[int, ...] | None,
    ) -> None:
        if not 0.0 <= maximum <= 1.0:
            raise ValueError("mechanism cap must be in [0, 1]")
        target = set(int(game) for game in game_ids) if game_ids is not None else None
        if game_ids is not None and len(target) != len(game_ids):
            raise ValueError("mechanism cap game_ids must be unique")
        known = {int(game) for game in self.games.game_id}
        if target is not None and not target.issubset(known):
            raise ValueError("mechanism cap references an unknown game")
        for row, game_id in enumerate(self.games.game_id):
            if target is None or int(game_id) in target:
                column = int(mechanism)
                self._mechanism_caps[row, column] = min(
                    self._mechanism_caps[row, column], maximum
                )
        self._enforce_mechanism_caps()

    def _enforce_mechanism_caps(self) -> None:
        np.minimum(
            self.games.monetisation,
            self._mechanism_caps,
            out=self.games.monetisation,
        )

    def configure_audit_regime(
        self,
        *,
        interval_days: int | None = None,
        sensitivity: float | None = None,
        specificity: float | None = None,
        random_fraction: float | None = None,
    ) -> None:
        if interval_days is not None:
            if interval_days <= 0 or interval_days % self.config.run.tick_days:
                raise ValueError("audit interval must align with tick_days")
            self._audit_interval = interval_days
        for value, name in (
            (sensitivity, "sensitivity"),
            (specificity, "specificity"),
            (random_fraction, "random_fraction"),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"audit {name} must be in [0, 1]")
        for state in self.states:
            if sensitivity is not None:
                state.audit_sensitivity = sensitivity
            if specificity is not None:
                state.audit_specificity = specificity
            if random_fraction is not None:
                state.random_audit_fraction = random_fraction

    def configure_subsidy_regime(
        self,
        *,
        budget_cents_per_state: int | None = None,
        interval_days: int | None = None,
        quality_weight: float | None = None,
        design_safety_weight: float | None = None,
        accessibility_weight: float | None = None,
    ) -> None:
        if budget_cents_per_state is not None and budget_cents_per_state < 0:
            raise ValueError("subsidy budget must be non-negative")
        if interval_days is not None:
            if interval_days <= 0 or interval_days % self.config.run.tick_days:
                raise ValueError("subsidy interval must align with tick_days")
            self._subsidy_interval = interval_days
        for value, name in (
            (quality_weight, "quality_weight"),
            (design_safety_weight, "design_safety_weight"),
            (accessibility_weight, "accessibility_weight"),
        ):
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"subsidy {name} must be in [0, 1]")
        for state in self.states:
            if budget_cents_per_state is not None:
                state.state.subsidy_budget_cents = budget_cents_per_state
            if quality_weight is not None:
                state.subsidy_quality_weight = quality_weight
            if design_safety_weight is not None:
                state.subsidy_safe_revenue_weight = design_safety_weight
            if accessibility_weight is not None:
                state.subsidy_accessibility_weight = accessibility_weight
    def outcome_snapshot(self, *, tick: int | None = None) -> OutcomeSnapshot:
        return outcome_snapshot(self, tick=tick)

    def step(self) -> WorldStep:
        return advance_day(self)

    def run(self, cycles: int | None = None) -> OutcomeSnapshot:
        count = self.config.run.cycles if cycles is None else cycles
        return advance_cycles(self, count)
