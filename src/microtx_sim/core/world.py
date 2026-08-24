from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from ..agents.companies import FirmAgent
from ..agents.jurisdictions import StateAgent, SubsidyApplicationView
from ..agents.players import PlayerTable
from ..config import ConfigurationError, SimulationConfig
from ..data.profiles import ProfileBundle, load_profile_bundle
from ..domain.games import GameTable
from ..metrics.outcomes import OutcomeRecorder, OutcomeSnapshot
from ..rng import CounterRNG, stable_stream_id
from ..systems.firm_strategy import (
    FirmResolution,
    FirmStrategySystem,
    PublicRankingSnapshot,
    capture_period_telemetry,
    create_firms,
)
from ..systems.initialization import initialize_player_table
from ..systems.player_dynamics import (
    PlayerDynamicsConfig,
    PlayerDynamicsSystem,
    StepResult,
)
from ..systems.popularity import PopularitySystem, PublishedRanking
from ..systems.regulation import (
    AuditResolution,
    FirmComplianceTruth,
    ObservableFirmMetrics,
    RegulationSystem,
)
from ..types import EventKind, HarmDimension, MonetisationMechanism
from .events import EventQueue, ScheduledEvent
from .ledger import Ledger


IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

_COMPLAINT_STREAM = stable_stream_id("player-complaint-report")
_INCOME_STREAM = stable_stream_id("monthly-income-shock")
_INT64_MAX = np.iinfo(np.int64).max


def _checked_accumulate(
    target: IntArray,
    increment: IntArray,
    *,
    label: str,
) -> None:
    values = np.asarray(increment, dtype=np.int64)
    if values.shape != target.shape or np.any(values < 0) or np.any(target < 0):
        raise ValueError(f"{label} needs aligned non-negative int64 arrays")
    if np.any(target > _INT64_MAX - values):
        raise OverflowError(f"{label} would overflow int64")
    target += values


@dataclass(frozen=True, slots=True)
class WorldStep:
    tick: int
    player_result: StepResult
    firm_resolution: FirmResolution | None
    published_ranking: PublishedRanking | None
    audit_resolutions: tuple[AuditResolution, ...]
    subsidies_paid_cents: int
    outcome: OutcomeSnapshot


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

        self._schedule_initial_events()

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
        return tuple(self._step_history)

    def _schedule_initial_events(self) -> None:
        self.events.schedule(0, EventKind.FIRM_DECISION, priority=0)
        self.events.schedule(0, EventKind.PUBLISH_RANKING, priority=20)
        self.events.schedule(0, EventKind.AUDIT_DUE, priority=30)
        self.events.schedule(0, EventKind.SUBSIDY_REVIEW, priority=40)
        self.events.schedule(30, "income_renewal", priority=-10)

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

    def _renew_income(self, tick: int) -> None:
        essential_share = self.player_system.config.essential_spend_share
        adult_inflow = np.rint(
            self.players.monthly_disposable_income_cents
            * (1.0 - essential_share)
        ).astype(np.int64)
        inflow = np.where(
            self.players.is_minor,
            self.players.allowance_cents,
            adult_inflow,
        ).astype(np.int64)
        maximum = np.iinfo(np.int64).max
        if np.any(self.players.liquidity_cents > maximum - inflow):
            raise OverflowError("player liquidity would overflow")
        self.players.liquidity_cents[:] += inflow
        for row in np.flatnonzero(inflow > 0):
            player_id = int(self.players.player_id[row])
            jurisdiction = int(self.players.jurisdiction[row])
            self.ledger.transfer(
                tick=tick,
                debit_account=f"external:income:{jurisdiction}",
                credit_account=f"player:{player_id}:liquid",
                amount_cents=int(inflow[row]),
                kind="disposable_income",
                reference=f"income:{tick}:{player_id}",
            )

    def _run_firm_decision(self, tick: int) -> FirmResolution:
        telemetry = capture_period_telemetry(
            tick=tick,
            games=self.games,
            firms=self.firms,
            rng=self.rng,
            period_revenue_cents=self._period_game_revenue_cents,
            active_players=self._latest_game_active_players,
        )
        intents = self.firm_system.collect_intents(
            tick=tick,
            games=self.games,
            period_telemetry=telemetry,
        )
        result = self.firm_system.resolve(
            tick=tick,
            games=self.games,
            intents=intents,
            ledger=self.ledger,
            period_telemetry=telemetry,
        )
        self._period_game_revenue_cents.fill(0)
        self._enforce_mechanism_caps()
        self._promotion_pressure *= 0.65
        self._promotion_pressure += result.promotion_pressure
        self._pending_subsidies.extend(
            replace(
                application,
                eligible_jurisdictions=(
                    int(self._firm_home_jurisdiction[application.firm_id]),
                ),
            )
            for application in result.subsidy_applications
        )
        self._last_firm_resolution = result
        self.events.schedule(
            tick + self.config.market.firm_decision_interval,
            EventKind.FIRM_DECISION,
            priority=0,
        )
        return result

    def _credit_firm_revenue(self, result: StepResult) -> None:
        by_firm_int = np.zeros(len(self.firms), dtype=np.int64)
        unsafe_int = np.zeros(len(self.firms), dtype=np.int64)
        # Integer scatter-add preserves every cent; it never passes money
        # through floating point and does not sample games or firms.
        np.add.at(by_firm_int, self.games.company_id, result.game_revenue_cents)
        np.add.at(
            unsafe_int,
            self.games.company_id,
            result.game_unsafe_revenue_cents,
        )
        for firm in self.firms:
            firm.state.cash_cents += int(by_firm_int[firm.firm_id])
        _checked_accumulate(
            self.firm_revenue_cents,
            by_firm_int,
            label="cumulative firm revenue",
        )
        _checked_accumulate(
            self.firm_unsafe_revenue_cents,
            unsafe_int,
            label="cumulative unsafe firm revenue",
        )

    def _publish_ranking(
        self, tick: int, result: StepResult
    ) -> PublishedRanking | None:
        self.popularity_system.observe_truth(
            tick=tick,
            players=self.players,
            games=self.games,
            period_revenue_cents=result.game_revenue_cents,
        )
        published = self.popularity_system.publish(
            tick=tick,
            games=self.games,
            rng=self.rng,
            promotion_pressure=self._promotion_pressure,
        )
        if published is not None:
            # Publication happens after firm decisions in this tick and becomes
            # available only to later decisions.
            self.firm_system.record_public_ranking(
                PublicRankingSnapshot.from_game_table(
                    as_of=published.published_tick,
                    data_tick=published.data_tick,
                    games=self.games,
                )
            )
        self._last_published_ranking = published
        self.events.schedule(
            tick + self.config.market.ranking_interval,
            EventKind.PUBLISH_RANKING,
            priority=20,
        )
        return published

    def _firm_for_player(self) -> IntArray:
        result = np.full(len(self.players), -1, dtype=np.int64)
        for row, game_id in enumerate(self.games.game_id):
            mask = self.players.current_game == int(game_id)
            result[mask] = int(self.games.company_id[row])
        return result

    def _observable_firm_metrics(
        self,
        *,
        tick: int,
        result: StepResult,
        jurisdiction_id: int,
    ) -> tuple[ObservableFirmMetrics, ...]:
        firm_for_player = self._firm_for_player()
        jurisdiction = self.players.jurisdiction == jurisdiction_id
        valid = jurisdiction & (firm_for_player >= 0)
        regret = self.players.harm_state[:, HarmDimension.REGRET].astype(np.float64)
        unauthorised = result.player_unauthorised_spend_cents > 0
        report_probability = np.clip(0.02 + 0.32 * regret + 0.70 * unauthorised, 0.0, 1.0)
        reports = valid & self.rng.bernoulli(
            self.players.player_id,
            tick,
            _COMPLAINT_STREAM,
            jurisdiction_id,
            probability=report_probability,
        )
        burden = np.divide(
            result.player_spend_cents.astype(np.float64),
            np.maximum(1, self.players.monthly_disposable_income_cents),
        )
        # Spending burden is latent until a player/household report exposes it.
        anomalous = reports & (burden > 0.10)
        minor_report = reports & self.players.is_minor
        metrics: list[ObservableFirmMetrics] = []
        for firm_id in range(len(self.firms)):
            exposed = valid & (firm_for_player == firm_id)
            denominator = max(1, int(np.count_nonzero(exposed)))
            minor_denominator = max(
                1, int(np.count_nonzero(exposed & self.players.is_minor))
            )
            metrics.append(
                ObservableFirmMetrics(
                    firm_id=firm_id,
                    complaint_rate=float(np.count_nonzero(reports & exposed) / denominator),
                    reported_minor_harm_rate=float(
                        np.count_nonzero(minor_report & exposed) / minor_denominator
                    ),
                    public_spend_anomaly=float(
                        np.count_nonzero(anomalous & exposed) / denominator
                    ),
                    past_public_detection=float(
                        min(1.0, self._public_detections[firm_id] / 3.0)
                    ),
                    signal_precision=0.65,
                    signal_age_days=0,
                )
            )
        return tuple(metrics)

    def _compliance_truth(
        self,
        *,
        state: StateAgent,
        result: StepResult,
    ) -> dict[int, FirmComplianceTruth]:
        if self._last_firm_resolution is None:
            kernel = {
                firm.firm_id: (firm.compliance_culture, 0.0) for firm in self.firms
            }
        else:
            kernel = {
                item.firm_id: (item.compliance_effectiveness, item.evasion_level)
                for item in self._last_firm_resolution.firm_kernel_state
            }
        firm_for_player = self._firm_for_player()
        truth: dict[int, FirmComplianceTruth] = {}
        for firm in self.firms:
            rows = np.flatnonzero(self.games.company_id == firm.firm_id)
            mechanisms = self.games.monetisation[rows]
            compliance, evasion = kernel[firm.firm_id]
            breaches: list[str] = []
            if (
                state.rules.paid_random_rewards_restricted
                and np.any(mechanisms[:, MonetisationMechanism.RANDOM_REWARD] > 0.02)
            ):
                breaches.append("paid_random_rewards")
            if (
                state.rules.odds_disclosure_required
                and np.any(mechanisms[:, MonetisationMechanism.RANDOM_REWARD] > 0.02)
                and compliance < 0.70
            ):
                breaches.append("odds_disclosure")
            if (
                state.rules.real_money_price_required
                and np.any(mechanisms[:, MonetisationMechanism.PRICE_OBFUSCATION] > compliance)
            ):
                breaches.append("price_transparency")
            if np.any(
                mechanisms[:, MonetisationMechanism.POWER_SALE]
                > state.rules.maximum_power_sale_intensity
            ):
                breaches.append("power_sale_limit")
            firm_minor_unauthorised = (
                (firm_for_player == firm.firm_id)
                & (self.players.jurisdiction == state.jurisdiction_id)
                & (result.player_unauthorised_spend_cents > 0)
            )
            if (
                state.rules.parental_authorisation_required
                and np.any(firm_minor_unauthorised)
                and compliance < 0.85
            ):
                breaches.append("parental_authorisation")
            if (
                state.rules.direct_exhortation_to_minors_banned
                and np.any(
                    0.5
                    * (
                        mechanisms[:, MonetisationMechanism.SOCIAL_PRESSURE]
                        + mechanisms[:, MonetisationMechanism.ARTIFICIAL_SCARCITY]
                    )
                    > 0.60
                )
                and compliance < 0.65
            ):
                breaches.append("minor_exhortation")
            truth[firm.firm_id] = FirmComplianceTruth(
                firm_id=firm.firm_id,
                actual_breaches=tuple(breaches),
                auditable_controls=(
                    "price_display",
                    "probability_disclosure",
                    "parental_authorisation",
                    "transaction_log",
                ),
                evasion_intensity=evasion,
                maximum_fine_cents=self.config.regulation.maximum_fine_cents,
            )
        return truth

    def _run_audits(self, tick: int, result: StepResult) -> tuple[AuditResolution, ...]:
        all_resolutions: list[AuditResolution] = []
        for state in self.states:
            # ``audit_budget_cents`` is the period appropriation, while the
            # treasury is the cash constraint. Re-authorising it here permits
            # genuinely periodic inspections without inventing money.
            state.state.audit_budget_cents = min(
                state.state.treasury_cents,
                state.state.audit_capacity_per_cycle
                * state.state.inspection_cost_cents,
            )
            metrics = self._observable_firm_metrics(
                tick=tick,
                result=result,
                jurisdiction_id=state.jurisdiction_id,
            )
            observation = self.regulation_system.build_observation(
                tick=tick,
                firms=metrics,
                public_harm_index=float(
                    np.clip(
                        np.mean(
                            [
                                0.45 * item.complaint_rate
                                + 0.35 * item.reported_minor_harm_rate
                                + 0.20 * item.public_spend_anomaly
                                for item in metrics
                            ]
                        ),
                        0.0,
                        1.0,
                    )
                ),
                treasury_pressure=float(
                    1.0 - state.state.treasury_cents / max(1, 36_000_000)
                ),
                sector_employment_estimate=50.0 * len(self.firms),
            )
            intents = self.regulation_system.select(
                tick=tick,
                state=state,
                observation=observation,
                rng=self.rng,
            )
            resolutions = self.regulation_system.resolve(
                tick=tick,
                state=state,
                intents=intents,
                truth_by_firm=self._compliance_truth(state=state, result=result),
                rng=self.rng,
            )
            audit_cost = len(resolutions) * state.state.inspection_cost_cents
            if audit_cost:
                self.ledger.transfer(
                    tick=tick,
                    debit_account=f"state:{state.jurisdiction_id}:treasury",
                    credit_account="sector:audit-services",
                    amount_cents=audit_cost,
                    kind="regulatory_audit",
                    reference=f"audit:{tick}:{state.jurisdiction_id}",
                )
            for resolution in resolutions:
                firm = self.firms[resolution.intent.firm_id]
                firm_id = firm.firm_id
                assessed = resolution.fine_cents
                if self.firm_fine_assessed_cents[firm_id] > _INT64_MAX - assessed:
                    raise OverflowError("cumulative assessed fines would overflow int64")
                self.firm_fine_assessed_cents[firm_id] += assessed
                collected = min(firm.state.cash_cents, resolution.fine_cents)
                if collected:
                    firm.state.cash_cents -= collected
                    if self.firm_fine_paid_cents[firm_id] > _INT64_MAX - collected:
                        raise OverflowError("cumulative paid fines would overflow int64")
                    self.firm_fine_paid_cents[firm_id] += collected
                    state.state.treasury_cents += collected
                    self.ledger.transfer(
                        tick=tick,
                        debit_account=f"firm:{firm.firm_id}:cash",
                        credit_account=f"state:{state.jurisdiction_id}:treasury",
                        amount_cents=collected,
                        kind="regulatory_fine",
                        reference=(
                            f"fine:{tick}:{state.jurisdiction_id}:{firm.firm_id}"
                        ),
                    )
                if resolution.evidence.detected_breaches:
                    self._public_detections[firm.firm_id] += 1
            all_resolutions.extend(resolutions)
        self.events.schedule(
            tick + self._audit_interval,
            EventKind.AUDIT_DUE,
            priority=30,
        )
        return tuple(all_resolutions)

    def _pay_subsidies(self, tick: int) -> int:
        total = 0
        # A firm can reapply between review dates. Regulators see its latest
        # dossier once, preventing duplicate awards from repeated submissions.
        mature = [
            replace(
                application,
                evidence_age_days=(
                    application.evidence_age_days
                    + tick
                    - application.submitted_tick
                ),
            )
            for application in self._pending_subsidies
            if application.submitted_tick < tick
        ]
        future = [
            application
            for application in self._pending_subsidies
            if application.submitted_tick >= tick
        ]
        latest_by_firm = {
            application.firm_id: application
            for application in mature
        }
        for state in self.states:
            applications = tuple(
                application
                for application in latest_by_firm.values()
                if state.jurisdiction_id in application.eligible_jurisdictions
            )
            awards = state.award_subsidies(applications)
            for award in awards:
                available = min(
                    state.state.treasury_cents,
                    state.state.subsidy_budget_cents,
                )
                paid = min(available, award.award_cents)
                if paid <= 0:
                    continue
                firm = self.firms[award.firm_id]
                state.state.treasury_cents -= paid
                state.state.subsidy_budget_cents -= paid
                firm.state.cash_cents += paid
                if self.firm_subsidy_cents[award.firm_id] > _INT64_MAX - paid:
                    raise OverflowError("cumulative firm subsidy would overflow int64")
                if (
                    self.state_subsidy_outlay_cents[state.jurisdiction_id]
                    > _INT64_MAX - paid
                ):
                    raise OverflowError("cumulative state subsidy would overflow int64")
                self.firm_subsidy_cents[award.firm_id] += paid
                self.state_subsidy_outlay_cents[state.jurisdiction_id] += paid
                total += paid
                self.ledger.transfer(
                    tick=tick,
                    debit_account=f"state:{state.jurisdiction_id}:treasury",
                    credit_account=f"firm:{firm.firm_id}:cash",
                    amount_cents=paid,
                    kind="conditional_subsidy",
                    reference=f"subsidy:{tick}:{state.jurisdiction_id}:{firm.firm_id}",
                )
        self._pending_subsidies[:] = future
        self.events.schedule(
            tick + self._subsidy_interval,
            EventKind.SUBSIDY_REVIEW,
            priority=40,
        )
        return total

    def _accrue_interest(self) -> None:
        principal = self._initial_credit_limit_cents - self.players.credit_limit_cents
        raw_interest = (
            principal.astype(np.float64)
            * self.config.behavior.daily_credit_interest_rate
            * self.config.run.tick_days
        )
        if (
            not np.all(np.isfinite(raw_interest))
            or np.any(raw_interest > 2**53)
            or np.any(raw_interest < 0.0)
        ):
            raise OverflowError("interest calculation exceeded exact-cent range")
        interest = np.rint(raw_interest).astype(np.int64)
        _checked_accumulate(
            self.player_interest_cents,
            interest,
            label="cumulative player interest",
        )

    def outcome_snapshot(self, *, tick: int | None = None) -> OutcomeSnapshot:
        cash_values = [firm.state.cash_cents for firm in self.firms]
        if any(value < 0 or value > _INT64_MAX for value in cash_values):
            raise OverflowError("firm cash is outside the reportable int64 range")
        firm_cash = np.asarray(cash_values, dtype=np.int64)
        outstanding_fines = (
            self.firm_fine_assessed_cents - self.firm_fine_paid_cents
        )
        margin_values = [
            int(firm_cash[index])
            - int(self._initial_firm_cash_cents[index])
            - int(self.firm_subsidy_cents[index])
            - int(outstanding_fines[index])
            for index in range(len(self.firms))
        ]
        int64_min = np.iinfo(np.int64).min
        if any(value < int64_min or value > _INT64_MAX for value in margin_values):
            raise OverflowError("firm margin is outside the reportable int64 range")
        margin = np.asarray(margin_values, dtype=np.int64)
        safe_share = np.divide(
            self.firm_revenue_cents - self.firm_unsafe_revenue_cents,
            self.firm_revenue_cents,
            out=np.ones(len(self.firms), dtype=np.float64),
            where=self.firm_revenue_cents > 0,
        )
        debt = (
            self._initial_credit_limit_cents
            - self.players.credit_limit_cents
        )
        _checked_accumulate(
            debt,
            self.player_interest_cents,
            label="reported player debt",
        )
        return OutcomeSnapshot(
            tick=self.tick if tick is None else tick,
            player_harm=self.players.harm_state.astype(np.float64, copy=True),
            player_spend_cents=self.player_total_spend_cents.copy(),
            player_income_cents=self.players.monthly_disposable_income_cents.copy(),
            player_debt_cents=debt.astype(np.int64, copy=False),
            firm_cash_cents=firm_cash,
            firm_operating_margin_cents=margin,
            firm_safe_revenue_share=safe_share,
            state_subsidy_outlay_cents=self.state_subsidy_outlay_cents.copy(),
        )

    def step(self) -> WorldStep:
        tick = self.tick
        due = self.events.pop_due(tick)
        firm_resolution: FirmResolution | None = None
        published: PublishedRanking | None = None
        audits: tuple[AuditResolution, ...] = ()
        subsidies_paid = 0

        for event in due:
            if event.kind == "income_renewal":
                self._renew_income(tick)
                self.events.schedule(tick + 30, "income_renewal", priority=-10)
            elif event.kind == EventKind.FIRM_DECISION:
                firm_resolution = self._run_firm_decision(tick)

        player_result = self.player_system.step(
            self.players,
            self.games,
            self.rng,
            self.ledger,
            tick=tick,
        )
        self._last_player_result = player_result
        _checked_accumulate(
            self.player_total_spend_cents,
            player_result.player_spend_cents,
            label="cumulative player spend",
        )
        _checked_accumulate(
            self.player_total_unsafe_spend_cents,
            player_result.player_unsafe_spend_cents,
            label="cumulative unsafe player spend",
        )
        _checked_accumulate(
            self.player_total_unauthorised_cents,
            player_result.player_unauthorised_spend_cents,
            label="cumulative unauthorised player spend",
        )
        _checked_accumulate(
            self._period_game_revenue_cents,
            player_result.game_revenue_cents,
            label="period game revenue",
        )
        self._latest_game_active_players[:] = self.games.active_players
        self._credit_firm_revenue(player_result)
        self._accrue_interest()

        for event in due:
            if event.kind == EventKind.PUBLISH_RANKING:
                published = self._publish_ranking(tick, player_result)
            elif event.kind == EventKind.AUDIT_DUE:
                audits = self._run_audits(tick, player_result)
            elif event.kind == EventKind.SUBSIDY_REVIEW:
                subsidies_paid = self._pay_subsidies(tick)

        # Content becomes less novel between releases; this supplies a smooth
        # endogenous incentive for future updates.
        self.games.novelty[:] *= np.exp(-0.01 * self.config.run.tick_days)
        self.ledger.assert_balanced()
        outcome = self.outcome_snapshot(tick=tick)
        self.recorder.record(outcome)
        result = WorldStep(
            tick=tick,
            player_result=player_result,
            firm_resolution=firm_resolution,
            published_ranking=published,
            audit_resolutions=audits,
            subsidies_paid_cents=subsidies_paid,
            outcome=outcome,
        )
        self._step_history.append(result)
        self.tick += self.config.run.tick_days
        return result

    def run(self, cycles: int | None = None) -> OutcomeSnapshot:
        count = self.config.run.cycles if cycles is None else cycles
        if count <= 0:
            raise ValueError("cycles must be positive")
        latest: OutcomeSnapshot | None = None
        for _ in range(count):
            latest = self.step().outcome
        assert latest is not None
        return latest
