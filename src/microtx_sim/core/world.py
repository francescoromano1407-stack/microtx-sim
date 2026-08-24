from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..agents.companies import FirmAgent
from ..agents.jurisdictions import StateAgent, SubsidyApplicationView
from ..agents.players import PlayerTable
from ..config import SimulationConfig
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
                essential_spend_share=0.68,
                game_choice_temperature=config.behavior.game_choice_temperature,
                switching_cost=config.behavior.switching_cost,
                base_purchase_logit=config.behavior.base_purchase_logit,
                harm_decay=config.behavior.harm_decay,
            )
        )
        self.firm_system = FirmStrategySystem(
            firms,
            rng=rng,
            public_signal_delay=config.information.public_signal_delay,
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
        self.state_subsidy_outlay_cents = np.zeros(state_count, dtype=np.int64)
        self._public_detections = np.zeros(firm_count, dtype=np.int64)
        self._promotion_pressure = np.zeros(config.market.game_count, dtype=np.float64)
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
        for row, game_id in enumerate(self.games.game_id):
            if target is None or int(game_id) in target:
                self.games.monetisation[row, int(mechanism)] = min(
                    self.games.monetisation[row, int(mechanism)], maximum
                )

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
        )
        result = self.firm_system.step(
            tick=tick,
            games=self.games,
            period_telemetry=telemetry,
            ledger=self.ledger,
        )
        self._promotion_pressure *= 0.65
        self._promotion_pressure += result.promotion_pressure
        self._pending_subsidies.extend(result.subsidy_applications)
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
        self.firm_revenue_cents += by_firm_int
        self.firm_unsafe_revenue_cents += unsafe_int

    def _publish_ranking(self, tick: int, result: StepResult) -> PublishedRanking:
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
        # Publication happens after firm decisions in this tick and becomes a
        # signal for the following tick.
        self.firm_system.record_public_ranking(
            PublicRankingSnapshot.from_game_table(as_of=tick + 1, games=self.games)
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
        anomalous = valid & (burden > 0.10)
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
                    np.clip(self.players.harm_state.mean(), 0.0, 1.0)
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
                collected = min(firm.state.cash_cents, resolution.fine_cents)
                if collected:
                    firm.state.cash_cents -= collected
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
            tick + self.config.regulation.audit_interval,
            EventKind.AUDIT_DUE,
            priority=30,
        )
        return tuple(all_resolutions)

    def _pay_subsidies(self, tick: int) -> int:
        total = 0
        # A firm can reapply between review dates. Regulators see its latest
        # dossier once, preventing duplicate awards from repeated submissions.
        latest_by_firm = {
            application.firm_id: application
            for application in self._pending_subsidies
        }
        applications = tuple(latest_by_firm.values())
        for state in self.states:
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
        self._pending_subsidies.clear()
        self.events.schedule(
            tick + self.config.regulation.subsidy_interval,
            EventKind.SUBSIDY_REVIEW,
            priority=40,
        )
        return total

    def _accrue_interest(self) -> None:
        principal = self._initial_credit_limit_cents - self.players.credit_limit_cents
        interest = np.rint(
            principal.astype(np.float64)
            * self.config.behavior.daily_credit_interest_rate
            * self.config.run.tick_days
        ).astype(np.int64)
        self.player_interest_cents += interest

    def outcome_snapshot(self, *, tick: int | None = None) -> OutcomeSnapshot:
        firm_cash = np.asarray(
            [firm.state.cash_cents for firm in self.firms], dtype=np.int64
        )
        margin = firm_cash - self._initial_firm_cash_cents - self.firm_subsidy_cents
        safe_share = np.divide(
            self.firm_revenue_cents - self.firm_unsafe_revenue_cents,
            self.firm_revenue_cents,
            out=np.ones(len(self.firms), dtype=np.float64),
            where=self.firm_revenue_cents > 0,
        )
        debt = (
            self._initial_credit_limit_cents
            - self.players.credit_limit_cents
            + self.player_interest_cents
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
        self.player_total_spend_cents += player_result.player_spend_cents
        self.player_total_unsafe_spend_cents += player_result.player_unsafe_spend_cents
        self.player_total_unauthorised_cents += (
            player_result.player_unauthorised_spend_cents
        )
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
