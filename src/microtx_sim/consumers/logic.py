"""Vectorised player choice, activity, purchasing, and harm dynamics.

This module is deliberately a system over passive tables, not an agent policy
with access to a ``World`` object.  Game choice sees public ranking signals,
published prices/monetisation, a player's current-game experience, and an
explicit lagged household-peer signal. Counter-based random fields keep results
independent of population chunking.

All monetary columns consumed here are *simulation purchasing-power cents*.
They are comparable integer minor units, not nominal GBP, KRW, JPY, or EUR.
Currency conversion and purchasing-power normalisation are responsibilities of
the profile loader before it constructs :class:`PlayerTable` and
:class:`GameTable` instances.

The system does not diagnose addiction or any clinical condition.  It updates
seven separately reported harm indicators and uses an explicit operational
``unsafe`` revenue flag for audit/research aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral
from typing import Final

import numpy as np
import numpy.typing as npt

from ..agents.players import PlayerTable, TRAIT_NAMES
from ..core.ledger import Ledger, LedgerEntry
from ..domain.games import GameTable
from ..rng import CounterRNG
from ..types import HarmDimension, MonetisationMechanism, Motive


Int64Array = npt.NDArray[np.int64]
Float64Array = npt.NDArray[np.float64]

_STREAM_DISCOVERY: Final[str] = "player-dynamics:game-discovery"
_STREAM_HOUSEHOLD_DISCOVERY: Final[str] = (
    "player-dynamics:household-peer-discovery"
)
_STREAM_GAME_TASTE: Final[str] = "player-dynamics:game-taste"
_STREAM_OUTSIDE_TASTE: Final[str] = "player-dynamics:outside-taste"
_STREAM_ACTIVITY: Final[str] = "player-dynamics:activity"
_STREAM_ACTIVITY_QUALITY: Final[str] = "player-dynamics:activity-quality-experience"
_STREAM_ACTIVITY_TIME: Final[str] = "player-dynamics:activity-time"
_STREAM_MATCH_COUNT: Final[str] = "player-dynamics:match-count"
_STREAM_STABLE_SKILL: Final[str] = "player-dynamics:stable-skill"
_STREAM_PERFORMANCE: Final[str] = "player-dynamics:performance"
_STREAM_PURCHASE_CONSIDERATION: Final[str] = (
    "player-dynamics:purchase-consideration"
)
_STREAM_PURCHASE_CONVERSION: Final[str] = "player-dynamics:purchase-conversion"
_STREAM_PURCHASE_QUALITY: Final[str] = "player-dynamics:purchase-quality-experience"
_STREAM_PURCHASE_TAIL: Final[str] = "player-dynamics:purchase-tail"
_STREAM_UNAUTHORISED_CARD: Final[str] = "player-dynamics:unauthorised-card"


@dataclass(frozen=True, slots=True)
class PlayerDynamicsConfig:
    """Illustrative behavioural parameters for one player-system step.

    The rare-card value is a daily hazard.  The default matches the explicit
    sensitivity-analysis prior in ``configs/jurisdictions.toml``; it is not a
    prevalence estimate. ``household_peer_influence`` is likewise a synthetic
    sensitivity parameter, not an estimated network effect.
    """

    tick_days: int = 1
    chunk_size: int = 4096
    base_unauthorised_card_hazard_per_exposed_minor_day: float = 5e-7
    low_supervision_threshold: float = 0.35
    unauthorised_household_fraction: float = 0.02
    unauthorised_household_cap_cents: int = 20_000
    max_packages_per_tick: int = 64
    essential_spend_share: float = 0.68
    game_choice_temperature: float = 0.30
    switching_cost: float = 0.12
    household_peer_influence: float = 0.0
    base_purchase_logit: float = -3.55
    harm_decay: float = 0.985

    def __post_init__(self) -> None:
        _plain_int(self.tick_days, name="tick_days", minimum=1)
        _plain_int(self.chunk_size, name="chunk_size", minimum=1)
        _plain_int(
            self.unauthorised_household_cap_cents,
            name="unauthorised_household_cap_cents",
            minimum=0,
        )
        _plain_int(
            self.max_packages_per_tick,
            name="max_packages_per_tick",
            minimum=1,
        )
        for name in (
            "base_unauthorised_card_hazard_per_exposed_minor_day",
            "low_supervision_threshold",
            "unauthorised_household_fraction",
            "essential_spend_share",
            "switching_cost",
            "household_peer_influence",
            "harm_decay",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not np.isfinite(self.game_choice_temperature)
            or self.game_choice_temperature <= 0.0
        ):
            raise ValueError("game_choice_temperature must be finite and positive")
        if not np.isfinite(self.base_purchase_logit):
            raise ValueError("base_purchase_logit must be finite")


@dataclass(frozen=True, slots=True)
class PlayerStepCounters:
    players: int
    known_player_game_pairs: int
    changed_games: int
    active_players: int
    competitive_players: int
    purchase_intents: int
    payers: int
    unauthorised_card_events: int
    unauthorised_payers: int
    credit_payers: int
    total_spend_cents: int
    unsafe_revenue_cents: int


@dataclass(frozen=True, slots=True)
class StepResult:
    """Exact per-step player outputs and auditable revenue aggregates.

    ``game_revenue_cents`` and ``game_unsafe_revenue_cents`` are flows for this
    step, not the cumulative columns in :class:`GameTable`.  Unsafe revenue is
    an operational research flag: unauthorised spend, credit-funded spend,
    severe affordability burden, or high monetisation pressure interacting
    with measured vulnerability.  It is not a clinical classification.
    """

    tick: int
    counters: PlayerStepCounters
    chosen_game: npt.NDArray[np.int32]
    active: npt.NDArray[np.bool_]
    activity_minutes: npt.NDArray[np.int32]
    matches_played: npt.NDArray[np.int16]
    matchmaking_bracket: npt.NDArray[np.int16]
    competitive_rank: npt.NDArray[np.int32]
    competitive_rating: npt.NDArray[np.float32]
    player_spend_cents: Int64Array
    player_unsafe_spend_cents: Int64Array
    player_unauthorised_spend_cents: Int64Array
    unauthorised_card_event: npt.NDArray[np.bool_]
    harm_delta: npt.NDArray[np.float32]
    game_revenue_cents: Int64Array
    game_unsafe_revenue_cents: Int64Array

    def __post_init__(self) -> None:
        n_players = self.player_spend_cents.size
        player_columns = (
            self.chosen_game,
            self.active,
            self.activity_minutes,
            self.matches_played,
            self.matchmaking_bracket,
            self.competitive_rank,
            self.competitive_rating,
            self.player_unsafe_spend_cents,
            self.player_unauthorised_spend_cents,
            self.unauthorised_card_event,
        )
        if any(column.ndim != 1 or column.size != n_players for column in player_columns):
            raise ValueError("StepResult player columns have inconsistent shapes")
        if self.harm_delta.shape != (n_players, len(HarmDimension)):
            raise ValueError("harm_delta must have one row and seven dimensions per player")
        if self.game_revenue_cents.ndim != 1 or (
            self.game_unsafe_revenue_cents.shape != self.game_revenue_cents.shape
        ):
            raise ValueError("StepResult game revenue columns have inconsistent shapes")
        for values in (
            self.player_spend_cents,
            self.player_unsafe_spend_cents,
            self.player_unauthorised_spend_cents,
            self.game_revenue_cents,
            self.game_unsafe_revenue_cents,
        ):
            if values.dtype != np.dtype(np.int64) or np.any(values < 0):
                raise TypeError("money outputs must be non-negative int64 cents")
        if np.any(self.player_unsafe_spend_cents > self.player_spend_cents):
            raise ValueError("unsafe spend cannot exceed player spend")
        if np.any(self.player_unauthorised_spend_cents > self.player_spend_cents):
            raise ValueError("unauthorised spend cannot exceed player spend")
        total = _exact_sum(self.player_spend_cents)
        unsafe = _exact_sum(self.player_unsafe_spend_cents)
        if total != _exact_sum(self.game_revenue_cents):
            raise ValueError("player and game revenue flows do not reconcile")
        if unsafe != _exact_sum(self.game_unsafe_revenue_cents):
            raise ValueError("player and game unsafe revenue flows do not reconcile")
        if total != self.counters.total_spend_cents:
            raise ValueError("total-spend counter does not reconcile")
        if unsafe != self.counters.unsafe_revenue_cents:
            raise ValueError("unsafe-revenue counter does not reconcile")

    @property
    def unsafe_revenue_cents(self) -> int:
        return self.counters.unsafe_revenue_cents

    @property
    def spend_cents(self) -> Int64Array:
        """Short alias retained for system composition."""

        return self.player_spend_cents

    @property
    def unauthorized_card_event(self) -> npt.NDArray[np.bool_]:
        """US-spelling alias for external analysis code."""

        return self.unauthorised_card_event

    @property
    def player_unauthorized_spend_cents(self) -> Int64Array:
        return self.player_unauthorised_spend_cents


@dataclass(frozen=True, slots=True)
class _ActivityOutcome:
    active: npt.NDArray[np.bool_]
    minutes: npt.NDArray[np.int32]
    matches: npt.NDArray[np.int16]
    bracket: npt.NDArray[np.int16]
    rank: npt.NDArray[np.int32]
    rating: npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class _PurchasePlan:
    intent: npt.NDArray[np.bool_]
    card_event: npt.NDArray[np.bool_]
    spend: Int64Array
    unsafe_spend: Int64Array
    unauthorised_spend: Int64Array
    liquid_used: Int64Array
    credit_used: Int64Array
    household_after: Int64Array
    pressure: Float64Array
    selected_monetisation: Float64Array
    game_revenue: Int64Array
    game_unsafe_revenue: Int64Array


@dataclass(frozen=True, slots=True)
class _HouseholdPeerIndex:
    """Sparse pre-tick household/game counts for block-local peer signals."""

    household_rows: Int64Array
    household_sizes: Int64Array
    previous_game_rows: npt.NDArray[np.int32]
    pair_keys: Int64Array
    pair_counts: Int64Array

    def share(self, start: int, stop: int, game_count: int) -> Float64Array:
        """Return leave-one-out peer shares for one player block."""

        block_households = self.household_rows[start:stop]
        block_previous_games = self.previous_game_rows[start:stop]
        block_size = stop - start
        query_keys = (
            block_households[:, None] * game_count
            + np.arange(game_count, dtype=np.int64)[None, :]
        )
        locations = np.searchsorted(self.pair_keys, query_keys)
        valid = locations < self.pair_keys.size
        if self.pair_keys.size:
            clipped = np.minimum(locations, self.pair_keys.size - 1)
            valid &= self.pair_keys[clipped] == query_keys
        else:
            clipped = np.zeros_like(locations)
        peer_counts = np.zeros((block_size, game_count), dtype=np.int64)
        peer_counts[valid] = self.pair_counts[clipped[valid]]
        has_game = block_previous_games >= 0
        if np.any(has_game):
            peer_counts[
                np.flatnonzero(has_game), block_previous_games[has_game]
            ] -= 1
        peer_denominator = self.household_sizes[block_households] - 1
        result = np.zeros((block_size, game_count), dtype=np.float64)
        np.divide(
            peer_counts,
            peer_denominator[:, None],
            out=result,
            where=peer_denominator[:, None] > 0,
        )
        return result


class PlayerDynamicsSystem:
    """Small state-free facade suitable for a simulation kernel."""

    __slots__ = ("config",)

    def __init__(self, config: PlayerDynamicsConfig | None = None) -> None:
        self.config = config or PlayerDynamicsConfig()

    def step(
        self,
        players: PlayerTable,
        games: GameTable,
        rng: CounterRNG,
        ledger: Ledger,
        *,
        tick: int,
    ) -> StepResult:
        return step_player_dynamics(
            players,
            games,
            rng,
            ledger,
            tick=tick,
            config=self.config,
        )


def step_player_dynamics(
    players: PlayerTable,
    games: GameTable,
    rng: CounterRNG,
    ledger: Ledger,
    *,
    tick: int,
    config: PlayerDynamicsConfig | None = None,
    chunk_size: int | None = None,
) -> StepResult:
    """Advance player behaviour by one tick without receiving latent ``World``.

    Dense player-by-game work is restricted to ``chunk_size`` rows.  Every
    known game is nevertheless evaluated for every player; no alternative is
    sampled or pruned.  All counter draws are indexed by stable player/game IDs,
    so changing the block size cannot change the result.
    """

    tick = _plain_int(tick, name="tick", minimum=0)
    if not isinstance(players, PlayerTable):
        raise TypeError("players must be a PlayerTable")
    if not isinstance(games, GameTable):
        raise TypeError("games must be a GameTable")
    if not isinstance(rng, CounterRNG):
        raise TypeError("rng must be a CounterRNG")
    if not isinstance(ledger, Ledger):
        raise TypeError("ledger must be a Ledger")
    resolved = config or PlayerDynamicsConfig()
    if chunk_size is not None:
        resolved = replace(
            resolved,
            chunk_size=_plain_int(chunk_size, name="chunk_size", minimum=1),
        )
    _validate_dynamic_tables(players, games, resolved)

    previous_game = players.current_game.copy()
    chosen_game, known_pairs = _choose_games_exact_blocked(
        players, games, rng, tick, resolved, previous_game=previous_game
    )
    game_rows = _rows_for_game_ids(chosen_game, games.game_id)
    activity = _activity_and_competition(players, games, game_rows, rng, tick)
    purchase = _plan_purchases(
        players,
        games,
        game_rows,
        activity,
        rng,
        tick,
        resolved,
    )
    new_harm, harm_delta = _next_harm_state(
        players, games, game_rows, activity, purchase, resolved
    )
    ledger_entries = _ledger_entries(
        players, games, game_rows, purchase, tick
    )
    _preflight_mutations(players, games, purchase)
    ledger.append_many(ledger_entries)

    # Mutations are deliberately delayed until all calculations and the atomic
    # ledger append succeed.
    players.current_game[...] = chosen_game
    players.liquidity_cents[...] -= purchase.liquid_used
    players.credit_limit_cents[...] -= purchase.credit_used
    players.household_liquidity_cents[...] = purchase.household_after
    players.harm_state[...] = new_harm

    active_by_game = np.zeros(len(games.game_id), dtype=np.int64)
    if np.any(activity.active):
        np.add.at(active_by_game, game_rows[activity.active], 1)
    games.active_players[...] = active_by_game
    games.revenue_cents[...] += purchase.game_revenue

    changed = previous_game != chosen_game
    counters = PlayerStepCounters(
        players=len(players),
        known_player_game_pairs=known_pairs,
        changed_games=int(np.count_nonzero(changed)),
        active_players=int(np.count_nonzero(activity.active)),
        competitive_players=int(np.count_nonzero(activity.matches > 0)),
        purchase_intents=int(np.count_nonzero(purchase.intent)),
        payers=int(np.count_nonzero(purchase.spend > 0)),
        unauthorised_card_events=int(np.count_nonzero(purchase.card_event)),
        unauthorised_payers=int(np.count_nonzero(purchase.unauthorised_spend > 0)),
        credit_payers=int(np.count_nonzero(purchase.credit_used > 0)),
        total_spend_cents=_exact_sum(purchase.spend),
        unsafe_revenue_cents=_exact_sum(purchase.unsafe_spend),
    )
    return StepResult(
        tick=tick,
        counters=counters,
        chosen_game=chosen_game.copy(),
        active=activity.active.copy(),
        activity_minutes=activity.minutes.copy(),
        matches_played=activity.matches.copy(),
        matchmaking_bracket=activity.bracket.copy(),
        competitive_rank=activity.rank.copy(),
        competitive_rating=activity.rating.copy(),
        player_spend_cents=purchase.spend.copy(),
        player_unsafe_spend_cents=purchase.unsafe_spend.copy(),
        player_unauthorised_spend_cents=purchase.unauthorised_spend.copy(),
        unauthorised_card_event=purchase.card_event.copy(),
        harm_delta=harm_delta,
        game_revenue_cents=purchase.game_revenue.copy(),
        game_unsafe_revenue_cents=purchase.game_unsafe_revenue.copy(),
    )


def _choose_games_exact_blocked(
    players: PlayerTable,
    games: GameTable,
    rng: CounterRNG,
    tick: int,
    config: PlayerDynamicsConfig,
    *,
    previous_game: npt.NDArray[np.int32],
) -> tuple[npt.NDArray[np.int32], int]:
    n_players = len(players)
    n_games = len(games.game_id)
    chosen = np.full(n_players, -1, dtype=np.int32)
    if not n_players or not n_games:
        return chosen, 0

    game_ids = games.game_id
    score_signal = _unit_scale(games.public_score)
    if n_games == 1:
        rank_signal = np.ones(1, dtype=np.float64)
    else:
        rank_signal = 1.0 - (games.public_rank.astype(np.float64) - 1.0) / (
            n_games - 1.0
        )
    visibility = 0.38 * score_signal + 0.62 * rank_signal
    monetisation_mean = games.monetisation.mean(axis=1)
    price = games.price_cents.astype(np.float64)
    novelty = games.novelty.astype(np.float64)
    random_reward = games.monetisation[:, MonetisationMechanism.RANDOM_REWARD]
    scarcity = games.monetisation[:, MonetisationMechanism.ARTIFICIAL_SCARCITY]
    power_sale = games.monetisation[:, MonetisationMechanism.POWER_SALE]
    social_pressure = games.monetisation[:, MonetisationMechanism.SOCIAL_PRESSURE]

    peer_index: _HouseholdPeerIndex | None = None
    if config.household_peer_influence > 0.0:
        peer_index = _build_household_peer_index(
            players.household_id,
            previous_game,
            game_ids,
        )

    known_pairs = 0
    literacy_index = TRAIT_NAMES.index("financial_literacy")
    social_susceptibility_index = TRAIT_NAMES.index("social_susceptibility")
    for start in range(0, n_players, config.chunk_size):
        stop = min(start + config.chunk_size, n_players)
        ids = players.player_id[start:stop, None]
        awareness = players.awareness[start:stop].astype(np.float64)[:, None]
        peer_share_block = (
            None
            if peer_index is None
            else peer_index.share(start, stop, n_games)
        )
        social_susceptibility = (
            None
            if peer_share_block is None
            else players.traits[
                start:stop, social_susceptibility_index
            ].astype(np.float64)[:, None]
        )
        discovery_probability = np.clip(
            0.015 + awareness * (0.12 + 0.85 * visibility[None, :]),
            0.0,
            1.0,
        )
        discovery_draw = rng.uniform(
            ids,
            tick,
            _STREAM_DISCOVERY,
            game_ids[None, :],
        )
        known = discovery_draw < discovery_probability
        if peer_share_block is not None:
            assert social_susceptibility is not None
            peer_discovery_probability = np.clip(
                config.household_peer_influence
                * social_susceptibility
                * peer_share_block,
                0.0,
                1.0,
            )
            peer_discovery_draw = rng.uniform(
                ids,
                tick,
                _STREAM_HOUSEHOLD_DISCOVERY,
                game_ids[None, :],
            )
            known |= peer_discovery_draw < peer_discovery_probability
        current = previous_game[start:stop, None].astype(np.int64)
        known |= current == game_ids[None, :]
        known_pairs += int(np.count_nonzero(known))

        motives = players.motive_weights[start:stop].astype(np.float64)
        competition = motives[:, Motive.COMPETITION, None]
        collection = motives[:, Motive.COLLECTION, None]
        social = motives[:, Motive.SOCIAL, None]
        exploration = motives[:, Motive.EXPLORATION, None]
        relaxation = motives[:, Motive.RELAXATION]
        literacy = players.traits[start:stop, literacy_index].astype(np.float64)[:, None]
        income = players.monthly_disposable_income_cents[start:stop].astype(
            np.float64
        )[:, None]

        public_evidence = (
            0.60 * score_signal[None, :] + 0.40 * rank_signal[None, :]
        )
        motive_marketing_fit = (
            competition * power_sale[None, :]
            + 0.55 * collection * (random_reward + scarcity)[None, :]
            + social * social_pressure[None, :]
        )
        affordability = np.log1p(
            price[None, :] / np.maximum(income / 30.0, 1.0)
        )
        inertia = (0.50 + 2.33 * config.switching_cost) * (
            current == game_ids[None, :]
        )
        systematic = (
            -0.28
            + awareness * (0.75 + competition + 0.35 * social) * public_evidence
            + 0.68 * exploration * novelty[None, :]
            + 0.48 * motive_marketing_fit
            - 0.70 * awareness * literacy * monetisation_mean[None, :]
            - 0.20 * affordability
            + inertia
        )
        if peer_share_block is not None:
            assert social_susceptibility is not None
            systematic += (
                config.household_peer_influence
                * social_susceptibility
                * peer_share_block
            )
        taste_uniform = np.clip(
            rng.uniform(ids, tick, _STREAM_GAME_TASTE, game_ids[None, :]),
            np.finfo(np.float64).tiny,
            1.0 - np.finfo(np.float64).eps,
        )
        gumbel_taste = -np.log(-np.log(taste_uniform))
        utility = systematic + config.game_choice_temperature * gumbel_taste
        utility[~known] = -np.inf

        best_row = np.argmax(utility, axis=1)
        best_utility = utility[np.arange(stop - start), best_row]
        outside_uniform = np.clip(
            rng.uniform(
                players.player_id[start:stop],
                tick,
                _STREAM_OUTSIDE_TASTE,
                0,
            ),
            np.finfo(np.float64).tiny,
            1.0 - np.finfo(np.float64).eps,
        )
        outside_utility = (
            -0.10 + 0.25 * relaxation + 0.22 * (-np.log(-np.log(outside_uniform)))
        )
        plays = best_utility > outside_utility
        selected = np.full(stop - start, -1, dtype=np.int64)
        selected[plays] = game_ids[best_row[plays]]
        chosen[start:stop] = selected.astype(np.int32)
    return chosen, known_pairs


def _build_household_peer_index(
    household_id: npt.NDArray[np.int64],
    previous_game: npt.NDArray[np.int32],
    game_ids: Int64Array,
) -> _HouseholdPeerIndex:
    """Index sparse pre-tick household/game counts without using raw IDs as rows."""

    if household_id.ndim != 1 or previous_game.shape != household_id.shape:
        raise ValueError("household and previous-game columns must be aligned 1-D arrays")
    game_count = game_ids.size
    _, household_rows = np.unique(household_id, return_inverse=True)
    household_rows = household_rows.astype(np.int64, copy=False)
    household_sizes = np.bincount(household_rows).astype(np.int64, copy=False)
    previous_game_rows = _rows_for_game_ids(previous_game, game_ids)
    has_game = previous_game_rows >= 0
    if (
        household_sizes.size
        and game_count > np.iinfo(np.int64).max // household_sizes.size
    ):
        raise OverflowError("household/game index exceeds int64")
    pair_keys = (
        household_rows[has_game] * game_count
        + previous_game_rows[has_game].astype(np.int64)
    )
    unique_keys, pair_counts = np.unique(pair_keys, return_counts=True)
    return _HouseholdPeerIndex(
        household_rows=household_rows,
        household_sizes=household_sizes,
        previous_game_rows=previous_game_rows,
        pair_keys=unique_keys.astype(np.int64, copy=False),
        pair_counts=pair_counts.astype(np.int64, copy=False),
    )


def _activity_and_competition(
    players: PlayerTable,
    games: GameTable,
    game_rows: npt.NDArray[np.int32],
    rng: CounterRNG,
    tick: int,
) -> _ActivityOutcome:
    n_players = len(players)
    has_game = game_rows >= 0
    quality = np.zeros(n_players, dtype=np.float64)
    novelty = np.zeros(n_players, dtype=np.float64)
    integrity = np.zeros(n_players, dtype=np.float64)
    if len(games.game_id) and np.any(has_game):
        quality[has_game] = games.quality[game_rows[has_game]]
        novelty[has_game] = games.novelty[game_rows[has_game]]
        integrity[has_game] = games.competitive_integrity[game_rows[has_game]]
    quality_experience = np.clip(
        quality
        + 0.10
        * rng.normal(players.player_id, tick, _STREAM_ACTIVITY_QUALITY, 0),
        0.0,
        1.0,
    )
    competition = players.motive(Motive.COMPETITION).astype(np.float64)
    relaxation = players.motive(Motive.RELAXATION).astype(np.float64)
    impairment = players.harm_state[:, HarmDimension.FUNCTIONING_IMPAIRMENT].astype(
        np.float64
    )
    activity_probability = _sigmoid(
        -0.95
        + 1.30 * quality_experience
        + 0.70 * novelty
        + 0.70 * relaxation
        + 0.55 * competition
        - 0.70 * impairment
    )
    active = has_game & (
        rng.uniform(players.player_id, tick, _STREAM_ACTIVITY, 0)
        < activity_probability
    )
    time_noise = rng.normal(players.player_id, tick, _STREAM_ACTIVITY_TIME, 0)
    minutes_float = (
        18.0
        + 105.0 * activity_probability
        + 95.0 * competition
        + 45.0 * relaxation
        + 24.0 * time_noise
    )
    minutes = np.where(
        active,
        np.clip(np.rint(minutes_float), 10.0, 480.0),
        0.0,
    ).astype(np.int32)

    expected_matches = np.where(
        active,
        np.clip(
            minutes.astype(np.float64)
            / 42.0
            * (0.12 + 1.45 * competition)
            * (0.45 + 0.55 * integrity),
            0.0,
            24.0,
        ),
        0.0,
    )
    whole_matches = np.floor(expected_matches)
    match_draw = rng.uniform(players.player_id, tick, _STREAM_MATCH_COUNT, 0)
    matches = (
        whole_matches + (match_draw < (expected_matches - whole_matches))
    ).astype(np.int16)

    impulsivity = players.trait("impulsivity").astype(np.float64)
    reward = players.trait("reward_sensitivity").astype(np.float64)
    self_control = players.trait("self_control").astype(np.float64)
    stable_skill_noise = rng.normal(
        players.player_id, 0, _STREAM_STABLE_SKILL, 0
    )
    skill = np.clip(
        0.18
        + 0.28 * self_control
        + 0.22 * reward
        + 0.24 * competition
        + 0.08 * (1.0 - impulsivity)
        + 0.08 * stable_skill_noise,
        0.0,
        1.0,
    )
    performance_noise = rng.normal(
        players.player_id, tick, _STREAM_PERFORMANCE, 0
    )
    match_scale = np.sqrt(np.maximum(matches.astype(np.float64), 1.0))
    rating_float = (
        1_000.0
        + 680.0 * (skill - 0.5)
        + 75.0 * (1.30 - integrity) * performance_noise / match_scale
        + 18.0 * np.log1p(matches.astype(np.float64))
    )
    competitor = matches > 0
    rating_float = np.where(
        competitor, np.clip(rating_float, 100.0, 2_500.0), 0.0
    )
    rating = rating_float.astype(np.float32)
    bracket = np.where(
        competitor,
        np.clip(np.floor((rating_float - 400.0) / 140.0), 0.0, 11.0),
        -1.0,
    ).astype(np.int16)

    rank = np.zeros(n_players, dtype=np.int32)
    for game_row in range(len(games.game_id)):
        positions = np.flatnonzero(competitor & (game_rows == game_row))
        if not positions.size:
            continue
        # Rating descending, then stable player ID ascending for exact ties.
        order = np.lexsort(
            (players.player_id[positions], -rating_float[positions])
        )
        rank[positions[order]] = np.arange(1, positions.size + 1, dtype=np.int32)
    return _ActivityOutcome(active, minutes, matches, bracket, rank, rating)


def _plan_purchases(
    players: PlayerTable,
    games: GameTable,
    game_rows: npt.NDArray[np.int32],
    activity: _ActivityOutcome,
    rng: CounterRNG,
    tick: int,
    config: PlayerDynamicsConfig,
) -> _PurchasePlan:
    n_players = len(players)
    n_games = len(games.game_id)
    has_game = game_rows >= 0
    selected_monetisation = np.zeros(
        (n_players, len(MonetisationMechanism)), dtype=np.float64
    )
    if n_games and np.any(has_game):
        selected_monetisation[has_game] = games.monetisation[game_rows[has_game]]
    price = np.zeros(n_players, dtype=np.int64)
    quality = np.zeros(n_players, dtype=np.float64)
    novelty = np.zeros(n_players, dtype=np.float64)
    if n_games and np.any(has_game):
        price[has_game] = games.price_cents[game_rows[has_game]]
        quality[has_game] = games.quality[game_rows[has_game]]
        novelty[has_game] = games.novelty[game_rows[has_game]]
    quality_experience = np.clip(
        quality
        + 0.10
        * rng.normal(players.player_id, tick, _STREAM_PURCHASE_QUALITY, 0),
        0.0,
        1.0,
    )

    motives = players.motive_weights.astype(np.float64)
    impulsivity = players.trait("impulsivity").astype(np.float64)
    reward = players.trait("reward_sensitivity").astype(np.float64)
    social_trait = players.trait("social_susceptibility").astype(np.float64)
    loss_aversion = players.trait("loss_aversion").astype(np.float64)
    literacy = players.trait("financial_literacy").astype(np.float64)
    self_control = players.trait("self_control").astype(np.float64)
    mechanism_weights = np.column_stack(
        (
            0.15 + motives[:, Motive.COMPETITION] + 0.35 * reward,
            0.15 + motives[:, Motive.COLLECTION] + 0.55 * reward,
            0.15 + motives[:, Motive.COLLECTION] + 0.45 * loss_aversion,
            0.15 + motives[:, Motive.SOCIAL] + 0.55 * social_trait,
            0.15 + 0.85 * (1.0 - literacy),
            0.15 + 0.55 * impulsivity + 0.45 * (1.0 - self_control),
        )
    )
    pressure = np.divide(
        np.sum(selected_monetisation * mechanism_weights, axis=1),
        np.sum(mechanism_weights, axis=1),
        out=np.zeros(n_players, dtype=np.float64),
        where=np.sum(mechanism_weights, axis=1) > 0.0,
    )
    vulnerability = players.baseline_vulnerability.astype(np.float64)
    age_gradient = np.clip(
        (25.0 - players.age_years.astype(np.float64)) / 17.0, 0.0, 1.0
    )
    income_window = np.maximum(
        players.monthly_disposable_income_cents.astype(np.float64)
        * config.tick_days
        / 30.0,
        1.0,
    )
    price_burden = np.divide(
        price.astype(np.float64),
        income_window,
        out=np.zeros(n_players, dtype=np.float64),
        where=has_game,
    )
    engagement = np.log1p(activity.minutes.astype(np.float64) / 30.0)
    payment_ease = players.has_stored_payment_access.astype(np.float64)
    safe_price = np.maximum(price.astype(np.float64), 1.0)
    liquid_coverage = np.clip(
        np.log1p(players.liquidity_cents.astype(np.float64) / safe_price),
        0.0,
        5.0,
    )
    credit_coverage = np.clip(
        np.log1p(players.credit_limit_cents.astype(np.float64) / safe_price),
        0.0,
        5.0,
    )
    consideration_latent = (
        0.60 * config.base_purchase_logit
        + 0.48 * engagement
        + 1.75 * pressure
        + 0.58 * vulnerability
        + 0.30 * payment_ease
        + 0.28 * age_gradient * reward
        + 0.30 * motives[:, Motive.COLLECTION]
    )
    conversion_latent = (
        0.40 * config.base_purchase_logit
        + 2.75 * pressure
        + 0.55 * quality_experience
        + 0.35 * novelty
        + 0.48 * vulnerability
        + 0.22 * impulsivity
        - 0.42 * self_control
        - 0.72 * np.log1p(price_burden)
        + 0.09 * liquid_coverage
        + 0.05 * credit_coverage * (1.0 - literacy)
        - 0.03 * credit_coverage * literacy
        - 0.22 * literacy * players.awareness.astype(np.float64)
    )
    considered = rng.uniform(
        players.player_id, tick, _STREAM_PURCHASE_CONSIDERATION, 0
    ) < _sigmoid(consideration_latent)
    converted = rng.uniform(
        players.player_id, tick, _STREAM_PURCHASE_CONVERSION, 0
    ) < _sigmoid(conversion_latent)
    intent = has_game & activity.active & considered & converted

    exposed_minor = (
        players.is_minor
        & ~players.guardian_consent
        & players.has_stored_payment_access
        & (players.guardian_supervision < config.low_supervision_threshold)
    )
    daily_hazard = config.base_unauthorised_card_hazard_per_exposed_minor_day
    if daily_hazard >= 1.0:
        period_hazard = 1.0
    elif daily_hazard <= 0.0:
        period_hazard = 0.0
    else:
        period_hazard = float(
            -np.expm1(config.tick_days * np.log1p(-daily_hazard))
        )
    card_event = exposed_minor & (
        rng.uniform(players.player_id, tick, _STREAM_UNAUTHORISED_CARD, 0)
        < period_hazard
    )

    tail_uniform = np.clip(
        rng.uniform(players.player_id, tick, _STREAM_PURCHASE_TAIL, 0),
        0.0,
        1.0 - np.finfo(np.float64).eps,
    )
    tail_scale = np.exp(
        np.clip(
            -0.20
            + 1.25 * pressure
            + 0.72 * vulnerability
            + 0.35 * impulsivity
            - 0.30 * self_control,
            -2.0,
            3.0,
        )
    )
    desired_packages = 1 + np.floor(
        -np.log1p(-tail_uniform) * tail_scale
    ).astype(np.int64)
    desired_packages = np.minimum(desired_packages, config.max_packages_per_tick)

    liquid_before = players.liquidity_cents
    credit_before = players.credit_limit_cents
    available_regular = liquid_before.copy()
    adult = ~players.is_minor
    if np.any(adult):
        available_regular[adult] = _saturating_add_nonnegative(
            liquid_before[adult], credit_before[adult]
        )
    regular_permitted = adult | players.guardian_consent
    regular_candidate = intent & regular_permitted
    affordable_packages = np.zeros(n_players, dtype=np.int64)
    affordable_packages[has_game] = (
        available_regular[has_game] // price[has_game]
    )
    regular_packages = np.where(
        regular_candidate,
        np.minimum(desired_packages, affordable_packages),
        0,
    ).astype(np.int64)
    regular_spend = regular_packages * price

    unauthorised_candidate = intent & card_event
    unauthorised_spend, household_after = _allocate_household_card_spend(
        players,
        unauthorised_candidate,
        desired_packages,
        price,
        config,
    )
    spend = regular_spend + unauthorised_spend
    liquid_used = np.minimum(regular_spend, liquid_before).astype(np.int64)
    credit_used = (regular_spend - liquid_used).astype(np.int64)

    discretionary_window = income_window * (1.0 - config.essential_spend_share)
    severe_burden = spend.astype(np.float64) > np.maximum(
        2.0 * discretionary_window,
        price.astype(np.float64),
    )
    pressure_interaction = (
        (pressure >= 0.62)
        & (vulnerability >= 0.55)
        & (impulsivity + reward > self_control + 0.55)
    )
    unsafe = (
        (unauthorised_spend > 0)
        | (credit_used > 0)
        | severe_burden
        | pressure_interaction
    ) & (spend > 0)
    unsafe_spend = np.where(unsafe, spend, 0).astype(np.int64)

    game_revenue = _aggregate_money_by_game(spend, game_rows, n_games)
    game_unsafe = _aggregate_money_by_game(unsafe_spend, game_rows, n_games)
    return _PurchasePlan(
        intent=intent,
        card_event=card_event,
        spend=spend,
        unsafe_spend=unsafe_spend,
        unauthorised_spend=unauthorised_spend,
        liquid_used=liquid_used,
        credit_used=credit_used,
        household_after=household_after,
        pressure=pressure,
        selected_monetisation=selected_monetisation,
        game_revenue=game_revenue,
        game_unsafe_revenue=game_unsafe,
    )


def _allocate_household_card_spend(
    players: PlayerTable,
    candidates: npt.NDArray[np.bool_],
    desired_packages: Int64Array,
    prices: Int64Array,
    config: PlayerDynamicsConfig,
) -> tuple[Int64Array, Int64Array]:
    n_players = len(players)
    spend = np.zeros(n_players, dtype=np.int64)
    household_after = players.household_liquidity_cents.copy()
    if not np.any(candidates):
        return spend, household_after

    households, inverse = np.unique(players.household_id, return_inverse=True)
    del households
    household_count = int(inverse.max()) + 1 if inverse.size else 0
    minimum = np.full(household_count, np.iinfo(np.int64).max, dtype=np.int64)
    maximum = np.zeros(household_count, dtype=np.int64)
    np.minimum.at(minimum, inverse, players.household_liquidity_cents)
    np.maximum.at(maximum, inverse, players.household_liquidity_cents)
    if not np.array_equal(minimum, maximum):
        raise ValueError("members of one household must share one liquidity balance")

    proportional_cap = np.floor(
        minimum.astype(np.float64) * config.unauthorised_household_fraction
    ).astype(np.int64)
    budget = np.minimum(
        minimum,
        np.minimum(proportional_cap, config.unauthorised_household_cap_cents),
    )
    remaining_access = budget.copy()
    positions = np.flatnonzero(candidates)
    order = np.lexsort((players.player_id[positions], inverse[positions]))
    for position in positions[order]:
        household_row = inverse[position]
        package_price = int(prices[position])
        if package_price <= 0:
            continue
        possible_packages = min(
            int(desired_packages[position]),
            int(remaining_access[household_row]) // package_price,
        )
        amount = possible_packages * package_price
        if amount:
            spend[position] = amount
            remaining_access[household_row] -= amount

    household_spend = np.zeros(household_count, dtype=np.int64)
    np.add.at(household_spend, inverse, spend)
    remaining_balance = minimum - household_spend
    household_after = remaining_balance[inverse]
    return spend, household_after


def _next_harm_state(
    players: PlayerTable,
    games: GameTable,
    game_rows: npt.NDArray[np.int32],
    activity: _ActivityOutcome,
    purchase: _PurchasePlan,
    config: PlayerDynamicsConfig,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    del games, game_rows  # Effects arrive through selected mechanism columns.
    old = players.harm_state.astype(np.float64)
    spend = purchase.spend.astype(np.float64)
    credit = purchase.credit_used.astype(np.float64)
    unauthorised = purchase.unauthorised_spend.astype(np.float64)
    vulnerability = players.baseline_vulnerability.astype(np.float64)
    income_window = np.maximum(
        players.monthly_disposable_income_cents.astype(np.float64)
        * config.tick_days
        / 30.0,
        1.0,
    )
    spend_ratio = spend / income_window
    remaining_liquidity_ratio = np.divide(
        players.liquidity_cents.astype(np.float64) - purchase.liquid_used,
        np.maximum(players.monthly_disposable_income_cents.astype(np.float64), 1.0),
    )
    low_liquidity = np.clip(0.35 - remaining_liquidity_ratio, 0.0, 0.35) / 0.35
    susceptibility = 0.55 + 0.90 * vulnerability
    random_reward = purchase.selected_monetisation[
        :, MonetisationMechanism.RANDOM_REWARD
    ]
    obfuscation = purchase.selected_monetisation[
        :, MonetisationMechanism.PRICE_OBFUSCATION
    ]
    friction_removal = purchase.selected_monetisation[
        :, MonetisationMechanism.PAYMENT_FRICTION_REMOVAL
    ]
    impulsivity = players.trait("impulsivity").astype(np.float64)
    reward = players.trait("reward_sensitivity").astype(np.float64)
    loss_aversion = players.trait("loss_aversion").astype(np.float64)
    literacy = players.trait("financial_literacy").astype(np.float64)
    self_control = players.trait("self_control").astype(np.float64)
    payer = (spend > 0).astype(np.float64)

    incidence = np.zeros_like(old)
    incidence[:, HarmDimension.FINANCIAL_STRESS] = susceptibility * (
        0.030 * np.clip(spend_ratio, 0.0, 5.0)
        + 0.025 * payer * low_liquidity
        + 0.030 * np.clip(credit / income_window, 0.0, 4.0)
    )
    discretionary_window = income_window * (1.0 - config.essential_spend_share)
    displaced = np.maximum(spend - discretionary_window, 0.0) / income_window
    incidence[:, HarmDimension.ESSENTIAL_SPEND_DISPLACEMENT] = (
        susceptibility * 0.060 * np.clip(displaced, 0.0, 5.0)
    )
    incidence[:, HarmDimension.DEBT] = susceptibility * 0.085 * np.clip(
        credit / income_window, 0.0, 5.0
    )
    minor_resources = np.maximum(
        players.allowance_cents.astype(np.float64) * config.tick_days / 30.0,
        1.0,
    )
    incidence[:, HarmDimension.UNAUTHORISED_SPEND] = (
        0.18 * np.clip(unauthorised / minor_resources, 0.0, 4.0)
        + 0.04 * purchase.card_event.astype(np.float64)
    )
    loss_control_signal = (
        purchase.pressure
        * (0.30 + 0.45 * impulsivity + 0.35 * reward)
        * (1.20 - 0.55 * self_control)
        * (0.60 + 0.70 * vulnerability)
    )
    incidence[:, HarmDimension.LOSS_OF_CONTROL] = (
        payer
        * (0.025 + 0.045 * np.clip(spend_ratio, 0.0, 3.0))
        * loss_control_signal
    )
    excess_hours = np.maximum(
        activity.minutes.astype(np.float64) / 60.0 - 3.0 * config.tick_days,
        0.0,
    )
    incidence[:, HarmDimension.FUNCTIONING_IMPAIRMENT] = (
        0.018 * excess_hours
        + 0.018
        * payer
        * loss_control_signal
        * np.clip(activity.minutes.astype(np.float64) / 180.0, 0.0, 2.0)
    )
    regret_signal = (
        0.35 * random_reward * reward
        + 0.35 * obfuscation * (1.0 - literacy)
        + 0.20 * friction_removal * impulsivity
        + 0.10 * loss_aversion
    )
    incidence[:, HarmDimension.REGRET] = payer * (
        0.045 * regret_signal
        + 0.030 * np.clip(spend_ratio - 0.5, 0.0, 3.0)
    )

    daily_decay = (1.0 - config.harm_decay) * np.asarray(
        (2.0 / 3.0, 0.8, 4.0 / 15.0, 2.0, 0.8, 1.0, 7.0 / 3.0),
        dtype=np.float64,
    )
    retention = np.power(np.clip(1.0 - daily_decay, 0.0, 1.0), config.tick_days)
    new = np.clip(old * retention[None, :] + incidence, 0.0, 1.0)
    delta = (new - old).astype(np.float32)
    return new.astype(np.float32), delta


def _ledger_entries(
    players: PlayerTable,
    games: GameTable,
    game_rows: npt.NDArray[np.int32],
    purchase: _PurchasePlan,
    tick: int,
) -> tuple[LedgerEntry, ...]:
    entries: list[LedgerEntry] = []
    for position in np.flatnonzero(purchase.spend > 0):
        player_id = int(players.player_id[position])
        row = int(game_rows[position])
        game_id = int(games.game_id[row])
        company_id = int(games.company_id[row])
        credit_account = f"firm:{company_id}:cash"
        liquid = int(purchase.liquid_used[position])
        credit = int(purchase.credit_used[position])
        unauthorised = int(purchase.unauthorised_spend[position])
        if liquid:
            entries.append(
                LedgerEntry(
                    tick=tick,
                    debit_account=f"player:{player_id}:liquid",
                    credit_account=credit_account,
                    amount_cents=liquid,
                    kind="purchase",
                    reference=f"purchase:{tick}:{player_id}:{game_id}:liquid",
                )
            )
        if credit:
            entries.append(
                LedgerEntry(
                    tick=tick,
                    debit_account=f"player:{player_id}:credit",
                    credit_account=credit_account,
                    amount_cents=credit,
                    kind="purchase",
                    reference=f"purchase:{tick}:{player_id}:{game_id}:credit",
                )
            )
        if unauthorised:
            household_id = int(players.household_id[position])
            entries.append(
                LedgerEntry(
                    tick=tick,
                    debit_account=f"household:{household_id}:liquid",
                    credit_account=credit_account,
                    amount_cents=unauthorised,
                    kind="purchase_unauthorised",
                    reference=(
                        f"purchase:{tick}:{player_id}:{game_id}:unauthorised"
                    ),
                )
            )
    return tuple(entries)


def _preflight_mutations(
    players: PlayerTable,
    games: GameTable,
    purchase: _PurchasePlan,
) -> None:
    if np.any(purchase.liquid_used > players.liquidity_cents):
        raise AssertionError("purchase plan exceeds player liquidity")
    if np.any(purchase.credit_used > players.credit_limit_cents):
        raise AssertionError("purchase plan exceeds remaining credit")
    maximum = np.iinfo(np.int64).max
    if np.any(games.revenue_cents > maximum - purchase.game_revenue):
        raise OverflowError("cumulative game revenue would overflow int64 cents")


def _validate_dynamic_tables(
    players: PlayerTable,
    games: GameTable,
    config: PlayerDynamicsConfig,
) -> None:
    games.validate()
    game_ids = games.game_id
    if game_ids.ndim != 1 or np.unique(game_ids).size != game_ids.size:
        raise ValueError("game_id values must be one-dimensional and unique")
    if np.any(game_ids < 0) or np.any(game_ids > np.iinfo(np.int32).max):
        raise ValueError("game IDs must fit non-negative PlayerTable int32 IDs")
    if games.monetisation.shape[1:] != (len(MonetisationMechanism),):
        raise ValueError("games need one monetisation column per mechanism")
    for values, name in (
        (games.public_score, "public_score"),
        (games.quality, "quality"),
        (games.novelty, "novelty"),
        (games.competitive_integrity, "competitive_integrity"),
        (games.monetisation, "monetisation"),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite values")
    if np.any(players.liquidity_cents < 0) or np.any(players.credit_limit_cents < 0):
        raise ValueError("player financial balances cannot be negative")
    if np.any(players.household_liquidity_cents < 0):
        raise ValueError("household liquidity cannot be negative")
    current_rows = _rows_for_game_ids(players.current_game, game_ids)
    if np.any((players.current_game >= 0) & (current_rows < 0)):
        raise ValueError("current_game references a game absent from GameTable")
    if game_ids.size:
        maximum_price = int(games.price_cents.max())
        if maximum_price > np.iinfo(np.int64).max // config.max_packages_per_tick:
            raise OverflowError("price times maximum packages would overflow int64 cents")


def _rows_for_game_ids(
    values: npt.NDArray[np.integer], game_ids: Int64Array
) -> npt.NDArray[np.int32]:
    result = np.full(values.shape, -1, dtype=np.int32)
    if not game_ids.size:
        return result
    order = np.argsort(game_ids, kind="stable")
    sorted_ids = game_ids[order]
    present = values >= 0
    if not np.any(present):
        return result
    candidates = values[present].astype(np.int64)
    locations = np.searchsorted(sorted_ids, candidates)
    valid = locations < sorted_ids.size
    clipped = np.minimum(locations, sorted_ids.size - 1)
    valid &= sorted_ids[clipped] == candidates
    mapped = np.full(candidates.shape, -1, dtype=np.int32)
    mapped[valid] = order[clipped[valid]].astype(np.int32)
    result[present] = mapped
    return result


def _aggregate_money_by_game(
    values: Int64Array,
    game_rows: npt.NDArray[np.int32],
    game_count: int,
) -> Int64Array:
    result = np.zeros(game_count, dtype=np.int64)
    positive = values > 0
    if np.any(positive):
        total = _exact_sum(values[positive])
        if total > np.iinfo(np.int64).max:
            raise OverflowError("aggregate game flow would overflow int64 cents")
        np.add.at(result, game_rows[positive], values[positive])
    return result


def _unit_scale(values: npt.ArrayLike) -> Float64Array:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return array.copy()
    low = float(array.min())
    high = float(array.max())
    if high == low:
        return np.full(array.shape, 0.5, dtype=np.float64)
    return (array - low) / (high - low)


def _sigmoid(values: npt.ArrayLike) -> Float64Array:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _saturating_add_nonnegative(left: Int64Array, right: Int64Array) -> Int64Array:
    maximum = np.iinfo(np.int64).max
    room = maximum - left
    return left + np.minimum(right, room)


def _exact_sum(values: npt.NDArray[np.integer]) -> int:
    return sum(int(value) for value in values)


def _plain_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return integer


# Friendly function aliases for kernels that use advance/run terminology.
advance_player_dynamics = step_player_dynamics
run_player_step = step_player_dynamics


__all__ = [
    "PlayerDynamicsConfig",
    "PlayerDynamicsSystem",
    "PlayerStepCounters",
    "StepResult",
    "advance_player_dynamics",
    "run_player_step",
    "step_player_dynamics",
]
