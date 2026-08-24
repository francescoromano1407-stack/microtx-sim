"""Immutable information boundaries and signal-driven beliefs.

This module deliberately has no dependency on a simulation ``World``.  A
decision policy can receive an :class:`ObservationView`, while a
:class:`BeliefBook` can be updated only with :class:`Signal` instances.  Latent
state therefore cannot leak into an agent merely through an object reference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np

from microtx_sim.types import InformationSource


SubjectId: TypeAlias = int | str | None
BeliefKey: TypeAlias = tuple[str, SubjectId]


def _freeze(value: Any) -> Any:
    """Defensively copy values into recursively immutable containers."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _plain_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class Signal:
    """A fallible observation, including its source, delay, and precision."""

    topic: str
    value: Any
    observed_tick: int
    source: InformationSource
    precision: float
    received_tick: int | None = None
    subject_id: SubjectId = None
    cost_cents: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.topic, str) or not self.topic:
            raise ValueError("topic must be a non-empty string")
        observed_tick = _plain_int(self.observed_tick, name="observed_tick")
        if observed_tick < 0:
            raise ValueError("observed_tick must be non-negative")
        received_tick = (
            observed_tick
            if self.received_tick is None
            else _plain_int(self.received_tick, name="received_tick")
        )
        if received_tick < observed_tick:
            raise ValueError("received_tick cannot precede observed_tick")
        if not isinstance(self.source, InformationSource):
            try:
                object.__setattr__(self, "source", InformationSource(self.source))
            except (TypeError, ValueError) as exc:
                raise TypeError("source must be an InformationSource") from exc
        precision = float(self.precision)
        if not math.isfinite(precision) or not 0.0 <= precision <= 1.0:
            raise ValueError("precision must be finite and in [0, 1]")
        cost_cents = _plain_int(self.cost_cents, name="cost_cents")
        if cost_cents < 0:
            raise ValueError("cost_cents must be non-negative integer minor units")
        if self.subject_id is not None and not isinstance(self.subject_id, (int, str)):
            raise TypeError("subject_id must be int, str, or None")

        object.__setattr__(self, "observed_tick", observed_tick)
        object.__setattr__(self, "received_tick", received_tick)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "cost_cents", cost_cents)
        object.__setattr__(self, "value", _freeze(self.value))

    @property
    def name(self) -> str:
        """Alias for call sites that use ``name`` rather than ``topic``."""

        return self.topic

    def age(self, at_tick: int) -> int:
        """Return information age measured from the underlying observation."""

        tick = _plain_int(at_tick, name="at_tick")
        if tick < self.received_tick:
            raise ValueError("the signal is not available at that tick")
        return tick - self.observed_tick

    def is_available(self, at_tick: int) -> bool:
        return _plain_int(at_tick, name="at_tick") >= self.received_tick


@dataclass(frozen=True, slots=True)
class Belief:
    """Immutable estimate derived from one or more received signals."""

    topic: str
    subject_id: SubjectId
    estimate: Any
    precision: float
    updated_tick: int
    evidence_count: int
    evidence_weight: float
    sources: frozenset[InformationSource]


@dataclass(frozen=True, slots=True)
class ObservationView:
    """A defensive, immutable snapshot supplied to an agent policy."""

    observer_id: int | str
    tick: int
    signals: tuple[Signal, ...] = field(default_factory=tuple)
    beliefs: tuple[Belief, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        tick = _plain_int(self.tick, name="tick")
        if tick < 0:
            raise ValueError("tick must be non-negative")
        signals = tuple(self.signals)
        beliefs = tuple(self.beliefs)
        if any(not isinstance(signal, Signal) for signal in signals):
            raise TypeError("ObservationView accepts only Signal instances")
        if any(not signal.is_available(tick) for signal in signals):
            raise ValueError("ObservationView cannot expose a signal before receipt")
        if any(not isinstance(belief, Belief) for belief in beliefs):
            raise TypeError("beliefs must contain only Belief instances")
        object.__setattr__(self, "tick", tick)
        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "beliefs", beliefs)

    def for_topic(
        self, topic: str, *, subject_id: SubjectId = None
    ) -> tuple[Signal, ...]:
        """Return matching signals without exposing the backing collection."""

        return tuple(
            signal
            for signal in self.signals
            if signal.topic == topic
            and (subject_id is None or signal.subject_id == subject_id)
        )

    def latest(
        self, topic: str, *, subject_id: SubjectId = None
    ) -> Signal | None:
        """Return the newest matching signal, resolving ties by receipt time."""

        candidates = self.for_topic(topic, subject_id=subject_id)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda signal: (signal.observed_tick, signal.received_tick),
        )


class BeliefBook:
    """Mutable private beliefs whose sole input type is :class:`Signal`.

    Numeric estimates are precision-weighted and therefore independent of the
    order in which equally available signals are processed.  Categorical
    estimates retain the most recently observed signal.  No method accepts a
    truth store or arbitrary state mapping.
    """

    __slots__ = ("_beliefs", "_residual_uncertainty")

    def __init__(self, signals: Iterable[Signal] = ()) -> None:
        self._beliefs: dict[BeliefKey, Belief] = {}
        self._residual_uncertainty: dict[BeliefKey, float] = {}
        self.update_many(signals)

    def __len__(self) -> int:
        return len(self._beliefs)

    @staticmethod
    def _key(signal: Signal) -> BeliefKey:
        return signal.topic, signal.subject_id

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return isinstance(value, Real) and not isinstance(value, bool)

    def update(self, signal: Signal) -> Belief:
        """Update one belief, rejecting all non-signal inputs."""

        if not isinstance(signal, Signal):
            raise TypeError("BeliefBook can be updated only from Signal instances")
        key = self._key(signal)
        previous = self._beliefs.get(key)
        residual = self._residual_uncertainty.get(key, 1.0)
        combined_residual = residual * (1.0 - signal.precision)
        combined_precision = 1.0 - combined_residual

        if previous is None:
            estimate = signal.value
            evidence_weight = signal.precision
            updated_tick = signal.received_tick
            sources = frozenset((signal.source,))
            evidence_count = 1
        else:
            evidence_weight = previous.evidence_weight + signal.precision
            if self._is_numeric(previous.estimate) and self._is_numeric(signal.value):
                if evidence_weight > 0.0:
                    estimate = (
                        float(previous.estimate) * previous.evidence_weight
                        + float(signal.value) * signal.precision
                    ) / evidence_weight
                else:
                    estimate = previous.estimate
                updated_tick = max(previous.updated_tick, signal.received_tick)
            elif signal.observed_tick >= previous.updated_tick:
                estimate = signal.value
                updated_tick = signal.received_tick
            else:
                estimate = previous.estimate
                updated_tick = previous.updated_tick
            sources = previous.sources | frozenset((signal.source,))
            evidence_count = previous.evidence_count + 1

        belief = Belief(
            topic=signal.topic,
            subject_id=signal.subject_id,
            estimate=_freeze(estimate),
            precision=combined_precision,
            updated_tick=updated_tick,
            evidence_count=evidence_count,
            evidence_weight=evidence_weight,
            sources=sources,
        )
        self._beliefs[key] = belief
        self._residual_uncertainty[key] = combined_residual
        return belief

    def update_many(self, signals: Iterable[Signal]) -> tuple[Belief, ...]:
        """Update from an iterable after validating every element up front."""

        received = tuple(signals)
        if any(not isinstance(signal, Signal) for signal in received):
            raise TypeError("BeliefBook can be updated only from Signal instances")
        return tuple(self.update(signal) for signal in received)

    def get(self, topic: str, subject_id: SubjectId = None) -> Belief | None:
        return self._beliefs.get((topic, subject_id))

    def snapshot(self) -> tuple[Belief, ...]:
        """Return an immutable, deterministic snapshot for an observation view."""

        return tuple(
            belief
            for _, belief in sorted(
                self._beliefs.items(),
                key=lambda item: (item[0][0], type(item[0][1]).__name__, repr(item[0][1])),
            )
        )


__all__ = ["Belief", "BeliefBook", "ObservationView", "Signal"]
