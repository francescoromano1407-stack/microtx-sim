"""Stable discrete-event scheduling with O(log n) insertion and removal.

Cancellation is intentionally lazy: cancelling an event only removes its ID
from the live-event table.  The stale heap entry is discarded when it reaches
the head.  This makes cancellation O(1) and avoids scans through the heap.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
import heapq
from numbers import Integral
from types import MappingProxyType
from typing import Any, TypeAlias

from microtx_sim.types import EventKind


EventHandle: TypeAlias = int


def _freeze(value: Any) -> Any:
    """Copy common containers into recursively immutable equivalents."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _require_plain_int(
    value: object, *, name: str, minimum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return integer


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """An immutable event returned by :class:`EventQueue`.

    Monetary values in payloads should be integer minor units (for example,
    cents).  The scheduler never coerces model values to floats.
    """

    tick: int
    kind: EventKind | str
    entity_id: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 0
    event_id: EventHandle = 0

    def __post_init__(self) -> None:
        tick = _require_plain_int(self.tick, name="tick", minimum=0)
        _require_plain_int(self.priority, name="priority")
        event_id = _require_plain_int(self.event_id, name="event_id", minimum=0)
        if self.entity_id is not None:
            entity_id = _require_plain_int(self.entity_id, name="entity_id", minimum=0)
            object.__setattr__(self, "entity_id", entity_id)
        if not isinstance(self.kind, (EventKind, str)):
            raise TypeError("kind must be EventKind or str")
        if isinstance(self.kind, str) and not self.kind:
            raise ValueError("kind must not be empty")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        object.__setattr__(self, "tick", tick)
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "payload", _freeze(self.payload))


class EventQueue:
    """A stable min-heap ordered by ``tick``, ``priority``, then insertion.

    Lower integer priorities run first.  Events with equal tick and priority
    are always popped in scheduling order, independent of their kind or entity
    ID.  Handles are monotonic and are never reused within a queue.
    """

    __slots__ = ("_heap", "_live", "_next_event_id", "_next_sequence")

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, EventHandle, ScheduledEvent]] = []
        self._live: dict[EventHandle, ScheduledEvent] = {}
        self._next_event_id = 1
        self._next_sequence = 0

    def __len__(self) -> int:
        """Number of live events, excluding lazily invalidated entries."""

        return len(self._live)

    def __bool__(self) -> bool:
        return bool(self._live)

    @property
    def heap_entries(self) -> int:
        """Physical heap size, exposed for diagnostics of lazy invalidation."""

        return len(self._heap)

    def schedule(
        self,
        tick: int,
        kind: EventKind | str,
        *,
        entity_id: int | None = None,
        payload: Mapping[str, Any] | None = None,
        priority: int = 0,
    ) -> EventHandle:
        """Schedule an event and return its cancellation handle."""

        event = ScheduledEvent(
            tick=tick,
            kind=kind,
            entity_id=entity_id,
            payload={} if payload is None else payload,
            priority=priority,
        )
        return self.push(event)

    def push(self, event: ScheduledEvent) -> EventHandle:
        """Schedule a pre-built event, assigning it a fresh queue-local ID."""

        if not isinstance(event, ScheduledEvent):
            raise TypeError("event must be a ScheduledEvent")
        event_id = self._next_event_id
        self._next_event_id += 1
        sequence = self._next_sequence
        self._next_sequence += 1
        queued = replace(event, event_id=event_id)
        self._live[event_id] = queued
        heapq.heappush(
            self._heap,
            (queued.tick, queued.priority, sequence, event_id, queued),
        )
        return event_id

    def cancel(self, event_id: EventHandle) -> bool:
        """Lazily invalidate an event; return whether it was still live."""

        _require_plain_int(event_id, name="event_id", minimum=1)
        return self._live.pop(event_id, None) is not None

    def reschedule(self, event_id: EventHandle, *, tick: int) -> EventHandle:
        """Invalidate an event and schedule an immutable copy at a new tick."""

        _require_plain_int(tick, name="tick", minimum=0)
        event = self._live.get(event_id)
        if event is None:
            raise KeyError(event_id)
        self.cancel(event_id)
        return self.push(replace(event, tick=tick, event_id=0))

    def _discard_stale_head(self) -> None:
        while self._heap:
            event_id = self._heap[0][3]
            event = self._heap[0][4]
            if self._live.get(event_id) is event:
                return
            heapq.heappop(self._heap)

    def peek(self) -> ScheduledEvent | None:
        """Return the next live event without removing it, or ``None``."""

        self._discard_stale_head()
        return self._heap[0][4] if self._heap else None

    @property
    def next_tick(self) -> int | None:
        event = self.peek()
        return None if event is None else event.tick

    def pop(self) -> ScheduledEvent:
        """Remove and return the next live event.

        ``IndexError`` is raised when no live events remain, mirroring
        :func:`heapq.heappop` and making accidental empty-loop pops visible.
        """

        self._discard_stale_head()
        if not self._heap:
            raise IndexError("pop from an empty EventQueue")
        _, _, _, event_id, event = heapq.heappop(self._heap)
        del self._live[event_id]
        return event

    def pop_due(self, through_tick: int) -> list[ScheduledEvent]:
        """Pop all live events whose tick is at most ``through_tick``."""

        _require_plain_int(through_tick, name="through_tick", minimum=0)
        due: list[ScheduledEvent] = []
        while True:
            event = self.peek()
            if event is None or event.tick > through_tick:
                return due
            due.append(self.pop())

    def iter_pending(self) -> Iterator[ScheduledEvent]:
        """Iterate over a stable snapshot without mutating the queue."""

        entries = [
            entry for entry in self._heap if self._live.get(entry[3]) is entry[4]
        ]
        for _, _, _, _, event in sorted(entries):
            yield event


# Friendly aliases for modules that use scheduler/event terminology.
EventScheduler = EventQueue
Event = ScheduledEvent

__all__ = [
    "Event",
    "EventHandle",
    "EventQueue",
    "EventScheduler",
    "ScheduledEvent",
]
