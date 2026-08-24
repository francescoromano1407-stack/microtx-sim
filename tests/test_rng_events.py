from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import IntEnum
import unittest

import numpy as np

from microtx_sim.core import BeliefBook, EventQueue, ObservationView, Signal
from microtx_sim.rng import CounterRNG, stable_stream_id
from microtx_sim.types import EventKind, InformationSource


class TestStream(IntEnum):
    PLAYER_CHOICE = 17
    RARE_EVENT = 23


class CounterRNGTests(unittest.TestCase):
    def test_reproducible_and_stable_named_streams(self) -> None:
        entity_ids = np.arange(1_000, dtype=np.int64)
        first = CounterRNG(20260824).uniform(
            entity_ids, 12, TestStream.PLAYER_CHOICE, 3
        )
        second = CounterRNG(20260824).uniform(
            entity_ids, 12, int(TestStream.PLAYER_CHOICE), 3
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(stable_stream_id("player-choice"), 6680835771791415058)
        np.testing.assert_array_equal(
            CounterRNG(20260824).uint64(
                np.array([1, 2, 3], dtype=np.int64), 12, 17, 3
            ),
            np.array(
                [
                    4806970298738978266,
                    13731674267806983033,
                    12714090901024793952,
                ],
                dtype=np.uint64,
            ),
        )
        self.assertEqual(
            stable_stream_id("player-choice"), stable_stream_id("player-choice")
        )
        self.assertNotEqual(
            stable_stream_id("player-choice"), stable_stream_id("rare-event")
        )

    def test_independent_of_entity_iteration_order(self) -> None:
        rng = CounterRNG(91)
        entity_ids = np.array([91, 7, 400, 18, 2], dtype=np.int64)
        permutation = np.array([2, 4, 0, 3, 1])

        for method in ("uniform", "normal"):
            with self.subTest(method=method):
                ordered = getattr(rng, method)(
                    entity_ids, 8, TestStream.PLAYER_CHOICE, 0
                )
                shuffled = getattr(rng, method)(
                    entity_ids[permutation], 8, TestStream.PLAYER_CHOICE, 0
                )
                np.testing.assert_array_equal(shuffled, ordered[permutation])

    def test_coordinates_and_vectorised_distributions_are_separate(self) -> None:
        rng = CounterRNG(404)
        entities = np.arange(20_000, dtype=np.int64)

        baseline = rng.uint64(entities, 2, 10, 0)
        self.assertTrue(np.any(baseline != rng.uint64(entities, 3, 10, 0)))
        self.assertTrue(np.any(baseline != rng.uint64(entities, 2, 11, 0)))
        self.assertTrue(np.any(baseline != rng.uint64(entities, 2, 10, 1)))

        normal = rng.normal(entities, 5, 12, 0)
        bernoulli = rng.bernoulli(
            entities, 5, 13, 0, probability=np.linspace(0.0, 1.0, entities.size)
        )
        self.assertEqual(normal.shape, entities.shape)
        self.assertEqual(normal.dtype, np.float64)
        self.assertEqual(bernoulli.shape, entities.shape)
        self.assertEqual(bernoulli.dtype, np.bool_)
        self.assertFalse(bernoulli[0])
        self.assertTrue(bernoulli[-1])


class EventQueueTests(unittest.TestCase):
    def test_stable_order_and_lazy_cancellation(self) -> None:
        queue = EventQueue()
        late = queue.schedule(9, EventKind.AUDIT_DUE, entity_id=4)
        first_equal = queue.schedule(4, "first", entity_id=1, priority=2)
        cancelled = queue.schedule(4, "cancelled", entity_id=2, priority=1)
        second_equal = queue.schedule(4, "second", entity_id=3, priority=2)

        self.assertTrue(
            all(
                handle > 0
                for handle in (late, first_equal, cancelled, second_equal)
            )
        )
        self.assertTrue(queue.cancel(cancelled))
        self.assertEqual(len(queue), 3)
        # Cancellation does not scan/remove an arbitrary heap entry.
        self.assertEqual(queue.heap_entries, 4)

        due = queue.pop_due(4)
        self.assertEqual([event.kind for event in due], ["first", "second"])
        self.assertEqual(
            [event.event_id for event in due], [first_equal, second_equal]
        )
        self.assertEqual(queue.next_tick, 9)
        self.assertEqual(queue.pop().event_id, late)
        self.assertIsNone(queue.peek())
        with self.assertRaises(IndexError):
            queue.pop()

    def test_payload_is_frozen_and_reschedule_invalidates_old(self) -> None:
        queue = EventQueue()
        source_payload = {"fine_cents": 10_000, "evidence": ["complaint"]}
        original = queue.schedule(
            7, EventKind.AUDIT_RESOLUTION, payload=source_payload
        )
        replacement = queue.reschedule(original, tick=3)
        source_payload["fine_cents"] = 0
        source_payload["evidence"].append("latent_truth")

        self.assertEqual(len(queue), 1)
        self.assertEqual(queue.heap_entries, 2)
        event = queue.pop()
        self.assertEqual(event.event_id, replacement)
        self.assertEqual(event.tick, 3)
        self.assertEqual(event.payload["fine_cents"], 10_000)
        self.assertEqual(event.payload["evidence"], ("complaint",))


class InformationBoundaryTests(unittest.TestCase):
    def test_view_is_immutable_and_detached_from_source_values(self) -> None:
        mutable_value = {"ranking": [1, 2, 3]}
        signal = Signal(
            topic="public_ranking",
            value=mutable_value,
            observed_tick=3,
            received_tick=5,
            source=InformationSource.PUBLIC_RANKING,
            precision=0.7,
        )
        source_signals = [signal]
        view = ObservationView(observer_id=8, tick=5, signals=source_signals)

        mutable_value["ranking"].append(99)
        source_signals.clear()

        self.assertEqual(len(view.signals), 1)
        self.assertEqual(view.signals[0].value["ranking"], (1, 2, 3))
        self.assertEqual(view.signals[0].age(8), 5)
        with self.assertRaises(TypeError):
            view.signals[0].value["ranking"] = (4,)
        with self.assertRaises(FrozenInstanceError):
            view.tick = 9

    def test_beliefs_require_signals_and_view_rejects_future_information(self) -> None:
        low = Signal(
            "market_size",
            100,
            observed_tick=1,
            received_tick=2,
            source=InformationSource.PUBLIC_RANKING,
            precision=0.25,
            subject_id="game-a",
        )
        high = Signal(
            "market_size",
            200,
            observed_tick=2,
            received_tick=2,
            source=InformationSource.PAID_RESEARCH,
            precision=0.75,
            subject_id="game-a",
            cost_cents=50_000,
        )
        book = BeliefBook((low, high))
        belief = book.get("market_size", "game-a")

        self.assertIsNotNone(belief)
        assert belief is not None
        self.assertAlmostEqual(belief.estimate, 175.0)
        self.assertAlmostEqual(belief.precision, 0.8125)
        with self.assertRaises(TypeError):
            book.update({"market_size": 999})

        future = Signal(
            "audit_risk",
            0.8,
            observed_tick=3,
            received_tick=6,
            source=InformationSource.AUDIT_EVIDENCE,
            precision=0.9,
        )
        with self.assertRaisesRegex(ValueError, "before receipt"):
            ObservationView(observer_id="firm-1", tick=5, signals=(future,))


if __name__ == "__main__":
    unittest.main()
