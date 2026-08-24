"""Deterministic, stateless random numbers for agent-based simulations.

The generator in this module is *counter based*: a draw is a pure function of
``(seed, entity_id, tick, stream, draw_index)``.  There is no mutable generator
cursor, so changing iteration order, population chunking, or the number of
draws made by another subsystem cannot perturb an agent's draw.

This is not a cryptographic random-number generator.  SplitMix64 is used as a
fast integer mixer and its output is converted to NumPy ``float64`` values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt


UIntArray: TypeAlias = npt.NDArray[np.uint64]
FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntegerInput: TypeAlias = int | IntEnum | npt.ArrayLike

_UINT64_MASK = (1 << 64) - 1
_FNV_OFFSET_BASIS = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_TWO_NEG_53 = 1.0 / (1 << 53)

# Domain constants make the five coordinates non-interchangeable.  Their
# literal values are part of the reproducibility contract and must not be
# generated through Python's process-randomised ``hash`` function.
_SEED_DOMAIN = np.uint64(0x243F6A8885A308D3)
_ENTITY_DOMAIN = np.uint64(0x13198A2E03707344)
_TICK_DOMAIN = np.uint64(0xA4093822299F31D0)
_STREAM_DOMAIN = np.uint64(0x082EFA98EC4E6C89)
_DRAW_DOMAIN = np.uint64(0x452821E638D01377)
_NORMAL_FIRST_LANE = np.uint64(0xBE5466CF34E90C6C)
_NORMAL_SECOND_LANE = np.uint64(0xC0AC29B7C97C50DD)


def stable_stream_id(name: str | bytes) -> int:
    """Return a stable unsigned 64-bit ID for a named random stream.

    FNV-1a is deliberately implemented here rather than using ``hash(name)``:
    Python hashes strings with a per-process salt, which would silently break
    experiment reproducibility.  Numeric stream IDs and :class:`IntEnum`
    members may be passed directly to :class:`CounterRNG` methods.
    """

    if isinstance(name, str):
        encoded = name.encode("utf-8")
    elif isinstance(name, bytes):
        encoded = name
    else:
        raise TypeError("stream names must be str or bytes")

    value = _FNV_OFFSET_BASIS
    # A fixed namespace prevents accidental equivalence with an FNV ID used by
    # another application while retaining a documented, portable algorithm.
    for byte in b"microtx-sim\0" + encoded:
        value ^= byte
        value = (value * _FNV_PRIME) & _UINT64_MASK
    return value


def _integer_array(value: IntegerInput, *, name: str) -> UIntArray:
    """Convert integer-like scalar/array input to uint64 without mutation."""

    array = np.asarray(value)
    if array.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integers")
    # Converting signed values to uint64 gives a documented two's-complement
    # mapping.  This also gives negative sentinel entity IDs stable draws.
    return array.astype(np.uint64, copy=False)


def _stream_array(stream: IntegerInput | str | bytes) -> UIntArray:
    if isinstance(stream, (str, bytes)):
        return np.asarray(stable_stream_id(stream), dtype=np.uint64)
    return _integer_array(stream, name="stream")


def _splitmix64(value: UIntArray | np.uint64) -> UIntArray:
    """Vectorised SplitMix64 finalisation with intentional modulo overflow."""

    z = np.asarray(value, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return np.asarray(z ^ (z >> np.uint64(31)), dtype=np.uint64)


@dataclass(frozen=True, slots=True)
class CounterRNG:
    """A reproducible random field indexed by simulation coordinates.

    The first four arguments of distribution methods are intentionally
    positional as well as keyword-compatible.  ``entity_ids`` and any other
    integer coordinate may be scalar or broadcastable NumPy arrays.  A player
    vector therefore receives an array of the same shape, while callers can
    vary draw indices per entity when needed.
    """

    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed) & _UINT64_MASK)

    def _words(
        self,
        entity_ids: IntegerInput,
        tick: IntegerInput,
        stream: IntegerInput | str | bytes,
        draw_index: IntegerInput,
        *,
        lane: np.uint64 = np.uint64(0),
    ) -> UIntArray:
        entities, ticks, streams, draws = np.broadcast_arrays(
            _integer_array(entity_ids, name="entity_ids"),
            _integer_array(tick, name="tick"),
            _stream_array(stream),
            _integer_array(draw_index, name="draw_index"),
        )
        seed = np.asarray(self.seed, dtype=np.uint64)
        # Mix each coordinate before combining it.  This avoids low-bit
        # regularity and gives semantically different coordinates their own
        # domains even when their integer values happen to match.
        combined = (
            _splitmix64(seed ^ _SEED_DOMAIN)
            ^ _splitmix64(entities ^ _ENTITY_DOMAIN)
            ^ _splitmix64(ticks ^ _TICK_DOMAIN)
            ^ _splitmix64(streams ^ _STREAM_DOMAIN)
            ^ _splitmix64(draws ^ _DRAW_DOMAIN)
            ^ lane
        )
        return _splitmix64(combined)

    def uint64(
        self,
        entity_ids: IntegerInput,
        tick: IntegerInput,
        stream: IntegerInput | str | bytes,
        draw_index: IntegerInput,
    ) -> UIntArray:
        """Return raw deterministic 64-bit words for the given coordinates."""

        return self._words(entity_ids, tick, stream, draw_index)

    # A descriptive alias is useful at call sites that distinguish raw words
    # from fixed-width model state.
    random_u64 = uint64

    @staticmethod
    def _unit_interval(words: UIntArray) -> FloatArray:
        # Retain the 53 high bits, exactly the precision of a float64 mantissa.
        return ((words >> np.uint64(11)).astype(np.float64)) * _TWO_NEG_53

    def uniform(
        self,
        entity_ids: IntegerInput,
        tick: IntegerInput,
        stream: IntegerInput | str | bytes,
        draw_index: IntegerInput,
        low: npt.ArrayLike = 0.0,
        high: npt.ArrayLike = 1.0,
    ) -> FloatArray:
        """Return vectorised uniforms on ``[low, high)``."""

        low_array = np.asarray(low, dtype=np.float64)
        high_array = np.asarray(high, dtype=np.float64)
        if np.any(~np.isfinite(low_array)) or np.any(~np.isfinite(high_array)):
            raise ValueError("uniform bounds must be finite")
        if np.any(high_array <= low_array):
            raise ValueError("uniform requires high > low")
        unit = self._unit_interval(
            self._words(entity_ids, tick, stream, draw_index)
        )
        return np.asarray(low_array + (high_array - low_array) * unit, dtype=np.float64)

    def normal(
        self,
        entity_ids: IntegerInput,
        tick: IntegerInput,
        stream: IntegerInput | str | bytes,
        draw_index: IntegerInput,
        loc: npt.ArrayLike = 0.0,
        scale: npt.ArrayLike = 1.0,
    ) -> FloatArray:
        """Return vectorised normal draws using a domain-separated Box--Muller map."""

        scale_array = np.asarray(scale, dtype=np.float64)
        loc_array = np.asarray(loc, dtype=np.float64)
        if np.any(~np.isfinite(scale_array)) or np.any(scale_array < 0.0):
            raise ValueError("normal scale must be finite and non-negative")
        if np.any(~np.isfinite(loc_array)):
            raise ValueError("normal location must be finite")

        first = self._unit_interval(
            self._words(
                entity_ids,
                tick,
                stream,
                draw_index,
                lane=_NORMAL_FIRST_LANE,
            )
        )
        # A raw uniform can be exactly zero.  Moving only that endpoint to the
        # centre of its representable bin prevents log(0) without consuming a
        # variable number of counters or introducing iteration-order effects.
        first = np.maximum(first, 0.5 * _TWO_NEG_53)
        second = self._unit_interval(
            self._words(
                entity_ids,
                tick,
                stream,
                draw_index,
                lane=_NORMAL_SECOND_LANE,
            )
        )
        standard = np.sqrt(-2.0 * np.log(first)) * np.cos(2.0 * np.pi * second)
        return np.asarray(loc_array + scale_array * standard, dtype=np.float64)

    def bernoulli(
        self,
        entity_ids: IntegerInput,
        tick: IntegerInput,
        stream: IntegerInput | str | bytes,
        draw_index: IntegerInput,
        probability: npt.ArrayLike = 0.5,
    ) -> BoolArray:
        """Return vectorised Bernoulli draws for probabilities in ``[0, 1]``."""

        probability_array = np.asarray(probability, dtype=np.float64)
        if np.any(~np.isfinite(probability_array)) or np.any(
            (probability_array < 0.0) | (probability_array > 1.0)
        ):
            raise ValueError("Bernoulli probabilities must be finite and in [0, 1]")
        unit = self._unit_interval(
            self._words(entity_ids, tick, stream, draw_index)
        )
        return np.asarray(unit < probability_array, dtype=np.bool_)


__all__ = ["CounterRNG", "stable_stream_id"]
