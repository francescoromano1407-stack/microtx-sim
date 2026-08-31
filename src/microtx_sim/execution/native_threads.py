"""Fail-closed native NumPy thread-pool control for host parallelism."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class NativeThreadAttestation:
    runtime: str
    library_path: str
    library_sha256: str
    getter_symbol: str
    setter_symbol: str
    previous_thread_count: int
    enforced_thread_count: int

    def snapshot(self) -> dict[str, object]:
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }

    def identity_snapshot(self) -> dict[str, object]:
        """Stable resume contract, excluding invocation-history state."""

        return {
            "runtime": self.runtime,
            "library_path": self.library_path,
            "library_sha256": self.library_sha256,
            "getter_symbol": self.getter_symbol,
            "setter_symbol": self.setter_symbol,
            "enforced_thread_count": self.enforced_thread_count,
        }


class NativeThreadControlError(RuntimeError):
    """Raised when native worker oversubscription cannot be prevented."""


_LOADED_LIBRARIES: list[ctypes.CDLL] = []


def enforce_numpy_native_thread_limit(
    limit: int = 1,
) -> NativeThreadAttestation:
    """Set and verify the loaded OpenBLAS worker count.

    NumPy is already imported by the model, so environment variables alone are
    not accepted as proof.  The active vendor runtime is called directly and
    queried after the update.  Unknown runtimes fail closed.
    """

    if type(limit) is not int or limit != 1:
        raise ValueError("the execution contract requires native thread limit 1")
    numpy_root = Path(np.__file__).resolve(strict=True).parent.parent
    candidates = sorted((numpy_root / "numpy.libs").glob("*openblas*"))
    if len(candidates) != 1 or not candidates[0].is_file():
        raise NativeThreadControlError(
            "cannot identify exactly one NumPy OpenBLAS runtime; bounded host "
            "parallelism is not authorized"
        )
    library_path = candidates[0].resolve(strict=True)
    library = ctypes.CDLL(str(library_path))
    symbol_pairs = (
        (
            "scipy_openblas_get_num_threads64_",
            "scipy_openblas_set_num_threads64_",
        ),
        ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
        ("openblas_get_num_threads", "openblas_set_num_threads"),
    )
    selected = next(
        (
            pair
            for pair in symbol_pairs
            if hasattr(library, pair[0]) and hasattr(library, pair[1])
        ),
        None,
    )
    if selected is None:
        raise NativeThreadControlError(
            "NumPy OpenBLAS runtime does not expose verifiable thread controls"
        )
    getter_name, setter_name = selected
    getter = getattr(library, getter_name)
    setter = getattr(library, setter_name)
    getter.argtypes = []
    getter.restype = ctypes.c_int
    setter.argtypes = [ctypes.c_int]
    setter.restype = None
    previous = int(getter())
    setter(limit)
    observed = int(getter())
    if observed != limit:
        raise NativeThreadControlError(
            "native NumPy thread limit could not be enforced"
        )
    _LOADED_LIBRARIES.append(library)
    return NativeThreadAttestation(
        runtime="scipy-openblas",
        library_path=library_path.as_posix(),
        library_sha256=_file_sha256(library_path),
        getter_symbol=getter_name,
        setter_symbol=setter_name,
        previous_thread_count=previous,
        enforced_thread_count=observed,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "NativeThreadAttestation",
    "NativeThreadControlError",
    "enforce_numpy_native_thread_limit",
]
