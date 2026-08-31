"""Crash-released single-coordinator lease for one execution attempt."""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType


class ConcurrentExecutionError(RuntimeError):
    """Raised when another process already coordinates the same attempt."""


class AttemptCoordinatorLease:
    """Hold a non-blocking one-byte OS lock for the lifetime of a launch."""

    def __init__(self, path: str | Path) -> None:
        selected = Path(path)
        if not selected.is_absolute():
            raise ValueError("execution lease path must be absolute")
        self.path = selected
        self._handle = None
        self._locked = False

    def acquire(self) -> "AttemptCoordinatorLease":
        if self._handle is not None:
            raise RuntimeError("execution lease is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ValueError("execution lease path cannot be a symlink")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        self._handle = handle
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                os.fsync(handle.fileno())
            handle.seek(0)
            self._lock_nonblocking()
            self._locked = True
            return self
        except BaseException:
            handle.close()
            self._handle = None
            self._locked = False
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            if self._locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._locked = False
            self._handle = None
            handle.close()

    def _lock_nonblocking(self) -> None:
        assert self._handle is not None
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(
                    self._handle.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
        except OSError as exc:
            raise ConcurrentExecutionError(
                "another process already coordinates execution attempt "
                f"{self.path.name.removeprefix('.').removesuffix('.lock')}"
            ) from exc

    def __enter__(self) -> "AttemptCoordinatorLease":
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


__all__ = ["AttemptCoordinatorLease", "ConcurrentExecutionError"]
