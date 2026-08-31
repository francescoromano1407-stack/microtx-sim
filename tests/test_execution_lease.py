from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from microtx_sim.execution.lease import (
    AttemptCoordinatorLease,
    ConcurrentExecutionError,
)


def test_second_attempt_coordinator_is_rejected_and_release_is_reusable() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory).resolve() / ".attempt-000002.lock"
        first = AttemptCoordinatorLease(path).acquire()
        try:
            with pytest.raises(
                ConcurrentExecutionError,
                match="another process already coordinates",
            ):
                AttemptCoordinatorLease(path).acquire()
        finally:
            first.release()

        with AttemptCoordinatorLease(path):
            assert path.is_file()


def test_attempt_coordinator_requires_absolute_non_symlink_path() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        AttemptCoordinatorLease(Path("relative.lock"))
