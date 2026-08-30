"""Atomic exploratory progress artifacts with no resume or monetary claims."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import stat
from typing import Final, Sequence

from ..causal.batch import PolicyBatchCheckpoint
from .exploratory import EXPLORATORY_INTERPRETATION_WORDING
from .writers import write_csv_atomic, write_json_atomic


EXPLORATORY_CHECKPOINT_SCHEMA_VERSION: Final[str] = "1.0"
EXPLORATORY_CHECKPOINT_OUTPUT_PROFILE: Final[str] = (
    "exploratory_nonmonetary_unweighted_seed_scenario_diagnostics"
)
EXPLORATORY_CHECKPOINT_COLUMNS: Final[tuple[str, ...]] = (
    "scenario_id",
    "scenario_label",
    "seed",
    "cohort_digest",
    "days",
    "player_count",
    "mean_harm",
    "harm_variance_players",
    "harm_p10",
    "harm_p50",
    "harm_p90",
    "mean_opportunity_cost_score",
    "mean_sleep_burden",
    "mean_education_work_burden",
    "mean_social_burden",
    "mean_wellbeing_burden",
    "mean_enjoyment",
    "high_risk_count",
    "high_risk_share",
    "mean_harm_effect_vs_safe",
    "interpretation",
)


@dataclass(slots=True)
class ExploratoryCheckpointRecorder:
    """Persist the last complete seed prefix in a fresh attempt directory."""

    attempt_dir: Path
    expected_seeds: tuple[int, ...]
    config_sha256: str
    exploratory_plan_id: str
    exploratory_plan_sha256: str
    launch_command: tuple[str, ...]
    _last_retained_count: int = 0

    @classmethod
    def start(
        cls,
        progress_root: str | Path,
        *,
        expected_seeds: Sequence[int],
        config_sha256: str,
        exploratory_plan_id: str,
        exploratory_plan_sha256: str,
        launch_command: Sequence[str],
    ) -> "ExploratoryCheckpointRecorder":
        root = Path(progress_root)
        root.mkdir(parents=True, exist_ok=True)
        root_status = root.lstat()
        reparse_marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if (
            stat.S_ISLNK(root_status.st_mode)
            or bool(
                getattr(root_status, "st_file_attributes", 0)
                & reparse_marker
            )
            or not stat.S_ISDIR(root_status.st_mode)
        ):
            raise ValueError("exploratory progress root must be a real directory")
        attempt_dir: Path | None = None
        for attempt_number in range(1, 1_000_000):
            candidate = root / f"attempt-{attempt_number:06d}"
            try:
                candidate.mkdir()
            except FileExistsError:
                continue
            attempt_dir = candidate
            break
        if attempt_dir is None:
            raise RuntimeError("exploratory checkpoint attempt space exhausted")
        recorder = cls(
            attempt_dir=attempt_dir,
            expected_seeds=tuple(expected_seeds),
            config_sha256=_sha256(config_sha256, name="config_sha256"),
            exploratory_plan_id=_identifier(
                exploratory_plan_id,
                name="exploratory_plan_id",
            ),
            exploratory_plan_sha256=_sha256(
                exploratory_plan_sha256,
                name="exploratory_plan_sha256",
            ),
            launch_command=tuple(launch_command),
        )
        if not recorder.expected_seeds:
            raise ValueError("checkpoint expected_seeds cannot be empty")
        if not recorder.launch_command or any(
            type(item) is not str or not item for item in recorder.launch_command
        ):
            raise ValueError("checkpoint launch_command must be non-empty text")
        recorder._write_progress(status="IN_PROGRESS")
        return recorder

    @property
    def attempt_id(self) -> str:
        return self.attempt_dir.name

    @property
    def progress_path(self) -> Path:
        return self.attempt_dir / "progress.json"

    @property
    def partial_results_path(self) -> Path:
        return self.attempt_dir / "seed_scenario_diagnostics.partial.csv"

    def __call__(self, checkpoint: PolicyBatchCheckpoint) -> None:
        if type(checkpoint) is not PolicyBatchCheckpoint:
            raise TypeError("checkpoint recorder requires PolicyBatchCheckpoint")
        retained = checkpoint.retained_seed_count
        if retained != self._last_retained_count + 1:
            raise ValueError(
                "exploratory checkpoints must arrive once per completed seed"
            )
        if checkpoint.completed_seeds != self.expected_seeds[:retained]:
            raise ValueError(
                "checkpoint completed seeds differ from the declared fixed order"
            )
        rows = checkpoint.nonmonetary_diagnostic_rows()
        if any("cents" in key for row in rows for key in row):
            raise RuntimeError(
                "exploratory checkpoint attempted to expose monetary units"
            )
        write_csv_atomic(
            self.partial_results_path,
            rows,
            canonical_columns=EXPLORATORY_CHECKPOINT_COLUMNS,
            allow_extra_columns=False,
        )
        self._last_retained_count = retained
        self._write_progress(status="IN_PROGRESS")

    def mark_model_batch_complete(self) -> None:
        if self._last_retained_count != len(self.expected_seeds):
            raise ValueError(
                "cannot complete checkpoint attempt before every seed is retained"
            )
        self._write_progress(status="MODEL_BATCH_COMPLETE_EXPORT_PENDING")

    def mark_complete(self) -> None:
        if self._last_retained_count != len(self.expected_seeds):
            raise ValueError(
                "cannot complete checkpoint attempt before every seed is retained"
            )
        self._write_progress(status="COMPLETE")

    def mark_interrupted(self) -> None:
        self._write_progress(status="INTERRUPTED")

    def mark_failed(self, error: BaseException) -> None:
        self._write_progress(
            status="FAILED",
            failure={
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def _write_progress(
        self,
        *,
        status: str,
        failure: dict[str, str] | None = None,
    ) -> None:
        partial_exists = self.partial_results_path.is_file()
        payload: dict[str, object] = {
            "schema_version": EXPLORATORY_CHECKPOINT_SCHEMA_VERSION,
            "output_profile": EXPLORATORY_CHECKPOINT_OUTPUT_PROFILE,
            "attempt_id": self.attempt_id,
            "status": status,
            "retained_seed_count": self._last_retained_count,
            "expected_seed_count": len(self.expected_seeds),
            "completed_seeds": list(
                self.expected_seeds[: self._last_retained_count]
            ),
            "remaining_seeds": list(
                self.expected_seeds[self._last_retained_count :]
            ),
            "last_completed_seed": (
                self.expected_seeds[self._last_retained_count - 1]
                if self._last_retained_count
                else None
            ),
            "config_sha256": self.config_sha256,
            "exploratory_plan_id": self.exploratory_plan_id,
            "exploratory_plan_sha256": self.exploratory_plan_sha256,
            "launch_command": list(self.launch_command),
            "partial_results_file": (
                self.partial_results_path.name if partial_exists else None
            ),
            "partial_results_sha256": (
                _file_sha256(self.partial_results_path)
                if partial_exists
                else None
            ),
            "partial_results_role": (
                "UNWEIGHTED_NONMONETARY_DIAGNOSTIC_ONLY_NOT_PRIMARY_ESTIMAND"
            ),
            "population_weighted_estimand_available": False,
            "monetary_output_present": False,
            "resume_supported": False,
            "restart_behavior": (
                "A new launch starts from seed zero in a new attempt directory; "
                "this attempt remains preserved."
            ),
            "interpretation_wording": EXPLORATORY_INTERPRETATION_WORDING,
            "campaign_ready": False,
        }
        if failure is not None:
            payload["failure"] = failure
        write_json_atomic(self.progress_path, payload)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _identifier(value: object, *, name: str) -> str:
    if type(value) is not str or not value or any(
        character.isspace() for character in value
    ):
        raise ValueError(f"{name} must be a non-empty identifier")
    return value


__all__ = [
    "EXPLORATORY_CHECKPOINT_COLUMNS",
    "EXPLORATORY_CHECKPOINT_OUTPUT_PROFILE",
    "EXPLORATORY_CHECKPOINT_SCHEMA_VERSION",
    "ExploratoryCheckpointRecorder",
]
