"""Atomic deterministic writers for batch simulation artifacts.

The generic CSV writer retains undeclared keys in lexical order for ad-hoc use.
Versioned policy exports disable that extension behavior and reject undeclared
keys against the exhaustive contracts in :mod:`microtx_sim.outputs.schema`.
A missing declared key produces an empty cell, preserving structural checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .schema import (
    ARTIFACT_FILENAMES,
    EPGC_FINANCING_COLUMNS,
    SCENARIO_SUMMARY_COLUMNS,
    SEED_RESULT_COLUMNS,
    SENSITIVITY_COLUMNS,
    stamp_manifest_schema,
)


Row = Mapping[str, object]


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Replace ``path`` atomically with UTF-8 ``text`` in the same directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_csv_atomic(
    path: str | Path,
    rows: Sequence[Row],
    *,
    canonical_columns: Sequence[str] = (),
    allow_extra_columns: bool = True,
) -> Path:
    """Write deterministic CSV, optionally rejecting undeclared row keys."""

    text = _render_csv(
        rows,
        canonical_columns=canonical_columns,
        allow_extra_columns=allow_extra_columns,
    )
    return write_text_atomic(path, text)


def preflight_csv_rows(
    rows: Sequence[Row],
    *,
    canonical_columns: Sequence[str] = (),
    allow_extra_columns: bool = True,
) -> None:
    """Validate row structure and its column contract without writing a file."""

    _prepare_csv_rows(
        rows,
        canonical_columns=canonical_columns,
        allow_extra_columns=allow_extra_columns,
    )


def write_json_atomic(path: str | Path, payload: object) -> Path:
    """Write canonical human-readable JSON, rejecting non-finite numbers."""

    return write_text_atomic(path, _render_json(payload))


def write_batch_artifacts(
    output_dir: str | Path,
    seed_rows: Sequence[Row],
    summary_rows: Sequence[Row],
    epgc_rows: Sequence[Row],
    sensitivity_rows: Sequence[Row],
    manifest: Mapping[str, object],
) -> dict[str, Path]:
    """Write the five stable tabular/metadata artifacts for one batch.

    ``manifest`` may contain arbitrary JSON-compatible domain metadata.  The
    writer owns the output version, manifest version/fingerprint, and artifact
    filenames so a caller cannot describe a different on-disk contract.
    """

    destination = Path(output_dir)
    manifest_payload = stamp_manifest_schema(
        manifest,
        artifact_files=ARTIFACT_FILENAMES,
    )

    # Render every table and the manifest before touching the destination.  A
    # late-table schema/value error therefore cannot leave a partial bundle or
    # replace files in an existing bundle.
    csv_payloads = {
        "seed_results": (
            "seed_results.csv",
            _render_csv(
                seed_rows,
                canonical_columns=SEED_RESULT_COLUMNS,
                allow_extra_columns=False,
            ),
        ),
        "scenario_summary": (
            "scenario_summary.csv",
            _render_csv(
                summary_rows,
                canonical_columns=SCENARIO_SUMMARY_COLUMNS,
                allow_extra_columns=False,
            ),
        ),
        "epgc_financing": (
            "epgc_financing.csv",
            _render_csv(
                epgc_rows,
                canonical_columns=EPGC_FINANCING_COLUMNS,
                allow_extra_columns=False,
            ),
        ),
        "sensitivity": (
            "sensitivity.csv",
            _render_csv(
                sensitivity_rows,
                canonical_columns=SENSITIVITY_COLUMNS,
                allow_extra_columns=False,
            ),
        ),
    }
    manifest_text = _render_json(manifest_payload)

    paths = {
        name: write_text_atomic(destination / filename, text)
        for name, (filename, text) in csv_payloads.items()
    }
    paths["manifest"] = write_text_atomic(destination / "manifest.json", manifest_text)
    return paths


def _prepare_csv_rows(
    rows: Sequence[Row],
    *,
    canonical_columns: Sequence[str],
    allow_extra_columns: bool,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    materialized = _materialize_rows(rows)
    columns = _column_order(
        canonical_columns,
        materialized,
        allow_extra_columns=allow_extra_columns,
    )
    return materialized, columns


def _render_csv(
    rows: Sequence[Row],
    *,
    canonical_columns: Sequence[str],
    allow_extra_columns: bool,
) -> str:
    materialized, columns = _prepare_csv_rows(
        rows,
        canonical_columns=canonical_columns,
        allow_extra_columns=allow_extra_columns,
    )
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    if columns:
        writer.writerow(columns)
        for row in materialized:
            writer.writerow(_csv_value(row.get(column)) for column in columns)
    return buffer.getvalue()


def _render_json(payload: object) -> str:
    normalized = _json_compatible(payload)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return text + "\n"


def _materialize_rows(rows: Sequence[Row]) -> list[dict[str, object]]:
    if isinstance(rows, (str, bytes, bytearray)):
        raise TypeError("CSV rows must be a sequence of mappings")
    result: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"CSV row {index} is not a mapping")
        converted: dict[str, object] = {}
        for key, value in row.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"CSV row {index} has a non-string or empty key")
            converted[key] = value
        result.append(converted)
    return result


def _column_order(
    canonical_columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
    *,
    allow_extra_columns: bool,
) -> tuple[str, ...]:
    canonical = tuple(canonical_columns)
    if len(set(canonical)) != len(canonical) or any(
        not isinstance(column, str) or not column for column in canonical
    ):
        raise ValueError("canonical CSV columns must be unique non-empty strings")
    extra = tuple(
        sorted(
            {
                key
                for row in rows
                for key in row
                if key not in canonical
            }
        )
    )
    if extra and not allow_extra_columns:
        raise ValueError(
            "versioned CSV rows contain undeclared columns: " + ", ".join(extra)
        )
    return canonical + extra


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("CSV values must not contain NaN or infinity")
        if value == 0.0:
            return "0"
        return repr(value)
    if isinstance(value, (Mapping, list, tuple, set, frozenset)) or is_dataclass(value):
        return json.dumps(
            _json_compatible(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    scalar = _scalar_item(value)
    if scalar is not value:
        return _csv_value(scalar)
    return str(value)


def _json_compatible(value: object) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value))
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("JSON mapping keys must be non-empty strings")
            converted[key] = _json_compatible(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_compatible(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    scalar = _scalar_item(value)
    if scalar is not value:
        return _json_compatible(scalar)
    raise TypeError(f"value of type {type(value).__name__} is not JSON compatible")


def _scalar_item(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


__all__ = [
    "preflight_csv_rows",
    "write_batch_artifacts",
    "write_csv_atomic",
    "write_json_atomic",
    "write_text_atomic",
]
