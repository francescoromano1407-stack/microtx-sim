"""Atomic deterministic writers for batch simulation artifacts.

Rows are supplied as sequences of string-keyed mappings.  Canonical keys are
documented in :mod:`microtx_sim.outputs.schema`; callers may add extra keys,
which are emitted in lexical order after the canonical columns.  A missing key
produces an empty CSV cell, allowing empty and partially populated structural
checks to use the same file schema as full batches.
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
    OUTPUT_SCHEMA_VERSION,
    SCENARIO_SUMMARY_COLUMNS,
    SEED_RESULT_COLUMNS,
    SENSITIVITY_COLUMNS,
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
) -> Path:
    """Write a deterministic RFC-4180-style CSV with a stable column order."""

    materialized = _materialize_rows(rows)
    columns = _column_order(canonical_columns, materialized)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    if columns:
        writer.writerow(columns)
        for row in materialized:
            writer.writerow(_csv_value(row.get(column)) for column in columns)
    return write_text_atomic(path, buffer.getvalue())


def write_json_atomic(path: str | Path, payload: object) -> Path:
    """Write canonical human-readable JSON, rejecting non-finite numbers."""

    normalized = _json_compatible(payload)
    text = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    return write_text_atomic(path, text + "\n")


def write_batch_artifacts(
    output_dir: str | Path,
    seed_rows: Sequence[Row],
    summary_rows: Sequence[Row],
    epgc_rows: Sequence[Row],
    sensitivity_rows: Sequence[Row],
    manifest: Mapping[str, object],
) -> dict[str, Path]:
    """Write the five stable tabular/metadata artifacts for one batch.

    ``manifest`` may contain arbitrary JSON-compatible metadata.  The writer
    owns ``output_schema_version`` and ``artifact_files`` so a caller cannot
    accidentally describe a different on-disk contract.
    """

    destination = Path(output_dir)
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    manifest_payload = dict(manifest)
    declared_version = manifest_payload.get("output_schema_version")
    if declared_version not in (None, OUTPUT_SCHEMA_VERSION):
        raise ValueError(
            "manifest output_schema_version conflicts with the writer schema"
        )
    declared_files = manifest_payload.get("artifact_files")
    if declared_files not in (None, ARTIFACT_FILENAMES):
        if declared_files != list(ARTIFACT_FILENAMES):
            raise ValueError("manifest artifact_files conflicts with stable filenames")
    manifest_payload["output_schema_version"] = OUTPUT_SCHEMA_VERSION
    manifest_payload["artifact_files"] = list(ARTIFACT_FILENAMES)

    paths = {
        "seed_results": write_csv_atomic(
            destination / "seed_results.csv",
            seed_rows,
            canonical_columns=SEED_RESULT_COLUMNS,
        ),
        "scenario_summary": write_csv_atomic(
            destination / "scenario_summary.csv",
            summary_rows,
            canonical_columns=SCENARIO_SUMMARY_COLUMNS,
        ),
        "epgc_financing": write_csv_atomic(
            destination / "epgc_financing.csv",
            epgc_rows,
            canonical_columns=EPGC_FINANCING_COLUMNS,
        ),
        "sensitivity": write_csv_atomic(
            destination / "sensitivity.csv",
            sensitivity_rows,
            canonical_columns=SENSITIVITY_COLUMNS,
        ),
    }
    paths["manifest"] = write_json_atomic(
        destination / "manifest.json", manifest_payload
    )
    return paths


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
    canonical_columns: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> tuple[str, ...]:
    canonical = tuple(canonical_columns)
    if len(set(canonical)) != len(canonical) or any(
        not isinstance(column, str) or not column for column in canonical
    ):
        raise ValueError("canonical CSV columns must be unique non-empty strings")
    extra = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in canonical
        }
    )
    return canonical + tuple(extra)


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
    "write_batch_artifacts",
    "write_csv_atomic",
    "write_json_atomic",
    "write_text_atomic",
]
