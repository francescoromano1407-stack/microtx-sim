from __future__ import annotations

import csv
from decimal import Decimal
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Callable

import pytest

from microtx_sim.data.calibration import (
    CalibrationBundleValidationError,
    CalibrationBundleVerificationError,
    EstimandRole,
    EvidenceStatus,
    POPULATION_WEIGHT_CSV_COLUMNS,
    TARGET_CSV_COLUMNS,
    UK_ADULTS_2024_POPULATION_COUNT,
    UNSUPPORTED_CONCEPTS,
    load_uk_adults_2024_calibration_bundle,
)


_POPULATION_CELLS = (
    ("18-24", 18, 24, "FEMALE", 2_821_237),
    ("18-24", 18, 24, "MALE", 2_970_284),
    ("25-34", 25, 34, "FEMALE", 4_754_911),
    ("25-34", 25, 34, "MALE", 4_590_686),
    ("35-44", 35, 44, "FEMALE", 4_805_400),
    ("35-44", 35, 44, "MALE", 4_499_492),
    ("45-54", 45, 54, "FEMALE", 4_343_516),
    ("45-54", 45, 54, "MALE", 4_169_961),
    ("55-64", 55, 64, "FEMALE", 4_554_117),
    ("55-64", 55, 64, "MALE", 4_350_983),
)


def _csv_bytes(columns: tuple[str, ...], rows: list[dict[str, object]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _target(
    target_id: str,
    concept: str,
    role: str,
    status: str,
    source_id: str,
    source_file: str,
    source_locator: str,
    *,
    value: str = "",
    unit: str = "not_identified",
    lower_ci: str = "",
    upper_ci: str = "",
) -> dict[str, object]:
    return {
        "target_id": target_id,
        "concept": concept,
        "estimand_role": role,
        "evidence_status": status,
        "geography": "UK",
        "population": "UK resident adults aged 18 to 64",
        "reference_period": "2024",
        "subgroup": "ALL",
        "value": value,
        "unit": unit,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "source_id": source_id,
        "source_file": source_file,
        "source_locator": source_locator,
        "transformation": "Exact transcription from the named source record.",
        "runtime_mapping": "NOT_CONNECTED",
        "limitations": "The evidence bundle does not itself alter runtime inputs.",
    }


def _source(
    source_id: str,
    relative_path: str,
    content: bytes,
    evidence_role: str,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "relative_path": relative_path,
        "sha256": sha256(content).hexdigest(),
        "byte_length": len(content),
        "official_url": f"https://example.org/{relative_path}",
        "licence": "Test fixture licence",
        "notes": "",
        "publisher": "Test publisher",
        "title": f"Fixture {source_id}",
        "version": "1",
        "publication_date": "2024-01-01",
        "retrieved_at": None,
        "evidence_role": evidence_role,
    }


def _write_fixture(
    root: Path,
    *,
    mutate_targets: Callable[[list[dict[str, object]]], None] | None = None,
    mutate_weights: Callable[[list[dict[str, object]]], None] | None = None,
    mutate_manifest: Callable[[dict[str, object]], None] | None = None,
    mutate_bundle: Callable[[dict[str, object]], None] | None = None,
) -> tuple[Path, Path]:
    raw_cache = root / "data" / "public_calibration_sources_uk_adults_2024"
    raw_cache.mkdir(parents=True)
    source_contents = {
        "frs.txt": b"Published FRS rounded margins: 5,14,18,14,11,9,7,5,4,3,11\n",
        "index.md": b"Evidence gaps remain unquantified or normative.\n",
        "ons_mid_2024.txt": b"ONS mid-2024 age-by-sex population cells.\n",
        "ons_time_use_mar_2024.txt": b"ONS March 2024 time-use calibration records.\n",
        "ons_time_use_sep_oct_2023.txt": b"ONS Sep-Oct 2023 holdout records.\n",
    }
    for relative_path, content in source_contents.items():
        (raw_cache / relative_path).write_bytes(content)

    sources = [
        _source("frs_2023_24", "frs.txt", source_contents["frs.txt"], "CALIBRATION"),
        _source("gap_index", "index.md", source_contents["index.md"], "REFERENCE"),
        _source(
            "ons_mid_2024",
            "ons_mid_2024.txt",
            source_contents["ons_mid_2024.txt"],
            "CALIBRATION",
        ),
        _source(
            "ons_time_use_mar_2024",
            "ons_time_use_mar_2024.txt",
            source_contents["ons_time_use_mar_2024.txt"],
            "CALIBRATION",
        ),
        _source(
            "ons_time_use_sep_oct_2023",
            "ons_time_use_sep_oct_2023.txt",
            source_contents["ons_time_use_sep_oct_2023.txt"],
            "VALIDATION",
        ),
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "bundle_id": "uk-adults-2024-v1",
        "raw_cache_root": "data/public_calibration_sources_uk_adults_2024",
        "raw_cache_tracked": False,
        "verified_at": "2026-09-01",
        "sources": sources,
    }
    if mutate_manifest is not None:
        mutate_manifest(manifest)

    targets = [
        _target(
            "frs.rounded.margin",
            "gross_weekly_household_income_published_margin",
            "CALIBRATION",
            "QUANTIFIED",
            "frs_2023_24",
            "frs.txt",
            "Table_2_5:published_percentage_row",
            value="101",
            unit="percent",
        ),
        _target(
            "ons.population.total",
            "uk_adult_population_18_64",
            "CALIBRATION",
            "QUANTIFIED",
            "ons_mid_2024",
            "ons_mid_2024.txt",
            "MYE2:ages_18_64",
            value="41860587",
            unit="persons",
        ),
        _target(
            "ons.sleep.mar_2024",
            "daily_sleep_and_rest_minutes",
            "CALIBRATION",
            "QUANTIFIED",
            "ons_time_use_mar_2024",
            "ons_time_use_mar_2024.txt",
            "Table_1:Sleeping_and_resting",
            value="518.5",
            unit="minutes_per_day",
            lower_ci="513.4",
            upper_ci="523.7",
        ),
        _target(
            "ons.sleep.sep_oct_2023",
            "daily_sleep_and_rest_minutes",
            "VALIDATION",
            "QUANTIFIED",
            "ons_time_use_sep_oct_2023",
            "ons_time_use_sep_oct_2023.txt",
            "Table_1:Sleeping_and_resting",
            value="522.8",
            unit="minutes_per_day",
            lower_ci="517.3",
            upper_ci="528.2",
        ),
    ]
    for concept in UNSUPPORTED_CONCEPTS:
        status = "NORMATIVE" if concept == "composite_harm_weights" else "UNQUANTIFIED"
        targets.append(
            _target(
                f"gap.{concept}",
                concept,
                "DIAGNOSTIC",
                status,
                "gap_index",
                "index.md",
                f"gap:{concept}",
            )
        )
    if mutate_targets is not None:
        mutate_targets(targets)

    weights: list[dict[str, object]] = []
    for age_band, age_min, age_max, sex, count in _POPULATION_CELLS:
        weight = Decimal(count) / Decimal(UK_ADULTS_2024_POPULATION_COUNT)
        weights.append(
            {
                "age_band": age_band,
                "age_min_inclusive": age_min,
                "age_max_inclusive": age_max,
                "sex": sex,
                "population_count": count,
                "adult_population_weight": format(weight, ".18f"),
                "source_id": "ons_mid_2024",
                "estimand_role": "CALIBRATION",
            }
        )
    if mutate_weights is not None:
        mutate_weights(weights)

    bundle_root = root / "inputs" / "calibration" / "uk-adults-2024-v1"
    bundle_root.mkdir(parents=True)
    companion_bytes = {
        "targets.csv": _csv_bytes(TARGET_CSV_COLUMNS, targets),
        "population_weights.csv": _csv_bytes(POPULATION_WEIGHT_CSV_COLUMNS, weights),
        "source_manifest.json": (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8"),
    }
    for filename, content in companion_bytes.items():
        (bundle_root / filename).write_bytes(content)
    bundle: dict[str, object] = {
        "schema_version": 1,
        "bundle_id": "uk-adults-2024-v1",
        "status": "PARTIAL",
        "files": {
            filename: {
                "sha256": sha256(content).hexdigest(),
                "byte_length": len(content),
            }
            for filename, content in companion_bytes.items()
        },
        "frs_rounded_margin": {
            "target_id": "frs.rounded.margin",
            "published_sum_percent": 101,
            "normalization_applied": False,
        },
        "unsupported_concepts": list(UNSUPPORTED_CONCEPTS),
        "campaign_ready": False,
        "blockers": sorted(f"unsupported:{concept}" for concept in UNSUPPORTED_CONCEPTS),
    }
    if mutate_bundle is not None:
        mutate_bundle(bundle)
    bundle_path = bundle_root / "calibration_bundle.json"
    bundle_path.write_text(
        json.dumps(
            bundle,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )
    return bundle_root, raw_cache


def test_loads_verified_partial_bundle_and_retains_missing_retrieval_dates(
    tmp_path: Path,
) -> None:
    bundle_root, _ = _write_fixture(tmp_path)

    bundle = load_uk_adults_2024_calibration_bundle(
        bundle_root,
        repository_root=tmp_path,
    )

    assert bundle.schema_version == 1
    assert bundle.status == "PARTIAL"
    assert bundle.campaign_ready is False
    assert len(bundle.targets) == 9
    assert len(bundle.population_weights) == 10
    assert sum(weight.population_count for weight in bundle.population_weights) == (
        41_860_587
    )
    assert all(source.retrieved_at is None for source in bundle.sources)
    assert bundle.targets[0].estimand_role is EstimandRole.CALIBRATION
    assert bundle.targets[0].evidence_status is EvidenceStatus.QUANTIFIED
    with pytest.raises(CalibrationBundleValidationError, match="cannot authorize"):
        bundle.validate_for_campaign()


def test_rejects_companion_file_hash_mismatch(tmp_path: Path) -> None:
    bundle_root, _ = _write_fixture(tmp_path)
    target_path = bundle_root / "targets.csv"
    target_path.write_bytes(target_path.read_bytes() + b"\n")

    with pytest.raises(
        CalibrationBundleVerificationError,
        match="byte length differs",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_raw_source_hash_mismatch(tmp_path: Path) -> None:
    bundle_root, raw_cache = _write_fixture(tmp_path)
    source_path = raw_cache / "ons_mid_2024.txt"
    original = source_path.read_bytes()
    source_path.write_bytes(b"X" + original[1:])

    with pytest.raises(CalibrationBundleVerificationError, match="SHA-256 differs"):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_promotion_of_unsupported_concept(tmp_path: Path) -> None:
    def promote(rows: list[dict[str, object]]) -> None:
        row = next(row for row in rows if row["concept"] == "decision.temperature")
        row["estimand_role"] = "CALIBRATION"
        row["evidence_status"] = "QUANTIFIED"
        row["value"] = "0.35"
        row["source_id"] = "ons_mid_2024"
        row["source_file"] = "ons_mid_2024.txt"

    bundle_root, _ = _write_fixture(tmp_path, mutate_targets=promote)

    with pytest.raises(
        CalibrationBundleValidationError,
        match="cannot be promoted to CALIBRATION",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_normalized_frs_rounded_margin(tmp_path: Path) -> None:
    def normalize(bundle: dict[str, object]) -> None:
        margin = bundle["frs_rounded_margin"]
        assert isinstance(margin, dict)
        margin["published_sum_percent"] = 100
        margin["normalization_applied"] = True

    bundle_root, _ = _write_fixture(tmp_path, mutate_bundle=normalize)

    with pytest.raises(
        CalibrationBundleValidationError,
        match="observed 101 percent sum",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_changed_ons_population_cell(tmp_path: Path) -> None:
    def alter_count(rows: list[dict[str, object]]) -> None:
        rows[0]["population_count"] = 2_821_238

    bundle_root, _ = _write_fixture(tmp_path, mutate_weights=alter_count)

    with pytest.raises(
        CalibrationBundleValidationError,
        match="count differs from the extracted ONS",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_calibration_validation_source_reuse(tmp_path: Path) -> None:
    def make_source_mixed(manifest: dict[str, object]) -> None:
        sources = manifest["sources"]
        assert isinstance(sources, list)
        source = next(
            source
            for source in sources
            if source["source_id"] == "ons_time_use_mar_2024"
        )
        source["evidence_role"] = "MIXED"

    def reuse_source(rows: list[dict[str, object]]) -> None:
        row = next(row for row in rows if row["target_id"] == "ons.sleep.sep_oct_2023")
        row["source_id"] = "ons_time_use_mar_2024"
        row["source_file"] = "ons_time_use_mar_2024.txt"
        row["source_locator"] = "Table_1:distinct_holdout_record"

    bundle_root, _ = _write_fixture(
        tmp_path,
        mutate_manifest=make_source_mixed,
        mutate_targets=reuse_source,
    )

    with pytest.raises(
        CalibrationBundleValidationError,
        match="disjoint source_id values",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_unsafe_raw_cache_path(tmp_path: Path) -> None:
    def escape(manifest: dict[str, object]) -> None:
        manifest["raw_cache_root"] = "../outside"

    bundle_root, _ = _write_fixture(tmp_path, mutate_manifest=escape)

    with pytest.raises(
        CalibrationBundleValidationError,
        match="safe repository-relative POSIX path",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )
