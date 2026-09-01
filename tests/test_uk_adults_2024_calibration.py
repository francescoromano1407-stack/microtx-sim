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
    DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
    CalibrationBundleValidationError,
    CalibrationBundleVerificationError,
    EstimandRole,
    EvidenceStatus,
    POPULATION_WEIGHT_CSV_COLUMNS,
    TARGET_CSV_COLUMNS,
    UK_ADULTS_2024_POPULATION_COUNT,
    UNSUPPORTED_CONCEPTS,
    _secure_read_regular_file,
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
    runtime_mapping: str = "NOT_CONNECTED",
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
        "runtime_mapping": runtime_mapping,
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
        "ons_time_use_mar_2024_alias.txt": (
            b"ONS March 2024 time-use calibration records.\n"
        ),
        "ons_time_use_sep_oct_2023.txt": b"ONS Sep-Oct 2023 holdout records.\n",
    }
    for relative_path, content in source_contents.items():
        (raw_cache / relative_path).write_bytes(content)

    sources = [
        _source("frs_2023_24", "frs.txt", source_contents["frs.txt"], "MIXED"),
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
    gap_source = next(
        source for source in sources if source["source_id"] == "gap_index"
    )
    gap_source["official_url"] = None
    gap_source["publication_date"] = None
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

    frs_values = (
        ("frs_income_under_200", "5"),
        ("frs_income_200_399", "14"),
        ("frs_income_400_599", "18"),
        ("frs_income_600_799", "14"),
        ("frs_income_800_999", "11"),
        ("frs_income_1000_1199", "9"),
        ("frs_income_1200_1399", "7"),
        ("frs_income_1400_1599", "5"),
        ("frs_income_1600_1799", "4"),
        ("frs_income_1800_1999", "3"),
        ("frs_income_2000_plus", "11"),
    )
    targets = [
        *[
            _target(
                target_id,
                "gross_weekly_household_income_share",
                "CALIBRATION",
                "QUANTIFIED",
                "frs_2023_24",
                "frs.txt",
                f"Table_2_5:{target_id}",
                value=value,
                unit="published_percent",
                runtime_mapping=(
                    "not_connected:household_to_personal_income_bridge"
                ),
            )
            for target_id, value in frs_values
        ],
        _target(
            "frs_income_published_sum",
            "gross_weekly_household_income_share_sum",
            "DIAGNOSTIC",
            "QUANTIFIED",
            "frs_2023_24",
            "frs.txt",
            "Table_2_5:published_percentage_row",
            value="101",
            unit="published_percent",
            runtime_mapping="audit_only",
        ),
        _target(
            "population_18_64_total",
            "population_count",
            "CALIBRATION",
            "QUANTIFIED",
            "ons_mid_2024",
            "ons_mid_2024.txt",
            "MYE2:ages_18_64",
            value="41860587",
            unit="persons",
            runtime_mapping="not_connected:population_projection",
        ),
        _target(
            "population_18_64_female",
            "population_count",
            "CALIBRATION",
            "QUANTIFIED",
            "ons_mid_2024",
            "ons_mid_2024.txt",
            "MYE2:Female_ages_18_64",
            value="21279181",
            unit="persons",
            runtime_mapping="not_connected:sex_unsupported_in_PlayerTable",
        ),
        _target(
            "population_18_64_male",
            "population_count",
            "CALIBRATION",
            "QUANTIFIED",
            "ons_mid_2024",
            "ons_mid_2024.txt",
            "MYE2:Male_ages_18_64",
            value="20581406",
            unit="persons",
            runtime_mapping="not_connected:sex_unsupported_in_PlayerTable",
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
            "target_id": "frs_income_published_sum",
            "published_sum_percent": 101,
            "normalization_applied": False,
        },
        "unsupported_concepts": list(UNSUPPORTED_CONCEPTS),
        "campaign_ready": False,
        "blockers": sorted(
            f"{concept}: unsupported in the synthetic fixture"
            for concept in UNSUPPORTED_CONCEPTS
        ),
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
    assert len(bundle.targets) == 22
    assert len(bundle.population_weights) == 10
    assert sum(weight.population_count for weight in bundle.population_weights) == (
        41_860_587
    )
    assert all(source.retrieved_at is None for source in bundle.sources)
    assert bundle.source_by_id["gap_index"].official_url is None
    assert bundle.source_by_id["gap_index"].publication_date is None
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


def test_raw_source_attestation_reads_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    content = b"x" * (2 * 1024 * 1024 + 17)
    source.write_bytes(content)
    original_open = Path.open
    read_sizes: list[int] = []

    class GuardedReader:
        def __init__(self, stream: object) -> None:
            self._stream = stream

        def __enter__(self) -> GuardedReader:
            return self

        def __exit__(self, *args: object) -> object:
            return self._stream.__exit__(*args)  # type: ignore[attr-defined,no-any-return]

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            assert 0 < size <= 1024 * 1024
            return self._stream.read(size)  # type: ignore[attr-defined,no-any-return]

    def guarded_open(path: Path, *args: object, **kwargs: object) -> GuardedReader:
        return GuardedReader(original_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", guarded_open)

    result = _secure_read_regular_file(
        source,
        maximum_bytes=64 * 1024 * 1024,
        expected_byte_length=len(content),
        expected_sha256=sha256(content).hexdigest(),
        description="streamed fixture",
        return_content=False,
    )

    assert result == b""
    assert read_sizes == [1024 * 1024, 1024 * 1024, 17, 1]


def test_rejects_promotion_of_unsupported_concept(tmp_path: Path) -> None:
    def promote(rows: list[dict[str, object]]) -> None:
        row = next(row for row in rows if row["concept"] == "decision.temperature")
        row["estimand_role"] = "CALIBRATION"
        row["evidence_status"] = "QUANTIFIED"
        row["value"] = "0.35"
        row["unit"] = "percent"
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


def test_rejects_blocker_that_only_mentions_unsupported_concept(
    tmp_path: Path,
) -> None:
    def weaken_prefix(bundle: dict[str, object]) -> None:
        blockers = bundle["blockers"]
        assert isinstance(blockers, list)
        blocker = next(
            item
            for item in blockers
            if str(item).startswith("decision.temperature:")
        )
        blockers.remove(blocker)
        blockers.append(
            "runtime_note: decision.temperature appears here without its exact prefix"
        )
        blockers.sort()

    bundle_root, _ = _write_fixture(tmp_path, mutate_bundle=weaken_prefix)

    with pytest.raises(
        CalibrationBundleValidationError,
        match="must use one exact concept prefix",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_unrealistic_negative_duration(tmp_path: Path) -> None:
    def make_negative(rows: list[dict[str, object]]) -> None:
        row = next(row for row in rows if row["target_id"] == "ons.sleep.mar_2024")
        row["value"] = "-1"
        row["lower_ci"] = "-2"
        row["upper_ci"] = "0"

    bundle_root, _ = _write_fixture(tmp_path, mutate_targets=make_negative)

    with pytest.raises(
        CalibrationBundleValidationError,
        match="outside the plausible range for minutes_per_day",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1" * 65, "exceeds the decimal character limit"),
        ("0." + "1" * 25, "exceeds the decimal precision limit"),
    ],
)
def test_rejects_excessive_decimal_size(
    tmp_path: Path,
    value: str,
    message: str,
) -> None:
    def alter_value(rows: list[dict[str, object]]) -> None:
        row = next(row for row in rows if row["target_id"] == "ons.sleep.mar_2024")
        row["value"] = value
        row["lower_ci"] = ""
        row["upper_ci"] = ""

    bundle_root, _ = _write_fixture(tmp_path, mutate_targets=alter_value)

    with pytest.raises(CalibrationBundleValidationError, match=message):
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


def test_rejects_frs_target_rows_that_do_not_match_published_margin(
    tmp_path: Path,
) -> None:
    def alter_component(rows: list[dict[str, object]]) -> None:
        row = next(row for row in rows if row["target_id"] == "frs_income_under_200")
        row["value"] = "6"

    bundle_root, _ = _write_fixture(tmp_path, mutate_targets=alter_component)

    with pytest.raises(
        CalibrationBundleValidationError,
        match="differs from the published rounded margin",
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


def test_rejects_population_target_that_disagrees_with_weights(tmp_path: Path) -> None:
    def alter_target(rows: list[dict[str, object]]) -> None:
        row = next(
            row for row in rows if row["target_id"] == "population_18_64_total"
        )
        row["value"] = "41860588"

    bundle_root, _ = _write_fixture(tmp_path, mutate_targets=alter_target)

    with pytest.raises(
        CalibrationBundleValidationError,
        match="does not reconcile to population_weights.csv",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_calibration_validation_record_reuse(tmp_path: Path) -> None:
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
        row["source_locator"] = "Table_1:Sleeping_and_resting"

    bundle_root, _ = _write_fixture(
        tmp_path,
        mutate_manifest=make_source_mixed,
        mutate_targets=reuse_source,
    )

    with pytest.raises(
        CalibrationBundleValidationError,
        match="reuse source records",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_source_alias_with_identical_content_fingerprint(
    tmp_path: Path,
) -> None:
    def alias_calibration_bytes(manifest: dict[str, object]) -> None:
        sources = manifest["sources"]
        assert isinstance(sources, list)
        calibration = next(
            source
            for source in sources
            if source["source_id"] == "ons_time_use_mar_2024"
        )
        validation = next(
            source
            for source in sources
            if source["source_id"] == "ons_time_use_sep_oct_2023"
        )
        validation["relative_path"] = "ons_time_use_mar_2024_alias.txt"
        validation["sha256"] = calibration["sha256"]
        validation["byte_length"] = calibration["byte_length"]

    bundle_root, _ = _write_fixture(
        tmp_path,
        mutate_manifest=alias_calibration_bytes,
    )

    with pytest.raises(
        CalibrationBundleValidationError,
        match="repeats a source content fingerprint under aliases",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_excessive_source_count_before_source_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def add_sources(manifest: dict[str, object]) -> None:
        sources = manifest["sources"]
        assert isinstance(sources, list)
        template = dict(sources[0])
        while len(sources) <= 128:
            clone = dict(template)
            clone["source_id"] = f"extra_{len(sources):03d}"
            sources.append(clone)

    bundle_root, _ = _write_fixture(tmp_path, mutate_manifest=add_sources)
    original_read = _secure_read_regular_file

    def reject_source_read(*args: object, **kwargs: object) -> bytes:
        if str(kwargs.get("description", "")).startswith("source "):
            pytest.fail("source I/O occurred before the source-count guard")
        return original_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "microtx_sim.data.calibration._secure_read_regular_file",
        reject_source_read,
    )

    with pytest.raises(
        CalibrationBundleValidationError,
        match="cannot declare more than 128 sources",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_rejects_excessive_declared_source_bytes_before_source_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def inflate_sources(manifest: dict[str, object]) -> None:
        sources = manifest["sources"]
        assert isinstance(sources, list)
        for source in sources[:3]:
            source["byte_length"] = 1024 * 1024 * 1024

    bundle_root, _ = _write_fixture(
        tmp_path,
        mutate_manifest=inflate_sources,
    )
    original_read = _secure_read_regular_file

    def reject_source_read(*args: object, **kwargs: object) -> bytes:
        if str(kwargs.get("description", "")).startswith("source "):
            pytest.fail("source I/O occurred before the aggregate-byte guard")
        return original_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        "microtx_sim.data.calibration._secure_read_regular_file",
        reject_source_read,
    )

    with pytest.raises(
        CalibrationBundleValidationError,
        match="total declared bytes exceed the verification limit",
    ):
        load_uk_adults_2024_calibration_bundle(
            bundle_root,
            repository_root=tmp_path,
        )


def test_allows_predeclared_distinct_records_from_one_mixed_source(
    tmp_path: Path,
) -> None:
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
        row["source_locator"] = "Table_1:predeclared_temporal_holdout"

    bundle_root, _ = _write_fixture(
        tmp_path,
        mutate_manifest=make_source_mixed,
        mutate_targets=reuse_source,
    )

    bundle = load_uk_adults_2024_calibration_bundle(
        bundle_root,
        repository_root=tmp_path,
    )

    assert bundle.target_by_id["ons.sleep.sep_oct_2023"].source_id == (
        "ons_time_use_mar_2024"
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


def test_repository_companion_descriptors_match_committed_bytes() -> None:
    bundle_root = DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH
    declaration = json.loads(
        (bundle_root / "calibration_bundle.json").read_text(encoding="utf-8")
    )

    for filename, descriptor in declaration["files"].items():
        content = (bundle_root / filename).read_bytes()
        assert len(content) == descriptor["byte_length"]
        assert sha256(content).hexdigest() == descriptor["sha256"]


def test_re_attests_repository_bundle_when_local_source_cache_is_available() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    raw_cache = repository_root / "data" / "public_calibration_sources_uk_adults_2024"
    if not raw_cache.is_dir():
        pytest.skip("ignored UK-adults source cache is not available")

    bundle = load_uk_adults_2024_calibration_bundle(
        DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
        repository_root=repository_root,
    )

    assert len(bundle.targets) == 75
    assert len(bundle.sources) == 15
    assert bundle.campaign_ready is False
    assert (
        bundle.target_by_id["time_gaming_mean_march_2024"].estimand_role
        is EstimandRole.DIAGNOSTIC
    )
    assert bundle.target_by_id["open_play_weekly_play_mean"].value == Decimal(
        "1004.2698795181"
    )
    assert bundle.source_by_id["ons_time_use_march_2024"].evidence_role.value == (
        "MIXED"
    )
