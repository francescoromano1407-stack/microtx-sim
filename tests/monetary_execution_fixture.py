"""Reusable registered test-only inputs for monetary output execution."""

from __future__ import annotations

from datetime import date
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from microtx_sim.data.monetary_execution import MONETARY_OUTPUT_AGGREGATION_UNIT
from microtx_sim.data.profiles import ProfileBundle, load_profile_bundle
from microtx_sim.data.rate_evidence import (
    RateEvidenceMethod,
    exact_csv_rational_recipe_json,
)


ROOT = Path(__file__).resolve().parents[1]
JURISDICTIONS = ROOT / "configs" / "jurisdictions.toml"
SOURCES = ROOT / "data" / "provenance" / "sources.toml"

SOURCE_ID = "TEST_ONLY_MONETARY_EXECUTION"
RETRIEVED_ON = date(2026, 8, 24)
PERIOD_START = date(2025, 1, 1)
PERIOD_END = date(2025, 12, 31)
DEFAULT_TARGET_PER_SIMULATION = {
    "UK": Fraction(1, 2),
    "KR": Fraction(3, 2),
    "JP": Fraction(2, 1),
    "BE": Fraction(5, 2),
}
_CSV_HEADER = (
    "source_id,jurisdiction_code,source_currency,target_currency,method,"
    "rate_period_start,rate_period_end,retrieved_on,rate_numerator,"
    "rate_denominator\n"
)


def write_monetary_execution_fixture(
    root: Path,
    *,
    target_per_simulation: Mapping[str, Fraction] | None = None,
    rounding_scope: str = "PER_OBSERVATION",
    aggregation_unit: str = MONETARY_OUTPUT_AGGREGATION_UNIT,
) -> tuple[ProfileBundle, Path]:
    """Write a registered algebraic fixture, never an empirical rate claim."""

    target_ratios = dict(
        DEFAULT_TARGET_PER_SIMULATION
        if target_per_simulation is None
        else target_per_simulation
    )
    if set(target_ratios) != {"UK", "KR", "JP", "BE"} or any(
        type(value) is not Fraction or value <= 0
        for value in target_ratios.values()
    ):
        raise ValueError(
            "target_per_simulation must exactly cover UK, KR, JP, and BE "
            "with positive Fraction values"
        )
    sources_text = SOURCES.read_text(encoding="utf-8") + f'''

[[source]]
id = "{SOURCE_ID}"
publisher = "Test fixture"
title = "Content-addressed test-only monetary execution rates"
url = "https://example.invalid/monetary-execution-rates"
period = "2025-01-01/2025-12-31"
geography = "Test jurisdictions"
supports = ["foreign_exchange_rate", "median_equivalised_disposable_income"]
calibration_status = "CALIBRATED"
'''
    sources_path = root / "sources.toml"
    sources_path.write_text(sources_text, encoding="utf-8", newline="")
    source_registry_sha256 = sha256(sources_path.read_bytes()).hexdigest()

    base = load_profile_bundle(
        JURISDICTIONS,
        sources_path,
        source_bundle_path=None,
    )
    scales = {scale.jurisdiction_code: scale for scale in base.money_scales}
    currencies = {code: scale.currency for code, scale in scales.items()}
    rate_ratios = {
        code: scale.currency_scale_to_sim * target_ratios[code]
        for code, scale in scales.items()
    }
    binding_ids = {
        code: f"{code.lower()}-{currencies[code].lower()}-tst-execution-2025"
        for code in currencies
    }

    rows = []
    for code in sorted(currencies):
        ratio = rate_ratios[code]
        rows.append(
            f"{SOURCE_ID},{code},{currencies[code]},TST,FX,"
            "2025-01-01,2025-12-31,2026-08-24,"
            f"{ratio.numerator},{ratio.denominator}\n"
        )
    artifact = (_CSV_HEADER + "".join(rows)).encode("utf-8")
    artifact_root = root / "rate_artifacts"
    artifact_root.mkdir()
    artifact_path = artifact_root / "rates.csv"
    artifact_path.write_bytes(artifact)

    binding_tables = []
    for code in sorted(currencies, key=lambda item: binding_ids[item]):
        ratio = rate_ratios[code]
        recipe = exact_csv_rational_recipe_json(
            source_id=SOURCE_ID,
            jurisdiction_code=code,
            source_currency=currencies[code],
            target_currency="TST",
            method=RateEvidenceMethod.FX,
            rate_period_start=PERIOD_START,
            rate_period_end=PERIOD_END,
            retrieved_on=RETRIEVED_ON,
        )
        binding_tables.append(
            f'''
[[bindings]]
binding_id = "{binding_ids[code]}"
artifact_id = "test-rates"
source_id = "{SOURCE_ID}"
jurisdiction_code = "{code}"
source_currency = "{currencies[code]}"
target_currency = "TST"
method = "FX"
rate_period_start = 2025-01-01
rate_period_end = 2025-12-31
retrieved_on = 2026-08-24
rate_numerator = {ratio.numerator}
rate_denominator = {ratio.denominator}
recipe_json = {json.dumps(recipe)}
'''
        )
    source_bundle_text = f'''schema_version = 1
bundle_id = "test-monetary-output-execution"
provenance_status = "CALIBRATED"
source_registry_sha256 = "{source_registry_sha256}"
artifact_root = "rate_artifacts"
notes = "Test-only exact execution fixture; not an empirical rate choice."

[[artifacts]]
artifact_id = "test-rates"
relative_path = "rates.csv"
media_type = "text/csv"
sha256 = "{sha256(artifact).hexdigest()}"
byte_length = {len(artifact)}
{''.join(binding_tables)}
[signature]
status = "MISSING"
algorithm = "NONE"
key_id = ""
value = ""
'''
    source_bundle_path = root / "source_bundle.toml"
    source_bundle_path.write_text(
        source_bundle_text,
        encoding="utf-8",
        newline="",
    )

    jurisdiction_text = JURISDICTIONS.read_text(encoding="utf-8")
    jurisdiction_text = jurisdiction_text.replace(
        "schema_version = 2",
        "schema_version = 3",
        1,
    )
    jurisdiction_text = jurisdiction_text.replace(
        "median_equivalised_disposable_income_minor_units = 300000\n"
        'income_status = "ILLUSTRATIVE"',
        "median_equivalised_disposable_income_minor_units = 300000\n"
        f'income_source = "{SOURCE_ID}"\n'
        'income_status = "ILLUSTRATIVE"',
        1,
    )
    jurisdiction_text = jurisdiction_text.replace(
        'income_status = "ANCHORED"',
        'income_status = "CALIBRATED"',
    ).replace(
        'income_status = "ILLUSTRATIVE"',
        'income_status = "CALIBRATED"',
    ).replace(
        'income_status = "CALIBRATED"',
        'income_status = "CALIBRATED"\n'
        'currency_scale_status = "CALIBRATED"',
    )
    conversion_tables = []
    for code in currencies:
        ratio = rate_ratios[code]
        conversion_tables.append(
            f'''
[[monetary_conversion]]
conversion_id = "{code.lower()}-to-tst-execution-2025"
rate_binding_id = "{binding_ids[code]}"
jurisdiction_code = "{code}"
source_currency = "{currencies[code]}"
target_currency = "TST"
method = "FX"
rate_numerator = {ratio.numerator}
rate_denominator = {ratio.denominator}
rate_period_start = "2025-01-01"
rate_period_end = "2025-12-31"
target_price_period_start = "2025-01-01"
target_price_period_end = "2025-12-31"
estimand = "test-only target-currency-equivalent player outcome"
population_base = "test-only common player population"
comparison_group = "test-only monetary execution basis"
rounding_method = "nearest_minor_unit_half_away_from_zero"
rounding_scope = "{rounding_scope}"
aggregation_unit = "{aggregation_unit}"
status = "CALIBRATED"
source_ids = ["{SOURCE_ID}"]
retrieved_on = "2026-08-24"
notes = "Model-scale conversion fixture; not recovered local money."
'''
        )
    jurisdictions_path = root / "jurisdictions.toml"
    jurisdictions_path.write_text(
        jurisdiction_text + "".join(conversion_tables),
        encoding="utf-8",
        newline="",
    )
    return (
        load_profile_bundle(
            jurisdictions_path,
            sources_path,
            source_bundle_path=source_bundle_path,
        ),
        artifact_path,
    )


__all__ = [
    "DEFAULT_TARGET_PER_SIMULATION",
    "JURISDICTIONS",
    "PERIOD_END",
    "PERIOD_START",
    "SOURCES",
    "write_monetary_execution_fixture",
]
