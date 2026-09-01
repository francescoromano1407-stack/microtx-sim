"""Verify the partial UK-adults-2024 calibration without running simulations."""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path

from microtx_sim.data.calibration import (
    DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
    CalibrationBundleValidationError,
    CalibrationTarget,
    UKAdults2024CalibrationBundle,
    load_uk_adults_2024_calibration_bundle,
)


def _target_by_id(
    bundle: UKAdults2024CalibrationBundle,
    target_id: str,
) -> CalibrationTarget:
    try:
        return bundle.target_by_id[target_id]
    except KeyError as exc:
        raise RuntimeError(f"expected exactly one target named {target_id}") from exc


def _required_value(
    bundle: UKAdults2024CalibrationBundle,
    target_id: str,
) -> Decimal:
    target = _target_by_id(bundle, target_id)
    if target.value is None:
        raise RuntimeError(f"target {target_id} has no quantified value")
    return target.value


def _inside_interval(value: Decimal, target: CalibrationTarget) -> bool:
    if target.lower_ci is None or target.upper_ci is None:
        raise RuntimeError(f"target {target.target_id} has no complete interval")
    return target.lower_ci <= value <= target.upper_ci


def build_diagnostics(bundle: UKAdults2024CalibrationBundle) -> dict[str, object]:
    """Return reproducible compatibility diagnostics for the verified bundle."""

    population_total = sum(
        weight.population_count for weight in bundle.population_weights
    )
    female_total = sum(
        weight.population_count
        for weight in bundle.population_weights
        if weight.sex == "FEMALE"
    )
    male_total = sum(
        weight.population_count
        for weight in bundle.population_weights
        if weight.sex == "MALE"
    )

    march_sleep = _target_by_id(bundle, "time_sleeping_mean_march_2024")
    holdout_sleep = _required_value(bundle, "holdout_sleeping_mean_sep_oct_2023")
    march_gaming = _target_by_id(bundle, "time_gaming_mean_march_2024")
    holdout_gaming = _required_value(bundle, "holdout_gaming_mean_sep_oct_2023")
    open_play_weekly = _required_value(bundle, "open_play_weekly_play_mean")
    ofcom_weekly = _required_value(bundle, "ofcom_gamer_weekly_play_mean")
    open_play_psqi = _required_value(
        bundle,
        "open_play_psqi_first_complete_sleep_mean",
    )
    ons_sleeping = _required_value(bundle, "time_sleeping_mean_march_2024")

    return {
        "schema_version": 1,
        "bundle_id": bundle.bundle_id,
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_status": bundle.status,
        "bundle_verification": "PASSED",
        "campaign_ready": bundle.campaign_ready,
        "simulation_execution_performed": False,
        "checks": {
            "population_age_sex_reconciliation": {
                "status": "PASSED",
                "persons_18_64": population_total,
                "female": female_total,
                "male": male_total,
                "female_plus_male": female_total + male_total,
            },
            "frs_published_rounding": {
                "status": "PASSED_RETAINED_AS_PUBLISHED",
                "published_sum_percent": str(
                    _required_value(bundle, "frs_income_published_sum")
                ),
                "normalization_applied": False,
            },
            "temporal_holdout_point_checks": {
                "status": "DIAGNOSTIC_ONLY",
                "sleeping_holdout_point_inside_march_95_ci": _inside_interval(
                    holdout_sleep,
                    march_sleep,
                ),
                "gaming_holdout_point_inside_march_95_ci": _inside_interval(
                    holdout_gaming,
                    march_gaming,
                ),
                "limitation": (
                    "A holdout point falling inside a calibration confidence interval "
                    "is not a formal predictive-coverage test."
                ),
            },
            "open_play_population_transport": {
                "status": "EXCLUDED_FROM_POPULATION_LEVEL_CALIBRATION",
                "open_play_to_ofcom_weekly_mean_ratio": str(
                    open_play_weekly / ofcom_weekly
                ),
                "open_play_mean_minutes_per_week": str(open_play_weekly),
                "ofcom_mean_minutes_per_week": str(ofcom_weekly),
                "limitation": (
                    "Open Play is a selected UK age-18-40 trace-sharing sample; "
                    "Ofcom is a gamer-only age-16-64 third-party self-report."
                ),
            },
            "sleep_construct_comparison": {
                "status": "INCOMPARABLE_DIAGNOSTIC",
                "open_play_first_psqi_minus_ons_sleeping_minutes": str(
                    open_play_psqi - ons_sleeping
                ),
                "limitation": (
                    "PSQI self-reported sleep in selected gamers and ONS all-adult "
                    "primary-activity sleeping have different samples and measures."
                ),
            },
            "open_play_uk_timeuse_provenance": {
                "status": "BLOCKED_AND_EXCLUDED",
                "rows": str(
                    _required_value(bundle, "open_play_uk_timeuse_rows_excluded")
                ),
                "limitation": (
                    "UK-linked rows conflict with the manuscript's US-only diary "
                    "design and cannot be used until provenance is resolved."
                ),
            },
        },
        "blockers": list(bundle.blockers),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-attest the UK-adults-2024 evidence bundle and print diagnostics; "
            "no scenario or campaign is executed."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_UK_ADULTS_2024_CALIBRATION_PATH,
        help="Bundle directory or calibration_bundle.json path.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help="Required only when validating a bundle outside this checkout.",
    )
    parser.add_argument(
        "--campaign-gate",
        action="store_true",
        help="Also enforce campaign readiness; schema v1 intentionally fails.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    bundle = load_uk_adults_2024_calibration_bundle(
        args.bundle,
        repository_root=args.repository_root,
    )
    if args.campaign_gate:
        try:
            bundle.validate_for_campaign()
        except CalibrationBundleValidationError as exc:
            print(
                json.dumps(
                    {
                        "campaign_gate": "FAILED_CLOSED",
                        "error": str(exc),
                        "simulation_execution_performed": False,
                    },
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            raise SystemExit(1) from None
    print(
        json.dumps(
            build_diagnostics(bundle),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
