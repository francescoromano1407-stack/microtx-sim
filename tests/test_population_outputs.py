from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from microtx_sim.data.population_evidence import PopulationEstimandRole
from microtx_sim.metrics.population_estimands import (
    EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
    POPULATION_ESTIMAND_SCHEMA_VERSION,
    ExactPopulationWeights,
    PopulationAnalysisUnit,
    PopulationContrast,
    PopulationEstimandAlgorithm,
    PopulationEstimandResult,
    PopulationEstimandSpec,
    PopulationInclusionField,
    PopulationInclusionRule,
    PopulationInclusionTiming,
    PopulationMetricKind,
    PopulationMetricScale,
    PopulationNormalization,
    PopulationPeriodSemantics,
    weighted_mean,
)
from microtx_sim.outputs.population import write_target_population_estimands
from microtx_sim.outputs.schema import (
    ARTIFACT_FILENAMES,
    POLICY_ARTIFACT_FILENAMES,
    TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES,
    TARGET_POPULATION_ESTIMAND_COLUMNS,
    TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256,
    TARGET_POPULATION_ESTIMAND_SCHEMA_VERSION,
    TARGET_POPULATION_OUTPUT_PROFILE,
    stamp_target_population_estimand_schema,
    target_population_estimand_schema_descriptor,
)


def _weights() -> ExactPopulationWeights:
    return ExactPopulationWeights(
        schema_version=EXACT_POPULATION_WEIGHTS_SCHEMA_VERSION,
        player_ids=(10, 20),
        weight_numerators=(1, 1),
        weight_denominators=(2, 2),
    )


def _spec(
    estimand_id: str,
    *,
    weights: ExactPopulationWeights | None = None,
) -> PopulationEstimandSpec:
    bound_weights = weights or _weights()
    return PopulationEstimandSpec(
        schema_version=POPULATION_ESTIMAND_SCHEMA_VERSION,
        estimand_id=estimand_id,
        target_population_id="target.players.test",
        target_evidence_sha256="a" * 64,
        design_weights_sha256=bound_weights.design_sha256,
        runtime_projection_sha256="b" * 64,
        balance_report_sha256="c" * 64,
        metric_contract_sha256="d" * 64,
        output_profile_id=TARGET_POPULATION_OUTPUT_PROFILE,
        output_profile_schema_sha256=(
            TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256
        ),
        analysis_unit=PopulationAnalysisUnit.PLAYER_PERSON,
        inclusion_rule=PopulationInclusionRule(
            rule_id="all.target.eligible",
            description="All target-eligible players before treatment.",
            source_fields=(
                PopulationInclusionField.AGE_YEARS,
                PopulationInclusionField.JURISDICTION,
            ),
            timing=PopulationInclusionTiming.PRETREATMENT,
            evidence_role=PopulationEstimandRole.CALIBRATION,
        ),
        metric_name="composite_harm",
        metric_kind=PopulationMetricKind.SCORE,
        metric_scale=PopulationMetricScale.NONADDITIVE,
        contrast=PopulationContrast.NONE,
        algorithm=PopulationEstimandAlgorithm.WEIGHTED_MEAN_V1,
        normalization=PopulationNormalization.DIVIDE_BY_WEIGHT_SUM,
        period=PopulationPeriodSemantics(
            period_start=date(2025, 1, 1),
            period_end=date(2025, 1, 31),
            description="One declared simulation horizon.",
        ),
    )


def _pair(
    estimand_id: str,
    outcomes: tuple[object, object],
) -> tuple[PopulationEstimandSpec, PopulationEstimandResult]:
    weights = _weights()
    spec = _spec(estimand_id, weights=weights)
    return spec, weighted_mean(spec, weights, outcomes)


class TargetPopulationOutputSchemaTests(unittest.TestCase):
    def test_profile_descriptor_and_fingerprint_are_frozen(self) -> None:
        descriptor = target_population_estimand_schema_descriptor()
        self.assertEqual(
            descriptor["output_profile"], TARGET_POPULATION_OUTPUT_PROFILE
        )
        self.assertEqual(
            descriptor["output_profile_schema_version"],
            TARGET_POPULATION_ESTIMAND_SCHEMA_VERSION,
        )
        self.assertEqual(
            descriptor["artifact_files"],
            list(TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES),
        )
        self.assertEqual(
            descriptor["table_columns"]["target_population_estimands.csv"],
            list(TARGET_POPULATION_ESTIMAND_COLUMNS),
        )
        encoded = json.dumps(
            descriptor,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.assertEqual(
            sha256(encoded).hexdigest(),
            TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256,
        )
        self.assertEqual(
            TARGET_POPULATION_ESTIMAND_SCHEMA_SHA256,
            "0e5d6c3a8b8334d44cca3b079c7392b542791396c246912d3b0323aaf431e53a",
        )

    def test_profile_does_not_change_legacy_artifact_sets(self) -> None:
        self.assertNotIn(
            "target_population_estimands.csv",
            ARTIFACT_FILENAMES,
        )
        self.assertNotIn(
            "target_population_estimands.csv",
            POLICY_ARTIFACT_FILENAMES,
        )

    def test_stamper_fixes_scope_and_rejects_bundle_or_readiness_claims(self) -> None:
        stamped = stamp_target_population_estimand_schema({"note": "test"})
        self.assertTrue(stamped["synthetic_only"])
        self.assertFalse(stamped["campaign_ready"])
        self.assertFalse(stamped["empirical_validation_claimed"])
        self.assertFalse(stamped["full_output_bundle"])
        self.assertFalse(stamped["manifest_envelope"])
        self.assertFalse(stamped["upstream_digests_independently_resolved"])
        self.assertNotIn("output_schema_version", stamped)

        invalid = (
            {"output_schema_version": "3.0"},
            {"manifest_schema_version": "1.0"},
            {"manifest_sha256": "0" * 64},
            {"manifest": {}},
            {"campaign_ready": True},
            {"synthetic_only": False},
            {"full_output_bundle": True},
            {"output_profile": TARGET_POPULATION_OUTPUT_PROFILE},
            {"output_profile_schema_sha265": "0" * 64},
        )
        for metadata in invalid:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    stamp_target_population_estimand_schema(metadata)


class TargetPopulationOutputWriterTests(unittest.TestCase):
    def test_exact_header_metadata_identity_and_order_are_deterministic(self) -> None:
        first = _pair("population.estimand.a", (0, 1))
        second = _pair("population.estimand.b", (Fraction(1, 3), 1))
        with (
            tempfile.TemporaryDirectory() as left_dir,
            tempfile.TemporaryDirectory() as right_dir,
        ):
            left = Path(left_dir)
            right = Path(right_dir)
            write_target_population_estimands(
                left,
                (second, first),
                metadata={"run_label": "test"},
            )
            write_target_population_estimands(
                right,
                (first, second),
                metadata={"run_label": "test"},
            )

            for filename in TARGET_POPULATION_ESTIMAND_ARTIFACT_FILENAMES:
                self.assertEqual(
                    (left / filename).read_bytes(),
                    (right / filename).read_bytes(),
                )

            with (left / "target_population_estimands.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
                self.assertEqual(
                    tuple(rows[0]), TARGET_POPULATION_ESTIMAND_COLUMNS
                )
            self.assertEqual(
                [row["estimand_id"] for row in rows],
                ["population.estimand.a", "population.estimand.b"],
            )
            self.assertEqual(rows[0]["numerator_decimal"], "1")
            self.assertEqual(rows[0]["denominator_decimal"], "2")
            self.assertEqual(rows[0]["weight_sum_numerator_decimal"], "1")
            self.assertEqual(rows[0]["weight_sum_denominator_decimal"], "1")
            self.assertEqual(rows[0]["player_count_decimal"], "2")

            metadata = json.loads(
                (left / "target_population_estimand_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["record_count_decimal"], "2")
            self.assertEqual(
                metadata["ordered_estimand_ids"],
                ["population.estimand.a", "population.estimand.b"],
            )
            self.assertTrue(metadata["synthetic_only"])
            self.assertFalse(metadata["campaign_ready"])
            self.assertFalse(metadata["upstream_digests_independently_resolved"])
            identities = metadata["upstream_identity_declarations"][0]
            self.assertEqual(identities["target_evidence_sha256"], "a" * 64)
            self.assertEqual(
                identities["design_weights_sha256"],
                first[0].design_weights_sha256,
            )
            self.assertEqual(identities["runtime_projection_sha256"], "b" * 64)
            self.assertEqual(identities["balance_report_sha256"], "c" * 64)
            self.assertEqual(identities["metric_contract_sha256"], "d" * 64)

    def test_incompatible_profile_binding_and_pair_binding_fail_closed(self) -> None:
        pair = _pair("population.estimand.test", (0, 1))
        wrong_schema = replace(
            pair[0], output_profile_schema_sha256="f" * 64
        )
        wrong_result_spec = _spec("population.estimand.other")

        invalid_pairs = (
            ((wrong_schema, pair[1]), ValueError),
            ((wrong_result_spec, pair[1]), ValueError),
        )
        with tempfile.TemporaryDirectory() as directory:
            for invalid_pair, expected in invalid_pairs:
                with self.subTest(invalid_pair=invalid_pair[0].estimand_id):
                    with self.assertRaises(expected):
                        write_target_population_estimands(
                            directory, (invalid_pair,)
                        )

    def test_subclasses_tampering_duplicates_and_order_ambiguity_are_rejected(
        self,
    ) -> None:
        pair = _pair("population.estimand.test", (0, 1))

        class SpecSubclass(PopulationEstimandSpec):
            pass

        class ResultSubclass(PopulationEstimandResult):
            pass

        spec_subclass = SpecSubclass(
            **{
                field: getattr(pair[0], field)
                for field in pair[0].__dataclass_fields__
            }
        )
        result_subclass = ResultSubclass(
            **{
                field: getattr(pair[1], field)
                for field in pair[1].__dataclass_fields__
            }
        )
        tampered = _pair("population.estimand.tampered", (0, 1))
        object.__setattr__(tampered[1], "numerator", 3)

        with tempfile.TemporaryDirectory() as directory:
            invalid = (
                ((spec_subclass, pair[1]), TypeError),
                ((pair[0], result_subclass), TypeError),
                (tampered, ValueError),
            )
            for invalid_pair, expected in invalid:
                with self.subTest(expected=expected.__name__):
                    with self.assertRaises(expected):
                        write_target_population_estimands(directory, (invalid_pair,))

            with self.assertRaisesRegex(ValueError, "duplicate estimand IDs"):
                write_target_population_estimands(
                    directory,
                    (pair, pair),
                )
            with self.assertRaisesRegex(TypeError, "exact two-item tuple"):
                write_target_population_estimands(
                    directory,
                    ([pair[0], pair[1]],),  # type: ignore[arg-type]
                )

    def test_all_preflight_errors_happen_before_either_file_is_replaced(self) -> None:
        pair = _pair("population.estimand.test", (0, 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            estimands_path = root / "target_population_estimands.csv"
            metadata_path = root / "target_population_estimand_metadata.json"
            estimands_path.write_text("old estimands\n", encoding="utf-8")
            metadata_path.write_text("old metadata\n", encoding="utf-8")

            with self.assertRaises(TypeError):
                write_target_population_estimands(
                    root,
                    (pair,),
                    metadata={"late_invalid_value": object()},
                )

            self.assertEqual(
                estimands_path.read_text(encoding="utf-8"), "old estimands\n"
            )
            self.assertEqual(
                metadata_path.read_text(encoding="utf-8"), "old metadata\n"
            )


if __name__ == "__main__":
    unittest.main()
