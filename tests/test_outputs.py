from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from microtx_sim.analysis.sensitivity import (
    SensitivityCase,
    run_sensitivity_analysis,
)
from microtx_sim.causal.batch import PolicyBatchSpec, run_policy_batch
from microtx_sim.consumers.decision import DecisionParameters
from microtx_sim.consumers.population import CountryProfile
from microtx_sim.outputs.plots import (
    render_epgc_subsidy_requirement_svg,
    render_harm_distribution_svg,
    render_harm_revenue_frontier_svg,
    render_opportunity_cost_decomposition_svg,
    render_spending_distribution_svg,
    write_harm_distribution_svg,
)
from microtx_sim.outputs.schema import (
    ARTIFACT_FILENAMES,
    EPGC_FINANCING_COLUMNS,
    OPPORTUNITY_DECOMPOSITION_COLUMNS,
    OUTPUT_SCHEMA_VERSION,
    PLAYER_OUTCOME_COLUMNS,
    POLICY_ARTIFACT_FILENAMES,
    SCENARIO_SUMMARY_COLUMNS,
    SCENARIO_SUMMARY_V1_PREFIX_COLUMNS,
    SEED_RESULT_COLUMNS,
    SEED_RESULT_V1_PREFIX_COLUMNS,
    SENSITIVITY_COLUMNS,
    SENSITIVITY_V1_PREFIX_COLUMNS,
)
from microtx_sim.outputs.writers import (
    write_batch_artifacts,
    write_csv_atomic,
    write_json_atomic,
)


class OutputSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = PolicyBatchSpec(
            seeds=(41, 42),
            days=1,
            player_count=4,
            decision_parameters=DecisionParameters(step_minutes=240),
        )
        cls.profiles = (CountryProfile(code="XX"),)
        cls.batch = run_policy_batch(cls.spec, country_profiles=cls.profiles)
        cls.sensitivity = run_sensitivity_analysis(
            cls.spec,
            cases=(
                SensitivityCase(
                    "paid_random_rewards",
                    (0.0, 0.7),
                    expected_direction="increasing",
                ),
            ),
            country_profiles=cls.profiles,
        )

    def test_schema_version_columns_and_filenames_are_stable(self) -> None:
        self.assertEqual(OUTPUT_SCHEMA_VERSION, "2.0")
        self.assertEqual(
            ARTIFACT_FILENAMES,
            (
                "seed_results.csv",
                "scenario_summary.csv",
                "epgc_financing.csv",
                "sensitivity.csv",
                "manifest.json",
            ),
        )
        for columns in (
            SEED_RESULT_COLUMNS,
            SCENARIO_SUMMARY_COLUMNS,
            EPGC_FINANCING_COLUMNS,
            SENSITIVITY_COLUMNS,
            PLAYER_OUTCOME_COLUMNS,
            OPPORTUNITY_DECOMPOSITION_COLUMNS,
        ):
            self.assertTrue(columns)
            self.assertEqual(len(columns), len(set(columns)))
            self.assertTrue(all(isinstance(column, str) and column for column in columns))

    def test_v2_preserves_v1_prefix_and_released_nonempty_order(self) -> None:
        migrations = {
            "seed_results.csv": (
                self.batch.seed_rows(),
                SEED_RESULT_V1_PREFIX_COLUMNS,
                SEED_RESULT_COLUMNS,
            ),
            "scenario_summary.csv": (
                self.batch.scenario_rows(),
                SCENARIO_SUMMARY_V1_PREFIX_COLUMNS,
                SCENARIO_SUMMARY_COLUMNS,
            ),
            "sensitivity.csv": (
                self.sensitivity.rows,
                SENSITIVITY_V1_PREFIX_COLUMNS,
                SENSITIVITY_COLUMNS,
            ),
        }
        for filename, (rows, v1_prefix, v2_columns) in migrations.items():
            emitted_keys = set(rows[0])
            released_nonempty_order = v1_prefix + tuple(
                sorted(emitted_keys.difference(v1_prefix))
            )
            with self.subTest(filename=filename):
                self.assertEqual(v2_columns[: len(v1_prefix)], v1_prefix)
                self.assertEqual(v2_columns, released_nonempty_order)

    def test_v2_schema_fingerprint_is_frozen(self) -> None:
        payload = {
            "version": OUTPUT_SCHEMA_VERSION,
            "artifacts": POLICY_ARTIFACT_FILENAMES,
            "columns": {
                "seed_results.csv": SEED_RESULT_COLUMNS,
                "scenario_summary.csv": SCENARIO_SUMMARY_COLUMNS,
                "epgc_financing.csv": EPGC_FINANCING_COLUMNS,
                "sensitivity.csv": SENSITIVITY_COLUMNS,
                "player_outcomes.csv": PLAYER_OUTCOME_COLUMNS,
                "opportunity_cost_decomposition.csv": (
                    OPPORTUNITY_DECOMPOSITION_COLUMNS
                ),
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            sha256(encoded).hexdigest(),
            "439df3faab6000565ccb8a33ce81acbc69baeb6e3058cb0e67bf3fd1390ffd80",
        )

    def test_emitted_policy_keys_and_headers_exactly_match_v2_schema(self) -> None:
        tables = {
            "seed_results.csv": (self.batch.seed_rows(), SEED_RESULT_COLUMNS),
            "scenario_summary.csv": (
                self.batch.scenario_rows(),
                SCENARIO_SUMMARY_COLUMNS,
            ),
            "epgc_financing.csv": (
                self.batch.epgc_rows(),
                EPGC_FINANCING_COLUMNS,
            ),
            "sensitivity.csv": (self.sensitivity.rows, SENSITIVITY_COLUMNS),
            "player_outcomes.csv": (
                self.batch.player_rows(),
                PLAYER_OUTCOME_COLUMNS,
            ),
            "opportunity_cost_decomposition.csv": (
                self.batch.opportunity_rows(),
                OPPORTUNITY_DECOMPOSITION_COLUMNS,
            ),
        }
        for filename, (rows, columns) in tables.items():
            with self.subTest(filename=filename, contract="keys"):
                self.assertTrue(rows)
                for row in rows:
                    self.assertEqual(set(row), set(columns))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_batch_artifacts(
                root,
                self.batch.seed_rows(),
                self.batch.scenario_rows(),
                self.batch.epgc_rows(),
                self.sensitivity.rows,
                {},
            )
            write_csv_atomic(
                root / "player_outcomes.csv",
                self.batch.player_rows(),
                canonical_columns=PLAYER_OUTCOME_COLUMNS,
                allow_extra_columns=False,
            )
            write_csv_atomic(
                root / "opportunity_cost_decomposition.csv",
                self.batch.opportunity_rows(),
                canonical_columns=OPPORTUNITY_DECOMPOSITION_COLUMNS,
                allow_extra_columns=False,
            )
            for filename, (_, columns) in tables.items():
                with self.subTest(filename=filename, contract="header"):
                    with (root / filename).open(
                        encoding="utf-8", newline=""
                    ) as handle:
                        self.assertEqual(next(csv.reader(handle)), list(columns))


class AtomicWriterTests(unittest.TestCase):
    def test_batch_writes_fixed_names_escaping_and_manifest_schema(self) -> None:
        scenario = 'safe, "quoted"\nline & <tag>'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_batch_artifacts(
                root,
                seed_rows=(
                    {
                        "scenario_id": scenario,
                        "scenario_label": scenario,
                        "seed": 17,
                        "player_count": 0,
                        "mean_harm": 0.0,
                        "cohort_digest": "cohort-é",
                    },
                ),
                summary_rows=({"scenario_id": scenario, "seed_count": 1},),
                epgc_rows=(
                    {
                        "scenario_id": scenario,
                        "minimum_public_contribution_cents": 0,
                    },
                ),
                sensitivity_rows=(
                    {
                        "parameter": "alpha",
                        "parameter_value": 0.0,
                        "scenario_id": scenario,
                    },
                ),
                manifest={"run_id": "batch-é", "nested": {"b": 2, "a": 1}},
            )

            self.assertEqual(
                {path.name for path in paths.values()}, set(ARTIFACT_FILENAMES)
            )
            with (root / "seed_results.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["scenario_id"], scenario)
            self.assertEqual(rows[0]["player_count"], "0")
            self.assertEqual(rows[0]["mean_harm"], "0")
            self.assertEqual(rows[0]["cohort_digest"], "cohort-é")

            manifest = json.loads((root / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["output_schema_version"], OUTPUT_SCHEMA_VERSION)
            self.assertEqual(manifest["artifact_files"], list(ARTIFACT_FILENAMES))
            self.assertEqual(manifest["run_id"], "batch-é")
            self.assertFalse(any(root.glob(".*.tmp")))

    def test_identical_inputs_are_byte_deterministic(self) -> None:
        arguments = dict(
            seed_rows=({"seed": 2, "scenario_id": "z", "mean_harm": 0.0},),
            summary_rows=({"scenario_id": "z", "seed_count": 1},),
            epgc_rows=(),
            sensitivity_rows=(),
            manifest={"seeds": [2], "config": {"beta": 2, "alpha": 1}},
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            write_batch_artifacts(first, **arguments)
            write_batch_artifacts(second, **arguments)
            for filename in ARTIFACT_FILENAMES:
                self.assertEqual(
                    (first / filename).read_bytes(), (second / filename).read_bytes()
                )

    def test_versioned_batch_tables_reject_undeclared_columns(self) -> None:
        arguments = {
            "seed_rows": (),
            "summary_rows": (),
            "epgc_rows": (),
            "sensitivity_rows": (),
            "manifest": {},
        }
        for row_argument in (
            "seed_rows",
            "summary_rows",
            "epgc_rows",
            "sensitivity_rows",
        ):
            with (
                self.subTest(table=row_argument),
                tempfile.TemporaryDirectory() as directory,
            ):
                invalid = dict(arguments)
                invalid[row_argument] = ({"not_in_schema": 1},)
                destination = Path(directory) / "bundle"
                with self.assertRaisesRegex(
                    ValueError,
                    "undeclared columns: not_in_schema",
                ):
                    write_batch_artifacts(destination, **invalid)
                self.assertFalse(destination.exists())

    def test_batch_preflight_leaves_existing_bundle_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            destination.mkdir()
            (destination / "sentinel.txt").write_bytes(b"keep-me")
            (destination / "seed_results.csv").write_bytes(b"old-seed-results")
            before = {
                path.name: path.read_bytes()
                for path in destination.iterdir()
                if path.is_file()
            }
            with self.assertRaisesRegex(
                ValueError,
                "undeclared columns: not_in_schema",
            ):
                write_batch_artifacts(
                    destination,
                    seed_rows=({"scenario_id": "valid"},),
                    summary_rows=({"scenario_id": "valid"},),
                    epgc_rows=({"scenario_id": "valid"},),
                    sensitivity_rows=({"not_in_schema": 1},),
                    manifest={},
                )
            after = {
                path.name: path.read_bytes()
                for path in destination.iterdir()
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_v1_manifest_cannot_describe_a_v2_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            with self.assertRaisesRegex(
                ValueError,
                "output_schema_version conflicts",
            ):
                write_batch_artifacts(
                    destination,
                    (),
                    (),
                    (),
                    (),
                    {"output_schema_version": "1.0"},
                )
            self.assertFalse(destination.exists())

    def test_generic_writer_retains_ad_hoc_extra_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ad_hoc.csv"
            write_csv_atomic(
                path,
                (
                    {
                        "seed": 2,
                        "custom_b": 2,
                        "custom_a": {"z": 1, "a": "é"},
                    },
                ),
                canonical_columns=("seed",),
            )
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                tuple(rows[0]),
                ("seed", "custom_a", "custom_b"),
            )
            self.assertEqual(rows[0]["custom_a"], '{"a":"é","z":1}')

    def test_empty_rows_have_headers_and_zero_values_remain_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_batch_artifacts(root, (), (), (), (), {})
            contracts = {
                "seed_results.csv": SEED_RESULT_COLUMNS,
                "scenario_summary.csv": SCENARIO_SUMMARY_COLUMNS,
                "epgc_financing.csv": EPGC_FINANCING_COLUMNS,
                "sensitivity.csv": SENSITIVITY_COLUMNS,
            }
            for filename, columns in contracts.items():
                with (root / filename).open(encoding="utf-8", newline="") as handle:
                    content = list(csv.reader(handle))
                self.assertEqual(content, [list(columns)])

            write_csv_atomic(
                root / "zero.csv",
                ({"value": 0, "negative_zero": -0.0},),
                canonical_columns=("value", "negative_zero"),
            )
            self.assertEqual(
                (root / "zero.csv").read_text("utf-8"),
                "value,negative_zero\n0,0\n",
            )

    def test_atomic_json_replacement_and_nonfinite_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            write_json_atomic(path, {"value": 1})
            write_json_atomic(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text("utf-8")), {"value": 2})
            with self.assertRaises(ValueError):
                write_json_atomic(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text("utf-8")), {"value": 2})


class SvgPlotTests(unittest.TestCase):
    def test_all_plots_handle_empty_and_zero_inputs_as_valid_svg(self) -> None:
        documents = (
            render_harm_distribution_svg([]),
            render_spending_distribution_svg([0, 0, 0]),
            render_harm_revenue_frontier_svg([]),
            render_opportunity_cost_decomposition_svg(
                ({"component": "sleep", "value": 0},)
            ),
            render_epgc_subsidy_requirement_svg(
                ({"scenario": "EPGC", "minimum_public_contribution_cents": 0},)
            ),
        )
        for document in documents:
            root = ET.fromstring(document)
            self.assertTrue(root.tag.endswith("svg"))
            self.assertEqual(root.attrib["viewBox"], "0 0 800 480")
            self.assertTrue(any(child.tag.endswith("title") for child in root))

    def test_svg_is_deterministic_and_escapes_titles_and_labels(self) -> None:
        title = 'Harm & revenue <test> "quoted"'
        rows = (
            {"scenario": "A & <B>", "producer_revenue_cents": 0, "mean_harm": 0},
            {"scenario": 'C "D"', "producer_revenue_cents": 100, "mean_harm": 0.2},
        )
        first = render_harm_revenue_frontier_svg(rows, title=title)
        second = render_harm_revenue_frontier_svg(rows, title=title)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))
        root = ET.fromstring(first)
        svg_title = next(child for child in root if child.tag.endswith("title"))
        self.assertEqual(svg_title.text, title)
        self.assertIn("A &amp; &lt;B&gt;", first)
        self.assertNotIn("A & <B>", first)

    def test_harm_revenue_plot_marks_only_non_dominated_points(self) -> None:
        document = render_harm_revenue_frontier_svg(
            (
                {"scenario": "dominated", "producer_revenue_cents": 100, "mean_harm": 0.5},
                {"scenario": "efficient-high", "producer_revenue_cents": 100, "mean_harm": 0.3},
                {"scenario": "efficient-low", "producer_revenue_cents": 50, "mean_harm": 0.1},
            )
        )
        root = ET.fromstring(document)
        circles = [node for node in root.iter() if node.tag.endswith("circle")]
        self.assertEqual(len(circles), 3)
        fills_by_label = {
            next(child for child in circle if child.tag.endswith("title"))
            .text.split(":", 1)[0]: circle.attrib["fill"]
            for circle in circles
        }
        self.assertEqual(fills_by_label["dominated"], "#BAB0AC")
        self.assertEqual(fills_by_label["efficient-high"], "#54A24B")
        self.assertEqual(fills_by_label["efficient-low"], "#54A24B")
        self.assertIn("Pareto-efficient frontier", document)

    def test_atomic_svg_writer_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "one.svg"
            second = Path(directory) / "two.svg"
            write_harm_distribution_svg(first, [0.0, 0.25, 0.5], bins=3)
            write_harm_distribution_svg(second, [0.0, 0.25, 0.5], bins=3)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            ET.parse(first)


if __name__ == "__main__":
    unittest.main()
