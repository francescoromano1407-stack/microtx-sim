from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

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
    OUTPUT_SCHEMA_VERSION,
    SCENARIO_SUMMARY_COLUMNS,
    SEED_RESULT_COLUMNS,
    SENSITIVITY_COLUMNS,
)
from microtx_sim.outputs.writers import (
    write_batch_artifacts,
    write_csv_atomic,
    write_json_atomic,
)


class OutputSchemaTests(unittest.TestCase):
    def test_schema_version_columns_and_filenames_are_stable(self) -> None:
        self.assertEqual(OUTPUT_SCHEMA_VERSION, "1.0")
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
        ):
            self.assertTrue(columns)
            self.assertEqual(len(columns), len(set(columns)))
            self.assertTrue(all(isinstance(column, str) and column for column in columns))


class AtomicWriterTests(unittest.TestCase):
    def test_batch_writes_fixed_names_escaping_and_manifest_schema(self) -> None:
        scenario = 'safe, "quoted"\nline & <tag>'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = write_batch_artifacts(
                root,
                seed_rows=(
                    {
                        "scenario": scenario,
                        "seed": 17,
                        "players": 0,
                        "mean_harm": 0.0,
                        "extra_payload": {"z": 1, "a": "é"},
                    },
                ),
                summary_rows=({"scenario": scenario, "seed_count": 1},),
                epgc_rows=(
                    {
                        "scenario": scenario,
                        "minimum_public_contribution_cents": 0,
                    },
                ),
                sensitivity_rows=(
                    {
                        "parameter": "alpha",
                        "parameter_value": 0.0,
                        "scenario": scenario,
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
            self.assertEqual(rows[0]["scenario"], scenario)
            self.assertEqual(rows[0]["players"], "0")
            self.assertEqual(rows[0]["mean_harm"], "0")
            self.assertEqual(rows[0]["extra_payload"], '{"a":"é","z":1}')

            manifest = json.loads((root / "manifest.json").read_text("utf-8"))
            self.assertEqual(manifest["output_schema_version"], OUTPUT_SCHEMA_VERSION)
            self.assertEqual(manifest["artifact_files"], list(ARTIFACT_FILENAMES))
            self.assertEqual(manifest["run_id"], "batch-é")
            self.assertFalse(any(root.glob(".*.tmp")))

    def test_identical_inputs_are_byte_deterministic(self) -> None:
        arguments = dict(
            seed_rows=({"seed": 2, "scenario": "z", "custom_b": 2, "custom_a": 1},),
            summary_rows=({"scenario": "z", "seed_count": 1},),
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
