"""Tests for build_figure_aggregate.py: schema contract, determinism, and the numerical contract."""

from __future__ import annotations

import csv
import json
import shutil
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.theorem_audit import build_figure_aggregate as builder  # noqa: E402
from experiments.theorem_audit.plot_supportive_evidence import _load_frozen_aggregate  # noqa: E402

CAMPAIGN = builder.DEFAULT_CAMPAIGN
TRACKED = builder.DEFAULT_OUTPUT


def _campaign_available() -> bool:
    return (CAMPAIGN / "results" / "broad" / "findings.json").is_file() and (
        CAMPAIGN / "results" / "controls_compatible" / "runs.csv"
    ).is_file()


@unittest.skipUnless(_campaign_available(), "frozen campaign aggregates are not present")
class BuildFigureAggregateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.record = builder.build(CAMPAIGN)
        cls.rows, _ = builder.load_control_rows(CAMPAIGN)
        cls.runs = builder.load_training_runs(CAMPAIGN)

    def test_record_passes_the_plot_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            path.write_text(builder.serialize(self.record), encoding="utf-8")
            loaded = _load_frozen_aggregate(path)
        self.assertEqual(loaded["raw_source"]["rows"], 1080)

    def test_build_is_deterministic(self) -> None:
        self.assertEqual(builder.serialize(self.record), builder.serialize(builder.build(CAMPAIGN)))

    def test_tracked_file_matches_rebuild(self) -> None:
        self.assertEqual(TRACKED.read_text(encoding="utf-8"), builder.serialize(self.record))

    def test_width_diagnostics_are_pooled_medians(self) -> None:
        panel = self.record["panels"]["width_diagnostics"]
        for width in builder.WIDTHS:
            rows = [run for run in self.runs if int(run["width"]) == width]
            self.assertEqual(len(rows), 110)
            cell = panel[str(width)]
            self.assertEqual(cell["gap_retained_denominator"], 110)
            self.assertEqual(
                cell["gap_retained_at_saved_checkpoints"],
                sum(bool(run["gap_preserved_through_training"]) for run in rows),
            )
            self.assertEqual(
                cell["median_initial_relative_kernel_error"],
                statistics.median(float(run["initial_reference_error_operator"]) for run in rows),
            )
            self.assertEqual(
                cell["final_frozen_trajectory_discrepancy"],
                statistics.median(float(run["final_relative_error_to_frozen_dynamics"]) for run in rows),
            )
        self.assertEqual(
            [panel[str(width)]["gap_retained_at_saved_checkpoints"] for width in builder.WIDTHS],
            [17, 110, 110, 110],
        )

    def test_conditioning_cells_are_five_seed_medians_of_contrast_runs(self) -> None:
        panel = self.record["panels"]["conditioning_width_16384"]
        for family in builder.EXPECTED_FAMILIES:
            for epsilon in builder.EPSILONS:
                cell = panel[family][f"{epsilon:g}"]
                rows = [
                    row
                    for row in self.rows
                    if row["family"] == family
                    and row["width"] == "16384"
                    and row["target_mode"] == "contrast"
                    and row["epsilon"] not in ("", "None")
                    and float(row["epsilon"]) == epsilon
                ]
                self.assertEqual(len(rows), 5)
                ratios = sorted(float(row["final_to_initial_loss_ratio"]) for row in rows)
                self.assertEqual(cell["median_final_to_initial_loss_ratio"], ratios[2])
                gaps = sorted(float(row["lambda_min"]) / float(row["lambda_max"]) for row in rows)
                self.assertEqual(cell["median_relative_gap"], gaps[2])
                self.assertAlmostEqual(cell["median_relative_gap"] * cell["median_initial_condition_number"], 1.0, places=12)

    def test_supporting_counts(self) -> None:
        counts = self.record["supporting_counts"]
        self.assertEqual(counts["positive_definite"]["runs"], 960)
        self.assertEqual(counts["positive_definite"]["loss_reduced"], 959)
        self.assertEqual(counts["positive_definite"]["manufactured_loss_reduced_at_least_10x"], 480)
        self.assertEqual(counts["compatible_singular"]["runs"], 120)
        self.assertEqual(counts["compatible_singular"]["final_loss_below_1e_minus_12"], 118)
        self.assertEqual(counts["compatible_singular"]["width_ge_256_final_loss_below_1e_minus_12"], 90)
        singular = self.record["panels"]["compatible_singular"]
        self.assertEqual(sum(singular[str(width)]["runs"] for width in builder.WIDTHS), 120)
        self.assertEqual(
            sum(singular[str(width)]["final_loss_below_1e_minus_12"] for width in builder.WIDTHS),
            counts["compatible_singular"]["final_loss_below_1e_minus_12"],
        )

    def test_raw_source_hash_is_the_table_hash(self) -> None:
        table = CAMPAIGN / "results" / "controls_compatible" / "runs.csv"
        self.assertEqual(self.record["raw_source"]["sha256"], builder._sha256(table))

    def test_short_table_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "campaign"
            shutil.copytree(CAMPAIGN / "results" / "controls_compatible", copy / "results" / "controls_compatible")
            shutil.copytree(CAMPAIGN / "results" / "broad", copy / "results" / "broad", ignore=shutil.ignore_patterns("figures", "tables"))
            for name in ("broad.json", "controls.json"):
                shutil.copy(CAMPAIGN / name, copy / name)
            table = copy / "results" / "controls_compatible" / "runs.csv"
            with table.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                fieldnames = list(reader.fieldnames or [])
            with table.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows[:-1])
            with self.assertRaises(ValueError):
                builder.build(copy)

    def test_incomplete_findings_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "campaign"
            shutil.copytree(CAMPAIGN / "results" / "controls_compatible", copy / "results" / "controls_compatible")
            (copy / "results" / "broad").mkdir(parents=True)
            findings = json.loads((CAMPAIGN / "results" / "broad" / "findings.json").read_text(encoding="utf-8"))
            findings["runs"][0]["status"] = "missing"
            findings["coverage"]["completed"] = 769
            (copy / "results" / "broad" / "findings.json").write_text(json.dumps(findings), encoding="utf-8")
            for name in ("broad.json", "controls.json"):
                shutil.copy(CAMPAIGN / name, copy / name)
            with self.assertRaises(ValueError):
                builder.build(copy)


if __name__ == "__main__":
    unittest.main()
