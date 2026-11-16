"""Build the tracked figure aggregate of the finite-width GD evidence figure from a campaign.

The manuscript's two-panel figure (``plot_supportive_evidence.py``) reads
``data/compatible_rank_convergence_conditioning.json``. This script derives
every number in that file from the frozen campaign directory, so the figure is
a function of the campaign records and not of a transcription:

* panel (a), the finite-width diagnostics per width, from
  ``results/broad/findings.json`` (770 completed training runs);
* the compatible-singular counts and the width-16,384 conditioning medians of
  panel (b), from ``results/controls_compatible/runs.csv`` (1,080 compatible
  two-row controls) and its ``aggregate.json`` coverage record;
* the supporting counts quoted in the experiments appendix.

Usage::

    uv run python experiments/theorem_audit/build_figure_aggregate.py
    uv run python experiments/theorem_audit/build_figure_aggregate.py --check

``--check`` rebuilds the record in memory and exits with status 1 if it differs
from the tracked file. The output contains no timestamps, so a rebuild from the
same campaign is byte-identical.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.theorem_audit.plot_supportive_evidence import (  # noqa: E402
    EPSILONS,
    EXPECTED_FAMILIES,
    WIDTHS,
    _load_frozen_aggregate,
)

DEFAULT_CAMPAIGN = ROOT / "experiments" / "theorem_audit" / "results" / "campaign_cluster_20261115"
DEFAULT_OUTPUT = ROOT / "experiments" / "theorem_audit" / "data" / "compatible_rank_convergence_conditioning.json"

EXPECTED_TRAINING_RUNS = 770
EXPECTED_PROBLEMS = 11
EXPECTED_SEEDS = 10
EXPECTED_CONTROL_ROWS = 1080
RUNS_PER_WIDTH = EXPECTED_PROBLEMS * EXPECTED_SEEDS
SINGULAR_RUNS_PER_WIDTH = 30
SEEDS_PER_CONDITIONING_CELL = 5
DUPLICATE_ROW_DESIGN = "epsilon_0"
CONTRAST_TARGET = "contrast"
MANUFACTURED_TARGET = "manufactured"
TENFOLD = 0.1
HUNDREDFOLD = 1.0e-2
LOSS_THRESHOLD = 1.0e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("median of an empty list")
    return float(statistics.median(values))


def load_training_runs(campaign: Path) -> list[dict[str, Any]]:
    """Return the 770 completed training rows of ``results/broad/findings.json`` after validation."""
    findings = json.loads((campaign / "results" / "broad" / "findings.json").read_text(encoding="utf-8"))
    coverage = findings.get("coverage", {})
    runs = findings.get("runs", [])
    if coverage.get("expected") != EXPECTED_TRAINING_RUNS or coverage.get("completed") != EXPECTED_TRAINING_RUNS:
        raise ValueError(f"findings.json coverage is {coverage}, expected {EXPECTED_TRAINING_RUNS} completed of {EXPECTED_TRAINING_RUNS}")
    if findings.get("issues"):
        raise ValueError(f"findings.json records {len(findings['issues'])} issues")
    if len(runs) != EXPECTED_TRAINING_RUNS or any(run.get("status") != "complete" for run in runs):
        raise ValueError("findings.json must hold 770 complete training rows")
    problems = {run["problem"] for run in runs}
    widths = sorted({int(run["width"]) for run in runs})
    seeds = {int(run["seed"]) for run in runs}
    if len(problems) != EXPECTED_PROBLEMS or len(seeds) != EXPECTED_SEEDS or not set(WIDTHS) <= set(widths):
        raise ValueError(f"unexpected training grid: {len(problems)} problems, widths {widths}, {len(seeds)} seeds")
    for key in (
        "initial_reference_error_operator",
        "max_kernel_drift_operator",
        "final_relative_error_to_frozen_dynamics",
        "gap_preserved_through_training",
    ):
        if any(run.get(key) is None for run in runs):
            raise ValueError(f"training row without {key}")
    return runs


def load_control_rows(campaign: Path) -> tuple[list[dict[str, str]], Path]:
    """Return the 1,080 compatible control rows and the path of the validated table."""
    directory = campaign / "results" / "controls_compatible"
    table = directory / "runs.csv"
    aggregate = json.loads((directory / "aggregate.json").read_text(encoding="utf-8"))
    if not aggregate.get("coverage_complete") or aggregate.get("completed_runs") != EXPECTED_CONTROL_ROWS:
        raise ValueError(f"controls_compatible/aggregate.json does not record {EXPECTED_CONTROL_ROWS} completed runs")
    if not aggregate.get("target_functionally_compatible_only"):
        raise ValueError("controls_compatible/aggregate.json is not restricted to compatible targets")
    with table.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EXPECTED_CONTROL_ROWS:
        raise ValueError(f"{table} has {len(rows)} rows, expected {EXPECTED_CONTROL_ROWS}")
    if any(row.get("target_functionally_compatible") != "True" for row in rows):
        raise ValueError("every compatible control row must have target_functionally_compatible=True")
    return rows, table


def width_diagnostics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    panel: dict[str, Any] = {
        "provenance": {
            "description": (
                "Pooled medians over the 110 completed training runs per width in results/broad/findings.json "
                "of the frozen campaign; displayed as table completed_width_results of the manuscript"
            ),
            "pde_systems": EXPECTED_PROBLEMS,
            "runs_per_width": RUNS_PER_WIDTH,
        }
    }
    for width in WIDTHS:
        rows = [run for run in runs if int(run["width"]) == width]
        if len(rows) != RUNS_PER_WIDTH:
            raise ValueError(f"expected {RUNS_PER_WIDTH} training runs at width {width}, found {len(rows)}")
        panel[str(width)] = {
            "median_initial_relative_kernel_error": _median([float(r["initial_reference_error_operator"]) for r in rows]),
            "maximum_relative_kernel_drift": _median([float(r["max_kernel_drift_operator"]) for r in rows]),
            "final_frozen_trajectory_discrepancy": _median([float(r["final_relative_error_to_frozen_dynamics"]) for r in rows]),
            "gap_retained_at_saved_checkpoints": sum(bool(r["gap_preserved_through_training"]) for r in rows),
            "gap_retained_denominator": RUNS_PER_WIDTH,
        }
    return panel


def compatible_singular(rows: list[dict[str, str]]) -> dict[str, Any]:
    singular = [row for row in rows if row["row_design"] == DUPLICATE_ROW_DESIGN]
    panel: dict[str, Any] = {}
    for width in WIDTHS:
        cell = [row for row in singular if int(row["width"]) == width]
        if len(cell) != SINGULAR_RUNS_PER_WIDTH:
            raise ValueError(f"expected {SINGULAR_RUNS_PER_WIDTH} duplicate-row runs at width {width}, found {len(cell)}")
        final = [float(row["final_loss"]) for row in cell]
        ratios = [float(row["final_to_initial_loss_ratio"]) for row in cell]
        panel[str(width)] = {
            "runs": len(cell),
            "loss_reduced_at_least_100x": sum(ratio <= HUNDREDFOLD for ratio in ratios),
            "final_loss_below_1e_minus_12": sum(value < LOSS_THRESHOLD for value in final),
            "median_final_loss": _median(final),
            "maximum_final_loss": max(final),
        }
    return panel


def conditioning_width_16384(rows: list[dict[str, str]]) -> dict[str, Any]:
    panel: dict[str, Any] = {}
    for family in EXPECTED_FAMILIES:
        family_panel: dict[str, Any] = {}
        for epsilon in EPSILONS:
            cell = [
                row
                for row in rows
                if row["family"] == family
                and int(row["width"]) == 16384
                and row["target_mode"] == CONTRAST_TARGET
                and row["epsilon"] not in ("", "None")
                and float(row["epsilon"]) == epsilon
            ]
            if len(cell) != SEEDS_PER_CONDITIONING_CELL:
                raise ValueError(f"expected {SEEDS_PER_CONDITIONING_CELL} contrast runs for {family}, epsilon={epsilon:g}, found {len(cell)}")
            gaps = [float(row["lambda_min"]) / float(row["lambda_max"]) for row in cell]
            family_panel[f"{epsilon:g}"] = {
                "runs": len(cell),
                "median_relative_gap": _median(gaps),
                "median_initial_condition_number": _median([1.0 / gap for gap in gaps]),
                "median_final_to_initial_loss_ratio": _median([float(row["final_to_initial_loss_ratio"]) for row in cell]),
            }
        panel[family] = family_panel
    return panel


def supporting_counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    positive = [row for row in rows if row["row_design"] != DUPLICATE_ROW_DESIGN]
    singular = [row for row in rows if row["row_design"] == DUPLICATE_ROW_DESIGN]
    ratio = lambda row: float(row["final_to_initial_loss_ratio"])  # noqa: E731
    final = lambda row: float(row["final_loss"])  # noqa: E731
    wide = [row for row in positive if int(row["width"]) >= 256]
    widest = [row for row in positive if int(row["width"]) == 16384]
    manufactured = [row for row in positive if row["target_mode"] == MANUFACTURED_TARGET]
    singular_wide = [row for row in singular if int(row["width"]) >= 256]
    return {
        "positive_definite": {
            "runs": len(positive),
            "loss_reduced": sum(ratio(row) < 1.0 for row in positive),
            "manufactured_runs": len(manufactured),
            "manufactured_loss_reduced_at_least_10x": sum(ratio(row) <= TENFOLD for row in manufactured),
            "width_ge_256_runs": len(wide),
            "width_ge_256_loss_reduced": sum(ratio(row) < 1.0 for row in wide),
            "width_16384_runs": len(widest),
            "width_16384_loss_reduced": sum(ratio(row) < 1.0 for row in widest),
            "width_16384_median_final_to_initial_loss_ratio": _median([ratio(row) for row in widest]),
        },
        "compatible_singular": {
            "runs": len(singular),
            "loss_reduced_at_least_100x": sum(ratio(row) <= HUNDREDFOLD for row in singular),
            "final_loss_below_1e_minus_12": sum(final(row) < LOSS_THRESHOLD for row in singular),
            "width_ge_256_runs": len(singular_wide),
            "width_ge_256_final_loss_below_1e_minus_12": sum(final(row) < LOSS_THRESHOLD for row in singular_wide),
        },
    }


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build(campaign: Path) -> dict[str, Any]:
    """Assemble the figure aggregate from the campaign directory."""
    runs = load_training_runs(campaign)
    rows, table = load_control_rows(campaign)
    broad_config = json.loads((campaign / "broad.json").read_text(encoding="utf-8"))
    record = {
        "schema_version": 1,
        "description": "Figure-level aggregate of the manuscript's finite-width GD evidence figure, computed from the frozen campaign records.",
        "raw_source": {
            "description": "Validated compatible-only control table",
            "rows": len(rows),
            "sha256": _sha256(table),
        },
        "panels": {
            "width_diagnostics": width_diagnostics(runs),
            "compatible_singular": compatible_singular(rows),
            "conditioning_width_16384": conditioning_width_16384(rows),
        },
        "supporting_counts": supporting_counts(rows),
        "provenance": {
            "campaign": _relative(campaign),
            "campaign_name": broad_config.get("name"),
            "broad_config_sha256": _sha256(campaign / "broad.json"),
            "controls_config_sha256": _sha256(campaign / "controls.json"),
            "findings_sha256": _sha256(campaign / "results" / "broad" / "findings.json"),
            "controls_aggregate_sha256": _sha256(campaign / "results" / "controls_compatible" / "aggregate.json"),
            "builder": _relative(Path(__file__)),
            "builder_sha256": _sha256(Path(__file__)),
        },
    }
    return record


def serialize(record: dict[str, Any]) -> str:
    return json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN, help="frozen campaign directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="tracked aggregate to write")
    parser.add_argument("--check", action="store_true", help="rebuild in memory and compare with --output instead of writing")
    args = parser.parse_args(argv)

    record = build(args.campaign)
    text = serialize(record)
    if args.check:
        current = args.output.read_text(encoding="utf-8")
        if current != text:
            print(f"[build_figure_aggregate] {args.output} differs from a rebuild of {args.campaign}", file=sys.stderr)
            return 1
        print(f"[build_figure_aggregate] {args.output} matches the campaign {_relative(args.campaign)}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    _load_frozen_aggregate(args.output)
    widths = record["panels"]["width_diagnostics"]
    retained = [widths[str(width)]["gap_retained_at_saved_checkpoints"] for width in WIDTHS]
    print(
        f"[build_figure_aggregate] wrote {_relative(args.output)} from {_relative(args.campaign)}: "
        f"{record['raw_source']['rows']} compatible controls, gap retained {retained} of {RUNS_PER_WIDTH} at widths {list(WIDTHS)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
