"""Share of the initial residual in the weak eigendirection of the two-row controls.

For a two-row rank control of :mod:`rank_controls` (family, mixing parameter
``epsilon``, target mode) the empirical kernel at initialization is the
``2 x 2`` matrix ``K_0 = J_0 J_0^T / d_1`` and the residual is
``e_0 = P(theta_0) - y``.  Full-batch gradient descent on ``||P - y||^2`` with
step ``eta = c / lambda_max(K_0)`` follows, for the frozen kernel, the
recursion ``e_{k+1} = (I - 2 eta K_0) e_k``.  With the eigenpairs
``(lambda_j, v_j)`` of ``K_0`` and the shares

    s_j = <v_j, e_0>^2 / ||e_0||^2,            sum_j s_j = 1,

the frozen-kernel loss after ``T`` steps is

    L_T / L_0 = sum_j s_j (1 - 2 c lambda_j / lambda_max)^(2 T).

The strong direction (``lambda_j = lambda_max``) is multiplied by
``(1 - 2c)^(2T)``, which is negligible for ``c = 0.2`` and ``T = 3000``.  The
weak direction (``lambda_j = lambda_min``) is multiplied by
``(1 - 2 c rho)^(2T)`` with ``rho = lambda_min / lambda_max``; for
``epsilon = 0.01`` the factor is close to one, so the frozen ratio is close to
the weak share ``s_min``.  This script records, for every configured family,
``epsilon`` and seed, the eigenvalues of ``K_0``, the shares, the two factors
and the predicted ratio, takes medians over seeds, and, when the tracked
aggregate of the control campaign is present, lists the observed median
final-to-initial loss ratio next to the prediction.

The initialization is the same nested-prefix bank as the control runs (same
seed, same maximum width), so ``K_0`` and ``e_0`` here are exactly the
``K_initial`` and ``initial_error`` arrays of the corresponding control
record.  The comparison with the observed ratio is an observation at the
configured settings: the trained network does not keep the kernel frozen, and
the difference between the two ratios is the effect of the kernel drift.
Nothing here verifies a theorem.

All settings come from the configuration file.  Run::

    uv run python -m experiments.theorem_audit.weak_direction_share \\
        --config experiments/theorem_audit/configs/weak_direction_share_smoke.json
    uv run python -m experiments.theorem_audit.weak_direction_share \\
        --config experiments/theorem_audit/configs/weak_direction_share.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .engine import evaluate_finite, make_initialization_bank
from .features import compile_constraints
from .rank_controls import FAMILIES, make_control
from .run import AuditError, PROJECT_ROOT, canonical_config_hash

SCHEMA_VERSION = 1
SUMMARY_NAME = "weak_direction_share_summary.json"
REQUIRED_KEYS = (
    "schema_version",
    "output_dir",
    "families",
    "epsilons",
    "target_mode",
    "contrast_amplitude",
    "width",
    "maximum_width_initialization_bank",
    "seeds",
    "steps",
    "learning_rate_factor",
)


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise AuditError("configuration must be a JSON object")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise AuditError(f"configuration is missing {missing}")
    if config["schema_version"] != SCHEMA_VERSION:
        raise AuditError(f"schema_version must be {SCHEMA_VERSION}")
    families = config["families"]
    if not isinstance(families, list) or not families:
        raise AuditError("families must be a non-empty list")
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise AuditError(f"unknown control families {unknown}; known: {list(FAMILIES)}")
    epsilons = config["epsilons"]
    if not isinstance(epsilons, list) or not epsilons:
        raise AuditError("epsilons must be a non-empty list")
    for epsilon in epsilons:
        if epsilon is None:
            continue
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise AuditError("each epsilon must be null (independent rows) or a positive finite number")
    if config["target_mode"] not in {"manufactured", "contrast"}:
        raise AuditError("target_mode must be manufactured or contrast")
    amplitude = config["contrast_amplitude"]
    if not isinstance(amplitude, (int, float)) or not math.isfinite(amplitude):
        raise AuditError("contrast_amplitude must be a finite number")
    width = config["width"]
    bank = config["maximum_width_initialization_bank"]
    if not isinstance(width, int) or width <= 0 or not isinstance(bank, int) or bank < width:
        raise AuditError("width must be a positive integer not larger than maximum_width_initialization_bank")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise AuditError("seeds must be a non-empty list of non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise AuditError("seeds must be distinct")
    steps = config["steps"]
    if not isinstance(steps, int) or steps <= 0:
        raise AuditError("steps must be a positive integer")
    factor = config["learning_rate_factor"]
    if not isinstance(factor, (int, float)) or not 0 < factor < 1:
        raise AuditError("learning_rate_factor must lie in (0, 1)")
    threads = config.get("threads", 1)
    if not isinstance(threads, int) or threads <= 0:
        raise AuditError("threads must be a positive integer")


def residual_shares(kernel: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending) of the symmetrized kernel and the residual shares.

    The shares are the squared coefficients of ``error`` in the orthonormal
    eigenbasis, divided by ``||error||^2``; they sum to one.
    """

    matrix = np.asarray(kernel, dtype=np.float64)
    vector = np.asarray(error, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or vector.shape != (matrix.shape[0],):
        raise AuditError("kernel must be square and error must be a vector of matching length")
    norm = float(vector @ vector)
    if not norm > 0:
        raise AuditError("the initial residual vanishes; shares are undefined")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    coefficients = eigenvectors.T @ vector
    return eigenvalues, coefficients**2 / norm


def frozen_prediction(
    eigenvalues: np.ndarray, shares: np.ndarray, learning_rate_factor: float, steps: int
) -> tuple[float, np.ndarray]:
    """Frozen-kernel ``L_T / L_0`` and the per-direction factors ``(1 - 2 c lambda_j / lambda_max)^(2T)``."""

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    shares = np.asarray(shares, dtype=np.float64)
    lambda_max = float(eigenvalues.max())
    if not lambda_max > 0:
        raise AuditError("lambda_max must be positive")
    factors = (1.0 - 2.0 * learning_rate_factor * eigenvalues / lambda_max) ** (2 * steps)
    return float(np.dot(shares, factors)), factors


def evaluate_cell(
    family: str, epsilon: float | None, seed: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Kernel, residual shares and frozen prediction for one control at one seed."""

    problem, metadata = make_control(
        family, epsilon, config["target_mode"], float(config["contrast_amplitude"])
    )
    compiled = compile_constraints(problem)
    bank = make_initialization_bank(
        int(config["maximum_width_initialization_bank"]), problem.input_dim, problem.output_dim, seed
    )
    state = bank.prefix(int(config["width"]))
    with torch.no_grad():
        initial = evaluate_finite(state, compiled)
    kernel = initial.kernel.cpu().numpy()
    error = initial.error.cpu().numpy()
    eigenvalues, shares = residual_shares(kernel, error)
    predicted, factors = frozen_prediction(
        eigenvalues, shares, float(config["learning_rate_factor"]), int(config["steps"])
    )
    lambda_min, lambda_max = float(eigenvalues[0]), float(eigenvalues[-1])
    return {
        "family": family,
        "epsilon": epsilon,
        "seed": seed,
        "problem_name": problem.name,
        "exact_structural_rank": metadata["exact_structural_rank"],
        "width": int(config["width"]),
        "kernel_initial": kernel.tolist(),
        "initial_error": error.tolist(),
        "initial_loss": float(error @ error),
        "eigenvalues_ascending": eigenvalues.tolist(),
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "relative_gap": lambda_min / lambda_max,
        "condition_number": lambda_max / lambda_min if lambda_min > 0 else math.inf,
        "shares_ascending": shares.tolist(),
        "weak_share": float(shares[0]),
        "direction_factors": factors.tolist(),
        "weak_direction_factor": float(factors[0]),
        "weak_direction_decrease": 1.0 - float(factors[0]),
        "predicted_final_to_initial_loss_ratio": predicted,
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def summarize(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Medians over seeds for every (family, epsilon) in the order of first appearance."""

    groups: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for cell in cells:
        groups.setdefault((cell["family"], cell["epsilon"]), []).append(cell)
    summary = []
    for (family, epsilon), members in groups.items():
        summary.append(
            {
                "family": family,
                "epsilon": epsilon,
                "seeds": [cell["seed"] for cell in members],
                "runs": len(members),
                "median_relative_gap": _median([cell["relative_gap"] for cell in members]),
                "median_condition_number": _median([cell["condition_number"] for cell in members]),
                "median_weak_share": _median([cell["weak_share"] for cell in members]),
                "median_weak_direction_decrease": _median([cell["weak_direction_decrease"] for cell in members]),
                "median_predicted_final_to_initial_loss_ratio": _median(
                    [cell["predicted_final_to_initial_loss_ratio"] for cell in members]
                ),
            }
        )
    return summary


def _epsilon_key(epsilon: float | None) -> str:
    return "independent" if epsilon is None else f"{epsilon:g}"


def observed_medians(config: Mapping[str, Any]) -> dict[str, Any]:
    """Observed median loss ratios from the tracked control aggregate, if configured and present."""

    relative = config.get("observed_aggregate")
    if not relative:
        return {"available": False, "reason": "no observed_aggregate configured"}
    path = PROJECT_ROOT / relative
    if not path.exists():
        return {"available": False, "reason": f"{relative} not found", "path": relative}
    with path.open("r", encoding="utf-8") as handle:
        aggregate = json.load(handle)
    panel_name = config.get("observed_panel", "")
    panel = aggregate.get("panels", {}).get(panel_name)
    if not isinstance(panel, dict):
        return {"available": False, "reason": f"panel {panel_name!r} not found", "path": relative}
    values: dict[str, dict[str, float]] = {}
    for family, by_epsilon in panel.items():
        if isinstance(by_epsilon, dict):
            values[family] = {
                key: entry["median_final_to_initial_loss_ratio"]
                for key, entry in by_epsilon.items()
                if isinstance(entry, dict) and "median_final_to_initial_loss_ratio" in entry
            }
    return {
        "available": True,
        "path": relative,
        "panel": panel_name,
        "aggregate_description": aggregate.get("description"),
        "raw_source_sha256": aggregate.get("raw_source", {}).get("sha256"),
        "median_final_to_initial_loss_ratio": values,
    }


def compare(summary: list[dict[str, Any]], observed: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not observed.get("available"):
        return []
    table = observed["median_final_to_initial_loss_ratio"]
    rows = []
    for entry in summary:
        value = table.get(entry["family"], {}).get(_epsilon_key(entry["epsilon"]))
        rows.append(
            {
                "family": entry["family"],
                "epsilon": entry["epsilon"],
                "predicted": entry["median_predicted_final_to_initial_loss_ratio"],
                "observed": value,
                "difference": None if value is None else entry["median_predicted_final_to_initial_loss_ratio"] - value,
            }
        )
    return rows


def build_record(config: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config.get("threads", 1)))
    torch.use_deterministic_algorithms(True)
    cells = [
        evaluate_cell(family, epsilon, seed, config)
        for family in config["families"]
        for epsilon in config["epsilons"]
        for seed in config["seeds"]
    ]
    summary = summarize(cells)
    observed = observed_medians(config)
    return {
        "declared_cells": len(config["families"]) * len(config["epsilons"]) * len(config["seeds"]),
        "found_cells": len(cells),
        "cells": cells,
        "summary": summary,
        "observed": observed,
        "comparison": compare(summary, observed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    started = time.perf_counter()
    result = build_record(config)
    elapsed = time.perf_counter() - started

    from .run import _git_metadata, _portable_path  # noqa: PLC0415 - provenance only

    record = {
        "schema_version": SCHEMA_VERSION,
        "config": dict(config),
        "config_path": _portable_path(str(arguments.config)).replace("\\", "/"),
        "config_sha256": canonical_config_hash(config),
        **result,
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": torch.get_num_threads(),
            "git": _git_metadata(),
        },
    }

    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / SUMMARY_NAME
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=False)
        handle.write("\n")

    observed = {
        (row["family"], row["epsilon"]): row["observed"] for row in result["comparison"]
    }
    print(
        f"{'family':<20} {'epsilon':>8} {'runs':>4} {'median rho':>12} {'weak share':>11} "
        f"{'weak decrease':>14} {'frozen L_T/L_0':>15} {'observed':>9}"
    )
    for entry in result["summary"]:
        value = observed.get((entry["family"], entry["epsilon"]))
        shown = "n/a" if value is None else f"{value:.3f}"
        print(
            f"{entry['family']:<20} {_epsilon_key(entry['epsilon']):>8} {entry['runs']:>4} "
            f"{entry['median_relative_gap']:>12.3e} {entry['median_weak_share']:>11.3f} "
            f"{entry['median_weak_direction_decrease']:>14.4f} "
            f"{entry['median_predicted_final_to_initial_loss_ratio']:>15.3f} {shown:>9}"
        )
    shown = destination.relative_to(PROJECT_ROOT) if destination.is_relative_to(PROJECT_ROOT) else destination
    print(f"\nwrote {shown.as_posix()} in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
