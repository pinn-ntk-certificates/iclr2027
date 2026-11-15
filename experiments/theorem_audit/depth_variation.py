"""Kernel stability and gradient descent at depth ``H`` for audited PDE designs.

The two-layer theorem ``thm:linearPDEsconvergence`` does not cover deeper
networks.  This script measures, for the depth-``H`` network of
``def:depth_h_network`` with uniform hidden width ``d``, the quantities that
the two-layer theory controls: the smallest eigenvalue of the tangent Gram
matrix at initialization (in total and per layer block, see
``lem:layerwise_decomposition``), its drift along full-batch gradient descent,
the discrepancy from the frozen-kernel recursion, and the training loss.

Network (``sigma = tanh``, bias scale ``gamma``, all parameters trained)::

    t1 = W1 x + gamma b1,                 y1 = tanh(t1),
    tl = Wl y(l-1) / sqrt(d) + gamma bl,  yl = tanh(tl),   l = 2..H,
    u  = Wout yH / sqrt(d).

Every coordinate is i.i.d. ``Uniform[-1, 1]``.  Within a seed the widths are
nested: the width-``d`` network uses the leading ``d`` rows and columns of one
maximum-width bank.  ``H = 1`` is the two-layer network of the paper.

Rows, group scaling and targets are those of ``features.compile_constraints``.
The prediction of row ``i`` is ``P_i = sum_t c_t d^alpha_t u_{q_t}(x_t)`` over
its terms, of differential order at most two; the spatial derivatives are
propagated through the layers by the chain rule (``predictions``) and checked
in the tests against nested forward-mode differentiation.  The Jacobian
``J = dP/dtheta`` is reverse-mode.  The loss is ``||P - y||^2``
and one step is ``theta <- theta - 2 eta J^T (P - y)`` with
``eta = learning_rate_factor / lambda_max(K_0)``, ``K = J J^T``.

Per run the record holds ``lambda_min`` and ``lambda_max`` of ``K_0``,
``lambda_min`` of each layer block ``K_0^[l]``, and over the saved checkpoints
``C`` the diagnostics

- drift ``max_{k in C} ||K_k - K_0||_op / ||K_0||_op``,
- gap ratio ``min_{k in C} lambda_min(K_k) / lambda_min(K_0)``,
- frozen discrepancy ``||r_T - rt_T|| / ||rt_T||`` with
  ``rt_{k+1} = (I - 2 eta K_0) rt_k``, ``rt_0 = r_0``,
- loss ratio ``L_T / L_0``.

These are observations at the configured depths, widths and seeds.  There is
no limiting reference kernel at depth here, so the gap ratio is relative to
the run's own ``K_0``, unlike the two-layer table of the paper.

All settings come from the configuration file.  Run::

    uv run python -m experiments.theorem_audit.depth_variation \\
        --config experiments/theorem_audit/configs/depth_variation_smoke.json
    uv run python -m experiments.theorem_audit.depth_variation \\
        --config experiments/theorem_audit/configs/depth_variation.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.func import jacfwd, jacrev, vmap

from .features import CompiledConstraints, compile_constraints
from .problems import get_problem, list_problems
from .run import AuditError, PROJECT_ROOT, canonical_config_hash

SCHEMA_VERSION = 1
SUMMARY_NAME = "depth_variation_summary.json"
TABLE_NAME = "depth_variation_table.tex"
DTYPE = torch.float64
TINY = float(np.finfo(np.float64).tiny)
MAX_ORDER = 2
REQUIRED_FIELDS = (
    "output_dir",
    "problems",
    "depths",
    "widths",
    "seeds",
    "bias_scale",
    "steps",
    "checkpoint_every",
    "learning_rate_factor",
    "gap_fraction",
    "processes",
    "threads_per_process",
)

Params = dict[str, torch.Tensor]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise AuditError("configuration must be a JSON object")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AuditError(f"unsupported schema_version: {config.get('schema_version')!r}")
    for key in REQUIRED_FIELDS:
        if key not in config:
            raise AuditError(f"configuration is missing required field {key!r}")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    known = set(list_problems())
    for key in ("problems", "depths", "widths", "seeds"):
        values = config[key]
        if not values or len(set(values)) != len(values):
            raise AuditError(f"{key} must be a nonempty list without repetitions")
    unknown = [name for name in config["problems"] if name not in known]
    if unknown:
        raise AuditError(f"unknown problems: {unknown}")
    for name in config["problems"]:
        if int(compile_constraints(get_problem(name)).orders.max()) > MAX_ORDER:
            raise AuditError(f"{name} has rows of differential order above {MAX_ORDER}")
    if any(int(depth) < 1 for depth in config["depths"]):
        raise AuditError("depths must be positive integers")
    if any(int(width) < 1 for width in config["widths"]):
        raise AuditError("widths must be positive integers")
    if float(config["bias_scale"]) < 0.0:
        raise AuditError("bias_scale must be nonnegative")
    steps, every = int(config["steps"]), int(config["checkpoint_every"])
    if steps < 1 or every < 1 or steps % every != 0:
        raise AuditError("steps must be a positive multiple of checkpoint_every")
    if not 0.0 < float(config["learning_rate_factor"]) <= 0.5:
        raise AuditError("learning_rate_factor must lie in (0, 0.5]")
    if not 0.0 < float(config["gap_fraction"]) < 1.0:
        raise AuditError("gap_fraction must lie in (0, 1)")
    for key in ("processes", "threads_per_process"):
        if int(config[key]) < 1:
            raise AuditError(f"{key} must be a positive integer")


# ---------------------------------------------------------------------------
# Network


def initial_parameters(
    input_dim: int, output_dim: int, depth: int, width: int, max_width: int, seed: int
) -> Params:
    """Leading width-``width`` block of the seed's maximum-width ``Uniform[-1, 1]`` bank."""
    if width > max_width:
        raise AuditError("width exceeds the maximum width of the bank")
    generator = torch.Generator().manual_seed(int(seed))

    def uniform(*shape: int) -> torch.Tensor:
        return 2.0 * torch.rand(*shape, generator=generator, dtype=DTYPE) - 1.0

    params: Params = {}
    params["W1"] = uniform(max_width, input_dim)[:width].clone()
    params["b1"] = uniform(max_width)[:width].clone()
    for layer in range(2, depth + 1):
        params[f"W{layer}"] = uniform(max_width, max_width)[:width, :width].clone()
        params[f"b{layer}"] = uniform(max_width)[:width].clone()
    params["Wout"] = uniform(output_dim, max_width)[:, :width].clone()
    return params


def depth_of(params: Params) -> int:
    return sum(1 for name in params if name.startswith("b"))


def layer_names(params: Params) -> list[list[str]]:
    """Parameter names of the layer blocks ``1..H+1`` of ``eqn:layer_jacobian_blocks``."""
    depth = depth_of(params)
    return [[f"W{layer}", f"b{layer}"] for layer in range(1, depth + 1)] + [["Wout"]]


def network(params: Params, x: torch.Tensor, bias_scale: float) -> torch.Tensor:
    """``u(x; theta)`` of ``eqn:depth_network`` for one point ``x`` of shape ``[d0]``."""
    depth = depth_of(params)
    y = torch.tanh(params["W1"] @ x + bias_scale * params["b1"])
    for layer in range(2, depth + 1):
        weight = params[f"W{layer}"]
        y = torch.tanh(weight @ y / math.sqrt(weight.shape[1]) + bias_scale * params[f"b{layer}"])
    out = params["Wout"]
    return out @ y / math.sqrt(out.shape[1])


def _alpha_indices(alphas: torch.Tensor) -> list[tuple[int, ...]]:
    """Each multi-index as a tuple of coordinate indices, e.g. ``(1, 2) -> (0, 1, 1)``."""
    return [
        tuple(coordinate for coordinate, power in enumerate(alpha) for _ in range(int(power)))
        for alpha in alphas.tolist()
    ]


def predictions(params: Params, compiled: CompiledConstraints, bias_scale: float) -> torch.Tensor:
    """Loss-scaled row predictions ``P(theta)`` of shape ``[N]``, rows of order at most two.

    The value, the first and the second spatial derivatives of every hidden
    layer are propagated for all term points at once.  With ``y = tanh(t)``,
    ``s1 = 1 - y^2`` and ``s2 = -2 y s1``::

        d_j y = s1 d_j t,    d_jk y = s2 d_j t d_k t + s1 d_jk t,

    and ``t`` is affine in the layer below.  ``predictions_reference`` computes
    the same quantity by nested forward-mode differentiation.
    """
    if int(compiled.orders.max()) > MAX_ORDER:
        raise AuditError(f"rows of differential order above {MAX_ORDER} are not supported")
    depth = depth_of(params)
    input_dim = compiled.input_dim
    points = compiled.points  # [T, d0]
    pairs = [(j, k) for j in range(input_dim) for k in range(j, input_dim)]

    weight = params["W1"]
    t = points @ weight.T + bias_scale * params["b1"]
    first_t = [weight[:, j].expand_as(t) for j in range(input_dim)]
    second_t = {pair: torch.zeros_like(t) for pair in pairs}
    for layer in range(1, depth + 1):
        if layer > 1:
            weight = params[f"W{layer}"] / math.sqrt(params[f"W{layer}"].shape[1])
            t = y @ weight.T + bias_scale * params[f"b{layer}"]
            first_t = [value @ weight.T for value in first]
            second_t = {pair: value @ weight.T for pair, value in second.items()}
        y = torch.tanh(t)
        s1 = 1.0 - y * y
        s2 = -2.0 * y * s1
        first = [s1 * value for value in first_t]
        second = {
            (j, k): s2 * first_t[j] * first_t[k] + s1 * second_t[(j, k)] for (j, k) in pairs
        }
    out = params["Wout"].T / math.sqrt(params["Wout"].shape[1])  # [d, q]

    indices = _alpha_indices(compiled.alphas)
    values = torch.zeros(compiled.term_count, dtype=DTYPE)
    rows = torch.arange(compiled.term_count)
    for order, index in sorted(set(zip(compiled.orders.tolist(), indices))):
        terms = torch.tensor([t_ for t_ in range(compiled.term_count) if indices[t_] == index])
        hidden = y if order == 0 else first[index[0]] if order == 1 else second[index]
        field = hidden @ out  # [T, q]
        values = values.index_add(0, terms, field[rows[terms], compiled.outputs[terms]])
    return torch.zeros(compiled.count, dtype=DTYPE).index_add(
        0, compiled.constraint_index, compiled.coefficients * values
    )


def predictions_reference(
    params: Params, compiled: CompiledConstraints, bias_scale: float
) -> torch.Tensor:
    """``predictions`` by nested forward-mode differentiation of ``network``; any order."""
    indices = _alpha_indices(compiled.alphas)
    values = torch.zeros(compiled.term_count, dtype=DTYPE)
    for order in sorted(set(compiled.orders.tolist())):
        terms = torch.nonzero(compiled.orders == order).flatten()

        def derivative(x: torch.Tensor):
            function = lambda point: network(params, point, bias_scale)  # noqa: E731
            for _ in range(order):
                function = jacfwd(function)
            return function(x)

        tensors = vmap(derivative)(compiled.points[terms])  # [T_k, q, d0, ..., d0]
        selector: list[torch.Tensor] = [torch.arange(terms.numel()), compiled.outputs[terms]]
        for slot in range(order):
            selector.append(torch.tensor([indices[int(term)][slot] for term in terms]))
        values = values.index_add(0, terms, tensors[tuple(selector)])
    return torch.zeros(compiled.count, dtype=DTYPE).index_add(
        0, compiled.constraint_index, compiled.coefficients * values
    )


def layer_jacobians(
    params: Params, compiled: CompiledConstraints, bias_scale: float
) -> list[torch.Tensor]:
    """Column blocks ``J^[1], ..., J^[H+1]`` of ``dP/dtheta``, each ``[N, D_l]``."""
    jacobian = jacrev(lambda p: predictions(p, compiled, bias_scale))(params)
    return [
        torch.cat([jacobian[name].reshape(compiled.count, -1) for name in names], dim=1)
        for names in layer_names(params)
    ]


def gram(blocks: list[torch.Tensor]) -> torch.Tensor:
    matrix = sum(block @ block.T for block in blocks)
    return 0.5 * (matrix + matrix.T)


def loss(params: Params, compiled: CompiledConstraints, bias_scale: float) -> torch.Tensor:
    residual = predictions(params, compiled, bias_scale) - compiled.targets
    return residual @ residual


# ---------------------------------------------------------------------------
# One run


def run_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Train one ``(problem, depth, width, seed)`` cell and return its record."""
    torch.set_num_threads(int(task["threads_per_process"]))
    started = time.perf_counter()
    problem = get_problem(task["problem"])
    compiled = compile_constraints(problem)
    bias_scale = float(task["bias_scale"])
    depth, width, seed = int(task["depth"]), int(task["width"]), int(task["seed"])
    params = initial_parameters(
        problem.input_dim, problem.output_dim, depth, width, int(task["max_width"]), seed
    )
    parameter_count = sum(value.numel() for value in params.values())

    blocks = layer_jacobians(params, compiled, bias_scale)
    kernel_0 = gram(blocks)
    eigenvalues = torch.linalg.eigvalsh(kernel_0)
    lambda_min_0, lambda_max_0 = float(eigenvalues[0]), float(eigenvalues[-1])
    layer_lambda_min = [float(torch.linalg.eigvalsh(block @ block.T)[0]) for block in blocks]
    eta = float(task["learning_rate_factor"]) / lambda_max_0

    residual_0 = predictions(params, compiled, bias_scale) - compiled.targets
    loss_0 = float(residual_0 @ residual_0)
    frozen = residual_0.clone()
    frozen_step = torch.eye(compiled.count, dtype=DTYPE) - 2.0 * eta * kernel_0

    steps, every = int(task["steps"]), int(task["checkpoint_every"])
    max_drift, min_gap_ratio = 0.0, 1.0
    checkpoint_losses = [loss_0]
    names = list(params)
    for step in range(1, steps + 1):
        leaves = {name: params[name].requires_grad_(True) for name in names}
        update = torch.autograd.grad(loss(leaves, compiled, bias_scale), [leaves[name] for name in names])
        params = {
            name: (leaves[name] - eta * slope).detach() for name, slope in zip(names, update)
        }
        frozen = frozen_step @ frozen
        if step % every == 0:
            kernel = gram(layer_jacobians(params, compiled, bias_scale))
            drift = float(torch.linalg.matrix_norm(kernel - kernel_0, ord=2)) / max(lambda_max_0, TINY)
            max_drift = max(max_drift, drift)
            ratio = float(torch.linalg.eigvalsh(kernel)[0]) / max(lambda_min_0, TINY)
            min_gap_ratio = min(min_gap_ratio, ratio)
            checkpoint_losses.append(float(loss(params, compiled, bias_scale)))

    residual = predictions(params, compiled, bias_scale) - compiled.targets
    loss_final = float(residual @ residual)
    discrepancy = float(torch.linalg.vector_norm(residual - frozen)) / max(
        float(torch.linalg.vector_norm(frozen)), TINY
    )
    return {
        "problem": task["problem"],
        "depth": depth,
        "width": width,
        "seed": seed,
        "N": compiled.count,
        "parameter_count": parameter_count,
        "lambda_min_K0": lambda_min_0,
        "lambda_max_K0": lambda_max_0,
        "layer_lambda_min_K0": layer_lambda_min,
        "eta": eta,
        "loss_initial": loss_0,
        "loss_final": loss_final,
        "loss_ratio": loss_final / max(loss_0, TINY),
        "loss_nonincreasing_at_checkpoints": all(
            later <= earlier for earlier, later in zip(checkpoint_losses, checkpoint_losses[1:])
        ),
        "max_kernel_drift": max_drift,
        "min_gap_ratio": min_gap_ratio,
        "gap_retained": min_gap_ratio >= float(task["gap_fraction"]),
        "frozen_discrepancy": discrepancy,
        "seconds": time.perf_counter() - started,
    }


def tasks_of(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    shared = {
        key: config[key]
        for key in (
            "bias_scale",
            "steps",
            "checkpoint_every",
            "learning_rate_factor",
            "gap_fraction",
            "threads_per_process",
        )
    }
    max_width = max(int(width) for width in config["widths"])
    return [
        {"problem": problem, "depth": depth, "width": width, "seed": seed, "max_width": max_width, **shared}
        for problem in config["problems"]
        for depth in config["depths"]
        for width in config["widths"]
        for seed in config["seeds"]
    ]


# ---------------------------------------------------------------------------
# Aggregation


def aggregate(config: Mapping[str, Any], runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Medians and counts over seeds per ``(problem, depth, width)``; fails on a shortfall."""
    declared = (
        len(config["problems"]) * len(config["depths"]) * len(config["widths"]) * len(config["seeds"])
    )
    keys = {(run["problem"], run["depth"], run["width"], run["seed"]) for run in runs}
    if len(runs) != declared or len(keys) != declared:
        raise AuditError(
            f"found {len(runs)} run records ({len(keys)} distinct) but the configuration declares {declared}"
        )
    cells = []
    for problem in config["problems"]:
        for depth in config["depths"]:
            for width in config["widths"]:
                group = [
                    run
                    for run in runs
                    if (run["problem"], run["depth"], run["width"]) == (problem, depth, width)
                ]
                median = lambda key: statistics.median(run[key] for run in group)  # noqa: E731
                cells.append(
                    {
                        "problem": problem,
                        "depth": depth,
                        "width": width,
                        "runs": len(group),
                        "median_lambda_min_K0": median("lambda_min_K0"),
                        "median_best_layer_lambda_min_K0": statistics.median(
                            max(run["layer_lambda_min_K0"]) for run in group
                        ),
                        "median_max_kernel_drift": median("max_kernel_drift"),
                        "median_frozen_discrepancy": median("frozen_discrepancy"),
                        "median_loss_ratio": median("loss_ratio"),
                        "median_min_gap_ratio": median("min_gap_ratio"),
                        "gap_retained": sum(bool(run["gap_retained"]) for run in group),
                        "loss_decreased": sum(run["loss_final"] < run["loss_initial"] for run in group),
                        "loss_nonincreasing_at_checkpoints": sum(
                            bool(run["loss_nonincreasing_at_checkpoints"]) for run in group
                        ),
                    }
                )
    return {"declared_runs": declared, "found_runs": len(runs), "cells": cells}


def scientific(value: float) -> str:
    if value == 0.0:
        return "$0$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def tex_table(record: Mapping[str, Any]) -> str:
    seeds = len(record["config"]["seeds"])
    lines = [
        f"% Generated by experiments/theorem_audit/depth_variation.py from the configuration with "
        f"SHA256 {record['config_sha256']}; do not edit numerical values manually.",
        r"\begin{tabular}{@{}lrrcccccc@{}}",
        r"\toprule",
        r"Design & $H$ & $d$ & $\lambda_{\min}(\mathbf K_{(0)})$ & Max.\ drift & Frozen discr.\ "
        r"& $\mathcal L_T/\mathcal L_0$ & Gap kept & Loss decreased \\",
        r"\midrule",
    ]
    for cell in record["cells"]:
        lines.append(
            " & ".join(
                [
                    cell["problem"].replace("_", r"\_"),
                    str(cell["depth"]),
                    f"{cell['width']:,}".replace(",", "{,}"),
                    scientific(cell["median_lambda_min_K0"]),
                    scientific(cell["median_max_kernel_drift"]),
                    scientific(cell["median_frozen_discrepancy"]),
                    scientific(cell["median_loss_ratio"]),
                    f"{cell['gap_retained']}/{seeds}",
                    f"{cell['loss_decreased']}/{seeds}",
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args(argv)

    config = load_config(arguments.config)
    tasks = tasks_of(config)
    started = time.perf_counter()
    if int(config["processes"]) == 1:
        runs = [run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(config["processes"])) as pool:
            runs = list(pool.map(run_task, tasks))
    elapsed = time.perf_counter() - started
    summary = aggregate(config, runs)

    from .run import _git_metadata, _portable_path  # noqa: PLC0415 - provenance only

    record = {
        "schema_version": SCHEMA_VERSION,
        "config": dict(config),
        "config_path": _portable_path(str(arguments.config)).replace("\\", "/"),
        "config_sha256": canonical_config_hash(config),
        **summary,
        "runs": runs,
        "elapsed_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "git": _git_metadata(),
        },
    }
    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / SUMMARY_NAME
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")
    (output_dir / TABLE_NAME).write_text(tex_table(record), encoding="utf-8", newline="\n")

    print(
        f"{'design':<22}{'H':>2}{'d':>6} {'lmin(K0)':>10} {'best layer':>10} {'drift':>9} "
        f"{'frozen':>9} {'L_T/L_0':>9} {'gap':>5} {'decr':>5}"
    )
    seeds = len(config["seeds"])
    for cell in summary["cells"]:
        print(
            f"{cell['problem']:<22}{cell['depth']:>2}{cell['width']:>6} "
            f"{cell['median_lambda_min_K0']:>10.2e} {cell['median_best_layer_lambda_min_K0']:>10.2e} "
            f"{cell['median_max_kernel_drift']:>9.2e} {cell['median_frozen_discrepancy']:>9.2e} "
            f"{cell['median_loss_ratio']:>9.2e} {cell['gap_retained']:>3}/{seeds} "
            f"{cell['loss_decreased']:>3}/{seeds}"
        )
    print(
        f"\n{summary['found_runs']} of {summary['declared_runs']} runs; "
        f"wrote {destination.relative_to(PROJECT_ROOT).as_posix()} in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
