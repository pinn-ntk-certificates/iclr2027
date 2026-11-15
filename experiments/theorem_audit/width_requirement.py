"""Width required by ``eqn:linear_pde_convergence_conditions`` for the audited designs.

For every registered PDE design of the theorem audit this script evaluates the
three width terms of ``thm:linearPDEsconvergence`` and compares them with the
widths that were trained.  The network is the audited one: two layers,
``sigma = tanh``, per-neuron parameters ``z = (w, b, a)`` with independent
``Uniform[-1, 1]`` coordinates, so ``E[a_q a_r] = delta_{qr} / 3``.

Per design, with ``h_i(z) = a^T f_i(w, b)`` the per-neuron prediction of row
``i`` and ``Psi(z)`` its ``N x d_theta`` Jacobian:

1. Limiting kernel.  ``K_infty = E[Psi Psi^T]`` is computed by tensor-product
   Gauss-Legendre quadrature over ``(w, b)`` with the output weights
   integrated in closed form.  Two quadrature orders are compared and the run
   fails when ``lambda_min`` changes by more than the configured relative
   tolerance.  The gap used below is ``lambda_0 = lambda_min(K_infty)``, the
   largest value the theorem admits.
2. Envelope constants of ``def:local_row_regularity``.
   ``A_{0,i} = sup |h_i|`` and ``R_K = sup ||Psi||_2^2`` over the support, and
   ``B_i = sup ||grad h_i||_2``, ``L_i = sup ||Hess h_i||_2`` over the tube
   ``{z_0 + R u : z_0 in supp mu, ||u||_2 <= 1}``.  Each supremum is replaced
   by a maximum over random points that lie in the stated set (the corners of
   the support are included), so every reported constant is a LOWER bound of
   the true constant.
3. Residual bound.  ``H_0^2 = E ||h||^2 = tr(K_out) / 3`` by the same
   quadrature and ``rbar_delta = ||v_T|| + H_0 + A_{0,*} sqrt(2 log(2/delta))``.
4. Width.  ``T1 = R_K log(2N/delta) / (c_Ch lambda_0)``,
   ``T2 = (320 + 128 sqrt 6) L_*^2 B_*^2 rbar_delta^2 / lambda_0^3``,
   ``T3 = 16 B_*^2 rbar_delta^2 / (R^2 lambda_0^2)``, and the requirement is
   ``max(T1, T2, T3)`` minimized over the configured radii ``R``.

Every term is nondecreasing in the envelope constants, so the reported width
is a lower bound of the width the theorem requires at ``lambda_0 =
lambda_min(K_infty)``, up to quadrature and finite-difference error.  It is a
statement about the sufficient condition of the theorem, not about the width
at which gradient descent converges.

The record also reports the gap that would make the requirement equal to the
largest trained width.  Since ``lambda_0 <= lambda_min(K_infty)`` is forced,
a value above ``lambda_min(K_infty)`` means that no certificate can bring the
trained widths into the regime of the theorem with these constants.

All settings come from the configuration file.  Run::

    uv run python -m experiments.theorem_audit.width_requirement \\
        --config experiments/theorem_audit/configs/width_requirement_smoke.json
    uv run python -m experiments.theorem_audit.width_requirement \\
        --config experiments/theorem_audit/configs/width_requirement.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .features import CompiledConstraints, compile_constraints, evaluate_features
from .problems import get_problem, list_problems
from .run import AuditError, PROJECT_ROOT, canonical_config_hash

SCHEMA_VERSION = 1
SUMMARY_NAME = "width_requirement_summary.json"
TABLE_NAME = "width_requirement_table.tex"
DTYPE = torch.float64
OUTPUT_WEIGHT_SECOND_MOMENT = 1.0 / 3.0  # E[a^2] for a ~ Uniform[-1, 1]
C_CHERNOFF = 0.25 + 0.75 * math.log(0.75)
DRIFT_PREFACTOR = 320.0 + 128.0 * math.sqrt(6.0)
REQUIRED_FIELDS = (
    "output_dir",
    "problems",
    "delta",
    "tube_radii",
    "quadrature_orders",
    "quadrature_relative_tolerance",
    "quadrature_chunk",
    "support_samples",
    "tube_samples",
    "sample_chunk",
    "seed",
    "finite_difference_step",
    "tested_widths",
    "threads",
)


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
    problems = config["problems"]
    if not problems or len(set(problems)) != len(problems):
        raise AuditError("problems must be a nonempty list without repetitions")
    unknown = [name for name in problems if name not in known]
    if unknown:
        raise AuditError(f"unknown problems: {unknown}")
    if not 0.0 < float(config["delta"]) < 1.0:
        raise AuditError("delta must lie in (0, 1)")
    radii = config["tube_radii"]
    if not radii or any(float(radius) <= 0.0 for radius in radii):
        raise AuditError("tube_radii must be a nonempty list of positive numbers")
    orders = config["quadrature_orders"]
    for name in problems:
        hidden = str(get_problem(name).input_dim + 1)
        pair = orders.get(hidden)
        if pair is None or len(pair) != 2 or not 2 <= int(pair[0]) < int(pair[1]):
            raise AuditError(
                f"quadrature_orders[{hidden!r}] must be two increasing orders (needed by {name})"
            )
    if float(config["quadrature_relative_tolerance"]) <= 0.0:
        raise AuditError("quadrature_relative_tolerance must be positive")
    for key in ("quadrature_chunk", "support_samples", "tube_samples", "sample_chunk", "threads"):
        if int(config[key]) < 1:
            raise AuditError(f"{key} must be a positive integer")
    if float(config["finite_difference_step"]) <= 0.0:
        raise AuditError("finite_difference_step must be positive")
    widths = config["tested_widths"]
    if not widths or any(int(width) < 1 for width in widths):
        raise AuditError("tested_widths must be a nonempty list of positive integers")


# ---------------------------------------------------------------------------
# Limiting kernel by Gauss-Legendre quadrature


def limiting_kernel(
    compiled: CompiledConstraints, order: int, chunk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(K_infty, K_out)`` for ``(w, b, a)`` uniform on ``[-1, 1]^d_theta``."""
    hidden_dim = compiled.input_dim + 1
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    nodes_1d = torch.tensor(nodes_1d, dtype=DTYPE)
    weights_1d = torch.tensor(weights_1d / 2.0, dtype=DTYPE)  # probability weights
    nodes = torch.cartesian_prod(*([nodes_1d] * hidden_dim)).reshape(-1, hidden_dim)
    weights = torch.cartesian_prod(*([weights_1d] * hidden_dim)).reshape(-1, hidden_dim).prod(dim=1)
    count = compiled.count
    k_out = torch.zeros(count, count, dtype=DTYPE)
    k_hid = torch.zeros(count, count, dtype=DTYPE)
    for start in range(0, nodes.shape[0], chunk):
        xi = nodes[start : start + chunk]
        weight = weights[start : start + chunk]
        batch = evaluate_features(
            compiled,
            xi[:, : compiled.input_dim].contiguous(),
            xi[:, compiled.input_dim].contiguous(),
        )
        k_out += torch.einsum("m,miq,mjq->ij", weight, batch.output_basis, batch.output_basis)
        k_hid += OUTPUT_WEIGHT_SECOND_MOMENT * torch.einsum(
            "m,miqh,mjqh->ij", weight, batch.hidden_basis, batch.hidden_basis
        )
    k_out = 0.5 * (k_out + k_out.T)
    k_hid = 0.5 * (k_hid + k_hid.T)
    return k_out + k_hid, k_out


# ---------------------------------------------------------------------------
# Per-neuron prediction, Jacobian and Hessian


def per_neuron(
    compiled: CompiledConstraints, z: torch.Tensor, *, hessian_step: float | None = None
) -> dict[str, torch.Tensor]:
    """``h`` ``[S, N]``, ``Psi`` ``[S, N, d_theta]`` and optionally ``Hess`` at ``z``.

    The mixed block of the Hessian and the first derivatives are the exact
    hand-derived features.  The ``(w, b)`` block is a central difference of the
    exact first derivatives with step ``hessian_step``.
    """
    input_dim = compiled.input_dim
    hidden_dim = input_dim + 1
    xi = z[:, :hidden_dim]
    a = z[:, hidden_dim:]

    def features(points: torch.Tensor):
        return evaluate_features(
            compiled, points[:, :input_dim].contiguous(), points[:, input_dim].contiguous()
        )

    batch = features(xi)
    output_basis = batch.output_basis  # [S, N, q]
    hidden_basis = batch.hidden_basis  # [S, N, q, hidden_dim]
    result = {
        "h": torch.einsum("snq,sq->sn", output_basis, a),
        "Psi": torch.cat((torch.einsum("snqh,sq->snh", hidden_basis, a), output_basis), dim=-1),
    }
    if hessian_step is not None:
        samples, count = result["h"].shape
        total = hidden_dim + compiled.output_dim
        hessian = torch.zeros(samples, count, total, total, dtype=DTYPE)
        hessian[:, :, :hidden_dim, hidden_dim:] = hidden_basis.permute(0, 1, 3, 2)
        hessian[:, :, hidden_dim:, :hidden_dim] = hidden_basis
        for coordinate in range(hidden_dim):
            shift = torch.zeros(hidden_dim, dtype=DTYPE)
            shift[coordinate] = hessian_step
            difference = (
                features(xi + shift).hidden_basis - features(xi - shift).hidden_basis
            ) / (2.0 * hessian_step)
            hessian[:, :, coordinate, :hidden_dim] = torch.einsum("snqh,sq->snh", difference, a)
        block = hessian[:, :, :hidden_dim, :hidden_dim]
        hessian[:, :, :hidden_dim, :hidden_dim] = 0.5 * (block + block.transpose(-1, -2))
        result["Hess"] = hessian
    return result


# ---------------------------------------------------------------------------
# Sampled envelope constants (lower bounds of the suprema)


def support_points(samples: int, dimension: int, generator: torch.Generator) -> torch.Tensor:
    """Uniform points of ``[-1, 1]^dimension``."""
    return 2.0 * torch.rand(samples, dimension, generator=generator, dtype=DTYPE) - 1.0


def support_corners(dimension: int) -> torch.Tensor:
    return torch.tensor(list(itertools.product((-1.0, 1.0), repeat=dimension)), dtype=DTYPE)


def tube_points(
    samples: int, dimension: int, radius: float, generator: torch.Generator
) -> torch.Tensor:
    """Points ``z_0 + radius * u`` with ``z_0`` in the support and ``||u||_2 = 1``.

    Even rows start from a uniform ``z_0``.  Odd rows start from a uniform
    corner of the support and move along a direction with the same signs, which
    reaches the part of the tube farthest from the origin.
    """
    base = support_points(samples, dimension, generator)
    direction = torch.randn(samples, dimension, generator=generator, dtype=DTYPE)
    direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    corner = torch.sign(base[1::2])
    corner[corner == 0] = 1.0
    base[1::2] = corner
    direction[1::2] = direction[1::2].abs() * corner
    return base + radius * direction


def envelope_constants(compiled: CompiledConstraints, config: Mapping[str, Any]) -> dict[str, Any]:
    dimension = compiled.input_dim + 1 + compiled.output_dim
    count = compiled.count
    chunk = int(config["sample_chunk"])
    step = float(config["finite_difference_step"])
    generator = torch.Generator().manual_seed(int(config["seed"]))

    a0 = torch.zeros(count, dtype=DTYPE)
    r_k = 0.0
    corners = support_corners(dimension)
    remaining = int(config["support_samples"])
    batches = [corners]
    while remaining > 0:
        size = min(chunk, remaining)
        batches.append(support_points(size, dimension, generator))
        remaining -= size
    for points in batches:
        values = per_neuron(compiled, points)
        a0 = torch.maximum(a0, values["h"].abs().max(dim=0).values)
        r_k = max(r_k, float(torch.linalg.matrix_norm(values["Psi"], ord=2).max() ** 2))

    tube: dict[str, dict[str, float]] = {}
    for radius in config["tube_radii"]:
        gradient = torch.zeros(count, dtype=DTYPE)
        curvature = torch.zeros(count, dtype=DTYPE)
        remaining = int(config["tube_samples"])
        while remaining > 0:
            size = min(chunk, remaining)
            points = tube_points(size, dimension, float(radius), generator)
            values = per_neuron(compiled, points, hessian_step=step)
            gradient = torch.maximum(
                gradient, torch.linalg.vector_norm(values["Psi"], dim=-1).max(dim=0).values
            )
            curvature = torch.maximum(
                curvature, torch.linalg.matrix_norm(values["Hess"], ord=2).max(dim=0).values
            )
            remaining -= size
        tube[repr(float(radius))] = {
            "B_star_squared": float((gradient**2).sum()),
            "L_star_squared": float((curvature**2).sum()),
        }
    return {"A0_star_squared": float((a0**2).sum()), "R_K": r_k, "tube": tube}


# ---------------------------------------------------------------------------
# Width terms


def width_terms(
    *,
    count: int,
    delta: float,
    radius: float,
    lambda0: float,
    r_k: float,
    b_star_squared: float,
    l_star_squared: float,
    residual_bound: float,
) -> dict[str, float]:
    """The three terms of ``eqn:linear_pde_convergence_conditions`` with ``Lambda = B_*^2``."""
    if lambda0 <= 0.0:
        raise AuditError("the width terms need a positive gap lambda_0")
    return {
        "chernoff": r_k * math.log(2.0 * count / delta) / (C_CHERNOFF * lambda0),
        "drift": DRIFT_PREFACTOR * l_star_squared * b_star_squared * residual_bound**2 / lambda0**3,
        "localization": 16.0 * b_star_squared * residual_bound**2 / (radius**2 * lambda0**2),
    }


def gap_needed_for_width(
    width: float,
    *,
    count: int,
    delta: float,
    radius: float,
    r_k: float,
    b_star_squared: float,
    l_star_squared: float,
    residual_bound: float,
) -> float:
    """Smallest ``lambda_0`` for which all three terms are at most ``width``."""
    unit = width_terms(
        count=count,
        delta=delta,
        radius=radius,
        lambda0=1.0,
        r_k=r_k,
        b_star_squared=b_star_squared,
        l_star_squared=l_star_squared,
        residual_bound=residual_bound,
    )
    return max(
        unit["chernoff"] / width,
        (unit["drift"] / width) ** (1.0 / 3.0),
        (unit["localization"] / width) ** 0.5,
    )


def residual_bound(target_norm: float, h0: float, a0_star: float, delta: float) -> float:
    return target_norm + h0 + a0_star * math.sqrt(2.0 * math.log(2.0 / delta))


# ---------------------------------------------------------------------------


def design_record(name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    compiled = compile_constraints(get_problem(name))
    count = compiled.count
    delta = float(config["delta"])
    coarse, fine = (int(order) for order in config["quadrature_orders"][str(compiled.input_dim + 1)])
    chunk = int(config["quadrature_chunk"])
    kernel_coarse, _ = limiting_kernel(compiled, coarse, chunk)
    kernel, kernel_out = limiting_kernel(compiled, fine, chunk)
    eigenvalues = torch.linalg.eigvalsh(kernel)
    lambda_min, lambda_max = float(eigenvalues[0]), float(eigenvalues[-1])
    change = abs(float(torch.linalg.eigvalsh(kernel_coarse)[0]) - lambda_min)
    if lambda_min <= 0.0:
        raise AuditError(f"{name}: the quadrature kernel is not positive definite")
    if change > float(config["quadrature_relative_tolerance"]) * lambda_min:
        raise AuditError(
            f"{name}: lambda_min changes by {change:.3e} between quadrature orders "
            f"{coarse} and {fine} (lambda_min = {lambda_min:.3e})"
        )

    envelope = envelope_constants(compiled, config)
    a0_star = math.sqrt(envelope["A0_star_squared"])
    h0 = math.sqrt(OUTPUT_WEIGHT_SECOND_MOMENT * float(torch.trace(kernel_out)))
    target_norm = float(torch.linalg.vector_norm(compiled.targets))
    rbar = residual_bound(target_norm, h0, a0_star, delta)
    largest_tested = float(max(int(width) for width in config["tested_widths"]))

    per_radius: dict[str, Any] = {}
    for key, constants in envelope["tube"].items():
        arguments = {
            "count": count,
            "delta": delta,
            "radius": float(key),
            "r_k": envelope["R_K"],
            "b_star_squared": constants["B_star_squared"],
            "l_star_squared": constants["L_star_squared"],
            "residual_bound": rbar,
        }
        terms = width_terms(lambda0=lambda_min, **arguments)
        per_radius[key] = {
            **constants,
            "terms": terms,
            "width_required": max(terms.values()),
            "gap_needed_for_largest_tested_width": gap_needed_for_width(largest_tested, **arguments),
        }
    best = min(per_radius, key=lambda key: per_radius[key]["width_required"])
    width_required = per_radius[best]["width_required"]
    gap_needed = min(item["gap_needed_for_largest_tested_width"] for item in per_radius.values())
    return {
        "N": count,
        "input_dim": compiled.input_dim,
        "output_dim": compiled.output_dim,
        "quadrature_orders": [coarse, fine],
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "lambda_min_output_block": float(torch.linalg.eigvalsh(kernel_out)[0]),
        "lambda_min_change_between_orders": change,
        "R_K": envelope["R_K"],
        "A0_star": a0_star,
        "H0": h0,
        "target_norm": target_norm,
        "residual_bound": rbar,
        "per_radius": per_radius,
        "best_radius": float(best),
        "width_required": width_required,
        "dominant_term": max(per_radius[best]["terms"], key=per_radius[best]["terms"].get),
        "width_required_over_largest_tested_width": width_required / largest_tested,
        "gap_needed_for_largest_tested_width": gap_needed,
        "gap_needed_exceeds_lambda_min": bool(gap_needed > lambda_min),
        "gap_needed_exceeds_lambda_max": bool(gap_needed > lambda_max),
        "seconds": time.perf_counter() - started,
    }


def build_record(config: Mapping[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(int(config["threads"]))
    designs = {name: design_record(name, config) for name in config["problems"]}
    declared, found = len(config["problems"]), len(designs)
    if found != declared:  # pragma: no cover - guarded by the comprehension above
        raise AuditError(f"computed {found} designs but the configuration declares {declared}")
    widths = [item["width_required"] for item in designs.values()]
    return {
        "designs": designs,
        "design_count": found,
        "largest_tested_width": max(int(width) for width in config["tested_widths"]),
        "smallest_width_required": min(widths),
        "largest_width_required": max(widths),
        "designs_within_tested_widths": [
            name for name, item in designs.items() if item["width_required_over_largest_tested_width"] <= 1.0
        ],
        "designs_where_gap_needed_exceeds_lambda_max": [
            name for name, item in designs.items() if item["gap_needed_exceeds_lambda_max"]
        ],
    }


DISPLAY_NAMES = {
    "poisson_1d": "Poisson 1D",
    "variable_elliptic_2d": "Variable elliptic 2D",
    "heat_1d": "Heat 1D",
    "transport_1d": "Transport 1D",
    "wave_1d": "Wave 1D",
    "biharmonic_2d": "Biharmonic 2D",
    "stokes_2d": "Stokes 2D",
    "weak_poisson_1d": "Weak Poisson 1D",
    "nonlocal_diffusion_1d": "Nonlocal diffusion 1D",
    "integro_diff_1d": "Integro-differential 1D",
    "caputo_diffusion_1d": "Caputo diffusion 1D",
}


def scientific(value: float) -> str:
    """``value`` to three significant digits as LaTeX, e.g. ``$1.72\\times10^{18}$``."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def tex_table(record: Mapping[str, Any]) -> str:
    """Rows of the table in ``width_requirement.tex``, one line per design."""
    lines = [
        f"% Generated by experiments/theorem_audit/width_requirement.py from the configuration with "
        f"SHA256 {record['config_sha256']}; do not edit numerical values manually.",
        r"\begin{tabular}{@{}lrcccccc@{}}",
        r"\toprule",
        r"Design & $N$ & $\lambda_{\min}(\mathbf K_{(0)}^\infty)$ & $\lambda_{\max}(\mathbf K_{(0)}^\infty)$"
        r" & $T_1$ & $T_2$ & $T_3$ & $\lambda_0^{\dagger}$ \\",
        r"\midrule",
    ]
    for name, item in record["designs"].items():
        terms = item["per_radius"][repr(item["best_radius"])]["terms"]
        cells = [
            DISPLAY_NAMES.get(name, name.replace("_", r"\_")),
            str(item["N"]),
            scientific(item["lambda_min"]),
            scientific(item["lambda_max"]),
            scientific(terms["chernoff"]),
            scientific(terms["drift"]),
            scientific(terms["localization"]),
            scientific(item["gap_needed_for_largest_tested_width"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


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
        f"{'design':<24}{'N':>3} {'lambda_min':>11} {'lambda_max':>11} {'Chernoff':>10} "
        f"{'drift':>10} {'localiz.':>10} {'width':>10} {'gap needed':>11}"
    )
    for name, item in result["designs"].items():
        terms = item["per_radius"][repr(item["best_radius"])]["terms"]
        print(
            f"{name:<24}{item['N']:>3} {item['lambda_min']:>11.3e} {item['lambda_max']:>11.3e} "
            f"{terms['chernoff']:>10.2e} {terms['drift']:>10.2e} {terms['localization']:>10.2e} "
            f"{item['width_required']:>10.2e} {item['gap_needed_for_largest_tested_width']:>11.3e}"
        )
    print(
        f"\n{result['design_count']} of {len(config['problems'])} designs computed; "
        f"largest tested width {result['largest_tested_width']}; required width between "
        f"{result['smallest_width_required']:.2e} and {result['largest_width_required']:.2e}; "
        f"designs within the tested widths: {len(result['designs_within_tested_widths'])}"
    )
    print(f"wrote {destination.relative_to(PROJECT_ROOT).as_posix()} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
