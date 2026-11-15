"""Reproduce every constant of ``ex:fully_numerical_gd_certificate``.

The example instantiates ``thm:linearPDEsconvergence`` for one row (``N = 1``),
the continuum mean observation

    lambda(f) = (1 / (2 X)) * int_{-X}^{X} f(x) dx,        target v_T = 0,

of the two-layer network of ``def:ntk_parameterization`` with ``sigma = tanh``
and ``d_0 = d_2 = 1``.  The per-neuron prediction is ``h(z) = a F(w, b)`` with
``z = (w, b, a)`` and ``F(w, b) = lambda(tanh(w x + gamma b))``.  Neurons are
i.i.d. with independent coordinates

    w ~ Unif[-W, W],    b ~ Unif[b_low, b_high],    a ~ Unif[-A, A].

Derivation (same order as the example; ``m_1 = X/2`` and ``m_2 = X^2/3`` are
the first absolute and second moments of the mean over ``[-X, X]``):

1. gap:        ``w x + gamma b >= gamma b_low - W X > 0`` on the support, so
               ``K_infty >= E[F^2] >= tanh(gamma b_low - W X)^2 =: lambda_0``.
2. first order: ``|F| <= 1``, ``|F_w| <= m_1``, ``|F_b| <= gamma`` because
               ``|tanh| <= 1`` and ``|tanh'| <= 1``.  With
               ``g = m_1^2 + gamma^2`` and ``grad h = (a F_w, a F_b, F)``:
               ``R_K <= 1 + A^2 g`` on the support and
               ``B_*^2 = 1 + (A + R)^2 g`` on the radius-``R`` tube.
3. second order: ``|tanh''| <= c_2 = 4 / (3 sqrt 3)`` gives
               ``|F_ww| <= c_2 m_2``, ``|F_wb| <= gamma c_2 m_1``,
               ``|F_bb| <= gamma^2 c_2``, and the Frobenius norm of the Hessian
               of ``h`` on the tube is bounded by
               ``L_*^2 = (A + R)^2 c_2^2 (m_2^2 + 2 gamma^2 m_1^2 + gamma^4) + 2 g``.
4. residual:   ``A_{0,*} = A``, ``H_0^2 = E[h^2] <= E[a^2] = A^2 / 3``, so
               ``rbar_delta <= ||v_T|| + A / sqrt 3 + A sqrt(2 log(2 / delta)) =: r_hat``.
5. width:      the three terms of ``eqn:linear_pde_convergence_conditions`` with
               ``Lambda = B_*^2`` and ``c_Ch = 1/4 + (3/4) log(3/4)``;
               step size ``eta <= 1 / (2 Lambda)``.

Rational constants are computed exactly with ``fractions.Fraction``.  Every
transcendental quantity is enclosed by outward-rounded interval arithmetic
(``mpmath.iv``), and each comparison with a value printed in the paper is
decided on exact rational endpoints.  The envelope bounds of steps 1-4 are
analytic; ``sampled_checks`` additionally evaluates the left-hand sides on
random parameters, which is an empirical check of the derivation and not a
proof.

All settings come from the configuration file.  Run::

    uv run python -m experiments.theorem_audit.example_gd_certificate \\
        --config experiments/theorem_audit/configs/example_gd_certificate_smoke.json
    uv run python -m experiments.theorem_audit.example_gd_certificate \\
        --config experiments/theorem_audit/configs/example_gd_certificate.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import mpmath
import numpy as np
from mpmath import iv

from .run import AuditError, PROJECT_ROOT, canonical_config_hash

SCHEMA_VERSION = 1
SUMMARY_NAME = "example_gd_certificate_summary.json"
NUMBER_OF_ROWS = 1
REQUIRED_FIELDS = (
    "output_dir",
    "gamma",
    "domain_half_width",
    "hidden_weight_bound",
    "hidden_bias_low",
    "hidden_bias_high",
    "output_weight_bound",
    "tube_radius",
    "delta",
    "target_norm",
    "interval_decimal_digits",
    "report_decimal_digits",
    "check_samples",
    "check_seed",
    "check_quadrature_nodes",
    "reported",
)
REPORTED_FIELDS = (
    "lambda0_lower",
    "R_K",
    "B_star_squared",
    "L_star_squared_exact",
    "L_star_squared",
    "r_hat_upper",
    "width_terms_upper",
    "width",
    "eta_max",
    "r_hat_rounded",
    "second_term_with_rounded_r_hat",
    "width_with_rounded_r_hat",
)


def _fraction(config: Mapping[str, Any], key: str) -> Fraction:
    value = config[key]
    if not isinstance(value, str):
        raise AuditError(f"{key} must be a decimal or rational string, got {value!r}")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise AuditError(f"{key} is not a decimal or rational string: {value!r}") from error


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
    for key in REPORTED_FIELDS:
        if key not in config["reported"]:
            raise AuditError(f"configuration is missing required field reported.{key!r}")
    validate_setting(config)
    return config


def validate_setting(config: Mapping[str, Any]) -> None:
    """Reject settings for which the analytic derivation does not apply."""
    gamma = _fraction(config, "gamma")
    half_width = _fraction(config, "domain_half_width")
    weight = _fraction(config, "hidden_weight_bound")
    bias_low = _fraction(config, "hidden_bias_low")
    bias_high = _fraction(config, "hidden_bias_high")
    output = _fraction(config, "output_weight_bound")
    radius = _fraction(config, "tube_radius")
    delta = _fraction(config, "delta")
    if min(gamma, half_width, weight, output, radius) <= 0:
        raise AuditError("gamma, domain_half_width, the weight bounds, and tube_radius must be positive")
    if not bias_low < bias_high:
        raise AuditError("hidden_bias_low must be smaller than hidden_bias_high")
    if not 0 < delta < 1:
        raise AuditError("delta must lie in (0, 1)")
    if _fraction(config, "target_norm") < 0:
        raise AuditError("target_norm must be nonnegative")
    if gamma * bias_low - weight * half_width <= 0:
        raise AuditError(
            "the gap bound needs gamma * hidden_bias_low > hidden_weight_bound * domain_half_width"
        )
    if int(config["check_samples"]) < 1 or int(config["check_quadrature_nodes"]) < 2:
        raise AuditError("check_samples and check_quadrature_nodes are too small")
    if len(config["reported"]["width_terms_upper"]) != 3:
        raise AuditError("reported.width_terms_upper must contain exactly three entries")


# ---------------------------------------------------------------------------
# Exact and interval arithmetic


def _interval(value: Fraction):
    """Interval enclosing the rational ``value`` (integers convert exactly)."""
    return iv.mpf(value.numerator) / iv.mpf(value.denominator)


def _endpoint(value) -> Fraction:
    """Exact rational value of a binary floating-point interval endpoint."""
    sign, mantissa, exponent, _ = mpmath.mpf(value)._mpf_
    result = Fraction(int(mantissa)) * Fraction(2) ** int(exponent)
    return -result if sign else result


def _bounds(value) -> tuple[Fraction, Fraction]:
    if isinstance(value, Fraction):
        return value, value
    return _endpoint(value.a), _endpoint(value.b)


def _decimal(value: Fraction, digits: int, *, upward: bool) -> str:
    """Decimal string of ``value`` rounded toward -inf or +inf at ``digits`` places."""
    scaled = value * 10**digits
    integer = math.ceil(scaled) if upward else math.floor(scaled)
    sign = "-" if integer < 0 else ""
    whole, fractional = divmod(abs(integer), 10**digits)
    return f"{sign}{whole}.{fractional:0{digits}d}"


def _tanh(value):
    exponential = iv.exp(2 * value)
    return (exponential - 1) / (exponential + 1)


def exact_constants(config: Mapping[str, Any]) -> dict[str, Fraction]:
    """The constants of the example that are rational in the settings."""
    gamma = _fraction(config, "gamma")
    half_width = _fraction(config, "domain_half_width")
    output = _fraction(config, "output_weight_bound")
    radius = _fraction(config, "tube_radius")
    first_moment = half_width / 2
    second_moment = half_width**2 / 3
    first_order = first_moment**2 + gamma**2
    tube_output = output + radius
    second_derivative_sup_squared = Fraction(16, 27)  # (4 / (3 sqrt 3))^2 = sup |tanh''|^2
    hessian = (
        tube_output**2
        * second_derivative_sup_squared
        * (second_moment**2 + 2 * gamma**2 * first_moment**2 + gamma**4)
        + 2 * first_order
    )
    b_star_squared = 1 + tube_output**2 * first_order
    return {
        "first_absolute_moment": first_moment,
        "second_moment": second_moment,
        "R_K": 1 + output**2 * first_order,
        "B_star_squared": b_star_squared,
        "L_star_squared": hessian,
        "A_0_star": output,
        "eta_max": 1 / (2 * b_star_squared),
    }


def interval_constants(config: Mapping[str, Any]) -> dict[str, Any]:
    """Enclosures of the transcendental constants of the example."""
    iv.dps = int(config["interval_decimal_digits"])
    gamma = _fraction(config, "gamma")
    half_width = _fraction(config, "domain_half_width")
    weight = _fraction(config, "hidden_weight_bound")
    bias_low = _fraction(config, "hidden_bias_low")
    output = _interval(_fraction(config, "output_weight_bound"))
    delta = _interval(_fraction(config, "delta"))
    smallest_argument = _interval(gamma * bias_low - weight * half_width)
    lambda0 = _tanh(smallest_argument) ** 2
    r_hat = (
        _interval(_fraction(config, "target_norm"))
        + output / iv.sqrt(iv.mpf(3))
        + output * iv.sqrt(2 * iv.log(2 / delta))
    )
    return {
        "lambda0": lambda0,
        "r_hat": r_hat,
        "c_Ch": iv.mpf(1) / 4 + iv.mpf(3) / 4 * iv.log(iv.mpf(3) / 4),
        "drift_prefactor": 320 + 128 * iv.sqrt(iv.mpf(6)),
        "log_2N_over_delta": iv.log(2 * NUMBER_OF_ROWS / delta),
    }


def width_terms(
    config: Mapping[str, Any],
    *,
    r_k: Fraction,
    b_star_squared: Fraction,
    l_star_squared: Fraction,
    r_hat=None,
) -> list[Any]:
    """The three width terms of ``eqn:linear_pde_convergence_conditions``."""
    enclosures = interval_constants(config)
    lambda0 = enclosures["lambda0"]
    residual = enclosures["r_hat"] if r_hat is None else r_hat
    radius = _interval(_fraction(config, "tube_radius"))
    big_lambda = _interval(b_star_squared)
    chernoff = _interval(r_k) * enclosures["log_2N_over_delta"] / (enclosures["c_Ch"] * lambda0)
    drift = (
        enclosures["drift_prefactor"]
        * _interval(l_star_squared)
        * big_lambda
        * residual**2
        / lambda0**3
    )
    localization = 16 * big_lambda * residual**2 / (radius**2 * lambda0**2)
    return [chernoff, drift, localization]


def certified_width(terms: list[Any]) -> int:
    """Smallest integer that provably dominates all three width terms."""
    return max(math.ceil(_bounds(term)[1]) for term in terms)


# ---------------------------------------------------------------------------
# Comparison with the values printed in the paper


def _check(name: str, relation: str, computed, reported_text: str, digits: int) -> dict[str, Any]:
    reported = Fraction(reported_text)
    lower, upper = _bounds(computed)
    if relation == "computed > reported":
        holds = lower > reported
    elif relation == "computed < reported":
        holds = upper < reported
    elif relation == "computed <= reported":
        holds = upper <= reported
    elif relation == "computed == reported":
        holds = lower == upper == reported
    else:  # pragma: no cover - programming error
        raise AuditError(f"unknown relation {relation!r}")
    return {
        "name": name,
        "relation": relation,
        "computed_lower": _decimal(lower, digits, upward=False),
        "computed_upper": _decimal(upper, digits, upward=True),
        "reported": reported_text,
        "holds": bool(holds),
    }


def reported_checks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    reported = config["reported"]
    digits = int(config["report_decimal_digits"])
    exact = exact_constants(config)
    enclosures = interval_constants(config)

    def value(key: str) -> Fraction:
        return Fraction(reported[key])

    checks = [
        _check("lambda_0 = tanh(gamma b_low - W X)^2", "computed > reported", enclosures["lambda0"], reported["lambda0_lower"], digits),
        _check("R_K", "computed <= reported", exact["R_K"], reported["R_K"], digits),
        _check("B_*^2", "computed <= reported", exact["B_star_squared"], reported["B_star_squared"], digits),
        _check("L_*^2 (exact rational)", "computed == reported", exact["L_star_squared"], reported["L_star_squared_exact"], digits),
        _check("L_*^2 (value used in the width)", "computed < reported", exact["L_star_squared"], reported["L_star_squared"], digits),
        _check("r_hat", "computed < reported", enclosures["r_hat"], reported["r_hat_upper"], digits),
        _check("eta_max = 1 / (2 B_*^2)", "computed == reported", exact["eta_max"], reported["eta_max"], digits),
    ]

    # The paper substitutes its printed constants R_K, B_*^2, L_*^2 and the
    # exact expressions for lambda_0 and r_hat.
    terms = width_terms(
        config,
        r_k=value("R_K"),
        b_star_squared=value("B_star_squared"),
        l_star_squared=value("L_star_squared"),
    )
    names = ("Chernoff term", "drift term", "localization term")
    for name, term, bound in zip(names, terms, reported["width_terms_upper"]):
        checks.append(_check(f"width: {name}", "computed < reported", term, bound, digits))
    width = certified_width(terms)
    largest_lower = max(_bounds(term)[0] for term in terms)
    checks.append(
        {
            "name": "smallest integer width satisfying the three terms",
            "relation": "computed == reported",
            "computed_lower": str(width),
            "computed_upper": str(width),
            "reported": str(int(reported["width"])),
            "holds": width == int(reported["width"]) and largest_lower > width - 1,
        }
    )

    # The remark on display rounding: replacing r_hat by its printed rounding.
    rounded_terms = width_terms(
        config,
        r_k=value("R_K"),
        b_star_squared=value("B_star_squared"),
        l_star_squared=value("L_star_squared"),
        r_hat=_interval(value("r_hat_rounded")),
    )
    rounded_lower, rounded_upper = _bounds(rounded_terms[1])
    printed = value("second_term_with_rounded_r_hat")
    half_unit = Fraction(1, 2) * Fraction(1, 10) ** 2
    rounded_width = certified_width(rounded_terms)
    checks.append(
        {
            "name": "drift term with r_hat replaced by its printed rounding, to two decimals",
            "relation": "|computed - reported| < 0.005",
            "computed_lower": _decimal(rounded_lower, digits, upward=False),
            "computed_upper": _decimal(rounded_upper, digits, upward=True),
            "reported": reported["second_term_with_rounded_r_hat"],
            "holds": printed - half_unit < rounded_lower and rounded_upper < printed + half_unit,
        }
    )
    checks.append(
        {
            "name": "integer width with r_hat replaced by its printed rounding",
            "relation": "computed == reported",
            "computed_lower": str(rounded_width),
            "computed_upper": str(rounded_width),
            "reported": str(int(reported["width_with_rounded_r_hat"])),
            "holds": rounded_width == int(reported["width_with_rounded_r_hat"]),
        }
    )
    return checks


# ---------------------------------------------------------------------------
# Empirical check of the analytic envelopes (float64, not a proof)


def feature_derivatives(w: np.ndarray, b: np.ndarray, config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """``F`` and its derivatives through order two by Gauss-Legendre quadrature."""
    gamma = float(_fraction(config, "gamma"))
    half_width = float(_fraction(config, "domain_half_width"))
    nodes, weights = np.polynomial.legendre.leggauss(int(config["check_quadrature_nodes"]))
    x = half_width * nodes[None, :]
    mean = 0.5 * weights[None, :]
    t = np.tanh(w[:, None] * x + gamma * b[:, None])
    first = 1.0 - t * t
    second = -2.0 * t * first
    return {
        "F": np.sum(mean * t, axis=1),
        "F_w": np.sum(mean * x * first, axis=1),
        "F_b": gamma * np.sum(mean * first, axis=1),
        "F_ww": np.sum(mean * x * x * second, axis=1),
        "F_wb": gamma * np.sum(mean * x * second, axis=1),
        "F_bb": gamma * gamma * np.sum(mean * second, axis=1),
    }


def sampled_checks(config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the bounded quantities on random support and tube points."""
    exact = exact_constants(config)
    lambda0_lower = float(_bounds(interval_constants(config)["lambda0"])[0])
    weight = float(_fraction(config, "hidden_weight_bound"))
    bias_low = float(_fraction(config, "hidden_bias_low"))
    bias_high = float(_fraction(config, "hidden_bias_high"))
    output = float(_fraction(config, "output_weight_bound"))
    radius = float(_fraction(config, "tube_radius"))
    samples = int(config["check_samples"])
    generator = np.random.default_rng(int(config["check_seed"]))

    w0 = generator.uniform(-weight, weight, samples)
    b0 = generator.uniform(bias_low, bias_high, samples)
    a0 = generator.uniform(-output, output, samples)
    # Include the corners of the support, where the envelopes are approached.
    corners = np.array(
        [(sw * weight, sb, sa * output) for sw in (-1, 0, 1) for sb in (bias_low, bias_high) for sa in (-1, 1)]
    )
    w0 = np.concatenate([corners[:, 0], w0])
    b0 = np.concatenate([corners[:, 1], b0])
    a0 = np.concatenate([corners[:, 2], a0])

    support = feature_derivatives(w0, b0, config)
    gradient_support = a0**2 * (support["F_w"] ** 2 + support["F_b"] ** 2) + support["F"] ** 2

    # Tube points z = z_0 + R u with u on the unit sphere; half of them push the
    # output weight to the extreme |a| = A + R, where B_* and L_* are approached.
    direction = generator.standard_normal((w0.size, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    direction[::2] = np.array([0.0, 0.0, 1.0]) * np.sign(a0[::2] + (a0[::2] == 0))[:, None]
    w1 = w0 + radius * direction[:, 0]
    b1 = b0 + radius * direction[:, 1]
    a1 = a0 + radius * direction[:, 2]
    tube = feature_derivatives(w1, b1, config)
    gradient_tube = a1**2 * (tube["F_w"] ** 2 + tube["F_b"] ** 2) + tube["F"] ** 2
    hessian = np.zeros((w1.size, 3, 3))
    hessian[:, 0, 0] = a1 * tube["F_ww"]
    hessian[:, 0, 1] = hessian[:, 1, 0] = a1 * tube["F_wb"]
    hessian[:, 1, 1] = a1 * tube["F_bb"]
    hessian[:, 0, 2] = hessian[:, 2, 0] = tube["F_w"]
    hessian[:, 1, 2] = hessian[:, 2, 1] = tube["F_b"]
    hessian_operator_squared = np.max(np.abs(np.linalg.eigvalsh(hessian)), axis=1) ** 2

    def entry(observed: float, bound: float, relation: str) -> dict[str, Any]:
        holds = observed >= bound if relation == "observed >= bound" else observed <= bound
        return {"observed": float(observed), "bound": float(bound), "relation": relation, "holds": bool(holds)}

    interior = slice(corners.shape[0], None)  # i.i.d. draws only, for the two means
    return {
        "samples": samples,
        "corner_points": int(corners.shape[0]),
        "seed": int(config["check_seed"]),
        "quadrature_nodes": int(config["check_quadrature_nodes"]),
        "min_F_squared_on_support_vs_lambda0": entry(np.min(support["F"] ** 2), lambda0_lower, "observed >= bound"),
        "max_gradient_norm_squared_on_support_vs_R_K": entry(np.max(gradient_support), float(exact["R_K"]), "observed <= bound"),
        "max_gradient_norm_squared_on_tube_vs_B_star_squared": entry(np.max(gradient_tube), float(exact["B_star_squared"]), "observed <= bound"),
        "max_hessian_operator_norm_squared_on_tube_vs_L_star_squared": entry(np.max(hessian_operator_squared), float(exact["L_star_squared"]), "observed <= bound"),
        "max_abs_h_on_support_vs_A_0_star": entry(np.max(np.abs(a0 * support["F"])), float(exact["A_0_star"]), "observed <= bound"),
        "mean_h_squared_vs_A_squared_over_3": entry(np.mean((a0 * support["F"])[interior] ** 2), output**2 / 3.0, "observed <= bound"),
        "monte_carlo_limiting_kernel": float(np.mean(gradient_support[interior])),
    }


# ---------------------------------------------------------------------------


def build_record(config: Mapping[str, Any]) -> dict[str, Any]:
    digits = int(config["report_decimal_digits"])
    exact = exact_constants(config)
    enclosures = interval_constants(config)
    terms = width_terms(
        config,
        r_k=exact["R_K"],
        b_star_squared=exact["B_star_squared"],
        l_star_squared=exact["L_star_squared"],
    )

    def enclosure(value) -> dict[str, str]:
        lower, upper = _bounds(value)
        return {"lower": _decimal(lower, digits, upward=False), "upper": _decimal(upper, digits, upward=True)}

    checks = reported_checks(config)
    sampled = sampled_checks(config)
    sampled_hold = all(item["holds"] for item in sampled.values() if isinstance(item, dict))
    return {
        "exact_constants": {key: str(value) for key, value in exact.items()},
        "interval_constants": {key: enclosure(value) for key, value in enclosures.items()},
        "width_terms_with_exact_constants": {
            "chernoff": enclosure(terms[0]),
            "drift": enclosure(terms[1]),
            "localization": enclosure(terms[2]),
            "certified_integer_width": certified_width(terms),
        },
        "reported_checks": checks,
        "all_reported_checks_hold": all(item["holds"] for item in checks),
        "sampled_checks": sampled,
        "all_sampled_checks_hold": sampled_hold,
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
            "mpmath": mpmath.__version__,
            "numpy": np.__version__,
            "git": _git_metadata(),
        },
    }

    output_dir = PROJECT_ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / SUMMARY_NAME
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"{'claim':<72} {'computed (outward-rounded)':<46} {'reported':>14}  ok")
    for item in result["reported_checks"]:
        computed = (
            item["computed_lower"]
            if item["computed_lower"] == item["computed_upper"]
            else f"[{item['computed_lower'][:20]}, {item['computed_upper'][:20]}]"
        )
        print(f"{item['name']:<72} {computed:<46} {item['reported']:>14}  {item['holds']}")
    print()
    for name, item in result["sampled_checks"].items():
        if isinstance(item, dict):
            print(f"sampled {name}: {item['observed']:.6g} vs {item['bound']:.6g} ({item['relation']}): {item['holds']}")
    print(f"sampled Monte Carlo limiting kernel E|grad h|^2: {result['sampled_checks']['monte_carlo_limiting_kernel']:.6f}")
    print(f"\nwrote {destination.relative_to(PROJECT_ROOT).as_posix()} in {elapsed:.2f}s")

    if not (result["all_reported_checks_hold"] and result["all_sampled_checks_hold"]):
        print("FAILED: at least one check does not hold")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
