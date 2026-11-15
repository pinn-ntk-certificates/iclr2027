"""Numerical-contract tests for :mod:`width_requirement`."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from .features import compile_constraints, evaluate_features
from .problems import get_problem, list_problems
from .run import AuditError, PROJECT_ROOT
from .width_requirement import (
    C_CHERNOFF,
    DRIFT_PREFACTOR,
    SCHEMA_VERSION,
    SUMMARY_NAME,
    TABLE_NAME,
    scientific,
    tex_table,
    build_record,
    design_record,
    envelope_constants,
    gap_needed_for_width,
    limiting_kernel,
    load_config,
    per_neuron,
    residual_bound,
    support_corners,
    support_points,
    tube_points,
    width_terms,
)

PAPER_CONFIG = PROJECT_ROOT / "experiments/theorem_audit/configs/width_requirement.json"
SMOKE_CONFIG = PROJECT_ROOT / "experiments/theorem_audit/configs/width_requirement_smoke.json"
DTYPE = torch.float64


def _tiny(**override) -> dict:
    config = load_config(SMOKE_CONFIG)
    config.update(
        {
            "problems": ["poisson_1d", "nonlocal_diffusion_1d"],
            "quadrature_orders": {"2": [48, 64], "3": [16, 24]},
            "quadrature_relative_tolerance": 1e-4,
            "support_samples": 600,
            "tube_samples": 400,
            "sample_chunk": 400,
            "threads": 1,
        }
    )
    config.update(override)
    return config


def _write(config: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(config, handle)
    handle.close()
    return Path(handle.name)


def _prediction(compiled, z: torch.Tensor, row: int) -> torch.Tensor:
    """``h_row(z)`` for one neuron, differentiable in ``z`` by autograd."""
    hidden = compiled.input_dim + 1
    batch = evaluate_features(compiled, z[None, : compiled.input_dim], z[None, compiled.input_dim])
    return batch.output_basis[0, row] @ z[hidden:]


class ConfigValidation(unittest.TestCase):
    def test_rejects_wrong_schema_version(self) -> None:
        with self.assertRaises(AuditError):
            load_config(_write({**_tiny(), "schema_version": SCHEMA_VERSION + 1}))

    def test_rejects_missing_field(self) -> None:
        broken = {key: value for key, value in _tiny().items() if key != "tested_widths"}
        with self.assertRaises(AuditError):
            load_config(_write(broken))

    def test_rejects_unknown_or_repeated_problem(self) -> None:
        for problems in (["no_such_design"], ["poisson_1d", "poisson_1d"], []):
            with self.assertRaises(AuditError):
                load_config(_write({**_tiny(), "problems": problems}))

    def test_rejects_bad_numerical_settings(self) -> None:
        for key, value in (
            ("delta", 0.0),
            ("delta", 1.0),
            ("tube_radii", []),
            ("tube_radii", [0.1, -0.5]),
            ("quadrature_orders", {"2": [64, 48], "3": [16, 24]}),
            ("quadrature_orders", {"3": [16, 24]}),
            ("finite_difference_step", 0.0),
            ("support_samples", 0),
            ("tested_widths", []),
        ):
            with self.assertRaises(AuditError, msg=f"{key}={value}"):
                load_config(_write({**_tiny(), key: value}))

    def test_paper_and_smoke_configs_share_the_setting(self) -> None:
        paper, smoke = load_config(PAPER_CONFIG), load_config(SMOKE_CONFIG)
        numerical = {
            "output_dir",
            "quadrature_orders",
            "quadrature_relative_tolerance",
            "support_samples",
            "tube_samples",
            "sample_chunk",
        }
        for key in paper:
            if key not in numerical:
                self.assertEqual(paper[key], smoke[key], key)
        self.assertEqual(tuple(paper["problems"]), tuple(list_problems()))


class LimitingKernel(unittest.TestCase):
    def test_quadrature_matches_monte_carlo_of_the_full_jacobian(self) -> None:
        # The closed-form integration over the output weights uses E[a^2] = 1/3.
        compiled = compile_constraints(get_problem("poisson_1d"))
        kernel, kernel_out = limiting_kernel(compiled, 64, 4096)
        generator = torch.Generator().manual_seed(0)
        z = support_points(400_000, 3, generator)
        psi = per_neuron(compiled, z)["Psi"]
        monte_carlo = torch.einsum("sip,sjp->ij", psi, psi) / z.shape[0]
        scale = float(torch.linalg.matrix_norm(kernel, ord=2))
        self.assertLess(float(torch.linalg.matrix_norm(monte_carlo - kernel, ord=2)), 2e-2 * scale)
        self.assertTrue(torch.equal(kernel, kernel.T))
        # K_infty - K_out is the hidden block, which is positive semidefinite.
        self.assertGreater(float(torch.linalg.eigvalsh(kernel - kernel_out)[0]), -1e-12)

    def test_orders_agree_and_kernel_is_positive_definite(self) -> None:
        compiled = compile_constraints(get_problem("poisson_1d"))
        coarse, _ = limiting_kernel(compiled, 96, 4096)
        fine, _ = limiting_kernel(compiled, 128, 4096)
        low, high = float(torch.linalg.eigvalsh(coarse)[0]), float(torch.linalg.eigvalsh(fine)[0])
        self.assertGreater(high, 0.0)
        self.assertLess(abs(low - high), 1e-8 * high)

    def test_chunking_does_not_change_the_kernel(self) -> None:
        compiled = compile_constraints(get_problem("weak_poisson_1d"))
        whole, _ = limiting_kernel(compiled, 24, 10**6)
        pieces, _ = limiting_kernel(compiled, 24, 100)
        self.assertLess(float((whole - pieces).abs().max()), 1e-13)


class PerNeuronDerivatives(unittest.TestCase):
    def test_jacobian_and_hessian_match_autograd(self) -> None:
        generator = torch.Generator().manual_seed(1)
        for name in ("poisson_1d", "stokes_2d", "nonlocal_diffusion_1d"):
            compiled = compile_constraints(get_problem(name))
            dimension = compiled.input_dim + 1 + compiled.output_dim
            z = 1.2 * support_points(3, dimension, generator)
            values = per_neuron(compiled, z, hessian_step=1e-4)
            for sample in range(z.shape[0]):
                for row in (0, compiled.count - 1):
                    point = z[sample].clone()
                    function = lambda v, row=row: _prediction(compiled, v, row)  # noqa: E731
                    gradient = torch.autograd.functional.jacobian(function, point)
                    hessian = torch.autograd.functional.hessian(function, point)
                    scale = 1.0 + float(hessian.abs().max())
                    self.assertLess(float((values["Psi"][sample, row] - gradient).abs().max()), 1e-10 * scale, name)
                    self.assertLess(float((values["Hess"][sample, row] - hessian).abs().max()), 1e-5 * scale, name)
                    self.assertAlmostEqual(float(values["h"][sample, row]), float(function(point)), delta=1e-12 * scale)


class Sampling(unittest.TestCase):
    def test_tube_points_lie_in_the_tube(self) -> None:
        generator = torch.Generator().manual_seed(2)
        for radius in (0.1, 1.0):
            points = tube_points(5001, 4, radius, generator)
            # Distance to the support [-1, 1]^4 is at most the radius.
            distance = torch.linalg.vector_norm((points.abs() - 1.0).clamp(min=0.0), dim=1)
            self.assertLessEqual(float(distance.max()), radius + 1e-12)
            # The corner rows reach distance exactly equal to the radius.
            self.assertAlmostEqual(float(distance[1::2].min()), radius, delta=1e-12)

    def test_support_points_and_corners(self) -> None:
        generator = torch.Generator().manual_seed(3)
        points = support_points(1000, 3, generator)
        self.assertLessEqual(float(points.abs().max()), 1.0)
        corners = support_corners(3)
        self.assertEqual(tuple(corners.shape), (8, 3))
        self.assertTrue(bool((corners.abs() == 1.0).all()))

    def test_envelopes_are_deterministic_and_monotone_in_the_radius(self) -> None:
        compiled = compile_constraints(get_problem("poisson_1d"))
        first = envelope_constants(compiled, _tiny())
        second = envelope_constants(compiled, _tiny())
        self.assertEqual(first, second)
        # R_K is a supremum of ||Psi||_2^2 <= ||Psi||_F^2 <= sum_i B_i^2 over a smaller set.
        self.assertLessEqual(first["R_K"], first["tube"]["1.0"]["B_star_squared"])
        self.assertGreater(first["A0_star_squared"], 0.0)

    def test_sampled_constants_do_not_exceed_a_denser_sample(self) -> None:
        # Not a proof of the lower-bound property, which holds because every
        # sampled point lies in the set of the supremum.  This checks that more
        # samples from the same stream can only raise the maxima.
        compiled = compile_constraints(get_problem("poisson_1d"))
        small = envelope_constants(compiled, _tiny(tube_radii=[0.5], support_samples=400, tube_samples=400))
        large = envelope_constants(compiled, _tiny(tube_radii=[0.5], support_samples=4000, tube_samples=400))
        self.assertLessEqual(small["R_K"], large["R_K"])
        self.assertLessEqual(small["A0_star_squared"], large["A0_star_squared"])


class WidthTerms(unittest.TestCase):
    ARGUMENTS = {
        "count": 1,
        "delta": 0.05,
        "radius": 0.1,
        "r_k": 1.0125,
        "b_star_squared": 1.05,
        "l_star_squared": 2.5382,
        "residual_bound": 0.1 / math.sqrt(3.0) + 0.1 * math.sqrt(2.0 * math.log(40.0)),
    }

    def test_reproduces_the_worked_example_of_the_paper(self) -> None:
        # ex:fully_numerical_gd_certificate: 212.6122, 1355.9790, 692.2517.
        terms = width_terms(lambda0=math.tanh(0.9) ** 2, **self.ARGUMENTS)
        self.assertAlmostEqual(terms["chernoff"], 212.6122, delta=1e-3)
        self.assertAlmostEqual(terms["drift"], 1355.9790, delta=1e-3)
        self.assertAlmostEqual(terms["localization"], 692.2517, delta=1e-3)
        self.assertEqual(math.ceil(max(terms.values())), 1356)

    def test_constants(self) -> None:
        self.assertAlmostEqual(C_CHERNOFF, 0.0342384, delta=1e-7)
        self.assertAlmostEqual(DRIFT_PREFACTOR, 633.5347, delta=1e-4)
        self.assertAlmostEqual(residual_bound(2.0, 0.5, 1.0, 0.05), 2.5 + math.sqrt(2.0 * math.log(40.0)))

    def test_scaling_in_the_gap(self) -> None:
        one = width_terms(lambda0=0.5, **self.ARGUMENTS)
        two = width_terms(lambda0=0.25, **self.ARGUMENTS)
        self.assertAlmostEqual(two["chernoff"] / one["chernoff"], 2.0)
        self.assertAlmostEqual(two["drift"] / one["drift"], 8.0)
        self.assertAlmostEqual(two["localization"] / one["localization"], 4.0)

    def test_rejects_nonpositive_gap(self) -> None:
        with self.assertRaises(AuditError):
            width_terms(lambda0=0.0, **self.ARGUMENTS)

    def test_gap_needed_inverts_the_requirement(self) -> None:
        for width in (64.0, 16384.0, 1e9):
            gap = gap_needed_for_width(width, **self.ARGUMENTS)
            self.assertAlmostEqual(max(width_terms(lambda0=gap, **self.ARGUMENTS).values()) / width, 1.0, places=10)
            self.assertGreater(max(width_terms(lambda0=0.99 * gap, **self.ARGUMENTS).values()), width)


class Record(unittest.TestCase):
    def test_record_is_complete_consistent_and_serializable(self) -> None:
        config = _tiny()
        record = build_record(config)
        json.dumps(record, allow_nan=False)
        self.assertEqual(record["design_count"], len(config["problems"]))
        self.assertEqual(record["largest_tested_width"], 16384)
        for name in config["problems"]:
            item = record["designs"][name]
            self.assertGreater(item["lambda_min"], 0.0)
            self.assertLessEqual(item["lambda_min_output_block"], item["lambda_min"] * (1 + 1e-12))
            self.assertLessEqual(item["H0"], item["A0_star"] * 1.05)
            best = item["per_radius"][repr(item["best_radius"])]
            self.assertEqual(item["width_required"], max(best["terms"].values()))
            self.assertEqual(
                item["width_required"], min(entry["width_required"] for entry in item["per_radius"].values())
            )
            self.assertEqual(
                item["gap_needed_exceeds_lambda_min"],
                item["width_required_over_largest_tested_width"] > 1.0,
            )

    def test_scientific_formatting(self) -> None:
        self.assertEqual(scientific(1.7234e18), r"$1.72\times10^{18}$")
        self.assertEqual(scientific(2.4621e-4), r"$2.46\times10^{-4}$")
        self.assertEqual(scientific(9.996), r"$1.00\times10^{1}$")

    def test_tracked_table_matches_tracked_summary(self) -> None:
        # The LaTeX table and the JSON summary are written by the same run.
        config = load_config(PAPER_CONFIG)
        summary = PROJECT_ROOT / config["output_dir"] / SUMMARY_NAME
        table = PROJECT_ROOT / config["output_dir"] / TABLE_NAME
        if not summary.is_file():
            self.skipTest("the full configuration has not been run in this checkout")
        record = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(record["design_count"], len(config["problems"]))
        self.assertEqual(list(record["designs"]), config["problems"])
        self.assertEqual(record["config"], config)
        if table.is_file():
            self.assertEqual(table.read_text(encoding="utf-8"), tex_table(record))

    def test_quadrature_disagreement_fails_loudly(self) -> None:
        config = _tiny(quadrature_orders={"2": [2, 3], "3": [2, 3]}, quadrature_relative_tolerance=1e-12)
        with self.assertRaises(AuditError):
            design_record("poisson_1d", config)


if __name__ == "__main__":
    unittest.main()
