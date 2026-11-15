"""Numerical contract of ``depth_variation.py``."""

from __future__ import annotations

import copy
import math
import unittest

import torch

from experiments.theorem_audit import depth_variation as dv
from experiments.theorem_audit.features import compile_constraints, evaluate_features
from experiments.theorem_audit.problems import get_problem
from experiments.theorem_audit.run import AuditError

CONFIG = {
    "schema_version": 1,
    "output_dir": "experiments/theorem_audit/results/depth_variation_test",
    "problems": ["poisson_1d"],
    "depths": [1, 2],
    "widths": [8],
    "seeds": [0, 1],
    "bias_scale": 1.0,
    "steps": 20,
    "checkpoint_every": 10,
    "learning_rate_factor": 0.2,
    "gap_fraction": 0.5,
    "processes": 1,
    "threads_per_process": 1,
}


def two_layer_gram(compiled, params) -> torch.Tensor:
    """Gram matrix of the paper's two-layer model from the hand-derived features."""
    width = params["W1"].shape[0]
    batch = evaluate_features(compiled, params["W1"], params["b1"])
    a = params["Wout"].T  # [m, q]
    output = batch.output_basis.permute(1, 0, 2).reshape(compiled.count, -1)
    hidden = torch.einsum("mq,mnqh->nmh", a, batch.hidden_basis).reshape(compiled.count, -1)
    jacobian = torch.cat((output, hidden), dim=1) / math.sqrt(width)
    return jacobian @ jacobian.T


class NetworkTests(unittest.TestCase):
    def test_depth_one_matches_two_layer_features(self) -> None:
        for name in ("poisson_1d", "heat_1d"):
            problem = get_problem(name)
            compiled = compile_constraints(problem)
            params = dv.initial_parameters(problem.input_dim, problem.output_dim, 1, 12, 12, seed=3)
            kernel = dv.gram(dv.layer_jacobians(params, compiled, 1.0))
            reference = two_layer_gram(compiled, params)
            error = float(torch.linalg.matrix_norm(kernel - reference, ord=2))
            self.assertLess(error, 1e-11 * float(torch.linalg.matrix_norm(reference, ord=2)), name)

    def test_chain_rule_predictions_match_forward_mode_reference(self) -> None:
        for name in ("poisson_1d", "heat_1d", "variable_elliptic_2d"):
            problem = get_problem(name)
            compiled = compile_constraints(problem)
            for depth in (1, 2, 3):
                params = dv.initial_parameters(
                    problem.input_dim, problem.output_dim, depth, 7, 9, seed=depth
                )
                fast = dv.predictions(params, compiled, 0.7)
                reference = dv.predictions_reference(params, compiled, 0.7)
                self.assertLess(float(torch.max(torch.abs(fast - reference))), 1e-12, (name, depth))

    def test_orders_above_two_are_rejected(self) -> None:
        compiled = compile_constraints(get_problem("biharmonic_2d"))
        params = dv.initial_parameters(2, 1, 2, 4, 4, seed=0)
        with self.assertRaises(AuditError):
            dv.predictions(params, compiled, 1.0)
        with self.assertRaises(AuditError):
            dv.validate_config({**CONFIG, "problems": ["biharmonic_2d"]})

    def test_spatial_derivatives_match_finite_differences_at_depth(self) -> None:
        problem = get_problem("poisson_1d")
        compiled = compile_constraints(problem)
        params = dv.initial_parameters(1, 1, 3, 6, 6, seed=5)
        value = lambda x: float(dv.network(params, torch.tensor([x], dtype=dv.DTYPE), 1.0)[0])  # noqa: E731
        step = 1e-4
        expected = []
        for term in range(compiled.term_count):
            x = float(compiled.points[term, 0])
            if int(compiled.orders[term]) == 2:
                derivative = (value(x + step) - 2.0 * value(x) + value(x - step)) / step**2
            else:
                derivative = value(x)
            expected.append(float(compiled.coefficients[term]) * derivative)
        got = dv.predictions(params, compiled, 1.0)
        for index in range(compiled.count):
            # one term per row in this design
            self.assertAlmostEqual(float(got[index]), expected[index], delta=1e-6)

    def test_jacobian_matches_finite_differences_at_depth(self) -> None:
        problem = get_problem("heat_1d")
        compiled = compile_constraints(problem)
        params = dv.initial_parameters(2, 1, 2, 5, 5, seed=7)
        blocks = dv.layer_jacobians(params, compiled, 1.0)
        names = dv.layer_names(params)
        self.assertEqual(names, [["W1", "b1"], ["W2", "b2"], ["Wout"]])
        step = 1e-6
        for block, (name, *_rest) in zip(blocks, names):
            shifted_up, shifted_down = copy.deepcopy(params), copy.deepcopy(params)
            shifted_up[name].view(-1)[1] += step
            shifted_down[name].view(-1)[1] -= step
            numeric = (
                dv.predictions(shifted_up, compiled, 1.0) - dv.predictions(shifted_down, compiled, 1.0)
            ) / (2.0 * step)
            self.assertLess(float(torch.max(torch.abs(block[:, 1] - numeric))), 1e-7, name)

    def test_layerwise_decomposition(self) -> None:
        problem = get_problem("poisson_1d")
        compiled = compile_constraints(problem)
        params = dv.initial_parameters(1, 1, 3, 16, 16, seed=11)
        blocks = dv.layer_jacobians(params, compiled, 1.0)
        self.assertEqual(len(blocks), 4)
        kernel = dv.gram(blocks)
        full = torch.cat(blocks, dim=1)
        self.assertLess(float(torch.max(torch.abs(kernel - full @ full.T))), 1e-12)
        best_layer = max(float(torch.linalg.eigvalsh(block @ block.T)[0]) for block in blocks)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(kernel)[0]), best_layer - 1e-14)

    def test_widths_are_nested_prefixes(self) -> None:
        small = dv.initial_parameters(2, 1, 3, 4, 16, seed=2)
        large = dv.initial_parameters(2, 1, 3, 16, 16, seed=2)
        self.assertTrue(torch.equal(small["W1"], large["W1"][:4]))
        self.assertTrue(torch.equal(small["W3"], large["W3"][:4, :4]))
        self.assertTrue(torch.equal(small["Wout"], large["Wout"][:, :4]))
        for value in large.values():
            self.assertEqual(value.dtype, torch.float64)
            self.assertLessEqual(float(value.abs().max()), 1.0)

    def test_one_step_agrees_with_frozen_recursion_to_second_order(self) -> None:
        problem = get_problem("poisson_1d")
        compiled = compile_constraints(problem)
        params = dv.initial_parameters(1, 1, 2, 64, 64, seed=1)
        kernel = dv.gram(dv.layer_jacobians(params, compiled, 1.0))
        residual = dv.predictions(params, compiled, 1.0) - compiled.targets
        errors = []
        for factor in (1e-2, 1e-3):
            eta = factor / float(torch.linalg.eigvalsh(kernel)[-1])
            update = torch.func.grad(lambda p: dv.loss(p, compiled, 1.0))(params)
            moved = {name: value - eta * update[name] for name, value in params.items()}
            after = dv.predictions(moved, compiled, 1.0) - compiled.targets
            frozen = residual - 2.0 * eta * kernel @ residual
            errors.append(float(torch.linalg.vector_norm(after - frozen)))
        self.assertLess(errors[1], errors[0] * 2e-2)  # quadratic in the step


class RunTests(unittest.TestCase):
    def test_run_record_is_deterministic_and_loss_decreases(self) -> None:
        task = dv.tasks_of(CONFIG)[-1]
        first, second = dv.run_task(task), dv.run_task(task)
        for record in (first, second):
            record.pop("seconds")
        self.assertEqual(first, second)
        self.assertEqual(first["depth"], 2)
        self.assertEqual(len(first["layer_lambda_min_K0"]), 3)
        self.assertLess(first["loss_final"], first["loss_initial"])
        self.assertGreaterEqual(first["lambda_min_K0"], max(first["layer_lambda_min_K0"]) - 1e-14)
        self.assertAlmostEqual(first["eta"] * first["lambda_max_K0"], 0.2, places=12)

    def test_aggregate_counts_and_fails_on_shortfall(self) -> None:
        runs = [dv.run_task(task) for task in dv.tasks_of(CONFIG)]
        summary = dv.aggregate(CONFIG, runs)
        self.assertEqual(summary["declared_runs"], 4)
        self.assertEqual(summary["found_runs"], 4)
        self.assertEqual([cell["runs"] for cell in summary["cells"]], [2, 2])
        self.assertIn(r"\bottomrule", dv.tex_table({"config": CONFIG, "config_sha256": "x", **summary}))
        with self.assertRaises(AuditError):
            dv.aggregate(CONFIG, runs[:-1])
        with self.assertRaises(AuditError):
            dv.aggregate(CONFIG, runs[:-1] + runs[:1])

    def test_config_validation(self) -> None:
        dv.validate_config(CONFIG)
        for key, value in (
            ("problems", ["no_such_problem"]),
            ("depths", [0]),
            ("widths", [8, 8]),
            ("steps", 25),
            ("learning_rate_factor", 0.6),
            ("gap_fraction", 1.0),
        ):
            with self.assertRaises(AuditError, msg=key):
                dv.validate_config({**CONFIG, key: value})


if __name__ == "__main__":
    unittest.main()
