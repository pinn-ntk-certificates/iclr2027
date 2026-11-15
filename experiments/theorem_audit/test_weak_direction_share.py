"""Numerical-contract tests for :mod:`weak_direction_share`."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .engine import evaluate_finite, make_initialization_bank
from .features import compile_constraints
from .rank_controls import make_control
from .run import AuditError, PROJECT_ROOT
from .weak_direction_share import (
    SCHEMA_VERSION,
    SUMMARY_NAME,
    build_record,
    compare,
    evaluate_cell,
    frozen_prediction,
    load_config,
    main,
    observed_medians,
    residual_shares,
    summarize,
    validate_config,
)

PAPER_CONFIG = PROJECT_ROOT / "experiments/theorem_audit/configs/weak_direction_share.json"
SMOKE_CONFIG = PROJECT_ROOT / "experiments/theorem_audit/configs/weak_direction_share_smoke.json"


class ResidualShareTests(unittest.TestCase):
    def test_shares_are_squared_eigencoefficients_and_sum_to_one(self) -> None:
        kernel = np.array([[2.0, 0.5], [0.5, 1.0]])
        error = np.array([0.3, -0.7])
        eigenvalues, shares = residual_shares(kernel, error)
        self.assertTrue(np.all(np.diff(eigenvalues) >= 0))
        self.assertAlmostEqual(float(shares.sum()), 1.0, places=14)
        values, vectors = np.linalg.eigh(kernel)
        expected = (vectors.T @ error) ** 2 / (error @ error)
        np.testing.assert_allclose(shares, expected, rtol=0, atol=1e-15)

    def test_residual_along_one_eigenvector_has_unit_share(self) -> None:
        kernel = np.diag([1e-5, 1.0])
        _, shares = residual_shares(kernel, np.array([0.0, 3.0]))
        np.testing.assert_allclose(shares, [0.0, 1.0], atol=1e-15)
        _, shares = residual_shares(kernel, np.array([-2.0, 0.0]))
        np.testing.assert_allclose(shares, [1.0, 0.0], atol=1e-15)

    def test_asymmetric_input_is_symmetrized(self) -> None:
        kernel = np.array([[2.0, 0.6], [0.4, 1.0]])
        eigenvalues, _ = residual_shares(kernel, np.array([1.0, 1.0]))
        expected = np.linalg.eigvalsh(np.array([[2.0, 0.5], [0.5, 1.0]]))
        np.testing.assert_allclose(eigenvalues, expected, rtol=0, atol=1e-15)

    def test_zero_residual_is_rejected(self) -> None:
        with self.assertRaises(AuditError):
            residual_shares(np.eye(2), np.zeros(2))


class FrozenPredictionTests(unittest.TestCase):
    def test_prediction_matches_explicit_frozen_recursion(self) -> None:
        """``sum_j s_j (1 - 2 c lambda_j/lambda_max)^(2T)`` equals ``||(I - 2 eta K)^T e||^2 / ||e||^2``."""

        rng = np.random.default_rng(3)
        root = rng.standard_normal((2, 5))
        kernel = root @ root.T
        error = rng.standard_normal(2)
        eigenvalues, shares = residual_shares(kernel, error)
        steps, factor = 25, 0.2
        predicted, factors = frozen_prediction(eigenvalues, shares, factor, steps)
        eta = factor / eigenvalues.max()
        propagator = np.linalg.matrix_power(np.eye(2) - 2.0 * eta * kernel, steps)
        explicit = float((propagator @ error) @ (propagator @ error) / (error @ error))
        self.assertAlmostEqual(predicted, explicit, places=12)
        self.assertEqual(factors.shape, (2,))

    def test_strong_direction_factor_and_weak_limit(self) -> None:
        eigenvalues = np.array([1e-6, 1.0])
        predicted, factors = frozen_prediction(eigenvalues, np.array([0.0, 1.0]), 0.2, 3000)
        self.assertAlmostEqual(factors[1], (1.0 - 0.4) ** 6000, places=15)
        self.assertLess(predicted, 1e-300)
        predicted, factors = frozen_prediction(eigenvalues, np.array([1.0, 0.0]), 0.2, 3000)
        self.assertAlmostEqual(factors[0], (1.0 - 0.4e-6) ** 6000, places=15)
        self.assertAlmostEqual(predicted, factors[0], places=15)

    def test_prediction_is_linear_in_the_shares(self) -> None:
        eigenvalues = np.array([0.001, 1.0])
        ratio_weak, _ = frozen_prediction(eigenvalues, np.array([1.0, 0.0]), 0.2, 3000)
        ratio_strong, _ = frozen_prediction(eigenvalues, np.array([0.0, 1.0]), 0.2, 3000)
        mixed, _ = frozen_prediction(eigenvalues, np.array([0.3, 0.7]), 0.2, 3000)
        self.assertAlmostEqual(mixed, 0.3 * ratio_weak + 0.7 * ratio_strong, places=15)

    def test_non_positive_kernel_is_rejected(self) -> None:
        with self.assertRaises(AuditError):
            frozen_prediction(np.array([-1.0, 0.0]), np.array([0.5, 0.5]), 0.2, 10)


class ConfigTests(unittest.TestCase):
    def test_paper_and_smoke_configs_load(self) -> None:
        paper = load_config(PAPER_CONFIG)
        smoke = load_config(SMOKE_CONFIG)
        self.assertEqual(paper["width"], 16384)
        self.assertEqual(paper["maximum_width_initialization_bank"], smoke["maximum_width_initialization_bank"])
        self.assertEqual(paper["steps"], smoke["steps"])
        self.assertEqual(paper["learning_rate_factor"], smoke["learning_rate_factor"])
        self.assertEqual(paper["target_mode"], "contrast")

    def test_paper_config_matches_the_control_campaign_settings(self) -> None:
        controls = json.loads(
            (PROJECT_ROOT / "experiments/theorem_audit/configs/rank_controls_paper.json").read_text(encoding="utf-8")
        )
        paper = load_config(PAPER_CONFIG)
        self.assertEqual(paper["steps"], controls["steps"])
        self.assertEqual(paper["learning_rate_factor"], controls["learning_rate_factor"])
        self.assertEqual(paper["contrast_amplitude"], controls["contrast_amplitude"])
        self.assertEqual(paper["maximum_width_initialization_bank"], max(controls["widths"]))
        self.assertIn(paper["width"], controls["widths"])
        self.assertEqual(paper["families"], controls["families"])
        for epsilon in paper["epsilons"]:
            self.assertIn(epsilon, controls["epsilons"])

    def test_invalid_settings_are_rejected(self) -> None:
        base = load_config(SMOKE_CONFIG)
        for key, value in [
            ("families", ["no_such_family"]),
            ("epsilons", [-0.1]),
            ("epsilons", [0.0]),
            ("target_mode", "random"),
            ("width", 32768),
            ("seeds", [0, 0]),
            ("steps", 0),
            ("learning_rate_factor", 1.0),
            ("threads", 0),
            ("schema_version", 2),
        ]:
            config = copy.deepcopy(base)
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AuditError):
                validate_config(config)
        config = copy.deepcopy(base)
        del config["steps"]
        with self.assertRaises(AuditError):
            validate_config(config)


class CellTests(unittest.TestCase):
    def test_cell_reproduces_the_control_run_initial_state(self) -> None:
        """The kernel and residual are those of ``rank_controls.run_task`` at the same seed and width."""

        config = load_config(SMOKE_CONFIG)
        cell = evaluate_cell("pointwise_poisson", 0.1, 1, config)
        problem, _ = make_control("pointwise_poisson", 0.1, "contrast", 0.5)
        compiled = compile_constraints(problem)
        state = make_initialization_bank(16384, problem.input_dim, problem.output_dim, 1).prefix(256)
        with torch.no_grad():
            initial = evaluate_finite(state, compiled)
        np.testing.assert_allclose(np.array(cell["kernel_initial"]), initial.kernel.numpy(), rtol=0, atol=0)
        np.testing.assert_allclose(np.array(cell["initial_error"]), initial.error.numpy(), rtol=0, atol=0)
        self.assertAlmostEqual(cell["initial_loss"], float(initial.loss), places=14)
        self.assertAlmostEqual(sum(cell["shares_ascending"]), 1.0, places=13)
        self.assertEqual(cell["exact_structural_rank"], 2)
        self.assertGreater(cell["lambda_min"], 0.0)
        self.assertEqual(cell["weak_share"], cell["shares_ascending"][0])
        self.assertAlmostEqual(cell["weak_direction_decrease"], 1.0 - cell["direction_factors"][0], places=15)

    def test_smaller_epsilon_gives_smaller_relative_gap(self) -> None:
        config = load_config(SMOKE_CONFIG)
        wide = evaluate_cell("weak_boundary", 0.1, 0, config)
        narrow = evaluate_cell("weak_boundary", 0.01, 0, config)
        self.assertLess(narrow["relative_gap"], wide["relative_gap"])
        self.assertLess(narrow["weak_direction_decrease"], wide["weak_direction_decrease"])


class SummaryTests(unittest.TestCase):
    def test_summary_takes_medians_over_seeds(self) -> None:
        cells = [
            {"family": "f", "epsilon": 0.01, "seed": seed, "relative_gap": gap, "condition_number": 1 / gap,
             "weak_share": share, "weak_direction_decrease": 0.05, "predicted_final_to_initial_loss_ratio": share}
            for seed, gap, share in [(0, 1e-5, 0.1), (1, 3e-5, 0.4), (2, 2e-5, 0.2)]
        ]
        summary = summarize(cells)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["runs"], 3)
        self.assertEqual(summary[0]["seeds"], [0, 1, 2])
        self.assertAlmostEqual(summary[0]["median_relative_gap"], 2e-5)
        self.assertAlmostEqual(summary[0]["median_weak_share"], 0.2)
        self.assertAlmostEqual(summary[0]["median_predicted_final_to_initial_loss_ratio"], 0.2)

    def test_comparison_uses_the_tracked_aggregate_when_present(self) -> None:
        config = load_config(PAPER_CONFIG)
        observed = observed_medians(config)
        if not observed["available"]:
            self.skipTest(observed["reason"])
        summary = [{"family": "vector_stokes", "epsilon": 0.01, "median_predicted_final_to_initial_loss_ratio": 0.5}]
        rows = compare(summary, observed)
        self.assertEqual(len(rows), 1)
        self.assertIsNotNone(rows[0]["observed"])
        self.assertAlmostEqual(rows[0]["difference"], 0.5 - rows[0]["observed"], places=15)
        missing = compare([{"family": "vector_stokes", "epsilon": 7.0, "median_predicted_final_to_initial_loss_ratio": 0.5}], observed)
        self.assertIsNone(missing[0]["observed"])

    def test_missing_aggregate_is_reported_not_fatal(self) -> None:
        config = dict(load_config(SMOKE_CONFIG))
        config["observed_aggregate"] = "experiments/theorem_audit/data/does_not_exist.json"
        observed = observed_medians(config)
        self.assertFalse(observed["available"])
        self.assertEqual(compare([], observed), [])


class EndToEndTests(unittest.TestCase):
    def test_smoke_record_and_summary_file(self) -> None:
        config = load_config(SMOKE_CONFIG)
        record = build_record(config)
        self.assertEqual(record["declared_cells"], 2 * 2 * 2)
        self.assertEqual(record["found_cells"], record["declared_cells"])
        self.assertEqual(len(record["summary"]), 4)
        for cell in record["cells"]:
            self.assertAlmostEqual(sum(cell["shares_ascending"]), 1.0, places=13)
            self.assertTrue(0.0 <= cell["weak_share"] <= 1.0)
            self.assertTrue(0.0 < cell["predicted_final_to_initial_loss_ratio"] <= 1.0)
        with tempfile.TemporaryDirectory() as directory:
            override = dict(config)
            override["output_dir"] = str(Path(directory).resolve())
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(override), encoding="utf-8")
            self.assertEqual(main(["--config", str(path)]), 0)
            written = json.loads((Path(directory) / SUMMARY_NAME).read_text(encoding="utf-8"))
        self.assertEqual(written["schema_version"], SCHEMA_VERSION)
        self.assertEqual(written["found_cells"], 8)
        self.assertIn("config_sha256", written)
        self.assertIn("git", written["environment"])


if __name__ == "__main__":
    unittest.main()
