"""Numerical-contract tests for :mod:`example_gd_certificate`."""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from .example_gd_certificate import (
    SCHEMA_VERSION,
    _bounds,
    _decimal,
    build_record,
    certified_width,
    exact_constants,
    feature_derivatives,
    interval_constants,
    load_config,
    reported_checks,
    sampled_checks,
    width_terms,
)
from .run import AuditError, PROJECT_ROOT

PAPER_CONFIG = PROJECT_ROOT / "experiments/theorem_audit/configs/example_gd_certificate.json"
SMOKE_CONFIG = PROJECT_ROOT / "experiments/theorem_audit/configs/example_gd_certificate_smoke.json"


def _paper() -> dict:
    config = load_config(PAPER_CONFIG)
    config["check_samples"] = 4000
    config["check_quadrature_nodes"] = 32
    return config


def _write(config: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(config, handle)
    handle.close()
    return Path(handle.name)


class ConfigValidation(unittest.TestCase):
    def test_rejects_wrong_schema_version(self) -> None:
        with self.assertRaises(AuditError):
            load_config(_write({**_paper(), "schema_version": SCHEMA_VERSION + 1}))

    def test_rejects_missing_field(self) -> None:
        broken = {key: value for key, value in _paper().items() if key != "delta"}
        with self.assertRaises(AuditError):
            load_config(_write(broken))

    def test_rejects_missing_reported_field(self) -> None:
        broken = copy.deepcopy(_paper())
        del broken["reported"]["width"]
        with self.assertRaises(AuditError):
            load_config(_write(broken))

    def test_rejects_float_settings(self) -> None:
        # Binary floats would make 0.1 inexact; settings must be strings.
        with self.assertRaises(AuditError):
            load_config(_write({**_paper(), "tube_radius": 0.1}))

    def test_rejects_delta_outside_unit_interval(self) -> None:
        for delta in ("0", "1", "1.5"):
            with self.assertRaises(AuditError):
                load_config(_write({**_paper(), "delta": delta}))

    def test_rejects_support_where_the_gap_bound_fails(self) -> None:
        # gamma * b_low <= W * X: tanh(w x + gamma b) can vanish on the support.
        with self.assertRaises(AuditError):
            load_config(_write({**_paper(), "hidden_weight_bound": "1"}))

    def test_smoke_and_paper_configs_share_the_setting(self) -> None:
        paper, smoke = load_config(PAPER_CONFIG), load_config(SMOKE_CONFIG)
        numerical = {"output_dir", "interval_decimal_digits", "report_decimal_digits", "check_samples", "check_quadrature_nodes"}
        for key in paper:
            if key not in numerical:
                self.assertEqual(paper[key], smoke[key], key)


class ExactConstants(unittest.TestCase):
    def test_rational_constants_of_the_example(self) -> None:
        exact = exact_constants(_paper())
        self.assertEqual(exact["R_K"], Fraction(81, 80))
        self.assertEqual(exact["B_star_squared"], Fraction(21, 20))
        self.assertEqual(exact["L_star_squared"], Fraction(30839, 12150))
        self.assertEqual(exact["eta_max"], Fraction(10, 21))
        self.assertEqual(exact["A_0_star"], Fraction(1, 10))

    def test_hessian_bound_matches_the_displayed_sum(self) -> None:
        # 0.2^2 c_2^2 (1/9 + 1/2 + 1) + 2 (1/4 + 1) with c_2^2 = 16/27.
        displayed = Fraction(1, 25) * Fraction(16, 27) * (Fraction(1, 9) + Fraction(1, 2) + 1) + 2 * Fraction(5, 4)
        self.assertEqual(exact_constants(_paper())["L_star_squared"], displayed)

    def test_second_derivative_supremum(self) -> None:
        # sup |tanh''| = 4 / (3 sqrt 3), attained where tanh^2 = 1/3.
        t = np.linspace(-6.0, 6.0, 2_000_001)
        second = np.abs(-2.0 * np.tanh(t) * (1.0 - np.tanh(t) ** 2))
        self.assertAlmostEqual(float(second.max()), 4.0 / (3.0 * math.sqrt(3.0)), places=10)
        self.assertLessEqual(float(second.max()) ** 2, 16.0 / 27.0 + 1e-15)


class IntervalConstants(unittest.TestCase):
    def test_enclosures_contain_the_float_values(self) -> None:
        enclosures = interval_constants(_paper())
        floats = {
            "lambda0": math.tanh(0.9) ** 2,
            "r_hat": 0.1 / math.sqrt(3.0) + 0.1 * math.sqrt(2.0 * math.log(40.0)),
            "c_Ch": 0.25 + 0.75 * math.log(0.75),
            "drift_prefactor": 320.0 + 128.0 * math.sqrt(6.0),
            "log_2N_over_delta": math.log(40.0),
        }
        for key, value in floats.items():
            lower, upper = _bounds(enclosures[key])
            self.assertLessEqual(lower, upper)
            self.assertLess(float(upper - lower), 1e-40, key)
            self.assertAlmostEqual(float(lower), value, delta=1e-13 * max(1.0, abs(value)), msg=key)

    def test_directed_decimal_rounding(self) -> None:
        third = Fraction(1, 3)
        self.assertEqual(_decimal(third, 4, upward=False), "0.3333")
        self.assertEqual(_decimal(third, 4, upward=True), "0.3334")
        self.assertEqual(_decimal(-third, 4, upward=False), "-0.3334")
        self.assertEqual(_decimal(Fraction(5, 4), 2, upward=True), "1.25")


class WidthTerms(unittest.TestCase):
    def _terms(self, config: dict, **override):
        reported = config["reported"]
        arguments = {
            "r_k": Fraction(reported["R_K"]),
            "b_star_squared": Fraction(reported["B_star_squared"]),
            "l_star_squared": Fraction(reported["L_star_squared"]),
        }
        arguments.update(override)
        return width_terms(config, **arguments)

    def test_terms_match_a_float_evaluation(self) -> None:
        config = _paper()
        lam = math.tanh(0.9) ** 2
        r_hat = 0.1 / math.sqrt(3.0) + 0.1 * math.sqrt(2.0 * math.log(40.0))
        expected = [
            1.0125 * math.log(40.0) / ((0.25 + 0.75 * math.log(0.75)) * lam),
            (320.0 + 128.0 * math.sqrt(6.0)) * 2.5382 * 1.05 * r_hat**2 / lam**3,
            16.0 * 1.05 * r_hat**2 / (0.1**2 * lam**2),
        ]
        for term, value in zip(self._terms(config), expected):
            lower, upper = _bounds(term)
            self.assertAlmostEqual(float(lower), value, delta=1e-9 * value)
            self.assertLess(float(upper - lower), 1e-30)

    def test_reported_width_and_displayed_bounds(self) -> None:
        config = _paper()
        terms = self._terms(config)
        self.assertEqual(certified_width(terms), 1356)
        for term, bound in zip(terms, ("212.613", "1355.980", "692.252")):
            self.assertLess(_bounds(term)[1], Fraction(bound))
        # 1356 is not improvable with these constants: the drift term exceeds 1355.
        self.assertGreater(_bounds(terms[1])[0], 1355)

    def test_exact_hessian_constant_gives_the_same_width(self) -> None:
        config = _paper()
        exact = exact_constants(config)
        terms = self._terms(config, l_star_squared=exact["L_star_squared"])
        self.assertEqual(certified_width(terms), 1356)

    def test_width_grows_as_confidence_and_gap_tighten(self) -> None:
        base = certified_width(self._terms(_paper()))
        smaller_delta = certified_width(self._terms({**_paper(), "delta": "0.01"}))
        smaller_gap = certified_width(self._terms({**_paper(), "hidden_bias_low": "0.5"}))
        self.assertGreater(smaller_delta, base)
        self.assertGreater(smaller_gap, base)

    def test_all_reported_checks_hold(self) -> None:
        checks = reported_checks(_paper())
        self.assertEqual(len(checks), 13)
        self.assertTrue(all(item["holds"] for item in checks), [c for c in checks if not c["holds"]])

    def test_wrong_reported_values_are_detected(self) -> None:
        for key, wrong in (("width", 1355), ("lambda0_lower", "0.5131"), ("r_hat_upper", "0.32935"), ("eta_max", "1/2")):
            config = copy.deepcopy(_paper())
            config["reported"][key] = wrong
            failed = [item for item in reported_checks(config) if not item["holds"]]
            self.assertTrue(failed, key)


class SampledEnvelopes(unittest.TestCase):
    def test_quadrature_matches_the_closed_form(self) -> None:
        # F(w, b) = (log cosh(w + b) - log cosh(b - w)) / (2 w) for X = gamma = 1.
        config = _paper()
        w = np.array([-0.1, 0.03, 0.1, 0.2])
        b = np.array([1.0, 1.4, 2.0, 2.1])
        closed = (np.log(np.cosh(w + b)) - np.log(np.cosh(b - w))) / (2.0 * w)
        derived = feature_derivatives(w, b, config)
        np.testing.assert_allclose(derived["F"], closed, rtol=0, atol=1e-14)
        step = 1e-5
        for name, dw, db in (("F_w", step, 0.0), ("F_b", 0.0, step)):
            plus = feature_derivatives(w + dw, b + db, config)["F"]
            minus = feature_derivatives(w - dw, b - db, config)["F"]
            np.testing.assert_allclose(derived[name], (plus - minus) / (2 * step), rtol=0, atol=1e-9)
        for name, first, dw, db in (("F_ww", "F_w", step, 0.0), ("F_wb", "F_w", 0.0, step), ("F_bb", "F_b", 0.0, step)):
            plus = feature_derivatives(w + dw, b + db, config)[first]
            minus = feature_derivatives(w - dw, b - db, config)[first]
            np.testing.assert_allclose(derived[name], (plus - minus) / (2 * step), rtol=0, atol=1e-9)

    def test_envelopes_hold_on_samples_for_the_example(self) -> None:
        result = sampled_checks(_paper())
        for name, item in result.items():
            if isinstance(item, dict):
                self.assertTrue(item["holds"], name)
        # The certified gap is a lower bound of the limiting kernel, not its value.
        self.assertGreater(result["monte_carlo_limiting_kernel"], float(Fraction("0.51308")))

    def test_envelopes_hold_on_samples_for_a_different_setting(self) -> None:
        config = {
            **_paper(),
            "gamma": "0.7",
            "domain_half_width": "1.5",
            "hidden_weight_bound": "0.3",
            "hidden_bias_low": "0.8",
            "hidden_bias_high": "1.5",
            "output_weight_bound": "0.25",
            "tube_radius": "0.05",
        }
        load_config(_write(config))
        for name, item in sampled_checks(config).items():
            if isinstance(item, dict):
                self.assertTrue(item["holds"], name)

    def test_sampling_is_deterministic(self) -> None:
        self.assertEqual(sampled_checks(_paper()), sampled_checks(_paper()))


class Record(unittest.TestCase):
    def test_record_is_complete_and_serializable(self) -> None:
        record = build_record(_paper())
        json.dumps(record, allow_nan=False)
        self.assertTrue(record["all_reported_checks_hold"])
        self.assertTrue(record["all_sampled_checks_hold"])
        self.assertEqual(record["width_terms_with_exact_constants"]["certified_integer_width"], 1356)
        self.assertEqual(record["exact_constants"]["L_star_squared"], "30839/12150")
        lower = record["interval_constants"]["lambda0"]["lower"]
        upper = record["interval_constants"]["lambda0"]["upper"]
        self.assertTrue(lower.startswith("0.51308263885165839"))
        self.assertLessEqual(Fraction(lower), Fraction(upper))


if __name__ == "__main__":
    unittest.main()
