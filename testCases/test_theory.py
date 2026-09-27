"""The Eggers, Lister & Stone (1999) law and its two leading-order velocity forms."""

from __future__ import annotations

import math
import unittest

import numpy as np

from coalescence.analysis.theory import r_eggers, tau_eggers, u_eggers, u_leading_log


class TheoryTests(unittest.TestCase):
    def test_inversion_round_trip(self) -> None:
        tau = np.exp(np.linspace(-40.0, -1.05, 50))
        self.assertLess(np.max(np.abs(tau_eggers(r_eggers(tau)) / tau - 1.0)), 1e-12)

    def test_velocity_is_the_time_derivative(self) -> None:
        tau, h = 1e-7, 1e-12
        numerical = (r_eggers(tau + h) - r_eggers(tau - h)) / (2 * h)
        self.assertAlmostEqual(float(u_eggers(r_eggers(tau))), float(numerical), places=5)

    def test_forms_differ_by_the_iterated_logarithm(self) -> None:
        tau = np.exp(np.linspace(-35.0, -3.0, 20))
        R = r_eggers(tau)
        expected = (np.log(np.abs(np.log(tau))) - math.log(math.pi) - 1.0) / math.pi
        self.assertLess(np.max(np.abs(u_eggers(R) - u_leading_log(R) - expected)), 1e-12)

    def test_outside_the_invertible_range_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            tau_eggers(np.array([0.2]))


if __name__ == "__main__":
    unittest.main()
