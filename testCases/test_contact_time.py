"""The power-law contact-time fit recovers a known shift."""

from __future__ import annotations

import unittest

import numpy as np

from coalescence.analysis.contact_time import power_law_contact_time, stokes_law_contact_time
from coalescence.analysis.theory import r_eggers


class ContactTimeTests(unittest.TestCase):
    def test_power_law_shift_is_recovered(self) -> None:
        A, n, tc, R0 = 0.4, 0.9, 3e-7, 1e-6
        t = np.geomspace(1e-9, 1e-3, 400)
        R = A * (t + tc) ** n
        fit = power_law_contact_time(t, R, R0)
        self.assertAlmostEqual(fit.t_con / tc, 1.0, places=6)
        self.assertAlmostEqual(fit.exponent, n, places=6)

    def test_stokes_law_shift_puts_the_curve_on_the_law(self) -> None:
        tau = np.geomspace(1e-10, 1e-3, 5000)
        shift = 2e-8
        t, R = tau - shift, r_eggers(tau)
        keep = t > 0
        self.assertAlmostEqual(stokes_law_contact_time(t[keep], R[keep], 1e-5) / shift, 1.0, places=4)

    def test_stokes_law_shift_refuses_bad_input(self) -> None:
        t, R = np.array([1.0, 2.0, 3.0]), np.array([1e-6, 2e-6, 3e-6])
        with self.assertRaises(ValueError):
            stokes_law_contact_time(t, R, 1e-5)                         # outside the range
        with self.assertRaises(ValueError):
            stokes_law_contact_time(t, np.array([1e-6, 3e-6, 2e-6]), 2.5e-6)   # not increasing


if __name__ == "__main__":
    unittest.main()
