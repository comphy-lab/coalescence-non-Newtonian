"""Linear modes of a viscous drop: the three limits that pin the dispersion relation."""

from __future__ import annotations

import math
import unittest

from coalescence.analysis.drop_modes import (
    amplitude_transform,
    mode_root,
    oscillatory_mode,
    stokes_relaxation_rate,
)


class DropModeTests(unittest.TestCase):
    def test_inviscid_limit_is_rayleigh(self) -> None:
        Oh = 1e-4
        for s in (0.3 + 0.2j, 0.05 - 0.4j):
            exact = s / (s * s + 8 * Oh ** 2)
            self.assertLess(abs(amplitude_transform(s, Oh) / exact - 1), 1e-12)

    def test_weak_viscosity_approaches_lamb(self) -> None:
        errors = []
        for Oh in (1e-2, 1e-3, 1e-4):
            m = oscillatory_mode(Oh)
            errors.append(abs(m["decay_rate"] / m["decay_rate_lamb"] - 1))
            self.assertLess(abs(m["omega"] / m["omega_rayleigh"] - 1), 2e-3)
        # The correction falls like Oh^(1/2): a factor of about sqrt(10) per decade.
        for big, small in zip(errors, errors[1:]):
            self.assertGreater(big / small, 2.5)
            self.assertLess(big / small, 4.0)
        self.assertLess(errors[-1], 1e-2)

    def test_stokes_limit_is_quasi_static(self) -> None:
        for l in (2, 3):
            rate = stokes_relaxation_rate(l)
            # The potential and viscous parts become degenerate as q -> 0; at Oh = 100 the
            # residual cancellation still leaves about ten digits.
            s = mode_root(100.0, -rate, l, tol=1e-10)
            self.assertLess(abs(s.imag), 1e-9)
            self.assertLess(abs(-s.real / rate - 1), 1e-4)
        self.assertAlmostEqual(stokes_relaxation_rate(2), 20.0 / 19.0, places=14)

    def test_rayleigh_frequency_constant(self) -> None:
        self.assertAlmostEqual(oscillatory_mode(0.01)["omega_rayleigh"], math.sqrt(8) * 0.01, places=15)


if __name__ == "__main__":
    unittest.main()
