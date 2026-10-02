"""The Newton-Krylov backward-Euler step reproduces the backward-Euler decay rate of a P2 drop.

Backward Euler damps mode n by 1/(1 + lambda_n dt) per step, so the fitted rate is
ln(1 + lambda_2 dt)/dt with lambda_2 = 20/19; the exact solve of the implicit
equations must reproduce it (the linearly implicit step does so only because the
problem is nearly linear here).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from coalescence.bim.newton import stalled, step_backward_euler
from coalescence.bim.solver import resample
from testCases.test_bim_relaxation import _drop


class NewtonTests(unittest.TestCase):
    def test_backward_euler_rate_of_p2_mode(self) -> None:
        dt, mer, t = 0.1, _drop(1e-3), 0.0
        dk = dict(k=0.1, n_tip=16, h_max=0.02)
        ts, amps = [], []
        for _ in range(4):
            moved, _, rep = step_backward_euler(mer, dt, dk, tol=1e-8)
            self.assertTrue(rep.converged)
            mer = resample(moved, 0.1, 16, 0.02)
            t += dt
            ts.append(t)
            amps.append((mer.z[-1] - mer.R_n) / 1.5)
        rate = -np.polyfit(ts, np.log(amps), 1)[0]
        scheme = math.log(1.0 + 20.0 / 19.0 * dt) / dt
        self.assertAlmostEqual(rate / scheme, 1.0, delta=2e-3)

    def test_stall_acceptance_only_near_tolerance(self) -> None:
        tol = 1e-3
        history = [8.5e7, 8.3e4, 10.4, 1.8e-2, 3.0e-3]
        self.assertTrue(stalled(history + [2.1e-3], tol, 5.0))     # floor just above tol
        self.assertFalse(stalled(history + [1.4e-3], tol, 5.0))    # still contracting
        self.assertFalse(stalled([2.0e-2, 1.9e-2], tol, 5.0))      # stalled far above tol
        self.assertFalse(stalled(history + [2.1e-3], tol, 0.0))    # disabled


if __name__ == "__main__":
    unittest.main()
