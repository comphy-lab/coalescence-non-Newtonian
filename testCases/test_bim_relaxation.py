"""Gate D1: small P2 deformation of a free viscous drop relaxes at the Lamb rate.

For a drop of radius a, viscosity mu and surface tension gamma in a passive
exterior, Lamb's interior solution with zero tangential stress and normal stress
-gamma (n-1)(n+2) zeta / a^2 gives the decay rate of mode n
lambda_n = (gamma / mu a) n (n+2)(2n+1) / (2 (2n^2 + 4n + 3)); lambda_2 = 20/19.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from coalescence.bim.geometry import Meridian
from coalescence.bim.solver import resample, step_extrapolated


def _drop(eps: float, nodes: int = 80) -> Meridian:
    th = np.linspace(math.pi / 2, 0.0, nodes)
    rho = 1.0 + eps * 0.5 * (3.0 * np.cos(th) ** 2 - 1.0)
    r, z = rho * np.sin(th), rho * np.cos(th)
    z[0] = 0.0
    return Meridian(R_n=r[0], X=r - r[0], z=z)


class RelaxationTests(unittest.TestCase):
    def test_p2_mode_decays_at_twenty_nineteenths(self) -> None:
        dt, mer, t = 0.1, _drop(1e-3), 0.0
        ts, amps = [], []
        for _ in range(10):
            moved, _ = step_extrapolated(mer, dt)
            mer = resample(moved, 0.1, 16, 0.02)
            t += dt
            ts.append(t)
            amps.append((mer.z[-1] - mer.R_n) / 1.5)
        rate = -np.polyfit(ts, np.log(amps), 1)[0]
        z = -20.0 / 19.0 * dt
        scheme = -math.log(2.0 / (1.0 - z / 2.0) ** 2 - 1.0 / (1.0 - z)) / dt
        self.assertAlmostEqual(rate / scheme, 1.0, delta=1e-3)
        self.assertAlmostEqual(rate / (20.0 / 19.0), 1.0, delta=5e-3)


if __name__ == "__main__":
    unittest.main()
