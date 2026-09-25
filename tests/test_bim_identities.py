"""Gate K2: the boundary-integral identity for exact interior Stokes flows."""

from __future__ import annotations

import math
import unittest

import numpy as np

from independent.bim_stokes.geometry import Meridian, anthony_initial_meridian
from independent.bim_stokes.operators import Assembler


def _flows(mer):
    (_, _), (Nr, Nz), _, _, _ = mer.node_frame()
    r, z = mer.r, mer.z
    strain = (np.concatenate([-r / 2, z]), np.concatenate([-Nr, 2 * Nz]))
    p = r * r - 2 * z * z                                    # u = (r z^2, -2 z^3/3)
    fr = (-p + 2 * z * z) * Nr + 2 * r * z * Nz
    fz = 2 * r * z * Nr + (-p - 4 * z * z) * Nz
    quadratic = (np.concatenate([r * z * z, -2 * z**3 / 3]), np.concatenate([fr, fz]))
    return {"strain": strain, "quadratic": quadratic}


def _residual(mer):
    S, K = Assembler(mer).assemble()
    n = mer.n + 1
    out = {}
    for name, (U, F) in _flows(mer).items():
        a, b, c = 0.5 * U, K @ U / (8 * math.pi), S @ F / (8 * math.pi)
        res = a + b - c
        res[n] = 0.0          # u_z at the neck and u_r at the pole are fixed by symmetry
        res[mer.n] = 0.0
        local = np.maximum.reduce([np.abs(a), np.abs(b), np.abs(c)])
        out[name] = (np.abs(res), max(np.abs(a).max(), np.abs(b).max(), np.abs(c).max()), local)
    return out


class IdentityTests(unittest.TestCase):
    def test_sphere_identity_converges_at_fourth_order(self) -> None:
        errs = []
        for n in (60, 120):
            th = np.linspace(0.0, math.pi / 2, n)
            res, scale, _ = _residual(Meridian(R_n=1.0, X=np.cos(th) - 1.0, z=np.sin(th)))["strain"]
            errs.append(res.max() / scale)
        self.assertLess(errs[1], 1e-8)
        self.assertGreater(errs[0] / errs[1], 10.0)

    def test_exact_bridge_identity_holds_to_the_double_precision_floor(self) -> None:
        mer = anthony_initial_meridian(1e-6, 5e-13)
        for name, (res, scale, local) in _residual(mer).items():
            self.assertLess(res.max() / scale, 1e-7, name)
            # At the 1e-13..1e-19 tip scales: below 1e-6 of the row's own terms, or at the
            # absolute double-precision floor where the exact field itself is ~z.
            tip = np.concatenate([np.hypot(mer.X, mer.z) < 1e-12] * 2)
            ok = (res[tip] <= 1e-6 * local[tip]) | (res[tip] <= 1e-15)
            self.assertTrue(np.all(ok), name)


if __name__ == "__main__":
    unittest.main()
