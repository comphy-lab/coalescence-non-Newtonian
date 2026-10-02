"""Gate K2: the half-body boundary-integral identity for exact interior Stokes flows."""

from __future__ import annotations

import math
import unittest

import numpy as np

from coalescence.bim.geometry import Meridian, anthony_initial_meridian, disc_for
from coalescence.bim.operators import EDGE_COEFFICIENT, Assembler, edge_coefficient_from_identities


def _flows(ops):
    ru, zu = ops.velocity_points()
    r, z, Nr, Nz = ops.traction_points()
    out = {"strain": (np.concatenate([-ru / 2, zu]), np.concatenate([-Nr, 2 * Nz]))}
    p = r * r - 2 * z * z                                    # u = (r z^2, -2 z^3/3)
    out["quadratic"] = (np.concatenate([ru * zu * zu, -2 * zu**3 / 3]),
                        np.concatenate([(-p + 2 * z * z) * Nr + 2 * r * z * Nz,
                                        2 * r * z * Nr + (-p - 4 * z * z) * Nz]))
    out["shear"] = (np.concatenate([ru * zu, -zu * zu]),     # u = (r z, -z^2), p = -2 z
                    np.concatenate([4 * z * Nr + r * Nz, r * Nr - 2 * z * Nz]))
    return out


def _residual(mer):
    ops = Assembler(mer, disc_for(mer)).assemble()
    out = {}
    for name, (U, F) in _flows(ops).items():
        a, b, c = ops.C @ U, ops.K @ U / (8 * math.pi), ops.S @ F / (8 * math.pi)
        res = a + b - c
        res[ops.N] = 0.0            # r-rows on the axis (pole and disc centre) are trivial
        res[ops.N + ops.M] = 0.0
        local = np.maximum.reduce([np.abs(a), np.abs(b), np.abs(c)])
        out[name] = (np.abs(res), max(np.abs(a).max(), np.abs(b).max(), np.abs(c).max()), local)
    return ops, out


class IdentityTests(unittest.TestCase):
    def test_edge_coefficient_is_that_of_a_right_angle(self) -> None:
        th = np.linspace(0.0, math.pi / 2, 80)
        ops, _ = _residual(Meridian(R_n=1.0, X=np.cos(th) - 1.0, z=np.sin(th)))
        c = edge_coefficient_from_identities(ops)
        self.assertLess(np.abs(c - EDGE_COEFFICIENT).max(), 1e-6)

    def test_hemisphere_identity_converges_at_fourth_order(self) -> None:
        errs = []
        for n in (80, 160):
            th = np.linspace(0.0, math.pi / 2, n)
            _, out = _residual(Meridian(R_n=1.0, X=np.cos(th) - 1.0, z=np.sin(th)))
            res, scale, _ = out["quadratic"]
            errs.append(res.max() / scale)
        self.assertLess(errs[1], 2e-8)
        self.assertGreater(errs[0] / errs[1], 10.0)

    def test_exact_bridge_identity_holds_to_the_double_precision_floor(self) -> None:
        mer = anthony_initial_meridian(1e-6, 5e-13)
        ops, out = _residual(mer)
        r, z = ops.velocity_points()
        for name, (res, scale, local) in out.items():
            self.assertLess(res.max() / scale, 1e-7, name)
            # At the 1e-13..1e-19 tip scales: below 1e-6 of the row's own terms, or at the
            # absolute double-precision floor where the exact field itself is ~z.
            tip = np.concatenate([np.hypot(r - mer.R_n, z) < 1e-12] * 2)
            ok = (res[tip] <= 1e-6 * local[tip]) | (res[tip] <= 1e-15)
            self.assertTrue(np.all(ok), name)


if __name__ == "__main__":
    unittest.main()
