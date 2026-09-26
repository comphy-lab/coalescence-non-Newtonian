"""Net capillary force consistency: the drop approach velocity must not depend on h_max.

The approach velocity of the drops is resisted only by the neck, so it amplifies
an error in the net axial capillary force on the drop by ~1/R_n.  With the
traction taken from the spline curvature at the quadrature points the net force
telescopes exactly, and the approach velocity converges as the far drop is
refined; with interpolated nodal traction it is dominated by an O(h_max^4)
force error.  The global force balance (disc force = net free-surface force) is
imposed on the discrete solution, since the drop velocity amplifies any residual
force imbalance of the discretisation by the same factor.
"""

from __future__ import annotations

import unittest

from independent.bim_stokes.geometry import anthony_initial_meridian, disc_for
from independent.bim_stokes.operators import Assembler
from independent.bim_stokes.solver import solve_velocity


def _velocities(R0: float, h_max: float, k: float = 0.1):
    mer = anthony_initial_meridian(R0, 0.5 * R0 * R0, k=k, h_max=h_max)
    ops = Assembler(mer, disc_for(mer, k=k, h_max=h_max)).assemble()
    ur, uz = solve_velocity(mer, 0.0, ops=ops)
    return ur[0], uz[-1]


class NetForceTests(unittest.TestCase):
    def test_drop_approach_velocity_is_resolution_independent(self) -> None:
        U1, V1 = _velocities(1e-4, 0.02)
        U2, V2 = _velocities(1e-4, 0.01)
        self.assertLess(abs(U1 / U2 - 1.0), 1e-5)
        self.assertLess(abs(V1 / V2 - 1.0), 0.1)

    def test_drop_approach_velocity_converges_with_grading_at_small_neck(self) -> None:
        # Global force balance: without it a 1e-9 relative force error on the unit drop,
        # amplified ~1/R_n by the neck, made V grow ~3.5x per halving of k at R0 = 1e-6.
        _, V1 = _velocities(1e-6, 0.02, k=0.1)
        _, V2 = _velocities(1e-6, 0.02, k=0.05)
        self.assertLess(abs(V1 / V2 - 1.0), 0.01)


if __name__ == "__main__":
    unittest.main()
