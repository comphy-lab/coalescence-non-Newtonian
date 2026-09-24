"""Contracts for the mapped Stokes tip and in-step Newton control."""

from __future__ import annotations

import math
import tempfile
import unittest
from unittest.mock import patch

from pyoomph import Problem

from problems.newtonian.coalescence_stokes_tip import MappedTipMesh, StokesTipCoalescence, validate_upper_interface


class TipMapTests(unittest.TestCase):
    def test_linear_core_joins_power_map_with_value_and_slope_continuity(self) -> None:
        mesh = MappedTipMesh()
        mesh._T = 1.0
        mesh._L = 1.0
        mesh._alpha = 0.5
        mesh._beta = 0.4
        mesh._linear_core_d = 4e-11
        mesh._linear_core_g = mesh._linear_core_d**mesh._alpha
        mesh._dc = 1e3 * mesh._linear_core_d
        mesh._dcp = mesh._dc**mesh._alpha
        mesh._A = mesh._dc ** (mesh._alpha - mesh._beta)

        d0 = mesh._linear_core_d
        self.assertAlmostEqual(mesh._g(d0) / mesh._linear_core_g, 1.0)
        eps = 1e-5
        left = (mesh._g(d0) - mesh._g(d0 * (1 - eps))) / (eps * d0)
        right = (mesh._g(d0 * (1 + eps)) - mesh._g(d0)) / (eps * d0)
        self.assertAlmostEqual(left / right, 1.0, places=4)
        for d in [d0 / 16, d0 / 2, d0, 10 * d0, mesh._dc, 2 * mesh._dc]:
            x, y = mesh._fwd(mesh._T + d / math.sqrt(2), d / math.sqrt(2))
            back_x, back_y = mesh._inv(x, y)
            self.assertLess(math.hypot(back_x - mesh._T - d / math.sqrt(2), back_y - d / math.sqrt(2)), 8 * math.ulp(mesh._T))

    def test_linear_core_avoids_quadratic_tip_overclustering(self) -> None:
        mesh = MappedTipMesh()
        mesh._L = 1.0
        mesh._alpha = 0.5
        mesh._dc = float("inf")
        rho = 4e-11
        mesh._linear_core_d = rho
        mesh._linear_core_g = math.sqrt(rho)
        first = mesh._ginv(mesh._g(rho) / 16)
        self.assertGreater(first / rho, 0.04)
        self.assertLess(first / rho, 0.08)
        mesh._linear_core_d = 0.0
        self.assertLess(mesh._ginv(mesh._g(rho) / 16) / rho, 0.005)


class NewtonGateTests(unittest.TestCase):
    def test_moving_frame_equations_can_be_defined(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(
                R0=1e-6, Z0=5e-13, output_dir=out,
                tip_map_alpha=0.5, neck_frame=True, neck_frame_moving=True,
            )
            with patch.object(Problem, "add_equations", lambda self, equations: None):
                problem.define_problem()

    def test_extra_newton_is_rejected_and_gate_uses_original_solve_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError, "post-step Newton"):
                StokesTipCoalescence(R0=1e-3, Z0=5e-7, output_dir=out, extra_newton_iterations=2)
            problem = StokesTipCoalescence(
                R0=1e-3, Z0=5e-7, output_dir=out, min_newton_iterations=3
            )
            problem._newton_tolerance_before_gate = 0.01
            problem.newton_solver_tolerance = 0.01
            with patch.object(Problem, "actions_before_newton_step", lambda self: None), patch.object(
                Problem, "actions_before_newton_convergence_check", lambda self: None
            ):
                seen = []
                for _ in range(3):
                    problem.actions_before_newton_step()
                    problem.actions_before_newton_convergence_check()
                    seen.append(problem.newton_solver_tolerance)
            self.assertEqual(seen, [0.0, 0.0, 0.01])

    def test_curvature_jump_is_rejected_before_remeshing(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(
                R0=1e-6, Z0=5e-13, output_dir=out, curvature_step_limit=0.2
            )
            problem._template = object()
            problem._step_state_before_newton = {"two_H": -1.0e16, "tip_radius": 1e-16}
            with patch.object(problem, "last_newton_step_failed", return_value=False), patch.object(
                problem, "interface_polyline", return_value=[(1e-6, 0), (1.1e-6, 1e-10), (1.2e-6, 2e-10), (1.3e-6, 3e-10)]
            ), patch.object(
                problem, "neck_state", return_value={"two_H": -1.3e16, "tip_radius": 1e-16}
            ), patch.object(problem, "tip_unresolved", return_value=False), patch.object(
                problem, "_remesh_reason", side_effect=AssertionError("remesh ran")
            ):
                with self.assertRaisesRegex(RuntimeError, "tip curvature changed"):
                    problem.actions_after_newton_solve()

    def test_archived_folded_interface_is_rejected_before_remeshing(self) -> None:
        poly = [(1.0247792e-6, 0.0), (1.02500895e-6, -1.3285e-10),
                (1.0253e-6, 2e-10), (1.026e-6, 4e-10)]
        with self.assertRaisesRegex(RuntimeError, "crosses below"):
            validate_upper_interface(poly, 1e-6)

    def test_positive_q2_nodes_can_still_hide_a_plane_crossing(self) -> None:
        poly = [(1e-6, 0.0), (1e-6, 1e-21), (1e-6, 3e-20),
                (1e-6, 4e-20), (1e-6, 5e-20)]
        # The first edge rises at its nodes but dips below the plane in between.
        with self.assertRaisesRegex(RuntimeError, "quadratic edge crosses"):
            validate_upper_interface(poly, 1e-6)

    def test_invalid_tip_map_and_newton_controls_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError, "requires 0 < tip_map_alpha < 1"):
                StokesTipCoalescence(
                    R0=1e-6, Z0=5e-13, output_dir=out,
                    tip_map_alpha=3.1, tip_map_linear_core=1.0,
                )
            with self.assertRaisesRegex(ValueError, "between 0 and 12"):
                StokesTipCoalescence(
                    R0=1e-6, Z0=5e-13, output_dir=out, min_newton_iterations=13
                )

    def test_linear_core_precision_floor_keeps_a_resolvable_tip(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(
                R0=1e-6, Z0=5e-13, output_dir=out,
                tip_map_alpha=0.5, tip_map_linear_core=1.0,
            )
            problem._mesh_receipt = {"h_floor_phys": 1.25e-18}
            self.assertFalse(problem.tip_unresolved(4e-17))
            self.assertTrue(problem.tip_unresolved(2e-17))
            problem.tip_map_linear_core = 0.0
            self.assertTrue(problem.tip_unresolved(4e-17))


if __name__ == "__main__":
    unittest.main()
