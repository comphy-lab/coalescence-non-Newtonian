"""Contracts for the mapped Stokes tip and in-step Newton control."""

from __future__ import annotations

import math
import tempfile
import unittest
from unittest.mock import Mock, patch

from pyoomph import Problem
from pyoomph.solvers.scipy import SuperLUSerial
from scipy.sparse import csr_matrix

from coalescence.fem.stokes_tip import (
    HiddenInterfacePlaneCrossing, InvalidMovingFrameGeometry, MappedTipMesh,
    StokesTipCoalescence, shallow_interface_refinement_points, validate_upper_interface,
)
from coalescence.fem.q2_geometry import signed_jacobian_range
from stokes_block_audit import _json_safe, _solve_serial_block


class TipMapTests(unittest.TestCase):
    def test_signed_q2_jacobian_catches_midside_fold_with_unchanged_vertices(self) -> None:
        import numpy as np

        local = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]])
        folded = local.copy()
        folded[3, 1] = 0.5
        self.assertEqual(signed_jacobian_range(local, local), (1.0, 1.0))
        minimum, maximum = signed_jacobian_range(local, folded)
        self.assertLess(minimum, 0)
        self.assertGreater(maximum, 0)

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
    def test_fresh_history_translation_cap_is_laboratory_frame_only(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            common = dict(R0=1e-6, Z0=5e-13, output_dir=out,
                          tip_map_alpha=0.5, neck_frame=True, dt_initial=1e-15)
            fixed = StokesTipCoalescence(**common)
            moving = StokesTipCoalescence(**common, neck_frame_moving=True)
            state = {"h_tip_now": 1e-20, "u_neck": 5.0}
            self.assertEqual(fixed.limit_dt_after_history_reset(1e-11, state), 1e-15)
            self.assertEqual(moving.limit_dt_after_history_reset(1e-11, state), 1e-11)

    def test_audit_receipt_serialises_nonfinite_diagnostics(self) -> None:
        import json
        import numpy as np

        receipt = _json_safe({"residual": [np.float64(float("nan")), float("inf")]})
        encoded = json.dumps(receipt, allow_nan=False)
        self.assertIn("nonfinite", encoded)

    def test_moving_frame_rejects_bulk_node_below_plane(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(
                R0=1e-6, Z0=5e-13, output_dir=out, spatial_scale=1e-6,
                tip_map_alpha=0.5, h_tip_floor=1e-30,
                neck_frame=True, neck_frame_moving=True,
            )
            interior = Mock()
            interior.x.side_effect = [5e-5, -7e-5]
            bulk = Mock()
            bulk.nodes.return_value = [interior]
            with patch.object(problem, "_frame_shift_now", return_value=1e-6), patch.object(
                problem, "get_mesh", return_value=bulk
            ):
                with self.assertRaisesRegex(InvalidMovingFrameGeometry, "left the physical quadrant") as caught:
                    problem.validate_domain_geometry()
                self.assertEqual(caught.exception.kind, "bulk_quadrant")

    def test_valid_state_recovery_remesh_checks_cap_continuity(self) -> None:
        before = {"t": 1e-8, "R_min": 1e-6, "volume": 4.18879, "tip_radius": 1e-18}
        after = {**before, "volume": before["volume"] * (1 + 2e-7)}
        StokesTipCoalescence.verify_recovery_remesh(before, after)
        after["tip_radius"] = 3e-18
        with self.assertRaisesRegex(RuntimeError, "tip radius excessively"):
            StokesTipCoalescence.verify_recovery_remesh(before, after)

    def test_block_audit_uses_serial_superlu_without_mpi(self) -> None:
        import numpy as np

        problem = Problem()
        matrix = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
        rhs = np.array([1.0, 2.0])
        correction = _solve_serial_block(SuperLUSerial(problem), matrix, rhs)
        np.testing.assert_allclose(matrix @ correction, rhs, atol=1e-14)

    def test_moving_frame_equations_can_be_defined(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(
                R0=1e-6, Z0=5e-13, output_dir=out,
                tip_map_alpha=0.5, neck_frame=True, neck_frame_moving=True,
            )
            with patch.object(Problem, "add_equations", lambda self, equations: None):
                problem.define_problem()

    def test_moving_axis_is_snapped_before_history_reset(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(
                R0=1e-6, Z0=5e-13, output_dir=out, spatial_scale=1e-6,
                tip_map_alpha=0.5, neck_frame=True, neck_frame_moving=True,
            )
            nodes = [Mock(), Mock()]
            axis, bulk = Mock(), Mock()
            axis.nodes.return_value = nodes
            with patch.object(
                problem, "_lookup_mesh", side_effect=lambda name: axis if name == "drop/axis" else bulk
            ), patch.object(problem, "invalidate_cached_mesh_data"):
                problem._snap_moving_axis(radius=1e-6)
            for node in nodes:
                node.set_x.assert_called_once_with(0, -1.0)
            bulk.set_lagrangian_nodal_coordinates.assert_called_once_with()

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

    def test_first_step_on_fresh_mesh_skips_only_the_curvature_gate(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(
                R0=1e-6, Z0=5e-13, output_dir=out, curvature_step_limit=0.2
            )
            problem._template = object()
            problem._fresh_mesh = True
            problem._step_state_before_newton = {"two_H": -1.0e16, "tip_radius": 1e-16}
            with patch.object(problem, "last_newton_step_failed", return_value=False), patch.object(
                problem, "interface_polyline", return_value=[(1e-6, 0), (1.1e-6, 1e-10), (1.2e-6, 2e-10), (1.3e-6, 3e-10)]
            ), patch.object(
                problem, "neck_state", return_value={"two_H": -1.3e16, "tip_radius": 1e-16}
            ), patch.object(problem, "tip_unresolved", return_value=False), patch.object(
                problem, "validate_domain_geometry"
            ) as geometry, patch.object(problem, "_remesh_reason", return_value=None), patch.object(
                Problem, "actions_after_newton_solve"
            ):
                problem.actions_after_newton_solve()
                geometry.assert_called_once()

    def test_archived_folded_interface_is_rejected_before_remeshing(self) -> None:
        poly = [(1.0247792e-6, 0.0), (1.02500895e-6, -1.3285e-10),
                (1.0253e-6, 2e-10), (1.026e-6, 4e-10)]
        with self.assertRaisesRegex(RuntimeError, "crosses below"):
            validate_upper_interface(poly, 1e-6)

    def test_positive_q2_nodes_can_still_hide_a_plane_crossing(self) -> None:
        poly = [(1e-6, 0.0), (1e-6, 1e-21), (1e-6, 3e-20),
                (1e-6, 4e-20), (1e-6, 5e-20)]
        # The first edge rises at its nodes but dips below the plane in between.
        with self.assertRaisesRegex(HiddenInterfacePlaneCrossing, "quadratic edge crosses") as caught:
            validate_upper_interface(poly, 1e-6)
        self.assertEqual(caught.exception.segment, 0)
        self.assertLess(caught.exception.minimum_z, 0.0)

    def test_shallow_off_neck_trough_requests_refinement_without_moving_interface(self) -> None:
        poly = [(0.0, 1e-3), (0.5, 1.2e-3), (1.0, 1e-2)]
        targets = shallow_interface_refinement_points(poly, tip_radius=1e-7)
        self.assertEqual(len(targets), 1)
        self.assertGreater(targets[0][1], 0.0)
        self.assertLess(targets[0][2], 0.05)
        self.assertEqual(shallow_interface_refinement_points(poly, tip_radius=1e-4), [])

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

    def test_interface_grading_controls_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError, "interface_grading"):
                StokesTipCoalescence(R0=1e-6, Z0=5e-13, output_dir=out, interface_grading=-0.1)
            with self.assertRaisesRegex(ValueError, "interface_size_growth"):
                StokesTipCoalescence(R0=1e-6, Z0=5e-13, output_dir=out, interface_size_growth=0.0)

    def test_restart_state_sets_the_remeshed_start(self) -> None:
        state = {
            "poly": [(0.0, 0.0), (1e-20, 1e-20), (-1.1e-6, 1.0)], "t": 2.8e-12,
            "R_min": 1.000008e-6, "u_neck": 3.12, "tip_radius_lagged": 2.4e-14,
            "rho_at_remesh": 2.4e-14, "n_remesh": 7, "steps": 69, "dt": 7.8e-14,
            "frame_shift_phys": 1.000008e-6,
        }
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(ValueError, "moving neck frame"):
                StokesTipCoalescence(R0=1e-6, Z0=5e-13, output_dir=out, restart=state)
            problem = StokesTipCoalescence(
                R0=1e-6, Z0=5e-13, output_dir=out, tip_map_alpha=0.5,
                neck_frame=True, neck_frame_moving=True, restart=state,
            )
            self.assertEqual(problem.R_start, state["R_min"])
            self.assertEqual(problem.frame_shift_phys, state["R_min"])
            self.assertEqual(problem.rmin_at_remesh, state["R_min"])
            self.assertEqual(problem.tip_radius_lagged, state["tip_radius_lagged"])
            self.assertEqual(problem.n_remesh, state["n_remesh"] + 1)
            self.assertEqual(problem._steps, state["steps"])
            self.assertEqual(problem.restart_poly[1], (1e-20, 1e-20))
            self.assertTrue(problem._fresh_mesh)


if __name__ == "__main__":
    unittest.main()
