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
    StokesTipCoalescence, capped_tip_scale, crest_interface_refinement_points, fit_circle_kasa, gap_half_height,
    point_distance_expression, shallow_interface_refinement_points, tip_fit_indices, validate_upper_interface,
)
from coalescence.fem.q2_geometry import signed_jacobian_range
from coalescence.fem.stokes_block_audit import _json_safe, _solve_serial_block


class TipMapTests(unittest.TestCase):
    def test_late_mapped_geometry_honours_physical_size_and_local_apex(self) -> None:
        import itertools
        import numpy as np

        pole = math.sqrt(0.2 + math.sqrt(2.04))
        z = np.linspace(0.0, pole, 65)
        coarse_concave = list(zip(0.2 * z**2 - 0.5 * z**4, z))
        fine_z = np.linspace(0.0, pole, 257)
        fine_concave = list(zip(0.2 * fine_z**2 - 0.5 * fine_z**4, fine_z))
        angle = np.linspace(0.0, math.pi / 2, 65)
        convex = list(zip(np.cos(angle) - 1.0, np.sin(angle)))
        overshooting = [(0.0, 0.0), (0.02, 0.005), (0.04, 0.01), (0.05, 0.505), (-1.0, 1.0)]
        tiny_radius = 5e-13
        tiny_neck = [(tiny_radius * (1 - math.cos(a)), tiny_radius * math.sin(a))
                     for a in (0.0, 0.02, 0.04)] + [(0.001, 0.5), (-1.0, 1.0)]
        fixtures = (("coarse concave", coarse_concave, False, 8.0),
                    ("fine concave", fine_concave, True, 8.0),
                    ("convex", convex, False, 8.0),
                    ("overshooting arc", overshooting, False, 8.0),
                    ("tiny neck", tiny_neck, True, tiny_radius))
        for name, poly, has_apex_arc, radius in fixtures:
            with self.subTest(geometry=name), tempfile.TemporaryDirectory() as out:
                poly = [*poly[:-1], (-1.0, poly[-1][1])]
                problem = StokesTipCoalescence(
                    R0=1e-6, Z0=5e-13, output_dir=out, h_max=0.02,
                    spatial_scale=1e-6, tip_map_alpha=0.5, tip_map_outer=0.4,
                    tip_map_linear_core=1.0, neck_frame=True, neck_frame_moving=True,
                )
                problem.restart_poly = poly
                problem.frame_shift_phys = 1.0
                problem.tip_radius_lagged = radius
                problem._apex_arc = {'stale': True}
                mesh = MappedTipMesh()
                callbacks, parameters, curves = [], {}, []

                def control_points(pieces, scale, size_at, **kwargs):
                    callbacks.append(size_at)
                    curves.append(pieces)
                    return [mesh._fwd(*(x / scale for x in pieces[0](0.0))),
                            mesh._fwd(*(x / scale for x in pieces[-1](1.0)))]

                with patch.multiple(
                    mesh, get_problem=Mock(return_value=problem), is_first_time=Mock(return_value=True),
                    _control_points=Mock(side_effect=control_points),
                    set_gmsh_parameter=Mock(side_effect=lambda k, v: parameters.__setitem__(k, v)),
                    point=Mock(side_effect=itertools.count()), spline=Mock(), line=Mock(),
                    plane_surface=Mock(), add_mesh_size_field=Mock(side_effect=itertools.count()),
                    set_mesh_size_background_field=Mock(),
                ):
                    mesh.define_geometry()
                if radius == 8.0:
                    self.assertEqual(problem.current_h_tip(), problem.h_max)
                for distance in (0.01, 0.1, 1.0):
                    mapped = mesh._fwd(mesh._T + distance / problem.S, 0.0)
                    dp = math.hypot(mapped[0] - mesh._T, mapped[1])
                    request = callbacks[0](mapped)
                    physical_chord = request * distance / dp
                    self.assertLessEqual(physical_chord, problem.tip_map_control_fraction * problem.h_max * (1 + 1e-12))
                    self.assertLessEqual(parameters['Mesh.MeshSizeMin'], request / problem.tip_map_control_fraction)
                if has_apex_arc:
                    receipt = problem._apex_arc
                    cap = min(problem.tip_apex_zone * radius, problem.h_max)
                    self.assertEqual(receipt['z_apex'], cap)
                    join = receipt['nodes_replaced']
                    self.assertGreater(join, 0)
                    self.assertEqual(join % 2, 0)
                    self.assertEqual(receipt['join_height'], poly[join][1])
                    self.assertLessEqual(receipt['join_height'], cap)
                    self.assertLessEqual(receipt['max_height'], cap)
                    apex = curves[0][0]
                    heights = [apex(t)[1] for t in np.linspace(0.0, 1.0, 257)]
                    self.assertLessEqual(max(heights), cap)
                    self.assertEqual(max(heights), receipt['max_height'])
                    for actual, expected in zip(apex(1.0), poly[join]):
                        self.assertAlmostEqual(actual / radius, expected / radius)
                    expected_tail = mesh._q2_pieces(poly[join:])
                    self.assertEqual(len(curves[0]), 1 + len(expected_tail))
                    for actual, expected in zip(curves[0][1:], expected_tail):
                        for t in np.linspace(0.0, 1.0, 17):
                            self.assertEqual(actual(t), expected(t))
                else:
                    self.assertEqual(problem._apex_arc, {})
                    expected_q2 = mesh._q2_pieces(poly)
                    self.assertEqual(len(curves[0]), len(expected_q2))
                    for actual, expected in zip(curves[0], expected_q2):
                        for t in np.linspace(0.0, 1.0, 17):
                            self.assertEqual(actual(t), expected(t))

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
    def test_tip_fit_does_not_reenter_the_distant_chord_angle_lobe(self) -> None:
        import numpy as np
        z = np.linspace(0.0, 2.0, 41)
        amplitude = math.tan(0.15) / (0.5 * math.exp(-0.5))
        r = amplitude * z**2 * np.exp(-2.0 * z**2)
        for perturbation in (-1e-14, 0.0, 1e-14):
            changed = r.copy()
            changed[10] += perturbation
            support = tip_fit_indices(changed, z, 0, 0.15, 0.05)
            self.assertGreaterEqual(len(support), 7)
            self.assertLessEqual(max(support), 10)

    def test_tip_fit_support_keeps_the_meniscus_behind_a_noisy_flat_face(self) -> None:
        import numpy as np
        rho, h = 1.0, 0.002                       # concave meniscus resolved by 500 nodes per radius
        z = np.arange(0.0, 0.6, h)
        r = 5.0 + rho - np.sqrt(rho**2 - z**2)
        r[2] += 1e-5                               # node noise turns the chord angle back
        lobe = tip_fit_indices(r, z, 0, 0.15, 0.0)
        self.assertEqual(lobe, tip_fit_indices(r, z, 0, 0.15, 0.0, 0.0))     # default is the plain lobe
        self.assertLess(max(z[lobe]), 0.02 * rho)  # collapsed onto the first few nodes
        kept = tip_fit_indices(r, z, 0, 0.15, 0.0, 0.25 * rho)
        self.assertGreater(max(z[kept]), 0.24 * rho)
        (_, _), radius = fit_circle_kasa([(float(r[i]), float(z[i])) for i in kept])
        self.assertAlmostEqual(radius, rho, delta=0.01 * rho)
        apex = tip_fit_indices(r, z, 0, 0.15, 0.05 * rho, 0.25 * rho)
        self.assertGreaterEqual(min(z[i] for i in apex if i != 0), 0.05 * rho)  # apex zone still excluded

    def test_gap_scale_caps_only_a_tip_radius_beyond_it(self) -> None:
        # Ordered Q2 interface (vertex, midside, vertex, ...): a bulb, then the gap surface beyond it.
        poly = [(0.0, 0.0), (0.01, 0.10), (0.03, 0.30), (0.06, 0.40), (0.04, 0.20), (0.06, 0.25), (0.08, 0.50)]
        self.assertAlmostEqual(gap_half_height(poly, 0.035), 0.19375)   # minimum inside the last edge (t = 1/8)
        self.assertEqual(gap_half_height(poly, 1.0), math.inf)
        self.assertEqual(capped_tip_scale(5.0, 0.2, 0.0), 5.0)      # off
        self.assertAlmostEqual(capped_tip_scale(5.0, 0.2, 6.0), 1.2)
        self.assertEqual(capped_tip_scale(1.0, 0.2, 6.0), 1.0)      # below the cap: unchanged
        self.assertEqual(capped_tip_scale(5.0, math.inf, 6.0), 5.0)

    def test_tip_scale_floor_bounds_a_capped_scale_from_below(self) -> None:
        self.assertEqual(capped_tip_scale(0.05, 0.2, 6.0), 0.05)                  # no floor: unchanged
        self.assertAlmostEqual(capped_tip_scale(0.05, 0.2, 6.0, 0.4), 0.48)       # floor 0.4 * 6 * 0.2
        self.assertAlmostEqual(capped_tip_scale(0.6, 0.2, 6.0, 0.4), 0.6)         # between floor and cap
        self.assertAlmostEqual(capped_tip_scale(5.0, 0.2, 6.0, 0.4), 1.2)         # cap still applies
        self.assertEqual(capped_tip_scale(0.05, 0.2, 0.0, 0.4), 0.05)             # cap off: floor off
        self.assertEqual(capped_tip_scale(0.05, math.inf, 6.0, 0.4), 0.05)        # no gap: unchanged

    def test_tip_scale_floor_breaks_the_flat_round_remesh_cycle(self) -> None:
        # A late flat apex read alternately as the flat face (fitted radius beyond 6 G, capped) and
        # as the meniscus turn (radius below one G), G = 1.25e-4.  Without the floor the scale
        # alternates by more than the grow trigger (3); with floor 0.4 it stays within 2.5 of the cap.
        gap, cap, grow, shrink = 1.25e-4, 6.0, 3.0, 0.7
        flat, turn = 1.48e-3, 1.06e-4
        hi, lo = capped_tip_scale(flat, gap, cap), capped_tip_scale(turn, gap, cap)
        self.assertGreater(hi / lo, grow)
        self.assertLess(lo / hi, shrink)
        hi_f, lo_f = capped_tip_scale(flat, gap, cap, 0.4), capped_tip_scale(turn, gap, cap, 0.4)
        self.assertLess(hi_f / lo_f, grow)          # the flat reading no longer triggers a coarser remesh

    def test_tip_scale_floor_engages_only_after_an_accepted_capped_state(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4)
            self.assertFalse(problem.cap_engaged)
            gap = 1e-3
            problem.note_cap_engaged({"tip_radius": 5e-3, "gap_half_height": gap})   # round tip below the cap
            self.assertFalse(problem.cap_engaged)
            problem.note_cap_engaged({"tip_radius": 7e-3, "tip_scale": 6e-3})        # no gap recorded
            self.assertFalse(problem.cap_engaged)
            problem.note_cap_engaged({"tip_radius": 7e-3, "gap_half_height": gap})   # flat: beyond 6 G
            self.assertTrue(problem.cap_engaged)
            off = self._moving_problem(out, tip_gap_cap=6.0)
            off.note_cap_engaged({"tip_radius": 7e-3, "gap_half_height": gap})
            self.assertFalse(off.cap_engaged)                                         # floor off: never engaged

    def test_tip_scale_floor_keeps_a_capped_state_seen_before_a_remesh_in_an_accepted_step(self) -> None:
        # pyoomph remeshes inside solve: the accepted step's post-remesh fit can be uncapped
        # although the pre-remesh state was capped.  The observation survives to the commit.
        gap = 1.25e-4
        capped = {"R_min": 5.2e-2, "tip_radius": 1.48e-3, "tip_scale": 7.5e-4, "gap_half_height": gap}
        uncapped = {"R_min": 5.2e-2, "tip_radius": 1.06e-4, "tip_scale": 3e-4, "gap_half_height": gap}
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4)
            with patch.object(problem, "neck_state", return_value=capped), \
                    patch.object(problem, "interface_polyline", return_value=[(0.0, 0.0), (1.0, 1.0)]), \
                    patch.object(problem, "write_restart_state"), \
                    patch.object(Problem, "actions_before_remeshing"):
                problem.actions_before_remeshing(set())
            self.assertFalse(problem.cap_engaged)                # pending until the step is accepted
            problem.note_cap_engaged(uncapped)
            self.assertTrue(problem.cap_engaged)
            self.assertAlmostEqual(capped_tip_scale(uncapped["tip_radius"], gap, 6.0, 0.4), 3e-4)
            self.assertFalse(problem._pending_cap_observation)  # the commit clears the observation
            off = self._moving_problem(out, tip_gap_cap=6.0)
            with patch.object(off, "neck_state", return_value=capped), \
                    patch.object(off, "interface_polyline", return_value=[(0.0, 0.0), (1.0, 1.0)]), \
                    patch.object(off, "write_restart_state"), \
                    patch.object(Problem, "actions_before_remeshing"):
                off.actions_before_remeshing(set())
            self.assertFalse(off._pending_cap_observation)        # floor off: nothing recorded

    def test_remesh_checkpoint_of_the_first_capped_state_restarts_engaged(self) -> None:
        import numpy
        from pathlib import Path
        gap = 1.25e-4
        capped = {"t": 3.5, "R_min": 5.2e-2, "u_neck": 6e-3, "tip_radius": 1.48e-3, "tip_scale": 7.5e-4,
                  "gap_half_height": gap}
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4)
            Path(out).mkdir(parents=True, exist_ok=True)
            with patch.object(problem, "neck_state", return_value=capped), \
                    patch.object(problem, "interface_polyline", return_value=[(0.0, 0.0), (1e-3, 1e-4), (-1e-2, 1.0)]), \
                    patch.object(Problem, "actions_before_remeshing"):
                problem.actions_before_remeshing(set())
            self.assertFalse(problem.cap_engaged)                  # not yet accepted
            files = sorted((Path(out) / "restart").glob("remesh_*.npz"))
            self.assertEqual(len(files), 1)
            saved = dict(numpy.load(files[0]))
            self.assertTrue(bool(saved["cap_engaged"]))            # the checkpointed state is capped
            restarted = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4, restart=saved)
            self.assertTrue(restarted.cap_engaged)

    def test_failed_capped_trial_does_not_engage_the_recovery_checkpoint(self) -> None:
        # A capped trial remeshes and then fails; the campaign restores the uncapped state and a
        # recovery remesh checkpoints it.  That checkpoint must not carry the trial's observation.
        import inspect
        import numpy
        from pathlib import Path
        gap = 1.25e-4
        capped = {"t": 3.5, "R_min": 5.2e-2, "u_neck": 6e-3, "tip_radius": 1.48e-3, "tip_scale": 7.5e-4,
                  "gap_half_height": gap}
        restored = {"t": 3.49, "R_min": 5.19e-2, "u_neck": 6e-3, "tip_radius": 1.06e-4, "tip_scale": 1.06e-4,
                    "gap_half_height": gap}
        poly = [(0.0, 0.0), (1e-3, 1e-4), (-1e-2, 1.0)]
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4)
            with patch.object(problem, "interface_polyline", return_value=poly), \
                    patch.object(Problem, "actions_before_remeshing"):
                with patch.object(problem, "neck_state", return_value=capped):
                    problem.actions_before_remeshing(set())          # the trial's remesh
                self.assertTrue(problem._pending_cap_observation)
                problem._discard_trial_observations()                # the failed attempt
                problem.n_remesh += 1
                with patch.object(problem, "neck_state", return_value=restored):
                    problem.actions_before_remeshing(set())          # the recovery remesh
            files = sorted((Path(out) / "restart").glob("remesh_*.npz"))
            self.assertEqual(len(files), 2)
            self.assertFalse(bool(dict(numpy.load(files[-1]))["cap_engaged"]))
            problem.note_cap_engaged(restored)
            self.assertFalse(problem.cap_engaged)
        # The campaign's failure handler discards the observations before any recovery remesh.
        src = inspect.getsource(StokesTipCoalescence.run_campaign)
        handler = src[src.index("except Exception as exc:  # Newton failure"):]
        self.assertLess(handler.index("_discard_trial_observations()"), handler.index("force_remesh"))

    def test_snapshot_restart_keeps_the_engaged_floor(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4)
            problem.cap_engaged = True
            state = {"t": 1.0, "R_min": 1e-2, "u_neck": 1e-2, "tip_radius": 1.06e-4, "tip_scale": 3e-4}
            with patch.object(problem, "interface_polyline", return_value=[(0.0, 0.0), (1e-3, 1e-4), (-1e-2, 1.0)]):
                problem.write_snapshot(state)
            import numpy
            from pathlib import Path
            saved = dict(numpy.load(Path(out) / "snapshots" / "snap_0000.npz"))
            self.assertTrue(bool(saved["cap_engaged"]))
            restarted = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4, restart=saved)
            self.assertTrue(restarted.cap_engaged)

    def test_gap_is_exact_on_q2_edges_and_unchanged_by_subdivision(self) -> None:
        def curve(t):                              # one quadratic edge whose minimum lies between its nodes
            return (2.0 + t, (t - 0.75) ** 2 + 1e-4)
        coarse = [curve(0.0), curve(0.5), curve(1.0)]
        fine = [curve(k / 4.0) for k in range(5)]  # exact subdivision of the same quadratic
        for r_out in (1.0, 2.6, 2.8):
            exact = 1e-4 if r_out <= 2.75 else (r_out - 2.75) ** 2 + 1e-4
            self.assertAlmostEqual(gap_half_height(coarse, r_out), exact, delta=1e-12)
            self.assertAlmostEqual(gap_half_height(fine, r_out), exact, delta=1e-12)
            self.assertAlmostEqual(capped_tip_scale(1.0, gap_half_height(coarse, r_out), 6.0),
                                   capped_tip_scale(1.0, gap_half_height(fine, r_out), 6.0), delta=1e-12)

    def test_gap_crossing_survives_a_nearly_linear_radial_edge(self) -> None:
        def curve(t):                              # straight edge rising outwards: the minimum is the r_out crossing
            return (1e-3 * t, 2e-4 + 1.8e-3 * t)
        coarse = [curve(0.0), curve(0.5), curve(1.0)]
        fine = [curve(k / 4.0) for k in range(5)]
        for bump in (0.0, 1.0, -1.0):              # one-ulp radial midpoint perturbations
            edge = list(coarse)
            edge[1] = (math.nextafter(edge[1][0], bump * math.inf) if bump else edge[1][0], edge[1][1])
            self.assertAlmostEqual(gap_half_height(edge, 4e-4), 9.2e-4, delta=1e-12)
        self.assertAlmostEqual(gap_half_height(fine, 4e-4), 9.2e-4, delta=1e-12)

    def test_flat_capped_apex_recovery_compares_curvature_on_the_gap_scale(self) -> None:
        # Recovery remesh of a flat capped apex: both fitted radii far beyond the cap, the sign of the
        # fitted curvature flipped between meshes, geometry unchanged.
        base = {"t": 3.2744812722860583, "R_min": 0.05006481021292418, "volume": 4.1888}
        before = dict(base, tip_radius=0.03022655696937398, kappa_m=33.08349015778462,
                      tip_scale=0.009020116180324654, gap_half_height=0.0015033526967207758)
        after = dict(base, tip_radius=0.01347727990839988, kappa_m=-74.19894866001394,
                     tip_scale=0.009020096777068418, gap_half_height=0.0015033494628447365)
        StokesTipCoalescence.verify_recovery_remesh(before, after)
        # A radius that drops far below the cap turns the apex over on the gap scale:
        # |dkappa| G = (1/1.2e-3 + 33) * 1.5e-3 = 1.3 > 0.5, refused on the curvature.
        with self.assertRaisesRegex(RuntimeError, "curvature"):
            StokesTipCoalescence.verify_recovery_remesh(before, dict(after, tip_radius=1.2e-3, kappa_m=-1 / 1.2e-3,
                                                                     tip_scale=1.2e-3))
        # A gap that changes by more than a factor of two is a changed geometry: refused on G.
        with self.assertRaisesRegex(RuntimeError, "gap half-height"):
            StokesTipCoalescence.verify_recovery_remesh(before, dict(after, gap_half_height=4e-3,
                                                                     tip_scale=after["tip_radius"]))

    def test_capped_recovery_accepts_a_radius_below_the_cap_within_the_gap_curvature_bound(self) -> None:
        # Recovery remesh of a capped flat apex whose fitted radius on the new mesh falls below the
        # cap: the scale changes by 2.5x, the gap is unchanged and |dkappa| G = 0.47 <= 0.5.
        base = {"t": 3.5468312722860524, "R_min": 0.051948024604720935, "volume": 4.1888}
        before = dict(base, tip_radius=0.029426388469859777, kappa_m=33.98310333000118,
                      tip_scale=0.010446792993391383, gap_half_height=0.0017411321655652304)
        after = dict(base, tip_radius=0.004221368044435637, kappa_m=-236.89002936338187,
                     tip_scale=0.004221368044435637, gap_half_height=0.0017415814368085558)
        StokesTipCoalescence.verify_recovery_remesh(before, after)
        StokesTipCoalescence.verify_recovery_remesh(after, before)
        # Without recorded gaps the capped scales must still agree within a factor of two.
        strip = lambda s: {k: v for k, v in s.items() if k != "gap_half_height"}   # noqa: E731
        with self.assertRaisesRegex(RuntimeError, "tip scale"):
            StokesTipCoalescence.verify_recovery_remesh(strip(before), strip(after))
        # Consistent cap = 1 states: radii 2G, scales G, opposite curvature signs: |dkappa| G = 1 > 0.5, refused.
        gap = 1e-3
        flat = dict(base, tip_radius=2 * gap, tip_scale=gap, gap_half_height=gap)
        with self.assertRaisesRegex(RuntimeError, "curvature"):
            StokesTipCoalescence.verify_recovery_remesh(dict(flat, kappa_m=1 / (2 * gap)), dict(flat, kappa_m=-1 / (2 * gap)))

    def test_size_field_distance_parses_negative_centres(self) -> None:
        # A refinement centre at negative mapped x must still give a valid Gmsh distance expression.
        expr = point_distance_expression(-0.15034367107065086, 122.04557883366792)
        self.assertNotIn("--", expr)
        value = eval(expr.replace("^", "**"), {"sqrt": math.sqrt, "x": 1.0, "y": 120.0})   # noqa: S307 (test literal)
        self.assertAlmostEqual(value, math.hypot(1.15034367107065086, 2.04557883366792), places=12)

    def test_engaged_floor_recovery_judges_continuity_on_the_gap(self) -> None:
        # Recovery remeshes of late flat apexes (geometry unchanged, fits on the new mesh disagree):
        # each is refused on its fitted radius or curvature without the floor, accepted with it.
        cases = [
            ((1.084e-4, -9224.0, 1.084e-4, 1.8423666e-4), (1.028e-3, 973.0, 1.028e-3, 1.8424831e-4), "tip radius"),
            ((1.059e-3, 944.0, 8.929e-4, 1.4881991e-4), (3.591e-4, -2785.0, 3.591e-4, 1.4880942e-4), "curvature"),
            ((2.315e-4, 4320.0, 2.315e-4, 1.4381122e-4), (1.817e-05, -55042.0, 1.817e-05, 1.4381127e-4), "tip radius"),
        ]
        base = {"t": 1.0, "R_min": 1.5e-2, "volume": 4.1888}
        for (rb, kb, sb, gb), (ra, ka, sa, ga), refused in cases:
            before = dict(base, tip_radius=rb, kappa_m=kb, tip_scale=sb, gap_half_height=gb)
            after = dict(base, tip_radius=ra, kappa_m=ka, tip_scale=sa, gap_half_height=ga)
            with self.assertRaisesRegex(RuntimeError, refused):
                StokesTipCoalescence.verify_recovery_remesh(before, after)
            floored = lambda st: dict(st, tip_scale=capped_tip_scale(st["tip_radius"], st["gap_half_height"], 6.0, 0.4),  # noqa: E731
                                      floor_engaged=1.0)
            StokesTipCoalescence.verify_recovery_remesh(floored(before), floored(after))
            with self.assertRaisesRegex(RuntimeError, "gap half-height"):
                StokesTipCoalescence.verify_recovery_remesh(floored(before), dict(floored(after), gap_half_height=3 * ga))
            with self.assertRaisesRegex(RuntimeError, "volume"):
                StokesTipCoalescence.verify_recovery_remesh(floored(before), dict(floored(after), volume=4.2))

    def test_engaged_floor_keeps_the_radius_check_between_floor_and_cap(self) -> None:
        # Latched floor, both scales strictly between floor (2.4 G) and cap (6 G): a 2.2x radius change
        # of an unclamped tip is still refused.
        base = {"t": 1.0, "R_min": 1.5e-2, "volume": 4.1888, "gap_half_height": 1e-3, "floor_engaged": 1.0}
        before = dict(base, tip_radius=2.5e-3, kappa_m=-400.0, tip_scale=2.5e-3)
        after = dict(base, tip_radius=5.5e-3, kappa_m=-1 / 5.5e-3, tip_scale=5.5e-3)
        with self.assertRaisesRegex(RuntimeError, "tip radius"):
            StokesTipCoalescence.verify_recovery_remesh(before, after)

    def test_clamped_flat_apex_curvature_does_not_steer_the_time_step(self) -> None:
        # A floored late apex (G = 4.95e-4, cap 6, floor 0.4: scale >= 1.19e-3) whose fit on an unrefined
        # mesh shrinks to 1.4e-4 while 2H grows twentyfold: the reading must not cut dt.
        gap = 4.95e-4
        floored = {"t": 1.3115, "R_min": 3.289e-2, "u_neck": 1.14e-2, "two_H": -7122.0, "tip_radius": 1.4e-4,
                   "tip_scale": capped_tip_scale(1.4e-4, gap, 6.0, 0.4), "gap_half_height": gap, "floor_engaged": 1.0}
        between = dict(floored, tip_radius=2e-3, tip_scale=2e-3)                    # unclamped: still trusted
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4)
            self.assertTrue(problem.curvature_reading_ignored(floored))
            self.assertFalse(problem.curvature_reading_ignored(between))
            self.assertFalse(problem.curvature_reading_ignored({k: v for k, v in floored.items() if k != "floor_engaged"}))
            problem._dt, problem._last_curvature_ratio, problem._regrow = 2.4e-5, 4.0, False
            self.assertAlmostEqual(problem.choose_dt(floored), 2.4e-5 * problem.dt_growth)   # grows, not cut
            self.assertLess(problem.choose_dt(between), 2.4e-5)                              # trusted reading cuts it
            off = self._moving_problem(out, tip_gap_cap=6.0)
            off._dt, off._last_curvature_ratio, off._regrow = 2.4e-5, 4.0, False
            self.assertLess(off.choose_dt({k: v for k, v in floored.items() if k != "floor_engaged"}), 2.4e-5)

    def test_capillary_ceiling_caps_the_step_at_n_steps_per_crest_period(self) -> None:
        # Oh = 3e-4, N = 100: at R_min = 0.0156 the ceiling 4*R_min**3/(Oh*N) = 2.03e-4 binds below the
        # bridge-growth ceiling and the grown previous step; the controller then stops regrowing.
        oh = 3e-4
        state = {"t": 1.0, "R_min": 0.0156, "u_neck": 1.6e-2, "two_H": -400.0, "tip_radius": 1e-3,
                 "tip_scale": 1e-3}
        with tempfile.TemporaryDirectory() as out:
            on = self._moving_problem(out, Oh=oh, capillary_steps_per_period=100.0)
            on._dt, on._last_curvature_ratio, on._regrow = 1e-2, 1e-12, True
            ceiling = 4.0 * 0.0156 ** 3 / (oh * 100.0)
            self.assertAlmostEqual(on.capillary_dt_ceiling(state), ceiling)
            self.assertAlmostEqual(on.choose_dt(state), ceiling)
            self.assertFalse(on._regrow)
            self.assertAlmostEqual(on.limit_dt_for_campaign(1.0, state), ceiling)   # caps any attempt (retries too)
            before = dict(state, R_min=5.0 * oh)                        # below the onset 10*Oh: no ceiling
            self.assertEqual(on.capillary_dt_ceiling(before), math.inf)
            at_onset = dict(state, R_min=10.0 * oh)
            self.assertAlmostEqual(on.capillary_dt_ceiling(at_onset), 4.0 * (10.0 * oh) ** 3 / (oh * 100.0))
            twice = self._moving_problem(out, Oh=oh, capillary_steps_per_period=200.0)
            self.assertAlmostEqual(twice.capillary_dt_ceiling(state), 0.5 * ceiling)

    def test_capillary_ceiling_off_leaves_the_controller_unchanged(self) -> None:
        oh = 3e-4
        states = [{"t": 0.1 * k, "R_min": r, "u_neck": u, "two_H": -1e3, "tip_radius": 1e-3, "tip_scale": 1e-3}
                  for k, (r, u) in enumerate([(3e-4, 0.4), (6e-3, 0.05), (0.0156, 1.6e-2), (0.05, 6e-3)])]
        with tempfile.TemporaryDirectory() as out:
            plain = self._moving_problem(out, Oh=oh)
            off = self._moving_problem(out, Oh=oh, capillary_steps_per_period=0.0, capillary_period_coefficient=9.0)
            for st in states:
                for problem in (plain, off):
                    problem._dt, problem._last_curvature_ratio, problem._regrow = 2e-3, 0.03, False
                self.assertEqual(off.capillary_dt_ceiling(st), math.inf)
                self.assertEqual(off.choose_dt(st), plain.choose_dt(st))
                self.assertEqual(off.limit_dt_for_campaign(1.0, st), plain.limit_dt_for_campaign(1.0, st))
            with self.assertRaisesRegex(ValueError, "capillary_steps_per_period"):
                self._moving_problem(out, Oh=oh, capillary_steps_per_period=-1.0)
            with self.assertRaisesRegex(ValueError, "finite Oh"):
                self._moving_problem(out, capillary_steps_per_period=100.0)

    def test_alternating_clamped_and_trusted_readings_do_not_shrink_the_step(self) -> None:
        # gap 1e-3, cap 6, floor 0.4: radius 7e-3 is capped (scale 6e-3), 5.5e-3 is trusted.  Every accepted
        # step has an ignored endpoint, so the controller never shortens dt (12 alternating steps).
        gap = 1e-3
        def state(rho):
            return {"R_min": 3e-2, "u_neck": 1e-2, "two_H": -1.0 / rho, "tip_radius": rho,
                    "tip_scale": capped_tip_scale(rho, gap, 6.0, 0.4), "gap_half_height": gap, "floor_engaged": 1.0}
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, tip_gap_cap=6.0, tip_scale_floor=0.4)
            problem._dt, problem._regrow, problem._trial_curvature_ratio = 1e-4, False, None
            seq = [state(7e-3 if k % 2 == 0 else 5.5e-3) for k in range(13)]
            for prev, cur in zip(seq[:-1], seq[1:]):
                problem._last_curvature_ratio = problem.step_curvature_ratio(prev, cur)
                self.assertEqual(problem._last_curvature_ratio, 1e-12)
                problem._dt = problem.choose_dt(cur)
            self.assertGreaterEqual(problem._dt, 1e-4)
            # Two trusted endpoints still report their change.
            self.assertGreater(problem.step_curvature_ratio(state(5.5e-3), state(4e-3)), 0.0)
            self.assertNotEqual(problem.step_curvature_ratio(state(5.5e-3), state(4e-3)), 1e-12)

    def test_step_curvature_limit_skips_a_clamped_floored_state(self) -> None:
        import inspect
        src = inspect.getsource(StokesTipCoalescence.actions_after_newton_solve)
        self.assertIn("self.curvature_reading_ignored(self._step_state_before_newton)", src)
        self.assertIn("self.curvature_reading_ignored(current_state)", src)

    def test_neck_state_flags_an_engaged_floor_only(self) -> None:
        import inspect
        src = inspect.getsource(StokesTipCoalescence.neck_state)
        self.assertIn('extra["floor_engaged"] = 1.0', src)
        self.assertLess(src.index("floor = self.tip_scale_floor if self.cap_engaged else 0.0"),
                        src.index('extra["floor_engaged"] = 1.0'))

    def test_capped_recovery_still_checks_the_fitted_curvature(self) -> None:
        base = {"t": 1.0, "R_min": 1e-2, "volume": 4.0}
        before = dict(base, tip_radius=1e-4, kappa_m=-1e4, tip_scale=1e-5)
        StokesTipCoalescence.verify_recovery_remesh(before, dict(base, tip_radius=1.2e-4, kappa_m=-1 / 1.2e-4,
                                                                 tip_scale=1e-5))
        with self.assertRaisesRegex(RuntimeError, "curvature"):     # capped, but turns over on the mesh scale
            StokesTipCoalescence.verify_recovery_remesh(before, dict(base, tip_radius=2e-5, kappa_m=5e4,
                                                                     tip_scale=1e-5))
        with self.assertRaisesRegex(RuntimeError, "tip scale"):
            StokesTipCoalescence.verify_recovery_remesh(before, dict(base, tip_radius=1e-4, kappa_m=-1e4,
                                                                     tip_scale=4e-5))
        uncapped = dict(base, tip_radius=1e-4, kappa_m=-1e4, tip_scale=1e-4)     # cap off: radius ratio as before
        with self.assertRaisesRegex(RuntimeError, "tip radius"):
            StokesTipCoalescence.verify_recovery_remesh(uncapped, dict(base, tip_radius=3e-4, kappa_m=-1 / 3e-4,
                                                                       tip_scale=3e-4))

    def test_step_curvature_uses_matched_support_at_an_angle_threshold(self) -> None:
        import numpy as np
        z = np.linspace(0.0, 2.0, 41)
        amplitude = math.tan(0.15) / (0.5 * math.exp(-0.5))
        r = amplitude * z**2 * np.exp(-2.0 * z**2)
        changed = r.copy()
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(R0=1e-6, Z0=5e-13, output_dir=out)
            problem.tip_radius_lagged = 1.0
            mesh = Mock()
            mesh.evaluate_observable.return_value = 4.18879
            with patch.object(problem, '_interface_nodes',
                              side_effect=lambda: (None, changed, z, np.zeros(len(z)), np.zeros(len(z)))), patch.object(
                problem, '_frame_shift_now', return_value=1.0
            ), patch.object(problem, 'get_mesh', return_value=mesh), patch.object(
                problem, 'get_current_time', return_value=0.0
            ):
                before = problem.neck_state()
                problem._step_tip_fit_support = problem._last_tip_fit_support
                frozen = problem._step_tip_fit_support
                for perturbation in (-1e-14, 1e-14):
                    changed[10] = r[10] + perturbation
                    after = problem.neck_state()
                    self.assertEqual(problem._step_tip_fit_support, frozen)
                    self.assertLess(after['kappa_m'], 0.0)
                    self.assertLess(problem.curvature_change_ratio(before['two_H'], after['two_H']), 1e-6)

    def test_curvature_control_is_finite_across_a_flat_neck(self) -> None:
        ratio = StokesTipCoalescence.curvature_change_ratio
        self.assertAlmostEqual(ratio(-1e16, -1.3e16), 0.3)
        self.assertAlmostEqual(ratio(-0.04, 0.04), 0.04)
        self.assertEqual(ratio(0.0, 2.0), 1.0)
        for value in (float('nan'), float('inf')):
            with self.assertRaisesRegex(RuntimeError, 'non-finite tip curvature'):
                ratio(value, 1.0)

    def test_curvature_sign_crossing_does_not_collapse_the_controller(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = StokesTipCoalescence(R0=1e-6, Z0=5e-13, output_dir=out)
            problem._dt = 1e-3
            problem._last_curvature_ratio = problem.curvature_change_ratio(-1e-9, 1e-9)
            with patch.object(problem, 'tip_unresolved', return_value=False):
                dt = problem.choose_dt({'R_min': 1.0, 'u_neck': 0.1, 'tip_radius': 1.0})
            self.assertGreater(dt, problem._dt)

    def test_campaign_caps_actual_restart_steps_and_saves_the_endpoint(self) -> None:
        for start in (9.99, 9.997):
            with self.subTest(start=start), tempfile.TemporaryDirectory() as out:
                problem = StokesTipCoalescence(
                    R0=1e-6, Z0=5e-13, output_dir=out, dt_initial=0.004,
                    dt_max=0.005, t_stop=10.0, R_stop=10.0, tip_map_alpha=0.5,
                    neck_frame=True, neck_frame_moving=True,
                    snapshot_dt=0.005, field_frames=True,
                )
                problem._restart = {'dt': 0.02}
                live = {'t': start, 'R_min': 1.0, 'u_neck': 0.1, 'two_H': 1.0,
                        'tip_radius': 1.0, 'volume': 4.18879, 'h_tip_now': 0.01}
                attempted, frames = [], []

                def solve(*, timestep, **kwargs):
                    attempted.append(timestep)
                    live['t'] += timestep

                with patch.multiple(
                    problem, is_initialised=Mock(return_value=True), refine_tip=Mock(),
                    neck_state=Mock(side_effect=lambda: live.copy()), write_row=Mock(),
                    write_profile=Mock(), interface_polyline=Mock(return_value=[]),
                    ndof=Mock(return_value=0), choose_dt=Mock(return_value=0.02),
                    _dof_snapshot=Mock(return_value={}), solve=Mock(side_effect=solve),
                    write_snapshot=Mock(),
                    write_field_frame=Mock(side_effect=lambda st: frames.append(st['t'])),
                ):
                    summary = problem.run_campaign(max_steps=3)
                self.assertEqual(summary['status'], 'reached_t_stop')
                self.assertEqual(summary['final']['t'], 10.0)
                self.assertTrue(attempted)
                self.assertLessEqual(max(attempted), 0.005)
                if start == 9.997:
                    self.assertLess(attempted[-1], problem.dt_initial)
                self.assertEqual(frames[-1], 10.0)

    def test_invalid_timestep_controls_do_not_create_output(self) -> None:
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'absent'
            for options in ({'dt_initial': 0.0}, {'dt_initial': float('nan')},
                            {'dt_max': float('inf')}, {'dt_initial': 0.01, 'dt_max': 0.005}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    StokesTipCoalescence(R0=1e-6, Z0=5e-13, output_dir=out, **options)
                self.assertFalse(out.exists())

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

    def _moving_problem(self, out, **kwargs):
        return StokesTipCoalescence(
            R0=1e-4, Z0=5e-9, output_dir=out, spatial_scale=1e-4,
            tip_map_alpha=0.5, h_tip_floor=1e-30, neck_frame=True, neck_frame_moving=True, **kwargs,
        )

    def test_quality_safeguards_are_off_by_default_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out)
            self.assertFalse(problem.recover_signed_jacobian)
            self.assertEqual(problem.quality_remesh_ratio, 0.0)
            folded = InvalidMovingFrameGeometry("signed_jacobian", "folded")
            self.assertFalse(problem._is_recoverable_geometry(folded))
            self.assertTrue(problem._is_recoverable_geometry(InvalidMovingFrameGeometry("bulk_quadrant", "out")))
            self.assertTrue(problem._is_recoverable_geometry(HiddenInterfacePlaneCrossing(0, -1e-20)))
            problem.min_jacobian_ratio, problem._jacobian_ratio_at_remesh = 0.01, 0.8
            problem._steps, problem._step_at_remesh = 10, 0
            self.assertIsNone(problem._quality_remesh_reason())
            for bad in (-0.1, 1.0, float("nan")):
                with self.assertRaises(ValueError):
                    self._moving_problem(out, quality_remesh_ratio=bad)

    def test_signed_jacobian_is_recoverable_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, recover_signed_jacobian=True)
            self.assertTrue(problem._is_recoverable_geometry(InvalidMovingFrameGeometry("signed_jacobian", "folded")))
            self.assertFalse(problem._is_recoverable_geometry(InvalidMovingFrameGeometry("axis", "off axis")))
            self.assertFalse(problem._is_recoverable_geometry(RuntimeError("OomphException")))

    def test_quality_remesh_triggers_only_after_degradation_from_a_good_mesh(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, quality_remesh_ratio=0.3)
            problem.min_jacobian_ratio = 0.2          # no reference yet: never triggers on its own
            self.assertIsNone(problem._quality_remesh_reason())
            problem._jacobian_ratio_at_remesh = 0.7
            problem.min_jacobian_ratio = 0.31
            self.assertIsNone(problem._quality_remesh_reason())
            problem.min_jacobian_ratio = 0.2
            self.assertIn("element quality", problem._quality_remesh_reason())
            problem._jacobian_ratio_at_remesh = 0.25  # fresh mesh already below: no remesh loop
            self.assertIsNone(problem._quality_remesh_reason())
            problem._jacobian_ratio_at_remesh = 0.7
            problem._steps = problem._step_at_remesh  # first accepted step after a remesh is eligible
            with patch.object(problem, "neck_state", return_value={"R_min": 1e-4, "tip_radius": 1e-9}):
                self.assertIn("element quality", problem._remesh_reason())

    def test_quality_baseline_is_the_pre_step_mesh_so_a_first_step_crossing_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out, quality_remesh_ratio=0.3)

            def gate(value):
                return lambda: setattr(problem, "min_jacobian_ratio", value)

            with patch.object(problem, "validate_domain_geometry", side_effect=gate(0.7)) as geometry:
                problem._record_quality_baseline()      # initial mesh, before the first solve
                problem._record_quality_baseline()      # already recorded: not re-measured
            geometry.assert_called_once()
            self.assertEqual(problem._jacobian_ratio_at_remesh, 0.7)
            problem.min_jacobian_ratio = 0.2           # the first step degrades it below the threshold
            self.assertIn("element quality", problem._quality_remesh_reason())
            off = self._moving_problem(out)
            with patch.object(off, "validate_domain_geometry") as geometry:
                off._record_quality_baseline()
            geometry.assert_not_called()               # default runs are untouched

    def test_recovery_reasons_name_the_rejected_geometry(self) -> None:
        reason = StokesTipCoalescence._recovery_reason
        self.assertEqual(reason(InvalidMovingFrameGeometry("bulk_quadrant", "out")),
                         "valid-state recovery after rejected bulk-quadrant trial")
        self.assertEqual(reason(InvalidMovingFrameGeometry("signed_jacobian", "folded")),
                         "valid-state recovery after rejected signed-jacobian trial")
        self.assertEqual(reason(HiddenInterfacePlaneCrossing(0, -1e-20)),
                         "valid-state recovery after rejected hidden-Q2 interface trial")

    def test_geometry_gate_records_worst_jacobian_ratio(self) -> None:
        import numpy as np

        local = [(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)]

        def element(coords):
            nodes = []
            for k, current in enumerate(coords):          # reference geometry: the unit triangle
                node = Mock()
                node.x.side_effect = lambda i, c=current: c[i]
                node.x_lagr.side_effect = lambda i, k=k: local[k][i]
                nodes.append(node)
            e = Mock()
            e.nnode.return_value = 6
            e.local_coordinate_of_node.side_effect = lambda i: local[i]
            e.node_pt.side_effect = lambda i: nodes[i]
            return e

        affine = element([(2 * x + 1.0, 2 * y + 1.0) for x, y in local])
        bent = [(x + 1.0, y + 1.0) for x, y in local]
        bent[3] = (1.5, 1.2)                          # mid-side node pushed inwards, still valid
        curved = element(bent)
        drop, axis = Mock(), Mock()
        drop.nodes.return_value, axis.nodes.return_value = [], []
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out)
            for elements, expected in (([affine], 1.0), ([affine, curved], None)):
                drop.elements.return_value = elements
                with patch.object(problem, "_frame_shift_now", return_value=1e-4), patch.object(
                    problem, "get_mesh", side_effect=lambda name: axis if name == "drop/axis" else drop
                ):
                    problem.validate_domain_geometry()
                if expected is not None:
                    self.assertAlmostEqual(problem.min_jacobian_ratio, expected)
                else:
                    lo, hi = signed_jacobian_range(np.array(local, float), np.array(bent, float))
                    self.assertGreater(lo, 0.0)
                    self.assertAlmostEqual(problem.min_jacobian_ratio, lo / hi)
                    self.assertLess(problem.min_jacobian_ratio, 1.0)

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

    @staticmethod
    def _parabola_crest(r_neck=1.0, radius=2e-3, x0=1.00605, height=1.2e-2, n=121, h=1e-4, scale=1.0, offset=0.0):
        # z = H - (x - x0)^2 / (2R) sampled at Q2 nodes: exactly quadratic per edge, curvature 1/R at
        # its apex (x0 inside an edge); z > 0 along the whole profile.
        xs = [r_neck + i * h for i in range(n)]
        return [(offset + scale * x, scale * (height - (x - x0) ** 2 / (2 * radius))) for x in xs]

    def test_crest_refinement_finds_a_sharp_convex_crest_near_the_neck(self) -> None:
        r_neck, radius, x0 = 1.0, 2e-3, 1.00605
        targets = crest_interface_refinement_points(self._parabola_crest(), r_neck, window=0.25)
        self.assertEqual(len(targets), 1)
        x, z, fitted = targets[0]
        self.assertAlmostEqual(x, x0, delta=1e-9)
        self.assertAlmostEqual(fitted / radius, 1.0, delta=1e-6)       # exact Q2 curvature at the apex
        self.assertEqual(crest_interface_refinement_points(self._parabola_crest(), r_neck, window=0.001), [])
        rising = [(r_neck + i * 1e-4, i * 1e-5) for i in range(41)]  # collinear: no crest
        self.assertEqual(crest_interface_refinement_points(rising, r_neck, window=0.25), [])
        self.assertEqual(crest_interface_refinement_points(self._parabola_crest()[:3], r_neck, window=0.25), [])

    def test_crest_refinement_is_scale_invariant(self) -> None:
        reference = crest_interface_refinement_points(self._parabola_crest(), 1.0, window=0.25)
        s, offset = 1e-6, 1e-4                       # microscopic crest on an offset frame
        scaled = crest_interface_refinement_points(self._parabola_crest(scale=s, offset=offset), s, window=0.25)
        self.assertEqual(len(scaled), 1)
        self.assertAlmostEqual(scaled[0][2] / (s * reference[0][2]), 1.0, delta=1e-6)
        self.assertAlmostEqual((scaled[0][0] - offset) / (s * reference[0][0]), 1.0, delta=1e-9)

    def test_crest_refinement_finds_a_crest_inside_a_q2_edge(self) -> None:
        # Vertex heights rise monotonically, but the edge over nodes 2-4 has an interior maximum.
        z = [0, 0.5, 1, 4, 2, 2.5, 3]
        poly = [(0.001 * i, 0.001 * zi) for i, zi in enumerate(z)]
        targets = crest_interface_refinement_points(poly, 1.0, window=0.25)
        self.assertEqual(len(targets), 1)
        x, height, radius = targets[0]
        self.assertAlmostEqual(x, 0.0031, delta=1e-12)              # xi = 0.1 on the edge 0.002-0.004
        self.assertAlmostEqual(height, 0.004025, delta=1e-12)
        self.assertTrue(math.isfinite(radius) and radius > 0.0)

    def test_crest_refinement_finds_a_smooth_maximum_at_a_shared_vertex(self) -> None:
        # Both one-sided slopes vanish at the vertex (index 2): a smooth vertex-centred maximum.
        poly = [(i / 1024, z / 1024) for i, z in enumerate([0, 3, 4, 3.5, 2, 3, 6])]
        targets = crest_interface_refinement_points(poly, 1.0, window=0.25)
        self.assertEqual(len(targets), 1)
        x, height, radius = targets[0]
        self.assertEqual((x, height), poly[2])
        self.assertAlmostEqual(radius, 0.00048828125, delta=1e-15)  # the sharper one-sided radius
        flat = [(i / 1024, 1.0) for i in range(7)]                  # flat: no maximum
        self.assertEqual(crest_interface_refinement_points(flat, 1.0, window=0.25), [])

    def test_stationary_monotonic_joins_are_not_crests_and_do_not_displace_one(self) -> None:
        # Staircase of stationary joins (concave in, convex out, still rising) before the real crest.
        z = [0, 3, 4, 4.5, 6, 7.5, 8, 8.5, 10, 11.5, 12, 12.5, 14, 15.5, 16, 16.125, 16.5, 16.75, 16.5, 16, 15]
        poly = [(i / 1024, zi / 1024) for i, zi in enumerate(z)]
        targets = crest_interface_refinement_points(poly, 1.0, window=0.25)
        self.assertEqual([(x, h) for x, h, _ in targets], [poly[17]])  # only the true crest (index 17)
        join = [(i / 1024, zi / 1024) for i, zi in enumerate([0, 3, 4, 4.5, 6, 7.5, 9])]
        self.assertEqual(crest_interface_refinement_points(join, 1.0, window=0.25), [])

    def test_crest_cap_holds_physically_in_every_direction(self) -> None:
        mesh = MappedTipMesh()
        mesh._T, mesh._L, mesh._alpha, mesh._beta = 0.0, 1.0, 0.5, 0.5
        mesh._linear_core_d, mesh._dc, mesh._dcp, mesh._A = 0.0, float("inf"), float("inf"), 1.0
        x, z, radius, fraction = 1e-3, 1e-3, 2e-5, 0.1             # oblique crest: anisotropic map
        centre, cap_p, reach = mesh._crest_cap(x, z, radius, 1.0, fraction, floor=0.0)
        target = fraction * radius
        lengths = []
        for k in range(32):
            theta = 2 * math.pi * k / 32
            back = mesh._inv(centre[0] + cap_p * math.cos(theta), centre[1] + cap_p * math.sin(theta))
            lengths.append(math.hypot(back[0] - x, back[1] - z))
        self.assertLessEqual(max(lengths), 1.01 * target)           # never coarser than requested
        self.assertGreaterEqual(max(lengths), 0.95 * target)        # and tight in the weakest direction
        self.assertGreaterEqual(reach, cap_p)

    def test_crest_refinement_options_are_off_by_default_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out)
            self.assertEqual(problem.crest_refine_fraction, 0.0)
            for bad in ({"crest_refine_fraction": -0.1}, {"crest_refine_fraction": 1.0},
                        {"crest_window": 0.0}, {"crest_window": float("nan")}):
                with self.assertRaises(ValueError):
                    self._moving_problem(out, **bad)

    def test_blunt_apex_options_are_off_by_default_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            problem = self._moving_problem(out)
            self.assertEqual((problem.tip_fit_support, problem.tip_gap_cap), (0.0, 0.0))
            for bad in ({"tip_fit_support": -0.1}, {"tip_fit_support": 1.0}, {"tip_fit_support": float("nan")},
                        {"tip_gap_cap": -1.0}, {"tip_gap_cap": float("inf")}):
                with self.assertRaises(ValueError):
                    self._moving_problem(out, **bad)
            self.assertEqual(problem.tip_scale_floor, 0.0)
            for bad in ({"tip_scale_floor": -0.1, "tip_gap_cap": 6.0}, {"tip_scale_floor": 1.0, "tip_gap_cap": 6.0},
                        {"tip_scale_floor": float("nan"), "tip_gap_cap": 6.0}, {"tip_scale_floor": 0.4}):
                with self.assertRaises(ValueError):
                    self._moving_problem(out, **bad)

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
