"""Eq. 2 bridge/sphere junction and Eq. 6 dividing ellipse; no pyoomph required."""

from __future__ import annotations

import math
import unittest

from problems.newtonian.geometry import (
    EQ6_AXIS_SLOPE,
    EQ6_BETA,
    approximate_point_contact_z0,
    clip_ellipse_to_initial_interface,
    clip_ellipse_to_polyline,
    decimate_polyline,
    eggers_stokes_rmin,
    r_free_surface,
    eq6_axis_point,
    eq6_conic_coeffs,
    eq6_conic_slope,
    eq6_conic_value,
    eq6_ellipse_parameters,
    eq6_ellipse_point,
    eq6_ellipse_slope,
    eq6_interface_hit,
    eq6_ellipse_points,
    eq6_plane_point,
    eq2_bridge_value,
    point_in_initial_drop,
    sphere_bridge_junction,
    sphere_centre,
    sphere_centre_offset,
    sphere_value,
    viscocapillary_velocity_slope,
)


class BridgeGeometryTests(unittest.TestCase):
    def test_point_contact_z0(self) -> None:
        self.assertEqual(approximate_point_contact_z0(1e-3), 5e-7)

    def test_t0_junction_sits_on_both_curves(self) -> None:
        r0, z0 = 1e-3, 5e-7
        r, z = sphere_bridge_junction(r0, z0)
        _, z_c = sphere_centre(r0, z0)
        self.assertGreater(r, r0)
        self.assertGreater(z, 0.0)
        self.assertAlmostEqual(r * r + (z - z_c) ** 2, 1.0, places=12)
        c = r0 + z0
        self.assertAlmostEqual((r - c) ** 2 + z * z, z0 * z0, places=12)
        self.assertAlmostEqual(z_c, 1.0, places=8)
        self.assertLess(z, 10.0 * z0)
        self.assertLess(r - r0, 10.0 * z0)

    def test_fig3_depth_junction(self) -> None:
        r0, z0 = 1e-6, 5e-13
        r, z = sphere_bridge_junction(r0, z0)
        _, z_c = sphere_centre(r0, z0)
        self.assertGreater(r, r0)
        self.assertAlmostEqual(r * r + (z - z_c) ** 2, 1.0, places=12)
        self.assertAlmostEqual(z_c, 1.0, places=12)

    def test_centre_offset_is_retained_below_unit_ulp(self) -> None:
        r0, z0 = 1e-6, 5e-13
        offset = sphere_centre_offset(r0, z0)
        self.assertTrue(math.isfinite(offset))
        self.assertAlmostEqual(offset, 0.5 * r0**3, delta=2e-19)
        # The centre coordinate itself may round to one, but the explicit
        # offset used by the inverse sphere formulas must remain nonzero.
        self.assertEqual(sphere_centre(r0, z0)[1], 1.0)

    def test_free_surface_is_continuous_at_tangent_junction(self) -> None:
        for r0 in (1e-3, 1e-4, 1e-6):
            z0 = approximate_point_contact_z0(r0)
            rj, zj = sphere_bridge_junction(r0, z0)
            self.assertAlmostEqual(r_free_surface(zj, r0, z0), rj, delta=2e-14)
            self.assertAlmostEqual(eq2_bridge_value(rj, zj, r0, z0), 0.0, delta=2e-24)
            self.assertAlmostEqual(sphere_value(rj, zj, r0, z0), 0.0, delta=2e-15)

    def test_eggers_small_tau(self) -> None:
        tau = 1e-4
        rmin = eggers_stokes_rmin(tau)
        self.assertGreater(rmin, 0.0)
        self.assertAlmostEqual(rmin, -(tau / math.pi) * math.log(tau))
        self.assertAlmostEqual(viscocapillary_velocity_slope(), 1.0 / math.pi)

    def test_eggers_tau_inverts(self) -> None:
        from problems.newtonian.geometry import eggers_tau_for_rmin

        rmin = 1e-3
        tau = eggers_tau_for_rmin(rmin)
        self.assertAlmostEqual(eggers_stokes_rmin(tau) / rmin, 1.0, places=6)


class Eq6EllipseTests(unittest.TestCase):
    def test_eq6_endpoints_and_slopes_fig2(self) -> None:
        rmin = 0.18
        coeffs = eq6_conic_coeffs(rmin)
        r1, z1 = eq6_axis_point(rmin)
        r2, z2 = eq6_plane_point(rmin)
        self.assertEqual((r1, z1), (0.0, rmin))
        self.assertEqual((r2, z2), (2.0 * rmin, 0.0))
        self.assertAlmostEqual(eq6_conic_value(r1, z1, coeffs), 0.0, places=12)
        self.assertAlmostEqual(eq6_conic_value(r2, z2, coeffs), 0.0, places=12)
        self.assertAlmostEqual(eq6_conic_slope(r1, z1, coeffs), EQ6_AXIS_SLOPE, places=10)
        self.assertAlmostEqual(eq6_conic_slope(r2, z2, coeffs), 0.0, places=10)
        a, b, c, _, _ = coeffs
        disc = b * b - 4.0 * a * c
        self.assertLess(disc, 0.0)

    def test_declared_ellipse_parameters(self) -> None:
        for rmin in (1e-3, 1e-4, 1e-6, 0.18, 0.5):
            p = eq6_ellipse_parameters(rmin)
            self.assertAlmostEqual(p.c, 4.0 * rmin**2 / (4.0 * rmin + p.m * p.H))
            self.assertAlmostEqual(p.a, 2.0 * rmin - p.c)
            self.assertAlmostEqual(p.b2, p.H * p.a**2 / (p.m * p.c))
            self.assertAlmostEqual(eq6_ellipse_slope(rmin, 0.0), p.m, places=10)
            self.assertEqual(eq6_ellipse_point(rmin, 0.0), eq6_axis_point(rmin))
            self.assertEqual(eq6_ellipse_point(rmin, 1.0), eq6_plane_point(rmin))

    def test_beta_is_rejected(self) -> None:
        self.assertIsNone(EQ6_BETA)
        with self.assertRaises(ValueError):
            eq6_conic_coeffs(0.18, beta=-0.9)

    def test_interface_hit_is_resolved_analytically(self) -> None:
        for r0 in (1e-3, 1e-4, 1e-6):
            z0 = approximate_point_contact_z0(r0)
            t, (r, z) = eq6_interface_hit(r0, r0, z0)
            self.assertGreater(t, 0.0)
            self.assertLess(t, 1.0)
            self.assertAlmostEqual(eq6_conic_value(r, z, eq6_conic_coeffs(r0)), 0.0, delta=2e-12)
            self.assertAlmostEqual(r, r_free_surface(z, r0, z0), delta=2e-14)

    def test_eq6_t0_clip_hits_free_surface_inside_ellipse_scale(self) -> None:
        r0, z0 = 1e-3, 5e-7
        pts = clip_ellipse_to_initial_interface(r0, r0, z0)
        self.assertGreaterEqual(len(pts), 3)
        self.assertAlmostEqual(pts[0][0], 0.0, places=12)
        self.assertAlmostEqual(pts[0][1], r0, places=12)
        r_hit, z_hit = pts[-1]
        self.assertLessEqual(r_hit, 2.0 * r0 + 1e-9)
        self.assertGreater(r_hit, r0)
        self.assertGreaterEqual(z_hit, 0.0)
        self.assertAlmostEqual(r_hit, r_free_surface(z_hit, r0, z0), places=7)

    def test_eq6_samples_stay_in_first_quadrant(self) -> None:
        for rmin in (1e-3, 0.18, 0.4):
            for r, z in eq6_ellipse_points(rmin, n=24):
                self.assertGreaterEqual(r, -1e-12)
                self.assertGreaterEqual(z, -1e-12)

    def test_eq6_polyline_clip_stays_in_bridge_region(self) -> None:
        r0, z0 = 1e-3, 5e-7
        zs = [0.0] + [z0 * i / 4.0 for i in range(1, 5)] + [1e-3, 0.5, 1.0, 1.999]
        iface = [(r_free_surface(z, r0, z0), z) for z in zs]
        iface[0] = (r0, 0.0)
        pts = clip_ellipse_to_polyline(eq6_ellipse_points(r0), iface)
        r_hit, z_hit = pts[-1]
        self.assertLessEqual(r_hit, 3.0 * r0)
        self.assertLess(z_hit, 10.0 * r0)

    def test_resample_keeps_ends(self) -> None:
        from problems.newtonian.geometry import resample_polyline

        pts = [(float(i), float(i) ** 2) for i in range(50)]
        thin = resample_polyline(pts, 8)
        self.assertEqual(thin[0], pts[0])
        self.assertEqual(thin[-1], pts[-1])
        self.assertEqual(len(thin), 8)


if __name__ == "__main__":
    unittest.main()
