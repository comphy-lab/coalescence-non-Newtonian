"""Pure-Stokes (La = 0) coalescence on a tip-graded unstructured ALE mesh.

Fresh solver route for the Anthony, Harris & Basaran (2020) T0 gate, written
against the pinned public pyoomph release only.  It deliberately avoids the
structured four-block chart, the algebraic Eq. 6 divider constraint and the
private backend used by the earlier campaign.

Why this can be simple
----------------------
In the Stokes limit the velocity field has no history: the only evolving
state is the interface geometry.  Remeshing therefore costs a geometric
interpolation of the interface and nothing else, so the mesh may be rebuilt
as often as the tip demands.  The meniscus tip is the single small feature of
the problem; its radius of curvature is ``Z0`` initially and tends to
``delta ~ R_min**3`` (Eggers, Lister & Stone 1999; Anthony et al. 2020,
Fig. 3 inset).  A size field that grows geometrically with distance from the
tip resolves a 1e-9 tip inside a unit drop with a few thousand triangles.

Nondimensionalisation: drop radius, viscocapillary time ``mu R / gamma`` and
capillary pressure ``gamma / R`` (Anthony et al. 2020, Sec. II.A with
``1/Oh = 0``).  The exterior is passive with zero density and viscosity.
"""

from __future__ import annotations

import csv
import json
import math
import time

import numpy
from pathlib import Path
from typing import Any

from pyoomph import *
from pyoomph import _pyoomph_core as _pyoomph
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import (
    AxisymmetryBC,
    EnforcedBC,
    GlobalLagrangeMultiplier,
    InitialCondition,
    IntegralObservables,
    RemeshWhen,
    RemeshingOptions,
    ScalarField,
    WeakContribution,
)
from pyoomph.expressions import cartesian, dot, exp, vector, scale_factor, nondim, pi, matrix, diff
from pyoomph.expressions.coordsys import AxisymmetricCoordinateSystem
from pyoomph.equations.navier_stokes import NavierStokesFreeSurface, StokesEquations

from .geometry import sphere_bridge_junction, sphere_centre


def smooth_polyline_arclength(pts: list[tuple[float, float]], half_window: int = 3) -> list[tuple[float, float]]:
    """Local least-squares quadratic smoothing of (r,z)(s) in chord-length parameter.

    Removes element-scale oscillation of the reconstructed interface (which would
    otherwise be read as curvature by a finer mesh) while keeping the systematic
    error at O(h^2 kappa).  End points are kept exactly.
    """
    n = len(pts)
    if n < 2 * half_window + 1:
        return list(pts)
    import numpy as np

    arr = np.asarray(pts, dtype=float)
    seg = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    s_par = np.concatenate([[0.0], np.cumsum(seg)])
    out = arr.copy()
    for i in range(1, n - 1):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        if hi - lo < 4:
            continue
        ss = s_par[lo:hi] - s_par[i]
        scale = max(abs(ss).max(), 1e-300)
        ss = ss / scale
        A = np.vstack([np.ones_like(ss), ss, ss * ss]).T
        for c in (0, 1):
            coef, *_ = np.linalg.lstsq(A, arr[lo:hi, c], rcond=None)
            out[i, c] = coef[0]
    return [(float(x), float(y)) for x, y in out]


def fit_circle_kasa(points) -> tuple[tuple[float, float], float] | None:
    """Algebraic (Kasa) least-squares circle through >=3 points; None if degenerate."""
    import numpy as np

    P = np.asarray(points, dtype=float)
    if P.shape[0] < 3:
        return None
    x0 = P.mean(axis=0)
    Q = P - x0  # centre the data for conditioning
    # ... and scale it to O(1): with a 5e-13 tip on a 1e-6 neck the centred coordinates are
    # ~1e-14 and the columns of A differ by 1e14, which lstsq's rank cut-off treats as
    # rank deficient (R0=1e-6 run 41 read 4.3e-14 for an exactly circular 5e-13 tip).
    scale = float(np.abs(Q).max())
    if not (scale > 0.0) or not math.isfinite(scale):
        return None
    Q = Q / scale
    A = np.column_stack([2.0 * Q[:, 0], 2.0 * Q[:, 1], np.ones(len(Q))])
    b = (Q ** 2).sum(axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if not (r2 > 0.0) or not math.isfinite(r2):
        return None
    return (float(cx * scale + x0[0]), float(cy * scale + x0[1])), float(math.sqrt(r2) * scale)


def circle_through(p0, p1, p2) -> tuple[tuple[float, float], float] | None:
    """Centre and radius of the circle through three points, or None if collinear."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if d == 0.0:
        return None
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy), math.hypot(ax - ux, ay - uy)


class TipGradedQuadrantMesh(GmshTemplate):
    """One quadrant of one drop; triangles graded geometrically from the neck tip."""

    def _piecewise_arcs(self, pts, p_first, p_last):
        """Interface as circle arcs through consecutive node triples (p_i, p_i+1, p_i+2).

        pyoomph inverts a circle arc analytically (atan2 about the centre), whereas its
        Catmull-Rom spline inversion is a Gauss-Newton iteration that fails once the spline
        spans many decades of segment length.  Adjacent arcs are C0 with tangent kinks of
        order h^2 dkappa/ds, the same continuity class as the Q2 boundary itself.
        """
        curves = []
        n = len(pts)
        i = 0
        start_pt = p_first
        while i < n - 1:
            if i + 2 <= n - 1:
                p0, p1, p2 = pts[i], pts[i + 1], pts[i + 2]
                end_pt = p_last if i + 2 == n - 1 else self.point(*p2)
                circ = circle_through(p0, p1, p2)
                if circ is not None and circ[1] < 1e6 * math.hypot(p2[0] - p0[0], p2[1] - p0[1]):
                    curves.append(self.circle_arc(start_pt, end_pt, center=circ[0], name="interface"))
                else:
                    mid_pt = self.point(*p1)
                    curves.append(self.line(start_pt, mid_pt, name="interface"))
                    curves.append(self.line(mid_pt, end_pt, name="interface"))
                start_pt = end_pt
                i += 2
            else:
                curves.append(self.line(start_pt, p_last, name="interface"))
                i += 1
        return curves

    def define_geometry(self):
        pb = self.get_problem()
        assert isinstance(pb, StokesTipCoalescence)
        self.mesh_mode = "quads" if pb.tip_refine else "tris"
        self.order = 2
        S = pb.S
        h_tip = pb.current_h_tip() / S   # Gmsh works in solver (scaled) units
        h_max = pb.h_max / S
        k = pb.grading
        self.default_resolution = h_max
        self.set_gmsh_parameter("General.NumThreads", 1)
        self.set_gmsh_parameter("Geometry.Tolerance", 1e-15)
        self.set_gmsh_parameter("Mesh.MeshSizeMin", h_tip)
        self.set_gmsh_parameter("Mesh.MeshSizeMax", h_max)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeFromPoints", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeExtendFromBoundary", 0)
        self.set_gmsh_parameter("Mesh.Algorithm", 6)

        if self.is_first_time():
            r0, z0 = pb.R0, pb.Z0
            _, zc = sphere_centre(r0, z0)
            jr, jz = sphere_bridge_junction(r0, z0)
            p_o = self.point(0.0, 0.0)
            p_n = self.point(r0, 0.0)
            p_j = self.point(jr, jz)
            p_t = self.point(0.0, zc + 1.0)
            meniscus = self.circle_arc(p_n, p_j, center=(r0 + z0, 0.0), name="interface")
            sphere = self.circle_arc(p_j, p_t, center=(0.0, zc), name="interface")
            interface = [meniscus, sphere]
            pb._mesh_receipt = {"kind": "initial", "h_tip": h_tip * S, "r_neck": r0}
        else:
            pts, arc = pb.interface_polyline_for_remesh()
            r_neck = pts[0][0]
            z_pole = pts[-1][1]
            p_o = self.point(0.0, 0.0)
            p_n = self.point(r_neck, 0.0)
            p_t = self.point(0.0, z_pole)
            if arc is not None:
                # Tip as a circle arc through three actual interface nodes (neck, ~15 deg,
                # ~30 deg); the bisection pass refines along it through the macro element,
                # so the tip survives a remesh even far below the Gmsh floor.
                _kind, pm, pj = arc
                assert pts[1] == pj
                p_j = self.point(pj[0], pj[1])
                # Centre from the three nodes in physical units: pyoomph's through_point
                # route declares the tiny tip collinear (|det| < 1e-10 in solver units)
                # and would silently substitute a straight line.
                circ = circle_through((r_neck, 0.0), pm, pj)
                if circ is None:
                    raise RuntimeError("tip arc nodes are collinear")
                (cx, cz), _rf = circ
                tip_arc = self.circle_arc(p_n, p_j, center=(cx, cz), name="interface")
                inner = [self.point(x, y) for (x, y) in pts[2:-1]]
                interface = [tip_arc, self.spline([p_j, *inner, p_t], name="interface")]
            else:
                inner = [self.point(x, y) for (x, y) in pts[1:-1]]
                interface = [self.spline([p_n, *inner, p_t], name="interface")]
            pb._mesh_receipt = {
                "kind": "remesh",
                "h_tip": h_tip * S,
                "r_neck": r_neck,
                "n_interface_points": len(pts),
            }

        axis = self.line(p_t, p_o, name="axis")
        plane = self.line(p_o, p_n, name="plane")
        self.plane_surface(plane, *interface, axis, name="drop")

        dist = self.add_mesh_size_field("Distance", PointsList=[p_n])
        field = self.add_mesh_size_field(
            "MathEval", F=f"min({h_max!r}, max({h_tip!r}, {k!r}*F{dist}))"
        )
        self.set_mesh_size_background_field(field)


# Timing around Gmsh generation (a remesh at the R0=1e-6 asymptotic tip stalled silently for
# 15+ minutes in runs 48 and 57 between define_geometry and the mesh load).
import pyoomph.meshes.gmsh as _pyoomph_gmsh_module

_orig_generate_mesh_to_file = _pyoomph_gmsh_module.generate_mesh_to_file


def _timed_generate_mesh_to_file(*args, **kwargs):
    t0 = time.time()
    print("GMSH: generate start", flush=True)
    result = _orig_generate_mesh_to_file(*args, **kwargs)
    print(f"GMSH: generate done in {time.time() - t0:.1f} s", flush=True)
    return result


_pyoomph_gmsh_module.generate_mesh_to_file = _timed_generate_mesh_to_file


class NeckFrameAxisymmetric(AxisymmetricCoordinateSystem):
    """Axisymmetric coordinates with the radial mesh coordinate measured from a moving origin.

    The mesh stores X = r - R_shift, with R_shift a global parameter (solver units) reset to
    the neck radius at every remesh.  In double precision the tip nodes then carry absolute
    errors of eps*|X| ~ eps*rho instead of eps*R_neck, which is what a tip radius twelve
    decades below the neck radius (R0 = 1e-6) needs; see decision record 10-la0-restart §4j.
    Only the places where the physical radius r enters are overridden: the 2 pi r measure,
    the error-estimation Jacobian and the hoop terms of vector gradient and divergence.
    Derivatives with respect to X equal those with respect to r.
    """

    def __init__(self, shift):
        super().__init__()
        self.shift = shift

    def _shift(self, with_scales: bool):
        return self.shift * scale_factor("spatial") if with_scales else self.shift

    def integral_dx(self, nodal_dim, edim, with_scale, spatial_scale, lagrangian):
        if edim >= 3:
            raise RuntimeError("Axisymmetry does not work for dimension " + str(edim))
        edim_offs = edim + 1
        r = (nondim("lagrangian_x") if lagrangian else nondim("coordinate_x")) + self.shift
        dx = nondim("dX") if lagrangian else nondim("dx")
        if with_scale:
            return spatial_scale ** edim_offs * 2 * pi * r * dx
        return 2 * pi * r * dx

    def geometric_jacobian(self):
        if self.cartesian_error_estimation:
            return Expression(1)
        return 2 * pi * (nondim("coordinate_x") + self.shift)

    def vector_gradient(self, arg, ndim, edim, with_scales, lagrangian):
        if ndim != 2:
            raise RuntimeError("NeckFrameAxisymmetric supports two-dimensional (r,z) meshes only")
        if arg.nops() != 3:
            raise RuntimeError("Cannot take a 2d axisymmetric vector gradient from a vector with dim!=2:  " + str(arg))
        x, y = self.get_coords(ndim, with_scales, lagrangian)
        r = x + self._shift(with_scales)
        res = [[diff(arg[0], x), diff(arg[0], y), 0],
               [diff(arg[1], x), diff(arg[1], y), 0], [0, 0, arg[0] / r]]
        return matrix(res)

    def vector_divergence(self, arg, ndim, edim, with_scales, lagrangian):
        if ndim != 2:
            raise RuntimeError("NeckFrameAxisymmetric supports two-dimensional (r,z) meshes only")
        coords = self.get_coords(ndim, with_scales, lagrangian)
        r = coords[0] + self._shift(with_scales)
        return diff(arg[0], coords[0]) + diff(arg[1], coords[1]) + arg[0] / r


class MappedTipMesh(GmshTemplate):
    """One quadrant of one drop, generated in tip-magnifying coordinates.

    Gmsh cannot build elements smaller than about 1e-12 of the model size, whatever the
    spatial scale (its Delaunay predicates are relative to the bounding box), and a
    Catmull-Rom spline joined to a tip arc kinks once the point spacing Gmsh can honour
    exceeds the tip radius (R0=1e-4 dev runs 26-34).  Anthony et al. avoid both by
    generating their structured mesh in stretched coordinates.  The same idea works for an
    unstructured mesh: every geometric input is mapped by the radial power law

        x' = T + (x - T) * (d'/d),   d' = L^(1-a) d^a,   d = |x - T|,

    about the neck tip T = (R_neck, 0), with 0 < a < 1 and L the drop radius.  A tip of
    radius rho becomes rho' = rho^a (L = 1): 1e-13 -> 1.6e-8 for a = 0.6, ten thousand
    times the Gmsh floor.  Gmsh generates an isotropic, linearly graded mesh in x' and the
    nodes are mapped back on loading (add_node_unique), so the solver sees a physical mesh
    whose elements are elongated radially by 1/a and whose size grows like k d towards the
    far field, exactly as before.  Directions from T are preserved by the map, so the
    symmetry plane z = 0 stays a straight line and the interface tangent at T stays
    perpendicular to it; the axis r = 0 becomes a smooth curve, represented by a spline
    through mapped samples, and its nodes return to r = 0 to spline accuracy (the axis
    Dirichlet condition then pins them exactly).  No curved entities are registered: the
    Gmsh second-order nodes already lie on the mapped curves, and mapping is exact.
    """

    def __init__(self):
        super().__init__()
        self._mapping = False
        self._T = 0.0
        self._alpha = 1.0
        self._L = 1.0
        self._shift_s = 0.0      # X = r - shift (solver units) when the neck frame is on
        self._beta = 0.5
        self._dc = float("inf")
        self._dcp = float("inf")
        self._A = 1.0

    # ---------------------------------------------------------------- the map
    # Composite radial map: exponent alpha inside the core d <= d_c (alpha = 1/2 makes the
    # mapped tip an exact parabola, which Gmsh's Catmull-Rom spline reproduces exactly; any
    # other exponent misplaces the innermost nodes, run 52), exponent beta < alpha outside,
    # continuous at d_c.  The outer compression shrinks the mapped far field and with it the
    # ratio of model size to mapped tip element that Gmsh must resolve (run 53: eleven decades
    # at the asymptotic R0=1e-6 tip is beyond its floor; beta = 0.4 gains a factor 40).
    def _g(self, d: float) -> float:
        if d <= self._dc:
            return (self._L ** (1.0 - self._alpha)) * d ** self._alpha
        return self._A * d ** self._beta

    def _ginv(self, dp: float) -> float:
        if dp <= self._dcp:
            return (dp / self._L ** (1.0 - self._alpha)) ** (1.0 / self._alpha)
        return (dp / self._A) ** (1.0 / self._beta)

    def _fwd(self, x: float, y: float) -> tuple[float, float]:
        dx, dy = x - self._T, y
        d = math.hypot(dx, dy)
        if d == 0.0:
            return x, y
        s = self._g(d) / d
        return self._T + dx * s, dy * s

    def _inv(self, xp: float, yp: float) -> tuple[float, float]:
        dx, dy = xp - self._T, yp
        dp = math.hypot(dx, dy)
        if dp == 0.0:
            return xp, yp
        s = self._ginv(dp) / dp
        return self._T + dx * s, dy * s

    def _size_mapped(self, dp: float, h_tip_p: float, k: float) -> float:
        return max(h_tip_p, k * dp)

    # Nodes arrive from the .msh file in mapped coordinates; store them physically.
    def add_node_unique(self, x, y, z):  # type: ignore[override]
        if self._mapping:
            x, y = self._inv(x + self._shift_s, y)
            x -= self._shift_s
        return super().add_node_unique(x, y, z)

    def add_node(self, x, y, z):  # type: ignore[override]
        if self._mapping:
            x, y = self._inv(x + self._shift_s, y)
            x -= self._shift_s
        return super().add_node(x, y, z)

    def _load_mesh(self, mshfilename):  # type: ignore[override]
        t0 = time.time()
        print("MESH LOAD: start", flush=True)
        super()._load_mesh(mshfilename)
        print(f"MESH LOAD: done in {time.time() - t0:.1f} s", flush=True)
        tag = getattr(self, "_dbg_tag", None)
        if tag:
            numpy.save(tag, numpy.asarray(self._mesh.points))
            self._dbg_tag = None

    def _read_curved_entities(self, fname):  # type: ignore[override]
        # The .geo_unrolled curves live in mapped coordinates; they must not be attached to
        # the (physical) facets.  Gmsh's own second-order nodes carry the geometry.
        self._curved_entities1d = {}

    # ------------------------------------------------------------- geometry
    # A curve is a list of pieces, each an exact evaluator xi in [0, 1] -> physical point:
    # Q2 interface elements (vertex, midside, vertex), analytic circle arcs, straight lines.
    @staticmethod
    def _q2_pieces(poly):
        pieces = []
        for i in range(0, len(poly) - 2, 2):
            v0, m, v1 = poly[i], poly[i + 1], poly[i + 2]

            def ev(t, v0=v0, m=m, v1=v1):
                xi = 2.0 * t - 1.0
                w0, wm, w1 = 0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)
                return (w0 * v0[0] + wm * m[0] + w1 * v1[0], w0 * v0[1] + wm * m[1] + w1 * v1[1])

            pieces.append(ev)
        if len(poly) % 2 == 0:            # odd number of segments: last one linear
            a_, b_ = poly[-2], poly[-1]
            pieces.append(lambda t, a_=a_, b_=b_: (a_[0] + t * (b_[0] - a_[0]), a_[1] + t * (b_[1] - a_[1])))
        return pieces

    @staticmethod
    def _arc_piece(cx, cz, rad, a0, a1):
        return lambda t: (cx + rad * math.cos(a0 + t * (a1 - a0)), cz + rad * math.sin(a0 + t * (a1 - a0)))

    @staticmethod
    def _line_piece(p, q):
        return lambda t: (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    def _control_points(self, pieces, S, size_at, n_fine=24, end_shrink=32, shrink_start=True):
        """Spline control points on the mapped image of the curve.

        Each piece is sampled at n_fine parameters, mapped, and the cumulative mapped chord
        length is used to place points at spacing size_at(p') with chords halving towards both
        ends (Catmull-Rom end tangents follow the end chords).  Every control point is
        evaluated exactly on the piece (the parameter is interpolated, not the position), so
        the geometry is reproduced to roundoff whatever the fine sampling; a linearly
        interpolated fine polyline gave a 4e-7 inward bias per remesh at R0=1e-3.
        """
        params, mapped = [], []
        for ip, ev in enumerate(pieces):
            for j in range(n_fine + (1 if ip == len(pieces) - 1 else 0)):
                t = j / n_fine
                x, y = ev(t)
                params.append((ip, t))
                mapped.append(self._fwd(x / S, y / S))
        s = [0.0]
        for i in range(1, len(mapped)):
            s.append(s[-1] + math.hypot(mapped[i][0] - mapped[i - 1][0], mapped[i][1] - mapped[i - 1][1]))
        total = s[-1]

        def exact_at(pos):
            lo, hi = 0, len(s) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if s[mid] <= pos:
                    lo = mid
                else:
                    hi = mid
            seg = s[hi] - s[lo]
            w = 0.0 if seg <= 0 else min(1.0, max(0.0, (pos - s[lo]) / seg))
            (ip0, t0), (ip1, t1) = params[lo], params[hi]
            if ip1 != ip0:            # fine interval crossing a piece boundary
                t1 = 1.0
            x, y = pieces[ip0](t0 + w * (t1 - t0))
            return self._fwd(x / S, y / S)

        out = [mapped[0]]
        pos = 0.0
        while True:
            tgt = size_at(out[-1])
            remaining = total - pos
            step = min(tgt, max(0.5 * remaining, tgt / end_shrink))
            if shrink_start:
                step = min(step, max(0.5 * pos, tgt / end_shrink) if pos > 0 else tgt / end_shrink)
            if remaining <= 1.5 * tgt / end_shrink:
                break
            pos += step
            out.append(exact_at(pos))
        out.append(mapped[-1])
        return out

    def define_geometry(self):
        pb = self.get_problem()
        assert isinstance(pb, StokesTipCoalescence)
        self.mesh_mode = "tris"
        self.order = 2
        S = pb.S
        L = 1.0 / S                                  # drop radius in solver units
        a = pb.tip_map_alpha
        k = pb.grading
        h_tip = pb.current_h_tip() / S               # physical tip element size, solver units
        rho = max(pb.tip_radius_lagged, pb.h_tip_floor) / S
        h_max = pb.h_max / S
        h_tip_phys = pb.current_h_tip()

        shift = pb.frame_shift_phys                 # X = r - shift (0 in the laboratory frame)
        if self.is_first_time():
            r_neck = (pb.R0 - shift) / S
            r0, z0 = pb.R0, pb.Z0
            _, zc = sphere_centre(r0, z0)
            jr, jz = sphere_bridge_junction(r0, z0)
            z_pole = zc + 1.0
            a_j = math.atan2(jz, jr - (r0 + z0))
            # Meniscus arc from the tip (angle pi) to the junction, then the sphere to the pole,
            # each split into pieces of roughly equal physical grading so the fine sampling
            # resolves the geometric grading near the tip.
            pieces = []
            ang = math.pi
            while ang > a_j:
                x, y = r0 + z0 + z0 * math.cos(ang), z0 * math.sin(ang)
                da = 0.5 * min(pb.h_max, max(h_tip_phys, k * math.hypot(x - r0, y))) / z0
                a_next = max(a_j, ang - da)
                pieces.append(self._arc_piece(r0 + z0, 0.0, z0, ang, a_next))
                ang = a_next
            ang = math.atan2(jz - zc, jr)
            while ang < math.pi / 2:
                x, y = math.cos(ang), zc + math.sin(ang)
                da = 0.5 * min(pb.h_max, max(h_tip_phys, k * math.hypot(x - r0, y)))
                a_next = min(math.pi / 2, ang + da)
                pieces.append(self._arc_piece(0.0, zc, 1.0, ang, a_next))
                ang = a_next
            if shift:
                pieces = [(lambda t, ev=ev: (ev(t)[0] - shift, ev(t)[1])) for ev in pieces]
            kind = "initial"
        else:
            poly = pb.interface_polyline()          # frame coordinate X
            r_neck = poly[0][0] / S
            z_pole = poly[-1][1]
            poly[-1] = (-shift, z_pole)             # pole on the axis r = 0
            pieces = self._q2_pieces(poly)
            kind = "remesh"
        T = r_neck                                  # tip in frame coordinates (solver units)
        r_neck_phys = r_neck * S + shift

        self._T, self._alpha, self._L = T, a, L
        b = pb.tip_map_outer if pb.tip_map_outer > 0 else a
        self._beta = b
        self._dc = pb.tip_map_core * rho if b != a else float("inf")
        self._dcp = (L ** (1.0 - a)) * self._dc ** a if b != a else float("inf")
        self._A = (L ** (1.0 - a)) * self._dc ** (a - b) if b != a else 1.0
        self._mapping = True
        # Double-precision floor.  Node coordinates near the tip carry an absolute error of
        # about eps*R_neck, so an interface element of size h on a tip of radius rho, whose
        # sagitta is h^2/(2 rho), is geometric noise unless h^2/(2 rho) >> eps R_neck.  Nodes
        # closer to the tip than h_floor are not placed (R0=1e-6 run 42: control points at
        # 1e-21 of a 1e-6 neck gave a sign-flipping curvature and a Newton divergence).  At
        # R0=1e-4 the floor is 1e-15 against a smallest tip element of 4e-14 and never acts.
        eps = 2.2e-16
        # Coordinate scale of the tip nodes: R_neck in the laboratory frame, the tip radius
        # itself in the neck-anchored frame (X = r - R_shift is O(rho) there).
        coord_scale = rho * S if pb.neck_frame else r_neck_phys
        h_floor_phys = math.sqrt(2.0 * pb.tip_roundoff_factor * eps * coord_scale * rho * S)
        # Solve-accuracy floor: node positions are O(R_neck) unknowns whose converged error is a
        # fixed fraction of R_neck, so tip elements below tip_rel_floor * R_neck are moved by
        # more than their size in one step (R0=1e-6 run 44: an element expanded 155x at
        # h = 7e-13 R_neck).  Zero disables it; 1e-11 leaves R0 >= 1e-4 untouched.
        h_floor_phys = max(h_floor_phys, pb.tip_rel_floor * r_neck_phys)
        h_floor_p = (L ** (1.0 - a)) * ((h_floor_phys / S) ** a)
        h_tip_p = max((L ** (1.0 - a)) * (rho ** a) / pb.n_tip, h_floor_p)   # mapped tip size
        expo = (a - 1.0) / a

        # Physical cap h_max in mapped units along the interface, h' = h_max * g(d)/d (the
        # tangential stretch; radial elements are 1/exponent times longer), in each branch.
        expo_o = (b - 1.0) / b
        A = self._A

        def size_mapped(p):
            dp = max(math.hypot(p[0] - T, p[1]), 1e-3 * h_tip_p)   # clamp: negative powers
            cap = h_max * (dp / L) ** expo
            if b != a:
                cap = min(cap, h_max * A * (dp / A) ** expo_o)
            return min(cap, max(h_tip_p, k * dp))

        f_ctrl = pb.tip_map_control_fraction
        t_geo = time.time()
        iface_p = self._control_points(pieces, S, lambda p: max(f_ctrl * size_mapped(p), h_floor_p),
                                       shrink_start=False)
        # Axis r = 0 (X = -shift) from the pole to the origin: straight, curved in the map.
        axis_pieces = []
        z = z_pole
        while z > 0.0:
            dz = 0.5 * min(pb.h_max, max(h_tip_phys, k * math.hypot(r_neck_phys, z)))
            z_next = max(0.0, z - dz)
            axis_pieces.append(self._line_piece((-shift, z), (-shift, z_next)))
            z = z_next
        axis_p = self._control_points(axis_pieces, S, lambda p: f_ctrl * size_mapped(p))
        fine = []

        import os
        dbg = os.environ.get("LA0_MAP_DEBUG")
        if dbg:
            numpy.save(f"{dbg}/ctrl_iface_{pb.n_remesh:03d}.npy", numpy.array(iface_p))
            numpy.save(f"{dbg}/ctrl_axis_{pb.n_remesh:03d}.npy", numpy.array(axis_p))
            self._dbg_tag = f"{dbg}/nodes_{pb.n_remesh:03d}.npy"
        self.default_resolution = h_max
        self.set_gmsh_parameter("General.NumThreads", 1)
        self.set_gmsh_parameter("Geometry.Tolerance", 1e-15)
        self.set_gmsh_parameter("Mesh.MeshSizeMin", h_tip_p)
        self.set_gmsh_parameter("Mesh.MeshSizeMax", 10.0 * h_max)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeFromPoints", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeExtendFromBoundary", 0)
        self.set_gmsh_parameter("Mesh.Algorithm", 6)
        # Gmsh perturbs the boundary points of its 2D Delaunay by RandomFactor times the
        # model size (default 1e-9); with the mapped tip at 1e-11 of the model size that
        # perturbation is the meshing floor (run 49: surface returned empty at 7e-12).
        self.set_gmsh_parameter("Mesh.RandomFactor", pb.gmsh_random_factor)

        self._shift_s = 0.0          # geometry is already in frame coordinates
        p_n = self.point(T, 0.0, consider_spatial_scale=False)
        p_t = self.point(*iface_p[-1], consider_spatial_scale=False)
        p_o = self.point(*axis_p[-1], consider_spatial_scale=False)
        assert math.hypot(iface_p[-1][0] - axis_p[0][0], iface_p[-1][1] - axis_p[0][1]) < 1e-9 * L, "pole mismatch"
        inner = [self.point(x, y, consider_spatial_scale=False) for (x, y) in iface_p[1:-1]]
        interface = self.spline([p_n, *inner, p_t], name="interface", with_macro_element=False)
        ax_inner = [self.point(x, y, consider_spatial_scale=False) for (x, y) in axis_p[1:-1]]
        axis = self.spline([p_t, *ax_inner, p_o], name="axis", with_macro_element=False)
        plane = self.line(p_o, p_n, name="plane")
        self.plane_surface(plane, interface, axis, name="drop")

        dist = self.add_mesh_size_field("Distance", PointsList=[p_n])
        # Physical cap h_max expressed in mapped units: h' = h (d'/d) = h_max (d'/L)^((a-1)/a).
        dclamp = f"max(F{dist},{1e-3 * h_tip_p!r})"
        cap_expr = f"{h_max!r}*({dclamp}/{L!r})^({expo!r})"
        if b != a:
            cap_expr = f"min({cap_expr}, {h_max * A!r}*({dclamp}/{A!r})^({expo_o!r}))"
        field = self.add_mesh_size_field(
            "MathEval",
            F=f"min({cap_expr}, max({h_tip_p!r}, {k!r}*F{dist}))",
        )
        self.set_mesh_size_background_field(field)
        print(f"MESH GEOMETRY: {kind}, {len(iface_p)} interface + {len(axis_p)} axis control points, "
              f"h_tip_mapped={h_tip_p:.3e}, bbox_mapped~{self._g(2.0 * L):.3e}, h_floor={h_floor_phys:.3e}, shift={pb.frame_shift_phys:.6e}, "
              f"{time.time() - t_geo:.1f} s", flush=True)
        pb._mesh_receipt = {
            "kind": kind,
            "h_tip": h_tip * S,
            "h_tip_mapped": h_tip_p,
            "h_floor_phys": h_floor_phys,
            "r_neck": r_neck_phys,
            "alpha": a,
            "n_interface_points": len(iface_p),
            "n_axis_points": len(axis_p),
        }

class StokesTipCoalescence(Problem):
    """Driver: fixed-ratio implicit stepping, tip-tracking remeshes, neck diagnostics."""

    def __init__(
        self,
        *,
        R0: float,
        Z0: float,
        output_dir: str | Path,
        n_tip: int = 16,
        grading: float = 0.2,
        h_max: float = 0.1,
        remesh_growth: float = 1.5,
        tip_shrink_remesh: float = 0.7,
        tip_grow_remesh: float = 3.0,
        dt_fraction: float = 0.02,
        dt_initial: float = 1e-9,
        dt_growth: float = 1.3,
        curvature_change_target: float = 0.05,
        newton_tolerance: float = 1e-7,
        R_stop: float = 0.03,
        h_tip_floor: float = 1e-11,
        smooth_half_window: int = 3,
        smooth_tip_span: float = 200.0,
        detect_inversions: bool = False,
        neck_stretch: bool = True,
        tip_translation_span: float = 0.3,
        tip_refine: bool = False,
        gmsh_floor: float = 1e-11,
        refine_span: float = 40.0,
        max_refine_rounds: int = 6,
        bisect_floor: float = 2e-13,
        spatial_scale: float | None = None,
        interface_translation: bool = True,
        anticipation: float = 0.5,
        max_bisect_levels: int = 5,
        tip_map_alpha: float = 0.0,
        tip_map_control_fraction: float = 0.25,
        tip_fit_angle: float = 0.15,
        tip_roundoff_factor: float = 100.0,
        tip_rel_floor: float = 0.0,
        neck_frame: bool = False,
        gmsh_random_factor: float = 1e-9,
        tip_map_outer: float = 0.0,
        tip_map_core: float = 1e3,
        max_residuals: float = 1e10,
        extra_newton_iterations: int = 0,
    ):
        super().__init__()
        self.R0 = float(R0)
        self.Z0 = float(Z0)
        self.n_tip = int(n_tip)
        self.grading = float(grading)
        self.h_max = float(h_max)
        self.remesh_growth = float(remesh_growth)
        self.tip_shrink_remesh = float(tip_shrink_remesh)
        self.tip_grow_remesh = float(tip_grow_remesh)
        self.dt_fraction = float(dt_fraction)
        self.dt_initial = float(dt_initial)
        self.dt_growth = float(dt_growth)
        self.curvature_change_target = float(curvature_change_target)
        self._last_curvature_ratio = 0.0
        self.newton_tolerance = float(newton_tolerance)
        self.R_stop = float(R_stop)
        self.h_tip_floor = float(h_tip_floor)
        self.smooth_half_window = int(smooth_half_window)
        self.smooth_tip_span = float(smooth_tip_span)
        self.detect_inversions = bool(detect_inversions)
        self.neck_stretch = bool(neck_stretch)
        self.tip_translation_span = float(tip_translation_span)
        self.tip_refine = bool(tip_refine)
        self.gmsh_floor = float(gmsh_floor)
        self.refine_span = float(refine_span)
        self.max_refine_rounds = int(max_refine_rounds)
        # Absolute element-size floor for bisection.  Below ~1e-13 in a unit domain the
        # assembled system loses too many digits and Newton diverges at any step size
        # (R0=1e-4 dev runs: marginal at 1.5e-13, fatal at 3.7e-14).
        self.bisect_floor = float(bisect_floor)
        # pyoomph spatial scale: the solver stores coordinates as r/S.  With S = R0 the
        # tip elements are O(1e-9) in solver units instead of O(1e-13), which moves the
        # absolute-size cliff seen at R0=1e-4 out of reach.  Physical values are used
        # everywhere in this class; only Gmsh sizes and element midpoints need converting.
        self.S = float(spatial_scale) if spatial_scale else 1.0
        self.interface_translation = bool(interface_translation)
        self.anticipation = float(anticipation)
        # Bisection depth below the Gmsh base mesh.  Seven or more levels diverged in every
        # test (R0=1e-4 runs 21/22/24 at level 7; R0=1e-3 with a 1e-6 floor at 13 and 15);
        # five was always stable.
        self.max_bisect_levels = int(max_bisect_levels)
        # Exponent of the tip-magnifying map used by MappedTipMesh; 0 selects the plain
        # tip-graded template.
        self.tip_map_alpha = float(tip_map_alpha)
        # Spline control-point spacing in the mapped plane as a fraction of the local mesh
        # size; Gmsh's geo spline is a uniform-parameter Catmull-Rom, whose interpolation error
        # on a unit-curvature curve is ~1e-5 at 0.25*0.2 chords and ~1e-9 at 0.25*0.02.
        self.tip_map_control_fraction = float(tip_map_control_fraction)
        # Circle-fit window for the tip radius: chord angle from the tip tangent (radians).
        self.tip_fit_angle = float(tip_fit_angle)
        # Smallest tip-element sagitta allowed, in units of eps*R_neck (MappedTipMesh floor).
        self.tip_roundoff_factor = float(tip_roundoff_factor)
        # Smallest tip element relative to R_neck (MappedTipMesh); 0 = off.
        self.tip_rel_floor = float(tip_rel_floor)
        # Neck-anchored radial frame X = r - R_shift (NeckFrameAxisymmetric); requires tip_map.
        self.neck_frame = bool(neck_frame)
        self.gmsh_random_factor = float(gmsh_random_factor)
        # Composite map: outer exponent (0 = same as tip_map_alpha) beyond tip_map_core tip radii.
        self.tip_map_outer = float(tip_map_outer)
        self.tip_map_core = float(tip_map_core)
        # oomph-lib Newton residual cap; scales with 1/(Z0 S^2) like the tolerance.
        self.max_residuals_cap = float(max_residuals)
        self.extra_newton_iterations = int(extra_newton_iterations)
        self.frame_shift_phys = float(R0) if neck_frame else 0.0
        self._R_shift = None
        self._needs_tip_refine = False
        self.n_refine_events = 0
        self._thin_boost = 1.0
        self._R_ref = None

        self.output_root = Path(output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "profiles").mkdir(exist_ok=True)

        # Lagged mesh parameters (set at each remesh).
        self.tip_radius_lagged = self.Z0
        self.rho_at_remesh = self.Z0
        self.rmin_at_remesh = self.R0
        self.last_rmin = self.R0
        self.n_remesh = 0
        self._step_at_remesh = 0
        self._mesh_receipt: dict[str, Any] = {}
        self._pending_remesh_reason: str | None = None
        self._template: TipGradedQuadrantMesh | MappedTipMesh | None = None
        self._neck_csv = self.output_root / "neck.csv"
        self._progress_path = self.output_root / "progress.jsonl"
        self._wrote_header = False
        self._last_profile_bin: int | None = None
        self._steps = 0
        self._dt = self.dt_initial
        self._dt_prev: float | None = None
        self._predict_now = False
        self._wall0 = time.time()

    # ------------------------------------------------------------------ mesh
    def current_h_tip(self) -> float:
        floor = self.gmsh_floor if self.tip_refine else self.h_tip_floor
        return max(self.tip_radius_lagged / self.n_tip, floor)

    def rho_resolvable(self) -> float:
        """Smallest tip radius the mapped mesh can resolve with n_tip elements, given the
        double-precision floors recorded at the last mesh build (0 when no floor applies)."""
        h_floor = float(self._mesh_receipt.get("h_floor_phys", 0.0) or 0.0)
        return self.n_tip * h_floor

    def tip_unresolved(self, rho: float) -> bool:
        """True while the measured tip radius is within a factor 4 of the resolvable one: the
        fit then reads mesh noise and must drive neither remeshing nor the time step (R0=1e-6
        run 45 thrashed between 'shrank' and 'grew' remeshes at one instant)."""
        rr = self.rho_resolvable()
        return rr > 0.0 and (not math.isfinite(rho) or rho < 4.0 * rr)

    def define_problem(self):
        if self.neck_frame:
            if not self.tip_map_alpha > 0:
                raise RuntimeError("neck_frame requires the mapped template (tip_map_alpha > 0)")
            self._R_shift = self.define_global_parameter(R_shift=self.frame_shift_phys / self.S)
            self.set_coordinate_system(NeckFrameAxisymmetric(self._R_shift))
        else:
            self.set_coordinate_system("axisymmetric")
        if self.S != 1.0:
            self.set_scaling(spatial=self.S)
        self.set_output_directory(str(self.output_root / "pyoomph"))
        self.newton_solver_tolerance = self.newton_tolerance
        self.max_residuals = self.max_residuals_cap
        self.max_newton_iterations = 12
        if self.tip_refine:
            self.max_refinement_level = 20
        self.write_states = False
        self._template = MappedTipMesh() if self.tip_map_alpha > 0 else TipGradedQuadrantMesh()
        self.add_mesh(self._template)

        eqs = StokesEquations(dynamic_viscosity=1.0, mode="TH")
        eqs += LaplaceSmoothedMesh()
        # Safety net only; the physics-driven triggers live in actions_after_newton_solve.
        eqs += RemeshWhen(
            RemeshingOptions(
                max_expansion=4.0,
                min_expansion=0.25,
                min_quality_decrease=0.2,
                on_inverted_element=self.detect_inversions,
            )
        )
        eqs += IntegralObservables(volume=1)
        # Physical radius offset as a dimensional expression (zero in the laboratory frame).
        shift_phys = self._R_shift * scale_factor("spatial") if self.neck_frame else 0
        if self.neck_frame:
            # The axis r = 0 sits at X = -R_shift; AxisymmetryBC would pin X = 0.
            eqs += DirichletBC(velocity_x=0, mesh_x=-shift_phys) @ "axis"
        else:
            eqs += AxisymmetryBC() @ "axis"
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "plane"
        eqs += NavierStokesFreeSurface(surface_tension=1.0) @ "interface"
        if self.neck_stretch:
            # Make the mesh translate with the neck.  A global unknown R_neck equals the
            # neck node's radial position (point constraint at the interface/plane corner,
            # admissible because the corner sits at r=R_min, not on the axis).  Along the
            # symmetry plane the nodes stretch affinely with it, x = X R_neck/R_ref, where
            # X is the Lagrangian coordinate and R_ref the neck radius at the last remesh.
            # The stretch multiplier is pinned at the corner itself, where the condition
            # would duplicate the point constraint.
            self._R_ref = self.define_global_parameter(R_ref=self.R0)
            self.add_equations(
                (GlobalLagrangeMultiplier(R_neck=0) + InitialCondition(R_neck=self.R0)) @ "globals"
            )
            Rn = var("R_neck", domain="globals")
            eqs += EnforcedBC(mesh_x=var("mesh_x") + shift_phys - (var("lagrangian_x") + shift_phys) * Rn / self._R_ref) @ "plane"
            eqs += DirichletBC(_lagr_enf_bc_mesh_x=0) @ "plane/interface"
            if self.neck_frame:
                eqs += DirichletBC(_lagr_enf_bc_mesh_x=0) @ "plane/axis"
            eqs += WeakContribution(var("mesh_x") + shift_phys - Rn, testfunction("R_neck", domain="globals")) @ "interface/plane"
            # Tangential motion of the interface: a radial translation by (R_neck - R_ref)
            # decaying with Lagrangian distance from the neck over L = tip_translation_span*R_ref,
            # imposed through a tangential Lagrange multiplier.  The normal motion stays with
            # the kinematic condition.  Pinned at both corners where the tangent is already fixed.
        if self.neck_stretch and self.interface_translation:
            X = var("lagrangian")
            s2 = (X[0] + shift_phys - self._R_ref) ** 2 + X[1] ** 2
            shift = (Rn - self._R_ref) * exp(-s2 / (self.tip_translation_span * self._R_ref) ** 2)
            tvec = vector(-var("normal_y"), var("normal_x"))
            target = var("mesh") - X - vector(shift, 0)
            iface = ScalarField("lam_t", space="C2")
            iface += WeakContribution(var("lam_t"), dot(tvec, testfunction("mesh")), coordsys=cartesian)
            iface += WeakContribution(dot(target, tvec), testfunction("lam_t"), coordsys=cartesian)
            iface += DirichletBC(lam_t=0) @ "plane"
            iface += DirichletBC(lam_t=0) @ "axis"
            eqs += iface @ "interface"
        self.add_equations(eqs @ "drop")

    # ----------------------------------------------------------- geometry io
    def _interface_nodes(self):
        from pyoomph.meshes.meshdatacache import MeshDataCache

        data = MeshDataCache(tesselate_tri=False, nondimensional=False).get_data(
            self.get_mesh("drop/interface")
        )
        # Frame coordinate X (= r - R_shift in the neck frame): all tip-local geometry is done
        # in X; adding the shift first would round the tip away (eps R_neck ~ 2e-22 against
        # innermost-chord sagittas ~1e-23 at a 1e-18 tip, runs 48-55).
        r = data.get_data("coordinate_x")
        z = data.get_data("coordinate_y")
        u = data.get_data("velocity_x")
        v = data.get_data("velocity_y")
        return data, r, z, u, v

    def interface_polyline(self) -> list[tuple[float, float]]:
        from pyoomph.meshes.ordering import sort_line_segments

        data, r, z, _, _ = self._interface_nodes()
        segs, _ = data.get_interface_line_segments()
        pts = data.get_coordinates()
        segs = sort_line_segments(pts, segs, sort_along_axis="y+", whom="interface")
        poly = [(float(pts[0, i]), float(pts[1, i])) for seg in segs for i in seg]
        # Remove consecutive duplicates that segment joins produce.
        out: list[tuple[float, float]] = []
        for p in poly:
            if not out or (abs(out[-1][0] - p[0]) > 0 or abs(out[-1][1] - p[1]) > 0):
                out.append(p)
        return out

    def interface_polyline_for_remesh(self):
        """Return (points, arc) for the remesh geometry.

        ``arc`` is None for the plain spline route.  With tip bisection it is
        ``((cx, cz), rho, p_join)``: a least-squares circle through the interface nodes
        within one tip radius of the neck and the join point (projected onto that
        circle) where the spline takes over; ``points`` then start at ``p_join``.
        """
        pts = self.interface_polyline()
        if len(pts) < 4:
            raise RuntimeError("interface polyline too short for remeshing")
        # Smooth only the tip neighbourhood (within smooth_tip_span tip radii of the neck);
        # the bulk points stay exactly on the old discrete interface.
        span = self.smooth_tip_span * max(self.tip_radius_lagged, self.h_tip_floor)
        r0, z0 = pts[0]
        n_tip_pts = 1
        while n_tip_pts < len(pts) - 1 and math.hypot(pts[n_tip_pts][0] - r0, pts[n_tip_pts][1] - z0) < span:
            n_tip_pts += 1
        n_tip_pts = min(len(pts) - 1, n_tip_pts + self.smooth_half_window)
        if not self.tip_refine:
            head = smooth_polyline_arclength(pts[: n_tip_pts + 1], self.smooth_half_window)
            pts = head + pts[n_tip_pts + 1 :]
        if self.tip_refine:
            # Bisected tip nodes may be far below the Gmsh floor; Gmsh's kernel and the
            # macro-element spline inversion cannot take spacing that small in a unit
            # domain, so thin the polyline to the floor.  The bisection pass after the
            # remesh restores the tip resolution on the new spline.
            # The tip itself is carried by the circle arc; beyond it the spline only needs
            # Gmsh-floor resolution, and pyoomph's spline parameter inversion fails on
            # segments much shorter than the floor ("Cannot invert spline").
            # Geometrically graded sampling (spacing ~ k * distance from the neck, floored at
            # twice the Gmsh floor): abrupt spacing jumps between bisected and Gmsh-sized
            # nodes make the Catmull-Rom spline wiggly and pyoomph's Gauss-Newton parameter
            # inversion then fails ("Cannot invert spline").
            # Thin only the tip zone (bisected nodes), grading the spacing from twice the
            # Gmsh floor up to the Gmsh node spacing; keep every node beyond it, or the far
            # interface and the drop volume are misrepresented (run28: 0.5% volume loss).
            r_t, z_t = pts[0]
            h_g = self.current_h_tip()
            zone = h_g / self.grading * 4.0   # where Gmsh's graded field takes over
            kept = [pts[0]]
            for q in pts[1:-1]:
                d_tip = math.hypot(q[0] - r_t, q[1] - z_t)
                if d_tip > zone:
                    kept.append(q)
                    continue
                gap = max(4.0 * self.gmsh_floor * self._thin_boost, 0.5 * self.grading * d_tip)
                if math.hypot(q[0] - kept[-1][0], q[1] - kept[-1][1]) >= gap:
                    kept.append(q)
            kept.append(pts[-1])
            pts = kept
        # Exact symmetry-plane and axis end points.
        pts[0] = (pts[0][0], 0.0)
        pts[-1] = (0.0, pts[-1][1])
        if pts[0][1] != 0.0 or pts[-1][0] != 0.0 or pts[0][0] <= 0.0:
            raise RuntimeError(f"unexpected interface ends: {pts[0]}, {pts[-1]}")
        if not self.tip_refine:
            return pts, None
        # Circle fit over the raw nodes within one tip radius of the neck.
        raw = self.interface_polyline()
        st = self.neck_state()
        rho = st["tip_radius"]
        r0 = pts[0][0]
        near = [q for q in raw if math.hypot(q[0] - r0, q[1]) <= 1.0 * rho]
        circ = fit_circle_kasa(near) if len(near) >= 5 else None
        if circ is None:
            return pts, None
        (cx, cz), rf = circ
        # Join at ~30 degrees up the meniscus (z ~ rf/2), where the profile is still
        # circular to high accuracy, and pass the arc through actual interface nodes
        # (neck, a node near 15 degrees, the join node) so the join has no tangent kink.
        z_join = 0.8 * rf
        j = next((i for i, q in enumerate(raw) if q[1] >= z_join), None)
        if j is None or j < 2:
            return pts, None
        pj = raw[j]
        m = next((i for i, q in enumerate(raw) if q[1] >= 0.5 * z_join), None)
        if m is None or m < 1 or m >= j:
            return pts, None
        pm = raw[m]
        # Keep the true neck first, then the join, then the (thinned) polyline beyond it.
        rest = [pts[0], pj] + [q for q in pts[1:] if q[1] > pj[1] + 1e-300]
        if len(rest) < 5:
            return pts, None
        return rest, ("through", pm, pj)

    # ---------------------------------------------------------- diagnostics
    def neck_state(self) -> dict[str, float]:
        _, r, z, u, v = self._interface_nodes()
        n = len(r)
        # Neck node: on the plane z=0 with maximal r.
        # Plane nodes carry mesh_y = 0 exactly (Dirichlet).  Any finite tolerance fails on the
        # mapped R0=1e-4 mesh, whose first interface nodes above the plane sit at z ~ 3e-14
        # (1e-14 and 1e-13 both picked one of them as the neck in runs 38 and 39).
        on_plane = [i for i in range(n) if abs(z[i]) <= 1e-30]
        i0 = max(on_plane, key=lambda i: r[i]) if on_plane else int(z.argmin())
        p0 = (float(r[i0]), float(z[i0]))
        rmin = p0[0] + self.frame_shift_phys      # physical neck radius
        # Least-squares circle over the interface nodes within ~1 lagged tip radius of the
        # neck (at least the five nearest).  Far more robust than a three-node circle when the
        # element size at the tip varies between remeshes.
        d2 = (r - p0[0]) ** 2 + (z - p0[1]) ** 2
        order = sorted(range(n), key=lambda i: d2[i])
        # Scale-free fit window: nodes whose chord from the neck makes an angle of at most
        # tip_fit_angle with the tip tangent (the z axis), i.e. about 2*tip_fit_angle of arc
        # on a circle, at least seven nodes.  A fixed node count spans a thousandth of a tip
        # radius on the mapped mesh (run 36), and a window in lagged radii collapses once the
        # tip radius grows again as R_min^3 (run 37: 5.6e-14 read for a 1.4e-12 tip).
        near = [i0]
        for i in order[1:]:
            # Walk outwards in distance and stop at the first node beyond the angle; nodes
            # far up the drop near the axis have |r - R_min| small and must not qualify.
            # Nodes at or below the tip level (other plane nodes) are skipped, not fatal.
            if z[i] - p0[1] <= 0.0:
                continue
            if math.atan2(abs(r[i] - p0[0]), z[i] - p0[1]) > self.tip_fit_angle:
                break
            near.append(i)
        if len(near) < 7:
            near = order[: min(7, n)]
        p1 = (float(r[order[1]]), float(z[order[1]]))
        circ = fit_circle_kasa([(float(r[i]), float(z[i])) for i in near])
        if circ is None:
            kappa_m = 0.0
            tip_radius = float("inf")
        else:
            (cx, _), rho = circ
            tip_radius = rho
            kappa_m = -1.0 / rho if cx > p0[0] else 1.0 / rho
        two_h = kappa_m + 1.0 / rmin
        h_tip_now = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        vol = float(self.get_mesh("drop").evaluate_observable("volume"))
        return {
            "t": float(self.get_current_time(dimensional=False, as_float=True)),
            "R_min": rmin,
            "u_neck": float(u[i0]),
            "kappa_m": kappa_m,
            "two_H": two_h,
            "tip_radius": tip_radius,
            "h_tip_now": h_tip_now,
            "volume": vol,
        }

    def bridge_half_height(self, rmin: float, poly: list[tuple[float, float]]) -> float:
        target = 1.05 * rmin - self.frame_shift_phys      # poly is in the frame coordinate
        for (r1, z1), (r2, z2) in zip(poly[:-1], poly[1:]):
            if (r1 - target) * (r2 - target) <= 0.0 and r1 != r2:
                w = (target - r1) / (r2 - r1)
                return z1 + w * (z2 - z1)
        return float("nan")

    def write_row(self, state: dict[str, float], dt: float, newton_iters: int | None) -> None:
        poly = self.interface_polyline()
        row = dict(state)
        row["Z_b"] = self.bridge_half_height(state["R_min"], poly)
        row["dt"] = dt
        row["ndof"] = int(self.ndof())
        row["n_remesh"] = self.n_remesh
        row["newton_iters"] = newton_iters if newton_iters is not None else -1
        row["curv_change"] = self._last_curvature_ratio
        row["wall_s"] = time.time() - self._wall0
        fields = [
            "t", "R_min", "u_neck", "two_H", "kappa_m", "tip_radius", "Z_b",
            "h_tip_now", "volume", "dt", "ndof", "n_remesh", "newton_iters", "curv_change", "wall_s",
        ]
        new = not self._neck_csv.exists()
        with self._neck_csv.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if new or not self._wrote_header:
                w.writeheader()
                self._wrote_header = True
            w.writerow({k: row[k] for k in fields})
        with self._progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "physical_time": state["t"],
                "primary_observable": state["R_min"],
                "degrees_of_freedom": int(self.ndof()),
                "nonlinear_iterations": newton_iters if newton_iters is not None else -1,
                "timestep": dt,
                "remesh_events": self.n_remesh,
                "neck_velocity": state["u_neck"],
                "two_H": state["two_H"],
                "wall_s": row["wall_s"],
            }) + "\n")
        rb = math.floor(math.log10(state["R_min"]) * 10.0)  # every 0.1 decade
        if self._last_profile_bin != rb:
            self._last_profile_bin = rb
            self.write_profile(poly, state)

    def write_profile(self, poly: list[tuple[float, float]], state: dict[str, float], tag: str = "") -> None:
        path = self.output_root / "profiles" / (
            f"interface_t{state['t']:.8e}_R{state['R_min']:.8e}{tag}.dat"
        )
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# t={state['t']} R_min={state['R_min']} two_H={state['two_H']}\n")
            fh.write("r z\n")
            for r, z in poly:
                fh.write(f"{r + self.frame_shift_phys:.16e} {z:.16e}\n")

    # ---------------------------------------------------------- h-refinement
    def refine_tip(self) -> int:
        """Bisect quads near the neck tip until their size is below rho_tip/n_tip.

        Gmsh cannot produce elements much below ~1e-12 in a unit domain (its Delaunay
        predicates are relative to the model size), so the template stops at
        ``gmsh_floor`` and oomph-lib's quadtree refinement takes the tip further.  New
        boundary nodes land on the macro-element geometry.  The time integrator is
        restarted impulsively afterwards, as after a remesh.
        """
        if not self.tip_refine:
            return 0
        mesh = self.get_mesh("drop")
        total = 0
        for _round in range(self.max_refine_rounds):
            st = self.neck_state()
            rho = st["tip_radius"]
            # On a fresh Gmsh mesh the nearest nodes sit on floor-sized elements and the
            # circle fit over-reports the tip radius; the pre-remesh value is the better
            # estimate until the tip is resolved again.
            if math.isfinite(self.tip_radius_lagged) and self.tip_radius_lagged > 0:
                rho = min(rho, self.tip_radius_lagged) if (math.isfinite(rho) and rho > 0) else self.tip_radius_lagged
            if not (math.isfinite(rho) and rho > 0):
                break
            # Bisection happens only on a fresh Gmsh mesh (exact macro-element geometry);
            # resolve the tip for the shrinkage expected before the next remesh trigger.
            target = max(self.anticipation * rho / self.n_tip, self.bisect_floor,
                         self.current_h_tip() / 2 ** self.max_bisect_levels)
            r0, z0 = st["R_min"], 0.0
            idx = []
            S = self.S
            for i, e in enumerate(mesh.elements()):
                # Element size from its vertex nodes (physical units); the cached size API
                # was observed to over-report after a remesh and drove bisection far past
                # the floor.
                nv = e.nvertex_node()
                xs = [e.vertex_node_pt(k).x(0) * S for k in range(nv)]
                ys = [e.vertex_node_pt(k).x(1) * S for k in range(nv)]
                # sqrt of the polygon area of the vertex ring (shoelace); vertex order for a
                # Q element is (0,1,3,2) in oomph-lib's tensor-product numbering.
                order = (0, 1, 3, 2) if nv == 4 else tuple(range(nv))
                area = 0.0
                for a_, b_ in zip(order, order[1:] + order[:1]):
                    area += xs[a_] * ys[b_] - xs[b_] * ys[a_]
                h = math.sqrt(abs(area) * 0.5)
                if h <= target:
                    continue
                x, y = sum(xs) / nv, sum(ys) / nv
                d = math.hypot(x - r0, y - z0)
                # Same geometric grading as the Gmsh field: an element is too coarse only if it
                # is larger than both the tip target and k times its distance from the tip.
                if h > max(target, self.grading * d):
                    idx.append(i)
            if not idx:
                break
            self.actions_before_adapt()
            mesh.refine_selected_elements(idx)
            self.relink_external_data()
            self.actions_after_adapt()
            total += len(idx)
        if total:
            # Children inherit the parent's recorded size, which would trip RemeshWhen's
            # min_expansion test at once; the bisected mesh is the new reference.
            for e in mesh.elements():
                e.set_initial_cartesian_nondim_size(e.get_current_cartesian_nondim_size())
                e.set_initial_quality_factor(e.get_quality_factor())
            self.assign_initial_values_impulsive()
            self.timestepper.set_num_unsteady_steps_done(0)
            self._taken_already_an_unsteady_step = False
            self._dt_prev = None
            self.n_refine_events += 1
            st = self.neck_state()
            print(
                f"TIP REFINE: {total} elements bisected in rounds; ndof={self.ndof()} "
                f"h_tip_now={st['h_tip_now']:.3e} rho={st['tip_radius']:.3e}",
                flush=True,
            )
        return total

    # -------------------------------------------------------------- snapshot
    def _dof_snapshot(self):
        """Mesh-agnostic snapshot: time, all current values (dofs and pinned, positions
        included).  Unlike Problem._snapshot_state it does not rebuild the mesh on restore,
        so it survives selective bisection."""
        t = float(self.get_current_time(dimensional=False, as_float=True))
        dofs, _pos, pinned = self._get_all_values_at_current_time(True)
        return t, numpy.array(dofs, copy=True), numpy.array(pinned, copy=True)

    def _dof_restore(self, snap) -> None:
        t, dofs, pinned = snap
        self.set_all_values_at_current_time(dofs, pinned, True)
        self.set_current_time(t, dimensional=False)
        # A failed unsteady solve has shifted the history; restart impulsively (BDF1 step).
        self.assign_initial_values_impulsive()
        self.timestepper.set_num_unsteady_steps_done(0)
        self._taken_already_an_unsteady_step = False
        self._dt_prev = None
        self.invalidate_cached_mesh_data()

    # -------------------------------------------------------------- predictor
    def actions_before_newton_solve(self):
        """Seed Newton with a linear extrapolation of all dofs (positions included).

        At this point oomph-lib has shifted the history, so history level 1 is x_n and
        level 2 is x_{n-1}; the current values still equal x_n.  Starting Newton from
        x_n + (dt/dt_prev)(x_n - x_{n-1}) places the tip mesh where it will be after the
        translation, which is what makes u*dt >> tip radius steps converge.  After an
        impulsive restart the two history levels coincide and the predictor is inert.
        """
        if self._predict_now and self._dt_prev is not None and self._dt_prev > 0:
            try:
                x1 = numpy.asarray(self.get_history_dofs(1), dtype=float)
                x2 = numpy.asarray(self.get_history_dofs(2), dtype=float)
                if x1.shape == x2.shape and x1.size > 0:
                    self.set_current_dofs(x1 + (self._dt / self._dt_prev) * (x1 - x2))
            except Exception as exc:  # never let the predictor kill a step
                print(f"predictor skipped: {exc!r}", flush=True)
        super().actions_before_newton_solve()

    # -------------------------------------------------------------- remeshing
    def actions_after_newton_solve(self):
        if not self.last_newton_step_failed() and self._template is not None:
            reason = self._remesh_reason()
            if reason is not None:
                self._pending_remesh_reason = reason
                self._domains_to_remesh.add(self._template)
        super().actions_after_newton_solve()

    def _remesh_reason(self) -> str | None:
        st = self.neck_state()
        self.last_rmin = st["R_min"]
        rho = st["tip_radius"]
        if self._steps - self._step_at_remesh < 3:
            return None
        if st["R_min"] >= self.remesh_growth * self.rmin_at_remesh:
            return f"bridge grew x{st['R_min'] / self.rmin_at_remesh:.2f}"
        ref = self.rho_at_remesh
        if self.tip_unresolved(rho):
            return None
        if math.isfinite(rho) and rho < self.tip_shrink_remesh * ref:
            return f"tip radius shrank to {rho:.3e} from {ref:.3e}"
        if math.isfinite(rho) and rho > self.tip_grow_remesh * ref:
            return f"tip radius grew to {rho:.3e} from {ref:.3e}"
        return None

    def actions_before_remeshing(self, active_remeshers):
        st = self.neck_state()
        rho = st["tip_radius"]
        if self.tip_unresolved(rho):
            # Build the next mesh at the floor; the measured radius is noise there.
            rho = self.rho_resolvable()
        if math.isfinite(rho) and rho > 0:
            self.tip_radius_lagged = rho
            self.rho_at_remesh = rho
        self.rmin_at_remesh = st["R_min"]
        self.last_rmin = st["R_min"]
        if self._R_ref is not None:
            self._R_ref.value = st["R_min"]
        self._pre_remesh_state = st
        self._pre_remesh_poly = self.interface_polyline()
        if self.neck_frame:
            # Move the frame origin to the neck: translate every node by -dX and advance the
            # parameter, a physically identical state, before the new mesh is generated and
            # the old one interpolated in the same frame.
            new_shift = st["R_min"]
            dX = (new_shift - self.frame_shift_phys) / self.S
            if dX != 0.0:
                for node in self.get_mesh("drop").nodes():
                    node.set_x(0, node.x(0) - dX)
                self._R_shift.value = new_shift / self.S
                self.frame_shift_phys = new_shift
        super().actions_before_remeshing(active_remeshers)

    def actions_after_remeshing(self):
        super().actions_after_remeshing()
        # Stokes carries no velocity history, and the interpolated position history of a
        # refined tip is accurate only to the OLD element scale, which corrupts the BDF2
        # mesh velocity.  Restart the time integrator impulsively from the new geometry
        # (one first-order step) instead of trusting the interpolated history.
        self.assign_initial_values_impulsive()
        self.timestepper.set_num_unsteady_steps_done(0)
        self._taken_already_an_unsteady_step = False
        self._dt_prev = None
        self._step_at_remesh = self._steps
        self._needs_tip_refine = True
        self.n_remesh += 1
        st_new = self.neck_state()
        st_old = getattr(self, "_pre_remesh_state", {})
        rec = {
            "index": self.n_remesh,
            "t": st_new["t"],
            "reason": self._pending_remesh_reason,
            "before": st_old,
            "after": st_new,
            "mesh": self._mesh_receipt,
            "ndof": int(self.ndof()),
        }
        self._pending_remesh_reason = None
        # Interface before (old mesh nodes) and after (new mesh nodes) for geometric audit.
        self.write_profile(getattr(self, "_pre_remesh_poly", []), st_old or st_new, tag=f"_remesh{self.n_remesh:03d}_before")
        self.write_profile(self.interface_polyline(), st_new, tag=f"_remesh{self.n_remesh:03d}_after")
        with (self.output_root / "remesh.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            f"REMESH #{self.n_remesh} at t={st_new['t']:.6e}: {rec['reason']}; "
            f"h_tip={self._mesh_receipt.get('h_tip'):.3e}, ndof={rec['ndof']}, "
            f"volume {st_old.get('volume', float('nan')):.12e} -> {st_new['volume']:.12e}",
            flush=True,
        )

    # ------------------------------------------------------------------ drive
    def choose_dt(self, st: dict[str, float]) -> float:
        """Step-size controller.

        Two ceilings: a fraction of the bridge-growth time ``R_min/u`` (quasi-steady
        regime) and a PI-like adjustment that keeps the relative change of the neck
        curvature per step near ``curvature_change_target``.  In the similarity
        regime ``|2H| ~ R_min**-3`` so both agree; during the initial reshaping of
        the meniscus the curvature criterion dominates and shortens the step.
        """
        u = max(abs(st["u_neck"]), 1e-3)
        dt_phys = self.dt_fraction * st["R_min"] / u
        ratio = max(self._last_curvature_ratio, 1e-12)
        if self.tip_unresolved(st["tip_radius"]):
            ratio = 1e-12          # curvature reading is noise below the resolvable tip radius
        factor = math.sqrt(self.curvature_change_target / ratio)
        growth = 2.0 if getattr(self, "_regrow", False) else self.dt_growth
        factor = min(growth, max(0.5, factor))
        dt = min(dt_phys, self._dt * factor)
        if dt >= 0.9 * dt_phys or factor < growth:
            self._regrow = False
        return max(dt, self.dt_initial)

    def run_campaign(self, *, max_steps: int = 100000, max_wall_s: float | None = None) -> dict[str, Any]:
        self.initialise()
        self.refine_tip()
        st = self.neck_state()
        self.write_row(st, 0.0, None)
        self.write_profile(self.interface_polyline(), st, tag="_initial")
        print(
            f"START R_min={st['R_min']:.6e} two_H={st['two_H']:.6e} "
            f"tip_radius={st['tip_radius']:.3e} ndof={self.ndof()}",
            flush=True,
        )
        self._dt = self.dt_initial
        status = "running"
        while self._steps < max_steps:
            if st["R_min"] >= self.R_stop:
                status = "reached_R_stop"
                break
            if max_wall_s is not None and time.time() - self._wall0 > max_wall_s:
                status = "wall_limit"
                break
            if self.tip_refine and self._needs_tip_refine:
                self._needs_tip_refine = False
                if self.refine_tip():
                    st = self.neck_state()
            dt = self.choose_dt(st)
            ok = False
            if self._dt_prev is None:
                # Fresh integrator (after remesh, bisection or a rejected step): no predictor is
                # available, so keep the first displacement within ~20 tip elements.
                dt = min(dt, 20.0 * st["h_tip_now"] / max(abs(st["u_neck"]), 1e-3))
                dt = max(dt, self.dt_initial)
                self._regrow = True
            for _attempt in range(6):
                snapshot = self._dof_snapshot()
                self._dt = dt
                self._predict_now = True
                try:
                    self.solve(timestep=dt, do_not_set_IC=True)
                    # The max-residual test is dominated by far-field rows whose weights exceed
                    # the tip rows by ~20 decades at R0=1e-6, so Newton stops with the tip
                    # equations unconverged and a grid-scale wrinkle grows on the tip (run 58).
                    # Re-enter oomph-lib's Newton on the same time-discrete system (history
                    # already shifted, weights unchanged); always_take_one_newton_step makes each
                    # call do at least one more quadratic iteration.
                    for _extra in range(self.extra_newton_iterations):
                        _pyoomph.Problem.newton_solve(self, 0)
                    ok = True
                    break
                except Exception as exc:  # Newton failure: restore the pre-step state, retry smaller
                    print(f"step failed at dt={dt:.3e}: {exc!r}; restoring and reducing", flush=True)
                    self._predict_now = False
                    if "Cannot invert spline" in repr(exc) and self._template is not None:
                        # The failure is in building the NEW mesh's macro elements; the old mesh is
                        # still the live one.  Retry the remesh with coarser tip-zone sampling.
                        self._thin_boost *= 2.0
                        print(f"remesh spline inversion failed; retrying remesh with thin_boost={self._thin_boost}", flush=True)
                        try:
                            self._dof_restore(snapshot)
                            self.force_remesh({self._template})
                            self._needs_tip_refine = True
                            self.refine_tip()
                            st = self.neck_state()
                            continue
                        except Exception as exc2:
                            print(f"remesh retry failed: {exc2!r}", flush=True)
                    try:
                        self._dof_restore(snapshot)
                    except Exception as exc3:
                        # A quality remesh inside the failed step replaced the mesh; the snapshot
                        # no longer fits.  Continue from the remeshed state with a smaller step.
                        print(f"snapshot restore skipped after mid-step remesh: {exc3!r}", flush=True)
                        self._dt_prev = None
                    dt *= 0.3
                finally:
                    self._predict_now = False
            if not ok:
                status = "newton_failure"
                break
            self._dt_prev = dt
            self._dt = dt
            self._steps += 1
            st_prev = st
            st = self.neck_state()
            if self.tip_refine and math.isfinite(st["tip_radius"]) and st["tip_radius"] > 0 and st["h_tip_now"] <= st["tip_radius"] / 4:
                self.tip_radius_lagged = st["tip_radius"]  # trust the measurement only when resolved
            self._last_curvature_ratio = abs(st["two_H"] - st_prev["two_H"]) / max(abs(st_prev["two_H"]), 1e-30)
            iters = None
            try:
                iters = int(self.get_last_newton_iterations())
            except Exception:
                pass
            self.write_row(st, dt, iters)
            if self._steps % 10 == 0 or self._steps < 10:
                print(
                    f"step {self._steps} t={st['t']:.6e} dt={dt:.3e} R={st['R_min']:.6e} "
                    f"u={st['u_neck']:.5f} 2H={st['two_H']:.4e} rho={st['tip_radius']:.3e} "
                    f"vol={st['volume']:.10e} remesh={self.n_remesh}",
                    flush=True,
                )
        summary = {
            "status": status,
            "steps": self._steps,
            "final": st,
            "n_remesh": self.n_remesh,
            "wall_s": time.time() - self._wall0,
            "parameters": {
                "R0": self.R0, "Z0": self.Z0, "n_tip": self.n_tip, "grading": self.grading,
                "h_max": self.h_max, "remesh_growth": self.remesh_growth,
                "tip_shrink_remesh": self.tip_shrink_remesh, "tip_grow_remesh": self.tip_grow_remesh,
                "dt_fraction": self.dt_fraction, "dt_initial": self.dt_initial,
                "dt_growth": self.dt_growth, "curvature_change_target": self.curvature_change_target,
                "newton_tolerance": self.newton_tolerance,
                "R_stop": self.R_stop,
            },
        }
        (self.output_root / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"END {status} after {self._steps} steps, {self.n_remesh} remeshes", flush=True)
        return summary
