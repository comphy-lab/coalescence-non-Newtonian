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
from pyoomph.expressions import cartesian, dot, exp, vector
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
    return (float(cx + x0[0]), float(cy + x0[1])), float(math.sqrt(r2))


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

    def define_geometry(self):
        pb = self.get_problem()
        assert isinstance(pb, StokesTipCoalescence)
        self.mesh_mode = "quads" if pb.tip_refine else "tris"
        self.order = 2
        h_tip = pb.current_h_tip()
        h_max = pb.h_max
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
            pb._mesh_receipt = {"kind": "initial", "h_tip": h_tip, "r_neck": r0}
        else:
            pts = pb.interface_polyline_for_remesh()
            r_neck = pts[0][0]
            z_pole = pts[-1][1]
            p_o = self.point(0.0, 0.0)
            p_n = self.point(r_neck, 0.0)
            p_t = self.point(0.0, z_pole)
            inner = [self.point(x, y) for (x, y) in pts[1:-1]]
            interface = [self.spline([p_n, *inner, p_t], name="interface")]
            pb._mesh_receipt = {
                "kind": "remesh",
                "h_tip": h_tip,
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
        self._needs_tip_refine = False
        self.n_refine_events = 0
        self._R_ref = None

        self.output_root = Path(output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "profiles").mkdir(exist_ok=True)

        # Lagged mesh parameters (set at each remesh).
        self.tip_radius_lagged = self.Z0
        self.rmin_at_remesh = self.R0
        self.last_rmin = self.R0
        self.n_remesh = 0
        self._step_at_remesh = 0
        self._mesh_receipt: dict[str, Any] = {}
        self._pending_remesh_reason: str | None = None
        self._template: TipGradedQuadrantMesh | None = None
        self._neck_csv = self.output_root / "neck.csv"
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

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.set_output_directory(str(self.output_root / "pyoomph"))
        self.newton_solver_tolerance = self.newton_tolerance
        self.max_newton_iterations = 12
        if self.tip_refine:
            self.max_refinement_level = 20
        self.write_states = False
        self._template = TipGradedQuadrantMesh()
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
            eqs += EnforcedBC(mesh_x=var("mesh_x") - var("lagrangian_x") * Rn / self._R_ref) @ "plane"
            eqs += DirichletBC(_lagr_enf_bc_mesh_x=0) @ "plane/interface"
            eqs += WeakContribution(var("mesh_x") - Rn, testfunction("R_neck", domain="globals")) @ "interface/plane"
            # Tangential motion of the interface: a radial translation by (R_neck - R_ref)
            # decaying with Lagrangian distance from the neck over L = tip_translation_span*R_ref,
            # imposed through a tangential Lagrange multiplier.  The normal motion stays with
            # the kinematic condition.  Pinned at both corners where the tangent is already fixed.
            X = var("lagrangian")
            s2 = (X[0] - self._R_ref) ** 2 + X[1] ** 2
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

        data = MeshDataCache(tesselate_tri=False, nondimensional=True).get_data(
            self.get_mesh("drop/interface")
        )
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

    def interface_polyline_for_remesh(self) -> list[tuple[float, float]]:
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
        head = smooth_polyline_arclength(pts[: n_tip_pts + 1], self.smooth_half_window)
        pts = head + pts[n_tip_pts + 1 :]
        # Exact symmetry-plane and axis end points.
        pts[0] = (pts[0][0], 0.0)
        pts[-1] = (0.0, pts[-1][1])
        if pts[0][1] != 0.0 or pts[-1][0] != 0.0 or pts[0][0] <= 0.0:
            raise RuntimeError(f"unexpected interface ends: {pts[0]}, {pts[-1]}")
        return pts

    # ---------------------------------------------------------- diagnostics
    def neck_state(self) -> dict[str, float]:
        _, r, z, u, v = self._interface_nodes()
        n = len(r)
        # Neck node: on the plane z=0 with maximal r.
        on_plane = [i for i in range(n) if abs(z[i]) <= 1e-14]
        i0 = max(on_plane, key=lambda i: r[i]) if on_plane else int(z.argmin())
        p0 = (float(r[i0]), float(z[i0]))
        rmin = p0[0]
        # Least-squares circle over the interface nodes within ~1 lagged tip radius of the
        # neck (at least the five nearest).  Far more robust than a three-node circle when the
        # element size at the tip varies between remeshes.
        d2 = (r - p0[0]) ** 2 + (z - p0[1]) ** 2
        order = sorted(range(n), key=lambda i: d2[i])
        # Seven nearest Q2 nodes: three elements, about 0.2 tip radii of arc at n_tip=16.
        # Local enough to be the tip, averaged enough not to flip with the element size.
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
        target = 1.05 * rmin
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
                fh.write(f"{r:.16e} {z:.16e}\n")

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
            if not (math.isfinite(rho) and rho > 0):
                break
            target = rho / self.n_tip
            r0, z0 = st["R_min"], 0.0
            idx = []
            for i, e in enumerate(mesh.elements()):
                h = math.sqrt(max(e.get_current_cartesian_nondim_size(), 0.0))
                if h <= target:
                    continue
                x, y = e.get_Eulerian_midpoint()[:2]
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
        if math.isfinite(rho) and rho < self.tip_shrink_remesh * self.tip_radius_lagged:
            return f"tip radius shrank to {rho:.3e} from {self.tip_radius_lagged:.3e}"
        if math.isfinite(rho) and rho > self.tip_grow_remesh * self.tip_radius_lagged:
            return f"tip radius grew to {rho:.3e} from {self.tip_radius_lagged:.3e}"
        return None

    def actions_before_remeshing(self, active_remeshers):
        st = self.neck_state()
        rho = st["tip_radius"]
        if math.isfinite(rho) and rho > 0:
            self.tip_radius_lagged = rho
        self.rmin_at_remesh = st["R_min"]
        self.last_rmin = st["R_min"]
        if self._R_ref is not None:
            self._R_ref.value = st["R_min"]
        self._pre_remesh_state = st
        self._pre_remesh_poly = self.interface_polyline()
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
        factor = math.sqrt(self.curvature_change_target / ratio)
        factor = min(self.dt_growth, max(0.5, factor))
        dt = min(dt_phys, self._dt * factor)
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
            if self.tip_refine and (self._needs_tip_refine or st["h_tip_now"] > 1.3 * st["tip_radius"] / self.n_tip):
                self._needs_tip_refine = False
                if self.refine_tip():
                    st = self.neck_state()
            dt = self.choose_dt(st)
            ok = False
            for _attempt in range(6):
                snapshot = self._snapshot_state()
                self._dt = dt
                self._predict_now = True
                try:
                    self.solve(timestep=dt, do_not_set_IC=True)
                    ok = True
                    break
                except Exception as exc:  # Newton failure: restore the pre-step state, retry smaller
                    print(f"step failed at dt={dt:.3e}: {exc!r}; restoring and reducing", flush=True)
                    self._predict_now = False
                    self._restore_state(snapshot)
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
