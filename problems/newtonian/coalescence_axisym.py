"""One-quadrant axisymmetric Newtonian coalescence with Anthony's initial bridge.

Nondimensionalisation follows Anthony, Harris & Basaran, Phys. Rev. Fluids 5,
033608 (2020), Sec. II.A. Length is the drop radius. The Stokes limit uses the
viscocapillary time; a finite-Oh run uses the inertial-capillary time. Overlay
of the two is ``tau_v = tau / Oh`` and ``u_v = u_min * Oh``.

The mesh is the unstructured translation of their Sec. II.B construction:
Eq. 2 circular bridge, spherical drop, Eq. 6 dividing ellipse (lagged
``R_min``), two regions so elements cannot cross the ellipse, and remesh
whenever ``R_min`` grows by four. Neck spacing follows the bridge height
``~ R_min^2`` (their second scale). Hyperelastic ALE stands in for their
elliptic mesh generation.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from pyoomph import *
from pyoomph.equations.ALE import HyperelasticSmoothedMesh
from pyoomph.equations.generic import (
    AxisymmetryBC,
    ExtremumObservables,
    IntegralObservables,
    RemeshWhen,
    RemeshingOptions,
)
from pyoomph.equations.navier_stokes import (
    NavierStokesEquations,
    NavierStokesFreeSurface,
    StokesEquations,
)
from pyoomph.meshes.zeta import (
    AssignZetaCoordinatesByArclength,
    AssignZetaCoordinatesByEulerianCoordinate,
)

from .geometry import (
    clip_ellipse_to_initial_interface,
    clip_ellipse_to_polyline,
    eq6_axis_point,
    eq6_ellipse_points,
    hypot_sq,
    resample_polyline,
    sphere_bridge_junction,
    sphere_centre,
    twice_mean_curvature_at_neck,
)


def load_case(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != "pyoomph-case-v1":
        raise ValueError(f"not a pyoomph-case-v1 file: {path}")
    return data


def neck_mesh_size(rmin: float, abs_2h: float | None = None) -> float:
    """Resolve Anthony's second and third scales: height ``~ R_min^2`` and ``δ ~ 1/|2H|``."""
    h_height = max(rmin * rmin / 16.0, 1e-16)
    if abs_2h is None or abs_2h <= 0.0:
        return h_height
    h_curve = 4.0 / abs_2h
    return max(min(h_height, h_curve), 1e-12)


class CoalescenceMesh(GmshTemplate):
    """Quarter drop split by the Eq. 6 ellipse, fine only in the bridge region."""

    def define_geometry(self):
        problem = self.get_problem()
        assert isinstance(problem, CoalescenceProblem)
        r0 = float(problem.R0)
        z0 = float(problem.Z0)
        rmin = float(problem.rmin_lagged)
        hmin = neck_mesh_size(rmin, problem.abs_2h_lagged)
        hmax = float(problem.hmax)
        h_ell = min(max(rmin / 8.0, 8.0 * hmin), hmax)
        d_grade = max(0.2, 4.0 * rmin)

        self.mesh_mode = "tris"
        self.default_resolution = hmax
        self.set_gmsh_parameter("General.NumThreads", 1)
        self.set_gmsh_parameter("Geometry.Tolerance", 1e-14)
        self.set_gmsh_parameter("Mesh.MeshSizeMin", hmin)
        self.set_gmsh_parameter("Mesh.MeshSizeMax", hmax)
        self.set_gmsh_parameter("Mesh.MeshSizeFromCurvature", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeFromPoints", 0)
        self.set_gmsh_parameter("Mesh.MeshSizeExtendFromBoundary", 0)
        self.set_gmsh_parameter("Mesh.Algorithm", 6)

        _, z_c = sphere_centre(r0, z0)
        _, z1 = eq6_axis_point(rmin)
        z_north = z_c + 1.0
        p_origin = self.point(0.0, 0.0, size=hmin)

        if self.is_first_time():
            p_ell_axis = self.point(0.0, min(z1, 0.5 * z_north), size=h_ell)
            p_north = self.point(0.0, z_north, size=hmax)
            p_neck = self.point(r0, 0.0, size=hmin)
            jr, jz = sphere_bridge_junction(r0, z0)
            p_junc = self.point(jr, jz, size=hmin)
            centre = (r0 + z0, 0.0)
            divider_pts = clip_ellipse_to_initial_interface(rmin, r0, z0)
            p_ell_hit = self.point(divider_pts[-1][0], divider_pts[-1][1], size=h_ell)
            bridge_arc = self.circle_arc(p_neck, p_junc, center=centre, name="interface")
            neck_size_curves = [bridge_arc]
            # Sphere from the Eq. 2 junction to the ellipse hit, then to the pole.
            if divider_pts[-1][1] > jz * 1.05:
                sphere_low = self.circle_arc(
                    p_junc, p_ell_hit, center=(0.0, z_c), name="interface"
                )
                sphere_high = self.circle_arc(
                    p_ell_hit, p_north, center=(0.0, z_c), name="interface"
                )
                iface_bridge = [bridge_arc, sphere_low]
                iface_bulk = [sphere_high]
            else:
                sphere = self.circle_arc(p_junc, p_north, center=(0.0, z_c), name="interface")
                iface_bridge = [bridge_arc]
                iface_bulk = [sphere]
                p_ell_hit = p_junc
            ell_samples = [(r, z) for r, z in divider_pts[1:-1]]
            divider_nodes = [p_ell_axis] + [
                self.point(r, z, size=h_ell) for r, z in ell_samples
            ] + [p_ell_hit]
        else:
            interface = self.get_boundary_coordinates(
                "drop/interface", sort_along_axis="y+", nondimensional=True
            )[0]
            interface = [(float(x), float(y)) for x, y in interface]
            raw_ell = eq6_ellipse_points(rmin)
            divider_pts = clip_ellipse_to_polyline(raw_ell, interface)
            r_hit, z_hit = divider_pts[-1]
            low, high = _split_interface(interface, r_hit, z_hit)
            if z_hit > 20.0 * rmin or r_hit > 6.0 * rmin:
                raise RuntimeError(
                    f"Eq. 6 hit left the bridge region: r={r_hit}, z={z_hit}, Rmin={rmin}"
                )
            low = resample_polyline(low, 16)
            high = resample_polyline(high, 32)
            z_axis = min(max(divider_pts[0][1], 2.0 * hmin), 0.9 * interface[-1][1])
            p_ell_axis = self.point(0.0, z_axis, size=h_ell)
            p_neck = self.point(low[0][0], 0.0, size=hmin)
            p_north = self.point(0.0, high[-1][1], size=hmax)
            p_ell_hit = self.point(r_hit, z_hit, size=h_ell)
            pts_low = [p_neck] + [
                self.point(x, y, size=hmin if i < 4 else h_ell)
                for i, (x, y) in enumerate(low[1:-1])
            ] + [p_ell_hit]
            pts_high = [p_ell_hit] + [
                self.point(x, y, size=h_ell if i < 3 else hmax)
                for i, (x, y) in enumerate(high[1:-1])
            ] + [p_north]
            iface_bridge = _curve_through(self, pts_low, "interface")
            iface_bulk = _curve_through(self, pts_high, "interface")
            neck_size_curves = list(iface_bridge)
            divider_nodes = [p_ell_axis] + [
                self.point(r, z, size=h_ell) for r, z in divider_pts[1:-1]
            ] + [p_ell_hit]

        axis_bridge = self.create_lines(p_origin, "axis", p_ell_axis)
        axis_bulk = self.create_lines(p_ell_axis, "axis", p_north)
        plane = self.create_lines(p_origin, "plane", p_neck)
        divider = self.spline(divider_nodes, name="divider")

        self.plane_surface(*axis_bridge, *plane, *iface_bridge, divider, name="drop")
        self.plane_surface(*axis_bulk, *iface_bulk, divider, name="drop")

        dist = self.add_mesh_size_field(
            "Distance",
            CurvesList=list(neck_size_curves),
            Sampling=400,
        )
        near = self.add_mesh_size_field(
            "Threshold",
            IField=dist,
            LcMin=hmin,
            LcMax=h_ell,
            DistMin=max(4.0 * hmin, 0.25 * rmin * rmin),
            DistMax=max(rmin, 40.0 * hmin),
            StopAtDistMax=1,
        )
        bulk = self.add_mesh_size_field(
            "MathEval",
            F=(
                f"min({hmax},{h_ell}+({hmax}-{h_ell})"
                f"*sqrt((x-{rmin})*(x-{rmin})+y*y)/{d_grade})"
            ),
        )
        size = self.add_mesh_size_field("Min", FieldsList=[near, bulk])
        self.set_mesh_size_background_field(size)


def _curve_through(mesh: CoalescenceMesh, pts: list, name: str) -> list:
    if len(pts) < 2:
        raise RuntimeError(f"need at least two points for {name}")
    if len(pts) == 2:
        line = mesh.line(pts[0], pts[1], name=name)
        if line is None:
            raise RuntimeError(f"gmsh rejected a two-point {name} curve")
        return line if isinstance(line, list) else [line]
    return [mesh.spline(pts, name=name)]


def _split_interface(
    points: list[tuple[float, float]], r_hit: float, z_hit: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    j = min(range(len(points)), key=lambda k: hypot_sq(points[k], (r_hit, z_hit)))
    hit = (r_hit, z_hit)
    low = points[: j + 1]
    high = points[j:]
    if hypot_sq(low[-1], hit) > 0.0:
        low = low + [hit]
    if hypot_sq(high[0], hit) > 0.0:
        high = [hit] + high
    if len(low) < 2:
        low = [points[0], hit]
    if len(high) < 2:
        high = [hit, points[-1]]
    return low, high


class CoalescenceProblem(Problem):
    def __init__(self, case: dict[str, Any], output_dir: str | Path):
        super().__init__()
        physics = case["physics"]
        bridge = physics["initial_bridge"]
        mesh = case["mesh"]
        tolerances = case["tolerances"]
        self.case = case
        self.case_id = str(case["case_id"])
        self.R0 = float(bridge["R0"])
        self.Z0 = float(bridge["Z0"])
        self.inertia = bool(physics.get("inertia", True))
        oh = physics.get("Oh")
        self.Oh = None if oh is None else float(oh)
        self.newton_tol = float(tolerances["newton"])
        self.temporal_error = float(tolerances["time_adaptive_target"])
        self.hmax = float(mesh["characteristic_size"])
        self.remesh_factor = 4.0
        self.stop_rmin = 0.5
        self.rmin_lagged = self.R0
        self.abs_2h_lagged = 1.0 / self.Z0 + 1.0 / self.R0
        self.output_root = Path(output_dir)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._rmin_at_remesh = self.R0
        self._abs2h_at_remesh = self.abs_2h_lagged
        self._in_remesh = False
        self._neck_csv = self.output_root / "neck.csv"
        self._profile_dir = self.output_root / "profiles"
        self._profile_dir.mkdir(parents=True, exist_ok=True)
        self._wrote_header = False
        self._last_profile_log10 = None
        self._last_neck_t = None
        self._dump_fields = False
        self.strict_output_diagnostics = False

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self.set_output_directory(str(self.output_root / "pyoomph"))
        # With |2H| ~ 1/Z0 the discrete residual plateaus near 1e-8.
        self.newton_solver_tolerance = max(self.newton_tol, 5e-8)
        self.max_newton_iterations = 20
        self.DTSF_minimum_dt = 1e-18
        self.write_states = False
        self.add_mesh(CoalescenceMesh())

        if self.inertia:
            if self.Oh is None:
                raise ValueError("finite-Oh runs require physics.Oh")
            flow = NavierStokesEquations(
                mass_density=1.0, dynamic_viscosity=self.Oh, mode="TH"
            )
        else:
            flow = StokesEquations(
                dynamic_viscosity=1.0, mode="TH", mass_density=1.0, boussinesq=True
            )

        eqs = MeshFileOutput()
        eqs += flow
        eqs += HyperelasticSmoothedMesh()
        # Anthony remeshes on R_min ×4, not when the meniscus compresses an element.
        eqs += RemeshWhen(
            RemeshingOptions(
                max_expansion=1.0e9,
                min_expansion=1.0e-12,
                min_quality_decrease=1.0e-12,
                on_inverted_element=True,
            )
        )
        eqs += IntegralObservables(volume=1)
        eqs += AxisymmetryBC() @ "axis"
        eqs += DirichletBC(mesh_x=0) @ "axis"
        eqs += DirichletBC(mesh_y=True, velocity_y=0) @ "plane"
        eqs += NavierStokesFreeSurface(surface_tension=1.0) @ "interface"
        eqs += ExtremumObservables(r=var("mesh_x")) @ "interface"
        eqs += AssignZetaCoordinatesByArclength(sort_along_axis="y+") @ "interface"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axis"
        eqs += AssignZetaCoordinatesByEulerianCoordinate("x") @ "plane"
        eqs += DirichletBC(pressure=0) @ "axis/plane"
        self.add_equations(eqs @ "drop")

    def _interface_points(self) -> list[tuple[float, float]]:
        from pyoomph.meshes.meshdatacache import MeshDataCache
        from pyoomph.meshes.ordering import sort_line_segments

        mesh = self.get_mesh("drop/interface")
        data = MeshDataCache(tesselate_tri=False, nondimensional=True).get_data(mesh)
        segs, _ = data.get_interface_line_segments()
        pts = data.get_coordinates()
        segs = sort_line_segments(pts, segs, sort_along_axis="y+", whom="interface")
        return [(float(pts[0, i]), float(pts[1, i])) for seg in segs for i in seg]

    def neck_state(self) -> dict[str, float]:
        from pyoomph.meshes.meshdatacache import MeshDataCache

        plane = MeshDataCache(tesselate_tri=False, nondimensional=True).get_data(
            self.get_mesh("drop/plane")
        )
        r = plane.get_data("coordinate_x")
        u = plane.get_data("velocity_x")
        if r is None or len(r) == 0:
            raise RuntimeError("plane mesh has no coordinates")
        idx = int(r.argmax())
        rmin = float(r[idx])
        umin = float(u[idx]) if u is not None else float("nan")
        iface = self._interface_points()
        v_bod = self._back_of_drop_velocity()
        data = {
            "t": float(self.get_current_time()),
            "R_min": rmin,
            "u_min": umin,
            "abs_2H": twice_mean_curvature_at_neck(iface),
            "Z_b": self._bridge_half_height(rmin, iface),
            "V_bod": v_bod,
        }
        return data

    def _back_of_drop_velocity(self) -> float:
        from pyoomph.meshes.meshdatacache import MeshDataCache

        axis = MeshDataCache(tesselate_tri=False, nondimensional=True).get_data(
            self.get_mesh("drop/axis")
        )
        z = axis.get_data("coordinate_y")
        v = axis.get_data("velocity_y")
        if z is None or len(z) == 0 or v is None:
            return float("nan")
        return float(v[int(z.argmax())])

    def _bridge_half_height(self, rmin: float, iface: list[tuple[float, float]]) -> float:
        target = 1.05 * rmin
        best = float("nan")
        dist = float("inf")
        for r, z in iface:
            gap = abs(r - target)
            if gap < dist:
                dist = gap
                best = z
        return best

    def write_neck_row(self, *, profile: bool = False) -> dict[str, float]:
        row = self.neck_state()
        t = row["t"]
        if self._last_neck_t is not None and abs(t - self._last_neck_t) < 1e-18:
            return row
        self._last_neck_t = t
        fieldnames = ["t", "R_min", "u_min", "Z_b", "abs_2H", "V_bod"]
        new_file = not self._neck_csv.exists()
        with self._neck_csv.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if new_file or not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow({key: row[key] for key in fieldnames})
        if profile or self._should_write_profile(row["R_min"]):
            self._write_profile(row)
        return row

    def _should_write_profile(self, rmin: float) -> bool:
        if rmin <= 0.0:
            return False
        decade = math.floor(math.log10(rmin) * 20.0)  # every 0.05 decade
        if self._last_profile_log10 != decade:
            self._last_profile_log10 = decade
            return True
        return False

    def _write_profile(self, row: dict[str, float]) -> None:
        coords = self._interface_points()
        path = self._profile_dir / f"interface_t{row['t']:.8e}_R{row['R_min']:.8e}.dat"
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"# t={row['t']} R_min={row['R_min']}\n")
            handle.write("r z\n")
            for r, z in coords:
                handle.write(f"{r:.16e} {z:.16e}\n")

    def maybe_remesh(self, rmin: float, abs_2h: float | None = None) -> bool:
        if self._in_remesh:
            return False
        grow_r = rmin >= self.remesh_factor * self._rmin_at_remesh
        if not grow_r:
            return False
        self._in_remesh = True
        try:
            print(
                f"Anthony remesh at t={self.get_current_time()} "
                f"R_min={rmin} |2H|={abs_2h} "
                f"(R factor {rmin / self._rmin_at_remesh:.2f}, "
                f"H factor {abs_2h / self._abs2h_at_remesh if abs_2h and self._abs2h_at_remesh else float('nan'):.2f})",
                flush=True,
            )
            self.rmin_lagged = rmin
            if abs_2h is not None:
                self.abs_2h_lagged = abs_2h
            self.force_remesh()
            self.solve()
            self._rmin_at_remesh = rmin
            if abs_2h is not None:
                self._abs2h_at_remesh = abs_2h
            self._dump_fields = True
            self.output("timestep")
            self.write_neck_row(profile=True)
            return True
        finally:
            self._in_remesh = False

    def output(self, stage="", quiet=None, **kwargs):
        dump = self._dump_fields or stage == "end"
        try:
            time = float(self.get_current_time())
        except Exception:
            time = 0.0
        dump = dump or time == 0.0
        self._dump_fields = False
        if dump:
            super().output(stage, quiet, **kwargs)
        if stage in ("", "timestep", "end"):
            try:
                self.write_neck_row()
            except Exception:
                if self.strict_output_diagnostics:
                    raise

    def run_until(self, *, max_time: float, startstep: float, maxstep: float) -> None:
        self.initialise()
        row = self.neck_state()
        self._rmin_at_remesh = row["R_min"]
        self._abs2h_at_remesh = row["abs_2H"]
        self.rmin_lagged = row["R_min"]
        self.abs_2h_lagged = row["abs_2H"]
        self.write_neck_row(profile=True)
        time = row["t"]
        first = True
        while row["R_min"] < self.stop_rmin and time < max_time:
            hmin = neck_mesh_size(row["R_min"], row["abs_2H"])
            u = max(abs(row["u_min"]), 1.0)
            maxstep_now = min(maxstep, max(startstep, 0.4 * hmin / u))
            chunk = min(
                max(8.0 * startstep, 0.1 * max(time, startstep)),
                12.0 * maxstep_now,
            )
            target = min(time + chunk, max_time)
            kwargs: dict[str, Any] = dict(
                maxstep=maxstep_now,
                temporal_error=self.temporal_error,
                outstep=True,
                do_not_set_IC=True,
            )
            if first:
                kwargs["startstep"] = startstep
                first = False
            self.run(target, **kwargs)
            time = float(self.get_current_time())
            row = self.neck_state()
            self.rmin_lagged = row["R_min"]
            self.abs_2h_lagged = row["abs_2H"]
            self.maybe_remesh(row["R_min"], row["abs_2H"])
            startstep = min(maxstep_now, max(startstep, 0.05 * time if time > 0 else startstep))


def viscocapillary_time_and_velocity(
    row: dict[str, float], *, inertia: bool, oh: float | None
) -> tuple[float, float]:
    """Map a recorded row onto Fig. 3 coordinates (tau_v, u_v). Contact time is applied later."""
    t = row["t"]
    u = row["u_min"]
    if inertia:
        if oh is None:
            raise ValueError("Oh is required to rescale an inertial run")
        return t / oh, u * oh
    return t, u
