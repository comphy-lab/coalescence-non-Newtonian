"""Run the La=0 tip-graded Stokes coalescence solver for one case file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from coalescence.fem.stokes_tip import StokesTipCoalescence  # noqa: E402


# The case supplies the physics; numerical parameters and the stop radius come from the options.
CASE_FIELDS_USED = ["schema", "physics.inertia", "physics.initial_bridge.R0", "physics.initial_bridge.Z0"]
INERTIAL_FIELDS_USED = ["physics.Oh", "physics.initial_velocity"]
UNITS = "visco-capillary: length R, velocity gamma/mu, time mu R/gamma, pressure gamma/R; density 1/Oh^2"


def component_commit() -> str:
    """HEAD of the component, marked ``-dirty`` when tracked files differ from it."""
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
    return commit + ("-dirty" if dirty else "")

def exit_status(summary: dict) -> int:
    """0 only for a run that reached its stop radius; any other ending is a non-zero exit."""
    return 0 if summary["status"] == "reached_R_stop" else 1


def write_manifest(out: Path, manifest: dict, stem: str) -> Path:
    """Create ``<stem>.json`` exclusively, adding ``-2``, ``-3``... so no earlier record is replaced."""
    for k in range(1, 1000):
        path = out / (f"{stem}.json" if k == 1 else f"{stem}-{k}.json")
        try:
            with path.open("x", encoding="utf-8") as fh:
                fh.write(json.dumps(manifest, indent=1))
            return path
        except FileExistsError:
            continue
    raise SystemExit(f"no free manifest name for {stem} in {out}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-tip", type=int, default=16)
    ap.add_argument("--grading", type=float, default=0.2)
    ap.add_argument("--h-max", type=float, default=0.1)
    ap.add_argument("--dt-fraction", type=float, default=0.02)
    ap.add_argument("--dt-initial", type=float, default=1e-9)
    ap.add_argument("--remesh-growth", type=float, default=1.5)
    ap.add_argument("--newton-tol", type=float, default=None, help="default 1e-7*(5e-7/Z0): the max-residual roundoff floor scales with the initial capillary pressure 1/Z0")
    ap.add_argument("--r-stop", type=float, default=0.03)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--max-wall-s", type=float, default=None)
    ap.add_argument("--no-neck-stretch", action="store_true")
    ap.add_argument("--no-interface-translation", action="store_true")
    ap.add_argument("--h-tip-floor", type=float, default=1e-11)
    ap.add_argument("--tip-refine", action="store_true", help="quads + oomph-lib bisection below the Gmsh floor")
    ap.add_argument("--gmsh-floor", type=float, default=1e-11)
    ap.add_argument("--bisect-floor", type=float, default=2e-13)
    ap.add_argument("--spatial-scale", type=float, default=None, help="pyoomph spatial scale; 0 or negative means R0")
    ap.add_argument("--tip-map", type=float, default=0.0, help="exponent a of the tip-magnifying mesh map (MappedTipMesh); 0 = off, 0.6 recommended")
    ap.add_argument("--tip-map-outer", type=float, default=0.0, help="outer exponent of the composite map beyond --tip-map-core tip radii (0 = single exponent)")
    ap.add_argument("--tip-map-core", type=float, default=1e3, help="core radius of the composite map in tip radii")
    ap.add_argument("--tip-map-linear-core", type=float, default=0.0, help="linear apex core radius in lagged tip radii; 0 preserves the historical map")
    ap.add_argument("--max-residuals", type=float, default=1e10, help="oomph-lib Newton max-residual cap")
    ap.add_argument("--line-search", action="store_true", help="use pyoomph's globally convergent Newton line search")
    ap.add_argument("--audit-blocks", action="store_true", help="stop after one accepted step and audit frozen Stokes and complement Jacobian blocks")
    ap.add_argument("--audit-dt", type=float, default=None, help="fixed BDF1 timestep for --audit-blocks")
    ap.add_argument("--seed-frozen-stokes", action="store_true", help="seed algebraic Stokes velocity/pressure on the untouched initial geometry")
    ap.add_argument("--energy-budget", action="store_true", help="record kinetic energy, free-surface area and viscous dissipation in neck.csv")
    ap.add_argument("--extra-newton", type=int, default=0, help="retired: post-step Newton changes BDF history; nonzero values are rejected")
    ap.add_argument("--min-newton", type=int, default=0, help="minimum Newton iterations within each original time-discrete solve")
    ap.add_argument("--curvature-step-limit", type=float, default=0.0, help="reject a step whose relative tip curvature change exceeds this value; 0 disables")
    ap.add_argument("--neck-frame", action="store_true", help="radial mesh coordinate measured from the neck (NeckFrameAxisymmetric); requires --tip-map")
    ap.add_argument("--neck-frame-moving", action="store_true", help="solve the neck origin within each timestep; requires --neck-frame")
    ap.add_argument("--tip-rel-floor", type=float, default=0.0, help="smallest tip element relative to R_neck on the mapped mesh (double-precision solve floor); 0 = off")
    ap.add_argument("--interface-grading", type=float, default=0.0, help="tip-distance grading on the interface (mapped mesh); 0 = --grading")
    ap.add_argument("--interface-size-growth", type=float, default=0.3, help="growth of the element size with mapped distance from the interface")
    ap.add_argument("--curvature-change-target", type=float, default=0.05, help="target relative change of the neck curvature per step")
    ap.add_argument("--tip-shrink-remesh", type=float, default=0.7, help="remesh when the tip radius falls below this fraction of its value at the last remesh")
    ap.add_argument("--restart-from", type=Path, default=None, help="restart/remesh_NNNN.npz written by an earlier moving-frame run")
    ap.add_argument("--zone-grading", type=float, default=0.0, help="finer grading between --zone-inner and --zone-outer neck radii from the tip (0 = off)")
    ap.add_argument("--zone-inner", type=float, default=1e-3)
    ap.add_argument("--zone-outer", type=float, default=2.0)
    ap.add_argument("--tip-apex-zone", type=float, default=0.05, help="apex zone in lagged tip radii: excluded from the tip fit and replaced by a circle at remeshing (0 = off)")
    return ap


def build_problem(args: argparse.Namespace, bridge: dict, Oh: float | None,
                  restart: dict | None) -> StokesTipCoalescence:
    """The solver for parsed options, the case's initial bridge and Oh (None in the Stokes limit)."""
    spatial_scale = float(bridge["R0"]) if (args.spatial_scale is not None and args.spatial_scale <= 0) else args.spatial_scale
    S = spatial_scale if spatial_scale else 1.0
    # Max-residual roundoff floor: proportional to the initial capillary pressure 1/Z0 and to
    # 1/S^2 under a pyoomph spatial scale S (measured: 2e-8 at Z0=5e-7, S=1; 0.016 at S=1e-3).
    newton_tol = args.newton_tol if args.newton_tol is not None else 1e-7 * (5e-7 / float(bridge["Z0"])) / S**2
    return StokesTipCoalescence(
        R0=float(bridge["R0"]),
        Z0=float(bridge["Z0"]),
        output_dir=args.out,
        n_tip=args.n_tip,
        grading=args.grading,
        h_max=args.h_max,
        dt_fraction=args.dt_fraction,
        dt_initial=args.dt_initial,
        remesh_growth=args.remesh_growth,
        newton_tolerance=newton_tol,
        R_stop=args.r_stop,
        neck_stretch=not args.no_neck_stretch,
        interface_translation=not args.no_interface_translation,
        h_tip_floor=args.h_tip_floor,
        tip_refine=args.tip_refine,
        gmsh_floor=args.gmsh_floor,
        bisect_floor=args.bisect_floor,
        spatial_scale=spatial_scale,
        tip_map_alpha=args.tip_map,
        tip_rel_floor=args.tip_rel_floor,
        neck_frame=args.neck_frame,
        neck_frame_moving=args.neck_frame_moving,
        tip_map_outer=args.tip_map_outer,
        tip_map_core=args.tip_map_core,
        tip_map_linear_core=args.tip_map_linear_core,
        max_residuals=args.max_residuals,
        line_search=args.line_search,
        extra_newton_iterations=args.extra_newton,
        min_newton_iterations=args.min_newton,
        curvature_step_limit=args.curvature_step_limit,
        interface_grading=args.interface_grading,
        curvature_change_target=args.curvature_change_target,
        tip_shrink_remesh=args.tip_shrink_remesh,
        interface_size_growth=args.interface_size_growth,
        restart=restart,
        zone_grading=args.zone_grading,
        tip_apex_zone=args.tip_apex_zone,
        zone_inner=args.zone_inner,
        zone_outer=args.zone_outer,
        Oh=Oh,
        energy_budget=args.energy_budget,
    )


def main() -> int:
    args = build_parser().parse_args()

    case = json.loads(args.case.read_text())
    if case.get("schema") != "pyoomph-case-v1":
        raise SystemExit("not a pyoomph-case-v1 case")
    bridge = case["physics"]["initial_bridge"]
    Oh = None
    if case["physics"].get("inertia", True):
        Oh = float(case["physics"]["Oh"])
        if case["physics"].get("initial_velocity", "quiescent") != "quiescent":
            raise SystemExit("an inertial case must start from rest")
        if args.seed_frozen_stokes:
            raise SystemExit("the frozen-Stokes seed sets the initial velocity; an inertial case starts from rest")
        if args.restart_from is not None:
            raise SystemExit("restart files hold the interface only, the complete state only in the Stokes limit")
    if args.restart_from is not None and (args.out / "neck.csv").exists():
        raise SystemExit("a restart segment needs its own output directory; join it to its parent with `continues` in the run list")
    if args.restart_from is not None and not args.seed_frozen_stokes:
        raise SystemExit("--restart-from requires --seed-frozen-stokes: the velocity is re-solved on the restart mesh")
    if args.seed_frozen_stokes and not args.neck_frame_moving:
        raise SystemExit("frozen-Stokes seed is limited to the moving-frame route")
    if args.audit_blocks and (args.audit_dt is None or args.audit_dt <= 0 or float(bridge["R0"]) != 1e-6):
        raise SystemExit("block audit requires the exact R0=1e-6 case and positive --audit-dt")
    args.out.mkdir(parents=True, exist_ok=True)
    commit = component_commit()
    import pyoomph

    manifest = {
        "case_file": str(args.case),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "case_id": case["case_id"],
        "component_commit": commit,
        "case_fields_used": CASE_FIELDS_USED + (INERTIAL_FIELDS_USED if Oh is not None else []),
        "Oh": Oh,
        "units": UNITS,
        "pyoomph_module": pyoomph.__file__,
        "argv": sys.argv,
    }
    restart = None
    if args.restart_from is not None:
        import numpy as np
        with np.load(args.restart_from) as data:
            restart = {k: (data[k].tolist() if data[k].ndim else data[k].item()) for k in data.files}
        manifest["restart_from"] = {"path": str(args.restart_from),
                                    "sha256": hashlib.sha256(args.restart_from.read_bytes()).hexdigest(),
                                    "t": restart["t"], "R_min": restart["R_min"]}
    restarting_here = restart is not None and (args.out / "run-manifest.json").exists()
    write_manifest(args.out, manifest,
                   f"run-manifest-step{int(restart['steps']):06d}" if restarting_here else "run-manifest")
    with (args.out / "progress.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "start", "case_id": case["case_id"], "component_commit": commit}) + "\n")

    pb = build_problem(args, bridge, Oh, restart)
    pb.quiet()
    if args.seed_frozen_stokes:
        from stokes_block_audit import seed_frozen_stokes
        pb.initialise()
        seed_frozen_stokes(pb, args.out, args.dt_initial)
    if args.audit_blocks:
        summary = pb.run_campaign(max_steps=1, max_wall_s=args.max_wall_s)
        if summary["steps"] != 1:
            return 1
        from stokes_block_audit import audit_one_step
        audit_one_step(pb, args.out, args.audit_dt)
        return 0
    summary = pb.run_campaign(max_steps=args.max_steps, max_wall_s=args.max_wall_s)
    return exit_status(summary)


if __name__ == "__main__":
    raise SystemExit(main())
