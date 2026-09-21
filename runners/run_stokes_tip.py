"""Run the La=0 tip-graded Stokes coalescence solver for one case file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problems.newtonian.coalescence_stokes_tip import StokesTipCoalescence  # noqa: E402


def main() -> int:
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
    ap.add_argument("--tip-rel-floor", type=float, default=0.0, help="smallest tip element relative to R_neck on the mapped mesh (double-precision solve floor); 0 = off")
    args = ap.parse_args()

    case = json.loads(args.case.read_text())
    if case.get("schema") != "pyoomph-case-v1":
        raise SystemExit("not a pyoomph-case-v1 case")
    bridge = case["physics"]["initial_bridge"]
    if case["physics"].get("inertia", True):
        raise SystemExit("this runner is for the La=0 Stokes limit only")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"
    import pyoomph

    manifest = {
        "case_file": str(args.case),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "case_id": case["case_id"],
        "component_commit": commit,
        "pyoomph_module": pyoomph.__file__,
        "argv": sys.argv,
    }
    (args.out / "run-manifest.json").write_text(json.dumps(manifest, indent=1))
    with (args.out / "progress.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"event": "start", "case_id": case["case_id"], "component_commit": commit}) + "\n")

    spatial_scale = float(bridge["R0"]) if (args.spatial_scale is not None and args.spatial_scale <= 0) else args.spatial_scale
    S = spatial_scale if spatial_scale else 1.0
    # Max-residual roundoff floor: proportional to the initial capillary pressure 1/Z0 and to
    # 1/S^2 under a pyoomph spatial scale S (measured: 2e-8 at Z0=5e-7, S=1; 0.016 at S=1e-3).
    newton_tol = args.newton_tol if args.newton_tol is not None else 1e-7 * (5e-7 / float(bridge["Z0"])) / S**2
    pb = StokesTipCoalescence(
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
    )
    pb.quiet()
    summary = pb.run_campaign(max_steps=args.max_steps, max_wall_s=args.max_wall_s)
    return 0 if summary["status"] in ("reached_R_stop", "wall_limit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
