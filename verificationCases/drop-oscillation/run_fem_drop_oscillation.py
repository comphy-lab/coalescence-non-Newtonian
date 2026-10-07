"""Run the free-drop P2 oscillation used to verify the inertial finite-element solver."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coalescence.fem.drop_oscillation import OscillatingDropProblem, rayleigh_lamb  # noqa: E402

FIELDS = ["t", "z_pole", "r_equator", "volume", "kinetic", "area", "dissipation", "wall_s"]


def component_commit() -> str:
    commit_file = ROOT / "COMMIT"
    if commit_file.is_file():
        return commit_file.read_text().strip()
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"
    return commit + ("-dirty" if dirty else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Oh", type=float, required=True)
    ap.add_argument("--eps", type=float, default=0.01, help="initial P2 amplitude relative to the radius")
    ap.add_argument("--resolution", type=float, default=0.05, help="Gmsh element size")
    ap.add_argument("--periods", type=float, default=3.0, help="duration in Rayleigh periods")
    ap.add_argument("--steps-per-period", type=int, default=200)
    ap.add_argument("--spatial-scale", type=float, default=1.0, help="pyoomph spatial scale (physics must not depend on it)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.out / "oscillation.csv").exists():
        raise SystemExit(f"{a.out} already holds a run")
    a.out.mkdir(parents=True, exist_ok=True)
    theory = rayleigh_lamb(a.Oh)
    period = 2.0 * math.pi / theory["omega"]
    dt = period / a.steps_per_period
    n_steps = int(round(a.periods * a.steps_per_period))
    manifest = {"solver": "coalescence.fem.drop_oscillation", "component_commit": component_commit(),
                "Oh": a.Oh, "eps": a.eps, "resolution": a.resolution, "spatial_scale": a.spatial_scale, "dt": dt, "steps": n_steps,
                "theory": theory, "units": "visco-capillary; density 1/Oh^2", "argv": sys.argv}
    (a.out / "run-manifest.json").write_text(json.dumps(manifest, indent=1))
    pb = OscillatingDropProblem(Oh=a.Oh, eps=a.eps, resolution=a.resolution, spatial_scale=a.spatial_scale)
    pb.set_output_directory(str(a.out / "pyoomph"))
    pb.quiet()
    pb.initialise()
    wall0 = time.time()
    with (a.out / "oscillation.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerow({**pb.state(), "wall_s": 0.0})
        for k in range(n_steps):
            pb.solve(timestep=dt, do_not_set_IC=(k > 0))
            w.writerow({**pb.state(), "wall_s": time.time() - wall0})
            fh.flush()
    (a.out / "summary.json").write_text(json.dumps({"status": "completed", "steps": n_steps,
                                                    "wall_s": time.time() - wall0}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
