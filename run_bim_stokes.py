"""Run the independent boundary-integral Stokes solver for one La = 0 case file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from coalescence.bim.geometry import Meridian, anthony_initial_meridian  # noqa: E402
from coalescence.bim.solver import RunConfig, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=float, default=0.1)
    ap.add_argument("--n-tip", type=int, default=16)
    ap.add_argument("--h-max", type=float, default=0.02)
    ap.add_argument("--dt-initial", type=float, default=None, help="default 1e-3 * Z0")
    ap.add_argument("--dt-fraction", type=float, default=0.02)
    ap.add_argument("--curvature-target", type=float, default=0.05)
    ap.add_argument("--r-stop", type=float, default=0.03)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--max-wall-s", type=float, default=1e9)
    ap.add_argument("--restart-from", type=Path, default=None)
    ap.add_argument("--newton", action="store_true", help="backward Euler solved by Newton-Krylov (stiff limit)")
    ap.add_argument("--newton-tol", type=float, default=1e-3)
    ap.add_argument("--newton-stall", type=float, default=0.0,
                    help="accept a Newton iteration that stalls below this multiple of the tolerance (0: never)")
    ap.add_argument("--newton-retries", type=int, default=0, help="halve dt and retry an unconverged Newton step")
    a = ap.parse_args()
    case = json.loads(a.case.read_text())
    if case["physics"].get("inertia", True):
        raise SystemExit("the boundary-integral solver is for the La = 0 Stokes limit only")
    bridge = case["physics"]["initial_bridge"]
    R0, Z0 = float(bridge["R0"]), float(bridge["Z0"])
    cfg = RunConfig(k=a.k, n_tip=a.n_tip, h_max=a.h_max, dt_initial=a.dt_initial or 1e-3 * Z0,
                    dt_fraction=a.dt_fraction, curvature_target=a.curvature_target, R_stop=a.r_stop,
                    max_steps=a.max_steps, max_wall_s=a.max_wall_s, newton=a.newton, newton_tol=a.newton_tol,
                    newton_stall=a.newton_stall, newton_retries=a.newton_retries)
    a.out.mkdir(parents=True, exist_ok=True)
    commit_file = ROOT / "COMMIT"
    if commit_file.is_file():                      # source materialised from an archive
        commit = commit_file.read_text().strip()
    else:
        try:
            commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
                                             stderr=subprocess.DEVNULL).strip()
        except Exception:
            commit = "unknown"
    manifest = {"case_file": str(a.case), "case_sha256": hashlib.sha256(a.case.read_bytes()).hexdigest(),
                "case_id": case["case_id"], "solver": "coalescence.bim", "component_commit": commit,
                "config": asdict(cfg), "argv": sys.argv}
    t0, dt0, step0 = 0.0, None, 0
    if a.restart_from is not None:
        st = np.load(a.restart_from)
        mer = Meridian(R_n=float(st["R_n"]), X=st["X"], z=st["z"])
        t0, dt0, step0 = float(st["t"]), float(st["dt"]), int(st["step"])
        manifest["restart_from"] = {"path": str(a.restart_from),
                                    "sha256": hashlib.sha256(a.restart_from.read_bytes()).hexdigest()}
    else:
        mer = anthony_initial_meridian(R0, Z0, k=a.k, n_tip=a.n_tip, h_max=a.h_max)
    (a.out / "run-manifest.json").write_text(json.dumps(manifest, indent=1))
    summary = run(mer, a.out, cfg, t0=t0, dt0=dt0, steps0=step0)
    return 0 if summary["status"] == "reached_R_stop" else 1


if __name__ == "__main__":
    raise SystemExit(main())
