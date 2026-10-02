#!/usr/bin/env python3
"""Rebuild the Stokes velocity and pressure fields of a moving-frame run at its saved remesh states.

A Stokes run carries no velocity history: the velocity is fixed by the interface
geometry at that instant. A moving-frame run saves the interface before every remesh
(``restart/remesh_NNNN.npz``), so the full field at those times is recovered by
building the mesh that followed the saved state, exactly as a restart does, and solving
the frozen-geometry Stokes problem on it (the seed the run itself started from). Frame 0
is the initial bridge of the case. Each frame is written as ``fields/frame-NNNN.npz``:

* ``points``: (N, 2) nodal (X, z) with X = r - shift, the solver's neck-frame coordinate
  (r would lose the tip to round-off; add ``shift`` only after scaling);
* ``cells``: (M, 6) P2 triangles in VTK order (corners, then midsides 01, 12, 20);
* ``velocity`` (N, 2) as (u_r, u_z) in the laboratory frame, ``pressure`` (N,);
* ``shift`` (the neck radius), ``t``, ``R_min``, ``u_neck`` re-solved here and
  ``u_neck_recorded`` (the run's value before the remesh; ``nan`` for frame 0).

Visco-capillary units throughout (length R, velocity gamma/mu, time mu R/gamma). Each
frame runs in its own process, ``--workers`` at a time.

    python postProcess/reconstruct_stokes_fields.py <runtime> --out <folder> [--frames 0:141] [--workers 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def frame_list(spec: str, n_restart: int) -> list[int]:
    if ":" in spec:
        lo, hi = (int(v) for v in spec.split(":"))
        frames = list(range(lo, min(hi, n_restart + 1)))
    else:
        frames = [int(v) for v in spec.split(",") if v.strip()]
    if any(k < 0 or k > n_restart for k in frames):
        raise SystemExit(f"frames must lie in 0..{n_restart}")
    return frames


def solve_one(runtime: Path, out: Path, k: int) -> dict:
    """Frozen-geometry Stokes solve on the mesh that followed remesh state k (k = 0: initial bridge)."""
    import run_fem_stokes_tip as runner
    from stokes_block_audit import seed_frozen_stokes
    from pyoomph.meshes.meshdatacache import MeshDataCache

    manifest = json.loads((runtime / "run-manifest.json").read_text())
    args = runner.build_parser().parse_args(manifest["argv"][1:])
    case_path = Path(args.case)
    if hashlib.sha256(case_path.read_bytes()).hexdigest() != manifest["case_sha256"]:
        raise SystemExit(f"{case_path} does not match the case hash in the run manifest")
    case = json.loads(case_path.read_text())
    if case["physics"].get("inertia", True):
        raise SystemExit("an inertial run carries velocity history; its field is not fixed by the geometry")
    if not (args.neck_frame_moving and args.seed_frozen_stokes):
        raise SystemExit("reconstruction follows the moving-frame restart route")
    work = out / "work" / f"frame-{k:04d}"
    work.mkdir(parents=True, exist_ok=True)
    args.out = work
    restart, recorded = None, float("nan")
    if k > 0:
        with np.load(runtime / "restart" / f"remesh_{k:04d}.npz") as data:
            restart = {n: (data[n].tolist() if data[n].ndim else data[n].item()) for n in data.files}
        recorded = float(restart["u_neck"])
    wall0 = time.time()
    pb = runner.build_problem(args, case["physics"]["initial_bridge"], None, restart)
    pb.quiet()
    pb.initialise()
    seed = seed_frozen_stokes(pb, work, args.dt_initial)
    st = pb.neck_state()
    entry = MeshDataCache(tesselate_tri=False, nondimensional=False).get_data(pb.get_mesh("drop"))
    types = np.unique(np.asarray(entry.elem_types))
    if list(types) != [9]:
        raise SystemExit(f"expected only six-node triangles (type 9), got {types}")
    cells = np.asarray(entry.elem_indices, dtype=np.int64)[:, :6]
    points = np.column_stack([entry.get_data("coordinate_x"), entry.get_data("coordinate_y")])
    velocity = np.column_stack([entry.get_data("velocity_x"), entry.get_data("velocity_y")])
    pressure = np.asarray(entry.get_data("pressure"), dtype=float)
    shift = pb._frame_shift_now()
    (out / "fields").mkdir(parents=True, exist_ok=True)
    dest = out / "fields" / f"frame-{k:04d}.npz"
    np.savez_compressed(
        dest, points=points, cells=cells, velocity=velocity, pressure=pressure, shift=shift,
        t=float(st["t"]), R_min=float(st["R_min"]), u_neck=float(st["u_neck"]),
        u_neck_recorded=recorded, tip_radius=float(st["tip_radius"]), frame=k,
    )
    rec = {"frame": k, "t": float(st["t"]), "R_min": float(st["R_min"]), "u_neck": float(st["u_neck"]),
           "u_neck_recorded": recorded, "u_neck_relative_change": float(st["u_neck"] / recorded - 1) if k else None,
           "nodes": int(points.shape[0]), "cells": int(cells.shape[0]),
           "seed_residual_after_max": seed["residual_after_max"], "wall_s": time.time() - wall0}
    (work / "frame.json").write_text(json.dumps(rec, indent=1, allow_nan=True) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runtime", type=Path, help="runtime folder holding run-manifest.json and restart/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", default=None, help="lo:hi or a comma list; default all saved states and frame 0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--one", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    runtime = a.runtime.resolve()
    out = a.out.resolve()
    if a.one is not None:
        print(json.dumps(solve_one(runtime, out, a.one)), flush=True)
        return 0
    n_restart = len(list((runtime / "restart").glob("remesh_*.npz")))
    frames = frame_list(a.frames or f"0:{n_restart + 1}", n_restart)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = {"source_runtime": str(runtime), "component_commit": commit, "frames": frames,
                "python": sys.executable, "argv": sys.argv}
    (out / "reconstruct-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    def run(k: int) -> tuple[int, int]:
        if (out / "fields" / f"frame-{k:04d}.npz").exists():
            return k, 0
        with (out / "logs" / f"frame-{k:04d}.log").open("w") as log:
            proc = subprocess.run([sys.executable, __file__, str(runtime), "--out", str(out), "--one", str(k)],
                                  stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
        print(f"frame {k}: exit {proc.returncode}", flush=True)
        return k, proc.returncode

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        codes = dict(pool.map(run, frames))
    failed = sorted(k for k, c in codes.items() if c)
    print(f"{len(frames) - len(failed)} of {len(frames)} frames written; failed: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
