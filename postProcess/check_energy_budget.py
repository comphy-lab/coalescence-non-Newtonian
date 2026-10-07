#!/usr/bin/env python3
"""Energy budget of a coalescence run: d/dt(kinetic + area) = -dissipation.

For a run written with ``--energy-budget``, the left side is differenced between
consecutive rows of neck.csv and compared with the trapezoidal mean of the dissipation
rate. Steps that span a remesh are excluded (a rebuilt interface changes the area by
the reconstruction error, not by flow). In the Stokes limit the kinetic energy is zero
and the check is dA/dt = -dissipation. Unit surface tension and viscosity
(visco-capillary units). Prints and writes a JSON summary by decade of R_min.

    python postProcess/check_energy_budget.py <runtime folder> [...] --out <file.json>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def budget(runtime: Path) -> dict:
    with (runtime / "neck.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{runtime}: neck.csv has no rows")
    if "dissipation" not in rows[0]:
        raise SystemExit(f"{runtime}: neck.csv has no energy columns (run with --energy-budget)")
    col = {k: np.array([float(r[k]) for r in rows]) for k in ("t", "R_min", "kinetic", "area", "dissipation", "n_remesh")}
    E = col["kinetic"] + col["area"]
    dt = np.diff(col["t"])
    lhs = np.diff(E) / dt
    rhs = -0.5 * (col["dissipation"][1:] + col["dissipation"][:-1])
    same_mesh = np.diff(col["n_remesh"]) == 0
    first = np.zeros(lhs.size, bool)
    first[:1] = True                                  # the start from rest is first order
    use = same_mesh & ~first & (np.abs(rhs) > 0)
    if not use.any():
        raise SystemExit(f"{runtime}: no usable energy interval (need a second step on an unchanged mesh)")
    rel = (lhs[use] - rhs[use]) / np.abs(rhs[use])
    R = np.sqrt(col["R_min"][1:] * col["R_min"][:-1])[use]
    bands = {}
    for lo in 10.0 ** np.arange(np.floor(np.log10(R.min())), np.ceil(np.log10(R.max()))):
        k = (R >= lo) & (R < 10 * lo)
        if k.any():
            bands[f"{lo:g}..{10 * lo:g}"] = {"n": int(k.sum()), "median": float(np.median(rel[k])),
                                             "max_abs": float(np.max(np.abs(rel[k])))}
    return {"runtime": str(runtime), "steps_used": int(use.sum()), "steps_total": int(lhs.size),
            "relative_residual_median": float(np.median(rel)), "relative_residual_p95_abs": float(np.percentile(np.abs(rel), 95)),
            "by_R_min": bands}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runtimes", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = [budget(p) for p in a.runtimes]
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
