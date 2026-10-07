#!/usr/bin/env python3
"""Compare the neck velocity of the runs in a run list with its reference run, at equal R_min.

For each run of the list other than the one with ``role = "reference"``, the relative
difference u_v/u_v,ref − 1 is evaluated at equal R_min (logarithmic interpolation of both
histories) over the shared range of R_min. The figure shows (a) u_v(R_min) for every run
and (b) |u_v/u_v,ref − 1| against R_min. The JSON gives the difference at a set of radii
(in units of the reference R0), its range, and, for runs written with ``--energy-budget``,
the energy-budget residual of ``check_energy_budget.py``. Neck histories are located
through the data roots and checked against the run list's SHA-256.

    python postProcess/compare_neck_histories.py <runs.toml> --out <output prefix>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
from check_energy_budget import budget  # noqa: E402
from coalescence.analysis.runs import load_runs, neck_file  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}", "pdf.fonttype": 42,
})

AT = (1.001, 1.01, 1.1, 2.0, 3.0, 10.0, 30.0)


def read(path: Path) -> dict[str, np.ndarray]:
    """Steps only: the t = 0 row of a run started without a seed records no velocity."""
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))[1:]
    return {k: np.array([float(r[k]) for r in rows]) for k in ("t", "R_min", "u_neck")}


def ratio(run: dict, ref: dict, R: np.ndarray) -> np.ndarray:
    lr = np.log(R)
    return np.interp(lr, np.log(run["R_min"]), run["u_neck"]) / np.interp(lr, np.log(ref["R_min"]), ref["u_neck"]) - 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = load_runs(a.runs)
    refs = [r for r in runs if r.role == "reference"]
    if len(refs) != 1:
        raise SystemExit("the run list needs exactly one run with role = \"reference\"")
    ref_run = refs[0]
    paths = {r.id: neck_file(r) for r in runs}
    data = {r.id: read(paths[r.id]) for r in runs}
    ref = data[ref_run.id]
    receipt = {"reference": ref_run.id, "R0_reference": ref_run.R0, "runs": {}}
    with paths[ref_run.id].open(newline="") as fh:
        if "dissipation" in next(csv.reader(fh)):
            b = budget(paths[ref_run.id].parent)
            receipt["reference_energy_budget"] = {"median": b["relative_residual_median"],
                                                  "p95_abs": b["relative_residual_p95_abs"]}
    fig, ax = plt.subplots(1, 2, figsize=(15.0, 6.2), constrained_layout=True)
    colours = plt.cm.viridis(np.linspace(0.0, 0.85, max(len(runs) - 1, 1)))
    ax[0].plot(ref["R_min"], ref["u_neck"], color="k", lw=2.4, label=ref_run.label or ref_run.id)
    for r, c in zip((r for r in runs if r is not ref_run), colours):
        d = data[r.id]
        lo, hi = max(d["R_min"][0], ref["R_min"][0]), min(d["R_min"][-1], ref["R_min"][-1])
        entry = {"R_shared": [float(lo), float(hi)]}
        if hi > lo:
            R = np.geomspace(lo, hi, 600)
            e = ratio(d, ref, R)
            entry["relative_range"] = [float(e.min()), float(e.max())]
            entry["relative_at_R_over_R0"] = {f"{x:g}": float(ratio(d, ref, np.array([x * ref_run.R0]))[0])
                                              for x in AT if lo <= x * ref_run.R0 <= hi}
            ax[1].loglog(R, np.abs(e), color=c, lw=1.8, label=r.label or r.id)
        with paths[r.id].open(newline="") as fh:
            header = next(csv.reader(fh))
        if "dissipation" in header:
            b = budget(paths[r.id].parent)
            entry["energy_budget"] = {"median": b["relative_residual_median"], "p95_abs": b["relative_residual_p95_abs"]}
        receipt["runs"][r.id] = entry
        ax[0].plot(d["R_min"], d["u_neck"], color=c, lw=1.6, label=r.label or r.id)
    ax[0].set(xscale="log")
    ax[0].set_xlabel(r"$R_{\min}$", fontsize=24)
    ax[0].set_ylabel(r"$u_v$", fontsize=24)
    ax[1].set_xlabel(r"$R_{\min}$", fontsize=24)
    ax[1].set_ylabel(r"$|u_v/u_{v,\mathrm{ref}}-1|$", fontsize=24)
    for axi, tag in zip(ax, ("(a)", "(b)")):
        axi.legend(frameon=False, fontsize=13)
        axi.tick_params(which="both", direction="out", width=1.6, labelsize=18, pad=6)
        axi.tick_params(which="major", length=8)
        axi.tick_params(which="minor", length=4)
        for sp in axi.spines.values():
            sp.set_linewidth(1.6)
        axi.text(-0.02, 1.02, tag, transform=axi.transAxes, fontsize=22, ha="right", va="bottom")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1)
    fig.savefig(a.out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.1, dpi=110)
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
