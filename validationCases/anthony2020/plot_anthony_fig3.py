#!/usr/bin/env python3
"""Anthony, Harris & Basaran (2020) Figure 3 from the R0 = 1e-6 runs of this folder.

(a) R_min against tau_v = t + t_con with an inset |2H| against R_min; (b) u_v against
R_min. The Stokes run is required and the Oh = 0.6 run optional; each is the output folder
of ``run_stokes.sh`` or ``run_oh0.6.sh``. Markers are the digitised published series
(``data/``); t_con follows the paper's procedure, a power law fitted once R_min has grown by
one decade (10 R0 to 100 R0) and extrapolated to zero radius
(``coalescence.analysis.contact_time``). Black lines are the Stokes-theory curves drawn in
the published figure. No quantity is fitted to the published data. The .json gives, for
each run, the solver commit, the neck-history checksum and the relative differences from
the published markers by range of abscissa.

    python validationCases/anthony2020/plot_anthony_fig3.py <stokes run> [--finite <Oh 0.6 run>] \
        --out <output prefix>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from coalescence.analysis.contact_time import power_law_contact_time, stokes_law_contact_time  # noqa: E402
from coalescence.analysis.neck import read_neck  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "text.usetex": True, "text.latex.preamble": r"\usepackage{amsmath}", "pdf.fonttype": 42,
})


def published(path: Path, xcol: str, ycol: str) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["series"], []).append((float(r[xcol]), float(r[ycol])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def relative(ref: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """100 (present/published - 1) at the published abscissae inside the computed range."""
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("the computed abscissa must increase strictly for interpolation")
    k = (ref[:, 0] >= xs.min()) & (ref[:, 0] <= xs.max())
    return ref[k, 0], 100 * (np.interp(np.log(ref[k, 0]), np.log(xs), ys) / ref[k, 1] - 1)


BRANCH_OH = {"stokes": None, "oh0p6": 0.6}   # Oh recorded in the run manifest; None is Stokes


def load_run(folder: Path, branch: str) -> dict:
    """Neck history of a completed run folder of ``branch``, with its solver commit and R0 (the t = 0 row)."""
    path = folder / "neck.csv"
    summary = json.loads((folder / "summary.json").read_text())
    if summary["status"] != "reached_R_stop":
        raise SystemExit(f"{folder} did not reach R_stop: {summary['status']}")
    manifest = json.loads((folder / "run-manifest.json").read_text())
    if manifest.get("Oh") != BRANCH_OH[branch]:
        raise SystemExit(f"{folder} is a run at Oh = {manifest.get('Oh')}, not the {branch} branch")
    d = read_neck(path, extra=("two_H",))
    if d["t"][0] != 0.0:
        raise SystemExit(f"{folder}: the neck history must start with the initial bridge at t = 0")
    return {"folder": folder, "t": d["t"], "R": d["R_min"], "u": d["u_neck"], "H2": np.abs(d["two_H"]),
            "R0": float(d["R_min"][0]), "case_id": manifest["case_id"],
            "component_commit": manifest["component_commit"],
            "neck_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compare(run: dict, branch: str, ref_r, ref_u, ref_k) -> dict:
    """Contact time and relative differences of one run from the published ``branch`` series."""
    t, R, u, H2, R0 = run["t"], run["R"], run["u"], run["H2"], run["R0"]
    fit = power_law_contact_time(t, R, R0)
    run["tau"] = t + fit.t_con
    ok = run["tau"] > 0
    xr, er = relative(ref_r[f"fig3_radius_{branch}"], run["tau"][ok], R[ok])
    xu, eu = relative(ref_u[f"fig3_velocity_{branch}"], R, u)
    xk, ek = relative(ref_k[f"fig3_curvature_{branch}"], R, H2)
    return {
        "case_id": run["case_id"], "component_commit": run["component_commit"],
        "neck_sha256": run["neck_sha256"], "R0": R0, "R_end": float(R[-1]),
        "t_con_power_law": {"value": fit.t_con, "A": fit.prefactor, "n": fit.exponent,
                            "fit_window_R": list(fit.window), "samples": fit.samples},
        "t_con_sensitivity": {"stokes_law_at_10R0": stokes_law_contact_time(t, R, 10 * R0), "zero": 0.0},
        "R_vs_tau_relative_percent": bands(xr, er, [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-1]),
        "u_v_relative_percent": bands(xu, eu, [1e-7, 2e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1.0]),
        "abs_2H_relative_percent": bands(xk, ek, [1e-7, 2e-6, 1e-5, 1e-4, 1e-3, 1e-1]),
        "abs_2H_R3_end": float(H2[-1] * R[-1] ** 3),
    }


def bands(x: np.ndarray, e: np.ndarray, edges: list[float]) -> dict:
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        if k.any():
            out[f"{lo:g}..{hi:g}"] = {"n": int(k.sum()), "median_percent": float(np.median(e[k])),
                                      "min_percent": float(e[k].min()), "max_percent": float(e[k].max())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stokes", type=Path, help="output folder of the Stokes run (run_stokes.sh)")
    ap.add_argument("--finite", type=Path, default=None, help="output folder of the Oh = 0.6 run (run_oh0.6.sh)")
    ap.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data",
                    help="folder with the anthony2020-fig3-*-digitized.csv files")
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.pdf, .png, .json)")
    a = ap.parse_args()

    files = {q: a.data / f"anthony2020-fig3-{q}-digitized.csv" for q in ("radius", "velocity", "curvature")}
    ref_r = published(files["radius"], "tau_v", "R_min")
    ref_u = published(files["velocity"], "R_min", "u_v")
    ref_k = published(files["curvature"], "R_min", "abs_two_H")
    runs = {"stokes": load_run(a.stokes, "stokes")}
    if a.finite is not None:
        runs["oh0p6"] = load_run(a.finite, "oh0p6")
    receipt = {"published_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()},
               **{branch: compare(run, branch, ref_r, ref_u, ref_k) for branch, run in runs.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")

    blue, grey, light = "#1f4e79", "0.15", "#c07a2c"
    fig, ax = plt.subplots(1, 2, figsize=(16.5, 7.0), constrained_layout=True)
    # (a) R_min(tau_v)
    tt = np.logspace(-8, -1, 400); good = -(tt / np.pi) * np.log(tt) < 0.03
    ax[0].plot(tt[good], -(tt[good] / np.pi) * np.log(tt[good]), "k-", lw=1.3,
               label=r"$R_{\min}=-(\tau_v/\pi)\ln\tau_v$")
    ax[0].plot(ref_r["fig3_radius_oh0p6"][:, 0], ref_r["fig3_radius_oh0p6"][:, 1], "^", ms=5,
               mfc="none", mec=light, mew=1.1, label=r"Anthony et al., $\mathrm{Oh}=0.6$")
    ax[0].plot(ref_r["fig3_radius_stokes"][:, 0], ref_r["fig3_radius_stokes"][:, 1], "o", ms=5,
               mfc="none", mec=grey, mew=1.1, label=r"Anthony et al., Stokes")
    st = runs["stokes"]
    ok = st["tau"] > 0
    ax[0].plot(st["tau"][ok], st["R"][ok], color=blue, lw=2.4, label=r"present, Stokes")
    if "oh0p6" in runs:
        fo = runs["oh0p6"]
        okf = fo["tau"] > 0
        ax[0].plot(fo["tau"][okf], fo["R"][okf], color=light, lw=2.4, label=r"present, $\mathrm{Oh}=0.6$")
    ax[0].set(xscale="log", yscale="log", xlim=(1e-8, 0.2), ylim=(5e-7, 0.1))
    ax[0].set_xlabel(r"$\tau_v$", fontsize=26); ax[0].set_ylabel(r"$R_{\min}$", fontsize=26)
    ax[0].legend(frameon=False, fontsize=15, loc="lower right")
    ins = ax[0].inset_axes([0.1, 0.56, 0.36, 0.38])
    rr = np.logspace(-6, -1, 100)
    ins.plot(rr, rr ** -3, "k-", lw=1.0)
    ins.plot(ref_k["fig3_curvature_oh0p6"][:, 0], ref_k["fig3_curvature_oh0p6"][:, 1], "^", ms=3.5,
             mfc="none", mec=light, mew=0.9)
    ins.plot(ref_k["fig3_curvature_stokes"][:, 0], ref_k["fig3_curvature_stokes"][:, 1], "o", ms=3.5,
             mfc="none", mec=grey, mew=0.9)
    ins.plot(st["R"], st["H2"], color=blue, lw=1.8)
    if "oh0p6" in runs:
        ins.plot(fo["R"], fo["H2"], color=light, lw=1.8)
    ins.set(xscale="log", yscale="log")
    ins.set_xlabel(r"$R_{\min}$", fontsize=15, labelpad=1); ins.set_ylabel(r"$|2H|$", fontsize=15, labelpad=1)
    ins.tick_params(which="both", labelsize=11, width=1.0)
    # (b) u_v(R_min)
    rr = np.logspace(-6, 0, 400)
    ax[1].plot(rr, -np.log(rr) / np.pi, "k-", lw=1.3, label=r"$u_v=-(1/\pi)\ln R_{\min}$")
    ax[1].plot(ref_u["fig3_velocity_oh0p6"][:, 0], ref_u["fig3_velocity_oh0p6"][:, 1], "^", ms=5,
               mfc="none", mec=light, mew=1.1, label=r"Anthony et al., $\mathrm{Oh}=0.6$")
    ax[1].plot(ref_u["fig3_velocity_stokes"][:, 0], ref_u["fig3_velocity_stokes"][:, 1], "o", ms=5,
               mfc="none", mec=grey, mew=1.1, label=r"Anthony et al., Stokes")
    ax[1].plot(st["R"], st["u"], color=blue, lw=2.4, label=r"present, Stokes")
    if "oh0p6" in runs:
        ax[1].plot(fo["R"], fo["u"], color=light, lw=2.4, label=r"present, $\mathrm{Oh}=0.6$")
    ax[1].set(xscale="log", xlim=(7e-7, 1.3))
    ax[1].set_xlabel(r"$R_{\min}$", fontsize=26); ax[1].set_ylabel(r"$u_v$", fontsize=26)
    ax[1].legend(frameon=False, fontsize=15, loc="upper right")
    for axi in ax:
        axi.tick_params(which="both", direction="out", width=1.8, labelsize=19, pad=6)
        axi.tick_params(which="major", length=8); axi.tick_params(which="minor", length=4)
        for sp in axi.spines.values():
            sp.set_linewidth(1.8)
    for lab, axi in zip("ab", ax):
        axi.set_title(rf"$({lab})$", loc="left", fontsize=22)
    fig.savefig(a.out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.1, dpi=300)
    fig.savefig(a.out.with_suffix(".png"), bbox_inches="tight", pad_inches=0.1, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
