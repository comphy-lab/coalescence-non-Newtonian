#!/usr/bin/env python3
"""Stokes startups from several initial neck radii R0: u_v(R_min) and their merger.

(a) u_v(R_min) for every run in the run list, with the digitised Stokes markers of
Anthony, Harris & Basaran (2020, Fig. 3b) and the two leading-order forms of the
Eggers, Lister & Stone (1999) law (``coalescence.analysis.theory``); (b) the deviation
of each run from the smallest-R0 run at equal R_min, against R_min/R0. Every run must
have reached R_stop. No velocity offset or time shift is fitted.

    python postProcess/plot_startup_family.py \
        verificationCases/fem-startup-r0-independence/runs.toml \
        --anthony validationCases/anthony2020-fig3-stokes/anthony2020-fig3-velocity-digitized.csv \
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coalescence.analysis.neck import interp_log, peak  # noqa: E402
from coalescence.analysis.runs import load_runs, load_series, neck_file  # noqa: E402
from coalescence.analysis.theory import R_MAX_STOKES, u_eggers, u_leading_log  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "pdf.fonttype": 42,
})

COLOURS = ["#1f4e79", "#2a9d8f", "#b34e33", "#e9a23b", "#3a7d44", "#7b4f9d", "#8c6d1f", "#555555"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (6, 2))]


def r0_label(r0: float) -> str:
    lg = np.log10(r0)
    if abs(lg - round(4 * lg) / 4) < 1e-9:                 # integer, half or quarter decade
        h = round(4 * lg) / 4
        return rf"$R_0=10^{{{int(h)}}}$" if h == int(h) else rf"$R_0=10^{{{h:g}}}$"
    m, e = f"{r0:.0e}".split("e")
    return rf"$R_0={m}\times10^{{{int(e)}}}$"


def anthony_stokes(path: Path) -> np.ndarray:
    with path.open(newline="") as fh:
        pts = np.array([(float(r["R_min"]), float(r["u_v"])) for r in csv.DictReader(fh)
                        if r["series"] == "fig3_velocity_stokes"])
    return pts[np.argsort(pts[:, 0])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, help="run list (runs.toml)")
    ap.add_argument("--anthony", type=Path, default=None, help="digitised Anthony et al. (2020) Fig. 3(b) markers")
    ap.add_argument("--endpoint", type=float, default=0.03)
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.pdf, .png, .json)")
    a = ap.parse_args()

    runs = sorted(load_runs(a.runs), key=lambda r: r.R0)
    series = []
    for run in runs:
        summary = json.loads((neck_file(run).parent / "summary.json").read_text())
        if summary["status"] != "reached_R_stop":
            raise ValueError(f"{run.id} did not reach R_stop: {summary['status']}")
        series.append(load_series(runs, run))
    ref_run, ref = runs[0], series[0]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), layout="constrained")
    metrics = {"run_list": str(a.runs), "reference": ref_run.id, "runs": {},
               "theory": {"leading_log": "u_v = -(1/pi) ln R_min",
                          "eggers": "u_v = -(1 + ln tau_v)/pi, R_min = -(tau_v/pi) ln tau_v"}}
    if a.anthony is not None:
        an = anthony_stokes(a.anthony)
        axes[0].plot(an[:, 0], an[:, 1], "o", ms=3.5, mfc="none", mec="0.45", mew=0.9, zorder=2,
                     label=r"Anthony \textit{et al.} (2020), Stokes")
        metrics["anthony_sha256"] = hashlib.sha256(a.anthony.read_bytes()).hexdigest()
    Rt = np.geomspace(ref_run.R0, R_MAX_STOKES, 400)
    axes[0].plot(Rt, u_leading_log(Rt), color="k", lw=1.0, ls=":", zorder=4, label=r"$u_v=-\pi^{-1}\ln R_{\min}$")
    axes[0].plot(Rt, u_eggers(Rt), color="k", lw=1.2, ls="--", zorder=4,
                 label=r"$u_v=\mathrm{d}R_{\min}/\mathrm{d}\tau_v$, $R_{\min}=-\pi^{-1}\tau_v\ln\tau_v$")
    for k, (run, d) in enumerate(zip(runs, series)):
        c, ls = COLOURS[k % len(COLOURS)], STYLES[k % len(STYLES)]
        axes[0].plot(d["R_min"], d["u_neck"], color=c, ls=ls, lw=1.8, zorder=3, label=r0_label(run.R0))
        u_pk, R_pk = peak(d["R_min"], d["u_neck"])
        entry = {"R0": run.R0, "u_peak": u_pk, "R_at_peak_over_R0": R_pk / run.R0,
                 "u_at_endpoint": float(interp_log(a.endpoint, d["R_min"], d["u_neck"]))}
        if k > 0:
            R = np.geomspace(max(d["R_min"][0], ref["R_min"][0]), min(a.endpoint, d["R_min"][-1], ref["R_min"][-1]), 800)
            dev = 100.0 * (interp_log(R, d["R_min"], d["u_neck"]) / interp_log(R, ref["R_min"], ref["u_neck"]) - 1.0)
            axes[1].plot(R / run.R0, np.abs(dev), color=c, ls=ls, lw=1.8)
            within = {}
            for tol in (1.0, 0.3, 0.1):
                bad = np.nonzero(np.abs(dev) > tol)[0]
                within[f"{tol}%"] = (1.0 if not bad.size else
                                     float(R[bad[-1] + 1] / run.R0) if bad[-1] + 1 < R.size else None)
            entry.update({"deviation_percent_at_endpoint": float(dev[-1]), "R_over_R0_after_which_within": within})
        metrics["runs"][run.id] = entry
    axes[0].set(xscale="log", xlabel=r"$R_{\min}$", ylabel=r"$u_v$")
    axes[1].set(xscale="log", yscale="log", xlabel=r"$R_{\min}/R_0$",
                ylabel=r"$100\,|u_v/u_v^{\mathrm{ref}}-1|$ [\%]")
    for y in (1.0, 0.1):
        axes[1].axhline(y, color="0.6", lw=0.8, ls=":")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9.5,
               columnspacing=1.4, handlelength=2.4)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
        ax.grid(True, which="major", alpha=0.18)
        ax.text(0.0, 1.02, tag, transform=ax.transAxes, va="bottom", ha="left", fontsize=11)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    a.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
