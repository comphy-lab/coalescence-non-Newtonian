#!/usr/bin/env python3
"""Boundary-integral against finite-element Stokes startup at one initial neck radius.

(a) u_v(R_min) in the coordinates of Anthony, Harris & Basaran (2020, Fig. 3b), with
their digitised Stokes markers and the two leading-order forms of the Eggers, Lister &
Stone (1999) law (a function of R_min only, so drawn in (a) only); (b) the startup
against (R_min - R0)/R0; (c) 100 (BIM/FEM - 1) for
u_v and for the tip radius. Boundary-integral speeds are assigned to the geometric-mean
radius of each step (``coalescence.analysis.neck``) and compared at those samples with
the interpolated finite-element curve. The tip radius of the boundary-integral run is
divided by ``--estimator-factor``, the ratio of the two codes' tip-radius estimators.
No offsets are fitted.

    python postProcess/plot_bim_vs_fem.py verificationCases/bim-vs-fem-stokes-startup/runs.toml \
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
from coalescence.analysis.neck import interp_log, peak, relative_deviation  # noqa: E402
from coalescence.analysis.runs import load_runs, load_series  # noqa: E402
from coalescence.analysis.theory import u_eggers, u_leading_log  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "pdf.fonttype": 42,
})

COLOURS = ["#b34e33", "#3a7d44", "#7b4f9d", "#e9a23b"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, help="run list (runs.toml): one role=reference FEM run and BIM series")
    ap.add_argument("--anthony", type=Path, required=True, help="digitised Anthony et al. (2020) Fig. 3(b) markers")
    ap.add_argument("--estimator-factor", type=float, default=0.962)
    ap.add_argument("--out", type=Path, required=True, help="output prefix (.pdf, .png, .json)")
    a = ap.parse_args()

    runs = load_runs(a.runs)
    ref_run = next(r for r in runs if r.role == "reference")
    R0 = ref_run.R0
    f = load_series(runs, ref_run)
    leaves = [r for r in runs if r.solver == "bim" and not any(
        c.continues == r.id and c.series == r.series for c in runs)]
    with a.anthony.open(newline="") as fh:
        an = np.array([(float(r["R_min"]), float(r["u_v"])) for r in csv.DictReader(fh)
                       if r["series"] == "fig3_velocity_stokes"])

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6), layout="constrained")
    ax[0].plot(an[:, 0], an[:, 1], "o", ms=3.5, mfc="none", mec="0.45", mew=0.9, zorder=2,
               label=r"Anthony \textit{et al.} (2020), Stokes")
    for axis, x in ((ax[0], f["R_min"]), (ax[1], f["R_min"] / R0 - 1)):
        axis.plot(x, f["u_neck"], color="#1f4e79", lw=1.8, zorder=3, label=ref_run.label or "finite element")
    # The theory is a function of R_min alone, for a neck grown from point contact; it is drawn
    # against R_min only, not against (R_min - R0)/R0, where it would appear as a spurious plateau.
    Rt = R0 * (1 + np.geomspace(1e-9, 40.0, 500))
    ax[0].plot(Rt, u_leading_log(Rt), color="k", lw=1.0, ls=":", zorder=5, label=r"$u_v=-\pi^{-1}\ln R_{\min}$")
    ax[0].plot(Rt, u_eggers(Rt), color="k", lw=1.2, ls="--", zorder=5,
               label=r"$u_v=\mathrm{d}R_{\min}/\mathrm{d}\tau_v$, $R_{\min}=-\pi^{-1}\tau_v\ln\tau_v$")
    metrics = {"run_list": str(a.runs), "R0": R0, "estimator_factor": a.estimator_factor,
               "anthony_sha256": hashlib.sha256(a.anthony.read_bytes()).hexdigest(), "series": {}}
    R_end = R0
    for k, leaf in enumerate(leaves):
        b = load_series(runs, leaf)
        c, mk = COLOURS[k % len(COLOURS)], ("o", "s", "D", "^")[k % 4]
        style = dict(ms=2.6, mfc="none", mec=c, mew=0.8, ls="none", zorder=4, label=leaf.label)
        ax[0].plot(b["R_step"], b["u_neck"], marker=mk, **style)
        ax[1].plot(b["R_step"] / R0 - 1, b["u_neck"], marker=mk, **style)
        Ru, du = relative_deviation(b["R_step"], b["u_neck"], f["R_min"], f["u_neck"])
        Rr, dr = relative_deviation(b["R_min"], b["tip_radius"] / a.estimator_factor, f["R_min"], f["tip_radius"])
        short = leaf.label.removeprefix("boundary integral, ")
        ax[2].plot(Ru / R0 - 1, du, color=c, lw=1.6, label=rf"$u_v$, {short}")
        ax[2].plot(Rr / R0 - 1, dr, color=c, lw=1.1, ls="--", label=rf"tip radius, {short}")
        u_pk, R_pk = peak(b["R_step"], b["u_neck"])
        metrics["series"][leaf.series] = {
            "runs": [r.id for r in runs if r.series == leaf.series],
            "R_first": float(b["R_step"][0]), "R_end": float(b["R_min"][-1]),
            "u_max": u_pk, "R_at_u_max": R_pk,
            "u_v_dev_percent": {"min": float(du.min()), "max": float(du.max()), "median": float(np.median(du))},
            "tip_dev_percent_max_abs": float(np.abs(dr).max())}
        R_end = max(R_end, b["R_min"][-1])
    u_f, R_f = peak(f["R_min"], f["u_neck"])
    metrics["reference_peak"] = {"u_max": u_f, "R_at_u_max": R_f}
    ax[0].set(xscale="log", xlabel=r"$R_{\min}$", ylabel=r"$u_v$", xlim=(0.95 * R0, 1.05 * R_end))
    ax[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    ax[1].set(xscale="log", xlabel=r"$(R_{\min}-R_0)/R_0$", ylabel=r"$u_v$",
              xlim=(1e-9, 1.3 * (R_end / R0 - 1)), ylim=(2.4, 5.05))
    ax[2].axhline(0.0, color="0.5", lw=0.8)
    ax[2].set(xscale="log", xlabel=r"$(R_{\min}-R_0)/R_0$", ylabel=r"$100\,(\mathrm{BIM}/\mathrm{FEM}-1)$ [\%]",
              xlim=ax[1].get_xlim())
    ax[2].legend(frameon=False, fontsize=8.5, loc="lower left", ncol=2)
    for axis, tag in zip(ax, ("(a)", "(b)", "(c)")):
        axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
        axis.grid(True, which="major", alpha=0.18)
        axis.text(0.0, 1.02, tag, transform=axis.transAxes, va="bottom", ha="left", fontsize=11)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    a.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
