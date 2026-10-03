#!/usr/bin/env python3
"""Build the public Anthony/FEM/BIM validation overview.

The figure keeps the three tests distinct in the machine-readable receipt while
putting their shared velocity observable on one page: published Stokes and
Oh = 0.6 markers, the present FEM curves, and the independent Stokes BIM
startup. No time or velocity offsets are fitted.

    python postProcess/plot_public_validation.py \
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
from coalescence.analysis.neck import relative_deviation  # noqa: E402
from coalescence.analysis.runs import load_runs, load_series, neck_file  # noqa: E402

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "pdf.fonttype": 42,
})


def read_markers(path: Path, series: str) -> np.ndarray:
    with path.open(newline="") as fh:
        out = np.array([(float(row["R_min"]), float(row["u_v"]))
                        for row in csv.DictReader(fh) if row["series"] == series])
    return out[np.argsort(out[:, 0])]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style(axis: plt.Axes, xlabel: str, ylabel: str) -> None:
    axis.set_xscale("log")
    axis.set_xlabel(xlabel, fontsize=21)
    axis.set_ylabel(ylabel, fontsize=21)
    axis.tick_params(which="both", direction="in", top=True, right=True,
                     width=1.4, labelsize=15, pad=5)
    axis.tick_params(which="major", length=7)
    axis.tick_params(which="minor", length=3.5)
    axis.grid(True, which="major", alpha=0.16)
    for spine in axis.spines.values():
        spine.set_linewidth(1.4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="new output prefix; existing files are never overwritten")
    args = parser.parse_args()
    targets = [args.out.with_suffix(ext) for ext in (".pdf", ".png", ".json")]
    if any(path.exists() for path in targets):
        raise SystemExit("output exists: choose a new render name")

    validation_list = REPO / "validationCases/anthony2020-fig3-oh06/runs.toml"
    bim_list = REPO / "verificationCases/bim-vs-fem-stokes-startup/runs.toml"
    data_dir = REPO / "validationCases/anthony2020-fig3-stokes"
    validation_runs = load_runs(validation_list)
    finite_run = next(run for run in validation_runs if run.id == "la0-oh06-r0-1e-6-16fc4a0")
    stokes_run = next(run for run in validation_runs if run.id == "la0-fig3-r0-1e-6-zone-6c9f06c")
    finite = load_series(validation_runs, finite_run)
    stokes = load_series(validation_runs, stokes_run)

    anthony_oh = read_markers(data_dir / "anthony2020-fig3-velocity-digitized.csv",
                              "fig3_velocity_oh0p6")
    anthony_stokes = read_markers(data_dir / "anthony2020-fig3-velocity-digitized.csv",
                                  "fig3_velocity_stokes")

    bim_runs = load_runs(bim_list)
    bim_reference = next(run for run in bim_runs if run.role == "reference")
    bim_leaves = [run for run in bim_runs if run.solver == "bim" and not any(
        child.continues == run.id and child.series == run.series for child in bim_runs)]
    bim_series = [(run, load_series(bim_runs, run)) for run in bim_leaves]

    blue, orange, grey = "#0072B2", "#D55E00", "#4D4D4D"
    bim_colours = ["#009E73", "#CC79A7", "#E69F00"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), layout="constrained")

    velocity = axes[0]
    velocity.plot(stokes["R_min"], stokes["u_neck"], color=grey, lw=2.0, ls="--",
                  label="present FEM, Stokes", zorder=3)
    velocity.plot(finite["R_min"], finite["u_neck"], color=blue, lw=2.2,
                  label=r"present FEM, $\mathrm{Oh}=0.6$", zorder=4)
    velocity.plot(anthony_stokes[:, 0], anthony_stokes[:, 1], "o", ms=4.2,
                  mfc="none", mec=grey, mew=0.9, label="Anthony et al., Stokes", zorder=5)
    velocity.plot(anthony_oh[:, 0], anthony_oh[:, 1], "^", ms=4.2,
                  mfc="none", mec=orange, mew=0.9, label=r"Anthony et al., $\mathrm{Oh}=0.6$", zorder=5)
    for (run, data), colour in zip(bim_series, bim_colours):
        velocity.plot(data["R_step"], data["u_neck"], color=colour, lw=1.15,
                      alpha=0.9, label=run.label or "present BIM", zorder=2)
    style(velocity, r"$R_{\min}$", r"$u_v$")
    velocity.set_xlim(9e-7, 4e-2)
    velocity.set_ylim(2.35, 5.05)
    velocity.legend(frameon=False, fontsize=9.2, loc="lower left", ncol=2,
                    columnspacing=1.0, handlelength=2.2)

    residual = axes[1]
    for markers, data, colour, label, marker in (
        (anthony_stokes, stokes, grey, "FEM / Anthony, Stokes", "o"),
        (anthony_oh, finite, orange, r"FEM / Anthony, $\mathrm{Oh}=0.6$", "^"),
    ):
        x, error = relative_deviation(markers[:, 0], markers[:, 1],
                                      data["R_min"], data["u_neck"])
        residual.plot(x, error, marker=marker, ms=3.8, ls="-",
                      lw=1.0, color=colour, mfc="white", label=label)
    for (run, data), colour in zip(bim_series, bim_colours):
        x, error = relative_deviation(data["R_step"], data["u_neck"],
                                      stokes["R_min"], stokes["u_neck"])
        residual.plot(x, error, color=colour, lw=1.15,
                      label=(run.label or "BIM") + " / FEM")
    residual.axhline(0.0, color="0.45", lw=0.9)
    style(residual, r"$R_{\min}$", r"relative difference [\%]")
    # The published startup markers are contact-time sensitive and span several
    # hundred percent at the first point. Keep this panel readable for the
    # developed comparison; the startup adjudication is shown in the BIM figure.
    residual.set_xlim(1e-5, 4e-2)
    residual.set_ylim(-1.2, 1.2)
    residual.legend(frameon=False, fontsize=9.2, loc="lower left", ncol=2,
                    columnspacing=1.0, handlelength=2.2)

    for axis, tag in zip(axes, ("(a)", "(b)")):
        axis.text(0.0, 1.02, tag, transform=axis.transAxes,
                  va="bottom", ha="left", fontsize=17)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(args.out.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    metrics = {
        "figure": "public validation overview",
        "no_fitted_offsets": True,
        "inputs": {
            "validation_runs": str(validation_list.relative_to(REPO)),
            "bim_runs": str(bim_list.relative_to(REPO)),
            "anthony_velocity": digest(data_dir / "anthony2020-fig3-velocity-digitized.csv"),
            "finite_oh_neck": finite_run.neck_sha256,
            "stokes_neck": stokes_run.neck_sha256,
        },
        "bim": {
            "solver": "axisymmetric boundary-integral method",
            "not_bem": True,
            "series": [run.series for run, _ in bim_series],
        },
        "comparison": {
            "finite_oh_max_abs_percent": float(np.max(np.abs(
                relative_deviation(anthony_oh[:, 0], anthony_oh[:, 1], finite["R_min"], finite["u_neck"])[1]))),
            "stokes_max_abs_percent": float(np.max(np.abs(
                relative_deviation(anthony_stokes[:, 0], anthony_stokes[:, 1], stokes["R_min"], stokes["u_neck"])[1]))),
            "bim_max_abs_percent": {
                run.series: float(np.max(np.abs(relative_deviation(
                    data["R_step"], data["u_neck"], stokes["R_min"], stokes["u_neck"])[1])))
                for run, data in bim_series
            },
        },
    }
    args.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
