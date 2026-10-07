#!/usr/bin/env python3
"""Rebuild the finite-Oh comparison, deficit, consistency and R0-family figures."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from public_validation_data import DEFAULT_DATA, REPO, at_radius, export_inputs, load_inputs, stats
from plot_anthony_fig3 import bands, power_law_contact_time, published, relative

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#4D4D4D"


def style(axis, xlabel, ylabel, logy=False):
    axis.set(xscale="log", xlabel=xlabel, ylabel=ylabel)
    if logy:
        axis.set_yscale("log")
    axis.xaxis.label.set_size(25)
    axis.yaxis.label.set_size(25)
    axis.tick_params(which="both", direction="out", width=1.4, labelsize=18, pad=5)
    axis.tick_params(which="major", length=7)
    axis.tick_params(which="minor", length=3.5)
    for spine in axis.spines.values():
        spine.set_linewidth(1.4)


def save(fig, prefix, tag):
    for i, axis in enumerate(fig.axes):
        axis.text(0, 1.02, f"({chr(97 + i)})", transform=axis.transAxes, fontsize=20)
    for extension, dpi in (("pdf", 300), ("png", 160)):
        fig.savefig(f"{prefix}-{tag}.{extension}", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def comparison_metrics(histories):
    fits, reference, comparison = {}, {}, {}
    directory = REPO / "validationCases/anthony2020-fig3-stokes"
    for name in ("finite", "stokes"):
        data = histories[name]
        fits[name] = power_law_contact_time(data["t"], data["R_min"], 1e-6)
    for quantity, xcol, ycol in (("radius", "tau_v", "R_min"),
                                ("velocity", "R_min", "u_v"),
                                ("curvature", "R_min", "abs_two_H")):
        reference[quantity] = published(directory / f"anthony2020-fig3-{quantity}-digitized.csv", xcol, ycol)
        data = histories["finite"]
        xs = data["t"] + fits["finite"].t_con if quantity == "radius" else data["R_min"]
        ys = {"radius": data["R_min"], "velocity": data["u_neck"],
              "curvature": np.abs(data["two_H"])}[quantity]
        x, error = relative(reference[quantity][f"fig3_{quantity}_oh0p6"], xs, ys)
        comparison[quantity] = {"abscissa": xcol,
                               "bands_percent": bands(x, error, [1e-5, 1e-4, 1e-3, 0.01, 0.1]),
                               "coverage": [float(xs.min()), float(xs.max())]}
    return fits, reference, comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output prefix")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--archive", action="store_true", help="load checksum-verified registered histories")
    parser.add_argument("--export-data", type=Path, help="export a new compact bundle from registered histories")
    args = parser.parse_args()
    tags = ("anthony", "deficit", "convergence", "r0")
    targets = [Path(f"{args.out}-{tag}.{ext}") for tag in tags for ext in ("png", "pdf")]
    if args.out.with_suffix(".json").exists() or any(path.exists() for path in targets):
        raise SystemExit("output exists: choose a new render name")
    histories, metadata = load_inputs(None if args.archive else args.data)
    if args.export_data:
        if not args.archive:
            raise SystemExit("--export-data requires --archive")
        export_inputs(args.export_data, histories, metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    finite, stokes = histories["finite"], histories["stokes"]
    fits, reference, comparison = comparison_metrics(histories)
    receipt = {"time_units": "visco-capillary; tau_v=t_v+t_con,v; no division by Oh",
               "no_marker_fitted_offsets": True,
               "contact_time": {name: asdict(fit) for name, fit in fits.items()},
               "comparison": comparison, "input_histories": metadata["histories"]}

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2), layout="constrained")
    for axis, quantity, xlabel, ylabel in zip(
            axes, ("radius", "velocity", "curvature"),
            (r"$\tau_v$", r"$R_{\min}$", r"$R_{\min}$"),
            (r"$R_{\min}$", r"$u_v$", r"$\lvert 2H\rvert$")):
        for name, data, colour, linestyle, label in (
                ("stokes", stokes, GREY, "--", "present FEM, Stokes"),
                ("finite", finite, BLUE, "-", r"present FEM, $\mathrm{Oh}=0.6$")):
            x = data["t"] + fits[name].t_con if quantity == "radius" else data["R_min"]
            y = {"radius": data["R_min"], "velocity": data["u_neck"],
                 "curvature": np.abs(data["two_H"])}[quantity]
            axis.plot(x, y, color=colour, ls=linestyle, lw=2.2, label=label)
        for series, colour, marker, label in (
                ("stokes", GREY, "o", "Anthony et al., Stokes"),
                ("oh0p6", ORANGE, "^", r"Anthony et al., $\mathrm{Oh}=0.6$")):
            axis.plot(*reference[quantity][f"fig3_{quantity}_{series}"].T,
                      marker, mfc="none", color=colour, ms=5, label=label)
        style(axis, xlabel, ylabel, logy=(quantity != "velocity"))
        axis.set_xlim((1e-7, 0.025) if quantity == "radius" else (9e-7, 0.04))
    axes[0].set_ylim(8e-7, 0.04)
    axes[0].legend(frameon=False, fontsize=13, loc="lower right")
    axes[1].set_ylim(1.3, 5.1)
    save(fig, args.out, "anthony")

    grid = np.geomspace(1e-5, 0.03, 1601)
    u, us = at_radius(finite, grid), at_radius(stokes, grid)
    half_window = 0.1 * np.log(10)
    centres = grid[(grid * np.exp(-half_window) >= grid[0]) &
                   (grid * np.exp(half_window) <= grid[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2), layout="constrained")
    axes[0].plot(grid, 100 * (u / us - 1), color=BLUE, lw=2.2)
    axes[1].plot(grid, u - us, color=BLUE, lw=2.2)
    for data, colour, label in ((stokes, GREY, "Stokes"), (finite, BLUE, r"$\mathrm{Oh}=0.6$")):
        slope = -(at_radius(data, centres * np.exp(half_window)) -
                  at_radius(data, centres * np.exp(-half_window))) / (2 * half_window)
        axes[2].plot(centres, np.pi * slope, color=colour, label=label, lw=2)
    axes[2].axhline(1, color="0.7", lw=1)
    axes[2].legend(frameon=False, fontsize=16)
    for axis, ylabel in zip(axes, (r"$100(u_v/u_{v,\mathrm{S}}-1)$",
                                  r"$u_v-u_{v,\mathrm{S}}$", r"$-\pi\,\Delta u_v/\Delta\ln R_{\min}$")):
        style(axis, r"$R_{\min}$", ylabel)
        axis.set_xlim(1e-5, 0.03)
    save(fig, args.out, "deficit")
    receipt["stokes_deficit"] = []
    for radius in (1e-5, 1e-4, 1e-3, 1e-2, 0.03):
        velocity, reference_velocity = at_radius(finite, radius), at_radius(stokes, radius)
        receipt["stokes_deficit"].append({
            "R_min": radius, "relative_percent": float(100 * (velocity / reference_velocity - 1)),
            "absolute": float(velocity - reference_velocity),
            "effective_C_ratio": float(np.exp(np.pi * (velocity - reference_velocity)))})

    fig, axis = plt.subplots(figsize=(10.5, 4.5), layout="constrained")
    receipt["convergence"] = {}
    for name, ref_name, colour, label in (
            ("moving", "lab", BLUE, "moving / laboratory frame"),
            ("single_map", "lab", ORANGE, "single / composite tip map"),
            ("half_dt", "single_map", "#009E73", "half / production time step")):
        data, ref = histories[name], histories[ref_name]
        radius = np.geomspace(1.01e-3, min(data["R_min"][-1], ref["R_min"][-1]), 600)
        error = 100 * (at_radius(data, radius) / at_radius(ref, radius) - 1)
        axis.plot(radius, error, color=colour, lw=2, label=label)
        receipt["convergence"][name] = stats(error)
    axis.axhline(0, color="0.5", lw=0.8)
    style(axis, r"$R_{\min}$", r"relative difference [\%]")
    axis.legend(frameon=False, fontsize=17)
    save(fig, args.out, "convergence")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.2), layout="constrained")
    colours = [GREY, "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#D55E00", "#0072B2", "#000000"]
    for (name, data), colour in zip(
            [(name, data) for name, data in histories.items() if name.startswith("family_")], colours):
        R0 = metadata["histories"][name]["R0"]
        exponent = int(np.floor(np.log10(R0)))
        mantissa = R0 / 10 ** exponent
        number = rf"10^{{{exponent}}}" if np.isclose(mantissa, 1) else rf"{mantissa:.3g}\times10^{{{exponent}}}"
        axes[0].plot(data["R_min"], data["u_neck"], color=colour, lw=1.6, label=rf"$R_0={number}$")
    axes[0].legend(frameon=False, fontsize=11, ncol=2)
    for data, colour, label in ((finite, BLUE, r"$R_0=10^{-6}$"),
                                (histories["lab"], ORANGE, r"$R_0=10^{-3}$")):
        axes[1].plot(data["R_min"], data["u_neck"], color=colour, lw=2, label=label)
    radius = np.geomspace(3e-3, min(finite["R_min"][-1], histories["lab"]["R_min"][-1]), 600)
    error = 100 * (at_radius(histories["lab"], radius) / at_radius(finite, radius) - 1)
    receipt["finite_oh_R0_independence"] = stats(error)
    axes[2].plot(radius, error, color=BLUE, lw=2)
    axes[2].axhline(0, color="0.5", lw=0.8)
    style(axes[2], r"$R_{\min}$", r"$100(u_{v,10^{-3}}/u_{v,10^{-6}}-1)$")
    axes[2].set_xlim(3e-3, 0.03)
    axes[2].set_xticks([3e-3, 1e-2, 3e-2],
                      labels=[r"$3\times10^{-3}$", r"$10^{-2}$", r"$3\times10^{-2}$"])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    for axis in axes[:2]:
        style(axis, r"$R_{\min}$", r"$u_v$")
        axis.set_xlim(9e-7, 0.04)
    axes[1].legend(frameon=False, fontsize=14)
    save(fig, args.out, "r0")
    receipt["volume_relative_drift"] = float(finite["volume"][-1] / finite["volume"][0] - 1)
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
