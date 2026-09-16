"""Fig. 3 coordinates of Anthony, Harris & Basaran (2020): R_min(tau_v) and u_v(R_min)."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def read_neck(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return {key: np.array([float(row[key]) for row in rows], dtype=float) for key in rows[0]}


def eggers_tau_for_rmin(rmin: float) -> float:
    """Invert Eggers so the first point can be placed on the theoretical time origin."""
    if rmin <= 0.0 or rmin >= 0.1:
        return 0.0
    lo, hi = 1e-16, 1.0 / math.e
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        val = -(mid / math.pi) * math.log(mid)
        if val > rmin:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def contact_time(t: np.ndarray, rmin: np.ndarray) -> float:
    """Power-law extrapolation of R_min(t) to zero after one decade of growth.

    Until that decade exists, use the Eggers time at the initial radius. That is
    a time origin, not a test of the R_min(τ) law.
    """
    r0 = rmin[0]
    mask = rmin >= 10.0 * r0
    if np.count_nonzero(mask) < 4:
        mask = rmin >= 2.0 * r0
    if np.count_nonzero(mask) >= 3:
        tt = np.log(np.clip(t[mask], 1e-30, None))
        rr = np.log(rmin[mask])
        slope, intercept = np.polyfit(tt, rr, 1)
        if abs(slope) >= 1e-12 and slope < 0:
            return float(max(0.0, math.exp(-intercept / slope)))
    return eggers_tau_for_rmin(float(r0))


def fig3_series(neck: dict[str, np.ndarray], *, inertia: bool, oh: float | None) -> dict[str, np.ndarray]:
    t_con = contact_time(neck["t"], neck["R_min"])
    tau = neck["t"] + t_con
    if inertia:
        if oh is None:
            raise ValueError("Oh required for an inertial series")
        tau_v = tau / oh
        u_v = neck["u_min"] * oh
    else:
        tau_v = tau
        u_v = neck["u_min"]
    return {
        "tau_v": tau_v,
        "R_min": neck["R_min"],
        "u_v": u_v,
        "t_con": np.full_like(tau_v, t_con),
    }


def eggers(tau_v: np.ndarray) -> np.ndarray:
    tau = np.clip(tau_v, 1e-16, 0.999)
    return -(tau / math.pi) * np.log(tau)


def style(ax) -> None:
    ax.tick_params(which="both", direction="out", width=3, labelsize=30, pad=10)
    ax.tick_params(which="major", length=12)
    ax.tick_params(which="minor", length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()


def plot_fig3(series: list[tuple[str, dict[str, np.ndarray]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_r, ax_u = axes
    colours = {"Stokes": "tab:blue", "Oh = 0.6": "tab:red"}
    tau_min = min(float(np.min(item[1]["tau_v"])) for item in series)
    tau_max = max(float(np.max(item[1]["tau_v"])) for item in series)
    tau_th = np.logspace(math.log10(max(tau_min * 0.3, 1e-8)), math.log10(min(max(tau_max, 0.03), 0.3)), 400)
    ax_r.loglog(tau_th, eggers(tau_th), "k--", lw=2, label=r"$R_{\min}=-(\tau_v/\pi)\ln\tau_v$")
    ax_r.loglog(tau_th, tau_th, "k:", lw=2, label=r"$R_{\min}=\tau_v$")
    for name, data in series:
        ax_r.loglog(
            data["tau_v"],
            data["R_min"],
            "o",
            ms=5,
            color=colours.get(name, "tab:gray"),
            label=name,
        )
        ax_u.semilogx(
            data["R_min"],
            data["u_v"],
            "o",
            ms=5,
            color=colours.get(name, "tab:gray"),
            label=name,
        )
    r_line = np.logspace(-6, 0, 400)
    ax_u.semilogx(r_line, -(1.0 / math.pi) * np.log(r_line), "k-", lw=2, label=r"$u_v=-(1/\pi)\ln R_{\min}$")
    ax_r.set_xlabel(r"$\tau_v$", fontsize=40, labelpad=15)
    ax_r.set_ylabel(r"$R_{\min}$", fontsize=40, labelpad=15)
    ax_u.set_xlabel(r"$R_{\min}$", fontsize=40, labelpad=15)
    ax_u.set_ylabel(r"$u_v$", fontsize=40, labelpad=15)
    ax_r.set_xlim(tau_th[0], tau_th[-1])
    ax_r.set_ylim(1e-6, 1.0)
    ax_u.set_xlim(1e-6, 1.0)
    ax_r.set_box_aspect(1)
    ax_u.set_box_aspect(1)
    style(ax_r)
    style(ax_u)
    ax_r.legend(fontsize=18, frameon=False)
    ax_u.legend(fontsize=18, frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", dpi=300, pad_inches=0.1)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output, bbox_inches="tight", dpi=300, pad_inches=0.1)
    plt.close(fig)


def parse_run(spec: str) -> tuple[str, Path, bool, float | None]:
    name, path, kind, *rest = spec.split(":")
    inertia = kind == "inertia"
    oh = float(rest[0]) if rest else None
    return name, Path(path), inertia, oh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="name:neck.csv:stokes  or  name:neck.csv:inertia:0.6",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    series = []
    for spec in args.run:
        name, path, inertia, oh = parse_run(spec)
        series.append((name, fig3_series(read_neck(path), inertia=inertia, oh=oh)))
    plot_fig3(series, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
