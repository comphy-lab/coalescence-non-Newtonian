#!/usr/bin/env python3
"""Coalescence video with the neck history: the nested-zoom row above, the neck plots below.

Row 1 is the three-panel hybrid of ``render_hybrid_video.py`` (the pair, the neck and the
meniscus as nested windows centred on the tip), drawn by the same routine. Row 2 shows
the neck history of the same run:

* **(d)** R_min against tau_v = t + t_con, with the published Stokes curve
  R_min = -(tau_v/pi) ln tau_v, as in Anthony, Harris & Basaran (2020), Figure 3(a);
* **(e)** u_v against R_min, with u_v = -(1/pi) ln R_min, as in their Figure 3(b);
* **(f)** u_v against R_min for the runs of a family of initial radii R0.

In (d) and (e), and for the family member with the run's own R0 in (f), the computed curve
grows with the video: each frame draws the history up to its own time (in (f) the whole
history of the run and its continuations, like (e)) and marks the current state. The other
family members are static. t_con follows the paper's procedure (a power law
fitted once R_min has grown by one decade, 10 R0 to 100 R0, extrapolated to zero radius;
``coalescence.analysis.contact_time``), and no quantity is fitted to the published data.
The digitised published series are drawn as markers when ``--anthony`` is given. The
history of a restart continuation replaces its parent's rows from the restart time on.

The clock, the colour limits and the protocol are those of ``render_hybrid_video.py``.

    python postProcess/render_case_dashboard_video.py <fields> [<fields> ...] \\
        --runtimes <runtime> [<continuation runtime> ...] --out dashboard.mp4 \\
        [--anthony validationCases/anthony2020/data] \\
        [--family <run list of the initial-radius family>]
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402  (backend and fonts set by render_neck_video)
import numpy as np  # noqa: E402

import render_drop_pair_video as pair  # noqa: E402
import render_hybrid_video as hybrid  # noqa: E402
from coalescence.analysis.contact_time import power_law_contact_time  # noqa: E402
from coalescence.analysis.runs import load_runs, load_series  # noqa: E402
from render_neck_video import FS_LABEL, FS_TICK, Frame, load_frame, png_size  # noqa: E402

BLUE, GREY, LIGHT = "#1f4e79", "0.15", "#c07a2c"
FS_LEGEND = 12


# ------------------------------------------------------------------- data

def read_neck(runtime: Path) -> dict[str, np.ndarray]:
    with (runtime / "neck.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("t", "R_min", "u_neck")}


def history(runtimes: list[Path]) -> dict[str, np.ndarray]:
    """Neck history of a run and its restart continuations, in time order."""
    out = read_neck(runtimes[0])
    for rt in runtimes[1:]:
        restart = json.loads((rt / "run-manifest.json").read_text()).get("restart_from")
        if not restart:
            raise SystemExit(f"{rt} is not a restart continuation")
        t_r = float(restart["t"])
        child = read_neck(rt)
        keep_parent, keep_child = out["t"] <= t_r, child["t"] > t_r
        out = {k: np.concatenate([out[k][keep_parent], child[k][keep_child]]) for k in out}
    if np.any(np.diff(out["t"]) <= 0):
        raise SystemExit("the spliced history does not advance in time")
    return out


def published(path: Path, xcol: str, ycol: str) -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    with path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["series"], []).append((float(r[xcol]), float(r[ycol])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


@dataclass(frozen=True)
class Plots:
    t: np.ndarray
    R: np.ndarray
    u: np.ndarray
    t_con: float
    radius: dict[str, np.ndarray] | None
    velocity: dict[str, np.ndarray] | None
    family: tuple[tuple[float, np.ndarray, np.ndarray], ...]


def initial_radius(h: dict[str, np.ndarray]) -> float:
    """R0 of the case: the FEM runner writes the initial bridge as the t = 0 row of neck.csv.

    A history without that row (a continuation given first, or a solver that records only
    after its first step) would shift the 10 R0 to 100 R0 fit window, so it is refused.
    """
    if h["t"][0] != 0.0:
        raise SystemExit("the neck history must start with the initial bridge at t = 0 (pass the root run first)")
    return float(h["R_min"][0])


def load_plots(runtimes: list[Path], anthony: Path | None, family: Path | None) -> Plots:
    h = history(runtimes)
    t_con = power_law_contact_time(h["t"], h["R_min"], initial_radius(h)).t_con
    radius = velocity = None
    if anthony is not None:
        radius = published(anthony / "anthony2020-fig3-radius-digitized.csv", "tau_v", "R_min")
        velocity = published(anthony / "anthony2020-fig3-velocity-digitized.csv", "R_min", "u_v")
    fam = []
    if family is not None:
        # As in the family figure (plot_startup_family.py): rows produced by solver steps only,
        # so the t = 0 placeholder of an unseeded run (u = 0) is not drawn.
        runs = load_runs(family)
        for run in sorted(runs, key=lambda r: r.R0):
            d = load_series(runs, run)
            fam.append((float(run.R0), d["R_min"], d["u_neck"]))
    return Plots(h["t"], h["R_min"], h["u_neck"], float(t_con), radius, velocity, tuple(fam))


# ----------------------------------------------------------------- layout

@dataclass(frozen=True)
class Dashboard:
    row1: hybrid.Layout
    row2: tuple[tuple[float, float, float, float], ...]


def make_dashboard(limits, dpi: int, panel_h: float = 5.2, plot_h: float = 4.6, plot_w: float = 5.4) -> Dashboard:
    widths = (panel_h * pair.R_LIM / pair.Z_LIM, panel_h, panel_h)
    left, cbar, gap = 0.95, 1.95, 0.45
    bottom, row_gap, top = 0.95, 1.35, 1.15
    fig_w = left + sum(widths) + 3 * cbar + 2 * gap
    fig_h = bottom + plot_h + row_gap + panel_h + top
    x0, rects1, rects2 = left, [], []
    for w in widths:
        rects1.append((x0 / fig_w, (bottom + plot_h + row_gap) / fig_h, w / fig_w, panel_h / fig_h))
        rects2.append((x0 / fig_w, bottom / fig_h, plot_w / fig_w, plot_h / fig_h))
        x0 += w + cbar + gap
    return Dashboard(hybrid.Layout(fig_w, fig_h, tuple(rects1), dpi, limits), tuple(rects2))


def style(ax, xlabel: str, ylabel: str, tag: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=3)
    ax.tick_params(which="both", direction="out", width=1.2, labelsize=FS_TICK, pad=4)
    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_LABEL)


def draw_history(fig, rects, fr: Frame, p: Plots) -> None:
    now = p.t <= fr.t * (1 + 1e-12) + 1e-300
    tau = p.t + p.t_con

    # (d) R_min(tau_v)
    ax = fig.add_axes(rects[0])
    tt = np.logspace(-8, np.log10(0.25), 300)
    ax.plot(tt, -(tt / np.pi) * np.log(tt), "k-", lw=1.3, label=r"$R_{\min}=-(\tau_v/\pi)\ln\tau_v$")
    if p.radius is not None:
        ax.plot(*p.radius["fig3_radius_oh0p6"].T, "^", ms=4, mfc="none", mec=LIGHT, mew=1.0,
                label=r"Anthony et al., $\mathrm{Oh}=0.6$")
        ax.plot(*p.radius["fig3_radius_stokes"].T, "o", ms=4, mfc="none", mec=GREY, mew=1.0,
                label=r"Anthony et al., Stokes")
    ax.plot(tau[now], p.R[now], color=BLUE, lw=2.4, label=r"present, Stokes")
    ax.plot([fr.t + p.t_con], [fr.R_min], "o", ms=9, color=BLUE, mec="white", mew=1.5, zorder=8)
    ax.set(xscale="log", yscale="log", xlim=(1e-8, 2.0), ylim=(5e-7, 1.5))
    style(ax, r"$\tau_v$", r"$R_{\min}$", "(d)")
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="lower right")

    # (e) u_v(R_min)
    ax = fig.add_axes(rects[1])
    rr = np.logspace(-6, 0, 300)
    ax.plot(rr, -np.log(rr) / np.pi, "k-", lw=1.3, label=r"$u_v=-(1/\pi)\ln R_{\min}$")
    if p.velocity is not None:
        ax.plot(*p.velocity["fig3_velocity_oh0p6"].T, "^", ms=4, mfc="none", mec=LIGHT, mew=1.0,
                label=r"Anthony et al., $\mathrm{Oh}=0.6$")
        ax.plot(*p.velocity["fig3_velocity_stokes"].T, "o", ms=4, mfc="none", mec=GREY, mew=1.0,
                label=r"Anthony et al., Stokes")
    ax.plot(p.R[now], p.u[now], color=BLUE, lw=2.4, label=r"present, Stokes")
    ax.plot([fr.R_min], [fr.U], "o", ms=9, color=BLUE, mec="white", mew=1.5, zorder=8)
    ax.set(xscale="log", xlim=(7e-7, 1.3), ylim=(0.0, 5.3))
    style(ax, r"$R_{\min}$", r"$u_v$", "(e)")
    ax.legend(frameon=False, fontsize=FS_LEGEND, loc="upper right")

    # (f) u_v(R_min) for the family of initial radii
    ax = fig.add_axes(rects[2])
    ax.plot(rr, -np.log(rr) / np.pi, "k-", lw=1.3, label=r"$u_v=-(1/\pi)\ln R_{\min}$")
    colours = plt.cm.viridis(np.linspace(0.0, 0.88, max(len(p.family), 1)))
    R0_run = float(p.R[0])
    for (R0, R, u), c in zip(p.family, colours):
        m, e = f"{R0:.2e}".split("e")
        label = rf"$R_0 = 10^{{{int(e)}}}$" if float(m) == 1.0 else rf"$R_0 = {float(m):.3g}\times10^{{{int(e)}}}$"
        if abs(R0 / R0_run - 1.0) < 1e-9:
            # This run: its whole history, growing with the video; from the first computed step,
            # as for the rest of the family.
            grow = now.copy()
            grow[0] = False
            # Drawn above the static family so its growth stays visible where they collapse.
            ax.plot(p.R[grow], p.u[grow], color=c, lw=2.2, label=label, zorder=6)
            ax.plot([fr.R_min], [fr.U], "o", ms=8, color=c, mec="white", mew=1.4, zorder=8)
        else:
            ax.plot(R, u, color=c, lw=1.8, label=label)
    ax.set(xscale="log", xlim=(7e-7, 1.3), ylim=(0.0, 5.3))
    style(ax, r"$R_{\min}$", r"$u_v$", "(f)")
    ax.legend(frameon=False, fontsize=FS_LEGEND - 1, loc="upper right", ncol=2, columnspacing=0.8,
              handlelength=1.4)


def draw_frame(fr: Frame, dash: Dashboard, p: Plots, dest: Path) -> Path:
    lay = dash.row1
    fig = plt.figure(figsize=(lay.fig_w, lay.fig_h))
    hybrid.draw_panels(fig, fr, lay)
    draw_history(fig, dash.row2, fr, p)
    hybrid.draw_stamp(fig, fr, lay.fig_h)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=lay.dpi)
    plt.close(fig)
    return dest


def _render_one(item: tuple[int, str], dash: Dashboard, p: Plots, outdir: str) -> str | None:
    index, path = item
    try:
        return str(draw_frame(load_frame(index, Path(path)), dash, p, Path(outdir) / f"frame-{index:05d}.png"))
    except Exception as exc:  # noqa: BLE001 - one bad frame is counted, not fatal to the pool
        print(f"  frame {index} failed: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fields", type=Path, nargs="+", help="frame folders in time order")
    ap.add_argument("--runtimes", type=Path, nargs="+", required=True, help="runtime folders: run, then continuations")
    ap.add_argument("--anthony", type=Path, default=None, help="folder of the digitised Anthony et al. series")
    ap.add_argument("--family", type=Path, default=None, help="run list of the initial-radius family")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stills", type=str, default="0.5")
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--t-switch", type=float, default=0.05)
    ap.add_argument("--hold-first", type=float, default=0.6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args(argv)
    files = hybrid.collect(a.fields)
    times = np.array([float(np.load(f)["t"]) for f in files])
    if times[0] != 0.0 or np.any(np.diff(times) <= 0):
        raise SystemExit("frames must start at t = 0 and advance in time")
    if not a.stills_only:
        hybrid.check_playback(times, a.duration, a.hold_first, a.fps)
    plots = load_plots([r.resolve() for r in a.runtimes], a.anthony, a.family)
    if plots.t[-1] < times[-1] * (1 - 1e-9):
        raise SystemExit("the neck history ends before the last frame")
    print(f"{len(files)} frames; history {plots.t.size} rows to t = {plots.t[-1]:.4g}; t_con = {plots.t_con:.4e}; "
          f"family {len(plots.family)} runs", flush=True)
    limits = hybrid.colour_limits(files)
    dash = make_dashboard(limits, a.dpi)
    for frac in (float(v) for v in a.stills.split(",") if v.strip()):
        i = min(len(files) - 1, int(round(frac * (len(files) - 1))))
        print(f"wrote {draw_frame(load_frame(i, files[i]), dash, plots, a.out.parent / f'{a.out.stem}-still-{i:04d}.png')}",
              flush=True)
    if a.stills_only:
        return 0
    frames_dir = a.out.parent / f"{a.out.stem}-frames"
    if frames_dir.is_dir():
        for old in frames_dir.glob("frame-*.png"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    items = [(i, str(f)) for i, f in enumerate(files)]
    with Pool(a.workers) as pool:
        made = [Path(m) for m in pool.map(partial(_render_one, dash=dash, p=plots, outdir=str(frames_dir)), items) if m]
    print(f"rendered {len(made)} of {len(items)} frames", flush=True)
    if len(made) != len(items):
        raise SystemExit("frame numbering would have gaps; ffmpeg would silently truncate the video")
    sizes = {png_size(f) for f in made}
    if len(sizes) != 1:
        raise SystemExit(f"frame sizes are not uniform: {sorted(sizes)}")
    seq = hybrid.playback_sequence(times, a.t_switch, a.duration, a.fps, a.hold_first)
    seq_dir = a.out.parent / f"{a.out.stem}-sequence"
    hybrid.link_sequence(seq, frames_dir, seq_dir)
    encode = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
              "-i", str(seq_dir / "seq-%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
              "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(a.out)]
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH; encode with:\n  " + " ".join(encode), file=sys.stderr)
        return 2
    if subprocess.run(encode, check=False).returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB, {len(seq)} frames at {a.fps} fps "
          f"= {len(seq) / a.fps:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
