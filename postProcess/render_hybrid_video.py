#!/usr/bin/env python3
"""Hybrid coalescence video: the whole drop pair, the neck and the meniscus in one frame.

Input is one or more frame folders written by ``reconstruct_stokes_fields.py``, in time
order, for example the remesh states of a startup run followed by the uniform-time
snapshots of its continuation. Frames that do not advance the time are dropped, so a
restart state present in both folders is drawn once.

Three panels per frame share one (t, R_min, rho) stamp:

* **(a) the pair** at a fixed physical scale (as ``render_drop_pair_video.py``): the speed
  |u| left of the axis and log10 Phi right, both drops shown. A dashed box marks the window
  of (b) once it is larger than a pixel.
* **(b) the neck**, lengths over R_min (the ``neck`` view of ``render_neck_video.py``): |u|
  in the upper drop and log10(Phi R_min^2) in the lower one. A dashed box marks the window
  of (c) once it is larger than a pixel.
* **(c) the meniscus**, comoving with the tip: (r - R_min, z)/rho, the speed relative to the
  advancing neck |u - U e_r| above the plane and log10(Phi rho^2) below. When the window
  reaches past the axis (late times, rho comparable with R_min) the field is continued
  by its mirror image in the axis.

Playback follows a warped clock, s = ln t for t < t_s and s = ln t_s + (t - t_s)/t_s beyond
(continuous in value and slope): the startup is shown on a logarithmic clock, slowly, and
the late stage on a linear one. Each saved state is held on screen for its share of s
(sample and hold at a constant output rate); states are never interpolated or blended, and
the stamp always shows the physical time of the state drawn.

Phi = 2 mu E:E with the hoop strain, from the P2 velocity gradient in neck-frame
coordinates (``render_neck_video.py``). Visco-capillary units. The protocol is that of the
two single-scale renderers: fixed windows, ticks, figure size and colour limits
(area-weighted samples of a spread of frames), numeric gapless frame order, uniform frame
size before encoding, mathtext, and no text on the frames beyond labels, panel tags and the
stamp.

    python postProcess/render_hybrid_video.py <fields> [<fields> ...] --out hybrid.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402  (backend and fonts set by render_neck_video)
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.tri import LinearTriInterpolator, Triangulation  # noqa: E402

import render_drop_pair_video as pair  # noqa: E402
from render_neck_video import (  # noqa: E402
    _SUBTRIS, CMAP_DISS, CMAP_SPEED, DECADES, FS_CBAR_LABEL, FS_CBAR_TICK, FS_LABEL, FS_TICK, FS_TITLE,
    VIEWS, Frame, interface_chain, load_frame, png_size, read_series, sci, strain_fields,
)

NECK = next(v for v in VIEWS if v.name == "neck")
MEN_HALF = 6.0          # meniscus window half-width in tip radii
BOX_STYLE = dict(fill=False, ls=(0, (4, 3)), lw=1.4, ec="0.2", zorder=8)


def collect(folders: list[Path]) -> list[Path]:
    """Frame files of all folders in order, keeping only frames that advance the time."""
    out, last = [], -np.inf
    for folder in folders:
        for path in read_series(folder.resolve()):
            with np.load(path) as d:
                t = float(d["t"])
            if t > last * (1 + 1e-12) or (t == 0.0 and not out):
                out.append(path)
                last = t
    return out


def warped_clock(t: np.ndarray, t_switch: float) -> np.ndarray:
    """s(t) = ln t below t_switch, continued linearly with the same slope above it."""
    t = np.asarray(t, dtype=float)
    return np.where(t < t_switch, np.log(np.maximum(t, 1e-300)), np.log(t_switch) + (t - t_switch) / t_switch)


def playback_sequence(times: np.ndarray, t_switch: float, duration: float, fps: int, hold_first: float) -> np.ndarray:
    """Source frame shown at each output frame: sample and hold on the warped clock.

    Frame 0 (t = 0, where ln t is undefined) is held for ``hold_first`` seconds; the rest of
    the duration spans s(t_1) .. s(t_last) uniformly.
    """
    s = warped_clock(times[1:], t_switch)
    n_first = int(round(hold_first * fps))
    n_rest = int(round((duration - hold_first) * fps))
    clock = np.linspace(s[0], s[-1], n_rest)
    rest = 1 + np.searchsorted(s, clock, side="right") - 1
    return np.concatenate([np.zeros(n_first, dtype=int), np.clip(rest, 1, len(times) - 1)])


def fmt(v: float) -> str:
    return f"{v:.3f}" if 1e-2 <= abs(v) < 10 else ("0" if v == 0 else sci(v))


# -------------------------------------------------------------- meniscus panel

def meniscus_coords(fr: Frame, mirror_axis: bool) -> tuple[np.ndarray, np.ndarray]:
    """(r - R_min, z)/rho from the solver's X = r - R_neck; in the axis mirror r -> -r."""
    X = fr.points[:, 0]
    x = (-X - 2.0 * fr.shift) if mirror_axis else X
    return x / fr.rho, fr.points[:, 1] / fr.rho


def meniscus_relative_speed(fr: Frame, mirror_axis: bool) -> np.ndarray:
    u_r = -fr.velocity[:, 0] if mirror_axis else fr.velocity[:, 0]
    return np.hypot(u_r - fr.U, fr.velocity[:, 1])


def window_tris(x: np.ndarray, y: np.ndarray, fr: Frame, half: float, sign_y: float) -> Triangulation | None:
    tris = fr.cells[:, _SUBTRIS].reshape(-1, 3)
    pad = 0.05 * half
    near = np.any((np.abs(x[tris]) <= half + pad) & (y[tris] <= half + pad), axis=1)
    tris = tris[near]
    if not len(tris):
        return None
    x0, y0 = x[tris[:, 0]], y[tris[:, 0]]
    area = 0.5 * np.abs((x[tris[:, 1]] - x0) * (y[tris[:, 2]] - y0) - (x[tris[:, 2]] - x0) * (y[tris[:, 1]] - y0))
    tris = tris[area > (2e-9 * half) ** 2]
    return Triangulation(x, sign_y * y, tris) if len(tris) else None


def axis_in_window(fr: Frame) -> bool:
    return fr.R_min - MEN_HALF * fr.rho < 0.0


# ------------------------------------------------------------------- drawing

@dataclass(frozen=True)
class Layout:
    fig_w: float
    fig_h: float
    rects: tuple[tuple[float, float, float, float], ...]
    dpi: int
    limits: dict[str, dict[str, float]]


def make_layout(limits: dict[str, dict[str, float]], dpi: int, panel_h: float = 5.2) -> Layout:
    widths = (panel_h * pair.R_LIM / pair.Z_LIM, panel_h, panel_h)
    left, cbar, gap, bottom, top = 0.95, 1.95, 0.45, 0.95, 1.15
    fig_w = left + sum(widths) + 3 * cbar + 2 * gap
    fig_h = bottom + panel_h + top
    x0, rects = left, []
    for w in widths:
        rects.append((x0 / fig_w, bottom / fig_h, w / fig_w, panel_h / fig_h))
        x0 += w + cbar + gap
    return Layout(fig_w, fig_h, tuple(rects), dpi, limits)


def style_axes(ax, xlabel: str, ylabel: str, tag: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FS_LABEL, labelpad=4)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=4)
    ax.tick_params(which="both", direction="out", width=1.2, labelsize=FS_TICK, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_LABEL)


def colourbars(fig, lay: Layout, rect, lim: dict[str, float], upper_label: str, lower_label: str) -> None:
    x0, y0, w, h = rect
    cw, off = 0.20 / lay.fig_w, 0.22 / lay.fig_w
    cax_u = fig.add_axes([x0 + w + off, y0 + 0.52 * h, cw, 0.46 * h])
    cax_l = fig.add_axes([x0 + w + off, y0 + 0.02 * h, cw, 0.46 * h])
    cb_u = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_SPEED, norm=plt.Normalize(0.0, lim["vmax"])), cax=cax_u)
    cb_l = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_DISS, norm=plt.Normalize(lim["dmin"], lim["dmax"])), cax=cax_l)
    cb_u.set_label(upper_label, fontsize=FS_CBAR_LABEL, labelpad=6)
    cb_l.set_label(lower_label, fontsize=FS_CBAR_LABEL, labelpad=6)
    for cb in (cb_u, cb_l):
        cb.ax.tick_params(labelsize=FS_CBAR_TICK)


def draw_frame(fr: Frame, lay: Layout, dest: Path) -> Path:
    fig = plt.figure(figsize=(lay.fig_w, lay.fig_h))
    phi = strain_fields(fr)["phi"]
    log_phi = np.log10(np.maximum(phi, 1e-300))
    chain = interface_chain(fr)
    r_c = fr.points[chain, 0] + fr.shift
    z_c = fr.points[chain, 1]

    # (a) the pair, fixed physical scale
    lim = lay.limits["pair"]
    ax = fig.add_axes(lay.rects[0])
    sp = np.hypot(fr.velocity[:, 0], fr.velocity[:, 1])
    for sz in (1.0, -1.0):
        ax.tripcolor(pair.quadrant_triangulation(fr, -1.0, sz), sp, shading="gouraud", cmap=CMAP_SPEED,
                     vmin=0.0, vmax=lim["vmax"], rasterized=True)
        ax.tripcolor(pair.quadrant_triangulation(fr, 1.0, sz), log_phi, shading="gouraud", cmap=CMAP_DISS,
                     vmin=lim["dmin"], vmax=lim["dmax"], rasterized=True)
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            ax.plot(sx * r_c, sz * z_c, color="black", lw=1.4, zorder=6)
    ax.axvline(0.0, color="grey", lw=0.7, ls=(0, (6, 6)), zorder=5)
    if fr.R_min > 0.006 * pair.Z_LIM:
        ax.add_patch(Rectangle((0.0, -fr.R_min), 2.0 * fr.R_min, 2.0 * fr.R_min, **BOX_STYLE))
    ax.set_xlim(-pair.R_LIM, pair.R_LIM)
    ax.set_ylim(-pair.Z_LIM, pair.Z_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-1.5, 1.5 + 1e-9, 0.5))
    ax.set_xticklabels(["", "-1", "", "0", "", "1", ""])
    ax.set_yticks(np.arange(-2.0, 2.0 + 1e-9, 1.0))
    style_axes(ax, r"$r$", r"$z$", "(a)")
    colourbars(fig, lay, lay.rects[0], lim, r"$|\mathbf{u}|$  (left)", r"$\log_{10}\Phi$  (right)")

    # (b) the neck, lengths over R_min
    lim = lay.limits["neck"]
    ax = fig.add_axes(lay.rects[1])
    ax.tripcolor(NECK.triangulation(fr, mirror=False), NECK.upper(fr), shading="gouraud", cmap=CMAP_SPEED,
                 vmin=0.0, vmax=lim["vmax"], rasterized=True)
    ax.tripcolor(NECK.triangulation(fr, mirror=True), NECK.lower(fr, phi), shading="gouraud", cmap=CMAP_DISS,
                 vmin=lim["dmin"], vmax=lim["dmax"], rasterized=True)
    xb, yb = NECK.coords(fr)
    ax.plot(xb[chain], yb[chain], color="black", lw=1.6, zorder=6)
    ax.plot(xb[chain], -yb[chain], color="black", lw=1.6, zorder=6)
    ax.plot([0.0, 1.0], [0.0, 0.0], color="grey", lw=0.7, ls=(0, (6, 6)), zorder=5)
    half_b = MEN_HALF * fr.rho / fr.R_min
    if 0.006 * (NECK.xlim[1] - NECK.xlim[0]) < half_b < 1.0:      # visible and inside (b)
        ax.add_patch(Rectangle((1.0 - half_b, -half_b), 2.0 * half_b, 2.0 * half_b, **BOX_STYLE))
    ax.set_xlim(*NECK.xlim)
    ax.set_ylim(*NECK.ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(NECK.xlim[0], NECK.xlim[1] + 1e-9, NECK.tick))
    ax.set_yticks(np.arange(NECK.ylim[0], NECK.ylim[1] + 1e-9, NECK.tick))
    style_axes(ax, NECK.xlabel, NECK.ylabel, "(b)")
    colourbars(fig, lay, lay.rects[1], lim, NECK.upper_label, NECK.lower_label)

    # (c) the meniscus, comoving with the tip, lengths over rho
    lim = lay.limits["meniscus"]
    ax = fig.add_axes(lay.rects[2])
    log_phi_rho = np.log10(np.maximum(phi * fr.rho ** 2, 1e-300))
    mirrors = (False, True) if axis_in_window(fr) else (False,)
    for m in mirrors:
        x, y = meniscus_coords(fr, m)
        up, lo = window_tris(x, y, fr, MEN_HALF, 1.0), window_tris(x, y, fr, MEN_HALF, -1.0)
        if up is not None:
            ax.tripcolor(up, meniscus_relative_speed(fr, m), shading="gouraud", cmap=CMAP_SPEED,
                         vmin=0.0, vmax=lim["vmax"], rasterized=True)
        if lo is not None:
            ax.tripcolor(lo, log_phi_rho, shading="gouraud", cmap=CMAP_DISS,
                         vmin=lim["dmin"], vmax=lim["dmax"], rasterized=True)
        ax.plot(x[chain], y[chain], color="black", lw=1.6, zorder=6)
        ax.plot(x[chain], -y[chain], color="black", lw=1.6, zorder=6)
    x_axis = -fr.R_min / fr.rho
    ax.plot([max(-MEN_HALF, x_axis), 0.0], [0.0, 0.0], color="grey", lw=0.7, ls=(0, (6, 6)), zorder=5)
    if x_axis > -MEN_HALF:
        ax.axvline(x_axis, color="grey", lw=0.7, ls=(0, (6, 6)), zorder=5)
    ax.set_xlim(-MEN_HALF, MEN_HALF)
    ax.set_ylim(-MEN_HALF, MEN_HALF)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-MEN_HALF, MEN_HALF + 1e-9, 2.0))
    ax.set_yticks(np.arange(-MEN_HALF, MEN_HALF + 1e-9, 2.0))
    style_axes(ax, r"$(r-R_{\min})/\rho$", r"$z/\rho$", "(c)")
    colourbars(fig, lay, lay.rects[2], lim, r"$|\mathbf{u}-U\mathbf{e}_r|$", r"$\log_{10}(\Phi\rho^2)$")

    fig.text(0.5, 1.0 - 0.42 / lay.fig_h,
             rf"$t = {fmt(fr.t)}$,   $R_{{\min}} = {fmt(fr.R_min)}$,   $\rho = {fmt(fr.rho)}$",
             ha="center", va="center", fontsize=FS_TITLE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=lay.dpi)
    plt.close(fig)
    return dest


def _render_one(item: tuple[int, str], lay: Layout, outdir: str) -> str | None:
    index, path = item
    try:
        return str(draw_frame(load_frame(index, Path(path)), lay, Path(outdir) / f"frame-{index:05d}.png"))
    except Exception as exc:  # noqa: BLE001 - one bad frame is counted, not fatal to the pool
        print(f"  frame {index} failed: {exc}", file=sys.stderr)
        return None


# ------------------------------------------------------------ colour limits

def colour_limits(files: list[Path], n_probe: int = 16) -> dict[str, dict[str, float]]:
    """Per panel: 99.8th percentiles of area-weighted samples (uniform grids over the upper half of
    each window) of a spread of frames; dissipation over DECADES below its maximum."""
    probes = [load_frame(int(i), files[int(i)])
              for i in np.unique(np.linspace(0, len(files) - 1, min(n_probe, len(files))).astype(int))]
    grids = {
        "pair": np.meshgrid(np.linspace(0.0, pair.R_LIM, 161), np.linspace(0.0, pair.Z_LIM, 211)),
        "neck": np.meshgrid(np.linspace(*NECK.xlim, 241), np.linspace(0.0, NECK.ylim[1], 121)),
        "meniscus": np.meshgrid(np.linspace(-MEN_HALF, MEN_HALF, 241), np.linspace(0.0, MEN_HALF, 121)),
    }
    samples = {k: ([], []) for k in grids}
    for fr in probes:
        phi = strain_fields(fr)["phi"]
        fields = {
            "pair": (pair.quadrant_triangulation(fr, 1.0, 1.0), np.hypot(*fr.velocity.T), np.log10(np.maximum(phi, 1e-300))),
            "neck": (NECK.triangulation(fr, mirror=False), NECK.upper(fr), NECK.lower(fr, phi)),
        }
        x, y = meniscus_coords(fr, False)
        tri = window_tris(x, y, fr, MEN_HALF, 1.0)
        if tri is not None:
            fields["meniscus"] = (tri, meniscus_relative_speed(fr, False), np.log10(np.maximum(phi * fr.rho ** 2, 1e-300)))
        for k, (tri, up, lo) in fields.items():
            gx, gy = grids[k]
            su = LinearTriInterpolator(tri, up)(gx, gy).compressed()
            sl = LinearTriInterpolator(tri, lo)(gx, gy).compressed()
            if su.size:
                samples[k][0].append(np.percentile(su, 99.8))
                samples[k][1].append(np.percentile(sl, 99.8))
    out = {}
    for k, (speeds, highs) in samples.items():
        dmax = float(np.ceil(2.0 * max(highs)) / 2.0)
        out[k] = {"vmax": float(np.ceil(10.0 * max(speeds)) / 10.0), "dmin": dmax - DECADES, "dmax": dmax}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fields", type=Path, nargs="+", help="frame folders in time order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stills", type=str, default="0.5", help="comma list of frame fractions for stills")
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--fps", type=int, default=30, help="output frame rate")
    ap.add_argument("--duration", type=float, default=32.0, help="video length in seconds")
    ap.add_argument("--t-switch", type=float, default=0.05, help="time at which the clock turns from logarithmic to linear")
    ap.add_argument("--hold-first", type=float, default=0.6, help="seconds the t = 0 frame is shown")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args(argv)
    files = collect(a.fields)
    times = np.array([float(np.load(f)["t"]) for f in files])
    if times[0] != 0.0 or np.any(np.diff(times) <= 0):
        raise SystemExit("frames must start at t = 0 and advance in time")
    limits = colour_limits(files)
    print(f"{len(files)} frames from {len(a.fields)} folders", flush=True)
    for k, lim in limits.items():
        print(f"  colour limits {k}: speed 0 -> {lim['vmax']:.3g}; log dissipation {lim['dmin']:.3g} -> {lim['dmax']:.3g}",
              flush=True)
    lay = make_layout(limits, a.dpi)
    for frac in (float(v) for v in a.stills.split(",") if v.strip()):
        i = min(len(files) - 1, int(round(frac * (len(files) - 1))))
        dest = a.out.parent / f"{a.out.stem}-still-{i:04d}.png"
        print(f"wrote {draw_frame(load_frame(i, files[i]), lay, dest)}", flush=True)
    if a.stills_only:
        return 0
    frames_dir = a.out.parent / f"{a.out.stem}-frames"
    if frames_dir.is_dir():
        for old in frames_dir.glob("frame-*.png"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    items = [(i, str(p)) for i, p in enumerate(files)]
    with Pool(a.workers) as pool:
        made = [Path(m) for m in pool.map(partial(_render_one, lay=lay, outdir=str(frames_dir)), items) if m]
    print(f"rendered {len(made)} of {len(items)} frames", flush=True)
    if len(made) != len(items):
        raise SystemExit("frame numbering would have gaps; ffmpeg would silently truncate the video")
    sizes = {png_size(p) for p in made}
    if len(sizes) != 1:
        raise SystemExit(f"frame sizes are not uniform: {sorted(sizes)}")
    # Playback on the warped clock: a gapless numbered sequence of links to the held frames.
    seq = playback_sequence(times, a.t_switch, a.duration, a.fps, a.hold_first)
    seq_dir = a.out.parent / f"{a.out.stem}-sequence"
    if seq_dir.is_dir():
        for old in seq_dir.glob("seq-*.png"):
            old.unlink()
    seq_dir.mkdir(parents=True, exist_ok=True)
    for j, k in enumerate(seq):
        (seq_dir / f"seq-{j:05d}.png").symlink_to(frames_dir / f"frame-{int(k):05d}.png")
    held = np.bincount(seq, minlength=len(files)) / a.fps
    shown = np.count_nonzero(held)
    print(f"playback: {len(seq)} output frames; {shown} of {len(files)} states shown; longest hold "
          f"{held.max():.2f} s (state {int(held.argmax())}, t = {times[int(held.argmax())]:.3e})", flush=True)
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
