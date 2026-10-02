#!/usr/bin/env python3
"""Animate the whole coalescing drop pair at a fixed physical scale: speed left of the axis, dissipation right.

Input is the frame folder written by ``reconstruct_stokes_fields.py``. This is the late-time
companion of ``render_neck_video.py``: once the neck radius is a sizeable fraction of the
drop radius, a window that follows R_min no longer isolates anything, and the evolution is
best seen in one fixed window holding both drops.

What is drawn
-------------
The computed quadrant (upper drop, z > 0, r >= 0) is mirrored in the symmetry plane z = 0
to give the lower drop and in the axis r = 0 to give the full cross-section:

* **left of the axis** the speed |u| (laboratory frame);
* **right of the axis** log10 Phi, Phi = 2 mu E:E the viscous dissipation rate per unit
  volume, computed from the P2 velocity gradient with the hoop strain (see
  ``render_neck_video.py``, whose routines are used here).

Lengths in the drop radius, velocity in gamma/mu, time in mu R/gamma. The protocol is that of
``render_neck_video.py``: fixed window, ticks, figure size and colour limits (area-weighted
samples of a spread of frames), numeric gapless frame order, uniform frame size before
encoding, mathtext, and no text on the frames beyond labels and the (t, R_min) stamp.

    python postProcess/render_drop_pair_video.py <fields folder> --out pair.mp4
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
from matplotlib.tri import LinearTriInterpolator, Triangulation  # noqa: E402

from render_neck_video import (  # noqa: E402
    _SUBTRIS, CMAP_DISS, CMAP_SPEED, DECADES, FS_CBAR_LABEL, FS_CBAR_TICK, FS_LABEL, FS_TICK, FS_TITLE,
    Frame, interface_chain, load_frame, png_size, read_series, sci, strain_fields,
)

R_LIM = 1.6
Z_LIM = 2.1


def quadrant_triangulation(fr: Frame, sx: float, sz: float) -> Triangulation:
    """Sub-triangulation of the computed quadrant mapped to (sx r, sz z); sub-pixel slivers dropped."""
    r = fr.points[:, 0] + fr.shift
    z = fr.points[:, 1]
    tris = fr.cells[:, _SUBTRIS].reshape(-1, 3)
    x0, y0 = r[tris[:, 0]], z[tris[:, 0]]
    area = 0.5 * np.abs((r[tris[:, 1]] - x0) * (z[tris[:, 2]] - y0) - (r[tris[:, 2]] - x0) * (z[tris[:, 1]] - y0))
    return Triangulation(sx * r, sz * z, tris[area > (1e-9 * R_LIM) ** 2])


def log_dissipation(fr: Frame) -> np.ndarray:
    return np.log10(np.maximum(strain_fields(fr)["phi"], 1e-300))


def speed(fr: Frame) -> np.ndarray:
    return np.hypot(fr.velocity[:, 0], fr.velocity[:, 1])


@dataclass(frozen=True)
class Layout:
    fig_w: float
    fig_h: float
    rect: tuple[float, float, float, float]
    dpi: int
    vmax: float
    dmin: float
    dmax: float


def make_layout(limits: dict[str, float], dpi: int, panel_h_in: float = 7.0) -> Layout:
    panel_w_in = panel_h_in * R_LIM / Z_LIM
    left_in, right_in, bottom_in, top_in = 1.0, 1.9, 0.95, 0.85
    fig_w, fig_h = left_in + panel_w_in + right_in, bottom_in + panel_h_in + top_in
    rect = (left_in / fig_w, bottom_in / fig_h, panel_w_in / fig_w, panel_h_in / fig_h)
    return Layout(fig_w, fig_h, rect, dpi, limits["vmax"], limits["dmin"], limits["dmax"])


def draw_frame(fr: Frame, lay: Layout, dest: Path) -> Path:
    fig = plt.figure(figsize=(lay.fig_w, lay.fig_h))
    ax = fig.add_axes(lay.rect)
    sp, ld = speed(fr), log_dissipation(fr)
    for sz in (1.0, -1.0):
        ax.tripcolor(quadrant_triangulation(fr, -1.0, sz), sp, shading="gouraud", cmap=CMAP_SPEED,
                     vmin=0.0, vmax=lay.vmax, rasterized=True)
        ax.tripcolor(quadrant_triangulation(fr, 1.0, sz), ld, shading="gouraud", cmap=CMAP_DISS,
                     vmin=lay.dmin, vmax=lay.dmax, rasterized=True)
    chain = interface_chain(fr)
    r = fr.points[chain, 0] + fr.shift
    z = fr.points[chain, 1]
    for sx in (1.0, -1.0):
        for sz in (1.0, -1.0):
            ax.plot(sx * r, sz * z, color="black", lw=1.6, zorder=6)
    ax.axvline(0.0, color="grey", lw=0.7, ls=(0, (6, 6)), zorder=5)
    ax.set_xlim(-R_LIM, R_LIM)
    ax.set_ylim(-Z_LIM, Z_LIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(-1.5, 1.5 + 1e-9, 0.5))
    ax.set_yticks(np.arange(-2.0, 2.0 + 1e-9, 1.0))
    ax.set_xlabel(r"$r$", fontsize=FS_LABEL, labelpad=4)
    ax.set_ylabel(r"$z$", fontsize=FS_LABEL, labelpad=4)
    ax.tick_params(which="both", direction="out", width=1.2, labelsize=FS_TICK, pad=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.text(0.5, 1.0 - 0.40 / lay.fig_h, rf"$t = {fr.t:.3f}$,   $R_{{\min}} = {fr.R_min:.3f}$",
             ha="center", va="center", fontsize=FS_TITLE)
    x0, y0, w, h = lay.rect
    cw = 0.20 / lay.fig_w
    cax_u = fig.add_axes([x0 + w + 0.22 / lay.fig_w, y0 + 0.52 * h, cw, 0.46 * h])
    cax_l = fig.add_axes([x0 + w + 0.22 / lay.fig_w, y0 + 0.02 * h, cw, 0.46 * h])
    cb_u = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_SPEED, norm=plt.Normalize(0.0, lay.vmax)), cax=cax_u)
    cb_l = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_DISS, norm=plt.Normalize(lay.dmin, lay.dmax)), cax=cax_l)
    cb_u.set_label(r"$|\mathbf{u}|$  (left)", fontsize=FS_CBAR_LABEL, labelpad=6)
    cb_l.set_label(r"$\log_{10}\Phi$  (right)", fontsize=FS_CBAR_LABEL, labelpad=6)
    for cb in (cb_u, cb_l):
        cb.ax.tick_params(labelsize=FS_CBAR_TICK)
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


def colour_limits(files: list[Path], n_probe: int = 12) -> dict[str, float]:
    """99.8th percentiles of area-weighted samples (uniform grid on the quadrant) of a spread of frames."""
    gx, gy = np.meshgrid(np.linspace(0.0, R_LIM, 161), np.linspace(0.0, Z_LIM, 211))
    speeds, highs = [], []
    for i in np.unique(np.linspace(0, len(files) - 1, min(n_probe, len(files))).astype(int)):
        fr = load_frame(int(i), files[int(i)])
        tri = quadrant_triangulation(fr, 1.0, 1.0)
        speeds.append(np.percentile(LinearTriInterpolator(tri, speed(fr))(gx, gy).compressed(), 99.8))
        highs.append(np.percentile(LinearTriInterpolator(tri, log_dissipation(fr))(gx, gy).compressed(), 99.8))
    dmax = float(np.ceil(2.0 * max(highs)) / 2.0)
    return {"vmax": float(np.ceil(10.0 * max(speeds)) / 10.0), "dmin": dmax - DECADES, "dmax": dmax}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fields", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args(argv)
    files = read_series(a.fields.resolve())
    limits = colour_limits(files)
    print(f"{len(files)} frames; colour limits |u| 0 -> {limits['vmax']:.3g}, log10 Phi {limits['dmin']:.3g} -> "
          f"{limits['dmax']:.3g}", flush=True)
    lay = make_layout(limits, a.dpi)
    for name, i in {"early": 0, "middle": len(files) // 2, "late": len(files) - 1}.items():
        print(f"wrote {draw_frame(load_frame(i, files[i]), lay, a.out.parent / f'{a.out.stem}-still-{name}.png')}",
              flush=True)
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
    encode = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
              "-i", str(frames_dir / "frame-%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
              "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(a.out)]
    if shutil.which("ffmpeg") is None:
        print("ffmpeg is not on PATH; encode with:\n  " + " ".join(encode), file=sys.stderr)
        return 2
    if subprocess.run(encode, check=False).returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB, {len(made)} frames at {a.fps} fps "
          f"= {len(made) / a.fps:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
