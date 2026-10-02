#!/usr/bin/env python3
"""Animate the neck of a coalescing drop pair: speed above the symmetry plane, dissipation below.

Input is the frame folder written by ``reconstruct_stokes_fields.py`` (one ``.npz`` per
saved state: the P2 mesh in the neck frame, the velocity, the neck radius, the neck speed
and the tip radius of curvature).

What is drawn
-------------
The computed quadrant is the upper drop, z > 0, r >= 0. The lower drop is its mirror image
in the symmetry plane z = 0, so each panel shows both drops with a different field in each,
with equal scales in r and z:

* **(a) the neck**, lengths scaled by the instantaneous neck radius R_min: the speed |u|
  (laboratory frame) in the upper drop and log10(Phi R_min^2) in the lower one, Phi = 2 mu
  E:E the viscous dissipation rate per unit volume and E the rate-of-strain tensor. Velocity
  is O(1) in these units and gradients scale as u/R_min, hence the factor R_min^2. The gap
  between the drops, of thickness ~R_min^2, is thinner than a pixel at small R_min and shows
  as the interface line along z = 0 outside the neck.
* **(b) the meniscus**, lengths measured from the neck, (r - R_min, z), and scaled by the
  tip radius of curvature rho: the speed relative to the advancing neck, |u - U e_r| with
  U = dR_min/dt the neck speed (the tip region translates almost rigidly at U), and
  log10(Phi rho^2).

Both windows follow the neck as it opens over several decades; nothing is drawn at a fixed
physical scale.

Dissipation
-----------
Phi is computed from the P2 velocity field, not taken from the solver. For each element the
isoparametric Jacobian is evaluated at the six nodal parametric points and inverted, giving
the exact elementwise gradient of the quadratic field; contributions are averaged at shared
nodes. With mu = 1:

    Phi = 2 ( E_rr^2 + E_zz^2 + E_tt^2 + 2 E_rz^2 ),   E_tt = u_r / r  (hoop strain).

Gradients are evaluated in the solver's neck-frame coordinates (X = r - R_neck, z): near
the tip the elements are many decades smaller than R_min and would be lost to round-off in
r. The degeneracy test is relative to each element's size for the same reason. The
reconstruction is checked rather than trusted: rms(div u)/rms(|grad u|) from the same
nodal gradients is reported for a spread of frames (zero for the exact field).

Protocol
--------
Fixed windows, ticks, figure size and colour limits for the whole sequence (limits from
area-weighted samples, a uniform grid over each window, of a spread of frames, so the
tip's node cluster does not set them); numeric frame order; gapless frame numbering;
uniform PNG size checked before encoding; mathtext rather than usetex, because frames
render in a process pool. No explanatory text is drawn on the frames: only axis and
colourbar labels and the (R_min, rho, t) stamp.

    python postProcess/render_neck_video.py <fields folder> --out neck.mp4 [--stills-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"

CMAP_SPEED = "Blues"
CMAP_DISS = "hot_r"

FS_TITLE = 22
FS_LABEL = 19
FS_TICK = 14
FS_CBAR_LABEL = 17
FS_CBAR_TICK = 12

DECADES = 5.0
"""Range of the log-dissipation colour scale."""


@dataclass(frozen=True)
class Frame:
    index: int
    t: float
    R_min: float
    shift: float          # neck radius: origin of the solver's X = r - shift
    rho: float            # tip radius of curvature
    U: float              # neck speed dR_min/dt
    points: np.ndarray    # (N, 2) as (X, z)
    cells: np.ndarray     # (M, 6) triangle6, VTK order
    velocity: np.ndarray  # (N, 2) as (u_r, u_z), laboratory frame


def read_series(folder: Path) -> list[Path]:
    """Frame files in numeric order of the frame number (never lexicographic), without gaps."""
    files = sorted(folder.glob("frame-*.npz"), key=lambda p: int(p.stem.split("-")[-1]))
    if not files:
        raise SystemExit(f"no frame-*.npz under {folder}")
    numbers = [int(p.stem.split("-")[-1]) for p in files]
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
        missing = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
        raise SystemExit(f"frame numbering has gaps: {missing[:10]}")
    return files


def load_frame(index: int, path: Path) -> Frame:
    with np.load(path) as d:
        return Frame(index=index, t=float(d["t"]), R_min=float(d["R_min"]), shift=float(d["shift"]),
                     rho=float(d["tip_radius"]), U=float(d["u_neck"]),
                     points=np.asarray(d["points"], float), cells=np.asarray(d["cells"], np.int64),
                     velocity=np.asarray(d["velocity"], float))


# ------------------------------------------------------------------- P2 fields

# VTK quadratic triangle: corners 0,1,2 then midsides 3=(0,1), 4=(1,2), 5=(2,0).
_NODE_XI = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
_SUBTRIS = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])


def _dshape(xi: np.ndarray) -> np.ndarray:
    """``(6, 2)`` derivatives dN_i/dxi_a of the six P2 shape functions at one point."""
    x, y = float(xi[0]), float(xi[1])
    l0 = 1.0 - x - y
    return np.array([
        [-(4.0 * l0 - 1.0), -(4.0 * l0 - 1.0)],
        [4.0 * x - 1.0, 0.0],
        [0.0, 4.0 * y - 1.0],
        [4.0 * (l0 - x), -4.0 * x],
        [4.0 * y, 4.0 * x],
        [-4.0 * y, 4.0 * (l0 - y)],
    ])


def nodal_velocity_gradient(fr: Frame) -> np.ndarray:
    """Nodal grad[n, b, c] = du_c/dx_b (x_0 = r, x_1 = z), averaged over the elements at a node."""
    coords = fr.points[fr.cells]
    uvals = fr.velocity[fr.cells]
    # Element size squared, for a degeneracy test that is relative, not absolute.
    edge = coords[:, [1, 2, 0], :] - coords[:, [0, 1, 2], :]
    size2 = np.max(np.sum(edge ** 2, axis=2), axis=1)
    acc = np.zeros((fr.points.shape[0], 2, 2))
    count = np.zeros(fr.points.shape[0])
    for k in range(6):
        dN = _dshape(_NODE_XI[k])
        jac = np.einsum("ia,eib->eab", dN, coords)
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        ok = np.abs(det) > 1e-10 * size2
        dudxi = np.einsum("ia,eic->eac", dN, uvals[ok])
        grad_e = np.einsum("eba,eac->ebc", np.linalg.inv(jac[ok]), dudxi)
        np.add.at(acc, fr.cells[ok, k], grad_e)
        np.add.at(count, fr.cells[ok, k], 1.0)
    count[count == 0] = np.nan
    return acc / count[:, None, None]


def strain_fields(fr: Frame) -> dict[str, np.ndarray]:
    """Nodal dissipation 2 E:E (mu = 1), the divergence and the gradient norm."""
    g = nodal_velocity_gradient(fr)
    r = fr.points[:, 0] + fr.shift
    u_r = fr.velocity[:, 0]
    e_rr, e_zz = g[:, 0, 0], g[:, 1, 1]
    e_rz = 0.5 * (g[:, 1, 0] + g[:, 0, 1])
    # u_r vanishes linearly on the axis, so u_r/r -> du_r/dr there.
    axis = r <= 1e-9 * fr.R_min
    e_tt = np.where(axis, e_rr, u_r / np.where(axis, 1.0, r))
    phi = 2.0 * (e_rr ** 2 + e_zz ** 2 + e_tt ** 2 + 2.0 * e_rz ** 2)
    return {"phi": phi, "div": e_rr + e_zz + e_tt,
            "grad_norm": np.sqrt(np.nansum(g ** 2, axis=(1, 2)) + e_tt ** 2)}


# ----------------------------------------------------------------------- views

@dataclass(frozen=True)
class View:
    """One panel: coordinates, window and the two fields, all scaled with the frame."""

    name: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    tick: float
    xlabel: str
    ylabel: str
    upper_label: str
    lower_label: str

    def length(self, fr: Frame) -> float:
        return fr.R_min if self.name == "neck" else fr.rho

    def coords(self, fr: Frame) -> tuple[np.ndarray, np.ndarray]:
        L = self.length(fr)
        # Both from the solver's X = r - R_neck, so the tip keeps its digits.
        x = (fr.points[:, 0] + fr.shift) / L if self.name == "neck" else fr.points[:, 0] / L
        return x, fr.points[:, 1] / L

    def upper(self, fr: Frame) -> np.ndarray:
        if self.name == "neck":
            return np.hypot(fr.velocity[:, 0], fr.velocity[:, 1])
        return np.hypot(fr.velocity[:, 0] - fr.U, fr.velocity[:, 1])

    def lower(self, fr: Frame, phi: np.ndarray) -> np.ndarray:
        return np.log10(np.maximum(phi * self.length(fr) ** 2, 1e-300))

    def triangulation(self, fr: Frame, mirror: bool) -> Triangulation:
        """Four straight sub-triangles per P2 element, kept only near the window.

        Sub-triangles smaller than 1e-9 of the window are dropped: they lie in the tip
        cluster, far below a pixel, and can be degenerate once the coordinates are scaled.
        """
        x, y = self.coords(fr)
        width = self.xlim[1] - self.xlim[0]
        pad = 0.05 * width
        tris = fr.cells[:, _SUBTRIS].reshape(-1, 3)
        near = np.any((x[tris] >= self.xlim[0] - pad) & (x[tris] <= self.xlim[1] + pad)
                      & (y[tris] <= self.ylim[1] + pad), axis=1)
        tris = tris[near]
        x0, y0 = x[tris[:, 0]], y[tris[:, 0]]
        area = 0.5 * np.abs((x[tris[:, 1]] - x0) * (y[tris[:, 2]] - y0)
                            - (x[tris[:, 2]] - x0) * (y[tris[:, 1]] - y0))
        tris = tris[area > (1e-9 * width) ** 2]
        return Triangulation(x, -y if mirror else y, tris)


VIEWS = (
    View("neck", (0.0, 2.0), (-1.0, 1.0), 0.5, r"$r/R_{\min}$", r"$z/R_{\min}$",
         r"$|\mathbf{u}|$", r"$\log_{10}(\Phi R_{\min}^2)$"),
    View("meniscus", (-6.0, 6.0), (-6.0, 6.0), 2.0, r"$(r-R_{\min})/\rho$", r"$z/\rho$",
         r"$|\mathbf{u}-U\mathbf{e}_r|$", r"$\log_{10}(\Phi\rho^2)$"),
)


def divergence_check(fr: Frame) -> dict[str, float]:
    """rms(div u)/rms(|grad u|) over all nodes and over the nodes in each view's window."""
    s = strain_fields(fr)
    ok = np.isfinite(s["div"])
    out = {"all": float(np.sqrt(np.mean(s["div"][ok] ** 2)) / np.sqrt(np.mean(s["grad_norm"][ok] ** 2)))}
    for v in VIEWS:
        x, y = v.coords(fr)
        m = ok & (x >= v.xlim[0]) & (x <= v.xlim[1]) & (y <= v.ylim[1])
        out[v.name] = float(np.sqrt(np.mean(s["div"][m] ** 2)) / np.sqrt(np.mean(s["grad_norm"][m] ** 2)))
    return out


def interface_chain(fr: Frame) -> np.ndarray:
    """Node indices of the free surface in order, walked from the neck.

    It is the mesh boundary minus the axis (r = 0) and the symmetry plane (z = 0); the
    neck node is the one interface corner on the plane.
    """
    local = ((0, 3, 1), (1, 4, 2), (2, 5, 0))
    seen: dict[tuple[int, int], list] = {}
    for a, m, b in local:
        for ea, em, eb in zip(fr.cells[:, a], fr.cells[:, m], fr.cells[:, b]):
            key = (min(ea, eb), max(ea, eb))
            if key in seen:
                seen[key][1] += 1
            else:
                seen[key] = [(int(ea), int(em), int(eb)), 1]
    r = fr.points[:, 0] + fr.shift
    z = fr.points[:, 1]
    adj: dict[int, list[tuple[int, int]]] = {}
    for (a, m, b), n in seen.values():
        if n != 1:
            continue
        if z[a] == 0.0 and z[b] == 0.0:
            continue                                          # symmetry plane
        if abs(r[a]) <= 1e-9 * fr.R_min and abs(r[b]) <= 1e-9 * fr.R_min:
            continue                                          # axis
        adj.setdefault(a, []).append((m, b))
        adj.setdefault(b, []).append((m, a))
    start = next(n for n in adj if z[n] == 0.0)
    order, visited, node = [start], {start}, start
    while True:
        nxt = [(m, b) for m, b in adj.get(node, []) if b not in visited]
        if not nxt:
            break
        m, b = nxt[0]
        order.extend([m, b])
        visited.add(b)
        node = b
    return np.array(order)


# --------------------------------------------------------------------- drawing

@dataclass(frozen=True)
class Layout:
    fig_w: float
    fig_h: float
    rects: tuple[tuple[float, float, float, float], ...]
    dpi: int
    limits: tuple[dict[str, float], ...]


def make_layout(limits: tuple[dict[str, float], ...], dpi: int, panel_in: float = 5.6) -> Layout:
    left_in, cbar_in, gap_in, bottom_in, top_in = 1.05, 1.95, 0.55, 0.95, 1.15
    fig_w = left_in + 2 * (panel_in + cbar_in) + gap_in
    fig_h = bottom_in + panel_in + top_in
    x0 = left_in
    rects = []
    for _ in VIEWS:
        rects.append((x0 / fig_w, bottom_in / fig_h, panel_in / fig_w, panel_in / fig_h))
        x0 += panel_in + cbar_in + gap_in
    return Layout(fig_w=fig_w, fig_h=fig_h, rects=tuple(rects), dpi=dpi, limits=limits)


def sci(v: float, digits: int = 2) -> str:
    m, e = f"{v:.{digits}e}".split("e")
    return rf"{m}\times 10^{{{int(e)}}}"


def draw_frame(fr: Frame, lay: Layout, dest: Path) -> Path:
    fig = plt.figure(figsize=(lay.fig_w, lay.fig_h))
    phi = strain_fields(fr)["phi"]
    chain = interface_chain(fr)
    for v, rect, lim, tag in zip(VIEWS, lay.rects, lay.limits, ("(a)", "(b)")):
        ax = fig.add_axes(rect)
        ax.tripcolor(v.triangulation(fr, mirror=False), v.upper(fr), shading="gouraud",
                     cmap=CMAP_SPEED, vmin=0.0, vmax=lim["vmax"], rasterized=True)
        ax.tripcolor(v.triangulation(fr, mirror=True), v.lower(fr, phi), shading="gouraud",
                     cmap=CMAP_DISS, vmin=lim["dmin"], vmax=lim["dmax"], rasterized=True)
        x, y = v.coords(fr)
        ax.plot(x[chain], y[chain], color="black", lw=1.6, zorder=6)
        ax.plot(x[chain], -y[chain], color="black", lw=1.6, zorder=6)
        # Symmetry plane through the liquid, from the axis (or window edge) to the neck.
        x_neck = float(x[chain[0]])
        ax.plot([v.xlim[0], x_neck], [0.0, 0.0], color="grey", lw=0.7, ls=(0, (6, 6)), zorder=5)
        ax.set_xlim(*v.xlim)
        ax.set_ylim(*v.ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(v.xlim[0], v.xlim[1] + 1e-9, v.tick))
        ax.set_yticks(np.arange(v.ylim[0], v.ylim[1] + 1e-9, v.tick))
        ax.set_xlabel(v.xlabel, fontsize=FS_LABEL, labelpad=4)
        ax.set_ylabel(v.ylabel, fontsize=FS_LABEL, labelpad=4)
        ax.tick_params(which="both", direction="out", width=1.2, labelsize=FS_TICK, pad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_LABEL)
        x0, y0, w, h = rect
        cw = 0.20 / lay.fig_w
        cax_u = fig.add_axes([x0 + w + 0.22 / lay.fig_w, y0 + 0.52 * h, cw, 0.46 * h])
        cax_l = fig.add_axes([x0 + w + 0.22 / lay.fig_w, y0 + 0.02 * h, cw, 0.46 * h])
        cb_u = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_SPEED, norm=plt.Normalize(0.0, lim["vmax"])), cax=cax_u)
        cb_l = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_DISS, norm=plt.Normalize(lim["dmin"], lim["dmax"])),
                            cax=cax_l)
        cb_u.set_label(v.upper_label, fontsize=FS_CBAR_LABEL, labelpad=6)
        cb_l.set_label(v.lower_label, fontsize=FS_CBAR_LABEL, labelpad=6)
        for cb in (cb_u, cb_l):
            cb.ax.tick_params(labelsize=FS_CBAR_TICK)
    t_text = sci(fr.t) if fr.t > 0 else "0"
    fig.text(0.5, 1.0 - 0.40 / lay.fig_h,
             rf"$R_{{\min}} = {sci(fr.R_min)}$,   $\rho = {sci(fr.rho)}$,   $t = {t_text}$",
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


# ------------------------------------------------------------ colour limits, QA

def colour_limits(files: list[Path], n_probe: int = 12) -> tuple[dict[str, float], ...]:
    """Fixed limits per view from area-weighted samples (a uniform grid over the upper half
    of the window) of a spread of frames: 99.8th percentiles, dissipation over DECADES."""
    probes = [load_frame(int(i), files[int(i)])
              for i in np.unique(np.linspace(0, len(files) - 1, min(n_probe, len(files))).astype(int))]
    phis = [strain_fields(fr)["phi"] for fr in probes]
    out = []
    for v in VIEWS:
        gx, gy = np.meshgrid(np.linspace(*v.xlim, 241), np.linspace(0.0, v.ylim[1], 121))
        speeds, highs = [], []
        for fr, phi in zip(probes, phis):
            tri = v.triangulation(fr, mirror=False)
            speeds.append(np.percentile(LinearTriInterpolator(tri, v.upper(fr))(gx, gy).compressed(), 99.8))
            highs.append(np.percentile(LinearTriInterpolator(tri, v.lower(fr, phi))(gx, gy).compressed(), 99.8))
        dmax = float(np.ceil(2.0 * max(highs)) / 2.0)
        out.append({"vmax": float(np.ceil(10.0 * max(speeds)) / 10.0), "dmin": dmax - DECADES, "dmax": dmax})
    return tuple(out)


def png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fields", type=Path, help="folder of frame-NNNN.npz from reconstruct_stokes_fields.py")
    ap.add_argument("--out", type=Path, required=True, help="MP4 path")
    ap.add_argument("--frames-dir", type=Path, default=None)
    ap.add_argument("--stills-dir", type=Path, default=None)
    ap.add_argument("--stills-only", action="store_true")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=120)
    a = ap.parse_args(argv)
    files = read_series(a.fields.resolve())
    print(f"{len(files)} frames", flush=True)
    for i in np.unique(np.linspace(0, len(files) - 1, 5).astype(int)):
        fr = load_frame(int(i), files[int(i)])
        chk = divergence_check(fr)
        print(f"  frame {i}: R_min = {fr.R_min:.3e}, rms(div u)/rms(|grad u|): "
              + ", ".join(f"{k} {val:.2e}" for k, val in chk.items()), flush=True)
    limits = colour_limits(files)
    for v, lim in zip(VIEWS, limits):
        print(f"colour limits {v.name}: upper 0 -> {lim['vmax']:.3g}; log dissipation {lim['dmin']:.3g} -> "
              f"{lim['dmax']:.3g}", flush=True)
    lay = make_layout(limits, a.dpi)
    stills_dir = a.stills_dir or a.out.parent
    stills_dir.mkdir(parents=True, exist_ok=True)
    for name, i in {"early": 0, "middle": len(files) // 2, "late": len(files) - 1}.items():
        dest = draw_frame(load_frame(i, files[i]), lay, stills_dir / f"{a.out.stem}-still-{name}.png")
        print(f"wrote {dest}", flush=True)
    if a.stills_only:
        return 0
    frames_dir = a.frames_dir or (a.out.parent / f"{a.out.stem}-frames")
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
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if subprocess.run(encode, check=False).returncode != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {a.out} ({a.out.stat().st_size / 1e6:.2f} MB, {len(made)} frames at {a.fps} fps "
          f"= {len(made) / a.fps:.1f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
