"""Render an interface-profile video from saved (r, z) snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "cm"


def load_profile(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    t = float("nan")
    rs: list[float] = []
    zs: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                for part in line[1:].split():
                    if part.startswith("t="):
                        t = float(part[2:])
                continue
            if line.startswith("r"):
                continue
            r, z = line.split()
            rs.append(float(r))
            zs.append(float(z))
    return t, np.array(rs), np.array(zs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    files = sorted(args.profiles.glob("interface_*.dat"))
    if not files:
        raise SystemExit(f"no profiles in {args.profiles}")
    frames = [load_profile(path) for path in files]
    first, mid, last = frames[0], frames[len(frames) // 2], frames[-1]
    for label, frame in (("first", first), ("middle", mid), ("last", last)):
        if frame[1].size < 4:
            raise SystemExit(f"{label} profile is empty")
    fig, ax = plt.subplots(figsize=(8, 10))
    (line,) = ax.plot([], [], "k-", lw=2)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$r$", fontsize=20)
    ax.set_ylabel(r"$z$", fontsize=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 2.05)
    time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=16)
    writer = FFMpegWriter(fps=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, args.output, dpi=150):
        for t, r, z in frames:
            order = np.argsort(z)
            line.set_data(r[order], z[order])
            time_text.set_text(rf"$t={t:.3e}$")
            writer.grab_frame()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
