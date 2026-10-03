#!/usr/bin/env python3
"""Standard coalescence video of one case: the nested-zoom hybrid, from the run folders to a checked MP4.

This is the one entry point for case videos. Given the runtime folders of a case in time
order (a run, then any restart continuations), it

1. gathers field frames: the run's own ``fields/`` (written with ``--field-frames``,
   valid for any rheology and inertia) or, for a Stokes run without them, frames rebuilt
   from its saved states with ``reconstruct_stokes_fields.py`` (the Stokes velocity is
   fixed by the geometry, so this is exact);
2. renders the hybrid video with ``render_hybrid_video.py``: the whole pair, the neck and
   the meniscus as strictly nested windows centred on the tip, on a clock logarithmic in t
   up to ``--t-switch`` and linear beyond;
3. checks the result: frames sampled from the encoded file itself are tiled into a QA grid,
   the encode is probed (codec, size, frame count, duration), and a manifest records the
   sources, the component commit, the settings and SHA-256 sums of the outputs.

An inertial run without ``fields/`` cannot be rebuilt (its velocity carries history), and
is refused with that reason.

    python postProcess/make_case_video.py <runtime> [<runtime> ...] --out-dir <folder> \\
        --name <stem> [--pyoomph-python <python with the pinned pyoomph>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def field_folder(runtime: Path) -> Path | None:
    folder = runtime / "fields"
    return folder if folder.is_dir() and any(folder.glob("frame-*.npz")) else None


def rebuild_stokes(runtime: Path, out: Path, python: str, workers: int) -> Path:
    """Rebuild the Stokes fields of a run that did not write them."""
    manifest = json.loads((runtime / "run-manifest.json").read_text())
    if manifest.get("Oh") is not None:
        raise SystemExit(f"{runtime}: inertial run without fields/; its velocity carries history and "
                         "cannot be rebuilt from the interface (rerun with --field-frames)")
    states = "snapshots" if (runtime / "snapshots").is_dir() else "remesh"
    cmd = [python, str(HERE / "reconstruct_stokes_fields.py"), str(runtime), "--out", str(out),
           "--states", states, "--workers", str(workers)]
    print("rebuilding:", " ".join(cmd[1:]), flush=True)
    if subprocess.run(cmd, check=False).returncode != 0:
        raise SystemExit(f"field reconstruction failed for {runtime}")
    return out / "fields"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def probe(video: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration",
                          "-of", "json", str(video)], capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return {**data["streams"][0], "duration": float(data["format"]["duration"])}


def qa_grid(video: Path, dest: Path, times: list[float]) -> None:
    """Tile six frames decoded from the encoded file (not the PNGs) at the given playback times."""
    if len(times) != 6:
        raise ValueError("the QA grid takes six playback times")
    tmp = []
    for i, s in enumerate(times):
        frame = dest.parent / f".qa-{i}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{s:.3f}", "-i", str(video),
                        "-frames:v", "1", str(frame)], check=True)
        tmp.append(frame)
    graph = ("".join(f"[{i}]scale=1100:-1[s{i}];" for i in range(6))
             + "[s0][s1]hstack[r0];[s2][s3]hstack[r1];[s4][s5]hstack[r2];[r0][r1][r2]vstack=inputs=3")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *sum((["-i", str(f)] for f in tmp), []),
                    "-filter_complex", graph, str(dest)], check=True)
    for f in tmp:
        f.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runtimes", type=Path, nargs="+", help="runtime folders of one case, in time order")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", required=True, help="file stem of the video (add the date for a new render)")
    ap.add_argument("--pyoomph-python", default=sys.executable,
                    help="interpreter with the pinned pyoomph, for rebuilding Stokes fields")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--duration", type=float, default=32.0)
    ap.add_argument("--t-switch", type=float, default=0.05)
    a = ap.parse_args()
    out = a.out_dir.resolve()
    (out / "media").mkdir(parents=True, exist_ok=True)
    folders, sources = [], []
    for i, rt in enumerate(r.resolve() for r in a.runtimes):
        own = field_folder(rt)
        folder = own or rebuild_stokes(rt, out / f"rebuilt-{i}", a.pyoomph_python, a.workers)
        folders.append(folder)
        sources.append({"runtime": str(rt), "fields": str(folder), "origin": "run" if own else "rebuilt-stokes"})
    video = out / "media" / f"{a.name}.mp4"
    cmd = [sys.executable, str(HERE / "render_hybrid_video.py"), *map(str, folders), "--out", str(video),
           "--workers", str(a.workers), "--duration", str(a.duration), "--t-switch", str(a.t_switch),
           "--stills", "0,0.5,1"]
    if subprocess.run(cmd, check=False).returncode != 0:
        raise SystemExit("render failed")
    info = probe(video)
    grid = out / "media" / f"{a.name}-qa-grid.png"
    d = info["duration"]
    qa_grid(video, grid, [0.03 * d, 0.1 * d, 0.3 * d, 0.55 * d, 0.8 * d, 0.98 * d])
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    manifest = {"schema": "case-video-v1", "component_commit": commit + ("-dirty" if dirty else ""),
                "sources": sources, "video": str(video), "qa_grid": str(grid), "encode": info,
                "settings": {"duration_s": a.duration, "t_switch": a.t_switch,
                             "windows": "ell_b = min(R_min, 0.7), ell_c = min(6 rho, ell_b/10), tip-centred"},
                "sha256": {p.name: sha256(p) for p in sorted((out / "media").glob("*")) if p.is_file()}}
    (out / "media" / f"{a.name}-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"video": str(video), "qa_grid": str(grid), "encode": info}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
