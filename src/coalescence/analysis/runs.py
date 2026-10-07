"""Run lists: which simulations an evidence case uses, and where their outputs live.

A run list (``runs.toml``, kept outside the repository) has one ``[[run]]`` table per simulation:

    id           simulation identifier (the name of its directory under a data root)
    solver       "fem" or "bim"
    case         case file, relative to the repository root
    R0           initial neck radius of the case
    commit       component commit that produced the run
    runtime      folder holding neck.csv, relative to the simulation directory
    neck_sha256  SHA-256 of that neck.csv
    role, series, label, continues   optional; ``continues`` names the run whose final
                                     state this run was restarted from

Run lists carry no machine paths. Simulation directories are looked up under the roots
listed in an untracked ``data-roots.toml`` at the repository root (``roots = [...]``) or
in the ``COALESCENCE_DATA_ROOTS`` environment variable (``os.pathsep``-separated).
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .neck import read_neck, step_radius

REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Run:
    id: str
    solver: str
    case: str
    R0: float
    commit: str
    runtime: str
    neck_sha256: str
    role: str = ""
    series: str = ""
    label: str = ""
    continues: str = ""


def load_runs(path: Path) -> list[Run]:
    with Path(path).open("rb") as fh:
        table = tomllib.load(fh)
    return [Run(**r) for r in table["run"]]


def data_roots() -> list[Path]:
    env = os.environ.get("COALESCENCE_DATA_ROOTS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    cfg = REPO / "data-roots.toml"
    if cfg.exists():
        with cfg.open("rb") as fh:
            return [Path(p) for p in tomllib.load(fh)["roots"]]
    raise FileNotFoundError("no data roots: create data-roots.toml or set COALESCENCE_DATA_ROOTS")


def neck_file(run: Run, roots: list[Path] | None = None) -> Path:
    """Locate the run's neck.csv under the data roots whose SHA-256 matches the run list."""
    mismatched = []
    for root in roots or data_roots():
        path = root / run.id / run.runtime / "neck.csv"
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() == run.neck_sha256:
                return path
            mismatched.append(str(path))
    if mismatched:
        raise ValueError(f"{run.id}: neck.csv SHA-256 differs from the run list at {', '.join(mismatched)}")
    raise FileNotFoundError(f"{run.id}: not found under any data root")


def load_series(runs: list[Run], last: Run, roots: list[Path] | None = None) -> dict[str, np.ndarray]:
    """Neck history of the series ending in ``last``, following ``continues`` back to t = 0.

    Each continuation replaces its parent's rows from the restart time on, and the radius
    before its first step is the parent's radius at the restart. Only rows produced by
    runs of the same ``series`` as ``last`` are returned; ``R_step`` holds step radii.
    """
    by_id = {r.id: r for r in runs}
    chain, seen = [last], {last.id}
    while chain[-1].continues:
        parent = chain[-1].continues
        if parent in seen:
            raise ValueError(f"cyclic continuation through {parent}")
        seen.add(parent)
        chain.append(by_id[parent])
    chain.reverse()
    data = None
    for i, run in enumerate(chain):
        seg = read_neck(neck_file(run, roots), drop_initial=(run.solver == "fem" and i == 0))
        R_before = run.R0
        if data is not None:
            before = data["t"] < seg["t"][0]
            if not before.any():
                raise ValueError(f"{run.id} starts at t = {seg['t'][0]:g}, before any retained sample "
                                 f"of the run it continues, {chain[i - 1].id}")
            R_before = data["R_min"][before][-1]
            data = {k: v[before] for k, v in data.items()}
        seg["R_step"] = step_radius(seg["R_min"], R_before)
        seg["own"] = np.full(seg["t"].size, run.series == last.series)
        data = seg if data is None else {k: np.concatenate([data[k], seg[k]]) for k in seg}
    own = data.pop("own")
    return {k: v[own] for k, v in data.items()}
