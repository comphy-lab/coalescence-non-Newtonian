"""Checksum-verified plot inputs, from registered runs or a compact public bundle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from coalescence.analysis.neck import read_neck  # noqa: E402
from coalescence.analysis.runs import load_runs, load_series, neck_file  # noqa: E402

DEFAULT_DATA = REPO / "docs/finite-oh-validation/data"
FINITE_ID = "la0-oh06-r0-1e-6-16fc4a0"
STOKES_ID = "la0-fig3-r0-1e-6-zone-6c9f06c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_run(runs, run_id):
    matches = [run for run in runs if run.id == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected one registered run {run_id!r}, found {len(matches)}")
    return matches[0]


def load_inputs(data: Path | None = DEFAULT_DATA):
    if data is not None:
        manifest = json.loads((data / "manifest.json").read_text())
        archive = data / "curves.npz"
        if manifest["schema_version"] != 1 or digest(archive) != manifest["curves_sha256"]:
            raise ValueError("public plot-input bundle schema or checksum mismatch")
        with np.load(archive, allow_pickle=False) as packed:
            histories = {
                name: {column: packed[f"{name}__{column}"] for column in item["columns"]}
                for name, item in manifest["histories"].items()
            }
        return histories, manifest

    histories, metadata = {}, {}

    def add(name, runs, run, full=False):
        if full:
            path = neck_file(run)
            summary = json.loads((path.parent / "summary.json").read_text())
            if summary["status"] != "reached_R_stop":
                raise ValueError(f"incomplete validation run {run.id}: {summary['status']}")
            manifest = json.loads((path.parent / "run-manifest.json").read_text())
            if name == "finite" and "visco-capillary" not in manifest.get("units", ""):
                raise ValueError(f"expected visco-capillary units for {run.id}")
            if name == "finite" and manifest["Oh"] != 0.6:
                raise ValueError("finite-Oh validation requires Oh=0.6")
            values = read_neck(path, extra=("two_H", "volume"))
        else:
            values = load_series(runs, run)
        histories[name] = values
        chain, ancestor = [], run
        while True:
            chain.append({"id": ancestor.id, "neck_sha256": ancestor.neck_sha256,
                          "commit": ancestor.commit})
            if not ancestor.continues:
                break
            ancestor = find_run(runs, ancestor.continues)
        metadata[name] = {"R0": run.R0, "solver": run.solver,
                          "label": run.label, "sources": chain,
                          "columns": list(values)}

    runs = load_runs(REPO / "validationCases/anthony2020-fig3-oh06/runs.toml")
    add("finite", runs, find_run(runs, FINITE_ID), full=True)
    add("stokes", runs, find_run(runs, STOKES_ID), full=True)
    runs = load_runs(REPO / "verificationCases/fem-oh06-consistency/runs.toml")
    for name, run_id in (("lab", "oh06-r1e-3-lab"), ("moving", "oh06-r1e-3-mov"),
                         ("single_map", "oh06-r1e-3-prod"),
                         ("half_dt", "v4-oh06-r1e-3-halfdt")):
        add(name, runs, find_run(runs, run_id))
    runs = load_runs(REPO / "verificationCases/bim-vs-fem-stokes-startup/runs.toml")
    for run in runs:
        if run.solver == "bim" and not any(
                child.continues == run.id and child.series == run.series for child in runs):
            add(f"bim_{run.series}", runs, run)
    runs = load_runs(REPO / "verificationCases/fem-startup-r0-independence/runs.toml")
    for i, run in enumerate(sorted(runs, key=lambda run: run.R0)):
        add(f"family_{i}", runs, run)
    return histories, {"schema_version": 1, "units": "visco-capillary",
                       "histories": metadata}


def export_inputs(data: Path, histories: dict, manifest: dict):
    targets = [data / "curves.npz", data / "manifest.json"]
    if any(path.exists() for path in targets):
        raise FileExistsError("plot-input bundle exists: choose a new directory")
    data.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(targets[0], **{
        f"{name}__{column}": values for name, columns in histories.items()
        for column, values in columns.items()
    })
    manifest = {**manifest, "curves_sha256": digest(targets[0])}
    targets[1].write_text(json.dumps(manifest, indent=2) + "\n")


def at_radius(data, radius, column="u_neck"):
    radius = np.asarray(radius)
    if np.any(radius < data["R_min"][0]) or np.any(radius > data["R_min"][-1]):
        raise ValueError("radius interpolation would extrapolate")
    return np.interp(np.log(radius), np.log(data["R_min"]), data[column])


def stats(values):
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("no comparison samples in the requested range")
    return {"n": int(values.size), "median_percent": float(np.median(values)),
            "min_percent": float(values.min()), "max_percent": float(values.max()),
            "p95_abs_percent": float(np.percentile(np.abs(values), 95))}
