#!/usr/bin/env python3
"""Fit the free-drop P2 oscillation and compare with Rayleigh's frequency and Lamb's decay rate.

For each run folder (``run_fem_drop_oscillation.py`` output), the pole height is fitted
with z_pole(t) = c + A exp(-delta t) cos(omega t + phi) after the first quarter period
(the response of a drop released from rest departs from the normal-mode form at short
times; Prosperetti 1980). The energy budget d/dt(kinetic + area) = -dissipation is
checked step by step with the trapezoidal rule, after the first period. Writes a JSON
summary to --out and prints it.

    python postProcess/fit_drop_oscillation.py <run folder> [...] --out <file.json>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def read(path: Path) -> dict[str, np.ndarray]:
    with (path / "oscillation.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def fit_run(path: Path) -> dict:
    m = json.loads((path / "run-manifest.json").read_text())
    d = read(path)
    omega0, delta0 = m["theory"]["omega"], m["theory"]["decay_rate"]
    period = 2 * np.pi / omega0
    keep = d["t"] >= 0.25 * period
    t, z = d["t"][keep], d["z_pole"][keep]
    model = lambda tt, c, A, dl, om, ph: c + A * np.exp(-dl * tt) * np.cos(om * tt + ph)
    p0 = (1.0, m["eps"], delta0, omega0, 0.0)
    (c, A, dl, om, ph), cov = curve_fit(model, t, z, p0=p0, maxfev=20000)
    resid = z - model(t, c, A, dl, om, ph)
    # Energy budget after the first period, trapezoidal in time.
    E = d["kinetic"] + d["area"]
    dt = np.diff(d["t"])
    lhs = np.diff(E) / dt
    rhs = -0.5 * (d["dissipation"][1:] + d["dissipation"][:-1])
    late = d["t"][1:] >= period
    budget = (lhs[late] - rhs[late]) / np.abs(rhs[late])
    return {
        "run": path.name, "Oh": m["Oh"], "eps": m["eps"], "resolution": m["resolution"],
        "steps_per_period": round(period / m["dt"]),
        "omega_fit": float(om), "omega_rayleigh": omega0, "omega_relative_error": float(om / omega0 - 1),
        "decay_fit": float(dl), "decay_lamb": delta0, "decay_relative_error": float(dl / delta0 - 1),
        "fit_rms_over_amplitude": float(np.sqrt(np.mean(resid ** 2)) / abs(A)),
        "volume_drift": float(d["volume"][-1] / d["volume"][0] - 1),
        "energy_budget_relative_residual": {"median": float(np.median(budget)),
                                            "max_abs": float(np.max(np.abs(budget)))},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    results = [fit_run(p) for p in a.runs]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
