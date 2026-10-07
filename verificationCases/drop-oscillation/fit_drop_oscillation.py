#!/usr/bin/env python3
"""Fit the free-drop P2 oscillation and compare with Rayleigh's frequency and Lamb's decay rate.

For each run folder (``run_fem_drop_oscillation.py`` output), the pole height is fitted
with z_pole(t) = c + A exp(-delta t) cos(omega t + phi) after the first quarter period
(the response of a drop released from rest departs from the normal-mode form at short
times; Prosperetti 1980). The fitted frequency and decay rate are compared with the
leading-order Rayleigh and Lamb values and with the exact linear normal mode at the same
Oh (``coalescence.analysis.drop_modes``), which carries the finite-Oh corrections.

The energy budget d/dt(kinetic + area) = -dissipation is checked over whole periods after
the first, where the kinetic-surface exchange cancels and only dissipation remains. The
surface energy is taken at fixed volume, A - (dA/dV)(V - V0) with dA/dV = 2 for the unit
sphere: a volume drift of 1e-6 otherwise appears as a surface-energy change comparable to
what a weakly damped mode dissipates in a period. The uncorrected value is reported too.
Writes a JSON summary to --out and prints it.

    python verificationCases/drop-oscillation/fit_drop_oscillation.py <run folder> [...] --out <file.json>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from coalescence.analysis.drop_modes import oscillatory_mode  # noqa: E402


def read(path: Path) -> dict[str, np.ndarray]:
    with (path / "oscillation.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def energy_budget_per_period(t: np.ndarray, energy: np.ndarray, dissipated: np.ndarray, period: float) -> list[float]:
    """(E(b) - E(a) + W(b) - W(a)) / (W(b) - W(a)) over whole periods [a, b] = [k T, (k + 1) T], k >= 1.

    E and the dissipated work W are interpolated linearly to the exact period boundaries,
    so the window does not depend on where the time steps fall.
    """
    out = []
    k = 1
    while (k + 1) * period <= t[-1] * (1 + 1e-12):
        bounds = np.array([k * period, min((k + 1) * period, t[-1])])
        E_a, E_b = np.interp(bounds, t, energy)
        W_a, W_b = np.interp(bounds, t, dissipated)
        lost = W_b - W_a
        out.append(float((E_b - E_a + lost) / lost))
        k += 1
    return out


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
    mode = oscillatory_mode(m["Oh"])
    # Energy budget over whole periods after the first, at fixed volume.
    W = np.concatenate([[0.0], np.cumsum(0.5 * (d["dissipation"][1:] + d["dissipation"][:-1]) * np.diff(d["t"]))])
    E_raw = d["kinetic"] + d["area"]
    E_fix = E_raw - 2.0 * (d["volume"] - d["volume"][0])
    per_period = energy_budget_per_period(d["t"], E_fix, W, period)
    per_period_raw = energy_budget_per_period(d["t"], E_raw, W, period)
    return {
        "run": path.name, "Oh": m["Oh"], "eps": m["eps"], "resolution": m["resolution"],
        "steps_per_period": round(period / m["dt"]),
        "omega_fit": float(om), "omega_rayleigh": omega0, "omega_relative_error": float(om / omega0 - 1),
        "decay_fit": float(dl), "decay_lamb": delta0, "decay_relative_error": float(dl / delta0 - 1),
        "omega_mode": mode["omega"], "omega_vs_mode": float(om / mode["omega"] - 1),
        "decay_mode": mode["decay_rate"], "decay_vs_mode": float(dl / mode["decay_rate"] - 1),
        "fit_rms_over_amplitude": float(np.sqrt(np.mean(resid ** 2)) / abs(A)),
        "volume_drift": float(d["volume"][-1] / d["volume"][0] - 1),
        "energy_budget_per_period_fixed_volume": per_period,
        "energy_budget_per_period_uncorrected": per_period_raw,
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
