"""Neck histories written by both solvers, and their comparison at equal neck radius.

Both runners write ``neck.csv`` with at least the columns ``t``, ``R_min``, ``u_neck``
and ``tip_radius``. The finite-element row at t = 0 is the initial state and is dropped.
The finite-element speed is instantaneous at the recorded radius. The boundary-integral
speed is the implicit speed of the step that ends at the recorded radius, so it is
assigned to the geometric mean of the radii at the two ends of the step.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

COLUMNS = ("t", "R_min", "u_neck", "tip_radius")


def read_neck(path: Path, drop_initial: bool = False) -> dict[str, np.ndarray]:
    """Read the standard columns of a neck.csv; optionally drop the t = 0 row."""
    with Path(path).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    data = {k: np.array([float(r[k]) for r in rows]) for k in COLUMNS}
    if drop_initial:
        data = {k: v[1:] for k, v in data.items()}
    if np.any(np.diff(data["t"]) <= 0.0):
        raise ValueError(f"time is not strictly increasing: {path}")
    return data


def step_radius(R: np.ndarray, R_before: float) -> np.ndarray:
    """Geometric-mean radius of each step, given the radius before the first step."""
    return np.sqrt(np.concatenate([[R_before], R[:-1]]) * R)


def interp_log(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation in log x."""
    return np.interp(np.log(x), np.log(xp), fp)


def relative_deviation(R: np.ndarray, u: np.ndarray, R_ref: np.ndarray, u_ref: np.ndarray):
    """100 (u / u_ref - 1) at the samples R inside the reference range, and those samples."""
    inside = (R >= R_ref[0]) & (R <= R_ref[-1])
    return R[inside], 100.0 * (u[inside] / interp_log(R[inside], R_ref, u_ref) - 1.0)


def peak(R: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """(u_max, R at u_max)."""
    i = int(np.argmax(u))
    return float(u[i]), float(R[i])
