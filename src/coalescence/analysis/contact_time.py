"""Contact time of a computation that starts from a finite bridge.

Anthony, Harris & Basaran (2020, Sec. III) fit the simulated R_min(t) to a power law
once the bridge has grown by an order of magnitude and extrapolate it to zero radius;
the fitted shift t_con places t = 0 of the simulation at tau = t + t_con after contact.
Here the law R_min = A (t + t_con)^n is fitted over R0 * window[0] <= R_min <=
R0 * window[1] (default one decade, 10 R0 to 100 R0) by unconstrained least squares from
t_con = R0. The shift affects only the first decades of R_min(tau); the Stokes-law value
at 10 R0 serves as a sensitivity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, curve_fit


@dataclass(frozen=True)
class ContactTime:
    t_con: float
    prefactor: float
    exponent: float
    window: tuple[float, float]
    samples: int


def power_law_contact_time(t, R, R0: float, window: tuple[float, float] = (10.0, 100.0)) -> ContactTime:
    """Fit R_min = A (t + t_con)^n over the radius window and return t_con, A and n."""
    t, R = np.asarray(t, dtype=float), np.asarray(R, dtype=float)
    lo, hi = window[0] * R0, window[1] * R0
    m = (R >= lo) & (R <= hi)
    if np.count_nonzero(m) < 4:
        raise ValueError(f"the fit window [{lo:.3g}, {hi:.3g}] holds fewer than four samples")
    law = lambda tt, A, n, tc: A * (tt + tc) ** n
    with np.errstate(invalid="ignore"):          # trial shifts may leave t + t_con < 0
        (A, n, tc), _ = curve_fit(law, t[m], R[m], p0=(0.5, 1.0, R0), maxfev=20000)
    if not (np.isfinite(tc) and tc >= 0.0 and n > 0.0):
        raise ValueError(f"the power-law fit gave t_con = {tc:.3g}, n = {n:.3g}")
    return ContactTime(t_con=float(tc), prefactor=float(A), exponent=float(n), window=(lo, hi),
                       samples=int(np.count_nonzero(m)))


def stokes_law_contact_time(t, R, R_match: float) -> float:
    """Shift that puts R_min(t) on R_min = -(tau/pi) ln tau at R_min = R_match."""
    t, R = np.asarray(t, dtype=float), np.asarray(R, dtype=float)
    if np.any(np.diff(R) <= 0.0):
        raise ValueError("R_min must increase strictly to be interpolated")
    if not R[0] <= R_match <= R[-1]:
        raise ValueError(f"R_match = {R_match:g} lies outside the sampled range {R[0]:.3g} to {R[-1]:.3g}")
    tau = brentq(lambda x: -(x / np.pi) * np.log(x) - R_match, 1e-300, 1.0 / np.e)
    return float(tau - np.interp(R_match, R, t))
