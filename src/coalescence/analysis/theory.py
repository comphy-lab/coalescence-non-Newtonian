"""Stokes-regime theory for the neck of two coalescing drops.

Eggers, Lister & Stone (1999, J. Fluid Mech. 401, 293) give, to leading logarithmic
order and in visco-capillary units (lengths by the drop radius, times by mu R/gamma),

    R_min = -(tau_v / pi) ln tau_v,

with tau_v the time since contact. Two forms of the neck velocity u_v(R_min) follow at
the same order:

- ``u_leading_log``: u_v = -(1/pi) ln R_min, the straight line on a semilogarithmic plot
  drawn in Figure 3(b) of Anthony, Harris & Basaran (2020, Phys. Rev. Fluids 5, 033608);
- ``u_eggers``: u_v = dR_min/dtau_v = -(1 + ln tau_v)/pi, with tau_v found by inverting
  the law on 0 < tau_v < 1/e, where R_min increases monotonically.

They differ by (ln|ln tau_v| - ln pi - 1)/pi, a term beyond the accuracy of the leading
order. Anthony et al. quote the law as valid for R_min < 0.03.
"""

from __future__ import annotations

import numpy as np

R_MAX_STOKES = 0.03


def r_eggers(tau):
    """Neck radius R_min(tau_v) = -(tau_v/pi) ln tau_v."""
    tau = np.asarray(tau, dtype=float)
    return -(tau / np.pi) * np.log(tau)


def tau_eggers(R):
    """Invert R_min = -(tau_v/pi) ln tau_v on 0 < tau_v < 1/e (bisection in ln tau_v).

    The root lies in 2 ln R_min - 1 < ln tau_v < -1 for every admissible R_min, since
    R_min(tau_v) < R_min at tau_v = R_min^2/e.
    """
    R = np.asarray(R, dtype=float)
    if np.any(R <= 0.0) or np.any(R >= 1.0 / (np.pi * np.e)):
        raise ValueError("R_min must lie in (0, 1/(pi e)), where the law is invertible")
    lo, hi = 2.0 * np.log(R) - 1.0, np.full(R.shape, -1.0)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        big = -(np.exp(mid) / np.pi) * mid > R
        hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
    return np.exp(0.5 * (lo + hi))


def u_eggers(R):
    """u_v = dR_min/dtau_v = -(1 + ln tau_v)/pi at the given R_min."""
    return -(1.0 + np.log(tau_eggers(R))) / np.pi


def u_leading_log(R):
    """u_v = -(1/pi) ln R_min, as drawn in Anthony et al. (2020), Figure 3(b)."""
    return -np.log(np.asarray(R, dtype=float)) / np.pi
