"""Exact signed-Jacobian range for a six-node quadratic triangle."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _basis(points: np.ndarray) -> np.ndarray:
    s, t = points[:, 0], points[:, 1]
    return np.column_stack((np.ones(len(points)), s, t, s * s, s * t, t * t))


@lru_cache(maxsize=4)
def _inverse_basis(local_flat: tuple[float, ...]) -> np.ndarray:
    return np.linalg.inv(_basis(np.asarray(local_flat).reshape(6, 2)))


def _quadratic_range(coefficients: np.ndarray) -> tuple[float, float]:
    c0, cs, ct, css, cst, ctt = coefficients

    def value(s: float, t: float) -> float:
        return float(c0 + cs * s + ct * t + css * s * s + cst * s * t + ctt * t * t)

    corners = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    candidates = [value(*point) for point in corners]
    for start, end in ((corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0])):
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        q0, qm, q1 = value(*start), value(*midpoint), value(*end)
        a = 2 * (q0 - 2 * qm + q1)
        b = q1 - q0 - a
        if a != 0:
            tau = -b / (2 * a)
            if 0 < tau < 1:
                candidates.append(value(start[0] + tau * (end[0] - start[0]),
                                        start[1] + tau * (end[1] - start[1])))
    hessian = np.array([[2 * css, cst], [cst, 2 * ctt]])
    if abs(np.linalg.det(hessian)) > 1e-14 * max(np.linalg.norm(hessian, ord=np.inf) ** 2, 1e-300):
        s, t = np.linalg.solve(hessian, -np.array([cs, ct]))
        if s > 0 and t > 0 and s + t < 1:
            candidates.append(value(float(s), float(t)))
    return min(candidates), max(candidates)


def signed_jacobian_range(local: np.ndarray, physical: np.ndarray) -> tuple[float, float]:
    """Return the exact minimum and maximum det(dx/d(s,t)) on the triangle."""
    local = np.asarray(local, dtype=np.float64)
    physical = np.asarray(physical, dtype=np.float64)
    if local.shape != (6, 2) or physical.shape != (6, 2):
        raise ValueError("six local and physical Q2 node coordinates are required")
    inverse = _inverse_basis(tuple(float(x) for x in local.flat))
    x, y = inverse @ physical[:, 0], inverse @ physical[:, 1]

    def det_at(s: float, t: float) -> float:
        xs, xt = x[1] + 2 * x[3] * s + x[4] * t, x[2] + x[4] * s + 2 * x[5] * t
        ys, yt = y[1] + 2 * y[3] * s + y[4] * t, y[2] + y[4] * s + 2 * y[5] * t
        return float(xs * yt - xt * ys)

    det_coefficients = inverse @ np.array([det_at(float(s), float(t)) for s, t in local])
    return _quadratic_range(det_coefficients)
