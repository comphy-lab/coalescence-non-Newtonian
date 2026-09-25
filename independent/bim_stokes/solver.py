"""Linearly implicit time stepping of the free-surface boundary-integral problem.

With the geometry frozen at step n, the traction at the new position is
linearised in the normal displacement eta = dt (u . N),

    kappa^{n+1} ~ kappa^n - Lap_s eta - (k1^2 + k2^2) eta,

so that the boundary-integral equation for the new velocity is the dense
linear system

    [1/2 I + K/(8 pi) - (dt/8 pi) S D] u = -(1/8 pi) S (kappa^n N),
    D u = N (Lap_s + k1^2 + k2^2)(N . u),

which removes the capillary stability limit of the smallest panel.  Points move
with the normal velocity in the neck-anchored frame; the curve is then resampled
along its spline with the tip-graded spacing.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Meridian, target_spacing
from .operators import Assembler


def laplace_beltrami(mer: Meridian) -> np.ndarray:
    """Nodal matrix of Lap_s eta = eta'' + (r'/r) eta' (2 eta'' at the pole), eta even at both ends."""
    n = mer.n + 1
    s = mer.sigma
    (tX, _), _, _, _, _ = mer.node_frame()
    r = mer.r
    Lm = np.zeros((n, n))
    for i in range(n):
        if i == 0:
            hm = hp = s[1] - s[0]
            im, ip = 1, 1
        elif i == n - 1:
            hm = hp = s[i] - s[i - 1]
            im, ip = i - 1, i - 1
        else:
            hm, hp = s[i] - s[i - 1], s[i + 1] - s[i]
            im, ip = i - 1, i + 1
        c_m = 2.0 / (hm * (hm + hp))
        c_0 = -2.0 / (hm * hp)
        c_p = 2.0 / (hp * (hm + hp))
        d_m = -hp / (hm * (hm + hp))
        d_0 = (hp - hm) / (hm * hp)
        d_p = hm / (hp * (hm + hp))
        if i == n - 1:
            cm2, c02, cp2 = 2 * c_m, 2 * c_0, 2 * c_p
            Lm[i, im] += cm2 + cp2
            Lm[i, i] += c02
            continue
        g = tX[i] / r[i] if i > 0 else 0.0      # r'/r; zero at the neck (r' = 0 there)
        Lm[i, im] += c_m + g * d_m
        Lm[i, i] += c_0 + g * d_0
        Lm[i, ip] += c_p + g * d_p
    return Lm


def operators(mer: Meridian, assembler_kwargs=None):
    """Assembled S and K for the current geometry (reusable across implicit steps)."""
    return Assembler(mer, **(assembler_kwargs or {})).assemble()


def solve_velocity(mer: Meridian, dt: float, assembler_kwargs=None, ops=None):
    """Nodal velocity (u_r, u_z) of the linearly implicit step of size dt."""
    S, K = ops if ops is not None else operators(mer, assembler_kwargs)
    n = mer.n + 1
    (_, _), (Nr, Nz), k1, k2, _ = mer.node_frame()
    kappa = k1 + k2
    G0 = -np.concatenate([kappa * Nr, kappa * Nz])
    L = laplace_beltrami(mer) + np.diag(k1 * k1 + k2 * k2)
    # Linearised in the laboratory normal displacement dt (u . N): this is the shape
    # change (the frame-relative update in advance() differs only by a tangential slip),
    # and for a translating circular tip -Lap_s N_r - k1^2 N_r = 0, so the tip's
    # translation adds no spurious curvature.
    Nmat = np.hstack([np.diag(Nr), np.diag(Nz)])                 # (n, 2n): N . u
    D = np.vstack([np.diag(Nr), np.diag(Nz)]) @ L @ Nmat         # (2n, 2n)
    A = 0.5 * np.eye(2 * n) + K / (8 * math.pi) - (dt / (8 * math.pi)) * (S @ D)
    rhs = S @ G0 / (8 * math.pi)
    for row in (n, n - 1):          # u_z = 0 at the neck, u_r = 0 at the pole
        A[row, :] = 0.0
        A[row, row] = 1.0
        rhs[row] = 0.0
    U = np.linalg.solve(A, rhs)
    return U[:n], U[n:]


def advance(mer: Meridian, dt: float, ur: np.ndarray, uz: np.ndarray) -> Meridian:
    """Move the nodes with the frame-relative normal velocity (no resampling)."""
    (_, _), (Nr, Nz), _, _, _ = mer.node_frame()
    U = float(ur[0])
    # Normal displacement relative to the neck frame; the frame itself moves by U dt.
    # This equals the laboratory normal motion plus a tangential slip U dt t_r t.
    wn = ur * Nr + uz * Nz - U * Nr
    X = mer.X + dt * wn * Nr
    z = mer.z + dt * wn * Nz
    X[0], z[0] = 0.0, 0.0
    return Meridian(R_n=mer.R_n + dt * U, X=X, z=z)


def step_extrapolated(mer: Meridian, dt: float, assembler_kwargs=None):
    """Second-order, L-stable step: 2 x(two half steps) - x(one full step).

    Both paths move the same nodes (no resampling inside the step), so the two
    surfaces are combined node by node; the full step reuses the operators of the
    first half step.  Returns the unresampled meridian and the mean neck speed.
    """
    ops0 = operators(mer, assembler_kwargs)
    ur, uz = solve_velocity(mer, 0.5 * dt, ops=ops0)
    half = advance(mer, 0.5 * dt, ur, uz)
    ur2, uz2 = solve_velocity(half, 0.5 * dt, assembler_kwargs)
    hh = advance(half, 0.5 * dt, ur2, uz2)
    urf, uzf = solve_velocity(mer, dt, ops=ops0)
    full = advance(mer, dt, urf, uzf)
    X = 2.0 * hh.X - full.X
    z = 2.0 * hh.z - full.z
    X[0], z[0] = 0.0, 0.0
    R = 2.0 * hh.R_n - full.R_n
    return Meridian(R_n=R, X=X, z=z), (R - mer.R_n) / dt


def resample(mer: Meridian, k: float, n_tip: int, h_max: float, growth: float = 1.3) -> Meridian:
    """New nodes along the spline with spacing clamp(k d, rho/n_tip, h_max), d the distance from the neck."""
    rho = mer.tip_radius()
    s_end = mer.sigma[-1]
    sig = [0.0]
    h_prev = rho / n_tip
    while True:
        X, z = mer.eval(np.array([sig[-1]]))
        d = math.hypot(float(X[0]), float(z[0]))
        h = min(float(target_spacing(d, rho, k, n_tip, h_max)), growth * h_prev)
        h_prev = h
        nxt = sig[-1] + h
        if nxt >= s_end - 0.5 * h:
            break
        sig.append(nxt)
    sig.append(s_end)
    sig = np.array(sig)
    X, z = mer.eval(sig)
    X[0], z[0] = 0.0, 0.0
    return Meridian(R_n=mer.R_n, X=X, z=z)


@dataclass
class RunConfig:
    k: float = 0.1
    n_tip: int = 16
    h_max: float = 0.02
    dt_initial: float = 1e-15
    dt_fraction: float = 0.02
    curvature_target: float = 0.05
    dt_growth: float = 1.3
    R_stop: float = 0.03
    max_steps: int = 100000
    max_wall_s: float = 1e9
    restart_every: int = 25
    assembler: dict = field(default_factory=dict)


def run(mer: Meridian, out: Path, cfg: RunConfig, t0: float = 0.0, dt0: float | None = None,
        steps0: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "restart").mkdir(exist_ok=True)
    log = (out / "neck.csv").open("a", encoding="utf-8")
    if log.tell() == 0:
        log.write("t,R_min,u_neck,two_H,k1,tip_radius,volume,n_nodes,dt,wall_s\n")
    wall0 = time.time()
    t, dt = t0, (dt0 or cfg.dt_initial)
    V0 = mer.volume()
    k1_prev = mer.neck_curvature()[0]
    ratio = 0.0
    step = steps0
    status = "running"
    U = 0.0
    while True:
        if mer.R_n >= cfg.R_stop:
            status = "reached_R_stop"
            break
        if step - steps0 >= cfg.max_steps or time.time() - wall0 > cfg.max_wall_s:
            status = "limit"
            break
        moved, U = step_extrapolated(mer, dt, cfg.assembler)
        mer = resample(moved, cfg.k, cfg.n_tip, cfg.h_max)
        t += dt
        step += 1
        k1, H2 = mer.neck_curvature()
        ratio = abs(k1 - k1_prev) / abs(k1_prev)
        k1_prev = k1
        V = mer.volume()
        log.write(f"{t:.17g},{mer.R_n:.17g},{U:.17g},{H2:.17g},{k1:.17g},{1.0/abs(k1):.17g},"
                  f"{V:.17g},{mer.n + 1},{dt:.6g},{time.time() - wall0:.3f}\n")
        log.flush()
        if step % cfg.restart_every == 0:
            np.savez(out / "restart" / f"state_{step:06d}.npz", R_n=mer.R_n, X=mer.X, z=mer.z, t=t, dt=dt, step=step)
        dt_phys = cfg.dt_fraction * mer.R_n / max(abs(U), 1e-3)
        factor = min(cfg.dt_growth, max(0.5, math.sqrt(cfg.curvature_target / max(ratio, 1e-12))))
        dt = min(dt_phys, dt * factor)
    log.close()
    summary = {"status": status, "steps": step, "t": t, "R_min": mer.R_n, "u_neck": float(U) if step > steps0 else None,
               "volume_drift": (mer.volume() - V0) / V0, "wall_s": time.time() - wall0}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(out / "restart" / "final.npz", R_n=mer.R_n, X=mer.X, z=mer.z, t=t, dt=dt, step=step)
    return summary
