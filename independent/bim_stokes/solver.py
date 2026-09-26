"""Linearly implicit time stepping of the free-surface boundary-integral problem.

With the geometry frozen at step n, the traction at the new position is
linearised in the normal displacement eta = dt (u . N),

    kappa^{n+1} ~ kappa^n + J eta,   J = -(Lap_s + k1^2 + k2^2) + O(h^2),

with J the exact Jacobian of the nodal spline curvature (Meridian.
curvature_jacobian) and the translation of the tip with the neck taken out of
it (curvature_response), so that the boundary-integral equation for the new
velocity is the dense linear system

    [C + K/(8 pi) - (dt/8 pi) S_f D] u - (1/8 pi) S_D f_z = -(1/8 pi) S_f (kappa^n N),
    D u = -N (d kappa / d(dt u)) u,

on the half body (free surface S_f plus the symmetry-plane disc D, on which u_r
and f_z are the unknowns); this removes the capillary stability limit of the
smallest panel.  The kinematics are linearly implicit as well (Kinematics): nodes
near the neck move with it, nodes far away stay in the laboratory, and the normal
displacement relative to that reference is advected implicitly along the surface.
After each step the curve is resampled along its spline with the tip-graded spacing.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Meridian, disc_for, target_spacing
from .operators import Assembler, Operators


def translation_weight(mer: Meridian) -> np.ndarray:
    """1 within R_n/4 of the neck, 0 beyond R_n/2, smooth in between."""
    x = (np.hypot(mer.X, mer.z) / mer.R_n - 0.25) / 0.25
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - x * x * (3.0 - 2.0 * x)


@dataclass
class Kinematics:
    """Linearly implicit kinematics of one stage.

    The reference curve moves radially with w U (with the neck near the tip, fixed in
    the laboratory far from it); relative to it the nodes move along the normal by

        eta = A^{-1} dt [(u - w U e_r) . N],   A = I + dt diag(v_t) d/ds,

    the backward-Euler form of the kinematic condition in which a displacement is
    advected by the tangential velocity v_t = (u* - w U* e_r) . t of the surface
    relative to the nodes (u* a predictor).  Treating this advection explicitly is
    unstable once U dt exceeds the node spacing on the flanks of the gap, where the
    surface, static in the laboratory, streams past nodes that move with the neck.
    """

    Ainv: np.ndarray       # (n, n)
    w: np.ndarray          # (n,)


def kinematics(mer: Meridian, dt: float, ur_star: np.ndarray, uz_star: np.ndarray) -> Kinematics:
    (tX, tz), _, _, _, _ = mer.node_frame()
    w = translation_weight(mer)
    vt = ur_star * tX + uz_star * tz - w * ur_star[0] * tX
    A = np.eye(mer.n + 1) + dt * vt[:, None] * mer.arclength_derivative()
    return Kinematics(Ainv=np.linalg.inv(A), w=w)


def curvature_response(mer: Meridian, kin: Kinematics | None = None) -> np.ndarray:
    """d kappa / d(dt u) at the nodes, (n, 2n) in the columns (u_r, u_z) of the free surface.

    Near the neck the laboratory displacement of a step is dominated by the translation
    U dt N_r of the tip with the neck (U = u_r at P), which can exceed the tip radius
    many times over; its exact effect is only the change -N_r U dt / r^2 of k2.  The
    curvature is therefore linearised about the reference curve translated by w U dt
    (see Kinematics):

        d kappa = J eta + T U dt,   eta = A^{-1} dt [N . u - w N_r U],
        T = -w N_r / r^2 + J (w N_r) - w (J N_r),

    where T is the curvature change of the reference translation (exact where w = 1,
    linear in the transition).  The Jacobian thus acts only on the deformation relative
    to the moving tip and never on the translation itself.
    """
    n = mer.n + 1
    (_, _), (Nr, Nz), _, _, _ = mer.node_frame()
    J = mer.curvature_jacobian()
    w = translation_weight(mer) if kin is None else kin.w
    Ainv = np.eye(n) if kin is None else kin.Ainv
    r = mer.r
    with np.errstate(divide="ignore", invalid="ignore"):
        k2_shift = np.where(w > 0.0, Nr / np.where(r > 0.0, r * r, 1.0), 0.0)
    M = np.hstack([np.diag(Nr), np.diag(Nz)])                  # N . u
    M[:, 0] -= w * Nr                                          # - w N_r U
    R = J @ Ainv @ M
    R[:, 0] += -w * k2_shift + J @ (w * Nr) - w * (J @ Nr)
    return R


def operators(mer: Meridian, assembler_kwargs=None, disc_kwargs=None) -> Operators:
    """Assembled half-body operators for the current geometry (reusable across implicit steps)."""
    disc = disc_for(mer, **(disc_kwargs or {}))
    return Assembler(mer, disc, **(assembler_kwargs or {})).assemble()


def solve_velocity(mer: Meridian, dt: float, assembler_kwargs=None, ops: Operators | None = None,
                   disc_kwargs=None, full: bool = False, kin: Kinematics | None = None):
    """Free-surface nodal velocity (u_r, u_z) of the linearly implicit step of size dt.

    With full=True also returns the disc radial velocity (nodes 1..M) and the disc
    normal traction f_z (nodes 0..M).
    """
    if ops is None:
        ops = operators(mer, assembler_kwargs, disc_kwargs)
    N, M, nr, nu, nf = ops.N, ops.M, ops.n_row, ops.n_u, ops.n_f
    n = N + 1
    (_, _), (Nr, Nz), k1, k2, _ = mer.node_frame()
    kappa = k1 + k2
    G0 = -np.concatenate([kappa * Nr, kappa * Nz])
    D = -np.vstack([np.diag(Nr), np.diag(Nz)]) @ curvature_response(mer, kin)   # (2n, 2n)
    k8 = 8.0 * math.pi
    sf_f = np.concatenate([np.arange(n), nf + np.arange(n)])     # free-surface traction columns
    sf_u = np.concatenate([np.arange(n), nu + np.arange(n)])     # free-surface velocity columns
    S_f = ops.S[:, sf_f]
    Au = ops.C + ops.K / k8
    Au[:, sf_u] -= (dt / k8) * (S_f @ D)
    Af = -ops.S[:, nf + N + 1 + np.arange(M + 1)] / k8           # disc f_z columns
    rhs = (ops.cap if ops.cap is not None else S_f @ G0) / k8   # net capillary force exact
    fixed = {nu + 0, N, N + M} | {nu + N + q for q in range(1, M + 1)}   # u_z(P), u_r(pole), u_r(axis), disc u_z
    u_cols = np.array([c for c in range(2 * nu) if c not in fixed])
    rows = np.array([r for r in range(2 * nr) if r not in (N, N + M)])  # trivial r-rows on the axis
    A = np.hstack([Au[np.ix_(rows, u_cols)], Af[rows]])
    x = np.linalg.solve(A, rhs[rows])
    U = np.zeros(2 * nu)
    U[u_cols] = x[:u_cols.size]
    ur, uz = U[:n], U[nu:nu + n]
    if full:
        return ur, uz, U[n:nu], x[u_cols.size:]
    return ur, uz


def advance(mer: Meridian, dt: float, ur: np.ndarray, uz: np.ndarray, kin: Kinematics) -> Meridian:
    """Move the nodes: reference translation w U dt e_r plus the normal displacement eta.

    In the neck-anchored frame (which moves by U dt) this is X -> X - (1 - w) U dt +
    eta N_r, z -> z + eta N_z: nodes near the neck move with it, nodes far from it stay
    in the laboratory.  No resampling.
    """
    (_, _), (Nr, Nz), _, _, _ = mer.node_frame()
    U = float(ur[0])
    eta = kin.Ainv @ (dt * (ur * Nr + uz * Nz - kin.w * U * Nr))
    X = mer.X - (1.0 - kin.w) * U * dt + eta * Nr
    z = mer.z + eta * Nz
    X[0], z[0] = 0.0, 0.0
    return Meridian(R_n=mer.R_n + dt * U, X=X, z=z)


def _stage(mer: Meridian, dt: float, ops: Operators, predictor) -> Meridian:
    kin = kinematics(mer, dt, *predictor)
    ur, uz = solve_velocity(mer, dt, ops=ops, kin=kin)
    return advance(mer, dt, ur, uz, kin)


def step_extrapolated(mer: Meridian, dt: float, assembler_kwargs=None, disc_kwargs=None):
    """Second-order step: 2 x(two half steps) - x(one full step).

    Both paths move the same nodes (no resampling inside the step), so the two
    surfaces are combined node by node; the full step reuses the operators of the
    first half step.  Each stage uses the explicit velocity at its starting geometry
    as the predictor of the kinematics.  Returns the unresampled meridian and the mean
    neck speed.
    """
    ops0 = operators(mer, assembler_kwargs, disc_kwargs)
    pred0 = solve_velocity(mer, 0.0, ops=ops0)
    half = _stage(mer, 0.5 * dt, ops0, pred0)
    ops1 = operators(half, assembler_kwargs, disc_kwargs)
    hh = _stage(half, 0.5 * dt, ops1, solve_velocity(half, 0.0, ops=ops1))
    full = _stage(mer, dt, ops0, pred0)
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
        moved, U = step_extrapolated(mer, dt, cfg.assembler,
                                     dict(k=cfg.k, n_tip=cfg.n_tip, h_max=cfg.h_max))
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
