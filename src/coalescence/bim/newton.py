"""Backward-Euler BIM step solved exactly by a Jacobian-free Newton-Krylov method.

When U dt exceeds the tip radius by many orders of magnitude, the linearly
implicit step lands on the equilibrium of the problem linearised about the old
geometry (frozen single- and double-layer operators), not on the backward-Euler
solution, and the slaved tip shape carries a resolution-dependent bias.  Here the
step is the root of the exact backward-Euler condition.

Unknowns: the normal displacement eta_i of every node (i >= 1; the neck point is
the frame origin) along its old normal, relative to the reference curve that
moves radially with w U dt (with the neck near it, fixed in the laboratory far
from it), and the new neck speed U.  Residual, evaluated with a complete
boundary-integral solve on the new geometry y:

    F_i = (N_i^old . N_i(y)) eta_i - dt [u_i(y) - w_i U e_r] . N_i(y),
    F_U = U - u_r(P; y).

The disc nodes are scaled with R_n and not regenerated, so F is smooth in the
unknowns.  Unknowns and residuals are expressed in units of the local node
spacing (and of U), so finite-difference perturbations are relative at every
scale.  GMRES is right-preconditioned by the capillary response (what the
linearly implicit step captures); it supplies the geometric part.

The finite-difference Jacobian-vector products and the capped Krylov solves set
a residual floor.  An iteration that stalls (reduces the residual by less than a
factor 2) below ``stall_factor * tol`` is accepted and reported as stalled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres

from .geometry import Disc, Meridian, disc_for
from .operators import Assembler
from . import solver as S


@dataclass
class NewtonReport:
    converged: bool
    iterations: int
    residuals: list = field(default_factory=list)
    krylov: list = field(default_factory=list)
    evaluations: int = 0
    stalled: bool = False


def stalled(residuals, tol: float, stall_factor: float) -> bool:
    """True when the last iteration stagnated at a residual floor within stall_factor * tol."""
    if len(residuals) < 2 or residuals[-1] >= stall_factor * tol:
        return False
    return residuals[-1] > 0.5 * residuals[-2]


class BackwardEulerStep:
    def __init__(self, mer: Meridian, dt: float, disc_kwargs: dict, assembler_kwargs=None, jv_eps: float = 1e-6,
                 verbose: bool = False):
        self.verbose = verbose
        self.m, self.dt = mer, dt
        self.ak = assembler_kwargs or {}
        self.disc0 = disc_for(mer, **disc_kwargs)
        (_, _), (self.Nr0, self.Nz0), _, _, _ = mer.node_frame()
        self.w = S.translation_weight(mer)
        s = mer.sigma
        h = np.empty_like(s)
        h[1:-1] = 0.5 * (s[2:] - s[:-2])
        h[0], h[-1] = s[1] - s[0], s[-1] - s[-2]
        self.h = h
        self.n = mer.n + 1
        self.jv_eps = jv_eps
        self.evaluations = 0
        self.Us = 1.0

    # ---------------------------------------------------------------- mapping
    def unpack(self, x):
        eta = np.zeros(self.n)
        eta[1:] = x[:-1] * self.h[1:]
        return eta, x[-1] * self.Us

    def pack(self, eta, U):
        return np.concatenate([eta[1:] / self.h[1:], [U / self.Us]])

    def geometry(self, eta, U):
        m, dt = self.m, self.dt
        X = m.X - (1.0 - self.w) * U * dt + eta * self.Nr0
        z = m.z + eta * self.Nz0
        X[0], z[0] = 0.0, 0.0
        R = m.R_n + U * dt
        return Meridian(R_n=R, X=X, z=z), Disc(R_n=R, X=self.disc0.X * (R / m.R_n))

    # --------------------------------------------------------------- residual
    def residual(self, x):
        eta, U = self.unpack(x)
        y, disc = self.geometry(eta, U)
        ops = Assembler(y, disc, **self.ak).assemble()
        self.evaluations += 1
        ur, uz = S.solve_velocity(y, 0.0, ops=ops)
        (_, _), (Nry, Nzy), _, _, _ = y.node_frame()
        Fi = (self.Nr0 * Nry + self.Nz0 * Nzy) * eta - self.dt * (ur * Nry + uz * Nzy - self.w * U * Nry)
        FU = U - ur[0]
        return np.concatenate([Fi[1:] / self.h[1:], [FU / self.Us]]), (y, ops, ur, uz)

    # ---------------------------------------------------------- preconditioner
    def preconditioner(self, y: Meridian, ops, ur=None, uz=None, U=None):
        """LU of the scaled capillary-response Jacobian at geometry y."""
        n, dt = self.n, self.dt
        N, M, nr, nu, nf = ops.N, ops.M, ops.n_row, ops.n_u, ops.n_f
        k8 = 8.0 * math.pi
        sf_f = np.concatenate([np.arange(n), nf + np.arange(n)])
        Au = ops.C + ops.K / k8
        Af = -ops.S[:, nf + N + 1 + np.arange(M + 1)] / k8
        fixed = {nu + 0, N, N + M} | {nu + N + q for q in range(1, M + 1)}
        u_cols = np.array([c for c in range(2 * nu) if c not in fixed])
        rows = np.array([r for r in range(2 * nr) if r not in (N, N + M)])
        A0 = np.hstack([Au[np.ix_(rows, u_cols)], Af[rows]])
        (_, _), (Nr, Nz), _, _, _ = y.node_frame()
        J = y.curvature_jacobian()
        B = -(ops.S[:, sf_f] @ (np.vstack([np.diag(Nr), np.diag(Nz)]) @ J)) / k8   # traction -dkappa N
        X = lu_solve(lu_factor(A0), B[rows])
        Uf = np.zeros((2 * nu, n))
        Uf[u_cols] = X[:u_cols.size]
        dur, duz = Uf[:n], Uf[nu:nu + n]
        W = Nr[:, None] * dur + Nz[:, None] * duz           # d(u.N)_i / d eta_j through the curvature
        P = np.zeros((n + 1, n + 1))
        P[:n, :n] = np.diag(self.Nr0 * Nr + self.Nz0 * Nz) - dt * W
        if ur is not None:
            # rotation of the normal by a displacement gradient: dN = -(d eta/ds) t, so the
            # relative velocity projects on it with the tangential component v_t
            (tX, tz), _, _, _, _ = y.node_frame()
            vt = ur * tX + uz * tz - self.w * U * tX
            P[:n, :n] += dt * vt[:, None] * S.upwind_derivative(y, vt)
        P[:n, n] = dt * self.w * Nr
        P[n, :n] = -dur[0]
        P[n, n] = 1.0
        idx = np.arange(1, n + 1)
        D = np.concatenate([self.h[1:], [self.Us]])
        Ps = P[np.ix_(idx, idx)] * D[None, :] / D[:, None]
        return lu_factor(Ps)

    # ------------------------------------------------------------------ solve
    def solve(self, eta0, U0, tol=1e-6, maxit=8, krylov_rtol=1e-3, restart=30, maxkrylov=60, stall_factor=0.0):
        self.Us = max(abs(U0), 1e-3)
        x = self.pack(eta0, U0)
        F, info = self.residual(x)
        rep = NewtonReport(converged=False, iterations=0, residuals=[float(np.abs(F).max())])
        if self.verbose:
            print(f"   newton 0: |F|_max = {rep.residuals[0]:.3e}", flush=True)
        for it in range(maxit):
            if np.abs(F).max() < tol:
                rep.converged = True
                break
            y, ops, ur_y, uz_y = info
            plu = self.preconditioner(y, ops, ur_y, uz_y, self.unpack(x)[1])
            Minv = LinearOperator((x.size, x.size), matvec=lambda r: lu_solve(plu, r))
            Fx, xx = F.copy(), x.copy()

            def jv(v, Fx=Fx, xx=xx):
                nv = np.abs(v).max()
                if nv == 0.0:
                    return np.zeros_like(v)
                eps = self.jv_eps / nv
                F2, _ = self.residual(xx + eps * v)
                return (F2 - Fx) / eps

            Jop = LinearOperator((x.size, x.size), matvec=jv)
            count = [0]

            def cb(res):
                count[0] += 1
                if self.verbose:
                    print(f"      gmres {count[0]:3d}: preconditioned residual {float(res):.3e}", flush=True)

            dx, _ = gmres(Jop, -F, M=Minv, rtol=krylov_rtol, restart=restart, maxiter=max(1, maxkrylov // restart),
                          callback=cb, callback_type="pr_norm")
            rep.krylov.append(count[0])
            # backtracking on the max-norm of the residual
            lam = 1.0
            for _ in range(6):
                F_new, info_new = self.residual(x + lam * dx)
                if np.abs(F_new).max() < np.abs(F).max() or lam < 0.05:
                    break
                lam *= 0.5
            x, F, info = x + lam * dx, F_new, info_new
            rep.iterations = it + 1
            rep.residuals.append(float(np.abs(F).max()))
            if self.verbose:
                print(f"   newton {it + 1}: |F|_max = {rep.residuals[-1]:.3e} (step {lam:g}, {count[0]} krylov, {self.evaluations} evaluations)", flush=True)
            if rep.residuals[-1] >= tol and stalled(rep.residuals, tol, stall_factor):
                rep.converged = rep.stalled = True
                break
        rep.converged = rep.converged or np.abs(F).max() < tol
        rep.evaluations = self.evaluations
        eta, U = self.unpack(x)
        return info[0], U, rep


def step_backward_euler(mer: Meridian, dt: float, disc_kwargs: dict, assembler_kwargs=None,
                        tol: float = 1e-3, maxit: int = 6, stall_factor: float = 0.0):
    """One backward-Euler step solved by Newton-Krylov, from the linearly implicit prediction.

    Returns the unresampled new meridian, the new neck speed and the Newton report.
    """
    ops = S.operators(mer, assembler_kwargs, disc_kwargs)
    pred = S.solve_velocity(mer, 0.0, ops=ops)
    kin = S.kinematics(mer, dt, *pred)
    ur, uz = S.solve_velocity(mer, dt, ops=ops, kin=kin)
    (_, _), (Nr, Nz), _, _, _ = mer.node_frame()
    eta0 = kin.Ainv @ (dt * (ur * Nr + uz * Nz - kin.w * ur[0] * Nr))
    be = BackwardEulerStep(mer, dt, disc_kwargs, assembler_kwargs)
    return be.solve(eta0, float(ur[0]), tol=tol, maxit=maxit, stall_factor=stall_factor)
