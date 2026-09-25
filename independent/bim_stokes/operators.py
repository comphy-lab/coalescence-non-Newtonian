"""Collocation matrices of the half-body boundary-integral equation.

The upper drop (z > 0) is bounded by its free surface S_f, from the neck point
P = (R_n, 0) to the pole, and by the disc D = {z = 0, r <= R_n} of the symmetry
plane.  With the free-space kernels, for x0 on the boundary,

    c(x0) u(x0) + (1/8 pi) PV int u_i T_ijk N_k dS = (1/8 pi) int f_i G_ij dS,

with c = I/2 at smooth points.  On S_f the traction f = -kappa N is known; on D
the symmetry gives u_z = 0 and f_r = 0, while u_r and f_z are unknown.

The free-slip image across z = 0 is deliberately not used: it would turn the
thin void gap ahead of the neck into a crack between the surface and its image,
across which velocity jumps are nearly invisible to the second-kind equation
(the smallest singular value then falls roughly as R0^3.5).  Here nothing lies
across the gap.  At P the boundary has a right-angle edge whose free-term
coefficient is the exact right-angle value (checked against the discrete
translation and straining identities).

Unknown numbering.  Rows (collocation points) and velocity nodes: free-surface
nodes 0..N (0 = P), then disc nodes 1..M (M on the axis).  Traction nodes:
free-surface nodes 0..N, then disc nodes 0..M (the disc traction at P is
distinct from the free-surface traction there).  Matrices are ordered
component-major: index = alpha * n + node, alpha = 0 (r), 1 (z).

Densities are interpolated on each panel by cubic Lagrange polynomials on the
stencil (j-1, j, j+1, j+2), one-sided at P and with mirror ghosts across the
axis.  Quadrature: Gauss-Legendre on well-separated panels; for a panel within
NEAR_FACTOR of its length from the target, a composite rule graded geometrically
towards the closest point; for a panel ending at the target, the same grading
down to 2^-LEVELS of the panel length, with source positions, normals and
N.(x - x_i) from the exact local expansion of the segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import sparse

from .geometry import Disc, Meridian, local_differences
from .kernels import ring_kernels


@lru_cache(maxsize=None)
def _gauss(order: int):
    return np.polynomial.legendre.leggauss(order)


ORDER_FAR = 10
ORDER_SUB = 8
LEVELS = 46
NEAR_FACTOR = 2.0
AXIS_SIGN = np.array([-1.0, 1.0])          # (r, z) parity of a vector field about the axis


def _lagrange(p: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    out = np.ones((sigma.size, 4))
    for a in range(4):
        for b in range(4):
            if a != b:
                out[:, a] *= (sigma - p[b]) / (p[a] - p[b])
    return out


def _stencils(s: np.ndarray):
    """Four-node stencils per panel: one-sided at node 0, axis ghost beyond the last node."""
    N = len(s) - 1
    if N < 3:
        raise ValueError("a boundary piece needs at least four nodes")
    idx = np.empty((N, 4), dtype=int)
    sgn = np.ones((N, 4, 2))
    sig = np.empty((N, 4))
    for j in range(N):
        lo = 0 if j == 0 else j - 1
        for q, m in enumerate(range(lo, lo + 4)):
            if m > N:
                idx[j, q], sig[j, q] = 2 * N - m, 2.0 * s[N] - s[2 * N - m]
                sgn[j, q] = AXIS_SIGN
            else:
                idx[j, q], sig[j, q] = m, s[m]
    return idx, sgn, sig


class _Piece:
    """Panels of one boundary component and the maps from stencil slots to unknowns."""

    def __init__(self, name, sigma, geom, u_dof, f_dof, row_of_node):
        self.name = name
        self.s = sigma
        self.geom = geom
        self.a, self.b = sigma[:-1], sigma[1:]
        self.L = self.b - self.a
        self.st_idx, self.st_sgn, self.st_sig = _stencils(sigma)
        self.u_dof = np.asarray(u_dof)
        self.f_dof = np.asarray(f_dof)
        self.row_of_node = np.asarray(row_of_node)
        self.node_of_row = {int(r): q for q, r in enumerate(self.row_of_node)}
        self._bm = {}

    @property
    def n_pan(self) -> int:
        return len(self.a)

    def basis(self, j: int, sigma: np.ndarray) -> np.ndarray:
        return _lagrange(self.st_sig[j], sigma)

    def basis_matrix(self, beta: int, kind: str, n_cols: int):
        key = (beta, kind)
        if key not in self._bm:
            dof = self.u_dof if kind == "u" else self.f_dof
            vals = self.st_sgn[:, :, beta].ravel()
            rows = np.arange(self.n_pan * 4)
            self._bm[key] = sparse.csr_matrix((vals, (rows, dof[self.st_idx].ravel())),
                                              shape=(self.n_pan * 4, n_cols))
        return self._bm[key]

    def singular_side(self, row: int, j: int) -> int:
        """+1 if panel j starts at the target node, -1 if it ends there, else 0."""
        q = self.node_of_row.get(row)
        if q is None:
            return 0
        if j == q:
            return +1
        if j == q - 1:
            return -1
        return 0


@dataclass
class Operators:
    """Assembled half-body operators; see the module docstring for the numbering."""

    K: np.ndarray        # (2 n_row, 2 n_u)   double layer
    S: np.ndarray        # (2 n_row, 2 n_f)   single layer
    C: np.ndarray        # (2 n_row, 2 n_u)   free term
    mer: Meridian
    disc: Disc

    @property
    def N(self) -> int:
        return self.mer.n

    @property
    def M(self) -> int:
        return self.disc.n

    @property
    def n_row(self) -> int:
        return self.N + 1 + self.M

    n_u = n_row

    @property
    def n_f(self) -> int:
        return self.N + 2 + self.M

    def velocity_points(self):
        """(r, z) of the velocity nodes (= collocation points)."""
        return (np.concatenate([self.mer.r, self.disc.r[1:]]),
                np.concatenate([self.mer.z, np.zeros(self.M)]))

    def traction_points(self):
        """(r, z, N_r, N_z) of the traction nodes."""
        (_, _), (Nr, Nz), _, _, _ = self.mer.node_frame()
        M1 = self.M + 1
        return (np.concatenate([self.mer.r, self.disc.r]), np.concatenate([self.mer.z, np.zeros(M1)]),
                np.concatenate([Nr, np.zeros(M1)]), np.concatenate([Nz, -np.ones(M1)]))


class Assembler:
    def __init__(self, mer: Meridian, disc: Disc, order_far: int = ORDER_FAR, order_sub: int = ORDER_SUB,
                 levels: int = LEVELS, near_factor: float = NEAR_FACTOR, chunk: int = 16):
        if disc.R_n != mer.R_n:
            raise ValueError("disc and meridian disagree on the neck radius")
        self.m, self.d = mer, disc
        self.order_far, self.order_sub, self.levels = order_far, order_sub, levels
        self.near_factor, self.chunk = near_factor, chunk
        N, M = mer.n, disc.n
        self.N, self.M = N, M
        self.n_row = self.n_u = N + 1 + M
        self.n_f = N + 2 + M
        self.tX = np.concatenate([mer.X, disc.X[1:]])
        self.tz = np.concatenate([mer.z, np.zeros(M)])
        self.tr = self.tX + mer.R_n

        def surf_geom(sig):
            X, z = mer.eval(sig)
            (_, _), (Nr, Nz), _, _, sp = mer.frame(sig)
            return X, z, Nr, Nz, sp

        self.free = _Piece("free", mer.sigma, surf_geom, u_dof=np.arange(N + 1), f_dof=np.arange(N + 1),
                           row_of_node=np.arange(N + 1))
        disc_rows = np.concatenate([[0], N + np.arange(1, M + 1)])
        self.disc = _Piece("disc", disc.sigma, Disc.geom, u_dof=disc_rows, f_dof=N + 1 + np.arange(M + 1),
                           row_of_node=disc_rows)

    # ------------------------------------------------------------ assembly
    def assemble(self) -> Operators:
        nr, nu, nf = self.n_row, self.n_u, self.n_f
        K = np.zeros((2, nr, 2, nu))
        S = np.zeros((2, nr, 2, nf))
        xg, wg = _gauss(self.order_far)
        for pc in (self.free, self.disc):
            a, b, L = pc.a, pc.b, pc.L
            sq = 0.5 * L[:, None] * xg[None, :] + 0.5 * (a + b)[:, None]
            wq = 0.5 * L[:, None] * wg[None, :]
            Xq, zq, Nrq, Nzq, spq = pc.geom(sq)
            Bq = np.stack([pc.basis(j, sq[j]) for j in range(pc.n_pan)])
            near = self._near_pairs(pc, Xq, zq)
            rq = Xq + self.m.R_n
            for i0 in range(0, nr, self.chunk):
                I = np.arange(i0, min(nr, i0 + self.chunk))
                dr = Xq[None] - self.tX[I, None, None]
                dz = zq[None] - self.tz[I, None, None]
                Mk, Qk = ring_kernels(self.tr[I, None, None], rq[None], dr, dz, Nrq[None], Nzq[None])
                w = (wq * spq)[None] * (~near[I]).astype(float)[..., None]
                self._scatter(K, S, I, Mk, Qk, w, Bq, pc)
            self._near_contributions(K, S, pc, near)
        K = K.reshape(2 * nr, 2 * nu)
        S = S.reshape(2 * nr, 2 * nf)
        ops = Operators(K=K, S=S, C=np.zeros((2 * nr, 2 * nu)), mer=self.m, disc=self.d)
        ops.C = self._free_term(ops)
        return ops

    def _scatter(self, K, S, I, Mk, Qk, w, Bq, pc):
        nI = len(I)
        for beta in range(2):
            Pu = pc.basis_matrix(beta, "u", self.n_u)
            Pf = pc.basis_matrix(beta, "f", self.n_f)
            for alpha in range(2):
                cq = np.einsum("ijq,jqs->ijs", Qk[alpha, beta] * w, Bq).reshape(nI, -1)
                K[alpha, I, beta, :] += np.asarray(cq @ Pu)
                cm = np.einsum("ijq,jqs->ijs", Mk[alpha, beta] * w, Bq).reshape(nI, -1)
                S[alpha, I, beta, :] += np.asarray(cm @ Pf)

    # ------------------------------------------------------- near and singular
    def _near_pairs(self, pc: _Piece, Xq, zq):
        """Boolean (n_row, n_pan): panel within near_factor * L of the target."""
        Xa, za = pc.geom(pc.a)[:2]
        Xb, zb = pc.geom(pc.b)[:2]
        Xe = np.concatenate([Xq, Xa[:, None], Xb[:, None]], axis=1)
        ze = np.concatenate([zq, za[:, None], zb[:, None]], axis=1)
        d = np.hypot(Xe[None] - self.tX[:, None, None], ze[None] - self.tz[:, None, None]).min(axis=2)
        return d < self.near_factor * pc.L[None, :]

    def _graded_offsets(self, lo: float, hi: float, centre: float, delta: float):
        """Composite Gauss points on [lo, hi] graded geometrically about centre."""
        xs, ws = _gauss(self.order_sub)
        span = hi - lo
        eps0 = max(0.5 * delta, span * 2.0 ** (-self.levels))
        brk = {lo, hi}
        for side in (-1.0, 1.0):
            e = eps0
            while True:
                p = centre + side * e
                if p <= lo or p >= hi:
                    break
                brk.add(p)
                e *= 2.0
        brk = np.array(sorted(brk))
        a, b = brk[:-1], brk[1:]
        pts = (0.5 * (b - a)[:, None] * xs[None] + 0.5 * (a + b)[:, None]).ravel()
        wts = (0.5 * (b - a)[:, None] * ws[None]).ravel()
        return pts, wts

    def _near_contributions(self, K, S, pc: _Piece, near) -> None:
        mer = self.m
        rows, B_l, cu_l, cf_l, sg_l, w_l = [], [], [], [], [], []
        r0_l, r_l, dX_l, dz_l, Nr_l, Nz_l, c0_l = [], [], [], [], [], [], []
        for i, j in zip(*np.nonzero(near)):
            a, b = pc.a[j], pc.b[j]
            side = pc.singular_side(int(i), int(j))
            if side:
                q = pc.node_of_row[int(i)]
                d, w = self._graded_offsets(0.0, b - a, 0.0, 0.0)
                d = d * side
                sigma = pc.s[q] + d
                if pc is self.free:
                    ea, eb = mer.local_expansion(q, side)
                    dX, dz, c0, speed, (Nr, Nz) = local_differences(ea, eb, d)
                else:                                   # straight disc, target on it
                    dX, dz, Nr, Nz, speed = -d, 0.0 * d, 0.0 * d, -np.ones_like(d), np.ones_like(d)
                    c0 = 0.0 * d
            else:
                sfine = np.linspace(a, b, 33)
                Xf, zf = pc.geom(sfine)[:2]
                dist = np.hypot(Xf - self.tX[i], zf - self.tz[i])
                k = int(np.argmin(dist))
                sigma, w = self._graded_offsets(a, b, sfine[k], dist[k])
                Xs, zs, Nr, Nz, speed = pc.geom(sigma)
                dX, dz = Xs - self.tX[i], zs - self.tz[i]
                c0 = Nr * dX + Nz * dz
            m = sigma.size
            rows.append(np.full(m, i))
            B_l.append(pc.basis(j, sigma))
            cu_l.append(np.broadcast_to(pc.u_dof[pc.st_idx[j]], (m, 4)))
            cf_l.append(np.broadcast_to(pc.f_dof[pc.st_idx[j]], (m, 4)))
            sg_l.append(np.broadcast_to(pc.st_sgn[j], (m, 4, 2)))
            w_l.append(w * speed)
            r0_l.append(np.full(m, self.tr[i]))
            r_l.append(self.tr[i] + dX)
            dX_l.append(dX); dz_l.append(dz); Nr_l.append(Nr); Nz_l.append(Nz); c0_l.append(c0)
        if not rows:
            return
        cat = np.concatenate
        rows, B, cu, cf, sg, w = cat(rows), cat(B_l), cat(cu_l), cat(cf_l), cat(sg_l), cat(w_l)
        Mk, Qk = ring_kernels(cat(r0_l), cat(r_l), cat(dX_l), cat(dz_l), cat(Nr_l), cat(Nz_l), c0=cat(c0_l))
        flat_rows = np.repeat(rows, 4)
        for beta in range(2):
            for alpha in range(2):
                for mat, ker, cols, n in ((K, Qk, cu, self.n_u), (S, Mk, cf, self.n_f)):
                    vals = ((ker[alpha, beta] * w)[:, None] * B * sg[:, :, beta]).ravel()
                    block = np.zeros(self.n_row * n)
                    np.add.at(block, flat_rows * n + cols.ravel(), vals)
                    mat[alpha, :, beta, :] += block.reshape(self.n_row, n)

    # ------------------------------------------------------------- free term
    def _free_term(self, ops: Operators) -> np.ndarray:
        """I/2 at smooth points; the exact right-angle edge coefficient at P.

        At P the free surface leaves the disc vertically (tangent +e_z) while the
        disc runs towards the axis (tangent -e_r): a right-angle edge with the
        fluid in r < R_n, z > 0 whatever the neck radius.  Its free-term
        coefficient is purely local, EDGE_COEFFICIENT below; edge_coefficient_
        from_identities() recovers it from the discrete operators (to 1e-6 on a
        hemisphere).  Fitting it from the identities in production is not
        accurate at small R_n: the straining flow has u_r(P) = R_n, so the fit
        divides quadrature-level residuals by R_n (about 4e-4 error at
        R_n = 1e-6, enough to make the neck point lag its neighbours and the tip
        sharpen without bound).
        """
        nr, nu = self.n_row, self.n_u
        C = np.zeros((2 * nr, 2 * nu))
        i = np.arange(nr)
        C[i, i] = 0.5
        C[nr + i, nu + i] = 0.5
        c = EDGE_COEFFICIENT
        C[0, 0], C[0, nu], C[nr, 0], C[nr, nu] = c[0, 0], c[0, 1], c[1, 0], c[1, 1]
        return C


EDGE_COEFFICIENT = np.array([[0.25, -0.5 / math.pi], [-0.5 / math.pi, 0.25]])


def edge_coefficient_from_identities(ops: Operators) -> np.ndarray:
    """c(P) from the discrete operators: exact for the translation e_z and the straining flow (r, -2z)."""
    nr, nu = ops.n_row, ops.n_u
    ru, zu = ops.velocity_points()
    _, _, Nr, Nz = ops.traction_points()
    k8 = 8.0 * math.pi
    Ku = ops.K @ np.concatenate([np.zeros(nu), np.ones(nu)]) / k8
    c_rz, c_zz = -Ku[0], -Ku[nr]
    us = np.concatenate([ru, -2.0 * zu])
    res = (ops.S @ np.concatenate([2.0 * Nr, -4.0 * Nz]) - ops.K @ us) / k8
    return np.array([[res[0] / us[0], c_rz], [res[nr] / us[0], c_zz]])
