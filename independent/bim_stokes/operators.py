"""Collocation matrices of the single- and double-layer operators on the meridian.

Unknowns are nodal (u_r, u_z) at nodes 0..N of the upper half; the lower half is
the mirror image (u_r even, u_z odd).  For a target node i and nodal densities,

    (S g)_i = int_C M(x_i, x) g(x) dl,   (K u)_i = PV int_C Q(x_i, x) u(x) dl,

over the full closed meridian (upper half and its mirror).  Densities are
interpolated on each panel by cubic Lagrange polynomials in the chord parameter
sigma on the stencil (j-1, j, j+1, j+2), with mirror ghosts at the neck and the
pole.  Quadrature: Gauss-Legendre for well-separated panels; for a panel within
NEAR_FACTOR of its length from the target, a composite rule graded geometrically
towards the closest point; for a panel ending at the target, the same grading
down to 2^-LEVELS of the panel length, with source positions, normals and
N.(x - x_i) from the exact local expansion of the spline segment.
"""

from __future__ import annotations

import numpy as np

from functools import lru_cache

from scipy import sparse

from .geometry import Meridian, local_differences
from .kernels import ring_kernels


@lru_cache(maxsize=None)
def _gauss(order: int):
    return np.polynomial.legendre.leggauss(order)

ORDER_FAR = 10
ORDER_SUB = 8
LEVELS = 46
NEAR_FACTOR = 2.0
SIGN_MIRROR = np.array([1.0, -1.0])        # (r, z) components under z -> -z


class Assembler:
    def __init__(self, mer: Meridian, order_far: int = ORDER_FAR, order_sub: int = ORDER_SUB,
                 levels: int = LEVELS, near_factor: float = NEAR_FACTOR, chunk: int = 16):
        self.m = mer
        self.N = mer.n
        self.order_far, self.order_sub, self.levels = order_far, order_sub, levels
        self.near_factor, self.chunk = near_factor, chunk
        self._stencils()

    # --------------------------------------------------------------- basis
    def _stencils(self) -> None:
        N, s = self.N, self.m.sigma
        idx = np.empty((N, 4), dtype=int)
        sgn = np.ones((N, 4, 2))
        sig = np.empty((N, 4))
        for j in range(N):
            for q, m in enumerate(range(j - 1, j + 3)):
                if m < 0:                                   # mirror across the plane of node 1
                    idx[j, q], sig[j, q] = 1, -s[1]
                    sgn[j, q] = (1.0, -1.0)
                elif m > N:                                 # mirror across the axis of node N-1
                    idx[j, q], sig[j, q] = N - 1, 2.0 * s[N] - s[N - 1]
                    sgn[j, q] = (-1.0, 1.0)
                else:
                    idx[j, q], sig[j, q] = m, s[m]
        self.st_idx, self.st_sgn, self.st_sig = idx, sgn, sig

    def basis(self, j: int, sigma: np.ndarray) -> np.ndarray:
        """Cubic Lagrange weights (npts, 4) on the stencil of panel j."""
        p = self.st_sig[j]
        out = np.ones((sigma.size, 4))
        for a in range(4):
            for b in range(4):
                if a != b:
                    out[:, a] *= (sigma - p[b]) / (p[a] - p[b])
        return out

    # ------------------------------------------------------------ assembly
    def assemble(self):
        """Return S and K, each (2(N+1), 2(N+1)), rows (i, alpha), columns (node, beta)."""
        N = self.N
        n = N + 1
        S = np.zeros((2, n, 2, n))
        K = np.zeros((2, n, 2, n))
        mer = self.m
        xg, wg = _gauss(self.order_far)
        a, b = mer.sigma[:-1], mer.sigma[1:]
        L = b - a
        sq = 0.5 * L[:, None] * xg[None, :] + 0.5 * (a + b)[:, None]          # (N, q)
        wq = 0.5 * L[:, None] * wg[None, :]
        Xq, zq = mer.eval(sq)
        (_, _), (Nrq, Nzq), _, _, spq = mer.frame(sq)
        Bq = np.stack([self.basis(j, sq[j]) for j in range(N)])               # (N, q, 4)
        rq = Xq + mer.R_n
        Xi, zi, ri = mer.X, mer.z, mer.r
        near = self._near_pairs(Xq, zq, L)
        for i0 in range(0, n, self.chunk):
            I = np.arange(i0, min(n, i0 + self.chunk))
            for mirror in (False, True):
                zs = -zq if mirror else zq
                Nzs = -Nzq if mirror else Nzq
                dr = Xq[None] - Xi[I, None, None]
                dz = zs[None] - zi[I, None, None]
                Mk, Qk = ring_kernels(ri[I, None, None], rq[None], dr, dz, Nrq[None], Nzs[None])
                w = (wq * spq)[None] * (~near[mirror][I]).astype(float)[..., None]   # (nI, N, q)
                self._scatter(S, K, I, Mk, Qk, w, Bq, mirror)
        self._near_contributions(S, K, near, L)
        return S.reshape(2 * n, 2 * n), K.reshape(2 * n, 2 * n)

    def _basis_matrix(self, beta: int, mirror: bool):
        """Sparse (N*4, n) map from panel-stencil slots to nodal columns, with ghost signs."""
        key = (beta, mirror)
        cache = self.__dict__.setdefault("_bm", {})
        if key not in cache:
            N = self.N
            sm = SIGN_MIRROR[beta] if mirror else 1.0
            vals = (self.st_sgn[:, :, beta] * sm).ravel()
            rows = np.arange(N * 4)
            cache[key] = sparse.csr_matrix((vals, (rows, self.st_idx.ravel())), shape=(N * 4, N + 1))
        return cache[key]

    def _scatter(self, S, K, I, Mk, Qk, w, Bq, mirror):
        # Mk, Qk: (2, 2, nI, N, q); w: (nI, N, q); Bq: (N, q, 4)
        nI = len(I)
        for beta in range(2):
            P = self._basis_matrix(beta, mirror)
            for alpha in range(2):
                for mat, ker in ((S, Mk), (K, Qk)):
                    contrib = np.einsum("ijq,jqs->ijs", ker[alpha, beta] * w, Bq).reshape(nI, -1)
                    mat[alpha, I, beta, :] += np.asarray(contrib @ P)

    # ------------------------------------------------------- near and singular
    def _near_pairs(self, Xq, zq, L):
        """Boolean (n, N) masks per mirror flag: panel within near_factor * L of the target."""
        mer = self.m
        n = self.N + 1
        out = {}
        Xe = np.concatenate([Xq, mer.X[:-1, None], mer.X[1:, None]], axis=1)
        ze = np.concatenate([zq, mer.z[:-1, None], mer.z[1:, None]], axis=1)
        for mirror in (False, True):
            zz = -ze if mirror else ze
            d = np.hypot(Xe[None] - mer.X[:, None, None], zz[None] - mer.z[:, None, None]).min(axis=2)
            out[mirror] = d < self.near_factor * L[None, :]
        del n
        return out

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

    def _near_contributions(self, S, K, near, L) -> None:
        mer = self.m
        n = self.N + 1
        rows, cols_l, sg_l, B_l, w_l = [], [], [], [], []
        r0_l, r_l, dX_l, dz_l, Nr_l, Nz_l, c0_l = [], [], [], [], [], [], []
        for mirror in (False, True):
            sm = SIGN_MIRROR if mirror else np.ones(2)
            ii, jj = np.nonzero(near[mirror])
            for i, j in zip(ii, jj):
                a, b = mer.sigma[j], mer.sigma[j + 1]
                side = 0
                if not mirror and i == j:
                    side = +1
                elif not mirror and i == j + 1:
                    side = -1
                elif mirror and i == 0 and j == 0:
                    side = +1
                if side:
                    ea, eb = mer.local_expansion(i, side)
                    d, w = self._graded_offsets(0.0, b - a, 0.0, 0.0)
                    d = d * side
                    dX, dz, c0, speed, (Nr, Nz) = local_differences(ea, eb, d)
                    if mirror:
                        dz, Nz = -dz, -Nz
                    sigma = mer.sigma[i] + d
                else:
                    sfine = np.linspace(a, b, 33)
                    Xf, zf = mer.eval(sfine)
                    zf = -zf if mirror else zf
                    dist = np.hypot(Xf - mer.X[i], zf - mer.z[i])
                    k = int(np.argmin(dist))
                    sigma, w = self._graded_offsets(a, b, sfine[k], dist[k])
                    Xs, zs = mer.eval(sigma)
                    (_, _), (Nr, Nz), _, _, speed = mer.frame(sigma)
                    if mirror:
                        zs, Nz = -zs, -Nz
                    dX, dz = Xs - mer.X[i], zs - mer.z[i]
                    c0 = Nr * dX + Nz * dz
                m = sigma.size
                rows.append(np.full(m, i))
                B_l.append(self.basis(j, sigma))
                cols_l.append(np.broadcast_to(self.st_idx[j], (m, 4)))
                sg_l.append(np.broadcast_to(self.st_sgn[j] * sm[None, :], (m, 4, 2)))
                w_l.append(w * speed)
                r0_l.append(np.full(m, mer.r[i]))
                r_l.append(mer.r[i] + dX)
                dX_l.append(dX); dz_l.append(dz); Nr_l.append(Nr); Nz_l.append(Nz); c0_l.append(c0)
        if not rows:
            return
        cat = np.concatenate
        rows, B, cols, sg, w = cat(rows), cat(B_l), cat(cols_l), cat(sg_l), cat(w_l)
        Mk, Qk = ring_kernels(cat(r0_l), cat(r_l), cat(dX_l), cat(dz_l), cat(Nr_l), cat(Nz_l), c0=cat(c0_l))
        flat_rows = np.repeat(rows, 4)
        flat_cols = cols.ravel()
        for beta in range(2):
            for alpha in range(2):
                for mat, ker in ((S, Mk), (K, Qk)):
                    vals = ((ker[alpha, beta] * w)[:, None] * B * sg[:, :, beta]).ravel()
                    block = np.zeros(n * n)
                    np.add.at(block, flat_rows * n + flat_cols, vals)
                    mat[alpha, :, beta, :] += block.reshape(n, n)
