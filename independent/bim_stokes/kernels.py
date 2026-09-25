"""Axisymmetric ring kernels of the Stokeslet and the stresslet.

For a target x0 = (r0, z0) and a source ring through x = (r, z) with outward
unit normal N = (N_r, N_z), the azimuthally integrated kernels are

    M_ab = int_0^{2pi} (e_a(x0))_i G_ij(x, x0) (e_b(x))_j r dphi,
    Q_ab = int_0^{2pi} (e_a(x0))_j T_ijk(x, x0) (e_b(x))_i N_k(x) r dphi,

with G_ij = delta_ij/rho + xh_i xh_j/rho^3, T_ijk = -6 xh_i xh_j xh_k/rho^5,
xh = x - x0 and a, b in {r, z}.  The single-layer velocity is
int M_ab g_b dl and the double-layer term int Q_ab u_b dl over the meridian.

Closed forms.  With t = sin^2(phi/2) the separation is
s = rho^2 = dd + 2 b t, where dd = (r - r0)^2 + (z - z0)^2 and b = 2 r r0, and
every numerator is a polynomial in t whose coefficients use the differences
d_r = r - r0 and d_z = z - z0.  The azimuthal integrals

    T(n, p) = int_0^{2pi} t^n s^{-p/2} dphi
            = 4 (a + b)^{-p/2} int_0^{pi/2} cos^{2n}(th) Delta^{-p} dth,

a + b = (r + r0)^2 + d_z^2, Delta^2 = 1 - k^2 sin^2(th), k^2 = 2b/(a+b), follow
from cos^2(th) = (Delta^2 - m1)/k^2 with m1 = 1 - k^2 = dd/(a + b) and the four
integrals of Delta^q (q = 1, -1, -3, -5) in terms of the complete elliptic
integrals.  No term is formed by subtracting two nearly equal quantities, so
the forms keep full relative precision as the points coalesce (m1 -> 0).
They lose precision as k -> 0 (well separated points, or points near the
axis), where the azimuthal integrand is smooth and the trapezoidal rule in phi
converges spectrally; that branch is used below K2_SPLIT.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ellipe, ellipkm1

K2_SPLIT = 0.05
N_PHI_SMOOTH = 64


def _delta_integrals(m1):
    """int_0^{pi/2} Delta^q for q = 1, -1, -3, -5, with m1 = 1 - k^2."""
    F = ellipkm1(m1)
    E = ellipe(1.0 - m1)
    i1 = E
    im1 = F
    im3 = E / m1
    im5 = (2.0 * (1.0 + m1) * E - m1 * F) / (3.0 * m1 * m1)
    return {1: i1, -1: im1, -3: im3, -5: im5}


def _t_integrals(apb, m1):
    """T(n, p) for the (n, p) pairs used by the kernels."""
    k2 = 1.0 - m1
    I = _delta_integrals(m1)
    out = {}
    binom = {0: [1], 1: [1, 1], 2: [1, 2, 1], 3: [1, 3, 3, 1]}
    for p, nmax in ((1, 1), (3, 2), (5, 3)):
        pref = 4.0 * apb ** (-p / 2.0)
        for n in range(nmax + 1):
            acc = 0.0
            for l in range(n + 1):
                acc = acc + binom[n][l] * (-m1) ** (n - l) * I[2 * l - p]
            out[(n, p)] = pref * acc / k2**n
    return out


def _closed_forms(r0, r, dr, dz, Nr, Nz, c0):
    apb = (r + r0) ** 2 + dz * dz
    dd = dr * dr + dz * dz
    m1 = dd / apb
    T = _t_integrals(apb, m1)
    M = np.empty((2, 2) + np.shape(r))
    M[0, 0] = r * (T[(0, 1)] - 2.0 * T[(1, 1)] + dr * dr * T[(0, 3)] - 2.0 * dr * dr * T[(1, 3)]
                   - 4.0 * r * r0 * T[(2, 3)])
    M[0, 1] = r * dz * (dr * T[(0, 3)] - 2.0 * r * T[(1, 3)])
    M[1, 0] = r * dz * (dr * T[(0, 3)] + 2.0 * r0 * T[(1, 3)])
    M[1, 1] = r * (T[(0, 1)] + dz * dz * T[(0, 3)])
    c1 = 2.0 * Nr * r0
    t5 = [T[(n, 5)] for n in range(4)]
    Q = np.empty((2, 2) + np.shape(r))
    # Q_rr: (d_r - 2 r t)(d_r + 2 r0 t)(c0 + c1 t)
    Q[0, 0] = (dr * dr * c0 * t5[0] + (dr * dr * c1 - 2.0 * dr * dr * c0) * t5[1]
               + (-2.0 * dr * dr * c1 - 4.0 * r * r0 * c0) * t5[2] - 4.0 * r * r0 * c1 * t5[3])
    # Q_rz: target r, source z: (d_r - 2 r t) d_z (c0 + c1 t)
    Q[0, 1] = dz * (dr * c0 * t5[0] + (dr * c1 - 2.0 * r * c0) * t5[1] - 2.0 * r * c1 * t5[2])
    # Q_zr: target z, source r: d_z (d_r + 2 r0 t)(c0 + c1 t)
    Q[1, 0] = dz * (dr * c0 * t5[0] + (dr * c1 + 2.0 * r0 * c0) * t5[1] + 2.0 * r0 * c1 * t5[2])
    # Q_zz: d_z^2 (c0 + c1 t)
    Q[1, 1] = dz * dz * (c0 * t5[0] + c1 * t5[1])
    Q *= -6.0 * r
    return M, Q


def _integrand(phi, r0, r, dr, dz, Nr, Nz):
    """Azimuthal integrands of M and Q at angles phi (last axis)."""
    sh = np.sin(0.5 * phi) ** 2
    cph, sph = np.cos(phi), np.sin(phi)
    x1 = dr[..., None] - 2.0 * r[..., None] * sh          # xh . e_r(x0)
    xr = dr[..., None] + 2.0 * r0[..., None] * sh          # xh . e_r(x)
    zz = dz[..., None]
    s = (dr * dr + dz * dz)[..., None] + 4.0 * (r * r0)[..., None] * sh
    rs = np.sqrt(s)
    inv1, inv3, inv5 = 1.0 / rs, 1.0 / (s * rs), 1.0 / (s * s * rs)
    rr = r[..., None]
    m = np.empty((2, 2) + x1.shape)
    m[0, 0] = rr * (cph * inv1 + x1 * xr * inv3)
    m[0, 1] = rr * x1 * zz * inv3
    m[1, 0] = rr * zz * xr * inv3
    m[1, 1] = rr * (inv1 + zz * zz * inv3)
    xn = Nr[..., None] * xr + Nz[..., None] * zz
    q = np.empty_like(m)
    q[0, 0] = xr * x1 * xn
    q[0, 1] = zz * x1 * xn
    q[1, 0] = xr * zz * xn
    q[1, 1] = zz * zz * xn
    q *= -6.0 * rr * inv5
    del sph
    return m, q


def _trapezoid(r0, r, dr, dz, Nr, Nz, n_phi=N_PHI_SMOOTH):
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    m, q = _integrand(phi, r0, r, dr, dz, Nr, Nz)
    w = 2.0 * np.pi / n_phi
    return m.sum(axis=-1) * w, q.sum(axis=-1) * w


def ring_kernels(r0, r, dr, dz, Nr, Nz, c0=None):
    """Ring kernels M and Q, shape (2, 2) + broadcast shape of the inputs.

    dr = r - r0 and dz = z - z0 must be supplied as accurate differences, and
    c0 = N . (x - x0) may be supplied when it is known more accurately than the
    product formed here (singular panels, where it is O(kappa ds^2)).
    """
    r0, r, dr, dz, Nr, Nz = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (r0, r, dr, dz, Nr, Nz)))
    if c0 is None:
        c0 = Nr * dr + Nz * dz
    c0 = np.broadcast_to(np.asarray(c0, dtype=float), r.shape)
    shape = r.shape
    M = np.empty((2, 2) + shape)
    Q = np.empty((2, 2) + shape)
    apb = (r + r0) ** 2 + dz * dz
    k2 = 4.0 * r * r0 / np.where(apb > 0, apb, 1.0)
    far = k2 < K2_SPLIT
    near = ~far
    if np.any(near):
        Mn, Qn = _closed_forms(r0[near], r[near], dr[near], dz[near], Nr[near], Nz[near], c0[near])
        M[:, :, near] = Mn
        Q[:, :, near] = Qn
    if np.any(far):
        Mf, Qf = _trapezoid(r0[far], r[far], dr[far], dz[far], Nr[far], Nz[far])
        M[:, :, far] = Mf
        Q[:, :, far] = Qf
    return M, Q


def ring_kernels_reference(r0, r, dr, dz, Nr, Nz, n_panels=48, order=24):
    """Independent reference: direct azimuthal quadrature of the 3D kernels.

    The integrand peaks at phi = 0 with width ~ rho/sqrt(r r0); the substitution
    tan(phi/2) = eps sinh(v), eps = rho/(2 sqrt(r r0)), resolves it uniformly
    for any separation.  Composite Gauss-Legendre in v on [0, V(phi_c)] and in
    phi on [phi_c, pi]; the integrand is even in phi.  Scalars only.
    """
    r0, r, dr, dz, Nr, Nz = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (r0, r, dr, dz, Nr, Nz))
    x, w = np.polynomial.legendre.leggauss(order)
    rho = np.sqrt(dr * dr + dz * dz)
    phi_c = 0.5
    total_m = np.zeros((2, 2))
    total_q = np.zeros((2, 2))
    if r0[0] > 0 and r[0] > 0:
        eps = rho[0] / (2.0 * np.sqrt(r[0] * r0[0]))
        V = np.arcsinh(np.tan(0.5 * phi_c) / eps)
        edges = np.linspace(0.0, V, n_panels + 1)
        for a_, b_ in zip(edges[:-1], edges[1:]):
            v = 0.5 * (b_ - a_) * x + 0.5 * (b_ + a_)
            phi = 2.0 * np.arctan(eps * np.sinh(v))
            jac = 2.0 * eps * np.cosh(v) / (1.0 + (eps * np.sinh(v)) ** 2)
            m, q = _integrand(phi, r0, r, dr, dz, Nr, Nz)
            ww = 0.5 * (b_ - a_) * w * jac
            total_m += (m[..., 0, :] * ww).sum(axis=-1)
            total_q += (q[..., 0, :] * ww).sum(axis=-1)
        lo = phi_c
    else:
        lo = 0.0
    edges = np.linspace(lo, np.pi, n_panels + 1)
    for a_, b_ in zip(edges[:-1], edges[1:]):
        phi = 0.5 * (b_ - a_) * x + 0.5 * (b_ + a_)
        m, q = _integrand(phi, r0, r, dr, dz, Nr, Nz)
        ww = 0.5 * (b_ - a_) * w
        total_m += (m[..., 0, :] * ww).sum(axis=-1)
        total_q += (q[..., 0, :] * ww).sum(axis=-1)
    return 2.0 * total_m, 2.0 * total_q
