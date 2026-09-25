"""Meridian of the merged axisymmetric drop, in a neck-anchored frame.

The upper half of the meridian runs from the neck point (r, z) = (R_n, 0) on the
symmetry plane to the pole (0, z_p) on the axis.  Radial positions are stored as
X = r - R_n, so that near the neck coordinates carry round-off relative to the
local scale rather than to R_n.  The curve is a parametric cubic spline in
cumulative chord length sigma, built on the nodes extended by mirror images
across the plane (X even, z odd about the neck) and across the axis (r odd, z
even about the pole), so that the symmetry conditions hold to spline accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

N_GHOST = 3


def chord_parameter(X: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X), np.diff(z)))])


@dataclass
class Meridian:
    R_n: float
    X: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float).copy()
        self.z = np.asarray(self.z, dtype=float).copy()
        if self.X[0] != 0.0 or self.z[0] != 0.0:
            raise ValueError("node 0 must be the neck at the frame origin")
        self.X[-1] = -self.R_n                     # pole on the axis
        self.sigma = chord_parameter(self.X, self.z)
        self._build_splines()

    # ------------------------------------------------------------ splines
    def _build_splines(self) -> None:
        s, X, z, R = self.sigma, self.X, self.z, self.R_n
        g = N_GHOST
        s_lo = -s[g:0:-1]
        X_lo = X[g:0:-1]
        z_lo = -z[g:0:-1]
        sN = s[-1]
        s_hi = 2.0 * sN - s[-2:-2 - g:-1]
        r_hi = -(X[-2:-2 - g:-1] + R)
        X_hi = r_hi - R
        z_hi = z[-2:-2 - g:-1]
        self._s_ext = np.concatenate([s_lo, s, s_hi])
        self._off = g
        self.spX = CubicSpline(self._s_ext, np.concatenate([X_lo, X, X_hi]))
        self.spz = CubicSpline(self._s_ext, np.concatenate([z_lo, z, z_hi]))

    @property
    def n(self) -> int:
        return len(self.X) - 1

    @property
    def r(self) -> np.ndarray:
        return self.X + self.R_n

    def segment_coefficients(self, j: int):
        """Cubic coefficients of the spline on [sigma_j, sigma_{j+1}], highest power first."""
        k = j + self._off
        return self.spX.c[:, k], self.spz.c[:, k]

    def eval(self, sigma, nu: int = 0):
        return self.spX(sigma, nu), self.spz(sigma, nu)

    # ------------------------------------------------------ differential geometry
    def frame(self, sigma):
        """Unit tangent, outward unit normal, meridional and azimuthal curvature."""
        X1, z1 = self.eval(sigma, 1)
        X2, z2 = self.eval(sigma, 2)
        speed = np.hypot(X1, z1)
        tX, tz = X1 / speed, z1 / speed
        Nr, Nz = tz, -tX
        k1 = (X1 * z2 - z1 * X2) / speed**3
        X0, _ = self.eval(sigma, 0)
        r = X0 + self.R_n
        with np.errstate(divide="ignore", invalid="ignore"):
            k2 = np.where(r > 0.0, Nr / np.where(r > 0.0, r, 1.0), k1)
        return (tX, tz), (Nr, Nz), k1, k2, speed

    def node_frame(self):
        return self.frame(self.sigma)

    def neck_curvature(self) -> tuple[float, float]:
        """Meridional curvature and |2H| = |k1 + k2| at the neck."""
        _, _, k1, k2, _ = self.frame(np.array([0.0]))
        return float(k1[0]), float(abs(k1[0] + k2[0]))

    def tip_radius(self) -> float:
        k1, _ = self.neck_curvature()
        return 1.0 / abs(k1)

    def curvature_jacobian(self) -> np.ndarray:
        """J_ij = d(k1 + k2)_i / d eta_j for normal displacements eta of the nodes.

        The linearisation of exactly the spline curvature used for the traction:
        the perturbation eta_j N_j is interpolated by the same ghost-extended
        spline, at fixed parameter, so that the implicit capillary term has the
        stiffness of the discrete curvature operator (a three-point Laplacian
        underestimates it threefold for the grid-scale mode).
        """
        g, n = N_GHOST, self.n + 1
        s = self.sigma
        (_, _), (Nr, Nz), k1, _, sp = self.frame(s)
        Er, Ez = np.diag(Nr), np.diag(Nz)
        Yr = np.concatenate([Er[g:0:-1], Er, -Er[-2:-2 - g:-1]])      # r even at the neck, odd at the axis
        Yz = np.concatenate([-Ez[g:0:-1], Ez, Ez[-2:-2 - g:-1]])      # z odd at the neck, even at the axis
        spr, spz = CubicSpline(self._s_ext, Yr, axis=0), CubicSpline(self._s_ext, Yz, axis=0)
        dr1, dr2, dz1, dz2 = spr(s, 1), spr(s, 2), spz(s, 1), spz(s, 2)
        X1, z1 = (v[:, None] for v in self.eval(s, 1))
        X2, z2 = (v[:, None] for v in self.eval(s, 2))
        q = sp[:, None]
        dot = X1 * dr1 + z1 * dz1
        dk1 = (dr1 * z2 + X1 * dz2 - dz1 * X2 - z1 * dr2) / q**3 - 3.0 * k1[:, None] * dot / q**2
        r = self.r[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            dk2 = dz1 / (q * r) - z1 * dot / (q**3 * r) - (z1 / (q * r * r)) * Er
        dk2[-1] = dk1[-1]                                           # k2 = k1 on the axis
        return dk1 + dk2

    def volume(self, order: int = 16) -> float:
        """V = (2 pi / 3) oint r (x . N) dl over the full meridian (twice the upper half)."""
        x, w = np.polynomial.legendre.leggauss(order)
        total = 0.0
        for j in range(self.n):
            a, b = self.sigma[j], self.sigma[j + 1]
            s = 0.5 * (b - a) * x + 0.5 * (b + a)
            X, z = self.eval(s)
            (_, _), (Nr, Nz), _, _, speed = self.frame(s)
            r = X + self.R_n
            total += np.sum(w * 0.5 * (b - a) * speed * r * (r * Nr + z * Nz))
        return 2.0 * (2.0 * math.pi / 3.0) * total

    # ------------------------------------------------- local expansions (singular panels)
    def local_expansion(self, i: int, side: int):
        """Taylor coefficients (a1, a2, a3), (b1, b2, b3) of X and z about node i.

        side = +1 uses segment i (towards the pole), side = -1 segment i-1.  The
        expansion is exact for the cubic segment: X(sigma_i + d) - X_i =
        a1 d + a2 d^2 + a3 d^3 with d > 0 for side = +1 and d < 0 for side = -1.
        """
        if side > 0:
            cX, cz = self.segment_coefficients(i)
            return (cX[2], cX[1], cX[0]), (cz[2], cz[1], cz[0])
        cX, cz = self.segment_coefficients(i - 1)
        h = self.sigma[i] - self.sigma[i - 1]

        def about_right(c):
            c3, c2, c1, c0 = c[0], c[1], c[2], c[3]   # highest power first: c3 u^3 + ...
            d1 = c1 + 2.0 * c2 * h + 3.0 * c3 * h * h
            d2 = c2 + 3.0 * c3 * h
            return d1, d2, c3

        return about_right(cX), about_right(cz)


def local_differences(a, b, d):
    """Exact differences and N . (x - x_i) for offsets d about a node.

    Returns dX, dz, c0 and the speed |x'| at the offset points.  c0 is formed
    from the cancellation-free series (a1 b2 - a2 b1) d^2 + 2 (a1 b3 - a3 b1) d^3
    + (a2 b3 - a3 b2) d^4, divided by the local speed.
    """
    a1, a2, a3 = a
    b1, b2, b3 = b
    dX = d * (a1 + d * (a2 + d * a3))
    dz = d * (b1 + d * (b2 + d * b3))
    Xp = a1 + d * (2.0 * a2 + 3.0 * a3 * d)
    zp = b1 + d * (2.0 * b2 + 3.0 * b3 * d)
    speed = np.hypot(Xp, zp)
    c0 = d * d * ((a1 * b2 - a2 * b1) + d * (2.0 * (a1 * b3 - a3 * b1) + d * (a2 * b3 - a3 * b2))) / speed
    return dX, dz, c0, speed, (zp / speed, -Xp / speed)


@dataclass
class Disc:
    """The symmetry-plane disc z = 0, 0 <= r <= R_n, bounding the upper drop.

    Node 0 is the neck point P (X = 0), node M is on the axis (X = -R_n).  The
    parameter is the distance from P, sigma = -X; the outward normal of the
    upper drop is -e_z.
    """

    R_n: float
    X: np.ndarray

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float).copy()
        if self.X[0] != 0.0:
            raise ValueError("disc node 0 must be the neck point")
        self.X[-1] = -self.R_n
        self.sigma = -self.X

    @property
    def n(self) -> int:
        return len(self.X) - 1

    @property
    def r(self) -> np.ndarray:
        return self.X + self.R_n

    @staticmethod
    def geom(sigma):
        """X, z, N_r, N_z and |x'| at parameters sigma."""
        sigma = np.asarray(sigma, dtype=float)
        one = np.ones_like(sigma)
        return -sigma, 0.0 * one, 0.0 * one, -one, one


def disc_for(mer: Meridian, k: float = 0.1, n_tip: int = 16, h_max: float = 0.02,
             growth: float = 1.3) -> Disc:
    """Disc nodes with spacing min(max(k d, rho/n_tip), h_max, R_n/10), d the distance from P."""
    rho, R = mer.tip_radius(), mer.R_n
    h_cap = min(h_max, 0.1 * R)
    sig = [0.0]
    h_prev = rho / n_tip
    while True:
        h = min(max(k * sig[-1], rho / n_tip), h_cap, growth * h_prev)
        h_prev = h
        if sig[-1] + h >= R - 0.5 * h:
            break
        sig.append(sig[-1] + h)
    sig.append(R)
    if len(sig) < 5:
        sig = list(np.linspace(0.0, R, 5))
    return Disc(R_n=R, X=-np.array(sig))


# ----------------------------------------------------------------- spacing
def target_spacing(d, rho, k, n_tip, h_max):
    return np.clip(k * d, rho / n_tip, h_max)


# ------------------------------------------------------ initial condition (Eq. 2)
def anthony_initial_meridian(R0: float, Z0: float, k: float = 0.1, n_tip: int = 16,
                             h_max: float = 0.02, growth: float = 1.3) -> Meridian:
    """Eq. 2 bridge [r - (R0 + Z0)]^2 + z^2 = Z0^2 joined to the tangent unit sphere.

    Independent implementation from the paper's definition.  The unit sphere is
    centred on the axis at z_c with (R0 + Z0)^2 + z_c^2 = (1 + Z0)^2 (external
    tangency), and z_c = 1 - delta with delta formed without cancellation.
    """
    num = R0 * R0 + 2.0 * R0 * Z0 - 2.0 * Z0          # (R0+Z0)^2 - 2 Z0 - Z0^2
    zc = math.sqrt((1.0 + Z0) ** 2 - (R0 + Z0) ** 2)
    delta = num / (1.0 + zc)
    # Junction: on the segment from the torus centre (R0+Z0, 0) to (0, z_c), at Z0.
    cx = R0 + Z0
    ux, uz = -cx / (1.0 + Z0), zc / (1.0 + Z0)
    theta_j = math.atan2(uz, ux)                      # torus angle of the junction, in (pi/2, pi)
    alpha_j = math.atan2(cx + Z0 * ux, zc - Z0 * uz)  # sphere angle from the bottom

    def torus(theta):  # theta from pi (neck) down to theta_j
        X = Z0 * 2.0 * np.cos(0.5 * theta) ** 2       # r - R0 = Z0 (1 + cos theta)
        return X, Z0 * np.sin(theta)

    def sphere(alpha):  # alpha from alpha_j to pi (pole)
        r = np.sin(alpha)
        z = 2.0 * np.sin(0.5 * alpha) ** 2 - delta    # z_c - cos(alpha)
        return r - R0, z

    rho = Z0
    X_nodes, z_nodes = [0.0], [0.0]
    h_prev = rho / n_tip
    theta, alpha = math.pi, None
    arc_t = Z0 * (math.pi - theta_j)
    s_t = 0.0
    while True:
        d = math.hypot(X_nodes[-1], z_nodes[-1])
        h = min(float(target_spacing(d, rho, k, n_tip, h_max)), growth * h_prev)
        h_prev = h
        if alpha is None:
            if s_t + h < arc_t - 0.25 * h:
                s_t += h
                theta = math.pi - s_t / Z0
                X, z = torus(theta)
            else:
                alpha = alpha_j + (s_t + h - arc_t)
                X, z = sphere(alpha)
        else:
            alpha = alpha + h
            if alpha >= math.pi - 0.5 * h:
                break
            X, z = sphere(alpha)
        X_nodes.append(float(X))
        z_nodes.append(float(z))
    X_nodes.append(-R0)
    z_nodes.append(float(2.0 - delta))
    return Meridian(R_n=R0, X=np.array(X_nodes), z=np.array(z_nodes))
