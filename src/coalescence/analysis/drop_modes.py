"""Linear shape modes of a viscous drop in a passive exterior, at any Ohnesorge number.

A drop of radius a, viscosity mu, density rho and surface tension gamma, with interface
r = a + zeta(t) P_l(cos theta), released from rest. Visco-capillary units (length a,
velocity gamma/mu, time mu a/gamma): rho = 1/Oh^2 and nu = Oh^2. The linearised
Navier-Stokes equations are Laplace transformed in time (variable s; the initial velocity
is zero). The interior solution regular at the centre is the sum of a potential part and
a poloidal viscous part,

    u = grad phi + curl curl (r psi P_l),   phi = A r^l P_l,   psi = B i_l(q r),
    q^2 = s/nu,   p = -rho s phi,

with i_l the modified spherical Bessel function. On r = 1 the kinematic condition, zero
tangential stress, and the normal-stress balance with the capillary pressure
(l - 1)(l + 2) zeta give, with B' = B i_l(q) and g = q i_l'(q)/i_l(q),

    s Z - zeta_0                         = l A + l(l + 1) B'
    2(l - 1) A + [q^2 + 2(l^2 + l - 1) - 2 g] B'          = 0
    (rho s + 2 l(l - 1)) A + 2 l(l + 1)(g - 1) B' + (l - 1)(l + 2) Z = 0

for the transformed amplitude Z(s). Its poles are the normal modes. The relation reduces
to Rayleigh's inviscid response Z = zeta_0 s/(s^2 + omega_R^2), omega_R^2 = l(l - 1)(l + 2)
Oh^2, and its least-damped oscillatory root approaches Lamb's weak-viscosity decay rate
(l - 1)(2l + 1) Oh^2. In the Stokes limit its slowest root tends to the quasi-static rate
-l(l + 2)(2l + 1)/(2(2l^2 + 4l + 3)) obtained independently from Lamb's general solution of
the Stokes equations (both checked in testCases/test_drop_modes.py). Only the ratio
i_{l-1}/i_l enters, evaluated with exponentially scaled Bessel functions, so large |q| does
not overflow.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import ive


def bessel_ratio(q: complex, l: int) -> complex:
    """q i_l'(q) / i_l(q), from i_l' = i_{l-1} - (l + 1) i_l / q."""
    return q * ive(l - 0.5, q) / ive(l + 0.5, q) - (l + 1)


def interface_matrix(s: complex, Oh: float, l: int = 2) -> np.ndarray:
    """Coefficients of (A, B', Z) in the three interface conditions; the source is the kinematic row."""
    rho = 1.0 / Oh ** 2
    q = np.sqrt(complex(s)) / Oh
    g = bessel_ratio(q, l)
    return np.array([[-l, -l * (l + 1), s],
                     [2 * (l - 1), q * q + 2 * (l * l + l - 1) - 2 * g, 0],
                     [rho * s + 2 * l * (l - 1), 2 * l * (l + 1) * (g - 1), (l - 1) * (l + 2)]], dtype=complex)


def amplitude_transform(s: complex, Oh: float, l: int = 2) -> complex:
    """Z(s) for a unit initial amplitude released from rest."""
    return complex(np.linalg.solve(interface_matrix(s, Oh, l), np.array([1, 0, 0], dtype=complex))[2])


def mode_root(Oh: float, guess: complex, l: int = 2, tol: float = 1e-13) -> complex:
    """Pole of Z(s) nearest ``guess``, by Newton iteration on 1/Z with a central-difference derivative."""
    f = lambda s: 1.0 / amplitude_transform(s, Oh, l)
    s = complex(guess)
    for _ in range(100):
        h = 1e-7 * abs(s)
        step = f(s) / ((f(s + h) - f(s - h)) / (2 * h))
        s -= step
        if abs(step) < tol * abs(s):
            return s
    raise RuntimeError(f"no convergence to a mode root from {guess} at Oh = {Oh}")


def oscillatory_mode(Oh: float, l: int = 2) -> dict[str, float]:
    """Frequency and decay rate of the least-damped oscillatory mode, with the leading-order values.

    Started from Rayleigh's frequency and Lamb's rate, so valid where the mode oscillates
    (small and moderate Oh).
    """
    omega_R = math.sqrt(l * (l - 1) * (l + 2)) * Oh
    lamb = (l - 1) * (2 * l + 1) * Oh ** 2
    s = mode_root(Oh, complex(-lamb, omega_R), l)
    if not s.imag > 0:
        raise RuntimeError(f"the root found at Oh = {Oh} is not oscillatory: {s}")
    return {"omega": s.imag, "decay_rate": -s.real, "omega_rayleigh": omega_R, "decay_rate_lamb": lamb}


def stokes_relaxation_rate(l: int = 2) -> float:
    """Quasi-static relaxation rate of mode l in the Stokes limit (units gamma/(mu a))."""
    return l * (l + 2) * (2 * l + 1) / (2.0 * (2 * l * l + 4 * l + 3))
