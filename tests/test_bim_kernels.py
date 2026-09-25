"""Gate K1: axisymmetric ring kernels against direct azimuthal quadrature."""

from __future__ import annotations

import unittest

import numpy as np

from independent.bim_stokes.kernels import ring_kernels, ring_kernels_reference


def _cases(seed: int = 1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(12):
        r0, r = rng.uniform(0.01, 1.0, 2)
        th = rng.uniform(0.0, 2.0 * np.pi)
        out.append((r0, r, r - r0, rng.uniform(-1, 1) - rng.uniform(-1, 1), np.cos(th), np.sin(th)))
    for rel in (1e-2, 1e-7, 1e-12, 1e-18, 1e-24, 1e-30):
        r0 = 1e-6 * (1.0 + rng.uniform(-0.1, 0.1))
        ang, th = rng.uniform(0.0, 2.0 * np.pi, 2)
        dr, dz = rel * 1e-6 * np.cos(ang), rel * 1e-6 * np.sin(ang)
        out.append((r0, r0 + dr, dr, dz, np.cos(th), np.sin(th)))
    for _ in range(4):
        r0, r = rng.uniform(1e-8, 1e-3), rng.uniform(0.1, 1.0)
        th = rng.uniform(0.0, 2.0 * np.pi)
        out.append((r0, r, r - r0, rng.uniform(-1, 1), np.cos(th), np.sin(th)))
    return out


class RingKernelTests(unittest.TestCase):
    def test_closed_forms_match_direct_quadrature_from_far_field_to_coincidence(self) -> None:
        for case in _cases():
            M, Q = ring_kernels(*[np.array([v]) for v in case])
            Mr, Qr = ring_kernels_reference(*case)
            self.assertLess(np.abs(M[..., 0] - Mr).max() / np.abs(Mr).max(), 1e-11, case)
            self.assertLess(np.abs(Q[..., 0] - Qr).max() / np.abs(Qr).max(), 1e-11, case)

    def test_single_layer_is_symmetric_under_exchange_up_to_the_ring_weight(self) -> None:
        # G is symmetric, so r0 M_ab(x0 <- x) = r M_ba(x <- x0).
        r0, r, dz = 0.3, 0.7, 0.2
        M1, _ = ring_kernels(np.array([r0]), np.array([r]), np.array([r - r0]), np.array([dz]), np.array([1.0]), np.array([0.0]))
        M2, _ = ring_kernels(np.array([r]), np.array([r0]), np.array([r0 - r]), np.array([-dz]), np.array([1.0]), np.array([0.0]))
        np.testing.assert_allclose(r0 * M1[..., 0], r * M2[..., 0].T, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
