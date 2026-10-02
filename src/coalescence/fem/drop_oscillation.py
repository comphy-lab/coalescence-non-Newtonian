"""Small-amplitude shape oscillation of a free drop: verification of the inertial solver.

A drop of unit radius in a passive exterior starts from rest with the shape
r(theta) = a [1 + eps P_2(cos theta)], a chosen so that the volume is 4 pi/3 to O(eps^2).
In visco-capillary units (length R, velocity gamma/mu, time mu R/gamma, density 1/Oh^2)
the l = 2 mode oscillates at the Rayleigh frequency omega = sqrt(8) Oh and its amplitude
decays at Lamb's small-viscosity rate (l - 1)(2l + 1) nu / R^2 = 5 Oh^2 (Lamb 1932,
Hydrodynamics, Arts. 354-356; Prosperetti 1980, J. Fluid Mech. 100, 333). By symmetry
one quadrant is computed: axis r = 0 and equatorial plane z = 0, as in the coalescence
problem, with the same equations (Taylor-Hood Navier-Stokes, Laplace-smoothed ALE mesh,
free surface with unit surface tension).
"""

from __future__ import annotations

import math

import numpy as np
from pyoomph import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.generic import AxisymmetryBC, IntegralObservables
from pyoomph.equations.navier_stokes import NavierStokesEquations, NavierStokesFreeSurface
from pyoomph.expressions import dot, double_dot, grad, transpose


def p2(x):
    return 0.5 * (3.0 * x * x - 1.0)


def mean_radius(eps: float) -> float:
    """Mean radius a for which r = a [1 + eps P_2] encloses 4 pi/3 to O(eps^2)."""
    return (1.0 + 3.0 * eps * eps / 5.0) ** (-1.0 / 3.0)


class OscillatingDropMesh(GmshTemplate):
    def __init__(self, eps: float, resolution: float, n_interface: int = 64):
        super().__init__()
        self.eps, self.resolution, self.n_interface = float(eps), float(resolution), int(n_interface)

    def define_geometry(self):
        self.default_resolution = self.resolution
        self.mesh_mode = "tris"
        a = mean_radius(self.eps)
        theta = np.linspace(0.5 * math.pi, 0.0, self.n_interface)       # equator -> pole
        r = a * (1.0 + self.eps * p2(np.cos(theta)))
        pts = [self.point(float(ri * math.sin(t)), float(ri * math.cos(t))) for ri, t in zip(r, theta)]
        pts[0] = self.point(float(r[0]), 0.0)
        pts[-1] = self.point(0.0, float(r[-1]))
        self.spline(pts, name="interface")
        self.create_lines(pts[-1], "axis", (0.0, 0.0), "plane", pts[0])
        self.plane_surface("axis", "plane", "interface", name="drop")


class OscillatingDropProblem(Problem):
    def __init__(self, *, Oh: float, eps: float = 0.01, resolution: float = 0.05):
        super().__init__()
        if not Oh > 0:
            raise ValueError("Oh must be positive")
        self.Oh, self.eps, self.resolution = float(Oh), float(eps), float(resolution)
        self.rho = 1.0 / self.Oh ** 2

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        self += OscillatingDropMesh(self.eps, self.resolution)
        u = var("velocity")
        G = grad(u)
        D = 0.5 * (G + transpose(G))
        eqs = NavierStokesEquations(dynamic_viscosity=1.0, mass_density=self.rho, mode="TH")
        eqs += LaplaceSmoothedMesh()
        eqs += IntegralObservables(volume=1, kinetic=0.5 * self.rho * dot(u, u),
                                   dissipation=2.0 * double_dot(D, D))
        eqs += AxisymmetryBC() @ "axis"
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "plane"
        eqs += NavierStokesFreeSurface(surface_tension=1.0) @ "interface"
        eqs += IntegralObservables(area=1) @ "interface"
        self.add_equations(eqs @ "drop")

    def state(self) -> dict[str, float]:
        """Pole height, equatorial radius and the energy terms of the computed quadrant."""
        iface = self.get_mesh("drop/interface")
        pole, equator = 0.0, 0.0
        for n in iface.nodes():
            x, y = n.x(0), n.x(1)
            if abs(x) < 1e-12:
                pole = max(pole, y)
            if abs(y) < 1e-12:
                equator = max(equator, x)
        drop = self.get_mesh("drop")
        return {
            "t": float(self.get_current_time(dimensional=False, as_float=True)),
            "z_pole": float(pole),
            "r_equator": float(equator),
            "volume": float(drop.evaluate_observable("volume")),
            "kinetic": float(drop.evaluate_observable("kinetic")),
            "area": float(iface.evaluate_observable("area")),
            "dissipation": float(drop.evaluate_observable("dissipation")),
        }


def rayleigh_lamb(Oh: float, l: int = 2) -> dict[str, float]:
    """Rayleigh frequency and Lamb decay rate of mode l in visco-capillary units (rho = 1/Oh^2)."""
    return {"omega": math.sqrt(l * (l - 1) * (l + 2)) * Oh, "decay_rate": (l - 1) * (2 * l + 1) * Oh ** 2}
