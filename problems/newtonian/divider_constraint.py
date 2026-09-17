"""Lagged Anthony Eq. 6 constraint for the mapped interior divider.

The transient problem owns ``R_lagged`` as a pyoomph ``GlobalParameter`` and
updates its value only after an accepted time step.  This module deliberately
does not implement that time-stepping hook: it supplies the symbolic geometry
and the two equations that consume the parameter.

For the coalescence regime used here, ``0 < R_lagged < 1/2`` and hence
``H=R_lagged``.  The divider is constrained by the scalar ellipse residual

``(mesh_x-c)^2/a^2 + mesh_y^2/b^2 - 1 = 0``.

Only ``mesh_x`` is paired with a Lagrange multiplier, leaving ``mesh_y`` under
the tangential elliptic-grid equation.  The multiplier uses a Cartesian line
measure: an axisymmetric measure would vanish at the axis endpoint.  pyoomph's
``EnforcedBC`` pins the endpoint multiplier automatically because ``mesh_x``
is strongly pinned on ``axis``.  The remaining axis/divider corner coordinate
must be set separately with :func:`axis_divider_corner_condition`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyoomph import DirichletBC, EnforcedBC
from pyoomph.expressions import cartesian, square_root, var


@dataclass(frozen=True)
class Eq6SymbolicParameters:
    """Symbolic translated-ellipse parameters for ``R_lagged < 1/2``."""

    H: Any
    m: Any
    c: Any
    a: Any
    b2: Any


def eq6_symbolic_parameters(r_lagged: Any) -> Eq6SymbolicParameters:
    """Return Anthony's translated-ellipse coefficients as expressions.

    ``r_lagged`` may be a pyoomph ``GlobalParameter`` or a numeric scalar.
    The caller is responsible for maintaining ``0 < r_lagged < 1/2``; this is
    the branch used by the initial bridge and by the intended transient.
    """

    H = r_lagged
    m = square_root(3)
    c = 4 * r_lagged**2 / (4 * r_lagged + m * H)
    a = 2 * r_lagged - c
    b2 = H * a**2 / (m * c)
    return Eq6SymbolicParameters(H=H, m=m, c=c, a=a, b2=b2)


def eq6_divider_constraint(
    r_lagged: Any,
    *,
    radial_coordinate: Any | None = None,
    axial_coordinate: Any | None = None,
) -> Any:
    """Return the scalar Eq. 6 ellipse residual on the parent bulk mesh."""

    radial_coordinate = (
        var("mesh_x", domain="..") if radial_coordinate is None else radial_coordinate
    )
    axial_coordinate = (
        var("mesh_y", domain="..") if axial_coordinate is None else axial_coordinate
    )
    p = eq6_symbolic_parameters(r_lagged)
    return (
        (radial_coordinate - p.c) ** 2 / p.a**2
        + axial_coordinate**2 / p.b2
        - 1
    )


def anthony_divider_constraint(r_lagged: Any) -> EnforcedBC:
    """Create the C2 Cartesian multiplier equation for the interior divider.

    Add the returned equation on ``"drop/divider"`` (or add it to the bulk
    equation set and restrict it with ``@ "divider"``).  The adjusted
    ``mesh_x`` field belongs to the parent bulk domain, hence ``domain=".."``.
    """

    return EnforcedBC(
        mesh_x=eq6_divider_constraint(r_lagged),
        domain="..",
        space="C2",
        coordsys=cartesian,
    )


def axis_divider_corner_condition(r_lagged: Any) -> DirichletBC:
    """Strongly set ``mesh_y=R_lagged`` at the axis/divider corner.

    Apply this equation on the boundary-intersection path
    ``"drop/divider/axis"``.  This point condition is required independently
    of the ellipse multiplier because ``mesh_x`` is pinned on the axis and a
    point contribution with an axisymmetric measure would be identically zero.
    """

    return DirichletBC(mesh_y=r_lagged)


__all__ = [
    "Eq6SymbolicParameters",
    "anthony_divider_constraint",
    "axis_divider_corner_condition",
    "eq6_divider_constraint",
    "eq6_symbolic_parameters",
]
