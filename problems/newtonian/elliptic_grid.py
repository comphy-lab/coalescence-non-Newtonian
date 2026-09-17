"""Scale-covariant controlled elliptic mesh equations for a logical 2-D grid.

This is a project-local reconstruction of a Winslow/inverse-harmonic grid
functional.  It is not Anthony, Harris and Basaran's unpublished control
equation.  Geometry, free-surface tangential control and remeshing belong to
separate layers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from pyoomph.equations.ALE import BaseMovingMeshEquations
from pyoomph.expressions import cartesian, determinant, grad, var


LogicalPoint = tuple[float, float]
LogicalResolver = Callable[[LogicalPoint], Sequence[float] | None]


class LogicalCoordinateTable:
    """Strict physical-position to logical-coordinate lookup.

    With ``tolerance=0`` positions must match a table key exactly.  A positive
    tolerance uses the maximum coordinate distance and rejects both missing and
    multiply matching entries.  Tolerances are deliberately caller-supplied:
    an absolute project-wide tolerance would itself introduce a length scale.
    """

    def __init__(
        self,
        entries: Mapping[Sequence[float], Sequence[float]]
        | Iterable[tuple[Sequence[float], Sequence[float]]],
        *,
        tolerance: float = 0.0,
    ) -> None:
        if tolerance < 0.0 or not math.isfinite(tolerance):
            raise ValueError("logical-coordinate tolerance must be finite and non-negative")
        items = entries.items() if isinstance(entries, Mapping) else entries
        self._entries: list[tuple[LogicalPoint, LogicalPoint]] = []
        for physical, logical in items:
            p = _point2(physical, "physical table coordinate")
            q = _point2(logical, "logical table coordinate")
            if any(p == old_p for old_p, _ in self._entries):
                raise ValueError(f"duplicate physical coordinate in logical table: {p!r}")
            self._entries.append((p, q))
        if not self._entries:
            raise ValueError("logical-coordinate table must not be empty")
        self.tolerance = float(tolerance)
        self._exact = dict(self._entries) if tolerance == 0.0 else None

    def __call__(self, position: LogicalPoint) -> LogicalPoint:
        if self._exact is not None:
            try:
                return self._exact[position]
            except KeyError as exc:
                raise RuntimeError(
                    f"no logical coordinate for physical node {position!r}"
                ) from exc
        matches = [
            logical
            for physical, logical in self._entries
            if max(abs(position[0] - physical[0]), abs(position[1] - physical[1]))
            <= self.tolerance
        ]
        if not matches:
            raise RuntimeError(
                f"no logical coordinate within tolerance {self.tolerance:g} "
                f"of physical node {position!r}"
            )
        if len(matches) != 1:
            raise RuntimeError(
                f"ambiguous logical coordinate for physical node {position!r}: "
                f"{len(matches)} entries are within tolerance {self.tolerance:g}"
            )
        return matches[0]


class ControlledEllipticMesh(BaseMovingMeshEquations):
    r"""Positive-branch 2-D Winslow/inverse-harmonic mesh energy.

    For the physical map ``x(Xi)`` from fixed logical coordinates
    ``Xi=(xi,eta)``, let ``F=dx/dXi``, ``J=det(F)``,
    ``g_xi=|x_,xi|^2`` and ``g_eta=|x_,eta|^2``.  This class minimises

    ``Integral_Xi (m_xi*g_xi + m_eta*g_eta)/(2*J) dXi``.

    ``m_xi`` and ``m_eta`` are dimensionless, strictly positive monitor/control
    expressions.  With both equal to one this is the scale-covariant Winslow
    functional.  Uniform physical scaling multiplies the metric and ``J`` by
    the same factor squared, leaving the integrand unchanged.

    This functional is meaningful on the ``J>0`` branch; it does not by itself
    prove or enforce global orientation preservation.  No Jacobian floor is
    used: it would break scale covariance and hide an inverted grid.  Call
    :func:`mesh_quality` with ``require_positive=True`` around solves, and use
    pyoomph's inverted-element handling for production.

    ``logical_coordinates`` may be a strict table or a callable receiving the
    current physical ``(x,y)`` of each node.  Assignment happens after the base
    macro-element mapping, which otherwise resets Lagrangian coordinates to
    the physical ones.
    """

    def __init__(
        self,
        *,
        monitor_xi: Any = 1.0,
        monitor_eta: Any = 1.0,
        logical_coordinates: LogicalCoordinateTable
        | Mapping[Sequence[float], Sequence[float]]
        | LogicalResolver
        | None = None,
        logical_tolerance: float = 0.0,
    ) -> None:
        super().__init__(coordsys=cartesian)
        _check_positive_numeric_monitor(monitor_xi, "monitor_xi")
        _check_positive_numeric_monitor(monitor_eta, "monitor_eta")
        self.monitor_xi = monitor_xi
        self.monitor_eta = monitor_eta
        if logical_coordinates is None or isinstance(logical_coordinates, LogicalCoordinateTable):
            self.logical_coordinates = logical_coordinates
        elif isinstance(logical_coordinates, Mapping):
            self.logical_coordinates = LogicalCoordinateTable(
                logical_coordinates, tolerance=logical_tolerance
            )
        elif callable(logical_coordinates):
            if logical_tolerance != 0.0:
                raise ValueError(
                    "logical_tolerance applies only to a coordinate table, not a callable resolver"
                )
            self.logical_coordinates = logical_coordinates
        else:
            raise TypeError("logical_coordinates must be a mapping, callable, table, or None")

    def get_squared_spatial_factor(self):
        # The functional is dimensionless and contains no physical stiffness.
        return 1

    def define_residuals(self) -> None:
        if self.get_nodal_dimension() != 2:
            raise RuntimeError("ControlledEllipticMesh is implemented only for 2-D nodal coordinates")
        deformation_gradient = grad(var("mesh"), lagrangian=True, coordsys=cartesian)
        g_xi = deformation_gradient[0, 0] ** 2 + deformation_gradient[1, 0] ** 2
        g_eta = deformation_gradient[0, 1] ** 2 + deformation_gradient[1, 1] ** 2
        jacobian = determinant(deformation_gradient)
        energy = (self.monitor_xi * g_xi + self.monitor_eta * g_eta) / (2 * jacobian)
        self.add_functional_minimization(
            energy,
            dimensional_testfunctions=False,
            coordsys=cartesian,
            lagrangian=True,
        )

    def after_mapping_on_macro_elements(self) -> None:
        super().after_mapping_on_macro_elements()
        resolver = self.logical_coordinates
        if resolver is None:
            return
        mesh = self.get_mesh()
        for node in mesh.nodes():
            position = (float(node.x(0)), float(node.x(1)))
            try:
                logical = resolver(position)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to resolve logical coordinates for node at {position!r}"
                ) from exc
            if logical is None:
                raise RuntimeError(f"logical-coordinate resolver returned None for {position!r}")
            xi = _point2(logical, f"logical coordinate for node {position!r}")
            node.set_x_lagr(0, xi[0])
            node.set_x_lagr(1, xi[1])
        mesh.bump_topology_generation()


@dataclass(frozen=True)
class MeshQuality:
    """Sampled logical-to-physical map diagnostics for quadrilaterals."""

    element_count: int
    min_jacobian: float
    max_condition_number: float
    max_column_aspect: float
    sample_count_per_element: int


_GL3 = math.sqrt(3.0 / 5.0)
# The Q2 interpolation lattice supplies all corners, edge midpoints and the
# centre.  The independent 3x3 Gauss-Legendre lattice also probes the element
# interior.  Their union contains 17 points because both contain the centre.
_QUALITY_SAMPLE_COORDINATES = tuple(dict.fromkeys(
    [(s0, s1) for s1 in (-1.0, 0.0, 1.0) for s0 in (-1.0, 0.0, 1.0)]
    + [(s0, s1) for s1 in (-_GL3, 0.0, _GL3) for s0 in (-_GL3, 0.0, _GL3)]
))


def mesh_quality(mesh: Any, *, require_positive: bool = False) -> MeshQuality:
    """Return scale-aware sampled quality metrics for a C1/C2 quad mesh.

    The Jacobian is ``det(dx/dXi)``.  The 2-norm condition number and ratio of
    the two mapped logical-coordinate column lengths are invariant under
    uniform physical scaling.  The logical equation is Cartesian; no
    axisymmetric measure is included.

    Every element is sampled on the tensor-product Q2 interpolation lattice
    ``{-1,0,1}^2`` (corners, edge midpoints and centre) and the 3x3
    Gauss-Legendre lattice ``{-sqrt(3/5),0,sqrt(3/5)}^2``.  This catches the
    relevant nodal and integration-point folds, but remains a sampled gate; it
    is not an analytic proof that the determinant is positive everywhere.
    """

    count = 0
    min_jacobian = math.inf
    max_condition = 0.0
    max_aspect = 0.0
    for element in mesh.elements():
        count += 1
        for s0, s1 in _QUALITY_SAMPLE_COORDINATES:
            physical, logical = _element_tangents(element, s0, s1)
            f00, f01, f10, f11 = _map_gradient(physical, logical, sample=(s0, s1))
            jacobian = f00 * f11 - f01 * f10
            min_jacobian = min(min_jacobian, jacobian)

            col0 = math.hypot(f00, f10)
            col1 = math.hypot(f01, f11)
            if col0 == 0.0 or col1 == 0.0:
                aspect = math.inf
            else:
                aspect = max(col0, col1) / min(col0, col1)
            max_aspect = max(max_aspect, aspect)

            # Stable closed form for the two singular values.  Forming
            # trace(F^T F)^2-4*det(F)^2 loses the small anisotropy by
            # cancellation close to an isotropic map.
            sum_norm = math.hypot(f00 + f11, f10 - f01)
            difference_norm = math.hypot(f00 - f11, f10 + f01)
            sigma_max = 0.5 * (sum_norm + difference_norm)
            sigma_min = 0.5 * abs(sum_norm - difference_norm)
            condition = sigma_max / sigma_min if sigma_min > 0.0 else math.inf
            max_condition = max(max_condition, condition)

    if count == 0:
        raise RuntimeError("cannot evaluate mesh quality on an empty mesh")
    if require_positive and not (min_jacobian > 0.0):
        raise RuntimeError(
            f"controlled elliptic mesh is inverted or degenerate: min det(dx/dXi)={min_jacobian:g}"
        )
    return MeshQuality(
        count,
        min_jacobian,
        max_condition,
        max_aspect,
        len(_QUALITY_SAMPLE_COORDINATES),
    )


def _point2(value: Sequence[float], what: str) -> LogicalPoint:
    if len(value) != 2:
        raise ValueError(f"{what} must have exactly two components")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{what} must be finite, got {result!r}")
    return result


def _check_positive_numeric_monitor(value: Any, name: str) -> None:
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")


def _lagrange_basis(order: int, coordinate: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if order == 1:
        return (
            ((1.0 - coordinate) / 2.0, (1.0 + coordinate) / 2.0),
            (-0.5, 0.5),
        )
    if order == 2:
        return (
            (
                0.5 * coordinate * (coordinate - 1.0),
                1.0 - coordinate * coordinate,
                0.5 * coordinate * (coordinate + 1.0),
            ),
            (coordinate - 0.5, -2.0 * coordinate, coordinate + 0.5),
        )
    raise ValueError(f"unsupported tensor-product order {order}")


def _element_tangents(
    element: Any, s0: float, s1: float
) -> tuple[tuple[LogicalPoint, LogicalPoint], tuple[LogicalPoint, LogicalPoint]]:
    nnode = element.nnode()
    if nnode == 4:
        order = 1
    elif nnode == 9:
        order = 2
    else:
        raise RuntimeError(f"mesh quality supports only 4- or 9-node quadrilaterals, got {nnode}")

    basis0, derivative0 = _lagrange_basis(order, s0)
    basis1, derivative1 = _lagrange_basis(order, s1)
    nodes_per_direction = order + 1

    def tangent(lagrangian: bool, direction: int) -> LogicalPoint:
        result = [0.0, 0.0]
        for j in range(nodes_per_direction):
            for i in range(nodes_per_direction):
                index = j * nodes_per_direction + i
                weight = (
                    derivative0[i] * basis1[j]
                    if direction == 0
                    else basis0[i] * derivative1[j]
                )
                node = element.node_pt(index)
                for component in range(2):
                    coordinate = node.x_lagr(component) if lagrangian else node.x(component)
                    result[component] += weight * float(coordinate)
        return result[0], result[1]

    physical = (tangent(False, 0), tangent(False, 1))
    logical = (tangent(True, 0), tangent(True, 1))
    return physical, logical


def _map_gradient(
    physical: tuple[LogicalPoint, LogicalPoint],
    logical: tuple[LogicalPoint, LogicalPoint],
    *,
    sample: LogicalPoint | None = None,
) -> tuple[float, float, float, float]:
    l00, l10 = logical[0]
    l01, l11 = logical[1]
    logical_det = l00 * l11 - l01 * l10
    if logical_det == 0.0 or not math.isfinite(logical_det):
        where = "" if sample is None else f" at local coordinate {sample!r}"
        raise RuntimeError(f"logical coordinate map is degenerate{where}")
    inv00, inv01 = l11 / logical_det, -l01 / logical_det
    inv10, inv11 = -l10 / logical_det, l00 / logical_det
    p00, p10 = physical[0]
    p01, p11 = physical[1]
    return (
        p00 * inv00 + p01 * inv10,
        p00 * inv01 + p01 * inv11,
        p10 * inv00 + p11 * inv10,
        p10 * inv01 + p11 * inv11,
    )


__all__ = [
    "ControlledEllipticMesh",
    "LogicalCoordinateTable",
    "MeshQuality",
    "mesh_quality",
]
