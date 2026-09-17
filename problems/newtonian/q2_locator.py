"""Deterministic, scale-aware point location in a :class:`StructuredQ2Graph`.

The locator deliberately uses the complete nine-node isoparametric map.  It
does not have a nearest-node or affine fallback: a point either has a
rounding-resolved inverse in the graph, or an explicit exception is raised.
All physical comparisons are derived from the candidate element geometry and
binary64 error propagation, which is important for the ``O(R0**3)`` spacings
in the Anthony mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .structured_blocks import NodeKey, Point, Q2Element, StructuredQ2Graph


_EPS = math.ulp(1.0)
_TENSOR_INDICES = (
    (0, 0), (2, 0), (2, 2), (0, 2),
    (1, 0), (2, 1), (1, 2), (0, 1), (1, 1),
)
_EDGE_INDICES = {
    "v0": (0, 4, 1),
    "u1": (1, 5, 2),
    "v1": (3, 6, 2),
    "u0": (0, 7, 3),
}


class PointLocationError(RuntimeError):
    """Base class for deterministic point-location failures."""


class MissingSourceLocation(PointLocationError):
    """Raised when no Q2 element contains the requested physical point."""


class AmbiguousSourceLocation(PointLocationError):
    """Raised when several non-shared Q2 inverses satisfy the error bounds."""


@dataclass(frozen=True)
class LocationCandidate:
    """One accepted inverse-map result and its numerical receipt."""

    element_id: tuple[int, int, int]
    u: float
    v: float
    physical_residual: float
    local_coordinate_uncertainty: float


@dataclass(frozen=True)
class SourceLocation:
    """Accepted source point, including all conforming shared-edge matches."""

    element_id: tuple[int, int, int]
    u: float
    v: float
    physical_residual: float
    local_coordinate_uncertainty: float
    shared_edge: bool
    candidates: tuple[LocationCandidate, ...]


@dataclass(frozen=True)
class _ElementGeometry:
    element: Q2Element
    points: tuple[Point, ...]
    anchor: Point
    local_scale: float
    bbox: tuple[float, float, float, float]
    bbox_padding: tuple[float, float]


def _basis(value: float) -> tuple[float, float, float]:
    return (
        2.0 * (value - 0.5) * (value - 1.0),
        4.0 * value * (1.0 - value),
        2.0 * value * (value - 0.5),
    )


def _basis_derivative(value: float) -> tuple[float, float, float]:
    return (4.0 * value - 3.0, 4.0 - 8.0 * value, 4.0 * value - 1.0)


def q2_weights(u: float, v: float) -> tuple[float, ...]:
    """Return graph-ordered tensor Q2 interpolation weights."""
    lu, lv = _basis(float(u)), _basis(float(v))
    return tuple(lu[i] * lv[j] for i, j in _TENSOR_INDICES)


def q1_weights(u: float, v: float) -> tuple[float, float, float, float]:
    """Return graph-corner-ordered bilinear weights."""
    u, v = float(u), float(v)
    return ((1.0 - u) * (1.0 - v), u * (1.0 - v), u * v, (1.0 - u) * v)


def _centred_map(geometry: _ElementGeometry, u: float, v: float) -> Point:
    """Evaluate map minus anchor, avoiding cancellation under translation."""
    ax, ay = geometry.anchor
    x = y = 0.0
    for weight, point in zip(q2_weights(u, v), geometry.points):
        x += weight * (point[0] - ax)
        y += weight * (point[1] - ay)
    return x, y


def _centred_jacobian(
    geometry: _ElementGeometry, u: float, v: float
) -> tuple[float, float, float, float]:
    lu, lv = _basis(u), _basis(v)
    du, dv = _basis_derivative(u), _basis_derivative(v)
    ax, ay = geometry.anchor
    dxdu = dxdv = dydu = dydv = 0.0
    for point, (i, j) in zip(geometry.points, _TENSOR_INDICES):
        x, y = point[0] - ax, point[1] - ay
        dxdu += du[i] * lv[j] * x
        dxdv += lu[i] * dv[j] * x
        dydu += du[i] * lv[j] * y
        dydv += lu[i] * dv[j] * y
    return dxdu, dxdv, dydu, dydv


def _smallest_singular_value(jacobian: tuple[float, float, float, float]) -> float:
    a, b, c, d = jacobian
    frob2 = a * a + b * b + c * c + d * d
    det = a * d - b * c
    disc = max(0.0, frob2 * frob2 - 4.0 * det * det)
    sigma_max = math.sqrt(max(0.0, 0.5 * (frob2 + math.sqrt(disc))))
    # det(J) = sigma_max*sigma_min.  This quotient avoids the catastrophic
    # cancellation in ``frob2 - sqrt(disc)`` for Anthony's thin bridge cells.
    return abs(det) / sigma_max if sigma_max > 0.0 else 0.0


def _scaled_qr_solve(
    jacobian: tuple[float, float, float, float], residual: Point
) -> Point | None:
    """Solve ``J step = residual`` by column-scaled two-column QR."""
    a, b, c, d = jacobian
    s0, s1 = math.hypot(a, c), math.hypot(b, d)
    if s0 == 0.0 or s1 == 0.0:
        return None
    q0x, q0y = a / s0, c / s0
    c1x, c1y = b / s1, d / s1
    r01 = q0x * c1x + q0y * c1y
    ox, oy = c1x - r01 * q0x, c1y - r01 * q0y
    r11 = math.hypot(ox, oy)
    if r11 <= 64.0 * _EPS:
        return None
    q1x, q1y = ox / r11, oy / r11
    y0 = q0x * residual[0] + q0y * residual[1]
    y1 = q1x * residual[0] + q1y * residual[1]
    z1 = y1 / r11
    z0 = y0 - r01 * z1
    return z0 / s0, z1 / s1


def _rounding_bound(geometry: _ElementGeometry, point: Point, u: float, v: float) -> float:
    """Conservative propagated binary64 bound for the centred map residual."""
    weights = q2_weights(u, v)
    ax, ay = geometry.anchor
    sums = []
    for component, anchor in enumerate((ax, ay)):
        interpolation_sum = sum(
            abs(weight) * abs(node[component] - anchor)
            for weight, node in zip(weights, geometry.points)
        )
        target_offset = abs(point[component] - anchor)
        # 9 products and 8 sums plus basis evaluation and the two anchor
        # subtractions.  Explicit ulps retain the translated-coordinate cost.
        gamma = 48.0 * _EPS
        sums.append(
            gamma * (interpolation_sum + target_offset + geometry.local_scale)
            + 8.0 * (math.ulp(point[component]) + math.ulp(anchor))
        )
    return max(math.hypot(*sums), 32.0 * _EPS * geometry.local_scale)


def _bernstein_controls(values: Sequence[float]) -> tuple[float, ...]:
    """Convert graph-ordered Q2 samples to a tensor Bernstein control net."""
    grid = [[0.0] * 3 for _ in range(3)]
    for value, (i, j) in zip(values, _TENSOR_INDICES):
        grid[j][i] = value
    # Convert each row, then each column.  A quadratic Bernstein control
    # polygon bounds the polynomial exactly, unlike a merely sampled AABB.
    rows = []
    for row in grid:
        rows.append((row[0], 2.0 * row[1] - 0.5 * (row[0] + row[2]), row[2]))
    controls = [[0.0] * 3 for _ in range(3)]
    for i in range(3):
        controls[0][i] = rows[0][i]
        controls[1][i] = 2.0 * rows[1][i] - 0.5 * (rows[0][i] + rows[2][i])
        controls[2][i] = rows[2][i]
    return tuple(controls[j][i] for j in range(3) for i in range(3))


def _make_geometry(graph: StructuredQ2Graph, element: Q2Element) -> _ElementGeometry:
    points = tuple(graph.nodes[key].coordinates for key in element.node_keys)
    anchor = points[8]
    x_controls = _bernstein_controls([point[0] for point in points])
    y_controls = _bernstein_controls([point[1] for point in points])
    xmin, xmax = min(x_controls), max(x_controls)
    ymin, ymax = min(y_controls), max(y_controls)
    local_scale = max(xmax - xmin, ymax - ymin, max(math.ulp(value) for point in points for value in point))
    padx = 32.0 * _EPS * local_scale + 8.0 * max(math.ulp(xmin), math.ulp(xmax))
    pady = 32.0 * _EPS * local_scale + 8.0 * max(math.ulp(ymin), math.ulp(ymax))
    return _ElementGeometry(element, points, anchor, local_scale, (xmin, xmax, ymin, ymax), (padx, pady))


def _reference_allowance(uncertainty: float) -> float:
    return max(64.0 * _EPS, 4.0 * uncertainty)


def _topological_entity(
    geometry: _ElementGeometry, candidate: LocationCandidate
) -> frozenset[NodeKey] | None:
    allowance = _reference_allowance(candidate.local_coordinate_uncertainty)
    u, v = candidate.u, candidate.v
    at_u0, at_u1 = abs(u) <= allowance, abs(u - 1.0) <= allowance
    at_v0, at_v1 = abs(v) <= allowance, abs(v - 1.0) <= allowance
    keys = geometry.element.node_keys
    if (at_u0 or at_u1) and (at_v0 or at_v1):
        index = 0 if at_u0 and at_v0 else 1 if at_u1 and at_v0 else 2 if at_u1 else 3
        return frozenset((keys[index],))
    if at_v0:
        return frozenset(keys[index] for index in _EDGE_INDICES["v0"])
    if at_u1:
        return frozenset(keys[index] for index in _EDGE_INDICES["u1"])
    if at_v1:
        return frozenset(keys[index] for index in _EDGE_INDICES["v1"])
    if at_u0:
        return frozenset(keys[index] for index in _EDGE_INDICES["u0"])
    return None


class Q2PointLocator:
    """Precomputed deterministic full-Q2 locator for one source graph."""

    def __init__(self, graph: StructuredQ2Graph):
        self.graph = graph
        self._geometries = tuple(_make_geometry(graph, element) for element in graph.elements)
        self._by_id = {geometry.element.element_id: geometry for geometry in self._geometries}

    def _invert(self, geometry: _ElementGeometry, point: Point) -> LocationCandidate | None:
        target = (point[0] - geometry.anchor[0], point[1] - geometry.anchor[1])
        accepted: list[LocationCandidate] = []
        # Deterministic bounded multistart: centres of a 4x4 reference-cell
        # subdivision plus its 5x5 vertices.  This reaches curved Q2 basins
        # without allowing an iterate to escape to an unrelated map branch.
        seeds = [(i / 4.0, j / 4.0) for j in range(5) for i in range(5)]
        seeds += [((i + 0.5) / 4.0, (j + 0.5) / 4.0) for j in range(4) for i in range(4)]
        for seed_u, seed_v in seeds:
            u, v = seed_u, seed_v
            for _ in range(32):
                mapped = _centred_map(geometry, u, v)
                residual_vector = (mapped[0] - target[0], mapped[1] - target[1])
                residual = math.hypot(*residual_vector)
                jacobian = _centred_jacobian(geometry, u, v)
                sigma_min = _smallest_singular_value(jacobian)
                rounding = _rounding_bound(geometry, point, u, v)
                uncertainty = (residual + rounding) / sigma_min if sigma_min > 0.0 else math.inf
                allowance = _reference_allowance(uncertainty)
                if residual <= 4.0 * rounding and -allowance <= u <= 1.0 + allowance and -allowance <= v <= 1.0 + allowance:
                    # Clamp only excursions accounted for by the inverse-map
                    # uncertainty.  Materially exterior coordinates never get
                    # projected onto the cell.
                    cu = 0.0 if u < 0.0 else 1.0 if u > 1.0 else u
                    cv = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
                    remapped = _centred_map(geometry, cu, cv)
                    final_residual = math.hypot(remapped[0] - target[0], remapped[1] - target[1])
                    final_rounding = _rounding_bound(geometry, point, cu, cv)
                    final_sigma = _smallest_singular_value(_centred_jacobian(geometry, cu, cv))
                    final_uncertainty = (
                        (final_residual + final_rounding) / final_sigma
                        if final_sigma > 0.0 else math.inf
                    )
                    if final_residual <= 4.0 * final_rounding:
                        accepted.append(LocationCandidate(
                            geometry.element.element_id, cu, cv,
                            final_residual, final_uncertainty,
                        ))
                    break
                step = _scaled_qr_solve(jacobian, residual_vector)
                if step is None:
                    break
                # Backtracking is deterministic and accepts the best bounded
                # decrease.  A small halo lets Newton resolve edge points.
                old_residual = residual
                moved = False
                for power in range(9):
                    factor = 0.5**power
                    trial_u = min(1.125, max(-0.125, u - factor * step[0]))
                    trial_v = min(1.125, max(-0.125, v - factor * step[1]))
                    trial = _centred_map(geometry, trial_u, trial_v)
                    trial_residual = math.hypot(trial[0] - target[0], trial[1] - target[1])
                    if trial_residual < old_residual or trial_residual <= 4.0 * _rounding_bound(geometry, point, trial_u, trial_v):
                        u, v, moved = trial_u, trial_v, True
                        break
                if not moved:
                    break
        if not accepted:
            return None
        accepted.sort(key=lambda item: (item.physical_residual, item.u, item.v))
        best = accepted[0]
        # Multiple multistarts that reach a genuinely distinct inverse branch
        # inside one element are ambiguous, rather than silently selected.
        for other in accepted[1:]:
            allowance = 8.0 * max(best.local_coordinate_uncertainty, other.local_coordinate_uncertainty, 16.0 * _EPS)
            if math.hypot(best.u - other.u, best.v - other.v) > allowance:
                raise AmbiguousSourceLocation(
                    f"point {point!r} has multiple inverse branches in element {best.element_id!r}"
                )
        return best

    def locate(self, point: Sequence[float]) -> SourceLocation:
        """Locate a physical point or raise an explicit missing/ambiguous error."""
        if len(point) != 2:
            raise ValueError("point must have exactly two coordinates")
        physical = (float(point[0]), float(point[1]))
        if not all(math.isfinite(value) for value in physical):
            raise ValueError("point coordinates must be finite")
        plausible = []
        for geometry in self._geometries:
            xmin, xmax, ymin, ymax = geometry.bbox
            padx, pady = geometry.bbox_padding
            if xmin - padx <= physical[0] <= xmax + padx and ymin - pady <= physical[1] <= ymax + pady:
                plausible.append(geometry)
        hits = [hit for geometry in plausible if (hit := self._invert(geometry, physical)) is not None]
        if not hits:
            raise MissingSourceLocation(
                f"point {physical!r} is outside every rounding-padded Q2 element"
            )
        hits.sort(key=lambda item: (item.physical_residual, item.element_id))
        if len(hits) > 1:
            entities = [
                _topological_entity(self._by_id[hit.element_id], hit)
                for hit in hits
            ]
            if any(entity is None for entity in entities):
                raise AmbiguousSourceLocation(
                    f"point {physical!r} has {len(hits)} non-edge Q2 locations"
                )
            # All hits must describe one conforming topological edge/vertex.
            if not any(set(entity).intersection(*map(set, entities[1:])) for entity in entities[:1]):
                raise AmbiguousSourceLocation(
                    f"point {physical!r} lies on unrelated Q2 element boundaries"
                )
        best = hits[0]
        return SourceLocation(
            best.element_id, best.u, best.v, best.physical_residual,
            best.local_coordinate_uncertainty, len(hits) > 1, tuple(hits),
        )


def _evaluate_candidate(
    graph: StructuredQ2Graph,
    candidate: LocationCandidate,
    nodal_values: Mapping[NodeKey, float],
    *,
    q1: bool,
) -> float:
    element = next(element for element in graph.elements if element.element_id == candidate.element_id)
    indices = range(4) if q1 else range(9)
    weights = q1_weights(candidate.u, candidate.v) if q1 else q2_weights(candidate.u, candidate.v)
    return math.fsum(weights[index] * float(nodal_values[element.node_keys[index]]) for index in indices)


def _continuous_value(values: Sequence[float], location: SourceLocation) -> float:
    first = values[0]
    scale = max(1.0, *(abs(value) for value in values))
    uncertainty = max(candidate.local_coordinate_uncertainty for candidate in location.candidates)
    tolerance = 64.0 * _EPS * scale + 8.0 * uncertainty * scale
    if any(abs(value - first) > tolerance for value in values[1:]):
        raise AmbiguousSourceLocation(
            f"field is discontinuous across shared candidates {tuple(c.element_id for c in location.candidates)!r}"
        )
    return math.fsum(values) / len(values)


def evaluate_q2(
    graph: StructuredQ2Graph,
    location: SourceLocation,
    nodal_values: Mapping[NodeKey, float],
) -> float:
    """Interpolate a scalar Q2 nodal field, verifying shared-edge continuity."""
    values = [_evaluate_candidate(graph, candidate, nodal_values, q1=False) for candidate in location.candidates]
    return _continuous_value(values, location)


def evaluate_q1(
    graph: StructuredQ2Graph,
    location: SourceLocation,
    nodal_values: Mapping[NodeKey, float],
) -> float:
    """Interpolate a scalar corner-Q1 field, verifying shared-edge continuity."""
    values = [_evaluate_candidate(graph, candidate, nodal_values, q1=True) for candidate in location.candidates]
    return _continuous_value(values, location)


__all__ = [
    "AmbiguousSourceLocation",
    "LocationCandidate",
    "MissingSourceLocation",
    "PointLocationError",
    "Q2PointLocator",
    "SourceLocation",
    "evaluate_q1",
    "evaluate_q2",
    "q1_weights",
    "q2_weights",
]
