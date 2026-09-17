"""Independent geometry and conservation gates for a runtime remesh.

The mapped remesher has its own graph, locator and mesh-quality helpers.  This
module is deliberately a separate evaluator for the *live* pyoomph mesh.  It
reads element nodes and history slots, evaluates the tensor-product basis here,
and performs the meridian integrals with a fixed Gauss rule.  In particular it
does not call ``IntegralObservables``, ``mesh_quality`` or the remap locator.

The evaluator is useful in two places:

* :func:`diagnose_mesh` makes an immutable snapshot before a recreation remesh
  invalidates the old live nodes.
* :func:`compare_remesh` turns two snapshots into a small, executable receipt
  with volume, interface, Jacobian, divergence and flux gates.

Coordinates are ``(r,z)`` in an axisymmetric meridian.  Bulk C2 elements use
the pyoomph row-major node order ``[SW,S,SE,W,C,E,NW,N,NE]`` and local
coordinates in ``[-1,1]^2``.  Velocity fields are Q2.  A few helpers accept
Q1 geometry as well, which makes the evaluator convenient for small synthetic
tests, but the runtime remesh path requires nine-node bulk elements.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


# Eight-point Gauss--Legendre on [-1,1].  The order is intentionally fixed so
# receipts do not depend on pyoomph's current integration scheme or on any
# production equation's quadrature setting.
GAUSS8_POINTS: tuple[float, ...] = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
)
GAUSS8_WEIGHTS: tuple[float, ...] = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)

POSITION_SLOTS: tuple[int, ...] = (0, 1, 2)
Q2_ROW_MAJOR_LOCAL_COORDINATES: tuple[tuple[float, float], ...] = (
    (-1.0, -1.0),
    (0.0, -1.0),
    (1.0, -1.0),
    (-1.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (1.0, 1.0),
)
# Row-major runtime order: SW, S, SE, W, C, E, NW, N, NE.
Q2_CORNER_INDICES: tuple[int, ...] = (0, 2, 6, 8)
Q2_NODE_ORDER = Q2_ROW_MAJOR_LOCAL_COORDINATES
Q1_CORNER_INDICES = Q2_CORNER_INDICES
TWO_PI = 2.0 * math.pi


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _as_slots(value: Iterable[int] | int | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        result = default
    elif isinstance(value, int) and not isinstance(value, bool):
        result = (value,)
    else:
        result = tuple(value)  # type: ignore[arg-type]
    if not result or any(isinstance(slot, bool) or int(slot) != slot or int(slot) < 0 for slot in result):
        raise ValueError(f"history slots must be non-empty non-negative integers, got {result!r}")
    return tuple(int(slot) for slot in result)


def _lagrange_1d(coordinate: float, order: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return one-dimensional Lagrange values and derivatives.

    The Q2 nodes are at ``(-1, 0, 1)``.  The Q1 branch is only a convenience
    for synthetic quadrilateral probes; all runtime remesh elements use Q2.
    """
    s = _finite(coordinate, "local coordinate")
    if order == 1:
        return ((0.5 * (1.0 - s), 0.5 * (1.0 + s)), (-0.5, 0.5))
    if order == 2:
        return (
            (0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)),
            (s - 0.5, -2.0 * s, s + 0.5),
        )
    raise ValueError(f"unsupported tensor-product order {order}")


def q2_basis(coordinate: float) -> tuple[float, float, float]:
    """Return the independent Q2 basis at one local coordinate."""
    return _lagrange_1d(coordinate, 2)[0]  # type: ignore[return-value]


def q2_basis_derivative(coordinate: float) -> tuple[float, float, float]:
    """Return derivatives of the independent Q2 basis."""
    return _lagrange_1d(coordinate, 2)[1]  # type: ignore[return-value]


def q1_basis(coordinate: float) -> tuple[float, float]:
    """Return the bilinear (pressure/C1) basis in one direction."""
    return _lagrange_1d(coordinate, 1)[0]  # type: ignore[return-value]


def q1_basis_derivative(coordinate: float) -> tuple[float, float]:
    """Return derivatives of the bilinear (pressure/C1) basis."""
    return _lagrange_1d(coordinate, 1)[1]  # type: ignore[return-value]


q2_basis_derivatives = q2_basis_derivative
q1_basis_derivatives = q1_basis_derivative


def q1_weights(coordinate: tuple[float, float]) -> tuple[float, float, float, float]:
    """Return row-major corner weights for a Q1 field on a Q2 element."""
    basis_x = q1_basis(coordinate[0])
    basis_y = q1_basis(coordinate[1])
    # C1 pressure lives only on SW, SE, NW, NE corners.  Keep that order
    # explicit rather than treating the pressure's constrained midpoint slots
    # as independent Q2 data.
    return (
        basis_x[0] * basis_y[0],
        basis_x[1] * basis_y[0],
        basis_x[0] * basis_y[1],
        basis_x[1] * basis_y[1],
    )


def evaluate_q1(values: Sequence[float], coordinate: tuple[float, float]) -> float:
    """Evaluate four corner values with the independent Q1 basis."""
    if len(values) != 4:
        raise ValueError(f"Q1 evaluation requires four corner values, got {len(values)}")
    return math.fsum(weight * float(value) for weight, value in zip(q1_weights(coordinate), values))


def evaluate_q1_field(
    element: Any,
    coordinate: tuple[float, float],
    *,
    history_slot: int = 0,
    field_name: str = "pressure",
    field_index: int | None = None,
) -> float:
    """Evaluate a Q1/C1 nodal field (pressure) on a live Q2 element.

    pyoomph keeps constrained C1 values on all nine geometric nodes for a
    convenient storage layout, but only the four corner values are the
    interpolating pressure degrees of freedom.  This helper makes that
    distinction explicit for callers that want an independent pressure probe.
    """
    values: list[float] = []
    for local in Q2_CORNER_INDICES:
        node = _node(element, local)
        index = field_index
        if index is None and hasattr(element, "get_nodal_index_by_name"):
            index = int(element.get_nodal_index_by_name(node, field_name))
        if index is None or index < 0:
            raise ValueError(f"element does not provide Q1 field {field_name!r} at corner {local}")
        values.append(_node_value(node, history_slot, index))
    return evaluate_q1(values, coordinate)


def _element_order(element: Any) -> int:
    nnode = int(element.nnode())
    if nnode == 9:
        return 2
    if nnode == 4:
        return 1
    raise ValueError(
        "independent diagnostics support only 4- or 9-node quadrilaterals, "
        f"got {nnode}"
    )


def _node(element: Any, index: int) -> Any:
    try:
        return element.node_pt(index)
    except AttributeError:
        return tuple(element.nodes())[index]


def _node_position(node: Any, slot: int, component: int) -> float:
    if hasattr(node, "x_at_t"):
        return _finite(node.x_at_t(int(slot), int(component)), "nodal position")
    if int(slot) == 0 and hasattr(node, "x"):
        return _finite(node.x(int(component)), "nodal position")
    raise AttributeError("diagnostic node has neither x_at_t nor x")


def _node_value(node: Any, slot: int, index: int) -> float:
    if hasattr(node, "value_at_t"):
        try:
            return _finite(node.value_at_t(int(slot), int(index)), "nodal field value")
        except (IndexError, RuntimeError) as exc:
            raise ValueError(
                f"nodal field index {index} or history slot {slot} is unavailable"
            ) from exc
    if int(slot) == 0 and hasattr(node, "value"):
        return _finite(node.value(int(index)), "nodal field value")
    raise AttributeError("diagnostic node has neither value_at_t nor value")


def _position_values(element: Any, slot: int) -> tuple[tuple[float, float], ...]:
    return tuple(
        (
            _node_position(_node(element, index), slot, 0),
            _node_position(_node(element, index), slot, 1),
        )
        for index in range(int(element.nnode()))
    )


def _tensor_value_and_derivatives(
    values: Sequence[float], coordinate: tuple[float, float], order: int
) -> tuple[float, float, float]:
    """Evaluate a tensor field and its two local derivatives.

    ``values`` is in row-major order with the first local direction varying
    fastest.  ``math.fsum`` keeps the cancellation in the tiny bridge cells
    reproducible.
    """
    basis_x, derivative_x = _lagrange_1d(coordinate[0], order)
    basis_y, derivative_y = _lagrange_1d(coordinate[1], order)
    side = order + 1
    expected = side * side
    if len(values) != expected:
        raise ValueError(f"tensor field has {len(values)} values, expected {expected}")
    value_terms: list[float] = []
    dx_terms: list[float] = []
    dy_terms: list[float] = []
    for j in range(side):
        for i in range(side):
            nodal = float(values[j * side + i])
            value_terms.append(basis_x[i] * basis_y[j] * nodal)
            dx_terms.append(derivative_x[i] * basis_y[j] * nodal)
            dy_terms.append(basis_x[i] * derivative_y[j] * nodal)
    return math.fsum(value_terms), math.fsum(dx_terms), math.fsum(dy_terms)


def _geometry_at(
    element: Any, slot: int, coordinate: tuple[float, float]
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
    """Return ``(r,z)`` and ``(r_xi,r_eta,z_xi,z_eta)`` from live nodes."""
    order = _element_order(element)
    points = _position_values(element, slot)
    # Derivatives are evaluated after subtracting a nodal anchor.  The basis
    # derivative sums are zero, so this removes large absolute-coordinate
    # cancellation without changing the polynomial derivative.
    anchor_r, anchor_z = points[0]
    # Evaluate the map in centred form.  This is important when a local
    # bridge cell is translated by O(1) but has an O(R0^3) width.
    centred_r, r_xi, r_eta = _tensor_value_and_derivatives(
        [point[0] - anchor_r for point in points], coordinate, order
    )
    centred_z, z_xi, z_eta = _tensor_value_and_derivatives(
        [point[1] - anchor_z for point in points], coordinate, order
    )
    return (
        (anchor_r + centred_r, anchor_z + centred_z),
        (r_xi, r_eta, z_xi, z_eta),
    )


def _determinant(jacobian: tuple[float, float, float, float]) -> float:
    r_xi, r_eta, z_xi, z_eta = jacobian
    return r_xi * z_eta - r_eta * z_xi


def _quality_from_jacobian(jacobian: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return signed scaled determinant and 2-norm condition number."""
    r_xi, r_eta, z_xi, z_eta = jacobian
    determinant = _determinant(jacobian)
    column_x = math.hypot(r_xi, z_xi)
    column_y = math.hypot(r_eta, z_eta)
    denominator = column_x * column_y
    scaled = determinant / denominator if denominator > 0.0 else float("nan")

    # Stable singular values for a 2x2 matrix.  The sum/difference form gives
    # an exact zero for the antisymmetric part of a scaled rotation (where a
    # Frobenius/discriminant formula loses several digits); the determinant
    # quotient then avoids cancellation for the small singular value in thin
    # bridge cells.
    sum_norm = math.hypot(r_xi + z_eta, z_xi - r_eta)
    difference_norm = math.hypot(r_xi - z_eta, z_xi + r_eta)
    sigma_max = 0.5 * (sum_norm + difference_norm)
    if sigma_max == 0.0:
        return scaled, float("inf")
    sigma_min = abs(determinant) / sigma_max if sigma_max > 0.0 else 0.0
    condition = sigma_max / sigma_min if sigma_min > 0.0 else float("inf")
    return scaled, condition


def _sample_coordinates(include_endpoints: bool = True) -> tuple[tuple[float, float], ...]:
    coordinates = [(xi, eta) for eta in GAUSS8_POINTS for xi in GAUSS8_POINTS]
    if include_endpoints:
        coordinates.extend((xi, eta) for eta in (-1.0, 0.0, 1.0) for xi in (-1.0, 0.0, 1.0))
    return tuple(dict.fromkeys(coordinates))


@dataclass(frozen=True)
class InterfaceSegment:
    """One oriented live Q2 line element, stored as three ``(r,z)`` points."""

    start: tuple[float, float]
    midpoint: tuple[float, float]
    end: tuple[float, float]

    def evaluate(self, parameter: float) -> tuple[float, float]:
        t = _finite(parameter, "interface parameter")
        if not -1.0 <= t <= 1.0:
            raise ValueError("interface parameter must lie in [-1,1]")
        basis, _ = _lagrange_1d(t, 2)
        return (
            math.fsum((basis[0] * self.start[0], basis[1] * self.midpoint[0], basis[2] * self.end[0])),
            math.fsum((basis[0] * self.start[1], basis[1] * self.midpoint[1], basis[2] * self.end[1])),
        )

    def derivative(self, parameter: float) -> tuple[float, float]:
        t = _finite(parameter, "interface parameter")
        if not -1.0 <= t <= 1.0:
            raise ValueError("interface parameter must lie in [-1,1]")
        _, derivative = _lagrange_1d(t, 2)
        return (
            math.fsum((derivative[0] * self.start[0], derivative[1] * self.midpoint[0], derivative[2] * self.end[0])),
            math.fsum((derivative[0] * self.start[1], derivative[1] * self.midpoint[1], derivative[2] * self.end[1])),
        )

    def arclength(self) -> float:
        # The local line coordinate already spans [-1,1], so the Jacobian of
        # this one-dimensional map is one (not one half as it would be for a
        # parameter normalised to [0,1]).
        return math.fsum(
            weight * math.hypot(*self.derivative(point))
            for point, weight in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS)
        )

    @property
    def scale(self) -> float:
        return max(
            self.arclength(),
            math.dist(self.start, self.end),
            math.dist(self.start, self.midpoint),
            math.dist(self.midpoint, self.end),
        )


def _endpoint_scale(points: Iterable[tuple[float, float]]) -> float:
    values = [abs(component) for point in points for component in point]
    return max(1.0, *values) if values else 1.0


def _point_close(first: tuple[float, float], second: tuple[float, float]) -> bool:
    """Compare endpoint coordinates at the binary64 scale."""
    return all(
        a == b or abs(a - b) <= 16.0 * max(math.ulp(a), math.ulp(b))
        for a, b in zip(first, second)
    )


def _interface_chain(interface_mesh: Any, slot: int) -> tuple[InterfaceSegment, ...]:
    elements = tuple(interface_mesh.elements())
    if not elements:
        raise ValueError("interface mesh has no elements")
    faces: list[tuple[Any, Any, Any]] = []
    incident: dict[int, list[int]] = {}
    points: dict[int, tuple[float, float]] = {}
    for index, element in enumerate(elements):
        if int(element.nnode()) != 3:
            raise ValueError("interface diagnostics require three-node Q2 line elements")
        nodes = tuple(element.node_pt(i) for i in range(3))
        if len({id(node) for node in nodes}) != 3:
            raise ValueError("an interface line element repeats a node")
        faces.append(nodes)
        for node in (nodes[0], nodes[2]):
            key = id(node)
            points[key] = (
                _node_position(node, slot, 0),
                _node_position(node, slot, 1),
            )
            incident.setdefault(key, []).append(index)
    if any(len(entries) > 2 for entries in incident.values()):
        raise ValueError("interface endpoint graph branches")
    terminals = [key for key, entries in incident.items() if len(entries) == 1]
    if len(terminals) != 2 or any(len(entries) not in (1, 2) for entries in incident.values()):
        raise ValueError("interface elements do not form one open chain")

    scale = _endpoint_scale(points.values())
    tolerance = max(256.0 * math.ulp(scale), 1e-14 * scale)
    necks = [key for key in terminals if abs(points[key][1]) <= tolerance]
    poles = [key for key in terminals if abs(points[key][0]) <= tolerance]
    # A negative-control perturbation may move an endpoint by a few ulps.  If
    # it is still unambiguously the lowest-z/highest-r endpoint, retain it and
    # let the displacement gate report the material change.
    if len(necks) != 1:
        lowest = min(terminals, key=lambda key: abs(points[key][1]))
        if sum(abs(points[key][1]) <= 64.0 * max(abs(points[lowest][1]), tolerance) for key in terminals) != 1:
            raise ValueError("interface has no unique plane neck endpoint")
        necks = [lowest]
    if len(poles) != 1:
        lowest = min(terminals, key=lambda key: abs(points[key][0]))
        if sum(abs(points[key][0]) <= 64.0 * max(abs(points[lowest][0]), tolerance) for key in terminals) != 1:
            raise ValueError("interface has no unique axis pole endpoint")
        poles = [lowest]
    if necks[0] == poles[0]:
        raise ValueError("interface neck and pole endpoints coincide")

    current = necks[0]
    previous_element: int | None = None
    visited: set[int] = set()
    result: list[InterfaceSegment] = []
    while True:
        choices = [index for index in incident[current] if index != previous_element]
        if not choices:
            break
        if len(choices) != 1:
            raise ValueError("interface chain continuation is ambiguous")
        index = choices[0]
        if index in visited:
            raise ValueError("interface chain contains a cycle")
        visited.add(index)
        first, middle, last = faces[index]
        if id(first) == current:
            result.append(
                InterfaceSegment(
                    (_node_position(first, slot, 0), _node_position(first, slot, 1)),
                    (_node_position(middle, slot, 0), _node_position(middle, slot, 1)),
                    (_node_position(last, slot, 0), _node_position(last, slot, 1)),
                )
            )
            next_node = last
        elif id(last) == current:
            result.append(
                InterfaceSegment(
                    (_node_position(last, slot, 0), _node_position(last, slot, 1)),
                    (_node_position(middle, slot, 0), _node_position(middle, slot, 1)),
                    (_node_position(first, slot, 0), _node_position(first, slot, 1)),
                )
            )
            next_node = first
        else:  # pragma: no cover - protected by endpoint incidence above
            raise ValueError("interface endpoint adjacency lost a node")
        previous_element, current = index, id(next_node)
    if current != poles[0] or len(visited) != len(faces):
        raise ValueError("interface chain is disconnected or misses an element")
    return tuple(result)


@dataclass(frozen=True)
class InterfaceMetrics:
    """Immutable geometry summary of a neck-to-pole Q2 interface chain."""

    segments: tuple[InterfaceSegment, ...]
    neck: tuple[float, float]
    pole: tuple[float, float]
    arclength: float
    minimum_radius: float
    maximum_segment_scale: float
    minimum_segment_scale: float

    @property
    def neck_radius(self) -> float:
        return self.neck[0]

    @property
    def pole_position(self) -> tuple[float, float]:
        return self.pole

    @property
    def length(self) -> float:
        return self.arclength


def interface_geometry(interface_mesh: Any, *, position_slot: int = 0) -> InterfaceMetrics:
    """Capture a live Q2 interface independently of the remesher."""
    segments = _interface_chain(interface_mesh, int(position_slot))
    scales = tuple(segment.scale for segment in segments)
    samples = [segment.evaluate(point)[0] for segment in segments for point in (-1.0, *GAUSS8_POINTS, 0.0, 1.0)]
    return InterfaceMetrics(
        segments=segments,
        neck=segments[0].start,
        pole=segments[-1].end,
        arclength=math.fsum(segment.arclength() for segment in segments),
        minimum_radius=min(samples),
        maximum_segment_scale=max(scales),
        minimum_segment_scale=min(scales),
    )


def _segment_partial_arclength(segment: InterfaceSegment, parameter: float) -> float:
    t = _finite(parameter, "interface parameter")
    if not -1.0 <= t <= 1.0:
        raise ValueError("interface parameter must lie in [-1,1]")
    if t == -1.0:
        return 0.0
    if t == 1.0:
        return segment.arclength()
    midpoint = 0.5 * (t - 1.0)
    half_width = 0.5 * (t + 1.0)
    return half_width * math.fsum(
        weight * math.hypot(*segment.derivative(midpoint + half_width * point))
        for point, weight in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS)
    )


def _curve_at_arclength(metrics: InterfaceMetrics, distance: float) -> tuple[tuple[float, float], tuple[float, float]]:
    total = metrics.arclength
    target = min(total, max(0.0, float(distance)))
    walked = 0.0
    for segment in metrics.segments:
        length = segment.arclength()
        if target <= walked + length or segment is metrics.segments[-1]:
            local_target = target - walked
            lo, hi = -1.0, 1.0
            for _ in range(72):
                mid = 0.5 * (lo + hi)
                if _segment_partial_arclength(segment, mid) < local_target:
                    lo = mid
                else:
                    hi = mid
            parameter = 0.5 * (lo + hi)
            tangent = segment.derivative(parameter)
            speed = math.hypot(*tangent)
            if speed == 0.0:
                raise ValueError("interface segment has a zero tangent")
            return segment.evaluate(parameter), (tangent[0] / speed, tangent[1] / speed)
        walked += length
    return metrics.segments[-1].end, (1.0, 0.0)  # pragma: no cover - total is finite


@dataclass(frozen=True)
class InterfaceDisplacement:
    """Before/after interface displacement resolved in local tangent frames."""

    maximum_displacement: float
    maximum_normal_error: float
    maximum_tangential_error: float
    maximum_normal_error_relative: float
    maximum_tangential_error_relative: float
    neck_change: float
    neck_radius_change: float
    pole_change: float
    pole_radius_change: float
    sample_count: int

    @property
    def local_normal_error(self) -> float:
        return self.maximum_normal_error_relative

    @property
    def local_tangential_error(self) -> float:
        return self.maximum_tangential_error_relative

    @property
    def normal_relative(self) -> float:
        return self.maximum_normal_error_relative

    @property
    def tangential_relative(self) -> float:
        return self.maximum_tangential_error_relative


def compare_interface_geometry(
    before: InterfaceMetrics, after: InterfaceMetrics
) -> InterfaceDisplacement:
    """Compare two captured interfaces by normalised arclength.

    The comparison uses the before-interface tangent to resolve displacement
    into normal and tangential parts.  Each component is normalised by the
    *after* segment's own Q2 arclength/chord scale, so a tiny bridge segment
    cannot be hidden by a global drop radius.
    """
    if before.arclength <= 0.0 or after.arclength <= 0.0:
        raise ValueError("interface arclength must be positive")
    # Recreation with the same graph can preserve every interface node
    # bit-for-bit while the arclength inversion below still accumulates a few
    # ulps.  Report the exact identity as exact; this also makes the identity
    # gate a direct node-preservation assertion rather than a quadrature
    # self-consistency test.
    def _machine_same(first: InterfaceSegment, second: InterfaceSegment) -> bool:
        return all(
            a == b or abs(a - b) <= 8.0 * max(math.ulp(a), math.ulp(b))
            for first_point, second_point in zip(
                (first.start, first.midpoint, first.end),
                (second.start, second.midpoint, second.end),
            )
            for a, b in zip(first_point, second_point)
        )

    if len(before.segments) == len(after.segments) and all(
        _machine_same(first, second) for first, second in zip(before.segments, after.segments)
    ):
        return InterfaceDisplacement(
            maximum_displacement=0.0,
            maximum_normal_error=0.0,
            maximum_tangential_error=0.0,
            maximum_normal_error_relative=0.0,
            maximum_tangential_error_relative=0.0,
            neck_change=0.0,
            neck_radius_change=0.0,
            pole_change=0.0,
            pole_radius_change=0.0,
            sample_count=len(after.segments) * 11,
        )
    cumulative = 0.0
    maximum_displacement = maximum_normal = maximum_tangent = 0.0
    maximum_normal_relative = maximum_tangent_relative = 0.0
    sample_count = 0
    for segment in after.segments:
        scale = max(segment.scale, math.ulp(after.arclength), math.ulp(1.0))
        for parameter in (-1.0, 0.0, 1.0, *GAUSS8_POINTS):
            partial = _segment_partial_arclength(segment, parameter)
            fraction = (cumulative + partial) / after.arclength
            old_point, old_tangent = _curve_at_arclength(before, fraction * before.arclength)
            new_point = segment.evaluate(parameter)
            displacement = (new_point[0] - old_point[0], new_point[1] - old_point[1])
            tangent_error = abs(displacement[0] * old_tangent[0] + displacement[1] * old_tangent[1])
            normal_error = abs(-displacement[0] * old_tangent[1] + displacement[1] * old_tangent[0])
            magnitude = math.hypot(*displacement)
            maximum_displacement = max(maximum_displacement, magnitude)
            maximum_normal = max(maximum_normal, normal_error)
            maximum_tangent = max(maximum_tangent, tangent_error)
            maximum_normal_relative = max(maximum_normal_relative, normal_error / scale)
            maximum_tangent_relative = max(maximum_tangent_relative, tangent_error / scale)
            sample_count += 1
        cumulative += segment.arclength()
    neck_delta = math.dist(after.neck, before.neck)
    pole_delta = math.dist(after.pole, before.pole)
    return InterfaceDisplacement(
        maximum_displacement=maximum_displacement,
        maximum_normal_error=maximum_normal,
        maximum_tangential_error=maximum_tangent,
        maximum_normal_error_relative=maximum_normal_relative,
        maximum_tangential_error_relative=maximum_tangent_relative,
        neck_change=neck_delta,
        neck_radius_change=abs(after.neck[0] - before.neck[0]),
        pole_change=pole_delta,
        pole_radius_change=abs(after.pole[0] - before.pole[0]),
        sample_count=sample_count,
    )


def _field_indices(
    element: Any,
    mesh: Any,
    names: tuple[str, str],
    supplied: Mapping[str, int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    mesh_fields: Mapping[str, int] = {}
    if supplied is not None:
        mesh_fields = supplied
    elif hasattr(mesh, "get_nodal_field_indices"):
        try:
            mesh_fields = dict(mesh.get_nodal_field_indices())
        except Exception:
            mesh_fields = {}
    for name in names:
        indices = []
        for local in range(int(element.nnode())):
            node = _node(element, local)
            index: int | None = None
            if hasattr(element, "get_nodal_index_by_name"):
                try:
                    candidate = int(element.get_nodal_index_by_name(node, name))
                    if candidate >= 0:
                        index = candidate
                except Exception:
                    index = None
            if index is None and name in mesh_fields:
                index = int(mesh_fields[name])
            if index is None or index < 0:
                raise ValueError(f"element does not provide Q2 velocity field {name!r} at local node {local}")
            indices.append(index)
        result.append(tuple(indices))
    return result[0], result[1]


def _velocity_fields(value: Sequence[str] | Mapping[str, str]) -> tuple[str, str]:
    if isinstance(value, Mapping):
        radial = value.get("r", value.get("velocity_r", value.get("velocity_x")))
        axial = value.get("z", value.get("velocity_z", value.get("velocity_y")))
        if radial is None or axial is None:
            raise ValueError("velocity field mapping must provide radial/r and axial/z names")
        return str(radial), str(axial)
    names = tuple(value)
    if len(names) != 2:
        raise ValueError("velocity_fields must contain radial and axial field names")
    return str(names[0]), str(names[1])


def _history_storage(mesh: Any) -> int:
    nodes = tuple(mesh.nodes()) if hasattr(mesh, "nodes") else ()
    if not nodes:
        elements = tuple(mesh.elements()) if hasattr(mesh, "elements") else ()
        if elements and int(elements[0].nnode()) > 0:
            nodes = (_node(elements[0], 0),)
        else:
            raise ValueError("bulk mesh has no nodes")
    if not hasattr(nodes[0], "ntstorage"):
        return 1
    storage = int(nodes[0].ntstorage())
    if storage <= 0:
        raise ValueError("bulk mesh has no field history slots")
    return storage


@dataclass(frozen=True)
class DivergenceMetrics:
    """Axisymmetric divergence norms indexed by velocity history slot."""

    l2_by_slot: Mapping[int, float]
    rms_by_slot: Mapping[int, float]
    maximum_abs_by_slot: Mapping[int, float]
    signed_integral_by_slot: Mapping[int, float]
    measure_by_slot: Mapping[int, float]
    sample_count_by_slot: Mapping[int, int]

    @property
    def l2_norm(self) -> float:
        return self.l2_by_slot[min(self.l2_by_slot)]

    @property
    def rms_norm(self) -> float:
        return self.rms_by_slot[min(self.rms_by_slot)]

    @property
    def maximum_abs(self) -> float:
        return self.maximum_abs_by_slot[min(self.maximum_abs_by_slot)]

    @property
    def l2(self) -> float:
        return self.l2_norm

    @property
    def pointwise_max(self) -> float:
        return self.maximum_abs

    @property
    def l2_by_history(self) -> Mapping[int, float]:
        return self.l2_by_slot


def axisymmetric_divergence_at(
    element: Any,
    coordinate: tuple[float, float],
    *,
    position_slot: int = 0,
    velocity_slot: int = 0,
    velocity_fields: Sequence[str] | Mapping[str, str] = ("velocity_x", "velocity_y"),
    field_indices: Mapping[str, int] | None = None,
) -> float:
    """Evaluate ``d u_r/dr + d u_z/dz + u_r/r`` at one local point.

    At the axis the regular axisymmetric limit ``u_r/r -> d u_r/dr`` is used
    when the nodal field vanishes there.  A non-zero radial velocity on the
    axis is reported as a singular divergence rather than silently clipped.
    """
    fields = _velocity_fields(velocity_fields)
    point, jacobian = _geometry_at(element, int(position_slot), coordinate)
    determinant = _determinant(jacobian)
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("cannot evaluate divergence on a degenerate physical Jacobian")
    radial_indices, axial_indices = _field_indices(element, None, fields, field_indices)
    radial_values = [
        _node_value(_node(element, i), velocity_slot, radial_indices[i])
        for i in range(int(element.nnode()))
    ]
    axial_values = [
        _node_value(_node(element, i), velocity_slot, axial_indices[i])
        for i in range(int(element.nnode()))
    ]
    radial, radial_xi, radial_eta = _tensor_value_and_derivatives(radial_values, coordinate, _element_order(element))
    axial, axial_xi, axial_eta = _tensor_value_and_derivatives(axial_values, coordinate, _element_order(element))
    r_xi, r_eta, z_xi, z_eta = jacobian
    radial_r = (radial_xi * z_eta - radial_eta * z_xi) / determinant
    axial_z = (-axial_xi * r_eta + axial_eta * r_xi) / determinant
    scale = max(1.0, abs(point[0]), abs(point[1]), *(abs(value) for value in radial_values))
    axis_tolerance = 256.0 * math.ulp(scale)
    if abs(point[0]) <= axis_tolerance:
        if abs(radial) <= axis_tolerance:
            hoop = radial_r
        else:
            return math.copysign(float("inf"), radial)
    else:
        hoop = radial / point[0]
    return radial_r + axial_z + hoop


def axisymmetric_divergence(
    mesh: Any,
    *,
    position_slot: int = 0,
    velocity_slots: Iterable[int] | int | None = None,
    velocity_fields: Sequence[str] | Mapping[str, str] = ("velocity_x", "velocity_y"),
    field_indices: Mapping[str, int] | None = None,
    quadrature_order: int = 8,
) -> DivergenceMetrics:
    """Return independently integrated weak/pointwise axisymmetric divergence."""
    if quadrature_order != 8:
        raise ValueError("the independent diagnostics use the fixed eight-point Gauss rule")
    slots = _as_slots(velocity_slots, tuple(range(_history_storage(mesh))))
    elements = tuple(mesh.elements())
    if not elements:
        raise ValueError("cannot evaluate divergence on an empty mesh")
    sums = {slot: [] for slot in slots}
    squares = {slot: [] for slot in slots}
    measures = {slot: [] for slot in slots}
    maxima = {slot: 0.0 for slot in slots}
    counts = {slot: 0 for slot in slots}
    for element in elements:
        if int(element.nnode()) != 9:
            raise ValueError("runtime velocity diagnostics require nine-node Q2 elements")
        # Resolve indices once for this element and use the public pointwise
        # evaluator's same direct basis below.  Supplying a fake mesh to the
        # helper is unnecessary because indices are passed explicitly.
        fields = _velocity_fields(velocity_fields)
        radial_indices, axial_indices = _field_indices(element, mesh, fields, field_indices)
        order = _element_order(element)
        radial_nodal = {
            slot: [_node_value(_node(element, i), slot, radial_indices[i]) for i in range(9)]
            for slot in slots
        }
        axial_nodal = {
            slot: [_node_value(_node(element, i), slot, axial_indices[i]) for i in range(9)]
            for slot in slots
        }
        for xi, weight_x in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS):
            for eta, weight_y in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS):
                point, jacobian = _geometry_at(element, position_slot, (xi, eta))
                determinant = _determinant(jacobian)
                r_xi, r_eta, z_xi, z_eta = jacobian
                if not math.isfinite(determinant) or determinant == 0.0:
                    # Keep the snapshot usable as a negative-control receipt:
                    # a folded/degenerate cell has infinite divergence rather
                    # than aborting before the Jacobian gate can explain it.
                    for slot in slots:
                        squares[slot].append(float("inf"))
                        measures[slot].append(0.0)
                        sums[slot].append((0.0, 0.0))
                        maxima[slot] = float("inf")
                        counts[slot] += 1
                    continue
                weight = weight_x * weight_y * TWO_PI * abs(point[0]) * abs(determinant)
                signed_weight = weight_x * weight_y * TWO_PI * point[0] * determinant
                if weight == 0.0:
                    continue
                for slot in slots:
                    radial, radial_xi, radial_eta = _tensor_value_and_derivatives(radial_nodal[slot], (xi, eta), order)
                    axial, axial_xi, axial_eta = _tensor_value_and_derivatives(axial_nodal[slot], (xi, eta), order)
                    radial_r = (radial_xi * z_eta - radial_eta * z_xi) / determinant
                    axial_z = (-axial_xi * r_eta + axial_eta * r_xi) / determinant
                    axis_tolerance = 256.0 * math.ulp(max(1.0, abs(point[0]), abs(point[1]), abs(radial)))
                    if abs(point[0]) <= axis_tolerance:
                        hoop = radial_r if abs(radial) <= axis_tolerance else math.copysign(float("inf"), radial)
                    else:
                        hoop = radial / point[0]
                    value = radial_r + axial_z + hoop
                    sums[slot].append((value, signed_weight))
                    squares[slot].append(value * value * weight)
                    measures[slot].append(weight)
                    maxima[slot] = max(maxima[slot], abs(value))
                    counts[slot] += 1
    l2 = {slot: math.sqrt(math.fsum(squares[slot])) for slot in slots}
    measure = {slot: math.fsum(measures[slot]) for slot in slots}
    rms = {slot: math.sqrt(math.fsum(squares[slot]) / measure[slot]) if measure[slot] > 0.0 else 0.0 for slot in slots}
    signed = {slot: math.fsum(value * weight for value, weight in sums[slot]) for slot in slots}
    return DivergenceMetrics(l2, rms, maxima, signed, measure, counts)


@dataclass(frozen=True)
class BoundaryFluxMetrics:
    """Axisymmetric outward volume flux through one meridian boundary."""

    boundary: str
    flux_by_slot: Mapping[int, float]
    sample_count: int

    @property
    def flux(self) -> float:
        return self.flux_by_slot[min(self.flux_by_slot)] if self.flux_by_slot else 0.0

    @property
    def net_flux(self) -> float:
        return self.flux


def _boundary_chain(boundary_mesh: Any, boundary: str, slot: int) -> tuple[InterfaceSegment, ...]:
    if boundary == "interface":
        return _interface_chain(boundary_mesh, slot)
    elements = tuple(boundary_mesh.elements())
    if not elements:
        return ()
    # Axis and plane facets are already one-dimensional Q2 InterfaceElements.
    # Build a chain by endpoint identity, then orient it from the natural
    # lower/axis endpoint.  Orientation is used only for traversal; plane and
    # axis normals below are prescribed physically.
    faces = []
    incidence: dict[int, list[int]] = {}
    points: dict[int, tuple[float, float]] = {}
    for index, element in enumerate(elements):
        if int(element.nnode()) != 3:
            raise ValueError(f"{boundary} boundary requires three-node line elements")
        nodes = tuple(element.node_pt(i) for i in range(3))
        faces.append(nodes)
        for node in (nodes[0], nodes[2]):
            key = id(node)
            points[key] = (_node_position(node, slot, 0), _node_position(node, slot, 1))
            incidence.setdefault(key, []).append(index)
    terminals = [key for key, entries in incidence.items() if len(entries) == 1]
    if len(terminals) != 2:
        raise ValueError(f"{boundary} boundary does not form one open chain")
    if boundary == "plane":
        start = min(terminals, key=lambda key: abs(points[key][0]))
    else:  # axis
        start = min(terminals, key=lambda key: points[key][1])
    current = start
    previous: int | None = None
    visited: set[int] = set()
    result = []
    while True:
        choices = [index for index in incidence[current] if index != previous]
        if not choices:
            break
        if len(choices) != 1:
            raise ValueError(f"{boundary} boundary continuation is ambiguous")
        index = choices[0]
        if index in visited:
            raise ValueError(f"{boundary} boundary has a cycle")
        visited.add(index)
        first, middle, last = faces[index]
        if id(first) == current:
            result.append(InterfaceSegment(
                (_node_position(first, slot, 0), _node_position(first, slot, 1)),
                (_node_position(middle, slot, 0), _node_position(middle, slot, 1)),
                (_node_position(last, slot, 0), _node_position(last, slot, 1)),
            ))
            next_node = last
        else:
            result.append(InterfaceSegment(
                (_node_position(last, slot, 0), _node_position(last, slot, 1)),
                (_node_position(middle, slot, 0), _node_position(middle, slot, 1)),
                (_node_position(first, slot, 0), _node_position(first, slot, 1)),
            ))
            next_node = first
        previous, current = index, id(next_node)
    if len(visited) != len(faces):
        raise ValueError(f"{boundary} boundary is disconnected")
    return tuple(result)


def boundary_flux(
    boundary_mesh: Any,
    boundary: str,
    *,
    velocity_slots: Iterable[int] | int | None = None,
    velocity_fields: Sequence[str] | Mapping[str, str] = ("velocity_x", "velocity_y"),
    field_indices: Mapping[str, int] | None = None,
) -> BoundaryFluxMetrics:
    """Integrate outward ``2*pi*r*u.n`` on plane/axis/interface facets."""
    if boundary not in {"axis", "plane", "interface"}:
        raise ValueError("flux is meaningful for axis, plane and interface boundaries only")
    fields = _velocity_fields(velocity_fields)
    # Interface meshes expose no own nodes; get the storage from a line node.
    elems = tuple(boundary_mesh.elements())
    if not elems:
        return BoundaryFluxMetrics(boundary, {}, 0)
    node = elems[0].node_pt(0)
    slots = _as_slots(velocity_slots, tuple(range(int(node.ntstorage())) if hasattr(node, "ntstorage") else (0,)))
    segments = _boundary_chain(boundary_mesh, boundary, 0)
    flux = {slot: [] for slot in slots}
    sample_count = 0
    for element in elems:
        if int(element.nnode()) != 3:
            raise ValueError("boundary flux requires three-node Q2 line elements")
        # Segment topology is used for geometry.  Match the line element's
        # own node values; no interpolation helper from pyoomph is called.
        radial_indices, axial_indices = _field_indices(element, boundary_mesh, fields, field_indices)
        nodes = tuple(element.node_pt(i) for i in range(3))
        radial_values = {slot: [_node_value(nodes[i], slot, radial_indices[i]) for i in range(3)] for slot in slots}
        axial_values = {slot: [_node_value(nodes[i], slot, axial_indices[i]) for i in range(3)] for slot in slots}
        basis_geometry = InterfaceSegment(
            (_node_position(nodes[0], 0, 0), _node_position(nodes[0], 0, 1)),
            (_node_position(nodes[1], 0, 0), _node_position(nodes[1], 0, 1)),
            (_node_position(nodes[2], 0, 0), _node_position(nodes[2], 0, 1)),
        )
        # Find the corresponding physical orientation.  Boundary elements are
        # generally in the same order as the chain, but interface elements in
        # this mesh deliberately include one reversed cap face.
        original_start = basis_geometry.start
        original_end = basis_geometry.end
        reversed_geometry = False
        if boundary == "interface":
            # Match both endpoints.  Matching only the first endpoint would
            # select the preceding segment at every shared vertex and reverse
            # all but the first line element.
            matching = next(
                (
                    segment
                    for segment in segments
                    if _point_close(segment.start, original_start)
                    and _point_close(segment.end, original_end)
                ),
                None,
            )
            if matching is None:
                matching = next(
                    (
                        segment
                        for segment in segments
                        if _point_close(segment.start, original_end)
                        and _point_close(segment.end, original_start)
                    ),
                    None,
                )
                reversed_geometry = matching is not None
            if reversed_geometry:
                basis_geometry = InterfaceSegment(basis_geometry.end, basis_geometry.midpoint, basis_geometry.start)
        for point, quadrature_weight in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS):
            position = basis_geometry.evaluate(point)
            tangent = basis_geometry.derivative(point)
            speed = math.hypot(*tangent)
            if speed == 0.0:
                raise ValueError("boundary Q2 line has a zero tangent")
            if boundary == "plane":
                normal = (0.0, -1.0)
            elif boundary == "axis":
                normal = (-1.0, 0.0)
            else:
                normal = (tangent[1] / speed, -tangent[0] / speed)
            # Geometry in ``basis_geometry`` may be reversed relative to node
            # values.  Use the corresponding local parameter for values.
            value_parameter = -point if reversed_geometry else point
            value_basis, _ = _lagrange_1d(value_parameter, 2)
            for slot in slots:
                radial = math.fsum(value_basis[i] * radial_values[slot][i] for i in range(3))
                axial = math.fsum(value_basis[i] * axial_values[slot][i] for i in range(3))
                flux[slot].append(
                    quadrature_weight * TWO_PI * position[0] * (radial * normal[0] + axial * normal[1]) * speed
                )
            sample_count += 1
    return BoundaryFluxMetrics(boundary, {slot: math.fsum(values) for slot, values in flux.items()}, sample_count)


def axisymmetric_volume(mesh_or_problem: Any, *, position_slot: int = 0) -> float:
    """Integrate one position history slot without requiring velocity fields."""
    bulk, _ = _resolve_bulk_and_interface(mesh_or_problem, None)
    elements = tuple(bulk.elements())
    if not elements:
        raise ValueError("cannot integrate an empty bulk mesh")
    terms: list[float] = []
    for element in elements:
        if int(element.nnode()) != 9:
            raise ValueError("axisymmetric volume requires nine-node Q2 elements")
        for xi, weight_x in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS):
            for eta, weight_y in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS):
                point, jacobian = _geometry_at(element, int(position_slot), (xi, eta))
                terms.append(weight_x * weight_y * TWO_PI * point[0] * _determinant(jacobian))
    return math.fsum(terms)


@dataclass(frozen=True)
class MeshDiagnostics:
    """Immutable direct-evaluation receipt for one live mesh state."""

    position_slots: tuple[int, ...]
    volume_by_slot: Mapping[int, float]
    minimum_physical_jacobian_by_slot: Mapping[int, float]
    minimum_scaled_jacobian_by_slot: Mapping[int, float]
    maximum_condition_number_by_slot: Mapping[int, float]
    minimum_radius_by_slot: Mapping[int, float]
    element_count: int
    node_count: int
    interface: InterfaceMetrics | None
    divergence: DivergenceMetrics
    boundary_fluxes: Mapping[str, BoundaryFluxMetrics]

    @property
    def volume(self) -> float:
        return self.volume_by_slot[min(self.volume_by_slot)]

    @property
    def volumes(self) -> Mapping[int, float]:
        return self.volume_by_slot

    @property
    def minimum_physical_jacobian(self) -> float:
        return self.minimum_physical_jacobian_by_slot[min(self.minimum_physical_jacobian_by_slot)]

    @property
    def minimum_scaled_jacobian(self) -> float:
        return self.minimum_scaled_jacobian_by_slot[min(self.minimum_scaled_jacobian_by_slot)]

    @property
    def maximum_condition_number(self) -> float:
        return self.maximum_condition_number_by_slot[min(self.maximum_condition_number_by_slot)]

    @property
    def min_jacobian(self) -> float:
        return self.minimum_physical_jacobian

    @property
    def min_scaled_jacobian(self) -> float:
        return self.minimum_scaled_jacobian

    @property
    def max_condition_number(self) -> float:
        return self.maximum_condition_number

    @property
    def divergence_l2(self) -> float:
        return self.divergence.l2_norm

    @property
    def divergence_norm(self) -> float:
        return self.divergence_l2

    @property
    def neck(self) -> tuple[float, float] | None:
        return None if self.interface is None else self.interface.neck

    @property
    def pole(self) -> tuple[float, float] | None:
        return None if self.interface is None else self.interface.pole

    @property
    def fluxes(self) -> Mapping[str, BoundaryFluxMetrics]:
        return self.boundary_fluxes

    @property
    def net_flux_by_slot(self) -> Mapping[int, float]:
        result: dict[int, float] = {}
        for receipt in self.boundary_fluxes.values():
            for slot, value in receipt.flux_by_slot.items():
                result[slot] = result.get(slot, 0.0) + value
        return result


def _resolve_bulk_and_interface(mesh_or_problem: Any, interface_mesh: Any | None) -> tuple[Any, Any | None]:
    bulk = mesh_or_problem
    if not hasattr(bulk, "elements") and hasattr(bulk, "get_mesh"):
        bulk = bulk.get_mesh("drop")
    if interface_mesh is None:
        try:
            interface_mesh = bulk.get_problem().get_mesh("drop/interface")
        except Exception:
            try:
                interface_mesh = mesh_or_problem.get_mesh("drop/interface")
            except Exception:
                interface_mesh = None
    return bulk, interface_mesh


def diagnose_mesh(
    mesh_or_problem: Any,
    *,
    interface_mesh: Any | None = None,
    position_slots: Iterable[int] | int | None = POSITION_SLOTS,
    velocity_slots: Iterable[int] | int | None = None,
    velocity_fields: Sequence[str] | Mapping[str, str] = ("velocity_x", "velocity_y"),
    field_indices: Mapping[str, int] | None = None,
    boundaries: Iterable[str] = ("axis", "plane", "interface"),
    quadrature_order: int = 8,
) -> MeshDiagnostics:
    """Capture direct geometry, conservation and divergence diagnostics.

    No problem solve, timestep or pyoomph observable is invoked.  The returned
    object owns only Python floats and Q2 interface points and is therefore
    safe to retain across ``force_remesh``.
    """
    if quadrature_order != 8:
        raise ValueError("independent diagnostics use the fixed eight-point Gauss rule")
    bulk, interface_mesh = _resolve_bulk_and_interface(mesh_or_problem, interface_mesh)
    elements = tuple(bulk.elements())
    if not elements:
        raise ValueError("cannot diagnose an empty bulk mesh")
    slots = _as_slots(position_slots, POSITION_SLOTS)
    volumes: dict[int, float] = {}
    minimum_jacobian: dict[int, float] = {}
    minimum_scaled: dict[int, float] = {}
    maximum_condition: dict[int, float] = {}
    minimum_radius: dict[int, float] = {}
    samples = _sample_coordinates(True)
    for slot in slots:
        volume_terms: list[float] = []
        jacobians: list[float] = []
        scaled_values: list[float] = []
        conditions: list[float] = []
        radii: list[float] = []
        for element in elements:
            if int(element.nnode()) != 9:
                raise ValueError("runtime remesh geometry diagnostics require nine-node Q2 elements")
            for coordinate in samples:
                point, jacobian = _geometry_at(element, slot, coordinate)
                determinant = _determinant(jacobian)
                scaled, condition = _quality_from_jacobian(jacobian)
                jacobians.append(determinant)
                scaled_values.append(scaled)
                conditions.append(condition)
                radii.append(point[0])
            for xi, weight_x in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS):
                for eta, weight_y in zip(GAUSS8_POINTS, GAUSS8_WEIGHTS):
                    point, jacobian = _geometry_at(element, slot, (xi, eta))
                    volume_terms.append(weight_x * weight_y * TWO_PI * point[0] * _determinant(jacobian))
        volumes[slot] = math.fsum(volume_terms)
        minimum_jacobian[slot] = min(jacobians)
        minimum_scaled[slot] = min(scaled_values)
        maximum_condition[slot] = max(conditions)
        minimum_radius[slot] = min(radii)

    divergence = axisymmetric_divergence(
        bulk,
        position_slot=0,
        velocity_slots=velocity_slots,
        velocity_fields=velocity_fields,
        field_indices=field_indices,
        quadrature_order=quadrature_order,
    )
    interface = interface_geometry(interface_mesh) if interface_mesh is not None else None
    boundary_receipts: dict[str, BoundaryFluxMetrics] = {}
    if interface_mesh is not None:
        for boundary in tuple(boundaries):
            if boundary == "interface":
                target = interface_mesh
            else:
                try:
                    target = bulk.get_problem().get_mesh(f"drop/{boundary}")
                except Exception:
                    try:
                        target = mesh_or_problem.get_mesh(f"drop/{boundary}")
                    except Exception:
                        continue
            boundary_receipts[boundary] = boundary_flux(
                target,
                boundary,
                velocity_slots=velocity_slots,
                velocity_fields=velocity_fields,
                field_indices=field_indices,
            )
    node_count = int(bulk.nnode()) if hasattr(bulk, "nnode") else len(tuple(bulk.nodes()))
    return MeshDiagnostics(
        position_slots=slots,
        volume_by_slot=volumes,
        minimum_physical_jacobian_by_slot=minimum_jacobian,
        minimum_scaled_jacobian_by_slot=minimum_scaled,
        maximum_condition_number_by_slot=maximum_condition,
        minimum_radius_by_slot=minimum_radius,
        element_count=len(elements),
        node_count=node_count,
        interface=interface,
        divergence=divergence,
        boundary_fluxes=boundary_receipts,
    )


@dataclass(frozen=True)
class RemeshGateThresholds:
    """Default gates for the frozen-state runtime remesh verification."""

    relative_volume: float = 1.0e-10
    relative_interface_normal: float = 5.0e-3
    relative_interface_tangential: float = 5.0e-3
    relative_neck: float = 5.0e-3
    relative_pole: float = 5.0e-3
    absolute_divergence: float = 1.0e-9
    absolute_divergence_correction: float = 1.0e-9
    absolute_flux_change: float = 1.0e-9
    relative_flux_change: float = 1.0e-10

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} threshold must be finite and non-negative")


@dataclass(frozen=True)
class RemeshReceipt:
    """Executable before/after receipt and named gate outcomes."""

    before: MeshDiagnostics
    after: MeshDiagnostics
    relative_volume_change_by_slot: Mapping[int, float]
    maximum_relative_volume_change: float
    interface: InterfaceDisplacement | None
    relative_neck_change: float
    relative_pole_change: float
    divergence_after_by_slot: Mapping[int, float]
    divergence_norm_change_by_slot: Mapping[int, float]
    divergence_correction_by_slot: Mapping[int, float]
    divergence_after: float
    divergence_correction: float
    flux_change_by_boundary_and_slot: Mapping[str, Mapping[int, float]]
    gates: Mapping[str, bool]
    failures: tuple[str, ...]
    thresholds: RemeshGateThresholds
    passed: bool

    @classmethod
    def from_meshes(cls, before: Any, after: Any, **kwargs: Any) -> "RemeshReceipt":
        """Construct a receipt directly from live meshes or snapshots."""
        return compare_remesh(before, after, **kwargs)

    @property
    def ok(self) -> bool:
        return self.passed

    @property
    def gate_passed(self) -> bool:
        return self.passed

    @property
    def volume_relative_change(self) -> float:
        return self.maximum_relative_volume_change

    @property
    def relative_volume_changes(self) -> Mapping[int, float]:
        return self.relative_volume_change_by_slot

    @property
    def max_relative_volume_change(self) -> float:
        return self.maximum_relative_volume_change

    @property
    def interface_normal_error_relative(self) -> float:
        return 0.0 if self.interface is None else self.interface.maximum_normal_error_relative

    @property
    def interface_tangential_error_relative(self) -> float:
        return 0.0 if self.interface is None else self.interface.maximum_tangential_error_relative

    @property
    def local_interface_normal_error(self) -> float:
        return self.interface_normal_error_relative

    @property
    def local_interface_tangential_error(self) -> float:
        return self.interface_tangential_error_relative

    @property
    def local_interface_reconstruction_normal_error(self) -> float:
        return self.interface_normal_error_relative

    @property
    def local_interface_reconstruction_tangential_error(self) -> float:
        return self.interface_tangential_error_relative

    @property
    def neck_change(self) -> float:
        return 0.0 if self.interface is None else self.interface.neck_change

    @property
    def pole_change(self) -> float:
        return 0.0 if self.interface is None else self.interface.pole_change

    @property
    def divergence_norm_before(self) -> float:
        return max(self.before.divergence.l2_by_slot.values(), default=0.0)

    @property
    def divergence_norm_after(self) -> float:
        return self.divergence_after

    @property
    def divergence_norm(self) -> float:
        return self.divergence_after

    @property
    def field_correction(self) -> float:
        return self.divergence_correction

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly machine-readable receipt."""
        return {
            "passed": self.passed,
            "gates": dict(self.gates),
            "failures": list(self.failures),
            "relative_volume_change_by_slot": dict(self.relative_volume_change_by_slot),
            "maximum_relative_volume_change": self.maximum_relative_volume_change,
            "interface_normal_error_relative": self.interface_normal_error_relative,
            "interface_tangential_error_relative": self.interface_tangential_error_relative,
            "relative_neck_change": self.relative_neck_change,
            "relative_pole_change": self.relative_pole_change,
            "divergence_after_by_slot": dict(self.divergence_after_by_slot),
            "divergence_norm_change_by_slot": dict(self.divergence_norm_change_by_slot),
            "divergence_correction_by_slot": dict(self.divergence_correction_by_slot),
            "flux_change_by_boundary_and_slot": {
                name: dict(values) for name, values in self.flux_change_by_boundary_and_slot.items()
            },
        }

    to_dict = as_dict


def _coerce_thresholds(value: RemeshGateThresholds | Mapping[str, float] | None) -> RemeshGateThresholds:
    if value is None:
        return RemeshGateThresholds()
    if isinstance(value, RemeshGateThresholds):
        return value
    aliases = {
        "volume_relative": "relative_volume",
        "relative_volume_change": "relative_volume",
        "interface_normal_relative": "relative_interface_normal",
        "interface_tangential_relative": "relative_interface_tangential",
        "neck_relative": "relative_neck",
        "pole_relative": "relative_pole",
        "divergence": "absolute_divergence",
        "divergence_correction": "absolute_divergence_correction",
        "flux_absolute": "absolute_flux_change",
        "flux_relative": "relative_flux_change",
    }
    fields = {}
    for key, target in aliases.items():
        if key in value and target not in value:
            fields[target] = float(value[key])
    fields.update({field: float(value[field]) for field in RemeshGateThresholds.__dataclass_fields__ if field in value})
    return RemeshGateThresholds(**fields)


def _coerce_snapshot(value: MeshDiagnostics | Any, **kwargs: Any) -> MeshDiagnostics:
    if isinstance(value, MeshDiagnostics):
        return value
    return diagnose_mesh(value, **kwargs)


def compare_remesh(
    before: MeshDiagnostics | Any,
    after: MeshDiagnostics | Any,
    *,
    thresholds: RemeshGateThresholds | Mapping[str, float] | None = None,
    before_interface_mesh: Any | None = None,
    after_interface_mesh: Any | None = None,
    interface_before: Any | None = None,
    interface_after: Any | None = None,
    **diagnostic_kwargs: Any,
) -> RemeshReceipt:
    """Build and evaluate a before/after runtime-remesh receipt."""
    first_kwargs = dict(diagnostic_kwargs)
    second_kwargs = dict(diagnostic_kwargs)
    if before_interface_mesh is not None or interface_before is not None:
        first_kwargs["interface_mesh"] = (
            before_interface_mesh if before_interface_mesh is not None else interface_before
        )
    if after_interface_mesh is not None or interface_after is not None:
        second_kwargs["interface_mesh"] = (
            after_interface_mesh if after_interface_mesh is not None else interface_after
        )
    first = _coerce_snapshot(before, **first_kwargs)
    second = _coerce_snapshot(after, **second_kwargs)
    limits = _coerce_thresholds(thresholds)
    slots = tuple(sorted(set(first.volume_by_slot) & set(second.volume_by_slot)))
    if not slots:
        raise ValueError("before/after snapshots have no common position history slots")
    relative_volume = {
        slot: abs(second.volume_by_slot[slot] - first.volume_by_slot[slot])
        / max(abs(first.volume_by_slot[slot]), math.ulp(1.0))
        for slot in slots
    }
    maximum_volume = max(relative_volume.values())
    interface = None
    if first.interface is not None and second.interface is not None:
        interface = compare_interface_geometry(first.interface, second.interface)
    relative_neck = 0.0
    relative_pole = 0.0
    if interface is not None:
        relative_neck = interface.neck_radius_change / max(abs(first.interface.neck_radius), math.ulp(1.0))
        pole_scale = max(math.dist((0.0, 0.0), first.interface.pole), math.ulp(1.0))
        relative_pole = interface.pole_change / pole_scale

    divergence_slots = tuple(sorted(set(first.divergence.l2_by_slot) & set(second.divergence.l2_by_slot)))
    divergence_after = {slot: second.divergence.l2_by_slot[slot] for slot in divergence_slots}
    divergence_change = {
        slot: abs(second.divergence.l2_by_slot[slot] - first.divergence.l2_by_slot[slot])
        for slot in divergence_slots
    }
    divergence_correction = dict(divergence_change)
    maximum_divergence_after = max(divergence_after.values(), default=0.0)
    maximum_divergence_correction = max(divergence_correction.values(), default=0.0)

    boundary_names = set(first.boundary_fluxes) & set(second.boundary_fluxes)
    flux_changes: dict[str, dict[int, float]] = {}
    for name in sorted(boundary_names):
        common = set(first.boundary_fluxes[name].flux_by_slot) & set(second.boundary_fluxes[name].flux_by_slot)
        flux_changes[name] = {
            slot: abs(second.boundary_fluxes[name].flux_by_slot[slot] - first.boundary_fluxes[name].flux_by_slot[slot])
            for slot in sorted(common)
        }
    maximum_flux_change = max((value for changes in flux_changes.values() for value in changes.values()), default=0.0)
    maximum_flux_scale = max(
        (abs(first.boundary_fluxes[name].flux_by_slot[slot])
         for name in boundary_names
         for slot in set(first.boundary_fluxes[name].flux_by_slot) & set(second.boundary_fluxes[name].flux_by_slot)),
        default=0.0,
    )

    gates: dict[str, bool] = {
        "volume": maximum_volume <= limits.relative_volume,
        "jacobian": (
            all(value > 0.0 for value in second.minimum_physical_jacobian_by_slot.values())
            and all(value > 0.0 for value in second.minimum_scaled_jacobian_by_slot.values())
            and all(value >= 0.0 for value in second.minimum_radius_by_slot.values())
        ),
        "interface": interface is not None and (
            interface.maximum_normal_error_relative <= limits.relative_interface_normal
            and interface.maximum_tangential_error_relative <= limits.relative_interface_tangential
        ),
        "neck": interface is None or relative_neck <= limits.relative_neck,
        "pole": interface is None or relative_pole <= limits.relative_pole,
        "divergence": maximum_divergence_after <= limits.absolute_divergence,
        "divergence_correction": maximum_divergence_correction <= limits.absolute_divergence_correction,
        "flux": maximum_flux_change <= limits.absolute_flux_change
        + limits.relative_flux_change * max(1.0, maximum_flux_scale),
    }
    failures = tuple(name for name, passed in gates.items() if not passed)
    return RemeshReceipt(
        before=first,
        after=second,
        relative_volume_change_by_slot=relative_volume,
        maximum_relative_volume_change=maximum_volume,
        interface=interface,
        relative_neck_change=relative_neck,
        relative_pole_change=relative_pole,
        divergence_after_by_slot=divergence_after,
        divergence_norm_change_by_slot=divergence_change,
        divergence_correction_by_slot=divergence_correction,
        divergence_after=maximum_divergence_after,
        divergence_correction=maximum_divergence_correction,
        flux_change_by_boundary_and_slot=flux_changes,
        gates=gates,
        failures=failures,
        thresholds=limits,
        passed=not failures,
    )


def assert_remesh_gates(receipt: RemeshReceipt) -> RemeshReceipt:
    """Raise with named failures if a receipt is not accepted."""
    if not isinstance(receipt, RemeshReceipt):
        raise TypeError("assert_remesh_gates expects a RemeshReceipt")
    if not receipt.passed:
        raise AssertionError("runtime remesh diagnostics failed gates: " + ", ".join(receipt.failures))
    return receipt


# Short aliases make the public entry points easy to discover without tying
# callers to one spelling used in the first implementation.
measure_mesh = diagnose_mesh
mesh_diagnostics = diagnose_mesh
capture_diagnostics = diagnose_mesh
capture_remap_diagnostics = diagnose_mesh
evaluate_mesh_diagnostics = diagnose_mesh
remesh_diagnostics = compare_remesh
build_remesh_receipt = compare_remesh
build_before_after_receipt = compare_remesh
make_remesh_receipt = compare_remesh
interface_displacement = compare_interface_geometry
evaluate_divergence = axisymmetric_divergence
sample_mesh_quality = diagnose_mesh
volume = axisymmetric_volume
measure_axisymmetric_volume = axisymmetric_volume
RemapDiagnostics = MeshDiagnostics
BeforeAfterRemeshReceipt = RemeshReceipt
RemeshGateReceipt = RemeshReceipt


__all__ = [
    "GAUSS8_POINTS",
    "GAUSS8_WEIGHTS",
    "POSITION_SLOTS",
    "Q2_ROW_MAJOR_LOCAL_COORDINATES",
    "Q2_NODE_ORDER",
    "Q1_CORNER_INDICES",
    "InterfaceSegment",
    "InterfaceMetrics",
    "InterfaceDisplacement",
    "DivergenceMetrics",
    "BoundaryFluxMetrics",
    "MeshDiagnostics",
    "RemeshGateThresholds",
    "RemeshReceipt",
    "q2_basis",
    "q2_basis_derivative",
    "q2_basis_derivatives",
    "q1_basis",
    "q1_basis_derivative",
    "q1_basis_derivatives",
    "q1_weights",
    "evaluate_q1",
    "evaluate_q1_field",
    "interface_geometry",
    "compare_interface_geometry",
    "interface_displacement",
    "axisymmetric_divergence_at",
    "axisymmetric_divergence",
    "evaluate_divergence",
    "boundary_flux",
    "axisymmetric_volume",
    "measure_axisymmetric_volume",
    "volume",
    "diagnose_mesh",
    "measure_mesh",
    "mesh_diagnostics",
    "capture_diagnostics",
    "capture_remap_diagnostics",
    "evaluate_mesh_diagnostics",
    "compare_remesh",
    "remesh_diagnostics",
    "build_remesh_receipt",
    "build_before_after_receipt",
    "make_remesh_receipt",
    "sample_mesh_quality",
    "RemapDiagnostics",
    "BeforeAfterRemeshReceipt",
    "RemeshGateReceipt",
    "assert_remesh_gates",
]
