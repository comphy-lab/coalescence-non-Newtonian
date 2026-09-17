"""Pure-Python structured Q2 blocks for the Anthony initial quadrant.

The module deliberately has no pyoomph (or mesh-library) dependency.  A node
is identified by an integer topological key, not by rounding or deduplicating
its coordinates.  This makes the shared-edge contract explicit and keeps
physical spacings such as ``O(R0**3)`` visible at small bridge radii.

Each block uses a Coons/transfinite map.  At cap resolution ``c``, the bridge
has ``2*c`` elements in the divider direction so that its full divider edge
matches the two ``c``-element cap half-edges.  Its independently controlled
normal resolution defaults to two elements.  Each of the three cap patches is
``c x c``.  At the default resolutions this is the original seven-element
four-patch graph, with every shared edge represented by the same topological
keys at every resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Iterable

from .geometry import (
    eq6_ellipse_point,
    eq6_interface_hit,
    eq6_axis_point,
    initial_interface_hit_to_pole,
    initial_interface_neck_to_hit,
    sphere_centre_offset,
)


Point = tuple[float, float]
NodeKey = tuple[int, ...]
Curve = Callable[[float], Point]


# A scaled Jacobian is the sine of the angle between the two mapped
# coordinate directions.  Values below sin(20 degrees) already represent a
# nearly collapsed corner even when the raw determinant remains positive.
# The four-patch topology is required to retain this margin across the R0
# convergence sequence; production-resolution grids will impose additional
# element-size and conditioning gates.
MIN_TOPOLOGY_SCALED_JACOBIAN = 0.35


# Q2 local ordering is corners counter-clockwise, edge midpoints, then centre.
# Coordinates are in [0,1] so they can be used directly by the Coons maps.
Q2_REFERENCE_COORDINATES: tuple[Point, ...] = (
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (0.5, 0.0),
    (1.0, 0.5),
    (0.5, 1.0),
    (0.0, 0.5),
    (0.5, 0.5),
)
_Q2_GRID_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (2, 0),
    (2, 2),
    (0, 2),
    (1, 0),
    (2, 1),
    (1, 2),
    (0, 1),
    (1, 1),
)


def _clamp01(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("parameter must be finite")
    return min(1.0, max(0.0, value))


def _same_coordinate_to_ulps(first: float, second: float, *, ulps: int = 4) -> bool:
    """Compare two independently evaluated edge coordinates at machine scale.

    Shared block edges call the same analytical curve with reversed or split
    parameterisations.  Those paths can differ by one rounding unit even when
    they name the same topological point.  An ULP check permits only that
    evaluation noise; unlike an absolute tolerance it cannot hide a material
    error at the bridge-height or curvature scales.
    """
    if first == second:
        return True
    if not (math.isfinite(first) and math.isfinite(second)):
        return False
    tolerance = ulps * max(math.ulp(first), math.ulp(second))
    return abs(first - second) <= tolerance


@dataclass(frozen=True)
class Grading:
    """A monotone map of a unit interval and its inverse."""

    forward: Callable[[float], float]
    inverse_function: Callable[[float], float] | None = None

    def __call__(self, value: float) -> float:
        value = _clamp01(value)
        out = float(self.forward(value))
        if not math.isfinite(out) or out < -1e-14 or out > 1.0 + 1e-14:
            raise ValueError("grading must map [0,1] into [0,1]")
        return _clamp01(out)

    def inverse(self, value: float) -> float:
        value = _clamp01(value)
        if self.inverse_function is not None:
            out = float(self.inverse_function(value))
            if not math.isfinite(out):
                raise ValueError("grading inverse returned a non-finite value")
            return _clamp01(out)
        # A bounded inverse makes arbitrary user-supplied monotone callables
        # usable without importing a numerical package.  The fixed iteration
        # count gives deterministic node coordinates.
        lo, hi = 0.0, 1.0
        for _ in range(72):
            mid = 0.5 * (lo + hi)
            if self(mid) < value:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


uniform_grading = Grading(lambda value: value, lambda value: value)


def power_grading(power: float) -> Grading:
    """Return the monotone ``t**power`` grading used by a block direction."""
    power = float(power)
    if not math.isfinite(power) or power <= 0.0:
        raise ValueError("grading power must be positive and finite")
    return Grading(lambda value: value**power, lambda value: value ** (1.0 / power))


def _as_grading(value: Grading | Callable[[float], float] | None) -> Grading:
    if value is None:
        return uniform_grading
    if isinstance(value, Grading):
        # Check the endpoints and midpoint early, before any topology is made.
        value(0.0)
        value(1.0)
        value(0.5)
        if value(0.0) > value(0.5) or value(0.5) > value(1.0):
            raise ValueError("grading must be monotone increasing")
        return value
    if not callable(value):
        raise TypeError("grading must be callable")
    wrapped = Grading(value)
    wrapped(0.0)
    wrapped(1.0)
    wrapped(0.5)
    if wrapped(0.0) > wrapped(0.5) or wrapped(0.5) > wrapped(1.0):
        raise ValueError("grading must be monotone increasing")
    return wrapped


def validate_resolution(value: int, name: str) -> int:
    """Return a strictly positive integer mesh resolution.

    ``bool`` is excluded explicitly even though it is an ``int`` subclass;
    accepting it would make configuration mistakes silently select levels.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def line_curve(p0: Point, p1: Point) -> Curve:
    """Return an affine curve from ``p0`` to ``p1``."""
    def curve(t: float) -> Point:
        t = _clamp01(t)
        return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))

    return curve


def subcurve(curve: Curve, start: float, end: float) -> Curve:
    """Restrict a curve to ``[start,end]`` while preserving its orientation."""
    start, end = float(start), float(end)
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("curve interval must be finite")

    def restricted(t: float) -> Point:
        return curve(start + (end - start) * _clamp01(t))

    return restricted


def coons_map(
    bottom: Curve, right: Curve, top: Curve, left: Curve
) -> Callable[[float, float], Point]:
    """Build a bilinearly blended Coons map from four oriented boundaries.

    The boundary orientation is ``bottom: p00->p10``, ``right: p10->p11``,
    ``top: p01->p11`` and ``left: p00->p01``.  Explicit boundary returns avoid
    injecting round-off at a shared edge into the interior blend.
    """
    p00 = bottom(0.0)
    p10 = bottom(1.0)
    p01 = top(0.0)
    p11 = top(1.0)

    def mapping(u: float, v: float) -> Point:
        u, v = _clamp01(u), _clamp01(v)
        if v == 0.0:
            return bottom(u)
        if v == 1.0:
            return top(u)
        if u == 0.0:
            return left(v)
        if u == 1.0:
            return right(v)
        b = bottom(u)
        t = top(u)
        l = left(v)
        r = right(v)
        bilinear = (
            (1.0 - u) * (1.0 - v) * p00[0]
            + u * (1.0 - v) * p10[0]
            + (1.0 - u) * v * p01[0]
            + u * v * p11[0],
            (1.0 - u) * (1.0 - v) * p00[1]
            + u * (1.0 - v) * p10[1]
            + (1.0 - u) * v * p01[1]
            + u * v * p11[1],
        )
        return (
            (1.0 - v) * b[0] + v * t[0] + (1.0 - u) * l[0] + u * r[0] - bilinear[0],
            (1.0 - v) * b[1] + v * t[1] + (1.0 - u) * l[1] + u * r[1] - bilinear[1],
        )

    return mapping


# A descriptive alias for callers that use the transfinite-map terminology.
transfinite_map = coons_map


@dataclass(frozen=True)
class _Edge:
    edge_id: int
    curve: Curve
    start: NodeKey
    end: NodeKey
    label: str | None = None


@dataclass(frozen=True)
class _EdgeRef:
    edge: _Edge
    start: Fraction
    end: Fraction

    @property
    def orientation(self) -> int:
        return 1 if self.end >= self.start else -1


def _graded_edge_curve(ref: _EdgeRef, local_grading: Grading, edge_grading: Grading) -> Curve:
    """Curve wrapper that makes a graded edge conform across split blocks."""
    start, end = float(ref.start), float(ref.end)

    def curve(graded_parameter: float) -> Point:
        # The block map passes a graded coordinate.  Invert it to recover the
        # ungraded local coordinate, then apply grading on the global edge
        # coordinate.  Thus a bridge edge [0,1] and two cap pieces [0,1/2],
        # [1/2,1] evaluate exactly at the same physical locations.
        local = local_grading.inverse(graded_parameter)
        # ``edge_grading`` is shared by every patch touching this edge.  The
        # local block may use a different grading in its transverse direction,
        # but the physical position at a given ungraded edge coordinate must
        # remain identical on both sides.
        global_parameter = edge_grading(start + (end - start) * local)
        return ref.edge.curve(global_parameter)

    return curve


@dataclass
class Q2Patch:
    """A mapped structured block with explicit Q2 element subdivision."""

    patch_id: int
    name: str
    mapping: Callable[[float, float], Point]
    corners: tuple[NodeKey, NodeKey, NodeKey, NodeKey]
    edge_refs: tuple[_EdgeRef, _EdgeRef, _EdgeRef, _EdgeRef]
    edge_grading: Grading
    nu: int = 1
    nv: int = 1
    element_ids: list[tuple[int, int, int]] = field(default_factory=list)
    node_keys: set[NodeKey] = field(default_factory=set)

    def map(self, u: float, v: float) -> Point:
        return self.mapping(u, v)


@dataclass
class MeshNode:
    """A physical node keyed by topology, retaining all patch references."""

    key: NodeKey
    coordinates: Point
    patches: set[int] = field(default_factory=set)
    logical_coordinates: dict[int, Point] = field(default_factory=dict)


@dataclass(frozen=True)
class Q2Element:
    """One explicit nine-node quadrilateral element."""

    element_id: tuple[int, int, int]
    patch_id: int
    local_index: tuple[int, int]
    node_keys: tuple[NodeKey, ...]
    reference_coordinates: tuple[Point, ...] = Q2_REFERENCE_COORDINATES

    @property
    def connectivity(self) -> tuple[NodeKey, ...]:
        return self.node_keys


@dataclass(frozen=True)
class BoundaryFacet:
    """One Q2 edge facet, oriented canonically along its topological edge."""

    facet_id: tuple[int, int, int]
    patch_id: int
    element_id: tuple[int, int, int]
    side: str
    edge_id: int
    orientation: int
    node_keys: tuple[NodeKey, NodeKey, NodeKey]

    @property
    def connectivity(self) -> tuple[NodeKey, NodeKey, NodeKey]:
        return self.node_keys


@dataclass
class StructuredQ2Graph:
    """Conforming four-patch graph and metadata for later mesh integration."""

    R0: float
    Z0: float
    nodes: dict[NodeKey, MeshNode]
    elements: tuple[Q2Element, ...]
    boundary_facets: dict[str, tuple[BoundaryFacet, ...]]
    patches: dict[int, Q2Patch]
    shared_node_keys: frozenset[NodeKey]
    vertices: dict[str, NodeKey]
    edges: dict[int, _Edge]
    hit_parameter: float
    divider_hit: Point
    cap_resolution: int
    bridge_normal_resolution: int

    @property
    def node_coordinates(self) -> dict[NodeKey, Point]:
        return {key: node.coordinates for key, node in self.nodes.items()}

    @property
    def connectivity(self) -> tuple[tuple[NodeKey, ...], ...]:
        return tuple(element.node_keys for element in self.elements)

    @property
    def patch_identity(self) -> dict[tuple[int, int, int], int]:
        return {element.element_id: element.patch_id for element in self.elements}

    def element_map(self, element: Q2Element, u: float, v: float) -> Point:
        """Evaluate the nine-node Q2 interpolant for ``element``."""
        u, v = _clamp01(u), _clamp01(v)
        lx = (2.0 * (u - 0.5) * (u - 1.0), 4.0 * u * (1.0 - u), 2.0 * u * (u - 0.5))
        ly = (2.0 * (v - 0.5) * (v - 1.0), 4.0 * v * (1.0 - v), 2.0 * v * (v - 0.5))
        points = [self.nodes[key].coordinates for key in element.node_keys]
        tensor_indices = ((0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1))
        x = y = 0.0
        for point, (ix, iy) in zip(points, tensor_indices):
            weight = lx[ix] * ly[iy]
            x += weight * point[0]
            y += weight * point[1]
        return (x, y)

    def element_jacobian(self, element: Q2Element, u: float, v: float) -> tuple[float, float, float, float]:
        """Return ``(dx/du, dx/dv, dz/du, dz/dv)`` for a Q2 interpolant."""
        u, v = _clamp01(u), _clamp01(v)
        lx = (2.0 * (u - 0.5) * (u - 1.0), 4.0 * u * (1.0 - u), 2.0 * u * (u - 0.5))
        ly = (2.0 * (v - 0.5) * (v - 1.0), 4.0 * v * (1.0 - v), 2.0 * v * (v - 0.5))
        dlx = (4.0 * u - 3.0, 4.0 - 8.0 * u, 4.0 * u - 1.0)
        dly = (4.0 * v - 3.0, 4.0 - 8.0 * v, 4.0 * v - 1.0)
        points = [self.nodes[key].coordinates for key in element.node_keys]
        tensor_indices = ((0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1))
        dxdu = dxdv = dzdu = dzdv = 0.0
        for point, (ix, iy) in zip(points, tensor_indices):
            dxdu += dlx[ix] * ly[iy] * point[0]
            dxdv += lx[ix] * dly[iy] * point[0]
            dzdu += dlx[ix] * ly[iy] * point[1]
            dzdv += lx[ix] * dly[iy] * point[1]
        return dxdu, dxdv, dzdu, dzdv

    def scaled_jacobian(self, element: Q2Element, u: float, v: float) -> float:
        """Return determinant divided by local edge-vector magnitudes."""
        dxdu, dxdv, dzdu, dzdv = self.element_jacobian(element, u, v)
        det = dxdu * dzdv - dxdv * dzdu
        scale = math.hypot(dxdu, dzdu) * math.hypot(dxdv, dzdv)
        return det / scale if scale > 0.0 else float("nan")

    def minimum_scaled_jacobian(self, samples: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0)) -> float:
        values = [self.scaled_jacobian(element, u, v) for element in self.elements for u in samples for v in samples]
        return min(values) if values else float("nan")


def _vertex_key(number: int) -> NodeKey:
    return (0, int(number))


def _edge_node_key(edge_id: int, parameter: Fraction) -> NodeKey:
    parameter = Fraction(parameter)
    if parameter <= 0 or parameter >= 1:
        raise ValueError("edge interior node parameter must lie strictly inside (0,1)")
    return (1, int(edge_id), parameter.numerator, parameter.denominator)


def _face_node_key(patch_id: int, i: int, j: int) -> NodeKey:
    return (2, int(patch_id), int(i), int(j))


def _make_patch(
    patch_id: int,
    name: str,
    corners: tuple[NodeKey, NodeKey, NodeKey, NodeKey],
    edge_refs: tuple[_EdgeRef, _EdgeRef, _EdgeRef, _EdgeRef],
    *,
    nu: int,
    nv: int,
    grading_u: Grading,
    grading_v: Grading,
    edge_grading: Grading,
) -> Q2Patch:
    if nu < 1 or nv < 1:
        raise ValueError("Q2 subdivision counts must be positive")
    bottom, right, top, left = edge_refs
    base = coons_map(
        _graded_edge_curve(bottom, grading_u, edge_grading),
        _graded_edge_curve(right, grading_v, edge_grading),
        _graded_edge_curve(top, grading_u, edge_grading),
        _graded_edge_curve(left, grading_v, edge_grading),
    )

    def mapping(u: float, v: float) -> Point:
        return base(grading_u(_clamp01(u)), grading_v(_clamp01(v)))

    return Q2Patch(
        patch_id=patch_id,
        name=name,
        mapping=mapping,
        corners=corners,
        edge_refs=edge_refs,
        edge_grading=edge_grading,
        nu=nu,
        nv=nv,
    )


def _local_node_key(patch: Q2Patch, i: int, j: int) -> NodeKey:
    max_i, max_j = 2 * patch.nu, 2 * patch.nv
    if (i, j) == (0, 0):
        return patch.corners[0]
    if (i, j) == (max_i, 0):
        return patch.corners[1]
    if (i, j) == (max_i, max_j):
        return patch.corners[2]
    if (i, j) == (0, max_j):
        return patch.corners[3]
    if j == 0:
        ref = patch.edge_refs[0]
        q = ref.start + (ref.end - ref.start) * Fraction(i, max_i)
        return _edge_node_key(ref.edge.edge_id, q)
    if i == max_i:
        ref = patch.edge_refs[1]
        q = ref.start + (ref.end - ref.start) * Fraction(j, max_j)
        return _edge_node_key(ref.edge.edge_id, q)
    if j == max_j:
        ref = patch.edge_refs[2]
        q = ref.start + (ref.end - ref.start) * Fraction(i, max_i)
        return _edge_node_key(ref.edge.edge_id, q)
    if i == 0:
        ref = patch.edge_refs[3]
        q = ref.start + (ref.end - ref.start) * Fraction(j, max_j)
        return _edge_node_key(ref.edge.edge_id, q)
    return _face_node_key(patch.patch_id, i, j)


def _register_patch(
    patch: Q2Patch,
    nodes: dict[NodeKey, MeshNode],
    elements: list[Q2Element],
    facets: dict[str, list[BoundaryFacet]],
) -> None:
    max_i, max_j = 2 * patch.nu, 2 * patch.nv
    for j in range(max_j + 1):
        for i in range(max_i + 1):
            key = _local_node_key(patch, i, j)
            coordinate = patch.mapping(i / max_i, j / max_j)
            if key[0] == 1:
                # Independently parameterised full and half edges can reach
                # the same rational station through slightly different float
                # arithmetic (notably thirds).  Evaluate every edge node from
                # its canonical Fraction key so shared coordinates are bitwise
                # deterministic without widening the ULP consistency gate.
                edge_id = key[1]
                edge = next(
                    ref.edge for ref in patch.edge_refs if ref.edge.edge_id == edge_id
                )
                q = Fraction(key[2], key[3])
                coordinate = edge.curve(patch.edge_grading(float(q)))
            if not all(math.isfinite(value) for value in coordinate):
                raise ValueError(f"non-finite coordinate in patch {patch.patch_id}: {coordinate}")
            patch.node_keys.add(key)
            node = nodes.get(key)
            if node is None:
                node = MeshNode(key=key, coordinates=coordinate)
                nodes[key] = node
            else:
                # A shared topological key must have one physical coordinate.
                # Do not merge any other keys, even if their coordinates happen
                # to coincide.
                if not all(
                    _same_coordinate_to_ulps(first, second)
                    for first, second in zip(node.coordinates, coordinate)
                ):
                    raise ValueError(
                        f"inconsistent coordinates for shared key {key}: "
                        f"{node.coordinates!r} != {coordinate!r}"
                    )
            node.patches.add(patch.patch_id)
            node.logical_coordinates[patch.patch_id] = (i / max_i, j / max_j)

    for ey in range(patch.nv):
        for ex in range(patch.nu):
            i0, j0 = 2 * ex, 2 * ey
            local_keys = tuple(
                _local_node_key(patch, i0 + di, j0 + dj)
                for di, dj in _Q2_GRID_OFFSETS
            )
            element_id = (patch.patch_id, ex, ey)
            element = Q2Element(element_id=element_id, patch_id=patch.patch_id, local_index=(ex, ey), node_keys=local_keys)
            patch.element_ids.append(element_id)
            elements.append(element)

            side_data = (
                ("bottom", 0, (0, 0), (2, 0), (1, 0)),
                ("right", 1, (2, 0), (2, 2), (0, 1)),
                ("top", 2, (0, 2), (2, 2), (1, 0)),
                ("left", 3, (0, 0), (0, 2), (0, 1)),
            )
            for side, side_number, first, last, delta in side_data:
                ref = patch.edge_refs[side_number]
                if side == "bottom" and ey != 0:
                    continue
                if side == "right" and ex != patch.nu - 1:
                    continue
                if side == "top" and ey != patch.nv - 1:
                    continue
                if side == "left" and ex != 0:
                    continue
                fx, fy = first
                dx, dy = delta
                local_facet = tuple(
                    _local_node_key(patch, i0 + fx + k * dx, j0 + fy + k * dy)
                    for k in range(3)
                )
                facet_nodes = local_facet if ref.orientation > 0 else tuple(reversed(local_facet))
                label = ref.edge.label
                if label is not None:
                    facet = BoundaryFacet(
                        facet_id=(patch.patch_id, ex, ey),
                        patch_id=patch.patch_id,
                        element_id=element_id,
                        side=side,
                        edge_id=ref.edge.edge_id,
                        orientation=ref.orientation,
                        node_keys=facet_nodes,
                    )
                    facets.setdefault(label, []).append(facet)


def _edge_ref(edge: _Edge, start: Fraction | int | float, end: Fraction | int | float) -> _EdgeRef:
    return _EdgeRef(edge=edge, start=Fraction(start), end=Fraction(end))


def _add_edge(
    edges: dict[int, _Edge], edge_id: int, curve: Curve, start: NodeKey, end: NodeKey, label: str | None
) -> _Edge:
    edge = _Edge(edge_id=edge_id, curve=curve, start=start, end=end, label=label)
    edges[edge_id] = edge
    return edge


def build_curve_driven_four_patch_graph(
    R0: float,
    Z0: float | None = None,
    *,
    neck: Point,
    pole: Point,
    interface: Curve,
    interface_hit_parameter: float,
    divider_hit_parameter: float,
    cap_resolution: int = 1,
    bridge_normal_resolution: int = 2,
    grading_u: Grading | Callable[[float], float] | None = None,
    grading_v: Grading | Callable[[float], float] | None = None,
) -> StructuredQ2Graph:
    """Build the four-patch graph from an oriented neck-to-pole interface.

    ``interface`` is a full current free-surface curve with parameter zero at
    ``neck`` and one at ``pole``.  Its intersection parameter and the matching
    Eq. 6 divider parameter are explicit receipts from the geometry layer.
    The topology, resolution and grading contracts are otherwise identical to
    the analytical initial graph.
    """
    R0 = float(R0)
    if not math.isfinite(R0) or R0 <= 0.0:
        raise ValueError("R0 must be positive and finite")
    if Z0 is None:
        Z0 = 0.5 * R0 * R0
    Z0 = float(Z0)
    if not math.isfinite(Z0) or Z0 <= 0.0:
        raise ValueError("Z0 must be positive and finite")
    cap_resolution = validate_resolution(cap_resolution, "cap_resolution")
    bridge_normal_resolution = validate_resolution(
        bridge_normal_resolution, "bridge_normal_resolution"
    )
    gu, gv = _as_grading(grading_u), _as_grading(grading_v)

    neck = (float(neck[0]), float(neck[1]))
    pole = (float(pole[0]), float(pole[1]))
    if not all(math.isfinite(value) for point in (neck, pole) for value in point):
        raise ValueError("neck and pole must be finite")
    if not _same_coordinate_to_ulps(neck[0], R0, ulps=8) or neck[1] != 0.0:
        raise ValueError("neck must be (R0, 0) to binary64 resolution")
    if pole[0] != 0.0:
        raise ValueError("pole must lie on the axis")
    if not callable(interface):
        raise TypeError("interface must be callable")
    interface_hit_parameter = float(interface_hit_parameter)
    hit_parameter = float(divider_hit_parameter)
    if not 0.0 < interface_hit_parameter < 1.0:
        raise ValueError("interface hit parameter must lie strictly inside (0,1)")
    if not 0.0 < hit_parameter < 1.0:
        raise ValueError("divider hit parameter must lie strictly inside (0,1)")
    if not all(
        _same_coordinate_to_ulps(first, second, ulps=8)
        for first, second in zip(interface(0.0), neck)
    ):
        raise ValueError("interface start does not match neck")
    if not all(
        _same_coordinate_to_ulps(first, second, ulps=8)
        for first, second in zip(interface(1.0), pole)
    ):
        raise ValueError("interface end does not match pole")
    hit = interface(interface_hit_parameter)
    analytical_hit = eq6_ellipse_point(R0, hit_parameter)
    if not all(
        _same_coordinate_to_ulps(first, second, ulps=8)
        for first, second in zip(hit, analytical_hit)
    ):
        raise ValueError("interface and Eq. 6 hit receipts are inconsistent")
    axis_divider = eq6_axis_point(R0)
    origin: Point = (0.0, 0.0)

    # Integer topological vertex keys are intentionally independent of the
    # coordinate values and of the grading functions.
    vertices = {
        "origin": _vertex_key(0),
        "neck": _vertex_key(1),
        "divider_axis": _vertex_key(2),
        "divider_hit": _vertex_key(3),
        "pole": _vertex_key(4),
        # Midpoint keys are edge-owned.  This makes the midpoint of the
        # bridge's two-element divider edge identical to the cap endpoint,
        # without a coordinate-based merge.
        "divider_mid": _edge_node_key(0, Fraction(1, 2)),
        "interface_mid": _edge_node_key(1, Fraction(1, 2)),
        "axis_mid": _edge_node_key(2, Fraction(1, 2)),
        "interior": _vertex_key(8),
    }
    points = {
        vertices["origin"]: origin,
        vertices["neck"]: neck,
        vertices["divider_axis"]: axis_divider,
        vertices["divider_hit"]: hit,
        vertices["pole"]: pole,
    }

    def divider(t: float) -> Point:
        t = _clamp01(t)
        # Use the captured interface coordinate at the common endpoint.  The
        # safeguarded intersection gate has already established that it lies
        # on Eq. 6 to binary64 accuracy, and this explicit endpoint makes all
        # patch registrations bitwise deterministic.
        return hit if t == 1.0 else eq6_ellipse_point(R0, hit_parameter * t)

    bridge_interface = subcurve(interface, 0.0, interface_hit_parameter)
    cap_interface = subcurve(interface, interface_hit_parameter, 1.0)
    axis_outer: Curve = line_curve(axis_divider, pole)

    # Edge midpoints are topological q=1/2 stations.  Apply the shared edge
    # grading here as well, so the explicit vertex coordinates agree with the
    # graded boundary maps when grading_u is non-uniform.
    mid_q = gu(0.5)
    mid_divider = divider(mid_q)
    mid_interface = cap_interface(mid_q)
    mid_axis = axis_outer(mid_q)
    # Use the curved-boundary midpoint triangle, not the triangle of corner
    # vertices.  As R0 -> 0 all three corner radii vanish even though the
    # spherical free surface reaches O(1) radius; their centroid therefore
    # drives the three internal cap edges into a nearly collinear meeting.
    # The midpoint centroid follows the actual curved domain and keeps the
    # central block angles bounded independently of R0.
    interior = (
        (mid_divider[0] + mid_interface[0] + mid_axis[0]) / 3.0,
        (mid_divider[1] + mid_interface[1] + mid_axis[1]) / 3.0,
    )
    points.update(
        {
            vertices["divider_mid"]: mid_divider,
            vertices["interface_mid"]: mid_interface,
            vertices["axis_mid"]: mid_axis,
            vertices["interior"]: interior,
        }
    )

    # Curves are retained analytically on the outer boundaries.  Interior
    # subdivision edges are straight and use exactly the same callable from
    # both adjacent blocks.
    edges: dict[int, _Edge] = {}
    e_divider = _add_edge(edges, 0, divider, vertices["divider_axis"], vertices["divider_hit"], "divider")
    e_interface = _add_edge(edges, 1, cap_interface, vertices["divider_hit"], vertices["pole"], "interface")
    e_axis = _add_edge(edges, 2, axis_outer, vertices["divider_axis"], vertices["pole"], "axis")
    e_plane = _add_edge(edges, 3, line_curve(origin, neck), vertices["origin"], vertices["neck"], "plane")
    e_bridge_interface = _add_edge(edges, 4, bridge_interface, vertices["neck"], vertices["divider_hit"], "interface")
    e_bridge_axis = _add_edge(edges, 5, line_curve(origin, axis_divider), vertices["origin"], vertices["divider_axis"], "axis")
    e_mab_p = _add_edge(edges, 6, line_curve(mid_divider, interior), vertices["divider_mid"], vertices["interior"], None)
    e_mca_p = _add_edge(edges, 7, line_curve(mid_axis, interior), vertices["axis_mid"], vertices["interior"], None)
    e_p_mbc = _add_edge(edges, 8, line_curve(interior, mid_interface), vertices["interior"], vertices["interface_mid"], None)

    patches = {
        0: _make_patch(
            0,
            "bridge",
            (vertices["origin"], vertices["neck"], vertices["divider_hit"], vertices["divider_axis"]),
            (_edge_ref(e_plane, 0, 1), _edge_ref(e_bridge_interface, 0, 1), _edge_ref(e_divider, 0, 1), _edge_ref(e_bridge_axis, 0, 1)),
            nu=2 * cap_resolution,
            nv=bridge_normal_resolution,
            grading_u=gu,
            grading_v=gv,
            edge_grading=gu,
        ),
        1: _make_patch(
            1,
            "cap-divider",
            (vertices["divider_axis"], vertices["divider_mid"], vertices["interior"], vertices["axis_mid"]),
            (_edge_ref(e_divider, 0, Fraction(1, 2)), _edge_ref(e_mab_p, 0, 1), _edge_ref(e_mca_p, 0, 1), _edge_ref(e_axis, 0, Fraction(1, 2))),
            nu=cap_resolution,
            nv=cap_resolution,
            grading_u=gu,
            grading_v=gv,
            edge_grading=gu,
        ),
        2: _make_patch(
            2,
            "cap-interface-lower",
            (vertices["divider_mid"], vertices["divider_hit"], vertices["interface_mid"], vertices["interior"]),
            (_edge_ref(e_divider, Fraction(1, 2), 1), _edge_ref(e_interface, 0, Fraction(1, 2)), _edge_ref(e_p_mbc, 0, 1), _edge_ref(e_mab_p, 0, 1)),
            nu=cap_resolution,
            nv=cap_resolution,
            grading_u=gu,
            grading_v=gv,
            edge_grading=gu,
        ),
        3: _make_patch(
            3,
            "cap-axis-upper",
            (vertices["axis_mid"], vertices["interior"], vertices["interface_mid"], vertices["pole"]),
            (_edge_ref(e_mca_p, 0, 1), _edge_ref(e_p_mbc, 0, 1), _edge_ref(e_interface, 1, Fraction(1, 2)), _edge_ref(e_axis, Fraction(1, 2), 1)),
            nu=cap_resolution,
            nv=cap_resolution,
            grading_u=gu,
            grading_v=gv,
            edge_grading=gu,
        ),
    }

    nodes: dict[NodeKey, MeshNode] = {}
    elements: list[Q2Element] = []
    facets: dict[str, list[BoundaryFacet]] = {}
    for patch in patches.values():
        _register_patch(patch, nodes, elements, facets)

    shared = frozenset(key for key, node in nodes.items() if len(node.patches) > 1)
    return StructuredQ2Graph(
        R0=R0,
        Z0=Z0,
        nodes=nodes,
        elements=tuple(elements),
        boundary_facets={label: tuple(items) for label, items in facets.items()},
        patches=patches,
        shared_node_keys=shared,
        vertices=vertices,
        edges=edges,
        hit_parameter=hit_parameter,
        divider_hit=hit,
        cap_resolution=cap_resolution,
        bridge_normal_resolution=bridge_normal_resolution,
    )


def build_initial_four_patch_graph(
    R0: float,
    Z0: float | None = None,
    *,
    cap_resolution: int = 1,
    bridge_normal_resolution: int = 2,
    grading_u: Grading | Callable[[float], float] | None = None,
    grading_v: Grading | Callable[[float], float] | None = None,
) -> StructuredQ2Graph:
    """Build the analytical Anthony initial graph through the curve API.

    This remains the source-compatible public constructor.  The composite
    interface is split at parameter one half so its two branches evaluate the
    same analytical formulae as the pre-refactor initial builder.
    """
    R0 = float(R0)
    if not math.isfinite(R0) or R0 <= 0.0:
        raise ValueError("R0 must be positive and finite")
    if Z0 is None:
        Z0 = 0.5 * R0 * R0
    Z0 = float(Z0)
    if not math.isfinite(Z0) or Z0 <= 0.0:
        raise ValueError("Z0 must be positive and finite")
    hit_parameter, hit = eq6_interface_hit(R0, R0, Z0)
    neck: Point = (R0, 0.0)
    pole: Point = (0.0, 2.0 - sphere_centre_offset(R0, Z0))

    def interface(parameter: float) -> Point:
        parameter = _clamp01(parameter)
        if parameter == 0.0:
            return neck
        if parameter == 0.5:
            return hit
        if parameter == 1.0:
            return pole
        if parameter <= 0.5:
            return initial_interface_neck_to_hit(2.0 * parameter, R0, Z0, hit)
        return initial_interface_hit_to_pole(2.0 * parameter - 1.0, R0, Z0, hit)

    return build_curve_driven_four_patch_graph(
        R0,
        Z0,
        neck=neck,
        pole=pole,
        interface=interface,
        interface_hit_parameter=0.5,
        divider_hit_parameter=hit_parameter,
        cap_resolution=cap_resolution,
        bridge_normal_resolution=bridge_normal_resolution,
        grading_u=grading_u,
        grading_v=grading_v,
    )


# Short aliases used by integration code and exploratory tests.
build_four_patch_graph = build_initial_four_patch_graph
build_initial_graph = build_initial_four_patch_graph


__all__ = [
    "BoundaryFacet",
    "Curve",
    "Grading",
    "MeshNode",
    "MIN_TOPOLOGY_SCALED_JACOBIAN",
    "NodeKey",
    "Point",
    "Q2Element",
    "Q2Patch",
    "Q2_REFERENCE_COORDINATES",
    "StructuredQ2Graph",
    "build_curve_driven_four_patch_graph",
    "build_four_patch_graph",
    "build_initial_four_patch_graph",
    "build_initial_graph",
    "coons_map",
    "line_curve",
    "power_grading",
    "subcurve",
    "transfinite_map",
    "uniform_grading",
    "validate_resolution",
]
