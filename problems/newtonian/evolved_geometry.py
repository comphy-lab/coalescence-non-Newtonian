"""Exact piecewise-Q2 geometry for rebuilding an evolved four-patch mesh.

The runtime free surface is a chain of oriented three-node line elements.  This
module preserves those elements as quadratic polynomials; it never replaces
them by a polyline or a global spline.  The only approximation reported here
is therefore the error relative to an independently supplied reference curve.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

from .geometry import (
    eq6_conic_coeffs,
    eq6_conic_value,
    eq6_ellipse_parameters,
    eq6_ellipse_point,
)


Point = tuple[float, float]
Curve = Callable[[float], Point]
_EPS = math.ulp(1.0)
_GAUSS8_X = (
    -0.9602898564975363, -0.7966664774136267,
    -0.5255324099163290, -0.1834346424956498,
     0.1834346424956498,  0.5255324099163290,
     0.7966664774136267,  0.9602898564975363,
)
_GAUSS8_W = (
    0.1012285362903763, 0.2223810344533745,
    0.3137066458778873, 0.3626837833783620,
    0.3626837833783620, 0.3137066458778873,
    0.2223810344533745, 0.1012285362903763,
)


def _point(value: Sequence[float], name: str) -> Point:
    if len(value) != 2:
        raise ValueError(f"{name} must have two coordinates")
    out = float(value[0]), float(value[1])
    if not all(math.isfinite(component) for component in out):
        raise ValueError(f"{name} must be finite")
    return out


def _point_close(first: Point, second: Point, *, ulps: int = 8) -> bool:
    return all(
        a == b or abs(a - b) <= ulps * max(math.ulp(a), math.ulp(b))
        for a, b in zip(first, second)
    )


@dataclass(frozen=True)
class OrientedQ2Segment:
    """One oriented Q2 segment, stored as start/midpoint/end nodal values."""

    start: Point
    midpoint: Point
    end: Point

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _point(self.start, "start"))
        object.__setattr__(self, "midpoint", _point(self.midpoint, "midpoint"))
        object.__setattr__(self, "end", _point(self.end, "end"))
        scale = max(math.dist(self.start, self.midpoint),
                    math.dist(self.midpoint, self.end), math.dist(self.start, self.end))
        if scale == 0.0:
            raise ValueError("Q2 segment has zero extent")
        # The derivative is affine.  Its minimum norm is obtained by one
        # projection onto that affine line, so this is an exact fold/cusp gate
        # up to binary64 evaluation error rather than a sampling heuristic.
        d0 = self.derivative(0.0)
        d1 = self.derivative(1.0)
        delta = d1[0] - d0[0], d1[1] - d0[1]
        denominator = delta[0] * delta[0] + delta[1] * delta[1]
        station = 0.0 if denominator == 0.0 else min(
            1.0, max(0.0, -(d0[0] * delta[0] + d0[1] * delta[1]) / denominator)
        )
        speed = math.hypot(*self.derivative(station))
        if speed <= 128.0 * _EPS * scale:
            raise ValueError("Q2 segment is folded or has an unresolved cusp")

    def evaluate(self, parameter: float) -> Point:
        t = float(parameter)
        if not math.isfinite(t) or not 0.0 <= t <= 1.0:
            raise ValueError("segment parameter must lie in [0,1]")
        l0 = (1.0 - t) * (1.0 - 2.0 * t)
        lm = 4.0 * t * (1.0 - t)
        l1 = t * (2.0 * t - 1.0)
        return (
            l0 * self.start[0] + lm * self.midpoint[0] + l1 * self.end[0],
            l0 * self.start[1] + lm * self.midpoint[1] + l1 * self.end[1],
        )

    __call__ = evaluate

    def derivative(self, parameter: float) -> Point:
        t = float(parameter)
        if not math.isfinite(t) or not 0.0 <= t <= 1.0:
            raise ValueError("segment parameter must lie in [0,1]")
        d0, dm, d1 = 4.0 * t - 3.0, 4.0 - 8.0 * t, 4.0 * t - 1.0
        return (
            d0 * self.start[0] + dm * self.midpoint[0] + d1 * self.end[0],
            d0 * self.start[1] + dm * self.midpoint[1] + d1 * self.end[1],
        )

    def second_derivative(self) -> Point:
        """Return the constant second derivative of this Q2 segment."""
        return (
            4.0 * self.start[0] - 8.0 * self.midpoint[0] + 4.0 * self.end[0],
            4.0 * self.start[1] - 8.0 * self.midpoint[1] + 4.0 * self.end[1],
        )

    def arclength(self) -> float:
        """Return the segment arclength by fixed eighth-order Gauss quadrature."""
        return 0.5 * math.fsum(
            weight * math.hypot(*self.derivative(0.5 * (abscissa + 1.0)))
            for abscissa, weight in zip(_GAUSS8_X, _GAUSS8_W)
        )

    def split(self, parameter: float) -> tuple[OrientedQ2Segment, OrientedQ2Segment]:
        t = float(parameter)
        if not math.isfinite(t) or not 0.0 < t < 1.0:
            raise ValueError("split parameter must lie strictly inside (0,1)")
        point = self.evaluate(t)
        return (
            OrientedQ2Segment(self.start, self.evaluate(0.5 * t), point),
            OrientedQ2Segment(point, self.evaluate(t + 0.5 * (1.0 - t)), self.end),
        )

    def subsegment(self, start: float, end: float) -> OrientedQ2Segment:
        a, b = float(start), float(end)
        if not (math.isfinite(a) and math.isfinite(b) and 0.0 <= a < b <= 1.0):
            raise ValueError("subsegment bounds must satisfy 0 <= start < end <= 1")
        return OrientedQ2Segment(
            self.evaluate(a), self.evaluate(0.5 * (a + b)), self.evaluate(b)
        )

    def reversed(self) -> OrientedQ2Segment:
        return OrientedQ2Segment(self.end, self.midpoint, self.start)


@dataclass(frozen=True)
class PiecewiseQ2Curve:
    """Continuous oriented Q2 segment chain with a uniform global parameter."""

    segments: tuple[OrientedQ2Segment, ...]

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("piecewise Q2 curve needs at least one segment")
        if not all(isinstance(segment, OrientedQ2Segment) for segment in segments):
            raise TypeError("segments must be OrientedQ2Segment instances")
        for first, second in zip(segments, segments[1:]):
            if not _point_close(first.end, second.start):
                raise ValueError(
                    f"disconnected oriented Q2 segments: {first.end!r} != {second.start!r}"
                )
        object.__setattr__(self, "segments", segments)

    @classmethod
    def from_nodes(cls, nodes: Iterable[Sequence[float]]) -> PiecewiseQ2Curve:
        """Build from oriented Q2 nodes ``p0, pm, p1, pm, p2, ...``."""
        points = tuple(_point(node, "node") for node in nodes)
        if len(points) < 3 or len(points) % 2 == 0:
            raise ValueError("an oriented Q2 chain needs 2*n+1 nodes")
        return cls(tuple(
            OrientedQ2Segment(points[index], points[index + 1], points[index + 2])
            for index in range(0, len(points) - 2, 2)
        ))

    @property
    def start(self) -> Point:
        return self.segments[0].start

    @property
    def end(self) -> Point:
        return self.segments[-1].end

    def _local(self, parameter: float) -> tuple[int, float]:
        t = float(parameter)
        if not math.isfinite(t) or not 0.0 <= t <= 1.0:
            raise ValueError("curve parameter must lie in [0,1]")
        count = len(self.segments)
        if t == 1.0:
            return count - 1, 1.0
        scaled = t * count
        index = min(count - 1, int(scaled))
        return index, scaled - index

    def evaluate(self, parameter: float) -> Point:
        index, local = self._local(parameter)
        return self.segments[index].evaluate(local)

    __call__ = evaluate

    def derivative(self, parameter: float) -> Point:
        index, local = self._local(parameter)
        derivative = self.segments[index].derivative(local)
        count = len(self.segments)
        return derivative[0] * count, derivative[1] * count

    def arclength(self) -> float:
        return math.fsum(segment.arclength() for segment in self.segments)

    def subcurve(self, start: float, end: float) -> PiecewiseQ2Curve:
        a, b = float(start), float(end)
        if not (math.isfinite(a) and math.isfinite(b) and 0.0 <= a < b <= 1.0):
            raise ValueError("subcurve bounds must satisfy 0 <= start < end <= 1")
        count = len(self.segments)
        first_i, first_t = self._local(a)
        last_i, last_t = self._local(b)
        pieces = []
        if first_i == last_i:
            pieces.append(self.segments[first_i].subsegment(first_t, last_t))
        else:
            if first_t == 0.0:
                pieces.append(self.segments[first_i])
            else:
                pieces.append(self.segments[first_i].subsegment(first_t, 1.0))
            pieces.extend(self.segments[first_i + 1:last_i])
            if last_t == 1.0:
                pieces.append(self.segments[last_i])
            elif last_t > 0.0:
                pieces.append(self.segments[last_i].subsegment(0.0, last_t))
        return PiecewiseQ2Curve(tuple(pieces))

    def reversed(self) -> PiecewiseQ2Curve:
        return PiecewiseQ2Curve(tuple(segment.reversed() for segment in reversed(self.segments)))


def axisymmetric_twice_mean_curvature_at_start(
    curve: PiecewiseQ2Curve,
) -> float:
    """Return ``k_m + k_phi`` at the neck of a neck-to-pole Q2 chain.

    For tangent ``(dr,dz)``, the outward meridional normal is
    ``(dz,-dr)/|t|``.  This fixes both curvature signs independently of the
    interface element's native orientation.
    """
    segment = curve.segments[0]
    radius = segment.start[0]
    dr, dz = segment.derivative(0.0)
    d2r, d2z = segment.second_derivative()
    speed = math.hypot(dr, dz)
    if radius <= 0.0 or speed <= 0.0:
        raise ValueError("axisymmetric neck curvature requires positive radius and speed")
    meridional = (d2r * dz - d2z * dr) / speed**3
    hoop = (dz / speed) / radius
    result = meridional + hoop
    if not math.isfinite(result):
        raise ValueError("axisymmetric neck curvature is non-finite")
    return result


@dataclass(frozen=True)
class Eq6Intersection:
    interface_parameter: float
    divider_parameter: float
    point: Point
    conic_residual: float
    transversality: float


@dataclass(frozen=True)
class ReconstructionError:
    maximum: float
    rms: float
    sample_count: int


def reconstruction_error(
    curve: PiecewiseQ2Curve,
    reference: Curve,
    *,
    samples_per_segment: int = 16,
) -> ReconstructionError:
    """Compare the Q2 chain with an independent equally parameterised curve."""
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least two")
    count = len(curve.segments) * samples_per_segment + 1
    errors = []
    for index in range(count):
        parameter = index / (count - 1)
        errors.append(math.dist(curve(parameter), _point(reference(parameter), "reference point")))
    return ReconstructionError(max(errors), math.sqrt(math.fsum(error * error for error in errors) / count), count)


def _divider_parameter(rmin: float, point: Point) -> float:
    parameters = eq6_ellipse_parameters(rmin)
    theta_axis = math.acos(-parameters.c / parameters.a)
    theta = math.atan2(point[1] / parameters.b, (point[0] - parameters.c) / parameters.a)
    return min(1.0, max(0.0, 1.0 - theta / theta_axis))


def find_unique_eq6_intersection(
    curve: PiecewiseQ2Curve,
    rmin: float,
    *,
    subdivisions_per_segment: int = 32,
) -> Eq6Intersection:
    """Find the unique transverse Eq. 6 crossing by safeguarded Newton/bisection."""
    rmin = float(rmin)
    if not math.isfinite(rmin) or rmin <= 0.0:
        raise ValueError("rmin must be positive and finite")
    if subdivisions_per_segment < 8:
        raise ValueError("subdivisions_per_segment must be at least eight")
    coefficients = eq6_conic_coeffs(rmin)

    def value(parameter: float) -> float:
        return eq6_conic_value(*curve(parameter), coefficients)

    intervals = []
    samples = len(curve.segments) * subdivisions_per_segment
    previous_t, previous_f = 0.0, value(0.0)
    for index in range(1, samples + 1):
        current_t = index / samples
        current_f = value(current_t)
        if previous_f == 0.0:
            intervals.append((previous_t, previous_t))
        elif current_f == 0.0 or (previous_f < 0.0 < current_f) or (previous_f > 0.0 > current_f):
            intervals.append((previous_t, current_t))
        previous_t, previous_f = current_t, current_f
    if previous_f == 0.0:
        intervals.append((1.0, 1.0))
    # Merge duplicate brackets caused by a root exactly on a sample boundary.
    merged = []
    for bracket in intervals:
        if not merged or bracket[0] > merged[-1][1] + 8.0 * _EPS:
            merged.append(list(bracket))
        else:
            merged[-1][1] = max(merged[-1][1], bracket[1])
    if len(merged) != 1:
        raise ValueError(f"expected one Eq. 6/interface intersection, found {len(merged)}")

    lo, hi = merged[0]
    if lo == hi:
        root = lo
    else:
        flo, fhi = value(lo), value(hi)
        root = 0.5 * (lo + hi)
        for _ in range(96):
            point = curve(root)
            tangent = curve.derivative(root)
            a, b, c, d, e = coefficients
            gradient = (2.0 * a * point[0] + b * point[1] + d,
                        b * point[0] + 2.0 * c * point[1] + e)
            derivative = gradient[0] * tangent[0] + gradient[1] * tangent[1]
            trial = root - value(root) / derivative if derivative != 0.0 else math.inf
            if not lo < trial < hi or not math.isfinite(trial):
                trial = 0.5 * (lo + hi)
            ftrial = value(trial)
            if ftrial == 0.0:
                root = trial
                break
            if (flo < 0.0 < ftrial) or (flo > 0.0 > ftrial):
                hi, fhi = trial, ftrial
            else:
                lo, flo = trial, ftrial
            root = 0.5 * (lo + hi)
            if hi - lo <= 8.0 * max(math.ulp(lo), math.ulp(hi), _EPS):
                break

    point = curve(root)
    tangent = curve.derivative(root)
    a, b, c, d, e = coefficients
    gradient = (2.0 * a * point[0] + b * point[1] + d,
                b * point[0] + 2.0 * c * point[1] + e)
    transversality = abs(gradient[0] * tangent[0] + gradient[1] * tangent[1])
    gradient_scale = math.hypot(*gradient) * math.hypot(*tangent)
    if transversality <= 256.0 * _EPS * gradient_scale:
        raise ValueError("Eq. 6/interface intersection is tangent or unresolved")
    divider_t = _divider_parameter(rmin, point)
    divider_point = eq6_ellipse_point(rmin, divider_t)
    mismatch = math.dist(point, divider_point)
    geometry_scale = max(rmin, abs(point[0]), abs(point[1]))
    if mismatch > 256.0 * _EPS * geometry_scale:
        raise ValueError(
            f"Eq. 6 intersection is not binary64-resolved (mismatch {mismatch:.3e})"
        )
    return Eq6Intersection(root, divider_t, point, abs(value(root)), transversality)


def build_evolved_four_patch_graph(
    interface: PiecewiseQ2Curve,
    *,
    cap_resolution: int = 1,
    bridge_normal_resolution: int = 2,
    grading_u=None,
    grading_v=None,
):
    """Build the curve-driven graph from a captured neck-to-pole Q2 chain."""
    neck, pole = interface.start, interface.end
    if abs(neck[1]) > 16.0 * math.ulp(max(1.0, abs(neck[1]))):
        raise ValueError("evolved interface must start on the symmetry plane")
    if abs(pole[0]) > 16.0 * math.ulp(max(1.0, abs(pole[0]))):
        raise ValueError("evolved interface must end on the axis")
    intersection = find_unique_eq6_intersection(interface, neck[0])
    from .structured_blocks import (
        MIN_TOPOLOGY_SCALED_JACOBIAN,
        build_curve_driven_four_patch_graph,
    )
    graph = build_curve_driven_four_patch_graph(
        neck[0],
        neck=neck,
        pole=pole,
        interface=interface,
        interface_hit_parameter=intersection.interface_parameter,
        divider_hit_parameter=intersection.divider_parameter,
        cap_resolution=cap_resolution,
        bridge_normal_resolution=bridge_normal_resolution,
        grading_u=grading_u,
        grading_v=grading_v,
    )
    # Recheck the rebuilt Q2 maps on a denser reference lattice than the
    # construction-time diagnostic; this is still tiny (at most a few dozen
    # elements here) and prevents a centre-only or corner-only fold gate.
    minimum = graph.minimum_scaled_jacobian(tuple(index / 16.0 for index in range(17)))
    if not math.isfinite(minimum) or minimum <= MIN_TOPOLOGY_SCALED_JACOBIAN:
        raise ValueError(
            "evolved four-patch geometry is folded or violates the local angle "
            f"gate ({minimum!r} <= {MIN_TOPOLOGY_SCALED_JACOBIAN!r})"
        )
    return graph, intersection


__all__ = [
    "Eq6Intersection",
    "OrientedQ2Segment",
    "PiecewiseQ2Curve",
    "ReconstructionError",
    "build_evolved_four_patch_graph",
    "axisymmetric_twice_mean_curvature_at_start",
    "find_unique_eq6_intersection",
    "reconstruction_error",
]
