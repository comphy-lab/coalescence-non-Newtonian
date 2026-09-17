"""Pure software tests for exact evolved piecewise-Q2 geometry."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import unittest

from problems.newtonian.evolved_geometry import (
    OrientedQ2Segment,
    PiecewiseQ2Curve,
    build_evolved_four_patch_graph,
    find_unique_eq6_intersection,
    reconstruction_error,
)
from problems.newtonian.structured_blocks import (
    MIN_TOPOLOGY_SCALED_JACOBIAN,
    build_initial_four_patch_graph,
)


def _captured_interface(graph):
    """Extract the graph's oriented Q2 interface without a polyline fit."""
    remaining = []
    for facet in graph.boundary_facets["interface"]:
        remaining.append((
            facet.node_keys[0],
            facet.node_keys[-1],
            OrientedQ2Segment(*(graph.nodes[key].coordinates for key in facet.node_keys)),
            facet.element_id,
            facet.side,
        ))
    current = graph.vertices["neck"]
    ordered = []
    element_sides = []
    while remaining:
        matches = [index for index, item in enumerate(remaining) if item[0] == current]
        if len(matches) != 1:
            raise AssertionError("interface facets do not form one oriented chain")
        start, current, segment, element_id, side = remaining.pop(matches[0])
        ordered.append(segment)
        element = next(item for item in graph.elements if item.element_id == element_id)
        if side == "right":
            local_start = graph.element_map(element, 1.0, 0.0)
        elif side == "top":
            local_start = graph.element_map(element, 0.0, 1.0)
        elif side == "left":
            local_start = graph.element_map(element, 0.0, 0.0)
        else:
            local_start = graph.element_map(element, 0.0, 0.0)
        element_sides.append((element_id, side, math.dist(local_start, segment.start) > 1e-15))
    if current != graph.vertices["pole"]:
        raise AssertionError("interface chain does not terminate at the pole")
    return PiecewiseQ2Curve(tuple(ordered)), tuple(element_sides)


def _independent_boundary_evaluator(graph, element_sides):
    count = len(element_sides)

    def evaluate(parameter):
        if parameter == 1.0:
            index, local = count - 1, 1.0
        else:
            scaled = parameter * count
            index, local = int(scaled), scaled - int(scaled)
        element_id, side, reversed_parameter = element_sides[index]
        if reversed_parameter:
            local = 1.0 - local
        element = next(item for item in graph.elements if item.element_id == element_id)
        if side == "right":
            return graph.element_map(element, 1.0, local)
        if side == "top":
            return graph.element_map(element, local, 1.0)
        if side == "left":
            return graph.element_map(element, 0.0, local)
        if side == "bottom":
            return graph.element_map(element, local, 0.0)
        raise AssertionError(side)

    return evaluate


class Q2SegmentTests(unittest.TestCase):
    def test_evaluation_derivative_and_gauss_arclength(self):
        segment = OrientedQ2Segment((0.0, 0.0), (0.5, 0.25), (1.0, 1.0))
        for parameter in (0.0, 0.125, 0.5, 0.875, 1.0):
            self.assertEqual(segment(parameter)[0], parameter)
            self.assertAlmostEqual(segment(parameter)[1], parameter * parameter, places=15)
            self.assertAlmostEqual(segment.derivative(parameter)[0], 1.0, places=15)
            self.assertAlmostEqual(segment.derivative(parameter)[1], 2.0 * parameter, places=15)
        exact = 0.5 * math.sqrt(5.0) + 0.25 * math.asinh(2.0)
        self.assertAlmostEqual(segment.arclength(), exact, delta=6e-10)
        with self.assertRaises(FrozenInstanceError):
            segment.start = (1.0, 1.0)

    def test_split_subcurve_and_orientation_are_exact_q2_restrictions(self):
        segment = OrientedQ2Segment((0.0, 0.0), (0.4, 0.7), (1.0, 0.2))
        left, right = segment.split(0.37)
        for index in range(33):
            local = index / 32.0
            self.assertLess(math.dist(left(local), segment(0.37 * local)), 4e-16)
            self.assertLess(math.dist(right(local), segment(0.37 + 0.63 * local)), 4e-16)
        curve = PiecewiseQ2Curve((left, right))
        subcurve = curve.subcurve(0.2, 0.8)
        for index in range(33):
            local = index / 32.0
            self.assertLess(
                math.dist(subcurve(local), curve(0.2 + 0.6 * local)), 8e-16
            )
        reverse = curve.reversed()
        for index in range(33):
            local = index / 32.0
            self.assertLess(math.dist(reverse(local), curve(1.0 - local)), 8e-16)

    def test_folded_and_disconnected_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "folded|cusp"):
            OrientedQ2Segment((0.0, 0.0), (0.5, 0.0), (0.0, 0.0))
        first = OrientedQ2Segment((0.0, 0.0), (0.25, 0.1), (0.5, 0.2))
        second = OrientedQ2Segment((0.6, 0.2), (0.8, 0.4), (1.0, 0.5))
        with self.assertRaisesRegex(ValueError, "disconnected"):
            PiecewiseQ2Curve((first, second))


class EvolvedGraphTests(unittest.TestCase):
    scales = (1e-3, 1e-6)
    levels = (1, 2)

    def test_captured_q2_reconstruction_intersection_and_graph_gate(self):
        receipts = []
        for r0 in self.scales:
            for level in self.levels:
                source = build_initial_four_patch_graph(r0, cap_resolution=level)
                interface, element_sides = _captured_interface(source)
                error = reconstruction_error(
                    interface,
                    _independent_boundary_evaluator(source, element_sides),
                    samples_per_segment=24,
                )
                intersection = find_unique_eq6_intersection(interface, r0)
                rebuilt, rebuilt_intersection = build_evolved_four_patch_graph(
                    interface, cap_resolution=level
                )
                self.assertEqual(intersection, rebuilt_intersection)
                self.assertLessEqual(error.maximum, 8e-16)
                self.assertLess(math.dist(intersection.point, source.divider_hit), 1e-15)
                self.assertLessEqual(intersection.conic_residual, 2e-15)
                self.assertGreater(intersection.transversality, 1.0)
                self.assertEqual(len(rebuilt.elements), len(source.elements))
                self.assertEqual(len(rebuilt.nodes), len(source.nodes))
                self.assertEqual(
                    {label: len(facets) for label, facets in rebuilt.boundary_facets.items()},
                    {label: len(facets) for label, facets in source.boundary_facets.items()},
                )
                minimum = rebuilt.minimum_scaled_jacobian(
                    tuple(index / 40.0 for index in range(41))
                )
                self.assertGreater(minimum, MIN_TOPOLOGY_SCALED_JACOBIAN)
                receipts.append((r0, level, len(interface.segments), error.maximum,
                                 intersection.conic_residual, minimum))
        self.assertEqual([item[2] for item in receipts], [4, 6, 4, 6])

    def test_multiple_or_missing_eq6_crossings_are_rejected(self):
        rmin = 0.1
        # Two straight Q2 segments cross the upper Eq. 6 arc once each.
        ambiguous = PiecewiseQ2Curve((
            OrientedQ2Segment((0.3, 0.05), (0.15, 0.05), (0.0, 0.05)),
            OrientedQ2Segment((0.0, 0.05), (0.15, 0.05), (0.3, 0.05)),
        ))
        with self.assertRaisesRegex(ValueError, "expected one"):
            find_unique_eq6_intersection(ambiguous, rmin)
        missing = PiecewiseQ2Curve((
            OrientedQ2Segment((0.4, 0.2), (0.5, 0.25), (0.6, 0.3)),
        ))
        with self.assertRaisesRegex(ValueError, "found 0"):
            find_unique_eq6_intersection(missing, rmin)

    def test_endpoint_contract_rejects_misoriented_runtime_capture(self):
        source = build_initial_four_patch_graph(1e-3)
        interface, _ = _captured_interface(source)
        with self.assertRaisesRegex(ValueError, "symmetry plane"):
            build_evolved_four_patch_graph(interface.reversed())


if __name__ == "__main__":
    unittest.main()
