"""Software-contract tests for deterministic full-Q2 source point location."""

from __future__ import annotations

from collections import Counter
import copy
import math
import unittest

from problems.newtonian.q2_locator import (
    AmbiguousSourceLocation,
    MissingSourceLocation,
    Q2PointLocator,
    evaluate_q1,
    evaluate_q2,
)
from problems.newtonian.structured_blocks import (
    MeshNode,
    Q2Element,
    Q2_REFERENCE_COORDINATES,
    StructuredQ2Graph,
    build_initial_four_patch_graph,
)


GAUSS = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))
INTERIOR = ((0.137, 0.619), (0.331, 0.277), (0.863, 0.421))
SIDE_INDICES = ((0, 4, 1), (1, 5, 2), (3, 6, 2), (0, 7, 3))


def _element(graph, element_id):
    return next(element for element in graph.elements if element.element_id == element_id)


def _independent_q2_map(graph, element, u, v):
    """Direct polynomial evaluator independent of q2_locator's helpers."""
    one_d_u = (
        (1.0 - u) * (1.0 - 2.0 * u),
        4.0 * u * (1.0 - u),
        u * (2.0 * u - 1.0),
    )
    one_d_v = (
        (1.0 - v) * (1.0 - 2.0 * v),
        4.0 * v * (1.0 - v),
        v * (2.0 * v - 1.0),
    )
    tensor = ((0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1), (1, 1))
    coordinates = [graph.nodes[key].coordinates for key in element.node_keys]
    return tuple(
        math.fsum(one_d_u[i] * one_d_v[j] * point[component]
                  for point, (i, j) in zip(coordinates, tensor))
        for component in range(2)
    )


def _single_element_graph(origin=(0.0, 0.0), columns=((2.0, 0.25), (-0.2, 1.5))):
    keys = tuple((99, index) for index in range(9))
    points = []
    for u, v in Q2_REFERENCE_COORDINATES:
        points.append((
            origin[0] + columns[0][0] * u + columns[1][0] * v,
            origin[1] + columns[0][1] * u + columns[1][1] * v,
        ))
    nodes = {key: MeshNode(key, point) for key, point in zip(keys, points)}
    element = Q2Element((9, 0, 0), 9, (0, 0), keys)
    return StructuredQ2Graph(
        R0=1.0, Z0=1.0, nodes=nodes, elements=(element,),
        boundary_facets={}, patches={}, shared_node_keys=frozenset(),
        vertices={}, edges={}, hit_parameter=0.0, divider_hit=(0.0, 0.0),
        cap_resolution=1, bridge_normal_resolution=1,
    )


class Q2PointLocatorTests(unittest.TestCase):
    def test_round_trip_nodes_gauss_and_interior_points_at_anthony_scales(self):
        receipts = []
        for r0 in (1e-3, 1e-6):
            for level in (1, 2):
                graph = build_initial_four_patch_graph(r0, cap_resolution=level)
                locator = Q2PointLocator(graph)
                checked = shared = 0
                max_residual = max_uncertainty = 0.0
                references = tuple(Q2_REFERENCE_COORDINATES) + tuple(
                    (u, v) for u in GAUSS for v in GAUSS
                ) + INTERIOR
                for source in graph.elements:
                    for u, v in references:
                        target = _independent_q2_map(graph, source, u, v)
                        location = locator.locate(target)
                        found = _element(graph, location.element_id)
                        reconstructed = graph.element_map(found, location.u, location.v)
                        error = math.hypot(
                            reconstructed[0] - target[0],
                            reconstructed[1] - target[1],
                        )
                        local_scale = max(
                            math.dist(graph.nodes[found.node_keys[0]].coordinates,
                                      graph.nodes[found.node_keys[2]].coordinates),
                            math.ulp(target[0]), math.ulp(target[1]),
                        )
                        self.assertLessEqual(error, 512.0 * math.ulp(1.0) * local_scale + 16.0 * max(math.ulp(target[0]), math.ulp(target[1])))
                        self.assertGreaterEqual(location.local_coordinate_uncertainty, 0.0)
                        self.assertTrue(0.0 <= location.u <= 1.0)
                        self.assertTrue(0.0 <= location.v <= 1.0)
                        checked += 1
                        shared += location.shared_edge
                        max_residual = max(max_residual, location.physical_residual)
                        max_uncertainty = max(max_uncertainty, location.local_coordinate_uncertainty)
                receipts.append((r0, level, checked, shared, max_residual, max_uncertainty))

        self.assertEqual([receipt[2] for receipt in receipts], [112, 320, 112, 320])
        self.assertTrue(all(receipt[4] < 2e-15 for receipt in receipts))
        # Tiny bridge cells are ill-conditioned in physical coordinates, so
        # the local-coordinate receipt must honestly grow at R0=1e-6.
        self.assertGreater(receipts[2][5], receipts[0][5])

    def test_curved_non_affine_element_uses_full_q2_map(self):
        graph = _single_element_graph()
        centre_key = graph.elements[0].node_keys[8]
        graph.nodes[centre_key].coordinates = (
            graph.nodes[centre_key].coordinates[0] + 0.19,
            graph.nodes[centre_key].coordinates[1] - 0.11,
        )
        edge_key = graph.elements[0].node_keys[5]
        graph.nodes[edge_key].coordinates = (
            graph.nodes[edge_key].coordinates[0] + 0.07,
            graph.nodes[edge_key].coordinates[1] + 0.08,
        )
        locator = Q2PointLocator(graph)
        for u, v in INTERIOR + tuple((a, b) for a in GAUSS for b in GAUSS):
            target = _independent_q2_map(graph, graph.elements[0], u, v)
            location = locator.locate(target)
            self.assertAlmostEqual(location.u, u, delta=5e-14)
            self.assertAlmostEqual(location.v, v, delta=5e-14)
            self.assertFalse(location.shared_edge)

    def test_shared_divider_and_internal_edges_return_all_conforming_hits(self):
        graph = build_initial_four_patch_graph(1e-6, cap_resolution=2)
        locator = Q2PointLocator(graph)
        incidence = {}
        for element in graph.elements:
            for side in SIDE_INDICES:
                entity = frozenset(element.node_keys[index] for index in side)
                incidence.setdefault(entity, []).append((element, side))
        shared_entities = [items for items in incidence.values() if len(items) == 2]
        patch_pairs = Counter()
        q2_field = {key: 1.25 - 0.75 * node.coordinates[0] + 0.5 * node.coordinates[1]
                    for key, node in graph.nodes.items()}
        for items in shared_entities:
            first, side = items[0]
            target = graph.nodes[first.node_keys[side[1]]].coordinates
            location = locator.locate(target)
            self.assertTrue(location.shared_edge)
            self.assertGreaterEqual(len(location.candidates), 2)
            patch_pairs[tuple(sorted(element.patch_id for element, _ in items))] += 1
            expected = 1.25 - 0.75 * target[0] + 0.5 * target[1]
            self.assertAlmostEqual(evaluate_q2(graph, location, q2_field), expected, delta=2e-15)
        self.assertTrue(any(first != second for first, second in patch_pairs))
        self.assertTrue(any(first == second for first, second in patch_pairs))

    def test_search_is_global_across_patch_blocks(self):
        graph = build_initial_four_patch_graph(1e-3, cap_resolution=2)
        locator = Q2PointLocator(graph)
        for patch_id in sorted(graph.patches):
            source = next(element for element in graph.elements if element.patch_id == patch_id)
            target = _independent_q2_map(graph, source, 0.37, 0.63)
            location = locator.locate(target)
            self.assertEqual(_element(graph, location.element_id).patch_id, patch_id)

    def test_exterior_and_non_finite_points_fail_without_fallback(self):
        for r0 in (1e-3, 1e-6):
            graph = build_initial_four_patch_graph(r0, cap_resolution=2)
            locator = Q2PointLocator(graph)
            xs = [node.coordinates[0] for node in graph.nodes.values()]
            ys = [node.coordinates[1] for node in graph.nodes.values()]
            for point in ((min(xs) - 1e-8, min(ys)),
                          (max(xs) + 1e-8, max(ys)),
                          (0.5 * (min(xs) + max(xs)), max(ys) + 1e-8)):
                with self.assertRaises(MissingSourceLocation):
                    locator.locate(point)
            for point in ((math.nan, 0.0), (0.0, math.inf)):
                with self.assertRaises(ValueError):
                    locator.locate(point)

    def test_affine_q2_and_q1_fields_reproduce_to_roundoff(self):
        graph = _single_element_graph(origin=(0.125, -0.375))
        locator = Q2PointLocator(graph)
        physical_field = {
            key: 2.0 + 1.75 * node.coordinates[0] - 0.625 * node.coordinates[1]
            for key, node in graph.nodes.items()
        }
        reference_field = {
            key: -0.4 + 2.25 * uv[0] - 1.125 * uv[1]
            for key, uv in zip(graph.elements[0].node_keys, Q2_REFERENCE_COORDINATES)
        }
        for u, v in INTERIOR + tuple((a, b) for a in GAUSS for b in GAUSS):
            target = _independent_q2_map(graph, graph.elements[0], u, v)
            location = locator.locate(target)
            expected_q2 = 2.0 + 1.75 * target[0] - 0.625 * target[1]
            expected_q1 = -0.4 + 2.25 * u - 1.125 * v
            self.assertAlmostEqual(evaluate_q2(graph, location, physical_field), expected_q2, delta=3e-15)
            self.assertAlmostEqual(evaluate_q1(graph, location, reference_field), expected_q1, delta=3e-15)

    def test_translated_tiny_element_resolves_1e_18_perturbation_and_reports_uncertainty(self):
        graph = _single_element_graph(
            origin=(1e-6, -2e-6),
            columns=((1.1e-9, 0.2e-9), (-0.15e-9, 0.8e-9)),
        )
        locator = Q2PointLocator(graph)
        u, v = 0.413, 0.587
        base = _independent_q2_map(graph, graph.elements[0], u, v)
        target = (base[0] + 1e-18, base[1] - 1e-18)
        location = locator.locate(target)
        reconstructed = graph.element_map(graph.elements[0], location.u, location.v)
        residual = math.dist(target, reconstructed)
        self.assertLess(residual, 2e-21)
        self.assertGreater(location.local_coordinate_uncertainty, 0.0)
        self.assertLess(location.local_coordinate_uncertainty, 1e-11)
        self.assertGreater(abs(location.u - u) + abs(location.v - v), 1e-10)

    def test_shared_field_discontinuity_is_explicitly_ambiguous(self):
        # This negative control constructs two geometrically coincident
        # elements with distinct field keys.  Geometry may be shared, but a
        # discontinuous field must never be silently averaged.
        graph = _single_element_graph()
        other = copy.deepcopy(graph.elements[0])
        other_keys = tuple((100, index) for index in range(9))
        for key, original in zip(other_keys, graph.elements[0].node_keys):
            graph.nodes[key] = MeshNode(key, graph.nodes[original].coordinates)
        other = Q2Element((10, 0, 0), 10, (0, 0), other_keys)
        graph.elements = (graph.elements[0], other)
        # Coincident interiors are correctly rejected already at location.
        with self.assertRaises(AmbiguousSourceLocation):
            Q2PointLocator(graph).locate((0.7, 0.6))


if __name__ == "__main__":
    unittest.main()
