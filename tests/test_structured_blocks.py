"""Pure-Python four-patch Q2 graph checks at Anthony's tiny bridge scales."""

from __future__ import annotations

import math
from collections import Counter
import unittest

from problems.newtonian.geometry import (
    approximate_point_contact_z0,
    eq2_bridge_value,
    eq6_conic_coeffs,
    eq6_conic_value,
    sphere_value,
    sphere_bridge_junction,
)
from problems.newtonian.structured_blocks import (
    MIN_TOPOLOGY_SCALED_JACOBIAN,
    Q2_REFERENCE_COORDINATES,
    build_initial_four_patch_graph,
    power_grading,
)


class StructuredBlockTests(unittest.TestCase):
    scales = (1e-3, 1e-4, 1e-6)
    jacobian_samples = tuple(index / 100.0 for index in range(101))

    @staticmethod
    def expected_counts(cap_resolution: int, bridge_normal_resolution: int = 2) -> tuple[int, int]:
        c = cap_resolution
        b = bridge_normal_resolution
        elements = 2 * c * b + 3 * c * c
        boundary_facets = 6 * c + 2 * b
        nodes = 4 * elements + boundary_facets + 1
        return elements, nodes

    def test_deterministic_four_patch_counts_and_q2_connectivity(self) -> None:
        for r0 in self.scales:
            graph = build_initial_four_patch_graph(r0)
            self.assertEqual(len(graph.patches), 4)
            self.assertEqual(len(graph.elements), 7)
            self.assertEqual(len(graph.nodes), 39)
            self.assertEqual(len(set(graph.connectivity)), len(graph.elements))
            for key in graph.nodes:
                self.assertTrue(all(isinstance(part, int) for part in key))
            for element in graph.elements:
                self.assertEqual(len(element.node_keys), 9)
                self.assertEqual(len(set(element.node_keys)), 9)
                self.assertEqual(element.reference_coordinates, Q2_REFERENCE_COORDINATES)

    def test_resolution_levels_have_deterministic_counts_and_boundaries(self) -> None:
        for level in (1, 2, 3):
            with self.subTest(level=level):
                graph = build_initial_four_patch_graph(1e-3, cap_resolution=level)
                expected_elements, expected_nodes = self.expected_counts(level)
                self.assertEqual(len(graph.elements), expected_elements)
                self.assertEqual(len(graph.nodes), expected_nodes)
                self.assertEqual(graph.cap_resolution, level)
                self.assertEqual(graph.bridge_normal_resolution, 2)
                self.assertEqual(
                    {patch.name: (patch.nu, patch.nv) for patch in graph.patches.values()},
                    {
                        "bridge": (2 * level, 2),
                        "cap-divider": (level, level),
                        "cap-interface-lower": (level, level),
                        "cap-axis-upper": (level, level),
                    },
                )
                self.assertEqual(
                    {label: len(facets) for label, facets in graph.boundary_facets.items()},
                    {
                        "axis": 2 + 2 * level,
                        "plane": 2 * level,
                        "interface": 2 + 2 * level,
                        "divider": 4 * level,
                    },
                )

                # Every element side has incidence one (physical boundary) or
                # two (conforming interior); this covers the three unlabeled
                # cap interfaces as well as the named divider.
                side_offsets = ((0, 4, 1), (1, 5, 2), (3, 6, 2), (0, 7, 3))
                side_incidence = Counter(
                    frozenset(element.node_keys[index] for index in side)
                    for element in graph.elements
                    for side in side_offsets
                )
                physical_boundary = 6 * level + 4
                self.assertEqual(set(side_incidence.values()), {1, 2})
                self.assertEqual(
                    sum(count == 1 for count in side_incidence.values()),
                    physical_boundary,
                )
                self.assertEqual(
                    sum(count == 2 for count in side_incidence.values()),
                    2 * expected_elements - physical_boundary // 2,
                )

                # Each of the 2*c divider facets is represented once from the
                # bridge and once from a cap, using exactly the same keys.
                divider_incidence = Counter(
                    facet.node_keys for facet in graph.boundary_facets["divider"]
                )
                self.assertEqual(len(divider_incidence), 2 * level)
                self.assertEqual(set(divider_incidence.values()), {2})

                # Distinct keys remain distinct nodes; sharing is topological,
                # never a coordinate-based merge.
                coordinates = [node.coordinates for node in graph.nodes.values()]
                self.assertEqual(len(set(coordinates)), len(coordinates))

    def test_bridge_normal_resolution_is_independent(self) -> None:
        cap_resolution = 2
        for bridge_normal_resolution in (1, 3):
            graph = build_initial_four_patch_graph(
                1e-3,
                cap_resolution=cap_resolution,
                bridge_normal_resolution=bridge_normal_resolution,
            )
            self.assertEqual(
                (len(graph.elements), len(graph.nodes)),
                self.expected_counts(cap_resolution, bridge_normal_resolution),
            )
            self.assertEqual(
                (graph.patches[0].nu, graph.patches[0].nv),
                (2 * cap_resolution, bridge_normal_resolution),
            )

    def test_invalid_resolution_controls_fail(self) -> None:
        for name in ("cap_resolution", "bridge_normal_resolution"):
            for invalid in (True, False, 1.0, "2", None):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(TypeError):
                    build_initial_four_patch_graph(1e-3, **{name: invalid})
            for invalid in (0, -1):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    build_initial_four_patch_graph(1e-3, **{name: invalid})

    def test_shared_topology_and_coordinates_are_explicit(self) -> None:
        graph = build_initial_four_patch_graph(1e-6)
        self.assertGreaterEqual(len(graph.shared_node_keys), 11)
        for key in graph.shared_node_keys:
            node = graph.nodes[key]
            self.assertGreaterEqual(len(node.patches), 2)
            self.assertTrue(all(math.isfinite(value) for value in node.coordinates))
            for patch_id, logical in node.logical_coordinates.items():
                point = graph.patches[patch_id].map(*logical)
                self.assertEqual(point, node.coordinates)

        # Shared boundary segments carry the same ordered topological keys on
        # both sides of the divider and all internal cap edges.
        divider = [facet.node_keys for facet in graph.boundary_facets["divider"]]
        self.assertEqual(divider[0], divider[2])
        self.assertEqual(divider[1], divider[3])

    def test_positive_locally_scaled_q2_jacobians(self) -> None:
        for r0 in self.scales:
            graph = build_initial_four_patch_graph(r0)
            values = [
                graph.scaled_jacobian(element, u, v)
                for element in graph.elements
                for u in self.jacobian_samples
                for v in self.jacobian_samples
            ]
            self.assertTrue(all(math.isfinite(value) and value > 0.0 for value in values))
            self.assertGreater(min(values), MIN_TOPOLOGY_SCALED_JACOBIAN)
            self.assertGreater(
                graph.minimum_scaled_jacobian(self.jacobian_samples),
                MIN_TOPOLOGY_SCALED_JACOBIAN,
            )

    def test_refined_levels_retain_tiny_bridge_angle_gate(self) -> None:
        for level in (1, 2, 3):
            graph = build_initial_four_patch_graph(1e-6, cap_resolution=level)
            self.assertGreater(
                graph.minimum_scaled_jacobian(self.jacobian_samples),
                MIN_TOPOLOGY_SCALED_JACOBIAN,
            )

    def test_analytical_boundary_residuals(self) -> None:
        for r0 in self.scales:
            z0 = approximate_point_contact_z0(r0)
            graph = build_initial_four_patch_graph(r0)
            coeffs = eq6_conic_coeffs(r0)
            for key, node in graph.nodes.items():
                if key[0] == 1 and key[1] == 0:
                    self.assertAlmostEqual(eq6_conic_value(*node.coordinates, coeffs), 0.0, delta=2e-12)
                if key[0] == 1 and key[1] == 4:
                    eq2_relative = abs(
                        eq2_bridge_value(*node.coordinates, r0, z0)
                    ) / (z0 * z0)
                    sphere_roundoff = abs(
                        sphere_value(*node.coordinates, r0, z0)
                    ) / math.ulp(1.0)
                    self.assertTrue(
                        eq2_relative < 1e-8 or sphere_roundoff < 64.0
                    )
                if key[0] == 1 and key[1] == 1:
                    self.assertLess(
                        abs(sphere_value(*node.coordinates, r0, z0)) / math.ulp(1.0),
                        64.0,
                    )

            rj, zj = sphere_bridge_junction(r0, z0)
            junction_key = (1, 4, 1, 2)
            self.assertIn(junction_key, graph.nodes)
            self.assertAlmostEqual(graph.nodes[junction_key].coordinates[0], rj, delta=2e-14)
            self.assertAlmostEqual(graph.nodes[junction_key].coordinates[1], zj, delta=2e-14)

    def test_small_physical_spacing_is_not_coordinate_deduplicated(self) -> None:
        graph = build_initial_four_patch_graph(1e-6)
        bridge_keys = list(graph.patches[0].node_keys)
        distances = [
            math.hypot(
                graph.nodes[first].coordinates[0] - graph.nodes[second].coordinates[0],
                graph.nodes[first].coordinates[1] - graph.nodes[second].coordinates[1],
            )
            for i, first in enumerate(bridge_keys)
            for second in bridge_keys[i + 1 :]
        ]
        self.assertGreater(min(distances), 0.0)
        self.assertLess(min(distances), 1e-8)

    def test_bridge_residual_gate_rejects_a_grossly_wrong_tiny_height(self) -> None:
        r0 = 1e-6
        z0 = approximate_point_contact_z0(r0)
        wrong_z = 1e-8
        relative = abs(eq2_bridge_value(r0, wrong_z, r0, z0)) / (z0 * z0)
        self.assertGreater(relative, 1e6)

    def test_grading_utility_is_monotone(self) -> None:
        grading = power_grading(1.5)
        values = [grading(i / 20.0) for i in range(21)]
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
        for value in values:
            self.assertAlmostEqual(grading(grading.inverse(value)), value, places=14)


if __name__ == "__main__":
    unittest.main()
