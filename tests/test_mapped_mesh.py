"""Integration checks for the explicit Anthony four-patch pyoomph mesh."""

from __future__ import annotations

from contextlib import contextmanager
import math
import tempfile
import unittest

from pyoomph import DirichletBC, ElementSpace, Problem

from problems.newtonian.elliptic_grid import (
    ControlledEllipticMesh,
    LogicalCoordinateTable,
    mesh_quality,
)
from problems.newtonian.mapped_mesh import AnthonyMappedQuadMesh
from problems.newtonian.structured_blocks import build_initial_four_patch_graph


@contextmanager
def _initialised_problem(
    r0: float,
    *,
    cap_resolution: int = 1,
    bridge_normal_resolution: int = 2,
):
    """Yield a mesh-only Problem in a run-owned temporary output directory."""
    graph = build_initial_four_patch_graph(
        r0,
        cap_resolution=cap_resolution,
        bridge_normal_resolution=bridge_normal_resolution,
    )
    # The elliptic mesh equation uses the physical mesh coordinates as its
    # reference chart here.  A strict table is intentional: it catches any
    # rounding or coordinate-based node lookup during pyoomph generation.
    references = LogicalCoordinateTable(
        {node.coordinates: node.coordinates for node in graph.nodes.values()}
    )

    class MappedMeshProblem(Problem):
        def __init__(self) -> None:
            super().__init__()
            self.initial_adaption_steps = 0

        def define_problem(self) -> None:
            self.set_linear_solver("superlu")
            self.add_mesh(
                AnthonyMappedQuadMesh(
                    r0,
                    cap_resolution=cap_resolution,
                    bridge_normal_resolution=bridge_normal_resolution,
                )
            )
            equations = ControlledEllipticMesh(logical_coordinates=references)
            equations += ElementSpace("C2")
            # The outer boundary is fixed for this mesh-only initialisation;
            # the divider remains unpinned and is checked as an interior facet.
            equations += DirichletBC(mesh_x=True, mesh_y=True) @ [
                "axis",
                "plane",
                "interface",
            ]
            self.add_equations(equations @ "drop")

    problem = MappedMeshProblem()
    with tempfile.TemporaryDirectory(prefix="mapped-mesh-test-") as output_dir:
        problem.set_output_directory(output_dir)
        with problem:
            problem.initialise()
            yield problem, graph


class AnthonyMappedQuadMeshTests(unittest.TestCase):
    scales = (1e-3, 1e-4, 1e-6)

    def test_invalid_resolution_controls_fail_at_construction(self) -> None:
        for name in ("cap_resolution", "bridge_normal_resolution"):
            for invalid in (True, False, 1.0, "2", None):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(TypeError):
                    AnthonyMappedQuadMesh(1e-3, **{name: invalid})
            for invalid in (0, -1):
                with self.subTest(name=name, invalid=invalid), self.assertRaises(ValueError):
                    AnthonyMappedQuadMesh(1e-3, **{name: invalid})

    def test_template_and_runtime_preserve_all_nodes_and_q2_topology(self) -> None:
        """The pyoomph mesh has the graph's 39 nodes and seven Q2 elements."""
        for r0 in self.scales:
            with self.subTest(r0=r0), _initialised_problem(r0) as (problem, graph):
                mesh = problem.get_mesh("drop")
                runtime_nodes = tuple(mesh.nodes())
                runtime_elements = tuple(mesh.elements())
                template = mesh._templatemesh._get_template()
                quality = mesh_quality(mesh, require_positive=True)

                self.assertEqual(len(runtime_nodes), len(graph.nodes))
                self.assertEqual(len(runtime_nodes), 39)
                self.assertEqual(len(runtime_elements), len(graph.elements))
                self.assertEqual(len(runtime_elements), 7)
                self.assertEqual(quality.element_count, 7)
                self.assertTrue(math.isfinite(quality.min_jacobian))
                self.assertGreater(quality.min_jacobian, 0.0)
                self.assertEqual(set(template.node_key_to_template_index), set(graph.nodes))
                self.assertEqual(
                    set(template.node_key_to_template_index.values()),
                    set(range(len(graph.nodes))),
                )

                # add_node indices are retained by the generated mesh.  This
                # direct index check is also the no-runtime-node-merging gate.
                for key, index in template.node_key_to_template_index.items():
                    node = runtime_nodes[index]
                    expected = graph.nodes[key].coordinates
                    self.assertEqual((node.x(0), node.x(1)), expected)
                    self.assertEqual((node.x_lagr(0), node.x_lagr(1)), expected)

                runtime_index = {id(node): index for index, node in enumerate(runtime_nodes)}
                runtime_connectivity = tuple(
                    tuple(runtime_index[id(element.node_pt(index))] for index in range(element.nnode()))
                    for element in runtime_elements
                )
                expected_connectivity = tuple(
                    template.template_element_indices[element.element_id]
                    for element in graph.elements
                )
                self.assertEqual(runtime_connectivity, expected_connectivity)

    def test_boundary_labels_and_interior_divider_are_registered(self) -> None:
        """All four labels survive generation, with divider facets internal."""
        with _initialised_problem(1e-6) as (problem, graph):
            mesh = problem.get_mesh("drop")
            template = mesh._templatemesh._get_template()

            self.assertEqual(
                set(mesh.get_boundary_names()), {"axis", "plane", "interface", "divider"}
            )
            self.assertEqual(template._interior_boundaries, {"divider"})

            expected_elements = {
                label: len(facets) for label, facets in graph.boundary_facets.items()
            }
            expected_template_facets = {
                label: len(facets)
                for label, facets in template.template_boundary_facets.items()
            }
            # The divider is entered once per unique topological facet, while
            # the runtime interior boundary exposes both adjacent element sides.
            self.assertEqual(expected_template_facets, {
                "axis": 4,
                "plane": 2,
                "interface": 4,
                "divider": 2,
            })
            self.assertEqual(expected_elements, {
                "axis": 4,
                "plane": 2,
                "interface": 4,
                "divider": 4,
            })
            for label, expected in expected_elements.items():
                self.assertEqual(mesh.nboundary_element(mesh.get_boundary_index(label)), expected)
                self.assertGreater(len(tuple(mesh.boundary_nodes(label))), 0)
            for facets in template.template_boundary_facets.values():
                self.assertTrue(all(len(facet) == 3 for facet in facets))

            n_facets, n_boundary, n_interior, max_incidence = mesh.facet_adjacency_summary()
            self.assertEqual((n_facets, n_boundary, n_interior, max_incidence), (19, 10, 9, 2))

    def test_tiny_mesh_quality_and_dofs_remain_bounded(self) -> None:
        """The R0=1e-6 path is finite, positive and does not create a huge system."""
        with _initialised_problem(1e-6) as (problem, _):
            quality = mesh_quality(problem.get_mesh("drop"), require_positive=True)
            dof_count = problem.ndof()

        self.assertEqual(quality.element_count, 7)
        self.assertTrue(math.isfinite(quality.min_jacobian))
        self.assertGreater(quality.min_jacobian, 0.0)
        self.assertTrue(math.isfinite(quality.max_condition_number))
        self.assertGreater(quality.max_condition_number, 0.0)
        self.assertTrue(math.isfinite(quality.max_column_aspect))
        self.assertGreater(quality.max_column_aspect, 0.0)
        self.assertGreater(dof_count, 0)
        self.assertLessEqual(dof_count, 2 * 39)

    def test_level_two_runtime_preserves_nodes_connectivity_and_bounded_dofs(self) -> None:
        """A refined template initialises in pyoomph without a flow solve."""
        level = 2
        with _initialised_problem(1e-6, cap_resolution=level) as (problem, graph):
            mesh = problem.get_mesh("drop")
            runtime_nodes = tuple(mesh.nodes())
            runtime_elements = tuple(mesh.elements())
            template = mesh._templatemesh._get_template()

            self.assertEqual((len(graph.elements), len(graph.nodes)), (20, 97))
            self.assertEqual(len(runtime_nodes), len(graph.nodes))
            self.assertEqual(len(runtime_elements), len(graph.elements))
            self.assertEqual(
                set(template.node_key_to_template_index.values()),
                set(range(len(graph.nodes))),
            )
            runtime_index = {id(node): index for index, node in enumerate(runtime_nodes)}
            runtime_connectivity = tuple(
                tuple(runtime_index[id(element.node_pt(index))] for index in range(element.nnode()))
                for element in runtime_elements
            )
            self.assertEqual(
                runtime_connectivity,
                tuple(
                    template.template_element_indices[element.element_id]
                    for element in graph.elements
                ),
            )
            self.assertEqual(
                {label: len(facets) for label, facets in template.template_boundary_facets.items()},
                {"axis": 6, "plane": 4, "interface": 6, "divider": 4},
            )
            self.assertGreater(problem.ndof(), 0)
            self.assertLessEqual(problem.ndof(), 2 * len(graph.nodes))


if __name__ == "__main__":
    unittest.main()
