"""Software-contract tests for the lagged Anthony Eq. 6 divider."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import tempfile
import unittest

import numpy
from pyoomph import DirichletBC, ElementSpace, Problem
from pyoomph.expressions import cartesian

from problems.newtonian.divider_constraint import (
    anthony_divider_constraint,
    axis_divider_corner_condition,
    eq6_divider_constraint,
    eq6_symbolic_parameters,
)
from problems.newtonian.elliptic_grid import (
    ControlledEllipticMesh,
    LogicalCoordinateTable,
)
from problems.newtonian.geometry import eq6_ellipse_parameters
from problems.newtonian.mapped_mesh import AnthonyMappedQuadMesh
from problems.newtonian.structured_blocks import build_initial_four_patch_graph


@contextmanager
def _divider_problem(r0: float):
    graph = build_initial_four_patch_graph(r0)
    references = LogicalCoordinateTable(
        {node.coordinates: node.coordinates for node in graph.nodes.values()}
    )

    class DividerProblem(Problem):
        def __init__(self) -> None:
            super().__init__()
            self.initial_adaption_steps = 0
            self.R_lagged = self.define_global_parameter(R_lagged=r0)

        def define_problem(self) -> None:
            self.set_linear_solver("superlu")
            self.add_mesh(AnthonyMappedQuadMesh(r0))
            equations = ControlledEllipticMesh(logical_coordinates=references)
            equations += ElementSpace("C2")
            equations += DirichletBC(mesh_x=0) @ "axis"
            equations += DirichletBC(mesh_y=0) @ "plane"
            equations += DirichletBC(mesh_x=True, mesh_y=True) @ "interface"
            equations += anthony_divider_constraint(self.R_lagged) @ "divider"
            equations += axis_divider_corner_condition(self.R_lagged) @ "divider/axis"
            self.add_equations(equations @ "drop")

    problem = DividerProblem()
    diagnostics = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="divider-constraint-test-") as output_dir:
        problem.set_output_directory(output_dir)
        with problem, redirect_stdout(diagnostics), redirect_stderr(diagnostics):
            problem.initialise()
            yield problem, graph, diagnostics


def _divider_multiplier_rows(problem: Problem) -> tuple[int, ...]:
    mesh = problem.get_mesh("drop")
    interface_id = mesh.has_interface_dof_id("_lagr_enf_bc_mesh_x")
    if interface_id < 0:
        raise AssertionError("divider multiplier was not registered on the bulk mesh")
    rows: set[int] = set()
    for node in mesh.boundary_nodes("divider"):
        value_index = node.additional_value_index(interface_id)
        equation = node.eqn_number(value_index)
        if equation >= 0:
            rows.add(equation)
    return tuple(sorted(rows))


class AnthonyDividerExpressionTests(unittest.TestCase):
    def test_equation_factory_uses_parent_c2_cartesian_contract(self) -> None:
        equation = anthony_divider_constraint(1e-3)
        self.assertEqual(equation.domain, "..")
        self.assertEqual(equation.space, "C2")
        self.assertIs(equation.coordsys, cartesian)
        self.assertEqual(set(equation.constraints), {"mesh_x"})

    def test_symbolic_parameters_match_geometry_reconstruction(self) -> None:
        for r_lagged in (1e-3, 1e-6):
            with self.subTest(r_lagged=r_lagged):
                symbolic = eq6_symbolic_parameters(r_lagged)
                reference = eq6_ellipse_parameters(r_lagged)
                self.assertAlmostEqual(
                    float(symbolic.H), reference.H, delta=1e-15 * r_lagged
                )
                self.assertAlmostEqual(float(symbolic.m), reference.m, delta=1e-15)
                self.assertAlmostEqual(
                    float(symbolic.c), reference.c, delta=1e-15 * r_lagged
                )
                self.assertAlmostEqual(
                    float(symbolic.a), reference.a, delta=1e-15 * r_lagged
                )
                self.assertAlmostEqual(
                    float(symbolic.b2), reference.b2, delta=1e-15 * r_lagged**2
                )

    def test_constraint_vanishes_at_representative_divider_nodes(self) -> None:
        for r_lagged in (1e-3, 1e-6):
            graph = build_initial_four_patch_graph(r_lagged)
            coordinates = []
            for facet in graph.boundary_facets["divider"]:
                coordinates.extend(graph.nodes[key].coordinates for key in facet.node_keys)
            unique = tuple(dict.fromkeys(coordinates))
            self.assertGreaterEqual(len(unique), 5)
            for radial, axial in unique:
                residual = float(
                    eq6_divider_constraint(
                        r_lagged,
                        radial_coordinate=radial,
                        axial_coordinate=axial,
                    )
                )
                self.assertLess(abs(residual), 3e-14)


class AnthonyDividerIntegrationTests(unittest.TestCase):
    def test_real_interior_divider_jits_with_nonsingular_axis_endpoint(
        self,
    ) -> None:
        with _divider_problem(1e-3) as (problem, _, diagnostics):
            rows = _divider_multiplier_rows(problem)
            mesh = problem.get_mesh("drop")
            multiplier_id = mesh.has_interface_dof_id("_lagr_enf_bc_mesh_x")
            corner_nodes = [
                node for node in mesh.boundary_nodes("divider") if node.x(0) == 0.0
            ]
            self.assertEqual(len(corner_nodes), 1)
            corner = corner_nodes[0]
            multiplier_index = corner.additional_value_index(multiplier_id)
            self.assertTrue(corner.variable_position_pt().is_pinned(0))
            self.assertTrue(corner.variable_position_pt().is_pinned(1))
            self.assertTrue(corner.is_pinned(multiplier_index))
            self.assertAlmostEqual(corner.x(1), 1e-3, delta=2e-18)
            residual, jacobian = problem.assemble_jacobian(with_residual=True)
            initial = numpy.asarray(residual)[list(rows)]
            message = diagnostics.getvalue().lower()

        self.assertGreater(len(rows), 0)
        # The mapped edge is a polynomial Q2 interpolation through exact
        # ellipse nodes, not a rational conic element.  Its between-node weak
        # defect is therefore nonzero but remains small on the R0-scaled arc.
        self.assertLess(numpy.linalg.norm(initial, ord=numpy.inf), 5e-6)
        self.assertTrue(numpy.isfinite(jacobian.data).all())
        self.assertNotIn("redundant", message)
        self.assertNotIn("singular", message)

    def test_global_parameter_change_reassembles_constraint_and_jacobian(
        self,
    ) -> None:
        with _divider_problem(1e-3) as (problem, _, _):
            rows = _divider_multiplier_rows(problem)
            residual0, jacobian0 = problem.assemble_jacobian(with_residual=True)
            problem.R_lagged.value = 1.08e-3
            residual1, jacobian1 = problem.assemble_jacobian(with_residual=True)

        selected0 = numpy.asarray(residual0)[list(rows)]
        selected1 = numpy.asarray(residual1)[list(rows)]
        self.assertLess(numpy.linalg.norm(selected0, ord=numpy.inf), 5e-6)
        self.assertGreater(numpy.linalg.norm(selected1 - selected0, ord=numpy.inf), 1e-8)
        jacobian_change = jacobian1 - jacobian0
        self.assertGreater(numpy.linalg.norm(jacobian_change.data, ord=numpy.inf), 1e-8)


if __name__ == "__main__":
    unittest.main()
