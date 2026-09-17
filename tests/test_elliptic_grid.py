"""Focused software tests for the project-local controlled elliptic grid."""

from __future__ import annotations

from contextlib import contextmanager
import math
import tempfile
import unittest

import numpy

from pyoomph import DirichletBC, ElementSpace, Problem, RectangularQuadMesh

from problems.newtonian.elliptic_grid import (
    ControlledEllipticMesh,
    LogicalCoordinateTable,
    mesh_quality,
)


def _logical(position, scale):
    return position[0] / scale, position[1] / scale


@contextmanager
def _temporary_problem(problem):
    with tempfile.TemporaryDirectory(prefix="elliptic-grid-test-") as output_dir:
        problem.set_output_directory(output_dir)
        with problem:
            yield problem


class _GridProblem(Problem):
    def __init__(self, scale=1.0, *, table=None):
        super().__init__()
        self.scale = scale
        self.table = table
        self.initial_adaption_steps = 0

    def define_problem(self):
        self.set_linear_solver("superlu")
        self.add_mesh(RectangularQuadMesh(N=2, size=[self.scale, self.scale]))
        resolver = self.table or (lambda position: _logical(position, self.scale))
        equations = ControlledEllipticMesh(logical_coordinates=resolver)
        equations += ElementSpace("C2")
        equations += DirichletBC(mesh_x=True, mesh_y=True) @ [
            "left", "right", "bottom", "top"
        ]
        self.add_equations(equations @ "domain")


def _interior_nodes(problem):
    scale = problem.scale
    return [
        node
        for node in problem.get_mesh("domain").nodes()
        if 1e-12 * scale < node.x(0) < (1.0 - 1e-12) * scale
        and 1e-12 * scale < node.x(1) < (1.0 - 1e-12) * scale
    ]


def _distort(problem):
    nodes = _interior_nodes(problem)
    for node in nodes:
        xi, eta = node.x_lagr(0), node.x_lagr(1)
        envelope = xi * (1.0 - xi) * eta * (1.0 - eta)
        node.set_x(0, node.x(0) + problem.scale * 0.8 * envelope)
        node.set_x(1, node.x(1) - problem.scale * 0.55 * envelope)
    return nodes


class LogicalCoordinateTests(unittest.TestCase):
    def test_exact_table_assigns_a_nonphysical_logical_chart(self):
        scale = 0.25
        coords = (0.0, 0.5, 1.0)
        table = LogicalCoordinateTable({
            (scale * x, scale * y): (2.0 * x + 0.125, 3.0 * y - 0.25)
            for x in coords for y in coords
        })

        class P(Problem):
            def define_problem(self):
                self.add_mesh(RectangularQuadMesh(N=1, size=[scale, scale]))
                equations = ControlledEllipticMesh(logical_coordinates=table)
                equations += ElementSpace("C2")
                equations += DirichletBC(mesh_x=True, mesh_y=True) @ [
                    "left", "right", "bottom", "top"
                ]
                self.add_equations(equations @ "domain")

        with _temporary_problem(P()) as problem:
            problem.initialise()
            for node in problem.get_mesh("domain").nodes():
                x, y = node.x(0), node.x(1)
                self.assertAlmostEqual(node.x_lagr(0), 2.0 * x / scale + 0.125)
                self.assertAlmostEqual(node.x_lagr(1), 3.0 * y / scale - 0.25)

    def test_table_rejects_missing_and_ambiguous_entries(self):
        exact = LogicalCoordinateTable({(0.0, 0.0): (0.0, 0.0)})
        with self.assertRaisesRegex(RuntimeError, "no logical coordinate"):
            exact((1.0, 1.0))
        ambiguous = LogicalCoordinateTable(
            [((0.0, 0.0), (0.0, 0.0)), ((0.1, 0.0), (1.0, 0.0))],
            tolerance=0.1,
        )
        with self.assertRaisesRegex(RuntimeError, "ambiguous"):
            ambiguous((0.05, 0.0))


class ControlledEllipticMeshTests(unittest.TestCase):
    def test_q2_positive_centre_negative_corner_is_rejected(self):
        class Node:
            def __init__(self, s, t):
                self._logical = (s, t)
                self._physical = (s, t * (1.0 + 18.0 * s))

            def x(self, component):
                return self._physical[component]

            def x_lagr(self, component):
                return self._logical[component]

        class Element:
            def __init__(self):
                self._nodes = [Node(s, t) for t in (-1.0, 0.0, 1.0)
                               for s in (-1.0, 0.0, 1.0)]

            def nnode(self):
                return 9

            def node_pt(self, index):
                return self._nodes[index]

        class Mesh:
            def elements(self):
                return iter((Element(),))

        quality = mesh_quality(Mesh())
        self.assertEqual(quality.sample_count_per_element, 17)
        self.assertAlmostEqual(quality.min_jacobian, -17.0)
        with self.assertRaisesRegex(RuntimeError, "inverted or degenerate"):
            mesh_quality(Mesh(), require_positive=True)

    def test_distorted_q2_grid_solves_back_to_harmonic_map(self):
        with _temporary_problem(_GridProblem()) as problem:
            problem.initialise()
            nodes = _distort(problem)
            before = max(math.hypot(node.x(0) - node.x_lagr(0),
                                    node.x(1) - node.x_lagr(1)) for node in nodes)
            before_quality = mesh_quality(problem.get_mesh("domain"), require_positive=True)
            problem.solve(newton_solver_tolerance=1e-12, max_newton_iterations=20)
            after = max(math.hypot(node.x(0) - node.x_lagr(0),
                                   node.x(1) - node.x_lagr(1)) for node in nodes)
            quality = mesh_quality(problem.get_mesh("domain"), require_positive=True)
            convergence = list(problem.get_last_residual_convergence())

        self.assertGreater(before, 1e-3)
        self.assertGreater(before_quality.min_jacobian, 0.0)
        self.assertLess(after, 2e-10)
        self.assertGreater(quality.min_jacobian, 0.999999999)
        self.assertLess(quality.max_condition_number, 1.00000001)
        self.assertTrue(convergence)
        self.assertLess(convergence[-1], 1e-12)

    def test_analytic_jacobian_matches_scale_relative_directional_difference(self):
        scale = 1e-4
        with _temporary_problem(_GridProblem(scale=scale)) as problem:
            problem.initialise()
            nodes = _distort(problem)
            residual, jacobian = problem.assemble_jacobian(with_residual=True)
            direction = numpy.zeros(problem.ndof())
            saved = []
            for index, node in enumerate(nodes):
                dx = scale * (0.3 + 0.07 * index)
                dy = scale * (-0.2 + 0.05 * index)
                ex = node.variable_position_pt().eqn_number(0)
                ey = node.variable_position_pt().eqn_number(1)
                self.assertGreaterEqual(ex, 0)
                self.assertGreaterEqual(ey, 0)
                direction[ex], direction[ey] = dx, dy
                saved.append((node, node.x(0), node.x(1), dx, dy))

            analytic = jacobian.dot(direction)
            epsilon = 2e-6
            for node, x, y, dx, dy in saved:
                node.set_x(0, x + epsilon * dx)
                node.set_x(1, y + epsilon * dy)
            plus, _ = problem.assemble_jacobian(with_residual=True)
            for node, x, y, dx, dy in saved:
                node.set_x(0, x - epsilon * dx)
                node.set_x(1, y - epsilon * dy)
            minus, _ = problem.assemble_jacobian(with_residual=True)
            for node, x, y, _, _ in saved:
                node.set_x(0, x)
                node.set_x(1, y)

        finite_difference = (numpy.asarray(plus) - numpy.asarray(minus)) / (2 * epsilon)
        error = numpy.linalg.norm(analytic - finite_difference, ord=numpy.inf)
        reference = max(1.0, numpy.linalg.norm(finite_difference, ord=numpy.inf))
        self.assertLess(error / reference, 3e-7)

    def test_uniform_physical_scaling_preserves_scaled_residual_and_quality(self):
        receipts = []
        for scale in (1.0, 1e-6):
            with _temporary_problem(_GridProblem(scale=scale)) as problem:
                problem.initialise()
                _distort(problem)
                residual, jacobian = problem.assemble_jacobian(with_residual=True)
                quality = mesh_quality(problem.get_mesh("domain"), require_positive=True)
                receipts.append((
                    scale * numpy.asarray(residual),
                    scale * scale * jacobian.toarray(),
                    quality,
                ))

        residual0, jacobian0, quality0 = receipts[0]
        residual1, jacobian1, quality1 = receipts[1]
        self.assertTrue(numpy.allclose(residual0, residual1, rtol=2e-10, atol=2e-11))
        self.assertTrue(numpy.allclose(jacobian0, jacobian1, rtol=2e-9, atol=2e-10))
        self.assertAlmostEqual(quality0.min_jacobian,
                               quality1.min_jacobian / (1e-6 ** 2), places=10)
        self.assertAlmostEqual(quality0.max_condition_number,
                               quality1.max_condition_number, places=10)
        self.assertAlmostEqual(quality0.max_column_aspect,
                               quality1.max_column_aspect, places=10)


if __name__ == "__main__":
    unittest.main()
