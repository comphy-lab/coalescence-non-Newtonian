"""Real serial ``force_remesh`` gates for the mapped Q2 transaction."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from problems.newtonian.coalescence_mapped import MappedCoalescenceProblem
from problems.newtonian.remap_transfer import pyoomph_local_coordinates
from problems.newtonian.runtime_remesh import _runtime_nodes_by_template_index
from problems.newtonian.structured_blocks import build_initial_four_patch_graph


ROOT = Path(__file__).resolve().parents[1]


def _case() -> dict:
    return json.loads(
        (ROOT / "cases/anthony2020/T0-stokes-R0-1e-3.json").read_text()
    )


def _assert_close(test: unittest.TestCase, actual: float, expected: float) -> None:
    tolerance = 2.0e-10 * max(1.0, abs(actual), abs(expected))
    test.assertLessEqual(abs(actual - expected), tolerance)


def _nodes_by_key(problem: MappedCoalescenceProblem):
    mesh = problem.get_mesh("drop")
    template = problem.mapped_mesh_template
    assert template is not None
    by_index = _runtime_nodes_by_template_index(template, mesh)
    return {
        key: by_index[index]
        for key, index in template.node_key_to_template_index.items()
    }


def _seed_histories(problem: MappedCoalescenceProblem) -> None:
    mesh = problem.get_mesh("drop")
    fields = mesh.get_nodal_field_indices()
    for ordinal, node in enumerate(_nodes_by_key(problem).values()):
        for slot in range(1, 7):
            node.set_x_at_t(slot, 0, 10.0 * slot + 0.01 * ordinal)
            node.set_x_at_t(slot, 1, -10.0 * slot - 0.02 * ordinal)
        x, y = float(node.x(0)), float(node.x(1))
        for slot in range(node.ntstorage()):
            node.set_value_at_t(slot, fields["velocity_x"], 100.0 * slot + 2.0 * x - y)
            node.set_value_at_t(slot, fields["velocity_y"], -50.0 * slot - x + 3.0 * y)
            node.set_value_at_t(slot, fields["pressure"], 7.0 + 0.25 * slot)


def _snapshot(problem: MappedCoalescenceProblem):
    fields = problem.get_mesh("drop").get_nodal_field_indices()
    result = {}
    for key, node in _nodes_by_key(problem).items():
        result[key] = (
            tuple((float(node.x_at_t(slot, 0)), float(node.x_at_t(slot, 1))) for slot in range(7)),
            {
                name: tuple(float(node.value_at_t(slot, index)) for slot in range(node.ntstorage()))
                for name, index in fields.items()
            },
        )
    return result


def _install_synthetic_runtime_graph(
    problem: MappedCoalescenceProblem, graph
) -> None:
    """Verification-only geometry replacement; this is not a physical solve."""
    nodes = _nodes_by_key(problem)
    if set(nodes) != set(graph.nodes):
        raise AssertionError("synthetic graph topology differs from the live template")
    for key, node in nodes.items():
        x, y = graph.nodes[key].coordinates
        node.set_x(0, x)
        node.set_x(1, y)
        node.set_x_lagr(0, x)
        node.set_x_lagr(1, y)


class RuntimeMappedRemeshTests(unittest.TestCase):
    def test_identity_remesh_preserves_all_seven_slots_fields_time_and_ndof(self) -> None:
        with tempfile.TemporaryDirectory(prefix="runtime-remesh-identity-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                _seed_histories(problem)
                before = _snapshot(problem)
                time_before = problem.get_current_time(dimensional=False, as_float=True)
                ndof_before = problem.ndof()

                transaction = problem.prepare_mapped_remesh(problem.R0)
                with self.assertRaisesRegex(RuntimeError, "already pending"):
                    problem.prepare_mapped_remesh(problem.R0)
                self.assertIs(problem._prepared_mapped_remesh, transaction)
                self.assertIs(problem.mapped_mesh_template.pending_remesh_graph, transaction.new_graph)
                self.assertEqual(
                    (transaction.diagnostics.old_nodes, transaction.diagnostics.old_elements),
                    (39, 7),
                )
                self.assertTrue(problem.execute_prepared_remesh())

                self.assertTrue(transaction.completed)
                self.assertEqual(transaction.destination_count, 39)
                self.assertEqual((transaction.missing_count, transaction.fallback_count), (0, 0))
                self.assertEqual(problem.ndof(), ndof_before)
                self.assertEqual(
                    problem.get_current_time(dimensional=False, as_float=True), time_before
                )
                after = _snapshot(problem)
                self.assertEqual(set(after), set(before))
                for key in before:
                    for slot in range(7):
                        for component in range(2):
                            _assert_close(
                                self, after[key][0][slot][component], before[key][0][slot][component]
                            )
                    for name in before[key][1]:
                        for actual, expected in zip(after[key][1][name], before[key][1][name]):
                            _assert_close(self, actual, expected)

                # Known interface-only multipliers are reconstructed, not
                # mistaken for transported bulk state.
                mesh = problem.get_mesh("drop")
                for name in ("_kin_bc", "_lagr_enf_bc_mesh_x"):
                    field_id = mesh.has_interface_dof_id(name)
                    self.assertGreaterEqual(field_id, 0)
                    for node in mesh.nodes():
                        index = node.additional_value_index(field_id)
                        if index >= 0:
                            self.assertTrue(
                                all(node.value_at_t(slot, index) == 0.0 for slot in range(node.ntstorage()))
                            )

    def test_level_one_to_two_reproduces_q2_velocity_and_q1_pressure_histories(self) -> None:
        with tempfile.TemporaryDirectory(prefix="runtime-remesh-level2-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                fields = problem.get_mesh("drop").get_nodal_field_indices()
                for node in problem.get_mesh("drop").nodes():
                    x, y = float(node.x(0)), float(node.x(1))
                    for slot in range(node.ntstorage()):
                        node.set_value_at_t(slot, fields["velocity_x"], 3.0 * x - 2.0 * y + slot)
                        node.set_value_at_t(slot, fields["velocity_y"], -x + 4.0 * y - 2.0 * slot)
                        node.set_value_at_t(slot, fields["pressure"], 11.0 + 0.5 * slot)

                time_before = problem.get_current_time(dimensional=False, as_float=True)
                transaction = problem.prepare_mapped_remesh(
                    problem.R0, cap_resolution=2, bridge_normal_resolution=2
                )
                self.assertEqual(
                    (transaction.diagnostics.new_nodes, transaction.diagnostics.new_elements),
                    (97, 20),
                )
                self.assertGreater(transaction.diagnostics.new_minimum_scaled_jacobian, 0.0)
                problem.execute_prepared_remesh()

                mesh = problem.get_mesh("drop")
                self.assertEqual((mesh.nnode(), mesh.nelement()), (97, 20))
                self.assertEqual(transaction.destination_count, 97)
                self.assertEqual((transaction.missing_count, transaction.fallback_count), (0, 0))
                self.assertEqual(
                    problem.get_current_time(dimensional=False, as_float=True), time_before
                )
                for node in mesh.nodes():
                    x, y = float(node.x(0)), float(node.x(1))
                    for slot in range(node.ntstorage()):
                        _assert_close(
                            self, node.value_at_t(slot, fields["velocity_x"]),
                            3.0 * x - 2.0 * y + slot,
                        )
                        _assert_close(
                            self, node.value_at_t(slot, fields["velocity_y"]),
                            -x + 4.0 * y - 2.0 * slot,
                        )
                        _assert_close(
                            self, node.value_at_t(slot, fields["pressure"]),
                            11.0 + 0.5 * slot,
                        )

    def test_reconstruction_shift_changes_only_coordinate_like_history_slots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="runtime-remesh-shift-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                _seed_histories(problem)
                template = problem.mapped_mesh_template
                assert template is not None and template.graph is not None
                boundary_keys = {
                    key
                    for facets in template.graph.boundary_facets.values()
                    for facet in facets
                    for key in facet.node_keys
                }
                moved_key = next(key for key in sorted(template.graph.nodes) if key not in boundary_keys)
                old_node = _nodes_by_key(problem)[moved_key]
                old_node.set_x(0, old_node.x(0) + 1.0e-8)

                transaction = problem.prepare_mapped_remesh(problem.R0)
                new_point = transaction.new_graph.nodes[moved_key].coordinates
                location = transaction.locator.locate(new_point)
                source_element = transaction.source_elements[location.element_id]
                local = pyoomph_local_coordinates(location)
                source_positions = [
                    source_element.get_interpolated_position_at_s(slot, local, False)
                    for slot in range(7)
                ]
                displacement = tuple(
                    new_point[component] - source_positions[0][component]
                    for component in range(2)
                )
                self.assertGreater(max(map(abs, displacement)), 0.0)
                problem.execute_prepared_remesh()

                node = _nodes_by_key(problem)[moved_key]
                for slot in (1, 2, 6):
                    for component in range(2):
                        _assert_close(
                            self,
                            node.x_at_t(slot, component),
                            source_positions[slot][component] + displacement[component],
                        )
                for slot in (3, 4, 5):
                    for component in range(2):
                        _assert_close(
                            self,
                            node.x_at_t(slot, component),
                            source_positions[slot][component],
                        )

    def test_synthetic_evolved_four_r0_remesh_precomputes_plan_and_epoch(self) -> None:
        """Verification geometry at 4R0 exercises a genuinely evolved epoch."""
        with tempfile.TemporaryDirectory(prefix="runtime-remesh-evolved-") as output_dir:
            state_file = str(Path(output_dir) / "evolved.state")
            problem = MappedCoalescenceProblem(_case(), Path(output_dir) / "writer")
            with problem:
                problem.initialise()
                evolved_rmin = 4.0 * problem.R0
                synthetic = build_initial_four_patch_graph(
                    evolved_rmin,
                    cap_resolution=1,
                    bridge_normal_resolution=2,
                )
                self.assertGreater(synthetic.minimum_scaled_jacobian(), 0.0)
                _install_synthetic_runtime_graph(problem, synthetic)
                problem._set_lagged_rmin(evolved_rmin)
                _seed_histories(problem)

                transaction = problem.prepare_mapped_remesh(
                    evolved_rmin,
                    cap_resolution=2,
                    bridge_normal_resolution=2,
                )
                self.assertEqual(len(transaction.source_locations), 97)
                self.assertEqual(transaction.destination_count, 97)
                self.assertEqual((transaction.missing_count, transaction.fallback_count), (0, 0))
                self.assertGreater(transaction.diagnostics.new_minimum_scaled_jacobian, 0.0)
                # Location is a preparation-time operation.  Execution must
                # only bind the frozen locations to destination nodes.
                with patch.object(
                    transaction.locator,
                    "locate",
                    side_effect=AssertionError("post-replacement locator call"),
                ):
                    problem.execute_prepared_remesh()

                self.assertTrue(transaction.completed)
                self.assertEqual(
                    (problem.get_mesh("drop").nnode(), problem.get_mesh("drop").nelement()),
                    (97, 20),
                )
                self.assertEqual(problem._rmin_at_remesh, evolved_rmin)
                self.assertEqual(problem._abs2h_at_remesh, problem.abs_2h_lagged)
                trigger = problem.remesh_factor * evolved_rmin
                self.assertFalse(problem.maybe_remesh(math.nextafter(trigger, 0.0)))
                with self.assertRaisesRegex(RuntimeError, "remap verification gate"):
                    problem.maybe_remesh(trigger)
                residual, jacobian = problem.assemble_jacobian(with_residual=True)
                self.assertEqual(residual.shape, (problem.ndof(),))
                self.assertEqual(jacobian.shape, (problem.ndof(), problem.ndof()))
                self.assertTrue(all(math.isfinite(float(value)) for value in residual))
                problem.save_state(state_file, quiet=True)

            reader = MappedCoalescenceProblem(_case(), Path(output_dir) / "reader")
            with reader:
                reader.initialise()
                reader.load_state(state_file, quiet=True)
                self.assertEqual(reader._rmin_at_remesh, evolved_rmin)
                self.assertEqual(reader._abs2h_at_remesh, problem.abs_2h_lagged)
                self.assertFalse(reader.maybe_remesh(math.nextafter(trigger, 0.0)))

    def test_destination_location_failure_is_atomic_before_template_pending(self) -> None:
        with tempfile.TemporaryDirectory(prefix="runtime-remesh-atomic-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                template = problem.mapped_mesh_template
                assert template is not None and template.graph is not None
                graph_before = template.graph
                coordinates_before = {
                    key: (float(node.x(0)), float(node.x(1)))
                    for key, node in _nodes_by_key(problem).items()
                }

                from problems.newtonian import runtime_remesh

                real_builder = runtime_remesh.build_evolved_four_patch_graph

                def outside_builder(*args, **kwargs):
                    graph, intersection = real_builder(*args, **kwargs)
                    key = next(iter(sorted(graph.nodes)))
                    nodes = dict(graph.nodes)
                    nodes[key] = replace(nodes[key], coordinates=(100.0, 100.0))
                    return replace(graph, nodes=nodes), intersection

                with patch.object(
                    runtime_remesh,
                    "build_evolved_four_patch_graph",
                    side_effect=outside_builder,
                ), self.assertRaisesRegex(RuntimeError, "outside every"):
                    problem.prepare_mapped_remesh(problem.R0)

                self.assertIsNone(problem._prepared_mapped_remesh)
                self.assertIsNone(template.pending_remesh_graph)
                self.assertIs(template.graph, graph_before)
                self.assertEqual(
                    {
                        key: (float(node.x(0)), float(node.x(1)))
                        for key, node in _nodes_by_key(problem).items()
                    },
                    coordinates_before,
                )


if __name__ == "__main__":
    unittest.main()
