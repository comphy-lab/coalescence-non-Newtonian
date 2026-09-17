"""Executable restart gate for a real mapped ``force_remesh`` transaction."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy

from problems.newtonian.coalescence_mapped import MappedCoalescenceProblem
from problems.newtonian.mapped_mesh import _graph_state_payload
from problems.newtonian.runtime_remesh import (
    KNOWN_INTERFACE_FIELDS,
    _runtime_nodes_by_template_index,
    bind_runtime_elements,
)


ROOT = Path(__file__).resolve().parents[1]


def _case() -> dict:
    return json.loads(
        (ROOT / "cases/anthony2020/T0-stokes-R0-1e-3.json").read_text()
    )


def _nodes_by_key(problem: MappedCoalescenceProblem):
    mesh = problem.get_mesh("drop")
    template = problem.mapped_mesh_template
    assert template is not None
    by_index = _runtime_nodes_by_template_index(template, mesh)
    return {
        key: by_index[index]
        for key, index in template.node_key_to_template_index.items()
    }


def _seed_frozen_state(problem: MappedCoalescenceProblem) -> None:
    """Populate every history family without advancing physical time."""
    mesh = problem.get_mesh("drop")
    fields = mesh.get_nodal_field_indices()
    nodes = _nodes_by_key(problem)
    for ordinal, key in enumerate(sorted(nodes)):
        node = nodes[key]
        x, y = float(node.x(0)), float(node.x(1))
        for slot in range(1, 7):
            node.set_x_at_t(slot, 0, x + 1.0e-7 * (slot + ordinal))
            node.set_x_at_t(slot, 1, y - 2.0e-7 * (slot + ordinal))
        for slot in range(1, node.ntstorage()):
            node.set_value_at_t(
                slot, fields["velocity_x"], 0.25 * slot + 2.0 * x - y
            )
            node.set_value_at_t(
                slot, fields["velocity_y"], -0.5 * slot - x + 3.0 * y
            )
        for slot in range(node.ntstorage()):
            node.set_value_at_t(slot, fields["pressure"], 7.0 + 0.125 * slot)

    for field_ordinal, name in enumerate(sorted(KNOWN_INTERFACE_FIELDS)):
        field_id = mesh.has_interface_dof_id(name)
        if field_id < 0:
            continue
        for ordinal, key in enumerate(sorted(nodes)):
            node = nodes[key]
            index = int(node.additional_value_index(field_id))
            if index < 0:
                continue
            for slot in range(1, node.ntstorage()):
                node.set_value_at_t(
                    slot,
                    index,
                    0.01 * (1 + field_ordinal) + 1.0e-4 * ordinal + 1.0e-3 * slot,
                )

    problem.set_current_time(0.125, dimensional=False, as_float=True)
    time = problem.timestepper.time_pt()
    for slot in range(time.ndt()):
        time.set_dt(slot, 0.01 * (slot + 1))
    problem.timestepper.set_num_unsteady_steps_done(4)
    problem._suggested_next_dt = 0.0075
    problem.timestepper.set_weights()


def _snapshot(problem: MappedCoalescenceProblem) -> dict:
    mesh = problem.get_mesh("drop")
    template = problem.mapped_mesh_template
    assert template is not None and template.graph is not None
    nodes = _nodes_by_key(problem)
    boundary_data = {
        name: (mesh.get_boundary_index(name), set(mesh.boundary_nodes(name)))
        for name in mesh.get_boundary_names()
        if mesh.is_boundary_coordinate_defined(mesh.get_boundary_index(name))
    }
    interface_ids = {
        name: mesh.has_interface_dof_id(name)
        for name in sorted(KNOWN_INTERFACE_FIELDS)
    }
    node_state = {}
    for key, node in nodes.items():
        interface_state = {}
        for name, field_id in interface_ids.items():
            if field_id < 0:
                continue
            index = int(node.additional_value_index(field_id))
            if index >= 0:
                interface_state[name] = (
                    index,
                    tuple(
                        float(node.value_at_t(slot, index))
                        for slot in range(node.ntstorage())
                    ),
                    bool(node.is_pinned(index)),
                )
        node_state[key] = {
            "positions": tuple(
                tuple(float(node.x_at_t(slot, component)) for component in range(node.ndim()))
                for slot in range(7)
            ),
            "lagrangian": tuple(float(node.x_lagr(component)) for component in range(node.ndim())),
            "values": tuple(
                tuple(float(node.value_at_t(slot, index)) for index in range(node.nvalue()))
                for slot in range(node.ntstorage())
            ),
            "pinned": tuple(bool(node.is_pinned(index)) for index in range(node.nvalue())),
            "position_pinned": tuple(
                bool(node.position_is_pinned(component)) for component in range(node.ndim())
            ),
            "interface": interface_state,
            "zeta": {
                name: tuple(float(value) for value in node.get_coordinates_on_boundary(index))
                for name, (index, members) in boundary_data.items()
                if node in members
            },
        }

    runtime_index = {id(node): key for key, node in nodes.items()}
    element_state = {
        element_id: tuple(runtime_index[id(element.node_pt(index))] for index in range(element.nnode()))
        for element_id, element in bind_runtime_elements(template, mesh).items()
    }
    dof_types, dof_names = problem.get_dof_description()
    time = problem.timestepper.time_pt()
    return {
        "graph": _graph_state_payload(template.graph),
        "nodes": node_state,
        "elements": element_state,
        "counts": (mesh.nnode(), mesh.nelement(), problem.ndof()),
        "dof_description": (tuple(int(value) for value in dof_types), tuple(dof_names)),
        "R_lagged": float(problem.R_lagged_parameter.value),
        "rmin_lagged": float(problem.rmin_lagged),
        "time": problem.get_current_time(dimensional=False, as_float=True),
        "dts": tuple(float(time.dt(slot)) for slot in range(time.ndt())),
        "steps": int(problem.timestepper.get_num_unsteady_steps_done()),
        "suggested_next_dt": problem._suggested_next_dt,
    }


class MappedRemeshRestartTests(unittest.TestCase):
    def _assert_roundtrip(self, destination_cap_resolution: int) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-remesh-restart-") as output_dir:
            state_file = str(Path(output_dir) / "mapped.state")
            writer = MappedCoalescenceProblem(_case(), Path(output_dir) / "writer")
            writer.set_linear_solver("superlu")
            with writer:
                writer.initialise()
                transaction = writer.prepare_mapped_remesh(
                    writer.R0,
                    cap_resolution=destination_cap_resolution,
                    bridge_normal_resolution=2,
                )
                writer.execute_prepared_remesh()
                self.assertTrue(transaction.completed)
                _seed_frozen_state(writer)
                before = _snapshot(writer)
                residual_before, jacobian_before = writer.assemble_jacobian(
                    with_residual=True
                )
                writer.save_state(state_file, quiet=True)

            # Deliberately use the untouched level-one case.  The state file,
            # rather than external case mutation, must select a remeshed graph.
            reader = MappedCoalescenceProblem(_case(), Path(output_dir) / "reader")
            reader.set_linear_solver("superlu")
            with reader:
                reader.initialise()
                reader.load_state(state_file, quiet=True)
                after = _snapshot(reader)
                residual_after, jacobian_after = reader.assemble_jacobian(
                    with_residual=True
                )

            self.assertEqual(after, before)
            # The replacement lifecycle changes accumulation order for two
            # seeded nonzero residual rows (observed 2.7e-23) and interface
            # geometric derivatives (observed 2.7e-16), even with SuperLU.
            numpy.testing.assert_allclose(
                residual_after,
                residual_before,
                rtol=0.0,
                atol=1.0e-11,
            )
            self.assertEqual(jacobian_after.shape, jacobian_before.shape)
            numpy.testing.assert_array_equal(jacobian_after.indptr, jacobian_before.indptr)
            numpy.testing.assert_array_equal(jacobian_after.indices, jacobian_before.indices)
            numpy.testing.assert_allclose(
                jacobian_after.data,
                jacobian_before.data,
                rtol=0.0,
                atol=1.0e-11,
            )

    def test_level_one_to_two_state_roundtrip_after_real_force_remesh(self) -> None:
        self._assert_roundtrip(destination_cap_resolution=2)

    def test_identity_remesh_state_roundtrip(self) -> None:
        self._assert_roundtrip(destination_cap_resolution=1)

    def test_z_loaded_restart_graph_is_one_shot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-remesh-one-shot-") as output_dir:
            state_file = str(Path(output_dir) / "mapped.state")
            writer = MappedCoalescenceProblem(_case(), Path(output_dir) / "writer")
            with writer:
                writer.initialise()
                writer.prepare_mapped_remesh(writer.R0, cap_resolution=2)
                writer.execute_prepared_remesh()
                writer.save_state(state_file, quiet=True)

            reader = MappedCoalescenceProblem(_case(), Path(output_dir) / "reader")
            with reader:
                reader.initialise()
                reader.load_state(state_file, quiet=True)
                self.assertIsNone(reader.mapped_mesh_template._restart_graph)
                with self.assertRaisesRegex(RuntimeError, "no prepared remesh graph"):
                    reader.force_remesh(
                        only_domains={reader.mapped_mesh_template}, num_adapt=0
                    )


if __name__ == "__main__":
    unittest.main()
