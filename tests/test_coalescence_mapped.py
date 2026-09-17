"""Initialise-only integration gates for the mapped Newtonian problem."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from problems.newtonian.coalescence_mapped import MappedCoalescenceProblem


ROOT = Path(__file__).resolve().parents[1]


def _load_case(name: str) -> dict:
    return json.loads((ROOT / "cases" / "anthony2020" / name).read_text())


class MappedCoalescenceInitialisationTests(unittest.TestCase):
    def test_t0_and_t1_initialise_on_mapped_mesh_without_gmsh(self) -> None:
        for case_name in (
            "T0-stokes-R0-1e-3.json",
            "T1-Oh0.6-R0-1e-03.json",
        ):
            with self.subTest(case=case_name), tempfile.TemporaryDirectory(
                prefix="coalescence-mapped-test-"
            ) as output_dir:
                case = _load_case(case_name)
                problem = MappedCoalescenceProblem(case, output_dir)
                with problem:
                    problem.initialise()
                    mesh = problem.get_mesh("drop")
                    self.assertEqual(mesh.nelement(), 7)
                    self.assertEqual(mesh.nnode(), 39)
                    self.assertLess(problem.ndof(), 400)
                    self.assertAlmostEqual(
                        float(problem.R_lagged_parameter.value), problem.R0
                    )
                    self.assertGreater(
                        mesh.has_interface_dof_id("_lagr_enf_bc_mesh_x"), -1
                    )
                    self.assertEqual(
                        set(mesh.get_boundary_names()),
                        {"axis", "plane", "interface", "divider"},
                    )
                    origin = [
                        node
                        for node in mesh.nodes()
                        if node.x(0) == 0.0 and node.x(1) == 0.0
                    ]
                    self.assertEqual(len(origin), 1)
                    pressure_index = mesh.element_pt(0).get_jit_code().get_nodal_field_indices()[
                        "pressure"
                    ]
                    self.assertFalse(origin[0].is_pinned(pressure_index))

                self.assertFalse((Path(output_dir) / "pyoomph" / "_gmsh").exists())

    def test_lag_update_and_remesh_gate_fail_closed_without_time_advance(self) -> None:
        case = _load_case("T0-stokes-R0-1e-3.json")
        with tempfile.TemporaryDirectory(prefix="coalescence-mapped-test-") as output_dir:
            problem = MappedCoalescenceProblem(case, output_dir)
            with problem:
                problem.initialise()
                problem._set_lagged_rmin(1.25 * problem.R0)
                self.assertAlmostEqual(
                    float(problem.R_lagged_parameter.value), 1.25 * problem.R0
                )
                self.assertFalse(problem.maybe_remesh(3.99 * problem.R0))
                with self.assertRaisesRegex(RuntimeError, "remap verification gate"):
                    problem.maybe_remesh(4.0 * problem.R0)


if __name__ == "__main__":
    unittest.main()
