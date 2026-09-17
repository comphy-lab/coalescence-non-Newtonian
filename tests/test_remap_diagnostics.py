"""Independent geometry/conservation gates for the real runtime remesh."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from problems.newtonian.coalescence_mapped import MappedCoalescenceProblem
from problems.newtonian.remap_diagnostics import (
    RemeshBaselineCalibration,
    RemeshGateThresholds,
    axisymmetric_divergence_at,
    calibrate_remesh_baseline,
    compare_remesh,
    diagnose_mesh,
)


ROOT = Path(__file__).resolve().parents[1]


def _case(r0: float = 1.0e-3) -> dict:
    case = json.loads((ROOT / "cases/anthony2020/T0-stokes-R0-1e-3.json").read_text())
    case["physics"]["initial_bridge"]["R0"] = r0
    case["physics"]["initial_bridge"]["Z0"] = 0.5 * r0 * r0
    return case


def _seed_divergence_free_velocity(problem: MappedCoalescenceProblem, amplitude: float = 0.125) -> None:
    mesh = problem.get_mesh("drop")
    fields = mesh.get_nodal_field_indices()
    for node in mesh.nodes():
        radius, height = float(node.x(0)), float(node.x(1))
        for slot in range(node.ntstorage()):
            # u_r=a*r, u_z=-2*a*z gives
            # d(u_r)/dr + u_r/r + d(u_z)/dz = a+a-2a = 0.
            node.set_value_at_t(slot, fields["velocity_x"], amplitude * radius)
            node.set_value_at_t(slot, fields["velocity_y"], -2.0 * amplitude * height)


def _rescaled_snapshot(snapshot, *, divergence: float, flux: float):
    """Make a deterministic nonzero-residual receipt without a solve."""
    div = replace(
        snapshot.divergence,
        l2_by_slot={slot: divergence for slot in snapshot.divergence.l2_by_slot},
        rms_by_slot={slot: divergence for slot in snapshot.divergence.rms_by_slot},
        maximum_abs_by_slot={slot: divergence for slot in snapshot.divergence.maximum_abs_by_slot},
        signed_integral_by_slot={slot: divergence for slot in snapshot.divergence.signed_integral_by_slot},
    )
    fluxes = {
        name: replace(
            receipt,
            flux_by_slot={slot: flux for slot in receipt.flux_by_slot},
        )
        for name, receipt in snapshot.boundary_fluxes.items()
    }
    return replace(snapshot, divergence=div, boundary_fluxes=fluxes)


class IndependentRemeshDiagnosticsTests(unittest.TestCase):
    def test_identity_remesh_is_roundoff_conservative_and_divergence_free(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remap-diagnostics-identity-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                _seed_divergence_free_velocity(problem)
                before = diagnose_mesh(problem, velocity_slots=range(7))
                transaction = problem.prepare_mapped_remesh(problem.R0)
                self.assertEqual(
                    (transaction.diagnostics.old_nodes, transaction.diagnostics.old_elements),
                    (39, 7),
                )
                problem.execute_prepared_remesh()
                after = diagnose_mesh(problem, velocity_slots=range(7))
                receipt = compare_remesh(before, after)

                self.assertTrue(receipt.passed, receipt.as_dict())
                self.assertLessEqual(receipt.maximum_relative_volume_change, 1.0e-10)
                self.assertEqual(receipt.interface.maximum_normal_error_relative, 0.0)
                self.assertEqual(receipt.interface.maximum_tangential_error_relative, 0.0)
                self.assertLessEqual(receipt.divergence_after, 1.0e-12)
                self.assertLessEqual(receipt.divergence_correction, 1.0e-12)
                self.assertEqual(set(receipt.divergence_after_by_slot), set(range(7)))
                # The boundary flux receipt uses the same 2*pi*r meridian
                # measure; for the manufactured divergence-free field its
                # closed-boundary sum is roundoff.
                for slot, volume_integral in after.net_flux_by_slot.items():
                    self.assertLessEqual(abs(volume_integral), 1.0e-10)

                # Probe the regular r->0 limit explicitly at an axis node.  A
                # direct u_r/r division there would be undefined; the
                # evaluator uses d(u_r)/dr instead.
                axis_element = next(
                    element
                    for element in problem.get_mesh("drop").elements()
                    if float(element.node_pt(0).x(0)) == 0.0
                    and float(element.node_pt(3).x(0)) == 0.0
                )
                self.assertLess(
                    abs(axisymmetric_divergence_at(axis_element, (-1.0, 0.0))),
                    1.0e-12,
                )

    def test_level_one_to_two_passes_volume_interface_and_velocity_gates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remap-diagnostics-level2-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                _seed_divergence_free_velocity(problem)
                before = diagnose_mesh(problem, velocity_slots=range(7))
                transaction = problem.prepare_mapped_remesh(
                    problem.R0,
                    cap_resolution=2,
                    bridge_normal_resolution=2,
                )
                self.assertEqual(
                    (transaction.diagnostics.new_nodes, transaction.diagnostics.new_elements),
                    (97, 20),
                )
                problem.execute_prepared_remesh()
                after = diagnose_mesh(problem, velocity_slots=range(7))
                receipt = compare_remesh(before, after)

                self.assertTrue(receipt.passed, receipt.as_dict())
                self.assertLessEqual(receipt.maximum_relative_volume_change, 1.0e-10)
                self.assertLessEqual(
                    receipt.interface.maximum_normal_error_relative,
                    RemeshGateThresholds().relative_interface_normal,
                )
                self.assertLessEqual(
                    receipt.interface.maximum_tangential_error_relative,
                    RemeshGateThresholds().relative_interface_tangential,
                )
                self.assertLessEqual(receipt.divergence_after, 1.0e-12)
                self.assertLessEqual(receipt.divergence_correction, 1.0e-12)
                for slot, volume_integral in after.net_flux_by_slot.items():
                    self.assertLessEqual(abs(volume_integral), 1.0e-10)

    def test_perturbed_interface_control_fails_the_local_reconstruction_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remap-diagnostics-negative-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                _seed_divergence_free_velocity(problem)
                before = diagnose_mesh(problem, velocity_slots=range(7))

                # Move a sphere-side Q2 interface control node without doing a
                # physical solve.  This is intentionally outside the remesh
                # contract and should be reported as a geometric failure.
                control = problem.get_mesh("drop/interface").element_pt(2).node_pt(1)
                control.set_x(0, float(control.x(0)) + 2.0e-2)
                control.set_x(1, float(control.x(1)) - 2.0e-2)
                after = diagnose_mesh(problem, velocity_slots=range(7))
                receipt = compare_remesh(before, after)

                self.assertFalse(receipt.passed, receipt.as_dict())
                self.assertFalse(receipt.gates["interface"])
                self.assertGreater(receipt.interface.maximum_normal_error_relative, 5.0e-3)
                self.assertIn("interface", receipt.failures)

    def test_real_state_calibration_is_baseline_relative_but_manufactured_gate_stays_strict(self) -> None:
        """A nonzero accepted-state residual needs an explicit bounded policy."""
        with tempfile.TemporaryDirectory(prefix="remap-diagnostics-calibration-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                _seed_divergence_free_velocity(problem)
                manufactured = diagnose_mesh(problem, velocity_slots=range(7))

                # Model the first accepted Taylor--Hood receipt having a
                # finite solver residual and nonzero boundary flux.  This is
                # receipt calibration only: no physical solve or timestep is
                # performed in this test.
                baseline = _rescaled_snapshot(
                    manufactured,
                    divergence=2.0e-6,
                    flux=2.0e-4,
                )
                after = _rescaled_snapshot(
                    manufactured,
                    divergence=2.05e-6,
                    flux=2.01e-4,
                )

                strict = compare_remesh(baseline, after)
                self.assertFalse(strict.passed, strict.as_dict())
                self.assertFalse(strict.gates["divergence"])
                self.assertFalse(strict.gates["divergence_correction"])
                self.assertFalse(strict.gates["flux"])
                self.assertEqual(strict.divergence_gate_mode, "absolute-manufactured")

                calibration = calibrate_remesh_baseline(
                    baseline,
                    relative_divergence_growth=0.10,
                    relative_divergence_correction=0.10,
                    relative_flux_change=0.10,
                )
                self.assertIsInstance(calibration, RemeshBaselineCalibration)
                restored_calibration = RemeshBaselineCalibration.from_dict(
                    json.loads(json.dumps(calibration.as_dict()))
                )
                self.assertTrue(restored_calibration.matches(baseline))
                receipt = compare_remesh(baseline, after, calibration=calibration)
                self.assertTrue(receipt.passed, receipt.as_dict())
                self.assertTrue(receipt.calibration_matches)
                self.assertTrue(receipt.baseline_relative_used)
                self.assertEqual(receipt.divergence_gate_mode, "baseline-relative")
                self.assertAlmostEqual(receipt.divergence_relative_change_by_slot[0], 0.025)
                self.assertAlmostEqual(receipt.flux_relative_change["interface"][0], 0.005)
                restored_receipt = compare_remesh(
                    baseline,
                    after,
                    calibration=restored_calibration,
                )
                self.assertTrue(restored_receipt.passed)

                # A zero manufactured baseline retains the absolute floor even
                # when passed through the explicit calibration path.
                zero_calibration = calibrate_remesh_baseline(manufactured)
                zero_after = _rescaled_snapshot(
                    manufactured,
                    divergence=2.0e-6,
                    flux=2.0e-4,
                )
                zero_receipt = compare_remesh(
                    manufactured,
                    zero_after,
                    calibration=zero_calibration,
                )
                self.assertFalse(zero_receipt.passed)
                self.assertFalse(zero_receipt.gates["divergence"])
                self.assertFalse(zero_receipt.gates["flux"])

    def test_calibration_cannot_be_reused_for_a_different_baseline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="remap-diagnostics-calibration-mismatch-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                _seed_divergence_free_velocity(problem)
                baseline = _rescaled_snapshot(
                    diagnose_mesh(problem, velocity_slots=range(7)),
                    divergence=2.0e-6,
                    flux=2.0e-4,
                )
                calibration = calibrate_remesh_baseline(baseline)
                other = replace(
                    baseline,
                    volume_by_slot={
                        slot: value + 1.0e-4
                        for slot, value in baseline.volume_by_slot.items()
                    },
                )
                receipt = compare_remesh(other, baseline, calibration=calibration)
                self.assertFalse(receipt.passed)
                self.assertFalse(receipt.calibration_matches)
                self.assertFalse(receipt.baseline_relative_used)
                self.assertFalse(receipt.gates["calibration"])
                self.assertIn("calibration", receipt.failures)


if __name__ == "__main__":
    unittest.main()
