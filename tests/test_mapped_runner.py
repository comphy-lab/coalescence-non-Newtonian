"""Configuration and frozen-state gates for the mapped production runner."""

from __future__ import annotations

import json
import math
import io
import hashlib
import copy
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from problems.newtonian.coalescence_mapped import (
    MappedCoalescenceProblem,
    mapped_completion_reached,
)
from problems.newtonian.coalescence_axisym import load_case
from problems.newtonian.runtime_remesh import _runtime_nodes_by_template_index
from problems.newtonian.structured_blocks import build_initial_four_patch_graph
from runners.run_mapped_case import main, numerical_policy


ROOT = Path(__file__).resolve().parents[1]


def _case() -> dict:
    return json.loads(
        (ROOT / "cases" / "anthony2020" / "T0-stokes-R0-1e-3.json").read_text()
    )


class MappedRunnerConfigurationTests(unittest.TestCase):
    def test_exact_fig3_r0_1e_6_cases_have_finite_q2_neck_state(self) -> None:
        for case_name in (
            "T5-stokes-R0-1e-06.json",
            "T5-Oh0.6-R0-1e-06.json",
        ):
            with self.subTest(case=case_name), tempfile.TemporaryDirectory(
                prefix="mapped-run-fig3-"
            ) as output_dir:
                case = load_case(ROOT / "cases" / "anthony2020" / case_name)
                problem = MappedCoalescenceProblem(case, output_dir)
                with problem:
                    problem.initialise()
                    row = problem.neck_state()
                    expected = 1.0 / problem.Z0 + 1.0 / problem.R0
                    self.assertEqual(
                        (problem.get_mesh("drop").nnode(), problem.get_mesh("drop").nelement()),
                        (1345, 320),
                    )
                    self.assertEqual(problem.ndof(), 5649)
                    self.assertEqual(
                        problem.get_current_time(dimensional=False, as_float=True), 0.0
                    )
                    self.assertTrue(math.isfinite(row["abs_2H"]))
                    self.assertLessEqual(abs(row["abs_2H"] / expected - 1.0), 0.05)
                    bound = problem.live_mapped_timestep_bound(
                        maxstep=float(case["run"]["maxstep"]),
                        rmin=row["R_min"],
                        abs_2h=row["abs_2H"],
                    )
                    components = (
                        bound.bulk_singular_length,
                        bound.interface_segment_scale,
                        bound.neck_quadratic_scale,
                        bound.curvature_scale,
                    )
                    self.assertEqual(bound.selected_length_scale, min(components))
                    self.assertLess(bound.selected_length_scale, 1.0e-12)
                    self.assertGreater(
                        bound.selected_length_scale, bound.coordinate_ulp_floor
                    )
                    self.assertEqual(bound.maximum_sampled_speed, 0.0)
                    self.assertEqual(
                        bound.selected_dt_cap, float(case["run"]["maxstep"])
                    )
                    self.assertEqual(bound.speed_source, "sampled-fluid-q2")
                    high_curvature = problem.live_mapped_timestep_bound(
                        maxstep=float(case["run"]["maxstep"]),
                        rmin=1.0e-6,
                        abs_2h=1.0e18,
                    )
                    self.assertEqual(high_curvature.curvature_scale, 4.0e-18)
                    self.assertEqual(
                        high_curvature.selected_length_scale, 4.0e-18
                    )
                    self.assertLess(
                        high_curvature.coordinate_ulp_floor,
                        high_curvature.selected_length_scale,
                    )
                    self.assertLess(
                        high_curvature.selected_length_scale, 1.0e-14
                    )

    def test_q2_neck_curvature_converges_at_exact_fig3_scale(self) -> None:
        source = load_case(
            ROOT / "cases" / "anthony2020" / "T5-stokes-R0-1e-06.json"
        )
        for resolution, tolerance in ((4, 0.17), (8, 0.05), (16, 0.013)):
            with self.subTest(resolution=resolution), tempfile.TemporaryDirectory(
                prefix="mapped-run-curvature-"
            ) as output_dir:
                case = copy.deepcopy(source)
                case["mesh"]["cap_resolution"] = resolution
                case["mesh"]["bridge_normal_resolution"] = resolution
                problem = MappedCoalescenceProblem(case, output_dir)
                with problem:
                    problem.initialise()
                    curvature = problem.neck_state()["abs_2H"]
                    expected = 1.0 / problem.Z0 + 1.0 / problem.R0
                    self.assertTrue(math.isfinite(curvature))
                    self.assertLessEqual(abs(curvature / expected - 1.0), tolerance)

    def test_live_bound_uses_seeded_ale_and_relative_speeds_after_t0(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-ale-speed-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                dt = 0.01
                mesh_velocity = (2.0, -1.0)
                for node in problem.get_mesh("drop").nodes():
                    for component in range(2):
                        node.set_x_at_t(
                            1,
                            component,
                            float(node.x_at_t(0, component))
                            - dt * mesh_velocity[component],
                        )
                problem.set_current_time(0.1, dimensional=False, as_float=True)
                problem.time_pt().set_dt(0, dt)
                row = problem.neck_state()
                bound = problem.live_mapped_timestep_bound(
                    maxstep=1.0,
                    rmin=row["R_min"],
                    abs_2h=row["abs_2H"],
                )
                expected_speed = math.hypot(*mesh_velocity)
                self.assertEqual(bound.maximum_fluid_speed, 0.0)
                self.assertAlmostEqual(bound.maximum_mesh_speed, expected_speed)
                self.assertAlmostEqual(bound.maximum_relative_speed, expected_speed)
                self.assertAlmostEqual(bound.maximum_sampled_speed, expected_speed)
                self.assertEqual(
                    bound.speed_source, "sampled-max-fluid-mesh-relative-q2"
                )
                self.assertAlmostEqual(
                    bound.selected_dt_cap,
                    0.4 * bound.selected_length_scale / expected_speed,
                )

    def test_numerical_policy_is_case_hashed_and_strict(self) -> None:
        case = load_case(
            ROOT / "cases" / "anthony2020" / "T5-stokes-R0-1e-06.json"
        )
        missing = copy.deepcopy(case)
        del missing["run"]["maxstep"]
        with self.assertRaisesRegex(ValueError, "run.maxstep"):
            numerical_policy(missing)
        wrong_solver = copy.deepcopy(case)
        wrong_solver["run"]["linear_solver"] = "pardiso"
        with self.assertRaisesRegex(ValueError, "superlu"):
            numerical_policy(wrong_solver)
        too_tight = copy.deepcopy(case)
        too_tight["tolerances"]["newton"] = 1.0e-9
        with self.assertRaisesRegex(ValueError, ">= 5e-8"):
            numerical_policy(too_tight)
        no_states = copy.deepcopy(case)
        no_states["output"]["write_states"] = False
        with self.assertRaisesRegex(ValueError, "write_states"):
            numerical_policy(no_states)

    def test_endpoint_status_is_success_only_for_rmin_completion(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-endpoint-") as output_dir:
            incomplete = MappedCoalescenceProblem(_case(), Path(output_dir) / "incomplete")
            with incomplete:
                with self.assertRaisesRegex(RuntimeError, "max_time"):
                    incomplete.run_until(max_time=0.0, startstep=1.0e-12, maxstep=1.0e-6)
                self.assertEqual(
                    incomplete.get_current_time(dimensional=False, as_float=True), 0.0
                )
            complete = MappedCoalescenceProblem(_case(), Path(output_dir) / "complete")
            complete.stop_rmin = complete.R0
            with complete:
                complete.run_until(max_time=1.0, startstep=1.0e-12, maxstep=1.0e-6)
                self.assertEqual(
                    complete.get_current_time(dimensional=False, as_float=True), 0.0
                )
    def test_exact_fig3_case_contains_effective_hashed_numerical_policy(self) -> None:
        case = load_case(
            ROOT / "cases" / "anthony2020" / "T5-stokes-R0-1e-06.json"
        )
        policy = numerical_policy(case)
        self.assertEqual(policy.linear_solver, "superlu")
        self.assertEqual(policy.remesh_max_trigger_ratio, 4.05)
        self.assertEqual(policy.completion_threshold, 0.03)
        self.assertEqual((policy.startstep, policy.maxstep), (5.0e-15, 5.0e-8))

    def test_accepted_step_completion_policy_and_predicate(self) -> None:
        case = load_case(
            ROOT / "cases" / "anthony2020" / "T5-stokes-R0-1e-06.json"
        )
        case["completion"] = {
            "kind": "event",
            "value": "accepted_steps >= 1",
            "observable": "accepted_steps",
            "operator": ">=",
            "threshold": 1,
        }
        policy = numerical_policy(case)
        self.assertEqual(policy.completion_observable, "accepted_steps")
        self.assertEqual(policy.completion_threshold, 1)
        self.assertFalse(mapped_completion_reached(
            "accepted_steps", 1, rmin=1.0e-6, accepted_steps=0
        ))
        self.assertTrue(mapped_completion_reached(
            "accepted_steps", 1, rmin=1.0e-6, accepted_steps=1
        ))
        invalid = copy.deepcopy(case)
        invalid["completion"]["threshold"] = 1.5
        with self.assertRaisesRegex(ValueError, "positive integer"):
            numerical_policy(invalid)

    def test_mocked_one_step_endpoint_calibrates_before_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-one-step-") as output_dir:
            receipts = Path(output_dir) / "receipts.jsonl"
            progress: list[dict] = []
            problem = MappedCoalescenceProblem(_case(), output_dir)
            problem.configure_automatic_remeshing(receipts)
            problem.set_progress_callback(progress.append)
            with problem:
                problem.initialise()

                def accepted_step(**_kwargs):
                    problem.set_current_time(1.0e-6, dimensional=False, as_float=True)
                    problem.time_pt().set_dt(0, 1.0e-6)
                    problem.timestepper.set_num_unsteady_steps_done(1)
                    row = problem.neck_state()
                    problem._set_lagged_rmin(row["R_min"])
                    problem.abs_2h_lagged = row["abs_2H"]
                    problem.record_accepted_remesh_baseline_for_verification(
                        physical_time=1.0e-6,
                        accepted_steps=1,
                    )
                    return 1.0e-6

                with patch.object(problem, "solve", side_effect=accepted_step) as solve:
                    problem.run_until(
                        max_time=1.0,
                        startstep=1.0e-6,
                        maxstep=1.0e-4,
                        completion_observable="accepted_steps",
                        completion_threshold=1,
                    )
                    solve.assert_called_once()
                self.assertEqual(problem.automatic_remesh_state, "armed")
                self.assertEqual(progress[-1]["event"], "endpoint")
                self.assertEqual(progress[-1]["endpoint"], "accepted_steps")
                self.assertEqual(progress[-1]["accepted_steps"], 1)
            receipt_rows = [
                json.loads(line) for line in receipts.read_text().splitlines()
            ]
            self.assertEqual(
                receipt_rows[0]["kind"], "accepted-state-identity-calibration"
            )
            self.assertTrue(receipt_rows[0]["passed"])

    def test_configured_problem_is_not_armed_before_live_calibration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-unarmed-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            problem.configure_automatic_remeshing(Path(output_dir) / "receipts.jsonl")
            self.assertEqual(problem.automatic_remesh_state, "awaiting-accepted-step")
            with self.assertRaisesRegex(RuntimeError, "accepted-step"):
                problem.maybe_remesh(problem.remesh_factor * problem.R0)

    def test_mapped_output_propagates_diagnostic_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-output-failure-") as output_dir:
            problem = MappedCoalescenceProblem(_case(), output_dir)
            with problem:
                problem.initialise()
                with patch.object(
                    problem,
                    "write_neck_row",
                    side_effect=RuntimeError("synthetic output failure"),
                ), self.assertRaisesRegex(RuntimeError, "synthetic output failure"):
                    problem.output("timestep")

    def test_live_identity_calibration_arms_without_advancing_time(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-calibration-") as output_dir:
            receipts = Path(output_dir) / "receipts.jsonl"
            problem = MappedCoalescenceProblem(_case(), output_dir)
            problem.configure_automatic_remeshing(receipts)
            with problem:
                problem.initialise()
                time_before = problem.get_current_time(
                    dimensional=False, as_float=True
                )
                problem.record_accepted_remesh_baseline_for_verification(
                    physical_time=1.0e-12, accepted_steps=1
                )
                result = problem.calibrate_automatic_remeshing()
                self.assertTrue(result["passed"], result)
                self.assertEqual(problem.automatic_remesh_state, "armed")
                self.assertEqual(
                    problem.get_current_time(dimensional=False, as_float=True),
                    time_before,
                )
                self.assertFalse(problem.maybe_remesh(3.99 * problem.R0))
            records = [json.loads(line) for line in receipts.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["kind"], "accepted-state-identity-calibration")
            self.assertTrue(records[0]["passed"])
            self.assertEqual(records[0]["missing_locations"], 0)
            self.assertEqual(records[0]["fallback_locations"], 0)

    def test_accepted_calibration_and_epoch_restore_without_recalibration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-restart-policy-") as output_dir:
            state = str(Path(output_dir) / "accepted.state")
            writer = MappedCoalescenceProblem(_case(), Path(output_dir) / "writer")
            writer.configure_automatic_remeshing(
                Path(output_dir) / "writer-receipts.jsonl"
            )
            with writer:
                writer.initialise()
                writer.record_accepted_remesh_baseline_for_verification(
                    physical_time=1.0e-12, accepted_steps=1
                )
                writer.calibrate_automatic_remeshing()
                baseline = writer._rmin_at_remesh
                writer.save_state(state, quiet=True)

            reader = MappedCoalescenceProblem(_case(), Path(output_dir) / "reader")
            reader.configure_automatic_remeshing(
                Path(output_dir) / "reader-receipts.jsonl"
            )
            with reader:
                reader.initialise()
                reader.load_state(state, quiet=True)
                self.assertEqual(reader.automatic_remesh_state, "armed")
                self.assertEqual(reader._rmin_at_remesh, baseline)
                self.assertIsNotNone(reader._automatic_remesh_calibration)
                time_before = reader.get_current_time(
                    dimensional=False, as_float=True
                )
                with self.assertRaisesRegex(RuntimeError, "max_time"):
                    reader.run_until(
                        max_time=time_before,
                        startstep=1.0e-12,
                        maxstep=1.0e-6,
                    )
                self.assertEqual(
                    reader.get_current_time(dimensional=False, as_float=True),
                    time_before,
                )
                self.assertEqual(reader._rmin_at_remesh, baseline)

    def test_automatic_remesh_checkpoint_restores_post_remesh_calibration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-remesh-restart-") as output_dir:
            state = str(Path(output_dir) / "post-remesh.state")
            writer = MappedCoalescenceProblem(_case(), Path(output_dir) / "writer")
            writer.configure_automatic_remeshing(
                Path(output_dir) / "writer-receipts.jsonl"
            )
            with writer:
                writer.initialise()
                writer.record_accepted_remesh_baseline_for_verification(
                    physical_time=1.0e-12, accepted_steps=1
                )
                writer.calibrate_automatic_remeshing()
                evolved_rmin = writer.remesh_factor * writer.R0
                graph = build_initial_four_patch_graph(
                    evolved_rmin, cap_resolution=1, bridge_normal_resolution=2
                )
                template = writer.mapped_mesh_template
                assert template is not None
                runtime_nodes = _runtime_nodes_by_template_index(
                    template, writer.get_mesh("drop")
                )
                for key, index in template.node_key_to_template_index.items():
                    node = runtime_nodes[index]
                    x, y = graph.nodes[key].coordinates
                    node.set_x(0, x)
                    node.set_x(1, y)
                    node.set_x_lagr(0, x)
                    node.set_x_lagr(1, y)
                row = writer.neck_state()
                writer._set_lagged_rmin(row["R_min"])
                writer.abs_2h_lagged = row["abs_2H"]
                writer.record_accepted_remesh_baseline_for_verification(
                    physical_time=2.0e-12, accepted_steps=2
                )
                self.assertTrue(writer.maybe_remesh(row["R_min"], row["abs_2H"]))
                self.assertEqual(writer.automatic_remesh_state, "armed")
                writer.save_state(state, quiet=True)
                remesh_baseline = writer._rmin_at_remesh

            reader = MappedCoalescenceProblem(_case(), Path(output_dir) / "reader")
            reader.configure_automatic_remeshing(
                Path(output_dir) / "reader-receipts.jsonl"
            )
            with reader:
                reader.initialise()
                reader.load_state(state, quiet=True)
                self.assertEqual(reader.automatic_remesh_state, "armed")
                self.assertEqual(reader._rmin_at_remesh, remesh_baseline)

    def test_mesh_only_runner_claims_exclusive_segment_and_writes_progress(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-attempt-") as directory:
            root = Path(directory)
            attempt = root / "runs" / "attempts" / "attempt-0001"
            (attempt / "base").mkdir(parents=True)
            (attempt / "segments").mkdir()
            (attempt / "runtime").mkdir()
            case = ROOT / "cases" / "anthony2020" / "T5-stokes-R0-1e-06.json"
            runner = ROOT / "runners" / "run_mapped_case.py"
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "schema": "pyoomph-run-manifest-v1",
                "case_id": "T5-stokes-R0-1e-06",
                "case": {"path": str(case.resolve()), "sha256": digest(case)},
                "runner": {"path": str(runner.resolve()), "sha256": digest(runner)},
            }))
            (attempt / "attempt.json").write_text(json.dumps({
                "schema": "pyoomph-attempt-v1",
                "attempt_id": "attempt-0001",
                "manifest": {"path": str(manifest), "sha256": digest(manifest)},
            }))
            argv = [
                "--case", str(case),
                "--attempt-dir", str(attempt),
                "--segment-id", "base",
                "--mesh-only",
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            progress = attempt / "runtime" / "progress.jsonl"
            records = [json.loads(line) for line in progress.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["event"], "segment-start")
            self.assertEqual(records[0]["segment_id"], "base")
            self.assertTrue((attempt / "base" / "case.json").is_file())
            with self.assertRaises(FileExistsError):
                main(argv)

    def test_armed_preparation_failure_disables_state_without_mesh_mutation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-failure-") as output_dir:
            receipts = Path(output_dir) / "receipts.jsonl"
            problem = MappedCoalescenceProblem(_case(), output_dir)
            problem.configure_automatic_remeshing(receipts)
            with problem:
                problem.initialise()
                problem.record_accepted_remesh_baseline_for_verification(
                    physical_time=1.0e-12, accepted_steps=1
                )
                problem.calibrate_automatic_remeshing()
                mesh = problem.get_mesh("drop")
                coordinates = tuple(
                    (float(node.x(0)), float(node.x(1))) for node in mesh.nodes()
                )
                baseline = problem._rmin_at_remesh
                time_before = problem.get_current_time(
                    dimensional=False, as_float=True
                )
                with patch.object(
                    problem,
                    "prepare_mapped_remesh",
                    side_effect=RuntimeError("synthetic preflight failure"),
                ), self.assertRaisesRegex(RuntimeError, "synthetic preflight failure"):
                    problem.maybe_remesh(
                        problem.remesh_factor * baseline,
                        problem.abs_2h_lagged,
                    )
                self.assertEqual(problem.automatic_remesh_state, "failed")
                self.assertEqual(problem._rmin_at_remesh, baseline)
                self.assertIsNone(problem._prepared_mapped_remesh)
                self.assertIsNone(problem.mapped_mesh_template.pending_remesh_graph)
                self.assertEqual(
                    tuple((float(node.x(0)), float(node.x(1))) for node in mesh.nodes()),
                    coordinates,
                )
                self.assertEqual(
                    problem.get_current_time(dimensional=False, as_float=True),
                    time_before,
                )
            records = [json.loads(line) for line in receipts.read_text().splitlines()]
            self.assertEqual([record["passed"] for record in records], [True, False])
            self.assertEqual(records[-1]["kind"], "automatic-rmin-x4")
            self.assertEqual(records[-1]["error_type"], "RuntimeError")

    def test_configured_trigger_overshoot_fails_before_preparation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mapped-run-overshoot-") as output_dir:
            receipts = Path(output_dir) / "receipts.jsonl"
            problem = MappedCoalescenceProblem(_case(), output_dir)
            problem.configure_automatic_remeshing(
                receipts, max_trigger_ratio=4.05
            )
            with problem:
                problem.initialise()
                problem.record_accepted_remesh_baseline_for_verification(
                    physical_time=1.0e-12, accepted_steps=1
                )
                problem.calibrate_automatic_remeshing()
                with patch.object(problem, "prepare_mapped_remesh") as prepare:
                    with self.assertRaisesRegex(RuntimeError, "overshot"):
                        problem.maybe_remesh(4.051 * problem._rmin_at_remesh)
                    prepare.assert_not_called()
                self.assertEqual(problem.automatic_remesh_state, "failed")


if __name__ == "__main__":
    unittest.main()
