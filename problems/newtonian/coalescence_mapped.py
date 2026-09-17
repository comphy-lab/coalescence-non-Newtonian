"""Newtonian coalescence on the structured controlled-elliptic mesh.

This problem keeps pyoomph's axisymmetric Taylor--Hood flow and free-surface
weak forms, but replaces the Gmsh/hyperelastic mesh with the explicit
multi-block Q2 template and project-local Winslow grid equation.  Remapping is
deliberately fail-closed until its field/history transfer gate is implemented.
"""

from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pyoomph import DirichletBC, MeshFileOutput, var
from pyoomph.equations.generic import (
    AxisymmetryBC,
    ElementSpace,
    ExtremumObservables,
    IntegralObservables,
)
from pyoomph.equations.navier_stokes import (
    NavierStokesEquations,
    NavierStokesFreeSurface,
    StokesEquations,
)
from pyoomph.meshes.zeta import (
    AssignZetaCoordinatesByArclength,
    AssignZetaCoordinatesByEulerianCoordinate,
)

from .coalescence_axisym import CoalescenceProblem
from .divider_constraint import (
    anthony_divider_constraint,
    axis_divider_corner_condition,
)
from .elliptic_grid import ControlledEllipticMesh
from .evolved_geometry import axisymmetric_twice_mean_curvature_at_start
from .mapped_mesh import AnthonyMappedQuadMesh
from .remap_diagnostics import (
    MeshDiagnostics,
    RemeshBaselineCalibration,
    RemeshGateThresholds,
    calibrate_remesh_baseline,
    compare_remesh,
    diagnose_mesh,
    interface_geometry,
    q2_basis,
    q2_basis_derivative,
)
from .runtime_remesh import (
    PreparedMappedRemesh,
    capture_interface_q2_chain,
    prepare_mapped_transaction,
)


@dataclass(frozen=True)
class MappedTimestepBound:
    bulk_singular_length: float
    interface_segment_scale: float
    neck_quadratic_scale: float
    curvature_scale: float
    selected_length_scale: float
    maximum_fluid_speed: float
    maximum_mesh_speed: float
    maximum_relative_speed: float
    maximum_sampled_speed: float
    advective_dt_cap: float
    selected_dt_cap: float
    coordinate_ulp_floor: float
    speed_source: str = "sampled-fluid-q2"

    def as_dict(self) -> dict[str, Any]:
        return {
            "bulk_singular_length": self.bulk_singular_length,
            "interface_segment_scale": self.interface_segment_scale,
            "neck_quadratic_scale": self.neck_quadratic_scale,
            "curvature_scale": self.curvature_scale,
            "selected_length_scale": self.selected_length_scale,
            "maximum_fluid_speed": self.maximum_fluid_speed,
            "maximum_mesh_speed": self.maximum_mesh_speed,
            "maximum_relative_speed": self.maximum_relative_speed,
            "maximum_sampled_speed": self.maximum_sampled_speed,
            "advective_dt_cap": (
                self.advective_dt_cap
                if math.isfinite(self.advective_dt_cap)
                else None
            ),
            "selected_dt_cap": self.selected_dt_cap,
            "coordinate_ulp_floor": self.coordinate_ulp_floor,
            "speed_source": self.speed_source,
        }


def mapped_completion_reached(
    observable: str,
    threshold: float | int,
    *,
    rmin: float,
    accepted_steps: int,
) -> bool:
    """Evaluate one validated mapped production endpoint."""
    if observable == "R_min":
        return float(rmin) >= float(threshold)
    if observable == "accepted_steps":
        return int(accepted_steps) >= int(threshold)
    raise ValueError(f"unsupported mapped completion observable {observable!r}")


class MappedCoalescenceProblem(CoalescenceProblem):
    """Anthony-style mapped-grid replacement for the Newtonian problem.

    The Eq. 6 ellipse uses ``R_lagged`` from the preceding accepted timestep.
    :meth:`actions_after_transient_solve` is the only hook that advances this
    value, so rejected adaptive retries cannot move the divider.
    """

    def __init__(self, case: dict[str, Any], output_dir: str | Path):
        super().__init__(case, output_dir)
        # Problem.initialise() prints/prepares the output directory before it
        # calls define_problem(), so establish the run-owned path here as well
        # as in define_problem().  This prevents the default ``__main__``
        # directory from appearing in the source checkout during tests.
        self.set_output_directory(str(self.output_root / "pyoomph"))
        mesh = case["mesh"]
        self.cap_resolution = int(mesh.get("cap_resolution", 1))
        self.bridge_normal_resolution = int(mesh.get("bridge_normal_resolution", 2))
        self.R_lagged_parameter = None
        self.mapped_mesh_template: AnthonyMappedQuadMesh | None = None
        self._prepared_mapped_remesh: PreparedMappedRemesh | None = None
        self._automatic_remesh_state = "disabled"
        self._automatic_remesh_receipt_path: Path | None = None
        self._automatic_remesh_sequence = 0
        self._automatic_remesh_events = 0
        self._automatic_remesh_calibration: RemeshBaselineCalibration | None = None
        self._accepted_remesh_snapshot: MeshDiagnostics | None = None
        self._accepted_remesh_marker: dict[str, Any] | None = None
        self._progress_callback: Callable[[dict[str, Any]], None] | None = None
        self._checkpoint_callback: Callable[[dict[str, Any]], None] | None = None
        self._production_maxstep: float | None = None
        self._last_timestep_bound: MappedTimestepBound | None = None
        mesh_policy = case.get("mesh", {})
        self.remesh_factor = float(mesh_policy.get("remesh_factor", 4.0))
        gate_policy = dict(mesh_policy.get("remesh_gates", {}))
        self._remesh_gate_thresholds = RemeshGateThresholds(**{
            name: float(gate_policy[name])
            for name in RemeshGateThresholds.__dataclass_fields__
            if name in gate_policy
        })
        self._calibration_allowances = {
            "relative_divergence_growth": float(
                gate_policy.get("calibration_relative_divergence_growth", 0.25)
            ),
            "relative_divergence_correction": float(
                gate_policy.get("calibration_relative_divergence_correction", 0.25)
            ),
            "relative_flux_change": float(
                gate_policy.get("calibration_relative_flux_change", 0.25)
            ),
        }
        self._automatic_remesh_max_trigger_ratio = 4.05
        # A production mapped run must stop if a diagnostic/output contract
        # fails; the legacy exploratory runner retains its best-effort mode.
        self.strict_output_diagnostics = True

    def define_problem(self) -> None:
        self.set_coordinate_system("axisymmetric")
        self.set_output_directory(str(self.output_root / "pyoomph"))
        tolerances = self.case["tolerances"]
        # The mapped high-curvature residual has a documented discretisation
        # floor near 5e-8; cases may request looser, never tighter, solves.
        self.newton_solver_tolerance = max(self.newton_tol, 5.0e-8)
        self.max_newton_iterations = int(
            tolerances.get("max_newton_iterations", 20)
        )
        self.DTSF_minimum_dt = float(tolerances.get("minimum_dt", 1e-18))
        self.write_states = False

        self.R_lagged_parameter = self.define_global_parameter(R_lagged=self.R0)
        self.mapped_mesh_template = AnthonyMappedQuadMesh(
            self.R0,
            self.Z0,
            cap_resolution=self.cap_resolution,
            bridge_normal_resolution=self.bridge_normal_resolution,
        )
        self.mapped_mesh_template.remesh_rmin_baseline = self._rmin_at_remesh
        self.mapped_mesh_template.remesh_abs2h_baseline = self._abs2h_at_remesh
        self.add_mesh(self.mapped_mesh_template)

        if self.inertia:
            if self.Oh is None:
                raise ValueError("finite-Oh runs require physics.Oh")
            flow = NavierStokesEquations(
                mass_density=1.0,
                dynamic_viscosity=self.Oh,
                mode="TH",
            )
        else:
            flow = StokesEquations(
                dynamic_viscosity=1.0,
                mode="TH",
                mass_density=1.0,
                boussinesq=True,
            )

        equations = MeshFileOutput()
        equations += flow
        equations += ControlledEllipticMesh()
        equations += ElementSpace("C2")
        equations += IntegralObservables(volume=1)
        equations += AxisymmetryBC() @ "axis"
        equations += DirichletBC(mesh_x=0) @ "axis"
        equations += DirichletBC(mesh_y=0, velocity_y=0) @ "plane"
        equations += NavierStokesFreeSurface(surface_tension=1.0) @ "interface"
        equations += anthony_divider_constraint(self.R_lagged_parameter) @ "divider"
        equations += axis_divider_corner_condition(self.R_lagged_parameter) @ "divider/axis"
        equations += ExtremumObservables(r=var("mesh_x")) @ "interface"
        equations += AssignZetaCoordinatesByArclength(
            sort_along_axis="y+"
        ) @ "interface"
        equations += AssignZetaCoordinatesByEulerianCoordinate("y") @ "axis"
        equations += AssignZetaCoordinatesByEulerianCoordinate("x") @ "plane"

        # A prescribed exterior traction supplies the pressure datum.  Adding
        # a pointwise pressure pin here would over-constrain that physical
        # condition; the static-drop verification gate tests this separately.
        self.add_equations(equations @ "drop")

    def _set_lagged_rmin(self, rmin: float) -> None:
        rmin = float(rmin)
        if not math.isfinite(rmin) or rmin <= 0.0:
            raise ValueError("accepted R_min must be positive and finite")
        if self.R_lagged_parameter is None:
            raise RuntimeError("R_lagged parameter is not initialised")
        self.rmin_lagged = rmin
        self.R_lagged_parameter.value = rmin

    def actions_after_transient_solve(self) -> None:
        # Problem dispatches this hook only after an accepted timestep, after
        # temporal rejection/adaptation has settled.
        super().actions_after_transient_solve()
        row = self.neck_state()
        accepted_steps = int(self.timestepper.get_num_unsteady_steps_done()) + 1
        self._set_lagged_rmin(row["R_min"])
        self.abs_2h_lagged = row["abs_2H"]
        if self._automatic_remesh_state in {
            "awaiting-accepted-step", "baseline-ready", "armed"
        }:
            snapshot = diagnose_mesh(self, velocity_slots=range(7))
            self._record_accepted_remesh_baseline(
                snapshot,
                physical_time=float(row["t"]),
                accepted_steps=accepted_steps,
            )
        if self._progress_callback is not None:
            progress = {
                "event": "accepted-step",
                "case_id": self.case_id,
                "physical_time": float(row["t"]),
                "primary_observable": float(row["R_min"]),
                "degrees_of_freedom": int(self.ndof()),
                "timestep": float(self.time_pt().dt(0)),
                "remesh_events": self._automatic_remesh_events,
                "accepted_steps": accepted_steps,
                "R_min": float(row["R_min"]),
                "u_min": float(row["u_min"]),
                "abs_2H": float(row["abs_2H"]),
                "Z_b": float(row["Z_b"]),
                "V_bod": float(row["V_bod"]),
                "remesh_baseline": float(self._rmin_at_remesh),
                "automatic_remesh_state": self._automatic_remesh_state,
            }
            if self._production_maxstep is not None:
                bound = self.live_mapped_timestep_bound(
                    maxstep=self._production_maxstep,
                    rmin=float(row["R_min"]),
                    abs_2h=float(row["abs_2H"]),
                )
                self._last_timestep_bound = bound
                progress["timestep_bound"] = bound.as_dict()
            self._progress_callback(progress)

    def neck_state(self) -> dict[str, float]:
        """Use the exact first Q2 face for mapped neck curvature."""
        data = super().neck_state()
        if self.mapped_mesh_template is None or self.mapped_mesh_template.graph is None:
            raise RuntimeError("mapped mesh graph is unavailable")
        curve = capture_interface_q2_chain(
            self.get_mesh("drop/interface"),
            len(self.mapped_mesh_template.graph.boundary_facets["interface"]),
        )
        data["R_min"] = curve.start[0]
        data["abs_2H"] = abs(axisymmetric_twice_mean_curvature_at_start(curve))
        return data

    def maybe_remesh(self, rmin: float, abs_2h: float | None = None) -> bool:
        if self._automatic_remesh_state == "baseline-ready":
            self.calibrate_automatic_remeshing()
        if rmin < self.remesh_factor * self._rmin_at_remesh:
            return False
        if self._automatic_remesh_state == "armed":
            return self._execute_automatic_remesh(rmin, abs_2h)
        if self._automatic_remesh_state == "awaiting-accepted-step":
            raise RuntimeError(
                "mapped automatic remeshing has no accepted-step calibration baseline"
            )
        if self._automatic_remesh_state == "failed":
            raise RuntimeError("mapped automatic remeshing is disabled after a failed gate")
        raise RuntimeError(
            "mapped-grid fourfold remeshing is not enabled: field and history "
            "transfer must pass the remap verification gate first"
        )

    @property
    def automatic_remesh_state(self) -> str:
        return self._automatic_remesh_state

    def configure_automatic_remeshing(
        self, receipt_path: str | Path, *, max_trigger_ratio: float = 4.05
    ) -> None:
        """Request production remeshing; calibration is still mandatory."""
        if self._automatic_remesh_state != "disabled":
            raise RuntimeError("automatic remeshing has already been configured")
        if self.remesh_factor != 4.0:
            raise RuntimeError("mapped production remeshing requires the R_min x4 rule")
        if not bool(self.case.get("mesh", {}).get("remeshing", False)):
            raise RuntimeError("the selected case does not declare remeshing")
        path = Path(receipt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        max_trigger_ratio = float(max_trigger_ratio)
        if not math.isfinite(max_trigger_ratio) or max_trigger_ratio < self.remesh_factor:
            raise ValueError("maximum remesh trigger ratio must be finite and at least x4")
        self._automatic_remesh_max_trigger_ratio = max_trigger_ratio
        self._automatic_remesh_receipt_path = path
        self._automatic_remesh_state = "awaiting-accepted-step"

    def set_progress_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        self._progress_callback = callback

    def set_checkpoint_callback(
        self, callback: Callable[[dict[str, Any]], None]
    ) -> None:
        self._checkpoint_callback = callback

    def _refresh_automatic_calibration(self, snapshot: MeshDiagnostics) -> None:
        calibration = calibrate_remesh_baseline(
            snapshot, **self._calibration_allowances
        )
        self._accepted_remesh_snapshot = snapshot
        self._automatic_remesh_calibration = calibration
        if self.mapped_mesh_template is not None:
            self.mapped_mesh_template.automatic_remesh_calibration_payload = (
                self._calibration_payload(calibration)
            )
            self.mapped_mesh_template.automatic_remesh_accepted_marker = (
                self._accepted_remesh_marker
            )

    @staticmethod
    def _calibration_payload(
        calibration: RemeshBaselineCalibration,
    ) -> dict[str, Any]:
        return {
            "baseline_divergence_by_slot": [
                [int(slot), float(value)]
                for slot, value in sorted(calibration.baseline_divergence_by_slot.items())
            ],
            "baseline_flux_by_boundary_and_slot": {
                name: [[int(slot), float(value)] for slot, value in sorted(values.items())]
                for name, values in sorted(
                    calibration.baseline_flux_by_boundary_and_slot.items()
                )
            },
            "relative_divergence_growth": calibration.relative_divergence_growth,
            "relative_divergence_correction": calibration.relative_divergence_correction,
            "relative_flux_change": calibration.relative_flux_change,
            "baseline_signature": calibration.baseline_signature,
            "source": calibration.source,
        }

    @staticmethod
    def _calibration_from_payload(
        payload: dict[str, Any],
    ) -> RemeshBaselineCalibration:
        def tuples(value: Any) -> Any:
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value

        return RemeshBaselineCalibration(
            baseline_divergence_by_slot={
                int(slot): float(value)
                for slot, value in payload["baseline_divergence_by_slot"]
            },
            baseline_flux_by_boundary_and_slot={
                name: {int(slot): float(value) for slot, value in values}
                for name, values in payload["baseline_flux_by_boundary_and_slot"].items()
            },
            relative_divergence_growth=float(payload["relative_divergence_growth"]),
            relative_divergence_correction=float(
                payload["relative_divergence_correction"]
            ),
            relative_flux_change=float(payload["relative_flux_change"]),
            baseline_signature=tuples(payload["baseline_signature"]),
            source=str(payload["source"]),
        )

    def _record_accepted_remesh_baseline(
        self,
        snapshot: MeshDiagnostics,
        *,
        physical_time: float,
        accepted_steps: int,
    ) -> None:
        if accepted_steps < 1 or not math.isfinite(physical_time) or physical_time <= 0.0:
            raise RuntimeError("remesh calibration requires at least one accepted physical timestep")
        marker = {
            "physical_time": float(physical_time),
            "accepted_steps": int(accepted_steps),
            "R_min": float(self.rmin_lagged),
            "abs_2H": float(self.abs_2h_lagged),
        }
        self._accepted_remesh_marker = marker
        self._refresh_automatic_calibration(snapshot)
        if self._automatic_remesh_state != "armed":
            self._automatic_remesh_state = "baseline-ready"
        if self.mapped_mesh_template is not None:
            self.mapped_mesh_template.automatic_remesh_calibration_payload = (
                self._calibration_payload(self._automatic_remesh_calibration)
            )
            self.mapped_mesh_template.automatic_remesh_accepted_marker = marker

    def record_accepted_remesh_baseline_for_verification(
        self,
        *,
        physical_time: float,
        accepted_steps: int,
    ) -> None:
        """Inject an accepted-step marker in tests without advancing time."""
        if not self.is_initialised():
            raise RuntimeError("verification baseline requires an initialised problem")
        self._record_accepted_remesh_baseline(
            diagnose_mesh(self, velocity_slots=range(7)),
            physical_time=physical_time,
            accepted_steps=accepted_steps,
        )

    def _emit_remesh_receipt(self, payload: dict[str, Any]) -> None:
        path = self._automatic_remesh_receipt_path
        if path is None:
            raise RuntimeError("automatic-remesh receipt path is not configured")
        record = {
            "schema": "mapped-remesh-receipt-v1",
            "case_id": self.case_id,
            "sequence": self._automatic_remesh_sequence,
            **payload,
        }
        encoded = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._automatic_remesh_sequence += 1

    def calibrate_automatic_remeshing(self) -> dict[str, Any]:
        """Arm x4 remeshing only after a live identity transaction passes."""
        if self._automatic_remesh_state != "baseline-ready":
            raise RuntimeError("automatic remeshing has no accepted-step baseline to verify")
        if not self.is_initialised():
            raise RuntimeError("automatic remesh calibration requires an initialised problem")
        if self._accepted_remesh_snapshot is None or self._automatic_remesh_calibration is None:
            raise RuntimeError("accepted-step calibration snapshot is missing")
        self._automatic_remesh_state = "calibrating"
        time_before = self.get_current_time(dimensional=False, as_float=True)
        baseline_before = self._rmin_at_remesh
        abs2h_baseline_before = self._abs2h_at_remesh
        try:
            before = diagnose_mesh(self, velocity_slots=range(7))
            if not self._automatic_remesh_calibration.matches(before):
                raise RuntimeError("accepted-step calibration does not match the live mesh")
            transaction = self.prepare_mapped_remesh(
                self.rmin_lagged, abs_2h=self.abs_2h_lagged
            )
            self.execute_prepared_remesh()
            after = diagnose_mesh(self, velocity_slots=range(7))
            diagnostics = compare_remesh(
                before,
                after,
                calibration=self._automatic_remesh_calibration,
                thresholds=self._remesh_gate_thresholds,
            )
            payload = {
                "kind": "accepted-state-identity-calibration",
                "passed": diagnostics.passed,
                "physical_time_before": time_before,
                "physical_time_after": self.get_current_time(
                    dimensional=False, as_float=True
                ),
                "baseline_before": baseline_before,
                "baseline_after": self._rmin_at_remesh,
                "accepted_step_marker": self._accepted_remesh_marker,
                "destination_nodes": transaction.destination_count,
                "missing_locations": transaction.missing_count,
                "fallback_locations": transaction.fallback_count,
                "diagnostics": diagnostics.as_dict(),
            }
            self._emit_remesh_receipt(payload)
            if not diagnostics.passed or payload["physical_time_after"] != time_before:
                self._automatic_remesh_state = "failed"
                raise RuntimeError("live mapped-remesh calibration failed")
            # Calibration is an identity verification, not a new x4 epoch.
            self._rmin_at_remesh = baseline_before
            self._abs2h_at_remesh = abs2h_baseline_before
            if self.mapped_mesh_template is not None:
                self.mapped_mesh_template.remesh_rmin_baseline = baseline_before
                self.mapped_mesh_template.remesh_abs2h_baseline = abs2h_baseline_before
            self._refresh_automatic_calibration(after)
            self._automatic_remesh_state = "armed"
            if self._checkpoint_callback is not None:
                self._checkpoint_callback({
                    "kind": "accepted-state-calibration",
                    "physical_time": payload["physical_time_after"],
                })
            return payload
        except Exception as exc:
            if self._automatic_remesh_state != "failed":
                self._automatic_remesh_state = "failed"
                self._emit_remesh_receipt({
                    "kind": "accepted-state-identity-calibration",
                    "passed": False,
                    "physical_time_before": time_before,
                    "baseline_before": baseline_before,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            raise

    def _execute_automatic_remesh(
        self, rmin: float, abs_2h: float | None
    ) -> bool:
        if self._in_remesh:
            return False
        previous_baseline = self._rmin_at_remesh
        self._in_remesh = True
        self._automatic_remesh_state = "remeshing"
        try:
            before = diagnose_mesh(self, velocity_slots=range(7))
            trigger_ratio = rmin / previous_baseline
            if trigger_ratio > self._automatic_remesh_max_trigger_ratio:
                raise RuntimeError(
                    "accepted step overshot the configured remesh trigger: "
                    f"ratio={trigger_ratio:.8g} > {self._automatic_remesh_max_trigger_ratio:.8g}"
                )
            if (
                self._automatic_remesh_calibration is None
                or not self._automatic_remesh_calibration.matches(before)
            ):
                raise RuntimeError("automatic remesh baseline is missing or mismatched")
            transaction = self.prepare_mapped_remesh(
                rmin, abs_2h=abs_2h
            )
            self.execute_prepared_remesh()
            after = diagnose_mesh(self, velocity_slots=range(7))
            diagnostics = compare_remesh(
                before,
                after,
                calibration=self._automatic_remesh_calibration,
                thresholds=self._remesh_gate_thresholds,
            )
            payload = {
                "kind": "automatic-rmin-x4",
                "passed": diagnostics.passed,
                "physical_time": self.get_current_time(
                    dimensional=False, as_float=True
                ),
                "previous_baseline": previous_baseline,
                "trigger_ratio": trigger_ratio,
                "maximum_trigger_ratio": self._automatic_remesh_max_trigger_ratio,
                "accepted_rmin": rmin,
                "accepted_abs_2h": abs_2h,
                "destination_nodes": transaction.destination_count,
                "missing_locations": transaction.missing_count,
                "fallback_locations": transaction.fallback_count,
                "diagnostics": diagnostics.as_dict(),
            }
            self._emit_remesh_receipt(payload)
            if not diagnostics.passed:
                self._automatic_remesh_state = "failed"
                raise RuntimeError("automatic mapped remesh failed its diagnostics gates")
            self._refresh_automatic_calibration(after)
            self._automatic_remesh_state = "armed"
            self._automatic_remesh_events += 1
            if self.mapped_mesh_template is not None:
                self.mapped_mesh_template.automatic_remesh_events = (
                    self._automatic_remesh_events
                )
            self._dump_fields = True
            if self._checkpoint_callback is not None:
                self._checkpoint_callback({
                    "kind": "automatic-remesh",
                    "physical_time": payload["physical_time"],
                    "accepted_rmin": rmin,
                    "trigger_ratio": trigger_ratio,
                })
            return True
        except Exception as exc:
            if self._automatic_remesh_state != "failed":
                self._automatic_remesh_state = "failed"
                self._emit_remesh_receipt({
                    "kind": "automatic-rmin-x4",
                    "passed": False,
                    "physical_time": self.get_current_time(
                        dimensional=False, as_float=True
                    ),
                    "previous_baseline": previous_baseline,
                    "attempted_rmin": rmin,
                    "attempted_abs_2h": abs_2h,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            raise
        finally:
            self._in_remesh = False


    def prepare_mapped_remesh(
        self,
        rmin: float,
        *,
        cap_resolution: int | None = None,
        bridge_normal_resolution: int | None = None,
        abs_2h: float | None = None,
    ) -> PreparedMappedRemesh:
        """Freeze a complete serial remesh transaction without executing it."""
        if self._prepared_mapped_remesh is not None:
            raise RuntimeError("a mapped remesh transaction is already pending")
        if self.mapped_mesh_template is None or not self.is_initialised():
            raise RuntimeError("the mapped problem must be initialised before preparing a remesh")
        transaction = prepare_mapped_transaction(
            self.mapped_mesh_template,
            self.get_mesh("drop"),
            self.get_mesh("drop/interface"),
            rmin,
            cap_resolution=(
                self.cap_resolution if cap_resolution is None else cap_resolution
            ),
            bridge_normal_resolution=(
                self.bridge_normal_resolution
                if bridge_normal_resolution is None
                else bridge_normal_resolution
            ),
            accepted_abs2h=self.abs_2h_lagged if abs_2h is None else abs_2h,
        )
        self._prepared_mapped_remesh = transaction
        return transaction

    def execute_prepared_remesh(self) -> bool:
        """Verification-only entry point; public remeshing remains fail-closed."""
        transaction = self._prepared_mapped_remesh
        if transaction is None or self.mapped_mesh_template is None:
            raise RuntimeError("no mapped remesh transaction is prepared")
        result = self.force_remesh(
            only_domains={self.mapped_mesh_template},
            num_adapt=0,
            interpolator=transaction.interpolator_factory(),
        )
        if result is False:
            raise RuntimeError("pyoomph declined the explicitly prepared mapped remesh")
        # A no-op adaptive pass establishes pyoomph's canonical element-walk
        # node order.  Initialisation and state loading already take this
        # path, whereas force_remesh(num_adapt=0) otherwise leaves a different
        # equation permutation for the identical mesh.
        mesh = self.get_mesh("drop")
        if mesh._reorder_nodes_if_needed():
            transaction._record_transferred_state(mesh)
            with self.custom_adapt(True):
                pass
            transaction.restore_after_remesh_hooks(mesh)
        # load_state recomputes these weights after restoring the stored dts.
        # Keep the live post-remesh problem in the same assembly state before
        # it can be saved and compared with its reader.
        self.timestepper.set_weights()
        return True

    def actions_after_remeshing(self) -> None:
        # Clear only after pyoomph has completed interpolation, mapping removal,
        # redistribution and the base post-remesh hooks successfully.
        super().actions_after_remeshing()
        transaction = self._prepared_mapped_remesh
        if transaction is not None:
            transaction.restore_after_remesh_hooks(self.get_mesh("drop"))
            transaction.completed = True
            if self.mapped_mesh_template is not None:
                self.mapped_mesh_template.clear_remesh_graph()
            self.cap_resolution = transaction.new_graph.cap_resolution
            self.bridge_normal_resolution = transaction.new_graph.bridge_normal_resolution
            self._rmin_at_remesh = transaction.accepted_rmin
            if transaction.accepted_abs2h is not None:
                self._abs2h_at_remesh = transaction.accepted_abs2h
            if self.mapped_mesh_template is not None:
                self.mapped_mesh_template.remesh_rmin_baseline = self._rmin_at_remesh
                self.mapped_mesh_template.remesh_abs2h_baseline = self._abs2h_at_remesh
            self._prepared_mapped_remesh = None
        elif self.mapped_mesh_template is not None:
            # Loading a state may replace the constructor's graph with the
            # graph embedded in the file.  Adopt the effective template so a
            # subsequent mapped remesh operates on that restored epoch.
            effective = self.mapped_mesh_template._get_template()
            if effective is not self.mapped_mesh_template:
                for index, template in enumerate(self._meshtemplate_list):
                    if template is self.mapped_mesh_template:
                        self._meshtemplate_list[index] = effective
                        break
                self.mapped_mesh_template = effective
            if self.mapped_mesh_template.graph is not None:
                self.cap_resolution = self.mapped_mesh_template.graph.cap_resolution
                self.bridge_normal_resolution = (
                    self.mapped_mesh_template.graph.bridge_normal_resolution
                )
                self._rmin_at_remesh = float(
                    self.mapped_mesh_template.remesh_rmin_baseline
                )
                if self.mapped_mesh_template.remesh_abs2h_baseline is not None:
                    self._abs2h_at_remesh = float(
                        self.mapped_mesh_template.remesh_abs2h_baseline
                    )
            self.mapped_mesh_template.restore_restart_slot_one(
                self.get_mesh("drop")
            )
            if self._automatic_remesh_state == "awaiting-accepted-step":
                calibration_payload = (
                    self.mapped_mesh_template.automatic_remesh_calibration_payload
                )
                marker = self.mapped_mesh_template.automatic_remesh_accepted_marker
                if (calibration_payload is None) != (marker is None):
                    self._automatic_remesh_state = "failed"
                    raise RuntimeError("restart has an incomplete automatic-remesh baseline")
                if calibration_payload is not None and marker is not None:
                    calibration = self._calibration_from_payload(calibration_payload)
                    snapshot = diagnose_mesh(self, velocity_slots=range(7))
                    if not calibration.matches(snapshot):
                        self._automatic_remesh_state = "failed"
                        raise RuntimeError(
                            "restart automatic-remesh baseline does not match restored state"
                        )
                    if int(marker.get("accepted_steps", 0)) < 1:
                        self._automatic_remesh_state = "failed"
                        raise RuntimeError("restart remesh baseline lacks an accepted-step marker")
                    self._automatic_remesh_calibration = calibration
                    self._accepted_remesh_snapshot = snapshot
                    self._accepted_remesh_marker = dict(marker)
                    self._automatic_remesh_events = int(
                        self.mapped_mesh_template.automatic_remesh_events
                    )
                    self._automatic_remesh_state = "armed"
        if self.R_lagged_parameter is not None:
            self.rmin_lagged = float(self.R_lagged_parameter.value)

    def live_mapped_timestep_bound(
        self,
        *,
        maxstep: float,
        rmin: float | None = None,
        abs_2h: float | None = None,
        curvature_factor: float = 4.0,
        courant_factor: float = 0.4,
    ) -> MappedTimestepBound:
        """Sample a physical live-mesh length/speed timestep bound.

        pyoomph does not expose an ALE-relative velocity field at arbitrary
        points here, so the conservative available fallback is the maximum
        sampled Q2 fluid-velocity magnitude. No dimensional hard floor is
        applied; the ULP scale is used only to reject unresolved geometry.
        """
        maxstep = float(maxstep)
        rmin = self.neck_state()["R_min"] if rmin is None else float(rmin)
        abs_2h = self.neck_state()["abs_2H"] if abs_2h is None else float(abs_2h)
        if not all(math.isfinite(value) and value > 0.0 for value in (
            maxstep, rmin, abs_2h, curvature_factor, courant_factor
        )):
            raise ValueError("mapped timestep inputs must be positive and finite")

        mesh = self.get_mesh("drop")
        fields = mesh.get_nodal_field_indices()
        velocity_indices = (fields["velocity_x"], fields["velocity_y"])
        samples = (-1.0, -0.5, 0.0, 0.5, 1.0)
        minimum_singular = math.inf
        minimum_singular_floor = 0.0
        maximum_fluid_speed = 0.0
        maximum_mesh_speed = 0.0
        maximum_relative_speed = 0.0
        current_time = self.get_current_time(dimensional=False, as_float=True)
        previous_dt = float(self.time_pt().dt(0))
        ale_speed_available = (
            current_time > 0.0 and math.isfinite(previous_dt) and previous_dt > 0.0
        )
        for element in mesh.elements():
            if int(element.nnode()) != 9:
                raise RuntimeError("mapped timestep sampling requires Q2 bulk elements")
            nodes = tuple(element.node_pt(index) for index in range(9))
            points = tuple((float(node.x(0)), float(node.x(1))) for node in nodes)
            element_ulp_floor = 64.0 * max(
                math.ulp(value) for point in points for value in point
            )
            velocities = tuple(
                (
                    float(node.value(velocity_indices[0])),
                    float(node.value(velocity_indices[1])),
                )
                for node in nodes
            )
            mesh_velocities = (
                tuple(
                    (
                        (float(node.x_at_t(0, 0)) - float(node.x_at_t(1, 0)))
                        / previous_dt,
                        (float(node.x_at_t(0, 1)) - float(node.x_at_t(1, 1)))
                        / previous_dt,
                    )
                    for node in nodes
                )
                if ale_speed_available
                else ((0.0, 0.0),) * len(nodes)
            )
            for xi in samples:
                lx, dlx = q2_basis(xi), q2_basis_derivative(xi)
                for eta in samples:
                    ly, dly = q2_basis(eta), q2_basis_derivative(eta)
                    r_xi = r_eta = z_xi = z_eta = 0.0
                    velocity_r = velocity_z = 0.0
                    mesh_velocity_r = mesh_velocity_z = 0.0
                    for j in range(3):
                        for i in range(3):
                            index = 3 * j + i
                            r, z = points[index]
                            weight = lx[i] * ly[j]
                            r_xi += dlx[i] * ly[j] * r
                            r_eta += lx[i] * dly[j] * r
                            z_xi += dlx[i] * ly[j] * z
                            z_eta += lx[i] * dly[j] * z
                            velocity_r += weight * velocities[index][0]
                            velocity_z += weight * velocities[index][1]
                            mesh_velocity_r += weight * mesh_velocities[index][0]
                            mesh_velocity_z += weight * mesh_velocities[index][1]
                    determinant = r_xi * z_eta - r_eta * z_xi
                    sum_norm = math.hypot(r_xi + z_eta, z_xi - r_eta)
                    difference_norm = math.hypot(r_xi - z_eta, z_xi + r_eta)
                    sigma_max = 0.5 * (sum_norm + difference_norm)
                    sigma_min = (
                        abs(determinant) / sigma_max if sigma_max > 0.0 else 0.0
                    )
                    if sigma_min <= element_ulp_floor:
                        raise RuntimeError(
                            "mapped bulk singular length is below its element-local ULP scale"
                        )
                    if sigma_min < minimum_singular:
                        minimum_singular = sigma_min
                        minimum_singular_floor = element_ulp_floor
                    maximum_fluid_speed = max(
                        maximum_fluid_speed, math.hypot(velocity_r, velocity_z)
                    )
                    if ale_speed_available:
                        maximum_mesh_speed = max(
                            maximum_mesh_speed,
                            math.hypot(mesh_velocity_r, mesh_velocity_z),
                        )
                        maximum_relative_speed = max(
                            maximum_relative_speed,
                            math.hypot(
                                velocity_r - mesh_velocity_r,
                                velocity_z - mesh_velocity_z,
                            ),
                        )

        interface = interface_geometry(self.get_mesh("drop/interface"))
        interface_scale = math.inf
        interface_ulp_floor = 0.0
        for segment in interface.segments:
            segment_scale = segment.scale
            segment_floor = 64.0 * max(
                math.ulp(value)
                for point in (segment.start, segment.midpoint, segment.end)
                for value in point
            )
            if segment_scale <= segment_floor:
                raise RuntimeError(
                    "mapped interface scale is below its segment-local ULP scale"
                )
            if segment_scale < interface_scale:
                interface_scale = segment_scale
                interface_ulp_floor = segment_floor
        quadratic_scale = rmin * rmin / 16.0
        curvature_scale = curvature_factor / abs_2h
        components = (
            minimum_singular, interface_scale, quadratic_scale, curvature_scale
        )
        component_floors = (
            minimum_singular_floor,
            interface_ulp_floor,
            64.0 * max(
                math.ulp(rmin), math.ulp(rmin * rmin), math.ulp(quadratic_scale)
            ),
            64.0 * max(math.ulp(rmin), math.ulp(curvature_scale)),
        )
        if any(
            not math.isfinite(value) or value <= floor
            for value, floor in zip(components, component_floors)
        ):
            raise RuntimeError(
                "mapped timestep scale is non-finite or below its local ULP scale"
            )
        selected_index = min(range(len(components)), key=components.__getitem__)
        selected_length = components[selected_index]
        ulp_floor = component_floors[selected_index]
        maximum_speed = (
            max(
                maximum_fluid_speed,
                maximum_mesh_speed,
                maximum_relative_speed,
            )
            if ale_speed_available
            else maximum_fluid_speed
        )
        advective_cap = (
            math.inf
            if maximum_speed == 0.0
            else courant_factor * selected_length / maximum_speed
        )
        return MappedTimestepBound(
            minimum_singular,
            interface_scale,
            quadratic_scale,
            curvature_scale,
            selected_length,
            maximum_fluid_speed,
            maximum_mesh_speed,
            maximum_relative_speed,
            maximum_speed,
            advective_cap,
            min(maxstep, advective_cap),
            ulp_floor,
            (
                "sampled-max-fluid-mesh-relative-q2"
                if ale_speed_available
                else "sampled-fluid-q2"
            ),
        )

    def run_until(
        self,
        *,
        max_time: float,
        startstep: float,
        maxstep: float,
        completion_observable: str = "R_min",
        completion_threshold: float | int | None = None,
    ) -> None:
        """Advance one accepted step at a time and service remeshes safely."""
        restored_entry = self.is_initialised()
        if not restored_entry:
            self.initialise()
        row = self.neck_state()
        self._set_lagged_rmin(row["R_min"])
        self.abs_2h_lagged = row["abs_2H"]
        self.write_neck_row(profile=True)
        time = float(row["t"])
        if completion_threshold is None:
            completion_threshold = self.stop_rmin
        accepted_steps = int(self.timestepper.get_num_unsteady_steps_done())
        current_step = float(startstep)
        self._production_maxstep = float(maxstep)
        initial_bound = self.live_mapped_timestep_bound(
            maxstep=maxstep,
            rmin=row["R_min"],
            abs_2h=row["abs_2H"],
        )
        self._last_timestep_bound = initial_bound
        if self._progress_callback is not None:
            self._progress_callback({
                "event": (
                    "restored-segment-state" if restored_entry else "initial-state"
                ),
                "case_id": self.case_id,
                "physical_time": time,
                "R_min": float(row["R_min"]),
                "u_min": float(row["u_min"]),
                "Z_b": float(row["Z_b"]),
                "abs_2H": float(row["abs_2H"]),
                "V_bod": float(row["V_bod"]),
                "degrees_of_freedom": int(self.ndof()),
                "remesh_baseline": float(self._rmin_at_remesh),
                "automatic_remesh_state": self._automatic_remesh_state,
                "timestep_bound": initial_bound.as_dict(),
            })
        while (
            not mapped_completion_reached(
                completion_observable,
                completion_threshold,
                rmin=row["R_min"],
                accepted_steps=accepted_steps,
            )
            and time < max_time
        ):
            bound = self.live_mapped_timestep_bound(
                maxstep=maxstep,
                rmin=row["R_min"],
                abs_2h=row["abs_2H"],
            )
            self._last_timestep_bound = bound
            step_cap = bound.selected_dt_cap
            timestep = min(current_step, step_cap, max_time - time)
            if timestep <= 0.0 or not math.isfinite(timestep):
                raise RuntimeError("mapped live timestep bound is not positive and finite")
            suggested = self.solve(
                timestep=timestep,
                temporal_error=self.temporal_error,
                do_not_set_IC=True,
            )
            time = float(self.get_current_time())
            accepted_steps = int(self.timestepper.get_num_unsteady_steps_done())
            row = self.neck_state()
            self.maybe_remesh(row["R_min"], row["abs_2H"])
            self.output("timestep")
            current_step = min(
                step_cap,
                float(suggested),
            )
        completed = mapped_completion_reached(
            completion_observable,
            completion_threshold,
            rmin=row["R_min"],
            accepted_steps=accepted_steps,
        )
        endpoint = completion_observable if completed else "max_time"
        endpoint_record = {
            "event": "endpoint",
            "case_id": self.case_id,
            "endpoint": endpoint,
            "physical_time": time,
            "R_min": float(row["R_min"]),
            "accepted_steps": accepted_steps,
            "completion_observable": completion_observable,
            "completion_threshold": completion_threshold,
            "max_time": float(max_time),
        }
        if self._progress_callback is not None:
            self._progress_callback(endpoint_record)
        self.output("end")
        if self._checkpoint_callback is not None:
            self._checkpoint_callback({
                "kind": f"endpoint-{endpoint}",
                "physical_time": time,
            })
        if not completed:
            raise RuntimeError(
                "mapped run reached max_time before its completion threshold"
            )


__all__ = [
    "MappedCoalescenceProblem",
    "MappedTimestepBound",
    "mapped_completion_reached",
]
