"""One-state Jacobian block audit for the moving-frame Stokes solver.

This is a diagnostic correction, never an accepted physical timestep.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _digest(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


def _backward_error(matrix, solution, rhs) -> float:
    numerator = np.abs(matrix @ solution - rhs)
    denominator = np.abs(matrix) @ np.abs(solution) + np.abs(rhs)
    return float(np.max(numerator / np.maximum(denominator, np.finfo(float).tiny)))


def _solve_serial_block(solver, matrix, rhs) -> np.ndarray:
    """Use the production SuperLU backend directly for a serial CSR submatrix."""
    matrix = matrix.tocsr()
    n = len(rhs)
    if matrix.shape != (n, n):
        raise RuntimeError("block matrix is not square")
    solver._note_external_serial_solve()
    solution = np.ascontiguousarray(rhs, dtype=np.float64).copy()
    solver.solve_serial(1, n, matrix.nnz, 1, matrix.data, matrix.indices, matrix.indptr, solution, 0, 1)
    solver.solve_serial(2, n, matrix.nnz, 1, matrix.data, matrix.indices, matrix.indptr, solution, 0, 1)
    return solution


def _field_indices(problem) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[str]]:
    types, names = problem.get_dof_description()
    resolved = [names[int(kind)].rsplit("/", 1)[-1] if kind >= 0 else "unknown" for kind in types]
    fluid = np.array([i for i, field in enumerate(resolved)
                      if field in {"velocity_x", "velocity_y", "pressure"}], dtype=np.int64)
    other = np.array([i for i, field in enumerate(resolved)
                      if field not in {"velocity_x", "velocity_y", "pressure"}], dtype=np.int64)
    counts = {field: resolved.count(field) for field in sorted(set(resolved))}
    if len(fluid) < 100 or len(other) < 100:
        raise RuntimeError("unexpected Stokes/moving-mesh block partition")
    return fluid, other, counts, resolved


def _max_by_field(values, indices, resolved) -> dict[str, float]:
    result: dict[str, float] = {}
    for value, index in zip(values, indices):
        name = resolved[int(index)]
        result[name] = max(result.get(name, 0.0), abs(float(value)))
    return result


def _argmax_by_field(values, indices, resolved) -> dict[str, int]:
    result: dict[str, tuple[float, int]] = {}
    for value, index in zip(values, indices):
        name = resolved[int(index)]
        candidate = (abs(float(value)), int(index))
        if name not in result or candidate[0] > result[name][0]:
            result[name] = candidate
    return {name: index for name, (_, index) in result.items()}


def _mesh_peak_location(problem, dof: int, axis: int, correction_solver: float) -> dict:
    mesh = problem.get_mesh("drop")
    location, node_type = problem._search_dof_in_mesh(mesh, dof)
    result = {"dof": dof, "node_type": node_type,
              "correction_solver": correction_solver,
              "correction_physical_abs": abs(correction_solver) * problem.S}
    if location is None:
        return result
    x0, y0 = float(location[0]), float(location[1])
    result["frame_coordinate_solver"] = [x0, y0]
    result["physical_coordinate"] = [problem.S * x0 + problem._frame_shift_now(), problem.S * y0]
    nearest = float("inf")
    for element in mesh.elements():
        nodes = [element.node_pt(i) for i in range(element.nnode())]
        if not any(node.variable_position_pt().eqn_number(axis) == dof for node in nodes):
            continue
        for node in nodes:
            gap = math.hypot(node.x(0) - x0, node.x(1) - y0)
            if gap > 1e-30:
                nearest = min(nearest, gap)
    if math.isfinite(nearest):
        result["nearest_incident_node_gap_physical"] = nearest * problem.S
        result["correction_over_nearest_gap"] = abs(correction_solver) / nearest
    return result


def seed_frozen_stokes(problem, output_dir: Path, initial_dt: float) -> dict:
    """Solve only algebraic Stokes fields on the untouched t=0 geometry."""
    if not problem.is_initialised() or initial_dt <= 0:
        raise RuntimeError("Stokes seed requires an initialised problem and positive dt")
    problem.initialise_dt(initial_dt)
    problem.assign_initial_values_impulsive()
    fluid, other, _, resolved = _field_indices(problem)
    U0 = np.asarray(problem.get_current_dofs()[0], dtype=np.float64).copy()
    frozen_hash = _digest(U0[other])
    F0, J0 = problem.assemble_jacobian(with_residual=True)
    F0 = np.asarray(F0, dtype=np.float64)
    A = J0[fluid, :][:, fluid].tocsr()
    df = _solve_serial_block(problem.get_la_solver(), A, -F0[fluid])
    if not np.all(np.isfinite(df)):
        raise RuntimeError("non-finite initial frozen-Stokes correction")
    U1 = U0.copy()
    U1[fluid] += df
    problem.set_current_dofs(U1)
    problem.invalidate_cached_mesh_data()
    F1 = np.asarray(problem.get_residuals(), dtype=np.float64)
    result = {
        "schema": "frozen-stokes-seed-v1", "R0": problem.R0, "Z0": problem.Z0,
        "initial_dt": initial_dt, "ndof": len(U0), "fluid_dofs": len(fluid),
        "residual_before_max": float(np.max(np.abs(F0[fluid]))),
        "residual_after_max": float(np.max(np.abs(F1[fluid]))),
        "componentwise_backward_error": _backward_error(A, df, -F0[fluid]),
        "correction_max_by_field": _max_by_field(df, fluid, resolved),
        "frozen_geometry_and_neck_unchanged": _digest(U1[other]) == frozen_hash,
        "neck_after_seed": problem.neck_state(),
    }
    (output_dir / "seed-stokes.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not result["frozen_geometry_and_neck_unchanged"] or result["residual_after_max"] > 1e-6 * max(result["residual_before_max"], 1):
        raise RuntimeError("initial frozen-Stokes seed failed its algebraic or geometry gate")
    problem.assign_initial_values_impulsive()
    problem.timestepper.set_num_unsteady_steps_done(0)
    problem._taken_already_an_unsteady_step = False
    problem._dt_prev = None
    return result


def audit_one_step(problem, output_dir: Path, audit_dt: float) -> dict:
    """Apply one frozen-Stokes correction, then one complementary correction."""
    if problem._steps != 1 or problem.n_remesh != 0 or audit_dt <= 0:
        raise RuntimeError("block audit requires exactly one accepted, unremeshed step and positive dt")
    before_time = float(problem.get_current_time(dimensional=False, as_float=True))
    # Reproduce the driver's first BDF1 retry from its accepted first state.
    problem._dof_restore(problem._dof_snapshot())
    problem.initialise_dt(audit_dt)
    problem.timestepper.set_num_unsteady_steps_done(0)
    problem._taken_already_an_unsteady_step = True
    if float(problem.get_current_time(dimensional=False, as_float=True)) != before_time:
        raise RuntimeError("audit timestep setup changed physical time")

    fluid, other, counts, resolved = _field_indices(problem)
    U0 = np.asarray(problem.get_current_dofs()[0], dtype=np.float64).copy()
    h1 = _digest(problem.get_history_dofs(1))
    h2 = _digest(problem.get_history_dofs(2))
    F0, J0 = problem.assemble_jacobian(with_residual=True)
    F0 = np.asarray(F0, dtype=np.float64)
    if J0.shape != (len(U0), len(U0)) or len(F0) != len(U0):
        raise RuntimeError("assembled system dimension mismatch")
    solver = problem.get_la_solver()
    result = {
        "schema": "stokes-moving-block-audit-v1",
        "R0": problem.R0,
        "Z0": problem.Z0,
        "audit_dt": audit_dt,
        "physical_time": before_time,
        "neck_before": problem.neck_state(),
        "ndof": len(U0),
        "nnz": int(J0.nnz),
        "field_counts": counts,
        "frozen_state": {"complement_dofs_sha256": _digest(U0[other]),
                         "history_1_sha256": h1, "history_2_sha256": h2},
        "full_initial_residual_max": float(np.max(np.abs(F0))),
    }
    output = output_dir / "block-audit.json"
    try:
        A = J0[fluid, :][:, fluid].tocsr()
        rhs = -F0[fluid]
        df = _solve_serial_block(solver, A, rhs)
        if not np.all(np.isfinite(df)):
            raise RuntimeError("non-finite frozen-Stokes correction")
        U1 = U0.copy()
        U1[fluid] += df
        problem.set_current_dofs(U1)
        problem.invalidate_cached_mesh_data()
        F1 = np.asarray(problem.get_residuals(), dtype=np.float64)
        frozen_unchanged = (_digest(U1[other]) == result["frozen_state"]["complement_dofs_sha256"]
                            and _digest(problem.get_history_dofs(1)) == h1
                            and _digest(problem.get_history_dofs(2)) == h2)
        result["stokes_block"] = {
            "dofs": len(fluid),
            "nnz": int(A.nnz),
            "residual_before_max": float(np.max(np.abs(F0[fluid]))),
            "residual_after_max": float(np.max(np.abs(F1[fluid]))),
            "correction_max": float(np.max(np.abs(df))),
            "correction_max_by_field": _max_by_field(df, fluid, resolved),
            "correction_argmax_dof_by_field": _argmax_by_field(df, fluid, resolved),
            "neck_after_stokes": problem.neck_state(),
            "componentwise_backward_error": _backward_error(A, df, rhs),
            "linear_prediction_mismatch_max": float(np.max(np.abs(F1[fluid] - (F0[fluid] + A @ df)))),
            "complement_dofs_and_histories_unchanged": bool(frozen_unchanged),
        }
        if not frozen_unchanged:
            raise RuntimeError("frozen geometry or time history changed during Stokes correction")
        if result["stokes_block"]["residual_after_max"] > 1e-6 * max(result["stokes_block"]["residual_before_max"], 1):
            result["complement_block"] = {"skipped": "frozen-Stokes residual did not fall sufficiently"}
        else:
            F1, J1 = problem.assemble_jacobian(with_residual=True)
            F1 = np.asarray(F1, dtype=np.float64)
            D = J1[other, :][:, other].tocsr()
            rhs_other = -F1[other]
            try:
                dg = _solve_serial_block(solver, D, rhs_other)
                if not np.all(np.isfinite(dg)):
                    raise RuntimeError("non-finite complementary correction")
                complement = {
                    "dofs": len(other), "nnz": int(D.nnz),
                    "residual_before_max": float(np.max(np.abs(F1[other]))),
                    "correction_max": float(np.max(np.abs(dg))),
                    "correction_max_by_field": _max_by_field(dg, other, resolved),
                    "correction_argmax_dof_by_field": _argmax_by_field(dg, other, resolved),
                    "residual_before_max_by_field": _max_by_field(F1[other], other, resolved),
                    "componentwise_backward_error": _backward_error(D, dg, rhs_other),
                    "direction_samples": [],
                }
                complement["Rn_correction_physical_abs"] = problem.S * complement["correction_max_by_field"].get("R_neck", 0.0)
                complement["mesh_correction_max_physical_abs"] = problem.S * max(
                    complement["correction_max_by_field"].get("mesh_x", 0.0),
                    complement["correction_max_by_field"].get("mesh_y", 0.0),
                )
                complement["mesh_correction_over_tip_element_bound"] = (
                    complement["mesh_correction_max_physical_abs"] / max(result["neck_before"]["h_tip_now"], 1e-300)
                )
                complement["mesh_peak_locations"] = {}
                for field, axis in (("mesh_x", 0), ("mesh_y", 1)):
                    peak_dof = complement["correction_argmax_dof_by_field"].get(field)
                    if peak_dof is not None:
                        local_index = int(np.flatnonzero(other == peak_dof)[0])
                        complement["mesh_peak_locations"][field] = _mesh_peak_location(
                            problem, peak_dof, axis, float(dg[local_index])
                        )
                complement["Rn_correction_over_u_dt_abs"] = (
                    complement["Rn_correction_physical_abs"]
                    / max(abs(result["stokes_block"]["neck_after_stokes"]["u_neck"]) * audit_dt, 1e-300)
                )
                # Directional finite differences use the ACTUAL represented
                # DOF displacement, rather than assuming tiny perturbations
                # survive floating-point storage unchanged.
                max_fraction = min(
                    1e-2,
                    0.01 * result["neck_before"]["h_tip_now"]
                    / max(complement["mesh_correction_max_physical_abs"], 1e-300),
                    0.01 * problem.R0 / max(complement["Rn_correction_physical_abs"], 1e-300),
                )
                directions = {
                    "all_complement": dg,
                    "geometry_and_neck": np.where(
                        np.isin([resolved[int(i)] for i in other], ["mesh_x", "mesh_y", "R_neck"]), dg, 0.0
                    ),
                    "multipliers_only": np.where(
                        np.isin([resolved[int(i)] for i in other], ["mesh_x", "mesh_y", "R_neck"]), 0.0, dg
                    ),
                }
                for direction_name, direction in directions.items():
                    base_fraction = 1e-2 if direction_name == "multipliers_only" else max_fraction
                    for fraction in (base_fraction, base_fraction / 10, base_fraction / 100):
                        sample = {"direction": direction_name, "fraction": fraction}
                        try:
                            plus = U1.copy()
                            plus[other] += fraction * direction
                            problem.set_current_dofs(plus)
                            actual_plus = np.asarray(problem.get_current_dofs()[0], dtype=np.float64).copy()
                            F_plus = np.asarray(problem.get_residuals(), dtype=np.float64)
                            minus = U1.copy()
                            minus[other] -= fraction * direction
                            problem.set_current_dofs(minus)
                            actual_minus = np.asarray(problem.get_current_dofs()[0], dtype=np.float64).copy()
                            F_minus = np.asarray(problem.get_residuals(), dtype=np.float64)
                            represented = actual_plus - actual_minus
                            predicted = J1 @ represented
                            mismatch = F_plus - F_minus - predicted
                            requested_inf = float(np.max(np.abs(2 * fraction * direction)))
                            actual_inf = float(np.max(np.abs(represented[other])))
                            sample.update({
                                "requested_delta_inf": requested_inf,
                                "represented_delta_inf": actual_inf,
                                "represented_over_requested": actual_inf / max(requested_inf, 1e-300),
                                "mismatch_max": float(np.max(np.abs(mismatch))),
                                "predicted_change_max": float(np.max(np.abs(predicted))),
                                "mismatch_by_residual_field_max": _max_by_field(
                                    mismatch, np.arange(len(mismatch)), resolved
                                ),
                                "worst_mismatch_dof": int(np.argmax(np.abs(mismatch))),
                                "worst_mismatch_field": resolved[int(np.argmax(np.abs(mismatch)))],
                            })
                        except Exception as exc:
                            sample["error"] = repr(exc)
                        finally:
                            problem.set_current_dofs(U1)
                        complement["direction_samples"].append(sample)
                result["complement_block"] = complement
            except Exception as exc:
                result["complement_block"] = {"error": repr(exc)}
    except Exception as exc:
        result["audit_error"] = repr(exc)
        raise
    finally:
        problem.set_current_dofs(U0)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
