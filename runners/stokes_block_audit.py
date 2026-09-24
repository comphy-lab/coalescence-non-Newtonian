"""One-state Jacobian block audit for the moving-frame Stokes solver.

This is a diagnostic correction, never an accepted physical timestep.
"""

from __future__ import annotations

import hashlib
import json
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


def _field_indices(problem) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    types, names = problem.get_dof_description()
    resolved = [names[int(kind)].rsplit("/", 1)[-1] if kind >= 0 else "unknown" for kind in types]
    fluid = np.array([i for i, field in enumerate(resolved)
                      if field in {"velocity_x", "velocity_y", "pressure"}], dtype=np.int64)
    other = np.array([i for i, field in enumerate(resolved)
                      if field not in {"velocity_x", "velocity_y", "pressure"}], dtype=np.int64)
    counts = {field: resolved.count(field) for field in sorted(set(resolved))}
    if len(fluid) < 100 or len(other) < 100:
        raise RuntimeError("unexpected Stokes/moving-mesh block partition")
    return fluid, other, counts


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

    fluid, other, counts = _field_indices(problem)
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
                U2 = U1.copy()
                U2[other] += dg
                problem.set_current_dofs(U2)
                F2 = np.asarray(problem.get_residuals(), dtype=np.float64)
                result["complement_block"] = {
                    "dofs": len(other), "nnz": int(D.nnz),
                    "residual_before_max": float(np.max(np.abs(F1[other]))),
                    "residual_after_max": float(np.max(np.abs(F2[other]))),
                    "fluid_residual_after_max": float(np.max(np.abs(F2[fluid]))),
                    "correction_max": float(np.max(np.abs(dg))),
                    "componentwise_backward_error": _backward_error(D, dg, rhs_other),
                    "linear_prediction_mismatch_max": float(np.max(np.abs(F2[other] - (F1[other] + D @ dg)))),
                }
            except Exception as exc:
                result["complement_block"] = {"error": repr(exc)}
    except Exception as exc:
        result["audit_error"] = repr(exc)
        raise
    finally:
        problem.set_current_dofs(U0)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
