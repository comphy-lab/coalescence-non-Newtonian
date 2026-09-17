"""Newtonian coalescence on the structured controlled-elliptic mesh.

This problem keeps pyoomph's axisymmetric Taylor--Hood flow and free-surface
weak forms, but replaces the Gmsh/hyperelastic mesh with the explicit
multi-block Q2 template and project-local Winslow grid equation.  Remapping is
deliberately fail-closed until its field/history transfer gate is implemented.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

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
from .mapped_mesh import AnthonyMappedQuadMesh
from .runtime_remesh import PreparedMappedRemesh, prepare_mapped_transaction


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

    def define_problem(self) -> None:
        self.set_coordinate_system("axisymmetric")
        self.set_output_directory(str(self.output_root / "pyoomph"))
        self.newton_solver_tolerance = max(self.newton_tol, 5e-8)
        self.max_newton_iterations = 20
        self.DTSF_minimum_dt = 1e-18
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
        self._set_lagged_rmin(self.neck_state()["R_min"])

    def maybe_remesh(self, rmin: float, abs_2h: float | None = None) -> bool:
        if rmin < self.remesh_factor * self._rmin_at_remesh:
            return False
        raise RuntimeError(
            "mapped-grid fourfold remeshing is not enabled: field and history "
            "transfer must pass the remap verification gate first"
        )

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
        if self.R_lagged_parameter is not None:
            self.rmin_lagged = float(self.R_lagged_parameter.value)


__all__ = ["MappedCoalescenceProblem"]
