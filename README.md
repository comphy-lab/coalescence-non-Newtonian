# Coalescence in non-Newtonian media

Sharp-interface, arbitrary Lagrangian–Eulerian finite-element simulations of
drop coalescence, built on [pyoomph](https://github.com/pyoomph/pyoomph), for
Newtonian, viscoelastic and yield-stress liquids.

## Problem

When two drops touch, a microscopic bridge forms and grows under a diverging
capillary pressure. For Newtonian drops in a passive gas the regimes of that
growth are settled: coalescence begins in a Stokes regime and, at low Ohnesorge
number, crosses over to an inviscid regime at a bridge radius that scales
linearly with Oh (Anthony, Harris & Basaran, *Phys. Rev. Fluids* **5**, 033608,
2020). Once the liquid carries elasticity, a yield stress, or both, the neck
law, the regime map and the existence of a fully coalesced final state are
open questions. This repository holds the solver and the case record used to
study them.

The configuration is a proper free surface: the exterior has zero density and
zero viscosity and exerts only a constant pressure, so the interior rheology
is the only physics beyond the Newtonian reference.

## Layout

| Path | Contents |
|---|---|
| `src/coalescence/fem/` | The production solver: pyoomph finite elements on a tip-graded ALE mesh. |
| `src/coalescence/bim/` | An independent axisymmetric boundary-integral Stokes solver (numpy and scipy only). |
| `src/coalescence/analysis/` | Solver-agnostic code: neck-history readers, contact time, run lists and Stokes-regime theory. |
| `simulationCases/` | The coalescence entry point `run_coalescence.py` and its default case `coalescence.json`. |
| `validationCases/anthony2020/` | Comparison with Anthony, Harris & Basaran (2020), Figure 3: the two cases, their run scripts, the digitised published data and the comparison plot. |
| `verificationCases/bim-stokes/` | The boundary-integral runner, an independent solution of the Stokes case. |
| `verificationCases/drop-oscillation/` | The inertial solver against the exact linear modes of a viscous drop. |
| `postProcess/` | Analysis, figure and case-video scripts that work on run outputs. |
| `testCases/` | Software tests. |

`fem` and `bim` never import each other and share only the case files, so agreement
between them is a test of the numerics of each. Simulation output stays outside this
repository. A published figure states the solver commit and the pinned pyoomph version
that produced it; reproducing it means rerunning the simulation with that commit and
version.

## Method

**Finite element (`fem`).** pyoomph Taylor–Hood triangles, a mesh moved by Laplace
smoothing, the kinematic condition imposed through a Lagrange multiplier, and
second-order backward differentiation in time with Newton iteration. The mesh is
generated in coordinates magnified about the neck tip and graded geometrically away
from it, the radial coordinate is measured from the neck, and the mesh is rebuilt as
the neck grows. By symmetry one quadrant of one drop is computed. The pyoomph
version is pinned in `pyproject.toml` and `uv.lock`.

**Boundary integral (`bim`).** In Stokes flow with a passive exterior the velocity
on the free surface follows from the capillary traction alone, through the
single- and double-layer integrals of the free-space Stokeslet and stresslet. In
axisymmetry the azimuthal integration is analytical, leaving integrals along the
meridian with complete elliptic integrals; one drop is represented together with the
symmetry plane inside the neck. The meridian is a cubic spline with nodes graded
towards the neck tip and redistributed at every step, the equations are collocated
at the nodes, and the global force balance on the drop is imposed on the discrete
solution. The interface moves with the normal velocity, either by a linearly
implicit step or by backward Euler solved with a Jacobian-free Newton–Krylov method.

## Running

```bash
python simulationCases/run_coalescence.py simulationCases/coalescence.json --out <output> [options]
validationCases/anthony2020/run_stokes.sh <output>
validationCases/anthony2020/run_oh0.6.sh <output>
python verificationCases/bim-stokes/run_bim_stokes.py validationCases/anthony2020/T5-stokes-R0-1e-06.json --out <output> [--newton]
python -m unittest discover -s testCases -t .
```

The runners take the physics (Oh or Stokes, R0, Z0) from the case file and every numerical
parameter, including the stop radius, from their options; each manifest lists the case
fields used and records the full command line. A restart (`--restart-from`) writes to
a new output directory; a runner refuses to restart into a directory that already holds
a run. The finite-element runner and its tests need the pinned pyoomph environment; the
boundary-integral solver needs only numpy and scipy.

Case videos: with `--field-frames` the finite-element runner saves the P2 velocity and
pressure fields at the start, at every remesh and at every `--snapshot-dt`. The command
`python postProcess/make_case_video.py <runtime> [<continuation> ...] --out-dir <folder>
--name <stem>` turns the run folders of one case into a video of three successive zooms
centred on the neck tip (the drop pair, the neck, the meniscus), each showing the speed
and the viscous dissipation rate, on a clock that is logarithmic in time early and linear
late. A Stokes run without saved fields is rebuilt from its saved interfaces, since its
velocity is fixed by the geometry; an inertial run needs `--field-frames`.
