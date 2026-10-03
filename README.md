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

The repository follows the CoMPhy simulation layout: reusable code in one package,
case definitions in `simulationCases/`, and evidence grouped by what the result is
compared with.

| Path | Contents |
|---|---|
| `src/coalescence/fem/` | The production solver: pyoomph finite elements on a tip-graded ALE mesh. |
| `src/coalescence/bim/` | An independent axisymmetric boundary-integral Stokes solver (numpy and scipy only), used to verify the finite-element solver in the Stokes limit. |
| `src/coalescence/analysis/` | Solver-agnostic code: neck-history readers, run lists and Stokes-regime theory. |
| `run_fem_stokes_tip.py`, `run_bim_stokes.py` | Runners: one case file in, one output directory out. |
| `simulationCases/anthony2020/` | The Newtonian case matrix of Anthony et al. (2020); see its README. |
| `verificationCases/` | Code against code: the boundary-integral solver against the finite element, and convergence studies. |
| `validationCases/` | Comparison with independent published data. |
| `modelComparisonCases/` | Comparison with asymptotic theory in its stated regime. |
| `postProcess/` | Scripts that rebuild each figure from run outputs. |
| `testCases/` | Software tests. |
| `docs/` | Approved documentation. |

`fem` and `bim` never import each other and share only the case files, so agreement
between them is a test of the numerics of each. Run output is never stored here.
Each evidence case lists its simulations in a `runs.toml` by identifier, commit and
SHA-256 of the neck history; the figure scripts find the outputs through an
untracked `data-roots.toml` (see `src/coalescence/analysis/runs.py`).

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
python run_fem_stokes_tip.py simulationCases/anthony2020/T5-stokes-R0-1e-06.json --out <output> [options]
python run_bim_stokes.py simulationCases/anthony2020/T5-stokes-R0-1e-06.json --out <output> [--newton]
python -m unittest discover -s testCases -t .
```

The runners take the physics (R0, Z0, 1/Oh = 0) from the case file and every numerical
parameter, including the stop radius, from their options; each manifest lists the case
fields used and records the full command line. A restart (`--restart-from`) writes to
a new output directory, and the run list joins the segments with `continues`; a runner
refuses to restart into a directory that already holds a run. The finite-element
runner and its tests need the pinned pyoomph environment; the boundary-integral solver
needs only numpy and scipy.

Case videos: with `--field-frames` the finite-element runner saves the P2 velocity and
pressure fields at the start, at every remesh and at every `--snapshot-dt`. The command
`python postProcess/make_case_video.py <runtime> [<continuation> ...] --out-dir <folder>
--name <stem>` turns the run folders of one case into a video of three successive zooms
centred on the neck tip (the drop pair, the neck, the meniscus), each showing the speed
and the viscous dissipation rate, on a clock that is logarithmic in time early and linear
late. A Stokes run without saved fields is rebuilt from its saved interfaces, since its
velocity is fixed by the geometry; an inertial run needs `--field-frames`.

## Status

The Stokes (La = 0) branch of Anthony et al. (2020) Figure 3 has been computed from
their exact initial bridge, R0 = 10⁻⁶ and Z0 = R0²/2, to R_min = 0.03. It agrees
with the published curve for R_min ≥ 10⁻⁴ but not during the startup transient; see
[`docs/la0-stokes-anthony-fig3.md`](docs/la0-stokes-anthony-fig3.md). The
finite-Ohnesorge branch and the non-Newtonian problems are not yet computed.

## Archived lines

- `experiment/anthony-fig3-f9b408b` (tag `abandoned/anthony-fig3-f9b408b-2026-09-20`):
  structured four-block Q2 chart with a lagged Eq. 6 divider constraint on an
  unreleased pyoomph fork. Closed on 20 September 2026 without an accepted
  trajectory; retained, locked, as the audit trail for its archived
  diagnostics. Do not merge into `main`.
- Tag `archive/structured-mapped-2026-09`: the structured and mapped-Q2
  finite-element problem classes, runners and tests as they stood on `main`
  before the move to `src/coalescence/`, superseded by the tip-graded solver.
