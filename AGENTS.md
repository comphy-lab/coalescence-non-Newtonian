# AGENTS.md

Operating notes for the `coalescence-non-Newtonian` repository. Read
`README.md` first for the problem and the layout.

## Scientific rules

- The Newtonian reference comes first. No non-Newtonian result is reported
  until the regimes of Anthony, Harris & Basaran (2020) and the crossover
  \(R_c(\mathrm{Oh})\) are reproduced with a stated initial bridge radius
  \(R_0\) and the approximate point-contact condition \(Z_0 = R_0^2/2\).
- The exterior is a passive gas with zero density and zero viscosity. Every
  case file says so. A drop in a dynamically active exterior is a different
  problem and is not the default here.
- State the initial bridge geometry in every case. A finite initial gap
  produces a Taylor–Culick regime whose duration is set by \(Z_0\); any linear
  early regime is checked against that before being interpreted.
- The Stokes-to-inviscid crossover is \(R_c \approx \mathrm{Oh}\), with
  \(\mathrm{Oh}^2 = R_c^2|\ln R_c|\) at small Oh. Do not use
  \(R_c \approx \mathrm{Oh}^2\).
- A neck-growth exponent fitted over a limited window is weak evidence for a
  balance. Prefer local dimensionless tests: neck velocity rescaled by the
  viscocapillary velocity, curvature scaling, similarity collapse.
- Oldroyd-B has no finite extensibility. Where the strongly elastic limit is
  the point, say whether the conclusion survives a finite-extensibility
  closure. Report the yield-stress regularisation alongside any claim about
  arrest or a final shape.
- Report verification, convergence in mesh and in \(R_0\), and comparison
  with published data as separate tests.

## Repository rules

- Layout: all library code in `src/coalescence/` (`fem`, `bim`, `analysis`); the
  coalescence entry point and its default case in `simulationCases/`; each evidence
  folder owns its cases, run scripts and comparison data (`validationCases/` for
  independent published data, `verificationCases/` for exact or independent numerical
  solutions); analysis and video scripts that act on run outputs in `postProcess/`;
  tests in `testCases/`. Do not add a second cases tree.
- `coalescence.fem` and `coalescence.bim` never import each other, and `bim` never
  imports pyoomph; `testCases/test_layout.py` enforces it. They share only case files.
- No simulation output, run lists (`runs.toml`), run identities, archive paths or
  machine names are committed; they are kept privately with `data-roots.toml`. A
  published figure states only the solver commit and pinned pyoomph version that
  produced it. Results and their discussion are not documented in this repository.
- pyoomph is pinned in `pyproject.toml` / `uv.lock`. Change the pin only
  through the pinned-environment workflow, never by hand; commit the lockfile
  whenever it changes.
- Local environment and build directories (`.venv*`, `.pyoomph-*`) are
  ignored and never committed.
- Every case validates against its schema before any run. Every run has a
  manifest binding the case hash, the repository commit and the pyoomph
  commit. Run output does not live in this repository.
- pyoomph's coordinate-aware integrals already carry the axisymmetric
  \(2\pi r\) measure; do not apply it twice.

## pyoomph traps that apply here

These produce a plausible, wrong answer if violated.

- Never use `DirichletBC` with a value that depends on an unknown; impose it
  through `EnforcedBC` / `EnforcedDirichlet`, which use a Lagrange multiplier.
- Point constraints on the symmetry axis vanish in axisymmetry, because the
  measure carries \(2\pi r\). Express axis constraints as surface integrals,
  or pass `coordinate_system=cartesian` to the contribution.
- `NavierStokesFreeSurface` resolves `static_interface="auto"` to a moving
  interface only when a `BaseMovingMeshEquations` is present on the bulk
  domain; `partial_t(..., ALE="auto")` is ALE-corrected only once mesh
  equations exist.
- In axisymmetry the azimuthal conformation component is named `aa`; a
  material free surface takes no conformation boundary condition.
- A polymer modulus of zero should remove the conformation unknowns entirely,
  giving an exact Newtonian baseline, not a decoupled zero field.

## Documentation boundary

<!-- documentation-boundary-v1 -->

All content here, including this file, is public-candidate and uses calibrated
research prose. Provisional results, run records and debugging narrative do
not belong in this repository.
