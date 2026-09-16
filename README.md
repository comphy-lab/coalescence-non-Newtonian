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

## Contents

- `cases/anthony2020/` — Newtonian reference cases reproducing the regimes and
  the Stokes-to-inviscid crossover of Anthony, Harris & Basaran (2020). See
  the README there for the case matrix and pass conditions.
- `problems/` — problem-class definitions, one module per physical problem.
- `runners/`, `postprocess/` — runner adapters; neck diagnostics, similarity
  collapse, regime fits.
- `tests/`, `verification/`, `validation/` — software tests, equation
  verification, comparison with published data.
- `docs/` — documentation.

## Method

pyoomph: Galerkin finite elements on a moving, remeshed mesh, fully implicit
in time. Axisymmetric free surface with surface tension; by symmetry, one
quadrant of one drop. The pyoomph version is pinned in `pyproject.toml` and
`uv.lock` so that any result can be traced to an exact solver commit.

## Status

Scaffold. Case matrix written; no solver code and no results yet.
