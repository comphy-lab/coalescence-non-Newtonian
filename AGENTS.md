# AGENTS.md

Operating manual for the `coalescence-non-Newtonian` component. Read `README.md`
for scope.

## What this component is

Solver, cases and post-processing for the coalescence of viscoelastic drops,
used as a controlled proxy for biomolecular condensate fusion. It is an ordinary
independent repository. Project lifecycle, background and provisional results
live in the owning project context, not here.

## Scientific rules

- Establish the Newtonian baseline before any viscoelastic claim, and report the
  resolution at which the initial Stokes regime of coalescence is recovered.
  Coalescence begins in a Stokes regime; an inertially limited viscous regime
  appears only when the drops start at finite separation.
- Oldroyd-B has no finite extensibility, so polymer stress is unbounded in strong
  extension. Where the strongly elastic limit is the point of the calculation,
  state whether the conclusion survives a finite-extensibility closure.
- A neck-growth exponent fitted over a limited window is weak evidence for a
  balance. Prefer a local dimensionless test of the balance being claimed.
- Distinguish the free-surface configuration from a drop in a dynamically active
  exterior phase. They are different problems and the biological case is the
  second.
- For the inverse problem, fit only what an experiment could actually see.
  Degrade synthetic traces to realistic spatial and temporal resolution and noise
  before inverting, and report identifiability, not a best-fit value.

## Documentation boundary

<!-- documentation-boundary-v1 -->

All content here, including this file, is public-candidate and uses calibrated
research prose. Keep provisional simulation records, run identifiers, data paths,
host names and debugging narrative in the project tracker and project context
`scratch/`, not in this repository. Promotion of internal material into a
manuscript, report, site or slide requires Vatsal's explicit approval of both the
content and the named target. Every figure task routes through the
`publication-plots` skill.
