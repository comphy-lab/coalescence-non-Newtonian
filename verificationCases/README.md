# verificationCases/

Tests of the numerics: the implemented equations against an independently computed
solution of the same equations, or against the same computation at different
numerical parameters. Each folder states the question, lists its simulations in
`runs.toml`, and names the `postProcess/` script that rebuilds its figure.

- [`bim-vs-fem-stokes-startup/`](bim-vs-fem-stokes-startup/) — the boundary-integral
  solver against the finite-element solver for the Stokes startup from R0 = 10⁻⁶.
- [`fem-startup-r0-independence/`](fem-startup-r0-independence/) — finite-element
  startups from eight initial neck radii at the same resolution settings.
- [`drop-oscillation-normal-mode/`](drop-oscillation-normal-mode/) — the inertial solver
  against the exact linear shape modes of a viscous drop.
- [`inertia-stokes-limit/`](inertia-stokes-limit/) — inertial runs at increasing Oh
  against the Stokes run, and the energy budget.
- [`fem-oh06-consistency/`](fem-oh06-consistency/) — Oh = 0.6 in two frames of reference,
  two tip maps, two time steps and two initial radii.
