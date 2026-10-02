# verificationCases/

Tests of the numerics: the implemented equations against an independently computed
solution of the same equations, or against the same computation at different
numerical parameters. Each folder states the question, lists its simulations in
`runs.toml`, and names the `postProcess/` script that rebuilds its figure.

- [`bim-vs-fem-stokes-startup/`](bim-vs-fem-stokes-startup/) — the boundary-integral
  solver against the finite-element solver for the Stokes startup from R0 = 10⁻⁶.
- [`fem-startup-r0-independence/`](fem-startup-r0-independence/) — finite-element
  startups from eight initial neck radii at the same resolution settings.
