# Boundary integral against finite element: Stokes startup from R0 = 10⁻⁶

**Question.** Does an independent discretisation of the same Stokes free-surface
problem reproduce the finite-element neck velocity u_v(R_min), including the startup
transient from the exact bridge of Anthony et al. (2020) with R0 = 10⁻⁶ and
Z0 = R0²/2?

**Comparator.** `coalescence.bim` against `coalescence.fem`. The two solvers share
only the case file
[`T5-stokes-R0-1e-06.json`](../../simulationCases/anthony2020/T5-stokes-R0-1e-06.json);
they differ in formulation (surface integral equation against a volume mesh),
geometry representation, tip resolution strategy and time integration.

**Runs** ([`runs.toml`](runs.toml)). The finite-element reference; boundary-integral
runs at two tip gradings k; and a backward-Euler (Newton) series, which removes the
linearisation error of the linearly implicit step in the stiff limit. The Newton
series has two segments: the first restarts from the linearly implicit run, and the
second continues the first from its final state with a later commit that accepts
iterations stalled near the tolerance and retries failed steps at half the time
step. The figure script joins the segments into one curve.

**Compared.** u_v at equal R_min, with boundary-integral speeds assigned to the
geometric-mean radius of each step, and the tip radius divided by the constant ratio
of the two codes' tip-radius estimators. No offsets are fitted. Each solver's own
convergence (grading and Newton continuation here; mesh, time step and R0 for the
finite element) bounds the agreement that can be expected.

**Reproduce.**

```bash
python postProcess/plot_bim_vs_fem.py verificationCases/bim-vs-fem-stokes-startup/runs.toml \
  --anthony validationCases/anthony2020-fig3-stokes/anthony2020-fig3-velocity-digitized.csv \
  --out <output prefix>
```
