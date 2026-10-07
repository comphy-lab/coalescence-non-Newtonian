# Boundary integral method against the finite-element method: Stokes coalescence

**Question.** Does an independent discretisation of the same Stokes free-surface problem
reproduce the finite-element neck velocity u_v(R_min), including the startup from the
exact bridge of Anthony et al. (2020)?

**Comparator.** `coalescence.bim`, an axisymmetric boundary-integral solver for Stokes
flow with a free surface, against `coalescence.fem`. The two share only the case file;
they differ in formulation (a surface integral equation against a volume mesh), in the
representation of the geometry, in the tip resolution and in the time integration, and
neither imports the other (`testCases/test_layout.py`).

**Run.** [`run_bim_stokes.py`](run_bim_stokes.py) takes the Stokes case of the Anthony
validation and writes its neck history to `neck.csv` with a run manifest.
`--newton` replaces the linearly implicit step by backward Euler solved with
Newton–Krylov iterations, removing the linearisation error in the stiff limit.

```bash
python verificationCases/bim-stokes/run_bim_stokes.py \
  validationCases/anthony2020/T5-stokes-R0-1e-06.json --out <folder> [--newton]
```

**Comparison.** u_v at equal R_min, with boundary-integral speeds assigned to the
geometric-mean radius of each step. No offsets are fitted.
[`postProcess/plot_bim_vs_fem.py`](../../postProcess/plot_bim_vs_fem.py) draws the
finite-element reference and the boundary-integral runs named in a run list (format in
`src/coalescence/analysis/runs.py`).
