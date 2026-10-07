# postProcess/

Each script rebuilds one figure from the run list of an evidence case and writes
`<prefix>.pdf`, `<prefix>.png` and `<prefix>.json` (the numbers behind the figure and
the SHA-256 of every input). Outputs are located through `data-roots.toml`, and every
neck history is checked against the SHA-256 in the run list before use.

| Script | Evidence case |
|---|---|
| `plot_startup_family.py` | `verificationCases/fem-startup-r0-independence/`, also used by `validationCases/anthony2020-fig3-stokes/` and `modelComparisonCases/eggers1999-stokes-law/` |
| `plot_bim_vs_fem.py` | `verificationCases/bim-vs-fem-stokes-startup/` |
| `plot_anthony_fig3.py` | `validationCases/anthony2020-fig3-stokes/` (Figure 3 panels, inset and the contact-time fit) |
| `plot_public_validation.py` | `validationCases/anthony2020-fig3-oh06/` (FEM, Anthony's two curves and independent Stokes BIM) |
| `plot_finite_oh_validation.py` | `docs/finite-oh-validation/` (three-quantity comparison, finite-Oh deficit, convergence and R0 family) |

Figures are not stored in the repository, except those in `docs/figures/` that belong
to approved documentation.
