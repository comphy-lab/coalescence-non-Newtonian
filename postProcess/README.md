# postProcess/

Analysis, figure and case-video scripts that act on run outputs. Figure scripts write
`<prefix>.pdf`, `<prefix>.png` and `<prefix>.json` (the numbers behind the figure and the
SHA-256 of every input). Scripts that compare several runs read a run list (`runs.toml`,
format in `src/coalescence/analysis/runs.py`) naming the output folders; run lists are
kept with the outputs, not in this repository.

| Script | Purpose |
|---|---|
| `compare_neck_histories.py` | u_v of several runs against a reference run at equal R_min, with energy budgets |
| `check_energy_budget.py` | d/dt(kinetic + area) = −dissipation step by step, for runs written with `--energy-budget` |
| `plot_startup_family.py` | Stokes runs from several initial radii R0, against the Stokes-regime law |
| `plot_bim_vs_fem.py` | boundary-integral runs against the finite-element reference |
| `make_case_video.py` | the standard case video (see the repository README) |
| `render_hybrid_video.py`, `render_case_dashboard_video.py`, `render_neck_video.py`, `render_drop_pair_video.py` | video renderers used by `make_case_video.py` and for the neck-history dashboard |
| `reconstruct_stokes_fields.py` | Stokes velocity and pressure rebuilt at saved states of a run without field frames |

The Anthony et al. (2020) comparison has its own script in `validationCases/anthony2020/`,
and the drop-oscillation fit in `verificationCases/drop-oscillation/`.
