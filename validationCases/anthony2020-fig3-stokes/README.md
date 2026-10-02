# Anthony, Harris & Basaran (2020), Figure 3: Stokes branch

**Question.** Does the finite-element computation from the paper's exact initial
condition (R0 = 10⁻⁶, Z0 = R0²/2, 1/Oh = 0) reproduce the published neck velocity
u_v(R_min), neck radius R_min(τ_v) and neck curvature |2H|(R_min)?

**Independent data.** Anthony, Harris & Basaran, *Phys. Rev. Fluids* **5**, 033608
(2020), Figure 3, computed with an independent finite-element code. The article
publishes no tables; the markers were extracted from the vector graphics of the
figure (marker centres, calibrated to the axis ticks, no fitting to any curve):

| File | Columns | Content |
|---|---|---|
| `anthony2020-fig3-velocity-digitized.csv` | series, R_min, u_v | panel (b) |
| `anthony2020-fig3-radius-digitized.csv` | series, tau_v, R_min | panel (a) |
| `anthony2020-fig3-curvature-digitized.csv` | series, R_min, abs_two_H | panel (a) inset |
| `anthony2020-fig3-extraction.json` | | source hash, page, tick calibration and its residuals, checks of the drawn theory lines, marker counts |

Series `*_stokes` are the Stokes markers; series `*_oh0p6` are the Oh = 0.6 markers.
The extraction confirms that the theory line drawn in panel (b) is
u_v = −(1/π) ln R_min. Where markers overlap in the graphic, the extracted positions
carry the corresponding uncertainty; they are not raw simulation data.

**Runs.** [`runs.toml`](runs.toml): the R0 = 10⁻⁶ computation from the paper's exact
initial condition. The initial-radius study behind it is
[`../../verificationCases/fem-startup-r0-independence/`](../../verificationCases/fem-startup-r0-independence/).

**Reproduce.**

```bash
python postProcess/plot_anthony_fig3.py validationCases/anthony2020-fig3-stokes/runs.toml \
  --data validationCases/anthony2020-fig3-stokes --out <output prefix>
```

The contact time follows the paper's procedure (`src/coalescence/analysis/contact_time.py`);
the `.json` gives the relative differences from the published markers by range of
abscissa. `postProcess/plot_startup_family.py` overlays the velocity markers on the
initial-radius family. See [`docs/la0-stokes-anthony-fig3.md`](../../docs/la0-stokes-anthony-fig3.md).
