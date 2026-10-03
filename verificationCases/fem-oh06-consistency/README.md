# Oh = 0.6: frame of reference, mesh, time step and initial radius

**Question.** Is the inertial neck history at Oh = 0.6 independent of the numerical
choices that the Stokes runs were shown to be independent of: the frame of reference
(laboratory or the moving neck frame, which adds the term −ρ (dR_neck/dt) ∂_X u), the tip
mesh map, the time step, and the initial radius R0 once the startup is forgotten?

**Comparator.** The laboratory-frame run with the composite tip map at R0 = 10⁻³ (role
`reference` in [`runs.toml`](runs.toml)).

**Runs.** At R0 = 10⁻³, to R_min = 0.03: the moving neck frame with the same map; the
laboratory frame with the single map of the Stokes production runs; that run with the
curvature-change target and time-step fraction halved. At R0 = 10⁻⁶: the moving neck
frame with the production settings of the Stokes R0 = 10⁻⁶ run, started from rest with
the frozen-Stokes field as the first Newton iterate (`--stokes-first-guess`; from u = 0
Newton diverges at this R0), to R_min = 0.03.

**Results** (u_v/u_v,ref − 1 at equal R_min).

| | range over 1.001–30 R0 | 2 R0 | 30 R0 |
|---|---|---|---|
| moving neck frame | −0.044% … +0.042% | +0.030% | +0.042% |
| single tip map | −0.045% … +0.012% | +0.0002% | +0.0008% |
| single tip map, half time step | −0.045% … +0.009% | −0.0014% | −0.0015% |
| R0 = 10⁻⁶ | +0.10% at 2 × 10⁻³, +0.03 … +0.04% beyond 3 × 10⁻³ | | |

The largest negative differences sit in the first steps from rest; the moving-frame run
drifts to +0.04% by 30 R0. The run from R0 = 10⁻⁶ joins
the R0 = 10⁻³ history once the latter has forgotten its own startup, as in the Stokes
initial-radius study. The energy budget closes with median relative residuals between
3 × 10⁻⁸ and 4 × 10⁻⁵ at R0 = 10⁻³ and 1.5 × 10⁻⁴ at R0 = 10⁻⁶; at R0 = 10⁻⁶ the 95th
percentile is large, consistent with area changes per step below the round-off of the
area integral at the smallest R_min, and has not been examined further.

**Not yet done.** Mesh-grading and remesh-interval partners; comparison with the Oh = 0.6
data of Anthony et al. (2020), Figure 3.

**Reproduce.**

```bash
python postProcess/compare_neck_histories.py verificationCases/fem-oh06-consistency/runs.toml --out <output prefix>
```
