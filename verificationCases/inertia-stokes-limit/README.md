# The inertial solver in the Stokes limit

**Question.** Does the inertial finite-element solver reduce to the verified Stokes
solver as the Ohnesorge number grows, and is the departure at a finite Oh a converged
property of the start from rest rather than a numerical artefact?

**Comparator.** The Stokes run at R0 = 10⁻³ of the initial-radius study (role
`reference` in [`runs.toml`](runs.toml)), computed with the same mesh and time-step
settings. The inertial runs differ only in the momentum equation (density 1/Oh² in
visco-capillary units) and start from rest. A finite Oh need not reproduce Stokes; the
requirement is that the difference vanishes as Oh → ∞.

**Runs.** Oh = 440 (the T3 case of Anthony et al. 2020), with a partner at half the time
step; Oh = 4400 and 44 000 to R_min = 1.1 R0; Oh = 10¹² for four steps. The three large-Oh
cases are in [`simulationCases/oh-limit/`](../../simulationCases/oh-limit/).

**Results** (u_v/u_v,Stokes − 1 at equal R_min).

| | 1.001 R0 | 1.01 R0 | 1.1 R0 | 2 R0 | 10 R0 | 30 R0 |
|---|---|---|---|---|---|---|
| Oh = 440 | −0.014% | −0.18% | −1.6% | −4.0% | −0.70% | −0.08% |
| Oh = 440, half time step | +0.003% | −0.18% | −1.6% | | | |
| Oh = 4400 | −0.014% | −0.16% | −0.70% | | | |
| Oh = 44 000 | −0.003% | −0.007% | −0.006% | | | |

At Oh = 10¹² the four steps reproduce the Stokes steps to 10⁻¹⁴. The difference
therefore vanishes as Oh grows. At Oh = 440 it is converged in the time step (half-step
partner within 3 × 10⁻⁵ of the full-step run from 1.003 R0 on), reaches −4% during the
startup and decays as the startup is forgotten (−0.08% at 30 R0). Its likely origin is the
start from rest: momentum diffuses across the drop in a time 1/Oh² (5 × 10⁻⁶ at
Oh = 440), longer than the time in which the initial bridge becomes a meniscus.

**Energy budget.** For runs written with `--energy-budget`, d/dt(kinetic + area) =
−dissipation holds step by step with a median relative residual of 1.7 × 10⁻⁷
(Oh = 440) and 3.2 × 10⁻⁵ (Oh = 44 000), excluding steps across a remesh
(`postProcess/check_energy_budget.py`).

**Reproduce.**

```bash
python postProcess/compare_neck_histories.py verificationCases/inertia-stokes-limit/runs.toml --out <output prefix>
```
