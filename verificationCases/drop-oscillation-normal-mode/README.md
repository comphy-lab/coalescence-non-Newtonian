# Inertial solver against the linear modes of a viscous drop

**Question.** Does the finite-element solver with inertia reproduce the small-amplitude
shape oscillation of a free viscous drop, its frequency and its decay rate, at several
Ohnesorge numbers? The test exercises the inertial terms, the free surface and the moving
mesh without the neck.

**Comparator.** The exact linear normal mode of a viscous drop in a passive exterior,
`coalescence.analysis.drop_modes`: the Laplace-transformed linear problem (potential plus
poloidal viscous part, three interface conditions), solved for its least-damped
oscillatory root. It is tested in `testCases/test_drop_modes.py` against Rayleigh's
inviscid frequency, Lamb's weak-viscosity decay rate (l − 1)(2l + 1)ν/R², which it
approaches with an Oh^{1/2} correction, and the quasi-static Stokes rate. The
leading-order Rayleigh and Lamb values are reported alongside.

**Runs.** One quadrant of a drop of unit radius released from rest with the shape
r = a[1 + εP₂(cos θ)], ε = 0.01, in visco-capillary units (density 1/Oh²), for three
Rayleigh periods: Oh = 0.05, 0.02 and 0.01, Gmsh element size 0.05 and 0.025, and 200 and
400 time steps per period (`run_fem_drop_oscillation.py`, solver commit 9178033). The pole
height is fitted with a damped cosine after a quarter period, and the energy budget
d/dt(kinetic + surface energy) = −dissipation is checked over whole periods with the
surface energy taken at fixed volume (`postProcess/fit_drop_oscillation.py`).

| Oh | element size | steps/period | ω vs mode (%) | decay vs mode (%) | ω vs Rayleigh (%) | decay vs Lamb (%) | energy residual per period | volume drift |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 200 | −0.044 | −0.08 | −1.10 | −12.6 | 8.8 × 10⁻⁴ | −1.1 × 10⁻⁶ |
| 0.05 | 0.025 | 200 | −0.045 | −0.09 | −1.10 | −12.6 | 8.6 × 10⁻⁴ | −3.1 × 10⁻⁶ |
| 0.02 | 0.05 | 200 | −0.044 | −0.11 | −0.31 | −7.6 | 7.5 × 10⁻⁴ | −1.8 × 10⁻⁷ |
| 0.02 | 0.025 | 200 | −0.044 | −0.13 | −0.32 | −7.6 | 7.5 × 10⁻⁴ | −5.6 × 10⁻⁶ |
| 0.02 | 0.05 | 400 | −0.019 | −0.06 | −0.29 | −7.5 | 2.2 × 10⁻⁴ | −2.6 × 10⁻⁶ |
| 0.01 | 0.05 | 200 | −0.042 | −0.05 | −0.14 | −5.2 | 5.1 × 10⁻⁴ | −1.5 × 10⁻⁸ |

**Reading.** The departures from Rayleigh and Lamb (up to 1.1% in frequency and 13% in
decay rate at Oh = 0.05) do not change with the mesh or the time step: they are the
finite-Oh corrections that the exact mode carries. Against the exact mode the frequency
agrees to −0.044% at 200 steps per period and −0.019% at 400; the remainder that does not
scale with the time step, about −0.01%, is of the size expected from the O(ε²) nonlinear
shift at ε = 0.01. The decay rate agrees to 0.05–0.13%; the fit window starts a quarter
period after release, where the initial-value response still carries a small viscous
transient. Halving the time step reduces the energy residual per period by a factor 3.4.
The same oscillation computed with the pyoomph spatial scale set to 10⁻³ (element size
and Newton tolerance scaled accordingly) reproduces the scale-1 history to 10⁻¹³.

**Not resolved.** The volume drifts by 10⁻⁸ … 6 × 10⁻⁶ over three periods, more on the
finer mesh; without the fixed-volume correction this drift dominates the energy budget
of a weakly damped oscillation.

**Reproduce.**

```bash
python run_fem_drop_oscillation.py --Oh 0.02 --resolution 0.05 --steps-per-period 200 --periods 3 --out <folder>
python postProcess/fit_drop_oscillation.py <folder> [<folder> ...] --out <summary.json>
```
