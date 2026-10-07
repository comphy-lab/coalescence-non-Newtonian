# Inertial solver against the linear modes of a viscous drop

**Question.** Does the finite-element solver with inertia reproduce the small-amplitude
shape oscillation of a free viscous drop, its frequency and its decay rate, at several
Ohnesorge numbers? The test exercises the inertial terms, the free surface and the moving
mesh without the neck.

**Comparator.** The exact linear normal mode of a viscous drop in a passive exterior,
`coalescence.analysis.drop_modes`: the Laplace-transformed linear problem (potential plus
poloidal viscous part, three interface conditions), solved for its least-damped
oscillatory root. `testCases/test_drop_modes.py` checks it against Rayleigh's inviscid
frequency, Lamb's weak-viscosity decay rate, which it approaches with an Oh^{1/2}
correction, and the quasi-static Stokes rate.

**Run.** One quadrant of a drop of unit radius released from rest with the shape
r = a[1 + εP₂(cos θ)], ε = 0.01, in visco-capillary units (density 1/Oh²)
([`run_fem_drop_oscillation.py`](run_fem_drop_oscillation.py)).
[`fit_drop_oscillation.py`](fit_drop_oscillation.py) fits the pole height with a damped
cosine after the first quarter period, compares the frequency and decay rate with the
exact mode and with the leading-order Rayleigh and Lamb values, and checks the energy
budget d/dt(kinetic + surface energy) = −dissipation over whole periods with the surface
energy taken at fixed volume.

```bash
python verificationCases/drop-oscillation/run_fem_drop_oscillation.py \
  --Oh 0.02 --resolution 0.05 --steps-per-period 200 --periods 3 --out <folder>
python verificationCases/drop-oscillation/fit_drop_oscillation.py <folder> [<folder> ...] \
  --out <summary.json>
```
