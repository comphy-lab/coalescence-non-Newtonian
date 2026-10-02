# Finite-element startups from eight initial neck radii

**Question.** Point contact is approximated by a bridge of radius R0 and half height
Z0 = R0²/2. After how many initial radii does the neck velocity forget R0, and is the
trajectory beyond that independent of R0?

**Comparator.** `coalescence.fem` at R0 = 10⁻⁶, 10^−5.75, 10^−5.5, 10⁻⁵, 10^−4.5,
10⁻⁴, 5 × 10⁻⁴ and 10⁻³ with the same resolution settings, each from its own t = 0.
The smallest R0 is the reference.

**Runs.** [`runs.toml`](runs.toml).

**Compared.** u_v at equal R_min (linear interpolation in ln R_min) against
R_min/R0; the radius after which each run stays within 1, 0.3 and 0.1% of the
reference; the height and position of the startup maximum. No offsets are fitted.

**Reproduce.**

```bash
python postProcess/plot_startup_family.py verificationCases/fem-startup-r0-independence/runs.toml \
  --anthony validationCases/anthony2020-fig3-stokes/anthony2020-fig3-velocity-digitized.csv \
  --out <output prefix>
```
