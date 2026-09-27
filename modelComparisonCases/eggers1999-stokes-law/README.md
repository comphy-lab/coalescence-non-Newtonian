# The Stokes coalescence law of Eggers, Lister & Stone (1999)

**Model.** For Stokes coalescence of two drops in a passive exterior, Eggers, Lister &
Stone (*J. Fluid Mech.* **401**, 293, 1999) give, to leading logarithmic order and in
visco-capillary units, R_min = −(τ_v/π) ln τ_v. Anthony et al. (2020) quote it as
valid for R_min < 0.03. Two forms of the neck velocity follow at the same order
(`src/coalescence/analysis/theory.py`):

- u_v = −(1/π) ln R_min, the straight line drawn in Figure 3(b) of Anthony et al.;
- u_v = dR_min/dτ_v = −(1 + ln τ_v)/π, with τ_v from inverting the law.

They differ by (ln|ln τ_v| − ln π − 1)/π, a term beyond the accuracy of the leading
order, so the spread between them measures how far the asymptotic law can be held to
the computed curve.

**Compared.** The converged finite-element trajectory u_v(R_min), beyond the startup
transient, against both forms for R_min ≤ 0.03: slope in ln R_min and offset.

**Runs.** The finite-element runs of
[`../../verificationCases/fem-startup-r0-independence/runs.toml`](../../verificationCases/fem-startup-r0-independence/runs.toml).

**Reproduce.** `postProcess/plot_startup_family.py` draws both forms in panel (a).
