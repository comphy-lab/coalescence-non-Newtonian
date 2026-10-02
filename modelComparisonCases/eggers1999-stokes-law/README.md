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

**Compared.** The theory is a function of R_min alone, for a neck grown from point
contact, so it is compared only where a computation has forgotten its initial bridge
(beyond a few R0, see `verificationCases/fem-startup-r0-independence/`) and only for
R_min ≤ 0.03. The leading order fixes the coefficient 1/π of the logarithm but not an
additive constant: u_v = (1/π) ln(C/R_min) with C undetermined at that order. C is
fitted to the smallest-R0 run over 10 R0 ≤ R_min ≤ 0.03 with the slope held at 1/π (a
free-slope fit is reported alongside), and the data are divided by the fitted law: a
trajectory that follows it lies on 1, so flatness tests the slope and departures show
where the law does not hold (the startup transient; finite size beyond 0.03).

**Runs.** The finite-element runs of
[`../../verificationCases/fem-startup-r0-independence/runs.toml`](../../verificationCases/fem-startup-r0-independence/runs.toml).

**Reproduce.** `postProcess/plot_startup_family.py`: both forms against R_min in panel (a),
the ratio to the fitted law in panel (b); the fit is written to the `.json`.
