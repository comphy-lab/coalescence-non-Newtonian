# Large-Oh cases for the Stokes limit of the inertial solver

The Oh = 440 case of Anthony, Harris & Basaran (2020), `../anthony2020/T3-Oh440-R0-1e-3.json`,
with Oh raised to 4400, 44 000 and 10¹² and nothing else changed (R0 = 10⁻³, Z0 = R0²/2,
passive exterior, quiescent start). They are not part of the published case matrix; they
test that the inertial solver recovers the Stokes solver as Oh → ∞
([`verificationCases/inertia-stokes-limit/`](../../verificationCases/inertia-stokes-limit/)).
The finite-element runner works in visco-capillary units (length R, time μR/γ, density
1/Oh²) and records them in each run manifest.
