# Newtonian gate: Anthony, Harris & Basaran (2020)

Case matrix reproducing *Phys. Rev. Fluids* **5**, 033608 (2020). All cases:
axisymmetric, one quadrant, passive exterior (zero density and viscosity),
initial bridge surface \([r-(R_0+Z_0)]^2+z^2=Z_0^2\), quiescent start.
Unless a case says otherwise, \(Z_0 = R_0^2/2\) (approximate point contact).

| Tier | Cases | Reproduces | Pass condition |
|---|---|---|---|
| T0 | Stokes limit, \(R_0=10^{-3}\) | Fig. 7 squares; shallower verification case | \(R_{\min}=-(\tau_v/\pi)\ln\tau_v\) for \(R_{\min}<0.03\); \(u_v\) vs \(\ln R_{\min}\) slope \(1/\pi\); \(\lvert 2H\rvert\sim R_{\min}^{-3}\); profile collapse with \(r/R_{\min}\), \(z/R_{\min}^2\) |
| T1 | Oh = 0.6, \(R_0=10^{-3},10^{-4}\) | Fig. 4 and initial-radius convergence | collapse onto Stokes with \(\tau_v=\tau/\mathrm{Oh}\); no linear regime |
| T2 | Oh = 0.6 and Stokes, \(Z_0\in\{10^{-5},10^{-4}\}\) | Fig. 5, 6 | Taylor–Culick regime ending at \(R_c=(2Z_0)^{1/2}\), present even without inertia |
| T3 | Oh ∈ {440, 20, 3, 0.3, 0.03, 0.001} | Fig. 7, 8, 9 | Oh ≥ 3 on T0; 0.3, 0.03 leave to \(\tau^{1/2}\) near \(R_c\approx\mathrm{Oh}\); 0.001 inviscid |
| T4 | Oh ∈ {1, 0.1, 0.01, 0.003} | Fig. 11 | \(R_c(\mathrm{Oh})\) from local Re = 1; agrees with linear and log-corrected relations |
| T5 | Stokes and Oh = 0.6, \(R_0=10^{-6}\) (plus Oh = 0.6 at \(10^{-5}\); Stokes at intermediate \(R_0\) for the initial-radius study) | exact Fig. 3 initial condition | both curves reproduce Fig. 3 after the paper's time-origin fit; convergence in mesh, time step and \(R_0\) is shown separately because the paper does not publish element counts |

T0–T3 are the minimum gate. T4 is the phase diagram. T5 is the stretch. The last
column states pass conditions, not results. So far only the Stokes cases (T0 and T5)
have been computed; their comparison with the published figure, including the startup
transient where the two differ, is in
[`docs/la0-stokes-anthony-fig3.md`](../../docs/la0-stokes-anthony-fig3.md).

Diagnostics every run records: \(R_{\min}(t)\); \(Z_b=z(r=1.05R_{\min})\);
\(u_{\min}\); \(\lvert 2H\rvert\) at the neck; interface profile at each remesh;
back-of-drop axial velocity; \(t_{con}\) by power-law extrapolation once
\(R_{\min}\) has grown by a decade.

Times: \(\tau=t+t_{con}\) from the singularity; \(\tau_v=\tau/\mathrm{Oh}\) in
viscocapillary units for comparison with the Stokes run.
