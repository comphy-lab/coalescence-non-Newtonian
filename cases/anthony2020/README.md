# Newtonian gate: Anthony, Harris & Basaran (2020)

Case matrix reproducing *Phys. Rev. Fluids* **5**, 033608 (2020). All cases:
axisymmetric, one quadrant, passive exterior (zero density and viscosity),
initial bridge surface \([r-(R_0+Z_0)]^2+z^2=Z_0^2\), quiescent start.
Unless a case says otherwise, \(Z_0 = R_0^2/2\) (approximate point contact).

| Tier | Cases | Reproduces | Pass condition |
|---|---|---|---|
| T0 | Stokes limit, \(R_0=10^{-3}\) | Fig. 3 reference, Fig. 7 squares | \(R_{\min}=-(\tau_v/\pi)\ln\tau_v\) for \(R_{\min}<0.03\); \(u_v\) vs \(\ln R_{\min}\) slope \(1/\pi\); \(\lvert 2H\rvert\sim R_{\min}^{-3}\); profile collapse with \(r/R_{\min}\), \(z/R_{\min}^2\) |
| T1 | Oh = 0.6, \(R_0=10^{-3},10^{-4}\) | Fig. 3, 4 | collapse onto T0 with \(\tau_v=\tau/\mathrm{Oh}\); no linear regime |
| T2 | Oh = 0.6 and Stokes, \(Z_0\in\{10^{-5},10^{-4}\}\) | Fig. 5, 6 | Taylor–Culick regime ending at \(R_c=(2Z_0)^{1/2}\), present even without inertia |
| T3 | Oh ∈ {440, 20, 3, 0.3, 0.03, 0.001} | Fig. 7, 8, 9 | Oh ≥ 3 on T0; 0.3, 0.03 leave to \(\tau^{1/2}\) near \(R_c\approx\mathrm{Oh}\); 0.001 inviscid |
| T4 | Oh ∈ {1, 0.1, 0.01, 0.003} | Fig. 11 | \(R_c(\mathrm{Oh})\) from local Re = 1; agrees with linear and log-corrected relations |
| T5 | Oh = 0.6, \(R_0=10^{-5},10^{-6}\) | Fig. 3 at \(10^{-6}\) | honest depth statement |

T0–T3 are the minimum gate. T4 is the phase diagram. T5 is the stretch.

Diagnostics every run records: \(R_{\min}(t)\); \(Z_b=z(r=1.05R_{\min})\);
\(u_{\min}\); \(\lvert 2H\rvert\) at the neck; interface profile at each remesh;
back-of-drop axial velocity; \(t_{con}\) by power-law extrapolation once
\(R_{\min}\) has grown by a decade.

Times: \(\tau=t+t_{con}\) from the singularity; \(\tau_v=\tau/\mathrm{Oh}\) in
viscocapillary units for comparison with the Stokes run.
