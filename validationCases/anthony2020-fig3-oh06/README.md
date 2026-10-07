# Anthony et al. (2020) Figure 3 at `Oh = 0.6`

This validation case compares the present finite-element solution with the
digitised `Oh = 0.6` markers of Anthony, Harris & Basaran (2020, Fig. 3),
using the paper's visco-capillary quantities. The Stokes finite-element run is
shown as the limiting reference, not as a fitted correction.

The independent boundary-integral method is documented separately in
[`verificationCases/bim-vs-fem-stokes-startup/`](../../verificationCases/bim-vs-fem-stokes-startup/).
It is an independently implemented Stokes BIM, used to verify the early
startup where present Stokes FEM and published Stokes markers differ.
It does not solve the inertial problem.

The promoted result, including the unit check, the FEM comparison, the measured
finite-Oh deficit relative to Stokes, and the R0-independence evidence, is in
[`docs/finite-oh-validation.md`](../../docs/finite-oh-validation.md).

The run identities and neck-history checksums are in [`runs.toml`](runs.toml).
The overview is rebuilt by `postProcess/plot_public_validation.py`; the
three-quantity comparison, consistency checks and R0 plots are rebuilt by
`postProcess/plot_finite_oh_validation.py`. Both use the compact public
plot-input bundle by default, or registered raw histories with `--archive`.
