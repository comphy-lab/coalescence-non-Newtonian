# Anthony et al. (2020) Figure 3 at `Oh = 0.6`

This validation case compares the present finite-element solution with the
digitised `Oh = 0.6` markers of Anthony, Harris & Basaran (2020, Fig. 3),
using the paper's visco-capillary quantities. The Stokes finite-element run is
shown as the limiting reference, not as a fitted correction.

The independent boundary-integral method is documented separately in
[`verificationCases/bim-vs-fem-stokes-startup/`](../../verificationCases/bim-vs-fem-stokes-startup/).
It is a Stokes BIM, not a BEM, and is used to adjudicate the early discrepancy
between the present Stokes FEM and the published Stokes startup markers.

The promoted result, including the unit check, the FEM comparison, the measured
finite-Oh deficit relative to Stokes, and the R0-independence evidence, is in
[`docs/finite-oh-validation.md`](../../docs/finite-oh-validation.md).

The run identities and neck-history checksums are in [`runs.toml`](runs.toml).
The figure is rebuilt by `postProcess/plot_public_validation.py`.
