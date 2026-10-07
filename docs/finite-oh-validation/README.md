# Finite-Oh validation report

[Checked report PDF](finite-oh-validation-v4.pdf), [source](main.tex) and
[bibliography](references.bib). The [web account](../finite-oh-validation.md)
contains the same comparisons, with the original finite-Oh figure retained.

## Build the report

From this directory, with TeX Live and `latexmk`:

```bash
make NAME=finite-oh-validation-v5
```

The PDF is written to `build/`. The build refuses an existing PDF job name.
The six files in [`figures/`](figures/) are reusable TeX figure environments
with their scientific captions. Their vector graphics live in
[`../figures/`](../figures/), shared with the web account.

## Rebuild the figures

From the repository root, using its pinned Python environment plus
Matplotlib and an external LaTeX installation:

```bash
python postProcess/plot_finite_oh_validation.py \
  --out <new-output-prefix>
python postProcess/plot_public_validation.py \
  --out <another-new-output-prefix>
```

These commands use the compact [`data/`](data/) bundle: only scalar plot
curves, not meshes or field output. Its manifest records each source
neck-history checksum and checks the bundle before reading it. The scripts
write PDF, PNG and JSON statistics and refuse to overwrite any output.
The comparison script calls the same contact-time and marker-comparison
helpers as `plot_anthony_fig3.py`. Radius statistics are binned in
$\tau_v$; velocity and curvature statistics are binned in $R_{\min}$.

To regenerate the bundle from the registered histories, configure the
repository's untracked `data-roots.toml`, then use:

```bash
python postProcess/plot_finite_oh_validation.py --archive \
  --export-data <new-bundle-directory> --out <new-output-prefix>
```

The Stokes BIM detail figure uses
[`plot_bim_vs_fem.py`](../../postProcess/plot_bim_vs_fem.py), the registered
[`runs.toml`](../../verificationCases/bim-vs-fem-stokes-startup/runs.toml),
and the digitised Anthony Stokes velocity markers. The overview includes
the BIM velocity curves from the compact bundle without raw-data access.
All Anthony marker inputs remain in
[`validationCases/anthony2020-fig3-stokes/`](../../validationCases/anthony2020-fig3-stokes/).

The checked figures use embedded Computer Modern fonts. The finite-Oh
two-radius check must not be confused with a completed intermediate-radius
family or a completed finite-Oh mesh/remeshing study.
