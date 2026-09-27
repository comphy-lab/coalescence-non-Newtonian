"""Drop coalescence in Newtonian and non-Newtonian media.

- ``coalescence.fem``: the pyoomph finite-element solver (production).
- ``coalescence.bim``: an independent axisymmetric boundary-integral Stokes solver,
  used only to verify the finite-element solver in the Stokes limit.
- ``coalescence.analysis``: solver-agnostic readers, comparisons and theory.

``fem`` and ``bim`` never import each other; they share only the case files.
"""
