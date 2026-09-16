# problems/

Problem-class definitions. One module per physical problem, parameterised by
case JSON.

- `newtonian/` — Newtonian free-surface coalescence (Anthony 2020 gate).
  Planned: `coalescence_axisym.py` — one-quadrant axisymmetric drop, passive
  exterior, initial bridge of Eq. 2, remeshing on the factor-of-four rule,
  neck diagnostics.

Later families (`viscoelastic/`, `elastoviscoplastic/`) are added only after
the Newtonian gate passes.
