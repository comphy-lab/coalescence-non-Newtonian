# problems/

Problem-class definitions. One module per physical problem, parameterised by
case JSON.

- `newtonian/` — Newtonian free-surface coalescence (Anthony 2020 gate).
  `coalescence_axisym.py` is the one-quadrant axisymmetric drop: passive
  exterior, initial bridge of Eq. 2, remeshing on the factor-of-four rule,
  neck diagnostics. `geometry.py` holds the Eq. 2/sphere junction.

Later families (`viscoelastic/`, `elastoviscoplastic/`) are added only after
the Newtonian gate passes.
