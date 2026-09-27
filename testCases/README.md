# testCases/

Software tests. From the repository root:

```bash
python -m unittest discover -s testCases -t .
```

`testCases/__init__.py` puts `src/` on the import path. The `test_bim_*`,
`test_theory`, `test_runs` and `test_layout` tests need only numpy and scipy;
`test_stokes_tip_method` and `test_bridge_geometry` exercise the finite-element code
and need the pinned pyoomph environment.
