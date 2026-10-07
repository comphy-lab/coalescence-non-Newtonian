# Coalescence of two equal drops

[`run_coalescence.py`](run_coalescence.py) runs the axisymmetric, tip-graded
finite-element solver (`coalescence.fem`) for one case file and writes one output folder:
the neck history `neck.csv`, the saved states, a run manifest binding the case hash, the
component commit and the pyoomph installation, and a summary with the stop status.
[`coalescence.json`](coalescence.json) is the default case.

```bash
python simulationCases/run_coalescence.py simulationCases/coalescence.json --out <folder> [options]
```

**Case file** (`pyoomph-case-v1`). The case supplies the physics; every numerical
parameter and the stop radius come from the options, which the manifest records.

| Field | Meaning |
|---|---|
| `physics.inertia` | `false` for Stokes flow (Oh = ∞); `true` with `physics.Oh` for finite Oh |
| `physics.Oh` | Ohnesorge number μ/√(ργR) |
| `physics.initial_bridge.R0`, `.Z0` | initial bridge radius and half-gap; Z0 = R0²/2 is approximate point contact |
| `physics.exterior` | passive: zero density and viscosity |
| `physics.initial_velocity` | `quiescent`: the drops start from rest |

The runner works in visco-capillary units (length R, velocity γ/μ, time μR/γ, pressure
γ/R; density 1/Oh²). For another case, copy `coalescence.json` and change `physics`.
Inertial starts from small R0 need `--stokes-first-guess`; the run scripts in
[`validationCases/anthony2020/`](../validationCases/anthony2020/) give complete production
option sets. `--field-frames` saves the velocity and pressure fields for the case video
(`postProcess/make_case_video.py`).
