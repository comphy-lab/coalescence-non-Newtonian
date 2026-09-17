# runners/

`run_case.py` loads one `pyoomph-case-v1` JSON file and integrates
`problems.newtonian.coalescence_axisym.CoalescenceProblem`. Run output is
written to the directory passed as `--outdir`, not to this repository.

`run_mapped_case.py` is the production entry point for the explicit mapped Q2
problem. It accepts a manifest-bound `--case`, a pre-created `--attempt-dir`,
and `--segment-id`. The hashed case's `mesh.remeshing` boolean is the sole
automatic-remesh enablement switch; there is no mutable numerical run config.

Enabled runs remain unarmed through initialisation. After the first accepted
physical timestep they capture a real-state residual/flux baseline, perform a
frozen identity remesh, and evaluate it with the independent volume,
interface, Jacobian, divergence and boundary-flux diagnostics. Only that
accepted-state receipt arms the exact Anthony `R_min x4` state machine. The
baseline is refreshed after every accepted step and passed to every remesh
comparison. Each calibration, automatic remesh, or failure is appended to the
configured JSONL receipt. Preparation completes all point-location and
field-schema checks before mesh recreation; a failed preparation leaves the
live mesh untouched. A failed post-remesh diagnostic stops the run and leaves
automatic remeshing disabled.

`--mesh-only` initialises without advancing time. Automatic remeshing therefore
remains `awaiting-accepted-step`; a quiescent initial mesh is never accepted as
a production calibration. Production execution requires an existing attempt
directory from the normal manifest/environment/reservation/run-contract
preflight. `--segment-id` selects the pre-created base or restart owner, and
the runner immediately appends an fsynced `segment-start` record to
`attempt/runtime/progress.jsonl`. Accepted-step and checkpoint records are
appended and fsynced there as the solve proceeds. The base segment is the
pre-created `attempt/base/`; restart segments must already contain the
`restart.json` produced by `pyoomph_project.py new-restart`. The runner verifies
the complete parent/checkpoint chain and never creates or replaces a segment.

For a non-base segment, the state named and hash-bound by `restart.json` is
loaded after initialisation. Automatic runs require it to contain a matching
accepted-step calibration and remesh epoch; they neither recalibrate from a
quiescent state nor reset the x4 baseline to the restart radius. A passing
accepted-state calibration and every successful remesh write an exclusive
state checkpoint plus an append-only `pyoomph-field-event-v1` record.

All numerical policy is case-hashed: Newton tolerance and iteration cap,
minimum timestep, linear solver, start/max/end times, remesh factor, maximum
accepted trigger ratio, and all diagnostic/calibration thresholds. The run
runner adds no mutable numerical overrides.

Completion is likewise case-hashed and supports either `R_min >= threshold`
or `accepted_steps >= positive_integer`. The latter is the calibration-run
path: it stops after exactly the requested accepted solves, but only after the
first accepted-state identity calibration and checkpoint have completed.
`max_time` remains a safety limit and is reported as an incomplete failure.

The accepted-step timestep cap is derived from the live mapped mesh. It uses
the minimum sampled bulk Q2 Jacobian singular length, minimum Q2 interface
segment scale, `R_min^2/16`, and `4/abs_2H`, divided by the maximum sampled Q2
speed magnitude with a 0.4 Courant factor. At the initial state this uses the
sampled Q2 fluid speed. After an accepted step, nodal average mesh velocity is
reconstructed from `(x[0]-x[1])/dt`; fluid, mesh and ALE-relative `u-w` are
all Q2-sampled and the largest magnitude controls the cap. Receipts distinguish
`sampled-fluid-q2` from `sampled-max-fluid-mesh-relative-q2` and record all
three maxima. Zero sampled speed leaves the advective cap unbounded and the
hashed `run.maxstep` controls the step. The only geometry floor is derived
from local coordinate ULPs; no fixed `1e-12` length remains. Every accepted
progress row records all component scales, sampled speeds and selected cap.
