# Anthony, Harris & Basaran (2020), Figure 3

**Comparison.** The neck radius R_min(τ_v), the neck velocity u_v(R_min) and the
neck curvature |2H|(R_min) of two equal drops coalescing from the paper's exact initial
bridge, R0 = 10⁻⁶ and Z0 = R0²/2, in a passive exterior and from rest, against the
digitised data of *Phys. Rev. Fluids* **5**, 033608 (2020), Figure 3, for two branches:

| Branch | Case | Run script |
|---|---|---|
| Stokes (Oh = ∞) | [`T5-stokes-R0-1e-06.json`](T5-stokes-R0-1e-06.json) | [`run_stokes.sh`](run_stokes.sh) |
| Oh = 0.6 | [`T5-Oh0.6-R0-1e-06.json`](T5-Oh0.6-R0-1e-06.json) | [`run_oh0.6.sh`](run_oh0.6.sh) |

Each script runs [`simulationCases/run_coalescence.py`](../../simulationCases/run_coalescence.py)
on its case with the numerical options of the published comparison, to R_min = 0.03.
The Stokes run starts from the frozen-geometry Stokes field; the Oh = 0.6 run starts from
rest with that field as its first Newton iterate and records the energy budget.

**Published data.** [`data/`](data/) holds the digitised series of the paper's Figure 3
(radius, velocity and curvature, Stokes and Oh = 0.6) and the record of their extraction
from the vector markers of the published PDF.

**Procedure.** The contact time t_con is the paper's: a power law fitted once R_min has
grown by one decade (10 R0 to 100 R0) and extrapolated to zero radius, so that
τ_v = t + t_con. No quantity is fitted to the published data.

```bash
validationCases/anthony2020/run_stokes.sh <stokes output>
validationCases/anthony2020/run_oh0.6.sh <oh0.6 output>
python validationCases/anthony2020/plot_anthony_fig3.py <stokes output> \
  --finite <oh0.6 output> --out <output prefix>
```

The plot script writes the figure and a JSON file with, for each run, the solver commit,
the neck-history checksum and the relative differences from the published markers.
A published comparison states the solver commit and pinned pyoomph version that
produced it.
