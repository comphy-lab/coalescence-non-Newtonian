# postprocess/

`fig3.py` overlays Stokes and finite-Oh neck histories in the coordinates of
Anthony, Harris & Basaran (2020), Fig. 3. It fits
`R_min=A(t+t_con)^beta` over the completed first decade of bridge growth and
refuses to report a fitted contact time before that decade exists. The printed
fit exponent and logarithmic RMS are part of the validation receipt; Eggers
inversion is not substituted for the paper's extrapolation.

The input may be a segment-local `neck.csv` or the attempt's append-only
`runtime/progress.jsonl`. The progress form is authoritative across restarts:
for an overlapping checkpoint time, the later segment record wins. Restart
receipts and checkpoint hashes define the selected terminal segment's parent
chain; sibling branches are excluded, and an ambiguous terminal branch must
be named in the `--run` specification.

`interface_video.py` encodes saved interface profiles. Figure files are not
stored in this repository.
