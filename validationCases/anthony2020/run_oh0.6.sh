#!/usr/bin/env bash
# Anthony, Harris & Basaran (2020) Figure 3, Oh = 0.6 branch: R0 = 1e-6, Z0 = R0^2/2,
# passive exterior, from rest, to R_min = 0.03. The numerical options are those of the
# computation compared with the published figure.
#
#   validationCases/anthony2020/run_oh0.6.sh <output folder>
#
# Set PYTHON to an interpreter with the pinned pyoomph (pyproject.toml, uv.lock).
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <output folder>" >&2
  exit 2
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../.." && pwd)

exec "${PYTHON:-python}" "$root/simulationCases/run_coalescence.py" "$here/T5-Oh0.6-R0-1e-06.json" --out "$1" \
  --h-max 0.02 --dt-initial 1e-15 --spatial-scale 0 \
  --tip-map 0.5 --tip-map-outer 0.4 --tip-map-core 1e3 --tip-map-linear-core 1 \
  --grading 0.1 --interface-grading 0.035 --zone-grading 0.05 --zone-inner 1e-3 --zone-outer 2 \
  --h-tip-floor 1e-30 --remesh-growth 1.1 --curvature-step-limit 0.2 \
  --min-newton 3 --max-residuals 1e13 --neck-frame --neck-frame-moving \
  --stokes-first-guess --energy-budget \
  --r-stop 0.03 --field-frames
