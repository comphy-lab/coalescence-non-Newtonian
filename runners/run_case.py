"""Run one Anthony 2020 case through the Newtonian coalescence problem class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.newtonian.coalescence_axisym import CoalescenceProblem, load_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--max-time", type=float, default=1.0)
    parser.add_argument("--startstep", type=float, default=None)
    parser.add_argument("--maxstep", type=float, default=None)
    parser.add_argument("--mesh-only", action="store_true")
    args = parser.parse_args()
    case = load_case(args.case)
    r0 = float(case["physics"]["initial_bridge"]["R0"])
    z0 = float(case["physics"]["initial_bridge"]["Z0"])
    startstep = args.startstep if args.startstep is not None else min(1e-4 * r0, 1e-2 * z0)
    maxstep = args.maxstep if args.maxstep is not None else 0.05 * r0
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "case.json").write_text(
        json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with CoalescenceProblem(case, args.outdir) as problem:
        if args.mesh_only:
            problem.initialise()
            row = problem.write_neck_row(profile=True)
            mesh = problem.get_mesh("drop")
            print(
                json.dumps(
                    {
                        "R_min": row["R_min"],
                        "u_min": row["u_min"],
                        "abs_2H": row["abs_2H"],
                        "Z_b": row["Z_b"],
                        "ndof": int(problem.ndof()),
                        "nelement": int(mesh.nelement()),
                    }
                )
            )
            return 0
        problem.run_until(max_time=args.max_time, startstep=startstep, maxstep=maxstep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
