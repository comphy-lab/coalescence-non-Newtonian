"""Run one Anthony case with the verified mapped-mesh production path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from problems.newtonian.coalescence_axisym import load_case
from problems.newtonian.coalescence_mapped import MappedCoalescenceProblem


@dataclass(frozen=True)
class NumericalPolicy:
    max_time: float
    startstep: float
    maxstep: float
    linear_solver: str
    remesh_max_trigger_ratio: float
    completion_observable: str
    completion_threshold: float | int


def numerical_policy(case: dict[str, Any]) -> NumericalPolicy:
    """Validate the manifest-hashed effective numerical policy."""
    tolerances = case.get("tolerances", {})
    mesh = case.get("mesh", {})
    run = case.get("run", {})
    required_tolerances = ("newton", "max_newton_iterations", "minimum_dt")
    required_run = (
        "max_time", "startstep", "maxstep", "linear_solver",
        "remesh_max_trigger_ratio",
    )
    missing = [f"tolerances.{name}" for name in required_tolerances if name not in tolerances]
    missing += [f"run.{name}" for name in required_run if name not in run]
    missing += ["mesh.remesh_factor"] if "remesh_factor" not in mesh else []
    missing += ["mesh.remesh_gates"] if "remesh_gates" not in mesh else []
    if missing:
        raise ValueError(f"case lacks mapped production policy: {missing}")
    for name in ("newton", "minimum_dt"):
        value = float(tolerances[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"tolerances.{name} must be positive and finite")
    if float(tolerances["newton"]) < 5.0e-8:
        raise ValueError(
            "mapped production cases must declare tolerances.newton >= 5e-8"
        )
    iterations = tolerances["max_newton_iterations"]
    if isinstance(iterations, bool) or int(iterations) != iterations or int(iterations) < 1:
        raise ValueError("tolerances.max_newton_iterations must be a positive integer")
    if float(mesh["remesh_factor"]) != 4.0:
        raise ValueError("mapped production policy requires mesh.remesh_factor=4.0")
    if not isinstance(mesh.get("remeshing"), bool):
        raise ValueError("mesh.remeshing must be boolean")
    if case.get("output", {}).get("write_states") is not True:
        raise ValueError("mapped production runs require output.write_states=true")
    gates = mesh["remesh_gates"]
    gate_names = {
        "relative_volume", "relative_interface_normal",
        "relative_interface_tangential", "relative_neck", "relative_pole",
        "absolute_divergence", "absolute_divergence_correction",
        "absolute_flux_change", "relative_flux_change",
        "calibration_relative_divergence_growth",
        "calibration_relative_divergence_correction",
        "calibration_relative_flux_change",
    }
    if not isinstance(gates, dict) or set(gates) != gate_names:
        raise ValueError("mesh.remesh_gates must contain the 12 hashed gate values")
    if any(not math.isfinite(float(gates[name])) or float(gates[name]) < 0.0 for name in gate_names):
        raise ValueError("mesh.remesh_gates values must be finite and non-negative")
    values = {name: float(run[name]) for name in ("max_time", "startstep", "maxstep", "remesh_max_trigger_ratio")}
    if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
        raise ValueError("run time controls must be positive and finite")
    if not values["startstep"] <= values["maxstep"] <= values["max_time"]:
        raise ValueError("run controls must satisfy startstep <= maxstep <= max_time")
    if values["remesh_max_trigger_ratio"] < 4.0:
        raise ValueError("run.remesh_max_trigger_ratio must be at least 4.0")
    if run["linear_solver"] != "superlu":
        raise ValueError("mapped production runs require run.linear_solver='superlu'")
    completion = case.get("completion", {})
    observable = completion.get("observable")
    if observable not in {"R_min", "accepted_steps"} or completion.get("operator") != ">=":
        raise ValueError(
            "mapped completion must be R_min>=threshold or accepted_steps>=threshold"
        )
    raw_threshold = completion.get("threshold")
    if observable == "accepted_steps":
        if (
            isinstance(raw_threshold, bool)
            or not isinstance(raw_threshold, int)
            or raw_threshold < 1
        ):
            raise ValueError("accepted_steps completion threshold must be a positive integer")
        completion_threshold: float | int = raw_threshold
    else:
        completion_threshold = float(raw_threshold)
        if not math.isfinite(completion_threshold) or completion_threshold <= 0.0:
            raise ValueError("R_min completion threshold must be positive and finite")
    return NumericalPolicy(
        values["max_time"], values["startstep"], values["maxstep"],
        "superlu", values["remesh_max_trigger_ratio"], observable,
        completion_threshold,
    )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"JSONL owner directory does not exist: {path.parent}")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_attempt(attempt: Path, case_path: Path) -> dict[str, Any]:
    if attempt.parent.name != "attempts" or attempt.parent.parent.name != "runs":
        raise ValueError("attempt must be stored under runs/attempts/<id>")
    for child in ("base", "segments", "runtime"):
        if not (attempt / child).is_dir():
            raise ValueError(f"attempt is missing its pre-created {child}/ directory")
    record = _load_json(attempt / "attempt.json")
    if record.get("schema") != "pyoomph-attempt-v1":
        raise ValueError("attempt.json has the wrong schema")
    manifest_link = record.get("manifest", {})
    manifest = Path(str(manifest_link.get("path", ""))).resolve()
    if not manifest.is_file() or _sha256(manifest) != manifest_link.get("sha256"):
        raise ValueError("attempt manifest is missing or changed")
    data = _load_json(manifest)
    if data.get("schema") != "pyoomph-run-manifest-v1":
        raise ValueError("attempt manifest has the wrong schema")
    if data.get("case_id") != _load_json(case_path).get("case_id"):
        raise ValueError("attempt manifest case_id differs from selected case")
    if Path(data["case"]["path"]).resolve() != case_path.resolve() or data["case"]["sha256"] != _sha256(case_path):
        raise ValueError("attempt manifest does not bind the selected case")
    runner = Path(__file__).resolve()
    if Path(data["runner"]["path"]).resolve() != runner or data["runner"]["sha256"] != _sha256(runner):
        raise ValueError("attempt manifest does not bind this mapped runner")
    return record


def _segment_owner(attempt: Path, segment_id: str) -> tuple[Path, Path | None]:
    if segment_id == "base":
        return attempt / "base", None
    owner = attempt / "segments" / segment_id
    restart = _load_json(owner / "restart.json")
    if restart.get("schema") != "pyoomph-restart-segment-v1" or restart.get("segment_id") != segment_id:
        raise ValueError("restart segment receipt is invalid")
    checkpoint = Path(str(restart.get("checkpoint", {}).get("path", ""))).resolve()
    if not checkpoint.is_file() or _sha256(checkpoint) != restart.get("checkpoint", {}).get("sha256"):
        raise ValueError("restart checkpoint is missing or changed")
    parent = str(restart.get("parent", ""))
    parent_owner = attempt / "base" if parent == "base" else attempt / "segments" / parent
    checkpoint.relative_to(parent_owner.resolve())
    seen = {segment_id}
    while parent != "base":
        if parent in seen:
            raise ValueError("restart parent chain contains a cycle")
        seen.add(parent)
        parent_record = _load_json(attempt / "segments" / parent / "restart.json")
        if (
            parent_record.get("schema") != "pyoomph-restart-segment-v1"
            or parent_record.get("segment_id") != parent
        ):
            raise ValueError("restart parent chain is invalid")
        parent_checkpoint = Path(
            str(parent_record.get("checkpoint", {}).get("path", ""))
        ).resolve()
        if (
            not parent_checkpoint.is_file()
            or _sha256(parent_checkpoint)
            != parent_record.get("checkpoint", {}).get("sha256")
        ):
            raise ValueError("restart parent checkpoint is missing or changed")
        grandparent = str(parent_record.get("parent", ""))
        grandparent_owner = (
            attempt / "base"
            if grandparent == "base"
            else attempt / "segments" / grandparent
        )
        parent_checkpoint.relative_to(grandparent_owner.resolve())
        parent = grandparent
    return owner, checkpoint


def _write_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--segment-id", required=True)
    parser.add_argument("--mesh-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case_path = args.case.resolve()
    case = load_case(case_path)
    policy = numerical_policy(case)
    attempt_dir = args.attempt_dir.resolve()
    _validate_attempt(attempt_dir, case_path)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.segment_id):
        raise ValueError("segment-id contains unsupported characters")
    segment_dir, restart_state = _segment_owner(attempt_dir, args.segment_id)
    if not segment_dir.is_dir():
        raise FileNotFoundError(f"pre-created segment directory is missing: {segment_dir}")
    allowed = {"restart.json"} if args.segment_id != "base" else set()
    unexpected = sorted(path.name for path in segment_dir.iterdir() if path.name not in allowed)
    if unexpected:
        raise FileExistsError(f"segment already contains artifacts: {unexpected}")
    progress = attempt_dir / "runtime" / "progress.jsonl"
    _append_jsonl(progress, {
        "event": "segment-start",
        "case_id": case["case_id"],
        "segment_id": args.segment_id,
        "case_sha256": _sha256(case_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
    })
    _write_exclusive(
        segment_dir / "case.json",
        json.dumps(case, indent=2, sort_keys=True) + "\n",
    )
    _write_exclusive(
        segment_dir / "runner.json",
        json.dumps({
            "schema": "mapped-runner-effective-v1",
            "case_sha256": _sha256(case_path),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "segment": args.segment_id,
        }, indent=2, sort_keys=True) + "\n",
    )

    with MappedCoalescenceProblem(case, segment_dir) as problem:
        problem.set_linear_solver(policy.linear_solver)
        if policy.completion_observable == "R_min":
            problem.stop_rmin = float(policy.completion_threshold)
        problem.set_progress_callback(
            lambda record: _append_jsonl(progress, {
                "segment_id": args.segment_id, **record
            })
        )
        checkpoint_sequence = {"value": 0}

        def checkpoint(event: dict[str, Any]) -> None:
            sequence = checkpoint_sequence["value"]
            target = segment_dir / f"checkpoint-{sequence:06d}.state"
            if target.exists():
                raise FileExistsError(f"checkpoint already exists: {target}")
            problem.save_state(str(target), quiet=True)
            digest = _sha256(target)
            _append_jsonl(segment_dir / "fields.jsonl", {
                "schema": "pyoomph-field-event-v1",
                "segment": args.segment_id,
                "time": float(event["physical_time"]),
                "field": "state",
                "path": str(target.resolve()),
                "sha256": digest,
            })
            _append_jsonl(progress, {
                "event": "checkpoint",
                "case_id": case["case_id"],
                "segment_id": args.segment_id,
                "physical_time": float(event["physical_time"]),
                "path": str(target.resolve()),
                "sha256": digest,
                "reason": event["kind"],
            })
            checkpoint_sequence["value"] += 1

        problem.set_checkpoint_callback(checkpoint)
        if bool(case["mesh"]["remeshing"]):
            problem.configure_automatic_remeshing(
                segment_dir / "mapped-remesh-receipts.jsonl",
                max_trigger_ratio=policy.remesh_max_trigger_ratio,
            )
        if restart_state is not None:
            problem.initialise()
            problem.load_state(str(restart_state), quiet=True)
            if bool(case["mesh"]["remeshing"]) and problem.automatic_remesh_state != "armed":
                raise RuntimeError(
                    "restart state lacks a matching accepted-step remesh calibration"
                )
        if args.mesh_only:
            if not problem.is_initialised():
                problem.initialise()
            row = problem.write_neck_row(profile=True)
            mesh = problem.get_mesh("drop")
            print(json.dumps({
                "R_min": row["R_min"],
                "u_min": row["u_min"],
                "abs_2H": row["abs_2H"],
                "Z_b": row["Z_b"],
                "ndof": int(problem.ndof()),
                "nelement": int(mesh.nelement()),
                "automatic_remesh_state": problem.automatic_remesh_state,
            }, sort_keys=True))
            return 0
        problem.run_until(
            max_time=policy.max_time,
            startstep=policy.startstep,
            maxstep=policy.maxstep,
            completion_observable=policy.completion_observable,
            completion_threshold=policy.completion_threshold,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
