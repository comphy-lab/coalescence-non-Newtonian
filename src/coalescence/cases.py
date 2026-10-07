"""Validation of ``pyoomph-case-v1`` case files before a run.

A case supplies the physics of one coalescence: Stokes flow or a finite Ohnesorge number,
the initial bridge and the exterior. Every runner validates the whole case before it
creates any output, so a malformed case fails before a run starts rather than part-way.
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA = "pyoomph-case-v1"


def _number(value: Any) -> bool:
    """A finite int or float; booleans and integers too large for a float are not numbers here."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _positive(value: Any) -> bool:
    return _number(value) and value > 0


def case_problems(case: Any, *, stokes_only: bool = False) -> list[str]:
    """Every way ``case`` departs from the schema the runners support; empty when it is valid."""
    if not isinstance(case, dict):
        return ["the case is not a JSON object"]
    problems = []
    if case.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    if not isinstance(case.get("case_id"), str) or not case["case_id"]:
        problems.append("case_id must be a non-empty string")
    geometry = case.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("kind") != "axisymmetric":
        problems.append("geometry.kind must be 'axisymmetric'")
    physics = case.get("physics")
    if not isinstance(physics, dict):
        return problems + ["physics must be an object"]
    exterior = physics.get("exterior")
    if (not isinstance(exterior, dict) or exterior.get("kind") != "passive"
            or not all(_number(exterior.get(k)) and exterior[k] == 0 for k in ("density", "viscosity"))):
        problems.append("physics.exterior must be passive, with zero density and viscosity")
    inertia = physics.get("inertia")
    if not isinstance(inertia, bool):
        problems.append("physics.inertia must be true or false")
    elif inertia:
        if stokes_only:
            problems.append("this solver is for the Stokes limit only (physics.inertia false)")
        if not _positive(physics.get("Oh")):
            problems.append("physics.Oh must be a positive number when physics.inertia is true")
        if physics.get("initial_velocity", "quiescent") != "quiescent":
            problems.append("an inertial case must start from rest (physics.initial_velocity 'quiescent')")
    bridge = physics.get("initial_bridge")
    if not isinstance(bridge, dict):
        problems.append("physics.initial_bridge must be an object")
    else:
        R0, Z0 = bridge.get("R0"), bridge.get("Z0")
        if not (_positive(R0) and R0 < 1):
            problems.append("physics.initial_bridge.R0 must lie in (0, 1)")
        if not _positive(Z0):
            problems.append("physics.initial_bridge.Z0 must be positive")
    return problems


def validate_case(case: Any, *, stokes_only: bool = False) -> None:
    """Refuse a case that :func:`case_problems` rejects, naming every problem."""
    problems = case_problems(case, stokes_only=stokes_only)
    if problems:
        raise SystemExit("invalid case: " + "; ".join(problems))
