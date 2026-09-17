"""Fig. 3 coordinates of Anthony, Harris & Basaran (2020): R_min(tau_v) and u_v(R_min)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Computer Modern Roman"]
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def read_neck(
    path: Path, *, terminal_segment: str | None = None
) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".jsonl":
        return read_progress_neck(path, terminal_segment=terminal_segment)
    if terminal_segment is not None:
        raise ValueError("terminal_segment applies only to attempt progress JSONL")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return {key: np.array([float(row[key]) for row in rows], dtype=float) for key in rows[0]}


def _validated_progress_chain(
    path: Path, segment_ids: set[str], terminal_segment: str | None
) -> tuple[str, ...]:
    if not segment_ids or any(not segment for segment in segment_ids):
        raise ValueError("state progress records require non-empty segment_id values")
    if segment_ids == {"base"}:
        if terminal_segment not in (None, "base"):
            raise ValueError(f"terminal segment is absent from progress: {terminal_segment}")
        return ("base",)
    attempt = path.parent.parent
    if path != attempt / "runtime" / "progress.jsonl":
        raise ValueError("multi-segment progress must remain under its attempt/runtime owner")
    parents: dict[str, str] = {}
    for segment in sorted(segment_ids - {"base"}):
        receipt_path = attempt / "segments" / segment / "restart.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid restart receipt for segment {segment}") from exc
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != "pyoomph-restart-segment-v1"
            or receipt.get("segment_id") != segment
        ):
            raise ValueError(f"invalid restart receipt for segment {segment}")
        parent = str(receipt.get("parent", ""))
        if parent not in segment_ids:
            raise ValueError(f"restart parent {parent!r} is absent from progress")
        checkpoint = receipt.get("checkpoint", {})
        checkpoint_path = Path(str(checkpoint.get("path", ""))).resolve()
        if (
            not checkpoint_path.is_file()
            or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            != checkpoint.get("sha256")
        ):
            raise ValueError(f"restart checkpoint for segment {segment} is missing or changed")
        parent_owner = attempt / "base" if parent == "base" else attempt / "segments" / parent
        try:
            checkpoint_path.relative_to(parent_owner.resolve())
        except ValueError as exc:
            raise ValueError(f"restart checkpoint for segment {segment} has the wrong owner") from exc
        parents[segment] = parent
    if terminal_segment is None:
        leaves = segment_ids - set(parents.values())
        if len(leaves) != 1:
            raise ValueError(
                "progress contains multiple restart branches; select terminal_segment"
            )
        terminal_segment = next(iter(leaves))
    if terminal_segment not in segment_ids:
        raise ValueError(f"terminal segment is absent from progress: {terminal_segment}")
    reverse = []
    current = terminal_segment
    seen: set[str] = set()
    while current != "base":
        if current in seen or current not in parents:
            raise ValueError("terminal restart chain is cyclic or incomplete")
        seen.add(current)
        reverse.append(current)
        current = parents[current]
    return ("base", *reversed(reverse))


def read_progress_neck(
    path: Path, *, terminal_segment: str | None = None
) -> dict[str, np.ndarray]:
    """Read accepted-step diagnostics with restart precedence.

    The run-owned progress stream is append-only.  A restart segment can
    repeat its parent's checkpoint time; the later record wins for that exact
    physical time, matching the segment-precedence rule used for fields.
    """
    path = path.resolve()
    required = ("physical_time", "R_min", "u_min", "Z_b", "abs_2H", "V_bod")
    state_records: list[tuple[int, dict[str, object]]] = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            if number == len(lines) and not line.endswith(("\n", "\r")):
                # A live append-only writer can be observed between its
                # write and newline/fsync.  Ignore only that incomplete
                # tail; malformed complete or interior rows remain fatal.
                break
            raise ValueError(f"invalid progress JSON at {path}:{number}") from exc
        if not isinstance(record, dict) or record.get("event") not in {
            "initial-state", "restored-segment-state", "accepted-step"
        }:
            continue
        state_records.append((number, record))
    segment_ids = {str(record.get("segment_id", "")) for _, record in state_records}
    chain = set(_validated_progress_chain(path, segment_ids, terminal_segment))
    by_time: dict[float, dict[str, float]] = {}
    for number, record in state_records:
        if str(record.get("segment_id", "")) not in chain:
            continue
        if any(name not in record for name in required):
            raise ValueError(f"state progress record lacks Fig. 3 fields at line {number}")
        values = {name: float(record[name]) for name in required}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite state progress record at line {number}")
        by_time[values["physical_time"]] = values
    if not by_time:
        raise ValueError(f"no state diagnostics in {path}")
    rows = [by_time[time] for time in sorted(by_time)]
    return {
        "t": np.array([row["physical_time"] for row in rows], dtype=float),
        "R_min": np.array([row["R_min"] for row in rows], dtype=float),
        "u_min": np.array([row["u_min"] for row in rows], dtype=float),
        "Z_b": np.array([row["Z_b"] for row in rows], dtype=float),
        "abs_2H": np.array([row["abs_2H"] for row in rows], dtype=float),
        "V_bod": np.array([row["V_bod"] for row in rows], dtype=float),
    }


def eggers_tau_for_rmin(rmin: float) -> float:
    """Invert Eggers so the first point can be placed on the theoretical time origin."""
    if rmin <= 0.0 or rmin >= 0.1:
        return 0.0
    lo, hi = 1e-16, 1.0 / math.e
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        val = -(mid / math.pi) * math.log(mid)
        if val > rmin:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


@dataclass(frozen=True)
class ContactTimeFit:
    """Power-law extrapolation receipt for ``R_min=A(t+t_con)^beta``."""

    t_con: float
    exponent: float
    prefactor: float
    log_rms: float
    point_count: int
    first_radius: float
    last_radius: float


def _validate_growth_series(
    t: np.ndarray, rmin: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    rmin = np.asarray(rmin, dtype=float)
    if t.ndim != 1 or rmin.ndim != 1 or t.size != rmin.size or t.size < 2:
        raise ValueError("time and R_min must be equal-length one-dimensional series")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(rmin)):
        raise ValueError("time and R_min must be finite")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("simulation time must increase strictly")
    if t[0] < 0.0 or np.any(rmin <= 0.0):
        raise ValueError("time must be non-negative and R_min must be positive")
    return t, rmin


def fit_contact_time(t: np.ndarray, rmin: np.ndarray) -> ContactTimeFit:
    """Fit Anthony et al.'s contact-time shift after one decade of growth.

    The fitted model is ``R_min=A(t+t_con)^beta`` with ``t_con>=0`` and
    ``beta>0``.  For each trial shift, ``A`` and ``beta`` are solved by linear
    least squares in logarithmic coordinates; a bounded scalar minimisation
    then selects the shift.  A shallower trajectory is refused rather than
    relabelled with the Eggers inversion, which is only a theoretical time
    origin and not the paper's fitted contact time.
    """
    t, rmin = _validate_growth_series(t, rmin)
    threshold = 10.0 * float(rmin[0])
    reached = np.flatnonzero(rmin >= threshold)
    if reached.size == 0:
        raise ValueError(
            "t_con requires R_min to have grown by one decade"
        )
    # Anthony et al. say that the data are fit once the bridge has grown by an
    # order of magnitude.  Use that completed first-decade trajectory, from
    # the finite initial bridge through the first threshold crossing.  Fitting
    # only later samples would make t_con depend on how long and how densely a
    # run happened to continue after satisfying the stated criterion.
    fit_indices = np.arange(int(reached[0]) + 1)
    if fit_indices.size < 4:
        raise ValueError(
            "t_con requires at least four samples across the first decade of R_min growth"
        )
    t_fit = t[fit_indices]
    r_fit = rmin[fit_indices]
    log_r = np.log(r_fit)
    scale = max(float(t_fit[-1]), float(t_fit[-1] - t_fit[0]), np.finfo(float).tiny)
    upper = 100.0 * scale

    def linear_fit(shift: float) -> tuple[float, float, float]:
        shifted = t_fit + float(shift)
        if np.any(shifted <= 0.0):
            return float("inf"), float("nan"), float("nan")
        log_t = np.log(shifted)
        exponent, log_prefactor = np.polyfit(log_t, log_r, 1)
        if not math.isfinite(exponent) or exponent <= 0.0:
            return float("inf"), float(exponent), float(log_prefactor)
        residual = log_r - (log_prefactor + exponent * log_t)
        return float(np.mean(residual * residual)), float(exponent), float(log_prefactor)

    result = minimize_scalar(
        lambda shift: linear_fit(float(shift))[0],
        bounds=(0.0, upper),
        method="bounded",
        options={"xatol": max(1.0e-15, 1.0e-12 * scale)},
    )
    candidates = [(0.0, linear_fit(0.0)), (float(result.x), linear_fit(float(result.x)))]
    shift, (mean_square, exponent, log_prefactor) = min(
        candidates, key=lambda item: item[1][0]
    )
    if not result.success or not math.isfinite(mean_square):
        raise ValueError("power-law contact-time fit did not converge")
    if shift >= 0.99 * upper:
        raise ValueError("power-law contact-time shift is not identifiable from this trajectory")
    return ContactTimeFit(
        t_con=float(shift),
        exponent=exponent,
        prefactor=float(math.exp(log_prefactor)),
        log_rms=float(math.sqrt(mean_square)),
        point_count=int(fit_indices.size),
        first_radius=float(r_fit[0]),
        last_radius=float(r_fit[-1]),
    )


def contact_time(t: np.ndarray, rmin: np.ndarray) -> float:
    """Return the paper-compatible fitted contact-time shift."""
    return fit_contact_time(t, rmin).t_con


def fig3_series(neck: dict[str, np.ndarray], *, inertia: bool, oh: float | None) -> dict[str, np.ndarray]:
    fit = fit_contact_time(neck["t"], neck["R_min"])
    t_con = fit.t_con
    tau = neck["t"] + t_con
    if inertia:
        if oh is None:
            raise ValueError("Oh required for an inertial series")
        tau_v = tau / oh
        u_v = neck["u_min"] * oh
    else:
        tau_v = tau
        u_v = neck["u_min"]
    return {
        "tau_v": tau_v,
        "R_min": neck["R_min"],
        "u_v": u_v,
        "t_con": np.full_like(tau_v, t_con),
        "fit_exponent": np.full_like(tau_v, fit.exponent),
        "fit_log_rms": np.full_like(tau_v, fit.log_rms),
    }


def eggers(tau_v: np.ndarray) -> np.ndarray:
    tau = np.clip(tau_v, 1e-16, 0.999)
    return -(tau / math.pi) * np.log(tau)


def style(ax) -> None:
    ax.tick_params(which="both", direction="out", width=3, labelsize=30, pad=10)
    ax.tick_params(which="major", length=12)
    ax.tick_params(which="minor", length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    ax.minorticks_on()


def plot_fig3(series: list[tuple[str, dict[str, np.ndarray]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_r, ax_u = axes
    colours = {"Stokes": "tab:blue", "Oh = 0.6": "tab:red"}
    tau_min = min(float(np.min(item[1]["tau_v"])) for item in series)
    tau_max = max(float(np.max(item[1]["tau_v"])) for item in series)
    tau_th = np.logspace(math.log10(max(tau_min * 0.3, 1e-8)), math.log10(min(max(tau_max, 0.03), 0.3)), 400)
    ax_r.loglog(tau_th, eggers(tau_th), "k--", lw=2, label=r"$R_{\min}=-(\tau_v/\pi)\ln\tau_v$")
    ax_r.loglog(tau_th, tau_th, "k:", lw=2, label=r"$R_{\min}=\tau_v$")
    for name, data in series:
        ax_r.loglog(
            data["tau_v"],
            data["R_min"],
            "o",
            ms=5,
            color=colours.get(name, "tab:gray"),
            label=name,
        )
        ax_u.semilogx(
            data["R_min"],
            data["u_v"],
            "o",
            ms=5,
            color=colours.get(name, "tab:gray"),
            label=name,
        )
    r_line = np.logspace(-6, 0, 400)
    ax_u.semilogx(r_line, -(1.0 / math.pi) * np.log(r_line), "k-", lw=2, label=r"$u_v=-(1/\pi)\ln R_{\min}$")
    ax_r.set_xlabel(r"$\tau_v$", fontsize=40, labelpad=15)
    ax_r.set_ylabel(r"$R_{\min}$", fontsize=40, labelpad=15)
    ax_u.set_xlabel(r"$R_{\min}$", fontsize=40, labelpad=15)
    ax_u.set_ylabel(r"$u_v$", fontsize=40, labelpad=15)
    ax_r.set_xlim(tau_th[0], tau_th[-1])
    ax_r.set_ylim(1e-6, 1.0)
    ax_u.set_xlim(1e-6, 1.0)
    ax_r.set_box_aspect(1)
    ax_u.set_box_aspect(1)
    style(ax_r)
    style(ax_u)
    ax_r.legend(fontsize=18, frameon=False)
    ax_u.legend(fontsize=18, frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", dpi=300, pad_inches=0.1)
    if output.suffix.lower() != ".pdf":
        fig.savefig(output, bbox_inches="tight", dpi=300, pad_inches=0.1)
    plt.close(fig)


def parse_run(spec: str) -> tuple[str, Path, bool, float | None, str | None]:
    name, path, kind, *rest = spec.split(":")
    inertia = kind == "inertia"
    if kind not in {"inertia", "stokes"}:
        raise ValueError(f"unknown run kind: {kind}")
    oh = float(rest[0]) if inertia and rest else None
    terminal = (rest[1] if len(rest) > 1 else None) if inertia else (rest[0] if rest else None)
    return name, Path(path), inertia, oh, terminal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=("name:neck.csv:stokes or name:progress.jsonl:stokes:terminal-segment; "
              "finite Oh uses name:path:inertia:0.6[:terminal-segment]"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    series = []
    for spec in args.run:
        name, path, inertia, oh, terminal = parse_run(spec)
        data = fig3_series(
            read_neck(path, terminal_segment=terminal), inertia=inertia, oh=oh
        )
        print(
            f"{name}: t_con={data['t_con'][0]:.16e} "
            f"beta={data['fit_exponent'][0]:.10g} "
            f"log_rms={data['fit_log_rms'][0]:.3e}"
        )
        series.append((name, data))
    plot_fig3(series, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
