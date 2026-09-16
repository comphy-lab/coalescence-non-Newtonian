"""Anthony, Harris & Basaran (2020) initial geometry and Eq. 6 dividing ellipse.

Drop radius is one. The simulated interface is Eq. 2, not the kissing parabolas
of their Eq. 1. Approximate point contact is ``Z0 = R0**2 / 2``. That parabolic
matching leaves the unit sphere at ``(0, 1)`` a distance ``O(R0**3)`` outside
the bridge circle, so the two curves do not meet. The mesh uses the unique unit
sphere that is externally tangent to Eq. 2: centre ``(0, z_c)`` with
``z_c = sqrt((1 + Z0)**2 - (R0 + Z0)**2)``, which differs from ``1`` by
``O(R0**3)``. The junction is then C1.

Eq. 6 is four independent conditions on a five-parameter ellipse (two points
and two slopes). The remaining degree of freedom is fixed by taking the member
of the one-parameter family with the most negative quadratic discriminant,
which is the plump ellipse that matches Fig. 2. The domain here is the upper
quadrant; Fig. 2 is the lower drop, so the axis slope is ``-tan(pi/3)``.
"""

from __future__ import annotations

import math


# Discriminant-maximising member of the Eq. 6 family (B = beta / R_min**2).
EQ6_BETA = -0.9
EQ6_AXIS_SLOPE = math.tan(math.pi / 3.0)


def approximate_point_contact_z0(r0: float) -> float:
    return 0.5 * r0 * r0


def bridge_centre(r0: float, z0: float) -> tuple[float, float]:
    return (r0 + z0, 0.0)


def sphere_centre(r0: float, z0: float) -> tuple[float, float]:
    """Centre of the unit sphere that is externally tangent to the bridge."""
    if r0 <= 0.0 or z0 <= 0.0:
        raise ValueError("R0 and Z0 must be positive")
    c = r0 + z0
    rad = 1.0 + z0
    disc = rad * rad - c * c
    if disc <= 0.0:
        raise ValueError("bridge circle is too far from the axis for a unit sphere")
    return (0.0, math.sqrt(disc))


def sphere_bridge_junction(r0: float, z0: float) -> tuple[float, float]:
    """Return the first-quadrant C1 join of the tangent unit sphere and Eq. 2."""
    if r0 <= 0.0 or z0 <= 0.0:
        raise ValueError("R0 and Z0 must be positive")
    c = r0 + z0
    _, z_c = sphere_centre(r0, z0)
    scale = 1.0 / (1.0 + z0)
    r = c * scale
    z = z_c * (1.0 - scale)
    if r <= r0 or z <= 0.0:
        raise ValueError(f"junction is not in the first-quadrant bridge, got {(r, z)}")
    sphere_res = r * r + (z - z_c) ** 2 - 1.0
    bridge_res = (r - c) ** 2 + z * z - z0 * z0
    if max(abs(sphere_res), abs(bridge_res)) > 1e-12:
        raise ValueError("junction does not lie on both curves")
    return r, z


def eq6_axis_point(rmin: float) -> tuple[float, float]:
    """Eq. 6a–b for the upper drop: ``(r1, z1) = (0, R_min)`` if ``R_min < 1/2``."""
    if rmin <= 0.0:
        raise ValueError("R_min must be positive")
    z1 = rmin if rmin < 0.5 else 0.5
    return (0.0, z1)


def eq6_plane_point(rmin: float) -> tuple[float, float]:
    """Eq. 6d–e: ``(r2, z2) = (2 R_min, 0)``."""
    if rmin <= 0.0:
        raise ValueError("R_min must be positive")
    return (2.0 * rmin, 0.0)


def eq6_conic_coeffs(rmin: float, *, beta: float = EQ6_BETA) -> tuple[float, float, float, float, float]:
    """Quadratic ``A r^2 + B r z + C z^2 + D r + E z + 1 = 0`` for the upper drop.

    Mirror of the lower-drop family that satisfies Eq. 6 with ``z -> -z``.
    """
    if rmin <= 0.0:
        raise ValueError("R_min must be positive")
    r2 = 2.0 * rmin
    inv = 1.0 / (rmin * rmin)
    b_lower = beta * inv
    root3 = math.sqrt(3.0)
    a = (b_lower * (2.0 - root3) + (4.0 - root3) * 0.5 * inv) / (2.0 * root3)
    c = -2.0 * b_lower - inv
    d = -2.0 * a * rmin - 1.0 / r2
    e_lower = -2.0 * b_lower * rmin
    # Upper drop: B and E flip with z -> -z.
    return a, -b_lower, c, d, -e_lower


def eq6_conic_value(
    r: float, z: float, coeffs: tuple[float, float, float, float, float]
) -> float:
    a, b, c, d, e = coeffs
    return a * r * r + b * r * z + c * z * z + d * r + e * z + 1.0


def eq6_conic_slope(
    r: float, z: float, coeffs: tuple[float, float, float, float, float]
) -> float:
    """``dr/dz`` on the conic."""
    a, b, c, d, e = coeffs
    num = -(b * r + 2.0 * c * z + e)
    den = 2.0 * a * r + b * z + d
    if abs(den) < 1e-18:
        return math.copysign(float("inf"), num) if num != 0.0 else 0.0
    return num / den


def _eq6_z_roots(r: float, coeffs: tuple[float, float, float, float, float]) -> list[float]:
    a, b, c, d, e = coeffs
    aa = c
    bb = b * r + e
    cc = a * r * r + d * r + 1.0
    if abs(aa) < 1e-18:
        if abs(bb) < 1e-18:
            return []
        return [-cc / bb]
    disc = bb * bb - 4.0 * aa * cc
    if disc < 0.0 and disc > -1e-16:
        disc = 0.0
    if disc < 0.0:
        return []
    s = math.sqrt(disc)
    return [(-bb - s) / (2.0 * aa), (-bb + s) / (2.0 * aa)]


def eq6_ellipse_points(
    rmin: float, n: int = 48, *, beta: float = EQ6_BETA
) -> list[tuple[float, float]]:
    """Sample the upper-drop Eq. 6 arc from the axis point to ``(2 R_min, 0)``."""
    if n < 4:
        raise ValueError("need at least four samples")
    coeffs = eq6_conic_coeffs(rmin, beta=beta)
    _, z1 = eq6_axis_point(rmin)
    r2, _ = eq6_plane_point(rmin)
    points: list[tuple[float, float]] = [(0.0, z1)]
    prev_z = z1
    for i in range(1, n - 1):
        r = r2 * (i / (n - 1))
        roots = [z for z in _eq6_z_roots(r, coeffs) if z >= -1e-18]
        if not roots:
            continue
        z = min(roots, key=lambda zz: abs(zz - prev_z))
        points.append((r, max(z, 0.0)))
        prev_z = z
    points.append((r2, 0.0))
    return points


def r_free_surface(z: float, r0: float, z0: float) -> float:
    """Radial coordinate of the t = 0 interface at height ``z`` (upper drop)."""
    if z < 0.0:
        raise ValueError("upper drop has z >= 0")
    _, z_c = sphere_centre(r0, z0)
    if z <= z0:
        gap2 = z0 * z0 - z * z
        if gap2 < 0.0:
            gap2 = 0.0
        return (r0 + z0) - math.sqrt(gap2)
    span2 = 1.0 - (z - z_c) ** 2
    if span2 <= 0.0:
        return 0.0
    return math.sqrt(span2)


def point_in_initial_drop(r: float, z: float, r0: float, z0: float) -> bool:
    if r < -1e-18 or z < -1e-18:
        return False
    return r <= r_free_surface(max(z, 0.0), r0, z0) + 1e-12


def _on_segment(p0: tuple[float, float], p1: tuple[float, float], t: float) -> tuple[float, float]:
    return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))


def clip_ellipse_to_initial_interface(
    rmin: float, r0: float, z0: float, n: int = 64
) -> list[tuple[float, float]]:
    """Eq. 6 arc from the axis through the liquid, stopping on the free surface."""
    raw = eq6_ellipse_points(rmin, n=n)
    clipped: list[tuple[float, float]] = []
    prev: tuple[float, float] | None = None
    prev_inside = False
    for r, z in raw:
        inside = point_in_initial_drop(r, z, r0, z0)
        if inside:
            clipped.append((r, z))
            prev = (r, z)
            prev_inside = True
            continue
        if prev_inside and prev is not None:
            lo, hi = 0.0, 1.0
            p_out = (r, z)
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                p = _on_segment(prev, p_out, mid)
                if point_in_initial_drop(p[0], p[1], r0, z0):
                    lo = mid
                else:
                    hi = mid
            clipped.append(_on_segment(prev, p_out, 0.5 * (lo + hi)))
        break
    if len(clipped) < 3:
        raise RuntimeError("Eq. 6 ellipse does not meet the initial free surface")
    return clipped


def _seg_intersect(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> tuple[float, float] | None:
    """Open-segment intersection in the r–z plane, or None."""
    ax, ay = a1[0] - a0[0], a1[1] - a0[1]
    bx, by = b1[0] - b0[0], b1[1] - b0[1]
    den = ax * by - ay * bx
    if abs(den) < 1e-30:
        return None
    dx, dy = b0[0] - a0[0], b0[1] - a0[1]
    t = (dx * by - dy * bx) / den
    u = (dx * ay - dy * ax) / den
    if t < -1e-12 or t > 1.0 + 1e-12 or u < -1e-12 or u > 1.0 + 1e-12:
        return None
    t = min(max(t, 0.0), 1.0)
    return (a0[0] + t * ax, a0[1] + t * ay)


def clip_ellipse_to_polyline(
    ellipse: list[tuple[float, float]], interface: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Keep ellipse samples until they cross the free-surface polyline (r, z)."""
    if len(interface) < 2 or len(ellipse) < 2:
        raise ValueError("need polylines")
    for i in range(len(ellipse) - 1):
        for j in range(len(interface) - 1):
            hit = _seg_intersect(ellipse[i], ellipse[i + 1], interface[j], interface[j + 1])
            if hit is None:
                continue
            return ellipse[: i + 1] + [hit]
    raise RuntimeError("Eq. 6 ellipse does not meet the current free surface")


def resample_polyline(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Uniform arclength resampling, keeping the endpoints."""
    if n < 2:
        raise ValueError("need at least two samples")
    if len(points) <= n:
        return list(points)
    s = [0.0]
    for i in range(1, len(points)):
        s.append(
            s[-1]
            + math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        )
    total = s[-1]
    if total <= 0.0:
        return [points[0], points[-1]]
    out = [points[0]]
    j = 0
    for k in range(1, n - 1):
        target = total * k / (n - 1)
        while j + 1 < len(s) and s[j + 1] < target:
            j += 1
        if j + 1 >= len(s):
            break
        span = s[j + 1] - s[j]
        t = 0.0 if span <= 0.0 else (target - s[j]) / span
        p0, p1 = points[j], points[j + 1]
        out.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))
    if hypot_sq(points[-1], out[-1]) > 0.0:
        out.append(points[-1])
    return out


def hypot_sq(p: tuple[float, float], q: tuple[float, float]) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def decimate_polyline(
    points: list[tuple[float, float]],
    min_ds: float,
    *,
    keep_head: int = 4,
) -> list[tuple[float, float]]:
    """Keep the neck, then samples at least ``min_ds`` apart. Splines of every
    quadratic node self-intersect in gmsh once the meniscus tightens."""
    if len(points) <= 2:
        return list(points)
    keep_head = max(1, min(keep_head, len(points) - 1))
    out = [points[0]]
    for p in points[1:keep_head]:
        if hypot_sq(p, out[-1]) > 0.0:
            out.append(p)
    acc = 0.0
    prev = out[-1]
    for p in points[keep_head:-1]:
        ds = math.hypot(p[0] - prev[0], p[1] - prev[1])
        acc += ds
        prev = p
        if acc >= min_ds and hypot_sq(p, out[-1]) > 0.0:
            out.append(p)
            acc = 0.0
    if hypot_sq(points[-1], out[-1]) > 0.0:
        out.append(points[-1])
    if len(out) < 2:
        return [points[0], points[-1]]
    return out


def split_polyline_at_r(
    points: list[tuple[float, float]], r_split: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Split an interface polyline (neck -> pole, increasing z) at radial station ``r_split``."""
    if len(points) < 2:
        raise ValueError("need a polyline")
    left: list[tuple[float, float]] = [points[0]]
    hit: tuple[float, float] | None = None
    rest_start = 1
    for i in range(1, len(points)):
        r0, z0 = points[i - 1]
        r1, z1 = points[i]
        crossed = (r0 - r_split) * (r1 - r_split) <= 0.0 and r0 != r1
        if hit is None and crossed:
            t = (r_split - r0) / (r1 - r0)
            hit = (r_split, z0 + t * (z1 - z0))
            left.append(hit)
            rest_start = i
            break
        left.append((r1, z1))
    if hit is None:
        # Fall back: nearest point.
        j = min(range(len(points)), key=lambda k: abs(points[k][0] - r_split))
        hit = points[j]
        left = points[: j + 1]
        right = points[j:]
        return left, right
    right = [hit] + points[rest_start:]
    return left, right


def meridional_curvature_at_neck(points: list[tuple[float, float]]) -> float:
    """Three-point curvature in the r–z plane at the first (neck) sample."""
    if len(points) < 3:
        return float("nan")
    (r0, z0), (r1, z1), (r2, z2) = points[0], points[1], points[2]
    a = math.hypot(r1 - r0, z1 - z0)
    b = math.hypot(r2 - r1, z2 - z1)
    c = math.hypot(r2 - r0, z2 - z0)
    if a * b * c < 1e-30:
        return float("nan")
    area2 = abs((r1 - r0) * (z2 - z0) - (z1 - z0) * (r2 - r0))
    if area2 < 1e-30:
        return 0.0
    return 4.0 * (0.5 * area2) / (a * b * c)


def twice_mean_curvature_at_neck(points: list[tuple[float, float]]) -> float:
    """``|2H|`` at the neck: meridional curvature plus hoop ``1/R_min``.

    Outward normal at the neck is ``+r``, so both principal curvatures add.
    """
    if not points:
        return float("nan")
    rmin = points[0][0]
    kappa_m = meridional_curvature_at_neck(points)
    if not math.isfinite(kappa_m) or rmin <= 0.0:
        return float("nan")
    return abs(kappa_m + 1.0 / rmin)


def eggers_stokes_rmin(tau_v: float) -> float:
    """Eggers, Lister & Stone 1999, valid for ``R_min < 0.03``."""
    if tau_v <= 0.0:
        return float("nan")
    return -(tau_v / math.pi) * math.log(tau_v)


def eggers_tau_for_rmin(rmin: float) -> float:
    """Invert Eggers ``R = -(τ/π) ln τ`` for the viscocapillary time origin."""
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


def viscocapillary_velocity_slope() -> float:
    return 1.0 / math.pi
