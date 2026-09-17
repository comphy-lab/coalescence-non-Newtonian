"""Anthony, Harris & Basaran (2020) initial geometry and Eq. 6 divider.

Drop radius is one.  The initial interface is Anthony's Eq. 2 circular bridge
joined tangentially to a unit sphere.  The approximate point-contact choice is
``Z0 = R0**2 / 2``.  With this choice the translated sphere centre differs
from one by an ``O(R0**3)`` term; the small offset is retained in the formulas
below instead of being lost in a subtraction near unity.

The Eq. 6 divider used by the structured mesh is the axis-aligned translated
ellipse

``(r-c)**2/a**2 + z**2/b**2 = 1``

with ``H = min(Rmin, 1/2)``, ``m = sqrt(3)``,
``c = 4 Rmin**2/(4 Rmin + m H)``, ``a = 2 Rmin - c`` and
``b**2 = H a**2/(m c)``.  It is the upper-drop convention, so the slope at
the axis is ``+sqrt(3)``.  The old arbitrary ``beta`` family is not part of
the implementation; the keyword remains only to give an explicit migration
error to callers of the old API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Kept as a sentinel so importing older client code gives a useful value.  A
# non-None beta passed to a legacy function is rejected below rather than
# silently selecting an unsupported member of the old conic family.
EQ6_BETA = None
EQ6_AXIS_SLOPE = math.sqrt(3.0)


@dataclass(frozen=True)
class Eq6EllipseParameters:
    """Parameters of the translated ellipse used for the Eq. 6 divider."""

    rmin: float
    H: float
    m: float
    c: float
    a: float
    b: float
    b2: float


def _validate_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def approximate_point_contact_z0(r0: float) -> float:
    r0 = _validate_positive("R0", r0)
    return 0.5 * r0 * r0


def bridge_centre(r0: float, z0: float) -> tuple[float, float]:
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    return (r0 + z0, 0.0)


def _sphere_delta(r0: float, z0: float) -> float:
    """Return ``1 - ((1+Z0)^2 - (R0+Z0)^2)`` without a large subtraction."""
    # Expanding the two squares cancels Z0**2 exactly and leaves the quantity
    # whose leading term is R0**3 for Z0=R0**2/2.  Keeping this expression
    # explicit is important when the leading correction is below machine eps.
    return r0 * r0 + 2.0 * r0 * z0 - 2.0 * z0


def sphere_centre_offset(r0: float, z0: float) -> float:
    """Return ``1-z_c`` using the rationalised square-root identity.

    The returned value remains representable for ``R0=1e-6`` even though the
    corresponding ``z_c`` itself rounds to one in binary64.  For approximate
    point contact it is ``R0**3/2 + O(R0**4)``.
    """
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    delta = _sphere_delta(r0, z0)
    radicand = 1.0 - delta
    if radicand <= 0.0:
        raise ValueError("bridge circle is too far from the axis for a unit sphere")
    root = math.sqrt(radicand)
    # 1 - sqrt(1-delta) = delta/(1+sqrt(1-delta)); unlike the left-hand
    # expression this retains the O(R0**3) correction as a standalone value.
    return delta / (1.0 + root)


def sphere_centre(r0: float, z0: float) -> tuple[float, float]:
    """Centre of the unit sphere that is externally tangent to the bridge."""
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    # Use the stable offset for the calculation.  The explicit ``1-offset``
    # form documents and preserves the geometric O(R0**3) term even when the
    # final binary64 coordinate rounds to exactly 1.0.
    offset = sphere_centre_offset(r0, z0)
    return (0.0, 1.0 - offset)


def sphere_bridge_junction(r0: float, z0: float) -> tuple[float, float]:
    """Return the first-quadrant C1 join of the tangent unit sphere and Eq. 2."""
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    c = r0 + z0
    offset = sphere_centre_offset(r0, z0)
    z_c = 1.0 - offset
    inv = 1.0 / (1.0 + z0)
    r = c * inv
    # z_c * Z0/(1+Z0), with the small centre offset kept explicit.  Writing
    # Z0/(1+Z0) avoids the cancellation in 1-1/(1+Z0).
    z = (1.0 - offset) * z0 * inv
    if r <= r0 or z <= 0.0:
        raise ValueError(f"junction is not in the first-quadrant bridge, got {(r, z)}")
    sphere_res = r * r + (z - z_c) ** 2 - 1.0
    bridge_res = (r - c) ** 2 + z * z - z0 * z0
    if max(abs(sphere_res), abs(bridge_res)) > 1e-12:
        raise ValueError("junction does not lie on both curves")
    return r, z


def _rationalised_one_minus_sqrt_one_minus_square(r: float) -> float:
    """Stable ``1-sqrt(1-r*r)`` for ``0 <= r <= 1``."""
    r = float(r)
    if not math.isfinite(r) or r < 0.0 or r > 1.0:
        raise ValueError("radial coordinate must lie in [0, 1]")
    r2 = r * r
    root = math.sqrt(max(0.0, 1.0 - r2))
    return r2 / (1.0 + root)


def sphere_lower_surface_z(r: float, r0: float, z0: float) -> float:
    """Lower unit-sphere branch, retaining the tiny centre offset."""
    offset = sphere_centre_offset(r0, z0)
    return -offset + _rationalised_one_minus_sqrt_one_minus_square(r)


def sphere_upper_surface_z(r: float, r0: float, z0: float) -> float:
    """Upper unit-sphere branch, retaining the tiny centre offset."""
    offset = sphere_centre_offset(r0, z0)
    return 2.0 - offset - _rationalised_one_minus_sqrt_one_minus_square(r)


def eq2_bridge_value(r: float, z: float, r0: float, z0: float) -> float:
    """Signed Eq. 2 residual ``(r-c)^2+z^2-Z0^2``."""
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    c = r0 + z0
    return (r - c) * (r - c) + z * z - z0 * z0


def sphere_value(r: float, z: float, r0: float, z0: float) -> float:
    """Signed unit-sphere residual for the tangent initial interface."""
    _, zc = sphere_centre(r0, z0)
    return r * r + (z - zc) * (z - zc) - 1.0


def eq6_axis_point(rmin: float) -> tuple[float, float]:
    """Eq. 6a-b for the upper drop."""
    rmin = _validate_positive("R_min", rmin)
    return (0.0, min(rmin, 0.5))


def eq6_plane_point(rmin: float) -> tuple[float, float]:
    """Eq. 6d–e: ``(r2, z2) = (2 R_min, 0)``."""
    rmin = _validate_positive("R_min", rmin)
    return (2.0 * rmin, 0.0)


def eq6_ellipse_parameters(rmin: float) -> Eq6EllipseParameters:
    """Return the declared translated-ellipse reconstruction of Eq. 6."""
    rmin = _validate_positive("R_min", rmin)
    H = min(rmin, 0.5)
    m = EQ6_AXIS_SLOPE
    c = 4.0 * rmin * rmin / (4.0 * rmin + m * H)
    a = 2.0 * rmin - c
    if c <= 0.0 or a <= 0.0:
        raise ValueError("Eq. 6 ellipse has non-positive semiaxis")
    b2 = H * a * a / (m * c)
    return Eq6EllipseParameters(rmin=rmin, H=H, m=m, c=c, a=a, b=math.sqrt(b2), b2=b2)


def _reject_legacy_beta(beta: float | None) -> None:
    if beta is not None:
        raise ValueError("Eq. 6 beta is obsolete; use the translated ellipse")


def eq6_conic_coeffs(
    rmin: float, *, beta: float | None = EQ6_BETA
) -> tuple[float, float, float, float, float]:
    """Return normalized coefficients ``A r²+B r z+C z²+D r+E z+1=0``.

    This is the translated ellipse, expanded and normalized to unit constant
    term.  ``beta`` is accepted solely for source compatibility and any
    non-None value raises an explicit migration error.
    """
    _reject_legacy_beta(beta)
    p = eq6_ellipse_parameters(rmin)
    # (r-c)^2/a^2 + z^2/b^2 - 1 = 0.  Since a²-c² is positive, multiplying
    # by -a²/(a²-c²) gives the legacy unit-constant convention without any
    # cross term.  Factor a²-c² as (a-c)(a+c) to avoid unnecessary loss.
    den = (p.a - p.c) * (p.a + p.c)
    return (
        -1.0 / den,
        0.0,
        -p.a * p.a / (p.b2 * den),
        2.0 * p.c / den,
        0.0,
    )


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
    rmin: float, n: int = 48, *, beta: float | None = EQ6_BETA
) -> list[tuple[float, float]]:
    """Sample the upper-drop Eq. 6 arc from the axis point to ``(2 R_min, 0)``."""
    _reject_legacy_beta(beta)
    rmin = _validate_positive("R_min", rmin)
    if n < 4:
        raise ValueError("need at least four samples")
    p = eq6_ellipse_parameters(rmin)
    theta_axis = math.acos(-p.c / p.a)
    out: list[tuple[float, float]] = []
    for i in range(n):
        t = i / (n - 1)
        if i == 0:
            out.append(eq6_axis_point(rmin))
            continue
        if i == n - 1:
            out.append(eq6_plane_point(rmin))
            continue
        theta = theta_axis * (1.0 - t)
        out.append((p.c + p.a * math.cos(theta), p.b * math.sin(theta)))
    return out


def eq6_ellipse_point(rmin: float, t: float) -> tuple[float, float]:
    """Evaluate the Eq. 6 arc, with ``t=0`` at the axis and ``t=1`` at plane."""
    rmin = _validate_positive("R_min", rmin)
    t = float(t)
    if not math.isfinite(t):
        raise ValueError("ellipse parameter must be finite")
    if t <= 0.0:
        return eq6_axis_point(rmin)
    if t >= 1.0:
        return eq6_plane_point(rmin)
    p = eq6_ellipse_parameters(rmin)
    theta = math.acos(-p.c / p.a) * (1.0 - t)
    return (p.c + p.a * math.cos(theta), p.b * math.sin(theta))


def eq6_ellipse_slope(rmin: float, t: float) -> float:
    """Evaluate ``dr/dz`` on the translated ellipse at parameter ``t``."""
    r, z = eq6_ellipse_point(rmin, t)
    return eq6_conic_slope(r, z, eq6_conic_coeffs(rmin))


def eq6_interface_hit(
    rmin: float, r0: float, z0: float, *, scan: int = 256
) -> tuple[float, tuple[float, float]]:
    """Return the first Eq. 6 parameter and point on the initial interface.

    The signed function is ``ellipse_r - interface_r``.  A short scan brackets
    the first crossing and bisection then resolves it without relying on a
    polyline or on coordinate deduplication.  This is deliberately evaluated
    in the physical (r,z) plane so it remains well-conditioned at tiny R0.
    """
    rmin = _validate_positive("R_min", rmin)
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    if scan < 8:
        raise ValueError("scan must be at least eight intervals")

    def signed(t: float) -> float:
        r, z = eq6_ellipse_point(rmin, t)
        return r - r_free_surface(z, r0, z0)

    ta = 0.0
    fa = signed(ta)
    if abs(fa) == 0.0:
        return ta, eq6_ellipse_point(rmin, ta)
    for i in range(1, scan + 1):
        tb = i / scan
        fb = signed(tb)
        if fb == 0.0:
            return tb, eq6_ellipse_point(rmin, tb)
        # The divider starts inside the drop and ends outside it.  If a
        # future non-standard parameter set produces an earlier reverse
        # crossing, the first sign change is still the geometrically useful
        # one for clipping.
        if (fa < 0.0 <= fb) or (fa > 0.0 >= fb):
            lo, hi = ta, tb
            flo = fa
            for _ in range(96):
                mid = 0.5 * (lo + hi)
                fm = signed(mid)
                if fm == 0.0:
                    lo = hi = mid
                    break
                if (flo < 0.0 <= fm) or (flo > 0.0 >= fm):
                    hi = mid
                else:
                    lo, flo = mid, fm
            t_hit = 0.5 * (lo + hi)
            return t_hit, eq6_ellipse_point(rmin, t_hit)
        ta, fa = tb, fb
    raise RuntimeError("Eq. 6 ellipse does not meet the initial free surface")


def _bridge_surface_point(r0: float, z0: float, t: float) -> tuple[float, float]:
    """Point on Eq. 2 from the neck (t=0) towards its sphere junction."""
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    t = min(1.0, max(0.0, float(t)))
    rj, zj = sphere_bridge_junction(r0, z0)
    c = r0 + z0
    theta_j = math.atan2(zj, rj - c)
    theta = math.pi + (theta_j - math.pi) * t
    return (c + z0 * math.cos(theta), z0 * math.sin(theta))


def _sphere_lower_point(r: float, r0: float, z0: float) -> tuple[float, float]:
    r = min(1.0, max(0.0, float(r)))
    return (r, sphere_lower_surface_z(r, r0, z0))


def _interface_angle(r: float) -> float:
    return math.asin(min(1.0, max(0.0, float(r))))


def initial_interface_neck_to_hit(
    t: float, r0: float, z0: float, hit: tuple[float, float]
) -> tuple[float, float]:
    """Parameterize the interface from the neck to an Eq. 6 hit point.

    The point is assumed to be on the lower sphere branch after the tangent
    junction, as it is for the initial bridge and the Eq. 6 construction.  A
    bridge-only fallback is retained for unusual, deliberately oversized
    divider parameters.
    """
    t = min(1.0, max(0.0, float(t)))
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    rh, zh = float(hit[0]), float(hit[1])
    rj, zj = sphere_bridge_junction(r0, z0)
    if zh <= zj + 64.0 * math.ulp(max(zj, 1.0)):
        c = r0 + z0
        theta_j = math.atan2(zh, rh - c)
        theta = math.pi + (theta_j - math.pi) * t
        return (c + z0 * math.cos(theta), z0 * math.sin(theta))
    alpha_j = _interface_angle(rj)
    alpha_h = _interface_angle(rh)
    # Keep the C1 junction at the explicit quadratic midpoint.  Besides making
    # the two analytical pieces inspectable, this retains the O(R0**2) bridge
    # height (and, through the tangent point, the O(R0**3) centre offset) in a
    # Q2 graph instead of hiding it in a nearly endpoint parameter interval.
    split = 0.5
    if t <= split or split >= 1.0:
        return _bridge_surface_point(r0, z0, t / max(split, 1e-300))
    u = (t - split) / (1.0 - split)
    alpha = alpha_j + (alpha_h - alpha_j) * u
    return _sphere_lower_point(math.sin(alpha), r0, z0)


def initial_interface_hit_to_pole(
    t: float, r0: float, z0: float, hit: tuple[float, float]
) -> tuple[float, float]:
    """Parameterize the initial interface from an Eq. 6 hit to the pole."""
    t = min(1.0, max(0.0, float(t)))
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    rh, zh = float(hit[0]), float(hit[1])
    rj, zj = sphere_bridge_junction(r0, z0)
    if zh <= zj + 64.0 * math.ulp(max(zj, 1.0)):
        # Include the remaining bridge and lower-sphere part if a caller gives
        # a hit before the tangent point.
        alpha_h = _interface_angle(max(rh, rj))
    else:
        alpha_h = _interface_angle(rh)
    alpha_j = _interface_angle(rj)
    l_lower = max(0.0, math.pi / 2.0 - alpha_h)
    l_upper = math.pi / 2.0
    split = l_lower / (l_lower + l_upper) if l_lower > 0.0 else 0.0
    if t <= split or split >= 1.0:
        u = t / max(split, 1e-300) if split > 0.0 else 1.0
        alpha = alpha_h + (math.pi / 2.0 - alpha_h) * u
        return _sphere_lower_point(math.sin(alpha), r0, z0)
    u = (t - split) / (1.0 - split)
    alpha = (math.pi / 2.0) * (1.0 - u)
    return (math.sin(alpha), sphere_upper_surface_z(math.sin(alpha), r0, z0))


def r_free_surface(z: float, r0: float, z0: float) -> float:
    """Radial coordinate of the t = 0 interface at height ``z`` (upper drop)."""
    r0 = _validate_positive("R0", r0)
    z0 = _validate_positive("Z0", z0)
    z = float(z)
    if not math.isfinite(z) or z < 0.0:
        raise ValueError("upper drop has z >= 0")
    c = r0 + z0
    rj, zj = sphere_bridge_junction(r0, z0)
    offset = sphere_centre_offset(r0, z0)
    zc = 1.0 - offset
    if z <= zj:
        # Product form is more accurate than z0**2-z**2 close to the join.
        gap2 = max(0.0, (z0 - z) * (z0 + z))
        return c - math.sqrt(gap2)
    if z <= zc:
        # Lower sphere: q = 1-sqrt(1-r²) = z-(zc-1) = z+offset.
        q = z + offset
        r2 = q * (2.0 - q)
        return math.sqrt(max(0.0, r2))
    if z <= zc + 1.0:
        # Upper sphere, written in the analogous form with
        # h = 1-sqrt(1-r²) = (zc+1)-z.
        h = (zc + 1.0) - z
        r2 = h * (2.0 - h)
        return math.sqrt(max(0.0, r2))
    return 0.0


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
    if n < 3:
        raise ValueError("need at least three samples")
    t_hit, _ = eq6_interface_hit(rmin, r0, z0)
    # Resample the analytical ellipse itself.  The previous implementation
    # linearly interpolated the final crossing, which left the endpoint off
    # both Eq. 6 and the free surface by O(sample spacing), especially at
    # R0=1e-6.
    return [
        eq6_ellipse_point(rmin, t_hit * i / (n - 1))
        for i in range(n)
    ]


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
