"""
Numba-accelerated contact resolve kernels for breve physics.

Optional: if numba is not installed, HAS_NUMBA is False and PhysicsWorld
uses the pure-Python resolver. Install with: pip install 'breve[fast]'
or: pip install numba
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

_env_default_off = os.environ.get("BREVE_NUMBA", "1").strip().lower() in (
    "0",
    "false",
    "no",
    "off",
)

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def deco(fn):
            return fn

        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return deco


# Runtime toggle (web UI / API). Default on when installed unless BREVE_NUMBA=0.
_USE_NUMBA = bool(NUMBA_AVAILABLE) and not _env_default_off

# Back-compat: “can we use the JIT path?”
HAS_NUMBA = NUMBA_AVAILABLE


def numba_available() -> bool:
    return bool(NUMBA_AVAILABLE)


def numba_enabled() -> bool:
    return bool(NUMBA_AVAILABLE and _USE_NUMBA)


def set_numba_enabled(enabled: bool) -> dict:
    """Enable/disable JIT at runtime. Returns status dict for the API."""
    global _USE_NUMBA
    want = bool(enabled)
    if want and not NUMBA_AVAILABLE:
        _USE_NUMBA = False
        return {
            "ok": False,
            "available": False,
            "enabled": False,
            "error": "numba is not installed (pip install 'breve[fast]')",
        }
    _USE_NUMBA = want and NUMBA_AVAILABLE
    return {
        "ok": True,
        "available": bool(NUMBA_AVAILABLE),
        "enabled": bool(_USE_NUMBA),
        "error": None,
    }


@njit(cache=True)
def _cross(
    ax: float, ay: float, az: float, bx: float, by: float, bz: float
) -> Tuple[float, float, float]:
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@njit(cache=True)
def _ang_denom(
    inv_I: np.ndarray,
    bi: int,
    rx: float,
    ry: float,
    rz: float,
    ax: float,
    ay: float,
    az: float,
) -> float:
    cx, cy, cz = _cross(rx, ry, rz, ax, ay, az)
    ix = inv_I[bi, 0, 0] * cx + inv_I[bi, 0, 1] * cy + inv_I[bi, 0, 2] * cz
    iy = inv_I[bi, 1, 0] * cx + inv_I[bi, 1, 1] * cy + inv_I[bi, 1, 2] * cz
    iz = inv_I[bi, 2, 0] * cx + inv_I[bi, 2, 1] * cy + inv_I[bi, 2, 2] * cz
    return cx * ix + cy * iy + cz * iz


@njit(cache=True)
def _vel_at(
    pos: np.ndarray,
    vel: np.ndarray,
    omega: np.ndarray,
    bi: int,
    px: float,
    py: float,
    pz: float,
) -> Tuple[float, float, float]:
    rx = px - pos[bi, 0]
    ry = py - pos[bi, 1]
    rz = pz - pos[bi, 2]
    cx, cy, cz = _cross(omega[bi, 0], omega[bi, 1], omega[bi, 2], rx, ry, rz)
    return vel[bi, 0] + cx, vel[bi, 1] + cy, vel[bi, 2] + cz


@njit(cache=True)
def _apply_impulse(
    pos: np.ndarray,
    vel: np.ndarray,
    omega: np.ndarray,
    inv_mass: np.ndarray,
    inv_I: np.ndarray,
    is_static: np.ndarray,
    bi: int,
    ix: float,
    iy: float,
    iz: float,
    px: float,
    py: float,
    pz: float,
) -> None:
    if is_static[bi] or inv_mass[bi] <= 0.0:
        return
    im = inv_mass[bi]
    vel[bi, 0] += ix * im
    vel[bi, 1] += iy * im
    vel[bi, 2] += iz * im
    rx = px - pos[bi, 0]
    ry = py - pos[bi, 1]
    rz = pz - pos[bi, 2]
    tx, ty, tz = _cross(rx, ry, rz, ix, iy, iz)
    omega[bi, 0] += inv_I[bi, 0, 0] * tx + inv_I[bi, 0, 1] * ty + inv_I[bi, 0, 2] * tz
    omega[bi, 1] += inv_I[bi, 1, 0] * tx + inv_I[bi, 1, 1] * ty + inv_I[bi, 1, 2] * tz
    omega[bi, 2] += inv_I[bi, 2, 0] * tx + inv_I[bi, 2, 1] * ty + inv_I[bi, 2, 2] * tz


@njit(cache=True)
def resolve_contacts_batch(
    pos: np.ndarray,
    vel: np.ndarray,
    omega: np.ndarray,
    inv_mass: np.ndarray,
    inv_I: np.ndarray,
    restitution: np.ndarray,
    friction: np.ndarray,
    is_static: np.ndarray,
    ca: np.ndarray,
    cb: np.ndarray,
    cn: np.ndarray,
    cp: np.ndarray,
    cpen: np.ndarray,
    cmscale: np.ndarray,
    slop: float,
    baumgarte: float,
    bounce_threshold: float,
    dt: float,
    max_bias: float,
    deep_bias: float,
    position_only: int,
    iterations: int,
) -> None:
    """
    Sequential impulse solver over a fixed contact manifold.

    position_only=1: nonlinear position projection only.
    position_only=0: projection + velocity (bounce/friction).
    """
    m = ca.shape[0]
    if m == 0:
        return
    for _it in range(iterations):
        for ci in range(m):
            ai = ca[ci]
            bi = cb[ci]
            nx = cn[ci, 0]
            ny = cn[ci, 1]
            nz = cn[ci, 2]
            px = cp[ci, 0]
            py = cp[ci, 1]
            pz = cp[ci, 2]
            pen_raw = cpen[ci] - slop
            if pen_raw < 0.0:
                pen_raw = 0.0
            mscale = cmscale[ci]

            # --- position projection ---
            if pen_raw > 0.0:
                inv = inv_mass[ai] + inv_mass[bi]
                if inv > 1e-12:
                    vs_static = is_static[ai] or is_static[bi]
                    if pen_raw > 0.08:
                        frac = 0.95 if vs_static else 0.85
                    elif pen_raw > 0.04:
                        frac = 0.80 if vs_static else 0.55
                    elif pen_raw > 0.02:
                        frac = 0.60 if vs_static else 0.35
                    else:
                        # shallow: still push hard vs static floors (anti-tunnel)
                        if position_only != 0:
                            frac = 0.55 if vs_static else 0.18
                        else:
                            frac = 0.40 if vs_static else 0.12
                    # Always honour manifold_scale (1/n_points). Overriding it with
                    # a high floor made 4-foot floor manifolds apply ~3× correction
                    # and, with a wrong-way normal past the slab midline, fired
                    # bodies straight through the floor.
                    if mscale < 1.0:
                        frac *= mscale
                    elif not vs_static and pen_raw <= 0.03:
                        frac *= mscale
                    if frac > 1.0:
                        frac = 1.0
                    corr = frac * pen_raw / inv
                    # Cap per-contact displacement so stacked multi-body
                    # resolves cannot fire a body through a floor in one step.
                    max_shift = 0.12 if vs_static else 0.08
                    if not is_static[ai]:
                        s = corr * inv_mass[ai]
                        if s > max_shift:
                            s = max_shift
                        pos[ai, 0] += nx * s
                        pos[ai, 1] += ny * s
                        pos[ai, 2] += nz * s
                    if not is_static[bi]:
                        s = corr * inv_mass[bi]
                        if s > max_shift:
                            s = max_shift
                        pos[bi, 0] -= nx * s
                        pos[bi, 1] -= ny * s
                        pos[bi, 2] -= nz * s

            if position_only != 0:
                continue

            vax, vay, vaz = _vel_at(pos, vel, omega, ai, px, py, pz)
            vbx, vby, vbz = _vel_at(pos, vel, omega, bi, px, py, pz)
            rvx = vax - vbx
            rvy = vay - vby
            rvz = vaz - vbz
            vel_n = rvx * nx + rvy * ny + rvz * nz

            ra = restitution[ai]
            rb = restitution[bi]
            if ra < 0.0:
                ra = 0.0
            if rb < 0.0:
                rb = 0.0
            e = (ra * rb) ** 0.5

            rax = px - pos[ai, 0]
            ray = py - pos[ai, 1]
            raz = pz - pos[ai, 2]
            rbx = px - pos[bi, 0]
            rby = py - pos[bi, 1]
            rbz = pz - pos[bi, 2]

            denom = (
                inv_mass[ai]
                + inv_mass[bi]
                + _ang_denom(inv_I, ai, rax, ray, raz, nx, ny, nz)
                + _ang_denom(inv_I, bi, rbx, rby, rbz, nx, ny, nz)
            )
            if denom <= 1e-12:
                continue

            pen = pen_raw
            impact = -vel_n if vel_n < 0.0 else 0.0
            resting = impact < bounce_threshold

            if resting:
                j = (-vel_n / denom) if vel_n < 0.0 else 0.0
                if pen > 0.03:
                    bias = pen * 6.0 * mscale
                    if bias > deep_bias:
                        bias = deep_bias
                    j += bias / denom
                if j < 0.0:
                    j = 0.0
            else:
                if pen > 0.0 and dt > 1e-4:
                    raw_bias = baumgarte * mscale * pen / dt
                elif pen > 0.0:
                    raw_bias = baumgarte * mscale * pen / 1e-4
                else:
                    raw_bias = 0.0
                bias = raw_bias if raw_bias < max_bias else max_bias
                if pen > 0.05 and bias < deep_bias * 0.5:
                    bias = deep_bias * 0.5
                vn_neg = vel_n if vel_n < 0.0 else 0.0
                j = (-(1.0 + e) * vn_neg - bias) / denom
                if j < 0.0:
                    j = 0.0

            if j > 50.0:
                j = 50.0
            if j > 0.0:
                _apply_impulse(
                    pos, vel, omega, inv_mass, inv_I, is_static, ai, nx * j, ny * j, nz * j, px, py, pz
                )
                _apply_impulse(
                    pos,
                    vel,
                    omega,
                    inv_mass,
                    inv_I,
                    is_static,
                    bi,
                    -nx * j,
                    -ny * j,
                    -nz * j,
                    px,
                    py,
                    pz,
                )

            # friction
            vax, vay, vaz = _vel_at(pos, vel, omega, ai, px, py, pz)
            vbx, vby, vbz = _vel_at(pos, vel, omega, bi, px, py, pz)
            rvx = vax - vbx
            rvy = vay - vby
            rvz = vaz - vbz
            vn = rvx * nx + rvy * ny + rvz * nz
            tx = rvx - nx * vn
            ty = rvy - ny * vn
            tz = rvz - nz * vn
            tlen = (tx * tx + ty * ty + tz * tz) ** 0.5
            if tlen > 1e-5:
                inv_t = 1.0 / tlen
                thx = tx * inv_t
                thy = ty * inv_t
                thz = tz * inv_t
                denom_t = (
                    inv_mass[ai]
                    + inv_mass[bi]
                    + _ang_denom(inv_I, ai, rax, ray, raz, thx, thy, thz)
                    + _ang_denom(inv_I, bi, rbx, rby, rbz, thx, thy, thz)
                )
                if denom_t > 1e-12:
                    jt = -(rvx * thx + rvy * thy + rvz * thz) / denom_t
                    mu = (friction[ai] + friction[bi]) * 0.5
                    if resting:
                        mu2 = mu * 1.4
                        if mu2 < 0.7:
                            mu2 = 0.7
                        mu = mu2
                        j_cap = j * mu
                        if j < 1e-6:
                            floor_cap = 0.08 * mu
                            if floor_cap > j_cap:
                                j_cap = floor_cap
                    else:
                        j_cap = j * mu
                    if jt > j_cap:
                        jt = j_cap
                    elif jt < -j_cap:
                        jt = -j_cap
                    _apply_impulse(
                        pos,
                        vel,
                        omega,
                        inv_mass,
                        inv_I,
                        is_static,
                        ai,
                        thx * jt,
                        thy * jt,
                        thz * jt,
                        px,
                        py,
                        pz,
                    )
                    _apply_impulse(
                        pos,
                        vel,
                        omega,
                        inv_mass,
                        inv_I,
                        is_static,
                        bi,
                        -thx * jt,
                        -thy * jt,
                        -thz * jt,
                        px,
                        py,
                        pz,
                    )


@njit(cache=True)
def integrate_bodies(
    pos: np.ndarray,
    vel: np.ndarray,
    omega: np.ndarray,
    quat: np.ndarray,
    inv_mass: np.ndarray,
    inv_I: np.ndarray,
    mass: np.ndarray,
    force: np.ndarray,
    torque: np.ndarray,
    is_static: np.ndarray,
    awake: np.ndarray,
    gx: float,
    gy: float,
    gz: float,
    dt: float,
    linear_damping: float,
    angular_damping: float,
) -> None:
    """Semi-implicit Euler + quaternion integrate for dynamic awake bodies."""
    n = pos.shape[0]
    ld = 1.0 - linear_damping
    if ld < 0.0:
        ld = 0.0
    ad = 1.0 - angular_damping
    if ad < 0.0:
        ad = 0.0
    for i in range(n):
        if is_static[i] or not awake[i]:
            force[i, 0] = 0.0
            force[i, 1] = 0.0
            force[i, 2] = 0.0
            torque[i, 0] = 0.0
            torque[i, 1] = 0.0
            torque[i, 2] = 0.0
            continue
        im = inv_mass[i]
        vel[i, 0] += (force[i, 0] + gx * mass[i]) * im * dt
        vel[i, 1] += (force[i, 1] + gy * mass[i]) * im * dt
        vel[i, 2] += (force[i, 2] + gz * mass[i]) * im * dt
        tx, ty, tz = torque[i, 0], torque[i, 1], torque[i, 2]
        if tx != 0.0 or ty != 0.0 or tz != 0.0:
            omega[i, 0] += (inv_I[i, 0, 0] * tx + inv_I[i, 0, 1] * ty + inv_I[i, 0, 2] * tz) * dt
            omega[i, 1] += (inv_I[i, 1, 0] * tx + inv_I[i, 1, 1] * ty + inv_I[i, 1, 2] * tz) * dt
            omega[i, 2] += (inv_I[i, 2, 0] * tx + inv_I[i, 2, 1] * ty + inv_I[i, 2, 2] * tz) * dt
        vel[i, 0] *= ld
        vel[i, 1] *= ld
        vel[i, 2] *= ld
        omega[i, 0] *= ad
        omega[i, 1] *= ad
        omega[i, 2] *= ad
        speed2 = vel[i, 0] * vel[i, 0] + vel[i, 1] * vel[i, 1] + vel[i, 2] * vel[i, 2]
        if speed2 > 3600.0:
            s = 60.0 / (speed2 ** 0.5)
            vel[i, 0] *= s
            vel[i, 1] *= s
            vel[i, 2] *= s
        w2 = omega[i, 0] * omega[i, 0] + omega[i, 1] * omega[i, 1] + omega[i, 2] * omega[i, 2]
        if w2 > 625.0:
            s = 25.0 / (w2 ** 0.5)
            omega[i, 0] *= s
            omega[i, 1] *= s
            omega[i, 2] *= s
        force[i, 0] = 0.0
        force[i, 1] = 0.0
        force[i, 2] = 0.0
        torque[i, 0] = 0.0
        torque[i, 1] = 0.0
        torque[i, 2] = 0.0

        # integrate pose
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        pos[i, 2] += vel[i, 2] * dt
        # quaternion integrate (w,x,y,z)
        ox, oy, oz = omega[i, 0], omega[i, 1], omega[i, 2]
        w, x, y, z = quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3]
        dq0 = 0.5 * (-ox * x - oy * y - oz * z)
        dq1 = 0.5 * (ox * w + oy * z - oz * y)
        dq2 = 0.5 * (-ox * z + oy * w + oz * x)
        dq3 = 0.5 * (ox * y - oy * x + oz * w)
        w2 = w + dq0 * dt
        x2 = x + dq1 * dt
        y2 = y + dq2 * dt
        z2 = z + dq3 * dt
        nrm = (w2 * w2 + x2 * x2 + y2 * y2 + z2 * z2) ** 0.5
        if nrm < 1e-15:
            quat[i, 0] = 1.0
            quat[i, 1] = 0.0
            quat[i, 2] = 0.0
            quat[i, 3] = 0.0
        else:
            inv = 1.0 / nrm
            quat[i, 0] = w2 * inv
            quat[i, 1] = x2 * inv
            quat[i, 2] = y2 * inv
            quat[i, 3] = z2 * inv


@njit(cache=True)
def quat_to_R(quat: np.ndarray, out_R: np.ndarray) -> None:
    """Fill (N,3,3) rotation matrices from (N,4) quaternions wxyz."""
    n = quat.shape[0]
    for i in range(n):
        w, x, y, z = quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3]
        out_R[i, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        out_R[i, 0, 1] = 2.0 * (x * y - z * w)
        out_R[i, 0, 2] = 2.0 * (x * z + y * w)
        out_R[i, 1, 0] = 2.0 * (x * y + z * w)
        out_R[i, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        out_R[i, 1, 2] = 2.0 * (y * z - x * w)
        out_R[i, 2, 0] = 2.0 * (x * z - y * w)
        out_R[i, 2, 1] = 2.0 * (y * z + x * w)
        out_R[i, 2, 2] = 1.0 - 2.0 * (x * x + y * y)


@njit(cache=True)
def inv_inertia_world_batch(
    R: np.ndarray, inv_I_local: np.ndarray, is_static: np.ndarray, out: np.ndarray
) -> None:
    """out[i] = R * diag(inv_I_local) * R^T"""
    n = R.shape[0]
    for i in range(n):
        if is_static[i]:
            for r in range(3):
                for c in range(3):
                    out[i, r, c] = 0.0
            continue
        ix, iy, iz = inv_I_local[i, 0], inv_I_local[i, 1], inv_I_local[i, 2]
        # M = R * diag(i)  → columns of R scaled
        m00 = R[i, 0, 0] * ix
        m01 = R[i, 0, 1] * iy
        m02 = R[i, 0, 2] * iz
        m10 = R[i, 1, 0] * ix
        m11 = R[i, 1, 1] * iy
        m12 = R[i, 1, 2] * iz
        m20 = R[i, 2, 0] * ix
        m21 = R[i, 2, 1] * iy
        m22 = R[i, 2, 2] * iz
        # out = M * R^T
        out[i, 0, 0] = m00 * R[i, 0, 0] + m01 * R[i, 0, 1] + m02 * R[i, 0, 2]
        out[i, 0, 1] = m00 * R[i, 1, 0] + m01 * R[i, 1, 1] + m02 * R[i, 1, 2]
        out[i, 0, 2] = m00 * R[i, 2, 0] + m01 * R[i, 2, 1] + m02 * R[i, 2, 2]
        out[i, 1, 0] = m10 * R[i, 0, 0] + m11 * R[i, 0, 1] + m12 * R[i, 0, 2]
        out[i, 1, 1] = m10 * R[i, 1, 0] + m11 * R[i, 1, 1] + m12 * R[i, 1, 2]
        out[i, 1, 2] = m10 * R[i, 2, 0] + m11 * R[i, 2, 1] + m12 * R[i, 2, 2]
        out[i, 2, 0] = m20 * R[i, 0, 0] + m21 * R[i, 0, 1] + m22 * R[i, 0, 2]
        out[i, 2, 1] = m20 * R[i, 1, 0] + m21 * R[i, 1, 1] + m22 * R[i, 1, 2]
        out[i, 2, 2] = m20 * R[i, 2, 0] + m21 * R[i, 2, 1] + m22 * R[i, 2, 2]


KIND_SPHERE = 0
KIND_BOX = 1


@njit(cache=True)
def _point_in_obb_pen(
    px: float,
    py: float,
    pz: float,
    bx: float,
    by: float,
    bz: float,
    R: np.ndarray,
    bi: int,
    hx: float,
    hy: float,
    hz: float,
) -> float:
    """Penetration depth if point inside OBB bi, else -1."""
    dx = px - bx
    dy = py - by
    dz = pz - bz
    # local = R^T * d
    lx = R[bi, 0, 0] * dx + R[bi, 1, 0] * dy + R[bi, 2, 0] * dz
    ly = R[bi, 0, 1] * dx + R[bi, 1, 1] * dy + R[bi, 2, 1] * dz
    lz = R[bi, 0, 2] * dx + R[bi, 1, 2] * dy + R[bi, 2, 2] * dz
    if abs(lx) > hx + 1e-9 or abs(ly) > hy + 1e-9 or abs(lz) > hz + 1e-9:
        return -1.0
    dxp = hx - abs(lx)
    dyp = hy - abs(ly)
    dzp = hz - abs(lz)
    pen = dxp
    if dyp < pen:
        pen = dyp
    if dzp < pen:
        pen = dzp
    return pen


@njit(cache=True)
def _sat_box_box(
    pos: np.ndarray,
    R: np.ndarray,
    half: np.ndarray,
    ai: int,
    bi: int,
) -> Tuple[float, float, float, float]:
    """
    Returns (pen, nx, ny, nz) with normal from bi toward ai, or pen<=0 if separated.
    """
    ax, ay, az = pos[ai, 0], pos[ai, 1], pos[ai, 2]
    bx, by, bz = pos[bi, 0], pos[bi, 1], pos[bi, 2]
    tx, ty, tz = bx - ax, by - ay, bz - az
    ha0, ha1, ha2 = half[ai, 0], half[ai, 1], half[ai, 2]
    hb0, hb1, hb2 = half[bi, 0], half[bi, 1], half[bi, 2]

    min_pen = 1e30
    bnx, bny, bnz = 0.0, 1.0, 0.0

    # rotation of B in A frame AbsR
    # Rrel[i,j] = Ae_i · Be_j
    for i in range(3):
        # A face axis i
        ax_x, ax_y, ax_z = R[ai, 0, i], R[ai, 1, i], R[ai, 2, i]
        ra = half[ai, i]
        rb = (
            hb0 * abs(R[ai, 0, i] * R[bi, 0, 0] + R[ai, 1, i] * R[bi, 1, 0] + R[ai, 2, i] * R[bi, 2, 0])
            + hb1 * abs(R[ai, 0, i] * R[bi, 0, 1] + R[ai, 1, i] * R[bi, 1, 1] + R[ai, 2, i] * R[bi, 2, 1])
            + hb2 * abs(R[ai, 0, i] * R[bi, 0, 2] + R[ai, 1, i] * R[bi, 1, 2] + R[ai, 2, i] * R[bi, 2, 2])
        )
        # t along axis
        t_a = ax_x * tx + ax_y * ty + ax_z * tz
        pen = ra + rb - abs(t_a)
        if pen <= 0.0:
            return -1.0, 0.0, 1.0, 0.0
        if pen < min_pen:
            min_pen = pen
            # orient axis along t (from a to b) then flip to from b toward a later
            if t_a < 0.0:
                bnx, bny, bnz = -ax_x, -ax_y, -ax_z
            else:
                bnx, bny, bnz = ax_x, ax_y, ax_z

    for i in range(3):
        bx_x, bx_y, bx_z = R[bi, 0, i], R[bi, 1, i], R[bi, 2, i]
        rb = half[bi, i]
        ra = (
            ha0 * abs(R[bi, 0, i] * R[ai, 0, 0] + R[bi, 1, i] * R[ai, 1, 0] + R[bi, 2, i] * R[ai, 2, 0])
            + ha1 * abs(R[bi, 0, i] * R[ai, 0, 1] + R[bi, 1, i] * R[ai, 1, 1] + R[bi, 2, i] * R[ai, 2, 1])
            + ha2 * abs(R[bi, 0, i] * R[ai, 0, 2] + R[bi, 1, i] * R[ai, 1, 2] + R[bi, 2, i] * R[ai, 2, 2])
        )
        t_b = bx_x * tx + bx_y * ty + bx_z * tz
        pen = ra + rb - abs(t_b)
        if pen <= 0.0:
            return -1.0, 0.0, 1.0, 0.0
        if pen < min_pen:
            min_pen = pen
            if t_b < 0.0:
                bnx, bny, bnz = -bx_x, -bx_y, -bx_z
            else:
                bnx, bny, bnz = bx_x, bx_y, bx_z

    # edge-edge axes
    for i in range(3):
        aex, aey, aez = R[ai, 0, i], R[ai, 1, i], R[ai, 2, i]
        for j in range(3):
            bex, bey, bez = R[bi, 0, j], R[bi, 1, j], R[bi, 2, j]
            cx, cy, cz = _cross(aex, aey, aez, bex, bey, bez)
            n2 = cx * cx + cy * cy + cz * cz
            if n2 < 1e-16:
                continue
            inv = 1.0 / (n2 ** 0.5)
            cx, cy, cz = cx * inv, cy * inv, cz * inv
            ra = (
                ha0 * abs(R[ai, 0, 0] * cx + R[ai, 1, 0] * cy + R[ai, 2, 0] * cz)
                + ha1 * abs(R[ai, 0, 1] * cx + R[ai, 1, 1] * cy + R[ai, 2, 1] * cz)
                + ha2 * abs(R[ai, 0, 2] * cx + R[ai, 1, 2] * cy + R[ai, 2, 2] * cz)
            )
            rb = (
                hb0 * abs(R[bi, 0, 0] * cx + R[bi, 1, 0] * cy + R[bi, 2, 0] * cz)
                + hb1 * abs(R[bi, 0, 1] * cx + R[bi, 1, 1] * cy + R[bi, 2, 1] * cz)
                + hb2 * abs(R[bi, 0, 2] * cx + R[bi, 1, 2] * cy + R[bi, 2, 2] * cz)
            )
            td = cx * tx + cy * ty + cz * tz
            pen = ra + rb - abs(td)
            if pen <= 0.0:
                return -1.0, 0.0, 1.0, 0.0
            if pen < min_pen:
                min_pen = pen
                if td < 0.0:
                    bnx, bny, bnz = -cx, -cy, -cz
                else:
                    bnx, bny, bnz = cx, cy, cz

    # normal from b toward a: should point roughly a - b
    abx, aby, abz = ax - bx, ay - by, az - bz
    if bnx * abx + bny * aby + bnz * abz < 0.0:
        bnx, bny, bnz = -bnx, -bny, -bnz
    return min_pen, bnx, bny, bnz


@njit(cache=True)
def _append_box_box_manifold(
    pos: np.ndarray,
    R: np.ndarray,
    half: np.ndarray,
    ai: int,
    bi: int,
    pen: float,
    nx: float,
    ny: float,
    nz: float,
    max_pts: int,
    out_a: np.ndarray,
    out_b: np.ndarray,
    out_n: np.ndarray,
    out_p: np.ndarray,
    out_pen: np.ndarray,
    out_scale: np.ndarray,
    count: int,
    max_m: int,
) -> int:
    """Add up to max_pts corner contacts; returns new count."""
    # gather candidate corners
    pens = np.empty(16, dtype=np.float64)
    cxs = np.empty(16, dtype=np.float64)
    cys = np.empty(16, dtype=np.float64)
    czs = np.empty(16, dtype=np.float64)
    nc = 0
    signs = (-1.0, 1.0)
    # corners of A inside B
    for sx in signs:
        for sy in signs:
            for sz in signs:
                lx, ly, lz = sx * half[ai, 0], sy * half[ai, 1], sz * half[ai, 2]
                wx = pos[ai, 0] + R[ai, 0, 0] * lx + R[ai, 0, 1] * ly + R[ai, 0, 2] * lz
                wy = pos[ai, 1] + R[ai, 1, 0] * lx + R[ai, 1, 1] * ly + R[ai, 1, 2] * lz
                wz = pos[ai, 2] + R[ai, 2, 0] * lx + R[ai, 2, 1] * ly + R[ai, 2, 2] * lz
                p = _point_in_obb_pen(
                    wx, wy, wz, pos[bi, 0], pos[bi, 1], pos[bi, 2], R, bi,
                    half[bi, 0], half[bi, 1], half[bi, 2],
                )
                if p > 1e-6 and nc < 16:
                    pens[nc] = p
                    cxs[nc] = wx
                    cys[nc] = wy
                    czs[nc] = wz
                    nc += 1
    # corners of B inside A
    for sx in signs:
        for sy in signs:
            for sz in signs:
                lx, ly, lz = sx * half[bi, 0], sy * half[bi, 1], sz * half[bi, 2]
                wx = pos[bi, 0] + R[bi, 0, 0] * lx + R[bi, 0, 1] * ly + R[bi, 0, 2] * lz
                wy = pos[bi, 1] + R[bi, 1, 0] * lx + R[bi, 1, 1] * ly + R[bi, 1, 2] * lz
                wz = pos[bi, 2] + R[bi, 2, 0] * lx + R[bi, 2, 1] * ly + R[bi, 2, 2] * lz
                p = _point_in_obb_pen(
                    wx, wy, wz, pos[ai, 0], pos[ai, 1], pos[ai, 2], R, ai,
                    half[ai, 0], half[ai, 1], half[ai, 2],
                )
                if p > 1e-6 and nc < 16:
                    pens[nc] = p
                    cxs[nc] = wx
                    cys[nc] = wy
                    czs[nc] = wz
                    nc += 1

    if nc == 0:
        # fallback midpoint of centers projected
        if count >= max_m:
            return count
        out_a[count] = ai
        out_b[count] = bi
        out_n[count, 0] = nx
        out_n[count, 1] = ny
        out_n[count, 2] = nz
        out_p[count, 0] = 0.5 * (pos[ai, 0] + pos[bi, 0])
        out_p[count, 1] = 0.5 * (pos[ai, 1] + pos[bi, 1])
        out_p[count, 2] = 0.5 * (pos[ai, 2] + pos[bi, 2])
        out_pen[count] = pen
        out_scale[count] = 1.0
        return count + 1

    # pick deepest unique points
    picked = 0
    pxs = np.empty(4, dtype=np.float64)
    pys = np.empty(4, dtype=np.float64)
    pzs = np.empty(4, dtype=np.float64)
    used = np.zeros(nc, dtype=np.bool_)
    for _k in range(max_pts):
        best = -1
        best_p = -1.0
        for i in range(nc):
            if used[i]:
                continue
            if pens[i] > best_p:
                best_p = pens[i]
                best = i
        if best < 0:
            break
        # de-dupe
        dup = False
        for j in range(picked):
            dx = cxs[best] - pxs[j]
            dy = cys[best] - pys[j]
            dz = czs[best] - pzs[j]
            if dx * dx + dy * dy + dz * dz < 1e-6:
                dup = True
                break
        used[best] = True
        if dup:
            continue
        pxs[picked] = cxs[best]
        pys[picked] = cys[best]
        pzs[picked] = czs[best]
        picked += 1
        if picked >= max_pts:
            break

    if picked == 0:
        return count
    scale = 1.0 / float(picked)
    for j in range(picked):
        if count >= max_m:
            break
        out_a[count] = ai
        out_b[count] = bi
        out_n[count, 0] = nx
        out_n[count, 1] = ny
        out_n[count, 2] = nz
        out_p[count, 0] = pxs[j]
        out_p[count, 1] = pys[j]
        out_p[count, 2] = pzs[j]
        out_pen[count] = pen  # SAT depth
        out_scale[count] = scale
        count += 1
    return count


@njit(cache=True)
def find_contacts_packed(
    pos: np.ndarray,
    R: np.ndarray,
    half: np.ndarray,
    kind: np.ndarray,
    radius: np.ndarray,
    is_static: np.ndarray,
    broad_r: np.ndarray,
    out_a: np.ndarray,
    out_b: np.ndarray,
    out_n: np.ndarray,
    out_p: np.ndarray,
    out_pen: np.ndarray,
    out_scale: np.ndarray,
) -> int:
    """
    Broadphase + sphere/box contacts. Returns number of contacts written.
    out_* preallocated; max is out_a.shape[0].
    """
    n = pos.shape[0]
    max_m = out_a.shape[0]
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if is_static[i] and is_static[j]:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            rsum = broad_r[i] + broad_r[j]
            if dx * dx + dy * dy + dz * dz > rsum * rsum:
                continue
            ki, kj = kind[i], kind[j]
            if ki == KIND_SPHERE and kj == KIND_SPHERE:
                if count >= max_m:
                    return count
                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                ra, rb = radius[i], radius[j]
                if dist <= 1e-12:
                    nx, ny, nz = 0.0, 1.0, 0.0
                    dist = 0.0
                else:
                    inv = 1.0 / dist
                    nx, ny, nz = dx * inv, dy * inv, dz * inv
                pen = ra + rb - dist
                if pen <= 0.0:
                    continue
                out_a[count] = i
                out_b[count] = j
                out_n[count, 0] = nx
                out_n[count, 1] = ny
                out_n[count, 2] = nz
                out_p[count, 0] = pos[j, 0] + nx * rb
                out_p[count, 1] = pos[j, 1] + ny * rb
                out_p[count, 2] = pos[j, 2] + nz * rb
                out_pen[count] = pen
                out_scale[count] = 1.0
                count += 1
            elif ki == KIND_BOX and kj == KIND_BOX:
                pen, nx, ny, nz = _sat_box_box(pos, R, half, i, j)
                if pen <= 0.0:
                    continue
                # Floor-like statics: once a body crosses the slab midplane, SAT
                # orients the normal the wrong way and separation pushes *into*
                # the floor. Force mostly-vertical contacts so the dynamic body
                # is always pushed toward the top face (tip-off still works —
                # torque comes from edge contact points, not a side normal).
                nx, ny, nz = _orient_floor_normal(
                    pos, R, half, is_static, i, j, nx, ny, nz
                )
                max_pts = 4 if (is_static[i] or is_static[j]) else 3
                count = _append_box_box_manifold(
                    pos, R, half, i, j, pen, nx, ny, nz, max_pts,
                    out_a, out_b, out_n, out_p, out_pen, out_scale, count, max_m,
                )
            elif ki == KIND_SPHERE and kj == KIND_BOX:
                # sphere i vs box j
                count = _sphere_box_contact(
                    pos, R, half, radius, i, j, True,
                    out_a, out_b, out_n, out_p, out_pen, out_scale, count, max_m,
                )
            elif ki == KIND_BOX and kj == KIND_SPHERE:
                count = _sphere_box_contact(
                    pos, R, half, radius, j, i, False,
                    out_a, out_b, out_n, out_p, out_pen, out_scale, count, max_m,
                )
    return count


@njit(cache=True)
def _orient_floor_normal(
    pos: np.ndarray,
    R: np.ndarray,
    half: np.ndarray,
    is_static: np.ndarray,
    ai: int,
    bi: int,
    nx: float,
    ny: float,
    nz: float,
) -> Tuple[float, float, float]:
    """
    For floor-like static boxes (local +Y ≈ world up), ensure the contact
    normal separates the *dynamic* body toward free space above the top face.

    Convention: normal points from b toward a; a moves +n, b moves -n.
    """
    # identify static floor-like partner
    if is_static[ai] and not is_static[bi]:
        si, di = ai, bi
        # dyn is b → moves -n; want -n ≈ +up ⇒ n · up < 0
        want_n_dot_up_negative = True
    elif is_static[bi] and not is_static[ai]:
        si, di = bi, ai
        # dyn is a → moves +n; want +n ≈ +up ⇒ n · up > 0
        want_n_dot_up_negative = False
    else:
        return nx, ny, nz

    upx, upy, upz = R[si, 0, 1], R[si, 1, 1], R[si, 2, 1]
    if upy < 0.85:
        return nx, ny, nz  # wall / steep ramp — leave SAT alone

    # Only correct mostly-vertical contacts (side hits of thick pads stay SAT)
    ndot = nx * upx + ny * upy + nz * upz
    if ndot * ndot < 0.25:  # |cos| < 0.5
        return nx, ny, nz

    # Only when dyn COM is over the static footprint (edge hang keeps SAT tip)
    dx = pos[di, 0] - pos[si, 0]
    dy = pos[di, 1] - pos[si, 1]
    dz = pos[di, 2] - pos[si, 2]
    lx = R[si, 0, 0] * dx + R[si, 1, 0] * dy + R[si, 2, 0] * dz
    lz = R[si, 0, 2] * dx + R[si, 1, 2] * dy + R[si, 2, 2] * dz
    hx, hz = half[si, 0], half[si, 2]
    if abs(lx) > hx or abs(lz) > hz:
        return nx, ny, nz

    if want_n_dot_up_negative:
        if ndot > 0.0:
            return -nx, -ny, -nz
    else:
        if ndot < 0.0:
            return -nx, -ny, -nz
    return nx, ny, nz


@njit(cache=True)
def _sphere_box_contact(
    pos: np.ndarray,
    R: np.ndarray,
    half: np.ndarray,
    radius: np.ndarray,
    si: int,
    bi: int,
    sphere_is_a: bool,
    out_a: np.ndarray,
    out_b: np.ndarray,
    out_n: np.ndarray,
    out_p: np.ndarray,
    out_pen: np.ndarray,
    out_scale: np.ndarray,
    count: int,
    max_m: int,
) -> int:
    if count >= max_m:
        return count
    # sphere center in box local
    dx = pos[si, 0] - pos[bi, 0]
    dy = pos[si, 1] - pos[bi, 1]
    dz = pos[si, 2] - pos[bi, 2]
    lx = R[bi, 0, 0] * dx + R[bi, 1, 0] * dy + R[bi, 2, 0] * dz
    ly = R[bi, 0, 1] * dx + R[bi, 1, 1] * dy + R[bi, 2, 1] * dz
    lz = R[bi, 0, 2] * dx + R[bi, 1, 2] * dy + R[bi, 2, 2] * dz
    hx, hy, hz = half[bi, 0], half[bi, 1], half[bi, 2]
    cx = lx
    if cx > hx:
        cx = hx
    if cx < -hx:
        cx = -hx
    cy = ly
    if cy > hy:
        cy = hy
    if cy < -hy:
        cy = -hy
    cz = lz
    if cz > hz:
        cz = hz
    if cz < -hz:
        cz = -hz
    # closest world
    cwx = pos[bi, 0] + R[bi, 0, 0] * cx + R[bi, 0, 1] * cy + R[bi, 0, 2] * cz
    cwy = pos[bi, 1] + R[bi, 1, 0] * cx + R[bi, 1, 1] * cy + R[bi, 1, 2] * cz
    cwz = pos[bi, 2] + R[bi, 2, 0] * cx + R[bi, 2, 1] * cy + R[bi, 2, 2] * cz
    ddx = pos[si, 0] - cwx
    ddy = pos[si, 1] - cwy
    ddz = pos[si, 2] - cwz
    dist = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
    r = radius[si]
    if dist < 1e-12:
        # center inside: push out least pen axis
        faces_pen = np.empty(6, dtype=np.float64)
        faces_pen[0] = hx - lx
        faces_pen[1] = hx + lx
        faces_pen[2] = hy - ly
        faces_pen[3] = hy + ly
        faces_pen[4] = hz - lz
        faces_pen[5] = hz + lz
        best = 0
        bestp = faces_pen[0]
        for k in range(1, 6):
            if faces_pen[k] < bestp:
                bestp = faces_pen[k]
                best = k
        if best == 0:
            nx, ny, nz = R[bi, 0, 0], R[bi, 1, 0], R[bi, 2, 0]
        elif best == 1:
            nx, ny, nz = -R[bi, 0, 0], -R[bi, 1, 0], -R[bi, 2, 0]
        elif best == 2:
            nx, ny, nz = R[bi, 0, 1], R[bi, 1, 1], R[bi, 2, 1]
        elif best == 3:
            nx, ny, nz = -R[bi, 0, 1], -R[bi, 1, 1], -R[bi, 2, 1]
        elif best == 4:
            nx, ny, nz = R[bi, 0, 2], R[bi, 1, 2], R[bi, 2, 2]
        else:
            nx, ny, nz = -R[bi, 0, 2], -R[bi, 1, 2], -R[bi, 2, 2]
        pen = bestp + r
    else:
        inv = 1.0 / dist
        nx, ny, nz = ddx * inv, ddy * inv, ddz * inv
        pen = r - dist
        if pen <= 0.0:
            return count
    # Contact stores a,b with normal from b toward a in our solver convention.
    # nx is from box surface toward sphere center (push sphere along +n).
    if sphere_is_a:
        # a=sphere, b=box — sphere moves +n, so n should push sphere out
        out_a[count] = si
        out_b[count] = bi
        out_n[count, 0] = nx
        out_n[count, 1] = ny
        out_n[count, 2] = nz
    else:
        # a=box, b=sphere — sphere moves -n, so store -nx so -n = push out
        out_a[count] = bi
        out_b[count] = si
        out_n[count, 0] = -nx
        out_n[count, 1] = -ny
        out_n[count, 2] = -nz

    # Floor-like static box: if sphere center is over the footprint and the
    # contact is mostly vertical, always push the sphere toward +static up
    # (stops tunnel-through when center drops past the slab midplane).
    upx, upy, upz = R[bi, 0, 1], R[bi, 1, 1], R[bi, 2, 1]
    if upy >= 0.85 and abs(lx) <= hx and abs(lz) <= hz:
        # direction that should push sphere along +up when applied to sphere
        if sphere_is_a:
            # sphere moves +out_n
            nd = out_n[count, 0] * upx + out_n[count, 1] * upy + out_n[count, 2] * upz
            if nd * nd >= 0.25 and nd < 0.0:
                out_n[count, 0] = -out_n[count, 0]
                out_n[count, 1] = -out_n[count, 1]
                out_n[count, 2] = -out_n[count, 2]
        else:
            # sphere is b, moves -out_n; want -out_n · up > 0 ⇒ out_n · up < 0
            nd = out_n[count, 0] * upx + out_n[count, 1] * upy + out_n[count, 2] * upz
            if nd * nd >= 0.25 and nd > 0.0:
                out_n[count, 0] = -out_n[count, 0]
                out_n[count, 1] = -out_n[count, 1]
                out_n[count, 2] = -out_n[count, 2]

    out_p[count, 0] = cwx
    out_p[count, 1] = cwy
    out_p[count, 2] = cwz
    out_pen[count] = pen
    out_scale[count] = 1.0
    return count + 1


@njit(cache=True)
def hard_depenetrate_static(
    pos: np.ndarray,
    vel: np.ndarray,
    R: np.ndarray,
    half: np.ndarray,
    kind: np.ndarray,
    radius: np.ndarray,
    is_static: np.ndarray,
) -> None:
    """
    Keep dynamic bodies from sinking through static *top* surfaces.

    For each dynamic body over one or more floor-like footprints:
      - If already near-resting on any of them (need ≈ 0), only micro-correct
        shallow sinks on the nearest surface — never yank up to a higher step
        whose footprint happens to overlap (stairs).
      - If sunk through every nearby surface, lift by the *smallest* positive
        need (nearest top face). No depth cap — multi-body shoves can bury a
        box metres deep in one frame; COM over footprint still means "on pad".
    Edge tip-off is preserved: COM past the rim → no footprint match.
    """
    n = pos.shape[0]
    near_eps = 0.025
    for di in range(n):
        if is_static[di]:
            continue

        best_need = 1e30
        best_si = -1
        has_near_support = False
        for si in range(n):
            if not is_static[si] or kind[si] != KIND_BOX:
                continue
            if R[si, 1, 1] < 0.85:
                continue
            hx, hy, hz = half[si, 0], half[si, 1], half[si, 2]
            dx = pos[di, 0] - pos[si, 0]
            dy = pos[di, 1] - pos[si, 1]
            dz = pos[di, 2] - pos[si, 2]
            lx = R[si, 0, 0] * dx + R[si, 1, 0] * dy + R[si, 2, 0] * dz
            ly = R[si, 0, 1] * dx + R[si, 1, 1] * dy + R[si, 2, 1] * dz
            lz = R[si, 0, 2] * dx + R[si, 1, 2] * dy + R[si, 2, 2] * dz
            # Strict COM footprint — preserves edge tip-off when COM past rim.
            if abs(lx) > hx or abs(lz) > hz:
                continue
            # Accurate half-extent of dyn along static up (handles rotation)
            if kind[di] == KIND_SPHERE:
                e = radius[di]
            else:
                ux, uy, uz = R[si, 0, 1], R[si, 1, 1], R[si, 2, 1]
                e = (
                    half[di, 0]
                    * abs(R[di, 0, 0] * ux + R[di, 1, 0] * uy + R[di, 2, 0] * uz)
                    + half[di, 1]
                    * abs(R[di, 0, 1] * ux + R[di, 1, 1] * uy + R[di, 2, 1] * uz)
                    + half[di, 2]
                    * abs(R[di, 0, 2] * ux + R[di, 1, 2] * uy + R[di, 2, 2] * uz)
                )
            need = (hy + e) - ly
            # Resting on / slightly above this surface
            if need <= near_eps:
                if need > -0.05:
                    has_near_support = True
                continue
            if need < best_need:
                best_need = need
                best_si = si

        if best_si < 0:
            continue
        # With near support (e.g. on a lower stair), only fix shallow sinks on
        # the nearest other surface — never a 0.5 m yank onto an upper tread.
        if has_near_support and best_need > 0.15:
            continue
        si = best_si
        ux, uy, uz = R[si, 0, 1], R[si, 1, 1], R[si, 2, 1]
        pos[di, 0] += ux * best_need
        pos[di, 1] += uy * best_need
        pos[di, 2] += uz * best_need
        vn = vel[di, 0] * ux + vel[di, 1] * uy + vel[di, 2] * uz
        if vn < 0.0:
            vel[di, 0] -= ux * vn
            vel[di, 1] -= uy * vn
            vel[di, 2] -= uz * vn


def warmup() -> None:
    """Force-compile kernels once (first sim step still pays if skipped)."""
    if not NUMBA_AVAILABLE:
        return
    n, m = 2, 1
    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    omega = np.zeros((n, 3), dtype=np.float64)
    quat = np.zeros((n, 4), dtype=np.float64)
    quat[:, 0] = 1.0
    inv_mass = np.array([1.0, 0.0], dtype=np.float64)
    inv_I = np.zeros((n, 3, 3), dtype=np.float64)
    inv_I[0, 0, 0] = inv_I[0, 1, 1] = inv_I[0, 2, 2] = 1.0
    rest = np.array([0.3, 0.3], dtype=np.float64)
    fric = np.array([0.4, 0.4], dtype=np.float64)
    is_static = np.array([False, True])
    mass = np.array([1.0, 0.0], dtype=np.float64)
    force = np.zeros((n, 3), dtype=np.float64)
    torque = np.zeros((n, 3), dtype=np.float64)
    awake = np.array([True, True])
    ca = np.array([0], dtype=np.int64)
    cb = np.array([1], dtype=np.int64)
    cn = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    cp = np.zeros((m, 3), dtype=np.float64)
    cpen = np.array([0.01], dtype=np.float64)
    cmscale = np.array([1.0], dtype=np.float64)
    integrate_bodies(
        pos, vel, omega, quat, inv_mass, inv_I, mass, force, torque, is_static, awake,
        0.0, -9.8, 0.0, 0.016, 0.02, 0.06,
    )
    R = np.zeros((n, 3, 3), dtype=np.float64)
    inv_local = np.ones((n, 3), dtype=np.float64)
    quat_to_R(quat, R)
    inv_inertia_world_batch(R, inv_local, is_static, inv_I)
    resolve_contacts_batch(
        pos, vel, omega, inv_mass, inv_I, rest, fric, is_static,
        ca, cb, cn, cp, cpen, cmscale,
        0.006, 0.15, 0.8, 0.016, 0.25, 0.8, 0, 1,
    )
    # contact find path
    kind = np.array([KIND_SPHERE, KIND_BOX], dtype=np.int64)
    half = np.zeros((n, 3), dtype=np.float64)
    half[1, :] = 0.5
    radius = np.array([0.2, 0.0], dtype=np.float64)
    broad = np.array([0.2, 0.9], dtype=np.float64)
    max_m = 32
    oa = np.empty(max_m, dtype=np.int64)
    ob = np.empty(max_m, dtype=np.int64)
    on = np.empty((max_m, 3), dtype=np.float64)
    op = np.empty((max_m, 3), dtype=np.float64)
    open_ = np.empty(max_m, dtype=np.float64)
    osc = np.empty(max_m, dtype=np.float64)
    find_contacts_packed(pos, R, half, kind, radius, is_static, broad, oa, ob, on, op, open_, osc)
