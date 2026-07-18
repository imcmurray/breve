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
                    if pen_raw > 0.08:
                        frac = 0.85
                    elif pen_raw > 0.04:
                        frac = 0.55
                    elif pen_raw > 0.02:
                        frac = 0.35
                    else:
                        frac = 0.18 if position_only != 0 else 0.12
                    floor = 0.4 if pen_raw > 0.03 else mscale
                    if mscale > floor:
                        floor = mscale
                    frac *= floor
                    corr = frac * pen_raw / inv
                    if not is_static[ai]:
                        s = corr * inv_mass[ai]
                        pos[ai, 0] += nx * s
                        pos[ai, 1] += ny * s
                        pos[ai, 2] += nz * s
                    if not is_static[bi]:
                        s = corr * inv_mass[bi]
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
