"""
Rigid-body physics for breve (pure Python / NumPy).

Bodies have mass, linear + angular velocity, orientation (quaternion), and
local inertia. Contact impulses are applied at the contact *point*, so offset
forces produce torque — stacks tip and fall off edges naturally.

Not a full constraint/joint solver (Rapier later); enough for spheres, boxes,
gravity demos, and demolition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from breve.physics_kernels import numba_enabled, warmup as _numba_warmup

if TYPE_CHECKING:
    from breve.objects import Real
    from breve.shapes import Shape

_NUMBA_WARMED = False


class ShapeKind(Enum):
    SPHERE = auto()
    BOX = auto()


# ---------------------------------------------------------------------------
# Quaternions  (w, x, y, z)
# ---------------------------------------------------------------------------

def _q_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _q_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-15:
        return _q_identity()
    return q / n


def _q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _q_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _q_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate orientation with world-space angular velocity."""
    ox, oy, oz = omega
    w, x, y, z = q
    dq = 0.5 * np.array(
        [
            -ox * x - oy * y - oz * z,
            ox * w + oy * z - oz * y,
            -ox * z + oy * w + oz * x,
            ox * y - oy * x + oz * w,
        ],
        dtype=np.float64,
    )
    return _q_normalize(q + dq * dt)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hand-rolled cross product — numpy.cross is ~30× slower for 3-vectors."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


def _cross3(
    ax: float, ay: float, az: float, bx: float, by: float, bz: float
) -> Tuple[float, float, float]:
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def _dot3(ax: float, ay: float, az: float, bx: float, by: float, bz: float) -> float:
    return ax * bx + ay * by + az * bz


def _len3(x: float, y: float, z: float) -> float:
    return (x * x + y * y + z * z) ** 0.5


# ---------------------------------------------------------------------------
# Collider / body
# ---------------------------------------------------------------------------

@dataclass
class Collider:
    kind: ShapeKind
    # sphere: [radius]; box: half-extents [hx, hy, hz] in *local* space
    data: np.ndarray


@dataclass
class RigidBody:
    owner: object
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray = field(default_factory=_q_identity)
    angular_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    mass: float = 1.0
    inv_mass: float = 1.0
    # local inverse inertia (diagonal stored as 3-vector for spheres/boxes)
    inv_inertia_local: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    restitution: float = 0.45
    friction: float = 0.4
    static: bool = False
    collider: Optional[Collider] = None
    force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    awake: bool = True
    # cached per physics step
    _R: Optional[np.ndarray] = field(default=None, repr=False)
    _inv_I_w: Optional[np.ndarray] = field(default=None, repr=False)
    _corners: Optional[np.ndarray] = field(default=None, repr=False)
    _sleep_frames: int = 0

    def set_static(self, static: bool = True) -> None:
        self.static = static
        if static:
            self.inv_mass = 0.0
            self.inv_inertia_local[:] = 0.0
            self.velocity[:] = 0.0
            self.angular_velocity[:] = 0.0
        else:
            self.inv_mass = 0.0 if self.mass <= 0 else 1.0 / self.mass
            self._recompute_local_inertia()

    def _recompute_local_inertia(self) -> None:
        if self.static or self.collider is None or self.mass <= 0:
            self.inv_inertia_local[:] = 0.0
            return
        m = self.mass
        if self.collider.kind == ShapeKind.SPHERE:
            r = float(self.collider.data[0])
            i = 0.4 * m * r * r  # 2/5 m r^2
            if i < 1e-12:
                self.inv_inertia_local[:] = 0.0
            else:
                self.inv_inertia_local[:] = 1.0 / i
        elif self.collider.kind == ShapeKind.BOX:
            hx, hy, hz = (float(x) for x in self.collider.data)
            sx, sy, sz = 2 * hx, 2 * hy, 2 * hz
            # solid box about center
            ixx = m / 12.0 * (sy * sy + sz * sz)
            iyy = m / 12.0 * (sx * sx + sz * sz)
            izz = m / 12.0 * (sx * sx + sy * sy)
            self.inv_inertia_local[:] = [
                0.0 if ixx < 1e-12 else 1.0 / ixx,
                0.0 if iyy < 1e-12 else 1.0 / iyy,
                0.0 if izz < 1e-12 else 1.0 / izz,
            ]

    def rotation_matrix(self) -> np.ndarray:
        if self._R is None:
            self._R = _q_to_matrix(self.orientation)
        return self._R

    def inv_inertia_world(self) -> np.ndarray:
        """World-space inverse inertia tensor (3×3)."""
        if self.static:
            return np.zeros((3, 3), dtype=np.float64)
        if self._inv_I_w is None:
            R = self.rotation_matrix()
            inv_local = np.diag(self.inv_inertia_local)
            self._inv_I_w = R @ inv_local @ R.T
        return self._inv_I_w

    def invalidate_cache(self) -> None:
        self._R = None
        self._inv_I_w = None
        self._corners = None

    def apply_impulse_at(self, impulse: np.ndarray, point: np.ndarray) -> None:
        if self.static or self.inv_mass <= 0:
            return
        self.apply_impulse_xyz(
            float(impulse[0]),
            float(impulse[1]),
            float(impulse[2]),
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )

    def apply_impulse_xyz(
        self,
        ix: float,
        iy: float,
        iz: float,
        px: float,
        py: float,
        pz: float,
    ) -> None:
        """In-place impulse at world point — no temporary arrays."""
        if self.static or self.inv_mass <= 0 or not self.awake:
            return
        im = self.inv_mass
        self.velocity[0] += ix * im
        self.velocity[1] += iy * im
        self.velocity[2] += iz * im
        rx = px - float(self.position[0])
        ry = py - float(self.position[1])
        rz = pz - float(self.position[2])
        tx, ty, tz = _cross3(rx, ry, rz, ix, iy, iz)
        Iw = self.inv_inertia_world()
        self.angular_velocity[0] += Iw[0, 0] * tx + Iw[0, 1] * ty + Iw[0, 2] * tz
        self.angular_velocity[1] += Iw[1, 0] * tx + Iw[1, 1] * ty + Iw[1, 2] * tz
        self.angular_velocity[2] += Iw[2, 0] * tx + Iw[2, 1] * ty + Iw[2, 2] * tz
        # Do NOT reset _sleep_frames here — every resting contact applies tiny
        # impulses each step, which was keeping sleep_frames stuck at 1 and
        # boxes skating forever (~4s mark). Rest logic gates sleep on energy.

    def velocity_at_point(self, point: np.ndarray) -> np.ndarray:
        vx, vy, vz = self.velocity_at_xyz(
            float(point[0]), float(point[1]), float(point[2])
        )
        return np.array([vx, vy, vz], dtype=np.float64)

    def velocity_at_xyz(self, px: float, py: float, pz: float) -> Tuple[float, float, float]:
        rx = px - float(self.position[0])
        ry = py - float(self.position[1])
        rz = pz - float(self.position[2])
        wx = float(self.angular_velocity[0])
        wy = float(self.angular_velocity[1])
        wz = float(self.angular_velocity[2])
        cx, cy, cz = _cross3(wx, wy, wz, rx, ry, rz)
        return (
            float(self.velocity[0]) + cx,
            float(self.velocity[1]) + cy,
            float(self.velocity[2]) + cz,
        )

    def angular_denom_xyz(
        self, rx: float, ry: float, rz: float, ax: float, ay: float, az: float
    ) -> float:
        """(r × axis) · I^{-1} (r × axis)."""
        if self.static:
            return 0.0
        cx, cy, cz = _cross3(rx, ry, rz, ax, ay, az)
        Iw = self.inv_inertia_world()
        ix = Iw[0, 0] * cx + Iw[0, 1] * cy + Iw[0, 2] * cz
        iy = Iw[1, 0] * cx + Iw[1, 1] * cy + Iw[1, 2] * cz
        iz = Iw[2, 0] * cx + Iw[2, 1] * cy + Iw[2, 2] * cz
        return cx * ix + cy * iy + cz * iz

    def world_corners(self) -> np.ndarray:
        """8 corner positions of an oriented box (8×3). Cached per step."""
        if self._corners is not None:
            return self._corners
        assert self.collider is not None and self.collider.kind == ShapeKind.BOX
        hx, hy, hz = (float(x) for x in self.collider.data)
        R = self.rotation_matrix()
        local = np.array(
            [
                [-hx, -hy, -hz],
                [hx, -hy, -hz],
                [-hx, hy, -hz],
                [hx, hy, -hz],
                [-hx, -hy, hz],
                [hx, -hy, hz],
                [-hx, hy, hz],
                [hx, hy, hz],
            ],
            dtype=np.float64,
        )
        self._corners = (R @ local.T).T + self.position
        return self._corners

    def bounding_radius(self) -> float:
        """Conservative sphere bound for broadphase."""
        if self.collider is None:
            return 0.0
        if self.collider.kind == ShapeKind.SPHERE:
            return float(self.collider.data[0])
        hx, hy, hz = (float(x) for x in self.collider.data)
        return float(np.sqrt(hx * hx + hy * hy + hz * hz))


@dataclass
class Contact:
    a: RigidBody
    b: RigidBody
    normal: np.ndarray  # unit, points from b toward a (separating direction for a)
    penetration: float
    point: np.ndarray
    # scale positional correction when a manifold has several points (avoid 4× explode)
    manifold_scale: float = 1.0


class PhysicsWorld:
    def __init__(self) -> None:
        self.gravity = np.array([0.0, -9.8, 0.0], dtype=np.float64)
        self.bodies: List[RigidBody] = []
        self._by_owner: dict = {}
        self.iterations: int = 6
        # Position-only passes after velocity solve (push deep overlaps apart).
        # Numba contact find is cheap — 2 passes kills residual floor sinks.
        self.position_iterations: int = 2
        # Allow a little overlap without velocity bias (kills rest hop).
        self.slop: float = 0.006
        self.baumgarte: float = 0.15
        # Cap Baumgarte separation goal (m/s) — uncapped bias causes perpetual bounce.
        self.max_bias: float = 0.25
        # Extra separation velocity allowed only for deep sinks (m/s).
        self.deep_bias: float = 0.8
        self.linear_damping: float = 0.02
        self.angular_damping: float = 0.06
        # Below this approach speed, contacts are "resting" (no restitution).
        self.bounce_threshold: float = 0.8
        self.enabled: bool = True
        # frames of near-zero energy while in contact before full rest zero
        self.sleep_frames_needed: int = 4
        # multi-point caps (static pairs need more feet for rest; dynamic can be thinner)
        self.max_manifold_static: int = 4
        self.max_manifold_dynamic: int = 3

    def set_gravity(self, x: float, y: float, z: float) -> None:
        self.gravity = np.array([float(x), float(y), float(z)], dtype=np.float64)

    def clear(self) -> None:
        self.bodies.clear()
        self._by_owner.clear()

    def get_body(self, owner: object) -> Optional[RigidBody]:
        return self._by_owner.get(id(owner))

    def remove_body(self, owner: object) -> None:
        body = self._by_owner.pop(id(owner), None)
        if body is not None and body in self.bodies:
            self.bodies.remove(body)

    def add_or_update(
        self,
        owner: "Real",
        *,
        static: bool,
        mass: float = 1.0,
        restitution: float = 0.45,
        friction: float = 0.4,
    ) -> RigidBody:
        pos = np.array(
            [owner.location.x, owner.location.y, owner.location.z], dtype=np.float64
        )
        vel = np.zeros(3, dtype=np.float64)
        if hasattr(owner, "velocity"):
            vel = np.array(
                [owner.velocity.x, owner.velocity.y, owner.velocity.z],
                dtype=np.float64,
            )
        collider = _collider_from_shape(getattr(owner, "shape", None))
        existing = self._by_owner.get(id(owner))

        if existing is None:
            body = RigidBody(
                owner=owner,
                position=pos.copy(),
                velocity=vel.copy(),
                mass=max(mass, 1e-6),
                restitution=restitution,
                friction=friction,
                collider=collider,
            )
            body.set_static(static)
            self.bodies.append(body)
            self._by_owner[id(owner)] = body
            return body

        # Update mass/shape only — do NOT stomp velocity/orientation each frame
        # (that destroyed tipping and left bodies skating forever).
        existing.mass = max(mass, 1e-6)
        existing.collider = collider
        was_static = existing.static
        if static != was_static:
            existing.set_static(static)
        elif not static:
            existing.inv_mass = 1.0 / existing.mass
            existing._recompute_local_inertia()
        # Explicit resync (teleport / scene reset)
        if getattr(owner, "_physics_resync", False):
            existing.position[:] = pos
            existing.velocity[:] = vel
            existing.orientation = _q_identity()
            existing.angular_velocity[:] = 0.0
            existing.awake = True
            owner._physics_resync = False  # type: ignore[attr-defined]
        return existing

    def sync_from_owners(self) -> None:
        """Pull pose from owners only when flagged (teleport)."""
        for body in self.bodies:
            owner = body.owner
            if owner is None or not getattr(owner, "enabled", True):
                continue
            if not getattr(owner, "_physics_resync", False):
                continue
            body.position[:] = [
                owner.location.x,
                owner.location.y,
                owner.location.z,
            ]
            if hasattr(owner, "velocity") and not body.static:
                body.velocity[:] = [
                    owner.velocity.x,
                    owner.velocity.y,
                    owner.velocity.z,
                ]
            body.orientation = _q_identity()
            body.angular_velocity[:] = 0.0
            body.collider = _collider_from_shape(getattr(owner, "shape", None))
            if not body.static:
                body._recompute_local_inertia()
            body.awake = True
            owner._physics_resync = False  # type: ignore[attr-defined]

    def sync_to_owners(self) -> None:
        from breve.vector import vector

        for body in self.bodies:
            owner = body.owner
            if owner is None or not getattr(owner, "enabled", True):
                continue
            owner.location = vector(
                float(body.position[0]),
                float(body.position[1]),
                float(body.position[2]),
            )
            if hasattr(owner, "velocity") and not body.static:
                owner.velocity = vector(
                    float(body.velocity[0]),
                    float(body.velocity[1]),
                    float(body.velocity[2]),
                )
            if hasattr(owner, "acceleration") and not body.static:
                if getattr(owner, "physics_enabled", False):
                    owner.acceleration = vector(0, 0, 0)
            # stash orientation for viewers
            owner._physics_quat = body.orientation.copy()  # type: ignore[attr-defined]
            owner._physics_omega = body.angular_velocity.copy()  # type: ignore[attr-defined]

    def step(self, dt: float) -> None:
        if not self.enabled or dt <= 0:
            return
        if numba_enabled() and self.bodies:
            self._step_numba(dt)
        else:
            self._step_python(dt)

    def _step_numba(self, dt: float) -> None:
        """Pack bodies → Numba integrate/find/resolve → unpack."""
        global _NUMBA_WARMED
        if not _NUMBA_WARMED:
            _numba_warmup()
            _NUMBA_WARMED = True

        from breve.physics_kernels import (
            KIND_BOX,
            KIND_SPHERE,
            find_contacts_packed,
            hard_depenetrate_static,
            integrate_bodies,
            inv_inertia_world_batch,
            quat_to_R,
            resolve_contacts_batch,
        )
        from breve.shapes import Box, Sphere

        bodies = self.bodies
        n = len(bodies)
        pos = np.empty((n, 3), dtype=np.float64)
        vel = np.empty((n, 3), dtype=np.float64)
        omega = np.empty((n, 3), dtype=np.float64)
        quat = np.empty((n, 4), dtype=np.float64)
        inv_mass = np.empty(n, dtype=np.float64)
        inv_I_local = np.empty((n, 3), dtype=np.float64)
        mass = np.empty(n, dtype=np.float64)
        force = np.empty((n, 3), dtype=np.float64)
        torque = np.empty((n, 3), dtype=np.float64)
        is_static = np.empty(n, dtype=np.bool_)
        awake = np.empty(n, dtype=np.bool_)
        restitution = np.empty(n, dtype=np.float64)
        friction = np.empty(n, dtype=np.float64)
        kind = np.empty(n, dtype=np.int64)
        half = np.zeros((n, 3), dtype=np.float64)
        radius = np.zeros(n, dtype=np.float64)
        broad_r = np.zeros(n, dtype=np.float64)

        for i, b in enumerate(bodies):
            pos[i, 0] = b.position[0]
            pos[i, 1] = b.position[1]
            pos[i, 2] = b.position[2]
            vel[i, 0] = b.velocity[0]
            vel[i, 1] = b.velocity[1]
            vel[i, 2] = b.velocity[2]
            omega[i, 0] = b.angular_velocity[0]
            omega[i, 1] = b.angular_velocity[1]
            omega[i, 2] = b.angular_velocity[2]
            quat[i, 0] = b.orientation[0]
            quat[i, 1] = b.orientation[1]
            quat[i, 2] = b.orientation[2]
            quat[i, 3] = b.orientation[3]
            # Asleep bodies act as immovable (infinite mass) until woken —
            # stops gravity+contact bias from skating them across the floor.
            frozen = b.static or not b.awake
            inv_mass[i] = 0.0 if frozen else b.inv_mass
            inv_I_local[i, 0] = b.inv_inertia_local[0]
            inv_I_local[i, 1] = b.inv_inertia_local[1]
            inv_I_local[i, 2] = b.inv_inertia_local[2]
            mass[i] = b.mass
            force[i, 0] = b.force[0]
            force[i, 1] = b.force[1]
            force[i, 2] = b.force[2]
            torque[i, 0] = b.torque[0]
            torque[i, 1] = b.torque[1]
            torque[i, 2] = b.torque[2]
            is_static[i] = frozen  # solver: sleepers ≡ static until woken
            awake[i] = b.awake
            restitution[i] = b.restitution
            friction[i] = b.friction
            col = b.collider
            if col is not None and col.kind == ShapeKind.BOX:
                kind[i] = KIND_BOX
                half[i, 0] = col.data[0]
                half[i, 1] = col.data[1]
                half[i, 2] = col.data[2]
                broad_r[i] = float(
                    (half[i, 0] ** 2 + half[i, 1] ** 2 + half[i, 2] ** 2) ** 0.5
                )
            elif col is not None and col.kind == ShapeKind.SPHERE:
                kind[i] = KIND_SPHERE
                radius[i] = col.data[0]
                broad_r[i] = radius[i]
            else:
                # fall back from owner shape
                sh = getattr(b.owner, "shape", None)
                if isinstance(sh, Box):
                    kind[i] = KIND_BOX
                    half[i, 0] = float(sh.size.x) * 0.5
                    half[i, 1] = float(sh.size.y) * 0.5
                    half[i, 2] = float(sh.size.z) * 0.5
                    broad_r[i] = float(
                        (half[i, 0] ** 2 + half[i, 1] ** 2 + half[i, 2] ** 2) ** 0.5
                    )
                else:
                    kind[i] = KIND_SPHERE
                    radius[i] = float(sh.radius) if isinstance(sh, Sphere) else 0.25
                    broad_r[i] = radius[i]

        R = np.empty((n, 3, 3), dtype=np.float64)
        inv_I = np.empty((n, 3, 3), dtype=np.float64)
        quat_to_R(quat, R)
        inv_inertia_world_batch(R, inv_I_local, is_static, inv_I)

        gx, gy, gz = float(self.gravity[0]), float(self.gravity[1]), float(self.gravity[2])
        integrate_bodies(
            pos, vel, omega, quat, inv_mass, inv_I, mass, force, torque,
            is_static, awake, gx, gy, gz, float(dt),
            float(self.linear_damping), float(self.angular_damping),
        )
        quat_to_R(quat, R)
        inv_inertia_world_batch(R, inv_I_local, is_static, inv_I)

        # Contact buffers (n*(n-1)/2 * 4 max points is plenty for demos)
        max_m = max(64, n * n * 2)
        ca = np.empty(max_m, dtype=np.int64)
        cb = np.empty(max_m, dtype=np.int64)
        cn = np.empty((max_m, 3), dtype=np.float64)
        cp = np.empty((max_m, 3), dtype=np.float64)
        cpen = np.empty(max_m, dtype=np.float64)
        cmscale = np.empty(max_m, dtype=np.float64)

        # Broadphase/contacts must see true static vs dynamic (not sleepers),
        # so rebuild a real-static mask for finding; sleepers stay in pairs.
        real_static = np.array([b.static for b in bodies], dtype=np.bool_)
        m = find_contacts_packed(
            pos, R, half, kind, radius, real_static, broad_r,
            ca, cb, cn, cp, cpen, cmscale,
        )
        last_m = m

        # Wake sleepers hit by energetic awake bodies (or deep new overlaps).
        # Low threshold so a tipping base re-awakens a frozen stack above it.
        if m > 0:
            wake_speed2 = 0.04 * 0.04
            wake_spin2 = 0.15 * 0.15
            for k in range(m):
                i, j = int(ca[k]), int(cb[k])
                if real_static[i] or real_static[j]:
                    continue
                si = float(vel[i, 0] ** 2 + vel[i, 1] ** 2 + vel[i, 2] ** 2)
                sj = float(vel[j, 0] ** 2 + vel[j, 1] ** 2 + vel[j, 2] ** 2)
                wi = float(omega[i, 0] ** 2 + omega[i, 1] ** 2 + omega[i, 2] ** 2)
                wj = float(omega[j, 0] ** 2 + omega[j, 1] ** 2 + omega[j, 2] ** 2)
                pen = float(cpen[k])
                j_hot = awake[j] and (sj > wake_speed2 or wj > wake_spin2 or pen > 0.03)
                i_hot = awake[i] and (si > wake_speed2 or wi > wake_spin2 or pen > 0.03)
                if not awake[i] and j_hot:
                    awake[i] = True
                    inv_mass[i] = bodies[i].inv_mass
                    is_static[i] = False
                if not awake[j] and i_hot:
                    awake[j] = True
                    inv_mass[j] = bodies[j].inv_mass
                    is_static[j] = False
            inv_inertia_world_batch(R, inv_I_local, is_static, inv_I)

        if m > 0:
            resolve_contacts_batch(
                pos, vel, omega, inv_mass, inv_I, restitution, friction, is_static,
                ca[:m], cb[:m], cn[:m], cp[:m], cpen[:m], cmscale[:m],
                float(self.slop), float(self.baumgarte), float(self.bounce_threshold),
                float(dt), float(self.max_bias), float(self.deep_bias),
                0, int(self.iterations),
            )
            def _hard_depen_awake() -> None:
                # Don't move sleeping bodies (they are frozen until woken).
                saved = []
                for bi in range(n):
                    if not awake[bi] and not real_static[bi]:
                        saved.append(
                            (bi, pos[bi, 0], pos[bi, 1], pos[bi, 2],
                             vel[bi, 0], vel[bi, 1], vel[bi, 2])
                        )
                hard_depenetrate_static(
                    pos, vel, R, half, kind, radius, real_static
                )
                for bi, px, py, pz, vx, vy, vz in saved:
                    pos[bi, 0], pos[bi, 1], pos[bi, 2] = px, py, pz
                    vel[bi, 0], vel[bi, 1], vel[bi, 2] = vx, vy, vz

            # Lift bodies shoved into floors before position iterations.
            _hard_depen_awake()
            # Position-only pass(es) with refreshed Numba contacts
            for _ in range(max(1, self.position_iterations)):
                m2 = find_contacts_packed(
                    pos, R, half, kind, radius, real_static, broad_r,
                    ca, cb, cn, cp, cpen, cmscale,
                )
                if m2 <= 0:
                    break
                last_m = m2
                resolve_contacts_batch(
                    pos, vel, omega, inv_mass, inv_I, restitution, friction, is_static,
                    ca[:m2], cb[:m2], cn[:m2], cp[:m2], cpen[:m2], cmscale[:m2],
                    float(self.slop), float(self.baumgarte), float(self.bounce_threshold),
                    float(dt), float(self.max_bias), float(self.deep_bias),
                    1, 2,
                )
                _hard_depen_awake()
            # Final hard floor/step separation (anti-tunnel).
            _hard_depen_awake()
            m2 = find_contacts_packed(
                pos, R, half, kind, radius, real_static, broad_r,
                ca, cb, cn, cp, cpen, cmscale,
            )
            # Always refresh count — stale contacts kept sleepers "supported"
            # and floating mid-air after a stack base fell out from under them.
            last_m = m2
        else:
            # still run solid depenetration if no manifold contacts this frame
            hard_depenetrate_static(
                pos, vel, R, half, kind, radius, real_static
            )
        m = last_m

        # Unpack + rest. Dynamic *support* only when the partner can prop
        # (COM below or contact normal carries weight) — not mere side contact.
        dyn_support = np.zeros(n, dtype=np.bool_)
        static_support = np.zeros(n, dtype=np.bool_)
        in_contact = np.zeros(n, dtype=np.bool_)
        support_pts: List[List[Tuple[float, float]]] = [[] for _ in range(n)]
        if m > 0:
            for k in range(m):
                i, j = int(ca[k]), int(cb[k])
                in_contact[i] = True
                in_contact[j] = True
                if real_static[i] and not real_static[j]:
                    static_support[j] = True
                elif real_static[j] and not real_static[i]:
                    static_support[i] = True
                # Contact points below a body's COM can carry its weight —
                # they form the support hull for the stability/sleep gate.
                cpx, cpy, cpz = float(cp[k, 0]), float(cp[k, 1]), float(cp[k, 2])
                if not real_static[i] and cpy < float(pos[i, 1]):
                    support_pts[i].append((cpx, cpz))
                if not real_static[j] and cpy < float(pos[j, 1]):
                    support_pts[j].append((cpx, cpz))
                if real_static[i] or real_static[j]:
                    continue
                ny = float(cn[k, 1])  # normal from j toward i
                # Significant under-support only (not same-layer side neighbors)
                if pos[j, 1] < pos[i, 1] - 0.10 or ny > 0.55:
                    dyn_support[i] = True
                if pos[i, 1] < pos[j, 1] - 0.10 or (-ny) > 0.55:
                    dyn_support[j] = True

        for i, body in enumerate(bodies):
            body.position[0] = pos[i, 0]
            body.position[1] = pos[i, 1]
            body.position[2] = pos[i, 2]
            body.velocity[0] = vel[i, 0]
            body.velocity[1] = vel[i, 1]
            body.velocity[2] = vel[i, 2]
            body.angular_velocity[0] = omega[i, 0]
            body.angular_velocity[1] = omega[i, 1]
            body.angular_velocity[2] = omega[i, 2]
            body.orientation[0] = quat[i, 0]
            body.orientation[1] = quat[i, 1]
            body.orientation[2] = quat[i, 2]
            body.orientation[3] = quat[i, 3]
            body.awake = bool(awake[i])
            body.force[:] = 0.0
            body.torque[:] = 0.0
            body.invalidate_cache()
            if body.static:
                continue
            stable = _com_over_support_xz(
                float(pos[i, 0]),
                float(pos[i, 2]),
                support_pts[i],
                _support_margin(body),
            )
            if not body.awake:
                # Stay frozen only while supported *and* stably so — a sleeper
                # whose base slid away (corner-only contact) must wake and tip.
                if (not static_support[i] and not dyn_support[i]) or not stable:
                    body.awake = True
                    body._sleep_frames = 0
                else:
                    body.velocity[:] = 0.0
                    body.angular_velocity[:] = 0.0
                continue
            if not in_contact[i]:
                body._sleep_frames = 0
                continue
            _apply_contact_rest(
                body,
                self.sleep_frames_needed,
                dt,
                has_dynamic_support=bool(dyn_support[i]),
                has_static_support=bool(static_support[i]),
                support_stable=stable,
            )

    def _step_python(self, dt: float) -> None:
        gx, gy, gz = float(self.gravity[0]), float(self.gravity[1]), float(self.gravity[2])
        ld = max(0.0, 1.0 - self.linear_damping)
        ad = max(0.0, 1.0 - self.angular_damping)

        for body in self.bodies:
            body.invalidate_cache()
            if body.static or not body.awake:
                body.force[:] = 0.0
                body.torque[:] = 0.0
                continue
            # integrate forces in-place
            im = body.inv_mass
            body.velocity[0] += (body.force[0] + gx * body.mass) * im * dt
            body.velocity[1] += (body.force[1] + gy * body.mass) * im * dt
            body.velocity[2] += (body.force[2] + gz * body.mass) * im * dt
            if body.torque[0] or body.torque[1] or body.torque[2]:
                Iw = body.inv_inertia_world()
                tx, ty, tz = float(body.torque[0]), float(body.torque[1]), float(body.torque[2])
                body.angular_velocity[0] += (Iw[0, 0] * tx + Iw[0, 1] * ty + Iw[0, 2] * tz) * dt
                body.angular_velocity[1] += (Iw[1, 0] * tx + Iw[1, 1] * ty + Iw[1, 2] * tz) * dt
                body.angular_velocity[2] += (Iw[2, 0] * tx + Iw[2, 1] * ty + Iw[2, 2] * tz) * dt
            body.velocity[0] *= ld
            body.velocity[1] *= ld
            body.velocity[2] *= ld
            body.angular_velocity[0] *= ad
            body.angular_velocity[1] *= ad
            body.angular_velocity[2] *= ad
            speed2 = (
                float(body.velocity[0]) ** 2
                + float(body.velocity[1]) ** 2
                + float(body.velocity[2]) ** 2
            )
            if speed2 > 3600.0:
                s = 60.0 / (speed2 ** 0.5)
                body.velocity[0] *= s
                body.velocity[1] *= s
                body.velocity[2] *= s
            w2 = (
                float(body.angular_velocity[0]) ** 2
                + float(body.angular_velocity[1]) ** 2
                + float(body.angular_velocity[2]) ** 2
            )
            if w2 > 625.0:
                s = 25.0 / (w2 ** 0.5)
                body.angular_velocity[0] *= s
                body.angular_velocity[1] *= s
                body.angular_velocity[2] *= s
            body.force[:] = 0.0
            body.torque[:] = 0.0

        for body in self.bodies:
            if body.static or not body.awake:
                continue
            body.position[0] += body.velocity[0] * dt
            body.position[1] += body.velocity[1] * dt
            body.position[2] += body.velocity[2] * dt
            body.orientation = _q_integrate(
                body.orientation, body.angular_velocity, dt
            )
            body.invalidate_cache()

        contacts = self._find_contacts()
        # warm inv-inertia caches before the iteration loop
        for body in self.bodies:
            if not body.static and body.awake:
                body.inv_inertia_world()
        for _ in range(self.iterations):
            for c in contacts:
                _resolve_contact(
                    c,
                    self.slop,
                    self.baumgarte,
                    self.bounce_threshold,
                    dt,
                    self.max_bias,
                    self.deep_bias,
                    position_only=False,
                )
        # Separate position projection so deep multi-body piles de-penetrate
        # without needing huge velocity bias (which causes rest hop).
        for _ in range(self.position_iterations):
            # refresh contacts after previous projection moves bodies
            contacts = self._find_contacts()
            if not contacts:
                break
            for c in contacts:
                _resolve_contact(
                    c,
                    self.slop,
                    self.baumgarte,
                    self.bounce_threshold,
                    dt,
                    self.max_bias,
                    self.deep_bias,
                    position_only=True,
                )

        # Hard floor/step separation (anti-tunnel) for pure-Python path.
        _hard_depenetrate_static_py(self.bodies)

        # Contact damping + rest zero. Free-fall (not in contact) is never damped.
        support = _contact_support_map(contacts)
        support_pts = _support_points_map(contacts)
        for body in self.bodies:
            if body.static:
                continue
            info = support.get(id(body))
            if info is None:
                if not body.awake:
                    body.awake = True
                body._sleep_frames = 0
                continue
            has_dyn, has_static = info
            stable = _com_over_support_xz(
                float(body.position[0]),
                float(body.position[2]),
                support_pts.get(id(body), []),
                _support_margin(body),
            )
            if not body.awake:
                # Frozen only while stably supported (matches Numba path).
                if (not has_static and not has_dyn) or not stable:
                    body.awake = True
                    body._sleep_frames = 0
                else:
                    body.velocity[:] = 0.0
                    body.angular_velocity[:] = 0.0
                    continue
            _apply_contact_rest(
                body,
                self.sleep_frames_needed,
                dt,
                has_dynamic_support=has_dyn,
                has_static_support=has_static,
                support_stable=stable,
            )

    def _find_contacts(self) -> List[Contact]:
        contacts: List[Contact] = []
        active = [
            b
            for b in self.bodies
            if b.collider is not None and getattr(b.owner, "enabled", True)
        ]
        n = len(active)
        # broadphase radii (cheap sphere test before SAT)
        radii = [b.bounding_radius() for b in active]
        for i in range(n):
            a = active[i]
            ra = radii[i]
            pa = a.position
            for j in range(i + 1, n):
                b = active[j]
                if a.static and b.static:
                    continue
                # sphere broadphase
                dx = float(pa[0] - b.position[0])
                dy = float(pa[1] - b.position[1])
                dz = float(pa[2] - b.position[2])
                rsum = ra + radii[j]
                if dx * dx + dy * dy + dz * dz > rsum * rsum:
                    continue
                hits = _collide(a, b)
                if hits:
                    contacts.extend(hits)
        return contacts


# ---------------------------------------------------------------------------
# Shape helpers / collision
# ---------------------------------------------------------------------------

def _collider_from_shape(shape: Optional["Shape"]) -> Optional[Collider]:
    if shape is None:
        return None
    from breve.shapes import Box, Sphere

    if isinstance(shape, Sphere):
        return Collider(
            ShapeKind.SPHERE, np.array([float(shape.radius)], dtype=np.float64)
        )
    if isinstance(shape, Box):
        hx = float(shape.size.x) * 0.5
        hy = float(shape.size.y) * 0.5
        hz = float(shape.size.z) * 0.5
        return Collider(
            ShapeKind.BOX, np.array([hx, hy, hz], dtype=np.float64)
        )
    r = float(shape.bounding_radius())
    return Collider(ShapeKind.SPHERE, np.array([r], dtype=np.float64))


def _collide(a: RigidBody, b: RigidBody) -> List[Contact]:
    assert a.collider and b.collider
    ka, kb = a.collider.kind, b.collider.kind
    if ka == ShapeKind.SPHERE and kb == ShapeKind.SPHERE:
        c = _sphere_sphere(a, b)
        return [c] if c is not None else []
    if ka == ShapeKind.SPHERE and kb == ShapeKind.BOX:
        c = _sphere_obb(a, b)
        return [c] if c is not None else []
    if ka == ShapeKind.BOX and kb == ShapeKind.SPHERE:
        c = _sphere_obb(b, a)
        if c is None:
            return []
        return [Contact(a, b, -c.normal, c.penetration, c.point)]
    if ka == ShapeKind.BOX and kb == ShapeKind.BOX:
        return _obb_obb_manifold(a, b)
    return []


def _sphere_sphere(a: RigidBody, b: RigidBody) -> Optional[Contact]:
    ra = float(a.collider.data[0])  # type: ignore[index]
    rb = float(b.collider.data[0])  # type: ignore[index]
    delta = a.position - b.position
    dist = float(np.linalg.norm(delta))
    if dist <= 1e-12:
        normal = np.array([0.0, 1.0, 0.0])
        dist = 0.0
    else:
        normal = delta / dist
    pen = ra + rb - dist
    if pen <= 0:
        return None
    point = b.position + normal * rb
    return Contact(a, b, normal, pen, point)


def _sphere_obb(sphere: RigidBody, box: RigidBody) -> Optional[Contact]:
    """Sphere vs oriented box."""
    r = float(sphere.collider.data[0])  # type: ignore[index]
    half = box.collider.data  # type: ignore[union-attr]
    R = box.rotation_matrix()
    # sphere center in box local space
    local = R.T @ (sphere.position - box.position)
    clamped = np.minimum(half, np.maximum(-half, local))
    closest_local = clamped
    closest = box.position + R @ closest_local
    delta = sphere.position - closest
    dist = float(np.linalg.norm(delta))

    if dist < 1e-12:
        # center inside: push out along least-penetration local axis
        faces = [
            (half[0] - local[0], R[:, 0]),
            (half[0] + local[0], -R[:, 0]),
            (half[1] - local[1], R[:, 1]),
            (half[1] + local[1], -R[:, 1]),
            (half[2] - local[2], R[:, 2]),
            (half[2] + local[2], -R[:, 2]),
        ]
        pen, normal = min(faces, key=lambda t: t[0])
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        return Contact(sphere, box, normal, float(pen) + r, closest)

    normal = delta / dist
    pen = r - dist
    if pen <= 0:
        return None
    return Contact(sphere, box, normal, pen, closest)


def _closest_point_on_body(body: RigidBody, point: np.ndarray) -> np.ndarray:
    """Closest point on the body's surface/volume to a world-space point."""
    assert body.collider is not None
    if body.collider.kind == ShapeKind.SPHERE:
        r = float(body.collider.data[0])
        d = point - body.position
        dist = float(np.linalg.norm(d))
        if dist < 1e-12:
            return body.position + np.array([0.0, r, 0.0])
        return body.position + d * (r / dist)
    R = body.rotation_matrix()
    half = body.collider.data
    local = R.T @ (point - body.position)
    clamped = np.minimum(half, np.maximum(-half, local))
    return body.position + R @ clamped


def _contact_point_on_normal(a: RigidBody, b: RigidBody, n: np.ndarray) -> np.ndarray:
    """
    Contact location for torque computation.

    Must be the actual surface region between the two shapes — not the midpoint
    of centers (that puts the force deep under a platform) and not SAT support
    corners of a large floor (that puts the force on the far side).

    Closest-point pair keeps the force under the upper body, on the rim when
    the COM hangs past the edge, so gravity produces a tipping torque.
    """
    pa = _closest_point_on_body(a, b.position)
    pb = _closest_point_on_body(b, a.position)
    return 0.5 * (pa + pb)


def _sat_obb_overlap(
    a: RigidBody, b: RigidBody
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Oriented box vs oriented box (SAT).
    Returns (normal_from_b_toward_a, min_penetration) or None if separated.
    """
    ha = a.collider.data  # type: ignore[union-attr]
    hb = b.collider.data  # type: ignore[union-attr]
    Ra = a.rotation_matrix()
    Rb = b.rotation_matrix()
    Ae = [Ra[:, 0], Ra[:, 1], Ra[:, 2]]
    Be = [Rb[:, 0], Rb[:, 1], Rb[:, 2]]

    t = b.position - a.position

    R = Ra.T @ Rb
    AbsR = np.abs(R) + 1e-8
    t_a = Ra.T @ t

    min_pen = 1e30
    best_nx = best_ny = best_nz = 0.0
    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])

    def consider(pen: float, ax: float, ay: float, az: float) -> bool:
        """Return False if separated (pen<=0 already handled). Updates best axis."""
        nonlocal min_pen, best_nx, best_ny, best_nz
        nlen = (ax * ax + ay * ay + az * az) ** 0.5
        if nlen < 1e-10:
            return True
        inv = 1.0 / nlen
        ax, ay, az = ax * inv, ay * inv, az * inv
        if ax * tx + ay * ty + az * tz < 0.0:
            ax, ay, az = -ax, -ay, -az
        if pen < min_pen:
            min_pen = pen
            best_nx, best_ny, best_nz = ax, ay, az
        return True

    for i in range(3):
        ra = float(ha[i])
        rb = float(hb[0] * AbsR[i, 0] + hb[1] * AbsR[i, 1] + hb[2] * AbsR[i, 2])
        pen = ra + rb - abs(float(t_a[i]))
        if pen <= 0:
            return None
        col = Ra[:, i]
        consider(pen, float(col[0]), float(col[1]), float(col[2]))

    t_b = Rb.T @ t
    for i in range(3):
        ra = float(ha[0] * AbsR[0, i] + ha[1] * AbsR[1, i] + ha[2] * AbsR[2, i])
        rb = float(hb[i])
        pen = ra + rb - abs(float(t_b[i]))
        if pen <= 0:
            return None
        col = Rb[:, i]
        consider(pen, float(col[0]), float(col[1]), float(col[2]))

    # Edge-edge axes (skip near-parallel edges)
    for i in range(3):
        aex, aey, aez = float(Ae[i][0]), float(Ae[i][1]), float(Ae[i][2])
        for j in range(3):
            bex, bey, bez = float(Be[j][0]), float(Be[j][1]), float(Be[j][2])
            ax, ay, az = _cross3(aex, aey, aez, bex, bey, bez)
            nlen2 = ax * ax + ay * ay + az * az
            if nlen2 < 1e-16:
                continue
            inv = 1.0 / (nlen2 ** 0.5)
            ax, ay, az = ax * inv, ay * inv, az * inv
            ra = (
                float(ha[0]) * abs(_dot3(float(Ae[0][0]), float(Ae[0][1]), float(Ae[0][2]), ax, ay, az))
                + float(ha[1]) * abs(_dot3(float(Ae[1][0]), float(Ae[1][1]), float(Ae[1][2]), ax, ay, az))
                + float(ha[2]) * abs(_dot3(float(Ae[2][0]), float(Ae[2][1]), float(Ae[2][2]), ax, ay, az))
            )
            rb = (
                float(hb[0]) * abs(_dot3(float(Be[0][0]), float(Be[0][1]), float(Be[0][2]), ax, ay, az))
                + float(hb[1]) * abs(_dot3(float(Be[1][0]), float(Be[1][1]), float(Be[1][2]), ax, ay, az))
                + float(hb[2]) * abs(_dot3(float(Be[2][0]), float(Be[2][1]), float(Be[2][2]), ax, ay, az))
            )
            pen = ra + rb - abs(ax * tx + ay * ty + az * tz)
            if pen <= 0:
                return None
            consider(pen, ax, ay, az)

    if min_pen >= 1e29:
        return None

    # normal from b toward a
    abx = float(a.position[0] - b.position[0])
    aby = float(a.position[1] - b.position[1])
    abz = float(a.position[2] - b.position[2])
    if best_nx * abx + best_ny * aby + best_nz * abz < 0.0:
        best_nx, best_ny, best_nz = -best_nx, -best_ny, -best_nz
    return np.array([best_nx, best_ny, best_nz], dtype=np.float64), float(min_pen)


def _corner_penetration(box: RigidBody, point: np.ndarray) -> Optional[float]:
    """Positive penetration depth if point is inside the OBB, else None."""
    assert box.collider is not None and box.collider.kind == ShapeKind.BOX
    half = box.collider.data
    R = box.rotation_matrix()
    local = R.T @ (point - box.position)
    if (
        abs(float(local[0])) > float(half[0]) + 1e-9
        or abs(float(local[1])) > float(half[1]) + 1e-9
        or abs(float(local[2])) > float(half[2]) + 1e-9
    ):
        return None
    # distance to nearest face
    dx = float(half[0]) - abs(float(local[0]))
    dy = float(half[1]) - abs(float(local[1]))
    dz = float(half[2]) - abs(float(local[2]))
    return min(dx, dy, dz)


def _hard_depenetrate_static_py(bodies: List[RigidBody]) -> None:
    """Python mirror of hard_depenetrate_static (nearest floor top-face push)."""
    statics = [
        b
        for b in bodies
        if b.static
        and b.collider is not None
        and b.collider.kind == ShapeKind.BOX
    ]
    near_eps = 0.025
    for di in bodies:
        if di.static or di.collider is None:
            continue
        best_need = 1e30
        best_up = None
        has_near_support = False
        for si in statics:
            R = si.rotation_matrix()
            up = R[:, 1]
            if float(up[1]) < 0.85:
                continue
            hx, hy, hz = (
                float(si.collider.data[0]),  # type: ignore[index]
                float(si.collider.data[1]),  # type: ignore[index]
                float(si.collider.data[2]),  # type: ignore[index]
            )
            local = R.T @ (di.position - si.position)
            lx, ly, lz = float(local[0]), float(local[1]), float(local[2])
            if abs(lx) > hx or abs(lz) > hz:
                continue
            if di.collider.kind == ShapeKind.SPHERE:
                e = float(di.collider.data[0])  # type: ignore[index]
            else:
                Rd = di.rotation_matrix()
                hd = di.collider.data
                e = (
                    float(hd[0]) * abs(float(np.dot(Rd[:, 0], up)))  # type: ignore[index]
                    + float(hd[1]) * abs(float(np.dot(Rd[:, 1], up)))  # type: ignore[index]
                    + float(hd[2]) * abs(float(np.dot(Rd[:, 2], up)))  # type: ignore[index]
                )
            need = (hy + e) - ly
            if need <= near_eps:
                if need > -0.05:
                    has_near_support = True
                continue
            if need < best_need:
                best_need = need
                best_up = up
        if best_up is None:
            continue
        if has_near_support and best_need > 0.15:
            continue
        di.position[0] += float(best_up[0]) * best_need
        di.position[1] += float(best_up[1]) * best_need
        di.position[2] += float(best_up[2]) * best_need
        di.invalidate_cache()
        vn = float(
            di.velocity[0] * best_up[0]
            + di.velocity[1] * best_up[1]
            + di.velocity[2] * best_up[2]
        )
        if vn < 0.0:
            di.velocity[0] -= float(best_up[0]) * vn
            di.velocity[1] -= float(best_up[1]) * vn
            di.velocity[2] -= float(best_up[2]) * vn


def _orient_floor_normal_py(a: RigidBody, b: RigidBody, n: np.ndarray) -> np.ndarray:
    """
    Floor-like statics: past the slab midplane SAT can point the normal the
    wrong way. Force mostly-vertical contacts so the dynamic body is pushed
    toward the top face when its COM is over the footprint.
    """
    if a.static and not b.static:
        static, dyn = a, b
        want_n_dot_up_negative = True
    elif b.static and not a.static:
        static, dyn = b, a
        want_n_dot_up_negative = False
    else:
        return n
    if static.collider is None or static.collider.kind != ShapeKind.BOX:
        return n
    R = static.rotation_matrix()
    up = R[:, 1]
    if float(up[1]) < 0.85:
        return n
    ndot = float(n[0] * up[0] + n[1] * up[1] + n[2] * up[2])
    if ndot * ndot < 0.25:
        return n
    half = static.collider.data
    local = R.T @ (dyn.position - static.position)
    if abs(float(local[0])) > float(half[0]) or abs(float(local[2])) > float(half[2]):
        return n
    if want_n_dot_up_negative and ndot > 0.0:
        return -n
    if not want_n_dot_up_negative and ndot < 0.0:
        return -n
    return n


def _obb_obb_manifold(a: RigidBody, b: RigidBody) -> List[Contact]:
    """
    Multi-point box-box manifold.

    SAT finds the separating axis / normal. Penetrating *corners* of each box
    into the other become contact points. Against a static floor we keep up to
    4 feet (stable rest + tip torque). Dynamic–dynamic pairs keep at most 2
    (cheaper, still spins from offset hits).
    """
    sat = _sat_obb_overlap(a, b)
    if sat is None:
        return []
    n, min_pen = sat
    n = _orient_floor_normal_py(a, b, n)

    max_pts = 4 if (a.static or b.static) else 3

    candidates: List[Tuple[float, float, float, float]] = []  # score, x, y, z

    # Prefer corners that are deep *along the SAT normal* (not just any face),
    # so manifold penetration matches the true overlap axis.
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])

    def score_corner(point: np.ndarray, from_a: bool) -> Optional[float]:
        """How deep this corner sits along the contact normal."""
        pen = _corner_penetration(b if from_a else a, point)
        if pen is None or pen <= 1e-6:
            return None
        # Project: corners on the contacting side score higher
        return float(pen)

    # Corners of A inside B
    for corner in a.world_corners():
        sc = score_corner(corner, True)
        if sc is not None:
            candidates.append(
                (sc, float(corner[0]), float(corner[1]), float(corner[2]))
            )

    # Corners of B inside A
    for corner in b.world_corners():
        sc = score_corner(corner, False)
        if sc is not None:
            candidates.append(
                (sc, float(corner[0]), float(corner[1]), float(corner[2]))
            )

    if not candidates:
        point = _contact_point_on_normal(a, b, n)
        return [Contact(a, b, n, min_pen, point)]

    candidates.sort(key=lambda t: t[0], reverse=True)
    picked: List[Tuple[float, float, float]] = []
    for _sc, x, y, z in candidates:
        if any((x - qx) ** 2 + (y - qy) ** 2 + (z - qz) ** 2 < 1e-6 for qx, qy, qz in picked):
            continue
        picked.append((x, y, z))
        if len(picked) >= max_pts:
            break

    # All points share the SAT overlap depth so projection is consistent.
    # Manifold scale keeps multi-point position correction from exploding.
    scale = 1.0 / float(len(picked))
    return [
        Contact(
            a,
            b,
            n,
            min_pen,
            np.array([x, y, z], dtype=np.float64),
            manifold_scale=scale,
        )
        for x, y, z in picked
    ]


def _orientation_face_stable(body: RigidBody, threshold: float = 0.92) -> bool:
    """True if a local face axis is nearly world-up (for diagnostics / tests)."""
    R = body.rotation_matrix()
    for i in range(3):
        if abs(float(R[1, i])) >= threshold:
            return True
    return False


def _nearest_up_axis(body: RigidBody) -> Tuple[int, float, float, float, float]:
    """Local axis closest to world-up: (index, |dot|, ax, ay, az) with ay>=0."""
    R = body.rotation_matrix()
    best_i, best_abs = 0, -1.0
    for i in range(3):
        d = abs(float(R[1, i]))
        if d > best_abs:
            best_abs = d
            best_i = i
    ax, ay, az = float(R[0, best_i]), float(R[1, best_i]), float(R[2, best_i])
    if ay < 0.0:
        ax, ay, az = -ax, -ay, -az
    return best_i, best_abs, ax, ay, az


def _contact_support_map(contacts: List[Contact]) -> dict:
    """body_id -> (has_dynamic_partner, has_static_partner)."""
    out: dict = {}
    for c in contacts:
        for body, other in ((c.a, c.b), (c.b, c.a)):
            if body.static:
                continue
            bid = id(body)
            dyn, st = out.get(bid, (False, False))
            if other.static:
                st = True
            else:
                dyn = True
            out[bid] = (dyn, st)
    return out


def _support_points_map(contacts: List[Contact]) -> dict:
    """body_id -> [(x, z), ...] of contact points below that body's COM."""
    out: dict = {}
    for c in contacts:
        px, py, pz = float(c.point[0]), float(c.point[1]), float(c.point[2])
        for body in (c.a, c.b):
            if body.static:
                continue
            if py < float(body.position[1]):
                out.setdefault(id(body), []).append((px, pz))
    return out


# COM must project this close to the support hull to count as balanced.
# Kept tight: at 0.015 a half-metre cube slept ~2° off edge-balance, which
# still reads as "frozen mid-tip". Scaled down further for small bodies.
SUPPORT_MARGIN = 0.006


def _support_margin(body: RigidBody) -> float:
    return min(SUPPORT_MARGIN, 0.05 * max(body.bounding_radius(), 1e-3))


def _dist_point_segment_xz(
    px: float, pz: float, a: Tuple[float, float], b: Tuple[float, float]
) -> float:
    ax, az = a
    bx, bz = b
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    if L2 < 1e-18:
        ex, ez = px - ax, pz - az
        return (ex * ex + ez * ez) ** 0.5
    t = ((px - ax) * dx + (pz - az) * dz) / L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex, ez = px - (ax + t * dx), pz - (az + t * dz)
    return (ex * ex + ez * ez) ** 0.5


def _convex_hull_xz(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Monotone-chain hull, CCW. Input is tiny (≤ ~8 points)."""
    uniq = sorted(set(pts))
    if len(uniq) <= 2:
        return uniq

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in uniq:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(uniq):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _com_over_support_xz(
    px: float,
    pz: float,
    pts: List[Tuple[float, float]],
    margin: float = SUPPORT_MARGIN,
) -> bool:
    """
    True if gravity through the COM is carried by the contact set: the COM's
    horizontal projection lies inside (or within `margin` of) the convex hull
    of the support points. Corner contact → point; edge → segment; face → poly.
    """
    n = len(pts)
    if n == 0:
        return False
    if n == 1:
        dx, dz = px - pts[0][0], pz - pts[0][1]
        return dx * dx + dz * dz <= margin * margin
    hull = _convex_hull_xz(pts)
    if len(hull) == 1:
        dx, dz = px - hull[0][0], pz - hull[0][1]
        return dx * dx + dz * dz <= margin * margin
    if len(hull) == 2:
        return _dist_point_segment_xz(px, pz, hull[0], hull[1]) <= margin
    inside = True
    hn = len(hull)
    for i in range(hn):
        x1, z1 = hull[i]
        x2, z2 = hull[(i + 1) % hn]
        if (x2 - x1) * (pz - z1) - (z2 - z1) * (px - x1) < 0.0:
            inside = False
            break
    if inside:
        return True
    d = min(
        _dist_point_segment_xz(px, pz, hull[i], hull[(i + 1) % hn])
        for i in range(hn)
    )
    return d <= margin


def _apply_contact_rest(
    body: RigidBody,
    sleep_frames_needed: int,
    dt: float = 1.0 / 60.0,
    *,
    has_dynamic_support: bool = False,
    has_static_support: bool = False,
    support_stable: bool = True,
) -> None:
    """
    Natural rest only — no tip nudges, no face snaps, no forced poses.

    Gravity + contacts do the tipping. A body may only rest-damp and sleep
    when its support configuration is *stable* (COM over the contact hull —
    see _com_over_support_xz). A cube balanced on an unsupported corner or
    edge stays awake and undamped so gravity torque finishes the tip.
    Truly floating bodies never sleep.
    """
    del dt
    if not has_static_support and not has_dynamic_support:
        body._sleep_frames = 0
        return

    vx, vy, vz = (
        float(body.velocity[0]),
        float(body.velocity[1]),
        float(body.velocity[2]),
    )
    wx, wy, wz = (
        float(body.angular_velocity[0]),
        float(body.angular_velocity[1]),
        float(body.angular_velocity[2]),
    )
    speed = (vx * vx + vy * vy + vz * vz) ** 0.5
    spin = (wx * wx + wy * wy + wz * wz) ** 0.5

    if not support_stable:
        # Gravity torque is unresolved: no artificial damping (it stalls the
        # tip below the sleep threshold — the frozen-on-a-corner bug) and no
        # normal sleep. Safety valve for poses numerics call unstable but
        # with negligible residual torque: a long truly-quiescent window.
        if speed < 0.05 and spin < 0.10:
            body._sleep_frames += 1
            if body._sleep_frames >= 40:
                body.velocity[:] = 0.0
                body.angular_velocity[:] = 0.0
                body.awake = False
        else:
            body._sleep_frames = 0
        return

    # Mild damping when slow (solver chatter + residual roll) — not a tip force
    if speed < 0.50 and spin < 1.5:
        body.velocity[0] *= 0.85
        body.velocity[1] *= 0.90
        body.velocity[2] *= 0.85
        body.angular_velocity[0] *= 0.82
        body.angular_velocity[1] *= 0.82
        body.angular_velocity[2] *= 0.82
        speed = (
            float(body.velocity[0]) ** 2
            + float(body.velocity[1]) ** 2
            + float(body.velocity[2]) ** 2
        ) ** 0.5
        spin = (
            float(body.angular_velocity[0]) ** 2
            + float(body.angular_velocity[1]) ** 2
            + float(body.angular_velocity[2]) ** 2
        ) ** 0.5

    # Stack chatter often leaves spin ~0.05–0.20 forever — allow sleep there
    if speed < 0.12 and spin < 0.35:
        body._sleep_frames += 1
        if body._sleep_frames >= max(3, sleep_frames_needed):
            body.velocity[:] = 0.0
            body.angular_velocity[:] = 0.0
            body.awake = False
    else:
        body._sleep_frames = 0


def _resolve_contact(
    c: Contact,
    slop: float,
    baumgarte: float,
    bounce_threshold: float,
    dt: float = 1.0 / 60.0,
    max_bias: float = 0.25,
    deep_bias: float = 0.8,
    position_only: bool = False,
) -> None:
    a, b = c.a, c.b
    nx, ny, nz = float(c.normal[0]), float(c.normal[1]), float(c.normal[2])
    px, py, pz = float(c.point[0]), float(c.point[1]), float(c.point[2])

    # Asleep bodies are immovable until woken (matches Numba frozen mass).
    im_a = 0.0 if (a.static or not a.awake) else a.inv_mass
    im_b = 0.0 if (b.static or not b.awake) else b.inv_mass

    pen_raw = c.penetration - slop
    if pen_raw < 0.0:
        pen_raw = 0.0
    mscale = c.manifold_scale

    # --- position projection (also used alone in position_iterations) ---
    if pen_raw > 0.0:
        inv = im_a + im_b
        if inv > 1e-12:
            vs_static = a.static or b.static or (not a.awake) or (not b.awake)
            if pen_raw > 0.08:
                frac = 0.95 if vs_static else 0.85
            elif pen_raw > 0.04:
                frac = 0.80 if vs_static else 0.55
            elif pen_raw > 0.02:
                frac = 0.60 if vs_static else 0.35
            else:
                frac = (0.55 if vs_static else 0.18) if position_only else (
                    0.40 if vs_static else 0.12
                )
            # Honour manifold_scale (1/n_points). A high static floor override
            # made multi-foot floor contacts over-correct and tunnel bodies.
            if mscale < 1.0:
                frac *= mscale
            elif not vs_static and pen_raw <= 0.03:
                frac *= mscale
            frac = min(1.0, frac)
            corr = frac * pen_raw / inv
            max_shift = 0.12 if vs_static else 0.08
            if im_a > 0.0:
                s = min(corr * im_a, max_shift)
                a.position[0] += nx * s
                a.position[1] += ny * s
                a.position[2] += nz * s
            if im_b > 0.0:
                s = min(corr * im_b, max_shift)
                b.position[0] -= nx * s
                b.position[1] -= ny * s
                b.position[2] -= nz * s
            if vs_static and position_only and pen_raw > slop * 2:
                pen2 = min(pen_raw * (1.0 - frac), max_shift)
                if pen2 > 0.0:
                    if im_a <= 0.0 and im_b > 0.0:
                        b.position[0] -= nx * pen2
                        b.position[1] -= ny * pen2
                        b.position[2] -= nz * pen2
                    elif im_b <= 0.0 and im_a > 0.0:
                        a.position[0] += nx * pen2
                        a.position[1] += ny * pen2
                        a.position[2] += nz * pen2

    if position_only:
        return

    vax, vay, vaz = a.velocity_at_xyz(px, py, pz)
    vbx, vby, vbz = b.velocity_at_xyz(px, py, pz)
    rvx, rvy, rvz = vax - vbx, vay - vby, vaz - vbz
    vel_n = rvx * nx + rvy * ny + rvz * nz

    e = (max(a.restitution, 0.0) * max(b.restitution, 0.0)) ** 0.5
    rax = px - float(a.position[0])
    ray = py - float(a.position[1])
    raz = pz - float(a.position[2])
    rbx = px - float(b.position[0])
    rby = py - float(b.position[1])
    rbz = pz - float(b.position[2])

    ang_a = 0.0 if im_a <= 0.0 else a.angular_denom_xyz(rax, ray, raz, nx, ny, nz)
    ang_b = 0.0 if im_b <= 0.0 else b.angular_denom_xyz(rbx, rby, rbz, nx, ny, nz)
    denom = im_a + im_b + ang_a + ang_b
    if denom <= 1e-12:
        return

    pen = pen_raw
    impact = -vel_n if vel_n < 0.0 else 0.0
    resting = impact < bounce_threshold

    if resting:
        # Kill approach; if sunk deep, add modest separation goal (not rest hop).
        j = (-vel_n / denom) if vel_n < 0.0 else 0.0
        if pen > 0.03:
            bias = pen * 6.0 * mscale
            if bias > deep_bias:
                bias = deep_bias
            j += bias / denom
        if j < 0.0:
            j = 0.0
    else:
        raw_bias = (baumgarte * mscale * pen / dt) if (pen > 0.0 and dt > 1e-4) else 0.0
        if dt <= 1e-4 and pen > 0.0:
            raw_bias = baumgarte * mscale * pen / 1e-4
        bias = raw_bias if raw_bias < max_bias else max_bias
        if pen > 0.05:
            # deep impact: allow higher separation goal
            if bias < deep_bias * 0.5:
                bias = deep_bias * 0.5
        vn_neg = vel_n if vel_n < 0.0 else 0.0
        j = (-(1.0 + e) * vn_neg - bias) / denom
        if j < 0.0:
            j = 0.0

    if j > 50.0:
        j = 50.0
    if j > 0.0:
        if im_a > 0.0:
            a.apply_impulse_xyz(nx * j, ny * j, nz * j, px, py, pz)
        if im_b > 0.0:
            b.apply_impulse_xyz(-nx * j, -ny * j, -nz * j, px, py, pz)

    # Friction
    vax, vay, vaz = a.velocity_at_xyz(px, py, pz)
    vbx, vby, vbz = b.velocity_at_xyz(px, py, pz)
    rvx, rvy, rvz = vax - vbx, vay - vby, vaz - vbz
    vn = rvx * nx + rvy * ny + rvz * nz
    tx, ty, tz = rvx - nx * vn, rvy - ny * vn, rvz - nz * vn
    tlen = (tx * tx + ty * ty + tz * tz) ** 0.5
    if tlen > 1e-5:
        inv_t = 1.0 / tlen
        thx, thy, thz = tx * inv_t, ty * inv_t, tz * inv_t
        ang_ta = 0.0 if im_a <= 0.0 else a.angular_denom_xyz(rax, ray, raz, thx, thy, thz)
        ang_tb = 0.0 if im_b <= 0.0 else b.angular_denom_xyz(rbx, rby, rbz, thx, thy, thz)
        denom_t = im_a + im_b + ang_ta + ang_tb
        if denom_t > 1e-12:
            jt = -(rvx * thx + rvy * thy + rvz * thz) / denom_t
            mu = (a.friction + b.friction) * 0.5
            if resting:
                mu = mu * 1.4 if mu * 1.4 > 0.7 else 0.7
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
            if im_a > 0.0:
                a.apply_impulse_xyz(thx * jt, thy * jt, thz * jt, px, py, pz)
            if im_b > 0.0:
                b.apply_impulse_xyz(-thx * jt, -thy * jt, -thz * jt, px, py, pz)
