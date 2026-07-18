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

if TYPE_CHECKING:
    from breve.objects import Real
    from breve.shapes import Shape


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
    return np.cross(a, b)


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
        return _q_to_matrix(self.orientation)

    def inv_inertia_world(self) -> np.ndarray:
        """World-space inverse inertia tensor (3×3)."""
        if self.static:
            return np.zeros((3, 3), dtype=np.float64)
        R = self.rotation_matrix()
        inv_local = np.diag(self.inv_inertia_local)
        return R @ inv_local @ R.T

    def apply_impulse_at(self, impulse: np.ndarray, point: np.ndarray) -> None:
        if self.static or self.inv_mass <= 0:
            return
        self.velocity = self.velocity + impulse * self.inv_mass
        r = point - self.position
        self.angular_velocity = self.angular_velocity + self.inv_inertia_world() @ _cross(
            r, impulse
        )
        self.awake = True

    def velocity_at_point(self, point: np.ndarray) -> np.ndarray:
        return self.velocity + _cross(self.angular_velocity, point - self.position)


@dataclass
class Contact:
    a: RigidBody
    b: RigidBody
    normal: np.ndarray  # unit, points from b toward a (separating direction for a)
    penetration: float
    point: np.ndarray


class PhysicsWorld:
    def __init__(self) -> None:
        self.gravity = np.array([0.0, -9.8, 0.0], dtype=np.float64)
        self.bodies: List[RigidBody] = []
        self._by_owner: dict = {}
        self.iterations: int = 12
        self.slop: float = 0.005
        self.baumgarte: float = 0.25
        self.linear_damping: float = 0.005
        self.angular_damping: float = 0.02
        self.bounce_threshold: float = 0.4
        self.enabled: bool = True

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

        for body in self.bodies:
            if body.static or not body.awake:
                body.force[:] = 0.0
                body.torque[:] = 0.0
                continue
            body.force = body.force + self.gravity * body.mass
            body.velocity = body.velocity + body.force * body.inv_mass * dt
            body.angular_velocity = (
                body.angular_velocity + body.inv_inertia_world() @ body.torque * dt
            )
            body.velocity *= max(0.0, 1.0 - self.linear_damping)
            body.angular_velocity *= max(0.0, 1.0 - self.angular_damping)
            # clamp runaway spin/speed from deep-contact iterations
            speed = float(np.linalg.norm(body.velocity))
            if speed > 60.0:
                body.velocity *= 60.0 / speed
            w = float(np.linalg.norm(body.angular_velocity))
            if w > 25.0:
                body.angular_velocity *= 25.0 / w
            body.force[:] = 0.0
            body.torque[:] = 0.0

        for body in self.bodies:
            if body.static or not body.awake:
                continue
            body.position = body.position + body.velocity * dt
            body.orientation = _q_integrate(
                body.orientation, body.angular_velocity, dt
            )

        contacts = self._find_contacts()
        for _ in range(self.iterations):
            for c in contacts:
                _resolve_contact(
                    c, self.slop, self.baumgarte, self.bounce_threshold, dt
                )

        # Rest only when actually in contact and nearly still — never damp
        # free-fall or the gravity accumulation / tipping torque dies.
        in_contact = set()
        for c in contacts:
            in_contact.add(id(c.a))
            in_contact.add(id(c.b))
        for body in self.bodies:
            if body.static or id(body) not in in_contact:
                continue
            speed = float(np.linalg.norm(body.velocity))
            spin = float(np.linalg.norm(body.angular_velocity))
            if speed < 0.04 and spin < 0.08:
                body.velocity[:] = 0.0
                body.angular_velocity[:] = 0.0

    def _find_contacts(self) -> List[Contact]:
        contacts: List[Contact] = []
        n = len(self.bodies)
        for i in range(n):
            a = self.bodies[i]
            if a.collider is None or not getattr(a.owner, "enabled", True):
                continue
            for j in range(i + 1, n):
                b = self.bodies[j]
                if b.collider is None or not getattr(b.owner, "enabled", True):
                    continue
                if a.static and b.static:
                    continue
                hit = _collide(a, b)
                if hit is not None:
                    contacts.append(hit)
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


def _collide(a: RigidBody, b: RigidBody) -> Optional[Contact]:
    assert a.collider and b.collider
    ka, kb = a.collider.kind, b.collider.kind
    if ka == ShapeKind.SPHERE and kb == ShapeKind.SPHERE:
        return _sphere_sphere(a, b)
    if ka == ShapeKind.SPHERE and kb == ShapeKind.BOX:
        return _sphere_obb(a, b)
    if ka == ShapeKind.BOX and kb == ShapeKind.SPHERE:
        c = _sphere_obb(b, a)
        if c is None:
            return None
        return Contact(a, b, -c.normal, c.penetration, c.point)
    if ka == ShapeKind.BOX and kb == ShapeKind.BOX:
        return _obb_obb(a, b)
    return None


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


def _obb_obb(a: RigidBody, b: RigidBody) -> Optional[Contact]:
    """
    Oriented box vs oriented box (SAT).
    Returns contact with normal pointing from b toward a.
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
    best_n = None

    def consider(pen: float, axis_world: np.ndarray) -> None:
        nonlocal min_pen, best_n
        nlen = float(np.linalg.norm(axis_world))
        if nlen < 1e-10:
            return
        axis_world = axis_world / nlen
        if float(np.dot(axis_world, t)) < 0:
            axis_world = -axis_world
        if pen < min_pen:
            min_pen = pen
            best_n = axis_world

    for i in range(3):
        ra = float(ha[i])
        rb = float(hb[0] * AbsR[i, 0] + hb[1] * AbsR[i, 1] + hb[2] * AbsR[i, 2])
        pen = ra + rb - abs(float(t_a[i]))
        if pen <= 0:
            return None
        consider(pen, Ae[i])

    t_b = Rb.T @ t
    for i in range(3):
        ra = float(ha[0] * AbsR[0, i] + ha[1] * AbsR[1, i] + ha[2] * AbsR[2, i])
        rb = float(hb[i])
        pen = ra + rb - abs(float(t_b[i]))
        if pen <= 0:
            return None
        consider(pen, Be[i])

    for i in range(3):
        for j in range(3):
            axis = _cross(Ae[i], Be[j])
            nlen = float(np.linalg.norm(axis))
            if nlen < 1e-8:
                continue
            axis_n = axis / nlen
            ra = float(
                ha[0] * abs(float(np.dot(Ae[0], axis_n)))
                + ha[1] * abs(float(np.dot(Ae[1], axis_n)))
                + ha[2] * abs(float(np.dot(Ae[2], axis_n)))
            )
            rb = float(
                hb[0] * abs(float(np.dot(Be[0], axis_n)))
                + hb[1] * abs(float(np.dot(Be[1], axis_n)))
                + hb[2] * abs(float(np.dot(Be[2], axis_n)))
            )
            pen = ra + rb - abs(float(np.dot(t, axis_n)))
            if pen <= 0:
                return None
            consider(pen, axis_n)

    if best_n is None or min_pen >= 1e29:
        return None

    n = best_n
    # normal from b toward a
    if float(np.dot(n, a.position - b.position)) < 0:
        n = -n

    point = _contact_point_on_normal(a, b, n)
    return Contact(a, b, n, float(min_pen), point)


def _resolve_contact(
    c: Contact,
    slop: float,
    baumgarte: float,
    bounce_threshold: float,
    dt: float = 1.0 / 60.0,
) -> None:
    a, b = c.a, c.b
    n = c.normal
    p = c.point

    # relative velocity at contact point (includes spin)
    va = a.velocity_at_point(p)
    vb = b.velocity_at_point(p)
    rv = va - vb
    vel_n = float(np.dot(rv, n))

    e = float(np.sqrt(max(a.restitution, 0.0) * max(b.restitution, 0.0)))
    ra = p - a.position
    rb = p - b.position

    def angular_denom(body: RigidBody, r: np.ndarray, axis: np.ndarray) -> float:
        """(r × axis) · I^{-1} (r × axis) — correct effective mass term."""
        if body.static:
            return 0.0
        r_x_axis = _cross(r, axis)
        return float(np.dot(r_x_axis, body.inv_inertia_world() @ r_x_axis))

    denom = (
        a.inv_mass
        + b.inv_mass
        + angular_denom(a, ra, n)
        + angular_denom(b, rb, n)
    )
    if denom <= 1e-12:
        return

    # Baumgarte bias: push out of penetration as a velocity goal
    pen = max(c.penetration - slop, 0.0)
    bias = (baumgarte * pen / max(dt, 1e-4)) if pen > 0 else 0.0

    # Resting: no bounce when impact is soft
    impact = max(0.0, -vel_n)
    e_eff = e if impact > bounce_threshold else 0.0

    # Remove relative normal velocity + bias (works for resting and impact)
    j = (-(vel_n + bias) - e_eff * min(vel_n, 0.0)) / denom
    # only apply if it separates (prevents sticky pull when already separating hard)
    if j < 0:
        # allowing small negative for bias-only cases: recompute
        j = -(vel_n + bias) / denom
    j = float(np.clip(j, 0.0, 50.0))  # repulsive only
    if j > 0:
        impulse = n * j
        a.apply_impulse_at(impulse, p)
        b.apply_impulse_at(-impulse, p)

    # Friction whenever we have a contact (kills skating / endless spin on floors)
    va = a.velocity_at_point(p)
    vb = b.velocity_at_point(p)
    rv = va - vb
    tangent = rv - n * float(np.dot(rv, n))
    tlen = float(np.linalg.norm(tangent))
    if tlen > 1e-5:
        t_hat = tangent / tlen
        denom_t = (
            a.inv_mass
            + b.inv_mass
            + angular_denom(a, ra, t_hat)
            + angular_denom(b, rb, t_hat)
        )
        if denom_t > 1e-12:
            jt = -float(np.dot(rv, t_hat)) / denom_t
            mu = (a.friction + b.friction) * 0.5
            # stronger grip when nearly resting
            if impact < bounce_threshold:
                mu = max(mu, 0.6)
            jt = float(np.clip(jt, -j * mu if j > 1e-8 else -mu * 2.0, j * mu if j > 1e-8 else mu * 2.0))
            a.apply_impulse_at(t_hat * jt, p)
            b.apply_impulse_at(-t_hat * jt, p)

    # Light linear split for deep penetration (after velocity solve)
    if pen > slop * 2:
        inv = a.inv_mass + b.inv_mass
        if inv > 1e-12:
            correction = n * (0.2 * pen / inv)
            if not a.static:
                a.position = a.position + correction * a.inv_mass
            if not b.static:
                b.position = b.position - correction * b.inv_mass
