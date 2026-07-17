"""
Lightweight rigid-body physics for breve (pure Python / NumPy).

Enough for Gravity (balls + static steps), stacking/demolition, and bouncing
worlds without ODE/PyBullet. Not a full multibody/joint solver — that comes
later (Rapier).
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


@dataclass
class Collider:
    kind: ShapeKind
    # sphere: radius; box: half-extents (x,y,z)
    data: np.ndarray


@dataclass
class RigidBody:
    """Internal physics body, linked to a breve Real via `owner`."""

    owner: object
    position: np.ndarray
    velocity: np.ndarray
    # angular velocity (world); orientation kept simple (no full quat yet for boxes)
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    mass: float = 1.0
    inv_mass: float = 1.0
    restitution: float = 0.45
    friction: float = 0.4
    static: bool = False
    collider: Optional[Collider] = None
    # accumulated forces this substep
    force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    awake: bool = True

    def set_static(self, static: bool = True) -> None:
        self.static = static
        if static:
            self.inv_mass = 0.0
            self.velocity[:] = 0.0
            self.angular_velocity[:] = 0.0
        else:
            self.inv_mass = 0.0 if self.mass <= 0 else 1.0 / self.mass

    def apply_force(self, f: np.ndarray) -> None:
        if not self.static:
            self.force += f
            self.awake = True

    def apply_impulse(self, impulse: np.ndarray) -> None:
        if not self.static and self.inv_mass > 0:
            self.velocity += impulse * self.inv_mass
            self.awake = True


@dataclass
class Contact:
    a: RigidBody
    b: RigidBody
    normal: np.ndarray  # from b → a (points out of b into a)
    penetration: float
    point: np.ndarray


class PhysicsWorld:
    """
    Gravity + sequential impulse contacts.

    Supports dynamic spheres and static/dynamic AABB boxes.
    """

    def __init__(self) -> None:
        self.gravity = np.array([0.0, -9.8, 0.0], dtype=np.float64)
        self.bodies: List[RigidBody] = []
        self._by_owner: dict = {}
        self.iterations: int = 8
        self.slop: float = 0.005
        self.baumgarte: float = 0.2
        self.linear_damping: float = 0.01
        self.sleep_eps: float = 0.04
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
        existing = self._by_owner.get(id(owner))
        pos = np.array(
            [owner.location.x, owner.location.y, owner.location.z], dtype=np.float64
        )
        vel = np.zeros(3, dtype=np.float64)
        if hasattr(owner, "velocity"):
            vel = np.array(
                [owner.velocity.x, owner.velocity.y, owner.velocity.z], dtype=np.float64
            )

        collider = _collider_from_shape(getattr(owner, "shape", None))

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

        existing.position[:] = pos
        existing.velocity[:] = vel
        existing.mass = max(mass, 1e-6)
        existing.restitution = restitution
        existing.friction = friction
        existing.collider = collider
        existing.set_static(static)
        return existing

    def sync_from_owners(self) -> None:
        """Pull positions/shapes from breve objects (e.g. after move/reset)."""
        for body in self.bodies:
            owner = body.owner
            if owner is None or not getattr(owner, "enabled", True):
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
            body.collider = _collider_from_shape(getattr(owner, "shape", None))

    def sync_to_owners(self) -> None:
        """Push physics state back onto breve objects."""
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
                # acceleration is managed by physics; zero script accel when physical
                if getattr(owner, "physics_enabled", False):
                    owner.acceleration = vector(0, 0, 0)

    def step(self, dt: float) -> None:
        if not self.enabled or dt <= 0:
            return

        # integrate velocities
        for body in self.bodies:
            if body.static or not body.awake:
                body.force[:] = 0.0
                continue
            # F = mg + user forces
            body.force += self.gravity * body.mass
            body.velocity += body.force * body.inv_mass * dt
            body.velocity *= max(0.0, 1.0 - self.linear_damping)
            body.force[:] = 0.0

        # integrate positions
        for body in self.bodies:
            if body.static or not body.awake:
                continue
            body.position += body.velocity * dt

        # contacts
        contacts = self._find_contacts()
        for _ in range(self.iterations):
            for c in contacts:
                _resolve_contact(c, self.slop, self.baumgarte, dt)

        # sleep small velocities
        for body in self.bodies:
            if body.static:
                continue
            speed = float(np.linalg.norm(body.velocity))
            if speed < self.sleep_eps:
                body.velocity *= 0.5
            if speed < self.sleep_eps * 0.25 and abs(body.position[1]) < 1e6:
                # don't fully sleep mid-air; only near support roughly
                pass

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


def _collider_from_shape(shape: Optional["Shape"]) -> Optional[Collider]:
    if shape is None:
        return None
    from breve.shapes import Box, Sphere

    if isinstance(shape, Sphere):
        return Collider(ShapeKind.SPHERE, np.array([float(shape.radius)], dtype=np.float64))
    if isinstance(shape, Box):
        # size is full extents in breve API
        hx = float(shape.size.x) * 0.5
        hy = float(shape.size.y) * 0.5
        hz = float(shape.size.z) * 0.5
        return Collider(ShapeKind.BOX, np.array([hx, hy, hz], dtype=np.float64))
    # fallback: bounding sphere
    r = float(shape.bounding_radius())
    return Collider(ShapeKind.SPHERE, np.array([r], dtype=np.float64))


def _collide(a: RigidBody, b: RigidBody) -> Optional[Contact]:
    assert a.collider and b.collider
    ka, kb = a.collider.kind, b.collider.kind
    if ka == ShapeKind.SPHERE and kb == ShapeKind.SPHERE:
        return _sphere_sphere(a, b)
    if ka == ShapeKind.SPHERE and kb == ShapeKind.BOX:
        return _sphere_box(a, b)
    if ka == ShapeKind.BOX and kb == ShapeKind.SPHERE:
        c = _sphere_box(b, a)
        if c is None:
            return None
        # flip normal to keep convention (out of second into first in our resolve)
        return Contact(a, b, -c.normal, c.penetration, c.point)
    if ka == ShapeKind.BOX and kb == ShapeKind.BOX:
        return _box_box(a, b)
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


def _sphere_box(sphere: RigidBody, box: RigidBody) -> Optional[Contact]:
    """Sphere vs AABB (axis-aligned box in world space)."""
    r = float(sphere.collider.data[0])  # type: ignore[index]
    half = box.collider.data  # type: ignore[union-attr]
    # closest point on AABB to sphere center
    local = sphere.position - box.position
    clamped = np.minimum(half, np.maximum(-half, local))
    closest = box.position + clamped
    delta = sphere.position - closest
    dist = float(np.linalg.norm(delta))

    # sphere center inside box
    if dist < 1e-12:
        # push out along minimum penetration axis
        faces = [
            (half[0] - local[0], np.array([1.0, 0.0, 0.0])),
            (half[0] + local[0], np.array([-1.0, 0.0, 0.0])),
            (half[1] - local[1], np.array([0.0, 1.0, 0.0])),
            (half[1] + local[1], np.array([0.0, -1.0, 0.0])),
            (half[2] - local[2], np.array([0.0, 0.0, 1.0])),
            (half[2] + local[2], np.array([0.0, 0.0, -1.0])),
        ]
        pen, normal = min(faces, key=lambda t: t[0])
        # sphere is inside: penetration = pen + r
        return Contact(sphere, box, normal, float(pen) + r, closest)

    normal = delta / dist
    pen = r - dist
    if pen <= 0:
        return None
    return Contact(sphere, box, normal, pen, closest)


def _box_box(a: RigidBody, b: RigidBody) -> Optional[Contact]:
    """AABB vs AABB (SAT on axes)."""
    ha = a.collider.data  # type: ignore[union-attr]
    hb = b.collider.data  # type: ignore[union-attr]
    d = a.position - b.position
    px = ha[0] + hb[0] - abs(d[0])
    py = ha[1] + hb[1] - abs(d[1])
    pz = ha[2] + hb[2] - abs(d[2])
    if px <= 0 or py <= 0 or pz <= 0:
        return None
    # min penetration axis
    if px <= py and px <= pz:
        n = np.array([1.0 if d[0] >= 0 else -1.0, 0.0, 0.0])
        pen = float(px)
    elif py <= px and py <= pz:
        n = np.array([0.0, 1.0 if d[1] >= 0 else -1.0, 0.0])
        pen = float(py)
    else:
        n = np.array([0.0, 0.0, 1.0 if d[2] >= 0 else -1.0])
        pen = float(pz)
    point = a.position - n * (ha @ np.abs(n))  # rough contact point
    return Contact(a, b, n, pen, point)


def _resolve_contact(c: Contact, slop: float, baumgarte: float, dt: float) -> None:
    a, b = c.a, c.b
    n = c.normal
    # relative velocity
    rv = a.velocity - b.velocity
    vel_n = float(np.dot(rv, n))

    e = min(a.restitution, b.restitution)
    inv_mass_sum = a.inv_mass + b.inv_mass
    if inv_mass_sum <= 0:
        return

    # only apply bounce when approaching
    if vel_n < 0:
        j = -(1.0 + e) * vel_n / inv_mass_sum
        impulse = n * j
        a.apply_impulse(impulse)
        b.apply_impulse(-impulse)

        # friction (coulomb-ish)
        rv = a.velocity - b.velocity
        tangent = rv - n * float(np.dot(rv, n))
        tlen = float(np.linalg.norm(tangent))
        if tlen > 1e-8:
            t = tangent / tlen
            jt = -float(np.dot(rv, t)) / inv_mass_sum
            mu = (a.friction + b.friction) * 0.5
            jt = float(np.clip(jt, -j * mu, j * mu))
            a.apply_impulse(t * jt)
            b.apply_impulse(-t * jt)

    # positional correction (Baumgarte)
    pen = max(c.penetration - slop, 0.0)
    if pen > 0:
        correction = n * (pen * baumgarte / inv_mass_sum)
        if not a.static:
            a.position += correction * a.inv_mass
            a.awake = True
        if not b.static:
            b.position -= correction * b.inv_mass
            b.awake = True
