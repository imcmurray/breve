"""Simulation engine: timestep loop, neighborhoods, physics, collisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from breve.physics import PhysicsWorld
from breve.vector import Vector, vector

if TYPE_CHECKING:
    from breve.objects import Control, Real

_engine: Optional["Engine"] = None


def get_engine() -> "Engine":
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


def set_engine(engine: Optional["Engine"]) -> None:
    global _engine
    _engine = engine


class Engine:
    """
    World stepper.

    - Registers Real bodies and the active Control
    - Integrates kinematic Mobile agents OR rigid-body physics
    - Neighborhood queries
    - Sphere collision *handlers* (script callbacks) for non-physics contacts
    """

    def __init__(self) -> None:
        self.control: Optional["Control"] = None
        self.objects: List["Real"] = []
        self.time: float = 0.0
        self.iteration_step: float = 0.1
        self.integration_step: float = 0.05
        self.running: bool = False
        self._neighbors: Dict[int, List["Real"]] = {}
        self._step_hooks: List = []
        self.physics = PhysicsWorld()
        self.physics_enabled: bool = False

    def reset(self) -> None:
        self.control = None
        self.objects.clear()
        self.time = 0.0
        self.running = False
        self._neighbors.clear()
        self._step_hooks.clear()
        self.physics.clear()
        self.physics_enabled = False

    def register_control(self, control: "Control") -> None:
        self.control = control

    def register_object(self, obj: "Real") -> None:
        if obj not in self.objects:
            self.objects.append(obj)

    def unregister_object(self, obj: "Real") -> None:
        if obj in self.objects:
            self.objects.remove(obj)
        self.physics.remove_body(obj)

    def set_iteration_step(self, dt: float) -> None:
        self.iteration_step = float(dt)

    def set_integration_step(self, dt: float) -> None:
        self.integration_step = float(dt)

    def set_gravity(self, g: Vector) -> None:
        self.physics.set_gravity(g.x, g.y, g.z)
        self.physics_enabled = True

    def enable_physics(self) -> None:
        self.physics_enabled = True

    def add_step_hook(self, fn) -> None:
        self._step_hooks.append(fn)

    def update_neighbors(self) -> None:
        self._neighbors.clear()
        candidates = [o for o in self.objects if o.enabled]
        for obj in candidates:
            radius = getattr(obj, "neighborhood_size", 0.0) or 0.0
            if radius <= 0:
                self._neighbors[id(obj)] = []
                continue
            r2 = radius * radius
            neighbors: List["Real"] = []
            for other in candidates:
                if other is obj:
                    continue
                d = obj.location - other.location
                if d.length_squared() <= r2:
                    neighbors.append(other)
            self._neighbors[id(obj)] = neighbors

    def get_neighbors(self, obj: "Real") -> List["Real"]:
        return self._neighbors.get(id(obj), [])

    def register_physics_body(self, obj: "Real", *, static: bool, mass: float = 1.0) -> None:
        self.physics_enabled = True
        self.physics.add_or_update(obj, static=static, mass=mass)

    def step(self) -> None:
        """Advance one iteration step (may substep integration)."""
        dt = self.iteration_step
        sub = max(self.integration_step, 1e-6)
        remaining = dt
        while remaining > 1e-12:
            h = min(sub, remaining)
            self._integrate(h)
            remaining -= h
        self._detect_script_collisions()
        for obj in list(self.objects):
            iterate = getattr(obj, "iterate", None)
            if callable(iterate):
                iterate()
        if self.control is not None:
            self.control.iterate()
        self.time += dt
        for hook in self._step_hooks:
            hook(self)

    def _integrate(self, h: float) -> None:
        from breve.objects import Mobile, Stationary

        if self.physics_enabled:
            # Ensure bodies exist once; physics is the authority for pose/velocity
            # after creation (avoids clobbering spin / friction every substep).
            for obj in self.objects:
                if not obj.enabled or obj.shape is None:
                    continue
                if isinstance(obj, Mobile) and getattr(obj, "physics_enabled", False):
                    if self.physics.get_body(obj) is None:
                        mass = getattr(obj, "mass", 1.0)
                        self.physics.add_or_update(obj, static=False, mass=mass)
                elif isinstance(obj, Stationary):
                    if self.physics.get_body(obj) is None:
                        self.physics.add_or_update(obj, static=True, mass=0.0)

            self.physics.step(h)
            self.physics.sync_to_owners()

        # kinematic mobiles (no physics flag)
        for obj in self.objects:
            if not isinstance(obj, Mobile) or obj.frozen:
                continue
            if getattr(obj, "physics_enabled", False):
                continue
            if obj.acceleration.length_squared() > 0:
                obj.velocity = obj.velocity + obj.acceleration * h
            if obj.max_speed is not None:
                obj.velocity = obj.velocity.limit(obj.max_speed)
            if obj.velocity.length_squared() > 0:
                obj.location = obj.location + obj.velocity * h
            if obj.rotational_velocity.length_squared() > 0:
                obj.rotation = obj.rotation + obj.rotational_velocity * h
            if obj.floor_y is not None and obj.location.y < obj.floor_y:
                obj.location.y = obj.floor_y
                if obj.velocity.y < 0:
                    obj.velocity.y = 0.0

    def _detect_script_collisions(self) -> None:
        """Script collision handlers (handle_collisions). Skip rigid-solver pairs."""
        from breve.objects import Floor, Mobile

        bodies = [o for o in self.objects if o.shape is not None and o.enabled]
        floors = [o for o in bodies if isinstance(o, Floor)]
        mobiles = [o for o in bodies if isinstance(o, Mobile)]

        for floor in floors:
            for mob in mobiles:
                if getattr(mob, "physics_enabled", False):
                    continue
                radius = mob.shape.bounding_radius() if mob.shape else 0.0
                if mob.location.y - radius <= 0.0:
                    self._fire_collision(mob, floor)
                    self._fire_collision(floor, mob)

        others = [o for o in bodies if not isinstance(o, Floor)]
        n = len(others)
        for i in range(n):
            a = others[i]
            ra = a.shape.bounding_radius() if a.shape else 0.0
            for j in range(i + 1, n):
                b = others[j]
                # rigid solver owns contact if both are in the physics world
                if (
                    self.physics.get_body(a) is not None
                    and self.physics.get_body(b) is not None
                ):
                    continue
                rb = b.shape.bounding_radius() if b.shape else 0.0
                delta = a.location - b.location
                rsum = ra + rb
                if delta.length_squared() <= rsum * rsum:
                    self._fire_collision(a, b)
                    self._fire_collision(b, a)

    def _fire_collision(self, owner: "Real", other: "Real") -> None:
        handlers = getattr(owner, "_collision_handlers", {})
        handler_name = handlers.get(type(other).__name__)
        if handler_name is None:
            for cls_name, method in handlers.items():
                if other.__class__.__name__ == cls_name or any(
                    base.__name__ == cls_name for base in other.__class__.__mro__
                ):
                    handler_name = method
                    break
        if handler_name is None:
            return
        method = getattr(owner, handler_name, None)
        if callable(method):
            method(other)

    def run(self, steps: Optional[int] = None, max_time: Optional[float] = None) -> None:
        self.running = True
        count = 0
        try:
            while self.running:
                self.step()
                count += 1
                if steps is not None and count >= steps:
                    break
                if max_time is not None and self.time >= max_time:
                    break
        finally:
            self.running = False

    def stop(self) -> None:
        self.running = False

    def snapshot_positions(self) -> List[Tuple[float, float, float]]:
        return [(o.location.x, o.location.y, o.location.z) for o in self.objects if o.enabled]
