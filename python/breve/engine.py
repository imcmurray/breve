"""Simulation engine: timestep loop, neighborhoods, collisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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
    - Integrates kinematic Mobile agents (velocity / acceleration)
    - Neighborhood queries (spatial hash)
    - Sphere-ish collision handlers
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

    def reset(self) -> None:
        self.control = None
        self.objects.clear()
        self.time = 0.0
        self.running = False
        self._neighbors.clear()
        self._step_hooks.clear()

    def register_control(self, control: "Control") -> None:
        self.control = control

    def register_object(self, obj: "Real") -> None:
        if obj not in self.objects:
            self.objects.append(obj)

    def unregister_object(self, obj: "Real") -> None:
        if obj in self.objects:
            self.objects.remove(obj)

    def set_iteration_step(self, dt: float) -> None:
        self.iteration_step = float(dt)

    def set_integration_step(self, dt: float) -> None:
        self.integration_step = float(dt)

    def add_step_hook(self, fn) -> None:
        """Optional callback after each step (for viewers)."""
        self._step_hooks.append(fn)

    def update_neighbors(self) -> None:
        """Rebuild neighbor lists for all objects that have a neighborhood size."""
        self._neighbors.clear()
        candidates = [o for o in self.objects if o.enabled]
        # Group by cell using the max neighborhood among objects (simple + correct)
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

    def step(self) -> None:
        """Advance one iteration step (may substep integration)."""
        dt = self.iteration_step
        sub = max(self.integration_step, 1e-6)
        remaining = dt
        while remaining > 1e-12:
            h = min(sub, remaining)
            self._integrate(h)
            remaining -= h
        self._detect_collisions()
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
        from breve.objects import Mobile

        for obj in self.objects:
            if not isinstance(obj, Mobile) or obj.frozen:
                continue
            if obj.acceleration.length_squared() > 0:
                obj.velocity = obj.velocity + obj.acceleration * h
            if obj.max_speed is not None:
                obj.velocity = obj.velocity.limit(obj.max_speed)
            if obj.velocity.length_squared() > 0:
                obj.location = obj.location + obj.velocity * h
            if obj.rotational_velocity.length_squared() > 0:
                obj.rotation = obj.rotation + obj.rotational_velocity * h
            # Optional floor clamp
            if obj.floor_y is not None and obj.location.y < obj.floor_y:
                obj.location.y = obj.floor_y
                if obj.velocity.y < 0:
                    obj.velocity.y = 0.0

    def _detect_collisions(self) -> None:
        from breve.objects import Floor, Mobile

        bodies = [o for o in self.objects if o.shape is not None and o.enabled]
        floors = [o for o in bodies if isinstance(o, Floor)]
        mobiles = [o for o in bodies if isinstance(o, Mobile)]

        # Floor = infinite plane at y=0 (sphere–sphere vs huge box is wrong)
        for floor in floors:
            for mob in mobiles:
                radius = mob.shape.bounding_radius() if mob.shape else 0.0
                if mob.location.y - radius <= 0.0:
                    self._fire_collision(mob, floor)
                    self._fire_collision(floor, mob)

        # Sphere–sphere for non-floor pairs
        others = [o for o in bodies if not isinstance(o, Floor)]
        n = len(others)
        for i in range(n):
            a = others[i]
            ra = a.shape.bounding_radius() if a.shape else 0.0
            for j in range(i + 1, n):
                b = others[j]
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
