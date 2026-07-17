"""Core object hierarchy: Object → Abstract/Real → Control/Mobile/…"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from breve.engine import get_engine
from breve.shapes import Shape, Box
from breve.vector import Vector, vector


class Object:
    """Root of the breve instance tree."""

    def __init__(self) -> None:
        self._name = self.__class__.__name__
        self.init()

    def init(self) -> None:
        """Override for setup (classic breve called init after construct)."""

    def is_a(self, class_name: str) -> bool:
        return any(base.__name__ == class_name for base in self.__class__.__mro__)

    def isA(self, class_name: str) -> bool:  # noqa: N802
        return self.is_a(class_name)


class Abstract(Object):
    """No physical body — controllers, data, algorithms."""


class Real(Object):
    """Object with a place (and optional shape) in the world."""

    def __init__(self) -> None:
        self.location: Vector = vector(0, 0, 0)
        self.rotation: Vector = vector(0, 0, 0)
        self.color: Vector = vector(1, 1, 1)
        self.shape: Optional[Shape] = None
        self.enabled: bool = True
        self.neighborhood_size: float = 0.0
        self._collision_handlers: Dict[str, str] = {}
        self._show_neighbor_lines: bool = False
        get_engine().register_object(self)
        super().__init__()

    def move(self, loc: Vector) -> None:
        self.location = loc if isinstance(loc, Vector) else vector(*loc)

    def offset(self, delta: Vector) -> None:
        d = delta if isinstance(delta, Vector) else vector(*delta)
        self.location = self.location + d

    def get_location(self) -> Vector:
        return self.location

    def getLocation(self) -> Vector:  # noqa: N802
        return self.get_location()

    def set_shape(self, shape: Shape) -> None:
        self.shape = shape

    def setShape(self, shape: Shape) -> None:  # noqa: N802
        self.set_shape(shape)

    def set_color(self, color: Vector) -> None:
        self.color = color if isinstance(color, Vector) else vector(*color)

    def setColor(self, color: Vector) -> None:  # noqa: N802
        self.set_color(color)

    def set_neighborhood_size(self, size: float) -> None:
        self.neighborhood_size = float(size)

    def setNeighborhoodSize(self, size: float) -> None:  # noqa: N802
        self.set_neighborhood_size(size)

    def get_neighbors(self) -> List["Real"]:
        return get_engine().get_neighbors(self)

    def getNeighbors(self) -> List["Real"]:  # noqa: N802
        return self.get_neighbors()

    def handle_collisions(self, other_type: str, method_name: str) -> None:
        self._collision_handlers[other_type] = method_name

    def handleCollisions(self, other_type: str, method_name: str) -> None:  # noqa: N802
        self.handle_collisions(other_type, method_name)

    def show_neighbor_lines(self) -> None:
        self._show_neighbor_lines = True

    def showNeighborLines(self) -> None:  # noqa: N802
        self.show_neighbor_lines()

    def hide_neighbor_lines(self) -> None:
        self._show_neighbor_lines = False

    def hideNeighborLines(self) -> None:  # noqa: N802
        self.hide_neighbor_lines()

    def remove(self) -> None:
        get_engine().unregister_object(self)
        self.enabled = False


class Stationary(Real):
    """Collidable body that does not integrate motion (static physics collider)."""

    def set_shape(self, shape) -> None:
        super().set_shape(shape)
        eng = get_engine()
        if eng.physics_enabled and shape is not None:
            eng.register_physics_body(self, static=True)


class Floor(Stationary):
    """Infinite-ish ground plane at y=0 for landing demos."""

    def init(self) -> None:
        # Thin large box centered at origin so sphere collisions can hit it
        self.set_shape(Box().init_with(vector(200, 0.05, 200)))
        self.move(vector(0, -0.025, 0))
        self.set_color(vector(0.35, 0.4, 0.35))


class Mobile(Real):
    """Movable agent — kinematic by default; call enable_physics() for rigid body."""

    def __init__(self) -> None:
        self.velocity: Vector = vector(0, 0, 0)
        self.acceleration: Vector = vector(0, 0, 0)
        self.rotational_velocity: Vector = vector(0, 0, 0)
        self.frozen: bool = False
        self.wander_range: Vector = vector(10, 10, 10)
        self.max_speed: Optional[float] = None
        self.floor_y: Optional[float] = None
        self.physics_enabled: bool = False
        self.mass: float = 1.0
        super().__init__()

    def set_velocity(self, v: Vector) -> None:
        self.velocity = v if isinstance(v, Vector) else vector(*v)

    def setVelocity(self, v: Vector) -> None:  # noqa: N802
        self.set_velocity(v)

    def get_velocity(self) -> Vector:
        return self.velocity

    def getVelocity(self) -> Vector:  # noqa: N802
        return self.get_velocity()

    def set_acceleration(self, a: Vector) -> None:
        self.acceleration = a if isinstance(a, Vector) else vector(*a)

    def setAcceleration(self, a: Vector) -> None:  # noqa: N802
        self.set_acceleration(a)

    def get_acceleration(self) -> Vector:
        return self.acceleration

    def set_rotational_velocity(self, w: Vector) -> None:
        self.rotational_velocity = w if isinstance(w, Vector) else vector(*w)

    def setRotationalVelocity(self, w: Vector) -> None:  # noqa: N802
        self.set_rotational_velocity(w)

    def set_wander_range(self, r: Vector) -> None:
        self.wander_range = r if isinstance(r, Vector) else vector(*r)

    def setWanderRange(self, r: Vector) -> None:  # noqa: N802
        self.set_wander_range(r)

    def randomize_location(self) -> None:
        import numpy as np

        r = self.wander_range
        self.move(
            vector(
                (np.random.random() - 0.5) * 2 * r.x,
                (np.random.random() - 0.5) * 2 * r.y,
                (np.random.random() - 0.5) * 2 * r.z,
            )
        )

    def randomizeLocation(self) -> None:  # noqa: N802
        self.randomize_location()

    def get_angle(self, other: Real) -> float:
        """Angle between this agent's velocity and direction to other."""
        from breve.vector import angle as vec_angle

        to_other = other.location - self.location
        return vec_angle(self.velocity, to_other)

    def getAngle(self, other: Real) -> float:  # noqa: N802
        return self.get_angle(other)

    def enable_physics(self, mass: float = 1.0) -> None:
        """Register this mobile as a dynamic rigid body."""
        self.physics_enabled = True
        self.mass = float(mass)
        get_engine().register_physics_body(self, static=False, mass=self.mass)

    def enablePhysics(self, mass: float = 1.0) -> None:  # noqa: N802
        self.enable_physics(mass)

    def disable_physics(self) -> None:
        self.physics_enabled = False
        get_engine().physics.remove_body(self)

    def disablePhysics(self) -> None:  # noqa: N802
        self.disable_physics()

    def set_mass(self, mass: float) -> None:
        self.mass = float(mass)
        if self.physics_enabled:
            get_engine().register_physics_body(self, static=False, mass=self.mass)

    def setMass(self, mass: float) -> None:  # noqa: N802
        self.set_mass(mass)

    def iterate(self) -> None:
        """Per-agent hook; override in subclasses."""


class Control(Abstract):
    """
    Simulation controller — one per run.

    Call `.run()` to start (classic breve auto-ran on construct).
    """

    def __init__(self) -> None:
        self.engine = get_engine()
        self.engine.register_control(self)
        self.camera_target = vector(0, 0, 0)
        self.camera_offset = vector(0, 5, 20)
        self.camera_zoom = 20.0
        self.background_color = vector(0.2, 0.2, 0.3)
        self.lighting = False
        self.smooth_drawing = False
        self.shadows = False
        self.light_position = vector(0, 20, 20)
        self._menus: list[tuple[str, str]] = []
        super().__init__()

    def set_integration_step(self, dt: float) -> None:
        self.engine.set_integration_step(dt)

    def setIntegrationStep(self, dt: float) -> None:  # noqa: N802
        self.set_integration_step(dt)

    def set_iteration_step(self, dt: float) -> None:
        self.engine.set_iteration_step(dt)

    def setIterationStep(self, dt: float) -> None:  # noqa: N802
        self.set_iteration_step(dt)

    def point_camera(self, target: Vector, location: Optional[Vector] = None) -> None:
        self.camera_target = target if isinstance(target, Vector) else vector(*target)
        if location is not None:
            self.camera_offset = (
                location if isinstance(location, Vector) else vector(*location)
            )

    def pointCamera(self, target: Vector, location: Optional[Vector] = None) -> None:  # noqa: N802
        self.point_camera(target, location)

    def aim_camera(self, location: Vector) -> None:
        self.camera_target = location if isinstance(location, Vector) else vector(*location)

    def aimCamera(self, location: Vector) -> None:  # noqa: N802
        self.aim_camera(location)

    def zoom_camera(self, distance: float) -> None:
        self.camera_zoom = float(distance)

    def zoomCamera(self, distance: float) -> None:  # noqa: N802
        self.zoom_camera(distance)

    def offset_camera(self, offset: Vector) -> None:
        self.camera_offset = offset if isinstance(offset, Vector) else vector(*offset)

    def offsetCamera(self, offset: Vector) -> None:  # noqa: N802
        self.offset_camera(offset)

    def move_light(self, pos: Vector) -> None:
        self.light_position = pos if isinstance(pos, Vector) else vector(*pos)

    def moveLight(self, pos: Vector) -> None:  # noqa: N802
        self.move_light(pos)

    def enable_lighting(self) -> None:
        self.lighting = True

    def enableLighting(self) -> None:  # noqa: N802
        self.enable_lighting()

    def enable_smooth_drawing(self) -> None:
        self.smooth_drawing = True

    def enableSmoothDrawing(self) -> None:  # noqa: N802
        self.enable_smooth_drawing()

    def enable_shadows(self) -> None:
        self.shadows = True

    def enableShadows(self) -> None:  # noqa: N802
        self.enable_shadows()

    def set_background_color(self, color: Vector) -> None:
        self.background_color = color if isinstance(color, Vector) else vector(*color)

    def setBackgroundColor(self, color: Vector) -> None:  # noqa: N802
        self.set_background_color(color)

    def set_background_texture_image(self, *_args: Any) -> None:
        """Stub — textures not yet implemented."""

    def setBackgroundTextureImage(self, *args: Any) -> None:  # noqa: N802
        self.set_background_texture_image(*args)

    def update_neighbors(self) -> None:
        self.engine.update_neighbors()

    def updateNeighbors(self) -> None:  # noqa: N802
        self.update_neighbors()

    def add_menu(self, name: str, method_name: str) -> None:
        self._menus.append((name, method_name))
        return None

    def addMenu(self, name: str, method_name: str) -> Any:  # noqa: N802
        return self.add_menu(name, method_name)

    def add_menu_separator(self) -> None:
        self._menus.append(("", ""))

    def addMenuSeparator(self) -> None:  # noqa: N802
        self.add_menu_separator()

    def click(self, item: Any) -> None:
        """Selection hook for interactive frontends."""

    def iterate(self) -> None:
        """Controller per-step hook. Subclasses should call super().iterate()."""

    def run(self, steps: Optional[int] = None, max_time: Optional[float] = None) -> None:
        self.engine.run(steps=steps, max_time=max_time)

    def stop(self) -> None:
        self.engine.stop()


class PhysicalControl(Control):
    """Controller that turns on world gravity and sensible physics timesteps."""

    def __init__(self) -> None:
        super().__init__()

    def init(self) -> None:
        self.set_integration_step(0.008)
        self.set_iteration_step(0.02)
        self.full_gravity()
        self.enable_lighting()
        self.point_camera(vector(0, 0, 0), vector(0, 10, 30))

    def set_gravity(self, g: Vector) -> None:
        self.engine.set_gravity(g if isinstance(g, Vector) else vector(*g))

    def setGravity(self, g: Vector) -> None:  # noqa: N802
        self.set_gravity(g)

    def full_gravity(self) -> None:
        self.set_gravity(vector(0.0, -9.8, 0.0))

    def fullGravity(self) -> None:  # noqa: N802
        self.full_gravity()

    def half_gravity(self) -> None:
        self.set_gravity(vector(0.0, -4.9, 0.0))

    def halfGravity(self) -> None:  # noqa: N802
        self.half_gravity()

    def double_gravity(self) -> None:
        self.set_gravity(vector(0.0, -19.6, 0.0))

    def doubleGravity(self) -> None:  # noqa: N802
        self.double_gravity()

    def zero_gravity(self) -> None:
        self.set_gravity(vector(0.0, 0.0, 0.0))

    def zeroGravity(self) -> None:  # noqa: N802
        self.zero_gravity()

    def enable_shadow_volumes(self) -> None:
        self.shadows = True

    def enableShadowVolumes(self) -> None:  # noqa: N802
        self.enable_shadow_volumes()


