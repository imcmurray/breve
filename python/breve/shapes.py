"""Primitive shapes for collision / rendering bounds."""

from __future__ import annotations

from dataclasses import dataclass, field

from breve.vector import Vector, vector


@dataclass
class Shape:
    """Base shape. Subclasses define extents used by the simple collider."""

    def bounding_radius(self) -> float:
        raise NotImplementedError

    def clone(self) -> "Shape":
        raise NotImplementedError


@dataclass
class Sphere(Shape):
    radius: float = 1.0

    def init_with(self, radius: float) -> "Sphere":
        self.radius = float(radius)
        return self

    def initWith(self, radius: float) -> "Sphere":  # noqa: N802
        return self.init_with(radius)

    def bounding_radius(self) -> float:
        return self.radius

    def clone(self) -> "Sphere":
        return Sphere(self.radius)


@dataclass
class Box(Shape):
    size: Vector = field(default_factory=lambda: vector(1, 1, 1))

    def init_with(self, size: Vector) -> "Box":
        self.size = size if isinstance(size, Vector) else vector(*size)
        return self

    def initWith(self, size: Vector) -> "Box":  # noqa: N802
        return self.init_with(size)

    def bounding_radius(self) -> float:
        s = self.size
        return 0.5 * (s.x**2 + s.y**2 + s.z**2) ** 0.5

    def clone(self) -> "Box":
        return Box(self.size.copy())


Cube = Box
