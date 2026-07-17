"""3D vector type with a breve-like API."""

from __future__ import annotations

from typing import Iterable, Iterator, SupportsFloat, Union

import numpy as np

Number = Union[float, int, np.floating, np.integer]


class Vector:
    """3D vector (float64). Supports classic breve arithmetic style."""

    __slots__ = ("_data",)

    def __init__(self, x: Number = 0.0, y: Number = 0.0, z: Number = 0.0):
        self._data = np.array([float(x), float(y), float(z)], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Vector":
        v = object.__new__(cls)
        v._data = np.asarray(arr, dtype=np.float64).reshape(3).copy()
        return v

    @property
    def x(self) -> float:
        return float(self._data[0])

    @x.setter
    def x(self, value: Number) -> None:
        self._data[0] = float(value)

    @property
    def y(self) -> float:
        return float(self._data[1])

    @y.setter
    def y(self, value: Number) -> None:
        self._data[1] = float(value)

    @property
    def z(self) -> float:
        return float(self._data[2])

    @z.setter
    def z(self, value: Number) -> None:
        self._data[2] = float(value)

    def copy(self) -> "Vector":
        return Vector.from_array(self._data)

    def length(self) -> float:
        return float(np.linalg.norm(self._data))

    def length_squared(self) -> float:
        return float(np.dot(self._data, self._data))

    def normalize(self) -> "Vector":
        n = self.length()
        if n == 0.0:
            return Vector(0, 0, 0)
        return Vector.from_array(self._data / n)

    def limit(self, max_length: float) -> "Vector":
        n = self.length()
        if n > max_length and n > 0:
            return Vector.from_array(self._data * (max_length / n))
        return self.copy()

    def dot(self, other: "Vector") -> float:
        return float(np.dot(self._data, other._data))

    def cross(self, other: "Vector") -> "Vector":
        return Vector.from_array(np.cross(self._data, other._data))

    def angle_to(self, other: "Vector") -> float:
        """Angle in radians between this vector and other (0 if either is zero)."""
        a = self.length()
        b = other.length()
        if a == 0.0 or b == 0.0:
            return 0.0
        cos = np.clip(self.dot(other) / (a * b), -1.0, 1.0)
        return float(np.arccos(cos))

    def __iter__(self) -> Iterator[float]:
        yield float(self._data[0])
        yield float(self._data[1])
        yield float(self._data[2])

    def __getitem__(self, index: int) -> float:
        return float(self._data[index])

    def __add__(self, other: "Vector") -> "Vector":
        return Vector.from_array(self._data + other._data)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector.from_array(self._data - other._data)

    def __mul__(self, scalar: Number) -> "Vector":
        return Vector.from_array(self._data * float(scalar))

    def __rmul__(self, scalar: Number) -> "Vector":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: Number) -> "Vector":
        return Vector.from_array(self._data / float(scalar))

    def __neg__(self) -> "Vector":
        return Vector.from_array(-self._data)

    def __bool__(self) -> bool:
        return self.length_squared() > 0.0

    def __repr__(self) -> str:
        return f"vector({self.x:g}, {self.y:g}, {self.z:g})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return bool(np.allclose(self._data, other._data))


def vector(x: Number = 0.0, y: Number = 0.0, z: Number = 0.0) -> Vector:
    """Factory matching classic `breve.vector(x, y, z)`."""
    if isinstance(x, Vector):
        return x.copy()
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)) and not isinstance(x, (int, float, np.floating, np.integer)):
        vals = list(x)  # type: ignore[arg-type]
        if len(vals) == 3 and float(y) == 0.0 and float(z) == 0.0:
            return Vector(vals[0], vals[1], vals[2])
    return Vector(float(x), float(y), float(z))  # type: ignore[arg-type]


def length(obj: Union[Vector, float, int, Iterable]) -> float:
    """Classic `breve.length` — vector magnitude or collection size."""
    if isinstance(obj, Vector):
        return obj.length()
    if isinstance(obj, (list, tuple, set)):
        return float(len(obj))
    if hasattr(obj, "__len__") and not isinstance(obj, (str, bytes, Vector)):
        try:
            return float(len(obj))  # type: ignore[arg-type]
        except TypeError:
            pass
    return float(obj)  # type: ignore[arg-type]


def angle(a: Vector, b: Vector) -> float:
    """Angle in radians between two vectors."""
    return a.angle_to(b)
