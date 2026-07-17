"""Helpers matching common breve scripting patterns."""

from __future__ import annotations

from typing import Any, Type, TypeVar

import numpy as np

from breve.vector import Vector, vector

T = TypeVar("T")


class object_list(list):
    """List of agents with broadcast-style helpers (subset of classic API)."""

    def randomize_location(self) -> None:
        for item in self:
            if hasattr(item, "randomize_location"):
                item.randomize_location()

    def randomizeLocation(self) -> None:  # noqa: N802
        self.randomize_location()

    def move(self, loc: Vector) -> None:
        for item in self:
            if hasattr(item, "move"):
                item.move(loc)

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [getattr(item, method)(*args, **kwargs) for item in self]

    def __getattr__(self, name: str):
        """Broadcast unknown methods to members (e.g. birds.flockNormally())."""
        if name.startswith("_"):
            raise AttributeError(name)

        def _broadcast(*args: Any, **kwargs: Any) -> list[Any]:
            results = []
            for item in self:
                method = getattr(item, name, None)
                if callable(method):
                    results.append(method(*args, **kwargs))
            return results

        return _broadcast


def create_instances(cls: Type[T], count: int = 1) -> T | object_list:
    """Create one instance or an object_list of `count` instances."""
    if count == 1:
        return cls()
    return object_list(cls() for _ in range(count))


def random_expression(v):
    """
    Classic `randomExpression`.

    - vector → uniform [0,x)×[0,y)×[0,z)
    - number → uniform [0, v)
    """
    if isinstance(v, Vector):
        return vector(
            np.random.random() * v.x,
            np.random.random() * v.y,
            np.random.random() * v.z,
        )
    return float(np.random.random() * float(v))


# CamelCase aliases used by auto-converted demos
randomExpression = random_expression
createInstances = create_instances
objectList = object_list
