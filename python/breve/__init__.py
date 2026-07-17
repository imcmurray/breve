"""
breve — modern Python 3 multi-agent / artificial-life simulator.

Preserves the Control / Mobile / iterate programming model from Jon Klein's
classic breve, without the legacy C++ / Python 2 engine.
"""

from __future__ import annotations

from breve.vector import vector, Vector, length, angle
from breve.engine import Engine, get_engine, set_engine
from breve.objects import (
    Object,
    Abstract,
    Real,
    Control,
    PhysicalControl,
    Mobile,
    Stationary,
    Floor,
)
from breve.shapes import Shape, Sphere, Cube, Box
from breve.util import object_list, create_instances, random_expression
from breve.physics import PhysicsWorld

# scene / AI helpers imported lazily in CLI to keep core light

__version__ = "0.1.0a1"
__all__ = [
    "vector",
    "Vector",
    "length",
    "angle",
    "Engine",
    "get_engine",
    "set_engine",
    "Object",
    "Abstract",
    "Real",
    "Control",
    "PhysicalControl",
    "Mobile",
    "Stationary",
    "Floor",
    "Shape",
    "Sphere",
    "Cube",
    "Box",
    "object_list",
    "create_instances",
    "random_expression",
    "PhysicsWorld",
    "__version__",
]

objectList = object_list
createInstances = create_instances
randomExpression = random_expression
