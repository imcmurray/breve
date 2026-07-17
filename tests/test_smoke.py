"""Smoke tests for the modern breve slice."""

from __future__ import annotations

import breve
from breve.engine import Engine, set_engine


def test_vector_ops():
    a = breve.vector(1, 2, 3)
    b = breve.vector(3, 2, 1)
    assert (a + b) == breve.vector(4, 4, 4)
    assert abs(a.length() - (1 + 4 + 9) ** 0.5) < 1e-9


def test_hello_world_runs():
    set_engine(Engine())

    class Hello(breve.Control):
        def __init__(self):
            self.ticks = 0
            super().__init__()

        def iterate(self):
            self.ticks += 1
            super().iterate()

    h = Hello()
    h.run(steps=3)
    assert h.ticks == 3
    assert h.engine.time == pytest_approx(0.3)


def pytest_approx(val, rel=1e-9):
    class A:
        def __eq__(self, other):
            return abs(other - val) <= rel * max(1.0, abs(val))

    return A()


def test_fountain_particles_move():
    set_engine(Engine())

    class P(breve.Mobile):
        def init(self):
            self.set_shape(breve.Sphere().init_with(0.2))
            self.set_velocity(breve.vector(0, 10, 0))
            self.set_acceleration(breve.vector(0, -9.8, 0))

    class C(breve.Control):
        def init(self):
            self.p = P()
            self.set_iteration_step(0.1)
            self.set_integration_step(0.05)

    c = C()
    y0 = c.p.location.y
    c.run(steps=5)
    # under gravity from vy=10, should still be above start for a short time or have moved
    assert c.p.location.y != y0 or c.p.velocity.y != 10.0


def test_collision_handler_fires():
    set_engine(Engine())
    hits = {"n": 0}

    class Ball(breve.Mobile):
        def init(self):
            self.set_shape(breve.Sphere().init_with(1.0))
            self.handle_collisions("Ball", "on_hit")

        def on_hit(self, other):
            hits["n"] += 1

    class C(breve.Control):
        def init(self):
            self.a = Ball()
            self.b = Ball()
            self.a.move(breve.vector(0, 0, 0))
            self.b.move(breve.vector(0.5, 0, 0))  # overlapping

    C().run(steps=1)
    assert hits["n"] >= 2  # each fires on the other
