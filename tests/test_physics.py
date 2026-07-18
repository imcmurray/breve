"""Rigid-body physics tests."""

from __future__ import annotations

import numpy as np

import breve
from breve.engine import Engine, set_engine


def test_ball_falls_under_gravity():
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.005)
            self.set_iteration_step(0.02)
            self.full_gravity()
            ground = breve.Stationary()
            ground.set_shape(breve.Box().init_with(breve.vector(4, 0.2, 4)))
            ground.move(breve.vector(0, -0.1, 0))
            breve.get_engine().register_physics_body(ground, static=True)
            self.ball = breve.Mobile()
            self.ball.set_shape(breve.Sphere().init_with(0.2))
            self.ball.move(breve.vector(0, 2.0, 0))
            self.ball.set_velocity(breve.vector(0, 0, 0))
            self.ball.enable_physics(mass=1.0)

    c = C()
    y0 = c.ball.location.y
    c.run(steps=40)
    assert c.ball.location.y < y0
    # should have hit the ground and settled near y≈0.2
    assert c.ball.location.y < 1.0
    assert c.ball.location.y > 0.05


def test_sphere_bounces_on_box():
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.004)
            self.set_iteration_step(0.016)
            self.full_gravity()
            step = breve.Stationary()
            step.set_shape(breve.Box().init_with(breve.vector(2, 0.1, 2)))
            step.move(breve.vector(0, 0, 0))
            breve.get_engine().register_physics_body(step, static=True)
            self.ball = breve.Mobile()
            self.ball.set_shape(breve.Sphere().init_with(0.15))
            self.ball.move(breve.vector(0, 1.5, 0))
            self.ball.set_velocity(breve.vector(0, 0, 0))
            self.ball.enable_physics(mass=1.0)

    c = C()
    c.run(steps=80)
    # not fallen through the step
    assert c.ball.location.y > 0.05


def test_box_tips_off_platform_edge():
    """Offset support + gravity + torque should drop a box past the rim."""
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.004)
            self.set_iteration_step(0.016)
            self.full_gravity()
            floor = breve.Stationary()
            floor.set_shape(breve.Box().init_with(breve.vector(2.0, 0.2, 2.0)))
            floor.move(breve.vector(0, -0.1, 0))
            breve.get_engine().register_physics_body(floor, static=True)
            # center near the edge; a push + gravity should tip it off
            self.box = breve.Mobile()
            self.box.set_shape(breve.Box().init_with(breve.vector(0.5, 0.5, 0.5)))
            self.box.move(breve.vector(0.85, 0.4, 0))
            self.box.set_velocity(breve.vector(1.5, 0, 0))
            self.box.enable_physics(mass=1.0)

    c = C()
    y0 = c.box.location.y
    c.run(steps=120)
    # fallen well below the platform
    assert c.box.location.y < y0 - 0.3


def test_offset_hit_spins_box():
    """Impulse away from COM should produce angular velocity."""
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.zero_gravity()
            self.set_integration_step(0.005)
            self.set_iteration_step(0.02)
            self.box = breve.Mobile()
            self.box.set_shape(breve.Box().init_with(breve.vector(1.0, 0.4, 0.4)))
            self.box.move(breve.vector(0, 1.0, 0))
            self.box.set_velocity(breve.vector(0, 0, 0))
            self.box.enable_physics(mass=2.0)
            ball = breve.Mobile()
            ball.set_shape(breve.Sphere().init_with(0.2))
            ball.move(breve.vector(-1.2, 1.15, 0))  # hit top half of box
            ball.set_velocity(breve.vector(8, 0, 0))
            ball.enable_physics(mass=1.0)

    c = C()
    body = breve.get_engine().physics.get_body(c.box)
    c.run(steps=15)
    omega = float(np.linalg.norm(body.angular_velocity))
    assert omega > 0.05


def test_gravity_demo_runs():
    set_engine(Engine())
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "demos" / "gravity.py"
    spec = importlib.util.spec_from_file_location("gravity_demo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    set_engine(Engine())
    sim = mod.Gravity()
    y0 = [b.location.y for b in sim.balls]
    sim.run(steps=50)
    y1 = [b.location.y for b in sim.balls]
    # overall, balls should have moved (fallen / rolled)
    assert any(abs(a - b) > 0.01 for a, b in zip(y0, y1))
