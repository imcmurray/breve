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
    """COM past the rim: gravity + contact torque at the edge must tip it off."""
    set_engine(Engine())

    class C(breve.PhysicalControl):
        def init(self):
            self.set_integration_step(0.004)
            self.set_iteration_step(0.016)
            self.full_gravity()
            # platform occupies x in [-1, 1]
            floor = breve.Stationary()
            floor.set_shape(breve.Box().init_with(breve.vector(2.0, 0.2, 2.0)))
            floor.move(breve.vector(0, -0.1, 0))
            breve.get_engine().register_physics_body(floor, static=True)
            # 0.5-wide box: COM at 1.05 is past the rim (x=1); only the
            # inner face still overlaps. No initial push — weight alone tips it.
            self.box = breve.Mobile()
            self.box.set_shape(breve.Box().init_with(breve.vector(0.5, 0.5, 0.5)))
            self.box.move(breve.vector(1.05, 0.35, 0))
            self.box.set_velocity(breve.vector(0, 0, 0))
            self.box.enable_physics(mass=1.0)

    c = C()
    c.run(steps=100)
    assert c.box.location.y < -0.5


def test_box_stack_tips_off_edge():
    """Stacked boxes with COMs past the platform edge must all fall."""
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
            self.boxes = []
            for i in range(3):
                b = breve.Mobile()
                b.set_shape(breve.Box().init_with(breve.vector(0.5, 0.5, 0.5)))
                b.move(breve.vector(1.05, 0.35 + i * 0.52, 0))
                b.enable_physics(mass=0.5)
                self.boxes.append(b)

    c = C()
    c.run(steps=220)
    assert all(b.location.y < -0.5 for b in c.boxes)


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
