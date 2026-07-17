#!/usr/bin/env python3
"""
Gravity — balls bounce down a staircase (classic Getting-Started/Gravity).

Balls continuously re-launch so the 3D view always has motion.
"""

from __future__ import annotations

import argparse
import sys

import breve
from breve.engine import Engine, set_engine


class Gravity(breve.PhysicalControl):
    def __init__(self):
        self.balls = None
        super().__init__()

    def init(self):
        self.set_integration_step(0.005)
        self.set_iteration_step(0.02)
        self.full_gravity()
        self.enable_lighting()
        self.set_background_color(breve.vector(0.12, 0.14, 0.18))

        # Bigger staircase so motion is obvious in 3D
        steps = [
            (breve.vector(-1.0, 1.2, 0.0), breve.vector(2.2, 0.12, 1.6)),
            (breve.vector(0.3, 0.7, 0.0), breve.vector(1.2, 0.12, 1.6)),
            (breve.vector(1.4, 0.2, 0.0), breve.vector(1.2, 0.12, 1.6)),
            (breve.vector(2.5, -0.3, 0.0), breve.vector(1.2, 0.12, 1.6)),
            (breve.vector(3.6, -0.8, 0.0), breve.vector(1.2, 0.12, 1.6)),
            (breve.vector(5.0, -1.3, 0.0), breve.vector(3.0, 0.12, 2.0)),
        ]
        for loc, size in steps:
            Step().create(loc, size)

        ground = breve.Stationary()
        ground.set_shape(breve.Box().init_with(breve.vector(20, 0.15, 12)))
        ground.move(breve.vector(3.0, -1.9, 0.0))
        ground.set_color(breve.vector(0.25, 0.3, 0.28))
        get_engine().register_physics_body(ground, static=True)
        gb = get_engine().physics.get_body(ground)
        if gb:
            gb.restitution = 0.7
            gb.friction = 0.25

        Ball._tier_i = 0
        self.balls = breve.create_instances(Ball, 10)
        # Stagger initial drops
        for i, b in enumerate(self.balls):
            b.cooldown = i * 0.35

        self.point_camera(breve.vector(2.0, 0.2, 0.0), breve.vector(8.0, 4.0, 10.0))
        self.camera_zoom = 12.0
        self._last_print = -1.0
        print(f"Gravity: {len(self.balls)} balls with mixed masses (size/color = weight)")
        print("  light = small yellow/green  ·  heavy = big orange/red")
        print("They re-launch when settled. Heavy ones knock lights around on impact.")

    def iterate(self):
        # Camera follows the action
        if self.balls:
            cx = sum(b.location.x for b in self.balls) / len(self.balls)
            cy = sum(b.location.y for b in self.balls) / len(self.balls)
            self.aim_camera(breve.vector(cx, cy, 0.0))

        if self.engine.time - self._last_print >= 0.5:
            self._last_print = self.engine.time
            speeds = [breve.length(b.velocity) for b in self.balls]
            print(
                f"t={self.engine.time:5.1f}s  "
                f"speed_avg={sum(speeds)/len(speeds):5.2f}  "
                f"y_avg={sum(b.location.y for b in self.balls)/len(self.balls):5.2f}  "
                f"(balls re-drop when slow)"
            )
        super().iterate()


class Step(breve.Stationary):
    def create(self, location, size_vector):
        self.set_shape(breve.Cube().init_with(size_vector))
        self.move(location)
        self.set_color(breve.vector(0.55, 0.5, 0.42))
        get_engine().register_physics_body(self, static=True)
        # high friction steps
        body = get_engine().physics.get_body(self)
        if body:
            body.friction = 0.35
            body.restitution = 0.65
        return self


# mass tiers for staircase balls (size + color encode weight)
_MASS_TIERS = [
    # mass, radius, base color
    (0.3, 0.10, breve.vector(0.95, 0.9, 0.35)),   # light
    (0.8, 0.14, breve.vector(0.35, 0.9, 0.45)),
    (1.5, 0.18, breve.vector(0.35, 0.55, 0.95)),
    (4.0, 0.24, breve.vector(0.9, 0.4, 0.2)),
    (9.0, 0.30, breve.vector(0.75, 0.15, 0.2)),   # heavy
]


class Ball(breve.Mobile):
    _tier_i = 0

    def init(self):
        self.cooldown = 0.0
        tier = _MASS_TIERS[Ball._tier_i % len(_MASS_TIERS)]
        Ball._tier_i += 1
        self.mass, self._radius, self._base_color = tier
        self.set_shape(breve.Sphere().init_with(self._radius))
        self.enable_physics(mass=self.mass)
        body = get_engine().physics.get_body(self)
        if body:
            body.restitution = 0.82 if self.mass < 2 else 0.7
            body.friction = 0.15 if self.mass < 2 else 0.25
        self.reset()

    def iterate(self):
        eng = get_engine()
        self.cooldown = max(0.0, self.cooldown - eng.iteration_step)
        speed = breve.length(self.velocity)
        settled = speed < 0.4
        if self.cooldown <= 0 and (
            self.location.y < -2.5
            or self.location.x > 9
            or self.location.x < -4
            or settled
        ):
            self.reset()
            self.cooldown = 0.5 + breve.random_expression(1.2)

    def reset(self):
        # Keep mass/size; jitter color slightly around the tier color
        self.set_color(
            self._base_color + breve.random_expression(breve.vector(0.08, 0.08, 0.08))
        )
        self.move(
            breve.vector(-1.6, 2.8, -0.5)
            + breve.random_expression(breve.vector(1.2, 0.6, 1.0))
        )
        # Same-ish launch speed — mass difference shows when they collide
        self.set_velocity(
            breve.vector(
                1.5 + breve.random_expression(1.5),
                0.2 + breve.random_expression(0.8),
                (breve.random_expression(1.0) - 0.5) * 1.2,
            )
        )
        get_engine().register_physics_body(self, static=False, mass=self.mass)
        body = get_engine().physics.get_body(self)
        if body is not None:
            body.position[:] = [self.location.x, self.location.y, self.location.z]
            body.velocity[:] = [self.velocity.x, self.velocity.y, self.velocity.z]
            body.mass = self.mass
            body.inv_mass = 1.0 / max(self.mass, 1e-6)
            body.awake = True
            body.restitution = 0.82 if self.mass < 2 else 0.7
            body.friction = 0.15 if self.mass < 2 else 0.25


def get_engine():
    return breve.get_engine()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Breve gravity / staircase demo")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)

    steps = args.steps if args.steps is not None else (None if args.viz else 400)

    set_engine(Engine())
    sim = Gravity()

    if args.viz:
        try:
            from breve.viz import run_with_viewer

            print("3D: drag=orbit  scroll=zoom  SPACE=pause  A=auto-orbit  ESC=quit")
            run_with_viewer(sim, steps=steps)
            return
        except ImportError:
            print("viz deps missing; pip install -e '.[viz]'", file=sys.stderr)

    sim.run(steps=steps if steps is not None else 400)
    print("done.")


if __name__ == "__main__":
    main()
