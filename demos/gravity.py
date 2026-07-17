#!/usr/bin/env python3
"""
Gravity — balls bounce down a staircase (classic Getting-Started/Gravity).

Uses the pure-Python rigid-body solver (PhysicalControl + enable_physics).
"""

from __future__ import annotations

import argparse
import sys

import breve
from breve.engine import Engine, set_engine


class Gravity(breve.PhysicalControl):
    def __init__(self):
        self.balls = None
        self.the_ball = None
        super().__init__()

    def init(self):
        # tighter steps for stable bouncing on small geometry
        self.set_integration_step(0.004)
        self.set_iteration_step(0.016)
        self.full_gravity()
        self.enable_lighting()
        self.set_background_color(breve.vector(0.15, 0.18, 0.22))

        # staircase (static boxes)
        steps = [
            (breve.vector(-0.5, 0.0, 0.0), breve.vector(1.0, 0.06, 1.0)),
            (breve.vector(0.2, -0.2, 0.0), breve.vector(0.35, 0.06, 1.0)),
            (breve.vector(0.5, -0.4, 0.0), breve.vector(0.35, 0.06, 1.0)),
            (breve.vector(0.8, -0.6, 0.0), breve.vector(0.35, 0.06, 1.0)),
            (breve.vector(1.1, -0.8, 0.0), breve.vector(0.35, 0.06, 1.0)),
            (breve.vector(1.4, -1.0, 0.0), breve.vector(0.35, 0.06, 1.0)),
            (breve.vector(2.2, -1.2, 0.0), breve.vector(2.0, 0.06, 1.2)),
        ]
        for loc, size in steps:
            Step().create(loc, size)

        # floor catcher
        ground = breve.Stationary()
        ground.set_shape(breve.Box().init_with(breve.vector(12, 0.08, 12)))
        ground.move(breve.vector(1.0, -1.8, 0.0))
        ground.set_color(breve.vector(0.3, 0.35, 0.32))
        get_engine().register_physics_body(ground, static=True)

        self.balls = breve.create_instances(Ball, 8)
        self.the_ball = self.balls[0] if self.balls else None
        self.point_camera(breve.vector(1.0, -0.6, 0.0), breve.vector(4.0, 1.5, 5.0))
        self.camera_zoom = 6.0
        print(f"Gravity: {len(self.balls)} balls on staircase (rigid body physics)")

    def reset_ball(self):
        if self.the_ball:
            self.the_ball.reset()

    def iterate(self):
        if int(self.engine.time * 10) % 20 == 0:
            ys = [b.location.y for b in self.balls]
            speeds = [breve.length(b.velocity) for b in self.balls]
            print(
                f"t={self.engine.time:5.2f}s  "
                f"y_avg={sum(ys)/len(ys):6.2f}  "
                f"speed_avg={sum(speeds)/len(speeds):5.2f}  "
                f"y_min={min(ys):6.2f}"
            )
        super().iterate()


class Step(breve.Stationary):
    def create(self, location, size_vector):
        self.set_shape(breve.Cube().init_with(size_vector))
        self.move(location)
        self.set_color(breve.vector(0.55, 0.5, 0.45))
        get_engine().register_physics_body(self, static=True)
        return self


class Ball(breve.Mobile):
    def init(self):
        r = 0.06 + breve.random_expression(0.10)
        self.set_shape(breve.Sphere().init_with(r))
        self.mass = max(0.2, r * 8)
        self.enable_physics(mass=self.mass)
        self.reset()

    def iterate(self):
        if self.location.y < -3.5:
            self.reset()

    def reset(self):
        self.set_color(
            breve.vector(0.4, 0.5, 0.7) + breve.random_expression(breve.vector(0.6, 0.5, 0.3))
        )
        self.move(
            breve.vector(-0.7, 0.7, -0.4)
            + breve.random_expression(breve.vector(0.9, 0.5, 0.8))
        )
        self.set_velocity(
            breve.vector(
                1.2 + breve.random_expression(0.8),
                0.4 + breve.random_expression(0.6),
                (breve.random_expression(1.0) - 0.5) * 0.6,
            )
        )
        # wake physics body with new state
        if self.physics_enabled:
            get_engine().register_physics_body(self, static=False, mass=self.mass)


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

    steps = args.steps
    if steps is None:
        steps = None if args.viz else 300

    set_engine(Engine())
    sim = Gravity()

    if args.viz:
        try:
            from breve.viz import run_with_viewer

            print("3D viewer: drag orbit, scroll zoom, SPACE pause, ESC quit")
            run_with_viewer(sim, steps=steps)
            return
        except ImportError:
            print("viz deps missing; pip install -e '.[viz]'", file=sys.stderr)

    sim.run(steps=steps if steps is not None else 300)
    print("done.")


if __name__ == "__main__":
    main()
