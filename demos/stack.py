#!/usr/bin/env python3
"""
Stack / demolition-lite — dynamic boxes collapse under gravity + a wrecking ball.

Shows sphere–box and box–box rigid contacts.
"""

from __future__ import annotations

import argparse
import sys

import breve
from breve.engine import Engine, set_engine


class Stack(breve.PhysicalControl):
    def __init__(self):
        self.blocks = None
        self.ball = None
        super().__init__()

    def init(self):
        self.set_integration_step(0.004)
        self.set_iteration_step(0.016)
        self.full_gravity()
        self.enable_lighting()
        self.set_background_color(breve.vector(0.12, 0.14, 0.18))

        # ground
        ground = breve.Stationary()
        ground.set_shape(breve.Box().init_with(breve.vector(14, 0.2, 14)))
        ground.move(breve.vector(0, -0.1, 0))
        ground.set_color(breve.vector(0.28, 0.32, 0.3))
        get_engine().register_physics_body(ground, static=True)

        # wall of boxes (slight gaps so they aren't born interpenetrating)
        self.blocks = breve.object_list()
        cols, rows = 5, 4
        size = 0.32
        gap = 0.04
        for j in range(rows):
            for i in range(cols):
                b = Block()
                x = (i - (cols - 1) / 2) * (size + gap)
                y = size * 0.5 + j * (size + gap)
                b.setup(breve.vector(x, y, 0), size)
                self.blocks.append(b)

        # wrecking ball
        self.ball = WreckingBall()
        self.point_camera(breve.vector(0, 1.2, 0), breve.vector(6, 3, 8))
        self.camera_zoom = 10.0
        print(f"Stack: {len(self.blocks)} blocks + wrecking ball")

    def iterate(self):
        # heartbeat ~0.5s
        if int(self.engine.time * 20) % 10 == 0:
            upright = sum(1 for b in self.blocks if b.location.y > 0.2)
            print(
                f"t={self.engine.time:5.2f}s  blocks_up≈{upright}/{len(self.blocks)}  "
                f"ball=({self.ball.location.x:5.2f},{self.ball.location.y:5.2f})  "
                f"|v|={breve.length(self.ball.velocity):5.2f}"
            )
        super().iterate()


class Block(breve.Mobile):
    def setup(self, loc, size):
        self.set_shape(breve.Box().init_with(breve.vector(size, size, size)))
        self.move(loc)
        self.set_color(
            breve.vector(0.6, 0.45, 0.3) + breve.random_expression(breve.vector(0.3, 0.2, 0.15))
        )
        self.mass = 0.8
        self.enable_physics(mass=self.mass)
        # zero initial velocity
        self.set_velocity(breve.vector(0, 0, 0))
        get_engine().register_physics_body(self, static=False, mass=self.mass)
        return self


class WreckingBall(breve.Mobile):
    def init(self):
        self.set_shape(breve.Sphere().init_with(0.4))
        self.set_color(breve.vector(0.85, 0.15, 0.15))
        self.mass = 12.0
        self._launch()

    def _launch(self):
        # Aim at mid-stack: wall centered near x=0, y≈0.7
        self.move(breve.vector(-3.2, 0.9, 0.0))
        self.set_velocity(breve.vector(14.0, 1.2, 0.0))
        self.enable_physics(mass=self.mass)
        body = get_engine().physics.get_body(self)
        if body is not None:
            body.restitution = 0.15
            body.friction = 0.05
            body.velocity[:] = [14.0, 1.2, 0.0]

    def iterate(self):
        if self.location.y < -1.5 or self.location.x > 8 or self.location.x < -8:
            self._launch()


def get_engine():
    return breve.get_engine()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Breve box stack / wrecking ball")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.seed is not None:
        import numpy as np

        np.random.seed(args.seed)

    steps = args.steps if args.steps is not None else (None if args.viz else 250)

    set_engine(Engine())
    sim = Stack()

    if args.viz:
        try:
            from breve.viz import run_with_viewer

            print("3D viewer: drag orbit, scroll zoom, SPACE pause, ESC quit")
            run_with_viewer(sim, steps=steps)
            return
        except ImportError:
            print("viz deps missing; pip install -e '.[viz]'", file=sys.stderr)

    sim.run(steps=steps if steps is not None else 250)
    print("done.")


if __name__ == "__main__":
    main()
