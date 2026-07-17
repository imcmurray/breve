#!/usr/bin/env python3
"""
Stack — wrecking ball repeatedly smashes a tower of boxes.

The ball re-launches every few seconds so the view always has action.
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
        self._last_print = -1.0
        self._rebuild_timer = 0.0
        super().__init__()

    def init(self):
        self.set_integration_step(0.005)
        self.set_iteration_step(0.02)
        self.full_gravity()
        self.enable_lighting()
        self.set_background_color(breve.vector(0.1, 0.12, 0.16))

        ground = breve.Stationary()
        ground.set_shape(breve.Box().init_with(breve.vector(16, 0.25, 10)))
        ground.move(breve.vector(0, -0.125, 0))
        ground.set_color(breve.vector(0.28, 0.32, 0.3))
        get_engine().register_physics_body(ground, static=True)
        gb = get_engine().physics.get_body(ground)
        if gb:
            gb.friction = 0.9

        self.blocks = breve.object_list()
        self._build_tower()

        self.ball = WreckingBall()
        self.point_camera(breve.vector(0, 1.0, 0), breve.vector(7, 4, 9))
        self.camera_zoom = 11.0
        print(f"Stack: {len(self.blocks)} blocks — wrecking ball re-launches forever")
        print("Watch the RED ball. It should fly in from the left every ~3 seconds.")

    def _build_tower(self):
        # Clear old blocks
        for b in list(self.blocks):
            b.remove()
        self.blocks = breve.object_list()

        cols, rows = 4, 5
        size = 0.38
        gap = 0.05
        for j in range(rows):
            for i in range(cols):
                b = Block()
                x = (i - (cols - 1) / 2) * (size + gap)
                y = size * 0.5 + j * (size + gap) + 0.02
                b.setup(breve.vector(x, y, 0), size)
                self.blocks.append(b)

    def iterate(self):
        eng = get_engine()
        self._rebuild_timer += eng.iteration_step

        # Follow ball + tower
        if self.ball:
            self.aim_camera(
                breve.vector(
                    self.ball.location.x * 0.35,
                    max(0.8, self.ball.location.y),
                    0.0,
                )
            )

        # Rebuild tower periodically so there's always a target
        upright = sum(1 for b in self.blocks if b.location.y > 0.25)
        if self._rebuild_timer > 8.0 or upright < 4:
            self._build_tower()
            self._rebuild_timer = 0.0
            print("  → tower rebuilt")

        if self.engine.time - self._last_print >= 0.4:
            self._last_print = self.engine.time
            print(
                f"t={self.engine.time:5.1f}s  blocks_up≈{upright}/{len(self.blocks)}  "
                f"ball=({self.ball.location.x:5.2f},{self.ball.location.y:5.2f})  "
                f"|v|={breve.length(self.ball.velocity):5.2f}"
            )
        super().iterate()


class Block(breve.Mobile):
    def setup(self, loc, size):
        self.set_shape(breve.Box().init_with(breve.vector(size, size, size)))
        self.move(loc)
        self.set_color(
            breve.vector(0.65, 0.48, 0.28)
            + breve.random_expression(breve.vector(0.25, 0.15, 0.1))
        )
        self.mass = 0.5
        self.set_velocity(breve.vector(0, 0, 0))
        self.enable_physics(mass=self.mass)
        body = get_engine().physics.get_body(self)
        if body:
            body.restitution = 0.1
            body.friction = 0.55
            body.velocity[:] = 0.0
        return self


class WreckingBall(breve.Mobile):
    def init(self):
        self.set_shape(breve.Sphere().init_with(0.45))
        self.set_color(breve.vector(0.95, 0.15, 0.12))
        self.mass = 18.0
        self._timer = 0.0
        self._launch()

    def _launch(self):
        self.move(breve.vector(-4.5, 1.35, 0.0))
        self.set_velocity(breve.vector(16.0, 0.8, 0.0))
        self.enable_physics(mass=self.mass)
        body = get_engine().physics.get_body(self)
        if body is not None:
            body.restitution = 0.2
            body.friction = 0.05
            body.position[:] = [-4.5, 1.35, 0.0]
            body.velocity[:] = [16.0, 0.8, 0.0]
            body.awake = True
        self._timer = 0.0

    def iterate(self):
        eng = get_engine()
        self._timer += eng.iteration_step
        speed = breve.length(self.velocity)
        # Re-launch on a steady cadence OR after stopping / flying away
        if self._timer > 3.0 or self.location.x > 6 or (
            self._timer > 1.2 and speed < 0.4
        ):
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

    steps = args.steps if args.steps is not None else (None if args.viz else 300)

    set_engine(Engine())
    sim = Stack()

    if args.viz:
        try:
            from breve.viz import run_with_viewer

            print("3D: drag=orbit  scroll=zoom  SPACE=pause  A=auto-orbit  ESC=quit")
            print("Watch the RED ball fly in from the left — it loops forever.")
            run_with_viewer(sim, steps=steps)
            return
        except ImportError:
            print("viz deps missing; pip install -e '.[viz]'", file=sys.stderr)

    sim.run(steps=steps if steps is not None else 300)
    print("done.")


if __name__ == "__main__":
    main()
