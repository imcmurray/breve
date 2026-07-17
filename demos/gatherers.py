#!/usr/bin/env python3
"""
Gatherers — collectors pick up food on collision (modern breve).

Simplified port of classic demos/Gatherers.py.
"""

from __future__ import annotations

import breve
from breve.engine import Engine, set_engine


class Gatherers(breve.Control):
    def __init__(self):
        self.collector_shape = None
        self.food_shape = None
        self.food_list = None
        super().__init__()

    def init(self):
        self.collector_shape = breve.Sphere().init_with(1.0)
        self.food_shape = breve.Sphere().init_with(0.5)
        breve.create_instances(Collector, 20)
        self.food_list = breve.create_instances(Food, 60)
        self.set_integration_step(0.05)
        self.set_iteration_step(0.1)
        self.point_camera(breve.vector(0, 0, 0), breve.vector(0, 40, 80))
        self.enable_lighting()
        print("Gatherers: 20 collectors, 60 food")

    def get_collector_shape(self):
        return self.collector_shape

    def get_food_shape(self):
        return self.food_shape

    def iterate(self):
        carried = sum(1 for o in self.engine.objects if isinstance(o, Collector) and o.carrying)
        free_food = sum(1 for o in self.engine.objects if isinstance(o, Food) and o.owner is None)
        if int(self.engine.time * 5) % 5 == 0:
            print(f"t={self.engine.time:5.1f}s  carrying={carried}  free_food={free_food}")
        super().iterate()


class Food(breve.Mobile):
    def __init__(self):
        self.owner = None
        super().__init__()

    def init(self):
        ctrl = self._controller()
        self.set_shape(ctrl.get_food_shape())
        self.set_color(breve.vector(0.2, 0.9, 0.2))
        self.set_wander_range(breve.vector(20, 0.5, 20))
        self.randomize_location()
        # food sits still unless carried
        self.frozen = True

    def _controller(self) -> Gatherers:
        return get_engine_control()

    def get_owner(self):
        return self.owner

    def set_owner(self, owner):
        self.owner = owner


class Collector(breve.Mobile):
    def __init__(self):
        self.carrying = None
        self.just_collided = 0
        super().__init__()

    def init(self):
        ctrl = get_engine_control()
        self.set_shape(ctrl.get_collector_shape())
        self.set_color(breve.vector(1, 1, 1))
        self.set_wander_range(breve.vector(20, 0.5, 20))
        self.randomize_location()
        self.handle_collisions("Food", "collide")
        # wander with gentle random velocity changes
        self.set_velocity(breve.random_expression(breve.vector(4, 0, 4)) - breve.vector(2, 0, 2))

    def collide(self, food: Food):
        if food.owner is not None:
            return
        if self.just_collided > 0:
            self.just_collided = 2
            return
        self.just_collided = 2
        if self.carrying is not None:
            # drop previous near this food
            new_loc = food.get_location() + (
                breve.random_expression(breve.vector(2, 0, 2)) - breve.vector(1, 0, 1)
            )
            self.carrying.move(new_loc)
            self.carrying.set_owner(None)
            self.carrying.frozen = True
            self.carrying = None
            return
        food.set_owner(self)
        food.frozen = True
        self.carrying = food

    def iterate(self):
        # bounce at wander bounds
        r = self.wander_range
        loc = self.location
        vel = self.velocity
        if abs(loc.x) > r.x:
            vel.x *= -1
        if abs(loc.z) > r.z:
            vel.z *= -1
        self.set_velocity(vel)
        if self.carrying is not None:
            self.carrying.move(self.location - breve.vector(1.2, 0, 0))
        self.just_collided = max(0, self.just_collided - 1)
        # slight random steering
        import numpy as np

        if np.random.random() < 0.05:
            self.set_velocity(
                breve.random_expression(breve.vector(4, 0, 4)) - breve.vector(2, 0, 2)
            )


def get_engine_control() -> Gatherers:
    eng = breve.get_engine()
    assert eng.control is not None
    return eng.control  # type: ignore[return-value]


def main(argv=None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Breve gatherers demo")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args(argv)

    set_engine(Engine())
    sim = Gatherers()
    if args.viz:
        try:
            from breve.viz import run_with_viewer

            run_with_viewer(sim, steps=args.steps)
            return
        except ImportError:
            print("viz deps missing; pip install -e '.[viz]'", file=sys.stderr)
    sim.run(steps=args.steps)
    print("done.")


if __name__ == "__main__":
    main()
