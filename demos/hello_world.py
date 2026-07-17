#!/usr/bin/env python3
"""Classic-style HelloWorld controller (modern breve)."""

from __future__ import annotations

import breve
from breve.engine import set_engine, Engine


class HelloWorld(breve.Control):
    def iterate(self):
        print(f"[{self.engine.time:6.2f}s] Hello, world!")
        super().iterate()


def main() -> None:
    set_engine(Engine())
    sim = HelloWorld()
    sim.run(steps=5)
    print("done.")


if __name__ == "__main__":
    main()
