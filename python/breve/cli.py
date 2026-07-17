"""Minimal CLI entry point."""

from __future__ import annotations

import argparse
import runpy
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="breve-run", description="Run a breve Python simulation file")
    parser.add_argument("script", help="Path to a .py simulation")
    parser.add_argument("--steps", type=int, default=None, help="Override: not used by all scripts")
    args = parser.parse_args(argv)
    # Scripts typically construct a Control and call run(); execute as __main__
    sys.path.insert(0, ".")
    runpy.run_path(args.script, run_name="__main__")


if __name__ == "__main__":
    main()
