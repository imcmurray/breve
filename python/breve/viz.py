"""Visualization entry points — prefer true 3D, fall back to 2D projection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from breve.objects import Control


def run_with_viewer(control: "Control", steps: Optional[int] = None) -> None:
    """
    Open a viewer for `control`.

    Tries the moderngl 3D orbit viewer first; falls back to the simple pyglet 2D
    top-down view if 3D deps/context fail.
    """
    try:
        from breve.viz3d import run_with_viewer as run_3d

        run_3d(control, steps=steps)
        return
    except Exception as exc:  # noqa: BLE001 — intentional fallback path
        print(f"[breve] 3D viewer unavailable ({exc}); falling back to 2D.")
        from breve.viz_pyglet import run_with_viewer as run_2d

        run_2d(control, steps=steps)
