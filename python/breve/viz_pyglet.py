"""Simple interactive 2D projection viewer (optional: pip install pyglet)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from breve.objects import Control


def run_with_viewer(control: "Control", steps: Optional[int] = None) -> None:
    """Top-down orthographic projection of agent positions."""
    try:
        import pyglet
        from pyglet import shapes
    except ImportError as e:
        raise ImportError("pyglet is required for --viz (pip install -e '.[viz]')") from e

    window = pyglet.window.Window(
        width=960, height=720, caption=f"breve {__import__('breve').__version__}", resizable=True
    )
    batch = pyglet.graphics.Batch()
    step_count = 0
    dots: list = []
    label = pyglet.text.Label(
        "",
        font_size=13,
        x=12,
        y=12,
        anchor_x="left",
        anchor_y="bottom",
        color=(220, 220, 230, 255),
    )

    def world_to_screen(x, z, zoom, cx, cz):
        # center world (cx,cz) on window; scale by zoom
        scale = min(window.width, window.height) / max(zoom * 2.0, 1.0)
        sx = window.width / 2 + (x - cx) * scale
        sy = window.height / 2 + (z - cz) * scale
        return sx, sy

    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        label.draw()

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            control.engine.stop()
            pyglet.app.exit()
        elif symbol == pyglet.window.key.SPACE:
            control.engine.running = not control.engine.running
        elif symbol == pyglet.window.key.N and hasattr(control, "flock_normally"):
            control.flock_normally()
        elif symbol == pyglet.window.key.O and hasattr(control, "flock_obediently"):
            control.flock_obediently()
        elif symbol == pyglet.window.key.W and hasattr(control, "flock_wackily"):
            control.flock_wackily()
        elif symbol == pyglet.window.key.S and hasattr(control, "squish"):
            control.squish()

    def tick(_dt):
        nonlocal step_count, dots
        if steps is not None and step_count >= steps:
            control.engine.stop()
            pyglet.app.exit()
            return
        if control.engine.running or step_count == 0:
            if not control.engine.running:
                control.engine.running = True
            control.engine.step()
            step_count += 1

        # rebuild dots each frame (simple; fine for hundreds of agents)
        dots.clear()
        # pyglet Batch doesn't easily clear; recreate batch contents via new shapes list
        # Use a fresh batch by drawing manually via shapes on a list we own
        tick.shapes = []  # type: ignore[attr-defined]
        cx, cy, cz = control.camera_target.x, control.camera_target.y, control.camera_target.z
        zoom = max(control.camera_zoom, 5.0)
        for obj in control.engine.objects:
            if not obj.enabled or obj.__class__.__name__ == "Floor":
                continue
            sx, sy = world_to_screen(obj.location.x, obj.location.z, zoom, cx, cz)
            r = 4
            if obj.shape is not None:
                r = max(3, min(12, obj.shape.bounding_radius() * 8))
            color = (
                int(max(0, min(255, obj.color.x * 255))),
                int(max(0, min(255, obj.color.y * 255))),
                int(max(0, min(255, obj.color.z * 255))),
            )
            tick.shapes.append(shapes.Circle(sx, sy, r, color=color, batch=batch))  # type: ignore[attr-defined]

        # force new batch by reassigning — shapes hold batch ref
        # Actually old shapes linger on batch; recreate batch each frame:
        on_draw.batch_shapes = tick.shapes  # type: ignore[attr-defined]

        mode = getattr(control, "mode", "")
        label.text = (
            f"t={control.engine.time:.1f}s  agents={len(control.engine.objects)}  "
            f"zoom={zoom:.1f}  {mode}  [SPACE pause | N/O/W flock | S squish | ESC quit]"
        )

    # Fix batch accumulation: rebuild batch each draw
    frame_shapes: list = []

    @window.event
    def on_draw():  # noqa: F811
        window.clear()
        for s in frame_shapes:
            s.draw()
        label.draw()

    def tick2(_dt):
        nonlocal step_count, frame_shapes
        if steps is not None and step_count >= steps:
            control.engine.stop()
            pyglet.app.exit()
            return
        if step_count == 0:
            control.engine.running = True
        if control.engine.running:
            control.engine.step()
            step_count += 1

        frame_shapes = []
        cx, cz = control.camera_target.x, control.camera_target.z
        zoom = max(control.camera_zoom, 5.0)
        for obj in control.engine.objects:
            if not obj.enabled or obj.__class__.__name__ == "Floor":
                continue
            sx, sy = world_to_screen(obj.location.x, obj.location.z, zoom, cx, cz)
            r = 4
            if obj.shape is not None:
                r = max(3, min(14, obj.shape.bounding_radius() * 10))
            color = (
                int(max(0, min(255, obj.color.x * 255))),
                int(max(0, min(255, obj.color.y * 255))),
                int(max(0, min(255, obj.color.z * 255))),
            )
            frame_shapes.append(shapes.Circle(sx, sy, r, color=color))

        mode = getattr(control, "mode", "")
        label.text = (
            f"t={control.engine.time:.1f}s  n={sum(1 for o in control.engine.objects if o.__class__.__name__!='Floor')}  "
            f"zoom={zoom:.1f}  {mode}  [SPACE|N/O/W|S|ESC]"
        )

    @window.event
    def on_key_press(symbol, modifiers):  # noqa: F811
        if symbol == pyglet.window.key.ESCAPE:
            control.engine.stop()
            pyglet.app.exit()
        elif symbol == pyglet.window.key.SPACE:
            control.engine.running = not control.engine.running
        elif symbol == pyglet.window.key.N and hasattr(control, "flock_normally"):
            control.flock_normally()
        elif symbol == pyglet.window.key.O and hasattr(control, "flock_obediently"):
            control.flock_obediently()
        elif symbol == pyglet.window.key.W and hasattr(control, "flock_wackily"):
            control.flock_wackily()
        elif symbol == pyglet.window.key.S and hasattr(control, "squish"):
            control.squish()

    control.engine.running = True
    pyglet.clock.schedule_interval(tick2, max(control.engine.iteration_step, 1 / 60))
    pyglet.app.run()
