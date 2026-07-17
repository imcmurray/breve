"""
Interactive 3D viewer for breve (moderngl-window).

Orbit camera, velocity-aligned agents, ground grid.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from breve.objects import Control

_VS = """
#version 330
in vec3 in_position;
in vec3 in_color;
uniform mat4 m_mvp;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = m_mvp * vec4(in_position, 1.0);
}
"""

_FS = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""

_POINT_VS = """
#version 330
in vec3 in_position;
in vec3 in_color;
uniform mat4 m_mvp;
uniform float u_point_size;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = m_mvp * vec4(in_position, 1.0);
    gl_PointSize = u_point_size;
}
"""

_POINT_FS = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    float d = length(c);
    if (d > 0.5) discard;
    float edge = smoothstep(0.5, 0.32, d);
    f_color = vec4(mix(vec3(0.1), v_color, edge), edge);
}
"""


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = -eye
    return m @ t


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _agent_mesh(
    pos: np.ndarray,
    vel: np.ndarray,
    color: np.ndarray,
    size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    speed = float(np.linalg.norm(vel))
    if speed < 1e-4:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = (vel / speed).astype(np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up))) > 0.9:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-9)

    nose = pos + forward * size
    left = pos - forward * size * 0.45 - right * size * 0.55 + up * size * 0.05
    right_v = pos - forward * size * 0.45 + right * size * 0.55 + up * size * 0.05
    verts = np.concatenate([nose, left, right_v]).astype(np.float32)
    cols = np.tile(color.astype(np.float32), 3)
    return verts, cols


def _grid_lines(half: float = 24.0, step: float = 2.0, y: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    pts: List[float] = []
    cols: List[float] = []
    c_major = (0.22, 0.30, 0.40)
    c_axis = (0.40, 0.55, 0.32)
    n = int(half / step)
    for i in range(-n, n + 1):
        t = i * step
        col = c_axis if i == 0 else c_major
        pts.extend([-half, y, t, half, y, t])
        cols.extend(list(col) * 2)
        pts.extend([t, y, -half, t, y, half])
        cols.extend(list(col) * 2)
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32)


def run_with_viewer(control: "Control", steps: Optional[int] = None) -> None:
    """Open an interactive 3D window and step the simulation."""
    try:
        import moderngl
        import moderngl_window as mglw
        from moderngl_window import geometry  # noqa: F401 — ensure package layout
    except ImportError as e:
        raise ImportError(
            "3D viewer needs moderngl-window. Install with: pip install -e '.[viz]'"
        ) from e

    # Capture in closure for WindowConfig
    state = {
        "control": control,
        "steps": steps,
        "step_count": 0,
        "paused": False,
        "azim": 0.75,
        "elev": 0.42,
        "dist": 30.0,
        "dragging": False,
        "last_mouse": (0, 0),
        "auto_orbit": True,
    }

    class Breve3D(mglw.WindowConfig):
        gl_version = (3, 3)
        title = f"breve {__import__('breve').__version__} — 3D"
        window_size = (1100, 720)
        aspect_ratio = None
        resizable = True
        samples = 4

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
            self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

            self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
            self.prog_pts = self.ctx.program(vertex_shader=_POINT_VS, fragment_shader=_POINT_FS)

            gv, gc = _grid_lines()
            self.n_grid = len(gv) // 3
            self.grid_vao = self.ctx.vertex_array(
                self.prog,
                [
                    (self.ctx.buffer(gv.tobytes()), "3f", "in_position"),
                    (self.ctx.buffer(gc.tobytes()), "3f", "in_color"),
                ],
            )

            self.agent_vbo = self.ctx.buffer(reserve=512 * 1024)
            self.agent_cbo = self.ctx.buffer(reserve=512 * 1024)
            self.agent_vao = self.ctx.vertex_array(
                self.prog,
                [
                    (self.agent_vbo, "3f", "in_position"),
                    (self.agent_cbo, "3f", "in_color"),
                ],
            )
            self.n_agent = 0

            self.point_vbo = self.ctx.buffer(reserve=128 * 1024)
            self.point_cbo = self.ctx.buffer(reserve=128 * 1024)
            self.point_vao = self.ctx.vertex_array(
                self.prog_pts,
                [
                    (self.point_vbo, "3f", "in_position"),
                    (self.point_cbo, "3f", "in_color"),
                ],
            )
            self.n_points = 0

            control.engine.running = True
            print(
                "Controls: drag=orbit  scroll=zoom  SPACE=pause  A=auto-orbit  "
                "N/O/W=flock  S=squish  ESC=quit"
            )

        def _target(self) -> np.ndarray:
            t = control.camera_target
            return np.array([t.x, t.y, t.z], dtype=np.float32)

        def _mvp(self) -> bytes:
            target = self._target()
            if state["auto_orbit"]:
                z = max(float(getattr(control, "camera_zoom", 20.0)), 4.0)
                state["dist"] = 0.88 * state["dist"] + 0.12 * (z * 1.15 + 6.0)

            el = float(np.clip(state["elev"], -1.15, 1.35))
            state["elev"] = el
            az, d = state["azim"], state["dist"]
            eye = np.array(
                [
                    target[0] + d * math.cos(el) * math.sin(az),
                    target[1] + d * math.sin(el),
                    target[2] + d * math.cos(el) * math.cos(az),
                ],
                dtype=np.float32,
            )
            view = _look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
            aspect = self.wnd.buffer_width / max(self.wnd.buffer_height, 1)
            proj = _perspective(50.0, aspect, 0.1, 500.0)
            mvp = proj @ view
            return np.ascontiguousarray(mvp.T).astype(np.float32).tobytes()

        def _rebuild(self) -> None:
            tri_v: List[float] = []
            tri_c: List[float] = []
            pt_v: List[float] = []
            pt_c: List[float] = []

            for obj in control.engine.objects:
                if not obj.enabled or obj.__class__.__name__ == "Floor":
                    continue
                pos = np.array(
                    [obj.location.x, obj.location.y, obj.location.z], dtype=np.float32
                )
                col = np.array(
                    [
                        float(np.clip(obj.color.x, 0, 1)),
                        float(np.clip(obj.color.y, 0, 1)),
                        float(np.clip(obj.color.z, 0, 1)),
                    ],
                    dtype=np.float32,
                )
                vel = None
                if hasattr(obj, "velocity"):
                    vel = np.array(
                        [obj.velocity.x, obj.velocity.y, obj.velocity.z],
                        dtype=np.float32,
                    )
                radius = 0.35
                if obj.shape is not None:
                    radius = max(0.2, min(1.2, obj.shape.bounding_radius()))

                if vel is not None and float(np.linalg.norm(vel)) > 0.05:
                    v, c = _agent_mesh(pos, vel, col, radius * 1.9)
                    tri_v.extend(v.tolist())
                    tri_c.extend(c.tolist())
                else:
                    pt_v.extend(pos.tolist())
                    pt_c.extend((col * 0.65 + 0.35).tolist())

            if tri_v:
                vb = np.array(tri_v, dtype=np.float32)
                cb = np.array(tri_c, dtype=np.float32)
                self.agent_vbo.write(vb.tobytes())
                self.agent_cbo.write(cb.tobytes())
                self.n_agent = len(vb) // 3
            else:
                self.n_agent = 0

            if pt_v:
                vb = np.array(pt_v, dtype=np.float32)
                cb = np.array(pt_c, dtype=np.float32)
                self.point_vbo.write(vb.tobytes())
                self.point_cbo.write(cb.tobytes())
                self.n_points = len(vb) // 3
            else:
                self.n_points = 0

        def on_render(self, time: float, frame_time: float) -> None:
            steps = state["steps"]
            if steps is not None and state["step_count"] >= steps:
                self.wnd.close()
                return

            if control.engine.running and not state["paused"]:
                control.engine.step()
                state["step_count"] += 1
                if state["auto_orbit"]:
                    state["azim"] += 0.35 * frame_time

            self._rebuild()

            bg = control.background_color
            self.ctx.clear(bg.x * 0.12, bg.y * 0.12, bg.z * 0.18 + 0.04, 1.0)
            mvp = self._mvp()
            self.prog["m_mvp"].write(mvp)
            self.prog_pts["m_mvp"].write(mvp)
            self.prog_pts["u_point_size"] = 16.0

            self.grid_vao.render(moderngl.LINES, vertices=self.n_grid)
            if self.n_agent:
                self.agent_vao.render(moderngl.TRIANGLES, vertices=self.n_agent)
            if self.n_points:
                self.point_vao.render(moderngl.POINTS, vertices=self.n_points)

            # status in window title (avoids 2D text / GL state fights)
            mode = getattr(control, "mode", "")
            n = sum(
                1
                for o in control.engine.objects
                if o.enabled and o.__class__.__name__ != "Floor"
            )
            flag = "PAUSED" if state["paused"] else "live"
            self.wnd.title = (
                f"breve 3D  |  t={control.engine.time:.1f}s  n={n}  "
                f"dist={state['dist']:.0f}  {mode}  {flag}"
            )

        def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int) -> None:
            state["azim"] -= dx * 0.008
            state["elev"] += dy * 0.008
            state["auto_orbit"] = False

        def on_mouse_scroll_event(self, x_offset: float, y_offset: float) -> None:
            state["dist"] = float(np.clip(state["dist"] * (0.9 ** y_offset), 3.0, 140.0))
            state["auto_orbit"] = False

        def on_key_event(self, key, action, modifiers):
            keys = self.wnd.keys
            if action != keys.ACTION_PRESS:
                return
            if key == keys.ESCAPE:
                control.engine.stop()
                self.wnd.close()
            elif key == keys.SPACE:
                state["paused"] = not state["paused"]
            elif key == keys.A:
                state["auto_orbit"] = not state["auto_orbit"]
            elif key == keys.N and hasattr(control, "flock_normally"):
                control.flock_normally()
            elif key == keys.O and hasattr(control, "flock_obediently"):
                control.flock_obediently()
            elif key == keys.W and hasattr(control, "flock_wackily"):
                control.flock_wackily()
            elif key == keys.S and hasattr(control, "squish"):
                control.squish()
            elif key == keys.R:
                state["azim"], state["elev"], state["dist"] = 0.75, 0.42, 30.0
                state["auto_orbit"] = True

    # Avoid moderngl_window swallowing argv (pytest / demo flags)
    sys.argv = [sys.argv[0]]
    mglw.run_window_config(Breve3D)
