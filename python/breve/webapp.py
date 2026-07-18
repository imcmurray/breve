"""
Breve web interface — chat with Grok to build scenes, watch them in the browser.

  pip install -e ".[ai,web]"
  export XAI_API_KEY=xai-...   # or paste key in the UI
  breve-web
  # open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from breve.ai_llm import generate_scene, refine_scene, get_api_key
from breve.scene import (
    build_scene,
    load_scene_file,
    save_scene,
    snapshot_state,
    validate_scene,
)
from breve.share import decode_scene, encode_scene

STATIC_DIR = Path(__file__).resolve().parent / "web_static"
SCENES_DIR = Path(__file__).resolve().parents[2] / "scenes"

# Curriculum order for first-run autoplay + chips (10 demos)
CURRICULUM = [
    {
        "id": "example_gravity",
        "label": "Gravity + mass",
        "blurb": "Heavy vs light balls — free-fall vs collisions",
    },
    {
        "id": "example_stairs",
        "label": "Stairs",
        "blurb": "Mixed masses rolling down steps (jittered starts)",
    },
    {
        "id": "example_tower",
        "label": "Wrecking ball",
        "blurb": "Aimed heavy ball smashes a box tower",
    },
    {
        "id": "example_pyramid",
        "label": "Pyramid",
        "blurb": "Cannonball into a pyramid of boxes",
    },
    {
        "id": "example_ramps",
        "label": "Ramp race",
        "blurb": "Light vs heavy down parallel ramps",
    },
    {
        "id": "example_arena",
        "label": "Bounce arena",
        "blurb": "Enclosed room — walls keep the action going",
    },
    {
        "id": "example_volley",
        "label": "Mass volley",
        "blurb": "Light swarm meets one heavy cannonball",
    },
    {
        "id": "example_funnel",
        "label": "Funnel",
        "blurb": "Drop into a V of walls — masses settle differently",
    },
    {
        "id": "example_flock",
        "label": "Flock",
        "blurb": "Local rules → swarming in 3D",
    },
    {
        "id": "example_wander",
        "label": "Wander",
        "blurb": "Decentralized agents roaming continuous space",
    },
]

app = FastAPI(title="breve", description="3D multi-agent / ALife with AI scene builder")

# session_id -> { "sim": SceneController, "paused": bool }
_sessions: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    scene: Optional[Dict[str, Any]] = None  # if set, refine existing
    history: Optional[list] = None


class ChatResponse(BaseModel):
    explanation: str
    scene: Dict[str, Any]
    title: str = ""
    share_token: str = ""


class RunRequest(BaseModel):
    scene: Dict[str, Any]
    session_id: Optional[str] = None


class ShareRequest(BaseModel):
    scene: Dict[str, Any]


class StatusResponse(BaseModel):
    has_server_key: bool
    version: str
    default_example: str = "example_gravity"


@app.get("/api/status")
def status() -> StatusResponse:
    from breve import __version__

    return StatusResponse(has_server_key=bool(get_api_key()), version=__version__)


@app.get("/api/curriculum")
def curriculum() -> Dict[str, Any]:
    """Ordered teaching demos for chips + first-run experience."""
    items = []
    for c in CURRICULUM:
        path = SCENES_DIR / f"{c['id']}.json"
        if not path.is_file():
            continue
        try:
            spec = load_scene_file(str(path))
        except Exception:
            continue
        items.append(
            {
                **c,
                "title": spec.get("title") or c["label"],
                "notes": spec.get("notes") or c["blurb"],
            }
        )
    return {"curriculum": items, "default": "example_gravity"}


@app.get("/api/examples")
def examples() -> Dict[str, Any]:
    items = []
    if SCENES_DIR.is_dir():
        for p in sorted(SCENES_DIR.glob("*.json")):
            try:
                spec = load_scene_file(str(p))
                items.append(
                    {
                        "id": p.stem,
                        "title": spec.get("title") or p.stem,
                        "notes": spec.get("notes") or "",
                        "path": p.name,
                    }
                )
            except Exception:
                continue
    return {"examples": items}


@app.get("/api/examples/{example_id}")
def get_example(example_id: str) -> Dict[str, Any]:
    path = SCENES_DIR / f"{example_id}.json"
    if not path.is_file():
        # allow path traversal safe stem only
        raise HTTPException(404, "Example not found")
    return load_scene_file(str(path))


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    try:
        if req.scene:
            result = refine_scene(
                req.scene,
                req.message,
                model=req.model,
                api_key=req.api_key,
            )
        else:
            result = generate_scene(
                req.message,
                history=req.history,
                model=req.model,
                api_key=req.api_key,
            )
    except RuntimeError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(502, f"AI request failed: {e}") from e

    errs = validate_scene(result["scene"])
    if errs:
        raise HTTPException(422, "Invalid scene from model: " + "; ".join(errs))

    # persist last generated for convenience
    try:
        SCENES_DIR.mkdir(parents=True, exist_ok=True)
        title = str(result["scene"].get("title") or "ai_scene")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:40]
        save_scene(result["scene"], str(SCENES_DIR / f"{safe or 'ai_scene'}.json"))
    except Exception:
        pass

    scene = result["scene"]
    try:
        token = encode_scene(scene)
    except Exception:
        token = ""

    return ChatResponse(
        explanation=result["explanation"],
        scene=scene,
        title=str(scene.get("title") or ""),
        share_token=token,
    )


@app.post("/api/share")
def make_share(req: ShareRequest) -> Dict[str, Any]:
    errs = validate_scene(req.scene)
    if errs:
        raise HTTPException(422, "Invalid scene: " + "; ".join(errs))
    token = encode_scene(req.scene)
    # warn if huge (some browsers cap URL length ~8k–32k)
    return {
        "token": token,
        "path": f"/?s={token}",
        "bytes": len(token),
        "ok": len(token) < 12000,
    }


@app.get("/api/share/{token}")
def load_share(token: str) -> Dict[str, Any]:
    try:
        scene = decode_scene(token)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    errs = validate_scene(scene)
    if errs:
        raise HTTPException(422, "Invalid shared scene: " + "; ".join(errs))
    return scene


@app.post("/api/session")
def create_session(req: RunRequest) -> Dict[str, Any]:
    errs = validate_scene(req.scene)
    if errs:
        raise HTTPException(422, "Invalid scene: " + "; ".join(errs))
    try:
        sim = build_scene(req.scene)
    except Exception as e:
        raise HTTPException(400, f"Failed to build scene: {e}") from e

    sid = req.session_id or uuid.uuid4().hex[:12]
    # tear down old
    _sessions.pop(sid, None)
    _sessions[sid] = {"sim": sim, "paused": False, "scene": req.scene}
    snap = snapshot_state(sim)
    return {"session_id": sid, "state": snap}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    _sessions.pop(session_id, None)
    return {"status": "ok"}


@app.websocket("/ws/sim/{session_id}")
async def sim_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    sess = _sessions.get(session_id)
    if not sess:
        await websocket.send_json({"error": "unknown session — call POST /api/session first"})
        await websocket.close()
        return

    sim = sess["sim"]
    # target ~30 Hz sim ticks (may be slower if physics is heavy)
    tick = 1.0 / 30.0
    try:
        await websocket.send_json({"type": "state", "state": snapshot_state(sim)})
        while True:
            # non-blocking receive for pause/resume/step commands
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=tick)
                data = json.loads(msg)
                cmd = data.get("cmd")
                if cmd == "pause":
                    sess["paused"] = True
                elif cmd == "resume":
                    sess["paused"] = False
                elif cmd == "reset":
                    sim = build_scene(sess["scene"])
                    sess["sim"] = sim
                    await websocket.send_json(
                        {"type": "state", "state": snapshot_state(sim)}
                    )
                elif cmd == "step":
                    sim.engine.step()
                    await websocket.send_json(
                        {"type": "state", "state": snapshot_state(sim)}
                    )
            except asyncio.TimeoutError:
                pass

            if not sess.get("paused", False):
                # catch-up a couple of engine steps per frame for snappier physics
                for _ in range(2):
                    sim.engine.step()
                await websocket.send_json(
                    {"type": "state", "state": snapshot_state(sim)}
                )
            else:
                await asyncio.sleep(tick)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
    finally:
        # keep session for reconnect; client may DELETE
        pass


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Breve web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    import uvicorn

    print(f"breve web → http://{args.host}:{args.port}")
    if get_api_key():
        print("  XAI_API_KEY detected (server-side)")
    else:
        print("  No XAI_API_KEY in env — paste a key in the web UI settings")
    uvicorn.run(
        "breve.webapp:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
