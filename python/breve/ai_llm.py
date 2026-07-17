"""
xAI / Grok client for natural-language → breve scene JSON.

Uses the OpenAI-compatible API at https://api.x.ai/v1
Env: XAI_API_KEY (required)
Optional: XAI_MODEL (default grok-4.5)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from breve.scene import SCENE_SCHEMA_DOC

DEFAULT_MODEL = os.environ.get("XAI_MODEL", "grok-4.5")
DEFAULT_BASE = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")


SYSTEM_PROMPT = f"""You are Breve Scene Architect — an expert at composing 3D multi-agent
and physics demos for the breve simulator (lightweight continuous 3D agents / ALife).

The user describes what they want to see. You respond with:
1) A short plain-language explanation (2-4 sentences) of what you built and what to watch for.
2) A single JSON scene object in a ```json fenced block```.

{SCENE_SCHEMA_DOC}

Output rules:
- ONLY use the JSON schema above — never invent Python code or extra keys.
- Prefer mode "physics" for gravity/bounce/mass demos; "kinematic" for flocking/wandering.
- Keep object counts reasonable (≤ 40 dynamic bodies, ≤ 80 flock agents).
- Always make the impact of the user's request *visible* (floor, camera, drop height, colors).
- For mass demos: vary mass AND radius AND color so weight is readable.
- For gravity demos: floor + drop from y≥3 + some horizontal velocity.
- Do not wrap JSON in commentary inside the fence.
"""


def get_api_key(override: Optional[str] = None) -> Optional[str]:
    if override and override.strip():
        return override.strip()
    return os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")


def require_client(api_key: Optional[str] = None):
    key = get_api_key(api_key)
    if not key:
        raise RuntimeError(
            "No API key found. Set XAI_API_KEY (from https://console.x.ai) "
            "or paste a key in the web UI:\n"
            "  export XAI_API_KEY=xai-...\n"
            "Optional: export XAI_MODEL=grok-4.5"
        )
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "AI features need the openai package:\n"
            "  pip install -e '.[ai]'"
        ) from e
    return OpenAI(api_key=key, base_url=DEFAULT_BASE)


def generate_scene(
    user_prompt: str,
    *,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call Grok to produce a scene dict from a natural language request.

    Returns {"explanation": str, "scene": dict, "raw": str}
    """
    client = require_client(api_key)
    model = model or DEFAULT_MODEL
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    raw = _chat(client, model, messages)
    explanation, scene = _parse_response(raw)
    return {"explanation": explanation, "scene": scene, "raw": raw}


def _chat(client, model: str, messages: List[Dict[str, str]]) -> str:
    """Prefer chat.completions; fall back to responses API."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.4,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": m["role"], "content": m["content"]} for m in messages
                ],
            )
            text = getattr(resp, "output_text", None)
            if text:
                return text
            return str(resp)
        except Exception as e:
            raise RuntimeError(f"xAI request failed: {e}") from e


def _parse_response(raw: str) -> tuple[str, Dict[str, Any]]:
    from breve.scene import loads_scene

    explanation = raw
    scene = None
    if "```" in raw:
        before = raw.split("```", 1)[0].strip()
        if before:
            explanation = before
        scene = loads_scene(raw)
    else:
        try:
            scene = loads_scene(raw)
            explanation = scene.get("notes") or "Built your scene."
        except Exception as e:
            raise RuntimeError(
                f"Model did not return valid scene JSON.\nParse error: {e}\n\nRaw:\n{raw[:1500]}"
            ) from e
    if not isinstance(scene, dict):
        raise RuntimeError("Scene JSON was not an object")
    return explanation, scene


def refine_scene(
    scene: Dict[str, Any],
    instruction: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask the model to edit an existing scene JSON per user instruction."""
    prompt = (
        "Here is the current scene JSON. Modify it according to the instruction. "
        "Return explanation + full updated JSON fence.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Current scene:\n```json\n{json.dumps(scene, indent=2)}\n```"
    )
    return generate_scene(prompt, model=model, api_key=api_key)
