"""Encode/decode scene JSON for shareable URLs (no server state required)."""

from __future__ import annotations

import base64
import json
import zlib
from typing import Any, Dict
from urllib.parse import quote, unquote


def encode_scene(scene: Dict[str, Any]) -> str:
    """Compress + base64url encode a scene dict for query strings."""
    raw = json.dumps(scene, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    token = base64.urlsafe_b64encode(compressed).decode("ascii").rstrip("=")
    return token


def decode_scene(token: str) -> Dict[str, Any]:
    """Inverse of encode_scene. Raises ValueError on bad input."""
    if not token or not isinstance(token, str):
        raise ValueError("empty share token")
    # restore padding
    pad = "=" * (-len(token) % 4)
    try:
        compressed = base64.urlsafe_b64decode(token + pad)
        raw = zlib.decompress(compressed)
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"invalid share token: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("share payload must be a JSON object")
    return data


def share_path(scene: Dict[str, Any], *, base: str = "/") -> str:
    """Return path+query suitable for location (relative)."""
    token = encode_scene(scene)
    return f"{base}?s={quote(token, safe='')}"
