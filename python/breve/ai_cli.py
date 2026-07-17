"""
Interactive AI scene builder for breve.

  export XAI_API_KEY=xai-...
  pip install -e ".[ai,viz]"
  breve-ai
  breve-ai "drop heavy and light balls to show gravity" --viz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="breve-ai",
        description="Describe a 3D multi-agent / physics scene in English; Grok builds it.",
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional one-shot prompt (otherwise interactive chat)",
    )
    parser.add_argument("--viz", action="store_true", help="Open 3D viewer after build")
    parser.add_argument("--steps", type=int, default=None, help="Sim steps (default: unlimited with --viz)")
    parser.add_argument("--save", type=str, default=None, help="Save scene JSON to path")
    parser.add_argument("--load", type=str, default=None, help="Load scene JSON and run (no LLM)")
    parser.add_argument("--model", type=str, default=None, help="xAI model (default XAI_MODEL or grok-4.5)")
    parser.add_argument("--no-run", action="store_true", help="Only generate/save JSON, do not simulate")
    args = parser.parse_args(argv)

    # Optional .env
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if args.load:
        from breve.scene import build_and_run, load_scene_file

        spec = load_scene_file(args.load)
        print(f"Loaded {args.load}")
        if not args.no_run:
            build_and_run(spec, viz=args.viz, steps=args.steps)
        return 0

    prompt = " ".join(args.prompt).strip()
    if prompt:
        return _oneshot(prompt, args)

    return _repl(args)


def _oneshot(prompt: str, args) -> int:
    from breve.ai_llm import generate_scene, get_api_key
    from breve.scene import build_and_run, save_scene, validate_scene

    if not get_api_key():
        _print_key_help()
        return 1

    print(f"Thinking… ({args.model or os.environ.get('XAI_MODEL', 'grok-4.5')})")
    try:
        result = generate_scene(prompt, model=args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print()
    print(result["explanation"])
    print()
    errs = validate_scene(result["scene"])
    if errs:
        print("Scene validation failed:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    path = args.save or _default_save_path(result["scene"])
    save_scene(result["scene"], path)
    print(f"Saved scene → {path}")

    if args.no_run:
        return 0

    print("Running…")
    try:
        build_and_run(result["scene"], viz=args.viz, steps=args.steps)
    except Exception as e:
        print(f"Run failed: {e}", file=sys.stderr)
        return 1
    return 0


def _repl(args) -> int:
    from breve.ai_llm import generate_scene, refine_scene, get_api_key
    from breve.scene import build_and_run, save_scene, validate_scene

    if not get_api_key():
        _print_key_help()
        return 1

    print("breve AI scene builder  ·  powered by xAI Grok")
    print("Describe a 3D world. Commands:  run | viz | save [path] | show | quit")
    print("Example: heavy red bowling ball and light yellow ping-pong balls falling on stairs")
    print()

    current = None
    history = []

    while True:
        try:
            line = input("breve-ai> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "q"):
            break
        if low == "show":
            if not current:
                print("No scene yet.")
            else:
                print(json_pretty(current))
            continue
        if low.startswith("save"):
            if not current:
                print("No scene yet.")
                continue
            parts = line.split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else _default_save_path(current)
            save_scene(current, path)
            print(f"Saved → {path}")
            continue
        if low in ("run", "viz", "go"):
            if not current:
                print("No scene yet — describe one first.")
                continue
            viz = args.viz or low == "viz"
            try:
                build_and_run(current, viz=viz, steps=args.steps)
            except Exception as e:
                print(f"Run failed: {e}")
            continue

        # treat as NL request
        print("Thinking…")
        try:
            if current is None:
                result = generate_scene(line, model=args.model)
            else:
                # refine existing
                result = refine_scene(current, line, model=args.model)
        except Exception as e:
            print(f"Error: {e}")
            continue

        print()
        print(result["explanation"])
        errs = validate_scene(result["scene"])
        if errs:
            print("Validation issues:")
            for e in errs:
                print(f"  - {e}")
            continue
        current = result["scene"]
        history.append({"role": "user", "content": line})
        history.append({"role": "assistant", "content": result["raw"][:4000]})
        print(f"[scene ready: {current.get('title', 'untitled')} — type 'viz' to open 3D, 'save', or refine]")

    return 0


def _default_save_path(scene: dict) -> str:
    title = str(scene.get("title") or "scene")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:40]
    out = Path("scenes")
    out.mkdir(exist_ok=True)
    return str(out / f"{safe or 'scene'}.json")


def _print_key_help() -> None:
    print(
        """
No XAI_API_KEY set.

1. Create a key at https://console.x.ai
2. export XAI_API_KEY=xai-your-key-here
3. pip install -e ".[ai,viz]"
4. breve-ai "show me gravity with heavy and light balls" --viz

(Optional) XAI_MODEL=grok-4.5
""".strip(),
        file=sys.stderr,
    )


def json_pretty(obj) -> str:
    return json.dumps(obj, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
