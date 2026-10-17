#!/usr/bin/env python3
"""Locate local FFmpeg tools without downloading or installing anything."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


ENVIRONMENT_NAMES = {
    "ffmpeg": ("FFMPEG_PATH", "HYPERFRAMES_FFMPEG_PATH"),
    "ffprobe": ("FFPROBE_PATH",),
}


def executable(path: str | Path) -> Path | None:
    value = Path(path).expanduser()
    if value.is_file() and os.access(value, os.X_OK):
        return value.resolve()
    return None


def search_roots(start: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    for origin in (start or Path.cwd(), Path(__file__).resolve().parent):
        for root in (origin, *origin.parents):
            if root not in roots:
                roots.append(root)
    return roots


def resolve_tool(name: str, explicit: str | None = None, start: Path | None = None) -> Path:
    """Explicit flag > environment > PATH > a nearby workspace tool installation."""
    if name not in ENVIRONMENT_NAMES:
        raise ValueError(f"Unsupported tool: {name}")
    if explicit:
        found = executable(explicit) or (
            executable(shutil.which(explicit)) if shutil.which(explicit) else None
        )
        if found:
            return found
        raise FileNotFoundError(f"--{name} is not an executable file: {explicit}")
    for key in ENVIRONMENT_NAMES[name]:
        if os.environ.get(key):
            found = executable(os.environ[key])
            if found:
                return found
    found_on_path = shutil.which(name)
    if found_on_path:
        return Path(found_on_path).resolve()
    package = "@ffmpeg-installer" if name == "ffmpeg" else "@ffprobe-installer"
    for root in search_roots(start):
        for parent in (root / ".render-tools", root):
            for candidate in sorted((parent / "node_modules" / package).glob(f"*/{name}*")):
                if candidate.name in (name, f"{name}.exe") and executable(candidate):
                    return candidate.resolve()
        for candidate in (root / ".venv" / "bin" / name, root / "bin" / name):
            if executable(candidate):
                return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot find {name}. Supply --{name} /absolute/path/{name}, set "
        f"{ENVIRONMENT_NAMES[name][0]}, or put it on PATH. No installation was attempted."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", help="Explicit FFmpeg executable")
    parser.add_argument("--ffprobe", help="Explicit FFprobe executable")
    parser.add_argument("--start", type=Path, default=Path.cwd(), help="Workspace to search upward from")
    args = parser.parse_args()
    result = {"python": sys.executable, "network_used": False, "installations_attempted": False}
    for name in ENVIRONMENT_NAMES:
        try:
            result[name] = str(resolve_tool(name, getattr(args, name), args.start))
        except FileNotFoundError as exc:
            result[name] = None
            result[f"{name}_error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(result.get(name) for name in ENVIRONMENT_NAMES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
