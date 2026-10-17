#!/usr/bin/env python3
"""Apply a fixed, scene-specific color look without changing motion or geometry.

Config format: {"profiles": {"daylight": {"saturation": 0.9,
"hue_adjustments": [{"center": 220, "width": 65, "multiplier": 0.78}],
"tone_curve": [[0, 0.012], [1, 0.935]],
"midtone_rgb_offset": [0.002, 0.001, -0.003]}}}.

Each profile becomes a 33-cube LUT. Hue adjustments use a cosine falloff;
saturation changes around Rec. 709 luminance, so neutrals remain neutral.
The fixed RGB tone curve and optional midtone tint are then applied to every
frame. No temporal estimation, auto exposure, resizing, or optical flow occurs.
"""
from __future__ import annotations

import argparse
import bisect
import colorsys
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

from resolve_runtime import resolve_tool


def number(value: object, label: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{label} must be finite and between {lower} and {upper}")
    return result


def validate_profile(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("The selected profile must be an object")
    saturation = number(value.get("saturation", 1.0), "saturation", 0, 2)
    adjustments = value.get("hue_adjustments", [])
    if not isinstance(adjustments, list):
        raise ValueError("hue_adjustments must be a list")
    hues = []
    for index, item in enumerate(adjustments):
        if not isinstance(item, dict):
            raise ValueError(f"hue_adjustments[{index}] must be an object")
        center = number(item.get("center"), f"hue_adjustments[{index}].center", 0, 360)
        width = number(item.get("width"), f"hue_adjustments[{index}].width", 0, 180)
        if width == 0:
            raise ValueError("A hue adjustment width must be greater than zero")
        multiplier = number(item.get("multiplier"), f"hue_adjustments[{index}].multiplier", 0, 2)
        hues.append({"center": center % 360, "width": width, "multiplier": multiplier})
    curve = value.get("tone_curve", [[0, 0], [1, 1]])
    if not isinstance(curve, list) or len(curve) < 2:
        raise ValueError("tone_curve needs at least two [input, output] points")
    points = []
    for index, point in enumerate(curve):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"tone_curve[{index}] must be [input, output]")
        points.append((number(point[0], "curve input", 0, 1),
                       number(point[1], "curve output", 0, 1)))
    if points[0][0] != 0 or points[-1][0] != 1:
        raise ValueError("tone_curve inputs must start at 0 and end at 1")
    if any(a[0] >= b[0] for a, b in zip(points, points[1:])):
        raise ValueError("tone_curve input positions must increase strictly")
    if any(a[1] > b[1] for a, b in zip(points, points[1:])):
        raise ValueError("tone_curve outputs must not decrease")
    offsets = value.get("midtone_rgb_offset", [0, 0, 0])
    if not isinstance(offsets, (list, tuple)) or len(offsets) != 3:
        raise ValueError("midtone_rgb_offset must contain three RGB values")
    offsets = tuple(number(item, "midtone_rgb_offset", -0.25, 0.25) for item in offsets)
    return {"saturation": saturation, "hue_adjustments": hues,
            "tone_curve": points, "midtone_rgb_offset": offsets}


def load_profile(path: Path, name: str) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("profiles"), dict):
        raise ValueError("Config must contain a profiles object")
    if name not in config["profiles"]:
        raise ValueError(f"Unknown profile {name!r}; available: {', '.join(config['profiles'])}")
    return validate_profile(config["profiles"][name])


def transform_rgb(rgb: tuple[float, float, float], profile: dict) -> tuple[float, float, float]:
    red, green, blue = rgb
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    hue = colorsys.rgb_to_hsv(red, green, blue)[0] * 360
    factor = profile["saturation"]
    for item in profile["hue_adjustments"]:
        distance = abs((hue - item["center"] + 180) % 360 - 180)
        if distance < item["width"]:
            weight = (1 + math.cos(math.pi * distance / item["width"])) / 2
            factor *= 1 + (item["multiplier"] - 1) * weight
    chroma_mapped = tuple(luma + (channel - luma) * factor for channel in rgb)
    points = profile["tone_curve"]
    positions = [point[0] for point in points]

    def tone(channel: float) -> float:
        channel = min(1.0, max(0.0, channel))
        index = min(len(points) - 2, max(0, bisect.bisect_right(positions, channel) - 1))
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        return y0 + (channel - x0) * (y1 - y0) / (x1 - x0)

    midtone_weight = 4 * luma * (1 - luma)
    return tuple(min(1.0, max(0.0, tone(channel) + offset * midtone_weight))
                 for channel, offset in zip(chroma_mapped, profile["midtone_rgb_offset"]))


def write_lut(path: Path, profile: dict, profile_name: str, size: int = 33) -> None:
    """Write a .cube LUT in the specified B/G/R loop order (red fastest)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not put user-supplied names in a quoted LUT title.
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write('# Fixed scene grade generated by life-npc\n')
        handle.write('TITLE "LifeNPC fixed grade"\n')
        handle.write(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        denominator = size - 1
        for blue in range(size):
            for green in range(size):
                for red in range(size):
                    mapped = transform_rgb((red / denominator, green / denominator, blue / denominator), profile)
                    handle.write("{:.8f} {:.8f} {:.8f}\n".format(*mapped))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-time", type=float, help="Render one PNG frame at this time in seconds")
    parser.add_argument("--lut-output", type=Path, help="Save LUT here; defaults to output with .cube suffix")
    parser.add_argument("--ffmpeg", help="Explicit FFmpeg executable; otherwise resolved locally")
    parser.add_argument("--crf", type=int, default=17)
    parser.add_argument("--preset", default="fast", choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        source = args.input.expanduser().resolve()
        target = args.output.expanduser().resolve()
        lut = args.lut_output.expanduser().resolve() if args.lut_output else target.with_suffix(".cube")
        if not source.is_file():
            raise ValueError(f"Input video does not exist: {source}")
        if source in (target, lut) or target == lut:
            raise ValueError("Input video, output, and LUT must have different paths")
        if target.exists() and not args.overwrite:
            raise ValueError(f"Output already exists; use --overwrite: {target}")
        if lut.exists() and not args.overwrite:
            raise ValueError(f"LUT already exists; use --overwrite: {lut}")
        if not 0 <= args.crf <= 51:
            raise ValueError("--crf must be between 0 and 51")
        if args.frame_time is not None:
            if not math.isfinite(args.frame_time) or args.frame_time < 0:
                raise ValueError("--frame-time must be a finite nonnegative number")
            if target.suffix.lower() != ".png":
                raise ValueError("A frame preview output must have the .png extension")
        elif target.suffix.lower() != ".mp4":
            raise ValueError("A video output must have the .mp4 extension")
        profile = load_profile(args.config.expanduser().resolve(), args.profile)
        ffmpeg = resolve_tool("ffmpeg", args.ffmpeg, source.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        write_lut(lut, profile, args.profile)
        # Use a short ASCII working filename, avoiding filter-language escaping
        # hazards for user paths with quotes, colons, Unicode, or backslashes.
        with tempfile.TemporaryDirectory(prefix="plv-grade-") as temporary:
            working_lut = Path(temporary) / "grade.cube"
            working_lut.write_bytes(lut.read_bytes())
            command = [str(ffmpeg), "-hide_banner", "-loglevel", "error",
                       "-y" if args.overwrite else "-n", "-i", str(source)]
            if args.frame_time is not None:
                command += ["-ss", str(args.frame_time)]
            command += ["-map", "0:v:0", "-vf", "lut3d=file=grade.cube:interp=tetrahedral"]
            if args.frame_time is not None:
                command += ["-frames:v", "1", "-an", str(target)]
            else:
                command += ["-map", "0:a?", "-c:v", "libx264", "-crf", str(args.crf),
                            "-preset", args.preset, "-pix_fmt", "yuv420p", "-vsync", "0",
                            "-c:a", "copy", "-movflags", "+faststart", str(target)]
            subprocess.run(command, cwd=temporary, check=True)
        if not target.is_file() or target.stat().st_size == 0:
            raise ValueError("FFmpeg did not produce an output (the preview time may exceed the clip)")
        print(json.dumps({"input": str(source), "output": str(target), "lut": str(lut),
                          "profile": args.profile, "frame_time": args.frame_time,
                          "geometry_changed": False, "temporal_interpolation": False}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"grade_video: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
