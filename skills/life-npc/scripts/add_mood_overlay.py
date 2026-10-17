#!/usr/bin/env python3
"""Add a tracked transparent mood icon to a finished video, preserving its motion.

Usage: python add_mood_overlay.py mood.json
JSON paths are relative to the configuration. Supply source_video, icon_png,
output, display_width, fade_in_seconds, and tracking_states. Tracking uses the
skeletal renderer's per-frame pelvis, scale, and upper_roll_degrees. Coordinates
source_anchor, source_track_point, and source_top_bounds are required when using
tracking_states and refer to the current character's source PNG.
Alternatively, omit tracking_states and supply a fixed top-left position [x,y].
The icon remains upright; no body pixels, poses, timing, or optical flow change.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import time
import uuid

from PIL import Image

from resolve_runtime import resolve_tool


def probe(tool, path):
    result = subprocess.run([str(tool), "-v", "error", "-count_frames", "-show_streams",
                             "-show_format", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def video_info(result):
    stream = next(s for s in result["streams"] if s["codec_type"] == "video")
    fps = Fraction(stream["avg_frame_rate"])
    count = int(stream.get("nb_read_frames", stream.get("nb_frames", 0)))
    if count <= 0 or fps <= 0:
        raise ValueError("Source requires a positive decoded frame count and frame rate")
    return {"width": int(stream["width"]), "height": int(stream["height"]),
            "fps": float(fps), "fps_fraction": str(fps), "frames": count,
            "duration": float(stream.get("duration", result["format"]["duration"]))}


def transform(point, state, anchor):
    angle = math.radians(float(state.get("upper_roll_degrees", 0)))
    dx, dy = point[0]-anchor[0], point[1]-anchor[1]
    scale = float(state["scale"])
    return (state["pelvis"][0]+scale*(math.cos(angle)*dx-math.sin(angle)*dy),
            state["pelvis"][1]+scale*(math.sin(angle)*dx+math.cos(angle)*dy))


def read_exact(stream, length):
    chunks, received = [], 0
    while received < length:
        value = stream.read(length-received)
        if not value:
            break
        chunks.append(value)
        received += len(value)
    return b"".join(chunks)


def tracking_geometry(config):
    """Validate source coordinates explicitly supplied for the current asset."""
    values = []
    for name, size in (("source_anchor", 2), ("source_track_point", 2),
                       ("source_top_bounds", 4)):
        if name not in config:
            raise ValueError(f"tracking_states requires explicit {name} for the current source PNG")
        value = config[name]
        if (not isinstance(value, (list, tuple)) or len(value) != size or
                any(isinstance(item, bool) or not isinstance(item, (int, float)) or
                    not math.isfinite(item) or item < 0 for item in value)):
            raise ValueError(f"{name} must contain {size} finite nonnegative source coordinates")
        values.append(tuple(float(item) for item in value))
    bounds = values[2]
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError("source_top_bounds must be [left, top, right, bottom] with positive width and height")
    return tuple(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="JSON configuration; all paths relative to this file")
    parser.add_argument("--ffmpeg", help="Explicit local FFmpeg executable")
    parser.add_argument("--ffprobe", help="Explicit local FFprobe executable")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    root = config_path.parent
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    geometry = tracking_geometry(cfg) if cfg.get("tracking_states") else None
    path = lambda key: (root/cfg[key]).resolve()
    source, icon_path, output = (path(key) for key in ("source_video", "icon_png", "output"))
    if output.exists() or output == source:
        raise FileExistsError(f"Refusing to overwrite an existing output/source: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    debug = (root/cfg.get("debug_dir", "mood-review")).resolve()
    debug.mkdir(parents=True, exist_ok=True)
    ffmpeg = resolve_tool("ffmpeg", args.ffmpeg, root)
    ffprobe = resolve_tool("ffprobe", args.ffprobe, root)
    started = time.monotonic()
    source_probe = probe(ffprobe, source)
    info = video_info(source_probe)
    width, height, count, fps = (info[key] for key in ("width", "height", "frames", "fps"))
    if width % 2 or height % 2:
        raise ValueError("H264 yuv420p output requires even canvas dimensions")
    icon = Image.open(icon_path).convert("RGBA")
    if not icon.getchannel("A").getbbox():
        raise ValueError("Icon alpha is empty")
    icon_width = int(cfg.get("display_width", 280))
    if icon_width <= 0:
        raise ValueError("display_width must be positive")
    icon_height = round(icon.height*icon_width/icon.width)
    icon = icon.resize((icon_width, icon_height), Image.Resampling.LANCZOS)
    original_alpha = icon.getchannel("A")
    fade = max(0., float(cfg.get("fade_in_seconds", .25)))
    states = None
    if cfg.get("tracking_states"):
        tracking = json.loads(path("tracking_states").read_text(encoding="utf-8"))
        states = tracking["frames"] if isinstance(tracking, dict) else tracking
        if len(states) != count:
            raise ValueError(f"Tracking has {len(states)} frames; decoded source has {count}")
        for frame, state in enumerate(states):
            if state.get("frame", frame) != frame or abs(float(state.get("time", frame/fps))-frame/fps) > 1./fps/4:
                raise ValueError(f"Tracking frame/time mismatch at frame {frame}")
        anchor, track_point, bounds = geometry
        corners = [(bounds[x], bounds[y]) for x in (0, 2) for y in (1, 3)]
    elif "position" not in cfg:
        raise ValueError("Supply tracking_states or a fixed position [x,y]")
    debug_frames = {max(0, min(count-1, round(float(t)*fps)))
                    for t in cfg.get("debug_times", [0, .5, (count-1)/fps/2, (count-1)/fps])}
    temporary = output.with_name(output.stem+".partial-"+uuid.uuid4().hex+".mp4")
    decoder_log = (debug/"decode.log").open("w", encoding="utf-8")
    encoder_log = (debug/"encode.log").open("w", encoding="utf-8")
    decoder = subprocess.Popen([str(ffmpeg), "-v", "error", "-i", str(source), "-map", "0:v:0",
                                "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                               stdout=subprocess.PIPE, stderr=decoder_log)
    encoder = subprocess.Popen([str(ffmpeg), "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{width}x{height}", "-r", info["fps_fraction"], "-i", "pipe:0",
                                "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
                                "-c:v", "libx264", "-preset", cfg.get("preset", "fast"),
                                "-crf", str(int(cfg.get("crf", 17))), "-pix_fmt", "yuv420p",
                                "-c:a", "copy", "-movflags", "+faststart", "-shortest", str(temporary)],
                               stdin=subprocess.PIPE, stderr=encoder_log)
    placements = []
    try:
        for frame in range(count):
            raw = read_exact(decoder.stdout, width*height*3)
            if len(raw) != width*height*3:
                raise RuntimeError(f"Source decode ended early at frame {frame}")
            image = Image.frombytes("RGB", (width, height), raw).convert("RGBA")
            top_y = None
            if states:
                state = states[frame]
                center_x = transform(track_point, state, anchor)[0]
                top_y = min(transform(corner, state, anchor)[1] for corner in corners)
                x = round(center_x-icon_width/2+float(cfg.get("offset_x", 0)))
                y = math.floor(top_y-float(cfg.get("clearance", 16))-icon_height)
            else:
                x, y = (round(float(v)) for v in cfg["position"])
            if x < 0 or y < 0 or x+icon_width > width or y+icon_height > height:
                raise ValueError(f"Icon canvas clipping at frame {frame}: {(x,y,icon_width,icon_height)}")
            alpha = min(1., frame/fps/fade) if fade else 1.
            faded = icon.copy()
            if alpha < 1:
                faded.putalpha(original_alpha.point(lambda value: round(value*alpha)))
            image.alpha_composite(faded, (x, y))
            image = image.convert("RGB")
            encoder.stdin.write(image.tobytes())
            if frame in debug_frames:
                image.save(debug/f"source-composite-{frame:06d}.png")
            placements.append({"frame": frame, "time": frame/fps,
                               "bbox": [x, y, x+icon_width, y+icon_height],
                               "opacity": alpha, "source_body_top": top_y,
                               "clearance_pixels": top_y-y-icon_height if top_y is not None else None})
        if decoder.stdout.read(1):
            raise RuntimeError("Source has extra decoded frames beyond probe count")
        decoder.stdout.close()
        encoder.stdin.close()
        if decoder.wait() != 0 or encoder.wait() != 0:
            raise RuntimeError(f"FFmpeg failed; see {debug}/decode.log and encode.log")
        result_probe = probe(ffprobe, temporary)
        result_info = video_info(result_probe)
        if (result_info["frames"], result_info["width"], result_info["height"], result_info["fps_fraction"]) != (count, width, height, info["fps_fraction"]):
            raise RuntimeError("Decoded output frame count/dimensions/FPS differ from source")
        if output.exists():
            raise FileExistsError(f"Output appeared during rendering; refusing to overwrite: {output}")
        temporary.rename(output)
        metadata = {"source_video": str(source), "icon_png": str(icon_path), "output": str(output),
                    "source": info, "result": result_info, "display_size": [icon_width, icon_height],
                    "elapsed_seconds": time.monotonic()-started,
                    "method": "decode source frames; upright RGBA overlay; H264 encode; no body recomputation",
                    "audio": "copied if present", "visual_qa_status": "not assessed by metadata",
                    "frames": placements}
        (debug/"overlay-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: metadata[key] for key in ("output", "result", "display_size", "elapsed_seconds", "visual_qa_status")}, ensure_ascii=False, indent=2))
    finally:
        for process in (decoder, encoder):
            if process.poll() is None:
                process.kill()
                process.wait()
        decoder_log.close()
        encoder_log.close()
        if temporary.exists():
            temporary.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
