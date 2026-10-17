#!/usr/bin/env python3
"""Render one alpha pose at a time over a fixed background, with rigid travel.

Usage: python render_layered_poses.py scene.json [--dry-run]
All relative paths resolve beside the configuration JSON. Anchor coordinates are
in each ORIGINAL pose canvas, before transparent-bounds cropping. Path x/y are
the output-pixel location of that anchor; scale is output/source pixel ratio.
No optical flow, alpha pose crossfade, silhouette editing or mesh warping occurs.

Minimal JSON (paths relative to JSON):
{"background":"background.png","output":"video.mp4","width":1080,
 "height":1920,"fps":30,"duration":6,
 "poses":[{"image":"pose.png","anchor":[470,1600],"phase":0}],
 "path":[{"time":0,"x":300,"y":1700,"scale":0.45},
         {"time":6,"x":700,"y":1500,"scale":0.4}]}
Anchor x/y use the original transparent source canvas. Additional poses use
increasing phases in [0,1), cycling every cycle_seconds (default 1.2). Or use
pose_schedule:[{"time":0,"index":0},{"time":1,"index":1}] for explicit cuts.
Optional: bob_pixels, shadow:false or shadow object, background_fit:'cover',
debug_dir, debug_times, save_source_frames:false, preset:'medium', crf:17.
Default review samples cover every displayed pose plus first/last frames.
Metadata and actual decoded MP4 review PNGs are saved in debug_dir.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import OrderedDict
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


from resolve_runtime import resolve_tool


def resolved(root: Path, value: str) -> Path:
    return (root / Path(value).expanduser()).resolve()


def finite(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def load_plan(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    width, height = config.get("width", 1080), config.get("height", 1920)
    fps = finite(config.get("fps", 30), "fps")
    duration = finite(config.get("duration", 6), "duration")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 or n % 2 for n in (width, height)):
        raise ValueError("width/height must be positive even integers for yuv420p")
    if fps <= 0 or duration <= 0 or abs(fps * duration - round(fps * duration)) > 1e-7:
        raise ValueError("duration * fps must be a positive whole number of output frames")
    bg_path = resolved(root, config["background"])
    with Image.open(bg_path) as original:
        background = original.convert("RGB")
    fit = config.get("background_fit", "exact")
    if background.size != (width, height):
        if fit == "exact":
            raise ValueError(f"Background is {background.size}, expected {(width, height)}. Use background_fit='cover' explicitly to crop.")
        if fit != "cover":
            raise ValueError("background_fit must be exact or cover")
        ratio = max(width / background.width, height / background.height)
        background = background.resize((round(background.width * ratio), round(background.height * ratio)), Image.Resampling.LANCZOS)
        left, top = (background.width-width)//2, (background.height-height)//2
        background = background.crop((left, top, left+width, top+height))
    background = background.convert("RGBA")
    poses = []
    for number, entry in enumerate(config["poses"]):
        image_path = resolved(root, entry["image"])
        with Image.open(image_path) as original:
            if "A" not in original.getbands() and "transparency" not in original.info:
                raise ValueError(f"Pose {number} has no alpha channel: {image_path}")
            image = original.convert("RGBA")
        alpha = image.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None or alpha.getextrema()[0] == 255:
            raise ValueError(f"Pose {number} is empty or fully opaque; supply genuine transparent PNG")
        anchor = entry.get("anchor", [(bbox[0]+bbox[2])/2, bbox[3]-1])
        if len(anchor) != 2:
            raise ValueError(f"Pose {number} anchor requires x,y")
        anchor = [finite(value, f"pose {number} anchor") for value in anchor]
        phase = finite(entry.get("phase", number / len(config["poses"])), f"pose {number} phase")
        if not 0 <= phase < 1:
            raise ValueError("Pose phases must lie in [0,1)")
        poses.append({"image": image.crop(bbox), "source": str(image_path), "canvas": image.size,
                      "bbox": bbox, "anchor": anchor, "phase": phase,
                      "opaque_alpha_pixels": int(np.count_nonzero(np.asarray(alpha) == 255)),
                      "visible_alpha_pixels": int(np.count_nonzero(np.asarray(alpha)))})
    if not poses:
        raise ValueError("At least one pose is required")
    phases = [pose["phase"] for pose in poses]
    if phases != sorted(set(phases)) or phases[0] != 0:
        raise ValueError("Pose phases must be unique, increasing and start at zero")
    nodes = config["path"]
    if not nodes or finite(nodes[0]["time"], "path time") != 0:
        raise ValueError("path must start at time zero")
    previous = -1
    for node in nodes:
        for name in ("time", "x", "y", "scale"):
            node[name] = finite(node[name], f"path {name}")
        if node["time"] <= previous or node["scale"] <= 0 or node["time"] > duration:
            raise ValueError("path times must increase within duration and scales must be positive")
        previous = node["time"]
    schedule = config.get("pose_schedule")
    if schedule is not None:
        previous = -1
        if not schedule or schedule[0]["time"] != 0:
            raise ValueError("pose_schedule must start at zero")
        for node in schedule:
            node["time"] = finite(node["time"], "pose_schedule time")
            index = node["index"]
            if (node["time"] <= previous or not 0 <= node["time"] < duration
                    or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(poses)):
                raise ValueError("Invalid pose_schedule time or pose index")
            previous = node["time"]
    cycle = finite(config.get("cycle_seconds", 1.2), "cycle_seconds")
    bob = finite(config.get("bob_pixels", 0), "bob_pixels")
    if cycle <= 0 or bob < 0:
        raise ValueError("cycle_seconds must be positive, bob_pixels nonnegative")
    output = resolved(root, config["output"])
    debug_dir = resolved(root, config.get("debug_dir", "render-review"))
    times = config.get("debug_times", [])
    debug_frames = sorted(set(min(round(duration*fps)-1, max(0, round(finite(time, "debug time")*fps))) for time in times))
    return {"config": config, "config_path": str(config_path), "root": root,
            "width": width, "height": height, "fps": fps, "duration": duration,
            "count": round(duration*fps), "background": background, "background_source": str(bg_path),
            "poses": poses, "phases": phases, "path": nodes, "schedule": schedule,
            "cycle": cycle, "bob": bob, "output": output,
            "debug_dir": debug_dir, "debug_frames": debug_frames}


def frame_state(plan: dict, number: int) -> dict:
    time = number / plan["fps"]
    nodes = plan["path"]
    index = max(0, bisect_right([n["time"] for n in nodes], time)-1)
    a = nodes[index]
    b = nodes[min(index+1, len(nodes)-1)]
    mix = (time-a["time"])/(b["time"]-a["time"]) if b["time"] > a["time"] else 0
    x, y, scale = [a[key]+(b[key]-a[key])*mix for key in ("x", "y", "scale")]
    phase = (time / plan["cycle"]) % 1
    if plan["schedule"]:
        schedule = plan["schedule"]
        pose_index = schedule[max(0, bisect_right([n["time"] for n in schedule], time)-1)]["index"]
    else:
        pose_index = max(0, bisect_right(plan["phases"], phase)-1)
    bob = -plan["bob"] * math.sin(phase * 2 * math.pi) ** 2
    return {"frame": number, "time": time, "pose_index": pose_index, "x": x, "y": y,
            "scale": scale, "bob": bob, "visible_pose_count": 1}


def soft_shadow(plan: dict, state: dict) -> tuple[Image.Image, tuple[int, int]] | None:
    cfg = plan["config"].get("shadow", {})
    if cfg is False or cfg.get("enabled", True) is False:
        return None
    # Native ellipse shadow is independent of the person's alpha and RGB.
    scale = state["scale"]
    radius_x = float(cfg.get("radius_x", 75)) * scale
    radius_y = float(cfg.get("radius_y", 16)) * scale
    blur = float(cfg.get("blur", 12)) * scale
    opacity = int(cfg.get("opacity", 48))
    if radius_x <= 0 or radius_y <= 0 or blur < 0 or not 0 <= opacity <= 255:
        raise ValueError("Invalid shadow dimensions/opacity")
    margin = math.ceil(blur*3+2)
    w, h = math.ceil(radius_x*2+margin*2), math.ceil(radius_y*2+margin*2)
    layer = Image.new("RGBA", (w,h))
    ImageDraw.Draw(layer).ellipse((margin, margin, w-margin, h-margin), fill=(0,0,0,opacity))
    if blur:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    x = state["x"]+float(cfg.get("offset_x",0))*scale
    y = state["y"]+float(cfg.get("offset_y",0))*scale
    return layer, (round(x-w/2), round(y-h/2))


def render_frame(plan: dict, state: dict, cache: dict) -> tuple[Image.Image, dict]:
    frame = plan["background"].copy()
    shadow = soft_shadow(plan, state)
    if shadow:
        frame.alpha_composite(shadow[0], shadow[1])
    pose = plan["poses"][state["pose_index"]]
    scale = state["scale"]
    size = (max(1, round(pose["image"].width*scale)), max(1, round(pose["image"].height*scale)))
    key = (state["pose_index"], size)
    if key not in cache:
        cache[key] = pose["image"].resize(size, Image.Resampling.LANCZOS)
        if len(cache) > 16:
            cache.popitem(last=False)
    image = cache[key]
    left = round(state["x"] - (pose["anchor"][0]-pose["bbox"][0])*scale)
    top = round(state["y"] + state["bob"] - (pose["anchor"][1]-pose["bbox"][1])*scale)
    bounds = [left,top,left+image.width,top+image.height]
    clipped = left < 0 or top < 0 or bounds[2] > plan["width"] or bounds[3] > plan["height"]
    if clipped and not plan["config"].get("allow_canvas_clipping", False):
        raise ValueError(f"Pose would leave output canvas at frame {state['frame']}: bounds={bounds}. Adjust path/scale; no automatic silhouette clipping allowed.")
    frame.alpha_composite(image, (left,top))
    return frame.convert("RGB"), {**state, "visible_pose_bounds": bounds, "canvas_clipping": clipped}


def probe(path: Path, ffprobe: Path) -> dict:
    result = subprocess.run([str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def main() -> int:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output video")
    args = parser.parse_args()
    process = None
    try:
        config_path = args.config.expanduser().resolve()
        plan = load_plan(config_path)
        ffmpeg = resolve_tool("ffmpeg", args.ffmpeg, config_path.parent)
        ffprobe = resolve_tool("ffprobe", args.ffprobe, config_path.parent)
        if plan["output"].exists() and not args.overwrite and not args.dry_run:
            raise ValueError("Output exists; choose a new path or pass --overwrite")
        cache, states, phase_samples = OrderedDict(), [], {}
        # Validate placement for every output frame before starting the encoder.
        for number in range(plan["count"]):
            state = frame_state(plan, number)
            phase_samples.setdefault(state["pose_index"], number)
            pose = plan["poses"][state["pose_index"]]
            left = round(state["x"]-(pose["anchor"][0]-pose["bbox"][0])*state["scale"])
            top = round(state["y"]+state["bob"]-(pose["anchor"][1]-pose["bbox"][1])*state["scale"])
            bounds = [left,top,left+round(pose["image"].width*state["scale"]),top+round(pose["image"].height*state["scale"])]
            if (min(bounds[:2]) < 0 or bounds[2] > plan["width"] or bounds[3] > plan["height"]) and not plan["config"].get("allow_canvas_clipping", False):
                raise ValueError(f"Pose leaves output canvas at frame {number}: {bounds}")
        # Always include every displayed source pose and first/last frames.
        plan["debug_frames"] = sorted(set(plan["debug_frames"] + [0, plan["count"]-1] + list(phase_samples.values())))
        preparation_done = time.perf_counter()
        summary = {"output": str(plan["output"]), "width": plan["width"], "height": plan["height"],
                   "fps": plan["fps"], "duration_seconds": plan["duration"], "frame_count": plan["count"],
                   "pose_count": len(plan["poses"]), "single_pose_only": True,
                   "method": "fixed background + one source-alpha pose + rigid placement; no optical flow or pose dissolves",
                   "dry_run": args.dry_run}
        if args.dry_run:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        plan["output"].parent.mkdir(parents=True, exist_ok=True)
        plan["debug_dir"].mkdir(parents=True, exist_ok=True)
        temporary = plan["output"].with_name(plan["output"].stem+".encoding.mp4")
        log_path = plan["debug_dir"] / "ffmpeg.log"
        command = [str(ffmpeg), "-hide_banner", "-loglevel", "warning", "-y", "-f", "rawvideo",
                   "-pixel_format", "rgb24", "-video_size", f"{plan['width']}x{plan['height']}",
                   "-framerate", str(plan["fps"]), "-i", "pipe:0", "-an", "-c:v", "libx264",
                   "-preset", plan["config"].get("preset", "medium"), "-crf", str(plan["config"].get("crf", 17)),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
        debug_entries = []
        with log_path.open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            assert process.stdin is not None
            for number in range(plan["count"]):
                frame, state = render_frame(plan, frame_state(plan, number), cache)
                process.stdin.write(frame.tobytes())
                states.append(state)
                if plan["config"].get("save_source_frames", False) and number in plan["debug_frames"]:
                    target = plan["debug_dir"] / f"source-composite-{number:06d}.png"
                    frame.save(target)
                    debug_entries.append({"frame": number, "image": str(target)})
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg failed; see {log_path}")
        encoding_done = time.perf_counter()
        actual = probe(temporary, ffprobe)
        stream = actual["streams"][0]
        if (int(stream.get("nb_read_frames", 0)) != plan["count"] or
                Fraction(stream["avg_frame_rate"]) != Fraction(str(plan["fps"])) or
                (int(stream["width"]), int(stream["height"])) != (plan["width"], plan["height"]) or
                abs(float(stream.get("duration", 0))-plan["duration"]) > 1/plan["fps"]):
            raise RuntimeError(f"Encoded dimensions/rate/count differ from plan: {stream}")
        temporary.replace(plan["output"])
        metadata_checked = time.perf_counter()
        review_dir = plan["debug_dir"] / "decoded-review"
        review_command = [sys.executable, str(Path(__file__).with_name("inspect_video.py")),
                          str(plan["output"]), "--output-dir", str(review_dir),
                          "--times", ",".join(str(n/plan["fps"]) for n in plan["debug_frames"]),
                          "--ffmpeg", str(ffmpeg), "--ffprobe", str(ffprobe)]
        review = subprocess.run(review_command, capture_output=True, text=True, check=True)
        inspection = json.loads((review_dir / "inspection.json").read_text(encoding="utf-8"))
        if len(inspection["samples"]) != len(plan["debug_frames"]):
            raise RuntimeError("Decoded review omitted requested frames")
        finished = time.perf_counter()
        timings = {"preparation_seconds": preparation_done-started,
                   "compose_and_encode_seconds": encoding_done-preparation_done,
                   "probe_seconds": metadata_checked-encoding_done,
                   "decoded_review_seconds": finished-metadata_checked,
                   "total_elapsed_seconds": finished-started}
        metadata = {**summary, "dry_run": False, "config": str(config_path),
                    "background": plan["background_source"], "ffmpeg": str(ffmpeg), "ffprobe": str(ffprobe),
                    "poses": [{key:value for key,value in pose.items() if key != "image"} for pose in plan["poses"]],
                    "frame_states": states, "source_composites": debug_entries, "probe": actual,
                    "encoded_metadata_verified": True,
                    "decoded_samples_verified": True,
                    "decoded_review": str(review_dir / "inspection.json"),
                    "timings": timings,
                    "environment": {"python": sys.version, "python_executable": sys.executable,
                                    "platform": platform.platform(), "machine": platform.machine(),
                                    "processor": platform.processor(), "numpy": np.__version__,
                                    "pillow": Image.__version__, "network_used": False,
                                    "installations_attempted": False},
                    "visual_quality_verified": False,
                    "manual_review_required": "Decode actual MP4 and inspect each displayed pose, hair/umbrella/hand/feet contours, path start/end and cadence."}
        target = plan["debug_dir"] / "render-metadata.json"
        target.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({**summary, "metadata": str(target), "encoded_frame_count": stream["nb_read_frames"],
                          "encoded_metadata_verified": True, "decoded_samples_verified": True,
                          "visual_quality_verified": False, "timings": timings}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, FileNotFoundError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
