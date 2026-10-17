#!/usr/bin/env python3
"""Interpolate small, explicitly bounded motions over an unchanged scene image.

Usage: python render_local_motion.py motion.json [--overwrite]
Paths in JSON are relative to that JSON file. Required schema:
{"base_image":"base.png","output":"video.mp4","width":1080,"height":1920,
 "fps":30,"duration":3,"frames":[{"image":"a.png","time":0},
 {"image":"b.png","time":1}],"regions":[{"name":"head",
 "box":[300,400,400,450],"feather":12}]}
The final anchor is held until duration. Only the named rectangular regions move.
Their feather bands must lie on stationary background, beyond all hair/hand edges.
Optional frame field "transition" controls arrival at that anchor: "mci" (default)
interpolates from the previous anchor; "cut" holds the previous source unchanged
until this anchor's time, then switches to this source without a dissolve.
For example: {"image":"look-down.png","time":1.2,"transition":"cut"}.
This performs motion-compensated interpolation; it does not guarantee correct anatomy.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

from resolve_runtime import resolve_tool


def positive_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def read_configuration(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    directory = path.parent
    def source(value, label):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a path string")
        target = (directory / value).expanduser().resolve()
        if not target.is_file():
            raise ValueError(f"{label} does not exist: {target}")
        return target
    base = source(raw.get("base_image"), "base_image")
    with Image.open(base) as image:
        native_size = image.size
    width = integer(raw.get("width", native_size[0]), "width", 2)
    height = integer(raw.get("height", native_size[1]), "height", 2)
    if width % 2 or height % 2:
        raise ValueError("width and height must be even for yuv420p output")
    if (width, height) != native_size:
        raise ValueError(f"base_image must already be {width}x{height}; resizing source images is not implicit")
    fps = integer(raw.get("fps", 30), "fps", 1)
    if fps > 120:
        raise ValueError("fps must be <= 120")
    duration = positive_number(raw.get("duration"), "duration")
    count = round(duration * fps)
    if not count or abs(count / fps - duration) > 1e-7:
        raise ValueError("duration * fps must be a positive integer, ensuring exact frame count")
    frames = []
    for index, item in enumerate(raw.get("frames", [])):
        if not isinstance(item, dict):
            raise ValueError(f"frames[{index}] must be an object")
        image_path = source(item.get("image"), f"frames[{index}].image")
        time = item.get("time")
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not math.isfinite(time) or time < 0 or time > duration:
            raise ValueError(f"frames[{index}].time must lie within 0..duration")
        if frames and time <= frames[-1]["time"]:
            raise ValueError("Anchor times must be strictly increasing")
        if frames and time - frames[-1]["time"] < 1 / fps - 1e-7:
            raise ValueError("Anchor spacing must be at least one output frame")
        with Image.open(image_path) as image:
            if image.size != native_size:
                raise ValueError(f"Anchor dimensions differ from base_image: {image_path}")
        transition = item.get("transition", "mci")
        if transition not in ("mci", "cut"):
            raise ValueError(f"frames[{index}].transition must be 'mci' or 'cut'")
        frames.append({"image": image_path, "time": float(time), "transition": transition})
    if not frames or frames[0]["time"] != 0:
        raise ValueError("At least one anchor is required; its time must be 0")
    regions = []
    names = set()
    for index, item in enumerate(raw.get("regions", [])):
        if not isinstance(item, dict):
            raise ValueError(f"regions[{index}] must be an object")
        name = item.get("name", f"region-{index}")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Region names must be non-empty and unique")
        names.add(name)
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{name}.box must be [x,y,width,height]")
        x, y, w, h = [integer(value, f"{name}.box[{i}]", 16 if i >= 2 else 0) for i, value in enumerate(box)]
        if x + w > width or y + h > height:
            raise ValueError(f"Region {name} extends outside the image")
        feather = integer(item.get("feather", 8), f"{name}.feather", 0)
        if feather * 2 >= min(w, h):
            raise ValueError(f"{name}.feather must be less than half its width and height")
        for previous in regions:
            a, b, c, d = previous["box"]
            if min(x+w, a+c) > max(x, a) and min(y+h, b+d) > max(y, b):
                raise ValueError(f"Regions {name} and {previous['name']} overlap. Use a single union box to avoid conflicting flow.")
        regions.append({"name": name, "box": [x, y, w, h], "feather": feather})
    if not regions:
        raise ValueError("At least one explicitly bounded motion region is required")
    output = raw.get("output")
    if not isinstance(output, str) or not output:
        raise ValueError("output must be a path string")
    output_path = (directory / output).expanduser().resolve()
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("output must end in .mp4")
    return {"base_image": base, "output": output_path, "width": width, "height": height,
            "fps": fps, "duration": duration, "frame_count": count, "frames": frames, "regions": regions}


def feather_mask(width: int, height: int, feather: int) -> Image.Image:
    if not feather:
        return Image.new("L", (width, height), 255)
    xs = np.minimum(np.arange(width), np.arange(width)[::-1])[None, :]
    ys = np.minimum(np.arange(height), np.arange(height)[::-1])[:, None]
    amount = np.clip(np.minimum(xs, ys) / feather, 0, 1)
    amount = amount * amount * (3 - 2 * amount)
    return Image.fromarray(np.rint(amount * 255).astype(np.uint8), "L")


def coverage_report(config: dict) -> dict:
    """Diagnostic differences only: these cannot semantically identify missed hair."""
    with Image.open(config["base_image"]) as image:
        base = np.asarray(image.convert("RGB"), dtype=np.int16)
    union = np.zeros(base.shape[:2], dtype=bool)
    for region in config["regions"]:
        x, y, w, h = region["box"]
        union[y:y+h, x:x+w] = True
    samples = []
    for frame in config["frames"]:
        with Image.open(frame["image"]) as image:
            delta = np.abs(np.asarray(image.convert("RGB"), dtype=np.int16) - base).max(axis=2)
        edge_changes = {}
        for region in config["regions"]:
            x, y, w, h = region["box"]
            crop = delta[y:y+h, x:x+w]
            thickness = max(2, region["feather"])
            edge = np.ones((h, w), dtype=bool)
            edge[thickness:-thickness, thickness:-thickness] = False
            edge_changes[region["name"]] = round(float((crop[edge] > 24).mean()), 6)
        outside = delta[~union]
        samples.append({"image": str(frame["image"]), "time": frame["time"],
                        "outside_roi_changed_fraction_above_24": round(float((outside > 24).mean()), 6) if outside.size else 0,
                        "roi_edge_changed_fraction_above_24": edge_changes})
    return {"manual_review_required": True,
            "meaning": "Changed pixels near ROI edges may indicate motion or hair outside the crop; background/exposure changes also trigger this diagnostic. Inspect the original and crop at full size.",
            "samples": samples}


def checked_run(command: list[str], log_path: Path) -> None:
    with log_path.open("wb") as log:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=log)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed ({result.returncode}):\n{log_path.read_text(errors='replace')[-6000:]}")


def probe_video(path: Path, ffprobe: Path) -> dict:
    result = subprocess.run([str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_streams", "-show_format", "-of", "json", str(path)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def render(config: dict, config_path: Path, ffmpeg: Path, ffprobe: Path, work: Path, crf: int) -> dict:
    width, height, fps = config["width"], config["height"], config["fps"]
    images = [Image.open(frame["image"]).convert("RGB") for frame in config["frames"]]
    region_runs = []
    for ri, region in enumerate(config["regions"]):
        x, y, w, h = region["box"]
        crops = [image.crop((x, y, x+w, y+h)) for image in images]
        segments = []
        for si in range(len(images) - 1):
            start, end = config["frames"][si]["time"], config["frames"][si+1]["time"]
            delta = end - start
            # A deliberate pose change holds the previous complete source until
            # the new anchor time. It never dissolves two faces or bodies.
            if config["frames"][si+1]["transition"] == "cut" or config["frames"][si]["image"] == config["frames"][si+1]["image"]:
                segments.append({"start": start, "end": end, "delta": delta, "kind": "cut"})
                continue
            folder = work / f"region-{ri:02d}" / f"segment-{si:03d}"
            folder.mkdir(parents=True)
            for index, image in enumerate((crops[si], crops[si], crops[si+1], crops[si+1], crops[si+1])):
                image.save(folder / f"in-{index:03d}.png")
            # Duplicate endpoints provide the four-frame context required by minterpolate.
            # The actual A->B interval occupies input time delta..2*delta.
            filter_text = f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:mb_size=8:search_param=16:vsbmc=1:scd=none"
            checked_run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-framerate", f"{1/delta:.12g}",
                         "-start_number", "0", "-i", str(folder / "in-%03d.png"), "-vf", filter_text,
                         "-vsync", "0", "-start_number", "0", str(folder / "flow-%06d.png")], folder / "ffmpeg.log")
            generated = sorted(folder.glob("flow-*.png"))
            if not generated:
                raise RuntimeError(f"Interpolation produced no frames for region {ri}, segment {si}")
            segments.append({"start": start, "end": end, "delta": delta, "files": generated, "kind": "mci"})
        region_runs.append({**region, "crops": crops, "segments": segments, "mask": feather_mask(w, h, region["feather"])})
    base = Image.open(config["base_image"]).convert("RGB")
    output = config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".partial.mp4")
    command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an",
               "-frames:v", str(config["frame_count"]), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial)]
    key_times = [frame["time"] for frame in config["frames"]]
    with (work / "encode.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for number in range(config["frame_count"]):
                time = number / fps
                index = max(0, bisect_right(key_times, time) - 1)
                image = base.copy()
                for region in region_runs:
                    x, y, _, _ = region["box"]
                    if index == len(images)-1 or abs(time-key_times[index]) < 1e-8:
                        patch = region["crops"][index]
                    else:
                        segment = region["segments"][index]
                        if segment["kind"] == "cut":
                            patch = region["crops"][index]
                        else:
                            sample = min(len(segment["files"])-1, max(0, round((segment["delta"] + time-segment["start"]) * fps)))
                            with Image.open(segment["files"][sample]) as source:
                                patch = source.convert("RGB")
                    image.paste(patch, (x, y), region["mask"])
                process.stdin.write(image.tobytes())
            process.stdin.close()
            if process.wait():
                raise RuntimeError((work / "encode.log").read_text(errors="replace")[-6000:])
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            partial.unlink(missing_ok=True)
            raise
    metadata = probe_video(partial, ffprobe)
    stream = metadata["streams"][0]
    actual_count = int(stream.get("nb_read_frames", stream.get("nb_frames", 0)))
    actual_duration = float(stream.get("duration", metadata["format"].get("duration", 0)))
    if actual_count != config["frame_count"] or abs(actual_duration-config["duration"]) > 1/fps + 0.001:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Encoded output length mismatch: {actual_count} frames, {actual_duration}s")
    partial.replace(output)
    return {"output": str(output), "frame_count": actual_count, "duration": actual_duration,
            "width": int(stream["width"]), "height": int(stream["height"]), "fps": stream["avg_frame_rate"],
            "method": "FFmpeg motion-compensated interpolation per non-overlapping ROI; fixed RGB base elsewhere",
            "background_preservation": "Outside ROI boxes the pre-encode frames equal base_image exactly; lossy H.264 encoding can change decoded pixel values.",
            "regions": config["regions"], "config": str(config_path), "config_sha256": file_hash(config_path),
            "sources": [{"path": str(frame["image"]), "time": frame["time"], "transition": frame["transition"], "sha256": file_hash(frame["image"])} for frame in config["frames"]],
            "base_sha256": file_hash(config["base_image"]), "ffmpeg": str(ffmpeg), "ffprobe": str(ffprobe),
            "coverage_diagnostics": coverage_report(config), "visual_quality_verified": False,
            "review_note": "Inspect decoded head/hair, hands, phone and ROI borders before delivery. This metadata verifies encoding properties, not anatomy or absence of ghosting."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="Local-motion configuration JSON")
    parser.add_argument("--ffmpeg", help="FFmpeg executable; otherwise env/PATH/workspace lookup")
    parser.add_argument("--ffprobe", help="FFprobe executable; otherwise env/PATH/workspace lookup")
    parser.add_argument("--work-dir", type=Path, help="Retain interpolation files in a new directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output video")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print its plan without rendering")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF 0..51 (default 18)")
    args = parser.parse_args()
    try:
        path = args.config.expanduser().resolve()
        config = read_configuration(path)
        ffmpeg = resolve_tool("ffmpeg", args.ffmpeg, path.parent)
        ffprobe = resolve_tool("ffprobe", args.ffprobe, path.parent)
        if not 0 <= args.crf <= 51:
            raise ValueError("--crf must be 0..51")
        if args.dry_run:
            print(json.dumps({**config, "ffmpeg": ffmpeg, "ffprobe": ffprobe}, default=str, ensure_ascii=False, indent=2))
            return 0
        if config["output"].exists() and not args.overwrite:
            raise ValueError(f"Output already exists: {config['output']}. Use --overwrite to replace it.")
        with ExitStack() as stack:
            if args.work_dir:
                work = args.work_dir.expanduser().resolve()
                work.mkdir(parents=True, exist_ok=False)
            else:
                work = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="local-motion-")))
            report = render(config, path, ffmpeg, ffprobe, work, args.crf)
            report_path = config["output"].with_suffix(".render.json")
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(config["output"]), "report": str(report_path),
                              "frame_count": report["frame_count"], "duration": report["duration"],
                              "manual_visual_review_required": True}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
