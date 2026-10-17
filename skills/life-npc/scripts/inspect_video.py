#!/usr/bin/env python3
"""Decode exact video frames and ROI crops for manual inspection; no visual pass claim.

Example: python inspect_video.py clip.mp4 --output-dir review --times 0,.5,1,2.9
Add --config motion.json to inherit motion region boxes, or repeat
--roi head:300,400,400,450. Exports metadata, decoded PNGs and labelled crop sheets.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

from resolve_runtime import resolve_tool
from render_local_motion import probe_video


def sheet(items: list[tuple[str, Path]], output: Path) -> None:
    thumb_width, thumb_height = 460, 440
    columns = min(3, len(items))
    rows = (len(items)+columns-1)//columns
    canvas = Image.new("RGB", (columns*thumb_width, rows*(thumb_height+28)), "#202020")
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(items):
        with Image.open(path) as original:
            image = original.convert("RGB")
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns)*thumb_width
        y = (index // columns)*(thumb_height+28)
        draw.text((x+8, y+6), label, fill="white")
        canvas.paste(image, (x+(thumb_width-image.width)//2, y+28))
    canvas.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--times", help="Comma-separated seconds; rounded to exact decoded frame indices")
    parser.add_argument("--config", type=Path, help="Inherit regions from local-motion JSON")
    parser.add_argument("--roi", action="append", default=[], help="name:x,y,width,height; repeat as needed")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args()
    try:
        video = args.video.expanduser().resolve()
        if not video.is_file():
            raise ValueError(f"Video does not exist: {video}")
        output = args.output_dir.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        ffmpeg = resolve_tool("ffmpeg", args.ffmpeg, video.parent)
        ffprobe = resolve_tool("ffprobe", args.ffprobe, video.parent)
        probe = probe_video(video, ffprobe)
        if not probe.get("streams"):
            raise ValueError("No video stream found")
        stream = probe["streams"][0]
        fps = Fraction(stream["avg_frame_rate"])
        count = int(stream.get("nb_read_frames", stream.get("nb_frames", 0)))
        if fps <= 0 or count <= 0:
            raise ValueError("Cannot determine video frame count/rate")
        width, height = int(stream["width"]), int(stream["height"])
        if args.times:
            times = [float(part.strip()) for part in args.times.split(",")]
            if any(time < 0 or time >= count/float(fps) for time in times):
                raise ValueError("Every requested time must be >= 0 and before the end of the video")
            indices = sorted(set(min(count-1, max(0, round(time*float(fps)))) for time in times))
        else:
            indices = sorted(set(round((count-1)*number/8) for number in range(9)))
        regions = []
        if args.config:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            regions.extend(config.get("regions", []))
        for spec in args.roi:
            name, values = spec.split(":", 1)
            regions.append({"name": name, "box": [int(value) for value in values.split(",")]})
        for region in regions:
            box = region.get("box", [])
            if len(box) != 4 or any(isinstance(n, bool) or not isinstance(n, int) for n in box):
                raise ValueError(f"Invalid ROI: {region}")
            x, y, w, h = box
            if min(x,y) < 0 or min(w,h) <= 0 or x+w > width or y+h > height:
                raise ValueError(f"ROI outside decoded frame: {region}")
        expression = "+".join(f"eq(n\\,{number})" for number in indices)
        command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                   "-vf", f"select={expression}", "-vsync", "0", "-start_number", "0", str(output/"selected-%04d.png")]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr[-5000:])
        frames, frame_items = [], []
        crop_items = {index: [] for index in range(len(regions))}
        for sample, number in enumerate(indices):
            staged = output/f"selected-{sample:04d}.png"
            if not staged.is_file():
                raise RuntimeError(f"Decoder omitted selected frame {number}")
            target = output/f"frame-{number:06d}.png"
            staged.replace(target)
            seconds = number/float(fps)
            frames.append({"frame_index": number, "cfr_time_seconds": seconds, "image": str(target)})
            frame_items.append((f"frame {number} | {seconds:.3f}s", target))
            with Image.open(target) as image:
                for index, region in enumerate(regions):
                    x,y,w,h = region["box"]
                    crop = output/f"roi-{index:02d}-frame-{number:06d}.png"
                    image.crop((x,y,x+w,y+h)).save(crop)
                    crop_items[index].append((f"ROI {index} | f{number} | {seconds:.3f}s", crop))
        sheet(frame_items, output/"frame-sheet.png")
        for index, items in crop_items.items():
            sheet(items, output/f"roi-{index:02d}-sheet.png")
        report = {"video": str(video), "width": width, "height": height, "frame_count": count,
                  "average_fps": str(fps), "duration_seconds": float(stream.get("duration", probe["format"].get("duration", 0))),
                  "cfr_time_note": "Labels assume constant frame rate. Frame indices are exact decoded indices; variable-rate sources require timestamp inspection.",
                  "samples": frames, "regions": regions, "ffmpeg": str(ffmpeg), "ffprobe": str(ffprobe),
                  "manual_review_required": True, "visual_quality_verified": False,
                  "review_targets": ["complete hair boundary against source", "hands and fingers", "feet if visible", "ROI border seams", "ghosting", "scene/action continuity"],
                  "observations": []}
        metadata_path = output/"inspection.json"
        metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({"inspection": str(metadata_path), "frame_sheet": str(output/"frame-sheet.png"),
                          "sample_count": len(frames), "frame_count": count,
                          "manual_review_required": True}, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
