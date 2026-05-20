"""Benchmark the detector and the processing pipeline on an image or video file.

Examples (from the repository root):

    python scripts/benchmark.py --source data/samples/test.jpg --frames 100
    python scripts/benchmark.py --source data/samples/video.mp4 --frames 300 --image-size 320
    python scripts/benchmark.py --source data/samples/video.mp4 --resize 320x240   # reproduces the original preprocessing

Frames are processed back to back with no display, so the numbers describe the
compute cost only. The first --warmup frames are processed but not measured.
Results are printed and written as JSON under reports/benchmarks/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402

from restricted_zone_monitor.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from restricted_zone_monitor.detector import Detector  # noqa: E402
from restricted_zone_monitor.event_manager import EventEngine, new_session_id  # noqa: E402
from restricted_zone_monitor.performance import percentile  # noqa: E402
from restricted_zone_monitor.pipeline import MonitorPipeline  # noqa: E402
from restricted_zone_monitor.video_source import FileSource, Frame  # noqa: E402
from restricted_zone_monitor.zone_manager import load_zones  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def frames_from(path: Path, count: int, resize: tuple[int, int] | None):
    """Yield up to ``count`` frames. An image is repeated; a video is read once from the start."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"Could not read image {path}")
        if resize:
            image = cv2.resize(image, resize)
        for i in range(count):
            yield Frame(image, i, i / 30.0, time.perf_counter())
        return
    with FileSource(path) as source:
        for _ in range(count):
            frame = source.read()
            if frame is None:
                return
            image = cv2.resize(frame.image, resize) if resize else frame.image
            yield Frame(image, frame.index, frame.stream_time, time.perf_counter())


def describe(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    stats = {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) >= 20:
        stats["p95"] = percentile(values, 95)
    return {k: round(v, 2) for k, v in stats.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="image or video file")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--frames", type=int, default=200, help="frames to measure (after warm-up)")
    parser.add_argument("--warmup", type=int, default=10, help="frames processed before measuring")
    parser.add_argument("--image-size", type=int, help="override model.image_size")
    parser.add_argument("--device", help="override model.device (auto/cpu/cuda/mps)")
    parser.add_argument("--resize", help="resize frames to WxH before inference, e.g. 320x240")
    parser.add_argument("--no-track", action="store_true", help="detection only (no ByteTrack, no zone events)")
    parser.add_argument("--output", type=Path, help="JSON result path (default reports/benchmarks/benchmark_<time>.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.model
    if args.image_size:
        model_cfg = replace(model_cfg, image_size=args.image_size)
    if args.device:
        model_cfg = replace(model_cfg, device=args.device)
    resize = tuple(int(v) for v in args.resize.lower().split("x")) if args.resize else None

    detector = Detector(model_cfg, cfg.tracking.tracker)
    engine = EventEngine(
        load_zones(cfg.zones.file), session_id=new_session_id(),
        entry_confirm_seconds=cfg.tracking.entry_confirm_seconds,
        exit_grace_seconds=cfg.tracking.exit_grace_seconds,
        track_timeout_seconds=cfg.tracking.track_timeout_seconds,
    )
    pipeline = MonitorPipeline(detector, engine, cfg.zones.anchor)

    inference, detect, total = [], [], []
    detections_per_frame = []
    input_size = None
    measured_start = None
    processed = 0
    for i, frame in enumerate(frames_from(args.source, args.warmup + args.frames, resize)):
        if i == args.warmup:
            measured_start = time.perf_counter()
        input_size = (frame.image.shape[1], frame.image.shape[0])
        if args.no_track:
            t0 = time.perf_counter()
            result = detector.detect(frame.image, track=False)
            elapsed = (time.perf_counter() - t0) * 1000
            inf, det, tot, n = result.inference_ms, result.call_ms, elapsed, len(result.detections)
        else:
            r = pipeline.process(frame)
            inf, det, tot, n = r.inference_ms, r.detect_ms, r.pipeline_ms, len(r.detections)
        if i >= args.warmup:
            processed += 1
            if inf is not None:
                inference.append(inf)
            detect.append(det)
            total.append(tot)
            detections_per_frame.append(n)
    if processed == 0:
        print("No frames were measured (source shorter than the warm-up?)", file=sys.stderr)
        return 1
    wall = time.perf_counter() - measured_start

    result = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": str(args.source),
        "model": detector.model_path.name,
        "device": detector.device,
        "half": detector.half,
        "image_size": model_cfg.image_size,
        "input_size": f"{input_size[0]}x{input_size[1]}",
        "mode": "detect" if args.no_track else "detect+track+zones",
        "warmup_frames": args.warmup,
        "measured_frames": processed,
        "throughput_fps": round(processed / wall, 2),
        "inference_ms": describe(inference),
        "detector_call_ms": describe(detect),
        "pipeline_ms": describe(total),
        "mean_detections_per_frame": round(statistics.fmean(detections_per_frame), 2),
        "platform": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
    }

    print(f"\nModel {result['model']} on {result['device']} | imgsz {result['image_size']} | "
          f"input {result['input_size']} | {result['mode']}")
    print(f"Measured {processed} frames after {args.warmup} warm-up frames; throughput {result['throughput_fps']} fps")
    for key in ("inference_ms", "detector_call_ms", "pipeline_ms"):
        s = result[key]
        if s:
            p95 = f"  p95 {s['p95']:.1f}" if "p95" in s else ""
            print(f"  {key:<17} mean {s['mean']:.1f}  median {s['median']:.1f}{p95}  min {s['min']:.1f}  max {s['max']:.1f}")

    output = args.output or ROOT / "reports" / "benchmarks" / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
