"""Export each condition of strict Bare/Auto/Probing replays as its own MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def export_video(frame_dir: Path, output: Path, fps: float = 12.0) -> None:
    frames = sorted(frame_dir.glob("step_*.png"))
    if not frames:
        raise FileNotFoundError(frame_dir)
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot decode {frames[0]}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open {output}")
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Cannot decode {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = []
    for summary_path in sorted(args.root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if not summary.get("strict_001"):
            continue
        case_dir = summary_path.parent
        for condition in ("bare", "auto", "probing"):
            output = args.output_dir / f"{case_dir.name}_{condition}.mp4"
            frame_roots = sorted((case_dir / condition / "sandbox_frames").glob("task_*"))
            if not frame_roots:
                raise FileNotFoundError(case_dir / condition / "sandbox_frames")
            export_video(frame_roots[-1], output)
            manifest.append({"case": case_dir.name, "condition": condition, "video": str(output)})
            print(output, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
