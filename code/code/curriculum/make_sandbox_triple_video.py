#!/usr/bin/env python3
"""Build a title-free three-panel video from a saved Sandbox triple replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CONDITIONS = ("bare", "auto", "probing")
LABELS = {
    "bare": "BARE",
    "auto": "AUTO-HARNESS",
    "probing": "AUTO + PROBING",
}


def frame_paths(condition_dir: Path) -> list[Path]:
    frame_dirs = sorted((condition_dir / "sandbox_frames").glob("task_*"))
    if len(frame_dirs) != 1:
        raise RuntimeError(
            f"expected one Sandbox frame directory under {condition_dir}, "
            f"found {len(frame_dirs)}"
        )
    paths = sorted(frame_dirs[0].glob("step_*.png"))
    if not paths:
        raise RuntimeError(f"no frames under {frame_dirs[0]}")
    return paths


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized = cv2.resize(
        image,
        (max(1, round(src_w * scale)), max(1, round(src_h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def overlay_label(panel: np.ndarray, label: str, success: bool) -> None:
    status = "SUCCESS" if success else "FAIL"
    color = (70, 210, 90) if success else (70, 90, 235)
    cv2.rectangle(panel, (10, 10), (285, 66), (15, 15, 15), -1)
    cv2.putText(panel, label, (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, status, (20, 57), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, color, 2, cv2.LINE_AA)


def build(case_dir: Path, output: Path, fps: float = 12.0) -> None:
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    rows = {name: frame_paths(case_dir / name) for name in CONDITIONS}
    panel_w, panel_h = 480, 360
    max_frames = max(len(paths) for paths in rows.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (panel_w * len(CONDITIONS), panel_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {output}")
    try:
        for index in range(max_frames):
            panels = []
            for condition in CONDITIONS:
                paths = rows[condition]
                path = paths[min(index, len(paths) - 1)]
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"failed to decode {path}")
                panel = fit(image, panel_w, panel_h)
                overlay_label(
                    panel,
                    LABELS[condition],
                    bool(summary["conditions"][condition]["success"]),
                )
                panels.append(panel)
            writer.write(cv2.hconcat(panels))
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()
    build(args.case_dir, args.output, args.fps)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
