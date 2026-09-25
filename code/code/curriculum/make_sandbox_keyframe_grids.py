#!/usr/bin/env python3
"""Create publication-ready 4x4 keyframe grids from Sandbox trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CONDITIONS = ("bare", "auto", "probing")
DISPLAY_NAMES = {
    "bare": "Bare",
    "auto": "Auto-Harness",
    "probing": "Auto + Probing Evolution",
}


def find_frame_dir(condition_dir: Path) -> Path:
    candidates = sorted((condition_dir / "sandbox_frames").glob("task_*"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one task frame directory under {condition_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def select_uniformly(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        raise RuntimeError("trajectory contains no PNG frames")
    if len(paths) <= count:
        return paths
    indices = np.rint(np.linspace(0, len(paths) - 1, count)).astype(int)
    return [paths[index] for index in indices]


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized = cv2.resize(
        image,
        (max(1, round(src_w * scale)), max(1, round(src_h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def build_grid(
    case_dir: Path,
    condition: str,
    output: Path,
    *,
    count: int,
) -> dict:
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    result = summary["conditions"][condition]
    frame_dir = find_frame_dir(case_dir / condition)
    all_frames = sorted(frame_dir.glob("step_*.png"))
    selected = select_uniformly(all_frames, count)

    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    image_w, image_h = 480, 270
    caption_h, header_h, gap = 38, 104, 8
    tile_h = image_h + caption_h
    canvas_w = columns * image_w + (columns - 1) * gap
    canvas_h = header_h + rows * tile_h + (rows - 1) * gap
    canvas = np.full((canvas_h, canvas_w, 3), 247, dtype=np.uint8)

    status = "SUCCESS" if result["success"] else "FAIL"
    status_color = (40, 145, 45) if result["success"] else (45, 45, 195)
    title = (
        f'Sandbox trajectory: {summary["name"]} | seed {summary["seed"]} | '
        f'{DISPLAY_NAMES[condition]}'
    )
    subtitle = (
        f'{status} | agent steps={result["steps"]} | reward={result["reward"]:.3g} | '
        f'saved environment frames={len(all_frames)}'
    )
    cv2.putText(canvas, title, (16, 37), cv2.FONT_HERSHEY_SIMPLEX,
                0.86, (28, 28, 28), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (16, 76), cv2.FONT_HERSHEY_SIMPLEX,
                0.70, status_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "16 uniformly sampled frames spanning the full trajectory",
                (16, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (90, 90, 90), 1, cv2.LINE_AA)

    selected_rows = []
    for position, path in enumerate(selected):
        row, column = divmod(position, columns)
        x = column * (image_w + gap)
        y = header_h + row * (tile_h + gap)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to decode {path}")
        canvas[y:y + image_h, x:x + image_w] = fit_image(image, image_w, image_h)
        frame_index = int(path.stem.split("_")[-1])
        progress = 100.0 * position / max(1, len(selected) - 1)
        caption = f'{position + 1:02d}/16   {path.stem}   trajectory {progress:5.1f}%'
        cv2.putText(canvas, caption, (x + 10, y + image_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45),
                    1, cv2.LINE_AA)
        selected_rows.append(
            {
                "position": position + 1,
                "frame_index": frame_index,
                "file": str(path),
                "trajectory_fraction": position / max(1, len(selected) - 1),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError(f"failed to write {output}")
    return {
        "case": case_dir.name,
        "task": summary["name"],
        "seed": summary["seed"],
        "condition": condition,
        "display_name": DISPLAY_NAMES[condition],
        "success": result["success"],
        "agent_steps": result["steps"],
        "reward": result["reward"],
        "total_saved_frames": len(all_frames),
        "sampling": "uniform over the complete saved-frame trajectory, endpoints included",
        "output": str(output),
        "selected_frames": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    records = []
    for case_dir in args.case_dirs:
        for condition in CONDITIONS:
            output = args.output_dir / f"{case_dir.name}_{condition}_{args.count}frames.png"
            record = build_grid(case_dir, condition, output, count=args.count)
            records.append(record)
            print(output, flush=True)

    manifest = {
        "n_images": len(records),
        "frames_per_image": args.count,
        "images": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
