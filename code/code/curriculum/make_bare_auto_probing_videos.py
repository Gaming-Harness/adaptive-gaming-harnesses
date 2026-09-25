"""Encode strict 0/0/1 Sandbox replays as three-panel MP4 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _frame_dir(condition_dir: Path) -> Path:
    dirs = sorted((condition_dir / "sandbox_frames").glob("task_*"))
    if not dirs:
        raise FileNotFoundError(f"no Sandbox frames under {condition_dir}")
    return dirs[-1]


def _frames(condition_dir: Path) -> list[Path]:
    rows = sorted(_frame_dir(condition_dir).glob("step_*.png"))
    if not rows:
        raise FileNotFoundError(condition_dir)
    return rows


def _fit(frame, width: int, height: int):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.copyMakeBorder(
        resized, 0, height - resized.shape[0], 0, width - resized.shape[1],
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def build(case_dir: Path, output: Path) -> None:
    summary = json.loads((case_dir / "summary.json").read_text())
    if not summary.get("strict_001"):
        raise ValueError(f"not a strict bare=0 auto=0 probing=1 case: {case_dir}")
    names = ("bare", "auto", "probing")
    rows = {name: _frames(case_dir / name) for name in names}
    panel_w, panel_h, header_h, footer_h = 480, 360, 82, 82
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 12.0,
        (panel_w * 3, header_h + panel_h + footer_h),
    )
    if not writer.isOpened():
        raise RuntimeError(output)
    labels = ("BARE", "AUTO-HARNESS", "PROBING-EVOLVED")
    colors = ((90, 190, 255), (120, 200, 255), (100, 240, 140))
    max_n = max(len(v) for v in rows.values())
    try:
        for i in range(max_n):
            panels = []
            for name in names:
                frame = cv2.imread(str(rows[name][min(i, len(rows[name]) - 1)]))
                if frame is None:
                    raise RuntimeError(f"cannot decode frame for {name}")
                panels.append(_fit(frame, panel_w, panel_h))
            body = cv2.hconcat(panels)
            canvas = cv2.copyMakeBorder(
                body, header_h, footer_h, 0, 0,
                cv2.BORDER_CONSTANT, value=(23, 23, 23),
            )
            cv2.putText(
                canvas, f'{summary["name"]} | identical Sandbox seed {summary["seed"]}',
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
            )
            for j, (label, color, name) in enumerate(zip(labels, colors, names)):
                x = j * panel_w + 18
                result = summary["conditions"][name]
                cv2.putText(canvas, label, (x, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2)
                cv2.putText(
                    canvas,
                    f'{"SUCCESS" if result["success"] else "FAIL"} steps={result["steps"]}',
                    (x + 205, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1,
                )
                if j:
                    cv2.line(canvas, (j * panel_w, header_h),
                             (j * panel_w, header_h + panel_h), (255, 255, 255), 2)
            y = header_h + panel_h
            cv2.putText(
                canvas, "STRICT RESULT: bare FAIL | ordinary auto FAIL | probing-evolved SUCCESS",
                (20, y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 245, 155), 2,
            )
            cv2.putText(
                canvas, "Frozen VLA; same task/seed/temperature; independent memory; no bandit writeback",
                (20, y + 63), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
            )
            writer.write(canvas)
    finally:
        writer.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    made = []
    for summary_path in sorted(args.root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if not summary.get("strict_001"):
            continue
        case_dir = summary_path.parent
        output = args.output_dir / f'{case_dir.name}_bare0_auto0_probing1.mp4'
        build(case_dir, output)
        made.append(str(output))
        print(output, flush=True)
    manifest = {"n_strict_videos": len(made), "videos": made}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
