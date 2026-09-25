"""Build paired videos for frozen merge-harness rescues.

The footer deliberately reports episode-level counters only.  The saved OpenHA
environment log does not contain the exact logical step of each harness
decision, so the video must not pretend to place rollback markers on frames.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2


TASKS = {
    "mine_block:sugar_cane",
    "mine_block:cocoa",
    "kill_entity:slime",
    "mine_block:spruce_leaves",
    "kill_entity:vindicator",
    "mine_block:cactus",
    "mine_block:chiseled_sandstone",
    "mine_block:diorite",
    "mine_block:sunflower",
    "mine_block:kelp_plant",
}


def load(root: Path) -> dict[tuple[str, int], tuple[dict, Path]]:
    out = {}
    for result_path in root.glob("shard_g*/results.jsonl"):
        image_dirs = []
        for path in (result_path.parent / "images").glob("task_*"):
            try:
                image_dirs.append((int(path.name.rsplit("_", 1)[1]), path))
            except ValueError:
                pass
        image_dirs.sort()
        for line in result_path.read_text().splitlines():
            row = json.loads(line)
            candidates = [item for item in image_dirs if item[0] <= row["timestamp"]]
            if candidates:
                out[(row["file"], row["episode_seed"])] = (row, candidates[-1][1])
    return out


def frames(path: Path) -> list[Path]:
    rows = sorted(path.glob("step_*.png"))
    if not rows:
        raise FileNotFoundError(path)
    return rows


def fit(frame, width: int, height: int):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.copyMakeBorder(
        resized, 0, height - resized.shape[0], 0, width - resized.shape[1],
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def build(bare_dir: Path, merge_dir: Path, output: Path, bare: dict, merge: dict) -> None:
    left, right = frames(bare_dir), frames(merge_dir)
    panel_w, panel_h, header_h, footer_h = 640, 480, 78, 94
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 12.0,
        (panel_w * 2, header_h + panel_h + footer_h),
    )
    if not writer.isOpened():
        raise RuntimeError(output)
    recover = int(merge.get("n_recover") or 0)
    probe = int(merge.get("n_probe") or 0)
    try:
        for i in range(max(len(left), len(right))):
            lf = cv2.imread(str(left[min(i, len(left) - 1)]))
            rf = cv2.imread(str(right[min(i, len(right) - 1)]))
            body = cv2.hconcat([fit(lf, panel_w, panel_h), fit(rf, panel_w, panel_h)])
            canvas = cv2.copyMakeBorder(
                body, header_h, footer_h, 0, 0, cv2.BORDER_CONSTANT, value=(23, 23, 23)
            )
            cv2.putText(canvas, f'{merge["task_name"]} | same episode seed 101', (20, 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(canvas, "BARE / frozen agent", (20, 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, (90, 190, 255), 2)
            cv2.putText(canvas, f'FAIL  steps={bare["steps"]}', (245, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
            cv2.putText(canvas, "MERGE KNOWLEDGE HARNESS", (panel_w + 20, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (100, 230, 130), 2)
            cv2.putText(canvas, f'SUCCESS  steps={merge["steps"]}', (panel_w + 350, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 210, 210), 1)
            cv2.line(canvas, (panel_w, header_h), (panel_w, header_h + panel_h),
                     (255, 255, 255), 2)
            y = header_h + panel_h
            cv2.putText(canvas, "PROMOTED KNOWLEDGE [reacquire_and_center]", (20, y + 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.61, (120, 240, 170), 2)
            cv2.putText(canvas,
                        "target lost / looking away -> face target -> center crosshair -> retry",
                        (20, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (235, 235, 235), 1)
            color = (90, 220, 255) if recover else (170, 170, 170)
            cv2.putText(canvas, f"EPISODE COUNTERS: probes={probe}  checkpoint rollbacks={recover}",
                        (20, y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.59, color, 2)
            if i >= len(right):
                overlay = canvas[header_h:header_h + panel_h, panel_w:]
                shade = overlay.copy()
                shade[:] = (15, 70, 25)
                cv2.addWeighted(shade, 0.62, overlay, 0.38, 0, overlay)
                cv2.putText(overlay, "SUCCESS", (225, panel_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 255, 150), 3)
            writer.write(canvas)
    finally:
        writer.release()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    bare_rows = load(repo / "curriculum/outputs/openha811_bare_mtv1")
    merge_rows = load(repo / "curriculum/outputs/openha811_merge_frozen_mtv1")
    output_dir = repo / "curriculum/outputs/comparison_videos/merge_knowledge_recovery_samples"
    for key, (merge, merge_dir) in merge_rows.items():
        if merge["task_name"] not in TASKS or not merge["success"] or key not in bare_rows:
            continue
        bare, bare_dir = bare_rows[key]
        if bare["success"]:
            continue
        safe_name = merge["task_name"].replace(":", "_")
        output = output_dir / f"bare_fail_vs_merge_knowledge_success_{safe_name}_seed101.mp4"
        build(bare_dir, merge_dir, output, bare, merge)
        print(output)


if __name__ == "__main__":
    main()
