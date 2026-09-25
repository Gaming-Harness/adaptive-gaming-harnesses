"""Build side-by-side qualitative harness comparison videos from saved frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def _frames(path: Path) -> list[Path]:
    rows = sorted(path.glob("step_*.png"))
    if not rows:
        raise FileNotFoundError(f"no step frames in {path}")
    return rows


def _fit(frame, width: int, height: int):
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    return cv2.copyMakeBorder(
        resized, 0, height - resized.shape[0], 0, width - resized.shape[1],
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def build(left_dir: Path, right_dir: Path, output: Path, *, left_label: str,
          right_label: str, task_label: str, left_result: str,
          right_result: str, left_end_label: str = "",
          right_end_label: str = "", fps: float = 12.0) -> None:
    left, right = _frames(left_dir), _frames(right_dir)
    panel_h, panel_w, header_h = 480, 640, 76
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (panel_w * 2, panel_h + header_h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {output}")
    try:
        for i in range(max(len(left), len(right))):
            lf = cv2.imread(str(left[min(i, len(left) - 1)]))
            rf = cv2.imread(str(right[min(i, len(right) - 1)]))
            if lf is None or rf is None:
                raise RuntimeError(f"failed to decode frame {i}")
            body = cv2.hconcat([_fit(lf, panel_w, panel_h), _fit(rf, panel_w, panel_h)])
            header = cv2.copyMakeBorder(body[:1], header_h - 1, 0, 0, 0,
                                        cv2.BORDER_CONSTANT, value=(24, 24, 24))
            frame = cv2.vconcat([header, body])
            cv2.putText(frame, task_label, (20, 27), cv2.FONT_HERSHEY_SIMPLEX,
                        0.68, (255, 255, 255), 2)
            cv2.putText(frame, left_label, (20, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72, (90, 190, 255), 2)
            cv2.putText(frame, left_result, (250, 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.56, (210, 210, 210), 1)
            cv2.putText(frame, right_label, (panel_w + 20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 230, 130), 2)
            cv2.putText(frame, right_result, (panel_w + 280, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (210, 210, 210), 1)
            cv2.line(frame, (panel_w, header_h), (panel_w, panel_h + header_h),
                     (255, 255, 255), 2)
            if i >= len(left) and left_end_label:
                overlay = frame[header_h:, :panel_w]
                shade = overlay.copy()
                shade[:] = (20, 20, 20)
                cv2.addWeighted(shade, 0.62, overlay, 0.38, 0, overlay)
                cv2.putText(overlay, left_end_label, (55, panel_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.95, (110, 220, 255), 3)
            if i >= len(right) and right_end_label:
                overlay = frame[header_h:, panel_w:]
                shade = overlay.copy()
                shade[:] = (15, 70, 25)
                cv2.addWeighted(shade, 0.62, overlay, 0.38, 0, overlay)
                cv2.putText(overlay, right_end_label, (65, panel_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.82, (120, 255, 150), 3)
                cv2.putText(overlay, "Environment returned reward=1.0",
                            (95, panel_h // 2 + 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (230, 255, 235), 2)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    repo = parser.parse_args().repo.resolve()
    out = repo / "curriculum/outputs/comparison_videos"
    build(
        repo / "curriculum/outputs/openha811_bare_mtv1/shard_g4/images/task_0_101_1787310440",
        repo / "curriculum/outputs/openha811_static_harness_mtv1/shard_g4/images/task_0_101_1787315898",
        out / "knowledge_bare_vs_fixed_spruce_leaves.mp4",
        left_label="BARE", right_label="FIXED HARNESS",
        task_label="mine_block:spruce_leaves | episode seed 101",
        left_result="FAIL | 100 steps | reward 2.50",
        right_result="SUCCESS | 16 steps | reward 1.15",
        right_end_label="SUCCESS - BLOCK MINED")
    build(
        repo / "curriculum/outputs/auto_harness_hard_k1_t0_legacy_actionable/search_0002_mine_block_netherrack/images/task_0_909_1786976281",
        repo / "curriculum/outputs/auto_harness_hard_k1_t0_legacy_actionable/search_0002_mine_block_netherrack/images/task_0_909_1786975864",
        out / "auto_h0_vs_hstar_netherrack_holdout909.mp4",
        left_label="H0 / SEED HARNESS", right_label="AUTO-HARNESS H*",
        task_label="mine_block:netherrack | holdout seed 909",
        left_result="FAIL | 100 steps | reward 0.40",
        right_result="SUCCESS | 21 steps | reward 1.15",
        right_end_label="SUCCESS - EPISODE TERMINATED")


if __name__ == "__main__":
    main()
