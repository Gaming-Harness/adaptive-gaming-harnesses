"""Create the six green transparent Harness Edit icons."""

from pathlib import Path
import zipfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent / "green_harness_edits"
GREEN = (56, 134, 22, 255)  # BGRA -> RGB #168638


def canvas():
    return np.zeros((256, 256, 4), np.uint8)


def camera():
    im = canvas()
    cv2.rectangle(im, (48, 78), (208, 184), GREEN, -1, cv2.LINE_AA)
    cv2.rectangle(im, (82, 60), (134, 88), GREEN, -1, cv2.LINE_AA)
    cv2.circle(im, (128, 131), 39, (0, 0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(im, (128, 131), 25, GREEN, 8, cv2.LINE_AA)
    cv2.circle(im, (182, 101), 8, (0, 0, 0, 0), -1, cv2.LINE_AA)
    return im


def crossed_swords():
    im = canvas()
    cv2.line(im, (66, 57), (190, 188), GREEN, 14, cv2.LINE_AA)
    cv2.line(im, (190, 57), (66, 188), GREEN, 14, cv2.LINE_AA)
    cv2.line(im, (59, 181), (85, 207), GREEN, 12, cv2.LINE_AA)
    cv2.line(im, (197, 181), (171, 207), GREEN, 12, cv2.LINE_AA)
    cv2.line(im, (51, 168), (78, 195), GREEN, 9, cv2.LINE_AA)
    cv2.line(im, (205, 168), (178, 195), GREEN, 9, cv2.LINE_AA)
    return im


def recovery():
    im = canvas()
    pts = cv2.ellipse2Poly((128, 132), (68, 68), 0, 35, 325, 3)
    cv2.polylines(im, [pts[:-7]], False, GREEN, 12, cv2.LINE_AA)
    cv2.arrowedLine(im, tuple(pts[-9]), tuple(pts[-1]), GREEN, 12,
                    cv2.LINE_AA, tipLength=.72)
    return im


def pickaxe():
    im = canvas()
    cv2.line(im, (78, 198), (158, 89), GREEN, 16, cv2.LINE_AA)
    curve = np.array([[63, 73], [93, 54], [134, 51], [177, 69], [197, 91],
                      [167, 79], [133, 77], [101, 83]], np.int32)
    cv2.polylines(im, [curve], False, GREEN, 13, cv2.LINE_AA)
    return im


def success_check():
    im = canvas()
    cv2.rectangle(im, (58, 58), (198, 198), GREEN, 11, cv2.LINE_AA)
    cv2.line(im, (88, 130), (117, 159), GREEN, 14, cv2.LINE_AA)
    cv2.line(im, (117, 159), (174, 98), GREEN, 14, cv2.LINE_AA)
    return im


def open_book():
    im = canvas()
    left = np.array([[42, 67], [79, 61], [119, 78], [119, 193],
                     [80, 174], [42, 180]], np.int32)
    right = np.array([[137, 78], [177, 61], [214, 67], [214, 180],
                      [176, 174], [137, 193]], np.int32)
    cv2.polylines(im, [left], True, GREEN, 11, cv2.LINE_AA)
    cv2.polylines(im, [right], True, GREEN, 11, cv2.LINE_AA)
    cv2.line(im, (128, 78), (128, 198), GREEN, 10, cv2.LINE_AA)
    cv2.line(im, (52, 194), (119, 204), GREEN, 9, cv2.LINE_AA)
    cv2.line(im, (137, 204), (204, 194), GREEN, 9, cv2.LINE_AA)
    return im


def main():
    ROOT.mkdir(exist_ok=True)
    icons = {
        "01_camera_alignment_controller.png": camera(),
        "02_hold_attack_policy.png": crossed_swords(),
        "03_recovery_loop_breaker.png": recovery(),
        "04_tool_selection_rule.png": pickaxe(),
        "05_success_verifier.png": success_check(),
        "06_task_knowledge_retrieval.png": open_book(),
    }
    for name, image in icons.items():
        cv2.imwrite(str(ROOT / name), image)
    with zipfile.ZipFile(ROOT / "green_harness_edit_icons.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for name in icons:
            zf.write(ROOT / name, name)
    print(f"wrote {len(icons)} green harness-edit icons")


if __name__ == "__main__":
    main()
