"""Create transparent Only-Turns and Stalls/Hourglass failure icons."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
COLORS = {
    "red": (50, 30, 197, 255),
    "black": (31, 24, 20, 255),
    "blue": (199, 89, 21, 255),
    "green": (56, 134, 22, 255),
    "orange": (0, 122, 230, 255),
    "gray": (111, 99, 89, 255),
}


def circular_arrow(color, clockwise=True):
    img = np.zeros((256, 256, 4), np.uint8)
    center = np.array([128.0, 132.0])
    radius = 66.0
    if clockwise:
        angles = np.deg2rad(np.linspace(55, 338, 150))
    else:
        angles = np.deg2rad(np.linspace(125, -158, 150))
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ]).astype(np.int32)
    # Stop the circular stroke before its endpoint, then draw a dedicated final
    # arrow segment. This makes the arrowhead unambiguously the path endpoint.
    cv2.polylines(img, [pts[:-16]], False, color, 12, cv2.LINE_AA)
    cv2.arrowedLine(img, tuple(pts[-18]), tuple(pts[-1]), color, 12,
                    cv2.LINE_AA, tipLength=0.72)
    return img


def no_completion_check(color):
    img = np.zeros((256, 256, 4), np.uint8)
    cv2.rectangle(img, (62, 62), (194, 194), color, 11, cv2.LINE_AA)
    cv2.line(img, (92, 92), (164, 164), color, 12, cv2.LINE_AA)
    cv2.line(img, (164, 92), (92, 164), color, 12, cv2.LINE_AA)
    return img


def hourglass(color):
    img = np.zeros((256, 256, 4), np.uint8)
    cv2.line(img, (72, 48), (184, 48), color, 12, cv2.LINE_AA)
    cv2.line(img, (72, 208), (184, 208), color, 12, cv2.LINE_AA)
    cv2.line(img, (84, 54), (172, 202), color, 10, cv2.LINE_AA)
    cv2.line(img, (172, 54), (84, 202), color, 10, cv2.LINE_AA)
    # Small sand piles keep the icon legible at slide scale.
    cv2.fillConvexPoly(img, np.array([[101, 75], [155, 75], [128, 116]]), color, cv2.LINE_AA)
    cv2.fillConvexPoly(img, np.array([[96, 183], [160, 183], [128, 139]]), color, cv2.LINE_AA)
    return img


def main():
    for name, color in COLORS.items():
        out = ROOT / name
        out.mkdir(exist_ok=True)
        cv2.imwrite(str(out / "17_only_turns_clockwise_endpoint_v3.png"), circular_arrow(color, True))
        cv2.imwrite(str(out / "17_only_turns_counterclockwise_v2.png"), circular_arrow(color, False))
        cv2.imwrite(str(out / "18_stalls_hourglass.png"), hourglass(color))
        cv2.imwrite(str(out / "19_no_completion_check.png"), no_completion_check(color))
    cv2.imwrite(str(ROOT / "17_only_turns_clockwise_endpoint_v3.png"), circular_arrow(COLORS["red"], True))
    cv2.imwrite(str(ROOT / "17_only_turns_counterclockwise_v2.png"), circular_arrow(COLORS["red"], False))
    cv2.imwrite(str(ROOT / "18_stalls_hourglass.png"), hourglass(COLORS["red"]))
    cv2.imwrite(str(ROOT / "19_no_completion_check.png"), no_completion_check(COLORS["red"]))
    print("wrote failure-mode icons")


if __name__ == "__main__":
    main()
