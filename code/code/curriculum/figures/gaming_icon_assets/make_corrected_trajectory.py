"""Draw a transparent trajectory icon with a correctly oriented middle arrow."""

from pathlib import Path
import cv2
import numpy as np


OUT = Path(__file__).resolve().parent
BLUE = (232, 100, 25, 255)  # BGRA for #1964E8


def bezier(p0, p1, p2, p3, n=80):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = map(lambda p: np.array(p, float), (p0, p1, p2, p3))
    return ((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3).astype(int)


def dashed(img, points, width=4, dash=9, gap=7):
    pts = np.asarray(points)
    for i in range(len(pts)-1):
        if (i % (dash + gap)) < dash:
            cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), BLUE, width, cv2.LINE_AA)


def arrow(img, tip, direction, size=13):
    d = np.array(direction, float); d /= np.linalg.norm(d)
    q = np.array([-d[1], d[0]])
    tip = np.array(tip, float)
    a = tip - d*size + q*size*.55
    b = tip - d*size - q*size*.55
    cv2.line(img, tuple(a.astype(int)), tuple(tip.astype(int)), BLUE, 5, cv2.LINE_AA)
    cv2.line(img, tuple(b.astype(int)), tuple(tip.astype(int)), BLUE, 5, cv2.LINE_AA)


def main():
    img = np.zeros((256, 256, 4), np.uint8)
    # Continuous progression: upper-left -> top -> upper-right -> lower-left -> right.
    top = bezier((36, 42), (78, 45), (88, 14), (124, 22))
    top2 = bezier((124, 22), (175, 24), (198, 48), (194, 78))
    middle = bezier((194, 78), (185, 125), (96, 116), (54, 166))
    bottom = bezier((54, 166), (35, 214), (120, 218), (205, 190))
    for curve in (top, top2, middle, bottom):
        dashed(img, curve)
    # Arrowheads follow the actual direction of travel.
    # First arrow tilts upward-right; later arrows continue downward.
    arrow(img, (190, 61), (0.55, -1.0), 12)
    # All trajectory arrows progress from the top toward the bottom.
    arrow(img, tuple(middle[58]), middle[58] - middle[50], 12)  # down-left
    arrow(img, (205, 190), bottom[-1] - bottom[-8], 14)
    for x, y in [(36,42), (124,22), (194,78), (54,166)]:
        cv2.circle(img, (x, y), 10, BLUE, -1, cv2.LINE_AA)
    target = OUT / "03_perception_action_trajectory_first_arrow_up_v4.png"
    cv2.imwrite(str(target), img)
    print(target)


if __name__ == "__main__":
    main()
