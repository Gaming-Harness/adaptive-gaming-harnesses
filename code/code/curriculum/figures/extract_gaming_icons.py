"""Remove baked checkerboard background and split the 4x4 gaming icon sheet."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent / "gaming_icon_assets"
SOURCE = ROOT / "gaming_icons_sprite_4x4.png"

NAMES = [
    "01_game_monitor_target",
    "02_game_controller",
    "03_perception_action_trajectory",
    "04_active_probe_verified",
    "05_target_alignment",
    "06_reachability_path",
    "07_attack_persistence",
    "08_tool_inventory",
    "09_success_verification",
    "10_task_knowledge",
    "11_probe_crosshair",
    "12_memory_database",
    "13_action_middleware",
    "14_frozen_gaming_model",
    "15_harness_proposer",
    "16_camera_alignment",
]


def transparent_sheet() -> Image.Image:
    bgr = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(SOURCE)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # The generated checkerboard is light and nearly achromatic. Identify only
    # components connected to the canvas boundary so enclosed white controller
    # details remain opaque.
    candidate = ((hsv[:, :, 1] < 48) & (hsv[:, :, 2] > 165)).astype(np.uint8)
    n, labels = cv2.connectedComponents(candidate, connectivity=8)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    background = np.isin(labels, list(border_labels)) & (candidate > 0)

    # Also clear tiny isolated checker components, while preserving large enclosed
    # white details such as buttons and checkmarks.
    for label in range(1, n):
        area = int(np.count_nonzero(labels == label))
        if area < 36:
            background |= labels == label

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = np.where(background, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba)


def main() -> None:
    sheet = transparent_sheet()
    sheet.save(ROOT / "gaming_icons_sprite_4x4_transparent.png")
    width, height = sheet.size
    for index, name in enumerate(NAMES):
        row, col = divmod(index, 4)
        x0, x1 = round(col * width / 4), round((col + 1) * width / 4)
        y0, y1 = round(row * height / 4), round((row + 1) * height / 4)
        icon = sheet.crop((x0, y0, x1, y1))
        alpha = icon.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            raise RuntimeError(f"empty icon cell: {name}")
        icon = icon.crop(bbox)
        padded = Image.new("RGBA", (icon.width + 48, icon.height + 48), (0, 0, 0, 0))
        padded.alpha_composite(icon, (24, 24))
        padded.save(ROOT / f"{name}.png")
    print(f"wrote {len(NAMES)} icons to {ROOT}")


if __name__ == "__main__":
    main()
