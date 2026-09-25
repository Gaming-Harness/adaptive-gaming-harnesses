"""Create consistent color variants of transparent gaming method icons."""

from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent / "gaming_icon_assets"
COLORS = {
    "black": (20, 24, 31),
    "blue": (21, 89, 199),
    "green": (22, 134, 56),
    "red": (197, 30, 50),
    "orange": (230, 122, 0),
    "gray": (89, 99, 111),
}


def recolor(source: Path, color: tuple[int, int, int]) -> Image.Image:
    rgba = np.array(Image.open(source).convert("RGBA"))
    visible = rgba[:, :, 3] > 0
    # Preserve intentional near-white controller buttons and check centers.
    white_detail = visible & np.all(rgba[:, :, :3] >= 238, axis=2)
    recolor_mask = visible & ~white_detail
    rgba[recolor_mask, 0] = color[0]
    rgba[recolor_mask, 1] = color[1]
    rgba[recolor_mask, 2] = color[2]
    return Image.fromarray(rgba, "RGBA")


def main() -> None:
    sources = sorted(ROOT.glob("[0-9][0-9]_*.png"))
    if len(sources) != 16:
        raise RuntimeError(f"expected 16 source icons, found {len(sources)}")
    written: list[Path] = []
    for name, color in COLORS.items():
        out_dir = ROOT / name
        out_dir.mkdir(exist_ok=True)
        for source in sources:
            target = out_dir / source.name
            recolor(source, color).save(target)
            written.append(target)

    archive = ROOT / "gaming_icon_assets_6colors.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in written:
            zf.write(path, path.relative_to(ROOT))
    print(f"wrote {len(written)} variants and {archive}")


if __name__ == "__main__":
    main()
