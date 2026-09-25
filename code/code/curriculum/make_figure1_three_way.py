#!/usr/bin/env python3
"""Build the three-way Figure 1 and export its real Sandbox keyframes."""

from __future__ import annotations

import base64
import html
import json
import os
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import cairosvg


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "outputs/probing_triple_video_t07_seed101/sand_seed101"
OUT = ROOT / "outputs/paper_figures"
FRAME_OUT = OUT / "figure1_three_way_frames"

RUNS = {
    "bare": RUN_ROOT / "bare/sandbox_frames/task_0_101_1789674563",
    "auto": RUN_ROOT / "auto/sandbox_frames/task_0_101_1789675755",
    "ours": RUN_ROOT / "probing/sandbox_frames/task_0_101_1789677040",
}

# Chosen independently for each trajectory; these are rendered frames, not env steps.
SELECTED = {
    "bare": [0, 132, 264, 400],
    "auto": [0, 108, 216, 432],
    "ours": [0, 73, 132, 338],
}

ENV_LABELS = {
    "bare": ["t=0", "t≈31", "t≈62", "t=100"],
    "auto": ["t=0", "t≈25", "t≈50", "t=100"],
    "ours": ["t=0", "t≈17", "t≈31", "t=79"],
}

COLORS = {
    "ink": "#26313D",
    "muted": "#6B7785",
    "line": "#D7DEE7",
    "bare": "#7B8794",
    "bare_bg": "#F4F6F8",
    "auto": "#C77547",
    "auto_bg": "#FCF2EB",
    "ours": "#3D7C70",
    "ours_bg": "#ECF6F3",
    "probe": "#587FA6",
    "knowledge": "#756A91",
    "fail": "#C95454",
    "success": "#2F7D62",
}


def image_data_uri(path: Path) -> str:
    mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def rect(x, y, w, h, *, fill="#fff", stroke="none", sw=1.0, rx=10):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def text(x, y, value, *, size=16, color=None, weight=400, anchor="start", italic=False):
    color = color or COLORS["ink"]
    style = "italic" if italic else "normal"
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" font-style="{style}">{esc(value)}</text>'
    )


def multiline(x, y, lines, *, size=14, color=None, weight=400, anchor="middle", gap=18):
    color = color or COLORS["ink"]
    spans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else gap
        spans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}">' + "".join(spans) + "</text>"
    )


def arrow(x1, y1, x2, y2, color, *, dashed=False, marker="arrow"):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="2"{dash} marker-end="url(#{marker})"/>'
    )


def node(x, y, w, h, label, *, fill, stroke, sub=None):
    parts = [rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.5, rx=9)]
    cy = y + h / 2 - (6 if sub else -5)
    parts.append(text(x + w / 2, cy, label, size=14, color=stroke, weight=700, anchor="middle"))
    if sub:
        parts.append(text(x + w / 2, cy + 19, sub, size=10.5, color=COLORS["muted"], anchor="middle"))
    return "".join(parts)


def export_frames() -> dict[str, list[Path]]:
    FRAME_OUT.mkdir(parents=True, exist_ok=True)
    exported: dict[str, list[Path]] = {}
    for mode, indices in SELECTED.items():
        mode_out = FRAME_OUT / mode
        mode_out.mkdir(parents=True, exist_ok=True)
        paths = []
        for order, idx in enumerate(indices, 1):
            src = RUNS[mode] / f"step_{idx:04d}.png"
            if not src.exists():
                raise FileNotFoundError(src)
            dst = mode_out / f"{order:02d}_frame_{idx:04d}.png"
            shutil.copy2(src, dst)
            paths.append(dst)
        exported[mode] = paths
    return exported


def frame_strip(mode: str, paths: list[Path], x: float, y: float, panel_w: float) -> str:
    gap = 8
    fw = (panel_w - 36 - gap * 3) / 4
    fh = 105
    out = []
    for i, (path, lab) in enumerate(zip(paths, ENV_LABELS[mode])):
        fx = x + 18 + i * (fw + gap)
        out.append(rect(fx - 1.5, y - 1.5, fw + 3, fh + 3, fill="#FFFFFF", stroke=COLORS["line"], sw=1, rx=5))
        out.append(
            f'<image x="{fx}" y="{y}" width="{fw}" height="{fh}" '
            f'preserveAspectRatio="xMidYMid slice" href="{image_data_uri(path)}" clip-path="url(#frameclip)"/>'
        )
        out.append(text(fx + fw / 2, y + fh + 18, lab, size=10.5, color=COLORS["muted"], anchor="middle"))
        if i < 3:
            out.append(arrow(fx + fw + 2, y + fh / 2, fx + fw + gap - 2, y + fh / 2, COLORS["muted"], marker="arrowSmall"))
    return "".join(out)


def build_svg(exported: dict[str, list[Path]]) -> str:
    W, H = 1800, 690
    margin, gap = 28, 18
    pw = (W - 2 * margin - 2 * gap) / 3
    xs = [margin + i * (pw + gap) for i in range(3)]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker>',
        '<marker id="arrowSmall" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker>',
        '<clipPath id="frameclip"><rect x="0" y="0" width="10000" height="10000" rx="4"/></clipPath>',
        '</defs>',
        rect(0, 0, W, H, fill="#FFFFFF", rx=0),
        text(28, 27, "Frozen VLA, same task and seed", size=12.5, color=COLORS["muted"], weight=600),
        text(W - 28, 27, "mine_block:sand · seed 101 · real Sandbox rollouts", size=12.5, color=COLORS["muted"], weight=600, anchor="end"),
    ]

    configs = [
        ("bare", "1", "Bare", "Direct execution", COLORS["bare"], COLORS["bare_bg"]),
        ("auto", "2", "Auto-Knowledge", "Improved, but insufficient", COLORS["auto"], COLORS["auto_bg"]),
        ("ours", "3", "Knowledge–Probing Co-Evolution", "Adaptive recovery", COLORS["ours"], COLORS["ours_bg"]),
    ]

    for i, (mode, num, title, subtitle, accent, bg) in enumerate(configs):
        x = xs[i]
        svg.append(rect(x, 42, pw, 620, fill="#FFFFFF", stroke=accent, sw=1.7 if mode == "ours" else 1.25, rx=16))
        svg.append(rect(x, 42, pw, 58, fill=bg, stroke="none", rx=16))
        svg.append(f'<circle cx="{x + 27}" cy="71" r="15" fill="{accent}"/>')
        svg.append(text(x + 27, 76, num, size=14, color="#FFFFFF", weight=700, anchor="middle"))
        svg.append(text(x + 51, 68, title, size=20 if mode != "ours" else 18, color=accent, weight=700))
        svg.append(text(x + 51, 88, subtitle, size=11.5, color=COLORS["muted"], weight=500))

        # Mechanism band
        y0 = 119
        svg.append(text(x + 18, y0, "METHOD", size=10.5, color=COLORS["muted"], weight=700))
        if mode == "bare":
            svg.append(node(x + 24, y0 + 26, 120, 62, "Observation", fill=COLORS["bare_bg"], stroke=accent, sub="+ task prompt"))
            svg.append(arrow(x + 146, y0 + 57, x + 184, y0 + 57, accent))
            svg.append(node(x + 187, y0 + 26, 145, 62, "Frozen VLA", fill="#FFFFFF", stroke=COLORS["ink"], sub="direct policy"))
            svg.append(arrow(x + 334, y0 + 57, x + 372, y0 + 57, accent))
            svg.append(node(x + 375, y0 + 26, 144, 62, "Action", fill=COLORS["bare_bg"], stroke=accent, sub="no recovery"))
        elif mode == "auto":
            svg.append(node(x + 19, y0 + 26, 108, 62, "Observation", fill=COLORS["auto_bg"], stroke=accent))
            svg.append(arrow(x + 129, y0 + 57, x + 160, y0 + 57, accent))
            svg.append(node(x + 163, y0 + 26, 122, 62, "Frozen VLA", fill="#FFFFFF", stroke=COLORS["ink"] ))
            svg.append(arrow(x + 287, y0 + 57, x + 318, y0 + 57, accent))
            svg.append(node(x + 321, y0 + 26, 114, 62, "Action", fill=COLORS["auto_bg"], stroke=accent))
            svg.append(node(x + 449, y0 + 26, 101, 62, "Memory", fill="#F5F2F8", stroke=COLORS["knowledge"], sub="static prior"))
            svg.append(arrow(x + 499, y0 + 91, x + 391, y0 + 103, COLORS["knowledge"], dashed=True))
        else:
            svg.append(node(x + 18, y0 + 26, 98, 62, "Observation", fill=COLORS["ours_bg"], stroke=accent))
            svg.append(arrow(x + 118, y0 + 57, x + 143, y0 + 57, accent))
            svg.append(node(x + 146, y0 + 26, 106, 62, "Frozen VLA", fill="#FFFFFF", stroke=COLORS["ink"] ))
            svg.append(arrow(x + 254, y0 + 57, x + 279, y0 + 57, accent))
            svg.append(node(x + 282, y0 + 26, 98, 62, "Action", fill=COLORS["ours_bg"], stroke=accent))
            svg.append(node(x + 398, y0 + 16, 145, 40, "Adaptive Probe", fill="#EEF4FA", stroke=COLORS["probe"] ))
            svg.append(node(x + 398, y0 + 75, 145, 40, "Knowledge Prior", fill="#F4F1F7", stroke=COLORS["knowledge"] ))
            svg.append(arrow(x + 470, y0 + 58, x + 470, y0 + 72, COLORS["probe"], marker="arrowSmall"))
            svg.append(arrow(x + 398, y0 + 95, x + 337, y0 + 90, COLORS["knowledge"], dashed=True))
            svg.append(arrow(x + 337, y0 + 91, x + 398, y0 + 42, COLORS["probe"], dashed=True))

        # What is actually injected into the frozen VLA context.
        inject_y = y0 + 121
        svg.append(rect(x + 18, inject_y, pw - 36, 45, fill=bg, stroke="none", rx=8))
        svg.append(text(x + 30, inject_y + 15, "INJECTED CONTEXT", size=9.5, color=accent, weight=800))
        if mode == "bare":
            injected = "∅  task instruction only"
        elif mode == "auto":
            injected = "reacquire_and_center  ·  fixed prior + coverage probing"
        else:
            injected = "look_down → center(sand) → forward → continuous attack"
        svg.append(text(x + 30, inject_y + 34, injected, size=11.5, color=COLORS["ink"], weight=650))

        # Trajectory band
        traj_y = 316
        svg.append(f'<line x1="{x + 18}" y1="{traj_y - 18}" x2="{x + pw - 18}" y2="{traj_y - 18}" stroke="{COLORS["line"]}"/>')
        svg.append(text(x + 18, traj_y, "REAL SANDBOX TRAJECTORY", size=10.5, color=COLORS["muted"], weight=700))
        svg.append(frame_strip(mode, exported[mode], x, traj_y + 18, pw))

        # Evidence/result band
        ry = 476
        if mode == "bare":
            svg.append(multiline(x + 18, ry, ["Target lost", "No corrective behavior"], size=12.5, color=COLORS["muted"], weight=600, anchor="start", gap=20))
            status, stat_color = "FAIL @ 100 steps", COLORS["fail"]
            reward = "reward 0.05"
        elif mode == "auto":
            svg.append(multiline(x + 18, ry, ["Target is partially reacquired", "Recovery occurs, but mining never completes"], size=12.5, color=accent, weight=600, anchor="start", gap=20))
            status, stat_color = "FAIL @ 100 steps", COLORS["fail"]
            reward = "reward 0.20  ↑4×"
        else:
            svg.append(multiline(x + 18, ry, ["Reorient → center target → approach and act", "Successful knowledge receives delayed credit"], size=12.5, color=accent, weight=700, anchor="start", gap=20))
            status, stat_color = "SUCCESS @ 79 steps", COLORS["success"]
            reward = "reward 1.20"
        svg.append(rect(x + 18, 540, pw - 36, 86, fill=bg, stroke="none", rx=10))
        svg.append(text(x + 34, 574, status, size=19, color=stat_color, weight=800))
        svg.append(text(x + pw - 34, 574, reward, size=14, color=accent, weight=700, anchor="end"))
        bar_x, bar_y, bar_w = x + 34, 593, pw - 68
        svg.append(rect(bar_x, bar_y, bar_w, 8, fill="#E3E7EC", stroke="none", rx=4))
        progress = {"bare": 0.04, "auto": 0.17, "ours": 1.0}[mode]
        svg.append(rect(bar_x, bar_y, bar_w * progress, 8, fill=accent, stroke="none", rx=4))
        svg.append(text(bar_x, 617, "task progress", size=10.5, color=COLORS["muted"]))
        svg.append(text(bar_x + bar_w, 617, {"bare": "0/1", "auto": "0/1", "ours": "1/1"}[mode], size=10.5, color=accent, weight=700, anchor="end"))

    svg.append(text(W / 2, 681, "Knowledge alone improves behavior; co-evolving how to probe and what to reuse turns partial recovery into task success.", size=14, color=COLORS["ink"], weight=700, anchor="middle"))
    svg.append("</svg>")
    return "".join(svg)


def make_contact_sheet(mode: str, paths: list[Path]) -> None:
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((480, 300), Image.Resampling.LANCZOS)
        thumbs.append(image.copy())
    width = 4 * 480 + 3 * 18
    canvas = Image.new("RGB", (width, 365), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=18)
    for i, (image, label) in enumerate(zip(thumbs, ENV_LABELS[mode])):
        x = i * (480 + 18)
        canvas.paste(image, (x, 20))
        draw.text((x + 240, 330), label, fill="#26313D", anchor="mm", font=font)
    canvas.save(FRAME_OUT / f"{mode}_four_keyframes.png", quality=95)


def main() -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    exported = export_frames()
    for mode, paths in exported.items():
        make_contact_sheet(mode, paths)
    svg = build_svg(exported)
    svg_path = OUT / "figure1_three_way_with_injections.svg"
    png_path = OUT / "figure1_three_way_with_injections.png"
    pdf_path = OUT / "figure1_three_way_with_injections.pdf"
    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), output_width=3600, output_height=1380)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(pdf_path))
    manifest = {
        "task": "mine_block:sand",
        "seed": 101,
        "frozen_vla": True,
        "results": {
            "bare": {"success": False, "steps": 100, "reward": 0.05},
            "auto_knowledge": {"success": False, "steps": 100, "reward": 0.20},
            "ours": {"success": True, "steps": 79, "reward": 1.20},
        },
        "rendered_frame_indices": SELECTED,
        "export_directory": str(FRAME_OUT),
    }
    (OUT / "figure1_three_way_with_injections.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    injections = """Bare
  Injection: none
  Effective context: Gather sand from the desert.
  Disabled: probe, knowledge probe, memory injection, recovery

Auto-Knowledge
  Fixed prior skill: reacquire_and_center
  Injected knowledge: Previous action failed. If the target sand is not centered in your view or you are looking away, re-orient to face and center it, then retry.
  Probe selection: coverage (fixed, not learned)
  Recovery: generic rollback/replan

Ours: Knowledge-Probing Co-Evolution
  Candidate knowledge: backup_and_turn; reacquire_and_center; approach_and_act
  Selected instruction: Gather sand from the desert. Look down, center the crosshair on a sand block, and continuously attack it.
  Selected recovery: If the target sand is not centered, re-orient and center it; if stuck, look down, move slightly forward, and attack the sand block.
  Executable actions: mouseMove down; move forward; left-click attack
  Probe selection: contextual UCB
  Feedback: immediate progress/reward/novelty plus delayed task-success credit
"""
    (OUT / "figure1_three_way_injections.txt").write_text(injections, encoding="utf-8")
    print(svg_path)
    print(png_path)
    print(pdf_path)
    print(FRAME_OUT)


if __name__ == "__main__":
    main()
