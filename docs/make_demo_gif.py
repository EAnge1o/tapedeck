#!/usr/bin/env python3
"""Render docs/demo.gif: the README demo, animated line by line.

Same principle as make_demo_svg.py -- the FLOW constants are real captured
output (see that file for the capture recipe), and this script only animates
their reveal. Regenerate with:  python docs/make_demo_gif.py
(Requires Pillow:  pip install pillow -- this is a docs tool, not a runtime
dependency of the library itself.)
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_demo_svg import COLORS, FLOW, LH, esc

WIDTH, HEIGHT = 780, 560
HEADER_H = 38
PAD_X, PAD_Y = 22, 16
FONT_SIZE = 15
BG, HEADER, BORDER = "#0d1117", "#161b22", "#30363d"
DPI_FONT = "consola.ttf"          # Windows; falls back to DejaVu on other OSes


def load_font() -> "ImageFont.FreeTypeFont":
    for name in (DPI_FONT, "Consolas", "DejaVuSansMono"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except OSError:
            continue
    return ImageFont.load_default()


def frame(reveal: int) -> "Image.Image":
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, WIDTH - 1, HEIGHT - 1], radius=10, fill=BG, outline=BORDER, width=2)
    d.rectangle([2, 2, WIDTH - 3, HEADER_H], fill=HEADER)
    for i, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        d.ellipse([22 + i * 22, 13, 34 + i * 22, 25], fill=color)
    d.text((WIDTH // 2, 19), "tapedeck — record · replay · diff (real output)",
           fill="#8b949e", font=font, anchor="mm")
    y = HEADER_H + PAD_Y
    for kind, text in FLOW[:reveal]:
        color = COLORS[kind]
        if kind == "cmd":
            d.text((PAD_X, y), "$", fill="#3fb950", font=font)
            d.text((PAD_X + 18, y), text, fill=color, font=font)
        else:
            d.text((PAD_X, y), text.replace("\u00a0", " "), fill=color, font=font)
        y += LH
    return img


def main() -> None:
    global font
    font = load_font()
    frames = [frame(n) for n in range(1, len(FLOW) + 1)]
    frames.append(frames[-1])                       # hold the finished terminal
    durations = [140] * (len(frames) - 2) + [600, 3200]
    out = Path(__file__).resolve().parent / "demo.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=durations,
        loop=0, optimize=True,
    )
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
