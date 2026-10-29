#!/usr/bin/env python3
"""Render the README terminal screenshots from real command output.

Same principle as nimbus-infer's make_demo_gif.py: the images in the README
are rendered from actual captured output of the commands shown -- not a
mockup, not a screenshot of someone else's terminal. Re-capture by rerunning
the demo (see README) and rerunning this script:

    python docs/make_demo_svg.py
"""
from __future__ import annotations

from pathlib import Path

# --- captured from real runs (2026-10-26, Windows, Python 3.13) -----------
FLOW = [
    # (kind, text) kind: cmd | out | dim | ok | bad
    ("cmd", "python examples/demo_agent.py --mode record --scenario stable"),
    ("out", "Your coffee is $3.50; with the 10% member discount you pay $6.30."),
    ("dim", "[recorded 4 steps -> .tapedeck/runs/20261026-151919-demo-stable | 284 tokens]"),
    ("cmd", "python examples/demo_agent.py --mode record --scenario drift"),
    ("out", "Your coffee is $3.50; with the 20% member discount you pay $5.60."),
    ("dim", "[recorded 4 steps -> .tapedeck/runs/20261026-151920-demo-drift | 284 tokens]"),
    ("cmd", "python examples/demo_agent.py --mode replay"),
    ("out", "Your coffee is $3.50; with the 10% member discount you pay $6.30."),
    ("ok",  "[replayed '...demo-stable': served 4 steps, 0 live calls, $0.00 spent]"),
    ("cmd", "python -m tapedeck diff demo-stable demo-drift; echo $?"),
    ("out", "run A: demo-stable (4 steps)"),
    ("out", "run B: demo-drift (4 steps)"),
    ("out", "step 2 (tool:compute_total) output:"),
    ("bad", "  A: 6.3"),
    ("ok",  "  B: 5.6"),
    ("dim", "  context (steps 0..1):"),
    ("dim", "    step 0 (model:model.plan) input: []"),
    ("dim", "    step 1 (tool:get_price) input: []"),
    ("out", "run A: $0.0001"),
    ("out", "run B: $0.0001"),
    ("bad", "1                          <- exit 1: output drift, CI red"),
]

REPORT = [
    ("cmd", "python -m tapedeck ls"),
    ("out", "run                           steps    tokens        usd  name"),
    ("out", "20261026-151920-demo-drift        4       284     0.0001  demo-drift"),
    ("out", "20261026-151919-demo-stable       4       284     0.0001  demo-stable"),
    ("cmd", "python -m tapedeck report demo-stable"),
    ("out", "    #  kind   name                          wall_ms  usage"),
    ("out", "    0  model  model.plan                        0.0  p=118 c=24"),
    ("out", "    1  tool   get_price                         0.0"),
    ("out", "    2  tool   compute_total                     0.0"),
    ("out", "    3  model  model.answer                      0.0  p=118 c=24"),
    ("out", "  total cost: $0.0001"),
]

COLORS = {
    "cmd": "#79c0ff",
    "out": "#c9d1d9",
    "dim": "#8b949e",
    "ok": "#7ee787",
    "bad": "#ff7b72",
}

FONT = "ui-monospace,'Cascadia Code','JetBrains Mono',Menlo,Consolas,monospace"
FS = 13.0          # font size
LH = 21.0          # line height
PAD_X = 20.0
CHAR_W = 7.9       # measured for 13px ui-monospace-ish stacks


def esc(text: str) -> str:
    # XML whitespace collapsing would destroy table alignment, so every
    # space becomes a no-break space (same advance width in monospace).
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "\u00a0")
    )


def render(lines, title: str) -> str:
    width = max(int(max(len(t) for _, t in lines) * CHAR_W) + 2 * PAD_X + 8, 560)
    header_h = 38.0
    height = header_h + PAD_X + len(lines) * LH + 14

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{int(height)}" '
        f'viewBox="0 0 {width} {int(height)}" role="img" aria-label="{esc(title)}">',
        "<!-- rendered from real command output by docs/make_demo_svg.py -->",
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" '
        f'fill="#0d1117" stroke="#30363d"/>',
        f'<path d="M0.5 10.5 a10 10 0 0 1 10 -10 h{width - 21} a10 10 0 0 1 10 10 v28 h-{width - 1} z" '
        f'fill="#161b22" stroke="none"/>',
        f'<line x1="0.5" y1="{header_h}" x2="{width - 0.5}" y2="{header_h}" stroke="#30363d"/>',
    ]
    for i, color in enumerate(("#ff5f56", "#ffbd2e", "#27c93f")):
        parts.append(f'<circle cx="{24 + i * 20}" cy="19" r="6" fill="{color}"/>')
    parts.append(
        f'<text x="{width / 2}" y="23.5" text-anchor="middle" font-family="{FONT}" '
        f'font-size="12" fill="#8b949e">{esc(title)}</text>'
    )
    y = header_h + PAD_X + 2
    for kind, text in lines:
        if kind == "cmd":
            parts.append(
                f'<text x="{PAD_X}" y="{y}" font-family="{FONT}" font-size="{FS}">'
                f'<tspan fill="#3fb950" font-weight="bold">$</tspan>'
                f'<tspan fill="{COLORS[kind]}" dx="8">{esc(text)}</tspan></text>'
            )
        else:
            parts.append(
                f'<text x="{PAD_X}" y="{y}" font-family="{FONT}" font-size="{FS}" '
                f'fill="{COLORS[kind]}">{esc(text)}</text>'
            )
        y += LH
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    out = Path(__file__).resolve().parent
    (out / "demo.svg").write_text(
        render(FLOW, "tapedeck — record · replay · diff (real output)"), encoding="utf-8"
    )
    (out / "report.svg").write_text(
        render(REPORT, "tapedeck — ls · report (real output)"), encoding="utf-8"
    )
    print(f"wrote {out / 'demo.svg'} and {out / 'report.svg'}")


if __name__ == "__main__":
    main()
