"""Benchmark chart (pixel style, logo palette) from the measured logs:
bendtune/logs/direct_nothink.jsonl, direct_think.jsonl, tuned400_test.jsonl (writer, real Bend2) and
e2e.jsonl (QwenBend end to end, 20 items). Run: python make_bench_chart.py LOGS_DIR OUT.png
"""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, UP = 800, 450, 2
BG, DEEP, CREAM, DIM = (56, 26, 75), (36, 11, 53), (245, 228, 205), (160, 130, 180)
PURPLE, GOLD, TEAL = (156, 81, 186), (233, 174, 74), (70, 200, 180)
MENLO = "/System/Library/Fonts/Menlo.ttc"
F = {k: ImageFont.truetype(MENLO, s, index=1 if b else 0) for k, (s, b) in
     dict(title=(18, True), h=(10, True), t=(9, False), s=(8, False)).items()}


def load(p):
    return [json.loads(x) for x in open(p)]


def main():
    logs, out = Path(sys.argv[1]), Path(sys.argv[2])
    nothink, think = load(logs / "direct_nothink.jsonl"), load(logs / "direct_think.jsonl")
    writer, e2e = load(logs / "tuned400_test.jsonl"), load(logs / "e2e.jsonl")
    ids = {r["i"] for r in e2e}

    def acc(rs, key="correct", sel=lambda r: True):
        s = [r for r in rs if sel(r)]
        return sum(bool(r[key]) for r in s) / len(s), len(s)

    rows = [  # (label, value, n, color)
        ("Qwen3.5-0.8B, answers directly", *acc(nothink), CREAM),
        ("Qwen3.5-0.8B, thinking (4k cap)", *acc(think), DIM),
        ("QwenBend writer: right program", *acc(writer), PURPLE),
        ("same 20 tasks: direct answer", *acc(nothink, sel=lambda r: r["i"] in ids), CREAM),
        ("same 20 tasks: QwenBend end to end", *acc(e2e), GOLD),
        ("QwenBend engine: trace == reference", sum(r["trace_equals_reference"] for r in e2e) / len(e2e), len(e2e), TEAL),
    ]
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.fontmode = "1"
    d.text((16, 14), "QWENBEND: MEASURED", font=F["title"], fill=CREAM)
    d.text((16, 38), "200 test tasks (arithmetic, loops, gcd, digits...), greedy; answers checked against expected values",
           font=F["s"], fill=DIM)
    x0, bw = 290, 440
    for i, (label, v, n, c) in enumerate(rows):
        y = 70 + 46 * i + (16 if i >= 3 else 0)
        d.text((16, y + 4), label, font=F["t"], fill=CREAM)
        d.text((16, y + 17), f"n = {n}", font=F["s"], fill=DIM)
        d.rectangle((x0, y, x0 + bw, y + 22), fill=DEEP)
        d.rectangle((x0, y, x0 + int(bw * v), y + 22), fill=c)
        d.text((x0 + int(bw * v) + 6 if v < 0.88 else x0 + int(bw * v) - 44, y + 6), f"{100 * v:.0f}%",
               font=F["h"], fill=CREAM if v < 0.88 else DEEP)
    d.line((16, 205, W - 16, 205), fill=DIM)
    d.text((16, 372), "Read: the engine never miscomputes (20/20 traces exact), but on these small tasks a 0.8B model",
           font=F["t"], fill=CREAM)
    d.text((16, 386), "answering directly is as good or better. QwenBend's failures are programs the writer gets wrong",
           font=F["t"], fill=CREAM)
    d.text((16, 400), "or that break Bend2's rules (8 of 10 failures: digit_count, a task family it never saw).",
           font=F["t"], fill=CREAM)
    d.text((16, 424), "logs: bendtune/logs/ in github.com/voidash/qwenbend", font=F["s"], fill=GOLD)
    img.resize((W * UP, H * UP), Image.NEAREST).save(out)
    print("wrote", out, [(r[0], round(r[1], 3), r[2]) for r in rows])


if __name__ == "__main__":
    main()
