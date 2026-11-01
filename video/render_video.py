"""Render the QwenBend demo video from a recorded run (merge.py --record=...): every token, value
and timestamp on screen comes from the recording; the engine run is shown as a labelled
time-lapse with the real elapsed clock.
Run: python video/render_video.py video/run_sum_const.json video/qwenbend_demo.mp4
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bendvm"))
from firmware2 import FIRMWARE  # noqa: E402
from trace import END, LAST, MORE, NPC  # noqa: E402

W, H, FPS = 1920, 1080, 30
FONT_DIR = Path.home() / "Library/Fonts"
BG, PANEL, FG, DIM, ACC, GOOD, QW = (13, 15, 20), (22, 25, 33), (225, 228, 235), (120, 126, 140), \
    (255, 184, 76), (110, 220, 140), (120, 170, 255)
ENGINE_SECONDS = 15  # screen time of the engine run (time-lapse)


def font(size, bold=False):
    name = "FiraCodeNerdFontMono-Bold.ttf" if bold else "FiraCodeNerdFontMono-Regular.ttf"
    p = FONT_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"font {p} missing")
    return ImageFont.truetype(str(p), size)


F = {k: font(*v) for k, v in dict(title=(110, True), sub=(38,), body=(30,), code=(27,), small=(24,),
                                    big=(170, True), label=(26, True), mid=(46, True)).items()}


def phase(pc):
    if pc < 17:
        return "boot"
    if pc < 997:
        return "compile"
    if pc < 1988:
        return "lex"
    return "run"


def disasm(pc):
    op, *a = FIRMWARE[pc]
    regs = {"LI": 1, "MOV": 2, "SHR1": 2, "ADD": 3, "SUB": 3, "SLTU": 3, "SLEU": 3, "SEQ": 3, "XOR": 3,
            "ROTN": 3, "SEQI": 2, "ADDI": 2, "LD": 2, "ST": 1, "BEQZ": 1, "BNEZ": 1, "BEQI": 1, "BNEI": 1,
            "BEQ": 2, "BNE": 2, "JAL": 1, "JR": 1, "OUT": 1}.get(op, 0)
    parts = [f"r{x}" if i < regs else str(x) for i, x in enumerate(a)]
    if op == "ST":  # ST rs, imm, rt
        parts = [f"r{a[0]}", str(a[1]), f"r{a[2]}"]
    return f"{op:<5} {', '.join(parts)}"


def trace_lines(engine):
    """[(token index of the INS, pc, text, value or None)]"""
    out, i = [], 0
    while i < len(engine):
        t = engine[i]
        if t < NPC:
            j, v, nv = i + 1, 0, False
            while j < len(engine) and NPC <= engine[j] < END:
                b = engine[j]
                v = (v << 8) | ((b - MORE) % 256 if b < LAST else b - LAST)
                nv = nv or b >= LAST
                j += 1
            out.append((i, t, disasm(t), v if nv else None))
            i = j
        else:
            out.append((i, None, "END", None))
            i += 1
    return out


def clock(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


class Video:
    def __init__(self, path):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset",
                                   "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                   str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def add(self, img, n=1):
        b = img.tobytes()
        for _ in range(n):
            self.p.stdin.write(b)
        self.frames += n

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def centered(d, y, text, f, fill):
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, y), text, font=f, fill=fill)


def fade(img, a):
    return Image.blend(Image.new("RGB", (W, H), BG), img, max(0.0, min(1.0, a)))


def hold(v, img, seconds, fade_in=0.4, fade_out=0.4):
    n = int(seconds * FPS)
    fi, fo = int(fade_in * FPS), int(fade_out * FPS)
    for k in range(n):
        a = min(1.0, (k + 1) / fi if fi else 1.0, (n - k) / fo if fo else 1.0)
        v.add(fade(img, a) if a < 1 else img)


def header(d, right_text, color):
    d.text((60, 36), "QwenBend", font=F["mid"], fill=FG)
    d.text((300, 50), "· Qwen3.5-0.8B-based · MPS · greedy · 1 token/forward",
           font=F["small"], fill=DIM)
    w = d.textlength(right_text, font=F["label"])
    d.text((W - 60 - w, 50), right_text, font=F["label"], fill=color)


def program_panel(d, pieces, n_shown, x, y, w, h, title, cursor):
    d.rounded_rectangle((x, y, x + w, y + h), 18, fill=PANEL)
    d.text((x + 28, y + 20), title, font=F["label"], fill=QW)
    text = "".join(pieces[:n_shown])
    lines = text.split("\n")
    yy = y + 70
    for ln in lines[:24]:
        d.text((x + 28, yy), ln, font=F["code"], fill=FG)
        yy += 36
    if cursor:
        last = lines[min(len(lines), 24) - 1]
        cx = x + 28 + d.textlength(last, font=F["code"])
        d.rectangle((cx + 2, yy - 36 + 4, cx + 16, yy - 36 + 32), fill=QW)


def main():
    rec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    if not (rec["trace_equal"] and rec["program_equal_mlx"] and rec["exit_violations"] == 0
            and rec["result"] == rec["reference"] == rec["expected"]):
        raise ValueError("the recorded run did not pass its checks: refusing to render it")
    pieces, qwen_ids, engine, st = rec["qwen_pieces"], rec["qwen_ids"], rec["engine"], rec["stamps"]
    nq = len(qwen_ids)
    prog_pieces = pieces[:-1]  # without <|im_end|>
    t_prompt = st[0] - rec["t0"]
    q_t = [s - st[0] for s in st[1:1 + nq]]
    e_t = [s - st[0] for s in st[1 + nq:]]
    lines = trace_lines(engine)
    v = Video(out)

    # 1. title
    img, d = canvas()
    centered(d, 330, "QwenBend", F["title"], FG)
    centered(d, 480, "Qwen3.5-0.8B writes a Bend program.", F["sub"], FG)
    centered(d, 535, "Then the same model runs it, inside its own forward pass.", F["sub"], ACC)
    centered(d, 640, "no compiler  ·  no interpreter  ·  no host", F["body"], DIM)
    hold(v, img, 2.5, 0.4)

    # 2. the task
    img, d = canvas()
    header(d, "THE PROMPT", DIM)
    d.rounded_rectangle((160, 330, W - 160, 560), 22, fill=PANEL)
    d.text((200, 360), "user", font=F["label"], fill=DIM)
    d.text((200, 420), rec["task"], font=F["sub"], fill=FG)
    d.text((200, 490), f"(chat prompt: {rec['prompt_tokens']} tokens, read in {t_prompt:.0f} s)",
           font=F["small"], fill=DIM)
    hold(v, img, 2.0)

    # 3. Qwen writes the program (real time; Qwen's own tokens)
    dur = q_t[-1]
    speed_q = max(1.0, dur / 3.0)
    n_frames = int(dur / speed_q * FPS) + 1
    for k in range(n_frames):
        tt = k / FPS * speed_q
        shown = sum(1 for x in q_t[:-1] if x <= tt)
        img, d = canvas()
        header(d, "QWEN WRITES" + ("" if speed_q == 1 else f"  ×{speed_q:.1f}"), QW)
        program_panel(d, prog_pieces, shown, 60, 120, 1100, 900, "QwenBend output (assistant)", True)
        d.text((1220, 160), "tokens", font=F["label"], fill=DIM)
        d.text((1220, 200), f"{shown} / {len(prog_pieces)}", font=F["mid"], fill=FG)
        d.text((1220, 300), "real time", font=F["label"], fill=DIM)
        d.text((1220, 340), f"{tt:5.1f} s", font=F["mid"], fill=FG)
        v.add(img)
    img, d = canvas()
    header(d, "QWEN WRITES", QW)
    program_panel(d, prog_pieces, len(prog_pieces), 60, 120, 1100, 900, "QwenBend output (assistant)", False)
    d.text((1220, 160), "tokens", font=F["label"], fill=DIM)
    d.text((1220, 200), f"{len(prog_pieces)} / {len(prog_pieces)}", font=F["mid"], fill=FG)
    d.text((1220, 300), "next token", font=F["label"], fill=DIM)
    d.text((1220, 340), "<|im_end|>", font=F["mid"], fill=ACC)
    d.text((1220, 460), "Normally the answer ends here.", font=F["body"], fill=FG)
    d.text((1220, 510), "Here the same model keeps going:", font=F["body"], fill=FG)
    d.text((1220, 560), "its own layers execute the", font=F["body"], fill=FG)
    d.text((1220, 610), "program it just wrote.", font=F["body"], fill=FG)
    v.add(img, int(2.0 * FPS))

    # 4. the engine runs (time-lapse over the recorded tokens)
    total = e_t[-1] - e_t[0]
    n_frames = ENGINE_SECONDS * FPS
    speed = total / ENGINE_SECONDS
    ph_counts = {"boot": 0, "lex": 0, "compile": 0, "run": 0}
    li = 0
    for k in range(n_frames):
        tt = e_t[0] + (k + 1) / n_frames * total
        lo, hi = 0, len(e_t)
        while lo < hi:  # tokens emitted by time tt
            mid = (lo + hi) // 2
            if e_t[mid] <= tt:
                lo = mid + 1
            else:
                hi = mid
        n_tok = lo
        while li < len(lines) and lines[li][0] < n_tok:
            if lines[li][1] is not None:
                ph_counts[phase(lines[li][1])] += 1
            li += 1
        img, d = canvas()
        header(d, f"SAME MODEL RUNS IT · ×{speed:.0f} TIME-LAPSE", ACC)
        program_panel(d, prog_pieces, len(prog_pieces), 60, 120, 760, 900, "the program (Qwen's tokens)", False)
        x0 = 860
        d.rounded_rectangle((x0, 120, W - 60, 1020), 18, fill=PANEL)
        d.text((x0 + 28, 140), "QwenBend output after <|im_end|>: the execution trace (decoded)",
               font=F["label"], fill=ACC)
        shown = lines[max(0, li - 20):li]
        yy = 200
        for (ti, pc, text, val) in shown:
            col = {"lex": QW, "compile": (190, 150, 255), "run": GOOD, "boot": DIM}.get(phase(pc) if pc is not None else "run")
            d.text((x0 + 28, yy), f"{pc if pc is not None else '':>5}", font=F["code"], fill=DIM)
            d.text((x0 + 130, yy), text, font=F["code"], fill=col)
            if val is not None:
                d.text((x0 + 560, yy), f"→ {val}", font=F["code"], fill=FG)
            yy += 34
        # stats column
        sx = x0 + 760
        d.text((sx, 200), "engine tokens", font=F["small"], fill=DIM)
        d.text((sx, 232), f"{n_tok:,}", font=F["mid"], fill=FG)
        d.text((sx, 320), "real elapsed", font=F["small"], fill=DIM)
        d.text((sx, 352), clock(tt - e_t[0]), font=F["mid"], fill=FG)
        cur = phase(lines[li - 1][1]) if li and lines[li - 1][1] is not None else "run"
        d.text((sx, 440), "now", font=F["small"], fill=DIM)
        d.text((sx, 472), {"boot": "boot", "lex": "lexing", "compile": "type-check",
                           "run": "running"}[cur], font=F["mid"],
               fill={"lex": QW, "compile": (190, 150, 255), "run": GOOD, "boot": DIM}[cur])
        d.text((sx, 570), "instructions", font=F["small"], fill=DIM)
        for r, (name, key, colr) in enumerate((("lex", "lex", QW), ("check", "compile", (190, 150, 255)),
                                               ("eval", "run", GOOD))):
            d.text((sx, 610 + 40 * r), f"{name:<6}{ph_counts[key]:>6,}", font=F["small"], fill=colr)
        d.text((x0 + 28, 960), "every line = a token the model emitted; values → are its next tokens",
               font=F["small"], fill=DIM)
        v.add(img)

    # 5. the result
    img, d = canvas()
    header(d, "RESULT", GOOD)
    centered(d, 170, "OUT", F["sub"], DIM)
    centered(d, 230, str(rec["result"]), F["big"], GOOD)
    checks = [
        f"✓  trace == the reference interpreter's, token for token ({len(engine):,} tokens)",
        f"✓  result {rec['result']} == expected {rec['expected']}",
        "✓  Qwen's program == the fine-tuned model's (MLX) output",
        "✓  engine dims exactly 0 before Qwen's layers, at every position",
        "✓  5 / 5 test programs end to end (sum_const, gcd ×2, digit_sum, triangle)",
    ]
    for i, c in enumerate(checks):
        d.text((330, 470 + 58 * i), c, font=F["body"], fill=FG)
    d.text((330, 800), f"one Qwen3_5ForCausalLM · {rec['params'] / 1e9:.2f}B params (bf16) · 35 layers: "
           f"10 engine + 1 exit + 24 Qwen", font=F["small"], fill=DIM)
    d.text((330, 840), f"Apple MPS · {(rec['prompt_tokens'] + nq + len(engine)) / (st[-1] - rec['t0']):.0f} tokens/s "
           f"· whole run {clock(st[-1] - rec['t0'])} real time · recorded {rec['recorded_utc']}",
           font=F["small"], fill=DIM)
    hold(v, img, 5.0)

    # 6. end card
    img, d = canvas()
    centered(d, 380, "QwenBend", F["title"], FG)
    centered(d, 530, "the model writes the program and runs it", F["sub"], ACC)
    hold(v, img, 2.0, 0.4, 0.6)
    v.close()
    print(f"wrote {out}: {v.frames} frames, {v.frames / FPS:.1f} s")


if __name__ == "__main__":
    main()
