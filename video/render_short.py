"""QwenBend short video (~15 s, 1280x720, pixel style in the logo's palette), modelled on the Qwenputer
demo: left the prompt / Qwen's program / the answer; right the engine's live state (registers and the
8,192-word memory held in the Gated DeltaNet heads) and the layer bar. The state is replayed exactly
from the recorded trace with trace.TraceMachine (== what the heads hold: verified bit for bit).
Run: python video/render_short.py video/run_sum_const.json assets/qwenbend.mp4
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bendvm"))
from capture import build_rom  # noqa: E402
from firmware2 import FIRMWARE  # noqa: E402
from lexer_fw import TOK, Stored, token_words  # noqa: E402
from trace import END, LAST, TraceMachine  # noqa: E402
from paths import qwen_dir  # noqa: E402

sys.path.insert(0, str(HERE))
from render_video import clock, phase  # noqa: E402

LW, LH, UP, FPS = 640, 360, 2, 30
BG, DEEP, CREAM, DIM = (56, 26, 75), (36, 11, 53), (245, 228, 205), (150, 120, 170)
PURPLE, GOLD, TEAL, RED = (156, 81, 186), (233, 174, 74), (70, 200, 180), (240, 90, 110)
MENLO = "/System/Library/Fonts/Menlo.ttc"
F = {k: ImageFont.truetype(MENLO, s, index=1 if b else 0) for k, (s, b) in
     dict(big=(26, True), mid=(12, True), txt=(10, False), sm=(8, False), code=(9, False)).items()}
REGIONS = [(512, 1536, "tape", GOLD), (4608, 5632, "code", TEAL), (5632, 6144, "stacks", RED),
           (6144, 8192, "program tokens", PURPLE)]
WORDS = 8192
FLASH = 40  # tokens a fresh write stays highlighted


def region_color(a):
    for lo, hi, _, c in REGIONS:
        if lo <= a < hi:
            return c
    return CREAM


def mix(c, a):
    return tuple(int(DEEP[i] + (c[i] - DEEP[i]) * a) for i in range(3))


class Video:
    def __init__(self, path):
        self.p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{LW * UP}x{LH * UP}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                   "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags",
                                   "+faststart", str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def add(self, img, n=1):
        b = img.resize((LW * UP, LH * UP), Image.NEAREST).tobytes()
        for _ in range(n):
            self.p.stdin.write(b)
        self.frames += n

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def canvas():
    img = Image.new("RGB", (LW, LH), BG)
    d = ImageDraw.Draw(img)
    d.fontmode = "1"  # hard pixels
    return img, d


def replay(engine, prog):
    """Per engine token: registers, latched pc, and the memory write (addr, value) it carried."""
    tm = TraceMachine(FIRMWARE, prog)
    mem0 = np.array(tm.mem[:WORDS], dtype=np.int64)
    regs, pcs, writes = [], [], []
    for t in engine:
        addr = None
        if LAST <= t < END and tm.latch is not None and FIRMWARE[tm.latch][0] == "ST":
            _, rs, imm, _ = FIRMWARE[tm.latch]
            addr = (tm.r[rs] + imm) % 2**32
        tm.feed(t)
        writes.append((addr, tm.mem[addr]) if addr is not None and addr < WORDS else None)
        regs.append(list(tm.r))
        pcs.append(tm.latch)
    return mem0, regs, pcs, writes


def main():
    rec = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    if not (rec["trace_equal"] and rec["program_equal_mlx"] and rec["exit_violations"] == 0
            and rec["result"] == rec["reference"] == rec["expected"]):
        raise ValueError("the recorded run did not pass its checks: refusing to render it")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(qwen_dir()))
    words = token_words(build_rom(lambda i: tok.decode([i]), len(tok)))
    nn = tok.encode("\n\n", add_special_tokens=False)
    pieces, engine, st = rec["qwen_pieces"], rec["engine"], rec["stamps"]
    prog_ids = rec["qwen_ids"][:-1]
    prog_text = "".join(pieces[:-1]).strip("\n")
    nq = len(pieces)
    mem0, regs, pcs, writes = replay(engine, Stored(nn + prog_ids, words))
    q_t = [s - st[0] for s in st[1:1 + nq]]
    e_t = np.array([s - st[0] for s in st[1 + nq:]])
    logo = Image.open(HERE.parent / "assets" / "logo.png").convert("RGB")
    logo_big, logo_sm = logo.resize((256, 143), Image.NEAREST), logo.resize((150, 84), Image.NEAREST)
    mem = np.zeros(WORDS, dtype=np.int64)
    seen = np.full(WORDS, -10**9)  # engine token index of each word's last write
    tok_slots = [[TOK + 512 * k + n for k in range(4)] for n in range(len(nn + prog_ids) + 1)]

    def left(d, img, text, answer, sub):
        img.paste(logo_sm, (10, 6))
        for i, s in enumerate(("Qwen3.5-0.8B writes a Bend program;", "the same model checks it and runs",
                               "it inside its own weights.")):
            d.text((12, 94 + 10 * i), s, font=F["sm"], fill=CREAM)
        d.text((12, 132), "you", font=F["sm"], fill=DIM)
        d.text((12, 142), rec["task"].replace("Write a BendLite program. Task: ", ""), font=F["txt"], fill=CREAM)
        d.text((12, 160), "Qwen wrote", font=F["sm"], fill=DIM)
        for i, ln in enumerate(text.split("\n")[:11]):
            d.text((12, 170 + 10 * i), ln, font=F["code"], fill=CREAM)
        d.text((12, 292), sub, font=F["sm"], fill=DIM)
        if answer is not None:
            d.text((12, 302), str(answer), font=F["big"], fill=GOLD)

    def right(d, k, active, label, label_c, note):
        x0, y0 = 232, 8
        d.text((x0, y0), "the engine's state, held in Gated DeltaNet heads", font=F["sm"], fill=DIM)
        r = regs[k] if k is not None else [0] * 16
        d.text((x0, y0 + 14), "regs", font=F["sm"], fill=DIM)
        for i in range(16):
            for b in range(32):
                on = (r[i] >> (31 - b)) & 1
                xx, yy = x0 + 30 + 6 * b, y0 + 14 + 4 * i
                d.rectangle((xx, yy, xx + 4, yy + 2), fill=TEAL if on else mix(TEAL, 0.18))
        d.text((x0, y0 + 82), "memory: 8,192 words = 64 heads x 128 rows", font=F["sm"], fill=DIM)
        gx, gy = x0, y0 + 94
        for a in range(WORDS):
            xx, yy = gx + 6 * (a // 128), gy + 2 * (a % 128)
            if k is not None and 0 <= k - seen[a] < FLASH:
                c = CREAM
            elif mem[a]:
                c = mix(region_color(a), 0.9)
            else:
                c = mix(region_color(a), 0.15)
            d.line((xx, yy, xx + 4, yy), fill=c)
        lx = x0
        for _, _, name, c in REGIONS:
            d.rectangle((lx, 352 - 22, lx + 4, 352 - 18), fill=c)
            d.text((lx + 7, 352 - 25), name, font=F["sm"], fill=DIM)
            lx += 18 + d.textlength(name, font=F["sm"])
        d.text((x0, 352 - 10), label, font=F["mid"], fill=label_c)
        d.text((LW - 14 - d.textlength(note, font=F["sm"]), 352 - 8), note, font=F["sm"], fill=DIM)
        bx = 622  # layer bar: 10 engine + 1 exit + 24 Qwen
        for i in range(35):
            yy = 20 + 8 * i
            c = TEAL if i < 10 else (DIM if i == 10 else PURPLE)
            lit = (active == "engine" and i < 10) or (active == "qwen" and i > 10)
            d.rectangle((bx, yy, bx + 6, yy + 5), fill=c if lit else mix(c, 0.25))

    v = Video(out)
    # 1. logo (1.5 s)
    img, d = canvas()
    img.paste(logo_big, ((LW - 256) // 2, 60))
    for i, s in enumerate(("writes a Bend program.", "checks it. runs it.", "inside its own weights.")):
        d.text(((LW - d.textlength(s, font=F["mid"])) // 2, 222 + 18 * i), s, font=F["mid"],
               fill=[CREAM, GOLD, TEAL][i])
    v.add(img, int(1.5 * FPS))

    # 2. Qwen writes (2.5 s): Qwen's layers active; the engine's capture stores each token as it goes
    n = int(2.5 * FPS)
    for f in range(n):
        tt = (f + 1) / n * q_t[-1]
        shown = sum(1 for x in q_t[:-1] if x <= tt)
        for slot in tok_slots[:len(nn) + shown]:
            mem[slot] = mem0[slot]
        img, d = canvas()
        left(d, img, "".join(pieces[:shown]).strip("\n"), None, "")
        right(d, None, "qwen", "QWEN WRITES BEND", PURPLE, f"real {tt:4.1f} s")
        v.add(img)
    mem[:] = mem0  # after <|im_end|>: the token store as capture left it (incl. EOF)

    # 3. the engine (9 s time-lapse): engine layers active, state replayed from the trace
    n = int(9.0 * FPS)
    speed = (e_t[-1] - e_t[0]) / 9.0
    done = 0
    for f in range(n):
        tt = e_t[0] + (f + 1) / n * (e_t[-1] - e_t[0])
        k = max(0, int(np.searchsorted(e_t, tt, side="right")) - 1)
        while done <= k:
            if writes[done] is not None:
                a, val = writes[done]
                mem[a], seen[a] = val, done
            done += 1
        ph = phase(pcs[k]) if pcs[k] is not None else "run"
        label, c = {"lex": ("SAME MODEL READS IT", GOLD), "compile": ("SAME MODEL CHECKS IT: BEND2 RULES", GOLD),
                    "run": ("SAME MODEL RUNS IT", TEAL), "boot": ("SAME MODEL BOOTS", TEAL)}[ph]
        img, d = canvas()
        left(d, img, prog_text, None, f"engine tokens {k + 1:,}")
        right(d, k, "engine", label, c, f"x{speed:.0f} time-lapse  real {clock(tt - e_t[0])}")
        v.add(img)

    # 4. the answer (2 s)
    img, d = canvas()
    left(d, img, prog_text, rec["result"], "answer, computed by the model")
    right(d, len(engine) - 1, "engine", "TRACE == REFERENCE, TOKEN FOR TOKEN", TEAL,
          f"{len(engine):,} tokens  {clock(st[-1] - rec['t0'])} real")
    v.add(img, int(2.0 * FPS))
    v.close()
    print(f"wrote {out}: {v.frames} frames, {v.frames / FPS:.1f} s")


if __name__ == "__main__":
    main()
