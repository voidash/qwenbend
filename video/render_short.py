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
from render_video import clock, phase, trace_lines  # noqa: E402

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


SR = 44100
NOTE = {n: 440.0 * 2 ** ((i - 9) / 12) for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def hz(name):
    """'C5' -> frequency."""
    return NOTE[name[:-1]] * 2 ** (int(name[-1]) - 4)


def tone(freq, dur, vol=0.18, duty=0.25, decay=6.0):
    """A pulse-wave note with a fast attack and exponential decay (8-bit style)."""
    t = np.arange(int(dur * SR)) / SR
    wave = np.where((t * freq) % 1.0 < duty, 1.0, -1.0)
    env = np.minimum(1.0, t / 0.004) * np.exp(-decay * t)
    return vol * wave * env


def noise(dur, vol=0.05):
    t = np.arange(int(dur * SR)) / SR
    return vol * np.random.default_rng(7).uniform(-1, 1, len(t)) * np.exp(-60 * t)


def soundtrack(events, seconds):
    """events: [(time s, kind, arg)] -> mono float track."""
    out = np.zeros(int(seconds * SR) + SR)

    def put(t0, x):
        i = int(t0 * SR)
        out[i:i + len(x)] += x[:len(out) - i]

    scales = {"lex": ["A4", "C5", "E5", "G5"], "compile": ["D5", "F5", "A5", "C6"],
              "run": ["C5", "E5", "G5", "B5"], "boot": ["C5", "G5", "C5", "G5"]}
    bass = {"lex": "A2", "compile": "D3", "run": "C3", "boot": "C3"}
    for t0, kind, arg in events:
        if kind == "intro":
            for i, n in enumerate(["C5", "E5", "G5", "C6"]):
                put(t0 + 0.11 * i, tone(hz(n), 0.35, 0.16))
        elif kind == "type":
            put(t0, tone(hz(["E6", "G6", "A6", "C7"][arg % 4]), 0.05, 0.07, 0.5, 40))
        elif kind == "step":
            ph, i = arg
            put(t0, tone(hz(scales[ph][i % 4]), 0.12, 0.11, 0.25, 18))
            if i % 2 == 0:
                put(t0, noise(0.05))
            if i % 4 == 0:
                put(t0, tone(hz(bass[ph]), 0.5, 0.13, 0.5, 3))
        elif kind == "phase":
            for j, n in enumerate(scales[arg][:3]):
                put(t0 + 0.05 * j, tone(hz(n) * 2, 0.15, 0.09, 0.125, 10))
        elif kind == "win":
            for i, n in enumerate(["G5", "C6", "E6", "G6"]):
                put(t0 + 0.1 * i, tone(hz(n), 0.25, 0.15))
            for n in ("C5", "E5", "G5", "C6"):
                put(t0 + 0.45, tone(hz(n), 1.4, 0.07, 0.5, 2))
    peak = np.abs(out).max()
    return out / peak * 0.6 if peak > 0 else out


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
    lines = trace_lines(engine)
    line_tok = np.array([ln[0] for ln in lines])  # token index of each line's INS
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
        # the decoded trace: every line is a token the model emitted (INS pc) plus its value tokens
        x0, y0 = 226, 6
        d.rectangle((x0, y0, 466, 330), outline=mix(label_c, 0.6))
        d.text((x0 + 4, y0 + 3), "the model's next tokens, decoded" if k is not None else
               "the model's next tokens (after <|im_end|>)", font=F["sm"], fill=DIM)
        if k is not None:
            li = int(np.searchsorted(line_tok, k, side="right"))
            for i, (_, pc, text, val) in enumerate(lines[max(0, li - 28):li]):
                yy = y0 + 15 + 11 * i
                c = {"lex": GOLD, "compile": PURPLE, "run": TEAL, "boot": DIM}[phase(pc) if pc is not None else "run"]
                d.text((x0 + 4, yy), f"{pc if pc is not None else '':>4}", font=F["code"], fill=DIM)
                d.text((x0 + 32, yy), text.replace("4294967295", "-1"), font=F["code"], fill=c)
                if val is not None:
                    d.text((x0 + 178, yy), f"={val}", font=F["code"], fill=CREAM)
        # the state column: registers (bits) and the 8,192-word memory (64 heads x 128 rows)
        sx = 474
        d.text((sx, y0), "registers", font=F["sm"], fill=DIM)
        r = regs[k] if k is not None else [0] * 16
        for i in range(16):
            for b in range(32):
                on = (r[i] >> (31 - b)) & 1
                xx, yy = sx + 4 * b, y0 + 12 + 4 * i
                d.rectangle((xx, yy, xx + 2, yy + 2), fill=TEAL if on else mix(TEAL, 0.18))
        d.text((sx, y0 + 80), "memory: DeltaNet heads", font=F["sm"], fill=DIM)
        gy = y0 + 92
        for a in range(WORDS):
            xx, yy = sx + 2 * (a // 128), gy + a % 128
            if k is not None and 0 <= k - seen[a] < FLASH:
                c = CREAM
            elif mem[a]:
                c = mix(region_color(a), 0.9)
            else:
                c = mix(region_color(a), 0.15)
            d.line((xx, yy, xx + 1, yy), fill=c)
        ly = gy + 136
        for i, (_, _, name, c) in enumerate(REGIONS):
            d.rectangle((sx, ly + 10 * i, sx + 4, ly + 4 + 10 * i), fill=c)
            d.text((sx + 7, ly - 3 + 10 * i), name, font=F["sm"], fill=DIM)
        d.text((x0, 336), label, font=F["mid"], fill=label_c)
        d.text((LW - 14 - d.textlength(note, font=F["sm"]), 340), note, font=F["sm"], fill=DIM)
        bx = 626  # layer bar: 10 engine + 1 exit + 24 Qwen
        for i in range(35):
            yy = 8 + 9 * i
            c = TEAL if i < 10 else (DIM if i == 10 else PURPLE)
            lit = (active == "engine" and i < 10) or (active == "qwen" and i > 10)
            d.rectangle((bx, yy, bx + 6, yy + 6), fill=c if lit else mix(c, 0.25))

    silent = out.with_suffix(".silent.mp4")
    v = Video(silent)
    events = [(0.05, "intro", None)]
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
        shown_before = shown if f else 0
        shown = sum(1 for x in q_t[:-1] if x <= tt)
        if shown > shown_before and f % 2 == 0:
            events.append((v.frames / FPS, "type", shown))
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
        if f == 0 or ph != last_ph:
            events.append((v.frames / FPS, "phase", ph))
        last_ph = ph
        if f % 4 == 0:  # an 8-bit pulse at 7.5 notes/s, in the phase's scale
            events.append((v.frames / FPS, "step", (ph, f // 4)))
        label, c = {"lex": ("SAME MODEL READS IT", GOLD), "compile": ("SAME MODEL CHECKS IT: BEND2 RULES", GOLD),
                    "run": ("SAME MODEL RUNS IT", TEAL), "boot": ("SAME MODEL BOOTS", TEAL)}[ph]
        img, d = canvas()
        left(d, img, prog_text, None, f"engine tokens {k + 1:,}")
        right(d, k, "engine", label, c, f"x{speed:.0f} time-lapse  real {clock(tt - e_t[0])}")
        v.add(img)

    # 4. the answer (2 s)
    events.append((v.frames / FPS, "win", None))
    img, d = canvas()
    left(d, img, prog_text, rec["result"], "answer, computed by the model")
    right(d, len(engine) - 1, "engine", "TRACE == REFERENCE, TOKEN FOR TOKEN", TEAL,
          f"{len(engine):,} tokens  {clock(st[-1] - rec['t0'])} real")
    v.add(img, int(2.0 * FPS))
    v.close()
    # the soundtrack, muxed onto the frames
    import wave
    track = soundtrack(events, v.frames / FPS)[:int(v.frames / FPS * SR)]
    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((track * 32767).astype(np.int16).tobytes())
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(out)])
    if r.returncode != 0:
        raise RuntimeError("ffmpeg mux failed")
    silent.unlink()
    wav.unlink()
    print(f"wrote {out}: {v.frames} frames, {v.frames / FPS:.1f} s, with a {len(events)}-event soundtrack")


if __name__ == "__main__":
    main()
