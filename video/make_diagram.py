"""The 'how QwenBend works' image (pixel style, logo palette): the token timeline of a run and the
layer stack one token goes through. Numbers are the measured ones (sum_const run, merge.py).
Run: python video/make_diagram.py assets/how_it_works.png
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
W, H, UP = 800, 450, 2
BG, DEEP, CREAM, DIM = (56, 26, 75), (36, 11, 53), (245, 228, 205), (160, 130, 180)
PURPLE, GOLD, TEAL, RED = (156, 81, 186), (233, 174, 74), (70, 200, 180), (240, 90, 110)
MENLO = "/System/Library/Fonts/Menlo.ttc"
F = {k: ImageFont.truetype(MENLO, s, index=1 if b else 0) for k, (s, b) in
     dict(title=(20, True), h=(11, True), t=(9, False), s=(8, False)).items()}


def box(d, x0, y0, x1, y1, fill, outline=None):
    d.rectangle((x0, y0, x1, y1), fill=fill, outline=outline)


def text_c(d, cx, y, s, f, fill):
    d.text((cx - d.textlength(s, font=f) / 2, y), s, font=f, fill=fill)


def arrow(d, x0, y, x1, color):
    d.line((x0, y, x1, y), fill=color, width=2)
    d.polygon([(x1, y), (x1 - 5, y - 4), (x1 - 5, y + 4)], fill=color)


def main():
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.fontmode = "1"
    logo = Image.open(HERE.parent / "assets" / "logo.png").convert("RGB").resize((112, 63), Image.NEAREST)
    img.paste(logo, (W - 124, 8))
    d.text((16, 14), "HOW QWENBEND WORKS", font=F["title"], fill=CREAM)
    d.text((16, 40), "one Qwen3_5ForCausalLM: Qwen writes the program, the same weights check and run it",
           font=F["t"], fill=DIM)

    # 1. the token stream of one run
    d.text((16, 72), "1  ONE SEQUENCE, ONE TOKEN PER FORWARD PASS", font=F["h"], fill=GOLD)
    segs = [("chat prompt", "242 tokens", 90, DIM), ("</think>", "capture on", 64, GOLD),
            ("Qwen writes Bend", "70 tokens", 150, PURPLE), ("<|im_end|>", "engine on", 70, GOLD),
            ("read   check   run", "7,869 engine tokens", 300, TEAL), ("770", "END", 52, GOLD)]
    x = 16
    for name, sub, w, c in segs:
        box(d, x, 92, x + w - 4, 128, DEEP, c)
        text_c(d, x + (w - 4) / 2, 98, name, F["t"], c if c != DIM else CREAM)
        text_c(d, x + (w - 4) / 2, 113, sub, F["s"], DIM)
        x += w
    d.text((16, 136), "capture: while Qwen writes, the engine layers store each token in memory. "
           "engine tokens: INS(pc) = one executed", font=F["s"], fill=DIM)
    d.text((16, 147), "instruction, then its value as bytes. The model reads its own last tokens back: "
           "that is how the CPU advances.", font=F["s"], fill=DIM)

    # 2. the layer stack one token goes through
    d.text((16, 172), "2  WHAT ONE TOKEN GOES THROUGH  (hidden 4096 = 1024 Qwen dims + 3072 engine dims)",
           font=F["h"], fill=GOLD)
    stack = [("embedding", "Qwen's row on Qwen dims\\n+ engine fields", 140, CREAM),
             ("10 engine layers", "DeltaNet heads = registers +\\n8,192-word memory (one-hot rows)\\n"
              "MLP units = exact AND gates:\\ndecode, ALU, compare, emit", 196, TEAL),
             ("exit layer", "sets every engine dim\\nto exactly 0, so Qwen\\nsees its normal input", 150, DIM),
             ("24 Qwen layers", "Qwen3.5-0.8B + BendLite\\nLoRA, zero-padded", 140, PURPLE),
             ("tied lm_head", "Qwen token, or the\\nengine's next token", 118, GOLD)]
    x = 16
    for name, sub, w, c in stack:
        box(d, x, 192, x + w - 14, 268, DEEP, c)
        text_c(d, x + (w - 14) / 2, 198, name, F["t"], c)
        for j, ln in enumerate(sub.split("\\n")):
            text_c(d, x + (w - 14) / 2, 214 + 11 * j, ln, F["s"], CREAM)
        if name != stack[-1][0]:
            arrow(d, x + w - 13, 230, x + w - 2, DIM)
        x += w

    # 3. why it is exact, and what it checks
    d.text((16, 290), "3  WHY IT IS EXACT", font=F["h"], fill=GOLD)
    for j, s in enumerate(["every value sits where bf16 math is exact: one-hot rows, powers of 2,",
                           "saturated sigmoid / SiLU; residual 'ballast' fixes every RMSNorm scale",
                           "checked: 39,913 positions vs an exact evaluator, 3,230 bit for bit in torch",
                           "trace == reference interpreter, token for token, on every end-to-end run"]):
        d.text((16, 308 + 12 * j), s, font=F["s"], fill=CREAM)
    d.text((410, 290), "4  WHAT IT CHECKS (BEND2 RULES, IN-WEIGHTS)", font=F["h"], fill=GOLD)
    for j, s in enumerate(["types: U32 / Nat / Bool (5 vs 5n)",
                           "affine use: a variable used twice needs +",
                           "termination: recursion shrinks the 1n+p binder",
                           "a rule broken -> FAIL, not a run"]):
        d.text((410, 308 + 12 * j), s, font=F["s"], fill=CREAM)

    # footer
    box(d, 0, 372, W, H, DEEP)
    d.text((16, 382), "3.97B dense = 0.75B Qwen + 53M engine (hand-set, not trained) + zero padding",
           font=F["t"], fill=CREAM)
    d.text((16, 398), "~20 tok/s on an M4 Pro: slow by design, one instruction per forward pass of the whole model",
           font=F["t"], fill=DIM)
    d.text((16, 414), "verifies the computation, not the intent: the program is only as right as the 0.8B writer",
           font=F["t"], fill=DIM)
    d.text((16, 432), "github.com/voidash/qwenbend", font=F["t"], fill=GOLD)
    out = Path(sys.argv[1])
    img.resize((W * UP, H * UP), Image.NEAREST).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
