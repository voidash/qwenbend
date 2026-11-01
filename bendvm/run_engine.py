"""Free-running engine from a program's own tokens: the real Qwen3_5ForCausalLM (bf16, MPS) built from
our engine circuit is fed CAPSTART, the BendLite program's Qwen tokens and EOF (its capture layers
store the tokens; nothing is preloaded), then generates greedily, one token per forward pass,
until END. The result is read from the trace (the bytes after the OUT instruction) and compared
with bendlite_ref.
Run: python bendvm/run_engine.py [N programs] [--long]
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bendlite_ref import BendLiteError, run_source  # noqa: E402
from engine_circuit import CAPSTART, EOF, FIRMWARE, MAX_N, TEXT0, build  # noqa: E402
from trace import END, LAST, MORE  # noqa: E402
from paths import BENDTUNE, qwen_dir  # noqa: E402

OUT_PCS = {pc for pc, ins in enumerate(FIRMWARE) if ins[0] == "OUT"}


def result(toks):
    """The value printed by OUT, from the trace: INS(OUT) then MORE_d..., LAST."""
    for i, t in enumerate(toks):
        if t in OUT_PCS:
            v, j = 0, i + 1
            while j < len(toks):
                b = toks[j]
                if MORE <= b < LAST:
                    v = (v << 8) | ((b - MORE) % 256)
                elif LAST <= b < END:
                    return (v << 8) | (b - LAST)
                j += 1
    return "ERR"


def run(m, prefix, limit=200_000):
    """Feed prefix (CAPSTART, program, EOF), then generate greedily until END. -> engine tokens."""
    cache, out = None, []
    with torch.no_grad():
        for pos, t in enumerate(prefix):
            o = m(input_ids=torch.tensor([[t]], device="mps"), past_key_values=cache, use_cache=True,
                  position_ids=torch.tensor([[pos]], device="mps"))
            cache = o.past_key_values
        pos = len(prefix)
        while len(out) < limit:
            t = int(o.logits[0, -1].argmax())
            out.append(t)
            if t == END:
                break
            o = m(input_ids=torch.tensor([[t]], device="mps"), past_key_values=cache, use_cache=True,
                  position_ids=torch.tensor([[pos]], device="mps"))
            cache = o.past_key_values
            pos += 1
    return out


def main():
    from transformers import AutoTokenizer

    from capture import build_rom
    from lexer_fw import token_words
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tok = AutoTokenizer.from_pretrained(str(qwen_dir()))
    words = token_words(build_rom(lambda i: tok.decode([i]), len(tok)))
    rows = [json.loads(x) for x in open(BENDTUNE / "data/test.jsonl")]
    picked = []
    for row in rows[::max(1, len(rows) // (4 * n))]:
        ids = tok.encode(row["messages"][2]["content"], add_special_tokens=False)
        if len(ids) < MAX_N and all(i in words for i in ids):
            picked.append((row, ids))
    if "--long" in sys.argv:
        picked.sort(key=lambda r: -len(r[1]))
    picked = picked[:n]
    text_ids = sorted({i for _, ids in picked for i in ids})
    local = {i: TEXT0 + k for k, i in enumerate(text_ids)}
    c, _ = build([(i, words[i]) for i in text_ids])
    m = c.to_hf()
    ok = 0
    for row, ids in picked:
        src = row["messages"][2]["content"]
        try:
            want = run_source(src)[0]
        except BendLiteError:
            want = "ERR"
        t0 = time.time()
        toks = run(m, [CAPSTART] + [local[i] for i in ids] + [EOF])
        dt = time.time() - t0
        got = result(toks)
        ok += got == want
        print(f"{row['meta']['family']:<16} {len(ids):>3} program tokens -> engine {got} reference {want} "
              f"{'OK' if got == want else 'MISMATCH'} | {len(toks)} engine tokens, {dt:.0f} s "
              f"({len(toks) / dt:.0f} tok/s)", flush=True)
    print(f"{ok}/{len(picked)} programs: from the program's own tokens, the model's forward passes "
          f"computed the reference's result")


if __name__ == "__main__":
    main()
