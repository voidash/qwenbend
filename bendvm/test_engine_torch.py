"""The engine circuit (engine_circuit.build) in the real Qwen3_5ForCausalLM (bf16, MPS, decode with
a cache) vs our evaluator, from the program's own tokens: CAPSTART, the program's Qwen tokens, EOF
(the circuit's capture writes the token store; nothing is preloaded), then the program's trace.
At every position the residual after the last engine layer must be equal bit for bit, and from
EOF on torch's top next token must be the trace's next token.
Run: python bendvm/test_engine_torch.py [N programs] [trace positions each]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_circuit import CAPSTART, EOF, FIRMWARE, MAX_N, TEXT0, build  # noqa: E402
from trace import trace  # noqa: E402
from paths import BENDTUNE, qwen_dir  # noqa: E402


def programs(n):
    """n dataset programs (shortest first, then spread) as (Qwen ids, ROM words)."""
    from transformers import AutoTokenizer

    from capture import build_rom
    from lexer_fw import token_words
    tok = AutoTokenizer.from_pretrained(str(qwen_dir()))
    words = token_words(build_rom(lambda i: tok.decode([i]), len(tok)))
    srcs = [json.loads(x)["messages"][2]["content"] for x in open(BENDTUNE / "data/test.jsonl")]
    out = []
    for s in srcs[::max(1, len(srcs) // n)]:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) < MAX_N and all(i in words for i in ids):
            out.append(ids)
    return out[:n], words


def main():
    from lexer_fw import Stored
    n_prog = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    n_pos = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    progs, words = programs(n_prog)
    text_ids = sorted({i for ids in progs for i in ids})
    local = {i: TEXT0 + k for k, i in enumerate(text_ids)}
    c, L = build([(i, words[i]) for i in text_ids])
    m = c.to_hf()
    last = c.cfg["n_layers"]  # hidden_states[last] = the residual after the last engine layer
    checked = 0
    for ids in progs:
        toks = trace(FIRMWARE, Stored(ids, words))[:n_pos + 1]
        seq = [CAPSTART] + [local[i] for i in ids] + [EOF] + toks
        c.reset()
        cache = None
        for pos, t in enumerate(seq[:-1]):
            h, _ = c.step(t)
            with torch.no_grad():
                o = m(input_ids=torch.tensor([[t]], device="mps"), past_key_values=cache, use_cache=True,
                      position_ids=torch.tensor([[pos]], device="mps"), output_hidden_states=True)
            cache = o.past_key_values
            got = o.hidden_states[last][0, -1].float().cpu().numpy().astype(np.float64)
            assert np.array_equal(got, h), (pos, t, float(np.abs(got - h).max()))
            if pos >= len(ids) + 1:  # from EOF on: the engine's tokens
                top = int(o.logits[0, -1].argmax())
                assert top == seq[pos + 1], (pos, "next token", top, seq[pos + 1])
            checked += 1
    print(f"engine circuit with capture, in the real Qwen3_5 (bf16, MPS): == evaluator bit for bit, and "
          f"torch's top next token == the trace's from EOF on: {checked} positions of {len(progs)} programs "
          f"(capture from the program's own tokens; nothing preloaded)")


if __name__ == "__main__":
    main()
