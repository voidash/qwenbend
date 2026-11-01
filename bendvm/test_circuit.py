"""C1c: the circuit IR + evaluator vs the real Qwen3_5ForCausalLM (bf16, MPS, decode mode with a
cache), on gadgets built in the IR: a register file (SET(r, v) writes v into DeltaNet row r,
GET(r) reads it back into an always-one-hot OUT field) and a SwiGLU table (NXT = OUT + 1 mod 16:
one unit per value, exactly 0 off-match, the identity region on-match, a ballast dim as the
constant input). The evaluator's logits must equal torch's bit for bit at every position, and
the top VAL / NXT tokens must be the stored value and its successor.
Run: python bendvm/test_circuit.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit import Circuit  # noqa: E402

H, NR, K = 256, 8, 3  # hidden, registers, residual energy 256 * 4^K
R0, V0, OP0, OUT0, NXT0, BAL0 = 0, 9, 26, 29, 46, 64  # fields: R (8 + none), V (16 + none),
# OP (SET GET BOS), OUT (16 + none), NXT (16 + none); ballast from BAL0
SET, GET, BOS = 0, 1, 2


def tok_set(r, v):
    return r * 16 + v


def tok_get(r):
    return 128 + r


TOK_BOS, VAL0, NVAL0, VOCAB = 136, 137, 153, 169


def ballast(row):
    """Top the row's energy up to H * 4^K with dims holding powers of 2 (base-4 digits)."""
    rest, d, j = H * 4**K - int((row ** 2).sum()), BAL0, 0
    while rest:
        for _ in range(rest % 4):
            row[d] = 2.0 ** j
            d += 1
        rest //= 4
        j += 1
    assert d <= H


def build():
    c = Circuit(hidden=H, n_layers=1, n_heads=1, intermediate=16, vocab=VOCAB)
    E = c.w["model.embed_tokens.weight"]
    for r in range(NR):
        for v in range(16):
            e = E[tok_set(r, v)]
            e[R0 + r] = e[V0 + v] = e[OP0 + SET] = e[OUT0 + 16] = e[NXT0 + 16] = 1.0
        e = E[tok_get(r)]
        e[R0 + r] = e[V0 + 16] = e[OP0 + GET] = e[OUT0 + 16] = e[NXT0 + 16] = 1.0
    e = E[TOK_BOS]
    e[R0 + 8] = e[V0 + 16] = e[OP0 + BOS] = e[OUT0 + 16] = e[NXT0 + 16] = 1.0
    for t in range(TOK_BOS + 1):
        ballast(E[t])
    assert all((E[t] == E[TOK_BOS])[BAL0:].all() for t in range(TOK_BOS))  # one ballast for all
    for v in range(16):
        E[VAL0 + v, OUT0 + v] = 1.0  # output tokens: logit = the OUT field
        E[NVAL0 + v, NXT0 + v] = 1.0  # and = the NXT field
    s = 2.0 ** K  # undo the input norm's 2^-K scale
    p = "model.layers.0.linear_attn."
    qkv = c.w[p + "in_proj_qkv.weight"]
    for r in range(NR):
        qkv[r, R0 + r] = 16 * s  # q: one-hot row r (c = 16 -> l2norm exact)
        qkv[128 + r, R0 + r] = 16 * s  # k
    for v in range(16):
        qkv[256 + v, V0 + v] = 8 * s  # v: nibble 0 = the value (8 >= SiLU's identity threshold)
    for j in range(1, 8):
        qkv[256 + 16 * j, OP0 + SET] = qkv[256 + 16 * j, OP0 + GET] = 8 * s  # nibbles 1..7 = 0
    c.w[p + "conv1d.weight"][:, 0, 3] = 1.0  # the current token only
    c.w[p + "in_proj_b.weight"][0, OP0 + SET] = 16 * s  # beta = 1 on SET
    c.w[p + "in_proj_b.weight"][0, OP0 + GET] = c.w[p + "in_proj_b.weight"][0, OP0 + BOS] = -128 * s
    c.w[p + "in_proj_z.weight"][:, OP0 + SET] = c.w[p + "in_proj_z.weight"][:, OP0 + GET] = 8 * s
    out = c.w[p + "out_proj.weight"]  # a read: 8 hot dims of 32.0 (= 4 * silu(8))
    for v in range(16):
        out[OUT0 + v, v] = 1 / 32
    out[OUT0 + 16, 16] = -1 / 32  # nibble 1 is always hot: clear OUT's none code
    # SwiGLU table NXT = OUT + 1 (mod 16). After the post norm a hot field reads 2^-K and the
    # ballast dim BAL0 reads b = E[BAL0] * 2^-K (a constant). Unit v: gate = 2^(K+8) x[OUT v]
    # - (224 / b) x[BAL0] = 256 - 224 = 32 on a match (SiLU identity), -224 otherwise (SiLU = 0);
    # up = x[BAL0] / (32 b) = 1/32, so silu(gate) * up = 1 exactly.
    b = E[TOK_BOS, BAL0] * 2.0 ** -K
    m = "model.layers.0.mlp."
    for v in range(16):
        c.w[m + "gate_proj.weight"][v, OUT0 + v] = 2.0 ** (K + 8)
        c.w[m + "gate_proj.weight"][v, BAL0] = -224 / b
        c.w[m + "up_proj.weight"][v, BAL0] = 1 / (32 * b)
        c.w[m + "down_proj.weight"][NXT0 + (v + 1) % 16, v] = 1.0
        c.w[m + "down_proj.weight"][NXT0 + 16, v] = -1.0  # clear NXT's none code
    c.check_weights()
    return c


def main():
    c = build()
    m = c.to_hf()
    rng = random.Random(7)
    worst, n_pos = 0, 0
    for trial in range(20):
        seq, regs = [TOK_BOS], {}
        for _ in range(40):
            if regs and rng.random() < 0.5:
                r = rng.choice(sorted(regs))
                seq.append(tok_get(r))
            else:
                r, v = rng.randrange(NR), rng.randrange(16)
                regs[r] = v
                seq.append(tok_set(r, v))
        c.reset()
        cache = None
        regs = {}
        for pos, t in enumerate(seq):
            _, logits = c.step(t)
            with torch.no_grad():  # explicit positions: a cache with no attention layer has no length
                o = m(input_ids=torch.tensor([[t]], device="mps"), past_key_values=cache, use_cache=True,
                      position_ids=torch.tensor([[pos]], device="mps"))
            cache = o.past_key_values
            got = o.logits[0, -1].float().cpu().numpy().astype(np.float64)
            diff = float(np.abs(got - logits).max())
            worst = max(worst, diff)
            n_pos += 1
            if t < 128:
                regs[t // 16] = t % 16
            if t < TOK_BOS:
                want = regs[t // 16] if t < 128 else regs[t - 128]
                top = int(np.argmax(logits[VAL0:VAL0 + 16]))
                nxt = int(np.argmax(logits[NVAL0:NVAL0 + 16]))
                assert top == want and nxt == (want + 1) % 16, (trial, t, top, nxt, want)
        assert worst == 0.0, f"trial {trial}: evaluator vs torch max |diff| {worst}"
    print(f"register file + SwiGLU table: {n_pos} decode positions over 20 sequences; evaluator == "
          f"torch bf16/MPS logits bit for bit (max |diff| {worst}); every read returned the stored "
          f"value and the table its successor")


if __name__ == "__main__":
    main()
