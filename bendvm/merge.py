"""C4: one Qwen3_5ForCausalLM that is both Qwen3.5-0.8B (+ the fused BendLite LoRA) and the engine
(DESIGN.md "C4 plan"). Hidden 4096: dims 0..1023 are Qwen's residual, the rest the engine's.

Layers: the 10 engine layers (engine_circuit.build; zero weights on Qwen dims, so Qwen's embedding
passes through unchanged and is absorbed by the engine's norms: its energy is < 2^-12 of the
engine's H 4^K), then an exit layer (MLP only) that clears every engine dim exactly (EMIT dims only
at Qwen-token positions, QTOK = 1), then Qwen's 24 layers zero-padded to hidden 4608, their norm
weights halved (fp32) for the mean over 4096 dims.
Vocab: Qwen's 248,320 ids, then the engine's tokens 0..END at V0 + t. CAPSTART = '</think>',
EOF = '<|im_end|>', a lexable Qwen token = a text token, any other Qwen token = 'plain' (QTOK only).
lm_head tied to the embedding: Qwen rows score on Qwen dims; an engine token's row carries its
emit pattern * EMIT_EPS on the EMIT dims, whose final-norm gain is 2^12 / EMIT_EPS so an engine
position's logits are the engine's.

Run (end to end): python bendvm/merge.py [N test prompts]
  Qwen writes the program from the chat prompt, '<|im_end|>' ends it, and the same model's
  forward passes then emit the program's trace until END; the result is read from the trace.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine_circuit import CAPSTART, EMIT_EPS, EOF, FIRMWARE, H, MAX_N, N_LAYERS, QWEN_DIMS, S, TEXT0, build  # noqa: E402
from qwen_fused import qwen_state  # noqa: E402
from trace import END  # noqa: E402
from paths import BENDTUNE, qwen_dir  # noqa: E402

EMIT_GAIN = 2.0 ** 12 / EMIT_EPS  # the tied rows carry the emit pattern * EMIT_EPS
N_QWEN_LAYERS = 24
EXIT = N_LAYERS  # the exit layer's index


def to_dev(x, device, dtype=torch.bfloat16):
    """numpy (any float) -> torch on device; the cast happens on the CPU (MPS has no fp64)."""
    t = torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x
    return t.to(dtype).to(device)


def norm_weight(w_qwen):
    """A Qwen RMSNorm weight for hidden H: (1 + w') = (1 + w) / sqrt(H / 1024) on Qwen dims, 0 elsewhere."""
    w = torch.full((H,), -1.0, dtype=torch.float32)
    w[:QWEN_DIMS] = (1.0 + w_qwen.float()) / math.sqrt(H / QWEN_DIMS) - 1.0
    return w


def pad_qwen(k, v):
    """Qwen layer weight -> hidden H (Qwen dims first)."""
    if k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight")):
        return norm_weight(v)
    if v.dim() == 2 and v.shape[1] == QWEN_DIMS and k.endswith(
            ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
             "q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight")):
        out = torch.zeros((v.shape[0], H), dtype=v.dtype)
        out[:, :QWEN_DIMS] = v
        return out
    if v.dim() == 2 and v.shape[0] == QWEN_DIMS and k.endswith(
            ("linear_attn.out_proj.weight", "o_proj.weight", "down_proj.weight")):
        out = torch.zeros((H, v.shape[1]), dtype=v.dtype)
        out[:QWEN_DIMS] = v
        return out
    if v.dim() == 2 and QWEN_DIMS in v.shape:
        raise ValueError(f"unhandled Qwen weight touching the residual: {k} {tuple(v.shape)}")
    return v


def build_merged(device="mps"):
    """-> (model, tokenizer, info dict)."""
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding

    from capture import build_rom
    from lexer_fw import token_words
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(str(qwen_dir()))
    words = token_words(build_rom(lambda i: tok.decode([i]), len(tok)))
    think, im_end = tok.convert_tokens_to_ids("</think>"), tok.convert_tokens_to_ids("<|im_end|>")
    if think in words or im_end in words:
        raise ValueError("CAPSTART / EOF must not be lexable text tokens")
    c, L = build([])
    qcfg, qsd = qwen_state()
    QV = qcfg.vocab_size
    V0, vocab = QV, QV + END + 1
    tc = json.loads((qwen_dir() / "config.json").read_text())["text_config"]
    tc.update(hidden_size=H, num_hidden_layers=N_LAYERS + 1 + N_QWEN_LAYERS, vocab_size=vocab,
              layer_types=["linear_attention"] * (N_LAYERS + 1) + list(tc["layer_types"]),
              tie_word_embeddings=True)
    cfg = Qwen3_5TextConfig(**tc)
    if cfg.rms_norm_eps != c.cfg["eps"] or cfg.intermediate_size != c.cfg["intermediate"]:
        raise ValueError("engine and Qwen disagree on eps / MLP width")
    with torch.device("meta"):
        m = Qwen3_5ForCausalLM(cfg)
    shapes = {k: v.shape for k, v in m.state_dict().items()}
    sd = {}

    # --- engine layers (numpy weights popped as they are converted) ---
    for k in [k for k in c.w if k.startswith("model.layers.")]:
        v = c.w.pop(k)
        sd[k] = to_dev(v, device, torch.float32 if k.endswith(("A_log", "dt_bias")) else torch.bfloat16)
    eng_embed, eng_lm = c.w.pop("model.embed_tokens.weight"), c.w.pop("lm_head.weight")

    # --- the exit layer: zero mixer; MLP unit i clears engine dim d: silu(32) * x_d * (-1/32) ---
    one, qtok = L.f["ONE"][0], L.f["QTOK"][0]
    emit = set(L.f["EK"]) | set(L.f["EB"]) | set(L.f["EP"])
    p = f"model.layers.{EXIT}."
    for k, shp in shapes.items():
        if k.startswith(p):
            sd[k] = torch.zeros(shp, dtype=torch.float32 if k.endswith(("A_log", "dt_bias")) else torch.bfloat16)
    # engine dims that can be nonzero: every field dim, and the ballast dims some row uses
    tmpl_plain, tmpl_text = np.zeros(H), np.zeros(H)
    L.fill(tmpl_plain, None)
    L.fill(tmpl_text, TEXT0, next(iter(words.items())))
    bal = np.array(L.f["BAL"])
    used = np.ones(H, dtype=bool)
    used[:QWEN_DIMS] = False
    used[bal] = ((eng_embed[:, bal] != 0).any(axis=0) | (tmpl_plain[bal] != 0) | (tmpl_text[bal] != 0))
    dims = np.flatnonzero(used)
    if len(dims) > cfg.intermediate_size:
        raise ValueError(f"exit layer: {len(dims)} engine dims > {cfg.intermediate_size} units")
    g, u, d = (torch.zeros(shapes[p + n], dtype=torch.float32) for n in
               ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"))
    for i, dim in enumerate(dims):
        g[i, qtok if dim in emit else one] = 32 * S
        u[i, dim] = S
        d[dim, i] = -1 / 32
    for n, w in (("gate_proj", g), ("up_proj", u), ("down_proj", d)):
        sd[p + f"mlp.{n}.weight"] = w.to(torch.bfloat16)

    # --- Qwen's layers ---
    for k in list(qsd):
        if k.startswith("model.layers."):
            j = int(k.split(".")[2])
            nk = f"model.layers.{N_LAYERS + 1 + j}." + k.split(".", 3)[3]
            sd[nk] = pad_qwen(nk, qsd.pop(k))
    qemb = qsd.pop("model.embed_tokens.weight")
    fw = torch.full((H,), -1.0, dtype=torch.float32)
    fw[:QWEN_DIMS] = norm_weight(qsd.pop("model.norm.weight"))[:QWEN_DIMS]
    fw[sorted(emit)] = EMIT_GAIN - 1.0
    sd["model.norm.weight"] = fw
    if qsd:
        raise ValueError(f"unused Qwen weights: {list(qsd)[:5]}")

    # --- embedding: Qwen rows (Qwen dims + plain / text / CAPSTART / EOF engine fields), engine rows ---
    E = torch.zeros((vocab, H), dtype=torch.bfloat16)
    E[:QV, :QWEN_DIMS] = qemb
    E[:QV, QWEN_DIMS:] = torch.from_numpy(tmpl_plain[QWEN_DIMS:]).to(torch.bfloat16)
    lex = sorted(words)
    for s in range(0, len(lex), 4096):
        block = np.zeros((len(lex[s:s + 4096]), H))
        for r, q in enumerate(lex[s:s + 4096]):
            L.fill(block[r], TEXT0, (q, words[q]))  # a text token
        E[lex[s:s + 4096], QWEN_DIMS:] = torch.from_numpy(block[:, QWEN_DIMS:]).to(torch.bfloat16)
    E[think, QWEN_DIMS:] = torch.from_numpy(eng_embed[CAPSTART, QWEN_DIMS:]).to(torch.bfloat16)
    E[im_end, QWEN_DIMS:] = torch.from_numpy(eng_embed[EOF, QWEN_DIMS:]).to(torch.bfloat16)
    E[V0:] = torch.from_numpy(eng_embed[:END + 1]).to(torch.bfloat16)
    sd["model.embed_tokens.weight"] = E
    sd["lm_head.weight"] = E  # tied: Qwen rows score on Qwen dims (engine dims are 0 after the
    # exit at Qwen positions); engine rows score through their tiny emit pattern
    if E[:QV][:, sorted(emit)].any():
        raise ValueError("Qwen rows must be zero on the EMIT dims")
    if E[:, :QWEN_DIMS][V0:].any() or (eng_embed[:, :QWEN_DIMS] != 0).any():
        raise ValueError("engine rows must be zero on Qwen dims")
    del E, qemb, c  # the CPU copies go once sd's tensors are moved

    missing, unexpected = set(shapes) - set(sd), set(sd) - set(shapes)
    if missing or unexpected:
        raise RuntimeError(f"merged state dict: missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}")
    moved = {}  # one device tensor per source tensor: the tied embedding / lm_head stay one tensor
    for k, v in sd.items():
        if tuple(v.shape) != tuple(shapes[k]):
            raise RuntimeError(f"{k}: {tuple(v.shape)} vs {tuple(shapes[k])}")
        if id(v) not in moved:
            moved[id(v)] = v.to(device)
        sd[k] = moved[id(v)]
    m.load_state_dict(sd, assign=True, strict=True)
    del sd, moved
    if m.lm_head.weight.data_ptr() != m.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("lm_head is not tied to the embedding")
    m.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=cfg, device=device)
    meta = [n for n, t in list(m.named_parameters()) + list(m.named_buffers()) if t.device.type == "meta"]
    if meta:
        raise RuntimeError(f"left on meta: {meta[:5]}")
    m.eval()
    n_params = sum({t.data_ptr(): t.numel() for t in m.parameters()}.values())  # a tied tensor once
    with torch.no_grad():  # nonzero weights: Qwen's (its dims and layers) vs the engine's
        nz = {"qwen": 0, "engine": 0}
        for n, t in m.named_parameters():
            if n == "model.embed_tokens.weight":
                nz["qwen"] += int((t[:, :QWEN_DIMS] != 0).sum())
                nz["engine"] += int((t[:, QWEN_DIMS:] != 0).sum())
            elif n.startswith("model.layers."):
                nz["engine" if int(n.split(".")[2]) <= EXIT else "qwen"] += int((t != 0).sum())
            elif n == "model.norm.weight":
                nz["qwen"] += int((t != 0).sum())
    info = dict(V0=V0, think=think, im_end=im_end, words=words, L=L, exit_dims=len(dims),
                params=n_params, nonzero=nz, build_s=time.time() - t0)
    return m, tok, info


def run(m, info, prompt_ids, max_qwen=600, limit=200_000, device="mps", stamps=None):
    """Feed the prompt, then generate greedily until the engine's END. Checks at every position
    that the exit layer left the engine dims at exactly 0 (EMIT dims excepted at engine / EOF
    positions). stamps: a list that gets time.time() at each generated token (Qwen's, then the
    engine's) and once when the prompt is fed. -> (qwen tokens, engine tokens (local), exit violations)."""
    V0, im_end, L = info["V0"], info["im_end"], info["L"]
    emit = sorted(set(L.f["EK"]) | set(L.f["EB"]) | set(L.f["EP"]))
    keep = torch.zeros(H, dtype=torch.bool, device=device)
    keep[emit] = True
    eng = torch.zeros(H, dtype=torch.bool, device=device)
    eng[QWEN_DIMS:] = True
    cache, qwen, engine, bad = None, [], [], []

    def step(t, pos):
        nonlocal cache
        o = m(input_ids=torch.tensor([[t]], device=device), past_key_values=cache, use_cache=True,
              output_hidden_states=True)
        cache = o.past_key_values
        hx = o.hidden_states[EXIT + 1][0, -1]
        engine_pos = t >= V0 or t == im_end
        must_zero = eng & ~keep if engine_pos else eng
        if bool((hx[must_zero] != 0).any()):
            bad.append((pos, t, int((hx[must_zero] != 0).sum())))
        return int(o.logits[0, -1].argmax())

    with torch.no_grad():
        for pos, t in enumerate(prompt_ids):
            nxt = step(t, pos)
        if stamps is not None:
            stamps.append(time.time())
        pos = len(prompt_ids)
        while len(engine) < limit:
            t = nxt
            if stamps is not None:
                stamps.append(time.time())
            if engine or (qwen and qwen[-1] == im_end):
                if t < V0:
                    raise RuntimeError(f"a Qwen token {t} after the engine started (pos {pos})")
                engine.append(t - V0)
                if t - V0 == END:
                    break
            else:
                if t >= V0:
                    raise RuntimeError(f"engine token {t - V0} while Qwen writes (pos {pos})")
                qwen.append(t)
                if len(qwen) > max_qwen:
                    raise RuntimeError("Qwen did not end its answer")
            nxt = step(t, pos)
            pos += 1
    return qwen, engine, bad


def main():
    from bendlite_ref import BendLiteError, run_source
    from lexer_fw import Stored
    from run_engine import result
    from trace import trace
    args = [a for a in sys.argv[1:] if not a.startswith("--record=")]
    record = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--record=")), None)
    n = int(args[0]) if args else 1
    m, tok, info = build_merged()
    print(f"merged model: hidden {H}, {N_LAYERS} engine + 1 exit + {N_QWEN_LAYERS} Qwen layers, vocab "
          f"{info['V0'] + END + 1}, {info['params'] / 1e9:.2f}B params (nonzero: Qwen {info['nonzero']['qwen'] / 1e9:.3f}B, "
          f"engine {info['nonzero']['engine'] / 1e6:.1f}M), exit clears {info['exit_dims']} "
          f"dims; built in {info['build_s']:.0f} s", flush=True)
    data = [json.loads(x) for x in open(BENDTUNE / "data/test.jsonl")]
    mlx = [json.loads(x) for x in open(BENDTUNE / "logs/tuned400_test.jsonl")]
    nn = tok.encode("\n\n", add_special_tokens=False)
    picked = []
    for r in mlx:
        ids = tok.encode(r["raw"], add_special_tokens=False)
        if r["correct"] and len(ids) + len(nn) < MAX_N and all(i in info["words"] for i in ids):
            picked.append(r)
    picked = picked[::max(1, len(picked) // n)][:n]
    ok = 0
    for r in picked:
        msgs = data[r["i"]]["messages"][:2]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        pids = tok.encode(prompt, add_special_tokens=False)
        if pids[-2:] != [info["think"]] + nn:
            raise ValueError(f"prompt does not end with '</think>\\n\\n': {pids[-4:]}")
        t0 = time.time()
        stamps = []
        qwen, engine, bad = run(m, info, pids, stamps=stamps)
        dt = time.time() - t0
        prog_ids = qwen[:-1]  # without <|im_end|>
        src = tok.decode(prog_ids)
        want_trace = trace(FIRMWARE, Stored(nn + prog_ids, info["words"]))
        try:
            want = run_source(src)[0]
        except BendLiteError:
            want = "ERR"
        got = result(engine)
        good = got == want and engine == want_trace and not bad
        ok += good
        print(f"{r['family']:<16} Qwen wrote {len(prog_ids)} tokens (== MLX: {src.strip() == r['raw'].strip()}); "
              f"engine {len(engine)} tokens (== trace: {engine == want_trace}); result {got} reference {want} "
              f"expected {r['expected']}; exit violations {len(bad)} {bad[:3]}; {dt:.0f} s "
              f"({(len(pids) + len(qwen) + len(engine)) / dt:.1f} tok/s) {'OK' if good else 'FAIL'}", flush=True)
        if record:  # the run, token by token, for the video (render_video.py)
            Path(record).write_text(json.dumps(dict(
                family=r["family"], task=data[r["i"]]["messages"][1]["content"], prompt=prompt,
                prompt_tokens=len(pids), qwen_ids=qwen, qwen_pieces=[tok.decode([t]) for t in qwen],
                engine=engine, t0=t0, stamps=stamps, result=got, reference=want, expected=r["expected"],
                trace_equal=engine == want_trace, program_equal_mlx=src.strip() == r["raw"].strip(),
                exit_violations=len(bad), params=info["params"], firmware_len=len(FIRMWARE),
                recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))
            record = None  # only the first program
    print(f"{ok}/{len(picked)} end to end: Qwen wrote the program, the same model's layers ran it")


if __name__ == "__main__":
    main()
