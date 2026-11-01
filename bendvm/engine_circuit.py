"""C2b: the engine circuit (DESIGN.md "C2 circuit plan"), built in our IR (circuit.py) and checked
position by position against trace.TraceMachine on real traces (teacher-forced).

Built so far: the embedding ROM, layer A (latch heads + byte-assemble head), layer R (the
register file) and the ALU MLPs of layers 1-3 (B-operand select + SHR1; adder nibble pairs; carry
adjust + carry-out + equality). Each residual field's hot dims are 1.0 in the residual; the norms scale by
2^-K (energy H * 4^K at every position, every layer: see Layout).

Run: python bendvm/engine_circuit.py [N programs] [logs.jsonl ...]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit import KDIM, Circuit  # noqa: E402
from firmware2 import FIRMWARE  # noqa: E402
from trace import COMPUTED, END, LAST, MORE, NPC, TraceMachine  # noqa: E402

H, K = 4096, 3  # hidden, energy H * 4^K (norm scale 2^-K)
QWEN_DIMS = 1024  # dims 0..1023: Qwen3.5-0.8B's own residual in the merged model (0 in the engine alone)
EMIT_EPS = 2.0 ** -7  # an engine token's embedding carries its emit pattern * EMIT_EPS (tied rows, merge.py)
NH = 16  # DeltaNet heads per layer (Qwen3.5-0.8B's own count: the merge keeps it)
MEM_LAYER0, MEM_HEADS = 5, 64  # memory: layers 5-8, 16 heads each (8,192 words)
SELECT_LAYER = MEM_LAYER0 + MEM_HEADS // NH - 1  # its MLP selects the value (after all reads)
N_LAYERS = SELECT_LAYER + 2  # + the emission layer (MLP only)
VALUE_SRC = {"ADD": "SUM", "SUB": "SUM", "ADDI": "SUM", "SHR1": "SHR", "MOV": "RSW", "OUT": "RSW",
             "XOR": "XORF", "ROTN": "ROTF",
             "ST": "RTW", "LD": "MEMVAL", "SLTU": "FLAG", "SEQ": "FLAG", "SEQI": "FLAG"}
S = 2.0 ** K  # a weight reading a 1.0 field multiplies by S to undo the norm
CAPSTART, EOF = END + 1, END + 2  # capture tokens; text (program) tokens follow: TEXT0 + i
TEXT0 = END + 3
NBLK, MAX_N = 12, 192  # capture index n = 16 b + r < 192 (DESIGN.md "C3c design")
OPS = sorted({ins[0] for ins in FIRMWARE})
REGWRITE = (COMPUTED - {"ST", "OUT"})  # ops whose value is written to rd at LAST
CONSTWRITE = {"LI", "JAL"}  # ops writing a pc-fixed value at INS


def fields(pc):
    """FIRMWARE[pc] -> dict of decoded fields (None = absent)."""
    op, *a = FIRMWARE[pc]
    f = dict(op=op, rd=None, rs=None, rt=None, imm=0, tgt=0, const=None)
    if op == "LI":
        f.update(rd=a[0], const=a[1] % 2**32)
    elif op in ("MOV", "SHR1"):
        f.update(rd=a[0], rs=a[1])
    elif op in ("ADD", "SUB", "SLTU", "SLEU", "SEQ", "XOR", "ROTN"):
        f.update(rd=a[0], rs=a[1], rt=a[2])
    elif op in ("SEQI", "ADDI", "LD"):
        f.update(rd=a[0], rs=a[1], imm=a[2] % 2**32)
    elif op == "ST":
        f.update(rs=a[0], imm=a[1] % 2**32, rt=a[2])
    elif op in ("BEQZ", "BNEZ"):
        f.update(rs=a[0], tgt=a[1])
    elif op in ("BEQI", "BNEI"):
        f.update(rs=a[0], imm=a[1] % 2**32, tgt=a[2])
    elif op in ("BEQ", "BNE"):
        f.update(rs=a[0], rt=a[1], tgt=a[2])
    elif op == "JMP":
        f.update(tgt=a[0])
    elif op == "JAL":
        f.update(rd=a[0], const=pc + 1, tgt=a[1])
    elif op in ("JR", "OUT"):
        f.update(rs=a[0])
    elif op not in ("HALT", "FAIL"):
        raise ValueError(op)
    return f


def nibbles(v, n):
    return [(v >> (4 * j)) & 15 for j in range(n)]


class Layout:
    """Residual dims by name. Reserves are dims at 1.0 in every engine token's embedding that a
    gated output cancels (writes -1) when it fires, so energy is the same whether it fires."""

    def __init__(self):
        self.next, self.f, self.reserves = 0, {}, []

    def alloc(self, name, n):
        self.f[name] = range(self.next, self.next + n)
        self.next += n
        assert self.next <= H, f"out of hidden dims at {name}"
        return self.f[name]

    def reserve(self, name, n):
        r = self.alloc(name, n)
        self.reserves.append(r)
        return r


def build(text_words=()):
    """text_words: [(qwen id, capture2.rom_words)] for the program tokens this model knows (the
    merged model will know all 122k; a test build only those of its programs). Local id of the
    i-th: TEXT0 + i."""
    from firmware2 import LABELS
    from lexer_fw import PATTERNS, pattern_label
    VOCAB = TEXT0 + len(text_words)
    L = Layout()
    L.alloc("QWEN", QWEN_DIMS)  # never read or written by the engine (merge.py)
    # --- embedding fields (inputs) ---
    kind = L.alloc("KIND", 6)  # INS MORE_1 MORE_2 MORE_3 LAST END
    INS_, LAST_, END_ = kind[0], kind[4], kind[5]
    op_ = L.alloc("OP", len(OPS))
    rd, rs, rt = L.alloc("RD", 17), L.alloc("RS", 17), L.alloc("RT", 17)  # 16 = none
    imm, pc1, tgt = L.alloc("IMM", 128), L.alloc("PC1", 48), L.alloc("TGT", 48)
    const = L.alloc("CONST", 128)
    # a byte field per value-token kind (MORE_1..3, LAST): lo nibble 0..15, hi nibble 16..31;
    # separate fields because a linear projection can't AND "is MORE_d" with "digit"
    bytef = [L.alloc(f"BYTE{d}", 32) for d in range(4)]  # 0 = LAST, d = MORE_d
    # --- layer A outputs ---
    lat0 = L.alloc("LAT0", 128)  # OP RD RS RT + 4 constant: 8 hot
    lat1 = L.alloc("LAT1", 128)  # IMM: 8 hot
    lat2 = L.alloc("LAT2", 128)  # TGT PC1 + 2 constant: 8 hot
    lat_res = [L.reserve(f"LAT{i}_RES", 8) for i in range(3)]
    word = L.alloc("WORD", 128)  # the assembled value at LAST
    word_res = L.reserve("WORD_RES", 8)
    # --- layer R outputs ---
    rsw, rtw = L.alloc("RSW", 128), L.alloc("RTW", 128)
    rsw_res, rtw_res = L.reserve("RSW_RES", 8), L.reserve("RTW_RES", 8)
    # --- ALU fields (MLPs of layers 1-3) ---
    one = L.alloc("ONE", 1)[0]  # 1.0 at every position: the AND gadget's constant
    bsel, bsel_res = L.alloc("BSEL", 128), L.reserve("BSEL_RES", 8)  # the adder's B operand
    shr, shr_res = L.alloc("SHR", 128), L.reserve("SHR_RES", 8)  # rs >> 1
    sm = L.alloc("SUM", 128)  # nibble sums, carries applied in place by layer 3
    gpe = L.alloc("GPE", 48)  # per nibble pairs: G (generate), P (propagate), E (equal)
    gpe_res = L.reserve("GPE_RES", 32)
    cout = L.alloc("COUT", 2)  # [no carry out, carry out]: starts [1, 0]
    eqf = L.alloc("EQ", 2)  # [not equal, equal]: starts [1, 0]
    L.alloc("XORF", 128), L.reserve("XORF_RES", 8)  # rs ^ BSEL (the adder's pair units)
    L.alloc("ROTF", 128), L.reserve("ROTF_RES", 8)  # rs rotated left by 4 (rt mod 8) bits
    # --- memory: address one-hots (layer 4 MLP), the read (layers 5-8) ---
    L.alloc("ROW", 128), L.reserve("ROW_RES", 1)  # address & 127
    L.alloc("HEAD", 64), L.reserve("HEAD_RES", 1)  # address >> 7 (none if address >= 8192)
    L.alloc("MEMVAL", 128), L.reserve("MEMVAL_RES", 8)
    # --- select (MLP of SELECT_LAYER) and emit (the last MLP) ---
    L.alloc("V", 128), L.reserve("V_RES", 8)
    L.alloc("EK", 6)  # the emitted token's kind: INS MORE_1 MORE_2 MORE_3 LAST END
    L.alloc("EB", 32)  # the emitted byte (lo nibble, hi nibble)
    L.alloc("EP", 48)  # the emitted pc (3 nibbles)
    L.reserve("EMIT_RES", 4)  # EK, then up to 3 digit dims
    # --- capture (DESIGN.md "C3c design") ---
    ck = L.alloc("CKIND", 3)  # TEXT CAPSTART EOF
    qtok = L.alloc("QTOK", 1)[0]  # a Qwen token whose position Qwen answers (all but EOF): merge.py
    # the stored words of a text / EOF token. They overlay the instruction-only input fields
    # (never set in a text / EOF row; their readers ignore them off INS: zero keys, beta 0)
    pool = [d for f in (op_, rd, rs, rt, imm, pc1, tgt, const) for d in f]
    pool += list(L.alloc("ROMW_X", 512 - len(pool)))
    romw = [pool[128 * k:128 * (k + 1)] for k in range(4)]
    for k in range(4):
        L.f[f"ROMW{k}"] = romw[k]
    for name in ("CNT1", "CNT2", "FX"):  # tiny exact powers of 2 (<= 2^-5: absorbed by the norms)
        L.alloc(name, 1)
    L.reserve("CNT_RES", 4)  # 2 reference dims per counter
    L.alloc("FLAG", NBLK), L.reserve("FLAG_RES", NBLK)  # FLAG b: n >= 16 b (FLAG 0 unused)
    L.alloc("BLK", NBLK), L.reserve("BLK_RES", 1)
    L.alloc("FINE", 16), L.reserve("FINE_RES", 16)  # FINE k: r >= k (FINE 0 unused)
    L.alloc("ROWC", 128), L.alloc("HC", 2), L.reserve("IDX_RES", 2)
    bal0 = L.alloc("BAL", H - L.next)

    c = Circuit(hidden=H, n_layers=N_LAYERS, n_heads=NH, intermediate=3584, vocab=VOCAB)  # Qwen's own MLP width
    E = c.w["model.embed_tokens.weight"]
    def fill(e, t, tw=None):
        """Write token t's embedding fields into the zero row e. t: a local id (tw = (qwen id, rom
        words) for a text token), or None: a Qwen token unknown to the engine (merge.py)."""
        if t is None:
            e[qtok] = 1
        elif t < NPC:
            f = fields(t)
            e[kind[0]] = 1
            e[op_[OPS.index(f["op"])]] = 1
            for fld, rng in (("rd", rd), ("rs", rs), ("rt", rt)):
                e[rng[16 if f[fld] is None else f[fld]]] = 1
            for j, d in enumerate(nibbles(f["imm"], 8)):
                e[imm[16 * j + d]] = 1
            for j, d in enumerate(nibbles(t + 1, 3)):
                e[pc1[16 * j + d]] = 1
            for j, d in enumerate(nibbles(f["tgt"], 3)):
                e[tgt[16 * j + d]] = 1
            if f["const"] is not None:
                for j, d in enumerate(nibbles(f["const"], 8)):
                    e[const[16 * j + d]] = 1
        elif t >= CAPSTART:
            if t == CAPSTART:
                e[ck[1]] = e[qtok] = 1
            else:
                e[qtok] = int(t != EOF)
                e[ck[0 if t >= TEXT0 else 2]] = 1
                if t == EOF:
                    ws = [LABELS["lex_eof"], 0, 0, 0]
                else:
                    pat, pay = tw[1]
                    ws = [LABELS[pattern_label(pat)]] + [pay[k] if k < len(pay) else 0 for k in range(3)]
                for k in range(4):
                    for j, d in enumerate(nibbles(ws[k], 8)):
                        e[romw[k][16 * j + d]] = 1
        elif t < END:
            b = (t - MORE) % 256
            d = 0 if t >= LAST else (t - MORE) // 256 + 1
            e[kind[d if d else 4]] = 1
            e[bytef[d][b & 15]] = e[bytef[d][16 + (b >> 4)]] = 1
        elif t == END:
            e[END_] = 1
        for r in L.reserves:
            e[list(r)] = 1
        e[one] = e[cout[0]] = e[eqf[0]] = 1
        ballast(e, bal0)

    for t in range(VOCAB):
        fill(E[t], t, text_words[t - TEXT0] if t >= TEXT0 else None)
    L.fill = fill

    # ===== layer A: latch heads 0-2 (row 0, written at INS), assemble head 3 (row 0 at LAST)
    p = "model.layers.0.linear_attn."
    qkv, z, b, cw, out = (c.w[p + n] for n in ("in_proj_qkv.weight", "in_proj_z.weight",
                                                "in_proj_b.weight", "conv1d.weight", "out_proj.weight"))
    q0, k0, v0 = 0, KDIM * NH, 2 * KDIM * NH  # channel offsets of q, k, v
    cw[:, 0, 3] = 1.0  # tap 0 (the current token) unless set otherwise below
    lat_src = [  # (latch head, [(residual dim, value dim)])
        (0, [(op_[i], i) for i in range(len(OPS))]
         + [(rd[i], 32 + i) for i in range(17)] + [(rs[i], 49 + i) for i in range(17)]
         + [(rt[i], 66 + i) for i in range(17)]),
        (1, [(imm[i], i) for i in range(128)]),
        (2, [(tgt[i], i) for i in range(48)] + [(pc1[i], 48 + i) for i in range(48)]),
    ]
    const_dims = {0: [124, 125, 126, 127], 1: [], 2: [126, 127]}  # pad every latch row to 8 hot
    assert len(OPS) <= 32
    for hd, pairs in lat_src:
        assert all(vd < 128 and vd not in const_dims[hd] for _, vd in pairs), "latch row overflow"
        for src, vd in pairs:
            qkv[v0 + KDIM * hd + vd, src] = 8 * S
        for vd in const_dims[hd]:
            qkv[v0 + KDIM * hd + vd, list(kind)] = 8 * S  # every token kind: a constant 8
        qkv[k0 + KDIM * hd + 0, INS_] = 16 * S  # key row 0 at INS (else k = 0)
        qkv[q0 + KDIM * hd + 0, list(kind)] = 16 * S  # query row 0 everywhere
        b[hd, INS_] = 16 * S  # beta 1 at INS
        b[hd, list(kind[1:])] = -128 * S
        z[KDIM * hd:KDIM * (hd + 1), list(kind)] = 8 * S  # read always (32.0 per hot dim)
        dst = {0: lat0, 1: lat1, 2: lat2}[hd]
        for vd in range(128):
            out[dst[vd], KDIM * hd + vd] = 1 / 32
        out[list(lat_res[hd]), KDIM * hd:KDIM * (hd + 1)] = -1 / 256  # the row sums to 8 * 32: -1 per reserve
    # assemble head 3: value channel (byte d from the end, nibble h, digit) at tap d
    hd = 3
    for d in range(4):
        for h in range(2):
            for digit in range(16):
                ch = v0 + KDIM * hd + 32 * d + 16 * h + digit  # nibble 2d + h
                cw[ch, 0, :] = 0
                cw[ch, 0, 3 - d] = 1.0
                qkv[ch, bytef[d][16 * h + digit]] = 8 * S  # LAST at tap 0, MORE_d at tap d
                if d and digit == 0:  # the 0-code when t-d is not a MORE_d
                    qkv[ch, [kind[i] for i in range(6) if i != d]] = 8 * S
    # (d = 0 channels also see a MORE token's byte at MORE positions: there beta = 0 and z = 0)
    qkv[k0 + KDIM * hd + 0, LAST_] = 16 * S
    qkv[q0 + KDIM * hd + 0, LAST_] = 16 * S
    b[hd, LAST_] = 16 * S
    b[hd, [kind[i] for i in range(6) if i != 4]] = -128 * S
    z[KDIM * hd:KDIM * (hd + 1), LAST_] = 8 * S
    for vd in range(128):
        out[word[vd], KDIM * hd + vd] = 1 / 32
    for j in range(8):  # each nibble sums to 32 when read: cancel one reserve dim per nibble
        out[word_res[j], KDIM * hd + 16 * j:KDIM * hd + 16 * j + 16] = -1 / 32

    # ===== layer R: register file, heads 0 (read rs) and 1 (read rt); heads 2, 3 unused
    p = "model.layers.1.linear_attn."
    qkv, z, b, cw, out = (c.w[p + n] for n in ("in_proj_qkv.weight", "in_proj_z.weight",
                                                "in_proj_b.weight", "conv1d.weight", "out_proj.weight"))
    cw[:, 0, 3] = 1.0
    L_RD = [lat0[32 + i] for i in range(16)]
    L_RS = [lat0[49 + i] for i in range(16)]
    L_RT = [lat0[66 + i] for i in range(16)]
    L_OP = [lat0[i] for i in range(len(OPS))]
    for hd, qsrc, dst, res in ((0, L_RS, rsw, rsw_res), (1, L_RT, rtw, rtw_res)):
        for r in range(16):
            qkv[k0 + KDIM * hd + r, L_RD[r]] = 16 * S  # key: the latched rd
            qkv[q0 + KDIM * hd + r, qsrc[r]] = 16 * S  # query: the latched rs / rt
        for vd in range(128):
            qkv[v0 + KDIM * hd + vd, word[vd]] = 8 * S
            qkv[v0 + KDIM * hd + vd, const[vd]] = 8 * S
        # beta: (LAST and latched op in REGWRITE) or (INS and op in LI / JAL)
        b[hd, LAST_] = 256 * S
        for i, o in enumerate(OPS):
            if o in REGWRITE:
                b[hd, L_OP[i]] = 256 * S
            if o in CONSTWRITE:
                b[hd, op_[i]] = 512 * S
        b[hd, list(kind)] += -384 * S
        z[KDIM * hd:KDIM * (hd + 1), list(kind)] = 8 * S
        for vd in range(128):
            out[dst[vd], KDIM * hd + vd] = 1 / 32
        for j in range(8):
            out[res[j], KDIM * hd + 16 * j:KDIM * hd + 16 * j + 16] = -1 / 32
    for hd in (2, 3):  # unused heads in layer R: beta 0
        b[hd, list(kind)] = -128 * S
    alu(c, L, op_, lat0, lat1, rsw, rtw)
    memory(c, L, lat0)
    capture_layers(c, L)
    select_emit(c, L, lat0, lat2)
    # tied output rows: an engine token's embedding also carries its emit pattern * EMIT_EPS (tiny:
    # absorbed by the norms; nothing reads EK / EB / EP but the logits), so the merged model's
    # lm_head can be its embedding
    emit_dims = list(L.f["EK"]) + list(L.f["EB"]) + list(L.f["EP"])
    E[:END + 1, emit_dims] += EMIT_EPS * c.w["lm_head.weight"][:END + 1][:, emit_dims]
    # every head is cleared at CAPSTART (decay exactly 0) and kept otherwise: in the merged model
    # the prompt runs through the engine first, and nothing before CAPSTART may reach the run
    cap = ck[1]
    for i in range(N_LAYERS):
        p = f"model.layers.{i}.linear_attn."
        c.w[p + "A_log"][:] = 0.0
        c.w[p + "in_proj_a.weight"][:, one] = -200 * S
        c.w[p + "in_proj_a.weight"][:, cap] = 400 * S
    c.check_weights()
    return c, L


class Mlp:
    """Allocates SwiGLU units of one layer's MLP as exact AND gadgets: a unit fires (silu(gate) *
    up = 1 exactly) iff all its n features are 1: gate = 256 (sum - n) + 128 is 128 or <= -128
    (SiLU exactly 0); up = 1/128. A feature is a residual dim at 0 / 1.0, or a list of dims of
    which at most one is 1 (an OR over one-hot codes)."""

    def __init__(self, c, layer, one):
        p = f"model.layers.{layer}.mlp."
        self.g, self.u, self.d = (c.w[p + n] for n in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"))
        self.one, self.n = one, 0

    def unit(self, feats, outs, negs=(), up=None):
        """negs: features that must be 0 (each counts as 1 - f); up: {dim: weight} instead of
        the constant 1/128 (a gated pass-through: output = 128 * up exactly when it fires)."""
        i = self.n
        self.n += 1
        assert i < self.g.shape[0], "out of MLP units"
        for f in feats:
            self.g[i, f if isinstance(f, list) else [f]] += 256 * S
        for f in negs:
            self.g[i, f if isinstance(f, list) else [f]] -= 256 * S
        self.g[i, self.one] += (-256 * (len(feats) + len(negs)) + 128 + 256 * len(negs)) * S
        for dim, w in (up or {self.one: 1 / 128}).items():
            self.u[i, dim] = S * w
        for dim, v in outs.items():
            self.d[dim, i] += v


def alu(c, L, op_, lat0, lat1, rsw, rtw):
    one = L.f["ONE"][0]
    L_OP = [lat0[i] for i in range(len(OPS))]
    bsel, sm, gpe, cout, eqf = (L.f[n] for n in ("BSEL", "SUM", "GPE", "COUT", "EQ"))
    shr = L.f["SHR"]
    bsel_res, shr_res, gpe_res = L.f["BSEL_RES"], L.f["SHR_RES"], L.f["GPE_RES"]
    # class features need one dim each: a unit's features are single dims, so a class flag is
    # one unit per op (each op is one latched one-hot dim)
    m1 = Mlp(c, 1, one)
    classes = [  # BSEL = rt | ~rt | imm by the latched op's class
        (("ADD", "SEQ", "BEQ", "BNE", "XOR"), rtw, False),
        (("SUB", "SLTU"), rtw, True),
        (("ADDI", "LD", "ST", "SEQI", "BEQI", "BNEI", "BEQZ", "BNEZ"), lat1, False),
    ]
    for ops_, src, comp in classes:
        cls = [L_OP[OPS.index(o)] for o in ops_ if o in OPS]
        for j in range(8):
            for dgt in range(16):
                out = dgt ^ 15 if comp else dgt
                m1.unit([cls, src[16 * j + dgt]], {bsel[16 * j + out]: 1.0, bsel_res[j]: -1.0})
    for j in range(8):  # SHR = rs >> 1: nibble j = (a_j >> 1) | (a_{j+1} & 1) << 3
        for k in range(8):
            hi_bits = (0, 1) if j < 7 else (0,)
            for hi in hi_bits:
                feats = [[rsw[16 * j + 2 * k], rsw[16 * j + 2 * k + 1]]]
                if j < 7:
                    feats.append([rsw[16 * (j + 1) + x] for x in range(16) if x & 1 == hi])
                m1.unit(feats, {shr[16 * j + (k | (hi << 3))]: 1.0, shr_res[j]: -1.0})
    rotf, rotf_res = L.f["ROTF"], L.f["ROTF_RES"]
    for k in range(8):  # ROTN: nibble j of rs moves to nibble (j + k) mod 8, k = rt mod 8
        for j in range(8):
            for d in range(16):
                m1.unit([[rtw[k], rtw[k + 8]], rsw[16 * j + d]],
                        {rotf[16 * ((j + k) % 8) + d]: 1.0, rotf_res[(j + k) % 8]: -1.0})
    # layer 2: the adder's nibble pairs: S_j = a + b mod 16; G (a + b >= 16), P (a + b = 15),
    # E (a = b), each a one-hot pair; A = rs, B = BSEL
    m2 = Mlp(c, 2, one)
    for j in range(8):
        for a in range(16):
            for b_ in range(16):
                t = a + b_
                outs = {sm[16 * j + t % 16]: 1.0, gpe[6 * j + int(t >= 16)]: 1.0,
                        gpe[6 * j + 2 + int(t == 15)]: 1.0, gpe[6 * j + 4 + int(a == b_)]: 1.0,
                        L.f["XORF"][16 * j + (a ^ b_)]: 1.0, L.f["XORF_RES"][j]: -1.0}
                for r in range(4):
                    outs[gpe_res[4 * j + r]] = -1.0
                m2.unit([rsw[16 * j + a], bsel[16 * j + b_]], outs)
    # layer 3: carries. carry into nibble i: some j < i generates (or the carry-in at j = -1)
    # and every nibble between propagates; the terms are mutually exclusive, so each unit moves
    # S_i's hot digit by one (energy unchanged). Carry-in = 1 for SUB / SLTU.
    m3 = Mlp(c, 3, one)
    cin_ops = [L_OP[OPS.index(o)] for o in ("SUB", "SLTU") if o in OPS]
    G = [gpe[6 * j + 1] for j in range(8)]
    P = [gpe[6 * j + 3] for j in range(8)]
    for i in range(9):  # i = 8: the carry out
        for jj in range(-1, i):
            props = P[jj + 1:i]
            starts = [[G[jj]]] if jj >= 0 else [[o] for o in cin_ops]
            for st in starts:
                if i == 8:
                    m3.unit(st + props, {cout[0]: -1.0, cout[1]: 1.0})
                    continue
                for x in range(16):
                    m3.unit(st + props + [sm[16 * i + x]], {sm[16 * i + x]: -1.0, sm[16 * i + (x + 1) % 16]: 1.0})
    m3.unit([gpe[6 * j + 5] for j in range(8)], {eqf[0]: -1.0, eqf[1]: 1.0})  # all nibbles equal
    MLP3_USED[0] = m3.n
    print(f"ALU MLP units: layer 1 {m1.n}, layer 2 {m2.n}, layer 3 {m3.n}")


def ballast(e, dims):
    """Top e's energy up to H * 4^K with powers of 2 (base-4 digits of the rest)."""
    rest = H * 4**K - int(round(float((e ** 2).sum())))
    assert rest >= 0
    d, j = 0, 0
    while rest:
        for _ in range(rest % 4):
            e[dims[d]] = 2.0 ** j
            d += 1
        rest //= 4
        j += 1


def decode_word(x, rng):
    """8 one-hot nibbles at 1.0 -> the value (or None if a nibble is not exactly one-hot 1.0)."""
    v = 0
    for j in range(8):
        nib = x[list(rng)[16 * j:16 * j + 16]]
        if sorted(nib.tolist()) != [0.0] * 15 + [1.0]:
            return None
        v += int(np.argmax(nib)) << (4 * j)
    return v


def memory(c, L, lat0):
    one = L.f["ONE"][0]
    sm, row, head, word, memv = (L.f[n] for n in ("SUM", "ROW", "HEAD", "WORD", "MEMVAL"))
    row_res, head_res, memv_res = L.f["ROW_RES"][0], L.f["HEAD_RES"][0], L.f["MEMVAL_RES"]
    m4 = Mlp(c, 4, one)

    def nib(j, ds):
        return [sm[16 * j + d] for d in ds]
    for a in range(16):  # ROW = n0 + 16 (n1 & 7)
        for b_ in range(8):
            m4.unit([sm[a], nib(1, (b_, b_ + 8))], {row[a + 16 * b_]: 1.0, row_res: -1.0})
    zero_hi = [sm[16 * j] for j in range(4, 8)]  # address < 8192: nibbles 4..7 are 0
    for h1 in range(2):  # HEAD = (n1 >> 3) + 2 n2 + 32 n3, n3 in {0, 1}
        for n2 in range(16):
            for n3 in range(2):
                m4.unit([nib(1, range(8 * h1, 8 * h1 + 8)), sm[32 + n2], sm[48 + n3]] + zero_hi,
                        {head[h1 + 2 * n2 + 32 * n3]: 1.0, head_res: -1.0})
    LAST_ = L.f["KIND"][4]
    st = lat0[OPS.index("ST")]
    q0, k0, v0 = 0, KDIM * NH, 2 * KDIM * NH
    for layer in range(MEM_LAYER0, MEM_LAYER0 + MEM_HEADS // NH):
        p = f"model.layers.{layer}.linear_attn."
        qkv, z, b, cw, out = (c.w[p + n] for n in ("in_proj_qkv.weight", "in_proj_z.weight",
                                                    "in_proj_b.weight", "conv1d.weight", "out_proj.weight"))
        cw[:, 0, 3] = 1.0
        for hd in range(NH):
            g = NH * (layer - MEM_LAYER0) + hd
            for r in range(128):
                qkv[q0 + KDIM * hd + r, row[r]] = 16 * S
                qkv[k0 + KDIM * hd + r, row[r]] = 16 * S
            for vd in range(128):
                qkv[v0 + KDIM * hd + vd, word[vd]] = 8 * S
            # write at LAST of an ST into the addressed head; read (z) wherever HEAD = g
            b[hd, [LAST_, st, head[g]]] = 256 * S
            b[hd, one] = -640 * S
            if g >= 48:  # the token store: region k = (g - 48) // 4, index n >> 7 = (g - 48) % 4
                kreg, hc = (g - 48) // 4, (g - 48) % 4
                if hc < 2:  # n < 192 uses 2 heads per region
                    for r in range(128):
                        qkv[k0 + KDIM * hd + r, L.f["ROWC"][r]] = 16 * S
                    for vd in range(128):
                        qkv[v0 + KDIM * hd + vd, L.f["ROMW%d" % kreg][vd]] = 8 * S
                    b[hd, [L.f["CKIND"][0], L.f["CKIND"][2], L.f["HC"][hc]]] = 384 * S
            z[KDIM * hd:KDIM * (hd + 1), head[g]] = 8 * S
            for vd in range(128):
                out[memv[vd], KDIM * hd + vd] = 1 / 32
            for j in range(8):
                out[memv_res[j], KDIM * hd + 16 * j:KDIM * hd + 16 * j + 16] = -1 / 32
    MLP4_USED[0] = m4.n
    print(f"memory: layer 4 MLP {m4.n} units; layers {MEM_LAYER0}-{SELECT_LAYER}, {MEM_HEADS} heads")


def select_emit(c, L, lat0, lat2):
    one = L.f["ONE"][0]
    kind = L.f["KIND"]
    L_OP = {o: lat0[i] for i, o in enumerate(OPS)}
    V, V_res = L.f["V"], L.f["V_RES"]
    cout, eqf = L.f["COUT"], L.f["EQ"]
    m = Mlp(c, SELECT_LAYER, one)
    srcs = {}
    for o, src in VALUE_SRC.items():
        if o in OPS:
            srcs.setdefault(src, []).append(L_OP[o])
    for src, ops_ in srcs.items():
        if src == "FLAG":
            continue
        for j in range(8):
            for d in range(16):
                m.unit([ops_, L.f[src][16 * j + d]], {V[16 * j + d]: 1.0, V_res[j]: -1.0})

    def word(v):
        return {V[16 * j + d]: 1.0 for j, d in enumerate(nibbles(v, 8))}
    allres = {V_res[j]: -1.0 for j in range(8)}
    if "SLTU" in OPS:  # rs < rt = no carry out of rs + ~rt + 1
        for cbit in (0, 1):
            m.unit([L_OP["SLTU"], cout[cbit]], {**word(1 - cbit), **allres})
    eq_ops = [L_OP[o] for o in ("SEQ", "SEQI") if o in OPS]
    for e in (0, 1):
        m.unit([eq_ops, eqf[e]], {**word(e), **allres})
    print(f"select: MLP {SELECT_LAYER} {m.n} units")

    # --- emission: EK + (EB byte | EP pc) ---
    EK, EB, EP, er = L.f["EK"], L.f["EB"], L.f["EP"], L.f["EMIT_RES"]
    e = Mlp(c, N_LAYERS - 1, one)
    INS_, LAST_, END_ = kind[0], kind[4], kind[5]
    value_ops = [L_OP[o] for o in VALUE_SRC if o in OPS]

    def emit_byte(feats, d_next, lo, hi):
        """emit MORE_d_next (or LAST if d_next == 0) with byte (hi, lo)"""
        ek = EK[4] if d_next == 0 else EK[d_next]
        e.unit(feats, {ek: 1.0, EB[lo]: 1.0, EB[16 + hi]: 1.0, er[0]: -1.0, er[1]: -1.0, er[2]: -1.0})
    # at INS of a value op: the first byte; n = 1 + the index of the highest non-zero byte
    for n in range(1, 5):
        zeros = [V[16 * j] for j in range(2 * n, 8)]
        for hi in range(16):
            for lo in range(16):
                if n > 1 and hi == 0 and lo == 0:
                    continue
                emit_byte([INS_, value_ops, V[16 * (2 * n - 2) + lo], V[16 * (2 * n - 1) + hi]] + zeros,
                          n - 1, lo, hi)
    # at MORE_d: byte d - 1
    for d in (1, 2, 3):
        for hi in range(16):
            for lo in range(16):
                emit_byte([kind[d], V[16 * (2 * d - 2) + lo], V[16 * (2 * d - 1) + hi]], d - 1, lo, hi)

    # the next pc: from PC1 (latched), TGT (latched) or rs (JR)
    PC1 = [lat2[48 + i] for i in range(48)]
    TGT = [lat2[i] for i in range(48)]
    RSWP = [L.f["RSW"][i] for i in range(48)]  # a pc in a register: nibbles 0..2

    def ops(*names):
        return [L_OP[o] for o in names if o in OPS]
    beq, bne = ops("BEQZ", "BEQI", "BEQ"), ops("BNEZ", "BNEI", "BNE")
    cases = [  # (features, source)
        ([LAST_], PC1),
        ([INS_, ops("LI")], PC1),
        ([INS_, beq, eqf[0]], PC1),
        ([INS_, bne, eqf[1]], PC1),
        ([INS_, ops("JMP", "JAL")], TGT),
        ([INS_, beq, eqf[1]], TGT),
        ([INS_, bne, eqf[0]], TGT),
        ([INS_, ops("JR")], RSWP),
    ]
    for feats, src in cases:
        e.unit(feats, {EK[0]: 1.0, er[0]: -1.0})
        for j in range(3):
            for d in range(16):
                e.unit(feats + [src[16 * j + d]], {EP[16 * j + d]: 1.0, er[1 + j]: -1.0})
    e.unit([INS_, ops("HALT", "FAIL")], {EK[5]: 1.0, er[0]: -1.0})
    e.unit([END_], {EK[5]: 1.0, er[0]: -1.0})
    eof = L.f["CKIND"][2]  # the end of the program: the engine starts at INS(0)
    e.unit([eof], {EK[0]: 1.0, EP[0]: 1.0, EP[16]: 1.0, EP[32]: 1.0, er[0]: -1.0, er[1]: -1.0,
                   er[2]: -1.0, er[3]: -1.0})
    print(f"emit: MLP {N_LAYERS - 1} {e.n} units")

    # untied lm_head: an engine token's row is its emit pattern (kind weighted 4)
    VOCAB = c.cfg["vocab"]
    lm = np.zeros((VOCAB, H))
    for t in range(END + 1):  # engine tokens only: capture / text tokens are never emitted here
        if t < NPC:
            lm[t, EK[0]] = 4
            for j, d in enumerate(nibbles(t, 3)):
                lm[t, EP[16 * j + d]] = 1
        elif t < END:
            b = (t - MORE) % 256
            dk = 4 if t >= LAST else (t - MORE) // 256 + 1
            lm[t, EK[dk]] = 4
            lm[t, EB[b & 15]] = lm[t, EB[16 + (b >> 4)]] = 1
        else:
            lm[t, EK[5]] = 4
    c.w["lm_head.weight"] = lm


def capture_layers(c, L):
    """Counters A1 (layer 0) / A2 (layer 2), block flags (layers 1, 3), block one-hot + the
    rescaled fine value (MLP 3), fine flags per block (layer 4), the index (MLP 4)."""
    one, ck = L.f["ONE"][0], L.f["CKIND"]
    TEXT_, CAP_, EOF_ = ck
    textlike = [TEXT_, EOF_]
    cnt_res, flag, flag_res = L.f["CNT_RES"], L.f["FLAG"], L.f["FLAG_RES"]
    q0, k0, v0 = 0, KDIM * NH, 2 * KDIM * NH

    def wts(layer):
        p = f"model.layers.{layer}.linear_attn."
        return [c.w[p + n] for n in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight",
                                     "in_proj_a.weight", "A_log", "out_proj.weight")]

    def counter(layer, hd, halve_b, dst, res):
        """A halving row: (count, R, R); reset to (12, R, R) at CAPSTART; reads 1.5 2^(-7-halvings)
        (the factor 1.5 keeps every value in the readout chain off 2^-126, where MPS's flush to
        zero would be decided by fp32 noise)."""
        qkv, z, b, a, alog, out = wts(layer)
        qkv[q0 + KDIM * hd, one] = qkv[k0 + KDIM * hd, one] = 16 * S
        qkv[v0 + KDIM * hd + 0, CAP_] = 12 * S
        qkv[v0 + KDIM * hd + 1, [TEXT_, EOF_, CAP_]] = 2.0 ** 16 * S
        qkv[v0 + KDIM * hd + 2, [TEXT_, EOF_, CAP_]] = 2.0 ** 16 * S
        for dim, w in halve_b.items():
            b[hd, dim] += w * S
        z[KDIM * hd:KDIM * (hd + 1), one] = 8 * S
        out[dst, KDIM * hd + 0] = 1.0  # 64 (c / R) 2^-k = 1.5 2^(-7-k); the refs (64) are not output

    def step_flag(layer, hd, src, m, dst, res, key=None):
        """beta = [n >= m] on src = 1.5 2^(-8-n): b = 256 - 128 2^(8+m) src; cleared (decay 0) at
        CAPSTART; writes 8 hot dims (8.0) to row 0 (or row = key's one-hot); reads 1.0 to dst."""
        qkv, z, b, a, alog, out = wts(layer)
        if key is None:
            qkv[q0 + KDIM * hd, one] = qkv[k0 + KDIM * hd, one] = 16 * S
        else:
            for r, kd in enumerate(key):
                qkv[q0 + KDIM * hd + r, kd] = qkv[k0 + KDIM * hd + r, kd] = 16 * S
        for vd in range(8):
            qkv[v0 + KDIM * hd + 120 + vd, one] = 8 * S
        b[hd, one] = 256 * S
        b[hd, src] = -128 * 2.0 ** (8 + m) * S
        alog[hd] = 0.0
        a[hd, one] = -200 * S
        a[hd, CAP_] = 400 * S  # CAPSTART: +200, decay exactly 0 (clear)
        z[KDIM * hd:KDIM * (hd + 1), one] = 8 * S
        out[dst, KDIM * hd + 120:KDIM * hd + 128] = 1 / 256
        out[res, KDIM * hd + 120:KDIM * hd + 128] = -1 / 256

    for layer in (2, 3, 4):  # mixers that were no-ops before capture: pass the current token
        c.w[f"model.layers.{layer}.linear_attn.conv1d.weight"][:, 0, 3] = 1.0
    cnt1, cnt2, fx = L.f["CNT1"][0], L.f["CNT2"][0], L.f["FX"][0]
    # A1 (layer 0, head 4): halves at text / EOF tokens, reset at CAPSTART, else kept
    counter(0, 4, {TEXT_: 128, EOF_: 128, one: -128, CAP_: 512}, cnt1, cnt_res[0:2])
    # block flags n >= 16 b, b = 1..6, on A1 (layer 1, heads 2..7)
    for b_ in range(1, 7):
        step_flag(1, 1 + b_, cnt1, 16 * b_, flag[b_], flag_res[b_])
    # A2 (layer 2, head 0): halves at text / EOF once n >= 96 (FLAG 6)
    counter(2, 0, {TEXT_: 128, EOF_: 128, flag[6]: 128, one: -256, CAP_: 512}, cnt2, cnt_res[2:4])
    # block flags b = 7..11 on A2 (layer 3, heads 0..4): A2 = 2^(-8-(n-96)) for n >= 96
    for b_ in range(7, NBLK):
        step_flag(3, b_ - 7, cnt2, 16 * b_ - 96, flag[b_], flag_res[b_])
    # MLP 3: BLK = flag_b and not flag_{b+1}; FX = the block's counter * 2^(16 b) = 1.5 2^(-8-r)
    m3 = Mlp(c, 3, one)
    m3.n = MLP3_USED[0]
    blk, blk_res = L.f["BLK"], L.f["BLK_RES"][0]
    for b_ in range(NBLK):
        feats = [flag[b_]] if b_ else []
        negs = [flag[b_ + 1]] if b_ + 1 < NBLK else []
        m3.unit(feats, {blk[b_]: 1.0, blk_res: -1.0}, negs=negs)
        src, sh = (cnt1, 16 * b_) if b_ < 6 else (cnt2, 16 * (b_ - 6))
        m3.unit(feats, {fx: 1.0}, negs=negs, up={src: 2.0 ** sh / 128})
    # fine flags r >= k, k = 1..15 (layer 4, heads 0..14), one row per block
    fine, fine_res = L.f["FINE"], L.f["FINE_RES"]
    for k in range(1, 16):
        step_flag(4, k - 1, fx, k, fine[k], fine_res[k], key=list(blk))
    # MLP 4: the index n = 16 b + r at text / EOF tokens -> ROWC (n & 127), HC (n >> 7)
    m4 = Mlp(c, 4, one)
    m4.n = MLP4_USED[0]
    rowc, hcf, idx_res = L.f["ROWC"], L.f["HC"], L.f["IDX_RES"]
    for b_ in range(NBLK):
        for r in range(16):
            n = 16 * b_ + r
            feats = [textlike, blk[b_]] + ([fine[r]] if r else [])
            negs = [fine[r + 1]] if r < 15 else []
            m4.unit(feats, {rowc[n & 127]: 1.0, hcf[n >> 7]: 1.0, idx_res[0]: -1.0, idx_res[1]: -1.0}, negs=negs)
    print(f"capture: MLP 3 +{2 * NBLK} units, MLP 4 +{16 * NBLK} units")


MLP3_USED, MLP4_USED = [0], [0]  # units already used by alu() / memory() (set there)


def preload(c, prog):
    """The stored tokens as capture will have written them (C3c): into the memory heads' state."""
    from bvm import VM
    from firmware2 import MEM_WORDS
    from lexer_fw import TOK, load_program
    vm = VM(FIRMWARE, memory_words=MEM_WORDS)
    load_program(vm, prog)
    n = len(prog.ids) + 1  # + EOF
    for base in (TOK, TOK + 512, TOK + 1024, TOK + 1536):
        for a in range(base, base + n):  # capture writes all 4 words of every token (+ EOF)
            g = a >> 7
            st = c.state[MEM_LAYER0 + g // NH][g % NH]
            st[a & 127, :] = 0
            for j, d in enumerate(nibbles(vm.mem[a], 8)):
                st[a & 127, 16 * j + d] = 8.0


def capture_run(c, L, prog, local):
    """Feed CAPSTART, the program's tokens (local ids) and EOF; check the capture fields at every
    position and the token store after EOF against what load_program stores. -> positions."""
    from bvm import VM
    from firmware2 import MEM_WORDS
    from lexer_fw import TOK, load_program
    vm = VM(FIRMWARE, memory_words=MEM_WORDS)
    load_program(vm, prog)
    seq = [CAPSTART] + [local[i] for i in prog.ids] + [EOF]
    for pos, t in enumerate(seq):
        h, logits = c.step(t)
        if t == CAPSTART:
            continue
        n = pos - 1
        if n <= 110:
            assert h[L.f["CNT1"][0]] == 1.5 * 2.0 ** (-8 - n), (n, "counter A1", h[L.f["CNT1"][0]])
        b_, r = divmod(n, 16)
        assert list(h[list(L.f["FLAG"])][1:]) == [float(n >= 16 * k) for k in range(1, NBLK)], (n, "flags")
        assert list(h[list(L.f["BLK"])]) == [float(k == b_) for k in range(NBLK)], (n, "block")
        assert h[L.f["FX"][0]] == 1.5 * 2.0 ** (-8 - r), (n, "fine value")
        assert list(h[list(L.f["FINE"])][1:]) == [float(r >= k) for k in range(1, 16)], (n, "fine flags")
        assert list(h[list(L.f["ROWC"])]) == [float(k == n % 128) for k in range(128)], (n, "row")
        assert list(h[list(L.f["HC"])]) == [float(k == n // 128) for k in range(2)], (n, "head")
    top = int(np.argmax(logits))
    assert top == 0, ("EOF must emit INS(0)", top)
    for k in range(4):  # the token store: region k, index n -> layer 8 head 4k + (n >> 7), row n & 127
        for n in range(len(prog.ids) + 1):
            g = 48 + 4 * k + (n >> 7)
            row = c.state[MEM_LAYER0 + g // NH][g % NH][n & 127]
            want = vm.mem[TOK + 512 * k + n]
            assert decode_word(row / 8.0, range(128)) == want, (k, n, "stored word", want)
    return len(seq)


def check(c, L, prog, max_pos=None, local=None):
    """Teacher-force the program's true trace; compare with TraceMachine at every position.
    -> positions checked."""
    from trace import trace
    toks = trace(FIRMWARE, prog)
    tm = TraceMachine(FIRMWARE, prog)
    c.reset()
    if local is None:
        preload(c, prog)
    else:  # the circuit's own capture fills the token store
        capture_run(c, L, prog, local)
    reg_state = c.state[1]  # layer R
    for i, t in enumerate(toks[:max_pos]):
        h, logits = c.step(t)
        if i + 1 < len(toks):
            top = int(np.argmax(logits))
            assert top == toks[i + 1] and (logits == logits[top]).sum() == 1, (i, t, "next token", top, toks[i + 1])
        tm.feed(t)
        f = fields(tm.latch)
        lat = h[list(L.f["LAT0"])]
        want_lat0 = np.zeros(128)
        want_lat0[OPS.index(f["op"])] = 1
        for fld, off in (("rd", 32), ("rs", 49), ("rt", 66)):
            want_lat0[off + (16 if f[fld] is None else f[fld])] = 1
        want_lat0[124:128] = 1
        assert (lat == want_lat0).all(), (i, t, "latch 0")
        want_lat2 = np.zeros(128)
        for j, d in enumerate(nibbles(f["tgt"], 3)):
            want_lat2[16 * j + d] = 1
        for j, d in enumerate(nibbles(tm.latch + 1, 3)):
            want_lat2[48 + 16 * j + d] = 1
        want_lat2[126:128] = 1
        assert (h[list(L.f["LAT2"])] == want_lat2).all(), (i, t, "latch 2")
        assert decode_word(h, L.f["LAT1"]) == f["imm"], (i, "latch imm")
        if LAST <= t < END:
            assert decode_word(h, L.f["WORD"]) == tm.window_value(), (i, "word")
        for r in range(16):
            row = reg_state[0, r]
            got = decode_word(row / 8.0, range(128)) if row.any() else None
            want = tm.r[r] if (r == 0 or i >= 16) else (0 if r <= i else None)
            if i >= 16 or got is not None:
                assert got == want, (i, t, "register", r, got, want)
        if f["rs"] is not None and i >= 16:
            assert decode_word(h, L.f["RSW"]) == tm.r[f["rs"]], (i, "rs read")
        if f["rt"] is not None and i >= 16:
            assert decode_word(h, L.f["RTW"]) == tm.r[f["rt"]], (i, "rt read")
        if i >= 16:  # the ALU
            check_alu(h, L, f, tm, i)
            if f["op"] == "LD" and t < LAST:  # the memory read (at INS and MORE positions)
                addr = (tm.r[f["rs"]] + f["imm"]) % 2**32
                assert decode_word(h, L.f["MEMVAL"]) == tm.mem[addr], (i, "memory read", addr)
    return min(len(toks), max_pos or len(toks))


def check_alu(h, L, f, tm, i):
    op, r = f["op"], tm.r
    rs = r[f["rs"]] if f["rs"] is not None else None
    rt = r[f["rt"]] if f["rt"] is not None else None
    M32 = 2**32
    if op in ("ADD", "SUB", "SLTU", "SEQ", "BEQ", "BNE", "XOR"):
        b = rt if op not in ("SUB", "SLTU") else (~rt) % M32
    elif op in ("ADDI", "LD", "ST", "SEQI", "BEQI", "BNEI", "BEQZ", "BNEZ"):
        b = f["imm"]
    else:
        b = None
    if rs is not None and b is not None:
        cin = int(op in ("SUB", "SLTU"))
        assert decode_word(h, L.f["BSEL"]) == b, (i, op, "bsel")
        assert decode_word(h, L.f["SUM"]) == (rs + b + cin) % M32, (i, op, "sum")
        want_c = int(rs + b + cin >= M32)
        assert list(h[list(L.f["COUT"])]) == [1 - want_c, want_c], (i, op, "carry out")
        want_e = int(rs == b)
        assert list(h[list(L.f["EQ"])]) == [1 - want_e, want_e], (i, op, "eq")
        assert decode_word(h, L.f["XORF"]) == rs ^ b, (i, op, "xor")
    if rs is not None and rt is not None:
        k = 4 * (rt % 8)
        want_rot = ((rs << k) | (rs >> (32 - k))) % M32 if k else rs
        assert decode_word(h, L.f["ROTF"]) == want_rot, (i, op, "rotn")
    if rs is not None:
        assert decode_word(h, L.f["SHR"]) == rs >> 1, (i, op, "shr")


if __name__ == "__main__":
    from lexer_fw import Stored, token_words
    from test_firmware import load_programs
    n_prog = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tok, rom, progs = load_programs(sys.argv[2:])
    words = token_words(rom)
    chosen = []
    for origin, src in progs[::max(1, len(progs) // n_prog)][:n_prog]:
        ids = tok.encode(src, add_special_tokens=False)
        if len(ids) < MAX_N and all(i in words for i in ids):
            chosen.append(ids)
    text_ids = sorted({i for ids in chosen for i in ids})
    local = {i: TEXT0 + k for k, i in enumerate(text_ids)}
    c, L = build([(i, words[i]) for i in text_ids])
    print(f"engine circuit: hidden {H}, {L.next - len(L.f['BAL'])} field dims, {c.cfg['n_layers']} layers, "
          f"{len(text_ids)} text tokens")
    done, pos = 0, 0
    for ids in chosen:
        pos += check(c, L, Stored(ids, words), local=local)
        done += 1
    print(f"engine circuit: capture (counters, flags, index, token store) exact, then next token == the "
          f"trace's at every position (and latch, word, registers, reads, ALU, memory == TraceMachine): "
          f"{done} programs, {pos} positions")
