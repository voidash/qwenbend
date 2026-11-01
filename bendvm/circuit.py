"""C1: our circuit IR and its exact evaluator, for Qwen3.5-architecture models (written from
scratch; the Qwen3.5 math is transformers' qwen3_5 decode path, read in DESIGN.md Phase 2).

IR: a Circuit is the HF config values plus numpy weights under the HF parameter names
(`model.layers.{i}.linear_attn.in_proj_qkv.weight`, ...), so to_hf() loads it into the stock
Qwen3_5ForCausalLM unchanged. Only linear-attention (Gated DeltaNet) layers so far.

Evaluator: Circuit.step(token) runs one decode position in float64, mirroring the HF code
operation by operation. Where torch casts to bf16, the value must be EXACTLY representable
(asserted), except where Qwen's eps makes a value inexact by design (RMSNormGated): there the
evaluator rounds to bf16 and asserts the value is at least MARGIN ulps away from a rounding
boundary, so torch's fp32 arithmetic lands on the same bf16 value. Every other nonlinearity
must sit in a region where it is exact in bf16 (test_circuit_prims.py): SiLU(x) = x for
x >= 5.97, 0 for x < -88.5 or x = 0; sigmoid = 1 for b >= 6.25, 0 for b <= -87.5; residual
norms at mean(x^2) = 4^k >= 64; l2-normalised q / k one-hot (c >= 8) or zero.
A gadget that is not exact by construction fails loudly instead of drifting.

Run: python bendvm/circuit.py   (self-check: test_circuit.py is the
     torch comparison)
"""

import math

import numpy as np

KDIM = VDIM = 128
MARGIN = 0.1  # ulps from a bf16 rounding boundary


class Inexact(Exception):
    pass


FTZ = 2.0 ** -126  # MPS flushes subnormals to zero (fp32 ops and bf16 casts; measured)


def bf16_round(x):
    """Round float64 values to the nearest bf16 (ties to even), as float64; |x| < 2^-126 -> 0
    (MPS flush-to-zero)."""
    a = np.asarray(x, dtype=np.float64)
    f = a.astype(np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    r = ((u + 0x7FFF + lsb) >> 16) << 16
    out = r.astype(np.uint32).view(np.float32).astype(np.float64)
    out = np.where(np.abs(out) < FTZ, 0.0, out)
    return np.where(np.isfinite(a), out, a)


def exact_bf16(x, what):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.abs(x) < FTZ, 0.0, x)  # MPS flushes subnormals to zero
    if not np.array_equal(bf16_round(x), x):
        bad = x[bf16_round(x) != x].ravel()[:4]
        raise Inexact(f"{what}: not exactly bf16: {bad}")
    return x


def ulp_bf16(x):
    x = np.abs(np.asarray(x, dtype=np.float64))
    e = np.floor(np.log2(np.where(x > 0, x, 1.0)))
    return np.where(x > 0, 2.0 ** (e - 7), 2.0 ** -133)


def round_with_margin(x, what):
    """bf16(x), asserting x is >= MARGIN ulp from a rounding boundary (so fp32 noise can't
    change the result)."""
    x = np.asarray(x, dtype=np.float64)
    near = (np.abs(x) >= FTZ * (1 - 2.0 ** -18)) & (np.abs(x) < FTZ)  # within fp32 noise
    if near.any():
        raise Inexact(f"{what}: too close to the flush-to-zero threshold: {x[near][:4]}")
    x = np.where(np.abs(x) < FTZ, 0.0, x)  # clearly below 2^-126: MPS flushes it to 0
    r = bf16_round(x)
    dist = 0.5 * ulp_bf16(r) - np.abs(x - r)
    if (dist < MARGIN * ulp_bf16(r)).any():
        raise Inexact(f"{what}: too close to a bf16 rounding boundary: {x[dist < MARGIN * ulp_bf16(r)][:4]}")
    return r


def silu_exact(x, what):
    """SiLU in bf16 where it is exact: x >= 5.96875 -> x, x < -88.5 -> 0, x == 0 -> 0."""
    x = exact_bf16(x, what)
    ok = (x >= 5.96875) | (x < -88.5) | (x == 0)
    if not ok.all():
        raise Inexact(f"{what}: SiLU input outside its exact bf16 regions: {x[~ok][:4]}")
    return np.where(x >= 5.96875, x, 0.0)


def sigmoid_exact(x, what):
    """sigmoid in bf16 where it is exact: 1 for b >= 6.25, 0 for b <= -87.5, 0.5 at b = 0 (an
    unused head: k = 0 there, so its write is exactly zero anyway)."""
    x = exact_bf16(x, what)
    ok = (x >= 6.25) | (x <= -87.5) | (x == 0)
    if not ok.all():
        raise Inexact(f"{what}: sigmoid input outside its exact regions: {x[~ok][:4]}")
    return np.where(x >= 6.25, 1.0, np.where(x == 0, 0.5, 0.0))


def linear(x, w, what):
    """bf16 nn.Linear: exact when the true sum is a bf16 value (every accumulation order then
    agrees, the sums being small dyadic values)."""
    return exact_bf16(w @ x, what)


TINY = 2.0 ** -5  # residual values at most this large are "absorbed" by the norms (C3c counters)


def rmsnorm(x, w, eps, what):
    """Qwen3_5RMSNorm: x / rms(x) * (1 + w) in fp32, cast to bf16; exact at mean(x^2) = 4^k >= 64.
    Tiny values (|x| <= 2^-5, exact powers of 2 such as the capture counters) are absorbed: their
    energy is asserted below 2^-26 of the rest, under half an fp32 ulp of the sum in any order, so
    torch's fp32 mean is the rest's mean exactly; they are still scaled exactly by 2^-k."""
    tiny = np.abs(x) <= TINY
    e_tiny = float(np.sum(x[tiny] ** 2))
    ms = float(np.mean(np.where(tiny, 0.0, x) ** 2))
    if e_tiny >= 2.0 ** -26 * ms * len(x):
        raise Inexact(f"{what}: tiny residual values carry too much energy ({e_tiny}) to be absorbed")
    k = round(math.log(ms, 4)) if ms > 0 else -1
    if ms == 4.0 ** k and ms >= 64:
        return exact_bf16(x * 2.0 ** -k * (1.0 + w), what)
    # near a power of 4 (small extra energy, e.g. the tiny emit patterns of tied rows plus an
    # emitted 1.0 on the same dim): the rsqrt shifts by < 2^-13 relative, and the bf16 cast
    # absorbs it wherever the value sits >= MARGIN ulp from a rounding boundary
    full = float(np.mean(x ** 2))
    if ms < 64 or abs(full / 4.0 ** k - 1) > 2.0 ** -12:
        raise Inexact(f"{what}: mean(x^2) = {full} is not (near) a power of 4 >= 64 (add ballast)")
    return round_with_margin(x / math.sqrt(full) * (1.0 + w), what)


class Circuit:
    def __init__(self, hidden, n_layers, n_heads, intermediate, vocab, eps=1e-6):
        self.cfg = dict(hidden=hidden, n_layers=n_layers, n_heads=n_heads,
                        intermediate=intermediate, vocab=vocab, eps=eps)
        H, NH, I = hidden, n_heads, intermediate
        z = np.zeros
        self.w = {"model.embed_tokens.weight": z((vocab, H)), "model.norm.weight": z(H)}
        for i in range(n_layers):
            p = f"model.layers.{i}."
            self.w.update({
                p + "input_layernorm.weight": z(H), p + "post_attention_layernorm.weight": z(H),
                p + "linear_attn.in_proj_qkv.weight": z((2 * KDIM * NH + VDIM * NH, H)),
                p + "linear_attn.in_proj_z.weight": z((VDIM * NH, H)),
                p + "linear_attn.in_proj_b.weight": z((NH, H)),
                p + "linear_attn.in_proj_a.weight": z((NH, H)),
                p + "linear_attn.conv1d.weight": z((2 * KDIM * NH + VDIM * NH, 1, 4)),
                p + "linear_attn.dt_bias": z(NH),
                p + "linear_attn.A_log": np.full(NH, -1000.0),  # decay exactly 1
                p + "linear_attn.norm.weight": np.ones(VDIM),
                p + "linear_attn.out_proj.weight": z((H, VDIM * NH)),
                p + "mlp.gate_proj.weight": z((I, H)), p + "mlp.up_proj.weight": z((I, H)),
                p + "mlp.down_proj.weight": z((H, I)),
            })
        self._sp = {}
        self.reset()

    def lin(self, x, key, what, rounded=False):
        """linear() with the weight's nonzeros cached (the circuit's weights are very sparse; fp64
        sums of small dyadic values are exact in any order, so the result is the same)."""
        sp = self._sp.get(key)
        if sp is None:
            w = self.w[key]
            r, c_ = np.nonzero(w)
            sp = self._sp[key] = (r, c_, w[r, c_], w.shape[0])
        r, c_, v, n = sp
        y = np.bincount(r, weights=v * x[c_], minlength=n)
        return bf16_round(y) if rounded else exact_bf16(y, what)

    def reset(self):
        c = self.cfg
        C = (2 * KDIM + VDIM) * c["n_heads"]
        self.conv = [np.zeros((C, 3)) for _ in range(c["n_layers"])]  # the last 3 conv inputs
        self.state = [np.zeros((c["n_heads"], KDIM, VDIM)) for _ in range(c["n_layers"])]

    def check_weights(self):
        for k, v in self.w.items():
            if k.endswith(("A_log", "dt_bias")):
                continue  # fp32 parameters in HF too (kept as given)
            exact_bf16(v, k)

    # --- one decode position ---
    def step(self, token):
        c, w = self.cfg, self.w
        h = exact_bf16(w["model.embed_tokens.weight"][token], "embedding")
        for i in range(c["n_layers"]):
            p = f"model.layers.{i}."
            x = rmsnorm(h, w[p + "input_layernorm.weight"], c["eps"], f"L{i} input norm")
            h = exact_bf16(h + self.deltanet(i, x), f"L{i} residual after mixer")
            x = rmsnorm(h, w[p + "post_attention_layernorm.weight"], c["eps"], f"L{i} post norm")
            g = self.lin(x, p + "mlp.gate_proj.weight", f"L{i} gate")
            u = self.lin(x, p + "mlp.up_proj.weight", f"L{i} up", rounded=True)
            exact_bf16(np.where(g >= 5.96875, u, 0.0), f"L{i} up (firing units)")  # off: 0 * u = 0
            a = exact_bf16(silu_exact(g, f"L{i} mlp act") * u, f"L{i} silu(gate) * up")
            h = exact_bf16(h + self.lin(a, p + "mlp.down_proj.weight", f"L{i} down"), f"L{i} residual after mlp")
        x = rmsnorm(h, w["model.norm.weight"], c["eps"], "final norm")
        # the logits leave the circuit: only their argmax is used, so require the top token to win
        # by a clear gap (>= 1/16 of the top logit: far beyond fp32 / bf16 rounding of the sums)
        head = w.get("lm_head.weight", w["model.embed_tokens.weight"])  # untied if given
        raw = head @ x
        top2 = np.partition(raw, -2)[-2:]
        if top2[1] - top2[0] < abs(top2[1]) / 16:
            raise Inexact(f"logits: top token not clearly ahead: {top2[1]} vs {top2[0]}")
        logits = bf16_round(raw)
        return h, logits

    def deltanet(self, i, x):
        c, w, p = self.cfg, self.w, f"model.layers.{i}.linear_attn."
        NH = c["n_heads"]
        mixed = self.lin(x, p + "in_proj_qkv.weight", f"L{i} qkv")
        cw = w[p + "conv1d.weight"][:, 0, :]
        window = np.concatenate([self.conv[i], mixed[:, None]], axis=1)  # (C, 4): t-3 .. t
        self.conv[i] = window[:, 1:]
        conv = exact_bf16((window * cw).sum(axis=1), f"L{i} conv")
        qkv = silu_exact(conv, f"L{i} conv act")
        q = qkv[:KDIM * NH].reshape(NH, KDIM)
        k = qkv[KDIM * NH:2 * KDIM * NH].reshape(NH, KDIM)
        v = qkv[2 * KDIM * NH:].reshape(NH, VDIM)
        zg = self.lin(x, p + "in_proj_z.weight", f"L{i} z").reshape(NH, VDIM)
        # b may round in bf16 (a beta-step far from its threshold: 256 - 192 2^j); only its
        # saturated region matters, and sigmoid_exact still requires b >= 6.25 or <= -87.5
        beta = sigmoid_exact(self.lin(x, p + "in_proj_b.weight", f"L{i} b", rounded=True), f"L{i} beta")
        a = self.lin(x, p + "in_proj_a.weight", f"L{i} a")
        # torch: g = -exp(A_log) softplus(a + dt_bias) in fp32, decay = exp(g). Exact cases: keep
        # (exp(A_log) underflows, or softplus(a) = 0 for a <= -104) or clear (g <= -104 -> 0)
        sp = np.logaddexp(np.float32(0), (a + w[p + "dt_bias"]).astype(np.float32)).astype(np.float64)
        ea = np.exp(w[p + "A_log"].astype(np.float32)).astype(np.float64)
        keep, clear = (ea * sp == 0), (ea * sp >= 104)
        if not (keep | clear).all():
            raise Inexact(f"L{i}: decay is not exactly 1 (keep) or 0 (clear): {ea * sp}")
        decay = np.where(clear, 0.0, 1.0)
        qn, kn = self.unit(q, f"L{i} q"), self.unit(k, f"L{i} k")
        S = self.state[i]  # (NH, KDIM, VDIM); updated in place, all heads at once
        S *= decay[:, None, None]
        kv_mem = np.einsum("hkv,hk->hv", S, kn)
        S += np.einsum("hk,hv->hkv", kn, beta[:, None] * (v - kv_mem))
        out = np.einsum("hkv,hk->hv", S, qn / math.sqrt(KDIM))
        out = round_with_margin(out, f"L{i} core out")  # core_attn_out.to(bf16)
        # Qwen3_5RMSNormGated per head: fp32 norm, cast to bf16, * weight, * silu(z) (fp32), cast
        var = (out * out).mean(axis=1, keepdims=True)
        hs = round_with_margin(out / np.sqrt(var + c["eps"]), f"L{i} gated norm")
        hs = exact_bf16(w[p + "norm.weight"] * hs, f"L{i} gated norm * weight")
        o = exact_bf16(hs * silu_exact(zg, f"L{i} z act"), f"L{i} gated out")
        return self.lin(o.reshape(-1), p + "out_proj.weight", f"L{i} out_proj")

    @staticmethod
    def unit(x, what):
        """l2norm: exact only for one-hot rows c * e_i with c >= 8, or all-zero rows."""
        out = np.zeros_like(x)
        for r in range(x.shape[0]):
            nz = np.flatnonzero(x[r])
            if len(nz) == 0:
                continue
            if len(nz) != 1 or x[r, nz[0]] < 8:
                raise Inexact(f"{what} head {r}: not a one-hot with c >= 8: {x[r, nz][:4]}")
            out[r, nz[0]] = 1.0
        return out

    # --- export ---
    def to_hf(self, device="mps"):
        import torch
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        c = self.cfg
        # + one all-zero full-attention layer at the end: it adds exactly 0 to the residual (zero
        # projections and MLP), and HF's cache needs an attention layer to know its length
        cfg = Qwen3_5TextConfig(
            hidden_size=c["hidden"], num_hidden_layers=c["n_layers"] + 1,
            layer_types=["linear_attention"] * c["n_layers"] + ["full_attention"],
            linear_num_value_heads=c["n_heads"],
            linear_num_key_heads=c["n_heads"], linear_key_head_dim=KDIM, linear_value_head_dim=VDIM,
            linear_conv_kernel_dim=4, intermediate_size=c["intermediate"], vocab_size=c["vocab"],
            num_attention_heads=2, num_key_value_heads=1, head_dim=128,
            tie_word_embeddings="lm_head.weight" not in self.w,
            rms_norm_eps=c["eps"], hidden_act="silu")
        m = Qwen3_5ForCausalLM(cfg)
        sd = {k: torch.tensor(v, dtype=torch.float32) for k, v in self.w.items()}
        sd.setdefault("lm_head.weight", sd["model.embed_tokens.weight"])
        last = f"model.layers.{c['n_layers']}."
        for k, v in m.state_dict().items():
            if k.startswith(last):
                sd[k] = torch.zeros_like(v)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if unexpected or [k for k in missing if k != "lm_head.weight"]:
            raise RuntimeError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
        return m.to(device=device, dtype=torch.bfloat16).eval()


if __name__ == "__main__":
    assert bf16_round(np.array([1.0 + 2**-9]))[0] == 1.0  # ties to even
    assert bf16_round(np.array([1.0 + 3 * 2**-9]))[0] == 1.0 + 2**-7
    c = Circuit(hidden=256, n_layers=1, n_heads=1, intermediate=16, vocab=8)
    try:
        c.step(0)
        raise AssertionError("an all-zero residual must be rejected by the norm check")
    except Inexact:
        pass
    print("circuit self-check ok")
