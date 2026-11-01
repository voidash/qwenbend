"""C1b: the exactness facts the circuit rests on, measured on the real Qwen3.5 code paths
(transformers qwen3_5 functions and modules, torch MPS, bf16 where the model is bf16).
Each check prints its measured threshold and asserts the property the circuit relies on.
Run: python bendvm/test_circuit_prims.py
"""

import math

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as Q

DEV = "mps"
BF = torch.bfloat16


def bf(x):
    return torch.tensor(x, dtype=BF, device=DEV)


def check(name, ok, detail):
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    if not ok:
        raise SystemExit(1)


def bf16_grid(lo, hi):
    """Every bf16 value in [lo, hi] (as a float32 tensor on DEV)."""
    xs = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(BF).float()
    xs = xs[torch.isfinite(xs) & (xs >= lo) & (xs <= hi)]
    return xs.unique().to(DEV)


def main():
    # 1. beta = sigmoid(b) in bf16 (Qwen3_5GatedDeltaNet.forward: b.sigmoid() on a bf16 tensor)
    g = bf16_grid(-200, 200).to(BF)
    s = g.sigmoid()
    one = g[s == 1.0].float().min().item()
    zero = g[s == 0.0].float().max().item()
    check("sigmoid -> exactly 1 / 0 (bf16)", one <= 16 and zero >= -128,
          f"sigmoid(b) == 1.0 for b >= {one}; == 0.0 for b <= {zero}")

    # 2. decay: g = -exp(A_log) * softplus(a + dt_bias) in fp32, then exp(g)
    a_log = bf(-1000.0).float()
    for a in (-100.0, 0.0, 100.0):
        gg = -a_log.exp() * F.softplus(torch.tensor(a, device=DEV) + torch.tensor(1.0, device=DEV))
        check("decay exactly 1 with A_log = -1000", gg.exp().item() == 1.0, f"a={a}: exp(g) = {gg.exp().item()!r}")

    # 3. SiLU on bf16 (conv1d's activation, the MLP's act_fn): identity and exact zero ranges
    g = bf16_grid(-1000, 1000).to(BF)
    y = F.silu(g)
    ident = g[(y == g) & (g > 0)].float()
    lo_ident = ident.min().item()
    above = g[g.float() >= lo_ident]
    all_ident = bool((F.silu(above) == above).all())
    check("SiLU(x) == x in bf16 above a threshold", all_ident,
          f"silu(x) == x for every bf16 x >= {lo_ident}")
    nz = g[(y != 0) & (g < 0)].float().min().item()  # the most negative x with silu(x) != 0
    below = g[g.float() < nz]
    all_zero = below.numel() > 0 and bool((F.silu(below) == 0).all())
    check("SiLU(x) == 0 in bf16 below a threshold", all_zero, f"silu(x) == 0 for every bf16 x < {nz}")

    # 4. l2norm of a scaled one-hot key (fp32, eps 1e-6) is exactly the one-hot
    bad = []
    for c in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 7.0, 10.0, 100.0]:
        k = torch.zeros(1, 1, 1, 128, device=DEV)
        k[..., 5] = c
        kn = Q.l2norm(k, dim=-1, eps=1e-6)
        if not (kn[..., 5].item() == 1.0 and int((kn != 0).sum()) == 1):
            bad.append((c, kn[..., 5].item()))
    check("l2norm(c * e_i) == e_i", not [b for b in bad if b[0] >= 8],
          f"exact for c >= 8; inexact: {bad}")

    # 5. the delta rule as an exact row memory (Q.torch_recurrent_gated_delta_rule, fp32 state)
    torch.manual_seed(0)
    H = 2
    state = torch.zeros(1, H, 128, 128, device=DEV)
    rows = {}
    ok = True
    for step in range(40):
        r = int(torch.randint(0, 128, ()).item())
        word = int(torch.randint(0, 2**32, ()).item())
        v = torch.zeros(1, 1, H, 128, device=DEV)
        for j in range(8):  # 8 one-hot nibbles
            v[0, 0, 0, 16 * j + (word >> (4 * j) & 15)] = 1.0
        k = torch.zeros(1, 1, H, 128, device=DEV)
        k[0, 0, :, r] = 16.0
        q = k.clone()
        beta = torch.zeros(1, 1, H, device=DEV)
        beta[0, 0, 0] = 1.0  # write head 0 only
        gdec = torch.zeros(1, 1, H, device=DEV)
        out, state = Q.torch_recurrent_gated_delta_rule(q, k, v, g=gdec, beta=beta, initial_state=state,
                                                        output_final_state=True, use_qk_l2norm_in_kernel=True)
        rows[r] = v[0, 0, 0].clone()
        ok &= bool((state[0, 0, r] == rows[r]).all()) and bool((state[0, 1] == 0).all())
        expect = rows[r] / math.sqrt(128)
        ok &= bool(((out[0, 0, 0] - expect).abs() < 1e-7).all())
        ok &= bool((out[0, 0, 1] == 0).all())
    reread = all(bool((state[0, 0, r] == v_).all()) for r, v_ in rows.items())
    check("delta rule: one-hot rows store words exactly", ok and reread,
          f"{len(rows)} rows rewritten 40 times; state rows exact, unwritten head reads exactly 0")

    # 6. RMSNormGated: the read row keeps a margin; z = 0 gates to exactly 0
    norm = Q.Qwen3_5RMSNormGated(128, eps=1e-6).to(DEV).to(BF)
    row = rows[next(iter(rows))].to(BF) / math.sqrt(128)
    o = norm(row[None].to(BF), bf(8.0).expand(1, 128))
    on, off = o[0][row > 0].float(), o[0][row == 0].float()
    check("RMSNormGated keeps a one-hot margin", bool(on.min() > 10 * off.abs().max() + 1),
          f"on = {on.unique().tolist()}, off = {off.abs().max().item()}")
    o0 = norm(row[None].to(BF), torch.zeros(1, 128, dtype=BF, device=DEV))
    check("SiLU(z = 0) gates a head to exactly 0", bool((o0 == 0).all()), "all zero")
    zero_row = torch.zeros(1, 128, dtype=BF, device=DEV)
    check("an unwritten row reads exactly 0 after RMSNormGated", bool((norm(zero_row, bf(8.0).expand(1, 128)) == 0).all()), "all zero")

    # 7. causal conv1d decode update with kernel [0, 0, 0, 1] is the identity before SiLU
    x = torch.tensor([[[3.0], [-5.0], [16.0]]], dtype=BF, device=DEV)  # (1, channels 3, seq 1)
    cs = torch.zeros(1, 3, 4, dtype=BF, device=DEV)
    w = torch.zeros(3, 4, dtype=BF, device=DEV)
    w[:, 3] = 1.0
    y = Q.causal_conv1d_update.__wrapped__(x, cs, w, None, None) if hasattr(Q.causal_conv1d_update, "__wrapped__") \
        else Q.causal_conv1d_update(x, cs, w, None, None)
    check("conv1d kernel [0,0,0,1] passes the current token", bool((y == x).all()), f"{y.flatten().tolist()}")

    # 8. the residual RMSNorm is an exact power-of-2 scale when mean(x^2) is a power of 4 >= 64
    #    (eps 1e-6 is below half an fp32 ulp there); one ballast dim keeps the energy fixed
    rn = Q.Qwen3_5RMSNorm(1024, eps=1e-6).to(DEV).to(BF)
    exact = []
    for k in range(3, 9):  # mean(x^2) = 4^k
        x = torch.zeros(1, 1024, dtype=BF, device=DEV)
        x[0, 1:9] = 1.0  # a one-hot-ish field: 8 ones
        x[0, 10] = -2.0
        rest, d, j = 1024 * 4**k - 8 - 4, 20, 0  # ballast: base-4 digits of the rest, as 2^j dims
        while rest:
            for _ in range(rest % 4):
                x[0, d] = 2.0 ** j
                d += 1
            rest //= 4
            j += 1
        assert float((x.float() ** 2).sum()) == 1024 * 4**k
        y = rn(x).float()
        scale = 2.0 ** -k
        exact.append((k, bool((y == x.float() * scale).all())))
    check("residual RMSNorm exact at mean 4^k", exact and all(e for _, e in exact), f"{exact}")

    # 9 (C3c). (a) a halving counter row: x = c0 2^-n, two reference dims R as fixed points
    #    (v = R, beta = 1/2); after RMSNormGated (z = 8) the counter dim is exactly 2^(-6-n)
    R, c0 = 8.0, 2.0 ** -9
    st = torch.zeros(1, 1, 128, 128, device=DEV)
    k = torch.zeros(1, 1, 1, 128, device=DEV)
    k[..., 0] = 16.0
    gn = Q.Qwen3_5RMSNormGated(128, eps=1e-6).to(DEV).to(BF)
    z8 = bf(8.0).expand(1, 128)
    ok_a, bad = True, []
    for n in range(0, 111):
        v = torch.zeros(1, 1, 1, 128, device=DEV)
        v[..., 1] = v[..., 2] = R
        if n == 0:
            v[..., 0] = c0
        beta = torch.full((1, 1, 1), 1.0 if n == 0 else 0.5, device=DEV)  # reset, then halve
        out, st = Q.torch_recurrent_gated_delta_rule(k, k, v, g=torch.zeros(1, 1, 1, device=DEV), beta=beta,
                                                     initial_state=st, output_final_state=True,
                                                     use_qk_l2norm_in_kernel=True)
        o = gn(out[0, 0].to(BF), z8).float()
        want = 2.0 ** (-6 - n)
        if not (o[0, 0].item() == want and o[0, 1].item() == 64.0 and o[0, 2].item() == 64.0):
            ok_a = False
            bad.append((n, o[0, 0].item(), want))
    check("halving counter reads an exact power of 2", ok_a, f"n = 0..110 exact; bad: {bad[:3]}")

    # (b) a beta-step flag "n >= m" on x = 2^(-6-n): b = 256 - 192 2^(m-n) (bf16 linear), exactly 0 / 1
    ok_b = True
    for m in range(1, 110, 7):
        w = torch.tensor([[-192.0 * 2.0 ** (6 + m), 256.0]], dtype=BF, device=DEV)  # [x, one]
        for n in range(0, 111):
            xin = torch.tensor([[2.0 ** (-6 - n), 1.0]], dtype=BF, device=DEV)
            bb = (xin @ w.T)[0, 0]
            beta = bb.sigmoid().item()
            if beta != float(n >= m):
                ok_b = False
                print("   step", m, n, bb.item(), beta)
    check("beta-step flag n >= m is exactly 0 / 1", ok_b, "m = 1, 8, .., 106 against n = 0..110")

    # (c) the rescale unit: silu(gate = 128) * up(= x 2^k / 128) is exactly x 2^k (the real MLP)
    mlp = Q.Qwen3_5MLP(type("C", (), {"hidden_size": 2, "hidden_act": "silu"})(), 1).to(DEV).to(BF)
    ok_c = True
    with torch.no_grad():
        mlp.gate_proj.weight.copy_(torch.tensor([[0.0, 128.0]]))
        mlp.down_proj.weight.copy_(torch.tensor([[1.0], [0.0]]))
        for kk in (16, 32, 48, 64, 80, 96):
            mlp.up_proj.weight.copy_(torch.tensor([[2.0 ** kk / 128, 0.0]]))
            for n in range(0, 111):
                xin = torch.tensor([[2.0 ** (-6 - n), 1.0]], dtype=BF, device=DEV)
                y = mlp(xin)[0, 0].item()
                if y != 2.0 ** (kk - 6 - n):
                    ok_c = False
    check("rescale unit (gate 128, up a power of 2) is exact", ok_c, "x 2^k for k = 16..96, n = 0..110")


if __name__ == "__main__":
    main()
