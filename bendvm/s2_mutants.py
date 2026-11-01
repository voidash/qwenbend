"""S2 ground truth: mutate verified BendLite programs in ways Bend2's static checks may reject
(types, literal kinds, affine use, termination, unreachable errors), and label each mutant with
real Bend2 (`--check-only`, then a run). Output: s2_mutants.jsonl
  {"kind", "program", "valid", "output", "error"}
Run: python3 bendvm/s2_mutants.py [--per-kind 25] [--workers 8]
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from paths import BENDTUNE, bend2_main  # noqa: E402

HERE = Path(__file__).resolve().parent
BEND = bend2_main()
OUT = HERE / "s2_mutants.jsonl"


def params_of(src):
    """[(def name, [(plus, name, type)])] for every def line."""
    out = []
    for m in re.finditer(r"^def (\w+)\((.*)\) -> (\w+):", src, re.M):
        ps = []
        for p in [x.strip() for x in m[2].split(",") if x.strip()]:
            pm = re.fullmatch(r"(\+?)(\w+): (\w+)", p)
            ps.append((pm[1] == "+", pm[2], pm[3]))
        out.append((m[1], ps))
    return out


def m_nat_to_u32(src, rng):  # 5n -> 5 somewhere (a U32 literal where a Nat is expected)
    lits = list(re.finditer(r"\b(\d+)n\b", src))
    if not lits:
        return None
    m = rng.choice(lits)
    if src[m.start() - 3:m.start()] == "1n+" or src[m.start() - 5:m.start()] == "case ":
        return None
    return src[:m.start()] + m[1] + src[m.end():]


def m_u32_to_nat(src, rng):  # 5 -> 5n in an expression (a Nat literal where a U32 is expected)
    lits = [m for m in re.finditer(r"(?<![\w.])(\d+)(?![\dn\w])", src)]
    if not lits:
        return None
    m = rng.choice(lits)
    return src[:m.start()] + m[1] + "n" + src[m.end():]


def m_drop_plus(src, rng):  # +x: T -> x: T (affine: fine if used at most once)
    ms = list(re.finditer(r"\+(\w+): ", src))
    if not ms:
        return None
    m = rng.choice(ms)
    return src[:m.start()] + src[m.start() + 1:]


def m_swap_params(src, rng):  # swap the first two params of a recursive def and its calls
    for name, ps in params_of(src):
        if len(ps) >= 2 and len(re.findall(rf"\b{name}\(", src)) >= 3:
            def swap_call(m):
                args, depth, cur = [], 0, ""
                for ch in m[1]:
                    if ch == "," and depth == 0:
                        args.append(cur)
                        cur = ""
                        continue
                    depth += ch == "("
                    depth -= ch == ")"
                    cur += ch
                args.append(cur)
                if len(args) < 2:
                    return m[0]
                args[0], args[1] = args[1].strip(), " " + args[0].strip()
                args[1] = args[1] if args[1].startswith(" ") else " " + args[1]
                return f"{name}({args[0]},{','.join(args[1:])})"
            out = re.sub(rf"\b{name}\(((?:[^()]|\((?:[^()]|\([^()]*\))*\))*)\)", swap_call, src)
            return out if out != src else None
    return None


def m_no_shrink(src, rng):  # a recursive call passes the matched param itself, not its binder
    m = re.search(r"match (\w+):\n(?:.*\n)*?\s+case 1n\+(\w+):", src)
    if not m:
        return None
    var, binder = m[1], m[2]
    rest = src[m.end():]
    if not re.search(rf"\b{binder}\b", rest):
        return None
    return src[:m.end()] + re.sub(rf"\b{binder}\b", var, rest, count=1)


def m_param_type(src, rng):  # one param's declared type changed
    ms = list(re.finditer(r"(\+?\w+): (U32|Nat|Bool)", src))
    if not ms:
        return None
    m = rng.choice(ms)
    new = rng.choice([t for t in ("U32", "Nat", "Bool") if t != m[2]])
    return src[:m.start(2)] + new + src[m.end(2):]


def m_ret_type(src, rng):  # one non-main def's return type changed
    ms = [m for m in re.finditer(r"^def (\w+)\(.*\) -> (U32|Nat|Bool):", src, re.M) if m[1] != "main"]
    if not ms:
        return None
    m = rng.choice(ms)
    new = rng.choice([t for t in ("U32", "Nat", "Bool") if t != m[2]])
    return src[:m.start(2)] + new + src[m.end(2):]


def m_dead_error(src, rng):  # an unknown variable in a match arm (maybe never taken)
    ms = list(re.finditer(r"(case 0n:\n\s+)(\S[^\n]*)", src))
    if not ms:
        return None
    m = rng.choice(ms)
    return src[:m.start(2)] + "zzz" + src[m.end(2):]


def m_bool_arith(src, rng):  # a comparison result used as a U32 operand
    ms = list(re.finditer(r"\((\w+) \+ ", src))
    if not ms:
        return None
    m = rng.choice(ms)
    return src[:m.start(1)] + f"({m[1]} < 1 : U32)" + src[m.end(1):]


def m_use_twice(src, rng):  # a non-+ param used twice
    for name, ps in params_of(src):
        for plus, p, ty in ps:
            if not plus and ty == "Bool":
                return None
    ms = list(re.finditer(r"\+(\w+): U32", src))
    if not ms:
        return None
    m = rng.choice(ms)
    return src[:m.start()] + src[m.start() + 1:]


def m_nat_arith(src, rng):  # Nat binder used in U32 arithmetic without U32.from_nat
    m = re.search(r"U32\.from_nat\((\w+)\)", src)
    if not m:
        return None
    return src[:m.start()] + m[1] + src[m.end():]


KINDS = {"nat_to_u32": m_nat_to_u32, "u32_to_nat": m_u32_to_nat, "drop_plus": m_drop_plus,
         "swap_params": m_swap_params, "no_shrink": m_no_shrink, "param_type": m_param_type,
         "ret_type": m_ret_type, "dead_error": m_dead_error, "bool_arith": m_bool_arith,
         "nat_arith": m_nat_arith, "original": lambda s, r: s}


def bend(src, td, i):
    path = Path(td) / f"mut_{i}.bend"
    path.write_text(src)
    try:
        c = subprocess.run(["bun", str(BEND), str(path), "--check-only"], capture_output=True,
                           text=True, timeout=120)
        valid = "ALL PROOFS CHECK" in c.stdout + c.stderr
        out, err = "", ""
        if valid:
            r = subprocess.run(["bun", str(BEND), str(path)], capture_output=True, text=True,
                               timeout=120)
            lines = (r.stdout + r.stderr).strip().splitlines()
            out = lines[-1] if lines else ""
        else:
            err = (c.stdout + c.stderr).strip()[-400:]
    except subprocess.TimeoutExpired:
        valid, out, err = False, "", "timeout"
    return {"valid": valid, "output": out, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    srcs = [json.loads(x)["messages"][2]["content"] for x in open(BENDTUNE / "data/train.jsonl")]
    muts, seen = [], set()
    for kind, fn in KINDS.items():
        got, tries = 0, 0
        while got < args.per_kind and tries < 2000:
            tries += 1
            out = fn(rng.choice(srcs), rng)
            if out and out not in seen:
                seen.add(out)
                muts.append((kind, out))
                got += 1
    with tempfile.TemporaryDirectory() as td, ThreadPoolExecutor(args.workers) as pool:
        res = list(pool.map(lambda x: bend(x[1][1], td, x[0]), enumerate(muts)))
    with open(OUT, "w") as f:
        for (kind, src), r in zip(muts, res):
            f.write(json.dumps({"kind": kind, "program": src, **r}) + "\n")
    by = {}
    for (kind, _), r in zip(muts, res):
        v = by.setdefault(kind, [0, 0])
        v[0] += 1
        v[1] += r["valid"]
    for k, (n, ok) in by.items():
        print(f"{k:<12} {ok}/{n} accepted by Bend2")


if __name__ == "__main__":
    main()
