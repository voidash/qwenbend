"""Verified task -> BendLite program dataset for fine-tuning Qwen3.5-0.8B.

Each example comes from a template family with randomized names, values and phrasing. Every
program is written to disk and run by real Bend2 (`bend FILE` and `bend FILE --check-only`);
it is kept only if both succeed and the printed value equals the Python value. Families listed
in HELD_OUT never appear in train/valid (test only).

Output (bendtune/data/): train.jsonl, valid.jsonl, test.jsonl in mlx_lm chat format
({"messages": [system, user, assistant]}) plus a "meta" field (family, expected), and
report.json (counts, rejects).
Run: python3 bendtune/gen_data.py [--n-train 800] [--n-valid 80] [--n-test 200] [--seed 0]
"""

import argparse
import json
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bendvm"))
from paths import bend2_main  # noqa: E402

HERE = Path(__file__).resolve().parent
BEND = bend2_main()
DATA = HERE / "data"
M = 2**32

SYSTEM = (
    "You write programs in BendLite, a subset of the Bend2 language. Rules: start with "
    "`import Base`; define each def before it is used; `def main() -> U32:` returns the answer. "
    "Types: U32 numbers (42), Nat counters (5n), Bool (True{}, False{}). A body is an expression "
    "or a `match` on a parameter: `case 0n:` / `case 1n+p:` for Nat, `case True{}:` / "
    "`case False{}:` for Bool. Arithmetic and comparisons go in `( ... : U32)` with spaces "
    "around operators: + - * / % < <= > >=. Equality is U32.is_eq(a, b); U32.from_nat(n) turns "
    "a Nat into a U32. Recursion must shrink the first changing argument (use a Nat fuel "
    "counter for loops). Use `+name: Type` for parameters used more than once. Two-space "
    "indentation. Reply with the program only."
)

FN = ["f", "go", "calc", "solve", "loop", "acc", "run", "step", "total", "work", "helper", "rec"]
NAT = ["n", "k", "m", "i", "c", "steps", "count", "fuel"]
U = ["x", "a", "b", "v", "y", "s", "t", "w", "q", "u"]  # no "z": templates use it


def pick(rng, pool, avoid=()):
    return rng.choice([p for p in pool if p not in avoid])


def prog(*defs, main):
    return "import Base\n\n" + "\n\n".join(defs) + f"\n\ndef main() -> U32:\n  {main}\n"


def bool_to_u32(name):
    return (f"def {name}(c: Bool) -> U32:\n  match c:\n    case True{{}}:\n      1\n"
            f"    case False{{}}:\n      0")


# --- families: rng -> (task, program, expected) ---------------------------------------------

def fam_sum_const(rng):
    n, k = rng.randint(1, 40), rng.randint(1, 50)
    f, a = pick(rng, FN), pick(rng, NAT)
    t = rng.choice([f"Add the number {k} to itself {n} times.", f"Compute {k} times {n} by repeated addition.",
                    f"Sum {n} copies of {k}."])
    d = (f"def {f}(+{a}: Nat) -> U32:\n  match {a}:\n    case 0n:\n      0\n    case 1n+p:\n"
         f"      ({k} + {f}(p) : U32)")
    return t, prog(d, main=f"{f}({n}n)"), n * k % M


def fam_triangle(rng):
    n = rng.randint(1, 60)
    f, a = pick(rng, FN), pick(rng, NAT)
    t = rng.choice([f"Compute 1 + 2 + ... + {n}.", f"Return the sum of the integers from 1 to {n}.",
                    f"What is the {n}th triangular number?"])
    d = (f"def {f}(+{a}: Nat) -> U32:\n  match {a}:\n    case 0n:\n      0\n    case 1n+p:\n"
         f"      (U32.from_nat({a}) + {f}(p) : U32)")
    return t, prog(d, main=f"{f}({n}n)"), n * (n + 1) // 2


def fam_sum_squares(rng):
    n = rng.randint(1, 30)
    f, a = pick(rng, FN), pick(rng, NAT)
    t = rng.choice([f"Compute 1^2 + 2^2 + ... + {n}^2.", f"Return the sum of the squares of 1 to {n}."])
    d = (f"def {f}(+{a}: Nat) -> U32:\n  match {a}:\n    case 0n:\n      0\n    case 1n+p:\n"
         f"      (U32.from_nat({a}) * U32.from_nat({a}) + {f}(p) : U32)")
    return t, prog(d, main=f"{f}({n}n)"), sum(i * i for i in range(1, n + 1))


def fam_factorial(rng):
    n = rng.randint(0, 15)
    f, a = pick(rng, FN), pick(rng, NAT)
    t = rng.choice([f"Compute {n}! (factorial), wrapping modulo 2^32.", f"Return the factorial of {n} as a U32.",
                    f"Multiply 1 * 2 * ... * {n}."])
    d = (f"def {f}(+{a}: Nat) -> U32:\n  match {a}:\n    case 0n:\n      1\n    case 1n+p:\n"
         f"      (U32.from_nat({a}) * {f}(p) : U32)")
    r = 1
    for i in range(1, n + 1):
        r = r * i % M
    return t, prog(d, main=f"{f}({n}n)"), r


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % M
    return a


def fam_fib(rng):
    n = rng.randint(0, 40)
    f, a = pick(rng, FN), pick(rng, NAT)
    x, y = rng.sample(U, 2)
    t = rng.choice([f"Return the {n}th Fibonacci number (fib(0) = 0, fib(1) = 1).",
                    f"Compute Fibonacci number {n}.", f"What is fib({n})?"])
    d = (f"def {f}(+{a}: Nat, +{x}: U32, +{y}: U32) -> U32:\n  match {a}:\n    case 0n:\n"
         f"      {x}\n    case 1n+p:\n      {f}(p, {y}, ({x} + {y} : U32))")
    return t, prog(d, main=f"{f}({n}n, 0, 1)"), fib(n)


def fam_power(rng):
    b, e = rng.randint(0, 12), rng.randint(0, 12)
    f, a = pick(rng, FN), pick(rng, NAT)
    x = pick(rng, U)
    t = rng.choice([f"Compute {b} to the power {e}.", f"Return {b}^{e} as a U32.", f"Raise {b} to the {e}th power."])
    d = (f"def {f}(+{x}: U32, +{a}: Nat) -> U32:\n  match {a}:\n    case 0n:\n      1\n    case 1n+p:\n"
         f"      ({x} * {f}({x}, p) : U32)")
    return t, prog(d, main=f"{f}({b}, {e}n)"), pow(b, e, M)


def fam_gcd(rng):
    a_, b_ = rng.randint(1, 5000), rng.randint(1, 5000)
    f, g = rng.sample(FN, 2)
    x, y = rng.sample(U, 2)
    t = rng.choice([f"Compute the greatest common divisor of {a_} and {b_}.", f"Return gcd({a_}, {b_})."])
    d1 = (f"def {f}(+fuel: Nat, +{x}: U32, +{y}: U32, z: Bool) -> U32:\n  match fuel:\n    case 0n:\n"
          f"      {x}\n    case 1n+r:\n      match z:\n        case True{{}}:\n          {x}\n"
          f"        case False{{}}:\n          {f}(r, {y}, ({x} % {y} : U32), U32.is_eq(({x} % {y} : U32), 0))")
    d2 = f"def {g}(+{x}: U32, +{y}: U32) -> U32:\n  {f}(64n, {x}, {y}, U32.is_eq({y}, 0))"
    return t, prog(d1, d2, main=f"{g}({a_}, {b_})"), gcd(a_, b_)


def fam_max(rng):
    a_, b_ = rng.randint(0, 9999), rng.randint(0, 9999)
    f = pick(rng, FN)
    x, y = rng.sample(U, 2)
    big = rng.random() < 0.5
    t = (rng.choice([f"Return the larger of {a_} and {b_}.", f"Compute max({a_}, {b_})."]) if big else
         rng.choice([f"Return the smaller of {a_} and {b_}.", f"Compute min({a_}, {b_})."]))
    op = "<" if big else "<="
    first, second = (y, x) if big else (x, y)
    d = (f"def {f}(+{x}: U32, +{y}: U32, c: Bool) -> U32:\n  match c:\n    case True{{}}:\n      {first}\n"
         f"    case False{{}}:\n      {second}")
    return t, prog(d, main=f"{f}({a_}, {b_}, ({a_} {op} {b_} : U32))"), max(a_, b_) if big else min(a_, b_)


def fam_absdiff(rng):
    a_, b_ = rng.randint(0, 9999), rng.randint(0, 9999)
    f = pick(rng, FN)
    x, y = rng.sample(U, 2)
    t = rng.choice([f"Return the absolute difference between {a_} and {b_}.", f"Compute |{a_} - {b_}|."])
    d = (f"def {f}(+{x}: U32, +{y}: U32, c: Bool) -> U32:\n  match c:\n    case True{{}}:\n      ({y} - {x} : U32)\n"
         f"    case False{{}}:\n      ({x} - {y} : U32)")
    return t, prog(d, main=f"{f}({a_}, {b_}, ({a_} < {b_} : U32))"), abs(a_ - b_)


def fam_parity(rng):
    n = rng.randint(0, 60)
    f, g, h = rng.sample(FN, 3)
    a = pick(rng, NAT)
    t = rng.choice([f"Return 1 if {n} is even and 0 otherwise.", f"Is {n} even? Answer 1 for yes, 0 for no."])
    d1 = (f"def {f}(c: Bool) -> Bool:\n  match c:\n    case True{{}}:\n      False{{}}\n    case False{{}}:\n"
          f"      True{{}}")
    d2 = (f"def {g}(+{a}: Nat) -> Bool:\n  match {a}:\n    case 0n:\n      True{{}}\n    case 1n+p:\n"
          f"      {f}({g}(p))")
    return t, prog(d1, d2, bool_to_u32(h), main=f"{h}({g}({n}n))"), 1 if n % 2 == 0 else 0


def fam_count_multiples(rng):
    n, d_ = rng.randint(1, 60), rng.randint(1, 9)
    f, g, h = rng.sample(FN, 3)
    a = pick(rng, NAT)
    x = pick(rng, U)
    t = rng.choice([f"Count how many of the numbers 1 to {n} are divisible by {d_}.",
                    f"How many multiples of {d_} are there from 1 to {n}?"])
    d2 = f"def {g}(+{x}: U32, +k: U32) -> U32:\n  {h}(U32.is_eq((k % {x} : U32), 0))"
    d3 = (f"def {f}(+{a}: Nat, +{x}: U32) -> U32:\n  match {a}:\n    case 0n:\n      0\n    case 1n+p:\n"
          f"      ({g}({x}, U32.from_nat({a})) + {f}(p, {x}) : U32)")
    return t, prog(bool_to_u32(h), d2, d3, main=f"{f}({n}n, {d_})"), n // d_


def fam_digit_sum(rng):
    v = rng.randint(0, 10**9)
    f = pick(rng, FN)
    x = pick(rng, U)
    t = rng.choice([f"Return the sum of the decimal digits of {v}.", f"Add up the digits of {v}."])
    d = (f"def {f}(+fuel: Nat, +{x}: U32) -> U32:\n  match fuel:\n    case 0n:\n      0\n    case 1n+r:\n"
         f"      (({x} % 10 : U32) + {f}(r, ({x} / 10 : U32)) : U32)")
    return t, prog(d, main=f"{f}(12n, {v})"), sum(map(int, str(v)))


def fam_arith(rng):
    a_, b_, c_ = rng.randint(0, 999), rng.randint(0, 999), rng.randint(1, 99)
    form = rng.choice(["a + b * c", "a * b - c", "a / c + b", "a % c * b", "a + b - c"])
    expr = form.replace("a", str(a_)).replace("b", str(b_)).replace("c", str(c_))
    val = eval(form.replace("/", "//"), {"a": a_, "b": b_, "c": c_}) % M  # noqa: S307
    t = rng.choice([f"Compute {expr} with 32-bit unsigned arithmetic.", f"Evaluate {expr} (U32, wrapping)."])
    return t, f"import Base\n\ndef main() -> U32:\n  ({expr} : U32)\n", val


# held-out families (test only)
def fam_digit_count(rng):
    v = rng.randint(1, 10**9)
    f, g = rng.sample(FN, 2)
    x = pick(rng, U)
    t = rng.choice([f"Return the number of decimal digits of {v}.", f"How many digits does {v} have?"])
    d = (f"def {f}(+fuel: Nat, +{x}: U32, z: Bool) -> U32:\n  match fuel:\n    case 0n:\n      0\n"
         f"    case 1n+r:\n      match z:\n        case True{{}}:\n          0\n        case False{{}}:\n"
         f"          (1 + {f}(r, ({x} / 10 : U32), U32.is_eq(({x} / 10 : U32), 0)) : U32)")
    return t, prog(d, main=f"{f}(12n, {v}, U32.is_eq({v}, 0))"), len(str(v))


def fam_sum_multiples(rng):
    n, d_ = rng.randint(1, 60), rng.randint(1, 9)
    f, g, h = rng.sample(FN, 3)
    a = pick(rng, NAT)
    x = pick(rng, U)
    t = rng.choice([f"Return the sum of the numbers from 1 to {n} that are divisible by {d_}.",
                    f"Add up all multiples of {d_} between 1 and {n}."])
    d1 = (f"def {h}(+k: U32, c: Bool) -> U32:\n  match c:\n    case True{{}}:\n      k\n    case False{{}}:\n"
          f"      0")
    d2 = (f"def {f}(+{a}: Nat, +{x}: U32) -> U32:\n  match {a}:\n    case 0n:\n      0\n    case 1n+p:\n"
          f"      ({h}(U32.from_nat({a}), U32.is_eq((U32.from_nat({a}) % {x} : U32), 0)) + {f}(p, {x}) : U32)")
    return t, prog(d1, d2, main=f"{f}({n}n, {d_})"), sum(k for k in range(1, n + 1) if k % d_ == 0)


def fam_reverse_digits(rng):
    v = rng.randint(1, 999999)
    f = pick(rng, FN)
    x, y = rng.sample(U, 2)
    t = rng.choice([f"Reverse the decimal digits of {v} and return the number.",
                    f"Return {v} written backwards as a number."])
    d = (f"def {f}(+fuel: Nat, +{x}: U32, +{y}: U32) -> U32:\n  match fuel:\n    case 0n:\n      {y}\n"
         f"    case 1n+r:\n      {f}(r, ({x} / 10 : U32), ({y} * 10 + {x} % 10 : U32))")
    digits = len(str(v))
    return t, prog(d, main=f"{f}({digits}n, {v}, 0)"), int(str(v)[::-1])


FAMILIES = {f.__name__[4:]: f for f in [
    fam_sum_const, fam_triangle, fam_sum_squares, fam_factorial, fam_fib, fam_power, fam_gcd,
    fam_max, fam_absdiff, fam_parity, fam_count_multiples, fam_digit_sum, fam_arith,
    fam_digit_count, fam_sum_multiples, fam_reverse_digits]}
HELD_OUT = {"digit_count", "sum_multiples", "reverse_digits"}


def bend_ok(src: str, want: int, tmp: Path, name: str):
    path = tmp / f"{name}.bend"
    path.write_text(src)
    r = subprocess.run(["bun", str(BEND), str(path)], capture_output=True, text=True, timeout=300)
    c = subprocess.run(["bun", str(BEND), str(path), "--check-only"], capture_output=True, text=True,
                       timeout=300)
    out = (r.stdout + r.stderr).strip().splitlines()
    ok = r.returncode == 0 and out[-1:] == [str(want)] and "ALL PROOFS CHECK" in c.stdout + c.stderr
    return ok, (out[-1] if out else "") + (" | " + (c.stdout + c.stderr)[-200:] if not ok else "")


def example(task, src, want, fam):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"Write a BendLite program. Task: {task}"},
                         {"role": "assistant", "content": src}],
            "meta": {"family": fam, "expected": want}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-valid", type=int, default=80)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen = set()
    inside = [k for k in FAMILIES if k not in HELD_OUT]

    def draw(n, fams):
        out = []
        while len(out) < n:
            fam = rng.choice(fams)
            task, src, want = FAMILIES[fam](rng)
            if task in seen:
                continue
            seen.add(task)
            out.append((task, src, want, fam))
        return out

    train = draw(args.n_train, inside)
    valid = draw(args.n_valid, inside)
    n_out = args.n_test // 2
    test = draw(args.n_test - n_out, inside) + draw(n_out, sorted(HELD_OUT))
    DATA.mkdir(exist_ok=True)
    report = {"seed": args.seed, "held_out_families": sorted(HELD_OUT), "rejects": []}
    with tempfile.TemporaryDirectory() as td, ThreadPoolExecutor(args.workers) as pool:
        tmp = Path(td)
        for split, rows in (("train", train), ("valid", valid), ("test", test)):
            futs = [pool.submit(bend_ok, src, want, tmp, f"{split}_{i}")
                    for i, (_, src, want, _) in enumerate(rows)]
            kept = []
            for (task, src, want, fam), fut in zip(rows, futs):
                ok, info = fut.result()
                if ok:
                    kept.append(example(task, src, want, fam))
                else:
                    report["rejects"].append({"split": split, "family": fam, "task": task, "info": info})
            with open(DATA / f"{split}.jsonl", "w") as f:
                for ex in kept:
                    f.write(json.dumps(ex) + "\n")
            fams = {}
            for ex in kept:
                fams[ex["meta"]["family"]] = fams.get(ex["meta"]["family"], 0) + 1
            report[split] = {"kept": len(kept), "drawn": len(rows), "families": fams}
            print(split, report[split]["kept"], "/", len(rows), flush=True)
    (DATA / "report.json").write_text(json.dumps(report, indent=2))
    print("rejects:", len(report["rejects"]))
    for r in report["rejects"][:10]:
        print(" ", r)


if __name__ == "__main__":
    main()
