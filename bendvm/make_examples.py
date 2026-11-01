"""Writes the BendLite hand examples to bendvm/examples/ and checks each with real Bend2:
`bend FILE` must print the Python-computed value and `bend FILE --check-only` must pass.
Writes examples/RESULTS.tsv (name, expected, bend output, check, ok).
Run: python3 bendvm/make_examples.py"""

import subprocess
import sys
from math import gcd
from pathlib import Path
from paths import bend2_main  # noqa: E402

HERE = Path(__file__).resolve().parent
BEND = bend2_main()
OUT = HERE / "examples"
U32 = 2**32


def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# (name, source, expected value)
EXAMPLES = [
    ("lit", "def main() -> U32:\n  42\n", 42),
    ("arith", "def main() -> U32:\n  (3 + 4 * 5 - 6 : U32)\n", 3 + 4 * 5 - 6),
    ("divmod", "def main() -> U32:\n  (100 / 7 + 100 % 7 : U32)\n", 100 // 7 + 100 % 7),
    ("wrap", "def main() -> U32:\n  (0 - 1 : U32)\n", (0 - 1) % U32),
    ("sum_ones", """def sum(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (1 + sum(p) : U32)

def main() -> U32:
  sum(10n)
""", 10),
    ("triangle", """def tri(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (U32.from_nat(n) + tri(p) : U32)

def main() -> U32:
  tri(20n)
""", 20 * 21 // 2),
    ("factorial", """def fact(+n: Nat) -> U32:
  match n:
    case 0n:
      1
    case 1n+p:
      (U32.from_nat(n) * fact(p) : U32)

def main() -> U32:
  fact(10n)
""", 3628800),
    ("fib", """def fib(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      match p:
        case 0n:
          1
        case 1n+q:
          (fib(p) + fib(q) : U32)

def main() -> U32:
  fib(12n)
""", fib(12)),
    ("fib_iter", """def fib_go(+n: Nat, +a: U32, +b: U32) -> U32:
  match n:
    case 0n:
      a
    case 1n+p:
      fib_go(p, b, (a + b : U32))

def main() -> U32:
  fib_go(30n, 0, 1)
""", fib(30)),
    ("power", """def pow(+b: U32, +e: Nat) -> U32:
  match e:
    case 0n:
      1
    case 1n+p:
      (b * pow(b, p) : U32)

def main() -> U32:
  pow(3, 7n)
""", 3**7),
    ("power_wrap", """def pow(+b: U32, +e: Nat) -> U32:
  match e:
    case 0n:
      1
    case 1n+p:
      (b * pow(b, p) : U32)

def main() -> U32:
  pow(7, 20n)
""", 7**20 % U32),
    ("max", """def pick(+a: U32, +b: U32, c: Bool) -> U32:
  match c:
    case True{}:
      b
    case False{}:
      a

def main() -> U32:
  pick(7, 12, (7 < 12 : U32))
""", 12),
    ("min3", """def pick(+a: U32, +b: U32, c: Bool) -> U32:
  match c:
    case True{}:
      a
    case False{}:
      b

def min(+a: U32, +b: U32) -> U32:
  pick(a, b, (a <= b : U32))

def main() -> U32:
  min(min(40, 17), 23)
""", 17),
    ("abs_diff", """def pick(+a: U32, +b: U32, c: Bool) -> U32:
  match c:
    case True{}:
      (b - a : U32)
    case False{}:
      (a - b : U32)

def main() -> U32:
  pick(9, 30, (9 < 30 : U32))
""", 21),
    ("gcd", """def gcd_go(+fuel: Nat, +a: U32, +b: U32, z: Bool) -> U32:
  match fuel:
    case 0n:
      a
    case 1n+f:
      match z:
        case True{}:
          a
        case False{}:
          gcd_go(f, b, (a % b : U32), U32.is_eq((a % b : U32), 0))

def gcd(+a: U32, +b: U32) -> U32:
  gcd_go(64n, a, b, U32.is_eq(b, 0))

def main() -> U32:
  gcd(1071, 462)
""", gcd(1071, 462)),
    ("is_even", """def flip(c: Bool) -> Bool:
  match c:
    case True{}:
      False{}
    case False{}:
      True{}

def even(+n: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+p:
      flip(even(p))

def to_u32(c: Bool) -> U32:
  match c:
    case True{}:
      1
    case False{}:
      0

def main() -> U32:
  to_u32(even(14n))
""", 1),
    ("count_multiples", """def bump(c: Bool) -> U32:
  match c:
    case True{}:
      1
    case False{}:
      0

def hit(+d: U32, +k: U32) -> U32:
  bump(U32.is_eq((k % d : U32), 0))

def count(+n: Nat, +d: U32) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (hit(d, U32.from_nat(n)) + count(p, d) : U32)

def main() -> U32:
  count(30n, 4)
""", sum(1 for k in range(1, 31) if k % 4 == 0)),
    ("sum_squares", """def sq(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (U32.from_nat(n) * U32.from_nat(n) + sq(p) : U32)

def main() -> U32:
  sq(10n)
""", sum(k * k for k in range(1, 11))),
    ("collatz_steps", """def odd_step(+x: U32, c: Bool) -> U32:
  match c:
    case True{}:
      (x / 2 : U32)
    case False{}:
      (3 * x + 1 : U32)

def steps(+fuel: Nat, +x: U32, done: Bool) -> U32:
  match fuel:
    case 0n:
      0
    case 1n+f:
      match done:
        case True{}:
          0
        case False{}:
          (1 + steps(f, odd_step(x, U32.is_eq((x % 2 : U32), 0)), U32.is_eq(odd_step(x, U32.is_eq((x % 2 : U32), 0)), 1)) : U32)

def main() -> U32:
  steps(200n, 27, U32.is_eq(27, 1))
""", None),  # filled below
    ("digit_sum", """def ds(+fuel: Nat, +x: U32) -> U32:
  match fuel:
    case 0n:
      0
    case 1n+f:
      ((x % 10 : U32) + ds(f, (x / 10 : U32)) : U32)

def main() -> U32:
  ds(12n, 987654321)
""", sum(int(c) for c in "987654321")),
]


def collatz(x):
    s = 0
    while x != 1:
        x = x // 2 if x % 2 == 0 else 3 * x + 1
        s += 1
    return s


EXAMPLES = [(n, s, collatz(27) if n == "collatz_steps" else v) for n, s, v in EXAMPLES]


def bend(path, *args):
    r = subprocess.run(["bun", str(BEND), str(path), *args], capture_output=True, text=True,
                       timeout=300)
    return r.returncode, (r.stdout + r.stderr).strip()


def main():
    OUT.mkdir(exist_ok=True)
    rows, bad = [], 0
    for name, src, want in EXAMPLES:
        path = OUT / f"{name}.bend"
        path.write_text("import Base\n\n" + src)
        rc, out = bend(path)
        crc, cout = bend(path, "--check-only")
        ok = rc == 0 and out.splitlines()[-1:] == [str(want)] and "ALL PROOFS CHECK" in cout
        bad += not ok
        rows.append((name, str(want), out.splitlines()[-1] if out else "", "check" if "ALL PROOFS CHECK" in cout else "FAIL", "ok" if ok else "MISMATCH"))
        print(f"{'ok  ' if ok else 'FAIL'} {name:<16} want {want:<12} bend {out.splitlines()[-1] if out else '' !s:<12}"
              + ("" if ok else f"\n  {out[-300:]}\n  {cout[-200:]}"))
    (OUT / "RESULTS.tsv").write_text("name\texpected\tbend\tcheck\tok\n"
                                     + "".join("\t".join(r) + "\n" for r in rows))
    print(f"{len(EXAMPLES) - bad}/{len(EXAMPLES)} examples ok")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
