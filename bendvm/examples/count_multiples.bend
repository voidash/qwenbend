import Base

def bump(c: Bool) -> U32:
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
