import Base

def tri(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (U32.from_nat(n) + tri(p) : U32)

def main() -> U32:
  tri(20n)
