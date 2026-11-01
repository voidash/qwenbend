import Base

def sq(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (U32.from_nat(n) * U32.from_nat(n) + sq(p) : U32)

def main() -> U32:
  sq(10n)
