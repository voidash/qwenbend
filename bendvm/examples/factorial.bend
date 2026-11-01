import Base

def fact(+n: Nat) -> U32:
  match n:
    case 0n:
      1
    case 1n+p:
      (U32.from_nat(n) * fact(p) : U32)

def main() -> U32:
  fact(10n)
