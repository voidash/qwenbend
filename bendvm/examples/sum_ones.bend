import Base

def sum(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (1 + sum(p) : U32)

def main() -> U32:
  sum(10n)
