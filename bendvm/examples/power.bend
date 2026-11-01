import Base

def pow(+b: U32, +e: Nat) -> U32:
  match e:
    case 0n:
      1
    case 1n+p:
      (b * pow(b, p) : U32)

def main() -> U32:
  pow(3, 7n)
