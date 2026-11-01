import Base

def ds(+fuel: Nat, +x: U32) -> U32:
  match fuel:
    case 0n:
      0
    case 1n+f:
      ((x % 10 : U32) + ds(f, (x / 10 : U32)) : U32)

def main() -> U32:
  ds(12n, 987654321)
