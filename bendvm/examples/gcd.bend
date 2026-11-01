import Base

def gcd_go(+fuel: Nat, +a: U32, +b: U32, z: Bool) -> U32:
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
