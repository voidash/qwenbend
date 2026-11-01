import Base

def odd_step(+x: U32, c: Bool) -> U32:
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
