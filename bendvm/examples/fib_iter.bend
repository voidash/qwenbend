import Base

def fib_go(+n: Nat, +a: U32, +b: U32) -> U32:
  match n:
    case 0n:
      a
    case 1n+p:
      fib_go(p, b, (a + b : U32))

def main() -> U32:
  fib_go(30n, 0, 1)
