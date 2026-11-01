import Base

def fib(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      match p:
        case 0n:
          1
        case 1n+q:
          (fib(p) + fib(q) : U32)

def main() -> U32:
  fib(12n)
