import Base

def pick(+a: U32, +b: U32, c: Bool) -> U32:
  match c:
    case True{}:
      (b - a : U32)
    case False{}:
      (a - b : U32)

def main() -> U32:
  pick(9, 30, (9 < 30 : U32))
