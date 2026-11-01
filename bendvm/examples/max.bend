import Base

def pick(+a: U32, +b: U32, c: Bool) -> U32:
  match c:
    case True{}:
      b
    case False{}:
      a

def main() -> U32:
  pick(7, 12, (7 < 12 : U32))
