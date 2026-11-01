import Base

def pick(+a: U32, +b: U32, c: Bool) -> U32:
  match c:
    case True{}:
      a
    case False{}:
      b

def min(+a: U32, +b: U32) -> U32:
  pick(a, b, (a <= b : U32))

def main() -> U32:
  min(min(40, 17), 23)
