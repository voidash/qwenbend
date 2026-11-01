import Base

def flip(c: Bool) -> Bool:
  match c:
    case True{}:
      False{}
    case False{}:
      True{}

def even(+n: Nat) -> Bool:
  match n:
    case 0n:
      True{}
    case 1n+p:
      flip(even(p))

def to_u32(c: Bool) -> U32:
  match c:
    case True{}:
      1
    case False{}:
      0

def main() -> U32:
  to_u32(even(14n))
