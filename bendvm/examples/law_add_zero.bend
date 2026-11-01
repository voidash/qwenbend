import Base

# CLAIM: for every nat x, x plus 0 equals x.
law add_zero:
  for x: Nat
  {Nat.add(x, 0n) == x : Nat}

# PROOF: induction on `x`, one rewrite (`%`) per step.
def add_zero(x):
  match x:
    case 0n:
      {==}
    case 1n+xp:
      %add_zero(xp) : {1n+Nat.add(xp, 0n) == 1n+_ : Nat}
      {==}

def sum(+n: Nat) -> U32:
  match n:
    case 0n:
      0
    case 1n+p:
      (1 + sum(p) : U32)

def main() -> IO(Unit):
  do IO<Unit>:
    IO.print(U32.show(sum(Nat.add(10n, 0n))))
