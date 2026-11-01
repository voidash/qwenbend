# BendLite: the Bend2 subset Qwen writes and BendVM runs (spec v1)

Every BendLite program is a valid Bend2 program. Real `bend` runs it, and `--check-only`
passes. The rules below were learned from real Bend2 (bend @ 9e8d7b7) and its guide, and are
verified by the 20 examples in `examples/` (`make_examples.py`; all 20 match Python values,
`examples/RESULTS.tsv`).

## Program shape

```
import Base

def NAME(PARAM, PARAM, ...) -> TYPE:
  BODY

...

def main() -> U32:
  EXPR
```

- `import Base` is the first line. `main` takes nothing and returns `U32`. Real `bend` prints
  the value.
- **Definitions must appear before they are used** (a def calling one written below it is
  rejected).
- Indentation is two spaces per level.

## Types

| Type | Values | Notes |
|---|---|---|
| `U32` | `0` … `4294967295`, written `42` | arithmetic wraps mod 2^32 |
| `Nat` | `0n`, `1n`, … | only for counting down (recursion and fuel); no arithmetic in BendLite |
| `Bool` | `True{}`, `False{}` | from comparisons and `U32.is_eq` |

## Parameters

`+x: T` (usable more than once) or `x: T` (used at most once, e.g. a Bool that is only
matched). When unsure, use `+`; for U32 and Nat it is always allowed.

## Body

A body is either an expression, or a `match` on a **parameter** (never on a computed value):

```
  match n:          # n: Nat
    case 0n:
      EXPR_OR_MATCH
    case 1n+p:      # p = n - 1, usable in this branch
      EXPR_OR_MATCH

  match c:          # c: Bool
    case True{}:
      EXPR_OR_MATCH
    case False{}:
      EXPR_OR_MATCH
```

Matches nest (`fib` matches `p` inside the `1n+p` branch).

## Expressions

- Literals: `42`, `5n`, `True{}`, `False{}`. Variables are parameters or `1n+p` binders.
- Calls: `f(a, b)` to an earlier def, plus exactly three Base functions:
  - `U32.from_nat(n)`: Nat → U32;
  - `U32.is_eq(a, b)`: Bool equality (there is no `==` operator);
  - `U32.min(a, b)` and `U32.max(a, b)` (allowed; not in the examples yet).
- Arithmetic and comparisons go **inside a typed group** `( ... : U32)`, with spaces around
  every operator:
  - `+ - * / %` give U32. Precedence: `* / %` over `+ -`; left-associative.
  - `< <= > >=` give Bool (`(a < b : U32)`).
  - Without `: U32` the operators are refused.

## Recursion

- Structural only. A recursive call must pass a smaller part of a parameter (`p` from
  `1n+p`).
- The arguments are read **left to right**: each must be passed unchanged until one is
  smaller. So **the shrinking parameter goes first** (`gcd_go(fuel, a, b, z)` works;
  `gcd_go(a, b, fuel, z)` is rejected).
- Loops that aren't naturally structural count down a `Nat` fuel parameter (gcd, collatz,
  digit sum).
- Mutual recursion is not allowed.

## Not in BendLite v1

`if` (Bend2 has none: use `match` on a Bool), `let` statements, lists, tuples, user types,
strings, floats, closures, arrays, IO, laws/proofs, `!` parallel calls, `@unsafe`.

## Examples (all verified against real `bend`)

lit, arith, divmod, wrap, sum_ones, triangle, factorial, fib, fib_iter, power, power_wrap,
max, min3, abs_diff, gcd, is_even, count_multiples, sum_squares, collatz_steps, digit_sum.
