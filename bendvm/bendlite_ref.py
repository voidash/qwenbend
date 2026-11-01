"""BendLite reference interpreter (step 5a): runs a BendLite program given as Qwen token ids.

This fixes the exact semantics BendVM must reproduce: no compiler, no Bend2. The input is what
Qwen writes (its token ids); token ids are decoded to bytes through the tokenizer's table
(the engine will hold that table in ROM), then the program is lexed, parsed and evaluated.

Semantics (BENDLITE.md): U32 wraps mod 2^32; Nat counts down via `case 1n+p`; Bool via
True{}/False{}; `( a op b ... : U32)` with * / % over + - (left-assoc) and one optional
comparison < <= > >= giving Bool; U32.from_nat, U32.is_eq, U32.min, U32.max; defs must precede
their use; main() returns U32. Anything outside the subset raises BendLiteError.

Run: python3 bendvm/bendlite_ref.py FILE.bend      (prints the value)
"""

import re
import sys
from dataclasses import dataclass

M = 2**32


class BendLiteError(Exception):
    pass


MAX_DEPTH = 256  # nested def calls (capture.MAX_DEPTH); deeper programs are errors


# --- lexer: lines with indentation -> tokens ------------------------------------------------

TOK = re.compile(r"\s*(\d+n|\d+|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*|->|<=|>=|"
                 r"1n\+|\{\}|[()+\-*/%<>:,=])")


def lines_of(src: str):
    out = []
    for raw in src.split("\n"):
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        ind = len(raw) - len(raw.lstrip(" "))
        if "\t" in raw[:ind + 1]:
            raise BendLiteError("tabs are not allowed")
        out.append((ind, raw.strip()))
    return out


def tokens_of(text: str):
    toks, i = [], 0
    while i < len(text):
        if text[i:].strip() == "":
            break
        m = TOK.match(text, i)
        if not m:
            raise BendLiteError(f"unexpected text {text[i:i + 15]!r}")
        toks.append(m[1])
        i = m.end()
    return toks


# --- AST ----------------------------------------------------------------------------------------

@dataclass
class Def:
    name: str
    params: list  # [(name, type)]
    ret: str
    body: object
    plus: list  # per parameter: declared `+` (usable more than once)


# bodies: ("match", var, [(pattern, body)]) | ("expr", expr)
# exprs: ("u32", v) | ("nat", v) | ("bool", b) | ("var", x) | ("call", f, [args])
#        | ("arith", items) where items alternate operands and ops | ("cmp", op, a, b)


class Parser:
    def __init__(self, src: str):
        self.lines = lines_of(src)
        self.i = 0

    def peek(self):
        return self.lines[self.i] if self.i < len(self.lines) else None

    def program(self):
        ind, text = self.peek() or (0, "")
        if text != "import Base":
            raise BendLiteError("a program starts with `import Base`")
        self.i += 1
        defs = {}
        while self.peek():
            d = self.define()
            if d.name in defs:
                raise BendLiteError(f"def {d.name} defined twice")
            defs[d.name] = d
        if "main" not in defs or defs["main"].params or defs["main"].ret != "U32":
            raise BendLiteError("needs `def main() -> U32:`")
        return defs

    def define(self):
        ind, text = self.peek()
        if ind != 0:
            raise BendLiteError(f"expected a def at column 0: {text!r}")
        m = re.fullmatch(r"def ([A-Za-z_][A-Za-z0-9_]*)\((.*)\) -> (U32|Nat|Bool):", text)
        if not m:
            raise BendLiteError(f"bad def line {text!r}")
        params, plus = [], []
        for p in [x.strip() for x in m[2].split(",") if x.strip()]:
            pm = re.fullmatch(r"(\+?)([A-Za-z_][A-Za-z0-9_]*): (U32|Nat|Bool)", p)
            if not pm:
                raise BendLiteError(f"bad parameter {p!r}")
            params.append((pm[2], pm[3]))
            plus.append(pm[1] == "+")
        self.i += 1
        body = self.body(2)
        return Def(m[1], params, m[3], body, plus)

    def body(self, ind):
        line = self.peek()
        if not line or line[0] != ind:
            raise BendLiteError("expected an indented body")
        text = line[1]
        mm = re.fullmatch(r"match ([A-Za-z_][A-Za-z0-9_]*):", text)
        if mm:
            self.i += 1
            arms = []
            while self.peek() and self.peek()[0] == ind + 2 and self.peek()[1].startswith("case "):
                pat = self.peek()[1][5:]
                if not pat.endswith(":"):
                    raise BendLiteError(f"bad case {pat!r}")
                pat = pat[:-1]
                if pat == "0n":
                    p = ("zero",)
                elif re.fullmatch(r"1n\+[A-Za-z_][A-Za-z0-9_]*", pat):
                    p = ("succ", pat[3:])
                elif pat in ("True{}", "False{}"):
                    p = ("bool", pat == "True{}")
                else:
                    raise BendLiteError(f"bad pattern {pat!r}")
                self.i += 1
                arms.append((p, self.body(ind + 4)))
            if len(arms) != 2:
                raise BendLiteError("a match needs exactly two cases")
            kinds = {a[0][0] for a in arms}
            if kinds not in ({"zero", "succ"}, {"bool"}):
                raise BendLiteError("mixed case patterns")
            return ("match", mm[1], arms)
        self.i += 1
        return ("expr", Expr(tokens_of(text)).whole())


class Expr:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, x=None):
        tok = self.peek()
        if tok is None or (x is not None and tok != x):
            raise BendLiteError(f"expected {x!r}, got {tok!r}")
        self.i += 1
        return tok

    def whole(self):
        e = self.atom()
        if self.peek() is not None:
            raise BendLiteError(f"trailing {self.t[self.i:]}")
        return e

    def atom(self):
        tok = self.eat()
        if re.fullmatch(r"\d+n", tok):
            if int(tok[:-1]) >= M:
                raise BendLiteError("Nat literal too large")
            return ("nat", int(tok[:-1]))
        if tok.isdigit():
            v = int(tok)
            if v >= M:
                raise BendLiteError("U32 literal too large")
            return ("u32", v)
        if tok in ("True", "False"):
            self.eat("{}")
            return ("bool", tok == "True")
        if tok == "(":
            items = [self.atom()]
            cmp = None
            while self.peek() in ("+", "-", "*", "/", "%", "<", "<=", ">", ">="):
                op = self.eat()
                if op in ("<", "<=", ">", ">="):
                    if cmp is not None:
                        raise BendLiteError("one comparison per group")
                    cmp = (op, len(items))  # the operator's index in items
                items += [op, self.atom()]
            if len(items) == 1 and self.peek() == ")":  # (e): a plain parenthesis
                self.eat(")")
                return items[0]
            self.eat(":")
            ty = self.eat()
            self.eat(")")
            if len(items) == 1:  # (e : T): an annotation; e must have type T
                if ty not in ("U32", "Nat", "Bool"):
                    raise BendLiteError(f"bad annotation {ty}")
                return ("annot", ty, items[0])
            if ty != "U32":  # Bend2 has (a + b : Nat) too: outside BendLite
                raise BendLiteError("operators are typed U32")
            if cmp:
                k = cmp[1]
                left, right = items[:k], items[k + 1:]

                def side(xs):
                    return xs[0] if len(xs) == 1 else ("arith", xs)
                return ("cmp", cmp[0], side(left), side(right))
            return ("arith", items)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", tok):
            if self.peek() == "(":
                self.eat("(")
                args = []
                if self.peek() != ")":
                    args.append(self.atom())
                    while self.peek() == ",":
                        self.eat(",")
                        args.append(self.atom())
                self.eat(")")
                return ("call", tok, args)
            return ("var", tok)
        raise BendLiteError(f"unexpected {tok!r}")


# --- evaluation -----------------------------------------------------------------------------------

BUILTINS = {"U32.from_nat", "U32.is_eq", "U32.min", "U32.max"}


def arith(items):
    # * / % bind tighter than + -, all left-associative
    terms, ops = [items[0]], []
    for op, v in zip(items[1::2], items[2::2]):
        if op in ("*", "/", "%"):
            a = terms.pop()
            # Bend2 (measured): x / 0 = 0 and x % 0 = x
            terms.append(a * v % M if op == "*" else (a // v if v else 0) if op == "/"
                         else (a % v if v else a))
        else:
            ops.append(op)
            terms.append(v)
    acc = terms[0]
    for op, v in zip(ops, terms[1:]):
        acc = (acc + v) % M if op == "+" else (acc - v) % M
    return acc


class Machine:
    def __init__(self, defs, max_steps=10_000_000):
        self.defs, self.steps, self.max_steps = defs, 0, max_steps
        self.depth = 0

    def call(self, name, args, order):
        if name in BUILTINS:
            if len(args) != (1 if name == "U32.from_nat" else 2) or any(isinstance(a, bool) for a in args):
                raise BendLiteError("builtin arity or Bool argument")
            if name == "U32.from_nat":
                return args[0] % M
            if name == "U32.is_eq":
                return args[0] == args[1]
            return min(args) if name == "U32.min" else max(args)
        d = self.defs.get(name)
        if d is None or order.get(name, 1 << 30) >= order["__current"]:
            raise BendLiteError(f"unknown or later def {name}")
        if len(args) != len(d.params):
            raise BendLiteError(f"{name} takes {len(d.params)} arguments")
        return self.run(d, dict(zip((p for p, _ in d.params), args)), order)

    def run(self, d, env, order):
        self.steps += 1
        if self.steps > self.max_steps:
            raise BendLiteError("step limit")
        saved = order["__current"]
        order["__current"] = order[d.name] + 1  # a def may call itself and earlier defs
        self.depth += 1
        try:
            if self.depth > MAX_DEPTH:
                raise BendLiteError("call depth")
            return self.body(d.body, env, order)
        finally:
            order["__current"] = saved
            self.depth -= 1

    def body(self, b, env, order):
        if b[0] == "expr":
            return self.expr(b[1], env, order)
        _, var, arms = b
        if var not in env:
            raise BendLiteError(f"match on unknown {var}")
        v = env[var]
        for pat, sub in arms:
            if pat[0] == "zero" and v == 0 and not isinstance(v, bool):
                return self.body(sub, env, order)
            if pat[0] == "succ" and not isinstance(v, bool) and v > 0:
                return self.body(sub, {**env, pat[1]: v - 1}, order)
            if pat[0] == "bool" and isinstance(v, bool) and v == pat[1]:
                return self.body(sub, env, order)
        raise BendLiteError(f"no case matches {var}")

    def expr(self, e, env, order):
        k = e[0]
        if k in ("u32", "nat", "bool"):
            return e[1]
        if k == "annot":
            return self.expr(e[2], env, order)
        if k == "var":
            if e[1] not in env:
                raise BendLiteError(f"unknown variable {e[1]}")
            return env[e[1]]
        if k == "call":
            return self.call(e[1], [self.expr(a, env, order) for a in e[2]], order)
        if k == "arith":
            vals = [self.expr(x, env, order) if i % 2 == 0 else x for i, x in enumerate(e[1])]
            if any(isinstance(v, bool) for v in vals[0::2]):
                raise BendLiteError("Bool in arithmetic")
            return arith(vals)
        if k == "cmp":
            a, b = self.expr(e[2], env, order), self.expr(e[3], env, order)
            if isinstance(a, bool) or isinstance(b, bool):
                raise BendLiteError("Bool in a comparison")
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[e[1]]
        raise BendLiteError(f"bad expression {e}")


# --- static checks: what Bend2 rejects before running (S2) -------------------------------------
# Rules measured with real Bend2 (s2_probe.py, s2_mutants.py) and read from bend2/bend.ts
# (match_flatten, term_descend, infer-ref, uses):
# - types: U32 / Nat / Bool, literals 5 vs 5n vs True{}; groups take U32 operands; a comparison
#   group is Bool; builtins U32.from_nat(Nat), U32.is_eq(U32, U32) -> Bool, U32.min/max; call
#   arguments and def bodies must have the declared types; every def is checked, used or not.
# - match: the subject must be a pending binder (the params in order; matching v closes the
#   binders before it and puts v's fields in its place); Nat needs 0n and 1n+p, Bool True{} and
#   False{}, each once. Inside an arm v IS its pattern (0n, 1n+p, True{}).
# - affine: a param without + (and a binder of a match on one) is used at most once; uses add
#   across an expression and take the max across arms; matching is not a use.
# - termination: a self-call's arguments are compared with the def's columns left to right
#   (each equal until one is smaller); a matched column is its pattern tree.

class V:
    """A variable: a parameter or a match binder."""
    def __init__(self, name, ty, many):
        self.name, self.ty, self.many, self.pat = name, ty, many, None  # pat: None | ("zero",)
        # | ("succ", V) | ("bool", b), set while checking inside that arm


def expand(v):
    if v.pat is None:
        return ("var", v)
    if v.pat[0] == "succ":
        return ("succ", expand(v.pat[1]))
    return ("zero",) if v.pat[0] == "zero" else (("true",) if v.pat[1] else ("false",))


def descend(a, p):
    """Bend2's term_descend for BendLite: 'EQ', 'LT' or 'GT'."""
    if a[0] == "lit" and p[0] != "var":
        a = ("zero",) if a[1] == 0 else ("succ", ("lit", a[1] - 1))
    if p[0] == "var":
        return "EQ" if a[0] == "var" and a[1] is p[1] else "GT"
    if a[0] == p[0]:
        if p[0] != "succ":
            return "EQ"
        fld = descend(a[1], p[1])
        if fld != "GT":
            return fld
        return "GT"  # the only field is the bad one
    if p[0] == "succ" and descend(a, p[1]) != "GT":
        return "LT"
    return "GT"


def add_uses(*us):
    out = {}
    for u in us:
        for k, n in u.items():
            out[k] = out.get(k, 0) + n
    return out


class Checker:
    def __init__(self, defs, order):
        self.defs, self.order = defs, order

    def check(self, d):
        self.d = d
        self.cols = [V(n, t, pl) for (n, t), pl in zip(d.params, d.plus)]
        scope = {v.name: v for v in self.cols}  # a later duplicate name wins
        uses = self.body(d.body, scope, list(self.cols))
        for v, n in uses.items():
            if n > 1 and not v.many:
                raise BendLiteError(f"{v.name} consumed more than once")

    def body(self, b, scope, pending):
        if b[0] == "expr":
            t, u = self.expr(b[1], scope)
            if t != self.d.ret:
                raise BendLiteError(f"body is {t}, expected {self.d.ret}")
            return u
        _, name, arms = b
        v = scope.get(name)
        if v is None or not any(v is w for w in pending):
            raise BendLiteError(f"{name} can't be matched in this position")
        i = next(k for k, w in enumerate(pending) if w is v)
        want = {"Nat": {"zero", "succ"}, "Bool": {True, False}}.get(v.ty)
        got = [p[0] if p[0] != "bool" else p[1] for p, _ in arms]
        if want is None or set(got) != want or len(got) != 2:
            raise BendLiteError(f"match on {v.ty} needs one case per constructor")
        uses = {}
        for pat, sub in arms:
            sc, fields = scope, []
            if pat[0] == "succ":
                bv = V(pat[1], "Nat", v.many)
                sc, fields, v.pat = {**scope, pat[1]: bv}, [bv], ("succ", bv)
            else:
                v.pat = pat
            try:
                u = self.body(sub, sc, fields + pending[i + 1:])
            finally:
                v.pat = None
            uses = {k: max(uses.get(k, 0), u.get(k, 0)) for k in {*uses, *u}}
        return uses

    def var_uses(self, v):
        if v.pat is None:
            return {v: 1}
        return self.var_uses(v.pat[1]) if v.pat[0] == "succ" else {}

    def expr(self, e, scope):
        k = e[0]
        if k in ("u32", "nat", "bool"):
            return {"u32": "U32", "nat": "Nat", "bool": "Bool"}[k], {}
        if k == "var":
            v = scope.get(e[1])
            if v is None:
                raise BendLiteError(f"unknown variable {e[1]}")
            return v.ty, self.var_uses(v)
        if k == "annot":
            t, u = self.expr(e[2], scope)
            if t != e[1]:
                raise BendLiteError(f"annotated {e[1]}, is {t}")
            return t, u
        if k in ("arith", "cmp"):
            parts = e[1][0::2] if k == "arith" else [e[2], e[3]]
            ts, us = zip(*(self.expr(x, scope) for x in parts))
            if any(t != "U32" for t in ts):
                raise BendLiteError("a group takes U32 operands")
            return ("U32" if k == "arith" else "Bool"), add_uses(*us)
        # call
        name, args = e[1], e[2]
        ts, us = zip(*(self.expr(a, scope) for a in args)) if args else ((), ())
        uses = add_uses(*us)
        if name in BUILTINS:
            sig = {"U32.from_nat": (["Nat"], "U32"), "U32.is_eq": (["U32", "U32"], "Bool"),
                   "U32.min": (["U32", "U32"], "U32"), "U32.max": (["U32", "U32"], "U32")}[name]
            if list(ts) != sig[0]:
                raise BendLiteError(f"{name} takes {sig[0]}")
            return sig[1], uses
        d = self.defs.get(name)
        if d is None or self.order[name] > self.order[self.d.name]:
            raise BendLiteError(f"unknown or later def {name}")
        if list(ts) != [t for _, t in d.params]:
            raise BendLiteError(f"{name} takes {[t for _, t in d.params]}")
        if d is self.d:
            ord_ = "EQ"
            for a, col in zip(args, self.cols):
                while a[0] == "annot":  # Bend2 strips annotations (term_strip)
                    a = a[2]
                at = (expand(scope[a[1]]) if a[0] == "var" else ("lit", a[1]) if a[0] == "nat"
                      else (("true",) if a[1] else ("false",)) if a[0] == "bool" else ("other",))
                ord_ = descend(at, expand(col))
                if ord_ != "EQ":
                    break
            if ord_ != "LT":
                raise BendLiteError("a decreasing self-call")
        return d.ret, uses


def check(defs):
    order = {name: i for i, name in enumerate(defs)}
    c = Checker(defs, order)
    for d in defs.values():
        c.check(d)


def run_source(src: str):
    defs = Parser(src).program()
    check(defs)
    order = {name: i for i, name in enumerate(defs)}
    order["__current"] = len(defs) + 1
    m = Machine(defs)
    v = m.run(defs["main"], {}, order)
    if isinstance(v, bool):
        raise BendLiteError("main must return U32")
    return v, m.steps


def run_tokens(ids, decode):
    """The engine's view: token ids in, value out. decode(id) -> the token's text."""
    return run_source("".join(decode(i) for i in ids))


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    print(run_source(open(sys.argv[1]).read())[0])
