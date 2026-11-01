"""BendVM capture (5b-1): Qwen token ids -> lexeme tape, as the engine's hardware will do it while
Qwen writes (no trace tokens), plus a tape walker that evaluates a program from the tape alone
(the logic the firmware will implement in 5b-3).

ROM: for every token id in the vocabulary, the token's text if it consists only of BendLite
characters, else INVALID. The capture state machine reads those characters and appends tape
entries:
  NAME (hash, length)  a run of [A-Za-z0-9_.] starting with a letter or _ (U32.is_eq is one name)
  NUM value            a run of digits not followed by 'n'
  NAT value            a run of digits followed by 'n'
  P code               punctuation: ( ) , : + - * / % < > { } and the pairs -> <= >=
  NL column            a line break; column = indentation of the next non-empty line
Names are compared by (hash, length) only, as the engine will do.

Run: python bendvm/capture.py   (self-check on the dataset)
"""

import json
import random
import sys
from pathlib import Path
from paths import BENDTUNE, qwen_dir  # noqa: E402

NAMECH = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
PUNCT = set("(),:+-*/%<>{}=")
ALLOWED = NAMECH | PUNCT | {" ", "\n"}
M = 2**32


class CaptureError(Exception):
    pass


# Name keys: an XOR / nibble-rotate hash, key <- rotl(key, 4) ^ CHAR_CODE[ch] per character, so a
# name split across tokens hashes the same piecewise (key <- rotl(key, 4 len(part)) ^ H(part)),
# and the in-circuit lexer needs only XOR and a nibble rotate (DESIGN.md "C3 design").
CHAR_CODE = {ch: random.Random(f"bendvm-char-{ch}").getrandbits(32) for ch in sorted(NAMECH)}


def rotl4(x: int, k: int = 1) -> int:
    k = (4 * k) % 32
    return ((x << k) | (x >> (32 - k))) & 0xFFFFFFFF if k else x


def name_hash(s: str) -> tuple[int, int]:
    h = 0
    for ch in s:
        h = rotl4(h) ^ CHAR_CODE[ch]
    return h, len(s)


def build_rom(decode, vocab_size: int) -> dict:
    """token id -> text (BendLite characters only); ids absent from the dict are INVALID."""
    rom = {}
    for i in range(vocab_size):
        s = decode(i)
        if s and all(ch in ALLOWED for ch in s):
            rom[i] = s
    return rom


class Capture:
    """The capture state machine. feed() takes one token's text; the state carries across
    tokens (a name or number may span several)."""

    def __init__(self):
        self.tape = []
        self.word = ""  # current name or number characters
        self.pending = None  # a punctuation char that may pair with the next (- < >)
        self.col = None  # counting indentation after a newline
        self.invalid = False

    def _flush_word(self):
        w = self.word
        if not w:
            return
        self.word = ""
        if w[0].isdigit():
            if w.isdigit():
                self.tape.append(("NUM", int(w)))
            elif w[:-1].isdigit() and w[-1] == "n":
                self.tape.append(("NAT", int(w[:-1])))
            else:
                self.invalid = True
        else:
            self.tape.append(("NAME", name_hash(w)))

    def _flush_pending(self, nxt=None):
        if self.pending is None:
            return False
        p, self.pending = self.pending, None
        if nxt is not None and p + nxt in ("->", "<=", ">="):
            self.tape.append(("P", p + nxt))
            return True
        self.tape.append(("P", p))
        return False

    def char(self, ch):
        if self.col is not None:
            if ch == " ":
                self.col += 1
                return
            if ch == "\n":
                self.col = 0
                return
            self.tape.append(("NL", self.col))
            self.col = None
        if ch in NAMECH and not (ch == "." and not self.word):
            if self.pending is not None:
                self._flush_pending()
            # a digit run followed by a letter other than the Nat suffix is invalid (caught later)
            self.word += ch
            return
        self._flush_word()
        if self.pending is not None and self._flush_pending(ch):
            return
        if ch == "\n":
            if self.tape:  # nothing emitted yet: a leading newline is ignored
                self.col = 0
        elif ch == " ":
            pass
        elif ch in "-<>":
            self.pending = ch
        elif ch in PUNCT:
            self.tape.append(("P", ch))
        else:
            self.invalid = True

    def feed(self, text):
        if text is None:
            self.invalid = True
            return
        for ch in text:
            self.char(ch)

    def finish(self):
        self._flush_word()
        self._flush_pending()
        if self.col is not None:
            self.tape.append(("NL", 0))
            self.col = None
        if self.invalid:
            raise CaptureError("a token outside BendLite")
        return self.tape


def capture(ids, rom) -> list:
    c = Capture()
    for i in ids:
        c.feed(rom.get(i))
    return c.finish()


# --- tape walker: the firmware's logic, from the tape alone ----------------------------------

K = {w: name_hash(w) for w in ["import", "Base", "def", "main", "match", "case", "U32", "Nat",
                               "Bool", "True", "False", "U32.from_nat", "U32.is_eq", "U32.min",
                               "U32.max"]}


class WalkError(Exception):
    pass


MAX_DEPTH = 256  # nested def calls; the engine's return stack is sized for this (bytecode.py)


class Walker:
    def __init__(self, tape, max_calls=10_000_000):
        self.t, self.calls, self.max_calls = tape, 0, max_calls
        self.depth = 0
        self.defs = {}  # name -> (order, params [(name, type)], body position, line column)
        self.index()

    def at(self, p, kind, val=None):
        return p < len(self.t) and self.t[p][0] == kind and (val is None or self.t[p][1] == val)

    def need(self, p, kind, val=None):
        if not self.at(p, kind, val):
            raise WalkError(f"expected {kind} {val} at {p}, got {self.t[p] if p < len(self.t) else 'end'}")
        return p + 1

    def index(self):
        t = self.t
        p = 0
        p = self.need(p, "NAME", K["import"])
        p = self.need(p, "NAME", K["Base"])
        p = self.need(p, "NL", 0)
        order = 0
        while p < len(t):
            p = self.need(p, "NAME", K["def"])
            if not self.at(p, "NAME"):
                raise WalkError("def name")
            name = t[p][1]
            p = self.need(p + 1, "P", "(")
            params = []
            while not self.at(p, "P", ")"):
                if params:
                    p = self.need(p, "P", ",")
                if self.at(p, "P", "+"):
                    p += 1
                if not self.at(p, "NAME"):
                    raise WalkError("param name")
                pname = t[p][1]
                p = self.need(p + 1, "P", ":")
                if not any(self.at(p, "NAME", K[ty]) for ty in ("U32", "Nat", "Bool")):
                    raise WalkError("param type")
                params.append((pname, t[p][1]))
                p += 1
            p = self.need(p, "P", ")")
            p = self.need(p, "P", "->")
            if not any(self.at(p, "NAME", K[ty]) for ty in ("U32", "Nat", "Bool")):
                raise WalkError("return type")
            ret = t[p][1]
            p = self.need(p + 1, "P", ":")
            p = self.need(p, "NL", 2)
            if name in self.defs:
                raise WalkError("def twice")
            self.defs[name] = (order, params, p, 2, ret)
            order += 1
            p = self.skip_block(p, 2)
        main = self.defs.get(K["main"])
        if not main or main[1] or main[4] != K["U32"]:
            raise WalkError("needs def main() -> U32")

    def skip_block(self, p, col):
        """Skip a body at indentation col: up to the next NL with column < col, or the end."""
        while p < len(self.t):
            if self.t[p][0] == "NL" and self.t[p][1] < col:
                return p + 1 if self.t[p][1] == 0 else p
            p += 1
        return p

    def run_main(self):
        return self.call(K["main"], [], len(self.defs))

    def call(self, name, args, limit):
        if name in (K["U32.from_nat"], K["U32.is_eq"], K["U32.min"], K["U32.max"]):
            if len(args) != (1 if name == K["U32.from_nat"] else 2) or any(isinstance(a, bool) for a in args):
                raise WalkError("builtin arity or Bool argument")
            if name == K["U32.from_nat"]:
                return args[0] % M
            if name == K["U32.is_eq"]:
                return args[0] == args[1]
            return min(args) if name == K["U32.min"] else max(args)
        d = self.defs.get(name)
        if d is None or d[0] > limit:
            raise WalkError("unknown or later def")
        order, params, pos, col, _ = d
        if len(args) != len(params):
            raise WalkError("arity")
        self.calls += 1
        if self.calls > self.max_calls:
            raise WalkError("call limit")
        env = {pn: a for (pn, _), a in zip(params, args)}
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise WalkError("call depth")
        try:
            v, _ = self.body(pos, col, env, order)
        finally:
            self.depth -= 1
        return v

    def body(self, p, col, env, order):
        t = self.t
        if self.at(p, "NAME", K["match"]):
            if not self.at(p + 1, "NAME"):
                raise WalkError("match subject")
            var = t[p + 1][1]
            if var not in env:
                raise WalkError("match on unknown")
            v = env[var]
            p = self.need(p + 2, "P", ":")
            p = self.need(p, "NL", col + 2)
            for arm in range(2):
                p = self.need(p, "NAME", K["case"])
                if self.at(p, "NAT", 0):
                    pat, p = ("zero",), p + 1
                elif self.at(p, "NAT", 1) and self.at(p + 1, "P", "+") and self.at(p + 2, "NAME"):
                    pat, p = ("succ", t[p + 2][1]), p + 3
                elif self.at(p, "NAME", K["True"]) or self.at(p, "NAME", K["False"]):
                    b = t[p][1] == K["True"]
                    p = self.need(p + 1, "P", "{")
                    p = self.need(p, "P", "}")
                    pat = ("bool", b)
                else:
                    raise WalkError("case pattern")
                p = self.need(p, "P", ":")
                p = self.need(p, "NL", col + 4)
                hit = ((pat[0] == "zero" and not isinstance(v, bool) and v == 0)
                       or (pat[0] == "succ" and not isinstance(v, bool) and v > 0)
                       or (pat[0] == "bool" and isinstance(v, bool) and v == pat[1]))
                if hit:
                    sub = {**env, pat[1]: v - 1} if pat[0] == "succ" else env
                    r, _ = self.body(p, col + 4, sub, order)
                    return r, None
                p = self.skip_block(p, col + 4)
                if arm == 0:
                    if not self.at(p, "NAME", K["case"]):
                        # skip_block stops at the NL before the next case; step past it
                        p = self.need(p, "NL", col + 2)
            raise WalkError("no case matched")
        v, p = self.expr(p, env, order)
        if not (p >= len(t) or t[p][0] == "NL"):
            raise WalkError("trailing tokens")
        return v, p

    def expr(self, p, env, order):
        t = self.t
        if self.at(p, "NUM"):
            if t[p][1] >= M:
                raise WalkError("U32 literal too large")
            return t[p][1], p + 1
        if self.at(p, "NAT"):
            return t[p][1], p + 1
        if self.at(p, "NAME", K["True"]) or self.at(p, "NAME", K["False"]):
            b = t[p][1] == K["True"]
            p = self.need(p + 1, "P", "{")
            return b, self.need(p, "P", "}")
        if self.at(p, "P", "("):
            items, p = [], p + 1
            v, p = self.expr(p, env, order)
            items.append(v)
            cmp = None
            while p < len(t) and t[p][0] == "P" and t[p][1] in ("+", "-", "*", "/", "%", "<", "<=", ">", ">="):
                op = t[p][1]
                if op in ("<", "<=", ">", ">="):
                    if cmp is not None:
                        raise WalkError("two comparisons")
                    cmp = len(items)
                v, p = self.expr(p + 1, env, order)
                items += [op, v]
            p = self.need(p, "P", ":")
            p = self.need(p, "NAME", K["U32"])
            p = self.need(p, "P", ")")
            if len(items) == 1:
                raise WalkError("group without operator")
            if cmp is None:
                return arith(items), p
            if any(isinstance(x, bool) for x in items[0::2]):
                raise WalkError("Bool in a comparison group")
            left, right = items[:cmp], items[cmp + 1:]
            a = left[0] if len(left) == 1 else arith(left)
            b = right[0] if len(right) == 1 else arith(right)
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[items[cmp]], p
        if self.at(p, "NAME"):
            name = t[p][1]
            if self.at(p + 1, "P", "("):
                p += 2
                args = []
                while not self.at(p, "P", ")"):
                    if args:
                        p = self.need(p, "P", ",")
                    v, p = self.expr(p, env, order)
                    args.append(v)
                return self.call(name, args, order), p + 1
            if name not in env:
                raise WalkError("unknown variable")
            return env[name], p + 1
        raise WalkError(f"unexpected {t[p] if p < len(t) else 'end'}")


def arith(items):
    terms, ops = [items[0]], []
    for op, v in zip(items[1::2], items[2::2]):
        if isinstance(v, bool) or isinstance(terms[-1], bool):
            raise WalkError("arithmetic on Bool")
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


def run_ids(ids, rom):
    w = Walker(capture(ids, rom))
    v = w.run_main()
    if isinstance(v, bool):
        raise WalkError("main returns Bool")
    return v, w.calls


if __name__ == "__main__":
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bendlite_ref import BendLiteError, run_source

    sys.setrecursionlimit(100000)
    tok = AutoTokenizer.from_pretrained(str(qwen_dir()))
    rom = build_rom(lambda i: tok.decode([i]), len(tok))
    print(f"ROM: {len(rom)} of {len(tok)} token ids are BendLite-only text", flush=True)
    srcs = []
    for split in ("train", "valid", "test"):
        for line in open(BENDTUNE / f"data/{split}.jsonl"):
            ex = json.loads(line)
            srcs.append((ex["messages"][2]["content"], ex["meta"]["expected"]))
    extra = [(json.loads(x)["program"], None) for f in sys.argv[1:] for x in open(f)]
    bad = n = 0
    for src, want in srcs + extra:
        ids = tok.encode(src, add_special_tokens=False)
        try:
            ref = run_source(src)[0]
        except (BendLiteError, RecursionError) as e:
            ref = "ERR"
        try:
            got = run_ids(ids, rom)[0]
        except (CaptureError, WalkError, RecursionError, KeyError, IndexError) as e:
            got = "ERR"
        n += 1
        if got != ref or (want is not None and got != want):
            bad += 1
            if bad <= 8:
                print("MISMATCH", repr(got), repr(ref), want, src[:120].replace("\n", " | "))
    print(f"tape walker vs reference: {n - bad}/{n} agree")
    sys.exit(1 if bad else 0)
