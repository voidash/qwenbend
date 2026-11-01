"""The engine's compile pass and evaluator, in Python (S1b + S2): the oracle for firmware2.py.

compile_tape(ks, vs) reads the capture tape once, applies Bend2's static checks (the same rules
as bendlite_ref.check: types, match position and coverage, affine use, structural termination;
see bendlite_ref.py) and emits direct-threaded code; any error rejects the program (None). run()
executes the code on a word memory laid out exactly as the BVM's, making the same memory
accesses as firmware2's handlers, so results and faults match. Checked programs cannot fail at
run time; only the engine's limits can (call depth, memory, the op limit).

Code words (at CODE; jump targets absolute; one word per value; s, b slot indexes):
  LIT v          push v (U32, Nat and Bool 0/1 alike)
  VAR s          push frame slot s
  ADD SUB MUL DIV MOD LT LE GT GE ISEQ MIN MAX    pop b, a; push a op b  (x/0 = 0, x%0 = x)
  CALL entry n   the top n values become the callee's frame; return stack <- (next ip, fp)
  ENTER k        the frame has k slots (params, then match binders): sp = fp + k
  AZERO s L      if slot s != 0 jump L                    (first arm 0n)
  ASUCC s b L    if slot s == 0 jump L, else slot b = s-1  (first arm 1n+b)
  ABOOL s x L    if slot s != x jump L                    (first arm True{} / False{})
  PRED s b       slot b = slot s - 1                      (second arm 1n+b)
  RET            the result replaces the frame
  OUT            output main's result and halt
U32.from_nat, (e) and (e : T) emit nothing (a Nat below 2^32 is its U32). Program: CALL main 0, OUT, then
each def's code in order (ENTER k, body).

Memory (BVM words): RSTACK 5632..6143, 2 words (return ip, saved fp) per call: 256 nested
calls; STACK 6144..8191 frames and operands; an access at or beyond MEM_TOP = 8192 faults.

Run: python3 bendvm/bytecode.py  (self-check; the full check is test_bytecode.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from firmware import K, NAME, NAT, NL, NUM, PUN, pcode  # noqa: E402

M = 2**32
OPS = ["LIT", "VAR", "ADD", "SUB", "MUL", "DIV", "MOD", "LT", "LE", "GT", "GE", "ISEQ", "MIN",
       "MAX", "CALL", "ENTER", "AZERO", "ASUCC", "ABOOL", "PRED", "RET", "OUT"]
ARGS = {"LIT": 1, "VAR": 1, "CALL": 2, "ENTER": 1, "AZERO": 2, "ASUCC": 3, "ABOOL": 3, "PRED": 2}
BINOP = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "%": "MOD",
         "<": "LT", "<=": "LE", ">": "GT", ">=": "GE"}
U32, NATT, BOOL = 1, 2, 3  # type codes
TYPE_OF = {K["U32"]: U32, K["Nat"]: NATT, K["Bool"]: BOOL}
BUILTIN = {K["U32.from_nat"]: (None, [NATT], U32), K["U32.is_eq"]: ("ISEQ", [U32, U32], BOOL),
           K["U32.min"]: ("MIN", [U32, U32], U32), K["U32.max"]: ("MAX", [U32, U32], U32)}
CODE = 4608  # BVM address of code word 0 (the 8,192-word map, firmware2.py)
RSTACK, STACK, MEM_TOP = 5632, 6144, 8192
# engine table sizes (firmware2's memory map); exceeding one rejects the program
MAX_DEFS, MAX_PARAMS, MAX_VARS, MAX_CODE, MAX_TAPE = 64, 256, 128, 512, 509
MAX_DEPTH = 256  # nested def calls, as capture.MAX_DEPTH: (STACK - RSTACK) / 2
assert (STACK - RSTACK) // 2 == MAX_DEPTH
P = {s: pcode(s) for s in ["(", ")", ",", ":", "+", "->", "{", "}", *BINOP]}
ARITH_OPS = {P[s] for s in ("+", "-", "*", "/", "%")}
CMP_OPS = {P[s] for s in ("<", "<=", ">", ">=")}
MUL_OPS = {P[s] for s in ("*", "/", "%")}
OP_OF = {P[s]: v for s, v in BINOP.items()}
# pattern of a variable while checking inside an arm of a match on it
NOPAT, PZERO, PSUCC, PTRUE, PFALSE = 0, 1, 2, 3, 4
# termination: node kinds (a variable's expansion, or a call argument's shape)
NVAR, NZERO, NSUCC, NTRUE, NFALSE, NLIT, NOTHER = 0, 1, 2, 3, 4, 5, 6
EQ, LT, GT = 0, 1, 2


class Reject(Exception):
    """The program is rejected before it runs."""


class Var:
    def __init__(self, key, ty, many):
        self.key, self.ty, self.many = key, ty, many
        self.pat, self.child, self.uses = NOPAT, -1, 0


class Compiler:
    def __init__(self, ks, vs, handlers=None):
        self.ks, self.vs = list(ks) + [0] * 4, list(vs) + [0] * 4  # reads past the end see kind 0
        self.h = handlers or {op: i for i, op in enumerate(OPS)}
        self.code = []
        self.defs = []  # [key, param keys, param types, param many, ret, entry]
        self.nparams = 0

    # --- tape access ---
    def at(self, p, kind, val=None):
        return self.ks[p] == kind and (val is None or self.vs[p] == val)

    def need(self, p, kind, val=None):
        if not self.at(p, kind, val):
            raise Reject(f"expected {kind} {val} at {p}")
        return p + 1

    def emit(self, op, *args):
        assert len(args) == ARGS.get(op, 0), op
        self.code += [self.h[op], *args]

    def addr(self):
        return CODE + len(self.code)

    def type_at(self, p):
        if not (self.ks[p] == NAME and self.vs[p] in TYPE_OF):
            raise Reject("a type")
        return TYPE_OF[self.vs[p]]

    # --- program ---
    def compile(self):
        p = self.need(0, NAME, K["import"])
        p = self.need(p, NAME, K["Base"])
        p = self.need(p, NL, 0)
        self.emit("CALL", 0, 0)  # main's entry, patched at the end
        self.emit("OUT")
        while self.ks[p] != 0:
            p = self.define(p)
            if len(self.code) > MAX_CODE:
                raise Reject("code area full")
        main = [d for d in self.defs if d[0] == K["main"]]
        if not main or main[0][1] or main[0][4] != U32:
            raise Reject("needs def main() -> U32")
        self.code[1] = main[0][5]
        return self.code

    def define(self, p):
        p = self.need(p, NAME, K["def"])
        if not self.at(p, NAME):
            raise Reject("def name")
        key = self.vs[p]
        p = self.need(p + 1, PUN, P["("])
        keys, tys, many = [], [], []
        while not self.at(p, PUN, P[")"]):
            if keys:
                p = self.need(p, PUN, P[","])
            plus = self.at(p, PUN, P["+"])
            p += plus
            if not self.at(p, NAME):
                raise Reject("param name")
            keys.append(self.vs[p])
            p = self.need(p + 1, PUN, P[":"])
            tys.append(self.type_at(p))
            many.append(plus)
            p += 1
        self.nparams += len(keys)
        if self.nparams > MAX_PARAMS or len(keys) > MAX_VARS:
            raise Reject("parameter table full")
        p = self.need(p, PUN, P[")"])
        p = self.need(p, PUN, P["->"])
        ret = self.type_at(p)
        p = self.need(p + 1, PUN, P[":"])
        p = self.need(p, NL, 2)
        if any(d[0] == key for d in self.defs):
            raise Reject("def twice")
        if len(self.defs) == MAX_DEFS:
            raise Reject("def table full")
        self.cur = [key, keys, tys, many, ret, self.addr()]
        self.defs.append(self.cur)
        self.emit("ENTER", 0)
        enter_at = len(self.code) - 1
        self.vars = [Var(k, t, m) for k, t, m in zip(keys, tys, many)]
        self.scope = list(range(len(keys)))  # var ids; the newest binding of a key wins
        p = self.body(p, 2, -1, 0)
        for v in self.vars[:len(keys)]:
            self.affine(v)
        self.code[enter_at] = len(self.vars)
        if not (self.ks[p] == 0 or self.at(p, NL, 0)):
            raise Reject("a def ends at column 0")
        return p + (self.ks[p] != 0)

    def affine(self, v):
        if v.uses > 1 and not v.many:
            raise Reject("consumed more than once")

    def lookup(self, key):
        for i in reversed(self.scope):
            if self.vars[i].key == key:
                return i
        return -1

    def use(self, i):
        """A variable occurrence: inside an arm a matched variable is its pattern."""
        while self.vars[i].pat == PSUCC:
            i = self.vars[i].child
        v = self.vars[i]
        if v.pat == NOPAT:
            v.uses = min(v.uses + 1, 2)

    # --- bodies: the body at column col; pending binders = (front var or -1, first open param) ---
    def body(self, p, col, front, j):
        if not self.at(p, NAME, K["match"]):
            p, ty, _ = self.expr(p)
            if ty != self.cur[4]:
                raise Reject("body type")
            self.emit("RET")
        else:
            p = self.match(p, col, front, j)
        if not (self.ks[p] == 0 or (self.ks[p] == NL and self.vs[p] < col)):
            raise Reject("a body ends with a line at a smaller column")
        return p

    def match(self, p, col, front, j):
        if not self.at(p + 1, NAME):
            raise Reject("match subject")
        s = self.lookup(self.vs[p + 1])
        if s < 0:
            raise Reject("match on an unknown name")
        nparams = len(self.cur[1])
        if s == front:
            j2 = j
        elif j <= s < nparams:
            j2 = s + 1
        else:
            raise Reject("can't be matched in this position")
        v = self.vars[s]
        if v.ty == U32:
            raise Reject("a constructor of U32")
        p = self.need(p + 2, PUN, P[":"])
        p = self.need(p, NL, col + 2)
        snap = [w.uses for w in self.vars]
        seen, miss_at, after0 = [], None, None
        for arm in range(2):
            p = self.need(p, NAME, K["case"])
            if self.at(p, NAT, 0):
                pat, p, binder = PZERO, p + 1, None
            elif self.at(p, NAT, 1) and self.at(p + 1, PUN, P["+"]) and self.at(p + 2, NAME):
                pat, binder, p = PSUCC, self.vs[p + 2], p + 3
            elif self.at(p, NAME, K["True"]) or self.at(p, NAME, K["False"]):
                pat, binder = (PTRUE if self.vs[p] == K["True"] else PFALSE), None
                p = self.need(p + 1, PUN, P["{"])
                p = self.need(p, PUN, P["}"])
            else:
                raise Reject("case pattern")
            if (pat in (PZERO, PSUCC)) != (v.ty == NATT) or pat in seen:
                raise Reject("one case per constructor")
            seen.append(pat)
            p = self.need(p, PUN, P[":"])
            p = self.need(p, NL, col + 4)
            if arm == 1:
                self.code[miss_at] = self.addr()
                after0 = [w.uses for w in self.vars[:len(snap)]]
                for w, n in zip(self.vars, snap):
                    w.uses = n
            nscope = len(self.scope)
            front2 = -1
            if pat == PSUCC:
                if len(self.vars) == MAX_VARS:
                    raise Reject("variable table full")
                b = len(self.vars)
                self.vars.append(Var(binder, NATT, v.many))
                self.scope.append(b)
                front2 = b
                v.child = b
                if arm == 0:
                    self.emit("ASUCC", s, b, 0)
                else:
                    self.emit("PRED", s, b)
            elif arm == 0:
                if pat == PZERO:
                    self.emit("AZERO", s, 0)
                else:
                    self.emit("ABOOL", s, int(pat == PTRUE), 0)
            if arm == 0:
                miss_at = len(self.code) - 1
            v.pat = pat
            p = self.body(p, col + 4, front2, j2)
            v.pat = NOPAT
            if pat == PSUCC:
                self.affine(self.vars[front2])
            del self.scope[nscope:]
            if arm == 0:
                p = self.need(p, NL, col + 2)
        for w, n in zip(self.vars, after0):  # uses: the max over the two arms
            w.uses = max(w.uses, n)
        return p

    # --- expressions: -> (p, type, termination shape (kind, payload)) ---
    def expr(self, p):
        k, v = self.ks[p], self.vs[p]
        if k == NUM:
            self.emit("LIT", v)
            return p + 1, U32, (NOTHER, 0)
        if k == NAT:
            self.emit("LIT", v)
            return p + 1, NATT, (NLIT, v)
        if k == NAME and v in (K["True"], K["False"]):
            p = self.need(p + 1, PUN, P["{"])
            p = self.need(p, PUN, P["}"])
            self.emit("LIT", int(v == K["True"]))
            return p, BOOL, ((NTRUE if v == K["True"] else NFALSE), 0)
        if k == PUN and v == P["("]:
            p, ty, shape = self.expr(p + 1)
            if self.at(p, PUN, P[")"]):  # (e): a plain parenthesis
                return p + 1, ty, shape
            if self.at(p, PUN, P[":"]):  # (e : T): an annotation; e must have type T
                if self.type_at(p + 1) != ty:
                    raise Reject("annotation type")
                return self.need(p + 2, PUN, P[")"]), ty, shape
            pending, cmp, tys = None, None, [ty]
            while self.ks[p] == PUN and (self.vs[p] in ARITH_OPS or self.vs[p] in CMP_OPS):
                op = self.vs[p]
                if op in CMP_OPS:
                    if cmp is not None:
                        raise Reject("two comparisons")
                    if pending:
                        self.emit(pending)
                    pending, cmp = None, OP_OF[op]
                    p, ty, _ = self.expr(p + 1)
                elif op in MUL_OPS:
                    p, ty, _ = self.expr(p + 1)
                    self.emit(OP_OF[op])
                else:
                    if pending:
                        self.emit(pending)
                    pending = OP_OF[op]
                    p, ty, _ = self.expr(p + 1)
                tys.append(ty)
            p = self.need(p, PUN, P[":"])
            p = self.need(p, NAME, K["U32"])
            p = self.need(p, PUN, P[")"])
            if any(t != U32 for t in tys):
                raise Reject("a group of U32 operands")
            if pending:
                self.emit(pending)
            if cmp:
                self.emit(cmp)
            return p, (BOOL if cmp else U32), (NOTHER, 0)
        if k != NAME:
            raise Reject("unexpected entry")  # includes BIG literals and the end of the tape
        if not self.at(p + 1, PUN, P["("]):
            i = self.lookup(v)
            if i < 0:
                raise Reject("unknown variable")
            self.use(i)
            self.emit("VAR", i)
            return p + 1, self.vars[i].ty, (NVAR, i)
        # a call
        if v in BUILTIN:
            op, want, ret = BUILTIN[v]
            d, entry = None, None
        else:
            d = next((d for d in self.defs if d[0] == v), None)
            if d is None:
                raise Reject("unknown or later def")
            op, want, ret = "CALL", d[2], d[4]
        selfcall = d is self.cur
        p, nargs, ord_ = p + 2, 0, EQ
        while not self.at(p, PUN, P[")"]):
            if nargs:
                p = self.need(p, PUN, P[","])
            p, ty, shape = self.expr(p)
            if nargs >= len(want) or ty != want[nargs]:
                raise Reject("argument count or type")
            if selfcall and ord_ == EQ:
                ord_ = self.descend(shape, self.node(nargs))
            nargs += 1
        if nargs != len(want):
            raise Reject("argument count")
        if selfcall and ord_ != LT:
            raise Reject("a decreasing self-call")
        if op == "CALL":
            self.emit("CALL", d[5], nargs)
        elif op:
            self.emit(op)
        return p + 1, ret, (NOTHER, 0)

    # --- termination (Bend2's term_descend, see bendlite_ref.descend) ---
    # a node is (kind, payload); a SUCC node's payload is its field: ("v", var) or ("l", k)
    def node(self, i):
        """The expansion of variable i."""
        v = self.vars[i]
        if v.pat == NOPAT:
            return NVAR, i
        if v.pat == PSUCC:
            return NSUCC, ("v", v.child)
        return {PZERO: NZERO, PTRUE: NTRUE, PFALSE: NFALSE}[v.pat], 0

    def field(self, n):
        return self.node(n[1][1]) if n[1][0] == "v" else (NLIT, n[1][1])

    def descend(self, a, p):
        if a[0] == NVAR:
            a = self.node(a[1])
        if a[0] == NLIT and p[0] != NVAR:
            a = (NZERO, 0) if a[1] == 0 else (NSUCC, ("l", a[1] - 1))
        if p[0] == NVAR:
            return EQ if a[0] == NVAR and a[1] == p[1] else GT
        if a[0] == p[0]:
            return EQ if p[0] != NSUCC else self.descend(self.field(a), self.field(p))
        if p[0] == NSUCC and self.descend(a, self.field(p)) != GT:
            return LT
        return GT


def compile_tape(ks, vs, handlers=None):
    """-> code words, or None if the program is rejected (including a tape longer than the
    engine's tape area, MAX_TAPE entries)."""
    if len(ks) - 1 > MAX_TAPE:
        return None
    try:
        return Compiler(ks, vs, handlers).compile()
    except Reject:
        return None


class VMError(Exception):
    pass


def run(code, max_ops=50_000_000):
    """Execute compiled code (default handler numbering) on a word memory laid out as the BVM's,
    making the same memory accesses the firmware handlers make. -> (value, ops executed)."""
    mem = [0] * MEM_TOP
    mem[CODE:CODE + len(code)] = code

    def rd(a):
        if not 0 <= a < MEM_TOP:
            raise VMError(f"fault at {a}")
        return mem[a]

    def wr(a, v):
        if not 0 <= a < MEM_TOP:
            raise VMError(f"fault at {a}")
        mem[a] = v

    ip, fp, sp, rsp, n = CODE, STACK, STACK, RSTACK, 0
    while True:
        n += 1
        if n > max_ops:
            raise VMError("op limit")
        op = OPS[rd(ip)]
        if op == "LIT":
            wr(sp, rd(ip + 1))
            sp += 1
        elif op == "VAR":
            wr(sp, rd(fp + rd(ip + 1)))
            sp += 1
        elif op in ("ADD", "SUB", "MUL", "DIV", "MOD", "LT", "LE", "GT", "GE", "ISEQ", "MIN", "MAX"):
            b, a = rd(sp - 1), rd(sp - 2)
            r = {"ADD": (a + b) % M, "SUB": (a - b) % M, "MUL": a * b % M,
                 "DIV": a // b if b else 0, "MOD": a % b if b else a,
                 "LT": int(a < b), "LE": int(a <= b), "GT": int(a > b), "GE": int(a >= b),
                 "ISEQ": int(a == b), "MIN": min(a, b), "MAX": max(a, b)}[op]
            wr(sp - 2, r)
            sp -= 1
        elif op == "CALL":
            if rsp == STACK:  # the return stack is full: depth MAX_DEPTH + 1
                raise VMError("call depth")
            wr(rsp, ip + 3)
            wr(rsp + 1, fp)
            rsp += 2
            fp = sp - rd(ip + 2)
            ip = rd(ip + 1)
            continue
        elif op == "ENTER":
            sp = fp + rd(ip + 1)
        elif op in ("AZERO", "ASUCC", "ABOOL"):
            v = rd(fp + rd(ip + 1))
            if op == "AZERO":
                hit, miss = v == 0, ip + 2
            elif op == "ASUCC":
                hit, miss = v != 0, ip + 3
                if hit:
                    wr(fp + rd(ip + 2), v - 1)
            else:
                hit, miss = v == rd(ip + 2), ip + 3
            if not hit:
                ip = rd(miss)
                continue
        elif op == "PRED":
            wr(fp + rd(ip + 2), rd(fp + rd(ip + 1)) - 1)
        elif op == "RET":
            wr(fp, rd(sp - 1))
            sp = fp + 1
            rsp -= 2
            ip, fp = rd(rsp), rd(rsp + 1)
            continue
        elif op == "OUT":
            return rd(sp - 1), n
        else:
            raise VMError(f"bad op {op}")
        ip += 1 + ARGS.get(op, 0)


if __name__ == "__main__":
    from capture import Capture
    from firmware import tape_words

    def run_src(src):
        c = Capture()
        c.feed(src)
        code = compile_tape(*tape_words(c.finish()))
        return None if code is None else run(code)[0]

    src = "import Base\n\ndef f(+n: Nat, +a: U32) -> U32:\n  match n:\n    case 0n:\n      a\n" \
          "    case 1n+p:\n      f(p, (a * 2 + 1 - 3 / 2 : U32))\n\ndef main() -> U32:\n  f(5n, 1)\n"
    assert run_src(src) == 32, run_src(src)
    assert run_src(src.replace("f(5n, 1)", "(2 + 3 * 4 < 15 : U32)")) is None  # main is Bool
    assert run_src(src.replace("f(5n, 1)", "(20 - 3 * 4 % 5 + 1 : U32)")) == 19
    assert run_src(src.replace("f(p, (", "f(n, (")) is None  # not decreasing
    assert run_src(src.replace("+a: U32", "a: U32").replace("(a * 2 + 1", "(a * a + 1")) is None
    deep = ("import Base\n\ndef d(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n"
            "    case 1n+p:\n      (1 + d(p) : U32)\n\ndef main() -> U32:\n  d(%dn)\n")
    assert run_src(deep % (MAX_DEPTH - 2)) == MAX_DEPTH - 2  # main + d(n)..d(0): depth MAX_DEPTH, ok
    try:
        run_src(deep % (MAX_DEPTH - 1))  # depth MAX_DEPTH + 1
        raise AssertionError("depth limit")
    except VMError:
        pass
    print("bytecode self-check ok")
