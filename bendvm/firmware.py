"""BendVM firmware (5b-3): the BendLite tape walker as BVM machine code.

It runs on bvm.VM over a tape the capture hardware wrote into memory, and is a one-to-one
port of capture.Walker (index -> callf -> eval_body / eval_expr). Every program error becomes
FAIL (the engine prints "..."). Equivalence with capture.Walker / bendlite_ref is checked by
test_firmware.py.

Memory (32-bit words):
  TAPE_K 512    tape entry i's kind at TAPE_K + i (kinds below; kind 0 = end of tape)
  TAPE_V 1024   tape entry i's value at TAPE_V + i  (at most 509 entries + end + 2 pads; the
                Phase 2 8,192-word map, DESIGN.md)
  DEFS   8192   8 words per def: key, nparams, param base, body pos, order, return-type key
  PARAMS 8704   one key per parameter (all defs)
  NDEFS  9800
  STACK 16384.. grows upward
Kinds: NAME 1 (value = name key), NUM 2, NAT 3, P 4 (value = char code; -> <= >= are
1001-1003), NL 5 (value = column), BIG 6 (a number >= 2^32: always FAIL).
Registers: r1 r2 results (value, tag: 0 number, 1 Bool), r3 tape position, r4-r8 work,
r9-r11 macro temporaries, r12 the current def's order (call limit), r13 frame pointer,
r14 stack pointer, r15 return address.
Call frame (callf): [saved r13, saved r15, saved r12, env head, header address]; env cells
[key, value, tag, next]; call header [key, nargs, arg0 value, arg0 tag, ...].
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bvm import Asm  # noqa: E402
from capture import K as WK  # noqa: E402

M = 2**32
TAPE, DEFS, PARAMS, NDEFS, STACK = 4096, 8192, 8704, 9800, 16384
TAPE_K, TAPE_V = 512, 1024  # kinds and values of the tape (up to 509 entries + end + 2 pads)
NAME, NUM, NAT, PUN, NL, BIG = 1, 2, 3, 4, 5, 6
PCODE = {"->": 1001, "<=": 1002, ">=": 1003}
SP, RA, FP, ORD = 14, 15, 13, 12
T1, T2, T3 = 9, 10, 11


def key_of(h_len):
    """A name's key: its XOR / nibble-rotate hash (capture.name_hash); the length is not used."""
    return h_len[0]


def pcode(s):
    return PCODE.get(s, ord(s) if len(s) == 1 else 0)


K = {w: key_of(v) for w, v in WK.items()}


def tape_words(tape):
    """capture tape -> (kinds, values) arrays, plus the end marker (kind 0)."""
    ks, vs = [], []
    for kind, val in tape:
        if kind == "NAME":
            k, v = NAME, key_of(val)
        elif kind in ("NUM", "NAT"):
            k, v = (BIG, 0) if val >= M else (NUM if kind == "NUM" else NAT, val)
        elif kind == "P":
            k, v = PUN, pcode(val)
        elif kind == "NL":
            k, v = NL, val
        else:
            raise ValueError(kind)
        ks.append(k)
        vs.append(v)
    return ks + [0], vs + [0]


def load_tape(vm, tape):
    """Write a capture tape into vm memory; raises ValueError if it does not fit."""
    ks, vs = tape_words(tape)
    ks, vs = ks + [0, 0], vs + [0, 0]  # 2 written pads: the compiler looks ahead up to 2 entries,
    # and the circuit needs every word it reads to have been written (DESIGN.md Phase 2)
    if len(ks) > TAPE_V - TAPE_K:
        raise ValueError(f"tape of {len(ks)} entries does not fit")
    vm.mem[TAPE_K:TAPE_K + len(ks)] = ks
    vm.mem[TAPE_V:TAPE_V + len(vs)] = vs


class Fw(Asm):
    def push(self, r):
        self.ST(SP, 0, r)
        self.ADDI(SP, SP, 1)

    def pop(self, r):
        self.ADDI(SP, SP, M - 1)
        self.LD(r, SP, 0)

    def is_at(self, dest, kind, val=None, rp=3, off=0):
        """dest = [tape[rp + off] has this kind (and value)]; uses T2."""
        end = self.fresh("isat")
        self.LD(T2, rp, TAPE_K + off)
        self.SEQI(dest, T2, kind)
        if val is not None:
            self.BEQZ(dest, end)
            self.LD(T2, rp, TAPE_V + off)
            self.SEQI(dest, T2, val)
        self.label(end)

    def expect(self, kind, val=None, store=None, val_reg=None):
        """Require tape[r3] to have kind (and value, or the value in val_reg), r3 += 1."""
        self.LD(T2, 3, TAPE_K)
        self.BNEI(T2, kind, "fail")
        if val is not None or val_reg is not None:
            self.LD(T2, 3, TAPE_V)
            if val is not None:
                self.BNEI(T2, val, "fail")
            else:
                self.BNE(T2, val_reg, "fail")
        if store is not None:
            self.LD(store, 3, TAPE_V)
        self.ADDI(3, 3, 1)

    def type_name(self):
        """Require tape[r3] to be U32, Nat or Bool; r3 += 1."""
        ok = self.fresh("tyok")
        for ty in ("U32", "Nat", "Bool"):
            self.is_at(8, NAME, K[ty])
            self.BNEZ(8, ok)
        self.JMP("fail")
        self.label(ok)
        self.ADDI(3, 3, 1)


def build():
    a = Fw()
    # ---------------------------------------------------------------- entry
    a.LI(SP, STACK)
    a.LI(FP, 0)
    a.LI(3, 0)
    a.expect(NAME, K["import"])
    a.expect(NAME, K["Base"])
    a.expect(NL, 0)
    a.LI(4, 0)  # number of defs
    a.LI(5, 0)  # number of parameters so far (all defs)
    # ---------------------------------------------------------------- index: the def table
    a.label("idx_loop")
    a.is_at(8, 0)
    a.BNEZ(8, "idx_done")
    a.expect(NAME, K["def"])
    a.expect(NAME, store=6)  # the def's key
    a.expect(PUN, pcode("("))
    a.LI(7, 0)  # parameters of this def
    a.label("plist")
    a.is_at(8, PUN, pcode(")"))
    a.BNEZ(8, "pend")
    a.BEQZ(7, "nocomma")
    a.expect(PUN, pcode(","))
    a.label("nocomma")
    a.is_at(8, PUN, pcode("+"))
    a.BEQZ(8, "noplus")
    a.ADDI(3, 3, 1)
    a.label("noplus")
    a.expect(NAME, store=8)
    a.ST(5, PARAMS, 8)
    a.ADDI(5, 5, 1)
    a.expect(PUN, pcode(":"))
    a.type_name()
    a.ADDI(7, 7, 1)
    a.JMP("plist")
    a.label("pend")
    a.expect(PUN, pcode(")"))
    a.expect(PUN, pcode("->"))
    a.LD(ORD, 3, TAPE_V)  # the return-type key (checked by type_name next)
    a.type_name()
    a.expect(PUN, pcode(":"))
    a.expect(NL, 2)
    # a def defined twice is an error
    a.LI(1, 0)
    a.label("dup_loop")
    a.SEQ(2, 1, 4)
    a.BNEZ(2, "dup_done")
    a.LI(2, 8)
    a.MUL(2, 1, 2)
    a.LD(2, 2, DEFS)
    a.SEQ(2, 2, 6)
    a.BNEZ(2, "fail")
    a.ADDI(1, 1, 1)
    a.JMP("dup_loop")
    a.label("dup_done")
    a.LI(2, 8)
    a.MUL(1, 4, 2)
    a.ST(1, DEFS, 6)
    a.ST(1, DEFS + 1, 7)
    a.SUB(2, 5, 7)
    a.ST(1, DEFS + 2, 2)
    a.ST(1, DEFS + 3, 3)
    a.ST(1, DEFS + 4, 4)
    a.ST(1, DEFS + 5, ORD)
    a.ADDI(4, 4, 1)
    a.LI(1, 2)
    a.JAL(RA, "skip")
    a.JMP("idx_loop")
    a.label("idx_done")
    a.ST(0, NDEFS, 4)
    # main must exist, take no parameters and return U32
    a.LI(1, 0)
    a.label("main_find")
    a.SEQ(2, 1, 4)
    a.BNEZ(2, "fail")
    a.LI(2, 8)
    a.MUL(7, 1, 2)
    a.LD(2, 7, DEFS)
    a.LI(T3, K["main"])
    a.SEQ(2, 2, T3)
    a.BNEZ(2, "main_found")
    a.ADDI(1, 1, 1)
    a.JMP("main_find")
    a.label("main_found")
    a.LD(2, 7, DEFS + 1)
    a.BNEZ(2, "fail")
    a.LD(2, 7, DEFS + 5)
    a.LI(T3, K["U32"])
    a.SEQ(2, 2, T3)
    a.BEQZ(2, "fail")
    # ---------------------------------------------------------------- run main
    a.MOV(6, SP)
    a.LI(1, K["main"])
    a.push(1)
    a.push(0)
    a.MOV(ORD, 4)  # limit: every def
    a.JAL(RA, "callf")
    a.BNEZ(2, "fail")  # main must return a number
    a.OUT(1)
    a.HALT()
    a.label("fail")
    a.FAIL()

    # ---------------------------------------------------------------- skip(r1 = col) -> r3
    # Skip a body: stop at an NL with column < col (stepping past it if its column is 0), or
    # at the end of the tape. Leaf; uses r1 r2 T1-T3.
    a.label("skip")
    a.label("sk_loop")
    a.LD(2, 3, TAPE_K)
    a.BEQZ(2, "sk_ret")
    a.LI(T3, NL)
    a.SEQ(T2, 2, T3)
    a.BEQZ(T2, "sk_next")
    a.LD(2, 3, TAPE_V)
    a.SLTU(T2, 2, 1)
    a.BEQZ(T2, "sk_next")
    a.BNEZ(2, "sk_ret")
    a.ADDI(3, 3, 1)
    a.JR(RA)
    a.label("sk_next")
    a.ADDI(3, 3, 1)
    a.JMP("sk_loop")
    a.label("sk_ret")
    a.JR(RA)

    # ---------------------------------------------------------------- lookup(r5 = key) -> r1 r2
    a.label("lookup")
    a.LD(T2, FP, 3)
    a.label("lk_loop")
    a.BEQZ(T2, "fail")
    a.LD(T3, T2, 0)
    a.SEQ(T3, T3, 5)
    a.BNEZ(T3, "lk_hit")
    a.LD(T2, T2, 3)
    a.JMP("lk_loop")
    a.label("lk_hit")
    a.LD(1, T2, 1)
    a.LD(2, T2, 2)
    a.JR(RA)

    # ---------------------------------------------------------------- callf(r6 = header)
    # header [key, nargs, args (value, tag)...] at the top of the stack; returns r1 r2 and
    # pops the header. r12 = the caller's limit (defs it may call); restores r12 r13 r15.
    a.label("callf")
    a.LD(4, 6, 0)  # key
    a.LD(5, 6, 1)  # nargs
    for name, n in (("U32.from_nat", 1), ("U32.is_eq", 2), ("U32.min", 2), ("U32.max", 2)):
        nxt = a.fresh("bi")
        a.BNEI(4, K[name], nxt)
        a.BNEI(5, n, "fail")
        a.LD(1, 6, 2)
        a.LD(2, 6, 3)
        a.BNEZ(2, "fail")  # numbers only
        if n == 2:
            a.LD(7, 6, 4)
            a.LD(2, 6, 5)
            a.BNEZ(2, "fail")
        if name == "U32.from_nat":
            a.LI(2, 0)
        elif name == "U32.is_eq":
            a.SEQ(1, 1, 7)
            a.LI(2, 1)
        else:
            pick = a.fresh("mm")
            a.SLEU(8, 1, 7)  # a <= b
            if name == "U32.min":
                a.BNEZ(8, pick)
            else:
                a.BEQZ(8, pick)
            a.MOV(1, 7)
            a.label(pick)
            a.LI(2, 0)
        a.MOV(SP, 6)
        a.JR(RA)
        a.label(nxt)
    # a user def: find it (order <= limit), check arity
    a.LD(8, 0, NDEFS)
    a.LI(7, 0)
    a.label("cf_find")
    a.SEQ(T3, 7, 8)
    a.BNEZ(T3, "fail")
    a.LI(T3, 8)
    a.MUL(T2, 7, T3)
    a.LD(T3, T2, DEFS)
    a.SEQ(T3, T3, 4)
    a.BNEZ(T3, "cf_found")
    a.ADDI(7, 7, 1)
    a.JMP("cf_find")
    a.label("cf_found")
    a.LD(T3, T2, DEFS + 4)
    a.SLEU(T3, T3, ORD)
    a.BEQZ(T3, "fail")
    a.LD(T3, T2, DEFS + 1)
    a.SEQ(T3, T3, 5)
    a.BEQZ(T3, "fail")
    # the frame
    a.MOV(8, SP)  # new frame pointer
    a.push(FP)
    a.push(RA)
    a.push(ORD)
    a.push(0)  # env head
    a.push(6)  # header
    a.MOV(FP, 8)
    # env cells for the parameters: [key, value, tag, next]
    a.LD(4, T2, DEFS + 2)  # param base
    a.LD(ORD, T2, DEFS + 4)  # the callee's order
    a.LD(3, T2, DEFS + 3)  # body position
    a.LI(7, 0)  # i
    a.label("cf_bind")
    a.SEQ(T3, 7, 5)
    a.BNEZ(T3, "cf_bound")
    a.ADD(T3, 4, 7)
    a.LD(T3, T3, PARAMS)
    a.MOV(8, SP)
    a.push(T3)  # key
    a.ADD(T3, 7, 7)
    a.ADD(T3, T3, 6)
    a.LD(T2, T3, 2)
    a.push(T2)  # value
    a.LD(T2, T3, 3)
    a.push(T2)  # tag
    a.LD(T2, FP, 3)
    a.push(T2)  # next
    a.ST(FP, 3, 8)
    a.ADDI(7, 7, 1)
    a.JMP("cf_bind")
    a.label("cf_bound")
    a.LI(8, 2)  # column of a def body
    a.JAL(RA, "eval_body")
    a.LD(6, FP, 4)
    a.LD(ORD, FP, 2)
    a.LD(RA, FP, 1)
    a.LD(FP, FP, 0)
    a.MOV(SP, 6)
    a.JR(RA)

    # ---------------------------------------------------------------- eval_body(r3, r8 = col)
    a.label("eval_body")
    a.push(RA)
    a.is_at(1, NAME, K["match"])
    a.BEQZ(1, "eb_expr")
    a.is_at(1, NAME, off=1)
    a.BEQZ(1, "fail")
    a.LD(5, 3, TAPE_V + 1)
    a.JAL(RA, "lookup")  # the subject -> r4 value, r5 tag
    a.MOV(4, 1)
    a.MOV(5, 2)
    a.ADDI(3, 3, 2)
    a.expect(PUN, pcode(":"))
    a.ADDI(1, 8, 2)
    a.expect(NL, val_reg=1)
    for arm in range(2):
        miss, nopat = a.fresh("miss"), a.fresh("pat")
        p_succ, p_bool, p_true, p_false, p_done = (a.fresh(x) for x in ("ps", "pb", "pt", "pf", "pd"))
        a.expect(NAME, K["case"])
        a.is_at(1, NAT, 0)
        a.BEQZ(1, p_succ)
        a.LI(6, 0)  # pattern: zero
        a.ADDI(3, 3, 1)
        a.JMP(p_done)
        a.label(p_succ)
        a.is_at(1, NAT, 1)
        a.BEQZ(1, p_bool)
        a.is_at(1, PUN, pcode("+"), off=1)
        a.BEQZ(1, p_bool)
        a.is_at(1, NAME, off=2)
        a.BEQZ(1, p_bool)
        a.LI(6, 1)  # pattern: succ, binder key in r7
        a.LD(7, 3, TAPE_V + 2)
        a.ADDI(3, 3, 3)
        a.JMP(p_done)
        a.label(p_bool)
        a.is_at(1, NAME, K["True"])
        a.BNEZ(1, p_true)
        a.is_at(1, NAME, K["False"])
        a.BEQZ(1, "fail")
        a.LI(7, 0)
        a.JMP(p_false)
        a.label(p_true)
        a.LI(7, 1)
        a.label(p_false)
        a.LI(6, 2)  # pattern: bool b = r7
        a.ADDI(3, 3, 1)
        a.expect(PUN, pcode("{"))
        a.expect(PUN, pcode("}"))
        a.label(p_done)
        a.expect(PUN, pcode(":"))
        a.ADDI(1, 8, 4)
        a.expect(NL, val_reg=1)
        # hit? zero: tag 0 and v == 0; succ: tag 0 and v > 0; bool: tag 1 and v == b
        hz, hs, hb, hit = a.fresh("hz"), a.fresh("hs"), a.fresh("hb"), a.fresh("hit")
        a.BEQI(6, 1, hs)
        a.BEQI(6, 2, hb)
        a.label(hz)
        a.BNEZ(5, miss)
        a.BNEZ(4, miss)
        a.JMP(hit)
        a.label(hs)
        a.BNEZ(5, miss)
        a.BEQZ(4, miss)
        a.JMP(hit)
        a.label(hb)
        a.BEQZ(5, miss)
        a.SEQ(T3, 4, 7)
        a.BEQZ(T3, miss)
        a.label(hit)
        bind_done = a.fresh("bd")
        a.BNEI(6, 1, bind_done)
        # succ: bind the binder to v - 1 in a new env cell
        a.MOV(1, SP)
        a.push(7)
        a.ADDI(2, 4, M - 1)
        a.push(2)
        a.push(0)
        a.LD(2, FP, 3)
        a.push(2)
        a.ST(FP, 3, 1)
        a.ADDI(8, 8, 4)
        a.JAL(RA, "eval_body")
        a.ADDI(SP, SP, M - 4)
        a.LD(4, SP, 3)  # the cell's next: the env head before the binding
        a.ST(FP, 3, 4)
        a.pop(RA)
        a.JR(RA)
        a.label(bind_done)
        a.ADDI(8, 8, 4)
        a.JAL(RA, "eval_body")
        a.pop(RA)
        a.JR(RA)
        a.label(miss)
        a.ADDI(1, 8, 4)
        a.JAL(RA, "skip")
        if arm == 0:
            a.is_at(1, NAME, K["case"])
            a.BNEZ(1, nopat)
            a.ADDI(1, 8, 2)
            a.expect(NL, val_reg=1)
            a.label(nopat)
    a.JMP("fail")
    a.label("eb_expr")
    a.JAL(RA, "eval_expr")
    a.LD(T2, 3, TAPE_K)
    a.BEQZ(T2, "eb_ret")
    a.BNEI(T2, NL, "fail")
    a.label("eb_ret")
    a.pop(RA)
    a.JR(RA)

    # ---------------------------------------------------------------- eval_expr(r3) -> r1 r2 r3
    a.label("eval_expr")
    a.push(RA)
    a.LD(4, 3, TAPE_K)
    a.LD(5, 3, TAPE_V)
    for kind in (NUM, NAT):
        nxt = a.fresh("ek")
        a.BNEI(4, kind, nxt)
        a.MOV(1, 5)
        a.LI(2, 0)
        a.ADDI(3, 3, 1)
        a.pop(RA)
        a.JR(RA)
        a.label(nxt)
    a.BNEI(4, NAME, "ee_notname")
    for word, b in (("True", 1), ("False", 0)):
        nxt = a.fresh("eb")
        a.BNEI(5, K[word], nxt)
        a.ADDI(3, 3, 1)
        a.expect(PUN, pcode("{"))
        a.expect(PUN, pcode("}"))
        a.LI(1, b)
        a.LI(2, 1)
        a.pop(RA)
        a.JR(RA)
        a.label(nxt)
    a.is_at(1, PUN, pcode("("), off=1)
    a.BNEZ(1, "ee_call")
    a.JAL(RA, "lookup")  # a variable: r5 = its key
    a.ADDI(3, 3, 1)
    a.pop(RA)
    a.JR(RA)
    # a call: header [key, nargs], then each argument's (value, tag)
    a.label("ee_call")
    a.push(0)  # slot for the tape position after ")": callf moves r3 (fix 1)
    a.MOV(6, SP)
    a.push(5)
    a.push(0)
    a.ADDI(3, 3, 2)
    a.label("ec_args")
    a.is_at(1, PUN, pcode(")"))
    a.BNEZ(1, "ec_close")
    a.LD(1, 6, 1)
    a.BEQZ(1, "ec_first")
    a.expect(PUN, pcode(","))
    a.label("ec_first")
    a.push(6)
    a.JAL(RA, "eval_expr")
    a.pop(6)
    a.push(1)
    a.push(2)
    a.LD(1, 6, 1)
    a.ADDI(1, 1, 1)
    a.ST(6, 1, 1)
    a.JMP("ec_args")
    a.label("ec_close")
    a.ADDI(3, 3, 1)
    a.ST(6, M - 1, 3)  # the slot below the header
    a.JAL(RA, "callf")  # leaves SP = r6
    a.pop(3)
    a.pop(RA)
    a.JR(RA)
    # a typed group: ( e op e ... : U32 ) with * / % over + - and one comparison
    # group frame at r7: [sum, sop (0 +, 1 -), term, cmp op (0 none), left, nops]
    a.label("ee_notname")
    a.BNEI(4, PUN, "fail")
    a.BNEI(5, pcode("("), "fail")
    a.ADDI(3, 3, 1)
    a.MOV(7, SP)
    for _ in range(6):
        a.push(0)
    a.push(7)
    a.JAL(RA, "eval_expr")
    a.pop(7)
    a.BNEZ(2, "fail")
    a.ST(7, 2, 1)  # term
    a.label("eg_loop")
    a.LD(4, 3, TAPE_K)
    a.BNEI(4, PUN, "eg_end")
    a.LD(5, 3, TAPE_V)
    ops = ["+", "-", "*", "/", "%", "<", "<=", ">", ">="]
    a.LI(4, 0)
    for op in ops:
        a.LI(T3, pcode(op))
        a.SEQ(T3, 5, T3)
        a.ADD(4, 4, T3)
    a.BEQZ(4, "eg_end")
    a.ADDI(3, 3, 1)
    a.LD(4, 7, 5)
    a.ADDI(4, 4, 1)
    a.ST(7, 5, 4)  # nops += 1
    a.push(7)
    a.push(5)
    a.JAL(RA, "eval_expr")
    a.pop(5)
    a.pop(7)
    a.BNEZ(2, "fail")  # numbers only in a group
    # r1 = the operand; r5 = the operator
    for op, lab in (("*", "eg_mul"), ("/", "eg_div"), ("%", "eg_mod"), ("+", "eg_add"),
                    ("-", "eg_sub")):
        a.BEQI(5, pcode(op), lab)
    a.JMP("eg_cmp")
    a.label("eg_mul")
    a.LD(4, 7, 2)
    a.MUL(4, 4, 1)
    a.ST(7, 2, 4)
    a.JMP("eg_loop")
    # Bend2 (measured): x / 0 = 0 and x % 0 = x
    a.label("eg_div")
    a.LD(4, 7, 2)
    a.BEQZ(1, "eg_div0")
    a.DIVU(4, 4, 1)
    a.ST(7, 2, 4)
    a.JMP("eg_loop")
    a.label("eg_div0")
    a.ST(7, 2, 0)
    a.JMP("eg_loop")
    a.label("eg_mod")
    a.BEQZ(1, "eg_loop")  # x % 0 = x: the term stays
    a.LD(4, 7, 2)
    a.REMU(4, 4, 1)
    a.ST(7, 2, 4)
    a.JMP("eg_loop")

    def fold(dest):
        """dest = sum sop term (frame at r7); uses T2 T3."""
        neg = a.fresh("neg")
        done = a.fresh("fd")
        a.LD(dest, 7, 0)
        a.LD(T2, 7, 2)
        a.LD(T3, 7, 1)
        a.BNEZ(T3, neg)
        a.ADD(dest, dest, T2)
        a.JMP(done)
        a.label(neg)
        a.SUB(dest, dest, T2)
        a.label(done)

    for lab, sop in (("eg_add", 0), ("eg_sub", 1)):
        a.label(lab)
        fold(4)
        a.ST(7, 0, 4)  # sum = sum sop term
        a.LI(4, sop)
        a.ST(7, 1, 4)
        a.ST(7, 2, 1)  # term = operand
        a.JMP("eg_loop")
    a.label("eg_cmp")
    a.LD(4, 7, 3)
    a.BNEZ(4, "fail")  # one comparison per group
    a.ST(7, 3, 5)
    fold(4)
    a.ST(7, 4, 4)  # left
    a.ST(7, 0, 0)
    a.ST(7, 1, 0)
    a.ST(7, 2, 1)
    a.JMP("eg_loop")
    a.label("eg_end")
    a.expect(PUN, pcode(":"))
    a.expect(NAME, K["U32"])
    a.expect(PUN, pcode(")"))
    a.LD(4, 7, 5)
    a.BEQZ(4, "fail")  # a group needs an operator
    fold(1)  # right (or the value)
    a.LD(4, 7, 3)
    a.LI(2, 0)
    a.BEQZ(4, "eg_ret")
    a.LD(6, 7, 4)  # left
    for op, form in (("<", ("SLTU", 6, 1)), ("<=", ("SLEU", 6, 1)), (">", ("SLTU", 1, 6)),
                     (">=", ("SLEU", 1, 6))):
        nxt = a.fresh("cmp")
        a.BNEI(4, pcode(op), nxt)
        a.emit(form[0], 1, form[1], form[2])
        a.LI(2, 1)
        a.JMP("eg_ret")
        a.label(nxt)
    a.JMP("fail")
    a.label("eg_ret")
    a.MOV(SP, 7)
    a.pop(RA)
    a.JR(RA)
    LABELS.update(a.labels)
    return a.link()


LABELS = {}  # label -> address (for debugging traces)
FIRMWARE = build()
