"""BendVM firmware v2 (S1b + S2): the engine's program, as BVM machine code. One compile pass over
the capture tape applies Bend2's static checks (types, match position and coverage, affine use,
structural termination) and writes direct-threaded code at CODE; any error is FAIL (the program
is rejected). Then the evaluator runs the code. It reproduces bytecode.py exactly: the code
words equal bytecode.compile_tape(..., HANDLERS) and the result equals bytecode.run's
(test_bytecode.py --bvm). The tape is loaded with firmware.load_tape.

Memory (8,192 words = 64 DeltaNet heads of 128 rows, DESIGN.md Phase 2): TAPE_K 512 / TAPE_V
1024 the tape; DEFS 1536 (8 words per def: key, nparams, param base, return type, entry);
PKEY 2048 / PTY 2304 / PMANY 2560 the parameters of all defs; globals at G = 2816; VARS 3072
(8 words per variable of the current def: key, type, many, pattern, child, uses; the
variable's id is its frame slot); SCOPE 4096 (variable ids, newest last); CODE 4608; RSTACK
5632; STACK 6144..8191 (the compiler's stack, then the evaluator's).

Compile registers: r3 tape position, r11 CP (next code word), r12 match frame, r13 SC (next
scope entry), r14 stack, r15 return address, r10 (T2) helper temporary, r9 emit / address
temporary, r1 r2 r4-r8 work.
Evaluator registers: r3 ip, r13 fp, r14 sp, r12 return-stack pointer, r11 the constant 1, r15
dispatch, r1 r2 temporaries.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bvm import Asm  # noqa: E402
from bytecode import (BOOL, CODE, EQ, GT, LT, MAX_CODE, MAX_DEFS, MAX_PARAMS, MAX_TAPE, MAX_VARS, NATT,  # noqa: E402
                      NFALSE, NLIT, NOTHER, NSUCC, NVAR, NZERO, OPS, PSUCC, PTRUE, RSTACK, STACK, U32)
from firmware import K, NAME, NAT, NL, NUM, PUN, T2, TAPE_K, TAPE_V, pcode  # noqa: E402
from lexer_fw import lexer  # noqa: E402

NSUCCL = 7  # a SUCC node whose field is a literal (a stepped Nat literal)
DEFS, PKEY, PTY, PMANY = 1536, 2048, 2304, 2560
G = 2816
NDEFS, NPARAMS, CUR, NVARS = G, G + 1, G + 2, G + 3
VARS, SCOPE = 3072, 4096
VKEY, VTY, VMANY, VPAT, VCHILD, VUSES = range(6)
assert DEFS + 8 * MAX_DEFS <= PKEY and PMANY + MAX_PARAMS <= G and VARS + 8 * MAX_VARS <= SCOPE
# the code size is checked after each def; one def writes at most 2 words per tape entry
# (MAX_TAPE entries), so an overlong program spills at most into RSTACK (unused while compiling)
assert SCOPE + MAX_VARS <= CODE and CODE + MAX_CODE + 2 * MAX_TAPE <= STACK
assert TAPE_K + MAX_TAPE + 3 <= TAPE_V and TAPE_V + MAX_TAPE + 3 <= DEFS  # + end + 2 pads
IP, FP, SP, RSP, ONE, H = 3, 13, 14, 12, 11, 15
CP, MF, SC, RA = 11, 12, 13, 15
BIN = ["ADD", "SUB", "MUL", "DIV", "MOD", "LT", "LE", "GT", "GE", "ISEQ", "MIN", "MAX"]
PUNS = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV", "%": "MOD", "<": "LT", "<=": "LE",
        ">": "GT", ">=": "GE"}


class Fw(Asm):
    def push(self, r):
        self.ST(SP, 0, r)
        self.ADDI(SP, SP, 1)

    def pop(self, r):
        self.ADDI(SP, SP, -1)
        self.LD(r, SP, 0)

    def is_at(self, dest, kind, val=None, off=0):
        """dest = [tape[r3 + off] has this kind (and value)]; uses T2."""
        end = self.fresh("isat")
        self.LD(T2, 3, TAPE_K + off)
        self.SEQI(dest, T2, kind)
        if val is not None:
            self.BEQZ(dest, end)
            self.LD(T2, 3, TAPE_V + off)
            self.SEQI(dest, T2, val)
        self.label(end)

    def need(self, kind, val=None, store=None, val_reg=None):
        """Require tape[r3] to have kind (and value / the value in val_reg), else FAIL; r3 += 1."""
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

    def type_at(self, dest, off=0):
        """dest = the type code of tape[r3 + off] (U32 / Nat / Bool), else FAIL; uses T2."""
        done = self.fresh("ty")
        self.LD(T2, 3, TAPE_K + off)
        self.BNEI(T2, NAME, "fail")
        self.LD(T2, 3, TAPE_V + off)
        for key, code in (("U32", U32), ("Nat", NATT), ("Bool", BOOL)):
            nxt = self.fresh("ty")
            self.BNEI(T2, K[key], nxt)
            self.LI(dest, code)
            self.JMP(done)
            self.label(nxt)
        self.JMP("fail")
        self.label(done)

    def emit_regs(self, *regs):
        """Append code words (register values) at CP."""
        for i, r in enumerate(regs):
            self.ST(CP, i, r)
        self.ADDI(CP, CP, len(regs))

    def emit_op(self, op, *regs):
        self.LI(9, f"op_{op}")
        self.emit_regs(9, *regs)

    def var_addr(self, dest, idr):
        """dest = 8 * variable id (the offset of its VARS entry)."""
        self.ADD(dest, idr, idr)
        self.ADD(dest, dest, dest)
        self.ADD(dest, dest, dest)

    def node(self, dk, dv, idr):
        """(dk, dv) = the termination node of variable idr: its pattern kind (0 = a free variable)
        and payload (the variable itself if free, else its child); uses r9. dk must not be idr."""
        free, done = self.fresh("nd"), self.fresh("nd")
        self.var_addr(9, idr)
        self.LD(dk, 9, VARS + VPAT)
        self.BEQZ(dk, free)
        self.LD(dv, 9, VARS + VCHILD)
        self.JMP(done)
        self.label(free)
        self.MOV(dv, idr)
        self.label(done)


# ==================================================================== evaluator
DIV_SUB = 3  # divu's subtraction fast path (measured: PROGRESS.md)


def dispatch(a):
    a.LD(H, IP, 0)
    a.JR(H)


def eval_handlers(a):
    a.label("op_LIT")
    a.LD(1, IP, 1)
    a.ST(SP, 0, 1)
    a.ADDI(SP, SP, 1)
    a.ADDI(IP, IP, 2)
    dispatch(a)

    a.label("op_VAR")
    a.LD(1, IP, 1)
    a.ADD(1, 1, FP)
    a.LD(1, 1, 0)
    a.ST(SP, 0, 1)
    a.ADDI(SP, SP, 1)
    a.ADDI(IP, IP, 2)
    dispatch(a)

    for op in BIN:
        a.label(f"op_{op}")
        a.LD(1, SP, -2)
        a.LD(2, SP, -1)
        if op in ("ADD", "SUB", "ISEQ"):
            a.emit({"ISEQ": "SEQ"}.get(op, op), 1, 1, 2)
        elif op == "MUL":  # no MUL / DIVU / REMU in the circuit's ALU: routines (DESIGN.md C2)
            a.JAL(10, "mulu")
            a.MOV(1, 4)
        elif op == "DIV":  # x / 0 = 0
            z, done = a.fresh("div"), a.fresh("div")
            a.BEQZ(2, z)
            a.JAL(10, "divu")
            a.MOV(1, 4)
            a.JMP(done)
            a.label(z)
            a.LI(1, 0)
            a.label(done)
        elif op == "MOD":  # x % 0 = x
            z = a.fresh("mod")
            a.BEQZ(2, z)
            a.JAL(10, "divu")
            a.MOV(1, 5)
            a.label(z)
        elif op in ("LT", "GT"):
            a.emit("SLTU", 1, *((1, 2) if op == "LT" else (2, 1)))
        elif op in ("LE", "GE"):  # a <= b is not (b < a): no SLEU in the circuit's ALU
            a.emit("SLTU", 1, *((2, 1) if op == "LE" else (1, 2)))
            a.SEQI(1, 1, 0)
        else:  # MIN / MAX: keep a unless b is smaller (MIN) / larger (MAX)
            keep = a.fresh("mm")
            if op == "MIN":
                a.SLTU(4, 2, 1)
            else:
                a.SLTU(4, 1, 2)
            a.BEQZ(4, keep)
            a.MOV(1, 2)
            a.label(keep)
        a.ST(SP, -2, 1)
        a.ADDI(SP, SP, -1)
        a.ADDI(IP, IP, 1)
        dispatch(a)

    a.label("op_CALL")
    a.BEQI(RSP, STACK, "fail")  # 4096 calls deep already
    a.ADDI(1, IP, 3)
    a.ST(RSP, 0, 1)
    a.ST(RSP, 1, FP)
    a.ADDI(RSP, RSP, 2)
    a.LD(1, IP, 2)
    a.SUB(FP, SP, 1)
    a.LD(IP, IP, 1)
    dispatch(a)

    a.label("op_ENTER")
    a.LD(1, IP, 1)
    a.ADD(SP, FP, 1)
    a.ADDI(IP, IP, 2)
    dispatch(a)

    for op, nargs in (("AZERO", 2), ("ASUCC", 3), ("ABOOL", 3)):
        miss = f"miss_{op}"
        a.label(f"op_{op}")
        a.LD(1, IP, 1)
        a.ADD(1, 1, FP)
        a.LD(1, 1, 0)
        if op == "AZERO":
            a.BNEZ(1, miss)
        elif op == "ABOOL":
            a.LD(2, IP, 2)
            a.BNE(1, 2, miss)
        else:
            a.BEQZ(1, miss)
            a.LD(2, IP, 2)
            a.ADD(2, 2, FP)
            a.ADDI(1, 1, -1)
            a.ST(2, 0, 1)
        a.ADDI(IP, IP, 1 + nargs)
        dispatch(a)
        a.label(miss)
        a.LD(IP, IP, nargs)
        dispatch(a)

    a.label("op_PRED")
    a.LD(1, IP, 1)
    a.ADD(1, 1, FP)
    a.LD(1, 1, 0)
    a.ADDI(1, 1, -1)
    a.LD(2, IP, 2)
    a.ADD(2, 2, FP)
    a.ST(2, 0, 1)
    a.ADDI(IP, IP, 3)
    dispatch(a)

    a.label("op_RET")
    a.LD(1, SP, -1)
    a.ST(FP, 0, 1)
    a.ADDI(SP, FP, 1)
    a.ADDI(RSP, RSP, -2)
    a.LD(IP, RSP, 0)
    a.LD(FP, RSP, 1)
    dispatch(a)

    # mulu(r1 = x, r2 = y) -> r4 = x * y mod 2^32; shift-add over the smaller operand's bits;
    # registers only (r4-r8), returns through r10
    a.label("mulu")
    a.SLTU(5, 1, 2)
    a.BEQZ(5, "mu_go")  # make y the smaller operand
    a.MOV(5, 1)
    a.MOV(1, 2)
    a.MOV(2, 5)
    a.label("mu_go")
    a.LI(4, 0)
    a.label("mu_loop")
    a.BEQZ(2, "mu_done")
    a.SHR1(5, 2)
    a.ADD(6, 5, 5)
    a.BEQ(6, 2, "mu_even")
    a.ADD(4, 4, 1)
    a.label("mu_even")
    a.ADD(1, 1, 1)
    a.MOV(2, 5)
    a.JMP("mu_loop")
    a.label("mu_done")
    a.JR(10)

    # divu(r1 = x, r2 = y != 0) -> r4 = x / y, r5 = x % y. Most quotients are small: up to
    # DIV_SUB subtractions first, then restoring division of the rest (accumulating into r4): double
    # d = y while d <= rem (and d < 2^31), then walk back down with SHR1; registers only (r4-r9)
    a.label("divu")
    a.MOV(5, 1)
    a.MOV(6, 2)
    a.LI(4, 0)
    for _ in range(DIV_SUB):
        a.SLTU(8, 5, 6)
        a.BNEZ(8, "dv_done")
        a.SUB(5, 5, 6)
        a.ADDI(4, 4, 1)
    a.LI(7, 1)
    a.LI(9, 0x80000000)
    a.label("dv_up")
    a.SLTU(8, 5, 6)
    a.BNEZ(8, "dv_loop")  # d > rem: start walking down
    a.SLTU(8, 6, 9)
    a.BEQZ(8, "dv_loop")  # d >= 2^31: can't double
    a.ADD(6, 6, 6)
    a.ADD(7, 7, 7)
    a.JMP("dv_up")
    a.label("dv_loop")
    a.BEQZ(7, "dv_done")
    a.SLTU(8, 5, 6)
    a.BNEZ(8, "dv_skip")
    a.SUB(5, 5, 6)
    a.ADD(4, 4, 7)
    a.label("dv_skip")
    a.SHR1(6, 6)
    a.SHR1(7, 7)
    a.JMP("dv_loop")
    a.label("dv_done")
    a.JR(10)

    a.label("op_OUT")
    a.LD(1, SP, -1)
    a.OUT(1)
    a.HALT()


def start_eval(a):
    a.LI(IP, CODE)
    a.LI(FP, STACK)
    a.LI(SP, STACK)
    a.LI(RSP, RSTACK)
    a.LI(ONE, 1)
    dispatch(a)


# ==================================================================== compile pass
def program(a):
    """As bytecode.Compiler.compile."""
    for r in range(16):  # boot: every register row is written once (the circuit reads rows, and
        a.LI(r, 0)  # a never-written row reads as all zeros, not as the word 0; r0 included)
    a.JMP("lex_start")  # the lexer (lexer_fw.py) turns the stored tokens into the tape
    a.label("lex_done")
    a.LI(SP, STACK)
    a.LI(3, 0)
    a.need(NAME, K["import"])
    a.need(NAME, K["Base"])
    a.need(NL, 0)
    a.LI(CP, CODE)
    a.emit_op("CALL", 0, 0)
    a.emit_op("OUT")
    a.ST(0, NDEFS, 0)
    a.ST(0, NPARAMS, 0)
    a.label("prog_loop")
    a.LD(T2, 3, TAPE_K)
    a.BEQZ(T2, "prog_end")
    a.JAL(RA, "define")
    a.LI(1, CODE + MAX_CODE + 1)
    a.SLTU(1, CP, 1)
    a.BEQZ(1, "fail")  # more than MAX_CODE words
    a.JMP("prog_loop")
    a.label("prog_end")
    a.LD(2, 0, NDEFS)
    a.ADD(2, 2, 2)
    a.ADD(2, 2, 2)
    a.ADD(2, 2, 2)
    a.LI(1, 0)
    a.label("main_find")
    a.BEQ(1, 2, "fail")
    a.LD(T2, 1, DEFS)
    a.BEQI(T2, K["main"], "main_found")
    a.ADDI(1, 1, 8)
    a.JMP("main_find")
    a.label("main_found")
    a.LD(T2, 1, DEFS + 1)
    a.BNEZ(T2, "fail")
    a.LD(T2, 1, DEFS + 3)
    a.BNEI(T2, U32, "fail")
    a.LD(T2, 1, DEFS + 4)
    a.ST(0, CODE + 1, T2)


def define(a):
    """define(r3) -> r3; as bytecode.Compiler.define."""
    a.label("define")
    a.push(RA)
    a.need(NAME, K["def"])
    a.need(NAME, store=6)  # the def's key
    a.need(PUN, pcode("("))
    a.LI(7, 0)  # its parameters
    a.LD(8, 0, NPARAMS)  # their base in PKEY / PTY / PMANY
    a.label("plist")
    a.is_at(1, PUN, pcode(")"))
    a.BNEZ(1, "pend")
    a.BEQZ(7, "nocomma")
    a.need(PUN, pcode(","))
    a.label("nocomma")
    a.is_at(5, PUN, pcode("+"))
    a.ADD(3, 3, 5)
    a.need(NAME, store=1)
    a.need(PUN, pcode(":"))
    a.type_at(2)
    a.ADDI(3, 3, 1)
    a.ADD(4, 8, 7)
    a.BEQI(4, MAX_PARAMS, "fail")
    a.BEQI(7, MAX_VARS, "fail")
    a.ST(4, PKEY, 1)
    a.ST(4, PTY, 2)
    a.ST(4, PMANY, 5)
    a.ADDI(7, 7, 1)
    a.JMP("plist")
    a.label("pend")
    a.need(PUN, pcode(")"))
    a.need(PUN, pcode("->"))
    a.type_at(4)
    a.ADDI(3, 3, 1)
    a.need(PUN, pcode(":"))
    a.need(NL, 2)
    a.LD(2, 0, NDEFS)  # a def defined twice is an error
    a.ADD(2, 2, 2)
    a.ADD(2, 2, 2)
    a.ADD(2, 2, 2)
    a.LI(1, 0)
    a.label("dup_loop")
    a.BEQ(1, 2, "dup_done")
    a.LD(T2, 1, DEFS)
    a.BEQ(T2, 6, "fail")
    a.ADDI(1, 1, 8)
    a.JMP("dup_loop")
    a.label("dup_done")
    a.BEQI(2, 8 * MAX_DEFS, "fail")
    a.ST(2, DEFS, 6)
    a.ST(2, DEFS + 1, 7)
    a.ST(2, DEFS + 2, 8)
    a.ST(2, DEFS + 3, 4)
    a.ST(2, DEFS + 4, CP)
    a.ST(0, CUR, 2)
    a.LD(1, 0, NDEFS)
    a.ADDI(1, 1, 1)
    a.ST(0, NDEFS, 1)
    a.ADD(1, 8, 7)
    a.ST(0, NPARAMS, 1)
    a.emit_op("ENTER", 0)
    a.ADDI(1, CP, -1)
    a.push(1)  # where ENTER's operand goes
    # the parameters become variables 0..n-1, all in scope
    a.ST(0, NVARS, 7)
    a.LI(SC, SCOPE)
    a.LI(1, 0)
    a.label("pv_loop")
    a.BEQ(1, 7, "pv_done")
    a.ADD(2, 8, 1)
    a.var_addr(9, 1)
    a.LD(4, 2, PKEY)
    a.ST(9, VARS + VKEY, 4)
    a.LD(4, 2, PTY)
    a.ST(9, VARS + VTY, 4)
    a.LD(4, 2, PMANY)
    a.ST(9, VARS + VMANY, 4)
    a.ST(9, VARS + VPAT, 0)
    a.ST(9, VARS + VUSES, 0)
    a.ST(SC, 0, 1)
    a.ADDI(SC, SC, 1)
    a.ADDI(1, 1, 1)
    a.JMP("pv_loop")
    a.label("pv_done")
    a.LI(4, 2)  # body(p, col 2, front -1, j 0)
    a.LI(5, -1)
    a.LI(6, 0)
    a.JAL(RA, "body")
    # affine: parameters
    a.LD(2, 0, CUR)
    a.LD(7, 2, DEFS + 1)
    a.LI(1, 0)
    a.label("pa_loop")
    a.BEQ(1, 7, "pa_done")
    a.JAL(RA, "affine")
    a.ADDI(1, 1, 1)
    a.JMP("pa_loop")
    a.label("pa_done")
    a.pop(1)
    a.LD(2, 0, NVARS)
    a.ST(1, 0, 2)
    a.LD(T2, 3, TAPE_K)
    a.BEQZ(T2, "df_ret")
    a.need(NL, 0)
    a.label("df_ret")
    a.pop(RA)
    a.JR(RA)


def routines(a):
    # ------------------------------------------------ affine(r1 = variable id); leaf, uses r9 T2
    a.label("affine")
    a.var_addr(9, 1)
    a.LD(T2, 9, VARS + VUSES)
    a.BNEI(T2, 2, "af_ok")
    a.LD(T2, 9, VARS + VMANY)
    a.BEQZ(T2, "fail")
    a.label("af_ok")
    a.JR(RA)

    # ------------------------------------------------ lookup(r1 = key) -> r2 id or -1; leaf, uses r8 r9 T2
    a.label("lookup")
    a.MOV(8, SC)
    a.label("lk_loop")
    a.BEQI(8, SCOPE, "lk_none")
    a.ADDI(8, 8, -1)
    a.LD(2, 8, 0)
    a.var_addr(9, 2)
    a.LD(T2, 9, VARS + VKEY)
    a.BNE(T2, 1, "lk_loop")
    a.JR(RA)
    a.label("lk_none")
    a.LI(2, -1)
    a.JR(RA)

    # ------------------------------------------------ use(r4 = variable id); leaf, uses r4 r9 T2
    # inside an arm a matched variable is its pattern: follow 1n+p to p; 0n / True{} use nothing
    a.label("use")
    a.var_addr(9, 4)
    a.LD(T2, 9, VARS + VPAT)
    a.BNEI(T2, PSUCC, "us_here")
    a.LD(4, 9, VARS + VCHILD)
    a.JMP("use")
    a.label("us_here")
    a.BNEZ(T2, "us_ret")
    a.LD(T2, 9, VARS + VUSES)
    a.BEQI(T2, 2, "us_ret")
    a.ADDI(T2, T2, 1)
    a.ST(9, VARS + VUSES, T2)
    a.label("us_ret")
    a.JR(RA)

    # ------------------------------------------------ body(r3, r4 col, r5 front, r6 j) -> r3
    # frame (SP-relative): RA -4, col -3, front -2, j -1
    a.label("body")
    a.push(RA)
    a.push(4)
    a.push(5)
    a.push(6)
    a.is_at(1, NAME, K["match"])
    a.BNEZ(1, "b_match")
    a.JAL(RA, "expr")
    a.LD(4, 0, CUR)
    a.LD(4, 4, DEFS + 3)
    a.BNE(1, 4, "fail")  # the body's type is the def's return type
    a.emit_op("RET")
    a.JMP("b_end")
    a.label("b_match")
    a.JAL(RA, "match")
    a.label("b_end")  # then the end of the tape or a line at a smaller column
    a.LD(T2, 3, TAPE_K)
    a.BEQZ(T2, "b_ret")
    a.BNEI(T2, NL, "fail")
    a.LD(T2, 3, TAPE_V)
    a.LD(4, SP, -3)
    a.SLTU(T2, T2, 4)
    a.BEQZ(T2, "fail")
    a.label("b_ret")
    a.ADDI(SP, SP, -4)
    a.LD(RA, SP, 0)
    a.JR(RA)

    # ------------------------------------------------ match: called by body (its frame below SP)
    # match frame at MF (after the saved RA and MF): 0 subject id, 1 j2, 2 col, 3 nsnap, 4 arm,
    # 5 first pattern, 6 miss operand address, 7 scope top, 8 front2 (binder), 9 pattern,
    # 10 subject's VARS offset; then the uses snapshot (nsnap words) and arm 0's uses (nsnap).
    a.label("match")
    a.LD(4, SP, -3)  # col, front, j from the body frame
    a.LD(5, SP, -2)
    a.LD(6, SP, -1)
    a.push(RA)
    a.push(MF)
    a.MOV(MF, SP)
    a.ADDI(SP, SP, 11)
    a.ST(MF, 2, 4)
    a.push(5)  # front, j (above the frame while the subject is resolved)
    a.push(6)
    a.is_at(1, NAME, off=1)
    a.BEQZ(1, "fail")
    a.LD(1, 3, TAPE_V + 1)
    a.JAL(RA, "lookup")
    a.BEQI(2, -1, "fail")
    a.ST(MF, 0, 2)
    a.pop(6)
    a.pop(5)
    a.BNE(2, 5, "m_notfront")
    a.MOV(7, 6)  # matching the front binder: j2 = j
    a.JMP("m_pos_ok")
    a.label("m_notfront")
    a.SLTU(T2, 2, 6)
    a.BNEZ(T2, "fail")  # s < j: a closed parameter
    a.LD(T2, 0, CUR)
    a.LD(T2, T2, DEFS + 1)
    a.SLTU(T2, 2, T2)
    a.BEQZ(T2, "fail")  # a binder that is not the front
    a.ADDI(7, 2, 1)  # j2 = s + 1
    a.label("m_pos_ok")
    a.ST(MF, 1, 7)
    a.var_addr(9, 2)
    a.ST(MF, 10, 9)
    a.LD(T2, 9, VARS + VTY)
    a.BEQI(T2, U32, "fail")  # a U32 has no constructors
    a.ADDI(3, 3, 2)
    a.need(PUN, pcode(":"))
    a.LD(4, MF, 2)
    a.ADDI(4, 4, 2)
    a.need(NL, val_reg=4)
    # snapshot the uses of every variable so far
    a.LD(7, 0, NVARS)
    a.ST(MF, 3, 7)
    a.LI(1, 0)
    a.label("m_snap")
    a.BEQ(1, 7, "m_snapped")
    a.var_addr(9, 1)
    a.LD(T2, 9, VARS + VUSES)
    a.push(T2)
    a.ADDI(1, 1, 1)
    a.JMP("m_snap")
    a.label("m_snapped")
    a.ADD(SP, SP, 7)  # room for arm 0's uses
    a.ST(MF, 4, 0)
    a.label("m_arm")
    a.need(NAME, K["case"])
    a.LI(8, 0)  # binder key
    a.is_at(1, NAT, 0)
    a.BEQZ(1, "m_p1")
    a.ADDI(3, 3, 1)
    a.LI(7, 1)  # PZERO
    a.JMP("m_pat")
    a.label("m_p1")
    a.is_at(1, NAT, 1)
    a.BEQZ(1, "m_p2")
    a.is_at(1, PUN, pcode("+"), off=1)
    a.BEQZ(1, "m_p2")
    a.is_at(1, NAME, off=2)
    a.BEQZ(1, "m_p2")
    a.LD(8, 3, TAPE_V + 2)
    a.ADDI(3, 3, 3)
    a.LI(7, PSUCC)
    a.JMP("m_pat")
    a.label("m_p2")
    a.is_at(1, NAME, K["True"])
    a.BNEZ(1, "m_pb")
    a.is_at(1, NAME, K["False"])
    a.BEQZ(1, "fail")
    a.label("m_pb")
    a.LD(T2, 3, TAPE_V)
    a.SEQI(7, T2, K["True"])
    a.SUB(7, 0, 7)
    a.ADDI(7, 7, 4)  # True -> PTRUE 3, False -> PFALSE 4
    a.ADDI(3, 3, 1)
    a.need(PUN, pcode("{"))
    a.need(PUN, pcode("}"))
    a.label("m_pat")
    # the pattern fits the subject's type (Nat: 0n / 1n+p; Bool: True / False), once each
    a.LD(9, MF, 10)
    a.LD(T2, 9, VARS + VTY)
    a.LI(1, 3)
    a.SLTU(1, 7, 1)  # 1 if a Nat pattern (PZERO 1 / PSUCC 2)
    a.SEQI(T2, T2, NATT)
    a.BNE(1, T2, "fail")
    a.LD(1, MF, 4)
    a.BEQZ(1, "m_first")
    a.LD(1, MF, 5)
    a.BEQ(1, 7, "fail")
    a.label("m_first")
    a.ST(MF, 5, 7)
    a.ST(MF, 9, 7)
    a.need(PUN, pcode(":"))
    a.LD(4, MF, 2)
    a.ADDI(4, 4, 4)
    a.need(NL, val_reg=4)
    a.LD(1, MF, 4)
    a.BEQZ(1, "m_scope")
    # arm 1: arm 0's miss target is here; keep arm 0's uses, restore the snapshot
    a.LD(1, MF, 6)
    a.ST(1, 0, CP)
    a.LD(6, MF, 3)  # nsnap
    a.ADDI(5, MF, 11)  # snapshot
    a.ADD(4, 5, 6)  # arm 0's uses
    a.LI(1, 0)
    a.label("m_keep")
    a.BEQ(1, 6, "m_scope")
    a.var_addr(9, 1)
    a.LD(T2, 9, VARS + VUSES)
    a.ST(4, 0, T2)
    a.LD(T2, 5, 0)
    a.ST(9, VARS + VUSES, T2)
    a.ADDI(4, 4, 1)
    a.ADDI(5, 5, 1)
    a.ADDI(1, 1, 1)
    a.JMP("m_keep")
    a.label("m_scope")
    a.ST(MF, 7, SC)
    a.LI(T2, -1)
    a.ST(MF, 8, T2)
    a.LD(2, MF, 0)  # s
    a.LD(1, MF, 4)  # arm
    a.LD(7, MF, 9)  # pattern
    a.BNEI(7, PSUCC, "m_nosucc")
    a.LD(5, 0, NVARS)  # a binder: variable b = NVARS, in scope, the subject's child
    a.BEQI(5, MAX_VARS, "fail")
    a.ADDI(4, 5, 1)
    a.ST(0, NVARS, 4)
    a.var_addr(9, 5)
    a.ST(9, VARS + VKEY, 8)
    a.LI(T2, NATT)
    a.ST(9, VARS + VTY, T2)
    a.LD(4, MF, 10)
    a.LD(T2, 4, VARS + VMANY)
    a.ST(9, VARS + VMANY, T2)
    a.ST(9, VARS + VPAT, 0)
    a.ST(9, VARS + VUSES, 0)
    a.ST(SC, 0, 5)
    a.ADDI(SC, SC, 1)
    a.ST(MF, 8, 5)
    a.ST(4, VARS + VCHILD, 5)
    a.BNEZ(1, "m_pred")
    a.emit_op("ASUCC", 2, 5, 0)
    a.JMP("m_emitted")
    a.label("m_pred")
    a.emit_op("PRED", 2, 5)
    a.JMP("m_emitted")
    a.label("m_nosucc")
    a.BNEZ(1, "m_emitted")
    a.BNEI(7, 1, "m_bool")
    a.emit_op("AZERO", 2, 0)
    a.JMP("m_emitted")
    a.label("m_bool")
    a.SEQI(4, 7, PTRUE)
    a.emit_op("ABOOL", 2, 4, 0)
    a.label("m_emitted")
    a.LD(1, MF, 4)
    a.BNEZ(1, "m_body")
    a.ADDI(1, CP, -1)
    a.ST(MF, 6, 1)
    a.label("m_body")
    a.LD(9, MF, 10)  # the subject is its pattern inside the arm
    a.LD(7, MF, 9)
    a.ST(9, VARS + VPAT, 7)
    a.LD(4, MF, 2)
    a.ADDI(4, 4, 4)
    a.LD(5, MF, 8)
    a.LD(6, MF, 1)
    a.JAL(RA, "body")
    a.LD(9, MF, 10)
    a.ST(9, VARS + VPAT, 0)
    a.LD(1, MF, 8)
    a.BEQI(1, -1, "m_nobinder")
    a.JAL(RA, "affine")
    a.label("m_nobinder")
    a.LD(SC, MF, 7)
    a.LD(1, MF, 4)
    a.BNEZ(1, "m_done")
    a.LD(4, MF, 2)
    a.ADDI(4, 4, 2)
    a.need(NL, val_reg=4)
    a.LI(1, 1)
    a.ST(MF, 4, 1)
    a.JMP("m_arm")
    a.label("m_done")  # uses: the max over the two arms
    a.LD(6, MF, 3)
    a.ADDI(4, MF, 11)
    a.ADD(4, 4, 6)  # arm 0's uses
    a.LI(1, 0)
    a.label("m_max")
    a.BEQ(1, 6, "m_ret")
    a.var_addr(9, 1)
    a.LD(T2, 9, VARS + VUSES)
    a.LD(2, 4, 0)
    a.SLTU(5, T2, 2)
    a.BEQZ(5, "m_maxnext")
    a.ST(9, VARS + VUSES, 2)
    a.label("m_maxnext")
    a.ADDI(4, 4, 1)
    a.ADDI(1, 1, 1)
    a.JMP("m_max")
    a.label("m_ret")
    a.MOV(SP, MF)
    a.pop(MF)
    a.pop(RA)
    a.JR(RA)

    # ------------------------------------------------ expr(r3) -> r3, r1 type, r2 shape kind,
    # r7 shape payload
    a.label("expr")
    a.push(RA)
    a.LD(1, 3, TAPE_K)
    a.LD(7, 3, TAPE_V)
    a.BEQI(1, NUM, "e_num")
    a.BEQI(1, NAT, "e_nat")
    a.BEQI(1, PUN, "e_pun")
    a.BNEI(1, NAME, "fail")
    a.BEQI(7, K["True"], "e_bool")
    a.BEQI(7, K["False"], "e_bool")
    a.is_at(4, PUN, pcode("("), off=1)
    a.BNEZ(4, "e_call")
    a.MOV(1, 7)
    a.JAL(RA, "lookup")
    a.BEQI(2, -1, "fail")
    a.MOV(4, 2)
    a.JAL(RA, "use")
    a.emit_op("VAR", 2)
    a.ADDI(3, 3, 1)
    a.MOV(7, 2)
    a.var_addr(9, 2)
    a.LD(1, 9, VARS + VTY)
    a.LI(2, NVAR)
    a.JMP("e_ret")
    a.label("e_num")
    a.emit_op("LIT", 7)
    a.ADDI(3, 3, 1)
    a.LI(1, U32)
    a.LI(2, NOTHER)
    a.JMP("e_ret")
    a.label("e_nat")
    a.emit_op("LIT", 7)
    a.ADDI(3, 3, 1)
    a.LI(1, NATT)
    a.LI(2, NLIT)
    a.JMP("e_ret")
    a.label("e_bool")
    a.SEQI(4, 7, K["True"])
    a.ADDI(3, 3, 1)
    a.need(PUN, pcode("{"))
    a.need(PUN, pcode("}"))
    a.emit_op("LIT", 4)
    a.LI(1, BOOL)
    a.SUB(2, 0, 4)
    a.ADDI(2, 2, NFALSE)  # True -> NTRUE 3, False -> NFALSE 4
    a.label("e_ret")
    a.pop(RA)
    a.JR(RA)

    # (e), (e : T), or a group; group frame (SP-relative): pending + / - -3, comparison -2,
    # spare -1
    a.label("e_pun")
    a.BNEI(7, pcode("("), "fail")
    a.ADDI(3, 3, 1)
    a.JAL(RA, "expr")
    a.is_at(4, PUN, pcode(")"))
    a.BEQZ(4, "e_annot")
    a.ADDI(3, 3, 1)
    a.JMP("e_ret")
    a.label("e_annot")
    a.is_at(4, PUN, pcode(":"))
    a.BEQZ(4, "e_group")
    a.type_at(4, off=1)
    a.BNE(4, 1, "fail")
    a.ADDI(3, 3, 2)
    a.need(PUN, pcode(")"))
    a.JMP("e_ret")
    a.label("e_group")
    a.BNEI(1, U32, "fail")
    a.ADDI(SP, SP, 3)
    a.ST(SP, -3, 0)
    a.ST(SP, -2, 0)
    a.label("g_loop")
    a.LD(1, 3, TAPE_K)
    a.BNEI(1, PUN, "g_end")
    a.LD(2, 3, TAPE_V)
    for s_, op in PUNS.items():
        nxt = a.fresh("gop")
        a.BNEI(2, pcode(s_), nxt)
        a.LI(5, f"op_{op}")
        a.JMP({"+": "g_add", "-": "g_add", "*": "g_mul", "/": "g_mul", "%": "g_mul"}.get(s_, "g_cmp"))
        a.label(nxt)
    a.JMP("g_end")
    a.label("g_add")
    a.LD(1, SP, -3)
    a.BEQZ(1, "g_a1")
    a.emit_regs(1)
    a.label("g_a1")
    a.ST(SP, -3, 5)
    a.JMP("g_operand")
    a.label("g_mul")
    a.ST(SP, -1, 5)
    a.ADDI(3, 3, 1)
    a.JAL(RA, "expr")
    a.BNEI(1, U32, "fail")
    a.LD(5, SP, -1)
    a.emit_regs(5)
    a.JMP("g_loop")
    a.label("g_cmp")
    a.LD(1, SP, -2)
    a.BNEZ(1, "fail")  # two comparisons
    a.LD(1, SP, -3)
    a.BEQZ(1, "g_c1")
    a.emit_regs(1)
    a.ST(SP, -3, 0)
    a.label("g_c1")
    a.ST(SP, -2, 5)
    a.label("g_operand")
    a.ADDI(3, 3, 1)
    a.JAL(RA, "expr")
    a.BNEI(1, U32, "fail")
    a.JMP("g_loop")
    a.label("g_end")
    a.need(PUN, pcode(":"))
    a.need(NAME, K["U32"])
    a.need(PUN, pcode(")"))
    a.LD(1, SP, -3)
    a.BEQZ(1, "g_e1")
    a.emit_regs(1)
    a.label("g_e1")
    a.LI(1, U32)
    a.LD(4, SP, -2)
    a.BEQZ(4, "g_e2")
    a.emit_regs(4)
    a.LI(1, BOOL)
    a.label("g_e2")
    a.LI(2, NOTHER)
    a.ADDI(SP, SP, -3)
    a.JMP("e_ret")

    # call frame (SP-relative): handler or 0 (U32.from_nat) / the def's offset -7, wanted arity -6,
    # the wanted type (builtin) or the PTY base (def) -5, is a def -4, self-call -3, arguments -2,
    # order -1
    a.label("e_call")
    a.ADDI(SP, SP, 7)
    a.ST(SP, -2, 0)
    a.ST(SP, -1, 0)  # EQ
    a.ST(SP, -3, 0)
    a.ST(SP, -4, 0)
    for name, op, n, ty in (("U32.from_nat", None, 1, NATT), ("U32.is_eq", "ISEQ", 2, U32),
                            ("U32.min", "MIN", 2, U32), ("U32.max", "MAX", 2, U32)):
        nxt = a.fresh("bi")
        a.BNEI(7, K[name], nxt)
        a.LI(1, f"op_{op}" if op else 0)
        a.ST(SP, -7, 1)
        a.LI(1, n)
        a.ST(SP, -6, 1)
        a.LI(1, ty)
        a.ST(SP, -5, 1)
        a.JMP("c_args")
        a.label(nxt)
    a.LD(4, 0, NDEFS)  # a def so far (earlier, or this one)
    a.ADD(4, 4, 4)
    a.ADD(4, 4, 4)
    a.ADD(4, 4, 4)
    a.LI(5, 0)
    a.label("c_scan")
    a.BEQ(5, 4, "fail")
    a.LD(6, 5, DEFS)
    a.BEQ(6, 7, "c_found")
    a.ADDI(5, 5, 8)
    a.JMP("c_scan")
    a.label("c_found")
    a.ST(SP, -7, 5)
    a.LD(1, 5, DEFS + 1)
    a.ST(SP, -6, 1)
    a.LD(1, 5, DEFS + 2)
    a.ST(SP, -5, 1)
    a.LI(1, 1)
    a.ST(SP, -4, 1)
    a.LD(1, 0, CUR)
    a.SEQ(1, 1, 5)
    a.ST(SP, -3, 1)
    a.label("c_args")
    a.ADDI(3, 3, 2)
    a.label("c_loop")
    a.is_at(1, PUN, pcode(")"))
    a.BNEZ(1, "c_end")
    a.LD(1, SP, -2)
    a.BEQZ(1, "c_first")
    a.need(PUN, pcode(","))
    a.label("c_first")
    a.JAL(RA, "expr")
    a.LD(4, SP, -2)  # nargs
    a.LD(5, SP, -6)
    a.SLTU(6, 4, 5)
    a.BEQZ(6, "fail")  # too many arguments
    a.LD(5, SP, -5)
    a.LD(6, SP, -4)
    a.BEQZ(6, "c_bity")
    a.ADD(5, 5, 4)
    a.LD(5, 5, PTY)
    a.label("c_bity")
    a.BNE(1, 5, "fail")  # the argument's type
    a.LD(6, SP, -3)
    a.BEQZ(6, "c_next")
    a.LD(6, SP, -1)
    a.BNEZ(6, "c_next")  # the order is decided already
    a.MOV(1, 2)  # descend(shape, node(param nargs))
    a.MOV(2, 7)
    a.node(5, 6, 4)
    a.MOV(4, 5)
    a.MOV(5, 6)
    a.JAL(RA, "descend")
    a.ST(SP, -1, 1)
    a.label("c_next")
    a.LD(4, SP, -2)
    a.ADDI(4, 4, 1)
    a.ST(SP, -2, 4)
    a.JMP("c_loop")
    a.label("c_end")
    a.LD(4, SP, -2)
    a.LD(5, SP, -6)
    a.BNE(4, 5, "fail")  # too few arguments
    a.LD(6, SP, -3)
    a.BEQZ(6, "c_emit")
    a.LD(6, SP, -1)
    a.BNEI(6, LT, "fail")  # a self-call must decrease
    a.label("c_emit")
    a.LD(6, SP, -4)
    a.LD(5, SP, -7)
    a.BEQZ(6, "c_builtin")
    a.LD(1, 5, DEFS + 4)
    a.emit_op("CALL", 1, 4)
    a.LD(1, 5, DEFS + 3)
    a.JMP("c_done")
    a.label("c_builtin")
    a.BEQZ(5, "c_noop")
    a.emit_regs(5)
    a.label("c_noop")
    a.LI(1, U32)
    a.LI(6, "op_ISEQ")
    a.BNE(5, 6, "c_done")
    a.LI(1, BOOL)
    a.label("c_done")
    a.ADDI(3, 3, 1)
    a.LI(2, NOTHER)
    a.ADDI(SP, SP, -7)
    a.JMP("e_ret")

    # ------------------------------------------------ descend(a = (r1, r2), p = (r4, r5)) -> r1
    # Bend2's term_descend (bytecode.Compiler.descend); recursive
    a.label("descend")
    a.push(RA)
    a.BNEI(1, NVAR, "d1")
    a.node(1, 2, 2)  # a variable -> its expansion
    a.label("d1")
    a.BNEI(1, NLIT, "d2")
    a.BEQI(4, NVAR, "d2")
    a.BNEZ(2, "d_litsucc")  # a literal against a constructor: 0n or 1n + (k-1)
    a.LI(1, NZERO)
    a.JMP("d2")
    a.label("d_litsucc")
    a.LI(1, NSUCCL)
    a.ADDI(2, 2, -1)
    a.label("d2")
    a.BNEI(4, NVAR, "d3")
    a.BNEI(1, NVAR, "d_gt")
    a.BNE(2, 5, "d_gt")
    a.LI(1, EQ)
    a.JMP("d_ret")
    a.label("d3")
    a.MOV(6, 1)  # a's constructor (a SUCC of a literal is a SUCC)
    a.BNEI(6, NSUCCL, "d3b")
    a.LI(6, NSUCC)
    a.label("d3b")
    a.BNE(6, 4, "d4")
    a.BNEI(4, NSUCC, "d_eq")
    a.node(4, 5, 5)  # both SUCC: compare the fields
    a.BNEI(1, NSUCCL, "d_afield")
    a.LI(1, NLIT)
    a.JMP("d_rec")
    a.label("d_afield")
    a.node(1, 2, 2)
    a.label("d_rec")
    a.JAL(RA, "descend")
    a.JMP("d_ret")
    a.label("d4")
    a.BNEI(4, NSUCC, "d_gt")
    a.node(4, 5, 5)
    a.JAL(RA, "descend")
    a.BEQI(1, GT, "d_gt")
    a.LI(1, LT)
    a.JMP("d_ret")
    a.label("d_eq")
    a.LI(1, EQ)
    a.JMP("d_ret")
    a.label("d_gt")
    a.LI(1, GT)
    a.label("d_ret")
    a.pop(RA)
    a.JR(RA)


def build(compile_only=False):
    """-> (code, handler addresses, labels). compile_only halts where evaluation would start."""
    a = Fw()
    program(a)
    n = len(a.code)
    start_eval(a)
    if compile_only:  # same length, so every later label keeps its address
        a.code[n:] = [("HALT",)] * (len(a.code) - n)
    a.label("fail")
    a.FAIL()
    define(a)
    routines(a)
    lexer(a)
    eval_handlers(a)
    return a.link(), {op: a.labels[f"op_{op}"] for op in OPS}, dict(a.labels)


FIRMWARE, HANDLERS, LABELS = build()
MEM_WORDS = 8192  # bvm.VM(FIRMWARE, memory_words=MEM_WORDS)
COMPILE_ONLY = build(compile_only=True)[0]


if __name__ == "__main__":
    # the MUL / DIV / MOD routines against Python on edge and random operands
    import random
    from bvm import VM
    halt = FIRMWARE.index(("HALT",))
    rng = random.Random(1)
    edge = [0, 1, 2, 3, 7, 10, 255, 256, 65535, 2**31 - 1, 2**31, 2**31 + 1, 2**32 - 2, 2**32 - 1]
    pairs = [(x, y) for x in edge for y in edge] + [
        (rng.getrandbits(rng.randint(1, 32)), rng.getrandbits(rng.randint(1, 32))) for _ in range(3000)]
    for x, y in pairs:
        for routine in ("mulu", "divu"):
            if routine == "divu" and y == 0:
                continue
            vm = VM(FIRMWARE, memory_words=MEM_WORDS)
            vm.pc, vm.r[1], vm.r[2], vm.r[10] = LABELS[routine], x, y, halt
            vm.run()
            want = [x * y % 2**32] if routine == "mulu" else [x // y, x % y]
            got = [vm.r[4]] if routine == "mulu" else [vm.r[4], vm.r[5]]
            assert got == want, (routine, x, y, got, want)
    print(f"firmware2 routines ok: mulu / divu on {len(pairs)} operand pairs")
