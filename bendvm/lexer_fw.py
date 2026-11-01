"""C3b: the firmware lexer (DESIGN.md "C3 design"): BVM code that turns the stored program tokens
into the tape, exactly as capture2.Lexer does. It runs first, before firmware2's compile pass.

Stored tokens (written by the circuit's capture as Qwen writes; by load_tokens() in tests): token n
has W0 at TOK + n (its pattern's handler pc), W1 / W2 / W3 at TOK + 512 / 1024 / 1536 + n (the
payloads, capture2.rom_words); after the last token, W0 = the EOF handler. TOK = 6144 lies in the
compiler's stack, which is unused until the lexer has finished.

Registers while lexing: r1 tape entries p, r2 emitted value, r3 token pointer, r4 r5 payloads,
r6 scratch, r7 emitted kind, r8 link (emit), r9 word kind (0 none, 1 name, 2 number, 3 Nat,
4 bad), r10 word value, r11 overflow (BIG), r12 pending '-' '<' '>' (0 none), r13 indentation
count (-1 = not counting), r14 link (flush routines), r15 dispatch / link (segment routines).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bytecode import MAX_TAPE  # noqa: E402
from capture import name_hash  # noqa: E402
from capture2 import NPAY  # noqa: E402
from firmware import BIG, NAME, NAT, NL, NUM, PUN, TAPE_K, TAPE_V  # noqa: E402

TOK, MAX_TOK = 6144, 511  # W0 region; W_i at TOK + 512 i; at most 511 tokens + EOF
PATTERNS = [tuple(p) for p in json.load(open(Path(__file__).resolve().parent / "lexer_patterns.json"))["patterns"]]
H_N = name_hash("n")[0]
LIM = (2**32 - 1) // 10  # 429496729: v * 10 + d overflows iff v > LIM or (v == LIM and d > 5)
NOCOL = 2**32 - 1


def pattern_label(p):
    return "lex_" + "_".join(p)


def lexer(a):
    """Emit the lexer (entered at 'lex_start', leaves to 'lex_done'). Word kinds in r9: 0 none,
    1 name (key r10), 2 number (value r10, digits so far r11), 3 Nat, 4 bad, 5 / 6 a number / Nat
    that overflowed 32 bits (BIG)."""
    a.label("lex_start")
    a.LI(1, 0)
    a.LI(3, TOK)
    a.LI(9, 0)
    a.LI(12, 0)
    a.LI(13, NOCOL)
    a.label("lex_next")
    a.LD(15, 3, 0)
    a.JR(15)

    def emit(kind, val_reg):
        """Append (kind, r[val_reg]) to the tape, inline."""
        a.BEQI(1, MAX_TAPE, "fail")  # more entries than the tape area holds
        a.LI(7, kind)
        a.ST(1, TAPE_K, 7)
        a.ST(1, TAPE_V, val_reg)
        a.ADDI(1, 1, 1)

    def times10_plus(dreg):
        """r10 = r10 * 10 + r[dreg] (mod 2^32), via adds; uses r6."""
        a.ADD(6, 10, 10)
        a.ADD(10, 6, 6)
        a.ADD(10, 10, 10)
        a.ADD(10, 10, 6)
        a.ADD(10, 10, dreg)

    # pattern handlers: per segment an inline fast path for the common state (not counting
    # indentation, no pending '-' '<' '>'; no word, or a name / short number in progress), else
    # the general routine; then the next token
    for p in PATTERNS:
        a.label(pattern_label(p))
        w = 1
        for seg in p:
            slow, cont = a.fresh("slow"), a.fresh("cont")
            off = [512 * (w + i) for i in range(NPAY[seg])]
            a.BNEI(13, NOCOL, slow)
            a.BNEZ(12, slow)
            if seg in ("SP", "NL", "PU"):  # flush a name in progress inline
                none = a.fresh("none")
                a.BEQZ(9, none)
                a.BNEI(9, 1, slow)
                emit(NAME, 10)
                a.LI(9, 0)
                a.label(none)
                if seg == "NL":
                    a.BEQZ(1, cont)  # nothing emitted yet: a leading newline is ignored
                    a.LI(13, 0)
                elif seg == "PU":
                    pend = a.fresh("pend")
                    a.LD(4, 3, off[0])
                    for ch in "-<>":
                        a.BEQI(4, ord(ch), pend)
                    emit(PUN, 4)
                    a.JMP(cont)
                    a.label(pend)
                    a.MOV(12, 4)
            elif seg in ("NMa", "NMdot"):
                new = a.fresh("new")
                if seg == "NMa":
                    a.BEQZ(9, new)
                a.BNEI(9, 1, slow)
                a.LD(4, 3, off[0])  # a name continues: key = rotl(key, 4 len) ^ H
                a.LD(5, 3, off[1])
                a.ROTN(10, 10, 4)
                a.XOR(10, 10, 5)
                if seg == "NMa":
                    a.JMP(cont)
                    a.label(new)  # a new name
                    a.LI(9, 1)
                    a.LD(10, 3, off[1])
            else:  # NMd
                new, name = a.fresh("new"), a.fresh("name")
                a.BEQZ(9, new)
                a.BEQI(9, 1, name)
                a.BNEI(9, 2, slow)
                a.LI(6, 9)
                a.SLTU(6, 11, 6)
                a.BEQZ(6, slow)  # the 10th digit or later: the overflow check
                a.LD(4, 3, off[0])
                times10_plus(4)
                a.ADDI(11, 11, 1)
                a.JMP(cont)
                a.label(new)
                a.LI(9, 2)
                a.LD(10, 3, off[0])
                a.LI(11, 1)
                a.JMP(cont)
                a.label(name)  # a digit continues a name
                a.LD(5, 3, off[1])
                a.LI(6, 1)
                a.ROTN(10, 10, 6)
                a.XOR(10, 10, 5)
            a.JMP(cont)
            a.label(slow)
            for r, o in zip((4, 5), off):
                a.LD(r, 3, o)
            a.JAL(15, "seg_" + seg)
            a.label(cont)
            w += NPAY[seg]
        a.ADDI(3, 3, 1)
        a.LD(15, 3, 0)
        a.JR(15)

    # ----- the general routines (the slow paths) -----
    # flush_word; link r14
    a.label("lx_flush_word")
    a.BEQZ(9, "fw_ret")
    a.BEQI(9, 4, "fail")  # a bad word: not BendLite
    for kind, wk in ((NAME, 1), (NUM, 2), (NAT, 3)):
        nxt = a.fresh("fw")
        a.BNEI(9, wk, nxt)
        emit(kind, 10)
        a.LI(9, 0)
        a.JR(14)
        a.label(nxt)
    emit(BIG, 0)  # 5 / 6: overflowed
    a.LI(9, 0)
    a.label("fw_ret")
    a.JR(14)

    # flush_pending (single, unpaired); link r14
    a.label("lx_flush_pend")
    a.BEQZ(12, "fp_ret")
    emit(PUN, 12)
    a.LI(12, 0)
    a.label("fp_ret")
    a.JR(14)

    def start_col():
        """A non-space, non-newline char while counting indentation: emit NL(col)."""
        skip = a.fresh("sc")
        a.BEQI(13, NOCOL, skip)
        emit(NL, 13)
        a.LI(13, NOCOL)
        a.label(skip)

    # SP (r4 = count)
    a.label("seg_SP")
    a.BEQI(13, NOCOL, "sp_sep")
    a.ADD(13, 13, 4)
    a.JR(15)
    a.label("sp_sep")
    a.JAL(14, "lx_flush_word")
    a.JAL(14, "lx_flush_pend")
    a.JR(15)

    # NL
    a.label("seg_NL")
    a.BEQI(13, NOCOL, "nl_first")
    a.LI(13, 0)
    a.JR(15)
    a.label("nl_first")
    a.JAL(14, "lx_flush_word")
    a.JAL(14, "lx_flush_pend")
    a.BEQZ(1, "nl_lead")  # nothing emitted yet: a leading newline is ignored
    a.LI(13, 0)
    a.label("nl_lead")
    a.JR(15)

    # PU (r4 = char code)
    a.label("seg_PU")
    start_col()
    a.JAL(14, "lx_flush_word")
    a.BEQZ(12, "pu_nopend")
    for first, second, code in (("-", ">", 1001), ("<", "=", 1002), (">", "=", 1003)):
        nxt = a.fresh("pu")
        a.BNEI(12, ord(first), nxt)
        a.BNEI(4, ord(second), nxt)
        a.LI(2, code)
        emit(PUN, 2)
        a.LI(12, 0)
        a.JR(15)
        a.label(nxt)
    a.JAL(14, "lx_flush_pend")  # the pending char stands alone
    a.label("pu_nopend")
    for ch in "-<>":
        a.BEQI(4, ord(ch), "pu_pend")
    emit(PUN, 4)
    a.JR(15)
    a.label("pu_pend")
    a.MOV(12, 4)
    a.JR(15)

    # name-char parts: NMa / NMdot (r4 = len mod 8, r5 = H), NMd (r4 = digit, r5 = H(digit))
    for seg in ("NMa", "NMdot", "NMd"):
        a.label("seg_" + seg)
        start_col()
        a.JAL(14, "lx_flush_pend")
        new, name, num, big, bad = (a.fresh(seg) for _ in range(5))
        a.BEQZ(9, new)
        a.BEQI(9, 1, name)
        a.BEQI(9, 2, num)
        a.BEQI(9, 5, big)
        a.JMP(bad)
        a.label(new)
        if seg == "NMdot":
            a.JMP("fail")  # '.' cannot start a word
        elif seg == "NMa":
            a.LI(9, 1)
            a.MOV(10, 5)
            a.JR(15)
        else:
            a.LI(9, 2)
            a.MOV(10, 4)
            a.LI(11, 1)
            a.JR(15)
        a.label(name)  # any name chars continue a name: key = rotl(key, 4 len) ^ H
        if seg == "NMd":
            a.LI(6, 1)
            a.ROTN(10, 10, 6)
        else:
            a.ROTN(10, 10, 4)
        a.XOR(10, 10, 5)
        a.JR(15)
        a.label(num)
        if seg == "NMd":  # v * 10 + d; from the 10th digit on, check for overflow
            check, ovf = a.fresh("nd"), a.fresh("nd")
            a.LI(6, 9)
            a.SLTU(6, 11, 6)
            a.BEQZ(6, check)
            times10_plus(4)
            a.ADDI(11, 11, 1)
            a.JR(15)
            a.label(check)
            a.LI(6, LIM)
            a.SLTU(6, 6, 10)
            a.BNEZ(6, ovf)
            a.BNEI(10, LIM, "nd_ok")
            a.LI(6, 5)
            a.SLTU(6, 6, 4)
            a.BNEZ(6, ovf)
            a.label("nd_ok")
            times10_plus(4)
            a.ADDI(11, 11, 1)
            a.JR(15)
            a.label(ovf)
            a.LI(9, 5)
            a.JR(15)
            a.label(big)  # an overflowed number stays BIG
            a.JR(15)
        elif seg == "NMa":  # the Nat suffix: exactly "n"
            a.BNEI(4, 1, bad)
            a.BNEI(5, H_N, bad)
            a.LI(9, 3)
            a.JR(15)
            a.label(big)
            a.BNEI(4, 1, bad)
            a.BNEI(5, H_N, bad)
            a.LI(9, 6)
            a.JR(15)
        else:
            a.label(big)
        a.label(bad)
        a.LI(9, 4)
        a.JR(15)

    # end of the program: flush, the final NL, the end marker and 2 pads (all written)
    a.label("lex_eof")
    a.JAL(14, "lx_flush_word")
    a.JAL(14, "lx_flush_pend")
    a.BEQI(13, NOCOL, "eof_nocol")
    emit(NL, 0)
    a.label("eof_nocol")
    for k in range(3):
        a.ST(1, TAPE_K + k, 0)
        a.ST(1, TAPE_V + k, 0)
    a.JMP("lex_done")


class Stored:
    """A program as the circuit's capture stores it: its token ids and the ROM words per id."""

    def __init__(self, ids, words):
        self.ids, self.words = list(ids), words


def token_words(rom):
    """capture.build_rom's id -> text  ->  id -> capture2.rom_words (supported tokens only)."""
    from capture2 import rom_words
    return {i: w for i, s in rom.items() if (w := rom_words(s)) is not None}


def load_program(vm, prog):
    from firmware2 import LABELS
    load_tokens(vm, prog.ids, prog.words, LABELS)


def load_tokens(vm, ids, words, labels):
    """Store a program's tokens as the circuit's capture will: W0..W3 per token, then EOF.
    words: token id -> capture2.rom_words(...) or None (unsupported: W0 = fail)."""
    if len(ids) > MAX_TOK:
        raise ValueError(f"{len(ids)} tokens: more than the engine stores")
    for n, i in enumerate(ids):
        w = words.get(i)
        if w is None or w[0] not in PATTERNS:
            vm.mem[TOK + n] = labels["fail"]
            continue
        pattern, pay = w
        vm.mem[TOK + n] = labels[pattern_label(pattern)]
        for k in range(3):  # all 4 words, absent payloads as 0 (as the circuit writes them)
            vm.mem[TOK + 512 * (k + 1) + n] = pay[k] if k < len(pay) else 0
    vm.mem[TOK + len(ids)] = labels["lex_eof"]
    for k in range(3):
        vm.mem[TOK + 512 * (k + 1) + len(ids)] = 0
