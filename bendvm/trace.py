"""C2a: the trace the circuit emits, and the circuit's per-position function in Python.

Token vocabulary (engine tokens, ids relative to the engine's first id):
  INS(pc)            0 .. NPC-1      the instruction at pc runs (self-describing record head)
  MORE_d(b)          NPC + 256 (d-1) + b, d = 1..3   a value byte d bytes before its LAST
  LAST(b)            NPC + 768 + b   the value's last (least significant) byte; bytes are
                                     most significant first
  END                NPC + 1024      after HALT or FAIL (the engine stops either way)
MORE carries its distance d to LAST so that at a LAST position a MORE_d at conv tap d is part of
the current value by construction (a plain MORE at tap 3 could belong to an earlier record:
MORE LAST INS LAST).
A record is INS(pc) followed by the value's bytes when the instruction writes a computed value
(MOV, ALU ops, SEQI, ADDI, LD, ST, OUT); LI / JAL (value fixed by pc), branches, JR, HALT and
FAIL are INS only (DESIGN.md Phase 2). At LAST, the MORE_d tokens at t-d plus LAST are the whole
value: the conv window (t-3 .. t) holds it, one byte per offset.

trace(code, tape)       -> the token list, from bvm.VM's execution
TraceMachine(code, tape) consumes tokens one at a time and predicts the next one using only
  what a position can know in the circuit: the current token, the conv window of the last 4
  tokens, and the state written by earlier positions (registers, memory, a latch written at
  each INS). generate() runs it on its own outputs; it must reproduce trace() exactly.

Run: python bendvm/trace.py [logs.jsonl ...]  (all programs)
"""

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bvm import M, OPS3, VM, VMFail  # noqa: E402
from firmware2 import FIRMWARE, MEM_WORDS  # noqa: E402
from lexer_fw import Stored, load_program, token_words  # noqa: E402

NPC = len(FIRMWARE)
MORE, LAST, END = NPC, NPC + 768, NPC + 1024  # MORE_d(b) = MORE + 256 (d - 1) + b
COMPUTED = OPS3 | {"MOV", "SHR1", "XOR", "ROTN", "SEQI", "ADDI", "LD", "ST", "OUT"}


def value_bytes(v):
    """Big-endian bytes of v, as few as possible (at least one)."""
    n = max(1, (v.bit_length() + 7) // 8)
    return [(v >> (8 * (n - 1 - i))) & 255 for i in range(n)]


def value_tokens(v):
    bs = value_bytes(v)
    n = len(bs)
    return [MORE + 256 * (n - 2 - i) + b for i, b in enumerate(bs[:-1])] + [LAST + bs[-1]]


class Tracer(VM):
    """bvm.VM that records the token stream."""

    def run(self):
        self.toks = []
        code, r = self.code, self.r
        while not self.halted:
            op, *a = code[self.pc]
            self.toks.append(self.pc)
            if op == "FAIL":
                self.toks.append(END)
                return self.toks
            if op in COMPUTED:
                v = self.computed_value(op, a)
                if v is not None:
                    self.toks += value_tokens(v)
            self.max_steps = self.steps + 1
            try:
                VM.run(self)
            except VMFail as e:
                if str(e) != "step limit":
                    raise
        self.toks.append(END)
        return self.toks

    def computed_value(self, op, a):
        r = self.r
        if op == "OUT":
            return r[a[0]]
        if op == "ST":
            return r[a[2]]
        if op == "MOV":
            return r[a[1]]
        if op == "SHR1":
            return r[a[1]] >> 1
        if op == "XOR":
            return r[a[1]] ^ r[a[2]]
        if op == "ROTN":
            k = 4 * (r[a[2]] % 8)
            return ((r[a[1]] << k) | (r[a[1]] >> (32 - k))) % M if k else r[a[1]]
        if op == "SEQI":
            return int(r[a[1]] == a[2] % M)
        if op == "ADDI":
            return (r[a[1]] + a[2]) % M
        if op == "LD":
            addr = (r[a[1]] + a[2]) % M
            return self.mem[addr] if 0 <= addr < len(self.mem) else None  # a fault: no value
        x, y = r[a[1]], r[a[2]]
        if op in ("DIVU", "REMU"):
            return None if y == 0 else (x // y if op == "DIVU" else x % y)
        return {"ADD": lambda: (x + y) % M, "SUB": lambda: (x - y) % M, "MUL": lambda: x * y % M,
                "SLTU": lambda: int(x < y), "SLEU": lambda: int(x <= y), "SEQ": lambda: int(x == y)}[op]()


def trace(code, prog):
    """prog: a lexer_fw.Stored program (the stored tokens; firmware2 lexes them first)."""
    vm = Tracer(code, memory_words=MEM_WORDS, max_steps=20_000_000)
    load_program(vm, prog)
    try:
        return vm.run()
    except VMFail:
        return vm.toks  # a fault: the trace stops at the faulting instruction's INS


class TraceMachine:
    """The circuit's per-position function. State: registers, memory, the latch (the INS of the
    current record) and the conv window; nothing else carries over between positions."""

    def __init__(self, code, prog):
        self.code = code
        vm = VM(code, memory_words=MEM_WORDS)
        load_program(vm, prog)
        self.mem = vm.mem  # the tokens as capture stored them
        self.r = [0] * 16
        self.latch = None  # pc of the current record's INS
        self.window = []  # the last 4 tokens (t-3 .. t), as the conv sees them

    def feed(self, tok):
        """Low layers at this position: the writes this token carries. -> None."""
        self.window = (self.window + [tok])[-4:]
        if tok < NPC:
            self.latch = tok
            op, *a = self.code[tok]
            if op == "LI":
                self.wreg(a[0], a[1])
            elif op == "JAL":
                self.wreg(a[0], tok + 1)
        elif LAST <= tok < END:
            v = self.window_value()
            op, *a = self.code[self.latch]
            if op == "ST":
                self.mem[(self.r[a[0]] + a[1]) % M] = v
            elif op not in ("OUT",):
                self.wreg(a[0], v)

    def window_value(self):
        """The value whose LAST byte is the current token: the MORE_d tokens at t-d."""
        v = self.window[-1] - LAST
        for d in (1, 2, 3):
            if d < len(self.window):
                t = self.window[-1 - d]
                if MORE <= t < LAST and (t - MORE) // 256 == d - 1:
                    v += ((t - MORE) % 256) << (8 * d)
        return v

    def wreg(self, rd, v):
        if rd:
            self.r[rd] = v % M

    def next_token(self):
        """High layers at this position: the next token, from state and the window."""
        tok = self.window[-1]
        if tok == END:
            return None
        pc = self.latch
        op, *a = self.code[pc]
        if tok < NPC:  # at INS(pc): the record's first value byte, or the next INS
            if op in COMPUTED:
                v = Tracer.computed_value(self, op, a)
                if v is None:
                    return None  # a fault (memory / division): the trace stops
                return value_tokens(v)[0]
            return self.next_pc(op, a, pc)
        # at a value byte: the next byte of the same value, or (after LAST) the next INS.
        # The value is recomputed from state (unchanged within the record); the byte index is
        # the run of value tokens since the INS (in the window).
        if tok >= LAST:
            return self.next_pc(op, a, pc)
        d = (tok - MORE) // 256 + 1  # this is MORE_d: the next byte is d - 1 before LAST
        toks = value_tokens(Tracer.computed_value(self, op, a))
        return toks[len(toks) - d]

    def next_pc(self, op, a, pc):
        r = self.r
        if op in ("HALT", "FAIL"):
            return END
        if op in ("BEQZ", "BNEZ"):
            take = (r[a[0]] == 0) == (op == "BEQZ")
            return a[1] if take else pc + 1
        if op in ("BEQI", "BNEI"):
            take = (r[a[0]] == a[1] % M) == (op == "BEQI")
            return a[2] if take else pc + 1
        if op in ("BEQ", "BNE"):
            take = (r[a[0]] == r[a[1]]) == (op == "BEQ")
            return a[2] if take else pc + 1
        if op in ("JMP", "JAL"):
            return a[-1]
        if op == "JR":
            return r[a[0]]
        return pc + 1

    def generate(self, limit=1_000_000):
        out = [0]  # the engine starts at INS(0)
        self.feed(0)
        while len(out) < limit:
            nxt = self.next_token()
            if nxt is None:
                return out
            out.append(nxt)
            self.feed(nxt)
        return out


if __name__ == "__main__":
    from test_firmware import load_programs

    tok, rom, progs = load_programs(sys.argv[1:])
    words = token_words(rom)
    n, bad, lens = 0, 0, []
    for origin, src in progs:
        prog = Stored(tok.encode(src, add_special_tokens=False), words)
        try:
            load_program(VM(FIRMWARE, memory_words=MEM_WORDS), prog)
        except ValueError:
            continue
        want = trace(FIRMWARE, prog)
        got = TraceMachine(FIRMWARE, prog).generate(limit=len(want) + 5)
        n += 1
        if got != want:
            bad += 1
            if bad <= 5:
                i = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
                print("MISMATCH", origin, "at", i, "got", got[i:i + 3], "want", want[i:i + 3])
        lens.append(len(want))
    q = statistics.quantiles(lens, n=10)
    print(f"TraceMachine.generate == trace on {n - bad}/{n} programs; tokens median "
          f"{int(statistics.median(lens))} p90 {int(q[-1])} max {max(lens)} (failing runs included)")
    sys.exit(1 if bad else 0)
