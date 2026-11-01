"""BVM (5b-2): BendVM's instruction set, a Python assembler DSL, and the step-exact reference VM
with the trace each instruction emits (the tokens the engine would generate).

Machine: 16 registers r0..r15 (32-bit; r0 reads 0), PC, word-addressed memory of 32-bit words.
Every instruction makes AT MOST ONE state write (one register or one memory word), besides
moving the PC. Its trace record (DESIGN.md, Phase 2: the circuit can't advance state without a
token) is INS(pc), naming the instruction, then the written value's tokens only when the value
is computed (not fixed by pc):
  MOV, ALU ops, SEQI, ADDI, LD (register write), ST (memory write), OUT    INS + value tokens
  LI, JAL (value fixed by pc), branches, JMP, JR, HALT, FAIL                INS only
A value is 1 token if < 2^val_bits, else 2 (val_bits 16: VAL16 / VALH+VALL), or with
val_bits 8 one token per byte of its significant bytes.

Instructions (rd, rs, rt registers; imm a constant):
  LI rd, imm        rd = imm (mod 2^32); imm may be a label (its address)
  MOV rd, rs
  SHR1 rd, rs       rd = rs >> 1 (logical; cheap in the circuit, used by firmware2's MUL/DIV/MOD)
  XOR rd, rs, rt    ROTN rd, rs, rt (rotate rs left by 4 (rt mod 8) bits): the lexer's name keys
  ADD SUB MUL DIVU REMU SLTU SLEU SEQ rd, rs, rt      (U32; DIVU/REMU by 0 -> FAIL)
  ADDI rd, rs, imm
  LD rd, rs, imm    rd = mem[rs + imm]
  ST rs, imm, rt    mem[rs + imm] = rt
  BEQZ rs, L / BNEZ rs, L / JMP L
  SEQI rd, rs, imm  rd = [rs == imm];  BEQI / BNEI rs, imm, L;  BEQ / BNE rs, rt, L
  JAL rd, L         rd = PC + 1; jump L          (call)
  JR rs             jump to rs                    (return)
  OUT rs / HALT / FAIL
"""

from dataclasses import dataclass, field

M = 2**32
OPS3 = {"ADD", "SUB", "MUL", "DIVU", "REMU", "SLTU", "SLEU", "SEQ"}


class VMFail(Exception):
    pass


@dataclass
class Asm:
    code: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)
    _n: int = 0

    def label(self, name):
        if name in self.labels:
            raise ValueError(f"label {name} twice")
        self.labels[name] = len(self.code)

    def fresh(self, base="L"):
        self._n += 1
        return f"_{base}{self._n}"

    def emit(self, op, *args):
        self.code.append((op, *args))

    def __getattr__(self, op):
        if op.isupper():
            return lambda *a: self.emit(op, *a)
        raise AttributeError(op)

    def link(self):
        out = []
        for ins in self.code:
            op, *a = ins
            a = [self.labels[x] if isinstance(x, str) and op in ("BEQZ", "BNEZ", "JMP", "JAL", "BEQI", "BNEI", "BEQ", "BNE", "LI") and x in self.labels else x
                 for x in a]
            for x in a:
                if isinstance(x, str):
                    raise ValueError(f"unresolved label {x} in {ins}")
            out.append((op, *a))
        return out


class VM:
    def __init__(self, code, memory_words=1 << 16, max_steps=50_000_000, val_bits=16):
        self.code, self.r = code, [0] * 16
        self.mem = [0] * memory_words
        self.pc = 0
        self.steps = self.tokens = 0
        self.out = []
        self.max_steps = max_steps
        self.halted = False
        self.trail = []
        self.val_bits = val_bits

    def val_tokens(self, v):
        if self.val_bits == 16:
            return 1 if v < 65536 else 2
        return max(1, (v.bit_length() + 7) // 8)

    def wreg(self, rd, v, computed=True):
        v %= M
        if rd != 0:
            self.r[rd] = v
        if computed:
            self.tokens += self.val_tokens(v)

    def wmem(self, a, v):
        if not 0 <= a < len(self.mem):
            raise VMFail(f"memory address {a}")
        v %= M
        self.mem[a] = v
        self.tokens += self.val_tokens(v)

    def run(self):
        code, r = self.code, self.r
        while not self.halted:
            if self.steps >= self.max_steps:
                raise VMFail("step limit")
            if not 0 <= self.pc < len(code):
                raise VMFail(f"pc {self.pc}")
            op, *a = code[self.pc]
            self.trail.append(self.pc)  # recent pcs, for debugging
            if len(self.trail) > 64:
                del self.trail[:32]
            self.steps += 1
            self.tokens += 1  # INS(pc)
            nxt = self.pc + 1
            if op == "LI":
                self.wreg(a[0], a[1], computed=False)
            elif op == "MOV":
                self.wreg(a[0], r[a[1]])
            elif op in OPS3:
                x, y = r[a[1]], r[a[2]]
                if op in ("DIVU", "REMU") and y == 0:
                    raise VMFail("division by zero")
                v = {"ADD": x + y, "SUB": x - y, "MUL": x * y, "DIVU": x // y if y else 0,
                     "REMU": x % y if y else 0, "SLTU": int(x < y), "SLEU": int(x <= y),
                     "SEQ": int(x == y)}[op]
                self.wreg(a[0], v)
            elif op == "SHR1":
                self.wreg(a[0], r[a[1]] >> 1)
            elif op == "XOR":
                self.wreg(a[0], r[a[1]] ^ r[a[2]])
            elif op == "ROTN":  # rotate left by 4 * (rt mod 8) bits: a nibble permutation
                k = 4 * (r[a[2]] % 8)
                self.wreg(a[0], ((r[a[1]] << k) | (r[a[1]] >> (32 - k))) % M if k else r[a[1]])
            elif op == "SEQI":
                self.wreg(a[0], int(r[a[1]] == a[2] % M))
            elif op in ("BEQI", "BNEI"):
                take = (r[a[0]] == a[1] % M) == (op == "BEQI")
                nxt = a[2] if take else nxt
            elif op in ("BEQ", "BNE"):
                take = (r[a[0]] == r[a[1]]) == (op == "BEQ")
                nxt = a[2] if take else nxt
            elif op == "ADDI":
                self.wreg(a[0], r[a[1]] + a[2])
            elif op == "LD":
                addr = (r[a[1]] + a[2]) % M
                if not 0 <= addr < len(self.mem):
                    raise VMFail(f"load address {addr}")
                self.wreg(a[0], self.mem[addr])
            elif op == "ST":
                self.wmem((r[a[0]] + a[1]) % M, r[a[2]])
            elif op in ("BEQZ", "BNEZ"):
                take = (r[a[0]] == 0) == (op == "BEQZ")
                nxt = a[1] if take else nxt
            elif op == "JMP":
                nxt = a[0]
            elif op == "JAL":
                self.wreg(a[0], self.pc + 1, computed=False)
                nxt = a[1]
            elif op == "JR":
                nxt = r[a[0]]
            elif op == "OUT":
                self.out.append(r[a[0]])
                self.tokens += self.val_tokens(r[a[0]])
            elif op == "HALT":
                self.halted = True
            elif op == "FAIL":
                raise VMFail("firmware FAIL")
            else:
                raise VMFail(f"bad op {op}")
            self.pc = nxt
        return self.out


if __name__ == "__main__":
    # self-check: iterative fib(30) and a call/return through JAL/JR with a memory stack
    a = Asm()
    a.LI(1, 30)       # n
    a.LI(2, 0)        # x
    a.LI(3, 1)        # y
    a.label("loop")
    a.BEQZ(1, "done")
    a.ADD(4, 2, 3)
    a.MOV(2, 3)
    a.MOV(3, 4)
    a.ADDI(1, 1, M - 1)
    a.JMP("loop")
    a.label("done")
    a.OUT(2)
    a.LI(14, 1000)    # sp
    a.LI(5, 7)
    a.JAL(15, "sq")
    a.OUT(5)
    a.HALT()
    a.label("sq")     # r5 = r5 * r5, saving r15 on the stack
    a.ST(14, 0, 15)
    a.ADDI(14, 14, 1)
    a.MUL(5, 5, 5)
    a.ADDI(14, 14, M - 1)
    a.LD(15, 14, 0)
    a.JR(15)
    vm = VM(a.link())
    out = vm.run()
    assert out == [832040, 49], out
    print("bvm self-check ok:", out, "steps", vm.steps, "trace tokens", vm.tokens)
