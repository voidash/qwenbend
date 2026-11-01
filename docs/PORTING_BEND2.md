# How Bend2 was ported onto our CPU

QwenBend has no host compiler. Qwen's tokens go straight into the model, so **the whole Bend2
front end runs on the CPU inside the weights**: lexing, Bend2's static checks, compiling and
evaluating. This page explains how that front end was built and how each layer was checked
against the one before, ending at real Bend2.

## 1. Choose what to run: BendLite, a subset of Bend2

[Bend2](https://github.com/bendlang/bend) (checked at commit `9e8d7b7`) is a dependently typed
language with proofs. We kept the part a small CPU can run and a 0.8B model can write. The full
subset is in [BENDLITE.md](BENDLITE.md).

| kept | details |
|---|---|
| `def` with typed parameters | `U32`, `Nat`, `Bool`; `+x` marks a parameter usable more than once (affine use) |
| `match` on a parameter | `case 0n:` / `case 1n+p:` for Nat, `case True{}:` / `case False{}:` for Bool |
| arithmetic and comparison | `+ - * / % < <= > >=` in `( … : U32)` groups; `U32` wraps mod 2³² |
| builtins | `U32.from_nat`, `U32.is_eq`, `U32.min`, `U32.max` |
| recursion | structural only: it must shrink a `1n+p` binder |

Proofs and laws are dropped. Bend2 itself erases them before running: we confirmed that its
compiled output for a program with a law and its proof contains neither. So "only the code runs"
is Bend2's own semantics. Edge cases follow real Bend2, measured on probe programs: `x / 0 = 0`
and `x % 0 = x`, with no error.

## 2. A chain of four implementations, each checked against the last

Writing a type checker straight in assembly, and then in weights, would be impossible to debug. So
the same language was written four times, from most readable to the weights. Each step is checked
against the previous one.

```
real Bend2  ──►  bendlite_ref.py  ──►  bytecode.py  ──►  firmware2.py  ──►  engine_circuit.py
(the truth)      reference             compile pass +    the same, in        the CPU in the
                 interpreter +         threaded          BVM assembly        weights, running
                 Bend2's checks        evaluator (Python) (2,230 instr.)     firmware2
```

| step | what it is | checked against | result |
|---|---|---|---|
| `bendlite_ref.py` | tokenizes Qwen's tokens, applies Bend2's static checks (types incl. `5` vs `5n`, match position and coverage, affine use, structural termination), interprets | real Bend2 | 1,100/1,100 values equal; **0 unsound** (never accepts what Bend2 rejects) on every model-written program, 440 Bend2-labelled mutants and 69 rule probes |
| `bytecode.py` | one pass over the token tape: the same checks, then emits direct-threaded code, which a small evaluator runs | `bendlite_ref.py` | 2,389/2,389 programs agree |
| `firmware2.py` | the lexer, compile pass and evaluator as BVM assembly, run by `bvm.VM` | `bytecode.py` | 0 code mismatches, 0 result mismatches on 2,389 programs |
| `engine_circuit.py` | the CPU as transformer weights, executing firmware2 one instruction per token | `bvm.VM` / `trace.TraceMachine` | every position exact (39,913 in the evaluator, 3,230 bit for bit in torch); end-to-end traces equal token for token |

## 3. The CPU (BVM)

- 32-bit words, 16 registers (`r0` reads 0), and 8,192 words of memory.
- **One rule shapes the ISA: an instruction makes at most one state write.** A value computed
  inside the network reaches later steps only through the token the model emits. So each
  instruction is one token, `INS(pc)`, followed by its written value as 1–4 byte tokens.

```
LI MOV ADD ADDI SUB SLTU SEQ SEQI XOR ROTN SHR1      ALU (no multiplier, no divider)
LD ST                                                memory
BEQZ BNEZ BEQ BNE BEQI BNEI JMP JAL JR               control
OUT HALT FAIL                                        output / stop
```

- There is no hardware multiply or divide (they would cost far too many weights). `MUL`, `DIV`
  and `MOD` are firmware routines: shift-and-add, and restoring division with a fast path for
  small quotients. They are self-tested on 3,196 operand pairs.

## 4. Lexing Qwen's tokens without a host

Qwen writes BPE tokens like `def`, ` run`, `(+`, `22n`, not characters. Each of the 122,274 Qwen
tokens that can occur in BendLite gets a **ROM entry baked into its embedding row**: the address
of a lexer handler (one of 21 segment patterns) plus up to three 32-bit payload words.
- Names are hashed to 32-bit keys with XOR / rotate.
- Digits accumulate into numbers, with overflow flagged.
- Indentation is counted from spaces after newlines.

While Qwen writes, the engine's capture layers store each token's four words in memory. At
`<|im_end|>` the firmware lexer walks them and builds a token tape for the compile pass.

## 5. The compile pass and the evaluator

The compile pass reads the tape once and **applies Bend2's static rules**. A violation jumps to
`FAIL`, so an unchecked program never runs. It emits direct-threaded code:

```
LIT v   VAR s   ADD SUB MUL DIV MOD LT LE GT GE ISEQ MIN MAX
CALL entry n   ENTER k   RET   OUT
AZERO s L   ASUCC s b L   ABOOL s x L   PRED s b        (match arms)
```

The evaluator is one firmware handler per code word: `op_VAR`, `op_CALL`, and so on. It runs
with a return stack and an operand stack in memory.

**Memory map:**

| range | holds |
|---|---|
| 512–1535 | the token tape (kinds, values) |
| 4608–5631 | compiled code |
| 5632–6143 | return stack: 256 nested calls |
| 6144–8191 | Qwen's captured tokens; reused as frames and operands once lexing is done |

**Limits:**
- 64 defs, 256 parameters, 128 variables;
- 512 code words, a 509-entry tape, and call depth 256;
- programs under 192 tokens (the capture index).

## 6. What a run costs

For a typical program the firmware spends most of its instructions **reading and checking** the
source, and the actual evaluation is small. "Sum 22 copies of 35" takes 7,869 trace tokens in
three phases:
1. lexing: about 1,350 instructions;
2. checking and compiling: about 1,200;
3. evaluating: the rest.

Across the dataset the median is about 7.5k tokens and p90 about 16k.
