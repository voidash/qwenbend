# BendVM: our own engine for a Bend2 subset inside Qwen (design v0, Phase 0)

No Qwenputer code is used.
- **Ideas we learn from:** published Qwenputer docs (MIT), credited: DeltaNet heads as RAM, and
  "one memory write per token".
- **A limit we found ourselves** (an earlier, unpublished experiment).

## Phase 0 checks (all done 2026-11-01, evidence in PROGRESS.md)

1. **Lowering hook.** Both of Bend2's backends emit from the same erased per-def form:
   `fun_of(k) → {live params, body h: HTerm}`, walked by `js_func` / `js_expr` / `js_match`
   (`bend/bend2/comp.ts:2909-3104`, revision 9e8d7b7).
   - Our compiler is a third backend over that form, emitting BendVM code.
   - What the JS output looks like for `sum`: `if (n == 0) return 0; else { p = n-1; x = sum(p);
     return (1+x)>>>0 }`.
2. **Law erasure.** `examples/law_add_zero.bend`:
   - Bend2's checker: "ALL PROOFS CHECK".
   - Running it prints `10`.
   - The emitted program has 0 occurrences of `add_zero` (the law and its proof def are gone),
     and `Nat.add(10n, 0n)` was lowered to `10 + 0`.
   - **So "proven by Bend2, then only the code runs" holds.**
3. **Step counts.** The lowered code is first-order and small (`sum`: one compare, one subtract,
   one call and one add per level). Exact counts come from the Phase 1 reference VM, which
   records them for every example.

## What runs (the Bend2 subset, tier 1)

| Kept | Lowered to |
|---|---|
| `def` over `U32`, `Nat`, `Bool`, user `type … is Data`, tuples, `List<U32>` | functions with N params, 1 return value |
| `match` on `0n` / `1n+p`, `True{}` / `False{}`, constructors | branch on a tag (constructors: tag + fields on the heap) |
| `+ - * / % < > ==` on U32 (`>>> 0` wraparound), on Nat (≤ 2^48 per Bend2) | ALU ops |
| recursion (structural; tail calls become loops in Bend2's lowering) | call / return with an explicit stack in memory |
| laws, proofs, types, erased args (`-x`) | **nothing**: erased by Bend2 |
| `IO.print(U32.show(x))` in `main` | the engine prints x's digits as text tokens; no IO otherwise |

Out of v1: `!` parallel calls (run sequentially; the code is pure, so results are equal), F32,
String/Char, closures, arrays, real IO, and `@unsafe` recursion.

## The machine (BendVM)

The constraint that shapes everything: a value computed above a memory layer reaches lower
layers only through the emitted token. So **one token = one state write**, as an
(address, value) pair split across tokens if needed.

- **State:** registers (pc, sp, hp, a small accumulator file), a stack region, a heap region
  (constructor nodes: tag + fields, fixed width), and a code ROM. All of these are DeltaNet
  memory heads. ROM is written once at boot, like a boot record.
- **Instructions (sketch, finalized in Phase 1 from what the lowering actually needs):**
  `LIT r,k` · `MOV r,s` · `ADD/SUB/MUL/DIV/MOD/LT/EQ r,s,t` (U32 and Nat) · `JZ r,L` / `JMP L` ·
  `CALL f` / `RET` (stack in memory) · `NEW tag,n` / `GET r,s,i` / `TAG r,s` (heap) ·
  `FREE s` (Bend2 is affine: a match frees what it opens) · `OUT r` · `HALT`.
- **Trace tokens:** our own vocabulary of chip tokens: register-write, memory-write
  (address, value) and step-end. A step is all the writes one instruction makes; the next
  instruction is fetched from state that is fully committed.
- **Reference VM:** Python, exact, the oracle for everything after it. It counts steps and
  tokens per example.

## Toolchain (all ours)

1. `bend2-eng` backend: a TypeScript emitter added to our local Bend2 clone next to `js_def`.
   Input: `fun_of(k)` bodies. Output: BendVM assembly. Kept as a patch file.
2. Assembler and Python reference VM. Phase 1 is done when the outputs on every example and on
   fuzzed inputs equal real `bend`.
3. Circuit compiler (Phase 2): BendVM as exact transformer sublayers (MLP gadgets in exact
   bf16 ranges, one-hot DeltaNet memory, norm ballast, muting Qwen on machine tokens). **This
   is the large, risky part.**
4. Merge into Qwen3.5-0.8B (Phase 3).

## Test discipline (lessons from the verify track)

- Oracle: real `bend` output.
- Fuzzing: random inputs to every example, plus generated random tier-1 programs.
- Adversarial programs: deep recursion, max U32 wraparound, Nat edges, big lists vs memory
  limits.
- Every circuit term exercised in any planning corpus (the fix-2 lesson).

## Examples (to be written in Phase 1, each run on real `bend` first)

1. `sum(n)`
2. `fib(n)`
3. list `reverse` with the law `reverse(reverse(xs)) == xs` and its proof
4. stretch: the README game law, if its runtime part fits

## Step 5b design: capture to a lexeme tape, then a firmware tape-walker (2026-11-01)

**Measured** (`token_survey.py` → `token_survey.tsv`): the 1,080 dataset programs use **151
distinct Qwen tokens**. Tokenization is regular but fused:
- prefix punctuation + name: `(+`, `(p`, `(r`, `(c`;
- `+p`, `+r`;
- closing pairs: `))`, `),`;
- names split across tokens: `.is` + `_eq`, `.from` + `_nat`, and `U32` as ` U` + `3` + `2`;
- digits one per token;
- indentation as separate whitespace tokens (`\n` then `' '`, `'   '`, `'     '` …, plus a
  leading-space keyword like `' match'`).

**Why not lex in firmware:** a firmware lexer working character by character would cost
roughly 20k+ machine instructions (tens of thousands of trace tokens) per program, which is
hours on the Mac. Instead:

1. **Capture (hardware, while Qwen writes, no trace tokens).** A ROM indexed by token id gives
   each token its lexeme decomposition, a short fixed sequence such as
   `[PUNCT '(' , NAMECHARS 'p']`, or `INVALID` for any token with characters outside
   BendLite.
   - The capture state machine appends lexemes to a **tape**:
     - names are accumulated across tokens into a running hash + length, so `count` `_m`
       `ultiples` is one name;
     - numbers are accumulated digit by digit;
     - indentation becomes the column count of the line.
   - The ROM covers the whole vocabulary, so model-written identifiers that aren't in the
     dataset work too.
2. **Firmware (machine instructions, traced).** A pre-pass indexes defs and parameters. Then a
   recursive-descent **tape walker** evaluates `main` with an explicit stack in memory.
   - Values are U32/Nat (32-bit) and Bool.
   - Target: about 3–6k trace tokens per median program (92 source tokens, 11 def calls).
     To be measured.
3. **Reference VM:** Python, step-exact. It must equal `bendlite_ref` on all 1,100 programs and
   on every model-written program.

Sub-steps:
- **5b-1:** capture ROM generator plus a Python capture model, tape-equal to `bendlite_ref`'s
  lexer on all programs.
- **5b-2:** BVM ISA and reference VM with the trace format.
- **5b-3:** firmware tape-walker.
- **5b-4:** equivalence and per-program token counts.

## Trace token width (added with S1a, 2026-11-01)

Since S1a, a written value takes 1 token when it is below 2^16 and 2 tokens otherwise. The
trace must stay self-describing, so the write-marker token encodes the width: `REG16_r` /
`REG32_r` (32 register markers) and `MEM16` / `MEM32`. A decoder (C3) therefore knows how many
value tokens follow from the marker alone. The token counts in `bvm.VM` already assume this.

## Phase 2 (C1): circuit facts, budget, and the trace format the circuit can build (2026-11-01)

**Facts read from the code** (transformers 5.19.0 `models/qwen3_5/modeling_qwen3_5.py` in
torch 2.14.1 MPS; config of Qwen/Qwen3.5-0.8B):
- **Shape.** hidden 1024; 24 layers: 18 Gated DeltaNet + 6 full attention (every 4th).
  - DeltaNet: 16 key heads / 16 value heads × 128.
  - Attention: 8 heads / 2 KV, head_dim 256, sigmoid output gate, partial RoPE 0.25.
  - SwiGLU MLP, intermediate 3584.
  - **tied embeddings** (the output head is the embedding matrix). vocab 248,320.
- **Gated DeltaNet decode (seq_len 1):**
  - `in_proj_qkv` (bf16) → **depthwise causal conv1d, kernel 4, then SiLU** on q, k and v.
    Each channel mixes positions t-3..t with its own 4 weights.
  - q and k are **l2-normalised in fp32** (eps 1e-6); q is then × 128^-0.5.
  - β = sigmoid(`in_proj_b`) in bf16.
  - Decay = exp(−exp(A_log) · softplus(`in_proj_a` + dt_bias)) in fp32, so A_log ≪ 0 gives
    exactly 1.
  - State update, in **fp32**: S ← S·decay; S += k ⊗ β(v − Sᵀk); the output is Sᵀq after the
    write.
  - Per head, the output goes through **RMSNormGated** (fp32, eps 1e-6) × weight × SiLU(z),
    with z = `in_proj_z`; then `out_proj`.
  - So a read value is **renormalised per head**, and z = 0 gates a head to exactly 0.
- **RMSNorm:** x/rms(x) · (1 + w) in fp32, cast to bf16. The residual stream is bf16.
- **The verify track's merge** (the upstream method, MIT, ideas citable) **widened** the model:
  hidden 1024 → 1664, plus 14 chip layers.
  - The engine therefore lives in extra residual dims and extra layers of a model with the same
    architecture. One forward pass, loaded by the stock HF class.
  - Per-layer counts (heads, intermediate size) are global config values, so Qwen's layers get
    zero padding.

**Consequences for the engine:**
1. **A word in memory = one DeltaNet row.** A 32-bit word as 8 one-hot nibbles fills the
   128-dim value exactly.
   - It has a **constant norm**, so the per-head RMSNormGated rescales every read by the same
     known factor.
   - Readouts are **margin-based** (a nibble digit is the argmax / a threshold), never raw
     floats.
   - An unwritten row reads exactly 0.
   - One head holds 128 words. A memory of W words needs W/128 heads: 8,192 words = 64 heads,
     spread over engine DeltaNet layers.
   - Head select: the write uses per-head β from address high bits; the read uses per-head z.
2. **State can't be advanced without a token.**
   - A value computed in layers above ℓ at position t reaches layers ≤ ℓ at t+1 only through
     the emitted token. DeltaNet and attention at layer ℓ do see what layers < ℓ computed at
     earlier positions, but nothing from above.
   - So **"PC + 1 for free" (the S1a accounting) is not buildable.**
   - **Decision: every executed instruction emits `INS(pc)`** (firmware about 1.2k
     instructions → about 1.2k token ids). The instruction is then *named* by its token, and
     the low layers know the opcode, rd, rs and imm from the embedding.
   - It is followed by value tokens **only when the written value is computed**: MOV, ALU ops,
     ADDI, LD, SEQI, ST, OUT. LI and JAL write values fixed by pc, so they need none.
   - Branches are 1 token; the taken/not-taken result is the next `INS`. A store needs no
     address token (the address is r[rs] + imm, recomputable from state).
   - The trace becomes fully self-describing: it is a proof of which instruction ran.
3. **A latch head** (one DeltaNet row, overwritten at each `INS` position) holds the current
   instruction's write target and phase. The register and memory writes at the value-token
   position read the latch rather than depending on conv offsets, because a 1- or 2-token value
   moves the INS from t-1 to t-2.
4. **Mode:** a residual bit. The engine is dormant while Qwen writes (β = 0, z = 0 on engine
   heads; capture only), active from the first `INS`.

**Open (C3):** capture in-circuit. The name keys must become an XOR/nibble-rotate hash (linear
over GF(2), computable per token from a ROM entry `(len, H(s))`) instead of the polynomial hash.
`capture.py` / `key_of` change accordingly. Equality-only use, so any injective-enough key works.

**Open (C2):** MUL / DIVU / REMU in-circuit vs firmware routines (they matter in the digit
families' loops).

## C2 circuit plan: one BVM step per position (2026-11-01)

Every position runs the whole step. INS(pc) positions compute the instruction; MORE positions
recompute the same value (state is unchanged inside a record) to emit the next byte; LAST
positions commit the write. `trace.TraceMachine` is the oracle for every intermediate field.

**Energy rule (generalises C1's).** The residual energy at every norm must be a power of 4
(× hidden) **determined by the token type alone**:
- every field a sublayer adds has data-independent energy (one-hot words: 8 hot dims of a fixed
  magnitude), or is gated to exactly 0 (SiLU(z = 0));
- per-type ballast closes the gap. The always-one-hot rule of C1 is the special case.

**Layers** (engine-only model; Qwen's dims are C4):
- **E (embedding, a free ROM).**
  - Token kind (INS / MORE / LAST / END).
  - For INS(pc), FIRMWARE[pc]'s fields:
    - op class, rd / rs / rt (one-hot 16 each);
    - imm (8 one-hot nibbles), pc + 1 and the branch target (3 nibbles each);
    - CONST, LI's value or JAL's pc + 1 (8 nibbles).
  - For MORE / LAST, the byte (2 nibbles).
- **A (DeltaNet).**
  - **Latch heads:** β = 1 at INS; they write all instruction fields to row 0 and read row 0 at
    every position, so MORE / LAST see their record's instruction.
  - **Assemble head:** its conv taps 0..3 turn the MORE run + LAST into the 32-bit word. It is
    written and read at row 0 at LAST only, and z gates it to exactly 0 elsewhere. The output
    field is WORD.
  - Why not write registers straight from the conv: at an INS position, taps 2–3 may still see a
    previous record's MORE bytes, so LI / JAL constants can't share the byte channels.
- **R (DeltaNet).** The register file: 16 rows. Two identical heads so rs and rt can both be read.
  - Write key = latched rd, v = WORD + CONST (one of the two is 0).
  - β = (LAST ∧ op writes rd) ∨ (INS ∧ op ∈ {LI, JAL}), never for r0.
- **D (MLPs).** Address and ALU on nibbles:
  - r[rs] + imm: carry via generate / propagate prefix units, an AND of one-hot features per
    unit, exact;
  - ADD, SUB, SLTU / SLEU / SEQ / SEQI, SHR1 (2-nibble tables).
- **M (DeltaNet, 64 heads).** Memory; head = addr >> 7, row = addr & 127.
  - Write v = WORD at LAST of ST; β only on the addressed head.
  - Read for LD, with z selecting the head. The result is MEMVAL.
- **S (MLPs).** Select and emit:
  - choose the value by op (ALU / MEMVAL / r[rs] / r[rt]) and the byte to emit (the first
    non-zero byte at INS; the next byte by the window's run length at MORE);
  - choose the next pc (branch compare / JR r[rs] / target / pc + 1);
  - **cancel every other field**, so the final residual holds only the EMIT field plus a
    ballast in dims where no engine token's embedding row is non-zero. Then
    logit(t) = E[t] · x is exactly the emit-field match score.
- The harness for all of C2 is teacher-forced: feed each program's true trace and require the
  circuit's top next token = the trace's next token at every position, with intermediate
  fields checked against TraceMachine (registers, memory, latch).

## C3 design: capture = an in-circuit token store + a firmware lexer (2026-11-01)

**Why not lex in the circuit:**
- The tape index needs a counter. A counter is a nonlinear recurrence, and a high layer can't
  feed a low layer at the next position without a token.
- Names span up to 4 tokens (the conv window), but numbers span one token per digit and need
  ×10 accumulation. A DeltaNet can't multiply its state.

**Counting** (upstream Qwenputer's idea, "Counting without positions", docs/q35-store.md, MIT):
- A row halves at every token: β = sigmoid(0) = ½ exactly, v = 0.
- Adapted to our exactness rules:
  - the row carries 2 reference dims with v = ref, a fixed point of the halving, so after
    RMSNormGated the counter dim reads as an exact power of 2;
  - its energy is far below fp32 rounding, so the residual norms stay exact.
- bf16's exponent range limits one counter to about 110 steps. Measured program lengths:
  median 92 tokens, p99 178, max 371.
- **So 3 cascaded counters.** Counter B (a higher layer) halves only when counter A's
  thresholded "n ≥ 100" flag is set at the same position, which is legal because it flows
  upward. C does the same over B.
- Threshold units then give a one-hot token index n < 320.

**Pieces** (measured over the 122,849 BendLite-only vocabulary ids / program tokens):
- Program tokens are 1 piece (101,143) or 2 (54,561), rarely 3 (170, e.g. `' (('`).
- Digits are always 1 per token (Qwen's tokenizer).
- **Two ROM words per token:**
  - **A:** the handler pc of the firmware lexer for this token's class, plus a small payload
    (length mod 8, punctuation codes, digit, boundary flags);
  - **B:** the token's name-part hash H(s).
- Capture writes A and B to two raw-token regions, two memory heads in different memory
  layers, at index n. No trace tokens.

**Name keys** become an XOR / nibble-rotate hash, key ← rotl(key, 4·len(s)) ⊕ H(s).
- It is consistent across any tokenisation of the same name, and computable per piece.
- It costs two cheap ALU instructions: XOR (nibble tables) and ROTN (rotate by 4·k bits, a
  nibble permutation).
- `capture.py` / `key_of` switch to it.

**Firmware lexer** (boot, before the compile pass):
- For each stored token: `LD A; JR` to its handler (threaded dispatch).
- The handlers merge name parts, accumulate digits (×10 = 8x + 2x by adds), count
  indentation, handle the two-char operators, and write the same tape entries capture.py
  produces.
- Estimated +25% trace tokens.
- Oracle: the lexer's tape == `capture.py`'s tape on every program.

**Program region:** the counters reset at the first program token. The end (the engine trigger)
writes an END-class word. The trigger itself is decided in C4.

## C3c design: one-hot(n) by coarse / fine β-step flags (2026-11-01, advisor-reviewed)

The only exact two-sided saturation is β = sigmoid(b): b ≥ 6.25 → exactly 1, ≤ −87.5 → exactly 0.
It is one bit per head, so n = 16b + r is decomposed to keep the head count small:
1. **Active flag:** one row; β = 1 at CAPSTART (on) and at EOF (off). Every capture β includes
   −128·(1 − active).
2. **Counters:**
   - A1 halves at every program token (β = ½ exactly, 2 reference dims as fixed points; reset with
     β = 1 at CAPSTART).
   - A2 holds until the monotone flag "n ≥ 96", then halves (b = 128·flag96 − 128, plus 256 at
     CAPSTART).
3. **Block flags** "n ≥ 16b", b = 1..11: one β-step head each, on A1 (b ≤ 6) or A2 (b ≥ 7). Each
   writes a fixed pattern into row 0 when true. Stale-true is correct (n only grows within a
   program); CAPSTART clears them.
4. **Block one-hot** = flag_b ∧ ¬flag_{b+1} (AND units).
5. **Fine value** x′ = x · 2^(16b): a block-gated AND unit (gate exactly 128, up a power of 2), so
   it is exact.
6. **Fine flags** "r ≥ k", k = 1..15: one β-step head each on x′, writing into **row = block**
   (key = one-hot(block)) and read with q = one-hot(block). They are fresh per block and monotone
   within it.
7. **one-hot(n)** = block ∧ fine (192 AND units) → ROW / HEAD for the 4 token-store writes, as in
   the ST path.

**Limits and checks:**
- Programs are capped at **n < 192 tokens** (p99 178). `MAX_TOK` is 191 in Python and circuit.
- Counter readouts are tiny residual values. Absorption needs them ≤ 2⁻⁵; this is asserted in the
  evaluator, not assumed.
- About 5 sequential stages precede the memory writes. Memory is re-indexed after the index MLP,
  and the full C2 harness is re-run.

**Rejected:**
- the ruler / attention argmax (q_norm RMS and bf16 make it fragile; MPS SDPA unverified);
- host-side capture (it voids the claim; any stopgap must be stated in PROGRESS).

**Required later:** the lexer's +62% tokens must be optimised before the end-to-end claim's cost
numbers.

## C4 plan: merging the engine into Qwen3.5-0.8B (2026-11-01)

**Facts:**
- The generation prompt (eval.py, `enable_thinking=False`) ends with
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- The fine-tuned model then writes the program and ends with `<|im_end|>`.

**Token roles:**
- **CAPSTART = `</think>`** (once per assistant turn).
- **EOF = `<|im_end|>`.** At its position the engine emits INS(0); Qwen does not get the turn
  back.
- The `\n\n` after `</think>` is captured, so **capture must not emit NL entries while the tape
  is empty** (leading whitespace). This is a semantic change in capture.py, capture2 and the
  firmware lexer; equal for programs without leading whitespace.

**Model shape** (one HF Qwen3_5 model, stock classes):
- hidden = 1024 (Qwen) + H_e (engine); layers = 10 engine layers **first**, then an exit MLP,
  then Qwen's 24.
- heads 16 and MLP width 3,584 are Qwen's own; Qwen's matrices are zero-padded on engine dims.
- vocab = Qwen's 248,320 + the engine tokens.
- **Tied embeddings stay tied:**
  - an engine token's row = its input fields + its emit pattern;
  - layer 0 cancels the input emit pattern (units on the token's own kind / digits);
  - after the exit only EMIT survives in engine dims, so logit(engine t) = its emit match.

**Exactness with Qwen's dims present (engine layers first):**
- At text tokens the residual holds Qwen's embedding (arbitrary energy). The engine ballast is
  made large (total H·4^K with K about 10, 2³²+), so Qwen's |e|² (≲ 2⁸) is **absorbed** by
  every engine norm, as the counters already are.
- Engine projections have zero weight on Qwen dims, so Qwen dims pass through unchanged.
- **Exit MLP:** zeroes every engine dim exactly (one-hot fields by AND units, the ballast and
  constants by ONE-gated units, the tiny counters by gated pass-through with up = −value).
  - It keeps only EMIT and a `tau` marker (text positions: engine rows get a large negative
    logit).
  - Qwen's layers then see their normal input, up to the norm's mean over more dims. Qwen's norm
    weights are rescaled by √(H / 1024): near-identical, not bit-identical.
- **Logits:** engine positions → EMIT with large final-norm gains, so the engine wins; text
  positions → tau keeps engine rows far below Qwen's.

**Numerics risks to measure:**
- the prompt must be fed in **decode mode** (no chunked prefill through the engine heads);
- Qwen's state decays while the engine runs (it doesn't get the turn back, so acceptable for
  the demo);
- model size: about (24 + 10) layers × hidden (1024 + H_e) plus the embedding, about 8–10 GB in
  bf16 on a 24 GB Mac. **Measure before committing; shrink H_e (overlay) if needed.**

**Steps:**
- **C4a:** fuse the iteration-400 LoRA into HF weights; check torch generation ≈ MLX on test
  prompts.
- **C4b:** the leading-whitespace capture rule; CAPSTART / EOF ids.
- **C4c:** the merge builder (large ballast, exit MLP, tied rows, norm rescale) + its checks:
  - the engine dims of the merged model == the engine-only model;
  - Qwen's text logits ≈ the fused model's.
- **C4d:** end to end on MPS: a prompt → Qwen writes the program → `<|im_end|>` → the engine
  runs → result. Measured: correctness vs the reference, tokens/s.

### C4 as built (supersedes the plan above where they differ)

**Hidden and norms.**
- hidden **4608**, not 1024 + 4096: the engine's 3,415 field dims plus Qwen's 1,024 did not fit in
  4096.
- Engine energy H·4^K is unchanged in form. Qwen's norms take (1+w)/√4.5 in fp32, so they are
  near-identical, not bit-identical (the measured generations matched MLX).

**Absorbing Qwen's embedding.**
- The engine norms do not absorb Qwen's embedding exactly. They round it away: the relative rsqrt
  shift is < 2⁻¹⁹, and every engine norm output is an exact bf16 value.

**The exit layer.**
- It uses the linear SiLU region, one unit per engine dim: gate 32 on ONE (or on QTOK for EMIT),
  up = x_d, down −1/32. That is 3,432 of 3,584 units.
- **Why it is exact:** 32·x_d and −x_d are exact in bf16, so each engine dim is cleared exactly.
- No per-field AND gadgets are needed.

**Vocab and lm_head.**
- The lm_head is untied (+2.3 GB) rather than overlaying emit patterns on the input rows.
- Engine tokens sit at 248,320 + t.

**Leading whitespace.**
- The leading-whitespace issue is solved in the lexer: newlines before the first tape entry are
  ignored.

**Isolating the prompt.**
- The prompt is isolated by clearing all engine heads at CAPSTART.
- Measured: 0 nonzero engine dims after the exit layer at every prompt position, so the engine's
  pre-CAPSTART activity never reaches Qwen.
