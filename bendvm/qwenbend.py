"""QwenBend from the command line: Qwen writes a BendLite program for your task, then the same model
type-checks it (Bend2's rules) and runs it inside its own forward passes; the answer is read from
the model's own trace and checked against the reference interpreter.

Run: python bendvm/qwenbend.py "Sum 22 copies of 35."
Needs torch + transformers with Qwen3.5 support and an Apple-silicon Mac (MPS) or a CUDA GPU with
about 9 GB free; Qwen3.5-0.8B and the LoRA adapter are downloaded from the Hugging Face Hub on
first use (see paths.py to point at local copies).
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bendlite_ref import BendLiteError, run_source  # noqa: E402
from engine_circuit import FIRMWARE, MAX_N  # noqa: E402
from lexer_fw import Stored  # noqa: E402
from merge import build_merged, run  # noqa: E402
from run_engine import result  # noqa: E402
from trace import trace  # noqa: E402

SYSTEM = (
    "You write programs in BendLite, a subset of the Bend2 language. Rules: start with `import Base`; "
    "define each def before it is used; `def main() -> U32:` returns the answer. Types: U32 numbers (42), "
    "Nat counters (5n), Bool (True{}, False{}). A body is an expression or a `match` on a parameter: "
    "`case 0n:` / `case 1n+p:` for Nat, `case True{}:` / `case False{}:` for Bool. Arithmetic and "
    "comparisons go in `( ... : U32)` with spaces around operators: + - * / % < <= > >=. Equality is "
    "U32.is_eq(a, b); U32.from_nat(n) turns a Nat into a U32. Recursion must shrink the first changing "
    "argument (use a Nat fuel counter for loops). Use `+name: Type` for parameters used more than once. "
    "Two-space indentation. Reply with the program only.")


def main():
    if len(sys.argv) < 2:
        raise SystemExit('usage: python bendvm/qwenbend.py "Sum 22 copies of 35."')
    task = " ".join(sys.argv[1:])
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else None
    if device is None:
        raise SystemExit("QwenBend needs an MPS or CUDA device (the model is about 8 GB in bf16)")
    m, tok, info = build_merged(device)
    print(f"QwenBend: {info['params'] / 1e9:.2f}B params (nonzero: Qwen {info['nonzero']['qwen'] / 1e9:.2f}B, "
          f"engine {info['nonzero']['engine'] / 1e6:.0f}M), built in {info['build_s']:.0f} s on {device}")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Write a BendLite program. Task: {task}"}]
    prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    pids = tok.encode(prompt, add_special_tokens=False)
    nn = tok.encode("\n\n", add_special_tokens=False)
    if pids[-2:] != [info["think"]] + nn:
        raise RuntimeError(f"the chat template no longer ends with '</think>\\n\\n': {pids[-4:]}")
    t0 = time.time()
    qwen, engine, bad = run(m, info, pids, device=device)
    dt = time.time() - t0
    prog_ids = qwen[:-1]
    src = tok.decode(prog_ids)
    print("\nQwen wrote:\n" + src.strip() + "\n")
    if len(nn + prog_ids) >= MAX_N or not all(i in info["words"] for i in prog_ids):
        print("note: the program is outside what the engine captures (too long or a token outside BendLite)")
    got = result(engine)
    try:
        want = run_source(src)[0]
    except BendLiteError as e:
        want = f"ERR ({e})"
    same = engine == trace(FIRMWARE, Stored(nn + prog_ids, info["words"]))
    print(f"the same model then emitted {len(engine):,} engine tokens ({len(engine) / dt:.0f} tok/s overall, "
          f"{dt:.0f} s)")
    print(f"answer, computed by the model: {got}")
    print(f"reference interpreter: {want}; trace == reference token for token: {same}; "
          f"exit-layer violations: {len(bad)}")


if __name__ == "__main__":
    main()
