"""Evaluate a model (optionally with a LoRA adapter) on writing BendLite programs.

For each test item: greedy-generate with the item's system + user messages (MLX), strip code
fences, then run real Bend2 on the program: `--check-only` (valid?) and a run (correct output =
the item's expected value). Writes one JSON line per item and prints a summary, overall and by
family (held-out families marked).

Run: python bendtune/eval.py --model Qwen/Qwen3.5-0.8B \
       [--adapter bendtune/adapters] --split test --out bendtune/logs/baseline.jsonl
"""

import argparse
import json
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bendvm"))
from paths import bend2_main  # noqa: E402

HERE = Path(__file__).resolve().parent
BEND = bend2_main()
HELD_OUT = {"digit_count", "sum_multiples", "reverse_digits"}


def extract(text: str) -> str:
    m = re.search(r"```(?:[a-zA-Z0-9]*)\n(.*?)```", text, re.S)
    body = m[1] if m else text
    body = body.split("<|im_end|>")[0]
    return body.strip() + "\n"


def bend_eval(src: str, want: int, path: Path) -> dict:
    path.write_text(src)
    try:
        c = subprocess.run(["bun", str(BEND), str(path), "--check-only"], capture_output=True,
                           text=True, timeout=120)
        valid = "ALL PROOFS CHECK" in c.stdout + c.stderr
        out = ""
        if valid:
            r = subprocess.run(["bun", str(BEND), str(path)], capture_output=True, text=True,
                               timeout=120)
            lines = (r.stdout + r.stderr).strip().splitlines()
            out = lines[-1] if lines else ""
        err = "" if valid else (c.stdout + c.stderr).strip()[-300:]
    except subprocess.TimeoutExpired:
        valid, out, err = False, "", "timeout"
    return {"valid": valid, "output": out, "correct": valid and out == str(want), "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in open(HERE / "data" / f"{args.split}.jsonl")]
    rows = rows[: args.limit] if args.limit else rows
    model, tok = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory() as td, ThreadPoolExecutor(4) as pool, \
            open(args.out, "w") as f:
        pending = []
        for i, row in enumerate(rows):
            msgs = row["messages"][:2]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=False)
            t = time.time()
            text = generate(model, tok, prompt, max_tokens=args.max_tokens, sampler=sampler)
            src = extract(text)
            meta = row["meta"]
            fut = pool.submit(bend_eval, src, meta["expected"], Path(td) / f"g{i}.bend")
            pending.append((i, row, text, src, time.time() - t, fut))
        for i, row, text, src, secs, fut in pending:
            res = fut.result()
            rec = {"i": i, "family": row["meta"]["family"], "held_out": row["meta"]["family"] in HELD_OUT,
                   "task": row["messages"][1]["content"], "expected": row["meta"]["expected"],
                   "program": src, "raw": text[-1500:], "gen_seconds": round(secs, 1), **res}
            results.append(rec)
            f.write(json.dumps(rec) + "\n")

    def pct(rs, k):
        return f"{sum(r[k] for r in rs)}/{len(rs)} ({100 * sum(r[k] for r in rs) / max(1, len(rs)):.0f}%)"

    print("ALL      valid", pct(results, "valid"), " correct", pct(results, "correct"))
    for name, sel in (("in-dist", [r for r in results if not r["held_out"]]),
                      ("held-out", [r for r in results if r["held_out"]])):
        print(f"{name:<8} valid", pct(sel, "valid"), " correct", pct(sel, "correct"))
    fams = sorted({r["family"] for r in results})
    for fam in fams:
        sel = [r for r in results if r["family"] == fam]
        print(f"  {fam:<16}{'*' if fam in HELD_OUT else ' '} valid {pct(sel, 'valid'):<14} correct {pct(sel, 'correct')}")


if __name__ == "__main__":
    main()
