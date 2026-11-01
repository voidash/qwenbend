"""Baseline: Qwen3.5-0.8B (no adapter) answers each test task directly, as a number, with or without
thinking. Same 200 items and expected values as eval.py. Greedy (MLX).

Run: python bendtune/bench_direct.py --model Qwen/Qwen3.5-0.8B \
       --out bendtune/logs/direct_nothink.jsonl [--think --max-tokens 4096]
"""

import argparse
import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

HERE = Path(__file__).resolve().parent
HELD_OUT = {"digit_count", "sum_multiples", "reverse_digits"}
SYSTEM = "Solve the task. End your reply with the final answer on its own line as: ANSWER: <integer>"


def final_number(text):
    """The integer after the last 'ANSWER:', else the last integer in the reply (None if none)."""
    body = text.split("</think>")[-1]
    m = re.findall(r"ANSWER:\s*(-?\d[\d,]*)", body)
    if not m:
        m = re.findall(r"-?\d[\d,]*", body)
    return int(m[-1].replace(",", "")) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [json.loads(x) for x in open(HERE / "data" / "test.jsonl")]
    model, tok = load(args.model)
    sampler = make_sampler(temp=0.0)
    results = []
    with open(args.out, "w") as f:
        for i, row in enumerate(rows):
            task = row["messages"][1]["content"].replace("Write a BendLite program. Task: ", "")
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                             enable_thinking=args.think)
            t = time.time()
            text = generate(model, tok, prompt, max_tokens=args.max_tokens, sampler=sampler)
            got = final_number(text)
            want = row["meta"]["expected"]
            rec = {"i": i, "family": row["meta"]["family"], "held_out": row["meta"]["family"] in HELD_OUT,
                   "task": task, "expected": want, "answer": got, "correct": got == want,
                   "seconds": round(time.time() - t, 2), "reply_tail": text[-400:]}
            results.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()

    def pct(rs):
        return f"{sum(r['correct'] for r in rs)}/{len(rs)} ({100 * sum(r['correct'] for r in rs) / max(1, len(rs)):.0f}%)"

    mode = "thinking" if args.think else "no thinking"
    print(f"direct answers ({mode}): correct {pct(results)}; in-dist {pct([r for r in results if not r['held_out']])}; "
          f"held-out {pct([r for r in results if r['held_out']])}; median {sorted(r['seconds'] for r in results)[100]} s/item")
    for fam in sorted({r["family"] for r in results}):
        print(f"  {fam:<16} {pct([r for r in results if r['family'] == fam])}")


if __name__ == "__main__":
    main()
