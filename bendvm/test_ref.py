"""Checks bendlite_ref on every dataset program (train/valid/test; all verified by real Bend2)
and the 20 hand examples, feeding each program as Qwen3.5 token ids (the engine's input).
The value must equal the expected one; also reports interpreter steps (def calls).
Optional: --model-outputs LOG.jsonl (eval.py output) checks programs the model wrote: wherever
real Bend2 ran them (valid), the reference must give Bend2's output; wherever Bend2 rejected
them, the reference should reject too (mismatches are listed; the subset may be stricter).
Run: python bendvm/test_ref.py [--model-outputs ...]"""

import argparse
import csv
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bendlite_ref import BendLiteError, run_tokens  # noqa: E402
from paths import BENDTUNE, qwen_dir  # noqa: E402

TOK = AutoTokenizer.from_pretrained(str(qwen_dir()))
DECODE = {}


def decode(i):
    if i not in DECODE:
        DECODE[i] = TOK.decode([i])
    return DECODE[i]


def run(src):
    ids = TOK.encode(src, add_special_tokens=False)
    return run_tokens(ids, decode), len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-outputs")
    args = ap.parse_args()
    sys.setrecursionlimit(100000)
    bad, n, steps, ntoks = 0, 0, [], []
    for split in ("train", "valid", "test"):
        for line in open(BENDTUNE / f"data/{split}.jsonl"):
            ex = json.loads(line)
            src, want = ex["messages"][2]["content"], ex["meta"]["expected"]
            try:
                (v, s), k = run(src)
            except BendLiteError as e:
                v, s, k = f"ERR {e}", 0, 0
            n += 1
            steps.append(s)
            ntoks.append(k)
            if v != want:
                bad += 1
                if bad <= 5:
                    print("MISMATCH", split, ex["meta"]["family"], v, want)
    ex_dir = Path(__file__).resolve().parent / "examples"
    with open(ex_dir / "RESULTS.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                (v, s), k = run((ex_dir / f"{row['name']}.bend").read_text())
            except BendLiteError as e:
                v = f"ERR {e}"
            n += 1
            if str(v) != row["bend"]:
                bad += 1
                print("MISMATCH example", row["name"], v, row["bend"])
    steps.sort()
    ntoks.sort()
    print(f"dataset + examples: {n - bad}/{n} equal to Bend2-verified values; "
          f"def calls median {steps[len(steps) // 2]} max {steps[-1]}; "
          f"source tokens median {ntoks[len(ntoks) // 2]} max {ntoks[-1]}")
    if args.model_outputs:
        agree = disagree = 0
        for line in open(args.model_outputs):
            r = json.loads(line)
            try:
                (v, _), _ = run(r["program"])
                ref = str(v)
            except (BendLiteError, RecursionError) as e:
                ref = f"ERR {e}"
            if r["valid"]:
                ok = ref == r["output"]
            else:
                ok = ref.startswith("ERR")
            agree += ok
            disagree += not ok
            if not ok and disagree <= 8:
                print("MODEL-PROGRAM DISAGREE", r["family"], "bend:", r["valid"], r["output"], "ref:", ref[:80])
        print(f"model-written programs: ref agrees with Bend2 on {agree}/{agree + disagree}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
