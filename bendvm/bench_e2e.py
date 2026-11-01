"""End-to-end QwenBend benchmark on a uniform sample of the 200 test items (every STEP-th; no filtering
by whether Qwen gets it right). Per item: Qwen writes the program in the merged model, the same model
runs it, and the answer read from its trace is compared with the expected value. Items whose program
the engine cannot capture (>= MAX_N tokens, or a token outside BendLite) count as failures and are
not run. Writes one JSON line per item.

Run: python bendvm/bench_e2e.py [STEP] --out bendtune/logs/e2e.jsonl
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bendlite_ref import BendLiteError, run_source  # noqa: E402
from engine_circuit import FIRMWARE, MAX_N  # noqa: E402
from lexer_fw import Stored  # noqa: E402
from merge import build_merged, run  # noqa: E402
from paths import BENDTUNE  # noqa: E402
from run_engine import result  # noqa: E402
from trace import trace  # noqa: E402


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--out=")]
    out = Path(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), BENDTUNE / "logs/e2e.jsonl"))
    step = int(args[0]) if args else 10
    data = [json.loads(x) for x in open(BENDTUNE / "data/test.jsonl")]
    mlx = {r["i"]: r for r in (json.loads(x) for x in open(BENDTUNE / "logs/tuned400_test.jsonl"))}
    m, tok, info = build_merged()
    nn = tok.encode("\n\n", add_special_tokens=False)
    done = {json.loads(x)["i"] for x in open(out)} if out.exists() else set()  # resumable
    with open(out, "a") as f:
        for i in range(0, len(data), step):
            if i in done:
                continue
            row, ref = data[i], mlx[i]
            want = row["meta"]["expected"]
            rec = {"i": i, "family": row["meta"]["family"], "held_out": ref["held_out"], "expected": want}
            ids = tok.encode(ref["raw"], add_special_tokens=False)  # Qwen's program (torch == MLX, C4a)
            if len(nn + ids) >= MAX_N or not all(t in info["words"] for t in ids):
                rec.update(outcome="outside engine scope", correct=False, program=ref["raw"])
            else:
                prompt = tok.apply_chat_template(row["messages"][:2], add_generation_prompt=True, tokenize=False,
                                                 enable_thinking=False)
                pids = tok.encode(prompt, add_special_tokens=False)
                t0 = time.time()
                qwen, engine, bad = run(m, info, pids)
                dt = time.time() - t0
                prog_ids = qwen[:-1]
                src = tok.decode(prog_ids)
                got = result(engine)
                try:
                    ref_val = run_source(src)[0]
                except BendLiteError:
                    ref_val = "ERR"
                same = engine == trace(FIRMWARE, Stored(nn + prog_ids, info["words"]))
                outcome = ("correct" if got == want else
                           "rejected by Bend2 rules" if got == "ERR" else "wrong program")
                rec.update(outcome=outcome, correct=got == want, answer=got, reference=ref_val,
                           trace_equals_reference=same, exit_violations=len(bad), engine_tokens=len(engine),
                           seconds=round(dt, 1), program_equals_mlx=src.strip() == ref["raw"].strip(), program=src)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i:>3} {rec['family']:<16} {rec['outcome']:<24} answer {rec.get('answer')} expected {want} "
                  f"trace==ref {rec.get('trace_equals_reference')} {rec.get('engine_tokens', 0):>6} tok "
                  f"{rec.get('seconds', 0):>5} s", flush=True)
    rs = [json.loads(x) for x in open(out)]
    print(f"QwenBend end to end: {sum(r['correct'] for r in rs)}/{len(rs)} correct; "
          f"engine == reference on {sum(r.get('trace_equals_reference', False) for r in rs)}/"
          f"{sum('trace_equals_reference' in r for r in rs)} runs")


if __name__ == "__main__":
    main()
