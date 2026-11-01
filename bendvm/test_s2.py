"""S2 check: bendlite_ref (with its static checks) vs real Bend2's verdict, on the labelled
mutants (s2_mutants.jsonl), the rule probes (s2_probes.jsonl) and on every model-written program in the eval logs.
Bend2's outcome is its printed value if `--check-only` passed, else ERR. A disagreement is UNSOUND
if the ref gives a value Bend2 does not, else a gap (the ref rejects a valid Bend2 program that
is outside BendLite). Required: 0 unsound anywhere, 0 gaps on model-written programs.
Run: python3 bendvm/test_s2.py [extra logs.jsonl ...]
"""

import json
import sys
from collections import Counter
from pathlib import Path
from paths import BENDTUNE  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bendlite_ref import BendLiteError, run_source  # noqa: E402

LOGS = [BENDTUNE / "logs/baseline_test.jsonl", BENDTUNE / "logs/tuned200_test.jsonl"]


def bend_outcome(r):
    out = r["output"].strip()
    return int(out) if r["valid"] and out.isdigit() else "ERR"


def ref_outcome(src):
    try:
        return run_source(src)[0]
    except (BendLiteError, RecursionError):
        return "ERR"


def main():
    sys.setrecursionlimit(100000)
    rows = [("mutant:" + r["kind"], r) for r in map(json.loads, open(HERE / "s2_mutants.jsonl"))]
    rows += [("probes", r) for r in map(json.loads, open(HERE / "s2_probes.jsonl"))]
    for f in LOGS + [Path(a) for a in sys.argv[1:]]:
        rows += [("model:" + f.stem, r) for r in map(json.loads, open(f))]
    agree, gap, unsound, total, shown = Counter(), Counter(), Counter(), Counter(), 0
    for origin, r in rows:
        want, got = bend_outcome(r), ref_outcome(r["program"])
        total[origin] += 1
        if want == got:
            agree[origin] += 1
            continue
        if got == "ERR":
            gap[origin] += 1  # rejected though Bend2 accepts: outside BendLite (a completeness gap)
        else:
            unsound[origin] += 1  # a value Bend2 does not give
        if shown < 12:
            shown += 1
            print(f"DISAGREE {origin}: ref {got} bend2 {want} | {r.get('error', '')[-160:]!r}")
            print("   " + r["program"].replace("\n", " | ")[:400])
    for k in total:
        print(f"{k:<28} {agree[k]}/{total[k]}" + (f"  gaps {gap[k]}" if gap[k] else "")
              + (f"  UNSOUND {unsound[k]}" if unsound[k] else ""))
    n, a = sum(total.values()), sum(agree.values())
    model_gaps = sum(v for k, v in gap.items() if k.startswith("model:"))
    print(f"ALL {a}/{n}; unsound {sum(unsound.values())}; gaps {sum(gap.values())} "
          f"(on model-written programs {model_gaps})")
    sys.exit(0 if not unsound and not model_gaps else 1)


if __name__ == "__main__":
    main()
