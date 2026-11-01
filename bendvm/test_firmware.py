"""BendVM firmware vs the tape walker (and so vs bendlite_ref / real Bend2) on every program:
Qwen token ids -> capture tape -> BVM memory -> firmware on bvm.VM. The result (value, or FAIL
as an error) must equal capture.Walker's (value, or error). Records steps and trace tokens.
Run: python bendvm/test_firmware.py [model_outputs.jsonl ...]
     [--fuzz N]   (also N random BendLite programs from fuzz.py, if present)
"""

import json
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer
from paths import BENDTUNE, qwen_dir  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bvm import VM, VMFail  # noqa: E402
from capture import CaptureError, WalkError, build_rom, capture, run_ids  # noqa: E402
from firmware import FIRMWARE, load_tape  # noqa: E402



def run_firmware(ids, rom):
    try:
        tape = capture(ids, rom)
    except CaptureError:
        return "ERR", 0, 0
    vm = VM(FIRMWARE, max_steps=20_000_000)
    try:
        load_tape(vm, tape)
    except ValueError:
        return "ERR", 0, 0  # the tape does not fit
    try:
        out = vm.run()
    except VMFail:
        return "ERR", vm.steps, vm.tokens
    return (out[0] if out else "ERR"), vm.steps, vm.tokens


def run_walker(ids, rom):
    try:
        return run_ids(ids, rom)[0]
    except (CaptureError, WalkError, RecursionError, KeyError, IndexError):
        return "ERR"


def load_programs(logs):
    """-> tokenizer, ROM, [(origin, source)]: the dataset plus each model-output log."""
    tok = AutoTokenizer.from_pretrained(str(qwen_dir()))
    rom = build_rom(lambda i: tok.decode([i]), len(tok))
    progs = []
    for split in ("train", "valid", "test"):
        for line in open(BENDTUNE / f"data/{split}.jsonl"):
            ex = json.loads(line)
            progs.append(("data", ex["messages"][2]["content"]))
    for f in logs:
        for line in open(f):
            progs.append((Path(f).name, json.loads(line)["program"]))
    return tok, rom, progs


def main():
    sys.setrecursionlimit(100000)
    tok, rom, progs = load_programs([a for a in sys.argv[1:] if not a.startswith("--")])
    bad, steps, toks, n_ok = 0, [], [], 0
    for origin, src in progs:
        ids = tok.encode(src, add_special_tokens=False)
        want = run_walker(ids, rom)
        got, s, t = run_firmware(ids, rom)
        if got != want:
            bad += 1
            if bad <= 8:
                print("MISMATCH", origin, "firmware", got, "walker", want, src[:100].replace("\n", " | "))
        if got != "ERR":
            n_ok += 1
            steps.append(s)
            toks.append(t)
    print(f"firmware vs walker: {len(progs) - bad}/{len(progs)} agree ({n_ok} ran to a value)")
    if toks:
        q = statistics.quantiles(toks, n=10)
        print(f"per program: steps median {int(statistics.median(steps))} max {max(steps)}; "
              f"trace tokens median {int(statistics.median(toks))} p90 {int(q[-1])} max {max(toks)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
