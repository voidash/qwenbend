"""Trace-token cost of firmware2 under the INS(pc) record format (bvm.VM accounting), for 16-bit
and 8-bit value tokens, on every program that runs (dataset + the given model-output logs).
Run: python bendvm/measure_trace.py [logs.jsonl ...]
"""

import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bvm import VM, VMFail  # noqa: E402
from lexer_fw import Stored, load_program, token_words  # noqa: E402
from firmware2 import FIRMWARE, MEM_WORDS  # noqa: E402
from test_firmware import load_programs  # noqa: E402


def main():
    tok, rom, progs = load_programs(sys.argv[1:])
    words = token_words(rom)
    res = {16: [], 8: []}
    steps = []
    for _, src in progs:
        prog = Stored(tok.encode(src, add_special_tokens=False), words)
        ok = True
        for bits in (16, 8):
            vm = VM(FIRMWARE, max_steps=20_000_000, val_bits=bits, memory_words=MEM_WORDS)
            try:
                load_program(vm, prog)
            except ValueError:
                ok = False
                break
            try:
                vm.run()
            except VMFail:
                ok = False
                break
            res[bits].append(vm.tokens)
        if ok:
            steps.append(vm.steps)
    for bits, t in res.items():
        q = statistics.quantiles(t, n=10)
        print(f"val_bits {bits:>2}: trace tokens median {int(statistics.median(t))} p90 {int(q[-1])} "
              f"max {max(t)} (programs {len(t)})")
    q = statistics.quantiles(steps, n=10)
    print(f"instructions (= INS tokens) median {int(statistics.median(steps))} p90 {int(q[-1])} max {max(steps)}")


if __name__ == "__main__":
    main()
