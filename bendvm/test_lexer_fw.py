"""C3b: the firmware lexer (lexer_fw.py, inside firmware2) vs capture2.Lexer: from the stored tokens
(load_tokens), firmware2 runs until 'lex_done'; the tape in memory must equal the Python lexer's
tape as the engine sees it (firmware.tape_words + the end marker), or both must reject.
Run: python bendvm/test_lexer_fw.py [logs.jsonl ...] [--fuzz N]
"""

import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bvm import VM, VMFail  # noqa: E402
from capture import CaptureError  # noqa: E402
from capture2 import lex, rom_words  # noqa: E402
from firmware import TAPE_K, TAPE_V, tape_words  # noqa: E402
from firmware2 import FIRMWARE, LABELS, MEM_WORDS  # noqa: E402
from lexer_fw import load_tokens  # noqa: E402
from test_firmware import load_programs  # noqa: E402


def fw_tape(ids, words):
    """-> (kinds, values) up to and including the end marker, or 'ERR'; and the instructions."""
    vm = VM(FIRMWARE, memory_words=MEM_WORDS, max_steps=2_000_000)
    try:
        load_tokens(vm, ids, words, LABELS)
    except ValueError:
        return "ERR", 0
    done = LABELS["lex_done"]
    try:
        while vm.pc != done:
            vm.max_steps = vm.steps + 1
            try:
                vm.run()
            except VMFail as e:
                if str(e) != "step limit":
                    raise
    except VMFail:
        return "ERR", vm.steps
    n = next(i for i in range(512) if vm.mem[TAPE_K + i] == 0)
    return (vm.mem[TAPE_K:TAPE_K + n + 1], vm.mem[TAPE_V:TAPE_V + n + 1]), vm.steps


def py_tape(ids, words):
    try:
        ks, vs = tape_words(lex(ids, words))
    except CaptureError:
        return "ERR"
    return ks, vs


def main():
    fuzz = int(sys.argv[sys.argv.index("--fuzz") + 1]) if "--fuzz" in sys.argv else 0
    tok, rom, progs = load_programs([a for a in sys.argv[1:] if a.endswith(".jsonl")])
    words = {i: w for i, s in rom.items() if (w := rom_words(s)) is not None}
    srcs = [s for _, s in progs]
    if fuzz:
        rng = random.Random(5)
        pieces = ["x", "fuel", "n", "5", "12", "0n", "1n+p", ".is", "U32.is_eq", "_a", "(", ")", ",", ":",
                  "+", "-", ">", "<", "=", "->", "<=", ">=", "*", "/", "%", "{", "}", " ", "  ", "\n",
                  "\n  ", "\n    ", "3x", "7n", "n5", "..", "a.b", "9999999999", "4294967296n", "429496729",
                  "4294967295", "4294967296"]
        srcs += ["".join(rng.choice(pieces) for _ in range(rng.randint(1, 25))) for _ in range(fuzz)]
    bad, steps = 0, []
    for src in srcs:
        ids = tok.encode(src, add_special_tokens=False)
        want = py_tape(ids, words)
        got, n = fw_tape(ids, words)
        if got != want:
            bad += 1
            if bad <= 5:
                print("MISMATCH", repr(src[:100]), "\n got ", str(got)[:160], "\n want", str(want)[:160])
        elif got != "ERR":
            steps.append(n)
    print(f"firmware lexer == capture2.Lexer on {len(srcs) - bad}/{len(srcs)} sources ({fuzz} fuzzed); "
          f"lexing instructions median {int(statistics.median(steps))} max {max(steps)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
