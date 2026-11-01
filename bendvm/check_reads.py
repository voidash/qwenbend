"""Does firmware2 ever read a memory word nobody wrote? (The circuit needs every read row to hold
an encoded word: an unwritten DeltaNet row reads as all zeros, which changes the residual
energy that keeps the norms exact; DESIGN.md Phase 2.) The stored tokens (all 4 words of every
token and of EOF, as capture writes them) count as written.
Run: python bendvm/check_reads.py [logs.jsonl ...]
"""

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bvm import VM, VMFail  # noqa: E402
from lexer_fw import TOK, Stored, load_program, token_words  # noqa: E402
from firmware2 import FIRMWARE, LABELS, MEM_WORDS  # noqa: E402
from test_firmware import load_programs  # noqa: E402


class Watch(VM):
    """bvm.VM plus the set of written addresses and the PCs of LDs from unwritten ones."""


def main():
    tok, rom, progs = load_programs(sys.argv[1:])
    words = token_words(rom)
    where, n_bad, n, steps = Counter(), 0, 0, []
    lab = sorted((v, k) for k, v in LABELS.items())
    for _, src in progs:
        prog = Stored(tok.encode(src, add_special_tokens=False), words)
        vm = Watch(FIRMWARE, memory_words=MEM_WORDS)
        try:
            load_program(vm, prog)
        except ValueError:
            continue
        vm.written = {TOK + 512 * k + n for k in range(4) for n in range(len(prog.ids) + 1)}
        vm.bad = Counter()
        try:
            while not vm.halted:
                vm.max_steps = vm.steps + 1
                op, *a = vm.code[vm.pc]
                if op == "LD":
                    addr = (vm.r[a[1]] + a[2]) % 2**32
                    if addr not in vm.written:
                        vm.bad[vm.pc] += 1
                if op == "ST":
                    vm.written.add((vm.r[a[0]] + a[1]) % 2**32)
                try:
                    VM.run(vm)
                except VMFail as e:
                    if str(e) != "step limit":
                        raise
        except VMFail:
            pass
        n += 1
        steps.append(vm.steps)
        if vm.bad:
            n_bad += 1
            for pc, c in vm.bad.items():
                name = max((k for v, k in lab if v <= pc), key=lambda k: LABELS[k])
                where[name] += c
    print(f"{n_bad}/{n} programs read an unwritten word ({sum(steps)} instructions checked); "
          f"by routine: {where.most_common(12)}")


if __name__ == "__main__":
    main()
