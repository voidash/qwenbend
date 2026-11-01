"""bytecode.py (compile pass with Bend2's static checks + VM) vs bendlite_ref (parser + checks +
interpreter) on every program: the dataset, the model-output logs, the S2 mutants and probes.
With --bvm also firmware2 on bvm.VM, from the program's stored tokens (its lexer, then the compile
pass, then the evaluator), vs bytecode.py on capture.py's tape: the code words it compiles must
equal compile_tape(..., HANDLERS) and its result must equal bytecode.run's.
Run: python bendvm/test_bytecode.py [--bvm] [model_outputs.jsonl ...]
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bendlite_ref import BendLiteError, run_source  # noqa: E402
from bvm import VM, VMFail  # noqa: E402
from bytecode import CODE, VMError, compile_tape, run  # noqa: E402
from capture import CaptureError, capture  # noqa: E402
from firmware import tape_words  # noqa: E402
from lexer_fw import Stored, load_program, token_words  # noqa: E402
from test_firmware import load_programs  # noqa: E402


def run_ref(src):
    try:
        return run_source(src)[0]
    except (BendLiteError, RecursionError):
        return "ERR"


def run_bytecode(ids, rom):
    """-> (value or ERR, the tape or None, code words or None, ops executed)."""
    try:
        tape = capture(ids, rom)
    except CaptureError:
        return "ERR", None, None, 0
    code = compile_tape(*tape_words(tape))
    if code is None:
        return "ERR", tape, None, 0
    try:
        v, n = run(code)
    except VMError:
        return "ERR", tape, code, 0
    return v, tape, code, n


def run_bvm(prog, want_code):
    """The program's stored tokens through firmware2 (lexer, compile pass, evaluator) -> (value or
    ERR, trace tokens, lex + compile tokens, code ok). The front end is measured with COMPILE_ONLY
    (which also yields the code)."""
    from firmware2 import COMPILE_ONLY, FIRMWARE, MEM_WORDS
    vc = VM(COMPILE_ONLY, max_steps=20_000_000, memory_words=MEM_WORDS)
    try:
        load_program(vc, prog)
    except ValueError:  # more tokens than the engine stores
        return "ERR", 0, 0, want_code is None
    try:
        vc.run()
        got_code = vc.mem[CODE:CODE + len(want_code)] if want_code is not None else None
        if want_code is not None and vc.r[11] != CODE + len(want_code):  # CP: nothing extra
            got_code = None
    except VMFail:
        got_code = None
    vm = VM(FIRMWARE, max_steps=20_000_000, memory_words=MEM_WORDS)
    load_program(vm, prog)
    try:
        out = vm.run()[0]
    except VMFail:
        out = "ERR"
    return out, vm.tokens, vc.tokens, got_code == want_code


def main():
    sys.setrecursionlimit(100000)
    use_bvm = "--bvm" in sys.argv
    tok, rom, progs = load_programs([a for a in sys.argv[1:] if not a.startswith("--")])
    for f in ("s2_mutants.jsonl", "s2_probes.jsonl"):
        progs += [(f, json.loads(x)["program"]) for x in open(HERE / f)]
    if use_bvm:
        from firmware2 import HANDLERS
        words = token_words(rom)
    bad, bad_bvm, bad_code, sizes, nops, toks, ctoks, n_ok = 0, 0, 0, [], [], [], [], 0
    for origin, src in progs:
        want = run_ref(src)
        got, tape, code, n = run_bytecode(tok.encode(src, add_special_tokens=False), rom)
        if got != want:
            bad += 1
            if bad <= 10:
                print("MISMATCH", origin, "bytecode", got, "ref", want, src[:160].replace("\n", " | "))
        if got != "ERR":
            n_ok += 1
            sizes.append(len(code))
            nops.append(n)
        if use_bvm:
            hcode = compile_tape(*tape_words(tape), HANDLERS) if tape is not None else None
            bv, t, ct, ok = run_bvm(Stored(tok.encode(src, add_special_tokens=False), words), hcode)
            if not ok:
                bad_code += 1
                if bad_code <= 5:
                    print("CODE MISMATCH", origin, src[:160].replace("\n", " | "))
            if bv != got:
                bad_bvm += 1
                if bad_bvm <= 8:
                    print("BVM MISMATCH", origin, "bvm", bv, "bytecode", got, src[:160].replace("\n", " | "))
            if got != "ERR":
                toks.append(t)
                ctoks.append(ct)
    print(f"bytecode vs bendlite_ref: {len(progs) - bad}/{len(progs)} agree ({n_ok} ran to a value)")
    if nops:
        print(f"code words median {int(statistics.median(sizes))} max {max(sizes)}; "
              f"ops executed median {int(statistics.median(nops))} max {max(nops)}")
    if toks:
        q, qc = statistics.quantiles(toks, n=10), statistics.quantiles(ctoks, n=10)
        print(f"firmware2 vs bytecode.py: {bad_code} code mismatches, {bad_bvm} result mismatches")
        print(f"firmware2 trace tokens (programs that ran): median {int(statistics.median(toks))} "
              f"p90 {int(q[-1])} max {max(toks)}; of which lexing + the compile pass median "
              f"{int(statistics.median(ctoks))} p90 {int(qc[-1])} max {max(ctoks)}")
    sys.exit(1 if bad or bad_bvm or bad_code else 0)


if __name__ == "__main__":
    main()
