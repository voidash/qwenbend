"""S2 rule probes: small programs, each isolating one static rule, run through real Bend2.
Prints ACCEPT/REJECT (+ the error's first line) per probe and writes s2_probes.jsonl
(labelled like s2_mutants.jsonl; test_s2.py checks bendlite_ref against it).
Run: python3 bendvm/s2_probe.py
"""

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from s2_mutants import bend

H = "import Base\n\n"
PROBES = {
    # termination
    "t_bool_first_unchanged": H + "def f(+c: Bool, +n: Nat) -> U32:\n  match c:\n    case True{}:\n      match n:\n        case 0n:\n          0\n        case 1n+p:\n          f(c, p)\n    case False{}:\n      0\n\ndef main() -> U32:\n  f(True{}, 3n)\n",
    "t_bool_first_literal": H + "def f(+c: Bool, +n: Nat) -> U32:\n  match c:\n    case True{}:\n      match n:\n        case 0n:\n          0\n        case 1n+p:\n          f(True{}, p)\n    case False{}:\n      0\n\ndef main() -> U32:\n  f(True{}, 3n)\n",
    "t_u32_first_literal": H + "def f(+x: U32, +n: Nat) -> U32:\n  match n:\n    case 0n:\n      x\n    case 1n+p:\n      f(5, p)\n\ndef main() -> U32:\n  f(1, 3n)\n",
    "t_second_param_shrinks": H + "def f(+n: Nat, +m: Nat) -> U32:\n  match m:\n    case 0n:\n      0\n    case 1n+q:\n      f(n, q)\n\ndef main() -> U32:\n  f(2n, 3n)\n",
    "t_wrong_binder_column": H + "def f(+n: Nat, +m: Nat) -> U32:\n  match m:\n    case 0n:\n      0\n    case 1n+q:\n      f(q, m)\n\ndef main() -> U32:\n  f(2n, 3n)\n",
    "t_nested_binder": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      match p:\n        case 0n:\n          1\n        case 1n+q:\n          f(q)\n\ndef main() -> U32:\n  f(5n)\n",
    "t_nested_outer_binder": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      match p:\n        case 0n:\n          1\n        case 1n+q:\n          f(p)\n\ndef main() -> U32:\n  f(5n)\n",
    "t_nat_literal_smaller": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      f(0n)\n\ndef main() -> U32:\n  f(5n)\n",
    "t_rec_in_zero_arm": H + "def f(+n: Nat, +m: Nat) -> U32:\n  match n:\n    case 0n:\n      match m:\n        case 0n:\n          0\n        case 1n+q:\n          f(n, q)\n    case 1n+p:\n      f(p, m)\n\ndef main() -> U32:\n  f(2n, 3n)\n",
    "t_rec_zero_arm_literal": H + "def f(+n: Nat, +m: Nat) -> U32:\n  match n:\n    case 0n:\n      match m:\n        case 0n:\n          0\n        case 1n+q:\n          f(0n, q)\n    case 1n+p:\n      f(p, m)\n\ndef main() -> U32:\n  f(2n, 3n)\n",
    "t_lexi_reset": H + "def f(+n: Nat, +m: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      f(p, 7n)\n\ndef main() -> U32:\n  f(2n, 3n)\n",
    "t_recursive_arg_call": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      (f(p) + f(p) : U32)\n\ndef main() -> U32:\n  f(5n)\n",
    "t_unmatched_body_rec": H + "def f(+n: Nat) -> U32:\n  f(n)\n\ndef main() -> U32:\n  f(5n)\n",
    # affine
    "a_lone_used_twice": H + "def f(x: U32) -> U32:\n  (x + x : U32)\n\ndef main() -> U32:\n  f(5)\n",
    "a_lone_unused": H + "def f(x: U32) -> U32:\n  7\n\ndef main() -> U32:\n  f(5)\n",
    "a_lone_once_per_arm": H + "def f(+c: Bool, x: U32) -> U32:\n  match c:\n    case True{}:\n      x\n    case False{}:\n      (x + 1 : U32)\n\ndef main() -> U32:\n  f(True{}, 5)\n",
    "a_lone_matched_then_used": H + "def f(n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      U32.from_nat(n)\n\ndef main() -> U32:\n  f(5n)\n",
    "a_lone_bool_matched_then_used": H + "def g(c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef f(c: Bool) -> U32:\n  match c:\n    case True{}:\n      g(c)\n    case False{}:\n      0\n\ndef main() -> U32:\n  f(True{})\n",
    "a_lone_binder_twice": H + "def f(n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      (U32.from_nat(p) + f(p) : U32)\n\ndef main() -> U32:\n  f(5n)\n",
    "a_plus_binder_twice": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      (U32.from_nat(p) + f(p) : U32)\n\ndef main() -> U32:\n  f(5n)\n",
    "a_plus_matched_then_used": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      (U32.from_nat(n) + f(p) : U32)\n\ndef main() -> U32:\n  f(5n)\n",
    "a_lone_in_compare_twice": H + "def f(x: U32) -> Bool:\n  (x < x : U32)\n\ndef g(c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef main() -> U32:\n  g(f(5))\n",
    "a_lone_twice_in_builtin": H + "def f(x: U32) -> U32:\n  U32.max(x, x)\n\ndef main() -> U32:\n  f(5)\n",
    "a_plus_bool": H + "def f(+c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef main() -> U32:\n  f(True{})\n",
    "a_lone_n_and_p": H + "def f(n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      (U32.from_nat(n) + U32.from_nat(p) : U32)\n\ndef main() -> U32:\n  f(5n)\n",
    "a_lone_n_twice_zero_arm": H + "def f(n: Nat) -> U32:\n  match n:\n    case 0n:\n      (U32.from_nat(n) + U32.from_nat(n) : U32)\n    case 1n+p:\n      0\n\ndef main() -> U32:\n  f(5n)\n",
    "a_lone_n_twice_succ_arm": H + "def f(n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      (U32.from_nat(n) + U32.from_nat(n) : U32)\n\ndef main() -> U32:\n  f(5n)\n",
    "a_lone_bool_twice_in_arm": H + "def g(+c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef f(c: Bool) -> U32:\n  match c:\n    case True{}:\n      (g(c) + g(c) : U32)\n    case False{}:\n      0\n\ndef main() -> U32:\n  f(True{})\n",
    "t_bool_literal_unmatched": H + "def f(+c: Bool, +n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      f(True{}, p)\n\ndef main() -> U32:\n  f(True{}, 3n)\n",
    "t_matched_succ_unchanged": H + "def f(+n: Nat, +m: Nat) -> U32:\n  match m:\n    case 0n:\n      0\n    case 1n+q:\n      match n:\n        case 0n:\n          0\n        case 1n+p:\n          f(n, q)\n\ndef main() -> U32:\n  f(2n, 3n)\n",
    "t_matched_wrong_lit": H + "def f(+n: Nat, +m: Nat) -> U32:\n  match n:\n    case 0n:\n      match m:\n        case 0n:\n          0\n        case 1n+q:\n          f(1n, q)\n    case 1n+p:\n      f(p, m)\n\ndef main() -> U32:\n  f(2n, 3n)\n",
    "t_rec_call_in_arg": H + "def f(+n: Nat) -> Nat:\n  match n:\n    case 0n:\n      0n\n    case 1n+p:\n      f(f(p))\n\ndef main() -> U32:\n  U32.from_nat(f(5n))\n",
    "t_rec_binder_lit_bigger": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      match p:\n        case 0n:\n          1\n        case 1n+q:\n          f(1n)\n\ndef main() -> U32:\n  f(5n)\n",
    "g_annot_u32": H + "def main() -> U32:\n  (5 : U32)\n",
    "g_annot_nested_group": H + "def main() -> U32:\n  ((1 + 2 : U32) : U32)\n",
    "g_plain_paren": H + "def main() -> U32:\n  (5)\n",
    "g_plain_paren_call": H + "def main() -> U32:\n  (U32.from_nat(3n))\n",
    "g_double_paren": H + "def main() -> U32:\n  ((5))\n",
    "g_paren_in_group": H + "def main() -> U32:\n  ((2) + 3 : U32)\n",
    "g_ops_no_annot": H + "def main() -> U32:\n  (1 + 2)\n",
    "g_annot_nat_as_u32": H + "def main() -> U32:\n  U32.from_nat((5n : U32))\n",
    "g_annot_nat": H + "def main() -> U32:\n  U32.from_nat((5n : Nat))\n",
    "g_annot_bool": H + "def g(c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef main() -> U32:\n  g((True{} : Bool))\n",
    "g_cmp_annot_bool": H + "def g(c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef main() -> U32:\n  g((1 < 2 : Bool))\n",
    "g_group_annot_nat": H + "def main() -> U32:\n  U32.from_nat((1n + 2n : Nat))\n",
    "g_bool_annot_u32": H + "def main() -> U32:\n  (True{} : U32)\n",
    "g_paren_bool": H + "def g(c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef main() -> U32:\n  g((True{}))\n",
    "g_paren_rec_arg": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      f((p))\n\ndef main() -> U32:\n  f(5n)\n",
    "g_annot_rec_arg": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 1n+p:\n      f((p : Nat))\n\ndef main() -> U32:\n  f(5n)\n",
    "g_paren_var_use_twice": H + "def f(x: U32) -> U32:\n  ((x) + 1 : U32)\n\ndef main() -> U32:\n  f(5)\n",
    # typing
    "y_u32_match_nat_pat": H + "def f(+x: U32) -> U32:\n  match x:\n    case 0n:\n      0\n    case 1n+p:\n      1\n\ndef main() -> U32:\n  f(5)\n",
    "y_nat_match_bool_pat": H + "def f(+n: Nat) -> U32:\n  match n:\n    case True{}:\n      0\n    case False{}:\n      1\n\ndef main() -> U32:\n  f(5n)\n",
    "y_from_nat_of_u32": H + "def main() -> U32:\n  U32.from_nat(5)\n",
    "y_is_eq_nat": H + "def g(c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef main() -> U32:\n  g(U32.is_eq(5n, 5n))\n",
    "y_bool_return_u32_body": H + "def f(+x: U32) -> Bool:\n  x\n\ndef main() -> U32:\n  7\n",
    "y_nat_return_literal": H + "def f(+x: U32) -> Nat:\n  3n\n\ndef main() -> U32:\n  U32.from_nat(f(1))\n",
    "y_group_one_cmp_bool": H + "def g(c: Bool) -> U32:\n  match c:\n    case True{}:\n      1\n    case False{}:\n      0\n\ndef main() -> U32:\n  g((1 + 2 < 4 : U32))\n",
    "y_nat_literal_in_group": H + "def main() -> U32:\n  (1n + 2 : U32)\n",
    "y_cmp_result_as_u32": H + "def main() -> U32:\n  (1 < 2 : U32)\n",
    "y_main_returns_nat_body": H + "def main() -> U32:\n  5n\n",
    "y_unused_def_type_error": H + "def f(+x: U32) -> U32:\n  5n\n\ndef main() -> U32:\n  7\n",
    "y_dead_arm_unknown_var": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      zzz\n    case 1n+p:\n      1\n\ndef main() -> U32:\n  f(5n)\n",
    "y_big_u32_literal": H + "def main() -> U32:\n  4294967296\n",
    "y_big_nat_literal": H + "def main() -> U32:\n  U32.from_nat(4294967296n)\n",
    "y_match_binder_shadows": H + "def f(+n: Nat, +p: U32) -> U32:\n  match n:\n    case 0n:\n      p\n    case 1n+p:\n      U32.from_nat(p)\n\ndef main() -> U32:\n  f(5n, 9)\n",
    "y_dup_param_names": H + "def f(+x: U32, +x: U32) -> U32:\n  x\n\ndef main() -> U32:\n  f(1, 2)\n",
    "y_arms_reversed": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 1n+p:\n      1\n    case 0n:\n      0\n\ndef main() -> U32:\n  f(5n)\n",
    "y_false_first": H + "def f(+c: Bool) -> U32:\n  match c:\n    case False{}:\n      0\n    case True{}:\n      1\n\ndef main() -> U32:\n  f(True{})\n",
    "y_same_arm_twice": H + "def f(+n: Nat) -> U32:\n  match n:\n    case 0n:\n      0\n    case 0n:\n      1\n\ndef main() -> U32:\n  f(5n)\n",
}


def main():
    with tempfile.TemporaryDirectory() as td, ThreadPoolExecutor(8) as pool:
        res = list(pool.map(lambda kv: bend(kv[1][1], td, kv[0]), enumerate(PROBES.items())))
    with open(Path(__file__).resolve().parent / "s2_probes.jsonl", "w") as f:
        for (name, src), r in zip(PROBES.items(), res):
            f.write(json.dumps({"kind": "probe:" + name, "program": src, **r}) + "\n")
    for (name, _), r in zip(PROBES.items(), res):
        first = ""
        if not r["valid"]:
            e = r["error"]
            i = e.find("Error:")
            first = e[i + 6:i + 130].replace("\n", " ") if i >= 0 else e[:120].replace("\n", " ")
        print(f"{name:<32} {'ACCEPT ' + r['output'] if r['valid'] else 'REJECT'}  {first}")


if __name__ == "__main__":
    main()
