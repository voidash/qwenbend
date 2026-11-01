"""C3a: capture as an in-circuit token store + a lexer over stored words (DESIGN.md "C3 design").

ROM: every BendLite-only token id -> a segment pattern and up to 3 payload words; the circuit
stores, per program token, W0 = the pattern (the firmware lexer's handler) and W1..W3 = the
payloads. Segments, in order:
  SP n      a run of n spaces (payload n: indentation counts spaces)
  NL        one newline
  PU c      a punctuation char, payload pcode(c) ('-' '<' '>' may pair with the next char)
  NMa       a name part starting with a letter or '_'  (payloads: len mod 8, H(part))
  NMdot     a name part starting with '.'             (payloads: len mod 8, H(part))
  NMd       a single digit                            (payloads: the digit, H(digit))
Qwen3.5's vocabulary has no token with two adjacent digits and none with a digit followed by a
letter (checked over all 122,849 BendLite-only ids), so a digit part is always one digit.
A token whose payloads need more than 3 words, or with more than 3 segments, is UNSUPPORTED
(the engine rejects the program).

lex(words) is the lexer the firmware implements, in Python, over the stored words only; it must
produce capture.capture()'s tape (or raise CaptureError where capture does) on every program.

Run: python bendvm/capture2.py [logs.jsonl ...] [--fuzz N]
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import NAMECH, CaptureError, name_hash, rotl4  # noqa: E402

M = 2**32
SEG = re.compile(r"[A-Za-z0-9_.]+|\n|[ ]+|[(),:+\-*/%<>{}=]")
NPAY = {"SP": 1, "NL": 0, "PU": 1, "NMa": 2, "NMdot": 2, "NMd": 2}


def segments(s):
    """token text -> [(type, payloads)] or None (unsupported / not BendLite)."""
    if "".join(SEG.findall(s)) != s:
        return None
    out = []
    for m in SEG.findall(s):
        if m == "\n":
            out.append(("NL", []))
        elif m[0] == " ":
            out.append(("SP", [len(m)]))
        elif m[0] in NAMECH:
            h = name_hash(m)[0]
            if m[0] == ".":
                out.append(("NMdot", [len(m) % 8, h]))
            elif not m[0].isdigit():
                out.append(("NMa", [len(m) % 8, h]))
            elif len(m) == 1:
                out.append(("NMd", [int(m), h]))
            else:  # never in Qwen3.5's vocabulary (see above)
                return None
        else:
            out.append(("PU", [ord(m)]))
    return out


def rom_words(s):
    """-> (pattern, [payload words]) or None if unsupported."""
    segs = segments(s)
    if segs is None or len(segs) > 3:
        return None
    pay = [w for _, p in segs for w in p]
    if len(pay) > 3:
        return None
    return tuple(t for t, _ in segs), pay


class Lexer:
    """The capture state machine over stored token words (what the firmware lexer does)."""

    def __init__(self):
        self.tape = []
        self.word = None  # None | ["name", key] | ["num", value, big] | ["nat", value, big] | ["bad"]
        self.pending = None  # '-' '<' '>'
        self.col = None
        self.invalid = False

    def flush_word(self):
        w, self.word = self.word, None
        if w is None:
            return
        if w[0] == "name":
            self.tape.append(("NAME", (w[1], None)))
        elif w[0] == "num":
            self.tape.append(("NUM", M if w[2] else w[1]))
        elif w[0] == "nat":
            self.tape.append(("NAT", M if w[2] else w[1]))
        else:
            self.invalid = True

    def flush_pending(self, nxt=None):
        if self.pending is None:
            return False
        p, self.pending = self.pending, None
        if nxt is not None and p + nxt in ("->", "<=", ">="):
            self.tape.append(("P", p + nxt))
            return True
        self.tape.append(("P", p))
        return False

    def start_col_char(self):
        """A non-space, non-newline char while counting indentation emits NL(col)."""
        if self.col is not None:
            self.tape.append(("NL", self.col))
            self.col = None

    def part(self, kind, pay):
        """A name-char part: NMa / NMdot [len mod 8, H] or NMd [digit, H(digit)]."""
        self.start_col_char()
        if self.pending is not None:
            self.flush_pending()
        w = self.word
        rot = 1 if kind == "NMd" else pay[0]
        if w is None:
            if kind == "NMdot":  # '.' cannot start a word: not BendLite
                self.invalid = True
            elif kind == "NMa":
                self.word = ["name", pay[1]]
            else:
                self.word = ["num", pay[0], False]
        elif w[0] == "name":  # any name chars continue a name
            w[1] = rotl4(w[1], rot) ^ pay[1]
        elif w[0] == "num" and kind == "NMd":  # value * 10 + digit, noting overflow
            big = w[2] or w[1] > (M - 1 - pay[0]) // 10
            w[1], w[2] = (w[1] * 10 + pay[0]) % M, big
        elif w[0] == "num" and kind == "NMa" and pay == [1, name_hash("n")[0]]:
            self.word = ["nat", w[1], w[2]]
        else:  # digits then other name chars, or anything after the Nat suffix
            self.word = ["bad"]

    def punct(self, c):
        self.start_col_char()
        self.flush_word()
        if self.pending is not None and self.flush_pending(c):
            return
        if c in "-<>":
            self.pending = c
        else:
            self.tape.append(("P", c))

    def space(self, n):
        if self.col is not None:
            self.col += n
            return
        self.flush_word()
        if self.pending is not None:
            self.flush_pending(" ")

    def newline(self):
        if self.col is not None:
            self.col = 0
            return
        self.flush_word()
        if self.pending is not None:
            self.flush_pending("\n")
        if self.tape:  # nothing emitted yet: a leading newline is ignored
            self.col = 0

    def feed(self, pattern, pay):
        pay = list(pay)
        for kind in pattern:
            p = [pay.pop(0) for _ in range(NPAY[kind])]
            if kind == "SP":
                self.space(p[0])
            elif kind == "NL":
                self.newline()
            elif kind == "PU":
                self.punct(chr(p[0]))
            else:
                self.part(kind, p)

    def finish(self):
        self.flush_word()
        self.flush_pending()
        if self.col is not None:
            self.tape.append(("NL", 0))
            self.col = None
        if self.invalid:
            raise CaptureError("a token outside BendLite")
        return self.tape


def lex(ids, words):
    """Token ids -> tape, using only each token's stored words (words: id -> rom_words or None)."""
    lx = Lexer()
    for i in ids:
        w = words.get(i)
        if w is None:
            raise CaptureError("a token outside BendLite (or unsupported by the engine)")
        lx.feed(*w)
    return lx.finish()


def key_tape(tape):
    """Compare tapes as the engine sees them (firmware.tape_words): NAME by key (capture.py carries
    (key, len), the lexer (key, None)); a number >= 2^32 is BIG either way."""
    return [(k, v[0] if k == "NAME" else min(v, M) if k in ("NUM", "NAT") else v) for k, v in tape]


if __name__ == "__main__":
    from capture import capture
    from test_firmware import load_programs
    fuzz = int(sys.argv[sys.argv.index("--fuzz") + 1]) if "--fuzz" in sys.argv else 0
    logs = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
    tok, rom, progs = load_programs(logs)
    words = {i: rom_words(s) for i, s in rom.items()}
    words = {i: w for i, w in words.items() if w is not None}
    print(f"ROM: {len(words)} of {len(rom)} BendLite-only tokens supported "
          f"({len({w[0] for w in words.values()})} patterns)")
    bad = 0
    for origin, src in progs:
        ids = tok.encode(src, add_special_tokens=False)
        try:
            want = key_tape(capture(ids, rom))
        except CaptureError:
            want = "ERR"
        try:
            got = key_tape(lex(ids, words))
        except CaptureError:
            got = "ERR"
        if got != want:
            bad += 1
            if bad <= 5:
                print("MISMATCH", origin, repr(src[:80]), "\n got ", str(got)[:200], "\n want", str(want)[:200])
    print(f"lexer over stored words == capture on {len(progs) - bad}/{len(progs)} programs")
    if fuzz:  # random strings over BendLite pieces, tokenized by Qwen: the edge cases
        import random
        rng = random.Random(3)
        pieces = ["x", "fuel", "n", "5", "12", "0n", "1n+p", ".is", "U32.is_eq", "_a", "(", ")", ",", ":",
                  "+", "-", ">", "<", "=", "->", "<=", ">=", "*", "/", "%", "{", "}", " ", "  ", "\n",
                  "\n  ", "\n    ", "3x", "7n", "n5", "..", "a.b", "9999999999", "4294967296n"]
        n_err = n_unsup = fbad = 0
        for _ in range(fuzz):
            src = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 25)))
            ids = tok.encode(src, add_special_tokens=False)
            if any(i in rom and i not in words for i in ids):
                n_unsup += 1  # an engine-unsupported token: rejected by design
                continue
            try:
                want = key_tape(capture(ids, rom))
            except CaptureError:
                want, n_err = "ERR", n_err + 1
            try:
                got = key_tape(lex(ids, words))
            except CaptureError:
                got = "ERR"
            if got != want:
                fbad += 1
                if fbad <= 5:
                    print("FUZZ MISMATCH", repr(src))
        print(f"fuzz: {fuzz - n_unsup - fbad}/{fuzz - n_unsup} agree ({n_err} capture errors; "
              f"{n_unsup} strings with an engine-unsupported token skipped)")
        bad += fbad
    sys.exit(1 if bad else 0)
