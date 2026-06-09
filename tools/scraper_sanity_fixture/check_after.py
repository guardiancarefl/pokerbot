"""Validate a patched SanityChecker run against the BEFORE-baseline fixture.

Stdlib-only; runs anywhere (built for the Windows scraper project).

Input you produce on the Windows side: for each streams/<name>.jsonl,
replay the records IN ORDER through the patched SanityChecker (fresh
checker state per stream) and write after/<name>.jsonl with one line per
frame:

    {"captured_at": "...", "suspect": bool, "suspect_reasons": [...]}

Then:

    python check_after.py --fixture <fixture dir> --after <after dir>

Gates (exit 0 iff all four pass):
  (a) must_admit_stack — the previously-rejected correct reads carry NO
      'seatX stack jump' reason anymore (frame may stay suspect for
      unrelated co-reasons, e.g. a board flag on the same frame).
  (b) must_reject — every over-ceiling read (11101/11104/9907/9607/9406/
      9407/11407) is suspect with a reason naming that seat, at every
      frame, INCLUDING the two hand-start frames the current checker
      admits as clean.
  (c) ceiling — no AFTER-clean frame contains any stack reading above the
      stream's consensus ceiling.
  (d) bit-identity on clean — every frame that is currently clean stays
      clean (suspect=False), except the (b) frames flagged
      currently_suspect=false (the two intended clean->reject deltas).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STACK_JUMP_RE = re.compile(r"seat([1-6])\s+stack\s+jump")


def load_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(Path(__file__).parent))
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    fx = Path(args.fixture)
    after_dir = Path(args.after)

    expectations = json.load(open(fx / "expectations.json"))
    failures = []
    counts = {"a_pass": 0, "a_fail": 0, "b_pass": 0, "b_fail": 0,
              "c_fail": 0, "d_pass": 0, "d_fail": 0, "frames": 0}

    for name, exp in expectations["streams"].items():
        stream = load_jsonl(fx / "streams" / f"{name}.jsonl")
        after_path = after_dir / f"{name}.jsonl"
        if not after_path.exists():
            failures.append(f"[missing] after/{name}.jsonl not found")
            continue
        after = {r["captured_at"]: r for r in load_jsonl(after_path)}
        ceiling = exp["consensus_ceiling"]

        must_admit = {(m["captured_at"], m["seat"])
                      for m in exp["must_admit_stack"]}
        must_reject = {m["captured_at"]: m for m in exp["must_reject"]}
        new_rejects = {c for c, m in must_reject.items()
                       if not m["currently_suspect"]}

        for fr in stream:
            cap = fr["captured_at"]
            counts["frames"] += 1
            a = after.get(cap)
            if a is None:
                failures.append(f"[{name}] {cap}: no AFTER record")
                continue
            a_reasons = [str(r) for r in (a.get("suspect_reasons") or [])]
            a_suspect = bool(a.get("suspect"))

            # (a)
            for m in exp["must_admit_stack"]:
                if m["captured_at"] != cap:
                    continue
                seat_n = m["seat"].replace("seat", "")
                hit = any(STACK_JUMP_RE.search(r) and f"seat{seat_n}" in r
                          for r in a_reasons)
                if hit:
                    counts["a_fail"] += 1
                    failures.append(
                        f"[{name}] (a) {cap}: correct read "
                        f"{m['seat']}={m['read_value']} still carries a "
                        f"stack-jump reason: {a_reasons}")
                else:
                    counts["a_pass"] += 1

            # (b)
            if cap in must_reject:
                m = must_reject[cap]
                named = any(m["seat"] in r for r in a_reasons)
                if a_suspect and named:
                    counts["b_pass"] += 1
                else:
                    counts["b_fail"] += 1
                    failures.append(
                        f"[{name}] (b) {cap}: over-ceiling "
                        f"{m['seat']}={m['value']} not rejected "
                        f"(suspect={a_suspect}, reasons={a_reasons})")

            # (c)
            if not a_suspect and ceiling is not None:
                for sname, v in (fr["record"].get("stacks") or {}).items():
                    if v is not None and v > ceiling:
                        counts["c_fail"] += 1
                        failures.append(
                            f"[{name}] (c) {cap}: clean frame admits "
                            f"{sname}={v} > ceiling {ceiling}")

            # (d)
            if not fr["record"].get("suspect") and cap not in new_rejects:
                if a_suspect:
                    counts["d_fail"] += 1
                    failures.append(
                        f"[{name}] (d) {cap}: currently-clean frame newly "
                        f"flagged: {a_reasons}")
                else:
                    counts["d_pass"] += 1

    print(f"frames checked:            {counts['frames']}")
    print(f"(a) admit correct reads:   {counts['a_pass']} pass / "
          f"{counts['a_fail']} fail")
    print(f"(b) reject over-ceiling:   {counts['b_pass']} pass / "
          f"{counts['b_fail']} fail")
    print(f"(c) ceiling on clean:      {counts['c_fail']} violations")
    print(f"(d) clean stays clean:     {counts['d_pass']} pass / "
          f"{counts['d_fail']} fail")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f_ in failures[:60]:
            print(" ", f_)
        if len(failures) > 60:
            print(f"  ... +{len(failures) - 60} more")
        sys.exit(1)
    print("\nALL GATES PASS")


if __name__ == "__main__":
    main()
