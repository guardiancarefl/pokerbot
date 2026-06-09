"""Build tools/scraper_sanity_fixture/ — the BEFORE-baseline validation
fixture for the Windows scraper SanityChecker P1+P2 fix.

P1 = relax-after-stale (N=4) + consensus-derived chips-in-play ceiling.
P2 = new-hand-detector widening + board-shrank-vs-stale-prev_board fix.

Reads every logs/live_dryrun_*.jsonl on the Contabo box, splits them into
sub-session streams at seq-resets (sender reconnects restart the counter),
and emits:

  streams/<log>__sN.jsonl   ordered scraper records (verbatim raw_record
                            + seq + captured_at), one stream per sub-session
  expectations.json         machine-generated per-frame gates (see README)
  before_summary.json       the BEFORE numbers the after-run is diffed against

Gate classes generated here:
  must_reject       any frame where a stack reading exceeds the stream's
                    consensus ceiling (mode of clean-hand-start pre-hand
                    sums). Includes the two currently-CLEAN 9907 hand-start
                    frames — the intended clean->reject deltas.
  must_admit_stack  relax-class: inside a run of >= RELAX_N consecutive
                    frames flagged 'seatX stack jump A->B' with the SAME
                    stale reference A, frames at run position > RELAX_N
                    whose read value B is <= ceiling must no longer carry
                    that stack-jump reason. (Positions 1..RELAX_N are left
                    unconstrained so the fixture doesn't pin the exact
                    off-by-one of the implementation.)
  board_shrank_fix  frames flagged 'board shrank X->Y' (stale prev_board
                    compared against the next hand's board): the
                    board-shrank reason must be gone after P2.
  bit_identity      every other frame: suspect flag + reasons unchanged.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tools" / "scraper_sanity_fixture"
RELAX_N = 4

STACK_JUMP_RE = re.compile(r"^seat([1-6]) stack jump (\d+)->(\d+)$")
BOARD_SHRANK_RE = re.compile(r"^board shrank (\d+)->(\d+)$")


def split_streams(log_path: Path):
    """Yield (stream_idx, [records]) split at seq-resets."""
    streams, cur, prev_seq = [], [], None
    for line in open(log_path):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rr = d.get("raw_record")
        if rr is None:
            continue
        seq = d.get("seq")
        if prev_seq is not None and seq is not None and seq < prev_seq:
            streams.append(cur)
            cur = []
        prev_seq = seq
        cur.append({"seq": seq, "captured_at": rr.get("captured_at"),
                    "record": rr})
    if cur:
        streams.append(cur)
    return streams


def hand_start_sums(stream):
    """Pre-hand chip sums of CLEAN hand-start frames, via the bridge's own
    parser (ground truth for the consensus ceiling)."""
    import sys
    sys.path.insert(0, str(REPO))
    from src.nlhe.integration.scraper_schema import (
        parse_frame, is_hand_start, pre_hand_stacks,
        ScraperParseError, ScraperSuspect, ScraperDataQuality)
    sums = []
    for fr in stream:
        try:
            f = parse_frame(fr["record"])
        except (ScraperParseError, ScraperSuspect, ScraperDataQuality):
            continue
        if is_hand_start(f):
            sums.append(sum(pre_hand_stacks(f)))
    return sums


def main():
    (OUT / "streams").mkdir(parents=True, exist_ok=True)
    expectations = {"relax_n": RELAX_N, "streams": {}}
    before = {"streams": {}}

    for log_path in sorted((REPO / "logs").glob("live_dryrun_*.jsonl")):
        for si, stream in enumerate(split_streams(log_path), start=1):
            name = f"{log_path.stem}__s{si}"
            with open(OUT / "streams" / f"{name}.jsonl", "w") as f:
                for fr in stream:
                    f.write(json.dumps(fr) + "\n")

            sums = hand_start_sums(stream)
            ceiling = Counter(sums).most_common(1)[0][0] if sums else None

            must_reject = []
            must_admit = []
            shrank_fix = []
            run_key, run_len = None, 0  # (seat, stale_ref) consecutive run

            n_suspect = 0
            reason_counts = Counter()
            for fr in stream:
                rr = fr["record"]
                cap = fr["captured_at"]
                reasons = rr.get("suspect_reasons") or []
                if rr.get("suspect"):
                    n_suspect += 1
                    for r in reasons:
                        reason_counts[re.sub(r"\d+", "N", str(r))] += 1

                # must_reject: any stack reading over the consensus ceiling
                if ceiling is not None:
                    for sname, v in (rr.get("stacks") or {}).items():
                        if v is not None and v > ceiling:
                            must_reject.append({
                                "captured_at": cap, "seat": sname,
                                "value": v,
                                "currently_suspect": bool(rr.get("suspect")),
                            })

                # relax-class run tracking over stack-jump reasons
                jump = None
                for r in reasons:
                    m = STACK_JUMP_RE.match(str(r).strip())
                    if m:
                        jump = (f"seat{m.group(1)}", int(m.group(2)),
                                int(m.group(3)))
                        break
                if jump and rr.get("suspect"):
                    key = (jump[0], jump[1])
                    run_len = run_len + 1 if key == run_key else 1
                    run_key = key
                    if (run_len > RELAX_N and ceiling is not None
                            and jump[2] <= ceiling):
                        must_admit.append({
                            "captured_at": cap, "seat": jump[0],
                            "stale_ref": jump[1], "read_value": jump[2],
                            "run_position": run_len,
                        })
                else:
                    run_key, run_len = None, 0

                # board-shrank fix class
                for r in reasons:
                    if BOARD_SHRANK_RE.match(str(r).strip()):
                        shrank_fix.append({
                            "captured_at": cap, "reason": str(r),
                            "other_reasons": [str(x) for x in reasons
                                              if not BOARD_SHRANK_RE.match(
                                                  str(x).strip())],
                        })

            expectations["streams"][name] = {
                "consensus_ceiling": ceiling,
                "hand_start_sums": sums,
                "must_reject": must_reject,
                "must_admit_stack": must_admit,
                "board_shrank_fix": shrank_fix,
            }
            before["streams"][name] = {
                "frames": len(stream),
                "suspect_frames": n_suspect,
                "reasons": dict(reason_counts),
            }

    with open(OUT / "expectations.json", "w") as f:
        json.dump(expectations, f, indent=1)
    with open(OUT / "before_summary.json", "w") as f:
        json.dump(before, f, indent=1)

    # Console summary
    tot_frames = sum(s["frames"] for s in before["streams"].values())
    tot_susp = sum(s["suspect_frames"] for s in before["streams"].values())
    tot_rej = sum(len(s["must_reject"])
                  for s in expectations["streams"].values())
    new_rej = sum(1 for s in expectations["streams"].values()
                  for r in s["must_reject"] if not r["currently_suspect"])
    tot_admit = sum(len(s["must_admit_stack"])
                    for s in expectations["streams"].values())
    tot_shrank = sum(len(s["board_shrank_fix"])
                     for s in expectations["streams"].values())
    ceilings = {s: e["consensus_ceiling"]
                for s, e in expectations["streams"].items()
                if e["consensus_ceiling"] is not None}
    print(f"streams: {len(before['streams'])}   frames: {tot_frames}   "
          f"currently-suspect: {tot_susp}")
    print(f"consensus ceilings: {Counter(ceilings.values())}")
    print(f"(b) must_reject frames (stack > ceiling): {tot_rej} "
          f"(of which currently CLEAN -> intended new rejects: {new_rej})")
    print(f"(a) must_admit_stack frames (relax-class, pos>{RELAX_N}): "
          f"{tot_admit}")
    print(f"(P2) board_shrank_fix frames: {tot_shrank}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
