"""Build the post-F1 synthetic of session-5 log 230149.

Task: the session-5 seat2/3/5 dealer-burst kill set (9c9h/KcQd/AcTs,
postmortem evals/session5_postmortem_20260613/POSTMORTEM.txt section [3])
reads the pointed (dealer) seat as stack=None AND empty=True on the RAW
frames, because the Windows ocr_int truthiness bug (F1) dropped the
consensus ZEROS — an all-in / just-busted seat genuinely displays "0".

The post-F1 world ships the literal 0 on an OCCUPIED box. So the faithful
post-F1 transform of a None/empty pointed seat is:  stack None -> 0  and
empty True -> False  (the box renders a real 0, the seat is not vacant).

We apply that transform to EVERY frame whose dealer seat reads
(stack is None) on raw — i.e. the whole "dealer points to a
None-reading seat" burst class (the dominant session-5 hand-killer).
We deliberately do NOT touch any other seat or field: this isolates F2's
behavior on exactly the kill class.

Output: synthetic_postf1_burst_230149.jsonl (same schema as the raw log,
session_header preserved).
"""
import json
import sys
from pathlib import Path

RAW = Path("/home/quant/pokerbot/logs/live_dryrun_20260612_230149.jsonl")
OUT = Path("/home/quant/pokerbot/evals/f2_abort_fixes_20260613/"
           "synthetic_postf1_burst_230149.jsonl")

SEATS = ["seat1", "seat2", "seat3", "seat4", "seat5", "seat6"]


def main():
    n_frames = 0
    n_patched = 0
    patched_seqs = []
    with open(RAW) as fin, open(OUT, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("record_type") == "session_header":
                fout.write(json.dumps(rec) + "\n")
                continue
            rr = rec.get("raw_record")
            if rr is not None:
                n_frames += 1
                dealer = rr.get("dealer")
                stacks = rr.get("stacks") or {}
                empty = rr.get("empty") or {}
                # Post-F1 transform: dealer seat reads None on raw ->
                # ship the real consensus 0 on an occupied box.
                if dealer in stacks and stacks.get(dealer) is None:
                    stacks[dealer] = 0
                    if dealer in empty:
                        empty[dealer] = False
                    n_patched += 1
                    patched_seqs.append(rec.get("seq"))
            fout.write(json.dumps(rec) + "\n")
    print(f"frames={n_frames} patched(dealer None->0,occupied)={n_patched}")
    print(f"patched seqs: {patched_seqs}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
