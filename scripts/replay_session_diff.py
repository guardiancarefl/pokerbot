"""Replay all 200 frames from logs/live_dryrun_20260608_152756.jsonl
through the bridge (raw_record → SessionTracker → replay_to_decision →
check_mid_hand_invariant). Outputs per-frame JSONL with hashes so two
runs can be diffed bit-for-bit.

Used to compare pre-fix and post-fix bridge behavior across the full
200-frame session — verify 170/192 are fixed AND other frames are
byte-identical.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.invariant import check_mid_hand_invariant
from src.nlhe.integration.replay import replay_to_decision, ReplayError
from src.nlhe.integration.scraper_schema import (
    ScraperDataQuality, ScraperParseError, ScraperSuspect,
    SessionTracker, derive_action_sequence, parse_frame,
)

LOG = "logs/live_dryrun_20260608_152756.jsonl"
STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"


def _hash(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
    return h.hexdigest()[:16]


def process(out_path: str):
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    # Load all records
    records = []
    with open(LOG) as f:
        for line in f:
            r = json.loads(line)
            records.append(r)

    tracker = SessionTracker()
    out_rows = []
    for rec in records:
        seq = int(rec["seq"])
        raw = rec["raw_record"]
        row = {"seq": seq, "captured_at": rec.get("captured_at")}
        try:
            frame = parse_frame(raw)
        except ScraperParseError as e:
            row["bridge_status"] = "parse_error"
            row["reason"] = str(e)[:60]
            out_rows.append(row); continue
        except ScraperSuspect as e:
            row["bridge_status"] = "scraper_suspect"
            row["reason"] = str(e)[:60]
            out_rows.append(row); continue
        except ScraperDataQuality as e:
            row["bridge_status"] = "scraper_data_quality"
            row["reason"] = str(e)[:60]
            out_rows.append(row); continue

        tracker.observe(frame)
        row["controls_present"] = bool(frame.controls_present)
        if not frame.controls_present:
            row["bridge_status"] = "not_hero_to_act"
            out_rows.append(row); continue
        if frame.alive[frame.hero_seat] and not frame.hero_cards:
            row["bridge_status"] = "hero_no_cards"
            out_rows.append(row); continue
        row["hero_cards"] = list(frame.hero_cards)
        row["hero_seat"] = int(frame.hero_seat)
        row["dealer_seat"] = int(frame.dealer_seat)
        row["street_idx"] = int(len(frame.board) and
                                 {3: 1, 4: 2, 5: 3}.get(len(frame.board), 0))
        row["blinds"] = [int(frame.blinds.sb), int(frame.blinds.bb),
                          int(frame.blinds.ante)]
        row["pot_total"] = int(frame.pot_total)

        pre_hand_override = tracker.pre_hand_for(frame)
        row["pre_hand_override"] = (list(pre_hand_override)
                                      if pre_hand_override else None)
        # Try replay with override, then fallback to no-override (same
        # as live_loop.make_decision)
        used_override = False
        pack = None
        replay_err = None
        try:
            pack = replay_to_decision(
                frame, structure, pre_hand_override=pre_hand_override)
            used_override = pre_hand_override is not None
        except ReplayError as e:
            if pre_hand_override is not None:
                try:
                    pack = replay_to_decision(
                        frame, structure, pre_hand_override=None)
                    used_override = False
                except ReplayError as e2:
                    replay_err = f"both_failed: {type(e2).__name__}: {str(e2)[:80]}"
            else:
                replay_err = f"{type(e).__name__}: {str(e)[:80]}"
        except Exception as e:
            replay_err = f"{type(e).__name__}: {str(e)[:80]}"

        if pack is None:
            row["bridge_status"] = "replay_error"
            row["reason"] = replay_err
            out_rows.append(row); continue

        row["pre_hand_override_used"] = used_override
        row["preflop_commit_per_alive"] = int(pack.preflop_commit_per_alive)
        # Derive action sequence ourselves (no fallback) for hashing
        try:
            asq = derive_action_sequence(
                frame, pre_hand_override=pack.pre_hand_stacks)
            asq_tuple = tuple(tuple(t) for t in asq)
            row["action_seq_hash"] = _hash(asq_tuple)
            row["action_seq_len"] = len(asq)
        except Exception as e:
            row["action_seq_hash"] = None
            row["action_seq_err"] = f"{type(e).__name__}: {str(e)[:60]}"

        # OpenSpiel state hash (contribution + money + current_player +
        # street_idx — the load-bearing state for the invariant)
        parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
        row["state_contrib"] = list(parsed["contribution"])
        row["state_money"] = list(parsed["money"])
        row["state_cp"] = int(parsed["current_player"])
        row["state_street"] = int(parsed["street_idx"])
        row["state_hash"] = _hash(
            row["state_contrib"], row["state_money"],
            row["state_cp"], row["state_street"])

        # Invariant
        inv = check_mid_hand_invariant(frame, pack)
        row["invariant_ok"] = bool(inv.ok)
        row["invariant_deltas"] = list(inv.deltas) if inv.deltas else None
        if inv.ok:
            row["bridge_status"] = "invariant_pass"
        else:
            row["bridge_status"] = "invariant_fail"
        out_rows.append(row)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    # Summary
    from collections import Counter
    cnt = Counter(r["bridge_status"] for r in out_rows)
    print(f"\nReplayed {len(out_rows)} frames")
    print(f"Status breakdown:")
    for k, v in cnt.most_common():
        print(f"  {k:<25s} {v:>5d}")
    inv_fails = [r for r in out_rows if r.get("bridge_status") == "invariant_fail"]
    if inv_fails:
        print(f"\nInvariant failures ({len(inv_fails)}):")
        for r in inv_fails:
            print(f"  seq={r['seq']}  hero={r.get('hero_cards')}  "
                  f"street={r.get('street_idx')}  "
                  f"deltas={r.get('invariant_deltas')}")
    print(f"\nWritten: {out_path}")


def diff(pre_path: str, post_path: str):
    pre = {r["seq"]: r for r in
            (json.loads(line) for line in open(pre_path))}
    post = {r["seq"]: r for r in
             (json.loads(line) for line in open(post_path))}
    assert set(pre) == set(post), f"seq sets differ"
    n = len(pre)
    changed = []
    same = 0
    for s in sorted(pre):
        a = pre[s]; b = post[s]
        keys_to_diff = (
            "bridge_status", "invariant_ok", "invariant_deltas",
            "action_seq_hash", "state_hash",
            "pre_hand_override_used", "controls_present",
        )
        diffs = {}
        for k in keys_to_diff:
            if a.get(k) != b.get(k):
                diffs[k] = (a.get(k), b.get(k))
        if diffs:
            changed.append((s, diffs, a, b))
        else:
            same += 1

    print(f"\nTotal frames:        {n}")
    print(f"Bit-identical pre↔post: {same}")
    print(f"Changed:                {len(changed)}")
    print()
    if changed:
        print(f"--- Changed frames ---")
        for s, diffs, a, b in changed:
            print(f"\nseq={s}  hero={a.get('hero_cards')}  "
                  f"street={a.get('street_idx')}")
            for k, (av, bv) in diffs.items():
                print(f"  {k:<22s}  {av!r}  →  {bv!r}")
        # Classify changes
        invariant_fixes = [r for r in changed
                            if r[2].get("invariant_ok") is False
                            and r[3].get("invariant_ok") is True]
        invariant_regressions = [r for r in changed
                                  if r[2].get("invariant_ok") is True
                                  and r[3].get("invariant_ok") is False]
        print(f"\n--- Classification ---")
        print(f"  invariant FIX (FAIL→PASS):    {len(invariant_fixes)}")
        for r in invariant_fixes:
            s, *_ = r
            print(f"    seq={s}")
        print(f"  invariant REGRESSION (PASS→FAIL):  {len(invariant_regressions)}")
        for r in invariant_regressions:
            s, *_ = r
            print(f"    seq={s}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("run")
    p1.add_argument("--out", required=True)
    p2 = sub.add_parser("diff")
    p2.add_argument("--pre", required=True)
    p2.add_argument("--post", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        process(args.out)
    elif args.cmd == "diff":
        diff(args.pre, args.post)


if __name__ == "__main__":
    main()
