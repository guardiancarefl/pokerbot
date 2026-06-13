"""Replay a live_dryrun JSONL's raw_records through make_decision and dump
per-frame behavioral fields, so two code revisions can be diffed
behaviorally (bit-identity proof at the make_decision level — one layer
above scripts/replay_session_diff.py, which stops at replay+invariant).

Mirrors run_live_dryrun's live config: fresh SessionTracker, fresh
DecisionCache, sample mode, fixed seed. Deterministic given (code, log,
checkpoint, seed): any pre/post diff is a code-behavior change.

Usage:
    python scripts/replay_make_decision_diff.py run \\
        --log logs/live_dryrun_verify1_20260609_202137.jsonl \\
        --out /tmp/pre_verify1.jsonl \\
        --checkpoint runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \\
        --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl
    python scripts/replay_make_decision_diff.py diff \\
        --pre /tmp/pre_verify1.jsonl --post /tmp/post_verify1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"

# Behavioral fields compared by `diff`. recovered_fields is read with
# getattr so the same script runs against pre-change code (field absent).
BEHAVIORAL_KEYS = (
    "status", "skip_reason", "hero_stack", "pot_total", "pot_corrected",
    "pre_hand_override_used", "street_idx", "facing_bet",
    "client_action", "chip_int", "decision_identity",
    "invariant_deltas", "click_plan", "recovered_fields",
    "anchor_refused",
)


def process(log_path: str, out_path: str, checkpoint: str,
            abstraction: str, seed: int,
            anchor_sum_floor: bool = False,
            bet_closure_recovery: bool = False,
            dead_button_handling: bool = False,
            commit_reconciliation: bool = False,
            allin_zero_stack: bool = False,
            mode: str = "sample"):
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from src.nlhe.integration.live_loop import DecisionCache, make_decision
    from src.nlhe.integration.scraper_schema import SessionTracker
    from src.nlhe.integration.click_target import plan_as_dict
    from scripts.eval_6max_self_play import _load_solver

    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    abstr = Abstraction.load(abstraction)
    solver = _load_solver(checkpoint, abstr, structure)
    tracker = SessionTracker(anchor_sum_floor=anchor_sum_floor,
                             bet_closure_recovery=bet_closure_recovery)
    cache = DecisionCache()
    rng = random.Random(seed)

    rows = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = rec.get("seq")
            raw = rec.get("raw_record")
            if raw is None:
                continue
            d = make_decision(raw, structure, solver, tracker, rng,
                              mode=mode, seq=seq,
                              decision_cache=cache,
                              bet_closure_recovery=bet_closure_recovery,
                              dead_button_handling=dead_button_handling,
                              commit_reconciliation=commit_reconciliation,
                              allin_zero_stack=allin_zero_stack)
            row = {"seq": seq, "captured_at": d.captured_at}
            row["status"] = d.status
            row["skip_reason"] = d.skip_reason
            row["hero_stack"] = d.hero_stack
            row["pot_total"] = d.pot_total
            row["pot_corrected"] = d.pot_corrected
            row["pre_hand_override_used"] = d.pre_hand_override_used
            row["street_idx"] = d.street_idx
            row["facing_bet"] = d.facing_bet
            row["client_action"] = d.client_action
            row["chip_int"] = d.resolver_raw_openspiel_chip_int
            row["decision_identity"] = (
                repr(d.decision_identity)
                if d.decision_identity is not None else None)
            row["invariant_deltas"] = d.invariant_deltas
            row["click_plan"] = (plan_as_dict(d.click_plan)
                                 if d.click_plan is not None else None)
            row["recovered_fields"] = getattr(d, "recovered_fields", None)
            row["anchor_refused"] = bool(getattr(d, "anchor_refused", False))
            rows.append(row)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")

    from collections import Counter
    cnt = Counter(r["status"] for r in rows)
    print(f"Replayed {len(rows)} frames -> {out_path}")
    for k, v in cnt.most_common():
        print(f"  {k:<28s} {v:>5d}")


def diff(pre_path: str, post_path: str, ignore_skip_reason_on_suspect: bool):
    # Key by captured_at: seq numbers COLLIDE across sub-sessions when the
    # Windows sender reconnects mid-log (e.g. 154557 holds 571 frames but
    # only 450 unique seqs — a seq-keyed diff silently drops 121 frames).
    def _load(path):
        out = {}
        for r in (json.loads(l) for l in open(path)):
            key = r.get("captured_at") or f"seq:{r['seq']}"
            assert key not in out, f"duplicate captured_at {key} in {path}"
            out[key] = r
        return out

    pre = _load(pre_path)
    post = _load(post_path)
    assert set(pre) == set(post), "frame sets differ"
    changed = []
    for s in sorted(pre):
        a, b = pre[s], post[s]
        diffs = {}
        for k in BEHAVIORAL_KEYS:
            if a.get(k) != b.get(k):
                diffs[k] = (a.get(k), b.get(k))
        # A suspect frame whose ONLY change is the skip_reason annotation
        # (recovery declined: ...) is the intended audit-trail delta, not
        # a behavioral change. click_plan embeds the reason string, so a
        # click_plan diff counts as annotation-only iff 'reason' is its
        # only differing key.
        def _click_plan_reason_only():
            pa, pb = a.get("click_plan"), b.get("click_plan")
            if not (isinstance(pa, dict) and isinstance(pb, dict)):
                return False
            if set(pa) != set(pb):
                return False
            return all(pa[k] == pb[k] for k in pa if k != "reason")

        annotation_only = (
            ignore_skip_reason_on_suspect
            and set(diffs) <= {"skip_reason", "click_plan"}
            and ("click_plan" not in diffs or _click_plan_reason_only())
            and a.get("status") == "skip_data_quality"
            and b.get("status") == "skip_data_quality"
            and "ScraperSuspect" in (a.get("skip_reason") or "")
        )
        if diffs:
            changed.append((s, diffs, annotation_only))

    n_annot = sum(1 for _, _, ao in changed if ao)
    n_behav = len(changed) - n_annot
    print(f"Total frames:                {len(pre)}")
    print(f"Identical:                   {len(pre) - len(changed)}")
    print(f"Suspect-annotation-only:     {n_annot}")
    print(f"BEHAVIORAL CHANGES:          {n_behav}")
    for s, diffs, ao in changed:
        tag = "annot" if ao else "BEHAV"
        print(f"\n[{tag}] seq={s}")
        for k, (av, bv) in diffs.items():
            print(f"  {k:<22s} {str(av)[:110]!r}")
            print(f"  {'':<22s} -> {str(bv)[:110]!r}")
    return n_behav


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("run")
    p1.add_argument("--log", required=True)
    p1.add_argument("--out", required=True)
    p1.add_argument("--checkpoint", required=True)
    p1.add_argument("--abstraction", required=True)
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--anchor-sum-floor", action="store_true",
                    help="arm the P1 anchor sum-floor guard in the "
                         "replayed SessionTracker (counterfactual runs)")
    p1.add_argument("--bet-closure-recovery", action="store_true",
                    help="arm P2 bet-closure recovery (requires "
                         "--anchor-sum-floor; counterfactual runs)")
    p1.add_argument("--dead-button-handling", action="store_true",
                    help="arm D2 dead-button position handling "
                         "(dealer-on-eliminated-seat frames parse; "
                         "counterfactual runs)")
    p1.add_argument("--commit-reconciliation", action="store_true",
                    help="arm CR anchored commit reconciliation (requires "
                         "--anchor-sum-floor; counterfactual runs)")
    p1.add_argument("--allin-zero-stack", action="store_true",
                    help="arm F2 zero-stack all-in handling (explicit-0 "
                         "committed seats parse as valid all-in states; "
                         "counterfactual runs)")
    p1.add_argument("--mode", default="sample",
                    choices=("sample", "argmax"),
                    help="policy mode for the replayed make_decision. "
                         "Default 'sample' mirrors live. 'argmax' draws "
                         "no rng, isolating structural diffs from "
                         "shared-rng-stream cascade when a flag adds new "
                         "sampled decisions.")
    p2 = sub.add_parser("diff")
    p2.add_argument("--pre", required=True)
    p2.add_argument("--post", required=True)
    p2.add_argument("--strict", action="store_true",
                    help="count suspect-frame skip_reason annotations as "
                         "behavioral changes too")
    args = ap.parse_args()
    if args.cmd == "run":
        process(args.log, args.out, args.checkpoint, args.abstraction,
                args.seed, anchor_sum_floor=args.anchor_sum_floor,
                bet_closure_recovery=args.bet_closure_recovery,
                dead_button_handling=args.dead_button_handling,
                commit_reconciliation=args.commit_reconciliation,
                allin_zero_stack=args.allin_zero_stack,
                mode=args.mode)
    elif args.cmd == "diff":
        n = diff(args.pre, args.post,
                 ignore_skip_reason_on_suspect=not args.strict)
        sys.exit(1 if n else 0)


if __name__ == "__main__":
    main()
