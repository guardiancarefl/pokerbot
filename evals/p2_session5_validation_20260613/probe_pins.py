"""Per-pin P2 introspection probe (session-5 validation, 2026-06-13).

Replays logs/live_dryrun_20260612_230149.jsonl through make_decision with
P1+P2 armed (identical to replay_make_decision_diff run --anchor-sum-floor
--bet-closure-recovery, seed 42), and at each pin frame re-runs the P2
internals AFTER make_decision has processed it (tracker state at that point
equals the state step-6b saw), printing every intermediate: invariant
deltas, signature classification, anchors, blind-posting evidence,
candidate construction, and the corrected frame's re-validation residuals.
Read-only diagnostic; no production code touched.
"""
from __future__ import annotations

import dataclasses
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOG = REPO_ROOT / "logs/live_dryrun_20260612_230149.jsonl"
CKPT = REPO_ROOT / "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
ABST = REPO_ROOT / "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE_YAML = str(REPO_ROOT / "configs/ignition_double_up_6max_turbo.yaml")

PINS = {"20260612_193706_536", "20260612_194941_036", "20260612_200631_846"}


def main():
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from src.nlhe.integration.live_loop import DecisionCache, make_decision
    from src.nlhe.integration.scraper_schema import (
        SessionTracker, parse_frame, classify_displacement_deltas,
        build_bet_closure_candidate,
    )
    from src.nlhe.integration.replay import replay_to_decision, ReplayError
    from src.nlhe.integration.invariant import check_mid_hand_invariant
    from scripts.eval_6max_self_play import _load_solver

    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    abstr = Abstraction.load(str(ABST))
    solver = _load_solver(str(CKPT), abstr, structure)
    tracker = SessionTracker(anchor_sum_floor=True, bet_closure_recovery=True)
    cache = DecisionCache()
    rng = random.Random(42)

    for line in open(LOG):
        rec = json.loads(line)
        raw = rec.get("raw_record")
        if raw is None:
            continue
        seq = rec.get("seq")
        d = make_decision(raw, structure, solver, tracker, rng,
                          mode="sample", seq=seq, decision_cache=cache,
                          bet_closure_recovery=True)
        if rec.get("captured_at") not in PINS:
            continue

        print("=" * 76)
        print(f"PIN seq={seq} captured_at={rec['captured_at']} "
              f"hero_cards={raw.get('hero_cards')}")
        print(f"make_decision: status={d.status}")
        print(f"  skip_reason={d.skip_reason}")
        print(f"  recovered_fields={d.recovered_fields}")

        frame = parse_frame(raw)
        corrected = tracker.corrected_pot_for(frame)
        if corrected is not None:
            frame = dataclasses.replace(frame, pot_total=int(corrected))
            print(f"  [pot correction applied: {corrected}]")
        pre = tracker.pre_hand_for(frame)
        print(f"  pre_hand_for (regular override): {pre}")
        try:
            pack = replay_to_decision(frame, structure,
                                      pre_hand_override=pre)
        except ReplayError as e:
            print(f"  replay_to_decision FAILED: {e}")
            continue
        inv = check_mid_hand_invariant(frame, pack)
        print(f"  invariant ok={inv.ok} deltas={inv.deltas}")
        kind, payload = classify_displacement_deltas(inv.deltas)
        print(f"  classify_displacement_deltas -> {kind!r}, {payload!r}")
        anchors, refused_why = tracker.bet_closure_anchors_for(frame)
        print(f"  bet_closure_anchors_for: anchors={anchors} "
              f"refused_why={refused_why!r}")
        evidence = tracker.blind_posting_evidence_for(frame)
        print(f"  blind_posting_evidence_for: {evidence!r}")
        print(f"  frame: dealer_seat={frame.dealer_seat} "
              f"hero_seat={frame.hero_seat} blinds=({frame.blinds.sb},"
              f"{frame.blinds.bb},{frame.blinds.ante})")
        print(f"  frame stacks={frame.stack} bets={frame.bet} "
              f"pot={frame.pot_total} alive={frame.alive} "
              f"empty={frame.empty} folded={frame.folded}")
        if kind != "closure":
            continue
        seat, x = payload
        for pre_a, source in (anchors or []):
            print(f"  -- anchor source={source} pre_hand={pre_a}")
            cand, why = build_bet_closure_candidate(frame, seat, x, pre_a)
            if cand is None:
                print(f"     candidate REFUSED: {why}")
                continue
            print(f"     candidate: stack[{seat}]={cand.stack[seat]} "
                  f"bet[{seat}]={cand.bet[seat]} pot={cand.pot_total}")
            eff = cand
            c2 = tracker.corrected_pot_for(cand)
            if c2 is not None:
                eff = dataclasses.replace(cand, pot_total=int(c2))
                print(f"     [candidate pot correction: {c2}]")
            try:
                pack2 = replay_to_decision(eff, structure,
                                           pre_hand_override=pre_a)
            except ReplayError as e:
                print(f"     candidate replay FAILED: {e}")
                continue
            inv2 = check_mid_hand_invariant(eff, pack2)
            print(f"     re-validation ok={inv2.ok} "
                  f"residual deltas={inv2.deltas}")


if __name__ == "__main__":
    main()
