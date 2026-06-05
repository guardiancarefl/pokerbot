"""Log-only batch resolver run on a 1500-stack corpus.

For each PASSING hero-to-act frame in the captured JSONL, runs the trained
blueprint policy net through the bridge, reverse-translates the chip_int
to a real-table action, and writes a structured JSONL decision log.

NEVER CLICKS. Output is the log that the user reads to evaluate
decision quality + the bet-sizing drift indicator.

The log line schema (one per decision):
    {
      "captured_at": "...",                # frame timestamp for cross-ref
      "line_no": int,                      # source line in corpus JSONL
      "level": int|null,                   # blind level (1..)
      "n_alive": int,
      "dealer_seat": int,
      "hero_seat": int,
      "hero_cards": ["AsKh"],
      "board": ["Jc","3d","9h"],
      "street_idx": int,                   # 0/1/2/3 = preflop/flop/turn/river
      "pot_total": int,                    # scraper-view pot
      "hero_stack": int,
      "hero_bet": int,                     # current-street bet
      "facing_bet": bool,
      "is_preflop_open_spot": bool,        # watch-list filter
      "resolver_raw_openspiel_chip_int": int,
      "client_action": {
        "kind": "fold"|"call"|"check"|"raise_to",
        "chip_amount": int|null,
        "raw_openspiel_chip_int": int,
      },
      "client_action_pot_frac": float|null,
      "client_action_bb_mult": float|null,
      "client_action_stack_frac": float|null,
    }

Usage:
    python scripts/run_logonly_resolver.py \\
        --jsonl data/<your-1500-capture>.jsonl \\
        --out logs/<run_name>.jsonl \\
        --checkpoint runs/k200_blueprint_ckpt_iter_2000.pt \\
        --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.invariant import check_mid_hand_invariant
from src.nlhe.integration.replay import ReplayError, replay_to_decision
from src.nlhe.integration.scraper_schema import (
    ScraperDataQuality, ScraperParseError, ScraperSuspect, parse_frame,
)
from src.nlhe.integration.translate import openspiel_to_real_action


WATCHLIST_LEVELS = {1, 2, 3}


def is_preflop_open_spot(frame, pack) -> bool:
    """A preflop hero-to-act frame where no opponent has voluntarily acted yet,
    AND hero is not the BB (BB's option isn't an open).

    The watch-list metric (DECISIONS.md "Phase 2 bridge") tracks how often
    the model opens vs folds on these spots, and whether opens get called.
    """
    if pack.street_idx != 0:
        return False
    if frame.hero_seat == pack.bb_seat:
        return False
    for i in range(6):
        if not frame.alive[i]:
            continue
        if i == pack.sb_seat or i == pack.bb_seat:
            continue
        if i == frame.hero_seat:
            continue
        if frame.bet[i] != 0:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="captured 1500-stack corpus")
    ap.add_argument("--out", required=True, help="decision log output (JSONL)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--abstraction", required=True)
    ap.add_argument("--structure",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--mode", choices=["sample", "argmax"], default="sample",
                    help="sample (DEFAULT) = stochastic per the trained "
                         "policy distribution — the CFR-theoretic deployment "
                         "of the average strategy; what the model converged "
                         "to. argmax = deterministic plurality pick; useful "
                         "for single-decision audit or reproducibility but "
                         "NOT the deployment-correct mode for a mixed-strategy "
                         "CFR policy. Flipped from argmax to sample on "
                         "2026-06-05 after diagnosing that argmax was an "
                         "artifact masking a depth-distinct mixed strategy.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    structure = TournamentStructure.from_yaml(args.structure)

    # Load solver once
    print(f"Loading solver...")
    import pickle
    with open(args.abstraction, "rb") as f:
        abstraction = pickle.load(f)
    from scripts.eval_6max_self_play import (
        _load_solver, _sample_action_from_policy,
    )
    solver = _load_solver(args.checkpoint, abstraction, structure)
    from src.nlhe.infoset6 import parse_state_6max

    rng = random.Random(args.seed)
    n_total = n_pass = n_skip = n_error = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(args.jsonl) as fin, open(out_path, "w") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                n_error += 1
                continue
            try:
                frame = parse_frame(record)
            except (ScraperParseError, ScraperSuspect, ScraperDataQuality):
                n_skip += 1
                continue
            if not frame.controls_present:
                n_skip += 1
                continue
            if frame.alive[frame.hero_seat] and not frame.hero_cards:
                n_skip += 1
                continue
            if frame.pot_total <= 0:
                n_skip += 1
                continue

            try:
                pack = replay_to_decision(frame, structure)
            except ReplayError:
                n_skip += 1
                continue
            inv = check_mid_hand_invariant(frame, pack)
            if not inv.ok:
                n_skip += 1
                continue

            n_pass += 1

            # Run resolver
            parsed = parse_state_6max(pack.state, observer=frame.hero_seat)
            parsed["dealer_seat"] = frame.dealer_seat
            try:
                chip_int = _sample_action_from_policy(
                    solver, parsed, pack.state, rng, mode=args.mode,
                )
            except Exception as e:
                n_error += 1
                continue

            # Reverse-translate
            scraper_min_raise = max(
                frame.blinds.bb,
                2 * max((b for b in frame.bet), default=0),
            )
            scraper_max_raise = (frame.stack[frame.hero_seat]
                                  + frame.bet[frame.hero_seat])
            client_action = openspiel_to_real_action(
                chip_int,
                scraper_min_raise=scraper_min_raise,
                scraper_max_raise=scraper_max_raise,
                scraper_facing_bet=frame.hero_facing_bet,
            )

            # Identify blind level
            level = None
            for bl in structure.blind_schedule:
                if (bl.small_blind == frame.blinds.sb
                        and bl.big_blind == frame.blinds.bb
                        and bl.ante == frame.blinds.ante):
                    level = bl.level
                    break

            # Decision-level framings (None for non-raises)
            chip_amt = client_action.get("chip_amount")
            pot_frac = bb_mult = stk_frac = None
            if chip_amt is not None:
                pot_frac = chip_amt / max(1, frame.pot_total)
                bb_mult = chip_amt / max(1, frame.blinds.bb)
                stk_frac = chip_amt / max(1, frame.stack[frame.hero_seat])

            log_line = {
                "captured_at": frame.captured_at,
                "line_no": line_no,
                "level": level,
                "n_alive": sum(frame.alive),
                "dealer_seat": frame.dealer_seat,
                "hero_seat": frame.hero_seat,
                "hero_cards": list(frame.hero_cards),
                "board": list(frame.board),
                "street_idx": pack.street_idx,
                "pot_total": frame.pot_total,
                "hero_stack": frame.stack[frame.hero_seat],
                "hero_bet": frame.bet[frame.hero_seat],
                "facing_bet": frame.hero_facing_bet,
                "is_preflop_open_spot": is_preflop_open_spot(frame, pack),
                "is_watchlist_level": level in WATCHLIST_LEVELS,
                "resolver_raw_openspiel_chip_int": int(chip_int),
                "client_action": client_action,
                "client_action_pot_frac": pot_frac,
                "client_action_bb_mult": bb_mult,
                "client_action_stack_frac": stk_frac,
                "scraper_min_raise": scraper_min_raise,
                "scraper_max_raise": scraper_max_raise,
            }
            fout.write(json.dumps(log_line) + "\n")

    wall = time.time() - t0
    print(f"\nDone in {wall:.1f}s.")
    print(f"  total records:    {n_total}")
    print(f"  passed + logged:  {n_pass}")
    print(f"  skipped:          {n_skip}")
    print(f"  errored:          {n_error}")
    print(f"  decision log -> {out_path}")
    print(f"\nNext: analyze the log to read the watch-list distribution.")
    print(f"      e.g. jq -c 'select(.is_preflop_open_spot and "
          f".is_watchlist_level)' {out_path}")
    print(f"\nSTILL NO CLICKING. Log-only.")


if __name__ == "__main__":
    main()
