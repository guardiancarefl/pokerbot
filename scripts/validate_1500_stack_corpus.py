"""Validate a freshly captured 1500-stack corpus before resolver work.

Run this on Contabo against the captured JSONL transferred from the Windows
scraper. Confirms:
  1. Schema compatibility with parse_frame (no silent dropped records)
  2. Stack-sum sanity: total chips per hand = 6 × 1500 = 9000 (modulo pot)
  3. Level distribution (need coverage of levels 1-3 for the watch-list)
  4. Hero-to-act volume (need enough preflop opens at levels 1-3 to make
     open-fold-frequency statistically meaningful — target 30-100)
  5. Scraper-quality breakdown (suspect / data_quality / hero-to-act)

Does NOT call the resolver. Does NOT click. This is a corpus-acceptance gate.

Usage:
    python scripts/validate_1500_stack_corpus.py --jsonl data/<your-capture>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.invariant import check_mid_hand_invariant
from src.nlhe.integration.replay import ReplayError, replay_to_decision
from src.nlhe.integration.scraper_schema import (
    ScraperDataQuality, ScraperParseError, ScraperSuspect, parse_frame,
)


EXPECTED_STARTING_STACK = 1500
EXPECTED_NUM_SEATS = 6
EXPECTED_TOTAL_CHIPS = EXPECTED_STARTING_STACK * EXPECTED_NUM_SEATS  # 9000

# Watch-list spec (from DECISIONS.md "Phase 2 bridge"):
#   "Track preflop open-fold-frequency-when-called at levels 1-3."
# To get a meaningful read on a binary outcome (call/fold), we want
# 30-100 preflop opens at levels 1-3.
WATCHLIST_LEVELS = {1, 2, 3}
WATCHLIST_MIN_OPENS = 30
WATCHLIST_TARGET_OPENS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--structure",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    args = ap.parse_args()

    structure = TournamentStructure.from_yaml(args.structure)
    if structure.starting_chips != EXPECTED_STARTING_STACK:
        print(f"WARNING: structure.starting_chips = "
              f"{structure.starting_chips}, expected {EXPECTED_STARTING_STACK}")

    print(f"=== Validating 1500-stack corpus: {args.jsonl} ===\n")

    n_total = 0
    n_parse_ok = 0
    status_counts = Counter()
    level_counts = Counter()
    nalive_counts = Counter()
    stack_sum_dist = Counter()
    preflop_open_candidates_by_level = Counter()  # nominal preflop opens
    parse_errors = []
    schema_quirks = Counter()

    with open(args.jsonl) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors.append((line_no, f"json: {e}"))
                continue

            # Schema sanity: required fields
            for k in ("captured_at", "blinds", "dealer", "hero_cards",
                       "board", "stacks", "bets", "folded", "empty", "pot"):
                if k not in record:
                    schema_quirks[f"missing:{k}"] += 1

            try:
                frame = parse_frame(record)
            except ScraperParseError as e:
                parse_errors.append((line_no, f"parse: {e}"))
                continue
            except ScraperSuspect:
                status_counts["suspect"] += 1
                continue
            except ScraperDataQuality:
                # n_alive<4 soft-drop, etc — counts as data_quality, not parse error
                status_counts["data_quality:parse_level_dq"] += 1
                continue

            n_parse_ok += 1

            # Level distribution
            level = None
            for bl in structure.blind_schedule:
                if (bl.small_blind == frame.blinds.sb
                        and bl.big_blind == frame.blinds.bb
                        and bl.ante == frame.blinds.ante):
                    level = bl.level
                    break
            level_counts[level if level else f"unknown:{frame.blinds.sb}/{frame.blinds.bb}+{frame.blinds.ante}"] += 1

            # Stack-sum sanity (per hand, modulo where the chips currently are).
            # Ignition's `pot.total` is ANIMATION-STATE dependent: at one moment
            # it includes in-flight bets in front of seats (chips pushed to pot),
            # at another moment it doesn't (chips still in bet boxes, animation
            # not done). Both states are valid for the same total chip count.
            # Empirically verified on live_1500 (2026-06-04):
            #   Frame state A (post-collection): stacks + pot == 6 × 1500
            #   Frame state B (pre-collection):  stacks + pot + bets == 6 × 1500
            # Accept frame as 1500-stack if EITHER formula matches.
            alive_stacks_sum = sum(
                frame.stack[i] for i in range(EXPECTED_NUM_SEATS)
                if frame.alive[i]
            )
            alive_bets_sum = sum(
                frame.bet[i] for i in range(EXPECTED_NUM_SEATS)
                if frame.alive[i]
            )
            chip_total_post_collection = alive_stacks_sum + frame.pot_total
            chip_total_pre_collection = chip_total_post_collection + alive_bets_sum
            # Pick the one that matches an expected bucket; if neither, use post.
            n_alive = sum(frame.alive)
            expected = EXPECTED_STARTING_STACK * n_alive
            if chip_total_post_collection == expected:
                alive_chip_total = chip_total_post_collection
            elif chip_total_pre_collection == expected:
                alive_chip_total = chip_total_pre_collection
            else:
                # Neither matches — pick the closer one for diagnostic display
                alive_chip_total = chip_total_post_collection
            n_alive = sum(frame.alive)
            expected = EXPECTED_STARTING_STACK * n_alive
            # Bucketed to detect 1500 vs 2000 etc.
            stack_sum_dist[alive_chip_total] += 1
            nalive_counts[n_alive] += 1

            # Frame status classification (matches test_integration_midhand.py)
            if getattr(frame, "suspect", False):
                status_counts["suspect"] += 1
                continue
            if not frame.controls_present:
                status_counts["not_hero_to_act"] += 1
                continue
            # data_quality filters (mirror the harness)
            if frame.alive[frame.hero_seat] and not frame.hero_cards:
                status_counts["data_quality:hero_cards_empty"] += 1
                continue
            if frame.pot_total <= 0:
                status_counts["data_quality:pot_missing"] += 1
                continue

            try:
                pack = replay_to_decision(frame, structure)
            except ReplayError:
                status_counts["replay_error"] += 1
                continue
            inv = check_mid_hand_invariant(frame, pack)
            if not inv.ok:
                status_counts["invariant_fail"] += 1
                continue
            status_counts["pass"] += 1

            # Preflop-open watch-list counter: passing preflop frames where
            # hero is one of the first 3 voluntary actors (UTG/UTG+1/MP-ish)
            # and is not the BB (BB's "option" isn't an open).
            if pack.street_idx == 0 and level in WATCHLIST_LEVELS:
                # Approximation: hero is "opening" if all preflop voluntary
                # commits so far are zero AND hero is not the BB.
                # (precise: hero acts before any voluntary raise on first lap)
                no_voluntary_action_yet = all(
                    frame.bet[i] == 0 for i in range(EXPECTED_NUM_SEATS)
                    if frame.alive[i] and i != pack.sb_seat
                    and i != pack.bb_seat
                )
                if no_voluntary_action_yet and frame.hero_seat != pack.bb_seat:
                    preflop_open_candidates_by_level[level] += 1

    # --- Report ---
    print(f"--- Records: {n_total} total, {n_parse_ok} parsed cleanly ---\n")

    if parse_errors:
        print(f"PARSE ERRORS ({len(parse_errors)} — first 10):")
        for ln, msg in parse_errors[:10]:
            print(f"  line {ln}: {msg}")
        print()
    if schema_quirks:
        print("SCHEMA QUIRKS (top fields missing across records):")
        for k, c in schema_quirks.most_common(10):
            print(f"  {k}: {c}")
        print()

    print("=== Stack-sum sanity (per-frame total chips in play) ===")
    # Bucketize: 1500-per-alive means total in {n_alive × 1500 : n=2..6}
    expected_buckets = {EXPECTED_STARTING_STACK * n for n in range(2, 7)}
    n_1500_ok = sum(
        c for s, c in stack_sum_dist.items() if s in expected_buckets
    )
    pct_1500 = (100.0 * n_1500_ok / max(1, n_parse_ok))
    print(f"  Frames matching 1500-stack expectation: "
          f"{n_1500_ok}/{n_parse_ok} ({pct_1500:.1f}%)")
    # Show top 5 actual sums for diagnosis
    print("  Top 5 actual chip-total values:")
    for s, c in stack_sum_dist.most_common(5):
        marker = " ✓" if s in expected_buckets else " ✗ (off-format)"
        print(f"    {s:>8}: {c:>4d} frames{marker}")
    print()

    print("=== Level distribution ===")
    for lvl, c in sorted(level_counts.items(), key=lambda kv: (
        kv[0] if isinstance(kv[0], int) else 99)):
        print(f"  level {lvl}: {c}")
    print()

    print("=== n_alive distribution ===")
    for n, c in sorted(nalive_counts.items()):
        print(f"  {n}-handed: {c}")
    print()

    print("=== Frame status (Phase 2 gate metric) ===")
    for k in ("suspect", "not_hero_to_act", "data_quality:hero_cards_empty",
              "data_quality:pot_missing", "replay_error", "invariant_fail",
              "pass"):
        print(f"  {k:<32s} {status_counts.get(k, 0):>6d}")

    n_hero = (status_counts["pass"] + status_counts["replay_error"]
              + status_counts["invariant_fail"])
    if n_hero:
        rate = 100.0 * status_counts["pass"] / n_hero
        print(f"\n  --> MID-HAND PASS RATE = {rate:.2f}%  "
              f"(reference gate = 99.0%; live9 calibration = 95.12%)")
    print()

    print("=== Watch-list metric coverage (preflop opens, levels 1-3) ===")
    total_opens = sum(preflop_open_candidates_by_level.values())
    for lvl in sorted(WATCHLIST_LEVELS):
        c = preflop_open_candidates_by_level.get(lvl, 0)
        print(f"  level {lvl}: {c} preflop opens (passing frames)")
    print(f"  TOTAL passing preflop opens at levels 1-3: {total_opens}")
    if total_opens < WATCHLIST_MIN_OPENS:
        print(f"  ⚠ BELOW MINIMUM ({WATCHLIST_MIN_OPENS}). Watch-list will "
              f"have wide confidence intervals — capture more hands.")
    elif total_opens < WATCHLIST_TARGET_OPENS:
        print(f"  Sufficient for an initial read (>= {WATCHLIST_MIN_OPENS}). "
              f"Target ({WATCHLIST_TARGET_OPENS}) gives tighter intervals.")
    else:
        print(f"  ✓ Hits target ({WATCHLIST_TARGET_OPENS}).")

    print()
    print("=== Verdict ===")
    # Format check: most common chip-sum bucket must be in the
    # 1500-stack-expected set AND there must be no significant cluster
    # at a different starting-stack bucket (e.g., 12000 = 6 × 2000 from
    # the old calibration format). Small (< 200-chip) deviations from
    # 9000 are accepted as scraper animation-state noise verified
    # empirically on live_1500 — these are real 1500-stack frames
    # captured mid-chip-animation, not format issues.
    top_bucket, top_count = stack_sum_dist.most_common(1)[0]
    other_format_buckets = {
        n * 1000 for n in [10, 11, 12, 13, 14]
    }  # 2000-stack starting and adjacent
    n_other_format = sum(
        c for s, c in stack_sum_dist.items()
        if any(abs(s - bucket) < 100 for bucket in other_format_buckets)
    )
    if top_bucket not in expected_buckets:
        print(f"  ✗ FAIL: dominant chip-sum bucket ({top_bucket}) is not "
              f"in the 1500-stack expected set. Wrong format captured.")
        sys.exit(1)
    if n_other_format > n_parse_ok * 0.05:
        print(f"  ✗ FAIL: {n_other_format} frames cluster near non-1500 "
              f"buckets (>5% of parseable). Wrong format captured.")
        sys.exit(1)
    if n_hero == 0:
        print("  ✗ FAIL: no hero-to-act frames at all.")
        sys.exit(1)
    if total_opens < WATCHLIST_MIN_OPENS:
        print(f"  ⚠ ACCEPT WITH WARNING: corpus is parseable and right "
              f"format, but watch-list metric is underpowered "
              f"({total_opens} opens < {WATCHLIST_MIN_OPENS} minimum). "
              f"The log-only run can proceed but the drift indicator "
              f"will have wide confidence intervals.")
        sys.exit(0)
    print("  ✓ ACCEPT: corpus is parseable, 1500-stack format, has enough "
          "hero-to-act frames and preflop opens at levels 1-3 for the "
          "watch-list metric. Ready for the log-only resolver run.")


if __name__ == "__main__":
    main()
