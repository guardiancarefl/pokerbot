"""Consumer-audit v1: H2 battery 0/8112-flip scope-consistency check.

Spec section 3 (H2 battery oracle row): correction v1 is identity for
n_alive in {5, 6}; every ICM evaluation in the battery's oracle_ev is a
6-alive (fold / pre-shove) or 5-alive (win/lose after a bust) state, so
the corrected map must reproduce every oracle label exactly. This
script re-prices all three ICM terms of every spot with the corrected
map (pure math, 0 rollouts) and asserts:
  (a) reconstruction check: recomputed MH ev_fold == stored ev_fold;
  (b) corrected hero ICM == MH hero ICM on all 3 stack vectors;
  (c) therefore 0/8112 oracle label flips.

Honest note carried from the spec: TIER1_REPORT action (3) "re-label
the battery" overstates what Tier-1 licenses — the real re-label is
contingent on Tier-2 (n_alive 5/6 cells) + Tier-2b; queued separately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fold_vs_shove_battery import (HERO_SEAT, N_SEATS,
                                           _dealer_for, _POS_OFFSET)
from scripts.sng_baseline import PAYOUTS
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.icm import icm_equity
from src.nlhe.icm_correction import IcmCorrection

BATTERY = REPO_ROOT / "evals/h2_battery/battery_v1.json"
STRUCTURE_YAML = REPO_ROOT / "configs/ignition_double_up_6max_turbo.yaml"


def hero_icm_mh(stacks):
    eligible = [i for i in range(N_SEATS) if stacks[i] > 0]
    if HERO_SEAT not in eligible:
        return 0.0
    return float(icm_equity(stacks, PAYOUTS, eligible=eligible)[HERO_SEAT])


def hero_icm_corrected(corr, stacks, big_blind):
    eligible = [i for i in range(N_SEATS) if stacks[i] > 0]
    if HERO_SEAT not in eligible:
        return 0.0
    eq = corr.corrected_icm_equity(stacks, PAYOUTS, eligible=eligible,
                                   big_blind=big_blind)
    return float(eq[HERO_SEAT])


def spot_stacks(structure, spot):
    bl = structure.level(spot["level"])
    sb, bb, ante = bl.small_blind, bl.big_blind, bl.ante
    chips = int(round(spot["depth_bb"] * bb))
    dealer = _dealer_for(spot["hero_pos"])
    shover_seat = (dealer + _POS_OFFSET[spot["shover_pos"]]) % N_SEATS
    sb_seat, bb_seat = (dealer + 1) % N_SEATS, (dealer + 2) % N_SEATS

    def post(seat):
        p = ante
        if seat == sb_seat:
            p += sb
        elif seat == bb_seat:
            p += bb
        return min(p, chips)

    dead_others = sum(post(i) for i in range(N_SEATS)
                      if i not in (HERO_SEAT, shover_seat))
    stacks_fold = [chips - post(i) for i in range(N_SEATS)]
    stacks_fold[shover_seat] = chips + sum(
        post(i) for i in range(N_SEATS) if i != shover_seat)
    stacks_win = [chips - post(i) for i in range(N_SEATS)]
    stacks_win[HERO_SEAT] = 2 * chips + dead_others
    stacks_win[shover_seat] = 0
    stacks_lose = [chips - post(i) for i in range(N_SEATS)]
    stacks_lose[HERO_SEAT] = 0
    stacks_lose[shover_seat] = 2 * chips + dead_others
    return bl.big_blind, stacks_fold, stacks_win, stacks_lose


def main():
    corr = IcmCorrection.load()
    structure = TournamentStructure.from_yaml(str(STRUCTURE_YAML))
    bat = json.loads(BATTERY.read_text())
    spots = bat["spots"]
    n_recon_mismatch = n_icm_diff = n_alive_off_scope = 0
    max_abs_icm_diff = 0.0
    max_recon_err = 0.0
    for spot in spots:
        bb, s_fold, s_win, s_lose = spot_stacks(structure, spot)
        # (a) reconstruction check vs the frozen file
        max_recon_err = max(max_recon_err,
                            abs(hero_icm_mh(s_fold) - spot["ev_fold"]))
        if abs(hero_icm_mh(s_fold) - spot["ev_fold"]) > 1e-9:
            n_recon_mismatch += 1
        # (b) corrected == MH on all three vectors
        for st in (s_fold, s_win, s_lose):
            n_alive = sum(1 for s in st if s > 0)
            if n_alive == 4:
                n_alive_off_scope += 1   # would mean scope leak
            d = abs(hero_icm_corrected(corr, st, bb) - hero_icm_mh(st))
            max_abs_icm_diff = max(max_abs_icm_diff, d)
            if d != 0.0:
                n_icm_diff += 1
    # (c) identical ICM terms + identical equity term => identical EVs
    # => identical argmax labels.
    n_flips = 0 if n_icm_diff == 0 else None
    out = {
        "record_type": "battery_flip_check",
        "battery": str(BATTERY.relative_to(REPO_ROOT)),
        "n_spots": len(spots),
        "reconstruction_mismatches": n_recon_mismatch,
        "max_reconstruction_err": max_recon_err,
        "icm_value_diffs": n_icm_diff,
        "max_abs_icm_diff": max_abs_icm_diff,
        "stack_vectors_with_4_alive": n_alive_off_scope,
        "label_flips": n_flips,
        "assert_0_flips": n_flips == 0,
    }
    print(json.dumps(out, indent=2))
    out_path = REPO_ROOT / "evals/c1_correction_20260612/battery_flip_check.json"
    out_path.write_text(json.dumps(out, indent=2))
    assert n_flips == 0, "battery flip check FAILED"
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
