"""Depth-invariance probe — inference only.

For the deployed encoder + checkpoint, hold true BB-depth fixed and vary
the blind level. If the net recovers depth-in-BB from the implicit
pot/to_call/contribution-over-1500 smear, a 5BB-shove/fold strategy
should look ~identical at L2, L5, L8 even though the chip counts (and
the depth-blind `eff_stack` slice) differ wildly. If it drifts, the
missing BB-normalized feature is provably costing strategic correctness.

Battery:
  - 169 starting hands (canonical class — AsAh for AA, AsKs for AKs,
    AsKh for AKo)
  - Hero in {BTN, SB, BB}
  - Two scenarios: {unopened, facing single min-raise}
  - Three depths: 5BB, 10BB, 15BB
  - Three levels (fixed-BB axis): L2, L5, L8 (real-ante: 25/50/10,
    100/200/30, 300/600/90)

Reports:
  A. Fixed-BB: TV distance of action distribution across L2/L5/L8 at
     each (hand, position, scenario, depth). Median, 90th-pct, count of
     primary-action flips.
  B. Fixed-chips mirror: 750 chips at L1/L5/L8 — strategy SHOULD change.
  C. 2-3 sample 236-vectors with eff_stack + pot + to_call called out.
  D. Game-2 short-stack decisions (L3/L4, hero ~4-5 BB) — replay the
     actual logged frames and dump the model's distribution. Coherent
     shove/fold? Or deep-stack min-raise/limp?

No training, no encoder changes, no model changes.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pyspiel

from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.integration.replay import deal_one_card_6max, replay_to_decision
from src.nlhe.integration.scraper_schema import (
    ScraperParseError, ScraperSuspect, ScraperDataQuality,
    SessionTracker, parse_frame,
)

CHECKPOINT = "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt"
ABSTRACTION = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
N_SEATS = 6
N_ACT = len(DiscreteAction)
HERO_SEAT = 0

RANKS = "23456789TJQKA"


def enumerate_169_hands():
    """169 starting-hand classes as (hero_card1, hero_card2, label, kind).
    kind in {'pair','suited','offsuit'}."""
    out = []
    for i in range(13):  # higher card
        for j in range(13):  # lower card
            r1 = RANKS[12 - i]
            r2 = RANKS[12 - j]
            if i == j:
                # pair
                out.append((r1 + "s", r1 + "h", f"{r1}{r1}", "pair"))
            elif j > i:
                # suited: i is higher card, but j > i means RANKS[12-j] < RANKS[12-i]
                # use suited: same suit
                out.append((r1 + "s", r2 + "s", f"{r1}{r2}s", "suited"))
            else:
                # offsuit
                out.append((r1 + "s", r2 + "h", f"{r1}{r2}o", "offsuit"))
    return out


def build_preflop_spot(structure, level_num, hero_pos, scenario, hero_cards,
                       hero_depth_bb=None, hero_chips_override=None):
    """Construct an OpenSpiel state at a preflop decision node.

    Args:
        structure: TournamentStructure
        level_num: 1..N (blind level)
        hero_pos: 'BTN', 'SB', or 'BB'
        scenario: 'unopened' or 'facing_minraise'
        hero_cards: tuple of 2 card strings ('As','Kh')
        hero_depth_bb: optional fixed hero depth in BBs
        hero_chips_override: optional fixed chip count (mutex w/ depth_bb)

    Returns: (state, hero_seat, dealer_seat, bb_amount)
    """
    bl = structure.level(level_num)
    sb, bb, ante = bl.small_blind, bl.big_blind, bl.ante
    n = N_SEATS

    if hero_chips_override is not None:
        hero_chips = int(hero_chips_override)
    elif hero_depth_bb is not None:
        hero_chips = int(round(hero_depth_bb * bb))
    else:
        raise ValueError("must supply hero_depth_bb or hero_chips_override")

    if hero_chips < bb + ante:
        # Not enough to post BB+ante; skip (would all-in from blinds)
        raise ValueError(
            f"hero_chips={hero_chips} < BB+ante={bb+ante} at L{level_num}"
        )

    # Depth-invariance construction: give every alive seat the SAME chip
    # count = hero_chips. This guarantees effective depth = hero_chips
    # at every level. (Chip-pool conservation relaxed for probe purposes;
    # the model sees per-seat stack/1500 and doesn't see pool totals.)
    stacks = [hero_chips] * n

    if hero_pos == "BTN":
        dealer = HERO_SEAT
    elif hero_pos == "SB":
        dealer = (HERO_SEAT - 1) % n
    elif hero_pos == "BB":
        dealer = (HERO_SEAT - 2) % n
    else:
        raise ValueError(f"unknown hero_pos {hero_pos!r}")

    game_str = structure.to_inner_game_string_for_state(
        blind_level=bl, stacks=stacks, dealer_seat=dealer)
    game = pyspiel.load_game(game_str)
    state = game.new_initial_state()

    # Walk through chance nodes (12 hole-card deals)
    safety = 50
    while state.is_chance_node():
        safety -= 1
        if safety < 0:
            raise RuntimeError("infinite chance loop")
        deal_one_card_6max(state, HERO_SEAT, hero_cards, target_board=())

    # Apply opponent actions per scenario
    utg_seat = (dealer + 3) % n
    sb_seat = (dealer + 1) % n
    bb_seat = (dealer + 2) % n
    if scenario == "unopened":
        # Action folds around to hero. Special case: when hero is BB,
        # one earlier player must enter the pot (limp) so the hand
        # reaches BB-with-check-option rather than ending when SB folds.
        # Pick SB to limp (call the BB) — most natural "unopened" path
        # for BB-check spots. Otherwise everyone folds.
        safety = 30
        while state.current_player() != HERO_SEAT:
            safety -= 1
            if safety < 0:
                raise RuntimeError("infinite unopened loop")
            if state.is_terminal():
                raise RuntimeError("terminal before hero")
            cp = state.current_player()
            legal = state.legal_actions()
            if hero_pos == "BB" and cp == sb_seat:
                # SB limps so the action gets to BB's check option.
                if 1 not in legal:
                    raise RuntimeError(f"SB can't limp; legal={legal[:5]}")
                state.apply_action(1)
            elif 0 not in legal:
                # opponent can't fold (rare — e.g. forced post) — try call
                if 1 in legal:
                    state.apply_action(1)
                else:
                    raise RuntimeError(f"opp can't fold or check; legal={legal[:5]}")
            else:
                state.apply_action(0)  # fold
    elif scenario == "facing_minraise":
        cp_now = state.current_player()
        if cp_now != utg_seat:
            raise RuntimeError(
                f"facing_minraise expects UTG first; got cp={cp_now} utg={utg_seat}"
            )
        legal = state.legal_actions()
        min_raise = 2 * bb
        if min_raise not in legal:
            raise RuntimeError(
                f"min-raise chip_int={min_raise} not in legal {legal[:10]}"
            )
        state.apply_action(min_raise)
        # fold around to hero
        safety = 20
        while state.current_player() != HERO_SEAT:
            safety -= 1
            if safety < 0:
                raise RuntimeError("infinite fold loop")
            if state.is_terminal():
                raise RuntimeError("terminal before hero")
            legal = state.legal_actions()
            if 0 not in legal:
                raise RuntimeError(f"opp can't fold; legal={legal[:5]}")
            state.apply_action(0)
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    return state, HERO_SEAT, dealer, bb


def query_policy(solver, state, hero_seat, dealer, rng_seed=0):
    """Return (policy_dist_9d, legal_mask_9d, parsed, discrete_to_chip).
    Uses the real encode_from_parsed (deployed encoder)."""
    parsed = parse_state_6max(state, observer=hero_seat)
    parsed["dealer_seat"] = dealer
    legal_chip = list(state.legal_actions())
    view = _build_view_6max(state, parsed)
    d2c = discretize_legal_actions(legal_chip, view)
    mask = np.zeros(N_ACT, dtype=np.float32)
    for da in d2c:
        mask[int(da)] = 1.0
    rng = random.Random(rng_seed)
    encoded = solver.encoder.encode_from_parsed(parsed, rng=rng)
    features = np.asarray(encoded, dtype=np.float32)
    policy = solver.policy_nets.inference_policy(
        hero_seat, features, mask)
    return np.asarray(policy, dtype=np.float64), mask, parsed, d2c, features


def total_variation(p, q, mask):
    """TV distance between two distributions, masked to common legal slots.
    Both must be valid distributions (sum to 1 on `mask`)."""
    p = np.where(mask > 0, p, 0.0)
    q = np.where(mask > 0, q, 0.0)
    return 0.5 * float(np.sum(np.abs(p - q)))


def primary_action(p, mask):
    """argmax over legal slots."""
    masked = np.where(mask > 0, p, -1.0)
    return int(np.argmax(masked))


def report_fixed_bb_sweep(solver, structure, hands, positions, scenarios,
                          depths_bb, levels, max_hands=None):
    """Section A: fixed BB depth, vary blind level. TV across levels."""
    print("\n" + "=" * 80)
    print(f"(A) FIXED-BB SWEEP  (depth-invariance)")
    print(f"     {len(hands)} hands × {len(positions)} pos × {len(scenarios)} scen "
          f"× {len(depths_bb)} depths × {len(levels)} levels")
    print(f"     Levels: {levels}  Depths: {depths_bb}BB")
    print("=" * 80)

    if max_hands is not None:
        hands = hands[:max_hands]

    # nested results: results[(depth, pos, scen)] = list of (label, [tv_pairs], flipped_bool)
    all_records = []
    skipped = []
    for depth in depths_bb:
        for pos in positions:
            for scen in scenarios:
                tv_pairs = []          # per-hand list of TV(L2,L5), TV(L2,L8), TV(L5,L8) values
                flips = 0              # primary action varies across levels
                n_ok = 0
                for c1, c2, label, kind in hands:
                    policies = []
                    masks = []
                    failed = False
                    for L in levels:
                        try:
                            state, hs, dealer, bb = build_preflop_spot(
                                structure, L, pos, scen, (c1, c2),
                                hero_depth_bb=depth,
                            )
                            p, m, _, _, _ = query_policy(solver, state, hs, dealer)
                            policies.append(p)
                            masks.append(m)
                        except Exception as e:
                            failed = True
                            skipped.append((depth, pos, scen, label, L,
                                             type(e).__name__, str(e)[:80]))
                            break
                    if failed:
                        continue
                    # Intersect legal masks
                    common = masks[0].copy()
                    for m in masks[1:]:
                        common = np.where((common > 0) & (m > 0), 1.0, 0.0)
                    # Re-normalize policies onto common legal slots
                    renorm = []
                    for p in policies:
                        p2 = np.where(common > 0, p, 0.0)
                        s = p2.sum()
                        if s > 0:
                            p2 = p2 / s
                        renorm.append(p2)
                    # TVs across all pairs
                    n_levels = len(levels)
                    these_tvs = []
                    for i in range(n_levels):
                        for j in range(i + 1, n_levels):
                            tv = total_variation(renorm[i], renorm[j], common)
                            these_tvs.append(tv)
                    tv_pairs.append(max(these_tvs))  # report max TV pair per hand
                    # Primary action flip check
                    primaries = [primary_action(p, common) for p in renorm]
                    if len(set(primaries)) > 1:
                        flips += 1
                    n_ok += 1
                if n_ok == 0:
                    print(f"\n  {pos:<4s} {scen:<18s} depth={depth}BB: ALL SKIPPED")
                    continue
                tvs_arr = np.asarray(tv_pairs)
                med = np.median(tvs_arr)
                p90 = np.percentile(tvs_arr, 90)
                flip_pct = 100 * flips / n_ok
                # color-flag findings
                tag = "  "
                if med > 0.10 or flip_pct > 10:
                    tag = "**"
                print(f"  {tag}{pos:<4s} {scen:<18s} depth={depth:>2d}BB:  "
                      f"n_hands={n_ok:>3d}  med_TV={med:.3f}  p90_TV={p90:.3f}  "
                      f"flips={flips}/{n_ok} ({flip_pct:.1f}%)")
                all_records.append({
                    "depth": depth, "pos": pos, "scenario": scen,
                    "n_ok": n_ok, "median_tv": float(med),
                    "p90_tv": float(p90), "flips": int(flips),
                    "flip_pct": float(flip_pct),
                })

    if skipped:
        print(f"\n  Skipped {len(skipped)} cells (showing first 6):")
        for s in skipped[:6]:
            print(f"    depth={s[0]} pos={s[1]} scen={s[2]} hand={s[3]} L={s[4]}  "
                  f"reason: {s[5]}: {s[6]}")
    return all_records


def report_fixed_chips_mirror(solver, structure, hands, positions, scenarios,
                              chip_count, levels):
    """Section B: fixed chip count, vary blind level. Strategy SHOULD change."""
    print("\n" + "=" * 80)
    print(f"(B) FIXED-CHIPS MIRROR  (sanity: strategy SHOULD change)")
    print(f"     chips={chip_count}  levels={levels}")
    print("=" * 80)

    records = []
    for pos in positions:
        for scen in scenarios:
            tvs = []
            flips = 0
            n_ok = 0
            depth_per_level = {}
            for c1, c2, label, kind in hands:
                policies = []
                masks = []
                failed = False
                for L in levels:
                    bl = structure.level(L)
                    depth_per_level[L] = chip_count / bl.big_blind
                    try:
                        state, hs, dealer, bb = build_preflop_spot(
                            structure, L, pos, scen, (c1, c2),
                            hero_chips_override=chip_count,
                        )
                        p, m, _, _, _ = query_policy(solver, state, hs, dealer)
                        policies.append(p)
                        masks.append(m)
                    except Exception:
                        failed = True
                        break
                if failed:
                    continue
                common = masks[0].copy()
                for m in masks[1:]:
                    common = np.where((common > 0) & (m > 0), 1.0, 0.0)
                renorm = []
                for p in policies:
                    p2 = np.where(common > 0, p, 0.0)
                    s = p2.sum()
                    if s > 0:
                        p2 = p2 / s
                    renorm.append(p2)
                these_tvs = []
                for i in range(len(levels)):
                    for j in range(i + 1, len(levels)):
                        tv = total_variation(renorm[i], renorm[j], common)
                        these_tvs.append(tv)
                tvs.append(max(these_tvs))
                primaries = [primary_action(p, common) for p in renorm]
                if len(set(primaries)) > 1:
                    flips += 1
                n_ok += 1
            if n_ok == 0:
                continue
            tvs_arr = np.asarray(tvs)
            med = np.median(tvs_arr)
            p90 = np.percentile(tvs_arr, 90)
            flip_pct = 100 * flips / n_ok
            print(f"  {pos:<4s} {scen:<18s} chips={chip_count} "
                  f"(={depth_per_level}):")
            print(f"      n_hands={n_ok:>3d}  med_TV={med:.3f}  p90_TV={p90:.3f}  "
                  f"flips={flips}/{n_ok} ({flip_pct:.1f}%)")
            records.append({
                "pos": pos, "scenario": scen, "chips": chip_count,
                "n_ok": n_ok, "median_tv": float(med),
                "p90_tv": float(p90), "flips": int(flips),
                "flip_pct": float(flip_pct),
            })
    return records


def report_sample_vectors(solver, structure, samples):
    """Section C: print 236-vectors for 2-3 spots."""
    print("\n" + "=" * 80)
    print(f"(C) SAMPLE 236-VECTORS — eyeball eff_stack + pot + to_call")
    print("=" * 80)
    for label, level, pos, scen, c1, c2, depth_bb in samples:
        try:
            state, hs, dealer, bb = build_preflop_spot(
                structure, level, pos, scen, (c1, c2),
                hero_depth_bb=depth_bb,
            )
            p, mask, parsed, d2c, features = query_policy(
                solver, state, hs, dealer)
        except Exception as e:
            print(f"\n  {label}: BUILD FAIL — {type(e).__name__}: {e}")
            continue
        print(f"\n  [{label}]  L{level}  {pos}  {scen}  "
              f"{c1+c2}  hero {depth_bb}BB = {depth_bb*bb} chips")
        print(f"    parsed money       = {parsed['money']}")
        print(f"    parsed contribution= {parsed['contribution']}")
        print(f"    parsed pot         = {parsed['pot']}")
        print(f"    parsed big_blind   = {parsed.get('big_blind', '?')}")
        print(f"    feature_dim        = {len(features)}")
        # Map back: bucket 200 + street 4 + pos 6 + stacks 6 + active 6 +
        # contrib 6 + pot 1 + tocall 1 + eff 1 + bet5
        off = 200 + 4 + 6
        stacks_slice = features[off:off+6]
        off += 6
        active_slice = features[off:off+6]
        off += 6
        contrib_slice = features[off:off+6]
        off += 6
        pot_val = features[off]; off += 1
        tocall_val = features[off]; off += 1
        eff_val = features[off]; off += 1
        bet_slice = features[off:off+5]
        print(f"    stacks/1500        = {stacks_slice.tolist()}")
        print(f"    active mask        = {active_slice.tolist()}")
        print(f"    contrib/1500       = {contrib_slice.tolist()}")
        print(f"    pot/1500           = {float(pot_val):.4f}  (raw pot={parsed['pot']})")
        print(f"    to_call/1500       = {float(tocall_val):.4f}  "
              f"(raw to_call={int(tocall_val*1500)})")
        print(f"    eff_stack/1500     = {float(eff_val):.4f}  "
              f"(raw eff={int(eff_val*1500)} chips ≈ {eff_val*1500/bb:.1f}BB)")
        print(f"    bet-history (5d)   = {bet_slice.tolist()}")
        legal_actions = [DiscreteAction(i).name for i in range(N_ACT) if mask[i] > 0]
        print(f"    legal disc actions = {legal_actions}")
        print(f"    policy (legal only):")
        for i in range(N_ACT):
            if mask[i] > 0:
                print(f"      {DiscreteAction(i).name:<10s}  p={float(p[i]):.4f}")


def report_game2_short_stack(solver, structure):
    """Section D: replay Game-2 short-stack decisions (~4-5BB at L3/L4),
    dump model's action distribution. Coherent shove/fold or deep-stack?"""
    print("\n" + "=" * 80)
    print(f"(D) GAME-2 SHORT-STACK DECISIONS  (hero ~4-5BB at L3/L4)")
    print(f"     log: live_dryrun_20260608_051036.jsonl")
    print("=" * 80)
    # The Game-2 log was captured before raw_record was added — so we
    # can't replay it through the full bridge from a single log line.
    # Instead, mirror Game-2's hand-segments by reading the per-frame
    # blinds / hero_seat / hero_cards / dealer_seat / hero_stack /
    # n_alive, and construct a SYNTHETIC equivalent spot (since we have
    # the relevant state inputs even without raw scraper fields).
    log_path = "logs/live_dryrun_20260608_051036.jsonl"
    short_stack_hands = []
    seen = set()  # dedupe by (level, dealer, hero_cards, hero_stack)
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") != "decision":
                continue
            level = r.get("level")
            if level is None or level < 3:
                continue
            hero_stack = r.get("hero_stack")
            blinds = r.get("blinds")
            if not (blinds and hero_stack):
                continue
            bb = blinds[1]
            stack_bb = hero_stack / bb if bb > 0 else 0
            if stack_bb > 6.0:
                continue
            key = (level, r["dealer_seat"], tuple(r["hero_cards"]),
                   hero_stack)
            if key in seen:
                continue
            seen.add(key)
            short_stack_hands.append(r)
            if len(short_stack_hands) >= 12:
                break
    print(f"  Found {len(short_stack_hands)} unique short-stack decisions.")

    for r in short_stack_hands:
        bb = r["blinds"][1]
        sb_amt = r["blinds"][0]
        ante = r["blinds"][2]
        L = r["level"]
        bl = structure.level(L)
        stack_bb = r["hero_stack"] / bb
        hero_pos_idx = (r["hero_seat"] - r["dealer_seat"]) % N_SEATS
        # In 6-handed: dealer=BTN, dealer+1=SB, dealer+2=BB, dealer+3=UTG,
        # dealer+4=MP, dealer+5=CO
        pos_names = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
        hero_pos_name = pos_names[hero_pos_idx]
        scen = "facing_bet" if r.get("facing_bet") else "unopened/check"
        # Build a synthetic equivalent: unopened spot at hero_pos with
        # the hero's actual chip count
        try:
            chips = int(r["hero_stack"])
            if hero_pos_name in ("BTN", "SB", "BB"):
                # synthetic unopened spot — easiest model query
                state, hs, dealer, bb_amt = build_preflop_spot(
                    structure, L, hero_pos_name, "unopened",
                    tuple(r["hero_cards"]),
                    hero_chips_override=chips,
                )
                p, mask, _, _, _ = query_policy(solver, state, hs, dealer)
            else:
                p, mask = None, None
        except Exception as e:
            print(f"\n  seq={r.get('seq')} L{L} {hero_pos_name} {r['hero_cards']} "
                  f"({stack_bb:.1f}BB): synthetic build failed — "
                  f"{type(e).__name__}: {str(e)[:80]}")
            continue
        print(f"\n  seq={r.get('seq')}  L{L} ({sb_amt}/{bb}/{ante})  "
              f"hero={hero_pos_name}  cards={r['hero_cards']}  "
              f"stack={chips} ({stack_bb:.1f}BB)  "
              f"facing_bet={r.get('facing_bet')}")
        print(f"    LIVE bot picked: {r.get('client_action')}")
        if p is not None:
            print(f"    SYNTHETIC unopened {hero_pos_name} policy (model on same hand+depth+level):")
            for i in range(N_ACT):
                if mask[i] > 0:
                    print(f"      {DiscreteAction(i).name:<10s}  p={float(p[i]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hands", type=int, default=None,
                     help="cap number of hands per cell (debug)")
    ap.add_argument("--skip-vectors", action="store_true")
    ap.add_argument("--skip-game2", action="store_true")
    args = ap.parse_args()

    print(f"Loading structure {STRUCTURE_YAML}...")
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    print(f"  starting_chips={structure.starting_chips}  "
          f"num_players={structure.num_players}")
    print(f"Loading abstraction {ABSTRACTION}...")
    with open(ABSTRACTION, "rb") as f:
        abstr = pickle.load(f)
    print(f"Loading solver {CHECKPOINT}...")
    from scripts.eval_6max_self_play import _load_solver
    solver = _load_solver(CHECKPOINT, abstr, structure)

    feat_dim = solver.encoder.feature_dim
    print(f"\n  encoder.feature_dim = {feat_dim}")
    assert feat_dim == 236, f"FATAL: feature_dim {feat_dim} != 236"
    print(f"  ✓ confirmed feature_dim == 236")

    hands = enumerate_169_hands()
    print(f"  enumerated {len(hands)} starting-hand classes")
    positions = ["BTN", "SB", "BB"]
    scenarios = ["unopened", "facing_minraise"]

    fixed_bb_records = report_fixed_bb_sweep(
        solver, structure, hands, positions, scenarios,
        depths_bb=[5, 10, 15],
        levels=[2, 5, 8],
        max_hands=args.max_hands,
    )

    mirror_records = report_fixed_chips_mirror(
        solver, structure,
        hands[:30] if args.max_hands else hands,
        positions, scenarios,
        chip_count=750,
        levels=[1, 5, 8],
    )

    if not args.skip_vectors:
        report_sample_vectors(solver, structure, samples=[
            ("5BB-BTN-unopened L2", 2, "BTN", "unopened", "As", "Ah", 5),
            ("5BB-BTN-unopened L8", 8, "BTN", "unopened", "As", "Ah", 5),
            ("10BB-SB-facingminraise L5", 5, "SB", "facing_minraise", "Ks", "Qs", 10),
        ])

    if not args.skip_game2:
        report_game2_short_stack(solver, structure)

    print("\n" + "=" * 80)
    print(f"done.")


if __name__ == "__main__":
    main()
