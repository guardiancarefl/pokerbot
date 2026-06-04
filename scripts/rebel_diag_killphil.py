"""Diagnostic harness: WHERE does the value-only ReBeL hero bleed chips to
KillPhilMTT?

The trusted plateau-check says hero loses −0.04 to −0.05 ICM-delta/hand vs a
FIXED, non-adaptive Shanky profile (KillPhilMTT). A fixed strategy beating us
means we have a specific systematic deviation from GTO — locatable by slicing.

This script plays N paired hands of (value-only ReBeL hero) vs (KillPhilMTT × 5
seats), records per-hand chip P&L AND ICM-delta plus categorical bucketing
features, and prints the slice tables:

  * by ending street (preflop fold / flop / turn / river fold / river showdown)
  * by hero starting-stack bucket (<10bb / 10-20bb / 20-40bb / 40+bb)
  * by hero position relative to button (BTN/SB/BB/UTG/MP/CO)
  * by hero's voluntary involvement (folded preflop vs saw flop)
  * by all-in (was hero all-in at any point in the hand)
  * by went-to-showdown
  * by blind level (early/mid/late tournament)

For each slice: count(hands), Σ(hero_chip_pl), mean(chip_pl), Σ(hero_icm_pl),
mean(icm_pl), share-of-total-loss (only over rows with negative icm).

CONCENTRATED vs DIFFUSE read: if a few buckets account for >50% of the total
ICM loss, the leak is concentrated and identifiable. Otherwise diffuse.
"""
import sys, os, random, argparse, json, statistics, collections
sys.path.insert(0, '.')
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import pyspiel

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.stack_sampler import sample_starting_state
from src.nlhe.icm import sng_payouts_6max_double_up
from src.nlhe.icm_returns import icm_adjust_returns
from src.nlhe.infoset6 import parse_state_6max
from src.nlhe.scripted_bots import ShankyProfilePolicy
from scripts.rebel_gate2 import (BlueprintHero, ReBeLHero, mlp,
                                 ABSTR, STRUCT, CKPT, NET)
from scripts.eval_6max_self_play import _load_solver

SHANKY_DIR = "data/shanky_profiles"
NUM_SEATS = 6
# Button-relative position labels — BTN=0, then clockwise SB,BB,UTG,MP,CO.
POS_FROM_BTN = ["BTN", "SB", "BB", "UTG", "MP", "CO"]


def hero_position(hero_seat, dealer_seat):
    return POS_FROM_BTN[(hero_seat - dealer_seat) % NUM_SEATS]


def stack_bucket(stack_chips, bb_chips):
    """Bucket hero's starting stack in big blinds."""
    if bb_chips <= 0:
        return "n/a"
    bb = stack_chips / bb_chips
    if bb < 10:   return "<10bb"
    if bb < 20:   return "10-20bb"
    if bb < 40:   return "20-40bb"
    return "40+bb"


def play_hand_diag(gs, sm, hero_seat, hero_policy, opp_policies, dealer, seed,
                   payouts, bb_chips):
    """Play one hand, return a per-hand record dict (or None if interrupted)."""
    rng = random.Random(seed)
    game = pyspiel.load_game(gs); state = game.new_initial_state()
    starting = list(sm["stacks"])
    hero_start_stack = int(starting[hero_seat])

    rec = {
        "seed": seed,
        "blind_level": int(sm["blind_level"].level),
        "bb_chips": int(bb_chips),
        "hero_start_chips": hero_start_stack,
        "hero_start_bb": (hero_start_stack / bb_chips) if bb_chips > 0 else 0.0,
        "stack_bucket": stack_bucket(hero_start_stack, bb_chips),
        "position": hero_position(hero_seat, dealer),
        "saw_flop": False, "saw_turn": False, "saw_river": False,
        "all_in": False,
        "ended_street": -1,  # 0=pre, 1=flop, 2=turn, 3=river
        "ended_action": "n/a",  # 'fold' | 'allin' | 'check_call_sd' | 'last_street_action'
        "hero_voluntary_in": False,  # invested more than the blind/ante (preflop raise/call to a raise)
        "went_to_showdown": False,
        "alive_at_start": int(sum(1 for s in starting if s > 0)),
        # per-street: chips contributed by hero (cumulative end-of-street contribution).
        "hero_contrib_end_pre": 0, "hero_contrib_end_flop": 0,
        "hero_contrib_end_turn": 0, "hero_contrib_end_river": 0,
        # the last action TAKEN by anyone (we track hero's last)
        "hero_last_action_was_fold": False,
        "hero_last_action_was_aggressive": False,
    }
    # Track hero's prior aggression (was hero the preflop aggressor?)
    hero_aggressive_actions = 0
    hero_total_actions = 0
    last_street = 0
    last_hero_contrib = 0

    for _ in range(500):
        if state.is_terminal():
            break
        if state.is_chance_node():
            o = state.chance_outcomes()
            state.apply_action(int(rng.choices([x[0] for x in o],
                                               weights=[x[1] for x in o], k=1)[0]))
            continue
        parsed = parse_state_6max(state); parsed["dealer_seat"] = dealer
        cur_street = int(parsed["street_idx"])
        if cur_street != last_street:
            # End of last street — record hero's cumulative contribution at that point.
            hc = int(parsed["contribution"][hero_seat])
            if last_street == 0: rec["hero_contrib_end_pre"] = hc
            elif last_street == 1: rec["hero_contrib_end_flop"] = hc
            elif last_street == 2: rec["hero_contrib_end_turn"] = hc
            elif last_street == 3: rec["hero_contrib_end_river"] = hc
            last_street = cur_street
        if cur_street >= 1: rec["saw_flop"] = True
        if cur_street >= 2: rec["saw_turn"] = True
        if cur_street >= 3: rec["saw_river"] = True
        cp = parsed["current_player"]
        pol = hero_policy if cp == hero_seat else opp_policies[cp]
        action = int(pol.select_action(parsed, state, rng, mode="sample"))
        if cp == hero_seat:
            hero_total_actions += 1
            # Heuristic: action 0 = fold; action 1 (the call/check) is passive; any
            # action index >= 2 in this universe is a bet/raise (open, 3-bet, etc.).
            if action == 0:
                rec["hero_last_action_was_fold"] = True
                rec["hero_last_action_was_aggressive"] = False
            elif action >= 2:
                hero_aggressive_actions += 1
                rec["hero_last_action_was_aggressive"] = True
                rec["hero_last_action_was_fold"] = False
            else:
                rec["hero_last_action_was_aggressive"] = False
                rec["hero_last_action_was_fold"] = False
            # Voluntary-in (VPIP): hero put real chips in pot beyond forced blind.
            # Compute to_call = max contribution at table minus hero's; any non-fold
            # action with to_call > 0 means hero CALLED a bet (defended BB or called a
            # raise from SB/late). Any raise (action >= 2) is also voluntary.
            contribs = parsed["contribution"]
            to_call = int(max(contribs)) - int(contribs[hero_seat])
            if cur_street == 0 and action != 0:
                if action >= 2:
                    rec["hero_voluntary_in"] = True
                elif to_call > 0:
                    rec["hero_voluntary_in"] = True
        state.apply_action(action)

    if not state.is_terminal():
        return None

    # Finalize: which street did the hand end on?
    last_parsed = None
    try:
        last_parsed = parse_state_6max(state, observer=0)
    except Exception:
        pass
    end_street = int(last_parsed["street_idx"]) if last_parsed else last_street
    rec["ended_street"] = end_street
    # capture hero contrib at the last street
    if last_parsed is not None:
        hc = int(last_parsed["contribution"][hero_seat])
        if end_street == 0: rec["hero_contrib_end_pre"] = hc
        elif end_street == 1: rec["hero_contrib_end_flop"] = hc
        elif end_street == 2: rec["hero_contrib_end_turn"] = hc
        elif end_street == 3: rec["hero_contrib_end_river"] = hc
    # Showdown = ended on river AND at least 2 players still in (didn't end on fold)
    # We approximate: if the river was reached AND no one's last action was a fold
    # AND >= 2 players had remaining commitments, it's a showdown.
    chip_returns = state.returns()
    rec["hero_chip_pl"] = float(chip_returns[hero_seat])
    icm = icm_adjust_returns(chip_returns=chip_returns,
                             starting_stacks=starting, payouts=payouts)
    rec["hero_icm_pl"] = float(icm[hero_seat])
    rec["went_to_showdown"] = (end_street == 3 and rec["hero_chip_pl"] != -rec["hero_contrib_end_river"])
    rec["hero_aggression_freq"] = (hero_aggressive_actions / max(hero_total_actions, 1))
    # All-in: hero's MAX cumulative contribution across any street >= 95% of starting stack.
    # contrib_end_* is cumulative-from-hand-start (zero where street wasn't reached),
    # so the max over reached streets is hero's total chips at risk by the end.
    max_contrib = max(rec["hero_contrib_end_pre"], rec["hero_contrib_end_flop"],
                      rec["hero_contrib_end_turn"], rec["hero_contrib_end_river"])
    rec["max_contrib"] = max_contrib
    rec["all_in"] = max_contrib >= int(0.95 * hero_start_stack) and hero_start_stack > 0
    return rec


def build_killphil_factory(structure):
    """KillPhilMTT at every seat (the harness's 'shanky' branch, fixed bot per level)."""
    path = os.path.join(SHANKY_DIR, "KillPhilMTT.txt")
    by_level = {}
    for bl in structure.blind_schedule:
        bb = bl.inflated_big_blind(6)
        by_level[bl.level] = ShankyProfilePolicy(name="KillPhilMTT",
                                                 profile_path=path, big_blind_chips=bb)
    return lambda lvl: [by_level[lvl]] * 6, by_level


def slice_table(records, key_fn, name, total_icm_loss):
    """Aggregate hero P&L over rows grouped by key_fn(rec). Print + return rows."""
    buckets = collections.OrderedDict()
    for r in records:
        k = key_fn(r)
        if k is None: continue
        b = buckets.setdefault(k, {"n": 0, "chip_sum": 0.0, "icm_sum": 0.0,
                                    "chip_loss": 0.0, "icm_loss": 0.0,
                                    "chip_win": 0.0, "icm_win": 0.0})
        b["n"] += 1
        b["chip_sum"] += r["hero_chip_pl"]; b["icm_sum"] += r["hero_icm_pl"]
        if r["hero_icm_pl"] < 0:
            b["icm_loss"] += r["hero_icm_pl"]
            b["chip_loss"] += r["hero_chip_pl"]
        else:
            b["icm_win"] += r["hero_icm_pl"]
            b["chip_win"] += r["hero_chip_pl"]
    print(f"\n=== by {name} ===")
    print(f"{'bucket':<14} {'n':>5} {'mean_chip':>10} {'mean_icm':>10} "
          f"{'Σ_icm':>10} {'Σ_icm_loss':>11} {'%loss':>7}")
    rows = sorted(buckets.items(), key=lambda kv: kv[1]["icm_sum"])
    out = []
    for k, b in rows:
        mc = b["chip_sum"]/b["n"] if b["n"] else 0.0
        mi = b["icm_sum"]/b["n"] if b["n"] else 0.0
        pct = (100.0 * b["icm_loss"] / total_icm_loss) if total_icm_loss < 0 else 0.0
        print(f"{str(k):<14} {b['n']:>5} {mc:>+10.2f} {mi:>+10.4f} "
              f"{b['icm_sum']:>+10.3f} {b['icm_loss']:>+11.3f} {pct:>6.1f}%")
        out.append({"bucket": k, **b, "mean_chip": mc, "mean_icm": mi, "pct_of_loss": pct})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--net", default=NET)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--kiters", type=int, default=150)
    ap.add_argument("--weighting", default="linear")
    ap.add_argument("--warmstart", type=int, default=1)
    ap.add_argument("--belief", default="uniform")
    ap.add_argument("--policy-net", default=None)
    ap.add_argument("--out-jsonl", default="evals/diag_killphil.jsonl")
    ap.add_argument("--out-json", default="evals/diag_killphil_summary.json")
    a = ap.parse_args()

    structure = TournamentStructure.from_yaml(STRUCT)
    abstraction = Abstraction.load(ABSTR)
    solver = _load_solver(CKPT, abstraction, structure)
    payouts = list(sng_payouts_6max_double_up())

    if a.net == "k200":
        hero = BlueprintHero(solver); hero_label = "k200"
    else:
        cck = torch.load(a.net, map_location="cpu", weights_only=False)
        cnet = mlp(cck["in_dim"], tuple(cck["hidden"]))
        cnet.load_state_dict({kk.replace("net.", "", 1): vv for kk, vv in cck["state_dict"].items()})
        cnet.eval()
        pnet = pck = None
        if a.policy_net:
            from scripts.rebel_gate2 import _load_policy_net
            pnet, pck = _load_policy_net(a.policy_net)
        hero = ReBeLHero(solver, abstraction, cnet, cck, payouts,
                        depth=a.depth, n_iters=a.kiters,
                        weighting=a.weighting, warm_start=bool(a.warmstart),
                        belief=a.belief, policy_net=pnet, policy_ck=pck)
        hero_label = os.path.basename(a.net).replace(".pt","")
        hero_label += f"_d{a.depth}k{a.kiters}{a.weighting[0]}"
        if pnet is not None: hero_label += "_pn"

    factory, by_level = build_killphil_factory(structure)
    print(f"DIAG vs KillPhilMTT — hero={hero_label}, {a.hands} hands, "
          f"depth={a.depth} kiters={a.kiters} belief={a.belief}", flush=True)

    records = []
    os.makedirs(os.path.dirname(a.out_jsonl) or ".", exist_ok=True)
    f = open(a.out_jsonl, "w")
    t0 = __import__("time").perf_counter()
    print_every = max(50, a.hands // 20)

    for i in range(a.hands):
        base = a.seed + i * 7919
        sm = sample_starting_state(structure, random.Random(base * 2 + 1), num_paid=3)
        alive = [s for s in range(6) if sm["stacks"][s] > 0]
        hero_seat = random.Random(base * 3 + 5).choice(alive)
        dealer = sm["dealer_seat"]
        gs = structure.to_inner_game_string_for_state(
            blind_level=sm["blind_level"], stacks=sm["stacks"], dealer_seat=dealer)
        opps = factory(sm["blind_level"].level)
        bb = sm["blind_level"].inflated_big_blind(6)
        hero.new_hand(dealer, bb=bb)
        r = play_hand_diag(gs, sm, hero_seat, hero, opps, dealer, base, payouts, bb)
        if r is None: continue
        r["hero_seat"] = hero_seat; r["dealer_seat"] = dealer
        records.append(r); f.write(json.dumps(r) + "\n"); f.flush()
        if (i + 1) % print_every == 0:
            elapsed = __import__("time").perf_counter() - t0
            rate = (i+1) / elapsed
            eta = (a.hands - (i+1)) / max(rate, 1e-9)
            mean_icm = statistics.mean(rr["hero_icm_pl"] for rr in records)
            print(f"  [{i+1}/{a.hands}] {rate:.1f}h/s ETA {eta/60:.1f}min  "
                  f"mean_icm_pl={mean_icm:+.4f}", flush=True)

    f.close()
    if not records:
        print("no completed hands!"); return

    n = len(records)
    mc = statistics.mean(r["hero_chip_pl"] for r in records)
    mi = statistics.mean(r["hero_icm_pl"] for r in records)
    sd = statistics.pstdev(r["hero_icm_pl"] for r in records)
    se = sd / (n ** 0.5)
    total_icm_loss = sum(r["hero_icm_pl"] for r in records if r["hero_icm_pl"] < 0)
    total_icm_win = sum(r["hero_icm_pl"] for r in records if r["hero_icm_pl"] > 0)
    print(f"\n=== SUMMARY ({n} hands vs KillPhilMTT) ===")
    print(f"mean_chip_pl = {mc:+.2f}    mean_icm_pl = {mi:+.4f} ± {2*se:.4f}  (2·SE)")
    print(f"Σ_icm = {sum(r['hero_icm_pl'] for r in records):+.3f}    "
          f"Σ_icm_loss = {total_icm_loss:+.3f}    Σ_icm_win = {total_icm_win:+.3f}")
    print(f"loss/win hand-share = {sum(1 for r in records if r['hero_icm_pl']<0)/n:.1%} "
          f"/ {sum(1 for r in records if r['hero_icm_pl']>0)/n:.1%}")

    summary = {"hero": hero_label, "hands": n, "mean_chip_pl": mc,
               "mean_icm_pl": mi, "se_icm": se,
               "sum_icm": float(sum(r['hero_icm_pl'] for r in records)),
               "sum_icm_loss": float(total_icm_loss),
               "sum_icm_win": float(total_icm_win), "slices": {}}

    end_label = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
    summary["slices"]["ended_street"] = slice_table(records,
        lambda r: end_label.get(r["ended_street"], "?"), "ended_street", total_icm_loss)
    summary["slices"]["stack_bucket"] = slice_table(records,
        lambda r: r["stack_bucket"], "starting stack (bb)", total_icm_loss)
    summary["slices"]["position"] = slice_table(records,
        lambda r: r["position"], "hero position", total_icm_loss)
    summary["slices"]["voluntary_in"] = slice_table(records,
        lambda r: "vpip=Y" if r["hero_voluntary_in"] else "vpip=N",
        "voluntarily put chips in pot", total_icm_loss)
    summary["slices"]["all_in"] = slice_table(records,
        lambda r: "allin=Y" if r["all_in"] else "allin=N",
        "hero all-in at any point", total_icm_loss)
    summary["slices"]["blind_level"] = slice_table(records,
        lambda r: f"L{r['blind_level']}", "blind level", total_icm_loss)
    summary["slices"]["alive_at_start"] = slice_table(records,
        lambda r: f"{r['alive_at_start']}-alive", "alive players at hand start",
        total_icm_loss)
    summary["slices"]["last_action"] = slice_table(records,
        lambda r: "fold" if r["hero_last_action_was_fold"]
                  else ("aggr" if r["hero_last_action_was_aggressive"] else "pass"),
        "hero's LAST action this hand", total_icm_loss)

    # Concentration metric: what % of total icm-loss is in the worst K=3 buckets
    # of (position × ended_street × stack_bucket)?
    cross_buckets = collections.Counter()
    cross_icm = collections.defaultdict(float)
    cross_n = collections.Counter()
    for r in records:
        k = (r["position"], end_label.get(r["ended_street"], "?"), r["stack_bucket"])
        cross_buckets[k] += 1
        cross_icm[k] += r["hero_icm_pl"]
        cross_n[k] += 1
    losers = sorted(cross_icm.items(), key=lambda kv: kv[1])[:10]
    print(f"\n=== top-10 most-losing CROSS buckets (position × ended_street × stack_bucket) ===")
    print(f"{'bucket':<28} {'n':>5} {'Σ_icm':>10} {'mean':>10} {'%total_loss':>11}")
    for k, v in losers:
        mean = v / max(cross_n[k], 1)
        pct = (100.0 * v / total_icm_loss) if total_icm_loss < 0 else 0.0
        bs = f"{k[0]}/{k[1]}/{k[2]}"
        print(f"{bs:<28} {cross_n[k]:>5} {v:>+10.3f} {mean:>+10.4f} {pct:>10.1f}%")
    summary["top10_loss_buckets"] = [{"pos": k[0], "ended": k[1], "stack": k[2],
                                      "n": cross_n[k], "sum_icm": v}
                                     for k, v in losers]

    with open(a.out_json, "w") as ff:
        json.dump(summary, ff, indent=2)
    print(f"\nwrote {a.out_jsonl} + {a.out_json}")


if __name__ == "__main__":
    main()
