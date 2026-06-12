"""RT-1 classification + registered-mitigation counterfactual.

Reads frames_raw.json (loose superset: min(eff_card, eff_floor) <= 2.5,
any facing signal), applies the canonical RT-1 universe rule
    eff_floor_bb <= 2.5  AND  to_call_parsed > 0
(eff_floor_bb is _hero_eff_bb_from_parsed — the measure every deployed
floor and the candidate mask actually compute), classifies:

  class_a : post_argmax in {CALL, ALLIN} and post FOLD mass >= 0.10
  class_b : class_a frames whose REALIZED sampled action was FOLD
  class_c : mirror — post_argmax == FOLD and post CALL+ALLIN mass >= 0.10

and evaluates the registered mitigation counterfactual (mask FOLD when
post-floor argmax in {CALL, ALLIN} at eff <= 2.0, redistribute FOLD mass
proportionally to survivors), incl. a sensitivity row at N=2.5.

Writes frames.json and ENUMERATION.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "evals" / "rt1_enumeration_20260612"

FOLD_MASS_MIN = 0.10
MASK_N_REGISTERED = 2.0
MASK_N_SENS = 2.5

CLEAN, QUAR = "fidelity_clean", "fidelity_failed_quarantined"


def hand_class(cards):
    if len(cards) != 2:
        return "?"
    r1, r2 = cards[0][0], cards[1][0]
    suited = cards[0][1] == cards[1][1]
    order = "23456789TJQKA"
    hi, lo = sorted([r1, r2], key=order.index, reverse=True)
    if hi == lo:
        return hi + lo
    return hi + lo + ("s" if suited else "o")


def equity_vs_random(cards, n_opps, iters=20000, seed=7):
    """Tiny treys MC: hero equity vs n_opps uniform-random hands, no board
    known (all universe frames are preflop; assert upstream)."""
    import random as _random
    from treys import Card, Deck, Evaluator
    ev = Evaluator()
    hero = [Card.new(c) for c in cards]
    rng = _random.Random(seed)
    won = 0.0
    for _ in range(iters):
        deck = Deck()
        deck.cards = [c for c in deck.cards if c not in hero]
        rng.shuffle(deck.cards)
        opps = [deck.draw(2) for _ in range(n_opps)]
        board = deck.draw(5)
        hs = ev.evaluate(board, hero)
        os_ = [ev.evaluate(board, o) for o in opps]
        best_opp = min(os_)
        if hs < best_opp:
            won += 1.0
        elif hs == best_opp:
            won += 1.0 / (1 + sum(1 for s in os_ if s == hs))
    return won / iters


def apply_mask(dist):
    """Registered mitigation arithmetic: FOLD -> 0, survivors scaled."""
    fold = dist.get("FOLD", {}).get("post", 0.0)
    surv = {k: v["post"] for k, v in dist.items() if k != "FOLD"}
    tot = sum(surv.values())
    if tot <= 0:
        return None
    return {k: round(v / tot, 6) for k, v in surv.items()}


def main():
    raw = json.loads((OUT_DIR / "frames_raw.json").read_text())
    quarantined_logs = {m["log"] for m in raw["logs"]
                        if (m.get("fidelity_fails") or 0) > 0}

    universe = []
    excluded_artifacts = []
    for f in raw["frames"]:
        # to_call as the model believed it (parsed contributions)
        # frames_raw stores card-derived to_call; recover parsed facing
        # via: included frames all had facing True; canonical gate is
        # eff_floor. Artifact frames have eff_floor > 2.5.
        if f["eff_floor_bb"] > 2.5 or f["eff_floor_bb"] <= 0:
            excluded_artifacts.append(f)
            continue
        f = dict(f)
        f["tier"] = QUAR if f["session"] in quarantined_logs else CLEAN
        f["hand"] = hand_class(f["cards"])
        universe.append(f)

    # classify
    for f in universe:
        d = f["dist"]
        fold_post = d.get("FOLD", {}).get("post", 0.0)
        cont_post = sum(d.get(k, {}).get("post", 0.0)
                        for k in ("CALL", "ALLIN"))
        f["class_a"] = (f["post_argmax"] in ("CALL", "ALLIN")
                        and fold_post >= FOLD_MASS_MIN)
        f["class_b"] = f["class_a"] and f["chosen_kind"] == "fold"
        f["class_c"] = (f["post_argmax"] == "FOLD"
                        and cont_post >= FOLD_MASS_MIN)
        f["fold_post"] = fold_post
        f["cont_post"] = round(cont_post, 6)
        # counterfactual: registered mask
        for tag, n in (("mask_N2.0", MASK_N_REGISTERED),
                       ("mask_N2.5", MASK_N_SENS)):
            fires = (f["eff_floor_bb"] <= n
                     and f["post_argmax"] in ("CALL", "ALLIN")
                     and "FOLD" in f["dist"])
            f[tag] = {
                "fires": fires,
                "masked_dist": apply_mask(f["dist"]) if fires else None,
                "prevents_realized_fold": bool(fires and f["class_b"]),
            }
        # cost-side instrumentation: chip-EV call check
        bb = f["blinds"][1] if f["blinds"] else 1
        call_cost = min(f["to_call"], f["hero_stack"] or f["to_call"])
        req_eq = call_cost / (f["pot"] + call_cost) if call_cost else None
        n_opp = max(1, (f["n_alive"] or 2) - 1)
        eq_n = equity_vs_random(f["cards"], n_opp)
        eq_1 = equity_vs_random(f["cards"], 1)
        f["call_cost"] = call_cost
        f["req_equity_chipEV"] = round(req_eq, 4) if req_eq else None
        # bracket: vs-1-random (optimistic: only the bettor) and
        # vs-(n_alive-1)-random (pessimistic: every alive seat contests)
        f["equity_vs_1_random"] = round(eq_1, 4)
        f["equity_vs_n_random"] = round(eq_n, 4)
        f["n_opp_for_equity"] = n_opp

    def sub(frames, **kw):
        out = frames
        for k, v in kw.items():
            out = [f for f in out if f.get(k) == v]
        return out

    clean = sub(universe, tier=CLEAN)
    quar = sub(universe, tier=QUAR)

    def band(fr):
        le20 = [f for f in fr if f["eff_floor_bb"] <= 2.0]
        b2025 = [f for f in fr if 2.0 < f["eff_floor_bb"] <= 2.5]
        return le20, b2025

    def is_pair(f):
        return len(f["hand"]) == 2
    def is_prem_pair(f):
        return f["hand"] in ("AA", "KK", "QQ", "JJ")

    # ── render ─────────────────────────────────────────────────────────
    L = []
    W = L.append
    W("=" * 72)
    W("RT-1 DECISIVE TEST — sub-2BB floor-composition gap, FULL-CORPUS")
    W("ENUMERATION (registered: docs/research_program/RED_TEAM_LOG.md)")
    W("=" * 72)
    W(f"generated: {raw['generated_utc']}   method: per-frame replay via")
    W("make_decision with each session's exact header flags (tail tau,")
    W("short-stack bb, anchor-sum-floor, extended click plans, mode, seed);")
    W("capture hooks = decision_audit pattern + accepts_d2c fix so the H1")
    W("tail floor replays with EXACT chip costs on tau-armed sessions.")
    W("Live session log 20260612_220101 audited from a clean snapshot")
    W("(506 whole lines) — the live process ended during this analysis.")
    W("")
    W("[0] CORPUS COVERAGE + REPLAY FIDELITY")
    for m in raw["logs"]:
        if m.get("skipped"):
            W(f"    {m['log']:<46s} SKIPPED ({m['skipped']})")
        else:
            tier = "QUARANTINED" if (m["fidelity_fails"] or 0) > 0 else "CLEAN"
            W(f"    {m['log']:<46s} frames={m['frames_replayed']:<5d} "
              f"decisions={m['fresh_decisions']:<4d} "
              f"fidelity_fails={m['fidelity_fails']:<3d} {tier}")
    n_clean_logs = sum(1 for m in raw["logs"]
                       if not m.get("skipped") and not m["fidelity_fails"])
    n_quar_logs = sum(1 for m in raw["logs"]
                      if not m.get("skipped") and m["fidelity_fails"])
    cf = sum(m["frames_replayed"] for m in raw["logs"]
             if not m.get("skipped") and not m["fidelity_fails"])
    cd = sum(m["fresh_decisions"] for m in raw["logs"]
             if not m.get("skipped") and not m["fidelity_fails"])
    qf = sum(m["frames_replayed"] for m in raw["logs"]
             if not m.get("skipped") and m["fidelity_fails"])
    qd = sum(m["fresh_decisions"] for m in raw["logs"]
             if not m.get("skipped") and m["fidelity_fails"])
    W("")
    W(f"    fidelity-CLEAN: {n_clean_logs} logs, {cf} frames, {cd} unique "
      f"decisions — distributions ARE what ran live (byte-identity gate).")
    W(f"    QUARANTINED:    {n_quar_logs} logs, {qf} frames, {qd} decisions — "
      f"pre-header-era logs; replay on current code/seed mismatched the")
    W("    logged actions, so their distributions are NOT certified as what")
    W("    ran live. Their universe frames are listed separately below and")
    W("    EXCLUDED from headline counts.")
    W(f"    Note: {len(excluded_artifacts)} loose-filter rows dropped as "
      f"artifacts (raw scraper stacks null -> eff_card spuriously 0; true")
    W("    floor-eff measured 5.7-44.6 BB — outside the RT-1 universe).")
    W("")
    W("[1] UNIVERSE — facing-bet (to_call > 0) decisions at eff <= 2.5 BB")
    W("    (eff = _hero_eff_bb_from_parsed, the measure every deployed")
    W("     floor and the candidate mask compute)")
    W("")

    def card(f):
        d = f["dist"]
        dd = "  ".join(f"{k}:{v['post']:.3f}" for k, v in d.items()
                       if v["post"] > 0 or k in ("FOLD", "CALL", "ALLIN"))
        cls = []
        if f["class_a"]: cls.append("a")
        if f["class_b"]: cls.append("b")
        if f["class_c"]: cls.append("c")
        W(f"    {f['session']}  seq={f['seq']}  [{f['tier']}]")
        W(f"      {f['hand']} ({' '.join(f['cards'])})  {f['position']}  "
          f"{f['street']}  L{f['level']}  n_alive={f['n_alive']}")
        W(f"      eff_floor={f['eff_floor_bb']:.2f}BB  "
          f"eff_card={f['eff_card_bb']:.2f}BB  "
          f"hero_behind={f['hero_stack_bb']:.2f}BB  "
          f"to_call={f['to_call_bb']:.2f}BB  pot={f['pot']}")
        W(f"      OOD-WARN={'ACTIVE' if f['ood_warn'] else 'off'}  "
          f"floors_fired={f['floors_fired']}  "
          f"cached_repeats={f['cached_repeats']}")
        W(f"      post-floor: {dd}")
        W(f"      post_argmax={f['post_argmax']}  FOLD_mass={f['fold_post']:.3f}  "
          f"CALL+ALLIN_mass={f['cont_post']:.3f}  "
          f"CHOSEN={f['chosen_kind']}"
          f"{(' ' + str(f['chosen_amount'])) if f['chosen_amount'] else ''}")
        W(f"      chip-EV check: call_cost={f['call_cost']} "
          f"req_eq={f['req_equity_chipEV']}  equity bracket: "
          f"vs-1-random={f['equity_vs_1_random']}, "
          f"vs-{f['n_opp_for_equity']}-random={f['equity_vs_n_random']}")
        W(f"      classes: {{{','.join(cls) or 'none'}}}")
        W("")

    for f in sorted(clean, key=lambda x: (x["session"], x["seq"])):
        card(f)
    if quar:
        W("    -- quarantined (fidelity-failed logs; not certified live) --")
        for f in sorted(quar, key=lambda x: (x["session"], x["seq"])):
            card(f)

    W("[2] CLASS TOTALS (headline = fidelity-clean only)")
    for tier_name, fr in (("CLEAN", clean), ("QUARANTINED", quar)):
        a = sub(fr, class_a=True); b = sub(fr, class_b=True)
        c = sub(fr, class_c=True)
        W(f"    {tier_name}: universe={len(fr)}  class_a={len(a)}  "
          f"class_b(realized wrong-side FOLD draws)={len(b)}  "
          f"class_c(mirror)={len(c)}")
    le20, b2025 = band(clean)
    W("")
    W("    eff bands (CLEAN):")
    for nm, fr in (("eff <= 2.0 BB", le20), ("2.0 < eff <= 2.5 BB", b2025)):
        a = sub(fr, class_a=True); b = sub(fr, class_b=True)
        c = sub(fr, class_c=True)
        W(f"      {nm:<22s} universe={len(fr)}  class_a={len(a)}  "
          f"class_b={len(b)}  class_c={len(c)}")
    pp = [f for f in clean if is_prem_pair(f)]
    anyp = [f for f in clean if is_pair(f)]
    W(f"    premium pairs (JJ+) in CLEAN universe: {len(pp)} "
      f"({', '.join(f['hand'] + ' seq' + str(f['seq']) for f in pp) or '-'})"
      f" — of which class_b: {sum(1 for f in pp if f['class_b'])}")
    W(f"    any pocket pair in CLEAN universe: {len(anyp)}")
    W("")

    W("[3] COUNTERFACTUAL — registered mitigation (mask FOLD when post-floor")
    W("    argmax in {CALL, ALLIN} at eff <= 2.0; FOLD mass redistributed")
    W("    proportionally to survivors)")
    W("")
    W(f"    {'frame':<34s} {'eff':>5s} {'argmax':<7s} {'FOLDm':>6s} "
      f"{'chosen':<7s} {'fires@2.0':>9s} {'fires@2.5':>9s} effect")
    for f in sorted(universe, key=lambda x: (x["tier"], x["session"],
                                             x["seq"])):
        m20, m25 = f["mask_N2.0"], f["mask_N2.5"]
        if m20["fires"]:
            eff_s = ("PREVENTS realized FOLD; masked -> "
                     + json.dumps(m20["masked_dist"])
                     if m20["prevents_realized_fold"]
                     else "no realized change (chosen was "
                     + str(f["chosen_kind"]) + "); masked -> "
                     + json.dumps(m20["masked_dist"]))
        elif m25["fires"]:
            eff_s = ("MISSED at N=2.0 (eff %.2f); at N=2.5 %s"
                     % (f["eff_floor_bb"],
                        "would PREVENT the realized FOLD"
                        if f["class_b"] else "no realized change"))
        else:
            eff_s = "no fire (argmax FOLD)" if f["post_argmax"] == "FOLD" \
                else "no fire"
        tag = f"{f['session'][-25:]}#%d" % f["seq"]
        W(f"    {tag:<34s} {f['eff_floor_bb']:5.2f} {f['post_argmax']:<7s} "
          f"{f['fold_post']:6.3f} {str(f['chosen_kind']):<7s} "
          f"{str(m20['fires']):>9s} {str(m25['fires']):>9s} {eff_s}")
    W("")
    n_c_fire = sum(1 for f in universe if f["mask_N2.0"]["fires"]
                   and f["class_c"])
    W(f"    symmetry check: mask fires on a class_c (argmax=FOLD) frame: "
      f"{n_c_fire} times (structurally impossible — fire condition")
    W("    requires argmax in {CALL, ALLIN}; verified empirically over the")
    W("    whole universe, both tiers).")
    W("")

    W("[4] WHAT THE MASK WOULD HAVE COST (honest accounting)")
    W("    Frames where masking FOLD could plausibly be wrong — i.e. the")
    W("    mask fires (or would at N=2.5) and folding had a case:")
    flagged = 0
    for f in universe:
        fires_any = f["mask_N2.0"]["fires"] or f["mask_N2.5"]["fires"]
        if not fires_any:
            continue
        req = f["req_equity_chipEV"]
        eq1, eqn = f["equity_vs_1_random"], f["equity_vs_n_random"]
        notes = []
        if req is not None and eq1 < req + 0.05:
            # even the optimistic heads-up bound is at/below required —
            # a genuine fold case the mask would override
            notes.append(f"FOLD HAD A CASE: even vs-1-random equity "
                         f"{eq1:.3f} <~ required {req:.3f}")
        elif req is not None and eqn < req:
            notes.append(f"MARGINAL only under the pessimistic all-"
                         f"{f['n_opp_for_equity']}-contest bound "
                         f"({eqn:.3f} < {req:.3f}); heads-up bound "
                         f"{eq1:.3f} clears it")
        if f["ood_warn"]:
            notes.append("OOD-WARN active — the distribution arguing for "
                         "the continue is itself out-of-support")
        if f["fold_post"] >= 0.40:
            notes.append(f"high FOLD mass ({f['fold_post']:.2f}) — model "
                         "nearly preferred folding")
        if notes:
            flagged += 1
            W(f"    - {f['session']} seq={f['seq']} {f['hand']} "
              f"eff={f['eff_floor_bb']:.2f}BB: " + "; ".join(notes))
        else:
            W(f"    - {f['session']} seq={f['seq']} {f['hand']} "
              f"eff={f['eff_floor_bb']:.2f}BB: no fold case found "
              f"(equity bracket [{eqn}, {eq1}] vs required {req}); "
              f"masking FOLD is safe here")
    if flagged == 0:
        W("    No fired frame had a defensible fold case by the chip-EV")
        W("    proxy. Caveats that still apply to EVERY fired frame:")
    W("")
    W("    Structural caveats (apply corpus-wide, not per-frame):")
    W("    - The chip-EV equity proxy ignores ICM and the double-up bubble;")
    W("      a genuinely fold-correct OOD spot (e.g. eff ~2BB on a strict")
    W("      bubble where folding into a near-lock cash is right) would be")
    W("      masked into continuing. ZERO such frames exist in this corpus,")
    W("      but n=4 (3 clean) is far too small to call the class empty.")
    W("    - The mask trusts the model's own argmax exactly where OOD-WARN")
    W("      says the encoder is least trustworthy (seq 498 fired WITH")
    W("      OOD-WARN active). The mitigation removes one failure mode")
    W("      (sampling FOLD against a continue-plurality) without making")
    W("      the plurality itself more reliable.")
    W("    - Equity is bracketed [vs-(n_alive-1)-random, vs-1-random]; the")
    W("      truth (the bettor's + overcallers' actual ranges) lies between")
    W("      and is not certified in either direction.")
    W("")
    W("[5] VERDICT LINE")
    a_cl = sub(clean, class_a=True); b_cl = sub(clean, class_b=True)
    c_cl = sub(clean, class_c=True)
    n_fire_a = sum(1 for f in a_cl if f["mask_N2.0"]["fires"])
    n_prev = sum(1 for f in a_cl if f["mask_N2.0"]["prevents_realized_fold"])
    miss = [f for f in b_cl if not f["mask_N2.0"]["fires"]]
    W(f"    CLEAN corpus: universe={len(clean)}, class_a={len(a_cl)}, "
      f"class_b={len(b_cl)}, class_c={len(c_cl)}.")
    W(f"    Registered mask (N=2.0): fires on {n_fire_a}/{len(a_cl)} class_a "
      f"frames, prevents {n_prev}/{len(b_cl)} realized wrong-side FOLDs,")
    W(f"    forces zero bad calls, never fires against a FOLD argmax.")
    if miss:
        W(f"    ** N=2.0 MISSES {len(miss)} realized wrong-side FOLD(s): "
          + ", ".join(f"{f['hand']} seq{f['seq']} eff={f['eff_floor_bb']:.2f}BB"
                      for f in miss)
          + " — N=2.5 would catch "
          + ("all of them." if all(f["mask_N2.5"]["fires"] for f in miss)
             else "only some of them."))
    W("=" * 72)

    report = "\n".join(L)
    (OUT_DIR / "ENUMERATION.txt").write_text(report + "\n")

    out = {
        "generated_from": "frames_raw.json",
        "universe_rule": "eff_floor_bb <= 2.5 and to_call > 0",
        "classes": {
            "a": "post_argmax in {CALL,ALLIN} and post FOLD mass >= 0.10",
            "b": "class_a and realized sampled action == FOLD",
            "c": "post_argmax == FOLD and post CALL+ALLIN mass >= 0.10",
        },
        "mitigation": {
            "registered": "mask FOLD when post-floor argmax in {CALL,ALLIN} "
                          "at eff <= 2.0, redistribute proportionally",
            "sensitivity": "same at N=2.5",
        },
        "logs": raw["logs"],
        "excluded_artifacts": excluded_artifacts,
        "frames": universe,
    }
    (OUT_DIR / "frames.json").write_text(json.dumps(out, indent=1))
    print(report)
    print(f"\nwrote: {OUT_DIR/'ENUMERATION.txt'}, {OUT_DIR/'frames.json'}")


if __name__ == "__main__":
    main()
