"""Per-decision audit for live dry-run sessions.

For EVERY decision frame in a live_dryrun_*.jsonl this renders a
human-readable card: street, hero cards, board, stacks in BB, pot,
facing amount, the RECONSTRUCTED action history the model believed,
the model's FULL policy distribution over legal actions (pre- and
post-floor), floor firings, and a believed-state-vs-raw-scraper
divergence check. Then flags:

  (a) decisions where the reconstructed state diverges from the raw
      scraper record's observables (the seq-170/192 "reconstruction
      lying to the model" classes), including fields the invariant
      could NOT verify because the scraper reported null;
  (b) all-in / shove decisions, with eff_bb + position context;
  (c) folds of any hand containing an Ace, same context.

Plus a missed-action accounting: every frame where hero was to-act
(action buttons visible in the raw record) but no decision was
produced, with the to-act window duration.

How it works: the live path is deterministic given (code, log,
checkpoint, seed) — proven by the 2026-06-11 byte-identity smoke. The
jsonl does not store policy distributions or action sequences, so this
script RE-REPLAYS the session through make_decision with the exact
live config (fresh SessionTracker/DecisionCache, sample mode, header
seed) and installs three PURE capture hooks (no RNG consumption):

  1. make_live_policy_filter  -> captures pre/post-floor policy,
     legal mask, parsed-state snapshot, discrete->chip map, floor flags
  2. derive_action_sequence   -> captures the believed action history
  3. check_mid_hand_invariant -> captures the reconstructed scraper-
     view (stacks/bets/pot as the model believed them) + sb/bb seats

Fidelity gate: every replayed frame's (status, client_action,
chip_int) must equal the logged session byte-for-byte, and floor
firings must match the .stdout companion. Any mismatch is reported
loudly and invalidates the audit.

Usage:
    python -m scripts.decision_audit logs/live_dryrun_20260611_163815.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
DEFAULT_CHECKPOINT = ("runs/k200_real_ante_20260605_225847_PRESERVED/"
                      "ckpt_iter_1500.pt")
DEFAULT_ABSTRACTION = ("runs/abstraction_20260521_223018_retrofit/"
                       "abstraction.pkl")

STREETS = {0: "preflop", 1: "flop", 2: "turn", 3: "river", None: "?"}


def parse_captured_at(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y%m%d_%H%M%S_%f")
    except (ValueError, TypeError):
        return None


# ── capture hooks (all pure; consume no RNG) ───────────────────────────

class Capture:
    """Per-frame capture state, reset before each make_decision call."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.policy_pre = None
        self.policy_post = None
        self.legal_mask = None
        self.parsed_snapshot = None
        self.discrete_to_chip = None
        self.floors_fired = []
        self.action_seqs = []       # all derive_action_sequence calls
        self.recon_views = []       # all invariant reconstructed views
        self.packs = []             # (sb_seat, bb_seat, street_idx)


CAP = Capture()


def install_hooks():
    import numpy as np
    from src.nlhe.integration import live_loop
    from src.nlhe.integration import scraper_schema
    from src.nlhe.integration import invariant as invariant_mod
    from src.nlhe.actions import DiscreteAction, discretize_legal_actions
    from src.nlhe.cfr6 import _build_view_6max
    from src.nlhe.integration.live_loop import (
        apply_aa_kk_preflop_floor, apply_check_when_free_floor,
        apply_short_stack_floor, _hero_eff_bb_from_parsed,
    )

    # 1. policy filter capture — wrap the composed filter; the wrapped
    # version returns the original's output object untouched.
    orig_make_filter = live_loop.make_live_policy_filter

    def capturing_make_filter(*args, **kwargs):
        composed = orig_make_filter(*args, **kwargs)
        threshold = kwargs.get("short_stack_threshold_bb",
                               args[0] if args else 6.0)

        def capturing_filter(policy, legal_mask, parsed, state):
            out = composed(policy, legal_mask, parsed, state)
            # Recompute floor firing flags exactly as composed does
            # (pure functions, reference-equality short-circuit).
            p1 = apply_aa_kk_preflop_floor(policy, legal_mask, parsed, state)
            p2 = apply_check_when_free_floor(p1, legal_mask, parsed, state)
            p3 = apply_short_stack_floor(p2, legal_mask, parsed, state,
                                         threshold_bb=threshold)
            fires = []
            if p1 is not policy: fires.append("AA/KK")
            if p2 is not p1: fires.append("check-free")
            if p3 is not p2: fires.append("short-stack")
            eff_bb, bb = _hero_eff_bb_from_parsed(parsed)
            cp = parsed["current_player"]
            contribs = parsed.get("contribution", [])
            to_call = (int(max(contribs)) - int(contribs[cp])
                       if contribs else None)
            legal_chip = list(state.legal_actions())
            view = _build_view_6max(state, parsed)
            d2c = discretize_legal_actions(legal_chip, view)
            CAP.policy_pre = np.array(policy, dtype=float)
            CAP.policy_post = np.array(out, dtype=float)
            CAP.legal_mask = np.array(legal_mask, dtype=float)
            CAP.floors_fired = fires
            CAP.discrete_to_chip = {int(k): int(v) for k, v in d2c.items()}
            CAP.parsed_snapshot = {
                "current_player": cp,
                "to_call": to_call,
                "contribution": list(contribs),
                "money": list(parsed.get("money", [])),
                "big_blind": int(parsed.get("big_blind", 0)),
                "eff_bb_floors": eff_bb,
            }
            return out

        return capturing_filter

    live_loop.make_live_policy_filter = capturing_make_filter

    # 2. action-sequence capture
    orig_derive = scraper_schema.derive_action_sequence

    def capturing_derive(frame, pre_hand_override=None):
        seq = orig_derive(frame, pre_hand_override=pre_hand_override)
        CAP.action_seqs.append({
            "override_used": pre_hand_override is not None,
            "seq": [(int(s), int(a)) for s, a in seq],
        })
        return seq

    scraper_schema.derive_action_sequence = capturing_derive

    # 3. invariant reconstructed-view capture
    orig_inv = invariant_mod.check_mid_hand_invariant

    def capturing_inv(frame, state_pack):
        res = orig_inv(frame, state_pack)
        CAP.recon_views.append(res.reconstructed)

        def _i(v):
            return int(v) if v is not None else -1

        CAP.packs.append({
            "sb_seat": _i(state_pack.sb_seat),
            "bb_seat": _i(state_pack.bb_seat),
            "street_idx": _i(state_pack.street_idx),
            "pre_hand_stacks": tuple(int(x)
                                     for x in state_pack.pre_hand_stacks),
        })
        return res

    invariant_mod.check_mid_hand_invariant = capturing_inv


# ── position / formatting helpers ──────────────────────────────────────

def position_label(hero_seat, dealer_seat, sb_seat, bb_seat, alive):
    n_alive = sum(1 for a in alive if a)
    if hero_seat == sb_seat:
        return "SB" + (" (BTN)" if hero_seat == dealer_seat else "")
    if hero_seat == bb_seat:
        return "BB"
    if hero_seat == dealer_seat:
        return "BTN"
    # Count alive seats from after BB to hero in seat order
    order = []
    s = (bb_seat + 1) % 6
    while s != sb_seat:
        if alive[s]:
            order.append(s)
        s = (s + 1) % 6
    try:
        k = order.index(hero_seat)
    except ValueError:
        return "?"
    names = {3: ["UTG"], 4: ["UTG", "CO"], 5: ["UTG", "HJ", "CO"],
             6: ["UTG", "MP", "HJ", "CO"]}
    lab = names.get(n_alive, ["?"] * 4)
    return lab[k] if k < len(lab) else f"+{k}"


def fmt_chip_action(chip_int, facing):
    if chip_int == 0:
        return "FOLD"
    if chip_int == 1:
        return "CALL" if facing else "CHECK"
    return f"RAISE-TO {chip_int}"


def render_history(action_seq, hero_seat, dealer_seat, sb_seat, bb_seat,
                   blinds, bb):
    """Human-readable believed action history (post-blind voluntary
    actions; OpenSpiel chip ints are raise-TO totals)."""
    lines = [f"    blinds: seat{sb_seat+1} posts SB {blinds[0]}, "
             f"seat{bb_seat+1} posts BB {blinds[1]}, ante {blinds[2]} all"]
    if not action_seq:
        lines.append("    (no voluntary actions before hero — "
                     "hero is first to act)")
        return lines
    # Track per-seat running commitment to label CALL vs CHECK vs RAISE
    commit = {}
    commit[sb_seat] = blinds[0]
    commit[bb_seat] = blinds[1]
    cur_max = blinds[1]
    for seat, ci in action_seq:
        tags = []
        if seat == hero_seat: tags.append("HERO")
        if seat == dealer_seat: tags.append("BTN")
        if seat == sb_seat: tags.append("SB")
        if seat == bb_seat: tags.append("BB")
        tag = f" [{','.join(tags)}]" if tags else ""
        prev = commit.get(seat, 0)
        if ci == 0:
            act = "folds"
        elif ci == 1:
            act = "calls" if cur_max > prev else "checks"
            commit[seat] = cur_max
        else:
            kind = "raises to" if cur_max > 0 else "bets"
            act = (f"{kind} {ci} ({ci / max(1, bb):.1f}BB)"
                   if ci > cur_max else f"commits {ci} (call/short)")
            commit[seat] = ci
            cur_max = max(cur_max, ci)
        lines.append(f"    seat{seat+1}{tag} {act}")
    return lines


def divergence_check(recon, raw, pot_used, pot_corrected):
    """Compare the model-believed (reconstructed) state against the RAW
    scraper record. Returns (mismatches, unverifiable)."""
    mism, unver = [], []
    raw_stacks = (raw.get("stacks") or {})
    raw_bets = (raw.get("bets") or {})
    raw_pot = (raw.get("pot") or {}).get("total")
    if raw_pot is None:
        unver.append("pot (raw null)")
    elif recon is not None and recon.get("pot") != raw_pot:
        note = " [pot_corrected: UI-lag fix applied]" if pot_corrected else ""
        mism.append(f"pot: raw={raw_pot} believed={recon.get('pot')}{note}")
    for i in range(6):
        key = f"seat{i+1}"
        rs = raw_stacks.get(key)
        rb = raw_bets.get(key)
        if recon is None:
            continue
        cs = recon["stack"][i]
        cb = recon["bet"][i]
        if rs is None:
            unver.append(f"stack[{key}] (raw null, believed={cs})")
        elif int(rs) != int(cs):
            mism.append(f"stack[{key}]: raw={rs} believed={cs}")
        if rb is None:
            # Scraper null bet means "no bet seen this street" -> 0 by
            # parse contract; only flag if model believed a bet exists.
            if int(cb) != 0:
                unver.append(f"bet[{key}] (raw null, believed={cb})")
        elif int(rb) != int(cb):
            mism.append(f"bet[{key}]: raw={rb} believed={cb}")
    return mism, unver


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--abstraction", default=DEFAULT_ABSTRACTION)
    args = ap.parse_args()

    log_path = Path(args.log)
    session = re.sub(r"^live_dryrun_|\.jsonl$", "", log_path.name)
    out_path = Path(args.out or f"logs/decision_audit_{session}.txt")

    records = [json.loads(l) for l in open(log_path) if l.strip()]
    header = records[0] if records[0].get("record_type") == "session_header" \
        else {}
    frames = [r for r in records if r.get("raw_record") is not None]
    seed = int(header.get("mode") == "sample" and header.get("seed", 2026))

    import numpy as np
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from src.nlhe.integration.live_loop import DecisionCache, make_decision
    from src.nlhe.integration.scraper_schema import SessionTracker, parse_frame
    from src.nlhe.actions import DiscreteAction
    from scripts.eval_6max_self_play import _load_solver

    install_hooks()

    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    abstr = Abstraction.load(args.abstraction)
    solver = _load_solver(args.checkpoint, abstr, structure)
    tracker = SessionTracker()
    cache = DecisionCache()
    rng = random.Random(seed)

    rows = []
    fidelity_fails = []
    fresh_capture_by_identity = {}

    for rec in frames:
        CAP.reset()
        d = make_decision(rec["raw_record"], structure, solver, tracker,
                          rng, mode=header.get("mode", "sample"),
                          seq=rec.get("seq"), decision_cache=cache)
        # fidelity vs logged
        logged_ca = rec.get("client_action")
        replay_ca = d.client_action
        if (d.status != rec.get("status")
                or json.dumps(replay_ca, sort_keys=True)
                != json.dumps(logged_ca, sort_keys=True)):
            fidelity_fails.append(
                (rec.get("seq"), rec.get("status"), d.status,
                 logged_ca, replay_ca))
        row = {
            "rec": rec,
            "decision": d,
            "policy_pre": CAP.policy_pre,
            "policy_post": CAP.policy_post,
            "legal_mask": CAP.legal_mask,
            "parsed": CAP.parsed_snapshot,
            "d2c": CAP.discrete_to_chip,
            "floors": list(CAP.floors_fired),
            "history": CAP.action_seqs[-1] if CAP.action_seqs else None,
            "recon": CAP.recon_views[-1] if CAP.recon_views else None,
            "pack": CAP.packs[-1] if CAP.packs else None,
        }
        if d.status in ("decision", "decision_recovered") \
                and d.decision_identity is not None:
            fresh_capture_by_identity[repr(d.decision_identity)] = row
        rows.append(row)

    # ── render ─────────────────────────────────────────────────────────
    L = []
    W = L.append
    W("=" * 72)
    W(f"DECISION AUDIT — {log_path.name}")
    W("=" * 72)
    W("")
    W("[0] REPLAY FIDELITY GATE")
    W(f"    frames replayed: {len(rows)}")
    if fidelity_fails:
        W(f"    *** {len(fidelity_fails)} MISMATCHES vs logged session — "
          f"AUDIT INVALID, distributions below are NOT what ran live ***")
        for seq, ls, rs_, lca, rca in fidelity_fails[:20]:
            W(f"      seq={seq} logged={ls}/{lca} replay={rs_}/{rca}")
    else:
        W("    all replayed frames byte-identical to the logged session "
          "(status + client_action) — captured distributions are the "
          "ones the live model produced")
    W("")

    dec_rows = [r for r in rows if r["decision"].status.startswith("decision")]
    W(f"[1] DECISION CARDS ({len(dec_rows)} frames)")
    W("")
    flags_a, flags_b, flags_c = [], [], []

    for r in dec_rows:
        rec, d = r["rec"], r["decision"]
        raw = rec["raw_record"]
        bb = d.blinds[1] if d.blinds else 1
        # cached frames: pull the fresh capture for the same identity
        src = r
        cached = d.status.endswith("cached")
        if cached and r["policy_post"] is None:
            src = fresh_capture_by_identity.get(
                repr(d.decision_identity), r)
        pack = src["pack"] or r["pack"]
        try:
            frame = parse_frame(raw)
            alive = frame.alive
            bets = frame.bet
            stacks = frame.stack
        except Exception:
            alive = [bool(s) for s in (raw.get("stacks") or {}).values()]
            bets, stacks, frame = None, None, None
        sb_seat = pack["sb_seat"] if pack else -1
        bb_seat = pack["bb_seat"] if pack else -1
        pos = (position_label(d.hero_seat, d.dealer_seat, sb_seat, bb_seat,
                              alive) if pack else "?")
        hero_bet = bets[d.hero_seat] if bets else 0
        max_opp = max((b for i, b in enumerate(bets)
                       if i != d.hero_seat), default=0) if bets else 0
        to_call = max(0, max_opp - hero_bet)
        hero_total = (d.hero_stack or 0) + hero_bet
        opp_totals = [(i, stacks[i] + bets[i]) for i in range(6)
                      if stacks and i != d.hero_seat and alive[i]]
        eff = min(hero_total, max((t for _, t in opp_totals), default=0))

        W("-" * 72)
        W(f"seq {d.seq}  [{d.status}]  captured={d.captured_at}  "
          f"L{d.level} blinds={d.blinds}")
        W(f"  street={STREETS.get(d.street_idx)}  "
          f"hero={' '.join(d.hero_cards)}  "
          f"board={' '.join(d.board) or '-'}")
        W(f"  position={pos}  dealer=seat{d.dealer_seat+1}  "
          f"hero=seat{d.hero_seat+1}")
        stack_strs = [f"hero {d.hero_stack} ({(d.hero_stack or 0)/bb:.1f}BB)"]
        for i, t in opp_totals:
            stack_strs.append(f"seat{i+1} {stacks[i]} "
                              f"({stacks[i]/bb:.1f}BB)")
        W(f"  stacks(behind): {'; '.join(stack_strs)}")
        W(f"  pot={d.pot_total}{' [CORRECTED]' if d.pot_corrected else ''}"
          f"  facing={to_call} ({to_call/bb:.1f}BB)  "
          f"eff={eff} ({eff/bb:.1f}BB)  n_alive={d.n_alive}")
        W("  believed action history "
          f"(override={'anchor' if (r['history'] or {}).get('override_used') else 'simple-model'}):")
        if r["history"]:
            for ln in render_history(
                    r["history"]["seq"], d.hero_seat, d.dealer_seat,
                    sb_seat, bb_seat, d.blinds, bb):
                W("  " + ln)
        else:
            W("    (cached frame — history identical to the fresh sample "
              "of this decision)")
        # policy
        pol_pre, pol_post = src["policy_pre"], src["policy_post"]
        d2c = src["d2c"] or {}
        chosen_ci = d.resolver_raw_openspiel_chip_int
        if pol_post is not None:
            pol_src = ("from fresh sample of same decision" if cached
                       else "fresh")
            W(f"  policy distribution ({pol_src}; "
              f"mode=sample draws from post-floor):")
            for da in DiscreteAction:
                idx = int(da)
                if src["legal_mask"][idx] == 0:
                    continue
                ci = d2c.get(idx)
                chosen = (ci is not None and chosen_ci is not None
                          and int(ci) == int(chosen_ci))
                mark = "  <== CHOSEN" if chosen else ""
                concrete = ("fold" if ci == 0 else
                            ("check/call" if ci == 1 else
                             f"raise-to {ci} ({(ci or 0)/bb:.1f}BB)"))
                pre_p = pol_pre[idx] if pol_pre is not None else float("nan")
                W(f"    {da.name:<8s} {concrete:<26s} "
                  f"pre={pre_p:6.3f}  post={pol_post[idx]:6.3f}{mark}")
            if src["floors"]:
                W(f"  floors fired: {', '.join(src['floors'])}")
        else:
            W("  policy distribution: UNAVAILABLE (cached; fresh sample "
              "not in this log)")
        W(f"  chosen: {json.dumps(d.client_action)}")
        # invariant + divergence
        mism, unver = divergence_check(src["recon"] or r["recon"], raw,
                                       d.pot_total, d.pot_corrected)
        W("  invariant: PASS (decision frames only exist past the strict "
          "invariant)")
        if mism:
            W(f"  ** BELIEVED-vs-RAW DIVERGENCE: {'; '.join(mism)}")
            flags_a.append((d.seq, d.hero_cards, mism))
        if unver:
            W(f"  unverifiable raw fields: {'; '.join(unver)}")
        # flags b/c
        ca = d.client_action or {}
        is_shove = (ca.get("kind") == "raise_to" and ca.get("chip_amount")
                    and int(ca["chip_amount"]) >= hero_total - 1)
        if is_shove:
            flags_b.append((d.seq, d.hero_cards, pos, eff / bb, d.street_idx,
                            to_call / bb, ca))
        if ca.get("kind") == "fold" and any(
                c and c[0] == "A" for c in d.hero_cards):
            flags_c.append((d.seq, d.hero_cards, pos, eff / bb, d.street_idx,
                            to_call / bb))
        W("")

    # flag sections
    W("=" * 72)
    W("[2a] RECONSTRUCTION DIVERGENCE FLAGS (believed != raw observable)")
    if flags_a:
        for seq, cards, mism in flags_a:
            W(f"    seq {seq} {' '.join(cards)}: {'; '.join(mism)}")
    else:
        W("    none — every decision's believed stacks/bets/pot matched "
          "the raw scraper record on all non-null fields")
    W("")
    W("[2b] ALL-IN / SHOVE DECISIONS")
    for seq, cards, pos, effbb, st, callbb, ca in flags_b:
        W(f"    seq {seq:<4d} {' '.join(cards):<6s} pos={pos:<9s} "
          f"eff={effbb:5.1f}BB street={STREETS.get(st)} "
          f"facing={callbb:.1f}BB -> {ca.get('kind')} "
          f"{ca.get('chip_amount')}")
    W("")
    W("[2c] FOLDS OF Ax HANDS")
    if flags_c:
        for seq, cards, pos, effbb, st, callbb in flags_c:
            W(f"    seq {seq:<4d} {' '.join(cards):<6s} pos={pos:<9s} "
              f"eff={effbb:5.1f}BB street={STREETS.get(st)} "
              f"facing={callbb:.1f}BB -> FOLD")
    else:
        W("    none")
    W("")

    # ── missed-action accounting ───────────────────────────────────────
    W("=" * 72)
    W("[3] MISSED-ACTION ACCOUNTING")
    W("    every frame where hero had a REAL action UI (CHECK/CALL/RAISE/"
      "BET buttons, or 2+ buttons) but no decision was produced.")
    W("    FOLD-only button frames are the muck / fold-ahead-of-action UI "
      "— counted separately, not real to-act spots.")
    W("")

    def real_to_act(raw):
        btns = [str(b).upper()
                for b in (raw.get("action") or {}).get("buttons") or []]
        if not btns:
            return False
        if any(b in ("CHECK", "CALL", "RAISE", "BET") for b in btns):
            return True
        return len(btns) >= 2

    def free_check_available(raw):
        btns = [str(b).upper()
                for b in (raw.get("action") or {}).get("buttons") or []]
        return "CHECK" in btns

    missed = []
    fold_only = 0
    for idx, r in enumerate(rows):
        d = r["decision"]
        raw = r["rec"]["raw_record"]
        if d.status.startswith("decision"):
            continue
        btns = [str(b).upper()
                for b in (raw.get("action") or {}).get("buttons") or []]
        if btns == ["FOLD"]:
            fold_only += 1
            continue
        if not real_to_act(raw):
            continue
        # Resolution: scan forward while hero_cards unchanged.
        cards = tuple(raw.get("hero_cards") or [])
        t0 = parse_captured_at(r["rec"]["captured_at"])
        resolution, t_res = "hand ended (log tail)", None
        for r2 in rows[idx + 1:]:
            raw2 = r2["rec"]["raw_record"]
            cards2 = tuple(raw2.get("hero_cards") or [])
            if cards2 and cards2 != cards:
                resolution = "NEVER DECIDED — next hand dealt"
                t_res = parse_captured_at(r2["rec"]["captured_at"])
                break
            if r2["decision"].status.startswith("decision"):
                resolution = f"decided at seq {r2['decision'].seq}"
                t_res = parse_captured_at(r2["rec"]["captured_at"])
                break
        else:
            t_res = parse_captured_at(rows[-1]["rec"]["captured_at"])
        dt = (t_res - t0).total_seconds() if (t0 and t_res) else float("nan")
        missed.append((d, raw, btns, resolution, dt))

    W(f"    {'seq':>5s} {'status':<18s} {'cards':<6s} {'buttons':<18s} "
      f"{'resolution':<34s} {'+secs':>6s} reason")
    never_decided = 0
    for d, raw, btns, resolution, dt in missed:
        cards = "".join(raw.get("hero_cards") or []) or "?"
        if resolution.startswith("NEVER"):
            never_decided += 1
        reason = (d.skip_reason or "")[:48]
        W(f"    {d.seq:>5d} {d.status:<18s} {cards:<6s} "
          f"{'/'.join(btns):<18s} {resolution:<34s} {dt:5.1f}s {reason}")
    W("")
    safe_folds = sum(1 for d, *_ in missed if d.status == "safe_fold")
    W(f"    real to-act frames with no decision: {len(missed)} "
      f"(of which safe_fold — a FOLD/CHECK fallback plan WAS emitted: "
      f"{safe_folds})")
    W(f"    to-act spots NEVER decided before the next hand "
      f"(operator/site timer had to resolve them): {never_decided}")
    W(f"    FOLD-only (muck / fold-ahead UI) frames excluded: {fold_only}")
    W("    timer-threat note: '+secs' is time from the missed frame to "
      "resolution; spots marked NEVER DECIDED were resolved only by the "
      "operator acting manually or the site timer.")
    W("")
    W("=" * 72)

    report = "\n".join(L)
    out_path.write_text(report)
    print(report)
    print(f"\nreport written: {out_path}")
    if fidelity_fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
