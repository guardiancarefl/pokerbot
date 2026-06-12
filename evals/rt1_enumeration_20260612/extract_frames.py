"""RT-1 decisive test — corpus enumeration of the sub-2BB floor-composition gap.

Replays every raw-record live_dryrun log through make_decision with the
EXACT per-session live config (header flags: short_stack_bb, tail_floor_tau,
anchor_sum_floor, extended_click_plans, mode, seed) and the decision_audit
capture-hook pattern, then dumps a JSON row for every unique DECISION with
eff <= 2.5 BB facing a bet (to_call > 0).

Differences from scripts/decision_audit.py (deliberate, both fidelity-load-bearing):
  1. The policy-filter capture hook sets accepts_d2c=True and forwards
     discrete_to_chip, so the H1 tail floor replays with EXACT chip costs
     (decision_audit's stock hook silently downgrades it to the approximate
     fallback on tau-armed sessions).
  2. Header flags are forwarded to make_decision / SessionTracker, matching
     each session's run_live_dryrun invocation.

Replay fidelity gate per log: replayed (status, client_action) must equal
the logged session byte-for-byte; mismatching logs are flagged and their
frames are quarantined out of headline counts.

Usage (light, nice'd — RT-1 protocol):
    nice -n 19 taskset -c 10-11 .venv/bin/python \
        evals/rt1_enumeration_20260612/extract_frames.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "evals" / "rt1_enumeration_20260612"
STRUCTURE_YAML = "configs/ignition_double_up_6max_turbo.yaml"
DEFAULT_CHECKPOINT = str(REPO_ROOT / "runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt")
DEFAULT_ABSTRACTION = str(REPO_ROOT / "runs/abstraction_20260521_223018_retrofit/abstraction.pkl")

# The live session is ACTIVE on this log — audit the snapshot instead.
LIVE_LOG_NAME = "live_dryrun_20260612_220101.jsonl"
LIVE_SNAPSHOT = OUT_DIR / "snapshot_live_dryrun_20260612_220101.jsonl"

EFF_FILTER_BB = 2.5
STREETS = {0: "preflop", 1: "flop", 2: "turn", 3: "river", None: "?"}


class Capture:
    def __init__(self):
        self.reset()

    def reset(self):
        self.policy_pre = None
        self.policy_post = None
        self.legal_mask = None
        self.parsed_snapshot = None
        self.discrete_to_chip = None
        self.floors_fired = []
        self.action_seqs = []
        self.recon_views = []
        self.packs = []


CAP = Capture()


def install_hooks():
    """decision_audit's three pure capture hooks, with the accepts_d2c fix."""
    import numpy as np
    from src.nlhe.integration import live_loop
    from src.nlhe.integration import scraper_schema
    from src.nlhe.integration import invariant as invariant_mod
    from src.nlhe.actions import discretize_legal_actions
    from src.nlhe.cfr6 import _build_view_6max
    from src.nlhe.integration.live_loop import (
        apply_aa_kk_preflop_floor, apply_check_when_free_floor,
        apply_short_stack_floor, apply_commitment_tail_floor,
        _hero_eff_bb_from_parsed,
    )

    orig_make_filter = live_loop.make_live_policy_filter

    def capturing_make_filter(*args, **kwargs):
        composed = orig_make_filter(*args, **kwargs)
        threshold = kwargs.get("short_stack_threshold_bb",
                               args[0] if args else 6.0)
        tau = kwargs.get("tail_floor_tau")

        def capturing_filter(policy, legal_mask, parsed, state,
                             discrete_to_chip=None):
            out = composed(policy, legal_mask, parsed, state,
                           discrete_to_chip=discrete_to_chip)
            # Recompute floor firing flags exactly as composed does
            # (pure functions, reference-equality short-circuit).
            p1 = apply_aa_kk_preflop_floor(policy, legal_mask, parsed, state)
            p2 = apply_check_when_free_floor(p1, legal_mask, parsed, state)
            p3 = apply_short_stack_floor(p2, legal_mask, parsed, state,
                                         threshold_bb=threshold)
            if tau is not None:
                p4 = apply_commitment_tail_floor(
                    p3, legal_mask, parsed, state, tau_max=tau,
                    discrete_to_chip=discrete_to_chip)
            else:
                p4 = p3
            fires = []
            if p1 is not policy: fires.append("AA/KK")
            if p2 is not p1: fires.append("check-free")
            if p3 is not p2: fires.append("short-stack")
            if p4 is not p3: fires.append("tail")
            eff_bb, bb = _hero_eff_bb_from_parsed(parsed)
            cp = parsed["current_player"]
            contribs = parsed.get("contribution", [])
            to_call = (int(max(contribs)) - int(contribs[cp])
                       if contribs else None)
            if discrete_to_chip is not None:
                d2c = discrete_to_chip
            else:
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
                "contribution": list(int(c) for c in contribs),
                "money": list(int(m) for m in parsed.get("money", [])),
                "big_blind": int(parsed.get("big_blind", 0)),
                "eff_bb_floors": eff_bb,
            }
            return out

        capturing_filter.accepts_d2c = True
        return capturing_filter

    live_loop.make_live_policy_filter = capturing_make_filter

    orig_derive = scraper_schema.derive_action_sequence

    def capturing_derive(frame, pre_hand_override=None):
        seq = orig_derive(frame, pre_hand_override=pre_hand_override)
        CAP.action_seqs.append({
            "override_used": pre_hand_override is not None,
            "seq": [(int(s), int(a)) for s, a in seq],
        })
        return seq

    scraper_schema.derive_action_sequence = capturing_derive

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
        })
        return res

    invariant_mod.check_mid_hand_invariant = capturing_inv


def load_records(path: Path):
    """Read jsonl, tolerating a partial trailing line (live snapshot)."""
    records = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"    [warn] {path.name}: dropping unparseable "
                      f"line {i+1} (partial write?)", flush=True)
    return records


def replay_log(path: Path, solver_cache: dict, structure):
    """Replay one log; return (meta, sub25_frames)."""
    import numpy as np
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.integration.live_loop import DecisionCache, make_decision
    from src.nlhe.integration.scraper_schema import SessionTracker, parse_frame
    from src.nlhe.actions import DiscreteAction
    from scripts.eval_6max_self_play import _load_solver
    from scripts.decision_audit import position_label

    records = load_records(path)
    if not records:
        return {"log": path.name, "skipped": "empty"}, []
    header = records[0] if records[0].get("record_type") == "session_header" \
        else {}
    frames = [r for r in records if r.get("raw_record") is not None]
    if not frames:
        return {"log": path.name, "skipped": "no raw_record frames "
                "(summary-tier log)"}, []

    # Exact decision_audit seed convention.
    seed = int(header.get("mode") == "sample" and header.get("seed", 2026))
    mode = header.get("mode", "sample")
    floors = header.get("floors", {}) or {}
    ss_bb = float(floors.get("short_stack_bb", 6.0))
    tail_tau = floors.get("tail_floor_tau")
    tail_tau = float(tail_tau) if tail_tau is not None else None
    anchor_sum_floor = bool(header.get("anchor_sum_floor", False))
    extended_click_plans = bool(header.get("extended_click_plans", False))
    bet_closure = bool(header.get("bet_closure_recovery", False))

    ckpt = header.get("ckpt_path", DEFAULT_CHECKPOINT)
    ckpt_key = str(Path(ckpt).resolve())
    if ckpt_key not in solver_cache:
        abstr = Abstraction.load(DEFAULT_ABSTRACTION)
        solver_cache.clear()  # keep at most one solver in RAM
        solver_cache[ckpt_key] = _load_solver(ckpt, abstr, structure)
    solver = solver_cache[ckpt_key]

    tracker = SessionTracker(anchor_sum_floor=anchor_sum_floor,
                             bet_closure_recovery=bet_closure)
    cache = DecisionCache()
    rng = random.Random(seed)

    fidelity_fails = []
    out_frames = []
    n_decisions = 0
    cached_repeats = {}      # identity -> n cached frames
    identity_of_row = {}

    for rec in frames:
        CAP.reset()
        try:
            d = make_decision(rec["raw_record"], structure, solver, tracker,
                              rng, mode=mode, seq=rec.get("seq"),
                              decision_cache=cache,
                              short_stack_floor_bb=ss_bb,
                              extended_click_plans=extended_click_plans,
                              tail_floor_tau=tail_tau,
                              bet_closure_recovery=bet_closure)
        except Exception as e:
            fidelity_fails.append((rec.get("seq"), "replay_exception",
                                   f"{type(e).__name__}: {e}", None, None))
            continue
        logged_ca = rec.get("client_action")
        if (d.status != rec.get("status")
                or json.dumps(d.client_action, sort_keys=True)
                != json.dumps(logged_ca, sort_keys=True)):
            fidelity_fails.append((rec.get("seq"), rec.get("status"),
                                   d.status, logged_ca, d.client_action))

        if not d.status.startswith("decision"):
            continue
        if d.status.endswith("cached"):
            key = repr(d.decision_identity)
            cached_repeats[key] = cached_repeats.get(key, 0) + 1
            continue
        # fresh decision
        n_decisions += 1
        if CAP.policy_post is None:
            continue
        bb = d.blinds[1] if d.blinds else 1
        parsed = CAP.parsed_snapshot or {}
        # decision_audit card-style eff / to_call from the RAW frame
        try:
            frame = parse_frame(rec["raw_record"])
            alive, bets, stacks = frame.alive, frame.bet, frame.stack
        except Exception:
            alive, bets, stacks = [True] * 6, None, None
        hero_bet = bets[d.hero_seat] if bets else 0
        max_opp = max((b for i, b in enumerate(bets)
                       if i != d.hero_seat), default=0) if bets else 0
        to_call_card = max(0, max_opp - hero_bet)
        hero_total = (d.hero_stack or 0) + hero_bet
        opp_totals = [(i, stacks[i] + bets[i]) for i in range(6)
                      if stacks and i != d.hero_seat and alive[i]]
        eff_card = min(hero_total,
                       max((t for _, t in opp_totals), default=0))
        eff_card_bb = eff_card / max(1, bb)
        eff_floor_bb = float(parsed.get("eff_bb_floors") or 0.0)
        to_call_parsed = parsed.get("to_call")
        facing = (to_call_card > 0) or (to_call_parsed or 0) > 0
        if not facing:
            continue
        if min(eff_card_bb, eff_floor_bb) > EFF_FILTER_BB:
            continue

        pack = CAP.packs[-1] if CAP.packs else None
        pos = (position_label(d.hero_seat, d.dealer_seat, pack["sb_seat"],
                              pack["bb_seat"], alive) if pack else "?")
        # OOD-WARN condition, exactly live_loop's
        ood_warn = bool(d.level is not None and (
            d.level >= 8 or
            (bb > 0 and d.hero_stack and d.hero_stack / bb < 2.0)))

        dist = {}
        for da in DiscreteAction:
            idx = int(da)
            if CAP.legal_mask[idx] == 0:
                continue
            ci = CAP.discrete_to_chip.get(idx)
            dist[da.name] = {
                "pre": round(float(CAP.policy_pre[idx]), 6),
                "post": round(float(CAP.policy_post[idx]), 6),
                "chip": ci,
            }
        post_argmax = max(dist, key=lambda k: dist[k]["post"])
        chosen_ci = d.resolver_raw_openspiel_chip_int
        ca = d.client_action or {}

        key = repr(d.decision_identity)
        row = {
            "session": path.name,
            "seq": d.seq,
            "captured_at": d.captured_at,
            "level": d.level,
            "blinds": d.blinds,
            "street": STREETS.get(d.street_idx),
            "cards": list(d.hero_cards),
            "board": list(d.board),
            "position": pos,
            "n_alive": d.n_alive,
            "pot": d.pot_total,
            "hero_stack": d.hero_stack,
            "hero_stack_bb": round((d.hero_stack or 0) / max(1, bb), 3),
            "eff_card_bb": round(eff_card_bb, 3),
            "eff_floor_bb": round(eff_floor_bb, 3),
            "to_call": int(to_call_card),
            "to_call_bb": round(to_call_card / max(1, bb), 3),
            "pot_odds": round(d.pot_total / to_call_card, 2)
                if to_call_card else None,
            "ood_warn": ood_warn,
            "floors_fired": list(CAP.floors_fired),
            "dist": dist,
            "post_argmax": post_argmax,
            "chosen_chip_int": int(chosen_ci)
                if chosen_ci is not None else None,
            "chosen_kind": ca.get("kind"),
            "chosen_amount": ca.get("chip_amount"),
            "decision_identity": key,
            "cached_repeats": 0,  # filled below
        }
        identity_of_row[key] = row
        out_frames.append(row)

    for key, n in cached_repeats.items():
        if key in identity_of_row:
            identity_of_row[key]["cached_repeats"] = n

    meta = {
        "log": path.name,
        "header_present": bool(header),
        "mode": mode, "seed": seed,
        "flags": {"short_stack_bb": ss_bb, "tail_floor_tau": tail_tau,
                  "anchor_sum_floor": anchor_sum_floor,
                  "extended_click_plans": extended_click_plans,
                  "bet_closure_recovery": bet_closure},
        "ckpt": ckpt,
        "frames_replayed": len(frames),
        "fresh_decisions": n_decisions,
        "fidelity_fails": len(fidelity_fails),
        "fidelity_fail_detail": [
            {"seq": s, "logged_status": ls, "replay_status": rs,
             "logged_ca": lca, "replay_ca": rca}
            for s, ls, rs, lca, rca in fidelity_fails[:25]],
        "sub25_facing_frames": len(out_frames),
    }
    return meta, out_frames


def main():
    from src.nlhe.game_strings import TournamentStructure

    install_hooks()
    structure = TournamentStructure.from_yaml(str(REPO_ROOT / STRUCTURE_YAML))

    logs = sorted((REPO_ROOT / "logs").glob("live_dryrun_*.jsonl"))
    paths = []
    for p in logs:
        if p.name == LIVE_LOG_NAME:
            paths.append(LIVE_SNAPSHOT)   # live session active: use snapshot
        else:
            paths.append(p)

    solver_cache = {}
    metas, all_frames = [], []
    for p in paths:
        print(f"[rt1] replaying {p.name} ...", flush=True)
        meta, fr = replay_log(p, solver_cache, structure)
        print(f"[rt1]   -> {json.dumps({k: meta.get(k) for k in ('skipped', 'frames_replayed', 'fresh_decisions', 'fidelity_fails', 'sub25_facing_frames')})}",
              flush=True)
        metas.append(meta)
        all_frames.extend(fr)

    out = {"generated_utc": __import__("datetime").datetime.utcnow()
           .isoformat() + "Z",
           "filter": {"eff_bb_max": EFF_FILTER_BB, "facing": "to_call > 0"},
           "logs": metas,
           "frames": all_frames}
    raw_path = OUT_DIR / "frames_raw.json"
    raw_path.write_text(json.dumps(out, indent=1))
    print(f"[rt1] wrote {raw_path} ({len(all_frames)} sub-2.5BB facing-bet "
          f"decisions)", flush=True)


if __name__ == "__main__":
    main()
