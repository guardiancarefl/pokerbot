"""Cumulative chosen-action-mass accounting across live sessions.

For every FRESH decision (each = exactly one rng.choices draw; cached
frames reuse the draw and are excluded) in the given session logs,
compute the post-floor probability mass of the chosen CLIENT action.
Masses of distinct DiscreteActions that map to the same chip action
(e.g. ALLIN and a clamped BET_200) are summed — the quantity reported
is P(emit this client action), which is what the draw realises.

Outputs:
  - per-decision table (session, seq, cards, street, chosen, mass)
  - histogram of chosen-action masses
  - every decision with chosen mass < 0.10 flagged
  - binomial sanity check: observed sub-10%-mass draws vs expectation
    E = sum_i q_i where q_i = total mass on sub-10% client actions at
    decision i (variance sum q_i(1-q_i)); plus mean chosen mass vs
    its expectation sum_a p_a^2.

Reuses scripts.decision_audit's capture hooks; each session is
replayed with the live config (header seed, fresh tracker/cache) and
is only counted if the replay is byte-identical to the log.

Usage:
    python -m scripts.tail_accounting logs/a.jsonl logs/b.jsonl \
        [--out evals/...txt]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.decision_audit import (  # noqa: E402
    CAP, install_hooks, STRUCTURE_YAML,
    DEFAULT_CHECKPOINT, DEFAULT_ABSTRACTION, STREETS,
)


def collect_session(log_path, solver, structure):
    import random as _random
    from src.nlhe.integration.live_loop import DecisionCache, make_decision
    from src.nlhe.integration.scraper_schema import SessionTracker

    records = [json.loads(l) for l in open(log_path) if l.strip()]
    header = records[0] if records[0].get("record_type") == "session_header" \
        else {}
    frames = [r for r in records if r.get("raw_record") is not None]
    tracker, cache = SessionTracker(), DecisionCache()
    rng = _random.Random(int(header.get("seed", 2026)))

    rows, mismatches = [], 0
    for rec in frames:
        CAP.reset()
        d = make_decision(rec["raw_record"], structure, solver, tracker,
                          rng, mode=header.get("mode", "sample"),
                          seq=rec.get("seq"), decision_cache=cache)
        if (d.status != rec.get("status")
                or json.dumps(d.client_action, sort_keys=True)
                != json.dumps(rec.get("client_action"), sort_keys=True)):
            mismatches += 1
        if d.status not in ("decision", "decision_recovered"):
            continue
        if CAP.policy_post is None:
            continue
        pol = CAP.policy_post
        d2c = CAP.discrete_to_chip or {}
        chosen_ci = int(d.resolver_raw_openspiel_chip_int)
        # client-action mass: sum the masses of all discrete actions
        # mapping to the chosen chip int
        mass = sum(float(pol[idx]) for idx, ci in d2c.items()
                   if int(ci) == chosen_ci)
        # per-client-action aggregated masses for q_i
        agg = {}
        for idx, ci in d2c.items():
            agg[int(ci)] = agg.get(int(ci), 0.0) + float(pol[idx])
        q_low = sum(p for p in agg.values() if p < 0.10)
        exp_mass = sum(p * p for p in agg.values())
        ca = d.client_action or {}
        kind = ca.get("kind", "?")
        chip = ca.get("chip_amount")
        rows.append({
            "seq": d.seq,
            "cards": "".join(d.hero_cards),
            "street": STREETS.get(d.street_idx, "?"),
            "facing": bool(d.facing_bet),
            "chosen": kind.upper() + (f" {chip}" if chip else ""),
            "bb_mult": d.client_action_bb_mult,
            "mass": mass,
            "q_low": q_low,
            "exp_mass": exp_mass,
        })
    return rows, mismatches, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--abstraction", default=DEFAULT_ABSTRACTION)
    args = ap.parse_args()

    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from scripts.eval_6max_self_play import _load_solver

    install_hooks()
    structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
    abstr = Abstraction.load(args.abstraction)
    solver = _load_solver(args.checkpoint, abstr, structure)

    all_rows = []
    L = []
    W = L.append
    W("=" * 72)
    W("CUMULATIVE CHOSEN-ACTION-MASS ACCOUNTING (fresh decisions only)")
    W("=" * 72)
    for lp in args.logs:
        rows, mism, n = collect_session(lp, solver, structure)
        sess = Path(lp).stem.replace("live_dryrun_", "")
        for r in rows:
            r["session"] = sess
        all_rows.extend(rows)
        gate = "GREEN" if mism == 0 else f"*** {mism} MISMATCHES — INVALID"
        W(f"  {sess}: {len(rows)} fresh decisions / {n} frames, "
          f"fidelity {gate}")
    W("")

    W(f"  {'session':<16s} {'seq':>4s} {'cards':<6s} {'street':<8s} "
      f"{'chosen':<16s} {'mass':>6s}")
    for r in all_rows:
        flag = "  <== sub-10% TAIL" if r["mass"] < 0.10 else ""
        W(f"  {r['session']:<16s} {r['seq']:>4d} {r['cards']:<6s} "
          f"{r['street']:<8s} {r['chosen']:<16s} {r['mass']:6.3f}{flag}")
    W("")

    # histogram
    W("  chosen-mass histogram:")
    bins = [(0.0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5),
            (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]
    for lo, hi in bins:
        k = sum(1 for r in all_rows if lo <= r["mass"] < hi)
        W(f"    [{lo:.1f},{hi if hi <= 1 else 1.0:.1f}): "
          f"{'#' * k}{' ' if k else ''}({k})")
    W("")

    n = len(all_rows)
    k_low = sum(1 for r in all_rows if r["mass"] < 0.10)
    e_low = sum(r["q_low"] for r in all_rows)
    var_low = sum(r["q_low"] * (1 - r["q_low"]) for r in all_rows)
    z = (k_low - e_low) / math.sqrt(var_low) if var_low > 0 else float("nan")
    mean_obs = sum(r["mass"] for r in all_rows) / max(1, n)
    mean_exp = sum(r["exp_mass"] for r in all_rows) / max(1, n)
    W("  binomial sanity (event: drawn client action had mass < 0.10):")
    W(f"    draws n={n}   observed sub-10% draws k={k_low}")
    W(f"    expected E=sum q_i = {e_low:.2f}   sd={math.sqrt(var_low):.2f}"
      f"   z=(k-E)/sd = {z:+.2f}")
    W(f"    mean chosen mass observed {mean_obs:.3f} vs expected "
      f"E[sum p^2] {mean_exp:.3f}")
    W("=" * 72)

    report = "\n".join(L)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
        print(f"\nreport written: {args.out}")


if __name__ == "__main__":
    main()
