"""H1 TG2 gate runner — commitment-scaled tail floor counterfactual.

Binding spec: docs/research_program/H1_TAIL_FLOOR_SPEC.md (TG2).

For every raw-record dry-run log (logs/live_dryrun_*.jsonl), re-replays
the session through make_decision with the exact live config — reusing
scripts/decision_audit.py's capture-hook machinery (install_hooks + CAP),
NOT a reimplementation — to recover the post-existing-floors pre-sample
policy distribution of every fresh decision frame exactly as deployed.
Fidelity gate per log: every replayed frame's (status, client_action)
must equal the logged session; mismatching logs are flagged loudly and
their frames marked non-faithful.

Then applies apply_commitment_tail_floor at tau_max in {0.05, 0.10, 0.15}
to each captured distribution and reports, per threshold:
  - n decision frames, n frames fired (altered-decision rate)
  - pruned-action census
  - historical SAMPLED actions that would have been excluded
    (session, seq, action, mass) — matched by the chosen chip action
  - n argmax changes (TG2 bar: expected 0)

TG2 bars checked in the output:
  (a) seq-315-class draw (session 20260611_163815, 7.5% ALLIN on 2d5c)
      caught at tau_max in {0.10, 0.15}
  (b) zero argmax changes at any threshold
  (c) altered-decision rate at tau_max=0.10 <= 5%

Outputs JSON + human-readable txt to evals/h1_tail_floor_20260612/.

Usage:
    # smoke (session-1 only):
    python -m scripts.tail_floor_counterfactual --smoke
    # full run (all raw-record logs):
    python -m scripts.tail_floor_counterfactual
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TAUS = (0.05, 0.10, 0.15)
OUT_DIR = Path("evals/h1_tail_floor_20260612")
SMOKE_LOG = "logs/live_dryrun_20260611_163815.jsonl"
INCIDENT = {"session": "20260611_163815", "seq": 315,
            "action": "ALLIN", "must_catch_at": (0.10, 0.15)}


def session_name(log_path: Path) -> str:
    import re
    return re.sub(r"^live_dryrun_|\.jsonl$", "", log_path.name)


def replay_log(log_path: Path, solver_cache: dict):
    """Replay one log through make_decision with decision_audit's hooks
    installed; return (frames, fidelity_fails, meta).

    Each frame dict: seq, captured_at, policy_post, legal_mask, parsed,
    d2c, chosen_chip, status.
    """
    from scripts.decision_audit import (
        CAP, STRUCTURE_YAML, DEFAULT_CHECKPOINT, DEFAULT_ABSTRACTION,
    )
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from src.nlhe.integration.live_loop import DecisionCache, make_decision
    from src.nlhe.integration.scraper_schema import SessionTracker
    from scripts.eval_6max_self_play import _load_solver

    records = [json.loads(l) for l in open(log_path) if l.strip()]
    header = records[0] if records and \
        records[0].get("record_type") == "session_header" else {}
    frames_in = [r for r in records if r.get("raw_record") is not None]

    mode = header.get("mode", "sample")
    seed = int(header.get("seed", 2026))
    ckpt = header.get("ckpt_path", DEFAULT_CHECKPOINT)
    if not Path(ckpt).exists():
        print(f"  [warn] header ckpt missing on disk ({ckpt}); "
              f"falling back to {DEFAULT_CHECKPOINT}")
        ckpt = DEFAULT_CHECKPOINT
    key = (str(ckpt), DEFAULT_ABSTRACTION)
    if key not in solver_cache:
        structure = TournamentStructure.from_yaml(STRUCTURE_YAML)
        abstr = Abstraction.load(DEFAULT_ABSTRACTION)
        solver_cache[key] = (_load_solver(str(ckpt), abstr, structure),
                             structure)
    solver, structure = solver_cache[key]

    tracker = SessionTracker(
        anchor_sum_floor=bool(header.get("anchor_sum_floor", False)))
    cache = DecisionCache()
    rng = random.Random(seed)
    ext_clicks = bool(header.get("extended_click_plans", False))

    frames_out, fidelity_fails = [], []
    for rec in frames_in:
        CAP.reset()
        d = make_decision(rec["raw_record"], structure, solver, tracker,
                          rng, mode=mode, seq=rec.get("seq"),
                          decision_cache=cache,
                          extended_click_plans=ext_clicks)
        if (d.status != rec.get("status")
                or json.dumps(d.client_action, sort_keys=True)
                != json.dumps(rec.get("client_action"), sort_keys=True)):
            fidelity_fails.append(
                (rec.get("seq"), rec.get("status"), d.status,
                 rec.get("client_action"), d.client_action))
        # Fresh decision frames only: these are the spots where a policy
        # was actually sampled (cached frames reuse the same draw).
        if d.status in ("decision", "decision_recovered") \
                and CAP.policy_post is not None:
            frames_out.append({
                "seq": d.seq,
                "captured_at": d.captured_at,
                "policy_post": CAP.policy_post.copy(),
                "legal_mask": CAP.legal_mask.copy(),
                "parsed": dict(CAP.parsed_snapshot),
                "d2c": dict(CAP.discrete_to_chip),
                "chosen_chip": int(d.resolver_raw_openspiel_chip_int)
                if d.resolver_raw_openspiel_chip_int is not None else None,
                "hero_cards": list(d.hero_cards),
                "status": d.status,
            })
    meta = {"n_raw_frames": len(frames_in),
            "n_fresh_decisions": len(frames_out),
            "n_fidelity_fails": len(fidelity_fails),
            "mode": mode, "seed": seed, "ckpt": str(ckpt),
            "headerless": not bool(header)}
    return frames_out, fidelity_fails, meta


def counterfactual(all_frames: list[tuple[str, dict]]):
    """Apply the tail floor at each tau to every captured frame."""
    import numpy as np
    from src.nlhe.actions import DiscreteAction
    from src.nlhe.integration.live_loop import apply_commitment_tail_floor

    results = {}
    for tau in TAUS:
        n_fired = 0
        census = Counter()
        excluded = []
        argmax_changes = []
        for sess, fr in all_frames:
            pol = fr["policy_post"]
            mask = fr["legal_mask"]
            out = apply_commitment_tail_floor(pol, mask, fr["parsed"],
                                              None, tau_max=tau,
                                              discrete_to_chip=fr["d2c"])
            if out is pol:
                continue
            n_fired += 1
            pruned = [i for i in range(len(pol))
                      if float(pol[i]) > 0.0 and float(out[i]) == 0.0]
            for i in pruned:
                census[DiscreteAction(i).name] += 1
            # historical sampled action excluded? Match by chip action.
            chosen = fr["chosen_chip"]
            for i in pruned:
                if fr["d2c"].get(i) == chosen and chosen is not None:
                    alias = any(fr["d2c"].get(j) == chosen
                                for j in range(len(pol))
                                if j not in pruned and float(out[j]) > 0.0)
                    excluded.append({
                        "session": sess, "seq": fr["seq"],
                        "action": DiscreteAction(i).name,
                        "mass": round(float(pol[i]), 6),
                        "hero_cards": fr["hero_cards"],
                        "chosen_chip": chosen,
                        "alias_ambiguous": alias,
                    })
            pre_am = int(np.argmax(pol * mask))
            post_am = int(np.argmax(out * mask))
            if pre_am != post_am:
                argmax_changes.append((sess, fr["seq"],
                                       DiscreteAction(pre_am).name,
                                       DiscreteAction(post_am).name))
        n_frames = len(all_frames)
        # altered decision = the HISTORICAL sampled action would have been
        # excluded (distinct from mere firing, which only reshapes tails
        # the historical draw missed anyway).
        altered_frames = {(e["session"], e["seq"]) for e in excluded}
        results[tau] = {
            "n_frames": n_frames,
            "n_fired": n_fired,
            "firing_rate": n_fired / n_frames if n_frames else 0.0,
            "n_altered_decisions": len(altered_frames),
            "altered_decision_rate": (len(altered_frames) / n_frames
                                      if n_frames else 0.0),
            "pruned_census": dict(census),
            "sampled_excluded": excluded,
            "n_argmax_changes": len(argmax_changes),
            "argmax_changes": argmax_changes,
        }
    return results


def render_txt(results, per_log_meta, all_fidelity, label):
    L = []
    W = L.append
    W("=" * 72)
    W(f"H1 TG2 — commitment-scaled tail floor counterfactual ({label})")
    W("=" * 72)
    W("")
    W("[0] REPLAY FIDELITY (per log)")
    for sess, meta in per_log_meta.items():
        tag = "OK" if meta["n_fidelity_fails"] == 0 else \
            f"*** {meta['n_fidelity_fails']} MISMATCHES — distributions " \
            f"for this log are NOT what ran live ***"
        W(f"    {sess:<28s} raw={meta['n_raw_frames']:>5d} "
          f"fresh_decisions={meta['n_fresh_decisions']:>4d} "
          f"{'(headerless: assumed sample/2026)' if meta['headerless'] else ''} "
          f"fidelity={tag}")
    W("")
    n_total = next(iter(results.values()))["n_frames"] if results else 0
    W(f"[1] COUNTERFACTUAL — {n_total} fresh decision frames, "
      f"tau_max in {list(TAUS)}")
    for tau, r in results.items():
        W("")
        W(f"  tau_max = {tau:.2f}")
        W(f"    frames fired (any tail pruned): {r['n_fired']} / "
          f"{r['n_frames']} ({100.0 * r['firing_rate']:.2f}%)")
        W(f"    altered decisions (historical sample excluded): "
          f"{r['n_altered_decisions']} / {r['n_frames']} "
          f"({100.0 * r['altered_decision_rate']:.2f}%)")
        W(f"    pruned-action census:  "
          f"{r['pruned_census'] if r['pruned_census'] else '(none)'}")
        W(f"    argmax changes:        {r['n_argmax_changes']}"
          f"{'  <-- TG2 BAR VIOLATION (expected 0)' if r['n_argmax_changes'] else ''}")
        for s, q, a, b in r["argmax_changes"]:
            W(f"        {s} seq={q}: {a} -> {b}")
        W(f"    historical SAMPLED actions excluded: "
          f"{len(r['sampled_excluded'])}")
        for e in r["sampled_excluded"]:
            W(f"        {e['session']} seq={e['seq']:<5d} "
              f"{e['action']:<8s} mass={e['mass']:.4f} "
              f"cards={''.join(e['hero_cards'])} chip={e['chosen_chip']}"
              f"{' [alias-ambiguous]' if e['alias_ambiguous'] else ''}")
    W("")
    W("[2] TG2 BAR CHECKS")
    # (a) seq-315 incident
    for tau in INCIDENT["must_catch_at"]:
        hit = any(e["session"] == INCIDENT["session"]
                  and e["seq"] == INCIDENT["seq"]
                  and e["action"] == INCIDENT["action"]
                  for e in results.get(tau, {}).get("sampled_excluded", []))
        W(f"    (a) seq-315 2d5c ALLIN excluded at tau_max={tau:.2f}: "
          f"{'YES — caught' if hit else '*** NO — BAR FAILED ***'}")
    # (b) argmax
    n_am = sum(r["n_argmax_changes"] for r in results.values())
    W(f"    (b) argmax changes across all thresholds: {n_am} "
      f"({'PASS (0 expected)' if n_am == 0 else '*** BAR FAILED ***'})")
    # (c) altered rate at 0.10
    rate10 = results.get(0.10, {}).get("altered_decision_rate", 0.0)
    W(f"    (c) altered-decision rate at tau_max=0.10: "
      f"{100.0 * rate10:.2f}% "
      f"({'PASS (<= 5%)' if rate10 <= 0.05 else '*** > 5% — STOP, do not proceed to TG3 ***'})")
    n_ff = sum(m["n_fidelity_fails"] for m in per_log_meta.values())
    if n_ff:
        W("")
        W(f"    NOTE: {n_ff} replay-fidelity mismatches across "
          f"{sum(1 for m in per_log_meta.values() if m['n_fidelity_fails'])} "
          f"log(s) — those logs' distributions are reconstructions under "
          f"CURRENT code, not byte-faithful to what ran live.")
    W("=" * 72)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help=f"single-log smoke on session-1 ({SMOKE_LOG})")
    ap.add_argument("--logs", nargs="*", default=None,
                    help="explicit log list (default: glob "
                         "logs/live_dryrun_*.jsonl)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    if args.smoke:
        logs = [Path(SMOKE_LOG)]
        label = "SMOKE: session-1 only"
        stem = "tail_floor_counterfactual_smoke"
    elif args.logs:
        logs = [Path(p) for p in args.logs]
        label = f"{len(logs)} explicit logs"
        stem = "tail_floor_counterfactual"
    else:
        logs = [Path(p) for p in
                sorted(glob.glob("logs/live_dryrun_*.jsonl"))]
        label = "ALL raw-record logs"
        stem = "tail_floor_counterfactual"

    from scripts.decision_audit import install_hooks
    install_hooks()

    solver_cache = {}
    all_frames = []
    per_log_meta = {}
    all_fidelity = {}
    for lp in logs:
        sess = session_name(lp)
        print(f"[replay] {lp} ...", flush=True)
        frames, fails, meta = replay_log(lp, solver_cache)
        if meta["n_raw_frames"] == 0:
            print(f"  no raw_record frames — skipped (pre-raw-record log)")
            continue
        per_log_meta[sess] = meta
        all_fidelity[sess] = fails
        all_frames.extend((sess, fr) for fr in frames)
        print(f"  raw={meta['n_raw_frames']} fresh={len(frames)} "
              f"fidelity_fails={len(fails)}", flush=True)
        if fails and args.smoke:
            print("*** SMOKE FIDELITY FAILURE — aborting before full run ***")
            for q, ls, rs, lca, rca in fails[:10]:
                print(f"    seq={q} logged={ls}/{lca} replay={rs}/{rca}")
            sys.exit(1)

    results = counterfactual(all_frames)
    txt = render_txt(results, per_log_meta, all_fidelity, label)
    print(txt)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "taus": list(TAUS),
        "per_log": per_log_meta,
        "fidelity_fails": {s: [list(x) for x in f]
                           for s, f in all_fidelity.items()},
        "results": {f"{tau:.2f}": r for tau, r in results.items()},
    }
    (out_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, default=str))
    (out_dir / f"{stem}.txt").write_text(txt + "\n")
    print(f"\nwritten: {out_dir / (stem + '.json')}")
    print(f"written: {out_dir / (stem + '.txt')}")


if __name__ == "__main__":
    main()
