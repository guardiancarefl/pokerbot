"""Shoviest-rows panel for the slate-2a shove-defense floor — REGISTERED.

Registration (BINDING, read first): docs/research_program/EXPERIMENT_LOG.md
entries 2026-06-12 ~23:05 (supplement §2 + §3), ~23:15 (selection), ~23:40
(launch condition met).

PANEL (pre-logged): lionmttv.10, modernmikemtt, itmstrikea,
minestackermttv.7.3. Per row: 2000 CRN-paired games on the sng_baseline
instrument (scripts/sng_baseline.play_sng_game), master seed 2026, per-game
seed schedule seed = 2026 + 7919*g — identical both arms.

  arm T  (tail-only):  hero = champion ckpt + deployed floor chain
                       make_live_policy_filter(short_stack_threshold_bb=6.0,
                       tail_floor_tau=0.10)
  arm TS (tail+shove): same chain + slate-2a shove-defense floor (tau=0)
                       run AFTER the chain (V1-after-chain composition,
                       verbatim scripts/shove_defense_probe._make_arm_filter)

Pairing: per-game hero-net delta TS − T on identical seeds. Pairs where
either arm taints are recorded but EXCLUDED from deltas.

BARS (registered): no row z <= -2 on the paired per-row delta; pooled
delta >= 0 within noise; gray zone z in (-2, 0) = registered PASS.

Usage (repo root, venv active; 2 shards per row, one core each):
  python -m scripts.a2a_shoviest_panel row --profile lionmttv.10 \
      --games 2000 --shards 2 --shard 0 \
      --out evals/a2a_shoviest_panel_20260612/row_lionmttv.10_s0.json
  python -m scripts.a2a_shoviest_panel combine \
      --out-dir evals/a2a_shoviest_panel_20260612
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.nlhe.game_strings import TournamentStructure
from scripts.shove_defense_probe import (
    BATTERY, CHECKPOINT, ABSTRACTION, STRUCTURE_YAML,
    make_shove_defense_floor, _make_arm_filter, _mean_se_z)

PANEL = ["lionmttv.10", "modernmikemtt", "itmstrikea", "minestackermttv.7.3"]
MASTER_SEED = 2026
TAIL_TAU = 0.10          # deployed floor chain config (OQ-2, armed standard)
SHOVE_TAU = 0.0          # slate-2a pre-registered tau
THRESHOLD_BB = 6.0       # deployed short-stack floor default


def run_row(profile_name: str, out_path: str, games: int, shards: int,
            shard: int, master_seed: int = MASTER_SEED):
    from src.nlhe.abstraction import Abstraction
    from scripts.eval_pool import CheckpointPolicy
    from scripts.bake_off_real_ante import build_shanky_pool
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    from scripts.eval_6max_self_play import _sample_action_from_policy
    from src.nlhe.integration.live_loop import make_live_policy_filter
    import scripts.sng_baseline as sb

    battery = json.loads((REPO_ROOT / BATTERY).read_text())
    structure = _RealAnteStructure(
        TournamentStructure.from_yaml(STRUCTURE_YAML))
    abstr = Abstraction.load(ABSTRACTION)
    base_hero = CheckpointPolicy("k200", ckpt_path=CHECKPOINT,
                                 abstraction=abstr, structure=structure)
    pool = build_shanky_pool("data/shanky_profiles")
    opp = next(p for p in pool if p.name == profile_name)
    eq_cache: dict = {}

    class ArmHero:
        """Champion ckpt sampled through the given composed filter."""
        def __init__(self, filt, name):
            self.filt = filt
            self.name = name

        def select_action(self, parsed, state, rng, mode="sample"):
            return _sample_action_from_policy(
                base_hero.solver, parsed, state, rng, mode=mode,
                policy_filter=self.filt)

    game_idxs = [i for i in range(games) if i % shards == shard]
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    games_path = Path(str(out_p) + ".games.jsonl")
    print(f"[row {profile_name} s{shard}/{shards}] {len(game_idxs)} paired "
          f"games  seeds {master_seed}+7919*g  tail_tau={TAIL_TAU}  "
          f"shove_tau={SHOVE_TAU}", flush=True)

    records = []
    t0 = time.time()
    with open(games_path, "w") as fg:
        for k, g in enumerate(game_idxs):
            seed = master_seed + 7919 * g     # registration seed schedule
            arm_out = {}
            for arm in ("T", "TS"):
                stats = {"n_calls": 0, "chain_fired": 0, "probe_fired": 0}
                chain = make_live_policy_filter(
                    short_stack_threshold_bb=THRESHOLD_BB,
                    tail_floor_tau=TAIL_TAU)
                probe = (make_shove_defense_floor(
                    structure, battery["ranges"], tau=SHOVE_TAU,
                    eq_cache=eq_cache) if arm == "TS" else None)
                filt = _make_arm_filter(chain, probe, stats)
                hero = ArmHero(filt, f"k200+chain{'+shove' if arm == 'TS' else ''}")
                rec = sb.play_sng_game(hero, opp, structure, seed=seed,
                                       hands_per_level=5, max_hands=200,
                                       mode="sample")
                pq = probe.stats["n_qualify"] if probe is not None else 0
                arm_out[arm] = (rec, stats, pq)
            rT, sT, _ = arm_out["T"]
            rTS, sTS, qTS = arm_out["TS"]
            tainted = bool(rT["tainted"] or rTS["tainted"])
            row = {
                "game": int(g), "seed": int(seed),
                "tainted": tainted,
                "t_net": (None if rT["tainted"] else float(rT["hero_net"])),
                "ts_net": (None if rTS["tainted"]
                           else float(rTS["hero_net"])),
                "delta": (None if tainted
                          else float(rTS["hero_net"] - rT["hero_net"])),
                "t_hands": rT.get("hands"), "ts_hands": rTS.get("hands"),
                "t_capped": rT.get("capped"), "ts_capped": rTS.get("capped"),
                "t_chain_fired": sT["chain_fired"],
                "ts_chain_fired": sTS["chain_fired"],
                "ts_shove_qualified": qTS,
                "ts_shove_fired": sTS["probe_fired"],
            }
            if tainted:
                row["t_exception"] = rT.get("exception")
                row["ts_exception"] = rTS.get("exception")
            records.append(row)
            fg.write(json.dumps(row) + "\n")
            if (k + 1) % 25 == 0:
                fg.flush()
                el = time.time() - t0
                rate = (k + 1) / el
                eta = (len(game_idxs) - k - 1) / max(rate, 1e-9)
                done = [r["delta"] for r in records if r["delta"] is not None]
                print(f"  [{profile_name} s{shard}] {k + 1}/{len(game_idxs)}"
                      f"  delta_so_far={np.mean(done):+.4f}"
                      f"  elapsed={el / 60:.1f}m  eta={eta / 60:.1f}m",
                      flush=True)

    elapsed = time.time() - t0
    summary = _row_summary(profile_name, records, games, shards, shard,
                           master_seed, elapsed)
    out_p.write_text(json.dumps(summary, indent=2))
    d = summary["paired_delta"]
    print(f"[row {profile_name} s{shard}] done n={d['n']}  "
          f"delta={d['mean']:+.5f}±{(d['se'] or float('nan')):.5f}  "
          f"z={(d['z'] if d['z'] is not None else float('nan')):+.2f}  "
          f"shove_fired={summary['firing']['ts_shove_fired']}", flush=True)
    print(f"[out] {out_p}\n[out] {games_path}")
    return summary


def _row_summary(profile_name, records, games, shards, shard, master_seed,
                 elapsed):
    deltas = [r["delta"] for r in records if r["delta"] is not None]
    n_taint = sum(1 for r in records if r["tainted"])
    return {
        "panel": "a2a shoviest-rows (registered supplement §2)",
        "profile": profile_name,
        "config": {
            "games_requested": games, "shards": shards, "shard": shard,
            "master_seed": master_seed,
            "seed_schedule": "2026 + 7919*g (identical both arms)",
            "checkpoint": CHECKPOINT, "abstraction": ABSTRACTION,
            "arm_T": f"chain(short_stack_threshold_bb={THRESHOLD_BB}, "
                     f"tail_floor_tau={TAIL_TAU})",
            "arm_TS": "arm_T + shove_defense_floor(tau=0) after chain",
        },
        "games_completed": len(records),
        "n_tainted_pairs_excluded": n_taint,
        "paired_delta": _mean_se_z(deltas),
        "nonzero_delta": _mean_se_z([d for d in deltas if d != 0.0]),
        "n_nonzero": sum(1 for d in deltas if d != 0.0),
        "arm_means": {
            "t_net": float(np.mean([r["t_net"] for r in records
                                    if r["t_net"] is not None])),
            "ts_net": float(np.mean([r["ts_net"] for r in records
                                     if r["ts_net"] is not None])),
        },
        "firing": {
            "t_chain_fired": sum(r["t_chain_fired"] for r in records),
            "ts_chain_fired": sum(r["ts_chain_fired"] for r in records),
            "ts_shove_qualified": sum(r["ts_shove_qualified"]
                                      for r in records),
            "ts_shove_fired": sum(r["ts_shove_fired"] for r in records),
            "games_with_shove_fire": sum(1 for r in records
                                         if r["ts_shove_fired"] > 0),
        },
        "elapsed_s": elapsed,
    }


def run_combine(out_dir: str):
    out_p = Path(out_dir)
    rows = {}
    pooled = []
    for prof in PANEL:
        recs = []
        for jl in sorted(out_p.glob(f"row_{prof}_s*.json.games.jsonl")):
            with open(jl) as f:
                for line in f:
                    recs.append(json.loads(line))
        recs.sort(key=lambda r: r["game"])
        summ = _row_summary(prof, recs, len(recs), 1, 0, MASTER_SEED, 0.0)
        del summ["elapsed_s"]
        rows[prof] = summ
        pooled.extend(r["delta"] for r in recs if r["delta"] is not None)

    pooled_stats = _mean_se_z(pooled)
    row_zs = {p: rows[p]["paired_delta"]["z"] for p in PANEL}
    any_sig_neg = any(z is not None and z <= -2.0 for z in row_zs.values())
    pooled_ok = (pooled_stats["mean"] is not None
                 and (pooled_stats["mean"] >= 0.0
                      or (pooled_stats["z"] is not None
                          and pooled_stats["z"] > -2.0)))
    verdict = "PASS" if (not any_sig_neg and pooled_ok) else "FAIL"
    out = {
        "panel": "a2a shoviest-rows (registered supplement §2 + §3)",
        "registration": "EXPERIMENT_LOG 2026-06-12 ~23:05 / ~23:15 / ~23:40",
        "rows": rows,
        "pooled_delta": pooled_stats,
        "bars": {
            "no_row_z_le_minus2": not any_sig_neg,
            "row_z": row_zs,
            "pooled_ge_0_within_noise": pooled_ok,
            "gray_zone_note": "z in (-2,0) = registered PASS (supplement §3)",
        },
        "verdict": verdict,
    }
    (out_p / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"row_z": row_zs, "pooled": pooled_stats,
                      "verdict": verdict}, indent=2))
    print(f"[combine] wrote {out_p / 'results.json'}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("row")
    r.add_argument("--profile", required=True, choices=PANEL)
    r.add_argument("--out", required=True)
    r.add_argument("--games", type=int, default=2000)
    r.add_argument("--shards", type=int, default=2)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--master-seed", type=int, default=MASTER_SEED)

    c = sub.add_parser("combine")
    c.add_argument("--out-dir", required=True)

    args = ap.parse_args()
    if args.cmd == "row":
        assert 0 <= args.shard < args.shards
        run_row(args.profile, args.out, args.games, args.shards, args.shard,
                master_seed=args.master_seed)
    else:
        run_combine(args.out_dir)


if __name__ == "__main__":
    main()
