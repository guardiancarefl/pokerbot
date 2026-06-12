"""RESEARCH_MAP a1 — bucket decision-entropy "blur map" for a deployed champion.

Self-play SNGs with the champion checkpoint in ALL SIX seats (reuses
scripts/sng_baseline.play_sng_game via its seat_to_policy seam, same blind
escalation / bust-carry / 3-alive-terminal rules as the official yardstick).
Every decision taken by any seat is a champion decision; at each one we
record:

  street_idx          0-3 (preflop/flop/turn/river)
  bucket              abstraction bucket id of (actor hole, board) — the
                      identical bucket the encoder one-hots for the net
  eff-depth band      actor effective stack in raw BB, banded
                      [<6, 6-10, 10-20, 20-40, 40+]
  pot_bb              pot in raw BB. NOTE: universal_poker's observation
                      "Pot:" field is the POTENTIAL pot (current max
                      commitment x num seats — verified 150 vs 70 actually
                      committed at a fresh L1 state), so we compute the real
                      pot as sum(PlayerContribution): chips actually
                      committed by all seats this hand, antes included.
  policy distribution over LEGAL discrete actions (deployment path:
                      networks6.inference_policy, masked)

Per decision we compute:
  Hn      = normalized entropy H / ln(n_legal) over the legal-action
            distribution (decisions with n_legal < 2 are counted but
            excluded from entropy stats — there is nothing to decide)
  stake   = pot_bb + eff_bb  (EV-at-stake proxy)

STAKE PROXY CHOICE (documented per task spec): stake = pot_bb + eff_bb where
eff_bb = min(actor stack, max alive opponent stack) / raw BB — i.e. the pot
already out there plus the actor's maximum *matchable* remaining commitment.
This is the ceiling of the chip swing reachable from the node. pot_bb alone
is also recorded per cell so the report can show both.

facing_allin flag: to_call >= actor stack (calling commits everything) —
recorded so the blur map can be cross-referenced against the known
5-15bb facing-shove over-calling hole.

Sharding: --shard i --num-shards k plays game indices g with g % k == i,
seed = master_seed + 7919*g (sng_baseline's CRN schedule). --aggregate then
merges shard jsons into blur_map.json + the stats section of
BLUR_MAP_REPORT.txt.

New-file-only harness; touches no library code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

DEPTH_BAND_LABELS = ["<6", "6-10", "10-20", "20-40", "40+"]
STREET_LABELS = ["preflop", "flop", "turn", "river"]
N_GAME_SEED_STRIDE = 7919  # sng_baseline CRN stride


def depth_band(eff_bb: float) -> int:
    if eff_bb < 6.0:
        return 0
    if eff_bb < 10.0:
        return 1
    if eff_bb < 20.0:
        return 2
    if eff_bb < 40.0:
        return 3
    return 4


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Recorder ────────────────────────────────────────────────────────────

class BlurRecorder:
    """Accumulates per-decision blur statistics.

    sb_entropy keeps the raw normalized-entropy list per (street, band) so
    the aggregate step can compute exact p90 (no streaming approximation).
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.sb_entropy: dict[tuple[int, int], list[float]] = {}
        self.sb_acc: dict[tuple[int, int], list[float]] = {}   # n, sum_stake, sum_pot
        self.bucket_acc: dict[tuple[int, int], dict] = {}      # (street, bucket)
        self.allin_acc: dict[tuple[int, int, int], list[float]] = {}  # (street, band, facing_allin)
        self.n_decisions = 0
        self.n_single_action = 0
        self.n_no_discrete = 0
        self.n_no_bb = 0
        self.n_no_bucket = 0

    def record(self, *, street: int, bucket, eff_bb: float, pot_bb: float,
               to_call_bb: float, facing_allin: bool,
               policy: np.ndarray, legal_idx: list[int]) -> None:
        self.n_decisions += 1
        if bucket is None:
            self.n_no_bucket += 1
            bucket = -1
        n_legal = len(legal_idx)
        if n_legal < 2:
            self.n_single_action += 1
            return
        p = np.asarray([policy[i] for i in legal_idx], dtype=np.float64)
        s = float(p.sum())
        if s <= 0.0:
            p = np.full(n_legal, 1.0 / n_legal)
        else:
            p = p / s
        h = float(-np.sum(p[p > 0.0] * np.log(p[p > 0.0])))
        hn = h / math.log(n_legal)
        stake = pot_bb + eff_bb
        band = depth_band(eff_bb)

        key = (street, band)
        self.sb_entropy.setdefault(key, []).append(round(hn, 5))
        acc = self.sb_acc.setdefault(key, [0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += stake
        acc[2] += pot_bb

        bkey = (street, int(bucket))
        b = self.bucket_acc.get(bkey)
        if b is None:
            b = {"n": 0, "sum_h": 0.0, "sum_h2": 0.0, "sum_stake": 0.0,
                 "sum_pot": 0.0, "sum_eff_bb": 0.0, "n_facing_allin": 0,
                 "policy_sum": [0.0] * self.n_actions}
            self.bucket_acc[bkey] = b
        b["n"] += 1
        b["sum_h"] += hn
        b["sum_h2"] += hn * hn
        b["sum_stake"] += stake
        b["sum_pot"] += pot_bb
        b["sum_eff_bb"] += eff_bb
        if facing_allin:
            b["n_facing_allin"] += 1
        for i in legal_idx:
            b["policy_sum"][i] += float(policy[i])

        akey = (street, band, 1 if facing_allin else 0)
        a = self.allin_acc.setdefault(akey, [0, 0.0, 0.0])
        a[0] += 1
        a[1] += hn
        a[2] += stake

    def to_jsonable(self) -> dict:
        return {
            "n_actions": self.n_actions,
            "counters": {
                "n_decisions": self.n_decisions,
                "n_single_action": self.n_single_action,
                "n_no_discrete": self.n_no_discrete,
                "n_no_bb": self.n_no_bb,
                "n_no_bucket": self.n_no_bucket,
            },
            "sb_entropy": {f"{k[0]}|{k[1]}": v for k, v in self.sb_entropy.items()},
            "sb_acc": {f"{k[0]}|{k[1]}": v for k, v in self.sb_acc.items()},
            "bucket_acc": {f"{k[0]}|{k[1]}": v for k, v in self.bucket_acc.items()},
            "allin_acc": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in self.allin_acc.items()},
        }


# ── Probe policy ────────────────────────────────────────────────────────

class BlurProbePolicy:
    """Champion policy with a recording tap.

    Mirrors eval_6max_self_play._sample_action_from_policy exactly
    (discretize -> legal mask -> encode -> inference_policy -> sample),
    inserting one read-only recording call before the sample. The recording
    consumes no RNG draws, so play within this harness is the deployment
    sampling path bit-for-bit.
    """
    name = "blur_probe"

    def __init__(self, solver, recorder: BlurRecorder, n_actions: int):
        self.solver = solver
        self.rec = recorder
        self.n_actions = n_actions

    def select_action(self, parsed, state, rng, mode: str = "sample") -> int:
        from src.nlhe.actions import DiscreteAction, discretize_legal_actions
        from src.nlhe.cfr6 import _build_view_6max

        cp = parsed["current_player"]
        legal_chip = list(state.legal_actions())
        view = _build_view_6max(state, parsed)
        discrete_to_chip = discretize_legal_actions(legal_chip, view)
        if not discrete_to_chip:
            self.rec.n_decisions += 1
            self.rec.n_no_discrete += 1
            return rng.choice(legal_chip)

        legal_mask = np.zeros(self.n_actions, dtype=np.float32)
        for da in discrete_to_chip:
            legal_mask[int(da)] = 1.0

        encoded = self.solver.encoder.encode_from_parsed(parsed, rng=rng)
        features = np.asarray(encoded, dtype=np.float32)
        policy = self.solver.policy_nets.inference_policy(cp, features, legal_mask)

        # ---- recording tap (read-only; bucket comes from the encoder's
        # per-state cache populated by encode_from_parsed, so no extra MC
        # runouts and no RNG consumption) ----
        bb = int(parsed.get("big_blind") or 0)
        if bb <= 0:
            self.rec.n_decisions += 1
            self.rec.n_no_bb += 1
        else:
            bucket = self.solver.encoder._get_bucket(parsed, None)  # cache hit
            my_stack = parsed["money"][cp]
            opp_stacks = [parsed["money"][i] for i in range(6)
                          if i != cp and parsed["money"][i] > 0]
            eff = min(my_stack, max(opp_stacks)) if opp_stacks else my_stack
            max_contrib = max(parsed["contribution"])
            to_call = max(0, max_contrib - parsed["contribution"][cp])
            # Real pot = chips actually committed (universal_poker's "Pot:"
            # observation field is the potential pot — see module docstring).
            real_pot = sum(parsed["contribution"])
            self.rec.record(
                street=int(parsed["street_idx"]),
                bucket=bucket,
                eff_bb=eff / bb,
                pot_bb=real_pot / bb,
                to_call_bb=to_call / bb,
                facing_allin=(to_call >= my_stack and to_call > 0),
                policy=policy,
                legal_idx=sorted(int(da) for da in discrete_to_chip),
            )

        if mode == "argmax":
            chosen_idx = int(np.argmax(policy))
        else:
            chosen_idx = rng.choices(range(self.n_actions),
                                     weights=policy.tolist(), k=1)[0]
        da = DiscreteAction(chosen_idx)
        chip = discrete_to_chip.get(da)
        if chip is None:
            chip = rng.choice(list(discrete_to_chip.values()))
        return int(chip)


# ── Shard runner ────────────────────────────────────────────────────────

def run_shard(args) -> int:
    import torch
    torch.set_num_threads(1)

    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from src.nlhe.networks6 import N_DISCRETE_ACTIONS
    from scripts.eval_6max_self_play import _load_solver
    from scripts.sng_baseline import play_sng_game
    from scripts.throwaway_query_real_ante import _RealAnteStructure

    def log(msg):
        print(f"{datetime.now().strftime('%H:%M:%S')}  [shard {args.shard}] {msg}",
              flush=True)

    structure = _RealAnteStructure(TournamentStructure.from_yaml(args.structure))
    abstr = Abstraction.load(args.abstraction)
    solver = _load_solver(args.ckpt, abstr, structure)
    recorder = BlurRecorder(N_DISCRETE_ACTIONS)
    probe = BlurProbePolicy(solver, recorder, N_DISCRETE_ACTIONS)
    seat_to_policy = [probe] * 6

    game_ids = [g for g in range(args.games) if g % args.num_shards == args.shard]
    log(f"{len(game_ids)} games (of {args.games} global), "
        f"master_seed={args.master_seed}, hpl={args.hands_per_level}")

    header = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "games_global": args.games,
        "games_shard": len(game_ids),
        "master_seed": args.master_seed,
        "hands_per_level": args.hands_per_level,
        "max_hands": args.max_hands,
        "mode": args.mode,
        "ckpt_path": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": _sha256_of_file(args.ckpt),
        "abstraction_sha256": _sha256_of_file(args.abstraction),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tainted = 0
    total_hands = 0
    t0 = time.time()

    def flush(done):
        shard = {"header": header, "games_done": done, "n_tainted": n_tainted,
                 "total_hands": total_hands, "elapsed_s": time.time() - t0,
                 "recorder": recorder.to_jsonable()}
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(shard))
        tmp.rename(out_path)

    for j, g in enumerate(game_ids):
        solver.encoder.reset_cache()    # bound memory; boards are per-game anyway
        seed = args.master_seed + N_GAME_SEED_STRIDE * g
        rec = play_sng_game(probe, probe, structure, seed=seed,
                            hands_per_level=args.hands_per_level,
                            max_hands=args.max_hands, mode=args.mode,
                            seat_to_policy=seat_to_policy)
        if rec["tainted"]:
            n_tainted += 1
            log(f"TAINTED game {g}: {rec['exception']}")
        total_hands += rec.get("hands", 0)
        if (j + 1) % args.flush_every == 0:
            flush(j + 1)
            rate = (j + 1) / (time.time() - t0)
            log(f"{j + 1}/{len(game_ids)} games  "
                f"decisions={recorder.n_decisions}  "
                f"{rate:.2f} games/s  eta={((len(game_ids) - j - 1) / rate):.0f}s")

    flush(len(game_ids))
    log(f"DONE  games={len(game_ids)} hands={total_hands} "
        f"decisions={recorder.n_decisions} tainted={n_tainted} "
        f"[{time.time() - t0:.0f}s]")
    return 0


# ── Aggregation / report ───────────────────────────────────────────────

def _merge_shards(paths):
    merged = None
    headers = []
    meta = {"games_done": 0, "n_tainted": 0, "total_hands": 0, "elapsed_s": 0.0}
    for p in paths:
        d = json.loads(Path(p).read_text())
        headers.append(d["header"])
        for k in meta:
            meta[k] += d[k]
        r = d["recorder"]
        if merged is None:
            merged = {"n_actions": r["n_actions"],
                      "counters": dict(r["counters"]),
                      "sb_entropy": {k: list(v) for k, v in r["sb_entropy"].items()},
                      "sb_acc": {k: list(v) for k, v in r["sb_acc"].items()},
                      "bucket_acc": {k: dict(v) for k, v in r["bucket_acc"].items()},
                      "allin_acc": {k: list(v) for k, v in r["allin_acc"].items()}}
            continue
        for k, v in r["counters"].items():
            merged["counters"][k] += v
        for k, v in r["sb_entropy"].items():
            merged["sb_entropy"].setdefault(k, []).extend(v)
        for k, v in r["sb_acc"].items():
            a = merged["sb_acc"].setdefault(k, [0, 0.0, 0.0])
            for i in range(3):
                a[i] += v[i]
        for k, v in r["bucket_acc"].items():
            b = merged["bucket_acc"].get(k)
            if b is None:
                merged["bucket_acc"][k] = dict(v)
            else:
                for f in ("n", "sum_h", "sum_h2", "sum_stake", "sum_pot",
                          "sum_eff_bb", "n_facing_allin"):
                    b[f] += v[f]
                b["policy_sum"] = [x + y for x, y in
                                   zip(b["policy_sum"], v["policy_sum"])]
        for k, v in r["allin_acc"].items():
            a = merged["allin_acc"].setdefault(k, [0, 0.0, 0.0])
            for i in range(3):
                a[i] += v[i]
    return merged, headers, meta


def aggregate(args) -> int:
    from src.nlhe.actions import DiscreteAction

    merged, headers, meta = _merge_shards(args.aggregate)
    na = merged["n_actions"]
    act_labels = [DiscreteAction(i).name for i in range(na)]

    # per (street, band) stats
    sb_rows = []
    for k, ent in sorted(merged["sb_entropy"].items(),
                         key=lambda kv: (int(kv[0].split("|")[0]),
                                          int(kv[0].split("|")[1]))):
        st, band = (int(x) for x in k.split("|"))
        acc = merged["sb_acc"][k]
        e = np.asarray(ent, dtype=np.float64)
        sb_rows.append({
            "street": STREET_LABELS[st], "depth_band": DEPTH_BAND_LABELS[band],
            "n": int(acc[0]),
            "mean_entropy": float(e.mean()),
            "p90_entropy": float(np.percentile(e, 90)),
            "frac_entropy_gt_0.5": float((e > 0.5).mean()),
            "mean_stake_bb": acc[1] / acc[0],
            "mean_pot_bb": acc[2] / acc[0],
        })

    # per-bucket stats within street
    bucket_rows = []
    for k, b in merged["bucket_acc"].items():
        st, bucket = (int(x) for x in k.split("|"))
        n = b["n"]
        mean_h = b["sum_h"] / n
        var_h = max(0.0, b["sum_h2"] / n - mean_h * mean_h)
        mean_stake = b["sum_stake"] / n
        avg_pol = [x / n for x in b["policy_sum"]]
        top = sorted(range(na), key=lambda i: -avg_pol[i])[:3]
        bucket_rows.append({
            "street": STREET_LABELS[st], "bucket": bucket, "n": n,
            "mean_entropy": mean_h, "sd_entropy": math.sqrt(var_h),
            "mean_stake_bb": mean_stake,
            "mean_pot_bb": b["sum_pot"] / n,
            "mean_eff_bb": b["sum_eff_bb"] / n,
            "frac_facing_allin": b["n_facing_allin"] / n,
            "blur_score": mean_h * mean_stake,
            "avg_policy_top3": [f"{act_labels[i]}:{avg_pol[i]:.3f}" for i in top],
        })

    top20 = {}
    for st in STREET_LABELS:
        rows = [r for r in bucket_rows if r["street"] == st and r["n"] >= args.min_n]
        rows.sort(key=lambda r: -r["blur_score"])
        top20[st] = rows[:20]

    # facing-allin table
    allin_rows = []
    for k, a in sorted(merged["allin_acc"].items(),
                       key=lambda kv: tuple(int(x) for x in kv[0].split("|"))):
        st, band, fa = (int(x) for x in k.split("|"))
        allin_rows.append({
            "street": STREET_LABELS[st], "depth_band": DEPTH_BAND_LABELS[band],
            "facing_allin": bool(fa), "n": int(a[0]),
            "mean_entropy": a[1] / a[0], "mean_stake_bb": a[2] / a[0],
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blur = {
        "record_type": "a1_blur_map",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "shard_headers": headers,
        "meta": meta,
        "counters": merged["counters"],
        "action_labels": act_labels,
        "stake_proxy": "pot_bb + eff_bb (eff = min(actor stack, max alive opp "
                       "stack) / raw BB): pot plus actor's matchable remaining "
                       "commitment — ceiling of the chip swing from the node. "
                       "pot = sum(PlayerContribution) / raw BB, the chips "
                       "actually committed (NOT universal_poker's potential-"
                       "pot 'Pot:' field)",
        "entropy_def": "H/ln(n_legal) over the masked deployment policy on "
                       "legal discrete actions; n_legal<2 decisions excluded",
        "street_band": sb_rows,
        "street_band_facing_allin": allin_rows,
        "top20_buckets_by_street": top20,
        "all_buckets": sorted(bucket_rows,
                              key=lambda r: (r["street"], -r["blur_score"])),
    }
    (out_dir / "blur_map.json").write_text(json.dumps(blur, indent=1))

    # ---- stats section of the report ----
    L = []
    L.append("A1 BLUR MAP — champion decision-entropy x stake, self-play")
    L.append(f"generated: {blur['generated_utc']}")
    h0 = headers[0]
    L.append(f"ckpt: {h0['ckpt_path']}")
    L.append(f"ckpt sha256: {h0['ckpt_sha256'][:16]}…  "
             f"abstraction sha256: {h0['abstraction_sha256'][:16]}…")
    L.append(f"games={meta['games_done']} (master_seed={h0['master_seed']}, "
             f"hpl={h0['hands_per_level']}, mode={h0['mode']}, all-champion 6 seats)  "
             f"hands={meta['total_hands']}  tainted={meta['n_tainted']}")
    c = merged["counters"]
    L.append(f"decisions={c['n_decisions']}  scored={c['n_decisions'] - c['n_single_action'] - c['n_no_discrete'] - c['n_no_bb']}  "
             f"single-action(skipped)={c['n_single_action']}  "
             f"no-discrete={c['n_no_discrete']}  no-bb={c['n_no_bb']}  "
             f"no-bucket={c['n_no_bucket']}")
    L.append("")
    L.append(f"stake proxy: {blur['stake_proxy']}")
    L.append(f"entropy: {blur['entropy_def']}")
    L.append("")
    L.append("== street x depth-band (normalized entropy) ==")
    L.append(f"{'street':<8} {'depth':<6} {'n':>8} {'meanH':>7} {'p90H':>7} "
             f"{'fr>0.5':>7} {'stake':>8} {'pot':>7}")
    for r in sb_rows:
        L.append(f"{r['street']:<8} {r['depth_band']:<6} {r['n']:>8d} "
                 f"{r['mean_entropy']:>7.3f} {r['p90_entropy']:>7.3f} "
                 f"{r['frac_entropy_gt_0.5']:>7.3f} {r['mean_stake_bb']:>8.1f} "
                 f"{r['mean_pot_bb']:>7.1f}")
    L.append("")
    L.append("== street x depth-band, split by facing-allin "
             "(to_call >= actor stack) ==")
    L.append(f"{'street':<8} {'depth':<6} {'allin':<6} {'n':>8} {'meanH':>7} "
             f"{'stake':>8}")
    for r in allin_rows:
        L.append(f"{r['street']:<8} {r['depth_band']:<6} "
                 f"{('YES' if r['facing_allin'] else 'no'):<6} {r['n']:>8d} "
                 f"{r['mean_entropy']:>7.3f} {r['mean_stake_bb']:>8.1f}")
    L.append("")
    for st in STREET_LABELS:
        rows = top20[st]
        if not rows:
            continue
        L.append(f"== top-20 {st} buckets by blur_score = meanH x mean_stake "
                 f"(n >= {args.min_n}) ==")
        L.append(f"{'bucket':>6} {'n':>7} {'meanH':>7} {'sdH':>6} {'stake':>8} "
                 f"{'pot':>7} {'effbb':>7} {'frAI':>6} {'score':>8}  avg-policy top3")
        for r in rows:
            L.append(f"{r['bucket']:>6d} {r['n']:>7d} {r['mean_entropy']:>7.3f} "
                     f"{r['sd_entropy']:>6.3f} {r['mean_stake_bb']:>8.1f} "
                     f"{r['mean_pot_bb']:>7.1f} {r['mean_eff_bb']:>7.1f} "
                     f"{r['frac_facing_allin']:>6.3f} {r['blur_score']:>8.2f}  "
                     f"{', '.join(r['avg_policy_top3'])}")
        L.append("")
    report = "\n".join(L) + "\n"
    (out_dir / "BLUR_MAP_REPORT.txt").write_text(report)
    print(report)
    print(f"wrote {out_dir / 'blur_map.json'} and {out_dir / 'BLUR_MAP_REPORT.txt'}")
    return 0


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="A1 blur map harness")
    ap.add_argument("--ckpt",
                    default="runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt")
    ap.add_argument("--abstraction",
                    default="runs/abstraction_20260521_223018_retrofit/abstraction.pkl")
    ap.add_argument("--structure",
                    default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--games", type=int, default=100,
                    help="GLOBAL game count across all shards")
    ap.add_argument("--master-seed", type=int, default=4242)
    ap.add_argument("--hands-per-level", type=int, default=5)
    ap.add_argument("--max-hands", type=int, default=200)
    ap.add_argument("--mode", default="sample", choices=["sample", "argmax"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default=None, help="shard json output path")
    ap.add_argument("--flush-every", type=int, default=25)
    ap.add_argument("--aggregate", nargs="+", default=None,
                    help="shard json paths; merges and writes the report")
    ap.add_argument("--out-dir", default="evals/a1_blur_map_20260612")
    ap.add_argument("--min-n", type=int, default=30)
    args = ap.parse_args()

    if args.aggregate:
        return aggregate(args)
    if args.out is None:
        args.out = f"{args.out_dir}/shard_{args.shard}.json"
    return run_shard(args)


if __name__ == "__main__":
    sys.exit(main())
