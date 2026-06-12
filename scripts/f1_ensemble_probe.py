"""f1 ensemble probe harness — ENSEMBLE_PROBE_DESIGN.md (2026-06-12).

The only new code the pre-registration licenses: an `EnsemblePolicy`
Policy-protocol wrapper (selector of design §1.1) and a thin driver that
calls scripts/sng_baseline.evaluate_profile with hero_policy=EnsemblePolicy.
No game-loop, scoring, or seeding code is touched.

Subcommands:
  p0        — selector-mechanics validation: 50 yardstick games vs
              killphilmtt with EnsemblePolicy(champion, champion) must be
              bit-identical to plain CheckpointPolicy champion (bar K1).
  collision — constructed-starting_stacks characterization of the §1.1
              1-chip-live-seat alive-undercount collision (test seam).
  row       — one P2 yardstick row: ensemble hero vs one Shanky profile,
              exact v1 yardstick configuration (2000 games, master_seed
              2026, hpl 5, mode sample), B3 identity stamping with BOTH
              member checkpoint sha256s.
  p1ext     — P1 analysis: convert attacker_extraction_eval bubble-mode
              per-game jsonl(s) to ICM-start-relative extraction/game
              (recomputes each game's harvested start row from the
              seed ^ 0xB0BB1E schedule), optionally CRN-paired against a
              reference jsonl set (B5).
  analyze   — P2 analysis: paired per-game deltas + stage_acc bubble
              pooling vs the champion v1 yardstick baselines; emits the
              pre-registered B1-B4 bar verdicts.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CHAMPION_CKPT = ("runs/k200_real_ante_20260605_225847_PRESERVED/"
                 "ckpt_iter_1500.pt")
ABSTRACTION = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE = "configs/ignition_double_up_6max_turbo.yaml"
BUBBLE_ARTIFACT = "data/training_dist_v1_bubble4.json.gz"


class EnsemblePolicy:
    """Champion everywhere; specialist on hands that start with exactly 4
    players alive (design §1.1). Zero parameters, deterministic, consumes
    no RNG draws itself; switches checkpoints, never representations.

    Hand-start alive count is inferred from the parsed public state:
        n_alive = #{i : money[i] + contribution[i] > 1}
    Busted seats are stack=1/ante=0 placeholders (money+contribution == 1);
    live seats total their hand-start stack (> 1, except the documented
    1-chip-live-seat collision, characterized by the `collision` subcommand).
    The count is invariant within a hand.
    """

    def __init__(self, champion, specialist, name: str = "ensemble"):
        self.name = name
        self.champion = champion
        self.specialist = specialist
        self.n_alive_log = None  # set to a list to record inferred counts

    @staticmethod
    def infer_n_alive(parsed) -> int:
        return sum(1 for m, c in zip(parsed["money"], parsed["contribution"])
                   if m + c > 1)

    def select_action(self, parsed, state, rng, mode: str = "sample") -> int:
        n_alive = self.infer_n_alive(parsed)
        if self.n_alive_log is not None:
            self.n_alive_log.append(n_alive)
        member = self.specialist if n_alive == 4 else self.champion
        return member.select_action(parsed, state, rng, mode=mode)


# ── shared loading ──────────────────────────────────────────────────────

def _load_world(abstraction_path: str):
    from src.nlhe.abstraction import Abstraction
    from src.nlhe.game_strings import TournamentStructure
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    structure = _RealAnteStructure(TournamentStructure.from_yaml(STRUCTURE))
    abstr = Abstraction.load(abstraction_path)
    return structure, abstr


def _ckpt_policy(name, ckpt, abstr, structure):
    from scripts.eval_pool import CheckpointPolicy
    return CheckpointPolicy(name=name, ckpt_path=ckpt, abstraction=abstr,
                            structure=structure)


def _profile_policy(name: str):
    from scripts.bake_off_real_ante import build_shanky_pool
    pool = [p for p in build_shanky_pool("data/shanky_profiles")
            if p.name == name]
    if not pool:
        raise SystemExit(f"unknown shanky profile: {name}")
    return pool[0]


def _header(args, member_ckpts: dict) -> dict:
    from scripts.sng_baseline import _sha256_of_file, _git_identity
    head, dirty = _git_identity()
    return {
        "record_type": "run_header",
        "probe": "f1_ensemble_probe_20260612",
        "design": "docs/research_program/ENSEMBLE_PROBE_DESIGN.md",
        "member_ckpts": {k: {"path": str(Path(v).resolve()),
                             "sha256": _sha256_of_file(v)}
                         for k, v in member_ckpts.items()},
        "abstraction_sha256": _sha256_of_file(args.abstraction),
        "config": {k: v for k, v in vars(args).items() if k != "func"},
        "git_head": head, "git_dirty": dirty,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }


def _log(msg):
    print(f"{datetime.now().strftime('%H:%M:%S')}  {msg}", flush=True)


# ── p0: bit-identity ────────────────────────────────────────────────────

def cmd_p0(args):
    from scripts.sng_baseline import evaluate_profile
    structure, abstr = _load_world(args.abstraction)
    opp = _profile_policy("killphilmtt")
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    champ_plain = _ckpt_policy("champion", args.champion_ckpt, abstr,
                               structure)
    # two independent instances so the selector genuinely switches objects
    champ_a = _ckpt_policy("champ_a", args.champion_ckpt, abstr, structure)
    champ_b = _ckpt_policy("champ_b", args.champion_ckpt, abstr, structure)
    ens = EnsemblePolicy(champ_a, champ_b, name="ens_self")

    header = _header(args, {"champion": args.champion_ckpt,
                            "specialist(=champion)": args.champion_ckpt})
    (out / "p0_header.json").write_text(json.dumps(header, indent=2))

    results = {}
    for tag, hero in (("plain", champ_plain), ("ensemble", ens)):
        path = out / f"p0_games_{tag}.jsonl"
        with open(path, "w") as fh:
            r = evaluate_profile(hero, opp, structure, n_games=args.games,
                                 master_seed=args.master_seed,
                                 hands_per_level=5, max_hands=200,
                                 mode="sample", games_fh=fh,
                                 log_every=50, log=_log)
        results[tag] = r
        _log(f"  {tag}: net/game={r['hero_net_per_game']:+.4f} "
             f"hands/game={r['mean_hands_per_game']:.1f}")

    a = (out / "p0_games_plain.jsonl").read_bytes()
    b = (out / "p0_games_ensemble.jsonl").read_bytes()
    identical = a == b
    verdict = {"bit_identical": identical, "n_games": args.games,
               "plain": results["plain"], "ensemble": results["ensemble"]}
    (out / "p0_verdict.json").write_text(json.dumps(verdict, indent=2))
    _log(f"P0 bit-identity: {'PASS' if identical else 'FAIL'}")
    return 0 if identical else 1


# ── collision characterization ──────────────────────────────────────────

def cmd_collision(args):
    from scripts.sng_baseline import play_one_hand_sng
    structure, abstr = _load_world(args.abstraction)
    champ_a = _ckpt_policy("champ_a", args.champion_ckpt, abstr, structure)
    champ_b = _ckpt_policy("champ_b", args.champion_ckpt, abstr, structure)
    ens = EnsemblePolicy(champ_a, champ_b)

    cases = [
        ("6_alive", [1500] * 6, 6),
        ("5_alive", [1500, 1500, 1500, 1500, 1500, 0], 5),
        ("4_alive", [1500, 1500, 1500, 1500, 0, 0], 4),
        ("4_alive_uneven", [2000, 2500, 3000, 1500, 0, 0], 4),
        ("5_alive_one_chip", [1500, 1500, 1500, 1500, 1, 0], 5),  # collision
    ]
    level = args.level
    bl = structure.level(level)
    _log(f"level {level}: sb={bl.small_blind} bb={bl.big_blind} "
         f"ante={bl.ante}")
    rows = []
    for name, stacks, true_alive in cases:
        ens.n_alive_log = []
        rng = random.Random(12345)
        try:
            play_one_hand_sng([ens] * 6, structure, list(stacks), level, 0,
                              rng, mode="sample")
            err = None
        except Exception as e:  # characterize, don't crash
            err = f"{type(e).__name__}: {e}"
        inferred = sorted(set(ens.n_alive_log))
        rows.append({"case": name, "stacks": stacks,
                     "true_hand_start_alive": true_alive,
                     "inferred_n_alive_values": inferred,
                     "n_decisions": len(ens.n_alive_log), "error": err})
        _log(f"  {name}: true={true_alive} inferred={inferred} "
             f"decisions={len(ens.n_alive_log)} err={err}")
    ens.n_alive_log = None
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "p0_collision.json").write_text(json.dumps(rows, indent=2))
    return 0


# ── row: one P2 ensemble yardstick row ──────────────────────────────────

def cmd_row(args):
    from scripts.sng_baseline import evaluate_profile
    structure, abstr = _load_world(args.abstraction)
    champ = _ckpt_policy("champion", args.champion_ckpt, abstr, structure)
    spec = _ckpt_policy("specialist", args.specialist_ckpt, abstr, structure)
    ens = EnsemblePolicy(champ, spec)
    opp = _profile_policy(args.profile)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    header = _header(args, {"champion": args.champion_ckpt,
                            "specialist": args.specialist_ckpt})
    _log(f"run header: {json.dumps(header)}")
    games_path = out / f"games_{args.profile}.jsonl"
    with open(games_path, "w") as fh:
        r = evaluate_profile(ens, opp, structure, n_games=args.games,
                             master_seed=args.master_seed,
                             hands_per_level=5, max_hands=200, mode="sample",
                             games_fh=fh, log_every=args.log_every, log=_log)
    r["opponent"] = f"shanky:{args.profile}"
    summary = dict(header)
    summary["record_type"] = "summary"
    summary["result"] = r
    (out / f"summary_{args.profile}.json").write_text(
        json.dumps(summary, indent=2))
    _log(f"  {args.profile}: net/game={r['hero_net_per_game']:+.4f} "
         f"+/- {r['stderr']:.4f} tainted={r['n_tainted']} "
         f"[{r['elapsed_s']:.0f}s]")
    return 0


# ── p1ext: bubble extraction analysis ───────────────────────────────────

def _load_games(paths):
    recs = {}
    for p in paths:
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                recs[r["game"]] = r
    return recs


def _bubble_start_equity(seed, bubble_rows):
    """Replays attacker_extraction_eval's row draw: returns hero's gross
    starting Malmuth-Harville ICM equity for the game with this seed."""
    from scripts.sng_baseline import N_SEATS, PAYOUTS
    from src.nlhe.icm import icm_equity
    row_rng = random.Random(seed ^ 0xB0BB1E)
    row = row_rng.choice(bubble_rows)
    alive = [i for i in range(N_SEATS) if row["stacks"][i] > 0]
    hero_seat = row_rng.choice(alive)
    e0 = icm_equity(row["stacks"], PAYOUTS, eligible=alive)[hero_seat]
    return hero_seat, float(e0)


def _mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def cmd_p1ext(args):
    artifact = json.load(gzip.open(args.bubble_artifact, "rt"))
    bubble_rows = artifact["hand_starts"]
    recs = _load_games(sorted(glob.glob(args.games_glob)))
    ref = (_load_games(sorted(glob.glob(args.ref_glob)))
           if args.ref_glob else {})

    ext, deltas = [], []
    for g, r in sorted(recs.items()):
        if r["tainted"]:
            continue
        hs, e0 = _bubble_start_equity(r["seed"], bubble_rows)
        assert hs == r["hero_seat"], (g, hs, r["hero_seat"])
        x = (r["hero_net"] + 1.0) - e0  # gross final minus gross start ICM
        ext.append(x)
        rr = ref.get(g)
        if rr is not None and not rr["tainted"]:
            deltas.append(x - ((rr["hero_net"] + 1.0) - e0))

    m, se = _mean_se(ext)
    out = {"label": args.label, "n": len(ext),
           "extraction_per_game": m, "stderr": se,
           "z": m / se,
           "s1_bar": "extraction >= +0.040 with z >= 2",
           "s1_pass": bool(m >= 0.040 and m / se >= 2.0)}
    if deltas:
        dm, dse = _mean_se(deltas)
        out["paired_vs_ref"] = {"n": len(deltas), "delta": dm,
                                "stderr": dse, "z": dm / dse}
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


# ── analyze: P2 bars ────────────────────────────────────────────────────

TARGET_ROWS = ["fixedlimitheadsup", "loosenluckymtt", "killphilmtt",
               "modernmikecash", "itmstrikea", "itmstrikec"]
CONTROL_ROWS = ["thefixersng", "littlegreen"]


def _stage4_stats(stage_acc):
    """Pooled n_alive=4 (count, sum, sumsq) from a stage_acc dict keyed
    'n_alive|level'."""
    n = s = ss = 0.0
    for k, v in stage_acc.items():
        if k.split("|")[0] == "4":
            n += v[0]; s += v[1]; ss += v[2]
    return n, s, ss


def cmd_analyze(args):
    """Baseline manifest: JSON {profile: dir}; each dir holds the champion
    POST-FIX (instrument a8933ea) summary.json with results[0] and
    games_<profile>.jsonl. The pre-fix 2026-06-10 merged yardstick is NOT
    used (standing rule, evals/e2_rebaseline_20260612/REPORT.txt: never mix
    pre-fix and post-fix instrument readings)."""
    arm_dir = Path(args.arm_dir)
    manifest = json.loads(Path(args.baseline_manifest).read_text())

    rows = {}
    pooled_target_deltas = []
    b1_n = b1_s = b1_ss = 0.0   # ensemble pooled bubble stage
    c1_n = c1_s = c1_ss = 0.0   # champion pooled bubble stage
    for prof in TARGET_ROWS + CONTROL_ROWS:
        s = json.loads((arm_dir / f"summary_{prof}.json").read_text())
        r = s["result"]
        ens_games = _load_games([arm_dir / f"games_{prof}.jsonl"])
        base_dir = Path(manifest[prof])
        bs = json.loads((base_dir / "summary.json").read_text())
        base_r = bs["results"][0]
        assert base_r["opponent"] == f"shanky:{prof}", (prof, base_r["opponent"])
        champ_games = _load_games([base_dir / f"games_{prof}.jsonl"])
        deltas = [ens_games[g]["hero_net"] - champ_games[g]["hero_net"]
                  for g in sorted(ens_games)
                  if not ens_games[g]["tainted"]
                  and g in champ_games and not champ_games[g]["tainted"]]
        dm, dse = _mean_se(deltas)
        n_ident = sum(1 for d in deltas if d == 0.0)
        en, es, ess = _stage4_stats(r["stage_acc"])
        cn, cs, css = _stage4_stats(base_r["stage_acc"])
        rows[prof] = {
            "ensemble_net": r["hero_net_per_game"],
            "ensemble_se": r["stderr"],
            "champion_net": base_r["hero_net_per_game"],
            "champion_se": base_r["stderr"],
            "paired_delta": dm, "paired_se": dse,
            "paired_z": dm / dse if dse > 0 else float("nan"),
            "n_paired": len(deltas), "n_outcome_identical": n_ident,
            "tainted": r["n_tainted"],
            "bubble_ens": {"hands": en, "mean": es / en if en else None},
            "bubble_champ": {"hands": cn, "mean": cs / cn if cn else None},
        }
        if prof in TARGET_ROWS:
            pooled_target_deltas.extend(deltas)
            b1_n += en; b1_s += es; b1_ss += ess
            c1_n += cn; c1_s += cs; c1_ss += css

    def stage_mean_se(n, s, ss):
        m = s / n
        var = ss / n - m * m
        return m, math.sqrt(max(0.0, var) / n)

    em, ese = stage_mean_se(b1_n, b1_s, b1_ss)
    cm, cse = stage_mean_se(c1_n, c1_s, c1_ss)
    b1_delta = em - cm
    b1_se = math.sqrt(ese * ese + cse * cse)   # unpaired (conservative);
    # stage_acc is pooled per run, no per-game stage records exist on the
    # champion baseline side — noted in the design's B1 row.
    b1_z = b1_delta / b1_se

    b2 = {p: {"z": rows[p]["paired_z"],
              "pass": bool(rows[p]["paired_z"] > -2.0)}
          for p in CONTROL_ROWS}
    kp = rows["killphilmtt"]
    b3_bar = kp["champion_net"] - 2.0 * kp["paired_se"]
    b3_pass = bool(kp["ensemble_net"] >= b3_bar)
    pm, pse = _mean_se(pooled_target_deltas)

    verdict = {
        "arm": args.label,
        "rows": rows,
        "B1": {"metric": "pooled 6-row bubble-stage delta/hand "
                         "(ensemble - champion)",
               "ensemble": {"hands": b1_n, "mean": em, "se": ese},
               "champion": {"hands": c1_n, "mean": cm, "se": cse},
               "delta": b1_delta, "se": b1_se, "z": b1_z,
               "pass": bool(b1_delta > 0 and b1_z >= 2.0)},
        "B2": {"controls": b2,
               "pass": bool(all(v["pass"] for v in b2.values()))},
        "B3": {"ensemble_killphil_net": kp["ensemble_net"],
               "bar": b3_bar, "pass": b3_pass},
        "B4": {"pooled_6row_paired_delta": pm, "se": pse,
               "positive": bool(pm > 0), "note": "sign check, not kill"},
    }
    verdict["arm_pass"] = bool(verdict["B1"]["pass"]
                               and verdict["B2"]["pass"] and b3_pass)
    txt = json.dumps(verdict, indent=2)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt)
    return 0


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--abstraction", default=ABSTRACTION)
    ap.add_argument("--champion-ckpt", default=CHAMPION_CKPT)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("p0")
    p0.add_argument("--games", type=int, default=50)
    p0.add_argument("--master-seed", type=int, default=2026)
    p0.add_argument("--out-dir", required=True)
    p0.set_defaults(func=cmd_p0)

    co = sub.add_parser("collision")
    co.add_argument("--level", type=int, default=4)
    co.add_argument("--out-dir", required=True)
    co.set_defaults(func=cmd_collision)

    row = sub.add_parser("row")
    row.add_argument("--specialist-ckpt", required=True)
    row.add_argument("--profile", required=True)
    row.add_argument("--games", type=int, default=2000)
    row.add_argument("--master-seed", type=int, default=2026)
    row.add_argument("--out-dir", required=True)
    row.add_argument("--log-every", type=int, default=400)
    row.set_defaults(func=cmd_row)

    p1 = sub.add_parser("p1ext")
    p1.add_argument("--games-glob", required=True)
    p1.add_argument("--ref-glob", default=None)
    p1.add_argument("--bubble-artifact", default=BUBBLE_ARTIFACT)
    p1.add_argument("--label", default="")
    p1.add_argument("--out", default=None)
    p1.set_defaults(func=cmd_p1ext)

    an = sub.add_parser("analyze")
    an.add_argument("--arm-dir", required=True)
    an.add_argument("--baseline-manifest", required=True)
    an.add_argument("--label", default="")
    an.add_argument("--out", default=None)
    an.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
