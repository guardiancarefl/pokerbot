"""Aggregate sharded TG3 runs (EXP_H1 H1.2) into the registered 24k verdict.

Reads tg3_24k_shard{K}.json[.games.jsonl] produced by scripts.tail_floor_ab
--shards N --shard K, verifies exact once-each coverage of games
0..games-1 (CRN seeds are shard-invariant — gate in tg3_harness_gates.txt),
reuses tail_floor_ab.summarize() for the stats, evaluates the pre-committed
F-TG3 ship bars, and writes the combined JSON + concatenated games.jsonl.

Usage:
  python -m scripts.tg3_aggregate \
      --pattern evals/h1_tail_floor_20260612/tg3_24k_shard{K}.json \
      --shards 8 --games 24000 \
      --out evals/h1_tail_floor_20260612/tg3_24k.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.tail_floor_ab import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help="shard JSON path with {K} placeholder")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--games", type=int, default=24000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = []
    shard_cfgs = []
    elapsed_total = 0.0
    elapsed_max = 0.0
    for k in range(args.shards):
        jpath = Path(args.pattern.replace("{K}", str(k)))
        gpath = Path(str(jpath) + ".games.jsonl")
        shard = json.loads(jpath.read_text())
        shard_cfgs.append(shard["config"])
        elapsed_total += shard["elapsed_s"]
        elapsed_max = max(elapsed_max, shard["elapsed_s"])
        with gpath.open() as f:
            for line in f:
                records.append(json.loads(line))

    # Coverage gate: every game exactly once, identical config across shards.
    idxs = [r["game"] for r in records]
    assert sorted(idxs) == list(range(args.games)), (
        f"coverage FAIL: {len(idxs)} records, "
        f"{len(set(idxs))} unique, expected 0..{args.games - 1}")
    base = {k: v for k, v in shard_cfgs[0].items() if k != "shard"}
    for c in shard_cfgs[1:]:
        assert {k: v for k, v in c.items() if k != "shard"} == base, \
            "config mismatch across shards"

    records.sort(key=lambda r: r["game"])
    ns = argparse.Namespace(
        games=args.games, base_seed=base["base_seed"], shards=args.shards,
        shard=-1,  # aggregate marker
        tau=(base["tau"] or 0.0), threshold_bb=base["threshold_bb"],
        hpl=base["hpl"], max_hands=base["max_hands"],
        starting_stack=base["starting_stack"])
    summary = summarize(records, ns, elapsed_s=elapsed_total)
    summary["aggregate_of_shards"] = args.shards
    summary["elapsed_s_wall_max_shard"] = elapsed_max

    ag, dv = summary["all_games"], summary["diverged_only"]
    bars = {
        "all_games_z_gt_minus2": ag["z"] is not None and ag["z"] > -2.0,
        "diverged_mean_positive": dv["mean"] is not None and dv["mean"] > 0,
        "diverged_z_ge_2": dv["z"] is not None and dv["z"] >= 2.0,
    }
    summary["ship_bars"] = bars
    summary["ship_verdict"] = "PASS" if all(bars.values()) else "FAIL"

    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    with (Path(str(out) + ".games.jsonl")).open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(json.dumps({k: summary[k] for k in (
        "games_completed", "all_games", "diverged_only", "n_diverged",
        "diverged_frac", "firing", "ship_bars", "ship_verdict")}, indent=2))


if __name__ == "__main__":
    main()
