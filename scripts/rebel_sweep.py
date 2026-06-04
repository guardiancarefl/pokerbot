"""Experiment-sweep around the trusted Gate-2 plateau-check ("smart spaghetti").

Every variant is run as B against the SAME fixed baseline A (130k net, depth-3,
K=150, linear, uniform leaf-belief) over the SAME seeds + panel — so the
tight-matchup Δ (B-A) is directly comparable across variants. Cheap-scan defaults
to small N (~120 hands) for a directional "which flicker?" read, not a verdict.

  # on each box, run an assigned subset:
  .venv/bin/python scripts/rebel_sweep.py --run d4,k50,d4k50 --hands 120
  # then aggregate all boxes' logs into one table:
  .venv/bin/python scripts/rebel_sweep.py --report
"""
import sys, os, glob, argparse, subprocess, re
sys.path.insert(0, '.')
PY = ".venv/bin/python"
NET130 = "/workspace/rebel_value_net_full.pt"
OUT = "/workspace/sweep"
TIGHT = ["KillPhilMTT", "timidtom", "NIT", "TAG"]

# Baseline A (fixed for every run): 130k net, depth 3, K=150, linear, uniform belief.
BASE = dict(net_a=NET130, depth_a=3, kiters_a=150, weighting_a="linear")

# variant name -> B-side knobs (only the listed key differs from baseline). PROBE_READY
# variants are pure config; others need a component built first (flagged build=...).
VARIANTS = {
    # --- probe-ready (config only) ---
    "d4":      dict(net_b=NET130, depth_b=4, kiters_b=150, weighting_b="linear"),
    "d5":      dict(net_b=NET130, depth_b=5, kiters_b=150, weighting_b="linear"),
    "k50":     dict(net_b=NET130, depth_b=3, kiters_b=50,  weighting_b="linear"),
    "vanilla": dict(net_b=NET130, depth_b=3, kiters_b=150, weighting_b="uniform"),
    "d4k50":   dict(net_b=NET130, depth_b=4, kiters_b=50,  weighting_b="linear"),
    # --- need a small/real build (placeholders; not run until built) ---
    "warmstart_off": dict(build="solver flag: start CFR regret at 0 vs blueprint-adv"),
    "precise_belief": dict(build="reach-posterior belief at leaves in the resolver"),
    "opp_model":      dict(build="ODCFR opponent-modeling bias in the leaf eval"),
    "lowvar_target":  dict(build="lower-variance leaf target -> regen + retrain net"),
}


def run_variant(name, hands, seed):
    v = VARIANTS[name]
    if "build" in v:
        print(f"[sweep] {name}: NEEDS BUILD ({v['build']}) — skipped"); return
    os.makedirs(OUT, exist_ok=True)
    log = f"{OUT}/sweep_{name}.log"
    cmd = [PY, "scripts/rebel_gate2.py", "--hands", str(hands), "--seed", str(seed),
           "--net-a", BASE["net_a"], "--depth-a", str(BASE["depth_a"]),
           "--kiters-a", str(BASE["kiters_a"]), "--weighting-a", BASE["weighting_a"],
           "--net-b", v["net_b"], "--depth-b", str(v["depth_b"]),
           "--kiters-b", str(v["kiters_b"]), "--weighting-b", v["weighting_b"]]
    print(f"[sweep] running {name} -> {log}", flush=True)
    with open(log, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.DEVNULL)


def report():
    rows = {}
    for log in sorted(glob.glob(f"{OUT}/sweep_*.log")):
        name = os.path.basename(log)[6:-4]
        d = {}
        for line in open(log):
            m = re.match(r"\s*(\w+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+([+-]\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)", line)
            if m and m.group(1) in TIGHT:
                d[m.group(1)] = (float(m.group(4)), float(m.group(5)), m.group(6))  # delta, noise, verdict
        if d:
            rows[name] = d
    print(f"\n=== SWEEP RESULTS — Δ(variant − baseline) on TIGHT matchups (flicker if |Δ|>2·SE) ===")
    print(f"{'variant':<16} " + " ".join(f"{t:>12}" for t in TIGHT) + "   flickers")
    print("-" * 90)
    for name, d in rows.items():
        cells = []; nflick = 0
        for t in TIGHT:
            if t in d:
                dl, se, verd = d[t]
                star = "*" if abs(dl) > se else " "
                if star == "*": nflick += 1
                cells.append(f"{dl:>+9.4f}{star}  ")
            else:
                cells.append(f"{'—':>12} ")
        print(f"{name:<16} " + "".join(cells) + f"   {nflick}/{len(TIGHT)}")
    print("\n(* = beyond noise band on that matchup. Escalate variants with flickers to full N=500.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="", help="comma list of variant names to run (or 'all')")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--hands", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    if a.report:
        report()
    elif a.run:
        names = [n for n in VARIANTS if "build" not in VARIANTS[n]] if a.run == "all" else a.run.split(",")
        for n in names:
            run_variant(n.strip(), a.hands, a.seed)
        print("[sweep] done. aggregate with --report", flush=True)
