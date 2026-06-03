"""ReBeL self-play iteration driver: one loop step Net_n -> Net_{n+1}.

  1. GENERATE net-leaf samples with Net_n at the leaves (search(Net_n) targets).
  2. MATERIALIZE + TRAIN Net_{n+1} on them (auto-sized + early-stop).
  3. PLATEAU-CHECK: Net_{n+1} vs Net_n (paired) AND vs k=200, across the opponent
     panel (Shanky tight bots + archetypes), via the Gate-2 harness.
  4. VERDICT: Net_{n+1} "climbs" if it beats Net_n beyond noise on the tight rows.
     Plateau = no longer beats Net_n beyond noise.

Composes the existing scripts (rebel_samplegen --leaf net / rebel_train_value_net_v3
/ rebel_gate2 --net-a/--net-b). Each step shells out so a run is resumable by hand.

  .venv/bin/python scripts/rebel_iterate.py --iter 1 \
      --prev-net /workspace/rebel_value_net_full.pt \
      --gen-target 4000000 --hands 500
"""
import sys, os, shutil, subprocess, argparse
sys.path.insert(0, '.')

PY = ".venv/bin/python"


def run(cmd, log=None):
    print(f"[iterate] $ {' '.join(cmd)}", flush=True)
    if log:
        with open(log, "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--prev-net", required=True, help="Net_n (.pt)")
    ap.add_argument("--gen-target", type=int, default=4_000_000)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--hands", type=int, default=500)
    ap.add_argument("--skip-gen", action="store_true", help="reuse existing samples dir")
    ap.add_argument("--skip-train", action="store_true")
    a = ap.parse_args()

    n = a.iter
    sdir = f"/workspace/rebel_iter{n}"
    snap = f"/workspace/rebel_iter{n}.npz"
    new_net = f"/workspace/rebel_value_net_iter{n}.pt"
    k200 = "k200"

    # 1. generate net-leaf samples with Net_n
    if not a.skip_gen:
        run([PY, "scripts/rebel_samplegen.py", "--workers", str(a.workers),
             "--target", str(a.gen_target), "--leaf", "net", "--net-path", a.prev_net,
             "--out", sdir, "--flush-every", "256"])
    # 2. materialize + train Net_{n+1}
    if not a.skip_train:
        run([PY, "-m", "src.rebel.sample_io", sdir, "--out", snap])
        run([PY, "scripts/rebel_train_value_net_v3.py", "--data", snap, "--out", new_net,
             "--logfile", f"/tmp/train_iter{n}.log", "--samples-per-param", "15",
             "--max-epochs", "200", "--patience", "10"])
        # BOUND DISK (MooseFS 19G quota): once Net_{n+1} is trained, the raw samples
        # are spent — keep only the .pt net, delete this iteration's shards + snapshot
        # BEFORE the next batch. Without this the quota fills after ~2 iterations and
        # generation stalls (happened at iter 1).
        if os.path.exists(new_net):
            shutil.rmtree(sdir, ignore_errors=True)
            if os.path.exists(snap):
                os.remove(snap)
            print(f"[iterate] disk bound: deleted {sdir} + {snap}, kept {new_net}", flush=True)
        else:
            print(f"[iterate] WARNING: {new_net} missing — NOT deleting samples", flush=True)
    # 3. plateau-check: Net_{n+1} vs Net_n, and Net_{n+1} vs k=200
    run([PY, "scripts/rebel_gate2.py", "--hands", str(a.hands),
         "--net-a", a.prev_net, "--net-b", new_net], log=f"/workspace/gate2_iter{n}_vs_prev.log")
    run([PY, "scripts/rebel_gate2.py", "--hands", str(a.hands),
         "--net-a", k200, "--net-b", new_net], log=f"/workspace/gate2_iter{n}_vs_k200.log")
    print(f"[iterate] iter {n} done. Compare /workspace/gate2_iter{n}_vs_prev.log "
          f"(climb?) and /workspace/gate2_iter{n}_vs_k200.log (baseline).", flush=True)


if __name__ == "__main__":
    main()
