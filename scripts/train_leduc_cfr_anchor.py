"""Step 0 — train + validate the tabular CFR+ Leduc Nash anchor.

Run as a module (uses src.leduc.* imports):
    python -m scripts.train_leduc_cfr_anchor --iterations 1000 --eval-every 100

PASS GATE: final exploitability < 5 mbb/g (the Nash bar in src/leduc/evaluate.py).
Exits non-zero if the gate fails. Saves the anchor table + metrics to --out.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.leduc.cfr_anchor import train_cfr_plus_anchor, save_anchor

NASH_BAR_MBB = 5.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--out", default=None,
                    help="artifact dir (default runs/leduc_cfr_anchor_<ts>)")
    args = ap.parse_args()

    out = args.out or f"runs/leduc_cfr_anchor_{time.strftime('%Y%m%d_%H%M%S')}"
    t0 = time.time()
    res = train_cfr_plus_anchor(args.iterations, args.eval_every)
    metrics = save_anchor(res, Path(out))
    dt = time.time() - t0

    print(f"CFR+ anchor: {res.iterations} iters in {dt:.1f}s "
          f"({metrics['n_info_states']} info states)")
    for it, e in res.history:
        print(f"  iter {it:5d}  exploitability = {e:9.4f} mbb/g")
    print(f"FINAL exploitability = {res.exploitability_mbb:.4f} mbb/g")
    gate = res.exploitability_mbb < NASH_BAR_MBB
    print(f"PASS GATE (<{NASH_BAR_MBB} mbb/g): {'PASS' if gate else 'FAIL'}")
    print(f"artifact: {out}")
    raise SystemExit(0 if gate else 1)


if __name__ == "__main__":
    main()
