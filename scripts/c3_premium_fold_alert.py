"""Premium-fold tell alert for the C3 retrain (read-only).

Watches the convergence CSV written by monitor_k200_convergence.py and
flags the C3 spec's tell: AA/KK sample-mode fold% > 10% at any
checkpoint past iter 700 (the depth-confusion signature that triggered
the original real-ante investigation).

Usage (one-shot scan or polling):
    python scripts/c3_premium_fold_alert.py --csv runs/c3_retrain_*/convergence.csv
    python scripts/c3_premium_fold_alert.py --csv ... --watch 300
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

THRESHOLD_PCT = 10.0
FROM_ITER = 700


def scan(csv_path: Path) -> tuple[int, list[str]]:
    """Return (rows_checked, alert_lines)."""
    alerts = []
    n = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                it = int(float(row.get("iter", 0)))
            except ValueError:
                continue
            if it <= FROM_ITER:
                continue
            n += 1
            # monitor CSV carries per-hand fold% columns; check every
            # AA/KK fold column present at either depth.
            for col, val in row.items():
                if "fold" not in col.lower():
                    continue
                if not ("aa" in col.lower() or "kk" in col.lower()
                        or "premium" in col.lower()):
                    continue
                try:
                    pct = float(val)
                except (TypeError, ValueError):
                    continue
                if pct > THRESHOLD_PCT:
                    alerts.append(
                        f"ALERT iter {it}: {col} = {pct:.1f}% "
                        f"(> {THRESHOLD_PCT:.0f}%)")
    return n, alerts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--watch", type=int, default=0,
                    help="poll interval seconds; 0 = one-shot")
    args = ap.parse_args()
    path = Path(args.csv)
    while True:
        if path.exists():
            n, alerts = scan(path)
            if alerts:
                for a in alerts:
                    print(a, flush=True)
            else:
                print(f"ok: {n} post-iter-{FROM_ITER} checkpoints, "
                      f"no premium-fold tell", flush=True)
        else:
            print(f"waiting: {path} not present yet", flush=True)
        if not args.watch:
            return 1 if path.exists() and scan(path)[1] else 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
