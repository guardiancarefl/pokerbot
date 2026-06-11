"""Watchdog firing simulation over recorded sessions (v1 vs v2 gate).

Feeds each session log's frames into a FallbackWatchdog with a
simulated clock derived from the frames' captured_at timestamps,
polling at 1s steps between frames (standing in for the live
heartbeat/tick path). Statuses come from the recorded session, so the
sim asks exactly: "given what the pipeline did, when would the
watchdog have fired?"

Caveat: live firing used listener wall-clock (capture + transport +
processing); the sim uses capture time, so firing instants can shift
by transport jitter. Firing SPOTS (armed hand + approximate seq) are
the comparison unit, not millisecond timing.

Usage:
    python -m scripts.replay_watchdog logs/live_dryrun_*.jsonl \
        [--seconds 7.0] [--v2]
With --compare, runs BOTH v1 and v2 and prints the firing-set diff
(the watchdog-v2 acceptance gate: v2 must reproduce every v1 firing
spot and may ADD spots v1 missed — e.g. the 9cAd intermittent-
visibility gap).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.integration.fallback import FallbackWatchdog  # noqa: E402


def ts(captured_at: str) -> float | None:
    try:
        return datetime.strptime(captured_at,
                                 "%Y%m%d_%H%M%S_%f").timestamp()
    except (ValueError, TypeError):
        return None


def simulate(log_path: str, seconds: float, v2: bool) -> list[dict]:
    w = FallbackWatchdog(seconds, hand_deadline=v2)
    fired = []
    last_t = None
    for rec in (json.loads(l) for l in open(log_path) if l.strip()):
        raw = rec.get("raw_record")
        if raw is None:
            continue
        t = ts(rec.get("captured_at", "")) or last_t
        if t is None:
            continue
        # poll at 1s steps across the gap (heartbeat/tick stand-in)
        if last_t is not None and t > last_t:
            pt = last_t + 1.0
            while pt < t:
                plan = w.poll(pt)
                if plan is not None:
                    fired.append({"via": "poll", "at": pt, "plan": plan})
                pt += 1.0
        plan = w.observe(raw, rec.get("status", ""), rec.get("seq"), t)
        if plan is not None:
            fired.append({"via": "frame", "at": t, "plan": plan})
        last_t = t
    return fired


def fmt(f) -> str:
    p = f["plan"]
    return (f"hand={''.join(p.hand_cards) or '?':<6s} armed_seq={p.armed_seq:<5} "
            f"action={p.action_kind:<5s} waited={p.waited_seconds:5.1f}s "
            f"via={f['via']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    for lp in args.logs:
        name = Path(lp).stem.replace("live_dryrun_", "")
        if not args.compare:
            for f in simulate(lp, args.seconds, args.v2):
                print(f"{name}  {fmt(f)}")
            continue
        v1 = simulate(lp, args.seconds, v2=False)
        v2 = simulate(lp, args.seconds, v2=True)
        key = lambda f: (tuple(f["plan"].hand_cards), f["plan"].armed_seq)
        k1 = {key(f) for f in v1}
        k2 = {key(f) for f in v2}
        print(f"== {name}: v1 fires {len(v1)}, v2 fires {len(v2)}")
        for f in v1:
            mark = "" if key(f) in k2 else "   *** LOST IN V2 ***"
            print(f"   v1 {fmt(f)}{mark}")
        for f in v2:
            if key(f) not in k1:
                print(f"   v2 ADDS {fmt(f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
