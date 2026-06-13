"""Abort-counter fix gate (task #10, 2026-06-13): replay a live_dryrun
session's recorded frame stream + statuses through the FallbackWatchdog
and compare the session-abort verdict under LEGACY hand bookkeeping
(every raw card-tuple boundary appends a counter entry — the shipped
behavior up to commit 4c4e4f9) vs the EVIDENCE-HAND bookkeeping (the
fix: hands with no real hero-to-act frame are ignored by the counters).

The replayed watchdog is the CURRENT code (evidence bookkeeping); the
legacy verdict is recomputed by a shadow counter driven by the exact
same boundary/fire event stream, so the two verdicts differ only by the
counting rule under test. The session's own recorded fallback rows are
the ground truth the replay is validated against (armed_seq lineup +
the live run's abort_recommended values must match the LEGACY shadow).

Time base: frame captured_at timestamps (YYYYMMDD_HHMMSS_mmm); 1-second
poll ticks are simulated between frames (the listener's tick path).

Usage: python replay_watchdog_abort.py <session.jsonl> [...]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/quant/pokerbot")

from src.nlhe.integration.fallback import FallbackWatchdog, is_real_to_act


def t_of(captured_at: str) -> float | None:
    try:
        return datetime.strptime(captured_at, "%Y%m%d_%H%M%S_%f").timestamp()
    except (ValueError, TypeError):
        return None


class LegacyShadow:
    """The pre-fix counters: every non-empty card-tuple change closes
    the previous hand into the flags list (no evidence filter)."""

    def __init__(self, window: int, fallbacks: int, consecutive: int):
        self.window, self.fallbacks, self.consecutive = (
            window, fallbacks, consecutive)
        self.cur = None
        self.cur_fb = False
        self.flags = []
        self.hands_seen = 0
        self.abort = False
        self.reason = None

    def on_frame(self, cards):
        if cards and cards != self.cur:
            if self.cur is not None:
                self.flags.append(self.cur_fb)
            self.cur_fb = False
            self.cur = cards
            self.hands_seen += 1

    def on_fire(self):
        self.cur_fb = True
        flags = self.flags + [self.cur_fb]
        if sum(flags[-self.window:]) >= self.fallbacks:
            self.abort = True
            self.reason = (f"{sum(flags[-self.window:])} fallback hands "
                           f"in last {len(flags[-self.window:])}")
        tail = flags[-self.consecutive:]
        if len(tail) == self.consecutive and all(tail):
            self.abort = True
            self.reason = f"fallbacks in {self.consecutive} consecutive hands"


def replay(path: Path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    header = rows[0] if rows and rows[0].get("record_type") == "session_header" else {}
    fb_seconds = header.get("fallback_seconds")
    if fb_seconds is None:
        return None
    v2 = bool(header.get("watchdog_v2"))
    recorded_fb = [r for r in rows if r.get("record_type") == "fallback"]
    frames = [r for r in rows if "raw_record" in r]

    w = FallbackWatchdog(float(fb_seconds), hand_deadline=v2)
    shadow = LegacyShadow(w.abort_window_hands, w.abort_fallbacks,
                          w.abort_consecutive)
    fires = []
    last_t = None
    for fr in frames:
        raw = fr["raw_record"]
        now = t_of(raw.get("captured_at", "")) or (
            (last_t or 0.0) + 0.4)
        # 1 Hz poll ticks between frames (the listener tick path).
        if last_t is not None:
            tk = last_t + 1.0
            while tk < now:
                plan = w.poll(tk)
                if plan is not None:
                    shadow.on_fire()
                    fires.append((plan, tk, None, shadow.abort,
                                  shadow.reason))
                tk += 1.0
        last_t = now
        shadow.on_frame(tuple(raw.get("hero_cards") or []))
        plan = w.observe(raw, fr.get("status", ""), fr.get("seq"), now)
        if plan is not None:
            shadow.on_fire()
            fires.append((plan, now, fr.get("seq"), shadow.abort,
                          shadow.reason))

    return {
        "session": path.name,
        "config": {"fallback_seconds": fb_seconds, "watchdog_v2": v2,
                   "abort_enforce": bool(header.get("abort_enforce"))},
        "recorded": recorded_fb,
        "fires": fires,
        "watchdog": w,
        "shadow": shadow,
    }


def report(res) -> str:
    out = []
    w = res["watchdog"]
    sh = res["shadow"]
    out.append("=" * 72)
    out.append(f"SESSION {res['session']}  config={res['config']}")
    out.append(f"  recorded fallbacks in log: {len(res['recorded'])} "
               f"(live abort_recommended: "
               f"{[r['abort_recommended'] for r in res['recorded']]})")
    out.append(f"  replayed fires: {len(res['fires'])}")
    out.append(f"  {'#':>2} {'armed_seq':>9} {'hand':<12} {'action':<6} "
               f"{'legacy_abort':<13} {'evidence_abort':<15} reason")
    for i, (plan, t, after_seq, leg_abort, leg_reason) in enumerate(
            res["fires"], 1):
        hand = "".join(plan.hand_cards) or "??"
        out.append(
            f"  {i:>2} {str(plan.armed_seq):>9} {hand:<12} "
            f"{plan.action_kind:<6} {str(leg_abort):<13} "
            f"{str(plan.abort_recommended):<15} "
            f"{plan.abort_reason or leg_reason or ''}")
    rec_seqs = [r.get("armed_seq") for r in res["recorded"]]
    rep_seqs = [p.armed_seq for p, *_ in res["fires"]]
    out.append(f"  armed_seq lineup  recorded={rec_seqs}")
    out.append(f"                    replayed={rep_seqs}"
               f"  {'MATCH' if rec_seqs == rep_seqs else 'MISMATCH'}")
    leg_verdict = res["fires"][-1][3] if res["fires"] else False
    # legacy verdict = any fire with legacy abort True
    leg_any = any(f[3] for f in res["fires"])
    new_any = w.abort_recommended
    out.append(f"  VERDICT  legacy abort: {leg_any}   "
               f"evidence-hand abort: {new_any}   "
               f"{'** VERDICT CHANGED **' if leg_any != new_any else '(unchanged)'}")
    out.append(f"  hands_seen  legacy={sh.hands_seen}  evidence={w.hands_seen}")
    return "\n".join(out)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        res = replay(Path(arg))
        if res is None:
            print(f"{arg}: no fallback watchdog armed in header; skipped")
            continue
        print(report(res))
