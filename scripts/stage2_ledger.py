"""STAGE-2 READINESS LEDGER builder (pre-registered, task #9).

Parses EVERY live dry-run session into docs/research_program/STAGE2_LEDGER.md:

  - per-session triage (logs/triage_<session>.txt; regenerated via
    scripts/dryrun_triage.py when missing),
  - stdout fallback / abort lines (logs/live_dryrun_<session>.stdout),
  - safe-fold class histogram from the triage [e] RED FLAGS section,
  - manual interventions from operator pin lists (recorded in
    docs/research_program/EXPERIMENT_LOG.md; mirrored in the
    MANUAL_INTERVENTIONS map below — update it when a new pin list lands),
  - a GAMES-PLAYED counter: each session's raw jsonl is segmented into
    SNG games (stack-reset to ~1500 chips + blind reset to L1 = new game
    boundary; hero bust / 2x-start terminal = outcome where determinable).

Re-runnable: the post-session pipeline calls

    python scripts/stage2_ledger.py

and the ledger doc is rewritten in full. Stdlib-only, no model load.

STAGE-2 GATE (pre-registered in the ledger header): 5 consecutive
sessions with zero never-decided to-act hands AND zero required manual
interventions (N=5 PROPOSED, awaiting operator confirmation).
"""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
OUT_MD = REPO / "docs" / "research_program" / "STAGE2_LEDGER.md"

# Logs that are not operator play sessions.
EXCLUDE_SUBSTRINGS = ("smoke",)

# ── operator-recorded facts (mirror of EXPERIMENT_LOG pin lists) ───────
# session id -> count of required manual interventions (operator had to
# overrule/rescue a spot the pipeline mishandled). None = not recorded.
MANUAL_INTERVENTIONS: dict[str, int | None] = {
    # 2026-06-12 session 5 pin list: 1 known — the seq-1086 AcAh rescue
    # (invariant_fail safe-fold on aces facing an all-in; operator called
    # manually, won with the set).
    "20260612_230149": 1,
}

# session id -> operator session label (from SESSION_LOG/EXPERIMENT_LOG).
OPERATOR_SESSION: dict[str, str] = {
    "20260611_201230": "op-session 3 (G1)",
    "20260611_204751": "op-session 3 (G2-3)",
    "20260612_230149": "op-session 5",
}

# Operator-stated game results used to reconcile the detector where the
# log's trailing frames are OCR garbage (bust hidden in the transition).
# session id -> {game_index_within_session: "W"/"L"}.
OPERATOR_GAME_OVERRIDES: dict[str, dict[int, str]] = {
    # 2026-06-12 op-session 5: "played 3 games, won the last".  Detector
    # finds FOUR stack-reset games with hero decisions in all four
    # (game 1 = seqs 1-172, hero bust at L2, seqs 160-162 read seat1=0);
    # operator count likely omits that first ~13-min bust-out — flagged
    # below.  Game 3 (seqs 383-975) ends in transition garbage with the
    # hero short (last reliable read 1167 < 1x start); operator result
    # ("won only the last") pins it as a loss.
    "20260612_230149": {3: "L"},
}

SUMMARY_RE = re.compile(
    r"^TRIAGE\s+(?P<file>\S+):\s+frames=(?P<frames>\d+)\s+"
    r"decisions=(?P<decisions>\d+)\s+\(fresh (?P<fresh>\d+)/cached (?P<cached>\d+)"
    r"/recovered (?P<recovered>\d+)\)\s+safe_folds=(?P<safe_folds>\d+)\s+"
    r"skips=(?P<skips>\d+)\s+\((?P<skip_pct>[\d.]+)%\)\s+hands=(?P<hands>\d+)\s+"
    r"hands_lost=(?P<hands_lost>\d+)\s+to_act_skipped=(?P<to_act>\d+)\s+"
    r"red_flags=(?P<red_flags>\d+)")

BLINDS_RE = re.compile(r"(\d+)/(\d+),\s*(\d+)\s*Ante")

HERO_SEAT = "seat1"
START_STACK = 1500


# ── triage parsing ─────────────────────────────────────────────────────

def session_id(jsonl: Path) -> str:
    return jsonl.stem.replace("live_dryrun_", "")


def ensure_triage(jsonl: Path) -> Path | None:
    sid = session_id(jsonl)
    triage = LOGS / f"triage_{sid}.txt"
    if triage.exists():
        return triage
    cmd = [sys.executable, str(REPO / "scripts" / "dryrun_triage.py"), str(jsonl)]
    try:
        subprocess.run(cmd, cwd=REPO, capture_output=True, timeout=300)
    except Exception as exc:  # pragma: no cover
        print(f"[ledger] triage regen failed for {jsonl.name}: {exc}")
    return triage if triage.exists() else None


def classify_safe_fold(reason: str) -> str:
    r = reason.lower()
    if "preflop_commit remainder" in r or "pre-hand derivation failed" in r:
        return "replay-derivation (preflop-commit remainder / dead-SB class)"
    if "terminal before hero" in r:
        return "replay terminal-before-hero"
    if "invariant_fail" in r:
        return "invariant_fail"
    if "replay_error" in r:
        return "replay_error (other)"
    return "other"


def parse_triage(triage: Path) -> dict:
    info: dict = {"safe_fold_classes": {}}
    text = triage.read_text(errors="replace")
    for line in text.splitlines():
        m = SUMMARY_RE.match(line.strip())
        if m:
            info.update({k: (v if k in ("file", "skip_pct") else int(v))
                         for k, v in m.groupdict().items()})
    # safe-fold class histogram from the [e] RED FLAGS section
    for m in re.finditer(r"SAFE-FOLD\s+seq=\d+.*?\n(?:.*?\n)*?\s+reason:\s*(.+)",
                         text):
        cls = classify_safe_fold(m.group(1))
        info["safe_fold_classes"][cls] = info["safe_fold_classes"].get(cls, 0) + 1
    return info


def parse_stdout(jsonl: Path) -> dict:
    """Fallback / abort accounting from the listener stdout companion."""
    out = {"fallbacks": None, "aborts_tripped": None, "fallback_detail": []}
    stdout = jsonl.with_suffix(".stdout")
    if not stdout.exists():
        return out
    n_fb = 0
    aborts = 0
    text = stdout.read_text(errors="replace")
    for m in re.finditer(r"session fallback #(\d+)", text):
        n_fb = max(n_fb, int(m.group(1)))
    for m in re.finditer(r"FALLBACK — NOT A MODEL DECISION ⚠\s+action=(\w+)\n"
                         r"\s*hand=(\S+)\s+armed at seq=(\d+)", text):
        out["fallback_detail"].append(
            {"action": m.group(1), "hand": m.group(2), "seq": int(m.group(3))})
    aborts += len(re.findall(r"SESSION ABORT (?:ENFORCED|RECOMMENDED)", text))
    out["fallbacks"] = n_fb
    out["aborts_tripped"] = aborts
    return out


# ── games-played counter ───────────────────────────────────────────────

def _reliable(stacks: dict, bb) -> bool:
    vals = [v for v in (stacks or {}).values() if isinstance(v, int) and v >= 100]
    return bb is not None and len(vals) >= 3


def segment_games(jsonl: Path) -> list[dict]:
    """Segment one session jsonl into SNG games.

    Boundary: a frame at L1 blinds (15/25) with >=5 seats reading a
    fresh-stack value (1430..1505), after the current game has shown a
    blind level above L1 on >=3 reliable frames.  Outcome:
      L  — last reliable hero read is 0 and >=3 of the last 5 reads are 0
           (bust observed), OR operator override;
      W  — last reliable hero read >= 2x start * 0.9 (2x-start terminal);
      ?  — otherwise (bust/win hidden in transition OCR garbage).
    """
    games: list[dict] = []
    cur: dict | None = None
    n_raw_frames = 0

    def close(seq_end):
        nonlocal cur
        if cur is None:
            return
        # drop pure-garbage segments (no decisions and <5 reliable frames)
        if cur["n_decisions"] == 0 and cur["n_reliable"] < 5:
            cur = None
            return
        cur["last_seq"] = seq_end
        reads = cur["hero_reads"]
        tail = reads[-5:]
        if tail and tail[-1] == 0 and sum(1 for v in tail if v == 0) >= 3:
            cur["outcome"] = "L"
        elif (reads and reads[-1] >= 2 * START_STACK * 0.9
                and sum(1 for v in reads[-6:]
                        if v >= 2 * START_STACK * 0.9) >= 2):
            # 2x-start terminal: last read >= 2x start AND >=2 corroborating
            # reads (a lone trailing high value is transition OCR garbage,
            # e.g. 3997 @230149 G3)
            cur["outcome"] = "W"
        else:
            cur["outcome"] = "?"
        cur["hero_final"] = reads[-1] if reads else None
        games.append(cur)
        cur = None

    with open(jsonl) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") in ("session_header", "fallback"):
                continue
            seq = rec.get("seq")
            raw = rec.get("raw_record") or {}
            stacks = raw.get("stacks") or {}
            if stacks:
                n_raw_frames += 1
            m = BLINDS_RE.search(raw.get("blinds") or "")
            sb, bb = (int(m.group(1)), int(m.group(2))) if m else (None, None)
            reliable = _reliable(stacks, bb)
            near = [v for v in stacks.values()
                    if isinstance(v, int) and 1430 <= v <= 1505]
            is_reset = sb == 15 and bb == 25 and len(near) >= 5

            if cur is not None and is_reset and cur["frames_above_l1"] >= 3:
                close(seq)
            if cur is None:
                cur = {"first_seq": seq, "last_seq": seq, "max_bb": 0,
                       "frames_above_l1": 0, "hero_reads": [],
                       "n_reliable": 0, "n_decisions": 0}
            cur["last_seq"] = seq
            if str(rec.get("status", "")).startswith("decision"):
                cur["n_decisions"] += 1
            if reliable:
                cur["n_reliable"] += 1
                if bb and bb > 25:
                    cur["frames_above_l1"] += 1
                    cur["max_bb"] = max(cur["max_bb"], bb)
                hs = stacks.get(HERO_SEAT)
                if isinstance(hs, int):
                    cur["hero_reads"].append(hs)
    close(cur["last_seq"] if cur else None)
    if n_raw_frames == 0:
        return []   # pre-aa2f505 log, no raw_record — segmentation impossible

    sid = session_id(jsonl)
    overrides = OPERATOR_GAME_OVERRIDES.get(sid, {})
    for i, g in enumerate(games, start=1):
        g["index"] = i
        if i in overrides:
            g["outcome"] = overrides[i]
            g["overridden"] = True
        else:
            g["overridden"] = False
    return games


# ── ledger emission ────────────────────────────────────────────────────

def build() -> str:
    jsonls = sorted(p for p in LOGS.glob("live_dryrun_*.jsonl")
                    if not any(s in p.name for s in EXCLUDE_SUBSTRINGS))
    rows = []
    for jl in jsonls:
        sid = session_id(jl)
        triage = ensure_triage(jl)
        t = parse_triage(triage) if triage else {}
        s = parse_stdout(jl)
        games = segment_games(jl)
        rows.append({"sid": sid, "jsonl": jl, "triage": t, "stdout": s,
                     "games": games})

    now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC")
    L: list[str] = []
    L.append("# STAGE-2 READINESS LEDGER")
    L.append("")
    L.append(f"Generated {now} by `python scripts/stage2_ledger.py` "
             "(re-runnable; the post-session pipeline regenerates this file "
             "in full — do not hand-edit rows).")
    L.append("")
    L.append("## Pre-registered STAGE-2 GATE")
    L.append("")
    L.append("**GATE: 5 consecutive sessions with ZERO never-decided to-act "
             "hands AND ZERO required manual interventions.**")
    L.append("")
    L.append("- N=5 is **PROPOSED, awaiting operator confirmation** "
             "(pre-registered 2026-06-13, before any session satisfied it).")
    L.append("- *Never-decided to-act hand* = triage hand-accounting row "
             "\"hands LOST to skips\": raw_record shows hero action buttons "
             "in >=1 frame of the hand and the pipeline produced zero "
             "decisions for it. A fired watchdog fallback does NOT clear the "
             "hand — fallbacks are guaranteed actions, not decisions.")
    L.append("- *Required manual intervention* = operator had to overrule a "
             "pipeline output to avoid a clear EV disaster (pin lists in "
             "EXPERIMENT_LOG; mirrored in `MANUAL_INTERVENTIONS` in the "
             "script). Sessions with no recorded pin list count as "
             "UNKNOWN, not zero — they cannot contribute to the gate streak.")
    L.append("- Fallback fires and safe-folds are tracked as trend columns "
             "but do not gate (they are the safety net working as designed).")
    L.append("")
    L.append("## Per-session ledger")
    L.append("")
    L.append("| session | op-session | frames | hands | decisions | "
             "never-decided hands | to-act frames skipped | fallback fires | "
             "aborts tripped | safe-folds | manual interv. | red flags | "
             "gate-clean |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    gate_streak = 0
    streaks = []
    for r in rows:
        t, s = r["triage"], r["stdout"]
        if not t or "frames" not in t:
            L.append(f"| {r['sid']} | {OPERATOR_SESSION.get(r['sid'], '')} "
                     f"| 0 | 0 | 0 | – | – | – | – | – | – | – | empty log |")
            streaks.append(None)
            continue
        mi = MANUAL_INTERVENTIONS.get(r["sid"])
        fb = s["fallbacks"]
        clean = (t["hands_lost"] == 0 and mi == 0)
        known = mi is not None
        if t["frames"] == 0:
            verdict = "empty"
        elif clean and known:
            verdict = "YES"
        elif t["hands_lost"] == 0 and mi is None:
            verdict = "no (interv. unknown)"
        else:
            verdict = "no"
        sf = t.get("safe_fold_classes", {})
        sf_str = "; ".join(f"{v}x {k.split(' (')[0]}" for k, v in
                           sorted(sf.items())) or "0"
        L.append(
            f"| {r['sid']} | {OPERATOR_SESSION.get(r['sid'], '')} "
            f"| {t['frames']} | {t['hands']} | {t['decisions']} "
            f"| {t['hands_lost']} | {t['to_act']} "
            f"| {fb if fb is not None else 'n/a'} "
            f"| {s['aborts_tripped'] if s['aborts_tripped'] is not None else 'n/a'} "
            f"| {t['safe_folds']} ({sf_str}) "
            f"| {mi if mi is not None else 'unknown'} | {t['red_flags']} "
            f"| {verdict} |")
        streaks.append(bool(clean and known))

    # trailing gate streak
    for v in reversed(streaks):
        if v:
            gate_streak += 1
        else:
            break

    L.append("")
    L.append("## Trend")
    L.append("")
    valid = [r for r in rows if r["triage"].get("frames")]
    if valid:
        lost = [r["triage"]["hands_lost"] for r in valid]
        L.append(f"- Sessions on record: {len(valid)} "
                 f"(+{len(rows) - len(valid)} empty logs).")
        L.append(f"- Never-decided hands per session (chronological): "
                 f"{lost} — latest = {lost[-1]}.")
        fbs = [r["stdout"]["fallbacks"] for r in valid
               if r["stdout"]["fallbacks"] is not None]
        L.append(f"- Fallback fires (sessions with stdout): {fbs}.")
        L.append(f"- **Current gate streak: {gate_streak} / 5.** "
                 "No session yet combines zero never-decided hands with a "
                 "recorded zero-intervention pin list.")
    L.append("")

    # games section
    L.append("## Games played (all-time, detector + operator reconciliation)")
    L.append("")
    L.append("Game boundary = stack reset to ~1500 for >=5 seats at L1 "
             "blinds (15/25) after the previous game escalated past L1. "
             "Outcome: L = hero bust observed (trailing seat1=0 reads); "
             "W = 2x-start terminal (last reliable hero read >= 2700); "
             "? = end hidden in transition OCR garbage. Games begun between "
             "listener runs are not counted, and pre-aa2f505 logs (no "
             "raw_record: the four 2026-06-08 logs) cannot be segmented at "
             "all (coverage caveat).")
    L.append("")
    L.append("| session | games | W | L | ? | per-game detail "
             "(first-last seq, max BB, hero final, outcome) |")
    L.append("|---|---|---|---|---|---|")
    tw = tl = tu = tg = 0
    for r in rows:
        gs = r["games"]
        if not gs:
            continue
        w = sum(1 for g in gs if g["outcome"] == "W")
        l = sum(1 for g in gs if g["outcome"] == "L")
        u = sum(1 for g in gs if g["outcome"] == "?")
        tw, tl, tu, tg = tw + w, tl + l, tu + u, tg + len(gs)
        det = "; ".join(
            f"G{g['index']} {g['first_seq']}-{g['last_seq']} bb{g['max_bb']} "
            f"hero={g['hero_final']} {g['outcome']}"
            + ("*" if g["overridden"] else "")
            for g in gs)
        L.append(f"| {r['sid']} | {len(gs)} | {w} | {l} | {u} | {det} |")
    L.append("")
    L.append(f"**All-time totals: {tg} tournaments — {tw} W / {tl} L / "
             f"{tu} unknown.**  (`*` = outcome set by operator statement, "
             "see OPERATOR_GAME_OVERRIDES.)")
    L.append("")
    L.append("### Calibration / reconciliation notes")
    L.append("")
    L.append("- 2026-06-12 op-session 5 (230149): operator ground truth = "
             "\"played 3 games, won the last\". Detector finds **4** games; "
             "the extra one is seqs 1-172 (~19:02-19:15 local): 7 hero "
             "decision spots (8h6d, 2cQc, 9dAc, Qh8d, 6h9s, Qc8s, 3cQh), "
             "hero short by L2 and seat1 reads 0 at seqs 160-162, fresh "
             "1500-stack table at seq 173. Evidence says the operator "
             "count omitted this first quick bust-out — **awaiting operator "
             "confirmation**. Outcomes agree: every game before the last "
             "was a loss; the last was the win (hero ~3617 at L4).")
    L.append("- op-session 3 (204751): SESSION_LOG records \"won G1+G3\" "
             "(G2 a loss), but the log shows hero ending BOTH 204751 games "
             "in the final 3 (5965 and 5887 chips at 3-handed terminal) — "
             "either a G-numbering mismatch (an unlogged game in the "
             "20:32-20:47 listener gap) or a SESSION_LOG error. Flagged, "
             "not silently corrected.")
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = build()
    OUT_MD.write_text(md)
    print(f"[ledger] wrote {OUT_MD} ({len(md.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
