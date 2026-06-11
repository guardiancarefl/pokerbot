"""Session triage report for live dry-run logs.

Input:  one logs/live_dryrun_*.jsonl (the full audit-trail log written by
        scripts/run_live_dryrun.py — every frame, including skips).
Output: a plain-text report at logs/triage_<session>.txt (override with
        --out) and the same text on stdout. One paste-back summary line
        at the end.

Stdlib-only, no model load — safe to run mid-session on a live log.

Sections
  a. Header echo — session_header (mode/seed/ckpt sha). Loudly warns if
     mode != sample, ckpt sha != deployed b79e82dd…, or header missing
     (pre-aa2f505 logs have none).
  b. Frame accounting — totals, decision split (fresh/cached/recovered),
     skips by status, and a sub-cause histogram of skip_data_quality
     (the blinds-string OCR failure class gets its own line).
  c. Hand accounting — hands seen vs hands with >=1 hero decision vs
     hands lost entirely to skips (the OCR-loss metric). Segmentation
     uses raw_record hero-card/dealer/board evidence (per the corrected
     2026-06-09 analysis — (dealer_seat, level) keying is blind to hands
     whose every frame lost the dealer button).
  d. Decision digest — per decision: seq, street, cards, board, facing,
     chosen action, fresh-vs-cached, floor fired. Floor firings are NOT
     in the JSONL (they go to stdout); if a companion .stdout file sits
     next to the log it is parsed and joined by captured_at.
  e. Red flags — safe-folds (full frame context), anchor_refused,
     nonzero invariant_deltas, FOLD-while-facing-0 (impossible after the
     check-when-free floor), and reconstruction-failure signatures
     (seq-192 bet-as-0 / seq-170 raise-under-reconstructed classes).
  f. Summary line for paste-back.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEPLOYED_CKPT_SHA_PREFIX = "b79e82dd"

DECISION_STATUSES = (
    "decision", "decision_cached",
    "decision_recovered", "decision_recovered_cached",
)
FRESH = ("decision", "decision_recovered")
STREETS = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


# ── skip_data_quality sub-cause classifier ─────────────────────────────

SKIP_CLASSES = [
    # (label, predicate substring). Order matters; first match wins.
    ("blinds-string OCR failure (ScraperParseError SB/BB/ante)",
     "could not extract SB/BB/ante from blinds string"),
    ("scraper suspect frame", "ScraperSuspect"),
    ("dealer field missing/empty", "dealer field missing/empty"),
    ("dealer points to non-alive/empty seat", "dealer points to seat"),
    ("n_alive below playable range", "below the trained model"),
    ("hero alive but no cards visible", "hero alive but no cards visible"),
]


def classify_skip(reason: str) -> str:
    for label, needle in SKIP_CLASSES:
        if needle in reason:
            return label
    # collapse volatile bits so the 'other' bucket still aggregates
    r = re.sub(r"captured_at=\S+", "", reason)
    r = re.sub(r"\d+", "N", r)
    return f"other: {r.strip()[:80]}"


# ── stdout [FLOOR] join ────────────────────────────────────────────────

FLOOR_RE = re.compile(r"\[FLOOR\] fired=\[([^\]]*)\]\s+(.*)")
BANNER_RE = re.compile(r"captured (\d{8}_\d{6}_\d{3})")


def parse_floor_firings(stdout_path: Path) -> dict[str, str]:
    """captured_at -> '[FLOOR] fired=…' detail. A [FLOOR] line prints
    during make_decision, immediately before the decision banner that
    carries `captured <ts>`; attach each firing to the next banner ts."""
    firings: dict[str, str] = {}
    pending: list[str] = []
    try:
        lines = stdout_path.read_text(errors="replace").splitlines()
    except OSError:
        return firings
    for line in lines:
        m = FLOOR_RE.search(line)
        if m:
            pending.append(f"{m.group(1)} ({m.group(2).strip()})")
            continue
        if pending:
            b = BANNER_RE.search(line)
            if b:
                firings[b.group(1)] = "; ".join(pending)
                pending = []
    return firings


# ── hand segmentation ──────────────────────────────────────────────────

def hero_pair(rec: dict) -> tuple | None:
    raw = rec.get("raw_record") or {}
    cards = raw.get("hero_cards") or rec.get("hero_cards") or []
    if len(cards) == 2:
        return tuple(sorted(cards))
    return None


def raw_dealer(rec: dict):
    raw = rec.get("raw_record") or {}
    return raw.get("dealer") or None


def board_len(rec: dict) -> int | None:
    raw = rec.get("raw_record") or {}
    board = raw.get("board")
    if board is None:
        board = rec.get("board")
    return len(board) if board is not None else None


def hero_to_act_evidence(rec: dict) -> bool:
    """Raw-record evidence hero had action buttons this frame — works
    even when the bridge skipped the frame for data quality."""
    raw = rec.get("raw_record") or {}
    action = raw.get("action") or {}
    if action.get("buttons"):
        return True
    controls = raw.get("controls") or {}
    return bool(controls.get("present"))


def segment_hands(records: list[dict]) -> list[dict]:
    """Greedy boundary detection: a new hand starts when the raw hero
    pair changes to a DIFFERENT non-empty pair, the raw dealer string
    changes (non-null -> different non-null), or the board shrinks."""
    hands: list[dict] = []
    cur = None
    prev_dealer = None
    prev_board_len = None
    for rec in records:
        pair = hero_pair(rec)
        dealer = raw_dealer(rec)
        blen = board_len(rec)
        boundary = cur is None
        if cur is not None:
            if pair is not None and cur["pair"] is not None and pair != cur["pair"]:
                boundary = True
            if dealer is not None and prev_dealer is not None and dealer != prev_dealer:
                boundary = True
            if (blen is not None and prev_board_len is not None
                    and blen < prev_board_len):
                boundary = True
        if boundary:
            cur = {"pair": pair, "frames": [], "first_seq": rec.get("seq"),
                   "last_seq": rec.get("seq")}
            hands.append(cur)
        if cur["pair"] is None and pair is not None:
            cur["pair"] = pair
        cur["frames"].append(rec)
        cur["last_seq"] = rec.get("seq")
        if dealer is not None:
            prev_dealer = dealer
        if blen is not None:
            prev_board_len = blen

    # Merge pair-less fragments (between-hands dead air: board clears a
    # frame or two before the dealer button and hero cards advance) into
    # the FOLLOWING hand; a trailing pair-less fragment merges backward.
    merged: list[dict] = []
    pending: list[dict] = []
    for h in hands:
        if h["pair"] is None:
            pending.append(h)
            continue
        if pending:
            h["frames"] = [f for frag in pending
                           for f in frag["frames"]] + h["frames"]
            h["first_seq"] = pending[0]["first_seq"]
            pending = []
        merged.append(h)
    for frag in pending:  # trailing fragments
        if merged:
            merged[-1]["frames"].extend(frag["frames"])
            merged[-1]["last_seq"] = frag["last_seq"]
        else:
            merged.append(frag)
    return merged


# ── invariant-delta signature classifier ───────────────────────────────

def delta_signatures(deltas) -> list[str]:
    """Known reconstruction-failure classes from the 152756 postmortem
    (seq-170 / seq-192). Each delta is [field, scraper_val, recon_val]."""
    sigs = []
    for d in deltas or []:
        try:
            field, scraper_val, recon_val = d[0], d[1], d[2]
        except (IndexError, TypeError):
            continue
        if not str(field).startswith("bet["):
            continue
        try:
            s, r = int(scraper_val), int(recon_val)
        except (TypeError, ValueError):
            continue
        if s > 0 and r == 0:
            sigs.append(f"bet-as-0 (seq-192 class): {field} scraper={s} recon=0")
        elif s > r > 0:
            sigs.append(f"raise-under-reconstructed / min-raise-as-call "
                        f"(seq-170 class): {field} scraper={s} recon={r}")
    return sigs


# ── report ─────────────────────────────────────────────────────────────

def fmt_action(rec: dict) -> str:
    ca = rec.get("client_action") or {}
    kind = ca.get("kind", "?")
    chip = ca.get("chip_amount")
    s = kind.upper() + (f" {chip}" if chip is not None else "")
    mult = rec.get("client_action_bb_mult")
    if mult is not None:
        s += f" ({mult:.1f}xBB)"
    return s


def frame_context_lines(rec: dict, indent: str = "      ") -> list[str]:
    raw = rec.get("raw_record") or {}
    lines = [
        f"{indent}level=L{rec.get('level')} blinds={rec.get('blinds')} "
        f"street={STREETS.get(rec.get('street_idx'), '?')} "
        f"n_alive={rec.get('n_alive')}",
        f"{indent}hero_cards={''.join(rec.get('hero_cards') or []) or '??'} "
        f"board={' '.join(rec.get('board') or []) or '(none)'} "
        f"pot={rec.get('pot_total')} hero_stack={rec.get('hero_stack')} "
        f"facing_bet={rec.get('facing_bet')}",
        f"{indent}reason: {rec.get('skip_reason')}",
    ]
    if raw.get("stacks"):
        lines.append(f"{indent}raw stacks={raw['stacks']}")
    if raw.get("bets"):
        bets = {k: v for k, v in raw["bets"].items() if v}
        lines.append(f"{indent}raw bets={bets or '{}'} dealer={raw.get('dealer')}")
    for d in rec.get("invariant_deltas") or []:
        lines.append(f"{indent}delta {d[0]}: scraper={d[1]} recon={d[2]}")
    return lines


def triage(log_path: Path, stdout_path: Path | None) -> tuple[str, str]:
    header = None
    records: list[dict] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") == "session_header":
                header = rec
                continue
            if "seq" not in rec:
                continue
            records.append(rec)

    session = log_path.name
    out: list[str] = []
    warn: list[str] = []
    rule = "=" * 72

    out += [rule, f"DRY-RUN TRIAGE — {session}", rule, ""]

    # a. header echo ----------------------------------------------------
    out.append("[a] SESSION HEADER")
    if header is None:
        msg = ("!! WARNING: no session_header record — pre-aa2f505 log; "
               "mode/seed/ckpt provenance NOT verifiable from the log")
        out.append(f"    {msg}")
        warn.append(msg)
    else:
        out.append(f"    mode={header.get('mode')}  seed={header.get('seed')}")
        out.append(f"    ckpt={header.get('ckpt_path')}")
        out.append(f"    ckpt_sha256={header.get('ckpt_sha256')}")
        out.append(f"    abstraction_sha256={header.get('abstraction_sha256')}")
        out.append(f"    floors={header.get('floors')}")
        out.append(f"    git_head={header.get('git_head')} "
                   f"dirty={header.get('git_dirty')} "
                   f"started={header.get('started_utc')}")
        if header.get("mode") != "sample":
            msg = (f"!! WARNING: mode={header.get('mode')!r} != 'sample' — "
                   "sample is the deployment invariant (argmax was the one "
                   "real deployment bug)")
            out.append(f"    {msg}")
            warn.append(msg)
        sha = header.get("ckpt_sha256") or ""
        if not sha.startswith(DEPLOYED_CKPT_SHA_PREFIX):
            msg = (f"!! WARNING: ckpt sha {sha[:16]}… is NOT the deployed "
                   f"{DEPLOYED_CKPT_SHA_PREFIX}… checkpoint")
            out.append(f"    {msg}")
            warn.append(msg)
    out.append("")

    # b. frame accounting -----------------------------------------------
    status_counts = Counter(r.get("status") for r in records)
    n_total = len(records)
    n_fresh = status_counts["decision"]
    n_cached = status_counts["decision_cached"]
    n_rec = status_counts["decision_recovered"]
    n_rec_cached = status_counts["decision_recovered_cached"]
    n_dec_all = n_fresh + n_cached + n_rec + n_rec_cached
    n_safe = status_counts["safe_fold"]
    skip_statuses = sorted(k for k in status_counts
                           if k and k.startswith("skip"))
    n_skip = sum(status_counts[k] for k in skip_statuses)

    out.append("[b] FRAME ACCOUNTING")
    out.append(f"    total frames        {n_total}")
    out.append(f"    decisions           {n_dec_all}  "
               f"(fresh {n_fresh} / cached {n_cached} / "
               f"recovered {n_rec} / recovered_cached {n_rec_cached})")
    out.append(f"    safe_fold           {n_safe}")
    for k in skip_statuses:
        out.append(f"    {k:<19s} {status_counts[k]}")
    dq = [r for r in records if r.get("status") == "skip_data_quality"]
    if dq:
        out.append(f"    skip_data_quality sub-causes ({len(dq)}):")
        sub = Counter(classify_skip(r.get("skip_reason") or "") for r in dq)
        for label, n in sub.most_common():
            out.append(f"        {n:>5d}  {label}")
    out.append("")

    # c. hand accounting ------------------------------------------------
    hands = segment_hands(records)
    hands_with_dec = [h for h in hands
                      if any(r.get("status") in DECISION_STATUSES
                             for r in h["frames"])]
    lost_hands = []
    for h in hands:
        if h in hands_with_dec:
            continue
        evid = [r for r in h["frames"] if hero_to_act_evidence(r)]
        if evid:
            lost_hands.append((h, evid))

    # Hero-to-act moments lost ANYWHERE (incl. mid-hand blackouts inside
    # hands that did get earlier decisions — the verify1 seq 83-94 class).
    lost_moments = [r for r in records
                    if hero_to_act_evidence(r)
                    and (r.get("status") or "").startswith("skip")]

    out.append("[c] HAND ACCOUNTING  (raw_record card/dealer/board segmentation)")
    out.append(f"    hands seen                       {len(hands)}")
    out.append(f"    hands with >=1 hero decision     {len(hands_with_dec)}")
    out.append(f"    hero-to-act frames skipped       {len(lost_moments)}  "
               f"(action buttons visible in raw_record, frame skipped)")
    out.append(f"    hands LOST to skips (OCR-loss)   {len(lost_hands)}  "
               f"(hero-to-act evidence present, zero decisions)")
    for h, evid in lost_hands:
        pair = "".join(h["pair"]) if h["pair"] else "??"
        reasons = Counter(classify_skip(r.get("skip_reason") or "")
                          for r in evid
                          if (r.get("status") or "").startswith("skip"))
        sf = sum(1 for r in evid if r.get("status") == "safe_fold")
        detail = ", ".join(f"{lbl} x{n}" for lbl, n in reasons.most_common())
        if sf:
            detail = (detail + ", " if detail else "") + f"safe_fold x{sf}"
        out.append(f"        seq {h['first_seq']}-{h['last_seq']}  "
                   f"hero={pair}  killed by: {detail or 'unknown'}")
    out.append("")

    # d. decision digest --------------------------------------------------
    floor_map: dict[str, str] = {}
    floor_note = "no .stdout companion found — floor column unavailable"
    if stdout_path and stdout_path.exists():
        floor_map = parse_floor_firings(stdout_path)
        floor_note = (f"floors joined from {stdout_path.name} "
                      f"({len(floor_map)} firing(s))")
    out.append(f"[d] DECISION DIGEST  ({floor_note})")
    out.append(f"    {'seq':>5s} {'mode':<7s} {'street':<8s} {'cards':<6s} "
               f"{'board':<16s} {'facing':<7s} {'action':<22s} floor")
    for r in records:
        if r.get("status") not in DECISION_STATUSES:
            continue
        mode = {"decision": "sample",
                "decision_cached": "cached",
                "decision_recovered": "REC",
                "decision_recovered_cached": "REC-c"}[r["status"]]
        street = STREETS.get(r.get("street_idx"), "?")
        cards = "".join(r.get("hero_cards") or []) or "??"
        board = " ".join(r.get("board") or []) or "-"
        facing = "bet" if r.get("facing_bet") else "no-bet"
        floor = floor_map.get(r.get("captured_at") or "", "-")
        out.append(f"    {str(r.get('seq')):>5s} {mode:<7s} {street:<8s} "
                   f"{cards:<6s} {board:<16s} {facing:<7s} "
                   f"{fmt_action(r):<22s} {floor}")
    out.append("")

    # e. red flags --------------------------------------------------------
    flags: list[str] = []
    out.append("[e] RED FLAGS")

    safe_folds = [r for r in records if r.get("status") == "safe_fold"]
    for r in safe_folds:
        flags.append(f"safe_fold at seq={r.get('seq')}")
        out.append(f"    SAFE-FOLD  seq={r.get('seq')} "
                   f"captured={r.get('captured_at')}")
        out += frame_context_lines(r)

    for r in records:
        if r.get("anchor_refused"):
            flags.append(f"anchor_refused at seq={r.get('seq')}")
            out.append(f"    ANCHOR-REFUSED  seq={r.get('seq')} "
                       f"captured={r.get('captured_at')} — hand-start anchor "
                       f"rejected by ceiling guard; layer-1 recovery disabled "
                       f"for this hand")
            out += frame_context_lines(r)

    for r in records:
        deltas = r.get("invariant_deltas")
        if deltas and r.get("status") != "safe_fold":
            flags.append(f"invariant_deltas on non-safe_fold seq={r.get('seq')}")
            out.append(f"    INVARIANT-DELTAS on {r.get('status')} "
                       f"seq={r.get('seq')}: {deltas}")

    for r in records:
        if (r.get("status") in DECISION_STATUSES
                and (r.get("client_action") or {}).get("kind") == "fold"
                and r.get("facing_bet") is False):
            flags.append(f"FOLD-facing-0 at seq={r.get('seq')}")
            out.append(f"    !! FOLD WHILE FACING 0  seq={r.get('seq')} — "
                       f"should be IMPOSSIBLE after the check-when-free "
                       f"floor; investigate before any further dry run")
            out += frame_context_lines(r)

    for r in records:
        for sig in delta_signatures(r.get("invariant_deltas")):
            flags.append(f"reconstruction signature at seq={r.get('seq')}")
            out.append(f"    RECON-SIGNATURE  seq={r.get('seq')} "
                       f"status={r.get('status')}: {sig}")

    for w in warn:
        flags.append("header warning")

    if not flags:
        out.append("    none")
    out.append("")

    # f. summary line -----------------------------------------------------
    skip_pct = (100.0 * n_skip / n_total) if n_total else 0.0
    summary = (f"TRIAGE {session}: frames={n_total} "
               f"decisions={n_dec_all} (fresh {n_fresh}/cached {n_cached}/"
               f"recovered {n_rec + n_rec_cached}) "
               f"safe_folds={n_safe} skips={n_skip} ({skip_pct:.1f}%) "
               f"hands={len(hands)} hands_lost={len(lost_hands)} "
               f"to_act_skipped={len(lost_moments)} red_flags={len(flags)}")
    out += [rule, summary, rule]
    return "\n".join(out) + "\n", summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", type=Path, help="live_dryrun_*.jsonl")
    ap.add_argument("--out", type=Path, default=None,
                    help="report path (default logs/triage_<session>.txt)")
    ap.add_argument("--stdout-log", type=Path, default=None,
                    help="companion stdout capture for [FLOOR] lines "
                         "(default: <log>.stdout if it exists)")
    args = ap.parse_args()

    stdout_path = args.stdout_log
    if stdout_path is None:
        cand = args.log.with_suffix(".stdout")
        stdout_path = cand if cand.exists() else None

    report, summary = triage(args.log, stdout_path)

    out_path = args.out
    if out_path is None:
        session = re.sub(r"\.jsonl(\.preserved)?$", "", args.log.name)
        session = session.replace("live_dryrun_", "")
        out_path = Path("logs") / f"triage_{session}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)

    sys.stdout.write(report)
    sys.stdout.write(f"\nreport written: {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
