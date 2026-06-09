"""Stage 1 of the staged auto-clicker: DRY-RUN / LOG-ONLY.

Consumes scraper records — either replayed from a JSONL file or streamed
from the Windows scraper over a TCP socket — and runs the full real-ante
bridge → resolver → click-plan pipeline in SAMPLE MODE. Each frame is
displayed in a banner-formatted block (READ / DECIDE / CLICK PLAN) and
appended to a JSONL log. NEVER CLICKS.

Modes
─────
  --replay-from-jsonl PATH
      Replays the given corpus through the same pipeline as live. Useful
      for verifying the dashboard, sizing, click-target boxes, and
      safe-fold banner BEFORE wiring the Windows sender. `--replay-pace`
      adds an inter-record sleep so frames are readable.

  --listen-port N
      Binds a TCP server (default host 127.0.0.1 — only accepts SSH-
      tunneled / Tailscale connections). Reads newline-delimited JSON
      messages from the Windows scraper sender:
        {"type":"frame","seq":N,"record":<scraper_record>}
        {"type":"heartbeat","seq":null,"ts":"…"}
      Logs gaps in `seq` and stalls (>2s with no message). Reconnects
      tolerated; loop runs until Ctrl-C.

Common flags
────────────
  --checkpoint PATH   Model checkpoint (real-ante k=200 production).
  --abstraction PATH  Matching abstraction (236-dim).
  --structure PATH    Tournament structure YAML.
  --mode {sample,argmax}   Default sample (production deployment).
  --seed N            RNG seed for sample mode.
  --out PATH          JSONL log output (every processed record, including
                      skips and safe-folds — full audit trail).

THIS SCRIPT NEVER CLICKS. The click plan is computed and displayed only;
the auto-clicker advances to actually performing clicks in Stage 2+,
which is implemented in a separate runner.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.click_target import plan_as_dict
from src.nlhe.integration.live_loop import (
    DecisionCache, LiveDecision, make_decision,
)
from src.nlhe.integration.scraper_schema import SessionTracker


# ─── ANSI helpers ──────────────────────────────────────────────────────

ESC = "\033["
RESET = ESC + "0m"
BOLD = ESC + "1m"
DIM = ESC + "2m"
GREEN = ESC + "32m"
RED = ESC + "31m"
YELLOW = ESC + "33m"
CYAN = ESC + "36m"
GRAY = ESC + "90m"

# Disable colors if stdout isn't a TTY (clean log redirection)
if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    ESC = RESET = BOLD = DIM = GREEN = RED = YELLOW = CYAN = GRAY = ""


def _color(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


# ─── Banner renderer ───────────────────────────────────────────────────

def render_decision(d: LiveDecision) -> str:
    """Render one LiveDecision as a multi-line banner. Caller prints +
    flushes."""
    lines = []
    rule = "─" * 70
    lines.append(rule)

    # Header
    lvl = f"L{d.level}" if d.level else "L?"
    blinds_str = ""
    if d.blinds:
        sb, bb, ante = d.blinds
        blinds_str = f"(sb={sb} bb={bb} ante={ante})"
    seq_str = f"seq={d.seq}" if d.seq is not None else ""
    pot_corr = " [pot UI-lag-corrected]" if d.pot_corrected else ""
    lines.append(
        f"  {BOLD}LIVE  {lvl} {blinds_str}{RESET}   "
        f"captured {d.captured_at}{pot_corr}  {DIM}{seq_str}{RESET}"
    )
    hero_str = f"seat{d.hero_seat+1}" if d.hero_seat is not None else "?"
    dealer_str = f"seat{d.dealer_seat+1}" if d.dealer_seat is not None else "?"
    street_name = ["preflop", "flop", "turn", "river"][d.street_idx] \
        if d.street_idx is not None and 0 <= d.street_idx <= 3 else "?"
    lines.append(
        f"  hero={hero_str}  dealer={dealer_str}  "
        f"n_alive={d.n_alive}  street={street_name}"
    )
    lines.append(rule)

    # READ panel
    lines.append(f"  {BOLD}READ{RESET}")
    cards_str = "".join(d.hero_cards) if d.hero_cards else "(none)"
    board_str = " ".join(d.board) if d.board else "(preflop)"
    lines.append(f"    cards   {cards_str:<12s}  board   {board_str}")
    lines.append(
        f"    pot     {str(d.pot_total):<12s}  hero_stack {d.hero_stack}"
    )
    facing_str = "facing bet" if d.facing_bet else "no bet to call"
    if d.status == "decision":
        inv_str = _color("✓ PASS", GREEN)
    elif d.status == "decision_recovered":
        inv_str = _color("✓ PASS (field recovered)", YELLOW)
    elif d.status == "safe_fold":
        inv_str = _color("✗ SAFE-FOLD", RED)
    else:
        inv_str = _color(d.status, GRAY)
    lines.append(f"    {facing_str:<14s}    invariant: {inv_str}")
    if d.recovered_fields:
        lines.append(
            f"    {YELLOW}recovered: {', '.join(d.recovered_fields)}{RESET}"
        )
    if d.skip_reason:
        # Wrap reason if long
        reason = d.skip_reason
        if len(reason) > 60:
            reason = reason[:57] + "..."
        lines.append(f"    {DIM}reason: {reason}{RESET}")
    if d.invariant_deltas:
        # Show the first 3 deltas inline
        for delta in d.invariant_deltas[:3]:
            field_name, scraper_val, recon_val = delta
            lines.append(
                f"    {DIM}{field_name:<24s} scraper={scraper_val!r} "
                f"recon={recon_val!r}{RESET}"
            )
        if len(d.invariant_deltas) > 3:
            lines.append(
                f"    {DIM}... +{len(d.invariant_deltas) - 3} more deltas{RESET}"
            )

    # DECIDE panel — only for full decisions
    if d.status in ("decision", "decision_recovered") and d.client_action:
        lines.append(rule)
        lines.append(f"  {BOLD}DECIDE{RESET} (sample mode)")
        kind = d.client_action["kind"]
        chip = d.client_action.get("chip_amount")
        if kind == "raise_to":
            action_str = f"RAISE TO {chip}"
        elif kind == "call":
            action_str = f"CALL {chip}" if chip is not None else "CALL"
        elif kind == "check":
            action_str = "CHECK"
        elif kind == "fold":
            action_str = "FOLD"
        else:
            action_str = kind.upper()
        lines.append(f"    action     {_color(action_str, CYAN + BOLD)}")
        if chip is not None and d.client_action_bb_mult is not None:
            sizing = (
                f"{d.client_action_bb_mult:.1f}×BB │ "
                f"{(d.client_action_pot_frac or 0) * 100:.0f}%pot │ "
                f"{(d.client_action_stack_frac or 0) * 100:.0f}%stack"
            )
            lines.append(f"    sizing     {sizing}")
        lines.append(
            f"    openspiel  chip_int={d.resolver_raw_openspiel_chip_int}"
        )

    # CLICK PLAN panel
    lines.append(rule)
    cp = d.click_plan
    if cp is None or cp.is_no_op():
        if d.status == "safe_fold":
            label = "CLICK PLAN  (DRY-RUN — no plan, frame rejected)"
        elif d.status.startswith("skip"):
            label = "CLICK PLAN  (DRY-RUN — no plan, frame skipped)"
        else:
            label = "CLICK PLAN  (DRY-RUN — no plan)"
        lines.append(f"  {BOLD}{label}{RESET}")
        if cp and cp.reason:
            lines.append(f"    {DIM}{cp.reason}{RESET}")
        lines.append(f"    {DIM}would do nothing this frame{RESET}")
    else:
        lines.append(
            f"  {BOLD}CLICK PLAN  "
            f"({_color('DRY-RUN — would do, not clicking', YELLOW)}{BOLD}){RESET}"
        )
        for i, s in enumerate(cp.steps, start=1):
            if s.kind == "click":
                box_str = (
                    f"[{s.box[0]:.3f}, {s.box[1]:.3f}, "
                    f"{s.box[2]:.3f}, {s.box[3]:.3f}]"
                )
                lines.append(f"    {i}. click {s.target:<12s} box {box_str}")
            elif s.kind == "type":
                lines.append(f"    {i}. type  \"{s.payload}\"")
    lines.append(rule)
    return "\n".join(lines) + "\n"


def _abridged_history_line(d: LiveDecision) -> str:
    """One-line history entry for scroll back compactness."""
    ts = d.captured_at[-6:] if d.captured_at else "?"
    if d.status in ("decision", "decision_cached",
                     "decision_recovered", "decision_recovered_cached"):
        kind = d.client_action["kind"]
        chip = d.client_action.get("chip_amount")
        sizing = ""
        if chip and d.client_action_bb_mult:
            sizing = f" ({d.client_action_bb_mult:.1f}×BB)"
        action_str = (f"{kind.upper()} {chip}" if chip else kind.upper()) + sizing
        cached = d.status.endswith("_cached")
        tag = "↻" if cached else "✓"
        suffix = f" {DIM}(locked-in){RESET}" if cached else ""
        if d.recovered_fields:
            suffix += f" {YELLOW}(recovered){RESET}"
        return f"  {ts}  {_color(tag, GREEN)}  {action_str}{suffix}"
    if d.status == "safe_fold":
        return (f"  {ts}  {_color('✗', RED)}  SAFE-FOLD "
                f"{DIM}({(d.skip_reason or '')[:50]}){RESET}")
    return (f"  {ts}  {_color('·', GRAY)}  {d.status} "
             f"{DIM}({(d.skip_reason or '')[:50]}){RESET}")


# ─── Logging ───────────────────────────────────────────────────────────

def _decision_as_dict(d: LiveDecision) -> dict:
    """Serializable form for JSONL log.

    The caller adds `raw_record` (the original scraper dict) at the
    top level so the log is 1-for-1 replayable through `make_decision`
    in a later session — necessary to reproduce a live decision
    offline (the live consumer reads from a TCP socket; without the
    raw record on disk, only the decision summary is captured)."""
    base = asdict(d)
    if d.click_plan is not None:
        base["click_plan"] = plan_as_dict(d.click_plan)
    return base


# ─── Record sources ────────────────────────────────────────────────────

def replay_records(jsonl_path: Path, pace_seconds: float) -> Iterator[tuple[int, dict]]:
    """Yield (seq, record) tuples from a JSONL file with pacing."""
    with open(jsonl_path) as f:
        for seq, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield seq, json.loads(line)
            except json.JSONDecodeError:
                continue
            if pace_seconds > 0:
                time.sleep(pace_seconds)


def socket_records(bind_host: str, port: int) -> Iterator[tuple[int | None, dict]]:
    """Yield (seq, record) tuples from a TCP socket.

    Newline-delimited JSON; envelope is one of:
      {"type":"frame", "seq":N, "record":<scraper_record>}
      {"type":"heartbeat", "seq":null, "ts":"…"}

    Heartbeats are consumed silently (stall detection happens in the
    main loop's elapsed-time check). On disconnect, accepts a fresh
    client and continues. Bad lines are logged and skipped.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_host, port))
    server.listen(1)
    print(_color(
        f"[run_live_dryrun] listening on {bind_host}:{port} "
        f"(waiting for Windows scraper sender to connect…)",
        CYAN,
    ), flush=True)

    while True:
        try:
            client, peer = server.accept()
        except KeyboardInterrupt:
            print("\n[run_live_dryrun] shutdown requested; closing", flush=True)
            server.close()
            return
        print(_color(
            f"[run_live_dryrun] sender connected from {peer[0]}:{peer[1]}",
            GREEN,
        ), flush=True)

        buf = b""
        try:
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    print(_color(
                        "[run_live_dryrun] sender disconnected; "
                        "waiting for reconnect…",
                        YELLOW,
                    ), flush=True)
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(_color(
                            f"[run_live_dryrun] bad message dropped: {e}",
                            DIM,
                        ), flush=True)
                        continue
                    mtype = msg.get("type")
                    if mtype == "heartbeat":
                        # Surface the heartbeat as a sentinel record;
                        # main loop tracks last-seen-time externally.
                        yield None, {"__heartbeat__": True}
                        continue
                    if mtype == "frame":
                        rec = msg.get("record")
                        if not isinstance(rec, dict):
                            print(_color(
                                "[run_live_dryrun] frame msg missing 'record' field",
                                DIM,
                            ), flush=True)
                            continue
                        yield msg.get("seq"), rec
                        continue
                    print(_color(
                        f"[run_live_dryrun] unknown message type: {mtype!r}",
                        DIM,
                    ), flush=True)
        finally:
            try:
                client.close()
            except Exception:
                pass


# ─── Main loop ─────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Stage 1 of the staged auto-clicker — DRY-RUN.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay-from-jsonl", type=Path,
                      help="Replay a JSONL corpus through the pipeline.")
    src.add_argument("--listen-port", type=int,
                      help="Bind TCP socket on this port for live scraper.")
    ap.add_argument("--bind-host", default="127.0.0.1",
                     help="(socket mode) Interface to bind; default localhost "
                          "so only SSH-tunneled / Tailscale connects work.")
    ap.add_argument("--replay-pace", type=float, default=0.4,
                     help="(replay mode) Seconds between records. Default 0.4s.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--abstraction", required=True)
    ap.add_argument("--structure",
                     default="configs/ignition_double_up_6max_turbo.yaml")
    ap.add_argument("--mode", default="sample",
                     choices=["sample", "argmax"])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True,
                     help="JSONL log output (one row per processed record).")
    ap.add_argument("--stall-warn-seconds", type=float, default=2.0,
                     help="(socket mode) Warn if no message arrives within "
                          "this many seconds. Default 2.0s.")
    args = ap.parse_args()

    structure = TournamentStructure.from_yaml(args.structure)

    print(_color("[run_live_dryrun] loading model + abstraction…",
                  CYAN), flush=True)
    abstr = Abstraction.load(args.abstraction)
    from scripts.eval_6max_self_play import _load_solver
    solver = _load_solver(args.checkpoint, abstr, structure)

    tracker = SessionTracker()
    decision_cache = DecisionCache()
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(out_path, "a", buffering=1)  # line-buffered

    n_total = n_decision = n_decision_cached = n_safe_fold = n_skip = 0

    print(_color(
        f"[run_live_dryrun] {BOLD}DRY-RUN ACTIVE{RESET}{CYAN}: bot will display "
        f"and log decisions but {BOLD}NEVER CLICK{RESET}{CYAN}. Mode={args.mode}.",
        CYAN,
    ), flush=True)
    print(_color(f"[run_live_dryrun] log: {out_path}", DIM), flush=True)

    # Pick source
    if args.replay_from_jsonl:
        source = replay_records(args.replay_from_jsonl, args.replay_pace)
    else:
        source = socket_records(args.bind_host, args.listen_port)

    last_msg_time = time.time()

    try:
        for seq, rec in source:
            # Heartbeat sentinel — just refresh the stall timer.
            if rec.get("__heartbeat__"):
                last_msg_time = time.time()
                continue

            # Stall check (socket mode only; replay mode's pacing is
            # deterministic so this never fires there).
            now = time.time()
            if args.listen_port and (now - last_msg_time) > args.stall_warn_seconds:
                print(_color(
                    f"[run_live_dryrun] ⚠ stall: {now - last_msg_time:.1f}s "
                    f"since last frame/heartbeat",
                    YELLOW,
                ), flush=True)
            last_msg_time = now

            n_total += 1
            d = make_decision(rec, structure, solver, tracker, rng,
                               mode=args.mode, seq=seq,
                               decision_cache=decision_cache)
            if d.status in ("decision", "decision_recovered"):
                n_decision += 1
            elif d.status in ("decision_cached", "decision_recovered_cached"):
                n_decision_cached += 1
            elif d.status == "safe_fold":
                n_safe_fold += 1
            else:
                n_skip += 1

            # Show fresh DECISIONS and SAFE_FOLDS prominently with the full
            # banner; cached decisions and skips fold into the one-line
            # history (the action was already shown the first time it was
            # decided — re-showing the banner per polling frame buries it).
            if d.status in ("decision", "decision_recovered", "safe_fold"):
                print(render_decision(d), flush=True)
            else:
                print(_abridged_history_line(d), flush=True)

            log_entry = _decision_as_dict(d)
            log_entry["raw_record"] = rec
            log_fh.write(json.dumps(log_entry, default=str) + "\n")
    except KeyboardInterrupt:
        print("\n[run_live_dryrun] shutdown requested", flush=True)
    finally:
        log_fh.close()

    print(_color(
        f"\n[run_live_dryrun] done. total={n_total} decision={n_decision} "
        f"decision_cached={n_decision_cached} "
        f"safe_fold={n_safe_fold} skip={n_skip}",
        CYAN,
    ), flush=True)
    print(_color("[run_live_dryrun] STILL NO CLICKING. Log-only.", CYAN),
           flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
