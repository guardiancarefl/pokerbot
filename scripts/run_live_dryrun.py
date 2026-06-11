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
                      argmax additionally requires --unsafe-argmax: sample
                      is the deployment invariant (DECISIONS.md "Process
                      learning: iter_500 throwaway probes" — the argmax
                      deployment flag was the one real deployment bug).
  --seed N            RNG seed for sample mode.
  --out PATH          JSONL log output (every processed record, including
                      skips and safe-folds — full audit trail).

THIS SCRIPT NEVER CLICKS. The click plan is computed and displayed only;
the auto-clicker advances to actually performing clicks in Stage 2+,
which is implemented in a separate runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
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


def render_fallback(plan, click_plan=None) -> str:
    """Loud banner for a fired guaranteed-action fallback. Visually
    distinct from the DECIDE panel on purpose: a fallback is NOT a
    model decision."""
    lines = []
    rule = "█" * 70
    lines.append(_color(rule, RED))
    lines.append(_color(
        f"  ⚠ FALLBACK — NOT A MODEL DECISION ⚠   "
        f"action={plan.action_kind.upper()}", RED + BOLD))
    lines.append(
        f"  hand={''.join(plan.hand_cards) or '?'}  "
        f"armed at seq={plan.armed_seq} ({plan.armed_captured_at})  "
        f"waited {plan.waited_seconds:.1f}s")
    fired_via = (f"frame seq={plan.fired_after_seq}"
                 if plan.fired_after_seq is not None
                 else "frameless poll (no frame at deadline)")
    lines.append(f"  fired via {fired_via}  "
                 f"check_available={plan.check_available}  "
                 f"session fallback #{plan.n_fallbacks_session}")
    if click_plan is not None and not click_plan.is_no_op():
        for i, s in enumerate(click_plan.steps, start=1):
            if s.kind == "click":
                lines.append(f"    {i}. click {s.target}")
            elif s.kind == "type":
                lines.append(f"    {i}. type \"{s.payload}\"")
    if plan.abort_recommended:
        lines.append(_color(
            f"  ⛔ SESSION ABORT RECOMMENDED — {plan.abort_reason} — "
            f"SIT OUT NOW", RED + BOLD))
    lines.append(_color(rule, RED))
    return "\n".join(lines) + "\n"


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


def socket_records(bind_host: str, port: int,
                   tick_seconds: float | None = None,
                   ) -> Iterator[tuple[int | None, dict]]:
    """Yield (seq, record) tuples from a TCP socket.

    Newline-delimited JSON; envelope is one of:
      {"type":"frame", "seq":N, "record":<scraper_record>}
      {"type":"heartbeat", "seq":null, "ts":"…"}

    Heartbeats are consumed silently (stall detection happens in the
    main loop's elapsed-time check). On disconnect, accepts a fresh
    client and continues. Bad lines are logged and skipped.

    tick_seconds (fallback watchdog only): when set, the client socket
    gets a recv timeout and a {"__tick__": True} sentinel is yielded on
    each timeout so the main loop can fire the guaranteed-action
    fallback framelessly — a frozen scraper must not suppress the
    deadline. When None (the default, and always when the watchdog is
    off) the recv blocks exactly as before this parameter existed.
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
        if tick_seconds is not None:
            client.settimeout(tick_seconds)
        try:
            while True:
                try:
                    chunk = client.recv(65536)
                except socket.timeout:
                    yield None, {"__tick__": True}
                    continue
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


# ─── Session identity stamping ─────────────────────────────────────────

def _sha256_of_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_identity() -> tuple[str, bool]:
    """(HEAD sha, dirty?) of the repo this script runs from. dirty counts
    TRACKED modifications only — this working tree permanently carries
    untracked artifact dirs, which would make the flag uninformative."""
    repo = Path(__file__).resolve().parent.parent
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True).strip()
        porcelain = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain",
             "--untracked-files=no"], text=True)
        return head, bool(porcelain.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", False


def build_session_header(args: argparse.Namespace,
                          short_stack_bb: float) -> dict:
    """Run-identity header — the first record of every session log.

    sha256s are computed here from the file bytes at the same paths the
    loaders read, immediately before loading; never from recorded
    constants."""
    head, dirty = _git_identity()
    return {
        "record_type": "session_header",
        "mode": args.mode,
        "seed": int(args.seed),
        "ckpt_path": str(Path(args.checkpoint).resolve()),
        "ckpt_sha256": _sha256_of_file(args.checkpoint),
        "abstraction_sha256": _sha256_of_file(args.abstraction),
        "floors": {
            "aa_kk": True,
            "check_when_free": True,
            "short_stack_bb": float(short_stack_bb),
        },
        "git_head": head,
        "git_dirty": dirty,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }


# ─── Main loop ─────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    ap.add_argument("--unsafe-argmax", action="store_true",
                     help="Required alongside --mode argmax. Sample mode is "
                          "the deployment invariant; argmax was the old "
                          "deployment bug (DECISIONS.md iter_500 entry).")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True,
                     help="JSONL log output (one row per processed record).")
    ap.add_argument("--stall-warn-seconds", type=float, default=2.0,
                     help="(socket mode) Warn if no message arrives within "
                          "this many seconds. Default 2.0s.")
    ap.add_argument("--anchor-sum-floor", action="store_true",
                     help="P1 anchor sum-floor guard (approved 2026-06-11): "
                          "refuse hand-start anchors whose pre-hand sum != "
                          "chips-in-play while all seats read alive (the "
                          "seq-1363 poisoned-anchor class). OFF by default.")
    ap.add_argument("--fallback-seconds", type=float, default=None,
                     help="Guaranteed-action fallback (operator-approved "
                          "2026-06-11): if hero is to-act and no decision "
                          "exists within N seconds, emit CHECK-if-free / "
                          "FOLD-otherwise, logged loudly as FALLBACK (never "
                          "a model decision). OFF by default in Stage 1. "
                          "Proposed live value: 7.0.")
    args = ap.parse_args(argv)

    if args.mode == "argmax" and not args.unsafe_argmax:
        ap.error(
            "--mode argmax requires --unsafe-argmax. Sample mode is the "
            "deployment invariant: DECISIONS.md 'Process learning: iter_500 "
            "throwaway probes' records the argmax deployment flag as the one "
            "real deployment bug of the 8-cycle arc (fixed argmax → sample); "
            "argmax caricatures the mixed strategy and is for audits only.")

    return args


def main() -> int:
    args = parse_args()

    structure = TournamentStructure.from_yaml(args.structure)

    # Run-identity header: hash the exact files about to be loaded, then load.
    from src.nlhe.integration.live_loop import _DEFAULT_SHORT_STACK_FLOOR_BB
    header = build_session_header(
        args, short_stack_bb=_DEFAULT_SHORT_STACK_FLOOR_BB)

    print(_color("[run_live_dryrun] loading model + abstraction…",
                  CYAN), flush=True)
    abstr = Abstraction.load(args.abstraction)
    from scripts.eval_6max_self_play import _load_solver
    solver = _load_solver(args.checkpoint, abstr, structure)
    # Convention gate (C3): the clean bridge view serves real-ante models
    # only; refuse anything unstamped/inflated. Reads the checkpoint's RAW
    # config_dict (or the sha whitelist) — per-checkpoint dispatch, never a
    # global flip. See src/nlhe/conventions.py.
    from src.nlhe.conventions import require_live_servable
    conv = require_live_servable(args.checkpoint)
    print(_color(f"[run_live_dryrun] ante convention: {conv} (servable)",
                  CYAN), flush=True)

    tracker = SessionTracker(anchor_sum_floor=args.anchor_sum_floor)
    if args.anchor_sum_floor:
        header["anchor_sum_floor"] = True
        print(_color("[run_live_dryrun] P1 anchor sum-floor guard ARMED",
                      YELLOW), flush=True)
    decision_cache = DecisionCache()
    rng = random.Random(args.seed)

    # Guaranteed-action fallback watchdog — OFF unless --fallback-seconds.
    # Strictly listener-side: never touches make_decision, DecisionCache,
    # or SessionTracker (see src/nlhe/integration/fallback.py).
    watchdog = None
    if args.fallback_seconds is not None:
        from src.nlhe.integration.fallback import FallbackWatchdog
        watchdog = FallbackWatchdog(args.fallback_seconds)
        header["fallback_seconds"] = float(args.fallback_seconds)
        print(_color(
            f"[run_live_dryrun] guaranteed-action fallback ARMED: "
            f"N={args.fallback_seconds:.1f}s, CHECK-if-free/FOLD, abort at "
            f"{watchdog.abort_fallbacks} fallbacks/{watchdog.abort_window_hands} "
            f"hands or {watchdog.abort_consecutive} consecutive hands",
            YELLOW,
        ), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(out_path, "a", buffering=1)  # line-buffered

    header_line = json.dumps(header)
    log_fh.write(header_line + "\n")
    print(header_line, flush=True)

    n_total = n_decision = n_decision_cached = n_safe_fold = n_skip = 0

    print(_color(
        f"[run_live_dryrun] {BOLD}DRY-RUN ACTIVE{RESET}{CYAN}: bot will display "
        f"and log decisions but {BOLD}NEVER CLICK{RESET}{CYAN}. Mode={args.mode}.",
        CYAN,
    ), flush=True)
    print(_color(f"[run_live_dryrun] log: {out_path}", DIM), flush=True)

    # Pick source. The socket gets a recv-timeout tick ONLY when the
    # fallback watchdog is armed (when off, recv blocks exactly as
    # before — live path unchanged).
    if args.replay_from_jsonl:
        source = replay_records(args.replay_from_jsonl, args.replay_pace)
    else:
        source = socket_records(
            args.bind_host, args.listen_port,
            tick_seconds=1.0 if watchdog is not None else None)

    last_msg_time = time.time()
    n_fallback = 0

    def emit_fallback(plan) -> None:
        nonlocal n_fallback
        n_fallback += 1
        from src.nlhe.integration.click_target import compute_click_target
        client_action = {"kind": plan.action_kind, "chip_amount": None,
                         "raw_openspiel_chip_int": None}
        try:
            cp = compute_click_target(client_action, plan.last_controls)
        except Exception:
            cp = None
        print(render_fallback(plan, cp), flush=True)
        row = plan.as_log_record()
        row["fired_at_utc"] = datetime.now(timezone.utc).isoformat()
        if cp is not None:
            row["click_plan"] = plan_as_dict(cp)
        log_fh.write(json.dumps(row, default=str) + "\n")

    try:
        for seq, rec in source:
            # Tick sentinel (watchdog mode only) — frameless deadline check.
            if rec.get("__tick__"):
                if watchdog is not None:
                    plan = watchdog.poll(time.time())
                    if plan is not None:
                        emit_fallback(plan)
                continue
            # Heartbeat sentinel — refresh the stall timer (+ frameless
            # fallback deadline check while the table is quiet).
            if rec.get("__heartbeat__"):
                last_msg_time = time.time()
                if watchdog is not None:
                    plan = watchdog.poll(last_msg_time)
                    if plan is not None:
                        emit_fallback(plan)
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

            if watchdog is not None:
                plan = watchdog.observe(rec, d.status, seq, time.time())
                if plan is not None:
                    emit_fallback(plan)
    except KeyboardInterrupt:
        print("\n[run_live_dryrun] shutdown requested", flush=True)
    finally:
        log_fh.close()

    fallback_summary = ""
    if watchdog is not None:
        fallback_summary = (f" fallback={n_fallback}"
                            + (" ABORT-RECOMMENDED"
                               if watchdog.abort_recommended else ""))
    print(_color(
        f"\n[run_live_dryrun] done. total={n_total} decision={n_decision} "
        f"decision_cached={n_decision_cached} "
        f"safe_fold={n_safe_fold} skip={n_skip}{fallback_summary}",
        CYAN,
    ), flush=True)
    print(_color("[run_live_dryrun] STILL NO CLICKING. Log-only.", CYAN),
           flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
