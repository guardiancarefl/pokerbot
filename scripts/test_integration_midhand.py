"""Phase 2 gate — hero-to-act corpus test.

Runs replay_to_decision + check_mid_hand_invariant on every
controls.present==True frame in a liveN.jsonl. Reports pass rate. This
mirrors test_integration_handstart.py but for the mid-hand pipeline.

Phase 2 gate: invariant pass rate >= 99% on hero-to-act frames before any
clicking is wired. <99% -> diagnose from per-field deltas before adding
pipeline complexity (same discipline as Phase 1).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.nlhe.game_strings import TournamentStructure  # noqa: E402
from src.nlhe.integration.scraper_schema import (  # noqa: E402
    parse_frame, ScraperSuspect, ScraperParseError, ScraperDataQuality,
    NUM_SEATS,
)
from src.nlhe.integration.replay import (  # noqa: E402
    replay_to_decision, ReplayError,
)
from src.nlhe.integration.invariant import (  # noqa: E402
    check_mid_hand_invariant,
)

STRUCT_YAML = "configs/ignition_double_up_6max_turbo.yaml"


def run_one(record: dict, structure: TournamentStructure,
            hero_seat_alias: str = "seat1",
            verbose: bool = False
            ) -> tuple[str, dict]:
    """Run parse -> hero-to-act detect -> replay_to_decision ->
    check_mid_hand_invariant on one record. Returns (status, info).
    Statuses:
      parse_error, suspect, data_quality, not_hero_to_act,
      replay_error, invariant_fail, pass
    """
    info: dict = {"captured_at": record.get("captured_at", "")}
    try:
        frame = parse_frame(record, hero_seat_alias=hero_seat_alias)
    except ScraperSuspect as e:
        info["reason"] = str(e)
        return "suspect", info
    except ScraperDataQuality as e:
        info["reason"] = str(e)
        return "data_quality", info
    except ScraperParseError as e:
        info["reason"] = str(e)
        return "parse_error", info

    # Phase 2 corpus filter: only frames where it's actually hero's turn
    # to act. Hand-start frames have controls.present=False; they're
    # covered by Phase 1's test harness.
    if not frame.controls_present:
        info["reason"] = "controls.present is False (not a hero-to-act frame)"
        return "not_hero_to_act", info

    info["n_alive"] = sum(frame.alive)
    info["dealer_seat"] = frame.dealer_seat
    info["hero_seat"] = frame.hero_seat
    info["hero_cards"] = list(frame.hero_cards)
    info["board"] = list(frame.board)
    info["pot_total"] = frame.pot_total

    # Data-quality filter: if hero is alive and we're nominally hero-to-act
    # but the scraper captured no hero_cards, the screenshot caught a
    # between-hands or mid-deal moment. A bot wouldn't act without knowing
    # its own cards anyway — treat as data_quality SKIP (same bucket as
    # suspect frames), not a replay_error. Verified on live9.jsonl
    # 2026-06-04: all 13 such frames are between-hands snapshots (controls
    # field was raw `None` in JSON; pot is 0 or partial-blind-posting on
    # an empty board).
    if frame.alive[frame.hero_seat] and not frame.hero_cards:
        info["reason"] = ("hero alive + controls_present but hero_cards "
                          "empty — scraper between-hands snapshot")
        return "data_quality", info

    # Data-quality filter: pot_total <= 0 with a non-empty board / alive
    # seats. The scraper either failed to extract pot (raw JSON `pot: None`
    # or `pot: {main: ...}` missing `total`), or it captured a between-
    # hands frame. A real hand-in-progress always has pot >= antes+blinds.
    # Verified on live9.jsonl 2026-06-04: the pot-missing frames are
    # screenshots taken at the exact moment the previous hand's pot is
    # being collected (frame.pot_total=0 with hero_cards already shown
    # because the hero-card field lags the pot-clear by a frame).
    if frame.pot_total <= 0:
        info["reason"] = ("pot_total<=0 — scraper failed to extract pot "
                          "(raw `pot: None` or between-hands)")
        return "data_quality", info

    try:
        pack = replay_to_decision(frame, structure)
    except ReplayError as e:
        info["reason"] = str(e).split('\n')[0][:200]
        return "replay_error", info

    info["sb_seat"] = pack.sb_seat
    info["bb_seat"] = pack.bb_seat
    info["street_idx"] = pack.street_idx
    info["preflop_commit_per_alive"] = pack.preflop_commit_per_alive
    info["n_actions_applied"] = pack.n_actions_applied

    res = check_mid_hand_invariant(frame, pack)
    info["deltas"] = res.deltas
    info["safe_action"] = res.safe_action

    if res.ok:
        return "pass", info
    if verbose:
        print(f"\n[mid-hand invariant FAIL] {info['captured_at']}:")
        for d in res.deltas:
            print(f"  {d[0]:<28s}  scraper={d[1]!r:<24s}  recon={d[2]!r}")
    return "invariant_fail", info


def run_jsonl(path: Path, structure: TournamentStructure,
              hero_seat_alias: str = "seat1") -> int:
    print(f"=== Mid-hand (hero-to-act) corpus: {path} ===\n")
    counts = Counter()
    hero_to_act_counts = Counter()
    n_total = 0
    n_hero = 0
    n_pass = 0
    failures = []

    t0 = time.time()
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["json_decode_error"] += 1
                continue
            status, info = run_one(record, structure,
                                    hero_seat_alias=hero_seat_alias,
                                    verbose=False)
            counts[status] += 1
            # Hero-to-act denominator excludes drops/non-hero-to-act
            if status in ("replay_error", "invariant_fail", "pass"):
                n_hero += 1
                hero_to_act_counts[status] += 1
                if status == "pass":
                    n_pass += 1
                elif status == "invariant_fail" and len(failures) < 100:
                    failures.append((line_no, info))
                elif status == "replay_error" and len(failures) < 100:
                    failures.append((line_no, info))

    wall = time.time() - t0
    print(f"Processed {n_total} records in {wall:.1f}s\n")
    print("=== Per-status counts (all records) ===")
    for k in ("parse_error", "suspect", "data_quality", "not_hero_to_act",
              "replay_error", "invariant_fail", "pass"):
        print(f"  {k:<18s}  {counts.get(k, 0):>6d}")
    print()
    if n_hero:
        pass_rate = 100.0 * n_pass / n_hero
        print(f"=== Hero-to-act records (Phase 2 gate metric) ===")
        print(f"  total hero-to-act processed:  {n_hero}")
        print(f"  replay_error:                 {hero_to_act_counts.get('replay_error', 0)}")
        print(f"  invariant_fail:               {hero_to_act_counts.get('invariant_fail', 0)}")
        print(f"  PASS:                         {hero_to_act_counts.get('pass', 0)}")
        print(f"  --> MID-HAND INVARIANT PASS RATE = {pass_rate:.2f}%  "
              f"(gate = 99.0%)")
        if failures:
            print(f"\n=== First {len(failures)} failures ===")
            for line_no, info in failures:
                print(f"\nline {line_no}  captured_at={info.get('captured_at')}")
                print(f"  n_alive={info.get('n_alive')}  "
                      f"dealer=seat{info.get('dealer_seat', -1)+1}  "
                      f"hero=seat{info.get('hero_seat', -1)+1}  "
                      f"street_idx={info.get('street_idx')}  "
                      f"pot={info.get('pot_total')}")
                if "reason" in info:
                    print(f"  reason: {info['reason'][:160]}")
                for d in info.get("deltas", [])[:8]:
                    print(f"    {d[0]:<28s}  scraper={d[1]!r:<24s}  recon={d[2]!r}")
    else:
        print("=== No hero-to-act records found in corpus ===")
    return 0 if (n_hero > 0 and n_pass == n_hero) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True,
                    help="path to liveN.jsonl capture")
    ap.add_argument("--hero-seat", default="seat1")
    args = ap.parse_args()

    structure = TournamentStructure.from_yaml(STRUCT_YAML)
    print(f"Loaded structure: {structure.format_name}  "
          f"starting_chips={structure.starting_chips}\n")

    return run_jsonl(Path(args.jsonl), structure,
                      hero_seat_alias=args.hero_seat)


if __name__ == "__main__":
    sys.exit(main())
