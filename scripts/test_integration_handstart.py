"""Phase-1 end-to-end test: parse -> replay -> invariant on real captures.

Two modes:
  --sample      run on the single embedded sample record (parser-binding only;
                expected: parse OK, replay+invariant will FAIL chip totals
                because the sample is from a 2000-chip variant while
                deployment is locked to 1500. The test reports the diff and
                exits OK if the diff is exactly the expected ante-mismatch
                pattern — i.e., we're proving the diff catches it.)
  --jsonl PATH  run on every hand-start record in a liveN.jsonl capture.
                Reports invariant pass-rate.

Strict invariant: ANY field mismatch -> REJECT. Reports per-field deltas
on every reject so we can diagnose at the field level (not just "fail").
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
    parse_frame, is_hand_start, chip_conservation_total, ScraperSuspect,
    ScraperParseError, NUM_SEATS,
)
from src.nlhe.integration.replay import replay_hand_start, ReplayError  # noqa: E402
from src.nlhe.integration.invariant import check_hand_start_invariant  # noqa: E402

STRUCT_YAML = "configs/ignition_double_up_6max_turbo.yaml"

# The literal sample record provided in the design exchange.
SAMPLE_RECORD = {
  "blinds": "% 15/25, 5 Ante No Limit Hold'em - Hold'em - TBL#1",
  "hero_cards": ["6h", "7c"],
  "board": [],
  "pot": {"total": 70, "main": 30},
  "stacks": {"seat1": 1960, "seat2": 1985, "seat3": 1935,
              "seat4": 1970, "seat5": 2030, "seat6": 2050},
  "folded": {f"seat{i+1}": False for i in range(6)},
  "empty":  {f"seat{i+1}": False for i in range(6)},
  "bets":   {"seat1": 25, "seat2": None, "seat3": None,
              "seat4": None, "seat5": None, "seat6": 15},
  "dealer": "seat5",
  "action": {"buttons": [], "amounts": [], "raw": ""},
  "controls": {"present": False, "bet_input": {}, "slider": {}, "shortcuts": []},
  "suspect": False,
  "captured_at": "20260604_115713_574",
}


def run_one(record: dict, structure: TournamentStructure,
            hero_seat_alias: str = "seat1", verbose: bool = True
            ) -> tuple[str, dict]:
    """Run parse -> handstart-detect -> replay -> invariant on one record.
    Returns (status, info_dict) where status is one of:
      'parse_error', 'suspect', 'not_handstart', 'replay_error',
      'invariant_fail', 'pass'
    """
    info: dict = {"captured_at": record.get("captured_at", "")}
    try:
        frame = parse_frame(record, hero_seat_alias=hero_seat_alias)
    except ScraperSuspect as e:
        info["reason"] = str(e)
        return "suspect", info
    except ScraperParseError as e:
        info["reason"] = str(e)
        return "parse_error", info

    info["chip_conservation_total"] = chip_conservation_total(frame)

    if not is_hand_start(frame):
        info["reason"] = "not a hand-start frame (board non-empty or non-blind bets)"
        return "not_handstart", info

    try:
        state_pack = replay_hand_start(frame, structure)
    except ReplayError as e:
        info["reason"] = str(e)
        return "replay_error", info

    info["sb_seat"] = state_pack.sb_seat
    info["bb_seat"] = state_pack.bb_seat
    info["n_alive"] = state_pack.n_alive
    info["level"] = state_pack.blind_level.level

    res = check_hand_start_invariant(frame, state_pack)
    info["reconstructed"] = res.reconstructed
    info["deltas"] = res.deltas
    info["safe_action"] = res.safe_action

    if res.ok:
        return "pass", info
    if verbose:
        print(f"\n[invariant FAIL] {record.get('captured_at', '')}:")
        print(res.format_deltas())
    return "invariant_fail", info


def run_sample(structure: TournamentStructure) -> int:
    print("=== Running on the embedded sample record ===\n")
    print(f"Record captured_at: {SAMPLE_RECORD['captured_at']}")
    print(f"Note: sample is from a 2000-chip-start tournament variant.")
    print(f"Deployment is locked to STARTING_CHIPS=1500. We EXPECT this sample's")
    print(f"chip arithmetic to be internally consistent (total 12000 = 6x2000)")
    print(f"but the replay state will match the scraper view exactly, since the")
    print(f"replay reads pre_hand_stacks from the frame itself, not from a")
    print(f"hardcoded starting_chips. So we expect invariant PASS on this record.")
    print()
    status, info = run_one(SAMPLE_RECORD, structure, verbose=True)
    print(f"\n>>> status: {status}")
    print(f">>> chip_conservation_total: {info.get('chip_conservation_total')} "
          f"(== 12000 means sample is from a 2000-chip variant; 9000 = 1500-chip)")
    if status == "pass":
        print(f">>> sb_seat={info['sb_seat']}  bb_seat={info['bb_seat']}  "
              f"n_alive={info['n_alive']}  level={info['level']}")
        print(f">>> reconstructed scraper-view fields:")
        recon = info["reconstructed"]
        print(f"    pot: {recon['pot']}")
        print(f"    stack: {recon['stack']}")
        print(f"    bet:   {recon['bet']}")
        print(f"\nEND-TO-END PIPELINE WORKS: parse -> handstart -> replay -> invariant PASS")
        return 0
    print(f">>> FAILURE — pipeline did not complete cleanly.")
    return 1


def run_jsonl(path: Path, structure: TournamentStructure,
              hero_seat_alias: str = "seat1") -> int:
    print(f"=== Running on jsonl corpus: {path} ===\n")
    counts = Counter()
    handstart_counts = Counter()
    n_total = 0
    n_handstart = 0
    n_invariant_pass = 0
    sample_failures = []

    t0 = time.time()
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                counts["json_decode_error"] += 1
                continue
            status, info = run_one(record, structure,
                                    hero_seat_alias=hero_seat_alias,
                                    verbose=False)
            counts[status] += 1
            if status not in ("not_handstart", "suspect", "parse_error"):
                n_handstart += 1
                handstart_counts[status] += 1
                if status == "pass":
                    n_invariant_pass += 1
                elif status == "invariant_fail":
                    if len(sample_failures) < 5:
                        sample_failures.append((line_no, info))

    wall = time.time() - t0
    print(f"Processed {n_total} records in {wall:.1f}s\n")
    print("=== Per-status counts (all records) ===")
    for k in ("parse_error", "suspect", "not_handstart", "replay_error",
              "invariant_fail", "pass"):
        print(f"  {k:<18s}  {counts.get(k, 0):>6d}")
    print()
    if n_handstart:
        pass_rate = 100.0 * n_invariant_pass / n_handstart
        print(f"=== Hand-start records (suspect/parse_error/not_handstart excluded) ===")
        print(f"  total hand-starts processed:  {n_handstart}")
        print(f"  replay_error:                 {handstart_counts.get('replay_error', 0)}")
        print(f"  invariant_fail:               {handstart_counts.get('invariant_fail', 0)}")
        print(f"  PASS:                         {handstart_counts.get('pass', 0)}")
        print(f"  --> INVARIANT PASS RATE = {pass_rate:.2f}%  "
              f"(gate = 99.0% before any clicking)")
        if sample_failures:
            print(f"\n=== Sample invariant failures (first 5) ===")
            for line_no, info in sample_failures:
                print(f"\nline {line_no}  captured_at={info.get('captured_at')}")
                for fld, scr, rec in info["deltas"][:10]:
                    print(f"  {fld:<24s}  scraper={scr!r:<20s}  recon={rec!r}")
    return 0 if (n_handstart > 0 and n_invariant_pass == n_handstart) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                    help="run on the embedded sample record only")
    ap.add_argument("--jsonl", default=None,
                    help="path to a liveN.jsonl capture; runs on every record")
    ap.add_argument("--hero-seat", default="seat1",
                    help="scraper alias for the hero seat (default seat1)")
    args = ap.parse_args()

    structure = TournamentStructure.from_yaml(STRUCT_YAML)
    print(f"Loaded structure: {structure.format_name}  "
          f"starting_chips={structure.starting_chips}  "
          f"n_levels={len(structure.blind_schedule)}\n")

    if args.sample:
        return run_sample(structure)
    if args.jsonl:
        return run_jsonl(Path(args.jsonl), structure,
                          hero_seat_alias=args.hero_seat)
    # Default: run sample
    return run_sample(structure)


if __name__ == "__main__":
    sys.exit(main())
