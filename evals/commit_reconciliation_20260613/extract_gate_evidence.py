"""Extract G2/G3/G4 gate evidence from the armed arm outputs.

Run after g2_armA_*/g2_armB_* sweeps complete:
    python evals/commit_reconciliation_20260613/extract_gate_evidence.py
Writes G2_pins.txt, G3_seq276.txt, G4_displacement_family.txt into this
directory and prints a PASS/FAIL line per gate.
"""
from __future__ import annotations

import json
from pathlib import Path

D = Path(__file__).resolve().parent

PINS = {
    "20260612_193706_536": dict(
        seq=780, hand="8c8d",
        # registered believed state (ground truth seq 781-786)
        pot_total=1017, hero_stack=2408, facing_bet=True,
        recovered_sub=["commit.seat4=50", "commit.seat5=100"]),
    "20260612_194941_036": dict(
        seq=1086, hand="AcAh",
        pot_total=2693, hero_stack=637, facing_bet=True,
        recovered_sub=["commit.seat2=25", "commit.seat3=25"]),
    "20260612_200631_846": dict(
        seq=1635, hand="5hTs",
        pot_total=509, hero_stack=3617, facing_bet=False,
        recovered_sub=["commit.seat4=75"]),
}

# Verified bonus recovery (NOT a registered pin; ground-truth verdict in
# SUMMARY.txt): 183701 seq 697 KhQc — the seq-1635 ante-bookkeeping
# sub-class on TWO callers (seat3/seat5 each mis-rendered by the 15 ante
# after seat4's 89-chip all-in-for-less poisoned the matched_all_in
# heuristic). Implied commits equal every visible bet exactly against the
# clean seq-693 anchor (1423,2495,1669,104,3309,0); zero restorations;
# the recon's bet=185 (= 200-15) is not a legal call amount in the hand.
VERIFIED_BONUS = {
    "20260612_151624_215": dict(
        log="live_dryrun_20260612_183701", seq=697,
        pot_total=664, hero_stack=1308, facing_bet=True),
}

# The P2 build's 9 gate-2 displacement-family frames (G4), from
# evals/p2_build_20260612/flag_on_armA_vs_armB.txt
G4_FRAMES = [
    ("live_dryrun_20260611_163815", "20260611_130121_831", 429),
    ("live_dryrun_20260611_201230", "20260611_161937_746", 13),
    ("live_dryrun_20260611_201230", "20260611_161939_102", 14),
    ("live_dryrun_20260611_222532", "20260611_183517_645", 1891),
    ("live_dryrun_20260611_222532", "20260611_184658_780", 2097),
    ("live_dryrun_20260611_222532", "20260611_204408_252", 2679),
    ("live_dryrun_20260611_222532", "20260611_204412_697", 2681),
    ("live_dryrun_20260612_035454", "20260612_000213_894", 153),
    ("live_dryrun_20260612_161347", "20260612_122700_038", 276),
]

CASUALTY_LOG = "live_dryrun_20260612_230149"
SEQ276_LOG = "live_dryrun_20260612_161347"


def load(arm: str, log: str) -> dict:
    out = {}
    for line in open(D / arm / f"{log}.jsonl"):
        r = json.loads(line)
        out[r.get("captured_at") or f"seq:{r['seq']}"] = r
    return out


def main() -> None:
    ok_all = True

    # ── G2: the three pins recover in armB, refused in armA ─────────
    lines = ["G2 — pinned-casualty recoveries (armB = P1+P2+CR)",
             "=" * 64]
    for mode in ("sample", "argmax"):
        a = load(f"g2_armA_{mode}", CASUALTY_LOG)
        b = load(f"g2_armB_{mode}", CASUALTY_LOG)
        lines.append(f"\n--- mode={mode} ---")
        for cap, exp in PINS.items():
            ra, rb = a[cap], b[cap]
            ok = (
                ra["status"] == "safe_fold"
                and rb["status"] == "decision_recovered"
                and rb["pot_total"] == exp["pot_total"]
                and rb["hero_stack"] == exp["hero_stack"]
                and rb["facing_bet"] == exp["facing_bet"]
                and all(any(sub in f for f in (rb["recovered_fields"] or []))
                        for sub in exp["recovered_sub"])
            )
            ok_all &= ok
            lines.append(
                f"seq {exp['seq']:>4} {exp['hand']}  "
                f"armA={ra['status']}  armB={rb['status']}  "
                f"pot={rb['pot_total']} hero={rb['hero_stack']} "
                f"facing={rb['facing_bet']}  "
                f"action={json.dumps(rb['client_action'])}  "
                f"[{'OK' if ok else 'FAIL'}]")
            lines.append(f"      recovered_fields={rb['recovered_fields']}")
    (D / "G2_pins.txt").write_text("\n".join(lines) + "\n")
    print("G2:", "PASS" if ok_all else "FAIL")

    # ── G2b: armB recoveries across ALL logs — every CR recovery must
    #    be a pin or an individually ground-truth-verified frame ───────
    extra, verified = [], []
    for mode in ("sample", "argmax"):
        for f in sorted((D / f"g2_armB_{mode}").glob("*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                rf = r.get("recovered_fields") or []
                if (r["status"].startswith("decision_recovered")
                        and any("commit_reconciliation" in x for x in rf)
                        and r.get("captured_at") not in PINS):
                    cap = r.get("captured_at")
                    v = VERIFIED_BONUS.get(cap)
                    if (v is not None
                            and r["pot_total"] == v["pot_total"]
                            and r["hero_stack"] == v["hero_stack"]
                            and r["facing_bet"] == v["facing_bet"]):
                        verified.append((mode, f.name, r.get("seq"), cap))
                    else:
                        extra.append((mode, f.name, r.get("seq"), cap))
    with open(D / "G2_pins.txt", "a") as fh:
        fh.write(f"\nCR recoveries beyond the 3 pins: "
                 f"{len(verified)} verified-in-class (seq-697 1635-family "
                 f"bonus; verdict in SUMMARY.txt), {len(extra)} "
                 f"UNVERIFIED\n")
        for e in verified:
            fh.write(f"  verified bonus: {e}\n")
        for e in extra:
            fh.write(f"  EXTRA (UNVERIFIED): {e}\n")
    g2b_ok = len(extra) == 0
    ok_all &= g2b_ok
    print("G2b (every CR recovery is a pin or verified):",
          "PASS" if g2b_ok else f"FAIL ({len(extra)} unverified)")

    # ── G3: seq 276 still refuses in armB ───────────────────────────
    lines = ["G3 — seq-276 dead-SB fixture must STILL refuse", "=" * 64]
    g3_ok = True
    for mode in ("sample", "argmax"):
        b = load(f"g2_armB_{mode}", SEQ276_LOG)
        r = b["20260612_122700_038"]
        ok = (r["status"] == "safe_fold"
              and "commit reconciliation declined" in (r["skip_reason"] or "")
              and "no clean REGULAR hand-start anchor" in (r["skip_reason"] or ""))
        g3_ok &= ok
        lines.append(f"\n--- mode={mode} ---")
        lines.append(f"status={r['status']}  [{'OK' if ok else 'FAIL'}]")
        lines.append(f"skip_reason={r['skip_reason']}")
    (D / "G3_seq276.txt").write_text("\n".join(lines) + "\n")
    ok_all &= g3_ok
    print("G3:", "PASS" if g3_ok else "FAIL")

    # ── G4: the 9 displacement-family frames — no silent recovery ───
    lines = ["G4 — P2 gate-2 displacement-family frames under armB "
             "(P1+P2+CR), frame-by-frame", "=" * 64]
    g4_ok = True
    for mode in ("sample", "argmax"):
        lines.append(f"\n--- mode={mode} ---")
        for log, cap, seq in G4_FRAMES:
            b = load(f"g2_armB_{mode}", log)
            r = b[cap]
            recovered = r["status"].startswith("decision_recovered")
            g4_ok &= not recovered
            lines.append(
                f"{log} seq={seq} cap={cap}\n"
                f"  status={r['status']}"
                f"{'  *** RECOVERED ***' if recovered else ''}\n"
                f"  skip_reason={r['skip_reason']}")
    (D / "G4_displacement_family.txt").write_text("\n".join(lines) + "\n")
    ok_all &= g4_ok
    print("G4:", "PASS" if g4_ok else "FAIL")

    print("OVERALL:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
