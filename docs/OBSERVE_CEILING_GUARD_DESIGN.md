# observe() consensus-ceiling anchor guard — Design for review

**Status: DESIGN ONLY — not built. User approval required before build.**
(Gate set 2026-06-09; same protocol as layer 1 and the layer-3 ledger.)

## 0. Why this is a correctness fix, not defense-in-depth

The original framing was residual risk: scraper P1 rejects over-ceiling
reads going forward, but can't retroactively stop a poisoned hand-start
from anchoring the bridge. Replaying `live_dryrun_20260609_154557.jsonl`
through CURRENT code (layer 1, `7d47e86`) shows the risk is already a
live defect, and layer 1 makes it worse than pre-layer-1 behavior:

- `SessionTracker.observe()` accepted both `9907` hand-starts (Σpre-hand
  = 17917) — its self-consistency check passes because both sides of the
  equation use the same wrong stack.
- Clean frames in those hands self-quarantine (`pre_hand_for`'s closure
  check fails against the poisoned anchor → simple-model fallback), BUT
- **layer-1 recovery consumed the poisoned anchor 7 times** —
  `anchor_for()` deliberately skips the closure check (the flagged field
  breaks closure by definition), so the conservation solve ran with
  Σpre = 17917 and derived phantom seat6 stacks that the invariant then
  validated, because `check_mid_hand_invariant` reconstructs from the
  SAME poisoned `state_pack.pre_hand_stacks` — the error cancels
  self-consistently. The Q3 trust argument ("five other stacks verify
  independently") holds for frame fields but NOT for the anchor itself;
  the anchor had no independent plausibility check. That is the hole.

The seven poisoned recoveries (captured_at, derived value):

```
20260609_115050_884  decision_recovered  stack.seat6=9897
20260609_115104_614  decision_recovered  stack.seat6=9822   <- frame's actual read was the CORRECT 905
20260609_115123_111  decision_recovered  stack.seat6=9822
20260609_115124_294  decision_recovered_cached   seat6=9822
20260609_115126_590  decision_recovered_cached   seat6=9822
20260609_115127_251  decision_recovered_cached   seat6=9822
20260609_120521_330  decision_recovered  stack.seat6=9907   <- "confirmed" the poison itself
```

Severity: hero's own stack/pot/board were correct in all seven, so no
chip-accounting error — but the model decided with a phantom ~9800-chip
opponent stack feature (true ~905), i.e. an unverified derived value
reached the model. Exactly what the invariant discipline exists to
prevent.

## 1. Mechanism

One enforcement point: `SessionTracker.observe()`, before accepting a
hand-start anchor. (If a poisoned anchor is never stored, `pre_hand_for`
and `anchor_for` cannot return one — no second check site needed.)

```
ceiling = chips-in-play consensus            (see open Q1)
pre     = pre_hand_stacks(frame)             (existing)
REFUSE the anchor (return without storing) if:
    any(pre[i] > ceiling)                    per-seat bound
or  sum(pre)  > ceiling                      table-total upper bound
(existing self-consistency check unchanged, runs after)
```

Refusal means the tracker keeps the previous hand's anchor; for frames
of the poisoned hand, `hand_key` mismatches → `pre_hand_for` /
`anchor_for` / `corrected_pot_for` all return None — identical to the
"missed hand-start" path that already exists and is handled everywhere
(simple-model fallback for clean frames; "no clean pre-hand anchor"
decline for recovery). No new states, no new failure modes. Refusals log
loudly (`[ANCHOR-REFUSED] ...`) for live audit, like `[FLOOR]`/`[OOD-WARN]`.

## 2. Open questions, with recommendations

### Q1 — Ceiling source / warm-up period

The scraper-side rule (consensus of recent clean hand-starts, never a
constant) has a bootstrap problem bridge-side: the bridge's consensus
would be built from the anchors observe() accepts, and the guard decides
what observe() accepts — at match start (zero history) a poisoned hand 1
would BE the consensus. Options:

- **(A) Config product**: `STARTING_CHIPS × NUM_SEATS` (= 9000). These
  are already deployment-format pins in `scraper_schema.py:33-34` that
  the entire training/eval/replay stack depends on; chips never leave a
  single 6-max table, so the true total is constant for the whole match.
  Not the scraper's `13000`-class magic number — if the format changes,
  this config changes or everything else breaks loudly first. No warm-up
  hole.
- **(B) Pure observed consensus**: mode of accepted hand-start sums,
  guard inactive for the first K hands. Honors "derive, don't constant"
  literally, but the warm-up hole is worst exactly where the data is
  thinnest, and a hand-1 poison seeds the consensus itself.
- **(C) RECOMMENDED — A as the ceiling, observed consensus as a
  cross-check**: guard with the config product from frame 1 (no warm-up
  hole); independently track the mode of accepted hand-start sums; if a
  session's observed mode ever disagrees with the config product, emit a
  loud `[ANCHOR-OOD]` log (wrong table format mounted — e.g. a
  2000-chip variant under the 1500 config), because every layer above is
  OOD at that point, not just the ceiling. The derivation requirement is
  honored where it matters (the check that fires is validated against
  observed table state every hand-start); the constant is load-bearing
  config, not a tolerance picked by hand.

### Q2 — Per-seat vs table-total

They catch different poisons; both are one-line upper bounds, so:
**RECOMMENDED — both, upper bounds only.**

- Per-seat (`pre[i] > ceiling`): catches a single huge stuck digit even
  when a co-occurring under-read pulls the SUM back under the ceiling
  (e.g. seat6=9907 with another seat hidden: sum 8,9xx passes, per-seat
  fires).
- Table-total (`Σpre > ceiling`): catches multi-seat moderate over-reads
  that each stay under the ceiling individually (two seats +500 each).
- **No lower bound** — deliberately. 30 of the corpus's 178 clean
  hand-starts sum BELOW 9000 (8950/8900/8200/…, the mis-alive
  hidden-chip under-reads). Those anchors are currently accepted and the
  mis-alive tolerance machinery in `_closure_plausible` exists
  specifically to work with them. A strict-equality or lower-bound check
  would refuse ~17% of good anchors — a real recovery/override
  regression for zero demonstrated benefit. Future tightening if
  under-read anchors ever cause a validated incident.

### Q3 — The v == total boundary

Unreachable in any frame that can reach observe(): `parse_frame` rejects
n_alive < 4 (`scraper_schema.py:272`), and every alive seat has
pre-hand > 0, so any single seat's pre-hand is ≤ total − 3 strictly. A
seat equal to the full chips-in-play implies everyone else at zero — the
match is over and no valid hand-start exists. **RECOMMENDED: strict `>`**
(mirrors the scraper P1 spec wording); `>=` would be equally safe and
the choice is behaviorally indistinguishable on reachable frames.
Document the unreachability argument at the check site.

## 3. test9907 fixture — the frames the guard must handle

All from `logs/live_dryrun_20260609_154557.jsonl` (in
`tools/scraper_sanity_fixture/streams/` as `…154557__s1/s2`) unless
noted. Verification = `scripts/replay_make_decision_diff.py` pre/post on
ALL FOUR raw-record dry-run logs (this time including 154557 — closing
the layer-1 audit gap that let this slip through) + unit tests on the
named frames + full bridge/integration suite.

**Must REFUSE anchor (2 frames — the only intended anchor deltas):**

| captured_at | why |
|---|---|
| `20260609_115039_013` | hand-start, seat6 pre-hand 9912 > 9000, Σpre 17917 |
| `20260609_120518_206` | hand-start, same poison, second sub-session |

**Must STILL anchor: all 176 other clean hand-starts** across the
corpus, explicitly including every under-read sum (8950, 8975, 8950,
8900, 8200×7, 8169, 8925, 8700, 8600, 8990, 8887, 8850, 7600, 7558,
8679, 6989) — zero anchor regressions tolerated.

**Downstream must-change (exactly the 7 poisoned recoveries):** each
becomes `skip_data_quality … (recovery declined: no clean pre-hand
anchor for this hand-key)` — the pre-layer-1 safe-fold, which is correct
here: with a poisoned anchor the bridge cannot verify anything in that
hand, so it must not decide.

**Downstream must-NOT-change:**
- the 6 legitimate recoveries in the same log (`115411_544`,
  `115415_453` → seat6=940; `115908_671` → 896; `115921_099`,
  `115928_815`, `115937_827` → 846) — their anchors are clean;
- verify1's 3 recoveries (seq 88/92/94) and both prior diff-corpora —
  re-run the full 4-log behavioral diff expecting deltas ONLY at the 7
  poisoned recoveries;
- all clean frames of the poisoned hands themselves (they already
  self-quarantined via the closure check / hand-key mismatch, so anchor
  refusal is behaviorally invisible to them — the diff must confirm).

## 4. Blast radius

~15 lines in `SessionTracker.observe()` + the consensus cross-check
state. No parser, replay, invariant, or recovery-logic changes. The
guard can only ever REFUSE an anchor that today would be accepted —
every refusal degrades to the already-tested no-anchor path. The
behavioral delta on the entire known corpus is exactly: 7 poisoned
recoveries → safe-folds.
