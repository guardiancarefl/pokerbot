# Layer 3 — Per-Street Contribution Ledger: Design & Desync Analysis

**Status: DESIGN ONLY — not built. Do not build without explicit approval
of the desync handling below.** (User gate, 2026-06-09.)

---

## 0. Evidence correction that reframes this design

The layer-3 ledger was motivated by the 2026-06-09 diagnosis that the
verify1 blackout frames (seq 83-94) carried a **second, unflagged bad
field** (seat5's stack "frozen at 1347"), which single-field conservation
recovery cannot handle (one equation, two unknowns).

**That diagnosis was wrong.** The analysis printed only `stacks.seat1` per
frame and extrapolated seat5 from an earlier frame. Full re-extraction
shows seat5 tracked correctly the whole way (1347 → 1197 after the flop
call → 717 after the river bet → 0 all-in → 87 final, matching the seq=95
post-hand ground truth). The incident was **pure single-field corruption**
— exactly the class layer 1 closes.

Empirical result after building layer 1 (commit 7d47e86): all three real
hero-to-act decisions in the blackout (seq 88, 92, 94) recover, with
derived values (1110 / 1110 / 150) confirmed against ground truth. Across
692 replayed frames (3 dry-run logs, 83 suspect frames), **zero**
hero-to-act suspect frames remain that a contribution ledger would have
additionally rescued.

So the honest framing: **layer 3 is no longer kick-risk-critical for any
observed incident.** It addresses a hypothetical-but-real residual class
(genuine multi-field corruption on a decision frame). This doc specifies
the design so it can be built quickly if that class is ever observed; the
recommendation in §6 is to hold.

## 1. What the ledger is

A per-hand, bridge-side accumulator of **voluntary chip commitments per
seat per street**, sourced from the `bets` fields of observed frames —
including suspect frames, whose bet fields are typically clean (verified
in the blackout: hero's flop 150 and seat5's river 480 were all readable
in the *dropped* frames).

It yields a per-seat stack prediction independent of stack OCR:

```
stack_pred[i] = pre_hand[i] − ante − Σ_streets commit[i][street]
```

which turns the single closure equation (1 unknown max) into 6
independent predictions — enabling multi-field recovery, strict/UI-lag
disambiguation, and validation of *unflagged* fields.

## 2. Data structure & update rules

```
ContributionLedger (one per SessionTracker, same lifetime)
  hand_key            (dealer_seat, sb, bb, ante)   # same key as tracker
  street_idx          int                            # last confirmed street
  committed[6][4]     finalized chips per seat per street
  pending[6]          candidate current-street bets (unstable until voted)
  stable[6]           per-seat (value, consecutive_count)
  diverged            bool — pot audit failed; ledger dead for this hand
```

Update on EVERY frame the bridge sees for the tracked hand (clean or
suspect — bet fields only; stack fields are never read into the ledger):

1. **Hand reset.** New hand_key ⇒ reset everything. The ledger never
   carries state across hands (see §4.4).
2. **Stability vote.** A bet value for seat *i* enters `pending[i]` only
   after being read identically on **2 consecutive frames** (kills the
   seq=91 `48`-for-480 mid-animation transient; polling gives ~1-3
   frames/s so a real bet is always seen ≥2×).
3. **Within-street monotonicity.** A seat's stable bet may only grow
   within a street (call → raise). A stable *decrease* without a street
   transition ⇒ mark `diverged` (animation artifact or missed
   transition; don't guess which).
4. **Street transition = collection point.** Confirmed by BOTH signals:
   board length increased AND all bets read 0/null. On confirmation:
   `committed[i][old_street] += pending[i]`, clear pending, advance
   street_idx. Board jumping by >1 street (missed window) is fine —
   intermediate streets had bets we never saw; the pot audit (below)
   decides whether anything was missed.
5. **Pot audit (every frame, the anti-silent-desync core).** The ledger
   must explain the observed pot under the same two conventions
   `_closure_plausible` accepts:
   `pot_pred_strict = Σ committed + antes + blinds_accounting` and
   `pot_pred_uilag = pot_pred_strict − Σ pending`. If the frame's pot
   (or its tracker-corrected value) matches neither ⇒ `diverged = True`
   for the remainder of the hand. No partial trust: diverged means the
   ledger proposes nothing until the next hand-start reset.

## 3. How it's consumed

Strictly as a **candidate generator inside the existing recovery path** —
never as a value source of its own:

- Extends `_attempt_suspect_stack_recovery`: when the single-unknown
  closure solve declines (multi-seat reasons, out-of-range candidate, or
  strict/UI-lag ambiguity), ledger-predicted stacks for the affected
  seats form one additional candidate frame.
- Every candidate still passes the **unchanged replay + invariant gate**
  (same as layer 1; no gate is loosened, anywhere).
- `diverged == True` ⇒ ledger contributes nothing ⇒ behavior is exactly
  layer-1-only.

The model can never see a ledger value that didn't survive: (a) 2-frame
stability vote, (b) per-frame pot audit, (c) full replay+invariant.

## 4. Desync handling (the approval-gating section)

The frame-only dead-button rule (DECISIONS.md) rejected history trackers
because they desync *silently*. The ledger's design answer is that every
failure mode below either self-detects on the next frame or degrades to
the current safe-fold behavior:

### 4.1 Missed frames (scraper gap spans a bet)
A bet placed and collected entirely inside a dropped window never enters
`committed`. The ledger's pot prediction then undershoots the observed
pot on the very next frame ⇒ pot audit fails ⇒ `diverged` ⇒ no
substitution for the rest of the hand ⇒ identical to today's behavior.
**A missed observation can suppress recovery; it cannot fabricate one** —
the pot audit is an independent per-frame observable, not accumulated
state, so divergence is detected at first use, not discovered later.

### 4.2 Mid-animation transient bets (the seq=91 "48")
Two defenses: the 2-consecutive-frame stability vote (a transient appears
once; the next poll reads the settled value) and within-street
monotonicity (a stable-looking value that later shrinks ⇒ diverged). A
transient that is stable across 2 frames AND consistent with a
same-direction pot misread would be needed to poison the ledger — and the
resulting candidate must *still* produce a legal OpenSpiel replay
matching all six stacks/bets/pot/cards. That is a strictly harder
coordinated-failure bar than the single stuck digit that caused the
actual incident.

### 4.3 Street-boundary UI lag (pot/board/bets update at different times)
Collection requires board-change AND bets-cleared together; frames inside
the transition window leave `pending` untouched and are absorbed by the
strict/UI-lag dual accounting in the pot audit — the same two-convention
tolerance the tracker already uses for pot correction.

### 4.4 Missed hands / resync
The ledger holds **zero cross-hand state**. It anchors at the same clean
hand-start frame SessionTracker anchors at (and, like the tracker, never
reads suspect frames for anchoring). Missed hand-start ⇒ ledger inactive
for that hand ⇒ layer-1-only behavior. Resync is therefore automatic and
total at every clean hand-start; maximum staleness is bounded by one
hand. This is the same state-lifetime contract SessionTracker already
operates under — the accepted precedent for "stateful but per-hand".

### 4.5 Residual silent-desync surface
The genuinely unclosable case: correlated wrong-but-stable bets plus a
consistently wrong pot, jointly encoding a *legal alternate hand history*
consistent with five other stacks, the board, and the button. No
field-level recovery can defend against a fully self-consistent forgery —
neither can the current pipeline (a clean-parsing frame with those
properties already passes today). The ledger does not widen this surface:
it only proposes candidates into the same gate clean frames pass through.

## 5. Cost & blast radius

~150-200 lines in `scraper_schema.py` (ledger class) + ~40 in
`live_loop.py` (candidate wiring), all behind the suspect-only branch —
non-suspect frames can't reach it, so the layer-1 bit-identity proof
methodology (replay_make_decision_diff.py) applies unchanged. Main
engineering risk is Ignition's street-boundary UI-lag patterns (§4.3);
that's also where test effort concentrates (the 0608/0609 corpora contain
every transition pattern observed so far).

## 6. Recommendation

**Hold layer 3.** Build it only when the audit trail shows it's needed:
layer 1 now annotates every declined recovery with its reason in the
dry-run JSONL, so the trigger condition is directly observable — **a
hero-to-act suspect frame declined for "no in-range closure candidate" /
"ambiguous" / "no candidate passed replay+invariant" that a correct
ledger would have saved.** Current count across all replayed corpora: 0.

Priority instead: **layer 2 (Windows scraper last-good freeze)** — still
real and now the binding residual risk. Specifically:

1. During the 12-frame suspect run the scraper kept rejecting the
   *correct* OCR (`150` at seq 93/94) against a 12-frame-stale reference
   (1260). Layer 1 bails this out bridge-side mid-hand, BUT:
2. **Anchor starvation**: if a stuck-digit suspect run survives into the
   next hand's hand-start frame, that frame arrives suspect, the tracker
   (correctly) refuses to anchor from it, and layer 1 loses its anchor
   for the entire next hand — recovery declines everywhere, full
   blackout again. The scraper-side fix (reference decay, or accepting a
   value that re-stabilizes N consecutive frames) is the root-cause cure
   and lives in the Windows scraper's SanityChecker.
