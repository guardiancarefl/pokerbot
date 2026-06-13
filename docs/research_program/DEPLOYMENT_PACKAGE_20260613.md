# SESSION-5 DEPLOYMENT PACKAGE — operator review (2026-06-13)

Six items from the session-5 work. Each: what it does, arming, gate
evidence, and any caveat. **Nothing here is armed** — arming is your
line per item. Standing rule: every floor/bridge change ships
flag-gated OFF; the live path is unchanged until you arm it.

Validation artifacts (deployed model):
- ckpt `b79e82dd…`  ·  abstraction `0fc20800…`

---

## 1. Shove-defense floor (EV — slate-2a)  — READY, arming = your call

- **What:** at facing-all-in 5–15bb nodes, an equity break-even gate
  (hero hand-class equity vs the frozen killphil shove range, ICM
  break-even from the same arithmetic the battery oracle uses); when
  equity < threshold (τ=0, pure break-even), move CALL/ALLIN mass to
  FOLD. Probe-only floor today — NOT yet wired into the deployed
  `make_live_policy_filter`; arming it for live = a wiring step + the
  H1-style live gate chain (flag, byte-identity, dry-run audit).
- **Gates (all pre-registered, all PASS):** M-A battery F1 (M1
  0.0867→≤0.0434 + call-mass band); M-B 4k paired self-play F2 (no EV
  cost, z>−2); M-C killphil holds-improves (−0.030 vs −0.080,
  extraction halved); **shoviest-rows panel** (your supplement): all 4
  rows positive, pooled **+0.0108 ± 0.0023, z=+4.7**, no row near z≤−2.
- **Evidence:** `evals/a2a_shove_floor_probe_20260612/`,
  `evals/a2a_shoviest_panel_20260612/results.json`.
- **Caveat (carried):** the oracle is killphil-range-specific and
  MH-ICM-based at 6-alive equal-stack cells; ICM Tier-2 will bound it.
  The self-play + panel EV evidence is the load-bearing case, not the
  oracle. **Recommendation:** strong EV case; if armed, treat the
  live-wiring as its own H1-style gated step.

## 2. P2 bet-closure recovery  — BUILT+GATED (3bf5d3f), arming = your call

- **What:** recovers displacement-signature misrender frames (one
  seat's bet/stack split corrupted) via anchor chip-conservation, full
  replay+invariant re-validation; dead-SB guard.
- **Gates:** flag-off byte-identity 6077/6077; flag-on 9 annotation-only
  refusals, 0 decision changes; 23 tests. Session-5 validation: it
  REFUSED all 3 of tonight's casualties **correctly** (they were the
  mirror class — see item 5). It still uniquely covers the seq-276
  misrender family.
- **Evidence:** `evals/p2_build_20260612/`,
  `evals/p2_session5_validation_20260613/`.
- **Recommendation:** safe to arm (zero decision changes, audit trail);
  it will NOT save the dealer-burst/commit class — item 5 does that.

## 3. Click-plan RAISE→CHECK fix  — COMMITTED (e0ca30e), rides next session

- **What:** removed the realization rung that clicked CHECK when no
  RAISE/BET was readable (a transient pre-action-panel misread became an
  irreversible forfeit). Now returns an unexecutable no-op → listener
  retries next frame; fallback watchdog still guarantees action.
- **Gates:** flag-off 0/1233 plan diffs over 24 logs; flag-on 19 diffs
  ALL in the exhibit class (seq194/610, prior 716, seq2629 ALLIN); 29
  click tests. **This is inside the `--extended-click-plans` path** — if
  you run that flag (you have been), the fix is live the next session.
- **Evidence:** `evals/clickplan_fix_20260613/`.

## 4. D2 dead-button handling  — COMMITTED (4c4e4f9), arming = your call

- **What:** the session-5 #1 hand-killer. Admits correctly-scraped
  DEAD buttons (button on an eliminated seat); BB-advances-one-active
  derivation with the frame's observed posts as ground truth. Flag
  `--dead-button-handling` (OFF), prefers Windows' additive
  `dealer_dead` key when present.
- **Gates:** flag-off byte-identity 9750/9750 (17 logs); flag-on
  argmax-isolated 223 diffs ALL in-class (15 recovered decisions across
  7 sessions incl. QdKc + TsAc); 24+239 tests; chip closure exact.
- **Caveat:** chip-equivalent dead-SB *labeling* residual on certain
  re-raise frames (state/chips/legality exact, position label off) —
  same P2-class limitation; the blind-structure-into-replay follow-up
  fixes labeling.
- **Evidence:** `evals/d2_dead_button_20260613/`.

## 5. CR anchored commit reconciliation  — COMMITTED (4c4e4f9), arming = your call

- **What:** the fix for tonight's actual casualties (8c8d, AcAh, 5hTs).
  When the frame conserves vs the anchor but the RECON under-counts a
  non-hero commit (swept folded blind, limp-then-fold), rebuild the
  believed action sequence from anchor-implied commits — frame never
  patched, full re-validation. Flag `--commit-reconciliation` (OFF,
  requires P1).
- **Gates:** combined-tree flag-off byte-identity 9750/9750; all 3
  session-5 pins recover (**AcAh → CALL 613, == your played
  ground truth**); seq-276 STILL refuses; all 9 displacement-family
  frames refuse; 27 tests; suite clean modulo pre-existing.
- **Caveat (arming-relevant):** seq 1635 recovers to the registered
  STATE but the *policy* then samples ALLIN (0.63 post-floor) rather
  than the CHECK that was logged — an RT-1/H1-tail interaction at the
  recovered short-stack spot, flagged for your arming judgment.
  Also: one bonus recovery (183701 seq697), ground-truth-verified.
- **Evidence:** `evals/commit_reconciliation_20260613/`.

## 6. F2 all-in-zero + abort-counter fixes  — SPLIT STATUS

The prior build agent died mid-run (model error). Manager verified the
working tree directly:

- **6a. Abort-counter phantom-reset fix (`fallback.py`): COMPLETE.**
  The "2 consecutive fallback hands" counter was reset by a phantom
  single-card hand, so abort never tripped tonight. Fix: counter keys
  on evidence-hands only. **Gate:** corrected counter flips the abort
  verdict on EXACTLY session 230149 (legacy False → evidence True), all
  16 other sessions unchanged — surgical. 41 unit tests pass; flag-off
  byte-identity 9750/9750 (manager-run). Inside the existing
  `--abort-enforce` path.
- **6b. F2 0-stack-in-hand = ALL-IN (`scraper_schema.py`+`live_loop.py`):
  CODE BUILT, FLAG-OFF SAFE, POSITIVE GATE IN PROGRESS.** Flag
  `--allin-zero-stack` (OFF). Unit tests pass; manager-run flag-off
  byte-identity 9750/9750 (0 default-behavior change). **NOT yet
  certified:** the flag-on counterfactual proving it rescues the
  KQo/ATo pointed-seat class was not produced before the agent died —
  finisher relaunched. Honesty note: tonight's raw KQo/ATo frames carry
  the Windows ocr_int bug (seat reads None, not 0), so F2 rescues the
  class only once Windows F1 ships the real zeros; the finisher is
  proving this on post-F1 synthetic fixtures.
- **Status:** 6a committable now; 6b commits with 6a as a unit once the
  F2 positive gate + SUMMARY land. Both are flag-off-safe in the tree.
- **Evidence:** `evals/f2_abort_fixes_20260613/` (abort gate +
  manager flag-off run present; F2 positive gate pending).

---

## What's deployed vs proposed (summary)

| # | item | state | arming |
|---|---|---|---|
| 1 | shove floor | gated PASS (probe-only; needs live-wiring) | your call |
| 2 | P2 | committed+gated | your call (safe) |
| 3 | click-plan | committed; rides `--extended-click-plans` | already live |
| 4 | D2 dead-button | committed+gated | your call |
| 5 | CR reconciliation | committed+gated (1635 note) | your call |
| 6a | abort-counter | gate-complete, uncommitted | your call |
| 6b | F2 all-in-zero | code+flag-off-safe; pos-gate pending | hold |

## Standing context
- H4 unlocks next session (485/500) — freeze checklist staged.
- Stage-2 ledger: gate streak 0/5 toward your confirmed N=5; items
  4/5/6 directly attack tonight's 8 never-decided hands.
- OQ-3 (pointed-seat OCR) routed to Windows; F1 their side, F2 bridge.
