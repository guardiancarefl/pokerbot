# H1 — Commitment-scaled tail floor (deployment-time policy filter)

**Status:** spec reconstructed 2026-06-12 from the operator-approved CAN-RIDE
line ("tail floor (policy call)", SESSION_LOG 2026-06-11) + the program brief.
The full prior spec was conversational, not on disk; THIS document is now the
binding spec. Gate definitions below are verbatim from the program brief.

## Problem

Sample mode draws from the full mixed strategy. Tail accounting across live
sessions 1–3 is statistically clean (z=+0.49..0.57 — no mechanical defect),
but individual low-mass draws can commit the whole stack: session-1 seq-315
sampled a 2d5c open-shove from a 7.5% ALLIN tail. The mixed strategy's value
in those tails is theoretical (balance vs adaptive opponents); the realized
cost in a 6-max SNG with ICM is concrete. Hypothesis: pruning low-probability
actions *in proportion to how much they commit* gains EV against the live
field without opening an exploitable leak.

## Definition

`apply_commitment_tail_floor(policy, legal_mask, parsed, state, tau_max)`:

- `hero_stack = parsed["money"][cp]`; no-op if ≤ 0 or parsed fields missing.
- `to_call = max(contribution) − contribution[cp]`, `pot = sum(contribution)`.
- Per legal action `a`, `commit(a)` = chips this action would ADD, using the
  same sizing semantics as the downstream chip translation
  (pot-fraction sizes ≈ `min(to_call + f·(pot + to_call), hero_stack)`;
  `CALL = min(to_call, hero_stack)`; `ALLIN = hero_stack`; `FOLD = 0`;
  CHECK ≡ CALL at to_call=0 ⇒ commit 0).
- `commit_frac(a) = commit(a) / hero_stack` ∈ [0, 1].
- **Threshold scales with commitment:** `τ(a) = tau_max · commit_frac(a)`.
- Prune every action with `0 < policy[a] < τ(a)`; renormalize remaining mass.
- Degenerate guard: if pruning would leave total mass ≤ 1e-12, return the
  input policy unchanged (identity reference).
- Identity-short-circuit contract: return the SAME `policy` reference when
  nothing fires (chain cost = reference check), matching the three existing
  floors in `src/nlhe/integration/live_loop.py`.

Consequences: FOLD and free CHECK are never pruned (commit 0 — bad folds are
the other floors' jurisdiction). A 7.5% ALLIN is pruned at tau_max ≥ 0.075.
A 7.5% quarter-stack bet needs tau_max ≥ 0.30 — untouched at all candidate
values. Calling off a shove is treated identically to jamming (commit ≈ 1).

## Wiring (flag-gated OFF, Stage-2 precedent — arming is an operator call)

- `make_live_policy_filter(..., tail_floor_tau: float | None = None)` —
  filter appended LAST in the chain (after short-stack floor) **only when
  tau is not None**. OFF ⇒ chain is byte-identical to today.
- `make_decision(..., tail_floor_tau=None)` pass-through.
- `scripts/run_live_dryrun.py --tail-floor-tau <float>` (default None = OFF).
- Firing log: `[FLOOR] fired=[tail] ...` with pruned action names + masses,
  same audit format as existing floors.
- **No live-path behavior change without operator approval line.** This build
  is flag-OFF; TG1 proves the OFF path byte-identical.

## Gates (pre-committed; defined in the program brief — may not be weakened)

- **TG1 — byte-identity flag-off.** `scripts/replay_make_decision_diff.py`
  over ALL raw-record dry-run logs (standing rule: all logs, never a subset).
  Bar: 0 behavioral diffs. Any diff ⇒ implementation rejected.
- **TG2 — counterfactual at three thresholds.** Replay all recorded decision
  frames; apply the floor at τ_max ∈ {0.05, 0.10, 0.15} to each pre-sample
  distribution. Report: firing rate, pruned-action census, historical sampled
  actions that would have been excluded, argmax changes.
  Bars: (a) seq-315-class draw (7.5% ALLIN) caught at τ_max ∈ {0.10, 0.15};
  (b) **zero argmax changes** at any threshold (the floor must reshape tails,
  never flip the modal decision); (c) altered-decision rate at τ_max=0.10
  ≤ 5% of decision frames — above that the floor is too aggressive: STOP and
  investigate, do not proceed to TG3.
- **TG3 — 2k-game CRN-paired panel, floor-on vs floor-off** (champion ckpt
  both arms; harness = `scripts/eval_icm_panel_floor.py` pairing method, or
  the short-stack-floor A/B harness `scripts/short_stack_floor_ab.py`
  adapted). τ_max for the armed arm = the TG2 winner.
  **Ship bar (falsification):** paired all-games delta NOT significantly
  negative (z > −2) **AND** diverged-games per-firing delta > 0 with z ≥ 2.
  If the paired eval shows no EV gain, **it does not ship** — verbatim
  program rule. No re-rolling seeds, no threshold-shopping after the fact:
  τ_max is fixed by TG2 before TG3 launches.
- **TG4 — attacker re-check.** Re-measure extraction (B4/B5-style batteries,
  floors ON incl. tail floor) reusing the already-trained
  `runs/attacker_v1/ckpt_iter_1000.pt` and
  `runs/attacker_bubble_v1/ckpt_iter_0600.pt` — measurement only, no attacker
  retraining. Bar: extraction no worse than champion's B4/B5 floors-ON
  baselines (−0.0400±0.0158 std, −0.0998±0.0133 bubble) beyond 2σ. A new
  leak opened by the floor ⇒ kill, regardless of TG3.

## Scheduling note

TG1/TG2 are light (minutes–1 h) and run anytime at nice/taskset. TG3/TG4 are
>30 min ⇒ run ONLY when no live dry-run is active (live session in progress
as of 2026-06-12 04:20 — both deferred, exact commands queued in
NEXT_RESUME.md).

## Cost estimate

Build + tests ≈ 2–3 h · TG1 ≈ 0.5 h · TG2 ≈ 1 h · TG3 ≈ 4–8 h CPU ·
TG4 ≈ 6–10 h CPU. Kill criteria as in the gates; impatience-kills prohibited
(kill only when the *type* of problem changes).
