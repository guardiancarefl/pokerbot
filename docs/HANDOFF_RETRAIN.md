# HANDOFF SPEC — Library Rewrite + Retrain

**Goal:** Replace the inflated-BB action-set distortion that caused the model
to learn an over-tight preflop strategy (verified 2026-06-05 via direct policy
query: 2.4% BTN open rate, 0% CO open rate at deep stacks; correct would be
40-50% / 25-35%).

**Root cause (single):** OpenSpiel's `universal_poker` does not support antes,
so antes were folded into the big blind as `inflated_BB = bb + N×ante`. That
inflated BB raises OpenSpiel's `min_bet = 2×BB` constraint above all small
pot-fraction bets, leaving the model with only `{fold, limp, 5.6×BB min-open,
all-in}` as its preflop action set. An ICM-aware agent trained against that
menu correctly learned to fold most hands because every "open" option is
either too large or all-in — both exploitable. Same root cause produces
runtime bet-sizing drift AND degenerate trained strategy.

**Decision:** Trigger library rewrite + retrain. Bridge has done its job
(surfaced the issue safely log-only).

---

## (1) THE LIBRARY FIX

### The constraint (load-bearing — read first)

OpenSpiel's `universal_poker` parameter list (verified via
`pyspiel.load_game(...).get_parameters()`):
```
betting, bettingAbstraction, blind, boardCards, calcOddsNumSims,
firstPlayer, handReaches, maxRaises, numBoardCards, numHoleCards,
numPlayers, numRanks, numRounds, numSuits, potSize, stack
```
**No `ante` parameter.** No other registered OpenSpiel poker game supports
antes either (`leduc_poker`, `kuhn_poker`, `repeated_poker` all checked).

So "thread ante through the game string" is not a free move — it requires
one of the following paths.

### Three realistic paths (pick one)

**Path A — Patch OpenSpiel `universal_poker` for native ante support (RECOMMENDED).**

Smallest C++ change in the right place. Estimated 30-50 lines in OpenSpiel's
`open_spiel/games/universal_poker/universal_poker.cc` + `.h`:
- Add `ante` to the game-parameter list (alongside `blind`)
- In `UniversalPokerState` constructor / new initial state: for each seat,
  deduct `ante` from `stack`, add to `pot`
- Optionally upstream PR (Anthropic-style courtesy) — others using
  universal_poker for tournament work would benefit
- Rebuild OpenSpiel from source against the patch; pin the build in
  `requirements.txt` or a vendored fork

Pros: correct, future-proof, no chip-arithmetic distortion anywhere.
Cons: C++ work + custom OpenSpiel build pipeline.

**Path B — Use the existing `bettingAbstraction=fcpa` or `bettingAbstraction=fullgame` with a custom min-raise floor.**

Probe what `bettingAbstraction` and `maxRaises` actually permit in universal_poker.
If there's a way to lower the enforced min-raise floor without breaking NL
rules, that could open the action set up without an ante param. **This needs
empirical verification** before committing.

Pros: pure-Python change.
Cons: speculative; may not exist.

**Path C — Drop antes from the trained game (no-ante approximation).**

Use `inflated_BB = bb` (real BB, no ante inflation). Antes simply don't exist
in the OpenSpiel game; the model trains on no-ante 6-max NLHE.

- Pre-action pot at level 1 = SB(15) + BB(25) = 40 chips (vs real 70)
- min_bet = 2×BB = 50 chips → 2×BB opens become legal
- Bet sizing action set: {BET_33=13, BET_50=20, BET_66=26, BET_100=40,
  BET_150=60, BET_200=80, ALLIN=1500} — all small fractions below min_bet
  still get filtered, BUT pot grows fast post-action, so most postflop
  fractions become available immediately

Pros: trivial code change (delete one method, change one formula).
Cons: training game is a different game than reality (43% less dead money
at level 1; pot dynamics are noticeably looser). May produce a model that
plays correctly for *no-ante* 6-max but is too aggressive against real
ante-paying Ignition tables. Acceptable distortion only if late-game push/
fold dynamics dominate (which they DO for double-up survival format) and
early-level deep-stack play is less load-bearing.

**My recommendation: try Path B first (cheap probe), fall back to Path A
(real fix) if it doesn't pan out. Path C is the escape hatch if both fail.**

### File-by-file change list (applies to any path)

These files need updates regardless of which path is chosen — they all
currently reference `inflated_big_blind`. Path A is the cleanest because
they just delete the inflation logic; Path C keeps the BlindLevel class
but removes the `n_players` argument.

**Production code (deletes + edits):**

| File | Current usage | Change after retrain |
|---|---|---|
| `src/nlhe/game_strings.py` | `BlindLevel.inflated_big_blind(num_players)`, `to_inner_game_string`, `to_inner_game_string_for_state`, `to_blind_schedule_string` | (Path A) Threads real `ante` through universal_poker game string. Delete `inflated_big_blind` method. Update all 3 game-string builders. |
| `src/nlhe/cfr6.py` | Uses inflated BB for sampling + CFR state init | Update to use real ante from BlindLevel. Search for `inflated_big_blind` calls and replace. |
| `src/nlhe/stack_sampler.py` | Samples preflop states using inflated BB | Same — replace inflation calls with real ante. |
| `src/nlhe/infoset6.py` | Feature encoding uses pot/stack normalized — does NOT directly reference inflated_BB but may have implicit assumptions | Audit normalization; values should be correct under either path. |
| `src/nlhe/integration/scraper_schema.py` | (bridge) BlindsLevel.inflated_bb | Delete `inflated_bb` (bridge no longer needed). |
| `src/nlhe/integration/translate.py` | (bridge) forward/reverse chip translation | **DELETE ENTIRELY** post-retrain. |
| `src/nlhe/integration/replay.py` | Uses bridge translate | Remove the bridge wiring; chip_ints flow 1:1. |
| `src/nlhe/integration/invariant.py` | (bridge) "ghost antes" bug-match workaround; bridge-aware loose pot check | Delete ghost-antes correction. Restore strict per-seat chip equality (the Phase 1 hand-start invariant) for mid-hand frames — exact chip match becomes possible again. Keep `_repair_folded_from_chip_deductions` import (separate scraper-side bug). |

**Eval scripts (review for distortion):**

| File | Why it matters |
|---|---|
| `scripts/candidate_bakeoff.py` | Reports numbers under the broken distribution — REOPEN bakeoff (section 3 below) |
| `scripts/rebel_gate2.py`, `scripts/match_length_probe.py`, `scripts/rebel_diag_killphil.py` | Diagnostic scripts that used inflated BB for stack sampling. Update to real ante; existing diagnostic numbers may shift. |

**Tests (update + add):**

| File | Change |
|---|---|
| `tests/test_game_strings.py` | Update `inflated_big_blind` tests for the chosen path. Add tests for real-ante game-string generation. |
| `tests/test_stack_sampler.py` | Update for the new sampling distribution. |
| `tests/test_integration_replay_to_decision.py` | Re-tighten chip-equality assertions (no more bridge tolerance). |
| (new) `tests/test_real_ante_game.py` | Sanity tests: load game, verify per-seat stacks deducted correctly, verify pot includes antes. |

**Bridge files — DELETE entirely post-retrain:**

```
src/nlhe/integration/translate.py
tests/test_integration_translate.py
```

**Bridge files — KEEP (orthogonal to the inflated-BB issue):**

```
src/nlhe/integration/scraper_schema.py::_repair_folded_from_chip_deductions
tests/test_integration_folded_repair.py
```
This handles a SCRAPER-side bug (Ignition's stale `folded` field), not a
library bug. Still needed.

---

## (2) THE RETRAIN

### What gets retrained

**Both** the blueprint (k200) AND the rebel value net. They share the same
training distribution (sampled via `stack_sampler` + `cfr6`), so any
distribution change affects both.

| Artifact | Current ckpt | Replace with |
|---|---|---|
| Blueprint (k200) | `runs/k200_blueprint_ckpt_iter_2000.pt` | Retrain from scratch on corrected distribution |
| ReBeL value net | `runs/rebel_policy_net_m3box2.pt` | Retrain after blueprint stabilizes |
| Other candidates (k500, k1000) | `runs/abstraction_k500_*`, `runs/abstraction_k1000_*` | Retrain in parallel if the bakeoff (section 3) is part of the redo |

### Config

The structural training config is unchanged — same Deep CFR loop, same
abstraction, same hyperparameters. The ONLY distribution change is real
antes replacing inflated BB. So:

- Use the existing `configs/six_max_phase4f_dcfr_candC_k200.yaml` (or whichever
  was the production training config) as a starting point
- Update `abstraction_path:` reference if abstraction also needs retraining
  (the abstraction itself uses EMD on equity histograms — equity is unaffected
  by ante structure, so **abstraction retrain is NOT required**; reuse
  `runs/abstraction_20260521_223018_retrofit/abstraction.pkl`)
- Iterations: match the original (k200 was 2000 iters)
- Action space: **unchanged in the abstraction**, but the effective set
  available at each decision node will be wider (more BET_FRACTION options
  legal because min_bet drops from ~110 to ~50 at level 1)

### Wall-time estimate

Calibrate from the existing `runs/k200_blueprint_ckpt_iter_2000.pt` training
log (the timestamp + iteration count gives sec/iter):

```bash
grep -E "iter.*time|wall|elapsed" runs/N7control_k200_train.log | head -5
```

The user has previously cited Contabo's 12-vCPU AMD EPYC as oversubscribed
with ~10× variance per iteration. RunPod RTX 4090 ($0.34/hr Community Cloud)
is the planned GPU. Recommend:

- **Contabo**: only for smoke / sanity (1-50 iters) before committing
- **RunPod**: full 2000-iter retrain. Estimate from prior k200 wall-time
  scaled by GPU speedup factor (~5-10× for CFR-on-GPU vs CPU)

Budget: ~$5-15 for a full k200 retrain on RunPod RTX 4090 if prior CPU
training took ~24h (becomes ~3-5h on GPU).

### Acceptance check (REUSABLE — built this session)

Re-query the BTN/CO preflop range with the new checkpoint. The query script
exists already:

```bash
python scripts/query_model_preflop_range.py \
    --checkpoint runs/<new_blueprint>.pt \
    --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
    --hero-pos BTN
python scripts/query_model_preflop_range.py --hero-pos CO ...
```

**Acceptance thresholds:**
- BTN open frequency: ≥ 30% (target 40-50%)
- CO open frequency: ≥ 20% (target 25-35%)
- Specifically: KK / AKs / AKo / KJs / KQs / AQs / TT / 99 / 88 should all
  OPEN from BTN (current model folds most of these)
- ATo and KJo should at least be MIXED (open at non-zero frequency)

If the retrained model still shows <20% BTN open rate, the fix didn't take —
debug before declaring the retrain complete.

---

## (3) THE BAKE-OFF — REOPENED

All prior bake-off results were measured on the broken distribution and are
invalidated. Re-decide the ship model from scratch:

- Candidates: k200, k500, k1000 (and any other Phase 4f candidates)
- Both fresh-trained on the corrected game
- Re-run `scripts/candidate_bakeoff.py` with the same eval protocol
- The prior winner (rebel_d3k150 or whichever it was — check `docs/STATUS.md`)
  is REOPENED — no longer the default ship candidate

Specific files to re-check:
- `evals/X0_resolver_vs_self.json`, `evals/k1000_gate_*.json` — pre-retrain
  numbers, archive but don't act on them
- `evals/resolver_shards/SWEEP_DONE.flag`, `X5_BUBBLE_DONE.flag` — flags
  marking completed sweeps; clear or rename to `.preretrain` suffix

---

## (4) CLEANUP — POST-RETRAIN

Once the retrained model passes the acceptance check (section 2), delete:

```bash
# Bridge (no longer needed — real chip actions are directly applicable)
rm src/nlhe/integration/translate.py
rm tests/test_integration_translate.py

# Bridge wiring in replay.py — remove the inline `real_to_openspiel_action` call
# (just delete the try/except wrap around `chip_int = real_to_openspiel_action(...)`)
# Edit: src/nlhe/integration/replay.py — search for `real_to_openspiel_action`

# Bridge-aware mid-hand invariant — restore strict chip-equality (Phase 1 style)
# Edit: src/nlhe/integration/invariant.py — remove scraper-self-consistency
# section, restore per-seat chip checks like check_hand_start_invariant does

# Ghost-antes bug-match workaround in invariant.py
# Edit: same file, openspiel_to_scraper_view — remove the
# `ghost_antes = (NUM_SEATS - n_alive) * ante` correction

# DECISIONS.md — mark the "Phase 2 bridge" entry as RESOLVED with the retrain
# Edit: docs/DECISIONS.md — append "RESOLVED 2026-XX-XX by library rewrite"
```

**KEEP (post-retrain — these handle SCRAPER bugs, not library bugs):**
- `src/nlhe/integration/scraper_schema.py::_repair_folded_from_chip_deductions`
  (Ignition's stale `folded` field is a scraper-side bug, independent of antes)
- `tests/test_integration_folded_repair.py`
- All scraper data-quality filters in `scripts/test_integration_midhand.py`
  and `scripts/validate_1500_stack_corpus.py` (hero_cards-empty, pot_missing)

---

## What stays running until retrain ships

- The current model + bridge + log-only resolver run can continue accumulating
  decision data on the live scraper. The decision log is useful AS-IS for
  building up a prod-mirror corpus to validate the retrained model against
  (run `query_model_preflop_range.py` + replay the corpus through the new
  blueprint and compare decisions).
- The `live_1500.jsonl` corpus + 124-decision log is the regression-test
  reference: after retrain, the same decisions should look qualitatively
  different (mix of opens for KJs/AT/pairs in late position).

---

## Verification checklist (before declaring retrain complete)

- [ ] Library fix landed: `pyspiel.load_game(...)` with the new game string
      shows correct per-seat stack deductions and pot inclusion of antes
- [ ] `python -m pytest tests/test_game_strings.py tests/test_stack_sampler.py`
      passes with updated assertions
- [ ] Retrained blueprint loads cleanly: `_load_solver(new_ckpt, ...)` works
- [ ] BTN range query: ≥30% open rate, opens include KK/AKs/AKo/KJs/KQs
- [ ] CO range query: ≥20% open rate
- [ ] `scripts/test_integration_midhand.py --jsonl data/live_1500.jsonl`
      pass rate ≥95% with the bridge code removed (strict chip-equality
      restored)
- [ ] Bake-off re-decision: new ship candidate identified, ICM win rate
      ≥ prior numbers (otherwise the retrain regressed)
- [ ] `docs/DECISIONS.md` updated: Phase 2 bridge entry marked RESOLVED;
      inflated_BB ghost-ante and heads-up SB/BB entries CLOSED (the
      original "FIX QUEUED" lines)
- [ ] `docs/STATUS.md` updated: Phase 2 marked complete; retrained model
      identified as ship candidate
- [ ] `docs/SESSION_LOG.md` appended with the retrain session block

---

## TL;DR for the bot-design program

1. **Patch OpenSpiel universal_poker** to accept `ante` parameter (or fall
   back to Path B / C if the patch is impractical)
2. **Edit ~6 production files** to use real ante instead of `inflated_big_blind`
3. **Retrain blueprint** on RunPod RTX 4090 (~3-5h, ~$5-15)
4. **Re-query BTN/CO range** with `scripts/query_model_preflop_range.py` —
   pass if BTN ≥30% open, CO ≥20%
5. **Reopen bake-off** to re-decide ship candidate
6. **Delete bridge code** when acceptance passes
7. **Keep folded-repair** (orthogonal scraper-side fix)

The current model is gated behind this retrain. No clicking until acceptance.
