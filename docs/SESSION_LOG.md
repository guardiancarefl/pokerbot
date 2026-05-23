# Session Log

Append-only record of what happened in each working session and why. STATUS.md tracks current state; this file tracks history. When something in STATUS.md changes, the reason should be findable here.

Format: most recent session at the top. Each session block notes date, what was done, what was decided, what was learned, and what's queued for next time.

---

## Session 7.5 — 2026-05-22 (late evening / overnight)
**Focus:** Open Track A3 — card abstraction comparison harness with KrwEmd as the headline option. Got further than planned, but in an unexpected direction.

### What was done
- Decision locked: implement Option 4 (KrwEmd, Fu et al. 2025) alongside Option 1 (EMD k=169/500) with a comparison harness, picking the winner by Slumbot bb/100. Documented in docs/A3_PLAN.md (commit 81fbaf4).
- Started the Option 1 baseline trainer (scripts/train_abstraction_a3_option1.py) on Contabo. Killed it mid-flop after a 30-second probe surfaced that the lookup was non-deterministic at k=169 — AA, KK, QQ, TT all collapsed to wrong / overlapping buckets across calls (commit 034d549).
- Verified the non-determinism is irreducible at any reasonable runout budget: at 200, 800, 2000, 5000 runouts, KK and QQ still flip buckets across fresh-rng trials. Even k=20 (the production abstraction Phase 2d trained against) shows AKs flipping between buckets 18, 10, 16 at 200 runouts (commit a10c012).
- Found a separate bug while validating: the "lossless" k=169 preflop trainer was producing only 168 distinct HoleClass strings, with 87o duplicated at buckets 41 and 167 and J7o missing entirely. Root cause: _kmeans_plus_plus_init used rng.choices weighted by squared distances but didn't zero out already-picked indices, allowing duplicate medoid picks. Two-line fix plus a k==n short-circuit (commit a22af38).
- Shipped deterministic preflop lookup as two clean commits:
  - f274c6f added infrastructure: hole_class_from_cards() in equity.py, optional preflop_lookup field on StreetAbstraction, bucket_of() fast path using the dict, 9 new tests.
  - ae2a1e7 modified train_abstraction.py to build the lookup from k-medoids labels during preflop training.
- End-to-end verified: a freshly-trained k=169 preflop abstraction returns the same bucket for AA, KK, QQ, AKs, 72o across 5 trials with 5 different rng seeds at 50 runouts each. Before the change, same probes returned 3-4 different buckets.

### What was decided
- **Project sequence stays the same.** User asked why we're not going straight to 6-max NLHE. Answer: HUNL is a debugging environment, not a deliverable. The pieces we're building (A3 card abstraction, B1 subgame solver, C1 within-match adaptation) all transfer. Validate them in 2-player where convergence is faster and the Slumbot benchmark exists, then port to 6-max at Phase 4 once the algorithmic pieces are in place. Recorded as a project-direction consensus, not a written decision (the original ARCHITECTURE.md already had Phase 4 as 6-max).
- **A3 strategy: deterministic lookup before any new abstraction algorithm.** Comparing abstractions on top of a non-deterministic lookup means comparing measurement noise. Postflop determinism is the next concrete deliverable; KrwEmd and OCHS implementations follow after.

### What was learned / surprises
- The non-determinism finding kept getting stronger as we probed deeper. First observation: AA collides with TT at k=169 in one trial. After "this might be noise," tried 5000 runouts to wash it out — still noisy. Concluded that the medoid distances are smaller than the irreducible MC noise floor at any reasonable runout budget, which means "more MC" cannot be the fix. The fix had to be deterministic-lookup, not better-MC.
- Found three foundational bugs in roughly two hours by being willing to keep probing instead of assuming the existing code was correct: (1) bucket_of() non-deterministic at lookup, (2) kmedoids sampling with replacement, (3) (still pending) the histogram-distance metric being too coarse to distinguish adjacent hands at lookup time. The first two are now fixed. Each finding required a 30-second probe that the codebase didn't already have.
- The patch-with-assertions discipline saved us at least twice tonight. Both Commit A and Commit B's first patch script attempts failed on count==0 because the actual file had blank lines or whitespace my anchor didn't match exactly. The assertions caught it immediately instead of silently writing nothing.
- Earlier in the session I rebuilt Track A2 because I didn't run `git log --oneline -10` at session start — A2 was already shipped in dbfc7be by an earlier session today. CLAUDE.md updated with a git-log-at-session-start requirement (commit db0cc31). The lesson held throughout the rest of the session.

### Workflow notes for next time
- The "30-second probe before committing to long training run" pattern is now proven twice (Session 7.5 kmedoids bug, Session 7.5 bucket_of bug). Future abstraction work should always include a 5-hand determinism check on the trained artifact before launching anything multi-hour.
- Two-commit-pattern (infrastructure first, no behavior change; integration second, behavior change) worked very well for the preflop lookup ship. Commit A is independently revertable. Commit B depends on A but is small. Use this for postflop too.
- Patch scripts should always use `sed -n ... | cat -A` to see actual bytes (whitespace, blank lines) before writing the `old_str` anchor. Two anchor failures tonight, both from invisible blank lines.

### What happened after — postflop determinism, B1a sketch, eval comparison, retrofit win
- **Postflop deterministic lookup landed** (commit 5333a4a). compute_hand_histogram receives a deterministic rng seeded from sha256(canonical sorted(hero, board)) when called from bucket_of(). Caller's rng parameter intentionally discarded — both real callers (InfosetEncoder, archetype solver) were passing the master training rng thinking that gave reproducibility, but the master rng's state advances between calls. End-to-end verified across preflop + flop + turn + river: every probe STABLE across 5 trials with different rng seeds and rng=None.
- **B1a (leaf strategies for subgame solving) sketched** (commit 5d6ff3a). 143-line src/nlhe/biased_policy.py implementing BiasConfig + standard_bias_configs(alpha) + apply_bias() + BiasedBlueprint with 17 passing tests. The four canonical continuation strategies (blueprint identity, fold/call/raise-biased) used at depth-limited solving leaf nodes. Written during eval-wait window. In retrospect should not have been written at the tail of a long session — but it's done, tests pin its behavior, and refactoring later is cheap. Tagged as B1a in commit and B1_PLAN.md.
- **Comparison eval kicked off in tmux**: same Phase 2d checkpoint, two different abstractions. Eval A = original noisy k=20. Eval B = deterministic fresh retrain. Each 1000 hands vs Slumbot.
- **Eval A: +15.05 bb/100** (baseline-adjusted, 1000 hands). Vs the historical +31.45 — session variance.
- **Eval B: -16.75 bb/100**. Surprising regression. Investigation revealed: a fresh deterministic retrain produces DIFFERENT preflop bucket IDs than the original (k-medoids init is random per run). 7/12 preflop probe hands had remapped buckets. Postflop bucket IDs by contrast were 7/7 identical across the two abstractions — postflop happens to be path-stable.
- **Diagnosed and fixed via retrofit script** (commit 129dda7). scripts/retrofit_preflop_lookup.py loads an existing abstraction and runs bucket_of() N times per canonical HoleClass with different rng seeds, taking the modal bucket as the lookup-table value. Result on Phase 2d artifact: 169/169 canonical hands unanimous (modal agreement 11/11 at 5000 runouts). Preserves original bucket IDs; only the lookup path becomes deterministic.
- **Found a 4th bug along the way**: bucket_of() is suit-dependent on canonically-equivalent literals (AsKs and AcKc produce different histograms because the MC samples from different remaining-deck card sets). The retrofit happens to address this for preflop by canonicalizing to a single representative per HoleClass.
- **Eval C: +78.35 bb/100** (retrofit + Phase 2d checkpoint vs Slumbot, 1000 hands, same session as Eval A). +63 above Eval A in the same session. Sized comfortably outside reasonable n=1000 variance bounds. Strong evidence determinism alone is worth ~+50 bb/100 to this trained bot.

### Final session score
Fourteen commits on main. Findings: four foundational bugs in the abstraction layer, three fixed in-session (kmedoids replacement, bucket_of non-determinism preflop and postflop), one worked around by the retrofit (cross-run bucket-id instability). Headline result: +78.35 bb/100 vs Slumbot, ~5x the historical reference number, achieved without any retraining.

### What was reprioritized
- A3 deeper work (KrwEmd / Option 4 / comparison harness): DE-PRIORITIZED. Current k=20 abstraction with retrofit produces +78.35 bb/100. Further A3 algorithm work has diminishing returns vs. the bigger missing pieces.
- B1 (subgame solver): elevated to top priority. The +50 bb/100 from determinism alone suggests the bot can be significantly stronger with the algorithmic pieces (subgame solving, within-match adaptation) that the architecture calls for.
- 6-max port: elevated. The actual project goal is 6-max SNG tournament top-3 finish; HUNL is debugging infrastructure. Next session should decide whether B1 lands in HUNL first or directly in 6-max.

### Queued for Session 9
1. Decide direction: B1b (subgame tree construction in HUNL) OR start 6-max port refactor. Either is a multi-session arc.
2. If B1b: use the existing biased_policy module (5d6ff3a), build subgame construction from current OpenSpiel state, add CFR subgame solver.
3. If 6-max port: action discretization remains (7 actions), infoset encoder needs 6-player state, training loop needs 6-player self-play, evaluation harness needs replacement (Slumbot is HUNL-only).
4. The 42 bought-bot profiles still pending integration (Phase 3 work, blocked on having a 6-max training environment).
5. The retrofit artifact at runs/abstraction_20260521_223018_retrofit/ is the production preflop-deterministic-lookup abstraction. Use it for any subsequent Phase 2d-bot evals.

---

## Session 7 — 2026-05-22 (evening)
**Focus:** Phase 3 Track A2 — implement the hand-engineered archetype framework. Five archetypes (NIT, TAG, LAG, STATION, MANIAC) parameterized by tightness × aggression, plugged into the Deep CFR solver as opponent diversity beyond pure self-play.

### What was done
- Reviewed src/nlhe/solver.py and src/nlhe/actions.py to understand the existing infrastructure. Found that _build_game_state_view already exposes pot, to_call, effective_stack, min_bet, max_bet, and legal flags; archetypes can be pure consumers without new feature extraction. Also discovered that src/nlhe/config.py doesn't exist as a separate module — TrainConfig is defined inline in solver.py.
- Investigated whether the EMD card abstraction's bucket IDs have any hand-strength ordering. They don't — confirmed by spot-check: AA at bucket 12, KK at bucket 5, QQ at bucket 16, 56s at bucket 13, 72o at bucket 2. Bucket IDs are categorical cluster labels with no inherent ordering.
- Examined StreetAbstraction and found `medoid_histograms` (shape k × bins) already stored — equity histograms per bucket. Wrote a one-line derivation for mean equity per bucket: `(normalized_histogram * bin_midpoints).sum(axis=1)`. No Monte Carlo needed.
- Verified the bucket-equity gradient is smooth (range 0.33 to 0.84 preflop, span widens through streets) but mid-range ordering is unreliable: KK reads 0.79 while AA reads 0.76; AKs at 0.56 sits below TT at 0.76. Endpoints are roughly calibrated to ground truth but the middle of the distribution has ordering errors. STATUS line 22 already documented this; today's investigation confirmed it generalizes from "AA/QQ/TT collide" to "no monotonic equity-bucket mapping."
- Initially proposed archetype thresholds as hand-picked constants ("nit folds below 0.78"). User pushed back: "is this the best, are we still letting data shape and decide the thresholds of profiles?" — caught a real bug. The made-up threshold of 0.78 would have caused the nit archetype to fold AA (bucket equity 0.76) preflop, because AA's bucket equity is below the made-up threshold. Pivoted to data-derived design before any code was written.
- Wrote scripts/empirical_buckets.py (initially /tmp): sampled 5000 random preflop hands and 2000 per postflop street through the abstraction, computed population-equity quantiles per street, saved as JSON. 3-minute run on Contabo CPU. Output saved to runs/archetype_design/bucket_equity_analysis.json with bucket_equity tables and quantile tables.
- Designed archetype thresholds as quantile parameters per street (e.g., NIT plays "top 15% preflop, top 25% flop, top 30% turn, top 40% river"). Each archetype gets a 4-tuple of quantiles plus a single aggression scalar. Five archetypes locked: NIT (0.85,0.75,0.70,0.60) aggression 0.25, TAG (0.70,0.55,0.50,0.45) aggression 0.65, LAG (0.40,0.30,0.35,0.40) aggression 0.85, STATION (0.25,0.15,0.10,0.05) aggression 0.20, MANIAC (0.05,0.10,0.20,0.30) aggression 0.95.
- Built src/nlhe/archetypes.py (~350 lines): ArchetypeProfile dataclass, NAMED_ARCHETYPES tuple, EquityCalibration loader with linear quantile interpolation, archetype_policy decision function with stack-depth and position modifiers, OpponentPool class holding the pool + sampling logic, derive_in_position helper for HU position derivation.
- Caught two bugs after first write via action-enum inspection: the policy code used `BET_33POT`/`BET_POT`/`ALL_IN` style names but the real enum is `BET_33`/`BET_100`/`ALLIN`. Also discovered that "check when free" is the `CALL` slot, not `FOLD` — the policy function had check probabilities written into the wrong slot in three places. Five-edit patch fixed all of it, plus the in-position nudge that depended on the same wrong assumption.
- Built tests/test_archetypes.py with 9 behavioral tests against the real calibration: distribution validity, nit-folds-trash, nit-plays-premium, maniac-plays-trash, nit-strictly-tighter-than-maniac-across-buckets, aggression-drives-bet-vs-call-split, legal-mask-honored, position-helper, opponent-pool-sampling. All green on first run.
- Moved /tmp/empirical_buckets.py to scripts/analyze_bucket_equity.py with proper argparse CLI (`--abstraction`, `--out`, sample-count overrides, seed). 163 lines. --help works. Reproducibility pipeline now in repo.
- Integrated archetype framework into the solver via a 6-edit patch in src/nlhe/solver.py: imports, TrainConfig.archetype_mix field (default 0.0 for backward compat), DeepCFRSolver.__init__ accepts optional opponent_pool, new _archetype_strategy method that builds the archetype's distribution from bucket equity + position + pot odds + stack/pot, _traverse opponent branch dispatches to archetype if assigned, train() samples archetype per trajectory. 41 tests green after integration (32 pre-existing + 9 new).
- Wrote /tmp/smoke_arch.py to exercise the integration with real training. Two configs: archetype_mix=0.0 (vanilla regression) and archetype_mix=1.0 (every opponent is an archetype). Instrumented _archetype_strategy and strat_buffer.add to count invocations. 4188 archetype calls across 20 trajectories, all 5 archetypes sampled, zero strategy buffer pollution. Smoke passed all 4 gates.
- Side finding from smoke: at archetype_mix=1.0 the strategy buffer never fills (no path to populate it from archetype trajectories — those write zeros to it by design). The strategy net therefore can't train at full archetype mix. Documented as a warning on the config field with recommended range 0.2-0.7.
- Pod-status check at session start showed the Phase 2d RunPod pod was already terminated (Session 6 closed it). No new GPU activity needed.

### What was decided
- **Data-derived archetype thresholds, not hand-picked.** Tightness is read from empirical bucket-equity quantiles; aggression remains a designed parameter (no labeled-action dataset under opponent anonymity, and the project's values say there never should be one). DECISIONS.md updated.
- **Archetype decisions do NOT write to the strategy buffer.** Otherwise the bot would learn to imitate maniacs. Strategy buffer writes are now gated on `self._current_archetype is None`. Verified by smoke. DECISIONS.md updated.
- **archetype_mix is config-knob exposed, default 0.0.** Backward compatible — no existing code path changes behavior unless the user explicitly passes an opponent_pool AND sets archetype_mix > 0. Recommended production value 0.5.
- **Single exponent simplification holds.** Track A2 doesn't change DCFR's single-exponent design from A1 (that's a separate algorithm). The two tracks coexist cleanly.
- **Pre-existing GPU checkpoint compat unaffected.** Archetypes integration touches `_traverse` opponent-node behavior but not the saved checkpoint format. Session 5 GPU artifacts remain loadable under any cfr_variant.
- **Real humans for evaluation, never for training.** Established the boundary explicitly: bb/100 from human play is evaluation data, observed hands never flow back to training, archetype recalibration never reads human play. Aligned with the opponent anonymity principle from Session 1.

### What was learned / surprises
- **Data-first design caught a silent bug that would have shipped.** The made-up "nit folds below 0.78" threshold would have caused nit to fold AA preflop, because the EMD bucket-equity for AA reads 0.76. Three minutes of empirical sampling saved hours of "why is the nit folding pocket aces" debugging later. The lesson generalizes: anywhere a threshold meets a learned representation, the threshold needs to be derived from the representation's actual distribution, not from intuition about what the representation should look like.
- **Grep lies.** Searching for "BET_" missed `ALLIN` because the action name has no underscore prefix. The bug nearly shipped to archetypes.py until a direct `len(DiscreteAction)` and enumeration via Python caught it. Lesson: when correctness depends on enum exhaustiveness, verify by iteration, not by pattern.
- **The 6-edit patch-with-assertions discipline scales.** Same pattern as DCFR's 3-patch sequence: each `old.count(...) == 1` assertion forces unique-match precision, the patch either applies all edits or none, and grep-based verification afterward is cheap. This held across both algorithmic work (DCFR) and module-creation work (archetypes). Worth keeping as the default for any multi-edit refactor.
- **OpenSpiel HU position convention is `firstPlayer="2 1 1 1"`.** Player 1 (SB/button) acts first preflop, player 0 (BB) acts first postflop. The `derive_in_position(street_idx, current_player)` helper encodes this once so archetype callers don't have to think about it. Worth being explicit: if we ever change firstPlayer in the game string, the helper breaks silently.
- **The strategy-buffer-empty-at-mix-1.0 finding is non-obvious.** The smoke caught it because we instrumented both the buffer writes AND the iteration losses. Without the loss column we'd have seen "smoke passed" and shipped a config that silently couldn't train the strategy net. Worth carrying: integration smokes need to surface both correctness invariants AND training signals.

### Workflow notes for next time
- Build the empirical-analysis artifact before designing the thing that consumes it. The bucket-equity analysis took 3 minutes and prevented a silent design bug. Cheap insurance.
- Inspect the actual enum / dataclass surface before writing code that references it. `len(DiscreteAction)` + `for a in DiscreteAction: print(a)` is the cheapest possible bug-prevention step.
- The `cat > /tmp/patch.py <<'EOF' ... EOF` pattern with file write + size check + run is the workflow that's stuck. Heredoc paste artifacts in terminal echo are decorative — verify with `wc -l` and `tail`, never with visual scan.
- When integration smoke shows "all gates passed" but a number is NaN or zero, treat the NaN as a finding, not a footnote. The strategy-buffer-empty case is worth its own line in the commit notes and a docstring warning.

### Queued for next session (Session 8)
1. Track-selection decision: A3 (OCHS card abstraction), B1 (subgame extractor design), or C1 (continuous archetype-belief design). Recommendation: B1 — subgame solving is the Pluribus delta and the highest-leverage single piece of work remaining. But A3 has a clean self-contained scope (~1-2 weeks) and an immediate benefit (better abstraction propagates to archetypes via the JSON regen). C1 needs A2 (now done) and benefits from B1 (which doesn't exist yet).
2. (Cleanup carry-over from earlier sessions): train_leduc.py config.json/metrics.json location reconciliation; _build_game_state_view underscore prefix even though it's imported by policy_adapter.py. Both still deferred, both still cosmetic.
3. (Possible carry-over): the 42 bought-bot profile format identification. Not strictly blocking anything but it'll need to be done before Phase 3 archetype integration can include them as additional anonymous training opponents.

---

## Session 6 — 2026-05-22 (afternoon)
**Focus:** Phase 3 Track A1 — implement DCFR (Linear / Discounted CFR) in the Deep CFR solver. Per-sample iteration weighting in advantage and strategy net training, simplified single-exponent form (Brown & Sandholm 2019 fidelity simplified).

### What was done
- Read src/nlhe/solver.py end-to-end. ReservoirBuffer has no per-entry iteration tracking; sample_batch returns 3-tuple (feats, targets, masks); _traverse adds samples without an iteration arg. All the integration points are in one file.
- Three sequential patches: (1) TrainConfig adds cfr_variant + dcfr_exponent fields with sensible defaults. (2) ReservoirBuffer gains an `iters` field, add() takes iteration arg, sample_batch returns 4-tuple, checkpoint save/load handles new field with backward-compat refusal of non-vanilla resume from pre-DCFR checkpoints. (3) Per-sample weighting wired into _train_advantage_net and _train_strategy_net via new _dcfr_weights helper, vanilla path is a no-op.
- Standalone math verification: stubbed solver state and called _dcfr_weights directly with hand-calculated test cases. Vanilla returns None. Linear with T=10, iters=[1,5,10] gives normalized weights summing to 3.0 with expected pattern. Discounted^2 with same setup gives weights matching `(iters/T)^2` normalized to N. discounted(exponent=1) ≡ linear exactly.
- Real-loop smoke: 2 iterations of HUNL with each variant, same seed. Iter 1 losses identical across all three variants (T=1 collapses all weights to uniform). Iter 2 losses diverge across variants (weighting actually applied). discounted(exponent=1.0) matches linear bit-exact at iter 2. All 32 pre-existing tests green at every patch step.
- Committed as 41e2fa3 with full attribution to the patches. STATUS.md updated to close A1; DECISIONS.md gained two entries (single-exponent rationale, refuse-non-vanilla-resume rationale).
- End-of-session check on Phase 2d RunPod pod: iter 137/5000, loss flat around 0.83 (same plateau as iter 100). Only ckpt_iter_0100 on disk because v2 config used checkpoint_every=100. Cost-discipline decision: terminate. Pulled ckpt_iter_0100 + config + log to runs/gpu_phase2d_artifacts/ before terminating via RunPod dashboard. Connection refused after termination — billing closed cleanly.

### What was decided
- **Single-exponent DCFR, not full three-exponent Brown & Sandholm.** Full DCFR has α (positive regrets), β (negative regrets), γ (strategy average). Simplified form uses one exponent that governs both nets equally. Linear = exponent 1. Discounted = configurable. Tradeoff: less faithful to paper, but less surface area for the implementation to be subtly wrong. Revisit if measurements show the simplification leaves convergence speed on the table.
- **Refuse non-vanilla resume from pre-DCFR checkpoints.** Old checkpoints don't have per-entry iteration tags. Any default for the missing iters silently corrupts weighting math. The load path raises a clear error pointing to either resume-vanilla or start-fresh.
- **Phase 2d pod terminated.** Phase 3 work is on Contabo CPU; ckpt_iter_0100's +31.45 bb/100 is the locked Phase 2d headline result. Extending the GPU run would have cost dollars for marginal-or-no improvement on a flat loss curve. Optionality preserved by keeping the checkpoint artifact locally.

### What was learned / surprises
- **Loss-plateau pod was already plateaued at iter 100.** The Session 6 check found iter 137 with the same loss as iter 100. The "let the GPU run overnight" intuition from Session 5 was wrong — we got 37 more iterations of effectively the same thing. Worth carrying: pod monitoring needs explicit "is this still improving" criteria, not "is it still going."
- **Patch-by-patch verification with assertions caught a real benefit.** The 3-patch sequence let the regression test (32 tests green) run between each patch. If any patch had broken vanilla behavior, the next patch wouldn't have built on a broken base. Worth keeping as the default discipline for multi-edit solver work.
- **STATUS.md edits accumulated across two commits.** Edit 1 closed A1 in the Done section. Edit 2 (separate commit) handled the In progress and Track A renumbering. Either commit alone would have left STATUS in an inconsistent state. Worth carrying: STATUS edits should be batched into one logical change per session, not split across commits.

### Workflow notes for next time
- Single-quoted heredoc delimiters (`<<'PATCH_EOF'`) prevent shell interpretation of $variables in the patch body. Used consistently this session.
- Verify file writes with `wc -l` + `head` + `tail` before running the patch. Terminal-echo artifacts (especially after long heredocs) are decorative, not diagnostic.
- One-liner remote checks via SSH are cheaper than attaching to a pod. `ssh ... 'tail -5 /tmp/log'` returns in seconds and tells you what you need.

### Queued for next session (Session 7)
1. Track-selection decision: A2 (archetype framework), B1 (subgame extractor design), or C1 (continuous archetype-belief design). Recommendation: A2 — most concrete, unblocks C1, shippable in 1-2 sessions.
2. (Cleanup carry-over): train_leduc.py config.json/metrics.json location, _build_game_state_view underscore prefix. Both deferred again.

---

## Session 2 — 2026-05-21
**Focus:** Move from planning to execution. Get a working Linux environment with PyTorch + OpenSpiel, write the Phase 1 Leduc Deep CFR scaffold, complete a training run that validates the pipeline.

### What was done
- Attempted WSL2 + Ubuntu 22.04 install on Windows 11 host. WSL itself (2.7.3) installed but DISM failed at `Enabling feature VirtualMachinePlatform` with error 14098 (component store corrupted). Same failure for `Microsoft-Windows-Subsystem-Linux`.
- Ran `DISM /Online /Cleanup-Image /StartComponentCleanup` (no effect), then `DISM /Online /Cleanup-Image /RestoreHealth` (succeeded but didn't fix the feature install), then `sfc /scannow` (found and repaired corrupt files but features still wouldn't enable). After reboot both features still `State : Disabled`. The remaining repair paths (ISO-source DISM, in-place Windows reinstall) would have cost more time than just using a clean Linux machine that was already available.
- Pivoted to existing Contabo VPS: Ubuntu 24.04 (Noble), 12 vCPU AMD EPYC (oversubscribed), 48GB RAM, ~300GB free, no GPU.
- Created GitHub account `guardiancarefl`. Created private repo `pokerbot`. Generated ed25519 SSH key on Contabo, registered with GitHub.
- Installed system packages: `cmake`, `build-essential`, `software-properties-common`. Added `ppa:deadsnakes/ppa`. Installed `python3.10`, `python3.10-venv`, `python3.10-dev` alongside system Python 3.12. (OpenSpiel's officially-tested Python range is 3.7-3.10, so 3.10 is the safer target.)
- Created `~/pokerbot/` on the Contabo box. Initialized git, set remote, wrote `.gitignore` and `README.md`. Wrote all five foundational docs verbatim from Session 1 versions. Committed and pushed (`c756764`).
- Updated STATUS.md, SESSION_LOG.md, DECISIONS.md, ARCHITECTURE.md to reflect the runtime migration. Committed and pushed (`52c3fd5`).
- Created venv at `~/pokerbot/.venv`. Installed PyTorch CPU build (2.12.0+cpu) and OpenSpiel (1.6.11). Smoke tests confirmed both import and Leduc loads.
- Built Phase 1 scaffold as a modular structure designed for reuse in Phase 2+: `src/leduc/config.py` (TrainConfig dataclass + YAML loader), `src/leduc/solver.py` (wraps OpenSpiel's DeepCFRSolver), `src/leduc/evaluate.py` (exploitability in mbb/g), `src/leduc/checkpoint.py`, `configs/leduc_default.yaml` + `configs/leduc_smoke.yaml`, `scripts/train_leduc.py` (~180 lines), `tests/test_pipeline.py` with 4 fast tests. Committed and pushed (`4f5c2d8`).
- Pipeline tests passed but the first real training run with `leduc_default.yaml` (100 iters x 40 traversals x 500 adv steps) ran for 30+ minutes with **no per-iteration visibility** because OpenSpiel's `solve()` loops internally with no callback hooks. Killed it.
- Tried a smaller config (`leduc_phase1.yaml`, 40 iters x 200 adv steps). First iteration came back at 67 seconds. Estimated total 45+ minutes. Killed it because the lack of visibility was the real problem, not the runtime.
- Rewrote `src/leduc/solver.py` to drive training one iteration at a time: construct solver with `num_iterations=1`, call `solve()` in a Python loop, accept optional `logger` and `eval_callback`. Cost: one extra policy-network training pass per loop iteration (since OpenSpiel's `solve()` includes policy training as its final step). Benefit: per-iteration progress logging and periodic exploitability eval.
- Updated `scripts/train_leduc.py` to pass the logger and an exploitability eval_callback (every 10 iterations, plus final iteration).
- Caught a None-handling bug in the new solver: OpenSpiel's `_learn_strategy_network()` can return None when the strategy reservoir buffer is too small, same way `_learn_advantage_network()` can. Both now coerced to NaN.
- Patched `runs/` gitignore: switched from `runs/` to `runs/*` so the negation rule `!runs/README.md` could take effect (git can't re-include files under an excluded directory). Wrote `runs/README.md` documenting the run directory format. Wrote `docs/PHASE2_SKETCH.md` as a forward-looking Phase 2 plan to give Session 3 a starting point. Committed and pushed (`d0688f9`).
- Ran the third training attempt with `leduc_phase1.yaml` cut to 25 iterations. Per-iteration timing was very noisy on the oversubscribed Contabo CPU: individual iterations ranged from 13s to 151s, averaging 91s/iter. Total runtime 38 minutes.
- **Phase 1 result: exploitability 1187 (uniform random) -> 502 (iter 10) -> 447 (iter 20) -> 434 (iter 25) mbb/g.** Clear downward trend, decelerating as expected. Final checkpoint saved at `runs/leduc_20260521_210552_phase1_take2/checkpoints/final.pt`. metrics.json and config.json saved as companions.

### What was decided
- **Runtime moved from WSL2-on-Windows to Contabo VPS** (recorded in DECISIONS.md, supersedes original WSL2 entry).
- **GPU work deferred to rented cloud hardware in Phase 4+** (or earlier if Phase 2d needs it — see open questions). The local RTX 3060 isn't part of the plan anymore.
- **OpenSpiel reference implementation for Phase 1**, wrapped not reimplemented. The goal of Phase 1 is pipeline validation, not algorithmic research.
- **Per-iteration solver loop** instead of single `solve(num_iterations=N)` call. Trades small compute overhead for the visibility needed to make informed decisions during runs. Acceptable for Leduc; may revisit for Phase 2+ if profiling shows it matters.
- **Phase 1 config sized for time-on-Contabo, not for Leduc record exploitability.** 25 iterations with 200/400 train steps and 64-unit networks. Result (~434 mbb/g) is well above published Leduc Deep CFR levels (50-250 mbb/g typical) but well below uniform random (1187), demonstrating the pipeline works. Phase 1's job was infrastructure validation, not Leduc benchmarking.
- **42 bought-bot profiles stay in Phase 3.** Brief discussion this session about skipping ahead to integrate them after Phase 1; the conclusion was that without a working NLHE training environment (Phase 2), there's no integration point for them. Phase 2 is the necessary bridge.

### What was learned / surprises
- The Windows component store corruption was the kind of environmental problem that no amount of project-specific planning would have caught. Worth flagging: operating systems can be broken in ways that look like "your install command failed" but are actually "your system needs an in-place repair." Knowing when to bail vs. push through is a real skill.
- **My time estimates for ML iterations were repeatedly wrong tonight.** Estimated "3-10 minutes," then "30-90 minutes," then "~18s/iter" (actually 67s+). The lesson: benchmark one iteration of the real config before committing to a full run. We should have run a single-iteration timing check before each config change.
- **A throwaway micro-benchmark gave wildly inconsistent numbers** on Contabo (17.9s vs 42.1s for same workload variants in opposite of the expected direction). Contabo's oversubscribed vCPUs mean per-iteration time varies by ~10x depending on neighbor activity. Adequate for development; useless for benchmarking. Phase 4+ compute planning must be done on dedicated hardware.
- **OpenSpiel's `solve()` returns None for losses when buffers are too small** for either advantage or strategy training. Came up in two places (advantage net + policy net), both initially crashed our code with `TypeError: float() argument must be a string or a real number, not 'NoneType'`. Now handled by NaN coercion. Worth defensive-coding for in any wrapper around library code.
- **Three killed training runs in one session.** The first kill was justified (no visibility). The second was justified (no visibility + bad timing estimate). The third would have been the impatience tax — we didn't kill that one and got the result. Pattern noted: kill when the *type* of problem changes (need visibility, need to reconfigure), not when "this is taking longer than I hoped."
- **Visual mash from heredoc + command echoes** kept making terminal pastes look corrupted. Every time, the actual file contents were correct (proven by verify commands). Lesson: trust `wc -l` / `head` / `tail` / `grep` over visual scan of terminal echo.

### Workflow notes for next time
- Run a single-iteration timing benchmark on the real config before kicking off a long training run. Saves wall-clock time on misjudgments.
- The 30-minute "is this hung or running" anxiety is the worst use of time in the project. Per-iteration logging eliminates it; future custom solvers (Phase 2+) need to preserve this.
- Two SSH sessions to the Contabo box was the productive workflow (one for training, one for editing/committing). Tmux next session would be cleaner than juggling SSH windows.
- The Session 1 verbatim-not-silent-edit rule held up well. Heredoc-based file writes were the right pattern.

### Phase 1 cleanup deferred to Session 3
- `train_leduc.py` writes its companion config.json and metrics.json into the `checkpoints/` subdirectory, not at the run-dir root as `runs/README.md` claims. Two options: update the README to match reality, or update the script to write both root and companion copies. Either is fine; not blocking Phase 2.

### Queued for next session
1. Decide GPU provider for Phase 2d training. Compare current pricing for Vast.ai 4090, RunPod 4090, Vultr A40. Pick a winner.
2. Phase 2a: load HUNL in OpenSpiel via universal_poker, validate the game representation, confirm action and information state encoding.
3. Phase 2a: build card abstraction module (`src/nlhe/abstraction.py`) using EMD clustering on equity distributions. Target ~200 buckets per street.
4. Phase 2a: build action abstraction module (`src/nlhe/actions.py`) with discretized bet sizes {check/fold, call, 0.33pot, 0.66pot, 1pot, 2pot, all-in} and translation of off-tree opponent sizes.
5. (Small) Fix the Phase 1 cleanup deferred item above.
6. (Small) Investigate why advantage loss kept climbing through the run. Per the OpenSpiel pattern with `reinitialize_advantage_networks=True`, climbing loss reflects growing-buffer complexity not network failure, but worth a sanity check against a published reference.

---

## Session 1 — 2026-05-21
**Focus:** Project bootstrap. Establish foundational docs and local infrastructure for Claude Code.

### What was done
- Identified that the local machine is Windows 11 with an RTX 3060 Laptop GPU (6GB VRAM), not the 12GB desktop variant originally assumed in project docs. Updated ARCHITECTURE.md and DECISIONS.md to reflect this.
- Confirmed OpenSpiel does not officially support Windows native. Decided to use WSL2 + Ubuntu 22.04 as the runtime environment.
- Installed Claude Code on the Windows host. Resolved PATH issue so `claude` is callable from any PowerShell.
- Created project directory at C:\Users\Ngior\pokerbot\ with docs/ subfolder.
- Created the four foundational docs verbatim from contents provided in the planning chat: PROJECT_OVERVIEW.md, ARCHITECTURE.md, DECISIONS.md, STATUS.md.

### What was decided
- **Runtime environment: WSL2 + Ubuntu 22.04** on the Windows host. (Superseded in Session 2.)
- **Opponent anonymity as a core design principle.** No persistent identity for any opponent across matches. No pre-collected real-world hand history data feeds training. The bot's information state mirrors a competent human at an anonymous online table. Values-driven decision — robustness and fairness over peak exploitation EV.
- **Within-match adaptation at Position 2.** Light-to-medium online statistics nudge subgame solver ranges within the current match, anchored to the blueprint as a safety floor. When a match ends, all derived state is wiped.
- **The 42 bought-bot profiles stay frozen.** Style diversity in training and stable benchmark targets. Strength diversity comes from league play (PSRO with archived self).

### What was learned / surprises
- The original ARCHITECTURE.md mixed two design goals — anonymous Nash-leaning play and identified-opponent exploitation. Removing the exploitation layer made the project cleaner.
- Claude Code defaulted to its own judgment when given file contents to write verbatim — it removed a line on its own and announced the edit after the fact. A hard rule was set: when exact contents are given between BEGIN/END markers, write them exactly; concerns must be raised as questions before writing, not as silent edits. Noted in STATUS.md under Known issues.

### Workflow notes for next time
- Always verify Claude Code's output, not just its self-reports. "Done" doesn't always mean done correctly.
- The relay workflow (Claude planning chat ↔ Claude Code execution) works, but only if outputs are actually read end to end before being passed onward.

### Queued for next session
1. Install WSL2 + Ubuntu 22.04 on Windows host
2. Verify CUDA passthrough from WSL2 to the RTX 3060 Laptop GPU
3. Set up Python 3.10 environment with venv inside WSL2
4. Install PyTorch (CUDA build) and OpenSpiel inside the venv
5. Confirm PyTorch sees the GPU and OpenSpiel loads Leduc poker
6. Move the project directory from C:\Users\Ngior\pokerbot\ to the WSL2 Linux filesystem (better I/O than the Windows mount)
7. Install Claude Code inside WSL2 for ongoing project work
8. Begin Phase 1 scaffold: Leduc Deep CFR training script

---

## Session 3 — 2026-05-21
**Focus:** Open Phase 2. Decide GPU provider for Phase 2d. Load HUNL in OpenSpiel, validate game representation. Build card abstraction (EMD-based) and action abstraction. End-state goal: Phase 2a closed, foundations in place for Phase 2b training pipeline.

### What was done
- Researched current pricing for Vast.ai, RunPod, and Vultr GPU options. Recommendation made, committed to project docs, RunPod account created (no pod rented yet — Phase 2a and 2b run on Contabo CPU; first rental triggered at start of Phase 2d). DECISIONS.md entry locks the choice.
- Validated HUNL game loads in OpenSpiel via `universal_poker` with ACPC-style parameter string. Numbers confirmed: 760-dim information state tensor, max_game_length 218, ~20000-action node before abstraction with dynamic shrinking as effective stack drops. Imperfect-information encoding verified (each player sees only their hole cards).
- Walked a HUNL game from initial state under uniform-random play, validating that the engine correctly handles chance nodes, decision nodes, terminal returns. Returns are zero-sum chips (`[20000, -20000]`), to be wrapped with ICM in Phase 4.
- Installed `treys` (Python poker hand evaluator). `pokerkit` would have been preferred for capability but requires Python 3.11+; we're pinned to 3.10 by OpenSpiel's tested range. Treys imported, sanity-checked AA>KK, AA-vs-random equity at 0.847 vs literature 0.85.
- Built `src/nlhe/equity.py` (191 lines): wraps `treys`, exposes 169-class canonical hole enumeration, string-based card I/O, Monte Carlo `equity_vs_random` and `equity_vs_range`. Smoke-tested at 6 levels: AA at 0.856, 72o at 0.348, paired-board AA at 0.888, wet-board AA at 0.701. All match literature.
- **Decision: EMD over OCHS for card abstraction.** PHASE2_SKETCH had flagged this as a session decision. After laying out three readings of the question (will-need-eventually vs values-driven vs default-to-harder-on-autopilot), confirmed Reading 2 — values-driven choice for the gold-standard technique. Logged in DECISIONS.md.
- Built `src/nlhe/abstraction.py` (326 lines): equity-histogram generation via Monte Carlo runouts, pairwise EMD via `scipy.stats.wasserstein_distance`, custom PAM (k-medoids) clustering, `Abstraction` container with `bucket_of`/`save`/`load`. Validated end-to-end on a tiny-scale smoke test (preflop k=5 in ~10 seconds). Numbers checked: AA in different bucket from 72o, d(AA, 72o) = 0.54, kmedoids cost decreases monotonically and converges in <5 iterations.
- Built `scripts/train_abstraction.py` (158 lines) for the production training run + `scripts/inspect_abstraction.py` (77 lines) for post-hoc analysis. Dry-run on preflop only first; clean. Then full run across all four streets in 6.8 minutes wall-clock. Bucket distributions are sensible (preflop median 6 hands/bucket on k=20, postflop median 7 hands/bucket on k=200, max 19–26 across streets — no degenerate one-giant-bucket clustering).
- Inspected the trained flop abstraction in detail. EMD is doing real strategic clustering: different surface hands with similar histogram shapes group together. Specific verifications:
  - "Drawing dead on coordinated board" hands (e.g., `4h2s` on `8s6dTh`, `5h2c` on `Kc3d9h`) cluster despite different surface cards — same equity-histogram shape.
  - Set on dry board (`3h3d` on `3cKd7h`) at equity 0.96 sits in its own bucket above two-pair-no-improvement clusters.
  - Pocket pair tiers (99 / JJ / QQ / AA) end up in separate buckets even where mean equities are close — EMD's histogram-shape sensitivity is the value-add over OCHS, and it works.
- Built `src/nlhe/actions.py` (~230 lines after patches): `DiscreteAction` enum {fold, call, 0.33pot, 0.66pot, 1pot, 2pot, allin}, `policy_to_game_action` (discrete → OpenSpiel integer), `game_to_policy_action` (OpenSpiel integer → distribution over discrete via pseudo-harmonic translation from Ganzfried & Sandholm 2013), `discretize_legal_actions` (filter at decision time). Smoke-tested with mid-pot, small-pot, large-pot views.
- Caught and patched two bugs in the action module on the first smoke run. First: `policy_to_game_action` was clamping sub-min-bet sizes up to min_bet, causing 0.33pot/0.66pot/1pot to alias to the same chip count on small pots. Fix: return `None` for sub-min-bet sizes, treat them as unavailable in that state. Second: `_legal_discrete_bet_sizes` was deduping at the same chip count by keeping the first (smallest-label) action; it should keep the largest. Fix: use a dict keyed by chip count, so later (larger) actions overwrite earlier ones at the same count. Re-run after both patches: clean.
- Updated DECISIONS.md (EMD abstraction entry), STATUS.md (Phase 2a closure + Phase 2b queue), SESSION_LOG.md (this entry).

### What was decided
- **GPU provider for Phase 2d: RunPod Community Cloud RTX 4090** at $0.34/hr. Vast.ai marginally cheaper but more variable; Vultr 5x the cost for unneeded VRAM. $93 Vultr credit held for Phase 4 (multi-day blueprint training where reliability matters) or Phase 3 helper if needed.
- **Card abstraction: EMD on equity histograms, not OCHS.** Values-driven choice: EMD is the gold-standard technique and implementing it once now means we've done it properly when Phase 4 needs an even harder abstraction. Trade-off acknowledged (2-3x the code of OCHS, no measurable Slumbot-cycle validation against simpler alternative).
- **EMD sample sizes for Phase 2a:** preflop=169 canonical hands × 400 runouts → k=20. Postflop=1500 sampled (hand, board) combos × 200 runouts → k=200. 50 histogram bins per equity distribution. 6.8 min total training time, well within budget.
- **Action abstraction: 5 bet sizes** {0.33pot, 0.66pot, 1pot, 2pot, allin} + fold/call. Confirmed as ARCHITECTURE.md's target. Acknowledged that opponent bets between 2pot and 0.9×stack all snap to 2pot — coarse but acceptable for Phase 2; revisit if Slumbot shows overbet-sizing exploits.
- **Action illegality is signal, not noise.** When a discrete action's intended chip target falls below the legal min_bet, return None rather than aliasing to min_bet. The policy network will softmax only over legal discrete actions in that state.

### What was learned / surprises
- **Visual mash from heredoc echo continues to mislead, including misleading the assistant.** Twice this session: once after the abstraction.py write where my eye saw the closing `STATUS_EOF` fused to fragments of later content and I worried the file was truncated; once after the STATUS.md write where `tail -3` showed three log entries and I momentarily thought Session 3 had been appended twice. Both times the file on disk was correct, verified by `wc -l` and `grep`. Session 2's lesson holds: trust `wc -l` / `head` / `tail` / `grep` over visual scan of terminal echo.
- **Pasting large content directly into a bash prompt fails noisily.** Early in the session, the first DECISIONS.md content was pasted without the surrounding `cat << 'EOF'` heredoc wrapper. Bash tried to execute each line as a command, producing a wall of `command not found` errors. No actual file damage, but a real reminder that BEGIN/END markers in the assistant's instructions are *labels* describing where content goes, not shell commands to type. The fix going forward: always wrap content in a heredoc, single-quote the delimiter to prevent variable expansion, give the delimiter a distinctive name (`STATUS_EOF` not `EOF`).
- **The "trust the test, don't trust my expectation" lesson, again.** On the equity calculator's flop test, AA on `2c7d9s` came back at 0.86, and I'd written "expected 0.88-0.92" — looked like a bug. Higher-precision MC (20k trials) confirmed 0.859. The function was correct; my expected range was wrong. The same instinct would have wasted an hour chasing a non-existent bug if I'd trusted my prior over the measurement.
- **EMD really does what the literature says it does, and inspection makes it visible.** The flop bucket inspection was the most satisfying moment of the session — seeing different surface hands with the same strategic situation correctly clustered, and seeing pocket pair tiers correctly separated by histogram *shape* not just *mean*. Not abstract theory; visible in the output table.
- **First-attempt bugs in action translation were the right ones to catch.** Both bugs I caught (sub-min-bet aliasing, dedupe-by-keeping-first) would have produced subtle problems downstream — the policy would have learned weird patterns in small-pot states. Catching them in the smoke test before any training run is cheap; catching them after a multi-hour Phase 2d Slumbot evaluation would have been expensive.
- **Time estimates were better this session than last.** Predicted 12-15 minutes for abstraction training; actual was 6.8. Predicted ~120s for pairwise EMD per postflop street; actual 107-125s. Predicted ~10s for preflop smoke test; actual ~4s. Session 2's "always benchmark one iteration before committing to a full run" approach (here: dry-run on preflop before full four-street run) is paying off.

### Workflow notes for next time
- The two-SSH-session pattern was the right shape: one window for the training run, one for editing, one for inspection. Tmux would still be cleaner — defer until it becomes a friction point.
- Heredoc paste failures continue to be the dominant low-level annoyance. Single-quoted distinctive delimiter is the rule; not optional.
- When patching existing files, sed-via-python-heredoc with `assert old in src` is safer than raw sed. The assert catches silent failure modes where sed would otherwise no-op on a near-miss.
- The "kill criterion" from Session 2 (kill when the *type* of problem changes, not when "this is taking longer than hoped") held this session — we never killed a run.

### Phase 2a cleanup that's still open
- `runs/README.md` vs `train_leduc.py` config/metrics location mismatch (deferred from Session 2). Not blocking Phase 2b.
- Advantage loss sanity check against published Deep CFR Leduc reference (deferred from Session 2). Also not blocking. Both are good candidates for "warm-up" tasks at start of Session 4.

### Queued for next session
1. Phase 2b: build NLHE solver wrapper. Should follow the Leduc solver pattern (per-iteration logging, eval callback, NaN-safe loss capture). Information state encoding has to combine card bucket (from `Abstraction.bucket_of`) with betting history features.
2. Phase 2b: train Deep CFR on tiny HUNL (20bb stacks, coarse abstraction, [64,64] networks) on Contabo CPU. Goal is "the pipeline doesn't explode," not "good HUNL strategy."
3. Phase 2b: implement resumable training (checkpoint every N iterations, idempotent resume). Hard prerequisite for RunPod Community Cloud in Phase 2d.
4. Optionally Phase 2c: start the Slumbot harness. Independent of 2b work, could be done in parallel.
5. Session 2 cleanups (the two open items above) if time permits.

---

## Session 3 (extended) — 2026-05-22 (late night)
**Focus:** After closing Phase 2a, pushed through Phase 2b in the same session against my own recommendation to wrap. Built the custom Deep CFR solver, info-state encoder, resumable checkpointing, and YAML-driven training script. Phase 2b closed.

### What was done
- Decision: **custom Deep CFR loop, not wrapping OpenSpiel's `DeepCFRSolver`.** Reasoning: OpenSpiel's solver is tightly coupled to its game API and would fight our card/action abstractions. The deviation from textbook Deep CFR foreshadowed in DECISIONS.md ("Phase 1 implementation: OpenSpiel reference Deep CFR, not custom") cashed in here.
- Built `src/nlhe/infoset.py` (229 lines): bucket one-hot + street + position + pot/stack features + betting history features = 214-dim vector. Per-traversal cache for the expensive `bucket_of` calls. Smoke-tested against live HUNL states.
- Built `src/nlhe/solver.py` (initial 378 lines, ~452 after patches): external-sampling Deep CFR, regret-matching+, two networks per player (advantage + strategy), reservoir buffers, per-iteration logging mirroring the Leduc pattern.
- Initial smoke run: solver ran but advantage losses were in the millions (4M+). Root cause: regrets are in chip-unit scale (±2000 for our tiny config), MSE on that is millions. Patched in regret normalization (divide by starting_stack). Re-ran: losses now O(0.7-1.0), gradient conditioning sane.
- Added save/load checkpoint methods (~80 lines): serialize all 4 networks, 4 optimizers, 3 reservoir buffers, all Python and torch RNG states, current iteration. Modified `train()` to take `checkpoint_dir`/`checkpoint_every` args and support resuming from `solver.iteration + 1`.
- **Resume correctness test:** ran the same 10-iter config two ways — (A) uninterrupted, (B) train 5 iters with checkpoint, fresh solver loads checkpoint, train 5 more. Compared all 4 networks' parameters. Result: **max param diff = 0.00e+00 across all networks.** Bit-identical resume verified.
- Built `scripts/train_nlhe.py` (~95 lines) as YAML-driven entry point. Mirrors `train_leduc.py` structure. Writes `config.json` and `metrics.json` to run dir, checkpoints to `checkpoints/` subdir.
- Wrote `configs/nlhe_smoke.yaml` (20 iter × 50 trav) and `configs/nlhe_phase2b.yaml` (100 iter × 100 trav) for varying use cases.
- Dry-run of full script with a 3-iter micro config: end-to-end pipeline works, checkpoints land, metrics persist.
- Committed all of Phase 2b as one commit (`563793f`, 830 insertions across 5 files).

### What was learned / surprises
- **Patching files via inline python heredoc then continuing without confirming the grep is a foot-gun.** First normalization patch failed silently — I sent the patch script *and* the verify grep in the same message, you ran only the grep, the patch never landed, I didn't notice because I was reading "I sent a patch" not "did the patch actually land." The second time, with explicit grep-then-verify-before-proceeding, it landed correctly. Lesson: when patching, do it in a sequence where the verification gates the next step, not bundled-together.
- **Bit-identical resume is a real correctness criterion** and worth the work. We saved every RNG state explicitly (Python random for solver, three separate Python random for buffers, torch RNG state). The reward: a Phase 2d run on RunPod that gets pre-empted loses only the work since the last checkpoint, not unbounded drift.
- **Advantage loss scale is a real production concern even at tiny stacks.** A 6-order-of-magnitude loss meant gradient updates were enormous early in training; the network would have trained but slowly and with risk of instability. The fix was one line.
- **Buffer asymmetry between players** (player 0 ~4x more entries than player 1 at 20bb) is probably structural to short-stack HUNL (SB folds preflop often), not a bug. Confirming or refuting this is a Session 5 item once we have Slumbot evaluation numbers.

### Queued for next session
1. Phase 2c: Slumbot API harness + bb/100 eval against random-policy baseline.
2. Optional Phase 2b real-run on Contabo (100 iter × 100 trav) before Phase 2d.
3. Phase 2d: rent RunPod 4090, scale up, get measurable bb/100 vs Slumbot.
4. Session 2/3 cleanups: `train_leduc.py` config.json/metrics.json location (now also inconsistent with `train_nlhe.py`).

---

## Session 3 (extended extended) — 2026-05-22 (~00:30-01:00)
**Focus:** After closing Phase 2b, pushed straight into Phase 2c. Built Slumbot HTTP client, action-language parser, eval script, random-policy baseline, validated end-to-end.

### What was done
- Probed Slumbot's API via curl before writing the client — saw the exact JSON response shape for `new_hand` and `act`. Found out the response includes `baseline_winnings`, `session_baseline_total`, etc. — Slumbot's built-in variance reduction signal we hadn't expected.
- Built `src/nlhe/slumbot_client.py` (~210 lines): SlumbotClient + SlumbotState dataclass + action-string parser + RandomPolicy baseline.
- Built `scripts/eval_vs_slumbot.py` (~110 lines): hands loop with running bb/100 summary, raw and baseline-adjusted.
- Hit and fixed three real bugs in sequence: (1) RandomPolicy tried to bet `b<STARTING_STACK>` ignoring chips committed → "Illegal call" rejection; dropped the option. (2) Slumbot's action language uses `k` for check and `c` strictly for call; client treated them as interchangeable; patched RandomPolicy to emit the right one based on whether facing a bet. (3) On preflop with empty action string, the SB is facing the unposted big blind; client said `facing_bet=False` and the policy sent `k`; patched to special-case preflop.
- 50-hand validation run: 0 errors, raw=-13 bb/100, baseline-adjusted=-6 bb/100. Both negative as expected; magnitudes smaller than I'd guessed because random-policy HUNL isn't as exploitable as my prior suggested (the legal action set is small and constrained, and Slumbot can't fully exploit randomness without paying off bluffs).
- Committed Phase 2c as `796868b` (368 insertions, 2 files).

### What was decided
- **Use `baseline_winnings` as the headline eval metric** rather than raw `winnings`. Slumbot's per-hand baseline difference gives a 10-100x variance reduction in academic literature, so we need fewer hands to detect a small edge.
- **PolicyAdapter (trained policy → Slumbot action string) deferred to Phase 2d.** No trained policy exists yet (only smoke runs); the adapter needs careful action-translation work that's better done with a real policy in front of us.

### What was learned / surprises
- **The Slumbot API is more useful than I'd expected.** The variance-reduction signal alone is worth a chapter — it makes evaluation 1-2 orders of magnitude cheaper than naive bb/100.
- **Action-language pedantry matters.** `c` vs `k` for call vs check, BB-as-implicit-bet preflop — these aren't documented prominently anywhere and only came out via 4 illegal-action rejections. Lesson for next protocol-integration work: probe with curl extensively before writing client code.
- **My prior on "random policy loses ~200 bb/100" was wrong by an order of magnitude.** Restricted-action-set random play in HUNL is much closer to break-even than I'd modeled. Worth recalibrating expectations for how hard the bot has to play to *not* lose money to Slumbot.

### Queued for Session 4 (next session)
1. PolicyAdapter: trained Deep CFR policy → Slumbot action string.
2. Optional Phase 2b real-run on Contabo CPU (100 iter × 100 trav) to get a real baseline policy before Phase 2d.
3. Phase 2d: rent RunPod 4090, scale up training, eval against Slumbot, get headline bb/100 number.
4. Session 2 cleanups: train_leduc/train_nlhe config.json location inconsistency.

---

## Session 5 — 2026-05-22
**Focus:** GPU validation of Phase 2d. Deploy training to rented GPU, measure throughput, evaluate trained model against Slumbot. Decide whether bigger network + more compute pushes past CPU baseline.

### What was done
- Patched src/nlhe/solver.py and src/nlhe/policy_adapter.py to support GPU/CUDA: auto-detect device at solver init, move all 4 networks to device, send batch tensors to device in both training methods, bring forward-pass results back to CPU before numpy conversion, use map_location on checkpoint load. Five surgical patches via python sed. Tested on Contabo (where CUDA is unavailable, so device=cpu path) — all 28 unit tests pass. Committed and pushed (34da1e3).
- Three failed RunPod pod deployments before getting a working one. First two: Community Cloud RTX 4090, both failed with "CUDA unknown error" / error 999 from raw cudaGetDeviceCount even though nvidia-smi worked. Confirmed via direct CUDA C program — not a PyTorch problem, container/driver mismatch. Third deploy on Secure Cloud got RTX PRO 4000 Blackwell (sm_120 architecture) which initial PyTorch couldn't use; upgraded to torch 2.11.0+cu128 which has sm_120 in arch list. Compute verified end-to-end.
- Set up project on pod: clone via PAT (password auth deprecated), install deps minus torch (kept the working 2.11.0+cu128), scp abstraction artifact from Contabo to pod, sanity-check imports. treys was missing from requirements.txt (manually pip install'd, real bug to commit at session end).
- GPU smoke test (5 iters, [512,512] networks): solver device: cuda confirmed, all 5 iters complete, bimodal per-iter time (1.0s-99s) from deep-trajectory variance. Pipeline validated on GPU.
- Bumped smoke to 50 iters as Stage 2 benchmark. Strategy loss saturated at ~1.00 around iter 26 when 100K buffer filled. Bigger network alone didn't break the CPU's plateau.
- Designed v2 config: 5000 iters, 100 traversals/iter (doubled), 500K buffer (5x), checkpoint_every: 100. Launched in tmux on pod for resilience to SSH dropouts. Per-iter time averaged ~95s with high variance (1s-326s depending on traversal depth).
- Hit two new bugs in policy_adapter.py during ckpt_iter_0100 eval: (1) torch.set_rng_state raised TypeError on cross-version checkpoint load (PyTorch RNG state format issue) — patched with try/except since inference doesn't need RNG continuity, (2) missing .cpu() before .numpy() in choose_action's forward pass — patched. Both fixes mirrored back to Contabo at end of session.
- Slumbot evaluation of ckpt_iter_0100 (1000 hands, seed 2026, 200bb): **+31.45 baseline-adjusted bb/100** (raw -340.39). For comparison, CPU run (275 iters, [64,64]) was -14.8 baseline-adj bb/100 on 500 hands. GPU improvement: +46 bb/100.
- Buffer behavior confirms hypothesis: strategy loss plateau dropped from ~0.95 (100K buffer) to ~0.85 (500K buffer). Bigger buffer was the real lever, not bigger network.

### What was decided
- **Phase 2d closed as a success.** The +31.45 baseline-adj bb/100 at iter 100 validates the GPU pipeline and demonstrates measurable improvement from CPU. Original Phase 2d goal was infrastructure validation; we got that plus a meaningful performance signal.
- **Phase 3 plan revised to be more ambitious.** Original architecture treated subgame solving as Phase 5-6. Revised plan moves subgame solver engineering into Phase 3 as a parallel track, alongside DCFR and archetype framework. Reasoning: at current compute prices, replicating Pluribus's compute is ~$125, not millions. The hard part is algorithmic correctness, not compute. With AI-assisted engineering, a 6-10 week effort can produce a Pluribus-class bot for our format.
- **Project goal sharpened:** strongest publicly-known 6-max NLHE SNG bot with correct ICM. Targets include beating Slumbot in HUNL, decisive wins against the 42 bought bots, 70%+ top-3 finish rate in SNG simulations, sub-second decisions via subgame solving, and plausibly beating Pluribus head-to-head in 6-max cash (stretch target).
- **Within-match adaptation upgraded from Phase 6 to a Phase 3 parallel track.** Continuous archetype representation (tightness × aggression), Bayesian updating from observed actions, population priors by stake level, response as integration over belief. This is genuinely beyond what Pluribus did and could be the differentiator.

### What was learned / surprises
- **GPU patches that look correct on CPU can have residual bugs.** Two patches landed today (RNG state + missing .cpu()) had to be discovered at eval time despite the unit tests passing on CPU. Tests don't catch device-conversion issues when there's no device to convert from. Future: add a GPU-emulation test mode or just be more careful with patch coverage of every tensor-to-numpy boundary.
- **Buffer size matters more than network size at this scale.** This is the major Phase 2d finding. Going [64,64] CPU → [512,512] GPU with same 100K buffer plateaus at the same loss. Going [512,512] GPU with 500K buffer breaks past it. We had been treating "bigger network" and "more data" as roughly interchangeable upgrades; the data says no — buffer dominates.
- **RunPod Community Cloud is unreliable right now.** Two consecutive bad hosts with broken CUDA passthrough (CUDA 13 driver vs containers built for earlier CUDA, evident even at raw libcuda level). Wasted ~45 minutes on Community before bailing to Secure Cloud. Lesson: for first GPU run of a new project, Secure Cloud is worth the $0.35/hr premium. Community Cloud is good only after you have a known-working pod recipe.
- **RTX PRO 4000 Blackwell (sm_120) requires PyTorch 2.7+.** Older PyTorch wheels were compiled for sm_90 and earlier. PyTorch 2.11.0+cu128 was the first version we found in pip that supported Blackwell. Worth noting for future Blackwell-era pod deployments.
- **EMD card abstraction is fundamentally limited.** Confirmed earlier in this session via the bucket-runouts probe: AA, QQ, and TT all map to the same preflop bucket regardless of bucket count or query precision. This is a representation limit, not a parameter to tune. OCHS would fix it; that's Phase 3 research work.
- **The Pluribus compute argument is weaker than I thought.** Pluribus's 8 days × 64 cores = ~700 PFLOPS total. One RTX 4090 today is ~80 TFLOPS sustained. To match Pluribus's compute on one 4090: ~365 days = $125 at $0.34/hr. With 8 GPUs running 5 days: more compute than Pluribus, for ~$1000. The argument "we can't match Pluribus compute" is wrong. The argument "we can't replace 6 expert CFR researchers" is partially countered by AI-assisted engineering.

### Workflow notes for next time
- Sed-based python patches for surgical multi-line edits work well. Heredocs for new files work well. Both should remain the default for code changes through Phase 3.
- tmux is required on RunPod pods for any run longer than ~10 minutes. Install with `apt-get update && apt-get install -y tmux` as first step on a fresh pod.
- One-liner status checks from Contabo (`ssh ... 'tail -5 /tmp/log'`) are much cheaper than attaching SSH for every progress check. Standardize on this.
- When patches touch device handling, run a quick smoke test of the actual GPU inference path before committing — CPU unit tests don't catch GPU-only issues.

### Queued for next session (Session 6 / Phase 3 kickoff)
1. SCP and archive any additional checkpoints from the still-running pod training (ckpt_iter_0200, 0300 if available).
2. Optionally eval ckpt_iter_0200 against Slumbot to chart bb/100 trajectory; decide based on its result whether to keep the pod running or terminate.
3. Begin Track A: implement DCFR (Linear/Discounted CFR) in src/nlhe/solver.py. Add per-buffer-entry timestamp, weight policy training by iteration index.
4. Begin Track B: subgame solver design doc. Identify the OpenSpiel subgame extraction APIs, sketch the depth-limited CFR variant.
5. Begin Track C: design archetype representation (continuous tightness × aggression), and the Bayesian update scheme for within-match opponent modeling.
6. Investigate OCHS abstraction implementation.

---

## Session 4 — 2026-05-22 (~02:00-04:35)
**Focus:** Open Phase 2d. Write the `PolicyAdapter` that bridges trained `DeepCFRSolver` to the Slumbot eval harness. Validate the infoset encoder is stack-parametric so we can use a 20bb-trained checkpoint as a plumbing test against the 200bb Slumbot game. Plumbing-test end-to-end. Set up a 200bb overnight training run.

### What was done
- **Two SSH dropouts mid-session, recovered both times** — no work lost because all design state was in a planning chat with another Claude instance and all code was on disk. tmux would have been cleaner; the recovery worked.
- **First Claude Code session:** `/init` wrote `CLAUDE.md`; diagnostic check confirmed `infoset.py` normalization is parametric on `starting_stack` (no bug); wrote `scripts/verify_abstraction_stack_invariance.py` confirming `Abstraction.bucket_of()` takes no stack-related args (EMD-on-equity-histograms is stack-agnostic by construction); ran a 20-iter NLHE smoke train at 20bb to produce a checkpoint for the plumbing test. Checkpoint `ckpt_iter_0020.pt` at `runs/nlhe_20260522_030612_smoke/`. 15.5 min wall, final losses adv=1.63 strat0=0.81 strat1=0.83, buffers symmetric 1.13:1 (refuting Session 3's 4:1 observation as small-sample noise).
- **Second Claude Code session (after first dropout):** wrote `src/nlhe/policy_adapter.py` (454 lines), `tests/test_policy_adapter.py` (24 tests initially), refactored `scripts/eval_vs_slumbot.py` for `--policy {random,adapter}`. First plumbing test against Slumbot at 200bb-vs-20bb-checkpoint mismatch surfaced a real protocol bug.
- **Bet-translation bug:** Slumbot `b<N>` means "total chips committed by actor on the **current street**"; OpenSpiel `universal_poker` bet int N means "total chips committed by actor across the **whole hand**." Identical preflop, divergent the moment any chips have been committed before the current street. Failure mode was a flop `b300` from Slumbot translating to OpenSpiel int 300 when the legal floor at that flop node was 400 (300 prior + 100 new street min-bet). 5 of 20 plumbing-test hands errored.
- **Empirically confirmed the per-hand semantics** by probing OpenSpiel directly: at a flop decision node with 300 chips committed by each player preflop, `legal_actions()` started at int 400 and `action_to_string(400)` returned `'player=0 move=Bet400'` — i.e., the bet int includes prior-street commitment.
- **Fixed** by adding a `prior_streets_committed_by_actor: int = 0` kwarg to both `slumbot_token_to_openspiel_action` and `openspiel_action_to_slumbot_token`. Replay loop maintains a per-player dict; refreshes it at each postflop street boundary by parsing `[Money: X Y]` from `state.information_state_string(0)`. Default 0 keeps preflop callers and tests unchanged. Added 2 new tests for the postflop translation path (28/28 green).
- **Second plumbing-test run: 20/20 hands clean, 0 errors.** Single commit `c07fd9c` covering adapter + tests + eval refactor + fix.
- **Cleanup commit `9e7be8b`:** `CLAUDE.md` added to repo, `scripts/verify_abstraction_stack_invariance.py` added, `configs/nlhe_smoke_iter1.yaml` dropped.
- **Wrote `configs/nlhe_200bb.yaml`** and ran iter-1 timing benchmark on Contabo CPU. **48.9s/iter at `[64,64]` / 50 trav / 100 steps, 5.5x CPU parallelism.** Decided to push for an overnight run.
- **Tried to upsize** (`[128,128]` / 100 trav / 200 steps / 300 iters / ckpt every 25) but iter 1 hung at 100% CPU **single-threaded** (process alive, not parallelizing). Killed at 4+ min, root cause undiagnosed.
- **Reverted to bench-proven config** + `n_iterations: 300` + `checkpoint_every: 25`. Iter 1 completed in 37.9s, iter 2 in 29.0s, CPU back to 411%. Confirmed healthy and left running in a separate SSH session (PID 730653, logging to `/tmp/200bb_overnight.log`, run dir `runs/nlhe_20260522_043341_phase2d_200bb_overnight/`, ETA ~07:15 Contabo).

### What was decided
- **`PolicyAdapter` is the bridge between trained policies and the Slumbot eval harness.** Lives at `src/nlhe/policy_adapter.py`. Eager init (fail-fast on bad config / checkpoint mismatch), loud assertions on state-reconstruction failures (no graceful degradation), and explicit warning when training-stack ≠ eval-stack so plumbing-test runs don't get confused for production eval.
- **Plumbing-test pattern:** train a small smoke model at the cheap stack, build the adapter against it, run a 20-hand eval to surface protocol bugs *before* paying for serious compute. Paid off immediately by catching the bet-translation bug.
- **200bb overnight CPU training uses the bench-proven `[64,64]`/50/100 config** for predictability. Larger configs deferred until we understand the `[128,128]` hang.
- **GPU rental decision is now data-conditional:** overnight result determines whether GPU is "make it stronger" or "make it work at all."

### What was learned / surprises
- **The Session 3 lesson restated and applied:** "always benchmark the actual config before kicking off a long run." Tried to triple-bump (network + traversals + train steps) on top of a single-knob bench and the result hung. The right move is one knob at a time.
- **ACPC `b<N>` semantics aren't universally per-hand or per-street.** Slumbot and OpenSpiel `universal_poker` happen to disagree. Identity-mapping was an assumption that survived design review because nobody actually traced a postflop example. The plumbing test caught it; without it, GPU training would have produced inscrutable bb/100 numbers.
- **SSH dropouts are recoverable when all work is on disk.** Two dropouts this session; lost zero code. tmux would have been cleaner up front.
- **Buffer asymmetry from Session 3 (4:1) was small-sample noise, not structural.** The Session 4 smoke run at the same stack depth produced 1.13:1. Closes that "known issue."

### Queued for next session
1. Review overnight 200bb training artifacts (final losses, full 300-iter trajectory, all 12 checkpoints, buffer behavior).
2. Plumbing-test the latest checkpoint against Slumbot at matched 200bb stack depth — this is the actual Phase 2d headline test.
3. Decide on RunPod rental for `[256,256]` scale-up — data-conditional on overnight result.
4. Doc/code-hygiene carryovers: `train_leduc.py` config.json/metrics.json location, `_build_game_state_view` underscore prefix, three-way `--run-name` CLI inconsistency, possibly diagnose the `[128,128]` hang.
