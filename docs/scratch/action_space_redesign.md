# Action-space redesign — preflop sizing void analysis + static proposal

**Status:** design only. NO code changes. NO config changes. NO training.
Written for operator review before commissioning a fresh blueprint train.

**HEAD at write time:** `f7ad2890f7743ed699e19a698e7ab3cb1a5964c5`
**Author session:** read-only design recon on `phase4f-league`.

---

## 1. Problem — what the current 7-action discretization misses

The deployed action set (`src/nlhe/actions.py`):

| label    | meaning                          |
|----------|----------------------------------|
| FOLD     | fold                             |
| CALL     | call (or check when nothing to call) |
| BET_33   | bet 0.33 × pot                   |
| BET_66   | bet 0.66 × pot                   |
| BET_100  | bet 1.00 × pot                   |
| BET_200  | bet 2.00 × pot                   |
| ALLIN    | bet effective stack              |

Every "BET\_X" is **pot-relative**. The pot used is `state.observation_string()`'s
`Pot:` field, which in `universal_poker` equals `num_players × max_contribution`,
not `sum(contributions)` — at SB=50 BB=100 6-max pre-action this reports
`Pot: 600` (=6×100), not `150`. The bot was trained against this convention, so
it is internally consistent; but it means "1pot bet" already corresponds to
**4× the sum of chips currently in the middle**, not 1×.

### 1.1 Measured void (probe on the deployed game, fresh chance-rolled states)

**Probe:** `src/nlhe.actions._legal_discrete_bet_sizes` called on
`_build_view_6max(state, parse_state_6max(state))` at varied stack depths,
SB=50/BB=100, six seats. Code path identical to what `cfr6.traverse_6max` sees
at decision time.

UTG **first-in opener** (smallest non-allin / largest non-allin → allin gap):

| stack | legal bet menu (bb)                          | gaps (bb)             | largest gap |
|-------|----------------------------------------------|-----------------------|-------------|
| 100bb | {3.96, 6.00, 12.00, 100.00}                  | 2.04, 6.00, **88.00** | 88.00       |
| 60bb  | {3.96, 6.00, 12.00, 60.00}                   | 2.04, 6.00, 48.00     | 48.00       |
| 30bb  | {3.96, 6.00, 12.00, 30.00}                   | 2.04, 6.00, 18.00     | 18.00       |
| 20bb  | {3.96, 6.00, 12.00, 20.00}                   | 2.04, 6.00,  8.00     |  8.00       |
| 15bb† | {3.96, 6.00, 12.00, 15.00}                   | 2.04, 6.00,  3.00     |  6.00       |
| 10bb  | {3.96, 6.00, 10.00}                          | 2.04, 4.00            |  4.00       |
|  8bb  | {3.96, 6.00,  8.00}                          | 2.04, 2.00            |  2.04       |
|  5bb  | {3.96, 5.00}                                 | 1.04                  |  1.04       |

† 15bb = production deployed config (`starting_stack: 1500`,
`configs/six_max_phase4f_dcfr_overnight.yaml`).

CALL (limp = 1bb) and FOLD are always available; allin is the right-anchor
of every row. `BET_33` (0.33pot = 1.98bb target) is **always dropped preflop
at every stack depth** because `1.98bb < min_bet=2bb`.

### 1.2 Three distinct pathologies, ranked by impact

**(P1) Deep-stack commitment band void.** At ≥30bb effective, the gap
between `BET_200` (2pot = 12bb) and `ALLIN` is ≥ 18bb and grows linearly.
At 100bb deep, the bot literally cannot represent a 4-bet to 20bb, a 5-bet
to 30bb, a 50bb shove-induce raise, or anything in between 12bb and the full
stack. Pseudo-harmonic translation handles inbound off-tree opponent bets but
does **not** widen the bot's outbound choice set. **This is the load-bearing
loss for any depth above SNG starting-stack territory.**

**(P2) Sub-2bb open hole.** The smallest legal preflop bet under current
discretization is `BET_66` ≈ 3.96bb at any blind level (because pot=6×BB
pre-action makes 0.66pot land at ~4bb). The bot cannot represent a 2bb
min-raise or a 2.5bb open. At 100bb depth this is "fine-but-wrong"; at SNG
depths where stacks are 10-30bb effective, opening to 4bb is materially
larger than canonical and burns ~1bb of fold-equity capital per open. Modest
EV impact, but persistent.

**(P3) Short-stack menu collapse.** At ≤8bb effective the BTN-facing-3bb-open
menu collapses to `{fold, call, jam}`; at 5bb the opener's own menu collapses
to `{fold, limp, ~4bb open, jam}`. This is **structurally correct** for SNG
push-fold — GTO at these depths is approximately binary — and is **not** a
priority to fix.

### 1.3 Where postflop stands

The 5-action postflop menu (BET\_33 / BET\_66 / BET\_100 / BET\_200 / ALLIN)
covers SPR ≤ 2.5 (the deployed SNG regime) adequately. Probed flop spots at
15bb deployment show gaps of 6bb and 3bb — fine. At 100bb depth postflop
inherits the same 12bb→100bb void in 3-bet pots, but the bot will rarely see
SPR > 4 in this SNG format because the chip ladder compresses fast.

**Recommendation:** keep postflop pot-relative. The cost/benefit is not there
to redesign postflop, and changing it broadens blast radius significantly
(every checkpoint, every league archive, every bias_factory builder).

---

## 2. Blast radius — what changes when the action set changes

Grep coverage of every action-space consumer in `src/` and `scripts/`.

### (a) Fixed-dimension surfaces (network output, checkpoint schemas)

These are the load-bearing dims. **A trained checkpoint at N=7 cannot be
loaded against an N>7 network.**

- `src/nlhe/networks6.py:31` — `N_DISCRETE_ACTIONS = len(DiscreteAction)`
  (auto-derives; net `out_dim` follows)
- `src/nlhe/networks6.py:99,116` — advantage + strategy nets `out_dim`
- `src/nlhe/solver.py:45`, `src/nlhe/archetypes.py:56` — same auto-derive
- `src/nlhe/cfr6.py:334` — `legal_mask = np.zeros(N_DISCRETE_ACTIONS)`
- **Every trained checkpoint:** the production blueprint
  (`runs/dcfr_anchor_2000`-class) and every league archive
  (`configs/league/...`) carry policy-net output dim 7. **All invalidated by
  N>7.** Mandatory fresh blueprint train.

### (b) Translation logic — handles new sizes automatically if data-driven

- `src/nlhe/actions.py:89-129` `policy_to_game_action` — iterates BET\_FRACTIONS;
  any new entry handled identically.
- `src/nlhe/actions.py:132-173` `_legal_discrete_bet_sizes` — iterates
  BET\_ACTIONS\_IN\_ORDER; same.
- `src/nlhe/actions.py:204-260` `game_to_policy_action` (pseudo-harmonic
  inbound translation) — iterates BET\_FRACTIONS for bracket-find; same.
- `src/nlhe/fast_view.py:107` `fast_discretize` — mirrors the legal-action
  logic with bisect; **must be re-checked** to ensure new entries flow through
  the bisect path correctly (single-line risk, but a known-quantity port site).
- **IF sizing becomes stack-relative:** `policy_to_game_action` must branch on
  pot-relative vs stack-relative per entry. New per-entry compute path.

### (c) Subgame tree builder + solver

- `src/nlhe/subgame.py:324` — decision-node children enumerated via
  `discretize_legal_actions` (auto-adapts to len(DiscreteAction)).
- `src/nlhe/subgame_solver.py:68` — `_N_ACTIONS = len(DiscreteAction)`;
  regret/strategy arrays sized off this. Auto.
- `src/nlhe/subgame_policy.py:45` — same auto.
- `src/nlhe/subgame_leaf.py` — uses DiscreteAction enum *names* for stub
  decisions; **new bet labels need explicit continuation-strategy weights at
  the leaf** (modest design work).

### (d) Bit-identity-critical — the C1c bias_factory path

- `src/nlhe/biased_policy.py:32-44` — `BiasConfig.multipliers` shape
  `(len(DiscreteAction),)` enforced in `__post_init__`. The C1c invariant is
  "at `confidence == 0` both raw-stats and archetype-Bayesian paths return
  all-ones multipliers." Growing N preserves this invariant trivially
  (`np.ones(N)`) — **but** every concrete BiasConfig declared with explicit
  multipliers must be regenerated.
- `src/nlhe/biased_policy.py:47-78` `standard_bias_configs` —
  fold-biased / call-biased / raise-biased archetypes need explicit choices
  for which new size labels get up/down weights.
- `src/nlhe/bias_configs.py` — raw-stats + archetype-Bayesian BiasConfig
  builders (Layer 4 C1c). Both must agree on the new vector layout for the
  bit-identity tests in `tests/test_biased_policy.py` to keep passing.
- `src/nlhe/archetypes.py:232-262` `ArchetypePolicy.action_dist_for_state` —
  hard-coded per-discrete-action weight tables (raise\_prob × {BET\_33: 0.5,
  BET\_66: 0.3, BET\_100: 0.15, BET\_200: 0.05}, etc). Adding sizes requires
  picking concrete weights for the new entries that preserve the archetype's
  intent (aggressive vs passive).

### (e) Cosmetic / informational

- `src/nlhe/cfr6.py:7` — docstring "7-action discrete abstraction"
- `src/nlhe/subgame.py:324` — comment "7-action DiscreteAction abstraction"
- `src/nlhe/policy_adapter.py:8` — docstring "samples a DiscreteAction"
- `src/nlhe/scripted_bots/policy.py` — Shanky parser/evaluator uses enum
  *names* (BET\_100, BET\_200, ALLIN); adding new size labels does not break
  existing scripted decisions but new labels won't be referenced by old
  Shanky rules without explicit edits.
- `src/nlhe/within_match.py:20` + `src/nlhe/infoset6.py:326` — both use
  `n_actions_street` (a *count of actions taken* this street in the
  betting history). Not the action-space size. **Unchanged** by this redesign.
- `src/nlhe/infoset6.py` feature dim 236 = bucket one-hot (200) + street (4) +
  per-player features (24) + game features (3) + betting features (5). **No
  dependency on action-space size.** Encoder unaffected.

---

## 3. Three concrete candidate sets

Default direction (per session prompt): **keep postflop pot-relative**, **add
stack-relative preflop sizing in the commitment band**.

### Candidate A — minimal additive (preflop-only, stack-relative)

Add three preflop-only stack-relative actions on top of the existing 7:

| new label   | semantics                       | rationale                       |
|-------------|---------------------------------|---------------------------------|
| PF\_RAISE\_2BB  | raise-to = 2 × BB (min-raise)   | fixes (P2): 2bb open option     |
| PF\_BET\_QTR\_STK | raise-to = 0.25 × eff\_stack    | fills (P1) 4-bet band           |
| PF\_BET\_HALF\_STK | raise-to = 0.5 × eff\_stack    | fills (P1) 5-bet / shove-induce |

Resulting action-space cardinality: **N = 10** (7 existing + 3 preflop-only).
Network out-dim: 10.

Street-conditional: yes. Postflop the three new entries are *always illegal*
(`_legal_discrete_bet_sizes` returns chip=0 path for them off-street),
so they never appear in the legal mask. Same blueprint network produces a
10-vector; mask zeros the three preflop entries on postflop nodes.

| change site                  | cost                                       |
|------------------------------|--------------------------------------------|
| `BET_FRACTIONS` split into pot-relative + stack-relative tables | small, mechanical |
| `policy_to_game_action` per-action branch                       | small             |
| `_legal_discrete_bet_sizes` street-conditional iteration        | small             |
| `fast_view.fast_discretize` mirror                              | small             |
| `pseudo_harmonic` inbound mapping with 3 new bracket sizes      | small, ~10 lines  |
| `BiasConfig.multipliers` shape 7 → 10                           | trivial           |
| every `standard_bias_configs` + `archetypes.py` weight table   | medium (deliberate weight choices needed) |
| every leaf-eval stub                                            | medium            |
| every subgame solver buffer (auto-sized)                        | free              |
| every trained checkpoint                                        | **invalidated**   |

### Candidate B — full preflop replacement (stack-relative only)

Replace the preflop bet menu entirely with stack-relative sizes:

| preflop label     | semantics                |
|-------------------|--------------------------|
| PF\_RAISE\_2BB    | raise-to 2×BB            |
| PF\_RAISE\_3BB    | raise-to 3×BB            |
| PF\_BET\_15PCT    | raise-to 0.15 × eff\_stack |
| PF\_BET\_30PCT    | raise-to 0.30 × eff\_stack |
| PF\_BET\_50PCT    | raise-to 0.50 × eff\_stack |
| ALLIN             | (shared)                 |

Postflop menu unchanged: {BET\_33, BET\_66, BET\_100, BET\_200, ALLIN}.
Union cardinality: **N = 11** (FOLD, CALL, ALLIN shared + 5 preflop-only +
4 postflop-only). Network out-dim: 11. Two strictly disjoint action sets
across streets.

| pros over A | cons vs A |
|-------------|-----------|
| cleaner mental model (sizes mean "fraction of decision-time chip equity") | one more action (11 vs 10) |
| 5bb / 7.5bb opens explicitly in the menu at SNG depths | every code site street-branches twice |
| 2-2.5-3bb preflop opens fully covered | bias / archetype weight design ~2× the work |

### Candidate C — finer-grain pot-relative (no stack-relative)

Add two pot-relative sizes that work on both streets:

- BET\_50  (0.5 pot, between BET\_33 and BET\_66)
- BET\_150 (1.5 pot, between BET\_100 and BET\_200)

Resulting cardinality: **N = 9**. No street-conditional logic. Net out-dim: 9.

**Verdict on C: rejected as the primary proposal.** Does not address (P1) at
all — at 100bb the 12bb → 100bb void remains 88bb wide. C is included only
as a sanity-baseline; "if A loses to the current N=7 blueprint, would C have
won?" is a useful falsification question (see §6).

### Recommended primary: Candidate A.

A captures the load-bearing fix (commitment-band coverage) at the smallest
code footprint and the smallest checkpoint-output growth (10 vs 11). The
preflop-only-additive shape is also the easiest to ablate: you can A/B the
new actions one at a time, by zeroing the corresponding column of the
post-softmax logits.

---

## 4. Street-conditional sizing — what it actually costs

Street conditionality (both A and B) adds these complexity surfaces:

1. **`_legal_discrete_bet_sizes` reads `parsed["street_idx"]`** to decide
   which fractions to iterate. Currently it reads view fields only; threading
   street through requires either extending `GameStateView` with a `street`
   field, or splitting the function in two. Extending the view is cheaper.
   (`infoset6.parse_state_6max` already exposes street.)

2. **`fast_view.fast_build_view` must populate the new `street` field**
   identically — the fold-in's "bit-identical to canonical" invariant in
   `_build_view_6max` would now mean "bit-identical including street."
   One additional regex hit per build; cost negligible.

3. **The advantage / strategy networks** still have a single output head of
   dim N=10 (A) or N=11 (B). The street feature is already encoded in the
   infoset 4-hot, so the network learns when each entry is meaningful from
   data; we don't need separate per-street heads. **Verified preserved
   bit-identity contract: no architectural change to the network beyond
   out\_dim.**

4. **Subgame tree builder** — `subgame.py` already enumerates children via
   `discretize_legal_actions`, which would return only the street-legal
   subset. No code change there.

5. **Pseudo-harmonic translation** at decision time of inbound opponent bets
   already uses BET\_FRACTIONS only. For A, the three new stack-relative
   sizes need to enter the inbound bracket-find path *only on preflop nodes*.
   Branching adds ~10 lines to `game_to_policy_action`.

**Estimate: A costs ~150 LOC of careful changes across 6 files, plus
the design work on bias / archetype weight tables.** B costs roughly 2×
that, mostly in weight tables.

---

## 5. Retrain implication — explicit

**Every trained checkpoint on disk is invalidated by this change.**

Concretely:
- The production blueprint (k=200 retrofit, ICM-correct DCFR-3400, the
  reference for the Stage 6-D verdict).
- The k=500 abstraction's prospective blueprint training.
- Every league archive in `configs/league/` (v1 + v2).
- The DCFR-2000 anchor.

The currently-committed `f7ad289` rewrite (vectorized EMD) unblocks a fresh
k=1000 abstraction overnight; that abstraction must be built **against the
new action set**, because the abstraction itself is action-set-agnostic but
the *blueprint trained on it* embeds the action set in its output layer.

**Sequence on green-light:**
1. Implement Candidate A (or B); add tests; verify C1c bit-identity at
   `confidence=0` over the new vector shape.
2. Build the k=1000 abstraction (cost projected: 2-4 h overnight, ref:
   commit `f7ad289`'s STOP message).
3. Train the new blueprint from scratch against the k=1000 abstraction with
   the new action set. Expected wall-clock: ~3 days at the previous DCFR
   3400-iter cadence; n=5000 iter target per the k=1000 design.
4. Re-run Stage 6-D verdict harness against the new blueprint to confirm
   subgame solving still lifts strength (architecture-level sanity).
5. Re-run league play to populate the v3 archive.

This is a **multi-week sequencing**, not a session of work. Do not commit to
the change without budgeting that.

---

## 6. Falsification — when to walk back

Project rule 4 requires **two independent benchmarks at σ ≥ 2.0**. Apply
both *gating* and *failing* criteria:

**Gating (must clear to ship):**
- F1: new-blueprint vs current k=200-retrofit DCFR-3400 blueprint, ≥5,000
  hands head-to-head, ICM-adjusted lift σ ≥ 2.0 in favor of new.
- F2: new-blueprint vs `runs/dcfr_anchor_2000`, same gate.

**Failing (ANY of these is grounds to revert):**
- (a) Neither F1 nor F2 clears σ ≥ 2.0 — added action granularity was not
  worth the extra dim.
- (b) F1 or F2 clears but **Stage 6-D rerun degrades** — i.e. the subgame
  PROFILE-vs-blueprint lift that motivates the whole B1c line collapses
  on the new action set. The architecture is now less tractable for
  subgame solving than the 7-action version.
- (c) Within-league exploitability (Nash gap proxy, BR-vs-blueprint pooled
  diff) grows by > 1.5 σ vs the old action space at matched training
  iterations. Larger action space is letting opponents exploit.
- (d) Wall-clock per training iter grows by > 1.5× without commensurate
  policy quality. The rewrite-EMD work bought time budget; we should
  not give it back.

**Cross-check experiment (cheap):** before committing to A, train a Candidate
C blueprint (N=9, pot-relative only, no street conditionality, much smaller
diff) at *matched* compute budget against the production. If C *also* clears
F1+F2, we've learned that the win is "more buckets" not "stack-relative
preflop" specifically — and the simpler change should ship instead. This
costs one extra training run but de-confounds the design hypothesis.

---

## 7. Open question — does this stay static or feed RL-CFR?

**Boundary of this proposal:** all three candidates (A, B, C) are **static**
discretizations chosen by the designer. The bot is trained to optimize over
a fixed action set known at config time. This is the same regime as the
current 7-action design — just with different fixed cardinality.

**Out of scope:** a later phase could replace the static menu with a learned
or game-state-conditional menu — RL-CFR / sequence-form policy gradient over
a continuous bet-size head, K-best ablation per-decision, etc. That is a
fundamentally different training architecture and is **not designed here.**
Flag it as a known follow-up; do not bake assumptions into the static
redesign that depend on it.

If RL-CFR is the eventual destination, the implications for *this* proposal
are:

- Candidate A's "preflop-only stack-relative additions" is the easiest
  bridge: a learned head could replace just the preflop branch later,
  leaving postflop pot-relative as a clean fallback path.
- Candidate B's symmetric street-disjoint design is also bridge-friendly.
- Candidate C is a dead-end bridge — its design choice is "more pot
  buckets," which a learned head doesn't help with.

This is **not** an argument to ship Candidate A over B today; it is a
reminder that today's choice can be undone or extended without rearchitecting.

---

## 8. Recommendation

**Ship Candidate A**, gated on the falsification criteria in §6 and the
explicit retrain budget in §5. Postflop stays pot-relative; preflop adds
three stack-relative entries for the commitment band. Net out-dim 10,
checkpoints invalidated, full retrain required.

**Do not ship without operator sign-off on the multi-week sequencing.** The
biggest cost here is not the code — it is the lost blueprint and the
forced rebuild of league archives + Stage 6-D verdict baseline.
