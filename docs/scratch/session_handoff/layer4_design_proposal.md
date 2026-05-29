# Layer 4 Design — Delta Against C1_PLAN.md

**Base document:** docs/C1_PLAN.md (120 lines, sub-phases C1a–C1f).
**Status of base document:** mostly correct; updates needed where it predates as-built code.
**This document's job:** carry forward what's still valid, flag what's stale, decide the
new integration point against the as-built subgame solver, and surface what the chat-Claude
conversation needs to decide before any C1a implementation.

**Recon log:** /tmp/layer4_recon_20260529T131740Z.log
**As-built files inspected:** src/nlhe/biased_policy.py (143L), subgame_policy.py (197L),
subgame_solver.py (711L), subgame_leaf.py, subgame.py, archetype6.py (243L), archetypes.py
(371L), infoset6.py (parse_state_6max), networks6.py, eval_pool.py (Policy Protocol).
**Design docs cross-referenced:** ARCHITECTURE.md §Layer 4, DECISIONS.md (anonymity +
Position-2 entries), SUBSTEP_5_DESIGN.md (gate at f≈0.27), SUBSTEP_6_DESIGN.md
(BR/PROFILE/blueprint 3-way lift verdict).

---

## 1. What C1_PLAN.md got right (carry forward unchanged)

These load-bearing decisions are still sound; recon found no reason to revisit them.

1. **Opponent anonymity & per-match wipe** (C1_PLAN §Constraints 1–2). DECISIONS.md
   re-locks this independently; no implementation choice can relax it. Concrete
   consequence: `wipe_match_state()` is structural, not a config knob.
2. **Position-2 (light bias, blueprint-anchored safety floor)** (C1_PLAN §Constraints 3,
   §"What C1 explicitly does NOT do"). Subgame_solver.py's existing `degraded → fall
   through to blueprint` design (SUBSTEP_5_DESIGN.md Decision 5.2) is the exact pattern
   Layer 4 inherits: if the bias produces garbage, the blueprint catches us.
3. **MatchObserver / SeatStats separation** (C1_PLAN §Module structure). The clean
   division — pure observation surface vs. bias generation — survives unchanged. The
   stats list (VPIP, PFR, aggression frequency by street, fold-to-bet by street,
   showdown win rate, average sizing relative to pot) is achievable from the
   `parse_state_6max` outputs (`sequences`, `pot`, `money`, `contribution`,
   `current_player`, `dealer_seat`) without new instrumentation. (Confirmed against
   infoset6.py lines 52–128.)
4. **Confidence ramp shape** (C1_PLAN line ~52: `<20 → 0`, `20–100 → 0→0.5`, `100–300 →
   0.5→1.0`, `300+ → 1.0`). The numbers are still empirical placeholders, but the
   *shape* (zero early, linear ramp, saturate late) is the right design — early reads
   in SNG are noisy and the blueprint anchor handles that for free.
5. **Sub-phase decomposition C1a–C1f.** The graph (observer → biaser → integrate →
   slumbot eval → bought-bot eval → wipe test) survives. Two phases need internal
   re-spec (see §8); the overall sequence does not.
6. **No NN trained on opponent observations** (C1_PLAN §"What C1 explicitly does NOT
   do"). A neural posterior on opponent behavior would need labeled data and risks
   overfitting to a 50-hand SNG sample. The hand-engineered stat → bias path is the
   right ceiling for this layer.
7. **No cross-decision caching of opponent state inside the solver.** SUBSTEP_5_DESIGN
   Decision 5.3 already commits the wrapper to be stateless between decisions; C1's
   per-seat state lives in the MatchObserver, not inside SubgamePolicy/SubgameSolveContext.

---

## 2. What C1_PLAN.md got wrong-by-evolution (stale assumptions to update)

### 2.1 Integration point

**C1_PLAN.md assumed** (line 66, verbatim):
> "The clean interface: B1's subgame solver currently takes a hand `state` and produces
> a refined policy. It uses the blueprint to estimate opponent ranges at every infoset.
> C1 extends this by passing a `range_bias_fn(opponent_seat, infoset) -> bias_multiplier`
> callable into B1's subgame construction."

**As-built reality** (subgame_solver.py + subgame_policy.py):
- The subgame solver has NO `range_bias_fn`, no `opp_policy`, no per-infoset opponent
  range estimate.  `SubgameSolveContext` (lines 76–141) contains only `blueprint`,
  `starting_stacks`, `payouts`, `hero_seat`, `n_iterations`, `rng`, `num_paid`,
  `average_weighting`. Adding a `range_bias_fn` field is feasible but is NOT how the
  solver currently expresses opponent strategy.
- Opponents inside the subgame are NOT represented as a "range" (distribution over
  hidden hands). They are represented as a fixed action distribution per decision
  node: `cache.sigma[nid] = RM+(_blueprint_adv(node))` (lines 269–287). Every
  non-hero, non-chance decision node weights its children by this σ in `_run_cfr`
  (lines 449–457).
- The opponent bias menu that does exist is `BiasedBlueprint` (biased_policy.py lines
  112–143), but it is consumed **only at leaf infosets**, by `evaluate_leaves` via
  `LeafEvalContext.biased_blueprint`. Not at intra-tree opponent decision nodes.
- SubgamePolicy hardcodes `self.biased = BiasedBlueprint()` (subgame_policy.py:64) with
  the default `standard_bias_configs(alpha=3.0)` — i.e. the k=4 menu is currently a
  process-level constant, not a per-seat or per-opponent quantity.

**Updated integration point.** Layer 4 has two distinct attachment surfaces, both
necessary for the design to actually bias opponent behavior in the solve:

  - **(A) Leaf-eval bias menu — per-opponent-seat BiasedBlueprint.** SubgamePolicy
    constructs `BiasedBlueprint` once with k=4 defaults. Replace this with a
    *per-seat* BiasedBlueprint built fresh on each `select_action` call from the
    current MatchObserver readings. This is the smallest change to as-built code:
    pass a `bias_factory: Callable[[seat], BiasedBlueprint] | None` into SubgamePolicy
    and into LeafEvalContext, and let evaluate_leaves choose the right factory output
    per opponent seat at each leaf infoset. None preserves current behavior bit-for-bit
    (the C1 == 0 case).
  - **(B) Intra-tree opponent-σ bias — warmup cache modification.** In
    `_build_warmup` (subgame_solver.py:269), after computing `cache.sigma[nid] =
    _strategy_from_advantages(adv, mask)`, optionally apply a per-seat bias
    via `apply_bias` (biased_policy.py:81) before storing. The CFR loop's opponent
    weighting (line 451: `sig = cache.sigma[nid]`) then uses the biased σ
    automatically. Surface for the hook: a `sigma_bias: Callable[[seat], BiasConfig] |
    None` field on SubgameSolveContext.

**Recommendation:** ship surface (A) first (less surface area, the leaf eval is where
the BR/PROFILE lift comes from per SUBSTEP_6_DESIGN.md §B). Add (B) only if measurement
shows the lift is bottlenecked by intra-tree opponent σ being blueprint-pure. This is a
STILL-OPEN decision for chat-Claude (see §6).

### 2.2 Other stale assumptions

- **C1_PLAN line 27: "bias multiplier" framing.** The framing is correct in shape (a
  per-action multiplier renormalized over legal actions) but C1_PLAN treats the
  multiplier as acting on a *range* (distribution over hands). The as-built
  BiasedBlueprint multiplier acts on an *action distribution at a given infoset*. The
  rename costs nothing — the same arithmetic — but the conceptual shift matters for §6
  Q3 (showdown observations cannot inform an action-distribution bias the same way they
  can inform a range estimate).
- **C1_PLAN §Open Questions Q4 ("6-max scaling — Phase 4 worry, not C1 blocker").**
  The repo is already 6-max as of the phase4f branch; this is no longer a future
  concern. Per-seat MatchObserver and 5-seat bias construction must work from C1a.
- **C1_PLAN line 79: "policy_adapter.py: extended to instantiate and wipe a
  MatchObserver per match."** policy_adapter.py exists but is the Slumbot wire-token
  translator (recon: src/nlhe/policy_adapter.py:1–18), not the per-match orchestrator.
  The correct host for `MatchObserver` instantiation is the hand-level eval loop in
  `scripts/eval_6max_self_play.py` (or whichever loop drives a "match"); the chat-Claude
  session needs to confirm where SNG match boundaries are actually identified in the
  current pipeline. (DECISIONS.md commits the wipe rule; recon did not find a "match
  end" hook in code.)
- **C1_PLAN §Module structure says `src/nlhe/within_match.py`.** Naming is fine. Note
  there is no existing module of that name; `policy_adapter.py` is taken (and orthogonal).
- **C1_PLAN's "1.5×–3×" bound (Pluribus citation).** standard_bias_configs uses
  `alpha=3.0` for the subgame leaf menu (biased_policy.py:28, comment cites the same
  Brown 2018 work but selects a tighter starting value). Layer 4's bias bound is a
  separate knob from the leaf-eval bias bound; do not collapse them. (See §4.)

---

## 3. The new integration architecture

```
                ┌────────────────────────────────────────────────┐
                │  Match-level loop (scripts/eval_6max_self_play  │
                │  or future match orchestrator)                  │
                │                                                 │
                │   per-match:                                    │
                │     observer = MatchObserver(num_seats=6)       │
                │     for each hand:                              │
                │        play_hand(observer, ...)                 │
                │     observer.wipe()  ← STRUCTURAL, not a flag   │
                └────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                ┌─────────────────────────────────────────────────┐
                │  Hand-level: each public action observed        │
                │      observer.update(seat, parsed, action, …)   │
                │      observer.note_showdown(...)  on showdown   │
                │                                                 │
                │  parse_state_6max already exposes:              │
                │    sequences ("ccc/cc") street_idx              │
                │    money/contribution/pot                       │
                │    current_player + dealer_seat (post-Step-7    │
                │    fix — cfr6.py:122)                           │
                └────────────────┬────────────────────────────────┘
                                 │
                                 ▼   (only when hero is to act)
                ┌─────────────────────────────────────────────────┐
                │  SubgamePolicy.select_action(parsed,state,…)    │
                │      gate evaluated (f≈0.27 solve, else blueprint)
                │      on SOLVE:                                  │
                │        ─────────────────────────                │
                │        bias_factory: seat → BiasedBlueprint     │
                │             ↑ built from observer.get_stats     │
                │             ↑ confidence ramp applied here      │
                │        passed into LeafEvalContext.biased_blueprint│
                │             (replaces hardcoded                 │
                │              BiasedBlueprint() at line 64)      │
                │        build_subgame_tree → evaluate_leaves     │
                │        → solve_subgame → extract_action         │
                └─────────────────────────────────────────────────┘
                                 │
                                 ▼
                          hero plays chip action
```

**Concrete code-touch list** (no implementation here — just where the edits land):

- `src/nlhe/within_match.py` (NEW). Contains `SeatStats` dataclass, `MatchObserver`
  class (`update()`, `note_showdown()`, `get_stats(seat)`, `wipe()`,
  `confidence(seat)`), and `stats_to_bias_configs(stats, confidence) → list[BiasConfig]`
  (the producer of the bias menu).
- `src/nlhe/subgame_policy.py:52` — extend `__init__` signature with
  `bias_factory: Callable[[int], BiasedBlueprint] | None = None`. Replace the
  hardcoded `self.biased = BiasedBlueprint()` at line 64 with: instantiate lazily per
  `select_action` using `bias_factory(opp_seat)` when present, else fall back to
  `BiasedBlueprint()` (preserves bit-identical pre-C1 behavior; SUBSTEP_6_DESIGN
  re-runnable).
- `src/nlhe/subgame_leaf.LeafEvalContext` — already takes `biased_blueprint`. The
  per-opponent-seat factory has to flow through this struct; either (a)
  `biased_blueprint` becomes per-seat (Mapping[int, BiasedBlueprint]), or (b) the
  factory is passed and evaluate_leaves looks it up at each leaf infoset. Pick (a)
  for the smaller diff; the per-seat dict is built once at select_action time.
- `scripts/eval_6max_self_play.py` — the per-match host. Construct the observer at
  match start, feed every observed action via observer.update, instantiate the
  SubgamePolicy with a `bias_factory=lambda seat: BiasedBlueprint(stats_to_bias_configs(
  observer.get_stats(seat), observer.confidence(seat)))` lambda. Call observer.wipe()
  at match end.
- `tests/test_within_match.py` (NEW). Per-C1_PLAN §Module structure: per-seat tracking
  correctness, decay/confidence ramp values, bias-config correctness against synthetic
  stats vectors, and the load-bearing wipe test (observer A's match-end state
  cannot affect observer B's behavior — no module-level globals, no `_warned_no_dealer`-
  style class latches that survive wipe).

**What does NOT change:**

- subgame_solver.py is untouched in path (A). Only path (B) — should chat-Claude opt
  in — adds a `sigma_bias` field to SubgameSolveContext.
- Gate logic (SUBSTEP_5_DESIGN Decision 5.1) is untouched. Layer 4 inherits f≈0.27.
- Blueprint training (cfr6/solver6) is untouched. Layer 4 is decision-time only.
- The `BiasedBlueprint.action_probs` API is untouched (biased_policy.py:131); Layer 4
  just constructs different `bias_configs` lists.

---

## 4. Bias parameterization — alignment with BiasedBlueprint

**BiasedBlueprint's actual parameterization** (biased_policy.py lines 31–110):

```python
BiasConfig(name, multipliers: ndarray(7,))   # one positive multiplier per DiscreteAction
apply_bias(probs, legal_mask, bias) = renormalize(probs * multipliers * legal_mask)
```

**Alignment status:** The C1_PLAN "1.5–3× multiplier" framing maps **directly** to a
BiasConfig.multipliers vector — same arithmetic, same renormalization, same fall-back
to uniform-over-legal when bias zeros the mass. The only translation work is mapping
each SeatStats reading to the right entry in the 7-element multiplier vector.

**Proposed mapping (initial; tune in C1b):**

| Observed stat (per opponent seat)        | Multiplies                                |
|------------------------------------------|-------------------------------------------|
| VPIP much higher than blueprint-typical  | CALL, BET_33, BET_66, BET_100 ↑           |
| VPIP much lower                          | FOLD ↑                                    |
| PFR much higher than blueprint-typical   | BET_33, BET_66, BET_100, BET_200 ↑        |
| Aggression freq high (street ≥ 1)        | BET_66, BET_100, BET_200, ALLIN ↑         |
| Aggression freq low                      | CALL ↑                                    |
| Fold-to-bet high (per street)            | FOLD ↑                                    |
| Showdown-win-rate high                   | (no direct multiplier; informs strength   |
|                                          | priors at leaves only — see §6 Q2)        |
| Avg sizing relative to pot ≥ ~0.85       | BET_100, BET_200, ALLIN ↑                 |

The per-action multiplier `m_a` for opponent seat `s` is computed:

```
m_a(s) = exp( confidence(s) · Σ_i w_i,a · z_i(s) )
clip  m_a(s) ∈ [1/α_C1, α_C1]              # α_C1 is C1's bias-cap, distinct from
                                            # standard_bias_configs's alpha=3.0
```

where `z_i(s)` is the standardized deviation of stat i from its blueprint-population
mean and `w_i,a` is the small mapping matrix above. `confidence(s) ∈ [0,1]` ramps per
C1_PLAN §"Decay and confidence" (carried forward).

**Why exp-of-linear and not directly multiplicative:** keeps multipliers strictly
positive (BiasConfig invariant, line 43–44) for any z, and `confidence=0` gives `m=1`
identically (recovers blueprint). Multiplicative-with-confidence-weight blending would
need extra clipping.

**Recommended initial α_C1 = 2.0** (tighter than the leaf-eval menu's 3.0). C1_PLAN
flagged 1.5–3.0 as the empirical zone; 2.0 is the conservative midpoint and matches our
values-driven robustness preference. Tune empirically in C1b.

**Open: SubgamePolicy currently builds a list of 4 BiasConfigs (k=4 menu) per
BiasedBlueprint.** Layer 4 needs to decide whether to:
  - replace the 4-entry menu wholesale with C1-derived configs (loses BR's
    bias-maximization-over-menu lift), or
  - keep the 4 standard configs and ADD a per-seat "observed-opponent" config that
    weights one of the leaf-eval k strategies. (See §6 Q1.)

---

## 5. Open questions from C1_PLAN.md — re-resolution

C1_PLAN.md lists four open questions. Re-classified:

**Q1: "Bias factor bounds. Pluribus reportedly used 1.5×–3× ranges. Our values-driven
robustness preference may push us toward the lower end (1.5×–2×). Decide empirically
during C1b–c."**

  - Classification: **PARTIALLY-RESOLVED-BY-AS-BUILT-CODE.** BiasedBlueprint's
    `standard_bias_configs` ships `alpha=3.0` (biased_policy.py:28) — the leaf-eval
    layer already committed to a value, but that's the *leaf-eval bias cap*, not the
    *C1 bias cap*. The two are now architecturally distinct and need separate values.
  - Resolution: Start `α_C1 = 2.0`; leave `alpha_leaf = 3.0` unchanged. Tune
    α_C1 in C1b–c. STILL-OPEN at the value level.

**Q2: "What counts as 'an action' for sample counting? Every public decision the
opponent makes, or every street-completing action?"**

  - Classification: **STILL-OPEN.**
  - Recommendation: "every public decision the opponent makes" — strictly more samples,
    smoother ramp, and the per-seat update is cheap. C1_PLAN already leaned this way
    ("probably 'every public decision'"). Chat-Claude should confirm before C1a.

**Q3: "Per-street vs aggregate biases. Should the bias multiplier vary by street?"**

  - Classification: **STILL-OPEN.**
  - Recommendation: aggregate first. The mapping table in §4 already conditions
    `aggression_freq` on `street ≥ 1`, so a partial per-street structure is baked in.
    Move to per-street multipliers only if C1b–c measurements justify.

**Q4: "6-max scaling — five opponents instead of one."**

  - Classification: **RESOLVED-BY-AS-BUILT-CODE.** The repo IS 6-max. C1a's per-seat
    MatchObserver is per-seat from the start. Per-decision cost: at most 5 BiasConfig
    constructions per select_action call (constant work, microseconds), only for the
    seats currently in the hand. Cheap. C1_PLAN's deferral to "Phase 4 when porting"
    is obsolete.

---

## 6. New open questions surfaced by the recon

These are not in C1_PLAN.md; the as-built code introduces or reframes them.

**Q1 (new): Single attachment point (leaf bias menu, §3 path A) vs dual (also intra-tree
opponent σ, path B).** Path A is the smallest change and inherits SUBSTEP_6_DESIGN's
BR/PROFILE lift mechanism. Path B is conceptually closer to C1_PLAN's original "nudge
opponent ranges" framing but requires modifying subgame_solver.py's warmup cache.
*Recommendation:* ship A first; measure lift in C1d; add B only if A's lift is small
and intra-tree opponent purity is the candidate explanation. Chat-Claude decision
before C1c.

**Q2 (new): Showdown observation handling.** A showdown reveals an opponent's actual
hand; that's an order of magnitude more signal than action-only observation. C1_PLAN
includes "showdown win rate" as a stat but does NOT spec how a revealed hand feeds back
into bias. Options: (a) ignore the revealed hand, only use the win/loss outcome —
preserves the "C1 is light" frame; (b) use revealed hands to build a per-seat *hand
range* posterior, which would be a genuine "range" estimate but requires range
infrastructure we don't have; (c) restrict to a calibrated "VPIP-adjusted showdown
strength" stat that nudges aggression-frequency interpretation. *Recommendation:* (a)
for C1a–c, decide between (b)/(c) before C1d. Pure (a) is the cheap default.

**Q3 (new): Archetype-mixture parameterization as an alternative to raw multipliers.**
The repo's archetype framework (archetype6.ArchetypePolicy, archetypes.py — NIT, TAG,
LAG, STATION, MANIAC) gives 5 named action distributions parameterized over
(bucket_id, in_position, pot_odds, stack_to_pot, legal_mask, facing_bet). A natural
alternative to "compute multipliers from stats" is "maintain a 5-dim posterior over
archetypes per seat, update from observed actions via the archetype likelihood, mix the
5 archetype action distributions weighted by posterior, blend with blueprint at low
confidence." Pros: principled (Bayesian over a known model class), uses real
infrastructure, the 5 archetypes ARE the prior basis. Cons: violates C1_PLAN §"What C1
explicitly does NOT do" ("Categorize opponents into archetypes online"). *Question for
chat-Claude:* is the C1_PLAN "no online categorization" rule still load-bearing, or was
it written when no archetype framework existed in code? STILL-OPEN.

**Q4 (new): Gate-fraction interaction with effective lift.** SUBSTEP_5_DESIGN measures
the solve gate at f≈0.27. Layer 4 only acts on the ~27% of decisions that go through
the solver — the other ~73% skip to the blueprint untouched. So effective per-hand
lift from C1 ≈ f × per-solved-decision-lift. With ~30 hero decisions per SNG hand and
the gate firing on ~8 of them, C1 has ~8 opportunities per hand to bias the opponent's
modeled behavior. The chat-Claude session should be honest about this ceiling before
C1d's eval budget is set.

**Q5 (new, light): Match-boundary identification.** C1_PLAN §Module structure cites
`policy_adapter.py` as the wipe host, but recon shows policy_adapter.py is the Slumbot
wire-token translator. The actual SNG match boundary in the current eval pipeline lives
somewhere in scripts/eval_6max_self_play.py / scripts/eval_pool*.py — not yet a named
event. Need a `match_id` or `match_started/ended` hook. Small but real C1a prereq.

---

## 7. Adjacent levers the user explicitly asked about

### Card abstraction (k=200 vs k=500 vs higher)

**Recommendation: keep k=200 for Layer 4 first cut.** Layer 4's value is *exploitation*
(detecting and biasing toward observed opponent style). Bigger k improves Nash quality
of the *blueprint* — the layer C1 sits on top of and adjusts. The two are orthogonal:
a stronger blueprint helps the C1 == 0 baseline equally, so the *lift from C1 over
blueprint* is largely abstraction-independent at this k regime.

The k=500 artifact (runs/abstraction_k500_20260529_013537) is a separate hedge that
should be evaluated on its own terms in a different track. If a k=500 blueprint is
ever trained, C1 plugs in unchanged — the bias-multiplier-on-action-distribution math
doesn't reference k.

### Deeper network

**Recommendation: no.** Layer 4 has no neural net in the C1_PLAN design and no new
neural component in this delta. The bias config is computed from O(few-dozen) scalar
stats via a small linear map (§4). The blueprint network (networks6.py MLP
hidden=[256,256]) is the only NN in the solve path, and it's frozen at decision time.
If a deeper net helps, it helps the blueprint; that's a separate track from C1.

### New features

C1_PLAN's stat list (VPIP, PFR, aggression frequency by street, fold-to-bet by street,
showdown win rate, average bet sizing relative to pot) is **all derivable from the
parse_state_6max output** plus per-street action records. Recon confirms the parser
exposes `sequences` (the betting string per street, e.g. "ccc/cc"), `money`,
`contribution`, `pot`, `current_player`, and `dealer_seat` (post-Step-7 fix). No new
features need to be exposed from the encoder or game state. The MatchObserver is a
pure consumer of parse_state output.

### CPU vs GPU

**CPU-only is fine; per-decision overhead is negligible.**
- Per-action observer.update: O(1) — a handful of float increments per stat.
- Per-decision bias construction: O(num_opponents × num_stats) ≈ 5 × 12 = 60 scalar
  ops, then 5 BiasConfig.multipliers vectors of length 7 → microseconds.
- Inside the solve, the bias multiplier already lives on the leaf-eval hot path
  (subgame_leaf evaluates the k=4 menu); replacing the k=4 menu with a per-seat
  per-decision menu does not change the leaf-eval algorithmic complexity, only what
  multipliers it uses.

Layer 4's per-decision compute is dwarfed by the leaf-eval (~6.7s blended per solve per
SUBSTEP_5_DESIGN.md §D). The chat-Claude session does not need to revisit the GPU
deferral on this layer's account.

---

## 8. Sub-phase status — what's the next implementable unit?

For each of C1a–C1f from C1_PLAN.md (lines 81–93):

- **C1a — observer + stats.** `implementable-as-described`. Caveats: (a) wire into
  `scripts/eval_6max_self_play.py`, not `policy_adapter.py` (per §2.2); (b) need a
  `match_started/ended` signal (§6 Q5); (c) Q2 from §5 (sample-count units) must be
  decided before observer.update is written.
- **C1b — range biaser → BIAS-CONFIG BUILDER.** `needs-update`. C1_PLAN's
  `RangeBiaser` → in this delta becomes `stats_to_bias_configs(stats, confidence) →
  list[BiasConfig]`, the producer of BiasedBlueprint inputs. The validation target
  (synthetic maniac/nit produce predictable bias multipliers) is unchanged.
- **C1c — B1 integration.** `needs-update`. C1_PLAN's "plug range_bias_fn into B1
  subgame construction" → in this delta becomes "pass per-seat BiasedBlueprint dict
  through SubgamePolicy → LeafEvalContext". Path (B) (intra-tree opponent σ bias) is a
  C1c+ option pending §6 Q1's resolution.
- **C1d — Slumbot eval with C1+B1.** `implementable-as-described` once C1c is done.
  The 3-way verdict structure from SUBSTEP_6_DESIGN.md (blueprint / PROFILE / BR) can
  be reused — add a 4th column: BR+C1 vs BR — and we get the C1 lift over BR for free.
- **C1e — bought-bot eval.** `implementable-as-described` against the
  `src/nlhe/scripted_bots/` profiles (Phase 5 already integrated 42 Shanky bots).
  The style-extreme profiles (maniac, nit) are exactly the regime where C1 should
  produce measurable lift.
- **C1f — wipe-test.** `implementable-as-described`. Critical: also test that no
  module-level singletons / class latches survive the wipe (cf.
  `ArchetypePolicy._warned_no_dealer` class-level latch at archetype6.py:78; a similar
  pattern in MatchObserver would silently violate the wipe rule).

**Next implementable target after chat-Claude approves this delta: C1a.** No
prerequisites blocked (the match-boundary hook is a small refactor, not a redesign).
C1b can proceed in parallel with C1a since they share no code.

---

## 9. What this delta does NOT decide

Hand-offs to chat-Claude before any code is written:

1. **§5 Q2 — sample-count units.** "Every public decision" vs "every street-completing
   action." Default recommendation: every public decision. Confirm.
2. **§5 Q3 — per-street vs aggregate biases.** Default recommendation: aggregate first,
   per-street if measurements justify. Confirm.
3. **§5 Q1 — α_C1 value.** Default recommendation: 2.0. Confirm direction (lower vs
   higher); empirical tuning happens in C1b–c.
4. **§6 Q1 — single attachment point (leaf menu, path A) vs dual (also intra-tree
   opponent σ, path B).** Default recommendation: A first, B as a measured-need
   follow-up. Confirm.
5. **§6 Q2 — showdown observation handling.** Default recommendation: outcome-only
   (option a) for C1a–c; revisit before C1d. Confirm.
6. **§6 Q3 — raw-multiplier vs archetype-mixture parameterization.** Default
   recommendation: raw multipliers per the existing BiasedBlueprint surface;
   archetype-mixture is the principled alternative but violates the "no online
   categorization" rule. Confirm whether that rule is still load-bearing.
7. **§6 Q4 — gate-fraction realism.** Not a decision per se, but a *budget honesty*
   item: the eval setup for C1d needs to account for the fact that C1 only acts on
   the gated-solve fraction (f≈0.27). Confirm we will report C1's lift both
   per-solved-decision and per-hand-average.
8. **§6 Q5 — match-boundary hook location.** A small refactor to scripts/eval_6max_self_play.py
   (or wherever a "match" boundary exists in the current eval driver). Confirm scope.

**Out of scope for this delta:** the active control training run at
`runs/phase4f_dcfr_control_2000_20260529_122715/` is orthogonal — Layer 4 is
decision-time only and consumes whatever blueprint exists at eval time. The chat-Claude
session can let that run finish and proceed with C1a in parallel without contention.
