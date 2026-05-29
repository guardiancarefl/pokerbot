# Layer 4 / C1 — Locked Decisions

Source proposal: /tmp/layer4_design_proposal.md
Verification reads: this file (Part A)
Status: locked by chat-Claude on 2026-05-29T13:37:14Z. Open items remain for C1b/d.

## Verification (Part A summary)

- **A.1 — archetype distribution surface.** `archetypes.archetype_policy(...)` returns
  the full length-7 ndarray over `DiscreteAction`. No refactor required to evaluate
  per-action likelihoods. (`archetype6.ArchetypePolicy.select_action` samples from this
  same dist; both are usable.)
- **A.2 — prior basis.** `NAMED_ARCHETYPES` is 5 explicit in-code profiles
  (NIT/TAG/LAG/STATION/MANIAC, each a hand-set `play_quantile_by_street` + `aggression`
  scalar); per-street equity quantiles loaded from `runs/archetype_design/
  bucket_equity_analysis_6max.json`. `archetype_policy` is deterministic in its body
  (rng is a leftover parameter, unused). Likelihood `P(action | arch, state)` is one
  O(1) function call per archetype, 5 total per observation — cheap. Caveat: requires
  the opponent's hidden `bucket_id`; resolve via uniform-population marginalization or
  abstraction-mean conditioning (chat-Claude C1b decision).
- **A.3 — dimensionality.** Both `BiasConfig.multipliers` and `archetype_policy` output
  are shape `(len(DiscreteAction),) = (7,)` over the same DiscreteAction enum. No
  translation needed.
- **VERIFICATION VERDICT: IMPLEMENTABLE-IN-C1B.** §6-Q3's archetype-prior path requires
  no new infrastructure.

## Locked

| ID | Question | Decision | Notes |
|----|----------|----------|-------|
| §5-Q1 | α_C1 bias cap | 2.0 (start; tune C1b–c) | Distinct from leaf-eval alpha=3.0 |
| §5-Q2 | Sample-count units | Every public decision | C1_PLAN's preference confirmed |
| §5-Q3 | Per-street vs aggregate biases | Aggregate first | Partial per-street in §4 mapping table |
| §6-Q1 | Path A only vs A+B | Path A only for C1 | Path B revisited only if C1d lift small |
| §6-Q2 | Showdown handling | Win/loss outcome only | Revisit only if C1e weak vs maniac/nit |
| §6-Q3 | Archetype prior | BOTH-PATHS-PERMITTED-IN-C1B | C1_PLAN "no online categorization" rule dropped; raw-stats and archetype-derived BiasConfigs implemented in parallel; empirical comparison picks winner (A's verdict justifies — no infrastructure cost) |
| §6-Q5 | Match-boundary hook | Part of C1a scope | Wipe-fires-against-this signal |

## Open (deferred to C1b)

- §6-Q3 final form: raw-stats z-scores OR archetype-derived z-scores in the §4 mapping
  table. Both paths converge on identical BiasConfig output. Empirical comparison in C1b.
- §6-Q3 sub-decision: archetype-likelihood bucket handling — uniform-population
  marginalization (~200× factor, still microseconds) vs abstraction-mean conditioning
  (free). Pick in C1b based on stat-recovery test against synthetic opponents.
- §6-Q4: gate-fraction × per-decision-lift accounting in C1d's eval design. Reporting
  format = (lift per gated-solve decision) AND (lift per hand-averaged).

## C1a scope (next implementable unit)

- New file: `src/nlhe/within_match.py` — `SeatStats` dataclass, `MatchObserver` class,
  `update()`, `note_showdown()`, `get_stats(seat)`, `wipe()`, `confidence(seat)`.
- Hook into `scripts/eval_6max_self_play.py` for match-boundary signals
  (`match_started`, `match_ended`) — required for wipe.
- Tests: `tests/test_within_match.py` covering per-seat stat correctness, confidence
  ramp values, wipe correctness (NO module-level globals, NO class latches surviving
  wipe — reference `archetype6.py:78` `_warned_no_dealer` as the anti-pattern).
- NOT in scope: BiasConfig generation (that's C1b), SubgamePolicy integration (C1c).

## C1b scope (parallelizable with C1a)

- New function: `stats_to_bias_configs(stats, confidence) → list[BiasConfig]`.
- Two implementations side-by-side per §6-Q3 lock:
  - Raw-stats path: §4 mapping table with z-scored stat deviations.
  - Archetype-derived path: 5-archetype posterior over `NAMED_ARCHETYPES`,
    Bayesian-updated from observed actions via `archetypes.archetype_policy`
    likelihoods, posterior-weighted mixture mapped to BiasConfig multipliers.
- Resolves §6-Q3 final form based on C1b empirical results.
- Tests: synthetic maniac/nit/station `SeatStats` produce predictable `BiasConfig`
  multipliers within α_C1=2.0 bounds; archetype-derived path posterior concentrates on
  the correct archetype after ~100 actions of a known-archetype synthetic opponent.

## C1c+ scope (after C1a + C1b)

- `subgame_policy.py`: `bias_factory` parameter, replace hardcoded `BiasedBlueprint()`.
- `subgame_leaf.LeafEvalContext`: per-seat `biased_blueprint` dict.
- Bit-identity acceptance gate: `bias_factory=None` must produce byte-identical
  `evaluate_leaves` output vs the current process-level `BiasedBlueprint()`. This
  protects SUBSTEP_6_DESIGN re-runs (the C1==0 evaluation must reproduce the recorded
  BR/PROFILE/blueprint lift verdict bit-for-bit).

## C1d, C1e, C1f

- As specified in `/tmp/layer4_design_proposal.md` §8, no changes.
