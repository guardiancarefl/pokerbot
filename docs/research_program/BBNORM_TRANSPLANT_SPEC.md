# BBNORM TRANSPLANT SPEC — bbnorm depth-cured encoder into the H2 league retrain

Registration (Addendum 5 slate 2b, 2026-06-12). Status: **PRE-REGISTERED,
CONDITIONAL** — executes only if the H2 probe PASSES all of
`H2_KILLPHIL_LEAGUE_SPEC.md` §5 (F-M1/F-M2a/F-M2b/F-iter). If the probe
FAILS, this spec is void and is recorded as such. Launch is an operator
decision (pod spend — Addendum 2); this document is the full pre-commitment
so nothing is decided after seeing results. Per Addendum 1: thresholds below
may not be weakened after launch.

Evidence base (all cited, nothing new asserted):
- **Organ verdict (a3):** `evals/a3_v2_decomposition_20260612/REPORT.txt` —
  the bbnorm encoder's depth cure paid EV exactly where champion depth
  confusion lives: paired Δ **+0.130/game at ≤6bb (z=+11.8, 15/19 profiles
  positive)** over 48,000 CRN-paired games. v2's gate failure decomposes into
  (i) a depth-flat killphil hole (policy defect → H2's target) and (ii) an
  **independent broad-spectrum over-folding regression** that travels with
  the run, not the encoder. Confound caveat: one run, one seed (G=12).
- **Representation probe:** `evals/attacker_ext_20260611_pergame/bbnorm_probe_20260611/REPORT.md`
  — median TV ratio 0.004–0.006 vs deployed 1.55–3.16 (PASS, stable iters
  100–500).
- **Live-servability:** v2 gate G4 PASS (`evals/v2_gate_20260611_pergame/REPORT.md`)
  — 1,263-frame bridge replay byte-identical status distribution, floors
  compose on the 237-d bbnorm encoder, `require_live_servable` → "real".
- **League mechanism:** H2 probe (`docs/research_program/H2_KILLPHIL_LEAGUE_SPEC.md`),
  verdict pending at registration time.

---

## 0. Exactly what is being transplanted (code-verified definitions)

Two SEPARATE encoder deltas exist; the v2 candidate carried BOTH. Do not
conflate them:

| delta | config key | encoder field | what it does | feature_dim |
|---|---|---|---|---|
| **eff-BB depth channel** (C3, 2026-06-10) | `encoder_eff_bb` | `InfosetEncoder6Max.include_eff_bb` | APPENDS one channel at the END: `min(hero, max alive opp)/bb / EFF_BB_SCALE(=100)` — same quantity as the deployment floors' `_hero_eff_bb`. First 236 offsets byte-identical to legacy. | **236 → 237** |
| **bbnorm money re-normalization** (Phase A, 2026-06-11) | `encoder_bb_norm` | `InfosetEncoder6Max.include_bb_norm` | CHANGES THE DENOMINATOR of the 15 money slots (6 per-seat stacks + 6 per-seat contributions + pot + to_call + legacy eff-stack) from `starting_stack` (the /1500 smear — the depth-confusion mechanism per the C3 NO-SHIP DECISIONS entry) to `current big_blind * BB_NORM_SCALE(=100)`. **Same slots, same count — feature_dim UNCHANGED.** Legacy eff-stack slot becomes a duplicate of the eff_bb channel (accepted, keeps layout stable). Falls back to starting_stack if bb≤0 (never occurs in this pipeline). | unchanged |

"The bbnorm encoder" in this spec = **both flags on** (`encoder_eff_bb: true`
+ `encoder_bb_norm: true`, 237-d), i.e. exactly the v2 candidate's encoder
(`configs/v2_retrain.yaml` in the pod bundle; mirror config at
`mirrors/v2_retrain_podmirror/v2_retrain/config.json`).

**Code location warning:** `include_bb_norm` does **NOT exist on
track-policy HEAD**. It lives only in the pod bundle
`mirrors/runpod-env_v2gate_20260611.bundle` (branch head `8f734ae`;
introduced in `de974de`, parallel wiring in `eb77691`). HEAD's
`src/nlhe/infoset6.py` has only `include_eff_bb`, and HEAD's
`TrainConfig6Max.__post_init__` **raises** if `encoder_eff_bb` is combined
with `parallel_groups > 0`. §1 ports this before anything runs.

---

## 1. Prerequisite code port (P0 — no training until DONE + tests green)

Port from `refs/bundles/v2gate` (fetch the bundle; commits `eb77691` parallel
wiring + `de974de` bbnorm) onto track-policy. Measured diff vs HEAD
(`git diff HEAD refs/bundles/v2gate -- <files>`): 238 insertions / 35
deletions across 7 files:

| file | delta | content |
|---|---|---|
| `src/nlhe/infoset6.py` | +30 | `include_bb_norm` field, `BB_NORM_SCALE`, denominator switch in `encode_from_parsed` |
| `src/nlhe/solver6.py` | +25/−x | `encoder_bb_norm` TrainConfig6Max field; **relax the `__post_init__` parallel guard** (bundle wires encoder flags through workers) |
| `src/nlhe/parallel/protocol.py` | +20 | `encoder_bb_norm` / `encoder_eff_bb` in `WorkerInput` |
| `src/nlhe/parallel/worker.py` | +63 | encoder construction with both flags |
| `src/nlhe/parallel/orchestrator.py` | +23 | pass-through |
| `scripts/eval_6max_self_play.py` | +9 | `_load_solver`: `encoder_bb_norm=saved.get("encoder_bb_norm", False)` |
| `scripts/train_c3.py` | +103 | launch asserts (encoder echo, `include_bb_norm` mismatch refusal) — port the asserts into the §2 entry point even if train_c3 itself is not used |

Port acceptance gates (all must pass before P1):
1. **Legacy bit-identity:** with both flags False, encoder output and a
   deployed-champion (`b79e82dd`) 100-hand eval are byte-identical to HEAD.
2. **Parallel bit-identity:** sequential vs G>0 with flags ON — the bundle's
   own gate passed ("bit-identity PASS, 4.13x at G=12", commit `eb77691`);
   re-verify on track-policy after merge (league code coexists here).
3. **Loader round-trip:** save a 1-iter bbnorm checkpoint, reload via
   `_load_solver`, assert `encoder.include_bb_norm is True` and
   `feature_dim == 237`.
4. Existing test suite green.

---

## 2. Config deltas — `configs/h2_probe_league.yaml` → full retrain

**Structure decision (pre-registered):** the run is **two-phase from
scratch**, composing the two evidence bases exactly rather than inventing an
untested third regime:
- Phase 1 reproduces the v2 recipe shape (bbnorm from scratch, pure
  self-play, parametric sampler) — v2 proved this trains stably to
  panel-neutral (G2 pooled z=−0.54) within 1100 iters.
- Phase 2 reproduces the H2 probe shape (league_mix 0.30 continuation from
  iter 1500) — the probe (if PASS) proved this teaches shove defense.
League from iter 0 is NOT what H2 tested; it is not used.

### Phase 1 — `configs/bbnorm_league_v1_phase1.yaml`
Deltas vs `configs/h2_probe_league.yaml` (everything not listed: identical —
abstraction_path, game shape, tournament_structure_path, num_paid, hidden_dim
[256,256], traversals_per_iter 150, train_steps_per_iter 200, batch_size 64,
learning_rate 0.001, buffer_capacity 500000, cfr_variant linear,
dcfr_exponent 1.0, bucket_runouts 30, max_traversal_depth 200,
checkpoint_every 100):

| key | h2_probe_league.yaml | phase 1 | why |
|---|---|---|---|
| `tag` | h2_probe_league | `bbnorm_league_v1` | new run identity |
| `encoder_eff_bb` | (absent = false) | **true** | the organ (§0) |
| `encoder_bb_norm` | (absent = false) | **true** | the organ (§0) |
| `ante_convention` | (absent = "real" default) | **real** (explicit) | checkpoint stamp, conventions.py live gate |
| `league_mix` | 0.30 | **(removed = 0.0)** | league enters at phase 2 only (probe-faithful) |
| `league_registry_path` | configs/league/registry_h2_probe.json | **(removed)** | ditto |
| `league_sample_strategy` | uniform | **(removed)** | ditto |
| `n_iterations` | 2000 | **1500** | phase-1 endpoint = the champion's resume point |
| `seed` | 2026 | **4242** | **MANDATORY change.** Phase 1 at seed 2026 + G=12 would deterministically replay `runs/v2_retrain` (bit-identical training is a project invariant) and inherit its over-folding regression. The new seed is also the a3 one-run-confound test (§5.4). |
| `parallel_groups` | 8 | 8 (Contabo) / **12** (pod) | requires §1 port; G recorded in run header |
| (launch flag) | `--resume <champion ckpt_1500>` | **NO resume — from scratch** | §3: champion checkpoint is incompatible |
| (launch assert) | — | `empirical_dist_path` UNSET; assert at launch (v2's G3-PARAMETRIC equivalent) | v2 isolation choice carried: parametric sampler, NOT C3's empirical distribution |

Entry point: `python -m scripts.train_6max --config configs/bbnorm_league_v1_phase1.yaml`
(train_6max passes yaml → `TrainConfig6Max(**cfg_dict)` generically; honors
parallel + league + resume). Port train_c3's launch echo/refusal asserts
(encoder feature_dim==237, include_bb_norm live) into the launch wrapper.
Benchmark 1–2 iterations before committing (standing rule).

### Phase 2 — `configs/bbnorm_league_v1_phase2.yaml`
Identical to phase 1 EXCEPT:

| key | phase 2 value |
|---|---|
| `n_iterations` | **2500** (cap; resume target 1500 + 500–1000 league iters under the stop rule below) |
| `league_mix` | **0.30** |
| `league_registry_path` | `configs/league/registry_h2_probe.json` — **the FROZEN H2 pool, unchanged**: {killphilmtt (shanky, post-a8933ea), champion ckpt_1000, champion ckpt_0500} |
| `league_sample_strategy` | uniform |

Launch: `python -m scripts.train_6max --config configs/bbnorm_league_v1_phase2.yaml --resume runs/bbnorm_league_v1/.../ckpt_iter_1500.pt`
(own phase-1 checkpoint; RNG state restored from checkpoint).

Note the 236-d past-self league entries are fine as OPPONENTS: league
checkpoints load via `league_pool.sample_opponent → CheckpointPolicy →
_load_solver`, each reconstructing its OWN encoder from its own
`config_dict`. Mixed encoder dims across hero/opponents are structurally
supported. Scripted/league decisions never enter the strategy buffer
(non-traverser short-circuit), same as the probe.

**Stop rule (phase 2, pre-committed):** per-checkpoint (every 100) run the
frozen fold-vs-shove battery M1 + a 2000-game killphil CRN mini-row. Stop at
the first checkpoint ≥2000 where neither M1 nor the killphil row improved
>1 SE over the preceding best in 3 consecutive checkpoints; candidate =
best-M1 checkpoint subject to killphil-row no-worse-than-best−1SE. Hard cap
2500. No re-rolls; an infrastructure kill may be relaunched once from the
last checkpoint, recorded.

**Mid-run kill checks (phase 1, regression tripwire):** at ckpt_0500 and
ckpt_1000, run a reduced floored-arm behavioral probe (6,000 paired games,
`scripts/short_stack_floor_ab.py`, hpl=5). If fold-facing-action mass
exceeds the §5.3 bin caps by >1.5× at either checkpoint → kill the run and
write the report (a reproduced regression at a new seed is itself the a3
confound answer). This is a "type of problem changed" kill, per the standing
kill criterion.

---

## 3. Checkpoint compatibility — can it resume from the champion? **NO.**

Two independent incompatibilities, either alone fatal:

1. **Hard (shape):** champion `b79e82dd` (k200_real_ante ckpt_iter_1500)
   has `config_dict` without encoder keys → 236-d encoder → all advantage
   and strategy nets have first-layer weights `[256, 236]`. The transplant
   encoder is 237-d (`encoder_eff_bb`) → `load_state_dict` size-mismatch
   error. Not patchable by zero-padding a column: see (2).
2. **Soft (semantics) — applies even where shapes match:** `encoder_bb_norm`
   changes the VALUES of 15 input slots (chips/1500 → chips/(bb·100))
   without changing dimension. At L1 (bb=25) the same stack encodes 1.0
   (legacy) vs 0.60 (bbnorm); at L8 (bb=600) 1.0 vs 0.025. Champion weights
   are trained against the legacy scaling; warm-starting them against
   re-scaled inputs is an uncontrolled perturbation, not a continuation.
   (This is also why a silent mis-load is dangerous — §4.1.)

Reservoir buffers store encoded feature vectors → equally incompatible, but
moot: the deployed champion checkpoint is SLIM (10.9 MB, no buffers — H2
spec §4 noted deviation).

**Considered and rejected:** resuming from v2 `ckpt_iter_1100` (encoder-
compatible, sha `68b13754…`, mirrors only). Rejected because a3's verdict is
that v2's over-folding regression travels with the RUN — resuming inherits
its weights/trajectory and destroys the single largest open question (seed
confound). **Consequence: this is a from-scratch ~2000–2500-iteration run.**
Cost in §6. The H2-probe-passed artifact itself (champion+league, 236-d) is
NOT carried forward as weights — only its VERDICT (league mechanism works)
and its instruments are.

---

## 4. Gate / live-path implications of the encoder change

### 4.1 The silent-load hazard (highest-priority guard)

`encoder_bb_norm` does not change `feature_dim`. A loader WITHOUT the §1
pass-through loads a bbnorm checkpoint with **no error** and serves it with
legacy-normalized features — silently corrupt at every non-L1 blind level.
(`encoder_eff_bb` alone fails loudly via shape mismatch; bbnorm alone does
not.) There is exactly ONE loader chokepoint:
`scripts/eval_6max_self_play._load_solver`, consumed by ~33 scripts
(verified by grep), including the live path (`scripts/run_live_dryrun.py`,
`src/nlhe/integration/live_loop.py` — its docstring pins this loader), the
panel/gate harnesses (`scripts/eval_pool.CheckpointPolicy` →
`sng_baseline.py`, `fold_vs_shove_battery.py`), the A/B harnesses
(`short_stack_floor_ab.py`, `tail_floor_ab.py`), `depth_invariance_probe.py`,
and `league_pool`. Required guard (port + new): after construction, assert
the encoder flags equal the checkpoint `config_dict` values, and **refuse to
load any checkpoint whose `config_dict` contains encoder keys the running
code does not recognize** (forward-compat refusal — prevents the
mirror-image hazard on stale deployments).

### 4.2 Inventory of 236-d assumptions (grep `feature_dim|236` on HEAD)

| site | assumption | action |
|---|---|---|
| `src/nlhe/networks6.py` (`input_dim: int = 236` default) | default only; solver constructs with `input_dim=self.encoder.feature_dim` | none |
| `scripts/eval_6max_self_play._load_solver` | rebuilds encoder from `config_dict`; HEAD lacks `encoder_bb_norm` key | §1 port + §4.1 guard — BLOCKING |
| `src/nlhe/solver6.py` `__post_init__` guard | refuses encoder flags + parallel | §1 port — BLOCKING |
| `src/nlhe/adaptive/model.py` `F_TOKEN: int = 236` (Test A6.2 pin) | Layer-4 within-match adaptation consumes encoder vectors as tokens; file states bumping = schema change invalidating trained adaptive checkpoints | NOT in the deployed live chain today. Pre-register: F_TOKEN bumps to 237 ONLY when/if the adaptive layer is retrained against the new champion; until then all adaptive artifacts remain champion(236)-bound and MUST NOT be composed with a 237-d blueprint. |
| `src/nlhe/subgame_leaf.py` (+ resolver/ReBeL track, `scripts/rebel_*`) | leaf encoders / value nets trained on champion-lineage 236-d features | resolver value nets are champion-specific; a shipped transplant invalidates them for the new blueprint (re-generation is a separate costed track, NOT part of this spec) |
| deployment floor chain (`live_loop.apply_*_floor`, `_hero_eff_bb_from_parsed`) | operates on `parsed` state + policy vectors, never on feature vectors | encoder-agnostic — no code change; behavior re-validated by G1(b)/G4 anyway |
| `src/nlhe/conventions.py` | per-checkpoint `ante_convention` stamp | satisfied (`ante_convention: real` stamped) |

### 4.3 Re-validation the live path needs before any ship

- **G4 bridge replay** on the new candidate (v2's G4 PASS is precedent for
  the encoder class, not for the new weights): full dry-run corpus, status
  distribution vs deployed, floor composition, `require_live_servable`.
- **Floor-chain A/B (G1b-style)** with the deployed floor chain incl. the
  now-ARMED `--tail-floor-tau 0.10` (OQ-2): floors must compose on the
  candidate; report all-games and diverged-only paired Δ.
- **Baseline re-record (BLOCKING, pre-training):** the recorded v1 panel
  baseline (`evals/sng_baseline_20260610`) is PRE-adapter-fix; a8933ea
  materially changed killphil-class as-played behavior (killphil row −0.174
  → −0.080). Re-record the full 24-profile × 2000-game CRN baseline
  ("v1.1") with the post-fix adapter and the SAME master seed 2026, and
  freeze §5 bars from it BEFORE phase-1 training completes. (e2_rebaseline
  covers only 9 rows.)
- **Per-hand instrumentation upgrade (BLOCKING, pre-gate):** implement a3's
  "cheapest harness fix" — G2/G3 runners append per-hand records
  (hand_idx, level, hero_stack_bb, n_alive, hand_net_chips) + CRN
  first-divergence hand id (~4 fields/hand, no extra compute; the ab harness
  already logs per-decision `hero_eff_bb`). The §5 depth gates are still
  DEFINED on the a3 terminal-proxy method for comparability; the per-hand
  data is recorded for the decomposition report.

---

## 5. Falsification plan (pre-committed; ship requires ALL of 5.1–5.3)

Candidate = phase-2 stop-rule checkpoint. Deployed anchor = champion
`b79e82dd` + deployed floor chain. All CRN master seed 2026 (eval side).
σ_diff = sqrt(se₁²+se₂²) per H2 spec convention.

### 5.1 H2 battery + killphil row (the transplant must keep the league win)
- **T-M1:** candidate M1 EV-loss on the FROZEN fold-vs-shove battery
  ≤ **0.0650** (champion baseline 0.0867 ± 0.0007; same −25% bar as the
  probe, on the same frozen file — labels not regenerated).
- **T-M2a:** killphilmtt row (2000 games, CRN 2026, post-fix adapter)
  ≥ **−0.0300** AND improvement vs the v1.1 killphil row ≥ 2σ_diff
  (v1.1 expected ≈ −0.0800 ± 0.0223 per e2; the frozen v1.1 number is
  authoritative).
- **T-M2b holds:** ticketmaster, sng, tighttom rows each degrade ≤ 2σ_diff
  vs v1.1 (e2 references +0.5400 ± 0.0188 / +0.8360 ± 0.0123 /
  +0.7780 ± 0.0140).

### 5.2 Full gate battery (v2-format, bars frozen at v1.1 re-record)
- **G1(a) depth probe:** median TV ratio < 1.0 (`depth_probe` harness;
  bbnorm reference 0.004 — failure here means the port broke the organ).
- **G1(b) floor A/B:** 24,000 paired games, hpl=5, seeds 1–24000, deployed
  floor chain both arms. DIAGNOSTIC, not ship-blocking, with pre-stated
  interpretation: if league training fixed shallow facing-action play, the
  diverged-only floor Δ should SHRINK vs v2's +0.320 ± 0.065; a value ≥
  +0.256 (deployed's own) means the floors remain fully load-bearing and the
  f1 hybrid-override option stays live.
- **G2 panel:** 24 profiles × 2000 CRN games vs v1.1 recorded baseline.
  C1: panel mean ≥ (v1.1 champion panel mean − 2σ_diff,mean). C2: = T-M2a.
  C3-bubble: bubble Δ/decision > v1.1 champion value. Frozen numerically in
  the v1.1 freeze block before phase-1 completes.
- **G3 calibration:** 2000 self-play games, candidate all six seats:
  |net/game| < 2σ from zero (v2 failed at 2.55σ).
- **G4 live-path bridge replay:** §4.3, byte-comparison format of the v2 G4.

### 5.3 a3 regression guard (NUMERIC, ship-blocking) — "not v2's disease"
Instrument: candidate floored arm of the G1(b) run (per-decision logs,
hpl=5, seeds 1–24000), facing-action decisions, binned by logged
`hero_eff_bb`. Reference values = deployed floored arm,
`evals/short_stack_floor_ab_hpl5` (seeds 100001+), as computed in
`evals/a3_v2_decomposition_20260612/analysis.json`
(`partB_behavior_by_depth_unpaired.deployed_floored_arm`). Cross-seed-space
comparison accepted exactly as in a3 (n ≥ 13k decisions/bin; sampling SE
negligible at these n).

Deployed reference (mean fold mass facing action / mean allin mass; n):

| bin | fold ref | allin ref | n |
|---|---|---|---|
| 0–4bb | 0.3884 | 0.2219 | 16,511 |
| 4–6bb | 0.5184 | 0.2247 | 13,141 |
| 6–10bb | 0.5562 | 0.1730 | 32,750 |
| 10–15bb | 0.6115 | 0.1445 | 51,872 |
| 15–25bb | 0.6034 | 0.1427 | 53,602 |
| >25bb | 0.6457 | 0.0956 | 229,034 |

**FAIL (regression) if ANY of:**
- Δfold(15–25bb) > **+0.04** or Δfold(>25bb) > **+0.04** — the H2 battery is
  5–15 BB; legitimate shove-defense tightening has no business at 15bb+.
  (v2: +0.0800 / −0.0015 → fails 15–25.)
- Δfold(6–10bb) > **+0.08** or Δfold(10–15bb) > **+0.08** — headroom for the
  LEGITIMATE H2 effect (champion over-CALLS shoves, 35.9% call mass vs 7.7%
  oracle; fixing it raises fold-facing mass in shove spots at these depths)
  while still catching v2-scale broad tightening. (v2: +0.0830 / +0.0641 →
  fails 6–10.)
- Δfold(0–4bb) > **+0.10** or Δfold(4–6bb) > **+0.10** (floors fire here;
  v2: +0.1830 / +0.1179 → fails both.)
- Δallin(10–15bb) < **−0.04** or Δallin(15–25bb) < **−0.04** — v2's
  aggression-mass drain (v2: −0.0295 / −0.0481 → fails 15–25).

v2 fails this gate on 5 of 8 clauses; the deployed champion trivially passes
(Δ=0). Bin caps were set BEFORE any transplant training exists and may not
be moved.

**Plus the EV-level mid-depth guard** (a3 method, terminal-depth proxy on
the G2 paired records vs the v1.1 baseline trajectory): pooled paired-Δ z at
6–12bb AND at 12–25bb each > **−3.0** (v2: −6.4 / −4.4 → fails both).

### 5.4 Organ-replication diagnostic (NOT ship-blocking; f-pillar bookkeeping)
From the same G2 decomposition: paired Δ at ≤6bb terminal depth, z ≥ **+3.0**
upgrades the f2 organ-bank entry from "economically signed at shallow, one
run" to "replicated across runs/seeds". z < +2.0 with ship gates otherwise
passing ⇒ record the shallow gain as possibly run-specific; the ship
decision is unaffected (it rests on 5.1–5.3), but the f-pillar entry is
downgraded and any FUTURE spec citing the +0.130 number must cite the
non-replication.

### Outcomes
- **PASS all of 5.1–5.3:** ship case goes to the operator (deployment is
  always an operator decision); RESEARCH_MAP f2 updated per 5.4; resolver
  value-net regeneration enters the queue as a separate costed track (§4.2).
- **FAIL any:** NO-SHIP, full report per Addendum 1, run preserved, organ
  verdict per 5.4 recorded either way. No seed re-rolls, no threshold
  shopping, one run (one infra relaunch allowed, recorded).

---

## 6. Cost table

Measured anchors: H2 probe benchmark **15.0–17.5 s/iter at G=8** on
contended Contabo (12 vCPU, oversubscribed ~10×); v2_retrain measured mean
**8.37 s/iter at G=12** on the 27-core pod (1119 iters in 2.60 h, identical
recipe + bbnorm; league adds cheap Shanky adapter calls). G2-class evals:
48,000 games in ~13 min wall at 12 workers (pod, measured 2026-06-11).
OPERATOR_QUEUE standing one-liner ("~3 h pod vs ~28 h Contabo") is the
optimistic-pod / contended-Contabo envelope of the same numbers.

| phase | Contabo (12 vCPU, contended) | 27-core pod |
|---|---|---|
| P0 code port + tests (§1) | 2–4 h dev, ~0.2 h compute | — (do locally) |
| P1 config + 1–2-iter benchmark | 0.5 h | 0.2 h |
| P2 v1.1 baseline re-record (24×2000) + bar freeze | 0.5–1 h | 0.3 h |
| P2b per-hand harness fields (§4.3) | 1–2 h dev | — |
| P3a phase-1 training, 1500 iters | 6.3–7.3 h (G=8 @ 15–17.5 s) | 3.5 h (G=12 @ 8.4 s) |
| P3b phase-2 training, 500–1000 iters + per-ckpt M1/killphil stop-rule evals | 2.1–4.9 h + ~1 h evals | 1.2–2.3 h + ~0.5 h |
| P4 gate battery: G1a (~1 h) + G1b 24k×2 arms (~1.5 h) + G2 48k (~1 h) + G3 (~0.5 h) + G4 (~0.2 h) + M1/M2 (~0.5 h) | 4.5–5.5 h | ~1.5 h if run pre-release |
| P5 decomposition + report | 2–3 h analysis (read-only) | — |
| **Total** | **~18–26 h compute + ~6–9 h dev/analysis** (matches the ~28 h queue envelope) | **~6–8 pod-hours** + ~4–6 h Contabo residual (P0/P2b/P5 + anything post-release); pod also frees all 12 local cores for eval parallelism |

Pod note: G=12 is the measured-safe setting (4.13× speedup verified with
bit-identity); higher G on 27 cores is plausible (~3 h total training) but
NOT assumed in the bars or the budget. Per the standing rule, benchmark one
iteration at the chosen G before committing.

---

## 7. Adversarial review (pre-launch, per Addendum 1)

- **R1 "Two deltas vs champion, not one":** the final recipe = champion +
  bbnorm + league band. Accepted deliberately: each delta is independently
  evidenced (bbnorm: Phase A probe + a3 economic signature; league: H2 probe
  PASS is a hard precondition of this spec), and the two-phase structure
  keeps each delta in the regime where its evidence was generated.
  Interaction risk is what §5.1–5.3 jointly test; there is no cheaper
  decisive experiment.
- **R2 "Phase 1 just re-rolls v2's dice":** yes — by design, at a NEW seed
  (4242). Either outcome is informative: a healthy phase 1 breaks the a3
  one-run confound; a reproduced over-folding regression at a new seed
  (caught by the mid-run tripwire) localizes the defect to the recipe, not
  the run, killing the transplant cheaply at ≤1000 iters.
- **R3 "The 5.3 caps could mask a legitimate H2 effect":** the 6–15bb caps
  carry +0.08 headroom precisely because shove-defense tightening is
  expected there; the strict caps sit at 15bb+ where the battery has no
  spots. If a candidate fails ONLY 6–15bb fold caps while T-M1/T-M2a pass
  strongly, that is recorded as a gate-design finding — but the gate still
  fails (no post-hoc weakening).
- **R4 "Cross-seed-space behavioral reference":** 5.3 compares seeds 1–24000
  (candidate) against 100001+ (deployed reference), unpaired — identical to
  the a3 method that produced the reference numbers; at n ≥ 13k/bin the
  sampling error is ~0.004–0.013 absolute, an order below the caps. NOTED.
- **R5 "Stale-loader deployments serve corrupt features":** the §4.1
  unknown-key refusal is BLOCKING in P0 precisely because bbnorm fails
  silently (same dim). Any environment that can load checkpoints (live box,
  eval shards, pod) must carry the ported loader before a bbnorm checkpoint
  is ever placed where it could be loaded.
- **R6 "v1.1 baseline freeze ordering":** bars must be frozen (P2) before
  phase-1 training completes, mirroring H2 §3's ordering discipline, so no
  bar is set with candidate results in view.

## 8. Registration checklist (fill at execution, in order)

- [ ] H2 probe verdict: PASS recorded at `evals/h2_battery/` + EXPERIMENT_LOG (precondition)
- [ ] Operator approval for pod spend / Contabo occupancy (Addendum 2)
- [ ] P0 port merged, 4 acceptance gates green (§1)
- [ ] P2 v1.1 baseline recorded, §5 bars frozen numerically (file:
      `evals/sng_baseline_v1_1_<date>/BARS_FROZEN.md`)
- [ ] P2b per-hand fields landed in G2/G3 runners
- [ ] Phase-1 launch (config sha + git head in run header); tripwires at 0500/1000
- [ ] Phase-2 launch from own ckpt_1500; stop rule log in dashboard
- [ ] Gate battery + 5.3/5.4 decomposition; verdict; RESEARCH_MAP + f2 update
