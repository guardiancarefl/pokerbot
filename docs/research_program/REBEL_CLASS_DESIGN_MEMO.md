# ReBeL-class rebuild — literature-grounded design memo (RESEARCH_MAP a2)

**Date:** 2026-06-12. **Slate:** Addendum 5, analysis slate item (f). **Compute used:** none (file-only + literature).
**Question (a2 verbatim):** what would a ReBeL-class (belief-state + search) rebuild require for THIS
format — 6-max SNG, top-3-equal ICM, CPU-only live serving (<1 s/decision), opponent anonymity —
given that the resolver closure was *leaf-value precision on a depth-limited resolve*, NOT a verdict
on a *learned belief-value architecture*?

**One-paragraph answer:** the rebuild is architecturally well-specified (the repo already wrote the
spec: `docs/REBEL_PBS_6MAX.md`), latency-feasible (the resolver's unshippable 82–99 s p95 was rollout
leaf-eval, which the value net replaces with a ~ms forward; the CFR solve itself is 0.109 s/1000
iters), and compute-affordable at "beat-the-blueprint" grade (not paper grade) using the repo's own
measured sample-gen rates. What is NOT established is that it attacks the binding leak: X0 measured
lift ≈ 0 for a *correct-model* resolver vs blueprint-alone, and the KillPhil diagnostic showed the
real loss is opponent-range-dependent — a self-play PBS value net still assumes blueprint-like
opponents. Verdict in §7: no build; one pre-registered zero-new-compute probe (belief-block R² on the
existing 2.1M shards) is justified as idle-time work, with hard go/no-go gates before anything bigger.

---

## 1. What the prior in-repo ReBeL track actually established — and where it stopped

The repo has TWO distinct closed tracks that get conflated. Keeping them separate is the whole point
of a2.

### 1a. The real-time resolver track (CLOSED net-negative, 2026-05-31)

Depth-limited subgame CFR at decision time with **rollout leaves** (PROFILE_SAMPLE / BEST_RESPONSE
over biased blueprint continuations), warm-started from the k=200 blueprint. Evidence
(`docs/STATUS.md` "Floor lock"; `docs/DECISIONS.md` "Blueprint-alone is the reference agent" +
"Foundation pivot"):

- **X0:** lift ≈ 0 vs a correct-model opponent — even when the leaf model is *right*, the search adds
  nothing measurable over blueprint-alone (`evals/X0_resolver_vs_self.json`; gate solved only ~37% of
  decisions, the rest blueprint passthrough).
- **J2:** condition-A −2.7 to −4.2 ICM-pts vs blueprint on all 4 legit profiles.
- **J3:** robust BR-leaves help partially, never reach blueprint-alone, and are **unshippable: action
  p95 82–99 s, 14.5% of actions >15 s at depth 3**.
- Depth sweep: d=3 is the *worst* point, non-monotone — not a "go deeper" story.
- X5 bubble slice: net-negative on both shards (killphil ≤15BB −0.039 σ−2.5; ticket >15BB −0.044
  σ−2.9).
- Recorded root cause: **"leaf-value precision/bias vs unsafe solve"** — the leaves, not the solver.

### 1b. The ReBeL track proper (`rebel-search` branch, 2026-06-02→04) — stopped, not falsified

What it **established** (survives, reusable):

1. **Safe re-solving primitive validated on Leduc** (`docs/REBEL_GATE1_FINDINGS.md`): after three
   failed hand-rolled gadget formulations (410 / 135 / 262 mbb/g — all *worse* than unsafe 54.6), the
   real **augmented-tree CFR-D gadget** (`src/rebel/gadget.py::AugGadgetCFR`) reached **0.263 mbb/g**
   vs Nash 0.123 — essentially lossless safe re-solve. GATE 1 PASSED. Also the negative control that
   matters for §5: **exact Nash continuation values used as STATIC leaves give 219 mbb/g** — fixed
   leaf values are unsafe even when *perfectly accurate*. This is the cleanest in-repo statement of
   why the resolver's static-leaf architecture was doomed independent of leaf precision.
2. **Search plumbing byte-exact** (Layers 1/2/3a: depth-limited CFR ≡ engine, PBS partition bijective,
   multi-root subgame solver exact).
3. **The 6-max PBS spec** (`docs/REBEL_PBS_6MAX.md`) — §2 below is built on it.
4. **Sample-generation economics measured** (`docs/REBEL_SAMPLEGEN_PILOT.md`, `_RESULTS.md`): leaf
   rollout is 90–99% of per-sample cost; rates in §3.
5. **A rough value net trained and honestly diagnosed**
   (`docs/REBEL_GATE2_TONIGHT_AND_TOMORROW.md`): 236-dim input (public block + acting-seat bucket
   one-hot, **no belief block**), scalar acting-player-relative target, 38k params on 558k samples →
   **held-out R² = 0.237** (beats predict-mean; plateaus from epoch 1). Diagnosed structural caps:
   (a) scalar head can't value non-hero leaves (perspective bug → "silent search corruption");
   (b) belief block missing from the encoding; (c) M=1 rollout label noise that more data won't fix.

Where and **why it stopped** (four compounding reasons, in order):

1. **Gate 2 (cash-rate vs k=200) was never validly run** — deferred for the perspective bug + missing
   Shanky benchmarks on the GPU box (`REBEL_GATE2_TONIGHT_AND_TOMORROW.md`). The fixes (per-seat
   6-vector targets, belief block, M>1 leaves) were specified for "tomorrow" and never executed.
2. **The KillPhil diagnostic re-pointed the program** (`docs/REBEL_KILLPHIL_DIAG_FINDINGS.md`): the
   persistent leak is **concentrated** (<10bb call-vs-shove to showdown, 43.5% of loss in 18% of
   hands) and **range-dependent** — the resolver's effective uniform-over-buckets opponent belief is
   too wide for KillPhil (7.1% shove range) and too narrow for STATION; a static fix is exactly
   net-zero across the panel. Conclusion recorded there: "closing it requires opponent-range
   modeling — the adaptive within-match layer." Program pivoted to the StratFormer-style adaptive
   track (`docs/REBEL_OVERNIGHT_PHASE1_REPORT.md` is that track's Phase-1, not ReBeL).
3. **The training distribution was invalidated**: the inflated-ante/ghost-ante convention bug
   (`docs/DECISIONS.md` ghost-ante + deployment-reversal entries) reopened the ship decision; only
   the blueprint was retrained real-ante; "the rebel value net's retrain was queued 'after blueprint
   stabilizes' and has not happened." The ~2.1M overnight samples were generated against the
   inflated-ante candC blueprint → stale distribution.
4. **Policy-net warm-start increment was NULL** (val_KL 1.20 vs uniform ~1.43 — too weak to redirect
   CFR; `REBEL_KILLPHIL_DIAG_FINDINGS.md` table row 1).

**Net:** the track stopped on *program-priority + data-validity* grounds with the decisive
experiment (belief-conditioned per-seat value net → Gate 2) designed but unrun. a2's premise is
correct: no in-repo evidence speaks to a learned belief-value architecture.

## 2. Belief-state sizing for 6-max with our abstraction

Per `docs/REBEL_PBS_6MAX.md` (pre-registered, unchanged here; numbers re-derived):

- **PBS = (public state, {r_seat : seat active})**, per-seat ranges over abstraction buckets,
  **product factorization** across seats (standard; ignores cross-seat card removal beyond
  bucketing — a flagged approximation, acceptable given the abstraction already collapses removal
  into buckets).
- **Dimension:** public block ~36–37 dims (street 4, seat-position 6, per-seat stacks 6, active mask
  6, contributions 6, pot/to_call/eff-stack 3, betting 5; +1 if the C3 `encoder_eff_bb` channel is
  carried) ⊕ belief block 6×k. Postflop k=200 → **≈1,237 dims**; preflop k=20 → ≈157. Unabstracted
  per-seat beliefs would be 6×1,326 = 7,956 — the k=200 abstraction buys a 6.4× reduction, at the
  cost that within-bucket distinctions are invisible (the known k=200 ceiling, RESEARCH_MAP a1) and
  bucket assignments carry irreducible MC noise (DECISIONS.md abstraction-noise entry: KK/QQ flip
  buckets even at 5,000 runouts).
- **How ICM enters:** three places. (i) *Public block:* the per-seat stack 6-vector + blinds level —
  the full ICM-relevant state for a top-3-equal payout (payout vector is constant, needs no
  encoding; busted seats enter via the active mask + zero stacks). (ii) *Targets:* per-seat
  **ICM-equity-delta 6-vector** (head A) — the exact unit `subgame_leaf.evaluate_leaves` and
  `solve_subgame` already back up, so the net is a drop-in leaf. (iii) *Sampling distribution:* ICM
  curvature concentrates at the bubble (4 alive, short stacks) — `stack_sampler` /
  `tournament_structure_path` machinery must oversample those configs or the net is accurate where
  it doesn't matter. Note `_sample_alive_count` floors at 4 alive — correct for this format (match
  ends at 3) and consistent with the heads-up-convention bug being moot (DECISIONS.md).
- **Output head:** start with head A (6-vector, per PBS doc §4 recommendation). Head B (per-bucket
  CFV 200-vector for the acting seat — closer to ReBeL/DeepStack's per-hand counterfactual values)
  only if A underfits; it multiplies label cost ~k×.
- **Net sizing:** input ~1.25k → e.g. 1024→512→6 ≈ 1.8M params; CPU forward on a batch of ~40–220
  leaves is single-digit ms — serving is not a constraint. At the repo's empirical ~10–30
  samples/param, a 0.5–2M-param net wants **5–60M samples** (the samplegen docs' "paper-grade ~18M
  params" is out of reach and unnecessary; the gate is beat-k200, not superhuman).

## 3. Value-net training compute — grounded in the repo's own pilot numbers

Measured generation rates (bootstrap round, depth-3, PROFILE_SAMPLE M=1, leaf rollout dominant):

| source | cores | rate | doc |
|---|---|---|---|
| pilot (GPU-box slice) | 13.6 | 14.6 samples/s = **52.5k/h** | REBEL_SAMPLEGEN_PILOT.md |
| RTX PRO 4500 box | 27.2 | 33.6 samples/s = **121k/h** | REBEL_SAMPLEGEN_RESULTS.md |
| Contabo (scaled, ~1.07 samples/s/core) | ~10 free | **~35–40k/h** (oversubscription-degraded; treat as upper bound) | scaled from pilot |

Per-sample stage costs: tree build ~0 s; **leaf rollout 0.5 s (M=1) → 2.1 s (M=5) PROFILE_SAMPLE,
5–20.6 s BEST_RESPONSE**; CFR solve ~0.05 s (150 iters). Post-net rounds are bounded by build+solve
≈ **0.1 s/sample** → ~36k/h/core — the self-play loop accelerates ~10–40× once a net exists
(bootstrap-then-accelerate plan, REBEL_SAMPLEGEN_RESULTS.md).

Realistic budget for a "beats-k200-or-not" net (NOT paper grade):

- **Bootstrap:** 2M samples at M=4 leaves (~2 s/sample): Contabo background ≈ 5–6 days; a 27-core
  CPU pod ≈ 40 h (~$15–25 at RunPod CPU rates). The existing 2.1M shards are **inflated-ante-stale**
  (§1b reason 3) — reusable for architecture probes only, NOT for a deployable net.
- **Accelerated rounds:** 2–3 self-play rounds × 10M net-leaf samples ≈ 23 h each on 12 Contabo
  cores, or ~10 h on a 27-core pod. ReBeL's poker results used ~2–3 value-net iterations beyond
  bootstrap at our scale analog.
- **Net training:** trivial — the 38k-param probe trained in 18.6 s on the 4090; a 1–2M-param net on
  20M samples is ~1–3 GPU-h/round (~$1–2 on the $0.34/h 4090).
- **Total full build:** order **1–2 weeks wall, $50–150 pod**, engineering dominant (regen pipeline
  on real-ante blueprint + 6-vector/belief sample schema + gadget integration + gate battery:
  realistically 2–4 sessions of careful work).

**Literature anchor (why "not paper grade" is the only sane target):** ReBeL's superhuman HUNL run
used up to 128 machines × 8 GPUs for data generation and 90 DGX-1 (8×V100 each) for training, 1,750
epochs × 2.56M examples (Brown, Bakhtin, Lerer & Gong, *Combining Deep Reinforcement Learning and
Search for Imperfect-Information Games*, NeurIPS 2020, arXiv:2007.13544; figures per NVIDIA's
technical writeup, https://developer.nvidia.com/blog/facebooks-ai-reinforcement-learning-model-outmatches-competitors-in-poker-new/)
— **4–5 orders of magnitude above this project's budget, for a 2-player game with a convergence
theorem we don't get in 6-max.** The cheaper precedent closer to our architecture is DeepStack
(Moravčík et al., *Science* 2017): counterfactual-value nets trained on ~10M solved random turn
situations + ~1M flop situations — i.e., the same "solve random PBSs, regress values" recipe our
samplegen already implements, at a sample count (~11M) that IS within a pod-week here. Pluribus
(Brown & Sandholm, *Science* 2019) is the existence proof that depth-limited search with
*blueprint-continuation* leaves works in 6-max without equilibrium guarantees — but note Pluribus's
leaves let opponents choose among k continuation strategies (exactly our BEST_RESPONSE leaf mode,
which at M=5 cost 20.6 s/leaf-eval — the thing the net amortizes).

## 4. Live-latency budget on Contabo (<1 s/decision)

Measured components (all in-repo):

| component | measured | source |
|---|---|---|
| subgame CFR solve, d=3 (~150–220 nodes), K=1000 iters | **0.109 s** | DECISIONS.md "vanilla full weighted traversal" (Stage 3-C) |
| solve, K=150 | ~0.05 s | REBEL_SAMPLEGEN_PILOT.md |
| leaf rollout (the resolver's killer) | 0.5–20.6 s; **deployed-config p95 82–99 s, 14.5% >15 s** | pilot; STATUS.md Floor lock (J3) |
| state-prep per step | 0.9 ms (0.30 ms fast-path gate) | DECISIONS.md best-response leaf entry |
| value-net leaf eval (replaces rollout) | one batched CPU forward, ~40–220 leaves × ~1.25k dims through ~1–2M params ≈ **2–10 ms** | §2 sizing; standard CPU MLP throughput |
| belief propagation per re-solve | blueprint policy queries for k buckets × O(10) public actions, batched ≈ 2,000 forwards of the 236-dim encoder net ≈ **10–50 ms** | derived |

**Budget arithmetic:** net-leaf re-solve ≈ tree build (~0) + beliefs (≤50 ms) + K×(traversal+leaf
forwards). At K=1000 with cached leaf values re-predicted only when beliefs update every ~10 iters:
**~0.2–0.4 s typical**. Against the <1 s bar that leaves 2.5–5× headroom for Contabo's ~10×
oversubscription variance — tight at the tail; mitigations: K adaptive on wall-clock (solve-until-
deadline), gate (X0 measured ~37% solve rate; blueprint passthrough otherwise), depth 3 only, and
the existing `FallbackWatchdog` pattern as the hard floor. Depth 4–5 multiplies leaves ~5–10× —
still ms-scale leaf cost with a net, but traversal grows; d=4 likely fits, d=5 marginal.

**Conclusion:** latency is **not** the binding risk. The resolver was unshippable because of rollout
leaves; a value net removes precisely that term. DeepStack's reference point — ~5 s/action on a 2017
laptop GPU with re-solve depth to street end — bounds us from above; our trees are far smaller
(discretized 7-action menu, depth 3).

## 5. Kill-risks, ranked honestly

1. **Wrong lever (highest).** X0: a resolver with a *correct* opponent model lifted ≈ 0 vs
   blueprint-alone. The measured binding leak (KillPhil diag) is *opponent-range-dependent* —
   beating it requires knowing THIS opponent's range, which self-play beliefs (opponents assumed
   blueprint-like) cannot supply, and the anonymity constraint forbids persistent per-opponent
   priors. A ReBeL-class agent is anonymity-compatible (beliefs are within-hand public-action
   inference) but inherits the panel-average range assumption — the exact failure that made the
   static fix net-zero. Search polish on top of an already-near-Nash-within-abstraction blueprint
   may be worth ~0 on this panel. *This, not leaf precision, is the most likely way a rebuild burns
   weeks for a null.* The a3 finding strengthens it: the killphil hole is depth-flat policy defect,
   feeding H2 (league training), not search.
2. **Leaf-value precision/underfit — the resolver's killer, partially answered.** Why a learned PBS
   net could beat rollout leaves: (i) Gate-1's negative control showed even *exact* static leaves
   are unsafe (219 mbb/g) — the failure was architectural (leaves that don't respond to the
   re-solve), and ReBeL's leaves are **belief-conditioned functions re-evaluated as beliefs update
   within the solve** (ReBeL Alg. 1), restoring the opponent's ability to "best respond" at the
   leaf; (ii) regression over millions of solved PBSs averages away the M=1 rollout noise that
   capped labels; (iii) the 2-player endgames this format collapses to can additionally use the
   validated AugGadgetCFR safe-resolve. Why it could still fail: the only trained probe hit **R² =
   0.237** — if the ceiling survives the three diagnosed fixes (belief block, 6-vector targets, M>1
   labels), the PBS→value signal under k=200 is too weak and the net's *systematic* bias re-creates
   the J2 failure. **This is exactly what the cheap probe (§6 P1) measures before any commitment.**
3. **No 6-max theory.** ReBeL's convergence proof is 2-player zero-sum. 6-max has no safety theorem;
   the product-factorized belief is an approximation; empirical gating (cash-rate panels + attacker
   battery) is the only correctness instrument — and e1 notes that instrument is abstraction-blind.
4. **Distribution-validity churn.** Precedent: one ante-convention bug silently invalidated 2.1M
   samples and the prior ship candidate. The net is glued to (blueprint sha, abstraction sha, game
   conventions); any blueprint retrain (C3, H2 league) staleness-cascades into full regen. A
   ReBeL-class layer makes every future blueprint change ~2× more expensive.
5. **Latency tail on oversubscribed Contabo (lowest).** §4: manageable with deadline-bounded K +
   gate + watchdog; would not kill the project, only clip solve quality on bad minutes.

## 6. Staged plan, go/no-go gates, cost table

Default at every gate is NO-GO (the program's active levers — H2 league probe, adaptive layer, H4
field model — attack risk #1 directly; this track only re-opens if they show the static-policy /
representation ceiling binds).

| stage | what | gate (pre-registered) | cost Contabo | cost pod |
|---|---|---|---|---|
| **P0** | this memo | — | done (0 compute) | — |
| **P1 — architecture probe (zero new samples)** | rebuild net inputs from the existing 2.1M stale shards: empirical belief block (shards over same public node = empirical belief, per REBEL_SAMPLEGEN_RESULTS.md) + per-seat 6-vector targets where derivable; retrain rough net | held-out **R² ≥ 0.5** (vs 0.237 scalar/no-belief) AND per-seat MAE beats the scalar net's perspective-bugged baseline. Below 0.4 → **dead, close a2 permanently** | ~2–4 h idle cores (Addendum-5 night shift class) | not needed |
| **P2 — real-ante regen probe** | 300–500k samples, current `k200_real_ante` blueprint, M=4 PROFILE_SAMPLE, 6-vector + belief schema | R² from P1 reproduces (±0.05) on the live distribution | ~12–30 h background | ~$5–10, ~8 h |
| **P3 — mini Gate-2 (the resolver post-mortem rematch)** | depth-3 search + net leaves vs blueprint-alone; paired CRN, 1k hands × {killphilmtt, STATION, mixed}; latency instrumented | (a) killphil Δ ≥ +0.03 ICM/hand outside noise **without** a STATION giveback (the static-fix failure signature); (b) action p95 < 1 s on Contabo. Fail either → close | ~1–2 days | ~$10 |
| **P4 — full build** (only past P3) | real-ante bootstrap 2M + 2–3 net-leaf self-play rounds (10–30M), gadget at 2-player endgames, full gate battery + attacker re-measure | beats blueprint-alone on the full 24-profile panel + attacker non-regression (floor rule absolute) | weeks (not recommended) | ~$50–150 + 2–4 sessions engineering |

Cumulative probe cost to a decisive answer: **P1+P2+P3 ≈ 3–5 days mostly-background CPU + <$20**,
vs the resolver track's multi-week sunk cost — the probes front-load exactly the two quantities
that were never measured (belief-conditioned value signal; panel-asymmetric EV).

## 7. Verdict

**Not worth a build; worth exactly one zero-new-compute probe (P1) queued as idle-time work, and
nothing past it unless BOTH P1's R²-gate clears AND an independent program result (a1 blur-map or
H2's league probe) shows the static-policy/representation ceiling — not opponent-range mismatch —
is what binds.** The evidence that would flip this to an active track: P3 showing a belief-
conditioned net moves the killphil-class leak *asymmetrically* (no STATION giveback) — the precise
test every static approach has failed — at <1 s p95. The evidence that buries it: P1 R² < 0.4, or
H2 closing the shove-defense hole by league training alone (which would confirm risk #1: the leak
never needed search).

---

### Sources

In-repo: `docs/REBEL_PBS_6MAX.md`, `docs/REBEL_GATE1_FINDINGS.md`,
`docs/REBEL_GATE2_TONIGHT_AND_TOMORROW.md`, `docs/REBEL_KILLPHIL_DIAG_FINDINGS.md`,
`docs/REBEL_SAMPLEGEN_PILOT.md`, `docs/REBEL_SAMPLEGEN_RESULTS.md`,
`docs/REBEL_OVERNIGHT_PHASE1_REPORT.md`, `docs/STATUS.md` (Floor lock), `docs/DECISIONS.md`
(resolver retirement ~L520, foundation pivot ~L543, Stage 3-C solver timing ~L363, ghost-ante
~L680, deployment reversal ~L1895), `docs/research_program/RESEARCH_MAP.md` (a2, a3, e1),
`evals/X0_resolver_vs_self.json`.

Literature: Brown, Bakhtin, Lerer, Gong, NeurIPS 2020, [arXiv:2007.13544](https://arxiv.org/abs/2007.13544)
(ReBeL; compute figures via [NVIDIA technical blog](https://developer.nvidia.com/blog/facebooks-ai-reinforcement-learning-model-outmatches-competitors-in-poker-new/));
Moravčík et al., *Science* 2017 (DeepStack — CFV nets from ~10M turn / ~1M flop solved situations;
~5 s/action re-solve on a GTX 1080); Brown & Sandholm, *Science* 2019 (Pluribus — 6-max depth-limited
search, multi-continuation leaves, no equilibrium guarantee); Burch, Johanson, Bowling, AAAI 2014
(CFR-D / re-solving gadget); Brown & Sandholm, NeurIPS 2017 (safe & nested subgame solving);
Ganzfried & Sandholm 2013 (pseudo-harmonic action translation); Schmid et al., *Science* 2023
(Student of Games — unified sound search, 2-player-zero-sum scope caveat applies equally).
