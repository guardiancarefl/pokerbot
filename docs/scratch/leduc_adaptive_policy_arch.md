# Leduc adaptive-policy net — Step-2 architecture (APPROVED 2026-05-31)

**Status:** architecture APPROVED 2026-05-31. Required additions (a–d) from
the approval message are integrated below:
- §3 — running-stats no-leakage confirmation
- §8 — information-leakage invariant + tested test plan (load-bearing)
- §9 — Step-4 eval plan: per-cell headroom-normalized gain + opp-model
       accuracy as a measured diagnostic

Step-3 (training spec) APPROVED 2026-05-31 with two corrections + locked
thresholds — see §12. Concrete leash VALUES (β, K, λ, gate shape) are
PROPOSED in §12 and awaiting user sign-off; do not build against them until
signed. No code written yet.

---

## 1. What this net is

A small sequence model that takes the within-match history of hero AND
opponent decisions and outputs (a) hero's policy at the current decision
point, regularized toward the CFR+ Nash anchor for safety, and (b) a
prediction of opponent's next-action distribution as an auxiliary task. The
within-match adaptation mechanism is the encoder's attention over the
opponent's action tokens plus an explicit running-frequencies summary fed
into the policy head.

The Step-4 pass bar (per addition d, §9) is reported in HEADROOM-NORMALIZED
terms per cell, using the locked Step-1 denominator
(`runs/leduc_headroom_20260531_160608/headroom.json`, commit `3584d49`).

---

## 2. Token format

One token per **decision point** in the match — hero's AND opponent's. Chance
events (cards) are encoded into the next token's public-state fields rather
than emitted as separate tokens (keeps sequences short). Hand boundaries are
marked by a flag bit, not a separator token.

Per-token raw feature vector, F_raw = **17 dims**:

| group | dims | content |
|---|---:|---|
| actor identity            | 1 | 0=hero, 1=opp |
| street one-hot            | 2 | preflop, flop |
| public card one-hot       | 3 | J, Q, K (zeros preflop) |
| facing-bet flag           | 1 | 1 if a bet is currently faced, else 0 |
| raises-this-street        | 2 | one-hot in {0, 1} (cap=2; "2" means raise illegal) |
| hero private card one-hot | 3 | J/Q/K — **non-zero ONLY if actor=hero** (opp's hand is private) |
| action taken              | 3 | one-hot fold/call/raise |
| hand-boundary flag        | 1 | 1 if this is the first decision of a new hand |
| (intentional pad to 17)   | 1 | reserved (zero) — future: betting position |

Notes:
- Hero's private card is part of the token ONLY at hero decisions; at opp
  decisions, those 3 dims are zero. Opp's private card has no slot
  anywhere in the token (load-bearing — see §8).
- Public card is identical for both actors after the flop (it's public).
- "raises-this-street" as the count BEFORE the action lets the model know
  whether raise was legal at this decision (cap=2).
- The action_taken field at the CURRENT hero decision (the one being
  predicted) is masked/absent; we use a query-token convention — see §4.

---

## 3. Running opponent-summary features (auxiliary, not in the token stream)

Six scalar features, F_stats = **6 dims**, computed cumulatively over all
prior opp decisions in the match (with mild Laplace smoothing, +1/+|legal|,
so a freshly-started match isn't a divide-by-zero):

1. opp fold rate when facing a bet (denominator = opp decisions where fold was legal)
2. opp call rate when facing a bet
3. opp raise rate when facing a bet
4. opp aggression factor = raises / (calls + raises) over all opp decisions
5. opp open-raise rate (rate of raise when not facing a bet, i.e. opening)
6. log(1 + n_opp_decisions) / log(1 + 100)  — confidence weight, capped near 1

**No-leakage confirmation (per addition c):** every one of these 6 features
is a function of OBSERVED OPP ACTIONS ONLY (and the public state context
needed to define "facing a bet" / "opening"). NONE of them references opp's
private hole card; opp's card is never observed by the model during a hand
(it's revealed only at showdown, which is a terminal state, not a decision
point in the token stream). Test C in §8 verifies this empirically by
counterfactual substitution of opp's hole card.

These give the policy head an explicit prior on opp tendencies even when the
token context is short or the match is long. The transformer alone could
learn the same statistics from raw tokens, but providing them explicitly is
a strong sample-efficiency prior and lets long matches degrade gracefully
once the token window is full.

---

## 4. Network architecture

**Type:** `nn.TransformerEncoder` (small). LSTM is an acceptable fallback if
transformer-on-Leduc proves unstable at this size, but I expect the
transformer to behave fine — Leduc tokens are tiny and sequences are short.

**Sizing:**
- d_model = **64**
- nhead = **4** (head_dim = 16)
- num_layers = **2**
- dim_feedforward = **128**
- dropout = **0.0** (deterministic; Leduc is small, no overfitting room — if
  it does overfit, raise to 0.1)
- max_seq_len = **128** tokens (covers ~25–40 Leduc hands depending on
  action density; long matches use a sliding window + running-stats prior)
- Positional encoding: **learned**, shape [max_seq_len, d_model]

**Match vs hand tokenization:** one continuous sequence per match. The
hand-boundary flag is the only intra-match structure; the transformer learns
how to use it. Cross-hand attention is the whole point — within-match
adaptation requires it.

**Forward pass:**
1. `x = token_proj(tokens)` — `Linear(17, 64)` → [B, T, 64]
2. `x = x + pos_embed[:T]` — learned positional encoding
3. `h = encoder(x, src_key_padding_mask=pad_mask)` — [B, T, 64]
4. `repr = h[:, current_idx]` — take the hidden state at the "current
   decision" position (we append a **query token** with actor=hero,
   action=zero at the end of the sequence; its hidden state is the
   prediction context). Shape: [B, 64]
5. `s = stats_proj(opp_stats)` — `Linear(6, 32)` → [B, 32]
6. `combined = concat([repr, s], dim=-1)` → [B, 96]
7. Two heads (§5).

**Approximate parameter count (~84k total):**
- Input proj 17→64: ~1.1k
- Positional emb 128×64: 8.2k
- Per transformer layer: ~37k (self-attn ~20k + FFN 64↔128 ~16.5k + LN)
- 2 layers: ~74k
- Stats proj 6→32: ~0.2k
- Policy head 96→3: ~0.3k
- Opp-model head 96→3: ~0.3k
- **Total: ~84k params** (~340 KB at fp32). Trains in seconds on Contabo CPU.

---

## 5. Heads

**Policy head (load-bearing):**
- `logits = Linear(96, 3)` → mask illegal actions to `-inf` → softmax → [B, 3]
- This is hero's action distribution at the current decision point.

**Opponent-model head (auxiliary loss in training, MEASURED DIAGNOSTIC in eval):**
- `opp_logits = Linear(96, 3)` → masked softmax → [B, 3]
- At **every opp decision point** in the sequence, predict opp's action
  distribution. Trained via cross-entropy against the observed opp action.
  Forces the encoder to build a useful opp representation, which should
  help generalization to held-out cells.
- In Step-4 eval the head's **prediction accuracy is reported per cell**
  (see §9, addition b) — it is evidence FOR WHY exploitation works:
  high accuracy on a cell = the net genuinely learned to read it;
  accuracy holding on held-out cells = the reading generalizes (not
  memorized).

Both heads share the encoder; the opp head is an aux loss with a small
weight (e.g. λ≈0.3 in the Step-3 phase-2 spec; final weight TBD with that
spec).

---

## 6. How opp context enters the policy (the adaptation mechanism)

Two paths, both wired into the policy head input:

1. **Encoder attention over opp action tokens.** Each opp action emits a
   token with actor=opp, action one-hot, and current public state. The
   transformer attends over these; the query-token's hidden state at the
   current hero decision aggregates this evidence. Short matches rely on this
   path (every opp action is visible to attention).

2. **Explicit opp_stats vector.** The six cumulative bucket rates plus the
   confidence weight are concatenated to the encoder output before the
   heads. Long matches (token sequence saturates the 128-token window) still
   carry the long-tail history through these stats, so the model degrades
   gracefully rather than forgetting the early hands.

Both paths see the SAME observed opp actions; the redundancy is intentional —
attention is expressive but sample-hungry, stats are crude but sample-cheap.
At Leduc scale we want both. (At 6-max we'll re-evaluate whether the explicit
stats are still load-bearing.)

---

## 7. End-to-end shapes (one forward pass, batch size B)

Input:
- `tokens`:    [B, T, 17]   T ≤ 128
- `pad_mask`:  [B, T]       True at padded positions
- `legal_mask`:[B, 3]       True at legal actions at the current decision
- `opp_stats`: [B, 6]

Computation:
- token_proj    → [B, T, 64]
- +pos_embed    → [B, T, 64]
- encoder(...)  → [B, T, 64]
- gather query  → [B, 64]
- stats_proj    → [B, 32]
- concat        → [B, 96]

Output:
- `policy_probs`:    [B, 3]  (masked softmax)
- `opp_pred_probs`:  [B, 3]  (masked softmax; only meaningful at opp decisions)

---

## 8. INFORMATION-LEAKAGE INVARIANT (load-bearing correctness property)

**This is the single most important correctness property of the entire
proof.** A leak invalidates every exploitation number. The invariant is
enforced by token construction AND verified by automated tests. Both must
hold before any training run.

### 8.1 Invariant statement

> At any model PREDICTION position — either (i) the policy head's query
> token (hero's current decision) or (ii) any opp-model head prediction
> position (an opp decision in the sequence) — the inputs reaching the
> model (the token stream up to and including that position, plus the
> opp_stats vector) contain **ZERO information about ANY opponent's
> private (hole) card** at any point in the match.
>
> The only private-card information ever encoded in the model's inputs is
> the HERO's private card, and only in tokens at HERO's own decision
> points. Opp's hole card has no slot in the token format and never
> influences the opp_stats vector.

### 8.2 Why this is plausible by construction

- The token format (§2) reserves the private-card slot for HERO ONLY; at
  opp decisions, those 3 dims are zero. Opp's card has no token slot.
- Opp's card is observed only at showdown (terminal state, not a decision
  point). Terminal states are NOT in the token stream — the stream ends at
  the last decision before showdown.
- The opp_stats vector (§3) is a function of OBSERVED OPP ACTIONS ONLY.
- The public card (flop) is symmetric — both players see it; it is not
  private information for either.

But "plausible by construction" is not enough. The invariant must be
TESTED (per addition a).

### 8.3 Test plan (required; must pass before any training)

All three tests live in `tests/test_leduc_token_no_leak.py`. The build of
the Step-3 training code is BLOCKED until these tests are written, run,
and passing.

**Test A — opp-private-card-slot-zero (necessary, easy).**
Generate N≥100 random Leduc match trajectories. For every token position
in every match, check:
- `actor == opp` ⇒ the 3-dim hero-private-card slot is all-zeros.
- No token field encodes opp's hole card (the build only allocates slots
  to documented fields; this test pins that down by hashing the token-byte
  layout against a fixed manifest).
- The hero-private-card slot at hero positions matches the hero's actual
  card.
Failure mode: a wiring bug stuffed opp's card into a token field. Fails
loudly via `pytest` with a precise position index.

**Test B — counterfactual-equivalence (load-bearing; the test that PROVES
no leakage by construction).**
For each sampled match transcript M = (public_card, hero_card, opp_card,
hero_actions, opp_actions, actor_order):
1. Build the token tensor `T(M)` and the opp_stats vector `S(M)`.
2. For every legal alternative opp_card' in the deck (consistent with
   public_card already drawn), construct M' = M with opp_card replaced by
   opp_card' and EVERYTHING ELSE identical (same public card, same hero
   card, same actions in the same order — actions are public history, so
   this counterfactual is well-defined as a test input even though in real
   play the action distribution would differ; that's exactly what we DON'T
   want the model to see).
3. Build `T(M')` and `S(M')`.
4. **Assert bit-identity: `T(M) == T(M')` and `S(M) == S(M')`** at every
   position, for at least 1000 (M, opp_card_substitution) pairs.

If Test B fails, opp-card information is leaking into the model's inputs
through SOME token field or stats feature; the build is incorrect. If Test
B passes, the model's outputs at every prediction position are PROVABLY
not a function of opp's hidden card via its inputs.

**Test C — running-stats counterfactual (subsumed by Test B but reported
separately for diagnostic clarity).**
Under the same M → M' substitution as Test B, assert `S(M) == S(M')` bit-
identically. Confirms the 6 stat features are action-frequency only (as
documented in §3).

### 8.4 Failure protocol

If any of Tests A/B/C fails: **DO NOT TRAIN.** Halt. The token-construction
code or stats-construction code has a leak and must be fixed before any
exploitation number is meaningful. Log the failing position(s) + a
minimal-repro transcript. Failures are blockers, not warnings.

---

## 9. Step-4 eval plan (per addition b, d)

### 9.1 Headroom-normalized exploitation-gain metric (per cell)

For each (archetype, strength) cell c, with values in mbb/g:

```
anchor_EV(c)   = (locked Step-1 baseline.json; from anchor vs cell)
BR_EV(c)       = (locked Step-1 headroom.json; from BR vs cell)
adaptive_EV(c) = (measured in Step-4; from the trained net vs cell)
headroom(c)    = BR_EV(c) - anchor_EV(c)
gain_frac(c)   = (adaptive_EV(c) - anchor_EV(c)) / headroom(c)
```

`gain_frac(c)` ∈ [0, 1] for a well-behaved adaptive policy (0 = no
improvement over anchor; 1 = matches BR's exact exploitation). May go
slightly negative if the adaptive policy hurts vs a cell — that's a real
signal of mis-adaptation.

### 9.2 Per-cell reporting (REQUIRED — not just an aggregate)

Step-4 must report `gain_frac` PER CELL, not as a single mean. Aggregates
hide the structure that makes the pass bar interpretable:

- **High-headroom EXPLOITATION cells (the headline test):**
  `always_raise@s1.00` (+1016 mbb/g headroom) and the two `random_uniform`
  cells (+833 mbb/g). These are where adaptive should show LARGE absolute
  gains AND large gain_frac. If gain_frac < 0.3 here, exploitation isn't
  working.
- **Held-out (TEST) GENERALIZATION cells:** `always_fold@s0.5` (+608),
  `over_folder@s0.5` (+550), `over_caller@s0.5` (+488). Same metric, never
  seen in training. If train-cell gain_frac >> test-cell gain_frac, the
  net memorized. If both are comparable, the reading generalizes.
- **Low-headroom FLOOR cells:** `always_fold@s1.00` (+214 — structural cap
  because BR vs always-fold maxes at 1 chip = the blind). A small
  absolute gain here is EXPECTED and FINE; do not treat it as failure.
  This is why per-cell reporting matters.

The PASS-BAR THRESHOLD on `gain_frac` (and the exploitability-safety bar
on the other half of Step-4) is **TBD by user**, to be set when the Step-3
training spec is approved.

### 9.3 Opp-model-head accuracy reported per cell (per addition b)

Alongside `gain_frac`, Step-4 reports the **opp-model head's prediction
accuracy per cell**, computed as:

```
opp_acc(c) = mean over opp decisions in matches vs c of:
             1 if argmax(opp_pred_probs) == observed_opp_action else 0
```

Also report the **mean KL divergence** between predicted and true opp
action distributions per cell (the archetype's true distribution is known
in closed form — it IS the archetype function). KL is more informative than
accuracy for mixed strategies like `over_caller`.

Reporting is **per cell, train AND held-out:**
- High `opp_acc` / low KL on exploitable archetypes = the net genuinely
  learned to read them (mechanism evidence for WHY exploitation works).
- Held-out cell accuracy/KL ≈ train cell accuracy/KL on the same
  archetype family = the reading generalizes (not memorized).
- The opp head's per-cell report makes it a MEASURED diagnostic in the
  Step-4 verdict, not decoration.

---

## 10. Step-3 sketch (OUT OF SCOPE here, context only)

The Step-3 training spec will be a separate proposal for review. Sketched
here only so §1–§9 don't paint Step 3 into a corner:

- **Phase 1 — distill to anchor.** Cross-entropy of policy_probs against the
  CFR+ anchor's distribution at every hero decision point in self-play
  sequences. Zero opp-model loss (no real opp present in self-play).
  Output: a net that ≈ matches the anchor's strength when there's no read.
- **Phase 2 — exploit-shift, regularized.** Play matches vs the TRAIN
  archetype cells (the 9 of 12 from `train_test_split()`). Policy gradient
  (REINFORCE or PPO-clip) to maximize chip-EV; opp-model aux CE against
  observed opp actions; KL-anchor regularizer `β · KL(policy || anchor)`
  keeps exploitability bounded near the anchor (the safety half of the
  Step-4 pass bar). β schedule + RL algo + opp-loss weight + confidence
  signal use are all TBD with the Step-3 spec.

---

## 11. What's approved here / what's still pending

**APPROVED 2026-05-31 (user, this message):**
- §2 token format (17-dim)
- §3 running-stats vector (6-dim) + no-leakage confirmation
- §4 transformer 2L/4H/64d/~84k-params, learned posenc, 128-token window,
  query-token convention
- §5 policy + opp-model heads from concat(encoder_out, stats_emb) → 96 → 3
- §6 dual adaptation mechanism: attention + explicit stats
- §7 end-to-end shapes
- §8 **information-leakage invariant + Tests A/B/C as a build blocker**
- §9 Step-4 eval plan: per-cell headroom-normalized gain + opp-model
  accuracy/KL diagnostic
- LSTM as a fallback if transformer training is unstable at this size

**APPROVED 2026-05-31 (Step-3 follow-up message) — see §12:**
- Step-3 two-phase training spec (phase-1 distill + exit gate, phase-2
  PPO + confidence-gated blend + KL-anchor leash, opp-aux loss)
- Two corrections locked: (1) the blend gate is keyed to a non-equilibrium
  READ signal from the opp-model head, NOT raw decision count; (2) E* is a
  PASS REQUIREMENT, not a diagnostic.
- Step-4 pass-bar thresholds locked: τ_distill ≤ 5 mbb/g; E0 ≤ anchor+2
  (≤~2.13); E* ≤ 50 mbb/g; gain_frac ≥ 0.40 on the 3 high-headroom cells;
  held-out gain_frac ≥ 0.5× train.

**PENDING user sign-off (this proposal, §12.5):**
- Concrete leash values: β_base, β_target, K, λ, and the exact gate shape
  g(read_confidence). Build is blocked until these are signed.

**EXPLICITLY NOT IN THIS APPROVAL:**
- The network code (not written yet — blocked on Step-3 approval)
- Test A/B/C code (will land alongside the network code; design is fixed
  here but implementation is a Step-3-build deliverable)
- Any training run (blocked on Step-3 approval + Tests A/B/C green)

---

## 12. Step-3 two-phase training spec (APPROVED 2026-05-31, with corrections)

Supersedes the §10 sketch. Thresholds in §12.4 are LOCKED. Leash values in
§12.5 are PROPOSED and awaiting sign-off — DO NOT BUILD against §12.5 until
the user signs.

### 12.1 Phase 1 — distill to anchor (+ exit gate)

- **Data:** self-play only; both seats play the frozen CFR+ tabular anchor.
  No archetype cells appear (no test-cell contact). Confidence/read signal
  is ~0 throughout (anchor-vs-anchor is an equilibrium read).
- **Loss:** at every hero decision, forward-KL distillation
  `L1 = Σ_t KL(π_anchor(·|I_t) ‖ π_net(·|I_t))` (mode-covering — reproduces
  the anchor's mixedness). Opp-model aux OFF (λ=0; no real opponent).
- **EXIT GATE (hard block on Phase 2):** freeze the net, induce its fixed
  Leduc strategy at empty context, compute best-response exploitability E0
  via the anchor's machinery. Require **E0(distilled) ≤ 5 mbb/g** (§12.4).
  Fail ⇒ Phase 2 BLOCKED. STOP and report E0 to user regardless.

### 12.2 Phase 2 — exploit-shift, leashed (trains on 9 TRAIN cells ONLY)

- **Algo:** PPO-clip (ε=0.2) over full-match episodes vs the 9 train cells,
  reward = terminal chip-EV (seat-averaged). REINFORCE+value-baseline is the
  approved fallback if PPO is unstable at this scale.
- **Output composition (the exact zero-evidence guarantee):**
  `π_final = (1 − g)·π_anchor + g·π_net_raw`, with the gate `g` keyed to a
  READ signal (§12.3, correction #1) and `gate(0)=0` enforced. At zero read,
  π_final = anchor bit-exactly.
- **KL-anchor leash:** reward penalty `β_eff · KL(π_final ‖ π_anchor)` per
  hero decision, `β_eff = β_base·(1−g) + β_target·g` (tight at no-read, loose
  at strong-read), with a global warmup over K updates.
- **Aux opp-model loss:** `λ · Σ CE(observed_opp_action, opp_pred_probs)` at
  every opp decision. NOTE: the opp-head is now load-bearing for the gate, so
  its calibration matters more than in the original sketch.
- **Discipline:** 3 held-out cells never instantiated in any rollout; Tests
  A/B/C in `tests/test_leduc_token_no_leak.py` GREEN before the first step.

### 12.3 The READ signal (CORRECTION #1 — gate keyed to read, not count)

The gate must open on a genuine NON-EQUILIBRIUM read, never on the clock.
Count enters at most as a secondary shrinkage cap.

```
per opp decision t:
  d_t  = KL( q_t ‖ π_eq(public_t) )      # opp-head pred vs anchor's action dist; "how non-equilibrium"
  w_t  = 1 − H(q_t)/log|A_legal,t|       # sharpness/reliability ∈ [0,1]; uncertain reads down-weighted
running read (evidence-weighted, prior-shrunk to 0):
  D̄    = ( Σ_t w_t·d_t ) / ( w0 + Σ_t w_t )
gate:
  g    = tanh( κ · D̄ )                   # gate(0)=0 exactly; bounded [0,1)
```

- Equilibrium-like opp ⇒ d_t≈0 ⇒ D̄≈0 ⇒ g≈0 ⇒ π_final = anchor.
- Uncertain/noisy head ⇒ w_t≈0 ⇒ shrunk ⇒ g small (no read ≠ many actions).
- Sharp, far-from-equilibrium read (always_fold, always_raise, random_uniform,
  over_caller …) ⇒ D̄ large ⇒ g opens ⇒ exploitation permitted.
- Count appears ONLY via the w0 shrinkage prior (the allowed "secondary cap").

### 12.4 LOCKED thresholds (Step-4 pass bar)

- **τ_distill** (Phase-1 exit gate): `E0(distilled) ≤ 5 mbb/g`. Hard block.
- **ε** (E0 safety / blend correctness): `E0 ≤ anchor + 2 mbb/g (≤ ~2.13)`.
  With gate(0)=0, E0 should ≈ anchor exactly; drift above ⇒ gate(0)≠0 bug ⇒
  fail loud.
- **E\*** (worst-case safety — REQUIRED, correction #2): `E* ≤ 50 mbb/g`.
  Measured over the adversarial probe set: the 12 cells' induced contexts +
  a short BR-search over opp action sequences, with construction bound
  `E* ≤ E0 + Δ(β, KL_max)`. By mixture convexity
  `KL(π_final‖anchor) ≤ g·KL(π_net‖anchor)`, so exposure scales linearly in
  g — the bound the 50 mbb/g bar leans on.
- The hard pass bar sits on **BOTH E0 AND E\*** (a policy safe at g=0 but wild
  at g=1 must fail).
- **gain_frac exploitation:** `≥ 0.40` on `always_raise@s1.00` and both
  `random_uniform` cells.
- **gain_frac generalization:** held-out `gain_frac ≥ 0.5 × train gain_frac`
  on comparable cells; corroborate with held-out opp-head accuracy ≈ train.

### 12.5 PROPOSED leash values (AWAITING SIGN-OFF — do not build yet)

See chat message 2026-05-31 for per-value justification. Values:
`β_base=1.0`, `β_target=0.1` (chip/nat), `K=200`, `λ=0.3`, `κ=1.0`,
`w0=2.0`. **SIGNED OFF 2026-05-31 as STARTING points.** Build order:
(i) network code, (ii) Tests A/B/C GREEN, (iii) Phase 1 + exit-gate, report
E0, STOP before Phase 2.

### 12.6 Result-reading guardrails (recorded 2026-05-31, post sign-off)

1. **κ=1.0 assumes a strong read ≈1 nat. MANDATORY pre-Phase-2 check:**
   measure the per-cell D̄ distribution (the distilled net's opp-head reads
   vs each train cell). If a clearly-exploitable archetype (esp.
   `always_raise@s1.00`) yields D̄ so low that g stays ≈0, the gate is
   mis-scaled → recalibrate κ BEFORE policy learning. "Gate never opens" is
   a failure that masquerades as safety — a no-exploitation result must NOT
   pass as a safety pass.
2. **w0=2.0 is regime-sensitive.** Fine for Leduc (long matches) but must be
   re-tuned at the 6-max SNG stage (short matches, few hands per opponent).
   Do NOT carry it forward blind.
3. **β_target=0.1 is the knob most in tension with the E*≤50 bar.** If
   Phase-2 shows good exploitation but E*>50, tighten β_target FIRST (before
   touching the gate). This tuning order is locked.
