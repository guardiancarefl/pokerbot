# Q0 RESEARCH QUEUE — correctly converting field data into a winning policy

The lab grows on the FREE Contabo bench. The pod is rented muscle for
training-RUNS only, in dense bursts. **Most of this queue is free** (read-only
analysis or no-gradient overlays) and may resolve Q0 — or kill methods — before
any pod is rented. Only the TIER-2 training methods need the burst.

Gates are the canonical block verbatim (`BURST_PLAN.md` G1–G4). No softening.

---

## TIER 0 — FREE read-only probes (run on Contabo NOW, like C)
These are pure analysis. They sharpen or pre-empt every training method, and a
couple could answer Q0 outright.

- **C — data-as-targeting** ✅ DONE. Headroom 0.087/spot (4.3× R6), DEFENSIVE
  (96% over-calling shoves in BB at 11–15bb). The leak map exists.
- **C2 — reachability / self-anchor-cost map** ★ HIGHEST FREE EVoI. For each
  jam-wall cluster C found, compute the EV cost of applying the oracle
  (fold-more) AGAINST THE CHAMPION'S OWN shove range (the self-anchor price).
  Read-only (oracle + champion, both on disk). **If the fold-more clusters carry
  LOW self-anchor cost → a context-gated fix (B/F) is essentially free to make
  safe; if HIGH → the exploit is fundamentally entangled and bounds every
  method.** This is the missing number that tells us, before training, whether
  ANY method can capture the headroom self-anchor-safely.
- **C3 — pool-shove fidelity vs the real 517** ★. The oracle is vs the POOL's
  shove range. Compare the pool's jam ranges to the OBSERVED real shoves (517
  hands / FIELD_DOSSIER). If faithful → the exploit transfers to humans; if not →
  we'd be exploiting a synthetic artifact (dispositive for whether to burst at
  all). Read-only.
- **C4 — over-call decomposition by hand class** (a3-style). WHICH hands does the
  champion over-call shoves with (BB, 11–15bb)? Sharpens the fix and the eval.
  Read-only.

## TIER 1 — near-free, NO gradient (Contabo)
- **F — oracle-table overlay** ★ may be the simplest correct method. The oracle
  best-response call/fold thresholds vs the pool are ALREADY computed
  (`battery_v1.json`). Overlay them as a context-gated lookup at jam-wall spots,
  champion untouched elsewhere. **Why train at all if we already know the best
  response?** Self-anchor-safe by the same gate as B (apply only vs field
  context). Tests "is the headroom capturable by direct application, no training."
  Gated by C2 (must be self-anchor-safe) + C3 (must be real).

## TIER 2 — training methods (need the pod burst)
Run as ONE dense burst once ≥3 are training-ready.
- **D — jam-wall-only training** ✅ BUILT. Restricts learning to facing-shove
  infosets; tests whether the imperfect-pool dimensions were the H4 poison.
- **B — exploit residual head** 🔨 BUILDING. Frozen champion + context-gated
  additive head; self-anchor-safe by construction. The predicted winner — and C
  strengthened it (the exploit is a context-specific defensive correction, B's
  native shape).
- **A — explicit-restriction RNR** (fallback). λ-anchor to frozen champion.
  C gives it a sharp test: the anchor pulls toward champion (folds LESS vs field)
  while the exploit needs fold-MORE — can any λ hold both?
- **E — output-head-only fine-tune**. Cheap partial-restriction control.
- **G — distributionally-robust field BR**. BR to a confidence SET around the
  field model (robust to pool imperfection) instead of the point estimate.
  Hedges the C3 risk. Build only if C3 shows fidelity gaps.
- **H — importance-weighted real-hand fine-tune**. Train directly on the 517 real
  hands (importance-weighted) at the jam-wall, bypassing the synthetic pool.
  Tiny/high-variance, but the most direct use of the real data. Cheap probe.

## TIER 3 — frontier / synthesis (design → cheap probe first)
- **I — bbnorm encoder organ for jam-defense**. v2's depth-invariant encoder paid
  EV at ≤6bb (a3) but the leak is at 11–15bb. Read-only: does the bbnorm
  representation separate the jam-wall clusters better than the 236-d legacy?
  If yes, an organ transplant could REPRESENT the defense the legacy encoder blurs
  (composes with B/D). One-delta discipline: probe representation only.
- **J — novel objective: min-self-anchor-regret s.t. field-gain ≥ X**. Flip RNR's
  framing — minimize blueprint deviation subject to capturing a floor of headroom,
  instead of maximizing headroom with a soft anchor. Design-stage.
- **K — explicit champion+specialist ensemble (MoE)**. A small jam-defense
  specialist trained only on C's cluster, hard-gated against the champion by
  context. B's two-model cousin; useful if B's single-head leakage is hard to
  bound.

---

## Sequencing discipline (cheapest-first, free-before-paid)
1. **Run TIER 0 free now** (C2, C3, C4) — they may answer Q0 or kill the burst.
2. **F (TIER 1)** if C2/C3 green — a no-training candidate answer.
3. **Build the TIER-2 survivors** (D done, B building; A/E/G/H as warranted).
4. **ONE pod burst** when ≥3 training-ready methods exist — run all in parallel,
   terminate. Coordinator G-sum ≤ 24.
5. **TIER-3** frontier probes interleave on the free bench; only their training
   escalations join a later burst.

**Burst trigger (for the operator):** I will flag "queue worth a burst" when ≥3
TIER-2 methods are training-ready AND the TIER-0 gates (C2 reachability, C3
fidelity) haven't pre-empted them. Until then everything runs free.

## Operator gates (unchanged)
pod-spend · escalate probe→program · ship/arm/deploy · weaken-a-gate — all yours.
The bench (TIER 0/1, builds) runs autonomously and free.
