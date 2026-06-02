# d=64 Ablation — Wide-Pool / Entropy-Floor Loss

Three-arm ablation distilling the candC_k200 Deep CFR blueprint at the small
diagnostic capacity (`d_model=64, num_layers=2, dim_ff=256`, ~125k params).
All measured on the committed 5-member ICM-panel floor (1000 hands/seat,
6 seats, paired CRN vs blueprint).

## Setup

| arm | loss            | pool                                          |
|-----|-----------------|-----------------------------------------------|
| A   | entropy-floor β | narrow 15bb (original)                        |
| B   | β=0             | wide stratified (`5352,2000,2000,1500` strata)|
| C   | entropy-floor β | wide stratified                               |

All arms share: seed 2026, 200 epochs × 15 steps, batch 256, lr 1e-3,
`λ_distill=1.0 / λ_aux=0.1`, `skip-g4-floor`, anchor candC_k200, abstraction
retrofit. The only intentional differences between B and C are the loss
term; A retains the old pool and is judged on its narrow depth range.

Panel JSONs in this dir:
- `icm_panel_pre_fix_d128_narrow.json` — the d=128 narrow-pool smoke that
  established the floor breach we set out to fix.
- `icm_panel_arm_B.json` — d=64, wide pool, β=0.
- `icm_panel_arm_C.json` — d=64, wide pool, entropy-floor loss.

(Arm A was panel-judged elsewhere as policy-level worse than B/C and was
not re-run here — see prior session notes.)

## Pooled deltas (student − blueprint, ICM points / hand)

| member       | pre-fix d=128 narrow | arm B (d=64 wide β=0) | arm C (d=64 wide β=loss) |
|--------------|----------------------|------------------------|--------------------------|
| blueprint    | −0.0163  σ 4.22      | −0.0143  σ 3.82        | −0.0169  σ 4.42          |
| minestacker  | −0.0089  σ 3.23      | −0.0023  σ 0.89        | −0.0044  σ 1.66          |
| killphil     | −0.0141  σ 4.34      | −0.0092  σ 3.02        | −0.0118  σ 3.83          |
| maniac       | −0.0208  σ 3.82      | −0.0277  σ 5.37        | −0.0284  σ 5.54          |
| nit          | −0.0021  σ 0.48      | +0.0001  σ 0.02        | −0.0052  σ 1.24          |

Floor gate (σ≥2 ⇒ member breach, plus per-seat rules): all three nets
**BREACH**.

## Findings

**1. Entropy-floor loss falsified.** With the wide pool fixed, C (β=loss)
is uniformly *worse* than B (β=0): every member's σ is larger for C, and
nit moves from +0.0001/σ 0.02 (clean) to −0.0052/σ 1.24. The loss term
doesn't recover the breach; it adds variance. **Dropped going forward.**

**2. Wide pool fixes minestacker, not the rest.** The only breach that
*moves* from pre-fix → wide-pool is minestacker (σ 3.23 → 0.89 in B). All
other significant breaches are within the margin of the pre-fix numbers:
blueprint σ 4.22 → 3.82 (B), killphil σ 4.34 → 3.02 (B), maniac σ 3.82 →
5.37 (B). Minestacker was a small-pool-specific weakness; the rest aren't.

> **Correction vs prior session brief.** The earlier framing called this
> "the breach *moved* — wide pool opens a new self-arm breach." That isn't
> what the panel data says: the d=128 narrow-pool pre-fix net was already
> breaching blueprint at σ 4.22. Going to d=64 wide didn't open a new
> breach — it preserved the pre-existing blueprint/killphil/maniac
> breaches and fixed minestacker. Maniac is the one member that actually
> degraded (σ 3.82 → 5.37 in B).

**3. Hypothesis = capacity-limited.** Two members (blueprint, killphil)
breach at very similar magnitudes across all three arms and both pool
widths; maniac is large at every measurement. The student can shrink
minestacker (which lives in the depth range that was *missing* from the
narrow pool) but can't simultaneously fit the wider distribution and stay
faithful to the blueprint at d=64. A larger student that has the
parameters to do both is the next test.

## Next: d=128 wide-pool retrain (`runs/arm_D128/`)

Single arm — "arm B but bigger." Holds every arm-B hyperparameter fixed
except `d_model=128, num_layers=4, dim_ff=512` (~843k params, 6.7× B).
Same seed (2026), same strata, same anchor, β=0, 200×15 steps.

Pre-registered decision rule:
- **PASS** (within |σ|<2 vs blueprint every member, no seat fail) →
  capacity was the wall; floor cleared. Stop scaling; exploitation head
  next.
- **Improves but breaches** (self-arm σ drops materially from B's 3.82,
  breaches shrink) → capacity is the right axis; report the d=64→d=128
  exploitability slope to decide whether d=256 is justified.
- **Barely moves** (self-arm still 3–4σ) → NOT capacity; bottleneck is
  structural (card abstraction k=200 or pool design). Stop and redirect.

Cheap preview before the panel: `L_distill` at epoch 200. Arm B plateaued
at **0.7708**. If d=128 drives it materially below that, capacity
hypothesis is showing early; if it stays ~0.77, that's a warning the
bottleneck isn't capacity.
