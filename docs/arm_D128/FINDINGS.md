# arm_D128 — Capacity Test (d=128 L=4 ff=512, ~843k params)

Single-arm scale-up of arm_B: hold every hyperparameter fixed, change only
the trunk size. Tests whether the floor breach in the d=64 wide-pool
ablations is capacity-limited.

## Setup

| field            | arm_B (d=64)             | arm_D128 (d=128)        |
|------------------|--------------------------|-------------------------|
| d_model          | 64                       | **128**                 |
| num_layers       | 2                        | **4**                   |
| dim_ff           | 256                      | **512**                 |
| params           | 125,587                  | **843,283 (6.7×)**      |
| seed             | 2026                     | 2026 (same)             |
| pool             | wide stratified          | wide stratified (same)  |
| strata           | `5352,2000,2000,1500`    | same                    |
| epochs × steps   | 200 × 15                 | same                    |
| β entropy-floor  | 0 (falsified in d=64)    | 0                       |
| anchor           | candC_k200               | same                    |
| abstraction      | retrofit (k=200)         | same                    |

Panel JSON: `docs/arm_D128/icm_panel_arm_D128.json` (1000 hands/seat × 6
seats × 5 members, paired CRN vs blueprint).

## Result 1 — `L_distill` is flat

Final epoch 200 / 200:
- arm_B (d=64):    `L_distill = 0.7708`
- arm_D128 (d=128): `L_distill = 0.7693`  (**Δ = −0.0015**)

A 6.7× parameter increase moved the fit by 0.2%. The bigger model
distills the zero-context blueprint policy at essentially the same loss.
Capacity is **not** the binding constraint on fitting the teacher.

## Result 2 — Panel verdict: **BREACH** (4 of 5 members)

| member        | pre-fix d128-narrow | arm_B d64-wide | arm_C d64-wide+β | **arm_D128 d128-wide** |
|---------------|---------------------|----------------|-------------------|------------------------|
| blueprint     | σ 4.22              | σ 3.82         | σ 4.42            | **σ 2.64 ↓** (better)  |
| minestacker   | σ 3.23              | σ 0.89         | σ 1.66            | **σ 3.32 ↑** (regressed back to broken) |
| killphil      | σ 4.34              | σ 3.02         | σ 3.83            | σ 2.82 (≈ B)           |
| maniac        | σ 3.82              | σ 5.37         | σ 5.54            | **σ 7.18 ↑** (worse)   |
| nit           | σ 0.48              | σ 0.02         | σ 1.24            | σ 0.65 (clean)         |

Capacity helped **self-arm fidelity** (blueprint σ 3.82 → 2.64, 31%
reduction) and **killphil** marginally. It **regressed** minestacker (which
arm_B had cleanly fixed) and made **maniac** materially worse. Bigger
model = better local fit to teacher's zero-context policy, but worse
generalization on the most aggressive opponents.

Per-seat: 14 seat-level breaches (mostly seat 0 + seat 4/5 on maniac).

## Result 3 — Teacher ceiling diagnostic

Blueprint-as-hero absolute ICM-equity-delta per hand vs each panel
opponent, extracted from any panel JSON's `pooled.blueprint_icm` field
(numbers are byte-identical across pre-fix/B/C/D128 because paired CRN
fixes the seeds — confirmed in `icm_panel_arm_D128.json`):

| opponent       | blueprint absolute | interpretation                  |
|----------------|--------------------|---------------------------------|
| blueprint      | +0.00432           | near-zero (self-play sanity ✓)  |
| minestacker    | +0.00886           | blueprint *mildly* beats it     |
| killphil       | **−0.00889**       | **blueprint loses (mild)**      |
| maniac         | **+0.30505**       | **blueprint dominates**         |
| nit            | **−0.03105**       | **blueprint loses badly**       |

This is the decisive new piece of information. Two distinct breach
classes:

- **Maniac breach** (huge teacher edge, student fails to extract):
  blueprint earns +0.305 ICM/hand, student earns +0.266 — student is on
  the right side of zero but capturing 87% of the available value.
  Per the pre-registered rule: blueprint strongly beats maniac →
  **student-input bottleneck** (the student can't fully represent /
  exploit the structure the teacher sees). Richer card abstraction is
  the candidate lever.

- **Killphil / nit / blueprint-self breaches**: the teacher itself loses
  or is at parity. The student matching the teacher within sampling
  noise IS the floor — improving the student can never clear it.
  These breaches require a **better teacher**, not a better student.

The two breach classes need different fixes; scaling the student
addresses neither cleanly, which is what arm_D128 shows.

## Decision per pre-registered rule

> "Barely moves (self-arm still ~3–4σ) → NOT capacity; the bottleneck is
> structural (card abstraction k=200 or the pool design)."

Self-arm went 3.82 → 2.64 (31% reduction) — not "barely" but also not
clearing |σ|<2; meanwhile minestacker regressed and maniac worsened.
L_distill is flat. The fit ceiling is not capacity. **Do not scale to
d=256.**

Per the teacher-ceiling rule:

- Blueprint strongly beats maniac → student-input bottleneck →
  candidate lever: **card abstraction k=200 → k=1000**.
- Blueprint loses to nit/killphil → teacher ceiling →
  candidate lever: **blueprint retrain on k=1000 abstraction**.

Both levers point at the same artifact.

## Readiness: k=1000 abstraction is on disk, blueprint not yet trained

- Abstraction: `runs/abstraction_k1000_retrofit_20260530_221744/abstraction.pkl`
  (726 KB, built 2026-05-30).
- Config: `configs/six_max_phase4f_dcfr_candC_k1000.yaml` (committed at
  `ba3a947`, 2026-05-30).
- Blueprint training was started 2026-05-30 22:26 but **only got to
  iter 8 / 2000** (`runs/candC_k1000_train.log`); `runs/six_max_*_k1000/
  checkpoints/` is empty. The abstraction is ready; the blueprint is
  **never strength-tested**, as flagged in the session brief.

## Bug surfaced during this run (already fixed at HEAD)

`scripts/six_max_adaptive_smoke.py:1346-1349` previously wrote a
hardcoded `{"d_model":64,"num_layers":2,"dim_ff":256,"smoke_only":True}`
to every checkpoint's `config` field regardless of the CLI flags
actually used. arm_A/B/C happened to be d=64 runs so the bug was
silent; arm_D128 saved d=128 weights with a config claiming d=64.

Same class as the panel-loader bug fixed at `0f6b90f`: a static config
dict drifting from reality and causing silent load failure or, worse,
load with mismatched shapes that crash much later.

Fixed in `ff5b01d`: config is now derived from `args` at save time.
The arm_D128 checkpoint was patched in-place (state_dict was already
correct; only the metadata was rewritten — backup retained at
`runs/arm_D128/smoke_net.pt.bak_bad_config`). Post-patch fingerprint:

| arm       | params  | policy_head.weight.abs().sum() |
|-----------|---------|--------------------------------|
| arm_B     | 125,587 | 51.2608                        |
| arm_C     | 125,587 | 50.4091                        |
| arm_D128  | 843,283 | **57.4038**                    |

Three distinct nets — d=128 confirmed loaded for the panel run.

## Next

Two candidate work items, both pointing at the k=1000 artifact:

1. **k=1000 student retrain** (cheap test of student-input lever):
   train the existing d=64 distillation at the larger abstraction.
   If maniac breach shrinks, abstraction was the bottleneck.

2. **k=1000 blueprint retrain** (~hours of CPU): retrain the Deep CFR
   blueprint on the larger abstraction. Tests whether the teacher
   itself improves vs nit/killphil/blueprint-self.

Stop scaling student capacity until at least (1) is run.
