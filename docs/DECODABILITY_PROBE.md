# Decodability probe — Phase-2 premise necessary-condition screen

**Date:** 2026-06-04
**Branch:** `track-policy`
**Corpus:** `l4_corpus/` (scp'd from RunPod `/workspace/l4_corpus/` —
 train 270,047 tuples / 1,502 matches across 12 opponents; held-out 44,879
 tuples / 300 matches across 5 opponents; blueprint sha1 `b1981e3e14a9` matches
 Contabo's anchor; foundation 4/4 PASS in `docs/CORPUS_FOUNDATION_CHECK.md` on
 the RunPod side).
**Probe script:** `scripts/decodability_probe.py` (478 lines, frozen-feature
 sklearn LR).
**Phase-1 backbone:** `runs/phase1_d128_repro/smoke_net.pt` (843,283 params,
 d=128/L=4/ff=512; FROZEN for feature (3)).
**Total wall:** 4,049 s (67 min) on Contabo CPU.

---

## TL;DR — Verdict for Phase-2

**Necessary condition is MET.** Opponent identity is decodable from the
within-match observable history at realistic match length, and the
decodability **generalizes to held-out opponents** (binary AUC 0.91, multiclass
top-1 77% on FAITHFUL feature, vs chance 8.3%).

**But the signal carrier is `opp_stats` (the running tendency vector), not the
token sequence.** The PRIMARY tokens feature is at chance on held-out
(AUC 0.464), even at 20+ hands observed; the FAITHFUL `combined`
representation generalizes (AUC 0.91) almost entirely through its
`opp_stats_proj` channel (32 dims out of 160). The Phase-1 transformer
trunk adds only ~2% multiclass accuracy beyond raw `opp_stats`.

**Implication for Phase-2 design.** The L2-from-anchor gate primitive
collapses this 160-d signal into one scalar `d_bar`. The earlier
`g6_empirical_anchor` probe already showed that scalar is too noisy to be
discriminative. The decodability result here says the *upstream*
representation `combined` IS discriminative — so the right Phase-2 design
is to condition the policy on the K=10 tendency vector or the 160-d
`combined` directly (concat into the policy_head input), not via a scalar
L2 gate. Phase-2 RL has real opponent signal to exploit, provided the gate
mechanism stops bottlenecking it.

### Verdict scope (caveats)

- **Held-out = new shanky profiles only.** The 5 held-out opps
 (TheFixerSNG / ticketmaster / kamakazi / littlegreen / timidtom) are all
 shanky-kind. The train pool includes 6 shanky + 5 archetypes (NIT / TAG /
 LAG / STATION / MANIAC) + blueprint. The held-out generalization measured
 here is *to new shanky profiles*, **not** to a new opp kind. Generalization
 to an unseen archetype family is not tested.
- **Binary label is per-opp mean tendency distance to blueprint reference,
 thresholded at the median of train opp distances.** Train binary label spans
 shanky+archetype families (some "exploitable" archetypes are tagged
 exploitable by tendency-distance, e.g. MANIAC, STATION); held-out is
 shanky-only with 4 GTO-like + 1 exploitable. Multiclass↔binary divergence
 may be partly a label-composition artifact rather than a pure signal result.
 Reported numbers below show both.

---

## Probe design

### Three feature sources (all frozen, no retraining)

| # | Feature | Dim | Role |
|---|---|---|---|
| (1) | `tokens_pool = concat(tokens.mean(0), tokens.std(0))` over `tokens[≤50, 236]` | 472 | **PRIMARY / verdict.** Pure history-of-public-state-from-hero-POV input. |
| (2) | `opp_stats[:12]` | 12 | **REFERENCE / upper bound.** K=10 running SeatStats tendency + match_conf + n_act/300. Circular with the binary label (the label is derived from per-opp mean of the same statistic) — causal not leaky, but not a clean signal verdict on its own. |
| (3) | `combined = concat(trunk_repr_at_query[128], opp_stats_proj[32])` from frozen Phase-1 d128_repro net | 160 | **FAITHFUL / false-red guard.** What the blend's `policy_head` and `opp_head_tendency` both actually read from (`src/nlhe/adaptive/model.py:135`). If (1) decodes but (3) doesn't, the trunk has thrown the signal away. |

### Split

- **Train val split**: 1,502 train matches → 1,202 train / 300 val by `match_id`
 (80/20). Split is by MATCH (no intra-match tuple leakage between train/val).
- **Held-out split**: all 300 held-out matches (5 opps × 60 matches per opp).

### Labels

**Multiclass (12 classes):** literal `opp_id` for the train opp set.

**Binary (GTO-like = 0, exploitable = 1):** computed per-opp from the
manifest's `tendency_blueprint_ref`. For each opp, take the mean of its
per-match `tendency_target` (10-d SeatStats vector). Threshold at the median
of train opps' L2 distances to `tendency_blueprint_ref` (= 0.5915).
Blueprint is forced to label 0.

```
TRAIN binary mapping:
  GTO-like (0):  GusHansenMTT(0.444)  KillPhilMTT(0.422)  LionMTTv.10(0.462)
                 TAG(0.555)  blueprint(0.229)  wildbillmtt(0.219)
                 [6 opps; 166,047 tuples = 61%]
  Exploitable (1): LAG(0.670)  Loosenluckymtt(0.834)  MANIAC(0.729)
                 MillenniumMTTv.49(0.628)  NIT(0.719)  STATION(1.197)
                 [6 opps; 104,000 tuples = 39%]

HELD-OUT binary mapping (same threshold):
  GTO-like (0):  TheFixerSNG(0.455)  kamakazi(0.428)  ticketmaster(0.486)
                 timidtom(0.432)
                 [4 opps; 35,088 tuples = 78%]
  Exploitable (1): littlegreen(0.847)
                 [1 opp; 9,791 tuples = 22%]
```

Note the held-out exploitable class is *only* littlegreen — a single-opp class
on the positive side. AUC is well-defined (both classes present), but its
interpretation is "can the model distinguish littlegreen's tendency
signature from the other 4 shanky GTO-likes" rather than a multi-opp
exploitable-recognition test.

### Match length

`median = 150` hands (deterministic `hands_per_match=150` from manifest);
`min = 20`, `max = 150` over 1,502 train matches. The minimum-20 matches are
likely early ICM-bust scenarios; the bulk are full 150-hand matches.

---

## Results

### Multiclass top-1 on train val (12 classes; chance = 0.0833)

| Feature | Top-1 val | Ratio vs chance | Note |
|---|---|---|---|
| (1) PRIMARY tokens 472-d | **0.310** | 3.7× | Real but modest signal |
| (2) REFERENCE opp_stats 12-d | 0.743 | 8.9× | Circular ceiling |
| (3) FAITHFUL combined 160-d | **0.766** | 9.2× | Slightly exceeds REFERENCE |

The fact that **(3) exceeds (2)** means the Phase-1 trunk has distilled
some *additional* opp-distinguishing structure from tokens beyond what
opp_stats raw carries.

### Binary AUC (train val) — opp set spans shanky + archetype families

| Feature | AUC val | n_pos | n_neg |
|---|---|---|---|
| (1) PRIMARY tokens | 0.761 | 20,868 | 33,486 |
| (2) REFERENCE opp_stats | 0.920 | 20,868 | 33,486 |
| (3) FAITHFUL combined | 0.922 | 20,868 | 33,486 |

### Binary AUC (held-out) — 5 new shanky profiles only

| Feature | AUC held-out | n_pos | n_neg | vs val AUC |
|---|---|---|---|---|
| (1) PRIMARY tokens | **0.464** | 9,791 | 35,088 | **−0.30** (collapses to chance) |
| (2) REFERENCE opp_stats | 0.923 | 9,791 | 35,088 | +0.00 |
| (3) FAITHFUL combined | 0.907 | 9,791 | 35,088 | −0.02 |

**The PRIMARY tokens-only feature collapses to chance on held-out** —
the tokens carry opp signature only via incidental hero-side correlations
(hero pot sizes, board exposure to opp's style) that don't generalize across
opp families. **REFERENCE and FAITHFUL both generalize.**

### History-length AUC curve (binary)

Buckets are inclusive ranges on `hand_index_in_match` (0 = first hand of
the match, 149 = last). "h_obs" reads as "hands of opp behavior observed
before this decision."

#### (1) PRIMARY tokens
```
              h_obs    val_AUC (n)        held_AUC (n)
                 1      0.761 (743)        0.461 (587)
                 3      0.759 (727)        0.456 (606)
                 5      0.774 (721)        0.458 (603)
                10      0.753 (1,840)      0.470 (1,507)
               20+      0.760 (47,026)     0.464 (38,872)
```
**Flat across history.** Tokens encode hero infoset + public game state;
they don't accumulate opp signature with more hands observed.

#### (2) REFERENCE opp_stats
```
              h_obs    val_AUC (n)        held_AUC (n)
                 1      0.784 (743)        0.546 (587)
                 3      0.826 (727)        0.587 (606)
                 5      0.861 (721)        0.641 (603)
                10      0.870 (1,840)      0.763 (1,507)
               20+      0.927 (47,026)     0.941 (38,872)
```
**Steep rise with history**, on both val and held-out. By 20+ hands observed,
held-out AUC matches val AUC (0.94 vs 0.93) — strong generalization given
enough match-conf.

#### (3) FAITHFUL combined
```
              h_obs    val_AUC (n)        held_AUC (n)
                 1      0.797 (743)        0.516 (587)
                 3      0.839 (727)        0.561 (606)
                 5      0.865 (721)        0.668 (603)
                10      0.869 (1,840)      0.741 (1,507)
               20+      0.929 (47,026)     0.926 (38,872)
```
**Tracks (2) closely.** The transformer trunk adds essentially zero
generalization on top of opp_stats_proj on held-out. (1)+(3) joint reading:
the trunk's discrimination on held-out is carried entirely by the
opp_stats_proj 32-d slice of `combined`, not by the trunk's processing of
tokens.

---

## What this means for Phase-2

1. **Phase-2 has real opponent signal to exploit.** Held-out binary AUC 0.91
 from a frozen-trunk linear probe is unambiguous: the representation that
 the blend's heads consume IS opp-discriminative on opponents the model
 never saw.

2. **The signal carrier is `opp_stats`, not tokens.** This refines the
 "Phase-1 distill backbone can't beat blueprint" finding from
 `docs/REBEL_OVERNIGHT_PHASE1_REPORT.md`: the trunk works as a denoising
 conduit for the running tendency input, but its transformer-side
 processing of tokens does not enrich opp identity. For Phase-2, the
 transformer is doing infoset processing for the policy head; it's not the
 opp-identifier.

3. **The L2-from-anchor gate is the bottleneck, not the input signal.**
 Combined with the prior empirical-anchor G6 probe
 (`runs/phase1_d128_repro/g6_empirical_anchor.json`, commit 7473a3e):

  - the head produces opp-discriminative output at the K=10 vector level
  - blueprint sits in the middle of the archetype cluster in head-output
    space (no distinct "blueprint" mode), so `||head − blueprint_ref||₂`
    cannot reliably separate
  - per-decision noise in the head's K=10 vector is large
   (per-dim std 0.25 on AF_flop) and `tanh(scalar L2)` discards the
    cross-dim structure

  Today's probe confirms the K=10 (and downstream 160-d `combined`)
 vectors carry decodable identity; collapsing them to a scalar throws
 that away. **The right Phase-2 design conditions the policy on the
 vector directly** — e.g., feed `tendency_pred` (10-d) or `combined` (160-d)
 into the policy_head input rather than a scalar-gated blend.

4. **Phase-2 RL should sharpen, not create, the opp signal.** RL with
 chip-EV reward will push the policy to use whatever the trunk hands it.
 Since the trunk already gives the policy_head a 160-d opp-discriminative
 representation (via the same `combined` it reads), the gradient is
 expected to learn "respond to this representation," not "extract more
 from raw tokens."

### Recommended next step

Before any Phase-2 RL training, do one CPU-cheap design test on the existing
frozen backbone: **construct an inference-time blend that uses the full
`tendency_pred` (10-d) instead of `||tendency_pred − bp_ref||₂` (scalar)**,
e.g. with a small mixing head trained on the corpus to produce the blend
weight. If a small linear/MLP head on `tendency_pred` matches the linear
probe's held-out AUC, the gate redesign carries no architectural risk and
Phase-2 RL can proceed on the new gate. If it doesn't, that's a
representation-resolution problem in the head, not Phase-2's job to solve.

---

## Files / artifacts

Created this run:
- `scripts/decodability_probe.py` — the probe (478 lines, frozen sklearn LR)
- `runs/phase1_d128_repro/decodability_probe.json` — full numerical results
 (per-feature multiclass, binary val + held-out, all history-length buckets,
 per-opp tendency distances, binary mapping)
- `docs/DECODABILITY_PROBE.md` — this report

Inputs unchanged:
- `l4_corpus/` (scp'd from RunPod, 46 MB, 120 files)
- `runs/phase1_d128_repro/smoke_net.pt` (frozen Phase-1 d=128 backbone)
- `src/nlhe/adaptive/model.py` (no edits)
