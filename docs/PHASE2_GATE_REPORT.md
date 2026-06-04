# Phase-2 go/no-go gate — final diagnosis

**Date:** 2026-06-04
**Branch:** `track-policy`
**Verdict:** **DO NOT start Phase-2 RL.** The within-match-from-scratch read,
as currently architected, is too slow for the real match-length window
(median 10 hands, p10 6 hands), and the trunk's contribution beyond `opp_stats`
collapses to chance on held-out opponents.

This report combines two cheap probes that gate the Phase-2 go/no-go:
JOB 1 (real match-length distribution under a real Ignition turbo schedule)
and JOB 2 (opp_stats ablation on the FAITHFUL feature from the prior
decodability probe).

---

## JOB 1 — real first-bust and 3-left windows

### Setup

- **Tournament structure:** `configs/ignition_double_up_6max_turbo.yaml`,
 captured from real Ignition tournament #80840020.
- **Starting depth:** 1500 chips, L1 SB=15 / BB=25 / ante=5 →
 **60 BB raw** (1500/25), 27.3 BB on the inflated-pot metric.
- **Blind escalation:** `hands_per_level=3` per the turbo convention
 (`game_strings.py:454` comment: "real Ignition's ~3 hands per 5-min Turbo
 level").
- **Natural termination:** match ends when ≤3 players remain (the
 Ignition Double-Up bubble).
- **Termination simulator:** built on top of `to_inner_game_string_for_state`,
 with external stack tracker. Sub-SB / sub-BB seats are blinded out
 pre-hand (OpenSpiel's `universal_poker` rejects `stack < blind`; real
 poker would auto-all-in for less, but at turbo blind levels stack < SB
 means functionally dead anyway).
- **n_matches:** 200 each policy.

### Two policies bracketing the answer

**Blueprint self-play** uses the same anchor as the prior decodability probe
(`runs/six_max_20260530_034023_phase4f_dcfr_candC_k200/checkpoints/ckpt_iter_2000.pt`).
**Critical caveat:** this blueprint was trained on the corpus's fixed-15-BB
`six_max_sng()` distribution (`game_strings.py:102`, BB=100, stack=1500).
In the simulator it plays at 60 BB → 30 BB → 15 BB → 10 BB → ... → 3.75 BB
sequentially across levels — **OOD-loose at the deep end** of the schedule.
Diagnostic tell: 14% of matches first-bust at hand 1 with blueprint, which is
unrealistic at 60 BB.

**Random-uniform** (uniform sampling from `state.legal_actions()`) is the
policy-independent reference. It's not a "floor" of sane play; it's the
maniacal ceiling — 6 random players raise all-in constantly.

### Numbers (200 matches each)

```
                              hands_to_first_bust              hands_to_three_left
                              median  p25  p10  min  max     median  p25  p10  min  max
Blueprint (OOD-loose):           4     2    1    1   14         10    7    6    3   24
Random-uniform:                  1     1    1    1    4          2    1    1    1    7
```

Histograms (5-hand buckets) for `hands_to_three_left`:

```
Blueprint:    0-4:  6   5-9: 87   10-14: 73   15-19: 27   20-24: 7
Random:       0-4: 190  5-9: 10   10-14:  0   15-19:  0   20-24: 0
```

### Diagnosis

- The YAML's own chip-math (`level_duration` × no-rebuys + L10 dead-money ratio)
 argues "game effectively cannot reach level 11 in practice"
 → real-Ignition max ≈ 30 hands.
- 60-BB starting depth is **real**, not a config artifact (provenance: in-client
 tournament info).
- The hand-1 bust tail is **partly** an OOD-blueprint artifact (the blueprint
 doesn't know how to play 60-BB carefully) but is **also** intrinsic to multi-way
 NL poker — random reproduces it harder.
- **The plausible real-Ignition match window with a sane in-distribution policy:
 ~15–30 hands. With OOD play (today's anchor): ~5–22 hands. The Phase-2 read
 has to function inside that window.**

---

## JOB 2 — opp_stats ablation on the FAITHFUL feature

### Question

The prior decodability probe (commit `ece08ae`, `docs/DECODABILITY_PROBE.md`)
showed that the 160-d FAITHFUL feature (= `combined` from frozen Phase-1 net)
achieves held-out binary AUC 0.907 — close to REFERENCE `opp_stats[:12]`
alone (0.923). Open question: does the FAITHFUL feature retain ANY held-out
signal once `opp_stats` is removed at the input, or does it collapse to the
PRIMARY-tokens chance floor (0.464)?

### Method

Run the Phase-1 forward pass with **`opp_stats` zeroed at the input**, then
extract `repr_` = transformer-pooled-at-query (128-d). Since
`opp_stats_proj(0)` is a constant bias term (32-d), it adds no linear
information to a Logistic Regression classifier — i.e. **zeroing
`opp_stats` at the input is mathematically equivalent to using `repr_`
alone as the feature**. Re-fit a binary LR on train, evaluate on val and
held-out, plot history-length AUC.

### Numbers

```
Feature                                       val_AUC    heldout_AUC
FAITHFUL combined 160-d (non-ablated, prior)   0.9220     0.9074
FAITHFUL ABLATED repr_ 128-d (opp_stats=0)     0.7072     0.4837   ← collapses on held-out
REFERENCE opp_stats[:12] (prior, circular)     0.9203     0.9231
PRIMARY tokens [mean,std] 472-d (prior)        0.7606     0.4637   ← chance floor on held-out
```

History-length AUC (binary, FAITHFUL ABLATED):

```
              h_obs   val_AUC (n)        held_AUC (n)
                 1     0.691 (743)        0.508 (587)
                 3     0.713 (727)        0.517 (606)
                 5     0.709 (721)        0.504 (603)
                10     0.697 (1,840)      0.487 (1,507)
               20+     0.708 (47,026)     0.481 (38,872)
```

### Diagnosis

- **Held-out AUC drops from 0.91 → 0.48** when `opp_stats` is removed at the
 input. That's near-chance — the same regime as PRIMARY tokens 472-d (0.46).
- The trunk has **not** learned an opp-discriminative representation from
 tokens alone that generalizes to new opponents. Whatever opp signal exists
 in `combined` is carried by the 32-d `opp_stats_proj` slice, which is a
 small linear projection of the running running-tendency input.
- The history-length curve is **flat at chance** for the ablated feature
 on held-out (0.48–0.52 across all buckets, 1 hand to 20+ observed). No
 within-match accumulation in the token-pooled trunk.

---

## What this means for Phase-2

### The collision

JOB 1 says **the gate has 5–30 hands** (more likely 10–22) to make a useful
read.

JOB 2 says **the only feature that decodes on held-out is `opp_stats`** —
the running tendency vector — and even that decodes weakly inside that
window (held-out binary AUC 0.55 at 1 hand observed, 0.64 at 5 hands, 0.76
at 10 hands, and only reaches 0.94 at 20+ hands).

The `match_conf` ramp in the current architecture (`model.py:170-184`) goes
from 0 → 1 over **300 actions** of designated-opp behavior. At 6 actions
per hand that's ~50 hands — well past most matches' end. By hand 10
(median match end), match_conf ≈ 0.2, gate ≈ tanh(d_bar) × 0.2 ≈ ~10%
effective opening even when d_bar is large. **The blend stays ~90% anchor
through most of the match.**

### Where the architecture fails

1. **`tokens` carry no transferable opp signal.** The Phase-1 trunk's
 transformer processing of public history doesn't extract opp identity that
 generalizes to held-out (JOB 2: 0.48 held-out AUC, indistinguishable from
 472-d raw token pool).
2. **`opp_stats` carries the signal, but slowly.** It needs 10–20 hands of
 observation to be useful, and most matches don't make it that far.
3. **The match_conf ramp is calibrated for a 50+ hand horizon** that this
 format doesn't have.

### Verdict — Phase-2 RL is OFF

Phase-2's premise was that an exploit gate would open during a match as
opp behavior is observed. The probes here show:

- The trunk doesn't help (JOB 2).
- The input that does help is too slow (JOB 1 × decodability buckets).
- The gate's calibration assumes a horizon the format doesn't have.

**Do not start Phase-2 RL on this stack.** The chip-EV gradient has nothing
to pull on inside the 10-hand window.

### What's salvageable (not in scope today; just flagged)

- **Stack-situation adaptation, not opponent adaptation.** ICM pressure and
 stack-depth ratios change deterministically over the match and are
 observable instantly. A "read" indexed on stack situation (not opp
 identity) is in-budget for a 10-hand window.
- **Re-calibrated match_conf.** A ramp that hits 1.0 around action 50, not
 300, would at least let the opp_stats-derived signal contribute.
- **Token-conditioned identification that actually generalizes** would
 require a different training objective (e.g., contrastive on opp identity,
 with held-out opps in training) — out of scope here, would need a corpus
 with held-out opps used as a contrastive signal in train.
- **Or: accept that opponent anonymity + 10-hand format means no opponent
 exploitation.** Build the best blueprint and stop.

These are framings, not recommendations. The deliverable today is the
diagnosis above.

---

## Files / artifacts

Committed this turn:
- `scripts/match_length_probe.py` — tournament sim with `--policy {blueprint, random}`
- `scripts/decodability_probe_ablation.py` — JOB 2 probe (repr_ 128-d feature)
- `docs/PHASE2_GATE_REPORT.md` — this report
- `runs/phase1_d128_repro/match_length_probe.json` — blueprint policy results
- `runs/phase1_d128_repro/match_length_probe_random.json` — random policy results
- `runs/phase1_d128_repro/decodability_ablation.json` — JOB 2 numerical results

Inputs unchanged:
- `l4_corpus/` (untracked; lives at `~/pokerbot/l4_corpus/`, 46 MB)
- `runs/phase1_d128_repro/smoke_net.pt` (frozen)
- `configs/ignition_double_up_6max_turbo.yaml` (turbo schedule)
- `src/nlhe/adaptive/model.py` (no edits)
