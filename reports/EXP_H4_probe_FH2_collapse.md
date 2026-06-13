# EXP_H4 — Field-RNR Probe: F-H2 Self-Anchor Collapse (VERDICT: FAIL AT PROBE)

**Date:** 2026-06-13. **Status:** CLOSED — probe failed its load-bearing gate;
program (Option-B full retrain) NOT triggered. **Experiment #4** (H1 PASS,
H2 FAIL, c1 MATERIAL, H4 FAIL-AT-PROBE).

**One-line verdict:** Global RNR-against-a-synthetic-Shanky-pool does NOT cleanly
teach field exploitation. The probe collapsed the self-anchor (−0.18/game,
z=−3.7, monotone over 4 checkpoints) while the only ΔFIELD "gain" was an
aggression-inflation artifact. The failure is a **METHOD bug, not a data bug**,
and it occurred on a **proven-bit-identical buffer start** — which kills the
last escape hatch ("league teaches if buffers are handled").

---

## 1. What ran

- **Champion:** `b79e82dd` (k200 candC, 236-d legacy encoder), preserved SLIM.
- **Buffer rebuild (P3):** generate-only (`populate_only`, frozen-net traversals,
  zero gradient steps). Verified bit-identical to the champion: **901,439 params,
  max|Δ|=0.000e+00** on the pod's `ckpt_full_rebuilt.pt` (iter 1900) — the
  deployed `strat_net` + all 6 advantage nets. Self-anchor START = clean noise.
- **Probe (P4):** `configs/h4_probe.yaml` — `league_mix=0.25`, weighted pool of
  8 coherent Shanky members (`registry_h4_field.json`), resumed the full rebuilt
  ckpt, iters 1901→2400, G=24 on the pod. Killed at iter ~2300 on the collapse
  trip-wire.
- **Faithfulness:** pod proven bit-identical to Contabo before any training
  (champion-recipe smoke: adv 0.5118/0.5068, strat 0.7703/0.8530, exact).
- **Evidence:** `evals/h4_probe_pod/progress.jsonl` (4 rows), checkpoints
  2000–2300 (`runs/h4_probe_pod/checkpoints/`).

## 2. The curve (monitor proxy, 400-game CRN-paired; noise floor ≈ ±0.05/SE)

| iter | self-anchor net | z | ΔFIELD | Δaggr | read |
|------|----------------|---|--------|-------|------|
| 2000 | −0.080 | −1.6 | +0.000 | +0.014 | clean baseline (noise) |
| 2100 | −0.085 | −1.7 | −0.020 | +0.058 | aggression drifting, no field payoff |
| 2200 | −0.135 | −2.7 | +0.025 | +0.046 | first breach of the −0.10/z≤−2 collapse band |
| 2300 | **−0.180** | **−3.7** | **+0.065** | +0.048 | **collapse confirmed; monotone x4** |

**The worst-quadrant signature:** self-anchor degrades monotonically
(−0.08→−0.18) while ΔFIELD stays sub-bar until 2300, when it crosses +0.05 ONLY
because aggression is inflated (Δaggr +0.048, ΔFIELD up TOGETHER) — the
**registered limp-gap artifact** (limitation 1: "ΔFIELD up WITH Δaggr inflated =
over-attacking pool passivity = NOT a green light"). So the +0.065 is a fake
win. The probe paid a −0.18 self-anchor collapse to buy a +0.065 *artifact*.

## 3. Root cause (the method bug)

**Our "RNR" had no explicit restriction to the frozen champion.** The intended
anchor was the 75% self-play mass in the traversal mix — but those self-play
opponents play the *current* policy, which drifts as training proceeds. The
"anchor" therefore anchors to the **drifting current policy, not to `b79e82dd`**.
There is no restoring force toward the champion specifically; nothing pins the
strategy net to the blueprint. True RNR (Johanson et al.) uses an explicit
restriction parameter to the blueprint; we approximated it with self-play
mixing, and **that approximation is the bug**. The monotone walk-away
(−0.08→−0.18, never recovering) is exactly what "no restoring force" predicts.

This is decisive precisely because the buffer start was PROVEN bit-identical:
the policy began *as* the champion and walked away under league exposure. Buffer
handling was never the issue (H2's open question) — **the league-RNR method
itself does not cleanly teach.**

## 4. Contributing factors (real, but secondary to §3)

- **Imperfect pool + near-zero headroom → blunt over-aggression.** The champion
  already beats the weak pool ~+0.8/game (little to gain), and the pool is
  jam-wall-faithful but limp-style-imperfect. Best-responding to *that* pool
  teaches an aggression shift that beats the synthetic pool but loses to the
  champion's sound play. The yellow-band Δaggr is this exact signature.
- **k=200 can only represent blunt shifts.** Session 5 established k=200 as the
  binding plateau (loses to 15/19 Shanky scripts). Under-resolved buckets can't
  encode a *targeted* exploit, so league training degenerates into a global
  aggression change — which is what we observe.

## 5. The headline

**OUR DATA IS VALID; OUR METHOD OF APPLYING IT WAS WRONG.** The 517 real hands,
the frozen field portrait, and the moment-matched pool are sound inputs. The
error was the *method* — a global RNR retrain of the blueprint with an anchor
that doesn't anchor. The top research question is now: **how to correctly
convert 517 hands of field data into a winning policy** — see RESEARCH_MAP.

## 6. What this closes (honesty clause, spec §7)

> "If the H4 probe fails F-H2 (anchor) despite Option A, the successor claim
> 'league training works if buffers are handled' is DEAD."

**Confirmed dead.** The H2 mechanism question is answered on clean ground:
buffer discontinuity was a real but separable problem (solved by generate-only);
the league-RNR-against-synthetic-pool *teaching* mechanism is independently
broken. No future arm should resume global league-mix RNR of the blueprint
without an explicit blueprint restriction — and the open question is whether the
blueprint is even the right place for exploitation at all (residual-head /
adaptation-layer alternative).

## 7. What survives (instruments + assets, all reusable)

- **Generate-only buffer rebuild** (`populate_only`) — bit-identical refill, a
  permanent tool; free-self-play refill retired.
- **The gate battery + dashboard** — caught the collapse early and cleanly
  (the H2 lesson, working as designed). Self-anchor |z|<2, §4.4 aggression
  caps, F-B2 bar: all validated as discriminating instruments.
- **The frozen field portrait + moment-matched pool** — valid data; the question
  is the application method, not the inputs.
- **The proven-faithful pod** — bit-identity smoke + RUNPOD_SETUP reproducible.

## 8. Disposition → the lab expansion

The next step is NOT another global-RNR variant. It is a parallel cheap-probe
sweep of the **competing data-application methods**, cheapest-first, under the
same gates that caught this failure (self-anchor ≈0 AND real ΔFIELD trend with
FLAT aggression):
- **C** data-as-targeting (dossier → surgical leak spots; data as diagnostic) —
  run first/always;
- **A** explicit-restriction RNR (real anchor to frozen champion, λ-swept) —
  the direct fix for §3;
- **B** exploit residual head (frozen champion + additive correction, zero by
  construction) — self-anchor-safe by construction;
- **D** jam-wall-only training (tests the §4 imperfect-pool factor).

Most should die cheap; the survivor earns the full battery. See the expansion
plan (OPERATOR_QUEUE / coordinator brief).
