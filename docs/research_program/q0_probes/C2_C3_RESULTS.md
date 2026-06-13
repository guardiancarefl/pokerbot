# C2 / C3 — the two free gating numbers (results)

Both free, read-only. They decide whether the exploit is worth capturing (C2) and
whether the pool is the right target (C3) — before any training investment.

## C3 — pool-shove fidelity vs the real 517 — VERDICT: GREEN (at C's regime)
Source: `evals/h4_pool_20260613/REPORT.txt` (pool moment-matched to the
FIELD_DOSSIER built from the real 517 hands). On the SHOVE dimension at C's exact
regime (5–15bb), the pool matches the real field:
- **T5_openjam 24.7% vs field 25.5% [21.4,30.1] — IN-CI**
- **T7_jam_5_15 8.5% vs 8.4% [6.7,10.5] — IN-CI** (the 5–15bb jam wall)
- **T7_5-10 12.5% vs 11.4% — IN-CI · T7_10-15 4.5% vs 5.9% — IN-CI** (C's 11–15bb)
- jam-wall + VPIP targets: **6/6 IN-CI.**
- Only OUT: T7_15-25 (1.5% vs 3.0%) and T8_15-25 (8.0% vs 12.9%) — ABOVE C's
  11–15bb core and explicitly LOW-EV (limitation [3]: "the high-EV 5–15bb jam wall
  is intact").

**⇒ The exploit (BB fold-more vs field shoves at 11–15bb) aims at a FAITHFUL
target.** The pool's shoves are real where it matters; the synthetic-artifact risk
is confined to 15-25bb+ sub-bands that are not C's headroom. Foundation confirmed.

## C2 — reachability / self-anchor-cost map — VERDICT: GREEN (REACHABLE/CHEAP)
`evals/c2_reachability_20260613.json`. Field headroom +0.0866/spot (gain vs field)
vs **self-anchor cost +0.0040/spot (price vs the champion's own shoves) = 4.6% of
the headroom.** Field-oracle and champ-oracle AGREE on **94.5%** of spots.
- **Cheapest exactly where the headroom concentrates:** by depth, cost falls
  0.0080 (5bb) → 0.0025 (11bb) → 0.0017 (15bb); disagreement 10% → 2.4%. C's
  11–15bb core is the SAFEST place to apply fold-more.
- BB (59% of the headroom): cost 0.0048 — still cheap.

**⇒ fold-more is nearly FREE vs the champion (4.6% of its field gain), and freest
at 11–15bb.** Decisive reframe: the champion's BB over-call is a genuine LEAK
(suboptimal vs ANY reasonable shover, not a champion-calibrated choice) — so
fixing it helps vs the field AND barely costs vs the champion. **This is why H4
collapsed: it learned OFFENSIVE over-aggression (which DOES cost), not the
DEFENSIVE fold-more (which is nearly free).** A method that targets fold-more
specifically captures ~+0.087 field at ~0.004 self-anchor cost = net hugely
positive and self-anchor-safe.

## Combined verdict → PROCEED to B (free, no pod)
- C3: target is REAL (pool shoves faithful at 11–15bb).
- C2: exploit is REACHABLE and nearly free (cost 4.6% of gain), safest at C's core.
- The exploit is barely entangled, so MULTIPLE methods should work: **B** (gated,
  zero self-anchor by construction — cleanest), **F** (un-gated oracle overlay may
  even pass the self-anchor gate, since fold-more costs only ~0.004/spot), **D**
  (jam-wall training). Build B's distillation trainer next (free); F is a cheap
  fallback/baseline. No pod needed for any of these.
