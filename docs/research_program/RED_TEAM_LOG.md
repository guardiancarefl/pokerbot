# RED_TEAM_LOG — adversarial review of the DEPLOYED floor composition

Standing function (registered 2026-06-12, operator directive): exhibits
against the composed deployment floors, each with severity, decisive
test, cheapest mitigation. Same discipline as every floor: evidence
first, registered gates, arming = operator line.

---

## RT-1 (2026-06-12, session 220101 seq=498) — sub-2BB FOLD remains samplable against a CALL argmax

**Exhibit (decision_audit, exact masses):** JcJs, MP, eff 1.9BB, pot
480, facing 200 (1BB to call, ~5.8:1), OOD-WARN active (<2BB, below
training support). Post-floor distribution: **FOLD 0.279 / CALL 0.442 /
ALLIN 0.279** — short-stack floor fired (restricted to its allowed set,
redistributed intermediate-bet mass) but the floor RESTRICTS-AND-
REWEIGHTS; it never masks FOLD facing a bet. Sample mode drew the 27.9%
FOLD. pre==post on all three surviving actions: the 28% fold-JJ-getting-
5.8:1 mass is the RAW policy's, an out-of-support artifact.

**Composition analysis (Q2, code-level):** facing a bet at ≤2BB:
- aa_kk floor: masks FOLD only for AA/KK preflop — JJ uncovered.
- check-free floor: inactive (to_call > 0).
- short-stack floor: keeps {FOLD, CALL, ALLIN} — by design never
  removes FOLD.
- tail floor (τ=0.10): prunes sub-τ tails only — cannot touch a 0.279
  mass; and it runs LAST, after the majority is already set.
⇒ **No deployed floor can prevent a material-mass FOLD draw against a
CALL/ALLIN argmax in this class.** The class is exactly where the
encoder is least trustworthy (sub-support OOD).

**Severity: HIGH-when-it-fires, LOW-frequency** (sub-2BB facing-bet
decisions are rare but maximal-stakes — tournament life at full pot
odds; corpus enumeration below quantifies).

**Decisive test:** enumerate every ≤2BB facing-bet decision in the full
historical corpus via decision_audit: post-floor FOLD mass when
post_argmax ∈ {CALL, ALLIN}; count realized wrong-side draws.

**Cheapest mitigation (CANDIDATE, NOT BUILT):** extend the short-stack
floor: at eff ≤ N BB (N=2.0 registered strawman) facing a bet, when
post-floor argmax ∈ {CALL, ALLIN}, MASK FOLD (redistribute its mass
proportionally to the survivors). Rationale: at ≤2BB with a committed
pot the fold-equity argument for keeping FOLD samplable is nil when the
model's own plurality says continue. Gates (pre-registered now): same
treatment as every floor — flag-gated OFF; flag-off byte-identity over
ALL raw-record logs; TG2-style counterfactual instrument over the
corpus enumeration; paired A/B (deployed chain vs +mask) before any
arming; arming = operator line.

**Status: corpus enumeration running; mitigation awaits its evidence.**

### RT-1 enumeration result (2026-06-13 ~00:40, `evals/rt1_enumeration_20260612/`)

Clean corpus (12 fidelity-clean logs, 522 replayed decisions; 4
pre-header logs quarantined, listed): sub-2.5BB facing-bet universe =
3 frames, all session 220101. **Class a = 2; realized wrong-side FOLD
draws = 2/2** (T8s seq474 eff 2.23, FOLD .333 vs ALLIN argmax .494;
JJ seq498 = the exhibit, masses reproduced exactly). Mirror class = 1
(86s, FOLD argmax .600 — the mask must not and does not touch it).

Counterfactual of the registered mask: **N=2.0 catches 1/2** (misses
T8s at 2.23); **N=2.5 catches 2/2**, fires 0× against FOLD argmaxes
(mirror-safe by construction, verified), forces 0 bad calls at the
optimistic equity bound. Honest caveats carried: n=3 universe; both
fires lean on OOD-adjacent argmaxes; ICM-bubble-fold spots absent from
but not ruled out by this corpus.

**Design selection (evidence-driven, pre-build):** mitigation proceeds
at **N=2.5** (catches both realized instances; the strawman N=2.0 was
registered before the enumeration; this selection happens BEFORE the
mitigation's own gates run, and the paired A/B tests the chosen N on
fresh data regardless). Build remains queued behind the operator's
deployment-package review; gates unchanged from the RT-1 registration.
