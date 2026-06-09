# Scraper SanityChecker P1+P2 — BEFORE-baseline validation fixture

Built on the Contabo side (2026-06-09) from every dry-run log that
carries raw scraper records. This is the validation gate for the
Windows-side SanityChecker changes:

- **P1**: relax-after-stale (N=4) + **consensus-derived** chips-in-play
  ceiling (replaces the `STACK_IMPOSSIBLE=13000` constant).
- **P2**: new-hand-detector widening (board-5→0 / pot-reset / button-move,
  animation-independent) + the board-shrank-vs-stale-`prev_board`
  ordering fix.

## Ceiling derivation — hard requirement

The ceiling must be the **consensus (mode/median) of pre-hand chip sums
across recent CLEAN hand-starts** — never a hardcoded constant, never a
single hand-start. This corpus contains the counterexample for both:
seat6 OCR'd `9907` ON a hand-start frame that the 13000 constant admitted
as clean (stream `live_dryrun_20260609_154557__s1`, captured_at
`20260609_115039_013`; the hand-start sums to 17917). A constant misses
it; a single-frame derivation would *adopt* it as the ceiling. The
consensus over this corpus is robustly **9000** in every stream
(`expectations.json: streams.*.consensus_ceiling`, derived independently
per stream; the raw per-start sums are included as `hand_start_sums` —
note the sub-9000 stragglers from mis-alive hidden-chip under-reads,
which is why consensus, not max or latest).

## What's in here

```
streams/<name>.jsonl    ordered scraper records (verbatim), one stream
                        per sub-session (seq resets at sender reconnects
                        — frames are keyed by captured_at, NOT seq)
expectations.json       machine-generated per-frame gates
before_summary.json     current (BEFORE) flag counts per stream
check_after.py          the verifier (stdlib-only)
```

The two incidents the fixture is built around:

1. **verify1** (`live_dryrun_verify1_20260609_202137__s1`): hero stack
   stuck-digit `11101/11104` for 10 frames (must stay rejected — the
   ceiling alone guarantees this even after relax), then the OCR
   self-corrected to `150` at captured `20260609_162633_543` /
   `20260609_162636_764` and the current checker rejected the CORRECT
   reads against its stale 1260 reference. Those two frames must admit.
2. **154557 s1** (`live_dryrun_20260609_154557__s1`): the mirror image —
   `9907` admitted clean AT the hand-start (became the poisoned
   reference), then 23 frames of correct reads (980/905/895) rejected
   against it. The 9907 frame must flip to rejected; the corrected reads
   from run-position 5 (= N+1) onward must admit. `s2` additionally has
   `9406/9407/11407`, a second `9907` hand-start, and the contrast case
   `9607` (bad value rejected at the door → correct read re-admitted in
   2 frames — the behavior P1 generalizes).

## How to run the validation (Windows side)

1. For each `streams/<name>.jsonl`: instantiate a FRESH patched
   SanityChecker, feed each line's `record` in order, and append
   `{"captured_at", "suspect", "suspect_reasons"}` per frame to
   `after/<name>.jsonl`.
2. `python check_after.py --after <after dir>`

Gates enforced (exit 0 = all pass):
- **(a)** 43 previously-rejected correct reads carry no stack-jump reason
  (frame may stay suspect for unrelated co-reasons on the same frame).
- **(b)** all 18 over-ceiling reads rejected at every frame, including
  the 2 currently-clean 9907 hand-starts (the only intended
  clean→suspect deltas in the corpus).
- **(c)** no AFTER-clean frame anywhere admits any stack above the
  stream's consensus ceiling.
- **(d)** every other currently-clean frame stays clean.

Suspect-frame reason *wording* is not gated — renaming "stack jump" to a
ceiling-style reason on the 11101 frames is fine; gate (b) only requires
the seat to be named. Run positions 1..4 of stale runs are deliberately
unconstrained so the fixture doesn't pin the relax-counter off-by-one.

The 12 board-shrank false-flags (P2) are listed in
`expectations.json: streams.*.board_shrank_fix` with their co-reasons.
They are all hand-transition artifacts (patterns 5→3, 5→4, 4→3 against a
stale `prev_board`). P2 should clear the board-shrank reason; gate (d)
will catch any NEW board flags P2 introduces on clean frames. Frames
whose only reason was board-shrank should come out fully clean — that is
visible in the (a)-style reason diff but not hard-gated, since the
Windows-local corpus (with its 48-frame board-shrank set) is the richer
testbed for P2; this fixture's 12 are the cross-check.

## Bridge interplay (no coordination needed)

The Contabo bridge's layer-1 recovery (`7d47e86`) reads ONLY
suspect-flagged frames. After P1, the verify1-93/94-class frames arrive
clean and recovery simply stops firing on them; mid-run frames that stay
correctly suspect (11101-class) remain covered by recovery. Per-field
suspect propagation is explicitly OUT of scope (shared frame-contract
change — separate decision with the Linux side).
