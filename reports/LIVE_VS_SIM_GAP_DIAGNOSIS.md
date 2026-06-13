# LIVE-vs-SIM GAP — DIAGNOSIS (2026-06-13)

**Question:** why does a near-optimal bot (wins +0.746/game vs the field in sim)
go **9W/13L/4U live**?

**Answer: it's the SCRAPER, not the model.** Live, the bot is *not playing its
policy* on a large fraction of its turns — it skips or force-folds on bad
perception. The +0.746 edge is real; it just isn't being *captured* because the
table can't be reliably *seen*.

## Evidence (3 live sessions, `scripts/dryrun_triage.py`)

| session | frames | fresh model decisions | hero-to-act SKIPPED (bad data) | SAFE-FOLDS (forced) | hands lost entirely |
|---|---|---|---|---|---|
| 220101 | 505 | 38 | 10 | 1 | 0 |
| 230149 | 1649 | 114 | 86 | 14 | 8 |
| 023826 | 789 | 50 | 53 | 3 | 10 |

~**30% of hero-to-act moments are lost to scraper data quality** (skipped or
force-folded). Catastrophic individual leaks:
- **Folded pocket ACES** (AcAh, seq=1086) on an `invariant_fail`.
- **Folded JsAs** on a `replay_error`.
- Scraper self-consistency failures (e.g. pot scraper=375 vs recon=1500).

A bot folding/skipping ~a third of its spots — including premiums — *will* go
underwater regardless of how good its policy is. **9W/13L is a capture problem.**

## Dominant skip causes (230149, 225 data-quality skips)
| cause | % | maps to |
|---|---|---|
| scraper suspect frame | 56% | SanityChecker / capture quality (OQ-4 wrong-window) |
| dealer → non-alive/empty seat | 24% | dealer-button detection |
| dealer field missing/empty | 9% | dealer-button detection |
| n_alive below playable | 9% | player-count / folded-flag (OQ-1) |
| blinds-string OCR fail | 2% | OCR |

Plus `invariant_fail` / `replay_error` reconstruction failures → the premium
safe-folds.

## What this means
- **The live-vs-sim gap is ~entirely scraper/capture reliability.** Fix the
  scraper data quality → the bot plays its model → it realizes its proven edge.
- The dominant causes (suspect frames 56%, dealer detection 33%) are **Windows
  scraper-side** — exactly OQ-4 (window-handle capture) + OQ-1 (folded-flag /
  n_alive). Both briefs are already written; both **await operator routing to a
  Windows CC session.**
- The safe-fold is conservative *by design* (don't act on bad data — shoving into
  a made hand on a misread is worse than folding AA). So the fix is **better data,
  not a looser safety valve.**
- **Stage-2 readiness is gated on the same thing** — the scraper SanityChecker is
  the binding Stage-2 constraint.

## Highest-EV action
Land the **Windows scraper reliability** work (OQ-4 capture + OQ-1 folded-flag) —
it directly converts the ~30% lost turns into played turns, capturing the proven
+0.746 edge. This is worth far more than any model tweak. **Needs operator routing
of the existing briefs to a Windows CC session** (the scraper runs on Windows;
this Contabo bench can sharpen the briefs and the Linux-side decision logic but
cannot fix the Windows capture itself).
