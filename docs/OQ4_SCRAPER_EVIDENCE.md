# OQ-4 — Windows scraper reliability: EVIDENCE PACKAGE (for routing to a Windows CC session)

The live-vs-sim gap (9W/13L) is ~entirely scraper data quality
(`reports/LIVE_VS_SIM_GAP_DIAGNOSIS.md`). This is the prioritized, evidence-backed
target list extracted from the live sessions (Contabo-side analysis; the fixes are
Windows-side). Route with `WINDOWS_TASK_SCRAPER_FOLDED_SHOWDOWN.md` (OQ-1).

## The two dominant skip classes (session 230149, 225 data-quality skips)
1. **scraper-suspect frames — 56%** (the scraper self-flags; breakdown below)
2. **dealer-button detection — 33%** (`dealer → non-alive/empty seat` 24% +
   `dealer field missing/empty` 9%)
(n_alive 9% — ties to OQ-1 folded-flag; blinds OCR 2%.)

## Suspect-reason histogram (3 sessions, 246 suspect frames) — Windows fix targets
| count | reason class | fix |
|---|---|---|
| **74** | **table not rendered** | **#1 — window-handle / WGC capture: don't capture (or auto-pause) when the target window isn't rendering the felt. THIS is the session-6 wrong-window failure (felt<0.10, OCR read back terminals/`_read_table.py`/`127.0.0.1:9000`).** |
| 18 | duplicate cards | card OCR robustness (impossible duplicate) |
| 23 | invalid board count (1/2) | board-card OCR / count validation |
| 8 | invalid hero count 1 | hero-card OCR |
| ~32 | seat stack "impossible" / "stack jump N->M" | **stack OCR robustness** (impossible values, single-frame jumps) |
| 13 | pot total unaccounted | pot OCR / reconstruction |
| 4 | board shrank 4->3 | board OCR flicker (cards disappearing) |
| 4 | bet > stack + pot | bet OCR sanity |

## Prioritized Windows work (by EV = frequency × leak)
1. **Window-handle capture (OQ-4 core)** — fixes "table not rendered" (30% of
   suspect frames) AND the wrong-window felt-collapse. Highest single lever.
   Add the felt-collapse auto-pause (don't act while felt < threshold).
2. **Dealer-button detection** — 33% of skips. Robust dealer read (currently points
   to non-alive/empty seats or goes missing).
3. **OCR robustness** — stacks (impossible/jumps), cards (duplicate/count), board
   (count/shrink), pot. These produce the suspect flags AND the invariant/replay
   reconstruction failures that force premium safe-folds (e.g. **pocket aces folded**,
   `live_dryrun_20260612_230149` seq=1086, invariant_fail).
4. **OQ-1 folded-flag** (separate brief) — fixes n_alive + fold-vs-shove integrity.

## Frame evidence pointers
- Suspect/dealer/recon signatures: `logs/triage_2026061{2_230149,3_023826,2_220101}.txt`
  (regenerate: `python scripts/dryrun_triage.py <log>`).
- Premium-fold smoking guns: 230149 seq=1086 (AcAh, invariant_fail), 220101 seq=386
  (JsAs, replay_error).
- Scraper sanity fixture for before/after proof: `tools/scraper_sanity_fixture/`.

**Why this is the project's highest-EV work:** it converts ~30% of lost hero-turns
into played turns (capturing the proven +0.746 edge) AND unblocks the Stage-2 gate
(0/5 clean — every session loses hands to these skips). Worth far more than any
model tweak.
