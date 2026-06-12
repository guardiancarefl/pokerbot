# WINDOWS_TASK — scraper `folded`-flag staleness fix + showdown hole-card capture

**Authorization:** OQ-1 RESOLVED 2026-06-12 — operator APPROVED, scope
expanded, priority raised. Part A (`folded` staleness) is the priority
half. Route to a Windows CC session; this doc is the complete brief.

**Process bar (operator-set):** forensics-grade, like the pot-spike
validator work — evidence from archived frame PNGs FIRST, then a fix
proposal, then replay proof over an archived session BEFORE any live
use. The bridge/Contabo side must be provably untouched (replay gate,
§5). Precedent for packaging/verification style:
`tools/scraper_sanity_fixture/README.md`.

---

## 1. Why (what the staleness corrupts)

- The per-seat `folded` flag in scraper frames is Ignition-side stale:
  it does not transition when seats fold. This corrupts
  **fold-vs-shove counting**, which is both **H3's field statistics**
  (opponent DB, `data/opponent_db/`) and **H2's planned probe metric**.
  Fold-vs-shove currently measures **0/23** facing-allin folds — an
  artifact, not a field read.
- The live bridge survives it by re-deriving folds from chip patterns
  (`scraper_schema._repair_folded_from_chip_deductions`), **but the
  repair treats pre-existing `folded=true` flags as ground truth**
  ("treated as already dropped, augmented"). A fix that introduces
  FALSE-POSITIVE folded flags would therefore be live-path dangerous.
  Hard bar in §4: zero new false positives; the fix may only reduce
  false negatives.

## 2. Contabo-side forensic evidence (committed, ready to use)

`tools/scraper_folded_showdown_task/stale_folded_candidates.jsonl`
(regenerate any time:
`python -m scripts.extract_stale_folded_candidates --out <path>`).

Ground-truth construction: a NON-BLIND dealt seat whose hand net is
exactly `-ante` (net via chip-conservation closure between clean
hand-start anchors) **provably folded preflop** — no other line loses
exactly the ante. Restricted to hands that reached postflop
(`max_street >= 1`), so the scraper had whole streets of frames to set
the flag.

Measured on the full 16-session archive (2026-06-12):

- **241 chip-proven preflop folds → 0/241 ever got a `folded`
  transition. 100% stale on this slice.**
- The entire corpus (432 hands, 1,466 actions) contains only **16**
  `folded_flag` transitions, several at hand-boundary seqs (likely
  segmentation edges, not real-time fold detection).
- The earlier ~30% figure (scraper_schema docstring, live_1500 corpus
  2026-06-04) was a different denominator — postflop nominal
  hero-to-act FRAMES with at least one stale seat. Per fold-EVENT, the
  flag essentially never fires.

Each candidate row carries `captured_first`/`captured_last` — the
`captured_at` range of the hand, which is the PNG-archive filename key
(format `YYYYMMDD_HHMMSS_mmm`), plus `file`, `hand_idx`, `seat`,
`first_seq`/`last_seq`, `max_street`, `ante`, `bb`.

## 3. Part A (PRIORITY) — `folded` staleness

### A-1. Forensics FIRST (no code until this is written)

From the Windows PNG archive, pull the frames for **>= 20 candidate
hands across >= 3 sessions** (sample `stale_folded_candidates.jsonl`;
include both early-blind and ante-level hands). For each, document:

| captured_at | seat | on-screen truth (PNG) | emitted `folded` | cause hypothesis |

On-screen truth = what a human sees: cards mucked / seat grayed /
"FOLD" badge / no cards in front. Then identify the MECHANISM — e.g.
detection region wrong for some seats, card-back detector defeated by
the fold animation, flag only computed on some frame types, sit-out
overlay confusion. Deliverable:
`FORENSICS_FOLDED.md` with the table + mechanism finding, BEFORE any
fix. (If the PNGs show the scraper's detection region never looks at
fold state for non-acting seats, say exactly that.)

### A-2. Fix proposal

Short written proposal (what changes, what cannot be affected) —
reviewed before implementation. Semantics to preserve: `folded[seat]`
means "this seat folded earlier THIS hand"; it must reset at hand
start; `null` stays legal (Contabo coerces `null` → False,
`scraper_schema.py:251`).

### A-3. Replay proof over archived sessions (Windows side)

Re-run the patched scraper OFFLINE over the archived PNG sequences of
**>= 2 full sessions** that appear in the candidates file; emit a
stream jsonl per session. Verify (script the checks; stdlib-only, in
the style of `tools/scraper_sanity_fixture/check_after.py`):

- **(a) Stale class closed:** every candidate (hand, seat) of those
  sessions has a `folded=true` transition no later than the first
  postflop frame of its hand.
- **(b) ZERO new false positives (hard bar, live-path safety):** no
  seat is flagged folded that subsequently has a chip-committing
  action in the same hand (bet-level rise across consecutive frames =
  the opponent-DB `frame_diff` evidence class). Any single violation
  ⇒ fix rejected, back to A-2.
- **(c) Everything else untouched:** all non-`folded` fields
  byte-identical to the originally archived stream for every frame.

Ship the before/after stream jsonls + the verifier output back to
Contabo with the patch (drop under
`tools/scraper_folded_showdown_task/after/` for the archive).

## 4. Part B — showdown hole-card capture (approved as specced)

- New **OPTIONAL** per-frame field:
  `"shown_cards": {"<seat>": ["As", "Kd"], ...}` — present ONLY when
  revealed opponent cards are actually visible (showdown, all-in
  runout). Absent otherwise (absent, not null/empty).
- New top-level `"schema_version": 2` stamped on every frame once the
  new sender deploys (lets the Contabo ingester gate on it).
- Card encoding: rank in `23456789TJQKA` + suit in `cdhs`, exactly the
  existing `hero_cards` convention.
- Windows gates: on archived PNG sessions containing showdowns,
  OCR'd `shown_cards` must match human PNG reading on **100%** of a
  >= 20-frame showdown sample; field absent on ALL non-showdown frames
  of the replayed sessions; document the confusable glyph pairs you
  checked (T/7, Q/O, 6/8, club/spade at small render sizes).

## 5. Contabo-side gates (this repo — run when the patch lands; pre-committed now)

- **G1 — bridge provably untouched.** No Contabo live-path code change
  is expected at all. Gate: (i) standing full replay gate — ALL
  raw-record dry-run logs through `make_decision`, byte-identical
  pre/post anything that lands from this task (rule: all logs, never a
  subset); (ii) **injection gate** — take the archived logs, inject
  `schema_version: 2` + synthetic `shown_cards` into every frame, and
  replay: decisions byte-identical to the un-injected replay (proves
  `parse_frame` ignores the new keys).
- **G2 — H3 ingester upgrade behind a schema version.** Parser gains
  `shown_cards` → `showdowns` table opponent rows (today structurally
  always 0), gated on frame `schema_version >= 2`; bump
  `parser_version`. Re-ingesting the EXISTING corpus must reproduce
  `FIELD_REPORT.txt` exactly (idempotence + no-behavior-change on v1
  frames). A synthetic v2 stream must populate opponent showdown rows.
- **G3 — post-deploy validation, first live session:** re-run
  `scripts/extract_stale_folded_candidates.py` scoped to the new
  session — stale rate target **<= 5%** (from 100%); fold-vs-shove
  events become countable (denominator > 0); ingest shows opponent
  showdown rows when showdowns occurred.

## 6. What this unblocks / non-goals

Unblocks: honest H3 field stats (VPIP folds, fold-vs-shove), H2's probe
metric, and upgrades H4's RNR target from action frequencies to
range-conditioned reads (showdown holdings).

Non-goals here: NO bridge changes; NO SanityChecker P1/P2 work (that is
the separate `tools/scraper_sanity_fixture/` task); NO auto-click /
arming implications — sender stays Stage-1 dry-run
(`tools/windows_scraper_sender/README.md`).
