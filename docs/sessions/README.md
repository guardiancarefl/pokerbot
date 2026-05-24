# Session summaries

This directory holds one Markdown file per working session,
`session_<N>_summary.md`. It is the per-session companion to the always-current
docs and the original append-only log:

- `docs/STATUS.md` — single snapshot of "where are we now" (overwritten each session).
- `docs/SESSION_LOG.md` — the original append-only narrative log (Sessions 1–9 live here).
- `docs/DECISIONS.md` — locked-in choices, append-only.
- `NEXT_SESSION.md` — handoff brief for the next session (rewritten each session).

## Why a per-session file (vs the monolithic SESSION_LOG.md)

`SESSION_LOG.md` grew into a single 480+-line file whose internal ordering
drifted out of sync with the actual session numbering — it documents through
Session 9 while commits already reference Sessions 11–12. One file per session:

- keeps each session's summary self-contained and easy to diff/review,
- makes the commit that closes a session touch exactly one new file,
- avoids merge-style churn in one ever-growing file,
- gives a stable, greppable filename per session.

`SESSION_LOG.md` is retained as the historical record for Sessions 1–9. New
sessions (13+) are summarized here. Sessions 10–12 were captured only in commit
messages and STATUS at the time and are **not** back-filled.

## Format

Each `session_<N>_summary.md` covers, in order:

1. **What was done** — concrete deliverables, each tied to its commit hash.
2. **What was decided** — pointers to the `DECISIONS.md` entries added.
3. **What was learned / measured** — diagnostics, numbers, surprises.
4. **State at close** — what's done, what's open, what the next session picks up.

Keep it factual and commit-anchored. The *why* behind a locked decision belongs
in `DECISIONS.md`; this file is the chronological *what happened*.
