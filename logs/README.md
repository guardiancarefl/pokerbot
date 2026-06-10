# logs/ — canonical live-play record

**The scraper JSONL files in this directory are the sole and permanent
record of live play.** There is no official hand-history source for this
site/format (Ignition does not expose downloadable HH for the double-up
turbo SNGs the bot plays), so these logs are the only ground truth for
what the deployed agent saw and decided.

## Rules

- **Never delete a live session log.** `live_dryrun_*.jsonl`,
  `live_*.jsonl`, and their `.stdout` companions are irreplaceable —
  re-running cannot reproduce a past live session (live frames are not
  deterministic). Deletion is data loss with no recovery.
- **Gzip, don't delete, for housekeeping.** Sessions older than 30 days
  may be `gzip`'d in place (`gzip live_dryrun_YYYYMMDD_*.jsonl`) to save
  space. Convention only — there is no cron; do it by hand if/when the
  directory grows. The replay tooling should learn to read `.jsonl.gz`
  before this is exercised at scale.
- **Each run is identified by its `session_header` line.** Since
  `aa2f505`, `scripts/run_live_dryrun.py` writes a `record_type:
  session_header` first line carrying mode, seed, ckpt_sha256,
  abstraction_sha256, active floors, git HEAD, and start time. That line
  ties a log file to the exact model + code that produced it. Older logs
  (pre-`aa2f505`) lack it — their provenance rests on file mtime +
  STATUS/DECISIONS dating.

## Why gitignored

`logs/` is in `.gitignore` (large, append-only, machine-local). The live
session logs live only on the Contabo host. If they ever need durable
off-box backup, that is a deliberate operator action — they are not in
git and a host loss would destroy them.

## Not live records

Many files here are training/gate/diagnostic logs (`*_train.log`,
`gate_*.log`, `diag_*.log`, `task*.log`, etc.), not live play. Only the
`live_*` / `live_dryrun_*` JSONL families are the canonical live record;
the rest are reproducible run artifacts.
