# Live dry-run pre-flight checklist

Stage 1, LOG-ONLY — the bot NEVER clicks. Follow top to bottom; every
step has a pass condition. If a pass condition fails, stop and fix
before going on. Written to be followable at midnight.

Deployed artifacts (do not substitute):

| artifact | path | sha256 starts |
|---|---|---|
| checkpoint | `runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt` | `b79e82dd` |
| abstraction | `runs/abstraction_20260521_223018_retrofit/abstraction.pkl` | `0fc20800` |

Start order matters: **Contabo listener first → tunnel → Windows sender.**

---

## 1. Contabo side — listener up first

```bash
cd ~/pokerbot

# 1a. Assert the deployed artifacts (must print the two prefixes above)
sha256sum runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \
          runs/abstraction_20260521_223018_retrofit/abstraction.pkl
```

PASS: hashes start `b79e82dd` and `0fc20800`. Anything else: STOP —
wrong artifacts.

```bash
# 1b. Start the listener in tmux, capturing stdout NEXT TO the jsonl
#     (the .stdout companion is how triage attributes [FLOOR] firings)
tmux new -s dryrun
TS=$(date +%Y%m%d_%H%M%S)
.venv/bin/python scripts/run_live_dryrun.py \
    --listen-port 9000 \
    --checkpoint runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \
    --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
    --fallback-seconds 7.0 \
    --tail-floor-tau 0.10 \
    --out logs/live_dryrun_${TS}.jsonl \
    |& tee logs/live_dryrun_${TS}.stdout
```

`--tail-floor-tau 0.10` is **STANDARD CONFIG as of 2026-06-12** (operator
approval: OQ-2). It arms the H1 commitment-scaled tail floor, last in
the floor chain — EXP_H1 verdict chain TG1–TG4 all PASS
(`docs/research_program/reports/EXP_H1_tail_floor.md` §7). The startup
banner must echo `ARMED: tau_max=0.100`; firings appear as
`[FLOOR] fired=[tail]` in the `.stdout` and are joined by triage.
Post-session: eyeball the tail firings in the triage decision digest.
Expected order of magnitude (self-play instrument numbers — live may
differ): distribution adjusted on ~70–75% of hero decisions, but the
SAMPLED action changes on only ~4.4% of decision frames (TG2
exact-commit). A session where tail firings change most argmaxes is
anomalous — the floor only prunes sub-τ tails.

`--fallback-seconds 7.0` is **operator-optional** (drop the line to run
without it): if hero is to-act and no decision exists within 7s, the
listener emits a CHECK-if-free / FOLD-otherwise plan in a loud red
FALLBACK banner — logged as a fallback, never as a model decision
(session-1 calibration: would have fired on all 8 never-decided spots,
zero false fires).

**Stage-2 must-land flags (all OFF by default; each replay-gated GREEN
2026-06-11; arm per operator call):**

| flag | what it does |
|---|---|
| `--anchor-sum-floor` | P1: refuse hand-start anchors whose sum ≠ 9000 with all 6 seats alive (the seq-1363 poisoned-anchor class) |
| `--extended-click-plans` | typed-raise verify step; ALLIN-button mapping for all-in intents; raise_to at call-only UI realizes as CALL (29/301 previously-unexecutable plans now executable) |
| `--watchdog-v2` | fallback deadline anchors per spot (hand+board), survives button flicker — closes the 9cAd gap; needs `--fallback-seconds` |
| `--abort-enforce` | abort criterion becomes enforced: click plans SUPPRESSED + SIT OUT NOW banner until `touch logs/ABORT_RESET`; needs `--fallback-seconds` |

PASS — all four of these appear before any frame:

1. The `session_header` JSON line echoes `"mode": "sample"` and
   `"ckpt_sha256": "b79e82dd…"` (the header is also the first line of
   the jsonl — triage re-asserts it).
2. `ante convention: real (servable)` — the conventions gate accepted
   the checkpoint.
3. `DRY-RUN ACTIVE: bot will display and log decisions but NEVER CLICK.
   Mode=sample.`
4. `listening on 127.0.0.1:9000 (waiting for Windows scraper sender…)`

Do NOT pass `--mode argmax` (it requires `--unsafe-argmax` for a
reason — argmax was the one real deployment bug).

## 2. Windows side

1. **Scraper window uncovered, on its own monitor.** The OCR reads
   pixels; any overlapping window, tooltip, or screensaver corrupts
   frames for as long as it covers the table. Disable sleep/screensaver
   for the session.
2. **Tunnel up (port 9000):**
   ```bat
   ssh -N -L 9000:localhost:9000 quant@<CONTABO_PUBLIC_IP>
   ```
   Keep this terminal open for the whole session. (Tailscale users:
   skip the tunnel and point the sender at the Contabo 100.x.y.z IP.)
3. **Sender integrated in the scraper** (`read_table.py`), per
   `tools/windows_scraper_sender/README.md` §2 — already wired for the
   06-08/06-09 sessions; just confirm the three call sites are present:
   - at startup: `sender = ScraperSocketSender("127.0.0.1", 9000)` +
     `sender.start()`
   - in the per-frame loop, next to the local JSONL append:
     `sender.send_frame(record)`
   - at shutdown: `sender.stop()`
4. Optional channel smoke (before starting the real scraper):
   ```bat
   python scraper_socket_sender.py --smoke 127.0.0.1 9000
   ```
   Contabo prints 5 `skip_data_quality` frames → channel is live.
5. **Start the scraper WITH a local log** (session 1 ran without one —
   the Contabo jsonl was the only copy of the session; never again):
   ```bat
   python read_table.py --live --loop --out scraper_local_%TS%.jsonl
   ```
   VERIFY the flag name first: `python read_table.py --help` on the
   Windows box (`read_table.py` is not in this repo). If the local-log
   flag is named differently, use that; if no such flag exists, the
   local JSONL append belongs right next to `sender.send_frame(record)`
   (see `tools/windows_scraper_sender/README.md` §2) — add it before
   the session.
   PASS: the local jsonl grows while the table renders.
6. Contabo prints `sender connected from 127.0.0.1:…`.

## 3. During the session — what to watch on the Contabo dashboard

- Fresh decisions print full READ / DECIDE / CLICK-PLAN banners;
  cached/skips print one-liners. That's normal.
- `[FLOOR] fired=[…]` — fine, that's the deployment floors working;
  triage will tabulate them afterward.
- `⚠ stall:` lines — tunnel or scraper hiccup; check the Windows side.
- **SAFE-FOLD banners or `[ANCHOR-REFUSED]`** — note the seq; these are
  red-flag material, triage will pull full context.
- **Red `FALLBACK — NOT A MODEL DECISION` banners** (only if
  `--fallback-seconds` is armed) — the pipeline went silent on a to-act
  spot and the watchdog emitted CHECK/FOLD. One is survivable; a
  `SESSION ABORT RECOMMENDED` line means sit out now.

## 4. After the session (or mid-session — it's read-only)

```bash
.venv/bin/python scripts/dryrun_triage.py logs/live_dryrun_${TS}.jsonl
```

Writes `logs/triage_${TS}.txt` and prints it. The `.stdout` companion
is picked up automatically for the floor column. Paste back the final
`TRIAGE …` summary line.

Then feed the session into the opponent DB (H3; idempotent, waits-10-min
guard means run it after shutdown): `.venv/bin/python -m scripts.ingest_session logs/live_dryrun_${TS}.jsonl --stats`

Red-section items that demand attention before the next session:
any safe-fold, any `anchor_refused`, any `FOLD WHILE FACING 0`
(impossible post-floor — would mean the floors aren't active), any
seq-170/seq-192-class reconstruction signature.

### Session-2 scoreboard (vs session-1 baselines, log 20260611_163815)

Compare the triage summary line against these. Baselines are session 1;
targets assume the Windows pot-spike-validator fix is deployed (and,
for never-decided, `--fallback-seconds 7.0` armed). Backstop sign-off:
the Contabo invariant gate PASSES the seq-188-class frames the fixed
scraper will now deliver (all 4 killed hands' to-act frames → full
decisions on replay; `evals/live_session1_audit_20260611/backstop_seq188_class.txt`).

| metric | session 1 | session 2 target | if missed |
|---|---|---|---|
| skip rate | 86.6% (421/486) | ≈83% (validator-FP class ≈18 frames returns to the decision path) | suspect sub-causes in triage [b] say which gate is still firing |
| hands lost to skips | 4 (8dAd, AdKs, 7s3s, 2sJh) | **0** | any lost hand = scraper fix incomplete; pull its frames |
| never-decided to-act spots | 8 | **0** (fallback guarantees an action) | each one should instead appear as a logged FALLBACK row |
| red flags (triage [e]) | 4 | ≤4, no new classes (scraper fix does NOT address replay/derivation safe-folds) | any seq-170/192 signature: stop, audit before next session |

## 5. Shutdown

Ctrl-C the listener (it prints `STILL NO CLICKING. Log-only.`), stop
the Windows scraper, close the tunnel. **Never delete the session log**
(`logs/README.md` — the jsonl + .stdout pair is the only record of live
play; there is no site hand-history export).
