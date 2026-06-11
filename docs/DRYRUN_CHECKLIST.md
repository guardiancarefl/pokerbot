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
    --out logs/live_dryrun_${TS}.jsonl \
    |& tee logs/live_dryrun_${TS}.stdout
```

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
5. Start the scraper. Contabo prints
   `sender connected from 127.0.0.1:…`.

## 3. During the session — what to watch on the Contabo dashboard

- Fresh decisions print full READ / DECIDE / CLICK-PLAN banners;
  cached/skips print one-liners. That's normal.
- `[FLOOR] fired=[…]` — fine, that's the deployment floors working;
  triage will tabulate them afterward.
- `⚠ stall:` lines — tunnel or scraper hiccup; check the Windows side.
- **SAFE-FOLD banners or `[ANCHOR-REFUSED]`** — note the seq; these are
  red-flag material, triage will pull full context.

## 4. After the session (or mid-session — it's read-only)

```bash
.venv/bin/python scripts/dryrun_triage.py logs/live_dryrun_${TS}.jsonl
```

Writes `logs/triage_${TS}.txt` and prints it. The `.stdout` companion
is picked up automatically for the floor column. Paste back the final
`TRIAGE …` summary line.

Red-section items that demand attention before the next session:
any safe-fold, any `anchor_refused`, any `FOLD WHILE FACING 0`
(impossible post-floor — would mean the floors aren't active), any
seq-170/seq-192-class reconstruction signature.

## 5. Shutdown

Ctrl-C the listener (it prints `STILL NO CLICKING. Log-only.`), stop
the Windows scraper, close the tunnel. **Never delete the session log**
(`logs/README.md` — the jsonl + .stdout pair is the only record of live
play; there is no site hand-history export).
