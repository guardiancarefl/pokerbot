# Windows scraper sender — staged auto-clicker Stage 1

`scraper_socket_sender.py` is the **Windows-side** half of the live
dry-run pipeline. It forwards your scraper's per-frame records over a
TCP socket to the Contabo box, which runs `scripts/run_live_dryrun.py`
and prints the bot's READ / DECIDE / CLICK PLAN for each frame.

> **Stage 1 is DRY-RUN.** The bot never clicks anything. The Contabo
> side does not send anything back. The only direction of data flow in
> Stage 1 is Windows → Contabo.

---

## 1. Files

| File | Purpose |
|---|---|
| `scraper_socket_sender.py` | The sender. Self-contained. Drop into the scraper project. |
| `README.md` | This file. |

No third-party dependencies (uses only `socket`, `threading`, `queue`, `json`).

---

## 2. Integration point — one line

In whatever file produces a scraper record (the place where you
currently append a JSON line to your local JSONL log), do this:

```python
from scraper_socket_sender import ScraperSocketSender

# Once, at scraper startup:
sender = ScraperSocketSender("127.0.0.1", 9000)
sender.start()

# In the per-frame loop, wherever you do something like
#   with open("scraper.jsonl", "a") as f: f.write(json.dumps(record) + "\n")
sender.send_frame(record)        # ← ADD THIS LINE

# Once, at scraper shutdown:
sender.stop()
```

`send_frame` is non-blocking and thread-safe. It just enqueues the
record onto a background socket-writer thread.

---

## 3. Network setup — the secure channel

The scraper data must NOT go over an open public port. Pick one of
the two recommended channels:

### Option A — SSH tunnel (simplest)

On Windows, in a terminal:

```bat
ssh -N -L 9000:localhost:9000 quant@<CONTABO_PUBLIC_IP>
```

Then point the sender at the local tunneled port:

```python
ScraperSocketSender("127.0.0.1", 9000)
```

The traffic is encrypted over SSH and reaches the Contabo box on
`localhost:9000`, where the dry-run consumer listens. Keep this SSH
session open while the bot is running.

### Option B — Tailscale / WireGuard

Install Tailscale on both Windows and Contabo
(<https://tailscale.com/download>), log into the same tailnet, then
use the Contabo box's Tailscale IP (looks like `100.x.y.z`):

```python
ScraperSocketSender("100.x.y.z", 9000)
```

This stays up across reboots and gives you a stable private-IP address
no matter where the Windows machine roams.

---

## 4. Start the Contabo-side consumer first

On the Contabo box, run:

```bash
cd /home/quant/pokerbot
.venv/bin/python scripts/run_live_dryrun.py \
    --listen-port 9000 \
    --checkpoint runs/k200_real_ante_20260605_225847_PRESERVED/ckpt_iter_1500.pt \
    --abstraction runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
    --out logs/live_dryrun_$(date +%Y%m%d_%H%M%S).jsonl
```

The consumer prints a banner and waits for a connection on port 9000.
**Always** start the consumer before the Windows sender — otherwise the
sender will just sit in its reconnect-backoff loop.

---

## 5. Connection smoke test

Before integrating with the scraper, verify the tunnel + ports work:

```bat
:: On Windows, after the SSH tunnel is up and run_live_dryrun.py is listening:
python scraper_socket_sender.py --smoke 127.0.0.1 9000
```

The smoke test sends 5 fake frames + heartbeats and exits. The Contabo
dashboard will skip them as `skip_data_quality` (they aren't real
scraper records) — but that's enough to prove the channel is live.

---

## 6. Wire protocol (for reference)

Newline-delimited JSON over TCP. One JSON object per line.

```
{"type":"frame","seq":17,"record":{...full scraper record...}}\n
{"type":"heartbeat","seq":null,"ts":"20260607_142312_581"}\n
```

`seq` is a monotonic counter incremented per frame on the Windows side.
Heartbeats fire every 1s. If the Contabo dashboard sees nothing for
≥2s it prints a STALLED warning (it does not disconnect or click).

Downward traffic (Contabo → Windows) is reserved for Stages 2+
(click commands, ack receipts). Stage 1 sends nothing back.

---

## 7. Operational notes

- **Disconnect handling.** If the SSH tunnel drops or the Contabo
  consumer restarts, the sender enters exponential backoff
  (1s → 2s → 4s → … capped 30s). When the consumer comes back, it
  accepts the next fresh connection. No frames are lost — they queue
  on the Windows side.

- **Queue backpressure.** The send queue holds up to 256 messages. If
  Contabo can't keep up (it should — local decisions are ~10ms), the
  sender drops the OLDEST queued frame so the newest state is always
  sent next. This is important: Stage 2+ would click on whichever
  frame arrives, and stale state would be a real bug.

- **No clicking, no return traffic in Stage 1.** Confirm both before
  starting: search the consumer's logs for the banner
  `STILL NO CLICKING. Log-only.` and check that the sender prints
  no socket-read activity (it only writes).

- **Where do logs go?** Contabo logs each decision as one JSONL line
  to `--out`. Windows logs (the sender's stdout) tell you about
  reconnects + dropped frames; pipe it to a file if you want history.

---

## 8. Stage 2+ (not yet built)

Stage 2 adds a downward `click` message from Contabo → Windows. The
sender becomes bidirectional; a separate Windows-side click-executor
consumes those messages. Stages 2 and beyond require a separate
human-in-the-loop gate before they get built.
