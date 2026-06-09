"""Windows-side socket sender for the staged auto-clicker.

Self-contained: paste this file into the Windows scraper project (or
import it from there). Provides a `ScraperSocketSender` that opens a
TCP connection to Contabo (via SSH tunnel or Tailscale), forwards each
scraper record as a JSON message, and sends a heartbeat every 1s so
the Contabo consumer can detect stalls.

Integration point — ONE LINE:
    sender = ScraperSocketSender("127.0.0.1", 9000)  # tunneled-loopback
    sender.start()
    # ...in the scraper's record-producing loop (wherever you currently
    #    append a JSON record to the local JSONL file):
    sender.send_frame(record_dict)
    # ...on shutdown:
    sender.stop()

Protocol (newline-delimited JSON over TCP):
    Upward (Windows → Contabo):
      {"type":"frame","seq":N,"record":<scraper_record>}\\n
      {"type":"heartbeat","seq":null,"ts":"YYYYMMDD_HHMMSS_ffffff"}\\n
    Downward (Contabo → Windows): reserved for Stages 2+ ("click", "ack"); NOT sent in Stage 1.

Resilience:
  - Connect on a background thread; reconnect with exponential backoff
    (1s → 2s → 4s → … capped 30s) on disconnect.
  - Heartbeat every 1s while connected.
  - Backpressure policy = DROP OLDEST: if the send queue is full
    (Contabo slow), discard the oldest queued frame so the NEWEST state
    is always sent next. This matters for Stages 2+ where stale frames
    lead to wrong clicks; in Stage 1 dry-run it's a no-op since the
    bridge is fast enough to keep up.
  - Failures are logged via the `log_cb` callback (default: print). They
    never raise into the scraper's main loop.

Standalone smoke test:
    python scraper_socket_sender.py --smoke 127.0.0.1 9000
will connect, send 5 fake "frame" messages + heartbeats, then exit.
"""
from __future__ import annotations

import argparse
import json
import queue
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Callable


def _ts() -> str:
    """ISO-ish compact timestamp matching the scraper's captured_at."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


class ScraperSocketSender:
    """Forward scraper records to a remote Contabo socket consumer.

    Drop-in: call `send_frame(record)` wherever the scraper currently
    emits a record (e.g. immediately before writing one line to the
    local JSONL log). The sender takes care of connecting, queuing,
    heartbeating, and reconnecting.

    Thread-safe. The scraper's main thread should call `send_frame` —
    that just enqueues the record and returns. A background thread
    handles the socket I/O.
    """

    def __init__(
        self,
        host: str,
        port: int,
        queue_max: int = 256,
        heartbeat_sec: float = 1.0,
        log_cb: Callable[[str], None] = print,
    ):
        self._host = host
        self._port = port
        self._queue_max = queue_max
        self._heartbeat_sec = heartbeat_sec
        self._log = log_cb

        self._q: queue.Queue[bytes] = queue.Queue(maxsize=queue_max)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._worker: threading.Thread | None = None
        self._heartbeat: threading.Thread | None = None

    # ─── public API ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the worker + heartbeat threads. Idempotent."""
        if self._worker and self._worker.is_alive():
            return
        self._stop_flag.clear()
        self._worker = threading.Thread(
            target=self._run_worker, name="scraper-sender", daemon=True
        )
        self._worker.start()
        self._heartbeat = threading.Thread(
            target=self._run_heartbeat, name="scraper-heartbeat", daemon=True
        )
        self._heartbeat.start()
        self._log(f"[scraper_socket_sender] started → {self._host}:{self._port}")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal threads to exit and wait briefly."""
        self._stop_flag.set()
        for t in (self._worker, self._heartbeat):
            if t and t.is_alive():
                t.join(timeout=timeout)
        self._log("[scraper_socket_sender] stopped")

    def send_frame(self, record: dict) -> None:
        """Enqueue a scraper record for transmission. Non-blocking.

        DROP-OLDEST backpressure: if the queue is full (consumer slow
        or disconnected), the OLDEST queued frame is discarded so the
        newest state is always queued next. Stage 2+ clicking on stale
        state would be a real bug; this prevents it by construction.
        """
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        msg = {"type": "frame", "seq": seq, "record": record}
        try:
            payload = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        except (TypeError, ValueError) as e:
            self._log(f"[scraper_socket_sender] frame serialize failed: {e}")
            return
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            # Drop oldest, put newest.
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(payload)
                self._log(
                    "[scraper_socket_sender] queue full — dropped oldest frame"
                )
            except queue.Full:
                self._log("[scraper_socket_sender] queue full — dropped newest frame")

    # ─── threads ───────────────────────────────────────────────────────

    def _run_worker(self) -> None:
        """Connect, drain queue, reconnect on drop. Exponential backoff."""
        backoff = 1.0
        while not self._stop_flag.is_set():
            sock = self._connect()
            if sock is None:
                # Connection failed. Sleep then retry.
                self._sleep_or_stop(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0  # reset on successful connect

            try:
                self._drain_to_socket(sock)
            except (BrokenPipeError, ConnectionError, OSError) as e:
                self._log(f"[scraper_socket_sender] connection error: {e}")
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
            self._log("[scraper_socket_sender] disconnected; will reconnect")

    def _run_heartbeat(self) -> None:
        """Enqueue a heartbeat every heartbeat_sec while running."""
        while not self._stop_flag.is_set():
            msg = {"type": "heartbeat", "seq": None, "ts": _ts()}
            payload = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                # Hearbeats are nice-to-have; if backed up, skip rather
                # than displace a real frame.
                pass
            self._sleep_or_stop(self._heartbeat_sec)

    def _connect(self) -> socket.socket | None:
        """One connection attempt. Returns the socket or None on fail."""
        try:
            s = socket.create_connection((self._host, self._port), timeout=5.0)
            s.settimeout(None)
            self._log(
                f"[scraper_socket_sender] connected to "
                f"{self._host}:{self._port}"
            )
            return s
        except OSError as e:
            self._log(
                f"[scraper_socket_sender] connect to "
                f"{self._host}:{self._port} failed: {e}"
            )
            return None

    def _drain_to_socket(self, sock: socket.socket) -> None:
        """Pull payloads from the queue and write to the socket until
        stop or error."""
        while not self._stop_flag.is_set():
            try:
                payload = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            sock.sendall(payload)

    def _sleep_or_stop(self, seconds: float) -> None:
        """Sleep but wake immediately on stop."""
        self._stop_flag.wait(timeout=seconds)


# ─── Standalone smoke test ─────────────────────────────────────────────

def _smoke(host: str, port: int) -> int:
    """Send 5 fake 'frame' messages + heartbeats, then exit. Useful for
    verifying connectivity to Contabo over the tunnel before integrating
    with the scraper."""
    sender = ScraperSocketSender(host, port)
    sender.start()
    print(f"[smoke] sending 5 fake frames to {host}:{port}…")
    try:
        for i in range(5):
            fake_record = {
                "captured_at": _ts(),
                "smoke_frame_no": i + 1,
                "note": "this is a smoke-test record, not a real scraper frame",
            }
            sender.send_frame(fake_record)
            print(f"[smoke]  → frame {i+1}/5")
            time.sleep(1.0)
        print("[smoke] sent. waiting 2s for queue drain, then exit.")
        time.sleep(2.0)
    finally:
        sender.stop()
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Windows-side socket sender for the staged auto-clicker."
    )
    p.add_argument("--smoke", nargs=2, metavar=("HOST", "PORT"),
                    help="Run the smoke test against HOST:PORT.")
    args = p.parse_args()
    if args.smoke:
        host = args.smoke[0]
        port = int(args.smoke[1])
        sys.exit(_smoke(host, port))
    print(__doc__)
    p.print_help()
    sys.exit(0)
